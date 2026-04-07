"""Exponentially Damped Temporal Coherence Coding (ED-TCC) for KV cache compression.

Implements the novel damped DPCM predictor from the NexusQuant paper (Section 3.5)
that eliminates error accumulation in temporal coding of KV cache sequences.

Core idea: consecutive KV vectors are highly correlated (rho ~ 0.78 across model
families). Instead of storing raw vectors, store I-frames (full precision) at regular
intervals and P-frames (quantized residuals) between them, analogous to video coding.

The key innovation is the DAMPED predictor:

    pred(t) = alpha * h_hat[t-1] + (1 - alpha) * h_hat[t-2]

Unlike standard DPCM (pred = 2*h[t-1] - h[t-2]) which EXTRAPOLATES and causes error
to grow as O(k), the damped predictor is a CONVEX COMBINATION that bounds accumulated
error to a finite quantity regardless of sequence length:

    Var(accumulated_error) <= sigma_q^2 / (1 - (1-alpha)^2)

At alpha=0.6, transient error from any step decays by 40% per subsequent step.
After 5 steps, contribution is (0.4)^5 = 1.02%, effectively zero.

Results:
  - Standard DPCM: +1.91% PPL at 256 tokens (error accumulation)
  - ED-TCC alpha=0.6: -0.52% PPL at 512 tokens (improving trend with context)
  - On Qwen-3B: ED-TCC recovers 8 percentage points of quality lost by spatial VQ

Reference: NexusQuant paper Section 3.5, Appendix C; Patent application ED-TCC.
"""

import math
import torch
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass, field


@dataclass
class TCCCompressed:
    """Container for TCC-compressed KV sequence.

    Stores I-frames at full precision and P-frames as residuals,
    along with the parameters needed for decompression.

    Attributes:
        frames: List of (frame_type, data) tuples.
            frame_type: "I" for intra-coded, "P" for predictive-coded.
            data: tensor of shape (*, dim) -- raw vector for I, residual for P.
        alpha: Damping coefficient used for prediction.
        i_interval: I-frame interval (number of tokens between I-frames).
        seq_len: Original sequence length.
        original_shape: Shape of the original KV tensor (e.g. (heads, seq, dim)).
    """
    frames: list  # List of (str, Tensor) tuples: ("I", vec) or ("P", residual)
    alpha: float
    i_interval: int
    seq_len: int
    original_shape: torch.Size


def forward_tcc(
    kv_sequence: torch.Tensor,
    alpha: float = 0.6,
    i_interval: int = 32,
    quantize_fn: Optional[callable] = None,
) -> TCCCompressed:
    """Encode a KV sequence using ED-TCC temporal predictive coding.

    Encodes a sequence of KV vectors into I-frames (intra-coded, stored at full
    precision) and P-frames (predictive-coded, stored as residuals). The damped
    predictor uses reconstructed (not original) reference frames to prevent
    encoder-decoder mismatch.

    The prediction for token t is:
        pred(t) = alpha * h_hat[t-1] + (1 - alpha) * h_hat[t-2]

    where h_hat are RECONSTRUCTED vectors (after quantizing the residual).

    I-frames are placed at positions 0, i_interval, 2*i_interval, etc.
    The first two tokens of each I-frame group are always I-frames (the predictor
    needs two reference frames).

    Args:
        kv_sequence: KV tensor with token dimension as second-to-last.
            Shape: (heads, seq_len, dim) or (batch, heads, seq_len, dim).
            The temporal coding operates along the seq_len dimension.
        alpha: Damping coefficient in (0, 1). Default 0.6 (paper-validated).
            Higher alpha = more weight on most recent frame.
            Lower alpha = more temporal smoothing.
        i_interval: Number of tokens between I-frames. Default 32.
            Shorter intervals = less error accumulation but lower compression.
            Paper uses 32; patent suggests adaptive selection per layer.
        quantize_fn: Optional function to quantize residual vectors.
            Signature: quantize_fn(tensor) -> quantized_tensor.
            If None, residuals are stored at full precision (useful for analysis).
            In production, this would be E8Lattice.quantize or similar.

    Returns:
        TCCCompressed container with frames and metadata for decompression.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if i_interval < 2:
        raise ValueError(f"i_interval must be >= 2, got {i_interval}")

    original_shape = kv_sequence.shape

    # Identify the sequence dimension (second-to-last)
    if kv_sequence.ndim < 2:
        raise ValueError(
            f"kv_sequence must have at least 2 dimensions (seq, dim), "
            f"got shape {kv_sequence.shape}"
        )

    seq_dim = -2
    seq_len = kv_sequence.shape[seq_dim]

    # Move sequence dim to a canonical position for iteration.
    # Work with shape (..., seq_len, dim) where ... is arbitrary batch dims.
    batch_shape = kv_sequence.shape[:-2]
    dim = kv_sequence.shape[-1]

    # Flatten batch dims for uniform processing: (B, seq, dim)
    x = kv_sequence.reshape(-1, seq_len, dim)
    B = x.shape[0]

    frames = []
    # Reconstructed buffer: stores h_hat for the two most recent positions
    h_hat_prev2 = None  # h_hat[t-2]
    h_hat_prev1 = None  # h_hat[t-1]

    for t in range(seq_len):
        raw = x[:, t, :]  # (B, dim)

        # Determine if this is an I-frame
        pos_in_group = t % i_interval
        is_i_frame = (pos_in_group == 0) or (pos_in_group == 1) or (t < 2)

        if is_i_frame:
            # I-frame: store raw vector, no prediction
            frames.append(("I", raw.clone()))
            # For I-frames, the reconstructed value IS the raw value
            # (I-frames are stored at full precision)
            h_hat = raw.clone()
        else:
            # P-frame: predict from two most recent reconstructed frames
            pred = alpha * h_hat_prev1 + (1.0 - alpha) * h_hat_prev2
            residual = raw - pred

            # Optionally quantize the residual
            if quantize_fn is not None:
                residual_q = quantize_fn(residual)
            else:
                residual_q = residual

            frames.append(("P", residual_q.clone()))

            # Reconstruct using quantized residual (encoder-decoder match)
            h_hat = pred + residual_q

        # Shift reference buffers
        h_hat_prev2 = h_hat_prev1
        h_hat_prev1 = h_hat

    return TCCCompressed(
        frames=frames,
        alpha=alpha,
        i_interval=i_interval,
        seq_len=seq_len,
        original_shape=original_shape,
    )


def inverse_tcc(compressed: TCCCompressed) -> torch.Tensor:
    """Decode an ED-TCC compressed KV sequence back to a full tensor.

    Reconstructs the original KV sequence from I-frames and P-frame residuals
    using the same damped predictor. Because forward_tcc uses reconstructed
    (not original) reference frames, the decoder reproduces the encoder's
    prediction exactly -- no encoder-decoder mismatch.

    Args:
        compressed: TCCCompressed container from forward_tcc().

    Returns:
        Reconstructed KV tensor with the same shape as the original input.
    """
    alpha = compressed.alpha
    frames = compressed.frames
    seq_len = compressed.seq_len
    original_shape = compressed.original_shape

    # Infer dimensions from first frame
    first_data = frames[0][1]
    B = first_data.shape[0]
    dim = first_data.shape[-1]
    device = first_data.device
    dtype = first_data.dtype

    # Allocate output
    output = torch.zeros(B, seq_len, dim, device=device, dtype=dtype)

    h_hat_prev2 = None
    h_hat_prev1 = None

    for t, (frame_type, data) in enumerate(frames):
        if frame_type == "I":
            h_hat = data
        elif frame_type == "P":
            pred = alpha * h_hat_prev1 + (1.0 - alpha) * h_hat_prev2
            h_hat = pred + data  # data is the residual
        else:
            raise ValueError(f"Unknown frame type: {frame_type}")

        output[:, t, :] = h_hat

        # Shift reference buffers
        h_hat_prev2 = h_hat_prev1
        h_hat_prev1 = h_hat

    return output.reshape(original_shape)


def compute_compression_stats(
    kv_sequence: torch.Tensor,
    alpha: float = 0.6,
    i_interval: int = 32,
) -> dict:
    """Analyze the compression potential of ED-TCC on a KV sequence.

    Computes temporal correlation, residual variance reduction, and
    theoretical bit savings without actually compressing.

    Args:
        kv_sequence: KV tensor, shape (..., seq_len, dim).
        alpha: Damping coefficient.
        i_interval: I-frame interval.

    Returns:
        Dictionary with:
          - mean_correlation: Average cosine similarity between consecutive tokens.
          - variance_ratio: Ratio of residual variance to raw variance.
          - theoretical_bit_saving: Estimated bits/dim saved by temporal coding.
          - i_frame_fraction: Fraction of frames that are I-frames.
          - effective_compression_factor: Including I-frame overhead.
    """
    seq_len = kv_sequence.shape[-2]
    dim = kv_sequence.shape[-1]
    x = kv_sequence.reshape(-1, seq_len, dim).float()

    # Temporal correlation (cosine similarity)
    if seq_len < 2:
        return {
            "mean_correlation": 0.0,
            "variance_ratio": 1.0,
            "theoretical_bit_saving": 0.0,
            "i_frame_fraction": 1.0,
            "effective_compression_factor": 1.0,
        }

    x_curr = x[:, 1:, :]
    x_prev = x[:, :-1, :]
    cos_sim = torch.nn.functional.cosine_similarity(x_curr, x_prev, dim=-1)
    mean_corr = cos_sim.mean().item()

    # Residual variance ratio
    raw_var = x.var().item()
    if raw_var < 1e-12:
        return {
            "mean_correlation": mean_corr,
            "variance_ratio": 1.0,
            "theoretical_bit_saving": 0.0,
            "i_frame_fraction": 1.0,
            "effective_compression_factor": 1.0,
        }

    # Simulate prediction to measure actual residual variance
    residual_vars = []
    h_prev2 = None
    h_prev1 = None
    for t in range(seq_len):
        raw = x[:, t, :]
        pos_in_group = t % i_interval
        is_i = (pos_in_group == 0) or (pos_in_group == 1) or (t < 2)
        if not is_i and h_prev1 is not None and h_prev2 is not None:
            pred = alpha * h_prev1 + (1.0 - alpha) * h_prev2
            residual = raw - pred
            residual_vars.append(residual.var().item())
        h_prev2 = h_prev1
        h_prev1 = raw

    if len(residual_vars) > 0:
        mean_residual_var = sum(residual_vars) / len(residual_vars)
        var_ratio = mean_residual_var / raw_var
    else:
        var_ratio = 1.0

    # Theoretical bit saving: 0.5 * log2(1 / var_ratio)
    if var_ratio > 0 and var_ratio < 1.0:
        bit_saving = 0.5 * math.log2(1.0 / var_ratio)
    else:
        bit_saving = 0.0

    # I-frame fraction
    n_i_frames = 0
    for t in range(seq_len):
        pos_in_group = t % i_interval
        if (pos_in_group == 0) or (pos_in_group == 1) or (t < 2):
            n_i_frames += 1
    i_frac = n_i_frames / seq_len

    # Effective compression: P-frames save bits, I-frames don't
    # Compression factor from temporal coding on P-frames only
    p_frac = 1.0 - i_frac
    if bit_saving > 0 and p_frac > 0:
        # Weighted average bit saving across I and P frames
        effective_factor = 1.0 / (i_frac + p_frac * (2 ** (-2 * bit_saving)))
    else:
        effective_factor = 1.0

    return {
        "mean_correlation": mean_corr,
        "variance_ratio": var_ratio,
        "theoretical_bit_saving": bit_saving,
        "i_frame_fraction": i_frac,
        "effective_compression_factor": effective_factor,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Validate ED-TCC forward/inverse roundtrip and error bounds."""
    torch.manual_seed(42)
    print("=" * 60)
    print("ED-TCC Self-Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Perfect roundtrip (no quantization)
    # ------------------------------------------------------------------
    heads, seq, dim = 8, 128, 128
    x = torch.randn(heads, seq, dim)
    compressed = forward_tcc(x, alpha=0.6, i_interval=32)
    x_recon = inverse_tcc(compressed)
    err = (x - x_recon).abs().max().item()
    print(f"Test 1 -- Roundtrip (8, 128, 128), no quant: max error = {err:.2e}", end="")
    assert err < 1e-5, f"Roundtrip error too large: {err}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 2: Frame type distribution
    # ------------------------------------------------------------------
    n_i = sum(1 for ft, _ in compressed.frames if ft == "I")
    n_p = sum(1 for ft, _ in compressed.frames if ft == "P")
    print(f"Test 2 -- Frame distribution: {n_i} I-frames, {n_p} P-frames "
          f"({100*n_p/seq:.0f}% P-frames)", end="")
    assert n_i + n_p == seq, "Frame count mismatch"
    assert n_p > n_i, "Should have more P-frames than I-frames"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 3: Residual variance reduction on correlated data
    # ------------------------------------------------------------------
    # Generate temporally correlated KV-like data (AR(1) process)
    rho = 0.8
    x_corr = torch.zeros(4, 256, 128)
    x_corr[:, 0, :] = torch.randn(4, 128)
    for t in range(1, 256):
        x_corr[:, t, :] = rho * x_corr[:, t-1, :] + math.sqrt(1 - rho**2) * torch.randn(4, 128)

    stats = compute_compression_stats(x_corr, alpha=0.6, i_interval=32)
    print(f"Test 3 -- Correlated data (rho=0.8):")
    print(f"         Measured correlation:    {stats['mean_correlation']:.3f}")
    print(f"         Variance ratio:          {stats['variance_ratio']:.3f}")
    print(f"         Bit saving:              {stats['theoretical_bit_saving']:.2f} bits/dim")
    print(f"         Effective compression:   {stats['effective_compression_factor']:.2f}x")
    assert stats['variance_ratio'] < 0.5, "Variance should be significantly reduced"
    assert stats['theoretical_bit_saving'] > 0.3, "Should save at least 0.3 bits/dim"
    print("         [PASS]")

    # ------------------------------------------------------------------
    # Test 4: Roundtrip with simulated quantization
    # ------------------------------------------------------------------
    def fake_quantize(x):
        """Simulates coarse quantization (round to nearest 0.1)."""
        return (x * 10).round() / 10

    compressed_q = forward_tcc(x_corr, alpha=0.6, i_interval=32, quantize_fn=fake_quantize)
    x_recon_q = inverse_tcc(compressed_q)
    err_q = (x_corr - x_recon_q).abs().mean().item()
    print(f"Test 4 -- Roundtrip with quantization: mean error = {err_q:.4f}", end="")
    # With quantization, some error is expected, but it should be bounded
    assert err_q < 1.0, f"Quantized reconstruction error too large: {err_q}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 5: Damped vs undamped error accumulation
    # ------------------------------------------------------------------
    # Create data and compare error growth
    x_test = torch.zeros(1, 128, 64)
    x_test[:, 0, :] = torch.randn(1, 64)
    for t in range(1, 128):
        x_test[:, t, :] = 0.9 * x_test[:, t-1, :] + 0.1 * torch.randn(1, 64)

    # Damped (alpha=0.6) -- errors should stay bounded
    comp_damped = forward_tcc(x_test, alpha=0.6, i_interval=128, quantize_fn=fake_quantize)
    recon_damped = inverse_tcc(comp_damped)
    err_first_half = (x_test[:, 2:64, :] - recon_damped[:, 2:64, :]).pow(2).mean().item()
    err_second_half = (x_test[:, 64:, :] - recon_damped[:, 64:, :]).pow(2).mean().item()
    growth = err_second_half / (err_first_half + 1e-12)
    print(f"Test 5 -- Error growth (damped, i_interval=128):")
    print(f"         1st half MSE: {err_first_half:.6f}")
    print(f"         2nd half MSE: {err_second_half:.6f}")
    print(f"         Growth ratio: {growth:.2f}x", end="")
    # Damped predictor: error should NOT grow unboundedly
    # Allow some growth due to statistics, but it should be moderate
    assert growth < 5.0, f"Error growth too large for damped predictor: {growth:.2f}x"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 6: 4D input shape (batch, heads, seq, dim)
    # ------------------------------------------------------------------
    x4d = torch.randn(2, 8, 64, 128)
    comp4d = forward_tcc(x4d, alpha=0.6, i_interval=16)
    recon4d = inverse_tcc(comp4d)
    err4d = (x4d - recon4d).abs().max().item()
    print(f"Test 6 -- 4D input (2, 8, 64, 128): max error = {err4d:.2e}", end="")
    assert err4d < 1e-5, f"4D roundtrip error too large: {err4d}"
    assert recon4d.shape == x4d.shape, f"Shape mismatch: {recon4d.shape} vs {x4d.shape}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 7: Short sequences (edge case)
    # ------------------------------------------------------------------
    x_short = torch.randn(2, 3, 64)
    comp_short = forward_tcc(x_short, alpha=0.6, i_interval=32)
    recon_short = inverse_tcc(comp_short)
    err_short = (x_short - recon_short).abs().max().item()
    # With seq_len=3 and i_interval=32, all frames should be I-frames
    n_i_short = sum(1 for ft, _ in comp_short.frames if ft == "I")
    print(f"Test 7 -- Short seq (2, 3, 64): {n_i_short} I-frames, "
          f"max error = {err_short:.2e}", end="")
    assert n_i_short == 3, "All frames should be I-frames for short sequences"
    assert err_short < 1e-5, f"Short seq error too large: {err_short}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 8: Alpha parameter validation
    # ------------------------------------------------------------------
    try:
        forward_tcc(x_short, alpha=0.0)
        assert False, "Should have raised ValueError for alpha=0"
    except ValueError:
        pass
    try:
        forward_tcc(x_short, alpha=1.0)
        assert False, "Should have raised ValueError for alpha=1"
    except ValueError:
        pass
    print("Test 8 -- Alpha validation (0, 1 rejected)  [PASS]")

    print("=" * 60)
    print("All ED-TCC tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()

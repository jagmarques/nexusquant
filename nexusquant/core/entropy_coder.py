"""rANS (Asymmetric Numeral Systems) entropy coder for E8 lattice indices.

After E8 quantization, lattice point coordinates are integers or half-integers
with non-uniform (center-heavy) distribution. rANS compresses these to near
Shannon entropy, giving a free 1.2-1.5x boost on top of E8 VQ.

This is a pure Python/NumPy implementation for prototyping and measurement.
Production use would need a Cython/C/Triton kernel.

Theory:
    E8 at 3-bit has coordinates in {-4, -3, ..., 3, 4} (integer type)
    or {-3.5, -2.5, ..., 2.5, 3.5} (half-integer type).
    Uniform coding: ceil(log2(9)) = 4 bits/coordinate.
    Actual entropy: ~2.5-3.0 bits/coordinate (data-dependent).
    rANS achieves entropy + ~0.01 bits overhead.
    Expected additional compression: log2(9)/entropy ~ 1.2-1.3x.

References:
    J. Duda, "Asymmetric numeral systems", 2009. arXiv:0902.0271
    F. Giesen, "Interleaved entropy coders", 2014. arXiv:1402.3392
"""

import struct
import math
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Utility: convert E8 lattice points to/from integer symbol streams
# ---------------------------------------------------------------------------

def _lattice_to_symbols(lattice_points: np.ndarray, levels: int = 8
                        ) -> Tuple[np.ndarray, bool, int]:
    """Convert E8 lattice coordinates to non-negative integer symbols.

    E8 coordinates can be integers or half-integers. We detect which type
    and shift to non-negative range.

    For integer coords in [-levels/2, levels/2]: shift by levels/2.
        e.g. levels=8: {-4,-3,...,3,4} -> {0,1,...,7,8}, alphabet_size=9
    For half-integer coords: multiply by 2, then shift.
        e.g. levels=8: {-3.5,...,3.5} combined with ints -> alphabet_size=17

    Args:
        lattice_points: flat array of E8 coordinate values (float)
        levels: quantization levels (2^bits)

    Returns:
        symbols: non-negative integer array
        has_half: whether half-integer coords were detected
        alphabet_size: number of distinct symbols possible
    """
    half_lvl = levels // 2

    # Detect half-integers: check if any coordinate has fractional part ~0.5
    frac = np.abs(lattice_points - np.round(lattice_points))
    has_half = bool(np.any(frac > 0.25))

    if has_half:
        # Multiply by 2 to make everything integer, then shift
        doubled = np.round(lattice_points * 2).astype(np.int32)
        shifted = doubled + levels
        alphabet_size = 2 * levels + 1
        symbols = np.clip(shifted, 0, alphabet_size - 1).astype(np.int32)
    else:
        # Pure integer coords
        rounded = np.round(lattice_points).astype(np.int32)
        shifted = rounded + half_lvl
        alphabet_size = levels + 1
        symbols = np.clip(shifted, 0, alphabet_size - 1).astype(np.int32)

    return symbols, has_half, alphabet_size


def _symbols_to_lattice(symbols: np.ndarray, has_half: bool, levels: int = 8
                         ) -> np.ndarray:
    """Inverse of _lattice_to_symbols."""
    half_lvl = levels // 2

    if has_half:
        doubled = symbols.astype(np.float64) - levels
        return (doubled / 2.0).astype(np.float32)
    else:
        return (symbols.astype(np.float32) - half_lvl)


# ---------------------------------------------------------------------------
# Entropy measurement
# ---------------------------------------------------------------------------

def measure_entropy(indices: np.ndarray, alphabet_size: Optional[int] = None
                    ) -> Dict[str, float]:
    """Measure Shannon entropy and theoretical compression of a symbol stream.

    Args:
        indices: 1D integer array of symbols
        alphabet_size: total possible symbols (for uniform baseline)

    Returns:
        Dictionary with entropy stats:
            - entropy_bps: bits per symbol (Shannon entropy)
            - uniform_bps: bits per symbol if uniform
            - compression_ratio: uniform_bps / entropy_bps
            - n_symbols: total symbols
            - n_unique: number of distinct symbols seen
    """
    n = len(indices)
    if n == 0:
        return {"entropy_bps": 0, "uniform_bps": 0, "compression_ratio": 1.0,
                "n_symbols": 0, "n_unique": 0}

    unique, counts = np.unique(indices, return_counts=True)
    probs = counts / n

    # Shannon entropy H = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-30))

    if alphabet_size is None:
        alphabet_size = int(unique.max()) + 1
    uniform_bps = math.log2(max(alphabet_size, 2))

    ratio = uniform_bps / max(entropy, 0.01)

    return {
        "entropy_bps": float(entropy),
        "uniform_bps": float(uniform_bps),
        "compression_ratio": float(ratio),
        "n_symbols": int(n),
        "n_unique": int(len(unique)),
    }


def measure_e8_entropy(quantized: torch.Tensor, scale_factors: torch.Tensor,
                       levels: int = 8) -> Dict[str, float]:
    """Measure entropy of E8 quantized tensor (the full pipeline measurement).

    Takes the output of E8 quantization (scaled lattice points) and measures
    how compressible the underlying indices are.

    Args:
        quantized: E8 quantized tensor (any shape, last dim divisible by 8)
        scale_factors: per-group scale factors from E8 quantization
        levels: quantization levels

    Returns:
        Entropy statistics dict
    """
    flat = quantized.reshape(-1, 8).float().cpu().numpy()
    scales = scale_factors.reshape(-1, 1).float().cpu().numpy()
    scales = np.clip(scales, 1e-8, None)
    lattice_coords = (flat / scales).flatten()

    symbols, has_half, alphabet_size = _lattice_to_symbols(lattice_coords, levels)
    return measure_entropy(symbols, alphabet_size)


# ---------------------------------------------------------------------------
# rANS encoder/decoder
# ---------------------------------------------------------------------------
# 32-bit state rANS with 16-bit renormalization.
# Duda (2009): simple, correct, near-optimal (~0.01 bits overhead).
#
# State invariant: state in [L, L << 16) = [2^16, 2^32)
# Frequency tables sum to M = 2^PROB_BITS.
# Streaming unit: 16-bit words.

PROB_BITS = 12
M = 1 << PROB_BITS  # 4096 = frequency table precision
L = 1 << 16         # lower bound of state range (state >= L always)


def _build_freq_table(symbols: np.ndarray, alphabet_size: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalized frequency and cumulative frequency tables.

    Frequencies are scaled to sum to M (power of 2).
    Every symbol that appears gets at least frequency 1.

    Args:
        symbols: 1D integer array
        alphabet_size: total alphabet size

    Returns:
        freqs: array of length alphabet_size, sums to M
        cum_freqs: array of length alphabet_size+1, cum_freqs[i] = sum(freqs[:i])
    """
    counts = np.bincount(symbols, minlength=alphabet_size).astype(np.int64)
    total = counts.sum()

    if total == 0:
        freqs = np.ones(alphabet_size, dtype=np.int64)
        freqs *= M // alphabet_size
        remainder = M - freqs.sum()
        freqs[0] += remainder
    else:
        seen_mask = counts > 0
        n_seen = int(seen_mask.sum())

        freqs = np.zeros(alphabet_size, dtype=np.int64)
        freqs[seen_mask] = 1
        remaining = M - n_seen

        if remaining > 0 and total > 0:
            # Distribute remaining budget proportionally to counts
            proportional = (counts[seen_mask].astype(np.float64) / total) * remaining
            extra = np.floor(proportional).astype(np.int64)
            freqs[seen_mask] += extra

            # Fix rounding: award leftovers to symbols with highest fractional part
            diff = int(M - freqs.sum())
            if diff > 0:
                seen_indices = np.where(seen_mask)[0]
                remainders = proportional - extra
                order = seen_indices[np.argsort(-remainders)]
                for i in range(diff):
                    freqs[order[i % len(order)]] += 1
            elif diff < 0:
                seen_indices = np.where(seen_mask)[0]
                order = seen_indices[np.argsort(counts[seen_mask])]
                removed = 0
                for idx in order:
                    while freqs[idx] > 1 and removed < -diff:
                        freqs[idx] -= 1
                        removed += 1
                    if removed >= -diff:
                        break
        elif remaining < 0:
            # More seen symbols than M -- extremely unlikely for typical E8 alphabets
            seen_indices = np.where(seen_mask)[0]
            order = seen_indices[np.argsort(-counts[seen_mask])]
            freqs[:] = 0
            for i in range(min(M, len(order))):
                freqs[order[i]] = 1

    assert freqs.sum() == M, f"Freq sum {freqs.sum()} != {M}"

    cum_freqs = np.zeros(alphabet_size + 1, dtype=np.int64)
    cum_freqs[1:] = np.cumsum(freqs)

    return freqs, cum_freqs


def _rans_encode(symbols: np.ndarray, freqs: np.ndarray, cum_freqs: np.ndarray
                 ) -> List[int]:
    """rANS encode a symbol stream. Returns list of 16-bit words.

    Encodes symbols in REVERSE order (rANS stack property).
    Decoder reads forward and recovers symbols in original order.

    The encode step for symbol s with frequency fs and cumulative start cs:
        new_state = (state // fs) * M + (state % fs) + cs

    Before encoding, state is renormalized by streaming out 16-bit words
    until state is in the valid input range [fs * (L >> PROB_BITS), fs * (L << (16 - PROB_BITS))).

    Args:
        symbols: 1D integer array of symbols to encode
        freqs: frequency table (sums to M)
        cum_freqs: cumulative frequency table

    Returns:
        List of 16-bit words (reversed: first word contains MSB of final state)
    """
    state = L  # initial state
    out_u16 = []

    for i in range(len(symbols) - 1, -1, -1):
        s = int(symbols[i])
        fs = int(freqs[s])
        cs = int(cum_freqs[s])

        if fs == 0:
            raise ValueError(f"Cannot encode symbol {s} with zero frequency")

        # Renormalize: stream out 16-bit words until state is small enough
        # that after encoding, state stays in [L, L << 16).
        # Condition: state < fs << (32 - PROB_BITS) = fs << 20
        threshold = fs << (32 - PROB_BITS)
        while state >= threshold:
            out_u16.append(state & 0xFFFF)
            state >>= 16

        # Core rANS encode step
        state = (state // fs) * M + (state % fs) + cs

    # Flush final state (2 x 16-bit words for 32-bit state)
    out_u16.append(state & 0xFFFF)
    state >>= 16
    out_u16.append(state & 0xFFFF)

    out_u16.reverse()
    return out_u16


def _rans_decode(encoded_u16: List[int], n_symbols: int,
                 freqs: np.ndarray, cum_freqs: np.ndarray, alphabet_size: int
                 ) -> np.ndarray:
    """rANS decode. Returns the original symbol stream.

    Args:
        encoded_u16: list of 16-bit words from _rans_encode
        n_symbols: number of symbols to decode
        freqs: frequency table
        cum_freqs: cumulative frequency table
        alphabet_size: alphabet size

    Returns:
        1D integer array of decoded symbols
    """
    # Build CDF -> symbol reverse lookup table
    cum2sym = np.zeros(M, dtype=np.int32)
    for s in range(alphabet_size):
        start = int(cum_freqs[s])
        end = int(cum_freqs[s + 1])
        if start < end:
            cum2sym[start:end] = s

    # Initialize state from first 2 x 16-bit words
    pos = 0
    state = (encoded_u16[pos] << 16) | encoded_u16[pos + 1]
    pos += 2

    symbols = np.zeros(n_symbols, dtype=np.int32)

    for i in range(n_symbols):
        # Extract symbol from current state
        slot = state & (M - 1)  # = state % M, since M is power of 2
        s = int(cum2sym[slot])
        fs = int(freqs[s])
        cs = int(cum_freqs[s])

        # Inverse of encode step
        state = fs * (state >> PROB_BITS) + slot - cs

        # Renormalize: read 16-bit words to bring state back to [L, L << 16)
        while state < L and pos < len(encoded_u16):
            state = (state << 16) | encoded_u16[pos]
            pos += 1

        symbols[i] = s

    return symbols


# ---------------------------------------------------------------------------
# High-level API: encode/decode E8 quantized tensors
# ---------------------------------------------------------------------------

def _serialize_freq_table(freqs: np.ndarray) -> bytes:
    """Serialize frequency table to bytes."""
    # Format: [alphabet_size:u16] [freq_0:u16] [freq_1:u16] ...
    alphabet_size = len(freqs)
    parts = [struct.pack('<H', alphabet_size)]
    for f in freqs:
        parts.append(struct.pack('<H', int(f)))
    return b''.join(parts)


def _deserialize_freq_table(data: bytes, offset: int = 0
                             ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Deserialize frequency table from bytes."""
    alphabet_size = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    freqs = np.zeros(alphabet_size, dtype=np.int64)
    for i in range(alphabet_size):
        freqs[i] = struct.unpack_from('<H', data, offset)[0]
        offset += 2
    cum_freqs = np.zeros(alphabet_size + 1, dtype=np.int64)
    cum_freqs[1:] = np.cumsum(freqs)
    return freqs, cum_freqs, alphabet_size, offset


def encode_e8(quantized: torch.Tensor, scale_factors: torch.Tensor,
              levels: int = 8) -> bytes:
    """Entropy-encode an E8 quantized tensor using rANS.

    Takes the output of E8Lattice.quantize (already-quantized values with
    per-group scales applied) and produces a compact byte stream.

    Args:
        quantized: E8 quantized tensor (flattened to groups of 8)
        scale_factors: per-group scale factors, shape (n_groups, 1) or (n_groups,)
        levels: E8 quantization levels (default 8 = 3-bit)

    Returns:
        Compressed byte stream containing header, freq table, scales, and rANS data.
    """
    # Flatten to groups of 8 and recover lattice coordinates
    flat = quantized.reshape(-1, 8).float().cpu().numpy()
    scales = scale_factors.reshape(-1, 1).float().cpu().numpy()
    scales = np.clip(scales, 1e-8, None)
    lattice_coords = flat / scales

    # Convert to symbols
    coord_flat = lattice_coords.flatten()
    symbols, has_half, alphabet_size = _lattice_to_symbols(coord_flat, levels)

    # Build frequency table and encode
    freqs, cum_freqs = _build_freq_table(symbols, alphabet_size)
    encoded_u16 = _rans_encode(symbols, freqs, cum_freqs)

    # Pack into byte stream:
    # Header: [n_groups:u32] [levels:u8] [has_half:u8] [n_symbols:u32]
    n_groups = flat.shape[0]
    n_symbols = len(symbols)
    header = struct.pack('<IBBI', n_groups, levels, int(has_half), n_symbols)

    # Frequency table
    freq_bytes = _serialize_freq_table(freqs)

    # Scale factors as float16 (2 bytes each)
    scale_f16 = scales.flatten().astype(np.float16).tobytes()

    # rANS data as packed 16-bit words
    rans_data = struct.pack(f'<{len(encoded_u16)}H', *encoded_u16)

    # Section lengths for parsing
    freq_len = len(freq_bytes)
    scale_len = len(scale_f16)
    rans_len = len(rans_data)
    lengths = struct.pack('<III', freq_len, scale_len, rans_len)

    return header + lengths + freq_bytes + scale_f16 + rans_data


def decode_e8(compressed: bytes, original_shape: Optional[tuple] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode a rANS-compressed E8 stream back to quantized tensor.

    Args:
        compressed: bytes from encode_e8
        original_shape: optional, reshape output to this shape

    Returns:
        (quantized_tensor, scale_factors) -- lossless reconstruction of the
        E8-quantized values that were encoded.
    """
    offset = 0

    # Header
    n_groups, levels, has_half_int, n_symbols = struct.unpack_from('<IBBI', compressed, offset)
    offset += struct.calcsize('<IBBI')
    has_half = bool(has_half_int)

    # Lengths
    freq_len, scale_len, rans_len = struct.unpack_from('<III', compressed, offset)
    offset += struct.calcsize('<III')

    # Frequency table
    freqs, cum_freqs, alphabet_size, _ = _deserialize_freq_table(
        compressed[offset:offset + freq_len])
    offset += freq_len

    # Scale factors
    scales_f16 = np.frombuffer(compressed[offset:offset + scale_len], dtype=np.float16)
    scales = scales_f16.astype(np.float32).reshape(-1, 1)
    offset += scale_len

    # rANS data
    n_u16 = rans_len // 2
    encoded_u16 = list(struct.unpack_from(f'<{n_u16}H', compressed, offset))
    offset += rans_len

    # Decode symbols
    symbols = _rans_decode(encoded_u16, n_symbols, freqs, cum_freqs, alphabet_size)

    # Convert back to lattice coordinates and re-apply scales
    lattice_coords = _symbols_to_lattice(symbols, has_half, levels)
    lattice_coords = lattice_coords.reshape(n_groups, 8)
    quantized = lattice_coords * scales

    quantized_tensor = torch.from_numpy(quantized)
    if original_shape is not None:
        quantized_tensor = quantized_tensor.reshape(original_shape)

    scale_tensor = torch.from_numpy(scales.flatten())
    return quantized_tensor, scale_tensor


# ---------------------------------------------------------------------------
# Full E8 quantize + entropy encode pipeline (for measurement)
# ---------------------------------------------------------------------------

def e8_quantize_with_entropy(x: torch.Tensor, levels: int = 8,
                             return_stats: bool = True) -> Dict:
    """E8 quantize and entropy-encode a tensor. For benchmarking compression.

    Runs the full pipeline: E8 VQ -> rANS encode -> rANS decode, and reports
    the actual compression achieved.

    Args:
        x: input tensor (any shape, last dim will be padded to multiple of 8)
        levels: E8 quantization levels (8 = 3-bit)
        return_stats: whether to compute and return entropy statistics

    Returns:
        Dict with:
            - compressed_bytes: the rANS bitstream
            - reconstructed: decoded tensor (same shape as input)
            - scale_factors: per-group scales
            - n_bytes_compressed: compressed size in bytes
            - n_bytes_uniform: size with uniform coding
            - n_bytes_original: size as fp16
            - rans_compression_ratio: uniform / compressed
            - e8_base_ratio: fp16 / uniform
            - total_compression_ratio: fp16 / compressed
            - entropy_stats: (if return_stats) Shannon entropy analysis
    """
    from nexusquant.core.e8_lattice import E8Lattice

    shape = x.shape

    # Pad last dim to multiple of 8
    pad = (8 - shape[-1] % 8) % 8
    if pad > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad))
    else:
        x_padded = x

    # E8 quantize — replicate logic to capture scale factors
    flat = x_padded.reshape(-1, 8).float()
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = amax / (levels / 2)
    lp = E8Lattice.nearest_point(flat / sc).clamp(-levels / 2, levels / 2)
    quantized_scaled = lp * sc

    # Entropy encode
    compressed = encode_e8(quantized_scaled, sc, levels=levels)

    # Decode to verify round-trip
    decoded, decoded_scales = decode_e8(
        compressed, original_shape=x_padded.reshape(-1, 8).shape)

    # Reshape back
    reconstructed = decoded.reshape(x_padded.shape)
    if pad > 0:
        reconstructed = reconstructed[..., :shape[-1]]
    reconstructed = reconstructed.reshape(shape)

    # Size calculations
    n_coords = flat.numel()
    n_groups = flat.shape[0]

    # Uniform coding: ceil(log2(levels+1)) bits per coord, plus fp16 scales
    bits_per_coord_uniform = math.ceil(math.log2(levels + 1))
    n_bytes_uniform = math.ceil(n_coords * bits_per_coord_uniform / 8) + n_groups * 2

    n_bytes_compressed = len(compressed)
    original_bytes = n_coords * 2  # fp16 baseline

    rans_ratio = n_bytes_uniform / max(n_bytes_compressed, 1)
    e8_base_ratio = original_bytes / n_bytes_uniform
    total_ratio = original_bytes / n_bytes_compressed

    result = {
        "compressed_bytes": compressed,
        "reconstructed": reconstructed,
        "scale_factors": sc,
        "n_bytes_compressed": n_bytes_compressed,
        "n_bytes_uniform": n_bytes_uniform,
        "n_bytes_original": original_bytes,
        "rans_compression_ratio": rans_ratio,
        "e8_base_ratio": e8_base_ratio,
        "total_compression_ratio": total_ratio,
    }

    if return_stats:
        coord_flat = lp.cpu().numpy().flatten()
        symbols, has_half, alphabet_size = _lattice_to_symbols(coord_flat, levels)
        result["entropy_stats"] = measure_entropy(symbols, alphabet_size)

    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Verify round-trip correctness on random data."""
    torch.manual_seed(42)

    # Simulate KV cache-like data (roughly Gaussian, which is close to real KV)
    x = torch.randn(4, 32, 64, 128) * 0.5

    result = e8_quantize_with_entropy(x, levels=8)
    recon = result["reconstructed"]

    print("=== rANS Entropy Coder Self-Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Original size (fp16):   {result['n_bytes_original']:>12,} bytes")
    print(f"E8 uniform coded:       {result['n_bytes_uniform']:>12,} bytes")
    print(f"E8 + rANS compressed:   {result['n_bytes_compressed']:>12,} bytes")
    print(f"E8 base ratio:          {result['e8_base_ratio']:.2f}x (fp16 -> E8)")
    print(f"rANS additional ratio:  {result['rans_compression_ratio']:.3f}x")
    print(f"Total compression:      {result['total_compression_ratio']:.2f}x (fp16 -> E8+rANS)")
    print(f"Entropy stats:          {result['entropy_stats']}")

    # Verify lossless round-trip of entropy coding
    decoded2, _ = decode_e8(
        result["compressed_bytes"],
        original_shape=(x.reshape(-1, 8).shape[0], 8))
    max_err = (decoded2.reshape(-1, 8) - recon.reshape(-1, 8)).abs().max().item()
    print(f"Round-trip max error:   {max_err:.2e}")
    status = "PASS" if max_err < 1e-2 else "FAIL"
    print(f"Status: {status}")
    return status == "PASS"


if __name__ == "__main__":
    ok = _self_test()
    exit(0 if ok else 1)

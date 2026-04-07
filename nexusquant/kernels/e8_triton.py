"""Triton GPU kernels for E8 lattice vector quantization.

Replaces the CPU-bound E8Lattice.nearest_point / quantize_perhead with
a fully GPU-parallel implementation. Each 8D group is processed by an
independent Triton program instance — perfect embarrassing parallelism.

Algorithm mirrors e8_lattice.py exactly:
    1. Integer candidate: round to Z^8, fix coordinate-sum parity
    2. Half-integer candidate: round to (Z+0.5)^8, apply relaxed parity fix
    3. Pick whichever candidate has smaller squared distance to input

The parity fix uses the same deliberate relaxation documented in
e8_lattice.py: the half-integer coset parity is intentionally loose
(matches the CPU reference), which empirically lowers distortion on
sub-Gaussian KV distributions by 0.3-0.4%.

Encoding (e8_encode):
    - Per-vector (per-head) fp16 scale stored as 1 scalar
    - Each 8D group of quantized integers packed as 8 x int8 = 8 bytes
    - With levels=4 (2-bit), valid range is [-2, 2] → fits comfortably in int8

Fused dequant-matmul (e8_dequant_matmul):
    - Loads int8 codes + scales, reconstructs float on-the-fly
    - Immediately multiplies with query matrix Q (head_dim, head_dim)
    - Never materialises the full float KV tensor → saves bandwidth + memory
    - Inspired by TurboQuant/tq-kv style fused decode-GEMV
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # triton not installed (CPU-only environments)
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _require_triton():
    """Raise a clear ImportError if Triton is not installed."""
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "nexusquant.kernels requires triton >= 2.1. "
            "Install with: pip install triton>=2.1  "
            "On CPU-only machines, use nexusquant.core.e8_lattice.E8Lattice instead."
        )


# ---------------------------------------------------------------------------
# Triton kernel definitions — only compiled when triton is available
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def e8_nearest_point_kernel(
        x_ptr,       # float32 input,  shape [N, 8]
        out_ptr,     # float32 output, shape [N, 8]
        N: tl.constexpr,
        BLOCK: tl.constexpr,  # number of 8D groups per program instance
    ):
        """Find the nearest (relaxed) E8 lattice point for BLOCK groups of 8 dims.

        Each program handles BLOCK consecutive 8D groups, loading all 8
        coordinates into registers and performing the full nearest-point
        algorithm without any memory round-trips for intermediate state.
        """
        pid = tl.program_id(0)
        group_start = pid * BLOCK  # first group index for this program

        # Lane offsets within the BLOCK x 8 tile
        # We iterate over groups sequentially within the block to keep register
        # pressure manageable while still amortising launch overhead.
        for b in tl.static_range(BLOCK):
            g = group_start + b
            if g >= N:
                break

            base = g * 8
            offs = base + tl.arange(0, 8)

            # ---- Load 8 coordinates ----------------------------------------
            x = tl.load(x_ptr + offs)   # shape [8]

            # ---- Integer candidate: round to nearest integer ----------------
            r_int = tl.math.round(x)

            # Coordinate sum parity fix
            s = tl.sum(r_int, axis=0)                      # scalar
            is_odd = (tl.math.fmod(s, 2.0) != 0.0)        # bool scalar

            gap = tl.abs(x - r_int)                        # [8]

            # argmin of gap — branchless using sequential comparisons
            # We find the index with the smallest gap and the sign of (x - r_int)
            # at that index.  The fix is: r_int[idx] += sign.
            #
            # Triton does not have a native argmin, so we materialise it with a
            # running min-scan across the 8 fixed-size lanes.  Because dim=8 is
            # a compile-time constant (constexpr) all branches are unrolled.
            min_val_i = gap[0]
            min_idx_i = 0
            for k in tl.static_range(1, 8):
                v = gap[k]
                cond = v < min_val_i
                min_val_i = tl.where(cond, v, min_val_i)
                min_idx_i = tl.where(cond, k, min_idx_i)

            # Build one-hot fix vector and direction sign
            lane_ids = tl.arange(0, 8)
            fix_mask_i = (lane_ids == min_idx_i)           # [8] bool
            # sign = +1 if (x - r_int) >= 0, else -1
            diff_i = x - r_int
            # Gather the diff at min_idx_i
            sign_val_i = tl.sum(tl.where(fix_mask_i, diff_i, tl.zeros_like(diff_i)), axis=0)
            sign_i = tl.where(sign_val_i >= 0.0, 1.0, -1.0)
            fix_i = tl.where(fix_mask_i, sign_i, 0.0)

            r_int = tl.where(is_odd, r_int + fix_i, r_int)

            # ---- Half-integer candidate -------------------------------------
            r_half = tl.math.round(x - 0.5) + 0.5

            s_h = tl.sum(r_half, axis=0)
            # Relaxed parity check (mirrors CPU: (sums_h * 2).round() % 2 != 0)
            s_h2 = tl.math.round(s_h * 2.0)
            is_odd_h = (tl.math.fmod(s_h2, 2.0) != 0.0)

            gap_h = tl.abs(x - r_half)

            min_val_h = gap_h[0]
            min_idx_h = 0
            for k in tl.static_range(1, 8):
                v = gap_h[k]
                cond = v < min_val_h
                min_val_h = tl.where(cond, v, min_val_h)
                min_idx_h = tl.where(cond, k, min_idx_h)

            fix_mask_h = (lane_ids == min_idx_h)
            diff_h = x - r_half
            sign_val_h = tl.sum(tl.where(fix_mask_h, diff_h, tl.zeros_like(diff_h)), axis=0)
            sign_h = tl.where(sign_val_h >= 0.0, 1.0, -1.0)
            fix_h = tl.where(fix_mask_h, sign_h, 0.0)

            r_half = tl.where(is_odd_h, r_half + fix_h, r_half)

            # ---- Pick closer candidate -------------------------------------
            d_int  = tl.sum((x - r_int)  ** 2, axis=0)
            d_half = tl.sum((x - r_half) ** 2, axis=0)

            result = tl.where(d_half < d_int, r_half, r_int)

            tl.store(out_ptr + offs, result)

    # ---------------------------------------------------------------------------
    # Per-group quantize kernel (scale per 8D group)
    # ---------------------------------------------------------------------------

    @triton.jit
    def e8_quantize_group_kernel(
        x_ptr,
        out_ptr,
        N: tl.constexpr,
        half_levels: tl.constexpr,   # levels / 2
        BLOCK: tl.constexpr,
    ):
        """Quantize with per-8D-group scaling (E8Lattice.quantize equivalent)."""
        pid = tl.program_id(0)

        for b in tl.static_range(BLOCK):
            g = pid * BLOCK + b
            if g >= N:
                break

            base = g * 8
            offs = base + tl.arange(0, 8)
            x = tl.load(x_ptr + offs)

            amax = tl.max(tl.abs(x), axis=0)
            amax = tl.where(amax < 1e-8, 1e-8, amax)
            sc = amax / tl.cast(half_levels, tl.float32)
            xn = x / sc

            # --- Integer candidate ---
            r_int = tl.math.round(xn)
            s = tl.sum(r_int, axis=0)
            is_odd = (tl.math.fmod(s, 2.0) != 0.0)
            gap = tl.abs(xn - r_int)
            lane_ids = tl.arange(0, 8)
            min_val_i = gap[0]; min_idx_i = 0
            for k in tl.static_range(1, 8):
                cond = gap[k] < min_val_i
                min_val_i = tl.where(cond, gap[k], min_val_i)
                min_idx_i = tl.where(cond, k, min_idx_i)
            fix_mask_i = (lane_ids == min_idx_i)
            diff_i = xn - r_int
            sv_i = tl.sum(tl.where(fix_mask_i, diff_i, tl.zeros_like(diff_i)), axis=0)
            fix_i = tl.where(fix_mask_i, tl.where(sv_i >= 0.0, 1.0, -1.0), 0.0)
            r_int = tl.where(is_odd, r_int + fix_i, r_int)

            # --- Half-integer candidate ---
            r_half = tl.math.round(xn - 0.5) + 0.5
            s_h2 = tl.math.round(tl.sum(r_half, axis=0) * 2.0)
            is_odd_h = (tl.math.fmod(s_h2, 2.0) != 0.0)
            gap_h = tl.abs(xn - r_half)
            min_val_h = gap_h[0]; min_idx_h = 0
            for k in tl.static_range(1, 8):
                cond = gap_h[k] < min_val_h
                min_val_h = tl.where(cond, gap_h[k], min_val_h)
                min_idx_h = tl.where(cond, k, min_idx_h)
            fix_mask_h = (lane_ids == min_idx_h)
            diff_h = xn - r_half
            sv_h = tl.sum(tl.where(fix_mask_h, diff_h, tl.zeros_like(diff_h)), axis=0)
            fix_h = tl.where(fix_mask_h, tl.where(sv_h >= 0.0, 1.0, -1.0), 0.0)
            r_half = tl.where(is_odd_h, r_half + fix_h, r_half)

            # --- Closer candidate ---
            d_int  = tl.sum((xn - r_int)  ** 2, axis=0)
            d_half = tl.sum((xn - r_half) ** 2, axis=0)
            lp = tl.where(d_half < d_int, r_half, r_int)

            hl = tl.cast(half_levels, tl.float32)
            lp = tl.minimum(tl.maximum(lp, -hl), hl)
            tl.store(out_ptr + offs, lp * sc)

    # ---------------------------------------------------------------------------
    # Per-head quantize: encode path (returns int8 codes + fp32 scale)
    # ---------------------------------------------------------------------------

    @triton.jit
    def e8_encode_kernel(
        x_ptr,        # float32 [M, head_dim_padded]  (M = total vectors)
        codes_ptr,    # int8    [M, head_dim_padded]   output codes
        scales_ptr,   # float32 [M]                   output scale per vector
        M: tl.constexpr,
        head_dim: tl.constexpr,   # original (unpadded) head dim
        head_dim_pad: tl.constexpr,  # padded to multiple of 8
        half_levels: tl.constexpr,
        BLOCK_VEC: tl.constexpr,  # vectors per program
    ):
        """Encode M vectors to int8 codes with per-vector fp32 scale.

        Each program handles BLOCK_VEC consecutive vectors.  Within each
        vector the 8D groups are processed sequentially; the inner loop is
        fully unrolled at compile time since head_dim_pad is constexpr.
        """
        pid = tl.program_id(0)
        n_groups = head_dim_pad // 8

        for bv in tl.static_range(BLOCK_VEC):
            row = pid * BLOCK_VEC + bv
            if row >= M:
                break

            # Compute per-vector scale from unpadded head_dim
            amax = 1e-8
            for d in tl.static_range(0, head_dim_pad, 8):
                offs_x = row * head_dim_pad + d + tl.arange(0, 8)
                # Only include dims within the original head_dim
                mask = (d + tl.arange(0, 8)) < head_dim
                v = tl.load(x_ptr + offs_x, mask=mask, other=0.0)
                amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))

            sc = amax / tl.cast(half_levels, tl.float32)
            tl.store(scales_ptr + row, sc)

            # Encode each 8D group
            for g in tl.static_range(n_groups):
                d = g * 8
                offs_x = row * head_dim_pad + d + tl.arange(0, 8)
                mask = (d + tl.arange(0, 8)) < head_dim
                x = tl.load(x_ptr + offs_x, mask=mask, other=0.0)
                xn = x / sc

                # Integer candidate
                r_int = tl.math.round(xn)
                s = tl.sum(r_int, axis=0)
                is_odd = (tl.math.fmod(s, 2.0) != 0.0)
                gap = tl.abs(xn - r_int)
                lane_ids = tl.arange(0, 8)
                min_val_i = gap[0]; min_idx_i = 0
                for k in tl.static_range(1, 8):
                    cond = gap[k] < min_val_i
                    min_val_i = tl.where(cond, gap[k], min_val_i)
                    min_idx_i = tl.where(cond, k, min_idx_i)
                fix_mask_i = (lane_ids == min_idx_i)
                diff_i = xn - r_int
                sv_i = tl.sum(tl.where(fix_mask_i, diff_i, tl.zeros_like(diff_i)), axis=0)
                fix_i = tl.where(fix_mask_i, tl.where(sv_i >= 0.0, 1.0, -1.0), 0.0)
                r_int = tl.where(is_odd, r_int + fix_i, r_int)

                # Half-integer candidate
                r_half = tl.math.round(xn - 0.5) + 0.5
                s_h2 = tl.math.round(tl.sum(r_half, axis=0) * 2.0)
                is_odd_h = (tl.math.fmod(s_h2, 2.0) != 0.0)
                gap_h = tl.abs(xn - r_half)
                min_val_h = gap_h[0]; min_idx_h = 0
                for k in tl.static_range(1, 8):
                    cond = gap_h[k] < min_val_h
                    min_val_h = tl.where(cond, gap_h[k], min_val_h)
                    min_idx_h = tl.where(cond, k, min_idx_h)
                fix_mask_h = (lane_ids == min_idx_h)
                diff_h = xn - r_half
                sv_h = tl.sum(tl.where(fix_mask_h, diff_h, tl.zeros_like(diff_h)), axis=0)
                fix_h = tl.where(fix_mask_h, tl.where(sv_h >= 0.0, 1.0, -1.0), 0.0)
                r_half = tl.where(is_odd_h, r_half + fix_h, r_half)

                d_int  = tl.sum((xn - r_int)  ** 2, axis=0)
                d_half = tl.sum((xn - r_half) ** 2, axis=0)
                lp = tl.where(d_half < d_int, r_half, r_int)

                hl = tl.cast(half_levels, tl.float32)
                lp = tl.minimum(tl.maximum(lp, -hl), hl)

                codes_out = tl.cast(lp, tl.int8)
                offs_c = row * head_dim_pad + d + tl.arange(0, 8)
                tl.store(codes_ptr + offs_c, codes_out)

    # ---------------------------------------------------------------------------
    # Decode kernel: int8 codes + scales → float32
    # ---------------------------------------------------------------------------

    @triton.jit
    def e8_decode_kernel(
        codes_ptr,   # int8    [M, head_dim_pad]
        scales_ptr,  # float32 [M]
        out_ptr,     # float32 [M, head_dim]  (unpadded, or head_dim_pad if same)
        M: tl.constexpr,
        head_dim: tl.constexpr,
        head_dim_pad: tl.constexpr,
        BLOCK_VEC: tl.constexpr,
    ):
        pid = tl.program_id(0)

        for bv in tl.static_range(BLOCK_VEC):
            row = pid * BLOCK_VEC + bv
            if row >= M:
                break

            sc = tl.load(scales_ptr + row)

            for g in tl.static_range(head_dim_pad // 8):
                d = g * 8
                offs = row * head_dim_pad + d + tl.arange(0, 8)
                mask = (d + tl.arange(0, 8)) < head_dim
                codes = tl.load(codes_ptr + offs)
                vals = tl.cast(codes, tl.float32) * sc
                tl.store(out_ptr + offs, vals, mask=mask)

    # ---------------------------------------------------------------------------
    # Fused dequant-matmul kernel
    # ---------------------------------------------------------------------------
    # Notation:
    #   K  = number of KV vectors (e.g. sequence length tokens × heads)
    #   H  = head_dim (= D, must be multiple of 8 for E8)
    #   Q  = query matrix, shape [H, H] — e.g. the W_Q projection or the query
    #        vector itself broadcast across KV positions
    #
    # For the common attention dequant-dot use-case we compute:
    #   out[k, :] = decode(codes[k, :], scales[k]) @ Q^T
    # i.e. for each KV token we decode its head_dim float vector and
    # immediately dot it against the query, avoiding materialising [K, H].
    #
    # out shape: [K, H] or [K, Q_cols] depending on Q shape.

    @triton.jit
    def e8_dequant_matmul_kernel(
        codes_ptr,   # int8    [K, H_pad]
        scales_ptr,  # float32 [K]
        q_ptr,       # float32 [H, Q_cols]   query / projection matrix
        out_ptr,     # float32 [K, Q_cols]
        K: tl.constexpr,
        H: tl.constexpr,         # original head_dim
        H_pad: tl.constexpr,     # padded to multiple of 8
        Q_cols: tl.constexpr,    # number of output columns
        BLOCK_K: tl.constexpr,   # KV rows per program
        BLOCK_Q: tl.constexpr,   # Q cols per tile
    ):
        """Fused decode + matrix multiply for attention.

        Each program handles BLOCK_K consecutive KV vectors and BLOCK_Q
        consecutive query columns.  The H-dimension reduction is accumulated
        in registers without materialising the decoded float tensor.

        Inner loop structure:
            for each 8D group g:
                decode 8 floats from int8 codes
                load 8-row slice of Q  [8, BLOCK_Q]
                accumulate outer product into acc [BLOCK_K, BLOCK_Q]
        """
        pid_k = tl.program_id(0)   # KV tile index
        pid_q = tl.program_id(1)   # Q-col tile index

        k_start = pid_k * BLOCK_K
        q_start = pid_q * BLOCK_Q

        # Accumulator
        acc = tl.zeros((BLOCK_K, BLOCK_Q), dtype=tl.float32)

        # Load per-vector scales for this KV tile
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        scales = tl.load(scales_ptr + k_offs, mask=k_mask, other=0.0)  # [BLOCK_K]

        q_offs = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < Q_cols

        n_groups = H_pad // 8
        for g in tl.static_range(n_groups):
            d = g * 8
            h_mask = (d + tl.arange(0, 8)) < H

            # Load codes for this group: [BLOCK_K, 8]
            # Flatten: codes_ptr[k, d:d+8]
            codes_offs = (k_offs[:, None]) * H_pad + (d + tl.arange(0, 8)[None, :])
            codes = tl.load(
                codes_ptr + codes_offs,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0,
            )
            # Dequantize: [BLOCK_K, 8]
            vals = tl.cast(codes, tl.float32) * scales[:, None]

            # Load Q slice: [8, BLOCK_Q]
            q_row_offs = (d + tl.arange(0, 8)[:, None]) * Q_cols + q_offs[None, :]
            q_slice = tl.load(
                q_ptr + q_row_offs,
                mask=h_mask[:, None] & q_mask[None, :],
                other=0.0,
            )

            # Accumulate: [BLOCK_K, 8] x [8, BLOCK_Q] -> [BLOCK_K, BLOCK_Q]
            acc += tl.dot(vals, q_slice)

        # Store result
        out_offs = k_offs[:, None] * Q_cols + q_offs[None, :]
        tl.store(out_ptr + out_offs, acc, mask=k_mask[:, None] & q_mask[None, :])


# ===========================================================================
# Python-facing API
# ===========================================================================

def _pad8(n: int) -> int:
    return n + (8 - n % 8) % 8


def e8_nearest_point(x: torch.Tensor) -> torch.Tensor:
    """GPU nearest-point on (relaxed) E8 lattice. Drop-in for E8Lattice.nearest_point.

    Args:
        x: (..., 8) float32 tensor.  Last dimension MUST be 8.

    Returns:
        Nearest E8 lattice point, same shape and dtype.
    """
    _require_triton()
    assert x.shape[-1] == 8, "last dim must be 8"
    shape = x.shape
    flat = x.contiguous().view(-1, 8).float()
    N = flat.shape[0]
    out = torch.empty_like(flat)

    BLOCK = 128  # groups per program; tune for L2 re-use vs. launch overhead
    grid = (triton.cdiv(N, BLOCK),)
    e8_nearest_point_kernel[grid](flat, out, N, BLOCK)

    return out.view(shape).to(x.dtype)


def e8_quantize_perhead(x: torch.Tensor, levels: int = 4) -> torch.Tensor:
    """GPU quantize with per-vector scale. Drop-in for E8Lattice.quantize_perhead.

    Each row (last dim = head_dim) gets one scale = amax / (levels/2).
    The 8D groups within that row share the per-vector scale, keeping
    overhead at 0.125 bits/dim.

    Args:
        x:      (..., head_dim) float tensor.  head_dim need not be mult of 8.
        levels: quantisation levels per dimension (4 = 2-bit, 8 = 3-bit).

    Returns:
        Quantised and dequantised tensor, same shape/dtype as x.
    """
    _require_triton()
    shape = x.shape
    head_dim = shape[-1]
    head_dim_pad = _pad8(head_dim)
    half_levels = levels // 2

    flat = x.reshape(-1, head_dim).contiguous().float()
    M = flat.shape[0]

    # Pad to multiple of 8
    if head_dim_pad != head_dim:
        flat_pad = torch.nn.functional.pad(flat, (0, head_dim_pad - head_dim))
    else:
        flat_pad = flat

    codes = torch.empty(M, head_dim_pad, dtype=torch.int8, device=x.device)
    scales = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_VEC = 4
    grid = (triton.cdiv(M, BLOCK_VEC),)
    e8_encode_kernel[grid](
        flat_pad, codes, scales,
        M, head_dim, head_dim_pad, half_levels, BLOCK_VEC,
    )

    # Decode back to float (used for the plain quantize_perhead path)
    out_pad = torch.empty(M, head_dim_pad, dtype=torch.float32, device=x.device)
    e8_decode_kernel[grid](codes, scales, out_pad, M, head_dim, head_dim_pad, BLOCK_VEC)

    out = out_pad[:, :head_dim].reshape(shape).to(x.dtype)
    return out


def e8_encode(
    x: torch.Tensor,
    levels: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode x to compact int8 codes + per-vector fp32 scales.

    Args:
        x:      (..., head_dim) float tensor.
        levels: quantisation levels (4 = 2-bit).

    Returns:
        codes:  int8 tensor [..., head_dim_padded]  (head_dim rounded up to 8)
        scales: float32 tensor [...] (one scalar per vector)

    Storage cost per vector of head_dim H (padded to H_p):
        codes:  H_p bytes (int8 per dimension)
        scales: 4 bytes
        total:  H_p + 4  bytes   vs.  2 * H bytes for float16
        ratio:  (2H) / (H_p + 4)   ≈ 2x for large H (on top of eviction)
    """
    _require_triton()
    shape = x.shape
    head_dim = shape[-1]
    head_dim_pad = _pad8(head_dim)
    half_levels = levels // 2

    flat = x.reshape(-1, head_dim).contiguous().float()
    M = flat.shape[0]

    if head_dim_pad != head_dim:
        flat_pad = torch.nn.functional.pad(flat, (0, head_dim_pad - head_dim))
    else:
        flat_pad = flat

    codes = torch.empty(M, head_dim_pad, dtype=torch.int8, device=x.device)
    scales = torch.empty(M, dtype=torch.float32, device=x.device)

    BLOCK_VEC = 4
    grid = (triton.cdiv(M, BLOCK_VEC),)
    e8_encode_kernel[grid](
        flat_pad, codes, scales,
        M, head_dim, head_dim_pad, half_levels, BLOCK_VEC,
    )

    codes_shape = shape[:-1] + (head_dim_pad,)
    scales_shape = shape[:-1]
    return codes.view(codes_shape), scales.view(scales_shape)


def e8_decode(
    codes: torch.Tensor,
    scales: torch.Tensor,
    levels: int = 4,          # kept for API symmetry; not needed for int8 path
    original_head_dim: int | None = None,
) -> torch.Tensor:
    """Decode int8 codes + scales back to float32.

    Args:
        codes:             int8 [..., head_dim_pad]
        scales:            float32 [...] (one per vector)
        levels:            quantisation levels (unused for int8 decode, kept for compat)
        original_head_dim: if provided, output is trimmed to this dim.
                           Otherwise assumed equal to codes.shape[-1].

    Returns:
        float32 [..., original_head_dim]
    """
    _require_triton()
    head_dim_pad = codes.shape[-1]
    head_dim = original_head_dim if original_head_dim is not None else head_dim_pad
    shape = codes.shape
    M = math.prod(shape[:-1])

    flat_codes = codes.reshape(M, head_dim_pad).contiguous()
    flat_scales = scales.reshape(M).contiguous().float()
    out = torch.empty(M, head_dim_pad, dtype=torch.float32, device=codes.device)

    BLOCK_VEC = 4
    grid = (triton.cdiv(M, BLOCK_VEC),)
    e8_decode_kernel[grid](
        flat_codes, flat_scales, out,
        M, head_dim, head_dim_pad, BLOCK_VEC,
    )

    out_shape = shape[:-1] + (head_dim,)
    return out[:, :head_dim].view(out_shape)


def e8_dequant_matmul(
    codes: torch.Tensor,
    scales: torch.Tensor,
    query: torch.Tensor,
    levels: int = 4,          # kept for API symmetry
) -> torch.Tensor:
    """Fused decode + matrix multiply — the hot path for compressed attention.

    Decodes the int8 E8-coded KV tensor and immediately multiplies with the
    query matrix without materialising the full float KV tensor.

    Args:
        codes:  int8  [K, H_pad]  — compressed KV vectors (from e8_encode)
        scales: float [K]         — per-vector scale
        query:  float [H, Q_cols] — query matrix (or single query broadcast)
                                    H must equal the original (unpadded) head_dim.
        levels: quantisation levels (unused in current int8 decode, kept for compat)

    Returns:
        float32 [K, Q_cols]   = dequant(codes, scales) @ query

    Example (attention score computation):
        # Compress the key cache once
        k_codes, k_scales = e8_encode(key_cache)  # key_cache [B, T, H]
        # At every decode step:
        scores = e8_dequant_matmul(k_codes, k_scales, query_vec.T)
    """
    _require_triton()
    assert codes.ndim == 2, "codes must be [K, H_pad]"
    assert query.ndim == 2, "query must be [H, Q_cols]"

    K, H_pad = codes.shape
    H, Q_cols = query.shape

    codes_c = codes.contiguous()
    scales_c = scales.contiguous().float()
    query_c = query.contiguous().float()
    out = torch.empty(K, Q_cols, dtype=torch.float32, device=codes.device)

    BLOCK_K = 32
    BLOCK_Q = 32
    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(Q_cols, BLOCK_Q))

    e8_dequant_matmul_kernel[grid](
        codes_c, scales_c, query_c, out,
        K, H, H_pad, Q_cols, BLOCK_K, BLOCK_Q,
    )
    return out

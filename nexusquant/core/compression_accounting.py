"""Byte-accurate compression measurement for E8-quantized KV cache.

Every overhead is counted:
  - E8 integer coordinates (int8)
  - Per-head fp16 scales
  - Temporal delta + zstd compressed index stream
  - PCA basis and mean vectors (if used)
  - A fixed header that records the config needed to decode

No rounding tricks. No partial counts.
"""

import struct
import numpy as np
import torch
import zstandard

from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.rope_utils import inverse_rope


# ---------------------------------------------------------------------------
# Header layout (24 bytes, always present)
#   [magic:4B] [n_layers:2B] [n_heads:2B] [head_dim:2B] [n_tokens:4B]
#   [bits:1B] [pca_dims:2B] [use_delta:1B] [zstd_level:1B] [reserved:5B]
# ---------------------------------------------------------------------------
_HEADER_MAGIC = b'NQC1'
_HEADER_SIZE = 24  # bytes


def _pack_header(n_layers, n_heads, head_dim, n_tokens,
                 bits, pca_dims, use_delta, zstd_level):
    pca = pca_dims if pca_dims is not None else 0
    return struct.pack(
        '<4sHHHIBHBBxxxxx',
        _HEADER_MAGIC,
        n_layers,
        n_heads,
        head_dim,
        n_tokens,
        bits,
        pca,
        int(use_delta),
        zstd_level,
    )


def _compress_with_delta_zstd(coords_int8, n_tokens, n_vectors_per_token, zstd_level=22):
    """Temporal delta + zstd on a flat int8 coordinate array.

    coords_int8: flat int8 array of all E8 coordinates
    n_tokens: sequence length
    n_vectors_per_token: n_layers * 2 * n_heads
    """
    coords_per_vector = len(coords_int8) // (n_tokens * n_vectors_per_token)
    reshaped = coords_int8.reshape(n_vectors_per_token, n_tokens, coords_per_vector)
    delta = np.zeros_like(reshaped)
    delta[:, 0, :] = reshaped[:, 0, :]
    delta[:, 1:, :] = reshaped[:, 1:, :] - reshaped[:, :-1, :]
    cctx = zstandard.ZstdCompressor(level=zstd_level)
    return cctx.compress(delta.astype(np.int8).tobytes())


def _compress_raw_zstd(coords_int8, zstd_level=22):
    cctx = zstandard.ZstdCompressor(level=zstd_level)
    return cctx.compress(coords_int8.astype(np.int8).tobytes())


def _decompress_with_delta(compressed, n_tokens, n_vectors_per_token, n_coords):
    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    coords_per_vector = n_coords // (n_tokens * n_vectors_per_token)
    delta = np.frombuffer(raw, dtype=np.int8).copy()
    reshaped = delta.reshape(n_vectors_per_token, n_tokens, coords_per_vector)
    recovered = np.cumsum(reshaped, axis=1).astype(np.int8)
    return recovered.reshape(-1)


def _decompress_raw(compressed):
    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    return np.frombuffer(raw, dtype=np.int8).copy()


def _apply_pca(vectors_2d, pca_dims):
    """Fit PCA and return (projected, Vh, mean).

    vectors_2d: (N, head_dim) numpy array
    Returns projected (N, pca_dims), Vh (pca_dims, head_dim), mean (head_dim,)
    all as fp32 numpy.
    """
    mean = vectors_2d.mean(axis=0)
    centered = vectors_2d - mean
    # SVD-based PCA
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    Vh_trunc = Vh[:pca_dims]          # (pca_dims, head_dim)
    projected = centered @ Vh_trunc.T  # (N, pca_dims)
    return projected, Vh_trunc, mean


def measure_compression(
    past_key_values,
    bits: int = 3,
    head_dim: int = 64,
    pca_dims: int = None,
    use_temporal_delta: bool = True,
    zstd_level: int = 22,
) -> dict:
    """Measure exact compression ratio with full overhead accounting.

    Parameters
    ----------
    past_key_values : DynamicCache from HuggingFace
        pkv.key_cache[layer] = tensor (1, n_heads, seq_len, head_dim)
    bits : int
        E8 quantization bits. 2 => levels=4, 3 => levels=8.
    head_dim : int
        Head dimension (auto-detected if cache tensors have enough dims).
    pca_dims : int or None
        If set, project each KV head down to this many dimensions before
        quantization. PCA is fit on the same data (in-sample).
    use_temporal_delta : bool
        Whether to apply temporal delta coding before zstd.
    zstd_level : int
        zstd compression level (1-22).

    Returns
    -------
    dict with:
        fp16_bytes               - original FP16 byte count
        index_raw_bytes          - E8 coordinate bytes before entropy coding
        index_compressed_bytes   - after temporal delta + zstd
        scale_bytes              - fp16 per-vector scales
        pca_basis_bytes          - PCA Vh + mean (0 if no PCA)
        header_bytes             - fixed metadata header
        total_compressed_bytes   - sum of all compressed components
        compression_ratio        - fp16_bytes / total_compressed_bytes
        bits_per_dim             - total_compressed_bytes * 8 / total_elements
        breakdown                - human-readable per-component string
    """
    levels = 2 ** bits

    # -----------------------------------------------------------------------
    # 1. Collect cache tensors
    #    Support both old DynamicCache (.key_cache list) and new DynamicCache
    #    (.to_legacy_cache() -> tuple of (K, V) tuples) and raw legacy tuples.
    # -----------------------------------------------------------------------
    if isinstance(past_key_values, tuple):
        # Already in legacy format: tuple of (K_tensor, V_tensor)
        legacy = past_key_values
    elif hasattr(past_key_values, 'key_cache'):
        # Old DynamicCache API
        legacy = tuple(
            (past_key_values.key_cache[i], past_key_values.value_cache[i])
            for i in range(len(past_key_values.key_cache))
        )
    else:
        # New DynamicCache API (transformers >= 4.45)
        legacy = past_key_values.to_legacy_cache()

    n_layers = len(legacy)
    # Auto-detect dimensions from the first layer K tensor
    ref = legacy[0][0]                       # (1, n_heads, n_tokens, head_dim)
    if ref.dim() == 4:
        n_heads = ref.shape[1]
        n_tokens = ref.shape[2]
        head_dim_actual = ref.shape[3]
    else:
        # (n_heads, n_tokens, head_dim) without batch dim
        n_heads = ref.shape[0]
        n_tokens = ref.shape[1]
        head_dim_actual = ref.shape[2]

    # Use the actual head_dim from the cache rather than the parameter default
    head_dim = head_dim_actual

    # Convenience accessors
    def _get_kv(layer_idx, is_value):
        t = legacy[layer_idx][1 if is_value else 0]
        if t.dim() == 4:
            return t.squeeze(0)   # (n_heads, n_tokens, head_dim)
        return t                  # already (n_heads, n_tokens, head_dim)

    # -----------------------------------------------------------------------
    # 2. Original FP16 byte count
    # -----------------------------------------------------------------------
    total_elements = n_layers * 2 * n_heads * n_tokens * head_dim
    fp16_bytes = total_elements * 2          # 2 bytes per fp16 element

    # -----------------------------------------------------------------------
    # 3. Header bytes (fixed, always written)
    # -----------------------------------------------------------------------
    header = _pack_header(
        n_layers, n_heads, head_dim, n_tokens,
        bits, pca_dims, use_temporal_delta, zstd_level
    )
    assert len(header) == _HEADER_SIZE
    header_bytes = _HEADER_SIZE

    # -----------------------------------------------------------------------
    # 4. Per-vector fp16 scales
    #    One scale per (layer, K/V, head, token) vector.
    #    For PCA: scale is on the projected vector, so same count.
    # -----------------------------------------------------------------------
    n_vectors = n_layers * 2 * n_heads * n_tokens
    scale_bytes = n_vectors * 2              # one fp16 per vector

    # -----------------------------------------------------------------------
    # 5. PCA basis bytes (per layer, per K/V, per head - independent basis)
    #    Vh : (pca_dims, head_dim) fp16 per head per K/V per layer
    #    mean: (head_dim,) fp16 per head per K/V per layer
    # -----------------------------------------------------------------------
    if pca_dims is not None:
        # Each head in each layer gets its own PCA basis
        n_bases = n_layers * 2 * n_heads
        vh_bytes_each = pca_dims * head_dim * 2   # fp16
        mean_bytes_each = head_dim * 2            # fp16
        pca_basis_bytes = n_bases * (vh_bytes_each + mean_bytes_each)
        quant_dim = pca_dims                      # quantize in projected space
    else:
        pca_basis_bytes = 0
        quant_dim = head_dim

    # -----------------------------------------------------------------------
    # 6. Quantize and extract integer coordinates
    # -----------------------------------------------------------------------
    # coords_per_vector: how many int8 values per input vector after E8
    # E8 groups 8 dims; each group yields 8 int8 coordinates
    groups_per_vector = (quant_dim + 7) // 8    # ceil, padding handled inside E8Lattice
    coords_per_vector = groups_per_vector * 8

    all_coords = []     # list of int8 arrays, one per (layer, kv, head, token) vector
    all_scales = []     # float, will store as fp16

    for layer_idx in range(n_layers):
        for kv_idx in range(2):                  # 0=key, 1=value
            tensor = _get_kv(layer_idx, is_value=(kv_idx == 1))
            # tensor: (n_heads, n_tokens, head_dim)

            # Remove RoPE from keys before compression (as per best practice)
            if kv_idx == 0:
                try:
                    tensor = inverse_rope(tensor)  # (n_heads, n_tokens, head_dim)
                except Exception:
                    pass  # skip if RoPE removal fails (e.g. non-Llama model)

            # Work in float32 for numerical accuracy
            tensor_f32 = tensor.float()          # (n_heads, n_tokens, head_dim)

            for head_idx in range(n_heads):
                head_data = tensor_f32[head_idx]  # (n_tokens, head_dim)
                vectors = head_data.numpy()        # (n_tokens, head_dim) float32

                if pca_dims is not None:
                    projected, Vh, mean = _apply_pca(vectors, pca_dims)
                    vectors = projected.astype(np.float32)  # (n_tokens, pca_dims)

                # Convert to torch for E8Lattice
                v_torch = torch.from_numpy(vectors)  # (n_tokens, quant_dim)

                # Per-vector (per-token) scaling
                amax = v_torch.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = amax / (levels / 2)          # (n_tokens, 1)

                # Normalize and quantize
                normalized = v_torch / scale          # (n_tokens, quant_dim)

                # Pad to multiple of 8 for E8 groups
                pad = (8 - quant_dim % 8) % 8
                if pad > 0:
                    normalized = torch.nn.functional.pad(normalized, (0, pad))

                # E8 nearest point per group
                flat = normalized.reshape(-1, 8)     # (n_tokens * groups_per_vector, 8)
                lp = E8Lattice.nearest_point(flat).clamp(-levels / 2, levels / 2)

                # Integer coordinates: multiply back by levels/2 to get integers
                # lp values are multiples of 0.5 or 1.0; rounding is exact
                coords_float = lp * (levels / 2)     # [-levels/2*levels/2 ... ] no, think again
                # lp is already in integer-or-half-integer lattice units
                # The actual integer index is round(lp): lp is the lattice point
                # in normalized space where scale = 1/(levels/2), so lp * (levels/2)
                # gives the integer coordinate in [-levels/2, levels/2]
                coords_int = coords_float.round().to(torch.int8).numpy()
                # Shape: (n_tokens * groups_per_vector * 8,)
                # Equivalently (n_tokens, groups_per_vector * 8) = (n_tokens, coords_per_vector)

                all_coords.append(coords_int.reshape(-1))
                all_scales.append(scale.squeeze(-1).numpy())  # (n_tokens,) fp32

    # -----------------------------------------------------------------------
    # 7. Concatenate and measure raw index bytes
    # -----------------------------------------------------------------------
    all_coords_flat = np.concatenate(all_coords, dtype=np.int8)  # total int8 count
    index_raw_bytes = len(all_coords_flat)  # 1 byte per int8 element

    # -----------------------------------------------------------------------
    # 8. Compress indices: temporal delta + zstd
    #    Layout for delta: (n_vectors_per_token, n_tokens, coords_per_vector)
    #    n_vectors_per_token = n_layers * 2 * n_heads
    # -----------------------------------------------------------------------
    n_vectors_per_token = n_layers * 2 * n_heads

    if use_temporal_delta:
        compressed_indices = _compress_with_delta_zstd(
            all_coords_flat, n_tokens, n_vectors_per_token, zstd_level
        )
    else:
        compressed_indices = _compress_raw_zstd(all_coords_flat, zstd_level)

    index_compressed_bytes = len(compressed_indices)

    # -----------------------------------------------------------------------
    # 9. Round-trip verification
    # -----------------------------------------------------------------------
    if use_temporal_delta:
        recovered = _decompress_with_delta(
            compressed_indices, n_tokens, n_vectors_per_token, len(all_coords_flat)
        )
    else:
        recovered = _decompress_raw(compressed_indices)

    if not np.array_equal(all_coords_flat, recovered):
        raise RuntimeError(
            f"Round-trip verification FAILED: "
            f"{np.sum(all_coords_flat != recovered)} mismatches out of {len(all_coords_flat)}"
        )

    # -----------------------------------------------------------------------
    # 10. Total and ratios
    # -----------------------------------------------------------------------
    total_compressed_bytes = (
        header_bytes
        + index_compressed_bytes
        + scale_bytes
        + pca_basis_bytes
    )

    compression_ratio = fp16_bytes / total_compressed_bytes
    bits_per_dim = (total_compressed_bytes * 8) / total_elements

    # -----------------------------------------------------------------------
    # 11. Human-readable breakdown
    # -----------------------------------------------------------------------
    def pct(b):
        return f"{b / total_compressed_bytes * 100:.1f}%"

    breakdown = (
        f"FP16 original:       {fp16_bytes:>10,} bytes\n"
        f"  header:            {header_bytes:>10,} bytes ({pct(header_bytes)})\n"
        f"  index raw:         {index_raw_bytes:>10,} bytes (uncompressed)\n"
        f"  index compressed:  {index_compressed_bytes:>10,} bytes ({pct(index_compressed_bytes)})\n"
        f"  scales (fp16):     {scale_bytes:>10,} bytes ({pct(scale_bytes)})\n"
        f"  pca basis:         {pca_basis_bytes:>10,} bytes ({pct(pca_basis_bytes)})\n"
        f"  TOTAL compressed:  {total_compressed_bytes:>10,} bytes\n"
        f"  ratio:             {compression_ratio:>10.3f}x\n"
        f"  bits/dim:          {bits_per_dim:>10.3f}"
    )

    return {
        'fp16_bytes': fp16_bytes,
        'index_raw_bytes': index_raw_bytes,
        'index_compressed_bytes': index_compressed_bytes,
        'scale_bytes': scale_bytes,
        'pca_basis_bytes': pca_basis_bytes,
        'header_bytes': header_bytes,
        'total_compressed_bytes': total_compressed_bytes,
        'compression_ratio': compression_ratio,
        'bits_per_dim': bits_per_dim,
        'breakdown': breakdown,
    }

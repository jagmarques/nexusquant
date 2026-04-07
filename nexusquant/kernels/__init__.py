"""NexusQuant GPU kernels — Triton implementations.

Exported functions are drop-in replacements for the CPU equivalents in
nexusquant.core.e8_lattice, plus encode/decode/fused-matmul primitives
for the compressed-attention hot path.

Usage
-----
    from nexusquant.kernels import (
        e8_nearest_point,      # drop-in for E8Lattice.nearest_point
        e8_quantize_perhead,   # drop-in for E8Lattice.quantize_perhead
        e8_encode,             # compress  → (int8 codes, fp32 scales)
        e8_decode,             # decompress← float32
        e8_dequant_matmul,     # fused decode + GEMM (attention hot-path)
    )

Requires: triton >= 2.1 (tested on 2.2 / 2.3, A100 / A10G).
Falls back gracefully: if Triton is not available, import raises ImportError
so callers can fall back to the CPU E8Lattice class.
"""

from nexusquant.kernels.e8_triton import (
    e8_nearest_point,
    e8_quantize_perhead,
    e8_encode,
    e8_decode,
    e8_dequant_matmul,
)

__all__ = [
    "e8_nearest_point",
    "e8_quantize_perhead",
    "e8_encode",
    "e8_decode",
    "e8_dequant_matmul",
]

from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix, fht, ifht
from nexusquant.core.dp_allocator import (
    dp_bit_allocation,
    calibrate_distortion_factors,
    DISTORTION_THEORETICAL,
    DISTORTION_EMPIRICAL,
)
from nexusquant.core.rope_utils import inverse_rope, forward_rope
from nexusquant.core.token_merger import merge_tokens, merge_and_drop
from nexusquant.core.entropy_coder import (
    encode_e8,
    decode_e8,
    measure_entropy,
    measure_e8_entropy,
    e8_quantize_with_entropy,
)
from nexusquant.core.temporal_codec import (
    compress_indices,
    decompress_indices,
    temporal_delta_encode,
    temporal_delta_decode,
    measure_compression,
)
from nexusquant.core.nsn import forward_nsn, inverse_nsn, NSNStats
from nexusquant.core.tcc import (
    forward_tcc,
    inverse_tcc,
    compute_compression_stats,
    TCCCompressed,
)
from nexusquant.core.optimal_shrinkage import (
    estimate_noise_sigma,
    optimal_shrinkage_frobenius,
    optimal_hard_threshold,
    effective_dims,
    variance_retained,
    reconstruct_with_shrinkage,
    mp_median,
)

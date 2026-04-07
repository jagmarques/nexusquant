"""NexusQuant: Training-Free KV Cache Compression via E8 Lattice VQ + Token Eviction.

GPU-validated numbers (Mistral-7B, A100, all overhead included):
  10.4x at +0.43% PPL (quality="high", 35% eviction)
  16.8x at +1.34% PPL (quality="balanced", 60% eviction)
  33.3x at +2.64% PPL (quality="max", 80% eviction)

Training-free. Zero calibration. One line of code. Apache 2.0.

Quick start:
    from nexusquant import nexusquant_evict

    with nexusquant_evict(model, quality="balanced"):
        output = model.generate(input_ids, max_new_tokens=100)
"""

__version__ = "0.4.0"

from nexusquant.pipeline import (
    NexusQuantFast,
    NexusQuantSimple,
    NexusQuantMax,
    NexusQuantEvict,
    compress_kv_cache,
)

from nexusquant.integrations.huggingface import (
    nexusquant,
    nexusquant_simple,
    nexusquant_max,
    nexusquant_evict,
)

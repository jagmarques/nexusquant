"""NexusQuant: Training-Free KV Cache Compression via E8 Lattice VQ + Token Eviction.

Two-tier compression:
  Quant-only: ~5x, lossless PPL, NIAH recall preserved. For quality-critical apps.
  Quant+evict: 10-33x, PPL +0.8-5%, NIAH degraded. For memory-critical apps.

GPU-validated numbers (all overhead included):
  ~5x  at ~0% PPL     (mode="quant_only", E8 3-bit, no eviction, Gemma-2-2b)
  ~9x  at +0.35% PPL  (quality="high",     K3V2, 35% eviction, Mistral-7B)
  ~17x at +0.82% PPL  (quality="balanced", K2V2, 60% eviction, Mistral-7B)
  ~33x at +2.13% PPL  (quality="max",      K2V2, 80% eviction, Mistral-7B)

Training-free. Zero calibration. One line of code. Apache 2.0.

Quick start (quant-only, lossless):
    from nexusquant import compress_kv_cache
    compressed_kv = compress_kv_cache(past_key_values, mode="quant_only")

Quick start (eviction, maximum compression):
    from nexusquant import nexusquant_evict
    with nexusquant_evict(model, quality="balanced"):
        output = model.generate(input_ids, max_new_tokens=100)
"""

__version__ = "0.5.0"

from nexusquant.pipeline import (
    NexusQuantFast,
    NexusQuantSimple,
    NexusQuantQuantOnly,
    NexusQuantAsymmetric,
    NexusQuantMax,
    NexusQuantEvict,
    NexusQuantEvictTruncate,
    compress_kv_cache,
)

from nexusquant.integrations.huggingface import (
    nexusquant,
    nexusquant_simple,
    nexusquant_max,
    nexusquant_evict,
)

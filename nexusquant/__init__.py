"""NexusQuant: Training-Free KV Cache Compression via E8 Lattice VQ + Token Eviction.

GPU-validated numbers (Mistral-7B, WikiText-2, all overhead included):
  ~9x  at +0.35% PPL  (quality="high",     K3V2, 35% eviction, A100, 3544-tok prefix)
  ~17x at +0.82% PPL  (quality="balanced", K2V2, 60% eviction, A10G, 1664-tok prefix)
  ~33x at +2.13% PPL  (quality="max",      K2V2, 80% eviction, A10G, 1664-tok prefix)

PPL delta is context-length dependent. Shorter prefixes (<500 tok) will show higher
degradation; longer prefixes (1664+ tok) benefit from better scorer discrimination.

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
    NexusQuantEvictTruncate,
    compress_kv_cache,
)

from nexusquant.integrations.huggingface import (
    nexusquant,
    nexusquant_simple,
    nexusquant_max,
    nexusquant_evict,
)

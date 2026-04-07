# Changelog

## [0.4.0] - 2026-04-07

### Added
- NexusQuantEvict pipeline class with attention-aware token eviction
- Context manager API: `with nexusquant_evict(model, quality="balanced")`
- Quality presets: high (10x), balanced (17x), max (33x)
- E8 lattice vector quantization with relaxed parity
- Hadamard incoherence preprocessing
- RoPE removal and reapplication for keys
- Temporal delta coding + zstd entropy compression
- Cross-architecture support (Mistral-7B, Llama-3-8B, TinyLlama)

### Validated
- 10-33x compression on Mistral-7B (A100 GPU)
- Context-length scaling: quality improves 5-6x from 500 to 2000 tokens
- 12 alternative approaches tested and documented

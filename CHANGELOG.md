# Changelog

## [0.5.0] - 2026-04-08

### Added
- **Real attention scorer**  - uses accumulated softmax weights (`attn_implementation='eager'`). Zero quality loss at 35% eviction (+0.00% PPL vs +0.66% with key-key proxy)
- **Asymmetric K/V compression**  - `key_bits` and `value_bits` params. K3V2 (3-bit keys, 2-bit values) gives 6.4x better quality than symmetric K2V2
- **Boundary layer protection**  - `protect_boundary=N` keeps first/last N layers at FP16. Mandatory for Qwen-family models
- **Physical KV tensor truncation**  - `truncate=True` physically removes evicted tokens, saving real GPU memory. RoPE remapped to contiguous positions
- **Deferred compression**  - `min_context_for_compression` skips compression for short contexts
- **Triton E8 GPU kernel**  - 5 kernels including fused dequant-matmul (written, benchmarking pending)
- **vLLM PagedAttention integration**  - compressed KV stored natively in paged blocks
- **New quality preset** `"asym"`  - K3V2 at 60% eviction
- Non-power-of-2 head_dim support (Phi-3 d=96 via zero-padding)

### Validated
- **3 models**: Mistral-7B, Phi-3-mini, Qwen2.5-7B (A100 + A10 GPUs)
- Real scorer: +0.00%/+0.16%/+0.66% at 35/60/80% eviction (Mistral-7B, A100)
- K3V2: +0.35% at 8.89x vs K2V2 +2.26% at 10.52x (Mistral-7B, A100)
- Qwen catastrophic without boundary protection  - confirmed TurboQuant+ finding
- K4V2 diminishing returns (+0.76% vs K3V2 +0.82%)

### Community
- Posted to vLLM, HuggingFace transformers, llama.cpp, ExLlamaV2, NVIDIA kvpress, mlx-lm
- Shared asymmetric K/V data with TurboQuant+ team

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

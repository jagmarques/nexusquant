<p align="center">
  <strong>NexusQuant</strong>
</p>
<p align="center">
  Compress your LLM's KV cache 10-33x. Training-free. One line of code.
</p>
<p align="center">
  <a href="https://pypi.org/project/nexusquant-kv/"><img src="https://img.shields.io/pypi/v/nexusquant-kv?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/jagmarques/nexusquant/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/jagmarques/nexusquant"><img src="https://img.shields.io/github/stars/jagmarques/nexusquant?style=social" alt="Stars"></a>
</p>

---

Token eviction + E8 lattice quantization, applied once after prefill. No training, no calibration data, no model modifications.

## Install

```bash
pip install nexusquant-kv
pip install "nexusquant-kv[hf]"  # with HuggingFace transformers
```

## Quickstart

```python
from nexusquant import nexusquant_evict

with nexusquant_evict(model, quality="balanced"):
    output = model.generate(input_ids, max_new_tokens=512)
```

## Why

| Without NexusQuant | With NexusQuant |
|---|---|
| 128K context → 80 GB KV cache | 128K context → 5 GB KV cache (17x) |
| OOM at 32K on a single A100 | 500K+ tokens on one A100 |
| Needs 8× A100 cluster for long context | Single GPU, single machine |
| Deploy a fine-tuned retrieval model | One `with` block, no code changes |

## Quality presets

Measured on Mistral-7B, Phi-3-mini, Qwen2.5-7B. Compression ratios include all overhead.

| Preset | Compression | PPL degradation | Context on 80 GB | Config |
|---|---|---|---|---|
| `high` | ~9x | <0.5% | ~1.2M tokens | K3V2 + real scorer + 35% evict |
| `asym` | ~14x | ~1% | ~1.8M tokens | K3V2 + 60% evict |
| `balanced` | ~17x | ~1.3% | ~2.2M tokens | K2V2 + 60% evict |
| `max` | ~33x | +0.66% | ~4.2M tokens | K2V2 + real scorer + 80% evict |

**NEW:** Asymmetric K/V compression (3-bit keys, 2-bit values) and real attention scorer dramatically improve quality. GPU-validated on Mistral-7B, Phi-3-mini, and Qwen2.5-7B across A100 and A10.

### Cross-architecture results (Cerebrium A10)

| Model | K2V2 35% | K3V2 35% | K2V2 60% | K3V2 60% |
|---|---|---|---|---|
| Mistral-7B (GQA 8:1) | +0.91% | +0.82% | +1.64% | +1.22% |
| Phi-3-mini (d=96) | +0.82% | +0.59% | +2.81% | +1.10% |
| Qwen2.5-7B | catastrophic | catastrophic | catastrophic | catastrophic |
| Qwen2.5-7B + boundary(2) | +7.9% | +8.7% | +23.8% | +23.3% |

> **Note:** Qwen-family models require `protect_boundary=2` (first/last 2 layers at FP16). Mistral and Phi-3 work without it.

## How it works

1. **Importance scoring** — rank tokens by attention weight. Two options: key-key proxy (fast, no extra pass) or **real attention scorer** (uses `attn_implementation='eager'`, zero quality loss at 35% eviction)
2. **Token eviction** — drop lowest-scoring tokens; always keep BOS and a recent sliding window
3. **RoPE removal** — undo rotary embeddings on keys so they share a common subspace
4. **Hadamard rotation** — spread energy uniformly across dimensions (handles non-power-of-2 head dims via zero-padding)
5. **E8 lattice quantization** — quantize 8-float groups onto the E8 root lattice. **Asymmetric:** 3-bit keys + 2-bit values (keys need more precision due to softmax amplification)
6. **Boundary protection** — optionally keep first/last N layers at FP16 (mandatory for Qwen-family)
7. **Delta coding + zstd** — consecutive tokens produce similar lattice indices; storing deltas then compressing with zstd yields another 2-3x

Token eviction reduces *count* (2.5x at 60% eviction). E8 quantization reduces *precision* (~7x after entropy coding). Combined: 17x.

## Compared to

| Method | Compression | PPL degradation | Training required | Notes |
|---|---|---|---|---|
| **NexusQuant (K3V2+scorer)** | **9-33x** | **+0.0-0.66%** | **No** | Includes eviction |
| **NexusQuant (K2V2)** | **10-33x** | **+0.4-2.6%** | **No** | Includes eviction |
| TurboQuant+ | 3.8-6.4x | ~0-1% | No | Quant-only, no eviction |
| KVTC (NVIDIA) | up to 20x | <1% | Yes (calibration) | |
| CommVQ (Apple) | ~8x | ~0% | Yes (retraining) | |
| Palu | 11x | ~25% rel | Yes (calibration) | |

NexusQuant ratios include token eviction (10-80% of tokens removed). TurboQuant+ ratios are pure quantization without eviction — not directly comparable. Competitor numbers from their papers.

## Supported models

Any HuggingFace causal LM using split-half RoPE (the standard since Llama-2):

- Llama family (Llama-2, Llama-3, Llama-3.1)
- Mistral / Mixtral
- Qwen
- Phi
- Gemma

Not yet supported: models with interleaved RoPE (GPT-NeoX, GPT-J).

## Limitations

- **Quality is text-dependent.** Creative/narrative text degrades more than structured/technical text. Test on your actual workload.
- **Short prefixes hurt.** Prefixes under 500 tokens see more degradation. The scorer needs enough tokens to distinguish signal from noise.
- **Architecture-dependent boundary protection.** Qwen-family models catastrophically fail without `protect_boundary=2`. Mistral and Phi-3 work without it. Always test your specific model.
- **E8 quantization is CPU-bound.** Triton GPU kernel is written (`nexusquant/kernels/e8_triton.py`) but not yet benchmarked for latency. Physical KV truncation (`truncate=True`) is implemented for actual VRAM savings.
- **Eviction is permanent.** Evicted tokens are gone. If your task requires precise recall of a specific token, measure eviction sensitivity first.
- **Results on 7B-class models only.** 70B validation pending.
- **Batch size > 1 is partially broken.** `NexusQuantSimple` only compresses batch index 0; other batch elements are silently dropped to the first element's compressed result. `NexusQuantEvictTruncate` computes one keep-mask from batch element 0 and applies it to all sequences — incorrect when sequences differ in importance. Validate batch inference results carefully.
- **Multi-turn chat (persistent KV cache) is not supported.** The hook compresses on every incoming prefill (seq > 1). If the same cache is reused across conversation turns, the second turn's user message triggers re-compression of an already-quantized cache. Use a fresh context manager per turn, or call `model.generate` with `past_key_values=None` to reset the cache between turns.
- **Speculative decoding is not supported.** Speculative decoding writes multiple draft tokens to the KV cache during the decode phase. Because the hook triggers on any batch of >1 new tokens, it will incorrectly fire on draft verification steps, compressing decode-phase tokens.
- **KV cache offloading is not supported.** `OffloadedCache` (used by HuggingFace's `accelerate` `max_memory` offloading) does not inherit from `DynamicLayer`, so the NexusQuant hooks do not intercept it. Compression silently does nothing when offloading is active.
- **Encoder-decoder models (T5, BART, Whisper) are not supported.** These models use cross-attention whose KV cache stores encoder representations rather than decoder tokens. RoPE removal in the pipeline assumes decoder self-attention with split-half rotary embeddings, which does not apply to T5-style relative position biases. Applying NexusQuant to encoder-decoder models will produce incorrect results.
- **Vision-language models (LLaVA, Qwen-VL, LLaVA-Next) are untested.** Model config detection handles nested `text_config`, but image tokens are scored for importance and evicted by the same heuristic as text tokens. High-information image tokens may be evicted. Results on VLMs have not been measured.
- **GGUF models are not supported.** GGUF format is typically run via llama.cpp or ctransformers, which do not use HuggingFace `DynamicCache`. The integration hooks have no effect. Only GPTQ/AWQ models loaded through `AutoModelForCausalLM` with HuggingFace are compatible.
- **rope_scaling (extended context) is not accounted for.** Models using linear or NTK rope scaling (e.g., Llama-3.1 at >8K context) read `rope_theta` but ignore `rope_scaling` config. At contexts beyond the original training length, the RoPE removal introduces a small frequency mismatch. Impact is unmeasured.

## Citation

```bibtex
@software{nexusquant2026,
  author  = {Marques, Jo\~{a}o Andr\'{e} Gomes},
  title   = {{NexusQuant}: Training-Free {KV} Cache Compression via {E8} Lattice Quantization and Attention-Aware Token Eviction},
  year    = {2026},
  url     = {https://github.com/jagmarques/nexusquant},
  license = {Apache-2.0},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

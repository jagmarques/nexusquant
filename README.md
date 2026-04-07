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

Measured on Mistral-7B, A100, FP16. Compression ratios include all overhead (scales, indices, metadata).

| Preset | Compression | PPL degradation | Context on 80 GB |
|---|---|---|---|
| `high` | 10x | +0.4% | ~1.3M tokens |
| `balanced` | 17x | +1.3% | ~2.2M tokens |
| `max` | 33x | +2.6% | ~4.2M tokens |

Validated on Mistral-7B, TinyLlama-1.1B, Llama-3-8B across academic, technical, and creative text.

## How it works

1. **Importance scoring** — rank tokens by cross-head attention weight (key-key dot product)
2. **Token eviction** — drop lowest-scoring tokens; always keep BOS and a recent sliding window
3. **RoPE removal** — undo rotary embeddings on keys so they share a common subspace, reducing quantization error ~0.7 pp
4. **Hadamard rotation** — spread energy uniformly across dimensions so no outlier dominates the quantization scale
5. **E8 lattice quantization** — quantize 8-float groups onto the E8 root lattice (densest sphere packing in 8D), 2 bits/dim
6. **Delta coding + zstd** — consecutive tokens produce similar lattice indices; storing deltas then compressing with zstd yields another 2-3x on the index stream

Token eviction reduces *count* (2.5x at 60% eviction). E8 quantization reduces *precision* (~7x after entropy coding). Combined: 17x.

## Compared to

| Method | Compression | PPL degradation | Training required |
|---|---|---|---|
| **NexusQuant** | **10-33x** | **+0.4-2.6%** | **No** |
| TurboQuant (Google) | ~5-6x | ~0% | No |
| KVTC (NVIDIA) | up to 20x | <1% | Yes (calibration, ~10 min) |
| CommVQ (Apple) | ~8x | ~0% | Yes (full retraining) |
| Palu | 11x | ~25% rel | Yes (calibration) |

NexusQuant is the highest-compression training-free method. KVTC achieves comparable ratios with better quality but requires calibration data. Competitor numbers are from their published papers, not reproduced on our hardware.

## Supported models

Any HuggingFace causal LM using split-half RoPE (the standard since Llama-2):

- Llama family (Llama-2, Llama-3, Llama-3.1)
- Mistral / Mixtral
- Qwen
- Phi
- Gemma

Not yet supported: models with interleaved RoPE (GPT-NeoX, GPT-J).

## Limitations

- **Quality is text-dependent.** Creative/narrative text degrades more than structured/technical text at the same compression ratio. Test on your actual workload before deploying.
- **Short prefixes hurt.** Prefixes under 500 tokens see more degradation than the numbers above, which were measured at 1600-3500 tokens. The importance scorer needs enough tokens to distinguish signal from noise.
- **E8 quantization is CPU-bound.** A production deployment needs Triton/CUDA kernels for the quantization step. The current implementation writes dequantized values back to the cache for compatibility — actual GPU memory savings require native compact storage.
- **Eviction is permanent.** Evicted tokens are gone. If your task requires precise recall of a specific token, measure eviction sensitivity on that task first.

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

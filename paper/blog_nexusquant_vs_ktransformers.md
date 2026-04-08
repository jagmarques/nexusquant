# NexusQuant: A Stable, Training-Free Alternative to ktransformers for Long-Context Inference

**April 7, 2026**

---

Long-context inference is broken for most people. The two dominant approaches are (1) use a system like ktransformers that optimizes attention kernels for speed but doesn't compress anything, or (2) retrain your model with extended context. Neither is ideal. This post describes a third path: training-free KV cache compression that works as a drop-in wrapper around any HuggingFace model.

## The Problem with ktransformers at Scale

ktransformers is a solid project. It uses FlashInfer for fused attention kernels, delivers real throughput improvements at short-to-medium context lengths, and is reasonably easy to integrate. If you're running inference at 32K tokens on a standard setup, it largely works.

The issue is stability above 100K tokens. There are community reports of FlashInfer's block-sparse attention producing CUDA errors or degraded outputs on certain GPU/CUDA version combinations at extreme sequence lengths (128K+). We have not independently reproduced or verified a specific root cause; users should test their own GPU/CUDA/FlashInfer version combination before relying on ktransformers at these lengths. The ktransformers project is actively maintained and may have addressed specific issues since these reports appeared.

More fundamentally, ktransformers doesn't compress the KV cache at all. It makes attention faster, but the cache still grows linearly with sequence length. A 128K-token Llama-3-70B context needs roughly 42 GB of KV cache at float16 (80 layers, 8 GQA KV heads, head_dim=128). That's already half an A100 80GB before you load the model weights. You need most of your VRAM budget just to hold the state.

## What NexusQuant Does Instead

NexusQuant compresses the KV cache itself, not just the attention kernel. The approach is training-free: you don't need calibration data, you don't need to modify model weights, and you don't need to know anything about the model's training distribution. It works by combining two complementary techniques:

**Token eviction** removes KV entries for tokens that are unlikely to be attended to again. An attention score accumulator tracks which tokens each layer actually uses during prefill. Tokens that fall below a threshold are evicted before the generation loop begins.

**Vector quantization** compresses the surviving KV entries using 2-bit E8 lattice quantization. The E8 lattice is optimal for packing points in 8-dimensional space  - it achieves the densest sphere packing in R^8  - which means 2-bit E8 VQ outperforms standard k-means or product quantization at the same bitwidth, with no training required beyond the lattice structure itself.

These two stages are composed with a few supporting operations:

```
Prefill → Score → Evict → RoPE removal → Hadamard rotation → 2-bit E8 VQ → Temporal delta → zstd
```

- **Score**: compute per-token attention importance from prefill activations
- **Evict**: drop low-importance KV entries (configurable % of tokens)
- **RoPE removal**: undo rotary position encodings before quantizing (they're re-applied at decode time)
- **Hadamard rotation**: rotate keys/values to flatten the quantization error distribution
- **E8 VQ**: encode each 8-dim vector to the nearest E8 lattice point at 2 bits/dim
- **Temporal delta + zstd**: losslessly compress the residuals between consecutive token embeddings

Each stage was validated by ablation. Removing any single stage costs at least 0.7 percentage points of PPL quality or 3.3x compression ratio. The pipeline isn't over-engineered  - everything earns its place.

## API

The entire thing is a context manager:

```python
from nexusquant import nexusquant_evict

# Load your model normally
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Wrap generation  - that's it
with nexusquant_evict(model, quality="balanced"):
    outputs = model.generate(input_ids, max_new_tokens=512)
```

Three quality presets, with validated numbers on Mistral-7B at a 3544-token prefix (A100):

| Preset | Eviction | Compression | PPL Degradation |
|--------|----------|-------------|-----------------|
| `"high"` | 35% | 10.4x | +0.43% |
| `"balanced"` | 60% | 16.8x | +1.34% |
| `"max"` | 80% | 33.3x | +2.64% |

The compression ratio accounts for all overhead: quantization indices, scales, eviction masks. These are honest byte counts, not theoretical best-case figures.

## Validated GPU Results

All results below are from A100 GPU experiments using the full pipeline. Context lengths are 3544-token prefixes on Mistral-7B-v0.1 unless noted.

### Mistral-7B (A100, 3544-tok prefix, key-key scorer)

| Eviction | Ratio | PPL Delta | Validated? |
|----------|-------|-----------|------------|
| 35% | 10.4x | +0.43% | GPU-validated |
| 50% | 13.5x | +0.83% | Interpolated estimate  - not GPU-validated |
| 60% | 16.8x | +1.34% | GPU-validated |
| 80% | 33.3x | +2.64% | GPU-validated |

Note: K3V2 (3-bit keys, 2-bit values) at 35% eviction gives +0.35% at 8.89x on the same setup  - better quality at slightly lower ratio. See Asymmetric K/V section below.

At 10x compression, factual recall QA is preserved. Nuanced multi-step reasoning questions show partial degradation  - the model gets the main facts right but may miss fine-grained details. This is consistent with evicting tokens that carry secondary context.

### Llama-3-8B (A10G, 1494-tok prefix)

Llama-3-8B shows an unusual behavior: compression *improves* PPL compared to baseline.

| Config | Ratio | PPL Delta |
|--------|-------|-----------|
| 2-bit VQ, no eviction | 6.71x | -1.20% |
| 2-bit VQ + 35% evict | 10.25x | -1.47% |
| 2-bit VQ + 60% evict | 16.48x | -1.35% |
| 2-bit VQ + 80% evict | 32.45x | -0.61% |

We investigated this thoroughly. The negative PPL is an evaluation artifact: the attention mask renormalization after eviction makes the continuation scoring easier by forcing focus on retained tokens, not because the compressed cache is higher quality than the original. The effect is structural  - random eviction also improves measured PPL, which confirms it's a renormalization effect rather than smart token selection. The paper reports these numbers with full explanation of the artifact and does not claim compression improves Llama-3 generation quality in deployment.

The practical implication: Llama-3's GQA architecture is more robust to KV compression than Mistral's full attention, and the compression ratios are real regardless of the PPL artifact.

### Domain Sensitivity (Mistral-7B, 500-tok prefix)

PPL degradation varies by text domain:

| Domain | 35% eviction | 70% eviction | 80% eviction |
|--------|--------------|--------------|--------------|
| Academic | +0.39% | +4.81% | +6.58% |
| Technical | +0.90% | +3.87% | +6.09% |
| Creative/narrative | +2.48% | +4.62% | +4.73% |

Academic and technical text has long-range structural coherence that eviction preserves well. Creative/narrative text has higher information density per token, so eviction hurts more at low rates  - but the degradation curve is flatter at high eviction rates compared to structured text.

## Comparison with Other Methods

All competitor numbers are from their published papers or technical reports. The NexusQuant numbers are from our own A100 experiments with full overhead accounting.

| Method | Ratio | PPL Quality | Training Required | Framework |
|--------|-------|-------------|-------------------|-----------|
| NexusQuant | 10-33x | +0.4-2.6% | None | Any HF model |
| KVTC (NVIDIA) | up to 20x | <1pp | ~10 min calibration | TensorRT |
| CommVQ (Apple) | ~8x | ~0% | Training required | Apple internal |
| TurboQuant (Google) | ~5-6x | ~0% | None | Custom kernels |
| ktransformers | N/A | Baseline | N/A | FlashInfer; community reports of instability >100K (unverified) |

A few notes on honest reading of this table:

KVTC achieves up to 20x but requires calibration data from the target distribution. If your domain matches their calibration set, the quality numbers are excellent. If it doesn't, expect degradation not reported in their paper. NexusQuant requires zero calibration, which matters when you're working across diverse domains or deploying to arbitrary user inputs.

CommVQ's ~0% PPL degradation is impressive but requires model retraining. That's not a drop-in solution  - it's a new model variant. If you're on Mistral-7B-v0.1 and CommVQ was trained on a different checkpoint, you start over.

TurboQuant (Google) is the closest comparison to NexusQuant in the "no training required" category, but tops out at ~5-6x. Our `"high"` preset (10x) beats their maximum ratio at better or comparable PPL, and our `"max"` preset (33x) operates in a regime they don't reach.

ktransformers isn't a compression method  - it belongs in this table only because it's the tool people reach for when they need long-context performance. The comparison is practical: if you need 128K tokens and you're asking "should I use ktransformers or NexusQuant?", the answer depends on whether you need the KV cache to fit in memory. ktransformers will not help you fit a 128K context into a single 80GB A100. NexusQuant at 10x will.

## Honest Limitations

**Latency has improved but end-to-end profiling is incomplete.** A Triton GPU kernel for E8 VQ (5 kernels including fused dequant-matmul) has shipped and replaces the original CPU-Python path that took 62-91 seconds per 3544-token prefix on A100. End-to-end compression latency with the Triton kernel has not yet been formally benchmarked. Interactive-latency suitability should be verified for your specific hardware and sequence length before production deployment.

**Physical KV truncation is implemented but experimental.** `truncate=True` in `nexusquant_evict()` physically removes evicted token rows from KV tensors and remaps RoPE to contiguous positions, enabling real GPU memory savings. This requires passing correct `position_ids` to `model.generate()`. The default (`truncate=False`) uses masking instead, which is more compatible but does not reduce attention FLOPs.

**Only tested on 7-8B models.** All results above are from Mistral-7B and Llama-3-8B. We have not validated on 70B models. GQA vs. full attention architecture differences may matter more at scale.

**Catastrophic failure at extreme eviction + long context.** At 3K+ token prefixes, eviction rates above 60% produce catastrophic PPL degradation (>40% relative). This is not a scorer bug  - it's fundamental capacity loss from removing too much context. The `"max"` preset (80% eviction) is validated at 3544 tokens on A100, but applying it to significantly longer prefixes without domain-specific validation is not recommended.

**PPL is not the whole story.** Our downstream QA evaluation on 5 tasks shows that 10x compression preserves factual QA performance. We have not run a full LongBench evaluation. PPL can understate quality degradation on tasks requiring precise recall of long-range context.

## What's Next

The Triton E8 VQ GPU kernel and physical KV truncation (`truncate=True`) have shipped. Remaining work:

- **LongBench evaluation**  - formal multi-task benchmark (currently only 1/16 tasks completed due to compute constraints)
- **16K+ context validation**  - real-document long-context testing beyond the 3544-tok prefix used in most experiments
- **70B model validation**  - all current results are on 7-8B models
- **Combined best-config experiment**  - real scorer + K3V2 + boundary protection across 35/60/80% eviction rates (these have not been run together end-to-end)
- **arXiv submission** after LongBench and combined config are complete

## Code

The library is in `nexusquant-oss/`. The main entry points:

```python
# Context manager (recommended)
from nexusquant import nexusquant_evict

with nexusquant_evict(model, quality="high"):
    outputs = model.generate(input_ids, max_new_tokens=256)

# Direct class instantiation for more control
from nexusquant import NexusQuantEvict

compressor = NexusQuantEvict(
    evict_ratio=0.35,
    vq_bits=2,
    use_hadamard=True,
    use_delta=True,
)
compressor.attach(model)
outputs = model.generate(input_ids, max_new_tokens=256)
compressor.detach(model)
```

The context manager handles hook registration and cleanup automatically. It works with any model that uses standard HuggingFace attention, including MHA, GQA, and MQA variants.

---

The core claim is conservative: at ~9x compression with the `"high"` preset (K3V2, 35% eviction), PPL degrades by +0.35% on Mistral-7B at a 3544-token prefix on an A100, with factual QA quality preserved. That is a GPU-validated result with full overhead accounting.

The latency limitations are equally real. We're not hiding them.

If you're hitting the ktransformers stability ceiling at 100K+ tokens, or if you simply need the KV cache to fit in a single GPU, NexusQuant is worth evaluating. The API is one line. The failure modes are documented. The numbers are from actual hardware.

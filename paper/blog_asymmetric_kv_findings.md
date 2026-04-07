# Why 3-bit keys + 2-bit values beats symmetric 2-bit: GPU-validated asymmetric KV cache compression

We've been running a lot of GPU experiments on KV cache compression, and one finding keeps surprising people: **keys need fundamentally different treatment than values**.

## The scorer result: zero degradation at 35% eviction

When using the real attention scorer, evicting 35% of KV tokens causes **+0.00% perplexity degradation** (A100, 764-token prefix, Mistral-7B). At 80% it's +0.66%.

The standard key-key proxy scorer gives +0.66% at 35% and +3.20% at 80%.

| Evict% | Key-Key | Real Scorer | Improvement |
|--------|---------|-------------|-------------|
| 35% | +0.66% | +0.00% | -0.66pp |
| 60% | +1.07% | +0.16% | -0.91pp |
| 80% | +3.20% | +0.66% | -2.54pp |

The real scorer runs one prefill forward pass with `output_attentions=True`, accumulates softmax weights column-wise across layers, and averages over heads. Requires `attn_implementation='eager'` — SDPA silently returns `None` for attention weights.

## Why keys need more bits

K3V2 (3-bit keys, 2-bit values) drops PPL delta from +2.26% to +0.35% on Mistral-7B at 35% eviction — a **6x improvement** at 15% ratio cost.

The mechanism: keys participate in softmax. Quantization noise in keys perturbs the full attention weight matrix — redistributing probability mass across *all* positions. Value noise only scales output proportionally to that value's attention weight. Keys are in the exponent; values are linear.

## Cross-architecture: Mistral, Phi-3, Qwen

| Model | K2V2 35% | K3V2 35% | K2V2 60% | K3V2 60% |
|-------|----------|----------|----------|----------|
| Mistral-7B (GQA 8:1) | +0.91% | +0.82% | +1.64% | +1.22% |
| Phi-3-mini (d=96) | +0.82% | +0.59% | +2.81% | +1.10% |
| Qwen2.5-7B | catastrophic | catastrophic | catastrophic | catastrophic |
| Qwen2.5-7B + boundary(2) | +7.9% | +8.7% | +23.8% | +23.3% |

**Phi-3** has head_dim=96 (non-power-of-2). We pad to 128 before the Hadamard transform, slice back after. Works fine.

**Qwen** requires boundary layer protection. Without `protect_boundary=2`, symmetric quantization gives +966,934% PPL. The first/last layers are critical for Qwen's context anchoring.

## K4V2: diminishing returns

K4V2 gives +0.76% vs K3V2's +0.82% at 35% eviction — barely any improvement. The bottleneck shifts to value quantization once key precision crosses a threshold. Don't bother with K4V2; spend the bits elsewhere.

## TurboQuant+ found the same thing

TurboQuant+ (TheTom) independently arrived at 3-bit keys + 2-bit values from a completely different approach (random orthogonal rotations + Lloyd-Max scalar quantization). Two teams, different theory, same conclusion. That's meaningful signal.

**Important:** NexusQuant's 8.89x includes token eviction. TurboQuant's ~5x is pure quantization. Not directly comparable ratios.

## Practical recommendations

- **Default:** K3V2 (`quality="high"`) for quality-critical tasks
- **Add `protect_boundary=2`** for Qwen-family models (mandatory)
- **Use real scorer** when you can load with `attn_implementation='eager'`
- **Skip K4V2** — diminishing returns, not worth the ratio cost

Code: https://github.com/jagmarques/nexusquant

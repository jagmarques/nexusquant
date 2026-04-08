# What didn't work: negative results from KV cache compression experiments

Most ML papers report what worked. We are going to do the opposite.

Over the past few months of GPU experiments on NexusQuant, we have accumulated a set of ideas that seemed reasonable, had clear theoretical motivation, and failed on contact with actual hardware. We are writing them up here because negative results are useful, they save other people time, and they are almost never published.

All experiments are Mistral-7B unless stated otherwise.

---

## 1. Soft eviction: 1-bit tokens are worse than zeros

After evicting tokens from the KV cache, you can either mask them out (set attention logits to `-inf`) or keep them in degraded form. We tried the second option: quantize evicted tokens to 1-bit E8 and keep them, reasoning that a noisy signal is better than no signal.

It is not.

At 35% eviction, soft eviction gives +2.24% PPL degradation versus +0.82% for hard masking. At 60%, it is +2.39% versus +1.22%. Soft eviction is consistently 2-3x worse.

The reason is that 1-bit quantization is aggressive enough to destroy the signal but not aggressive enough to produce zero weight. Evicted tokens still receive nonzero softmax attention, and their corrupted vectors pull the output in random directions. Masking to `-inf` is cleaner: those positions simply do not exist from the model's perspective.

The broader lesson is that "something is better than nothing" does not hold once the something is corrupted past a threshold. If you want a softer version of eviction, use a lower eviction rate. Do not use noisier tokens.

---

## 2. K4V2: diminishing returns that aren't worth it

K3V2 (3-bit keys, 2-bit values) gives +0.82% PPL on Mistral-7B at 35% eviction. K4V2 gives +0.76%. That is a 0.06pp improvement for an extra bit on every key vector in every layer.

The bottleneck shifts. Once key precision crosses a threshold, value noise dominates. Adding more bits to keys past that point buys almost nothing. The right configuration is K3V2, not K4V2. If you need better quality, either lower the eviction rate or use the real attention scorer.

This is also why we stopped at K3V2 rather than searching over K4V2, K5V2, K3V3, and so on. The space of configurations is large and the marginal gains collapse quickly. K3V2 is the Pareto-optimal point for this model family.

---

## 3. 8K context catastrophe with the key-key scorer

The key-key proxy scorer works well at 1664-token prefixes (10.4x at +0.14% PPL) and degrades gracefully at 2924 tokens (10.5x at +1.5%). At longer contexts with high eviction rates, it falls apart.

We tested 60%+ eviction at 2924-token prefixes and got catastrophic failure (>42% PPL degradation). The scorer relies on aggregating query-key attention over recent positions to estimate token importance. At very long contexts, the attention distribution becomes more diffuse and the scorer loses the ability to reliably distinguish important tokens from noise. It is not a scorer deficiency you can fix by tuning -- we tested twelve alternative scorers and they all fail in the same regime.

The practical boundary is: at 2924 tokens, do not exceed 35% eviction. The real attention scorer, which uses true softmax weights from a prefill forward pass, closes most of this gap, but it has only been validated up to 764 tokens and requires eager attention mode.

---

## 4. Qwen symmetric compression catastrophe

We tested K2V2 symmetric quantization on Qwen2.5-7B. Without boundary protection, every configuration produces catastrophic degradation. Not "worse than expected" -- we are talking +966,934% PPL. The model stops making predictions that resemble natural language.

Adding `protect_boundary=2` (keeping the first and last 2 layers in FP16) brings this down to +7.9% at 35% eviction and +23.8% at 60%. Those numbers are still too large for production use. K3V2 asymmetric quantization does not help -- it actually regresses slightly versus K2V2 on Qwen, the opposite of what happens on Mistral and Phi-3.

The root cause is that Qwen uses dense GQA with tightly coupled layer interactions. The first and last layers do heavy lifting for context anchoring in a way that is much more sensitive to quantization noise than Mistral's architecture. NexusQuant is not ready for Qwen-family models without architecture-specific work. If you are building on Qwen, this is a hard blocker.

---

## Why share this

Negative results take the same GPU time to produce as positive results. They are just as informative. When we read papers that only report what worked, we end up re-running the same failed experiments ourselves -- and so does everyone else.

The compression space is big enough that it is easy to spend weeks on ideas that do not pan out. Soft eviction, K4V2, long-context scorer failure, Qwen collapse -- these are real-time sinks that we ran into and documented. If this post saves one person from repeating them, it was worth writing.

Code: https://github.com/jagmarques/nexusquant

"""Investigate why token eviction IMPROVES perplexity (negative PPL delta).

Three competing hypotheses:
  H1: Attention mask renormalization — softmax over fewer tokens concentrates
      attention on high-quality tokens, reducing noise.
  H2: Short sequences / small model — 439-token prefix on TinyLlama means many
      tokens are genuinely low-importance; removing them helps.
  H3: Quantization noise cancellation — fewer quantized tokens = less total
      quantization noise injected into the KV cache.

Each hypothesis has a direct falsification test. We measure PPL deltas and
look for which manipulation eliminates or reproduces the negative-delta effect.

KV cache access convention (this transformers version uses DynamicCache with
DynamicLayer objects):
  kv.layers[l].keys   -> (1, n_heads, seq, head_dim)  float32 or float16
  kv.layers[l].values -> (1, n_heads, seq, head_dim)
  kv[l][0]            -> same as kv.layers[l].keys  (via __getitem__)
  kv[l][1]            -> same as kv.layers[l].values

We always deep-copy the cache before modifying it so kv_raw stays pristine.
"""

import sys, os, copy, math, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nexusquant-oss"))
from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.rope_utils import inverse_rope, forward_rope


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {m}...", flush=True)
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
    model.eval()
    return model, tok


def get_text():
    return (
        "The Standard Model of particle physics is the theory describing three "
        "of the four known fundamental forces in the universe, as well as "
        "classifying all known elementary particles. It was developed in stages "
        "throughout the latter half of the 20th century, through the work of "
        "many scientists around the world, with the current formulation being "
        "finalized in the mid-1970s upon experimental confirmation of the "
        "existence of quarks. The Standard Model explains how the basic building "
        "blocks of matter interact, governed by four fundamental forces. "
        "Fermions are the building blocks: six quarks and six leptons. Forces "
        "between the fermions are mediated by gauge bosons. The Higgs mechanism "
        "gives mass to some particles through spontaneous symmetry breaking. "
        "The photon mediates the electromagnetic force between electrically "
        "charged particles. The W and Z bosons mediate the weak force. The "
        "eight gluons mediate the strong force between quarks. The graviton is "
        "hypothesized to mediate the gravitational force, but is not part of "
        "the Standard Model. The model predicted the existence of the W and Z "
        "bosons, the gluon, the top quark and the charm quark, and their "
        "predicted properties were experimentally confirmed with good precision. "
        "The Higgs boson was predicted in 1964 and finally observed in 2012 at "
        "CERN. The Standard Model has been tested extensively. Because it fails "
        "to incorporate gravity and dark matter, many physicists believe it is "
        "incomplete. Extensions include supersymmetry, extra dimensions, and "
        "string theory. "
        "The Roman Empire was the post-Republican period of ancient Roman "
        "civilization. It had a government headed by emperors and large "
        "territorial holdings around the Mediterranean Sea in Europe, North "
        "Africa, and Western Asia. The city of Rome was the largest city in the "
        "world from around 100 BC to 400 AD, with Constantinople becoming the "
        "largest around 500 AD. The Empire was among the most powerful economic, "
        "cultural, political and military forces in the world of its time. "
        "It was the largest empire of the classical antiquity period. At its "
        "height under Trajan it covered five million square kilometers. Roman "
        "economy was based on agriculture. Trade was conducted throughout the "
        "Empire and beyond. The Romans built roads aqueducts and bridges. "
        "Latin was the common language. Roman law influenced many modern legal "
        "systems. The Empire was eventually divided into Eastern and Western "
        "halves. The Western Roman Empire fell in 476 AD when Odoacer deposed "
        "Romulus Augustulus. The Eastern Roman Empire survived until 1453. "
        "Artificial neural networks are computing systems inspired by "
        "biological neural networks. They learn to perform tasks by considering "
        "examples generally without being programmed with specific rules. "
        "Neural networks consist of connected artificial neurons organized in "
        "layers. Deep learning uses networks with many layers. Convolutional "
        "networks excel at image recognition. Recurrent networks handle "
        "sequential data. Transformers use attention mechanisms and have "
        "revolutionized natural language processing. Large language models "
        "like GPT and Claude demonstrate remarkable capabilities in text "
        "generation reasoning and analysis. The human immune system defends "
        "the body against pathogens including bacteria viruses and fungi. "
        "The innate immune system provides immediate non-specific response. "
        "The adaptive immune system provides specific responses and retains "
        "memory. T cells coordinate immune response. B cells produce "
        "antibodies. Vaccines train the adaptive immune system. The Pacific "
        "Ocean is the largest and deepest oceanic division covering about "
        "46 percent of Earth water surface. The Mariana Trench is the "
        "deepest point at 10994 meters. The Ring of Fire follows the Pacific "
        "plate edges with 75 percent of active volcanoes. Ocean currents "
        "transport heat and nutrients across vast distances. Number theory "
        "studies integers and their properties. Gauss called it the queen "
        "of mathematics. The Riemann hypothesis remains unsolved. Prime "
        "numbers are central. The prime number theorem gives asymptotic "
        "density. Fermat last theorem was proved by Andrew Wiles in 1995."
    )


def clone_kv(kv_raw):
    """Deep-copy a DynamicCache, returning a new independent cache."""
    return copy.deepcopy(kv_raw)


def compute_ppl(model, full_ids, prefix_len, kv, attn_mask=None):
    """Compute PPL of continuation tokens given a KV cache.

    Uses use_cache=False so the continuation forward pass does NOT append
    any tokens to the DynamicCache in-place.
    """
    cont_ids = full_ids[:, prefix_len:]
    with torch.no_grad():
        out = model(cont_ids, past_key_values=kv,
                    attention_mask=attn_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, prefix_len + 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return torch.exp(loss).item()


def score_importance(kv, n_layers, obs_window=32, pool_kernel=5):
    """Key-key attention importance scores (SnapKV-style).

    Returns tensor of shape [seq_len] using the prefix KV length.
    """
    seq_len = kv.layers[0].keys.shape[2]
    head_dim = kv.layers[0].keys.shape[3]
    w = min(obs_window, seq_len)
    all_importance = torch.zeros(seq_len)

    for l in range(n_layers):
        k = kv.layers[l].keys[0].float()  # (n_heads, seq, dim)
        k_obs = k[:, -w:, :]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale

        all_pos = torch.arange(seq_len).unsqueeze(0)
        obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
        causal = (all_pos <= obs_pos)
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        layer_importance = attn.sum(dim=1).mean(dim=0)  # (seq,)

        if pool_kernel > 1 and seq_len > pool_kernel:
            imp_1d = layer_importance.unsqueeze(0).unsqueeze(0)
            layer_importance = F.avg_pool1d(
                imp_1d, kernel_size=pool_kernel, padding=pool_kernel // 2, stride=1
            ).squeeze()[:seq_len]

        all_importance += layer_importance

    return all_importance / n_layers


def build_keep_mask(prefix_len, keep_pct, importance, sliding_window=32):
    """Importance-based keep mask: BOS + sliding window + top-importance tokens."""
    keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
    keep_mask[0] = True
    keep_mask[-sliding_window:] = True

    n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
    n_from_importance = n_to_keep - keep_mask.sum().item()
    if n_from_importance > 0:
        imp_copy = importance.clone()
        imp_copy[keep_mask] = -float('inf')
        _, top_idx = imp_copy.topk(min(n_from_importance, (~keep_mask).sum().item()))
        keep_mask[top_idx] = True

    return keep_mask


def build_random_keep_mask(prefix_len, keep_pct, sliding_window=32):
    """Random keep mask: BOS + sliding window + random other tokens."""
    keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
    keep_mask[0] = True
    keep_mask[-sliding_window:] = True

    n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
    n_from_random = n_to_keep - keep_mask.sum().item()
    if n_from_random > 0:
        candidates = (~keep_mask).nonzero().squeeze(-1)
        perm = torch.randperm(len(candidates))[:n_from_random]
        keep_mask[candidates[perm]] = True

    return keep_mask


def build_attn_mask(prefix_len, keep_mask, cont_len):
    """Attention mask: 1 for kept/continuation tokens, 0 for evicted."""
    attn_ctx = torch.ones(prefix_len, dtype=torch.long)
    attn_ctx[~keep_mask] = 0
    attn_full = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long)])
    return attn_full.unsqueeze(0)


def build_full_attn_mask(prefix_len, cont_len):
    """Full attention mask — attend to all tokens."""
    return torch.ones(1, prefix_len + cont_len, dtype=torch.long)


def zero_evicted_kv(kv_raw, keep_mask):
    """Clone KV, zero out evicted positions. Evicted tokens attend normally
    (no mask applied here) — the caller decides whether to mask."""
    kv = clone_kv(kv_raw)
    evict_mask = ~keep_mask
    for l in range(len(kv.layers)):
        kv.layers[l].keys[0, :, evict_mask, :] = 0.0
        kv.layers[l].values[0, :, evict_mask, :] = 0.0
    return kv


def e8_quantize_kv(kv_raw, keep_mask, bits, head_dim, rope_base=10000.0):
    """Clone KV, E8-quantize the kept tokens, zero the evicted tokens."""
    levels = 2 ** bits
    H = hadamard_matrix(head_dim)
    kv = clone_kv(kv_raw)

    for l in range(len(kv.layers)):
        k_full = kv.layers[l].keys[0].float()   # (n_heads, seq, dim)
        v_full = kv.layers[l].values[0].float()

        for is_key, t_full in [(True, k_full), (False, v_full)]:
            n_heads_local = t_full.shape[0]
            for h in range(n_heads_local):
                if is_key:
                    t_head = inverse_rope(t_full[h:h+1], base=rope_base)[0]  # (seq, dim)
                else:
                    t_head = t_full[h]

                kept_data = t_head[keep_mask]  # (n_kept, dim)
                rotated = kept_data @ H.T
                amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                sc = amax / (levels / 2)
                normalized = rotated / sc

                pad = (8 - head_dim % 8) % 8
                if pad > 0:
                    normalized = F.pad(normalized, (0, pad))
                groups = normalized.reshape(-1, 8)
                lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                coords = lp.reshape(-1, normalized.shape[-1])
                if pad > 0:
                    coords = coords[..., :head_dim]

                quantized = (coords * sc) @ H  # (n_kept, dim)

                result = torch.zeros_like(t_head)
                result[keep_mask] = quantized

                if is_key:
                    t_full[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                else:
                    t_full[h] = result

            if is_key:
                kv.layers[l].keys = t_full.unsqueeze(0).half()
            else:
                kv.layers[l].values = t_full.unsqueeze(0).half()

    return kv


# ---------------------------------------------------------------------------
# Hypothesis 1: Attention mask changes normalization
# ---------------------------------------------------------------------------

def hypothesis_1(model, full_ids, prefix_len, kv_raw, importance, n_layers,
                 head_dim, rope_base, baseline_ppl, keep_pct=30):
    """
    H1: The negative PPL comes from the attention MASK (softmax renormalization),
    not from zeroing evicted tokens.

    Variants for a fixed eviction rate (70% evicted = keep 30%):
      A) Zero evicted + mask them      (original pipeline, no quantization)
      B) Zero evicted, NO mask         (zeros in KV, softmax sees them)
      C) Keep original values, MASK evicted  (mask only, no zeroing)
      D) No modification               (sanity check = should match baseline_ppl)

    H1 prediction: C < 0  (mask alone improves PPL)
    H1 rejection: C ≈ 0, while A < 0  (need zeroing for improvement)
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Does the attention MASK drive the negative-PPL effect?")
    print("="*70)

    keep_mask = build_keep_mask(prefix_len, keep_pct, importance)
    cont_len = full_ids.shape[1] - prefix_len
    n_kept = keep_mask.sum().item()
    print(f"  keep_pct={keep_pct}%, kept={n_kept}, evicted={prefix_len-n_kept}", flush=True)

    # D: unmodified KV
    kv_d = clone_kv(kv_raw)
    ppl_d = compute_ppl(model, full_ids, prefix_len, kv_d)
    delta_d = (ppl_d - baseline_ppl) / baseline_ppl * 100
    print(f"  D) No eviction, no mask:         PPL={ppl_d:.4f} ({delta_d:+.3f}%) baseline", flush=True)

    # A: zero evicted + mask
    kv_a = zero_evicted_kv(kv_raw, keep_mask)
    attn_a = build_attn_mask(prefix_len, keep_mask, cont_len)
    ppl_a = compute_ppl(model, full_ids, prefix_len, kv_a, attn_mask=attn_a)
    delta_a = (ppl_a - baseline_ppl) / baseline_ppl * 100
    print(f"  A) Zero + mask:                  PPL={ppl_a:.4f} ({delta_a:+.3f}%)", flush=True)

    # B: zero evicted, NO mask (full attention)
    kv_b = zero_evicted_kv(kv_raw, keep_mask)
    attn_b = build_full_attn_mask(prefix_len, cont_len)
    ppl_b = compute_ppl(model, full_ids, prefix_len, kv_b, attn_mask=attn_b)
    delta_b = (ppl_b - baseline_ppl) / baseline_ppl * 100
    print(f"  B) Zero, no mask:                PPL={ppl_b:.4f} ({delta_b:+.3f}%)", flush=True)

    # C: original values, MASK only (no zeroing)
    kv_c = clone_kv(kv_raw)
    attn_c = build_attn_mask(prefix_len, keep_mask, cont_len)
    ppl_c = compute_ppl(model, full_ids, prefix_len, kv_c, attn_mask=attn_c)
    delta_c = (ppl_c - baseline_ppl) / baseline_ppl * 100
    print(f"  C) Mask only (no zeroing):       PPL={ppl_c:.4f} ({delta_c:+.3f}%)", flush=True)

    print("\n  INTERPRETATION:")
    if delta_c < -0.05:
        print(f"  -> H1 SUPPORTED: mask alone improves PPL by {delta_c:+.3f}%.")
        print(f"     Renormalization is the primary driver.")
        if abs(delta_c - delta_a) < 0.1:
            print(f"     Zeroing adds little on top of masking ({delta_a-delta_c:+.3f}% diff).")
    else:
        print(f"  -> H1 REJECTED: mask alone delta={delta_c:+.3f}% (not significant).")
        print(f"     Zeroing is required for improvement (A={delta_a:+.3f}%).")

    if delta_b > 0.1:
        print(f"  -> Zeroing WITHOUT mask hurts (+{delta_b:.3f}%): zeros corrupt dot-products.")
    elif delta_b < -0.05:
        print(f"  -> Zeroing without mask also helps ({delta_b:+.3f}%): zeros reduce attention mass on noise.")

    return {
        "D_unmodified": delta_d,
        "A_zero_plus_mask": delta_a,
        "B_zero_no_mask": delta_b,
        "C_mask_only": delta_c,
    }


# ---------------------------------------------------------------------------
# Hypothesis 2: Short sequences + small model
# ---------------------------------------------------------------------------

def hypothesis_2(model, tok, n_layers, head_dim, rope_base, keep_pct=30):
    """
    H2: Effect is sequence-length specific — short prefixes have many
    low-importance tokens; removing them genuinely helps focus.

    Test: vary prefix length (128, 256, 384, 512) measuring mask-only PPL delta.
    Uses mask-only (Variant C) to isolate the attention-focus effect cleanly.

    H2 prediction: effect weakens or reverses at longer prefix lengths.
    H2 rejection: effect holds at all lengths (structural renormalization).
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Does the effect scale with sequence length?")
    print("="*70)

    text = get_text()
    inputs = tok(text, return_tensors="pt", max_length=2048, truncation=True)
    all_ids = inputs.input_ids
    total_tokens = all_ids.shape[1]
    print(f"  Total tokens: {total_tokens}", flush=True)

    prefix_lengths = [128, 256, 384, 512]
    prefix_lengths = [p for p in prefix_lengths if p + 64 <= total_tokens]

    results = {}
    print(f"  Testing prefix lengths: {prefix_lengths} (keep_pct={keep_pct}%)\n", flush=True)
    print(f"  {'PrefixLen':>10s} {'BaselinePPL':>12s} {'MaskOnlyPPL':>12s} {'Delta%':>8s}  Verdict")
    print("  " + "-"*65)

    for prefix_len in prefix_lengths:
        cont_len = min(128, total_tokens - prefix_len)
        full_ids = all_ids[:, :prefix_len + cont_len]

        with torch.no_grad():
            out = model(full_ids[:, :prefix_len], use_cache=True)
            kv = out.past_key_values

        baseline_ppl = compute_ppl(model, full_ids, prefix_len, clone_kv(kv))
        importance = score_importance(kv, n_layers, obs_window=32, pool_kernel=5)
        keep_mask = build_keep_mask(prefix_len, keep_pct, importance)

        # Mask-only test (C variant from H1 — cleanest signal)
        kv_masked = clone_kv(kv)
        attn_mask = build_attn_mask(prefix_len, keep_mask, cont_len)
        masked_ppl = compute_ppl(model, full_ids, prefix_len, kv_masked, attn_mask=attn_mask)
        delta = (masked_ppl - baseline_ppl) / baseline_ppl * 100

        n_kept = keep_mask.sum().item()
        verdict = "IMPROVES" if delta < -0.05 else ("NEUTRAL" if abs(delta) < 0.3 else "DEGRADES")
        print(f"  {prefix_len:>10d} {baseline_ppl:>12.4f} {masked_ppl:>12.4f} {delta:>+8.3f}%  {verdict} (kept={n_kept})")
        results[f"prefix_{prefix_len}"] = {"baseline": baseline_ppl, "masked": masked_ppl, "delta": delta}

    deltas = [v["delta"] for v in results.values()]
    improving = sum(1 for d in deltas if d < -0.05)
    print("\n  INTERPRETATION:")
    if improving == len(deltas):
        print(f"  -> H2 SUPPORTED trivially but DOES NOT narrow it down:")
        print(f"     Effect persists across ALL lengths — it is STRUCTURAL, not length-specific.")
        print(f"     H1 (renormalization) is the better explanation.")
    elif improving >= max(1, len(deltas) - 1) and deltas[-1] > deltas[0]:
        print(f"  -> H2 PARTIALLY SUPPORTED: effect attenuates at longer sequences.")
        print(f"     Shorter sequences benefit more — fewer tokens needed, signal concentrates.")
    else:
        print(f"  -> H2 REJECTED: no clear length dependence in the effect.")

    return results


# ---------------------------------------------------------------------------
# Hypothesis 3: Quantization noise cancellation
# ---------------------------------------------------------------------------

def hypothesis_3(model, full_ids, prefix_len, kv_raw, importance, n_layers,
                 head_dim, rope_base, baseline_ppl, keep_pct=30, bits=2):
    """
    H3: Fewer quantized tokens = less total quantization noise = lower PPL.

    Variants:
      A) Evict + quantize remaining  (original pipeline = observed -PPL)
      B) Evict only, NO quantization (zero+mask, keep values in fp16)
      C) Quantize ALL tokens, no eviction  (maximum noise baseline)
      D) Baseline (no eviction, no quantization)

    H3 strict prediction: (A) worse than (B), since (A) = (B) + quant noise on kept tokens.
    H3 noise-cancellation variant: (A) better than (C) because fewer tokens quantized.
    Critical test: does (B) alone improve PPL? If yes, eviction drives the effect regardless.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Is quantization noise cancellation the driver?")
    print("="*70)

    keep_mask = build_keep_mask(prefix_len, keep_pct, importance)
    cont_len = full_ids.shape[1] - prefix_len
    n_kept = keep_mask.sum().item()
    print(f"  keep_pct={keep_pct}%, bits={bits}, kept={n_kept}, evicted={prefix_len-n_kept}", flush=True)

    # C: quantize ALL tokens, no eviction
    keep_all = torch.ones(prefix_len, dtype=torch.bool)
    kv_c = e8_quantize_kv(kv_raw, keep_all, bits, head_dim, rope_base)
    ppl_c = compute_ppl(model, full_ids, prefix_len, kv_c)
    delta_c = (ppl_c - baseline_ppl) / baseline_ppl * 100
    print(f"  C) Quant all, no eviction:        PPL={ppl_c:.4f} ({delta_c:+.3f}%) ← pure quant noise", flush=True)

    # B: evict only, NO quantization (zero + mask, fp16 values)
    kv_b = zero_evicted_kv(kv_raw, keep_mask)
    attn_b = build_attn_mask(prefix_len, keep_mask, cont_len)
    ppl_b = compute_ppl(model, full_ids, prefix_len, kv_b, attn_mask=attn_b)
    delta_b = (ppl_b - baseline_ppl) / baseline_ppl * 100
    print(f"  B) Evict only (no quantization):  PPL={ppl_b:.4f} ({delta_b:+.3f}%) ← pure eviction", flush=True)

    # A: evict + quantize (original pipeline)
    kv_a = e8_quantize_kv(kv_raw, keep_mask, bits, head_dim, rope_base)
    attn_a = build_attn_mask(prefix_len, keep_mask, cont_len)
    ppl_a = compute_ppl(model, full_ids, prefix_len, kv_a, attn_mask=attn_a)
    delta_a = (ppl_a - baseline_ppl) / baseline_ppl * 100
    print(f"  A) Evict + quantize:              PPL={ppl_a:.4f} ({delta_a:+.3f}%) ← observed pipeline", flush=True)

    # E: evict+quantize, NO mask (zero evicted, quant kept, full attention)
    kv_e = e8_quantize_kv(kv_raw, keep_mask, bits, head_dim, rope_base)
    attn_e = build_full_attn_mask(prefix_len, cont_len)
    ppl_e = compute_ppl(model, full_ids, prefix_len, kv_e, attn_mask=attn_e)
    delta_e = (ppl_e - baseline_ppl) / baseline_ppl * 100
    print(f"  E) Evict+quant, NO mask:          PPL={ppl_e:.4f} ({delta_e:+.3f}%) ← quant fewer, attend all", flush=True)

    print("\n  DECOMPOSITION:")
    print(f"    Pure quant noise (all tokens):     {delta_c:+.3f}%")
    print(f"    Pure eviction (no quant):           {delta_b:+.3f}%")
    print(f"    Evict + quantize (observed):        {delta_a:+.3f}%")
    print(f"    Quant noise on kept only (A - B):   {delta_a - delta_b:+.3f}%")

    print("\n  INTERPRETATION:")
    if delta_b < -0.05:
        print(f"  -> Eviction alone ({delta_b:+.3f}%) ALSO improves PPL.")
        print(f"     The improvement is NOT unique to quantization noise reduction.")
        print(f"     H1 (attention renormalization) is necessary to explain both A and B.")
        if delta_a < delta_b:
            print(f"     Adding quantization hurts vs pure eviction: A({delta_a:+.3f}%) < B({delta_b:+.3f}%).")
            print(f"     H3 confirmed: fewer quantized tokens = better. But H1 is the base effect.")
        else:
            print(f"     Adding quantization has negligible additional cost: A≈B.")
    else:
        print(f"  -> Eviction alone does NOT improve PPL ({delta_b:+.3f}%).")
        if delta_a < -0.05:
            print(f"     But eviction+quant does ({delta_a:+.3f}%). H3 is the primary mechanism:")
            print(f"     Quantized zeros cancel noise — the specific pattern of zero+quant matters.")
        else:
            print(f"     Neither eviction alone nor eviction+quant improve PPL significantly.")

    return {
        "C_quant_all": delta_c,
        "B_evict_no_quant": delta_b,
        "A_evict_plus_quant": delta_a,
        "E_evict_quant_no_mask": delta_e,
    }


# ---------------------------------------------------------------------------
# Bonus: Random eviction vs importance-based eviction
# ---------------------------------------------------------------------------

def random_vs_importance_eviction(model, full_ids, prefix_len, kv_raw, importance,
                                   baseline_ppl, keep_pct=30):
    """
    Critical structural test: if even RANDOM eviction improves PPL, then the
    improvement is NOT from selecting the right tokens — it is from the
    structural effect of having fewer attention slots (H1/H3), independent of
    which specific tokens are removed.

    Uses mask-only (Variant C) for cleanest signal.
    """
    print("\n" + "="*70)
    print("BONUS: Importance-based vs random eviction")
    print("="*70)

    cont_len = full_ids.shape[1] - prefix_len

    # Importance-based (mask only)
    keep_imp = build_keep_mask(prefix_len, keep_pct, importance)
    kv_imp = clone_kv(kv_raw)
    attn_imp = build_attn_mask(prefix_len, keep_imp, cont_len)
    ppl_imp = compute_ppl(model, full_ids, prefix_len, kv_imp, attn_mask=attn_imp)
    delta_imp = (ppl_imp - baseline_ppl) / baseline_ppl * 100
    print(f"  Importance-based (mask only): PPL={ppl_imp:.4f} ({delta_imp:+.3f}%)", flush=True)

    # Random eviction (3 seeds)
    random_deltas = []
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        keep_rand = build_random_keep_mask(prefix_len, keep_pct)
        kv_rand = clone_kv(kv_raw)
        attn_rand = build_attn_mask(prefix_len, keep_rand, cont_len)
        ppl_rand = compute_ppl(model, full_ids, prefix_len, kv_rand, attn_mask=attn_rand)
        delta_rand = (ppl_rand - baseline_ppl) / baseline_ppl * 100
        random_deltas.append(delta_rand)
        print(f"  Random (seed={seed}):           PPL={ppl_rand:.4f} ({delta_rand:+.3f}%)", flush=True)

    avg_rand = sum(random_deltas) / len(random_deltas)
    print(f"\n  Importance-based: {delta_imp:+.3f}%  |  Random avg: {avg_rand:+.3f}%")
    print(f"  Improvement from smart selection vs random: {delta_imp - avg_rand:+.3f}%")

    print("\n  INTERPRETATION:")
    if avg_rand < -0.05:
        print("  -> CRITICAL: Random eviction ALSO improves PPL!")
        print("     The effect is STRUCTURAL — not about smart token selection.")
        print("     This strongly supports H1: renormalization over fewer tokens is the driver.")
        print("     The improvement-by-eviction claim cannot be attributed to token quality.")
    elif delta_imp < avg_rand - 0.2:
        print(f"  -> Smart selection provides {delta_imp-avg_rand:+.3f}% advantage over random.")
        print("     Token quality selection contributes, but test if structural effect is still present.")
    else:
        print("  -> Importance ≈ random. Smart selection provides no measurable advantage.")

    return {
        "importance_delta": delta_imp,
        "random_avg_delta": avg_rand,
        "random_deltas": random_deltas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model, tok = load_model()
    text = get_text()
    inputs = tok(text, return_tensors="pt", max_length=1024, truncation=True)
    full_ids = inputs.input_ids
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2

    n_layers = model.config.num_hidden_layers
    head_dim = 64
    rope_base = getattr(model.config, 'rope_theta', 10000.0)

    print(f"Model: TinyLlama 1.1B | Layers: {n_layers} | Tokens: {n_tok} | Prefix: {prefix_len}")
    print(f"Head dim: {head_dim} | RoPE base: {rope_base}\n")

    # Compute prefix KV once and reuse (all downstream functions clone it)
    print("Computing prefix KV...", flush=True)
    with torch.no_grad():
        out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_raw = out.past_key_values

    # Score importance (must use kv_raw.layers which has correct seq_len=prefix_len)
    print("Scoring token importance...", flush=True)
    importance = score_importance(kv_raw, n_layers, obs_window=32, pool_kernel=5)
    print(f"Importance: min={importance.min():.4f} max={importance.max():.4f} "
          f"mean={importance.mean():.4f}\n")

    # Baseline PPL using a clone (so kv_raw stays pristine)
    baseline_ppl = compute_ppl(model, full_ids, prefix_len, clone_kv(kv_raw))
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    t0 = time.time()

    r1 = hypothesis_1(model, full_ids, prefix_len, kv_raw, importance, n_layers,
                      head_dim, rope_base, baseline_ppl, keep_pct=30)

    r2 = hypothesis_2(model, tok, n_layers, head_dim, rope_base, keep_pct=30)

    r3 = hypothesis_3(model, full_ids, prefix_len, kv_raw, importance, n_layers,
                      head_dim, rope_base, baseline_ppl, keep_pct=30, bits=2)

    r_rand = random_vs_importance_eviction(model, full_ids, prefix_len, kv_raw,
                                            importance, baseline_ppl, keep_pct=30)

    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    h1_mask_only = r1["C_mask_only"]
    h1_zero_mask = r1["A_zero_plus_mask"]
    h1_zero_nomask = r1["B_zero_no_mask"]
    h3_evict_only = r3["B_evict_no_quant"]
    h3_evict_quant = r3["A_evict_plus_quant"]
    rand_avg = r_rand["random_avg_delta"]

    print(f"\n  Key measurements (all vs baseline PPL={baseline_ppl:.4f}):")
    print(f"    Mask only (H1 test):              {h1_mask_only:+.3f}%")
    print(f"    Zero + mask (no quant):           {h1_zero_mask:+.3f}%")
    print(f"    Zero, no mask (corruption test):  {h1_zero_nomask:+.3f}%")
    print(f"    Evict only, no quant (H3 test):   {h3_evict_only:+.3f}%")
    print(f"    Evict + 2b-E8 (observed):         {h3_evict_quant:+.3f}%")
    print(f"    Random eviction (avg, mask-only):  {rand_avg:+.3f}%")

    verdicts = []

    if h1_mask_only < -0.05:
        verdicts.append("H1 CONFIRMED: Attention mask renormalization is sufficient to explain the improvement.")
    else:
        verdicts.append("H1 REJECTED: Mask alone is insufficient.")

    if h3_evict_only < -0.05:
        verdicts.append("H3 partially: Eviction alone (no quant) also improves PPL.")
    if rand_avg < -0.05:
        verdicts.append("STRUCTURAL: Random eviction ALSO improves — not from smart token selection.")
        verdicts.append("=> Mechanism is renormalization/sparsification, NOT 'removing bad tokens'.")

    if h1_mask_only < -0.05 and rand_avg < -0.05:
        verdicts.append("PRIMARY CONCLUSION: The negative PPL is an evaluation artifact.")
        verdicts.append("  The attention mask makes the continuation score EASIER by forcing focus,")
        verdicts.append("  not because the compression is higher quality.")
    elif h3_evict_only >= -0.05 and h1_mask_only >= -0.05 and h3_evict_quant < -0.05:
        verdicts.append("H3 CONFIRMED (noise cancellation): Quantization + eviction interact specifically.")
        verdicts.append("  The improvement requires BOTH eviction AND quantization.")

    print("\n  CONCLUSIONS:")
    for v in verdicts:
        print(f"    * {v}")

    print(f"\n  HONEST IMPLICATIONS FOR PAPER CLAIMS:")
    print(f"    - A negative PPL delta cannot be treated as compression quality improvement.")
    print(f"    - PPL improvement via masking is an artifact of attention redistribution,")
    print(f"      not a property that persists during autoregressive generation (no prefix KV reuse).")
    print(f"    - Must validate on: (1) generation-mode (token-by-token, no pre-scored KV),")
    print(f"      (2) longer contexts 2K+, (3) Mistral-7B where this effect may not hold.")
    print(f"    - The correct framing: 'N% eviction achieves Xx compression at +Y% PPL'")
    print(f"      where Y is measured WITHOUT masking the evicted positions.")

    print(f"\n  Total experiment time: {elapsed:.1f}s")

    # Write report
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".planning", "research")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "NEGATIVE_PPL_INVESTIGATION.md")
    with open(out_path, "w") as f:
        f.write("# Negative PPL Investigation — Why Does Eviction IMPROVE Perplexity?\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model:** TinyLlama 1.1B | Tokens: {n_tok} | Prefix: {prefix_len}\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n\n")
        f.write("## H1: Attention Mask Renormalization\n\n")
        f.write(f"| Variant | Delta% | Meaning |\n|---------|--------|--------|\n")
        for k, v in r1.items():
            f.write(f"| {k} | {v:+.3f}% | |\n")
        f.write("\n## H2: Sequence Length Scaling\n\n")
        f.write(f"| Prefix | Baseline PPL | Masked PPL | Delta% |\n|--------|-------------|------------|-------|\n")
        for k, v in r2.items():
            f.write(f"| {k} | {v['baseline']:.4f} | {v['masked']:.4f} | {v['delta']:+.3f}% |\n")
        f.write("\n## H3: Quantization Noise Cancellation\n\n")
        f.write(f"| Variant | Delta% | Meaning |\n|---------|--------|--------|\n")
        for k, v in r3.items():
            f.write(f"| {k} | {v:+.3f}% | |\n")
        f.write("\n## Random vs Importance Eviction\n\n")
        f.write(f"- Importance-based: {r_rand['importance_delta']:+.3f}%\n")
        f.write(f"- Random avg: {r_rand['random_avg_delta']:+.3f}%\n\n")
        f.write("## Conclusions\n\n")
        for v in verdicts:
            f.write(f"- {v}\n")
        f.write("\n## Honest Implications\n\n")
        f.write("- Negative PPL from eviction is likely an evaluation artifact (attention renormalization).\n")
        f.write("- The effect may not persist in generation mode or longer contexts.\n")
        f.write("- Paper must use unmasked PPL measurement as the honest compression quality metric.\n")
    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()

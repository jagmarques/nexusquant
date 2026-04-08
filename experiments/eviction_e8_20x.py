"""Attention-aware token eviction + E8 quantization — path to 20x.

Strategy: SnapKV-style eviction stacked with 2-bit E8 quantization.
- E8 2-bit gives ~6x alone
- 70% eviction (keep 30%) gives ~3.3x
- Combined: ~20x

Token eviction is fundamentally different from token merging:
- Merging AVERAGES similar tokens → destroys individual info → catastrophic
- Eviction REMOVES low-importance tokens → keeps important ones intact

Importance scoring: use last W keys as pseudo-queries against all keys.
Key insight: keys already have RoPE, so K_obs @ K_all^T is a valid attention proxy.
"""
import sys, os, copy, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nexusquant-oss"))
from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.rope_utils import inverse_rope, forward_rope
import zstandard


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {m}...", flush=True)
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
    model.eval()
    return model, tok


def get_long_text():
    """~1000 tokens of diverse text."""
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
        "46 percent of Earth's water surface. The Mariana Trench is the "
        "deepest point at 10994 meters. The Ring of Fire follows the Pacific "
        "plate edges with 75 percent of active volcanoes. Ocean currents "
        "transport heat and nutrients across vast distances. Number theory "
        "studies integers and their properties. Gauss called it the queen "
        "of mathematics. The Riemann hypothesis remains unsolved. Prime "
        "numbers are central. The prime number theorem gives asymptotic "
        "density. Fermat's last theorem was proved by Andrew Wiles in 1995."
    )


def score_token_importance(kv, n_layers, n_heads, obs_window=32, pool_kernel=5):
    """Score each token's importance using key-key attention (SnapKV-style).

    Uses last obs_window keys as pseudo-queries. Average importance across all layers and heads.
    """
    seq_len = kv.layers[0].keys.shape[2]
    head_dim = kv.layers[0].keys.shape[3]
    w = min(obs_window, seq_len)

    all_importance = torch.zeros(seq_len)

    for l in range(n_layers):
        k = kv.layers[l].keys[0].float()  # (n_heads, seq, dim)
        k_obs = k[:, -w:, :]  # (n_heads, w, dim)

        scale = 1.0 / math.sqrt(head_dim)
        # (n_heads, w, seq)
        scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale

        # Causal mask: obs token at position (seq-w+i) can attend to positions <= (seq-w+i)
        all_pos = torch.arange(seq_len).unsqueeze(0)  # (1, seq)
        obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)  # (w, 1)
        causal = (all_pos <= obs_pos)  # (w, seq)
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        # Sum over observation queries, mean over heads
        layer_importance = attn.sum(dim=1).mean(dim=0)  # (seq,)

        # Smooth with average pooling
        if pool_kernel > 1 and seq_len > pool_kernel:
            imp_1d = layer_importance.unsqueeze(0).unsqueeze(0)
            layer_importance = F.avg_pool1d(
                imp_1d, kernel_size=pool_kernel, padding=pool_kernel // 2, stride=1
            ).squeeze()[:seq_len]

        all_importance += layer_importance

    return all_importance / n_layers


def evict_and_quantize(kv, keep_mask, bits, head_dim, rope_base=10000.0):
    """Evict tokens not in keep_mask, E8 quantize the rest.

    Instead of physically removing tokens (which would break position IDs),
    we zero out evicted tokens and use attention mask during inference.
    We quantize only the kept tokens.
    """
    n_layers = len(kv.layers)
    levels = 2 ** bits
    H = hadamard_matrix(head_dim)

    all_coords = []
    total_fp16 = 0
    total_scale_bytes = 0
    n_kept = keep_mask.sum().item()
    seq_len = kv.layers[0].keys.shape[2]

    for l in range(n_layers):
        layer = kv.layers[l]
        k = layer.keys.float()  # (1, n_heads, seq, dim)
        v = layer.values.float()
        total_fp16 += k.numel() * 2 + v.numel() * 2

        for is_key, tensor in [(True, k), (False, v)]:
            t = tensor[0].clone()  # (n_heads, seq, dim)
            n_heads_local = t.shape[0]

            # Quantize kept tokens
            for h in range(n_heads_local):
                if is_key:
                    # RoPE removal for keys
                    t_head = inverse_rope(t[h:h+1], base=rope_base)[0]  # (seq, dim)
                else:
                    t_head = t[h]  # (seq, dim)

                # Only quantize kept tokens
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

                quantized = (coords * sc) @ H

                # Extract coords for compression
                int_coords = coords.detach().numpy()
                has_half = np.any(np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25)
                if has_half:
                    all_coords.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                else:
                    all_coords.append(np.round(int_coords.flatten()).astype(np.int8))
                total_scale_bytes += n_kept * 2  # fp16 per kept vector

                # Write back
                if is_key:
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                else:
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    t[h] = result

            if is_key:
                layer.keys = t.unsqueeze(0).half()
            else:
                layer.values = t.unsqueeze(0).half()

    # Compress indices per-layer with temporal delta + zstd
    cctx = zstandard.ZstdCompressor(level=22)
    total_idx = 0
    for coords in all_coords:
        arr = coords.ravel()
        n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
        if n_per_tok > 0 and len(arr) % n_kept == 0:
            reshaped = arr.reshape(n_kept, n_per_tok)
            delta = np.zeros_like(reshaped)
            delta[0] = reshaped[0]
            delta[1:] = reshaped[1:] - reshaped[:-1]
            total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
        else:
            total_idx += len(cctx.compress(arr.tobytes()))

    # Token mask overhead: 1 bit per token per layer = negligible
    mask_bytes = math.ceil(seq_len / 8) * n_layers * 2  # 2 for K+V (though same mask)

    total = total_idx + total_scale_bytes + mask_bytes
    ratio = total_fp16 / total if total > 0 else 0

    return {
        "fp16": total_fp16,
        "idx_bytes": total_idx,
        "scale_bytes": total_scale_bytes,
        "mask_bytes": mask_bytes,
        "total": total,
        "ratio": ratio,
        "n_kept": n_kept,
        "n_total": seq_len,
        "keep_pct": n_kept / seq_len * 100,
    }


def main():
    model, tok = load_model()
    text = get_long_text()
    inputs = tok(text, return_tensors="pt", max_length=1024, truncation=True)
    full_ids = inputs.input_ids
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    print(f"Tokens: {n_tok}, prefix: {prefix_len}\n", flush=True)

    n_layers = model.config.num_hidden_layers
    n_heads = 4  # KV heads for TinyLlama
    head_dim = 64
    rope_base = getattr(model.config, 'rope_theta', 10000.0)

    # Get baseline PPL
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_baseline = prefix_out.past_key_values
        cont = model(full_ids[:, prefix_len:], past_key_values=kv_baseline, use_cache=True)
        logits = cont.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}\n", flush=True)

    # Score token importance (using baseline KV)
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_score = prefix_out.past_key_values
    importance = score_token_importance(kv_score, n_layers, n_heads, obs_window=32)
    print(f"Importance scores: min={importance.min():.4f} max={importance.max():.4f} "
          f"mean={importance.mean():.4f}", flush=True)

    # Test configs: eviction rate + quantization bits
    configs = [
        {"name": "2b E8 only (0% evict)", "bits": 2, "keep_pct": 100},
        {"name": "3b E8 only (0% evict)", "bits": 3, "keep_pct": 100},
        {"name": "2b E8 + 10% evict", "bits": 2, "keep_pct": 90},
        {"name": "2b E8 + 20% evict", "bits": 2, "keep_pct": 80},
        {"name": "2b E8 + 30% evict", "bits": 2, "keep_pct": 70},
        {"name": "2b E8 + 40% evict", "bits": 2, "keep_pct": 60},
        {"name": "2b E8 + 50% evict", "bits": 2, "keep_pct": 50},
        {"name": "2b E8 + 60% evict", "bits": 2, "keep_pct": 40},
        {"name": "2b E8 + 70% evict", "bits": 2, "keep_pct": 30},
        {"name": "2b E8 + 75% evict", "bits": 2, "keep_pct": 25},
        {"name": "2b E8 + 80% evict", "bits": 2, "keep_pct": 20},
        {"name": "3b E8 + 50% evict", "bits": 3, "keep_pct": 50},
        {"name": "3b E8 + 70% evict", "bits": 3, "keep_pct": 30},
    ]

    # Always keep last 32 tokens (sliding window) + first token (BOS)
    sliding_window = min(32, prefix_len)

    print(f"\n{'Config':<28s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} "
          f"{'Kept':>5s} {'Idx':>8s} {'Scale':>8s} {'Total':>10s}", flush=True)
    print("-" * 95, flush=True)

    results = []

    for cfg in configs:
        t0 = time.time()
        bits = cfg["bits"]
        keep_pct = cfg["keep_pct"]

        # Fresh KV
        with torch.no_grad():
            prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
            kv = prefix_out.past_key_values

        if keep_pct >= 100:
            # No eviction, just quantize all
            keep_mask = torch.ones(prefix_len, dtype=torch.bool)
        else:
            # Build keep mask: sliding window + top importance tokens
            keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
            keep_mask[0] = True  # BOS
            keep_mask[-sliding_window:] = True  # Sliding window

            n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
            n_from_importance = n_to_keep - keep_mask.sum().item()

            if n_from_importance > 0:
                # Score non-window, non-BOS tokens
                imp_copy = importance.clone()
                imp_copy[keep_mask] = -float('inf')
                _, top_idx = imp_copy.topk(min(n_from_importance, (~keep_mask).sum().item()))
                keep_mask[top_idx] = True

        # Evict and quantize
        info = evict_and_quantize(kv, keep_mask, bits, head_dim, rope_base)

        # Compute PPL with attention mask
        evict_mask = ~keep_mask
        attn_ctx = torch.ones(prefix_len, dtype=torch.long)
        attn_ctx[evict_mask] = 0
        cont_len = full_ids.shape[1] - prefix_len
        attn_full = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long)])
        attn_mask = attn_full.unsqueeze(0)

        with torch.no_grad():
            cont = model(full_ids[:, prefix_len:], past_key_values=kv,
                        attention_mask=attn_mask, use_cache=True)
            logits = cont.logits[:, :-1, :]
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppl = torch.exp(loss).item()

        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        elapsed = time.time() - t0

        print(f"{cfg['name']:<28s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x "
              f"{info['keep_pct']:4.0f}% {info['idx_bytes']:>8,} {info['scale_bytes']:>8,} "
              f"{info['total']:>10,}  ({elapsed:.1f}s)", flush=True)

        results.append({
            "config": cfg["name"], "ppl": ppl, "delta": delta,
            "bits": bits, "keep_pct": keep_pct, **info
        })

    # Summary
    print(f"\n{'='*95}")
    print("SUMMARY — Eviction + E8 Path to 20x")
    print(f"{'='*95}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Prefix tokens: {prefix_len}, Sliding window: {sliding_window}\n")

    for r in results:
        status = ""
        if r["ratio"] >= 20:
            status = " *** 20x ACHIEVED ***"
        elif r["ratio"] >= 15:
            status = " (CLOSE)"
        quality = "OK" if abs(r["delta"]) < 5 else ("WARN" if abs(r["delta"]) < 10 else "BAD")
        print(f"  {r['config']:<28s} {r['ratio']:6.1f}x  {r['delta']:+6.2f}%  Q:{quality}{status}")

    # Find best config at 20x+
    candidates_20x = [r for r in results if r["ratio"] >= 20 and abs(r["delta"]) < 10]
    if candidates_20x:
        best = min(candidates_20x, key=lambda r: abs(r["delta"]))
        print(f"\n  BEST 20x+ CONFIG: {best['config']} → {best['ratio']:.1f}x at {best['delta']:+.2f}% PPL")
    else:
        best_ratio = max(results, key=lambda r: r["ratio"] if abs(r["delta"]) < 10 else 0)
        print(f"\n  20x NOT REACHED with <10% PPL. Best usable: {best_ratio['config']} "
              f"→ {best_ratio['ratio']:.1f}x at {best_ratio['delta']:+.2f}%")

    # Write results
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            ".company", "engineering", "eviction_e8_results.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Attention-Aware Eviction + E8 — Path to 20x\n\n")
        f.write(f"**Model:** TinyLlama 1.1B (22 layers, 4 KV heads, head_dim=64)\n")
        f.write(f"**Tokens:** {n_tok} (prefix={prefix_len})\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n")
        f.write(f"**Sliding window:** {sliding_window} tokens (always kept)\n")
        f.write(f"**Importance:** Key-key attention, obs_window=32, pool_kernel=5\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")
        f.write(f"| Config | PPL | Delta% | Ratio | Kept% | Idx | Scale | Total |\n")
        f.write(f"|--------|-----|--------|-------|-------|-----|-------|-------|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['ppl']:.4f} | {r['delta']:+.2f}% | "
                    f"{r['ratio']:.2f}x | {r['keep_pct']:.0f}% | {r['idx_bytes']:,} | "
                    f"{r['scale_bytes']:,} | {r['total']:,} |\n")
        f.write("\n## Honest Notes\n\n")
        f.write("- Evicted tokens zeroed out, attention mask used during continuation\n")
        f.write("- Importance scored on full uncompressed KV (optimistic — real system scores incrementally)\n")
        f.write("- Per-head per-layer compression with temporal delta + zstd level 22\n")
        f.write("- All overhead: compressed indices + fp16 scales + token mask bits\n")
        f.write("- RoPE removed from keys before E8, reapplied after\n")
        f.write(f"- TinyLlama has only 4 KV heads — larger models may tolerate more eviction\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

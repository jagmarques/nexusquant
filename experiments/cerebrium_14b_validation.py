"""Qwen2.5-14B validation on Cerebrium A10 (24GB).

Deploy instructions:
  1. Copy this file to cerebrium-exp/nexusquant-gpu/main_14b.py
  2. Add nexusquant core files to the same directory (or install as package)
  3. Deploy: cerebrium deploy nexusquant-gpu
  4. Call: POST https://<endpoint>/run_14b  {"prefix_len": 2000}

Qwen2.5-14B in FP16 = ~28GB — tight on A10 24GB with KV overhead.
We use device_map="auto" so the runtime will offload to CPU if needed.
Boundary protection is mandatory for Qwen architecture (boundary=2).

Test matrix:
  - Baseline:            no KV compression
  - K2V2+boundary 35%:   2-bit K+V, 35% eviction, protect 2 boundary layers
  - K2V2+boundary 60%:   2-bit K+V, 60% eviction, protect 2 boundary layers
  - K3V2+boundary 35%:   3-bit K, 2-bit V, 35% eviction, protect 2 boundary layers
  - K3V2+boundary 60%:   3-bit K, 2-bit V, 60% eviction, protect 2 boundary layers

Prefix: ~2000 tokens. Continuation: 200 tokens for PPL.

Overhead accounting (BRUTAL HONESTY):
  - Compressed layers: temporal delta-zstd E8 index bytes (analytic, zstd level 22)
  - Scale bytes: 1 fp16 per kept-token per KV-head per layer (K and V separately)
  - Protected layers: full FP16 for all prefix tokens, no eviction
  - Eviction mask: ceil(prefix_len/8) per compressed layer
"""
import os
import sys
import gc
import json
import time
import math


# Cerebrium expects a top-level callable. Wrap everything in run_14b().
def run_14b(prefix_len: int = 2000, cont_len: int = 200, **kwargs):
    """Entry point for Cerebrium deployment.

    Args:
        prefix_len: Number of prefix tokens to compress (default 2000).
        cont_len:   Number of continuation tokens for PPL measurement (default 200).

    Returns:
        dict with keys: model, baseline_ppl, results (list of per-config dicts).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # NexusQuant core — expected at /root/nexusquant or alongside this file
    _nq_paths = [
        "/root/nexusquant",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexusquant"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nexusquant-oss", "nexusquant"),
    ]
    for _p in _nq_paths:
        parent = os.path.dirname(_p)
        if parent not in sys.path and os.path.isdir(_p):
            sys.path.insert(0, parent)
            break

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    print("=" * 80)
    print("NEXUSQUANT — Qwen2.5-14B Validation (Cerebrium A10 24GB)")
    print(f"prefix_len={prefix_len}  cont_len={cont_len}")
    print("=" * 80)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    hf_token = os.environ.get("HF_TOKEN", "")
    model_name = "Qwen/Qwen2.5-14B"

    print(f"\nLoading {model_name} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim   = model.config.hidden_size // model.config.num_attention_heads
    rope_base  = getattr(model.config, "rope_theta", 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads} KV-heads, head_dim={head_dim}, rope_base={rope_base}")

    # ---- Corpus (same multi-topic structure as Mistral-7B experiments) ----
    CORPUS = " ".join([
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed through the work of many scientists throughout the latter half of the 20th century, with the current formulation finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by fundamental forces mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. Despite its success, the Standard Model does not incorporate gravity, dark matter, or dark energy.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation beginning in Britain and spreading to Western Europe and North America. The transition from hand production to machine manufacturing, new chemical processes, the rise of steam power, and the growth of the factory system fundamentally changed the nature of work and society. This period saw the emergence of the middle class, rapid urbanization, and the beginning of modern capitalism.",
        "The theory of evolution by natural selection, formulated by Darwin and Wallace, is the cornerstone of modern biology. Organisms with heritable traits better suited to their environment tend to survive and reproduce more. Over time these advantageous traits spread through the population. Genetic variation arises through mutation, recombination, and gene flow. Evolutionary theory is supported by the fossil record, comparative anatomy, molecular biology, and direct observation of evolution in real time.",
        "Quantum mechanics fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain black-body radiation, the photoelectric effect, or atomic stability. Max Planck introduced energy quanta in 1900. Einstein explained the photoelectric effect with photons in 1905. Bohr developed his hydrogen atom model in 1913. Heisenberg, Schrodinger, Dirac, and others built the mathematical framework of quantum mechanics in the 1920s.",
        "Artificial intelligence is the simulation of human intelligence by computer systems. Machine learning uses statistical techniques to enable computers to learn from data without explicit programming. Deep learning employs multi-layer neural networks to learn hierarchical representations. Transformer architectures, introduced by Vaswani et al. in 2017, revolutionized natural language processing and have since been applied to vision, audio, protein structure prediction, and many other domains.",
        "Climate change represents one of the defining challenges of the 21st century. Burning fossil fuels and deforestation have elevated greenhouse gas concentrations to levels not seen in at least 800,000 years. Global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels. Consequences include more frequent extreme weather events, rising sea levels, ocean acidification, and ecosystem disruption.",
        "The AdS/CFT correspondence, proposed by Maldacena in 1997, conjectures an exact duality between type IIB superstring theory on five-dimensional anti-de Sitter space and four-dimensional N=4 super Yang-Mills theory on the conformal boundary. This holographic principle has profound implications for quantum gravity and strongly coupled quantum field theories and has been applied to compute quark-gluon plasma properties.",
        "CRISPR-Cas9 gene editing exploits the bacterial adaptive immune system. The Cas9 endonuclease, guided by a single-guide RNA, introduces double-strand breaks at specific genomic loci, enabling precise modifications through nonhomologous end joining or homology-directed repair. Base editing and prime editing extend CRISPR capabilities to single-nucleotide changes without double-strand breaks.",
        "Topological data analysis uses persistent homology to track the birth and death of topological features across a filtration of simplicial complexes, producing stable persistence diagrams and barcodes. Applications include protein folding, brain connectivity, materials microstructure, and manifold learning. Stability theorems guarantee robustness under perturbation of the input data.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. Headed by emperors and spanning territories around the Mediterranean, the city of Rome was the world's largest city from about 100 BC to 400 AD. Roman engineering achievements included aqueducts, roads, and the Pantheon. Roman law formed the basis of many modern legal systems. Latin evolved into the Romance languages.",
    ])

    sliding_window = 32

    # ---- KV cache accessors ----
    def get_kv(cache, layer):
        if hasattr(cache, "key_cache"):
            return cache.key_cache[layer], cache.value_cache[layer]
        return cache.layers[layer].keys, cache.layers[layer].values

    def set_kv(cache, layer, k, v):
        if hasattr(cache, "key_cache"):
            cache.key_cache[layer] = k
            cache.value_cache[layer] = v
        else:
            cache.layers[layer].keys = k
            cache.layers[layer].values = v

    # ---- Importance scorer ----
    def score_importance(kv_cache, obs_window=32, pool_kernel=5):
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device="cpu")
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale_f = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale_f
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = all_pos <= obs_pos
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(
                    imp_1d, kernel_size=pool_kernel,
                    padding=pool_kernel // 2, stride=1
                ).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ---- Keep mask ----
    def build_keep_mask(seq_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(seq_len, dtype=torch.bool)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(seq_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float("inf")
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ---- Core evict + E8 quantize ----
    def evict_quantize(kv_cache, keep_mask, plen,
                       key_bits=2, value_bits=2, protect_boundary=2):
        """Boundary protection is mandatory for Qwen (always >= 2).

        Overhead accounting:
          - Compressed layers: temporal delta-zstd bytes (E8 indices)
          - Scale bytes: n_kept * n_kv_heads * 2 (K+V) * 2 (fp16) * n_compressed_layers
          - Protected layers: full FP16, no eviction
          - Mask bytes: ceil(plen/8) * n_compressed_layers
        """
        protect_boundary = max(protect_boundary, 2)  # Qwen: always protect >= 2 layers

        H = hadamard_matrix(head_dim).cpu()
        cctx = zstandard.ZstdCompressor(level=22)

        n_kept = keep_mask.sum().item()
        total_fp16 = 0

        protected_layers = set()
        for i in range(min(protect_boundary, n_layers)):
            protected_layers.add(i)
        for i in range(max(0, n_layers - protect_boundary), n_layers):
            protected_layers.add(i)
        n_compressed_layers = n_layers - len(protected_layers)

        all_key_coords = []
        all_val_coords = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            if l in protected_layers:
                continue

            for is_key, tensor, bits, coord_list in [
                (True,  k, key_bits,   all_key_coords),
                (False, v, value_bits, all_val_coords),
            ]:
                levels = 2 ** bits
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    kept_data = t_head[keep_mask]
                    rotated = kept_data @ H.T
                    amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc = amax / (levels / 2)
                    normalized = rotated / sc
                    groups = normalized.reshape(-1, 8)
                    lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords = lp.reshape(-1, head_dim)
                    quantized = (coords * sc) @ H
                    int_coords = coords.detach().numpy()
                    has_half = np.any(
                        np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25
                    )
                    if has_half:
                        coord_list.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                    else:
                        coord_list.append(np.round(int_coords.flatten()).astype(np.int8))
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    if is_key:
                        t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = result
                # Determine target device from existing tensor
                target_device = kl.device
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to(target_device), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to(target_device))

        total_idx = 0
        for coords_arr in all_key_coords + all_val_coords:
            arr = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped = arr.reshape(n_kept, n_per_tok)
                delta = np.zeros_like(reshaped)
                delta[0] = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx += len(cctx.compress(arr.tobytes()))

        scale_bytes      = n_kept * n_kv_heads * 2 * 2 * n_compressed_layers
        mask_bytes       = math.ceil(plen / 8) * n_compressed_layers
        protected_fp16   = len(protected_layers) * plen * n_kv_heads * head_dim * 2 * 2
        total_compressed = total_idx + scale_bytes + mask_bytes + protected_fp16

        return {
            "fp16": total_fp16,
            "idx":  total_idx,
            "scale": scale_bytes,
            "mask":  mask_bytes,
            "protected_fp16": protected_fp16,
            "total": total_compressed,
            "ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "n_kept": n_kept,
            "n_protected_layers": len(protected_layers),
            "n_compressed_layers": n_compressed_layers,
        }, kv_cache

    # ---- Tokenize ----
    inputs = tok(CORPUS, return_tensors="pt", max_length=4096, truncation=True)
    n_tok = inputs.input_ids.shape[1]

    actual_prefix = min(prefix_len, n_tok - cont_len)
    if actual_prefix < 300:
        return {"error": f"Corpus too short: {n_tok} tokens"}

    actual_cont = n_tok - actual_prefix
    print(f"\nTokens: total={n_tok}, prefix={actual_prefix}, continuation={actual_cont}")

    # Move input to GPU (first device from device_map)
    primary_device = next(model.parameters()).device
    full_ids = inputs.input_ids.to(primary_device)

    # ---- Baseline PPL ----
    try:
        with torch.no_grad():
            pout = model(full_ids[:, :actual_prefix], use_cache=True)
            cout = model(
                full_ids[:, actual_prefix:],
                past_key_values=pout.past_key_values,
                use_cache=True,
            )
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, actual_prefix + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"\nBaseline PPL (no compression): {baseline_ppl:.4f}")
    except Exception as exc:
        return {"error": f"Baseline failed: {exc}"}

    # ---- Score importance ----
    try:
        with torch.no_grad():
            pout_score = model(full_ids[:, :actual_prefix], use_cache=True)
        importance = score_importance(pout_score.past_key_values)
        del pout_score
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        return {"error": f"Importance scoring failed: {exc}"}

    # ---- Configs (boundary >= 2 always enforced inside evict_quantize) ----
    CONFIGS = [
        # K2V2+boundary at 35% and 60%
        {"name": "K2V2+boundary 35%ev", "evict_pct": 35, "key_bits": 2, "val_bits": 2, "protect_boundary": 2},
        {"name": "K2V2+boundary 60%ev", "evict_pct": 60, "key_bits": 2, "val_bits": 2, "protect_boundary": 2},
        # K3V2+boundary at 35% and 60%
        {"name": "K3V2+boundary 35%ev", "evict_pct": 35, "key_bits": 3, "val_bits": 2, "protect_boundary": 2},
        {"name": "K3V2+boundary 60%ev", "evict_pct": 60, "key_bits": 3, "val_bits": 2, "protect_boundary": 2},
    ]

    print(f"\n{'Config':<35s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot':>5s}")
    print("-" * 72)

    all_results = []
    for cfg in CONFIGS:
        torch.cuda.empty_cache()
        name             = cfg["name"]
        evict_pct        = cfg["evict_pct"]
        key_bits         = cfg["key_bits"]
        val_bits         = cfg["val_bits"]
        protect_boundary = cfg["protect_boundary"]

        try:
            with torch.no_grad():
                pout = model(full_ids[:, :actual_prefix], use_cache=True)
                kv = pout.past_key_values

            keep_mask = build_keep_mask(actual_prefix, evict_pct, importance)
            info, kv  = evict_quantize(
                kv, keep_mask, actual_prefix,
                key_bits=key_bits, value_bits=val_bits,
                protect_boundary=protect_boundary,
            )

            evict_mask = ~keep_mask
            attn_ctx = torch.ones(actual_prefix, dtype=torch.long, device=primary_device)
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(actual_cont, dtype=torch.long, device=primary_device),
            ])

            with torch.no_grad():
                cout = model(
                    full_ids[:, actual_prefix:],
                    past_key_values=kv,
                    attention_mask=attn_full.unsqueeze(0),
                    use_cache=True,
                )
                logits = cout.logits[:, :-1, :].float()
                targets = full_ids[:, actual_prefix + 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 1.0 else (" <<" if abs(delta) < 2.0 else "")
            n_prot = info.get("n_protected_layers", 0)
            print(f"{name:<35s} {ppl:8.4f} {delta:+8.3f}% {info['ratio']:6.2f}x "
                  f"{info['n_kept']:5d} {n_prot:4d}{tag}")
            all_results.append({
                "name": name, "evict_pct": evict_pct,
                "key_bits": key_bits, "val_bits": val_bits,
                "protect_boundary": protect_boundary,
                "ppl": ppl, "delta_pct": delta,
                "ratio": info["ratio"], "n_kept": info["n_kept"],
                "n_protected": n_prot,
                "baseline_ppl": baseline_ppl,
                "prefix_len": actual_prefix,
                "cont_len": actual_cont,
                "model": model_name,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"{name:<35s} OOM — skipped")
            gc.collect()
            torch.cuda.empty_cache()
            all_results.append({"name": name, "error": "OOM"})
        except Exception as exc:
            print(f"{name:<35s} ERROR: {exc}")
            gc.collect()
            torch.cuda.empty_cache()
            all_results.append({"name": name, "error": str(exc)})

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — {model_name}")
    print(f"Baseline PPL: {baseline_ppl:.4f}  |  prefix={actual_prefix}  cont={actual_cont}")
    print(f"Boundary protection: mandatory (protect_boundary>=2 for Qwen)")
    print(f"{'='*80}")
    print(f"\n{'Config':<35s} {'PPL Δ%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot L':>7s}")
    print("-" * 66)
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<35s}  ({r['error']})")
            continue
        tag = " <<<" if abs(r["delta_pct"]) < 1.0 else (" <<" if abs(r["delta_pct"]) < 2.0 else "")
        print(f"{r['name']:<35s} {r['delta_pct']:+8.3f}% {r['ratio']:6.2f}x "
              f"{r['n_kept']:5d} {r['n_protected']:6d}{tag}")

    # Best tradeoffs
    good = [r for r in all_results if "error" not in r and abs(r["delta_pct"]) < 1.0]
    if good:
        best = max(good, key=lambda r: r["ratio"])
        print(f"\nBest <1% PPL: {best['name']}  "
              f"delta={best['delta_pct']:+.3f}%  ratio={best['ratio']:.2f}x")
    else:
        valid = [r for r in all_results if "error" not in r]
        if valid:
            closest = min(valid, key=lambda r: abs(r["delta_pct"]))
            print(f"\nNo config <1% — closest: {closest['name']}  "
                  f"delta={closest['delta_pct']:+.3f}%  ratio={closest['ratio']:.2f}x")

    print(f"\n{'='*80}")
    print("Done.")
    print(f"{'='*80}")

    return {
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "prefix_len": actual_prefix,
        "cont_len": actual_cont,
        "results": all_results,
    }


# Allow direct execution for local testing
if __name__ == "__main__":
    import json as _json
    result = run_14b(prefix_len=2000, cont_len=200)
    print("\n--- JSON OUTPUT ---")
    print(_json.dumps(result, indent=2))

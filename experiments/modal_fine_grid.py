"""Fine-grid eviction sweep on Mistral-7B — narrow the cliff at 2b+35-38%.

Tests ONLY:
  2b + 33% evict
  2b + 34% evict
  2b + 36% evict
  2b + 37% evict

Goal: find max compression at <1% PPL degradation on BOTH easy and hard text.
Previous: 35% = 10.07x / 0.985% (pass), 38% = 10.55x / 1.94% (fail).
"""
import modal
import os

app = modal.App("nexusquant-fine-grid")

nq_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nexusquant-oss", "nexusquant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.44.0,<5.0.0",
        "accelerate>=0.27.0",
        "zstandard>=0.22.0",
        "numpy<2.0",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(nq_local, remote_path="/root/nexusquant")
)

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": "os.environ.get("HF_TOKEN", "")"})


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_fine_grid():
    import sys
    sys.path.insert(0, "/root")

    import time, math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    # ------------------------------------------------------------------ helpers
    def get_kv(cache, layer):
        if hasattr(cache, 'key_cache'):
            return cache.key_cache[layer], cache.value_cache[layer]
        return cache.layers[layer].keys, cache.layers[layer].values

    def set_kv(cache, layer, k, v):
        if hasattr(cache, 'key_cache'):
            cache.key_cache[layer] = k
            cache.value_cache[layer] = v
        else:
            cache.layers[layer].keys = k
            cache.layers[layer].values = v

    print("=" * 80)
    print("NEXUSQUANT — Fine Grid Sweep (Mistral-7B)")
    print("2b + {33, 34, 36, 37}% eviction — narrow the cliff")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers          # 32
    n_kv_heads = model.config.num_key_value_heads        # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_base={rope_base}")

    # ------------------------------------------------------------------ corpora (copied exactly from modal_quality_sweep.py)
    EASY_TEXT = " ".join([
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists around the world, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. Fermions are the building blocks: six quarks and six leptons. Forces between the fermions are mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation. It began in Britain and quickly spread to Western Europe and North America. The transition from hand production methods to machine manufacturing, new chemical processes, iron production, increased use of steam power, the development of machine tools, and the rise of the factory system fundamentally changed the nature of work and society.",
        "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better suited to their environment will tend to survive and produce more offspring. Over time, these advantageous traits become more common in the population.",
        "The development of quantum mechanics in the early 20th century fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain phenomena such as black-body radiation, the photoelectric effect, or the stability of atoms. Max Planck introduced the concept of energy quanta in 1900. Albert Einstein explained the photoelectric effect using photons in 1905.",
        "Mathematics has been essential to the development of science and technology throughout human history. From the ancient Babylonians who developed a base-60 number system that we still use for measuring time, to the development of calculus by Newton and Leibniz that made modern physics possible, mathematical ideas have been the foundation of scientific progress.",
        "The history of astronomy represents one of humanity's oldest scientific endeavors. Ancient civilizations observed the stars for navigation, agriculture, and religious purposes. The Babylonians recorded planetary positions, the Greeks developed geometric models of the cosmos, and Islamic astronomers preserved and extended this knowledge during the medieval period.",
        "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence was felt in literature, philosophy, art, music, politics, science, religion, and other aspects of intellectual inquiry.",
        "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons connected by trillions of synapses. Each neuron can form thousands of connections with other neurons, creating an intricate network that gives rise to thought, memory, emotion, and consciousness.",
        "Climate change represents one of the most significant challenges facing humanity in the 21st century. The burning of fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of greenhouse gases, particularly carbon dioxide and methane, to levels unprecedented in at least 800,000 years.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. It had a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around 100 BC to 400 AD.",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data. Deep learning uses artificial neural networks with many layers.",
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of the Earth's lithosphere. Tectonic plates move because of the relative density of oceanic lithosphere and the relative weakness of the asthenosphere.",
    ])

    HARD_TEXT = " ".join([
        "The renormalization group in quantum field theory provides a systematic framework for understanding how physical theories change with the energy scale of observation. Kenneth Wilson's formulation connects statistical mechanics and quantum field theory through the concept of universality classes, where disparate physical systems exhibit identical critical behavior near phase transitions due to shared symmetry properties and dimensionality.",
        "Homological algebra studies algebraic structures through chain complexes and their derived functors. The Ext and Tor functors provide fundamental invariants that measure the failure of exactness under the Hom and tensor product functors respectively. Spectral sequences provide computational tools for successive approximation of these derived functors.",
        "The Langlands program represents one of the most ambitious unifying frameworks in mathematics, connecting number theory, algebraic geometry, and representation theory. The geometric Langlands correspondence establishes deep connections between automorphic forms on reductive groups and l-adic representations of absolute Galois groups over function fields.",
        "CRISPR-Cas9 gene editing exploits the bacterial adaptive immune system's ability to incorporate foreign DNA fragments into clustered regularly interspaced short palindromic repeats. The Cas9 endonuclease, guided by a chimeric single-guide RNA, introduces double-strand breaks at specific genomic loci, enabling precise modifications through nonhomologous end joining or homology-directed repair pathways.",
        "The AdS/CFT correspondence, proposed by Juan Maldacena in 1997, conjectures an exact duality between type IIB superstring theory on five-dimensional anti-de Sitter space times a five-sphere and four-dimensional N=4 super Yang-Mills theory on the conformal boundary. This holographic principle has profound implications for quantum gravity and strongly coupled quantum field theories.",
        "Stochastic partial differential equations driven by space-time white noise arise naturally in the study of random interface growth, population genetics, and directed polymers in random environments. The Kardar-Parisi-Zhang equation describes the universal scaling behavior of growing interfaces, connecting to random matrix theory through the Tracy-Widom distribution.",
        "The tumor microenvironment comprises a complex ecosystem of cancer cells, immune cells, fibroblasts, endothelial cells, and extracellular matrix components that collectively determine tumor progression and therapeutic response. Immune checkpoint inhibitors targeting PD-1/PD-L1 and CTLA-4 have revolutionized oncology by releasing the brakes on anti-tumor immune responses.",
        "Quantum error correction addresses the fundamental challenge of protecting quantum information from decoherence and operational errors. Surface codes, implemented on a two-dimensional lattice of physical qubits, achieve fault-tolerant computation through topological protection, where logical qubits are encoded in the global topology of the code rather than individual physical qubits.",
        "The Navier-Stokes equations governing incompressible fluid flow remain one of the seven Millennium Prize Problems. The existence and smoothness of solutions in three dimensions is unresolved, with turbulent cascading energy transfer from large to small scales described by Kolmogorov's 1941 theory predicting the famous minus five-thirds power law in the inertial range of the energy spectrum.",
        "Persistent homology in topological data analysis provides multiscale shape descriptors for point cloud data by tracking the birth and death of topological features across a filtration of simplicial complexes. The resulting persistence diagrams and barcodes offer stable invariants under perturbation, with stability guarantees provided by the bottleneck and Wasserstein distances.",
        "Nonequilibrium statistical mechanics extends beyond the Boltzmann-Gibbs framework to describe systems driven away from thermal equilibrium. The Jarzynski equality and Crooks fluctuation theorem relate free energy differences to nonequilibrium work measurements, while large deviation theory provides the mathematical foundation for understanding rare events in stochastic processes.",
        "The hypothalamic-pituitary-adrenal axis orchestrates the neuroendocrine stress response through a cascade of hormonal signals. Corticotropin-releasing hormone from the paraventricular nucleus stimulates adrenocorticotropic hormone release from the anterior pituitary, which in turn drives cortisol synthesis in the adrenal cortex, with glucocorticoid receptors mediating negative feedback at multiple levels.",
    ])

    sliding_window = 32

    # ------------------------------------------------------------------ scorer
    def score_importance(kv_cache, obs_window=32, pool_kernel=5):
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = (all_pos <= obs_pos)
            scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                         padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict+quantize
    def evict_quantize(kv_cache, keep_mask, key_bits, val_bits, prefix_len):
        """Quantize + evict the KV cache.

        Compression accounting:
          - zstd-22 compressed indices (delta-coded per token)
          - fp16 scale per vector (one scale per kept token per head per layer, K+V)
          - ceil(prefix_len/8)*n_layers mask bits (shared mask, no *2)
        """
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_key_coords = []
        all_val_coords = []

        cctx = zstandard.ZstdCompressor(level=22)

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2  # bytes if fp16

            for is_key, tensor, bits, coord_list in [
                (True,  k, key_bits, all_key_coords),
                (False, v, val_bits, all_val_coords),
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
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        # compressed index bytes
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

        # fp16 scale: one scale per kept token per head per layer, for both K and V
        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2  # *2 for K+V, *2 for fp16
        # mask: shared single mask (no *2), ceil to byte boundary, one per layer
        mask_bytes = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes

        return {
            "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
            "mask": mask_bytes, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep_mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_pct = 100 - evict_pct
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True                       # always keep BOS
        keep_mask[-sliding_window:] = True        # always keep sliding window
        n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ run on text
    def run_configs_on_text(text_name, text, configs):
        inputs = tok(text, return_tensors="pt", max_length=2048, truncation=True)
        full_ids = inputs.input_ids.to("cuda")
        n_tok = full_ids.shape[1]
        prefix_len = n_tok // 2
        cont_len = n_tok - prefix_len
        print(f"\n{'='*80}")
        print(f"TEXT: {text_name} | tokens={n_tok}, prefix={prefix_len}, cont={cont_len}")
        print(f"{'='*80}")

        # Baseline (uncompressed)
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            cout = model(full_ids[:, prefix_len:],
                         past_key_values=pout.past_key_values, use_cache=True)
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"Baseline PPL: {baseline_ppl:.4f}")

        # Score importance once (shared across configs)
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
        importance = score_importance(pout.past_key_values)

        results = []
        print(f"\n{'Config':<20s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s}")
        print("-" * 56)

        for cfg in configs:
            torch.cuda.empty_cache()
            name      = cfg["name"]
            evict_pct = cfg["evict_pct"]
            key_bits  = cfg["key_bits"]
            val_bits  = cfg["val_bits"]

            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv = pout.past_key_values

            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            info, kv = evict_quantize(kv, keep_mask, key_bits, val_bits, prefix_len)

            evict_mask = ~keep_mask
            attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(cont_len, dtype=torch.long, device="cuda")
            ])

            with torch.no_grad():
                cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                             attention_mask=attn_full.unsqueeze(0), use_cache=True)
                logits = cout.logits[:, :-1, :].float()
                targets = full_ids[:, prefix_len + 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 1.0 else ""
            print(f"{name:<20s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x {info['n_kept']:5d}{tag}")
            results.append({
                "name": name, "evict_pct": evict_pct,
                "key_bits": key_bits, "val_bits": val_bits,
                "ppl": ppl, "delta": delta,
                "ratio": info["ratio"], "n_kept": info["n_kept"],
                "baseline": baseline_ppl, "text": text_name,
            })

        return results

    # ------------------------------------------------------------------ configs: tight 2b grid
    configs = [
        {"name": "2b+33%evict", "evict_pct": 33, "key_bits": 2, "val_bits": 2},
        {"name": "2b+34%evict", "evict_pct": 34, "key_bits": 2, "val_bits": 2},
        {"name": "2b+36%evict", "evict_pct": 36, "key_bits": 2, "val_bits": 2},
        {"name": "2b+37%evict", "evict_pct": 37, "key_bits": 2, "val_bits": 2},
    ]

    # ------------------------------------------------------------------ run both corpora
    all_results = []
    all_results.extend(run_configs_on_text("EASY (general science)", EASY_TEXT, configs))
    all_results.extend(run_configs_on_text("HARD (advanced technical)", HARD_TEXT, configs))

    # ------------------------------------------------------------------ summary table
    print(f"\n{'='*80}")
    print("FINE GRID SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Config':<20s} {'Easy Δ%':>8s} {'Hard Δ%':>8s} {'Max Δ%':>8s} {'Ratio':>7s} {'<1%?':>6s}")
    print("-" * 62)

    best_config = None
    best_ratio  = 0.0

    for cfg in configs:
        name = cfg["name"]
        easy = next((r for r in all_results if r["name"] == name and "EASY" in r["text"]), None)
        hard = next((r for r in all_results if r["name"] == name and "HARD" in r["text"]), None)
        if easy and hard:
            max_d    = max(abs(easy["delta"]), abs(hard["delta"]))
            avg_ratio = (easy["ratio"] + hard["ratio"]) / 2
            ok = "YES" if max_d < 1.0 else ("CLOSE" if max_d < 2.0 else "NO")
            marker = " <<<" if ok == "YES" else ""
            print(f"{name:<20s} {easy['delta']:+7.2f}% {hard['delta']:+7.2f}% "
                  f"{max_d:7.2f}% {avg_ratio:6.2f}x {ok:>5s}{marker}")
            if ok == "YES" and avg_ratio > best_ratio:
                best_ratio  = avg_ratio
                best_config = {
                    "name": name,
                    "easy_delta": easy["delta"],
                    "hard_delta": hard["delta"],
                    "ratio": avg_ratio,
                    "max_delta": max_d,
                }

    print()
    if best_config:
        print(f">>> BEST <1% CONFIG: {best_config['name']}")
        print(f"    Easy Δ%={best_config['easy_delta']:+.3f}%  "
              f"Hard Δ%={best_config['hard_delta']:+.3f}%  "
              f"Max={best_config['max_delta']:.3f}%  "
              f"Ratio={best_config['ratio']:.2f}x")
    else:
        print(">>> No config achieved <1% on BOTH texts in this grid.")
        combos = []
        for cfg in configs:
            name = cfg["name"]
            easy = next((r for r in all_results if r["name"] == name and "EASY" in r["text"]), None)
            hard = next((r for r in all_results if r["name"] == name and "HARD" in r["text"]), None)
            if easy and hard:
                max_d = max(abs(easy["delta"]), abs(hard["delta"]))
                combos.append((max_d, cfg["name"],
                                (easy["ratio"] + hard["ratio"]) / 2,
                                easy["delta"], hard["delta"]))
        combos.sort()
        if combos:
            md, nm, rt, ed, hd = combos[0]
            print(f"    Closest: {nm}  MaxΔ={md:.3f}%  ratio={rt:.2f}x  "
                  f"easy={ed:+.3f}%  hard={hd:+.3f}%")

    print(f"\nPeak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return all_results, best_config


@app.local_entrypoint()
def main():
    import time
    print("Launching fine-grid sweep on Modal A10G...")
    all_results, best_config = run_fine_grid.remote()

    out_dir = os.path.join(os.path.dirname(__file__), "..", ".planning", "research")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "FINE_GRID_RESULTS.md")

    with open(out, "w") as f:
        f.write("# Fine Grid Sweep Results — Mistral-7B\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Goal:** Find max compression at <1% PPL — tight grid around 35-38% eviction\n")
        f.write("**Context:** 35%=10.07x/0.985% (pass), 38%=10.55x/1.94% (fail)\n\n")

        f.write("## Full Results\n\n")
        f.write("| Text | Config | PPL | Delta% | Ratio | Kept |\n")
        f.write("|------|--------|-----|--------|-------|------|\n")
        for r in all_results:
            f.write(f"| {r['text'][:25]} | {r['name']} | {r['ppl']:.4f} "
                    f"| {r['delta']:+.3f}% | {r['ratio']:.2f}x | {r['n_kept']} |\n")

        f.write("\n## Summary Table\n\n")
        f.write("| Config | Easy Δ% | Hard Δ% | Max Δ% | Ratio | <1%? |\n")
        f.write("|--------|---------|---------|--------|-------|------|\n")

        seen_easy = {r["name"]: r for r in all_results if "EASY" in r["text"]}
        seen_hard = {r["name"]: r for r in all_results if "HARD" in r["text"]}
        for cfg in [{"name": "2b+33%evict"}, {"name": "2b+34%evict"},
                    {"name": "2b+36%evict"}, {"name": "2b+37%evict"}]:
            nm = cfg["name"]
            if nm in seen_easy and nm in seen_hard:
                e = seen_easy[nm]
                h = seen_hard[nm]
                max_d = max(abs(e["delta"]), abs(h["delta"]))
                avg_r = (e["ratio"] + h["ratio"]) / 2
                ok = "YES" if max_d < 1.0 else ("CLOSE" if max_d < 2.0 else "NO")
                f.write(f"| {nm} | {e['delta']:+.3f}% | {h['delta']:+.3f}% "
                        f"| {max_d:.3f}% | {avg_r:.2f}x | {ok} |\n")

        f.write("\n## Verdict\n\n")
        if best_config:
            f.write(f"**Best <1% config: {best_config['name']}**\n\n")
            f.write(f"- Easy Δ%: {best_config['easy_delta']:+.3f}%\n")
            f.write(f"- Hard Δ%: {best_config['hard_delta']:+.3f}%\n")
            f.write(f"- Max Δ%: {best_config['max_delta']:.3f}%\n")
            f.write(f"- Compression ratio: {best_config['ratio']:.2f}x\n")
        else:
            f.write("No config achieved <1% PPL on BOTH texts in this sweep.\n")
            f.write("The quality cliff is sharper than expected between 35% and 36%.\n")

    print(f"\nResults written to: {out}")

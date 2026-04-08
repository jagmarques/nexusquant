"""Eviction + E8 at 1024 prefix tokens — definitive 20x validation.

Uses 9 concatenated passages (same as final_standard_benchmark.py) for 2048 tokens.
Tests eviction rates 0-80% with 2-bit and 3-bit E8.
All overhead included. Honest compression accounting.
"""
import sys, os, copy, time, math, functools
import numpy as np
import torch
import torch.nn.functional as F

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nexusquant-oss"))
from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.rope_utils import inverse_rope, forward_rope
import zstandard

PASSAGES = [
    "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists around the world, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. Fermions are the building blocks: six quarks and six leptons. Forces between the fermions are mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. The photon mediates the electromagnetic force between electrically charged particles. The W and Z bosons mediate the weak force. The eight gluons mediate the strong force between quarks. The graviton is hypothesized to mediate the gravitational force, but is not part of the Standard Model.",
    "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation. It began in Britain and quickly spread to Western Europe and North America. The transition from hand production methods to machine manufacturing, new chemical processes, iron production, increased use of steam power, the development of machine tools, and the rise of the factory system fundamentally changed the nature of work and society. The textile industry was the first to use modern production methods. The iron and steel industries, along with the development of the steam engine, played central roles in the Industrial Revolution. The introduction of steam-powered ships and railways transformed transportation and commerce. Working conditions in factories were often harsh, leading to the development of labor movements and eventual reforms. The Industrial Revolution marks a major turning point in history, as it affected every aspect of daily life and led to unprecedented economic growth.",
    "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better suited to their environment will tend to survive and produce more offspring. Over time, these advantageous traits become more common in the population. Darwin spent decades gathering evidence before publishing On the Origin of Species in 1859. The book was controversial but the scientific evidence was overwhelming. Key mechanisms of evolution include natural selection, genetic drift, mutation, and gene flow. The fossil record provides evidence of species that lived millions of years ago and shows how life has changed over time. DNA evidence has confirmed and extended Darwin's insights, showing that all life on Earth shares common ancestors. Modern evolutionary biology integrates genetics, paleontology, ecology, and molecular biology into a comprehensive understanding of how life diversifies and adapts.",
    "The development of quantum mechanics in the early 20th century fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain phenomena such as black-body radiation, the photoelectric effect, or the stability of atoms. Max Planck introduced the concept of energy quanta in 1900. Albert Einstein explained the photoelectric effect using photons in 1905. Niels Bohr proposed a quantum model of the atom in 1913. The full mathematical framework was developed in the 1920s by Werner Heisenberg, Erwin Schrodinger, Max Born, and Paul Dirac. The Copenhagen interpretation, primarily due to Bohr and Heisenberg, holds that quantum mechanics describes probabilities rather than definite outcomes. The uncertainty principle states that certain pairs of physical properties cannot both be known to arbitrary precision. Quantum entanglement, described by Einstein as spooky action at a distance, has been experimentally verified and is now the basis for quantum computing and quantum cryptography.",
    "Mathematics has been essential to the development of science and technology throughout human history. From the ancient Babylonians who developed a base-60 number system that we still use for measuring time, to the development of calculus by Newton and Leibniz that made modern physics possible, mathematical ideas have been the foundation of scientific progress. The development of non-Euclidean geometry in the 19th century led to Einstein's general theory of relativity, which describes gravity as the curvature of spacetime. Group theory, originally developed to study the symmetries of polynomial equations, has become the language of modern particle physics, where symmetry groups determine the fundamental forces and particles. The advent of computers has transformed mathematics itself, enabling numerical simulations that reveal patterns invisible to analytical methods, and leading to entirely new fields such as computational complexity theory and algorithmic information theory. Modern mathematics continues to surprise with unexpected connections between seemingly unrelated areas, such as the link between number theory and cryptography that underlies all secure digital communication.",
    "The history of astronomy represents one of humanity's oldest scientific endeavors. Ancient civilizations observed the stars for navigation, agriculture, and religious purposes. The Babylonians recorded planetary positions, the Greeks developed geometric models of the cosmos, and Islamic astronomers preserved and extended this knowledge during the medieval period. The Copernican revolution, placing the Sun at the center of the solar system, fundamentally changed our understanding of Earth's place in the universe. Galileo's telescopic observations, Kepler's laws of planetary motion, and Newton's law of universal gravitation provided the theoretical framework that explained celestial mechanics. Modern astronomy has expanded to include the study of galaxies, cosmic microwave background radiation, dark matter, and dark energy, revealing a universe far larger and more complex than any previous generation could have imagined. Space-based observatories like the Hubble Space Telescope and the James Webb Space Telescope have provided unprecedented views of the universe, from nearby planets to galaxies billions of light-years away.",
    "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence was felt in literature, philosophy, art, music, politics, science, religion, and other aspects of intellectual inquiry. Renaissance scholars employed the humanist method in study and searched for realism and human emotion in art. Renaissance thinkers sought out learning from ancient texts, typically written in Latin or ancient Greek. The art of the Renaissance period was characterized by a focus on perspective, proportion, and the human form. Artists such as Leonardo da Vinci, Michelangelo, and Raphael are considered among the greatest painters of all time. The invention of the printing press by Johannes Gutenberg around 1440 greatly accelerated the spread of knowledge and ideas. The Scientific Revolution that followed the Renaissance laid the foundations for modern science, with figures like Copernicus, Galileo, Kepler, and Newton challenging traditional views of the natural world through observation and mathematical reasoning.",
    "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons connected by trillions of synapses. Each neuron can form thousands of connections with other neurons, creating an intricate network that gives rise to thought, memory, emotion, and consciousness. The cerebral cortex, the outer layer of the brain, is responsible for higher cognitive functions including language, abstract thinking, and decision making. The hippocampus plays a crucial role in the formation of new memories and spatial navigation. The amygdala processes emotions, particularly fear and anxiety. Neurotransmitters such as dopamine, serotonin, and glutamate carry signals between neurons and regulate mood, attention, and learning. Modern neuroscience has revealed that the brain exhibits remarkable plasticity, the ability to reorganize itself by forming new neural connections throughout life. Advances in brain imaging technology, including functional MRI and positron emission tomography, have allowed scientists to observe the brain in action, mapping which regions are active during specific tasks. Despite these advances, many fundamental questions about consciousness, free will, and the nature of subjective experience remain unanswered.",
    "Climate change represents one of the most significant challenges facing humanity in the 21st century. The burning of fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of greenhouse gases, particularly carbon dioxide and methane, to levels unprecedented in at least 800,000 years. The resulting warming has led to rising sea levels, more frequent and intense extreme weather events, shifts in ecosystems and species distributions, and threats to food and water security. The Paris Agreement of 2015 set a goal of limiting global warming to well below 2 degrees Celsius above pre-industrial levels. Achieving this goal requires rapid and deep reductions in greenhouse gas emissions across all sectors of the economy. Renewable energy sources including solar, wind, and hydroelectric power have become increasingly cost-competitive with fossil fuels. Carbon capture and storage technologies are being developed to remove carbon dioxide directly from the atmosphere. The transition to a low-carbon economy presents both challenges and opportunities, requiring significant investment in new infrastructure, changes in agricultural practices, and international cooperation on an unprecedented scale.",
]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {m}...")
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
    model.eval()
    return model, tok


def score_importance(kv, n_layers, n_heads, head_dim, obs_window=32):
    """Key-key attention importance scorer (SnapKV-style, causal)."""
    seq_len = kv.layers[0].keys.shape[2]
    w = min(obs_window, seq_len)
    all_imp = torch.zeros(seq_len)

    for l in range(n_layers):
        k = kv.layers[l].keys[0].float()
        k_obs = k[:, -w:, :]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale

        all_pos = torch.arange(seq_len).unsqueeze(0)
        obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
        causal = (all_pos <= obs_pos)
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        layer_imp = attn.sum(dim=1).mean(dim=0)

        if seq_len > 5:
            imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
            layer_imp = F.avg_pool1d(imp_1d, kernel_size=5, padding=2, stride=1).squeeze()[:seq_len]

        all_imp += layer_imp

    return all_imp / n_layers


def evict_quantize_measure(kv, keep_mask, bits, head_dim, rope_base, seq_len):
    """Evict tokens, E8 quantize kept tokens, measure compression."""
    n_layers = len(kv.layers)
    levels = 2 ** bits
    H = hadamard_matrix(head_dim)
    n_kept = keep_mask.sum().item()
    total_fp16 = 0
    all_coords = []

    for l in range(n_layers):
        layer = kv.layers[l]
        k = layer.keys.float()
        v = layer.values.float()
        total_fp16 += k.numel() * 2 + v.numel() * 2

        for is_key, tensor in [(True, k), (False, v)]:
            t = tensor[0].clone()
            n_heads_local = t.shape[0]

            for h in range(n_heads_local):
                if is_key:
                    t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                else:
                    t_head = t[h]

                kept_data = t_head[keep_mask]
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

                int_coords = coords.detach().numpy()
                has_half = np.any(np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25)
                if has_half:
                    all_coords.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                else:
                    all_coords.append(np.round(int_coords.flatten()).astype(np.int8))

                result = torch.zeros_like(t_head)
                result[keep_mask] = quantized
                if is_key:
                    t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                else:
                    t[h] = result

            if is_key:
                layer.keys = t.unsqueeze(0).half()
            else:
                layer.values = t.unsqueeze(0).half()

    # Compress per-head-layer with temporal delta + zstd
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

    scale_bytes = n_kept * n_layers * 2 * n_heads_local * 2  # fp16 per kept vector
    mask_bytes = math.ceil(seq_len / 8) * n_layers * 2
    total = total_idx + scale_bytes + mask_bytes

    return {
        "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
        "mask": mask_bytes, "total": total, "ratio": total_fp16 / total if total > 0 else 0,
        "n_kept": n_kept, "n_total": seq_len,
    }


def main():
    model, tok = load_model()

    # Concatenate passages to get 2048+ tokens
    full_text = " ".join(PASSAGES)
    inputs = tok(full_text, return_tensors="pt", max_length=2048, truncation=True)
    full_ids = inputs.input_ids
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"Total tokens: {n_tok}, prefix: {prefix_len}, continuation: {cont_len}")

    n_layers = model.config.num_hidden_layers
    n_heads = 4
    head_dim = 64
    rope_base = getattr(model.config, 'rope_theta', 10000.0)
    sliding_window = min(32, prefix_len)

    # Baseline PPL
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_bl = prefix_out.past_key_values
        cont = model(full_ids[:, prefix_len:], past_key_values=kv_bl, use_cache=True)
        logits = cont.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    # Score importance
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_score = prefix_out.past_key_values
    importance = score_importance(kv_score, n_layers, n_heads, head_dim)

    configs = [
        ("2b no-evict",       2, 100),
        ("3b no-evict",       3, 100),
        ("2b + 50% evict",    2,  50),
        ("3b + 50% evict",    3,  50),
        ("2b + 60% evict",    2,  40),
        ("3b + 60% evict",    3,  40),
        ("2b + 70% evict",    2,  30),
        ("3b + 70% evict",    3,  30),
        ("2b + 75% evict",    2,  25),
        ("3b + 75% evict",    3,  25),
        ("2b + 80% evict",    2,  20),
        ("3b + 80% evict",    3,  20),
    ]

    hdr = f"{'Config':<22s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Idx':>9s} {'Scale':>8s} {'Mask':>6s} {'Total':>10s}"
    print(hdr)
    print("-" * len(hdr))

    results = []

    for name, bits, keep_pct in configs:
        t0 = time.time()

        with torch.no_grad():
            prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
            kv = prefix_out.past_key_values

        if keep_pct >= 100:
            keep_mask = torch.ones(prefix_len, dtype=torch.bool)
        else:
            keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
            keep_mask[0] = True
            keep_mask[-sliding_window:] = True
            n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
            n_from_imp = n_to_keep - keep_mask.sum().item()
            if n_from_imp > 0:
                imp = importance.clone()
                imp[keep_mask] = -float('inf')
                _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
                keep_mask[top_idx] = True

        info = evict_quantize_measure(kv, keep_mask, bits, head_dim, rope_base, prefix_len)

        evict_mask = ~keep_mask
        attn_ctx = torch.ones(prefix_len, dtype=torch.long)
        attn_ctx[evict_mask] = 0
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

        print(f"{name:<22s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x "
              f"{info['n_kept']:5d} {info['idx']:>9,} {info['scale']:>8,} "
              f"{info['mask']:>6,} {info['total']:>10,}  ({elapsed:.1f}s)")

        results.append({"name": name, "bits": bits, "keep_pct": keep_pct,
                        "ppl": ppl, "delta": delta, **info})

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — prefix={prefix_len}, cont={cont_len}, baseline PPL={baseline_ppl:.4f}")
    print(f"{'='*80}")
    for r in results:
        tag = ""
        if r["ratio"] >= 20: tag = " *** 20x+ ***"
        elif r["ratio"] >= 15: tag = " (close)"
        print(f"  {r['name']:<22s} {r['ratio']:6.1f}x  {r['delta']:+6.2f}%{tag}")

    best_20x = [r for r in results if r["ratio"] >= 20 and r["delta"] < 5]
    if best_20x:
        b = min(best_20x, key=lambda x: x["delta"])
        print(f"\n  >>> BEST 20x+: {b['name']} = {b['ratio']:.1f}x at {b['delta']:+.2f}% PPL <<<")

    # Write results
    out = os.path.join(os.path.dirname(__file__), "..",
                       ".company", "engineering", "eviction_1024_results.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("# Eviction + E8 — 1024-Prefix Definitive Results\n\n")
        f.write(f"**Model:** TinyLlama 1.1B (22 layers, 4 KV heads, head_dim=64)\n")
        f.write(f"**Total tokens:** {n_tok} (prefix={prefix_len}, cont={cont_len})\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n")
        f.write(f"**Sliding window:** {sliding_window} (always kept + BOS)\n")
        f.write(f"**Scorer:** Key-key attention (last 32 tokens, causal mask)\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")
        f.write("| Config | PPL | Delta% | Ratio | Kept | Idx bytes | Scale | Mask | Total |\n")
        f.write("|--------|-----|--------|-------|------|-----------|-------|------|-------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['ppl']:.4f} | {r['delta']:+.2f}% | "
                    f"{r['ratio']:.2f}x | {r['n_kept']} | {r['idx']:,} | "
                    f"{r['scale']:,} | {r['mask']:,} | {r['total']:,} |\n")
        f.write("\n## Honest Notes\n\n")
        f.write("- All overhead included: compressed E8 indices + fp16 scales + token mask bits\n")
        f.write("- Evicted tokens zeroed, attention mask excludes them during continuation\n")
        f.write("- Importance scored after full prefill (standard for compress-after-prefill)\n")
        f.write("- Per-head per-layer compression with temporal delta + zstd level 22\n")
        f.write("- RoPE removed from keys before E8, reapplied after\n")
        f.write("- Single run, single text (9 concatenated passages)\n")
        f.write(f"- TinyLlama: 4 KV heads, head_dim=64 (2x scale overhead vs head_dim=128)\n")
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()

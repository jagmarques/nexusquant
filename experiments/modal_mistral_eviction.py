"""Modal GPU experiment: Eviction + E8 on Mistral-7B.

Validates 20x+ compression on a real 7B model with GPU.
Mistral-7B: 32 layers, 8 KV heads, head_dim=128, rope_theta=1e6.
head_dim=128 means 2x LESS scale overhead vs TinyLlama's head_dim=64.
"""
import modal
import os

app = modal.App("nexusquant-mistral-eviction")

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

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_experiment():
    import sys
    sys.path.insert(0, "/root")

    import time
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    # DynamicCache API helper — works with both old and new transformers
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

    def get_n_layers_cache(cache):
        if hasattr(cache, 'key_cache'):
            return len(cache.key_cache)
        return len(cache.layers)

    print("=" * 80)
    print("NEXUSQUANT — Mistral-7B GPU Validation")
    print("=" * 80)

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers  # 32
    n_kv_heads = model.config.num_key_value_heads  # 8
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base = getattr(model.config, 'rope_theta', 1000000.0)
    print(f"Config: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}, rope_theta={rope_base}")

    # Diverse text — target 2048+ tokens
    passages = [
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists around the world, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. Fermions are the building blocks: six quarks and six leptons. Forces between the fermions are mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation. It began in Britain and quickly spread to Western Europe and North America. The transition from hand production methods to machine manufacturing, new chemical processes, iron production, increased use of steam power, the development of machine tools, and the rise of the factory system fundamentally changed the nature of work and society. The textile industry was the first to use modern production methods.",
        "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better suited to their environment will tend to survive and produce more offspring. Over time, these advantageous traits become more common in the population. Darwin spent decades gathering evidence before publishing On the Origin of Species in 1859.",
        "The development of quantum mechanics in the early 20th century fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain phenomena such as black-body radiation, the photoelectric effect, or the stability of atoms. Max Planck introduced the concept of energy quanta in 1900. Albert Einstein explained the photoelectric effect using photons in 1905. Niels Bohr proposed a quantum model of the atom in 1913.",
        "Mathematics has been essential to the development of science and technology throughout human history. From the ancient Babylonians who developed a base-60 number system that we still use for measuring time, to the development of calculus by Newton and Leibniz that made modern physics possible, mathematical ideas have been the foundation of scientific progress. The development of non-Euclidean geometry in the 19th century led to Einstein's general theory of relativity.",
        "The history of astronomy represents one of humanity's oldest scientific endeavors. Ancient civilizations observed the stars for navigation, agriculture, and religious purposes. The Babylonians recorded planetary positions, the Greeks developed geometric models of the cosmos, and Islamic astronomers preserved and extended this knowledge during the medieval period. The Copernican revolution placed the Sun at the center of the solar system.",
        "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence was felt in literature, philosophy, art, music, politics, science, religion, and other aspects of intellectual inquiry. Artists such as Leonardo da Vinci, Michelangelo, and Raphael are considered among the greatest painters of all time.",
        "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons connected by trillions of synapses. Each neuron can form thousands of connections with other neurons, creating an intricate network that gives rise to thought, memory, emotion, and consciousness. The cerebral cortex is responsible for higher cognitive functions including language, abstract thinking, and decision making.",
        "Climate change represents one of the most significant challenges facing humanity in the 21st century. The burning of fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of greenhouse gases, particularly carbon dioxide and methane, to levels unprecedented in at least 800,000 years. The Paris Agreement of 2015 set a goal of limiting global warming to well below 2 degrees Celsius above pre-industrial levels.",
        "Python is a high-level general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms including structured, object-oriented, and functional programming. Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language.",
        "The Amazon River is the largest river in the world by discharge volume of water. It rises in the Andes mountains of Peru and flows eastward across the South American continent to empty into the Atlantic Ocean. The Amazon basin covers approximately 7 million square kilometers, making it the largest drainage basin in the world. The river system contains over 1100 tributaries.",
        "Modern economics traces its origins to Adam Smith's An Inquiry into the Nature and Causes of the Wealth of Nations, published in 1776. Smith argued that individuals pursuing their own self-interest through free markets would be guided by an invisible hand to promote the welfare of society as a whole. David Ricardo developed the theory of comparative advantage, showing that international trade benefits all countries.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. It had a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around 100 BC to 400 AD. The Empire was among the most powerful economic, cultural, political and military forces in the world of its time. At its height under Trajan it covered five million square kilometers. Roman roads, aqueducts, and engineering projects transformed the ancient world. Latin was the common language. Roman law influenced many modern legal systems.",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data. Deep learning uses artificial neural networks with many layers to learn representations of data. Natural language processing enables computers to understand and generate human language. Computer vision allows machines to interpret visual information from the world. Reinforcement learning trains agents through trial and error using rewards and penalties. AI has been applied to problems in healthcare, finance, transportation, and entertainment.",
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of the Earth's lithosphere. Tectonic plates move because of the relative density of oceanic lithosphere and the relative weakness of the asthenosphere. Convergent boundaries where plates collide create mountain ranges and subduction zones. Divergent boundaries where plates move apart create mid-ocean ridges and rift valleys. Transform boundaries where plates slide past each other create fault lines and earthquakes. The San Andreas Fault in California is a famous transform boundary. The Ring of Fire around the Pacific Ocean contains 75 percent of the world's active volcanoes.",
        "Music theory is the study of the practices and possibilities of music. Western music theory is founded on the division of the octave into twelve equally spaced semitones. Harmony describes how chords are constructed and progress. Counterpoint is the relationship between voices that are harmonically interdependent yet independent in rhythm. Johann Sebastian Bach's fugues remain the pinnacle of contrapuntal writing. The twentieth century saw twelve-tone technique by Arnold Schoenberg, chance music by John Cage, and electronic music that fundamentally changed how sound is produced. Jazz developed from African American communities in the early 20th century, combining African rhythmic traditions with European harmonic practices.",
    ]
    full_text = " ".join(passages)
    inputs = tok(full_text, return_tensors="pt", max_length=2048, truncation=True)
    full_ids = inputs.input_ids.to(model.device)
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"\nTokens: {n_tok}, prefix: {prefix_len}, continuation: {cont_len}")

    sliding_window = 32

    # ---------- Importance scorer ----------
    def score_importance(kv_cache, obs_window=32):
        """Key-key attention scorer (SnapKV-style, causal)."""
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')

        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()  # (n_heads, seq, dim)
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

    # ---------- Evict + quantize + measure ----------
    def evict_quantize(kv_cache, keep_mask, bits):
        """Evict tokens, E8 quantize kept, measure compression on CPU."""
        levels = 2 ** bits
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()   # (1, n_heads, seq, dim)
            v = vl.float().cpu()

            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor in [(True, k), (False, v)]:
                t = tensor[0].clone()  # (n_heads, seq, dim)

                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]

                    kept_data = t_head[keep_mask]  # (n_kept, dim)
                    rotated = kept_data @ H.T
                    amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc = amax / (levels / 2)
                    normalized = rotated / sc

                    # E8 quantize (head_dim=128, pad to multiple of 8 = already)
                    groups = normalized.reshape(-1, 8)
                    lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords = lp.reshape(-1, head_dim)
                    quantized = (coords * sc) @ H

                    # Coords for compression
                    int_coords = coords.detach().numpy()
                    has_half = np.any(np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25)
                    if has_half:
                        all_coords.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                    else:
                        all_coords.append(np.round(int_coords.flatten()).astype(np.int8))

                    # Write back
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    if is_key:
                        t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = result

                # Write modified KV back to GPU
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to(model.device), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to(model.device))

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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2  # fp16 per kept vector
        mask_bytes = math.ceil(prefix_len / 8) * n_layers * 2
        total = total_idx + scale_bytes + mask_bytes

        return {
            "fp16": total_fp16,
            "idx": total_idx,
            "scale": scale_bytes,
            "mask": mask_bytes,
            "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ---------- Baseline PPL ----------
    print("\nComputing baseline PPL...")
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_bl = prefix_out.past_key_values
        cont_out = model(full_ids[:, prefix_len:], past_key_values=kv_bl, use_cache=True)
        logits = cont_out.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ---------- Score importance ----------
    print("Scoring token importance...")
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_score = prefix_out.past_key_values
    importance = score_importance(kv_score)
    print(f"Importance: min={importance.min():.4f} max={importance.max():.4f}")

    # ---------- Configs ----------
    configs = [
        ("2b no-evict",       2, 100),
        ("3b no-evict",       3, 100),
        ("2b + 50% evict",    2,  50),
        ("3b + 50% evict",    3,  50),
        ("2b + 70% evict",    2,  30),
        ("3b + 70% evict",    3,  30),
        ("2b + 75% evict",    2,  25),
        ("3b + 75% evict",    3,  25),
        ("2b + 80% evict",    2,  20),
        ("3b + 80% evict",    3,  20),
    ]

    print(f"\n{'Config':<22s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} "
          f"{'Idx':>10s} {'Scale':>8s} {'Mask':>6s} {'Total':>10s}")
    print("-" * 100)

    results = []

    for name, bits, keep_pct in configs:
        torch.cuda.empty_cache()
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

        info, kv = evict_quantize(kv, keep_mask, bits)

        # PPL with attention mask
        evict_mask = ~keep_mask
        attn_ctx = torch.ones(prefix_len, dtype=torch.long, device=model.device)
        attn_ctx[evict_mask] = 0
        attn_full = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long, device=model.device)])
        attn_mask = attn_full.unsqueeze(0)

        with torch.no_grad():
            cont_out = model(full_ids[:, prefix_len:], past_key_values=kv,
                           attention_mask=attn_mask, use_cache=True)
            logits = cont_out.logits[:, :-1, :]
            targets = full_ids[:, prefix_len + 1:].contiguous()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppl = torch.exp(loss).item()

        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        elapsed = time.time() - t0

        print(f"{name:<22s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x "
              f"{info['n_kept']:5d} {info['idx']:>10,} {info['scale']:>8,} "
              f"{info['mask']:>6,} {info['total']:>10,}  ({elapsed:.1f}s)")

        results.append({"name": name, "bits": bits, "keep_pct": keep_pct,
                        "ppl": ppl, "delta": delta, **info})

    # Summary
    print(f"\n{'='*80}")
    print(f"MISTRAL-7B RESULTS — prefix={prefix_len}, cont={cont_len}, baseline={baseline_ppl:.4f}")
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

    # Compare with TinyLlama
    print(f"\n{'='*80}")
    print("COMPARISON: Mistral-7B vs TinyLlama-1.1B")
    print(f"{'='*80}")
    print("TinyLlama (head_dim=64): 3b+80%=21.5x@+1.82%, 2b+80%=29.0x@+3.19%")
    r_3b_80 = next((r for r in results if r['name'] == '3b + 80% evict'), None)
    r_2b_80 = next((r for r in results if r['name'] == '2b + 80% evict'), None)
    if r_3b_80:
        print(f"Mistral-7B (head_dim=128): 3b+80%={r_3b_80['ratio']:.1f}x@{r_3b_80['delta']:+.2f}%")
    if r_2b_80:
        print(f"Mistral-7B (head_dim=128): 2b+80%={r_2b_80['ratio']:.1f}x@{r_2b_80['delta']:+.2f}%")

    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return results


@app.local_entrypoint()
def main():
    print("Launching Mistral-7B experiment on Modal A10G GPU...")
    results = run_experiment.remote()

    # Write results locally
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            ".company", "engineering", "mistral_7b_results.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    import time
    with open(out_path, "w") as f:
        f.write("# Mistral-7B GPU Validation — Eviction + E8\n\n")
        f.write(f"**Model:** Mistral-7B-v0.1 (32 layers, 8 KV heads, head_dim=128)\n")
        f.write(f"**GPU:** A10G (24GB)\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")
        f.write("| Config | PPL | Delta% | Ratio | Kept | Idx | Scale | Mask | Total |\n")
        f.write("|--------|-----|--------|-------|------|-----|-------|------|-------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['ppl']:.4f} | {r['delta']:+.2f}% | "
                    f"{r['ratio']:.2f}x | {r['n_kept']} | {r['idx']:,} | "
                    f"{r['scale']:,} | {r['mask']:,} | {r['total']:,} |\n")
        f.write("\n## Honest Notes\n\n")
        f.write("- All overhead included: compressed E8 indices + fp16 scales + token mask bits\n")
        f.write("- Evicted tokens zeroed, attention mask excludes them during continuation\n")
        f.write("- FP16 model inference on GPU, E8 quantization on CPU\n")
        f.write("- Key-key attention scorer (SnapKV-style, causal mask)\n")
        f.write("- RoPE removed from keys before E8, reapplied after\n")
        f.write("- Per-head per-layer compression with temporal delta + zstd level 22\n")

    print(f"\nResults written to {out_path}")

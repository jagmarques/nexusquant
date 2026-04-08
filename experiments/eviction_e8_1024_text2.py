"""Second text validation — different passage order + different content.
Validates that 20x+ holds across texts. Quick run: only key configs.
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

# Completely different text: programming, geography, economics, music, philosophy
TEXT2 = """Python is a high-level general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms including structured, object-oriented, and functional programming. Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991. Python consistently ranks as one of the most popular programming languages. Python's standard library provides tools suited to many tasks including web frameworks, database connectors, text processing, and network protocols. The language supports modules and packages which encourages program modularity and code reuse. Python interpreters are available for many operating systems. CPython is the reference implementation written in C. Other implementations include PyPy which uses a just-in-time compiler for speed, Jython which runs on the Java virtual machine, and IronPython which targets the Common Language Runtime. Python's simple syntax and readability make it an excellent language for beginners while its powerful libraries make it equally suitable for complex scientific computing, machine learning, data analysis, and web development.
The Amazon River is the largest river in the world by discharge volume of water. It rises in the Andes mountains of Peru and flows eastward across the South American continent to empty into the Atlantic Ocean. The Amazon basin covers approximately 7 million square kilometers, making it the largest drainage basin in the world. The river system contains over 1,100 tributaries, of which 17 are longer than 1,500 kilometers. The Amazon rainforest, which the river passes through, is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers. It contains roughly 10 percent of all species on Earth and produces about 20 percent of the world's oxygen. Deforestation of the Amazon has accelerated in recent decades due to logging, cattle ranching, and soybean farming. The loss of tropical forest has significant implications for global climate patterns, biodiversity, and indigenous communities who depend on the forest for their livelihoods.
Modern economics traces its origins to Adam Smith's An Inquiry into the Nature and Causes of the Wealth of Nations, published in 1776. Smith argued that individuals pursuing their own self-interest through free markets would be guided by an invisible hand to promote the welfare of society as a whole. David Ricardo developed the theory of comparative advantage, showing that international trade benefits all countries even when one country can produce everything more efficiently than another. Karl Marx critiqued capitalism in Das Kapital, arguing that the extraction of surplus value from workers would lead to increasing inequality and eventual revolution. John Maynard Keynes revolutionized macroeconomics during the Great Depression, arguing that government spending could stabilize the economy during downturns. Milton Friedman and the Chicago School championed monetarism and free market policies. Modern economics has evolved to incorporate behavioral insights from psychology, network effects from technology platforms, and environmental externalities that classical models ignored. Central banks around the world use monetary policy tools including interest rates and quantitative easing to manage inflation and employment. The 2008 financial crisis revealed systematic risks in interconnected financial markets and led to new regulations aimed at preventing future crises.
Music theory is the study of the practices and possibilities of music. It describes the elements of music and includes both practical and scholarly traditions. Western music theory is founded on the division of the octave into twelve equally spaced semitones, known as equal temperament. This system was standardized during the Baroque period and enables modulation between any key. Harmony describes how chords are constructed and how they progress from one to another. Counterpoint is the relationship between voices that are harmonically interdependent yet independent in rhythm and contour. Johann Sebastian Bach's fugues remain the pinnacle of contrapuntal writing. Melody refers to the linear succession of musical tones that the listener perceives as a single entity. Rhythm describes the timing of sounds and silences in music. Time signatures organize beats into regular groups. Musical form provides the overall structure of a composition, from simple binary and ternary forms to complex sonata form and rondo. The twentieth century saw revolutionary developments including twelve-tone technique by Arnold Schoenberg, chance music by John Cage, and electronic music that fundamentally changed how sound is produced and organized.
Philosophy has been a central part of human intellectual life since ancient times. The pre-Socratic philosophers of ancient Greece were among the first to seek natural explanations for phenomena previously attributed to the gods. Socrates developed the dialectical method of inquiry, systematically questioning assumptions to arrive at truth. Plato proposed the theory of Forms, arguing that abstract ideals exist in a higher realm of reality. Aristotle established formal logic and made foundational contributions to ethics, politics, metaphysics, and natural science. Medieval philosophy was dominated by the integration of Christian theology with Greek philosophy, particularly by Thomas Aquinas. The Enlightenment brought renewed emphasis on reason and individual rights, with thinkers like John Locke, Voltaire, and Immanuel Kant challenging traditional authority. Kant's Critique of Pure Reason attempted to reconcile rationalism and empiricism. In the nineteenth century, existentialists like Kierkegaard and Nietzsche focused on individual existence and meaning. The twentieth century saw the development of analytic philosophy, which emphasizes logical analysis and clarity, and continental philosophy, which encompasses phenomenology, hermeneutics, and critical theory. Contemporary philosophy engages with questions raised by artificial intelligence, cognitive science, and bioethics."""


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {m}...")
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
    model.eval()
    return model, tok


def score_importance(kv, n_layers, n_heads, head_dim, obs_window=32):
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
    n_layers = len(kv.layers)
    levels = 2 ** bits
    H = hadamard_matrix(head_dim)
    n_kept = keep_mask.sum().item()
    total_fp16 = 0
    all_coords = []
    n_heads_local = kv.layers[0].keys.shape[1]

    for l in range(n_layers):
        layer = kv.layers[l]
        k = layer.keys.float()
        v = layer.values.float()
        total_fp16 += k.numel() * 2 + v.numel() * 2

        for is_key, tensor in [(True, k), (False, v)]:
            t = tensor[0].clone()
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

    scale_bytes = n_kept * n_layers * 2 * n_heads_local * 2
    mask_bytes = math.ceil(seq_len / 8) * n_layers * 2
    total = total_idx + scale_bytes + mask_bytes
    return {
        "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
        "mask": mask_bytes, "total": total, "ratio": total_fp16 / total if total > 0 else 0,
        "n_kept": n_kept, "n_total": seq_len,
    }


def main():
    model, tok = load_model()
    inputs = tok(TEXT2, return_tensors="pt", max_length=2048, truncation=True)
    full_ids = inputs.input_ids
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"Total tokens: {n_tok}, prefix: {prefix_len}, continuation: {cont_len}")

    n_layers = model.config.num_hidden_layers
    n_heads = 4; head_dim = 64
    rope_base = getattr(model.config, 'rope_theta', 10000.0)
    sliding_window = min(32, prefix_len)

    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_bl = prefix_out.past_key_values
        cont = model(full_ids[:, prefix_len:], past_key_values=kv_bl, use_cache=True)
        logits = cont.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_score = prefix_out.past_key_values
    importance = score_importance(kv_score, n_layers, n_heads, head_dim)

    # Only test key configs for speed
    configs = [
        ("2b no-evict",       2, 100),
        ("3b no-evict",       3, 100),
        ("2b + 75% evict",    2,  25),
        ("3b + 75% evict",    3,  25),
        ("2b + 80% evict",    2,  20),
        ("3b + 80% evict",    3,  20),
    ]

    print(f"{'Config':<22s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s}")
    print("-" * 60)

    for name, bits, keep_pct in configs:
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
        tag = " *** 20x+ ***" if info["ratio"] >= 20 else ""
        print(f"{name:<22s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x {info['n_kept']:5d}{tag}")

    print(f"\nBaseline: {baseline_ppl:.4f} | Text: programming/geography/economics/music/philosophy")


if __name__ == "__main__":
    main()

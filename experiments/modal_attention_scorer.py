"""Modal A100 experiment: key-key proxy vs real attention scorer on Mistral-7B.

Hypothesis: real accumulated softmax weights improve importance scoring and
save ~0.35pp PPL at 80% eviction vs the SnapKV-style key-key proxy.

Runs NexusQuantEvict at 35%, 60%, 80% eviction with BOTH scorers and reports
the per-eviction-rate PPL delta between them.

CRITICAL: real scorer requires attn_implementation='eager'. SDPA suppresses
attention weights silently regardless of output_attentions=True.
"""

import modal
import os

app = modal.App("nexusquant-attention-scorer")

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


@app.function(image=image, gpu="A100", timeout=1800, secrets=[HF_SECRET])
def run_scorer_comparison():
    import sys
    sys.path.insert(0, "/root")

    import time
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nexusquant.pipeline import NexusQuantEvict
    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    print("=" * 80)
    print("NEXUSQUANT — Key-Key vs Real Attention Scorer (Mistral-7B, A100)")
    print("=" * 80)

    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name} with attn_implementation='eager'...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

    # CRITICAL: eager is required to get real attention weights.
    # SDPA (the default) returns None for output_attentions regardless of the flag.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_layers = model.config.num_hidden_layers          # 32
    n_kv_heads = model.config.num_key_value_heads      # 8
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base = getattr(model.config, "rope_theta", 1000000.0)
    print(f"Config: {n_layers}L, {n_kv_heads} KV heads, head_dim={head_dim}, rope_theta={rope_base:.0f}")

    # Verify eager is active
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"attn_implementation: {attn_impl}")
    assert attn_impl == "eager", f"Expected 'eager', got {attn_impl!r}"

    # ------------------------------------------------------------------
    # Build ~3500-token prefix from diverse passages (same domain as
    # validated experiments to keep numbers comparable)
    # ------------------------------------------------------------------
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
        "The Roman Empire was the post-Republican period of ancient Roman civilization. It had a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around 100 BC to 400 AD. The Empire was among the most powerful economic, cultural, political and military forces in the world of its time. At its height under Trajan it covered five million square kilometers.",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data. Deep learning uses artificial neural networks with many layers to learn representations of data. Natural language processing enables computers to understand and generate human language.",
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of the Earth's lithosphere. Convergent boundaries where plates collide create mountain ranges and subduction zones. Divergent boundaries where plates move apart create mid-ocean ridges and rift valleys. Transform boundaries where plates slide past each other create fault lines and earthquakes.",
        "Music theory is the study of the practices and possibilities of music. Western music theory is founded on the division of the octave into twelve equally spaced semitones. Harmony describes how chords are constructed and progress. Counterpoint is the relationship between voices that are harmonically interdependent yet independent in rhythm. Johann Sebastian Bach's fugues remain the pinnacle of contrapuntal writing.",
    ]
    full_text = " ".join(passages)
    inputs = tok(full_text, return_tensors="pt", max_length=4096, truncation=True)
    full_ids = inputs.input_ids.to(model.device)
    n_tok = full_ids.shape[1]

    # Use ~half as prefix, rest as continuation for PPL measurement
    prefix_len = min(3500, n_tok // 2)
    cont_len = n_tok - prefix_len
    prefix_ids = full_ids[:, :prefix_len]
    print(f"\nTokens: total={n_tok}, prefix={prefix_len}, continuation={cont_len}")

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------
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

    def n_layers_cache(cache):
        if hasattr(cache, "key_cache"):
            return len(cache.key_cache)
        return len(cache.layers)

    # ------------------------------------------------------------------
    # E8 eviction + quantization (mirrors validated experiment logic)
    # ------------------------------------------------------------------
    def evict_and_quantize(kv_cache, keep_mask, bits=2):
        """Zero-out evicted tokens then E8-quantize survivors in-place."""
        levels = 2 ** bits
        H = hadamard_matrix(head_dim).to(model.device).float()
        n_l = n_layers_cache(kv_cache)

        for l in range(n_l):
            k, v = get_kv(kv_cache, l)
            k = k.float()
            v = v.float()
            seq = k.shape[2]

            mask_4d = keep_mask.unsqueeze(1).unsqueeze(-1).float().to(k.device)  # (b,1,seq,1)
            k = k * mask_4d
            v = v * mask_4d

            b, h = k.shape[0], k.shape[1]

            # Keys: RoPE-rm -> Hadamard -> E8 -> inv-Hadamard -> re-RoPE
            k_out_batches = []
            for bi in range(b):
                k_bi = k[bi]                                                     # (h, seq, d)
                k_nr = inverse_rope(k_bi, base=rope_base)
                k_rot = torch.einsum("hsd,de->hse", k_nr, H)
                k_flat = k_rot.reshape(-1, head_dim)
                k_q = E8Lattice.quantize_perhead(k_flat, levels=levels)
                k_back = torch.einsum("hsd,ed->hse", k_q.reshape(h, seq, head_dim), H)
                k_roped = forward_rope(k_back, base=rope_base)
                k_out_batches.append(k_roped)
            k_out = torch.stack(k_out_batches, dim=0).half()

            # Values: Hadamard -> E8 -> inv-Hadamard
            v_rot = torch.einsum("bhsd,de->bhse", v, H)
            v_flat = v_rot.reshape(-1, head_dim)
            v_q = E8Lattice.quantize_perhead(v_flat, levels=levels)
            v_out = torch.einsum("bhsd,ed->bhse", v_q.reshape(b, h, seq, head_dim), H).half()

            # Re-zero evicted positions (quantizer may shift zero slightly)
            k_out = k_out * mask_4d.half()
            v_out = v_out * mask_4d.half()

            set_kv(kv_cache, l, k_out, v_out)

        return kv_cache

    def compute_ppl(kv_cache, keep_mask):
        """Continuation PPL given a compressed/evicted kv_cache."""
        attn_ctx = keep_mask.long().to(model.device)           # (1, prefix_len)
        attn_full = torch.cat([
            attn_ctx,
            torch.ones(1, cont_len, dtype=torch.long, device=model.device),
        ], dim=1)

        with torch.no_grad():
            out = model(
                full_ids[:, prefix_len:],
                past_key_values=kv_cache,
                attention_mask=attn_full,
                use_cache=True,
            )
        logits = out.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return torch.exp(loss).item()

    # ------------------------------------------------------------------
    # Baseline PPL (no compression)
    # ------------------------------------------------------------------
    print("\nComputing baseline PPL (no compression)...")
    with torch.no_grad():
        bl_out = model(prefix_ids, use_cache=True)
        kv_bl = bl_out.past_key_values
        bl_full_mask = torch.ones(1, n_tok, dtype=torch.long, device=model.device)
        bl_cont_out = model(
            full_ids[:, prefix_len:],
            past_key_values=kv_bl,
            attention_mask=bl_full_mask,
            use_cache=True,
        )
    bl_logits = bl_cont_out.logits[:, :-1, :]
    bl_targets = full_ids[:, prefix_len + 1:].contiguous()
    bl_loss = F.cross_entropy(bl_logits.reshape(-1, bl_logits.shape[-1]), bl_targets.reshape(-1))
    baseline_ppl = torch.exp(bl_loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ------------------------------------------------------------------
    # Importance scorers
    # ------------------------------------------------------------------
    def score_key_key(kv_cache, obs_window=32):
        """SnapKV-style key-key proxy scorer (layer 0 only, matches NexusQuantEvict)."""
        k0, _ = get_kv(kv_cache, 0)
        b, h, seq, d = k0.shape
        scale = d ** -0.5
        obs = min(obs_window, seq)
        q = k0[:, :, -obs:, :].float()
        k = k0.float()
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        return attn.sum(dim=2).mean(dim=1)   # (b, seq)

    def score_real(prefix_ids):
        """Real attention scorer: accumulated softmax weights from all layers."""
        nq = NexusQuantEvict(
            head_dim=head_dim,
            bits=2,
            eviction_rate=0.0,  # scorer only — no actual eviction here
            rope_base=rope_base,
            scorer="real",
        )
        return nq._score_importance_real(model, prefix_ids)   # (b, seq)

    # ------------------------------------------------------------------
    # Run both scorers, then evict + quantize at each rate, measure PPL
    # ------------------------------------------------------------------
    eviction_rates = [0.35, 0.60, 0.80]
    bits = 2
    sliding_window = 32

    print("\nPre-computing real attention scores (one extra forward pass)...")
    t_real = time.time()
    real_importance = score_real(prefix_ids)   # (1, prefix_len)
    print(f"Real scorer done in {time.time() - t_real:.1f}s  "
          f"min={real_importance.min():.4f} max={real_importance.max():.4f}")

    print("\nPre-computing key-key proxy scores...")
    with torch.no_grad():
        pf_out = model(prefix_ids, use_cache=True)
        kv_for_kk = pf_out.past_key_values
    kk_importance = score_key_key(kv_for_kk)
    print(f"Key-key scorer done  "
          f"min={kk_importance.min():.4f} max={kk_importance.max():.4f}")

    def build_keep_mask(importance, eviction_rate):
        """BOS + top-(1-eviction_rate) + sliding window."""
        b, seq = importance.shape
        device = importance.device
        prefix_end = max(seq - sliding_window, 0)
        n_keep = max(1, int(prefix_end * (1.0 - eviction_rate)))

        mask = torch.zeros(b, seq, dtype=torch.bool, device=device)
        mask[:, 0] = True                                         # BOS always kept
        if sliding_window > 0:
            mask[:, max(0, seq - sliding_window):] = True         # sliding window

        if prefix_end > 1 and n_keep > 0:
            prefix_scores = importance[:, 1:prefix_end]           # exclude BOS
            topk_k = min(n_keep, prefix_scores.shape[1])
            if topk_k > 0:
                _, idx = torch.topk(prefix_scores, k=topk_k, dim=1)
                mask.scatter_(1, idx + 1, True)                   # +1: offset for BOS
        return mask

    print(f"\n{'Scorer':<10} {'Evict%':>7} {'PPL':>9} {'Delta%':>8}  {'Kept':>6}")
    print("-" * 48)

    results = []

    for scorer_name, importance in [("key-key", kk_importance), ("real", real_importance)]:
        for eviction_rate in eviction_rates:
            torch.cuda.empty_cache()

            keep_mask = build_keep_mask(importance, eviction_rate)
            n_kept = keep_mask.sum(dim=1).item()

            # Fresh prefill for each run
            with torch.no_grad():
                pf_out = model(prefix_ids, use_cache=True)
                kv = pf_out.past_key_values

            kv = evict_and_quantize(kv, keep_mask, bits=bits)
            ppl = compute_ppl(kv, keep_mask)
            delta = (ppl - baseline_ppl) / baseline_ppl * 100

            print(f"{scorer_name:<10} {eviction_rate:>6.0%}  {ppl:>9.4f} {delta:>+7.2f}%  {n_kept:>6}")
            results.append({
                "scorer": scorer_name,
                "eviction_rate": eviction_rate,
                "ppl": ppl,
                "delta": delta,
                "n_kept": n_kept,
            })

    # ------------------------------------------------------------------
    # Summary: real vs key-key diff at each eviction rate
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SCORER COMPARISON — Mistral-7B, prefix={prefix_len}, baseline={baseline_ppl:.4f}")
    print(f"{'=' * 60}")
    print(f"{'Evict%':>7}  {'key-key PPL':>12}  {'real PPL':>10}  {'real-kk diff':>14}")
    print("-" * 52)

    for eviction_rate in eviction_rates:
        kk_r = next(r for r in results if r["scorer"] == "key-key" and r["eviction_rate"] == eviction_rate)
        re_r = next(r for r in results if r["scorer"] == "real"    and r["eviction_rate"] == eviction_rate)
        diff = re_r["ppl"] - kk_r["ppl"]
        sign = "BETTER" if diff < 0 else "worse"
        print(f"{eviction_rate:>6.0%}   {kk_r['ppl']:>10.4f} ({kk_r['delta']:+.2f}%)"
              f"  {re_r['ppl']:>8.4f} ({re_r['delta']:+.2f}%)"
              f"  {diff:+.4f} pp  [{sign}]")

    print(f"\nHypothesis: real scorer saves ~0.35pp at 80% eviction.")
    print(f"Actual 80% diff: {next(r['ppl'] for r in results if r['scorer']=='real' and r['eviction_rate']==0.80) - next(r['ppl'] for r in results if r['scorer']=='key-key' and r['eviction_rate']==0.80):+.4f} pp")

    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return results


@app.local_entrypoint()
def main():
    print("Launching scorer comparison on Modal A100...")
    results = run_scorer_comparison.remote()

    print("\n=== LOCAL SUMMARY ===")
    baseline_ppl = None
    for r in results:
        print(f"  scorer={r['scorer']:<8} eviction={r['eviction_rate']:.0%}"
              f"  PPL={r['ppl']:.4f}  delta={r['delta']:+.2f}%  kept={r['n_kept']}")

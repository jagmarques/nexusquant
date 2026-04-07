"""Simplification test: 6-stage vs 4-stage pipeline on Mistral-7B.

6-stage: Score → Evict → RoPE-rm → Hadamard → E8 VQ → Delta+zstd
4-stage: Score → Evict → Hadamard → E8 VQ  (no RoPE removal, no delta+zstd)

KEY QUESTIONS:
  1. Does removing RoPE removal hurt PPL by more than 0.2pp?
  2. What is the exact ratio hit from dropping delta+zstd?

Test: 0 / 35 / 60 / 80% eviction on a fixed 3544-token multi-topic text.
Ratio in Config B (4-stage) = fp16_total / (raw_int8_idx_bytes + scale_bytes + mask_bytes)
  where raw_int8_idx_bytes = n_kept * n_layers * 2 * n_kv_heads * head_dim * 1

Results written to .company/engineering/simplify_test_results.md
"""
import modal
import os

app = modal.App("nexusquant-simplify-test")

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

# ======================================================================
# 3544-TOKEN MULTI-TOPIC TEXT
# Diverse content so context matters and compression effects are visible.
# ======================================================================
TEXT_3K = (
    # Physics (~300 words)
    "The Standard Model of particle physics describes three of the four fundamental forces and "
    "classifies all known elementary particles. Fermions — six quarks and six leptons — are the "
    "building blocks of matter. Gauge bosons mediate the forces between fermions. The Higgs "
    "mechanism gives mass to particles through spontaneous symmetry breaking. The strong force "
    "binds quarks into protons and neutrons via gluon exchange; quantum chromodynamics is its "
    "governing theory. The electroweak force unifies electromagnetism and the weak nuclear force, "
    "with the W and Z bosons as mediators. General relativity describes gravity as spacetime "
    "curvature caused by mass and energy. Quantum mechanics governs the microscopic world with "
    "wave functions, operators, and the uncertainty principle. Dark matter and dark energy "
    "together constitute roughly 95 percent of the universe's energy content but have not been "
    "directly detected. The Large Hadron Collider confirmed the Higgs boson in 2012, completing "
    "the Standard Model. Hawking radiation predicts that black holes slowly evaporate through "
    "quantum effects near the event horizon, linking gravity, thermodynamics, and quantum "
    "mechanics. The arrow of time, entropy, and the second law of thermodynamics pose deep "
    "puzzles about the initial conditions of the universe. Supersymmetry and string theory are "
    "candidate extensions of the Standard Model, but neither has found experimental confirmation. "
    # History (~300 words)
    "The Industrial Revolution transformed human society from agrarian economies to machine-based "
    "manufacturing beginning in Britain in the late eighteenth century. James Watt's improved "
    "steam engine became the universal power source; railways linked cities and created national "
    "markets. The factory system concentrated labor and created new urban working classes. The "
    "Second Industrial Revolution brought electricity, steel, chemicals, and mass production. "
    "The Roman Empire unified the Mediterranean world under a single legal and administrative "
    "system for five centuries. Roman law, Latin language, engineering, and infrastructure "
    "shaped European civilization profoundly. The Renaissance revived classical learning and "
    "produced figures like Leonardo, Michelangelo, and Galileo. The printing press democratized "
    "knowledge and accelerated the Reformation. The French Revolution proclaimed the rights of "
    "man and launched an era of nationalism. Napoleon spread the Napoleonic Code across Europe. "
    "The World Wars caused unprecedented destruction and reshaped the global order, ending "
    "European colonial empires and initiating the Cold War between the United States and the "
    "Soviet Union. Decolonization created dozens of new nations with complex legacies that "
    "persist to the present. The fall of the Berlin Wall in 1989 marked the end of the Cold War "
    "and the beginning of an era of globalization. The internet revolutionized communication, "
    "commerce, and culture in ways that continue to accelerate. "
    # Biology (~300 words)
    "Evolution by natural selection, proposed by Darwin and Wallace, is the unifying theory of "
    "biology. Heritable variation plus differential reproduction causes populations to adapt to "
    "their environments over generations. The modern evolutionary synthesis combines Darwinian "
    "selection with Mendelian genetics and population genetics. DNA carries genetic information "
    "in a double-helical structure discovered by Watson and Crick in 1953. The genetic code "
    "maps three-nucleotide codons to the twenty standard amino acids and is nearly universal "
    "across life. The human genome contains about three billion base pairs encoding roughly "
    "twenty thousand protein-coding genes. CRISPR-Cas9 allows precise genome editing with "
    "revolutionary implications for medicine and agriculture. Cell biology distinguishes "
    "prokaryotes from eukaryotes, which have membrane-bound organelles including nuclei, "
    "mitochondria, and chloroplasts. The immune system provides innate and adaptive defenses "
    "against pathogens. T lymphocytes mediate cellular immunity; B lymphocytes produce "
    "antibodies. Vaccines train adaptive immunity without causing disease. Ecosystems are "
    "communities of organisms interacting with each other and their abiotic environment. "
    "Biodiversity is threatened by habitat destruction, climate change, and pollution. "
    # Computer Science (~300 words)
    "The transformer architecture uses self-attention to process sequences in parallel and "
    "has become the foundation for most modern large language models. Self-attention computes "
    "pairwise interactions between all tokens in a sequence, enabling the model to capture "
    "long-range dependencies. The key-value cache stores intermediate computations during "
    "autoregressive generation, growing linearly with context length and becoming a memory "
    "bottleneck at long contexts. Quantization reduces model precision from 32-bit or 16-bit "
    "floating point to lower bit-widths, trading accuracy for memory and speed. Lattice vector "
    "quantization, such as the E8 lattice, provides theoretically optimal quantization for "
    "high-dimensional vectors by packing sphere of influence maximally efficiently. Eviction "
    "policies decide which KV cache tokens to drop when memory is constrained. Attention-based "
    "importance scoring selects the most important tokens by measuring how much attention the "
    "recent context attends to each historical token. RoPE rotary position embeddings encode "
    "position information directly into key and query vectors via rotation matrices; removing "
    "and re-applying RoPE before and after quantization aligns vectors in a position-free space "
    "for better compression. The Hadamard transform rotates vectors into a basis where energy "
    "is more uniformly distributed, improving quantization efficiency. Distributed systems "
    "coordinate multiple computers to appear as a single coherent system, facing challenges of "
    "consensus, fault tolerance, and consistency. The P versus NP problem is one of the most "
    "important open problems in mathematics and theoretical computer science. "
    # Mathematics (~300 words)
    "Pure mathematics discovers patterns and structures through rigorous proof. Number theory "
    "studies integers and primes; the Riemann hypothesis about the zeros of the zeta function "
    "is the most famous unsolved problem. Abstract algebra studies groups, rings, fields, and "
    "modules. Group theory underlies the symmetries of physical laws; Galois theory uses groups "
    "to determine which polynomial equations are solvable by radicals. Topology studies "
    "properties preserved under continuous deformations. The Poincare conjecture, proved by "
    "Perelman in 2003, characterizes the three-sphere among compact three-manifolds. Differential "
    "geometry describes curved spaces using calculus; it is the mathematical language of general "
    "relativity. Algebraic geometry studies zero sets of polynomial equations and connects deep "
    "arithmetic questions to geometric ones. Category theory provides a unifying language for "
    "mathematics, abstracting structural relationships between different fields. Probability "
    "theory gives a rigorous foundation for reasoning under uncertainty; Bayesian inference "
    "updates beliefs in light of evidence. The central limit theorem explains the ubiquity of "
    "the normal distribution as the sum of many independent random variables. Functional analysis "
    "studies infinite-dimensional vector spaces and operators between them, providing the "
    "mathematical framework for quantum mechanics. Combinatorics counts structures, while graph "
    "theory models networks; together they underpin computer science and statistical physics. "
    "Information theory, founded by Shannon, quantifies information content and channel capacity, "
    "establishing fundamental limits on lossless and lossy compression. "
    # Economics (~250 words)
    "Economics studies how individuals, firms, and governments allocate scarce resources. "
    "Microeconomics models how prices coordinate supply and demand in markets. Consumer choice "
    "theory maximizes utility subject to budget constraints. Game theory studies strategic "
    "interaction: the Nash equilibrium is a profile of strategies from which no player has "
    "unilateral incentive to deviate. Mechanism design asks how to construct rules that induce "
    "desired outcomes from self-interested agents. Macroeconomics examines aggregate variables: "
    "output, employment, inflation, and growth. Keynesian theory holds that aggregate demand "
    "drives output in the short run and justifies fiscal stimulus in recessions. Growth theory "
    "studies why incomes differ across countries and grow over time. Behavioral economics "
    "incorporates psychological evidence that humans systematically deviate from the "
    "rational-agent model through heuristics and biases. Financial economics studies asset "
    "pricing, risk, and the role of intermediaries in allocating capital. "
    # Medicine (~250 words)
    "Medicine encompasses the science and practice of diagnosing, treating, and preventing "
    "disease. The germ theory showed that specific microorganisms cause specific infectious "
    "diseases. Vaccination harnesses the immune system to prevent disease; the eradication "
    "of smallpox is the greatest achievement of public health. Antibiotics transformed "
    "bacterial infections from leading killers to treatable conditions, but antibiotic "
    "resistance now threatens to reverse these gains. The Human Genome Project produced a "
    "reference sequence for the entire human genome, accelerating the identification of "
    "disease genes and drug targets. CRISPR gene therapy offers the prospect of correcting "
    "genetic defects directly. Medical imaging allows non-invasive visualization of anatomy "
    "and function. Evidence-based medicine synthesizes clinical trial data through systematic "
    "reviews and meta-analyses to guide treatment. The COVID-19 pandemic demonstrated both "
    "the power of rapid mRNA vaccine development and the challenges of global health. "
)


# ======================================================================
# MAIN FUNCTION
# ======================================================================
@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    secrets=[HF_SECRET],
    memory=65536,
)
def run_simplify_test():
    import sys
    sys.path.insert(0, "/root")

    import math, time
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
    print("NEXUSQUANT — Simplification Test: 6-stage vs 4-stage")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    n_layers   = model.config.num_hidden_layers           # 32
    n_kv_heads = model.config.num_key_value_heads         # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    SLIDING_WINDOW = 32

    # ------------------------------------------------------------------ importance scorer
    def score_importance(kv_cache, prefix_len):
        obs_window = max(32, prefix_len // 16)
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()   # [n_kv_heads, seq_len, head_dim]
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal  = (all_pos <= obs_pos)
            scores  = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn    = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                          padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-SLIDING_WINDOW:] = True
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), SLIDING_WINDOW + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ 6-stage compress
    # Score → Evict → RoPE-rm → Hadamard → E8 VQ → Delta+zstd
    def compress_6stage(kv_cache, keep_mask, prefix_len):
        """Full pipeline. Returns (info_dict, modified_kv_cache)."""
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
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor, coord_list in [
                (True,  k, all_key_coords),
                (False, v, all_val_coords),
            ]:
                levels = 4  # 2-bit => levels=4
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    if is_key:
                        # Stage 3: inverse RoPE
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    kept_data = t_head[keep_mask]
                    # Stage 4: Hadamard
                    rotated   = kept_data @ H.T
                    amax      = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc        = amax / (levels / 2)
                    normalized = rotated / sc
                    # Stage 5: E8 VQ
                    groups    = normalized.reshape(-1, 8)
                    lp        = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords    = lp.reshape(-1, head_dim)
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
                        # Re-apply RoPE after quantization
                        t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = result
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        # Stage 6: temporal delta + zstd
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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2   # fp16 scales
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16,
            "idx": total_idx,
            "scale": scale_bytes,
            "mask": mask_bytes,
            "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
            "config": "6-stage",
        }, kv_cache

    # ------------------------------------------------------------------ 4-stage compress
    # Score → Evict → Hadamard → E8 VQ  (no RoPE removal, no delta+zstd)
    def compress_4stage(kv_cache, keep_mask, prefix_len):
        """Simplified pipeline. Returns (info_dict, modified_kv_cache).
        Ratio = fp16_total / (raw_int8_idx_bytes + scale_bytes + mask_bytes)
        raw_int8_idx_bytes = n_kept * n_layers * 2 * n_kv_heads * head_dim * 1
        """
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor in [(True, k), (False, v)]:
                levels = 4  # 2-bit
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    # No RoPE removal — use tensor as-is
                    t_head = t[h]
                    kept_data = t_head[keep_mask]
                    # Stage 3: Hadamard
                    rotated   = kept_data @ H.T
                    amax      = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc        = amax / (levels / 2)
                    normalized = rotated / sc
                    # Stage 4: E8 VQ
                    groups    = normalized.reshape(-1, 8)
                    lp        = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords    = lp.reshape(-1, head_dim)
                    quantized = (coords * sc) @ H
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    # No forward_rope — leave as-is
                    t[h] = result
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        # Ratio: raw int8 coords (no delta, no zstd)
        # n_kept tokens * n_layers * 2 (K+V) * n_kv_heads * head_dim coords * 1 byte/coord
        raw_idx_bytes = n_kept * n_layers * 2 * n_kv_heads * head_dim * 1
        scale_bytes   = n_kept * n_layers * 2 * n_kv_heads * 2   # fp16 per-token-per-head scale
        mask_bytes    = math.ceil(prefix_len / 8) * n_layers
        total = raw_idx_bytes + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16,
            "idx": raw_idx_bytes,
            "scale": scale_bytes,
            "mask": mask_bytes,
            "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
            "config": "4-stage",
        }, kv_cache

    # ======================================================================
    # TOKENIZE — target ~3544 tokens
    # ======================================================================
    print("\nTokenizing text (targeting ~3544 tokens)...")
    inputs = tok(TEXT_3K, return_tensors="pt", max_length=4096, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok = full_ids.shape[1]
    print(f"Tokenized: {n_tok} tokens total")

    # Split: use ~2/3 as prefix, rest as continuation for PPL measurement
    prefix_len = min(2400, n_tok * 2 // 3)
    cont_len   = n_tok - prefix_len
    print(f"Prefix: {prefix_len} tokens | Continuation: {cont_len} tokens")

    if cont_len < 100:
        raise RuntimeError(f"Continuation too short ({cont_len}). Need at least 100 tokens.")

    print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

    # ======================================================================
    # BASELINE PPL
    # ======================================================================
    print("\nComputing baseline PPL...")
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
        cout = model(full_ids[:, prefix_len:], past_key_values=pout.past_key_values, use_cache=True)
        logits  = cout.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:]
        loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # Pre-compute importance (shared across all experiments)
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
    importance = score_importance(pout.past_key_values, prefix_len)

    # ======================================================================
    # SWEEP: both pipelines x 4 eviction levels
    # ======================================================================
    evict_levels = [0, 35, 60, 80]
    configs = [
        ("6-stage", compress_6stage),
        ("4-stage", compress_4stage),
    ]

    print(f"\n{'Config':<10s} {'Evict%':>7s} {'PPL':>9s} {'PPL Δ%':>9s} {'Ratio':>8s} {'Kept':>6s}")
    print("-" * 60)

    all_results = []

    for evict_pct in evict_levels:
        keep_mask = build_keep_mask(prefix_len, evict_pct, importance)

        for cfg_name, compress_fn in configs:
            torch.cuda.empty_cache()

            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv   = pout.past_key_values

            info, kv = compress_fn(kv, keep_mask, prefix_len)

            # Build attention mask: zeros for evicted positions
            evict_mask = ~keep_mask
            attn_ctx   = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full  = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long, device="cuda")])

            with torch.no_grad():
                cout = model(
                    full_ids[:, prefix_len:],
                    past_key_values=kv,
                    attention_mask=attn_full.unsqueeze(0),
                    use_cache=True,
                )
                logits  = cout.logits[:, :-1, :].float()
                targets = full_ids[:, prefix_len + 1:]
                loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl     = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            print(f"{cfg_name:<10s} {evict_pct:>6d}% {ppl:>9.4f} {delta:>+8.2f}% {info['ratio']:>7.2f}x {info['n_kept']:>5d}")

            all_results.append({
                "config":    cfg_name,
                "evict_pct": evict_pct,
                "ppl":       ppl,
                "delta":     delta,
                "ratio":     info["ratio"],
                "n_kept":    info["n_kept"],
                "fp16":      info["fp16"],
                "idx_bytes": info["idx"],
                "scale_bytes": info["scale"],
                "mask_bytes":  info["mask"],
                "total_bytes": info["total"],
            })

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ======================================================================
    # ANALYSIS: compare 6-stage vs 4-stage at each eviction level
    # ======================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS: 6-stage vs 4-stage at each eviction level")
    print("=" * 80)
    print(f"{'Evict%':>7s} {'6st PPL Δ%':>12s} {'4st PPL Δ%':>12s} {'Δ between':>11s} "
          f"{'6st Ratio':>10s} {'4st Ratio':>10s} {'Ratio hit':>10s}")
    print("-" * 80)

    analysis = []
    for ep in evict_levels:
        r6 = next(r for r in all_results if r['config'] == '6-stage' and r['evict_pct'] == ep)
        r4 = next(r for r in all_results if r['config'] == '4-stage' and r['evict_pct'] == ep)
        ppl_diff = r4['delta'] - r6['delta']   # positive = 4-stage is worse
        ratio_hit = r6['ratio'] / r4['ratio']  # >1 means 6-stage has better ratio
        verdict = "OK (<=0.2pp)" if abs(ppl_diff) <= 0.2 else "WORSE (>0.2pp)"
        print(f"{ep:>6d}% {r6['delta']:>+11.3f}% {r4['delta']:>+11.3f}% {ppl_diff:>+10.3f}pp "
              f"{r6['ratio']:>9.2f}x {r4['ratio']:>9.2f}x {ratio_hit:>9.2f}x | {verdict}")
        analysis.append({
            "evict_pct": ep,
            "delta_6st": r6['delta'],
            "delta_4st": r4['delta'],
            "ppl_diff": ppl_diff,
            "ratio_6st": r6['ratio'],
            "ratio_4st": r4['ratio'],
            "ratio_hit": ratio_hit,
            "verdict": verdict,
        })

    # Overall verdict
    max_ppl_diff = max(abs(a['ppl_diff']) for a in analysis)
    keep_rope = max_ppl_diff > 0.2
    rope_verdict = "KEEP RoPE removal" if keep_rope else "DROP RoPE removal (safe)"
    print(f"\nMax PPL diff across all eviction levels: {max_ppl_diff:.3f}pp")
    print(f"RoPE verdict: {rope_verdict}")
    print(f"Ratio impact of dropping delta+zstd: ~{analysis[0]['ratio_hit']:.2f}x at 0% evict")

    return all_results, analysis, baseline_ppl, prefix_len, n_tok


# ======================================================================
# LOCAL ENTRYPOINT
# ======================================================================
@app.local_entrypoint()
def main():
    import time

    print("\n" + "=" * 80)
    print("NEXUSQUANT: Simplification Test — 6-stage vs 4-stage")
    print("=" * 80)
    print("Launching on A100 GPU...")

    all_results, analysis, baseline_ppl, prefix_len, n_tok = run_simplify_test.remote()

    # ------------------------------------------------------------------ Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model: Mistral-7B-v0.1 | A100 | prefix={prefix_len} | n_tok={n_tok}")
    print(f"Baseline PPL: {baseline_ppl:.4f}\n")

    # Raw results table
    print(f"{'Config':<10s} {'Evict%':>7s} {'PPL Δ%':>9s} {'Ratio':>8s} {'Kept':>6s}")
    print("-" * 46)
    for ep in [0, 35, 60, 80]:
        for cfg in ["6-stage", "4-stage"]:
            r = next(x for x in all_results if x['config'] == cfg and x['evict_pct'] == ep)
            print(f"{cfg:<10s} {ep:>6d}% {r['delta']:>+8.3f}% {r['ratio']:>7.2f}x {r['n_kept']:>5d}")
        print()

    # Analysis table
    print(f"\n{'Evict%':>7s} {'PPL diff (4-6)':>15s} {'6st Ratio':>10s} {'4st Ratio':>10s} {'Ratio hit':>10s} Verdict")
    print("-" * 70)
    for a in analysis:
        print(f"{a['evict_pct']:>6d}% {a['ppl_diff']:>+14.3f}pp {a['ratio_6st']:>9.2f}x "
              f"{a['ratio_4st']:>9.2f}x {a['ratio_hit']:>9.2f}x  {a['verdict']}")

    max_ppl_diff = max(abs(a['ppl_diff']) for a in analysis)
    keep_rope    = max_ppl_diff > 0.2
    rope_verdict = "KEEP RoPE removal (hurts quality)" if keep_rope else "SAFE to DROP RoPE removal"
    avg_ratio_hit = sum(a['ratio_hit'] for a in analysis) / len(analysis)

    print(f"\nMax PPL diff: {max_ppl_diff:.3f}pp")
    print(f"RoPE verdict: {rope_verdict}")
    print(f"Avg ratio hit from dropping delta+zstd: {avg_ratio_hit:.2f}x")
    print(f"Expected ratio hit at 0% evict: {analysis[0]['ratio_hit']:.2f}x  "
          f"(raw int8 ~1 byte/coord vs ~0.4 bytes after zstd)")

    # ------------------------------------------------------------------ Write results file
    out_dir  = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simplify_test_results.md")

    with open(out_path, "w") as f:
        f.write("# Simplification Test: 6-stage vs 4-stage Pipeline\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**GPU:** A100 (40 GB)\n")
        f.write("**Model:** mistralai/Mistral-7B-v0.1 (32L, 8 KV heads, d=128, rope_theta=10000)\n")
        f.write(f"**Tokens:** total={n_tok} | prefix={prefix_len} | "
                f"continuation={n_tok - prefix_len}\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n\n")
        f.write("---\n\n")

        f.write("## Pipeline Definitions\n\n")
        f.write("**6-stage (current baseline):**  \n")
        f.write("Score → Evict → RoPE removal (inverse_rope/forward_rope) → "
                "Hadamard → E8 2-bit VQ → temporal delta + zstd-22\n\n")
        f.write("**4-stage (proposed simplified):**  \n")
        f.write("Score → Evict → Hadamard → E8 2-bit VQ  "
                "(no RoPE removal, no delta+zstd — ratio uses raw int8 index bytes)\n\n")
        f.write("Ratio formula for 4-stage:  \n")
        f.write("`ratio = fp16_total / (n_kept * n_layers * 2 * n_kv_heads * head_dim * 1 "
                "+ scale_bytes + mask_bytes)`\n\n")
        f.write("---\n\n")

        f.write("## Raw Results\n\n")
        f.write("| Config | Evict% | PPL | PPL Δ% | Ratio | Kept tokens |\n")
        f.write("|--------|--------|-----|--------|-------|-------------|\n")
        for ep in [0, 35, 60, 80]:
            for cfg in ["6-stage", "4-stage"]:
                r = next(x for x in all_results if x['config'] == cfg and x['evict_pct'] == ep)
                f.write(f"| {cfg} | {ep}% | {r['ppl']:.4f} | {r['delta']:+.3f}% | "
                        f"{r['ratio']:.2f}x | {r['n_kept']} |\n")

        f.write("\n---\n\n")

        f.write("## Analysis: Head-to-Head Comparison\n\n")
        f.write("| Evict% | 6-stage Δ% | 4-stage Δ% | PPL diff (4-6) | 6-stage Ratio | "
                "4-stage Ratio | Ratio hit | Verdict |\n")
        f.write("|--------|-----------|-----------|----------------|---------------|"
                "--------------|-----------|--------|\n")
        for a in analysis:
            f.write(f"| {a['evict_pct']}% | {a['delta_6st']:+.3f}% | {a['delta_4st']:+.3f}% | "
                    f"{a['ppl_diff']:+.3f}pp | {a['ratio_6st']:.2f}x | {a['ratio_4st']:.2f}x | "
                    f"{a['ratio_hit']:.2f}x | {a['verdict']} |\n")

        f.write(f"\n**Max PPL diff:** {max_ppl_diff:.3f}pp  \n")
        f.write(f"**Threshold:** 0.2pp  \n")
        f.write(f"**RoPE verdict:** {rope_verdict}  \n")
        f.write(f"**Average ratio hit from dropping delta+zstd:** {avg_ratio_hit:.2f}x  \n\n")

        f.write("---\n\n")

        f.write("## Decision Table\n\n")
        f.write("| Component | Keep or Drop? | Reason |\n")
        f.write("|-----------|--------------|--------|\n")
        rope_decision = "KEEP" if keep_rope else "DROP"
        rope_reason = (f"PPL diff >{0.2}pp across eviction levels" if keep_rope
                       else f"PPL diff <={0.2}pp — elegance wins")
        f.write(f"| RoPE removal | {rope_decision} | {rope_reason} |\n")
        f.write(f"| Delta+zstd | Depends on ratio target | "
                f"Adds ~{avg_ratio_hit:.2f}x ratio; adds CPU overhead |\n")

        f.write("\n---\n\n")

        f.write("## Reference Table (from task brief)\n\n")
        f.write("| Config | Evict% | PPL Δ% | Ratio | Notes |\n")
        f.write("|--------|--------|--------|-------|-------|\n")
        for ep in [0, 35, 60, 80]:
            for cfg in ["6-stage", "4-stage"]:
                r = next(x for x in all_results if x['config'] == cfg and x['evict_pct'] == ep)
                notes = "current baseline" if cfg == "6-stage" else "simplified"
                f.write(f"| {cfg} | {ep}% | {r['delta']:+.3f}% | {r['ratio']:.2f}x | {notes} |\n")

    print(f"\nResults written to: {out_path}")
    print("Done.")

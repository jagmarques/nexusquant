"""Modal GPU experiment: Mistral-7B validation rerun + random eviction comparison.

Reproduces original 2b no-evict / 2b+70% / 2b+80% results on NEW text,
and adds random eviction baselines to prove importance scoring matters.

Original results (text set 1):
  2b no-evict  : 6.72x  at +0.36%
  2b+70% evict : 21.58x at +0.43%
  2b+80% evict : 31.99x at +0.94%

Reproducibility target: within ±0.5% PPL delta of originals.
Importance vs random: importance-based eviction should beat random by ≥0.5% PPL.
"""
import modal
import os

app = modal.App("nexusquant-mistral-validation-v2")

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
    timeout=1800,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_experiment():
    import sys
    sys.path.insert(0, "/root")

    import time
    import math
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    # ------------------------------------------------------------------ #
    # DynamicCache API helpers — compatible with transformers 4.44–4.99   #
    # ------------------------------------------------------------------ #
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
    print("NEXUSQUANT — Mistral-7B Validation Rerun + Random Eviction Comparison")
    print("=" * 80)

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,          # avoid deprecated torch_dtype
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_layers  = model.config.num_hidden_layers           # 32
    n_kv_heads = model.config.num_key_value_heads        # 8
    head_dim  = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}, rope_theta={rope_base}")

    # ------------------------------------------------------------------ #
    # NEW text — completely different topics from original experiment      #
    # ------------------------------------------------------------------ #
    passages = [
        # Biochemistry
        "Protein folding is one of the most complex problems in molecular biology. A protein's function depends critically on its three-dimensional structure, yet predicting this structure from the amino acid sequence alone remained unsolved for decades. The hydrophobic effect, hydrogen bonding, van der Waals interactions, and electrostatics all contribute to the native fold. Misfolded proteins are implicated in diseases such as Alzheimer's, Parkinson's, and prion diseases. AlphaFold2, developed by DeepMind, achieved near-experimental accuracy on the CASP14 benchmark in 2020, representing a landmark breakthrough. The protein universe contains hundreds of millions of sequences discovered by environmental sequencing.",
        # Cryptography
        "Modern cryptography is built on computational hardness assumptions. RSA relies on the difficulty of factoring large integers; elliptic-curve cryptography relies on the discrete logarithm problem on elliptic curves. A 256-bit elliptic-curve key provides roughly the same security as a 3072-bit RSA key. Post-quantum cryptography anticipates the threat from quantum computers running Shor's algorithm. NIST standardized CRYSTALS-Kyber for key encapsulation and CRYSTALS-Dilithium for digital signatures in 2022. Zero-knowledge proofs allow one party to prove knowledge of a secret without revealing it. The zk-SNARK construction underlies many modern blockchain privacy protocols.",
        # Ocean science
        "The deep ocean remains the least explored environment on Earth. Hadal zones—ocean trenches deeper than six thousand meters—host unique communities of amphipods, holothurians, and microbes adapted to pressures exceeding six hundred atmospheres. The Challenger Deep in the Mariana Trench reaches nearly eleven thousand meters. Hydrothermal vents support chemosynthetic ecosystems that derive energy from hydrogen sulfide rather than sunlight, challenging the assumption that all food webs depend on photosynthesis. Cold seeps emit methane and hydrogen sulfide from the seafloor, supporting similar chemosynthetic communities.",
        # Linguistics
        "Human language is unique in its capacity for recursion and displacement—the ability to refer to events removed in time and space. The Sapir-Whorf hypothesis proposes that the language one speaks shapes cognition, a claim supported by studies showing that speakers of languages without spatial terms for left and right navigate differently. Approximately seven thousand languages are spoken today, but more than half are endangered. The Swadesh list of about two hundred core vocabulary items has been used to estimate divergence times between language families. Proto-Indo-European is reconstructed as having been spoken roughly six thousand years ago on the Pontic steppe.",
        # Immunology
        "The adaptive immune system generates enormous receptor diversity through V(D)J recombination, in which gene segments are randomly combined to produce antibody heavy and light chains. Somatic hypermutation further diversifies antibodies after antigen exposure, allowing affinity maturation in germinal centers. T-helper cells coordinate the immune response by secreting cytokines; cytotoxic T cells directly kill infected or cancerous cells; regulatory T cells suppress autoimmune responses. Checkpoint inhibitors such as anti-PD-1 antibodies reinvigorate exhausted T cells in tumors and have transformed oncology. CAR-T cell therapy engineers patient T cells to recognize cancer antigens.",
        # Materials science
        "Two-dimensional materials have attracted enormous interest since the isolation of graphene from graphite by Scotch-tape exfoliation in 2004. Graphene has exceptional electrical conductivity, thermal conductivity, and mechanical strength—approximately two hundred times stronger than steel by weight. Transition metal dichalcogenides such as molybdenum disulfide exhibit a band gap in the monolayer limit, enabling field-effect transistors and photodetectors. Hexagonal boron nitride provides an atomically flat, electrically insulating substrate that dramatically improves graphene electron mobility. Twisted bilayer graphene at the magic angle of about 1.1 degrees exhibits unconventional superconductivity, discovered in 2018.",
        # Philosophy of science
        "Karl Popper argued that scientific theories must be falsifiable—capable of being proven false by experiment. The demarcation problem asks how science is distinguished from pseudoscience. Thomas Kuhn's The Structure of Scientific Revolutions introduced the concept of paradigm shifts: periods of normal science are punctuated by revolutionary breaks when anomalies accumulate and a new framework replaces the old. Imre Lakatos developed the notion of research programmes with a protective belt of auxiliary hypotheses. Paul Feyerabend's epistemological anarchism argued that no single scientific method is universally valid, summarized provocatively as 'anything goes'.",
        # Robotics
        "Simultaneous localization and mapping (SLAM) allows a robot to build a map of an unknown environment while tracking its own position within it. Loop closure detection—recognizing when the robot revisits a location—is essential for correcting accumulated drift. Visual odometry estimates motion from camera images; inertial measurement units provide complementary high-frequency pose estimates. Boston Dynamics' Atlas robot demonstrated dynamic bipedal locomotion and parkour using model predictive control and reinforcement learning. Soft robotics uses compliant materials and pneumatic actuators to interact safely with humans and unstructured environments.",
        # Economics of innovation
        "Joseph Schumpeter described capitalism as a process of creative destruction in which new technologies continuously displace old industries. The knowledge economy depends on intellectual property regimes, network effects, and increasing returns to scale. Venture capital provides equity financing to early-stage startups in exchange for ownership stakes, typically targeting ten-times returns within five to seven years. The patent system grants temporary monopoly rights in exchange for public disclosure of innovations. Silicon Valley's success has been attributed to dense networks of engineers, proximity to Stanford and Berkeley, permissive non-compete laws, and a culture of risk tolerance.",
        # Neuroscience of memory
        "The hippocampus is essential for the formation of episodic and semantic memories. Long-term potentiation—the activity-dependent strengthening of synapses—is widely considered the cellular mechanism of learning. The standard model of memory consolidation holds that memories are initially stored in the hippocampus and gradually transferred to the neocortex during sleep. Sharp-wave ripples in the hippocampus during slow-wave sleep replay recent experiences. The entorhinal cortex contains grid cells that tile space with hexagonally periodic firing fields; place cells in the hippocampus fire in specific locations. Together they form a cognitive map of the environment.",
        # Climate physics
        "Radiative forcing quantifies the change in energy flux in the atmosphere caused by a perturbation. A doubling of atmospheric CO2 produces approximately 3.7 W/m² of radiative forcing. The equilibrium climate sensitivity—the eventual warming after doubling CO2—is estimated at 2.5 to 4 degrees Celsius by the IPCC AR6. Positive feedbacks include water-vapor feedback, ice-albedo feedback, and lapse-rate feedback. Cloud feedbacks remain the largest source of uncertainty. The Atlantic Meridional Overturning Circulation transports warm surface water northward and cold deep water southward, moderating European climate. Evidence suggests AMOC is weakening.",
        # Number theory
        "Prime numbers have fascinated mathematicians since antiquity. The fundamental theorem of arithmetic states that every integer greater than one factors uniquely into primes. The Riemann hypothesis, one of the Millennium Prize Problems, asserts that all non-trivial zeros of the Riemann zeta function lie on the critical line with real part one-half. Its truth would imply tight bounds on the distribution of primes. The twin prime conjecture holds that there are infinitely many primes p such that p plus two is also prime. In 2013 Yitang Zhang proved a finite bounded gap between consecutive primes, a breakthrough toward the twin prime conjecture.",
        # Social network theory
        "Network science studies the structure and dynamics of complex networks. The small-world phenomenon, characterized by short average path lengths and high clustering coefficients, was famously demonstrated by Stanley Milgram's experiments suggesting that any two people are separated by about six degrees. Preferential attachment, in which new nodes are more likely to connect to high-degree nodes, produces power-law degree distributions observed in the World Wide Web and citation networks. Community detection algorithms partition networks into densely connected subgraphs. Epidemic spreading on networks depends critically on the reproductive number R-naught and the network's spectral properties.",
        # Space exploration
        "The James Webb Space Telescope, launched in December 2021, observes in the near- and mid-infrared with a 6.5-meter primary mirror composed of eighteen gold-coated beryllium segments. It operates at the L2 Lagrange point, 1.5 million kilometers from Earth. JWST has imaged galaxies formed within 300 million years of the Big Bang and detected signatures of carbon dioxide in exoplanet atmospheres. The Artemis program aims to return humans to the lunar surface by the mid-2020s using the Space Launch System rocket and the Orion capsule. SpaceX's Starship, designed for full reusability, targets missions to the Moon and Mars.",
        # Fluid dynamics
        "Turbulence is one of the last unsolved problems in classical physics. The Navier-Stokes equations govern the motion of viscous fluids, but whether smooth solutions always exist for three-dimensional flow remains an open Millennium Prize Problem. The Reynolds number—the ratio of inertial to viscous forces—determines whether flow is laminar or turbulent. Kolmogorov's 1941 theory predicts that energy cascades from large eddies to small ones with a five-thirds power-law spectrum. Computational fluid dynamics solves discretized Navier-Stokes equations to simulate aerodynamic flows around aircraft, with direct numerical simulation resolving all scales but limited to low Reynolds numbers.",
        # Music technology
        "Digital audio workstations transformed music production by moving recording, editing, and mixing entirely into software. The Fourier transform decomposes audio signals into frequency components; the fast Fourier transform algorithm makes this computationally tractable. Pulse-code modulation samples audio at discrete intervals—CD audio uses 44,100 samples per second at 16-bit depth. Lossy audio compression formats such as MP3 exploit psychoacoustic masking: frequencies that are inaudible due to simultaneous louder sounds are discarded. Neural audio synthesis models such as WaveNet and SoundStream generate natural-sounding speech and music by predicting audio samples autoregressively or using quantization codebooks.",
    ]

    full_text = " ".join(passages)
    inputs = tok(full_text, return_tensors="pt", max_length=2048, truncation=True)
    full_ids = inputs.input_ids.to(model.device)
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"\nTokens: {n_tok}, prefix: {prefix_len}, continuation: {cont_len}")

    sliding_window = 32

    # ------------------------------------------------------------------ #
    # Importance scorer (identical to original)                            #
    # ------------------------------------------------------------------ #
    def score_importance(kv_cache, obs_window=32):
        """Key-key attention scorer (SnapKV-style, causal)."""
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')

        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()  # (n_kv_heads, seq, dim)
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

    # ------------------------------------------------------------------ #
    # Random eviction mask builder                                         #
    # ------------------------------------------------------------------ #
    def build_random_mask(seq_len, keep_pct, sw=32, seed=42):
        """Keep BOS + sliding window tail + random selection for the rest.

        Identical structure to importance-based but token selection is uniform-random.
        """
        rng = random.Random(seed)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[0] = True                    # always keep BOS
        mask[-sw:] = True                 # always keep sliding window
        n_to_keep = max(int(seq_len * keep_pct / 100), sw + 1)
        n_from_rand = n_to_keep - mask.sum().item()
        if n_from_rand > 0:
            candidates = [i for i in range(1, seq_len - sw) if not mask[i]]
            chosen = rng.sample(candidates, min(n_from_rand, len(candidates)))
            for idx in chosen:
                mask[idx] = True
        return mask

    # ------------------------------------------------------------------ #
    # Evict + E8-quantize + measure compression (identical to original)   #
    # ------------------------------------------------------------------ #
    def evict_quantize(kv_cache, keep_mask, bits):
        """Evict tokens, E8 quantize kept, measure honest compression."""
        levels = 2 ** bits
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()   # (1, n_kv_heads, seq, dim)
            v = vl.float().cpu()

            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor in [(True, k), (False, v)]:
                t = tensor[0].clone()  # (n_kv_heads, seq, dim)

                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]

                    kept_data = t_head[keep_mask]       # (n_kept, dim)
                    rotated   = kept_data @ H.T
                    amax      = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc        = amax / (levels / 2)
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
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to(model.device), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to(model.device))

        # Compress: temporal delta + zstd level 22
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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2   # fp16 per kept vector
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers * 2
        total       = total_idx + scale_bytes + mask_bytes

        return {
            "fp16":   total_fp16,
            "idx":    total_idx,
            "scale":  scale_bytes,
            "mask":   mask_bytes,
            "total":  total,
            "ratio":  total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ #
    # Measure PPL for a given KV cache + keep mask                        #
    # ------------------------------------------------------------------ #
    def measure_ppl(kv, keep_mask):
        evict_mask = ~keep_mask
        attn_ctx  = torch.ones(prefix_len, dtype=torch.long, device=model.device)
        attn_ctx[evict_mask] = 0
        attn_full = torch.cat([attn_ctx,
                               torch.ones(cont_len, dtype=torch.long, device=model.device)])
        attn_mask = attn_full.unsqueeze(0)
        with torch.no_grad():
            cont_out = model(full_ids[:, prefix_len:], past_key_values=kv,
                             attention_mask=attn_mask, use_cache=True)
            logits  = cont_out.logits[:, :-1, :]
            targets = full_ids[:, prefix_len + 1:].contiguous()
            loss    = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return torch.exp(loss).item()

    # ------------------------------------------------------------------ #
    # Baseline PPL                                                         #
    # ------------------------------------------------------------------ #
    print("\nComputing baseline PPL...")
    with torch.no_grad():
        prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
        kv_bl = prefix_out.past_key_values
        cont_out = model(full_ids[:, prefix_len:], past_key_values=kv_bl, use_cache=True)
        logits  = cont_out.logits[:, :-1, :]
        targets = full_ids[:, prefix_len + 1:].contiguous()
        loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ------------------------------------------------------------------ #
    # Score importance once                                                #
    # ------------------------------------------------------------------ #
    print("Scoring token importance...")
    with torch.no_grad():
        prefix_out2 = model(full_ids[:, :prefix_len], use_cache=True)
        kv_score = prefix_out2.past_key_values
    importance = score_importance(kv_score)
    print(f"Importance stats: min={importance.min():.4f} max={importance.max():.4f} "
          f"mean={importance.mean():.4f}")

    # ------------------------------------------------------------------ #
    # Configs                                                              #
    # ------------------------------------------------------------------ #
    # (name, bits, keep_pct, use_random_eviction)
    configs = [
        # --- reproducibility configs (match originals) ---
        ("2b no-evict",            2, 100, False),
        ("2b+70% importance",      2,  30, False),
        ("2b+80% importance",      2,  20, False),
        # --- random eviction baselines ---
        ("2b+70% RANDOM",          2,  30, True),
        ("2b+80% RANDOM",          2,  20, True),
    ]

    print(f"\n{'Config':<26s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} "
          f"{'Idx':>10s} {'Scale':>8s} {'Total':>10s}")
    print("-" * 100)

    results = []

    for name, bits, keep_pct, use_random in configs:
        torch.cuda.empty_cache()
        t0 = time.time()

        with torch.no_grad():
            prefix_out = model(full_ids[:, :prefix_len], use_cache=True)
            kv = prefix_out.past_key_values

        if keep_pct >= 100:
            keep_mask = torch.ones(prefix_len, dtype=torch.bool)
        elif use_random:
            keep_mask = build_random_mask(prefix_len, keep_pct, sw=sliding_window)
        else:
            # Importance-based eviction (identical to original)
            keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
            keep_mask[0] = True
            keep_mask[-sliding_window:] = True
            n_to_keep     = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
            n_from_imp    = n_to_keep - keep_mask.sum().item()
            if n_from_imp > 0:
                imp = importance.clone()
                imp[keep_mask] = -float('inf')
                _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
                keep_mask[top_idx] = True

        info, kv = evict_quantize(kv, keep_mask, bits)
        ppl     = measure_ppl(kv, keep_mask)
        delta   = ((ppl - baseline_ppl) / baseline_ppl) * 100
        elapsed = time.time() - t0

        tag = " [RANDOM]" if use_random else ""
        print(f"{name:<26s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x "
              f"{info['n_kept']:5d} {info['idx']:>10,} {info['scale']:>8,} "
              f"{info['total']:>10,}  ({elapsed:.1f}s){tag}")

        results.append({
            "name":     name,
            "bits":     bits,
            "keep_pct": keep_pct,
            "random":   use_random,
            "ppl":      ppl,
            "delta":    delta,
            **info,
        })

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY — prefix={prefix_len}, cont={cont_len}, "
          f"baseline_ppl={baseline_ppl:.4f}")
    print(f"{'='*80}")

    # Original results for comparison
    originals = {
        "2b no-evict":   (6.72,  +0.36),
        "2b+70% importance": (21.58, +0.43),
        "2b+80% importance": (31.99, +0.94),
    }

    print("\nReproducibility check (target: within ±0.5% PPL delta of original):")
    print(f"  {'Config':<26s} {'Old Ratio':>10s} {'New Ratio':>10s} "
          f"{'Old Δ%':>8s} {'New Δ%':>8s} {'Δ diff':>8s} {'PASS?':>6s}")
    for r in results:
        if r["name"] in originals:
            old_ratio, old_delta = originals[r["name"]]
            diff = abs(r["delta"] - old_delta)
            passed = "YES" if diff <= 0.5 else "NO "
            print(f"  {r['name']:<26s} {old_ratio:>10.2f}x {r['ratio']:>10.2f}x "
                  f"{old_delta:>+8.2f}% {r['delta']:>+8.2f}% {diff:>+8.2f}% {passed:>6s}")

    print("\nImportance vs Random eviction (same compression ratio):")
    pairs = [("2b+70% importance", "2b+70% RANDOM"),
             ("2b+80% importance", "2b+80% RANDOM")]
    for imp_name, rnd_name in pairs:
        r_imp = next((r for r in results if r["name"] == imp_name), None)
        r_rnd = next((r for r in results if r["name"] == rnd_name), None)
        if r_imp and r_rnd:
            advantage = r_rnd["delta"] - r_imp["delta"]
            verdict = "IMPORTANCE WINS" if advantage > 0 else "RANDOM WINS(?!)"
            print(f"  {imp_name:<26s}  Δ={r_imp['delta']:+.2f}%  |  "
                  f"{rnd_name:<26s}  Δ={r_rnd['delta']:+.2f}%  |  "
                  f"Importance advantage: {advantage:+.2f}%  [{verdict}]")

    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return results


@app.local_entrypoint()
def main():
    import time
    print("Launching Mistral-7B validation rerun on Modal A10G GPU...")
    results = run_experiment.remote()

    # Write results locally
    out_path = os.path.join(
        os.path.dirname(__file__), "..",
        ".company", "engineering", "mistral_7b_validation_rerun.md"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# Mistral-7B Validation Rerun — Reproducibility + Random Eviction\n\n")
        f.write("**Model:** Mistral-7B-v0.1 (32 layers, 8 KV heads, head_dim=128)\n")
        f.write("**GPU:** A10G (24GB)\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Text:** New passages (biochemistry, cryptography, ocean science, "
                "linguistics, immunology, materials science, philosophy of science, "
                "robotics, economics, neuroscience, climate physics, number theory, "
                "network theory, space, fluid dynamics, music technology)\n\n")
        f.write("## Results\n\n")
        f.write("| Config | PPL | Delta% | Ratio | Kept | Idx | Scale | Total |\n")
        f.write("|--------|-----|--------|-------|------|-----|-------|-------|\n")
        for r in results:
            tag = " (random)" if r["random"] else ""
            f.write(f"| {r['name']}{tag} | {r['ppl']:.4f} | {r['delta']:+.2f}% | "
                    f"{r['ratio']:.2f}x | {r['n_kept']} | {r['idx']:,} | "
                    f"{r['scale']:,} | {r['total']:,} |\n")

        f.write("\n## Reproducibility Check\n\n")
        originals = {
            "2b no-evict":       (6.72,  +0.36),
            "2b+70% importance": (21.58, +0.43),
            "2b+80% importance": (31.99, +0.94),
        }
        f.write("| Config | Old Ratio | New Ratio | Old Δ% | New Δ% | Diff | Pass? |\n")
        f.write("|--------|-----------|-----------|--------|--------|------|-------|\n")
        for r in results:
            if r["name"] in originals:
                old_ratio, old_delta = originals[r["name"]]
                diff = abs(r["delta"] - old_delta)
                passed = "YES" if diff <= 0.5 else "NO"
                f.write(f"| {r['name']} | {old_ratio:.2f}x | {r['ratio']:.2f}x | "
                        f"{old_delta:+.2f}% | {r['delta']:+.2f}% | {diff:+.2f}% | {passed} |\n")

        f.write("\n## Importance vs Random Eviction\n\n")
        f.write("| Importance Config | Importance Δ% | Random Config | Random Δ% | Advantage |\n")
        f.write("|-------------------|---------------|---------------|-----------|----------|\n")
        pairs = [("2b+70% importance", "2b+70% RANDOM"),
                 ("2b+80% importance", "2b+80% RANDOM")]
        for imp_name, rnd_name in pairs:
            r_imp = next((r for r in results if r["name"] == imp_name), None)
            r_rnd = next((r for r in results if r["name"] == rnd_name), None)
            if r_imp and r_rnd:
                adv = r_rnd["delta"] - r_imp["delta"]
                f.write(f"| {imp_name} | {r_imp['delta']:+.2f}% | "
                        f"{rnd_name} | {r_rnd['delta']:+.2f}% | {adv:+.2f}% |\n")

        f.write("\n## Honest Notes\n\n")
        f.write("- All overhead included: compressed E8 indices + fp16 scales + token mask bits\n")
        f.write("- Evicted tokens zeroed, attention mask excludes them during continuation\n")
        f.write("- Random eviction: keeps BOS + last 32 tokens, remainder uniform-random (seed=42)\n")
        f.write("- Importance eviction: keeps BOS + last 32 tokens + top-k by key-key attention\n")
        f.write("- FP16 model inference on GPU, E8 quantization on CPU\n")
        f.write("- RoPE removed from keys before E8, reapplied after\n")
        f.write("- Per-head per-layer compression with temporal delta + zstd level 22\n")
        f.write("- dtype=torch.float16 (not deprecated torch_dtype)\n")
        f.write("- numpy<2.0, transformers>=4.44.0,<5.0.0\n")

    print(f"\nResults written to {out_path}")

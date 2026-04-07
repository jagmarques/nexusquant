"""Calibrated PCA rotation vs Hadamard: can per-head PCA cut quantization error?

Hypothesis: PCA rotation aligns with actual KV-cache data distribution, while
Hadamard is generic. Better alignment → lower quantization error → we can push
eviction harder and still stay under <1% PPL degradation.

Phase 1: Calibration
  - Run model on text prefix to collect KV cache
  - For each layer/head/KV: extract token vectors, apply inverse_rope on keys,
    compute SVD, store Vt (the PCA basis, head_dim × head_dim)

Phase 2: Quantization with PCA rotation
  - Rotate: data @ pca_basis.T  (instead of data @ H.T)
  - Per-vector amax scaling → E8 quantize → dequantize
  - Unrotate: (coords * sc) @ pca_basis  (instead of @ H)

Phase 3: Honest compression accounting
  - PCA basis overhead: n_layers * 2 * n_kv_heads * head_dim * head_dim * 2 bytes
  - For Mistral-7B: 32 * 2 * 8 * 128 * 128 * 2 = 16,777,216 bytes (16 MB)
  - Reported at current prefix length AND projected to 4096 tokens

Configs tested on EASY + HARD text:
  HAD: Hadamard baseline (2b no-evict, 2b+35%evict)
  PCA: PCA rotation      (2b no-evict, 2b+35/50/60/70%evict)
  ADP: Adaptive eviction (PCA + per-head MSE sensitivity scoring)
"""

import modal
import os

app = modal.App("nexusquant-calibrated")

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
    timeout=7200,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_calibrated():
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

    print("=" * 80)
    print("NEXUSQUANT — Calibrated PCA Rotation vs Hadamard (Mistral-7B)")
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

    # PCA basis storage: [n_layers][2][n_kv_heads] → (head_dim, head_dim) fp32
    # Index 0 = keys, 1 = values
    PCA_BASIS_BYTES = n_layers * 2 * n_kv_heads * head_dim * head_dim * 2  # fp16

    # ------------------------------------------------------------------ corpora
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

    # ------------------------------------------------------------------ DynamicCache helpers
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

    # ------------------------------------------------------------------ importance scorer
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

    # ------------------------------------------------------------------ build keep_mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_pct = 100 - evict_pct
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
        return keep_mask

    # ------------------------------------------------------------------ PCA calibration
    def calibrate_pca(kv_cache):
        """
        Extract per-layer, per-head PCA bases from a KV cache.

        Returns: bases[layer][kv_idx][head] = (head_dim, head_dim) float32 tensor
          kv_idx: 0=keys, 1=values
        """
        bases = []
        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            layer_bases = [[], []]  # [key_bases, val_bases]

            for kv_idx, tensor in enumerate([kl, vl]):
                # tensor shape: (1, n_kv_heads, seq_len, head_dim)
                t = tensor[0].float().cpu()  # (n_kv_heads, seq_len, head_dim)

                for h in range(n_kv_heads):
                    vecs = t[h]  # (seq_len, head_dim)

                    if kv_idx == 0:  # keys: remove rope first
                        vecs = inverse_rope(vecs.unsqueeze(0), base=rope_base)[0]
                        # vecs shape after squeeze: (seq_len, head_dim)

                    # Center the data (optional but standard for PCA)
                    vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)

                    # SVD: U (seq_len × k), S (k,), Vt (k × head_dim)
                    # Vt rows are principal components — our rotation basis
                    _, _, Vt = torch.linalg.svd(vecs_centered, full_matrices=False)
                    # Vt: (head_dim, head_dim) — full square rotation matrix

                    layer_bases[kv_idx].append(Vt)  # (head_dim, head_dim)

            bases.append(layer_bases)

        return bases

    # ------------------------------------------------------------------ per-head MSE sensitivity
    def compute_head_sensitivity(kv_cache, pca_bases, bits=2):
        """
        Compute per-head quantization MSE in PCA space, averaged over K and V.
        Returns: sensitivity[layer][head] = scalar MSE
        """
        levels = 2 ** bits
        sensitivity = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            layer_sens = []

            for h in range(n_kv_heads):
                head_mse = 0.0
                count = 0

                for kv_idx, tensor in enumerate([kl, vl]):
                    t = tensor[0].float().cpu()
                    vecs = t[h]  # (seq_len, head_dim)

                    if kv_idx == 0:  # keys
                        vecs = inverse_rope(vecs.unsqueeze(0), base=rope_base)[0]

                    basis = pca_bases[l][kv_idx][h]  # (head_dim, head_dim)
                    rotated = vecs @ basis.T  # (seq_len, head_dim)

                    amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc = amax / (levels / 2)
                    normalized = rotated / sc
                    groups = normalized.reshape(-1, 8)
                    lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords = lp.reshape(-1, head_dim)
                    dequant = (coords * sc) @ basis  # (seq_len, head_dim)

                    mse = ((vecs - dequant) ** 2).mean().item()
                    head_mse += mse
                    count += 1

                layer_sens.append(head_mse / count)

            sensitivity.append(layer_sens)

        return sensitivity

    # ------------------------------------------------------------------ adaptive keep mask per head
    def build_adaptive_keep_masks(prefix_len, sensitivity, importance):
        """
        Per-head eviction masks based on quantization sensitivity.
        Top 25% most sensitive heads: keep 90% tokens
        Middle 50%: keep 65% tokens
        Bottom 25% least sensitive: keep 50% tokens
        → average kept ~68.75% → average eviction ~31.25% ≈ 35%
        """
        flat_sens = [(sensitivity[l][h], l, h) for l in range(n_layers) for h in range(n_kv_heads)]
        flat_sens.sort(key=lambda x: x[0], reverse=True)  # descending by sensitivity
        total_heads = len(flat_sens)

        top25_idx    = set((l, h) for _, l, h in flat_sens[:total_heads // 4])
        bottom25_idx = set((l, h) for _, l, h in flat_sens[3 * total_heads // 4:])
        # middle 50%: everything else

        keep_rates = {}
        for _, l, h in flat_sens:
            if (l, h) in top25_idx:
                keep_rates[(l, h)] = 0.90
            elif (l, h) in bottom25_idx:
                keep_rates[(l, h)] = 0.50
            else:
                keep_rates[(l, h)] = 0.65

        # Build per-head masks
        masks = {}
        for l in range(n_layers):
            for h in range(n_kv_heads):
                rate = keep_rates[(l, h)]
                evict_pct = round((1 - rate) * 100)
                masks[(l, h)] = build_keep_mask(prefix_len, evict_pct, importance)

        # Also build a global (union) mask for attention mask construction
        # (a token is kept if ANY head keeps it — conservative for attention mask)
        global_keep = torch.zeros(prefix_len, dtype=torch.bool)
        for m in masks.values():
            global_keep |= m

        avg_kept = sum(m.sum().item() for m in masks.values()) / len(masks)
        avg_evict = (1 - avg_kept / prefix_len) * 100

        return masks, global_keep, avg_evict

    # ------------------------------------------------------------------ evict+quantize (Hadamard)
    def evict_quantize_hadamard(kv_cache, keep_mask, bits, prefix_len):
        """Standard Hadamard rotation baseline."""
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []
        cctx = zstandard.ZstdCompressor(level=22)

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor in [(True, k), (False, v)]:
                levels = 2 ** bits
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    kept_data = t_head[keep_mask]
                    rotated   = kept_data @ H.T
                    amax      = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc        = amax / (levels / 2)
                    normalized = rotated / sc
                    groups    = normalized.reshape(-1, 8)
                    lp        = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords    = lp.reshape(-1, head_dim)
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
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        total_idx = 0
        for coords_arr in all_coords:
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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total       = total_idx + scale_bytes + mask_bytes

        return {
            "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
            "mask": mask_bytes, "basis": 0, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ evict+quantize (PCA)
    def evict_quantize_pca(kv_cache, keep_mask, bits, prefix_len, pca_bases,
                           global_keep_mask=None):
        """PCA rotation: uses per-head Vt basis instead of Hadamard.

        global_keep_mask: if provided (adaptive case), it's the union mask
          used for set_kv (all heads share the same cache shape). Per-head
          masking is handled by zeroing evicted tokens per head.
        """
        if global_keep_mask is None:
            global_keep_mask = keep_mask

        n_kept = global_keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []
        cctx = zstandard.ZstdCompressor(level=22)

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for kv_idx, (is_key, tensor) in enumerate([(True, k), (False, v)]):
                levels = 2 ** bits
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    basis = pca_bases[l][kv_idx][h]  # (head_dim, head_dim)

                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]

                    # Use per-head mask if it was passed as a dict
                    if isinstance(keep_mask, dict):
                        head_mask = keep_mask[(l, h)]
                    else:
                        head_mask = keep_mask

                    kept_data = t_head[head_mask]

                    if kept_data.shape[0] == 0:
                        result = torch.zeros_like(t_head)
                    else:
                        rotated    = kept_data @ basis.T
                        amax       = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                        sc         = amax / (levels / 2)
                        normalized = rotated / sc
                        groups     = normalized.reshape(-1, 8)
                        lp         = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                        coords     = lp.reshape(-1, head_dim)
                        quantized  = (coords * sc) @ basis  # unrotate with PCA basis

                        int_coords = coords.detach().numpy()
                        has_half = np.any(
                            np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25
                        )
                        n_head_kept = head_mask.sum().item()
                        if has_half:
                            all_coords.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                        else:
                            all_coords.append(np.round(int_coords.flatten()).astype(np.int8))

                        result = torch.zeros_like(t_head)
                        result[head_mask] = quantized

                    if is_key:
                        t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = result

                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        # For compression accounting we use global_keep_mask n_kept
        n_kept_global = global_keep_mask.sum().item()
        total_idx = 0
        for coords_arr in all_coords:
            arr = coords_arr.ravel()
            # Each head has its own count; use actual length
            total_idx += len(cctx.compress(arr.tobytes()))

        scale_bytes = n_kept_global * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total_no_basis = total_idx + scale_bytes + mask_bytes
        total_with_basis = total_no_basis + PCA_BASIS_BYTES

        # Projected ratio at 4096 prefix tokens (scale idx+scale by 4096/prefix_len)
        scale_factor = 4096 / prefix_len
        total_idx_4k   = total_idx   * scale_factor
        scale_bytes_4k = scale_bytes * scale_factor
        fp16_4k        = total_fp16  * scale_factor
        total_4k = total_idx_4k + scale_bytes_4k + mask_bytes + PCA_BASIS_BYTES

        return {
            "fp16":  total_fp16,
            "idx":   total_idx,
            "scale": scale_bytes,
            "mask":  mask_bytes,
            "basis": PCA_BASIS_BYTES,
            "total": total_with_basis,
            "ratio": total_fp16 / total_with_basis if total_with_basis > 0 else 0,
            "ratio_4k": fp16_4k / total_4k if total_4k > 0 else 0,
            "n_kept": n_kept_global,
        }, kv_cache

    # ------------------------------------------------------------------ main experiment runner
    def run_on_text(text_name, text):
        inputs = tok(text, return_tensors="pt", max_length=2048, truncation=True)
        full_ids  = inputs.input_ids.to("cuda")
        n_tok     = full_ids.shape[1]
        prefix_len = n_tok // 2
        cont_len   = n_tok - prefix_len

        print(f"\n{'='*80}")
        print(f"TEXT: {text_name} | tokens={n_tok}, prefix={prefix_len}, cont={cont_len}")
        print(f"PCA basis overhead: {PCA_BASIS_BYTES/1e6:.1f} MB")
        fp16_kv_bytes = n_layers * n_kv_heads * prefix_len * head_dim * 2 * 2  # K+V
        print(f"KV cache (fp16): {fp16_kv_bytes/1e6:.1f} MB")
        print(f"Basis overhead fraction at current length: {PCA_BASIS_BYTES/fp16_kv_bytes*100:.1f}%")
        fp16_kv_4k = n_layers * n_kv_heads * 4096 * head_dim * 2 * 2
        print(f"Basis overhead fraction at 4096 tokens:   {PCA_BASIS_BYTES/fp16_kv_4k*100:.1f}%")
        print(f"{'='*80}")

        # Baseline (uncompressed)
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            cout = model(full_ids[:, prefix_len:],
                         past_key_values=pout.past_key_values, use_cache=True)
            logits  = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"Baseline PPL: {baseline_ppl:.4f}")

        # Collect KV cache once for calibration and importance scoring
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)

        importance = score_importance(pout.past_key_values)

        # PCA calibration (from same text — same distribution as what we'll compress)
        print("Calibrating PCA bases...")
        t_cal = time.time()
        pca_bases = calibrate_pca(pout.past_key_values)
        print(f"PCA calibration done in {time.time()-t_cal:.2f}s")

        # Compute per-head sensitivity
        print("Computing per-head quantization sensitivity...")
        sensitivity = compute_head_sensitivity(pout.past_key_values, pca_bases, bits=2)
        flat_sens = [sensitivity[l][h] for l in range(n_layers) for h in range(n_kv_heads)]
        print(f"Head sensitivity: min={min(flat_sens):.4f}, max={max(flat_sens):.4f}, mean={sum(flat_sens)/len(flat_sens):.4f}")

        results = []
        header = f"\n{'Config':<32s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Ratio@4K':>9s} {'Kept':>6s}"
        print(header)
        print("-" * 80)

        def eval_config(name, kv, keep_mask_or_dict, global_keep, use_pca, bits, evict_pct):
            torch.cuda.empty_cache()

            if use_pca:
                info, kv = evict_quantize_pca(kv, keep_mask_or_dict, bits, prefix_len,
                                              pca_bases, global_keep_mask=global_keep)
            else:
                info, kv = evict_quantize_hadamard(kv, keep_mask_or_dict, bits, prefix_len)
                info["ratio_4k"] = info["ratio"]  # Hadamard has no basis overhead

            evict_mask = ~global_keep
            attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(cont_len, dtype=torch.long, device="cuda")
            ])

            with torch.no_grad():
                cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                             attention_mask=attn_full.unsqueeze(0), use_cache=True)
                logits  = cout.logits[:, :-1, :].float()
                targets = full_ids[:, prefix_len + 1:]
                loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl     = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 0.5 else (" <<" if abs(delta) < 1.0 else "")
            print(f"{name:<32s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x "
                  f"{info['ratio_4k']:8.2f}x {info['n_kept']:5d}{tag}")

            return {
                "name": name, "evict_pct": evict_pct,
                "ppl": ppl, "delta": delta,
                "ratio": info["ratio"],
                "ratio_4k": info["ratio_4k"],
                "n_kept": info["n_kept"],
                "baseline": baseline_ppl, "text": text_name,
                "use_pca": use_pca,
            }

        # ---- Group 1: Hadamard baselines ----
        print(f"\n--- Hadamard Baseline ---")
        for evict_pct in [0, 35]:
            label = f"HAD 2b+{evict_pct}%evict" if evict_pct > 0 else "HAD 2b no-evict"
            torch.cuda.empty_cache()
            with torch.no_grad():
                pout2 = model(full_ids[:, :prefix_len], use_cache=True)
            kv = pout2.past_key_values
            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            r = eval_config(label, kv, keep_mask, keep_mask, False, 2, evict_pct)
            results.append(r)

        # ---- Group 2: PCA rotation ----
        print(f"\n--- PCA Rotation ---")
        for evict_pct in [0, 35, 50, 60, 70]:
            label = f"PCA 2b+{evict_pct}%evict" if evict_pct > 0 else "PCA 2b no-evict"
            torch.cuda.empty_cache()
            with torch.no_grad():
                pout2 = model(full_ids[:, :prefix_len], use_cache=True)
            kv = pout2.past_key_values
            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            r = eval_config(label, kv, keep_mask, keep_mask, True, 2, evict_pct)
            results.append(r)

        # ---- Group 3: PCA + adaptive eviction ----
        print(f"\n--- PCA + Adaptive Eviction ---")
        torch.cuda.empty_cache()
        with torch.no_grad():
            pout2 = model(full_ids[:, :prefix_len], use_cache=True)
        kv = pout2.past_key_values
        adaptive_masks, global_keep, avg_evict = build_adaptive_keep_masks(
            prefix_len, sensitivity, importance
        )
        avg_kept_pct = global_keep.sum().item() / prefix_len * 100
        print(f"Adaptive: avg evict={avg_evict:.1f}%, global union keeps {avg_kept_pct:.1f}% of tokens")
        r = eval_config(f"PCA+Adaptive (~{avg_evict:.0f}%evict)", kv,
                        adaptive_masks, global_keep, True, 2, avg_evict)
        results.append(r)

        return results

    # ------------------------------------------------------------------ run both corpora
    all_results = []
    easy_results = run_on_text("EASY (general science)", EASY_TEXT)
    hard_results = run_on_text("HARD (advanced technical)", HARD_TEXT)
    all_results.extend(easy_results)
    all_results.extend(hard_results)

    # ------------------------------------------------------------------ summary table
    print(f"\n{'='*80}")
    print("CALIBRATED COMPRESSION SUMMARY TABLE")
    print(f"{'='*80}")
    fmt = f"\n{'Config':<32s} {'Easy Δ%':>8s} {'Hard Δ%':>8s} {'Max Δ%':>8s} {'Ratio':>7s} {'Ratio@4K':>9s} {'<0.5%?':>7s}"
    print(fmt)
    print("-" * 85)

    best_config = None
    best_ratio  = 0.0
    summary_rows = []

    def find_result(results, name_prefix, text_key):
        for r in results:
            if r["name"].startswith(name_prefix) and text_key in r["text"]:
                return r
        return None

    all_config_names = list(dict.fromkeys(r["name"] for r in all_results))  # preserve order

    for name in all_config_names:
        easy = find_result(easy_results, name, "EASY")
        hard = find_result(hard_results, name, "HARD")
        if easy and hard:
            max_d     = max(abs(easy["delta"]), abs(hard["delta"]))
            avg_ratio = (easy["ratio"] + hard["ratio"]) / 2
            avg_r4k   = (easy["ratio_4k"] + hard["ratio_4k"]) / 2
            ok = "YES" if max_d < 0.5 else ("CLOSE" if max_d < 1.0 else "NO")
            marker = " <<<" if ok == "YES" else (" <<" if ok == "CLOSE" else "")
            print(f"{name:<32s} {easy['delta']:+7.2f}% {hard['delta']:+7.2f}% "
                  f"{max_d:7.2f}% {avg_ratio:6.2f}x {avg_r4k:8.2f}x {ok:>6s}{marker}")
            summary_rows.append({
                "name": name, "easy_delta": easy["delta"], "hard_delta": hard["delta"],
                "max_delta": max_d, "ratio": avg_ratio, "ratio_4k": avg_r4k, "ok": ok,
            })
            if ok == "YES" and avg_ratio > best_ratio:
                best_ratio  = avg_ratio
                best_config = summary_rows[-1]

    print()
    if best_config:
        print(f">>> BEST <0.5% CONFIG: {best_config['name']}")
        print(f"    Easy Δ%={best_config['easy_delta']:+.3f}%  "
              f"Hard Δ%={best_config['hard_delta']:+.3f}%  "
              f"Max={best_config['max_delta']:.3f}%  "
              f"Ratio={best_config['ratio']:.2f}x  Ratio@4K={best_config['ratio_4k']:.2f}x")
    else:
        print(">>> No config achieved <0.5% on BOTH texts.")
        close = [r for r in summary_rows if r["ok"] == "CLOSE"]
        if close:
            best_close = min(close, key=lambda x: x["max_delta"])
            print(f"    Closest <1%: {best_close['name']}  MaxΔ={best_close['max_delta']:.3f}%  "
                  f"Ratio={best_close['ratio']:.2f}x  Ratio@4K={best_close['ratio_4k']:.2f}x")

    # PCA improvement summary
    print(f"\n--- PCA vs Hadamard comparison at equivalent eviction ---")
    for evict_pct in [0, 35]:
        had_name = f"HAD 2b+{evict_pct}%evict" if evict_pct > 0 else "HAD 2b no-evict"
        pca_name = f"PCA 2b+{evict_pct}%evict" if evict_pct > 0 else "PCA 2b no-evict"
        had = next((r for r in summary_rows if r["name"] == had_name), None)
        pca = next((r for r in summary_rows if r["name"] == pca_name), None)
        if had and pca:
            improvement = had["max_delta"] - pca["max_delta"]
            print(f"  {evict_pct}% evict: HAD max={had['max_delta']:.3f}%  PCA max={pca['max_delta']:.3f}%  "
                  f"Improvement={improvement:+.3f}pp")

    print(f"\nPeak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return all_results, summary_rows, best_config


@app.local_entrypoint()
def main():
    import time
    print("Launching calibrated PCA compression experiment on Modal A10G...")
    all_results, summary_rows, best_config = run_calibrated.remote()

    out_dir = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "calibrated_results.md")

    with open(out, "w") as f:
        f.write("# Calibrated PCA Rotation Results — Mistral-7B\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Hypothesis:** PCA rotation per head reduces quantization error vs generic Hadamard\n\n")
        f.write("## Experiment Design\n")
        f.write("- **HAD**: Hadamard rotation (baseline), 2-bit E8 quantization\n")
        f.write("- **PCA**: Per-head PCA basis calibrated from same text prefix, 2-bit E8\n")
        f.write("- **PCA+Adaptive**: PCA + per-head MSE-based eviction budgets\n\n")
        f.write("## PCA Basis Overhead\n")
        f.write(f"- n_layers=32, n_kv_heads=8, head_dim=128: 32×2×8×128×128×2 = 16,777,216 bytes (16 MB)\n")
        f.write("- This overhead is included in ALL PCA compression ratios\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Config | Easy Δ% | Hard Δ% | Max Δ% | Ratio | Ratio@4K | <0.5%? |\n")
        f.write("|--------|---------|---------|--------|-------|----------|--------|\n")
        for row in summary_rows:
            f.write(f"| {row['name']} | {row['easy_delta']:+.3f}% | {row['hard_delta']:+.3f}% "
                    f"| {row['max_delta']:.3f}% | {row['ratio']:.2f}x | {row['ratio_4k']:.2f}x | {row['ok']} |\n")

        f.write("\n## Best Config\n\n")
        if best_config:
            f.write(f"**{best_config['name']}**\n\n")
            f.write(f"- Easy Δ%: {best_config['easy_delta']:+.3f}%\n")
            f.write(f"- Hard Δ%: {best_config['hard_delta']:+.3f}%\n")
            f.write(f"- Max Δ%: {best_config['max_delta']:.3f}%\n")
            f.write(f"- Compression ratio (at current prefix): {best_config['ratio']:.2f}x\n")
            f.write(f"- Compression ratio (projected at 4096 tokens): {best_config['ratio_4k']:.2f}x\n")
        else:
            f.write("No config achieved <0.5% PPL on BOTH texts in this sweep.\n")
            close = [r for r in summary_rows if r["ok"] == "CLOSE"]
            if close:
                best_close = min(close, key=lambda x: x["max_delta"])
                f.write(f"\nClosest (<1%): **{best_close['name']}**  "
                        f"Max Δ={best_close['max_delta']:.3f}%  "
                        f"Ratio={best_close['ratio']:.2f}x  "
                        f"Ratio@4K={best_close['ratio_4k']:.2f}x\n")

        f.write("\n## Full Results\n\n")
        f.write("| Text | Config | PPL | Delta% | Ratio | Ratio@4K | Kept |\n")
        f.write("|------|--------|-----|--------|-------|----------|------|\n")
        for r in all_results:
            f.write(f"| {r['text'][:25]} | {r['name']} | {r['ppl']:.4f} "
                    f"| {r['delta']:+.3f}% | {r['ratio']:.2f}x | {r['ratio_4k']:.2f}x | {r['n_kept']} |\n")

    print(f"\nResults written to: {out}")

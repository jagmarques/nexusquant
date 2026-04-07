"""Asymmetric K/V compression, boundary-layer protection, and deferred compression.

Three new features tested on Mistral-7B (A100) at ~3544-token prefix:

  1. Asymmetric K/V bits — separate key_bits / value_bits per compressed layer
  2. Boundary protection — first/last N layers stay at FP16 (no quantization)
  3. Deferred compression — only activate when context > threshold tokens
     (included as a sanity check; at 3544 tokens the threshold is always met)

Test matrix (35% and 60% eviction):
  - Baseline:        no compression
  - K2V2:            keys=2b, values=2b, no protection (current best)
  - K3V2:            keys=3b, values=2b, no protection
  - K4V2:            keys=4b, values=2b, no protection (keys near FP-like)
  - K2V2+boundary2:  keys=2b, values=2b, protect first/last 2 layers (FP16)
  - K3V2+boundary2:  keys=3b, values=2b, protect first/last 2 layers (FP16)

Overhead accounting (BRUTAL HONESTY — all overheads included):
  - Compressed layers: delta-zstd E8 index bytes (analytic, zstd level 22)
  - Scale bytes: 1 fp16 scale per kept token per head per layer (K and V separately)
  - Protected layers: full FP16 for ALL prefix tokens (no eviction applied — the
    protected layer budget is added on top of the compressed layers)
  - Eviction mask: ceil(prefix_len/8) * n_compressed_layers (one mask per layer)
  - For asymmetric: key index bytes use key_bits range, value index bytes use val_bits range

Run with: modal run experiments/modal_asymmetric_kv.py
"""
import modal
import os

app = modal.App("nexusquant-asymmetric-kv")

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
    gpu="A100",
    timeout=7200,
    secrets=[HF_SECRET],
    memory=65536,
)
def run_asymmetric_kv():
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
    print("NEXUSQUANT — Asymmetric K/V + Boundary Protection + Deferred Compression")
    print("Mistral-7B | A100 | ~3544-token prefix")
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

    n_layers   = model.config.num_hidden_layers           # 32
    n_kv_heads = model.config.num_key_value_heads         # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, "rope_theta", 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_base={rope_base}")

    # ------------------------------------------------------------------ long corpus
    # ~3544 tokens on Mistral-7B tokenizer — use multi-topic concatenation.
    # We concatenate general-science blocks (easy) + advanced-technical blocks (hard)
    # to produce a diverse, representative prefix.
    MULTI_TEXT = " ".join([
        # ---- General science (easy) ----
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists around the world, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. Fermions are the building blocks: six quarks and six leptons. Forces between the fermions are mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. Despite its success, the Standard Model does not incorporate gravity, dark matter, or dark energy, leaving physicists with significant open questions about the nature of the universe.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation. It began in Britain and quickly spread to Western Europe and North America. The transition from hand production methods to machine manufacturing, new chemical processes, iron production, increased use of steam power, the development of machine tools, and the rise of the factory system fundamentally changed the nature of work and society. This period saw the emergence of the middle class, the growth of cities, and the beginning of modern capitalism. Child labor was common in factories and mines, prompting early labor reforms.",
        "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better suited to their environment will tend to survive and produce more offspring. Over time, these advantageous traits become more common in the population. Genetic variation arises through mutation, recombination, and gene flow. Sexual selection is a special case where traits improve mating success rather than survival directly. Evolutionary theory is supported by evidence from the fossil record, comparative anatomy, molecular biology, and direct observation of evolution in real time.",
        "The development of quantum mechanics in the early 20th century fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain phenomena such as black-body radiation, the photoelectric effect, or the stability of atoms. Max Planck introduced the concept of energy quanta in 1900. Albert Einstein explained the photoelectric effect using photons in 1905. Niels Bohr developed his model of the hydrogen atom in 1913. Werner Heisenberg, Erwin Schrodinger, Paul Dirac, and others built the mathematical framework of modern quantum mechanics throughout the 1920s.",
        "Mathematics has been essential to the development of science and technology throughout human history. From the ancient Babylonians who developed a base-60 number system that we still use for measuring time, to the development of calculus by Newton and Leibniz that made modern physics possible, mathematical ideas have been the foundation of scientific progress. Abstract algebra, topology, and functional analysis — developed largely for theoretical reasons — later turned out to have profound physical applications. The unreasonable effectiveness of mathematics in the natural sciences remains a philosophical puzzle.",
        "The history of astronomy represents one of humanity's oldest scientific endeavors. Ancient civilizations observed the stars for navigation, agriculture, and religious purposes. The Babylonians recorded planetary positions, the Greeks developed geometric models of the cosmos, and Islamic astronomers preserved and extended this knowledge during the medieval period. Copernicus proposed the heliocentric model. Galileo used the telescope to observe Jupiter's moons, the phases of Venus, and sunspots. Newton derived his law of universal gravitation. Modern astronomy covers everything from exoplanet detection to gravitational wave observation.",
        "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence was felt in literature, philosophy, art, music, politics, science, religion, and other aspects of intellectual inquiry. The rediscovery of ancient Greek and Roman texts inspired new ways of thinking about human nature, government, and the natural world. Leonardo da Vinci, Michelangelo, Raphael, and Botticelli represent the artistic pinnacle of the movement. The printing press accelerated the dissemination of ideas across Europe.",
        "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons connected by trillions of synapses. Each neuron can form thousands of connections with other neurons, creating an intricate network that gives rise to thought, memory, emotion, and consciousness. The brain consumes roughly 20 percent of the body's energy despite comprising only about 2 percent of its mass. Neuroplasticity allows the brain to reorganize itself in response to experience. Modern neuroscience uses tools such as fMRI, EEG, and optogenetics to study how neural circuits give rise to behavior.",
        "Climate change represents one of the most significant challenges facing humanity in the 21st century. The burning of fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of greenhouse gases, particularly carbon dioxide and methane, to levels unprecedented in at least 800,000 years. Global average temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels. Consequences include more frequent extreme weather events, rising sea levels, ocean acidification, and disruption of ecosystems. The Paris Agreement aims to limit warming to 1.5 to 2 degrees Celsius through international cooperation.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. It had a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around 100 BC to 400 AD. Roman engineering achievements included aqueducts, roads, concrete construction, and the Pantheon. Roman law formed the basis of many modern legal systems. Latin evolved into the Romance languages — Italian, Spanish, French, Portuguese, and Romanian.",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data without being explicitly programmed. Deep learning uses artificial neural networks with many layers to learn hierarchical representations of data. Transformer architectures, introduced by Vaswani et al. in 2017, revolutionized natural language processing and have since been applied to vision, audio, protein structure prediction, and many other domains.",
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of the Earth's lithosphere. Tectonic plates move because of the relative density of oceanic lithosphere and the relative weakness of the asthenosphere. Convection currents in the mantle drive the motion of plates. Where plates converge, one may subduct beneath the other, producing volcanism and earthquakes. Where plates diverge, new oceanic crust forms at mid-ocean ridges. The theory explains the distribution of earthquakes, volcanoes, mountain ranges, and ocean trenches.",
        "The development of antibiotics in the 20th century represents one of medicine's greatest achievements. Alexander Fleming discovered penicillin in 1928, and its clinical use began in the 1940s. Antibiotics have saved hundreds of millions of lives by treating bacterial infections that were previously often fatal. However, the overuse and misuse of antibiotics has driven the emergence of antibiotic-resistant bacteria, posing a growing threat to public health. Methicillin-resistant Staphylococcus aureus and carbapenem-resistant Enterobacteriaceae are among the most dangerous resistant pathogens.",
        "The structure of DNA, elucidated by Watson and Crick in 1953 using X-ray crystallography data from Franklin and Wilkins, revealed how genetic information is stored and replicated. The double helix consists of two antiparallel strands of nucleotides connected by hydrogen bonds between complementary base pairs: adenine with thymine, guanine with cytosine. During cell division, the strands separate and each serves as a template for the synthesis of a new complementary strand. The central dogma of molecular biology describes how information flows from DNA to RNA to protein.",
        # ---- Advanced technical (hard) ----
        "The renormalization group in quantum field theory provides a systematic framework for understanding how physical theories change with the energy scale of observation. Kenneth Wilson's formulation connects statistical mechanics and quantum field theory through the concept of universality classes, where disparate physical systems exhibit identical critical behavior near phase transitions due to shared symmetry properties and dimensionality. The renormalization group explains why microscopic details are irrelevant to macroscopic critical phenomena and provides a mathematical framework for computing critical exponents.",
        "Homological algebra studies algebraic structures through chain complexes and their derived functors. The Ext and Tor functors provide fundamental invariants that measure the failure of exactness under the Hom and tensor product functors respectively. Spectral sequences provide computational tools for successive approximation of these derived functors. The derived category of an abelian category, constructed by formally inverting quasi-isomorphisms, provides a more natural setting for derived functor computations and connects to deformation theory and algebraic geometry.",
        "The Langlands program represents one of the most ambitious unifying frameworks in mathematics, connecting number theory, algebraic geometry, and representation theory. The geometric Langlands correspondence establishes deep connections between automorphic forms on reductive groups and l-adic representations of absolute Galois groups over function fields. Recent work by Fargues and Scholze on p-adic geometry has brought new tools to bear on the local Langlands correspondence, while Taylor's work has resolved many cases of the global correspondence.",
        "CRISPR-Cas9 gene editing exploits the bacterial adaptive immune system's ability to incorporate foreign DNA fragments into clustered regularly interspaced short palindromic repeats. The Cas9 endonuclease, guided by a chimeric single-guide RNA, introduces double-strand breaks at specific genomic loci, enabling precise modifications through nonhomologous end joining or homology-directed repair pathways. Base editing and prime editing extend CRISPR capabilities to single-nucleotide changes without double-strand breaks. Clinical trials using CRISPR to treat sickle-cell disease have shown promising results.",
        "The AdS/CFT correspondence, proposed by Juan Maldacena in 1997, conjectures an exact duality between type IIB superstring theory on five-dimensional anti-de Sitter space times a five-sphere and four-dimensional N=4 super Yang-Mills theory on the conformal boundary. This holographic principle has profound implications for quantum gravity and strongly coupled quantum field theories. The duality maps the bulk gravitational degrees of freedom to boundary conformal field theory data and has been used to compute properties of quark-gluon plasma and quantum entanglement entropy.",
        "Stochastic partial differential equations driven by space-time white noise arise naturally in the study of random interface growth, population genetics, and directed polymers in random environments. The Kardar-Parisi-Zhang equation describes the universal scaling behavior of growing interfaces, connecting to random matrix theory through the Tracy-Widom distribution. Hairer's theory of regularity structures provides a rigorous mathematical framework for renormalizing such equations and earned him the Fields Medal in 2014.",
        "The tumor microenvironment comprises a complex ecosystem of cancer cells, immune cells, fibroblasts, endothelial cells, and extracellular matrix components that collectively determine tumor progression and therapeutic response. Immune checkpoint inhibitors targeting PD-1/PD-L1 and CTLA-4 have revolutionized oncology by releasing the brakes on anti-tumor immune responses. CAR-T cell therapy engineers patient immune cells to target cancer-specific antigens. Tumor heterogeneity and immune evasion remain major obstacles to durable responses.",
        "Quantum error correction addresses the fundamental challenge of protecting quantum information from decoherence and operational errors. Surface codes, implemented on a two-dimensional lattice of physical qubits, achieve fault-tolerant computation through topological protection, where logical qubits are encoded in the global topology of the code rather than individual physical qubits. Threshold theorems guarantee that below a critical error rate, arbitrarily long quantum computations are possible with only polynomial overhead in qubit count.",
        "The Navier-Stokes equations governing incompressible fluid flow remain one of the seven Millennium Prize Problems. The existence and smoothness of solutions in three dimensions is unresolved, with turbulent cascading energy transfer from large to small scales described by Kolmogorov's 1941 theory predicting the famous minus five-thirds power law in the inertial range of the energy spectrum. Direct numerical simulation of turbulence requires resolving length scales spanning many orders of magnitude, making it computationally prohibitive at high Reynolds numbers.",
        "Persistent homology in topological data analysis provides multiscale shape descriptors for point cloud data by tracking the birth and death of topological features across a filtration of simplicial complexes. The resulting persistence diagrams and barcodes offer stable invariants under perturbation, with stability guarantees provided by the bottleneck and Wasserstein distances. Applications include the analysis of protein folding, brain connectivity, materials microstructure, and manifold learning.",
        "Nonequilibrium statistical mechanics extends beyond the Boltzmann-Gibbs framework to describe systems driven away from thermal equilibrium. The Jarzynski equality and Crooks fluctuation theorem relate free energy differences to nonequilibrium work measurements, while large deviation theory provides the mathematical foundation for understanding rare events in stochastic processes. Active matter systems, from bacteria to cytoskeletal filaments, exhibit collective behaviors that have no equilibrium analogue.",
        "The hypothalamic-pituitary-adrenal axis orchestrates the neuroendocrine stress response through a cascade of hormonal signals. Corticotropin-releasing hormone from the paraventricular nucleus stimulates adrenocorticotropic hormone release from the anterior pituitary, which in turn drives cortisol synthesis in the adrenal cortex, with glucocorticoid receptors mediating negative feedback at multiple levels. Chronic stress dysregulates this axis and has been implicated in depression, anxiety, metabolic syndrome, and cardiovascular disease.",
        # ---- Extra blocks to reach ~3544 tokens ----
        "The discovery of gravitational waves in 2015 by LIGO opened a new observational window on the universe. The merger of two black holes 1.3 billion light-years away produced a signal lasting a fraction of a second, causing spacetime distortions smaller than one-thousandth the diameter of a proton. Since then, dozens of gravitational wave events have been detected, including neutron star mergers that produced coincident electromagnetic counterparts and allowed independent measurements of the Hubble constant. Third-generation detectors such as the Einstein Telescope will probe the gravitational wave sky with much greater sensitivity.",
        "Transformer language models scale remarkably well with compute. The scaling laws of Kaplan et al. showed that model performance on language modeling follows a power law with model size, dataset size, and compute budget — each contributing roughly equally when optimally balanced. Chinchilla scaling revised the optimal compute allocation to require larger datasets relative to model size. Models trained on trillions of tokens with billions of parameters exhibit emergent capabilities not present in smaller models, such as few-shot reasoning, code generation, and instruction following.",
        "Topological insulators are materials that behave as insulators in their bulk but support conducting states on their surfaces or edges that are protected by time-reversal symmetry. These surface states arise from band inversions driven by strong spin-orbit coupling and are topologically distinct from ordinary insulators. The quantum spin Hall effect in two-dimensional topological insulators produces helical edge states where spin and momentum are locked together, making them robust against non-magnetic disorder. The discovery of topological phases has unified condensed matter physics around topological invariants such as Chern numbers and Z2 indices.",
        "Optimal transport theory provides a mathematical framework for comparing probability distributions by finding the most efficient way to transform one distribution into another. The Wasserstein distance, defined as the minimum cost of transporting mass from one distribution to another, has found applications in machine learning for generative modeling, domain adaptation, and fairness. The Monge-Kantorovich duality connects the primal transport problem to a dual problem involving potential functions, enabling efficient computational algorithms through entropic regularization and the Sinkhorn algorithm.",
        "The microbiome — the community of trillions of microorganisms living in and on the human body — plays a crucial role in health and disease. The gut microbiome aids in digestion, synthesizes vitamins, trains the immune system, and produces neuroactive compounds that influence mood and behavior through the gut-brain axis. Dysbiosis — disruption of the microbial community — has been associated with inflammatory bowel disease, obesity, type 2 diabetes, colorectal cancer, and neuropsychiatric disorders. Fecal microbiota transplantation is an effective treatment for recurrent Clostridioides difficile infection.",
    ])

    sliding_window = 32

    # ------------------------------------------------------------------ KV cache helpers
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

    # ------------------------------------------------------------------ scorer
    def score_importance(kv_cache, obs_window=32, pool_kernel=5):
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device="cpu")
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
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

    # ------------------------------------------------------------------ keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        """Build boolean keep mask with BOS and sliding-window anchor always kept."""
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_pct = 100 - evict_pct
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True                           # always keep BOS
        keep_mask[-sliding_window:] = True            # always keep sliding window
        n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float("inf")
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ core evict+quantize
    def evict_quantize(
        kv_cache,
        keep_mask,
        prefix_len,
        key_bits=2,
        value_bits=2,
        protect_boundary=0,
        deferred_threshold=0,
    ):
        """Asymmetric E8 quantization with boundary-layer protection and deferred activation.

        Args:
            kv_cache:           HuggingFace DynamicCache or equivalent
            keep_mask:          bool tensor [prefix_len] — which tokens to keep
            prefix_len:         int, full prefix length before eviction
            key_bits:           int, bits for key quantization in non-protected layers
            value_bits:         int, bits for value quantization in non-protected layers
            protect_boundary:   int, number of layers at each end to leave in FP16
            deferred_threshold: int, skip compression entirely if prefix_len < threshold
                                (0 = always compress)

        Overhead accounting (bytes):
            - Compressed layers: delta-zstd E8 index bytes (key_bits / value_bits)
            - Protected layers:  full FP16 for all prefix_len tokens (no eviction)
            - Scale bytes:       n_kept * n_kv_heads * 2 (for K+V) * 2 (fp16) per compressed layer
            - Mask bytes:        ceil(prefix_len/8) * n_compressed_layers
        """
        # Deferred compression guard
        if deferred_threshold > 0 and prefix_len < deferred_threshold:
            # Return identity — no compression, no eviction
            total_fp16 = 0
            for l in range(n_layers):
                kl, vl = get_kv(kv_cache, l)
                total_fp16 += kl.numel() * 2 + vl.numel() * 2
            return {"fp16": total_fp16, "idx": 0, "scale": 0, "mask": 0,
                    "total": total_fp16, "ratio": 1.0,
                    "n_kept": prefix_len, "deferred": True}, kv_cache

        H = hadamard_matrix(head_dim).cpu()
        cctx = zstandard.ZstdCompressor(level=22)

        n_kept = keep_mask.sum().item()
        total_fp16 = 0

        # Identify which layers are protected vs compressed
        # protect_boundary=2 → protect layers {0,1} and {30,31} in a 32-layer model
        protected_layers = set()
        if protect_boundary > 0:
            for i in range(min(protect_boundary, n_layers)):
                protected_layers.add(i)
            for i in range(max(0, n_layers - protect_boundary), n_layers):
                protected_layers.add(i)
        n_compressed_layers = n_layers - len(protected_layers)

        # Per-layer processing
        all_key_coords = []   # one entry per (compressed_layer * n_kv_heads)
        all_val_coords = []   # same

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2  # full FP16 baseline bytes

            if l in protected_layers:
                # Protected layer: no quantization, no eviction — keep all prefix tokens
                # Write back unchanged (stay fp16 on GPU)
                # (The tensor is already on GPU as fp16; no-op needed)
                continue

            # Compressed layer: evict + quantize keys and values
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
                    # detect half-integer offset (E8 relaxed parity)
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

        # ---- Compression ratio accounting ----
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

        # Scale bytes: 1 fp16 per (kept_token, head, K/V) for each compressed layer
        scale_bytes = n_kept * n_kv_heads * 2 * 2 * n_compressed_layers  # *2=K+V, *2=fp16

        # Mask bytes: 1 shared mask per compressed layer (not per K/V separately)
        mask_bytes = math.ceil(prefix_len / 8) * n_compressed_layers

        # Protected-layer FP16 cost: all prefix_len tokens × all heads × head_dim × 2bytes, K+V
        # These are already in total_fp16, but their compressed size equals their FP16 size
        # (they are not compressed). We account for them separately.
        protected_fp16_bytes = len(protected_layers) * prefix_len * n_kv_heads * head_dim * 2 * 2

        total_compressed = total_idx + scale_bytes + mask_bytes + protected_fp16_bytes

        return {
            "fp16": total_fp16,
            "idx": total_idx,
            "scale": scale_bytes,
            "mask": mask_bytes,
            "protected_fp16": protected_fp16_bytes,
            "total": total_compressed,
            "ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "n_kept": n_kept,
            "n_protected_layers": len(protected_layers),
            "n_compressed_layers": n_compressed_layers,
            "deferred": False,
        }, kv_cache

    # ------------------------------------------------------------------ PPL evaluator
    def measure_ppl(model, input_ids, past_kv, attention_mask):
        """Compute continuation PPL given a (possibly compressed) past_kv."""
        prefix_len = past_kv.key_cache[0].shape[2] if hasattr(past_kv, "key_cache") \
            else past_kv.layers[0].keys.shape[2]
        full_len = input_ids.shape[1]
        cont_ids = input_ids[:, prefix_len:]
        with torch.no_grad():
            out = model(cont_ids, past_key_values=past_kv,
                        attention_mask=attention_mask.unsqueeze(0), use_cache=True)
            logits = out.logits[:, :-1, :].float()
            targets = input_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return torch.exp(loss).item()

    # ------------------------------------------------------------------ run configs on text
    def run_configs_on_text(text_name, text, configs):
        # Tokenize to target prefix length
        inputs = tok(text, return_tensors="pt", max_length=8192, truncation=True)
        full_ids = inputs.input_ids.to("cuda")
        n_tok = full_ids.shape[1]
        # Use first 3544 tokens as prefix (or all if shorter)
        target_prefix = 3544
        prefix_len = min(target_prefix, n_tok - 100)  # leave at least 100 for continuation
        if prefix_len < 500:
            print(f"WARNING: text too short ({n_tok} tokens), skipping.")
            return []
        cont_len = n_tok - prefix_len
        print(f"\n{'='*80}")
        print(f"TEXT: {text_name} | tokens={n_tok}, prefix={prefix_len}, cont={cont_len}")
        print(f"{'='*80}")

        # Baseline PPL (no compression)
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            cout = model(full_ids[:, prefix_len:],
                         past_key_values=pout.past_key_values, use_cache=True)
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"Baseline PPL (no compression): {baseline_ppl:.4f}")

        # Score importance once (shared across all compressed configs for fairness)
        with torch.no_grad():
            pout_for_score = model(full_ids[:, :prefix_len], use_cache=True)
        importance = score_importance(pout_for_score.past_key_values)
        del pout_for_score
        torch.cuda.empty_cache()

        results = []
        header = f"\n{'Config':<30s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot':>5s}"
        print(header)
        print("-" * 76)

        for cfg in configs:
            torch.cuda.empty_cache()
            name            = cfg["name"]
            evict_pct       = cfg["evict_pct"]
            key_bits        = cfg.get("key_bits", 2)
            val_bits        = cfg.get("val_bits", 2)
            protect_boundary = cfg.get("protect_boundary", 0)
            deferred_threshold = cfg.get("deferred_threshold", 0)

            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv = pout.past_key_values

            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            info, kv = evict_quantize(
                kv, keep_mask, prefix_len,
                key_bits=key_bits, value_bits=val_bits,
                protect_boundary=protect_boundary,
                deferred_threshold=deferred_threshold,
            )

            if info["deferred"]:
                print(f"{name:<30s} (deferred — context below threshold)")
                continue

            # Build attention mask: zeros for evicted prefix positions, ones for kept + continuation
            evict_mask = ~keep_mask
            attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(cont_len, dtype=torch.long, device="cuda"),
            ])

            with torch.no_grad():
                cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                             attention_mask=attn_full.unsqueeze(0), use_cache=True)
                logits = cout.logits[:, :-1, :].float()
                targets = full_ids[:, prefix_len + 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 1.0 else (" <<" if abs(delta) < 2.0 else "")
            n_prot = info.get("n_protected_layers", 0)
            print(f"{name:<30s} {ppl:8.4f} {delta:+8.3f}% {info['ratio']:6.2f}x "
                  f"{info['n_kept']:5d} {n_prot:4d}{tag}")
            results.append({
                "name": name, "evict_pct": evict_pct,
                "key_bits": key_bits, "val_bits": val_bits,
                "protect_boundary": protect_boundary,
                "ppl": ppl, "delta": delta,
                "ratio": info["ratio"], "n_kept": info["n_kept"],
                "n_protected": n_prot,
                "baseline": baseline_ppl, "text": text_name,
                "prefix_len": prefix_len,
            })

        return results

    # ------------------------------------------------------------------ config matrix
    # Run each eviction rate separately for clarity
    EVICT_RATES = [35, 60]

    all_configs = []

    # No-compression baseline (only run once at evict=0, included in each eviction block)
    all_configs.append({
        "name": "Baseline (no compression)",
        "evict_pct": 0,
        "key_bits": 2,         # unused — no quantization at evict=0 would need special path
        "val_bits": 2,
        "protect_boundary": 0,
        "_is_baseline": True,
    })

    for evict_pct in EVICT_RATES:
        ep_label = f"{evict_pct}%ev"

        # K2V2: current best (symmetric 2-bit)
        all_configs.append({
            "name": f"K2V2 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 2, "val_bits": 2,
            "protect_boundary": 0,
        })

        # K3V2: asymmetric (keys at 3-bit, values at 2-bit)
        all_configs.append({
            "name": f"K3V2 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 3, "val_bits": 2,
            "protect_boundary": 0,
        })

        # K4V2: keys at 4-bit (near FP quality), values at 2-bit
        all_configs.append({
            "name": f"K4V2 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 4, "val_bits": 2,
            "protect_boundary": 0,
        })

        # K2V2 + boundary-2: protect first/last 2 layers
        all_configs.append({
            "name": f"K2V2+boundary2 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 2, "val_bits": 2,
            "protect_boundary": 2,
        })

        # K3V2 + boundary-2: asymmetric + boundary protection
        all_configs.append({
            "name": f"K3V2+boundary2 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 3, "val_bits": 2,
            "protect_boundary": 2,
        })

        # Deferred compression sanity check: threshold=1000 (will compress since prefix>3000)
        all_configs.append({
            "name": f"K2V2+defer1000 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 2, "val_bits": 2,
            "protect_boundary": 0,
            "deferred_threshold": 1000,  # should compress at 3544-token prefix
        })

        # Deferred compression: threshold=5000 (will NOT compress — shows deferred path)
        all_configs.append({
            "name": f"K2V2+defer5000 {ep_label}",
            "evict_pct": evict_pct, "key_bits": 2, "val_bits": 2,
            "protect_boundary": 0,
            "deferred_threshold": 5000,  # should skip compression
        })

    # ------------------------------------------------------------------ handle baseline
    # The "Baseline" config needs a special path: evict_pct=0 means no eviction, but we
    # also want no quantization. We handle this by overriding evict_quantize behavior:
    # evict_pct=0 keeps all tokens, and we can set bits to 16 to represent FP16 pass-through.
    # Easier: just compute the baseline PPL directly in run_configs_on_text.
    # The baseline config in all_configs with evict_pct=0 will exercise the keep_mask=all-ones
    # path and then quantize — that is NOT the true baseline. Remove it from configs and
    # handle it in the per-text function instead (already handled above via baseline_ppl).
    # Drop the sentinel entry to avoid double-counting.
    configs_to_run = [c for c in all_configs if not c.get("_is_baseline")]

    # ------------------------------------------------------------------ run experiment
    print(f"\nRunning {len(configs_to_run)} configs on MULTI_TEXT at ~3544-token prefix")
    all_results = run_configs_on_text("MULTI-TOPIC (diverse)", MULTI_TEXT, configs_to_run)

    # ------------------------------------------------------------------ summary table
    print(f"\n{'='*80}")
    print("ASYMMETRIC K/V — FULL SUMMARY TABLE")
    print(f"{'='*80}")

    # Group by eviction rate for cleaner reading
    for evict_pct in EVICT_RATES:
        group = [r for r in all_results if r["evict_pct"] == evict_pct]
        if not group:
            continue
        print(f"\n--- Eviction rate: {evict_pct}% ---")
        print(f"{'Config':<30s} {'PPL Δ%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot L':>7s}")
        print("-" * 65)
        for r in group:
            tag = " <<<" if abs(r["delta"]) < 1.0 else (" <<" if abs(r["delta"]) < 2.0 else "")
            print(f"{r['name']:<30s} {r['delta']:+8.3f}% {r['ratio']:6.2f}x "
                  f"{r['n_kept']:5d} {r['n_protected']:6d}{tag}")

    # ------------------------------------------------------------------ best config analysis
    print(f"\n{'='*80}")
    print("BEST QUALITY/RATIO TRADEOFF ANALYSIS")
    print(f"{'='*80}")

    # For each eviction rate, rank by ratio subject to |delta| < 1%
    for evict_pct in EVICT_RATES:
        group = [r for r in all_results if r["evict_pct"] == evict_pct and not r.get("deferred")]
        sub1pct = [r for r in group if abs(r["delta"]) < 1.0]
        print(f"\nEviction {evict_pct}%:")
        if sub1pct:
            best = max(sub1pct, key=lambda r: r["ratio"])
            print(f"  Best <1% PPL: {best['name']:30s}  "
                  f"delta={best['delta']:+.3f}%  ratio={best['ratio']:.2f}x")
        else:
            closest = min(group, key=lambda r: abs(r["delta"]))
            print(f"  No config <1% — closest: {closest['name']:25s}  "
                  f"delta={closest['delta']:+.3f}%  ratio={closest['ratio']:.2f}x")

    # ------------------------------------------------------------------ asymmetric gain analysis
    print(f"\n{'='*80}")
    print("ASYMMETRIC K/V GAIN vs K2V2 BASELINE")
    print(f"{'='*80}")
    print(f"\n{'Config':<30s} {'vs K2V2 delta':>14s} {'vs K2V2 ratio':>14s}")
    print("-" * 60)
    for evict_pct in EVICT_RATES:
        baseline_k2v2 = next(
            (r for r in all_results if r["evict_pct"] == evict_pct and r["name"] == f"K2V2 {evict_pct}%ev"),
            None
        )
        if baseline_k2v2 is None:
            continue
        for r in all_results:
            if r["evict_pct"] != evict_pct or r.get("deferred"):
                continue
            if r["name"] == f"K2V2 {evict_pct}%ev":
                continue
            delta_gain = r["delta"] - baseline_k2v2["delta"]
            ratio_diff = r["ratio"] - baseline_k2v2["ratio"]
            sign_d = "+" if delta_gain > 0 else ""
            sign_r = "+" if ratio_diff > 0 else ""
            print(f"{r['name']:<30s} {sign_d}{delta_gain:+.3f}% PPL      "
                  f"{sign_r}{ratio_diff:+.2f}x ratio")

    # ------------------------------------------------------------------ boundary analysis
    print(f"\n{'='*80}")
    print("BOUNDARY PROTECTION: QUALITY GAIN vs RATIO COST")
    print(f"{'='*80}")
    print(f"\n{'Pair':<50s} {'ΔPPL%':>8s} {'ΔRatio':>8s}")
    print("-" * 70)
    for evict_pct in EVICT_RATES:
        for base_name, bnd_name in [
            (f"K2V2 {evict_pct}%ev", f"K2V2+boundary2 {evict_pct}%ev"),
            (f"K3V2 {evict_pct}%ev", f"K3V2+boundary2 {evict_pct}%ev"),
        ]:
            base_r = next((r for r in all_results if r["name"] == base_name), None)
            bnd_r  = next((r for r in all_results if r["name"] == bnd_name), None)
            if base_r and bnd_r:
                d_ppl   = bnd_r["delta"] - base_r["delta"]
                d_ratio = bnd_r["ratio"] - base_r["ratio"]
                verdict = "quality win" if d_ppl < 0 and d_ratio > -0.5 else \
                          "ratio cost too high" if d_ratio < -1.0 else "neutral"
                print(f"{base_name} → +boundary2{'':<14s} "
                      f"{d_ppl:+.3f}%   {d_ratio:+.2f}x   [{verdict}]")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("\nDone.")
    return all_results


@app.local_entrypoint()
def main():
    import time, os, json

    print("Launching asymmetric K/V experiment on Modal A100...")
    t0 = time.time()
    all_results = run_asymmetric_kv.remote()
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Persist results to a local JSON for post-analysis
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".planning", "research")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "asymmetric_kv_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_json}")

    # Print markdown table for copy-paste
    print("\n## Markdown Summary Table\n")
    print("| Config | Evict% | PPL Δ% | Ratio | Kept | Protected Layers |")
    print("|--------|--------|--------|-------|------|-----------------|")
    for r in all_results:
        tag = " <<<" if abs(r["delta"]) < 1.0 else (" <<" if abs(r["delta"]) < 2.0 else "")
        print(f"| {r['name']} | {r['evict_pct']}% | {r['delta']:+.3f}%{tag} "
              f"| {r['ratio']:.2f}x | {r['n_kept']} | {r['n_protected']} |")

"""Modal A100 experiment: combined best-config sweep on Mistral-7B.

Tests the "holy grail" combination:
  - Real attention scorer (eager attn, one extra forward pass)
  - Asymmetric K3V2 quantization (3-bit keys, 2-bit values)
  - Boundary layer protection (first/last 2 layers at FP16)
  - Eviction at 35%, 60%, 80%

Full ablation matrix:
  | Config             | Scorer   | Key bits | Val bits | Boundary | Evict rates  |
  |--------------------|----------|----------|----------|----------|--------------|
  | Baseline           | -        | -        | -        | -        | 0%           |
  | Current best       | key-key  | 2        | 2        | 0        | 35/60/80%    |
  | New best           | real     | 3        | 2        | 2        | 35/60/80%    |
  | Ablation: real+K2V2| real     | 2        | 2        | 0        | 35/60/80%    |
  | Ablation: kk+K3V2  | key-key  | 3        | 2        | 0        | 35/60/80%    |

Overhead accounting is BRUTAL — all bytes included:
  - Compressed layers: delta-zstd E8 index bytes (key_bits / value_bits separately)
  - Scale bytes: 1 fp16 per (kept_token, kv_head, K+V) per compressed layer
  - Mask bytes: ceil(prefix_len/8) * n_compressed_layers
  - Boundary layers: full FP16 for ALL prefix_len tokens (no eviction applied)

SELF-CONTAINED: no NexusQuantEvict import, all logic inlined.

Run with: modal run experiments/modal_combined_best.py
"""

import modal
import os

app = modal.App("nexusquant-combined-best")

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
    gpu="A100",
    timeout=7200,
    secrets=[HF_SECRET],
    memory=65536,
)
def run_combined_best():
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 80)
    print("NEXUSQUANT — Combined Best Config (Mistral-7B, A100)")
    print("Real scorer + K3V2 + boundary protection ablation")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name} with attn_implementation='eager'...")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

    # CRITICAL: eager required to get real attention weights.
    # SDPA returns None for output_attentions regardless of the flag.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_layers   = model.config.num_hidden_layers           # 32
    n_kv_heads = model.config.num_key_value_heads         # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, "rope_theta", 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads} KV heads, head_dim={head_dim}, rope_base={rope_base}")

    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"attn_implementation: {attn_impl}")
    assert attn_impl == "eager", f"Expected eager, got {attn_impl!r}"

    # ------------------------------------------------------------------
    # Build ~3544-token prefix (same corpus as validated asymmetric_kv exp)
    # ------------------------------------------------------------------
    MULTI_TEXT = " ".join([
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists around the world, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. Fermions are the building blocks: six quarks and six leptons. Forces between the fermions are mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. Despite its success, the Standard Model does not incorporate gravity, dark matter, or dark energy, leaving physicists with significant open questions about the nature of the universe.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation. It began in Britain and quickly spread to Western Europe and North America. The transition from hand production methods to machine manufacturing, new chemical processes, iron production, increased use of steam power, the development of machine tools, and the rise of the factory system fundamentally changed the nature of work and society. This period saw the emergence of the middle class, the growth of cities, and the beginning of modern capitalism. Child labor was common in factories and mines, prompting early labor reforms.",
        "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better suited to their environment will tend to survive and produce more offspring. Over time, these advantageous traits become more common in the population. Genetic variation arises through mutation, recombination, and gene flow. Sexual selection is a special case where traits improve mating success rather than survival directly. Evolutionary theory is supported by evidence from the fossil record, comparative anatomy, molecular biology, and direct observation of evolution in real time.",
        "The development of quantum mechanics in the early 20th century fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain phenomena such as black-body radiation, the photoelectric effect, or the stability of atoms. Max Planck introduced the concept of energy quanta in 1900. Albert Einstein explained the photoelectric effect using photons in 1905. Niels Bohr developed his model of the hydrogen atom in 1913. Werner Heisenberg, Erwin Schrodinger, Paul Dirac, and others built the mathematical framework of modern quantum mechanics throughout the 1920s.",
        "Mathematics has been essential to the development of science and technology throughout human history. From the ancient Babylonians who developed a base-60 number system that we still use for measuring time, to the development of calculus by Newton and Leibniz that made modern physics possible, mathematical ideas have been the foundation of scientific progress. Abstract algebra, topology, and functional analysis developed largely for theoretical reasons later turned out to have profound physical applications. The unreasonable effectiveness of mathematics in the natural sciences remains a philosophical puzzle.",
        "The history of astronomy represents one of humanity's oldest scientific endeavors. Ancient civilizations observed the stars for navigation, agriculture, and religious purposes. The Babylonians recorded planetary positions, the Greeks developed geometric models of the cosmos, and Islamic astronomers preserved and extended this knowledge during the medieval period. Copernicus proposed the heliocentric model. Galileo used the telescope to observe Jupiter's moons, the phases of Venus, and sunspots. Newton derived his law of universal gravitation. Modern astronomy covers everything from exoplanet detection to gravitational wave observation.",
        "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence was felt in literature, philosophy, art, music, politics, science, religion, and other aspects of intellectual inquiry. The rediscovery of ancient Greek and Roman texts inspired new ways of thinking about human nature, government, and the natural world. Leonardo da Vinci, Michelangelo, Raphael, and Botticelli represent the artistic pinnacle of the movement. The printing press accelerated the dissemination of ideas across Europe.",
        "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons connected by trillions of synapses. Each neuron can form thousands of connections with other neurons, creating an intricate network that gives rise to thought, memory, emotion, and consciousness. The brain consumes roughly 20 percent of the body's energy despite comprising only about 2 percent of its mass. Neuroplasticity allows the brain to reorganize itself in response to experience. Modern neuroscience uses tools such as fMRI, EEG, and optogenetics to study how neural circuits give rise to behavior.",
        "Climate change represents one of the most significant challenges facing humanity in the 21st century. The burning of fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of greenhouse gases, particularly carbon dioxide and methane, to levels unprecedented in at least 800,000 years. Global average temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels. Consequences include more frequent extreme weather events, rising sea levels, ocean acidification, and disruption of ecosystems. The Paris Agreement aims to limit warming to 1.5 to 2 degrees Celsius through international cooperation.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. It had a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around 100 BC to 400 AD. Roman engineering achievements included aqueducts, roads, concrete construction, and the Pantheon. Roman law formed the basis of many modern legal systems. Latin evolved into the Romance languages: Italian, Spanish, French, Portuguese, and Romanian.",
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data without being explicitly programmed. Deep learning uses artificial neural networks with many layers to learn hierarchical representations of data. Transformer architectures, introduced by Vaswani et al. in 2017, revolutionized natural language processing and have since been applied to vision, audio, protein structure prediction, and many other domains.",
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of the Earth's lithosphere. Tectonic plates move because of the relative density of oceanic lithosphere and the relative weakness of the asthenosphere. Convection currents in the mantle drive the motion of plates. Where plates converge, one may subduct beneath the other, producing volcanism and earthquakes. Where plates diverge, new oceanic crust forms at mid-ocean ridges. The theory explains the distribution of earthquakes, volcanoes, mountain ranges, and ocean trenches.",
        "The development of antibiotics in the 20th century represents one of medicine's greatest achievements. Alexander Fleming discovered penicillin in 1928, and its clinical use began in the 1940s. Antibiotics have saved hundreds of millions of lives by treating bacterial infections that were previously often fatal. However, the overuse and misuse of antibiotics has driven the emergence of antibiotic-resistant bacteria, posing a growing threat to public health. Methicillin-resistant Staphylococcus aureus and carbapenem-resistant Enterobacteriaceae are among the most dangerous resistant pathogens.",
        "The structure of DNA, elucidated by Watson and Crick in 1953 using X-ray crystallography data from Franklin and Wilkins, revealed how genetic information is stored and replicated. The double helix consists of two antiparallel strands of nucleotides connected by hydrogen bonds between complementary base pairs: adenine with thymine, guanine with cytosine. During cell division the strands separate and each serves as a template for the synthesis of a new complementary strand. The central dogma of molecular biology describes how information flows from DNA to RNA to protein.",
        "The renormalization group in quantum field theory provides a systematic framework for understanding how physical theories change with the energy scale of observation. Kenneth Wilson's formulation connects statistical mechanics and quantum field theory through the concept of universality classes, where disparate physical systems exhibit identical critical behavior near phase transitions due to shared symmetry properties and dimensionality. The renormalization group explains why microscopic details are irrelevant to macroscopic critical phenomena and provides a mathematical framework for computing critical exponents.",
        "Homological algebra studies algebraic structures through chain complexes and their derived functors. The Ext and Tor functors provide fundamental invariants that measure the failure of exactness under the Hom and tensor product functors respectively. Spectral sequences provide computational tools for successive approximation of these derived functors. The derived category of an abelian category, constructed by formally inverting quasi-isomorphisms, provides a more natural setting for derived functor computations and connects to deformation theory and algebraic geometry.",
        "The Langlands program represents one of the most ambitious unifying frameworks in mathematics, connecting number theory, algebraic geometry, and representation theory. The geometric Langlands correspondence establishes deep connections between automorphic forms on reductive groups and l-adic representations of absolute Galois groups over function fields. Recent work by Fargues and Scholze on p-adic geometry has brought new tools to bear on the local Langlands correspondence.",
        "CRISPR-Cas9 gene editing exploits the bacterial adaptive immune system's ability to incorporate foreign DNA fragments into clustered regularly interspaced short palindromic repeats. The Cas9 endonuclease, guided by a chimeric single-guide RNA, introduces double-strand breaks at specific genomic loci, enabling precise modifications through nonhomologous end joining or homology-directed repair pathways. Base editing and prime editing extend CRISPR capabilities to single-nucleotide changes without double-strand breaks. Clinical trials using CRISPR to treat sickle-cell disease have shown promising results.",
        "The AdS/CFT correspondence, proposed by Juan Maldacena in 1997, conjectures an exact duality between type IIB superstring theory on five-dimensional anti-de Sitter space times a five-sphere and four-dimensional N=4 super Yang-Mills theory on the conformal boundary. This holographic principle has profound implications for quantum gravity and strongly coupled quantum field theories. The duality maps the bulk gravitational degrees of freedom to boundary conformal field theory data and has been used to compute properties of quark-gluon plasma and quantum entanglement entropy.",
        "The discovery of gravitational waves in 2015 by LIGO opened a new observational window on the universe. The merger of two black holes 1.3 billion light-years away produced a signal lasting a fraction of a second, causing spacetime distortions smaller than one-thousandth the diameter of a proton. Since then, dozens of gravitational wave events have been detected, including neutron star mergers that produced coincident electromagnetic counterparts and allowed independent measurements of the Hubble constant.",
        "Transformer language models scale remarkably well with compute. The scaling laws of Kaplan et al. showed that model performance on language modeling follows a power law with model size, dataset size, and compute budget, each contributing roughly equally when optimally balanced. Chinchilla scaling revised the optimal compute allocation to require larger datasets relative to model size. Models trained on trillions of tokens with billions of parameters exhibit emergent capabilities not present in smaller models, such as few-shot reasoning, code generation, and instruction following.",
        "Topological insulators are materials that behave as insulators in their bulk but support conducting states on their surfaces or edges that are protected by time-reversal symmetry. These surface states arise from band inversions driven by strong spin-orbit coupling and are topologically distinct from ordinary insulators. The quantum spin Hall effect in two-dimensional topological insulators produces helical edge states where spin and momentum are locked together, making them robust against non-magnetic disorder.",
        "Optimal transport theory provides a mathematical framework for comparing probability distributions by finding the most efficient way to transform one distribution into another. The Wasserstein distance, defined as the minimum cost of transporting mass from one distribution to another, has found applications in machine learning for generative modeling, domain adaptation, and fairness. The Monge-Kantorovich duality connects the primal transport problem to a dual problem involving potential functions, enabling efficient computational algorithms through entropic regularization and the Sinkhorn algorithm.",
        "The microbiome, the community of trillions of microorganisms living in and on the human body, plays a crucial role in health and disease. The gut microbiome aids in digestion, synthesizes vitamins, trains the immune system, and produces neuroactive compounds that influence mood and behavior through the gut-brain axis. Dysbiosis, disruption of the microbial community, has been associated with inflammatory bowel disease, obesity, type 2 diabetes, colorectal cancer, and neuropsychiatric disorders. Fecal microbiota transplantation is an effective treatment for recurrent Clostridioides difficile infection.",
    ])

    # ------------------------------------------------------------------
    # Tokenize — target 3544-token prefix
    # ------------------------------------------------------------------
    inputs = tok(MULTI_TEXT, return_tensors="pt", max_length=8192, truncation=True)
    full_ids = inputs.input_ids.to(model.device)
    n_tok = full_ids.shape[1]

    target_prefix = 3544
    prefix_len = min(target_prefix, n_tok - 100)
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

    # ------------------------------------------------------------------
    # Scorers
    # ------------------------------------------------------------------
    def score_key_key(kv_cache, obs_window=32):
        """SnapKV-style key-key proxy: all layers, causal masking, avg-pooled."""
        all_imp = torch.zeros(prefix_len, device="cpu")
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()  # (n_kv_heads, seq, head_dim)
            w = min(obs_window, prefix_len)
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            # causal mask: each obs position can only attend to earlier positions
            all_pos = torch.arange(prefix_len).unsqueeze(0)
            obs_pos = torch.arange(prefix_len - w, prefix_len).unsqueeze(1)
            causal = all_pos <= obs_pos
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)  # (seq,)
            all_imp += layer_imp
        return (all_imp / n_layers).to(model.device)  # (seq,)

    def score_real(prefix_ids):
        """Real attention scorer: accumulated softmax weights from ALL layers.

        Runs a single forward pass with output_attentions=True (requires eager).
        For each layer: sum attention weights column-wise (how much each token
        is attended to), average over heads. Sum across layers.
        Result: importance score per token, shape (prefix_len,).
        """
        print("  [real scorer] Running extra forward pass with output_attentions=True...")
        t0 = time.time()
        with torch.no_grad():
            out = model(
                prefix_ids,
                output_attentions=True,
                use_cache=False,
            )
        # out.attentions: tuple of (n_layers,) each (batch, n_heads, seq, seq)
        # For GQA models like Mistral-7B: n_heads=32, but KV heads=8;
        # attention weights are still (batch, n_heads, seq, seq).
        importance = torch.zeros(prefix_len, device=model.device)
        for layer_attn in out.attentions:
            if layer_attn is None:
                continue
            # layer_attn: (1, n_heads, seq, seq) — lower triangle (causal)
            # Sum over query dimension (dim=2) = how much each key token was used
            # Then average over heads
            col_sum = layer_attn[0].sum(dim=1)  # (n_heads, seq) — query-summed
            importance += col_sum.mean(dim=0)   # (seq,)
        del out
        torch.cuda.empty_cache()
        print(f"  [real scorer] Done in {time.time() - t0:.1f}s  "
              f"min={importance.min():.4f} max={importance.max():.4f}")
        return importance  # (prefix_len,)

    # ------------------------------------------------------------------
    # Keep mask builder
    # ------------------------------------------------------------------
    SLIDING_WINDOW = 32

    def build_keep_mask(importance, eviction_rate):
        """BOS + top-(1-eviction_rate) tokens + sliding window always kept."""
        seq = importance.shape[0]
        device = importance.device
        prefix_end = max(seq - SLIDING_WINDOW, 0)
        n_keep = max(1, int(prefix_end * (1.0 - eviction_rate)))

        mask = torch.zeros(seq, dtype=torch.bool, device=device)
        mask[0] = True                                       # BOS always
        if SLIDING_WINDOW > 0:
            mask[max(0, seq - SLIDING_WINDOW):] = True      # sliding window

        if prefix_end > 1 and n_keep > 0:
            prefix_scores = importance[1:prefix_end].clone()
            topk_k = min(n_keep, prefix_scores.shape[0])
            if topk_k > 0:
                _, idx = torch.topk(prefix_scores, k=topk_k)
                mask[idx + 1] = True                         # +1: offset for BOS
        return mask

    # ------------------------------------------------------------------
    # Core evict + quantize (asymmetric K/V + boundary protection)
    # ------------------------------------------------------------------
    def evict_and_quantize(kv_cache, keep_mask, key_bits=2, value_bits=2, protect_boundary=0):
        """Evict tokens then E8-quantize survivors. Boundary layers stay FP16.

        Overhead accounting (bytes, ALL included — no cherry-picking):
          - Compressed layers: delta-zstd E8 index bytes (key_bits / value_bits)
          - Scale bytes: n_kept * n_kv_heads * 2 (K+V) * 2 (fp16) per compressed layer
          - Mask bytes: ceil(prefix_len/8) * n_compressed_layers
          - Boundary layers: full FP16 for all prefix_len tokens (no compression)
        """
        H = hadamard_matrix(head_dim).cpu()
        cctx = zstandard.ZstdCompressor(level=22)

        n_kept = keep_mask.sum().item()
        keep_mask_cpu = keep_mask.cpu()

        # Identify protected vs compressed layers
        protected_layers = set()
        if protect_boundary > 0:
            for i in range(min(protect_boundary, n_layers)):
                protected_layers.add(i)
            for i in range(max(0, n_layers - protect_boundary), n_layers):
                protected_layers.add(i)
        n_compressed_layers = n_layers - len(protected_layers)

        total_fp16_bytes = 0
        all_key_coords = []
        all_val_coords = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16_bytes += k.numel() * 2 + v.numel() * 2

            if l in protected_layers:
                # No quantization, no eviction — tensors stay as FP16 on GPU
                continue

            # Per-head eviction + quantization
            k_t = k[0].clone()  # (n_kv_heads, seq, head_dim)
            v_t = v[0].clone()

            for is_key, tensor, bits, coord_list in [
                (True,  k_t, key_bits,   all_key_coords),
                (False, v_t, value_bits, all_val_coords),
            ]:
                levels = 2 ** bits
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(tensor[h:h+1], base=rope_base)[0]
                    else:
                        t_head = tensor[h]

                    kept_data = t_head[keep_mask_cpu]           # (n_kept, head_dim)
                    rotated   = kept_data @ H.T                 # Hadamard

                    # Per-token scale (E8 quantization)
                    amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc   = amax / (levels / 2)
                    normalized = rotated / sc
                    groups = normalized.reshape(-1, 8)
                    lp     = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords = lp.reshape(-1, head_dim)
                    quantized = (coords * sc) @ H               # inv-Hadamard

                    # Store integer coords for compression ratio accounting
                    int_c = coords.detach().numpy()
                    has_half = np.any(
                        np.abs(int_c.flatten() - np.round(int_c.flatten())) > 0.25
                    )
                    if has_half:
                        coord_list.append(np.round(int_c.flatten() * 2).astype(np.int8))
                    else:
                        coord_list.append(np.round(int_c.flatten()).astype(np.int8))

                    # Write back: zero evicted positions, re-RoPE keys
                    result = torch.zeros_like(t_head)
                    result[keep_mask_cpu] = quantized
                    if is_key:
                        tensor[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        tensor[h] = result

            set_kv(kv_cache, l, k_t.unsqueeze(0).half().to(model.device), vl)
            kl_now, _ = get_kv(kv_cache, l)
            set_kv(kv_cache, l, kl_now, v_t.unsqueeze(0).half().to(model.device))

        # ------------------------------------------------------------------
        # Compression ratio accounting (honest — all overhead)
        # ------------------------------------------------------------------
        total_idx_bytes = 0
        for coords_arr in all_key_coords + all_val_coords:
            arr = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped = arr.reshape(n_kept, n_per_tok)
                delta = np.zeros_like(reshaped)
                delta[0]  = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx_bytes += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx_bytes += len(cctx.compress(arr.tobytes()))

        # Scale: 1 fp16 per (kept_token, head) for K and V, per compressed layer
        scale_bytes = n_kept * n_kv_heads * 2 * 2 * n_compressed_layers

        # Mask: 1 shared boolean mask per compressed layer
        mask_bytes = math.ceil(prefix_len / 8) * n_compressed_layers

        # Boundary layers: full FP16, no eviction applied
        boundary_fp16_bytes = len(protected_layers) * prefix_len * n_kv_heads * head_dim * 2 * 2

        total_compressed = total_idx_bytes + scale_bytes + mask_bytes + boundary_fp16_bytes
        ratio = total_fp16_bytes / total_compressed if total_compressed > 0 else 0.0

        return kv_cache, {
            "fp16_bytes":      total_fp16_bytes,
            "idx_bytes":       total_idx_bytes,
            "scale_bytes":     scale_bytes,
            "mask_bytes":      mask_bytes,
            "boundary_bytes":  boundary_fp16_bytes,
            "total_bytes":     total_compressed,
            "ratio":           ratio,
            "n_kept":          n_kept,
            "n_protected":     len(protected_layers),
            "n_compressed":    n_compressed_layers,
        }

    # ------------------------------------------------------------------
    # PPL evaluator
    # ------------------------------------------------------------------
    def compute_ppl(kv_cache, keep_mask):
        """Continuation PPL given a (possibly compressed) KV cache."""
        attn_ctx = keep_mask.long().to(model.device)
        attn_full = torch.cat([
            attn_ctx,
            torch.ones(cont_len, dtype=torch.long, device=model.device),
        ]).unsqueeze(0)

        with torch.no_grad():
            out = model(
                full_ids[:, prefix_len:],
                past_key_values=kv_cache,
                attention_mask=attn_full,
                use_cache=True,
            )
        logits  = out.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return torch.exp(loss).item()

    # ------------------------------------------------------------------
    # Baseline PPL (no compression)
    # ------------------------------------------------------------------
    print("\nComputing baseline PPL (no compression)...")
    with torch.no_grad():
        bl_out  = model(prefix_ids, use_cache=True)
        bl_full_mask = torch.ones(1, n_tok, dtype=torch.long, device=model.device)
        bl_cont = model(
            full_ids[:, prefix_len:],
            past_key_values=bl_out.past_key_values,
            attention_mask=bl_full_mask,
            use_cache=True,
        )
    bl_logits  = bl_cont.logits[:, :-1, :].float()
    bl_targets = full_ids[:, prefix_len + 1:].contiguous()
    bl_loss    = F.cross_entropy(bl_logits.reshape(-1, bl_logits.shape[-1]), bl_targets.reshape(-1))
    baseline_ppl = torch.exp(bl_loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del bl_out, bl_cont
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pre-compute importance scores
    # ------------------------------------------------------------------
    print("\nPre-computing REAL attention scores (extra forward pass)...")
    real_importance = score_real(prefix_ids)   # (prefix_len,) on GPU

    print("\nPre-computing KEY-KEY proxy scores (prefill forward pass)...")
    with torch.no_grad():
        pf_out = model(prefix_ids, use_cache=True)
    kk_importance = score_key_key(pf_out.past_key_values)
    del pf_out
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Config matrix
    # ------------------------------------------------------------------
    EVICTION_RATES = [0.35, 0.60, 0.80]

    configs = []
    for evict in EVICTION_RATES:
        ep = f"{int(evict*100)}%"

        # Current best: key-key + K2V2 + no boundary
        configs.append({
            "name":             f"current_best  kk+K2V2       {ep}",
            "scorer":           "key-key",
            "key_bits":         2,
            "value_bits":       2,
            "protect_boundary": 0,
            "eviction_rate":    evict,
        })

        # New best: real + K3V2 + boundary-2
        configs.append({
            "name":             f"NEW_BEST       real+K3V2+b2  {ep}",
            "scorer":           "real",
            "key_bits":         3,
            "value_bits":       2,
            "protect_boundary": 2,
            "eviction_rate":    evict,
        })

        # Ablation: real scorer + K2V2 (isolates scorer effect)
        configs.append({
            "name":             f"ablation       real+K2V2     {ep}",
            "scorer":           "real",
            "key_bits":         2,
            "value_bits":       2,
            "protect_boundary": 0,
            "eviction_rate":    evict,
        })

        # Ablation: key-key + K3V2 (isolates asymmetric bits effect)
        configs.append({
            "name":             f"ablation       kk+K3V2       {ep}",
            "scorer":           "key-key",
            "key_bits":         3,
            "value_bits":       2,
            "protect_boundary": 0,
            "eviction_rate":    evict,
        })

    # ------------------------------------------------------------------
    # Run all configs
    # ------------------------------------------------------------------
    print(f"\n{'Config':<45} {'PPL':>9} {'Delta%':>8} {'Ratio':>7} {'Kept':>6} {'Prot':>5}")
    print("-" * 82)

    all_results = []

    for cfg in configs:
        torch.cuda.empty_cache()

        scorer           = cfg["scorer"]
        key_bits         = cfg["key_bits"]
        value_bits       = cfg["value_bits"]
        protect_boundary = cfg["protect_boundary"]
        eviction_rate    = cfg["eviction_rate"]
        name             = cfg["name"]

        importance = real_importance if scorer == "real" else kk_importance
        keep_mask  = build_keep_mask(importance, eviction_rate)

        # Fresh prefill for each run (don't share KV state between configs)
        with torch.no_grad():
            pf_out = model(prefix_ids, use_cache=True)
            kv = pf_out.past_key_values

        kv, overhead = evict_and_quantize(
            kv, keep_mask,
            key_bits=key_bits,
            value_bits=value_bits,
            protect_boundary=protect_boundary,
        )

        ppl   = compute_ppl(kv, keep_mask)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100

        tag = " <<<" if abs(delta) < 0.5 else (" <<" if abs(delta) < 1.0 else ("  <" if abs(delta) < 2.0 else ""))
        print(f"{name:<45} {ppl:9.4f} {delta:+8.3f}% {overhead['ratio']:6.2f}x "
              f"{overhead['n_kept']:5d} {overhead['n_protected']:4d}{tag}")

        all_results.append({
            "name":             name,
            "scorer":           scorer,
            "key_bits":         key_bits,
            "value_bits":       value_bits,
            "protect_boundary": protect_boundary,
            "eviction_rate":    eviction_rate,
            "ppl":              ppl,
            "delta":            delta,
            "ratio":            overhead["ratio"],
            "n_kept":           overhead["n_kept"],
            "n_protected":      overhead["n_protected"],
            "baseline_ppl":     baseline_ppl,
            "prefix_len":       prefix_len,
            "overhead":         overhead,
        })

        del kv
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary tables grouped by eviction rate
    # ------------------------------------------------------------------
    print(f"\n{'='*82}")
    print(f"COMBINED BEST — FULL SUMMARY  (Mistral-7B, prefix={prefix_len}, baseline={baseline_ppl:.4f})")
    print(f"{'='*82}")

    for evict in EVICTION_RATES:
        group = [r for r in all_results if r["eviction_rate"] == evict]
        print(f"\n--- Eviction {int(evict*100)}% ---")
        print(f"{'Config':<45} {'PPL Δ%':>8} {'Ratio':>7} {'Kept':>6} {'Prot':>5}")
        print("-" * 74)
        for r in group:
            tag = " <<<" if abs(r["delta"]) < 0.5 else (" <<" if abs(r["delta"]) < 1.0 else ("  <" if abs(r["delta"]) < 2.0 else ""))
            print(f"{r['name']:<45} {r['delta']:+8.3f}% {r['ratio']:6.2f}x "
                  f"{r['n_kept']:5d} {r['n_protected']:4d}{tag}")

    # ------------------------------------------------------------------
    # Component contribution analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*82}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*82}")
    print("Marginal gain of each component vs current_best at same eviction rate:\n")
    print(f"{'Evict':>6}  {'real scorer':>12}  {'K3V2 bits':>10}  {'boundary-2':>11}  {'combined':>10}")
    print("-" * 60)

    for evict in EVICTION_RATES:
        def find(sc, kb, vb, pb):
            for r in all_results:
                if (r["eviction_rate"] == evict and r["scorer"] == sc
                        and r["key_bits"] == kb and r["value_bits"] == vb
                        and r["protect_boundary"] == pb):
                    return r
            return None

        base   = find("key-key", 2, 2, 0)
        r_sc   = find("real",    2, 2, 0)   # scorer only
        r_bits = find("key-key", 3, 2, 0)   # bits only
        r_new  = find("real",    3, 2, 2)   # full new best

        if not all([base, r_sc, r_bits, r_new]):
            continue

        gain_sc   = r_sc["delta"]   - base["delta"]
        gain_bits = r_bits["delta"] - base["delta"]
        gain_new  = r_new["delta"]  - base["delta"]

        print(f"{int(evict*100):>5}%  "
              f"{gain_sc:>+10.3f}pp  "
              f"{gain_bits:>+8.3f}pp  "
              f"{'(incl. boundary)':>11}  "
              f"{gain_new:>+8.3f}pp")

    # ------------------------------------------------------------------
    # Best config per quality tier
    # ------------------------------------------------------------------
    print(f"\n{'='*82}")
    print("BEST CONFIG PER QUALITY TIER")
    print(f"{'='*82}")

    tiers = [
        ("ultra-high (<0.5% PPL)",  0.5),
        ("high      (<1.0% PPL)",   1.0),
        ("balanced  (<2.0% PPL)",   2.0),
    ]
    for tier_name, threshold in tiers:
        candidates = [r for r in all_results if abs(r["delta"]) < threshold]
        if candidates:
            best = max(candidates, key=lambda r: r["ratio"])
            print(f"\n{tier_name}:")
            print(f"  Best: {best['name']}")
            print(f"  PPL delta={best['delta']:+.3f}%  ratio={best['ratio']:.2f}x  "
                  f"evict={int(best['eviction_rate']*100)}%  "
                  f"key_bits={best['key_bits']}  val_bits={best['value_bits']}  "
                  f"boundary={best['protect_boundary']}")
        else:
            print(f"\n{tier_name}: no config achieves this threshold")

    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return all_results


@app.local_entrypoint()
def main():
    print("Launching combined best-config experiment on Modal A100...")
    results = run_combined_best.remote()

    print("\n=== LOCAL SUMMARY ===")
    for r in results:
        print(f"  {r['name']:<45}  PPL={r['ppl']:.4f}  delta={r['delta']:+.2f}%  "
              f"ratio={r['ratio']:.2f}x  kept={r['n_kept']}")

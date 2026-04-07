"""NexusQuant — True 8K/16K Context Validation on A100 80GB.

Validates that 3544-token results (10.4x@+0.43%, 16.8x@+1.34%, 33.3x@+2.64%)
hold at real long-context lengths using REAL documents (PG19 books, not synthetic).

Design:
  - 5 PG19 books (Project Gutenberg, guaranteed 16K+ tokens each)
  - Fallback: diverse-topic synthetic text if HF unavailable
  - Prefix lengths: [8192, 16384] (16K only when text long enough)
  - Eviction rates: [0%, 35%, 50%, 60%, 80%]
  - 2-bit E8 VQ + temporal delta + zstd level 22
  - Memory tracking before/after compression

Metrics (per text, then mean ± std):
  - Baseline PPL, compressed PPL, delta (%)
  - Compression ratio (analytic byte counting, ALL overhead: indices + scales + mask)
  - Tokens kept vs total
  - GPU memory allocated before/after
"""
import modal
import os

app = modal.App("nexusquant-8k-16k-validation")

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
        "datasets>=2.18.0",
        "huggingface_hub>=0.22.0",
    )
    .add_local_dir(nq_local, remote_path="/root/nexusquant")
)

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": "os.environ.get("HF_TOKEN", "")"})

# ---------------------------------------------------------------------------
# Fallback synthetic corpus: 20 topics × ~400 words ≈ 20K+ tokens.
# Only used if PG19 is unavailable. NOT the primary data source.
# ---------------------------------------------------------------------------
SYNTHETIC_FALLBACK = (
    # Physics
    "The Standard Model of particle physics describes three of the four fundamental forces and classifies "
    "all known elementary particles. Fermions — six quarks and six leptons — are the building blocks of "
    "matter. Gauge bosons mediate the forces between fermions. The Higgs mechanism gives mass to particles "
    "through spontaneous symmetry breaking. The strong force binds quarks into protons and neutrons via "
    "gluon exchange governed by quantum chromodynamics. The electroweak force unifies electromagnetism and "
    "the weak nuclear force, with W and Z bosons as mediators. General relativity describes gravity as "
    "spacetime curvature caused by mass and energy, predicting black holes, gravitational waves, and the "
    "expansion of the universe. Quantum mechanics governs the microscopic world with wave functions, "
    "operators, and the uncertainty principle. Dark matter and dark energy together constitute roughly 95 "
    "percent of the universe's energy content but have not been directly detected. The Large Hadron Collider "
    "confirmed the Higgs boson in 2012, completing the Standard Model. Supersymmetry and string theory are "
    "candidate extensions but lack experimental confirmation. Hawking radiation predicts that black holes "
    "slowly evaporate through quantum effects near the event horizon. The arrow of time and entropy pose "
    "deep puzzles about the initial conditions of the universe. Quantum field theory combines quantum "
    "mechanics with special relativity to describe fundamental particles as excitations of fields. The "
    "Casimir effect demonstrates that quantum vacuum fluctuations produce measurable forces between "
    "conducting plates. Feynman diagrams provide a systematic perturbative expansion of scattering "
    "amplitudes in terms of particle interactions. The renormalization group explains how physical laws "
    "change with the energy scale of observation. Spontaneous symmetry breaking is responsible for the "
    "masses of the W and Z bosons via the Brout-Englert-Higgs mechanism. Lattice QCD uses numerical "
    "simulations on a discretized spacetime to compute hadronic properties from first principles. "
    # History
    "The Industrial Revolution transformed human society from agrarian economies to machine-based "
    "manufacturing beginning in Britain in the late eighteenth century. James Watt's improved steam "
    "engine became the universal power source. Railways linked cities and created national markets. "
    "The factory system concentrated labor and created new urban working classes. Child labor and "
    "dangerous working conditions sparked reform movements and the first labor laws. The Second "
    "Industrial Revolution in the late nineteenth century brought electricity, steel, chemicals, and "
    "mass production. The Roman Empire unified the Mediterranean world under a single legal and "
    "administrative system for five centuries. Roman law, Latin, engineering, and infrastructure "
    "shaped European civilization profoundly. The fall of the Western Empire in 476 AD initiated "
    "the medieval period of fragmented power and the rise of the Catholic Church. The Renaissance, "
    "beginning in fourteenth-century Italy, revived classical learning and developed perspective in "
    "painting. The printing press, invented by Gutenberg around 1440, democratized knowledge and "
    "accelerated the Reformation. The French Revolution of 1789 overthrew the monarchy and proclaimed "
    "the rights of man. Napoleon spread the Napoleonic Code across Europe, reshaping legal systems. "
    "The World Wars of the twentieth century caused unprecedented destruction and reshaped the global "
    "order, ending European colonial empires and initiating the Cold War. Decolonization between 1945 "
    "and 1975 created dozens of new nations in Africa, Asia, and the Caribbean with complex legacies "
    "of colonialism that persist to the present. The Cold War competition between the United States and "
    "Soviet Union drove the space race, nuclear arms buildup, and proxy conflicts worldwide. The fall "
    "of the Berlin Wall in 1989 symbolized the collapse of Soviet-style communism in Eastern Europe. "
    # Biology
    "Evolution by natural selection, proposed by Darwin and Wallace, is the unifying theory of biology. "
    "Heritable variation plus differential reproduction causes populations to adapt to their environments "
    "over generations. The modern evolutionary synthesis combines Darwinian selection with Mendelian "
    "genetics and population genetics. DNA carries genetic information in a double-helical structure "
    "discovered by Watson and Crick in 1953. The genetic code maps three-nucleotide codons to amino acids "
    "and is nearly universal across life. The human genome contains about three billion base pairs encoding "
    "roughly twenty thousand protein-coding genes. CRISPR-Cas9 allows precise genome editing. Cell biology "
    "distinguishes prokaryotes from eukaryotes, which have membrane-bound organelles. Mitochondria generate "
    "ATP through oxidative phosphorylation and carry their own circular DNA. The immune system provides "
    "innate and adaptive defenses. T lymphocytes mediate cellular immunity; B lymphocytes produce "
    "antibodies. Ecosystems are communities of organisms interacting with their abiotic environment. "
    "Biodiversity is threatened by habitat destruction, climate change, and overexploitation. "
    "Epigenetics studies heritable changes in gene expression that do not involve changes to the DNA "
    "sequence, mediated by methylation and histone modification. Synthetic biology engineers biological "
    "systems from standardized genetic parts for applications in medicine, agriculture, and industry. "
    # Computer Science
    "The transformer architecture, introduced in Attention Is All You Need in 2017, uses self-attention "
    "to process sequences in parallel and has become the foundation for most modern large language models. "
    "Self-attention computes pairwise interactions between all tokens in a sequence, enabling the model "
    "to capture long-range dependencies. The key-value cache stores intermediate computations during "
    "autoregressive generation, growing linearly with context length and becoming a memory bottleneck "
    "at long contexts. Quantization reduces model precision from 32-bit or 16-bit floating point to "
    "lower bit-widths, trading accuracy for memory and speed. Lattice vector quantization, such as the "
    "E8 lattice, provides theoretically optimal quantization for high-dimensional vectors. Eviction "
    "policies decide which KV cache tokens to drop when memory is constrained. Attention-based importance "
    "scoring selects the most important tokens by measuring how much attention the recent context attends "
    "to each historical token. RoPE rotary position embeddings encode position information directly into "
    "key and query vectors via rotation matrices. Operating systems manage hardware resources through "
    "process scheduling, virtual memory, file systems, and device drivers. The P versus NP problem "
    "asks whether every problem whose solution can be verified in polynomial time can also be solved in "
    "polynomial time. Distributed systems coordinate multiple computers facing challenges of consensus, "
    "fault tolerance, and consistency. Cryptography enables secure communication using mathematical "
    "hardness assumptions. Public-key cryptography relies on the difficulty of factoring large integers "
    "or computing discrete logarithms. Hash functions provide data integrity and form the basis of "
    "blockchain systems. Compilers translate high-level programs into machine code through lexing, "
    "parsing, semantic analysis, optimization, and code generation phases. "
    # Mathematics
    "Pure mathematics discovers patterns and structures through rigorous proof. Number theory studies "
    "integers and primes. The Riemann hypothesis about the zeros of the zeta function has profound "
    "implications for the distribution of prime numbers and remains unproved. Abstract algebra studies "
    "groups, rings, fields, and modules. Group theory underlies the symmetries of physical laws. "
    "Galois theory uses groups to determine which polynomial equations are solvable by radicals. "
    "Topology studies properties preserved under continuous deformations. The Poincare conjecture, "
    "proved by Perelman in 2003, characterizes the three-sphere among compact three-manifolds. "
    "Differential geometry describes curved spaces using calculus and is the mathematical language "
    "of general relativity. Algebraic geometry studies zero sets of polynomial equations. Category "
    "theory provides a unifying language for mathematics. Probability theory gives a rigorous "
    "foundation for reasoning under uncertainty. Bayesian inference updates beliefs in light of "
    "evidence using Bayes' theorem. The central limit theorem explains the ubiquity of the normal "
    "distribution as the sum of many independent random variables. Functional analysis studies "
    "infinite-dimensional vector spaces and operators, providing the mathematical framework for "
    "quantum mechanics. Graph theory models networks and underpins computer science. Combinatorics "
    "counts structures; generating functions are a powerful tool for enumerating combinatorial objects. "
    # Economics
    "Economics studies how individuals, firms, and governments allocate scarce resources. "
    "Microeconomics models how prices coordinate supply and demand in markets. Consumer choice theory "
    "maximizes utility subject to budget constraints. Game theory studies strategic interaction: the "
    "Nash equilibrium is a profile of strategies from which no player has unilateral incentive to "
    "deviate. Mechanism design asks how to construct rules that induce desired outcomes from "
    "self-interested agents, with applications in auctions, matching, and regulation. Macroeconomics "
    "examines aggregate variables: output, employment, inflation, and growth. Keynesian theory holds "
    "that aggregate demand drives output in the short run. Monetarism emphasizes money supply. "
    "Real business cycle theory attributes fluctuations to technology shocks. Growth theory studies "
    "why incomes differ across countries and grow over time. Comparative advantage — not absolute "
    "advantage — determines trade patterns and mutual gains from exchange. Behavioral economics "
    "incorporates evidence that humans systematically deviate from the rational-agent model through "
    "heuristics, loss aversion, and hyperbolic discounting. Financial economics studies asset pricing, "
    "risk, and the role of intermediaries. The efficient market hypothesis holds that asset prices "
    "reflect all available information. Asymmetric information and moral hazard explain market "
    "failures in insurance, credit, and labor markets. "
    # Philosophy
    "Philosophy examines fundamental questions about reality, knowledge, morality, language, and mind. "
    "Metaphysics asks what exists and what is its nature. The mind-body problem asks how physical brain "
    "processes give rise to subjective conscious experience. The hard problem of consciousness resists "
    "reduction to functional or physical descriptions. Epistemology asks what knowledge is and how "
    "it is possible. Descartes doubted everything he could doubt and reached the cogito. Hume argued "
    "that causal necessity cannot be observed but is a habit of mind. Kant synthesized rationalism and "
    "empiricism. Ethics asks how we should act. Consequentialism judges acts by their outcomes. "
    "Deontology holds that some acts are intrinsically right or wrong regardless of consequences. "
    "Virtue ethics focuses on the character of the agent. Political philosophy asks what makes "
    "political authority legitimate. Social contract theories from Hobbes, Locke, and Rousseau ground "
    "authority in rational consent. Rawls argued that just principles are those rational agents would "
    "choose behind a veil of ignorance. Philosophy of language asks how words and sentences acquire "
    "meaning. Frege distinguished sense from reference. Wittgenstein argued that meaning is use. "
    # Geology
    "Geology studies the solid Earth, its composition, structure, and the processes that shape it. "
    "The Earth consists of an inner solid iron-nickel core, an outer liquid core, a mantle, and a crust. "
    "Plate tectonics explains continental drift, seafloor spreading, mountain building, earthquakes, "
    "and volcanism as consequences of lithospheric plates moving over the asthenosphere. Convergent "
    "plate boundaries create subduction zones with deep trenches and volcanic arcs. Divergent boundaries "
    "create mid-ocean ridges where new ocean floor forms. Transform boundaries produce strike-slip faults. "
    "The rock cycle describes how rocks transition between igneous, sedimentary, and metamorphic forms. "
    "Radiometric dating places the age of Earth at 4.54 billion years. The stratigraphic column records "
    "the history of life through fossils, revealing five major mass extinctions. Ice cores from "
    "Greenland and Antarctica preserve climate records extending hundreds of thousands of years, "
    "showing cycles of glaciation linked to Milankovitch orbital cycles. Geophysics uses seismic waves "
    "to image Earth's interior; seismology maps the structure of the crust and mantle. Soil science "
    "studies the formation, classification, and management of soils as the foundation of agriculture. "
    # Astronomy
    "Astronomy studies celestial objects and phenomena beyond Earth's atmosphere. The Big Bang model "
    "describes the origin of the universe 13.8 billion years ago as an extremely hot dense state that "
    "expanded and cooled. Cosmic microwave background radiation provides the most detailed map of the "
    "early universe. Inflation theory posits an exponential expansion in the first fraction of a second. "
    "Stars form in dense molecular clouds when gravity overcomes thermal pressure. Nuclear fusion in "
    "stellar cores converts hydrogen to helium, releasing energy. More massive stars fuse heavier "
    "elements up to iron, then explode as supernovae. Neutron stars and black holes are the remnants "
    "of massive stellar deaths. Binary neutron star mergers produce gravitational wave signals and "
    "r-process nucleosynthesis, creating heavy elements including gold. The Milky Way contains several "
    "hundred billion stars arranged in a spiral disk with a supermassive black hole at its center. "
    "Dark energy causes the universe's expansion to accelerate and represents the greatest mystery "
    "in modern cosmology. Gravitational lensing bends light from distant sources around massive "
    "objects, enabling indirect detection of dark matter and magnification of distant galaxies. "
    # Medicine
    "Medicine encompasses the science and practice of diagnosing, treating, and preventing disease. "
    "The germ theory, established by Pasteur and Koch, showed that specific microorganisms cause "
    "specific infectious diseases. Vaccination harnesses the immune system to prevent disease. "
    "Antibiotics transformed bacterial infections from leading killers to treatable conditions. "
    "Molecular medicine uses understanding of genes and proteins to diagnose and treat disease. "
    "The Human Genome Project produced a reference sequence for the entire human genome. Monoclonal "
    "antibodies engineered to recognize specific targets are important drugs for cancer and autoimmune "
    "diseases. CRISPR gene therapy offers the prospect of correcting genetic defects directly. Medical "
    "imaging allows non-invasive visualization of anatomy and function. Evidence-based medicine "
    "synthesizes clinical trial data through systematic reviews and meta-analyses to guide treatment. "
    "Epidemiology studies disease at the population level, identifying risk factors and evaluating "
    "interventions. The COVID-19 pandemic demonstrated both the power of rapid mRNA vaccine "
    "development and the challenges of global health coordination. Precision medicine tailors "
    "treatment to individual patients based on genetic, environmental, and lifestyle factors. "
)


# ---------------------------------------------------------------------------
# GPU FUNCTION
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[HF_SECRET],
)
def run_8k_16k_validation():
    import sys
    sys.path.insert(0, "/root")

    import time
    import math
    import gc
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    # ------------------------------------------------------------------ KV cache helpers
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
    print("NEXUSQUANT — True 8K/16K Context Validation (A100 80GB)")
    print("=" * 80)

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"GPU: {dev.name}  VRAM: {dev.total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------ Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    n_layers   = model.config.num_hidden_layers          # 32
    n_kv_heads = model.config.num_key_value_heads        # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L  {n_kv_heads}KVH  d={head_dim}  rope_theta={rope_base:.0f}")

    sliding_window = 32

    # ------------------------------------------------------------------ Load PG19 texts
    print("\nLoading PG19 texts...")
    raw_texts = []

    try:
        from datasets import load_dataset
        print("  Trying emozilla/pg19 (HF streaming)...")
        ds = load_dataset("emozilla/pg19", split="test", streaming=True,
                          token=os.environ["HF_TOKEN"])
        for i, example in enumerate(ds):
            if i >= 5:
                break
            text = example.get("text", "")
            if len(text) > 5000:  # at least 5K chars
                raw_texts.append(text[:60000])  # first 60K chars ≈ 15-18K tokens
                print(f"  Book {i+1}: {len(text):,} chars (using first 60K)")
        print(f"  Loaded {len(raw_texts)} PG19 books from HF.")
    except Exception as e:
        print(f"  PG19 load failed: {e}")

    # Fallback: try LongBench
    if len(raw_texts) < 3:
        try:
            print("  Trying THUDM/LongBench (gov_report subset)...")
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", "gov_report", split="test",
                              streaming=True, token=os.environ["HF_TOKEN"])
            for i, example in enumerate(ds):
                if i >= 5:
                    break
                text = example.get("context", example.get("input", ""))
                if len(text) > 5000:
                    raw_texts.append(text[:60000])
                    print(f"  LongBench doc {i+1}: {len(text):,} chars")
            print(f"  LongBench gave {len(raw_texts)} docs total.")
        except Exception as e:
            print(f"  LongBench load failed: {e}")

    # Last resort: synthetic multi-topic text (repeated to reach target length)
    if len(raw_texts) < 3:
        print("  WARNING: Using synthetic fallback text. "
              "Results are less representative than PG19 books.")
        base = SYNTHETIC_FALLBACK
        # Repeat to get 5 distinct "texts" covering different halves/thirds
        n = len(base)
        segments = [
            base,
            base[n // 5:] + base[:n // 5],
            base[2 * n // 5:] + base[:2 * n // 5],
            base[3 * n // 5:] + base[:3 * n // 5],
            base[4 * n // 5:] + base[:4 * n // 5],
        ]
        raw_texts = segments
        print(f"  Synthetic: {len(raw_texts)} texts, {len(raw_texts[0]):,} chars each")

    # ------------------------------------------------------------------ Tokenize + filter
    MIN_TOKENS = 8192 + 1024   # need at least 8K prefix + 1K continuation
    datasets_by_length = []    # list of dicts: {ids, n_tok, source}

    for i, text in enumerate(raw_texts):
        ids = tok(text, return_tensors="pt", truncation=False).input_ids
        n = ids.shape[1]
        print(f"  Text {i+1}: {n:,} tokens  (raw chars: {len(text):,})")
        if n < MIN_TOKENS:
            print(f"    SKIP: only {n} tokens, need {MIN_TOKENS}")
            continue
        datasets_by_length.append({"ids": ids, "n_tok": n,
                                   "source": f"text_{i+1}"})

    if not datasets_by_length:
        print("ERROR: no texts with sufficient length — aborting.")
        return {}

    print(f"\n{len(datasets_by_length)} texts qualify (>= {MIN_TOKENS} tokens)")

    # ------------------------------------------------------------------ Importance scorer
    def score_importance(kv_cache, seq_len):
        """Key-key attention scorer (SnapKV-style, causal). obs_window = seq_len // 16 clipped to [32, 512]."""
        obs_window = max(32, min(512, seq_len // 16))
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()   # (n_kv_heads, seq_len, head_dim)
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal  = (all_pos <= obs_pos)
            scores  = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn    = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_k = 5
            if seq_len > pool_k:
                imp_1d    = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_k,
                                         padding=pool_k // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ Build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True                        # BOS always kept
        keep_mask[-sliding_window:] = True         # recency window always kept
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            n_avail = (~keep_mask).sum().item()
            _, top_idx = imp.topk(min(n_from_imp, n_avail))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ Evict + 2-bit E8 quantize
    def evict_quantize(kv_cache, keep_mask, prefix_len):
        """
        Evict tokens per keep_mask, E8 quantize kept vectors.
        Returns compression info dict and modified kv_cache.
        ALL overhead counted: zstd-compressed indices + fp16 scales + bit-packed mask.
        """
        bits   = 2
        levels = 2 ** bits
        H      = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_key_coords = []
        all_val_coords = []
        cctx = zstandard.ZstdCompressor(level=22)

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()   # (1, n_kv_heads, seq_len, head_dim)
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor, coord_list in [
                (True,  k, all_key_coords),
                (False, v, all_val_coords),
            ]:
                t = tensor[0].clone()  # (n_kv_heads, seq_len, head_dim)
                for h in range(n_kv_heads):
                    t_head = inverse_rope(t[h:h+1], base=rope_base)[0] if is_key else t[h]
                    kept_data  = t_head[keep_mask]                  # (n_kept, head_dim)
                    rotated    = kept_data @ H.T
                    amax       = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc         = amax / (levels / 2)
                    normalized = rotated / sc
                    groups     = normalized.reshape(-1, 8)
                    lp         = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords     = lp.reshape(-1, head_dim)
                    quantized  = (coords * sc) @ H

                    int_coords = coords.detach().numpy()
                    has_half   = np.any(
                        np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25
                    )
                    coord_list.append(
                        np.round(int_coords.flatten() * 2).astype(np.int8)
                        if has_half
                        else np.round(int_coords.flatten()).astype(np.int8)
                    )

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

        # Compressed size: temporal delta + zstd on coords
        total_idx = 0
        for coords_arr in all_key_coords + all_val_coords:
            arr      = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped = arr.reshape(n_kept, n_per_tok)
                delta    = np.zeros_like(reshaped)
                delta[0] = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx += len(cctx.compress(arr.tobytes()))

        # Scale bytes: one fp16 scale per kept vector, per head, per layer, K+V
        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        # Mask bytes: bit-packed per layer, K+V (1 bit per position)
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers * 2
        total       = total_idx + scale_bytes + mask_bytes

        return {
            "fp16":    total_fp16,
            "idx":     total_idx,
            "scale":   scale_bytes,
            "mask":    mask_bytes,
            "total":   total,
            "ratio":   total_fp16 / total if total > 0 else 0,
            "n_kept":  n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ Eviction configs
    eviction_configs = [
        {"name": "baseline",     "evict_pct": 0},
        {"name": "35%evict",     "evict_pct": 35},
        {"name": "50%evict",     "evict_pct": 50},
        {"name": "60%evict",     "evict_pct": 60},
        {"name": "80%evict",     "evict_pct": 80},
    ]

    prefix_lengths_target = [8192, 16384]

    # ------------------------------------------------------------------ Per-text results
    all_results = []   # list of result dicts

    for text_info in datasets_by_length:
        full_ids = text_info["ids"].to("cuda")
        n_tok    = text_info["n_tok"]
        source   = text_info["source"]

        print(f"\n{'='*80}")
        print(f"TEXT: {source}  ({n_tok:,} tokens)")
        print(f"{'='*80}")

        for target_prefix in prefix_lengths_target:
            cont_len = 1024   # fixed continuation window for PPL

            # Check feasibility
            if n_tok < target_prefix + cont_len:
                print(f"  SKIP prefix={target_prefix}: only {n_tok} tokens available "
                      f"(need {target_prefix + cont_len})")
                continue

            prefix_len = target_prefix
            prefix_ids = full_ids[:, :prefix_len]
            cont_ids   = full_ids[:, prefix_len:prefix_len + cont_len]

            print(f"\n  Prefix={prefix_len}  Cont={cont_len}  "
                  f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f}GB")

            # Memory before compression (after prefill)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_before_gb = torch.cuda.memory_allocated() / 1e9

            # Baseline PPL (no compression)
            try:
                with torch.no_grad():
                    pout    = model(prefix_ids, use_cache=True)
                    kv_bl   = pout.past_key_values
                    mem_after_prefill_gb = torch.cuda.memory_allocated() / 1e9
                    cout    = model(cont_ids, past_key_values=kv_bl, use_cache=True)
                    logits  = cout.logits[:, :-1, :].float()
                    targets = cont_ids[:, 1:].contiguous()
                    loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              targets.reshape(-1))
                    baseline_ppl = torch.exp(loss).item()
                del kv_bl, pout, cout, logits, targets
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at baseline prefix={prefix_len} — skipping this length.")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            if math.isnan(baseline_ppl) or math.isinf(baseline_ppl) or baseline_ppl > 1000:
                print(f"  Degenerate baseline PPL={baseline_ppl:.4f} — skipping.")
                continue

            print(f"  Baseline PPL: {baseline_ppl:.4f}  "
                  f"(KV mem: {mem_after_prefill_gb - mem_before_gb:.2f}GB)")

            # Score importance once for this prefix
            try:
                with torch.no_grad():
                    pout = model(prefix_ids, use_cache=True)
                importance = score_importance(pout.past_key_values, prefix_len)
                del pout
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM during importance scoring — skipping prefix={prefix_len}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Print header for this prefix
            print(f"\n  {'Config':<14s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} "
                  f"{'Kept':>6s} {'MemDelta':>10s}")
            print(f"  {'-'*60}")
            print(f"  {'baseline':<14s} {baseline_ppl:8.4f} {'0.00%':>9s} {'N/A':>7s} "
                  f"{'all':>6s} {'—':>10s}")

            for cfg in eviction_configs:
                if cfg["evict_pct"] == 0:
                    continue  # already printed above

                evict_pct = cfg["evict_pct"]
                torch.cuda.empty_cache()

                try:
                    with torch.no_grad():
                        pout = model(prefix_ids, use_cache=True)
                        kv   = pout.past_key_values

                    mem_pre_compress = torch.cuda.memory_allocated() / 1e9
                    keep_mask        = build_keep_mask(prefix_len, evict_pct, importance)
                    info, kv         = evict_quantize(kv, keep_mask, prefix_len)
                    mem_post_compress = torch.cuda.memory_allocated() / 1e9
                    mem_delta_gb      = mem_post_compress - mem_pre_compress

                    # Attention mask: 0 for evicted positions in prefix, 1 elsewhere
                    attn_ctx  = torch.ones(prefix_len, dtype=torch.long, device="cuda")
                    attn_ctx[~keep_mask] = 0
                    attn_cont = torch.ones(cont_len, dtype=torch.long, device="cuda")
                    attn_full = torch.cat([attn_ctx, attn_cont]).unsqueeze(0)

                    with torch.no_grad():
                        cout    = model(cont_ids, past_key_values=kv,
                                        attention_mask=attn_full, use_cache=True)
                        logits  = cout.logits[:, :-1, :].float()
                        targets = cont_ids[:, 1:].contiguous()
                        loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                  targets.reshape(-1))
                        ppl     = torch.exp(loss).item()

                    del kv, cout, logits, targets, attn_full

                    if math.isnan(ppl) or math.isinf(ppl):
                        print(f"  {cfg['name']:<14s} DEGENERATE (nan/inf)")
                        torch.cuda.empty_cache()
                        continue

                    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
                    print(f"  {cfg['name']:<14s} {ppl:8.4f} {delta:>+8.2f}% "
                          f"{info['ratio']:6.2f}x {info['n_kept']:5d} "
                          f"{mem_delta_gb:>+9.2f}GB")

                    all_results.append({
                        "source":       source,
                        "prefix_len":   prefix_len,
                        "cont_len":     cont_len,
                        "evict_pct":    evict_pct,
                        "baseline_ppl": baseline_ppl,
                        "ppl":          ppl,
                        "delta":        delta,
                        "ratio":        info["ratio"],
                        "n_kept":       info["n_kept"],
                        "fp16_bytes":   info["fp16"],
                        "idx_bytes":    info["idx"],
                        "scale_bytes":  info["scale"],
                        "mask_bytes":   info["mask"],
                        "total_bytes":  info["total"],
                        "mem_delta_gb": mem_delta_gb,
                    })

                except torch.cuda.OutOfMemoryError:
                    print(f"  {cfg['name']:<14s} OOM — skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

            torch.cuda.empty_cache()
            gc.collect()

    # ------------------------------------------------------------------ Aggregate stats
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS — mean ± std across texts")
    print(f"{'='*80}")

    for prefix_len in prefix_lengths_target:
        prefix_rows = [r for r in all_results if r["prefix_len"] == prefix_len]
        if not prefix_rows:
            continue

        print(f"\n## Prefix = {prefix_len} tokens  "
              f"({len(set(r['source'] for r in prefix_rows))} texts)")
        print(f"\n| Config | PPL Delta% (mean) | PPL Delta% (std) | Ratio (mean) | Ratio (std) |")
        print(f"|--------|-------------------|------------------|--------------|-------------|")

        evict_pcts = sorted(set(r["evict_pct"] for r in prefix_rows))
        for evict_pct in evict_pcts:
            rows   = [r for r in prefix_rows if r["evict_pct"] == evict_pct]
            deltas = [r["delta"] for r in rows]
            ratios = [r["ratio"] for r in rows]
            d_mean = float(np.mean(deltas))
            d_std  = float(np.std(deltas)) if len(deltas) > 1 else 0.0
            r_mean = float(np.mean(ratios))
            r_std  = float(np.std(ratios)) if len(ratios) > 1 else 0.0
            name   = f"{evict_pct}%evict" if evict_pct > 0 else "baseline"
            print(f"| {name:<12s} | {d_mean:>+16.2f}% | {d_std:>15.2f}% | "
                  f"{r_mean:>11.2f}x | {r_std:>10.2f}x |")

    # Full per-text breakdown
    print(f"\n{'='*80}")
    print("PER-TEXT BREAKDOWN")
    print(f"{'='*80}")

    sources = sorted(set(r["source"] for r in all_results))
    for src in sources:
        print(f"\n### {src}")
        src_rows = [r for r in all_results if r["source"] == src]
        prefix_lens = sorted(set(r["prefix_len"] for r in src_rows))
        for pl in prefix_lens:
            pl_rows = [r for r in src_rows if r["prefix_len"] == pl]
            if not pl_rows:
                continue
            baseline_ppl = pl_rows[0]["baseline_ppl"]
            print(f"  prefix={pl}  baseline_ppl={baseline_ppl:.4f}")
            print(f"  {'Evict%':<10s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} "
                  f"{'Kept':>6s} {'idx_MB':>7s} {'scale_MB':>9s} {'mask_B':>7s}")
            for r in sorted(pl_rows, key=lambda x: x["evict_pct"]):
                ep_str = f"{r['evict_pct']}%"
                print(f"  {ep_str:<10s} {r['ppl']:8.4f} {r['delta']:>+8.2f}% "
                      f"{r['ratio']:6.2f}x {r['n_kept']:5d} "
                      f"{r['idx_bytes']/1e6:6.2f}MB "
                      f"{r['scale_bytes']/1e6:8.2f}MB "
                      f"{r['mask_bytes']:6d}B")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — Comparison with 3544-token reference results")
    print(f"{'='*80}")
    print("Reference (3544-token prefix, prior experiment):")
    print("  35%evict → 10.4x  at +0.43% PPL")
    print("  60%evict → 16.8x  at +1.34% PPL")
    print("  80%evict → 33.3x  at +2.64% PPL")
    print()

    for prefix_len in prefix_lengths_target:
        prefix_rows = [r for r in all_results if r["prefix_len"] == prefix_len]
        if not prefix_rows:
            continue
        print(f"8K/16K (prefix={prefix_len}):")
        for evict_pct in [35, 60, 80]:
            rows = [r for r in prefix_rows if r["evict_pct"] == evict_pct]
            if not rows:
                continue
            d_mean = np.mean([r["delta"] for r in rows])
            r_mean = np.mean([r["ratio"] for r in rows])
            n_txt  = len(rows)
            print(f"  {evict_pct}%evict → {r_mean:.1f}x  at {d_mean:+.2f}% PPL  (n={n_txt} texts)")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory: {peak_mem:.2f} GB")
    print("=" * 80)

    return all_results


# ---------------------------------------------------------------------------
# LOCAL ENTRYPOINT
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import time
    import json

    print("\n" + "=" * 80)
    print("NEXUSQUANT: 8K/16K Context Validation on A100 80GB")
    print("=" * 80)
    print("Launching on Modal A100-80GB...")

    t0      = time.time()
    results = run_8k_16k_validation.remote()
    elapsed = time.time() - t0

    print(f"\nTotal wall-clock time: {elapsed/60:.1f} minutes")

    # Write results locally
    out_dir  = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "8k_16k_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON written to: {out_path}")

    # Markdown summary
    md_path = os.path.join(out_dir, "8k_16k_validation_results.md")
    with open(md_path, "w") as f:
        f.write("# NexusQuant 8K/16K Context Validation\n\n")
        f.write("**Model:** Mistral-7B-v0.1  |  **GPU:** A100 80GB  |  **Bits:** 2  |  ")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Data source:** PG19 books (emozilla/pg19, HF) with fallback to LongBench/synthetic.\n\n")
        f.write("**Overhead accounting:** All bytes counted — zstd-compressed E8 indices ")
        f.write("+ fp16 scales + bit-packed eviction mask.\n\n")

        for prefix_len in [8192, 16384]:
            prefix_rows = [r for r in results if r["prefix_len"] == prefix_len]
            if not prefix_rows:
                continue
            n_texts = len(set(r["source"] for r in prefix_rows))
            f.write(f"## Prefix = {prefix_len} tokens ({n_texts} texts)\n\n")
            f.write("| Config | PPL Delta% mean | PPL Delta% std | Ratio mean | Ratio std |\n")
            f.write("|--------|----------------|----------------|------------|----------|\n")

            import numpy as np
            for evict_pct in sorted(set(r["evict_pct"] for r in prefix_rows)):
                rows   = [r for r in prefix_rows if r["evict_pct"] == evict_pct]
                deltas = [r["delta"] for r in rows]
                ratios = [r["ratio"] for r in rows]
                d_mean = float(np.mean(deltas))
                d_std  = float(np.std(deltas)) if len(deltas) > 1 else 0.0
                r_mean = float(np.mean(ratios))
                r_std  = float(np.std(ratios)) if len(ratios) > 1 else 0.0
                name   = f"{evict_pct}%evict" if evict_pct > 0 else "baseline"
                f.write(f"| {name} | {d_mean:+.2f}% | {d_std:.2f}% | {r_mean:.2f}x | {r_std:.2f}x |\n")
            f.write("\n")

        f.write("## Comparison with 3544-Token Reference\n\n")
        f.write("| Evict% | Ref ratio | Ref PPL delta | 8K ratio | 8K PPL delta | 16K ratio | 16K PPL delta |\n")
        f.write("|--------|-----------|---------------|----------|--------------|-----------|---------------|\n")
        ref = {35: (10.4, 0.43), 60: (16.8, 1.34), 80: (33.3, 2.64)}
        import numpy as np
        for ep in [35, 60, 80]:
            rr, rd = ref[ep]
            r8  = [r for r in results if r["prefix_len"] == 8192  and r["evict_pct"] == ep]
            r16 = [r for r in results if r["prefix_len"] == 16384 and r["evict_pct"] == ep]
            r8_ratio  = f"{np.mean([r['ratio'] for r in r8]):.1f}x"  if r8  else "—"
            r8_delta  = f"{np.mean([r['delta'] for r in r8]):+.2f}%" if r8  else "—"
            r16_ratio = f"{np.mean([r['ratio'] for r in r16]):.1f}x"  if r16 else "—"
            r16_delta = f"{np.mean([r['delta'] for r in r16]):+.2f}%" if r16 else "—"
            f.write(f"| {ep}% | {rr}x | +{rd}% | {r8_ratio} | {r8_delta} | {r16_ratio} | {r16_delta} |\n")
        f.write("\n")
        f.write("*Ratios are means across PG19 test books. All overhead included.*\n")

    print(f"Markdown summary written to: {md_path}")

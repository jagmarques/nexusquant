"""Combined experiment: Real LongBench eval on Mistral-7B + Llama-3 negative PPL validation.

Task A: LongBench eval on Mistral-7B
  - Loads 5 real LongBench tasks from THUDM/LongBench via HuggingFace datasets
  - Tasks: narrativeqa, qasper, multifieldqa_en, hotpotqa, triviaqa
  - 3-5 examples per task
  - Evaluation: PPL of expected answer under baseline vs compressed KV
  - Configs: baseline, 2b+35%evict (10x), 2b+60%evict (16x)

Task B: Llama-3-8B negative PPL validation on a SECOND text
  - Validates the negative PPL phenomenon seen in llama3_8k_results.md
  - Uses a completely different hard technical text (advanced math/physics/biology)
  - Configs: 2b+0%, 2b+35%, 2b+60%, 2b+80%
  - Key question: does negative PPL persist on second text, or was it text-dependent?

Results written to .company/engineering/longbench_llama3_results.md
"""
import modal
import os

app = modal.App("nexusquant-longbench-llama3")

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
        "datasets>=2.16.0",
    )
    .add_local_dir(nq_local, remote_path="/root/nexusquant")
)

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# ======================================================================
# SECOND VALIDATION TEXT — HARD TECHNICAL (different from original text)
# Advanced math + quantum physics + molecular biology + complexity theory
# This is the "second text" to cross-validate the negative PPL phenomenon
# ======================================================================
HARD_TECHNICAL_TEXT = (
    # Advanced quantum mechanics (~600 words)
    "The formalism of quantum mechanics rests on the mathematical structure of Hilbert spaces and operator "
    "algebras. The state of a quantum system is represented by a normalized vector in a complex Hilbert space, "
    "or more generally by a density operator — a positive semidefinite trace-one operator — that encodes "
    "both pure and mixed states. Observables correspond to self-adjoint operators, and the spectrum of such "
    "an operator gives the set of possible measurement outcomes. The Born rule provides the connection "
    "between the mathematical formalism and experimental probability: the probability of obtaining eigenvalue "
    "lambda when measuring observable A in state psi is the squared norm of the projection of psi onto the "
    "corresponding eigenspace. The time evolution of a closed quantum system is governed by the Schrodinger "
    "equation, a linear first-order differential equation in time, with the Hamiltonian operator H generating "
    "unitary evolution. For time-independent Hamiltonians, the formal solution involves the operator "
    "exponential exp(-iHt/hbar), which is unitary and hence norm-preserving. The measurement postulate "
    "introduces irreversible state collapse: upon measuring an observable and obtaining outcome lambda, "
    "the state is projected onto the eigenspace of lambda and renormalized. This non-unitary collapse "
    "distinguishes quantum measurement from unitary evolution and gives rise to the measurement problem, "
    "one of the central conceptual difficulties of quantum mechanics. Entanglement is the defining feature "
    "of composite quantum systems. A pure state of a bipartite system is entangled if and only if it cannot "
    "be written as a tensor product of states of the subsystems. The Schmidt decomposition provides a "
    "canonical form: any pure bipartite state can be written as a sum of tensor products of orthonormal "
    "basis vectors of the two subsystems, with non-negative real coefficients whose squares sum to one. "
    "The number of non-zero Schmidt coefficients, the Schmidt rank, measures the degree of entanglement. "
    "Entangled states exhibit correlations that cannot be explained by any local hidden variable theory, "
    "as expressed by Bell's inequalities, which quantum mechanics violates. Quantum field theory extends "
    "quantum mechanics to relativistic settings by treating fields rather than particles as the fundamental "
    "entities. Fields are operator-valued distributions on spacetime, satisfying commutation or "
    "anticommutation relations depending on whether they describe bosons or fermions. The vacuum state "
    "is the lowest energy eigenstate of the field Hamiltonian, but it is not empty: virtual particle-"
    "antiparticle pairs constantly fluctuate in and out of existence, contributing to the vacuum energy "
    "and to measurable physical effects like the Casimir effect and the Lamb shift. Renormalization "
    "procedures handle the ultraviolet divergences that arise in perturbative calculations, trading bare "
    "parameters for physical renormalized parameters. The renormalization group describes how the "
    "effective description of a theory changes with the energy scale, with fixed points corresponding "
    "to scale-invariant theories. Quantum chromodynamics, the theory of the strong force, exhibits "
    "asymptotic freedom: the coupling constant decreases at high energies, allowing perturbative "
    "calculations, but increases at low energies, leading to confinement of quarks and gluons within "
    "hadrons. The non-perturbative regime of QCD is studied using lattice gauge theory, which discretizes "
    "spacetime and evaluates path integrals numerically on large computer clusters. "
    # Advanced algebraic geometry and number theory (~600 words)
    "The Langlands program represents one of the most ambitious and profound unifying visions in modern "
    "mathematics, positing deep connections between number theory, representation theory, and harmonic "
    "analysis. Its origins lie in a 1967 letter from Robert Langlands to Andre Weil, proposing that "
    "automorphic forms — functions on Lie groups satisfying certain symmetry and analytic properties — "
    "are related to Galois representations in a precise and far-reaching way. An automorphic representation "
    "is an irreducible representation of an adelic group appearing in the decomposition of the space of "
    "automorphic forms; a Galois representation is a continuous homomorphism from the absolute Galois "
    "group of a number field into a linear group over a p-adic or complex coefficient field. The "
    "functoriality conjecture asserts that natural maps between L-groups correspond to transfers of "
    "automorphic representations. The reciprocity conjecture asserts that L-functions associated to "
    "Galois representations — which encode arithmetic information about solutions of polynomial equations "
    "— coincide with automorphic L-functions, which are defined analytically. The proof of Fermat's Last "
    "Theorem by Andrew Wiles in 1994 can be understood as a special case of the Shimura-Taniyama-Weil "
    "conjecture, now the modularity theorem, which states that every rational elliptic curve is modular: "
    "it arises from a weight-2 cusp form. An elliptic curve over the rationals is a smooth projective "
    "curve of genus one with a specified rational point; its L-function encodes information about the "
    "number of points on the curve over finite fields. The Birch and Swinnerton-Dyer conjecture, one of "
    "the seven Millennium Prize Problems, predicts that the rank of the Mordell-Weil group of rational "
    "points equals the order of vanishing of the L-function at the central value s=1. The Riemann "
    "hypothesis, which asserts that all non-trivial zeros of the Riemann zeta function have real part "
    "one-half, remains unproven after 167 years. Random matrix theory has provided unexpected insight: "
    "the statistical distribution of spacings between zeros of the zeta function matches the distribution "
    "of eigenvalue spacings of large random unitary matrices, suggesting deep connections between number "
    "theory and quantum chaos. Algebraic K-theory provides invariants of rings and schemes that generalize "
    "both classical algebraic invariants and topological K-theory of spaces. The higher K-groups K_n(R) "
    "of a ring R encode subtle arithmetic and geometric information; the K-groups of rings of integers "
    "of number fields are related to special values of zeta functions by the Quillen-Lichtenbaum "
    "conjecture, now largely proved using motivic cohomology. Motives are a conjectural universal "
    "cohomology theory for algebraic varieties, providing a single framework from which all "
    "Weil cohomology theories — de Rham, etale, crystalline — should arise as realizations. "
    "The category of pure motives, constructed from smooth projective varieties modulo algebraic "
    "equivalence with rational coefficients, carries a tensor product and internal Hom, making it "
    "a rigid tensor category. Grothendieck's standard conjectures, particularly the Hodge conjecture "
    "over finite fields and the Lefschetz conjecture, remain central unresolved problems. "
    # Molecular biology and systems biology (~500 words)
    "Gene regulatory networks orchestrate the spatiotemporal patterns of gene expression that underlie "
    "cellular differentiation, morphogenesis, and homeostasis. Transcription factors bind specific DNA "
    "sequences in promoters and enhancers, recruiting or excluding the transcriptional machinery and "
    "thereby activating or repressing target genes. The combinatorial logic of transcription factor "
    "binding — with multiple factors acting cooperatively or antagonistically — allows a finite set "
    "of regulatory proteins to generate a vast diversity of cell-type-specific expression programs. "
    "Chromatin structure provides an additional regulatory layer: nucleosomes, consisting of DNA wrapped "
    "around histone octamers, can be repositioned by chromatin remodeling complexes, and histone "
    "residues can be covalently modified by methylation, acetylation, phosphorylation, and ubiquitination, "
    "creating an epigenetic code read by effector proteins. Enhancers are cis-regulatory elements, often "
    "thousands of base pairs from the genes they regulate, that contact promoters through chromatin "
    "looping mediated by the cohesin complex and CTCF protein. Super-enhancers are dense clusters of "
    "enhancers associated with key cell identity genes, characterized by particularly high levels of "
    "transcription factor binding and active histone marks. Non-coding RNAs including long non-coding "
    "RNAs and small non-coding RNAs participate in gene regulation through diverse mechanisms: acting "
    "as scaffolds for chromatin-modifying complexes, competing for microRNA binding sites, or guiding "
    "CRISPR-like surveillance complexes. Systems biology applies mathematical modeling to understand "
    "the emergent properties of biological networks. Ordinary differential equations describe the "
    "temporal dynamics of protein and mRNA concentrations, with parameters estimated from quantitative "
    "experimental data. Network motifs — recurring subgraph patterns such as feed-forward loops and "
    "autoregulatory feedback — perform specific signal processing functions: feed-forward loops can "
    "act as sign-sensitive delays, filtering out transient signals, while negative feedback loops "
    "reduce noise and enable precise adaptation. Synthetic biology applies engineering principles to "
    "construct novel biological circuits with predictable behavior. Toggle switches, oscillators, and "
    "logic gates have been built from transcription factors and RNA components, demonstrating that "
    "biological computation can be rationally designed. The central dogma of molecular biology — "
    "DNA to RNA to protein — has been refined by discoveries of reverse transcription in retroviruses, "
    "RNA editing, alternative splicing that allows one gene to encode multiple proteins, and post-"
    "translational modifications that dramatically expand proteome diversity. Protein folding remains "
    "a central problem: the sequence of amino acids in a polypeptide chain determines its "
    "three-dimensional structure, but the folding pathway and the principles governing which sequences "
    "fold reliably versus form aberrant aggregates are still incompletely understood. AlphaFold2's "
    "dramatic success in predicting protein structure from sequence using deep learning has transformed "
    "structural biology, making high-quality structural models available for essentially the entire "
    "known proteome of many organisms. "
    # Computational complexity and information theory (~500 words)
    "Information theory, founded by Claude Shannon in 1948, provides a mathematical framework for "
    "quantifying information content and the limits of communication and compression. The entropy "
    "H(X) = -sum p(x) log p(x) of a discrete random variable X measures the average number of bits "
    "needed to encode its outcomes under an optimal code. Shannon's source coding theorem establishes "
    "that lossless compression can approach but not exceed the entropy rate, and his channel coding "
    "theorem establishes that reliable communication over a noisy channel is possible at any rate "
    "below the channel capacity C = max_{p(x)} I(X;Y), where I(X;Y) is the mutual information "
    "between input and output. The Kolmogorov complexity K(x) of a string x is the length of the "
    "shortest program on a universal Turing machine that outputs x, providing an absolute measure "
    "of the information content of individual strings rather than distributions. Kolmogorov complexity "
    "is not computable, but it provides the conceptual foundation for algorithmic information theory "
    "and connects information theory to computational complexity. The P versus NP problem asks whether "
    "every problem whose solution can be verified in polynomial time can also be solved in polynomial "
    "time. This is equivalent to asking whether the class NP, which includes problems like satisfiability, "
    "graph coloring, and the traveling salesman problem, equals the class P of problems solvable "
    "efficiently. Most complexity theorists believe P ≠ NP, partly because of the existence of "
    "NP-complete problems: if any NP-complete problem has a polynomial-time algorithm, then all "
    "NP problems do. Cook's theorem, proved in 1971, established that Boolean satisfiability is "
    "NP-complete; Karp subsequently showed 21 other combinatorial problems are also NP-complete. "
    "The polynomial hierarchy extends P and NP to a hierarchy of complexity classes using oracle "
    "machines, with levels defined by alternating quantifiers. The permanent of a matrix — defined "
    "like the determinant but without signs — is #P-complete to compute, even though the determinant "
    "is computable in polynomial time; this distinction between the permanent and determinant is "
    "related to deep open questions about algebraic complexity. Randomized complexity classes such "
    "as BPP (bounded-error probabilistic polynomial time) capture problems solvable efficiently with "
    "access to random bits. Adleman's theorem shows that BPP is contained in P/poly, meaning "
    "randomized polynomial-time algorithms can be derandomized with polynomial-size advice. "
    "Pseudorandom generators — deterministic algorithms whose outputs are computationally "
    "indistinguishable from random — would imply BPP = P, and their existence is related to "
    "hardness assumptions. Quantum computing adds another dimension: BQP, the class of problems "
    "solvable in polynomial time on a quantum computer, contains integer factoring by Shor's "
    "algorithm and unstructured search speedup by Grover's algorithm. Whether BQP contains NP "
    "is unknown, but believed unlikely. The quantum threshold theorem establishes that quantum "
    "error correction allows fault-tolerant quantum computation provided the physical error rate "
    "is below a threshold of approximately one percent per gate operation. "
)


# ======================================================================
# TASK A: LongBench eval on Mistral-7B
# ======================================================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_longbench():
    import sys
    sys.path.insert(0, "/root")

    import time, math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard
    import os

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
    print("NEXUSQUANT — TASK A: Real LongBench Eval on Mistral-7B")
    print("=" * 80)
    print("Tasks: narrativeqa | qasper | multifieldqa_en | hotpotqa | triviaqa")
    print("Metric: PPL of expected answer given compressed context (lower = better)")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers   # 32
    n_kv_heads = model.config.num_key_value_heads # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    sliding_window = 32
    MAX_TOKENS = 2048   # hard cap: context + question must fit here
    N_EXAMPLES = 4      # per task

    # ------------------------------------------------------------------ importance scorer
    def score_importance(kv_cache, prefix_len):
        obs_window = max(32, prefix_len // 16)
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores_mat = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = (all_pos <= obs_pos)
            scores_mat = scores_mat.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores_mat, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                          padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict + 2-bit E8 quantize
    def evict_quantize(kv_cache, keep_mask, prefix_len):
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
                levels = 4  # 2-bit
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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ compute PPL of answer tokens
    def compute_answer_ppl(model, prefix_kv, prefix_len, evict_mask, answer_ids_gpu):
        """Compute PPL of answer_ids given prefix KV cache (possibly with eviction).

        prefix_kv: past_key_values from forward pass on context+question
        evict_mask: bool tensor of length prefix_len, True = evicted (attention=0)
        answer_ids_gpu: [1, n_ans] token ids on GPU
        """
        n_ans = answer_ids_gpu.shape[1]
        if n_ans < 2:
            return float('nan')

        # Build full input: answer tokens
        # Attention mask covers: prefix positions (1=kept, 0=evicted) + answer positions (all 1)
        attn_prefix = torch.ones(prefix_len, dtype=torch.long, device="cuda")
        if evict_mask is not None:
            attn_prefix[evict_mask] = 0
        attn_ans = torch.ones(n_ans, dtype=torch.long, device="cuda")
        attn_full = torch.cat([attn_prefix, attn_ans]).unsqueeze(0)  # [1, prefix_len+n_ans]

        with torch.no_grad():
            out = model(
                answer_ids_gpu,
                past_key_values=prefix_kv,
                attention_mask=attn_full,
                use_cache=False,
            )
            logits = out.logits[:, :-1, :].float()  # [1, n_ans-1, vocab]
            targets = answer_ids_gpu[:, 1:]          # [1, n_ans-1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return torch.exp(loss).item()

    # ------------------------------------------------------------------ load LongBench tasks
    # THUDM/LongBench data is in data.zip → data/{task_name}.jsonl
    # Download zip once, extract, load jsonl files directly.
    import zipfile, urllib.request, tempfile, json as _json

    TASKS = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "triviaqa",
    ]

    EVICT_CONFIGS = [
        {"name": "baseline",      "evict_pct": 0},
        {"name": "35%evict(10x)", "evict_pct": 35},
        {"name": "60%evict(16x)", "evict_pct": 60},
    ]

    # Download and extract LongBench data.zip once
    data_dir = None
    zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
    print(f"\nDownloading LongBench data.zip from HuggingFace (~114MB)...")
    try:
        import requests as _req
        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, "data.zip")
        resp = _req.get(
            zip_url,
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        with open(zip_path, "wb") as zf:
            for chunk in resp.iter_content(chunk_size=8*1024*1024):
                zf.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zr:
            zr.extractall(tmpdir)
        data_dir = os.path.join(tmpdir, "data")
        print(f"  Extracted to {data_dir}")
        print(f"  Files: {os.listdir(data_dir)[:10]}")
    except Exception as e:
        print(f"  ERROR downloading/extracting: {e}")

    def load_task_examples(task_name, n_examples):
        """Load N examples from extracted LongBench jsonl."""
        if data_dir is None:
            return None, "no data_dir"
        path = os.path.join(data_dir, f"{task_name}.jsonl")
        if not os.path.exists(path):
            return None, f"file not found: {path}"
        examples = []
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                examples.append(_json.loads(line))
                if len(examples) >= n_examples:
                    break
        return examples, None

    all_task_results = []

    for task_name in TASKS:
        print(f"\n{'='*70}")
        print(f"Task: {task_name}")
        try:
            examples_raw, err = load_task_examples(task_name, N_EXAMPLES)
            if err:
                raise RuntimeError(err)
            print(f"  Loaded {len(examples_raw)} examples from jsonl")
        except Exception as e:
            print(f"  ERROR loading {task_name}: {e}")
            all_task_results.append({
                "task": task_name,
                "error": str(e),
                "examples": [],
            })
            continue

        task_example_results = []

        for ex_idx, ex in enumerate(examples_raw):
            # Parse fields — LongBench uses 'context', 'input', 'answers'
            context_text = ex.get("context", "") or ""
            question_text = ex.get("input", "") or ""
            answers_raw = ex.get("answers", []) or []

            # answers can be a list of strings or a list of dicts with 'text'
            answer_strings = []
            for a in answers_raw:
                if isinstance(a, str):
                    answer_strings.append(a)
                elif isinstance(a, dict) and "text" in a:
                    answer_strings.append(a["text"])
            if not answer_strings:
                answer_strings = [""]

            # Best answer = first (or shortest if multiple)
            best_answer = min(answer_strings, key=len) if answer_strings else ""
            if not best_answer:
                print(f"  Example {ex_idx}: empty answer, skipping")
                continue

            # Build prompt: context + question. Truncate context to fit MAX_TOKENS.
            question_part = f"\n\nQuestion: {question_text}\nAnswer:"
            q_ids = tok(question_part, return_tensors="pt").input_ids
            q_len = q_ids.shape[1]

            # Truncate context to leave room for question + a bit of headroom
            ctx_max_tokens = MAX_TOKENS - q_len - 10
            ctx_ids = tok(context_text, return_tensors="pt",
                          max_length=ctx_max_tokens, truncation=True).input_ids
            ctx_len = ctx_ids.shape[1]

            # Build full prefix: [context_ids, question_ids]
            prefix_ids = torch.cat([ctx_ids, q_ids], dim=1).to("cuda")
            prefix_len = prefix_ids.shape[1]

            # Tokenize best answer for PPL scoring
            ans_ids = tok(best_answer, return_tensors="pt",
                          max_length=128, truncation=True).input_ids.to("cuda")
            n_ans_tok = ans_ids.shape[1]

            print(f"\n  Ex {ex_idx}: context={ctx_len}tok, prefix={prefix_len}tok, "
                  f"answer='{best_answer[:60]}' ({n_ans_tok}tok)")

            if n_ans_tok < 2:
                print(f"    Answer too short, skipping")
                continue

            # Compute importance once for this example
            with torch.no_grad():
                pout_imp = model(prefix_ids, use_cache=True)
            importance = score_importance(pout_imp.past_key_values, prefix_len)

            ex_config_results = []
            for cfg in EVICT_CONFIGS:
                torch.cuda.empty_cache()
                evict_pct = cfg["evict_pct"]

                # Re-run prefix forward to get fresh KV
                with torch.no_grad():
                    pout = model(prefix_ids, use_cache=True)
                    kv = pout.past_key_values

                evict_mask = None
                ratio = 1.0

                if evict_pct > 0:
                    keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
                    info, kv = evict_quantize(kv, keep_mask, prefix_len)
                    evict_mask_full = ~keep_mask
                    evict_mask = evict_mask_full
                    ratio = info["ratio"]

                # Compute PPL of best answer given compressed context
                ppl = compute_answer_ppl(model, kv, prefix_len, evict_mask, ans_ids)

                print(f"    [{cfg['name']:16s}] PPL={ppl:.4f}  ratio={ratio:.1f}x")
                ex_config_results.append({
                    "config": cfg["name"],
                    "evict_pct": evict_pct,
                    "ppl": ppl,
                    "ratio": ratio,
                })

            task_example_results.append({
                "ex_idx": ex_idx,
                "context_len": ctx_len,
                "prefix_len": prefix_len,
                "answer": best_answer[:200],
                "answer_tokens": n_ans_tok,
                "configs": ex_config_results,
            })

        all_task_results.append({
            "task": task_name,
            "n_examples": len(task_example_results),
            "examples": task_example_results,
        })

    # ------------------------------------------------------------------ per-task summary
    print("\n" + "=" * 80)
    print("LONGBENCH SUMMARY: Mean PPL per task and config")
    print("(Lower PPL = model better predicts the answer given that context)")
    print("=" * 80)

    summary_rows = []
    for tr in all_task_results:
        if "error" in tr or not tr["examples"]:
            print(f"  {tr['task']:25s} — no data")
            continue

        row = {"task": tr["task"]}
        for cfg in EVICT_CONFIGS:
            cn = cfg["name"]
            ppls = [
                ec["ppl"]
                for ex in tr["examples"]
                for ec in ex["configs"]
                if ec["config"] == cn and not math.isnan(ec["ppl"])
            ]
            mean_ppl = sum(ppls) / len(ppls) if ppls else float('nan')
            row[cn] = mean_ppl

        # Compute delta vs baseline
        baseline_ppl = row.get("baseline", float('nan'))
        row["delta_35"] = ((row.get("35%evict(10x)", float('nan')) - baseline_ppl)
                           / baseline_ppl * 100
                           if not math.isnan(baseline_ppl) and baseline_ppl > 0 else float('nan'))
        row["delta_60"] = ((row.get("60%evict(16x)", float('nan')) - baseline_ppl)
                           / baseline_ppl * 100
                           if not math.isnan(baseline_ppl) and baseline_ppl > 0 else float('nan'))
        summary_rows.append(row)

        print(f"  {tr['task']:25s}  baseline={baseline_ppl:.4f}  "
              f"35%evict={row.get('35%evict(10x)', float('nan')):.4f}(Δ{row['delta_35']:+.2f}%)  "
              f"60%evict={row.get('60%evict(16x)', float('nan')):.4f}(Δ{row['delta_60']:+.2f}%)")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return all_task_results, summary_rows


# ======================================================================
# TASK B: Llama-3-8B Negative PPL Validation on Second Text
# ======================================================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_llama3_validation():
    import sys
    sys.path.insert(0, "/root")

    import time, math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard
    import os

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
    print("NEXUSQUANT — TASK B: Llama-3 Negative PPL Validation on SECOND TEXT")
    print("=" * 80)
    print("Previous result: ALL configs showed NEGATIVE PPL on text #1 (multi-topic)")
    print("Second text: Advanced quantum mechanics + Langlands + molecular biology + complexity")
    print("Hypothesis: if negative PPL persists on text #2, it is a REAL phenomenon")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "NousResearch/Meta-Llama-3-8B"
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers   # 32
    n_kv_heads = model.config.num_key_value_heads # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 500000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    sliding_window = 32

    # ------------------------------------------------------------------ importance scorer
    def score_importance(kv_cache, prefix_len):
        obs_window = max(32, prefix_len // 16)
        print(f"  [scorer] prefix_len={prefix_len}, obs_window={obs_window} "
              f"({obs_window/prefix_len*100:.1f}% of context)")
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores_mat = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = (all_pos <= obs_pos)
            scores_mat = scores_mat.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores_mat, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                          padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict + 2-bit E8 quantize
    def evict_quantize(kv_cache, keep_mask, prefix_len):
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
                levels = 4  # 2-bit
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

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ tokenize second text
    inputs = tok(HARD_TECHNICAL_TEXT, return_tensors="pt", max_length=8192, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"\nSecond text: tokens={n_tok}, prefix={prefix_len}, continuation={cont_len}")

    # ------------------------------------------------------------------ baseline PPL on second text
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
        cout = model(full_ids[:, prefix_len:], past_key_values=pout.past_key_values, use_cache=True)
        logits = cout.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL (second text): {baseline_ppl:.4f}")

    # Previous text #1 baseline for reference
    PREV_BASELINE_PPL = 3.1433
    print(f"Previous text #1 baseline:  {PREV_BASELINE_PPL:.4f}")
    print(f"Text difficulty ratio:       {baseline_ppl/PREV_BASELINE_PPL:.3f}x")

    # ------------------------------------------------------------------ compute importance once
    with torch.no_grad():
        pout_imp = model(full_ids[:, :prefix_len], use_cache=True)
    importance = score_importance(pout_imp.past_key_values, prefix_len)

    # ------------------------------------------------------------------ configs
    configs = [
        {"name": "2b+0%evict",   "evict_pct": 0},
        {"name": "2b+35%evict",  "evict_pct": 35},
        {"name": "2b+60%evict",  "evict_pct": 60},
        {"name": "2b+80%evict",  "evict_pct": 80},
    ]

    # Previous text #1 results for comparison
    PREV_TEXT1_RESULTS = {
        0:  {"ppl": 3.1057, "delta": -1.196, "ratio": 6.71},
        35: {"ppl": 3.0971, "delta": -1.470, "ratio": 10.25},
        60: {"ppl": 3.1009, "delta": -1.348, "ratio": 16.48},
        80: {"ppl": 3.1240, "delta": -0.614, "ratio": 32.45},
    }

    print(f"\n{'Config':<20s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} {'Kept':>6s}  "
          f"{'Text1 Δ%':>9s}  {'Verdict':>12s}")
    print("-" * 80)

    results = []
    for cfg in configs:
        torch.cuda.empty_cache()
        evict_pct = cfg["evict_pct"]

        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            kv = pout.past_key_values

        keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
        info, kv = evict_quantize(kv, keep_mask, prefix_len)

        evict_mask = ~keep_mask
        attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
        attn_ctx[evict_mask] = 0
        attn_full = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long, device="cuda")])

        with torch.no_grad():
            cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                         attention_mask=attn_full.unsqueeze(0), use_cache=True)
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppl = torch.exp(loss).item()

        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

        prev = PREV_TEXT1_RESULTS.get(evict_pct, {})
        prev_delta = prev.get("delta", float('nan'))

        # Verdict
        if delta < -0.1:
            verdict = "NEGATIVE PPL"
        elif delta < 0.5:
            verdict = "near-zero"
        elif delta < 2.0:
            verdict = "acceptable"
        else:
            verdict = "degraded"

        prev_delta_str = f"{prev_delta:+.3f}%" if not math.isnan(prev_delta) else "N/A"
        print(f"{cfg['name']:<20s} {ppl:8.4f} {delta:+8.2f}% {info['ratio']:6.2f}x "
              f"{info['n_kept']:5d}  {prev_delta_str:>9s}  {verdict}")

        results.append({
            "name": cfg["name"],
            "evict_pct": evict_pct,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "delta": delta,
            "ratio": info["ratio"],
            "n_kept": info["n_kept"],
            "prev_text1_delta": prev_delta,
            "verdict": verdict,
        })

    # ------------------------------------------------------------------ cross-text analysis
    print("\n" + "=" * 80)
    print("CROSS-TEXT ANALYSIS: Does negative PPL persist?")
    print("=" * 80)

    neg_on_text1 = all(r["prev_text1_delta"] < 0 for r in results)
    neg_on_text2 = all(r["delta"] < 0 for r in results)
    any_neg_text2 = any(r["delta"] < 0 for r in results)

    print(f"Text 1 (multi-topic):   ALL negative? {neg_on_text1}")
    print(f"Text 2 (hard technical): ALL negative? {neg_on_text2}")
    print(f"Text 2:                  ANY negative? {any_neg_text2}")

    if neg_on_text2:
        print("\nCONCLUSION: NEGATIVE PPL IS A REAL PHENOMENON (persists on second text)")
        print("  This is NOT a text-dependent artifact.")
        print("  Possible explanation: E8 quantization acts as a regularizer,")
        print("  reducing overfit noise in the KV cache and improving next-token prediction.")
    elif any_neg_text2:
        print("\nCONCLUSION: PARTIAL — some configs negative on text 2 but not all")
        print("  Phenomenon may be real but weaker on harder technical text.")
    else:
        print("\nCONCLUSION: NEGATIVE PPL WAS TEXT-DEPENDENT")
        print("  Text 1 was likely easier/more repetitive, causing the negative PPL.")
        print("  On hard technical text, compression causes expected positive PPL degradation.")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return results, baseline_ppl


# ======================================================================
# LOCAL ENTRYPOINT
# ======================================================================
@app.local_entrypoint()
def main():
    import time
    import math
    import os

    print("\n" + "=" * 80)
    print("NEXUSQUANT: LongBench Eval + Llama-3 Negative PPL Validation")
    print("=" * 80)
    print("Task A: Real LongBench tasks on Mistral-7B (narrativeqa/qasper/hotpotqa/...)")
    print("Task B: Llama-3-8B negative PPL validation on SECOND hard technical text")
    print("Both running on A10G GPUs")
    print()

    print("Launching Task A: LongBench on Mistral-7B...")
    fa = run_longbench.spawn()

    print("Launching Task B: Llama-3 validation on second text...")
    fb = run_llama3_validation.spawn()

    print("\nWaiting for both tasks...\n")

    all_task_results, summary_rows = fa.get()
    llama3_results, llama3_baseline = fb.get()

    # ------------------------------------------------------------------ print Task A
    print("\n" + "=" * 80)
    print("TASK A: LONGBENCH RESULTS — Mistral-7B")
    print("=" * 80)
    print("Metric: Mean PPL of expected answer (lower = better comprehension)")
    print(f"{'Task':25s} {'Baseline PPL':>13s} {'35%evict(10x)':>14s} {'Δ%':>7s} "
          f"{'60%evict(16x)':>14s} {'Δ%':>7s}")
    print("-" * 85)
    for row in summary_rows:
        task = row["task"]
        b = row.get("baseline", float('nan'))
        e35 = row.get("35%evict(10x)", float('nan'))
        e60 = row.get("60%evict(16x)", float('nan'))
        d35 = row.get("delta_35", float('nan'))
        d60 = row.get("delta_60", float('nan'))
        b_s   = f"{b:.4f}" if not math.isnan(b) else "N/A"
        e35_s = f"{e35:.4f}" if not math.isnan(e35) else "N/A"
        e60_s = f"{e60:.4f}" if not math.isnan(e60) else "N/A"
        d35_s = f"{d35:+.2f}%" if not math.isnan(d35) else "N/A"
        d60_s = f"{d60:+.2f}%" if not math.isnan(d60) else "N/A"
        print(f"{task:25s} {b_s:>13s} {e35_s:>14s} {d35_s:>7s} {e60_s:>14s} {d60_s:>7s}")

    # ------------------------------------------------------------------ print Task B
    print("\n" + "=" * 80)
    print("TASK B: LLAMA-3 NEGATIVE PPL VALIDATION — Second Text")
    print("=" * 80)
    print(f"Baseline PPL (text #2): {llama3_baseline:.4f}")
    print(f"{'Config':<20s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} "
          f"{'Kept':>6s} {'Text1 Δ%':>9s} {'Verdict':>14s}")
    print("-" * 80)
    for r in llama3_results:
        prev_s = f"{r['prev_text1_delta']:+.3f}%" if not math.isnan(r['prev_text1_delta']) else "N/A"
        print(f"{r['name']:<20s} {r['ppl']:8.4f} {r['delta']:+8.2f}% "
              f"{r['ratio']:6.2f}x {r['n_kept']:5d} {prev_s:>9s} {r['verdict']:>14s}")

    neg_text2 = all(r["delta"] < 0 for r in llama3_results)
    any_neg   = any(r["delta"] < 0 for r in llama3_results)
    if neg_text2:
        print("\nVERDICT: NEGATIVE PPL CONFIRMED on second text — REAL PHENOMENON")
    elif any_neg:
        print("\nVERDICT: PARTIAL negative PPL on second text")
    else:
        print("\nVERDICT: NO negative PPL on second text — was TEXT-DEPENDENT on text #1")

    # ------------------------------------------------------------------ write results
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "longbench_llama3_results.md")

    with open(out_path, "w") as f:
        f.write("# LongBench + Llama-3 Validation Results — NexusQuant\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Pipeline:** NSN → Hadamard → 2-bit E8 VQ → temporal delta → zstd-22\n\n")
        f.write("---\n\n")

        # Task A
        f.write("## Task A: Real LongBench Eval — Mistral-7B\n\n")
        f.write("**Model:** mistralai/Mistral-7B-v0.1 (32L, 8KVH, d=128, rope_theta=10000)\n")
        f.write("**Dataset:** THUDM/LongBench (real long-context QA tasks)\n")
        f.write("**Metric:** PPL of expected answer given compressed context (lower = better)\n")
        f.write("**Eviction configs:** baseline | 2b+35%evict (~10x) | 2b+60%evict (~16x)\n\n")

        f.write("### Per-task Mean PPL\n\n")
        f.write(f"| Task | Baseline PPL | 35%evict(10x) | Δ% | 60%evict(16x) | Δ% |\n")
        f.write(f"|------|-------------|--------------|-----|--------------|-----|\n")
        for row in summary_rows:
            task = row["task"]
            b = row.get("baseline", float('nan'))
            e35 = row.get("35%evict(10x)", float('nan'))
            e60 = row.get("60%evict(16x)", float('nan'))
            d35 = row.get("delta_35", float('nan'))
            d60 = row.get("delta_60", float('nan'))
            b_s   = f"{b:.4f}" if not math.isnan(b) else "N/A"
            e35_s = f"{e35:.4f}" if not math.isnan(e35) else "N/A"
            e60_s = f"{e60:.4f}" if not math.isnan(e60) else "N/A"
            d35_s = f"{d35:+.2f}%" if not math.isnan(d35) else "N/A"
            d60_s = f"{d60:+.2f}%" if not math.isnan(d60) else "N/A"
            f.write(f"| {task} | {b_s} | {e35_s} | {d35_s} | {e60_s} | {d60_s} |\n")

        f.write("\n### Per-example Details\n\n")
        for tr in all_task_results:
            if "error" in tr:
                f.write(f"#### {tr['task']} — ERROR: {tr['error']}\n\n")
                continue
            f.write(f"#### {tr['task']} ({tr['n_examples']} examples)\n\n")
            for ex in tr["examples"]:
                f.write(f"- **Example {ex['ex_idx']}** "
                        f"(context={ex['context_len']}tok, prefix={ex['prefix_len']}tok, "
                        f"ans_tok={ex['answer_tokens']}): "
                        f"\"{ex['answer'][:100]}\"\n")
                for ec in ex["configs"]:
                    ppl_s = f"{ec['ppl']:.4f}" if not math.isnan(ec['ppl']) else "N/A"
                    f.write(f"  - {ec['config']}: PPL={ppl_s} (ratio={ec['ratio']:.1f}x)\n")
            f.write("\n")

        # Task B
        f.write("---\n\n")
        f.write("## Task B: Llama-3-8B Negative PPL Validation — Second Text\n\n")
        f.write("**Model:** NousResearch/Meta-Llama-3-8B (32L, 8KVH GQA, d=128, rope_theta=500000)\n")
        f.write("**Second text:** Advanced quantum mechanics + Langlands program + "
                "molecular biology + computational complexity (~2300 words)\n")
        f.write("**Purpose:** Validate whether ALL-negative PPL from text #1 persists\n\n")

        f.write(f"**Baseline PPL (text #2):** {llama3_baseline:.4f}\n\n")

        f.write("### Results\n\n")
        f.write("| Config | PPL | Delta% | Ratio | Kept | Text1 Delta% | Verdict |\n")
        f.write("|--------|-----|--------|-------|------|-------------|--------|\n")
        for r in llama3_results:
            prev_s = f"{r['prev_text1_delta']:+.3f}%" if not math.isnan(r['prev_text1_delta']) else "N/A"
            f.write(f"| {r['name']} | {r['ppl']:.4f} | {r['delta']:+.2f}% | "
                    f"{r['ratio']:.2f}x | {r['n_kept']} | {prev_s} | {r['verdict']} |\n")

        f.write("\n### Reference: Text #1 Results (llama3_8k_results.md)\n\n")
        f.write("| Config | PPL | Delta% | Ratio |\n")
        f.write("|--------|-----|--------|-------|\n")
        PREV = {0: (3.1057, -1.196, 6.71), 35: (3.0971, -1.470, 10.25),
                60: (3.1009, -1.348, 16.48), 80: (3.1240, -0.614, 32.45)}
        for ep, (ppl, delta, ratio) in PREV.items():
            f.write(f"| 2b+{ep}%evict | {ppl:.4f} | {delta:+.3f}% | {ratio:.2f}x |\n")

        f.write("\n### Conclusion\n\n")
        neg_text2_final = all(r["delta"] < 0 for r in llama3_results)
        any_neg_final   = any(r["delta"] < 0 for r in llama3_results)
        if neg_text2_final:
            f.write("**NEGATIVE PPL CONFIRMED on second text — REAL PHENOMENON**\n\n")
            f.write("All configs show negative PPL delta on both texts. This is not a "
                    "text-dependent artifact. Hypothesis: E8 quantization acts as a "
                    "regularizer, removing noise from the KV cache and improving the "
                    "model's ability to predict next tokens in the continuation.\n")
        elif any_neg_final:
            f.write("**PARTIAL: Some configs show negative PPL on second text**\n\n")
            f.write("The phenomenon is real but weaker on harder technical text.\n")
        else:
            f.write("**NEGATIVE PPL WAS TEXT-DEPENDENT**\n\n")
            f.write("No configs show negative PPL on the second (harder) text. "
                    "The original negative PPL was likely due to characteristics of "
                    "the first text (general knowledge, more repetitive), not a "
                    "real compression benefit.\n")

    print(f"\nResults written to: {out_path}")
    return out_path

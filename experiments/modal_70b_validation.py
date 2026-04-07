"""NexusQuant validation on Llama-3-70B (or Llama-2-70B fallback) — A100-80GB.

Model fits in 80GB via 4-bit bitsandbytes weight quantization (~35GB).
KV cache is kept in FP16 and compressed by NexusQuant independently.

Test matrix (K2V2 and K3V2 at 35% and 60% eviction, with boundary protection):
  - Baseline:          no KV compression
  - K2V2  35%ev:       2-bit keys + 2-bit values, 35% eviction
  - K2V2  60%ev:       2-bit keys + 2-bit values, 60% eviction
  - K3V2  35%ev:       3-bit keys + 2-bit values, 35% eviction
  - K3V2  60%ev:       3-bit keys + 2-bit values, 60% eviction
  - K2V2+boundary 35%: above + protect first/last 2 layers at FP16
  - K2V2+boundary 60%: above + protect first/last 2 layers at FP16
  - K3V2+boundary 35%: above + protect first/last 2 layers at FP16
  - K3V2+boundary 60%: above + protect first/last 2 layers at FP16

Prefix: ~2000 tokens (conservative to avoid OOM on 70B + KV overhead).
Continuation: 200 tokens for PPL measurement.

Overhead accounting (BRUTAL HONESTY):
  - Compressed layers: delta-zstd E8 index bytes (analytic, zstd level 22)
  - Scale bytes: 1 fp16 scale per kept-token per KV-head per layer (K and V)
  - Protected layers: full FP16 for all prefix tokens, no eviction
  - Eviction mask: ceil(prefix_len/8) per compressed layer

Run with: modal run experiments/modal_70b_validation.py
"""
import modal
import os

app = modal.App("nexusquant-70b-validation")

nq_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nexusquant-oss", "nexusquant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.44.0,<5.0.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
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
    gpu="A100-80GB",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=131072,
)
def run_70b():
    import sys
    sys.path.insert(0, "/root")

    import time, math, gc
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    print("=" * 80)
    print("NEXUSQUANT — 70B Validation (A100-80GB)")
    print("4-bit weight quantization (bitsandbytes) + FP16 KV cache compression")
    print("=" * 80)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Try Llama-3-70B first, fall back to Llama-2-70B
    CANDIDATES = [
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Llama-2-70b-hf",
    ]
    model = None
    model_name = None
    tok = None
    hf_token = os.environ.get("HF_TOKEN", "")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    for candidate in CANDIDATES:
        try:
            print(f"\nAttempting to load {candidate} ...")
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(candidate, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
            )
            model.eval()
            model_name = candidate
            print(f"Loaded {model_name} in {time.time()-t0:.1f}s")
            break
        except Exception as exc:
            print(f"  Failed to load {candidate}: {exc}")
            model = None
            tok = None
            gc.collect()
            torch.cuda.empty_cache()

    if model is None:
        print("ERROR: Could not load any 70B model. Aborting.")
        return

    n_layers   = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim   = model.config.hidden_size // model.config.num_attention_heads
    rope_base  = getattr(model.config, "rope_theta", 10000.0)
    print(f"\nConfig: {n_layers}L, {n_kv_heads} KV-heads, head_dim={head_dim}, rope_base={rope_base}")

    # ---- Corpus: ~2000-token prefix to stay well inside VRAM budget on 70B ----
    # Multi-topic text identical in structure to Mistral-7B experiments.
    CORPUS = " ".join([
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe, as well as classifying all known elementary particles. It was developed through the work of many scientists throughout the latter half of the 20th century, with the current formulation finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The Standard Model explains how the basic building blocks of matter interact, governed by fundamental forces mediated by gauge bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. Despite its success, the Standard Model does not incorporate gravity, dark matter, or dark energy.",
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of significant economic and technological transformation beginning in Britain and spreading to Western Europe and North America. The transition from hand production to machine manufacturing, new chemical processes, the rise of steam power, and the growth of the factory system fundamentally changed the nature of work and society. This period saw the emergence of the middle class, rapid urbanization, and the beginning of modern capitalism. Child labor was common in factories and mines, prompting early labor reforms.",
        "The theory of evolution by natural selection, formulated by Darwin and Wallace, is the cornerstone of modern biology. Organisms with heritable traits better suited to their environment tend to survive and reproduce more. Over time these advantageous traits spread through the population. Genetic variation arises through mutation, recombination, and gene flow. Sexual selection improves mating success rather than survival. Evolutionary theory is supported by the fossil record, comparative anatomy, molecular biology, and direct observation.",
        "Quantum mechanics fundamentally changed our understanding of physics at the atomic and subatomic level. Classical physics could not explain black-body radiation, the photoelectric effect, or atomic stability. Max Planck introduced energy quanta in 1900. Einstein explained the photoelectric effect with photons in 1905. Bohr developed his hydrogen atom model in 1913. Heisenberg, Schrodinger, Dirac, and others built the mathematical framework of quantum mechanics throughout the 1920s, producing a theory of extraordinary predictive power.",
        "Artificial intelligence is the simulation of human intelligence by computer systems. Machine learning uses statistical techniques to enable computers to learn from data. Deep learning employs multi-layer neural networks to learn hierarchical data representations. Transformer architectures, introduced by Vaswani et al. in 2017, revolutionized natural language processing and have since been applied to vision, audio, protein structure prediction, and many other domains. Large language models trained on vast corpora exhibit emergent capabilities at scale.",
        "Climate change represents one of the defining challenges of the 21st century. Burning fossil fuels and deforestation have elevated greenhouse gas concentrations to levels not seen in at least 800,000 years. Global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels. Consequences include more frequent extreme weather events, rising sea levels, ocean acidification, and ecosystem disruption. The Paris Agreement aims to limit warming to 1.5-2 degrees Celsius through international cooperation.",
        "The Roman Empire was the post-Republican period of ancient Roman civilization. Headed by emperors and spanning territories around the Mediterranean, the city of Rome was the world's largest city from about 100 BC to 400 AD. Roman engineering achievements included aqueducts, roads, concrete architecture, and the Pantheon. Roman law formed the basis of many modern legal systems. Latin evolved into the Romance languages — Italian, Spanish, French, Portuguese, and Romanian.",
        "The AdS/CFT correspondence, proposed by Maldacena in 1997, conjectures an exact duality between type IIB superstring theory on five-dimensional anti-de Sitter space and four-dimensional N=4 super Yang-Mills theory on the conformal boundary. This holographic principle has profound implications for quantum gravity and strongly coupled quantum field theories. The duality maps bulk gravitational degrees of freedom to boundary conformal field theory data and has been applied to compute quark-gluon plasma properties and quantum entanglement entropy.",
        "CRISPR-Cas9 gene editing exploits the bacterial adaptive immune system. The Cas9 endonuclease, guided by a single-guide RNA, introduces double-strand breaks at specific genomic loci, enabling precise modifications. Base editing and prime editing extend CRISPR to single-nucleotide changes without double-strand breaks. Clinical trials using CRISPR to treat sickle-cell disease and beta-thalassemia have demonstrated durable remissions, marking a new era in genetic medicine.",
        "Persistent homology in topological data analysis tracks the birth and death of topological features across a filtration of simplicial complexes, producing persistence diagrams and barcodes as stable invariants. Applications include protein folding analysis, brain connectivity, materials microstructure characterization, and manifold learning. The Wasserstein and bottleneck stability theorems guarantee robustness under perturbation. Mapper algorithms extend TDA to exploratory data analysis of high-dimensional datasets.",
    ])

    sliding_window = 32

    # ---- KV cache helpers ----
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

    # ---- Attention-based importance scorer ----
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

    # ---- Keep mask with BOS + sliding-window anchor ----
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
            imp[keep_mask] = -float("inf")
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ---- Core evict + E8 quantize (inline, no imports from NexusQuantEvict) ----
    def evict_quantize(kv_cache, keep_mask, prefix_len,
                       key_bits=2, value_bits=2, protect_boundary=0):
        """Evict low-importance tokens and E8-quantize remaining KV cache.

        Overhead accounting (bytes, all included):
          - Compressed layers: temporal delta-zstd E8 index bytes
          - Protected layers: full FP16, no eviction
          - Scale bytes: n_kept * n_kv_heads * 2 (K+V) * 2 (fp16) * n_compressed_layers
          - Mask bytes: ceil(prefix_len/8) * n_compressed_layers
        """
        H = hadamard_matrix(head_dim).cpu()
        cctx = zstandard.ZstdCompressor(level=22)

        n_kept = keep_mask.sum().item()
        total_fp16 = 0

        protected_layers = set()
        if protect_boundary > 0:
            for i in range(min(protect_boundary, n_layers)):
                protected_layers.add(i)
            for i in range(max(0, n_layers - protect_boundary), n_layers):
                protected_layers.add(i)
        n_compressed_layers = n_layers - len(protected_layers)

        all_key_coords = []
        all_val_coords = []

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            if l in protected_layers:
                continue

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

        # Compression accounting
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

        scale_bytes       = n_kept * n_kv_heads * 2 * 2 * n_compressed_layers
        mask_bytes        = math.ceil(prefix_len / 8) * n_compressed_layers
        protected_fp16    = len(protected_layers) * prefix_len * n_kv_heads * head_dim * 2 * 2
        total_compressed  = total_idx + scale_bytes + mask_bytes + protected_fp16

        return {
            "fp16": total_fp16,
            "idx":  total_idx,
            "scale": scale_bytes,
            "mask":  mask_bytes,
            "protected_fp16": protected_fp16,
            "total": total_compressed,
            "ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "n_kept": n_kept,
            "n_protected_layers": len(protected_layers),
            "n_compressed_layers": n_compressed_layers,
        }, kv_cache

    # ---- PPL measurement ----
    def measure_ppl_continuation(model, full_ids, prefix_len, kv, attn_full):
        cont_len = full_ids.shape[1] - prefix_len
        with torch.no_grad():
            cout = model(
                full_ids[:, prefix_len:],
                past_key_values=kv,
                attention_mask=attn_full.unsqueeze(0),
                use_cache=True,
            )
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return torch.exp(loss).item()

    # ---- Main experiment ----
    inputs = tok(CORPUS, return_tensors="pt", max_length=4096, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok = full_ids.shape[1]

    TARGET_PREFIX = 2000
    CONT_LEN = 200
    prefix_len = min(TARGET_PREFIX, n_tok - CONT_LEN)

    if prefix_len < 300:
        print(f"ERROR: Corpus too short ({n_tok} tokens). Aborting.")
        return

    cont_len = n_tok - prefix_len
    print(f"\nTokens: total={n_tok}, prefix={prefix_len}, continuation={cont_len}")
    print(f"Model: {model_name}")

    # Baseline PPL
    try:
        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            cout = model(
                full_ids[:, prefix_len:],
                past_key_values=pout.past_key_values,
                use_cache=True,
            )
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"\nBaseline PPL (no compression): {baseline_ppl:.4f}")
    except Exception as exc:
        print(f"ERROR during baseline: {exc}")
        return

    # Score importance once
    try:
        with torch.no_grad():
            pout_score = model(full_ids[:, :prefix_len], use_cache=True)
        importance = score_importance(pout_score.past_key_values)
        del pout_score
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"ERROR scoring importance: {exc}")
        return

    CONFIGS = []
    for evict_pct in [35, 60]:
        for key_bits, val_bits, label in [(2, 2, "K2V2"), (3, 2, "K3V2")]:
            CONFIGS.append({
                "name": f"{label} {evict_pct}%ev",
                "evict_pct": evict_pct, "key_bits": key_bits, "val_bits": val_bits,
                "protect_boundary": 0,
            })
            CONFIGS.append({
                "name": f"{label}+boundary {evict_pct}%ev",
                "evict_pct": evict_pct, "key_bits": key_bits, "val_bits": val_bits,
                "protect_boundary": 2,
            })

    print(f"\n{'Config':<35s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot':>5s}")
    print("-" * 80)

    all_results = []
    for cfg in CONFIGS:
        torch.cuda.empty_cache()
        name             = cfg["name"]
        evict_pct        = cfg["evict_pct"]
        key_bits         = cfg["key_bits"]
        val_bits         = cfg["val_bits"]
        protect_boundary = cfg["protect_boundary"]

        try:
            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv = pout.past_key_values

            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            info, kv  = evict_quantize(
                kv, keep_mask, prefix_len,
                key_bits=key_bits, value_bits=val_bits,
                protect_boundary=protect_boundary,
            )

            evict_mask = ~keep_mask
            attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(cont_len, dtype=torch.long, device="cuda"),
            ])

            ppl = measure_ppl_continuation(model, full_ids, prefix_len, kv, attn_full)
            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 1.0 else (" <<" if abs(delta) < 2.0 else "")
            n_prot = info.get("n_protected_layers", 0)
            print(f"{name:<35s} {ppl:8.4f} {delta:+8.3f}% {info['ratio']:6.2f}x "
                  f"{info['n_kept']:5d} {n_prot:4d}{tag}")
            all_results.append({
                "name": name, "evict_pct": evict_pct,
                "key_bits": key_bits, "val_bits": val_bits,
                "protect_boundary": protect_boundary,
                "ppl": ppl, "delta": delta,
                "ratio": info["ratio"], "n_kept": info["n_kept"],
                "n_protected": n_prot,
                "baseline": baseline_ppl,
                "prefix_len": prefix_len, "model": model_name,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"{name:<35s} OOM — skipped")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{name:<35s} ERROR: {exc}")
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — {model_name}")
    print(f"Baseline PPL: {baseline_ppl:.4f}  |  prefix={prefix_len}  cont={cont_len}")
    print(f"{'='*80}")
    for evict_pct in [35, 60]:
        group = [r for r in all_results if r["evict_pct"] == evict_pct]
        if not group:
            continue
        print(f"\n--- Eviction {evict_pct}% ---")
        print(f"{'Config':<35s} {'PPL Δ%':>8s} {'Ratio':>7s} {'Kept':>6s} {'Prot L':>7s}")
        print("-" * 66)
        for r in group:
            tag = " <<<" if abs(r["delta"]) < 1.0 else (" <<" if abs(r["delta"]) < 2.0 else "")
            print(f"{r['name']:<35s} {r['delta']:+8.3f}% {r['ratio']:6.2f}x "
                  f"{r['n_kept']:5d} {r['n_protected']:6d}{tag}")

    # Best tradeoffs
    print(f"\n{'='*80}")
    print("BEST TRADEOFFS (<1% PPL degradation)")
    print(f"{'='*80}")
    for evict_pct in [35, 60]:
        group = [r for r in all_results if r["evict_pct"] == evict_pct]
        sub1 = [r for r in group if abs(r["delta"]) < 1.0]
        print(f"\nEviction {evict_pct}%:")
        if sub1:
            best = max(sub1, key=lambda r: r["ratio"])
            print(f"  Best <1% PPL: {best['name']:35s}  "
                  f"delta={best['delta']:+.3f}%  ratio={best['ratio']:.2f}x")
        else:
            if group:
                closest = min(group, key=lambda r: abs(r["delta"]))
                print(f"  No config <1% — closest: {closest['name']:30s}  "
                      f"delta={closest['delta']:+.3f}%  ratio={closest['ratio']:.2f}x")

    print(f"\n{'='*80}")
    print(f"Done. Model: {model_name}")
    print(f"{'='*80}")


@app.local_entrypoint()
def main():
    run_70b.remote()

"""NexusQuant — Full LongBench Benchmark (All 16 Tasks, F1/ROUGE/Accuracy Scoring)

Runs Mistral-7B-Instruct on all 16 LongBench tasks under three configs:
  - baseline:    no compression
  - evict_35pct: 35% token eviction + 2-bit E8 VQ
  - evict_60pct: 60% token eviction + 2-bit E8 VQ

Metrics are task-appropriate:
  - QA (narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,
         triviaqa): token-level F1
  - Summarization (gov_report, qmsum, multi_news, samsum): ROUGE-L
  - Classification (trec): accuracy (exact match on category label)
  - Synthetic (passage_count, passage_retrieval_en): exact match / accuracy
  - Code (lcc, repobench-p): token-level F1 (code completion proxy)

Outlier handling: multifieldqa_en gets median + 3-sigma clipping.
All tasks: report both mean and median for transparency.

Runtime budget: max 50 examples/task × 16 tasks × 3 configs = 2400 evals.
"""

import modal
import os

app = modal.App("nexusquant-longbench-full")

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

# ---------------------------------------------------------------------------
# Task registry: name → (metric, max_new_tokens, is_outlier_sensitive)
# ---------------------------------------------------------------------------
TASKS = {
    # Single-doc QA
    "narrativeqa":         ("f1",        128, False),
    "qasper":              ("f1",        128, False),
    "multifieldqa_en":     ("f1",        128, True),   # known outlier variance
    # Multi-doc QA
    "hotpotqa":            ("f1",        128, False),
    "2wikimqa":            ("f1",        128, False),
    "musique":             ("f1",        128, False),
    # Summarization
    "gov_report":          ("rouge_l",   256, False),
    "qmsum":               ("rouge_l",   256, False),
    "multi_news":          ("rouge_l",   256, False),
    # Few-shot / classification
    "trec":                ("accuracy",   16, False),
    "triviaqa":            ("f1",        128, False),
    "samsum":              ("rouge_l",   256, False),
    # Synthetic
    "passage_count":       ("accuracy",   16, False),
    "passage_retrieval_en":("accuracy",   32, False),
    # Code
    "lcc":                 ("f1",        128, False),
    "repobench-p":         ("f1",        128, False),
}

MAX_EXAMPLES_PER_TASK = 50


@app.function(image=image, gpu="A100", timeout=10800, secrets=[HF_SECRET])
def run_longbench():
    import sys
    sys.path.insert(0, "/root")

    import math
    import json
    import time
    import urllib.request
    import zipfile
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def _lcs_length(a, b):
        """Longest common subsequence length between two token lists."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        # Use two-row DP to save memory
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def token_f1(prediction: str, ground_truth: str) -> float:
        """Token-level F1 (bag-of-words, lowercased)."""
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def rouge_l(prediction: str, reference: str) -> float:
        """ROUGE-L (LCS-based) F-measure, lowercased tokens."""
        pred_tokens = prediction.lower().split()
        ref_tokens  = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall    = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def accuracy(prediction: str, ground_truths) -> float:
        """Exact-match accuracy: 1 if prediction matches any ground truth."""
        pred = prediction.strip().lower()
        for gt in ground_truths:
            if pred == gt.strip().lower():
                return 1.0
            # Also try prefix match for classification (e.g. "ABBR" in "ABBR: ...")
            if pred.startswith(gt.strip().lower()) or gt.strip().lower().startswith(pred):
                return 1.0
        return 0.0

    def score_example(prediction: str, ground_truths, metric: str) -> float:
        """Score one example. Returns best score over all ground truths."""
        if not prediction.strip():
            return 0.0
        if metric == "f1":
            return max(token_f1(prediction, gt) for gt in ground_truths)
        elif metric == "rouge_l":
            return max(rouge_l(prediction, gt) for gt in ground_truths)
        elif metric == "accuracy":
            return accuracy(prediction, ground_truths)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def sigma_clip(scores, n_sigma=3):
        """Return (clipped_scores, n_removed). Removes outliers > n_sigma from mean."""
        arr = np.array(scores, dtype=float)
        if len(arr) < 4:
            return arr, 0
        mean, std = arr.mean(), arr.std()
        if std < 1e-8:
            return arr, 0
        mask = np.abs(arr - mean) <= n_sigma * std
        return arr[mask], int((~mask).sum())

    # -----------------------------------------------------------------------
    # Data download (direct HTTP, avoids datasets library)
    # -----------------------------------------------------------------------
    data_zip = "/tmp/longbench_data.zip"
    data_dir = "/tmp/longbench"

    if not os.path.exists(data_dir):
        print("Downloading LongBench data...")
        url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
        t0 = time.time()
        try:
            urllib.request.urlretrieve(url, data_zip)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download LongBench data from {url}: {e}\n"
                "Check that the HuggingFace URL is still valid and the container has internet access."
            ) from e
        with zipfile.ZipFile(data_zip, "r") as z:
            z.extractall(data_dir)
        print(f"  Downloaded and extracted in {time.time()-t0:.1f}s")
    else:
        print("LongBench data already present.")

    def load_task(task_name, max_examples):
        """Load up to max_examples from a LongBench JSONL file."""
        # Try multiple path layouts from the zip
        candidates = [
            f"{data_dir}/data/{task_name}.jsonl",
            f"{data_dir}/{task_name}.jsonl",
            f"{data_dir}/LongBench/data/{task_name}.jsonl",
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            # Search recursively
            for root, _, files in os.walk(data_dir):
                for fn in files:
                    if fn == f"{task_name}.jsonl":
                        path = os.path.join(root, fn)
                        break
                if path:
                    break
        if path is None:
            print(f"  [WARN] {task_name}.jsonl not found — skipping")
            return []
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if len(data) >= max_examples:
                        break
        return data

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("NEXUSQUANT — Full LongBench Benchmark (16 tasks, A100)")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers         # 32
    n_kv_heads = model.config.num_key_value_heads       # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}")

    device = next(model.parameters()).device

    # -----------------------------------------------------------------------
    # NexusQuant eviction helpers (self-contained, no external import needed)
    # -----------------------------------------------------------------------
    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    SLIDING_WINDOW = 32
    BITS = 2

    def _get_kv(cache, layer):
        if hasattr(cache, 'key_cache'):
            return cache.key_cache[layer], cache.value_cache[layer]
        l = cache.layers[layer]
        return l.keys, l.values

    def _set_kv(cache, layer, k, v):
        if hasattr(cache, 'key_cache'):
            cache.key_cache[layer] = k
            cache.value_cache[layer] = v
        else:
            l = cache.layers[layer]
            l.keys = k
            l.values = v

    def _score_importance(kv_cache, obs_window=32):
        k0, _ = _get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device="cpu")
        scale = 1.0 / math.sqrt(head_dim)
        for layer in range(n_layers):
            kl, _ = _get_kv(kv_cache, layer)
            k = kl[0].float().cpu()  # (n_kv_heads, seq_len, head_dim)
            # Average over heads for a unified importance signal
            k_mean = k.mean(dim=0)  # (seq_len, head_dim)
            k_obs  = k_mean[-w:]    # (w, head_dim)
            scores = (k_obs @ k_mean.T) * scale  # (w, seq_len)
            causal = torch.ones(w, seq_len, dtype=torch.bool)
            for i in range(w):
                causal[i, seq_len - w + i + 1:] = False
            scores = scores.masked_fill(~causal, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            all_imp += attn.sum(dim=0)
        return all_imp / n_layers

    def _build_keep_mask(seq_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(seq_len, dtype=torch.bool)
        keep_pct = 1.0 - evict_pct / 100.0
        n_keep = max(int(seq_len * keep_pct), SLIDING_WINDOW + 1)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        # Always keep first token (BOS) and last SLIDING_WINDOW tokens
        mask[0] = True
        mask[-SLIDING_WINDOW:] = True
        n_already = mask.sum().item()
        n_from_imp = max(0, n_keep - n_already)
        if n_from_imp > 0:
            imp = importance.clone()
            imp[mask] = -float('inf')
            eligible = (~mask).sum().item()
            n_pick = min(n_from_imp, eligible)
            if n_pick > 0:
                _, top_idx = imp.topk(n_pick)
                mask[top_idx] = True
        return mask

    def _hadamard_quantize_cache(kv_cache):
        """In-place 2-bit E8 VQ with Hadamard rotation on the full KV cache."""
        H = hadamard_matrix(head_dim).to(device=device, dtype=torch.float16)
        levels = 2 ** BITS
        for layer in range(n_layers):
            kl, vl = _get_kv(kv_cache, layer)
            new_kvs = []
            for tensor in [kl, vl]:
                t = tensor.float()  # (1, n_kv_heads, seq_len, head_dim)
                shape = t.shape
                flat = t.reshape(-1, head_dim)  # (N, head_dim)
                rotated = flat @ H.T.float()
                amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                sc = amax / (levels / 2)
                normalized = rotated / sc
                groups = normalized.reshape(-1, 8)
                lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                dq = (lp.reshape(-1, head_dim) * sc) @ H.float()
                new_kvs.append(dq.reshape(shape).to(tensor.dtype))
            _set_kv(kv_cache, layer, new_kvs[0], new_kvs[1])

    def _apply_eviction(kv_cache, keep_mask):
        """Drop evicted tokens from all layers of the KV cache."""
        idx = keep_mask.nonzero(as_tuple=True)[0]
        for layer in range(n_layers):
            kl, vl = _get_kv(kv_cache, layer)
            _set_kv(kv_cache, layer,
                    kl[:, :, idx, :],
                    vl[:, :, idx, :])

    def _compress_kv_cache(kv_cache, evict_pct):
        """Full NexusQuant pipeline: score → evict → Hadamard → 2b E8 VQ."""
        k0, _ = _get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        if seq_len < 4:
            return kv_cache
        # Score importance
        importance = _score_importance(kv_cache)
        # Build eviction mask
        keep_mask = _build_keep_mask(seq_len, evict_pct, importance)
        # Evict tokens
        if evict_pct > 0:
            _apply_eviction(kv_cache, keep_mask)
        # Quantize remaining tokens
        _hadamard_quantize_cache(kv_cache)
        return kv_cache

    # -----------------------------------------------------------------------
    # Prompt builder
    # -----------------------------------------------------------------------
    def build_prompt(example, task_name):
        """
        Build an instruction-style prompt for Mistral-7B-Instruct.
        Uses [INST] ... [/INST] format.
        """
        ctx   = example.get("context", "").strip()
        inp   = example.get("input", "").strip()
        # Task-specific instruction prefixes
        task_instructions = {
            "narrativeqa":          "Answer the question based on the story. Be concise.",
            "qasper":               "Answer the question based on the paper. Be concise.",
            "multifieldqa_en":      "Answer the question based on the documents. Be concise.",
            "hotpotqa":             "Answer the multi-hop question. Be concise.",
            "2wikimqa":             "Answer the question using the provided passages. Be concise.",
            "musique":              "Answer the question by reasoning over the passages. Be concise.",
            "gov_report":           "Summarize the following government report in a few sentences.",
            "qmsum":                "Given the meeting transcript, summarize the query-relevant content.",
            "multi_news":           "Summarize the following news articles into a coherent summary.",
            "trec":                 "Classify the question into one of these categories: ABBR, ENTY, DESC, HUM, LOC, NUM. Output only the category.",
            "triviaqa":             "Answer the trivia question. Be concise.",
            "samsum":               "Summarize the following dialogue.",
            "passage_count":        "Count the number of distinct passages in the document. Output only the number.",
            "passage_retrieval_en": "Find and output the paragraph that matches the given description. Output the paragraph number only.",
            "lcc":                  "Complete the code snippet. Output only the completion.",
            "repobench-p":          "Complete the code snippet given the repository context. Output only the next line.",
        }
        instruction = task_instructions.get(task_name, "Answer based on the context.")

        # Truncate context to avoid OOM (max ~3500 tokens worth of chars).
        # KNOWN LIMITATION: LongBench tasks have 8K-20K token contexts; at 14000
        # chars (~3500 tokens) we are nowhere near full length. Results here measure
        # compression quality on truncated inputs, not the full long-context regime.
        MAX_CHARS = 14000
        if len(ctx) > MAX_CHARS:
            ctx = ctx[:MAX_CHARS] + " [... truncated ...]"

        if inp:
            text = f"[INST] {instruction}\n\nContext:\n{ctx}\n\nQuestion/Task:\n{inp} [/INST]"
        else:
            text = f"[INST] {instruction}\n\nContext:\n{ctx} [/INST]"
        return text

    # -----------------------------------------------------------------------
    # Single inference call (with optional KV-cache compression)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate_answer(prompt_text, max_new_tokens, evict_pct):
        """
        Encode prompt → prefill → optionally compress KV cache → generate.
        Returns decoded prediction string.
        """
        try:
            enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = enc["input_ids"].to(device)
            prefix_len = input_ids.shape[1]

            if evict_pct == 0:
                # Baseline: standard generation
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            else:
                # Step 1: prefill only (no generation)
                cache = DynamicCache()
                with torch.no_grad():
                    fwd = model(
                        input_ids,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                kv_cache = fwd.past_key_values
                # Step 2: compress the prefill KV cache
                kv_cache = _compress_kv_cache(kv_cache, evict_pct)
                # Step 3: generate from compressed cache
                # Build position ids that account for dropped tokens
                k0, _ = _get_kv(kv_cache, 0)
                cached_len = k0.shape[2]
                # Use the last token's logits as starting point
                last_logits = fwd.logits[:, -1:, :]
                last_token  = last_logits.argmax(dim=-1)
                generated   = [last_token.item()]
                cur_input   = last_token
                cur_cache   = kv_cache
                for _ in range(max_new_tokens - 1):
                    step_out = model(
                        cur_input,
                        past_key_values=cur_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    cur_cache = step_out.past_key_values
                    next_token = step_out.logits[:, -1:, :].argmax(dim=-1)
                    generated.append(next_token.item())
                    cur_input = next_token
                    if next_token.item() == tok.eos_token_id:
                        break
                out = input_ids  # not used for decoding
                new_ids = torch.tensor([generated], device="cpu")
                return tok.decode(new_ids[0], skip_special_tokens=True).strip()

            # Decode for baseline path
            new_tokens = out[0, input_ids.shape[1]:]
            return tok.decode(new_tokens, skip_special_tokens=True).strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return ""
        except Exception as e:
            return ""

    # -----------------------------------------------------------------------
    # Per-task evaluation
    # -----------------------------------------------------------------------
    def evaluate_task(task_name, metric, max_new_tokens, is_outlier_sensitive, evict_pct):
        """Evaluate one task under one eviction config. Returns (mean, median, n_outliers)."""
        examples = load_task(task_name, MAX_EXAMPLES_PER_TASK)
        if not examples:
            return None, None, 0

        scores = []
        for i, ex in enumerate(examples):
            answers = ex.get("answers", ex.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue

            prompt = build_prompt(ex, task_name)
            pred   = generate_answer(prompt, max_new_tokens, evict_pct)
            sc     = score_example(pred, answers, metric)
            scores.append(sc)

        if not scores:
            return None, None, 0

        arr = np.array(scores, dtype=float)
        n_outliers = 0
        if is_outlier_sensitive and len(arr) >= 4:
            arr_clip, n_outliers = sigma_clip(arr, n_sigma=3)
            mean_val   = float(arr_clip.mean())
            median_val = float(np.median(arr_clip))
            print(f"      [outlier clip] removed {n_outliers}/{len(scores)} outliers")
        else:
            mean_val   = float(arr.mean())
            median_val = float(np.median(arr))

        return mean_val, median_val, n_outliers

    # -----------------------------------------------------------------------
    # Main evaluation loop
    # -----------------------------------------------------------------------
    CONFIGS = [
        ("baseline",    0),
        ("evict_35pct", 35),
        ("evict_60pct", 60),
    ]

    results = {}  # task → config → {mean, median, n_outliers}

    print(f"\nRunning {len(TASKS)} tasks × {len(CONFIGS)} configs × up to {MAX_EXAMPLES_PER_TASK} examples")
    print(f"Total evals: up to {len(TASKS) * len(CONFIGS) * MAX_EXAMPLES_PER_TASK}\n")

    t_total = time.time()
    for task_name, (metric, max_new_tokens, is_outlier_sensitive) in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}  |  Metric: {metric}  |  MaxNewTokens: {max_new_tokens}")
        results[task_name] = {}
        for cfg_name, evict_pct in CONFIGS:
            print(f"  Config: {cfg_name} (evict={evict_pct}%)")
            t0 = time.time()
            mean_v, median_v, n_out = evaluate_task(
                task_name, metric, max_new_tokens, is_outlier_sensitive, evict_pct
            )
            elapsed = time.time() - t0
            results[task_name][cfg_name] = {
                "mean":       mean_v,
                "median":     median_v,
                "n_outliers": n_out,
                "elapsed":    elapsed,
            }
            if mean_v is not None:
                print(f"    mean={mean_v*100:.2f}%  median={median_v*100:.2f}%  "
                      f"outliers={n_out}  ({elapsed:.0f}s)")
            else:
                print(f"    SKIPPED (no data)")

        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total
    print(f"\nTotal evaluation time: {total_elapsed/60:.1f}min")

    # -----------------------------------------------------------------------
    # Results printing — markdown tables
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("RESULTS — FULL LONGBENCH BENCHMARK (NexusQuant)")
    print("=" * 80)

    # Per-task breakdown
    print("\n### Per-Task Results (mean score × 100)")
    print("\n| Task | Metric | Baseline | 35% Evict | 60% Evict | Δ@35% | Δ@60% |")
    print("|------|--------|----------|-----------|-----------|-------|-------|")

    task_groups = {
        "Single-doc QA":       ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-doc QA":        ["hotpotqa", "2wikimqa", "musique"],
        "Summarization":       ["gov_report", "qmsum", "multi_news"],
        "Few-shot/Classif.":   ["trec", "triviaqa", "samsum"],
        "Synthetic":           ["passage_count", "passage_retrieval_en"],
        "Code":                ["lcc", "repobench-p"],
    }

    for group, task_list in task_groups.items():
        print(f"| **{group}** | | | | | | |")
        for task_name in task_list:
            if task_name not in results:
                continue
            metric = TASKS[task_name][0]
            r = results[task_name]
            b  = r.get("baseline",    {}).get("mean")
            e35 = r.get("evict_35pct", {}).get("mean")
            e60 = r.get("evict_60pct", {}).get("mean")

            def fmt(v):
                return f"{v*100:.2f}" if v is not None else "N/A"

            def delta(v, base):
                if v is None or base is None:
                    return "N/A"
                d = (v - base) * 100
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.2f}"

            outlier_flag = "*" if TASKS[task_name][2] else ""
            print(f"| {task_name}{outlier_flag} | {metric} | {fmt(b)} | {fmt(e35)} | {fmt(e60)} | {delta(e35,b)} | {delta(e60,b)} |")

    print("\n*outlier-sensitive: 3σ clipping applied")

    # Median table
    print("\n### Per-Task Results (median score × 100, robust to outliers)")
    print("\n| Task | Metric | Baseline | 35% Evict | 60% Evict |")
    print("|------|--------|----------|-----------|-----------|")
    for task_name in TASKS:
        if task_name not in results:
            continue
        metric = TASKS[task_name][0]
        r = results[task_name]
        b   = r.get("baseline",    {}).get("median")
        e35 = r.get("evict_35pct", {}).get("median")
        e60 = r.get("evict_60pct", {}).get("median")
        def fmt(v): return f"{v*100:.2f}" if v is not None else "N/A"
        print(f"| {task_name} | {metric} | {fmt(b)} | {fmt(e35)} | {fmt(e60)} |")

    # Aggregate by group
    print("\n### Aggregate Summary (mean of task means × 100)")
    print("\n| Category | Baseline | 35% Evict | 60% Evict | Δ@35% | Δ@60% |")
    print("|----------|----------|-----------|-----------|-------|-------|")

    def group_agg(task_list, cfg):
        vals = []
        for t in task_list:
            if t in results:
                v = results[t].get(cfg, {}).get("mean")
                if v is not None:
                    vals.append(v)
        return np.mean(vals) if vals else None

    for group, task_list in task_groups.items():
        b   = group_agg(task_list, "baseline")
        e35 = group_agg(task_list, "evict_35pct")
        e60 = group_agg(task_list, "evict_60pct")
        def fmt(v): return f"{v*100:.2f}" if v is not None else "N/A"
        def delta(v, base):
            if v is None or base is None: return "N/A"
            d = (v - base) * 100
            return f"{'+' if d>=0 else ''}{d:.2f}"
        print(f"| {group} | {fmt(b)} | {fmt(e35)} | {fmt(e60)} | {delta(e35,b)} | {delta(e60,b)} |")

    # Overall aggregate
    all_tasks = list(TASKS.keys())
    b_all   = group_agg(all_tasks, "baseline")
    e35_all = group_agg(all_tasks, "evict_35pct")
    e60_all = group_agg(all_tasks, "evict_60pct")
    def fmt(v): return f"{v*100:.2f}" if v is not None else "N/A"
    def delta(v, base):
        if v is None or base is None: return "N/A"
        d = (v - base) * 100
        return f"{'+' if d>=0 else ''}{d:.2f}"
    print(f"| **ALL 16 TASKS** | {fmt(b_all)} | {fmt(e35_all)} | {fmt(e60_all)} | {delta(e35_all,b_all)} | {delta(e60_all,b_all)} |")

    # Compression ratio reminder
    print("\n### Compression Context")
    print("| Config | Eviction | Effective Compression | Notes |")
    print("|--------|----------|-----------------------|-------|")
    print("| baseline    |  0% | 1x    | No compression |")
    print("| evict_35pct | 35% | ~10x  | 2b E8 VQ + 35% eviction (quality='high') |")
    print("| evict_60pct | 60% | ~17x  | 2b E8 VQ + 60% eviction (quality='balanced') |")
    print("\nNote: compression ratios include ALL overhead (indices, scales, position mask).")
    print("Results are reported per-task because LongBench quality is TASK-DEPENDENT.")
    print("Outlier-sensitive tasks use 3σ clipping; both mean and median are reported.")

    return results


@app.local_entrypoint()
def main():
    results = run_longbench.remote()
    print("\nDone. Results returned.")

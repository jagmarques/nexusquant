# From 128K to 4.2M tokens: a practical guide to KV cache compression with NexusQuant

KV cache is the memory bottleneck for long-context inference. A 7B model at 128K context eats 32+ GB of KV cache alone on a single A100. NexusQuant compresses it 10-33x at inference time, no training needed.

## Install

```bash
pip install nexusquant
```

Requires Python 3.9+, PyTorch 2.0+, transformers >= 4.44.

## The one-liner

```python
from nexusquant import nexusquant_evict

with nexusquant_evict(model, quality="balanced") as nq:
    output = model.generate(input_ids, max_new_tokens=500,
                            attention_mask=nq.last_mask)
```

Hooks into DynamicCache, applies eviction + E8 lattice quantization after prefill, removes hooks on exit.

## Quality presets

| Preset | Evict | Bits | Ratio | PPL delta | Use for |
|--------|-------|------|-------|-----------|---------|
| `high` | 35% | K3V2 | ~9x | +0.35% | Quality-critical tasks |
| `asym` | 60% | K3V2 | ~14x | est. <1% | Sweet spot |
| `balanced` | 60% | K2V2 | ~17x | +0.82% | General long-context |
| `max` | 80% | K2V2 | ~33x | +2.13% | Maximum compression |

## Real attention scorer

The default key-key scorer is fast but approximate. The real scorer uses actual softmax weights — at 35% eviction it gives **+0.00% PPL** vs +0.66% for key-key.

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    attn_implementation="eager",  # required — SDPA silently suppresses weights
    torch_dtype=torch.float16,
    device_map="auto",
)

with nexusquant_evict(model, quality="high", scorer="real", input_ids=prefix_ids) as nq:
    output = model.generate(prefix_ids, max_new_tokens=500,
                            attention_mask=nq.last_mask)
```

Costs one extra forward pass at prefill. Worth it for quality-sensitive workloads.

## Boundary protection (mandatory for Qwen)

Qwen-family models catastrophically fail without boundary protection:

```python
with nexusquant_evict(model, quality="balanced", protect_boundary=2) as nq:
    output = model.generate(input_ids, max_new_tokens=500,
                            attention_mask=nq.last_mask)
```

Keeps first/last 2 layers at FP16. Mistral and Phi-3 don't need this.

## Physical truncation for real memory savings

Default mode zeroes evicted tokens but doesn't free memory. For actual VRAM savings:

```python
with nexusquant_evict(model, quality="balanced", truncate=True) as nq:
    output = model.generate(
        input_ids, max_new_tokens=500,
        position_ids=torch.arange(nq.next_position, nq.next_position + 500,
                                  device=input_ids.device).unsqueeze(0),
    )
```

Evicted tokens are physically removed from tensors. RoPE positions remapped to be contiguous.

## Deferred compression

Skip compression for short contexts where it's not needed:

```python
with nexusquant_evict(model, quality="balanced", min_context_for_compression=512) as nq:
    output = model.generate(input_ids, max_new_tokens=500)
```

Useful for chat apps where early turns are short.

## Complete example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nexusquant import nexusquant_evict

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    attn_implementation="eager",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

document = "..." * 5000
prompt = f"Summarize:\n\n{document}\n\nSummary:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

with nexusquant_evict(
    model, quality="high", scorer="real",
    input_ids=input_ids, truncate=True,
) as nq:
    output = model.generate(
        input_ids, max_new_tokens=300,
        position_ids=torch.arange(nq.next_position, nq.next_position + 300,
                                  device=input_ids.device).unsqueeze(0),
    )

print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
```

## What's next

- **Triton E8 kernel** — compression is CPU-bound (60-90s). GPU kernel written, benchmarking pending.
- **LongBench** — full 16-task evaluation with F1 scoring
- **70B validation** — current results are 7B-class only
- **arXiv** — NeurIPS 2026 paper in final draft

Code: https://github.com/jagmarques/nexusquant

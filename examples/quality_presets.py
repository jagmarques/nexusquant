"""Compare all three quality presets on the same prompt."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nexusquant import nexusquant_evict

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL)

# Use a prompt long enough to give the importance scorer real signal.
# Short prefixes (<500 tokens) see more degradation than the numbers below.
prompt = (
    "Large language models have transformed natural language processing. "
    "Attention mechanisms enable models to weigh the relevance of each token "
    "when generating the next one. As context lengths grow, the key-value cache "
    "that stores intermediate attention state becomes the dominant memory cost. "
    "This example shows how NexusQuant reduces that cost at inference time. "
) * 8  # ~640 tokens

input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

PRESETS = [
    ("high",     "10x compression  / +0.4% PPL"),
    ("balanced", "17x compression  / +1.3% PPL"),
    ("max",      "33x compression  / +2.6% PPL"),
]

print(f"Prompt length: {input_ids.shape[1]} tokens\n")

for preset, description in PRESETS:
    with nexusquant_evict(model, quality=preset):
        out = model.generate(input_ids, max_new_tokens=40, do_sample=False)

    continuation = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[{preset}] {description}")
    print(f"  {continuation.strip()}\n")

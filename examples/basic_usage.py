"""Compress KV cache with one line."""
from transformers import AutoModelForCausalLM, AutoTokenizer
from nexusquant import nexusquant_evict

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

input_ids = tok("The meaning of life is", return_tensors="pt").input_ids

with nexusquant_evict(model, quality="balanced"):
    output = model.generate(input_ids, max_new_tokens=50)

print(tok.decode(output[0]))

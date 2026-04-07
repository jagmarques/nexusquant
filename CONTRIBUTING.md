# Contributing

## Setup

```bash
git clone https://github.com/jagmarques/nexusquant
cd nexusquant
pip install -e ".[hf]"
```

## Running tests

```bash
python -m pytest nexusquant/test_core.py -v
```

Tests run on CPU with a synthetic cache — no GPU or model download required.

## Code style

- Black-formatted, 88-char line limit: `pip install black && black .`
- No type: ignore comments. Fix the types.
- Every new function needs a one-line docstring. Nothing more.
- If a change touches the compression pipeline, add a regression test that catches regressions in the PPL numbers table in README.md.

## What to contribute

Good bets:
- GPU kernels (Triton/CUDA) for the E8 nearest-point computation
- Support for interleaved-RoPE models (GPT-NeoX, GPT-J)
- Benchmarks on additional model families

Not useful right now:
- Additional quantization formats that don't beat E8 at the same bit-width
- New eviction heuristics without a measured PPL comparison against the current one

## PR process

1. Open an issue first for anything non-trivial.
2. Keep PRs focused — one change per PR.
3. Include a benchmark table if you're touching the compression pipeline.
4. PRs that add complexity without measurable gain will not be merged.

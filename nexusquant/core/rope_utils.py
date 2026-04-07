"""RoPE (Rotary Position Embedding) removal and re-application utilities.

HuggingFace Mistral/Llama use SPLIT-HALF RoPE: dim i paired with dim i+d/2.
Removing RoPE before PCA reduces key effective rank, enabling more aggressive
DP dimension dropping (8x at +0.45% vs +0.83% with RoPE on 50 passages).

Supported models (split-half RoPE):
    - Llama family (Llama-2, Llama-3, Llama-3.1)
    - Mistral family (Mistral-7B, Mixtral)
    - Qwen family
    - Phi family
    - Gemma family

NOT supported (interleaved RoPE, dim 2i paired with dim 2i+1):
    - GPT-NeoX
    - GPT-J
    - CodeGen
    These models pair consecutive dimensions (2i, 2i+1) instead of split halves.
    Using inverse_rope/forward_rope on these models will produce INCORRECT results.

Note on rope_scaling: Llama-3.1 and some other models use rope_scaling config
for extended context (e.g., linear or dynamic NTK scaling). This changes the
effective theta values at longer context lengths. The current implementation
reads rope_theta but does NOT apply rope_scaling corrections. At standard
context lengths (<= original training length), this has no effect. At extended
context lengths, the frequency mismatch may introduce small errors.
"""

import torch


def inverse_rope(keys: torch.Tensor, seq_offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    """Remove RoPE from key vectors (split-half style).

    Implements the inverse rotation for split-half RoPE layout where dim i
    is paired with dim i+d/2. This is the layout used by Llama, Mistral,
    Qwen, Phi, and Gemma. Do NOT use with interleaved-RoPE models (GPT-NeoX,
    GPT-J) -- see module docstring for details.

    Args:
        keys: (heads, seq, dim) or (batch, heads, seq, dim) tensor
        seq_offset: Starting position index
        base: RoPE frequency base (rope_theta from model config)

    Returns:
        Keys with RoPE removed, same shape
    """
    squeeze = False
    if keys.dim() == 3:
        keys = keys.unsqueeze(0)
        squeeze = True

    b, h, s, d = keys.shape
    d_half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    positions = torch.arange(s, dtype=torch.float32) + seq_offset
    freqs = torch.outer(positions, inv_freq)  # (s, d_half)
    cos_f = freqs.cos().to(keys.device, keys.dtype)
    sin_f = freqs.sin().to(keys.device, keys.dtype)

    first_half = keys[..., :d_half]
    second_half = keys[..., d_half:]
    result = keys.clone()
    result[..., :d_half] = first_half * cos_f + second_half * sin_f
    result[..., d_half:] = -first_half * sin_f + second_half * cos_f

    if squeeze:
        result = result.squeeze(0)
    return result


def forward_rope(keys: torch.Tensor, seq_offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    """Re-apply RoPE to key vectors (split-half style).

    Implements the forward rotation for split-half RoPE layout where dim i
    is paired with dim i+d/2. This is the layout used by Llama, Mistral,
    Qwen, Phi, and Gemma. Do NOT use with interleaved-RoPE models (GPT-NeoX,
    GPT-J) -- see module docstring for details.

    Args:
        keys: (heads, seq, dim) or (batch, heads, seq, dim) tensor
        seq_offset: Starting position index
        base: RoPE frequency base (rope_theta from model config)

    Returns:
        Keys with RoPE re-applied, same shape
    """
    squeeze = False
    if keys.dim() == 3:
        keys = keys.unsqueeze(0)
        squeeze = True

    b, h, s, d = keys.shape
    d_half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    positions = torch.arange(s, dtype=torch.float32) + seq_offset
    freqs = torch.outer(positions, inv_freq)
    cos_f = freqs.cos().to(keys.device, keys.dtype)
    sin_f = freqs.sin().to(keys.device, keys.dtype)

    first_half = keys[..., :d_half].clone()
    second_half = keys[..., d_half:].clone()
    result = keys.clone()
    result[..., :d_half] = first_half * cos_f - second_half * sin_f
    result[..., d_half:] = first_half * sin_f + second_half * cos_f

    if squeeze:
        result = result.squeeze(0)
    return result

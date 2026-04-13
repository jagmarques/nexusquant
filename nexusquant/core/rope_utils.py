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
supports linear and dynamic rope_scaling via the rope_scaling dict param
({"type": "linear"/"dynamic", "factor": float}). Both types divide frequencies
by the scale factor — this is exact for linear scaling and an approximation for
dynamic/NTK-aware scaling (full NTK base adjustment is a TODO). At standard
context lengths (<= original training length), rope_scaling has no effect.
"""

import torch


def _apply_rope_scaling(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    """Scale RoPE frequencies according to rope_scaling config dict.

    Args:
        inv_freq: Base inverse frequencies tensor (d_half,)
        rope_scaling: Dict with "type"/"rope_type" and "factor". Supported types:
            "linear"  -- exact: divide frequencies by factor.
            "dynamic" -- approximation: divide by factor (same as linear).
            "llama3"  -- piecewise: high-freq unchanged, low-freq scaled,
                         mid-freq smoothly interpolated. Used by Llama 3.1+.

    Returns:
        Scaled inv_freq tensor, same shape.
    """
    import math
    scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type", "linear"))
    factor = float(rope_scaling.get("factor", 1.0))
    if factor == 1.0:
        return inv_freq
    if scaling_type in ("linear", "dynamic"):
        return inv_freq / factor
    if scaling_type == "llama3":
        low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
        old_context_len = int(rope_scaling.get("original_max_position_embeddings", 8192))
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2.0 * math.pi / freq.item()
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1.0 - smooth) * freq / factor + smooth * freq)
        return torch.stack(new_freqs)
    return inv_freq


def inverse_rope(
    keys: torch.Tensor,
    seq_offset: int = 0,
    base: float = 10000.0,
    rope_scaling: dict = None,
) -> torch.Tensor:
    """Remove RoPE from key vectors (split-half style).

    Implements the inverse rotation for split-half RoPE layout where dim i
    is paired with dim i+d/2. This is the layout used by Llama, Mistral,
    Qwen, Phi, and Gemma. Do NOT use with interleaved-RoPE models (GPT-NeoX,
    GPT-J) -- see module docstring for details.

    Args:
        keys: (heads, seq, dim) or (batch, heads, seq, dim) tensor
        seq_offset: Starting position index
        base: RoPE frequency base (rope_theta from model config)
        rope_scaling: Optional dict from model config, e.g.
            {"type": "linear", "factor": 8.0} or {"type": "dynamic", "factor": 4.0}.
            None means no scaling.

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
    if rope_scaling is not None:
        inv_freq = _apply_rope_scaling(inv_freq, rope_scaling)
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


def inverse_rope_at_positions(
    keys: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    rope_scaling: dict = None,
) -> torch.Tensor:
    """Remove RoPE from keys encoded at arbitrary (non-contiguous) positions.

    Used when keys were encoded at positions like [0, 3, 7, 15] after eviction.
    Unlike inverse_rope(), this accepts an explicit per-token position tensor.

    Args:
        keys: (heads, seq, dim) float tensor with RoPE already applied
        positions: (seq,) long or float tensor of the original position for each token
        base: RoPE frequency base
        rope_scaling: Optional dict from model config, e.g.
            {"type": "linear", "factor": 8.0}. None means no scaling.

    Returns:
        Keys with RoPE removed at those positions, same shape (heads, seq, dim)
    """
    h, s, d = keys.shape
    d_half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=keys.device) / d))
    if rope_scaling is not None:
        inv_freq = _apply_rope_scaling(inv_freq, rope_scaling)
    positions_f = positions.float().to(keys.device)
    freqs = torch.outer(positions_f, inv_freq)     # (s, d_half)
    cos_f = freqs.cos().to(keys.dtype)
    sin_f = freqs.sin().to(keys.dtype)

    first_half = keys[..., :d_half]
    second_half = keys[..., d_half:]
    result = keys.clone()
    # inverse rotation: R^{-T}x = [x1*cos + x2*sin, -x1*sin + x2*cos]
    result[..., :d_half] = first_half * cos_f + second_half * sin_f
    result[..., d_half:] = -first_half * sin_f + second_half * cos_f
    return result


def forward_rope_at_positions(
    keys: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    rope_scaling: dict = None,
) -> torch.Tensor:
    """Apply RoPE to keys at arbitrary (non-contiguous) positions.

    Used to re-encode keys at contiguous positions [0, 1, 2, ...] after truncation.

    Args:
        keys: (heads, seq, dim) float tensor with RoPE removed
        positions: (seq,) long or float tensor of the target position for each token
        base: RoPE frequency base
        rope_scaling: Optional dict from model config, e.g.
            {"type": "linear", "factor": 8.0}. None means no scaling.

    Returns:
        Keys with RoPE applied at those positions, same shape (heads, seq, dim)
    """
    h, s, d = keys.shape
    d_half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=keys.device) / d))
    if rope_scaling is not None:
        inv_freq = _apply_rope_scaling(inv_freq, rope_scaling)
    positions_f = positions.float().to(keys.device)
    freqs = torch.outer(positions_f, inv_freq)     # (s, d_half)
    cos_f = freqs.cos().to(keys.dtype)
    sin_f = freqs.sin().to(keys.dtype)

    first_half = keys[..., :d_half].clone()
    second_half = keys[..., d_half:].clone()
    result = keys.clone()
    result[..., :d_half] = first_half * cos_f - second_half * sin_f
    result[..., d_half:] = first_half * sin_f + second_half * cos_f
    return result


def forward_rope(
    keys: torch.Tensor,
    seq_offset: int = 0,
    base: float = 10000.0,
    rope_scaling: dict = None,
) -> torch.Tensor:
    """Re-apply RoPE to key vectors (split-half style).

    Implements the forward rotation for split-half RoPE layout where dim i
    is paired with dim i+d/2. This is the layout used by Llama, Mistral,
    Qwen, Phi, and Gemma. Do NOT use with interleaved-RoPE models (GPT-NeoX,
    GPT-J) -- see module docstring for details.

    Args:
        keys: (heads, seq, dim) or (batch, heads, seq, dim) tensor
        seq_offset: Starting position index
        base: RoPE frequency base (rope_theta from model config)
        rope_scaling: Optional dict from model config, e.g.
            {"type": "linear", "factor": 8.0}. None means no scaling.

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
    if rope_scaling is not None:
        inv_freq = _apply_rope_scaling(inv_freq, rope_scaling)
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

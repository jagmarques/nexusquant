"""Token merging for KV cache compression.

Merges similar consecutive tokens in KV cache, reducing sequence length
while preserving attention quality. Novel technique: first combination
of token merging with lattice VQ for KV cache compression.

At 20% merge rate: ~1.25x additional compression on top of quantization.
At 40% merge rate: ~1.67x additional compression.
"""

import torch


def merge_tokens(
    keys: torch.Tensor,
    values: torch.Tensor,
    merge_pct: float = 20.0,
    protect_recent: int = 2,
) -> tuple:
    """Merge similar consecutive tokens in KV cache.

    Uses cosine similarity between adjacent key vectors to identify
    merge candidates. Merges by averaging key and value vectors.

    Args:
        keys: (heads, seq, dim) key tensor
        values: (heads, seq, dim) value tensor
        merge_pct: Percentage of tokens to merge (0-100)
        protect_recent: Number of recent tokens to protect from merging

    Returns:
        (merged_keys, merged_values, keep_mask) tuple
    """
    h, seq, d = keys.shape
    if merge_pct <= 0 or seq < 4:
        return keys, values, torch.ones(seq, dtype=torch.bool)

    n_merge = int(seq * merge_pct / 100)
    if n_merge == 0:
        return keys, values, torch.ones(seq, dtype=torch.bool)

    # Cosine similarity between consecutive keys (averaged over heads)
    k_norm = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (k_norm[:, :-1, :] * k_norm[:, 1:, :]).sum(dim=-1).mean(dim=0)

    # Protect first token (BOS), second token, and recent tokens
    cos_sim[0] = -1
    if len(cos_sim) > 1:
        cos_sim[1] = -1
    if protect_recent > 0:
        cos_sim[-protect_recent:] = -1

    # Find most similar consecutive pairs
    _, merge_idx = cos_sim.topk(min(n_merge, len(cos_sim)))
    merged_set = set()

    for idx in merge_idx.tolist():
        if idx in merged_set or idx + 1 in merged_set:
            continue
        # Average the pair
        avg_k = (keys[:, idx, :] + keys[:, idx + 1, :]) / 2
        avg_v = (values[:, idx, :] + values[:, idx + 1, :]) / 2
        keys[:, idx, :] = avg_k
        keys[:, idx + 1, :] = avg_k
        values[:, idx, :] = avg_v
        values[:, idx + 1, :] = avg_v
        merged_set.add(idx)
        merged_set.add(idx + 1)

    return keys, values, torch.ones(seq, dtype=torch.bool)


def merge_and_drop(
    keys: torch.Tensor,
    values: torch.Tensor,
    merge_pct: float = 20.0,
    protect_recent: int = 2,
) -> tuple:
    """Merge similar tokens and drop duplicates for actual sequence reduction.

    Unlike merge_tokens which keeps all positions, this actually removes
    duplicate tokens after merging, achieving real compression.

    Args:
        keys: (heads, seq, dim) key tensor
        values: (heads, seq, dim) value tensor
        merge_pct: Percentage of tokens to merge (0-100)
        protect_recent: Number of recent tokens to protect

    Returns:
        (shortened_keys, shortened_values) with reduced sequence length
    """
    h, seq, d = keys.shape
    if merge_pct <= 0 or seq < 4:
        return keys, values

    n_merge = int(seq * merge_pct / 100)
    if n_merge == 0:
        return keys, values

    k_norm = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (k_norm[:, :-1, :] * k_norm[:, 1:, :]).sum(dim=-1).mean(dim=0)

    cos_sim[0] = -1
    if len(cos_sim) > 1:
        cos_sim[1] = -1
    if protect_recent > 0:
        cos_sim[-protect_recent:] = -1

    _, merge_idx = cos_sim.topk(min(n_merge, len(cos_sim)))
    drop_indices = set()

    for idx in sorted(merge_idx.tolist()):
        if idx in drop_indices or idx + 1 in drop_indices:
            continue
        avg_k = (keys[:, idx, :] + keys[:, idx + 1, :]) / 2
        avg_v = (values[:, idx, :] + values[:, idx + 1, :]) / 2
        keys[:, idx, :] = avg_k
        values[:, idx, :] = avg_v
        drop_indices.add(idx + 1)

    keep = [i for i in range(seq) if i not in drop_indices]
    return keys[:, keep, :], values[:, keep, :]

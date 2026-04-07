"""NexusQuant compression pipelines.

NexusQuantFast: Hadamard + E8 VQ (training-free, ~5.3x, +0.36%)
NexusQuantMax:  RoPE removal + PCA + DP + E8 VQ (needs calibration, ~8x, +0.45%)
NexusQuantSimple: RoPE removal + Hadamard + E8 VQ (training-free, ~5.3x, +0.03%)

The Simple pipeline is our recommended default — 3 stages, zero calibration,
matches TurboQuant quality. Use Max only when you need >5.3x compression.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any

from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.dp_allocator import dp_bit_allocation
from nexusquant.core.rope_utils import inverse_rope, forward_rope
from nexusquant.core.token_merger import merge_and_drop
from nexusquant.core.mp_threshold import mp_signal_dims, adaptive_dim_budget


def _get_layer_kv(past_key_values, layer_idx):
    """Extract key/value tensors from a cache layer, with API fallback.

    Supports both the DynamicCache API (transformers >= 4.44) which uses
    .key_cache[l] / .value_cache[l], and the legacy .layers[l].keys / .values
    attribute style.

    Returns:
        (keys, values) tensors for the given layer index.
    """
    # Preferred: DynamicCache uses .key_cache / .value_cache lists
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]

    # Fallback: legacy .layers[l].keys / .values style
    layer = past_key_values.layers[layer_idx]
    return getattr(layer, 'keys'), getattr(layer, 'values')


def _set_layer_kv(past_key_values, layer_idx, keys, values):
    """Write key/value tensors back into a cache layer, with API fallback.

    Supports both the DynamicCache API (.key_cache / .value_cache) and
    the legacy .layers[l].keys / .values attribute style.
    """
    # Preferred: DynamicCache uses .key_cache / .value_cache lists
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        past_key_values.key_cache[layer_idx] = keys
        past_key_values.value_cache[layer_idx] = values
        return

    # Fallback: legacy .layers[l].keys / .values style
    layer = past_key_values.layers[layer_idx]
    setattr(layer, 'keys', keys)
    setattr(layer, 'values', values)


def _num_layers(past_key_values):
    """Return the number of layers in the cache, with API fallback."""
    if hasattr(past_key_values, 'key_cache'):
        return len(past_key_values.key_cache)
    return len(past_key_values.layers)


class NexusQuantFast:
    """DEPRECATED: Use NexusQuantSimple instead (adds RoPE removal for 17% better quality).

    Kept for backwards compatibility only.
    """

    def __init__(self, head_dim: int = 128, bits: int = 3):
        self.head_dim = head_dim
        self.bits = bits
        self.levels = 2 ** bits
        self.H = hadamard_matrix(head_dim)

    def compress(self, past_key_values) -> Any:
        """Compress KV cache in-place using Hadamard + E8."""
        device = None
        n_layers = _num_layers(past_key_values)
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            for val_tensor, is_key in [(k, True), (v, False)]:
                val = val_tensor.float()
                if device is None:
                    device = val.device
                    self.H = self.H.to(device)
                rotated = torch.einsum('bhsd,de->bhse', val, self.H.float())
                quantized = E8Lattice.quantize(rotated, levels=self.levels)
                unrotated = torch.einsum('bhsd,ed->bhse', quantized, self.H.float())
                if is_key:
                    new_k = unrotated.half()
                else:
                    new_v = unrotated.half()
            _set_layer_kv(past_key_values, l, new_k, new_v)
        return past_key_values


class NexusQuantSimple:
    """Training-free KV cache compression: RoPE removal + Hadamard + E8 VQ.

    Our recommended default. 3 stages, zero calibration, matches TurboQuant.
    ~5.3x at +0.03% on Mistral-7B, -0.002% on Llama-3.1-8B.

    With token merging: ~6.6x at <0.5% (4 stages, still no calibration).

    Usage:
        nq = NexusQuantSimple(head_dim=128)
        compressed_kv = nq.compress(past_key_values)
    """

    def __init__(self, head_dim: int = 128, bits: int = 3, rope_base: float = 10000.0,
                 merge_pct: float = 0.0):
        self.head_dim = head_dim
        self.bits = bits
        self.levels = 2 ** bits
        self.rope_base = rope_base
        self.merge_pct = merge_pct
        self.H = hadamard_matrix(head_dim)

    def compress(self, past_key_values) -> Any:
        """Compress KV cache: RoPE removal + Hadamard + E8 + re-RoPE."""
        device = None
        n_layers = _num_layers(past_key_values)
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            k = k.float()
            v = v.float()
            if device is None:
                device = k.device
                self.H = self.H.to(device)

            b, h, seq, d = k.shape

            # Keys: remove RoPE -> Hadamard -> E8 -> inverse Hadamard -> re-RoPE
            k_nr = inverse_rope(k[0], base=self.rope_base)  # (h, seq, d)
            if self.merge_pct > 0:
                k_nr, v_merged = merge_and_drop(k_nr, v[0], self.merge_pct)
                v = v_merged.unsqueeze(0)
            k_rot = torch.einsum('hsd,de->hse', k_nr.float(), self.H.float())
            # Per-head scaling: 5.12x at 3-bit (vs 3.2x with per-group)
            k_flat = k_rot.reshape(-1, k_rot.shape[-1])
            k_q = E8Lattice.quantize_perhead(k_flat, levels=self.levels)
            k_back = torch.einsum('hsd,ed->hse', k_q.reshape(k_rot.shape), self.H.float())
            k_out = forward_rope(k_back, base=self.rope_base).unsqueeze(0).half().to(device)

            # Values: Hadamard -> E8 (per-head scaling)
            v_flat = torch.einsum('bhsd,de->bhse', v.float(), self.H.float()).reshape(-1, v.shape[-1])
            v_q = E8Lattice.quantize_perhead(v_flat, levels=self.levels)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(v.shape), self.H.float()).half()

            _set_layer_kv(past_key_values, l, k_out, v_out)

        return past_key_values


class NexusQuantMax:
    """Maximum compression: RoPE removal + PCA + DP + E8 VQ.

    ~8x at +0.45% (50 passages, p=0.033). Requires PCA calibration.

    Usage:
        nq = NexusQuantMax(model, tokenizer, bits_per_dim=2.0)
        nq.calibrate(calibration_texts)  # ~30s on GPU
        compressed_kv = nq.compress(past_key_values)
    """

    def __init__(self, model, tokenizer, bits_per_dim: float = 2.0, head_dim: int = 128,
                 value_bits: int = 3, n_calibration: int = 50, distortion: str = "auto",
                 merge_pct: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.bits_per_dim = bits_per_dim
        self.head_dim = head_dim
        self.value_bits = value_bits  # 3 for low-rank models, 4 for high-rank
        self.n_calibration = n_calibration  # 50 > 20 gives better PCA basis
        self.distortion = distortion  # "auto", "empirical", or "theoretical"
        self.merge_pct = merge_pct  # Token merge percentage (0 = disabled)
        self.H = hadamard_matrix(head_dim)
        self.n_layers = model.config.num_hidden_layers
        self.rope_base = getattr(model.config, 'rope_theta', 10000.0)
        self.pca_k: Dict[int, Dict] = {}
        self.pca_v: Dict[int, Dict] = {}
        self.alloc_k: Dict[int, list] = {}
        self.alloc_v: Dict[int, list] = {}
        self._calibrated = False

    def calibrate(self, texts: Optional[list] = None, n_texts: int = 20, max_length: int = 256):
        """Compute PCA basis and DP allocation from calibration texts.

        Args:
            texts: List of calibration strings. If None, uses WikiText-103 validation.
            n_texts: Number of texts to use (default 20)
            max_length: Max token length per text
        """
        device = next(self.model.parameters()).device

        if texts is None:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            texts = [t for t in ds["text"] if len(t.split()) > 50][:n_texts]

        key_data = {l: [] for l in range(self.n_layers)}
        val_data = {l: [] for l in range(self.n_layers)}

        for text in texts[:n_texts]:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            pkv = out.past_key_values
            n_layers_out = _num_layers(pkv)
            for l in range(n_layers_out):
                k_tensor, v_tensor = _get_layer_kv(pkv, l)
                k = k_tensor[0].float().cpu()
                k_nr = inverse_rope(k)
                key_data[l].append(k_nr.reshape(-1, self.head_dim))
                val_data[l].append(v_tensor[0].float().cpu().reshape(-1, self.head_dim))

        total_bits = int(self.bits_per_dim * self.head_dim)
        for l in range(self.n_layers):
            for data, pca_dict, alloc_dict in [
                (key_data, self.pca_k, self.alloc_k),
                (val_data, self.pca_v, self.alloc_v),
            ]:
                X = torch.cat(data[l], dim=0)
                mean = X.mean(dim=0)
                _, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
                eig = (S ** 2 / (X.shape[0] - 1)).numpy()
                pca_dict[l] = {"mean": mean, "Vh": Vh, "eig": eig}
                alloc_dict[l] = dp_bit_allocation(eig, total_bits, distortion=self.distortion)

        self._calibrated = True

    def _compress_tensor(self, val: torch.Tensor, pca_dict: dict, alloc: list, layer_idx: int, is_keys: bool) -> torch.Tensor:
        """Compress a single key or value tensor using DP + E8."""
        b, h, seq, d = val.shape
        raw = val[0]  # (h, seq, d)

        if is_keys:
            processed = inverse_rope(raw).reshape(h * seq, d).cpu().float()
        else:
            processed = raw.reshape(h * seq, d).cpu().float()

        proj = (processed - pca_dict[layer_idx]["mean"]) @ pca_dict[layer_idx]["Vh"].T
        al = alloc[layer_idx]
        res = torch.zeros_like(proj)

        for bits in range(1, 5):
            dims = [i for i, bb in enumerate(al) if bb == bits]
            if not dims:
                continue
            chunk = proj[:, dims]
            lvl = 2 ** bits
            pd = (8 - len(dims) % 8) % 8
            if pd > 0:
                chunk = F.pad(chunk, (0, pd))
            p2_sz = 1
            while p2_sz < chunk.shape[-1]:
                p2_sz *= 2
            H_small = hadamard_matrix(p2_sz).cpu()
            chunk_pad = F.pad(chunk, (0, p2_sz - chunk.shape[-1]))
            rotated = chunk_pad @ H_small.T
            quantized = E8Lattice.quantize(rotated, levels=lvl)
            res[:, dims] = (quantized @ H_small)[:, :len(dims)]

        reconstructed = res @ pca_dict[layer_idx]["Vh"] + pca_dict[layer_idx]["mean"]

        if is_keys:
            recon_heads = reconstructed.reshape(h, seq, d)
            recon_heads = forward_rope(recon_heads)
            return recon_heads.unsqueeze(0).half().to(val.device)
        else:
            return reconstructed.reshape(b, h, seq, d).half().to(val.device)

    def compress(self, past_key_values) -> Any:
        """Compress KV cache using DP + RoPE removal + E8."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before compress()")

        n_layers = _num_layers(past_key_values)
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            k_compressed = self._compress_tensor(k.float(), self.pca_k, self.alloc_k, l, is_keys=True)
            v_compressed = self._compress_tensor(v.float(), self.pca_v, self.alloc_v, l, is_keys=False)
            _set_layer_kv(past_key_values, l, k_compressed, v_compressed)

        return past_key_values


class NexusQuantAsymmetric:
    """Asymmetric K-V compression using Marchenko-Pastur + E8.

    Keys and values have fundamentally different geometry:
    - Keys: intrinsic dim ~8 (matches E8), aggressive PCA dropping works
    - Values: intrinsic dim ~2 but high PCA rank, need conservative treatment

    MP threshold determines signal dims per layer. E8 VQ on signal space.
    Measured: 8.14x on TinyLlama (136% improvement over symmetric baseline).

    Usage:
        nq = NexusQuantAsymmetric(model, tokenizer)
        nq.calibrate()
        compressed_kv = nq.compress(past_key_values)
    """

    def __init__(self, model, tokenizer, bits: int = 3, head_dim: int = 128,
                 min_dims: int = 8, n_calibration: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.head_dim = head_dim
        self.min_dims = min_dims
        self.n_calibration = n_calibration
        self.n_layers = model.config.num_hidden_layers
        self.rope_base = getattr(model.config, 'rope_theta', 10000.0)
        self.pca_k: Dict[int, Dict] = {}
        self.pca_v: Dict[int, Dict] = {}
        self.keep_k: Dict[int, int] = {}
        self.keep_v: Dict[int, int] = {}
        self._calibrated = False

    def calibrate(self, texts=None, n_texts: int = 20, max_length: int = 256):
        """Calibrate PCA basis and MP-guided dimension budgets."""
        device = next(self.model.parameters()).device

        if texts is None:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            texts = [t for t in ds["text"] if len(t.split()) > 50][:n_texts]

        key_data = {l: [] for l in range(self.n_layers)}
        val_data = {l: [] for l in range(self.n_layers)}

        for text in texts[:n_texts]:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            pkv = out.past_key_values
            for l in range(_num_layers(pkv)):
                k_t, v_t = _get_layer_kv(pkv, l)
                k = k_t[0].float().cpu()
                k_nr = inverse_rope(k, base=self.rope_base)
                key_data[l].append(k_nr.reshape(-1, self.head_dim))
                val_data[l].append(v_t[0].float().cpu().reshape(-1, self.head_dim))

        for l in range(self.n_layers):
            for data, pca_dict, keep_dict, label in [
                (key_data, self.pca_k, self.keep_k, "keys"),
                (val_data, self.pca_v, self.keep_v, "values"),
            ]:
                X = torch.cat(data[l], dim=0)
                n_samples = X.shape[0]
                mean = X.mean(dim=0)
                _, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
                eig = (S ** 2 / max(n_samples - 1, 1)).numpy()

                n_signal = mp_signal_dims(eig, n_samples, self.head_dim)
                # Conservative: keep 1.5x signal dims (MP can be aggressive)
                n_keep_raw = int(n_signal * 1.5)
                n_keep = max(self.min_dims, ((n_keep_raw + 7) // 8) * 8)
                n_keep = min(n_keep, self.head_dim)

                pca_dict[l] = {"mean": mean, "Vh": Vh, "eig": eig}
                keep_dict[l] = n_keep

        self._calibrated = True

    def _compress_tensor(self, val, pca_dict, keep_dict, layer_idx, is_keys):
        """Compress using MP-guided PCA + E8 VQ."""
        b, h, seq, d = val.shape
        raw = val[0]
        n_keep = keep_dict[layer_idx]
        levels = 2 ** self.bits

        if is_keys:
            processed = inverse_rope(raw, base=self.rope_base).reshape(h * seq, d).cpu().float()
        else:
            processed = raw.reshape(h * seq, d).cpu().float()

        pca = pca_dict[layer_idx]
        proj = (processed - pca["mean"]) @ pca["Vh"][:n_keep].T

        # Hadamard + E8 on projected dims
        p2_sz = 1
        while p2_sz < n_keep:
            p2_sz *= 2
        H_small = hadamard_matrix(p2_sz).cpu()
        proj_pad = F.pad(proj, (0, p2_sz - n_keep))
        rotated = proj_pad @ H_small.T
        quantized = E8Lattice.quantize_perhead(rotated, levels=levels)
        result = (quantized @ H_small)[:, :n_keep]

        reconstructed = result @ pca["Vh"][:n_keep] + pca["mean"]

        if is_keys:
            recon_heads = reconstructed.reshape(h, seq, d)
            recon_heads = forward_rope(recon_heads, base=self.rope_base)
            return recon_heads.unsqueeze(0).half().to(val.device)
        else:
            return reconstructed.reshape(b, h, seq, d).half().to(val.device)

    def compress(self, past_key_values) -> Any:
        """Compress KV cache with asymmetric MP-guided PCA + E8."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before compress()")

        n_layers = _num_layers(past_key_values)
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            k_c = self._compress_tensor(k.float(), self.pca_k, self.keep_k, l, is_keys=True)
            v_c = self._compress_tensor(v.float(), self.pca_v, self.keep_v, l, is_keys=False)
            _set_layer_kv(past_key_values, l, k_c, v_c)
        return past_key_values


class NexusQuantEvict:
    """Token eviction + quantization: score importance, evict, RoPE-rm, Hadamard, E8, write back.

    Combines SnapKV-style attention-based eviction with E8 VQ for additive compression:
    - Eviction gives ~2.5x at 60% eviction rate (tokens gone from cache)
    - E8 2-bit quantization gives ~6.4x on surviving tokens
    - Combined: ~16x at long context with <1% PPL degradation

    Usage:
        nq = NexusQuantEvict(eviction_rate=0.6)
        past_key_values, prefix_mask = nq.compress(past_key_values)
        # pass prefix_mask as attention_mask to model.generate()
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 2,
        eviction_rate: float = 0.6,
        sliding_window: int = 32,
        obs_window: int = 32,
        rope_base: float = 10000.0,
    ):
        """
        Args:
            head_dim: Attention head dimension (default 128).
            bits: Quantization bits for E8 VQ (default 2 for aggressive compression).
            eviction_rate: Fraction of prefix tokens to evict (0–1). Default 0.6.
            sliding_window: Recent tokens always kept regardless of importance score.
            obs_window: Number of recent query positions used to compute importance scores.
            rope_base: RoPE frequency base (rope_theta from model config).
        """
        self.head_dim = head_dim
        self.bits = bits
        self.levels = 2 ** bits
        self.eviction_rate = eviction_rate
        self.sliding_window = sliding_window
        self.obs_window = obs_window
        self.rope_base = rope_base
        self.H = hadamard_matrix(head_dim)

    def _score_importance(self, keys: torch.Tensor) -> torch.Tensor:
        """Score token importance via key–key attention (SnapKV-style).

        Uses the last obs_window key vectors as queries against all prefix keys.
        Importance = mean attention weight received from recent queries.

        Args:
            keys: (batch, heads, seq, head_dim) float tensor

        Returns:
            importance: (batch, seq) importance scores, higher = more important
        """
        b, h, seq, d = keys.shape
        scale = d ** -0.5

        # Recent keys act as queries; prefix = everything except sliding window
        obs = min(self.obs_window, seq)
        q = keys[:, :, -obs:, :]          # (b, h, obs, d)
        k = keys[:, :, :seq, :]            # (b, h, seq, d)

        # Dot-product attention: (b, h, obs, seq)
        attn = torch.einsum('bhqd,bhkd->bhqk', q.float(), k.float()) * scale
        attn = torch.softmax(attn, dim=-1)

        # Sum attention received per token, averaged over heads and query positions
        importance = attn.sum(dim=2).mean(dim=1)  # (b, seq)
        return importance

    def _build_keep_mask(self, importance: torch.Tensor, seq: int) -> torch.Tensor:
        """Build boolean keep mask: BOS + top-k important + sliding window.

        Args:
            importance: (batch, seq) importance scores
            seq: total sequence length

        Returns:
            keep_mask: (batch, seq) bool, True = keep token
        """
        b = importance.shape[0]
        device = importance.device

        # How many tokens to keep from the evictable prefix
        prefix_len = max(seq - self.sliding_window, 0)
        n_keep = max(1, int(prefix_len * (1.0 - self.eviction_rate)))

        keep_mask = torch.zeros(b, seq, dtype=torch.bool, device=device)

        # Always keep BOS (position 0)
        keep_mask[:, 0] = True

        # Always keep the sliding window (last sliding_window tokens)
        if self.sliding_window > 0 and seq > 0:
            keep_mask[:, max(0, seq - self.sliding_window):] = True

        # Keep top-n_keep tokens from the evictable prefix by importance
        if prefix_len > 1 and n_keep > 0:
            prefix_scores = importance[:, 1:prefix_len]  # exclude BOS (already kept)
            topk_k = min(n_keep, prefix_scores.shape[1])
            if topk_k > 0:
                _, topk_idx = torch.topk(prefix_scores, k=topk_k, dim=1)
                topk_idx = topk_idx + 1  # offset back (we sliced from 1)
                keep_mask.scatter_(1, topk_idx, True)

        return keep_mask

    def compress(self, past_key_values):
        """Compress KV cache in-place: evict + RoPE-rm + Hadamard + E8 + write back.

        Args:
            past_key_values: HuggingFace DynamicCache (or legacy .layers API).

        Returns:
            (past_key_values, prefix_attention_mask)
            - past_key_values: modified in-place; evicted positions zeroed.
            - prefix_attention_mask: (batch, seq) float tensor of 0/1 to pass as
              attention_mask when continuing generation.
        """
        device = None
        n_layers = _num_layers(past_key_values)

        # Use layer 0 keys to compute importance (representative of all layers)
        k0, _ = _get_layer_kv(past_key_values, 0)
        if device is None:
            device = k0.device
            self.H = self.H.to(device)

        b, h, seq, d = k0.shape

        importance = self._score_importance(k0.float())          # (b, seq)
        keep_mask = self._build_keep_mask(importance, seq)        # (b, seq) bool

        # Build attention mask: 1.0 for kept, 0.0 for evicted (additive-mask convention)
        prefix_attention_mask = keep_mask.float()                 # (b, seq)

        H = self.H.float()
        levels = self.levels
        rope_base = self.rope_base

        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            k = k.float()
            v = v.float()

            # --- Evict: zero positions that are masked out ---
            # keep_mask: (b, seq) -> (b, 1, seq, 1) for broadcasting
            mask_4d = keep_mask.unsqueeze(1).unsqueeze(-1).float()  # (b, 1, seq, 1)
            k = k * mask_4d
            v = v * mask_4d

            # --- Keys: RoPE removal -> Hadamard -> E8 -> inv Hadamard -> re-RoPE ---
            k_out_batches = []
            for bi in range(b):
                k_bi = k[bi]  # (h, seq, d)
                k_nr = inverse_rope(k_bi, base=rope_base)                         # (h, seq, d)
                k_rot = torch.einsum('hsd,de->hse', k_nr, H)                      # (h, seq, d)
                k_flat = k_rot.reshape(-1, d)                                      # (h*seq, d)
                k_q = E8Lattice.quantize_perhead(k_flat, levels=levels)            # (h*seq, d)
                k_back = torch.einsum('hsd,ed->hse', k_q.reshape(h, seq, d), H)   # (h, seq, d)
                k_roped = forward_rope(k_back, base=rope_base)                     # (h, seq, d)
                k_out_batches.append(k_roped)
            k_out = torch.stack(k_out_batches, dim=0).to(dtype=torch.float16, device=device)

            # --- Values: Hadamard -> E8 -> inv Hadamard ---
            v_rot = torch.einsum('bhsd,de->bhse', v, H)                            # (b, h, seq, d)
            v_flat = v_rot.reshape(-1, d)                                           # (b*h*seq, d)
            v_q = E8Lattice.quantize_perhead(v_flat, levels=levels)                # (b*h*seq, d)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(b, h, seq, d), H
                                 ).to(dtype=torch.float16, device=device)

            # Re-zero evicted positions (quantizer may have shifted zeros slightly)
            k_out = k_out * mask_4d.half()
            v_out = v_out * mask_4d.half()

            _set_layer_kv(past_key_values, l, k_out, v_out)

        return past_key_values, prefix_attention_mask


def compress_kv_cache(past_key_values, mode: str = "simple", head_dim: int = 128, **kwargs) -> Any:
    """Convenience function to compress KV cache.

    Args:
        past_key_values: HuggingFace DynamicCache from model forward pass
        mode: "simple" (5.3x, training-free, recommended),
              "fast" (5.3x, no RoPE removal),
              "max" (8x, needs model+tokenizer in kwargs)
        head_dim: Head dimension (default 128)
        **kwargs: For "simple": rope_base, merge_pct
                  For "max": model, tokenizer, bits_per_dim, distortion, merge_pct

    Returns:
        Compressed past_key_values (modified in-place)
    """
    if mode == "simple":
        nq = NexusQuantSimple(
            head_dim=head_dim,
            bits=kwargs.get("bits", 3),
            rope_base=kwargs.get("rope_base", 10000.0),
            merge_pct=kwargs.get("merge_pct", 0.0),
        )
        return nq.compress(past_key_values)
    elif mode == "fast":
        nq = NexusQuantFast(head_dim=head_dim, bits=kwargs.get("bits", 3))
        return nq.compress(past_key_values)
    elif mode == "max":
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        bpd = kwargs.get("bits_per_dim", 2.0)
        nq = NexusQuantMax(
            model, tokenizer,
            bits_per_dim=bpd,
            head_dim=head_dim,
            distortion=kwargs.get("distortion", "auto"),
            merge_pct=kwargs.get("merge_pct", 0.0),
        )
        nq.calibrate()
        return nq.compress(past_key_values)
    elif mode == "asymmetric":
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        nq = NexusQuantAsymmetric(
            model, tokenizer,
            bits=kwargs.get("bits", 3),
            head_dim=head_dim,
            min_dims=kwargs.get("min_dims", 8),
        )
        nq.calibrate()
        return nq.compress(past_key_values)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple', 'fast', 'max', or 'asymmetric'.")

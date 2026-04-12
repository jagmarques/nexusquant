"""NexusQuant compression pipelines.

NexusQuantFast: Hadamard + E8 VQ (training-free, ~5.3x, +0.36%)
NexusQuantMax:  RoPE removal + PCA + DP + E8 VQ (needs calibration, ~8x, +0.45%)
NexusQuantSimple: RoPE removal + Hadamard + E8 VQ (training-free, ~5.3x, +0.03%)

The Simple pipeline is our recommended default — 3 stages, zero calibration,
matches TurboQuant quality. Use Max only when you need >5.3x compression.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Union

from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.dp_allocator import dp_bit_allocation
from nexusquant.core.rope_utils import inverse_rope, forward_rope, inverse_rope_at_positions, forward_rope_at_positions
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
            # Loop over batch elements to support batch_size > 1
            k_out_list = []
            v_list = [v[bi] for bi in range(b)]  # split values per batch for merge_pct
            for bi in range(b):
                k_nr = inverse_rope(k[bi], base=self.rope_base)  # (h, seq, d)
                if self.merge_pct > 0:
                    k_nr, v_merged = merge_and_drop(k_nr, v_list[bi], self.merge_pct)
                    v_list[bi] = v_merged
                k_rot = torch.einsum('hsd,de->hse', k_nr.float(), self.H.float())
                # Per-head scaling: 5.12x at 3-bit (vs 3.2x with per-group)
                k_flat = k_rot.reshape(-1, k_rot.shape[-1])
                k_q = E8Lattice.quantize_perhead(k_flat, levels=self.levels)
                k_back = torch.einsum('hsd,ed->hse', k_q.reshape(k_rot.shape), self.H.float())
                k_roped = forward_rope(k_back, base=self.rope_base)  # (h, seq, d)
                k_out_list.append(k_roped)
            k_out = torch.stack(k_out_list, dim=0).half().to(device)

            # Rebuild value tensor after possible merge_pct modification
            v = torch.stack(v_list, dim=0)  # (b, h, seq', d)

            # Values: Hadamard -> E8 (per-head scaling)
            v_flat = torch.einsum('bhsd,de->bhse', v.float(), self.H.float()).reshape(-1, v.shape[-1])
            v_q = E8Lattice.quantize_perhead(v_flat, levels=self.levels)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(v.shape), self.H.float()).half()

            _set_layer_kv(past_key_values, l, k_out, v_out)

        return past_key_values


class NexusQuantQuantOnly:
    """Quantization-only KV cache compression: RoPE removal + Hadamard + E8 VQ. No eviction.

    Lossless quality tier. ~5x compression, preserves NIAH factual recall.
    GPU-validated: -0.001% PPL on Gemma-2-2b, NIAH recall = YES.

    Use this when quality matters more than compression ratio.
    For higher compression (16-32x) at the cost of factual recall, use NexusQuantEvict.

    Usage:
        nq = NexusQuantQuantOnly(rope_base=10000.0)
        compressed_kv = nq.compress(past_key_values)
    """

    def __init__(self, bits: int = 3, rope_base: float = 10000.0,
                 compress_layers: str = "all"):
        self.bits = bits
        self.levels = 2 ** bits
        self.rope_base = rope_base
        self.compress_layers = compress_layers
        self._hadamard_cache = {}
        self._swa_layer_indices = None

    def set_model_config(self, model_config):
        """Detect SWA layers for hybrid models (Gemma 4, GLM4).

        When compress_layers="global_only", SWA layers are kept at f16.
        Mirrors NexusQuantEvict.set_model_config().
        """
        if self.compress_layers != "global_only":
            return
        # Gemma 4 nests text config inside text_config; unwrap if present
        cfg = getattr(model_config, 'text_config', model_config)
        swa_layers = set()
        n_layers = getattr(cfg, 'num_hidden_layers', 32)
        pattern = getattr(cfg, 'sliding_window_pattern', None)
        if pattern is not None:
            for i in range(n_layers):
                if (i + 1) % pattern != 0:
                    swa_layers.add(i)
        layer_types = getattr(cfg, 'layer_types', None)
        if layer_types is not None and not swa_layers:
            for i, lt in enumerate(layer_types):
                if lt in ('sliding_window', 'local', 'swa', 'sliding_attention'):
                    swa_layers.add(i)
        n_shared = getattr(cfg, 'num_kv_shared_layers', 0)
        if n_shared > 0 and not swa_layers:
            for i in range(n_layers - n_shared, n_layers):
                swa_layers.add(i)
        self._swa_layer_indices = swa_layers

    def _get_hadamard(self, size: int, device) -> torch.Tensor:
        if size not in self._hadamard_cache:
            self._hadamard_cache[size] = hadamard_matrix(size).to(device)
        H = self._hadamard_cache[size]
        if H.device != device:
            H = H.to(device)
            self._hadamard_cache[size] = H
        return H

    def compress(self, past_key_values) -> Any:
        """Compress KV cache in-place: RoPE removal + Hadamard + E8 + re-RoPE. No eviction."""
        n_layers = _num_layers(past_key_values)
        skip_layers = self._swa_layer_indices or set()
        for l in range(n_layers):
            if l in skip_layers:
                continue
            k, v = _get_layer_kv(past_key_values, l)
            k = k.float()
            v = v.float()
            b, h, seq, d = k.shape
            device = k.device
            H = self._get_hadamard(d, device).float()

            # Keys: RoPE removal -> Hadamard -> E8 per-head -> inv Hadamard -> re-RoPE
            k_out_list = []
            for bi in range(b):
                k_nr = inverse_rope(k[bi], base=self.rope_base)
                k_rot = torch.einsum('hsd,de->hse', k_nr, H)
                k_flat = k_rot.reshape(-1, d)
                k_q = E8Lattice.quantize_perhead(k_flat, levels=self.levels)
                k_back = torch.einsum('hsd,ed->hse', k_q.reshape(h, seq, d), H)
                k_out_list.append(forward_rope(k_back, base=self.rope_base))
            k_out = torch.stack(k_out_list, dim=0).half().to(device)

            # Values: Hadamard -> E8 per-head -> inv Hadamard
            v_rot = torch.einsum('bhsd,de->bhse', v, H)
            v_flat = v_rot.reshape(-1, d)
            v_q = E8Lattice.quantize_perhead(v_flat, levels=self.levels)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(b, h, seq, d), H).half()

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

        target_dtype = val.dtype
        if is_keys:
            recon_heads = reconstructed.reshape(h, seq, d)
            recon_heads = forward_rope(recon_heads)
            return recon_heads.unsqueeze(0).to(dtype=target_dtype, device=val.device)
        else:
            return reconstructed.reshape(b, h, seq, d).to(dtype=target_dtype, device=val.device)

    def compress(self, past_key_values) -> Any:
        """Compress KV cache using DP + RoPE removal + E8."""
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before compress()")

        n_layers = _num_layers(past_key_values)
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            orig_dtype = k.dtype
            k_compressed = self._compress_tensor(k.float(), self.pca_k, self.alloc_k, l, is_keys=True)
            v_compressed = self._compress_tensor(v.float(), self.pca_v, self.alloc_v, l, is_keys=False)
            k_compressed = k_compressed.to(dtype=orig_dtype)
            v_compressed = v_compressed.to(dtype=orig_dtype)
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
        key_bits: int = None,
        value_bits: int = None,
        eviction_rate: Union[float, str] = 0.6,
        sliding_window: int = 32,
        obs_window: int = 32,
        rope_base: float = 10000.0,
        scorer: str = "key-key",
        protected_layers: set = None,
        protect_boundary: Union[int, str] = 0,
        min_context_for_compression: int = 0,
        soft_eviction: bool = False,
        adaptive_context: bool = False,
        protected_positions: Optional[torch.Tensor] = None,
        compress_layers: str = "all",
        layer_bit_profile: str = "uniform",
        distance_graduated: bool = False,
    ):
        """
        Args:
            head_dim: Attention head dimension (default 128).
            bits: Quantization bits for E8 VQ (default 2). Applied to both K and V
                unless key_bits/value_bits override.
            key_bits: Quantization bits for keys. If None, uses bits.
            value_bits: Quantization bits for values. If None, uses bits.
            eviction_rate: Fraction of prefix tokens to evict (0–1). Default 0.6.
                Pass "auto" to measure attention entropy and choose the rate
                adaptively: high-entropy (creative/uniform) text → lower eviction;
                low-entropy (peaked/structured) text → higher eviction.
            sliding_window: Recent tokens always kept regardless of importance score.
            obs_window: Number of recent query positions used to compute importance scores.
            rope_base: RoPE frequency base (rope_theta from model config).
            scorer: Importance scoring method.
                "key-key" (default) -- SnapKV-style proxy: recent keys as queries
                    against all prefix keys. Fast, no extra forward pass.
                "real" -- Accumulated softmax weights from a dedicated prefill
                    forward pass with output_attentions=True. Requires the model to
                    be loaded with attn_implementation='eager' (SDPA suppresses
                    attention weights silently). Needs model + input_ids passed to
                    compress(). Expected ~0.35pp improvement at 80% eviction.
            protected_layers: Set of layer indices to skip quantization on (kept at
                FP16 after eviction). Default None (no protected layers).
            protect_boundary: Auto-protect first and last N layers. E.g. protect_boundary=2
                protects layers 0, 1 and the last 2. Default 0 (disabled).
                Pass "auto" to probe layer key magnitudes: if first/last layers are
                >2x brighter than the middle layer, 2-layer boundary protection is
                enabled automatically (result is cached after the first call).
            min_context_for_compression: Skip compression entirely when seq_len is below
                this threshold. 0 = always compress (default).
            soft_eviction: When True, low-importance tokens are quantized at 1-bit
                (levels=2, sign-only after Hadamard) instead of being zeroed. This
                preserves directional information in evicted tokens at the cost of
                slightly higher memory vs. hard eviction. The returned attention mask
                is all-ones (no tokens are removed; all positions remain visible).
                Default False (hard eviction, current behaviour).
            adaptive_context: When True, scale eviction rate down for short prefixes.
                Default False.
                  seq < 256  → evict_rate = 0.0
                  seq < 512  → evict_rate *= 0.5
                  seq < 1024 → evict_rate *= 0.75
                  seq >= 1024 → full evict_rate
            protected_positions: Optional 1-D tensor of token position indices that
                must never be evicted (e.g. image token spans in VLMs). Default None.
                Example: torch.tensor([1, 2, 3, 256, 257]) to protect 5 positions.
            compress_layers: Which layers to compress. Default "all".
                "all" -- compress every layer (standard behavior).
                "global_only" -- only compress global/full-attention layers, skip
                    sliding-window layers. For hybrid models like Gemma4 (50 SWA +
                    10 global), this means only the 10 global layers get compressed.
                    SWA layers have fixed memory cost regardless of context length,
                    so compressing them wastes quality for minimal savings.
                    Requires model config to be passed via set_model_config().
        """
        if scorer not in ("key-key", "real"):
            raise ValueError(f"scorer must be 'key-key' or 'real', got {scorer!r}")
        if not (isinstance(eviction_rate, (int, float)) or eviction_rate == "auto"):
            raise ValueError(
                f"eviction_rate must be a float in [0, 1] or 'auto', got {eviction_rate!r}"
            )
        if not (isinstance(protect_boundary, int) or protect_boundary == "auto"):
            raise ValueError(
                f"protect_boundary must be a non-negative int or 'auto', got {protect_boundary!r}"
            )
        self.head_dim = head_dim
        self.bits = bits
        self.key_bits = key_bits if key_bits is not None else bits
        self.value_bits = value_bits if value_bits is not None else bits
        self.key_levels = 2 ** self.key_bits
        self.value_levels = 2 ** self.value_bits
        self.levels = 2 ** bits  # kept for backward compat
        self.eviction_rate = eviction_rate  # float or "auto"
        self.sliding_window = sliding_window
        self.obs_window = obs_window
        self.rope_base = rope_base
        self.scorer = scorer
        self.protected_layers = set(protected_layers) if protected_layers else set()
        self.protect_boundary = protect_boundary  # int or "auto"
        self.min_context_for_compression = min_context_for_compression
        self.soft_eviction = soft_eviction
        self.adaptive_context = adaptive_context
        self.protected_positions = protected_positions  # Optional[torch.Tensor] of position indices
        self.compress_layers = compress_layers  # "all" or "global_only"
        self.layer_bit_profile = layer_bit_profile  # "uniform" or "graduated"
        self.distance_graduated = distance_graduated  # boost recent token importance
        self._swa_layer_indices = None  # cached set of SWA layer indices
        self._hadamard_cache = {}  # size -> Hadamard matrix on device

    def set_model_config(self, model_config):
        """Set model config for attention-type-aware compression.

        For hybrid models (Gemma4, GLM4), detects which layers use sliding-window
        attention vs global attention. When compress_layers="global_only", only
        global layers get compressed.

        Args:
            model_config: HuggingFace model config object.
        """
        if self.compress_layers != "global_only":
            return

        # Gemma 4 nests text config inside text_config; unwrap if present
        cfg = getattr(model_config, 'text_config', model_config)

        # Detect SWA layers from config
        swa_layers = set()
        n_layers = getattr(cfg, 'num_hidden_layers', 32)

        # Method 1: sliding_window_pattern (Gemma4 style)
        pattern = getattr(cfg, 'sliding_window_pattern', None)
        if pattern is not None:
            for i in range(n_layers):
                if (i + 1) % pattern != 0:
                    swa_layers.add(i)

        # Method 2: layer_types list (Gemma 4 E2B uses this)
        layer_types = getattr(cfg, 'layer_types', None)
        if layer_types is not None and not swa_layers:
            for i, lt in enumerate(layer_types):
                if lt in ('sliding_window', 'local', 'swa', 'sliding_attention'):
                    swa_layers.add(i)

        # Method 3: num_kv_shared_layers (Gemma4 E-series)
        n_shared = getattr(cfg, 'num_kv_shared_layers', 0)
        if n_shared > 0 and not swa_layers:
            for i in range(n_layers - n_shared, n_layers):
                swa_layers.add(i)

        self._swa_layer_indices = swa_layers

    def _get_hadamard(self, size: int, device) -> torch.Tensor:
        """Get or create a Hadamard matrix for the given dimension.

        Caches matrices by size so heterogeneous-head_dim models (Gemma 4:
        global_head_dim=512, head_dim=256) don't rebuild every layer.
        """
        if size not in self._hadamard_cache:
            self._hadamard_cache[size] = hadamard_matrix(size).to(device)
        H = self._hadamard_cache[size]
        if H.device != device:
            H = H.to(device)
            self._hadamard_cache[size] = H
        return H

    def _find_scoring_layer(self, past_key_values, n_layers: int) -> int:
        """Find the best layer for importance scoring.

        For hybrid models (compress_layers="global_only"), returns the first
        global attention layer — its keys span the full sequence. Layer 0 in
        Gemma 4 is SWA and only covers the last ~512 tokens, which would
        produce garbage importance scores for the full context.
        """
        if self.compress_layers == "global_only" and self._swa_layer_indices:
            for l in range(n_layers):
                if l not in self._swa_layer_indices:
                    return l
        return 0

    def _resolve_layer_levels(self, layer_idx: int, n_layers: int):
        """Get key/value quantization levels for a specific layer.

        With layer_bit_profile="graduated", boundary layers (first/last 15%)
        get 3-bit for both K and V, while middle layers use the configured
        key_levels/value_levels (typically K3V2). This is inspired by the
        APEX weight quant approach where layer-wise precision beats uniform.

        Returns (key_levels, value_levels) for this layer.
        """
        if self.layer_bit_profile == "graduated":
            boundary_count = max(1, int(n_layers * 0.15))
            if layer_idx < boundary_count or layer_idx >= n_layers - boundary_count:
                return 8, 8  # 3-bit K, 3-bit V for boundary layers
        return self.key_levels, self.value_levels

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

    def _score_importance_real(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Score token importance via real attention weights during prefill.

        Runs a dedicated forward pass with output_attentions=True and collects
        the full softmax attention weight matrices from every layer and head.
        For each token position j, importance[j] is the total attention it
        received from all later positions (i > j), summed across layers and heads
        and then averaged.

        REQUIRES: model loaded with attn_implementation='eager'.
        PyTorch SDPA (the default) suppresses attention weights silently —
        output_attentions=True returns None when SDPA is active.

        Args:
            model: HuggingFace causal LM loaded with attn_implementation='eager'.
            input_ids: (batch, seq) token ids for the prefix.

        Returns:
            importance: (batch, seq) float tensor, higher = more important.
        """
        device = input_ids.device
        b, seq = input_ids.shape

        with torch.no_grad():
            out = model(
                input_ids,
                use_cache=False,
                output_attentions=True,
            )

        attentions = out.attentions  # tuple of (b, heads, seq, seq) per layer
        if attentions is None or len(attentions) == 0 or attentions[0] is None:
            raise RuntimeError(
                "output_attentions=True returned None. "
                "Load the model with attn_implementation='eager' to get real attention weights. "
                "SDPA (the default) suppresses attention weight output regardless of this flag."
            )

        # Accumulate: for each token j, how much attention it received from
        # query positions i >= j (causal column sums).
        # The causal mask already zeroes the strictly-upper triangle, so
        # summing all query positions gives the correct causal column sum.
        importance = torch.zeros(b, seq, dtype=torch.float32, device=device)
        n_layers = len(attentions)

        for layer_attn in attentions:
            # layer_attn: (b, heads, seq_q, seq_k)
            layer_attn = layer_attn.float()           # ensure fp32
            col_sum = layer_attn.sum(dim=2)            # (b, heads, seq_k)
            importance += col_sum.mean(dim=1)          # (b, seq_k), avg over heads

        importance = importance / n_layers             # avg over layers
        return importance

    def _compute_adaptive_eviction_rate(self, importance: torch.Tensor, seq: int) -> float:
        """Compute eviction rate based on attention entropy.

        High entropy (uniform attention, creative text) → lower eviction (more conservative).
        Low entropy (peaked attention, structured text) → higher eviction (safe to remove more).

        Args:
            importance: (batch, seq) importance scores from _score_importance
            seq: total sequence length

        Returns:
            evict_rate: float in [0.1, 0.7]
        """
        # Normalize importance to a probability distribution
        probs = importance[0] / (importance[0].sum() + 1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = math.log(max(seq, 2))  # guard: log(1)=0 would cause div-by-zero
        normalized_entropy = entropy / max_entropy  # 0 = peaked, 1 = uniform

        # Map entropy to eviction rate:
        # Low entropy (0.3) → high eviction (0.55)
        # High entropy (0.9) → low eviction (0.25)
        rate = 0.7 - 0.5 * normalized_entropy  # linear mapping
        return max(0.1, min(0.7, rate))  # clamp to [0.1, 0.7]

    def _auto_detect_boundary(self, past_key_values) -> int:
        """Auto-detect if boundary protection is needed.

        Compares key magnitudes of the first and last layers against the middle
        layer.  If first/last are >2x brighter than the middle layer the model
        has strong boundary activations and 2-layer protection is enabled.

        The result is cached on self after the first call so subsequent
        compress() calls pay no probing cost.

        Returns:
            n_protect: 0 (no protection) or 2 (protect first+last 2 layers)
        """
        if hasattr(self, '_auto_boundary_cached'):
            return self._auto_boundary_cached

        n_layers = _num_layers(past_key_values)
        if n_layers < 6:
            self._auto_boundary_cached = 0
            return 0

        k_first, _ = _get_layer_kv(past_key_values, 0)
        k_last, _ = _get_layer_kv(past_key_values, n_layers - 1)
        k_mid, _ = _get_layer_kv(past_key_values, n_layers // 2)

        mag_boundary = (k_first.float().abs().mean() + k_last.float().abs().mean()) / 2
        mag_middle = k_mid.float().abs().mean()

        ratio = (mag_boundary / (mag_middle + 1e-8)).item()
        self._auto_boundary_cached = 2 if ratio > 2.0 else 0
        return self._auto_boundary_cached

    def _build_keep_mask(self, importance: torch.Tensor, seq: int,
                         eviction_rate: float = None) -> torch.Tensor:
        """Build boolean keep mask: BOS + top-k important + sliding window.

        Args:
            importance: (batch, seq) importance scores
            seq: total sequence length
            eviction_rate: fraction to evict; falls back to self.eviction_rate when
                None (used internally when the rate has already been resolved).

        Returns:
            keep_mask: (batch, seq) bool, True = keep token
        """
        if eviction_rate is None:
            eviction_rate = self.eviction_rate  # backward-compat default
        if not isinstance(eviction_rate, (int, float)):
            raise ValueError(
                f"_build_keep_mask requires a resolved float eviction_rate, got {eviction_rate!r}. "
                "If self.eviction_rate == 'auto', resolve it via _compute_adaptive_eviction_rate() first."
            )
        b = importance.shape[0]
        device = importance.device

        # How many tokens to keep from the evictable prefix
        prefix_len = max(seq - self.sliding_window, 0)
        n_keep = max(1, int(prefix_len * (1.0 - eviction_rate)))

        keep_mask = torch.zeros(b, seq, dtype=torch.bool, device=device)

        # Always keep BOS (position 0)
        keep_mask[:, 0] = True

        # Always keep the sliding window (last sliding_window tokens)
        if self.sliding_window > 0 and seq > 0:
            keep_mask[:, max(0, seq - self.sliding_window):] = True

        # Distance-graduated eviction: boost importance of recent tokens so they
        # survive eviction more easily. Distant tokens need higher raw importance
        # to be kept. This is like DCA's idea of treating recent vs distant context
        # differently, but applied through the importance scores.
        if getattr(self, 'distance_graduated', False) and prefix_len > 1:
            # Linear ramp: position 1 gets 0.0 boost, last prefix token gets 1.0 boost
            ramp = torch.linspace(0.0, 1.0, prefix_len - 1, device=device)
            # Scale boost to ~25% of max importance so it biases but doesn't dominate
            boost = ramp * importance[:, 1:prefix_len].max() * 0.25
            importance = importance.clone()
            importance[:, 1:prefix_len] = importance[:, 1:prefix_len] + boost

        # Keep top-n_keep tokens from the evictable prefix by importance
        if prefix_len > 1 and n_keep > 0:
            prefix_scores = importance[:, 1:prefix_len]  # exclude BOS (already kept)
            topk_k = min(n_keep, prefix_scores.shape[1])
            if topk_k > 0:
                _, topk_idx = torch.topk(prefix_scores, k=topk_k, dim=1)
                topk_idx = topk_idx + 1  # offset back (we sliced from 1)
                keep_mask.scatter_(1, topk_idx, True)

        # Force-keep VLM protected positions (e.g. image token spans)
        if self.protected_positions is not None:
            for pos in self.protected_positions:
                idx = int(pos)
                if 0 <= idx < seq:
                    keep_mask[:, idx] = True

        return keep_mask

    def compress(self, past_key_values, model=None, input_ids=None):
        """Compress KV cache in-place: evict + RoPE-rm + Hadamard + E8 + write back.

        Args:
            past_key_values: HuggingFace DynamicCache (or legacy .layers API).
            model: Required when self.scorer == "real". HuggingFace causal LM
                loaded with attn_implementation='eager'.
            input_ids: Required when self.scorer == "real". (batch, seq) token ids
                for the prefix that produced past_key_values.

        Returns:
            (past_key_values, prefix_attention_mask)
            - past_key_values: modified in-place; evicted positions zeroed.
            - prefix_attention_mask: (batch, seq) float tensor of 0/1 to pass as
              attention_mask when continuing generation.
        """
        n_layers = _num_layers(past_key_values)

        # Find the right layer for scoring: for hybrid models, use a global
        # attention layer (full sequence) instead of layer 0 (may be SWA).
        score_layer = self._find_scoring_layer(past_key_values, n_layers)
        k_score, _ = _get_layer_kv(past_key_values, score_layer)
        device = k_score.device
        b, _, seq, _ = k_score.shape

        # --- Score importance ---
        if self.scorer == "real":
            if model is None or input_ids is None:
                raise ValueError(
                    "scorer='real' requires model and input_ids to be passed to compress(). "
                    "Also ensure the model was loaded with attn_implementation='eager'."
                )
            importance = self._score_importance_real(model, input_ids)  # (b, seq)
        else:
            importance = self._score_importance(k_score.float())        # (b, seq)

        # --- Feature 1: Resolve eviction rate (fixed float or "auto") ---
        if self.eviction_rate == "auto":
            evict_rate = self._compute_adaptive_eviction_rate(importance, seq)
        else:
            evict_rate = float(self.eviction_rate)

        # --- Feature 2: Graduated context-aware scaling ---
        if self.adaptive_context:
            if seq < 256:
                evict_rate = 0.0
            elif seq < 512:
                evict_rate *= 0.5
            elif seq < 1024:
                evict_rate *= 0.75
            # else: full evict_rate

        keep_mask = self._build_keep_mask(importance, seq, eviction_rate=evict_rate)  # (b, seq) bool

        # Build attention mask: 1.0 for kept, 0.0 for evicted (additive-mask convention)
        prefix_attention_mask = keep_mask.float()                 # (b, seq)

        rope_base = self.rope_base

        # --- Feature 3: Resolve boundary protection (fixed int or "auto") ---
        if self.protect_boundary == "auto":
            n_protect = self._auto_detect_boundary(past_key_values)
        else:
            n_protect = int(self.protect_boundary)

        # Resolve protected layers (boundary + explicit + SWA layers for hybrid models)
        skip_layers = set(self.protected_layers)
        if n_protect > 0:
            for i in range(min(n_protect, n_layers)):
                skip_layers.add(i)
                skip_layers.add(n_layers - 1 - i)
        if self.compress_layers == "global_only" and self._swa_layer_indices:
            skip_layers.update(self._swa_layer_indices)

        # Deferred compression: skip entirely when context is too short
        if self.min_context_for_compression > 0 and seq < self.min_context_for_compression:
            return past_key_values, torch.ones(b, seq, device=device)

        if self.soft_eviction:
            # Soft eviction: kept tokens at full precision (key_levels/value_levels),
            # evicted tokens at 1-bit (levels=2, sign-only after Hadamard rotation).
            # No tokens removed — all positions remain visible; mask is all-ones.
            ones_mask = torch.ones(b, seq, dtype=torch.float32, device=device)

            for l in range(n_layers):
                k, v = _get_layer_kv(past_key_values, l)
                k = k.float()
                v = v.float()

                # Protected layers: keep at FP16, no quantization
                if l in skip_layers:
                    _set_layer_kv(past_key_values, l, k.half(), v.half())
                    continue

                # Per-layer shape: h and d may differ across layers (Gemma 4)
                _, h_l, seq_l, d_l = k.shape
                H = self._get_hadamard(d_l, device).float()

                # Resolve per-layer levels
                lkl, lvl = self._resolve_layer_levels(l, n_layers)

                # Build per-layer mask (seq_l may differ from scoring seq for SWA)
                mask_4d_l = keep_mask[:, :seq_l].unsqueeze(1).unsqueeze(-1).float()

                # --- Keys: RoPE removal -> Hadamard -> dual-precision E8 -> inv Hadamard -> re-RoPE ---
                k_out_batches = []
                for bi in range(b):
                    k_bi = k[bi]                                                                  # (h_l, seq_l, d_l)
                    k_nr = inverse_rope(k_bi, base=rope_base)                                     # (h_l, seq_l, d_l)
                    k_rot = torch.einsum('hsd,de->hse', k_nr, H)                                 # (h_l, seq_l, d_l)
                    k_flat = k_rot.reshape(-1, d_l)                                               # (h_l*seq_l, d_l)
                    k_q_full = E8Lattice.quantize_perhead(k_flat, levels=lkl)
                    k_q_1bit = E8Lattice.quantize_perhead(k_flat, levels=2)
                    keep_flat = mask_4d_l[bi].expand(h_l, seq_l, 1).reshape(h_l * seq_l, 1)
                    k_q = k_q_full * keep_flat + k_q_1bit * (1.0 - keep_flat)
                    k_back = torch.einsum('hsd,ed->hse', k_q.reshape(h_l, seq_l, d_l), H)
                    k_roped = forward_rope(k_back, base=rope_base)
                    k_out_batches.append(k_roped)
                k_out = torch.stack(k_out_batches, dim=0).to(dtype=torch.float16, device=device)

                # --- Values: Hadamard -> dual-precision E8 -> inv Hadamard ---
                v_rot = torch.einsum('bhsd,de->bhse', v, H)
                v_flat = v_rot.reshape(-1, d_l)
                v_q_full = E8Lattice.quantize_perhead(v_flat, levels=lvl)
                v_q_1bit = E8Lattice.quantize_perhead(v_flat, levels=2)
                keep_v_flat = mask_4d_l.expand(b, h_l, seq_l, 1).reshape(b * h_l * seq_l, 1)
                v_q = v_q_full * keep_v_flat + v_q_1bit * (1.0 - keep_v_flat)
                v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(b, h_l, seq_l, d_l), H
                                     ).to(dtype=torch.float16, device=device)

                _set_layer_kv(past_key_values, l, k_out, v_out)

            return past_key_values, ones_mask

        # --- Hard eviction (default behaviour, soft_eviction=False) ---
        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)
            k = k.float()
            v = v.float()

            # Per-layer shape: h and d may differ across layers (Gemma 4)
            _, h_l, seq_l, d_l = k.shape

            # Build per-layer mask (seq_l may differ from scoring seq for SWA)
            mask_4d_l = keep_mask[:, :seq_l].unsqueeze(1).unsqueeze(-1).float()

            # --- Evict: zero positions that are masked out ---
            k = k * mask_4d_l
            v = v * mask_4d_l

            # Protected layers: keep FP16 after eviction, skip quantization
            if l in skip_layers:
                k_out = k.to(dtype=torch.float16, device=device)
                v_out = v.to(dtype=torch.float16, device=device)
                _set_layer_kv(past_key_values, l, k_out, v_out)
                continue

            H = self._get_hadamard(d_l, device).float()

            # Resolve per-layer quantization levels (graduated profile gives
            # boundary layers more bits than middle layers)
            lkl, lvl = self._resolve_layer_levels(l, n_layers)

            # --- Keys: RoPE removal -> Hadamard -> E8 -> inv Hadamard -> re-RoPE ---
            k_out_batches = []
            for bi in range(b):
                k_bi = k[bi]  # (h_l, seq_l, d_l)
                k_nr = inverse_rope(k_bi, base=rope_base)
                k_rot = torch.einsum('hsd,de->hse', k_nr, H)
                k_flat = k_rot.reshape(-1, d_l)
                k_q = E8Lattice.quantize_perhead(k_flat, levels=lkl)
                k_back = torch.einsum('hsd,ed->hse', k_q.reshape(h_l, seq_l, d_l), H)
                k_roped = forward_rope(k_back, base=rope_base)
                k_out_batches.append(k_roped)
            k_out = torch.stack(k_out_batches, dim=0).to(dtype=torch.float16, device=device)

            # --- Values: Hadamard -> E8 -> inv Hadamard ---
            v_rot = torch.einsum('bhsd,de->bhse', v, H)
            v_flat = v_rot.reshape(-1, d_l)
            v_q = E8Lattice.quantize_perhead(v_flat, levels=lvl)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(b, h_l, seq_l, d_l), H
                                 ).to(dtype=torch.float16, device=device)

            # Re-zero evicted positions (quantizer may have shifted zeros slightly)
            k_out = k_out * mask_4d_l.half()
            v_out = v_out * mask_4d_l.half()

            _set_layer_kv(past_key_values, l, k_out, v_out)

        return past_key_values, prefix_attention_mask


class NexusQuantEvictTruncate(NexusQuantEvict):
    """Physical KV tensor truncation with contiguous RoPE remapping.

    Extends NexusQuantEvict by REMOVING evicted tokens from tensors entirely,
    rather than zeroing them. This delivers real GPU memory savings and faster
    attention (fewer tokens in KV = fewer FLOPs).

    The critical correctness fix: after removing tokens, the surviving keys
    carry RoPE encodings for their original (non-contiguous) positions. We
    strip those encodings and re-apply RoPE at contiguous positions [0,1,...,n-1].
    Generation then continues from position n (= number of kept tokens).

    Returns:
        (past_key_values, new_position_ids_start)
        - past_key_values: cache with physically truncated tensors
        - new_position_ids_start: int, pass as position_ids start for generation

    Usage:
        nq = NexusQuantEvictTruncate(eviction_rate=0.6)
        past_key_values, next_pos = nq.compress(past_key_values)
        # For generation: position_ids = torch.arange(next_pos, next_pos + max_new_tokens)
    """

    def compress(self, past_key_values):
        """Physically truncate KV cache: evict + remap RoPE to contiguous positions.

        Args:
            past_key_values: HuggingFace DynamicCache (or legacy .layers API).

        Returns:
            (past_key_values, new_position_ids_start)
            - past_key_values: modified in-place; evicted tokens physically removed.
            - new_position_ids_start: int, first position for the next generated token.
        """
        n_layers = _num_layers(past_key_values)

        # Find the right layer for scoring (global layer for hybrid models)
        score_layer = self._find_scoring_layer(past_key_values, n_layers)
        k_score, _ = _get_layer_kv(past_key_values, score_layer)
        device = k_score.device
        b, _, seq, _ = k_score.shape

        # Physical truncation requires identical sequence lengths across batch elements.
        if b > 1:
            raise ValueError(
                f"NexusQuantEvictTruncate does not support batch_size > 1 (got {b}). "
                "Physical truncation requires identical sequence lengths across batch elements. "
                "Use NexusQuantEvict (masking, not truncation) for batched inference."
            )

        # scorer="real" not supported in truncation path (no model/input_ids arg here)
        importance = self._score_importance(k_score.float())   # (b, seq)

        # --- Feature 1: Resolve eviction rate (fixed float or "auto") ---
        if self.eviction_rate == "auto":
            evict_rate = self._compute_adaptive_eviction_rate(importance, seq)
        else:
            evict_rate = float(self.eviction_rate)

        # --- Feature 2: Graduated context-aware scaling ---
        if self.adaptive_context:
            if seq < 256:
                evict_rate = 0.0
            elif seq < 512:
                evict_rate *= 0.5
            elif seq < 1024:
                evict_rate *= 0.75
            # else: full evict_rate

        keep_mask = self._build_keep_mask(importance, seq, eviction_rate=evict_rate)  # (b, seq) bool

        # kept_indices: positions that survive, shape (n_kept,)
        kept_indices = keep_mask[0].nonzero(as_tuple=True)[0]  # (n_kept,)
        n_kept = kept_indices.shape[0]

        # Original positions for RoPE inverse (non-contiguous after eviction)
        orig_positions = kept_indices.float()                   # (n_kept,)
        # New contiguous positions [0, 1, ..., n_kept-1]
        new_positions = torch.arange(n_kept, dtype=torch.float32, device=device)

        rope_base = self.rope_base

        # --- Feature 3: Resolve boundary protection (fixed int or "auto") ---
        if self.protect_boundary == "auto":
            n_protect = self._auto_detect_boundary(past_key_values)
        else:
            n_protect = int(self.protect_boundary)

        # Resolve protected layers (boundary + explicit + SWA layers for hybrid models)
        skip_layers = set(self.protected_layers)
        if n_protect > 0:
            for i in range(min(n_protect, n_layers)):
                skip_layers.add(i)
                skip_layers.add(n_layers - 1 - i)
        if self.compress_layers == "global_only" and self._swa_layer_indices:
            skip_layers.update(self._swa_layer_indices)

        # Deferred compression: skip entirely when context is too short
        if self.min_context_for_compression > 0 and seq < self.min_context_for_compression:
            return past_key_values, seq

        for l in range(n_layers):
            k, v = _get_layer_kv(past_key_values, l)

            # Per-layer shape (h and d may differ across layers)
            _, h_l, seq_l, d_l = k.shape

            # Truncation indices clipped to this layer's seq length
            layer_kept = kept_indices[kept_indices < seq_l]
            n_kept_l = layer_kept.shape[0]

            # Physical truncation: gather only kept token positions
            k_kept = k[:, :, layer_kept, :].float()
            v_kept = v[:, :, layer_kept, :].float()

            # Protected layers: keep FP16 at truncated positions, skip quantization
            if l in skip_layers:
                _set_layer_kv(past_key_values, l,
                              k_kept.to(dtype=torch.float16, device=device),
                              v_kept.to(dtype=torch.float16, device=device))
                continue

            H = self._get_hadamard(d_l, device).float()

            # Resolve per-layer levels
            lkl, lvl = self._resolve_layer_levels(l, n_layers)

            # Per-layer RoPE positions (clipped to this layer's range)
            layer_orig_pos = layer_kept.float()
            layer_new_pos = torch.arange(n_kept_l, dtype=torch.float32, device=device)

            # --- Keys: strip RoPE at original positions, re-apply at contiguous positions ---
            k_out_batches = []
            for bi in range(b):
                k_bi = k_kept[bi]   # (h_l, n_kept_l, d_l)
                k_nr = inverse_rope_at_positions(k_bi, layer_orig_pos, base=rope_base)
                k_rot = torch.einsum('hsd,de->hse', k_nr, H)
                k_flat = k_rot.reshape(-1, d_l)
                k_q = E8Lattice.quantize_perhead(k_flat, levels=lkl)
                k_back = torch.einsum('hsd,ed->hse', k_q.reshape(h_l, n_kept_l, d_l), H)
                k_roped = forward_rope_at_positions(k_back, layer_new_pos, base=rope_base)
                k_out_batches.append(k_roped)
            k_out = torch.stack(k_out_batches, dim=0).to(dtype=torch.float16, device=device)

            # --- Values: Hadamard + E8 (no RoPE on values) ---
            v_rot = torch.einsum('bhsd,de->bhse', v_kept, H)
            v_flat = v_rot.reshape(-1, d_l)
            v_q = E8Lattice.quantize_perhead(v_flat, levels=lvl)
            v_out = torch.einsum('bhsd,ed->bhse', v_q.reshape(b, h_l, n_kept_l, d_l), H
                                 ).to(dtype=torch.float16, device=device)

            _set_layer_kv(past_key_values, l, k_out, v_out)

        # Update DynamicCache's seen-token counter to reflect the shorter sequence
        if hasattr(past_key_values, '_seen_tokens'):
            past_key_values._seen_tokens = n_kept

        # new_position_ids_start: generation continues from position n_kept
        return past_key_values, n_kept


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
    elif mode == "quant_only":
        nq = NexusQuantQuantOnly(
            bits=kwargs.get("bits", 3),
            rope_base=kwargs.get("rope_base", 10000.0),
        )
        return nq.compress(past_key_values)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple', 'fast', 'max', 'asymmetric', or 'quant_only'.")

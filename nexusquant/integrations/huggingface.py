"""Zero-change HuggingFace integration for NexusQuant KV cache compression.

One import, one line. Works with any model that uses DynamicCache
(Llama, Mistral, Qwen, Phi, Gemma, and more).

Usage:
    from nexusquant.integrations.huggingface import nexusquant

    with nexusquant(model):
        output = model.generate(input_ids, max_new_tokens=100)

    # Or with manual forward pass:
    with nexusquant(model, mode="simple", bits=3):
        out = model(input_ids, use_cache=True)

    # Maximum compression (requires calibration texts):
    with nexusquant(model, mode="max", tokenizer=tokenizer, bits=2):
        output = model.generate(input_ids, max_new_tokens=100)
"""

import sys
import time
import warnings
import functools
from contextlib import contextmanager
from typing import Optional, Any

import torch

from nexusquant.pipeline import NexusQuantSimple, NexusQuantMax, NexusQuantEvict, NexusQuantEvictTruncate
from nexusquant.core.hadamard import hadamard_matrix
from nexusquant.core.e8_lattice import E8Lattice
from nexusquant.core.rope_utils import inverse_rope, forward_rope


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------

class _CompressionStats:
    """Track compression statistics across all layers during inference."""

    # E8 lattice VQ with per-group FP16 scale overhead:
    # For 8D groups: each group stores `bits`-bit lattice code (bits * 8 = data bits)
    # plus a 16-bit FP16 scale factor, amortized over 8 elements = 2 extra bits/element.
    # Effective bits/element = data_bits + 2 (scale overhead).
    E8_SCALE_OVERHEAD_BPD = 2.0

    def __init__(self, bits: int):
        self.bits = bits
        self.effective_bits = bits + self.E8_SCALE_OVERHEAD_BPD
        self.reset()

    def reset(self):
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        self.layers_compressed = 0
        self.prefill_compressions = 0
        self.tokens_seen = 0
        self.start_time = time.time()

    def record(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Record compression of one layer's KV cache."""
        # Original size: float16 = 2 bytes per element
        n_elements = key_states.numel() + value_states.numel()
        self.total_original_bytes += n_elements * 2  # fp16

        # Compressed size: effective bits per element includes per-group scale overhead
        # For 3-bit E8: 3 (data) + 2 (FP16 scale / 8 elements) = 5 bits/element
        self.total_compressed_bytes += n_elements * self.effective_bits / 8

        self.layers_compressed += 1

    @property
    def compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def memory_saved_mb(self) -> float:
        return (self.total_original_bytes - self.total_compressed_bytes) / (1024 * 1024)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        ratio = self.compression_ratio
        saved = self.memory_saved_mb
        lines = [
            "",
            "--- NexusQuant Compression Stats ---",
            f"  Compression ratio:    {ratio:.1f}x",
            f"  Effective bits:       {self.effective_bits:.1f}-bit ({self.bits}-bit data + {self.E8_SCALE_OVERHEAD_BPD:.0f}-bit scale overhead)",
            f"  Memory saved:         {saved:.1f} MB",
            f"  Layers compressed:    {self.layers_compressed}",
            f"  Wall time:            {elapsed:.2f}s",
            "------------------------------------",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model introspection: detect config from any HuggingFace model
# ---------------------------------------------------------------------------

def _detect_model_config(model):
    """Extract head_dim, num_layers, rope_base, num_kv_heads from a HF model."""
    config = model.config

    # Handle nested text configs (e.g., VLMs with model.config.text_config)
    text_config = getattr(config, "text_config", config)

    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(text_config, "hidden_size", 4096)
        num_heads = getattr(text_config, "num_attention_heads", 32)
        head_dim = hidden_size // num_heads

    num_layers = getattr(text_config, "num_hidden_layers", 32)
    rope_base = getattr(text_config, "rope_theta", 10000.0)

    num_kv_heads = getattr(text_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(text_config, "num_attention_heads", 32)

    rope_scaling = getattr(text_config, "rope_scaling", None)

    return {
        "head_dim": head_dim,
        "num_layers": num_layers,
        "rope_base": rope_base,
        "num_kv_heads": num_kv_heads,
        "rope_scaling": rope_scaling,
    }


# ---------------------------------------------------------------------------
# Compression logic applied inside the hook
# ---------------------------------------------------------------------------

def _compress_kv_simple(keys, values, compressor):
    """Apply NexusQuantSimple compression to a single layer's KV tensors.

    Args:
        keys: (batch, heads, seq, head_dim) tensor
        values: (batch, heads, seq, head_dim) tensor
        compressor: NexusQuantSimple instance

    Returns:
        (compressed_keys, compressed_values)
    """
    b, h, seq, d = keys.shape

    H = compressor.H
    if H.device != keys.device:
        H = H.to(keys.device)
        compressor.H = H

    rope_base = compressor.rope_base
    levels = compressor.levels

    # --- Keys: remove RoPE -> Hadamard -> E8 -> inv Hadamard -> re-RoPE ---
    k_float = keys.float()
    k_out_list = []
    for bi in range(b):
        k_nr = inverse_rope(k_float[bi], base=rope_base)  # (h, seq, d)
        k_rot = torch.einsum('hsd,de->hse', k_nr, H.float())
        k_q = E8Lattice.quantize(k_rot, levels=levels)
        k_back = torch.einsum('hsd,ed->hse', k_q, H.float())
        k_roped = forward_rope(k_back, base=rope_base)  # (h, seq, d)
        k_out_list.append(k_roped)
    k_compressed = torch.stack(k_out_list, dim=0).to(dtype=keys.dtype, device=keys.device)

    # --- Values: Hadamard -> E8 -> inv Hadamard ---
    v_float = values.float()
    v_rot = torch.einsum('bhsd,de->bhse', v_float, H.float())
    v_q = E8Lattice.quantize(v_rot, levels=levels)
    v_compressed = torch.einsum('bhsd,ed->bhse', v_q, H.float()).to(
        dtype=values.dtype, device=values.device
    )

    return k_compressed, v_compressed


def _compress_kv_max(keys, values, compressor, layer_idx):
    """Apply NexusQuantMax compression to a single layer's KV tensors.

    Args:
        keys: (batch, heads, seq, head_dim) tensor
        values: (batch, heads, seq, head_dim) tensor
        compressor: NexusQuantMax instance
        layer_idx: Layer index for per-layer PCA/DP parameters

    Returns:
        (compressed_keys, compressed_values)
    """
    k_compressed = compressor._compress_tensor(
        keys.float(), compressor.pca_k, compressor.alloc_k, layer_idx, is_keys=True
    )
    v_compressed = compressor._compress_tensor(
        values.float(), compressor.pca_v, compressor.alloc_v, layer_idx, is_keys=False
    )
    return k_compressed, v_compressed


# ---------------------------------------------------------------------------
# Hook installer / uninstaller
# ---------------------------------------------------------------------------

def _install_hooks(model, compressor, stats, min_seq_for_compression, mode):
    """Install compression hooks by monkey-patching DynamicLayer.update.

    Strategy: monkey-patch the class-level update() on DynamicLayer and
    DynamicSlidingWindowLayer. After the standard concat-to-cache, we
    apply NexusQuant compression in-place. This works because ALL
    HuggingFace models call layer.update(k, v) inside Cache.update().

    Same pattern as kvpress -- intercept at the cache layer level,
    completely model-agnostic.
    """
    try:
        from transformers.cache_utils import (
            DynamicCache,
            DynamicLayer,
            DynamicSlidingWindowLayer,
        )
    except ImportError:
        raise ImportError(
            "transformers >= 4.44 required for DynamicCache support. "
            "Install with: pip install 'transformers>=4.44'"
        )

    # Detect offloaded cache support and warn if the model uses it.
    # OffloadedStaticCache has a different update interface and cannot be
    # transparently patched the same way as DynamicCache layers.
    try:
        from transformers.cache_utils import OffloadedStaticCache as _OffloadedStaticCache
        _has_offloaded = True
    except ImportError:
        _OffloadedStaticCache = None
        _has_offloaded = False

    if _has_offloaded and hasattr(model, "_offload_cache"):
        warnings.warn(
            "NexusQuant detected KV cache offloading (_offload_cache). "
            "Compression hooks intercept DynamicCache layer updates and may not "
            "fire for offloaded cache writes. Results may be incorrect or "
            "compression may silently not apply. Disable KV cache offloading "
            "when using NexusQuant.",
            stacklevel=3,
        )

    # Save originals for clean restoration
    originals = {
        "DynamicLayer_update": DynamicLayer.update,
        "DynamicSlidingWindowLayer_update": DynamicSlidingWindowLayer.update,
    }

    def _make_hooked_update(original_method):
        """Wrap a cache layer's update method with NexusQuant compression."""

        def hooked_update(self, key_states, value_states, cache_kwargs=None):
            # Run the original update (concatenates new KV to existing cache)
            keys, values = original_method(self, key_states, value_states, cache_kwargs)

            incoming_seq_len = key_states.shape[-2]

            # Only compress on prefill (long sequences), not single-token decode
            if incoming_seq_len < min_seq_for_compression:
                return keys, values

            if mode == "simple":
                k_compressed, v_compressed = _compress_kv_simple(
                    keys, values, compressor
                )
            elif mode == "max":
                layer_idx = getattr(self, '_nq_layer_idx', None)
                if layer_idx is None:
                    # Without layer index, we cannot apply per-layer PCA.
                    # This should not happen if Cache.update is also patched.
                    return keys, values
                k_compressed, v_compressed = _compress_kv_max(
                    keys, values, compressor, layer_idx
                )
            else:
                return keys, values

            # Write compressed tensors back into the cache layer
            self.keys = k_compressed
            self.values = v_compressed

            stats.record(k_compressed, v_compressed)

            return k_compressed, v_compressed

        return hooked_update

    # Patch class-level update methods
    DynamicLayer.update = _make_hooked_update(originals["DynamicLayer_update"])
    DynamicSlidingWindowLayer.update = _make_hooked_update(
        originals["DynamicSlidingWindowLayer_update"]
    )

    # For max mode: patch Cache.update to inject layer indices into each layer
    # DynamicLayer doesn't inherently know its own index, but Cache.update receives it
    if mode == "max":
        original_cache_update = DynamicCache.update
        originals["DynamicCache_update"] = original_cache_update

        def cache_update_with_idx(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Ensure layers exist up to layer_idx
            if self.layer_class_to_replicate is not None:
                while len(self.layers) <= layer_idx:
                    self.layers.append(self.layer_class_to_replicate())
            # Tag the layer with its index so the hooked update can use it
            if layer_idx < len(self.layers):
                self.layers[layer_idx]._nq_layer_idx = layer_idx
            # Delegate to the original (which calls layer.update, now hooked)
            return original_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)

        DynamicCache.update = cache_update_with_idx
        originals["_patched_cache_update"] = True

    return originals


def _uninstall_hooks(originals):
    """Restore original DynamicLayer/Cache methods."""
    try:
        from transformers.cache_utils import (
            DynamicCache,
            DynamicLayer,
            DynamicSlidingWindowLayer,
        )
    except ImportError:
        return

    DynamicLayer.update = originals["DynamicLayer_update"]
    DynamicSlidingWindowLayer.update = originals["DynamicSlidingWindowLayer_update"]

    if originals.get("_patched_cache_update"):
        DynamicCache.update = originals["DynamicCache_update"]


# ---------------------------------------------------------------------------
# Public API: the context manager
# ---------------------------------------------------------------------------

@contextmanager
def nexusquant(
    model,
    mode: str = "simple",
    bits: int = 3,
    merge_pct: float = 0.0,
    rope_base: Optional[float] = None,
    tokenizer=None,
    calibration_texts: Optional[list] = None,
    bits_per_dim: Optional[float] = None,
    distortion: str = "auto",
    min_seq_for_compression: int = 2,
    verbose: bool = True,
):
    """Context manager that enables NexusQuant KV cache compression on any HuggingFace model.

    Monkey-patches DynamicCache layer updates to intercept KV cache writes
    and apply E8 lattice vector quantization with RoPE-aware processing.
    Cleanly restores original behavior on exit.

    Works with model.generate(), model(), and any code path that uses
    HuggingFace's DynamicCache (Llama, Mistral, Qwen, Phi, Gemma, etc.).

    Args:
        model: Any HuggingFace causal LM (AutoModelForCausalLM).
        mode: Compression mode.
            "simple" -- RoPE removal + Hadamard + E8 VQ. Training-free, ~5.3x. (default)
            "max"    -- PCA + DP bit allocation + E8 VQ. Needs calibration, ~8x.
        bits: Quantization bits (2, 3, or 4). Default 3 for ~5.3x compression.
        merge_pct: Token merge percentage (0-100). 0 = disabled. 20 = ~1.25x extra.
        rope_base: RoPE frequency base. Auto-detected from model config if None.
        tokenizer: Required for mode="max" (used in calibration).
        calibration_texts: Custom calibration texts for mode="max". Uses WikiText if None.
        bits_per_dim: Bits per dimension for mode="max". Default 2.0.
        distortion: Distortion model for DP allocation ("auto"/"empirical"/"theoretical").
        min_seq_for_compression: Minimum incoming sequence length to trigger compression.
            Sequences shorter than this (single-token decode steps) skip compression
            to avoid overhead. Default 2.
        verbose: Print compression stats on exit. Default True.

    Yields:
        The model (same reference, but with compression hooks active).

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from nexusquant import nexusquant
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>>
        >>> inputs = tokenizer("The meaning of life is", return_tensors="pt").to(model.device)
        >>> with nexusquant(model):
        ...     output = model.generate(**inputs, max_new_tokens=100)
        >>> print(tokenizer.decode(output[0]))
    """
    # --- Validate args ---
    if mode not in ("simple", "max"):
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'simple' (training-free, ~5.3x) "
            f"or 'max' (calibrated, ~8x)."
        )

    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4 (got {bits})")

    if mode == "max" and tokenizer is None:
        raise ValueError(
            "mode='max' requires a tokenizer for calibration. "
            "Pass tokenizer=your_tokenizer."
        )

    # --- Detect model config ---
    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]
    num_layers = model_cfg["num_layers"]
    num_kv_heads = model_cfg["num_kv_heads"]

    if rope_base is None:
        rope_base = model_cfg["rope_base"]

    # --- Create compressor ---
    stats = _CompressionStats(bits=bits)

    if mode == "simple":
        compressor = NexusQuantSimple(
            head_dim=head_dim,
            bits=bits,
            rope_base=rope_base,
            merge_pct=merge_pct,
        )
    elif mode == "max":
        if bits_per_dim is None:
            bits_per_dim = float(bits) - 1.0  # bits=3 -> 2.0 bpd

        compressor = NexusQuantMax(
            model=model,
            tokenizer=tokenizer,
            bits_per_dim=bits_per_dim,
            head_dim=head_dim,
            value_bits=bits,
            distortion=distortion,
            merge_pct=merge_pct,
        )
        if verbose:
            print(
                f"NexusQuant: calibrating max mode "
                f"({num_layers} layers, {head_dim}d, {bits_per_dim:.1f} bpd)...",
                flush=True,
            )
        compressor.calibrate(texts=calibration_texts)
        if verbose:
            print("NexusQuant: calibration complete.", flush=True)

    # --- Install hooks ---
    if verbose:
        mode_desc = {
            "simple": f"simple ({bits}-bit E8 VQ, ~{16/bits:.1f}x)",
            "max": f"max ({bits_per_dim:.1f} bpd, PCA+DP+E8, ~{16/(bits_per_dim or 2.0):.1f}x)",
        }
        model_name = getattr(model.config, "_name_or_path", model.__class__.__name__)
        print(
            f"NexusQuant: activating {mode_desc[mode]} on {model_name}",
            flush=True,
        )

    originals = _install_hooks(
        model=model,
        compressor=compressor,
        stats=stats,
        min_seq_for_compression=min_seq_for_compression,
        mode=mode,
    )

    try:
        yield model
    finally:
        _uninstall_hooks(originals)

        if verbose:
            print(stats.summary(), flush=True)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def nexusquant_simple(model, bits: int = 3, **kwargs):
    """Shorthand: nexusquant(model, mode='simple', bits=bits, ...)."""
    return nexusquant(model, mode="simple", bits=bits, **kwargs)


def nexusquant_max(model, tokenizer, bits_per_dim: float = 2.0, **kwargs):
    """Shorthand: nexusquant(model, mode='max', tokenizer=tokenizer, ...)."""
    return nexusquant(
        model, mode="max", tokenizer=tokenizer, bits_per_dim=bits_per_dim, **kwargs
    )


# ---------------------------------------------------------------------------
# Eviction context manager
# ---------------------------------------------------------------------------

_EVICT_PRESETS = {
    "high":     {"eviction_rate": 0.35, "bits": 2, "key_bits": 3},         # K3V2, ~9x, <0.5% PPL (A100, 3544-tok)
    "balanced": {"eviction_rate": 0.60, "bits": 2},                        # K2V2, ~17x at long ctx, <1% PPL
    "max":      {"eviction_rate": 0.80, "bits": 2},                        # K2V2, ~33x, +2% PPL
    "asym":     {"eviction_rate": 0.60, "bits": 2, "key_bits": 3},         # K3V2, ~14x, <1% PPL (asymmetric: 3-bit K, 2-bit V)
}


@contextmanager
def nexusquant_evict(
    model,
    eviction_rate=0.6,
    quality: str = "balanced",
    sliding_window: int = 32,
    obs_window: int = 32,
    bits: int = 2,
    key_bits: int = None,
    value_bits: int = None,
    truncate: bool = False,
    scorer: str = "key-key",
    input_ids=None,
    protected_layers: set = None,
    protect_boundary=0,
    min_context_for_compression: int = 0,
    soft_eviction: bool = False,
    adaptive_context: bool = False,
    protected_positions: Optional[torch.Tensor] = None,
    compress_layers: str = "all",
    verbose: bool = True,
):
    """Compress KV cache with attention-aware token eviction + E8 quantization.

    Intercepts the DynamicCache layer update, applies NexusQuantEvict once on
    the first prefill (seq_len > 1), and writes the compressed + eviction-masked
    cache back in-place.

    When truncate=False (default): evicted positions are zeroed and a float
    attention mask is returned via compressor.last_mask. Pass this mask as
    attention_mask when calling model.generate().

    When truncate=True: evicted tokens are PHYSICALLY REMOVED from all KV tensors,
    freeing real GPU memory and reducing attention FLOPs. RoPE is stripped at
    original positions and re-applied at contiguous positions [0,1,...,n_kept-1].
    The starting position for generation is stored in compressor.next_position.
    Pass position_ids=torch.arange(next_pos, next_pos + max_new_tokens) to generate().

    quality presets (override eviction_rate / bits / key_bits):
        "high":     K3V2, eviction_rate=0.35  (~9x,  +0.35% PPL, A100 3544-tok)
        "balanced": K2V2, eviction_rate=0.60  (~17x, +0.82% PPL, A10G 1664-tok)
        "max":      K2V2, eviction_rate=0.80  (~33x, +2.13% PPL, A10G 1664-tok)
        "asym":     K3V2, eviction_rate=0.60  (~14x, <1% PPL estimated — not fully validated end-to-end)

    Args:
        model: Any HuggingFace causal LM using DynamicCache.
        eviction_rate: Fraction of prefix tokens to evict (ignored when quality preset
            is one of the named presets; set quality=None to use this directly).
            Pass "auto" to compute per-call from attention entropy.
        quality: Preset name ("high", "balanced", "max") or None to use raw args.
        sliding_window: Recent tokens always kept (never evicted).
        obs_window: Number of recent positions used to score importance (key-key only).
        bits: E8 quantization bits for surviving tokens (2 recommended).
        truncate: If True, physically remove evicted tokens from KV tensors and
            remap RoPE to contiguous positions. Saves real GPU memory and FLOPs.
            Requires passing correct position_ids to model.generate(). Default False.
        scorer: Importance scoring method.
            "key-key" (default) -- SnapKV-style proxy using recent key vectors as
                queries. Fast, no extra forward pass needed.
            "real" -- Accumulated softmax weights from a dedicated prefill forward
                pass with output_attentions=True. Requires:
                  1. model loaded with attn_implementation='eager'
                  2. input_ids passed to this context manager
            Expected benefit over key-key: ~0.35pp at 80% eviction.
        input_ids: Required when scorer="real". The (batch, seq) prefix token ids
            used to run the importance-scoring forward pass before prefill hooks fire.
        protect_boundary: Auto-protect first/last N layers (int) or "auto" to probe
            key magnitudes and decide automatically. Default 0.
        soft_eviction: When True, low-importance tokens are quantized at 1-bit instead
            of being zeroed. The returned attention mask is all-ones (no tokens removed).
            Default False (hard eviction).
        adaptive_context: Scale eviction rate down for short prefixes (seq < 1024).
            Default False.
        verbose: Print stats on exit.

    Yields:
        compressor (NexusQuantEvict or NexusQuantEvictTruncate): active compressor.
            truncate=False: compressor.last_mask  — (batch, seq) float attention mask
            truncate=True:  compressor.next_position — int, first position for generation

    Example (masking, default):
        >>> with nexusquant_evict(model) as nq:
        ...     output = model.generate(input_ids, max_new_tokens=200,
        ...                             attention_mask=nq.last_mask)

    Example (real attention scorer):
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-v0.1", attn_implementation="eager", ...
        ... )
        >>> with nexusquant_evict(model, scorer="real", input_ids=prefix_ids) as nq:
        ...     output = model.generate(prefix_ids, max_new_tokens=200)

    Example (truncation, real memory savings):
        >>> with nexusquant_evict(model, truncate=True) as nq:
        ...     output = model.generate(input_ids, max_new_tokens=200)
    """
    # Apply quality preset if recognised
    if quality in _EVICT_PRESETS:
        preset = _EVICT_PRESETS[quality]
        eviction_rate = preset["eviction_rate"]
        bits = preset["bits"]
        if key_bits is None and "key_bits" in preset:
            key_bits = preset["key_bits"]
        if value_bits is None and "value_bits" in preset:
            value_bits = preset["value_bits"]

    # Validate real-scorer requirements early so the user gets a clear error
    if scorer == "real":
        attn_impl = getattr(model.config, "_attn_implementation", None)
        if attn_impl != "eager":
            raise ValueError(
                f"scorer='real' requires attn_implementation='eager' but model reports "
                f"attn_implementation={attn_impl!r}. "
                "SDPA suppresses attention weights silently. "
                "Reload with: AutoModelForCausalLM.from_pretrained("
                "..., attn_implementation='eager')"
            )
        if input_ids is None:
            raise ValueError(
                "scorer='real' requires input_ids to be passed to nexusquant_evict(). "
                "These are the (batch, seq) token ids for the prefix."
            )

    model_cfg = _detect_model_config(model)
    head_dim = model_cfg["head_dim"]
    rope_base = model_cfg["rope_base"]
    rope_scaling = model_cfg["rope_scaling"]  # None for most models; dict for Llama-3.1 etc.

    # Warn if the model uses KV cache offloading — our hooks target DynamicCache
    # and may silently miss offloaded cache updates.
    try:
        from transformers.cache_utils import OffloadedStaticCache as _OffloadedStaticCache
        _has_offloaded = True
    except ImportError:
        _has_offloaded = False

    if _has_offloaded and hasattr(model, "_offload_cache"):
        warnings.warn(
            "NexusQuantEvict detected KV cache offloading (_offload_cache). "
            "Compression hooks intercept DynamicCache layer updates and may not "
            "fire for offloaded cache writes. Results may be incorrect or "
            "compression may silently not apply. Disable KV cache offloading "
            "when using NexusQuant.",
            stacklevel=3,
        )

    compressor_cls = NexusQuantEvictTruncate if truncate else NexusQuantEvict
    compressor = compressor_cls(
        head_dim=head_dim,
        bits=bits,
        key_bits=key_bits,
        value_bits=value_bits,
        eviction_rate=eviction_rate,
        sliding_window=sliding_window,
        obs_window=obs_window,
        rope_base=rope_base,
        rope_scaling=rope_scaling,
        scorer=scorer,
        protected_layers=protected_layers,
        protect_boundary=protect_boundary,
        min_context_for_compression=min_context_for_compression,
        soft_eviction=soft_eviction,
        adaptive_context=adaptive_context,
        protected_positions=protected_positions,
    )
    compressor.last_mask = None       # set on first prefill (masking path)
    compressor.next_position = None   # set on first prefill (truncation path)

    # For scorer="real": run the importance-scoring forward pass NOW (before the
    # prefill hooks fire) so that the scores are ready when compress() is called
    # inside the hook. We cache them on the compressor instance.
    if scorer == "real":
        _real_importance = compressor._score_importance_real(model, input_ids)
        compressor._cached_real_importance = _real_importance
    else:
        compressor._cached_real_importance = None

    stats = _CompressionStats(bits=bits)

    try:
        from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    except ImportError:
        raise ImportError(
            "transformers >= 4.44 required for DynamicCache support. "
            "Install with: pip install 'transformers>=4.44'"
        )

    original_dl_update = DynamicLayer.update
    original_dsw_update = DynamicSlidingWindowLayer.update

    def _make_evict_hook(original_method):
        def hooked_update(self, key_states, value_states, cache_kwargs=None):
            keys, values = original_method(self, key_states, value_states, cache_kwargs)

            # Fix 2: skip if this cache layer was already compressed (multi-turn safety)
            if getattr(keys, '_nq_compressed', False):
                return keys, values

            incoming_seq = key_states.shape[-2]
            existing_seq = keys.shape[-2] - incoming_seq  # tokens in cache before this update

            # Fix 2+4: skip single-token decode steps
            if incoming_seq < 2:
                return keys, values

            # Fix 4: guard against speculative decoding draft-verification passes.
            # If there is already a large cache and the incoming chunk is tiny
            # relative to it, this is likely draft verification, not a real prefill.
            if existing_seq > 0 and incoming_seq < max(existing_seq * 0.1, 8):
                return keys, values

            # Wrap single-layer tensors into a minimal cache-like object for compress()
            class _SingleLayerCache:
                def __init__(self, k, v):
                    self.key_cache = [k]
                    self.value_cache = [v]

            single = _SingleLayerCache(keys, values)

            if scorer == "real" and compressor._cached_real_importance is not None:
                # Inject pre-computed real importance by temporarily shadowing
                # _score_importance on the instance so compress() uses them
                # without triggering another forward pass.
                cached_imp = compressor._cached_real_importance
                compressor._score_importance = lambda k, _imp=cached_imp: _imp
                result = compressor.compress(single)
                del compressor._score_importance   # remove instance override
            else:
                result = compressor.compress(single)

            if truncate:
                # result = (cache, next_position_int)
                _, next_pos = result
                compressor.next_position = next_pos
            else:
                # result = (cache, attention_mask)
                _, mask = result
                compressor.last_mask = mask

            new_k = single.key_cache[0]
            new_v = single.value_cache[0]

            # Fix 2: mark compressed so subsequent turns skip re-compression
            new_k._nq_compressed = True

            self.keys = new_k
            self.values = new_v

            stats.record(new_k, new_v)
            return new_k, new_v

        return hooked_update

    DynamicLayer.update = _make_evict_hook(original_dl_update)
    DynamicSlidingWindowLayer.update = _make_evict_hook(original_dsw_update)

    if verbose:
        mode_label = "truncate" if truncate else "mask"
        model_name = getattr(model.config, "_name_or_path", model.__class__.__name__)
        _kb = compressor.key_bits
        _vb = compressor.value_bits
        bits_label = f"K={_kb}-bit/V={_vb}-bit E8" if _kb != _vb else f"{_kb}-bit E8"
        boundary_label = f", protect_boundary={protect_boundary}" if protect_boundary else ""
        ctx_label = f", min_ctx={min_context_for_compression}" if min_context_for_compression else ""
        adaptive_label = ", adaptive_context=True" if adaptive_context else ""
        eviction_display = (
            "auto" if eviction_rate == "auto" else f"{eviction_rate:.0%}"
        )
        print(
            f"NexusQuantEvict: activating on {model_name} "
            f"(eviction={eviction_display}, {bits_label}, "
            f"sliding_window={sliding_window}, mode={mode_label}, scorer={scorer}"
            f"{boundary_label}{ctx_label}{adaptive_label})",
            flush=True,
        )

    try:
        yield compressor
    finally:
        DynamicLayer.update = original_dl_update
        DynamicSlidingWindowLayer.update = original_dsw_update

        if verbose:
            print(stats.summary(), flush=True)

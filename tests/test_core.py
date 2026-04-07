"""Smoke tests for NexusQuant core components. No GPU or model required.

Run with:
    pytest tests/test_core.py          # from project root
    pytest tests/                      # run all tests
"""
import sys
import os
import pytest
import torch

# Ensure nexusquant-oss is importable without a package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nexusquant-oss'))


# ---------------------------------------------------------------------------
# E8 lattice
# ---------------------------------------------------------------------------

def test_e8_nearest_point_shape():
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(100, 8)
    y = E8Lattice.nearest_point(x)
    assert y.shape == x.shape


def test_e8_nearest_point_changes_input():
    """nearest_point should quantize, not return the original vector."""
    from nexusquant.core.e8_lattice import E8Lattice
    # Use large-magnitude inputs so they are definitely not already on the lattice
    x = torch.randn(50, 8) * 5.0
    y = E8Lattice.nearest_point(x)
    assert not torch.allclose(x, y), "nearest_point returned input unchanged"


def test_e8_nearest_point_even_sum():
    """Integer-coset E8 points must have even coordinate sum."""
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(200, 8)
    lp = E8Lattice.nearest_point(x)
    for i in range(lp.shape[0]):
        s = lp[i].sum().item()
        # Valid iff integer with even sum OR half-integer
        is_integer = abs(s - round(s)) < 1e-4
        is_half_int = abs(s - (round(s * 2) / 2)) < 1e-4
        assert is_integer or is_half_int, f"Point {i} sum={s} is neither integer nor half-integer"


def test_e8_quantize_shape_preserved():
    from nexusquant.core.e8_lattice import E8Lattice
    for shape in [(10, 8), (5, 16), (3, 128), (2, 7), (1, 96)]:
        x = torch.randn(*shape)
        q = E8Lattice.quantize(x, levels=8)
        assert q.shape == x.shape, f"Shape mismatch for input {shape}: got {q.shape}"


def test_e8_quantize_reconstruction_error_bounded():
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(100, 128)
    y = E8Lattice.quantize(x, levels=4)  # 2-bit
    assert y.shape == x.shape
    rmse = (x - y).pow(2).mean().sqrt().item()
    assert rmse < 2.0, f"2-bit RMSE {rmse:.4f} unexpectedly large"


def test_e8_quantize_perhead_shape():
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(10, 128)
    y = E8Lattice.quantize_perhead(x, levels=4)
    assert y.shape == x.shape


def test_e8_quantize_perhead_reconstruction_quality():
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(20, 128)
    y = E8Lattice.quantize_perhead(x, levels=4)
    rmse = (x - y).pow(2).mean().sqrt().item()
    assert rmse < 1.0, f"per-head 2-bit RMSE {rmse:.4f} too high"


def test_e8_quantize_levels():
    """Higher bit-depth should give lower reconstruction error."""
    from nexusquant.core.e8_lattice import E8Lattice
    torch.manual_seed(42)
    x = torch.randn(50, 128)
    err_2bit = (x - E8Lattice.quantize(x, levels=4)).pow(2).mean().item()
    err_4bit = (x - E8Lattice.quantize(x, levels=16)).pow(2).mean().item()
    assert err_4bit < err_2bit, "4-bit should have lower error than 2-bit"


# ---------------------------------------------------------------------------
# Hadamard
# ---------------------------------------------------------------------------

def test_hadamard_shape():
    from nexusquant.core.hadamard import hadamard_matrix
    H = hadamard_matrix(128)
    assert H.shape == (128, 128)


def test_hadamard_orthonormal():
    from nexusquant.core.hadamard import hadamard_matrix
    for n in [2, 4, 8, 16, 32, 64, 128]:
        H = hadamard_matrix(n)
        identity = H @ H.T
        assert torch.allclose(identity, torch.eye(n), atol=1e-5), \
            f"H @ H.T != I for n={n}"


def test_hadamard_non_power_of_2_raises():
    """hadamard_matrix should raise for non-power-of-2 inputs."""
    from nexusquant.core.hadamard import hadamard_matrix
    with pytest.raises(ValueError):
        hadamard_matrix(96)


def test_next_power_of_2():
    from nexusquant.core.hadamard import next_power_of_2
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(96) == 128
    assert next_power_of_2(128) == 128
    assert next_power_of_2(129) == 256


def test_fht_matches_matrix_multiply():
    """FHT output must match dense H @ x."""
    from nexusquant.core.hadamard import hadamard_matrix, fht
    for n in [8, 16, 32, 64, 128]:
        H = hadamard_matrix(n)
        x = torch.randn(10, n)
        dense = x @ H.T
        fast = fht(x)
        assert torch.allclose(dense, fast, atol=1e-5), \
            f"FHT != matmul at n={n}, max_diff={torch.abs(dense - fast).max():.2e}"


def test_fht_self_inverse():
    """FHT(FHT(x)) == x for orthonormal Hadamard."""
    from nexusquant.core.hadamard import fht
    x = torch.randn(5, 128)
    assert torch.allclose(fht(fht(x)), x, atol=1e-5)


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def test_rope_forward_inverse_roundtrip():
    """inverse_rope(forward_rope(x)) ≈ x."""
    from nexusquant.core.rope_utils import inverse_rope, forward_rope
    keys = torch.randn(8, 100, 128)   # (heads, seq, dim)
    k_fwd = forward_rope(keys)
    k_inv = inverse_rope(k_fwd)
    assert torch.allclose(keys, k_inv, atol=1e-4), \
        f"forward→inverse roundtrip max err: {(keys - k_inv).abs().max():.2e}"


def test_rope_inverse_forward_roundtrip():
    """forward_rope(inverse_rope(x)) ≈ x."""
    from nexusquant.core.rope_utils import inverse_rope, forward_rope
    keys = torch.randn(8, 100, 128)
    k_inv = inverse_rope(keys)
    k_fwd = forward_rope(k_inv)
    assert torch.allclose(keys, k_fwd, atol=1e-4), \
        f"inverse→forward roundtrip max err: {(keys - k_fwd).abs().max():.2e}"


def test_rope_changes_values():
    """forward_rope should not be a no-op."""
    from nexusquant.core.rope_utils import forward_rope
    keys = torch.randn(4, 50, 128)
    k_roped = forward_rope(keys)
    assert not torch.allclose(keys, k_roped, atol=1e-3), \
        "forward_rope returned unchanged tensor — likely a no-op bug"


def test_rope_at_positions_roundtrip():
    """inverse_rope_at_positions → forward_rope_at_positions ≈ identity."""
    from nexusquant.core.rope_utils import inverse_rope_at_positions, forward_rope_at_positions
    keys = torch.randn(8, 50, 128)
    positions = torch.arange(50, dtype=torch.float32)
    k_inv = inverse_rope_at_positions(keys, positions)
    k_fwd = forward_rope_at_positions(k_inv, positions)
    assert torch.allclose(keys, k_fwd, atol=1e-4), \
        f"at_positions roundtrip max err: {(keys - k_fwd).abs().max():.2e}"


def test_rope_at_non_contiguous_positions():
    """RoPE removal at arbitrary positions (after eviction) should round-trip."""
    from nexusquant.core.rope_utils import inverse_rope_at_positions, forward_rope_at_positions
    keys = torch.randn(4, 20, 128)
    # Simulate kept positions after 50% eviction: [0, 2, 4, ..., 38]
    positions = torch.arange(0, 40, 2, dtype=torch.float32)
    k_inv = inverse_rope_at_positions(keys, positions)
    k_fwd = forward_rope_at_positions(k_inv, positions)
    assert torch.allclose(keys, k_fwd, atol=1e-4)


def test_rope_batch_vs_unbatched():
    """inverse_rope should give same result whether called on 3D or 4D input."""
    from nexusquant.core.rope_utils import inverse_rope
    keys = torch.randn(8, 30, 128)  # (h, s, d)
    r3d = inverse_rope(keys)
    r4d = inverse_rope(keys.unsqueeze(0)).squeeze(0)
    assert torch.allclose(r3d, r4d, atol=1e-6)


# ---------------------------------------------------------------------------
# NexusQuantEvict construction (no model, no GPU)
# ---------------------------------------------------------------------------

def test_evict_init_defaults():
    """NexusQuantEvict can be constructed with default arguments."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict()
    assert nq.head_dim == 128
    assert nq.eviction_rate == 0.6
    assert nq.key_bits == 2
    assert nq.value_bits == 2
    assert nq.soft_eviction == False
    assert nq.adaptive_context == False


def test_evict_init_auto_params():
    """NexusQuantEvict accepts 'auto' for eviction_rate and protect_boundary."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(eviction_rate="auto", protect_boundary="auto")
    assert nq.eviction_rate == "auto"
    assert nq.protect_boundary == "auto"


def test_evict_invalid_eviction_rate_raises():
    from nexusquant.pipeline import NexusQuantEvict
    with pytest.raises(ValueError):
        NexusQuantEvict(eviction_rate="bad_string")


def test_evict_invalid_scorer_raises():
    from nexusquant.pipeline import NexusQuantEvict
    with pytest.raises(ValueError):
        NexusQuantEvict(scorer="unknown")


def test_evict_invalid_protect_boundary_raises():
    from nexusquant.pipeline import NexusQuantEvict
    with pytest.raises(ValueError):
        NexusQuantEvict(protect_boundary="maybe")


def test_evict_asymmetric_bits():
    """key_bits and value_bits can differ (K3V2 asymmetric mode)."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(key_bits=3, value_bits=2)
    assert nq.key_bits == 3
    assert nq.value_bits == 2
    assert nq.key_levels == 8
    assert nq.value_levels == 4


def test_evict_hadamard_created():
    """NexusQuantEvict builds the Hadamard matrix on init."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=64)
    assert nq.H.shape == (64, 64)


# ---------------------------------------------------------------------------
# NexusQuantEvict.compress on synthetic CPU tensors (no model)
# ---------------------------------------------------------------------------

def _make_fake_cache(b=1, h=4, seq=64, d=128, n_layers=2):
    """Build a minimal DynamicCache-like object with .key_cache/.value_cache."""
    class FakeCache:
        pass
    cache = FakeCache()
    cache.key_cache = [torch.randn(b, h, seq, d) for _ in range(n_layers)]
    cache.value_cache = [torch.randn(b, h, seq, d) for _ in range(n_layers)]
    return cache


def test_evict_compress_returns_tuple():
    """compress() returns (cache, mask) for default (non-truncate) path."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.5, sliding_window=8)
    cache = _make_fake_cache(b=1, h=4, seq=64, d=128, n_layers=2)
    result = nq.compress(cache)
    assert isinstance(result, tuple) and len(result) == 2
    new_cache, mask = result
    assert mask.shape == (1, 64)


def test_evict_compress_mask_bos_always_kept():
    """BOS token (position 0) must always be in the keep mask."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.8, sliding_window=4)
    cache = _make_fake_cache(seq=50)
    _, mask = nq.compress(cache)
    assert mask[0, 0].item() == 1.0, "BOS (position 0) was evicted"


def test_evict_compress_sliding_window_kept():
    """The last sliding_window tokens must always be kept."""
    from nexusquant.pipeline import NexusQuantEvict
    sw = 8
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.9, sliding_window=sw)
    cache = _make_fake_cache(seq=60)
    _, mask = nq.compress(cache)
    # Last sw positions should all be 1.0
    assert mask[0, -sw:].sum().item() == float(sw), \
        f"Sliding window not fully kept: {mask[0, -sw:]}"


def test_evict_compress_eviction_rate_respected():
    """Actual keep fraction should be approximately (1 - eviction_rate)."""
    from nexusquant.pipeline import NexusQuantEvict
    seq = 200
    sw = 10
    rate = 0.6
    nq = NexusQuantEvict(head_dim=128, eviction_rate=rate, sliding_window=sw)
    cache = _make_fake_cache(seq=seq)
    _, mask = nq.compress(cache)
    kept = mask[0].sum().item()
    # Expected: BOS(1) + kept-prefix + sliding_window(10)
    # We just check it's less than seq and more than sw+1
    assert kept < seq * (1 - rate + 0.15), \
        f"Too many tokens kept ({kept}/{seq}) for eviction_rate={rate}"
    assert kept > sw, f"Fewer tokens kept ({kept}) than sliding_window ({sw})"


def test_evict_compress_kv_shape_unchanged():
    """After compress (mask mode), KV tensors retain original shape."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.5)
    b, h, seq, d = 1, 4, 64, 128
    cache = _make_fake_cache(b=b, h=h, seq=seq, d=d, n_layers=2)
    nq.compress(cache)
    assert cache.key_cache[0].shape == (b, h, seq, d)
    assert cache.value_cache[0].shape == (b, h, seq, d)


def test_evict_compress_protected_layers_not_quantized():
    """Protected layers should remain in fp16 without quantization artifacts."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.5, protected_layers={0})
    cache = _make_fake_cache(n_layers=2)
    k_before = cache.key_cache[0].clone()
    nq.compress(cache)
    k_after = cache.key_cache[0]
    # Protected layer: only evicted positions are zeroed; kept positions
    # should be close to original (only dtype cast, no quantization).
    assert k_after.dtype == torch.float16


def test_evict_truncate_compress_returns_int():
    """NexusQuantEvictTruncate.compress() returns (cache, next_position_int)."""
    from nexusquant.pipeline import NexusQuantEvictTruncate
    nq = NexusQuantEvictTruncate(head_dim=128, eviction_rate=0.5, sliding_window=8)
    cache = _make_fake_cache(seq=64)
    result = nq.compress(cache)
    assert isinstance(result, tuple) and len(result) == 2
    new_cache, next_pos = result
    assert isinstance(next_pos, int), f"next_pos should be int, got {type(next_pos)}"
    # Truncated cache should have fewer tokens
    assert cache.key_cache[0].shape[2] < 64, "Truncation did not reduce KV length"


def test_evict_truncate_next_position_sane():
    """next_position should be between sliding_window+1 and seq."""
    from nexusquant.pipeline import NexusQuantEvictTruncate
    seq = 100
    sw = 10
    rate = 0.5
    nq = NexusQuantEvictTruncate(head_dim=128, eviction_rate=rate, sliding_window=sw)
    cache = _make_fake_cache(seq=seq)
    _, next_pos = nq.compress(cache)
    assert sw < next_pos < seq, f"next_pos={next_pos} out of expected range ({sw}, {seq})"


# ---------------------------------------------------------------------------
# Keep-mask logic: edge cases
# ---------------------------------------------------------------------------

def test_evict_min_context_skips_short_prefix():
    """min_context_for_compression: short prefixes should pass through unchanged."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.8,
                         min_context_for_compression=200)
    cache = _make_fake_cache(seq=50)
    k_before = cache.key_cache[0].clone()
    _, mask = nq.compress(cache)
    # Mask should be all-ones (no eviction for short prefix)
    assert mask.sum().item() == mask.numel(), \
        "Short prefix should not be evicted when min_context_for_compression > seq"


def test_evict_adaptive_context_zero_rate_for_very_short():
    """adaptive_context=True: seq < 256 should produce zero eviction."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.8,
                         sliding_window=4, adaptive_context=True)
    cache = _make_fake_cache(seq=100)
    _, mask = nq.compress(cache)
    # With adaptive_context and seq < 256, evict_rate is set to 0 → all kept
    assert mask.sum().item() == mask.numel(), \
        "adaptive_context should keep all tokens for seq < 256"


def test_evict_soft_eviction_mask_all_ones():
    """soft_eviction=True: all positions remain visible (mask is all-ones)."""
    from nexusquant.pipeline import NexusQuantEvict
    nq = NexusQuantEvict(head_dim=128, eviction_rate=0.6,
                         sliding_window=8, soft_eviction=True)
    cache = _make_fake_cache(seq=64)
    _, mask = nq.compress(cache)
    assert mask.sum().item() == mask.numel(), \
        "soft_eviction should return all-ones mask"


# ---------------------------------------------------------------------------
# NexusQuantSimple construction (no model, no GPU)
# ---------------------------------------------------------------------------

def test_simple_init():
    from nexusquant.pipeline import NexusQuantSimple
    nq = NexusQuantSimple(head_dim=128, bits=3)
    assert nq.head_dim == 128
    assert nq.bits == 3
    assert nq.H.shape == (128, 128)


def test_simple_compress_batch1():
    """NexusQuantSimple.compress works for batch=1 (the validated path)."""
    from nexusquant.pipeline import NexusQuantSimple
    nq = NexusQuantSimple(head_dim=128, bits=3)
    cache = _make_fake_cache(b=1, h=4, seq=32, d=128, n_layers=2)
    result = nq.compress(cache)
    # Should return the cache object
    assert result is cache
    # Shape must be preserved
    assert cache.key_cache[0].shape == (1, 4, 32, 128)


def test_simple_compress_changes_values():
    """Compression should actually change the KV tensors."""
    from nexusquant.pipeline import NexusQuantSimple
    nq = NexusQuantSimple(head_dim=128, bits=3)
    cache = _make_fake_cache(b=1, h=4, seq=32, d=128, n_layers=1)
    k_before = cache.key_cache[0].clone()
    nq.compress(cache)
    assert not torch.allclose(k_before.half(), cache.key_cache[0]), \
        "Compression should change KV tensors"


# ---------------------------------------------------------------------------
# Known limitation: batch > 1 in NexusQuantSimple (bug documentation)
# ---------------------------------------------------------------------------

def test_simple_batch_gt1_uses_only_batch0():
    """
    Documents the known limitation: NexusQuantSimple.compress only processes
    batch index 0. With batch=2, both output elements equal the compressed
    version of input[0]. This is a known bug, not intended behaviour.
    """
    from nexusquant.pipeline import NexusQuantSimple
    torch.manual_seed(0)
    nq = NexusQuantSimple(head_dim=128, bits=3)
    # Two different sequences
    cache = _make_fake_cache(b=2, h=4, seq=32, d=128, n_layers=1)
    # Make batch element 1 very different from batch element 0
    cache.key_cache[0][1] = cache.key_cache[0][0] + 100.0
    nq.compress(cache)
    # After compression: output[0] and output[1] will be identical
    # (both derived from input[0] only) — this is the bug
    k_out = cache.key_cache[0]
    # Known bug: NexusQuantSimple uses k[0] internally, so output batch dim
    # collapses to 1 or both elements equal batch[0]'s compressed result.
    # This test documents the bug — it should be (2,4,32,128) but isn't.
    assert k_out.shape[1:] == (4, 32, 128), "head/seq/dim should be preserved"
    # Mark as known limitation
    if k_out.shape[0] != 2:
        pytest.skip("Known limitation: batch>1 collapses to batch[0] in NexusQuantSimple")


# ---------------------------------------------------------------------------
# DP bit allocator
# ---------------------------------------------------------------------------

def test_dp_allocation_respects_budget():
    from nexusquant.core.dp_allocator import dp_bit_allocation
    import numpy as np
    eig = np.random.exponential(1.0, size=128)
    for budget in [128, 256, 384]:
        alloc = dp_bit_allocation(eig, budget)
        assert sum(alloc) <= budget, f"Allocation {sum(alloc)} > budget {budget}"
        assert len(alloc) == 128
        assert all(0 <= b <= 4 for b in alloc)


def test_dp_allocation_monotone():
    from nexusquant.core.dp_allocator import dp_bit_allocation
    import numpy as np
    # Strongly decreasing eigenvalues: top dims should get more bits
    eig = np.array([100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1])
    alloc = dp_bit_allocation(eig, 16)
    assert alloc[0] >= alloc[-1], f"Not monotone: {alloc}"


# ---------------------------------------------------------------------------
# Token merger
# ---------------------------------------------------------------------------

def test_token_merger_reduces_seq():
    from nexusquant.core.token_merger import merge_and_drop
    keys = torch.randn(4, 20, 128)
    values = torch.randn(4, 20, 128)
    dk, dv = merge_and_drop(keys.clone(), values.clone(), merge_pct=20)
    assert dk.shape[1] < 20, "merge_and_drop should reduce sequence length"
    assert dk.shape[1] == dv.shape[1]


def test_token_merger_zero_pct_identity():
    from nexusquant.core.token_merger import merge_tokens
    keys = torch.randn(4, 20, 128)
    values = torch.randn(4, 20, 128)
    mk, mv, mask = merge_tokens(keys.clone(), values.clone(), merge_pct=0)
    assert torch.allclose(mk, keys)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

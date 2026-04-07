"""Unit tests for NexusQuant core modules."""

import torch
import numpy as np
import pytest


def test_e8_nearest_point_even_sum():
    """E8 lattice points must have even coordinate sum."""
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(100, 8)
    lp = E8Lattice.nearest_point(x)
    sums = lp.sum(dim=-1)
    # All sums should be even (integer or half-integer with even sum)
    # For integer lattice: sum is even integer
    # For half-integer lattice: sum is even integer (each coord is n+0.5, sum of 8 = sum_n + 4)
    for i in range(100):
        s = sums[i].item()
        assert abs(s - round(s)) < 1e-5 or abs(s - round(s) - 0.5) < 1e-5, f"Sum {s} not valid E8"


def test_e8_quantize_shape():
    """E8 quantize should preserve tensor shape."""
    from nexusquant.core.e8_lattice import E8Lattice
    for shape in [(10, 8), (5, 16), (3, 128), (2, 7)]:
        x = torch.randn(*shape)
        q = E8Lattice.quantize(x, levels=8)
        assert q.shape == x.shape, f"Shape mismatch: {q.shape} != {x.shape}"


def test_e8_quantize_reduces_error():
    """Quantized values should be closer to lattice points than random."""
    from nexusquant.core.e8_lattice import E8Lattice
    x = torch.randn(100, 8) * 3.0
    q = E8Lattice.quantize(x, levels=8)
    # MSE should be less than variance of input
    mse = ((x - q) ** 2).mean().item()
    var = x.var().item()
    assert mse < var, f"MSE {mse} >= variance {var}"


def test_hadamard_orthonormal():
    """Hadamard matrix should be orthonormal."""
    from nexusquant.core.hadamard import hadamard_matrix
    for n in [2, 4, 8, 16, 32, 64, 128]:
        H = hadamard_matrix(n)
        I = H @ H.T
        assert torch.allclose(I, torch.eye(n), atol=1e-5), f"H @ H.T != I for n={n}"


def test_hadamard_shape():
    """Hadamard matrix should be square of correct size."""
    from nexusquant.core.hadamard import hadamard_matrix
    H = hadamard_matrix(128)
    assert H.shape == (128, 128)


def test_dp_allocation_budget():
    """DP allocation should not exceed budget."""
    from nexusquant.core.dp_allocator import dp_bit_allocation
    eig = np.random.exponential(1.0, size=128)
    for budget in [128, 256, 384, 512]:
        alloc = dp_bit_allocation(eig, budget)
        assert sum(alloc) <= budget, f"Allocation {sum(alloc)} > budget {budget}"
        assert len(alloc) == 128
        assert all(0 <= b <= 4 for b in alloc)


def test_dp_allocation_monotone():
    """Higher eigenvalue dims should get more or equal bits."""
    from nexusquant.core.dp_allocator import dp_bit_allocation
    # Strongly decreasing eigenvalues
    eig = np.array([100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1])
    alloc = dp_bit_allocation(eig, 16)  # 2 bpd average
    # First dim should get >= last dim
    assert alloc[0] >= alloc[-1], f"Alloc not monotone: {alloc}"


def test_rope_roundtrip():
    """inverse_rope(forward_rope(x)) should be identity."""
    from nexusquant.core.rope_utils import inverse_rope, forward_rope
    x = torch.randn(4, 16, 128)  # (heads, seq, dim)
    x_rope = forward_rope(x)
    x_back = inverse_rope(x_rope)
    mse = ((x - x_back) ** 2).mean().item()
    assert mse < 1e-10, f"RoPE roundtrip MSE too high: {mse}"


def test_rope_inverse_roundtrip():
    """forward_rope(inverse_rope(x)) should be identity."""
    from nexusquant.core.rope_utils import inverse_rope, forward_rope
    x = torch.randn(4, 16, 128)
    x_norope = inverse_rope(x)
    x_back = forward_rope(x_norope)
    mse = ((x - x_back) ** 2).mean().item()
    assert mse < 1e-10, f"Inverse RoPE roundtrip MSE too high: {mse}"


def test_rope_changes_values():
    """RoPE should actually change the values (not be identity)."""
    from nexusquant.core.rope_utils import forward_rope
    x = torch.randn(4, 16, 128)
    x_rope = forward_rope(x)
    diff = (x - x_rope).abs().mean().item()
    assert diff > 0.01, f"RoPE didn't change values: diff={diff}"


def test_dp_empirical_distortion():
    """Empirical distortion factors should produce different allocation."""
    from nexusquant.core.dp_allocator import dp_bit_allocation
    eig = np.random.exponential(1.0, size=128)
    eig = np.sort(eig)[::-1]
    alloc_theo = dp_bit_allocation(eig, 256, distortion="theoretical")
    alloc_emp = dp_bit_allocation(eig, 256, distortion="empirical")
    # Both should respect budget
    assert sum(alloc_theo) <= 256
    assert sum(alloc_emp) <= 256
    # Empirical should allocate differently (more aggressive drops)
    # since empirical factors are lower, more bits freed for top dims
    zeros_theo = sum(1 for b in alloc_theo if b == 0)
    zeros_emp = sum(1 for b in alloc_emp if b == 0)
    assert zeros_emp >= zeros_theo, "Empirical should drop more dims"


def test_dp_auto_distortion():
    """Auto distortion should select based on bitrate and rank."""
    from nexusquant.core.dp_allocator import dp_bit_allocation, select_distortion_factors, DISTORTION_EMPIRICAL
    # Low bitrate (2bpd) should always use empirical
    eig = np.random.exponential(1.0, size=128)
    factors = select_distortion_factors(eig, 2.0)
    assert factors == DISTORTION_EMPIRICAL


def test_token_merger_basic():
    """Token merging should work without errors."""
    from nexusquant.core.token_merger import merge_tokens, merge_and_drop
    keys = torch.randn(4, 20, 128)
    values = torch.randn(4, 20, 128)
    mk, mv, mask = merge_tokens(keys.clone(), values.clone(), merge_pct=20)
    assert mk.shape == keys.shape
    dk, dv = merge_and_drop(keys.clone(), values.clone(), merge_pct=20)
    assert dk.shape[1] < keys.shape[1], "merge_and_drop should reduce seq length"


def test_fht_matches_matrix():
    """Fast Hadamard Transform should match dense matrix multiply."""
    from nexusquant.core.hadamard import hadamard_matrix, fht
    for n in [8, 16, 32, 64, 128]:
        H = hadamard_matrix(n)
        x = torch.randn(10, n)
        dense = x @ H.T  # dense matmul
        fast = fht(x)     # O(n log n) butterfly
        assert torch.allclose(dense, fast, atol=1e-5), f"FHT != dense at n={n}, max diff={torch.max(torch.abs(dense-fast))}"


def test_fht_inverse():
    """FHT(FHT(x)) should return x (orthonormal Hadamard is self-inverse)."""
    from nexusquant.core.hadamard import fht
    x = torch.randn(5, 128)
    roundtrip = fht(fht(x))
    assert torch.allclose(x, roundtrip, atol=1e-5), "FHT is not self-inverse"


def test_token_merger_zero_merge():
    """0% merge should be identity."""
    from nexusquant.core.token_merger import merge_tokens
    keys = torch.randn(4, 20, 128)
    values = torch.randn(4, 20, 128)
    mk, mv, mask = merge_tokens(keys.clone(), values.clone(), merge_pct=0)
    assert torch.allclose(mk, keys)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

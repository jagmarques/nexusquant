"""Walsh-Hadamard transform for KV cache decorrelation.

Provides both the full matrix (for compatibility) and the O(n log n)
Fast Hadamard Transform (FHT) for correctness testing / reference.

NOTE: The pipeline uses dense einsum (x @ H) for GPU, which is faster
than the Python-loop FHT for any practical batch size. The fht() function
is provided for correctness testing and CPU reference only.
"""

import math
import torch


def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def hadamard_matrix(n: int) -> torch.Tensor:
    """Generate normalized Walsh-Hadamard matrix of size n x n.

    For applying the transform, prefer fht() which is O(n log n) instead
    of the O(n^2) matrix multiply this matrix requires.

    Args:
        n: Matrix dimension (must be power of 2)

    Returns:
        Orthonormal Hadamard matrix (n, n)
    """
    if n & (n - 1) != 0:
        raise ValueError(f"n must be power of 2, got {n}. Use pad_to_power_of_2() first.")
    if n == 1:
        return torch.tensor([[1.0]])
    h = hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([h, h], dim=1),
        torch.cat([h, -h], dim=1)
    ], dim=0) / math.sqrt(2)


def fht(x: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard Transform -- O(n log n) butterfly, REFERENCE IMPLEMENTATION.

    WARNING: This uses a Python for-loop over butterfly stages, which incurs
    significant Python overhead. On GPU, the dense matmul (x @ H) via einsum
    is substantially faster for any practical batch size. This function is
    provided for correctness testing and CPU reference only. The main pipeline
    uses torch.einsum('...d,de->...e', x, H) for production workloads.

    Mathematically equivalent to x @ H where H = hadamard_matrix(d).

    Args:
        x: (..., d) tensor where d is a power of 2

    Returns:
        Hadamard-transformed tensor, same shape
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Last dim must be power of 2, got {d}"

    result = x.clone()
    h = 1
    while h < d:
        # Butterfly: pairs at distance h
        for i in range(0, d, h * 2):
            a = result[..., i:i+h].clone()
            b = result[..., i+h:i+2*h].clone()
            result[..., i:i+h] = a + b
            result[..., i+h:i+2*h] = a - b
        h *= 2

    # Normalize
    result = result / math.sqrt(d)
    return result


def ifht(x: torch.Tensor) -> torch.Tensor:
    """Inverse Fast Hadamard Transform.

    For orthonormal Hadamard, the inverse equals the forward transform.
    """
    return fht(x)

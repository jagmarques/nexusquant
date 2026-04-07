"""Dynamic programming optimal bit allocation for PCA dimensions.

Given eigenvalues from PCA decomposition and a total bit budget,
allocates {0,1,2,3,4} bits per dimension to minimize total
weighted distortion: sum(eigenvalue_i * distortion_factor(bits_i)).

Supports both theoretical (Gaussian assumption) and empirical distortion
factors. Empirical factors are 1.5-2.4x lower because E8 lattice VQ
performs better on real KV cache data than Gaussian theory predicts.
"""

from typing import List, Optional, Dict
import numpy as np


# Theoretical: Gaussian-optimal distortion factors (standard rate-distortion)
DISTORTION_THEORETICAL = {0: 1.0, 1: 0.363, 2: 0.117, 3: 0.034, 4: 0.0095}

# Empirical: Measured on Llama-3.1-8B KV cache after Hadamard + E8
# E8 is 1.5-2.4x better than Gaussian theory on real KV data
DISTORTION_EMPIRICAL = {0: 1.0, 1: 0.244, 2: 0.063, 3: 0.016, 4: 0.004}

# Legacy alias
DISTORTION_FACTORS = DISTORTION_THEORETICAL


def marchenko_pastur_threshold(eigenvalues: np.ndarray, n_samples: int, n_dims: int) -> int:
    """Compute signal dimension count using Marchenko-Pastur noise edge.

    Random Matrix Theory: for random (n, p) data with noise variance sigma^2,
    the bulk eigenvalue distribution has upper edge sigma^2 * (1 + sqrt(p/n))^2.
    Eigenvalues above this edge are signal; below are noise (safe to drop).

    Returns the number of signal dimensions.
    """
    gamma = n_dims / max(n_samples, 1)
    sigma2 = float(np.median(eigenvalues))
    edge = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return int(np.sum(eigenvalues > edge))


def select_distortion_factors(
    eigenvalues: np.ndarray,
    bits_per_dim: float,
) -> Dict[int, float]:
    """Adaptive distortion selection by bitrate and effective rank.

    Universal rule validated on 4 model families:
    - <=2.0 bpd (>=8x): Always empirical
    - 2.5 bpd (6.4x): Empirical (safe default)
    - >=3.0 bpd (<=5.3x): Auto-select by rank (rank>40 -> empirical, else theoretical)
    """
    rank = int(np.sum(eigenvalues > eigenvalues.max() * 0.01))
    if bits_per_dim <= 2.5:
        return DISTORTION_EMPIRICAL
    if rank > 40:
        return DISTORTION_EMPIRICAL
    return DISTORTION_THEORETICAL


def dp_bit_allocation(
    eigenvalues: np.ndarray,
    total_budget: int,
    max_bits: int = 4,
    distortion: str = "auto",
    custom_factors: Optional[Dict[int, float]] = None,
) -> List[int]:
    """Compute optimal bit allocation via dynamic programming.

    Args:
        eigenvalues: PCA eigenvalues (variance per dimension), shape (n_dims,)
        total_budget: Total bits to allocate across all dimensions
        max_bits: Maximum bits per dimension (default 4)
        distortion: "theoretical", "empirical", or "auto" (adaptive selection)
        custom_factors: Override with custom distortion factors dict

    Returns:
        List of bit allocations, one per dimension
    """
    n = len(eigenvalues)
    bits_per_dim = total_budget / max(n, 1)

    if custom_factors is not None:
        factors = custom_factors
    elif distortion == "empirical":
        factors = DISTORTION_EMPIRICAL
    elif distortion == "theoretical":
        factors = DISTORTION_THEORETICAL
    else:  # auto
        factors = select_distortion_factors(eigenvalues, bits_per_dim)

    dp = [[float('inf')] * (total_budget + 1) for _ in range(n + 1)]
    choice = [[0] * (total_budget + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        lam = max(float(eigenvalues[i - 1]), 1e-10)
        for b in range(total_budget + 1):
            for bits in range(min(max_bits + 1, b + 1)):
                cost = lam * factors[bits]
                if dp[i - 1][b - bits] + cost < dp[i][b]:
                    dp[i][b] = dp[i - 1][b - bits] + cost
                    choice[i][b] = bits

    alloc = []
    b = total_budget
    for i in range(n, 0, -1):
        alloc.append(choice[i][b])
        b -= choice[i][b]
    return list(reversed(alloc))


def calibrate_distortion_factors(
    data: np.ndarray,
    head_dim: int = 128,
    max_bits: int = 4,
) -> Dict[int, float]:
    """Calibrate empirical distortion factors from real KV cache data.

    Measures actual E8 quantization distortion at each bitrate on provided data,
    producing model-specific factors that are typically 1.5-2.4x lower than theory.

    Args:
        data: KV cache vectors in PCA space, shape (n_samples, head_dim)
        head_dim: Dimension of each vector
        max_bits: Maximum bits to calibrate

    Returns:
        Dictionary mapping bits -> distortion factor (relative to unquantized)
    """
    import torch
    import torch.nn.functional as F
    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    total_var = data.var().item()
    if total_var < 1e-10:
        return DISTORTION_THEORETICAL.copy()

    factors = {0: 1.0}
    for bits in range(1, max_bits + 1):
        levels = 2 ** bits
        pad = (8 - head_dim % 8) % 8
        padded = F.pad(data, (0, pad)) if pad > 0 else data

        p2 = 1
        while p2 < padded.shape[-1]:
            p2 *= 2
        H = hadamard_matrix(p2)
        padded = F.pad(padded, (0, p2 - padded.shape[-1]))
        rotated = padded @ H.T
        quantized = E8Lattice.quantize(rotated, levels=levels)
        reconstructed = (quantized @ H)[:, :head_dim]

        mse = ((data - reconstructed) ** 2).mean().item()
        factors[bits] = mse / total_var

    return factors

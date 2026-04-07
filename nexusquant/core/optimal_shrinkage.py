"""Optimal singular value shrinkage for PCA denoising.

Gavish-Donoho (2014) optimal hard/soft thresholding for
Frobenius-loss optimal denoising of low-rank matrices.

Applied to KV cache PCA: instead of hard MP cutoff (binary keep/drop),
shrink eigenvalues optimally. This preserves weak signal that hard
thresholding destroys, critical for quality at aggressive compression.

Reference: Gavish & Donoho, "The Optimal Hard Threshold for Singular
Values is 4/sqrt(3)", IEEE Trans. Inf. Theory 60(8), 2014.
"""

import numpy as np
from typing import Tuple


def mp_median(gamma: float, n_samples: int = 1000) -> float:
    """Numerically approximate the median of the Marchenko-Pastur distribution.

    The MP distribution on [beta_minus, beta_plus] has density:
        f(x) = sqrt((beta_plus - x)(x - beta_minus)) / (2*pi*gamma*x)

    We integrate by sampling to find the median.

    Args:
        gamma: aspect ratio p/n, must be in (0, 1]
        n_samples: quadrature points

    Returns:
        Approximate MP median (with sigma=1)
    """
    if gamma <= 0:
        return 1.0
    gamma = min(gamma, 1.0)
    beta_plus = (1.0 + np.sqrt(gamma)) ** 2
    beta_minus = (1.0 - np.sqrt(gamma)) ** 2

    if beta_plus <= beta_minus:
        return 1.0

    # Grid over the MP support
    x = np.linspace(beta_minus + 1e-10, beta_plus - 1e-10, n_samples)
    # MP density (unnormalised)
    density = np.sqrt(np.maximum((beta_plus - x) * (x - beta_minus), 0.0)) / (
        2.0 * np.pi * gamma * x
    )
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    # Normalise
    cdf /= cdf[-1]
    # Find median (CDF = 0.5)
    idx = np.searchsorted(cdf, 0.5)
    idx = int(np.clip(idx, 0, len(x) - 1))
    return float(x[idx])


def estimate_noise_sigma(
    eigenvalues: np.ndarray, n_samples: int, n_dims: int
) -> float:
    """Estimate noise level sigma from the eigenvalue spectrum.

    Uses the bulk eigenvalues (below the rough MP upper edge) to estimate
    sigma^2 via the MP median. Falls back to median of the lower half when
    gamma >= 1 (square or fat matrices).

    Args:
        eigenvalues: PCA eigenvalues (variance per component), sorted descending
        n_samples: number of data points used for PCA
        n_dims: original feature dimensionality

    Returns:
        Estimated noise standard deviation sigma >= 0
    """
    gamma = n_dims / max(n_samples, 1)

    if len(eigenvalues) == 0:
        return 1e-5

    # Rough upper edge estimate using all eigenvalues for sigma
    sigma2_rough = float(np.median(eigenvalues))
    rough_upper = sigma2_rough * (1.0 + np.sqrt(gamma)) ** 2

    # Select bulk eigenvalues below the rough upper edge
    bulk = eigenvalues[eigenvalues <= rough_upper]
    if len(bulk) == 0:
        # All eigenvalues above edge - very low-rank signal; use bottom quarter
        bulk = eigenvalues[len(eigenvalues) // 4 * 3 :]
    if len(bulk) == 0:
        bulk = eigenvalues

    med_bulk = float(np.median(bulk))

    # sigma^2 = median(bulk eigenvalues) / mp_median(gamma)
    # When gamma << 1 the MP median ~ 1, so this reduces to the raw median
    mp_med = mp_median(min(gamma, 1.0))
    if mp_med < 1e-10:
        mp_med = 1.0

    sigma2 = med_bulk / mp_med
    return float(np.sqrt(max(sigma2, 1e-10)))


def optimal_hard_threshold(
    n_samples: int, n_dims: int, sigma: float
) -> float:
    """Gavish-Donoho optimal hard threshold for singular values (Frobenius loss).

    Equation (11) from the paper - applies when sigma is known.

    Args:
        n_samples: rows of the data matrix
        n_dims: columns (dimensionality)
        sigma: noise standard deviation

    Returns:
        Threshold tau on the singular value scale (not eigenvalue scale).
        Singular values above tau are kept; below are zeroed.
    """
    gamma = n_dims / max(n_samples, 1)
    # Simplified optimal threshold formula from Gavish-Donoho eq (11)
    numerator = 8.0 * gamma
    denominator = (gamma + 1.0) + np.sqrt(gamma ** 2 + 14.0 * gamma + 1.0)
    tau = np.sqrt(2.0 * (gamma + 1.0) + numerator / denominator) * sigma
    return float(tau)


def optimal_shrinkage_frobenius(
    eigenvalues: np.ndarray,
    n_samples: int,
    n_dims: int,
    sigma: float = None,
) -> np.ndarray:
    """Gavish-Donoho optimal soft shrinkage (Frobenius loss).

    For each component with eigenvalue lambda_i = s_i^2 / (n-1):
      - If s_i^2 <= beta_plus  (inside MP bulk): eigenvalue set to 0
      - If s_i^2 > beta_plus   (signal): apply the SURE-optimal shrinkage

    The shrinkage formula (Gavish-Donoho 2014, eq. 20):
        s_shrunk = sqrt(max(0, (s^2 - beta_plus)(s^2 - beta_minus))) / s

    Args:
        eigenvalues: PCA eigenvalues (variance per component), any order
        n_samples: number of data points used in PCA
        n_dims: original dimensionality p
        sigma: noise level. If None, estimated from data.

    Returns:
        Shrunk eigenvalues, same shape. Noise dims -> 0.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    gamma = n_dims / max(n_samples, 1)

    if sigma is None:
        sigma = estimate_noise_sigma(eigenvalues, n_samples, n_dims)

    sigma2 = sigma ** 2
    n = max(n_samples - 1, 1)

    # Marchenko-Pastur bulk edges (on singular value squared scale)
    beta_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2 * n
    beta_minus = sigma2 * (1.0 - np.sqrt(gamma)) ** 2 * n

    # Convert PCA eigenvalues -> singular values squared
    # PCA eigenvalue = s^2 / (n_samples - 1)
    sv_sq = eigenvalues * n

    shrunk = np.zeros_like(eigenvalues)
    for i, s2 in enumerate(sv_sq):
        if s2 <= beta_plus:
            # Inside noise bulk - zero out
            shrunk[i] = 0.0
        else:
            # Signal component - apply optimal shrinkage
            above = s2 - beta_plus
            below = s2 - beta_minus
            if above > 0 and below > 0 and s2 > 0:
                # Shrunk singular value
                s_shrunk = np.sqrt(above * below) / np.sqrt(s2)
                # Convert back to eigenvalue scale
                shrunk[i] = (s_shrunk ** 2) / n

    return shrunk


def effective_dims(shrunk_eigenvalues: np.ndarray, threshold: float = 0.01) -> int:
    """Count non-negligible dimensions after shrinkage.

    Args:
        shrunk_eigenvalues: output of optimal_shrinkage_frobenius
        threshold: relative energy threshold (fraction of total)

    Returns:
        Number of components carrying at least threshold * total variance
    """
    total = shrunk_eigenvalues.sum()
    if total <= 0.0:
        return 0
    return int(np.sum(shrunk_eigenvalues > threshold * total))


def variance_retained(
    original_eigenvalues: np.ndarray, shrunk_eigenvalues: np.ndarray
) -> float:
    """Fraction of original total variance kept after shrinkage.

    Args:
        original_eigenvalues: before shrinkage
        shrunk_eigenvalues: after shrinkage

    Returns:
        Value in [0, 1]
    """
    orig_total = original_eigenvalues.sum()
    if orig_total <= 0.0:
        return 0.0
    return float(shrunk_eigenvalues.sum() / orig_total)


def reconstruct_with_shrinkage(
    X: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    original_eigenvalues: np.ndarray,
    shrunk_eigenvalues: np.ndarray,
) -> np.ndarray:
    """Reconstruct data matrix using shrunk eigenvalues.

    Compresses and reconstructs X using only components where the shrunk
    eigenvalue is positive, weighted by the shrinkage factor.

    Args:
        X: (n_samples, n_dims) data matrix
        components: (n_components, n_dims) PCA components (eigenvectors)
        mean: (n_dims,) PCA mean vector
        original_eigenvalues: (n_components,) original PCA eigenvalues
        shrunk_eigenvalues: (n_components,) shrunk eigenvalues

    Returns:
        (n_samples, n_dims) reconstructed matrix
    """
    X_centered = X - mean

    # Project onto PCA basis
    scores = X_centered @ components.T  # (n, n_components)

    # Apply shrinkage weights: scale each component by sqrt(lambda_shrunk / lambda_orig)
    weights = np.zeros(len(original_eigenvalues))
    for i in range(len(original_eigenvalues)):
        if original_eigenvalues[i] > 1e-12 and shrunk_eigenvalues[i] > 0:
            weights[i] = np.sqrt(shrunk_eigenvalues[i] / original_eigenvalues[i])

    scores_shrunk = scores * weights  # (n, n_components)

    # Reconstruct
    X_recon = scores_shrunk @ components + mean
    return X_recon

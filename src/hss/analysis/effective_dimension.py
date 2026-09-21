"""Participation dimensions of frozen MFA covariances and observed clusters.

No fitting or feature normalization. Low-rank trace identities avoid forming
the D x D covariance, and are exact up to floating-point roundoff.
"""
import numpy as np

from hss.analysis.cluster_profiles import participation_dimension


def covariance_dimensions(loadings, noise):
    w = np.asarray(loadings, dtype=np.float64)
    psi = np.asarray(noise, dtype=np.float64)
    if w.ndim != 2 or psi.shape != (w.shape[0],):
        raise ValueError("Expected loadings D x rank and noise D")
    if not np.isfinite(w).all() or not np.isfinite(psi).all() or (psi < 0).any():
        raise ValueError("Finite factors and nonnegative noise are required")
    gram = w.T @ w
    factor_trace = float(np.trace(gram))
    factor_square_trace = float(np.square(gram).sum())
    noise_trace = float(psi.sum())
    noise_square_trace = float(psi @ psi)
    cross_trace = float(psi @ np.square(w).sum(1))
    trace = factor_trace + noise_trace
    square_trace = factor_square_trace + 2 * cross_trace + noise_square_trace
    eigen = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.)
    fraction = eigen / factor_trace if factor_trace else np.zeros_like(eigen)
    positive = fraction > 0
    cumulative = np.cumsum(fraction)
    return dict(
        rank=w.shape[1], hidden_dim=w.shape[0],
        factor_pr=factor_trace ** 2 / factor_square_trace if factor_square_trace else 0.,
        model_pr=trace ** 2 / square_trace if square_trace else 0.,
        noise_pr=noise_trace ** 2 / noise_square_trace if noise_square_trace else 0.,
        factor_trace=factor_trace, noise_trace=noise_trace, total_trace=trace,
        total_square_trace=square_trace, cross_trace=cross_trace,
        factor_fraction=factor_trace / trace if trace else 0.,
        factor_entropy_rank=float(np.exp(-np.sum(fraction[positive] * np.log(fraction[positive])))) if positive.any() else 0.,
        factor_top1_fraction=float(fraction[0]) if len(fraction) else 0.,
        factor_d90=int(np.searchsorted(cumulative, .90) + 1) if factor_trace else 0,
        factor_d95=int(np.searchsorted(cumulative, .95) + 1) if factor_trace else 0,
        factor_eigenvalues=eigen.tolist(),
    )


def empirical_dimensions(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty observation matrix")
    trace = float(np.square(x - x.mean(0)).sum() / len(x))
    return dict(empirical_pr=participation_dimension(x), empirical_trace=trace,
                empirical_rank_ceiling=min(len(x) - 1, x.shape[1]))


def matched_size_dimensions(x, size, repeats, seed):
    """Fixed-size without-replacement sensitivity; repeat quantiles are not CIs."""
    if size < 2 or repeats < 1:
        raise ValueError("Need at least two rows and one repeat")
    if len(x) < size:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    return np.array([participation_dimension(x[rng.choice(len(x), size, replace=False)])
                     for _ in range(repeats)])


def weighted_quantile(values, weights, probabilities):
    """Inverse CDF of a weighted discrete distribution (no interpolation)."""
    x, w = np.asarray(values, float), np.asarray(weights, float)
    q = np.asarray(probabilities, float)
    if x.shape != w.shape or (w < 0).any() or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Invalid weighted distribution")
    if not len(x) or w.sum() <= 0 or ((q < 0) | (q > 1)).any():
        raise ValueError("Positive mass and probabilities in [0,1] required")
    keep = w > 0; x, w = x[keep], w[keep]
    order = np.argsort(x)
    cdf = np.cumsum(w[order]) / w.sum()
    return x[order[np.minimum(np.searchsorted(cdf, q, side='left'), len(order) - 1)]]

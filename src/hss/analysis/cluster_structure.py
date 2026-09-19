"""Descriptive covariance diagnostics for fixed, already-fitted clusters.

Clustering inputs and labels are never modified. PCA is used only for display.
Empirical spectra describe hard-assigned observations; MFA factor/noise traces
describe its fitted soft mixture covariance. These are deliberately separate.
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd


def empirical_spectrum(X, components=32, seed=42):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not len(X):
        raise ValueError("Expected a nonempty matrix")
    mean = X.mean(axis=0)
    centered = X - mean
    trace = float(np.square(centered).sum() / len(X))
    q = min(components, len(X) - 1, X.shape[1])
    if q < 1 or trace == 0:
        return dict(n=len(X), trace=trace, eigenvalues=[], top_fraction=[],
                    mean_norm=float(np.linalg.norm(mean))), np.zeros((0, X.shape[1]))
    _, singular, vt = randomized_svd(centered, n_components=q, n_iter=5, random_state=seed)
    eigen = np.maximum(singular ** 2 / len(X), 0)
    return dict(n=len(X), trace=trace, eigenvalues=eigen.tolist(),
                top_fraction=(np.cumsum(eigen) / trace).tolist(),
                mean_norm=float(np.linalg.norm(mean))), vt


def factor_structure(loadings, noise):
    W, psi = np.asarray(loadings, dtype=float), np.asarray(noise, dtype=float)
    if W.ndim != 2 or psi.shape != (len(W),) or (psi <= 0).any():
        raise ValueError("Invalid factor covariance")
    factor_trace, noise_trace = float(np.square(W).sum()), float(psi.sum())
    if W.shape[1]:
        U, singular, _ = np.linalg.svd(W, full_matrices=False)
        energies = singular ** 2
        stable = energies > max(float(energies.max()), 1.0) * 1e-12
        basis = U[:, stable]
    else:
        energies, basis = np.array([]), np.zeros((len(W), 0))
    return dict(factor_trace=factor_trace, noise_trace=noise_trace,
                total_trace=factor_trace + noise_trace,
                factor_fraction=factor_trace / (factor_trace + noise_trace),
                factor_eigenvalues=energies.tolist(),
                noise_min=float(psi.min()), noise_median=float(np.median(psi)),
                noise_max=float(psi.max())), basis


def subspace_overlap(a, b):
    """Mean squared cosine of principal angles, invariant to factor rotation."""
    q = min(a.shape[1], b.shape[1])
    return float(np.square(a.T @ b).sum() / q) if q else None


def variance_decomposition(X, assignments):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(assignments)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("Assignment rows must match X")
    mean = X.mean(0)
    total = float(np.square(X - mean).sum() / len(X))
    within, between = 0.0, 0.0
    for k in np.unique(y):
        part = X[y == k]
        center = part.mean(0)
        within += float(np.square(part - center).sum() / len(X))
        between += len(part) / len(X) * float(np.square(center - mean).sum())
    return dict(total=total, within=within, between=between,
                between_fraction=between / total if total else 0,
                identity_error=abs(total - within - between))

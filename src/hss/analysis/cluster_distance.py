"""Distances to frozen mixture components in the original feature space."""

import numpy as np
from scipy.stats import rankdata


def factor_distances(X, mean, loadings, noise):
    """MFA Mahalanobis distance and rotation-invariant decompositions.

    Q = ||z_MAP||^2 + ||delta - W z_MAP||^2_{Psi^-1}.
    Orthogonal parallel/perpendicular energies use the Euclidean W column span;
    they are a different decomposition from this penalized latent MAP identity.
    """
    delta = np.asarray(X, dtype=np.float64) - mean
    W, psi = np.asarray(loadings, dtype=float), np.asarray(noise, dtype=float)
    if np.any(psi <= 0) or W.shape[0] != delta.shape[1]:
        raise ValueError("Invalid MFA covariance")
    m = np.eye(W.shape[1]) + W.T @ (W / psi[:, None])
    z = np.linalg.solve(m, ((delta / psi) @ W).T).T
    residual = delta - z @ W.T
    latent_energy = np.square(z).sum(1)
    residual_energy = (np.square(residual) / psi).sum(1)
    if W.shape[1]:
        u, s, _ = np.linalg.svd(W, full_matrices=False)
        basis = u[:, s > max(s.max(), 1) * 1e-12]
        parallel = np.square(delta @ basis).sum(1)
    else:
        parallel = np.zeros(len(delta))
    euclidean = np.square(delta).sum(1)
    return dict(euclidean=np.sqrt(euclidean), mahalanobis=np.sqrt(latent_energy + residual_energy),
                latent_map_norm=np.sqrt(latent_energy), residual_psi_norm=np.sqrt(residual_energy),
                parallel=np.sqrt(np.maximum(parallel, 0)),
                perpendicular=np.sqrt(np.maximum(euclidean - parallel, 0)))


def cluster_percentiles(values, labels):
    """Within-cluster mid-ranks, for descriptive plots (not tail p-values)."""
    values, labels = np.asarray(values), np.asarray(labels)
    result = np.empty(len(values), dtype=float)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        result[idx] = (rankdata(values[idx], method="average") - .5) / len(idx)
    return result

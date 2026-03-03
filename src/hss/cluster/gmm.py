"""GMMModel with pure-numpy inference — no sklearn runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    from ..utils import f32
    from .base import ClusterModel
except ImportError:
    # Standalone testing
    def f32(x):
        x = np.asarray(x)
        if x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
            return x
        return np.ascontiguousarray(x, dtype=np.float32)

    class ClusterModel:
        pass

# =========================================================================
# Pure-numpy GMM inference helpers
# =========================================================================

def _log_det_cholesky(precisions_chol: np.ndarray, cov_type: str, n_features: int) -> np.ndarray:
    """Compute log-det from Cholesky of precisions, matching sklearn's internal logic."""
    if cov_type == "full":
        # precisions_chol: (K, D, D) lower-triangular
        return np.array([
            np.sum(np.log(np.diag(precisions_chol[k])))
            for k in range(precisions_chol.shape[0])
        ])
    if cov_type == "tied":
        # precisions_chol: (D, D)
        return np.array([np.sum(np.log(np.diag(precisions_chol)))])
    if cov_type == "diag":
        # precisions_chol: (K, D)
        return np.sum(np.log(precisions_chol), axis=1)
    if cov_type == "spherical":
        # precisions_chol: (K,)
        return n_features * np.log(precisions_chol)
    raise ValueError(f"Unknown covariance_type: {cov_type}")


def _estimate_log_prob(
    X: np.ndarray,
    means: np.ndarray,
    precisions_chol: np.ndarray,
    cov_type: str,
) -> np.ndarray:
    """Compute log N(X; mu_k, Sigma_k) for each sample and component.

    Returns (N, K) array of log-probabilities (up to a constant).
    """
    N, D = X.shape
    K = means.shape[0]

    log_det = _log_det_cholesky(precisions_chol, cov_type, D)

    if cov_type == "full":
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]  # (N, D)
            y = diff @ precisions_chol[k]  # (N, D)  — sklearn convention
            log_prob[:, k] = np.sum(y * y, axis=1)

    elif cov_type == "tied":
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            y = diff @ precisions_chol  # sklearn convention
            log_prob[:, k] = np.sum(y * y, axis=1)
        # log_det is scalar, broadcast to K
        log_det = np.full(K, log_det[0])

    elif cov_type == "diag":
        # precisions_chol: (K, D) — element-wise sqrt of precision
        # Mahalanobis: sum_d (prec_chol[k,d] * (x_d - mu_kd))^2
        precisions = precisions_chol ** 2  # (K, D)
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]  # (N, D)
            log_prob[:, k] = np.sum(diff * diff * precisions[k], axis=1)

    elif cov_type == "spherical":
        precisions = precisions_chol ** 2  # (K,)
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            log_prob[:, k] = precisions[k] * np.sum(diff * diff, axis=1)

    else:
        raise ValueError(f"Unknown covariance_type: {cov_type}")

    # log N(x; mu, Sigma) = -0.5 * D * log(2π) + log_det - 0.5 * mahalanobis
    return -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det


def _estimate_log_weights(weights: np.ndarray) -> np.ndarray:
    return np.log(weights)


def _log_resp(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    precisions_chol: np.ndarray,
    cov_type: str,
) -> np.ndarray:
    """Compute log-responsibilities (N, K), unnormalized."""
    log_prob = _estimate_log_prob(X, means, precisions_chol, cov_type)
    return log_prob + _estimate_log_weights(weights)  # (N, K)


# =========================================================================
# GMMModel
# =========================================================================

@dataclass(frozen=True)
class GMMModel(ClusterModel):
    """Gaussian Mixture Model with pure-numpy inference.

    After construction, all inference (predict, predict_proba, score_samples,
    bic, aic) uses only numpy — no sklearn objects are created or stored.
    sklearn is used ONLY in _fit_gmm() for the actual EM fitting.
    """

    weights_: np.ndarray           # (K,)
    means_: np.ndarray             # (K, D)
    covariances_: np.ndarray       # shape depends on cov_type
    precisions_cholesky_: np.ndarray  # shape depends on cov_type
    covariance_type: str
    reg_covar: float

    # ---- ClusterModel interface ----

    def n_clusters(self) -> int:
        return int(self.means_.shape[0])

    def centers(self) -> np.ndarray:
        return f32(self.means_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        lr = _log_resp(
            np.asarray(X, dtype=np.float64),
            np.asarray(self.weights_, dtype=np.float64),
            np.asarray(self.means_, dtype=np.float64),
            np.asarray(self.precisions_cholesky_, dtype=np.float64),
            self.covariance_type,
        )
        return np.argmax(lr, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lr = _log_resp(
            np.asarray(X, dtype=np.float64),
            np.asarray(self.weights_, dtype=np.float64),
            np.asarray(self.means_, dtype=np.float64),
            np.asarray(self.precisions_cholesky_, dtype=np.float64),
            self.covariance_type,
        )
        # log-sum-exp normalization
        log_norm = np.max(lr, axis=1, keepdims=True)
        lr_shifted = lr - log_norm
        resp = np.exp(lr_shifted)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp.astype(np.float32)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood: log p(x)."""
        lr = _log_resp(
            np.asarray(X, dtype=np.float64),
            np.asarray(self.weights_, dtype=np.float64),
            np.asarray(self.means_, dtype=np.float64),
            np.asarray(self.precisions_cholesky_, dtype=np.float64),
            self.covariance_type,
        )
        # log-sum-exp over components
        max_lr = np.max(lr, axis=1)
        return max_lr + np.log(np.sum(np.exp(lr - max_lr[:, None]), axis=1))

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood."""
        return float(np.mean(self.score_samples(X)))

    def bic(self, X: np.ndarray) -> float:
        N = X.shape[0]
        return -2.0 * float(np.sum(self.score_samples(X))) + self._n_parameters() * np.log(N)

    def aic(self, X: np.ndarray) -> float:
        return -2.0 * float(np.sum(self.score_samples(X))) + 2.0 * self._n_parameters()

    def _n_parameters(self) -> float:
        """Number of free parameters, matching sklearn convention."""
        K = self.n_clusters()
        D = self.means_.shape[1]
        mean_params = K * D
        weight_params = K - 1
        if self.covariance_type == "full":
            cov_params = K * D * (D + 1) / 2
        elif self.covariance_type == "diag":
            cov_params = K * D
        elif self.covariance_type == "tied":
            cov_params = D * (D + 1) / 2
        elif self.covariance_type == "spherical":
            cov_params = K
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
        return float(mean_params + weight_params + cov_params)

    # ---- Serialization ----

    def config(self) -> Dict[str, Any]:
        return {
            "kind": "gmm",
            "n_clusters": self.n_clusters(),
            "covariance_type": self.covariance_type,
            "reg_covar": float(self.reg_covar),
        }

    def state_arrays(self) -> Dict[str, np.ndarray]:
        # Keep precision matrices in float64 to avoid f32→f64 round-trip error in high dimensions.
        return {
            "weights": np.asarray(self.weights_, dtype=np.float64),
            "means": np.asarray(self.means_, dtype=np.float64),
            "covariances": np.asarray(self.covariances_, dtype=np.float64),
            "precisions_cholesky": np.asarray(self.precisions_cholesky_, dtype=np.float64),
        }

    @classmethod
    def from_state(
        cls, config: Dict[str, Any], arrays: Dict[str, np.ndarray]
    ) -> GMMModel:
        # Load in float64 so inference (predict, bic, etc.) does not lose precision.
        return cls(
            weights_=np.asarray(arrays["weights"], dtype=np.float64),
            means_=np.asarray(arrays["means"], dtype=np.float64),
            covariances_=np.asarray(arrays["covariances"], dtype=np.float64),
            precisions_cholesky_=np.asarray(arrays["precisions_cholesky"], dtype=np.float64),
            covariance_type=str(config.get("covariance_type", "diag")),
            reg_covar=float(config.get("reg_covar", 1e-6)),
        )


# =========================================================================
# Fitting (the ONLY place sklearn is used)
# =========================================================================

def _fit_gmm(
    X: np.ndarray,
    k: int,
    seed: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-2,
    n_init: int = 2,
    max_iter: int = 200,
) -> GMMModel:
    gm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    ).fit(X)
    # Store in float64 so serialization and inference preserve sklearn's precision.
    return GMMModel(
        weights_=np.asarray(gm.weights_, dtype=np.float64),
        means_=np.asarray(gm.means_, dtype=np.float64),
        covariances_=np.asarray(gm.covariances_, dtype=np.float64),
        precisions_cholesky_=np.asarray(gm.precisions_cholesky_, dtype=np.float64),
        covariance_type=covariance_type,
        reg_covar=reg_covar,
    )
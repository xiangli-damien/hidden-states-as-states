
"""GMMModel with pure-numpy inference — no sklearn runtime dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, kmeans_plusplus
from sklearn.mixture import GaussianMixture

try:
    from ..utils import f32
    from .base import ClusterModel
except ImportError:

    def f32(x):
        x = np.asarray(x)
        if x.dtype == np.float32 and x.flags['C_CONTIGUOUS']:
            return x
        return np.ascontiguousarray(x, dtype=np.float32)

    class ClusterModel:
        pass


_LOG2PI = float(np.log(2.0 * np.pi))
_DEFAULT_TOL = 1e-6
_DEFAULT_ADAPTIVE_ALPHA = 1e-2
_DEFAULT_ADAPTIVE_MIN = 1e-12
_DEFAULT_CHUNK_SIZE = 4096
_DEFAULT_AUTO_GPU_THRESHOLD = 50_000_000


def _log_det_cholesky(
    precisions_chol: np.ndarray, cov_type: str, n_features: int
) -> np.ndarray:
    """Compute log-det from Cholesky of precisions, matching sklearn's internal logic."""
    if cov_type == 'full':
        return np.array(
            [
                np.sum(np.log(np.diag(precisions_chol[k])))
                for k in range(precisions_chol.shape[0])
            ]
        )
    if cov_type == 'tied':
        return np.array([np.sum(np.log(np.diag(precisions_chol)))])
    if cov_type == 'diag':
        return np.sum(np.log(precisions_chol), axis=1)
    if cov_type == 'spherical':
        return n_features * np.log(precisions_chol)
    raise ValueError(f'Unknown covariance_type: {cov_type}')


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
    if cov_type == 'full':
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            y = diff @ precisions_chol[k]
            log_prob[:, k] = np.sum(y * y, axis=1)
    elif cov_type == 'tied':
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            y = diff @ precisions_chol
            log_prob[:, k] = np.sum(y * y, axis=1)
        log_det = np.full(K, log_det[0])
    elif cov_type == 'diag':
        precisions = precisions_chol**2
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            log_prob[:, k] = np.sum(diff * diff * precisions[k], axis=1)
    elif cov_type == 'spherical':
        precisions = precisions_chol**2
        log_prob = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            log_prob[:, k] = precisions[k] * np.sum(diff * diff, axis=1)
    else:
        raise ValueError(f'Unknown covariance_type: {cov_type}')
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
    return log_prob + _estimate_log_weights(weights)


@dataclass(frozen=True)
class GMMModel(ClusterModel):
    """Gaussian Mixture Model with pure-numpy inference.

    After construction, all inference (predict, predict_proba, score_samples,
    bic, aic) uses only numpy — no sklearn objects are created or stored.
    sklearn is used ONLY in the fitting helpers below.
    """

    weights_: np.ndarray
    means_: np.ndarray
    covariances_: np.ndarray
    precisions_cholesky_: np.ndarray
    covariance_type: str
    reg_covar: float

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
        max_lr = np.max(lr, axis=1)
        return max_lr + np.log(np.sum(np.exp(lr - max_lr[:, None]), axis=1))

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood."""
        return float(np.mean(self.score_samples(X)))

    def bic(self, X: np.ndarray) -> float:
        N = X.shape[0]
        return -2.0 * float(np.sum(self.score_samples(X))) + self._n_parameters() * np.log(
            N
        )

    def aic(self, X: np.ndarray) -> float:
        return -2.0 * float(np.sum(self.score_samples(X))) + 2.0 * self._n_parameters()

    def _n_parameters(self) -> float:
        """Number of free parameters, matching sklearn convention."""
        K = self.n_clusters()
        D = self.means_.shape[1]
        mean_params = K * D
        weight_params = K - 1
        if self.covariance_type == 'full':
            cov_params = K * D * (D + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = K * D
        elif self.covariance_type == 'tied':
            cov_params = D * (D + 1) / 2
        elif self.covariance_type == 'spherical':
            cov_params = K
        else:
            raise ValueError(f'Unknown covariance_type: {self.covariance_type}')
        return float(mean_params + weight_params + cov_params)

    def config(self) -> Dict[str, Any]:
        return {
            'kind': 'gmm',
            'n_clusters': self.n_clusters(),
            'covariance_type': self.covariance_type,
            'reg_covar': float(self.reg_covar),
        }

    def state_arrays(self) -> Dict[str, np.ndarray]:
        return {
            'weights': np.asarray(self.weights_, dtype=np.float64),
            'means': np.asarray(self.means_, dtype=np.float64),
            'covariances': np.asarray(self.covariances_, dtype=np.float64),
            'precisions_cholesky': np.asarray(
                self.precisions_cholesky_, dtype=np.float64
            ),
        }

    @classmethod
    def from_state(cls, config: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> "GMMModel":
        return cls(
            weights_=np.asarray(arrays['weights'], dtype=np.float64),
            means_=np.asarray(arrays['means'], dtype=np.float64),
            covariances_=np.asarray(arrays['covariances'], dtype=np.float64),
            precisions_cholesky_=np.asarray(
                arrays['precisions_cholesky'], dtype=np.float64
            ),
            covariance_type=str(config.get('covariance_type', 'diag')),
            reg_covar=float(config.get('reg_covar', 1e-6)),
        )


def _logsumexp_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    out = x_max + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _regularize_diag_covariances_np(
    raw_cov: np.ndarray,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
) -> np.ndarray:
    """Regularize diagonal covariances.

    Important implementation detail:
    the user-facing "adaptive floor" formula is interpreted as an *additive*
    scale-aware regularizer

        cov = raw_cov + max(alpha * raw_cov, adaptive_min)

    rather than `max(raw_cov, alpha * raw_cov)`, because the latter is a no-op
    for 0 < alpha < 1 and would not change EM at all.
    """
    cov = np.maximum(np.asarray(raw_cov, dtype=np.float64), 0.0)
    if adaptive_reg:
        reg = np.maximum(adaptive_alpha * cov, adaptive_min)
        cov = cov + reg
    else:
        cov = cov + float(reg_covar)
        np.maximum(cov, adaptive_min, out=cov)
    return cov


def _regularize_diag_covariances_torch(
    raw_cov,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
):
    import torch

    raw_cov = raw_cov.clamp_min(0.0)
    if adaptive_reg:
        reg = torch.clamp(raw_cov * adaptive_alpha, min=adaptive_min)
        return raw_cov + reg
    return torch.clamp(raw_cov + float(reg_covar), min=adaptive_min)


def _resolve_device(backend: str, device: str = 'auto') -> str:
    if device and device != 'auto':
        return str(device)
    if backend != 'gpu':
        return 'cpu'
    try:
        import torch  # noqa: F401
    except Exception:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_diag_model(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    reg_covar: float,
) -> GMMModel:
    covariances = np.asarray(covariances, dtype=np.float64)
    np.maximum(covariances, _DEFAULT_ADAPTIVE_MIN, out=covariances)
    precisions_cholesky = 1.0 / np.sqrt(covariances)
    return GMMModel(
        weights_=np.asarray(weights, dtype=np.float64),
        means_=np.asarray(means, dtype=np.float64),
        covariances_=covariances,
        precisions_cholesky_=np.asarray(precisions_cholesky, dtype=np.float64),
        covariance_type='diag',
        reg_covar=float(reg_covar),
    )


def _score_diag_parameters(
    X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, *, reg_covar: float
) -> float:
    model = _make_diag_model(weights, means, covariances, reg_covar=reg_covar)
    return model.score(np.asarray(X, dtype=np.float64))


def _squared_distance_np(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ centers.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _labels_from_centers_np(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = _squared_distance_np(X, centers)
    return np.argmin(d2, axis=1).astype(np.int32)


def _initial_stats_from_labels(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.RandomState,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, D = X.shape
    weights = np.zeros(k, dtype=np.float64)
    means = np.zeros((k, D), dtype=np.float64)
    raw_cov = np.zeros((k, D), dtype=np.float64)
    global_var = np.var(X, axis=0).astype(np.float64, copy=False)
    global_var = np.maximum(global_var, adaptive_min)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    empty = np.where(counts <= 0)[0]
    if empty.size > 0:
        replacement = rng.choice(N, size=empty.size, replace=N < empty.size)
        labels = labels.copy()
        labels[replacement] = empty
        counts = np.bincount(labels, minlength=k).astype(np.int64)
    for idx in range(k):
        mask = labels == idx
        n_k = int(mask.sum())
        if n_k <= 0:
            point = X[int(rng.randint(0, N))]
            means[idx] = point
            raw_cov[idx] = global_var
            weights[idx] = 1.0 / max(N, k)
            continue
        Xk = X[mask]
        weights[idx] = n_k / N
        means[idx] = Xk.mean(axis=0)
        centered = Xk - means[idx]
        raw_cov[idx] = np.mean(centered * centered, axis=0)
    covariances = _regularize_diag_covariances_np(
        raw_cov,
        adaptive_reg=adaptive_reg,
        reg_covar=reg_covar,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
    )
    weights_sum = np.sum(weights)
    if weights_sum <= 0:
        weights.fill(1.0 / k)
    else:
        weights /= weights_sum
    return weights, means, covariances


def _init_diag_from_kmeanspp(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    if k >= X.shape[0]:
        centers = X[rng.choice(X.shape[0], size=k, replace=True)]
    else:
        centers, _ = kmeans_plusplus(
            X,
            n_clusters=k,
            random_state=rng,
        )
    labels = _labels_from_centers_np(X, np.asarray(centers, dtype=np.float64))
    return _initial_stats_from_labels(
        X,
        labels,
        k,
        rng,
        adaptive_reg=adaptive_reg,
        reg_covar=reg_covar,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
    )


def _init_diag_from_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    km = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=min(max(256, 4 * int(k)), max(256, len(X))),
        n_init=1,
        max_iter=50,
        reassignment_ratio=0.0,
        random_state=int(seed),
    ).fit(X)
    labels = np.asarray(km.labels_, dtype=np.int32)
    return _initial_stats_from_labels(
        X,
        labels,
        k,
        rng,
        adaptive_reg=adaptive_reg,
        reg_covar=reg_covar,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
    )


def _init_diag_from_spectral(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
    spectral_affinity: str,
    spectral_n_neighbors: int,
    spectral_gamma: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    if X.shape[0] <= 2 or k <= 1:
        return _init_diag_from_kmeanspp(
            X,
            k,
            seed,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
        )
    kwargs = dict(
        n_clusters=int(k),
        random_state=int(seed),
        assign_labels='kmeans',
        affinity=str(spectral_affinity),
    )
    if str(spectral_affinity) == 'nearest_neighbors':
        kwargs['n_neighbors'] = int(max(2, min(int(spectral_n_neighbors), X.shape[0] - 1)))
    elif str(spectral_affinity) == 'rbf' and spectral_gamma is not None:
        kwargs['gamma'] = float(spectral_gamma)
    try:
        labels = SpectralClustering(**kwargs).fit_predict(X).astype(np.int32, copy=False)
    except Exception as exc:
        warnings.warn(
            f'spectral initialization failed with {exc!r}; falling back to kmeans++',
            RuntimeWarning,
            stacklevel=2,
        )
        return _init_diag_from_kmeanspp(
            X,
            k,
            seed,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
        )
    return _initial_stats_from_labels(
        X,
        labels,
        k,
        rng,
        adaptive_reg=adaptive_reg,
        reg_covar=reg_covar,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
    )


def _init_diag_parameters(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    init_method: str,
    adaptive_reg: bool,
    reg_covar: float,
    adaptive_alpha: float,
    adaptive_min: float,
    spectral_affinity: str,
    spectral_n_neighbors: int,
    spectral_gamma: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = str(init_method or 'kmeans++').strip().lower()
    if method in {'kmeans++', 'kmeans_plus_plus', 'kmeanspp'}:
        return _init_diag_from_kmeanspp(
            X,
            k,
            seed,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
        )
    if method in {'kmeans', 'mini_batch_kmeans', 'minibatch_kmeans'}:
        return _init_diag_from_kmeans(
            X,
            k,
            seed,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
        )
    if method in {'random', 'rand'}:
        rng = np.random.RandomState(int(seed))
        labels = rng.randint(0, int(k), size=X.shape[0], dtype=np.int32)
        return _initial_stats_from_labels(
            X,
            labels,
            k,
            rng,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
        )
    if method in {'spectral', 'spectral_gmm', 'spectral_init'}:
        return _init_diag_from_spectral(
            X,
            k,
            seed,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            spectral_gamma=spectral_gamma,
        )
    raise ValueError(
        f'Unknown GMM init_method={init_method!r}. Expected one of '
        '"kmeans++", "kmeans", "random", "spectral".'
    )


def _diag_sufficient_statistics_np(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    N, D = X.shape
    K = means.shape[0]
    inv_cov = 1.0 / covariances
    weighted_means = means * inv_cov
    mean_quad = np.sum(means * weighted_means, axis=1)
    log_cov_sum = np.sum(np.log(covariances), axis=1)
    log_weights = np.log(np.clip(weights, 1e-12, None))
    nk = np.zeros(K, dtype=np.float64)
    sum_x = np.zeros((K, D), dtype=np.float64)
    sum_x2 = np.zeros((K, D), dtype=np.float64)
    total_log_prob = 0.0
    block = max(1, int(chunk_size))
    for start in range(0, N, block):
        stop = min(N, start + block)
        Xb = X[start:stop]
        X2b = Xb * Xb
        maha = X2b @ inv_cov.T
        maha -= 2.0 * (Xb @ weighted_means.T)
        maha += mean_quad[None, :]
        log_prob = -0.5 * (D * _LOG2PI + log_cov_sum[None, :] + maha)
        log_prob_weighted = log_prob + log_weights[None, :]
        log_norm = _logsumexp_np(log_prob_weighted, axis=1)
        resp = np.exp(log_prob_weighted - log_norm[:, None])
        nk += resp.sum(axis=0)
        sum_x += resp.T @ Xb
        sum_x2 += resp.T @ X2b
        total_log_prob += float(np.sum(log_norm))
    mean_log_prob = total_log_prob / max(N, 1)
    return nk, sum_x, sum_x2, mean_log_prob


def _fit_gmm_adaptive(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    covariance_type: str = 'diag',
    reg_covar: float = 0.01,
    n_init: int = 2,
    max_iter: int = 200,
    tol: float = _DEFAULT_TOL,
    adaptive_reg: bool = True,
    adaptive_alpha: float = _DEFAULT_ADAPTIVE_ALPHA,
    adaptive_min: float = _DEFAULT_ADAPTIVE_MIN,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    init_method: str = 'kmeans++',
    spectral_affinity: str = 'nearest_neighbors',
    spectral_n_neighbors: int = 10,
    spectral_gamma: Optional[float] = None,
) -> GMMModel:
    """Pure-numpy diagonal-covariance EM with adaptive additive regularization."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f'Expected 2-D array, got shape {X.shape}')
    N, D = X.shape
    if N <= 0:
        raise ValueError('Cannot fit GMM on an empty array')
    if int(k) < 1:
        raise ValueError('k must be >= 1')
    if int(k) > N:
        raise ValueError(f'k={k} exceeds number of samples N={N}')
    if str(covariance_type).lower() != 'diag':
        warnings.warn(
            f'Adaptive EM only supports covariance_type="diag"; '
            f'falling back to sklearn for covariance_type={covariance_type!r}.',
            RuntimeWarning,
            stacklevel=2,
        )
        return _fit_gmm_sklearn(
            X,
            k,
            seed,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
        )

    best_model: Optional[GMMModel] = None
    best_score = float('-inf')
    global_var = np.maximum(np.var(X, axis=0), adaptive_min).astype(np.float64, copy=False)

    for init_idx in range(max(1, int(n_init))):
        init_seed = int(seed) + init_idx * 10007
        rng = np.random.RandomState(init_seed)
        weights, means, covariances = _init_diag_parameters(
            X,
            int(k),
            init_seed,
            init_method=init_method,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            spectral_gamma=spectral_gamma,
        )
        prev_score: Optional[float] = None
        converged = False

        for _ in range(max(1, int(max_iter))):
            nk, sum_x, sum_x2, score = _diag_sufficient_statistics_np(
                X,
                weights,
                means,
                covariances,
                chunk_size=chunk_size,
            )

            dead = nk <= 1e-8
            if np.any(dead):
                dead_idx = np.flatnonzero(dead)
                replacement = rng.choice(N, size=dead_idx.size, replace=N < dead_idx.size)
                repl_x = X[replacement]
                nk[dead_idx] = 1.0
                sum_x[dead_idx] = repl_x
                sum_x2[dead_idx] = repl_x * repl_x + global_var[None, :]

            weights = nk / np.sum(nk)
            means = sum_x / nk[:, None]
            raw_cov = sum_x2 / nk[:, None] - means * means
            np.maximum(raw_cov, 0.0, out=raw_cov)
            covariances = _regularize_diag_covariances_np(
                raw_cov,
                adaptive_reg=adaptive_reg,
                reg_covar=reg_covar,
                adaptive_alpha=adaptive_alpha,
                adaptive_min=adaptive_min,
            )

            if prev_score is not None and abs(score - prev_score) <= float(tol):
                converged = True
                break
            prev_score = score

        model = _make_diag_model(
            weights,
            means,
            covariances,
            reg_covar=0.0 if adaptive_reg else reg_covar,
        )
        final_score = model.score(X)
        if final_score > best_score or best_model is None:
            best_score = final_score
            best_model = model

        if converged and best_score == final_score:
            # Keep deterministic best-model tie-breaking while still evaluating all
            # restarts when requested.
            pass

    if best_model is None:
        raise RuntimeError('Adaptive diagonal GMM failed to produce a model')
    return best_model


def _fit_gmm_gpu(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    covariance_type: str = 'diag',
    reg_covar: float = 0.01,
    n_init: int = 2,
    max_iter: int = 200,
    tol: float = _DEFAULT_TOL,
    adaptive_reg: bool = True,
    adaptive_alpha: float = _DEFAULT_ADAPTIVE_ALPHA,
    adaptive_min: float = _DEFAULT_ADAPTIVE_MIN,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    init_method: str = 'kmeans++',
    spectral_affinity: str = 'nearest_neighbors',
    spectral_n_neighbors: int = 10,
    spectral_gamma: Optional[float] = None,
    device: str = 'auto',
) -> GMMModel:
    """PyTorch-accelerated diagonal-covariance EM.

    If CUDA is unavailable, this routine falls back to CPU PyTorch.
    """
    if str(covariance_type).lower() != 'diag':
        warnings.warn(
            f'GPU EM only supports covariance_type="diag"; '
            f'falling back to sklearn for covariance_type={covariance_type!r}.',
            RuntimeWarning,
            stacklevel=2,
        )
        return _fit_gmm_sklearn(
            X,
            k,
            seed,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
        )
    try:
        import torch
    except Exception as exc:
        warnings.warn(
            f'PyTorch is unavailable ({exc!r}); falling back to CPU adaptive EM.',
            RuntimeWarning,
            stacklevel=2,
        )
        return _fit_gmm_adaptive(
            X,
            k,
            seed,
            covariance_type='diag',
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            adaptive_reg=adaptive_reg,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
            chunk_size=chunk_size,
            init_method=init_method,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            spectral_gamma=spectral_gamma,
        )

    resolved_device = _resolve_device('gpu', device=device)
    torch_device = torch.device(resolved_device)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    X_np = np.asarray(X, dtype=np.float32, order='C')
    if X_np.ndim != 2:
        raise ValueError(f'Expected 2-D array, got shape {X_np.shape}')
    N, D = X_np.shape
    if int(k) > N:
        raise ValueError(f'k={k} exceeds number of samples N={N}')
    X_t = torch.as_tensor(X_np, dtype=torch.float32, device=torch_device)
    global_var = torch.clamp(X_t.var(dim=0, unbiased=False), min=float(adaptive_min))
    best_model: Optional[GMMModel] = None
    best_score = float('-inf')
    block = max(1, int(chunk_size))

    for init_idx in range(max(1, int(n_init))):
        init_seed = int(seed) + init_idx * 10007
        init_weights_np, init_means_np, init_cov_np = _init_diag_parameters(
            np.asarray(X, dtype=np.float64),
            int(k),
            init_seed,
            init_method=init_method,
            adaptive_reg=adaptive_reg,
            reg_covar=reg_covar,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            spectral_gamma=spectral_gamma,
        )
        weights_t = torch.as_tensor(init_weights_np, dtype=torch.float32, device=torch_device)
        means_t = torch.as_tensor(init_means_np, dtype=torch.float32, device=torch_device)
        cov_t = torch.as_tensor(init_cov_np, dtype=torch.float32, device=torch_device)
        prev_score: Optional[float] = None

        for _ in range(max(1, int(max_iter))):
            inv_cov = 1.0 / cov_t
            weighted_means = means_t * inv_cov
            mean_quad = torch.sum(means_t * weighted_means, dim=1)
            log_cov_sum = torch.sum(torch.log(cov_t), dim=1)
            log_weights = torch.log(torch.clamp(weights_t, min=1e-12))
            nk = torch.zeros(int(k), dtype=torch.float32, device=torch_device)
            sum_x = torch.zeros((int(k), D), dtype=torch.float32, device=torch_device)
            sum_x2 = torch.zeros((int(k), D), dtype=torch.float32, device=torch_device)
            total_log_prob = torch.zeros((), dtype=torch.float32, device=torch_device)

            for start in range(0, N, block):
                stop = min(N, start + block)
                Xb = X_t[start:stop]
                X2b = Xb * Xb
                maha = X2b @ inv_cov.T
                maha = maha - 2.0 * (Xb @ weighted_means.T)
                maha = maha + mean_quad.unsqueeze(0)
                log_prob = -0.5 * (D * _LOG2PI + log_cov_sum.unsqueeze(0) + maha)
                log_prob_weighted = log_prob + log_weights.unsqueeze(0)
                log_norm = torch.logsumexp(log_prob_weighted, dim=1)
                resp = torch.exp(log_prob_weighted - log_norm.unsqueeze(1))
                nk = nk + resp.sum(dim=0)
                sum_x = sum_x + resp.T @ Xb
                sum_x2 = sum_x2 + resp.T @ X2b
                total_log_prob = total_log_prob + log_norm.sum()

            dead = nk <= 1e-8
            if bool(dead.any()):
                dead_idx = torch.nonzero(dead, as_tuple=False).flatten()
                cpu_gen = torch.Generator(device='cpu')
                cpu_gen.manual_seed(init_seed + 97)
                replacement_cpu = torch.randint(
                    low=0,
                    high=N,
                    size=(int(dead_idx.numel()),),
                    generator=cpu_gen,
                    device='cpu',
                )
                replacement = replacement_cpu.to(torch_device)
                repl_x = X_t[replacement]
                nk[dead_idx] = 1.0
                sum_x[dead_idx] = repl_x
                sum_x2[dead_idx] = repl_x * repl_x + global_var.unsqueeze(0)

            weights_t = nk / nk.sum()
            means_t = sum_x / nk.unsqueeze(1)
            raw_cov = sum_x2 / nk.unsqueeze(1) - means_t * means_t
            cov_t = _regularize_diag_covariances_torch(
                raw_cov,
                adaptive_reg=adaptive_reg,
                reg_covar=reg_covar,
                adaptive_alpha=adaptive_alpha,
                adaptive_min=adaptive_min,
            )

            score = float((total_log_prob / max(N, 1)).detach().cpu().item())
            if prev_score is not None and abs(score - prev_score) <= float(tol):
                break
            prev_score = score

        model = _make_diag_model(
            weights_t.detach().cpu().numpy().astype(np.float64, copy=False),
            means_t.detach().cpu().numpy().astype(np.float64, copy=False),
            cov_t.detach().cpu().numpy().astype(np.float64, copy=False),
            reg_covar=0.0 if adaptive_reg else reg_covar,
        )
        final_score = model.score(np.asarray(X, dtype=np.float64))
        if final_score > best_score or best_model is None:
            best_score = final_score
            best_model = model

    if best_model is None:
        raise RuntimeError('GPU diagonal GMM failed to produce a model')
    return best_model


def _fit_gmm_sklearn(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    covariance_type: str = 'diag',
    reg_covar: float = 0.01,
    n_init: int = 2,
    max_iter: int = 200,
) -> GMMModel:
    gm = GaussianMixture(
        n_components=int(k),
        covariance_type=str(covariance_type),
        reg_covar=float(reg_covar),
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(np.asarray(X, dtype=np.float64))
    return GMMModel(
        weights_=np.asarray(gm.weights_, dtype=np.float64),
        means_=np.asarray(gm.means_, dtype=np.float64),
        covariances_=np.asarray(gm.covariances_, dtype=np.float64),
        precisions_cholesky_=np.asarray(gm.precisions_cholesky_, dtype=np.float64),
        covariance_type=str(covariance_type),
        reg_covar=float(reg_covar),
    )


def _fit_gmm_spectral(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    covariance_type: str = 'diag',
    reg_covar: float = 0.01,
    n_init: int = 2,
    max_iter: int = 200,
    tol: float = _DEFAULT_TOL,
    adaptive_reg: bool = True,
    adaptive_alpha: float = _DEFAULT_ADAPTIVE_ALPHA,
    adaptive_min: float = _DEFAULT_ADAPTIVE_MIN,
    backend: str = 'auto',
    auto_gpu_threshold: int = _DEFAULT_AUTO_GPU_THRESHOLD,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    device: str = 'auto',
    spectral_affinity: str = 'nearest_neighbors',
    spectral_n_neighbors: int = 10,
    spectral_gamma: Optional[float] = None,
) -> GMMModel:
    """Convenience wrapper: spectral initialization + standard GMM EM."""
    return _fit_gmm(
        X,
        k,
        seed,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        adaptive_reg=adaptive_reg,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
        backend=backend,
        auto_gpu_threshold=auto_gpu_threshold,
        chunk_size=chunk_size,
        device=device,
        init_method='spectral',
        spectral_affinity=spectral_affinity,
        spectral_n_neighbors=spectral_n_neighbors,
        spectral_gamma=spectral_gamma,
    )


def _choose_gmm_backend(
    X: np.ndarray,
    k: int,
    *,
    covariance_type: str,
    adaptive_reg: bool,
    backend: str,
    auto_gpu_threshold: int,
) -> str:
    requested = str(backend or 'auto').strip().lower()
    if requested not in {'auto', 'sklearn', 'cpu', 'gpu'}:
        raise ValueError(f'Unknown GMM backend={backend!r}')
    if requested == 'gpu' and str(covariance_type).lower() != 'diag':
        raise ValueError('The GPU GMM implementation supports diagonal covariance only')
    if str(covariance_type).lower() == 'diag' and requested in {'cpu', 'gpu'}:
        # Both custom kernels implement fixed as well as adaptive regularization.
        # An explicit GPU request must not silently become sklearn on the CPU.
        return requested
    if not adaptive_reg:
        return 'sklearn'
    if str(covariance_type).lower() != 'diag':
        return 'sklearn'
    if requested in {'sklearn', 'cpu', 'gpu'}:
        return requested
    if requested != 'auto':
        raise ValueError(
            f'Unknown GMM backend={backend!r}. Expected one of '
            '"auto", "sklearn", "cpu", "gpu".'
        )
    work = int(np.prod(np.asarray(X.shape, dtype=np.int64))) * int(k)
    try:
        import torch
    except Exception:
        return 'cpu'
    if torch.cuda.is_available() and work >= int(max(1, auto_gpu_threshold)):
        return 'gpu'
    return 'cpu'


def _gmm_kwargs_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse common GMM fit kwargs from ClusterSpec.params."""
    payload = dict(params or {})
    return {
        'covariance_type': str(payload.get('covariance_type', 'diag')),
        'reg_covar': float(payload.get('reg_covar', 0.01)),
        'n_init': int(payload.get('n_init', 2)),
        'max_iter': int(payload.get('max_iter', 200)),
        'tol': float(payload.get('tol', _DEFAULT_TOL)),
        'adaptive_reg': bool(payload.get('adaptive_reg', True)),
        'adaptive_alpha': float(
            payload.get('adaptive_alpha', _DEFAULT_ADAPTIVE_ALPHA)
        ),
        'adaptive_min': float(payload.get('adaptive_min', _DEFAULT_ADAPTIVE_MIN)),
        'backend': str(payload.get('backend', 'auto')),
        'auto_gpu_threshold': int(
            payload.get('auto_gpu_threshold', _DEFAULT_AUTO_GPU_THRESHOLD)
        ),
        'chunk_size': int(payload.get('chunk_size', _DEFAULT_CHUNK_SIZE)),
        'device': str(payload.get('device', 'auto')),
        'init_method': str(payload.get('init_method', 'kmeans++')),
        'spectral_affinity': str(
            payload.get('spectral_affinity', 'nearest_neighbors')
        ),
        'spectral_n_neighbors': int(payload.get('spectral_n_neighbors', 10)),
        'spectral_gamma': None
        if payload.get('spectral_gamma') is None
        else float(payload.get('spectral_gamma')),
    }


def _fit_gmm(
    X: np.ndarray,
    k: int,
    seed: int,
    *,
    covariance_type: str = 'diag',
    reg_covar: float = 0.01,
    n_init: int = 2,
    max_iter: int = 200,
    tol: float = _DEFAULT_TOL,
    adaptive_reg: bool = True,
    adaptive_alpha: float = _DEFAULT_ADAPTIVE_ALPHA,
    adaptive_min: float = _DEFAULT_ADAPTIVE_MIN,
    backend: str = 'auto',
    auto_gpu_threshold: int = _DEFAULT_AUTO_GPU_THRESHOLD,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    device: str = 'auto',
    init_method: str = 'kmeans++',
    spectral_affinity: str = 'nearest_neighbors',
    spectral_n_neighbors: int = 10,
    spectral_gamma: Optional[float] = None,
) -> GMMModel:
    """Unified GMM entry point.

    backend:
      - 'sklearn': original sklearn EM with fixed reg_covar
      - 'cpu': custom adaptive diagonal EM in numpy
      - 'gpu': custom adaptive diagonal EM in PyTorch
      - 'auto': use GPU when available and the problem is large enough,
                otherwise use CPU; always fall back to sklearn when
                adaptive_reg=False or covariance_type!='diag'
    """
    selected_backend = _choose_gmm_backend(
        np.asarray(X),
        int(k),
        covariance_type=str(covariance_type),
        adaptive_reg=bool(adaptive_reg),
        backend=str(backend),
        auto_gpu_threshold=int(auto_gpu_threshold),
    )
    if selected_backend == 'sklearn':
        if adaptive_reg and str(covariance_type).lower() != 'diag':
            warnings.warn(
                f'Adaptive GMM currently supports only diag covariance; '
                f'falling back to sklearn for covariance_type={covariance_type!r}.',
                RuntimeWarning,
                stacklevel=2,
            )
        return _fit_gmm_sklearn(
            X,
            k,
            seed,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
        )
    if selected_backend == 'gpu':
        return _fit_gmm_gpu(
            X,
            k,
            seed,
            covariance_type='diag',
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            adaptive_reg=adaptive_reg,
            adaptive_alpha=adaptive_alpha,
            adaptive_min=adaptive_min,
            chunk_size=chunk_size,
            init_method=init_method,
            spectral_affinity=spectral_affinity,
            spectral_n_neighbors=spectral_n_neighbors,
            spectral_gamma=spectral_gamma,
            device=device,
        )
    return _fit_gmm_adaptive(
        X,
        k,
        seed,
        covariance_type='diag',
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        adaptive_reg=adaptive_reg,
        adaptive_alpha=adaptive_alpha,
        adaptive_min=adaptive_min,
        chunk_size=chunk_size,
        init_method=init_method,
        spectral_affinity=spectral_affinity,
        spectral_n_neighbors=spectral_n_neighbors,
        spectral_gamma=spectral_gamma,
    )

"""Mixture of factor analyzers: Sigma_k = W_k W_k.T + diag(psi_k).

Exact latent-variable EM with component-specific diagonal noise. Both likelihood
and sufficient statistics use the Woodbury identity; no D x D covariance is
formed. CPU and optional CUDA use the same chunked updates. See Ghahramani and
Hinton, CRG-TR-96-1 (1996/1997). Parameters are persisted as ordinary arrays.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .base import ClusterModel


@dataclass
class MFAModel(ClusterModel):
    weights_: np.ndarray
    means_: np.ndarray
    loadings_: np.ndarray
    noise_: np.ndarray
    reg_covar: float = 1e-6
    history_: list = field(default_factory=list)
    converged_: bool = False
    backend_: str = "cpu"
    chunk_size: int = 1024

    def n_clusters(self):
        return len(self.weights_)

    def centers(self):
        return self.means_.astype(np.float32)

    def _log_joint(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.means_.shape[1]:
            raise ValueError("MFA feature dimension mismatch")
        out = np.empty((len(X), self.n_clusters()))
        for k, (mu, W, psi) in enumerate(zip(self.means_, self.loadings_, self.noise_)):
            inv = 1.0 / psi
            M = np.eye(W.shape[1]) + W.T @ (inv[:, None] * W)
            chol = np.linalg.cholesky(M)
            logdet = np.log(psi).sum() + 2 * np.log(chol.diagonal()).sum()
            for start in range(0, len(X), self.chunk_size):
                delta = X[start : start + self.chunk_size] - mu
                low = (delta * inv) @ W
                solved = np.linalg.solve(chol, low.T)
                mahal = (delta * delta * inv).sum(1) - (solved * solved).sum(0)
                out[start : start + len(delta), k] = np.log(self.weights_[k]) - 0.5 * (
                    X.shape[1] * np.log(2 * np.pi) + logdet + mahal
                )
        return out

    def predict(self, X):
        return self._log_joint(X).argmax(1).astype(np.int32)

    def predict_proba(self, X):
        log = self._log_joint(X)
        return np.exp(log - logsumexp(log, axis=1, keepdims=True))

    def score_samples(self, X):
        return logsumexp(self._log_joint(X), axis=1)

    def score(self, X):
        return float(self.score_samples(X).mean())

    def _n_parameters(self):
        k, d, q = self.loadings_.shape
        # Rotations of W leave W W.T unchanged: subtract q(q-1)/2.
        return k - 1 + k * (2 * d + d * q - q * (q - 1) // 2)

    def bic(self, X):
        return float(
            -2 * self.score_samples(X).sum() + self._n_parameters() * np.log(len(X))
        )

    def aic(self, X):
        return float(-2 * self.score_samples(X).sum() + 2 * self._n_parameters())

    def config(self):
        return dict(
            kind="mfa",
            n_clusters=self.n_clusters(),
            rank=self.loadings_.shape[2],
            reg_covar=self.reg_covar,
            converged=self.converged_,
            backend=self.backend_,
            chunk_size=self.chunk_size,
            history=self.history_,
        )

    def state_arrays(self):
        return dict(
            weights=self.weights_,
            means=self.means_,
            loadings=self.loadings_,
            noise=self.noise_,
        )

    @classmethod
    def from_state(cls, config: dict[str, Any], arrays):
        return cls(
            *(
                np.asarray(arrays[k], dtype=np.float64)
                for k in ("weights", "means", "loadings", "noise")
            ),
            reg_covar=config["reg_covar"],
            history_=config.get("history", []),
            converged_=config.get("converged", False),
            backend_=config.get("backend", "cpu"),
            chunk_size=config.get("chunk_size", 1024),
        )


class _Kernel:
    def __init__(self, backend, device):
        self.gpu = backend == "gpu"
        if self.gpu:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "MFA backend=gpu requires CUDA; no silent CPU fallback"
                )
            self.xp, self.device = torch, device if device != "auto" else "cuda:0"
        else:
            self.xp, self.device = np, None

    def array(self, x):
        if self.gpu:
            return self.xp.as_tensor(x, dtype=self.xp.float64, device=self.device)
        return np.asarray(x, dtype=np.float64)

    def zeros(self, shape):
        return self.array(np.zeros(shape))

    def eye(self, n):
        return self.array(np.eye(n))

    def numpy(self, x):
        return x.detach().cpu().numpy() if self.gpu else np.asarray(x)

    def lse(self, x):
        return self.xp.logsumexp(x, dim=1) if self.gpu else logsumexp(x, axis=1)


def fit_mfa(
    X,
    k,
    seed=42,
    *,
    rank=8,
    reg_covar=1e-6,
    n_init=2,
    max_iter=200,
    tol=1e-5,
    chunk_size=1024,
    backend="cpu",
    device="auto",
):
    """Fit without storing N x K x D tensors or a full feature covariance.

    rank=0 is a diagonal Gaussian mixture. CUDA streams input batches from host;
    only parameters and the current batch need reside on the device.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError("MFA backend must be explicitly cpu or gpu")
    X = np.asarray(X)
    if X.ndim != 2 or not 1 <= k <= len(X) or not 0 <= rank < X.shape[1]:
        raise ValueError("Require 1 <= k <= N and 0 <= rank < D")
    if min(n_init, max_iter, chunk_size) < 1 or reg_covar <= 0 or tol < 0:
        raise ValueError("Invalid MFA iteration, batching, or regularization setting")
    for start in range(0, len(X), chunk_size):
        if not np.isfinite(X[start : start + chunk_size]).all():
            raise ValueError("MFA requires finite input")
    kernel = _Kernel(backend, device)
    xp = kernel.xp
    n, d = X.shape
    best, best_score = None, -np.inf
    for restart in range(n_init):
        rng = np.random.default_rng(seed + restart)
        km = KMeans(n_clusters=k, n_init=1, random_state=seed + restart).fit(X)
        means = np.asarray(km.cluster_centers_, dtype=np.float64)
        variance = np.maximum(np.var(X, axis=0, dtype=np.float64), reg_covar)
        noise = np.stack(
            [
                np.maximum(
                    np.var(X[km.labels_ == j], axis=0, dtype=np.float64), reg_covar
                )
                if (km.labels_ == j).sum() > 1
                else variance
                for j in range(k)
            ]
        )
        loadings = (
            rng.normal(size=(k, d, rank))
            * np.sqrt(noise[:, :, None] / max(rank, 1))
            * 0.1
        )
        weights = np.bincount(km.labels_, minlength=k).clip(1) / n
        w, mu, W, psi = map(kernel.array, (weights, means, loadings, noise))
        history, converged = [], False
        for iteration in range(max_iter):
            inv = 1 / psi
            posterior, covz, logdet = [], [], []
            for j in range(k):
                M = kernel.eye(rank) + W[j].T @ (inv[j, :, None] * W[j])
                chol = xp.linalg.cholesky(M)
                C = xp.linalg.solve(M, kernel.eye(rank))
                covz.append(C)
                posterior.append(C @ (W[j].T * inv[j]))
                logdet.append(xp.log(psi[j]).sum() + 2 * xp.log(chol.diagonal()).sum())
            nk = kernel.zeros(k)
            R = kernel.zeros((k, rank + 1, rank + 1))
            T = kernel.zeros((k, d, rank + 1))
            xx = kernel.zeros((k, d))
            total_ll = kernel.zeros(())
            for start in range(0, n, chunk_size):
                batch = kernel.array(X[start : start + chunk_size])
                log = kernel.zeros((len(batch), k))
                for j in range(k):
                    delta = batch - mu[j]
                    ez = delta @ posterior[j].T
                    low = (delta * inv[j]) @ W[j]
                    mahal = (delta * delta * inv[j]).sum(axis=1) - (ez * low).sum(
                        axis=1
                    )
                    log[:, j] = xp.log(w[j]) - 0.5 * (
                        d * np.log(2 * np.pi) + logdet[j] + mahal
                    )
                normalizer = kernel.lse(log)
                resp = xp.exp(log - normalizer[:, None])
                total_ll = total_ll + normalizer.sum()
                for j in range(k):
                    r = resp[:, j]
                    count = r.sum()
                    ez = (batch - mu[j]) @ posterior[j].T
                    rz = r[:, None] * ez
                    nk[j] += count
                    R[j, 0, 0] += count
                    R[j, 0, 1:] += rz.sum(axis=0)
                    R[j, 1:, 0] += rz.sum(axis=0)
                    R[j, 1:, 1:] += ez.T @ rz + count * covz[j]
                    T[j, :, 0] += (r[:, None] * batch).sum(axis=0)
                    T[j, :, 1:] += batch.T @ rz
                    xx[j] += (r[:, None] * batch * batch).sum(axis=0)
            ll = float(kernel.numpy(total_ll)) / n
            if not np.isfinite(ll):
                raise FloatingPointError("Nonfinite MFA likelihood")
            history.append(ll)
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                converged = True
                break
            # Keep returned parameters paired with the likelihood actually measured.
            if iteration == max_iter - 1:
                break
            counts = kernel.numpy(nk)
            if (counts < 1e-8).any():
                raise FloatingPointError(
                    "Collapsed MFA component; reduce k or increase regularization"
                )
            for j in range(k):
                A = xp.linalg.solve(R[j], T[j].T).T
                mu[j] = A[:, 0]
                W[j] = A[:, 1:]
                psi[j] = ((xx[j] - (A * T[j]).sum(axis=1)) / nk[j]).clip(reg_covar)
            w = nk / n
        model = MFAModel(
            *(kernel.numpy(v).copy() for v in (w, mu, W, psi)),
            reg_covar=reg_covar,
            history_=history,
            converged_=converged,
            backend_=backend,
            chunk_size=chunk_size,
        )
        if history[-1] > best_score:
            best, best_score = model, history[-1]
    return best

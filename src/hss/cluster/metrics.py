from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from .gmm import GMMModel


def silhouette_sampled(
    X: np.ndarray,
    labels: np.ndarray,
    seed: int,
    max_n: int = 5000,
) -> float:
    n = len(labels)
    if len(np.unique(labels)) < 2:
        return float("-inf")
    if n > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_n, replace=False)
        return float(silhouette_score(X[idx], labels[idx], metric="euclidean"))
    return float(silhouette_score(X, labels, metric="euclidean"))


def compute_icl(model: Any, X: np.ndarray, mode: str) -> float:
    if not isinstance(model, GMMModel):
        return float("nan")
    bic = model.bic(X)
    tau = model.predict_proba(X)
    tau = np.clip(tau, 1e-12, None)
    ent = float(-np.sum(tau * np.log(tau)))
    if not np.isfinite(ent):
        return float("inf")
    if mode == "bic_plus_2entropy":
        return bic + 2.0 * ent
    if mode == "bic_minus_2entropy":
        return bic - 2.0 * ent
    return bic + 2.0 * ent
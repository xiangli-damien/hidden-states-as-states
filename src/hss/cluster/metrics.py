
from __future__ import annotations

from typing import Any
import warnings

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
        return float('-inf')
    if n > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_n, replace=False)
        return float(silhouette_score(X[idx], labels[idx], metric='euclidean'))
    return float(silhouette_score(X, labels, metric='euclidean'))


def compute_icl(model: Any, X: np.ndarray, mode: str) -> float:
    if not isinstance(model, GMMModel):
        return float('nan')
    bic = model.bic(X)
    tau = model.predict_proba(X)
    tau = np.clip(tau, 1e-12, None)
    ent = float(-np.sum(tau * np.log(tau)))
    if not np.isfinite(ent):
        return float('inf')

    normalized = str(mode or 'bic_plus_2entropy').strip().lower()
    if normalized in {'bic_plus_2entropy', 'icl', 'default'}:
        return bic + 2.0 * ent
    if normalized in {'bic_minus_2entropy', 'legacy_bic_minus_2entropy'}:
        warnings.warn(
            'bic_minus_2entropy is the legacy sign-flipped ICL. '
            'The default and paper-consistent setting is bic_plus_2entropy.',
            RuntimeWarning,
            stacklevel=2,
        )
        return bic - 2.0 * ent
    raise ValueError(
        f'Unknown ICL mode={mode!r}. Expected "bic_plus_2entropy" '
        f'or "bic_minus_2entropy".'
    )

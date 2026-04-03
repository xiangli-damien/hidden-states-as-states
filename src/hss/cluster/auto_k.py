
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .gmm import GMMModel, _fit_gmm, _gmm_kwargs_from_params
from .kmeans import KMeansModel, _fit_kmeans
from .metrics import compute_icl, silhouette_sampled


def select_k_kmeans(
    X: np.ndarray,
    *,
    k_range: Tuple[int, int],
    seed: int,
    tolerance: float = 0.05,
    params: Dict[str, Any],
) -> KMeansModel:
    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    k_hi = min(k_hi, len(X) - 1)
    k_lo = max(2, k_lo)
    candidates: List[Tuple[float, int, KMeansModel]] = []
    for k in range(k_lo, k_hi + 1):
        model = _fit_kmeans(
            X,
            k,
            seed,
            batch_size=int(params.get('batch_size', 1024)),
            n_init=int(params.get('n_init', 10)),
            max_iter=int(params.get('max_iter', 100)),
        )
        labels = model.predict(X)
        sil = silhouette_sampled(X, labels, seed)
        candidates.append((sil, k, model))
    if not candidates:
        raise ValueError('No valid k found')
    best_sil = max(c[0] for c in candidates)
    threshold = best_sil - tolerance * abs(best_sil)
    parsimonious = [c for c in candidates if c[0] >= threshold]
    parsimonious.sort(key=lambda c: c[1])
    return parsimonious[0][2]


def select_k_gmm(
    X: np.ndarray,
    *,
    k_range: Tuple[int, int],
    seed: int,
    tolerance: float = 0.05,
    icl_mode: str = 'bic_plus_2entropy',
    params: Dict[str, Any],
) -> GMMModel:
    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    k_hi = min(k_hi, len(X) - 1)
    k_lo = max(2, k_lo)
    fit_kwargs = _gmm_kwargs_from_params(params)
    candidates: List[Tuple[float, int, GMMModel]] = []
    for k in range(k_lo, k_hi + 1):
        try:
            model = _fit_gmm(X, k, seed, **fit_kwargs)
            icl = compute_icl(model, X, icl_mode)
            candidates.append((icl, k, model))
        except Exception:
            continue
    if not candidates:
        raise ValueError('No valid k found for GMM')
    best_icl = min(c[0] for c in candidates)
    abs_best = abs(best_icl) if abs(best_icl) > 1e-12 else 1.0
    threshold = best_icl + tolerance * abs_best
    parsimonious = [c for c in candidates if c[0] <= threshold]
    parsimonious.sort(key=lambda c: c[1])
    return parsimonious[0][2]

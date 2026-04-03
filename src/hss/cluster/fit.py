
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .auto_k import select_k_gmm, select_k_kmeans
from .base import BatchFactory, ClusterModel, _materialize
from .gmm import _fit_gmm, _gmm_kwargs_from_params
from .kmeans import _fit_kmeans


def fit_cluster(
    factory: BatchFactory,
    *,
    method: str,
    k: Optional[int],
    k_range: Tuple[int, int],
    seed: int,
    tolerance: float = 0.05,
    icl_mode: str = 'bic_plus_2entropy',
    params: Dict[str, Any],
) -> ClusterModel:
    m = method.lower()
    if m == 'kmeans':
        X = _materialize(factory)
        if k is not None:
            extra = {
                kk: params[kk]
                for kk in ('batch_size', 'n_init', 'max_iter')
                if kk in params
            }
            return _fit_kmeans(X, k, seed, **extra)
        return select_k_kmeans(X, k_range=k_range, seed=seed, tolerance=tolerance, params=params)
    if m == 'gmm':
        X = _materialize(factory)
        if k is not None:
            return _fit_gmm(X, k, seed, **_gmm_kwargs_from_params(params))
        return select_k_gmm(
            X,
            k_range=k_range,
            seed=seed,
            tolerance=tolerance,
            icl_mode=icl_mode,
            params=params,
        )
    raise ValueError(f'Unknown clustering method: {method}')

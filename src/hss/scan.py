from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from .cluster.gmm import _fit_gmm
from .cluster.kmeans import _fit_kmeans
from .cluster.metrics import compute_icl, silhouette_sampled
from .transform import build_chain_from_spec
from .types import ClusterSpec, StateProvider, TransformSpec
from .utils import f32, set_thread_env


@dataclass
class ScanConfig:
    k_range: Tuple[int, int] = (2, 40)
    seed: int = 42
    stability_repeats: int = 2
    eval_sample: int = 5000
    icl_mode: str = "bic_plus_2entropy"
    transform: TransformSpec = field(default_factory=TransformSpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    batch_size: int = 4096
    n_jobs: int = 1
    parallel_backend: str = "loky"


def _fit_one(
    X: np.ndarray,
    method: str,
    k: int,
    seed: int,
    params: Dict[str, Any],
):
    if method == "kmeans":
        return _fit_kmeans(
            X,
            k,
            seed,
            batch_size=int(params.get("batch_size", 1024)),
            n_init=int(params.get("n_init", 10)),
            max_iter=int(params.get("max_iter", 100)),
        )
    if method == "gmm":
        return _fit_gmm(
            X,
            k,
            seed,
            covariance_type=str(params.get("covariance_type", "diag")),
            reg_covar=float(params.get("reg_covar", 1e-2)),
            n_init=int(params.get("n_init", 2)),
            max_iter=int(params.get("max_iter", 200)),
        )
    raise ValueError(f"Unknown clustering method: {method}")


def _stability(
    X: np.ndarray,
    k: int,
    seed: int,
    repeats: int,
    method: str,
    params: Dict[str, Any],
) -> float:
    if repeats < 2:
        return float("nan")
    labels_list: List[np.ndarray] = []
    for r in range(repeats):
        model = _fit_one(X, method, k, seed + r * 10007, params)
        labels_list.append(model.predict(X))
    scores = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(len(labels_list))
        for j in range(i + 1, len(labels_list))
    ]
    return float(np.mean(scores))


def _scan_one_layer(
    provider: StateProvider,
    *,
    layer: int,
    config: ScanConfig,
    indices: Optional[np.ndarray],
) -> pd.DataFrame:
    chunks: List[np.ndarray] = []
    for batch in provider.iter_batches(
        layer=layer, indices=indices, batch_size=config.batch_size
    ):
        chunks.append(f32(batch.states))
    if not chunks:
        return pd.DataFrame()
    X_raw = np.concatenate(chunks, axis=0)

    transform = build_chain_from_spec(config.transform)

    def raw_factory():
        yield X_raw

    transform.fit(raw_factory)
    X = transform.transform(X_raw)

    rng = np.random.RandomState(config.seed + layer)
    n = len(X)
    eval_n = min(config.eval_sample, n)
    eval_idx = (
        rng.choice(n, eval_n, replace=False)
        if n > config.eval_sample
        else np.arange(n)
    )
    X_eval = X[eval_idx]

    k_lo = max(2, config.k_range[0])
    k_hi = min(config.k_range[1], n - 1)
    if k_lo > k_hi:
        return pd.DataFrame(
            columns=["layer", "k", "icl", "silhouette", "stability", "n_samples"]
        )

    method = config.cluster.method
    params = dict(config.cluster.params)
    rows: List[Dict[str, Any]] = []
    for k in range(k_lo, k_hi + 1):
        seed_k = config.seed + layer * 10007 + k
        try:
            model = _fit_one(X, method, k, seed_k, params)
            labels_eval = model.predict(X_eval)

            icl = compute_icl(model, X, config.icl_mode)
            sil = silhouette_sampled(X_eval, labels_eval, seed_k)
            stab = _stability(
                X,
                k,
                seed_k,
                config.stability_repeats,
                method,
                params,
            )

            rows.append({
                "layer": int(layer),
                "k": int(k),
                "icl": icl,
                "silhouette": sil,
                "stability": stab,
                "n_samples": int(n),
            })
        except Exception:
            rows.append({
                "layer": int(layer),
                "k": int(k),
                "icl": float("nan"),
                "silhouette": float("nan"),
                "stability": float("nan"),
                "n_samples": int(n),
            })

    return pd.DataFrame(rows)


def scan(
    provider: StateProvider,
    *,
    config: Optional[ScanConfig] = None,
    layers: Optional[List[int]] = None,
    indices: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    if config is None:
        config = ScanConfig()
    if layers is None:
        layers = [int(l) for l in provider.layers()]

    n_jobs = int(config.n_jobs)
    if n_jobs != 1:
        set_thread_env(1)

    if n_jobs == 1:
        all_dfs = [
            _scan_one_layer(provider, layer=l, config=config, indices=indices)
            for l in layers
        ]
    else:
        all_dfs = Parallel(
            n_jobs=n_jobs,
            backend=config.parallel_backend,
        )(
            delayed(_scan_one_layer)(
                provider, layer=l, config=config, indices=indices
            )
            for l in layers
        )

    all_dfs = [df for df in all_dfs if not df.empty]
    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if save_path is not None and not result.empty:
        result.to_parquet(save_path, index=False)

    return result
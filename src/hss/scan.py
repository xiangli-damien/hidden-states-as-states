
from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .cluster.gmm import _fit_gmm, _gmm_kwargs_from_params
from .cluster.kmeans import _fit_kmeans
from .cluster.metrics import compute_icl, silhouette_sampled
from .provider import MemmapProvider
from .transform import build_chain_from_spec
from .types import ClusterSpec, StateProvider, TransformSpec
from .utils import (
    f32,
    i64,
    materialize_provider_states,
    resolve_parallel_backend,
    set_thread_env,
)

_log = logging.getLogger(__name__)
_MAX_SEED = 2**31 - 1


@dataclass
class ScanConfig:
    k_range: Tuple[int, int] = (2, 40)
    seed: int = 42
    stability_repeats: int = 2
    eval_sample: int = 5000
    icl_mode: str = 'bic_plus_2entropy'
    transform: TransformSpec = field(default_factory=TransformSpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    batch_size: int = 4096
    n_jobs: int = 1
    parallel_backend: str = 'auto'
    prefer_memmap: bool = True
    progress: bool = True
    temp_root: Optional[str] = None


@dataclass
class _ScanParallelContext:
    provider: StateProvider
    backend: str
    temp_dir: Optional[tempfile.TemporaryDirectory]
    memmap_path: Optional[Path]
    memmap_layer_ids: Optional[List[int]]


def _fit_one(X: np.ndarray, method: str, k: int, seed: int, params: Dict[str, Any]):
    if method == 'kmeans':
        return _fit_kmeans(
            X,
            k,
            seed,
            batch_size=int(params.get('batch_size', 1024)),
            n_init=int(params.get('n_init', 10)),
            max_iter=int(params.get('max_iter', 100)),
        )
    if method == 'gmm':
        return _fit_gmm(X, k, seed, **_gmm_kwargs_from_params(params))
    raise ValueError(f'Unknown clustering method: {method}')


def _stability(
    X: np.ndarray,
    k: int,
    seed: int,
    repeats: int,
    method: str,
    params: Dict[str, Any],
) -> float:
    if int(repeats) < 2:
        return float('nan')
    labels_list: List[np.ndarray] = []
    for rep in range(int(repeats)):
        s = int(seed) + rep * 10007
        if s < 0 or s > _MAX_SEED:
            s %= _MAX_SEED + 1
        model = _fit_one(X, method, k, s, params)
        labels_list.append(model.predict(X))
    scores = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(len(labels_list))
        for j in range(i + 1, len(labels_list))
    ]
    return float(np.mean(scores)) if scores else float('nan')


def _scan_one_layer(
    provider: StateProvider,
    *,
    layer: int,
    config: ScanConfig,
    indices: Optional[np.ndarray],
) -> pd.DataFrame:
    chunks: List[np.ndarray] = []
    for batch in provider.iter_batches(
        layer=layer, indices=indices, batch_size=int(config.batch_size)
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

    rng = np.random.RandomState(int(config.seed) + int(layer))
    n = int(len(X))
    eval_n = min(int(config.eval_sample), n)
    eval_idx = rng.choice(n, eval_n, replace=False) if n > eval_n else np.arange(n)
    X_eval = X[eval_idx]

    k_lo = max(2, int(config.k_range[0]))
    k_hi = min(int(config.k_range[1]), n - 1)
    if k_lo > k_hi:
        return pd.DataFrame(
            columns=['layer', 'k', 'icl', 'silhouette', 'stability', 'n_samples']
        )

    method = str(config.cluster.method)
    params = dict(config.cluster.params)
    rows: List[Dict[str, Any]] = []
    total_k = k_hi - k_lo + 1

    for offset, k in enumerate(range(k_lo, k_hi + 1), start=1):
        seed_k = int(config.seed) + int(layer) * 10007 + int(k)
        if seed_k < 0 or seed_k > _MAX_SEED:
            seed_k %= _MAX_SEED + 1
        try:
            model = _fit_one(X, method, k, seed_k, params)
            labels_eval = model.predict(X_eval)
            icl = compute_icl(model, X, config.icl_mode)
            sil = silhouette_sampled(X_eval, labels_eval, seed_k)
            stab = _stability(X, k, seed_k, config.stability_repeats, method, params)
            rows.append(
                {
                    'layer': int(layer),
                    'k': int(k),
                    'icl': float(icl),
                    'silhouette': float(sil),
                    'stability': float(stab),
                    'n_samples': int(n),
                }
            )
        except Exception:
            rows.append(
                {
                    'layer': int(layer),
                    'k': int(k),
                    'icl': float('nan'),
                    'silhouette': float('nan'),
                    'stability': float('nan'),
                    'n_samples': int(n),
                }
            )
        if config.progress:
            _log.info('scan layer=%d progress=%d/%d', layer, offset, total_k)
    return pd.DataFrame(rows)


def _scan_one_layer_from_memmap(
    *,
    provider_path: str,
    provider_layer_ids: Sequence[int],
    config: ScanConfig,
    layer: int,
    indices: Optional[np.ndarray],
) -> pd.DataFrame:
    provider = MemmapProvider(provider_path, layer_ids=list(provider_layer_ids))
    return _scan_one_layer(provider, layer=layer, config=config, indices=indices)


def _prepare_parallel_context(
    provider: StateProvider,
    *,
    config: ScanConfig,
    layers: Sequence[int],
    indices: Optional[np.ndarray],
) -> _ScanParallelContext:
    backend = resolve_parallel_backend(
        config.parallel_backend,
        n_jobs=int(config.n_jobs),
        prefer_memmap=bool(config.prefer_memmap),
    )
    if backend != 'process':
        return _ScanParallelContext(
            provider=provider,
            backend=backend,
            temp_dir=None,
            memmap_path=None,
            memmap_layer_ids=None,
        )
    if isinstance(provider, MemmapProvider) and provider.path is not None:
        return _ScanParallelContext(
            provider=provider,
            backend='process',
            temp_dir=None,
            memmap_path=Path(provider.path),
            memmap_layer_ids=[int(x) for x in provider.layers()],
        )
    if config.temp_root:
        temp_root = Path(config.temp_root).expanduser().resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix='hss_scan_')
        temp_root = Path(temp_dir_obj.name)
    memmap_path = temp_root / 'scan_provider_states.npy'
    _log.info('scan backend=process; materializing provider to %s', memmap_path)
    materialize_provider_states(
        provider,
        path=memmap_path,
        layers=layers,
        indices=indices,
        batch_size=max(int(config.batch_size), 1024),
        dtype=np.float32,
    )
    working_provider = MemmapProvider(memmap_path, layer_ids=[int(x) for x in layers])
    return _ScanParallelContext(
        provider=working_provider,
        backend='process',
        temp_dir=temp_dir_obj,
        memmap_path=memmap_path,
        memmap_layer_ids=[int(x) for x in layers],
    )


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
    selected_layers = [int(x) for x in (layers if layers is not None else provider.layers())]
    idx = None if indices is None else i64(np.asarray(indices))
    if int(config.n_jobs) != 1:
        set_thread_env(1)

    parallel_ctx = _prepare_parallel_context(
        provider,
        config=config,
        layers=selected_layers,
        indices=idx,
    )
    working_provider = parallel_ctx.provider
    worker_indices = idx
    if (
        parallel_ctx.backend == 'process'
        and parallel_ctx.memmap_path is not None
        and (not isinstance(provider, MemmapProvider))
    ):
        worker_indices = None

    all_dfs: List[pd.DataFrame] = []
    if parallel_ctx.backend == 'serial':
        for done, layer in enumerate(selected_layers, start=1):
            all_dfs.append(
                _scan_one_layer(
                    working_provider,
                    layer=layer,
                    config=config,
                    indices=worker_indices,
                )
            )
            if config.progress:
                _log.info('scan completed layer %d (%d/%d)', layer, done, len(selected_layers))
    elif parallel_ctx.backend == 'thread':
        with ThreadPoolExecutor(max_workers=int(config.n_jobs)) as ex:
            futures = {
                ex.submit(
                    _scan_one_layer,
                    working_provider,
                    layer=layer,
                    config=config,
                    indices=worker_indices,
                ): layer
                for layer in selected_layers
            }
            completed = 0
            for future in as_completed(futures):
                layer = futures[future]
                all_dfs.append(future.result())
                completed += 1
                if config.progress:
                    _log.info(
                        'scan completed layer %d (%d/%d)',
                        layer,
                        completed,
                        len(selected_layers),
                    )
    else:
        assert parallel_ctx.memmap_path is not None and parallel_ctx.memmap_layer_ids is not None
        with ProcessPoolExecutor(max_workers=int(config.n_jobs)) as ex:
            futures = {
                ex.submit(
                    _scan_one_layer_from_memmap,
                    provider_path=str(parallel_ctx.memmap_path),
                    provider_layer_ids=list(parallel_ctx.memmap_layer_ids),
                    config=config,
                    layer=layer,
                    indices=None
                    if worker_indices is None
                    else np.asarray(worker_indices, dtype=np.int64),
                ): layer
                for layer in selected_layers
            }
            completed = 0
            for future in as_completed(futures):
                layer = futures[future]
                all_dfs.append(future.result())
                completed += 1
                if config.progress:
                    _log.info(
                        'scan completed layer %d (%d/%d)',
                        layer,
                        completed,
                        len(selected_layers),
                    )

    all_dfs = [df for df in all_dfs if df is not None and (not df.empty)]
    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if save_path is not None and (not result.empty):
        result.to_parquet(save_path, index=False)
    if parallel_ctx.temp_dir is not None:
        parallel_ctx.temp_dir.cleanup()
    return result

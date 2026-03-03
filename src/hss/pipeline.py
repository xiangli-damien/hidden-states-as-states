from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed

from .align import align_layers
from .cluster import ClusterModel, fit_cluster
from .storage import ArtifactStore
from .trajectory import build_global_labels
from .transform import L2NormTransform, TransformChain, build_chain_from_spec
from .types import (
    AlignmentResult,
    AlignSpec,
    HSSConfig,
    HSSResult,
    LayerResult,
    StateProvider,
)
from .utils import atomic_json, atomic_npy, f32, i64, read_json, set_thread_env

_log = logging.getLogger(__name__)

_MAX_SEED = 2**31 - 1


def _layer_seed(base_seed: int, layer: int) -> int:
    s = int(base_seed) + 10007 * int(layer)
    return int(s % (_MAX_SEED + 1)) if s > _MAX_SEED or s < 0 else s


@dataclass
class _LayerProcessResult:
    layer: int
    labels: np.ndarray
    centers_hidden: np.ndarray
    transform_config: Dict[str, Any]
    cluster_config: Dict[str, Any]


def _raw_batches(
    provider: StateProvider,
    *,
    layer: int,
    indices: Optional[np.ndarray],
    batch_size: int,
):
    """Yield f32 state matrices from the provider with dimension validation.

    Raises ValueError if any batch is not 2-D or has an inconsistent
    feature dimension compared to earlier batches from the same layer.
    """
    expected_dim: Optional[int] = None
    for batch in provider.iter_batches(
        layer=layer, indices=indices, batch_size=batch_size
    ):
        X = f32(batch.states)
        if X.ndim != 2:
            raise ValueError(
                f"Layer {layer}: expected 2-D batch, got shape {X.shape}"
            )
        if X.shape[0] == 0:
            continue
        if expected_dim is None:
            expected_dim = X.shape[1]
        elif X.shape[1] != expected_dim:
            raise ValueError(
                f"Layer {layer}: inconsistent state dim — "
                f"expected {expected_dim}, got {X.shape[1]}"
            )
        yield X


def _predict_all(
    provider: StateProvider,
    *,
    layer: int,
    indices: Optional[np.ndarray],
    n_items: int,
    batch_size: int,
    transform: TransformChain,
    model: ClusterModel,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is None:
        out = np.empty(n_items, dtype=np.int32)
    pos = 0
    for batch in provider.iter_batches(
        layer=layer, indices=indices, batch_size=batch_size
    ):
        X = f32(batch.states)
        b = X.shape[0]
        if b == 0:
            continue
        Xf = transform.transform(X)
        out[pos : pos + b] = model.predict(Xf)
        pos += b
    if pos != n_items:
        raise ValueError(f"Expected {n_items} labels, wrote {pos}")
    return out


def _process_one_layer(
    provider: StateProvider,
    *,
    config: HSSConfig,
    layer: int,
    indices: Optional[np.ndarray],
    n_items: int,
    store: Optional[ArtifactStore],
) -> _LayerProcessResult:
    seed = _layer_seed(config.seed, layer)
    transform = build_chain_from_spec(config.transform)

    def raw_factory():
        return _raw_batches(
            provider,
            layer=layer,
            indices=indices,
            batch_size=config.batch_size_fit,
        )

    transform.fit(raw_factory)

    def feat_factory():
        for X in raw_factory():
            yield transform.transform(X)

    model = fit_cluster(
        feat_factory,
        method=config.cluster.method,
        k=config.cluster.k,
        k_range=config.cluster.k_range,
        seed=seed,
        tolerance=config.parsimony_tolerance,
        icl_mode=config.icl_mode,
        params=config.cluster.params,
    )

    centers_feat = model.centers()
    centers_hidden = transform.inverse(f32(centers_feat))

    labels = _predict_all(
        provider,
        layer=layer,
        indices=indices,
        n_items=n_items,
        batch_size=config.batch_size_predict,
        transform=transform,
        model=model,
    )

    if store is not None:
        d = store.layer_dir(layer)
        atomic_npy(d / "labels.npy", labels)
        store.save_layer_model(layer, transform, model)

    _log.debug(
        "Layer %d: k=%d, n_items=%d", layer, model.n_clusters(), n_items
    )

    return _LayerProcessResult(
        layer=layer,
        labels=labels,
        centers_hidden=f32(centers_hidden),
        transform_config=transform.config(),
        cluster_config=model.config(),
    )


def discretize(
    provider: StateProvider,
    *,
    config: Optional[HSSConfig] = None,
    store_path: Optional[str | Path] = None,
    k_map: Optional[Dict[int, int]] = None,
) -> HSSResult:
    if config is None:
        config = HSSConfig()
    chain = build_chain_from_spec(config.transform)
    if any(isinstance(s, L2NormTransform) for s in chain.steps):
        raise ValueError(
            "Transform chain contains L2NormTransform, which has no inverse. "
            "discretize needs inverse to map cluster centers to hidden space; "
            "remove L2Norm from the chain or use a different pipeline."
        )
    indices = config.indices
    if indices is not None:
        indices = i64(np.asarray(indices))
    n_items = provider.n_items() if indices is None else len(indices)
    layers = (
        [int(l) for l in config.layers]
        if config.layers is not None
        else [int(l) for l in provider.layers()]
    )
    store: Optional[ArtifactStore] = None
    if store_path is not None:
        store = ArtifactStore(root=Path(store_path))

    def effective_config(layer: int) -> HSSConfig:
        if k_map is not None and layer in k_map:
            return replace(
                config,
                cluster=replace(config.cluster, k=k_map[layer]),
            )
        return config

    n_jobs = int(config.n_jobs)
    if n_jobs != 1:
        set_thread_env(1)

    _log.info(
        "discretize: %d layers, n_items=%d, n_jobs=%d",
        len(layers),
        n_items,
        n_jobs,
    )

    if n_jobs == 1:
        raw_results = [
            _process_one_layer(
                provider,
                config=effective_config(l),
                layer=l,
                indices=indices,
                n_items=n_items,
                store=store,
            )
            for l in layers
        ]
    else:
        raw_results = Parallel(
            n_jobs=n_jobs,
            backend=config.parallel_backend,
        )(
            delayed(_process_one_layer)(
                provider,
                config=effective_config(l),
                layer=l,
                indices=indices,
                n_items=n_items,
                store=store,
            )
            for l in layers
        )

    raw_results.sort(key=lambda x: x.layer)

    layer_results: List[LayerResult] = []
    centers_by_layer: List[np.ndarray] = []
    labels_by_layer: List[np.ndarray] = []
    for r in raw_results:
        layer_results.append(
            LayerResult(
                layer=r.layer,
                labels=r.labels,
                centers_hidden=r.centers_hidden,
                n_clusters=int(r.centers_hidden.shape[0]),
                transform_config=r.transform_config,
                cluster_config=r.cluster_config,
            )
        )
        centers_by_layer.append(r.centers_hidden)
        labels_by_layer.append(r.labels)

    alignment: Optional[AlignmentResult] = None
    global_labels: Optional[np.ndarray] = None
    if len(layers) > 1:
        alignment = align_layers(
            centers_by_layer,
            layers=layers,
            spec=config.alignment,
        )
        global_labels = build_global_labels(
            labels_by_layer, alignment.local_to_global
        )

    result = HSSResult(
        config=config,
        layer_results=layer_results,
        alignment=alignment,
        global_labels=global_labels,
    )

    if store is not None:
        store.save(result)
        # Record the effective per-layer k choices alongside the base config
        # so that a later load() can see what k was actually used per layer.
        effective_k: Dict[str, int] = {
            str(lr.layer): lr.n_clusters for lr in layer_results
        }
        atomic_json(store.root / "effective_k_map.json", effective_k)

    return result


def discretize_with_existing_models(
    provider: StateProvider,
    *,
    store_path: str | Path,
    layers: Optional[List[int]] = None,
    indices: Optional[np.ndarray] = None,
    align_spec: Optional[AlignSpec] = None,
    batch_size: int = 8192,
) -> HSSResult:
    store = ArtifactStore(root=Path(store_path))

    saved_config = HSSConfig()
    if store.config_path.exists():
        saved_config = HSSConfig.from_dict(read_json(store.config_path))

    if layers is None:
        layers = store.list_layers()
    if indices is not None:
        indices = i64(np.asarray(indices))
    n_items = provider.n_items() if indices is None else len(indices)

    layer_results: List[LayerResult] = []
    centers_by_layer: List[np.ndarray] = []
    labels_by_layer: List[np.ndarray] = []

    for l in layers:
        chain, model = store.load_layer_model(l)
        labels = np.empty(n_items, dtype=np.int32)
        pos = 0
        for batch in provider.iter_batches(
            layer=l, indices=indices, batch_size=batch_size
        ):
            X = f32(batch.states)
            b = X.shape[0]
            if b == 0:
                continue
            Xf = chain.transform(X)
            labels[pos : pos + b] = model.predict(Xf)
            pos += b
        centers_hidden = chain.inverse(f32(model.centers()))
        lr = LayerResult(
            layer=l,
            labels=labels,
            centers_hidden=centers_hidden,
            n_clusters=model.n_clusters(),
            transform_config=chain.config(),
            cluster_config=model.config(),
        )
        layer_results.append(lr)
        centers_by_layer.append(centers_hidden)
        labels_by_layer.append(labels)

    alignment: Optional[AlignmentResult] = None
    global_labels: Optional[np.ndarray] = None
    if len(layers) > 1:
        if align_spec is None:
            align_spec = saved_config.alignment
        alignment = align_layers(
            centers_by_layer, layers=layers, spec=align_spec
        )
        global_labels = build_global_labels(
            labels_by_layer, alignment.local_to_global
        )

    return HSSResult(
        config=saved_config,
        layer_results=layer_results,
        alignment=alignment,
        global_labels=global_labels,
    )
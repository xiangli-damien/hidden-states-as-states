from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .align import align_layers
from .cluster import ClusterModel, fit_cluster
from .provider import MemmapProvider
from .storage import ArtifactStore
from .trajectory import build_global_labels
from .transform import L2NormTransform, TransformChain, build_chain_from_spec
from .types import HSSConfig, HSSResult, LayerResult, RowMetadata, StateProvider
from .utils import (
    atomic_json,
    atomic_npy,
    f32,
    i64,
    materialize_provider_states,
    open_writable_memmap,
    read_json,
    resolve_parallel_backend,
    sanitize_metadata_fields,
    set_thread_env,
)

_log = logging.getLogger(__name__)

_MAX_SEED = 2**31 - 1


def _layer_seed(base_seed: int, layer: int) -> int:
    seed = int(base_seed) + 10007 * int(layer)
    if seed < 0 or seed > _MAX_SEED:
        seed %= (_MAX_SEED + 1)
    return int(seed)


@dataclass
class _LayerProcessResult:
    layer: int
    labels: Optional[np.ndarray]
    soft_labels: Optional[np.ndarray]
    centers_hidden: np.ndarray
    n_clusters: int
    transform_config: Dict[str, Any]
    cluster_config: Dict[str, Any]


@dataclass
class _ParallelContext:
    provider: StateProvider
    backend: str
    temp_dir: Optional[tempfile.TemporaryDirectory]
    memmap_path: Optional[Path]
    memmap_layer_ids: Optional[List[int]]


def _raw_batches(
    provider: StateProvider,
    *,
    layer: int,
    indices: Optional[np.ndarray],
    batch_size: int,
) -> Iterable[np.ndarray]:
    expected_dim: Optional[int] = None
    for batch in provider.iter_batches(layer=layer, indices=indices, batch_size=batch_size):
        X = f32(batch.states)
        if X.ndim != 2:
            raise ValueError(f"Layer {layer}: expected 2-D batch, got shape {X.shape}")
        if X.shape[0] == 0:
            continue
        if expected_dim is None:
            expected_dim = int(X.shape[1])
        elif int(X.shape[1]) != expected_dim:
            raise ValueError(
                f"Layer {layer}: inconsistent state dim, expected {expected_dim} but got {X.shape[1]}"
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
    out_labels: Optional[np.ndarray] = None,
    soft_labels_dtype: Optional[np.dtype] = None,
    soft_labels_out_path: Optional[Path] = None,
    log_every_batches: int = 0,
    progress_prefix: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if out_labels is None:
        out_labels = np.empty(int(n_items), dtype=np.int32)

    soft_labels: Optional[np.ndarray] = None
    soft_tmp_path: Optional[Path] = None
    if soft_labels_dtype is not None:
        shape = (int(n_items), int(model.n_clusters()))
        if soft_labels_out_path is not None:
            soft_tmp_path = soft_labels_out_path.with_name(soft_labels_out_path.name + ".tmp")
            soft_labels = open_writable_memmap(soft_tmp_path, shape=shape, dtype=soft_labels_dtype)
        else:
            soft_labels = np.empty(shape, dtype=soft_labels_dtype)

    pos = 0
    batch_count = 0
    for batch in provider.iter_batches(layer=layer, indices=indices, batch_size=batch_size):
        X = f32(batch.states)
        b = int(X.shape[0])
        if b == 0:
            continue
        Xf = transform.transform(X)
        out_labels[pos : pos + b] = model.predict(Xf)
        if soft_labels is not None:
            probs = model.predict_proba(Xf)
            soft_labels[pos : pos + b] = np.asarray(probs, dtype=soft_labels.dtype)
        pos += b
        batch_count += 1
        if log_every_batches > 0 and batch_count % int(log_every_batches) == 0:
            _log.info("%sassigned %d / %d rows", progress_prefix, pos, n_items)

    if pos != int(n_items):
        raise ValueError(f"Expected {n_items} assigned rows but wrote {pos}")

    if soft_tmp_path is not None:
        assert soft_labels_out_path is not None
        assert soft_labels is not None
        soft_labels.flush()
        del soft_labels
        soft_tmp_path.replace(soft_labels_out_path)
        soft_labels = np.load(soft_labels_out_path, mmap_mode="r")

    return out_labels, soft_labels


def _process_one_layer(
    provider: StateProvider,
    *,
    config: HSSConfig,
    layer: int,
    indices: Optional[np.ndarray],
    n_items: int,
    store_root: Optional[str | Path],
    return_arrays: bool,
) -> _LayerProcessResult:
    seed = _layer_seed(config.seed, layer)
    transform = build_chain_from_spec(config.transform)

    def raw_factory() -> Iterable[np.ndarray]:
        return _raw_batches(
            provider,
            layer=layer,
            indices=indices,
            batch_size=int(config.batch_size_fit),
        )

    _log.info("layer=%d stage=transform_fit", layer)
    transform.fit(raw_factory)

    def feat_factory() -> Iterable[np.ndarray]:
        for X in raw_factory():
            yield transform.transform(X)

    _log.info("layer=%d stage=cluster_fit", layer)
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

    labels_path: Optional[Path] = None
    soft_path: Optional[Path] = None
    if store_root is not None:
        d = ArtifactStore(Path(store_root)).layer_dir(layer)
        labels_path = d / "labels.npy"
        if config.store_soft_labels:
            soft_path = d / "soft_labels.npy"

    _log.info("layer=%d stage=predict", layer)
    labels, soft_labels = _predict_all(
        provider,
        layer=layer,
        indices=indices,
        n_items=n_items,
        batch_size=int(config.batch_size_predict),
        transform=transform,
        model=model,
        soft_labels_dtype=np.dtype(config.soft_labels_dtype) if config.store_soft_labels else None,
        soft_labels_out_path=soft_path,
        log_every_batches=int(config.log_every_batches),
        progress_prefix=f"layer={layer} ",
    )

    if store_root is not None:
        store = ArtifactStore(Path(store_root))
        d = store.layer_dir(layer)
        atomic_npy(d / "labels.npy", labels)
        atomic_npy(d / "centers_hidden.npy", centers_hidden.astype(np.float32, copy=False))
        atomic_json(d / "meta.json", {
            "layer": int(layer),
            "n_clusters": int(model.n_clusters()),
            "transform_config": transform.config(),
            "cluster_config": model.config(),
            "has_soft_labels": bool(config.store_soft_labels),
        })
        store.save_layer_model(layer, transform, model)
        if config.store_soft_labels and soft_labels is not None and soft_path is None:
            atomic_npy(d / "soft_labels.npy", soft_labels)

    if not return_arrays and store_root is not None:
        labels = None
        soft_labels = None

    return _LayerProcessResult(
        layer=int(layer),
        labels=labels,
        soft_labels=soft_labels,
        centers_hidden=f32(centers_hidden),
        n_clusters=int(model.n_clusters()),
        transform_config=transform.config(),
        cluster_config=model.config(),
    )


def _process_one_layer_from_memmap(
    *,
    provider_path: str,
    provider_layer_ids: Sequence[int],
    config_dict: Dict[str, Any],
    layer: int,
    indices: Optional[np.ndarray],
    n_items: int,
    store_root: Optional[str | Path],
    return_arrays: bool,
) -> _LayerProcessResult:
    provider = MemmapProvider(provider_path, layer_ids=list(provider_layer_ids))
    config = HSSConfig.from_dict(config_dict)
    return _process_one_layer(
        provider,
        config=config,
        layer=int(layer),
        indices=None if indices is None else np.asarray(indices, dtype=np.int64),
        n_items=int(n_items),
        store_root=store_root,
        return_arrays=bool(return_arrays),
    )


def _collect_row_metadata(
    provider: StateProvider,
    *,
    layers: Sequence[int],
    indices: Optional[np.ndarray],
    batch_size: int,
) -> Optional[RowMetadata]:
    if len(layers) == 0:
        return None
    n_items = provider.n_items() if indices is None else int(len(indices))
    unit = "item"
    if hasattr(provider, "unit") and callable(getattr(provider, "unit")):
        try:
            unit = str(provider.unit())
        except Exception:
            unit = "item"

    provider_manifest: Dict[str, Any] = {}
    if hasattr(provider, "metadata_manifest") and callable(getattr(provider, "metadata_manifest")):
        try:
            provider_manifest = dict(provider.metadata_manifest())
        except Exception:
            provider_manifest = {}

    row_ids: Optional[np.ndarray] = None
    if hasattr(provider, "row_ids") and callable(getattr(provider, "row_ids")):
        try:
            row_ids = np.asarray(provider.row_ids(indices), dtype=np.int64)
        except Exception:
            row_ids = None

    sample_ids: Optional[np.ndarray] = None
    if hasattr(provider, "sample_ids") and callable(getattr(provider, "sample_ids")):
        try:
            fetched = provider.sample_ids(indices)
            if fetched is not None:
                sample_ids = np.asarray(fetched, dtype=np.int64)
        except Exception:
            sample_ids = None

    fields: Dict[str, np.ndarray] = {}
    if hasattr(provider, "metadata_fields") and callable(getattr(provider, "metadata_fields")):
        try:
            fields = sanitize_metadata_fields(provider.metadata_fields(indices), expected_len=n_items)
        except Exception:
            fields = {}

    if row_ids is None or (sample_ids is None and not fields):
        row_chunks: List[np.ndarray] = []
        sample_chunks: List[np.ndarray] = []
        field_chunks: Dict[str, List[np.ndarray]] = {}
        first_layer = int(layers[0])
        for batch in provider.iter_batches(layer=first_layer, indices=indices, batch_size=int(batch_size)):
            b = int(batch.states.shape[0])
            if row_ids is None:
                if batch.ids is not None:
                    row_chunks.append(np.asarray(batch.ids, dtype=np.int64))
                else:
                    start = sum(len(x) for x in row_chunks)
                    row_chunks.append(np.arange(start, start + b, dtype=np.int64))
            if sample_ids is None and batch.sample_ids is not None:
                sample_chunks.append(np.asarray(batch.sample_ids, dtype=np.int64))
            if not fields and batch.fields:
                for key, value in sanitize_metadata_fields(batch.fields, expected_len=b).items():
                    field_chunks.setdefault(key, []).append(value)
        if row_ids is None:
            row_ids = np.concatenate(row_chunks, axis=0) if row_chunks else np.arange(n_items, dtype=np.int64)
        if sample_ids is None and sample_chunks:
            sample_ids = np.concatenate(sample_chunks, axis=0)
        if not fields and field_chunks:
            fields = {key: np.concatenate(parts, axis=0) for key, parts in field_chunks.items()}

    if row_ids is None:
        row_ids = np.arange(n_items, dtype=np.int64)

    return RowMetadata(
        unit=unit,
        row_ids=np.asarray(row_ids, dtype=np.int64),
        sample_ids=None if sample_ids is None else np.asarray(sample_ids, dtype=np.int64),
        fields=fields,
        provider_manifest=provider_manifest,
    )


def _prepare_parallel_context(
    provider: StateProvider,
    *,
    config: HSSConfig,
    layers: Sequence[int],
    indices: Optional[np.ndarray],
    store_path: Optional[str | Path],
) -> _ParallelContext:
    backend = resolve_parallel_backend(
        config.parallel_backend,
        n_jobs=int(config.n_jobs),
        prefer_memmap=bool(config.prefer_memmap),
    )
    if backend != "process":
        return _ParallelContext(
            provider=provider,
            backend=backend,
            temp_dir=None,
            memmap_path=None,
            memmap_layer_ids=None,
        )

    if isinstance(provider, MemmapProvider) and provider.path is not None:
        return _ParallelContext(
            provider=provider,
            backend="process",
            temp_dir=None,
            memmap_path=Path(provider.path),
            memmap_layer_ids=[int(x) for x in provider.layers()],
        )

    if store_path is not None:
        temp_root = Path(store_path).expanduser().resolve() / "_tmp_parallel"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_root_value = config.temp_root
        if temp_root_value:
            temp_root = Path(temp_root_value).expanduser().resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_dir_obj = None
        else:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="hss_parallel_")
            temp_root = Path(temp_dir_obj.name)
    memmap_path = temp_root / "provider_states.npy"
    _log.info("parallel backend=process; materializing provider to %s", memmap_path)
    materialize_provider_states(
        provider,
        path=memmap_path,
        layers=layers,
        indices=indices,
        batch_size=max(int(config.batch_size_fit), 1024),
        dtype=np.float32,
    )
    working_provider = MemmapProvider(memmap_path, layer_ids=[int(x) for x in layers])
    return _ParallelContext(
        provider=working_provider,
        backend="process",
        temp_dir=temp_dir_obj,
        memmap_path=memmap_path,
        memmap_layer_ids=[int(x) for x in layers],
    )


def _update_progress(
    store: Optional[ArtifactStore],
    *,
    stage: str,
    total_layers: int,
    completed_layers: int,
    current_layer: Optional[int] = None,
    backend: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if store is None:
        return
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "total_layers": int(total_layers),
        "completed_layers": int(completed_layers),
        "current_layer": None if current_layer is None else int(current_layer),
    }
    if backend is not None:
        payload["backend"] = str(backend)
    if extra:
        payload.update(extra)
    store.save_progress(payload)


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
    if any(isinstance(step, L2NormTransform) for step in chain.steps):
        raise ValueError(
            "Transform chain contains L2NormTransform, which has no inverse. "
            "discretize needs inverse to map cluster centers back to hidden space."
        )

    indices = None if config.indices is None else i64(np.asarray(config.indices))
    layers = [int(x) for x in (config.layers if config.layers is not None else provider.layers())]
    n_items = provider.n_items() if indices is None else int(len(indices))
    store = ArtifactStore(Path(store_path)) if store_path is not None else None

    if int(config.n_jobs) != 1:
        set_thread_env(1)

    row_metadata = None
    if config.capture_row_metadata:
        row_metadata = _collect_row_metadata(
            provider,
            layers=layers,
            indices=indices,
            batch_size=max(int(config.batch_size_predict), 1024),
        )

    parallel_ctx = _prepare_parallel_context(
        provider,
        config=config,
        layers=layers,
        indices=indices,
        store_path=store_path,
    )
    working_provider = parallel_ctx.provider
    worker_indices = indices
    if parallel_ctx.backend == "process" and parallel_ctx.memmap_path is not None and not isinstance(provider, MemmapProvider):
        # The materialized memmap already contains the selected subset in order.
        worker_indices = None

    def effective_config(layer: int) -> HSSConfig:
        if k_map is not None and layer in k_map:
            return replace(config, cluster=replace(config.cluster, k=int(k_map[layer])))
        return config

    _log.info(
        "discretize: layers=%d n_items=%d n_jobs=%d backend=%s",
        len(layers),
        n_items,
        int(config.n_jobs),
        parallel_ctx.backend,
    )
    _update_progress(store, stage="starting", total_layers=len(layers), completed_layers=0, backend=parallel_ctx.backend)

    raw_results: List[_LayerProcessResult] = []
    if parallel_ctx.backend == "serial":
        for done, layer in enumerate(layers, start=1):
            result = _process_one_layer(
                working_provider,
                config=effective_config(layer),
                layer=layer,
                indices=worker_indices,
                n_items=n_items,
                store_root=store_path,
                return_arrays=True,
            )
            raw_results.append(result)
            _update_progress(store, stage="running", total_layers=len(layers), completed_layers=done, current_layer=layer, backend=parallel_ctx.backend)
            if config.progress:
                _log.info("completed layer %d (%d/%d)", layer, done, len(layers))
    elif parallel_ctx.backend == "thread":
        with ThreadPoolExecutor(max_workers=int(config.n_jobs)) as ex:
            futures = {
                ex.submit(
                    _process_one_layer,
                    working_provider,
                    config=effective_config(layer),
                    layer=layer,
                    indices=worker_indices,
                    n_items=n_items,
                    store_root=store_path,
                    return_arrays=True,
                ): layer
                for layer in layers
            }
            completed = 0
            for future in as_completed(futures):
                layer = futures[future]
                raw_results.append(future.result())
                completed += 1
                _update_progress(store, stage="running", total_layers=len(layers), completed_layers=completed, current_layer=layer, backend=parallel_ctx.backend)
                if config.progress:
                    _log.info("completed layer %d (%d/%d)", layer, completed, len(layers))
    else:
        assert parallel_ctx.memmap_path is not None and parallel_ctx.memmap_layer_ids is not None
        return_arrays = store_path is None
        with ProcessPoolExecutor(max_workers=int(config.n_jobs)) as ex:
            futures = {
                ex.submit(
                    _process_one_layer_from_memmap,
                    provider_path=str(parallel_ctx.memmap_path),
                    provider_layer_ids=list(parallel_ctx.memmap_layer_ids),
                    config_dict=effective_config(layer).to_dict(),
                    layer=int(layer),
                    indices=None if worker_indices is None else np.asarray(worker_indices, dtype=np.int64),
                    n_items=int(n_items),
                    store_root=store_path,
                    return_arrays=return_arrays,
                ): layer
                for layer in layers
            }
            completed = 0
            for future in as_completed(futures):
                layer = futures[future]
                raw_results.append(future.result())
                completed += 1
                _update_progress(store, stage="running", total_layers=len(layers), completed_layers=completed, current_layer=layer, backend=parallel_ctx.backend)
                if config.progress:
                    _log.info("completed layer %d (%d/%d)", layer, completed, len(layers))

    raw_results.sort(key=lambda item: item.layer)

    layer_results: List[LayerResult] = []
    centers_by_layer: List[np.ndarray] = []
    labels_by_layer: List[np.ndarray] = []
    for item in raw_results:
        labels = item.labels
        soft_labels = item.soft_labels
        if labels is None:
            assert store is not None
            d = store.layer_dir(item.layer)
            labels = np.load(d / "labels.npy", mmap_mode="r")
            soft_path = d / "soft_labels.npy"
            if soft_path.exists():
                soft_labels = np.load(soft_path, mmap_mode="r")
        layer_results.append(
            LayerResult(
                layer=int(item.layer),
                labels=np.asarray(labels),
                centers_hidden=f32(item.centers_hidden),
                n_clusters=int(item.n_clusters),
                transform_config=dict(item.transform_config),
                cluster_config=dict(item.cluster_config),
                soft_labels=None if soft_labels is None else np.asarray(soft_labels),
            )
        )
        centers_by_layer.append(f32(item.centers_hidden))
        labels_by_layer.append(np.asarray(labels, dtype=np.int32))

    alignment = None
    global_labels = None
    if len(layers) > 1:
        alignment = align_layers(centers_by_layer, layers=layers, spec=config.alignment)
        global_labels = build_global_labels(labels_by_layer, alignment.local_to_global)

    result = HSSResult(
        config=config,
        layer_results=layer_results,
        alignment=alignment,
        global_labels=global_labels,
        row_metadata=row_metadata,
    )

    if store is not None:
        atomic_json(store.config_path, result.config.to_dict())
        if result.alignment is not None:
            store._save_alignment(result.alignment)
        if result.global_labels is not None:
            atomic_npy(store.root / "global_labels.npy", result.global_labels.astype(np.int32))
        if result.row_metadata is not None:
            store._save_row_metadata(result.row_metadata)
        effective_k = {str(lr.layer): int(lr.n_clusters) for lr in result.layer_results}
        atomic_json(store.root / "effective_k_map.json", effective_k)
        _update_progress(store, stage="done", total_layers=len(layers), completed_layers=len(layers), backend=parallel_ctx.backend)

    if parallel_ctx.temp_dir is not None:
        parallel_ctx.temp_dir.cleanup()

    return result


def discretize_with_existing_models(
    provider: StateProvider,
    *,
    store_path: str | Path,
    layers: Optional[List[int]] = None,
    indices: Optional[np.ndarray] = None,
    align_spec: Optional[Any] = None,
    batch_size: int = 8192,
) -> HSSResult:
    store = ArtifactStore(root=Path(store_path))
    saved_config = HSSConfig()
    if store.config_path.exists():
        saved_config = HSSConfig.from_dict(read_json(store.config_path))

    if layers is None:
        layers = store.list_layers()
    indices = None if indices is None else i64(np.asarray(indices))
    n_items = provider.n_items() if indices is None else int(len(indices))

    row_metadata = None
    if saved_config.capture_row_metadata:
        row_metadata = _collect_row_metadata(
            provider,
            layers=layers,
            indices=indices,
            batch_size=max(int(batch_size), 1024),
        )

    layer_results: List[LayerResult] = []
    centers_by_layer: List[np.ndarray] = []
    labels_by_layer: List[np.ndarray] = []

    for layer in layers:
        chain, model = store.load_layer_model(layer)
        labels, soft_labels = _predict_all(
            provider,
            layer=layer,
            indices=indices,
            n_items=n_items,
            batch_size=int(batch_size),
            transform=chain,
            model=model,
            soft_labels_dtype=np.dtype(saved_config.soft_labels_dtype) if saved_config.store_soft_labels else None,
        )
        centers_hidden = chain.inverse(f32(model.centers()))
        layer_results.append(
            LayerResult(
                layer=int(layer),
                labels=labels,
                centers_hidden=centers_hidden,
                n_clusters=int(model.n_clusters()),
                transform_config=chain.config(),
                cluster_config=model.config(),
                soft_labels=soft_labels,
            )
        )
        centers_by_layer.append(centers_hidden)
        labels_by_layer.append(labels)

    alignment = None
    global_labels = None
    if len(layers) > 1:
        spec = saved_config.alignment if align_spec is None else align_spec
        alignment = align_layers(centers_by_layer, layers=layers, spec=spec)
        global_labels = build_global_labels(labels_by_layer, alignment.local_to_global)

    return HSSResult(
        config=saved_config,
        layer_results=layer_results,
        alignment=alignment,
        global_labels=global_labels,
        row_metadata=row_metadata,
    )

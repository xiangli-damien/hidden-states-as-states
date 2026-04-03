from __future__ import annotations

import inspect
import logging
import os
from contextlib import nullcontext
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, get_type_hints

import hss
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.ablation import AblationSuiteResult, run_ablation_suite
from experiments.config import AblationConfig, DataConfig, DiscretizeConfig, PredictionConfig, ScanConfig, SelectConfig, StabilityConfig, VisualizationConfig
from experiments.core import BaseArtifacts
from experiments.data import LoadedInputs, load_from_memmap, load_from_numpy, load_inputs
from experiments.discretize import DiscretizeResult, ScanResult, SelectResult, run_discretize, run_scan, run_select
from experiments.information import EffectiveRankResult, InformationResult
from experiments.io_utils import ExperimentStore
from experiments.k_analysis import KAnalysisResult, run_k_analysis, summarize_k_map
from experiments.label_analysis import LabelAnalysisResult
from experiments.prediction import PredictionResult, run_prediction
from experiments.preliminary import PreliminaryResult, run_preliminary_suite
from experiments.runtime import configure_logging
from experiments.stability import PersistenceResult, StabilitySuiteResult, TrendStabilityResult, run_stability_suite
from experiments.statistics import DynamicsResult, SequenceStatsResult
from experiments.visualization import VisualizationResult, run_visualization_suite

from .models import AnalysisArtifact, AnalysisContext, DataSession, DiscretizeArtifact, ScanArtifact, SelectArtifact
from .store import ArtifactManifest, WorkspaceStore, now_utc, read_json, to_jsonable


T = TypeVar('T')


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, 'to_dict'):
        return dict(config.to_dict())
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return dict(config.__dict__)
    raise TypeError(f'Unsupported config type: {type(config)}')


def _merge_config(config: Any, cls: Type[T], kwargs: Dict[str, Any]) -> T:
    if config is None:
        if hasattr(cls, 'from_dict'):
            return cls.from_dict(kwargs)
        return cls(**kwargs)
    if isinstance(config, cls):
        payload = _config_to_dict(config)
        payload.update(kwargs)
        if hasattr(cls, 'from_dict'):
            return cls.from_dict(payload)
        return cls(**payload)
    if isinstance(config, dict):
        payload = dict(config)
        payload.update(kwargs)
        if hasattr(cls, 'from_dict'):
            return cls.from_dict(payload)
        return cls(**payload)
    raise TypeError(f'Expected {cls.__name__} or dict, got {type(config)}')


def _empty_scan_result(layers: Sequence[int]) -> ScanResult:
    return ScanResult(metrics_df=pd.DataFrame(), layers=list(layers), config={})


def _empty_select_result(k_map: Dict[int, int]) -> SelectResult:
    return SelectResult(k_map={int(k): int(v) for k, v in k_map.items()}, strategy='attached', config={'strategy': 'attached'})


def _empty_k_analysis() -> KAnalysisResult:
    return KAnalysisResult(
        tolerance_df=pd.DataFrame(),
        rel_icl_surface=np.empty((0, 0), dtype=np.float64),
        rel_icl_layers=np.empty((0,), dtype=np.int32),
        rel_icl_k_values=np.empty((0,), dtype=np.int32),
    )


def _load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {str(k): np.asarray(payload[k]) for k in payload.files}


def _qualname(func: Callable[..., Any]) -> str:
    module = getattr(func, '__module__', '') or ''
    name = getattr(func, '__qualname__', getattr(func, '__name__', 'callable'))
    return f'{module}:{name}' if module else str(name)


def _summarize_output(result: Any) -> Dict[str, Any]:
    if result is None:
        return {'return_type': 'None'}
    if isinstance(result, pd.DataFrame):
        return {'return_type': 'DataFrame', 'n_rows': int(len(result)), 'n_cols': int(len(result.columns))}
    if isinstance(result, np.ndarray):
        return {'return_type': 'ndarray', 'shape': list(result.shape), 'dtype': str(result.dtype)}
    if isinstance(result, PredictionResult):
        return {'return_type': 'PredictionResult', 'n_methods': int(len(result.summary_df)), 'n_folds': int(len(result.fold_df))}
    if isinstance(result, PreliminaryResult):
        return {
            'return_type': 'PreliminaryResult',
            'n_nodes': int(len(result.dynamics.node_df)),
            'n_flows': int(len(result.dynamics.flow_df)),
            'has_information': result.information is not None,
            'has_label_analysis': result.label_analysis is not None,
        }
    if isinstance(result, StabilitySuiteResult):
        return {
            'return_type': 'StabilitySuiteResult',
            'n_layers': int(len(result.trend.layers)),
            'seed_distance_mean': float(np.mean(result.persistence.seed_distances)),
        }
    if isinstance(result, VisualizationResult):
        return {'return_type': 'VisualizationResult', 'n_paths': int(len(result.paths))}
    if isinstance(result, dict):
        return {'return_type': 'dict', 'keys': sorted(result.keys())}
    if isinstance(result, (list, tuple)):
        return {'return_type': type(result).__name__, 'length': len(result)}
    return {'return_type': type(result).__name__}


def _auto_save_output(store: ExperimentStore, result: Any) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if result is None:
        return paths
    if isinstance(result, pd.DataFrame):
        store.save_csv('result/result.csv', result)
        paths['result_csv'] = 'result/result.csv'
        return paths
    if isinstance(result, np.ndarray):
        store.save_npy('result/result.npy', result)
        paths['result_npy'] = 'result/result.npy'
        return paths
    if isinstance(result, (dict, list, tuple, str, int, float, bool)):
        store.save_json('result/result.json', result)
        paths['result_json'] = 'result/result.json'
        return paths
    return paths




def _thread_limit_context(n: int = 1):
    limit = max(1, int(n))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(limit)
    try:
        return threadpool_limits(limits=limit)
    except Exception:
        return nullcontext()

def _materialize_provider(provider: Any, path: Path, layers: Sequence[int]) -> None:
    layers = [int(x) for x in layers]
    n_items = int(provider.n_items())
    state_dim = int(provider.state_dim())
    arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n_items, len(layers), state_dim))
    for pos, layer in enumerate(layers):
        offset = 0
        for batch in provider.iter_batches(layer=int(layer), indices=None, batch_size=4096):
            states = np.asarray(batch.states, dtype=np.float32)
            n_batch = int(states.shape[0])
            arr[offset:offset + n_batch, pos, :] = states
            offset += n_batch
    del arr


def _provider_metadata_from_labels(label_dict: Dict[str, np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
    row_ids = None
    for key in ('item_id', 'row_id', 'row_ids', 'sentence_row', 'token_row', 'zarr_row'):
        if key in label_dict:
            row_ids = np.asarray(label_dict[key], dtype=np.int64)
            break
    sample_ids = None
    if 'sample_id' in label_dict:
        sample_ids = np.asarray(label_dict['sample_id'], dtype=np.int64)
    metadata_fields: Dict[str, np.ndarray] = {}
    for key, value in label_dict.items():
        if key in {'label', 'is_correct', 'correct', 'sample_id'}:
            continue
        arr = np.asarray(value)
        if arr.ndim == 1:
            metadata_fields[str(key)] = arr
    return row_ids, sample_ids, metadata_fields


def _reconstruct_loaded_from_manifest(manifest: Dict[str, Any], root: Path) -> LoadedInputs:
    states_rel = manifest.get('states_path', 'states.npy')
    states_path = root / states_rel
    layers = [int(x) for x in manifest.get('layers', [])]
    y_rel = manifest.get('y_path')
    y = np.load(root / y_rel, allow_pickle=False) if y_rel else np.array([], dtype=np.int32)
    labels_rel = manifest.get('labels_path')
    label_dict = _load_npz_dict(root / labels_rel) if labels_rel else {}
    if y.size == 0 and label_dict:
        key = str(manifest.get('positive_label_key', 'correct'))
        if key in label_dict:
            y = np.asarray(label_dict[key]).astype(np.int32)
    row_ids, sample_ids, metadata_fields = _provider_metadata_from_labels(label_dict)
    provider = hss.MemmapProvider(
        states_path,
        layer_ids=layers if layers else None,
        row_ids=row_ids,
        sample_ids=sample_ids,
        metadata_fields=metadata_fields,
        unit=str(manifest.get('unit', 'item')),
        provider_manifest=dict(manifest.get('provider_manifest', {})),
    )
    loaded = LoadedInputs(
        provider=provider,
        y=np.asarray(y).astype(np.int32, copy=False),
        label_dict=label_dict,
        layers=layers if layers else [int(x) for x in provider.layers()],
        n_items=int(provider.n_items()),
        state_dim=int(provider.state_dim()),
    )
    return loaded


class Workspace:
    def __init__(self, root: str | Path, *, log_level: str = 'INFO', quiet: bool = True, thread_limit: int = 1) -> None:
        self.root = Path(root)
        self.store = WorkspaceStore(self.root)
        self.root_store = ExperimentStore(self.root)
        self._data: Optional[DataSession] = None
        self.thread_limit = max(1, int(thread_limit))
        self.log_path = configure_logging(self.root_store, log_level, quiet)
        workspace_meta = self.store.load_workspace_meta()
        if not workspace_meta:
            self.store.save_workspace_meta({
                'schema_version': 2,
                'created_at': now_utc(),
                'root': str(self.root),
                'log_path': str(self.log_path),
            })
        self.logger = logging.getLogger(__name__)

    def load_data(
        self,
        states: Any = None,
        *,
        y: Optional[np.ndarray] = None,
        labels: Optional[Dict[str, np.ndarray]] = None,
        layers: Optional[Sequence[int]] = None,
        config: Optional[DataConfig] = None,
        positive_label_key: str = 'correct',
        persist: bool = True,
    ) -> DataSession:
        if isinstance(states, LoadedInputs):
            loaded = states
            origin = {'source': 'loaded_inputs'}
        elif config is not None:
            data_cfg = _merge_config(config, DataConfig, {})
            loaded = load_inputs(data_cfg)
            origin = {'source': 'config', 'config': data_cfg.to_dict()}
            if positive_label_key == 'correct' and getattr(data_cfg, 'positive_label_key', None):
                positive_label_key = data_cfg.positive_label_key
        elif isinstance(states, np.ndarray):
            label_dict = labels or ({positive_label_key: np.asarray(y)} if y is not None else {})
            provider, y_vec, label_dict = load_from_numpy(states, layer_ids=list(layers) if layers is not None else None, labels=label_dict, positive_label_key=positive_label_key)
            if y is not None:
                y_vec = np.asarray(y).astype(np.int32)
                label_dict = dict(label_dict)
                label_dict.setdefault(positive_label_key, y_vec)
            loaded = LoadedInputs(
                provider=provider,
                y=y_vec,
                label_dict=label_dict,
                layers=list(layers) if layers is not None else [int(x) for x in provider.layers()],
                n_items=int(provider.n_items()),
                state_dim=int(provider.state_dim()),
            )
            origin = {'source': 'numpy'}
        elif isinstance(states, (str, Path)):
            label_dict = labels or ({positive_label_key: np.asarray(y)} if y is not None else {})
            provider, y_vec, label_dict = load_from_memmap(states, layer_ids=list(layers) if layers is not None else None, labels=label_dict, positive_label_key=positive_label_key)
            if y is not None:
                y_vec = np.asarray(y).astype(np.int32)
                label_dict = dict(label_dict)
                label_dict.setdefault(positive_label_key, y_vec)
            loaded = LoadedInputs(
                provider=provider,
                y=y_vec,
                label_dict=label_dict,
                layers=list(layers) if layers is not None else [int(x) for x in provider.layers()],
                n_items=int(provider.n_items()),
                state_dim=int(provider.state_dim()),
            )
            origin = {'source': 'memmap_path', 'path': str(states)}
        elif states is not None:
            provider = states
            use_layers = list(layers) if layers is not None else [int(x) for x in provider.layers()]
            y_vec = np.asarray(y).astype(np.int32) if y is not None else np.array([], dtype=np.int32)
            label_dict = dict(labels or {})
            if y is not None:
                label_dict.setdefault(positive_label_key, y_vec)
            loaded = LoadedInputs(
                provider=provider,
                y=y_vec,
                label_dict=label_dict,
                layers=use_layers,
                n_items=int(provider.n_items()),
                state_dim=int(provider.state_dim()),
            )
            origin = {'source': 'provider'}
        else:
            manifest = self.store.load_data_manifest()
            if not manifest:
                raise ValueError('No input data or data manifest available')
            loaded = _reconstruct_loaded_from_manifest(manifest, self.store.data_dir)
            origin = {'source': 'manifest'}
            positive_label_key = str(manifest.get('positive_label_key', positive_label_key))
        unit = 'item'
        if hasattr(loaded.provider, 'unit') and callable(getattr(loaded.provider, 'unit')):
            try:
                unit = str(loaded.provider.unit())
            except Exception:
                unit = 'item'
        provider_manifest = {}
        if hasattr(loaded.provider, 'metadata_manifest') and callable(getattr(loaded.provider, 'metadata_manifest')):
            try:
                provider_manifest = dict(loaded.provider.metadata_manifest())
            except Exception:
                provider_manifest = {}
        manifest = {
            'source': origin.get('source', 'unknown'),
            'origin': to_jsonable(origin),
            'positive_label_key': positive_label_key,
            'layers': [int(x) for x in loaded.layers],
            'n_items': int(loaded.n_items),
            'state_dim': int(loaded.state_dim),
            'has_labels': bool(len(loaded.y) > 0),
            'positive_rate': float(np.mean(loaded.y)) if len(loaded.y) > 0 else None,
            'unit': unit,
            'provider_manifest': to_jsonable(provider_manifest),
            'saved_at': now_utc(),
        }
        if persist:
            states_path = self.store.data_dir / 'states.npy'
            _materialize_provider(loaded.provider, states_path, loaded.layers)
            manifest['states_path'] = 'states.npy'
            if len(loaded.y) > 0:
                np.save(self.store.data_dir / 'y.npy', loaded.y.astype(np.int32))
                manifest['y_path'] = 'y.npy'
            if loaded.label_dict:
                np.savez_compressed(self.store.data_dir / 'labels.npz', **{str(k): np.asarray(v) for k, v in loaded.label_dict.items()})
                manifest['labels_path'] = 'labels.npz'
            self.store.save_data_manifest(manifest)
        session = DataSession(loaded=loaded, manifest=manifest)
        self._data = session
        self.store.save_table_index()
        return session

    def open_data(self) -> DataSession:
        if self._data is not None:
            return self._data
        manifest = self.store.load_data_manifest()
        if not manifest:
            raise FileNotFoundError('Workspace has no persisted data manifest')
        loaded = _reconstruct_loaded_from_manifest(manifest, self.store.data_dir)
        self._data = DataSession(loaded=loaded, manifest=manifest)
        return self._data

    def has_data(self) -> bool:
        return self._data is not None or bool(self.store.load_data_manifest())

    def list_scans(self) -> List[str]:
        return self.store.list_names('scan')

    def list_selects(self) -> List[str]:
        return self.store.list_names('select')

    def list_discretizations(self) -> List[str]:
        return self.store.list_names('discretize')

    def list_analyses(self) -> List[str]:
        return self.store.list_names('analysis')

    def scan_table(self) -> pd.DataFrame:
        return self.store.manifest_frame('scan')

    def select_table(self) -> pd.DataFrame:
        return self.store.manifest_frame('select')

    def discretization_table(self) -> pd.DataFrame:
        return self.store.manifest_frame('discretize')

    def analysis_table(self) -> pd.DataFrame:
        return self.store.manifest_frame('analysis')

    def run_scan(
        self,
        name: str,
        *,
        config: Optional[ScanConfig | Dict[str, Any]] = None,
        layers: Optional[Sequence[int]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> ScanArtifact:
        data = self.open_data()
        scan_cfg = _merge_config(config, ScanConfig, kwargs)
        root = self.store.prepare_artifact('scan', name, overwrite=overwrite)
        artifact_store = ExperimentStore(root)
        manifest = ArtifactManifest(
            name=name,
            kind='scan',
            status='running',
            config=scan_cfg.to_dict(),
            refs={'data': 'data/manifest.json'},
            paths={'root': str(root)},
        )
        self.store.save_manifest('scan', name, manifest)
        with _thread_limit_context(self.thread_limit):
            result = run_scan(data.loaded.provider, scan_cfg, layers=list(layers) if layers is not None else list(data.layers), store=artifact_store)
        metrics_path = 'scan/scan_metrics.parquet' if artifact_store.exists('scan/scan_metrics.parquet') else 'scan/scan_metrics.csv'
        manifest.status = 'done'
        manifest.updated_at = now_utc()
        manifest.summary = {
            'n_rows': int(len(result.metrics_df)),
            'n_layers': int(len(result.layers)),
            'k_min': int(result.metrics_df['k'].min()) if len(result.metrics_df) else None,
            'k_max': int(result.metrics_df['k'].max()) if len(result.metrics_df) else None,
        }
        manifest.paths.update({'metrics': metrics_path, 'config_json': 'scan/scan_config.json'})
        self.store.save_manifest('scan', name, manifest)
        self.store.save_table_index()
        return self.load_scan(name)

    def load_scan(self, name: str) -> ScanArtifact:
        manifest = self.store.load_manifest('scan', name)
        root = self.store.artifact_root('scan', name)
        artifact_store = ExperimentStore(root)
        metrics_path = 'scan/scan_metrics.parquet' if artifact_store.exists('scan/scan_metrics.parquet') else 'scan/scan_metrics.csv'
        metrics_df = artifact_store.load_parquet(metrics_path)
        config = artifact_store.load_json('scan/scan_config.json') if artifact_store.exists('scan/scan_config.json') else manifest.config
        layers = manifest.summary.get('layers') if isinstance(manifest.summary.get('layers'), list) else sorted(metrics_df['layer'].unique().tolist()) if len(metrics_df) else []
        result = ScanResult(metrics_df=metrics_df, layers=[int(x) for x in layers], config=dict(config))
        return ScanArtifact(name=name, root=root, manifest=manifest, store=artifact_store, result=result)

    def run_select(
        self,
        name: str,
        *,
        scan: Optional[str] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        k_map: Optional[Dict[int, int]] = None,
        selector: Optional[Callable[[pd.DataFrame], Dict[int, int]]] = None,
        config: Optional[SelectConfig | Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> SelectArtifact:
        root = self.store.prepare_artifact('select', name, overwrite=overwrite)
        artifact_store = ExperimentStore(root)
        refs: Dict[str, Any] = {}
        if metrics_df is None and scan is not None:
            scan_artifact = self.load_scan(scan)
            metrics_df = scan_artifact.metrics_df
            refs['scan'] = scan
        elif metrics_df is None and k_map is None:
            raise ValueError('run_select requires scan, metrics_df, or k_map')
        manifest = ArtifactManifest(name=name, kind='select', status='running', refs=refs, paths={'root': str(root)})
        self.store.save_manifest('select', name, manifest)
        if k_map is not None:
            result = SelectResult(k_map={int(k): int(v) for k, v in k_map.items()}, strategy='manual', config={'strategy': 'manual'})
            artifact_store.save_json('select/k_map.json', result.k_map)
            artifact_store.save_json('select/select_config.json', result.config)
            k_analysis = None
        elif selector is not None:
            selected = selector(metrics_df)
            result = SelectResult(k_map={int(k): int(v) for k, v in selected.items()}, strategy='callable', config={'strategy': 'callable', 'selector': _qualname(selector)})
            artifact_store.save_json('select/k_map.json', result.k_map)
            artifact_store.save_json('select/select_config.json', result.config)
            k_analysis = None
            if metrics_df is not None:
                base_cfg = _merge_config(config, SelectConfig, kwargs)
                k_analysis = run_k_analysis(metrics_df, base_cfg, store=artifact_store)
        else:
            select_cfg = _merge_config(config, SelectConfig, kwargs)
            result = run_select(metrics_df, select_cfg, store=artifact_store)
            k_analysis = run_k_analysis(metrics_df, select_cfg, store=artifact_store)
        summary = summarize_k_map(result.k_map)
        summary['strategy'] = result.strategy
        artifact_store.save_json('select/summary.json', summary)
        manifest.status = 'done'
        manifest.updated_at = now_utc()
        manifest.config = dict(result.config)
        manifest.summary = summary
        manifest.paths.update({
            'k_map': 'select/k_map.json',
            'config_json': 'select/select_config.json',
            'summary_json': 'select/summary.json',
        })
        if artifact_store.exists('select/tolerance_sweep.csv'):
            manifest.paths['tolerance_sweep'] = 'select/tolerance_sweep.csv'
        if artifact_store.exists('scan/rel_icl_surface.npy'):
            manifest.paths['rel_icl_surface'] = 'scan/rel_icl_surface.npy'
        self.store.save_manifest('select', name, manifest)
        self.store.save_table_index()
        return self.load_select(name)

    def load_select(self, name: str) -> SelectArtifact:
        manifest = self.store.load_manifest('select', name)
        root = self.store.artifact_root('select', name)
        artifact_store = ExperimentStore(root)
        k_map_raw = artifact_store.load_json('select/k_map.json') if artifact_store.exists('select/k_map.json') else {}
        k_map = {int(k): int(v) for k, v in dict(k_map_raw).items()}
        config = artifact_store.load_json('select/select_config.json') if artifact_store.exists('select/select_config.json') else manifest.config
        strategy = str(config.get('strategy', manifest.config.get('strategy', 'unknown')))
        result = SelectResult(k_map=k_map, strategy=strategy, config=dict(config))
        k_analysis = None
        if artifact_store.exists('select/tolerance_sweep.csv') and artifact_store.exists('scan/rel_icl_surface.npy'):
            tolerance_df = artifact_store.load_csv('select/tolerance_sweep.csv')
            rel_icl_surface = artifact_store.load_npy('scan/rel_icl_surface.npy')
            rel_icl_layers = artifact_store.load_npy('scan/rel_icl_layers.npy') if artifact_store.exists('scan/rel_icl_layers.npy') else np.empty((0,), dtype=np.int32)
            rel_icl_k_values = artifact_store.load_npy('scan/rel_icl_k_values.npy') if artifact_store.exists('scan/rel_icl_k_values.npy') else np.empty((0,), dtype=np.int32)
            k_analysis = KAnalysisResult(
                tolerance_df=tolerance_df,
                rel_icl_surface=rel_icl_surface,
                rel_icl_layers=rel_icl_layers,
                rel_icl_k_values=rel_icl_k_values,
            )
        return SelectArtifact(name=name, root=root, manifest=manifest, store=artifact_store, result=result, k_analysis=k_analysis)

    def run_discretize(
        self,
        name: str,
        *,
        select: Optional[str] = None,
        k_map: Optional[Dict[int, int]] = None,
        config: Optional[DiscretizeConfig | Dict[str, Any]] = None,
        layers: Optional[Sequence[int]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> DiscretizeArtifact:
        data = self.open_data()
        disc_cfg = _merge_config(config, DiscretizeConfig, kwargs)
        refs: Dict[str, Any] = {'data': 'data/manifest.json'}
        requested_k_map = None
        if select is not None:
            select_artifact = self.load_select(select)
            requested_k_map = dict(select_artifact.k_map)
            refs['select'] = select
            if 'scan' in select_artifact.manifest.refs:
                refs['scan'] = select_artifact.manifest.refs['scan']
        elif k_map is not None:
            requested_k_map = {int(k): int(v) for k, v in k_map.items()}
        root = self.store.prepare_artifact('discretize', name, overwrite=overwrite)
        artifact_store = ExperimentStore(root)
        manifest = ArtifactManifest(
            name=name,
            kind='discretize',
            status='running',
            config=disc_cfg.to_dict(),
            refs=refs,
            paths={'root': str(root)},
            extra={'requested_k_map': requested_k_map},
        )
        self.store.save_manifest('discretize', name, manifest)
        if requested_k_map is not None:
            artifact_store.save_json('discretize/requested_k_map.json', requested_k_map)
        with _thread_limit_context(self.thread_limit):
            result = run_discretize(
                data.loaded.provider,
                disc_cfg,
                k_map=requested_k_map,
                layers=list(layers) if layers is not None else list(data.layers),
                store=artifact_store,
            )
        manifest.status = 'done'
        manifest.updated_at = now_utc()
        manifest.summary = {
            'n_items': int(result.n_items),
            'n_layers': int(len(result.layers)),
            'n_global_states': int(result.n_global_states),
            'k_summary': summarize_k_map(result.k_map),
        }
        manifest.paths.update({
            'config_json': 'discretize/discretize_config.json',
            'effective_k_map': 'discretize/effective_k_map.json',
            'summary_json': 'discretize/summary.json',
            'global_labels': 'discretize/global_labels.npy',
            'hss_store': 'discretize/hss_artifacts',
        })
        self.store.save_manifest('discretize', name, manifest)
        self.store.save_table_index()
        return self.load_discretize(name)

    def load_discretize(self, name: str) -> DiscretizeArtifact:
        manifest = self.store.load_manifest('discretize', name)
        root = self.store.artifact_root('discretize', name)
        artifact_store = ExperimentStore(root)
        hss_store = hss.ArtifactStore(root / 'discretize' / 'hss_artifacts')
        hss_result = hss_store.load()
        effective_k_map_raw = artifact_store.load_json('discretize/effective_k_map.json') if artifact_store.exists('discretize/effective_k_map.json') else {}
        effective_k_map = {int(k): int(v) for k, v in dict(effective_k_map_raw).items()}
        config = artifact_store.load_json('discretize/discretize_config.json') if artifact_store.exists('discretize/discretize_config.json') else manifest.config
        layers = [int(lr.layer) for lr in hss_result.layer_results]
        n_items = int(hss_result.global_labels.shape[0]) if hss_result.global_labels is not None else int(hss_result.layer_results[0].labels.shape[0])
        n_global_states = int(hss_result.alignment.n_global_states) if hss_result.alignment is not None else 0
        result = DiscretizeResult(
            hss_result=hss_result,
            k_map=effective_k_map,
            layers=layers,
            n_items=n_items,
            n_global_states=n_global_states,
            config=dict(config),
        )
        requested_k_map = None
        if artifact_store.exists('discretize/requested_k_map.json'):
            payload = artifact_store.load_json('discretize/requested_k_map.json')
            requested_k_map = {int(k): int(v) for k, v in dict(payload).items()}
        return DiscretizeArtifact(name=name, root=root, manifest=manifest, store=artifact_store, result=result, requested_k_map=requested_k_map)

    def _resolve_refs(self, *, discretize: Optional[str], select: Optional[str], scan: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        disc_name = discretize
        select_name = select
        scan_name = scan
        if disc_name is not None:
            disc_manifest = self.store.load_manifest('discretize', disc_name)
            if select_name is None:
                select_name = disc_manifest.refs.get('select')
            if scan_name is None:
                scan_name = disc_manifest.refs.get('scan')
        if select_name is not None and scan_name is None:
            select_manifest = self.store.load_manifest('select', select_name)
            scan_name = select_manifest.refs.get('scan')
        return disc_name, select_name, scan_name

    def build_base(self, *, discretize: str, select: Optional[str] = None, scan: Optional[str] = None) -> BaseArtifacts:
        data = self.open_data()
        disc_name, select_name, scan_name = self._resolve_refs(discretize=discretize, select=select, scan=scan)
        discretize_artifact = self.load_discretize(disc_name)
        if select_name is not None:
            select_artifact = self.load_select(select_name)
        else:
            select_artifact = SelectArtifact(
                name='attached',
                root=Path('.'),
                manifest=ArtifactManifest(name='attached', kind='select', config={'strategy': 'attached'}),
                store=ExperimentStore(self.root),
                result=_empty_select_result(discretize_artifact.requested_k_map or discretize_artifact.k_map),
                k_analysis=None,
            )
        if scan_name is not None:
            scan_artifact = self.load_scan(scan_name)
        else:
            scan_artifact = ScanArtifact(
                name='attached',
                root=Path('.'),
                manifest=ArtifactManifest(name='attached', kind='scan'),
                store=ExperimentStore(self.root),
                result=_empty_scan_result(data.layers),
            )
        k_analysis = select_artifact.k_analysis
        if k_analysis is None and len(scan_artifact.metrics_df) > 0:
            cfg = SelectConfig.from_dict(select_artifact.result.config) if select_artifact.result.config and select_artifact.result.config.get('strategy') not in {'manual', 'callable', 'attached'} else None
            if cfg is not None:
                k_analysis = run_k_analysis(scan_artifact.metrics_df, cfg)
        if k_analysis is None:
            k_analysis = _empty_k_analysis()
        return BaseArtifacts(
            loaded=data.loaded,
            scan=scan_artifact.result,
            select=select_artifact.result,
            discretize=discretize_artifact.result,
            k_analysis=k_analysis,
            layers=list(discretize_artifact.layers),
        )

    def analysis_store(self, name: str) -> ExperimentStore:
        return ExperimentStore(self.store.artifact_root('analysis', name))

    def _prepare_context(
        self,
        *,
        name: str,
        analysis_type: Optional[str],
        store: ExperimentStore,
        discretize: Optional[str],
        select: Optional[str],
        scan: Optional[str],
        build_base: bool,
    ) -> AnalysisContext:
        data = self.open_data() if self.has_data() else None
        disc_name, select_name, scan_name = self._resolve_refs(discretize=discretize, select=select, scan=scan)
        scan_artifact = self.load_scan(scan_name) if scan_name is not None else None
        select_artifact = self.load_select(select_name) if select_name is not None else None
        discretize_artifact = self.load_discretize(disc_name) if disc_name is not None else None
        base = None
        if build_base and discretize_artifact is not None and data is not None:
            base = self.build_base(discretize=disc_name, select=select_name, scan=scan_name)
        return AnalysisContext(
            workspace=self,
            store=store,
            data=data,
            scan=scan_artifact,
            select=select_artifact,
            discretize=discretize_artifact,
            base=base,
            analysis_name=name,
            analysis_type=analysis_type,
            refs={'scan': scan_name, 'select': select_name, 'discretize': disc_name},
        )

    def _instantiate_cfg_from_function(self, func: Callable[..., Any], extra_kwargs: Dict[str, Any], bound: Dict[str, Any]) -> Dict[str, Any]:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if name in bound:
                continue
            ann = hints.get(name, param.annotation)
            if ann is inspect._empty:
                continue
            if hasattr(ann, 'from_dict') and isinstance(extra_kwargs, dict):
                field_names = getattr(ann, '__dataclass_fields__', {})
                if field_names:
                    matching = {k: extra_kwargs[k] for k in list(extra_kwargs.keys()) if k in field_names}
                    if matching:
                        bound[name] = ann.from_dict(matching)
                        for k in matching:
                            extra_kwargs.pop(k, None)
            if name in {'cfg', 'config'} and name not in bound and 'config' in extra_kwargs:
                bound[name] = extra_kwargs.pop('config')
        return bound

    def _invoke(self, func: Callable[..., Any], context: AnalysisContext, config: Any, extra_kwargs: Dict[str, Any]) -> Any:
        sig = inspect.signature(func)
        kwargs = dict(extra_kwargs)
        values: Dict[str, Any] = {
            'context': context,
            'ctx': context,
            'workspace': self,
            'ws': self,
            'store': context.store,
            'data': None if context.data is None else context.data.loaded,
            'loaded': None if context.data is None else context.data.loaded,
            'provider': context.provider,
            'state_provider': context.provider,
            'y': context.y,
            'label_dict': context.label_dict,
            'labels': context.label_dict,
            'layers': context.layers,
            'scan': None if context.scan is None else context.scan.result,
            'scan_result': None if context.scan is None else context.scan.result,
            'metrics_df': None if context.scan is None else context.scan.metrics_df,
            'select': None if context.select is None else context.select.result,
            'select_result': None if context.select is None else context.select.result,
            'k_map': None if context.select is None else context.select.k_map if context.select is not None else None,
            'discretize': None if context.discretize is None else context.discretize.result,
            'discretize_result': None if context.discretize is None else context.discretize.result,
            'hss_result': context.hss_result,
            'global_labels': context.global_labels,
            'base': context.base,
        }
        bound: Dict[str, Any] = {}
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for name, param in sig.parameters.items():
            if name in kwargs:
                bound[name] = kwargs.pop(name)
                continue
            if name in values and values[name] is not None:
                bound[name] = values[name]
                continue
            if name in {'cfg', 'config'} and config is not None:
                bound[name] = config
                continue
        bound = self._instantiate_cfg_from_function(func, kwargs, bound)
        missing = [name for name, param in sig.parameters.items() if param.default is inspect._empty and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD} and name not in bound]
        if missing:
            raise TypeError(f'Could not bind required arguments for {_qualname(func)}: {missing}')
        if accepts_kwargs:
            bound.update(kwargs)
        elif kwargs:
            unknown = sorted(kwargs.keys())
            raise TypeError(f'Unexpected extra kwargs for {_qualname(func)}: {unknown}')
        return func(**bound)

    def run_analysis(
        self,
        name: str,
        *,
        func: Callable[..., Any],
        discretize: Optional[str] = None,
        select: Optional[str] = None,
        scan: Optional[str] = None,
        config: Any = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        analysis_type: Optional[str] = None,
        overwrite: bool = False,
        build_base: bool = False,
    ) -> AnalysisArtifact:
        root = self.store.prepare_artifact('analysis', name, overwrite=overwrite)
        artifact_store = ExperimentStore(root)
        refs = {'scan': scan, 'select': select, 'discretize': discretize}
        manifest = ArtifactManifest(
            name=name,
            kind='analysis',
            status='running',
            analysis_type=analysis_type or getattr(func, '__name__', 'analysis'),
            config=_config_to_dict(config),
            refs=refs,
            paths={'root': str(root)},
            extra={'callable': _qualname(func), 'extra_kwargs': to_jsonable(extra_kwargs or {})},
        )
        self.store.save_manifest('analysis', name, manifest)
        context = self._prepare_context(name=name, analysis_type=manifest.analysis_type, store=artifact_store, discretize=discretize, select=select, scan=scan, build_base=build_base)
        with _thread_limit_context(self.thread_limit):
            result = self._invoke(func, context, config, dict(extra_kwargs or {}))
        auto_paths = _auto_save_output(artifact_store, result)
        manifest.status = 'done'
        manifest.updated_at = now_utc()
        manifest.summary = _summarize_output(result)
        manifest.paths.update(auto_paths)
        self.store.save_manifest('analysis', name, manifest)
        self.store.save_table_index()
        return AnalysisArtifact(name=name, root=root, manifest=manifest, store=artifact_store, result=result)

    def run_preliminary(self, name: str, *, discretize: str, select: Optional[str] = None, scan: Optional[str] = None, overwrite: bool = False) -> AnalysisArtifact:
        return self.run_analysis(name, func=_run_preliminary_builtin, discretize=discretize, select=select, scan=scan, analysis_type='preliminary', overwrite=overwrite, build_base=True)

    def run_prediction(
        self,
        name: str,
        *,
        discretize: str,
        config: Optional[PredictionConfig | Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> AnalysisArtifact:
        prediction_cfg = _merge_config(config, PredictionConfig, kwargs)
        return self.run_analysis(name, func=_run_prediction_builtin, discretize=discretize, config=prediction_cfg, analysis_type='prediction', overwrite=overwrite)

    def run_stability(
        self,
        name: str,
        *,
        discretize: str,
        select: Optional[str] = None,
        scan: Optional[str] = None,
        discretize_config: Optional[DiscretizeConfig | Dict[str, Any]] = None,
        stability_config: Optional[StabilityConfig | Dict[str, Any]] = None,
        overwrite: bool = False,
        disc_kwargs: Optional[Dict[str, Any]] = None,
        stab_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AnalysisArtifact:
        disc_cfg = _merge_config(discretize_config, DiscretizeConfig, disc_kwargs or {})
        stab_cfg = _merge_config(stability_config, StabilityConfig, stab_kwargs or {})
        return self.run_analysis(
            name,
            func=_run_stability_builtin,
            discretize=discretize,
            select=select,
            scan=scan,
            config={'discretize': disc_cfg.to_dict(), 'stability': stab_cfg.to_dict()},
            extra_kwargs={'disc_cfg': disc_cfg, 'stab_cfg': stab_cfg},
            analysis_type='stability',
            overwrite=overwrite,
            build_base=True,
        )

    def run_ablation(
        self,
        name: str,
        *,
        discretize: str,
        select: Optional[str] = None,
        scan: Optional[str] = None,
        discretize_config: Optional[DiscretizeConfig | Dict[str, Any]] = None,
        ablation_config: Optional[AblationConfig | Dict[str, Any]] = None,
        token_aggregation_providers: Optional[Dict[str, object]] = None,
        overwrite: bool = False,
        disc_kwargs: Optional[Dict[str, Any]] = None,
        abl_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AnalysisArtifact:
        disc_cfg = _merge_config(discretize_config, DiscretizeConfig, disc_kwargs or {})
        abl_cfg = _merge_config(ablation_config, AblationConfig, abl_kwargs or {})
        return self.run_analysis(
            name,
            func=_run_ablation_builtin,
            discretize=discretize,
            select=select,
            scan=scan,
            config={'discretize': disc_cfg.to_dict(), 'ablation': abl_cfg.to_dict()},
            extra_kwargs={'disc_cfg': disc_cfg, 'abl_cfg': abl_cfg, 'token_aggregation_providers': token_aggregation_providers},
            analysis_type='ablation',
            overwrite=overwrite,
            build_base=True,
        )

    def run_visualization(
        self,
        name: str,
        *,
        discretize: str,
        select: Optional[str] = None,
        scan: Optional[str] = None,
        preliminary: Optional[str] = None,
        prediction: Optional[str] = None,
        stability: Optional[str] = None,
        ablation: Optional[str] = None,
        config: Optional[VisualizationConfig | Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> AnalysisArtifact:
        viz_cfg = _merge_config(config, VisualizationConfig, kwargs)
        extra_kwargs = {
            'viz_cfg': viz_cfg,
            'preliminary_name': preliminary,
            'prediction_name': prediction,
            'stability_name': stability,
            'ablation_name': ablation,
        }
        return self.run_analysis(
            name,
            func=_run_visualization_builtin,
            discretize=discretize,
            select=select,
            scan=scan,
            config=viz_cfg,
            extra_kwargs=extra_kwargs,
            analysis_type='visualization',
            overwrite=overwrite,
            build_base=True,
        )

    def load_prediction(self, name: str) -> PredictionResult:
        artifact = self._load_analysis_record(name)
        summary_df = artifact.store.load_csv('prediction/summary.csv') if artifact.store.exists('prediction/summary.csv') else pd.DataFrame()
        fold_df = artifact.store.load_csv('prediction/fold_results.csv') if artifact.store.exists('prediction/fold_results.csv') else pd.DataFrame()
        config = artifact.store.load_json('prediction/prediction_config.json') if artifact.store.exists('prediction/prediction_config.json') else artifact.manifest.config
        return PredictionResult(summary_df=summary_df, fold_df=fold_df, config=dict(config))

    def load_preliminary(self, name: str) -> PreliminaryResult:
        artifact = self._load_analysis_record(name)
        disc_name = artifact.manifest.refs.get('discretize')
        layers = self.load_discretize(disc_name).layers if disc_name is not None else []
        node_df = artifact.store.load_csv('geometry/node_df.csv') if artifact.store.exists('geometry/node_df.csv') else pd.DataFrame()
        flow_df = artifact.store.load_csv('geometry/flow_df.csv') if artifact.store.exists('geometry/flow_df.csv') else pd.DataFrame()
        events_df = artifact.store.load_csv('geometry/events_df.csv') if artifact.store.exists('geometry/events_df.csv') else pd.DataFrame()
        self_trans = artifact.store.load_npy('geometry/self_transition.npy') if artifact.store.exists('geometry/self_transition.npy') else np.empty((0,), dtype=np.float64)
        active_per_layer = artifact.store.load_npy('geometry/active_per_layer.npy') if artifact.store.exists('geometry/active_per_layer.npy') else np.empty((0,), dtype=np.int32)
        dynamics = DynamicsResult(node_df=node_df, flow_df=flow_df, events_df=events_df, self_trans=self_trans, active_per_layer=active_per_layer, layers=list(layers))
        ham = artifact.store.load_npy('geometry/hamming_similarity.npy') if artifact.store.exists('geometry/hamming_similarity.npy') else None
        marginal_freq = artifact.store.load_npy('geometry/marginal_freq.npy') if artifact.store.exists('geometry/marginal_freq.npy') else np.empty((0,), dtype=np.float64)
        seq_meta = artifact.store.load_json('geometry/sequence_stats.json') if artifact.store.exists('geometry/sequence_stats.json') else {}
        sequence_stats = SequenceStatsResult(
            hamming_sim=ham,
            marginal_freq=marginal_freq,
            self_trans_mean=float(seq_meta.get('self_trans_mean', np.nan)),
            active_mean=float(seq_meta.get('active_mean', np.nan)),
        )
        information = None
        if artifact.store.exists('information/information_analysis.csv'):
            information = InformationResult(df=artifact.store.load_csv('information/information_analysis.csv'), layers=list(layers))
        effective_rank = EffectiveRankResult(
            df=artifact.store.load_csv('information/effective_rank.csv') if artifact.store.exists('information/effective_rank.csv') else pd.DataFrame(),
            layers=list(layers),
        )
        label_analysis = None
        if artifact.store.exists('geometry/delta_matrix.npy') and artifact.store.exists('geometry/lifecycle.json'):
            lifecycle = artifact.store.load_json('geometry/lifecycle.json')
            label_analysis = LabelAnalysisResult(
                delta_matrix=artifact.store.load_npy('geometry/delta_matrix.npy'),
                pos_rate_matrix=artifact.store.load_npy('geometry/pos_rate_matrix.npy'),
                count_matrix=artifact.store.load_npy('geometry/count_matrix.npy'),
                layers=list(layers),
                cluster_order=[int(x) for x in lifecycle.get('cluster_order', [])],
                baseline=float(lifecycle.get('baseline', np.nan)),
                birth={int(k): int(v) for k, v in dict(lifecycle.get('birth', {})).items()},
                death={int(k): int(v) for k, v in dict(lifecycle.get('death', {})).items()},
            )
        sink_states = artifact.store.load_csv('geometry/sink_states.csv') if artifact.store.exists('geometry/sink_states.csv') else pd.DataFrame()
        return PreliminaryResult(
            dynamics=dynamics,
            sequence_stats=sequence_stats,
            information=information,
            effective_rank=effective_rank,
            label_analysis=label_analysis,
            sink_states=sink_states,
        )

    def load_stability(self, name: str) -> StabilitySuiteResult:
        artifact = self._load_analysis_record(name)
        layers = artifact.store.load_npy('stability/layers.npy') if artifact.store.exists('stability/layers.npy') else np.empty((0,), dtype=np.int32)
        seed_k_curves = artifact.store.load_npy('stability/seed_k_curves.npy') if artifact.store.exists('stability/seed_k_curves.npy') else None
        subsample_k_curves = None
        if artifact.store.exists('stability/subsample_k_curves.npz'):
            payload = _load_npz_dict(artifact.root / 'stability' / 'subsample_k_curves.npz')
            subsample_k_curves = {float(k): np.asarray(v) for k, v in payload.items()}
        kmax_k_curves = None
        if artifact.store.exists('stability/kmax_k_curves.npz'):
            payload = _load_npz_dict(artifact.root / 'stability' / 'kmax_k_curves.npz')
            kmax_k_curves = {int(k): np.asarray(v) for k, v in payload.items()}
        trend = TrendStabilityResult(
            baseline_k_curve=artifact.store.load_npy('stability/baseline_k_curve.npy') if artifact.store.exists('stability/baseline_k_curve.npy') else np.empty((0,), dtype=np.int32),
            seed_k_curves=seed_k_curves,
            subsample_k_curves=subsample_k_curves,
            kmax_k_curves=kmax_k_curves,
            layers=layers,
        )
        sample_distances = artifact.store.load_npy('stability/sample_distances.npy') if artifact.store.exists('stability/sample_distances.npy') else None
        persistence = PersistenceResult(
            seed_distances=artifact.store.load_npy('stability/seed_distances.npy') if artifact.store.exists('stability/seed_distances.npy') else np.empty((0, 0), dtype=np.float64),
            sample_distances=sample_distances,
            random_distances=artifact.store.load_npy('stability/random_distances.npy') if artifact.store.exists('stability/random_distances.npy') else np.empty((0, 0), dtype=np.float64),
            layers=layers,
        )
        return StabilitySuiteResult(trend=trend, persistence=persistence)

    def load_visualization(self, name: str) -> VisualizationResult:
        artifact = self._load_analysis_record(name)
        manifest = artifact.store.load_json('figures/manifest.json') if artifact.store.exists('figures/manifest.json') else {}
        return VisualizationResult(paths={str(k): Path(v) for k, v in dict(manifest).items()})

    def _load_analysis_record(self, name: str) -> AnalysisArtifact:
        manifest = self.store.load_manifest('analysis', name)
        root = self.store.artifact_root('analysis', name)
        artifact_store = ExperimentStore(root)
        return AnalysisArtifact(name=name, root=root, manifest=manifest, store=artifact_store, result=None)

    def load_analysis(self, name: str, *, typed: bool = True) -> Any:
        record = self._load_analysis_record(name)
        if not typed:
            return record
        analysis_type = str(record.manifest.analysis_type or '')
        if analysis_type == 'prediction':
            return self.load_prediction(name)
        if analysis_type == 'preliminary':
            return self.load_preliminary(name)
        if analysis_type == 'stability':
            return self.load_stability(name)
        if analysis_type == 'visualization':
            return self.load_visualization(name)
        return record

    def save_artifact(self, relative_path: str | Path, obj: Any) -> Path:
        return self.store.save_any(relative_path, obj)

    def load_artifact(self, relative_path: str | Path) -> Any:
        return self.store.load_any(relative_path)

    def run_standard_bundle(
        self,
        *,
        scan_name: str = 'scan_default',
        select_name: str = 'select_default',
        discretize_name: str = 'discretize_default',
        preliminary_name: Optional[str] = 'preliminary_default',
        prediction_name: Optional[str] = 'prediction_default',
        stability_name: Optional[str] = None,
        ablation_name: Optional[str] = None,
        visualization_name: Optional[str] = None,
        scan_config: Optional[ScanConfig | Dict[str, Any]] = None,
        select_config: Optional[SelectConfig | Dict[str, Any]] = None,
        discretize_config: Optional[DiscretizeConfig | Dict[str, Any]] = None,
        prediction_config: Optional[PredictionConfig | Dict[str, Any]] = None,
        stability_config: Optional[StabilityConfig | Dict[str, Any]] = None,
        ablation_config: Optional[AblationConfig | Dict[str, Any]] = None,
        visualization_config: Optional[VisualizationConfig | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scan_artifact = self.run_scan(scan_name, config=scan_config)
        select_artifact = self.run_select(select_name, scan=scan_artifact.name, config=select_config)
        discretize_artifact = self.run_discretize(discretize_name, select=select_artifact.name, config=discretize_config)
        outputs: Dict[str, Any] = {
            'scan': scan_artifact,
            'select': select_artifact,
            'discretize': discretize_artifact,
        }
        if preliminary_name is not None:
            outputs['preliminary'] = self.run_preliminary(preliminary_name, discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name)
        if prediction_name is not None and len(self.open_data().y) > 0:
            outputs['prediction'] = self.run_prediction(prediction_name, discretize=discretize_artifact.name, config=prediction_config)
        if stability_name is not None:
            outputs['stability'] = self.run_stability(stability_name, discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, discretize_config=discretize_config, stability_config=stability_config)
        if ablation_name is not None:
            outputs['ablation'] = self.run_ablation(ablation_name, discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, discretize_config=discretize_config, ablation_config=ablation_config)
        if visualization_name is not None:
            outputs['visualization'] = self.run_visualization(
                visualization_name,
                discretize=discretize_artifact.name,
                select=select_artifact.name,
                scan=scan_artifact.name,
                preliminary=preliminary_name,
                prediction=prediction_name,
                stability=stability_name,
                ablation=ablation_name,
                config=visualization_config,
            )
        return outputs


def _run_preliminary_builtin(context: AnalysisContext, store: ExperimentStore) -> PreliminaryResult:
    if context.base is None:
        raise ValueError('Preliminary analysis requires a resolved base artifact set')
    return run_preliminary_suite(context.base, store=store)


def _run_prediction_builtin(context: AnalysisContext, store: ExperimentStore, cfg: PredictionConfig) -> PredictionResult:
    if context.discretize is None or context.data is None:
        raise ValueError('Prediction requires data and discretize artifacts')
    return run_prediction(context.provider, context.global_labels, context.y, context.layers, cfg, store=store)


def _run_stability_builtin(context: AnalysisContext, store: ExperimentStore, disc_cfg: DiscretizeConfig, stab_cfg: StabilityConfig) -> StabilitySuiteResult:
    if context.base is None:
        raise ValueError('Stability requires a resolved base artifact set')
    return run_stability_suite(context.base, disc_cfg, stab_cfg, store=store)


def _run_ablation_builtin(
    context: AnalysisContext,
    store: ExperimentStore,
    disc_cfg: DiscretizeConfig,
    abl_cfg: AblationConfig,
    token_aggregation_providers: Optional[Dict[str, object]] = None,
) -> AblationSuiteResult:
    if context.base is None:
        raise ValueError('Ablation requires a resolved base artifact set')
    return run_ablation_suite(context.base, disc_cfg, abl_cfg, token_aggregation_providers=token_aggregation_providers, store=store)


def _run_visualization_builtin(
    context: AnalysisContext,
    store: ExperimentStore,
    viz_cfg: VisualizationConfig,
    preliminary_name: Optional[str] = None,
    prediction_name: Optional[str] = None,
    stability_name: Optional[str] = None,
    ablation_name: Optional[str] = None,
) -> VisualizationResult:
    if context.base is None:
        raise ValueError('Visualization requires a resolved base artifact set')
    workspace: Workspace = context.workspace
    preliminary = workspace.load_preliminary(preliminary_name) if preliminary_name is not None else None
    prediction = workspace.load_prediction(prediction_name) if prediction_name is not None else None
    stability = workspace.load_stability(stability_name) if stability_name is not None else None
    ablation = workspace.load_analysis(ablation_name, typed=False) if ablation_name is not None else None
    return run_visualization_suite(context.base, viz_cfg, preliminary=preliminary, prediction=prediction, stability=stability, ablation=ablation, store=store)


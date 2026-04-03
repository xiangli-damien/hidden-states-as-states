
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hss
import numpy as np
import pandas as pd
from hss.types import StateProvider

from .config import DiscretizeConfig, ScanConfig, SelectConfig
from .io_utils import ExperimentStore

_log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    metrics_df: pd.DataFrame
    layers: List[int]
    config: Dict[str, Any]


@dataclass
class SelectResult:
    k_map: Dict[int, int]
    strategy: str
    config: Dict[str, Any]


@dataclass
class DiscretizeResult:
    hss_result: hss.HSSResult
    k_map: Dict[int, int]
    layers: List[int]
    n_items: int
    n_global_states: int
    config: Dict[str, Any]


def _cluster_params_from_cfg(cfg: Any) -> Dict[str, Any]:
    return {
        'covariance_type': cfg.covariance_type,
        'reg_covar': cfg.reg_covar,
        'backend': cfg.gmm_backend,
        'adaptive_reg': cfg.gmm_adaptive_reg,
        'adaptive_alpha': cfg.gmm_adaptive_alpha,
        'adaptive_min': cfg.gmm_adaptive_min,
        'tol': cfg.gmm_tol,
        'chunk_size': cfg.gmm_chunk_size,
        'device': cfg.gmm_device,
        'init_method': cfg.gmm_init_method,
        'auto_gpu_threshold': cfg.gmm_auto_gpu_threshold,
        'spectral_affinity': cfg.gmm_spectral_affinity,
        'spectral_n_neighbors': cfg.gmm_spectral_n_neighbors,
        'spectral_gamma': cfg.gmm_spectral_gamma,
    }


def _build_hss_config(
    cfg: DiscretizeConfig, layers: Optional[List[int]] = None
) -> hss.HSSConfig:
    return hss.HSSConfig(
        seed=cfg.seed,
        layers=layers,
        transform=hss.TransformSpec(
            steps=[hss.TransformStepSpec(**step) for step in cfg.transform_steps]
        ),
        cluster=hss.ClusterSpec(
            method=cfg.method,
            params=_cluster_params_from_cfg(cfg),
        ),
        alignment=hss.AlignSpec(
            similarity=cfg.alignment_similarity,
            method=cfg.alignment_method,
            threshold=cfg.alignment_threshold,
        ),
        batch_size_fit=cfg.batch_size_fit,
        batch_size_predict=cfg.batch_size_predict,
        n_jobs=cfg.n_jobs,
        parallel_backend=cfg.parallel_backend,
        parsimony_tolerance=cfg.parsimony_tolerance,
        icl_mode=cfg.icl_mode,
    )


def run_scan(
    provider: StateProvider,
    cfg: ScanConfig,
    *,
    layers: Optional[List[int]] = None,
    store: Optional[ExperimentStore] = None,
) -> ScanResult:
    scan_cfg = hss.ScanConfig(
        k_range=cfg.k_range,
        seed=cfg.seed,
        stability_repeats=cfg.stability_repeats,
        eval_sample=cfg.eval_sample,
        icl_mode=cfg.icl_mode,
        transform=hss.TransformSpec(
            steps=[hss.TransformStepSpec(**step) for step in cfg.transform_steps]
        ),
        cluster=hss.ClusterSpec(
            method=cfg.method,
            params=_cluster_params_from_cfg(cfg),
        ),
        batch_size=cfg.batch_size,
        n_jobs=cfg.n_jobs,
        parallel_backend=cfg.parallel_backend,
    )
    if layers is None:
        layers = [int(layer) for layer in provider.layers()]
    metrics_df = hss.scan(provider, config=scan_cfg, layers=layers)
    if store is not None:
        store.save_parquet('scan/scan_metrics.parquet', metrics_df)
        store.save_json('scan/scan_config.json', cfg.to_dict())
        if not metrics_df.empty:
            store.save_json(
                'scan/summary.json',
                {
                    'n_rows': int(len(metrics_df)),
                    'n_layers': int(metrics_df['layer'].nunique()),
                    'k_min': int(metrics_df['k'].min()),
                    'k_max': int(metrics_df['k'].max()),
                },
            )
    return ScanResult(metrics_df=metrics_df, layers=layers, config=cfg.to_dict())


def run_select(
    metrics_df: pd.DataFrame,
    cfg: SelectConfig,
    *,
    store: Optional[ExperimentStore] = None,
) -> SelectResult:
    if cfg.strategy == 'icl_parsimonious':
        k_map = hss.select_icl_parsimonious(
            metrics_df,
            relative_pct=cfg.relative_pct,
            stability_min=cfg.stability_min,
        )
    elif cfg.strategy == 'sil_stable':
        k_map = hss.select_sil_stable(metrics_df, stability_min=cfg.stability_min)
    elif cfg.strategy == 'auto_target':
        k_map = hss.select_auto_target_k(
            metrics_df,
            target_avg_k=cfg.target_avg_k,
            stability_min=cfg.stability_min,
            candidate_pcts=cfg.candidate_pcts,
        )
    else:
        raise ValueError(f'Unknown strategy: {cfg.strategy}')
    if store is not None:
        store.save_json('select/k_map.json', k_map)
        store.save_json('select/select_config.json', cfg.to_dict())
        if k_map:
            ks = list(k_map.values())
            store.save_json(
                'select/summary.json',
                {
                    'strategy': cfg.strategy,
                    'n_layers': len(k_map),
                    'k_mean': float(np.mean(ks)),
                    'k_std': float(np.std(ks)),
                    'k_min': int(np.min(ks)),
                    'k_max': int(np.max(ks)),
                },
            )
    return SelectResult(k_map=k_map, strategy=cfg.strategy, config=cfg.to_dict())


def run_discretize(
    provider: StateProvider,
    cfg: DiscretizeConfig,
    *,
    k_map: Optional[Dict[int, int]] = None,
    layers: Optional[List[int]] = None,
    store: Optional[ExperimentStore] = None,
) -> DiscretizeResult:
    hss_config = _build_hss_config(cfg, layers)
    store_path = None
    if store is not None:
        store_path = str(store.discretize_dir / 'hss_artifacts')
    result = hss.discretize(provider, config=hss_config, store_path=store_path, k_map=k_map)
    effective_layers = layers or [int(layer) for layer in provider.layers()]
    effective_k = {int(lr.layer): int(lr.n_clusters) for lr in result.layer_results}
    n_global = result.alignment.n_global_states if result.alignment else 0

    if store is not None:
        layer_df = pd.DataFrame(
            [{'layer': int(lr.layer), 'k': int(lr.n_clusters)} for lr in result.layer_results]
        )
        store.save_json('discretize/discretize_config.json', cfg.to_dict())
        store.save_json('discretize/effective_k_map.json', effective_k)
        store.save_csv('discretize/layer_summary.csv', layer_df)
        if result.global_labels is not None:
            store.save_npy('discretize/global_labels.npy', result.global_labels)
        if result.alignment is not None:
            store.save_json(
                'discretize/alignment_steps.json',
                [step.to_dict() for step in result.alignment.steps],
            )
        store.save_json(
            'discretize/summary.json',
            {
                'n_items': provider.n_items(),
                'n_layers': len(effective_layers),
                'n_global_states': n_global,
                'k_map': effective_k,
            },
        )

    return DiscretizeResult(
        hss_result=result,
        k_map=effective_k,
        layers=effective_layers,
        n_items=provider.n_items(),
        n_global_states=n_global,
        config=cfg.to_dict(),
    )


def run_tolerance_sweep(
    metrics_df: pd.DataFrame,
    *,
    tolerances: Optional[np.ndarray] = None,
    stability_min: float = 0.3,
    store: Optional[ExperimentStore] = None,
) -> pd.DataFrame:
    if tolerances is None:
        tolerances = np.linspace(0.0, 0.1, 51)
    records = []
    for tol in tolerances:
        k_map = hss.select_icl_parsimonious(
            metrics_df,
            relative_pct=float(tol),
            stability_min=stability_min,
        )
        if not k_map:
            continue
        ks = list(k_map.values())
        records.append(
            {
                'tolerance': float(tol),
                'avg_k': float(np.mean(ks)),
                'std_k': float(np.std(ks)),
                'min_k': int(np.min(ks)),
                'max_k': int(np.max(ks)),
                'n_layers': len(ks),
            }
        )
    df = pd.DataFrame(records)
    if store is not None:
        store.save_csv('select/tolerance_sweep.csv', df)
    return df


def build_rel_icl_surface(
    metrics_df: pd.DataFrame,
    *,
    store: Optional[ExperimentStore] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers_all = sorted(metrics_df['layer'].unique())
    k_all = sorted(metrics_df['k'].unique())
    layers_arr = np.array(layers_all)
    k_arr = np.array(k_all)
    rel_mat = np.full((len(layers_all), len(k_all)), np.nan, dtype=np.float64)
    for li, layer in enumerate(layers_all):
        sub = metrics_df[metrics_df['layer'] == layer].set_index('k')
        icl_vals = sub['icl'].reindex(k_all)
        valid = icl_vals.dropna()
        if len(valid) == 0:
            continue
        best = float(valid.min())
        scale = abs(best) if abs(best) > 1e-12 else 1.0
        for ki, k in enumerate(k_all):
            if k in valid.index:
                rel_mat[li, ki] = (float(valid[k]) - best) / scale
    if store is not None:
        store.save_npy('scan/rel_icl_surface.npy', rel_mat)
        store.save_npy('scan/rel_icl_layers.npy', layers_arr)
        store.save_npy('scan/rel_icl_k_values.npy', k_arr)
    return rel_mat, layers_arr, k_arr

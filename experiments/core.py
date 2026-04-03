from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import ExperimentConfig
from .data import LoadedInputs, load_inputs
from .discretize import DiscretizeResult, ScanResult, SelectResult, run_discretize, run_scan, run_select
from .io_utils import ExperimentStore
from .k_analysis import KAnalysisResult, run_k_analysis, summarize_k_map


@dataclass
class BaseArtifacts:
    loaded: LoadedInputs
    scan: ScanResult
    select: SelectResult
    discretize: DiscretizeResult
    k_analysis: KAnalysisResult
    layers: list[int]


def prepare_inputs(cfg: ExperimentConfig, *, store: Optional[ExperimentStore] = None) -> LoadedInputs:
    loaded = load_inputs(cfg.data)
    if store is not None:
        store.save_json('meta/data_summary.json', {
            'source': cfg.data.source,
            'n_items': int(loaded.n_items),
            'state_dim': int(loaded.state_dim),
            'n_layers': int(len(loaded.layers)),
            'has_labels': bool(len(loaded.y) > 0),
            'positive_rate': float(loaded.y.mean()) if len(loaded.y) > 0 else None,
        })
    return loaded


def run_core_pipeline(cfg: ExperimentConfig, *, loaded: Optional[LoadedInputs] = None, store: Optional[ExperimentStore] = None) -> BaseArtifacts:
    loaded = loaded or prepare_inputs(cfg, store=store)
    scan = run_scan(loaded.provider, cfg.scan, layers=loaded.layers, store=store)
    select = run_select(scan.metrics_df, cfg.select, store=store)
    discretize = run_discretize(loaded.provider, cfg.discretize, k_map=select.k_map, layers=loaded.layers, store=store)
    k_analysis = run_k_analysis(scan.metrics_df, cfg.select, store=store)
    if store is not None:
        store.save_json('meta/base_summary.json', {
            'n_items': int(loaded.n_items),
            'n_layers': int(len(loaded.layers)),
            'state_dim': int(loaded.state_dim),
            'k_summary': summarize_k_map(discretize.k_map),
            'n_global_states': int(discretize.n_global_states),
        })
    return BaseArtifacts(loaded=loaded, scan=scan, select=select, discretize=discretize, k_analysis=k_analysis, layers=list(loaded.layers))

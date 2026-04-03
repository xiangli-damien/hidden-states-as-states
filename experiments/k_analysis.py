from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SelectConfig
from .discretize import build_rel_icl_surface, run_tolerance_sweep
from .io_utils import ExperimentStore


@dataclass
class KAnalysisResult:
    tolerance_df: pd.DataFrame
    rel_icl_surface: np.ndarray
    rel_icl_layers: np.ndarray
    rel_icl_k_values: np.ndarray


def summarize_k_map(k_map: Dict[int, int]) -> Dict[str, float | int]:
    if not k_map:
        return {'n_layers': 0, 'k_mean': 0.0, 'k_std': 0.0, 'k_min': 0, 'k_max': 0}
    ks = np.array(list(k_map.values()), dtype=np.float64)
    return {
        'n_layers': int(len(ks)),
        'k_mean': float(np.mean(ks)),
        'k_std': float(np.std(ks)),
        'k_min': int(np.min(ks)),
        'k_max': int(np.max(ks)),
    }


def run_k_analysis(metrics_df: pd.DataFrame, select_cfg: SelectConfig, *, store: Optional[ExperimentStore] = None) -> KAnalysisResult:
    tolerance_df = run_tolerance_sweep(metrics_df, stability_min=select_cfg.stability_min, store=store)
    rel_icl_surface, rel_icl_layers, rel_icl_k_values = build_rel_icl_surface(metrics_df, store=store)
    if store is not None:
        store.save_json('select/k_analysis_summary.json', {
            'tolerance_rows': int(len(tolerance_df)),
            'surface_shape': [int(rel_icl_surface.shape[0]), int(rel_icl_surface.shape[1])],
        })
    return KAnalysisResult(tolerance_df=tolerance_df, rel_icl_surface=rel_icl_surface, rel_icl_layers=rel_icl_layers, rel_icl_k_values=rel_icl_k_values)

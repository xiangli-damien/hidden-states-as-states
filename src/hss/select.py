"""
Phase 2 (selection only): choose best k per layer from a metrics DataFrame.
Zero compute cost—pure filter/aggregation. Can be called repeatedly with
different relative_pct, stability_min, or criteria without re-running clustering.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def select_icl_parsimonious(
    metrics_df: pd.DataFrame,
    *,
    relative_pct: float = 0.02,
    stability_min: float = 0.30,
) -> Dict[int, int]:
    """
    Per layer: among rows with stability >= stability_min, find the best (min) ICL,
    then take the smallest k within relative_pct of that ICL.
    Returns {layer: k}.
    """
    if metrics_df is None or metrics_df.empty:
        return {}
    result: Dict[int, int] = {}
    for layer, g in metrics_df.groupby("layer"):
        g = g.copy()
        if "stability" in g.columns:
            stable = g[g["stability"] >= stability_min]
            if len(stable) == 0:
                stable = g
        else:
            stable = g

        valid = stable[stable["icl"].notna()]
        if len(valid) == 0:
            if "silhouette" in g.columns:
                best_row = g.loc[g["silhouette"].idxmax()]
            else:
                best_row = g.loc[g["k"].idxmin()]
            result[int(layer)] = int(best_row["k"])
            continue

        best_icl = float(valid["icl"].min())
        threshold = best_icl + abs(best_icl) * relative_pct
        candidates = valid[valid["icl"] <= threshold]
        result[int(layer)] = int(candidates.loc[candidates["k"].idxmin(), "k"])

    return result


def select_sil_stable(
    metrics_df: pd.DataFrame,
    *,
    stability_min: float = 0.60,
) -> Dict[int, int]:
    """
    Per layer: among rows with stability >= stability_min, pick k with max silhouette.
    """
    if metrics_df is None or metrics_df.empty:
        return {}
    result: Dict[int, int] = {}
    for layer, g in metrics_df.groupby("layer"):
        g = g.copy()
        stable = (
            g[g["stability"] >= stability_min]
            if "stability" in g.columns
            else g
        )
        if len(stable) == 0:
            stable = g
        valid = stable[stable["silhouette"].notna()]
        if len(valid) == 0:
            result[int(layer)] = int(g["k"].min())
            continue
        result[int(layer)] = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    return result


def select_auto_target_k(
    metrics_df: pd.DataFrame,
    *,
    target_avg_k: float = 8.0,
    stability_min: float = 0.30,
    candidate_pcts: Optional[List[float]] = None,
) -> Dict[int, int]:
    """
    Try multiple relative_pct values; choose the one whose per-layer k map
    has mean(k) closest to target_avg_k.
    """
    if metrics_df is None or metrics_df.empty:
        return {}
    if candidate_pcts is None:
        candidate_pcts = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

    best_k_map: Optional[Dict[int, int]] = None
    best_gap = float("inf")

    for pct in candidate_pcts:
        k_map = select_icl_parsimonious(
            metrics_df, relative_pct=pct, stability_min=stability_min
        )
        if not k_map:
            continue
        avg_k = float(np.mean(list(k_map.values())))
        gap = abs(avg_k - target_avg_k)
        if gap < best_gap:
            best_gap = gap
            best_k_map = k_map

    return best_k_map or {}

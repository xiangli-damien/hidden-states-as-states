from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_utils import ExperimentStore


@dataclass
class LabelAnalysisResult:
    delta_matrix: np.ndarray
    pos_rate_matrix: np.ndarray
    count_matrix: np.ndarray
    layers: List[int]
    cluster_order: List[int]
    baseline: float
    birth: Dict[int, int]
    death: Dict[int, int]


def _compute_matrices(
    global_labels: np.ndarray,
    y: np.ndarray,
    layers: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, L = global_labels.shape
    baseline = float(np.mean(y.ravel()))
    n_global = int(global_labels.max()) + 1

    pos_rate = np.full((L, n_global), np.nan, dtype=np.float64)
    counts = np.zeros((L, n_global), dtype=np.int64)
    delta = np.full((L, n_global), np.nan, dtype=np.float64)

    for li in range(L):
        col = global_labels[:, li]
        for gid in range(n_global):
            mask = col == gid
            c = int(mask.sum())
            if c == 0:
                continue
            counts[li, gid] = c
            pr = float(y.ravel()[mask].mean())
            pos_rate[li, gid] = pr
            delta[li, gid] = pr - baseline

    return delta, pos_rate, counts, baseline


def _compute_lifecycle(
    global_labels: np.ndarray,
    layers: List[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    n, L = global_labels.shape
    n_global = int(global_labels.max()) + 1
    birth, death = {}, {}

    for gid in range(n_global):
        first, last = None, None
        for li in range(L):
            if np.any(global_labels[:, li] == gid):
                if first is None:
                    first = layers[li]
                last = layers[li]
        if first is not None:
            birth[gid] = first
        if last is not None:
            death[gid] = last

    return birth, death


def _cluster_order(delta: np.ndarray, counts: np.ndarray) -> List[int]:
    n_global = delta.shape[1]
    mean_delta = np.nanmean(delta, axis=0)
    total = counts.sum(axis=0)
    active = np.where(total > 0)[0]
    return active[np.argsort(-mean_delta[active])].tolist()


def run_label_analysis(
    global_labels: np.ndarray,
    y: np.ndarray,
    layers: List[int],
    *,
    store: Optional[ExperimentStore] = None,
) -> LabelAnalysisResult:
    delta, pos_rate, counts, baseline = _compute_matrices(global_labels, y, layers)
    birth, death = _compute_lifecycle(global_labels, layers)
    order = _cluster_order(delta, counts)

    result = LabelAnalysisResult(
        delta_matrix=delta, pos_rate_matrix=pos_rate, count_matrix=counts,
        layers=layers, cluster_order=order, baseline=baseline,
        birth=birth, death=death,
    )

    if store is not None:
        store.save_npy("geometry/delta_matrix.npy", delta)
        store.save_npy("geometry/pos_rate_matrix.npy", pos_rate)
        store.save_npy("geometry/count_matrix.npy", counts)
        store.save_json("geometry/lifecycle.json", {
            "birth": birth, "death": death,
            "cluster_order": order, "baseline": baseline,
        })

    return result


def find_sink_states(
    result: LabelAnalysisResult,
    *,
    min_persistence: int = 5,
    min_count: int = 50,
    delta_threshold: float = -0.1,
) -> pd.DataFrame:
    delta, counts = result.delta_matrix, result.count_matrix
    n_global = delta.shape[1]
    L = len(result.layers)
    records = []

    for gid in range(n_global):
        active = [li for li in range(L) if counts[li, gid] > 0]
        if len(active) < min_persistence:
            continue
        total = int(counts[:, gid].sum())
        if total < min_count:
            continue
        md = float(np.nanmean(delta[active, gid]))
        if md > delta_threshold:
            continue
        records.append({
            "global_id": gid,
            "birth_layer": result.birth.get(gid, result.layers[0]),
            "n_active_layers": len(active),
            "mean_delta": md,
            "total_count": total,
        })

    return pd.DataFrame(records)


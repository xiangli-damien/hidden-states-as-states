from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from hss.types import StateProvider

from .io_utils import ExperimentStore

_log = logging.getLogger(__name__)


def _entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))


def _mutual_information_bits(c: np.ndarray, y: np.ndarray) -> float:
    n = len(c)
    if n == 0:
        return 0.0
    c_vals, y_vals = np.unique(c), np.unique(y)
    joint = np.zeros((len(c_vals), len(y_vals)), dtype=np.float64)
    c_map = {v: i for i, v in enumerate(c_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    for ci, yi in zip(c, y):
        joint[c_map[ci], y_map[yi]] += 1.0
    joint /= n
    p_c = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    mi = 0.0
    for i in range(len(c_vals)):
        for j in range(len(y_vals)):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (p_c[i] * p_y[j] + 1e-15) + 1e-15
                )
    return float(mi)


@dataclass
class InformationResult:
    df: pd.DataFrame
    layers: List[int]


def run_information_analysis(
    global_labels: np.ndarray,
    y: np.ndarray,
    layers: List[int],
    *,
    store: Optional[ExperimentStore] = None,
) -> InformationResult:
    n, L = global_labels.shape
    y = y.ravel()
    h_y = _entropy_bits(np.bincount(y, minlength=2))

    records = []
    for li in range(L):
        col = global_labels[:, li]
        valid = col >= 0
        col_v, y_v = col[valid], y[valid]
        c_counts = np.bincount(col_v)
        records.append({
            "layer": layers[li],
            "H_C": _entropy_bits(c_counts),
            "I_CY": _mutual_information_bits(col_v, y_v),
            "H_Y": h_y,
            "n_active_states": int(np.sum(c_counts > 0)),
        })

    df = pd.DataFrame(records)
    if store is not None:
        store.save_csv("information/information_analysis.csv", df)
    return InformationResult(df=df, layers=layers)


@dataclass
class EffectiveRankResult:
    df: pd.DataFrame
    layers: List[int]


def run_effective_rank(
    provider: StateProvider,
    *,
    layers: Optional[List[int]] = None,
    fracs: Optional[List[float]] = None,
    threshold: float = 0.90,
    seed: int = 42,
    n_repeats: int = 3,
    store: Optional[ExperimentStore] = None,
) -> EffectiveRankResult:
    if layers is None:
        layers = [int(l) for l in provider.layers()]
    if fracs is None:
        fracs = [0.3, 0.5, 0.7, 0.9]

    n = provider.n_items()
    records = []

    for frac in fracs:
        n_sub = max(10, int(n * frac))
        for rep in range(n_repeats):
            rng = np.random.RandomState(seed + rep)
            idx = np.sort(rng.choice(n, n_sub, replace=False)).astype(np.int64)
            for l in layers:
                chunks = []
                for batch in provider.iter_batches(layer=l, indices=idx, batch_size=4096):
                    chunks.append(np.asarray(batch.states, dtype=np.float32))
                X = np.concatenate(chunks, axis=0)
                try:
                    pca = PCA().fit(X)
                    cumvar = np.cumsum(pca.explained_variance_ratio_)
                    eff_rank = int(np.searchsorted(cumvar, threshold) + 1)
                except Exception:
                    eff_rank = 0
                records.append({
                    "frac": frac, "repeat": rep, "layer": l,
                    "effective_rank": eff_rank, "n_samples": n_sub,
                })

    df = pd.DataFrame(records)
    if store is not None:
        store.save_csv("information/effective_rank.csv", df)
    return EffectiveRankResult(df=df, layers=layers)


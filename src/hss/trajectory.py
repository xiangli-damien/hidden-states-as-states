from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import AlignmentResult

_HAMMING_MAX_N = 10000


def build_global_labels(
    labels_by_layer: List[np.ndarray],
    local_to_global: List[np.ndarray],
) -> np.ndarray:
    L = len(labels_by_layer)
    if L == 0:
        return np.empty((0, 0), dtype=np.int32)
    n = len(labels_by_layer[0])
    local = np.stack(
        [np.asarray(lb, dtype=np.int32).ravel() for lb in labels_by_layer],
        axis=1,
    )
    global_labels = np.full_like(local, -1, dtype=np.int32)
    for li in range(L):
        gmap = np.asarray(local_to_global[li], dtype=np.int32).ravel()
        lbl = local[:, li].astype(np.int64, copy=False)
        mask = (lbl >= 0) & (lbl < len(gmap))
        if np.any(mask):
            global_labels[mask, li] = gmap[lbl[mask]]
    return global_labels


def build_trajectory_df(
    labels_by_layer: List[np.ndarray],
    alignment: AlignmentResult,
    *,
    sample_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    L = len(labels_by_layer)
    if L == 0:
        return pd.DataFrame(
            columns=["sample_idx", "layer", "local_cluster", "global_cluster"]
        )
    n = len(labels_by_layer[0])
    layers = alignment.layers
    ltg = alignment.local_to_global
    if sample_ids is None:
        sample_ids = np.arange(n, dtype=np.int64)
    else:
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
    sample_col = np.tile(sample_ids, L)
    layer_col = np.repeat(np.asarray(layers, dtype=np.int32), n)
    local_chunks: List[np.ndarray] = []
    global_chunks: List[np.ndarray] = []
    for li, (lbl, gmap) in enumerate(zip(labels_by_layer, ltg)):
        lbl = np.asarray(lbl, dtype=np.int32).ravel()
        gmap = np.asarray(gmap, dtype=np.int32).ravel()
        local_chunks.append(lbl)
        gid = np.full(n, -1, dtype=np.int32)
        mask = (lbl >= 0) & (lbl < len(gmap))
        if np.any(mask):
            gid[mask] = gmap[lbl[mask]]
        global_chunks.append(gid)
    return pd.DataFrame(
        {
            "sample_idx": sample_col.astype(np.int64),
            "layer": layer_col.astype(np.int32),
            "local_cluster": np.concatenate(local_chunks).astype(np.int32),
            "global_cluster": np.concatenate(global_chunks).astype(np.int32),
        }
    )


def count_transitions(
    global_labels: np.ndarray,
    layers: List[int],
) -> pd.DataFrame:
    n, L = global_labels.shape
    rows = []
    for li in range(L - 1):
        layer_from = layers[li]
        layer_to = layers[li + 1]
        c_from = global_labels[:, li]
        c_to = global_labels[:, li + 1]
        valid = (c_from >= 0) & (c_to >= 0)
        if not np.any(valid):
            continue
        cf = c_from[valid]
        ct = c_to[valid]
        pairs, counts = np.unique(
            np.stack([cf, ct], axis=1), axis=0, return_counts=True
        )
        for (fr, to), cnt in zip(pairs, counts):
            rows.append(
                {
                    "layer_from": int(layer_from),
                    "layer_to": int(layer_to),
                    "cluster_from": int(fr),
                    "cluster_to": int(to),
                    "count": int(cnt),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "layer_from",
            "layer_to",
            "cluster_from",
            "cluster_to",
            "count",
        ]
    )


@dataclass(frozen=True)
class EventConfig:
    min_count: int = 20
    prob_threshold: float = 0.05


def detect_events(
    trans_df: pd.DataFrame,
    cfg: Optional[EventConfig] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = EventConfig()
    cols = ["event", "layer", "cluster", "detail", "count", "prob"]
    if trans_df.empty:
        return pd.DataFrame(columns=cols)
    rows: List[Dict] = []
    out_counts = (
        trans_df.groupby(["layer_from", "cluster_from"])["count"]
        .sum()
        .reset_index(name="out_count")
    )
    in_counts = (
        trans_df.groupby(["layer_to", "cluster_to"])["count"]
        .sum()
        .reset_index(name="in_count")
    )
    for layer in sorted(trans_df["layer_from"].unique()):
        out_l = out_counts[out_counts["layer_from"] == layer]
        layer_to = layer + 1
        in_l = in_counts[in_counts["layer_to"] == layer_to]
        out_set = set(out_l["cluster_from"].tolist())
        in_set = set(in_l["cluster_to"].tolist())
        for c in sorted(in_set - out_set):
            c_in = int(in_l[in_l["cluster_to"] == c]["in_count"].sum())
            if c_in >= cfg.min_count:
                rows.append(
                    {
                        "event": "birth",
                        "layer": int(layer_to),
                        "cluster": int(c),
                        "detail": "appears_in",
                        "count": c_in,
                        "prob": np.nan,
                    }
                )
        for c in sorted(out_set - in_set):
            c_out = int(
                out_l[out_l["cluster_from"] == c]["out_count"].sum()
            )
            if c_out >= cfg.min_count:
                rows.append(
                    {
                        "event": "death",
                        "layer": int(layer),
                        "cluster": int(c),
                        "detail": "disappears_out",
                        "count": c_out,
                        "prob": np.nan,
                    }
                )
    cnt = trans_df.copy()
    cnt["total_from"] = cnt.groupby(["layer_from", "cluster_from"])[
        "count"
    ].transform("sum")
    cnt["prob"] = cnt["count"] / cnt["total_from"].clip(lower=1)

    for (layer_from, c_from), g in cnt.groupby(["layer_from", "cluster_from"]):
        significant = g[g["prob"] >= cfg.prob_threshold]
        if len(significant) >= 2:
            total = int(g["count"].sum())
            if total >= cfg.min_count:
                targets = sorted(
                    significant["cluster_to"].astype(int).tolist()
                )
                rows.append(
                    {
                        "event": "split",
                        "layer": int(layer_from),
                        "cluster": int(c_from),
                        "detail": f"to={targets}",
                        "count": total,
                        "prob": float(significant["prob"].max()),
                    }
                )

    cnt["total_to"] = cnt.groupby(["layer_from", "cluster_to"])[
        "count"
    ].transform("sum")
    cnt["prob_in"] = cnt["count"] / cnt["total_to"].clip(lower=1)

    for (layer_from, c_to), g in cnt.groupby(["layer_from", "cluster_to"]):
        significant = g[g["prob_in"] >= cfg.prob_threshold]
        if len(significant) >= 2:
            total = int(g["count"].sum())
            if total >= cfg.min_count:
                sources = sorted(
                    significant["cluster_from"].astype(int).tolist()
                )
                rows.append(
                    {
                        "event": "merge",
                        "layer": int(layer_from + 1),
                        "cluster": int(c_to),
                        "detail": f"from={sources}",
                        "count": total,
                        "prob": float(significant["prob_in"].max()),
                    }
                )

    if rows:
        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(columns=cols)


def trajectory_hamming_similarity(
    global_labels: np.ndarray,
    *,
    max_n: int = _HAMMING_MAX_N,
    seed: int = 42,
) -> np.ndarray:
    n, L = global_labels.shape
    if n > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_n, replace=False)
        idx.sort()
        global_labels = global_labels[idx]
        n = max_n
    eq = np.zeros((n, n), dtype=np.float32)
    for li in range(L):
        col = global_labels[:, li]
        eq += (col[:, None] == col[None, :]).astype(np.float32)
    return eq / max(1, L)


def self_transition_prob(global_labels: np.ndarray) -> np.ndarray:
    n, L = global_labels.shape
    probs = np.zeros(L - 1, dtype=np.float64)
    for li in range(L - 1):
        valid = (global_labels[:, li] >= 0) & (global_labels[:, li + 1] >= 0)
        if valid.sum() == 0:
            probs[li] = np.nan
        else:
            probs[li] = float(
                (
                    global_labels[valid, li] == global_labels[valid, li + 1]
                ).mean()
            )
    return probs


def active_states_per_layer(global_labels: np.ndarray) -> np.ndarray:
    _, L = global_labels.shape
    counts = np.zeros(L, dtype=np.int32)
    for li in range(L):
        col = global_labels[:, li]
        counts[li] = len(np.unique(col[col >= 0]))
    return counts
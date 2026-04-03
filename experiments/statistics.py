from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import hss
from hss.trajectory import (
    count_transitions,
    detect_events,
    trajectory_hamming_similarity,
    self_transition_prob,
    active_states_per_layer,
    EventConfig,
)

from .io_utils import ExperimentStore


@dataclass
class DynamicsResult:
    node_df: pd.DataFrame
    flow_df: pd.DataFrame
    events_df: pd.DataFrame
    self_trans: np.ndarray
    active_per_layer: np.ndarray
    layers: List[int]


def _node_df(
    global_labels: np.ndarray,
    layers: List[int],
    y: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
) -> pd.DataFrame:
    n, L = global_labels.shape
    records = []
    for li in range(L):
        col = global_labels[:, li]
        for gid in np.unique(col[col >= 0]):
            mask = col == gid
            row = {"layer": layers[li], "cluster": int(gid), "count": int(mask.sum())}
            if y is not None and baseline is not None:
                pr = float(y[mask].mean())
                row["pos_rate"] = pr
                row["disc"] = pr - baseline
            records.append(row)
    return pd.DataFrame(records)


def _flow_df(
    global_labels: np.ndarray,
    layers: List[int],
    y: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
) -> pd.DataFrame:
    trans = count_transitions(global_labels, layers)
    if trans.empty:
        return trans

    total_out = trans.groupby(["layer_from", "cluster_from"])["count"].transform("sum")
    trans["prob_out"] = trans["count"] / total_out.clip(lower=1)

    if y is not None and baseline is not None:
        pr_list = []
        for _, row in trans.iterrows():
            li_f = layers.index(int(row["layer_from"]))
            li_t = layers.index(int(row["layer_to"]))
            mask = (
                (global_labels[:, li_f] == int(row["cluster_from"])) &
                (global_labels[:, li_t] == int(row["cluster_to"]))
            )
            pr_list.append(float(y[mask].mean()) if mask.sum() > 0 else baseline)
        trans["pos_rate"] = pr_list
        trans["disc"] = trans["pos_rate"] - baseline

    return trans


def _add_out_entropy(node_df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
    if flow_df.empty:
        node_df = node_df.copy()
        node_df["out_entropy"] = 0.0
        return node_df

    ent_map: Dict[tuple, float] = {}
    for (lf, cf), g in flow_df.groupby(["layer_from", "cluster_from"]):
        p = g["prob_out"].values
        p = p[p > 0]
        ent_map[(int(lf), int(cf))] = float(-np.sum(p * np.log(p + 1e-15))) if len(p) else 0.0

    node_df = node_df.copy()
    node_df["out_entropy"] = node_df.apply(
        lambda r: ent_map.get((int(r["layer"]), int(r["cluster"])), 0.0), axis=1,
    )
    return node_df


def run_dynamics(
    hss_result: hss.HSSResult,
    layers: List[int],
    y: Optional[np.ndarray] = None,
    *,
    store: Optional[ExperimentStore] = None,
) -> DynamicsResult:
    gl = hss_result.global_labels
    baseline = float(np.mean(y)) if y is not None and len(y) > 0 else None

    node = _node_df(gl, layers, y, baseline)
    flow = _flow_df(gl, layers, y, baseline)
    node = _add_out_entropy(node, flow)

    result = DynamicsResult(
        node_df=node,
        flow_df=flow,
        events_df=detect_events(flow, EventConfig()),
        self_trans=self_transition_prob(gl),
        active_per_layer=active_states_per_layer(gl),
        layers=layers,
    )

    if store is not None:
        store.save_csv("geometry/node_df.csv", node)
        store.save_csv("geometry/flow_df.csv", flow)
        store.save_csv("geometry/events_df.csv", result.events_df)
        store.save_npy("geometry/self_transition.npy", result.self_trans)
        store.save_npy("geometry/active_per_layer.npy", result.active_per_layer)

    return result


@dataclass
class SequenceStatsResult:
    hamming_sim: Optional[np.ndarray]
    marginal_freq: np.ndarray
    self_trans_mean: float
    active_mean: float


def run_sequence_stats(
    global_labels: np.ndarray,
    layers: List[int],
    *,
    max_n_hamming: int = 5000,
    seed: int = 42,
    store: Optional[ExperimentStore] = None,
) -> SequenceStatsResult:
    n_global = int(global_labels.max()) + 1
    freq = np.zeros(n_global, dtype=np.float64)
    for gid in range(n_global):
        freq[gid] = float(np.sum(global_labels == gid))
    freq /= freq.sum() + 1e-15

    self_trans = self_transition_prob(global_labels)
    active = active_states_per_layer(global_labels)

    ham = None
    if global_labels.shape[0] <= max_n_hamming:
        ham = trajectory_hamming_similarity(global_labels, max_n=max_n_hamming, seed=seed)

    result = SequenceStatsResult(
        hamming_sim=ham,
        marginal_freq=freq,
        self_trans_mean=float(np.nanmean(self_trans)),
        active_mean=float(active.mean()),
    )

    if store is not None:
        if ham is not None:
            store.save_npy("geometry/hamming_similarity.npy", ham)
        store.save_npy("geometry/marginal_freq.npy", freq)
        store.save_json("geometry/sequence_stats.json", {
            "self_trans_mean": result.self_trans_mean,
            "active_mean": result.active_mean,
        })

    return result


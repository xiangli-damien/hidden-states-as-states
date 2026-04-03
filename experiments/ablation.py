from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import hss
from hss.distance import cosine_similarity_matrix, euclidean_distance_sq
from hss.types import StateProvider

from .config import AblationConfig, DiscretizeConfig
from .discretize import _build_hss_config
from .io_utils import ExperimentStore

_log = logging.getLogger(__name__)


def _euc_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(euclidean_distance_sq(A, B), 0.0))


def _hungarian_cos(A, B):
    S = cosine_similarity_matrix(A, B).astype(np.float64)
    k = max(S.shape)
    pad = np.zeros((k, k), dtype=np.float64)
    pad[:S.shape[0], :S.shape[1]] = S
    ri, ci = linear_sum_assignment(-pad)
    return [(int(r), int(c), float(S[r, c]))
            for r, c in zip(ri, ci) if r < S.shape[0] and c < S.shape[1]]


def _hungarian_euc(A, B):
    D = _euc_dist(A, B).astype(np.float64)
    k = max(D.shape)
    pad = np.full((k, k), 1e12, dtype=np.float64)
    pad[:D.shape[0], :D.shape[1]] = D
    ri, ci = linear_sum_assignment(pad)
    return [(int(r), int(c), float(D[r, c]))
            for r, c in zip(ri, ci) if r < D.shape[0] and c < D.shape[1]]


def _overlap_cost(labs_prev, labs_curr):
    up, uc = np.unique(labs_prev), np.unique(labs_curr)
    mp = {c: i for i, c in enumerate(up)}
    mc = {c: i for i, c in enumerate(uc)}
    C = np.zeros((len(up), len(uc)), dtype=np.float64)
    for lp, lc in zip(labs_prev, labs_curr):
        C[mp[lp], mc[lc]] += 1.0
    return C


def _align_sequential(layers, labels_by_layer, cost_fn):
    """
    cost_fn(labels_prev, labels_curr, layer_prev, layer_curr) -> cost matrix.
    For label-based costs (e.g. overlap), use the first two args.
    For center-based costs (e.g. cosine), use layer_prev/layer_curr to index centers.
    """
    gl, maps, nxt = {}, {}, 0
    first = layers[0]
    uniq = np.unique(labels_by_layer[first])
    l2g = {}
    for c in uniq:
        l2g[int(c)] = nxt; nxt += 1
    gl[first] = np.array([l2g[int(c)] for c in labels_by_layer[first]], dtype=np.int32)
    maps[first] = l2g
    for i in range(1, len(layers)):
        layer, prev = layers[i], layers[i - 1]
        lp, lc = labels_by_layer[prev], labels_by_layer[layer]
        up, uc = np.unique(lp), np.unique(lc)
        C = cost_fn(lp, lc, prev, layer)
        k = max(C.shape)
        C_pad = np.zeros((k, k), dtype=np.float64)
        C_pad[:C.shape[0], :C.shape[1]] = C
        ri, ci = linear_sum_assignment(-C_pad)
        prev_l2g = maps[prev]
        curr_l2g, used = {}, set()
        for r, c in zip(ri, ci):
            if r < len(up) and c < len(uc):
                pl, cl = int(up[r]), int(uc[c])
                if pl in prev_l2g:
                    g = prev_l2g[pl]
                    if g not in used:
                        curr_l2g[cl] = g; used.add(g)
        for c in uc:
            if int(c) not in curr_l2g:
                curr_l2g[int(c)] = nxt; nxt += 1
        gl[layer] = np.array([curr_l2g[int(c)] for c in lc], dtype=np.int32)
        maps[layer] = curr_l2g
    return gl, maps, nxt


def _align_sequential_thresholded(layers, labels_by_layer, score_fn, accept_fn):
    gl, maps, nxt = {}, {}, 0
    acc_rates = []
    first = layers[0]
    uniq = np.unique(labels_by_layer[first])
    l2g = {}
    for c in uniq:
        l2g[int(c)] = nxt; nxt += 1
    gl[first] = np.array([l2g[int(c)] for c in labels_by_layer[first]], dtype=np.int32)
    maps[first] = l2g
    for i in range(1, len(layers)):
        layer, prev = layers[i], layers[i - 1]
        lc = labels_by_layer[layer]
        up, uc = np.unique(labels_by_layer[prev]), np.unique(lc)
        S = score_fn(prev, layer)
        k = max(S.shape)
        S_pad = np.zeros((k, k), dtype=np.float64)
        S_pad[:S.shape[0], :S.shape[1]] = S
        ri, ci = linear_sum_assignment(-S_pad)
        prev_l2g = maps[prev]
        curr_l2g, used, n_cand, n_acc = {}, set(), 0, 0
        for r, c in zip(ri, ci):
            if r < len(up) and c < len(uc):
                n_cand += 1
                pl, cl = int(up[r]), int(uc[c])
                if pl in prev_l2g and accept_fn(float(S[r, c])):
                    g = prev_l2g[pl]
                    if g not in used:
                        curr_l2g[cl] = g; used.add(g); n_acc += 1
        for c in uc:
            if int(c) not in curr_l2g:
                curr_l2g[int(c)] = nxt; nxt += 1
        gl[layer] = np.array([curr_l2g[int(c)] for c in lc], dtype=np.int32)
        maps[layer] = curr_l2g
        acc_rates.append(n_acc / max(n_cand, 1))
    return gl, nxt, acc_rates


def _align_growing_pool(layers, labels_by_layer, centers_by_layer,
                         metric="cosine", threshold=None):
    gl, nxt = {}, 0
    pool_gids, pool_centers, pool_sizes = [], [], []
    first = layers[0]
    uniq = np.unique(labels_by_layer[first])
    l2g = {}
    for c in uniq:
        l2g[int(c)] = nxt
        pool_gids.append(nxt); pool_centers.append(centers_by_layer[first][int(c)].copy())
        nxt += 1
    gl[first] = np.array([l2g[int(c)] for c in labels_by_layer[first]], dtype=np.int32)
    pool_sizes.append(len(pool_gids))
    for i in range(1, len(layers)):
        layer = layers[i]
        labs, centers = labels_by_layer[layer], centers_by_layer[layer]
        uc = np.unique(labs)
        pm = np.array(pool_centers)
        if metric == "cosine":
            S = cosine_similarity_matrix(pm, centers).astype(np.float64)
            k = max(S.shape); S_pad = np.zeros((k, k)); S_pad[:S.shape[0], :S.shape[1]] = S
            ri, ci = linear_sum_assignment(-S_pad)
        else:
            D = _euc_dist(pm, centers).astype(np.float64)
            k = max(D.shape); D_pad = np.full((k, k), 1e12); D_pad[:D.shape[0], :D.shape[1]] = D
            ri, ci = linear_sum_assignment(D_pad)
        curr_l2g, used = {}, set()
        for r, c in zip(ri, ci):
            if r < len(pool_gids) and c < len(uc):
                ok = True
                if threshold is not None:
                    ok = (float(S[r, c]) >= threshold) if metric == "cosine" else (float(D[r, c]) <= threshold)
                if ok:
                    g = pool_gids[r]
                    if g not in used:
                        curr_l2g[int(uc[c])] = g; used.add(g)
        for c in uc:
            if int(c) not in curr_l2g:
                curr_l2g[int(c)] = nxt
                pool_gids.append(nxt); pool_centers.append(centers[int(c)].copy()); nxt += 1
        gl[layer] = np.array([curr_l2g[int(c)] for c in labs], dtype=np.int32)
        pool_sizes.append(len(pool_gids))
    return gl, nxt, pool_sizes


def _align_anchor(layers, labels_by_layer, centers_by_layer, anchor_layer, metric="cosine"):
    gl, nxt = {}, 0
    ua = np.unique(labels_by_layer[anchor_layer])
    a_l2g = {}
    for c in ua:
        a_l2g[int(c)] = nxt; nxt += 1
    gl[anchor_layer] = np.array([a_l2g[int(c)] for c in labels_by_layer[anchor_layer]], dtype=np.int32)
    for l in layers:
        if l == anchor_layer:
            continue
        labs = labels_by_layer[l]
        ul = np.unique(labs)
        S = cosine_similarity_matrix(centers_by_layer[anchor_layer], centers_by_layer[l]) if metric == "cosine" \
            else -_euc_dist(centers_by_layer[anchor_layer], centers_by_layer[l])
        k = max(S.shape)
        S_pad = np.zeros((k, k), dtype=np.float64); S_pad[:S.shape[0], :S.shape[1]] = S.astype(np.float64)
        ri, ci = linear_sum_assignment(-S_pad)
        l2g, used = {}, set()
        for r, c in zip(ri, ci):
            if r < len(ua) and c < len(ul):
                g = a_l2g[int(ua[r])]
                if g not in used:
                    l2g[int(ul[c])] = g; used.add(g)
        for c in ul:
            if int(c) not in l2g:
                l2g[int(c)] = nxt; nxt += 1
        gl[l] = np.array([l2g[int(c)] for c in labs], dtype=np.int32)
    return gl, nxt


def _alignment_stats(layers, gl):
    st = []
    for i in range(len(layers) - 1):
        st.append(float(np.mean(gl[layers[i]] == gl[layers[i + 1]])))
    gids = set()
    for l in layers:
        gids.update(np.unique(gl[l]).tolist())
    n = len(gl[layers[0]])
    ch = np.zeros(n, dtype=np.int32)
    for i in range(len(layers) - 1):
        ch += (gl[layers[i]] != gl[layers[i + 1]]).astype(np.int32)
    return {
        "self_trans_by_layer": st,
        "self_trans_mean": float(np.mean(st)) if st else 0.0,
        "self_trans_min": float(np.min(st)) if st else 0.0,
        "total_unique_gids": len(gids),
        "avg_changes": float(np.mean(ch)),
    }


def _extract_dicts(hss_result):
    lbl, ctr = {}, {}
    for lr in hss_result.layer_results:
        lbl[lr.layer] = lr.labels
        ctr[lr.layer] = lr.centers_hidden
    return lbl, ctr



@dataclass
class ClusterMethodResult:
    results: Dict[str, hss.HSSResult]
    ari_df: pd.DataFrame
    summary_df: pd.DataFrame

def run_clustering_method_comparison(
    provider: StateProvider, disc_cfg: DiscretizeConfig, *,
    methods: Optional[List[str]] = None, layers: Optional[List[int]] = None,
    store: Optional[ExperimentStore] = None,
) -> ClusterMethodResult:
    if layers is None:
        layers = [int(l) for l in provider.layers()]
    if methods is None:
        methods = ["gmm", "kmeans"]
    cfgs = {
        "gmm": {"transform_steps": [{"name": "standardize"}], "method": "gmm"},
        "kmeans": {"transform_steps": [{"name": "l2"}], "method": "kmeans"},
    }
    results = {}
    for m in methods:
        results[m] = hss.discretize(provider, config=_build_hss_config(replace(disc_cfg, **cfgs.get(m, {})), layers))
    ari_records = []
    ml = list(results.keys())
    for i, m1 in enumerate(ml):
        for m2 in ml[i + 1:]:
            for li in range(len(layers)):
                ari_records.append({"method_a": m1, "method_b": m2, "layer": layers[li],
                    "ari": adjusted_rand_score(results[m1].layer_results[li].labels, results[m2].layer_results[li].labels)})
    ari_df = pd.DataFrame(ari_records)
    summary = [{"method": m, "k_mean": float(np.mean([lr.n_clusters for lr in r.layer_results])),
                "n_global_states": r.alignment.n_global_states if r.alignment else 0} for m, r in results.items()]
    summary_df = pd.DataFrame(summary)
    if store:
        store.save_csv("ablation/clustering_method_ari.csv", ari_df)
        store.save_csv("ablation/clustering_method_summary.csv", summary_df)
    return ClusterMethodResult(results=results, ari_df=ari_df, summary_df=summary_df)



@dataclass
class PreprocessResult:
    results: Dict[str, hss.HSSResult]
    ari_df: pd.DataFrame
    summary_df: pd.DataFrame

def run_preprocessing_comparison(
    provider: StateProvider, disc_cfg: DiscretizeConfig, *,
    variants: Optional[List[str]] = None, layers: Optional[List[int]] = None,
    store: Optional[ExperimentStore] = None,
) -> PreprocessResult:
    if layers is None:
        layers = [int(l) for l in provider.layers()]
    if variants is None:
        variants = ["standardize", "pca128", "standardize+pca128"]
    step_map = {
        "raw": [], "standardize": [{"name": "standardize"}],
        "pca128": [{"name": "pca", "n_components": 128}],
        "standardize+pca128": [{"name": "standardize"}, {"name": "pca", "n_components": 128}],
        "l2": [{"name": "l2"}],
        "l2+pca128": [{"name": "l2"}, {"name": "pca", "n_components": 128}],
    }
    results = {}
    for v in variants:
        results[v] = hss.discretize(provider, config=_build_hss_config(replace(disc_cfg, transform_steps=step_map.get(v, [])), layers))
    ari_records = []
    vl = list(results.keys())
    for i, v1 in enumerate(vl):
        for v2 in vl[i + 1:]:
            for li in range(len(layers)):
                ari_records.append({"variant_a": v1, "variant_b": v2, "layer": layers[li],
                    "ari": adjusted_rand_score(results[v1].layer_results[li].labels, results[v2].layer_results[li].labels)})
    ari_df = pd.DataFrame(ari_records)
    summary = [{"variant": v, "k_mean": float(np.mean([lr.n_clusters for lr in r.layer_results])),
                "n_global_states": r.alignment.n_global_states if r.alignment else 0} for v, r in results.items()]
    summary_df = pd.DataFrame(summary)
    if store:
        store.save_csv("ablation/preprocessing_ari.csv", ari_df)
        store.save_csv("ablation/preprocessing_summary.csv", summary_df)
    return PreprocessResult(results=results, ari_df=ari_df, summary_df=summary_df)



@dataclass
class TokenAggregationResult:
    results: Dict[str, hss.HSSResult]
    ari_df: pd.DataFrame

def run_token_aggregation_comparison(
    providers: Dict[str, StateProvider], disc_cfg: DiscretizeConfig, *,
    layers: Optional[List[int]] = None, store: Optional[ExperimentStore] = None,
) -> TokenAggregationResult:
    if layers is None:
        layers = [int(l) for l in next(iter(providers.values())).layers()]
    results = {n: hss.discretize(p, config=_build_hss_config(disc_cfg, layers)) for n, p in providers.items()}
    ari_records = []
    names = list(results.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            for li in range(len(layers)):
                ari_records.append({"agg_a": n1, "agg_b": n2, "layer": layers[li],
                    "ari": adjusted_rand_score(results[n1].layer_results[li].labels, results[n2].layer_results[li].labels)})
    ari_df = pd.DataFrame(ari_records)
    if store:
        store.save_csv("ablation/token_aggregation_ari.csv", ari_df)
    return TokenAggregationResult(results=results, ari_df=ari_df)



@dataclass
class CosineEuclideanResult:
    norm_df: pd.DataFrame
    agreement_df: pd.DataFrame
    rank_corr_df: pd.DataFrame

def run_cosine_vs_euclidean(
    hss_result: hss.HSSResult, layers: List[int], *,
    store: Optional[ExperimentStore] = None,
) -> CosineEuclideanResult:
    _, ctr = _extract_dicts(hss_result)
    norm_records = []
    for lr in hss_result.layer_results:
        norms = np.linalg.norm(lr.centers_hidden, axis=1)
        norm_records.append({"layer": lr.layer, "k": lr.n_clusters,
            "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
            "norm_cv": float(norms.std() / (norms.mean() + 1e-12))})
    norm_df = pd.DataFrame(norm_records)
    agree_records, rho_records = [], []
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        if l1 not in ctr or l2 not in ctr:
            continue
        c1, c2 = ctr[l1], ctr[l2]
        cos_p = set((r, c) for r, c, _ in _hungarian_cos(c1, c2))
        euc_p = set((r, c) for r, c, _ in _hungarian_euc(c1, c2))
        union = cos_p | euc_p
        agree_records.append({"layer_from": l1, "layer_to": l2,
            "agreement_pct": len(cos_p & euc_p) / len(union) * 100 if union else 0})
        S = cosine_similarity_matrix(c1, c2).ravel()
        D = _euc_dist(c1, c2).ravel()
        rho, pval = spearmanr(S, -D)
        rho_records.append({"layer_from": l1, "layer_to": l2, "spearman_rho": float(rho), "p_value": float(pval)})
    agree_df, rho_df = pd.DataFrame(agree_records), pd.DataFrame(rho_records)
    if store:
        store.save_csv("ablation/q1_norm_cv.csv", norm_df)
        store.save_csv("ablation/q1_matching_agreement.csv", agree_df)
        store.save_csv("ablation/q1_rank_correlation.csv", rho_df)
    return CosineEuclideanResult(norm_df=norm_df, agreement_df=agree_df, rank_corr_df=rho_df)



@dataclass
class FitVsHiddenResult:
    dominance_df: pd.DataFrame
    drift_df: pd.DataFrame

def run_fit_vs_hidden(
    hss_result: hss.HSSResult, layers: List[int], *,
    store: Optional[ExperimentStore] = None,
) -> FitVsHiddenResult:
    dom_records = []
    for lr in hss_result.layer_results:
        ch = lr.centers_hidden; k = ch.shape[0]
        mu = ch.mean(axis=0); mu_n = float(np.linalg.norm(mu))
        dev = float(np.mean(np.linalg.norm(ch - mu, axis=1)))
        pw = cosine_similarity_matrix(ch, ch)
        mask = ~np.eye(k, dtype=bool)
        dom_records.append({"layer": lr.layer, "mu_norm": mu_n, "mean_dev_norm": dev,
            "dominance_ratio": mu_n / (dev + 1e-12),
            "mean_pairwise_cos": float(pw[mask].mean()) if k > 1 else np.nan})
    dom_df = pd.DataFrame(dom_records)
    drift_records = []
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        lr1 = next((lr for lr in hss_result.layer_results if lr.layer == l1), None)
        lr2 = next((lr for lr in hss_result.layer_results if lr.layer == l2), None)
        if lr1 is None or lr2 is None:
            continue
        mu1, mu2 = lr1.centers_hidden.mean(axis=0), lr2.centers_hidden.mean(axis=0)
        drift_records.append({"layer_from": l1, "layer_to": l2,
            "cos_mu": float(np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-12)),
            "rel_l2_mu": float(np.linalg.norm(mu1 - mu2) / (np.linalg.norm(mu1) + 1e-12))})
    drift_df = pd.DataFrame(drift_records)
    if store:
        store.save_csv("ablation/q2_dominance.csv", dom_df)
        store.save_csv("ablation/q2_center_drift.csv", drift_df)
    return FitVsHiddenResult(dominance_df=dom_df, drift_df=drift_df)



@dataclass
class ThresholdSweepResult:
    cosine_df: pd.DataFrame
    separation_df: pd.DataFrame

def run_threshold_sweep(
    hss_result: hss.HSSResult, layers: List[int], *,
    n_thresholds: int = 50, store: Optional[ExperimentStore] = None,
) -> ThresholdSweepResult:
    lbl, ctr = _extract_dicts(hss_result)
    gl_ov, _, _ = _align_sequential(layers, lbl, lambda lp, lc, _lp, _lc: _overlap_cost(lp, lc))
    sep_records = []
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        if l1 not in ctr or l2 not in ctr:
            continue
        c1, c2 = ctr[l1], ctr[l2]
        S = cosine_similarity_matrix(c1, c2)
        matched = _hungarian_cos(c1, c2)
        m_set = set((r, c) for r, c, _ in matched)
        m_v = [float(S[r, c]) for r, c, _ in matched]
        um_v = [float(S[i, j]) for i in range(S.shape[0]) for j in range(S.shape[1]) if (i, j) not in m_set]
        sep_records.append({"layer_from": l1, "layer_to": l2,
            "matched_mean": float(np.mean(m_v)) if m_v else np.nan,
            "unmatched_mean": float(np.mean(um_v)) if um_v else np.nan})
    sep_df = pd.DataFrame(sep_records)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    sweep_records = []
    for t in thresholds:
        gl_t, _, acc = _align_sequential_thresholded(
            layers, lbl,
            lambda lp, lc: cosine_similarity_matrix(ctr[lp], ctr[lc]).astype(np.float64),
            lambda s, _t=float(t): s >= _t)
        stats = _alignment_stats(layers, gl_t)
        aris = [adjusted_rand_score(gl_ov[l], gl_t[l]) for l in layers]
        sweep_records.append({"threshold": float(t),
            "acceptance_rate": float(np.mean(acc)) if acc else 0.0,
            "self_trans_mean": stats["self_trans_mean"],
            "total_unique_gids": stats["total_unique_gids"],
            "ari_vs_overlap": float(np.mean(aris))})
    cos_df = pd.DataFrame(sweep_records)
    if store:
        store.save_csv("ablation/q3_threshold_sweep.csv", cos_df)
        store.save_csv("ablation/q3_matched_separation.csv", sep_df)
    return ThresholdSweepResult(cosine_df=cos_df, separation_df=sep_df)



@dataclass
class PropagationResult:
    stats_df: pd.DataFrame
    pool_sizes: Optional[List[int]]
    drift_df: pd.DataFrame

def run_propagation_comparison(
    hss_result: hss.HSSResult, layers: List[int], *,
    drift_strides: Optional[List[int]] = None,
    store: Optional[ExperimentStore] = None,
) -> PropagationResult:
    if drift_strides is None:
        drift_strides = [1, 2, 5, 10, 15, 20]
    lbl, ctr = _extract_dicts(hss_result)
    variants = {}
    gl_ov, _, _ = _align_sequential(layers, lbl, lambda lp, lc, _lp, _lc: _overlap_cost(lp, lc))
    variants["overlap"] = gl_ov
    gl_cos, _, _ = _align_sequential(layers, lbl,
        lambda _lp, _lc, layer_prev, layer_curr: cosine_similarity_matrix(ctr[layer_prev], ctr[layer_curr]).astype(np.float64))
    variants["cosine_seq"] = gl_cos
    gl_pool, _, pool_sizes = _align_growing_pool(layers, lbl, ctr)
    variants["cosine_pool"] = gl_pool
    mid = layers[len(layers) // 2]
    gl_anch, _ = _align_anchor(layers, lbl, ctr, anchor_layer=mid)
    variants["anchor_mid"] = gl_anch
    stats_records = []
    for name, gl in variants.items():
        s = _alignment_stats(layers, gl)
        aris = [adjusted_rand_score(gl_ov[l], gl[l]) for l in layers] if name != "overlap" else [1.0] * len(layers)
        stats_records.append({"variant": name, "self_trans_mean": s["self_trans_mean"],
            "total_unique_gids": s["total_unique_gids"], "avg_changes": s["avg_changes"],
            "ari_vs_overlap": float(np.mean(aris))})
    stats_df = pd.DataFrame(stats_records)
    drift_records = []
    for stride in drift_strides:
        if stride >= len(layers):
            continue
        aris = []
        for start in range(0, len(layers) - stride, max(1, stride)):
            end = min(start + stride, len(layers) - 1)
            aris.append(adjusted_rand_score(gl_ov[layers[start]], gl_ov[layers[end]]))
        if aris:
            drift_records.append({"stride": stride, "mean_ari": float(np.mean(aris))})
    drift_df = pd.DataFrame(drift_records)
    if store:
        store.save_csv("ablation/q4_propagation_stats.csv", stats_df)
        store.save_csv("ablation/q4_propagation_drift.csv", drift_df)
        if pool_sizes:
            store.save_json("ablation/q4_pool_sizes.json", pool_sizes)
    return PropagationResult(stats_df=stats_df, pool_sizes=pool_sizes, drift_df=drift_df)



@dataclass
class AlignmentComparisonResult:
    stats_df: pd.DataFrame
    ari_df: pd.DataFrame

def run_alignment_comparison(
    hss_result: hss.HSSResult, layers: List[int], *,
    store: Optional[ExperimentStore] = None,
) -> AlignmentComparisonResult:
    lbl, ctr = _extract_dicts(hss_result)
    gl_ov, _, _ = _align_sequential(layers, lbl, lambda lp, lc, _lp, _lc: _overlap_cost(lp, lc))
    gl_cos, _, _ = _align_sequential(layers, lbl,
        lambda _lp, _lc, layer_prev, layer_curr: cosine_similarity_matrix(ctr[layer_prev], ctr[layer_curr]).astype(np.float64))
    gl_euc, _, _ = _align_sequential(layers, lbl,
        lambda _lp, _lc, layer_prev, layer_curr: -_euc_dist(ctr[layer_prev], ctr[layer_curr]).astype(np.float64))
    gl_pool, _, _ = _align_growing_pool(layers, lbl, ctr)
    all_gls = {"overlap": gl_ov, "cosine_seq": gl_cos, "euclidean_seq": gl_euc, "cosine_pool": gl_pool}
    stats = [{"variant": n, **_alignment_stats(layers, gl)} for n, gl in all_gls.items()]
    stats_df = pd.DataFrame(stats)
    ari_records = []
    for n, gl in all_gls.items():
        if n == "overlap":
            continue
        for l in layers:
            ari_records.append({"variant": n, "layer": l, "ari_vs_overlap": adjusted_rand_score(gl_ov[l], gl[l])})
    ari_df = pd.DataFrame(ari_records)
    if store:
        store.save_csv("ablation/alignment_stats.csv", stats_df)
        store.save_csv("ablation/alignment_ari.csv", ari_df)
    return AlignmentComparisonResult(stats_df=stats_df, ari_df=ari_df)


from .core import BaseArtifacts


@dataclass
class AblationSuiteResult:
    clustering_method: Optional[ClusterMethodResult]
    preprocessing: Optional[PreprocessResult]
    token_aggregation: Optional[TokenAggregationResult]
    cosine_vs_euclidean: Optional[CosineEuclideanResult]
    fit_vs_hidden: Optional[FitVsHiddenResult]
    threshold_sweep: Optional[ThresholdSweepResult]
    propagation: Optional[PropagationResult]
    alignment: Optional[AlignmentComparisonResult]


def run_ablation_suite(
    base: BaseArtifacts,
    disc_cfg: DiscretizeConfig,
    abl_cfg: AblationConfig,
    *,
    token_aggregation_providers: Optional[Dict[str, StateProvider]] = None,
    store: Optional[ExperimentStore] = None,
) -> AblationSuiteResult:
    enabled = {name.lower() for name in abl_cfg.enabled}
    clustering_method = None
    preprocessing = None
    token_aggregation = None
    cosine_vs_euclidean = None
    fit_vs_hidden = None
    threshold_sweep = None
    propagation = None
    alignment = None
    if 'clustering_method' in enabled:
        clustering_method = run_clustering_method_comparison(
            base.loaded.provider,
            disc_cfg,
            methods=abl_cfg.clustering_methods,
            layers=base.layers,
            store=store,
        )
    if 'preprocessing' in enabled:
        preprocessing = run_preprocessing_comparison(
            base.loaded.provider,
            disc_cfg,
            variants=abl_cfg.preprocess_variants,
            layers=base.layers,
            store=store,
        )
    if 'token_aggregation' in enabled and token_aggregation_providers:
        token_aggregation = run_token_aggregation_comparison(
            token_aggregation_providers,
            disc_cfg,
            layers=base.layers,
            store=store,
        )
    if 'cosine_vs_euclidean' in enabled:
        cosine_vs_euclidean = run_cosine_vs_euclidean(base.discretize.hss_result, base.layers, store=store)
    if 'fit_vs_hidden' in enabled:
        fit_vs_hidden = run_fit_vs_hidden(base.discretize.hss_result, base.layers, store=store)
    if 'threshold_sweep' in enabled:
        threshold_sweep = run_threshold_sweep(base.discretize.hss_result, base.layers, n_thresholds=abl_cfg.threshold_points, store=store)
    if 'propagation' in enabled:
        propagation = run_propagation_comparison(base.discretize.hss_result, base.layers, drift_strides=abl_cfg.drift_strides, store=store)
    if 'alignment' in enabled:
        alignment = run_alignment_comparison(base.discretize.hss_result, base.layers, store=store)
    if store is not None:
        store.save_json('ablation/summary.json', {
            'enabled': sorted(enabled),
            'has_clustering_method': clustering_method is not None,
            'has_preprocessing': preprocessing is not None,
            'has_token_aggregation': token_aggregation is not None,
            'has_cosine_vs_euclidean': cosine_vs_euclidean is not None,
            'has_fit_vs_hidden': fit_vs_hidden is not None,
            'has_threshold_sweep': threshold_sweep is not None,
            'has_propagation': propagation is not None,
            'has_alignment': alignment is not None,
        })
    return AblationSuiteResult(
        clustering_method=clustering_method,
        preprocessing=preprocessing,
        token_aggregation=token_aggregation,
        cosine_vs_euclidean=cosine_vs_euclidean,
        fit_vs_hidden=fit_vs_hidden,
        threshold_sweep=threshold_sweep,
        propagation=propagation,
        alignment=alignment,
    )

from __future__ import annotations
import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional
import hss
import numpy as np
from hss.distance import cosine_similarity_matrix
from hss.types import StateProvider
from scipy.optimize import linear_sum_assignment
from .config import DiscretizeConfig, StabilityConfig
from .core import BaseArtifacts
from .discretize import _build_hss_config, _cluster_params_from_cfg
from .io_utils import ExperimentStore
_log = logging.getLogger(__name__)

@dataclass
class TrendStabilityResult:
    baseline_k_curve: np.ndarray
    seed_k_curves: Optional[np.ndarray]
    subsample_k_curves: Optional[Dict[float, np.ndarray]]
    kmax_k_curves: Optional[Dict[int, np.ndarray]]
    layers: np.ndarray

@dataclass
class PersistenceResult:
    seed_distances: np.ndarray
    sample_distances: Optional[np.ndarray]
    random_distances: np.ndarray
    layers: np.ndarray

@dataclass
class StabilitySuiteResult:
    trend: TrendStabilityResult
    persistence: PersistenceResult

def _discretize_quick(provider: StateProvider, cfg: DiscretizeConfig, layers: List[int], k_map: Optional[Dict[int, int]]=None, indices: Optional[np.ndarray]=None) -> hss.HSSResult:
    hss_cfg = _build_hss_config(cfg, layers)
    if indices is not None:
        hss_cfg = replace(hss_cfg, layers=layers, indices=indices)
    return hss.discretize(provider, config=hss_cfg, k_map=k_map)

def run_trend_stability(provider: StateProvider, disc_cfg: DiscretizeConfig, stab_cfg: StabilityConfig, *, layers: Optional[List[int]]=None, baseline_k_map: Optional[Dict[int, int]]=None, store: Optional[ExperimentStore]=None) -> TrendStabilityResult:
    if layers is None:
        layers = [int(l) for l in provider.layers()]
    layers_arr = np.array(layers, dtype=np.int32)
    res_base = _discretize_quick(provider, disc_cfg, layers, k_map=baseline_k_map)
    baseline_k = np.array([lr.n_clusters for lr in res_base.layer_results], dtype=np.int32)
    seed_curves = []
    for seed in stab_cfg.seeds:
        cfg_s = replace(disc_cfg, seed=seed)
        res_s = _discretize_quick(provider, cfg_s, layers, k_map=baseline_k_map)
        seed_curves.append([lr.n_clusters for lr in res_s.layer_results])
    seed_k_arr = np.array(seed_curves, dtype=np.int32) if seed_curves else None
    sub_k: Dict[float, np.ndarray] = {}
    n = provider.n_items()
    for frac in stab_cfg.subsample_fracs:
        if abs(frac - 1.0) < 1e-08:
            continue
        n_sub = max(2, int(n * frac))
        idx = np.sort(np.random.RandomState(disc_cfg.seed).choice(n, n_sub, replace=False)).astype(np.int64)
        res_sub = _discretize_quick(provider, disc_cfg, layers, indices=idx)
        sub_k[float(frac)] = np.array([lr.n_clusters for lr in res_sub.layer_results], dtype=np.int32)
    kmax_k: Dict[int, np.ndarray] = {}
    for kmax in stab_cfg.kmax_values:
        hss_cfg = _build_hss_config(disc_cfg, layers)
        cluster_km = hss.ClusterSpec(method=disc_cfg.method, k_range=(2, kmax), params=_cluster_params_from_cfg(disc_cfg))
        hss_cfg_km = replace(hss_cfg, layers=layers, cluster=cluster_km)
        res_km = hss.discretize(provider, config=hss_cfg_km)
        kmax_k[int(kmax)] = np.array([lr.n_clusters for lr in res_km.layer_results], dtype=np.int32)
    result = TrendStabilityResult(baseline_k_curve=baseline_k, seed_k_curves=seed_k_arr, subsample_k_curves=sub_k or None, kmax_k_curves=kmax_k or None, layers=layers_arr)
    if store is not None:
        store.save_npy('stability/layers.npy', layers_arr)
        store.save_npy('stability/baseline_k_curve.npy', baseline_k)
        if seed_k_arr is not None:
            store.save_npy('stability/seed_k_curves.npy', seed_k_arr)
        if sub_k:
            store.save_npz('stability/subsample_k_curves.npz', **{str(k): v for k, v in sub_k.items()})
        if kmax_k:
            store.save_npz('stability/kmax_k_curves.npz', **{str(k): v for k, v in kmax_k.items()})
        store.save_json('stability/trend_config.json', {'discretize': disc_cfg.to_dict(), 'stability': stab_cfg.to_dict()})
        store.save_json('stability/trend_summary.json', {'baseline_mean': float(np.mean(baseline_k)), 'baseline_std': float(np.std(baseline_k)), 'n_seed_curves': int(seed_k_arr.shape[0]) if seed_k_arr is not None else 0, 'n_subsample_curves': int(len(sub_k)), 'n_kmax_curves': int(len(kmax_k))})
    return result

def _bipartite_cosine_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    sim = cosine_similarity_matrix(c1, c2)
    k = max(sim.shape)
    pad = np.zeros((k, k), dtype=np.float64)
    pad[:sim.shape[0], :sim.shape[1]] = sim.astype(np.float64)
    ri, ci = linear_sum_assignment(-pad)
    matched = [float(sim[r, c]) for r, c in zip(ri, ci) if r < c1.shape[0] and c < c2.shape[0]]
    return 1.0 - float(np.mean(matched)) if matched else 1.0

def run_state_persistence(provider: StateProvider, disc_cfg: DiscretizeConfig, stab_cfg: StabilityConfig, *, layers: Optional[List[int]]=None, baseline_k_map: Optional[Dict[int, int]]=None, store: Optional[ExperimentStore]=None) -> PersistenceResult:
    if layers is None:
        layers = [int(l) for l in provider.layers()]
    layers_arr = np.array(layers, dtype=np.int32)
    res_base = _discretize_quick(provider, disc_cfg, layers, k_map=baseline_k_map)
    base_centers = [lr.centers_hidden for lr in res_base.layer_results]
    seed_dists = []
    for seed in stab_cfg.seeds:
        cfg_s = replace(disc_cfg, seed=seed)
        res_s = _discretize_quick(provider, cfg_s, layers, k_map=baseline_k_map)
        seed_dists.append([_bipartite_cosine_distance(base_centers[i], res_s.layer_results[i].centers_hidden) for i in range(len(layers))])
    seed_arr = np.array(seed_dists, dtype=np.float64)
    sample_dists = []
    n = provider.n_items()
    for frac in stab_cfg.subsample_fracs:
        if abs(frac - 1.0) < 1e-08:
            continue
        n_sub = max(2, int(n * frac))
        idx = np.sort(np.random.RandomState(disc_cfg.seed + 7777).choice(n, n_sub, replace=False)).astype(np.int64)
        res_sub = _discretize_quick(provider, disc_cfg, layers, k_map=baseline_k_map, indices=idx)
        sample_dists.append([_bipartite_cosine_distance(base_centers[i], res_sub.layer_results[i].centers_hidden) for i in range(len(layers))])
    sample_arr = np.array(sample_dists, dtype=np.float64) if sample_dists else None
    random_dists = []
    for i in range(stab_cfg.n_random_baselines):
        rng = np.random.RandomState(disc_cfg.seed + i * 999)
        random_dists.append([_bipartite_cosine_distance(base_centers[j], rng.randn(*base_centers[j].shape).astype(np.float32)) for j in range(len(layers))])
    random_arr = np.array(random_dists, dtype=np.float64)
    result = PersistenceResult(seed_distances=seed_arr, sample_distances=sample_arr, random_distances=random_arr, layers=layers_arr)
    if store is not None:
        store.save_npy('stability/layers.npy', layers_arr)
        store.save_npy('stability/seed_distances.npy', seed_arr)
        if sample_arr is not None:
            store.save_npy('stability/sample_distances.npy', sample_arr)
        store.save_npy('stability/random_distances.npy', random_arr)
        store.save_json('stability/persistence_config.json', {'discretize': disc_cfg.to_dict(), 'stability': stab_cfg.to_dict()})
        store.save_json('stability/persistence_summary.json', {'seed_mean': float(np.mean(seed_arr)), 'sample_mean': float(np.mean(sample_arr)) if sample_arr is not None and sample_arr.size else None, 'random_mean': float(np.mean(random_arr))})
    return result

def run_stability_suite(base: BaseArtifacts, disc_cfg: DiscretizeConfig, stab_cfg: StabilityConfig, *, store: Optional[ExperimentStore]=None) -> StabilitySuiteResult:
    trend = run_trend_stability(base.loaded.provider, disc_cfg, stab_cfg, layers=base.layers, baseline_k_map=base.select.k_map, store=store)
    persistence = run_state_persistence(base.loaded.provider, disc_cfg, stab_cfg, layers=base.layers, baseline_k_map=base.select.k_map, store=store)
    if store is not None:
        store.save_json('stability/summary.json', {'trend_baseline_mean': float(np.mean(trend.baseline_k_curve)), 'persistence_seed_mean': float(np.mean(persistence.seed_distances)), 'persistence_random_mean': float(np.mean(persistence.random_distances))})
    return StabilitySuiteResult(trend=trend, persistence=persistence)

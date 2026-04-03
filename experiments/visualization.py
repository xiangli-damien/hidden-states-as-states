from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .ablation import AblationSuiteResult
from .config import VisualizationConfig
from .core import BaseArtifacts
from .io_utils import ExperimentStore
from .preliminary import PreliminaryResult
from .prediction import PredictionResult
from .stability import StabilitySuiteResult
from .viz import (
    apply_style,
    plot_centroid_persistence,
    plot_dual_axis_information,
    plot_dynamics_absolute,
    plot_dynamics_relative,
    plot_dynamics_unlabeled,
    plot_effective_rank,
    plot_fold_distribution,
    plot_heatmap_diverging,
    plot_heatmap_probability,
    plot_information_bottleneck,
    plot_prediction_comparison,
    plot_rel_icl_surface,
    plot_tolerance_sweep,
    plot_trend_stability,
)


@dataclass
class VisualizationResult:
    paths: Dict[str, Path]


def _filter_layers(layers: list[int], cfg: VisualizationConfig) -> np.ndarray:
    if cfg.selected_layers is None:
        return np.array(layers, dtype=np.int32)
    keep = [int(l) for l in layers if int(l) in {int(x) for x in cfg.selected_layers}]
    return np.array(keep, dtype=np.int32)


def _filter_node_flow(node_df, flow_df, layers_arr):
    keep = set(int(x) for x in layers_arr.tolist())
    node_sub = node_df[node_df['layer'].isin(keep)].copy()
    flow_sub = flow_df[flow_df['layer_from'].isin(keep) & flow_df['layer_to'].isin(keep)].copy()
    return node_sub, flow_sub


def run_visualization_suite(base: BaseArtifacts, cfg: VisualizationConfig, *, preliminary: Optional[PreliminaryResult] = None, prediction: Optional[PredictionResult] = None, stability: Optional[StabilitySuiteResult] = None, ablation: Optional[AblationSuiteResult] = None, store: Optional[ExperimentStore] = None) -> VisualizationResult:
    apply_style()
    if store is None:
        raise ValueError('Visualization requires an ExperimentStore')
    paths: Dict[str, Path] = {}
    enabled = {name.lower() for name in cfg.enabled}
    layers_arr = _filter_layers(base.layers, cfg)
    if preliminary is not None:
        node_df, flow_df = _filter_node_flow(preliminary.dynamics.node_df, preliminary.dynamics.flow_df, layers_arr)
        if 'dynamics_unlabeled' in enabled and len(node_df) > 0:
            path = store.figures_dir / 'fig_dynamics_unlabeled'
            plot_dynamics_unlabeled(node_df, flow_df, layers_arr, base.loaded.n_items, out_path=path)
            paths['dynamics_unlabeled'] = path
        if len(base.loaded.y) > 0 and len(node_df) > 0:
            baseline = float(base.loaded.y.mean())
            if 'dynamics_absolute' in enabled:
                path = store.figures_dir / 'fig_dynamics_absolute'
                plot_dynamics_absolute(node_df, flow_df, layers_arr, base.loaded.n_items, baseline, out_path=path)
                paths['dynamics_absolute'] = path
            if 'dynamics_relative' in enabled:
                path = store.figures_dir / 'fig_dynamics_relative'
                plot_dynamics_relative(node_df, flow_df, layers_arr, base.loaded.n_items, baseline, out_path=path)
                paths['dynamics_relative'] = path
            if preliminary.label_analysis is not None:
                order = preliminary.label_analysis.cluster_order if cfg.cluster_order == 'mean_delta' else sorted(preliminary.label_analysis.birth)
                if 'heatmap_delta' in enabled:
                    path = store.figures_dir / 'fig_heatmap_delta'
                    plot_heatmap_diverging(preliminary.label_analysis.delta_matrix, np.array(base.layers, dtype=np.int32), order, preliminary.label_analysis.birth, preliminary.label_analysis.death, 'P(correct|cluster) - P(correct)', out_path=path)
                    paths['heatmap_delta'] = path
                if 'heatmap_probability' in enabled:
                    path = store.figures_dir / 'fig_heatmap_probability'
                    plot_heatmap_probability(preliminary.label_analysis.pos_rate_matrix, np.array(base.layers, dtype=np.int32), order, preliminary.label_analysis.birth, preliminary.label_analysis.death, preliminary.label_analysis.baseline, 'P(correct|cluster)', out_path=path)
                    paths['heatmap_probability'] = path
        if preliminary.information is not None:
            if 'information_curve' in enabled:
                path = store.figures_dir / 'fig_information_curve'
                plot_dual_axis_information(preliminary.information.df, out_path=path)
                paths['information_curve'] = path
            if 'information_bottleneck' in enabled:
                path = store.figures_dir / 'fig_information_bottleneck'
                plot_information_bottleneck(preliminary.information.df, out_path=path)
                paths['information_bottleneck'] = path
        if 'effective_rank' in enabled:
            path = store.figures_dir / 'fig_effective_rank'
            plot_effective_rank(preliminary.effective_rank.df, out_path=path)
            paths['effective_rank'] = path
    if 'rel_icl_surface' in enabled:
        path = store.figures_dir / 'fig_rel_icl_surface'
        plot_rel_icl_surface(base.k_analysis.rel_icl_surface, base.k_analysis.rel_icl_layers, base.k_analysis.rel_icl_k_values, k_sel_by_layer=base.select.k_map, rel_pct=base.select.config['relative_pct'], out_path=path)
        paths['rel_icl_surface'] = path
    if 'tolerance_sweep' in enabled and len(base.k_analysis.tolerance_df) > 0:
        path = store.figures_dir / 'fig_tolerance_sweep'
        plot_tolerance_sweep(base.k_analysis.tolerance_df, target_avg_k=base.select.config.get('target_avg_k', 8.0), out_path=path)
        paths['tolerance_sweep'] = path
    if stability is not None:
        if 'trend_stability' in enabled:
            path = store.figures_dir / 'fig_trend_stability'
            plot_trend_stability(stability.trend.baseline_k_curve, stability.trend.layers, stability.trend.seed_k_curves, stability.trend.subsample_k_curves, stability.trend.kmax_k_curves, out_path=path)
            paths['trend_stability'] = path
        if 'centroid_persistence' in enabled:
            path = store.figures_dir / 'fig_centroid_persistence'
            plot_centroid_persistence(stability.persistence.layers, stability.persistence.seed_distances, stability.persistence.random_distances, stability.persistence.sample_distances, out_path=path)
            paths['centroid_persistence'] = path
    if prediction is not None and len(prediction.summary_df) > 0:
        if 'prediction_auroc' in enabled:
            path = store.figures_dir / 'fig_prediction_auroc'
            plot_prediction_comparison(prediction.summary_df, metric='auroc', out_path=path)
            paths['prediction_auroc'] = path
        if 'prediction_folds' in enabled:
            path = store.figures_dir / 'fig_prediction_folds'
            plot_fold_distribution(prediction.fold_df, metric='auroc', out_path=path)
            paths['prediction_folds'] = path
    store.save_json('figures/manifest.json', {k: str(v) for k, v in paths.items()})
    return VisualizationResult(paths=paths)

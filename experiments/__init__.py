from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from .ablation import AlignmentComparisonResult, ClusterMethodResult, CosineEuclideanResult, FitVsHiddenResult, PreprocessResult, PropagationResult, ThresholdSweepResult, TokenAggregationResult, run_ablation_suite, run_alignment_comparison, run_clustering_method_comparison, run_cosine_vs_euclidean, run_fit_vs_hidden, run_preprocessing_comparison, run_propagation_comparison, run_threshold_sweep, run_token_aggregation_comparison
from .config import AblationConfig, DataConfig, DiscretizeConfig, ExecutionConfig, ExperimentConfig, PredictionConfig, ScanConfig, SelectConfig, StabilityConfig, VisualizationConfig, load_config, save_config
from .core import BaseArtifacts, prepare_inputs, run_core_pipeline
from .data import LoadedInputs, load_from_memmap, load_from_numpy, load_from_openact, load_inputs
from .discretize import DiscretizeResult, ScanResult, SelectResult, build_rel_icl_surface, run_discretize, run_scan, run_select, run_tolerance_sweep
from .information import EffectiveRankResult, InformationResult, run_effective_rank, run_information_analysis
from .io_utils import ExperimentStore
from .k_analysis import KAnalysisResult, run_k_analysis, summarize_k_map
from .label_analysis import LabelAnalysisResult, find_sink_states, run_label_analysis
from .prediction import PredictionResult, run_prediction
from .preliminary import PreliminaryResult, run_preliminary_suite
from .runner import ExperimentOutputs, run_experiment
from .stability import PersistenceResult, StabilitySuiteResult, TrendStabilityResult, run_stability_suite, run_state_persistence, run_trend_stability
from .statistics import DynamicsResult, SequenceStatsResult, run_dynamics, run_sequence_stats
from .visualization import VisualizationResult, run_visualization_suite

__all__ = [
    'AblationConfig', 'AlignmentComparisonResult', 'BaseArtifacts', 'ClusterMethodResult', 'CosineEuclideanResult', 'DataConfig', 'DiscretizeConfig', 'DiscretizeResult', 'DynamicsResult', 'EffectiveRankResult', 'ExecutionConfig', 'ExperimentConfig', 'ExperimentOutputs', 'ExperimentStore', 'FitVsHiddenResult', 'InformationResult', 'KAnalysisResult', 'LabelAnalysisResult', 'LoadedInputs', 'PersistenceResult', 'PredictionConfig', 'PredictionResult', 'PreliminaryResult', 'PreprocessResult', 'PropagationResult', 'ScanConfig', 'ScanResult', 'SelectConfig', 'SelectResult', 'SequenceStatsResult', 'StabilityConfig', 'StabilitySuiteResult', 'ThresholdSweepResult', 'TokenAggregationResult', 'TrendStabilityResult', 'VisualizationConfig', 'VisualizationResult', 'build_rel_icl_surface', 'find_sink_states', 'load_config', 'load_from_memmap', 'load_from_numpy', 'load_from_openact', 'load_inputs', 'prepare_inputs', 'run_ablation_suite', 'run_alignment_comparison', 'run_clustering_method_comparison', 'run_core_pipeline', 'run_discretize', 'run_dynamics', 'run_effective_rank', 'run_experiment', 'run_fit_vs_hidden', 'run_information_analysis', 'run_k_analysis', 'run_label_analysis', 'run_prediction', 'run_preliminary_suite', 'run_preprocessing_comparison', 'run_propagation_comparison', 'run_scan', 'run_select', 'run_sequence_stats', 'run_stability_suite', 'run_state_persistence', 'run_threshold_sweep', 'run_token_aggregation_comparison', 'run_tolerance_sweep', 'run_trend_stability', 'run_visualization_suite', 'save_config', 'summarize_k_map'
]

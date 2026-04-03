from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .core import BaseArtifacts
from .information import EffectiveRankResult, InformationResult, run_effective_rank, run_information_analysis
from .io_utils import ExperimentStore
from .label_analysis import LabelAnalysisResult, find_sink_states, run_label_analysis
from .statistics import DynamicsResult, SequenceStatsResult, run_dynamics, run_sequence_stats


@dataclass
class PreliminaryResult:
    dynamics: DynamicsResult
    sequence_stats: SequenceStatsResult
    information: Optional[InformationResult]
    effective_rank: EffectiveRankResult
    label_analysis: Optional[LabelAnalysisResult]
    sink_states: pd.DataFrame


def run_preliminary_suite(base: BaseArtifacts, *, store: Optional[ExperimentStore] = None) -> PreliminaryResult:
    gl = base.discretize.hss_result.global_labels
    y = base.loaded.y if len(base.loaded.y) > 0 else None
    dynamics = run_dynamics(base.discretize.hss_result, base.layers, y, store=store)
    sequence_stats = run_sequence_stats(gl, base.layers, store=store)
    information = run_information_analysis(gl, base.loaded.y, base.layers, store=store) if len(base.loaded.y) > 0 else None
    effective_rank = run_effective_rank(base.loaded.provider, layers=base.layers, store=store)
    label_analysis = run_label_analysis(gl, base.loaded.y, base.layers, store=store) if len(base.loaded.y) > 0 else None
    sink_states = find_sink_states(label_analysis) if label_analysis is not None else pd.DataFrame()
    if store is not None:
        if len(sink_states) > 0:
            store.save_csv('geometry/sink_states.csv', sink_states)
        store.save_json('preliminary/summary.json', {
            'self_trans_mean': float(sequence_stats.self_trans_mean),
            'active_mean': float(sequence_stats.active_mean),
            'n_sink_states': int(len(sink_states)),
            'has_information': information is not None,
            'has_labels': label_analysis is not None,
        })
    return PreliminaryResult(dynamics=dynamics, sequence_stats=sequence_stats, information=information, effective_rank=effective_rank, label_analysis=label_analysis, sink_states=sink_states)

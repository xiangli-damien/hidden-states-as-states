from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .ablation import AblationSuiteResult
from .config import ExperimentConfig
from .core import BaseArtifacts
from .data import LoadedInputs
from .io_utils import ExperimentStore
from .prediction import PredictionResult
from .preliminary import PreliminaryResult
from .stability import StabilitySuiteResult
from .visualization import VisualizationResult


@dataclass
class ExperimentOutputs:
    store: ExperimentStore
    base: BaseArtifacts
    preliminary: Optional[PreliminaryResult]
    prediction: Optional[PredictionResult]
    stability: Optional[StabilitySuiteResult]
    ablation: Optional[AblationSuiteResult]
    visualization: Optional[VisualizationResult]
    workspace: object


def run_experiment(cfg: ExperimentConfig, *, loaded: Optional[LoadedInputs] = None, token_aggregation_providers: Optional[Dict[str, object]] = None) -> ExperimentOutputs:
    from hss_workbench import Workspace

    ws = Workspace(cfg.output_dir, log_level=cfg.execution.log_level, quiet=cfg.execution.quiet)
    if loaded is not None:
        ws.load_data(loaded, persist=True)
    else:
        ws.load_data(config=cfg.data, persist=True)

    stages = {stage.lower() for stage in cfg.execution.stages}
    scan_artifact = ws.run_scan('scan', config=cfg.scan, overwrite=cfg.execution.overwrite)
    select_artifact = ws.run_select('select', scan=scan_artifact.name, config=cfg.select, overwrite=cfg.execution.overwrite)
    discretize_artifact = ws.run_discretize('discretize', select=select_artifact.name, config=cfg.discretize, overwrite=cfg.execution.overwrite)
    base = ws.build_base(discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name)

    preliminary = None
    prediction = None
    stability = None
    ablation = None
    visualization = None

    if 'preliminary' in stages:
        ws.run_preliminary('preliminary', discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, overwrite=cfg.execution.overwrite)
        preliminary = ws.load_preliminary('preliminary')
    if 'prediction' in stages and len(ws.open_data().y) > 0:
        ws.run_prediction('prediction', discretize=discretize_artifact.name, config=cfg.prediction, overwrite=cfg.execution.overwrite)
        prediction = ws.load_prediction('prediction')
    if 'stability' in stages:
        ws.run_stability('stability', discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, discretize_config=cfg.discretize, stability_config=cfg.stability, overwrite=cfg.execution.overwrite)
        stability = ws.load_stability('stability')
    if 'ablation' in stages:
        ws.run_ablation('ablation', discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, discretize_config=cfg.discretize, ablation_config=cfg.ablation, token_aggregation_providers=token_aggregation_providers, overwrite=cfg.execution.overwrite)
        ablation = ws.load_analysis('ablation', typed=False)
    if 'visualization' in stages and cfg.execution.save_figures:
        ws.run_visualization('visualization', discretize=discretize_artifact.name, select=select_artifact.name, scan=scan_artifact.name, preliminary='preliminary' if 'preliminary' in stages else None, prediction='prediction' if 'prediction' in stages and len(ws.open_data().y) > 0 else None, stability='stability' if 'stability' in stages else None, ablation='ablation' if 'ablation' in stages else None, config=cfg.visualization, overwrite=cfg.execution.overwrite)
        visualization = ws.load_visualization('visualization')

    return ExperimentOutputs(
        store=ExperimentStore(cfg.output_dir),
        base=base,
        preliminary=preliminary,
        prediction=prediction,
        stability=stability,
        ablation=ablation,
        visualization=visualization,
        workspace=ws,
    )

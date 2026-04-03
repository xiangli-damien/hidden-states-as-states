from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for path in [str(ROOT), str(SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from hss_workbench import Workspace

from .config import DiscretizeConfig, ExperimentConfig, PredictionConfig, ScanConfig
from .data import LoadedInputs, load_from_numpy
from .runner import run_experiment


def smoke_test_workspace(seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    states = rng.randn(24, 2, 4).astype(np.float32)
    y = rng.randint(0, 2, size=24).astype(np.int32)
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(tmpdir)
        ws.load_data(states, y=y, layers=[0, 1])
        ws.run_scan('gmm_default', config=ScanConfig(k_range=(2, 2), method='kmeans', stability_repeats=1))
        ws.run_select('icl_default', scan='gmm_default')
        ws.run_discretize('final_v1', select='icl_default', config=DiscretizeConfig(method='kmeans'))
        ws.run_prediction('prediction_v1', discretize='final_v1', config=PredictionConfig(methods=['HSS-NB', 'GaussianNB'], n_splits=3))
        gl = ws.load_discretize('final_v1').global_labels
        pred = ws.load_prediction('prediction_v1')
        assert gl is not None and gl.shape[0] == len(y)
        assert len(pred.summary_df) > 0
        ws2 = Workspace(tmpdir)
        gl2 = ws2.load_discretize('final_v1').global_labels
        assert gl2.shape == gl.shape


def smoke_test_runner(seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    states = rng.randn(24, 2, 4).astype(np.float32)
    y = rng.randint(0, 2, size=24).astype(np.int32)
    provider, y_vec, labels = load_from_numpy(states, layer_ids=[0, 1], labels={'correct': y})
    loaded = LoadedInputs(provider=provider, y=y_vec, label_dict=labels, layers=[0, 1], n_items=provider.n_items(), state_dim=provider.state_dim())
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ExperimentConfig(
            output_dir=tmpdir,
            scan=ScanConfig(k_range=(2, 2), method='kmeans', stability_repeats=1),
            discretize=DiscretizeConfig(method='kmeans'),
            prediction=PredictionConfig(methods=['HSS-NB', 'GaussianNB'], n_splits=3),
            execution=replace(ExperimentConfig().execution, stages=['preliminary', 'prediction']),
        )
        outputs = run_experiment(cfg, loaded=loaded)
        assert outputs.base.discretize.hss_result.global_labels is not None
        assert outputs.prediction is not None and len(outputs.prediction.summary_df) > 0


if __name__ == '__main__':
    smoke_test_workspace()
    smoke_test_runner()
    print('PASS')

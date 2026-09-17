import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from fixtures import openact_shards

from hss.cluster.gmm import GMMModel, _fit_gmm
from hss.data import DataSpec
from hss.experiments.config import (
    ClusterConfig,
    EvaluationConfig,
    ExecutionConfig,
    Experiment,
)
from hss.experiments.parallel import prefit_layers
from hss.experiments.runner import run_experiment


def test_prefit_uses_same_train_only_cache_as_serial(tmp_path):
    cfg = Experiment(
        name="prediction",
        data=DataSpec(
            [str(openact_shards(tmp_path / "source"))],
            representation="prompt_last",
            min_free_gib=0,
        ),
        cluster=ClusterConfig(k_values=[2, 3], max_iter=4, n_init=1),
        evaluation=EvaluationConfig(mode="prediction", methods=["HSS-NB"]),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "results"),
            memory_gib=4,
            min_available_gib=0,
        ),
    )
    seen = []
    snapshots = prefit_layers(
        [cfg],
        tmp_path / "progress",
        workers=2,
        on_ready=lambda c, p: seen.append(c.name),
    )
    assert seen == ["prediction"]
    fits = list((tmp_path / "cache/fits").glob("*/fit.json"))
    assert len(fits) == 6
    stamps = {str(p): p.stat().st_mtime_ns for p in fits}
    first = run_experiment(cfg, prepared=snapshots[cfg.name])
    assert stamps == {str(p): p.stat().st_mtime_ns for p in fits}
    second = run_experiment(
        replace(
            cfg,
            execution=replace(
                cfg.execution,
                cache_root=str(tmp_path / "independent_cache"),
                output_root=str(tmp_path / "independent_results"),
            ),
        )
    )
    np.testing.assert_array_equal(
        np.load(Path(first["path"]) / "states.npy"),
        np.load(Path(second["path"]) / "states.npy"),
    )
    assert (
        json.loads((Path(first["path"]) / "selection.json").read_text())[0]["selected"][
            "k"
        ]
        >= 2
    )


def test_gmm_cpu_records_convergence_and_restores_legacy():
    X = np.random.default_rng(7).normal(size=(30, 4))
    model = _fit_gmm(X, 2, 42, backend="cpu", adaptive_reg=False, n_init=1, max_iter=1)
    assert model.config()["converged"] is False
    assert model.config()["n_iter"] == 1
    assert GMMModel.from_state(model.config(), model.state_arrays()).n_iter_ == 1
    legacy = model.config()
    del legacy["converged"], legacy["n_iter"]
    assert GMMModel.from_state(legacy, model.state_arrays()).converged_ is None

from dataclasses import replace
import numpy as np
import pytest

from hss.experiments.config import ClusterConfig, Experiment
from hss.experiments.fitting import fit_one_candidate, select_candidate, load_fitted


def test_unconverged_minimum_cannot_set_parsimony_reference():
    cfg = ClusterConfig(require_convergence=True, parsimony_tolerance=0.02)
    records = [
        dict(k=2, criterion=100, icl=100, bic=10, converged=True),
        dict(k=5, criterion=99, icl=99, bic=9, converged=True),
        dict(k=8, criterion=-10000, icl=-10000, bic=-10001, converged=False),
    ]
    assert select_candidate(records, cfg)["k"] == 2
    assert select_candidate(records, replace(cfg, parsimony_tolerance=0))["k"] == 5
    assert select_candidate(records, replace(cfg, selection_criterion="bic"))["k"] == 5
    with pytest.raises(ValueError, match="No converged"):
        select_candidate(records[-1:], cfg)


def test_tolerance_and_bic_reuse_fit_and_preserve_both_assignments(tmp_path):
    X = np.random.default_rng(3).normal(size=(70, 6))
    cfg = Experiment()
    cfg.cluster = ClusterConfig(
        method="mfa",
        rank=1,
        mfa_init="svd",
        n_init=2,
        max_iter=5,
        save_candidate_assignments=True,
    )
    first = fit_one_candidate(X, cfg, {"layer": 3}, tmp_path, 2)
    from pathlib import Path

    path = Path(first["fit_path"])
    stamp = (path / "model.npz").stat().st_mtime_ns
    cfg.cluster = replace(cfg.cluster, parsimony_tolerance=0, selection_criterion="bic")
    second = fit_one_candidate(X, cfg, {"layer": 3}, tmp_path, 2)
    assert first["fit_path"] == second["fit_path"]
    assert (path / "model.npz").stat().st_mtime_ns == stamp
    model, _ = load_fitted(path)
    with np.load(path / "assignments.npz") as z:
        np.testing.assert_array_equal(z["posterior"], model.predict(X))
        assert z["nearest"].shape == (70,)
    assert len(list(path.glob("restart_*.npz"))) == 2

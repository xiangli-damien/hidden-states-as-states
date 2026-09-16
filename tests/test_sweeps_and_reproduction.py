from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest

from fixtures import openact_shards
from hss.experiments.config import (
    Experiment,
    ClusterConfig,
    EvaluationConfig,
    ExecutionConfig,
    load,
)
from hss.experiments.openact import DataSpec, prepare
from hss.experiments.runner import run_experiment, _load_layer
from hss.experiments.sweep import expand, plan, run_sweep
from hss.experiments.reporting import report
from hss.experiments.paper import generate_suite
from hss.experiments.fitting import selection_metrics
from hss.cluster.mfa import fit_mfa


def config(tmp):
    source = openact_shards(tmp / "source")
    return Experiment(
        data=DataSpec([str(source)], min_free_gib=0),
        cluster=ClusterConfig(k_values=[2, 3], n_init=1, max_iter=4),
        execution=ExecutionConfig(
            cache_root=str(tmp / "cache"),
            output_root=str(tmp / "outputs"),
            workers=2,
            min_available_gib=0,
            memory_gib=4,
        ),
    )


def test_grid_reuses_fits_frozen_snapshot_resume_and_report(tmp_path):
    cfg = config(tmp_path)
    cfg.grid = {"alignment.threshold": [0.3, 0.9]}
    first = run_sweep(cfg)
    assert (
        first["status"] == "complete" and first["trials"] == 2 and first["workers"] == 2
    )
    fits = list((tmp_path / "cache" / "fits").glob("*/fit.json"))
    assert len(fits) == 6  # 3 layers x 2 K; eta does not trigger duplicate fitting.
    stamps = {str(p): p.stat().st_mtime_ns for p in fits}
    second = run_sweep(cfg)
    assert all(r["cache_hit"] for r in second["results"])
    assert stamps == {str(p): p.stat().st_mtime_ns for p in fits}
    prepared = plan(cfg)
    assert prepared["candidate_fits_upper_bound"] == 6
    assert prepared["candidate_array_storage_gib_upper_bound"] > 0
    receipt = next((tmp_path / "source").glob("*/_COPY_VERIFIED.json"))
    receipt.write_text('{"changed":true}')
    assert plan(cfg)["data_snapshots"] == prepared["data_snapshots"]
    assert plan(cfg, refresh_data=True)["data_snapshots"] != prepared["data_snapshots"]
    rendered = report(tmp_path / "outputs")
    assert rendered["experiments"] == 2
    assert len(list(Path(rendered["path"]).glob("*/state_graph.png"))) == 2
    sharded = replace(cfg, execution=replace(cfg.execution, task_index=1, task_count=2))
    assert plan(sharded)["plan_id"] == plan(cfg)["plan_id"]
    assert run_sweep(sharded)["trials"] == 1


def test_test_features_cannot_change_fitted_map(tmp_path):
    cfg = config(tmp_path)
    cfg.data.representation = "prompt_last"
    cfg.evaluation = EvaluationConfig(mode="prediction", methods=["HSS-NB"])
    first = run_experiment(cfg)
    split = np.load(Path(first["path"]) / "split.npz")
    data = prepare(cfg.data, cfg.execution.cache_root)
    # Adversarially change only held-out features in a copied snapshot. This test
    # bypasses immutable source receipts deliberately to isolate fitting behavior.
    import shutil

    copied = tmp_path / "modified_snapshot"
    shutil.copytree(data.path, copied)
    marker = json.loads((copied / "_SUCCESS.json").read_text())
    marker["key"] = "heldout_modified"
    (copied / "_SUCCESS.json").write_text(json.dumps(marker))
    for layer in data.layers():
        X = np.load(copied / f"layer_{layer}.npy", mmap_mode="r+")
        X[split["test"]] += 1000
        X.flush()
    second = run_experiment(cfg, prepared=copied)
    for layer in data.layers():
        a, pa, sa = _load_layer(first["path"], layer)
        b, pb, sb = _load_layer(second["path"], layer)
        np.testing.assert_array_equal(a.centers(), b.centers())
        np.testing.assert_array_equal(pa.mean, pb.mean)
        assert sa["selected"]["k"] == sb["selected"]["k"]


def test_config_contract_suite_and_failure_resume(tmp_path):
    cfg = config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load(path, ['cluster.method="mfa"', "cluster.rank=2"]).cluster.rank == 2
    with pytest.raises(ValueError, match="Unknown config"):
        load(path, ["cluster.typo=1"])
    with pytest.raises(TypeError):
        Experiment.from_dict({**cfg.to_dict(), "typo": True})
    cfg.grid = {"execution.workers": [1, 2]}
    with pytest.raises(ValueError, match="Execution limits"):
        expand(cfg)
    cfg.grid = {}
    cfg.cluster.k = 1000
    assert run_sweep(cfg)["status"] == "failed"
    cfg.execution.retry_failed = False
    assert run_sweep(cfg)["results"][0]["status"] == "skipped_failed"
    generate_suite("/missing", tmp_path / "paper", tmp_path / "cache", tmp_path / "out")
    manifest = json.loads((tmp_path / "paper" / "suite.json").read_text())
    assert len(manifest["jobs"]) >= 35
    for job in manifest["jobs"]:
        load(tmp_path / "paper" / job["config"])
    assert any("belebele_pooled" in j["name"] for j in manifest["jobs"])


def test_icl_sign_and_entropy():
    X = np.random.default_rng(7).normal(size=(40, 3))
    model = fit_mfa(X, 2, rank=1, n_init=1, max_iter=4)
    score = selection_metrics(model, X, "mfa", 9, 42)
    assert score["entropy"] > 0
    assert score["criterion"] == pytest.approx(model.bic(X) + 2 * score["entropy"])


def test_global_control_and_fixed_k_suite_dependency(tmp_path):
    from hss.experiments.paper import run_suite

    cfg = config(tmp_path)
    cfg.cluster.k = 2
    cfg.evaluation.mode = "global_control"
    result = run_experiment(cfg)
    assert (Path(result["path"]) / "global_layer_purity.csv").exists()
    cfg.evaluation.mode = "geometry"
    cfg.grid = {}
    base = tmp_path / "base.json"
    base.write_text(json.dumps(cfg.to_dict()))
    target = replace(cfg, seed=43)
    child = tmp_path / "child.json"
    child.write_text(json.dumps(target.to_dict()))
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": "base", "config": "base.json", "depends_on": {}},
                    {
                        "name": "refit",
                        "config": "child.json",
                        "depends_on": {
                            "evaluation.fixed_k_map": {
                                "job": "base",
                                "file": "selection.json",
                            }
                        },
                    },
                ]
            }
        )
    )
    suite = run_suite(manifest, only=["refit"])
    assert suite["status"] == "complete"
    assert set(suite["jobs"]) == {"base", "refit"}
    reference = suite["jobs"]["base"]["results"][0]["path"]
    cfg.evaluation.fixed_map = reference
    reused = run_experiment(cfg)
    np.testing.assert_array_equal(
        np.load(Path(reference) / "states.npy"),
        np.load(Path(reused["path"]) / "states.npy"),
    )

from dataclasses import replace
import numpy as np
from hss.experiments.cluster_ablation import ClusterStudy, refinement_k
from hss.experiments.config import ClusterConfig
from test_sweeps_and_reproduction import config


def test_refinement_covers_parsimonious_and_minimum_icl_without_repeating():
    records = [
        dict(k=2, criterion=-980, icl=-980, bic=-985, converged=True),
        dict(k=8, criterion=-1000, icl=-1000, bic=-1005, converged=True),
        dict(k=16, criterion=-9000, icl=-9000, bic=-9005, converged=False),
    ]
    cfg = ClusterConfig(method="mfa", k_max=20, require_convergence=True)
    assert refinement_k(records, cfg, [0, 0.02], ["icl"], 1) == [3, 7, 9]


def test_small_study_exports_and_resumes_without_refitting(tmp_path):
    cfg = config(tmp_path)
    cfg.data.expected_samples = None
    cfg.cluster = replace(
        cfg.cluster,
        k_min=2,
        k_max=2,
        k_values=None,
        n_init=1,
        max_iter=30,
        tol=0.1,
        require_convergence=True,
        save_candidate_assignments=True,
        mfa_init="svd",
    )
    cfg.execution = replace(cfg.execution, threads_per_worker=1)
    recipe = dict(
        ranks=[0],
        icl_tolerances=[0, 0.02],
        criteria=["icl", "bic"],
        reference_tolerance=0.02,
        include_pre_final=False,
        kmeans_n_init=2,
        mfa_initial_k=[2],
        refine_radius=1,
        refine_rounds=1,
        stability_seeds=[43],
    )
    study = ClusterStudy(recipe, cfg, tmp_path / "study")
    study.run()
    assert study.state["phase"] == "finished"
    assert any(r["method"] == "mfa" for r in study.records.values())
    assert any(r["seed"] == 43 for r in study.records.values())
    tasks = [
        study.task("post", layer, "gmm", 0, 2)
        for layer in study.snapshots["post"].layers()
    ]
    assert all(r["status"] == "complete" for r in study.records.values())
    stamps = {
        p: p.stat().st_mtime_ns
        for p in (tmp_path / "study/cache/fits").glob("*/model.npz")
    }
    study.export_available("baseline")
    assert study.state["jobs"]
    study.run_tasks("baseline", tasks)
    assert stamps == {p: p.stat().st_mtime_ns for p in stamps}
    for item in study.records.values():
        from pathlib import Path

        with np.load(Path(item["record"]["fit_path"]) / "assignments.npz") as labels:
            assert len(labels["posterior"]) == len(study.snapshots["post"].meta)
    assert (study.root / "selection_ablation.csv").exists()

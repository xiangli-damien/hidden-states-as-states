import pandas as pd
from fixtures import openact_shards

from hss.data import DataSpec
from hss.experiments.artifacts import save_json
from hss.experiments.config import ClusterConfig, ExecutionConfig, Experiment
from hss.experiments.runner import run_experiment
from hss.viz.review import javascript, render_review


def test_review_contains_real_cases_and_safe_script_data(tmp_path):
    source = openact_shards(tmp_path / "source", n=5)
    shard = sorted(source.glob("shard_*"))[0]
    frame = pd.read_parquet(shard / "data.parquet")
    frame.loc[0, "response_text"] = '</script><img src=x onerror="window.pwned=1">'
    frame.to_parquet(shard / "data.parquet", index=False)
    cfg = Experiment(
        name="mean_gmm",
        data=DataSpec([str(source)], min_free_gib=0, expected_samples=10),
        cluster=ClusterConfig(k=2, max_iter=3, n_init=1),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "trials"),
            memory_gib=4,
            min_available_gib=0,
        ),
    )
    r = run_experiment(cfg)
    study = tmp_path / "study"
    save_json(study / "configs/mean_gmm.json", cfg.to_dict())
    save_json(
        study / "study.json",
        dict(
            status="stage_complete",
            jobs={"mean_gmm": dict(status="complete", path=r["path"])},
        ),
    )
    review = render_review(study, tmp_path / "review", max_trajectories=10, bootstrap=5)
    assert review["overview"]["n"] == 10
    data = (tmp_path / "review/cases.js").read_text()
    assert "</script>" not in data and "\\u003c/script\\u003e" in data
    states = (tmp_path / "review/trajectories.js").read_text()
    assert "math_0" in states
    quality = pd.read_csv(tmp_path / "review/comparison/cluster_quality.csv")
    assert len(quality) == 3 and quality.silhouette.notna().all()
    assert review["completed"] == ["mean_gmm"]
    inventory = tmp_path / "review/mean_gmm/manifest.json"
    stamp = inventory.stat().st_mtime_ns
    render_review(study, tmp_path / "review", max_trajectories=10, bootstrap=5)
    assert inventory.stat().st_mtime_ns == stamp
    assert "<img" not in javascript({"x": "<img>"})

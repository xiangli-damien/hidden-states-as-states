import numpy as np
import pandas as pd

from hss.experiments.config import (
    Experiment,
    ClusterConfig,
    EvaluationConfig,
    ExecutionConfig,
)
from hss.experiments.openact import DataSpec
from hss.experiments.evaluate import (
    grouped_split,
    calibrate_threshold,
    monitoring_metrics,
)
from hss.experiments.runner import run_experiment
from fixtures import openact_shards


def test_prefix_split_has_no_response_overlap_and_calibration_is_response_level():
    meta = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(40), 3),
            "label": np.repeat(np.arange(40) % 2, 3),
        }
    )
    split = grouped_split(meta, EvaluationConfig(mode="monitoring"))
    groups = {k: set(meta.iloc[v].group_id) for k, v in split.items()}
    assert [len(groups[k]) for k in ("train", "validation", "test")] == [16, 8, 16]
    assert not groups["train"] & groups["test"]
    scores = np.tile([0.1, 0.2, 0.9], 40)
    threshold, far = calibrate_threshold(
        meta.iloc[split["validation"]], scores[split["validation"]], 0.1
    )
    assert threshold == 0.9 and far == 0
    sample = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1],
            "label": [0, 0, 1, 1],
            "token_end": [5, 10, 5, 10],
            "n_tokens": [10] * 4,
            "is_final": [False, True] * 2,
        }
    )
    result, _ = monitoring_metrics(sample, np.array([0.1, 0.99, 0.1, 0.1]), 0.9)
    assert result["early_detection_rate"] == 0  # final-only failure alarm is not early.
    assert result["test_far"] == 0


def test_end_to_end_mfa_prediction_export_and_resume(tmp_path):
    source = openact_shards(tmp_path / "source")
    cfg = Experiment(
        name="test",
        data=DataSpec([str(source)], representation="prompt_last", min_free_gib=0),
        cluster=ClusterConfig(
            method="mfa", k=2, rank=1, n_init=1, max_iter=8, chunk_size=10
        ),
        evaluation=EvaluationConfig(mode="prediction", methods=["HSS-NB", "Logistic"]),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"), output_root=str(tmp_path / "out")
        ),
    )
    result = run_experiment(cfg)
    assert len(result["evaluation"]) == 2
    assert result["n_rows"] == 40 and not result["cache_hit"]
    assert run_experiment(cfg)["cache_hit"]
    from pathlib import Path

    split = np.load(Path(result["path"]) / "split.npz")
    assert len(split["train"]) == 16 and len(split["test"]) == 24
    assert not set(split["map_fit"]) & set(split["test"])


def test_monitoring_records_missing_output_baselines_without_future_leakage(tmp_path):
    source = openact_shards(tmp_path / "source")
    cfg = Experiment(
        data=DataSpec([str(source)], representation="prefix", min_free_gib=0),
        cluster=ClusterConfig(method="kmeans", k=2, n_init=1, max_iter=10),
        evaluation=EvaluationConfig(mode="monitoring"),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"), output_root=str(tmp_path / "out")
        ),
    )
    result = run_experiment(cfg)
    assert len(result["unavailable_baselines"]) == 2
    assert all(m["validation_far"] <= 0.1 for m in result["evaluation"])

from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from hss.analysis.tables import (
    state_map,
    trajectory_similarity,
    selection_surface,
    bootstrap_predictions,
)
from hss.viz import plots
from hss.viz.paper import render_paper, RECIPES
from hss.experiments.paper import generate_suite
from hss.experiments.config import (
    Experiment,
    ClusterConfig,
    ExecutionConfig,
    EvaluationConfig,
)
from hss.experiments.runner import run_experiment
from hss.data import DataSpec
from hss.results import Result
from fixtures import openact_shards


def test_state_graph_entropy_and_hamming_not_euclidean():
    states = np.array([[0, 2], [0, 3], [1, 3], [1, 3]])
    rows = pd.DataFrame(
        {"sample_id": ["a", "b", "c", "d"], "token_end": [2] * 4, "label": [1, 0, 1, 1]}
    )
    fake = SimpleNamespace(states=states, rows=rows, layers=[0, 1])
    nodes, edges = state_map(fake)
    assert nodes[(nodes.layer == 0) & (nodes.state == 0)].entropy.iloc[0] == 1
    assert nodes[(nodes.layer == 0) & (nodes.state == 1)].entropy.iloc[0] == 0
    assert nodes[nodes.layer == 1].entropy.isna().all()
    assert edges["count"].sum() == 4
    matrix, order = trajectory_similarity(fake, max_rows=4)
    expected = (
        states[order.result_row.to_numpy(), None, :]
        == states[None, order.result_row.to_numpy(), :]
    ).mean(axis=2)
    np.testing.assert_array_equal(matrix, expected)
    with pytest.raises(ValueError):
        trajectory_similarity(fake, linkage_method="ward")
    fig = plots.state_graph(nodes, edges)
    assert fig.axes[0].get_xlabel() == "Layer"
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_paper_coverage_missing_is_explicit_and_render_has_input_inventory(tmp_path):
    generated = generate_suite(
        "/missing", tmp_path / "study", tmp_path / "cache", tmp_path / "trials"
    )
    empty = render_paper(
        tmp_path / "trials",
        generated["suite"],
        tmp_path / "empty",
        only=["figure_03"],
        strict=True,
    )
    assert (
        empty["status"] == "failed"
        and empty["coverage"][0]["status"] == "missing_inputs"
    )
    source = openact_shards(tmp_path / "source")
    cfg = Experiment(
        data=DataSpec([str(source)], dataset_id="math", min_free_gib=0),
        cluster=ClusterConfig(k_values=[2, 3], n_init=1, max_iter=5),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "trials"),
            min_available_gib=0,
        ),
    )
    result = run_experiment(cfg)
    (tmp_path / "study/suite_status.json").write_text(
        json.dumps(
            {
                "default_map": {"status": "complete", "results": [result]},
                "qwen_math_map": {"status": "complete", "results": [result]},
            }
        )
    )
    rendered = render_paper(
        tmp_path / "trials",
        generated["suite"],
        tmp_path / "figures",
        only=[
            "figure_03",
            "figure_04",
            "figure_06",
            "figure_09",
            "figure_11",
            "figure_12",
        ],
        formats=["png"],
        max_trajectories=8,
    )
    assert rendered["status"] == "complete"
    manifest = json.loads((tmp_path / "figures/figure_04/manifest.json").read_text())
    assert manifest["inputs"][0]["trial_id"] == result["trial_id"]
    assert manifest["inputs"][0]["artifact_inventory_sha256"]
    assert manifest["parameters"]["max_trajectories"] == 8
    assert "trajectory_similarity.npy" in manifest["outputs"]
    assert len(pd.read_csv(tmp_path / "figures/figure_04/trajectory_order.csv")) == 8
    scan = selection_surface(Result(result["path"]))
    fig = plots.icl_surface(scan)
    assert (
        fig.axes[0].get_xlabel() == "Candidate K"
        and fig.axes[0].get_ylabel() == "Layer"
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert set(RECIPES) == {f"figure_{i:02}" for i in range(1, 13)} | {
        f"table_{i:02}" for i in range(1, 4)
    }


def test_bootstrap_has_fixed_response_unit(tmp_path):
    source = openact_shards(tmp_path / "source")
    cfg = Experiment(
        data=DataSpec([str(source)], representation="prompt_last", min_free_gib=0),
        cluster=ClusterConfig(k=2, n_init=1, max_iter=5),
        evaluation=EvaluationConfig(mode="prediction", methods=["HSS-NB"]),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "trials"),
            min_available_gib=0,
        ),
    )
    result = Result(run_experiment(cfg)["path"])
    first = bootstrap_predictions(result, 30)
    second = bootstrap_predictions(result, 30)
    pd.testing.assert_frame_equal(first, second)
    assert (first.lower_95 <= first.upper_95).all() and set(first.unit) == {
        "held-out response"
    }

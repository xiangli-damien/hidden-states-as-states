"""Scientific display invariants: normalization, filtering, provenance and refits."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hss.analysis.tables import state_map
from hss.viz.artifacts import FigureBundle
from hss.viz.publication import graph_display, refit_summary, state_graph


def test_display_filters_preserve_counts_entropy_and_edge_labels(tmp_path, monkeypatch):
    # Two equally likely observed successors; the other layer also has a third
    # state that this source never reaches. Normalize by 2, not all 3 states.
    result = SimpleNamespace(
        states=np.array([[0, 2], [0, 3], [1, 4], [1, 4]]),
        rows=pd.DataFrame({"label": [1, 0, 1, 1]}),
        layers=[0, 1],
    )
    nodes, edges = state_map(result)
    before = edges.copy(deep=True)
    cfg = tmp_path / "style.toml"
    cfg.write_text(
        "[publication]\nedge_min_count=1\nedge_min_probability=0\nedge_max_per_source=1\ncorrectness_limit=0.5\n"
    )
    monkeypatch.setenv("HSS_FIGURE_STYLE", str(cfg))
    nd, ed = graph_display(nodes, edges)
    assert nd.query("layer == 0 and state == 0").entropy_normalized.item() == 1
    assert nd.query("layer == 0 and state == 1").entropy_normalized.item() == 0
    assert nd.query("layer == 1").entropy_normalized.isna().all()
    assert ed["count"].sum() == 4 and ed.loc[ed.displayed, "count"].sum() == 3
    assert edges.query("to_state == 2").accuracy.item() == 1
    assert edges.query("to_state == 3").accuracy_delta.item() == -0.75
    pd.testing.assert_frame_equal(edges, before)
    fig = state_graph(nodes, edges, "accuracy_delta")
    assert len(fig.axes[0].texts) == len(nodes)  # no states silently removed
    assert fig._hss_encoding["transition_mass_retained"] == 0.75
    assert fig._hss_encoding["color_limits"] == [-0.5, 0.5]
    bundle = FigureBundle(tmp_path / "output", formats=("png",))
    bundle.figure("states", fig)
    saved = bundle.finish()
    assert saved["parameters"]["publication"]["correctness_limit"] == 0.5
    assert saved["visual_encodings"]["states"]["displayed_edges"] == 2
    assert json.loads((bundle.path / "manifest.json").read_text()) == saved


def test_refit_view_compares_to_declared_baseline_in_either_orientation():
    frame = pd.DataFrame(
        [
            dict(
                method="gmm",
                reference="base",
                target="seed",
                reference_seed=42,
                target_seed=43,
                reference_fit_fraction=1,
                target_fit_fraction=1,
            ),
            dict(
                method="gmm",
                reference="half",
                target="base",
                reference_seed=42,
                target_seed=42,
                reference_fit_fraction=0.5,
                target_fit_fraction=1,
            ),
            dict(
                method="gmm",
                reference="seed",
                target="half",
                reference_seed=43,
                target_seed=42,
                reference_fit_fraction=1,
                target_fit_fraction=0.5,
            ),
        ]
    )
    rows = refit_summary(frame, {"gmm": "base"})
    assert len(rows) == 2
    assert rows.refit_seed.tolist() == [43, 42]
    assert rows.refit_fraction.tolist() == [1, 0.5]
    assert rows.perturbation.tolist() == ["Seed", "Subsample"]


def test_single_layer_has_no_fake_entropy_or_edges():
    r = SimpleNamespace(
        states=np.zeros((3, 1), dtype=int),
        rows=pd.DataFrame({"label": [0, 1, 1]}),
        layers=[0],
    )
    nodes, edges = state_map(r)
    nd, ed = graph_display(nodes, edges)
    assert ed.empty and nd.entropy_normalized.isna().all()
    fig = state_graph(nodes, edges)
    assert fig._hss_encoding["displayed_edges"] == 0
    plt.close(fig)

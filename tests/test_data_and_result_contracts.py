from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import zarr

from hss.data import DataSpec, prepare
from hss.data.arrays import export_arrays
from hss.data.catalog import load_catalog, inspect_catalog
from hss.experiments.config import Experiment, ClusterConfig, ExecutionConfig
from hss.experiments.runner import run_experiment
from hss.results import Result, ResultCatalog, load_layer
from hss.provenance import stage_version
from hss.experiments.evaluate import (
    calibrate_threshold,
    monitoring_metrics,
    characterize,
)
from fixtures import openact_shards


def array_fixture(tmp_path):
    rng = np.random.default_rng(42)
    rows = pd.DataFrame(
        {
            "sample_id": [f"x{i}" for i in range(40)],
            "label": np.arange(40) % 2,
            "category": np.where(np.arange(40) % 2, "algebra", "geometry"),
        }
    )
    matrices = {
        j: rng.normal(size=(40, 6)) + (np.arange(40) % 2)[:, None] * 2
        for j in (0, 3, 8)
    }
    source = export_arrays(
        tmp_path / "source",
        matrices,
        rows,
        model=["test-model", "revision"],
        dataset_id="math",
    )
    return source, matrices, rows


def test_arrays_cache_labels_interchange_and_model_rejection(tmp_path):
    source, matrices, rows = array_fixture(tmp_path)
    spec = DataSpec([str(source)], source_format="arrays", min_free_gib=0)
    cache = prepare(spec, tmp_path / "cache")
    assert cache.info["dataset_id"] == "math"
    assert cache.n_items() == 40 and cache.meta.label.tolist() == rows.label.tolist()
    assert isinstance(cache.array(3), np.memmap)
    np.testing.assert_allclose(cache.array(3), matrices[3], rtol=1e-6)
    assert prepare(spec, tmp_path / "cache").path == cache.path
    with pytest.raises(ValueError, match="Unexpected"):
        prepare(replace(spec, expected_model="other"), tmp_path / "cache")
    with pytest.raises(ValueError, match="RMS"):
        prepare(replace(spec, final_norm="pre"), tmp_path / "cache")
    with pytest.raises(ValueError, match="Expected"):
        prepare(replace(spec, expected_samples=41), tmp_path / "cache")
    duplicate = rows.copy()
    duplicate.loc[1, "sample_id"] = "x0"
    duplicate.to_parquet(source / "rows.parquet")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(spec, tmp_path / "cache")


def test_sentence_is_local_span_and_prefix_is_cumulative(tmp_path):
    source = openact_shards(tmp_path / "source")
    spec = DataSpec(
        [str(source)],
        representation="sentence",
        batch_tokens=2,
        final_norm="pre",
        min_free_gib=0,
    )
    cache = prepare(spec, tmp_path / "cache")
    z = zarr.open_group(str(source / "shard_00000_00020/tensors.zarr"), mode="r")
    part = cache.meta[cache.meta.group_id == 0]
    start = 0
    for idx, end in zip(part.index, part.token_end):
        np.testing.assert_allclose(
            cache.array(2)[idx],
            z["final_norm/pre/per_token"][start:end].mean(0),
            atol=1e-6,
        )
        start = end
    assert start == 6


def test_array_prefix_requires_complete_ordered_response_boundaries(tmp_path):
    rows = pd.DataFrame(
        {
            "sample_id": ["x", "x"],
            "token_end": [3, 2],
            "n_tokens": [3, 3],
            "label": [1, 1],
        }
    )
    source = export_arrays(
        tmp_path / "source",
        {0: np.ones((2, 2))},
        rows,
        model=["m", "r"],
        representation="prefix",
    )
    with pytest.raises(ValueError, match="Boundaries"):
        prepare(
            DataSpec(
                [str(source)],
                source_format="arrays",
                representation="prefix",
                min_free_gib=0,
            ),
            tmp_path / "cache",
        )


def test_catalog_relative_paths_and_count_only_audit(tmp_path):
    source, _, _ = array_fixture(tmp_path)
    catalog = tmp_path / "catalog.toml"
    catalog.write_text(
        'schema_version=1\n[datasets."model/math"]\npaths=["source"]\nsource_format="arrays"\nexpected_samples=40\n'
    )
    assert load_catalog(catalog)["model/math"]["paths"] == [str(source)]
    assert inspect_catalog(catalog)["datasets"]["model/math"]["published_samples"] == 40


def test_stage_fingerprints_do_not_refit_on_figure_changes(tmp_path):
    root = tmp_path / "hss"
    (root / "viz").mkdir(parents=True)
    (root / "cluster").mkdir()
    (root / "viz/plots.py").write_text("original plot")
    (root / "cluster/mfa.py").write_text("original fit")
    before = {s: stage_version(s, root) for s in ("data", "fit", "trial", "figure")}
    (root / "viz/plots.py").write_text("new colors")
    assert stage_version("figure", root) != before["figure"]
    for s in ("data", "fit", "trial"):
        assert stage_version(s, root) == before[s]
    (root / "cluster/mfa.py").write_text("new optimizer")
    assert (
        stage_version("fit", root) != before["fit"]
        and stage_version("trial", root) != before["trial"]
    )


def test_result_can_move_without_raw_data_and_detects_corruption(tmp_path, monkeypatch):
    source, _, _ = array_fixture(tmp_path)
    cfg = Experiment(
        data=DataSpec([str(source)], source_format="arrays", min_free_gib=0),
        cluster=ClusterConfig(
            method="minibatch_kmeans", k=2, n_init=1, max_iter=8, batch_size=10
        ),
        execution=ExecutionConfig(
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "out"),
            min_available_gib=0,
        ),
    )
    summary = run_experiment(cfg)
    moved = tmp_path / "portable" / summary["trial_id"]
    shutil.copytree(summary["path"], moved)
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "cache")
    shutil.rmtree(tmp_path / "out")
    r = Result(moved)
    assert r.validate(full=True)["valid"]
    assert r.dataset == "math" and load_layer(moved, 3)[0].n_clusters() == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("A render may not fit or prepare")

    monkeypatch.setattr("hss.experiments.runner.run_experiment", forbidden)
    monkeypatch.setattr("hss.experiments.fitting.fit_candidates", forbidden)
    monkeypatch.setattr("hss.data.prepare", forbidden)
    from hss.experiments.reporting import report

    result = report(moved, formats=("png",))
    assert Path(result["path"]).parent.name == "reports"
    assert r.validate(full=True)["valid"]
    with pytest.raises(ValueError, match="outside immutable"):
        report(moved, moved / "plots", formats=("png",))
    raw = moved / "states.npy"
    arr = np.load(raw, mmap_mode="r+")
    arr[0, 0] += 1
    arr.flush()
    del arr
    audit = ResultCatalog(moved).validate(full=True)
    assert audit["status"] == "failed" and any(
        "Checksum" in x for x in audit["results"][0]["errors"]
    )


def test_far_scopes_and_nonfinal_denominator():
    meta = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2],
            "label": [1, 1, 0, 0, 1],
            "token_end": [5, 10, 5, 10, 10],
            "n_tokens": [10] * 5,
            "is_final": [False, True, False, True, True],
        }
    )
    scores = np.array([0.1, 0.95, 0.8, 0.9, 0.99])
    all_metrics, _ = monitoring_metrics(meta, scores, 0.7, "all_boundaries")
    early_metrics, _ = monitoring_metrics(meta, scores, 0.7, "nonfinal")
    assert all_metrics["test_far"] == 1 and early_metrics["test_far"] == 0
    assert early_metrics["early_detection_rate"] == 1
    threshold, far = calibrate_threshold(meta, scores, 0.5, "nonfinal")
    assert far == 0.5 and threshold < 0.1  # second success has no nonfinal boundary
    with pytest.raises(ValueError):
        calibrate_threshold(meta, scores, 0.1, "typo")


def test_characterization_reports_inference_unit_and_sparse_cells():
    meta = pd.DataFrame(
        {
            "group_id": [0, 1, 2, 3],
            "label": [0, 0, 1, 1],
            "category": ["a", "a", "b", "b"],
            "level": [1, 1, 2, 2],
            "subject": None,
            "language": "en",
        }
    )
    states = np.array([[0], [0], [1], [1]])
    assoc, _, _ = characterize(states, meta, [1])
    part = assoc[assoc.axis == "label"].iloc[0]
    assert (
        part.cramers_v == 1
        and part.p_value < 0.05
        and part.expected_below_5_fraction == 1
    )
    repeated = pd.concat([meta, meta], ignore_index=True)
    assoc, _, _ = characterize(np.tile(states, (2, 1)), repeated, [1])
    assert assoc[assoc.axis == "label"].p_value.isna().all()

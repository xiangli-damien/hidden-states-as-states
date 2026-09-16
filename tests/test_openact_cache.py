from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import zarr

from hss.experiments.openact import DataSpec, prepare
from fixtures import openact_shards


def test_shards_labels_norm_sides_and_cache_reuse(tmp_path):
    root = openact_shards(tmp_path / "source")
    spec = DataSpec([str(root)], min_free_gib=0, expected_samples=40)
    cache = prepare(spec, tmp_path / "cache")
    assert cache.n_items() == 40
    assert cache.meta.sample_id.nunique() == 40
    assert cache.meta.label.tolist() == [0, 1] * 20
    before = (cache.path / "layer_2.npy").stat().st_mtime_ns
    assert prepare(spec, tmp_path / "cache").path == cache.path
    assert (cache.path / "layer_2.npy").stat().st_mtime_ns == before
    pre = prepare(replace(spec, final_norm="pre"), tmp_path / "cache")
    np.testing.assert_allclose(pre.array(2), 3 * cache.array(2), atol=1e-6)
    np.testing.assert_array_equal(pre.array(0), cache.array(0))
    assert pre.path != cache.path
    assert isinstance(cache.array(0), np.memmap)


def test_prefixes_are_causal_and_groups_preserved(tmp_path):
    root = openact_shards(tmp_path / "source")
    spec = DataSpec(
        [str(root)],
        representation="prefix",
        layers=[0, 2],
        batch_tokens=2,
        token_entropy_path="metrics/token_entropy",
        min_free_gib=0,
    )
    data = prepare(spec, tmp_path / "cache")
    part = data.meta[data.meta.group_id == 0]
    assert part.token_end.tolist() == [2, 5, 6]
    z = zarr.open_group(str(root / "shard_00000_00020/tensors.zarr"), mode="r")
    for row, end in zip(part.index, part.token_end):
        np.testing.assert_allclose(
            data.array(2)[row], z["hidden_states/per_token"][:end, 2].mean(0), atol=1e-6
        )
        np.testing.assert_allclose(
            data.meta.loc[row, "prefix_entropy"],
            z["metrics/token_entropy"][:end].mean(),
        )
    assert data.meta.groupby("group_id").label.nunique().max() == 1


def test_incomplete_shards_ignored_and_full_coverage_required(tmp_path):
    root = openact_shards(tmp_path / "source")
    (root / "shard_00020_00040/_COPY_VERIFIED.json").unlink()
    spec = DataSpec([str(root)], min_free_gib=0, expected_samples=40)
    with pytest.raises(ValueError, match="Expected 40"):
        prepare(spec, tmp_path / "cache")
    data = prepare(replace(spec, expected_samples=None), tmp_path / "cache")
    assert data.n_items() == 20


def test_duplicate_samples_and_mixed_models_rejected(tmp_path):
    root = openact_shards(tmp_path / "source")
    p = root / "shard_00020_00040/data.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "sample_id"] = "math_0"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="Duplicate sample"):
        prepare(DataSpec([str(root)], min_free_gib=0), tmp_path / "cache")
    p = root / "shard_00020_00040/manifest.json"
    m = json.loads(p.read_text())
    m["model"]["revision"] = "other"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="different models"):
        prepare(DataSpec([str(root)], min_free_gib=0), tmp_path / "cache")

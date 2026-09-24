"""A portable NPY + Parquet interchange format for other collectors/datasets."""

from dataclasses import asdict
import glob
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .cache import CachedStates
from ..experiments.artifacts import digest, file_digest, lock, save_json
from ..provenance import stage_version


def export_arrays(
    path,
    matrices,
    rows,
    *,
    model,
    representation="mean",
    final_norm="post",
    dataset_id=None,
):
    """Write an explicit interchange dataset. This does not fit or label any data."""
    path = Path(path)
    if len(model) < 2:
        raise ValueError("Model must provide identifier and pinned revision")
    path.mkdir(parents=True, exist_ok=False)
    layers = {}
    shape = None
    for layer, matrix in matrices.items():
        matrix = np.asarray(matrix)
        if (
            matrix.ndim != 2
            or len(matrix) != len(rows)
            or (shape and matrix.shape != shape)
        ):
            raise ValueError(
                "All layer matrices must have the same (rows, hidden_dim) shape"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite hidden-state matrix")
        shape = matrix.shape
        layers[str(int(layer))] = f"layer_{int(layer)}.npy"
        np.save(
            path / layers[str(int(layer))],
            matrix.astype(np.float32),
            allow_pickle=False,
        )
    if not layers:
        raise ValueError("No layers supplied")
    rows.to_parquet(path / "rows.parquet", index=False)
    save_json(
        path / "dataset.json",
        {
            "format_version": 1,
            "model": list(model),
            "layers": layers,
            "rows": "rows.parquet",
            "representation": representation,
            "final_norm": final_norm,
            "dataset_id": dataset_id,
        },
    )
    return path


def prepare(spec, cache_root):
    started = time.perf_counter()
    manifests = sorted(
        {
            (Path(p) / "dataset.json" if Path(p).is_dir() else Path(p)).resolve()
            for pattern in spec.paths
            for p in glob.glob(str(Path(pattern).expanduser()))
        }
    )
    if spec.max_shards:
        manifests = manifests[: spec.max_shards]
    if not manifests:
        raise FileNotFoundError("No array dataset manifests matched")
    sources, parts, matrices, seen = [], [], [], set()
    signature, layers, width = None, None, None
    dataset_ids = set()
    for run_index, path in enumerate(manifests):
        manifest = json.loads(path.read_text())
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported array dataset format")
        if (
            manifest["representation"] != spec.representation
            or manifest["final_norm"] != spec.final_norm
        ):
            raise ValueError(
                "Array representation/RMS side differs from requested selection"
            )
        current = tuple(manifest["model"])
        if len(current) < 2:
            raise ValueError("Array manifest requires model identifier and revision")
        if manifest.get("dataset_id"):
            dataset_ids.add(manifest["dataset_id"])
        if signature is not None and current != signature:
            raise ValueError("Do not combine different models/revisions")
        signature = current
        if spec.expected_model and current[0] != spec.expected_model:
            raise ValueError("Unexpected array model identifier")
        available = sorted(map(int, manifest["layers"]))
        wanted = (
            available
            if spec.layers is None
            else [available[k] if k < 0 else k for k in spec.layers]
        )
        if len(set(wanted)) != len(wanted) or not set(wanted) <= set(available):
            raise ValueError("Missing or duplicate selected layers")
        if layers is not None and wanted != layers:
            raise ValueError("Inconsistent array layer IDs")
        layers = wanted
        files = [
            path,
            path.parent / manifest["rows"],
            *[path.parent / manifest["layers"][str(k)] for k in layers],
        ]
        sources.append(
            {
                "path": str(path),
                "files": {
                    str(p.relative_to(path.parent)): file_digest(p) for p in files
                },
            }
        )
        meta = pd.read_parquet(path.parent / manifest["rows"])
        if "sample_id" not in meta:
            raise ValueError("Array rows require sample_id")
        if (
            meta.sample_id.isna().any()
            or meta.sample_id.astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Sample IDs must be present and nonempty")
        meta["sample_id"] = meta.sample_id.astype(str)
        if spec.representation in ("prefix", "sentence", "tokens") and not {
            "token_end",
            "n_tokens",
        } <= set(meta):
            raise ValueError(
                "Prefix/token arrays require observed token_end and response n_tokens"
            )
        meta["token_end"] = meta.get("token_end", pd.Series(1, index=meta.index))
        meta["n_tokens"] = meta.get("n_tokens", meta.token_end)
        for field in ("token_end", "n_tokens"):
            values = meta[field].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values != np.floor(values)):
                raise ValueError("Token counts and boundaries must be finite integers")
            meta[field] = values.astype("int64")
        if (meta.token_end < 1).any() or (meta.token_end > meta.n_tokens).any():
            raise ValueError("Invalid token boundaries")
        if meta.duplicated(["sample_id", "token_end"]).any():
            raise ValueError("Duplicate sample/boundary rows")
        if meta.groupby("sample_id").n_tokens.nunique().max() != 1:
            raise ValueError("Inconsistent response token counts")
        if spec.representation in ("prefix", "sentence", "tokens"):
            for _, part in meta.groupby("sample_id", sort=False):
                ends = part.token_end.to_numpy()
                if np.any(np.diff(ends) <= 0) or ends[-1] != part.n_tokens.iloc[0]:
                    raise ValueError(
                        "Boundaries must increase and include the final response boundary"
                    )
                if spec.representation == "tokens" and not np.array_equal(
                    ends, np.arange(1, ends[-1] + 1)
                ):
                    raise ValueError(
                        "Token representations require every observed token"
                    )
        if (
            spec.representation in ("mean", "prompt_last")
            and meta.sample_id.duplicated().any()
        ):
            raise ValueError("Response-level arrays require one row per sample")
        mask = np.ones(len(meta), dtype=bool)
        if spec.only_valid and "status" in meta:
            mask &= meta.status.to_numpy() == 1
        if spec.exclude_truncated and "finish_reason" in meta:
            mask &= meta.finish_reason.to_numpy() != "length"
        ids = meta.loc[mask, "sample_id"].drop_duplicates().tolist()
        if spec.max_samples:
            ids = ids[: max(0, spec.max_samples - len(seen))]
        if seen.intersection(ids):
            raise ValueError("Duplicate sample ID across array datasets")
        seen.update(ids)
        selected = np.flatnonzero(mask & meta.sample_id.isin(ids).to_numpy())
        part = meta.iloc[selected].copy()
        if spec.label in part:
            part["label"] = part[spec.label]
        elif "label" not in part:
            part["label"] = np.nan
        if not part.label.dropna().isin([0, 1, True, False]).all():
            raise ValueError("Labels must be binary or missing")
        part["is_final"] = part.token_end == part.n_tokens
        part["source_run"] = run_index
        part["sample_idx"] = selected
        part["boundary"] = part.groupby("sample_id", sort=False).cumcount()
        for field in ("category", "level", "subject", "language", "finish_reason"):
            if field not in part:
                part[field] = None
        mapping = {}
        for layer in layers:
            X = np.load(
                path.parent / manifest["layers"][str(layer)],
                mmap_mode="r",
                allow_pickle=False,
            )
            if X.ndim != 2 or len(X) != len(meta) or (width and width != X.shape[1]):
                raise ValueError("Inconsistent array dimensions")
            width = X.shape[1]
            mapping[layer] = X
        parts.append(part)
        matrices.append((mapping, selected))
    rows = pd.concat(parts, ignore_index=True)
    if not len(rows) or (
        spec.expected_samples is not None and len(seen) != spec.expected_samples
    ):
        raise ValueError(f"Expected {spec.expected_samples} samples, found {len(seen)}")
    rows["group_id"] = pd.factorize(rows.sample_id, sort=False)[0]
    if rows.groupby("sample_id").label.nunique().max() > 1:
        raise ValueError("Inconsistent response labels across boundaries")
    identity = {
        "format": 1,
        "reader": stage_version("data"),
        "spec": asdict(spec),
        "sources": sources,
    }
    key = digest(identity)
    root = Path(cache_root).expanduser().resolve() / "data" / key
    with lock(root.with_suffix(".lock")):
        if (root / "_SUCCESS.json").exists():
            return CachedStates(root)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        size = len(rows) * len(layers) * width * 4
        if shutil.disk_usage(root).free < size + spec.min_free_gib * 1024**3:
            raise OSError("Insufficient space for array cache and reserve")
        for layer in layers:
            target = np.lib.format.open_memmap(
                root / f"layer_{layer}.npy",
                mode="w+",
                dtype="float32",
                shape=(len(rows), width),
            )
            offset = 0
            for mapping, indices in matrices:
                for start in range(0, len(indices), 1024):
                    values = mapping[layer][indices[start : start + 1024]]
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite array features")
                    target[offset : offset + len(values)] = values
                    offset += len(values)
            target.flush()
        rows.to_parquet(root / "rows.parquet", index=False)
        save_json(
            root / "_SUCCESS.json",
            {
                "identity": identity,
                "key": key,
                "layers": layers,
                "n_rows": len(rows),
                "n_samples": len(seen),
                "hidden_dim": width,
                "bytes": size,
                "build_seconds": time.perf_counter() - started,
                "segmentation": "provided-boundaries",
                "model": [*signature[:2], width, len(layers)],
                "dataset_id": spec.dataset_id
                or (next(iter(dataset_ids)) if len(dataset_ids) == 1 else None),
            },
        )
    return CachedStates(root)

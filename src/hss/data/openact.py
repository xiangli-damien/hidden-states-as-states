"""Read published OpenAct runs once, then serve contiguous per-layer memmaps.

Accepts one Run, a model directory containing verified shards, or explicit run
paths/globs. Unpublished staging/incoming/failed directories are never included.
No model weights, Transformers or OpenAct installation is required for analysis.
"""

from dataclasses import asdict
import glob
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import zarr

from ..experiments.artifacts import digest, file_digest, lock, save_json


from .spec import DataSpec
from .cache import CachedStates
from ..provenance import stage_version


def discover(spec):
    paths = set()
    for pattern in spec.paths:
        matches = glob.glob(str(Path(pattern).expanduser()))
        for match in matches:
            root = Path(match).resolve()
            candidates = (
                [root]
                if (root / "manifest.json").exists()
                else list(root.glob("shard_*"))
            )
            for p in candidates:
                if (
                    ".incoming-" in p.name
                    or ".failed-" in p.name
                    or not (p / "_SUCCESS").is_file()
                ):
                    continue
                if (
                    spec.require_copy_receipt
                    and not (p / "_COPY_VERIFIED.json").is_file()
                ):
                    continue
                if (p / "manifest.json").is_file():
                    paths.add(p)
    result = sorted(paths)
    if spec.max_shards:
        result = result[: spec.max_shards]
    if not result:
        raise FileNotFoundError(
            "No completed OpenAct runs matched; use a model directory, not a multi-model root"
        )
    return result


def sentence_boundaries(text, offsets, n_tokens):
    """Deterministic punctuation/newline boundaries mapped to observed whole tokens.

    The last boundary includes EOS if present. This segmentation rule is recorded
    explicitly because the manuscript does not specify its original tokenizer.
    """
    valid = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & (offsets[:, 1] > 0))
    ends = offsets[valid, 1]
    if len(ends) and np.any(np.diff(ends) < 0):
        raise ValueError(
            "Nonmonotonic token offsets cannot define causal sentence boundaries"
        )
    positions = []
    for m in re.finditer(r"[.!?](?=\s|$)|[。！？]|\n+", text):
        pos = int(np.searchsorted(ends, m.end()))
        if pos < len(valid):
            positions.append(int(valid[pos]) + 1)
    return sorted(set(p for p in positions + [n_tokens] if 0 < p <= n_tokens))


def prepare(spec: DataSpec, cache_root):
    spec.validate()
    started = time.perf_counter()
    runs = discover(spec)
    sources = []
    for p in runs:
        files = ["manifest.json", "data.parquet", "_SUCCESS"]
        files += [
            str(x.relative_to(p))
            for x in [
                p / "_COPY_VERIFIED.json",
                p / "labels" / (spec.label_file + ".parquet"),
            ]
            if x.exists()
        ]
        sources.append(
            {"path": str(p), "files": {f: file_digest(p / f) for f in files}}
        )
    identity = {
        "format": 1,
        "reader": stage_version("data"),
        "spec": asdict(spec),
        "sources": sources,
    }
    key = digest(identity)
    root = Path(cache_root).expanduser().resolve() / "data" / key
    root.parent.mkdir(parents=True, exist_ok=True)
    with lock(root.with_suffix(".lock")):
        if (root / "_SUCCESS.json").exists():
            return CachedStates(root)
        # Interrupted partial builds are private to this cache key; rebuild them.
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        rows, records, groups = [], [], set()
        signature, layers, positions, hidden_dim, last_layer = (
            None,
            None,
            None,
            None,
            None,
        )
        sample_count = 0
        for run_idx, p in enumerate(runs):
            manifest = json.loads((p / "manifest.json").read_text())
            model = manifest["model"]
            if spec.expected_model and model.get("identifier") != spec.expected_model:
                raise ValueError(
                    f"Expected model {spec.expected_model}, found {model.get('identifier')}"
                )
            current_signature = (
                model.get("identifier", model.get("name")),
                model.get("revision"),
                model["hidden_dim"],
                model["n_layers"],
            )
            if signature is not None and signature != current_signature:
                raise ValueError(
                    "Do not combine different models/revisions in one HSS state space"
                )
            signature = current_signature
            z = zarr.open_group(str(p / "tensors.zarr"), mode="r")
            # Derive selected layer IDs from capture metadata, not array positions.
            available = manifest.get("capture", {}).get("hidden_states_layers")
            available = (
                list(range(model["n_layers"])) if available is None else list(available)
            )
            wanted = (
                available
                if spec.layers is None
                else [available[x] if x < 0 else x for x in spec.layers]
            )
            if len(set(wanted)) != len(wanted) or any(
                x not in available for x in wanted
            ):
                raise ValueError("Requested layer IDs are missing or duplicated")
            if layers is not None and layers != wanted:
                raise ValueError("Inconsistent layer IDs across runs")
            layers, positions = wanted, [available.index(x) for x in wanted]
            hidden_dim = model["hidden_dim"]
            last_layer = model.get("n_decoder_layers", model["n_layers"] - 1)
            df = pd.read_parquet(p / "data.parquet")
            ptr = np.asarray(z["tokens/sample_ptr"][:], dtype=np.int64)
            if len(ptr) != len(df) + 1 or np.any(np.diff(ptr) < 0) or ptr[0] != 0:
                raise ValueError("Invalid OpenAct sample/token pointers")
            label_path = p / "labels" / (spec.label_file + ".parquet")
            labels = (
                pd.read_parquet(label_path) if label_path.exists() else pd.DataFrame()
            )
            if len(labels):
                if labels.sample_idx.duplicated().any():
                    raise ValueError("Duplicate label sample_idx")
                labels = labels.set_index("sample_idx")
            run_rows = []
            for local, row in df.iterrows():
                if spec.only_valid and row["status"] != 1:
                    continue
                if spec.exclude_truncated and row.get("finish_reason") == "length":
                    continue
                if spec.max_samples is not None and sample_count >= spec.max_samples:
                    break
                sid = str(row["sample_id"])
                if sid in groups:
                    raise ValueError(f"Duplicate sample ID across shards: {sid}")
                groups.add(sid)
                n_tokens = int(ptr[local + 1] - ptr[local])
                if n_tokens < 1:
                    raise ValueError(f"No response tokens for valid sample {sid}")
                label = None
                sample_idx = int(row["sample_idx"])
                if len(labels) and spec.label in labels and sample_idx in labels.index:
                    value = labels.at[sample_idx, spec.label]
                    if pd.notna(value):
                        if value not in (True, False, 0, 1):
                            raise ValueError(
                                f"Label {spec.label} must be binary, got {value!r}"
                            )
                        label = int(value)
                ends = [n_tokens]
                if spec.representation in ("prefix", "sentence"):
                    offsets = z["tokens/offsets"][ptr[local] : ptr[local + 1]]
                    ends = sentence_boundaries(
                        str(row["response_text"]), offsets, n_tokens
                    )
                elif spec.representation == "tokens":
                    ends = list(range(1, n_tokens + 1))
                first = len(rows)
                uncertainty = {}
                for name, metric_path in [
                    ("prefix_entropy", spec.token_entropy_path),
                    ("prefix_logprob", spec.token_logprob_path),
                ]:
                    if metric_path is not None:
                        if metric_path not in z or z[metric_path].shape != (
                            int(ptr[-1]),
                        ):
                            raise ValueError(
                                f"Missing aligned per-token metric: {metric_path}"
                            )
                        metric = np.asarray(
                            z[metric_path][ptr[local] : ptr[local + 1]], dtype=float
                        )
                        if not np.isfinite(metric).all():
                            raise ValueError(f"Nonfinite token metric {metric_path}")
                        uncertainty[name] = np.cumsum(metric) / np.arange(
                            1, n_tokens + 1
                        )
                for boundary, end in enumerate(ends):
                    rows.append(
                        dict(
                            sample_id=sid,
                            group_id=sample_count,
                            source_run=run_idx,
                            sample_idx=sample_idx,
                            label=label,
                            token_end=end,
                            n_tokens=n_tokens,
                            boundary=boundary,
                            is_final=end == n_tokens,
                            category=row.get("category"),
                            level=row.get("level"),
                            subject=row.get("subject"),
                            language=row.get("language"),
                            finish_reason=row.get("finish_reason"),
                            **{
                                name: values[end - 1]
                                for name, values in uncertainty.items()
                            },
                        )
                    )
                run_rows.append((local, first, ends, int(ptr[local]), n_tokens))
                sample_count += 1
            records.append((p, run_rows, positions.copy()))
        if not rows or (
            spec.expected_samples is not None and sample_count != spec.expected_samples
        ):
            raise ValueError(
                f"Expected {spec.expected_samples} samples, found {sample_count}; full data may still be collecting"
            )
        required = len(rows) * len(layers) * hidden_dim * 4
        if shutil.disk_usage(root).free < required + spec.min_free_gib * 1024**3:
            raise OSError(
                f"Insufficient local cache space: need {required / 1024**3:.2f} GiB plus reserve"
            )
        arrays = {
            layer: np.lib.format.open_memmap(
                root / f"layer_{layer}.npy",
                mode="w+",
                dtype="float32",
                shape=(len(rows), hidden_dim),
            )
            for layer in layers
        }
        for p, run_rows, pos in records:
            z = zarr.open_group(str(p / "tensors.zarr"), mode="r")
            for local, first, ends, offset, n_tokens in run_rows:
                if spec.representation in ("mean", "prompt_last"):
                    values = np.asarray(
                        z[f"hidden_states/{spec.representation}"][local]
                    )[pos]
                    if spec.final_norm == "pre" and last_layer in layers:
                        values[layers.index(last_layer)] = z[
                            f"final_norm/pre/{spec.representation}"
                        ][local]
                    for j, layer in enumerate(layers):
                        arrays[layer][first] = values[j]
                else:
                    sums = np.zeros((len(layers), hidden_dim), dtype=np.float64)
                    previous_boundary_sum = sums.copy()
                    previous_boundary_end = 0
                    next_end = 0
                    for start in range(0, n_tokens, spec.batch_tokens):
                        stop = min(start + spec.batch_tokens, n_tokens)
                        values = np.asarray(
                            z["hidden_states/per_token"].oindex[
                                offset + start : offset + stop, pos, :
                            ]
                        )
                        if spec.final_norm == "pre" and last_layer in layers:
                            values[:, layers.index(last_layer)] = z[
                                "final_norm/pre/per_token"
                            ][offset + start : offset + stop]
                        cumulative = np.cumsum(values, axis=0, dtype=np.float64) + sums
                        while next_end < len(ends) and ends[next_end] <= stop:
                            end = ends[next_end]
                            value = (
                                values[end - start - 1]
                                if spec.representation == "tokens"
                                else cumulative[end - start - 1] / end
                            )
                            if spec.representation == "sentence":
                                current = cumulative[end - start - 1]
                                value = (current - previous_boundary_sum) / (
                                    end - previous_boundary_end
                                )
                                previous_boundary_sum = current.copy()
                                previous_boundary_end = end
                            for j, layer in enumerate(layers):
                                arrays[layer][first + next_end] = value[j]
                            next_end += 1
                        sums = cumulative[-1]
        for arr in arrays.values():
            for start in range(0, len(arr), 1024):
                if not np.isfinite(arr[start : start + 1024]).all():
                    raise ValueError("Nonfinite activation data in cache")
            arr.flush()
        pd.DataFrame(rows).to_parquet(root / "rows.parquet", index=False)
        save_json(
            root / "_SUCCESS.json",
            dict(
                identity=identity,
                key=key,
                layers=layers,
                n_rows=len(rows),
                n_samples=sample_count,
                hidden_dim=hidden_dim,
                bytes=required,
                build_seconds=time.perf_counter() - started,
                segmentation="punctuation/newline-v1",
                model=signature,
                dataset_id=spec.dataset_id,
            ),
        )
        return CachedStates(root)

"""Named datasets decouple machines/storage paths from experimental protocols."""

import json
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .spec import DataSpec


def load_catalog(path):
    path = Path(path).expanduser().resolve()
    raw = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else tomllib.loads(path.read_text())
    )
    if set(raw) - {"schema_version", "datasets"} or raw.get("schema_version") != 1:
        raise ValueError("Dataset catalog needs schema_version=1 and datasets")
    entries = {}
    for name, entry in raw["datasets"].items():
        entry = dict(entry)
        entry.setdefault("dataset_id", name.split("/")[-1])
        paths = []
        for value in entry["paths"]:
            expanded = Path(os.path.expandvars(os.path.expanduser(value)))
            paths.append(
                str(expanded if expanded.is_absolute() else path.parent / expanded)
            )
        entry["paths"] = paths
        spec = DataSpec(**entry)
        spec.validate()
        entries[name] = entry
    return entries


def inspect_spec(spec):
    import glob
    import pyarrow.parquet as pq
    from .openact import discover

    try:
        if spec.source_format == "openact":
            paths = discover(spec)
            count = sum(
                pq.ParquetFile(p / "data.parquet").metadata.num_rows for p in paths
            )
        else:
            paths = [Path(p) for pattern in spec.paths for p in glob.glob(pattern)]
            if not paths:
                raise FileNotFoundError("No array manifests matched")
            count = 0
            for p in paths:
                p = p / "dataset.json" if p.is_dir() else p
                manifest = json.loads(p.read_text())
                count += (
                    pq.read_table(p.parent / manifest["rows"], columns=["sample_id"])
                    .to_pandas()
                    .sample_id.nunique()
                )
        return {
            "status": "available"
            if spec.expected_samples is None or count == spec.expected_samples
            else "incomplete",
            "published_samples": int(count),
            "expected_samples": spec.expected_samples,
            "sources": len(paths),
        }
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "missing_or_invalid", "reason": str(exc)}


def inspect_catalog(path):
    statuses = {
        name: inspect_spec(DataSpec(**entry))
        for name, entry in load_catalog(path).items()
    }
    return {
        "status": "complete"
        if statuses and all(s["status"] == "available" for s in statuses.values())
        else "incomplete",
        "datasets": statuses,
        "scope": "Published metadata counts only; prepare performs model, labels, tensor and sample checks.",
    }

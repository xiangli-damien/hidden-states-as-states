"""Public result reader, structural audit and immutable artifact inventory.

Reading or plotting a result never prepares raw activations or fits a model.
Legacy result directories remain readable; absent checksums are reported explicitly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.artifacts import save_json, file_digest


def seal_result(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Result artifacts cannot be symlinks")
        if (
            path.is_file()
            and path.name not in (".lock", "_SUCCESS.json", "artifacts.json")
            and ".tmp-" not in path.name
        ):
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
    save_json(root / "artifacts.json", {"schema_version": 1, "files": files})


class Result:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not (self.path / "_SUCCESS.json").exists():
            raise ValueError(f"Incomplete result: {self.path}")
        self.summary = self.json("summary.json")
        self.config = self.json("config.json")
        self.snapshot = self.json("data_snapshot.json")
        self._rows = None
        self._states = None

    def json(self, name):
        return json.loads((self.path / name).read_text())

    @property
    def rows(self):
        if self._rows is None:
            self._rows = pd.read_parquet(self.path / "rows.parquet")
        return self._rows

    @property
    def states(self):
        if self._states is None:
            self._states = np.load(
                self.path / "states.npy", mmap_mode="r", allow_pickle=False
            )
        return self._states

    @property
    def layers(self):
        return [row["layer"] for row in self.summary["profile"]]

    @property
    def dataset(self):
        explicit = self.config["data"].get("dataset_id") or self.snapshot.get(
            "dataset_id"
        )
        if explicit:
            return explicit
        paths = " ".join(self.config["data"]["paths"])
        for name in (
            "belebele_pooled",
            "belebele_en",
            "belebele_de",
            "belebele_zh",
            "math",
            "mmlu",
            "theoremqa",
            "jailbreakbench",
            "harmbench",
            "wildjailbreak",
        ):
            if name + "_full" in paths:
                return name
        return "unspecified"

    def table(self, name):
        path = self.path / name
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def metadata(self):
        c, s = self.config, self.summary
        return {
            "trial_id": s["trial_id"],
            "path": str(self.path),
            "name": s["name"],
            "model": s["model"][0],
            "model_revision": s["model"][1],
            "dataset": self.dataset,
            "mode": c["evaluation"]["mode"],
            "method": c["cluster"]["method"],
            "seed": c["seed"],
            "representation": c["data"]["representation"],
            "final_norm": c["data"]["final_norm"],
            "n_samples": s["n_samples"],
            "n_rows": s["n_rows"],
            "snapshot": s["snapshot"],
            "fit_fraction": c["evaluation"]["fit_fraction"],
            "k_max": c["cluster"]["k_max"],
            "standardize": c["transform"]["standardize"],
            "pca_components": c["transform"]["pca_components"],
            "alignment_threshold": c["alignment"]["threshold"],
            "seconds": s["seconds"],
        }

    def validate(self, full=False):
        errors, warnings = [], []
        required = [
            "summary.json",
            "config.json",
            "data_snapshot.json",
            "rows.parquet",
            "states.npy",
            "alignment.json",
            "selection.json",
            "split.npz",
            "diagnostics.json",
        ]
        errors += [
            f"Missing {name}" for name in required if not (self.path / name).is_file()
        ]
        inventory = self.path / "artifacts.json"
        if inventory.exists():
            for name, expected in self.json("artifacts.json")["files"].items():
                path = self.path / name
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != expected["bytes"]
                ):
                    errors.append(f"Missing, linked or resized artifact: {name}")
                elif full and file_digest(path) != expected["sha256"]:
                    errors.append(f"Checksum mismatch: {name}")
        else:
            warnings.append("Legacy result has no artifact checksum inventory")
        if errors:
            return {"valid": False, "errors": errors, "warnings": warnings}
        states, rows = self.states, self.rows
        if (
            states.shape != (len(rows), len(self.layers))
            or len(rows) != self.summary["n_rows"]
        ):
            errors.append("State/metadata/layer dimensions differ")
        if states.dtype.kind not in "iu" or np.any(states < 0):
            errors.append("Invalid discrete state IDs")
        if rows.duplicated(["sample_id", "token_end"]).any():
            errors.append("Duplicate sample/boundary rows")
        if rows.sample_id.nunique() != self.summary["n_samples"]:
            errors.append("Sample coverage differs from summary")
        if self.json("_SUCCESS.json")["trial_id"] != self.summary["trial_id"]:
            errors.append("Completion identity differs")
        alignment = self.json("alignment.json")["local_to_global"]
        if len(alignment) != len(self.layers):
            errors.append("Missing layer vocabularies")
        elif states.shape[1] == len(alignment):
            for j, vocab in enumerate(alignment):
                if not np.isin(states[:, j], vocab).all():
                    errors.append(f"Unknown state at layer {self.layers[j]}")
        with np.load(self.path / "split.npz", allow_pickle=False) as split:
            for name in ("train", "validation", "test", "map_fit"):
                indices = split[name]
                if (
                    indices.dtype.kind not in "iu"
                    or np.any(indices < 0)
                    or np.any(indices >= len(rows))
                ):
                    errors.append(f"Invalid {name} split indices")
            if not errors and self.config["evaluation"]["mode"] in (
                "prediction",
                "monitoring",
            ):
                groups = {
                    k: set(rows.iloc[split[k]].group_id)
                    for k in ("train", "validation", "test", "map_fit")
                }
                if (
                    groups["train"] & (groups["validation"] | groups["test"])
                    or groups["validation"] & groups["test"]
                ):
                    errors.append("Response leakage between splits")
                if not groups["map_fit"] <= groups["train"]:
                    errors.append("Map fit contains held-out responses")
        for layer in self.layers:
            for name in ("model.json", "model.npz", "projection.npz", "selection.json"):
                if not (self.path / "models" / f"layer_{layer}" / name).is_file():
                    errors.append(f"Missing layer {layer} {name}")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "full_checksums": full and inventory.exists(),
        }


class ResultCatalog:
    def __init__(self, roots):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        paths = set()
        for root in roots:
            root = Path(root).expanduser().resolve()
            candidates = (
                [root]
                if (root / "_SUCCESS.json").exists()
                else [p.parent for p in root.rglob("_SUCCESS.json")]
            )
            paths.update(
                p
                for p in candidates
                if (p / "summary.json").is_file() and (p / "config.json").is_file()
            )
        self.results = [Result(path) for path in sorted(paths)]

    def select(self, **criteria):
        return [
            r
            for r in self.results
            if all(r.metadata().get(k) == v for k, v in criteria.items())
        ]

    def frame(self):
        return pd.DataFrame([r.metadata() for r in self.results])

    def validate(self, full=False):
        checks = [
            {"path": str(r.path), "trial_id": r.summary["trial_id"], **r.validate(full)}
            for r in self.results
        ]
        return {
            "status": "complete"
            if checks and all(c["valid"] for c in checks)
            else "failed",
            "results": checks,
        }

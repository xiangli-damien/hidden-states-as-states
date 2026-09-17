"""Versioned render artifacts, separate from immutable fitted trial directories."""

from pathlib import Path
import matplotlib.pyplot as plt

from ..experiments.artifacts import save_json, file_digest
from ..provenance import stage_version


class FigureBundle:
    def __init__(self, path, results=(), parameters=None, formats=("png", "svg")):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        if not formats or set(formats) - {"png", "svg"}:
            raise ValueError("Figure formats: png, svg")
        self.formats = formats
        self.inputs = [
            {
                "trial_id": r.summary["trial_id"],
                "path": str(r.path),
                "summary_sha256": file_digest(r.path / "summary.json"),
                "snapshot": r.summary["snapshot"],
                "artifact_inventory_sha256": file_digest(r.path / "artifacts.json")
                if (r.path / "artifacts.json").exists()
                else None,
                "scope": "subset"
                if r.config["data"].get("max_samples")
                or r.config["data"].get("max_shards")
                or r.config["data"].get("expected_samples") is None
                else "declared_full_dataset",
                "n_samples": r.summary["n_samples"],
                "model": r.summary["model"][0],
            }
            for r in results
        ]
        self.parameters = parameters or {}
        self.outputs = []

    def table(self, name, frame):
        path = self.path / (name + ".csv")
        frame.to_csv(path, index=False)
        self.outputs.append(path)

    def figure(self, name, fig):
        if any(x["scope"] == "subset" for x in self.inputs):
            fig.suptitle(
                "SUBSET / PROTOCOL VALIDATION — not a full paper reproduction",
                fontsize=10,
                color="#97472b",
            )
        for extension in self.formats:
            path = self.path / (name + "." + extension)
            fig.savefig(path, bbox_inches="tight")
            self.outputs.append(path)
        plt.close(fig)

    def finish(self, **extra):
        payload = {
            "schema_version": 1,
            "figure_source_version": stage_version("figure"),
            "inputs": self.inputs,
            "parameters": self.parameters,
            "formats": list(self.formats),
            "outputs": {
                str(p.relative_to(self.path)): {
                    "bytes": p.stat().st_size,
                    "sha256": file_digest(p),
                }
                for p in self.outputs
            },
            **extra,
        }
        save_json(self.path / "manifest.json", payload)
        return payload

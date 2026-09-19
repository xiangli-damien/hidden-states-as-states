"""End-to-end small synthetic protocol validation, explicitly NOT paper results.

Exercises every figure recipe and every control renderer. Real OpenAct validation
is separate (benchmark_openact.py); this synthetic suite never needs model weights.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from hss.data.arrays import export_arrays
from hss.experiments.artifacts import save_json
from hss.experiments.paper import generate_suite, run_suite
from hss.experiments.config import load
from hss.experiments.sweep import expand
from hss.results import ResultCatalog
from hss.viz.paper import render_paper
from hss.viz.controls import render_controls


def run(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    generated = generate_suite(
        root / "source", root / "study", root / "cache", root / "trials"
    )
    suite = Path(generated["suite"])
    manifest = json.loads(suite.read_text())
    generated_sources = {}
    for job in manifest["jobs"]:
        path = suite.parent / job["config"]
        cfg = load(path)
        payload = cfg.to_dict()
        # Preserve each protocol and model/dataset role while bounding dimensions,
        # seeds, PCA rank, K range and repeated trials for an engineering smoke.
        variants = []
        for variant in expand(cfg):
            row = {}
            for key in (
                "seed",
                "transform.standardize",
                "transform.pca_components",
                "alignment.threshold",
                "alignment.similarity",
                "cluster.method",
                "cluster.rank",
                "data.representation",
                "data.final_norm",
                "evaluation.fit_fraction",
                "evaluation.far_scope",
            ):
                parts = key.split(".")
                a = variant
                for part in parts:
                    a = getattr(a, part)
                b = cfg
                for part in parts:
                    b = getattr(b, part)
                if a != b:
                    row[key] = a
            if "seed" in row:
                row["seed"] = 43
            if "cluster.rank" in row:
                row["cluster.rank"] = min(2, row["cluster.rank"])
            if (
                "transform.pca_components" in row
                and row["transform.pca_components"] is not None
            ):
                row["transform.pca_components"] = min(
                    3, row["transform.pca_components"]
                )
            if "evaluation.fit_fraction" in row:
                row["evaluation.fit_fraction"] = 0.7
            if row not in variants:
                variants.append(row)
        # Keep seed/fraction controls with at least two independent refits.
        if job["name"] == "reliability_centers":
            variants = [{"seed": 43}, {"seed": 44}, {"evaluation.fit_fraction": 0.7}]
        payload["grid"] = {"variants": variants} if variants else {}
        payload["name"] = "SMOKE_" + job["name"]
        payload["cluster"].update(
            k_values=[2, 3], rank=1, n_init=1, max_iter=5, batch_size=16
        )
        payload["evaluation"].update(
            max_iter=100,
            methods=["HSS-NB", "LDA", "GaussianNB", "Logistic", "LinearSVM", "MLP"],
        )
        payload["execution"].update(
            workers=1, threads_per_worker=1, memory_gib=4, min_available_gib=0
        )
        payload["data"].update(
            source_format="arrays",
            expected_samples=None,
            max_samples=40,
            min_free_gib=0,
            layers=[0, 1, 2],
            expected_model=cfg.data.expected_model,
        )

        # Each representation/norm variant points at its explicit source contract.
        def source_for(rep, norm):
            key = (cfg.data.expected_model, cfg.data.dataset_id, rep, norm)
            if key not in generated_sources:
                dest = root / "source" / str(len(generated_sources))
                rng = np.random.default_rng(321)
                labels = np.arange(40) % 2
                ends = (
                    [2, 4, 6]
                    if rep in ("prefix", "sentence")
                    else list(range(1, 7))
                    if rep == "tokens"
                    else [6]
                )
                rows = []
                matrices = {j: [] for j in (0, 1, 2)}
                for i, y in enumerate(labels):
                    values = {j: rng.normal(size=(6, 6)) + y * 1.5 for j in matrices}
                    previous = 0
                    for end in ends:
                        rows.append(
                            {
                                "sample_id": f"synthetic_{i}",
                                "label": int(y),
                                "category": "algebra" if y else "geometry",
                                "level": int(i % 5 + 1),
                                "token_end": end,
                                "n_tokens": 6,
                                "prefix_entropy": float(
                                    0.2 + (1 - y) * 0.3 + end * 0.01
                                ),
                                "prefix_logprob": float(
                                    -0.4 - (1 - y) * 0.4 - end * 0.01
                                ),
                            }
                        )
                        for j in matrices:
                            v = (
                                values[j][end - 1]
                                if rep == "tokens"
                                else values[j][0]
                                if rep == "prompt_last"
                                else values[j][previous:end].mean(0)
                                if rep == "sentence"
                                else values[j][:end].mean(0)
                            )
                            matrices[j].append(
                                v * (3 if norm == "pre" and j == 2 else 1)
                            )
                        previous = end
                export_arrays(
                    dest,
                    {j: np.asarray(x) for j, x in matrices.items()},
                    pd.DataFrame(rows),
                    model=[key[0], "SYNTHETIC_NOT_MODEL_ACTIVATIONS"],
                    representation=rep,
                    final_norm=norm,
                    dataset_id=key[1],
                )
                generated_sources[key] = str(dest)
            return generated_sources[key]

        payload["data"]["paths"] = [
            source_for(payload["data"]["representation"], payload["data"]["final_norm"])
        ]
        for variant in payload.get("grid", {}).get("variants", []):
            rep = variant.get("data.representation", payload["data"]["representation"])
            norm = variant.get("data.final_norm", payload["data"]["final_norm"])
            variant["data.paths"] = [source_for(rep, norm)]
        save_json(path, payload)
    manifest["protocol_notes"].insert(
        0,
        "SYNTHETIC ENGINEERING SMOKE: N=40, D=6, L=3, K=2/3, 5 EM iterations; never paper measurements.",
    )
    save_json(suite, manifest)
    experiment = run_suite(suite)
    if experiment["status"] != "complete":
        save_json(root / "smoke.json", experiment)
        return experiment
    audit = ResultCatalog(root / "trials").validate(full=True)
    figures = render_paper(
        root / "trials", suite, root / "figures", max_trajectories=40
    )
    controls = render_controls(root / "trials", suite, root / "controls", bootstrap=20)
    result = {
        "status": "complete"
        if all(x["status"] == "complete" for x in (audit, figures, controls))
        else "failed",
        "scope": "synthetic protocol validation, not numerical paper reproduction",
        "seconds": time.perf_counter() - start,
        "trials": len(audit["results"]),
        "jobs": len(manifest["jobs"]),
        "figures": {x["recipe"]: x["status"] for x in figures["coverage"]},
        "controls": {x["control"]: x["status"] for x in controls["coverage"]},
        "audit": audit["status"],
        "path": str(root),
    }
    save_json(root / "smoke.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    result = run(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "complete")

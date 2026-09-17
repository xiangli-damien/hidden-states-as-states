"""Paper figure/table recipes and honest coverage accounting.

A suite_status.json resolves job aliases to immutable trial IDs, including aliases
that deduplicate to the same trial. No raw data or fitting function is used here.
"""

from pathlib import Path
import json
import html

import numpy as np
import pandas as pd

from ..results import ResultCatalog
from ..analysis.tables import (
    state_map,
    dynamics,
    trajectory_similarity,
    selection_surface,
    collect_tables,
    compare_trials,
)
from ..experiments.artifacts import save_json, file_digest
from .artifacts import FigureBundle
from . import plots

CROSS = ["default_map", "qwen_math_map", "geometry_llama3_math"] + [
    f"geometry_{m}_{d}"
    for m in ("llama32", "qwen2", "llama3")
    for d in ("mmlu", "belebele_pooled")
]
PREDICTION = [
    f"prediction_{m}_{d}"
    for m in ("qwen2", "llama3")
    for d in ("math", "mmlu", "theoremqa")
] + [f"prediction_llama2_{d}" for d in ("jailbreakbench", "harmbench")]
RECIPES = {
    "figure_01": ("Conceptual state abstraction", []),
    "figure_02": ("Reproduction pipeline", []),
    "figure_03": ("Default entropy-colored state graph", ["default_map"]),
    "figure_04": (
        "State occupancy, trajectory similarity, layer dynamics",
        ["default_map"],
    ),
    "figure_05": (
        "Reliability and cross-model/dataset geometry",
        CROSS + ["reliability_trends", "reliability_centers"],
    ),
    "figure_06": ("State-label associations (Cramér's V)", ["qwen_math_map"]),
    "figure_07": ("MiniBatch KMeans unlabeled graph", ["kmeans_control"]),
    "figure_08": ("MiniBatch KMeans correctness graph", ["kmeans_control"]),
    "figure_09": ("Relative ICL surface", ["qwen_math_map"]),
    "figure_10": ("Prompt-last correctness graph", ["qwen_prompt_map"]),
    "figure_11": ("Generation-mean correctness graph", ["qwen_math_map"]),
    "figure_12": ("Global-state correctness bands", ["qwen_math_map"]),
    "table_01": ("Before-generation prediction", PREDICTION),
    "table_02": ("Sentence-prefix monitoring", ["sentence_monitoring"]),
    "table_03": ("Recorded protocol and data inventory", []),
}


def suite_aliases(suite, catalog):
    path = Path(suite).expanduser().resolve()
    status_path = path.parent / "suite_status.json"
    manifest = json.loads(path.read_text())
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    by_id = {r.summary["trial_id"]: r for r in catalog.results}
    aliases = {}
    issues = {}
    for job in manifest["jobs"]:
        name = job["name"]
        entry = status.get(name, {})
        aliases[name] = [
            by_id[item["trial_id"]]
            for item in entry.get("results", [])
            if item.get("trial_id") in by_id
        ]
        if entry.get("status") != "complete":
            issues[name] = entry.get(
                "error", f"Job status: {entry.get('status', 'not run')}"
            )
        elif len(aliases[name]) != len(entry.get("results", [])):
            issues[name] = (
                "Some completed trial artifacts are outside the supplied result roots"
            )
        elif entry.get("source_config_sha256") and entry[
            "source_config_sha256"
        ] != file_digest(path.parent / job["config"]):
            issues[name] = (
                "Study configuration changed since this job ran; rerun this job or use its original study"
            )
    return aliases, issues, manifest


def _render(name, bundle, inputs, aliases, params):
    if name in ("figure_01", "figure_02"):
        bundle.figure(name, plots.schematic(int(name[-2:])))
        return
    if name == "table_03":
        bundle.table("protocol_inventory", pd.DataFrame([r.metadata() for r in inputs]))
        configs = [
            {
                "trial_id": r.summary["trial_id"],
                "configuration": json.dumps(r.config, sort_keys=True),
            }
            for r in inputs
        ]
        bundle.table("resolved_parameters", pd.DataFrame(configs))
        return
    if name in ("table_01", "table_02"):
        frame = collect_tables(inputs)["evaluation"]
        bundle.table("metrics", frame)
        if len(frame):
            index = ["model", "dataset", "representation", "seed", "trial_id"]
            values = [
                c
                for c in (
                    "auroc",
                    "accuracy",
                    "auroc_half",
                    "auroc_final",
                    "test_far",
                    "early_detection_rate",
                    "saved_token_fraction_failed",
                )
                if c in frame
            ]
            wide = frame.set_index(index + ["predictor"])[values].unstack("predictor")
            wide.columns = [" / ".join(c) for c in wide.columns]
            bundle.table("comparison", wide.reset_index())
        return
    r = inputs[0]
    if name in ("figure_03", "figure_07", "figure_08", "figure_10", "figure_11"):
        nodes, edges = state_map(r)
        bundle.table("nodes", nodes)
        bundle.table("edges", edges)
        color = "entropy" if name in ("figure_03", "figure_07") else "accuracy_delta"
        if color == "accuracy_delta" and nodes.accuracy_delta.isna().all():
            raise ValueError("Correctness labels unavailable")
        bundle.figure(name, plots.state_graph(nodes, edges, color, RECIPES[name][0]))
    elif name == "figure_04":
        marginal, profile = dynamics(r)
        similarity, rows = trajectory_similarity(
            r, params["max_trajectories"], params["seed"], params["linkage"]
        )
        bundle.table("state_frequency", marginal)
        bundle.table("layer_dynamics", profile)
        bundle.table("trajectory_order", rows)
        matrix_path = bundle.path / "trajectory_similarity.npy"
        np.save(matrix_path, similarity, allow_pickle=False)
        bundle.outputs.append(matrix_path)
        bundle.figure(name, plots.geometry(marginal, profile, similarity))
    elif name == "figure_05":
        trend = collect_tables(aliases["reliability_trends"])["profiles"]
        refits = aliases["reliability_centers"]
        comparisons = compare_trials(refits, all_pairs=True)
        if len(comparisons) and not comparisons.equal_k.all():
            raise ValueError("Center reliability requires equal K at each layer")
        diagnostics = collect_tables(aliases["qwen_math_map"])["diagnostics"]
        cross = collect_tables([x for key in CROSS for x in aliases[key]])["profiles"]
        for key, table in [
            ("trends", trend),
            ("center_comparisons", comparisons),
            ("diagnostics", diagnostics),
            ("cross_model_profiles", cross),
        ]:
            bundle.table(key, table)
        if len(refits) < 2:
            raise ValueError("At least two independent fixed-K refits required")
        bundle.figure(name, plots.reliability(trend, comparisons, diagnostics, cross))
    elif name == "figure_06":
        frame = r.table("associations.csv")
        if frame.empty or frame.cramers_v.isna().all():
            raise ValueError("Association labels unavailable")
        bundle.table("associations", frame)
        bundle.figure(name, plots.associations(frame))
    elif name == "figure_09":
        frame = selection_surface(r)
        bundle.table("icl_surface", frame)
        bundle.figure(name, plots.icl_surface(frame))
    elif name == "figure_12":
        frame = r.table("state_tags.csv")
        if frame.empty or frame.accuracy_delta.isna().all():
            raise ValueError("Correctness labels unavailable")
        bundle.table("state_tags", frame)
        bundle.figure(name, plots.correctness_bands(frame))


def render_paper(
    roots,
    suite,
    destination,
    only=None,
    formats=("png", "svg"),
    max_trajectories=1000,
    seed=42,
    linkage="average",
    check=False,
    strict=False,
):
    catalog = ResultCatalog(roots)
    aliases, issues, manifest = suite_aliases(suite, catalog)
    selected = list(only or RECIPES)
    if set(selected) - set(RECIPES):
        raise ValueError(f"Unknown recipes: {set(selected) - set(RECIPES)}")
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target == r.path or r.path in target.parents for r in catalog.results):
        raise ValueError(
            "Render destination must be outside immutable fitted trial directories"
        )
    parameters = {
        "max_trajectories": max_trajectories,
        "seed": seed,
        "linkage": linkage,
        "trajectory_distance": "hamming",
        "icl_relative": "(criterion-minimum)/max(abs(minimum),1)",
        "near_optimal": "configured parsimony_tolerance",
        "reliability_pairs": "all fixed-K refit pairs",
        "error_bands": "K profiles: 2 sample SD; centers: 1 sample SD; not confidence intervals",
    }
    coverage = []
    audit = {r.summary["trial_id"]: r.validate() for r in catalog.results}
    for name in selected:
        title, dependencies = RECIPES[name]
        missing = [
            {"job": j, "reason": issues.get(j, "No completed matching results")}
            for j in dependencies
            if not aliases.get(j) or j in issues
        ]
        inputs = list(
            {
                r.summary["trial_id"]: r
                for j in dependencies
                for r in aliases.get(j, [])
            }.values()
        )
        if name == "table_03":
            inputs = catalog.results
        record = {
            "recipe": name,
            "title": title,
            "required_jobs": dependencies,
            "missing": missing,
            "trial_ids": [r.summary["trial_id"] for r in inputs],
        }
        if missing:
            record["status"] = "missing_inputs"
        elif any(not audit[r.summary["trial_id"]]["valid"] for r in inputs):
            record.update(
                status="invalid_inputs",
                audit=[
                    audit[r.summary["trial_id"]]
                    for r in inputs
                    if not audit[r.summary["trial_id"]]["valid"]
                ],
            )
        elif check:
            record["status"] = "ready"
        else:
            bundle = FigureBundle(target / name, inputs, parameters, formats)
            try:
                _render(name, bundle, inputs, aliases, parameters)
                unavailable = [
                    {"trial_id": r.summary["trial_id"], **item}
                    for r in inputs
                    for item in r.summary["unavailable_baselines"]
                ]
                record.update(
                    status="partial"
                    if unavailable and name == "table_02"
                    else "rendered",
                    unavailable_baselines=unavailable if name == "table_02" else [],
                )
                bundle.finish(
                    recipe=name,
                    title=title,
                    status=record["status"],
                    protocol_notes=manifest.get("protocol_notes", []),
                )
                record["manifest"] = str(bundle.path / "manifest.json")
            except (ValueError, KeyError, TypeError) as exc:
                record.update(status="failed", error=str(exc))
                bundle.finish(status="failed", error=str(exc))
        coverage.append(record)
    incomplete = any(
        r["status"] in ("missing_inputs", "invalid_inputs", "partial", "failed")
        for r in coverage
    )
    result = {
        "status": "failed"
        if any(r["status"] in ("failed", "invalid_inputs") for r in coverage)
        or (strict and incomplete)
        else "incomplete"
        if incomplete
        else "complete",
        "path": str(target),
        "coverage": coverage,
        "protocol_notes": manifest.get("protocol_notes", []),
        "meaning": "Rendered means this implementation produced artifacts; it does not assert numerical agreement with the manuscript.",
    }
    save_json(target / "coverage.json", result)
    rows = []
    for item in coverage:
        filename = f"{item['recipe']}/{item['recipe']}.png"
        preview = (
            f'<a href="{filename}"><img src="{filename}" style="max-width:100%;max-height:500px"></a>'
            if item["status"] in ("rendered", "partial")
            and (target / filename).exists()
            else ""
        )
        rows.append(
            f"<section><h2>{html.escape(item['recipe'] + ' · ' + item['title'])}</h2><p>{html.escape(item['status'])}</p>{preview}<pre>{html.escape(json.dumps(item.get('missing', []), indent=2))}</pre></section>"
        )
    (target / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>HSS paper artifacts</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;color:#223}section{border-top:1px solid #ccd;padding:16px 0}pre{white-space:pre-wrap}</style><h1>HSS paper artifact coverage</h1><p>Figures derive from recorded trials. Missing inputs and protocol differences are explicit.</p>'
        + "".join(rows)
    )
    return result

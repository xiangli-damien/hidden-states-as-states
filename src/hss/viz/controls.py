"""Render supplemental robustness checks from completed suite jobs."""

from pathlib import Path
import pandas as pd

from ..results import ResultCatalog
from ..analysis.tables import collect_tables, compare_trials, bootstrap_predictions
from ..experiments.artifacts import save_json
from .paper import suite_aliases
from .artifacts import FigureBundle
from . import plots

CONTROLS = [
    "reliability_trends",
    "reliability_centers",
    "preprocessing_control",
    "pca_control",
    "alignment_control",
    "kmeans_control",
    "global_clustering_control",
    "granularity_control",
    "token_granularity_control",
    "rms_control",
    "mfa_control",
    "preprocessing_prediction_control",
    "pca_prediction_control",
    "alignment_prediction_control",
    "method_prediction_control",
    "seed_prediction_control",
    "monitoring_far_scope_control",
]


def render_controls(
    roots, suite, destination, only=None, formats=("png", "svg"), bootstrap=0
):
    catalog = ResultCatalog(roots)
    aliases, issues, _ = suite_aliases(suite, catalog)
    selected = only or CONTROLS
    if set(selected) - set(CONTROLS):
        raise ValueError("Unknown control job")
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target == r.path or r.path in target.parents for r in catalog.results):
        raise ValueError(
            "Render destination must be outside immutable fitted trial directories"
        )
    coverage = []
    for name in selected:
        results = aliases.get(name, [])
        if not results or name in issues:
            coverage.append(
                {
                    "control": name,
                    "status": "missing_inputs",
                    "reason": issues.get(name, "No completed results"),
                }
            )
            continue
        checks = [r.validate() for r in results]
        if any(not c["valid"] for c in checks):
            coverage.append(
                {"control": name, "status": "invalid_inputs", "audit": checks}
            )
            continue
        bundle = FigureBundle(
            target / name,
            results,
            {"bootstrap_repeats": bootstrap, "bootstrap_unit": "held-out response"},
            formats,
        )
        tables = collect_tables(results)
        comparisons = compare_trials(results, all_pairs=True)
        tables["comparisons"] = comparisons
        if bootstrap:
            tables["prediction_intervals"] = pd.concat(
                [bootstrap_predictions(r, bootstrap) for r in results],
                ignore_index=True,
            )
        for key, frame in tables.items():
            bundle.table(key, frame)
        bundle.figure("profiles", plots.control_profiles(tables["profiles"]))
        bundle.figure(
            "diagnostics", plots.control_diagnostics(tables["diagnostics"], comparisons)
        )
        if len(tables["global_layer_purity"]):
            bundle.figure(
                "global_purity", plots.global_purity(tables["global_layer_purity"])
            )
        if len(tables["evaluation"]) and "auroc" in tables["evaluation"]:
            bundle.figure("prediction", plots.prediction_controls(tables["evaluation"]))
        bundle.finish(status="complete", control=name)
        coverage.append(
            {
                "control": name,
                "status": "rendered",
                "manifest": str(bundle.path / "manifest.json"),
            }
        )
    result = {
        "status": "failed"
        if any(c["status"] == "invalid_inputs" for c in coverage)
        else "incomplete"
        if any(c["status"] == "missing_inputs" for c in coverage)
        else "complete",
        "coverage": coverage,
        "path": str(target),
    }
    save_json(target / "coverage.json", result)
    return result

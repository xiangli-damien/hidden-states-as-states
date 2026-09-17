"""Compatibility entry point: regenerate diagnostics from saved results only."""

from pathlib import Path

from ..results import ResultCatalog
from ..analysis.tables import (
    collect_tables,
    compare_trials,
    state_map,
    dynamics,
    selection_surface,
    bootstrap_predictions,
)
from ..viz.artifacts import FigureBundle
from ..viz import plots
from .artifacts import save_json


def report(output_root, destination=None, *, formats=("png", "svg"), bootstrap=0):
    catalog = ResultCatalog(output_root)
    if not catalog.results:
        raise ValueError("No completed experiment outputs to report")
    root = Path(output_root).resolve()
    default = (
        root.parent / "reports" / root.name
        if (root / "_SUCCESS.json").exists()
        else root / "report"
    )
    target = Path(destination).resolve() if destination else default
    # Rendered outputs must not mutate a checksummed trial directory.
    if any(target == r.path or r.path in target.parents for r in catalog.results):
        raise ValueError(
            "Render destination must be outside immutable fitted trial directories"
        )
    target.mkdir(parents=True, exist_ok=True)
    audit = catalog.validate()
    if audit["status"] == "failed":
        raise ValueError(f"Result audit failed: {audit}")
    bundle = FigureBundle(
        target,
        catalog.results,
        {
            "comparison": "baseline controls + all fixed-K refit pairs",
            "bootstrap_repeats": bootstrap,
        },
        formats,
    )
    tables = collect_tables(catalog.results)
    tables["reliability"] = compare_trials(catalog.results)
    if bootstrap:
        import pandas as pd

        tables["bootstrap"] = pd.concat(
            [bootstrap_predictions(r, bootstrap) for r in catalog.results],
            ignore_index=True,
        )
    for name, frame in tables.items():
        bundle.table(name, frame)
    for r in catalog.results:
        child = FigureBundle(target / r.summary["trial_id"], [r], {}, formats)
        _, profile = dynamics(r)
        child.figure("profile", plots.profile_plot(profile))
        scan = selection_surface(r)
        child.table("selection_surface", scan)
        child.figure("selection_surface", plots.icl_surface(scan))
        nodes, edges = state_map(r)
        child.table("nodes", nodes)
        child.table("edges", edges)
        child.figure("state_graph", plots.state_graph(nodes, edges))
        child.finish(status="complete")
    bundle.finish(status="complete", audit=audit)
    catalog.frame().to_json(target / "experiments.json", orient="records", indent=2)
    result = {
        "status": "complete",
        "experiments": len(catalog.results),
        "path": str(target),
        "tables": [name + ".csv" for name in tables],
    }
    save_json(target / "index.json", result)
    return result

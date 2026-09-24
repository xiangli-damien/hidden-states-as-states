"""Bounded real OpenAct CPU check of the reorganized pipeline, never generation."""

import argparse
from dataclasses import replace
from pathlib import Path
import time
import json

from hss.data import DataSpec, prepare
from hss.experiments.artifacts import save_json, runtime_versions
from hss.experiments.config import (
    Experiment,
    ClusterConfig,
    ExecutionConfig,
    EvaluationConfig,
)
from hss.experiments.sweep import run_sweep
from hss.experiments.runner import run_experiment
from hss.experiments.reporting import report
from hss.experiments.paper import generate_suite
from hss.results import ResultCatalog
from hss.viz.paper import render_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    measured = {
        "scope": "Real OpenAct: first published 100 MATH responses, layers 0/8/final, K=2, CPU, 6 iterations. Engineering smoke only.",
        "runtime": runtime_versions(),
        "source": args.source,
    }

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        measured[name] = time.perf_counter() - start
        return result

    data = DataSpec(
        [args.source],
        dataset_id="math",
        max_samples=100,
        max_shards=1,
        expected_samples=100,
        layers=[0, 8, -1],
    )
    cache = timed("prepare_first_seconds", lambda: prepare(data, args.cache))
    timed("prepare_warm_seconds", lambda: prepare(data, args.cache))
    cfg = Experiment(
        name="real_openact_architecture_smoke",
        data=data,
        cluster=ClusterConfig(k=2, rank=2, n_init=1, max_iter=6, batch_size=32),
        execution=ExecutionConfig(
            cache_root=args.cache,
            artifact_cache_root=str(root / "fit_cache"),
            output_root=str(root / "trials"),
            workers=2,
            threads_per_worker=1,
            memory_gib=8,
            min_available_gib=16,
        ),
        grid={"cluster.method": ["gmm", "mfa", "minibatch_kmeans"]},
    )
    first = timed("three_method_sweep_seconds", lambda: run_sweep(cfg))
    if first["status"] != "complete":
        save_json(root / "validation.json", first)
        raise RuntimeError("Real-data sweep failed")
    second = timed("resume_seconds", lambda: run_sweep(cfg))
    assert all(r["cache_hit"] for r in second["results"])
    prediction = replace(
        cfg,
        name="real_prompt_prediction_smoke",
        grid={},
        data=replace(data, representation="prompt_last"),
        evaluation=EvaluationConfig(mode="prediction", methods=["HSS-NB", "Logistic"]),
    )
    pred = timed("prompt_prediction_seconds", lambda: run_experiment(prediction))
    audit = timed(
        "full_checksum_audit_seconds",
        lambda: ResultCatalog(root / "trials").validate(full=True),
    )
    assert audit["status"] == "complete"
    rendered = timed(
        "report_seconds",
        lambda: report(root / "trials", root / "report", bootstrap=100),
    )
    generated = generate_suite(
        "/lambda/nfs/dami/openact/runs", root / "study", args.cache, root / "trials"
    )
    baseline = next(
        r
        for r in first["results"]
        if json.loads((Path(r["path"]) / "config.json").read_text())["cluster"][
            "method"
        ]
        == "gmm"
    )
    save_json(
        root / "study/suite_status.json",
        {"default_map": {"status": "complete", "results": [baseline]}},
    )
    figures = timed(
        "paper_figure_seconds",
        lambda: render_paper(
            root / "trials",
            generated["suite"],
            root / "figures",
            only=["figure_03", "figure_04", "table_03"],
            max_trajectories=100,
        ),
    )
    measured.update(
        status="complete",
        data_cache={
            "path": str(cache.path),
            "bytes": cache.info["bytes"],
            "n_samples": cache.info["n_samples"],
        },
        trials=[
            {"id": r["trial_id"], "path": r["path"], "seconds": r["seconds"]}
            for r in first["results"]
        ],
        prediction_trial=pred["trial_id"],
        audit=audit["status"],
        resume_cache_hits=len(second["results"]),
        report=rendered["path"],
        figures=figures["status"],
    )
    save_json(root / "validation.json", measured)
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()

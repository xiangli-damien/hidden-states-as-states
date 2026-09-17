"""Bounded process grid search with frozen data plans and reusable fit caches."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from itertools import product
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import psutil

from .artifacts import digest, lock, save_json, source_version
from .config import Experiment, set_value
from ..data import CachedStates, prepare
from ..results import Result
from ..provenance import stage_version
from .runner import estimate_memory_gib, run_experiment, trial_identity
from .resources import storage_entries


def expand(cfg):
    base = cfg.to_dict()
    grid = base.pop("grid")
    variants = grid.pop("variants", [{}])
    if (
        not isinstance(variants, list)
        or not variants
        or any(not isinstance(v, dict) for v in variants)
    ):
        raise ValueError("grid.variants must be a nonempty list of dotted-key objects")
    if any(
        k.startswith("execution.")
        for k in list(grid) + [k for v in variants for k in v]
    ):
        raise ValueError(
            "Execution limits apply to the whole sweep; do not vary execution fields as scientific grid axes"
        )
    keys = sorted(grid)
    if any(not isinstance(grid[k], list) or not grid[k] for k in keys):
        raise ValueError("Every grid axis must be a nonempty list")
    count = math.prod(len(grid[k]) for k in keys) * len(variants)
    if count > cfg.execution.max_trials:
        raise ValueError(
            f"Grid expands to {count} trials, above max_trials={cfg.execution.max_trials}"
        )
    trials, seen = [], set()
    for values in product(*(grid[k] for k in keys)):
        for variant in variants:
            payload = json.loads(json.dumps(base))
            for key, value in list(zip(keys, values)) + list(variant.items()):
                set_value(payload, key, value)
            trial = Experiment.from_dict(payload)
            identifier = digest(trial.to_dict())
            if identifier not in seen:
                seen.add(identifier)
                trials.append(trial)
    return trials


def _worker(payload, prepared, version):
    cfg = Experiment.from_dict(payload)
    threads = str(cfg.execution.threads_per_worker)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = threads
    try:
        result = run_experiment(cfg, prepared=prepared, version=version)
        return {"status": "complete", **result}
    except Exception as exc:
        return {
            "status": "failed",
            "name": cfg.name,
            "error": repr(exc),
            "config": payload,
            "snapshot_path": str(prepared),
        }


def plan(cfg, *, refresh_data=False):
    version = source_version()
    configs = expand(cfg)
    # Sharding is a launch choice, not part of scientific identity or frozen data plan.
    canonical = cfg.to_dict()
    canonical["execution"].pop("task_index")
    canonical["execution"].pop("task_count")
    key = digest(
        {"config": canonical, "version": version, "reader": stage_version("data")}
    )
    folder = Path(cfg.execution.output_root).expanduser().resolve() / "sweeps" / key
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / ".plan.lock"):
        saved = folder / "plan.json"
        if saved.exists() and not refresh_data:
            result = json.loads(saved.read_text())
            for trial in result["trials"]:
                CachedStates(trial["snapshot_path"])
            return result
        datasets, trials, candidate_storage = {}, [], {}
        for config in configs:
            data_key = digest(asdict(config.data))
            if data_key not in datasets:
                datasets[data_key] = prepare(config.data, config.execution.cache_root)
            data = datasets[data_key]
            candidate_storage.update(storage_entries(config, data))
            trials.append(
                {
                    "config": config.to_dict(),
                    "snapshot_path": str(data.path),
                    "trial_id": digest(trial_identity(config, data, version)),
                    "estimated_ram_gib": estimate_memory_gib(data, config),
                    "n_samples": data.info["n_samples"],
                    "n_rows": data.n_items(),
                }
            )
        result = {
            "plan_id": key,
            "path": str(folder),
            "source_version": version,
            "trials": trials,
            "data_snapshots": [str(d.path) for d in datasets.values()],
            "created_at_epoch": time.time(),
            "candidate_fits_upper_bound": len(candidate_storage),
            "candidate_array_storage_gib_upper_bound": sum(candidate_storage.values())
            / 1024**3,
            "storage_estimate_note": "Candidate parameter arrays only, before existing-cache reuse. Excludes projected data, exports and filesystem overhead.",
        }
        save_json(saved, result)
        return result


def run_sweep(cfg, *, refresh_data=False):
    prepared_plan = plan(cfg, refresh_data=refresh_data)
    trials = prepared_plan["trials"][
        cfg.execution.task_index :: cfg.execution.task_count
    ]
    folder = Path(prepared_plan["path"])
    suffix = f"task_{cfg.execution.task_index:04d}"
    with lock(folder / (suffix + ".lock")):
        if not trials:
            return {"status": "complete", "trials": 0, "path": str(folder)}
        cached = {}
        for trial in trials:
            root = (
                Path(cfg.execution.output_root).expanduser().resolve()
                / trial["trial_id"]
            )
            if all(
                (root / name).exists()
                for name in (
                    "_SUCCESS.json",
                    "states.npy",
                    "rows.parquet",
                    "config.json",
                    "summary.json",
                )
            ):
                record = json.loads((root / "_SUCCESS.json").read_text())
                if record.get("trial_id") == trial["trial_id"]:
                    audit = Result(root).validate()
                    if not audit["valid"]:
                        raise ValueError(
                            f"Completed trial failed audit: {root}: {audit['errors']}"
                        )
                    cached[trial["trial_id"]] = {
                        "status": "complete",
                        **record,
                        "cache_hit": True,
                    }
        todo = [t for t in trials if t["trial_id"] not in cached]
        maximum = max((t["estimated_ram_gib"] for t in todo), default=0.0)
        available = (
            psutil.virtual_memory().available / 1024**3
            - cfg.execution.min_available_gib
        )
        ram_budget = min(cfg.execution.memory_gib, available)
        workers = (
            1
            if not todo
            else min(
                cfg.execution.workers,
                len(trials),
                int(ram_budget // maximum),
                max(1, (os.cpu_count() or 1) // cfg.execution.threads_per_worker),
            )
        )
        if workers < 1:
            raise MemoryError(
                f"One trial needs an estimated {maximum:.2f} GiB; budget currently {ram_budget:.2f} GiB"
            )
        print(
            json.dumps(
                {
                    "plan": str(folder),
                    "trials": len(trials),
                    "workers": workers,
                    "max_trial_ram_gib": maximum,
                    "ram_budget_gib": ram_budget,
                }
            ),
            flush=True,
        )
        results = []

        def accept(result):
            results.append(result)
            with (folder / (suffix + ".jsonl")).open("a") as f:
                f.write(json.dumps(result, allow_nan=False) + "\n")
            save_json(
                folder / (suffix + "_status.json"),
                {
                    "completed": len(results),
                    "total": len(trials),
                    "failed": sum(r["status"] != "complete" for r in results),
                    "workers": workers,
                },
            )
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(trials),
                        "status": result["status"],
                        "trial": result.get("trial_id"),
                    }
                ),
                flush=True,
            )

        iterator = iter(trials)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            pending = {}

            def submit():
                try:
                    trial = next(iterator)
                except StopIteration:
                    return False
                if trial["trial_id"] in cached:
                    accept(cached[trial["trial_id"]])
                    return True
                if not cfg.execution.retry_failed:
                    status = (
                        Path(cfg.execution.output_root)
                        / trial["trial_id"]
                        / "status.json"
                    )
                    if (
                        status.exists()
                        and json.loads(status.read_text()).get("status") == "failed"
                    ):
                        accept(
                            {"status": "skipped_failed", "trial_id": trial["trial_id"]}
                        )
                        return True
                # Execution overrides (worker/thread budgets) belong to this invocation.
                payload = trial["config"]
                payload["execution"] = asdict(cfg.execution)
                future = pool.submit(
                    _worker,
                    payload,
                    trial["snapshot_path"],
                    prepared_plan["source_version"],
                )
                pending[future] = trial
                return True

            while len(pending) < workers and submit():
                pass
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    trial = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "trial_id": trial["trial_id"],
                            "error": repr(exc),
                        }
                    accept(result)
                while len(pending) < workers and submit():
                    pass
        summary = {
            "status": "complete"
            if all(r["status"] == "complete" for r in results)
            else "failed",
            "path": str(folder),
            "trials": len(results),
            "workers": workers,
            "failed": sum(r["status"] != "complete" for r in results),
            "results": results,
        }
        save_json(folder / (suffix + "_summary.json"), summary)
        return summary

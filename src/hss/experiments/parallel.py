"""Precompute independent CPU layers; the ordinary runner assembles frozen maps.

Uses exactly the runner's split, transform and candidate cache identities. No
change to scientific selection, and no fitting on held-out prediction rows.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import time

import psutil
from threadpoolctl import threadpool_limits

from ..data import CachedStates, prepare
from ..provenance import stage_version
from .artifacts import save_json, digest
from .config import Experiment
from .fitting import fit_candidates, fit_transform
from .runner import fitting_split, _rows_digest, estimate_memory_gib


def _layer(payload, snapshot, layer, progress):
    cfg = Experiment.from_dict(payload)
    data = CachedStates(snapshot)
    meta = data.meta.copy()
    if cfg.evaluation.positive_label == 0:
        meta["label"] = 1 - meta.label
    _, train = fitting_split(meta, cfg)
    context = dict(
        version=stage_version("fit"),
        snapshot=data.info["key"],
        layer=layer,
        train=_rows_digest(train),
    )
    cache = cfg.execution.artifact_cache_root or cfg.execution.cache_root
    started = time.time()
    save_json(
        progress, dict(status="running", layer=layer, name=cfg.name, started_at=started)
    )
    with threadpool_limits(limits=cfg.execution.threads_per_worker):
        _, X, tk = fit_transform(data.array(layer), train, cfg, context, cache)
        if cfg.evaluation.fixed_k_map:
            choices = json.loads(Path(cfg.evaluation.fixed_k_map).read_text())
            ks = {r["layer"]: r["selected"]["k"] for r in choices}
            if set(ks) != set(data.layers()):
                raise ValueError("Fixed K reference must cover exactly the data layers")
            cfg = replace(cfg, cluster=replace(cfg.cluster, k=ks[layer]))
        model, scan = fit_candidates(X, cfg, {**context, "transform_key": tk}, cache)
    result = dict(
        status="complete",
        name=cfg.name,
        layer=layer,
        selected=scan["selected"],
        model=model.config(),
        seconds=time.time() - started,
    )
    save_json(progress, result)
    return result


def prefit_layers(configs, directory, *, workers=10, on_ready=None, layer_ids=None):
    """One bounded pool shared by all configs, checkpointing every K and layer."""
    directory = Path(directory)
    tasks, snapshots = [], {}
    pending = {}
    by_name = {c.name: c for c in configs}
    if len(by_name) != len(configs):
        raise ValueError("Parallel configuration names must be unique")
    maximum = 0
    for cfg in configs:
        cfg.validate()
        if (
            cfg.cluster.backend == "gpu"
            or cfg.evaluation.fixed_map
            or cfg.evaluation.mode == "global_control"
        ):
            raise ValueError("Layer prefit supports independent CPU maps only")
        data = prepare(cfg.data, cfg.execution.cache_root)
        assert data.info["identity"]["spec"] == asdict(cfg.data)
        snapshots[cfg.name] = str(data.path)
        layers = data.layers() if layer_ids is None else list(layer_ids)
        if (
            not layers
            or len(layers) != len(set(layers))
            or not set(layers) <= set(data.layers())
        ):
            raise ValueError(
                "Prefit layer IDs must be a nonempty subset without duplicates"
            )
        if layer_ids is not None and on_ready:
            raise ValueError("Subset prefit cannot signal that a full trial is ready")
        pending[cfg.name] = len(layers)
        maximum = max(maximum, estimate_memory_gib(data, cfg))
        for layer in layers:
            key = digest(
                dict(
                    config=cfg.to_dict(),
                    snapshot=data.info["key"],
                    layer=layer,
                    fit=stage_version("fit"),
                )
            )
            tasks.append(
                (cfg.to_dict(), str(data.path), layer, str(directory / f"{key}.json"))
            )
    reserve = max(c.execution.min_available_gib for c in configs)
    budget = min(
        min(c.execution.memory_gib for c in configs),
        psutil.virtual_memory().available / 1024**3 - reserve,
    )
    workers = min(workers, len(tasks), int(budget // maximum))
    if workers < 1:
        raise MemoryError("Insufficient RAM for a layer worker after reserve")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(_layer, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                save_json(
                    task[3],
                    dict(
                        status="failed",
                        name=task[0]["name"],
                        layer=task[2],
                        error=repr(exc),
                    ),
                )
                raise
            print(
                json.dumps({k: result[k] for k in ("name", "layer", "seconds")}),
                flush=True,
            )
            pending[result["name"]] -= 1
            if not pending[result["name"]] and on_ready:
                on_ready(by_name[result["name"]], snapshots[result["name"]])
    return snapshots


if __name__ == "__main__":
    import argparse
    from .config import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="+")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    prefit_layers(
        [load(p) for p in args.config],
        args.directory,
        workers=args.workers,
        layer_ids=args.layers,
    )

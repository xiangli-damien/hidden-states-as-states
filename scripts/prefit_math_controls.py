"""Overlap matched-K controls with a running full GMM layer scan.

Only completed baseline layers are eligible. The ordinary study runner later
reuses identical cache entries and remains the authority for final trial output.
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from hss.data import prepare
from hss.experiments.artifacts import lock, save_json
from hss.experiments.config import load
from hss.experiments.parallel import _layer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--tag", default="overlap")
    args = p.parse_args()
    root = Path(args.study)
    base = load(root / "configs/mean_gmm.json")
    data = prepare(base.data, base.execution.cache_root)
    layers = set(data.layers() if args.layers is None else args.layers)
    if (
        not layers
        or not layers <= set(data.layers())
        or not args.tag.replace("-", "").isalnum()
    ):
        raise ValueError("Invalid layer subset or progress tag")
    work = root / args.tag
    methods = ("mfa", "kmeans", "minibatch_kmeans")
    with (
        lock(root / f"{args.tag}.lock"),
        ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
    ):
        submitted, pending, completed = set(), {}, []
        while len(completed) < len(layers) * len(methods):
            for path in sorted((root / "progress").glob("*.json")):
                record = json.loads(path.read_text())
                layer = record.get("layer")
                if (
                    record.get("name") != "mean_gmm"
                    or record.get("status") != "complete"
                    or layer not in layers
                ):
                    continue
                for method in methods:
                    key = (layer, method)
                    if key in submitted:
                        continue
                    name = f"{args.tag}_{method}_L{layer}"
                    cfg = replace(
                        base,
                        name=name,
                        cluster=replace(
                            base.cluster, method=method, k=record["selected"]["k"]
                        ),
                    )
                    save_json(work / "configs" / f"{name}.json", cfg.to_dict())
                    future = pool.submit(
                        _layer,
                        cfg.to_dict(),
                        str(data.path),
                        layer,
                        str(work / f"{name}.json"),
                    )
                    pending[future] = key
                    submitted.add(key)
            for future in list(pending):
                if future.done():
                    result = future.result()
                    completed.append(pending.pop(future))
                    print(
                        json.dumps(
                            {k: result[k] for k in ("name", "layer", "seconds")}
                        ),
                        flush=True,
                    )
            save_json(
                work / "status.json",
                dict(
                    status="running"
                    if pending or len(completed) < len(layers) * len(methods)
                    else "complete",
                    completed=len(completed),
                    submitted=len(submitted),
                    total=len(layers) * len(methods),
                ),
            )
            if len(completed) < len(layers) * len(methods):
                time.sleep(10)

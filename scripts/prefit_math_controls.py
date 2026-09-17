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
    args = p.parse_args()
    root = Path(args.study)
    base = load(root / "configs/mean_gmm.json")
    data = prepare(base.data, base.execution.cache_root)
    methods = ("mfa", "kmeans", "minibatch_kmeans")
    with (
        lock(root / "overlap.lock"),
        ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
    ):
        submitted, pending, completed = set(), {}, []
        while len(completed) < len(data.layers()) * len(methods):
            for path in sorted((root / "progress").glob("*.json")):
                record = json.loads(path.read_text())
                layer = record.get("layer")
                if (
                    record.get("name") != "mean_gmm"
                    or record.get("status") != "complete"
                ):
                    continue
                for method in methods:
                    key = (layer, method)
                    if key in submitted:
                        continue
                    name = f"overlap_{method}_L{layer}"
                    cfg = replace(
                        base,
                        name=name,
                        cluster=replace(
                            base.cluster, method=method, k=record["selected"]["k"]
                        ),
                    )
                    save_json(root / "overlap/configs" / f"{name}.json", cfg.to_dict())
                    future = pool.submit(
                        _layer,
                        cfg.to_dict(),
                        str(data.path),
                        layer,
                        str(root / "overlap" / f"{name}.json"),
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
                root / "overlap/status.json",
                dict(
                    status="running"
                    if pending or len(completed) < len(data.layers()) * len(methods)
                    else "complete",
                    completed=len(completed),
                    submitted=len(submitted),
                    total=len(data.layers()) * len(methods),
                ),
            )
            if len(completed) < len(data.layers()) * len(methods):
                time.sleep(10)

"""Continue a frozen CPU study on one GPU, preserving fits and EM checkpoints.

The source controller must be stopped before migration. The source directory is
read-only. Completed candidates retain their original fit paths and provenance;
unfinished MFA restarts are copied into the GPU cache with a migration receipt.
No scientific hyperparameter or feature transform is changed.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from hss.data import CachedStates
from hss.experiments.artifacts import digest, file_digest, lock, save_json
from hss.experiments.cluster_ablation import ClusterStudy, method_config
from hss.experiments.config import Experiment
from hss.experiments.fitting import common_parameters
from hss.experiments.runner import _rows_digest
from hss.provenance import stage_version


class AcceleratedStudy(ClusterStudy):
    """MFA alone uses CUDA; GMM/KMeans and score/ICL evaluation remain on CPU."""

    def report(self):
        super().report()
        page = self.root / "index.html"
        body = page.read_text().replace(" · CPU</p>", " · MFA EM on GPU</p>")
        body = body.replace("<h2>Ablation results</h2>",
            "<p>Completed CPU fits are retained with their original provenance; "
            "new MFA fits continue in float64 on GPU. Scientific parameters are unchanged. "
            "<a href='migration.json'>Execution migration receipt</a></p><h2>Ablation results</h2>")
        page.write_text(body)

    def run_tasks(self, phase, tasks):
        cpu = [t for t in tasks if t["method"] != "mfa"]
        gpu = [t for t in tasks if t["method"] == "mfa"]
        if cpu:
            base, views = self.base, self.views
            try:
                self.base = replace(base, cluster=replace(base.cluster, backend="cpu"))
                self.views = {v: replace(c, cluster=replace(c.cluster, backend="cpu"))
                              for v, c in views.items()}
                super().run_tasks(phase + (":cpu" if gpu else ""), cpu)
            finally:
                self.base, self.views = base, views
        if gpu:
            super().run_tasks(phase, gpu)


def fit_key(task, cfg, snapshot_key):
    context = dict(version=stage_version("fit"), snapshot=snapshot_key,
                   layer=task["layer"], train=_rows_digest(np.arange(5000)))
    transform_key = digest({**context, "transform": asdict(cfg.transform),
                            "pca_seed": cfg.evaluation.split_seed})
    return digest({**context, "transform_key": transform_key,
                   "method": cfg.cluster.method, "params": common_parameters(cfg.cluster),
                   "k": task["k"], "seed": cfg.seed + 1009 * max(0, task["layer"])})


def assert_stopped(pid):
    if not psutil.pid_exists(pid):
        return
    parent = psutil.Process(pid)
    for proc in [parent, *parent.children(recursive=True)]:
        if proc.status() not in (psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE):
            raise RuntimeError(f"Source process {proc.pid} is still running")


def migrate(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Acceleration requires a separate study directory")
    protocol = json.loads((source / "protocol.json").read_text())
    identity = protocol["identity"]
    if identity["source"] != stage_version("trial"):
        raise ValueError("Scientific source changed; migration requires original fit code")
    old = Experiment.from_dict(identity["config"])
    if old.cluster.backend != "cpu" or old.evaluation.mode != "geometry":
        raise ValueError("Expected a CPU geometry study")
    if old.data.expected_samples != 5000:
        raise ValueError("This migration currently supports the frozen 5,000-row study")
    state = json.loads((source / "study.json").read_text())
    assert_stopped(state["pid"])
    recipe = dict(identity["recipe"], directory=str(destination))
    cfg = replace(old, cluster=replace(old.cluster, backend="gpu", device="cuda:0"),
                  execution=replace(old.execution, workers=1, threads_per_worker=2,
                                    min_gpu_free_gib=8))
    study = AcceleratedStudy(recipe, cfg, destination)
    receipt_path = destination / "migration.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text())["source"] != str(source):
            raise ValueError("Destination belongs to a different source study")
        return study
    study.prepare()
    receipt = dict(source=str(source), destination=str(destination), at=time.time(),
                   driver_sha256=file_digest(Path(__file__)), source_protocol_sha256=file_digest(source / "protocol.json"),
                   source_trial_version=identity["source"], target_trial_version=study.version,
                   execution_changes={"backend": ["cpu", "gpu"], "workers": [old.execution.workers, 1],
                                      "threads": [old.execution.threads_per_worker, 2],
                                      "routing": "MFA EM only on GPU; GMM, KMeans and ICL remain CPU"},
                   scientific_parameters_unchanged=True, candidates=[], checkpoints=[])
    for path in sorted((source / "candidates").glob("*.json")):
        item = json.loads(path.read_text())
        if item["status"] not in ("complete", "failed"):
            continue
        target = destination / "candidates" / path.name
        save_json(target, {**item, "imported_from": str(path)})
        study.records[path.stem] = {**item, "imported_from": str(path)}
        receipt["candidates"].append(dict(path=str(path), sha256=file_digest(path)))
    for task in state.get("active", []):
        if task["method"] != "mfa":
            continue
        view = task["view"]
        data = study.snapshots[view]
        old_cfg = method_config(old, recipe, "mfa", task["rank"], task["seed"])
        new_cfg = method_config(study.views[view], recipe, "mfa", task["rank"], task["seed"])
        src = source / "cache/fits" / fit_key(task, old_cfg, data.info["key"])
        dst = destination / "cache/fits" / fit_key(task, new_cfg, data.info["key"])
        if (src / "fit.json").exists():
            # A completed worker result may have been written just before STOP.
            # The candidate import above owns it; do not duplicate a finished fit.
            continue
        for checkpoint in sorted(src.glob("restart_*.npz")):
            with np.load(checkpoint, allow_pickle=False) as z:
                model = json.loads(str(z["config_json"]))
                if z["loadings"].shape != (task["k"], data.state_dim(), task["rank"]):
                    raise ValueError("Incompatible checkpoint shape")
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, dst / checkpoint.name)
            receipt["checkpoints"].append(dict(task=task, source=str(checkpoint),
                destination=str(dst / checkpoint.name), sha256=file_digest(checkpoint),
                n_iter=len(model.get("history", [])), converged=model.get("converged")))
    # Completed maps remain immutable and linked to their original CPU provenance.
    study.state["jobs"] = state.get("jobs", {})
    study.state.update(status="migrated", phase="ready_for_gpu", migrated_from=str(source))
    save_json(destination / "study.json", study.state)
    save_json(receipt_path, receipt)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    Path(args.directory).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with lock(Path(args.directory) / "study.lock"), threadpool_limits(2):
        study = migrate(args.source, args.directory)
        try:
            study.run()
        except Exception as exc:
            study.state.update(status="failed", error=repr(exc), updated_at=time.time())
            save_json(study.root / "study.json", study.state)
            raise

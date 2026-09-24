"""Restartable raw-state clustering study, with explicit search coverage.

Fits, selection, aligned maps and reporting are separate stages. Every finite
candidate is retained, but only converged candidates can enter selection.
No correctness label is used to select K, rank, initialization or a seed.
"""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import html
import json
import multiprocessing
import os
from pathlib import Path
import time

import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from ..data import CachedStates, prepare
from ..provenance import stage_version
from ..results import Result
from .artifacts import digest, file_digest, lock, save_json
from .config import Experiment, load
from .fitting import fit_one_candidate, fit_transform, select_candidate
from .runner import _rows_digest, estimate_memory_gib, fitting_split, run_experiment


def load_recipe(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(path).resolve()
    with path.open("rb") as f:
        recipe = tomllib.load(f)
    cfg = load(path.parent / recipe["base"])
    if (
        cfg.transform.standardize
        or cfg.transform.whiten
        or cfg.transform.pca_components is not None
    ):
        raise ValueError("This raw-state study forbids input transforms")
    if cfg.data.representation != "mean" or cfg.evaluation.mode != "geometry":
        raise ValueError("This study fits response means for descriptive geometry")
    if (
        not cfg.cluster.require_convergence
        or not cfg.cluster.save_candidate_assignments
    ):
        raise ValueError("Study requires convergence admission and saved assignments")
    if cfg.cluster.backend != "cpu":
        raise ValueError("This study reserves the GPU for OpenAct collection")
    if (
        not recipe["ranks"]
        or min(recipe["ranks"]) < 0
        or len(set(recipe["ranks"])) != len(recipe["ranks"])
    ):
        raise ValueError("Invalid or duplicate ranks")
    if set(recipe["criteria"]) - {"icl", "bic"} or not recipe["criteria"]:
        raise ValueError("Invalid criteria")
    if (
        min(recipe["icl_tolerances"]) < 0
        or recipe["refine_radius"] < 1
        or recipe["refine_rounds"] < 1
    ):
        raise ValueError("Invalid tolerance or refinement range")
    if not recipe["mfa_initial_k"] or any(
        k < cfg.cluster.k_min or k > cfg.cluster.k_max for k in recipe["mfa_initial_k"]
    ):
        raise ValueError("MFA coarse K grid lies outside the configured range")
    return recipe, cfg


def method_config(base, recipe, method, rank=0, seed=None):
    c = replace(base.cluster, method=method, rank=rank)
    if method == "kmeans":
        c = replace(c, n_init=recipe["kmeans_n_init"], max_iter=1000, tol=1e-4)
    return replace(base, cluster=c, seed=base.seed if seed is None else seed)


def task_key(task):
    return digest(
        {k: task[k] for k in ("view", "layer", "method", "rank", "k", "seed")}
    )


def candidate_worker(task, payload, snapshot, destination):
    cfg = Experiment.from_dict(payload)
    data = CachedStates(snapshot)
    _, train = fitting_split(data.meta, cfg)
    context = dict(
        version=stage_version("fit"),
        snapshot=data.info["key"],
        layer=task["layer"],
        train=_rows_digest(train),
    )
    started = time.time()
    result = dict(task, status="running", started_at=started, pid=os.getpid())
    save_json(destination, result)
    try:
        with threadpool_limits(limits=cfg.execution.threads_per_worker):
            _, X, transform_key = fit_transform(
                data.array(task["layer"]),
                train,
                cfg,
                context,
                cfg.execution.artifact_cache_root,
            )
            record = fit_one_candidate(
                X,
                cfg,
                {**context, "transform_key": transform_key},
                cfg.execution.artifact_cache_root,
                task["k"],
            )
        result.update(status="complete", record=record, seconds=time.time() - started)
    except Exception as exc:
        import traceback

        result.update(
            status="failed",
            error=repr(exc),
            traceback=traceback.format_exc(),
            seconds=time.time() - started,
        )
    save_json(destination, result)
    return result


def candidate_records(records, view, layer, method, rank, seed):
    return [
        r["record"]
        for r in records.values()
        if r["status"] == "complete"
        and (r["view"], r["layer"], r["method"], r["rank"], r["seed"])
        == (view, layer, method, rank, seed)
    ]


def refinement_k(records, cluster, tolerances, criteria, radius):
    """Densify around EACH selection policy; also works when raw ICL is negative."""
    tried = {r["k"] for r in records}
    wanted = set()
    for criterion in criteria:
        for tolerance in tolerances:
            try:
                chosen = select_candidate(
                    records,
                    replace(
                        cluster,
                        selection_criterion=criterion,
                        parsimony_tolerance=tolerance,
                    ),
                )["k"]
            except ValueError:
                continue
            wanted.update(
                range(
                    max(cluster.k_min, chosen - radius),
                    min(cluster.k_max, chosen + radius) + 1,
                )
            )
    return sorted(wanted - tried)


class ClusterStudy:
    def __init__(self, recipe, cfg, directory):
        self.recipe, self.base = recipe, cfg
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        cfg.execution = replace(
            cfg.execution,
            artifact_cache_root=str(self.root / "cache"),
            output_root=str(self.root / "trials"),
        )
        self.version = stage_version("trial")
        identity = dict(
            recipe=recipe,
            config=cfg.to_dict(),
            source=self.version,
            scheduler=file_digest(Path(__file__)),
        )
        marker = self.root / "protocol.json"
        if marker.exists() and json.loads(marker.read_text())["identity"] != identity:
            raise ValueError("Study protocol/source changed: use a new study directory")
        save_json(
            marker,
            dict(
                identity=identity,
                note=(
                    "Raw full-dimensional response means. No additional normalization/PCA. All fits use all responses; "
                    "this is descriptive, not a held-out prediction experiment. MFA adaptive K search is not exhaustive. "
                    "SVD initializes covariance factors only; the fitting inputs retain all original coordinates."
                ),
            ),
        )
        self.views, self.snapshots, self.records = {}, {}, {}
        self.state = dict(
            status="preparing",
            source_version=self.version,
            jobs={},
            scope="Qwen2-7B-Instruct / MATH raw response-mean cluster ablation",
        )
        previous = self.root / "study.json"
        if previous.exists():
            self.state = json.loads(previous.read_text())
        for p in (self.root / "candidates").glob("*.json"):
            r = json.loads(p.read_text())
            if r["status"] in ("complete", "failed"):
                self.records[p.stem] = r

    def prepare(self):
        post = replace(self.base, data=replace(self.base.data, final_norm="post"))
        data = prepare(post.data, post.execution.cache_root)
        self.views["post"], self.snapshots["post"] = post, data
        if self.recipe["include_pre_final"]:
            pre = replace(
                self.base,
                data=replace(
                    self.base.data, final_norm="pre", layers=[data.layers()[-1]]
                ),
            )
            self.views["pre_final"] = pre
            self.snapshots["pre_final"] = prepare(pre.data, pre.execution.cache_root)
        if max(self.recipe["ranks"]) >= data.state_dim():
            raise ValueError("MFA rank exceeds the source hidden dimension")
        for name, snapshot in self.snapshots.items():
            marker = self.root / "snapshots" / f"{name}.json"
            if (
                marker.exists()
                and json.loads(marker.read_text())["key"] != snapshot.info["key"]
            ):
                raise ValueError(
                    "Source data changed after the study snapshot was frozen"
                )
            save_json(marker, snapshot.info)
            snapshot.meta.to_parquet(
                self.root / "snapshots" / f"{name}_rows.parquet", index=False
            )
        self.save_status("prepared", 0, 0, [])

    def task(self, view, layer, method, rank, k, seed=None):
        return dict(
            view=view,
            layer=layer,
            method=method,
            rank=rank,
            k=k,
            seed=self.base.seed if seed is None else seed,
        )

    def save_status(self, phase, done, total, active):
        complete = [r for r in self.records.values() if r["status"] == "complete"]
        self.state.update(
            phase=phase,
            status="running",
            pid=os.getpid(),
            updated_at=time.time(),
            phase_completed=done,
            phase_total=total,
            active=active,
            completed_candidates=len(complete),
            converged_candidates=sum(
                r["record"].get("converged") is True for r in complete
            ),
            unconverged_candidates=sum(
                r["record"].get("converged") is not True for r in complete
            ),
            failed_candidates=sum(
                r["status"] == "failed" for r in self.records.values()
            ),
            ram_available_gib=psutil.virtual_memory().available / 1024**3,
        )
        save_json(self.root / "study.json", self.state)

    def run_tasks(self, phase, tasks):
        unique = {task_key(t): t for t in tasks}
        # Failed tasks are retained visibly, not silently retried forever.
        todo = [t for key, t in unique.items() if key not in self.records]
        done = len(unique) - len(todo)
        x = self.base.execution
        maximum = max(
            estimate_memory_gib(d, self.views[v]) for v, d in self.snapshots.items()
        )
        workers = min(
            x.workers,
            max(1, (os.cpu_count() or 1) // x.threads_per_worker),
            int(x.memory_gib // maximum),
        )
        if workers < 1:
            raise MemoryError("No layer worker fits the study memory budget")
        pending, index = {}, 0
        self.save_status(phase, done, len(unique), [])
        print(
            json.dumps(
                dict(phase=phase, requested=len(unique), cached=done, workers=workers)
            ),
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            while pending or index < len(todo):
                while (
                    index < len(todo)
                    and len(pending) < workers
                    and psutil.virtual_memory().available / 1024**3
                    >= x.min_available_gib + maximum
                ):
                    task = todo[index]
                    cfg = method_config(
                        self.views[task["view"]],
                        self.recipe,
                        task["method"],
                        task["rank"],
                        task["seed"],
                    )
                    destination = self.root / "candidates" / f"{task_key(task)}.json"
                    future = pool.submit(
                        candidate_worker,
                        task,
                        cfg.to_dict(),
                        str(self.snapshots[task["view"]].path),
                        str(destination),
                    )
                    pending[future] = task
                    index += 1
                if not pending:
                    self.save_status(phase + ":waiting_for_ram", done, len(unique), [])
                    time.sleep(10)
                    continue
                finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    task = pending.pop(future)
                    result = future.result()
                    self.records[task_key(task)] = result
                    done += 1
                    if result["status"] == "failed":
                        print(
                            json.dumps(dict(task=task, error=result["error"])),
                            flush=True,
                        )
                    if done % 20 == 0 or done == len(unique):
                        print(
                            json.dumps(
                                dict(phase=phase, completed=done, total=len(unique))
                            ),
                            flush=True,
                        )
                self.save_status(phase, done, len(unique), list(pending.values()))
        self.report()

    def get(self, view, layer, method, rank=0, seed=None):
        return candidate_records(
            self.records,
            view,
            layer,
            method,
            rank,
            self.base.seed if seed is None else seed,
        )

    def units(self):
        # The terminal layer is fitted first, then early/middle layers.
        return [
            (view, layer)
            for view, data in self.snapshots.items()
            for layer in [data.layers()[-1], *data.layers()[:-1]]
        ]

    def selections(
        self, view, method, rank, tolerance, criterion, seed=None, matched=False
    ):
        config = method_config(self.views[view], self.recipe, method, rank).cluster
        config = replace(
            config, parsimony_tolerance=tolerance, selection_criterion=criterion
        )
        choices = []
        for layer in self.snapshots[view].layers():
            records = self.get(view, layer, method, rank, seed)
            if matched and method == "mfa":
                try:
                    reference = select_candidate(
                        self.get(view, layer, "gmm"),
                        replace(
                            self.base.cluster,
                            method="gmm",
                            parsimony_tolerance=self.recipe["reference_tolerance"],
                        ),
                    )
                except ValueError:
                    return None
                records = [r for r in records if r["k"] == reference["k"]]
            try:
                selected = select_candidate(records, config)
            except ValueError:
                return None
            choices.append(
                dict(
                    layer=layer,
                    selected=selected,
                    candidates=records,
                    excluded_unconverged=[
                        r["k"] for r in records if not r.get("converged")
                    ],
                )
            )
        return choices

    def export(self, tag, view, method, rank, tolerance, criterion):
        name = f"{tag}_{view}_{method}_r{rank}_{criterion}_t{tolerance:g}"
        if name in self.state["jobs"]:
            return
        choices = self.selections(
            view, method, rank, tolerance, criterion, matched=tag == "matched_k"
        )
        if choices is None:
            return
        scan_path = self.root / "selections" / f"{name}.json"
        save_json(scan_path, choices)
        cfg = method_config(self.views[view], self.recipe, method, rank)
        cfg = replace(
            cfg,
            name=name,
            cluster=replace(
                cfg.cluster,
                selection_criterion=criterion,
                parsimony_tolerance=tolerance,
            ),
            evaluation=replace(cfg.evaluation, fixed_k_map=str(scan_path)),
        )
        for assignment in ("posterior", "nearest"):
            variant = replace(
                cfg,
                name=name + "_" + assignment,
                cluster=replace(cfg.cluster, assignment=assignment),
            )
            save_json(self.root / "configs" / f"{variant.name}.json", variant.to_dict())
            result = run_experiment(
                variant, prepared=str(self.snapshots[view].path), version=self.version
            )
            audit = Result(result["path"]).validate(full=True)
            if not audit["valid"]:
                raise ValueError(audit)
            self.state["jobs"][variant.name] = dict(
                status="complete", path=result["path"], audit=audit
            )
        self.state["jobs"][name] = dict(status="selection", path=str(scan_path))
        save_json(self.root / "study.json", self.state)

    def export_available(self, tag):
        for view in self.views:
            for method, ranks in [
                ("gmm", [0]),
                ("kmeans", [0]),
                ("mfa", self.recipe["ranks"]),
            ]:
                for rank in ranks:
                    for criterion in (
                        self.recipe["criteria"] if method != "kmeans" else ["icl"]
                    ):
                        for tolerance in (
                            self.recipe["icl_tolerances"]
                            if method != "kmeans"
                            else [0.0]
                        ):
                            self.export(tag, view, method, rank, tolerance, criterion)
        self.report()

    def report(self):
        records = []
        for item in self.records.values():
            row = {
                k: item[k]
                for k in ("view", "layer", "method", "rank", "k", "seed", "status")
            }
            row.update(item.get("record", {}))
            row["error"] = item.get("error", "")
            records.append(row)
        table = pd.DataFrame(records)
        table.to_csv(self.root / "candidate_metrics.csv", index=False)
        if "converged" not in table:
            table["converged"] = False
        summary = (
            table.groupby(["view", "method", "rank"], dropna=False).agg(
                candidates=("k", "size"), converged=("converged", "sum")
            )
            if len(table)
            else pd.DataFrame()
        )
        status = {
            k: self.state.get(k)
            for k in (
                "phase",
                "completed_candidates",
                "converged_candidates",
                "unconverged_candidates",
                "failed_candidates",
            )
        }
        page = "<!doctype html><meta charset='utf-8'><title>Qwen MATH cluster ablation</title>"
        page += "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>"
        page += "<h1>Qwen2-7B-Instruct · MATH 5,000</h1><p>Raw response means · no additional normalization or PCA · CPU</p>"
        page += "<p>Post-RMS final layer and a separate pre-RMS final-layer control. Only converged fits enter selection. MFA K search is adaptive, not exhaustive. These are descriptive full-data clusters, not held-out prediction results.</p>"
        page += (
            "<pre>"
            + html.escape(json.dumps(status, indent=2))
            + "</pre>"
            + summary.to_html()
        )
        page += "<p><a href='candidate_metrics.csv'>All candidate scores and fit paths</a> · <a href='protocol.json'>Frozen protocol</a> · <a href='study.json'>Live status and selected result paths</a></p>"
        page += "<p>Each candidate saves model.npz, fit.json, and assignments.npz (nearest/posterior). MFA also saves restart checkpoints and convergence histories. Row identities are in snapshots/*_rows.parquet; selected full-layer maps use the standard HSS Result format in trials/.</p>"
        from ..viz.cluster_ablation import render_ablation

        page += render_ablation(self.root, self.records, self.base, self.recipe)
        (self.root / "index.html").write_text(page)

    def run(self):
        self.prepare()
        lo, hi = self.base.cluster.k_min, self.base.cluster.k_max
        tasks = [
            self.task(view, layer, method, 0, k)
            for view, layer in self.units()
            for method in ("gmm", "kmeans")
            for k in range(lo, hi + 1)
        ]
        self.run_tasks("baseline_full_k", tasks)
        self.export_available("baseline")
        tasks = []
        for view, layer in self.units():
            try:
                reference = select_candidate(
                    self.get(view, layer, "gmm"),
                    replace(
                        self.base.cluster,
                        method="gmm",
                        parsimony_tolerance=self.recipe["reference_tolerance"],
                    ),
                )
            except ValueError:
                continue
            for rank in self.recipe["ranks"]:
                tasks.append(self.task(view, layer, "mfa", rank, reference["k"]))
        self.run_tasks("mfa_matched_k_rank", tasks)
        self.export_available("matched_k")
        tasks = [
            self.task(view, layer, "mfa", rank, k)
            for view, layer in self.units()
            for rank in self.recipe["ranks"]
            for k in self.recipe["mfa_initial_k"]
        ]
        self.run_tasks("mfa_independent_k_coarse", tasks)
        for round_index in range(self.recipe["refine_rounds"]):
            tasks = []
            for view, layer in self.units():
                for rank in self.recipe["ranks"]:
                    ks = refinement_k(
                        self.get(view, layer, "mfa", rank),
                        replace(self.base.cluster, method="mfa", rank=rank),
                        self.recipe["icl_tolerances"],
                        self.recipe["criteria"],
                        self.recipe["refine_radius"],
                    )
                    tasks.extend(self.task(view, layer, "mfa", rank, k) for k in ks)
            self.run_tasks(f"mfa_refine_{round_index + 1}", tasks)
        self.export_available("independent_k")
        tasks = []
        for view, layer in self.units():
            for method, ranks in [
                ("gmm", [0]),
                ("kmeans", [0]),
                ("mfa", self.recipe["ranks"]),
            ]:
                for rank in ranks:
                    for criterion in (
                        self.recipe["criteria"] if method != "kmeans" else ["icl"]
                    ):
                        for tolerance in (
                            self.recipe["icl_tolerances"]
                            if method != "kmeans"
                            else [0.0]
                        ):
                            try:
                                selected = select_candidate(
                                    self.get(view, layer, method, rank),
                                    replace(
                                        method_config(
                                            self.base, self.recipe, method, rank
                                        ).cluster,
                                        selection_criterion=criterion,
                                        parsimony_tolerance=tolerance,
                                    ),
                                )
                            except ValueError:
                                continue
                            tasks.extend(
                                self.task(
                                    view, layer, method, rank, selected["k"], seed
                                )
                                for seed in self.recipe["stability_seeds"]
                            )
        self.run_tasks("selected_seed_stability", tasks)
        self.state.update(
            status=(
                "finished_with_unresolved_fits"
                if self.state["failed_candidates"]
                or self.state["unconverged_candidates"]
                else "complete"
            ),
            phase="finished",
            updated_at=time.time(),
        )
        save_json(self.root / "study.json", self.state)
        self.report()


def run_study(recipe_path, directory=None):
    recipe, cfg = load_recipe(recipe_path)
    root = Path(directory or recipe["directory"])
    with lock(root / "study.lock"):
        study = ClusterStudy(recipe, cfg, root)
        try:
            study.run()
        except Exception as exc:
            study.state.update(status="failed", error=repr(exc), updated_at=time.time())
            save_json(study.root / "study.json", study.state)
            raise
    return study.state

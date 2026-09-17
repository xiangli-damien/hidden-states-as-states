"""Full Llama-MATH analysis with restartable, CPU-only layer jobs.

App code is deployed through Git. Source captures are read-only. Geometry uses
5,000 responses; prediction refits everything on its own 40% training split.
"""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from hss.experiments.artifacts import lock, save_json, source_version
from hss.experiments.config import load
from hss.experiments.parallel import prefit_layers
from hss.experiments.runner import run_experiment
from hss.results import Result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.toml")
    p.add_argument("--directory", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument(
        "--stage",
        choices=["core", "prediction", "reliability", "monitoring"],
        default="core",
    )
    args = p.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = load(args.base)
    base.execution = replace(
        base.execution,
        output_root=str(root / "trials"),
        workers=args.workers,
        threads_per_worker=1,
        memory_gib=48,
        min_available_gib=32,
    )
    state_path = root / "study.json"
    with lock(root / "study.lock"):
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "schema_version": 1,
                "jobs": {},
                "source_version": source_version(),
                "scope": "Llama-3.2-1B-Instruct / MATH 5000. Single-model paper analyses and explicit method controls; not cross-model paper numbers.",
                "protocol": "Raw hidden states; full dimensions/layers; cosine Hungarian eta=0.6; GMM diagonal ICL K=2..80, 2% parsimony. MFA rank=8 at GMM-selected K is a matched-K control, not an MFA ICL optimum.",
            }
        )
        if state["source_version"] != source_version():
            raise ValueError("Scientific code changed; use a new study directory")
        state.update(
            status="running", stage=args.stage, pid=os.getpid(), updated_at=time.time()
        )
        save_json(state_path, state)

        def reference(name):
            return state["jobs"][name]["path"]

        def ready(cfg, snapshot):
            result = run_experiment(cfg, prepared=snapshot)
            audit = Result(result["path"]).validate(full=True)
            if not audit["valid"]:
                raise ValueError(audit)
            state["jobs"][cfg.name] = dict(
                status="complete",
                path=result["path"],
                trial_id=result["trial_id"],
                audit=audit,
            )
            state["updated_at"] = time.time()
            save_json(state_path, state)
            print(
                json.dumps(dict(completed=cfg.name, trial_id=result["trial_id"])),
                flush=True,
            )

        def run(configs):
            for cfg in configs:
                save_json(root / "configs" / f"{cfg.name}.json", cfg.to_dict())
            prefit_layers(
                configs, root / "progress", workers=args.workers, on_ready=ready
            )

        def matched(name, method, ref, cfg=base, **extra):
            return replace(
                cfg,
                name=name,
                cluster=replace(cfg.cluster, method=method),
                evaluation=replace(
                    cfg.evaluation,
                    fixed_k_map=str(Path(reference(ref)) / "selection.json"),
                ),
                **extra,
            )

        try:
            if args.stage == "core":
                run([replace(base, name="mean_gmm")])
                run(
                    [
                        matched("mean_kmeans", "kmeans", "mean_gmm"),
                        matched(
                            "mean_minibatch_kmeans", "minibatch_kmeans", "mean_gmm"
                        ),
                        replace(
                            base,
                            name="selected_kmeans",
                            cluster=replace(
                                base.cluster, method="kmeans", parsimony_tolerance=0.0
                            ),
                        ),
                        replace(
                            base,
                            name="selected_minibatch_kmeans",
                            cluster=replace(
                                base.cluster,
                                method="minibatch_kmeans",
                                parsimony_tolerance=0.0,
                            ),
                        ),
                        matched("mean_mfa", "mfa", "mean_gmm"),
                    ]
                )
            elif args.stage == "prediction":
                prompt = replace(
                    base, data=replace(base.data, representation="prompt_last")
                )
                pred = replace(
                    prompt, evaluation=replace(base.evaluation, mode="prediction")
                )
                run(
                    [
                        replace(prompt, name="prompt_gmm"),
                        replace(pred, name="prediction_gmm"),
                    ]
                )
                configs = []
                # Continuous probes see identical features and splits for every
                # cluster method; fit/report them once in prediction_gmm.
                state_pred = replace(
                    pred, evaluation=replace(pred.evaluation, methods=["HSS-NB"])
                )
                for method in ("kmeans", "minibatch_kmeans", "mfa"):
                    configs += [
                        matched(f"prompt_{method}", method, "prompt_gmm", prompt),
                        matched(
                            f"prediction_{method}", method, "prediction_gmm", state_pred
                        ),
                    ]
                run(configs)
            elif args.stage == "reliability":
                configs = []
                for method in ("gmm", "kmeans", "minibatch_kmeans", "mfa"):
                    for seed, fraction in [(43, 1.0), (44, 1.0), (42, 0.5), (42, 0.8)]:
                        cfg = matched(
                            f"stability_{method}_s{seed}_f{fraction}",
                            method,
                            "mean_gmm",
                            seed=seed,
                        )
                        cfg.evaluation.fit_fraction = fraction
                        configs.append(cfg)
                run(configs)
            else:
                # Reuse fixed complexity chosen ONLY from the prompt training split;
                # all prefix mixture/probe fitting still uses monitoring train groups.
                prefix = replace(
                    base,
                    data=replace(base.data, representation="prefix"),
                    evaluation=replace(base.evaluation, mode="monitoring"),
                )
                run(
                    [
                        matched(f"monitor_{m}", m, "prediction_gmm", prefix)
                        for m in ("gmm", "kmeans", "minibatch_kmeans", "mfa")
                    ]
                )
        except Exception as exc:
            state.update(status="failed", error=repr(exc), updated_at=time.time())
            save_json(state_path, state)
            raise
        state.update(status="stage_complete", stage=args.stage, updated_at=time.time())
        save_json(state_path, state)


if __name__ == "__main__":
    main()

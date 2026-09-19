"""Generate an explicit main-paper and construction-control experiment suite."""

from pathlib import Path
import json

from .artifacts import save_json, file_digest, lock
from .config import Experiment, load, set_value


MODELS = {
    "llama32": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
}
SIZES = {
    "math": 5000,
    "mmlu": 14042,
    "theoremqa": 800,
    "belebele_en": 900,
    "belebele_de": 900,
    "belebele_zh": 900,
}


def generate_suite(
    data_root,
    directory,
    cache_root,
    output_root,
    check_data=False,
    catalog=None,
    artifact_cache_root=None,
    overwrite=False,
):
    directory = Path(directory).expanduser().resolve()
    if (directory / "suite.json").exists() and not overwrite:
        raise ValueError(
            "Study already exists; run/resume its suite, choose a new directory, or explicitly use --overwrite"
        )
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    from ..data.catalog import load_catalog

    entries = load_catalog(catalog) if catalog else {}

    def add(
        name,
        alias,
        dataset,
        mode="geometry",
        representation="mean",
        grid=None,
        overrides=None,
        dependencies=None,
    ):
        payload = Experiment().to_dict()
        payload.update(name=name, grid=grid or {})
        payload["data"].update(
            paths=[str(Path(data_root) / f"{dataset}_full_20260916" / alias)],
            expected_model=MODELS[alias],
            expected_samples=SIZES.get(dataset),
            representation=representation,
            dataset_id=dataset,
        )
        payload["evaluation"]["mode"] = mode
        payload["execution"].update(
            cache_root=str(cache_root),
            artifact_cache_root=str(
                artifact_cache_root or Path(output_root).parent / "fit_cache"
            ),
            output_root=str(output_root),
            workers=2,
            memory_gib=32.0,
        )
        for k, v in (overrides or {}).items():
            set_value(payload, k, v)
        if catalog:
            key = f"{alias}/{dataset}"
            if key in entries:
                payload["data"].update(entries[key])
            else:
                payload["data"]["paths"] = [
                    str(directory / "unconfigured_data" / alias / dataset)
                ]
            payload["data"]["representation"] = representation
        cfg = Experiment.from_dict(payload)
        config_path = directory / (name + ".json")
        save_json(config_path, cfg.to_dict())
        status = "not_checked"
        if check_data:
            from ..data.catalog import inspect_spec

            inspection = inspect_spec(cfg.data)
            status = inspection["status"]
            if status == "incomplete":
                status += f":{inspection['published_samples']}/{inspection['expected_samples']}"
        jobs.append(
            {
                "name": name,
                "config": config_path.name,
                "depends_on": dependencies or {},
                "data_status": status,
            }
        )

    add("default_map", "llama32", "math")
    add(
        "qwen_math_map",
        "qwen2",
        "math",
        overrides={
            "evaluation.diagnostics": ["dispersion", "separation", "effective_rank"]
        },
    )
    for model in ("llama32", "qwen2", "llama3"):
        for dataset in ("math", "mmlu", "belebele_en", "belebele_de", "belebele_zh"):
            if (model, dataset) in (("llama32", "math"), ("qwen2", "math")):
                continue
            add(f"geometry_{model}_{dataset}", model, dataset)
    for model in ("llama32", "qwen2", "llama3"):
        add(
            f"geometry_{model}_belebele_pooled",
            model,
            "belebele_pooled",
            overrides={
                "data.paths": [
                    str(Path(data_root) / f"belebele_{lang}_full_20260916" / model)
                    for lang in ("en", "de", "zh")
                ],
                "data.expected_samples": 2700,
            },
        )
    for model in ("qwen2", "llama3"):
        for dataset in ("math", "mmlu", "theoremqa"):
            add(
                f"prediction_{model}_{dataset}",
                model,
                dataset,
                "prediction",
                "prompt_last",
            )
    # The supplied manuscript uses these two benchmarks. WildJailbreak is not
    # silently substituted; it is available as a separately named extension.
    for dataset in ("jailbreakbench", "harmbench"):
        add(
            f"prediction_llama2_{dataset}",
            "llama2",
            dataset,
            "prediction",
            "prompt_last",
            overrides={"data.label": "is_safe", "data.label_file": "safety"},
        )
    add("sentence_monitoring", "qwen2", "math", "monitoring", "prefix")
    add(
        "characterization",
        "qwen2",
        "math",
        overrides={
            "evaluation.diagnostics": ["dispersion", "separation", "effective_rank"]
        },
    )
    add("qwen_prompt_map", "qwen2", "math", representation="prompt_last")
    add(
        "monitoring_far_scope_control",
        "qwen2",
        "math",
        "monitoring",
        "prefix",
        grid={"evaluation.far_scope": ["all_boundaries", "nonfinal"]},
    )
    add(
        "reliability_trends",
        "qwen2",
        "math",
        grid={
            "variants": [{"seed": s} for s in (42, 43, 44, 45, 46)]
            + [{"evaluation.fit_fraction": f} for f in (0.3, 0.5, 0.7, 0.9)]
            + [{"cluster.k_max": k} for k in (20, 40, 60)]
        },
    )
    add(
        "reliability_centers",
        "qwen2",
        "math",
        grid={
            "variants": [{"seed": s} for s in (43, 44, 45, 46)]
            + [{"evaluation.fit_fraction": f} for f in (0.3, 0.5, 0.7, 0.9)]
        },
        dependencies={
            "evaluation.fixed_k_map": {"job": "qwen_math_map", "file": "selection.json"}
        },
    )
    add(
        "preprocessing_control",
        "qwen2",
        "math",
        grid={"transform.standardize": [False, True], "seed": [42, 43, 44, 45, 46]},
    )
    add(
        "pca_control",
        "qwen2",
        "math",
        grid={"transform.pca_components": [None, 64, 128, 256, 512]},
    )
    add(
        "alignment_control",
        "qwen2",
        "math",
        grid={
            "alignment.threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "alignment.similarity": ["cosine", "euclidean"],
        },
    )
    add(
        "kmeans_control",
        "qwen2",
        "math",
        overrides={"cluster.method": "minibatch_kmeans"},
    )
    add(
        "global_clustering_control",
        "qwen2",
        "math",
        "global_control",
        overrides={"execution.memory_gib": 96.0, "execution.workers": 1},
    )
    add(
        "granularity_control",
        "qwen2",
        "math",
        grid={"data.representation": ["mean", "prompt_last", "sentence", "prefix"]},
    )
    add("rms_control", "qwen2", "math", grid={"data.final_norm": ["pre", "post"]})
    add(
        "mfa_control",
        "qwen2",
        "math",
        grid={"cluster.rank": [4, 8, 16, 32]},
        overrides={"cluster.method": "mfa"},
    )
    # The supplementary controls also assess downstream HSS-NB prediction.
    for name, grid in {
        "preprocessing_prediction_control": {"transform.standardize": [False, True]},
        "pca_prediction_control": {
            "transform.pca_components": [None, 64, 128, 256, 512]
        },
        "alignment_prediction_control": {
            "alignment.threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "alignment.similarity": ["cosine", "euclidean"],
        },
        "method_prediction_control": {
            "cluster.method": ["gmm", "minibatch_kmeans", "mfa"]
        },
        "seed_prediction_control": {"seed": [42, 43, 44, 45, 46]},
    }.items():
        add(
            name,
            "qwen2",
            "math",
            "prediction",
            "prompt_last",
            grid=grid,
            overrides={"evaluation.methods": ["HSS-NB"]},
        )
    # Token geometry is deliberately a separate expensive job, not hidden in a
    # default Cartesian product. Planning shows its full memory/storage cost.
    add(
        "token_granularity_control",
        "qwen2",
        "math",
        representation="tokens",
        overrides={"execution.workers": 1, "execution.memory_gib": 96.0},
    )
    suite = {
        "schema_version": 2,
        "dataset_catalog": str(Path(catalog).resolve()) if catalog else None,
        "jobs": jobs,
        "protocol_notes": [
            "Figures 1-2 are illustrative schematics; Figures 3-12 and Tables 1-3 have explicit recipes in hss.viz.paper.",
            "FAR: main-text default all_boundaries; appendix nonfinal is a separate control. Early detection always excludes final boundary.",
            "Reliability text uses 30-90% data; figure legend differs. Generated fractions are explicit 0.3,0.5,0.7,0.9; seeds 42-46; Kmax 20,40,60,80.",
            "Hamming heatmaps use average linkage by default; Ward requires Euclidean distances. Rendering records deterministic sampled trajectory IDs (default max 1000).",
            "Euclidean matching similarity is 1/(1+distance); numeric eta values have a different meaning from cosine similarity.",
            "KMeans controls use MiniBatchKMeans; ordinary KMeans remains an optional method.",
            "No reported paper scores are embedded; outputs are computed from supplied data.",
            "Main default: raw representations, diagonal GMM, ICL=BIC+2entropy, cosine Hungarian eta=.6.",
            "Monitoring validation fraction, sentence segmentation, mixture regularization/tolerance and some probe details are not fully specified by the manuscript; explicit implementation choices are recorded in configs.",
            "Current OpenAct collection has only response-aggregate entropy/perplexity; prefix entropy/logprob baselines need aligned per-token metrics and otherwise report unavailable.",
            "The manuscript names JailbreakBench/HarmBench; current WildJailbreak collection cannot substitute for exact safety-table reproduction.",
            "MFA and pre/post RMS comparisons are extensions, not claims of numerical paper reproduction.",
            "Full MMLU=14042 and pooled BELEBELE=2700 follow user choices; manuscript N=14000 and N=2100 require an unspecified subset for exact reproduction.",
            "Continuous baselines default to last_layer as in legacy code; all_layers is configurable. Exact probe layer selection is unspecified in the manuscript.",
        ],
    }
    save_json(directory / "suite.json", suite)
    return {
        "suite": str(directory / "suite.json"),
        "experiments": len(jobs),
        "data_status": {job["name"]: job["data_status"] for job in jobs},
    }


def run_suite(path, only=None, overrides=(), refresh_data=False):
    from .sweep import run_sweep

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    jobs = {j["name"]: j for j in manifest["jobs"]}
    selected = set(only or jobs)
    if selected - set(jobs):
        raise ValueError(f"Unknown suite jobs: {selected - set(jobs)}")

    def dependencies(name, visited):
        if name in visited:
            raise ValueError("Cyclic suite dependencies")
        for d in jobs[name]["depends_on"].values():
            if d["job"] not in jobs:
                raise ValueError(f"Missing dependency {d['job']}")
            dependencies(d["job"], visited | {name})
            selected.add(d["job"])

    for name in list(selected):
        dependencies(name, set())
    results = {}
    pending = [name for name in jobs if name in selected]
    while pending:
        for name in list(pending):
            job = jobs[name]
            if any(d["job"] not in results for d in job["depends_on"].values()):
                continue
            pending.remove(name)
            try:
                cfg = load(path.parent / job["config"], overrides)
                payload = cfg.to_dict()
                for key, dependency in job["depends_on"].items():
                    prior = results[dependency["job"]]
                    if prior["status"] != "complete":
                        raise ValueError(
                            "Prerequisite experiment failed or is unavailable"
                        )
                    if len(prior["results"]) != 1:
                        raise ValueError(
                            "A fixed-map dependency must resolve to one result"
                        )
                    set_value(
                        payload,
                        key,
                        str(Path(prior["results"][0]["path"]) / dependency["file"]),
                    )
                results[name] = run_sweep(
                    Experiment.from_dict(payload), refresh_data=refresh_data
                )
            except Exception as exc:
                results[name] = {"status": "failed", "error": repr(exc)}
            results[name]["source_config_sha256"] = (
                file_digest(path.parent / job["config"])
                if (path.parent / job["config"]).exists()
                else None
            )
            # Separate invocations may run different jobs. Preserve their outcomes,
            # including under concurrent writers, rather than replacing the catalog.
            with lock(path.parent / ".suite_status.lock"):
                status_path = path.parent / "suite_status.json"
                history = (
                    json.loads(status_path.read_text()) if status_path.exists() else {}
                )
                save_json(status_path, {**history, name: results[name]})
    return {
        "status": "complete"
        if all(r["status"] == "complete" for r in results.values())
        else "failed",
        "jobs": results,
    }

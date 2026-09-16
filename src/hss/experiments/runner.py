"""One reproducible experiment: split -> fit on train -> freeze -> evaluate."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ..align import align_layers
from ..types import AlignSpec
from ..cluster.registry import rebuild_model
from .artifacts import (
    digest,
    file_digest,
    lock,
    save_json,
    save_npz,
    source_version,
    runtime_versions,
)
from .evaluate import (
    CountNB,
    binary_metrics,
    calibrate_threshold,
    characterize,
    grouped_split,
    monitoring_metrics,
)
from .fitting import Projection, assign, fit_candidates, fit_transform
from .openact import CachedStates, prepare
from .diagnostics import layer_diagnostics
from .resources import model_bytes, candidates


def _rows_digest(indices):
    import hashlib

    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def estimate_memory_gib(data, cfg):
    n, d, l = data.n_items(), data.state_dim(), len(data.layers())
    feature = n * d * 8
    # Fit input, projections, sufficient stats, BLAS temporaries, runtime overhead.
    estimate = (
        feature * 8
        + model_bytes(cfg, cfg.transform.pca_components or d, max(candidates(cfg))) * 8
    )
    if cfg.evaluation.continuous_features == "all_layers" and cfg.evaluation.mode in (
        "prediction",
        "monitoring",
    ):
        estimate += feature * l * 4
    if cfg.evaluation.mode == "global_control":
        estimate += feature * l * 8
    return max(0.5, estimate / 1024**3 + 0.5)


def trial_identity(cfg, data, version):
    p = cfg.to_dict()
    references = {}
    if cfg.evaluation.fixed_k_map:
        references["fixed_k"] = file_digest(cfg.evaluation.fixed_k_map)
    if cfg.evaluation.fixed_map:
        folder = Path(cfg.evaluation.fixed_map)
        references["fixed_map"] = {
            str(p.relative_to(folder)): file_digest(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
            and (
                p.suffix == ".npz"
                or p.name in ("_SUCCESS.json", "config.json", "data_snapshot.json")
            )
        }
    return {
        "source_version": version,
        "references": references,
        "data": data.info["key"],
        **{
            k: p[k] for k in ("seed", "cluster", "transform", "alignment", "evaluation")
        },
    }


def _save_layer(root, layer, model, projection, scan):
    path = root / "models" / f"layer_{layer}"
    save_npz(path / "model.npz", **model.state_arrays())
    save_npz(path / "projection.npz", **projection.arrays())
    save_json(path / "model.json", model.config())
    save_json(path / "selection.json", scan)


def _load_layer(root, layer):
    path = Path(root) / "models" / f"layer_{layer}"
    config = json.loads((path / "model.json").read_text())
    with np.load(path / "model.npz", allow_pickle=False) as f:
        model = rebuild_model(config, {k: f[k] for k in f.files})
    with np.load(path / "projection.npz", allow_pickle=False) as f:
        projection = Projection(**{k: f[k] for k in f.files})
    return model, projection, json.loads((path / "selection.json").read_text())


def _continuous(data, indices, cfg):
    layers = (
        data.layers()
        if cfg.evaluation.continuous_features == "all_layers"
        else [data.layers()[-1]]
    )
    return np.concatenate([data.array(layer)[indices] for layer in layers], axis=1)


def _classifier(method, cfg):
    e = cfg.evaluation
    return {
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "GaussianNB": lambda: GaussianNB(),
        "Logistic": lambda: LogisticRegression(
            C=e.logistic_c, max_iter=e.max_iter, random_state=cfg.seed
        ),
        "LinearSVM": lambda: LinearSVC(
            C=e.svm_c, dual="auto", max_iter=e.max_iter, random_state=cfg.seed
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=tuple(e.mlp_hidden),
            max_iter=e.max_iter,
            random_state=cfg.seed,
        ),
    }[method]()


def _supervised(root, cfg, data, states, vocab, meta, split):
    y = meta.label.to_numpy(dtype=int)
    train, val, test = (split[k] for k in ("train", "validation", "test"))
    models = (
        cfg.evaluation.methods
        if cfg.evaluation.mode == "prediction"
        else ["HSS-NB", "Logistic"]
    )
    summary, scores = [], []
    for method in models:
        if method == "HSS-NB":
            estimator = CountNB(vocab, cfg.evaluation.alpha).fit(
                states[train], y[train]
            )
            pred = {
                key: estimator.predict_proba(states[idx])[:, 1]
                for key, idx in split.items()
                if len(idx)
            }
        elif method == "HSS-Markov":
            from ..predict import MarkovClassifier

            estimator = MarkovClassifier(alpha=cfg.evaluation.alpha).fit(
                states[train], y[train]
            )
            pred = {
                key: estimator.predict_proba(states[idx])[:, 1]
                for key, idx in split.items()
                if len(idx)
            }
        else:
            X = _continuous(data, train, cfg)
            scaler = (
                StandardScaler().fit(X)
                if cfg.evaluation.continuous_standardize
                else None
            )
            estimator = _classifier(method, cfg)
            estimator.fit(scaler.transform(X) if scaler else X, y[train])
            del X
            pred = {}
            for key, idx in split.items():
                if not len(idx):
                    continue
                X = _continuous(data, idx, cfg)
                if scaler:
                    X = scaler.transform(X)
                pred[key] = (
                    estimator.predict_proba(X)[:, 1]
                    if hasattr(estimator, "predict_proba")
                    else estimator.decision_function(X)
                )
                del X
        if cfg.evaluation.mode == "prediction":
            threshold = 0.0 if method == "LinearSVM" else 0.5
            summary.append(
                {
                    "method": method,
                    **binary_metrics(y[test], pred["test"], threshold=threshold),
                }
            )
            for idx, score in zip(test, pred["test"]):
                scores.append(
                    {
                        "row": int(idx),
                        "sample_id": str(meta.iloc[idx].sample_id),
                        "method": method,
                        "label": int(y[idx]),
                        "score": float(score),
                    }
                )
        else:
            danger_val, danger_test = 1 - pred["validation"], 1 - pred["test"]
            threshold, val_far = calibrate_threshold(
                meta.iloc[val], danger_val, cfg.evaluation.far_target
            )
            metrics, cases = monitoring_metrics(meta.iloc[test], danger_test, threshold)
            summary.append({"method": method, "validation_far": val_far, **metrics})
            cases.to_parquet(root / f"monitor_{method}.parquet", index=False)
            for idx, score in zip(test, danger_test):
                scores.append(
                    {
                        "row": int(idx),
                        "sample_id": str(meta.iloc[idx].sample_id),
                        "method": method,
                        "label": int(y[idx]),
                        "score": float(score),
                    }
                )
    missing = []
    if cfg.evaluation.mode == "monitoring":
        for name, field, sign in [
            ("Prefix entropy", "prefix_entropy", 1),
            ("Prefix logprob", "prefix_logprob", -1),
        ]:
            if field not in meta:
                missing.append(
                    {
                        "method": name,
                        "reason": "Per-token output metric was not captured. Full-response aggregates would leak future information.",
                    }
                )
                continue
            danger = sign * meta[field].to_numpy()
            threshold, val_far = calibrate_threshold(
                meta.iloc[val], danger[val], cfg.evaluation.far_target
            )
            metrics, cases = monitoring_metrics(
                meta.iloc[test], danger[test], threshold
            )
            summary.append({"method": name, "validation_far": val_far, **metrics})
            cases.to_parquet(root / f"monitor_{field}.parquet", index=False)
    pd.DataFrame(scores).to_parquet(root / "predictions.parquet", index=False)
    save_json(
        root / "evaluation.json", {"metrics": summary, "unavailable_baselines": missing}
    )
    return summary, missing


def run_experiment(cfg, *, prepared=None, version=None):
    cfg.validate()
    version = version or source_version()
    data = (
        CachedStates(prepared)
        if prepared
        else prepare(cfg.data, cfg.execution.cache_root)
    )
    if data.info["identity"]["spec"] != asdict(cfg.data):
        raise ValueError("Cached snapshot selection does not match data configuration")
    identity = trial_identity(cfg, data, version)
    key = digest(identity)
    root = Path(cfg.execution.output_root).expanduser().resolve() / key
    root.mkdir(parents=True, exist_ok=True)
    with (
        lock(root / ".lock"),
        threadpool_limits(limits=cfg.execution.threads_per_worker),
    ):
        success = root / "_SUCCESS.json"
        if success.exists():
            result = json.loads(success.read_text())
            for required in (
                "states.npy",
                "rows.parquet",
                "config.json",
                "summary.json",
            ):
                if not (root / required).exists():
                    raise ValueError(f"Completed result missing {required}")
            return {**result, "cache_hit": True}
        started = time.perf_counter()
        save_json(
            root / "status.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "host": platform.node(),
                "identity": identity,
            },
        )
        save_json(root / "config.json", cfg.to_dict())
        save_json(root / "data_snapshot.json", data.info)
        try:
            artifact_cache = (
                cfg.execution.artifact_cache_root or cfg.execution.cache_root
            )
            if estimate_memory_gib(data, cfg) > cfg.execution.memory_gib:
                raise MemoryError(
                    "Estimated trial RAM exceeds execution.memory_gib; reduce concurrency/features or raise explicit budget"
                )
            import psutil

            available = psutil.virtual_memory().available / 1024**3
            if (
                estimate_memory_gib(data, cfg)
                > available - cfg.execution.min_available_gib
            ):
                raise MemoryError("Insufficient available RAM after configured reserve")
            meta = data.meta.copy()
            if cfg.evaluation.positive_label == 0:
                meta["label"] = 1 - meta.label
            if cfg.evaluation.mode in ("prediction", "monitoring"):
                split = grouped_split(meta, cfg.evaluation)
                train = split["train"]
            else:
                split = {
                    "train": np.arange(len(meta)),
                    "validation": np.array([], dtype=int),
                    "test": np.array([], dtype=int),
                }
                train = split["train"]
            if cfg.evaluation.fit_fraction < 1:
                ids = np.unique(meta.iloc[train].group_id)
                chosen = np.random.default_rng(cfg.seed).choice(
                    ids,
                    max(2, int(len(ids) * cfg.evaluation.fit_fraction)),
                    replace=False,
                )
                train = train[np.isin(meta.iloc[train].group_id, chosen)]
            save_npz(root / "split.npz", **split, map_fit=train)
            meta.to_parquet(root / "rows.parquet", index=False)
            layers = data.layers()
            local = np.empty((len(meta), len(layers)), dtype=np.int32)
            centers, scans = [], []
            diagnostics = []
            fixed = cfg.evaluation.fixed_map
            fixed_k = {}
            if cfg.evaluation.fixed_k_map:
                fixed_k = {
                    s["layer"]: s["selected"]["k"]
                    for s in json.loads(Path(cfg.evaluation.fixed_k_map).read_text())
                }
                if set(fixed_k) != set(layers):
                    raise ValueError(
                        "Fixed-k reference must cover exactly the selected layers"
                    )
            if fixed:
                original = json.loads((Path(fixed) / "config.json").read_text())
                original_data = json.loads(
                    (Path(fixed) / "data_snapshot.json").read_text()
                )
                if (
                    not (Path(fixed) / "_SUCCESS.json").exists()
                    or original_data["model"] != data.info["model"]
                    or original_data["layers"] != layers
                    or any(
                        original["data"][k] != asdict(cfg.data)[k]
                        for k in ("representation", "final_norm")
                    )
                    or original["alignment"] != asdict(cfg.alignment)
                    or original["cluster"] != asdict(cfg.cluster)
                    or original["transform"] != asdict(cfg.transform)
                ):
                    raise ValueError(
                        "Frozen map is incomplete or incompatible with model, layers, representation or alignment"
                    )
            if cfg.evaluation.mode == "global_control":
                X = np.concatenate([data.array(layer) for layer in layers], axis=0)
                fit_rows = np.concatenate(
                    [train + j * len(meta) for j in range(len(layers))]
                )
                context = {
                    "version": version,
                    "snapshot": data.info["key"],
                    "layer": -1,
                    "train": _rows_digest(fit_rows),
                }
                projection, fitted_X, tk = fit_transform(
                    X, fit_rows, cfg, context, artifact_cache
                )
                model, scan = fit_candidates(
                    fitted_X, cfg, {**context, "transform_key": tk}, artifact_cache
                )
                del fitted_X
                for j, layer in enumerate(layers):
                    for start in range(0, len(meta), cfg.cluster.chunk_size):
                        stop = min(start + cfg.cluster.chunk_size, len(meta))
                        local[start:stop, j] = assign(
                            model,
                            projection.transform(data.array(layer)[start:stop]),
                            cfg.cluster.assignment,
                        )
                    scans.append({"layer": layer, **scan})
                    _save_layer(root, layer, model, projection, scan)
                vocab = [np.arange(model.n_clusters()) for _ in layers]
                states = local.copy()
                stacked = pd.DataFrame(
                    {"state": states.T.ravel(), "layer": np.repeat(layers, len(meta))}
                )
                purity = pd.crosstab(stacked.state, stacked.layer)
                (purity.max(axis=1) / purity.sum(axis=1)).rename("layer_purity").to_csv(
                    root / "global_layer_purity.csv"
                )
                del X
            else:
                for j, layer in enumerate(layers):
                    X = data.array(layer)
                    if fixed:
                        model, projection, scan = _load_layer(fixed, layer)
                    else:
                        context = {
                            "version": version,
                            "snapshot": data.info["key"],
                            "layer": layer,
                            "train": _rows_digest(train),
                        }
                        projection, fitted_X, tk = fit_transform(
                            X, train, cfg, context, artifact_cache
                        )
                        layer_cfg = (
                            replace(cfg, cluster=replace(cfg.cluster, k=fixed_k[layer]))
                            if fixed_k
                            else cfg
                        )
                        model, scan = fit_candidates(
                            fitted_X,
                            layer_cfg,
                            {**context, "transform_key": tk},
                            artifact_cache,
                        )
                        del fitted_X
                    diagnostics.append(
                        {
                            "layer": layer,
                            **layer_diagnostics(
                                X[train],
                                model,
                                projection,
                                cfg.cluster.assignment,
                                cfg.evaluation.diagnostics,
                            ),
                        }
                    )
                    for start in range(0, len(meta), cfg.cluster.chunk_size):
                        stop = min(start + cfg.cluster.chunk_size, len(meta))
                        local[start:stop, j] = assign(
                            model,
                            projection.transform(X[start:stop]),
                            cfg.cluster.assignment,
                        )
                    centers.append(projection.inverse(model.centers()))
                    scans.append({"layer": layer, **scan})
                    _save_layer(root, layer, model, projection, scan)
                    save_json(
                        root / "status.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "completed_layers": j + 1,
                            "total_layers": len(layers),
                            "layer": layer,
                        },
                    )
                alignment = align_layers(
                    centers,
                    layers=layers,
                    spec=AlignSpec(**asdict(cfg.alignment), allow_negative=True),
                )
                vocab = alignment.local_to_global
                states = np.column_stack(
                    [vocab[j][local[:, j]] for j in range(len(layers))]
                )
            save_json(
                root / "alignment.json",
                {"layers": layers, "local_to_global": [v.tolist() for v in vocab]},
            )
            np.save(root / "states.npy", states, allow_pickle=False)
            save_json(root / "selection.json", scans)
            save_json(root / "diagnostics.json", diagnostics)
            associations, tags, transitions = characterize(states, meta, layers)
            associations.to_csv(root / "associations.csv", index=False)
            tags.to_csv(root / "state_tags.csv", index=False)
            transitions.to_csv(root / "transitions.csv", index=False)
            evaluation, unavailable = [], []
            if cfg.evaluation.mode in ("prediction", "monitoring"):
                evaluation, unavailable = _supervised(
                    root, cfg, data, states, vocab, meta, split
                )
            profile = [
                {
                    "layer": layer,
                    "relative_depth": j / max(1, len(layers) - 1),
                    "k": scan["selected"]["k"],
                    "criterion": scan["selected"].get("criterion"),
                    "criterion_name": "negative_silhouette"
                    if cfg.cluster.method == "kmeans"
                    else "icl",
                    "self_transition": float((states[:, j] == states[:, j - 1]).mean())
                    if j
                    else None,
                }
                for j, (layer, scan) in enumerate(zip(layers, scans))
            ]
            summary = {
                "name": cfg.name,
                "trial_id": key,
                "path": str(root),
                "snapshot": data.info["key"],
                "n_samples": data.info["n_samples"],
                "n_rows": len(meta),
                "model": data.info["model"],
                "profile": profile,
                "n_global_states": int(states.max()) + 1,
                "evaluation": evaluation,
                "unavailable_baselines": unavailable,
                "seconds": time.perf_counter() - started,
                "estimated_ram_gib": estimate_memory_gib(data, cfg),
                "source_version": version,
                "runtime": runtime_versions(),
                "fixed_map_origin": fixed,
            }
            save_json(root / "summary.json", summary)
            save_json(root / "status.json", {"status": "complete", "trial_id": key})
            save_json(success, summary)
            return {**summary, "cache_hit": False}
        except Exception as exc:
            save_json(
                root / "status.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

"""Reusable per-layer transform/mixture fits for many downstream grid trials."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time
import os

import numpy as np
from scipy.special import xlogy
from sklearn.metrics import silhouette_score

from ..cluster.registry import rebuild_model
from ..cluster.methods import fit_method
from ..cluster.assignment import assign as assign
from .artifacts import digest, lock, save_json, save_npz
from ..transform.projection import Projection


def fit_transform(X, train, cfg, context, cache_root):
    key = digest(
        {
            **context,
            "transform": asdict(cfg.transform),
            "pca_seed": cfg.evaluation.split_seed,
        }
    )
    path = Path(cache_root) / "transforms" / key
    with lock(path.with_suffix(".lock")):
        if (path / "projection.npz").exists():
            with np.load(path / "projection.npz", allow_pickle=False) as f:
                projection = Projection(**{k: f[k] for k in f.files})
        else:
            projection = Projection.fit(
                X[train], cfg.transform, cfg.evaluation.split_seed
            )
            save_npz(path / "projection.npz", **projection.arrays())
    return projection, projection.transform(X[train]), key


def common_parameters(config):
    p = {
        k: getattr(config, k)
        for k in (
            "reg_covar",
            "n_init",
            "max_iter",
            "tol",
            "chunk_size",
            "backend",
            "device",
        )
    }
    if config.method == "mfa":
        p["rank"] = config.rank
        p["init_method"] = config.mfa_init
    elif config.method == "gmm":
        p.update(
            {
                k: getattr(config, k)
                for k in (
                    "covariance_type",
                    "adaptive_reg",
                    "adaptive_alpha",
                    "adaptive_min",
                    "init_method",
                )
            }
        )
    else:
        p = {"n_init": config.n_init, "max_iter": config.max_iter, "tol": config.tol}
    if config.method == "minibatch_kmeans":
        p["batch_size"] = config.batch_size
    return p


@contextmanager
def gpu_slot(cfg, needed_gib):
    if cfg.cluster.backend != "gpu":
        yield
        return
    # One fitting job per CUDA device across this machine, including separate sweeps.
    import torch

    name = cfg.cluster.device.replace(":", "_").replace("/", "_")
    with lock(Path("/tmp") / f"hss-gpu-{name}.lock"):
        free, _ = torch.cuda.mem_get_info(cfg.cluster.device)
        if free / 1024**3 < needed_gib + cfg.execution.min_gpu_free_gib:
            raise MemoryError(
                "Insufficient free GPU memory after reserve; retry when collection frees memory"
            )
        try:
            yield
        finally:
            torch.cuda.empty_cache()


def selection_metrics(model, X, method, chunk_size, seed):
    if method in ("kmeans", "minibatch_kmeans"):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), min(len(X), 2000), replace=False))
        labels = model.predict(X[idx])
        value = (
            float(silhouette_score(X[idx], labels))
            if 1 < len(np.unique(labels)) < len(idx)
            else -1.0
        )
        return {"criterion": -value, "silhouette": value, "bic": None, "entropy": None}
    loglik, entropy = 0.0, 0.0
    for start in range(0, len(X), chunk_size):
        block = X[start : start + chunk_size]
        loglik += float(model.score_samples(block).sum())
        p = model.predict_proba(block)
        entropy -= float(xlogy(p, p).sum())
    bic = -2 * loglik + model._n_parameters() * np.log(len(X))
    return {
        "criterion": float(bic + 2 * entropy),
        "icl": float(bic + 2 * entropy),
        "bic": float(bic),
        "entropy": float(entropy),
        "log_likelihood": float(loglik),
        "n_parameters": int(model._n_parameters()),
    }


def load_fitted(path):
    path = Path(path)
    info = json.loads((path / "fit.json").read_text())
    with np.load(path / "model.npz", allow_pickle=False) as f:
        return rebuild_model(info["model"], {k: f[k] for k in f.files}), info


def _fit_mfa_resumable(X, k, seed, params, path):
    from ..cluster.mfa import fit_mfa

    options = dict(params)
    restarts = options.pop("n_init")
    best = None
    audit = []
    for restart in range(restarts):
        saved = path / f"restart_{restart:02d}.npz"
        initial = None
        if saved.exists():
            with np.load(saved, allow_pickle=False) as z:
                initial = rebuild_model(
                    json.loads(str(z["config_json"])),
                    {name: z[name] for name in z.files if name != "config_json"},
                )

        def checkpoint(model, _):
            save_npz(
                saved,
                config_json=np.array(json.dumps(model.config())),
                **model.state_arrays(),
            )
            save_json(
                path / "progress.json",
                dict(
                    status="fitting",
                    pid=os.getpid(),
                    restart=restart,
                    restarts=restarts,
                    n_iter=len(model.history_),
                    average_log_likelihood=model.history_[-1],
                    last_delta=(
                        model.history_[-1] - model.history_[-2]
                        if len(model.history_) > 1
                        else None
                    ),
                    converged=model.converged_,
                    updated_at=time.time(),
                ),
            )

        model = fit_mfa(
            X,
            k,
            seed + restart,
            n_init=1,
            initial_model=initial,
            checkpoint=checkpoint,
            **options,
        )
        audit.append(
            dict(
                restart=restart,
                seed=seed + restart,
                converged=model.converged_,
                n_iter=len(model.history_),
                average_log_likelihood=model.history_[-1],
            )
        )
        if best is None or (model.converged_, model.history_[-1]) > (
            best.converged_,
            best.history_[-1],
        ):
            best = model
    save_json(path / "restarts.json", audit)
    return best


def fit_one_candidate(X, cfg, context, cache_root, k):
    """Persist one independent fit, including unsuccessful convergence, for audit."""
    c = cfg.cluster
    params = common_parameters(c)
    seed = cfg.seed + 1009 * max(0, int(context.get("layer", 0)))
    key = digest(
        {**context, "method": c.method, "params": params, "k": k, "seed": seed}
    )
    path = Path(cache_root) / "fits" / key
    with lock(path.with_suffix(".lock")):
        model = None
        if not (path / "fit.json").exists():
            start = time.perf_counter()
            with gpu_slot(
                cfg,
                max(
                    0.25, (X.nbytes * 8 + k * X.shape[1] * (c.rank + 1) * 128) / 1024**3
                ),
            ):
                if c.method == "mfa":
                    model = _fit_mfa_resumable(X, k, seed, params, path)
                else:
                    model = fit_method(c.method, X, k, seed, params)
            scores = selection_metrics(model, X, c.method, c.chunk_size, seed)
            if not np.isfinite(scores["criterion"]):
                raise FloatingPointError("Nonfinite model-selection criterion")
            save_npz(path / "model.npz", **model.state_arrays())
            save_json(
                path / "fit.json",
                dict(
                    key=key,
                    k=k,
                    model=model.config(),
                    scores=scores,
                    seconds=time.perf_counter() - start,
                    context=context,
                ),
            )
        record = json.loads((path / "fit.json").read_text())
        if c.save_candidate_assignments and not (path / "assignments.npz").exists():
            model = model or load_fitted(path)[0]
            labels = {name: [] for name in ("nearest", "posterior")}
            for start in range(0, len(X), c.chunk_size):
                block = X[start : start + c.chunk_size]
                for name in labels:
                    labels[name].append(assign(model, block, name))
            save_npz(
                path / "assignments.npz",
                **{name: np.concatenate(v) for name, v in labels.items()},
            )
        save_json(
            path / "progress.json",
            dict(
                status="complete",
                converged=record["model"].get("converged"),
                n_iter=record["model"].get(
                    "n_iter", len(record["model"].get("history", []))
                ),
                updated_at=time.time(),
            ),
        )
        return dict(
            k=k,
            fit_path=str(path),
            **record["scores"],
            converged=record["model"].get("converged"),
            seconds=record["seconds"],
        )


def select_candidate(results, config):
    """Select only admissible fits; selection policies never change fit identity."""
    eligible = [
        dict(r)
        for r in results
        if not config.require_convergence or r.get("converged") is True
    ]
    if not eligible:
        raise ValueError("No converged candidate is eligible for selection")
    if config.method in ("gmm", "mfa"):
        for record in eligible:
            record["criterion"] = (
                record.get("icl", record["criterion"])
                if config.selection_criterion == "icl"
                else record["bic"]
            )
    best = min(r["criterion"] for r in eligible)
    return min(
        (
            r
            for r in eligible
            if r["criterion"] <= best + config.parsimony_tolerance * max(abs(best), 1.0)
        ),
        key=lambda r: r["k"],
    )


def fit_candidates(X, cfg, context, cache_root):
    c = cfg.cluster
    candidates = (
        [c.k] if c.k is not None else (c.k_values or list(range(c.k_min, c.k_max + 1)))
    )
    candidates = sorted(set(int(k) for k in candidates))
    if c.k is not None and c.k > len(X):
        raise ValueError("Fixed k exceeds training size")
    requested = candidates
    candidates = [
        k
        for k in candidates
        if k <= (len(X) if c.k is not None else max(1, len(X) - 1))
    ]
    if not candidates:
        raise ValueError("No eligible k for the training split")
    results = []
    for k in candidates:
        results.append(fit_one_candidate(X, cfg, context, cache_root, k))
    selected = select_candidate(results, c)
    model, _ = load_fitted(selected["fit_path"])
    return model, dict(
        selected=selected,
        candidates=results,
        requested_k=requested,
        excluded_unconverged=(
            [r["k"] for r in results if r.get("converged") is not True]
            if c.require_convergence
            else []
        ),
        selection_criterion=(
            c.selection_criterion
            if c.method in ("gmm", "mfa")
            else "negative_silhouette"
        ),
        omitted_k=[k for k in requested if k not in candidates],
    )

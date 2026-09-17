"""Reusable per-layer transform/mixture fits for many downstream grid trials."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time

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
        "bic": float(bic),
        "entropy": float(entropy),
        "log_likelihood": float(loglik),
    }


def load_fitted(path):
    path = Path(path)
    info = json.loads((path / "fit.json").read_text())
    with np.load(path / "model.npz", allow_pickle=False) as f:
        return rebuild_model(info["model"], {k: f[k] for k in f.files}), info


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
    params = common_parameters(c)
    for k in candidates:
        seed = cfg.seed + 1009 * max(0, int(context.get("layer", 0)))
        key = digest(
            {**context, "method": c.method, "params": params, "k": k, "seed": seed}
        )
        path = Path(cache_root) / "fits" / key
        with lock(path.with_suffix(".lock")):
            if not (path / "fit.json").exists():
                start = time.perf_counter()
                with gpu_slot(
                    cfg,
                    max(
                        0.25,
                        (X.nbytes * 8 + k * X.shape[1] * (c.rank + 1) * 128) / 1024**3,
                    ),
                ):
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
        results.append(
            {
                "k": k,
                "fit_path": str(path),
                **record["scores"],
                "seconds": record["seconds"],
            }
        )
    best = min(r["criterion"] for r in results)
    selected = min(
        (
            r
            for r in results
            if r["criterion"] <= best + c.parsimony_tolerance * max(abs(best), 1.0)
        ),
        key=lambda r: r["k"],
    )
    model, _ = load_fitted(selected["fit_path"])
    return model, dict(
        selected=selected,
        candidates=results,
        requested_k=requested,
        omitted_k=[k for k in requested if k not in candidates],
    )

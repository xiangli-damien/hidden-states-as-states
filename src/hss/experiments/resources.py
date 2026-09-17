"""Conservative resource planning; estimates are not operating-system limits."""

from dataclasses import asdict

from .artifacts import digest


def candidates(cfg):
    c = cfg.cluster
    return (
        [c.k] if c.k is not None else (c.k_values or list(range(c.k_min, c.k_max + 1)))
    )


def model_bytes(cfg, d, k):
    if cfg.cluster.method == "mfa":
        return 8 * (k + k * d * (2 + cfg.cluster.rank))
    if cfg.cluster.method in ("kmeans", "minibatch_kmeans"):
        return 4 * k * d
    covariance = {"diag": k * d, "spherical": k, "tied": d * d, "full": k * d * d}[
        cfg.cluster.covariance_type
    ]
    return 8 * (k + k * d + 2 * covariance)  # covariance and precision Cholesky


def storage_entries(cfg, data):
    """Candidate-array upper bound, deduplicated across eta/probe-only variants.

    Does not subtract existing on-disk checkpoints or ineligible K, and excludes
    filesystem/JSON overhead, projected data, and final per-trial exported copies.
    """
    if cfg.evaluation.fixed_map:
        return {}
    e = cfg.evaluation
    context = {
        "snapshot": data.info["key"],
        "seed": cfg.seed,
        "cluster": {
            k: v
            for k, v in asdict(cfg.cluster).items()
            if k
            not in (
                "k",
                "k_values",
                "k_min",
                "k_max",
                "parsimony_tolerance",
                "assignment",
            )
        },
        "transform": asdict(cfg.transform),
        "pca_seed": e.split_seed,
        "training": {
            k: getattr(e, k)
            for k in (
                "mode",
                "train_fraction",
                "split_seed",
                "fit_fraction",
                "positive_label",
                "fixed_k_map",
            )
        },
    }
    d = cfg.transform.pca_components or data.state_dim()
    layers = [-1] if e.mode == "global_control" else data.layers()
    return {
        digest({**context, "layer": layer, "k": k}): model_bytes(cfg, d, k)
        for layer in layers
        for k in candidates(cfg)
    }

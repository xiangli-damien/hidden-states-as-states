"""Comparable geometry metrics on saved assignments, independent of labels.

These are descriptive diagnostics, not a second round of label-based selection.
Silhouette uses a seeded sample; CH and DB use every stored response.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from threadpoolctl import threadpool_limits

from ..results.models import load_layer
from ..data import CachedStates


def cluster_quality(result, cache_path, *, max_silhouette=2000, seed=42):
    data = CachedStates(cache_path)
    if data.info["key"] != result.summary["snapshot"]:
        raise ValueError("Quality features must match the fitted result snapshot")
    if not np.array_equal(data.meta.sample_id, result.rows.sample_id):
        raise ValueError("Quality feature row order differs from assignments")
    selected = np.sort(
        np.random.default_rng(seed).choice(
            len(result.rows), min(max_silhouette, len(result.rows)), replace=False
        )
    )
    rows = []
    with threadpool_limits(limits=1):
        for j, layer in enumerate(result.layers):
            model, projection, scan = load_layer(result.path, layer)
            X = projection.transform(data.array(layer))
            labels = np.asarray(result.states[:, j])
            _, counts = np.unique(labels, return_counts=True)
            probabilities = counts / counts.sum()
            eligible = 1 < len(counts) < len(X)
            sampled_k = len(np.unique(labels[selected]))
            cfg = model.config()
            rows.append(
                dict(
                    trial_id=result.summary["trial_id"],
                    method=result.config["cluster"]["method"],
                    representation=result.config["data"]["representation"],
                    layer=layer,
                    selected_k=model.n_clusters(),
                    active_k=len(counts),
                    silhouette=float(silhouette_score(X[selected], labels[selected]))
                    if 1 < sampled_k < len(selected)
                    else None,
                    silhouette_n=len(selected),
                    silhouette_seed=seed,
                    calinski_harabasz=float(calinski_harabasz_score(X, labels))
                    if eligible
                    else None,
                    davies_bouldin=float(davies_bouldin_score(X, labels))
                    if eligible
                    else None,
                    min_cluster_n=int(counts.min()),
                    max_cluster_fraction=float(probabilities.max()),
                    singleton_clusters=int((counts == 1).sum()),
                    occupancy_entropy_bits=float(
                        -(probabilities * np.log2(probabilities)).sum()
                    ),
                    converged=cfg.get("converged"),
                    n_iter=cfg.get("n_iter", len(cfg.get("history", [])) or None),
                    selection_criterion=scan["selected"].get("criterion"),
                    selected_fit_seconds=scan["selected"].get("seconds"),
                    k_policy="matched reference"
                    if result.config["evaluation"]["fixed_k_map"]
                    else "independent selection",
                )
            )
    return pd.DataFrame(rows)

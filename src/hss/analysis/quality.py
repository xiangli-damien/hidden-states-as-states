"""Comparable geometry metrics on saved assignments, independent of labels.

These are descriptive diagnostics, not a second round of label-based selection.
Silhouette uses a seeded sample; CH and DB use every stored response.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from threadpoolctl import threadpool_limits

from ..data import CachedStates
from ..results.models import load_layer


def state_outcomes(result):
    """Descriptive state membership, outcome and length; no label-based refit."""
    records = []
    for j, layer in enumerate(result.layers):
        for state in np.unique(result.states[:, j]):
            part = result.rows.loc[result.states[:, j] == state]
            ended = part.finish_reason.ne("length")
            records.append(
                dict(
                    layer=layer,
                    state=int(state),
                    n=len(part),
                    accuracy=float(part.label.mean()),
                    accuracy_delta=float(part.label.mean() - result.rows.label.mean()),
                    mean_tokens=float(part.n_tokens.mean()),
                    median_tokens=float(part.n_tokens.median()),
                    truncated_fraction=float((~ended).mean()),
                    n_nontruncated=int(ended.sum()),
                    accuracy_nontruncated=float(part.loc[ended, "label"].mean())
                    if ended.any()
                    else None,
                )
            )
    return pd.DataFrame(records)


def prediction_quality(result):
    """Held-out diagnostics including imbalance-aware baselines and PR-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    records = []
    frame = result.table("predictions.parquet")
    for method, part in frame.groupby("method"):
        y, scores = part.label.to_numpy(dtype=int), part.score.to_numpy()
        threshold = 0 if method == "LinearSVM" else 0.5
        prediction = scores >= threshold
        both = len(np.unique(y)) == 2
        records.append(
            dict(
                trial_id=result.summary["trial_id"],
                cluster_method=result.config["cluster"]["method"],
                predictor=method,
                n_test=len(y),
                positive_prevalence=float(y.mean()),
                majority_accuracy=float(max(y.mean(), 1 - y.mean())),
                auroc=float(roc_auc_score(y, scores)) if both else None,
                average_precision=float(average_precision_score(y, scores))
                if both
                else None,
                accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                mcc=float(matthews_corrcoef(y, prediction)),
                brier=float(brier_score_loss(y, scores))
                if method != "LinearSVM"
                else None,
            )
        )
    return pd.DataFrame(records)


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

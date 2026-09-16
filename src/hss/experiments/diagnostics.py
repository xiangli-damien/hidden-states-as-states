"""Geometry, state separation and matched-center reliability diagnostics."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def symmetric_diag_kl(mu_a, var_a, mu_b, var_b):
    va, vb = np.maximum(var_a, 1e-12), np.maximum(var_b, 1e-12)
    diff = np.square(mu_a - mu_b)
    return 0.25 * float(np.sum(va / vb + vb / va + diff * (1 / va + 1 / vb) - 2))


def layer_diagnostics(X, model, projection, assignment, requested):
    result = {}
    if "dispersion" in requested:
        result["covariance_trace"] = float(np.var(X, axis=0, dtype=np.float64).sum())
        result["mean_norm"] = float(
            np.sqrt(np.square(np.asarray(X, dtype=np.float64)).sum(1)).mean()
        )
    if "effective_rank" in requested:
        centered = np.asarray(X, dtype=np.float64) - X.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        eigen = singular**2
        p = eigen / max(eigen.sum(), 1e-300)
        positive = p > 0
        result["effective_rank"] = float(
            np.exp(-np.sum(p[positive] * np.log(p[positive])))
        )
    if (
        "separation" in requested
        and hasattr(model, "covariance_type")
        and model.covariance_type == "diag"
    ):
        # All statistics use the map-fitting data only, in the model's feature space.
        from .fitting import assign

        features = projection.transform(X)
        labels = assign(model, features, assignment)
        within, between = [], []
        for k in range(model.n_clusters()):
            part = features[labels == k]
            if len(part) > 1:
                within.append(
                    symmetric_diag_kl(
                        model.means_[k],
                        model.covariances_[k],
                        part.mean(0),
                        part.var(0),
                    )
                )
            for j in range(k):
                between.append(
                    symmetric_diag_kl(
                        model.means_[k],
                        model.covariances_[k],
                        model.means_[j],
                        model.covariances_[j],
                    )
                )
        result["within_fit_symmetric_kl"] = float(np.mean(within)) if within else None
        result["between_cluster_symmetric_kl"] = (
            float(np.mean(between)) if between else None
        )
    # Uniform random sample assignments, then recompute centroids and Hungarian
    # match to fitted centroids. This is distinct from random centroid pairing.
    if "separation" in requested:
        features = projection.transform(X)
        k = model.n_clusters()
        distances = []
        reference = projection.inverse(model.centers())
        a = reference / np.maximum(
            np.linalg.norm(reference, axis=1, keepdims=True), 1e-12
        )
        rng = np.random.default_rng(42)
        for _ in range(5):
            labels = rng.integers(k, size=len(X))
            centers = np.array(
                [X[labels == j].mean(0) for j in range(k) if np.any(labels == j)]
            )
            b = centers / np.maximum(
                np.linalg.norm(centers, axis=1, keepdims=True), 1e-12
            )
            sim = a @ b.T
            i, j = linear_sum_assignment(-sim)
            distances.append(float((1 - sim[i, j]).mean()))
        result["uniform_assignment_centroid_distance"] = float(np.mean(distances))
        result["uniform_assignment_repeats"] = len(distances)
    return result


def compare_results(reference, target):
    """Compare refits/ablations; never align unequal model hidden dimensions."""
    from .runner import _load_layer

    a, b = Path(reference), Path(target)
    am, bm = pd.read_parquet(a / "rows.parquet"), pd.read_parquet(b / "rows.parquet")
    sa, sb = np.load(a / "states.npy"), np.load(b / "states.npy")
    ak = json.loads((a / "summary.json").read_text())
    bk = json.loads((b / "summary.json").read_text())
    if ak["model"] != bk["model"]:
        raise ValueError(
            "Centroid reliability requires the same model/revision; compare normalized profiles across models instead"
        )
    columns = ["sample_id", "token_end"]
    joined = am.reset_index().merge(
        bm.reset_index(), on=columns, suffixes=("_a", "_b"), validate="one_to_one"
    )
    a_layers = [p["layer"] for p in ak["profile"]]
    b_layers = [p["layer"] for p in bk["profile"]]
    records = []
    for layer in a_layers:
        if layer not in b_layers:
            continue
        ma, pa, _ = _load_layer(a, layer)
        mb, pb, _ = _load_layer(b, layer)
        ca, cb = pa.inverse(ma.centers()), pb.inverse(mb.centers())
        an = ca / np.maximum(np.linalg.norm(ca, axis=1, keepdims=True), 1e-12)
        bn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
        similarity = an @ bn.T
        i, j = linear_sum_assignment(-similarity)
        records.append(
            {
                "layer": layer,
                "k_reference": len(ca),
                "k_target": len(cb),
                "centroid_cosine_distance": float((1 - similarity[i, j]).mean()),
                "unmatched_reference": len(ca) - len(i),
                "unmatched_target": len(cb) - len(j),
                "random_pair_cosine_distance": float((1 - similarity).mean()),
                "ari": float(
                    adjusted_rand_score(
                        sa[joined.index_a, a_layers.index(layer)],
                        sb[joined.index_b, b_layers.index(layer)],
                    )
                )
                if len(joined)
                else None,
                "common_rows": len(joined),
            }
        )
    return records

"""Descriptive diagnostics of frozen clusters; never used to choose a fit.

All distances use the original feature units. Metadata labels are used only
after fitting. Associations and Wilson intervals are in-sample descriptions.
"""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def participation_dimension(X):
    """Exact trace(C)^2 / trace(C^2), using the smaller sample Gram matrix.

    This is a descriptive effective dimension, not an estimate of MFA rank.
    """
    centered = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0, dtype=np.float64)
    energy = np.square(centered).sum()
    if not energy:
        return 0.0
    gram = centered @ centered.T if len(X) <= X.shape[1] else centered.T @ centered
    return float(energy * energy / np.square(gram).sum())


def association(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return dict(ari=float(adjusted_rand_score(a, b)),
                ami=float(adjusted_mutual_info_score(a, b)),
                rows=np.unique(a).astype(int).tolist(),
                columns=np.unique(b).astype(int).tolist(),
                counts=contingency_matrix(a, b).tolist())


def neighborhood_audit(X, coordinates, anchors, k=15):
    """Exact raw-space kNN overlap and trustworthiness on fixed anchors.

    Compare anchor-to-all distances; exclude the anchor itself. Trustworthiness
    penalizes new display neighbors by their original-space rank. kNN recall is
    more direct and intentionally harsh for a 3584-to-2 dimension projection.
    """
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X)
    anchors = np.asarray(anchors, dtype=int)
    n = len(X)
    if not 0 < k < (n - 1) / 2 or not len(anchors):
        raise ValueError("Need anchors and k < (n-1)/2")
    high = pairwise_distances(X[anchors], X, metric="euclidean", n_jobs=1)
    low = pairwise_distances(np.asarray(coordinates)[anchors], coordinates, n_jobs=1)
    high[np.arange(len(anchors)), anchors] = np.inf
    low[np.arange(len(anchors)), anchors] = np.inf
    order = np.argsort(high, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, n + 1), order.shape), axis=1)
    neighbors = np.argsort(low, axis=1, kind="stable")[:, :k]
    neighbor_ranks = np.take_along_axis(ranks, neighbors, axis=1)
    recall = (neighbor_ranks <= k).sum() / (len(anchors) * k)
    penalty = np.maximum(neighbor_ranks - k, 0).sum()
    trust = 1 - 2 * penalty / (len(anchors) * k * (2 * n - 3 * k - 1))
    return dict(k=k, anchors=len(anchors), recall=float(recall), trustworthiness=float(trust))

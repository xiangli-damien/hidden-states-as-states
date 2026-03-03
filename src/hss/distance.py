from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .utils import f32, normalize_l2

Metric = Literal["euclidean", "cosine"]


def cosine_similarity_matrix(
    A: np.ndarray, B: np.ndarray, *, eps: float = 1e-12
) -> np.ndarray:
    An = normalize_l2(f32(A), axis=1, eps=eps)
    Bn = normalize_l2(f32(B), axis=1, eps=eps)
    return (An @ Bn.T).astype(np.float32, copy=False)


def euclidean_distance_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = f32(A)
    B = f32(B)
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2.astype(np.float32, copy=False)


def pairwise_distance(
    A: np.ndarray,
    B: np.ndarray,
    *,
    metric: Metric = "euclidean",
    eps: float = 1e-12,
) -> np.ndarray:
    if metric == "euclidean":
        return np.sqrt(euclidean_distance_sq(A, B))
    if metric == "cosine":
        return (1.0 - cosine_similarity_matrix(A, B, eps=eps)).astype(
            np.float32, copy=False
        )
    raise ValueError(f"Unknown metric: {metric}")


def assign_nearest(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    metric: Metric = "euclidean",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    X = f32(X)
    centers = f32(centers)
    if metric == "euclidean":
        D = euclidean_distance_sq(X, centers)
        labels = np.argmin(D, axis=1).astype(np.int32)
        dists = np.sqrt(D[np.arange(len(X)), labels]).astype(np.float32)
        return labels, dists
    if metric == "cosine":
        S = cosine_similarity_matrix(X, centers, eps=eps)
        labels = np.argmax(S, axis=1).astype(np.int32)
        scores = S[np.arange(len(X)), labels].astype(np.float32)
        return labels, scores
    raise ValueError(f"Unknown metric: {metric}")


def assign_nearest_chunked(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    metric: Metric = "euclidean",
    chunk_size: int = 8192,
    eps: float = 1e-12,
) -> np.ndarray:
    X = f32(X)
    centers = f32(centers)
    n = len(X)
    out = np.empty(n, dtype=np.int32)
    cs = max(1, int(chunk_size))
    for i in range(0, n, cs):
        j = min(n, i + cs)
        out[i:j], _ = assign_nearest(X[i:j], centers, metric=metric, eps=eps)
    return out
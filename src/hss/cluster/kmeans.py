from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..distance import assign_nearest_chunked, euclidean_distance_sq
from ..utils import f32
from .base import ClusterModel


@dataclass(frozen=True)
class KMeansModel(ClusterModel):
    centers_: np.ndarray
    n_iter_: int | None = None
    converged_: bool | None = None
    inertia_: float | None = None

    def n_clusters(self) -> int:
        return int(self.centers_.shape[0])

    def centers(self) -> np.ndarray:
        return f32(self.centers_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return assign_nearest_chunked(f32(X), self.centers_, metric="euclidean")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a deterministic pseudo-soft assignment.

        K-means is not probabilistic, but downstream sanity checks and storage
        benefit from a dense assignment matrix. We therefore normalize inverse
        Euclidean distances into a simplex. This is intentionally labeled as a
        pseudo-probability in the model config.
        """

        X = f32(X)
        d2 = euclidean_distance_sq(X, self.centers_).astype(np.float64, copy=False)
        inv = 1.0 / np.maximum(np.sqrt(d2), 1e-6)
        inv_sum = np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
        prob = inv / inv_sum
        return prob.astype(np.float32, copy=False)

    def config(self) -> Dict[str, Any]:
        return {
            "kind": "kmeans",
            "n_clusters": self.n_clusters(),
            "soft_assignment": "inverse_distance_normalized",
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "inertia": self.inertia_,
        }

    def state_arrays(self) -> Dict[str, np.ndarray]:
        return {"centers": f32(self.centers_)}

    @classmethod
    def from_state(
        cls,
        config: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
    ) -> "KMeansModel":
        return cls(
            centers_=f32(arrays["centers"]),
            n_iter_=config.get("n_iter"),
            converged_=config.get("converged"),
            inertia_=config.get("inertia"),
        )


def _fit_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    batch_size: int = 1024,
    n_init: int = 10,
    max_iter: int = 100,
) -> KMeansModel:
    model = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=int(batch_size),
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit(X)
    return KMeansModel(centers_=f32(model.cluster_centers_))

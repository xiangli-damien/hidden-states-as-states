from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..distance import assign_nearest_chunked
from ..utils import f32
from .base import ClusterModel


@dataclass(frozen=True)
class KMeansModel(ClusterModel):
    centers_: np.ndarray

    def n_clusters(self) -> int:
        return int(self.centers_.shape[0])

    def centers(self) -> np.ndarray:
        return f32(self.centers_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return assign_nearest_chunked(f32(X), self.centers_, metric="euclidean")

    def config(self) -> Dict[str, Any]:
        return {"kind": "kmeans", "n_clusters": self.n_clusters()}

    def state_arrays(self) -> Dict[str, np.ndarray]:
        return {"centers": f32(self.centers_)}

    @classmethod
    def from_state(
        cls, config: Dict[str, Any], arrays: Dict[str, np.ndarray]
    ) -> KMeansModel:
        return cls(centers_=f32(arrays["centers"]))


def _fit_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    batch_size: int = 1024,
    n_init: int = 10,
    max_iter: int = 100,
) -> KMeansModel:
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    ).fit(X)
    return KMeansModel(centers_=f32(model.cluster_centers_))
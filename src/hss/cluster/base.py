from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ..types import BatchFactory
from ..utils import f32


class ClusterModel(ABC):
    @abstractmethod
    def n_clusters(self) -> int: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft assignments.

        Models with a natural probabilistic interpretation should override this
        method. The default implementation returns a one-hot encoding of hard
        assignments, which keeps the core persistence contract uniform even for
        simpler models.
        """

        labels = np.asarray(self.predict(X), dtype=np.int64).ravel()
        out = np.zeros((labels.shape[0], self.n_clusters()), dtype=np.float32)
        if labels.size:
            out[np.arange(labels.shape[0]), labels] = 1.0
        return out

    @abstractmethod
    def centers(self) -> np.ndarray: ...

    @abstractmethod
    def config(self) -> Dict[str, Any]: ...

    @abstractmethod
    def state_arrays(self) -> Dict[str, np.ndarray]: ...

    @classmethod
    @abstractmethod
    def from_state(
        cls,
        config: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
    ) -> "ClusterModel": ...


def _materialize(factory: BatchFactory) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for X in factory():
        X = f32(X)
        if X.ndim == 2 and X.shape[0] > 0:
            chunks.append(X)
    if not chunks:
        raise ValueError("No data to materialize")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

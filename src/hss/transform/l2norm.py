from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..utils import f32, normalize_l2
from .base import BatchFactory, Transform


@dataclass
class L2NormTransform(Transform):
    eps: float = 1e-12

    def fit(self, factory: BatchFactory) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return normalize_l2(f32(X), axis=1, eps=self.eps).astype(
            np.float32, copy=False
        )

    def inverse(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "L2NormTransform has no inverse: L2 normalization is not invertible. "
            "For pipelines that need centers in original space, avoid L2Norm as the last step."
        )

    def config(self) -> Dict[str, Any]:
        return {"name": "l2", "eps": float(self.eps)}

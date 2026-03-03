from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..utils import f32
from .base import BatchFactory, Transform


@dataclass
class IdentityTransform(Transform):
    def fit(self, factory: BatchFactory) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return f32(X)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return f32(X)

    def config(self) -> Dict[str, Any]:
        return {"name": "identity"}

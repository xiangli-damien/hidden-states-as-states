from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..utils import f32
from .base import BatchFactory, Transform


@dataclass
class StandardizeTransform(Transform):
    with_mean: bool = True
    with_std: bool = True
    eps: float = 1e-12
    mean_: Optional[np.ndarray] = field(default=None, repr=False)
    scale_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, factory: BatchFactory) -> None:
        count = 0
        mean: Optional[np.ndarray] = None
        m2: Optional[np.ndarray] = None
        for X in factory():
            X = f32(X).astype(np.float64, copy=False)
            b = int(X.shape[0])
            if b == 0:
                continue
            batch_sum = X.sum(axis=0)
            batch_sumsq = np.sum(X * X, axis=0)
            batch_mean = batch_sum / b
            batch_m2 = batch_sumsq - b * batch_mean * batch_mean
            np.maximum(batch_m2, 0.0, out=batch_m2)
            if mean is None:
                mean = batch_mean
                m2 = batch_m2
                count = b
                continue
            delta = batch_mean - mean
            total = count + b
            mean = mean + delta * (b / total)
            m2 = m2 + batch_m2 + (delta * delta) * (count * b / total)
            count = total
        if mean is None or m2 is None or count <= 0:
            raise ValueError("No data for StandardizeTransform.fit")
        var = (m2 / count).astype(np.float32, copy=False)
        scale = np.sqrt(np.maximum(var, 0.0) + self.eps).astype(
            np.float32, copy=False
        )
        self.mean_ = (
            mean.astype(np.float32)
            if self.with_mean
            else np.zeros_like(mean, dtype=np.float32)
        )
        self.scale_ = (
            scale if self.with_std else np.ones_like(scale, dtype=np.float32)
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardizeTransform is not fitted")
        return ((f32(X) - self.mean_) / self.scale_).astype(
            np.float32, copy=False
        )

    def inverse(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardizeTransform is not fitted")
        return (f32(X) * self.scale_ + self.mean_).astype(
            np.float32, copy=False
        )

    def config(self) -> Dict[str, Any]:
        return {
            "name": "standardize",
            "with_mean": bool(self.with_mean),
            "with_std": bool(self.with_std),
            "eps": float(self.eps),
        }

    def state_arrays(self) -> Dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            return {}
        return {"mean": f32(self.mean_), "scale": f32(self.scale_)}

    def load_state_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        self.mean_ = f32(arrays["mean"])
        self.scale_ = f32(arrays["scale"])

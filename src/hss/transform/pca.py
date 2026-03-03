from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import IncrementalPCA

from ..utils import f32
from .base import BatchFactory, Transform


@dataclass
class PCATransform(Transform):
    n_components: int = 128
    whiten: bool = False
    components_: Optional[np.ndarray] = field(default=None, repr=False)
    mean_: Optional[np.ndarray] = field(default=None, repr=False)
    explained_variance_: Optional[np.ndarray] = field(default=None, repr=False)
    explained_variance_ratio_sum_: float = 0.0

    def fit(self, factory: BatchFactory) -> None:
        ipca: Optional[IncrementalPCA] = None
        nc: Optional[int] = None
        carry: Optional[np.ndarray] = None
        for X in factory():
            X = f32(X)
            if X.shape[0] == 0:
                continue
            if ipca is None:
                d = int(X.shape[1])
                nc = min(int(self.n_components), d)
                ipca = IncrementalPCA(
                    n_components=nc, whiten=bool(self.whiten)
                )
            if carry is not None:
                X = np.concatenate([carry, X], axis=0).astype(
                    np.float32, copy=False
                )
                carry = None
            if nc is not None and int(X.shape[0]) < nc:
                carry = X
                continue
            ipca.partial_fit(X)
        if carry is not None and ipca is not None and nc is not None:
            if carry.shape[0] >= nc:
                ipca.partial_fit(carry)
        if ipca is None:
            raise ValueError("No data for PCATransform.fit")
        self.components_ = f32(ipca.components_)
        self.mean_ = f32(ipca.mean_)
        self.explained_variance_ = f32(ipca.explained_variance_)
        self.explained_variance_ratio_sum_ = float(
            np.sum(ipca.explained_variance_ratio_)
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCATransform is not fitted")
        Xc = f32(X) - self.mean_
        Z = (Xc @ self.components_.T).astype(np.float32, copy=False)
        if self.whiten and self.explained_variance_ is not None:
            Z = (Z / np.sqrt(self.explained_variance_ + 1e-12)).astype(
                np.float32, copy=False
            )
        return Z

    def inverse(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCATransform is not fitted")
        Z = f32(X)
        if self.whiten and self.explained_variance_ is not None:
            Z = (Z * np.sqrt(self.explained_variance_ + 1e-12)).astype(
                np.float32, copy=False
            )
        return (Z @ self.components_ + self.mean_).astype(
            np.float32, copy=False
        )

    def config(self) -> Dict[str, Any]:
        return {
            "name": "pca",
            "n_components": int(self.n_components),
            "whiten": bool(self.whiten),
            "explained_variance_ratio_sum": float(
                self.explained_variance_ratio_sum_
            ),
        }

    def state_arrays(self) -> Dict[str, np.ndarray]:
        if self.components_ is None or self.mean_ is None:
            return {}
        out: Dict[str, np.ndarray] = {
            "components": f32(self.components_),
            "mean": f32(self.mean_),
        }
        if self.explained_variance_ is not None:
            out["explained_variance"] = f32(self.explained_variance_)
        return out

    def load_state_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        self.components_ = f32(arrays["components"])
        self.mean_ = f32(arrays["mean"])
        if "explained_variance" in arrays:
            self.explained_variance_ = f32(arrays["explained_variance"])

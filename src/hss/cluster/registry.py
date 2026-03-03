from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import ClusterModel
from .gmm import GMMModel
from .kmeans import KMeansModel

_MODEL_REGISTRY: Dict[str, type] = {
    "kmeans": KMeansModel,
    "gmm": GMMModel,
}


def rebuild_model(
    config: Dict[str, Any], arrays: Dict[str, np.ndarray]
) -> ClusterModel:
    kind = str(config.get("kind", "")).lower()
    cls = _MODEL_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown model kind: {kind}")
    return cls.from_state(config, arrays)

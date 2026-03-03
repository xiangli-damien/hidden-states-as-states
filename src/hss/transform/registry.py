from __future__ import annotations

from typing import Any, Dict

from .base import Transform
from .identity import IdentityTransform
from .l2norm import L2NormTransform
from .pca import PCATransform
from .standardize import StandardizeTransform

_REGISTRY: Dict[str, type] = {
    "identity": IdentityTransform,
    "l2": L2NormTransform,
    "l2norm": L2NormTransform,
    "standardize": StandardizeTransform,
    "zscore": StandardizeTransform,
    "pca": PCATransform,
}


def _build_one(spec: Dict[str, Any]) -> Transform:
    name = str(spec.get("name", "identity")).lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown transform: {name}")
    if cls is IdentityTransform:
        return IdentityTransform()
    if cls is L2NormTransform:
        return L2NormTransform(eps=float(spec.get("eps", 1e-12)))
    if cls is StandardizeTransform:
        return StandardizeTransform(
            with_mean=bool(spec.get("with_mean", True)),
            with_std=bool(spec.get("with_std", True)),
            eps=float(spec.get("eps", 1e-12)),
        )
    if cls is PCATransform:
        return PCATransform(
            n_components=int(spec.get("n_components", 128)),
            whiten=bool(spec.get("whiten", False)),
        )
    known = {"eps", "with_mean", "with_std", "n_components", "whiten"}
    kwargs = {k: v for k, v in spec.items() if k in known}
    return cls(**kwargs)

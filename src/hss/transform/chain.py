from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

import numpy as np

from ..utils import f32
from .base import BatchFactory, Transform
from .registry import _build_one


@dataclass
class TransformChain:
    steps: List[Transform] = field(default_factory=list)

    def fit(self, factory: BatchFactory) -> None:
        fitted: List[Transform] = []
        for step in self.steps:
            prev_fitted = list(fitted)

            def step_factory(
                _prev=prev_fitted, _raw=factory
            ) -> Iterator[np.ndarray]:
                for X in _raw():
                    out = f32(X)
                    for s in _prev:
                        out = s.transform(out)
                    yield out

            step.fit(step_factory)
            fitted.append(step)

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = f32(X)
        for step in self.steps:
            out = step.transform(out)
        return out

    def inverse(self, X: np.ndarray) -> np.ndarray:
        out = f32(X)
        for step in reversed(self.steps):
            out = step.inverse(out)
        return out

    def config(self) -> Dict[str, Any]:
        return {"name": "chain", "steps": [s.config() for s in self.steps]}

    def state_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, step in enumerate(self.steps):
            for k, v in step.state_arrays().items():
                out[f"{i}.{k}"] = v
        return out

    def load_state_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        for i, step in enumerate(self.steps):
            prefix = f"{i}."
            sub = {
                k[len(prefix) :]: v
                for k, v in arrays.items()
                if k.startswith(prefix)
            }
            if sub:
                step.load_state_arrays(sub)


def build_chain(step_dicts: List[Dict[str, Any]]) -> TransformChain:
    return TransformChain(steps=[_build_one(s) for s in step_dicts])


def build_chain_from_spec(spec: Any) -> TransformChain:
    if spec is None:
        return TransformChain()
    if isinstance(spec, TransformChain):
        return spec
    if hasattr(spec, "steps"):
        return build_chain(
            [
                s.to_dict() if hasattr(s, "to_dict") else dict(s)
                for s in spec.steps
            ]
        )
    if isinstance(spec, dict):
        return build_chain(spec.get("steps", []))
    if isinstance(spec, list):
        return build_chain([dict(s) for s in spec])
    return TransformChain()
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

BatchFactory = Callable[[], Iterator[np.ndarray]]


@dataclass(frozen=True)
class Batch:
    states: np.ndarray
    ids: Optional[np.ndarray] = None


@runtime_checkable
class StateProvider(Protocol):
    def layers(self) -> Sequence[int]: ...
    def n_items(self) -> int: ...
    def state_dim(self) -> int: ...
    def iter_batches(
        self,
        *,
        layer: int,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> Iterator[Batch]: ...


@dataclass(frozen=True)
class TransformStepSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.params)
        d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TransformStepSpec:
        return cls(name=d["name"], params=dict(d.get("params", {})))


@dataclass(frozen=True)
class TransformSpec:
    steps: List[TransformStepSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TransformSpec:
        steps = d.get("steps", [])
        return cls(steps=[TransformStepSpec.from_dict(s) for s in steps])


@dataclass(frozen=True)
class ClusterSpec:
    method: Literal["kmeans", "gmm"] = "gmm"
    k: Optional[int] = None
    k_range: Tuple[int, int] = (2, 40)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "k": self.k,
            "k_range": list(self.k_range),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ClusterSpec:
        kr = d.get("k_range", [2, 40])
        return cls(
            method=d.get("method", "gmm"),
            k=d.get("k"),
            k_range=(int(kr[0]), int(kr[1])),
            params=dict(d.get("params", {})),
        )


@dataclass(frozen=True)
class AlignSpec:
    similarity: Literal["cosine", "euclidean"] = "cosine"
    method: Literal["hungarian", "greedy"] = "hungarian"
    threshold: float = 0.0
    allow_negative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity": self.similarity,
            "method": self.method,
            "threshold": float(self.threshold),
            "allow_negative": bool(self.allow_negative),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AlignSpec:
        return cls(
            similarity=d.get("similarity", "cosine"),
            method=d.get("method", "hungarian"),
            threshold=float(d.get("threshold", 0.0)),
            allow_negative=bool(d.get("allow_negative", False)),
        )


@dataclass(frozen=True)
class HSSConfig:
    seed: int = 42
    layers: Optional[List[int]] = None
    indices: Optional[np.ndarray] = field(default=None, repr=False)
    transform: TransformSpec = field(default_factory=TransformSpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    alignment: AlignSpec = field(default_factory=AlignSpec)
    batch_size_fit: int = 4096
    batch_size_predict: int = 8192
    n_jobs: int = 1
    parallel_backend: str = "threading"
    parsimony_tolerance: float = 0.05
    icl_mode: str = "bic_plus_2entropy"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "seed": self.seed,
            "layers": self.layers,
            "transform": self.transform.to_dict(),
            "cluster": self.cluster.to_dict(),
            "alignment": self.alignment.to_dict(),
            "batch_size_fit": self.batch_size_fit,
            "batch_size_predict": self.batch_size_predict,
            "n_jobs": self.n_jobs,
            "parallel_backend": self.parallel_backend,
            "parsimony_tolerance": self.parsimony_tolerance,
            "icl_mode": self.icl_mode,
        }
        if self.indices is not None:
            d["indices"] = self.indices.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HSSConfig:
        indices = None
        if d.get("indices") is not None:
            indices = np.array(d["indices"], dtype=np.int64)
        return cls(
            seed=int(d.get("seed", 42)),
            layers=d.get("layers"),
            indices=indices,
            transform=(
                TransformSpec.from_dict(d["transform"])
                if "transform" in d
                else TransformSpec()
            ),
            cluster=(
                ClusterSpec.from_dict(d["cluster"])
                if "cluster" in d
                else ClusterSpec()
            ),
            alignment=(
                AlignSpec.from_dict(d["alignment"])
                if "alignment" in d
                else AlignSpec()
            ),
            batch_size_fit=int(d.get("batch_size_fit", 4096)),
            batch_size_predict=int(d.get("batch_size_predict", 8192)),
            n_jobs=int(d.get("n_jobs", 1)),
            parallel_backend=str(d.get("parallel_backend", "threading")),
            parsimony_tolerance=float(d.get("parsimony_tolerance", 0.05)),
            icl_mode=str(d.get("icl_mode", "bic_plus_2entropy")),
        )


@dataclass(frozen=True)
class LayerResult:
    layer: int
    labels: np.ndarray
    centers_hidden: np.ndarray
    n_clusters: int
    transform_config: Dict[str, Any]
    cluster_config: Dict[str, Any]

    def meta_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata (excludes numpy arrays)."""
        return {
            "layer": self.layer,
            "n_clusters": self.n_clusters,
            "transform_config": self.transform_config,
            "cluster_config": self.cluster_config,
        }

    @classmethod
    def from_meta_and_arrays(
        cls,
        meta: Dict[str, Any],
        labels: np.ndarray,
        centers_hidden: np.ndarray,
    ) -> LayerResult:
        """Reconstruct from metadata dict and numpy arrays."""
        return cls(
            layer=int(meta["layer"]),
            labels=labels,
            centers_hidden=centers_hidden,
            n_clusters=int(meta["n_clusters"]),
            transform_config=meta["transform_config"],
            cluster_config=meta["cluster_config"],
        )


@dataclass(frozen=True)
class AlignmentStep:
    layer_from: int
    layer_to: int
    matched: List[Tuple[int, int, float]]
    births: List[int]
    deaths: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_from": int(self.layer_from),
            "layer_to": int(self.layer_to),
            "matched": [
                (int(a), int(b), float(v)) for a, b, v in self.matched
            ],
            "births": [int(x) for x in self.births],
            "deaths": [int(x) for x in self.deaths],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AlignmentStep:
        """Reconstruct from a dict (inverse of to_dict)."""
        return cls(
            layer_from=int(d["layer_from"]),
            layer_to=int(d["layer_to"]),
            matched=[(int(a), int(b), float(v)) for a, b, v in d["matched"]],
            births=[int(x) for x in d["births"]],
            deaths=[int(x) for x in d["deaths"]],
        )


@dataclass(frozen=True)
class AlignmentResult:
    layers: List[int]
    local_to_global: List[np.ndarray]
    steps: List[AlignmentStep]
    n_global_states: int

    def meta_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata (excludes numpy arrays)."""
        return {
            "layers": list(self.layers),
            "n_global_states": int(self.n_global_states),
        }

    def steps_list(self) -> List[Dict[str, Any]]:
        """Return steps as a JSON-serializable list of dicts."""
        return [s.to_dict() for s in self.steps]

    def local_to_global_dict(self) -> Dict[str, np.ndarray]:
        """Return local_to_global as a {str(layer): array} dict for npz saving."""
        return {
            str(layer): np.asarray(m, dtype=np.int32)
            for layer, m in zip(self.layers, self.local_to_global)
        }

    @classmethod
    def from_meta_steps_arrays(
        cls,
        meta: Dict[str, Any],
        steps_data: List[Dict[str, Any]],
        local_to_global_dict: Dict[int, np.ndarray],
    ) -> AlignmentResult:
        """Reconstruct from metadata, steps list, and local-to-global mapping."""
        layers = meta["layers"]
        local_to_global = [local_to_global_dict[l] for l in layers]
        steps = [AlignmentStep.from_dict(s) for s in steps_data]
        return cls(
            layers=layers,
            local_to_global=local_to_global,
            steps=steps,
            n_global_states=int(meta["n_global_states"]),
        )


@dataclass
class HSSResult:
    config: HSSConfig
    layer_results: List[LayerResult]
    alignment: Optional[AlignmentResult]
    global_labels: Optional[np.ndarray]
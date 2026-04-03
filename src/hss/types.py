from __future__ import annotations

"""Core public types for HSS.

This revision makes three architectural changes that are important for large
sentence-level and token-level experiments:

1. Stable row/sample tracking is first-class.
2. Soft assignments are part of the persisted result schema.
3. Provider metadata is generic and does not assume a single upstream format.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

BatchFactory = Callable[[], Iterator[np.ndarray]]


@dataclass(frozen=True)
class Batch:
    """One batch of states from a provider.

    Attributes:
        states:
            Hidden states with shape ``[batch, dim]``.
        ids:
            Stable row identifiers aligned with the batch. These are *not*
            required to be contiguous ``0..batch-1``. They should be the most
            useful external row identifier available (for example item_id,
            sentence_row, token_row, or absolute sample row).
        sample_ids:
            Optional per-row sample identifiers. This is the key field used to
            preserve sentence/token -> sample provenance through HSS.
        fields:
            Optional additional aligned metadata arrays. These are intended for
            numeric or compact string metadata such as ``sentence_index`` or
            ``token_position``.
    """

    states: np.ndarray
    ids: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None
    fields: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def row_ids(self) -> Optional[np.ndarray]:
        return self.ids


@runtime_checkable
class StateProvider(Protocol):
    """Minimal provider protocol used by HSS.

    Only the four methods below are required. Metadata-aware providers may also
    implement optional methods such as ``row_ids(...)``, ``sample_ids(...)``,
    ``metadata_fields(...)``, ``metadata_manifest()`` and ``unit()``. The core
    pipeline checks for these methods dynamically and falls back gracefully when
    they are absent.
    """

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
        payload = dict(self.params)
        payload["name"] = self.name
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformStepSpec":
        payload = dict(d)
        name = str(payload.pop("name", "identity"))
        params = dict(payload.pop("params", {}))
        params.update(payload)
        return cls(name=name, params=params)


@dataclass(frozen=True)
class TransformSpec:
    steps: List[TransformStepSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformSpec":
        return cls(
            steps=[TransformStepSpec.from_dict(step) for step in d.get("steps", [])]
        )


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
            "k_range": [int(self.k_range[0]), int(self.k_range[1])],
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClusterSpec":
        k_range = d.get("k_range", [2, 40])
        return cls(
            method=str(d.get("method", "gmm")),
            k=None if d.get("k") is None else int(d.get("k")),
            k_range=(int(k_range[0]), int(k_range[1])),
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
    def from_dict(cls, d: Dict[str, Any]) -> "AlignSpec":
        return cls(
            similarity=str(d.get("similarity", "cosine")),
            method=str(d.get("method", "hungarian")),
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
    parallel_backend: str = "auto"
    parsimony_tolerance: float = 0.05
    icl_mode: str = "bic_plus_2entropy"

    # New defaults for large-scale runs.
    store_soft_labels: bool = True
    soft_labels_dtype: str = "float16"
    capture_row_metadata: bool = True
    prefer_memmap: bool = True
    progress: bool = True
    log_every_batches: int = 16
    temp_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "seed": int(self.seed),
            "layers": None if self.layers is None else [int(x) for x in self.layers],
            "transform": self.transform.to_dict(),
            "cluster": self.cluster.to_dict(),
            "alignment": self.alignment.to_dict(),
            "batch_size_fit": int(self.batch_size_fit),
            "batch_size_predict": int(self.batch_size_predict),
            "n_jobs": int(self.n_jobs),
            "parallel_backend": str(self.parallel_backend),
            "parsimony_tolerance": float(self.parsimony_tolerance),
            "icl_mode": str(self.icl_mode),
            "store_soft_labels": bool(self.store_soft_labels),
            "soft_labels_dtype": str(self.soft_labels_dtype),
            "capture_row_metadata": bool(self.capture_row_metadata),
            "prefer_memmap": bool(self.prefer_memmap),
            "progress": bool(self.progress),
            "log_every_batches": int(self.log_every_batches),
            "temp_root": self.temp_root,
        }
        if self.indices is not None:
            payload["indices"] = np.asarray(self.indices, dtype=np.int64).tolist()
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HSSConfig":
        indices = None
        if d.get("indices") is not None:
            indices = np.asarray(d["indices"], dtype=np.int64)
        return cls(
            seed=int(d.get("seed", 42)),
            layers=None if d.get("layers") is None else [int(x) for x in d.get("layers", [])],
            indices=indices,
            transform=TransformSpec.from_dict(d.get("transform", {})),
            cluster=ClusterSpec.from_dict(d.get("cluster", {})),
            alignment=AlignSpec.from_dict(d.get("alignment", {})),
            batch_size_fit=int(d.get("batch_size_fit", 4096)),
            batch_size_predict=int(d.get("batch_size_predict", 8192)),
            n_jobs=int(d.get("n_jobs", 1)),
            parallel_backend=str(d.get("parallel_backend", "auto")),
            parsimony_tolerance=float(d.get("parsimony_tolerance", 0.05)),
            icl_mode=str(d.get("icl_mode", "bic_plus_2entropy")),
            store_soft_labels=bool(d.get("store_soft_labels", True)),
            soft_labels_dtype=str(d.get("soft_labels_dtype", "float16")),
            capture_row_metadata=bool(d.get("capture_row_metadata", True)),
            prefer_memmap=bool(d.get("prefer_memmap", True)),
            progress=bool(d.get("progress", True)),
            log_every_batches=int(d.get("log_every_batches", 16)),
            temp_root=d.get("temp_root"),
        )


@dataclass(frozen=True)
class RowMetadata:
    """Stable provider-side provenance for each row in the result order."""

    unit: str
    row_ids: np.ndarray
    sample_ids: Optional[np.ndarray] = None
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    provider_manifest: Dict[str, Any] = field(default_factory=dict)

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "unit": str(self.unit),
            "has_sample_ids": self.sample_ids is not None,
            "field_names": sorted(str(k) for k in self.fields.keys()),
            "provider_manifest": dict(self.provider_manifest),
        }

    def arrays_dict(self) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {
            "row_ids": np.asarray(self.row_ids),
        }
        if self.sample_ids is not None:
            arrays["sample_ids"] = np.asarray(self.sample_ids)
        for key, value in self.fields.items():
            arrays[f"field__{key}"] = np.asarray(value)
        return arrays

    @classmethod
    def from_meta_and_arrays(
        cls,
        meta: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
    ) -> "RowMetadata":
        fields: Dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            if key.startswith("field__"):
                fields[key[len("field__") :]] = np.asarray(value)
        sample_ids = arrays.get("sample_ids")
        return cls(
            unit=str(meta.get("unit", "item")),
            row_ids=np.asarray(arrays["row_ids"]),
            sample_ids=None if sample_ids is None else np.asarray(sample_ids),
            fields=fields,
            provider_manifest=dict(meta.get("provider_manifest", {})),
        )


@dataclass(frozen=True)
class LayerResult:
    layer: int
    labels: np.ndarray
    centers_hidden: np.ndarray
    n_clusters: int
    transform_config: Dict[str, Any]
    cluster_config: Dict[str, Any]
    soft_labels: Optional[np.ndarray] = None

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "layer": int(self.layer),
            "n_clusters": int(self.n_clusters),
            "transform_config": dict(self.transform_config),
            "cluster_config": dict(self.cluster_config),
            "has_soft_labels": self.soft_labels is not None,
        }

    @classmethod
    def from_meta_and_arrays(
        cls,
        meta: Dict[str, Any],
        labels: np.ndarray,
        centers_hidden: np.ndarray,
        soft_labels: Optional[np.ndarray] = None,
    ) -> "LayerResult":
        return cls(
            layer=int(meta["layer"]),
            labels=np.asarray(labels),
            centers_hidden=np.asarray(centers_hidden),
            n_clusters=int(meta["n_clusters"]),
            transform_config=dict(meta.get("transform_config", {})),
            cluster_config=dict(meta.get("cluster_config", {})),
            soft_labels=None if soft_labels is None else np.asarray(soft_labels),
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
            "matched": [(int(a), int(b), float(v)) for a, b, v in self.matched],
            "births": [int(x) for x in self.births],
            "deaths": [int(x) for x in self.deaths],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlignmentStep":
        return cls(
            layer_from=int(d["layer_from"]),
            layer_to=int(d["layer_to"]),
            matched=[(int(a), int(b), float(v)) for a, b, v in d.get("matched", [])],
            births=[int(x) for x in d.get("births", [])],
            deaths=[int(x) for x in d.get("deaths", [])],
        )


@dataclass(frozen=True)
class AlignmentResult:
    layers: List[int]
    local_to_global: List[np.ndarray]
    steps: List[AlignmentStep]
    n_global_states: int

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "layers": [int(x) for x in self.layers],
            "n_global_states": int(self.n_global_states),
        }

    def steps_list(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    def local_to_global_dict(self) -> Dict[str, np.ndarray]:
        return {
            str(int(layer)): np.asarray(mapping, dtype=np.int32)
            for layer, mapping in zip(self.layers, self.local_to_global)
        }

    @classmethod
    def from_meta_steps_arrays(
        cls,
        meta: Dict[str, Any],
        steps_data: List[Dict[str, Any]],
        local_to_global_dict: Dict[int | str, np.ndarray],
    ) -> "AlignmentResult":
        layers = [int(x) for x in meta.get("layers", [])]
        resolved: List[np.ndarray] = []
        for layer in layers:
            if layer in local_to_global_dict:
                resolved.append(np.asarray(local_to_global_dict[layer]))
            else:
                resolved.append(np.asarray(local_to_global_dict[str(layer)]))
        return cls(
            layers=layers,
            local_to_global=resolved,
            steps=[AlignmentStep.from_dict(step) for step in steps_data],
            n_global_states=int(meta.get("n_global_states", 0)),
        )


@dataclass
class HSSResult:
    config: HSSConfig
    layer_results: List[LayerResult]
    alignment: Optional[AlignmentResult]
    global_labels: Optional[np.ndarray]
    row_metadata: Optional[RowMetadata] = None

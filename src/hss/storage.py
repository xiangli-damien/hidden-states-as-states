from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cluster import ClusterModel, rebuild_model
from .transform import TransformChain, build_chain
from .types import AlignmentResult, HSSConfig, HSSResult, LayerResult, RowMetadata
from .utils import (
    atomic_json,
    atomic_npy,
    atomic_npz,
    ensure_dir,
    open_writable_memmap,
    read_json,
    to_jsonable,
)

_log = logging.getLogger(__name__)


def _parse_layer_num(name: str) -> Optional[int]:
    if not name.startswith("layer"):
        return None
    try:
        return int(name[5:])
    except ValueError:
        return None


@dataclass
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = ensure_dir(self.root)

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def alignment_dir(self) -> Path:
        return ensure_dir(self.root / "alignment")

    @property
    def row_metadata_dir(self) -> Path:
        return ensure_dir(self.root / "row_metadata")

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.json"

    def layer_dir(self, layer: int) -> Path:
        return ensure_dir(self.root / "layers" / f"layer{int(layer):03d}")

    def create_labels_memmap(
        self,
        *,
        layer: int,
        n_items: int,
    ) -> Tuple[Path, np.ndarray]:
        d = self.layer_dir(layer)
        p = d / "labels.tmp.npy"
        mm = open_writable_memmap(p, shape=(int(n_items),), dtype=np.int32)
        return p, mm

    def save_progress(self, payload: Dict[str, Any]) -> None:
        atomic_json(self.progress_path, payload)

    def save(self, result: HSSResult) -> None:
        atomic_json(self.config_path, result.config.to_dict())
        for layer_result in result.layer_results:
            self._save_layer_result(layer_result)
        if result.alignment is not None:
            self._save_alignment(result.alignment)
        if result.global_labels is not None:
            atomic_npy(self.root / "global_labels.npy", result.global_labels.astype(np.int32))
        if result.row_metadata is not None:
            self._save_row_metadata(result.row_metadata)

    def _save_row_metadata(self, metadata: RowMetadata) -> None:
        d = self.row_metadata_dir
        atomic_json(d / "meta.json", to_jsonable(metadata.meta_dict()))
        arrays = metadata.arrays_dict()
        if arrays:
            atomic_npz(d / "arrays.npz", **arrays)

    def _save_layer_result(self, lr: LayerResult) -> None:
        d = self.layer_dir(lr.layer)
        atomic_json(d / "meta.json", to_jsonable(lr.meta_dict()))
        atomic_npy(d / "labels.npy", np.asarray(lr.labels, dtype=np.int32))
        atomic_npy(d / "centers_hidden.npy", np.asarray(lr.centers_hidden, dtype=np.float32))
        if lr.soft_labels is not None:
            atomic_npy(d / "soft_labels.npy", np.asarray(lr.soft_labels))

    def _save_alignment(self, alignment: AlignmentResult) -> None:
        d = self.alignment_dir
        atomic_npz(d / "local_to_global.npz", **alignment.local_to_global_dict())
        atomic_json(d / "steps.json", to_jsonable(alignment.steps_list()))
        atomic_json(d / "meta.json", to_jsonable(alignment.meta_dict()))

    def load(self) -> HSSResult:
        config_dict = read_json(self.config_path)
        config = HSSConfig.from_dict(config_dict)
        layer_results = self._load_all_layer_results()
        alignment = self._load_alignment()
        global_labels = None
        gl_path = self.root / "global_labels.npy"
        if gl_path.exists():
            global_labels = np.load(gl_path, mmap_mode="r")
        row_metadata = self._load_row_metadata()
        return HSSResult(
            config=config,
            layer_results=layer_results,
            alignment=alignment,
            global_labels=global_labels,
            row_metadata=row_metadata,
        )

    def _load_row_metadata(self) -> Optional[RowMetadata]:
        meta_path = self.row_metadata_dir / "meta.json"
        arrays_path = self.row_metadata_dir / "arrays.npz"
        if not meta_path.exists() or not arrays_path.exists():
            return None
        meta = read_json(meta_path)
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {str(key): np.asarray(payload[key]) for key in payload.files}
        return RowMetadata.from_meta_and_arrays(meta, arrays)

    def _load_all_layer_results(self) -> List[LayerResult]:
        layers_root = self.root / "layers"
        if not layers_root.exists():
            return []
        results: List[LayerResult] = []
        for d in sorted(layers_root.iterdir()):
            if not d.is_dir():
                continue
            layer_num = _parse_layer_num(d.name)
            if layer_num is None:
                _log.warning("Skipping non-layer directory: %s", d.name)
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                _log.warning("Skipping layer directory without meta.json: %s", d.name)
                continue
            meta = read_json(meta_path)
            labels = np.load(d / "labels.npy", mmap_mode="r")
            centers = np.load(d / "centers_hidden.npy", mmap_mode="r")
            soft_labels = None
            soft_path = d / "soft_labels.npy"
            if soft_path.exists():
                soft_labels = np.load(soft_path, mmap_mode="r")
            results.append(
                LayerResult.from_meta_and_arrays(
                    meta,
                    labels=labels,
                    centers_hidden=centers,
                    soft_labels=soft_labels,
                )
            )
        return results

    def _load_alignment(self) -> Optional[AlignmentResult]:
        d = self.root / "alignment"
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        npz = np.load(d / "local_to_global.npz", allow_pickle=False)
        mappings: Dict[int | str, np.ndarray] = {
            key: np.asarray(npz[key], dtype=np.int32)
            for key in npz.files
        }
        steps_data = read_json(d / "steps.json")
        return AlignmentResult.from_meta_steps_arrays(meta, steps_data, mappings)

    def save_layer_model(
        self,
        layer: int,
        transform: TransformChain,
        model: ClusterModel,
    ) -> None:
        d = self.layer_dir(layer)
        atomic_json(d / "transform_config.json", transform.config())
        t_arrays = transform.state_arrays()
        if t_arrays:
            atomic_npz(d / "transform_arrays.npz", **t_arrays)
        else:
            atomic_npz(d / "transform_arrays.npz", _empty=np.array([], dtype=np.float32))
        atomic_json(d / "model_config.json", model.config())
        atomic_npz(d / "model_arrays.npz", **model.state_arrays())

    def load_layer_model(self, layer: int) -> Tuple[TransformChain, ClusterModel]:
        d = self.layer_dir(layer)
        t_cfg = read_json(d / "transform_config.json")
        chain = build_chain(t_cfg.get("steps", []))
        t_arrays_path = d / "transform_arrays.npz"
        if t_arrays_path.exists():
            arrays = dict(np.load(t_arrays_path, allow_pickle=False))
            arrays.pop("_empty", None)
            if arrays:
                chain.load_state_arrays(arrays)
        m_cfg = read_json(d / "model_config.json")
        m_arrays = dict(np.load(d / "model_arrays.npz", allow_pickle=False))
        model = rebuild_model(m_cfg, m_arrays)
        return chain, model

    def list_layers(self) -> List[int]:
        layers_root = self.root / "layers"
        if not layers_root.exists():
            return []
        out: List[int] = []
        for d in sorted(layers_root.iterdir()):
            if not d.is_dir():
                continue
            layer_num = _parse_layer_num(d.name)
            if layer_num is not None:
                out.append(layer_num)
        return out

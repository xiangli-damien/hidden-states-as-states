from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from hss import MemmapProvider, NumpyProvider
from hss.types import StateProvider

from .config import DataConfig


@dataclass
class LoadedInputs:
    provider: StateProvider
    y: np.ndarray
    label_dict: Dict[str, np.ndarray]
    layers: List[int]
    n_items: int
    state_dim: int


def _provider_metadata_from_labels(label_dict: Dict[str, np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
    row_ids = None
    for key in ("item_id", "row_id", "row_ids", "sentence_row", "token_row", "zarr_row"):
        if key in label_dict:
            row_ids = np.asarray(label_dict[key], dtype=np.int64)
            break
    sample_ids = None
    if "sample_id" in label_dict:
        sample_ids = np.asarray(label_dict["sample_id"], dtype=np.int64)
    metadata_fields: Dict[str, np.ndarray] = {}
    for key, value in label_dict.items():
        if key in {"label", "is_correct", "correct", "sample_id"}:
            continue
        arr = np.asarray(value)
        if arr.ndim == 1:
            metadata_fields[str(key)] = arr
    return row_ids, sample_ids, metadata_fields


def load_from_openact(cfg: DataConfig) -> Tuple[NumpyProvider, np.ndarray, Dict[str, np.ndarray]]:
    from openact_core import DataLoader, Run

    loader = DataLoader(Run(cfg.run_path), only_valid=cfg.only_valid)
    if cfg.n_points is not None:
        data = loader.load_trajectories(n_points=cfg.n_points, layers=cfg.layers, labels=cfg.labels, dtype=cfg.dtype)
    elif cfg.start_pct is not None or cfg.end_pct is not None:
        data = loader.load_range(start_pct=cfg.start_pct, end_pct=cfg.end_pct, reduction=cfg.reduction, layers=cfg.layers, labels=cfg.labels, dtype=cfg.dtype)
    else:
        data = loader.load(reduction=cfg.reduction, layers=cfg.layers, labels=cfg.labels, dtype=cfg.dtype)
    layer_ids = cfg.layers or list(range(data.hidden_states.shape[-2]))
    label_dict: Dict[str, np.ndarray] = {}
    if hasattr(data, "labels") and data.labels is not None:
        for key, value in data.labels.items():
            label_dict[key] = np.asarray(value)
    y = _extract_binary_label(label_dict, cfg.positive_label_key)
    row_ids, sample_ids, metadata_fields = _provider_metadata_from_labels(label_dict)
    provider = NumpyProvider(
        data.hidden_states,
        layer_ids=layer_ids,
        row_ids=row_ids,
        sample_ids=sample_ids,
        metadata_fields=metadata_fields,
        unit=str(getattr(cfg, "unit", "item") or "item"),
        provider_manifest={"source": "openact", "run_path": cfg.run_path},
    )
    return provider, y, label_dict


def load_from_numpy(
    states: np.ndarray,
    *,
    layer_ids: Optional[List[int]] = None,
    labels: Optional[Dict[str, np.ndarray]] = None,
    positive_label_key: str = "correct",
    unit: str = "item",
) -> Tuple[NumpyProvider, np.ndarray, Dict[str, np.ndarray]]:
    label_dict = labels or {}
    y = _extract_binary_label(label_dict, positive_label_key)
    row_ids, sample_ids, metadata_fields = _provider_metadata_from_labels(label_dict)
    provider = NumpyProvider(
        states,
        layer_ids=layer_ids,
        row_ids=row_ids,
        sample_ids=sample_ids,
        metadata_fields=metadata_fields,
        unit=str(unit),
    )
    return provider, y, label_dict


def load_from_memmap(
    path: Union[str, Path],
    *,
    layer_ids: Optional[List[int]] = None,
    labels: Optional[Dict[str, np.ndarray]] = None,
    positive_label_key: str = "correct",
    unit: str = "item",
) -> Tuple[MemmapProvider, np.ndarray, Dict[str, np.ndarray]]:
    label_dict = labels or {}
    y = _extract_binary_label(label_dict, positive_label_key)
    row_ids, sample_ids, metadata_fields = _provider_metadata_from_labels(label_dict)
    provider = MemmapProvider(
        path,
        layer_ids=layer_ids,
        row_ids=row_ids,
        sample_ids=sample_ids,
        metadata_fields=metadata_fields,
        unit=str(unit),
    )
    return provider, y, label_dict


def load_inputs(cfg: DataConfig) -> LoadedInputs:
    source = cfg.source.lower()
    if source == "openact":
        provider, y, label_dict = load_from_openact(cfg)
    elif source == "memmap":
        path = cfg.memmap_path or cfg.run_path
        label_dict = {}
        labels_path = getattr(cfg, "labels_path", "") or ""
        if labels_path:
            labels_obj = np.load(labels_path, allow_pickle=False)
            label_dict = {str(key): np.asarray(labels_obj[key]) for key in labels_obj.files}
        provider, y, label_dict = load_from_memmap(
            path,
            layer_ids=cfg.layers,
            labels=label_dict,
            positive_label_key=cfg.positive_label_key,
            unit=str(getattr(cfg, "unit", "item") or "item"),
        )
    else:
        raise ValueError(f"Unsupported data source: {cfg.source}")
    layers = cfg.layers or [int(layer) for layer in provider.layers()]
    return LoadedInputs(
        provider=provider,
        y=y,
        label_dict=label_dict,
        layers=layers,
        n_items=provider.n_items(),
        state_dim=provider.state_dim(),
    )


def _extract_binary_label(label_dict: Dict[str, np.ndarray], key: str = "correct") -> np.ndarray:
    if key not in label_dict:
        for existing in label_dict:
            low = existing.lower()
            if key.lower() in low or "correct" in low or "safe" in low:
                key = existing
                break
    if key not in label_dict:
        return np.array([], dtype=np.int32)
    raw = np.asarray(label_dict[key])
    return np.nan_to_num(raw, nan=0.0).astype(bool).astype(np.int32)


def discover_runs(base_dir: Union[str, Path]) -> Dict[str, Path]:
    base = Path(base_dir)
    return {d.name: d for d in sorted(base.iterdir()) if d.is_dir() and (d / "manifest.json").exists()}

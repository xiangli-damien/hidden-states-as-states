from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np


try:
    from hss.types import Batch
except Exception:
    @dataclass(frozen=True)
    class Batch:
        states: np.ndarray
        ids: Optional[np.ndarray] = None
        sample_ids: Optional[np.ndarray] = None
        fields: Dict[str, np.ndarray] = field(default_factory=dict)


class IndexedNpyProvider:
    """Memmap-backed provider for bundle views.

    Important detail: ``Batch.ids`` are the stable ``item_id`` values from the
    bundle index, not relative positions inside the current selection. That is
    exactly the provenance HSS should persist for sentence/token runs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        available_layers: Sequence[int],
        row_indices: Optional[np.ndarray] = None,
        selected_layers: Optional[Sequence[int]] = None,
        sample_ids: Optional[np.ndarray] = None,
        metadata_fields: Optional[Dict[str, np.ndarray]] = None,
        unit: str = "item",
        provider_manifest: Optional[Dict[str, Any]] = None,
        mmap_mode: str = "r",
    ) -> None:
        self.path = str(path)
        self._arr = np.load(self.path, mmap_mode=mmap_mode)
        if self._arr.ndim != 3:
            raise ValueError(f"Expected a 3D states array, found shape {self._arr.shape}")
        self._available_layers = [int(x) for x in available_layers]
        if len(self._available_layers) != int(self._arr.shape[1]):
            raise ValueError("available_layers length does not match array shape")
        self._layer_to_pos = {int(layer): pos for pos, layer in enumerate(self._available_layers)}
        self._selected_layers = [int(x) for x in (selected_layers if selected_layers is not None else self._available_layers)]
        missing = [x for x in self._selected_layers if x not in self._layer_to_pos]
        if missing:
            raise ValueError(f"Unknown layers requested: {missing}")
        if row_indices is None:
            self._row_indices = np.arange(int(self._arr.shape[0]), dtype=np.int64)
        else:
            idx = np.asarray(row_indices, dtype=np.int64).ravel()
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= int(self._arr.shape[0])):
                raise IndexError("row_indices are out of bounds")
            self._row_indices = idx
        self._sample_ids = None if sample_ids is None else np.asarray(sample_ids, dtype=np.int64).ravel()
        if self._sample_ids is not None and int(self._sample_ids.shape[0]) != int(self._row_indices.shape[0]):
            raise ValueError("sample_ids length does not match selected row count")
        self._metadata_fields: Dict[str, np.ndarray] = {}
        for key, value in dict(metadata_fields or {}).items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if int(arr.shape[0]) != int(self._row_indices.shape[0]):
                raise ValueError(f"metadata field '{key}' length does not match selected row count")
            self._metadata_fields[str(key)] = arr
        self._unit = str(unit)
        self._provider_manifest = dict(provider_manifest or {})

    def layers(self) -> Sequence[int]:
        return list(self._selected_layers)

    def n_items(self) -> int:
        return int(self._row_indices.shape[0])

    def state_dim(self) -> int:
        return int(self._arr.shape[2])

    def unit(self) -> str:
        return self._unit

    def metadata_manifest(self) -> Dict[str, Any]:
        return dict(self._provider_manifest)

    def row_ids(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            return self._row_indices.copy()
        rel = np.asarray(indices, dtype=np.int64).ravel()
        return self._row_indices[rel]

    def sample_ids(self, indices: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self._sample_ids is None:
            return None
        if indices is None:
            return self._sample_ids.copy()
        rel = np.asarray(indices, dtype=np.int64).ravel()
        return self._sample_ids[rel]

    def metadata_fields(self, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if indices is None:
            return {key: value.copy() for key, value in self._metadata_fields.items()}
        rel = np.asarray(indices, dtype=np.int64).ravel()
        return {key: value[rel] for key, value in self._metadata_fields.items()}

    def absolute_rows(self) -> np.ndarray:
        return self._row_indices.copy()

    def subset(
        self,
        row_indices: Optional[np.ndarray] = None,
        selected_layers: Optional[Sequence[int]] = None,
    ) -> "IndexedNpyProvider":
        if row_indices is None:
            next_rows = self._row_indices
            next_sample_ids = self._sample_ids
            next_fields = self._metadata_fields
        else:
            rel = np.asarray(row_indices, dtype=np.int64).ravel()
            if rel.size and (int(rel.min()) < 0 or int(rel.max()) >= int(self._row_indices.shape[0])):
                raise IndexError("subset indices are out of bounds")
            next_rows = self._row_indices[rel]
            next_sample_ids = None if self._sample_ids is None else self._sample_ids[rel]
            next_fields = {key: value[rel] for key, value in self._metadata_fields.items()}
        return IndexedNpyProvider(
            self.path,
            available_layers=self._available_layers,
            row_indices=next_rows,
            selected_layers=self._selected_layers if selected_layers is None else selected_layers,
            sample_ids=next_sample_ids,
            metadata_fields=next_fields,
            unit=self._unit,
            provider_manifest=self._provider_manifest,
        )

    def iter_batches(
        self,
        *,
        layer: int,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> Iterator[Batch]:
        layer = int(layer)
        if layer not in self._layer_to_pos:
            raise ValueError(f"Layer {layer} not found")
        pos = self._layer_to_pos[layer]
        rel = np.arange(self._row_indices.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64).ravel()
        if rel.size and (int(rel.min()) < 0 or int(rel.max()) >= int(self._row_indices.shape[0])):
            raise IndexError("indices are out of bounds")
        abs_rows = self._row_indices[rel]
        block = max(1, int(batch_size))
        for start in range(0, int(rel.shape[0]), block):
            end = min(int(rel.shape[0]), start + block)
            rel_chunk = rel[start:end]
            abs_chunk = abs_rows[start:end]
            states = np.asarray(self._arr[abs_chunk, pos, :], dtype=np.float32)
            sample_ids = None if self._sample_ids is None else self._sample_ids[rel_chunk]
            fields = {key: value[rel_chunk] for key, value in self._metadata_fields.items()}
            yield Batch(states=states, ids=abs_chunk, sample_ids=sample_ids, fields=fields)

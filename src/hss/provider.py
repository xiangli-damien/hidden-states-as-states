from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .types import Batch
from .utils import i64, iter_slices, sanitize_metadata_fields


class _ArrayProviderBase:
    _arr: np.ndarray
    _layer_ids: List[int]
    _layer_to_pos: Dict[int, int]
    _row_ids: np.ndarray
    _sample_ids: Optional[np.ndarray]
    _metadata_fields: Dict[str, np.ndarray]
    _unit: str
    _provider_manifest: Dict[str, Any]

    def _init_array(
        self,
        arr: np.ndarray,
        layer_ids: Optional[Sequence[int]],
        *,
        row_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        metadata_fields: Optional[Dict[str, np.ndarray]] = None,
        unit: str = "item",
        provider_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        if arr.ndim != 3:
            raise ValueError(f"Expected (N, L, D) array, got shape {arr.shape}")
        self._arr = arr
        self._layer_ids = list(layer_ids) if layer_ids is not None else list(range(arr.shape[1]))
        if len(self._layer_ids) != int(arr.shape[1]):
            raise ValueError("layer_ids length != array.shape[1]")
        self._layer_to_pos = {int(layer): int(pos) for pos, layer in enumerate(self._layer_ids)}
        n_items = int(arr.shape[0])
        self._row_ids = np.arange(n_items, dtype=np.int64) if row_ids is None else i64(np.asarray(row_ids).ravel())
        if int(self._row_ids.shape[0]) != n_items:
            raise ValueError("row_ids length != array.shape[0]")
        self._sample_ids = None
        if sample_ids is not None:
            sample_ids_arr = i64(np.asarray(sample_ids).ravel())
            if int(sample_ids_arr.shape[0]) != n_items:
                raise ValueError("sample_ids length != array.shape[0]")
            self._sample_ids = sample_ids_arr
        self._metadata_fields = sanitize_metadata_fields(metadata_fields, expected_len=n_items)
        self._unit = str(unit)
        self._provider_manifest = dict(provider_manifest or {})

    def layers(self) -> Sequence[int]:
        return list(self._layer_ids)

    def n_items(self) -> int:
        return int(self._arr.shape[0])

    def state_dim(self) -> int:
        return int(self._arr.shape[2])

    @property
    def path(self) -> Optional[str]:
        return None

    def unit(self) -> str:
        return self._unit

    def metadata_manifest(self) -> Dict[str, Any]:
        return dict(self._provider_manifest)

    def row_ids(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            return self._row_ids.copy()
        idx = i64(np.asarray(indices).ravel())
        return self._row_ids[idx]

    def sample_ids(self, indices: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self._sample_ids is None:
            return None
        if indices is None:
            return self._sample_ids.copy()
        idx = i64(np.asarray(indices).ravel())
        return self._sample_ids[idx]

    def metadata_fields(self, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if indices is None:
            return {key: value.copy() for key, value in self._metadata_fields.items()}
        idx = i64(np.asarray(indices).ravel())
        return {key: value[idx] for key, value in self._metadata_fields.items()}

    def _read_block(self, slc: slice, pos: int) -> np.ndarray:
        return np.asarray(self._arr[slc, pos, :])

    def _read_indices(self, idx: np.ndarray, pos: int) -> np.ndarray:
        return np.asarray(self._arr[idx, pos, :])

    def iter_batches(
        self,
        *,
        layer: int,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> Iterator[Batch]:
        pos = self._layer_to_pos.get(int(layer))
        if pos is None:
            raise ValueError(f"Layer {layer} not found")
        block = max(1, int(batch_size))
        if indices is None:
            rel = np.arange(self.n_items(), dtype=np.int64)
        else:
            rel = i64(np.asarray(indices).ravel())
        for start, end in iter_slices(int(rel.shape[0]), block):
            rel_chunk = rel[start:end]
            if indices is None:
                states = self._read_block(slice(start, end), pos)
            else:
                states = self._read_indices(rel_chunk, pos)
            sample_ids = None if self._sample_ids is None else self._sample_ids[rel_chunk]
            fields = {key: value[rel_chunk] for key, value in self._metadata_fields.items()}
            yield Batch(
                states=states,
                ids=self._row_ids[rel_chunk],
                sample_ids=sample_ids,
                fields=fields,
            )


class NumpyProvider(_ArrayProviderBase):
    def __init__(
        self,
        states: np.ndarray,
        *,
        layer_ids: Optional[Sequence[int]] = None,
        row_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        metadata_fields: Optional[Dict[str, np.ndarray]] = None,
        unit: str = "item",
        provider_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._init_array(
            np.asarray(states),
            layer_ids,
            row_ids=row_ids,
            sample_ids=sample_ids,
            metadata_fields=metadata_fields,
            unit=unit,
            provider_manifest=provider_manifest,
        )


class MemmapProvider(_ArrayProviderBase):
    def __init__(
        self,
        path: str | Path,
        *,
        layer_ids: Optional[Sequence[int]] = None,
        mmap_mode: str = "r",
        row_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        metadata_fields: Optional[Dict[str, np.ndarray]] = None,
        unit: str = "item",
        provider_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._path = str(path)
        arr = np.load(self._path, mmap_mode=mmap_mode)
        self._init_array(
            arr,
            layer_ids,
            row_ids=row_ids,
            sample_ids=sample_ids,
            metadata_fields=metadata_fields,
            unit=unit,
            provider_manifest=provider_manifest,
        )

    @property
    def path(self) -> Optional[str]:
        return self._path


class CallableProvider:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        n_items: int,
        state_dim: int,
        iter_batches_fn: Callable[[int, Optional[np.ndarray], int], Iterator[Batch | np.ndarray]],
        row_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        metadata_fields: Optional[Dict[str, np.ndarray]] = None,
        unit: str = "item",
        provider_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._layers = [int(x) for x in layers]
        self._n_items = int(n_items)
        self._state_dim = int(state_dim)
        self._iter_fn = iter_batches_fn
        self._row_ids = np.arange(self._n_items, dtype=np.int64) if row_ids is None else i64(np.asarray(row_ids).ravel())
        if int(self._row_ids.shape[0]) != self._n_items:
            raise ValueError("row_ids length != n_items")
        self._sample_ids = None if sample_ids is None else i64(np.asarray(sample_ids).ravel())
        if self._sample_ids is not None and int(self._sample_ids.shape[0]) != self._n_items:
            raise ValueError("sample_ids length != n_items")
        self._metadata_fields = sanitize_metadata_fields(metadata_fields, expected_len=self._n_items)
        self._unit = str(unit)
        self._provider_manifest = dict(provider_manifest or {})

    def layers(self) -> Sequence[int]:
        return list(self._layers)

    def n_items(self) -> int:
        return self._n_items

    def state_dim(self) -> int:
        return self._state_dim

    def unit(self) -> str:
        return self._unit

    def metadata_manifest(self) -> Dict[str, Any]:
        return dict(self._provider_manifest)

    def row_ids(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            return self._row_ids.copy()
        idx = i64(np.asarray(indices).ravel())
        return self._row_ids[idx]

    def sample_ids(self, indices: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self._sample_ids is None:
            return None
        if indices is None:
            return self._sample_ids.copy()
        idx = i64(np.asarray(indices).ravel())
        return self._sample_ids[idx]

    def metadata_fields(self, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if indices is None:
            return {key: value.copy() for key, value in self._metadata_fields.items()}
        idx = i64(np.asarray(indices).ravel())
        return {key: value[idx] for key, value in self._metadata_fields.items()}

    def iter_batches(
        self,
        *,
        layer: int,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> Iterator[Batch]:
        for payload in self._iter_fn(int(layer), indices, int(batch_size)):
            if isinstance(payload, Batch):
                row_ids = payload.ids
                if row_ids is None:
                    if indices is None:
                        raise ValueError(
                            "CallableProvider received a Batch without ids; provide ids or use row_ids metadata."
                        )
                    rel = i64(np.asarray(indices).ravel())
                    row_ids = self._row_ids[rel[: len(payload.states)]]
                sample_ids = payload.sample_ids
                if sample_ids is None and self._sample_ids is not None:
                    mapping = {int(v): i for i, v in enumerate(self._row_ids.tolist())}
                    rel_idx = np.array([mapping[int(v)] for v in np.asarray(row_ids).ravel()], dtype=np.int64)
                    sample_ids = self._sample_ids[rel_idx]
                fields = dict(payload.fields)
                if not fields and self._metadata_fields:
                    mapping = {int(v): i for i, v in enumerate(self._row_ids.tolist())}
                    rel_idx = np.array([mapping[int(v)] for v in np.asarray(row_ids).ravel()], dtype=np.int64)
                    fields = {key: value[rel_idx] for key, value in self._metadata_fields.items()}
                yield Batch(states=payload.states, ids=np.asarray(row_ids), sample_ids=sample_ids, fields=fields)
            else:
                raise TypeError(
                    "CallableProvider.iter_batches_fn must yield Batch objects in the metadata-aware implementation."
                )

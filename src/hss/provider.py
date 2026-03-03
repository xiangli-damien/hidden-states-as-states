from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .types import Batch
from .utils import i64, iter_slices


class _ArrayProviderBase:

    _arr: np.ndarray
    _layer_ids: List[int]
    _layer_to_pos: Dict[int, int]

    def _init_array(
        self, arr: np.ndarray, layer_ids: Optional[Sequence[int]]
    ) -> None:
        if arr.ndim != 3:
            raise ValueError(f"Expected (N, L, D) array, got shape {arr.shape}")
        self._arr = arr
        self._layer_ids = (
            list(layer_ids) if layer_ids is not None else list(range(arr.shape[1]))
        )
        if len(self._layer_ids) != arr.shape[1]:
            raise ValueError("layer_ids length != array.shape[1]")
        self._layer_to_pos = {
            int(lid): int(pos) for pos, lid in enumerate(self._layer_ids)
        }

    def layers(self) -> Sequence[int]:
        return list(self._layer_ids)

    def n_items(self) -> int:
        return int(self._arr.shape[0])

    def state_dim(self) -> int:
        return int(self._arr.shape[2])

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
        bs = max(1, int(batch_size))
        if indices is None:
            n = self.n_items()
            for i, j in iter_slices(n, bs):
                ids = np.arange(i, j, dtype=np.int64)
                yield Batch(states=self._read_block(slice(i, j), pos), ids=ids)
        else:
            idx = i64(np.asarray(indices).ravel())
            for i, j in iter_slices(len(idx), bs):
                chunk = idx[i:j]
                yield Batch(states=self._read_indices(chunk, pos), ids=chunk)


class NumpyProvider(_ArrayProviderBase):
    def __init__(
        self,
        states: np.ndarray,
        *,
        layer_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self._init_array(np.asarray(states), layer_ids)


class MemmapProvider(_ArrayProviderBase):
    def __init__(
        self,
        path: str | Path,
        *,
        layer_ids: Optional[Sequence[int]] = None,
        mmap_mode: str = "r",
    ) -> None:
        self._path = str(path)
        arr = np.load(self._path, mmap_mode=mmap_mode)
        self._init_array(arr, layer_ids)


class CallableProvider:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        n_items: int,
        state_dim: int,
        iter_batches_fn: Callable[[int, Optional[np.ndarray], int], Iterator[Batch]],
    ) -> None:
        self._layers = [int(x) for x in layers]
        self._n_items = int(n_items)
        self._state_dim = int(state_dim)
        self._iter_fn = iter_batches_fn

    def layers(self) -> Sequence[int]:
        return list(self._layers)

    def n_items(self) -> int:
        return self._n_items

    def state_dim(self) -> int:
        return self._state_dim

    def iter_batches(
        self,
        *,
        layer: int,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> Iterator[Batch]:
        return self._iter_fn(int(layer), indices, int(batch_size))
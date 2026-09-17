"""Read-only, per-layer float32 memmap contract shared by all data adapters."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..types import Batch


class CachedStates:
    def __init__(self, path):
        self.path = Path(path)
        self.info = json.loads((self.path / "_SUCCESS.json").read_text())
        self.meta = pd.read_parquet(self.path / "rows.parquet")
        self._arrays = {}
        if len(self.meta) != self.info["n_rows"]:
            raise ValueError("Cache row metadata is incomplete")

    def layers(self):
        return self.info["layers"]

    def n_items(self):
        return self.info["n_rows"]

    def state_dim(self):
        return self.info["hidden_dim"]

    def array(self, layer):
        if layer not in self.layers():
            raise ValueError(f"Layer {layer} is not in this snapshot")
        if layer not in self._arrays:
            arr = np.load(
                self.path / f"layer_{layer}.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                arr.shape != (self.n_items(), self.state_dim())
                or arr.dtype != np.float32
            ):
                raise ValueError("Incomplete or inconsistent layer cache")
            self._arrays[layer] = arr
        return self._arrays[layer]

    def iter_batches(self, *, layer, indices=None, batch_size=4096):
        indices = np.arange(self.n_items()) if indices is None else np.asarray(indices)
        arr = self.array(layer)
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            yield Batch(
                arr[idx], ids=idx, sample_ids=self.meta.group_id.to_numpy()[idx]
            )

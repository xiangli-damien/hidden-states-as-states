from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionSpec:
    item_ids: Optional[Sequence[int]] = None
    sample_ids: Optional[Sequence[int]] = None
    query: Optional[str] = None
    head: Optional[int] = None
    tail: Optional[int] = None
    frac: Optional[float] = None
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_ids": None if self.item_ids is None else [int(x) for x in self.item_ids],
            "sample_ids": None if self.sample_ids is None else [int(x) for x in self.sample_ids],
            "query": self.query,
            "head": self.head,
            "tail": self.tail,
            "frac": self.frac,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> SelectionSpec:
        return cls(
            item_ids=payload.get("item_ids"),
            sample_ids=payload.get("sample_ids"),
            query=payload.get("query"),
            head=payload.get("head"),
            tail=payload.get("tail"),
            frac=payload.get("frac"),
            random_state=int(payload.get("random_state", 42)),
        )


@dataclass(frozen=True)
class ViewManifest:
    name: str
    unit: str
    states_path: str
    index_path: str
    n_items: int
    n_layers: int
    hidden_dim: int
    dtype: str
    layer_ids: List[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["layer_ids"] = [int(x) for x in self.layer_ids]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> ViewManifest:
        return cls(
            name=str(payload["name"]),
            unit=str(payload["unit"]),
            states_path=str(payload["states_path"]),
            index_path=str(payload["index_path"]),
            n_items=int(payload["n_items"]),
            n_layers=int(payload["n_layers"]),
            hidden_dim=int(payload["hidden_dim"]),
            dtype=str(payload["dtype"]),
            layer_ids=[int(x) for x in payload.get("layer_ids", [])],
            extra=dict(payload.get("extra", {})),
        )


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int
    bundle_type: str
    source: Dict[str, Any]
    sample_table_path: str
    views: Dict[str, str]
    created_at: str
    selection: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "bundle_type": str(self.bundle_type),
            "source": dict(self.source),
            "sample_table_path": str(self.sample_table_path),
            "views": {str(k): str(v) for k, v in self.views.items()},
            "created_at": str(self.created_at),
            "selection": self.selection,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> BundleManifest:
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            bundle_type=str(payload.get("bundle_type", "hss_bundle")),
            source=dict(payload.get("source", {})),
            sample_table_path=str(payload["sample_table_path"]),
            views={str(k): str(v) for k, v in dict(payload.get("views", {})).items()},
            created_at=str(payload.get("created_at", "")),
            selection=payload.get("selection"),
            extra=dict(payload.get("extra", {})),
        )


@dataclass
class BridgeLoadedInputs:
    provider: Any
    y: np.ndarray
    label_dict: Dict[str, np.ndarray]
    layers: List[int]
    n_items: int
    state_dim: int
    item_table: pd.DataFrame
    sample_table: pd.DataFrame
    view_manifest: ViewManifest
    bundle_manifest: BundleManifest

    def to_experiments_loaded_inputs(self) -> Any:
        from experiments.data import LoadedInputs

        return LoadedInputs(
            provider=self.provider,
            y=self.y,
            label_dict=self.label_dict,
            layers=list(self.layers),
            n_items=int(self.n_items),
            state_dim=int(self.state_dim),
        )

    def joined_items(self, sample_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if sample_columns is None:
            preferred = ["sample_id", "is_correct", "zarr_row", "generated_text", "prompt", "question", "answer_token_count"]
            sample_columns = [c for c in preferred if c in self.sample_table.columns]
        cols = [c for c in sample_columns if c in self.sample_table.columns]
        base = self.sample_table[cols].drop_duplicates(subset=["sample_id"], keep="last") if "sample_id" in cols else self.sample_table[cols].copy()
        return self.item_table.merge(base, on="sample_id", how="left")

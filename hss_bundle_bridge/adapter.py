from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bundle import load_frame, numeric_columns, read_bundle_manifest, read_view_manifest
from .protocol import BridgeLoadedInputs, BundleManifest, SelectionSpec, ViewManifest
from .provider import IndexedNpyProvider


@dataclass
class OpenedView:
    provider: IndexedNpyProvider
    y: np.ndarray
    label_dict: Dict[str, np.ndarray]
    layers: List[int]
    n_items: int
    state_dim: int
    item_table: pd.DataFrame
    sample_table: pd.DataFrame
    view_manifest: ViewManifest
    bundle_manifest: BundleManifest

    def to_bridge_loaded_inputs(self) -> BridgeLoadedInputs:
        return BridgeLoadedInputs(
            provider=self.provider,
            y=self.y,
            label_dict=self.label_dict,
            layers=list(self.layers),
            n_items=int(self.n_items),
            state_dim=int(self.state_dim),
            item_table=self.item_table.copy(),
            sample_table=self.sample_table.copy(),
            view_manifest=self.view_manifest,
            bundle_manifest=self.bundle_manifest,
        )

    def to_experiments_loaded_inputs(self) -> Any:
        return self.to_bridge_loaded_inputs().to_experiments_loaded_inputs()

    def joined_items(self, sample_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return self.to_bridge_loaded_inputs().joined_items(sample_columns=sample_columns)


class BundleAdapter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.bundle_manifest = read_bundle_manifest(self.root)
        self.sample_table = load_frame(self.root / self.bundle_manifest.sample_table_path)
        if "sample_id" not in self.sample_table.columns:
            raise ValueError("sample table must contain a sample_id column")
        self.sample_table = self.sample_table.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)

    def list_views(self) -> List[str]:
        return sorted(self.bundle_manifest.views.keys())

    def view_manifest(self, name: str) -> ViewManifest:
        if name not in self.bundle_manifest.views:
            raise KeyError(f"Unknown view: {name}")
        return read_view_manifest(self.root, self.bundle_manifest.views[name])

    def item_table_for_view(self, name: str) -> pd.DataFrame:
        manifest = self.view_manifest(name)
        return load_frame(self.root / manifest.index_path)

    def open_view(
        self,
        name: str,
        *,
        selection: Optional[SelectionSpec] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> OpenedView:
        manifest = self.view_manifest(name)
        item_table = load_frame(self.root / manifest.index_path)
        if "item_id" not in item_table.columns or "sample_id" not in item_table.columns:
            raise ValueError("view index must contain item_id and sample_id columns")
        item_table = item_table.sort_values("item_id").reset_index(drop=True)
        selected_items = self._apply_selection(item_table, self.sample_table, selection)
        selected_items = selected_items.sort_values("item_id").reset_index(drop=True)

        sample_ids = selected_items["sample_id"].drop_duplicates().tolist()
        selected_samples = self.sample_table[self.sample_table["sample_id"].isin(sample_ids)].copy().reset_index(drop=True)
        sample_lookup = selected_samples.set_index("sample_id")

        if "is_correct" not in sample_lookup.columns:
            if "label" in sample_lookup.columns:
                y = sample_lookup.loc[selected_items["sample_id"], "label"].to_numpy(dtype=np.int32)
            else:
                y = np.zeros(len(selected_items), dtype=np.int32)
        else:
            y = sample_lookup.loc[selected_items["sample_id"], "is_correct"].to_numpy(dtype=np.int32)

        metadata_fields = self._build_provider_metadata_fields(selected_items, selected_samples)
        provider_manifest = {
            "bundle_root": str(self.root),
            "view_name": str(name),
            "view_unit": str(manifest.unit),
            "view_index_path": str((self.root / manifest.index_path).resolve()),
            "sample_table_path": str((self.root / self.bundle_manifest.sample_table_path).resolve()),
        }
        provider = IndexedNpyProvider(
            self.root / manifest.states_path,
            available_layers=manifest.layer_ids,
            row_indices=selected_items["item_id"].to_numpy(dtype=np.int64),
            selected_layers=manifest.layer_ids if layers is None else [int(x) for x in layers],
            sample_ids=selected_items["sample_id"].to_numpy(dtype=np.int64),
            metadata_fields=metadata_fields,
            unit=str(manifest.unit),
            provider_manifest=provider_manifest,
        )
        label_dict = self._build_label_dict(selected_items, selected_samples, y)
        layer_ids = list(provider.layers())
        return OpenedView(
            provider=provider,
            y=y,
            label_dict=label_dict,
            layers=layer_ids,
            n_items=provider.n_items(),
            state_dim=provider.state_dim(),
            item_table=selected_items,
            sample_table=selected_samples,
            view_manifest=manifest,
            bundle_manifest=self.bundle_manifest,
        )

    def _apply_selection(
        self,
        item_table: pd.DataFrame,
        sample_table: pd.DataFrame,
        selection: Optional[SelectionSpec],
    ) -> pd.DataFrame:
        if selection is None:
            return item_table.copy()
        out = item_table.copy()
        if selection.item_ids is not None:
            wanted = {int(x) for x in selection.item_ids}
            out = out[out["item_id"].isin(wanted)]
        if selection.sample_ids is not None:
            wanted = {int(x) for x in selection.sample_ids}
            out = out[out["sample_id"].isin(wanted)]
        if selection.query:
            merged = out.merge(sample_table, on="sample_id", how="left", suffixes=("", "__sample"))
            merged = merged.query(selection.query, engine="python")
            out = out[out["item_id"].isin(merged["item_id"].tolist())]
        if selection.frac is not None:
            frac = float(selection.frac)
            if frac <= 0:
                out = out.iloc[0:0].copy()
            elif frac < 1.0 and len(out) > 0:
                out = out.sample(frac=frac, random_state=int(selection.random_state)).sort_values("item_id")
        if selection.head is not None:
            out = out.head(int(selection.head))
        if selection.tail is not None:
            out = out.tail(int(selection.tail))
        return out.reset_index(drop=True)

    def _build_provider_metadata_fields(
        self,
        item_table: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        fields: Dict[str, np.ndarray] = {}
        for column in numeric_columns(item_table, exclude={"item_id", "sample_id"}):
            fields[column] = item_table[column].to_numpy()
        if len(sample_table) > 0:
            lookup = sample_table.set_index("sample_id")
            for column in numeric_columns(sample_table, exclude={"sample_id", "label", "is_correct"}):
                fields[f"sample__{column}"] = lookup.loc[item_table["sample_id"], column].to_numpy()
        return fields

    def _build_label_dict(
        self,
        item_table: pd.DataFrame,
        sample_table: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        labels: Dict[str, np.ndarray] = {
            "label": np.asarray(y, dtype=np.int32),
            "is_correct": np.asarray(y, dtype=np.int32),
            "item_id": item_table["item_id"].to_numpy(dtype=np.int64),
            "sample_id": item_table["sample_id"].to_numpy(dtype=np.int64),
        }
        for column in numeric_columns(item_table, exclude={"item_id", "sample_id"}):
            labels[column] = item_table[column].to_numpy()
        if len(sample_table) > 0:
            lookup = sample_table.set_index("sample_id")
            for column in numeric_columns(sample_table, exclude={"sample_id", "is_correct", "label"}):
                labels[column] = lookup.loc[item_table["sample_id"], column].to_numpy()
        return labels

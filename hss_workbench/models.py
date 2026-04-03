from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from experiments.core import BaseArtifacts
from experiments.data import LoadedInputs
from experiments.discretize import DiscretizeResult, ScanResult, SelectResult
from experiments.io_utils import ExperimentStore
from experiments.k_analysis import KAnalysisResult

from .store import ArtifactManifest


@dataclass
class DataSession:
    loaded: LoadedInputs
    manifest: Dict[str, Any]

    @property
    def provider(self):
        return self.loaded.provider

    @property
    def y(self) -> np.ndarray:
        return self.loaded.y

    @property
    def label_dict(self) -> Dict[str, np.ndarray]:
        return self.loaded.label_dict

    @property
    def layers(self) -> list[int]:
        return list(self.loaded.layers)


@dataclass
class ScanArtifact:
    name: str
    root: Path
    manifest: ArtifactManifest
    store: ExperimentStore
    result: ScanResult

    @property
    def metrics_df(self):
        return self.result.metrics_df

    @property
    def layers(self):
        return self.result.layers

    @property
    def config(self):
        return self.result.config


@dataclass
class SelectArtifact:
    name: str
    root: Path
    manifest: ArtifactManifest
    store: ExperimentStore
    result: SelectResult
    k_analysis: Optional[KAnalysisResult] = None

    @property
    def k_map(self):
        return self.result.k_map

    @property
    def strategy(self):
        return self.result.strategy

    @property
    def config(self):
        return self.result.config


@dataclass
class DiscretizeArtifact:
    name: str
    root: Path
    manifest: ArtifactManifest
    store: ExperimentStore
    result: DiscretizeResult
    requested_k_map: Optional[Dict[int, int]] = None

    @property
    def hss_result(self):
        return self.result.hss_result

    @property
    def global_labels(self):
        return self.result.hss_result.global_labels

    @property
    def k_map(self):
        return self.result.k_map

    @property
    def layers(self):
        return self.result.layers

    @property
    def n_items(self):
        return self.result.n_items

    @property
    def n_global_states(self):
        return self.result.n_global_states

    @property
    def config(self):
        return self.result.config


@dataclass
class AnalysisArtifact:
    name: str
    root: Path
    manifest: ArtifactManifest
    store: ExperimentStore
    result: Any = None

    @property
    def analysis_type(self) -> Optional[str]:
        return self.manifest.analysis_type

    def load_json(self, path: str) -> Any:
        return self.store.load_json(path)

    def load_csv(self, path: str):
        return self.store.load_csv(path)

    def load_npy(self, path: str):
        return self.store.load_npy(path)

    def load_parquet(self, path: str):
        return self.store.load_parquet(path)


@dataclass
class AnalysisContext:
    workspace: Any
    store: ExperimentStore
    data: Optional[DataSession]
    scan: Optional[ScanArtifact]
    select: Optional[SelectArtifact]
    discretize: Optional[DiscretizeArtifact]
    base: Optional[BaseArtifacts]
    analysis_name: str
    analysis_type: Optional[str]
    refs: Dict[str, Any]

    @property
    def provider(self):
        return None if self.data is None else self.data.loaded.provider

    @property
    def y(self):
        return np.array([], dtype=np.int32) if self.data is None else self.data.loaded.y

    @property
    def label_dict(self):
        return {} if self.data is None else self.data.loaded.label_dict

    @property
    def layers(self):
        if self.discretize is not None:
            return list(self.discretize.layers)
        if self.data is not None:
            return list(self.data.layers)
        if self.scan is not None:
            return list(self.scan.layers)
        return []

    @property
    def global_labels(self):
        if self.discretize is None:
            return None
        return self.discretize.global_labels

    @property
    def hss_result(self):
        if self.discretize is None:
            return None
        return self.discretize.hss_result

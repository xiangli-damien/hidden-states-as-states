from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config, save_config
from .io_utils import ExperimentStore


def read_config(path: str | Path) -> ExperimentConfig:
    return load_config(path)


def write_config(config: ExperimentConfig, path: str | Path) -> Path:
    return save_config(config, path)


def ensure_store(path: str | Path) -> ExperimentStore:
    return ExperimentStore(Path(path))

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import ExperimentConfig, load_config
from .runner import run_experiment


@dataclass(frozen=True)
class BatchJob:
    name: str
    config_path: str


def _run_config_path(path: str) -> tuple[str, str]:
    cfg = load_config(path)
    run_experiment(cfg)
    return cfg.name, cfg.output_dir


def run_batch(config_paths: Iterable[str | Path], *, max_workers: int = 1) -> List[Dict[str, str]]:
    paths = [str(Path(path)) for path in config_paths]
    if max_workers <= 1:
        out = []
        for path in paths:
            name, output_dir = _run_config_path(path)
            out.append({'name': name, 'output_dir': output_dir})
        return out
    out: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_config_path, path): path for path in paths}
        for future in as_completed(futures):
            name, output_dir = future.result()
            out.append({'name': name, 'output_dir': output_dir})
    return out

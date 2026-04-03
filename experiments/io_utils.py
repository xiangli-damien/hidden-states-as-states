from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


_log = logging.getLogger(__name__)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if not np.isfinite(v):
            _log.debug("Converting non-finite float %r to None for JSON", v)
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        _log.debug("Converting non-finite float %r to None for JSON", obj)
        return None
    return obj


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
    return path


@dataclass
class ExperimentStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def sub(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def config_dir(self) -> Path:
        return self.sub('config')

    @property
    def meta_dir(self) -> Path:
        return self.sub('meta')

    @property
    def logs_dir(self) -> Path:
        return self.sub('logs')

    @property
    def scan_dir(self) -> Path:
        return self.sub('scan')

    @property
    def select_dir(self) -> Path:
        return self.sub('select')

    @property
    def discretize_dir(self) -> Path:
        return self.sub('discretize')

    @property
    def preliminary_dir(self) -> Path:
        return self.sub('preliminary')

    @property
    def stability_dir(self) -> Path:
        return self.sub('stability')

    @property
    def geometry_dir(self) -> Path:
        return self.sub('geometry')

    @property
    def information_dir(self) -> Path:
        return self.sub('information')

    @property
    def prediction_dir(self) -> Path:
        return self.sub('prediction')

    @property
    def ablation_dir(self) -> Path:
        return self.sub('ablation')

    @property
    def figures_dir(self) -> Path:
        return self.sub('figures')

    def save_json(self, path: str, obj: Any, indent: int = 2) -> Path:
        p = self.root / path
        return _atomic_write_text(p, json.dumps(_to_jsonable(obj), indent=indent, sort_keys=True))

    def load_json(self, path: str) -> Any:
        return json.loads((self.root / path).read_text(encoding='utf-8'))

    def save_yaml(self, path: str, obj: Any) -> Path:
        p = self.root / path
        return _atomic_write_text(p, yaml.safe_dump(_to_jsonable(obj), sort_keys=False, allow_unicode=True))

    def load_yaml(self, path: str) -> Any:
        return yaml.safe_load((self.root / path).read_text(encoding='utf-8'))

    def save_text(self, path: str, text: str) -> Path:
        return _atomic_write_text(self.root / path, text)

    def save_csv(self, path: str, df: pd.DataFrame) -> Path:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return p

    def load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(self.root / path)

    def save_parquet(self, path: str, df: pd.DataFrame) -> Path:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(p, index=False)
            return p
        except Exception:
            fallback = p.with_suffix('.csv')
            df.to_csv(fallback, index=False)
            return fallback

    def load_parquet(self, path: str) -> pd.DataFrame:
        p = self.root / path
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        fallback = p.with_suffix('.csv')
        if fallback.exists():
            return pd.read_csv(fallback)
        return pd.read_parquet(p)

    def save_npy(self, path: str, arr: np.ndarray) -> Path:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        return p

    def load_npy(self, path: str) -> np.ndarray:
        return np.load(self.root / path, allow_pickle=False)

    def save_npz(self, path: str, **arrays: np.ndarray) -> Path:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arrays)
        return p

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        self.save_json('config/experiment.json', config_dict)
        self.save_yaml('config/experiment.yaml', config_dict)

    def save_run_meta(self, name: str, payload: Dict[str, Any]) -> Path:
        return self.save_json(f'meta/{name}.json', payload)

    def start_stage(self, stage: str) -> None:
        state = self._load_stage_state()
        state[stage] = {
            'status': 'running',
            'started_at': self._now(),
        }
        self.save_json('meta/stages.json', state)

    def finish_stage(self, stage: str, seconds: float, summary: Dict[str, Any] | None = None) -> None:
        state = self._load_stage_state()
        entry = dict(state.get(stage, {}))
        entry['status'] = 'done'
        entry['finished_at'] = self._now()
        entry['seconds'] = float(seconds)
        if summary is not None:
            entry['summary'] = _to_jsonable(summary)
        state[stage] = entry
        self.save_json('meta/stages.json', state)

    def fail_stage(self, stage: str, message: str) -> None:
        state = self._load_stage_state()
        entry = dict(state.get(stage, {}))
        entry['status'] = 'failed'
        entry['finished_at'] = self._now()
        entry['message'] = message
        state[stage] = entry
        self.save_json('meta/stages.json', state)

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def _load_stage_state(self) -> Dict[str, Any]:
        path = self.root / 'meta' / 'stages.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        return {}

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

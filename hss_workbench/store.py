from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from experiments.io_utils import ExperimentStore


_KIND_DIRS = {
    'scan': 'scans',
    'select': 'selects',
    'discretize': 'discretizations',
    'analysis': 'analyses',
}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, 'to_dict'):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:
            pass
    if is_dataclass(obj):
        return {str(k): to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)
    return path


def write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(yaml.safe_dump(to_jsonable(payload), sort_keys=False, allow_unicode=True), encoding='utf-8')
    tmp.replace(path)
    return path


@dataclass
class ArtifactManifest:
    name: str
    kind: str
    status: str = 'done'
    created_at: str = field(default_factory=now_utc)
    updated_at: str = field(default_factory=now_utc)
    analysis_type: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    refs: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'analysis_type': self.analysis_type,
            'config': to_jsonable(self.config),
            'refs': to_jsonable(self.refs),
            'summary': to_jsonable(self.summary),
            'paths': to_jsonable(self.paths),
            'extra': to_jsonable(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> ArtifactManifest:
        return cls(
            name=str(payload.get('name', '')),
            kind=str(payload.get('kind', '')),
            status=str(payload.get('status', 'done')),
            created_at=str(payload.get('created_at', now_utc())),
            updated_at=str(payload.get('updated_at', now_utc())),
            analysis_type=payload.get('analysis_type'),
            config=dict(payload.get('config', {})),
            refs=dict(payload.get('refs', {})),
            summary=dict(payload.get('summary', {})),
            paths=dict(payload.get('paths', {})),
            extra=dict(payload.get('extra', {})),
        )


@dataclass
class WorkspaceStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for kind in _KIND_DIRS:
            self.kind_dir(kind).mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.root / 'data'

    @property
    def meta_dir(self) -> Path:
        return self.root / 'meta'

    @property
    def index_dir(self) -> Path:
        return self.root / 'indexes'

    @property
    def logs_dir(self) -> Path:
        return self.root / 'logs'

    @property
    def workspace_meta_path(self) -> Path:
        return self.meta_dir / 'workspace.json'

    @property
    def data_manifest_path(self) -> Path:
        return self.data_dir / 'manifest.json'

    def kind_dir(self, kind: str) -> Path:
        if kind not in _KIND_DIRS:
            raise ValueError(f'Unknown artifact kind: {kind}')
        return self.root / _KIND_DIRS[kind]

    def artifact_root(self, kind: str, name: str) -> Path:
        return self.kind_dir(kind) / str(name)

    def index_path(self, kind: str) -> Path:
        return self.index_dir / f'{kind}.json'

    def list_names(self, kind: str) -> List[str]:
        payload = self.read_index(kind)
        return sorted(payload.keys())

    def read_index(self, kind: str) -> Dict[str, Any]:
        payload = read_json(self.index_path(kind), default={})
        return dict(payload or {})

    def write_index(self, kind: str, payload: Dict[str, Any]) -> Path:
        return write_json(self.index_path(kind), payload)

    def save_workspace_meta(self, payload: Dict[str, Any]) -> Path:
        return write_json(self.workspace_meta_path, payload)

    def load_workspace_meta(self) -> Dict[str, Any]:
        return dict(read_json(self.workspace_meta_path, default={}) or {})

    def save_data_manifest(self, payload: Dict[str, Any]) -> Path:
        return write_json(self.data_manifest_path, payload)

    def load_data_manifest(self) -> Dict[str, Any]:
        return dict(read_json(self.data_manifest_path, default={}) or {})

    def prepare_artifact(self, kind: str, name: str, *, overwrite: bool = False) -> Path:
        root = self.artifact_root(kind, name)
        if root.exists() and not overwrite:
            raise FileExistsError(f'{kind} artifact already exists: {name}')
        if root.exists() and overwrite:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def remove_artifact(self, kind: str, name: str) -> None:
        root = self.artifact_root(kind, name)
        if root.exists():
            shutil.rmtree(root)
        index = self.read_index(kind)
        if name in index:
            index.pop(name, None)
            self.write_index(kind, index)

    def save_manifest(self, kind: str, name: str, manifest: ArtifactManifest) -> Path:
        root = self.artifact_root(kind, name)
        root.mkdir(parents=True, exist_ok=True)
        payload = manifest.to_dict()
        write_json(root / 'manifest.json', payload)
        write_yaml(root / 'manifest.yaml', payload)
        index = self.read_index(kind)
        index[name] = payload
        self.write_index(kind, index)
        return root / 'manifest.json'

    def load_manifest(self, kind: str, name: str) -> ArtifactManifest:
        path = self.artifact_root(kind, name) / 'manifest.json'
        payload = read_json(path)
        if payload is None:
            raise FileNotFoundError(f'Manifest not found for {kind}:{name}')
        return ArtifactManifest.from_dict(payload)

    def list_manifests(self, kind: str) -> List[ArtifactManifest]:
        payload = self.read_index(kind)
        return [ArtifactManifest.from_dict(v) for _, v in sorted(payload.items())]

    def artifact_store(self, kind: str, name: str) -> ExperimentStore:
        return ExperimentStore(self.artifact_root(kind, name))

    def has_artifact(self, kind: str, name: str) -> bool:
        return (self.artifact_root(kind, name) / 'manifest.json').exists()

    def save_any(self, relative_path: str | Path, obj: Any) -> Path:
        path = self.root / Path(relative_path)
        suffix = path.suffix.lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pd.DataFrame):
            if suffix == '.parquet':
                obj.to_parquet(path, index=False)
                return path
            if suffix == '.csv':
                obj.to_csv(path, index=False)
                return path
            if suffix in {'.json', '.jsonl'}:
                obj.to_json(path, orient='records', indent=2)
                return path
            fallback = path.with_suffix('.csv') if suffix == '' else path
            fallback.parent.mkdir(parents=True, exist_ok=True)
            obj.to_csv(fallback, index=False)
            return fallback
        if isinstance(obj, np.ndarray):
            if suffix == '.npz':
                np.savez(path, data=obj)
                return path
            target = path if suffix == '.npy' else path.with_suffix('.npy')
            np.save(target, obj)
            return target
        if isinstance(obj, dict) and suffix == '.npz' and all(isinstance(v, np.ndarray) for v in obj.values()):
            np.savez(path, **obj)
            return path
        if suffix in {'.json', '.yaml', '.yml', '.txt'}:
            if suffix == '.json':
                return write_json(path, obj)
            if suffix in {'.yaml', '.yml'}:
                return write_yaml(path, obj)
            text = obj if isinstance(obj, str) else json.dumps(to_jsonable(obj), indent=2, sort_keys=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(text, encoding='utf-8')
            tmp.replace(path)
            return path
        if isinstance(obj, (dict, list, tuple, str, int, float, bool)):
            return write_json(path.with_suffix('.json') if suffix == '' else path, obj)
        raise TypeError(f'Unsupported artifact type for {path}: {type(obj)}')

    def load_any(self, relative_path: str | Path) -> Any:
        path = self.root / Path(relative_path)
        suffix = path.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(path)
        if suffix == '.parquet':
            return pd.read_parquet(path)
        if suffix == '.json':
            return read_json(path)
        if suffix in {'.yaml', '.yml'}:
            return yaml.safe_load(path.read_text(encoding='utf-8'))
        if suffix == '.npy':
            return np.load(path, allow_pickle=False)
        if suffix == '.npz':
            return {k: np.asarray(v) for k, v in np.load(path, allow_pickle=False).items()}
        if suffix == '.txt':
            return path.read_text(encoding='utf-8')
        raise ValueError(f'Unsupported artifact suffix: {suffix}')

    def manifest_frame(self, kind: str) -> pd.DataFrame:
        rows = [m.to_dict() for m in self.list_manifests(kind)]
        return pd.DataFrame(rows)

    def save_table_index(self) -> None:
        payload = {
            'scans': self.list_names('scan'),
            'selects': self.list_names('select'),
            'discretizations': self.list_names('discretize'),
            'analyses': self.list_names('analysis'),
        }
        write_json(self.meta_dir / 'table_index.json', payload)

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .protocol import BundleManifest, ViewManifest


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {str(k): _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    return out


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FrameSink:
    def __init__(self, path: str | Path) -> None:
        self.requested_path = Path(path)
        self.requested_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._frames: List[pd.DataFrame] = []
        self._schema = None
        self._final_path: Optional[Path] = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame is None or len(frame) == 0:
            return
        if pa is not None and pq is not None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(str(self.requested_path), table.schema)
                self._schema = table.schema
                self._final_path = self.requested_path
            self._writer.write_table(table)
            return
        self._frames.append(frame.copy())

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.close()
            return self._final_path or self.requested_path
        frame = pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame()
        try:
            frame.to_parquet(self.requested_path, index=False)
            self._final_path = self.requested_path
            return self.requested_path
        except Exception:
            fallback = self.requested_path.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            self._final_path = fallback
            return fallback


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(out, index=False)
        return out
    except Exception:
        fallback = out.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def load_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p)
    parquet_path = p.with_suffix(".parquet")
    csv_path = p.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(str(p))


def open_memmap_npy(path: str | Path, *, shape: tuple[int, ...], dtype: str | np.dtype) -> np.memmap:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(out, mode="w+", dtype=np.dtype(dtype), shape=shape)


def read_bundle_manifest(root: str | Path) -> BundleManifest:
    payload = read_json(Path(root) / "manifest.json")
    return BundleManifest.from_dict(payload)


def write_bundle_manifest(root: str | Path, manifest: BundleManifest) -> Path:
    return write_json(Path(root) / "manifest.json", manifest.to_dict())


def read_view_manifest(root: str | Path, relative_path: str) -> ViewManifest:
    payload = read_json(Path(root) / relative_path)
    return ViewManifest.from_dict(payload)


def write_view_manifest(root: str | Path, relative_path: str, manifest: ViewManifest) -> Path:
    return write_json(Path(root) / relative_path, manifest.to_dict())


def sanitize_scalar_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype(str)
    return series.map(lambda x: x if isinstance(x, (str, int, float, bool, np.integer, np.floating)) or x is None else str(x))


def compact_sample_frame(frame: pd.DataFrame) -> pd.DataFrame:
    kept: Dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column]
        try:
            kept[column] = sanitize_scalar_series(series)
        except Exception:
            continue
    out = pd.DataFrame(kept)
    if "sample_id" in out.columns:
        out = out.drop_duplicates(subset=["sample_id"], keep="last")
    return out.reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame, *, exclude: Optional[Iterable[str]] = None) -> List[str]:
    excluded = set(exclude or [])
    cols: List[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            cols.append(column)
    return cols

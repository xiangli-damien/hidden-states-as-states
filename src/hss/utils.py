from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Tuple

import numpy as np


def f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.float32)


def i32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.int32 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.int32)


def i64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.int64 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.int64)


def normalize_l2(
    x: np.ndarray, *, axis: int = 1, eps: float = 1e-12
) -> np.ndarray:
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, float(eps))


def now_utc() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_thread_env(n: int = 1) -> None:
    s = str(int(max(1, n)))
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[k] = s


def stable_hash(obj: Any, *, n_chars: int = 12) -> str:
    payload = json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n_chars]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(to_jsonable(obj), indent=indent, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(arr))
    tmp.replace(path)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    stem = path.with_suffix("")
    tmp_stem = stem.with_name(stem.name + "_tmp")
    np.savez(tmp_stem, **{k: np.asarray(v) for k, v in arrays.items()})
    tmp_npz = tmp_stem.with_suffix(".npz")
    if not tmp_npz.exists():
        tmp_npz = Path(str(tmp_stem) + ".npz")
    tmp_npz.replace(path)


def open_writable_memmap(
    path: str | Path, *, shape: Tuple[int, ...], dtype: np.dtype | str
) -> np.memmap:
    path = Path(path)
    ensure_dir(path.parent)
    return np.lib.format.open_memmap(
        str(path), mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
    )


def iter_slices(n: int, bs: int) -> Iterator[Tuple[int, int]]:
    bs = max(1, int(bs))
    i = 0
    while i < n:
        yield i, min(n, i + bs)
        i += bs


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
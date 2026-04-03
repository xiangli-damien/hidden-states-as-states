from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .types import StateProvider


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
    x: np.ndarray,
    *,
    axis: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    arr = np.asarray(x)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norms, float(eps))


def now_utc() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_thread_env(n: int = 1) -> None:
    value = str(int(max(1, n)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = value


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
    path: str | Path,
    *,
    shape: Tuple[int, ...],
    dtype: np.dtype | str,
) -> np.memmap:
    path = Path(path)
    ensure_dir(path.parent)
    return np.lib.format.open_memmap(
        str(path),
        mode="w+",
        dtype=np.dtype(dtype),
        shape=tuple(shape),
    )


def iter_slices(n: int, bs: int) -> Iterator[Tuple[int, int]]:
    block = max(1, int(bs))
    start = 0
    while start < n:
        yield start, min(n, start + block)
        start += block


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
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def sanitize_metadata_array(value: Any) -> Optional[np.ndarray]:
    """Convert provider metadata to a persistable numpy array.

    Numeric, boolean and unicode arrays are preserved. Generic object arrays are
    converted to unicode when possible; otherwise the field is dropped.
    """

    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype == object:
        try:
            arr = arr.astype(str)
        except Exception:
            return None
    if np.issubdtype(arr.dtype, np.datetime64):
        arr = arr.astype("datetime64[ns]").astype(str)
    return arr


def sanitize_metadata_fields(
    fields: Mapping[str, Any] | None,
    *,
    expected_len: Optional[int] = None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in dict(fields or {}).items():
        arr = sanitize_metadata_array(value)
        if arr is None:
            continue
        if expected_len is not None and int(arr.shape[0]) != int(expected_len):
            raise ValueError(
                f"metadata field '{key}' has length {arr.shape[0]} but expected {expected_len}"
            )
        out[str(key)] = arr
    return out


def resolve_parallel_backend(
    requested: str,
    *,
    n_jobs: int,
    prefer_memmap: bool,
) -> str:
    if int(n_jobs) <= 1:
        return "serial"
    mode = str(requested or "auto").strip().lower()
    if mode in {"serial", "none", "off"}:
        return "serial"
    if prefer_memmap:
        return "process"
    if mode in {"thread", "threads", "threading"}:
        return "thread"
    return "process"


def materialize_provider_states(
    provider: "StateProvider",
    *,
    path: str | Path,
    layers: Sequence[int],
    indices: Optional[np.ndarray] = None,
    batch_size: int = 4096,
    dtype: np.dtype | str = np.float32,
) -> Path:
    """Materialize selected provider rows/layers into one memmap-backed .npy.

    This is the key enabler for process-based parallelism without duplicating
    large in-memory arrays in each worker.
    """

    layer_ids = [int(x) for x in layers]
    if len(layer_ids) == 0:
        raise ValueError("layers must be non-empty")
    rows = provider.n_items() if indices is None else int(len(indices))
    cols = int(provider.state_dim())
    arr = open_writable_memmap(path, shape=(rows, len(layer_ids), cols), dtype=dtype)
    for pos, layer in enumerate(layer_ids):
        offset = 0
        for batch in provider.iter_batches(
            layer=int(layer),
            indices=indices,
            batch_size=int(batch_size),
        ):
            states = np.asarray(batch.states, dtype=dtype)
            if states.ndim != 2:
                raise ValueError(
                    f"Layer {layer}: expected a 2-D state batch, got shape {states.shape}"
                )
            b = int(states.shape[0])
            if b == 0:
                continue
            arr[offset : offset + b, pos, :] = states
            offset += b
        if offset != rows:
            raise ValueError(
                f"Layer {layer}: materialized {offset} rows but expected {rows}"
            )
    del arr
    return Path(path)

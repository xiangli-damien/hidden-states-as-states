"""Content-addressed artifacts and inter-process locks shared by grid workers."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid
import platform
from importlib.metadata import version, PackageNotFoundError

import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()[:24]


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def runtime_versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "scikit-learn", "pandas", "zarr", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def source_version():
    # Includes uncommitted implementation edits; never reuse stale algorithm caches.
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            "files": {
                str(p.relative_to(root)): file_digest(p)
                for p in sorted(root.rglob("*.py"))
            },
            "runtime": runtime_versions(),
        }
    )

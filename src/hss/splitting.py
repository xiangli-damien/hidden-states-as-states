from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def split_indices(
    n: int,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> Dict[str, np.ndarray]:
    fracs = np.array(
        [float(train_frac), float(val_frac), float(test_frac)], dtype=np.float64
    )
    if np.any(fracs < 0):
        raise ValueError("fractions must be non-negative")
    s = fracs.sum()
    if s <= 0:
        raise ValueError("fractions must sum to > 0")
    fracs = fracs / s
    rng = np.random.RandomState(int(seed))
    idx = np.arange(int(n), dtype=np.int64)
    if shuffle and n > 0:
        rng.shuffle(idx)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    n_train = min(n, max(0, n_train))
    n_val = min(n - n_train, max(0, n_val))
    return {
        "train": np.sort(idx[:n_train]),
        "val": np.sort(idx[n_train: n_train + n_val]),
        "test": np.sort(idx[n_train + n_val:]),
    }


def subsample_indices(
    n: int,
    *,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n, size=int(max_samples), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def kfold_indices(
    n: int,
    *,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n <= 0:
        return []
    k = int(n_splits)
    if k < 2:
        raise ValueError("n_splits must be >= 2")
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        rng.shuffle(idx)
    sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for sz in sizes:
        end = start + sz
        val = np.sort(idx[start:end])
        train = np.sort(np.concatenate([idx[:start], idx[end:]]))
        folds.append((train, val))
        start = end
    return folds


def stratified_split(
    y: np.ndarray,
    *,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    rng = np.random.RandomState(int(seed))
    classes = np.unique(y)
    train_parts, rest_parts = [], []
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_train = max(1, int(round(float(train_frac) * len(idx_c))))
        train_parts.append(idx_c[:n_train])
        rest_parts.append(idx_c[n_train:])
    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    rest = np.sort(np.concatenate(rest_parts)).astype(np.int64)
    return train, rest
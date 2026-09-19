"""Exact stratified empirical AUROC bootstrap without re-sorting every resample."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=3)
def _counts(labels,seed,repeats):
    y=np.frombuffer(labels,dtype=np.uint8)
    groups=[np.flatnonzero(y==label) for label in [0,1]]
    rng=np.random.default_rng(seed)
    counts=np.empty((repeats,len(y)),dtype=np.float32)
    for b in range(repeats):
        take=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        counts[b]=np.bincount(take,minlength=len(y))
    return counts


def auc_samples(y,score,seed,repeats):
    y=np.asarray(y,dtype=np.uint8);score=np.asarray(score)
    if not np.isfinite(score).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite scores and both binary classes required')
    weights=_counts(y.tobytes(),seed,repeats)
    order=np.argsort(score,kind='stable')
    starts=np.r_[0,1+np.flatnonzero(np.diff(score[order]))]
    negative=np.add.reduceat(weights[:,order]*(1-y[order]),starts,axis=1)
    positive=np.add.reduceat(weights[:,order]*y[order],starts,axis=1)
    below=np.cumsum(negative,axis=1,dtype=np.float64)-.5*negative
    return (positive*below).sum(1)/(int(y.sum())*int(len(y)-y.sum()))


def auc_ci(y,score,seed,repeats=1000):
    return np.quantile(auc_samples(y,score,seed,repeats),[.025,.975]).tolist()

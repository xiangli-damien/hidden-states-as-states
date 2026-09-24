"""Small numpy-only contracts shared by GPU and CPU revision stages.

GPU scripts deliberately do not import hss's eager sklearn imports: they run in
the pinned OpenAct environment. No fitted pickle is loaded by a model process.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def config(path):
    return json.loads(Path(path).read_text())


def provenance(cfg, files=()):
    return {'config': cfg, 'files': {str(p): sha(p) for p in files},
            'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[1], text=True).strip()}


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing experiment identity differs: {path}')
    else:
        write_json(path, value)


def nearest(x, centers, chunk=512):
    labels = []
    c = centers.astype(np.float64)
    for start in range(0, len(x), chunk):
        b = x[start:start + chunk].astype(np.float64)
        distance = (b*b).sum(1)[:, None] + (c*c).sum(1)[None, :] - 2*b@c.T
        labels.append(distance.argmin(1))
    return np.concatenate(labels)


def reconstruct(x, decoder, method):
    """All inputs are actual vectors; no mean-vector broadcasting to token slots."""
    x = np.asarray(x, dtype=np.float32)
    if method == 'identity':
        return x.copy()
    if method == 'zero':
        return np.zeros_like(x)
    if method == 'mean':
        return np.broadcast_to(decoder['train_mean'], x.shape).copy()
    if method.startswith('global_pca_'):
        basis = decoder['global_basis'][:int(method.rsplit('_', 1)[1])]
        center = decoder['train_mean']
        return center + ((x-center)@basis.T)@basis
    centers = decoder['kmeans_centers'] if method == 'kmeans_centroid' else decoder['centers']
    assignment = nearest(x, centers)
    result = centers[assignment].copy()
    if method.startswith('empirical_pca_'):
        rank=int(method.rsplit('_',1)[1])
        anchors=decoder['local_empirical_centers']
        result=anchors[assignment].copy()
        for k in np.unique(assignment):
            selected=assignment==k
            basis=decoder['local_empirical_basis'][k,:rank]
            result[selected]+=((x[selected]-anchors[k])@basis.T)@basis
    elif method.startswith('local_pca_'):
        rank = int(method.rsplit('_', 1)[1])
        for k in np.unique(assignment):
            selected = assignment == k
            basis = decoder['local_basis'][k, :rank]
            # The affine anchor is the fitted GMM center; PCA is fitted to its
            # nearest-assigned train residuals, without test-time refitting.
            result[selected] += ((x[selected]-centers[k])@basis.T)@basis
    elif method not in ('centroid', 'kmeans_centroid'):
        raise ValueError(method)
    return result.astype(np.float32)


def nmse_rows(x, recovered, train_mean):
    error = np.square(x.astype(np.float64)-recovered).sum(-1)
    denominator = np.square(x.astype(np.float64)-train_mean).sum(-1)
    return error, denominator


def paired_ratio_ci(numerator, denominator, seed=42, boot=1000):
    a, b = np.asarray(numerator), np.asarray(denominator)
    if b.sum() <= 0:
        return {'estimate': None, 'ci95': None, 'n': len(a)}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(boot):
        ix = rng.integers(len(a), size=len(a))
        den = b[ix].sum()
        if den > 0:
            samples.append(float(a[ix].sum()/den))
    return {'estimate': float(a.sum()/b.sum()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'n': len(a)}


def status(root, stage, **fields):
    write_json(Path(root)/f'{stage}_status.json', {'stage': stage, 'updated_unix': time.time(),
               'pid': os.getpid(), **fields})


class OncePatch:
    """Single prefill intervention, explicit position list, never cached decoding.

    The callback sees [n_selected, d]; batch size one is intentional. Continuing
    from any supplied KV cache is forbidden by the caller. No hook fires twice.
    """
    def __init__(self, positions, prefill_length, transform):
        self.positions = list(positions)
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise ValueError('Positions must be nonempty and unique')
        if min(self.positions) < 0 or max(self.positions) >= prefill_length:
            raise ValueError('Position outside prefill')
        self.prefill_length = prefill_length
        self.transform = transform
        self.calls = 0
        self.energy = None

    def __call__(self, module, args, output):
        # After first call decode hooks are no-ops, without touching the cache.
        if self.calls:
            return None
        import torch
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != self.prefill_length:
            raise ValueError('Patch must be applied to fresh, unpadded prefill')
        self.calls += 1
        before = h[0, self.positions].detach().clone()
        after = self.transform(before)
        if after.shape != before.shape or not torch.isfinite(after).all():
            raise ValueError('Invalid replacement')
        patched = h.clone()
        patched[0, self.positions] = after.to(h.dtype)
        self.energy = float((patched[0, self.positions].float()-before.float()).square().sum())
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

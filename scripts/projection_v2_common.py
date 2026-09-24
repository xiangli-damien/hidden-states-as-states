"""Frozen local-projection extensions. Pure NumPy; no fitting or outcome selection."""
import numpy as np


def catalog():
    base = dict(prefix=16, width=4, rank=8, alpha=1., operator='local')
    changes = [
        ('D01', dict(operator='wrong')), ('D02', dict(operator='shared', rank=64)),
        ('D03', dict(rank=16)), ('D04', dict(alpha=-.3)),
        ('D05', dict(width=1)), ('D06', dict(width=16)),
        ('D07', dict(prefix=8)), ('D08', dict(prefix=32)),
        ('D09', dict(operator='norm')), ('D10', dict(operator='common')),
    ]
    return [dict(base, **overrides, name=name) for name, overrides in changes]


def transfer_conditions():
    base = dict(prefix=16, width=4, rank=8, alpha=1.)
    return [dict(base, name=name, operator=op) for name, op in
            [('baseline', 'baseline'), ('local8', 'local'), ('shared8', 'shared'), ('wrong_local8', 'wrong')]]


def current_regions(x, centers):
    # Exact assignment expression used by the protected C1 implementation.
    distance = np.square(np.asarray(x, dtype=float)[:, None] - np.asarray(centers, dtype=float)[None]).sum(-1)
    return np.argsort(distance, axis=1, kind='stable')[:, 0], distance


def derangement(k, seed):
    if k < 2:
        raise ValueError('Wrong-region control requires at least two regions')
    order = np.random.default_rng(seed).permutation(k)
    result = np.empty(k, dtype=int)
    result[order] = np.roll(order, -1)
    assert np.all(result != np.arange(k))
    return result


def project(x, decoder, condition, permutation, near_zero=1e-12):
    """Same recipient anchor, row-orthonormal bases; snapshot whole window first."""
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(decoder['centers'], dtype=np.float64)
    current, _ = current_regions(x, centers)
    donor = np.asarray(permutation)[current] if condition['operator'] == 'wrong' else current.copy()
    rank = condition['rank']
    z = x.copy()
    if condition['operator'] != 'baseline':
        for k in np.unique(current):
            mask = current == k
            b = (decoder['shared_basis'][:rank] if condition['operator'] == 'shared'
                 else decoder['local_basis'][np.asarray(permutation)[k] if condition['operator'] == 'wrong' else k, :rank])
            b = b.astype(np.float64)
            if len(b) != rank:
                raise ValueError('Required compatible rank unavailable')
            residual = x[mask] - centers[k]
            z[mask] = centers[k] + (residual @ b.T) @ b
    fallback = np.zeros(len(x), dtype=bool)
    if condition['operator'] == 'norm':
        norm = np.linalg.norm(z, axis=1)
        fallback = norm <= near_zero
        z[~fallback] *= (np.linalg.norm(x[~fallback], axis=1) / norm[~fallback])[:, None]
        z[fallback] = x[fallback]
    elif condition['operator'] == 'common':
        z = x + (z - x).mean(axis=0, keepdims=True)
    result = x + condition['alpha'] * (z - x)
    return result, dict(current=current.tolist(), basis_region=donor.tolist(),
                        norm_fallback=fallback.tolist())


def geometry(x, actual, decoder, counts):
    x = np.asarray(x, dtype=float); actual = np.asarray(actual, dtype=float)
    original, distance = current_regions(x, decoder['centers'])
    after, _ = current_regions(actual, decoder['centers'])
    residual = x - decoder['centers'][original]
    projected = np.empty_like(residual)
    for k in np.unique(original):
        ix = original == k; b = decoder['local_basis'][k, :8].astype(float)
        projected[ix] = (residual[ix] @ b.T) @ b
    delta = actual - x
    den = np.square(residual).sum(1)
    off = np.divide(np.square(residual - projected).sum(1), den, out=np.zeros(len(x)), where=den > 0)
    return dict(energy=float(np.square(delta).sum()), modified_tokens=int(np.any(delta != 0, axis=1).sum()),
                region_before=original.tolist(), region_after=after.tolist(),
                region_retained=(original == after).tolist(),
                relative_change=(np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-30)).tolist(),
                offspace_fraction=off.tolist(), assignment_distance=np.sqrt(distance[np.arange(len(x)), original]).tolist(),
                training_question_support=np.asarray(counts)[original].tolist())


def paired_interval(values, draws=2000, seed=9242026):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError('Cannot report empty cohort')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return dict(estimate=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), n=len(values))

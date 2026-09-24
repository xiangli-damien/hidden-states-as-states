"""Within-answer distributions of adjacent-layer updates, never temporal deltas."""
import hashlib

import numpy as np


def pack_bf16_exact(values):
    """Lossless storage only: reject any float32 not exactly representable in bf16."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    if not np.isfinite(values).all() or np.any(bits & 65535):
        raise ValueError('Source activations are not finite exact bf16; refusing lossy cache')
    return (bits >> 16).astype(np.uint16)


def unpack_bf16(values):
    return (np.asarray(values, dtype=np.uint32) << 16).view(np.float32)


def stable_rng(sample_id, seed, purpose):
    digest = hashlib.sha256(f'{seed}:{purpose}:{sample_id}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def balanced_positions(n, sample_id, count, seed, purpose):
    """Exactly count units per question, independent of length and correctness."""
    if n < 1 or count < 1:
        raise ValueError('Nonempty units and positive count required')
    return stable_rng(sample_id, seed, purpose).choice(n, count, replace=n < count)


def segment_means(values, ends):
    """Exhaustive, nonoverlapping sentence means; includes final EOS if present."""
    values = np.asarray(values)
    ends = np.asarray(ends, dtype=int)
    if not len(ends) or ends[-1] != len(values) or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError('Boundaries must partition all response tokens')
    sums = np.concatenate([np.zeros((1,) + values.shape[1:]),
                           np.cumsum(values, axis=0, dtype=np.float64)], axis=0)
    shape = (-1,) + (1,) * (values.ndim - 1)
    return ((sums[ends] - sums[np.r_[0, ends[:-1]]]) /
            np.diff(np.r_[0, ends]).reshape(shape)).astype(np.float32)


def sequence_features(sequences, cardinalities, sample_ids, seed, shuffle=False):
    """Question-normalized occupancy and true adjacent-unit transition counts.

    Each item is [units, layers]. No edge ever crosses a question boundary.
    The same shuffle is used across layers, preserving within-unit depth routes.
    """
    n = len(sequences)
    offsets = np.cumsum([0] + list(cardinalities))
    edge_offsets = np.cumsum([0] + [k*k for k in cardinalities])
    occ = np.zeros((n, offsets[-1]), dtype=np.float64)
    edges = np.zeros((n, edge_offsets[-1]), dtype=np.float64)
    for i, original in enumerate(sequences):
        states = np.asarray(original, dtype=int)
        if states.ndim != 2 or states.shape[1] != len(cardinalities) or not len(states):
            raise ValueError('Invalid question sequence')
        if shuffle:
            order = stable_rng(sample_ids[i], seed, 'order_shuffle').permutation(len(states))
            states = states[order]
        for j, k in enumerate(cardinalities):
            col = states[:, j]
            if np.any((col < 0) | (col >= k)):
                raise ValueError('Cluster outside layer-local vocabulary')
            occ[i, offsets[j]:offsets[j+1]] = np.bincount(col, minlength=k) / len(col)
            if len(col) > 1:
                edges[i, edge_offsets[j]:edge_offsets[j+1]] = (
                    np.bincount(col[:-1]*k+col[1:], minlength=k*k) / (len(col)-1))
    return occ, edges


class SequenceBayes:
    """Smoothed class-conditional occupancy and row-conditional transitions.

    Every question contributes total mass one per layer to each table. Scores
    average evidence over units, preventing response length from scaling odds.
    Occupancy+transition is a composite score, not an exact sequence likelihood.
    """
    def __init__(self, cardinalities, alpha=1., transitions=False):
        self.cardinalities = list(cardinalities)
        self.alpha = alpha
        self.transitions = transitions

    def fit(self, x, y):
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != {0, 1} or self.alpha <= 0:
            raise ValueError('Need both classes and positive smoothing')
        self.prior = np.log(np.sum(y == 1) / np.sum(y == 0))
        self.occupancy_weights, self.edge_weights = [], []
        start, edge_start = 0, sum(self.cardinalities)
        for k in self.cardinalities:
            a = np.stack([x[y == c, start:start+k].sum(0) + self.alpha for c in [0, 1]])
            a /= a.sum(1, keepdims=True)
            self.occupancy_weights.extend(np.log(a[1]) - np.log(a[0]))
            start += k
            if self.transitions:
                b = np.stack([x[y == c, edge_start:edge_start+k*k].sum(0).reshape(k,k)
                              + self.alpha for c in [0, 1]])
                b /= b.sum(2, keepdims=True)
                self.edge_weights.extend((np.log(b[1]) - np.log(b[0])).ravel())
                edge_start += k*k
        self.weights = np.r_[self.occupancy_weights, self.edge_weights]
        if x.shape[1] != len(self.weights):
            raise ValueError('Feature width mismatch')
        return self

    def decision_function(self, x):
        return np.asarray(x) @ self.weights + self.prior

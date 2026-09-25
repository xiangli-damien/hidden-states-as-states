"""Layer-local empirical token-label lookup. No learned score aggregation."""
import numpy as np


def token_counts(sequence, cardinalities):
    sequence = np.asarray(sequence)
    if sequence.ndim != 2 or sequence.shape[1] != len(cardinalities):
        raise ValueError('Expected tokens x layers')
    if len(set(cardinalities)) != 1:
        raise ValueError('This frozen pilot uses equal K per layer')
    if not np.issubdtype(sequence.dtype, np.integer):
        raise ValueError('Cluster IDs must be integers')
    result = []
    for j, k in enumerate(cardinalities):
        if np.any((sequence[:, j] < 0) | (sequence[:, j] >= k)):
            raise ValueError('Invalid cluster ID')
        result.append(np.bincount(sequence[:, j], minlength=k))
    return np.asarray(result, dtype=np.int64)


def fit_error_lookup(counts, failure, training_mask):
    """Raw training-token ratio. A long answer supplies more tokens, by design."""
    counts = np.asarray(counts)
    training_mask = np.asarray(training_mask, dtype=bool)
    y = np.asarray(failure)[training_mask]
    if counts.ndim != 3 or len(counts) != len(training_mask) or (counts < 0).any():
        raise ValueError('Expected nonnegative question x layer x cluster counts')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Training labels must contain both classes')
    selected = counts[training_mask]
    wrong = selected[y == 1].sum(0)
    correct = selected[y == 0].sum(0)
    totals = correct + wrong
    prior = wrong.sum(1) / totals.sum(1)
    # Explicit fallback for an unseen training cluster; never look at test labels.
    q = np.broadcast_to(prior[:, None], totals.shape).copy()
    np.divide(wrong, totals, out=q, where=totals > 0)
    return {'wrong': wrong, 'correct': correct, 'q': q, 'fallback': totals == 0,
            'token_error_prior': prior}


def lookup_sequence(sequence, q):
    sequence = np.asarray(sequence)
    q = np.asarray(q, dtype=float)
    token_counts(sequence, [q.shape[1]] * q.shape[0])
    if not len(sequence) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Nonempty sequence and valid lookup probabilities required')
    per_token = q[np.arange(q.shape[0])[None, :], sequence]
    cumulative_mean = np.cumsum(per_token, axis=0) / np.arange(1, len(sequence) + 1)[:, None]
    return per_token, cumulative_mean


def score_layer_deltas(gmm, cluster_error_rates, delta_vectors):
    """Apply one frozen layer's GMM and lookup to NEW token updates."""
    regions = gmm.predict(np.asarray(delta_vectors))
    return regions, np.asarray(cluster_error_rates)[regions]

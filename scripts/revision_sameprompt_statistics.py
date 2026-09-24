"""Question-level inference and response-level FAR for the fixed two-check study."""
import numpy as np


def auc(failure, score):
    y, s = np.asarray(failure, int), np.asarray(score, float)
    if y.ndim != 1 or s.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError('Expected aligned binary outcomes and finite risk scores')
    positive, negative = s[y == 1], s[y == 0]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None]-negative[None, :]
    return float(np.mean((differences > 0)+.5*(differences == 0)))


def mean_interval(values, draws=2000, seed=42):
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Question effects must be finite')
    if not len(x):
        return {'n_questions': 0, 'estimate': None, 'low': None, 'high': None}
    ix = np.random.default_rng(seed).integers(len(x), size=(draws, len(x)))
    lo, hi = np.quantile(x[ix].mean(1), [.025, .975])
    return {'n_questions': len(x), 'estimate': float(x.mean()), 'low': float(lo), 'high': float(hi)}


def within_question(question, failure, first, second, draws=2000, seed=42):
    q = np.asarray(question, str); y = np.asarray(failure, int)
    a, b = np.asarray(first, float), np.asarray(second, float)
    if not (q.shape == y.shape == a.shape == b.shape):
        raise ValueError('Mismatched trajectory identities')
    rows = []
    for name in sorted(set(q)):
        mask = q == name; left, right = auc(y[mask], a[mask]), auc(y[mask], b[mask])
        if left is None:
            assert right is None
            continue
        rows.append({'question_group': name, 'valid_trajectories': int(mask.sum()),
                     'auc_first': left, 'auc_second': right, 'delta': left-right})
    return {'question_rows': rows, 'macro_first': mean_interval([r['auc_first'] for r in rows], draws, seed),
            'macro_second': mean_interval([r['auc_second'] for r in rows], draws, seed),
            'paired_delta': mean_interval([r['delta'] for r in rows], draws, seed)}


def far_threshold(correct_response_maxima, target=.1):
    """Calibrate once per correct response, after max across available boundaries.

    A strict greater-than alarm is conservative on ties. Negative infinity
    represents a response that finished before either fixed boundary.
    """
    x = np.asarray(correct_response_maxima, float)
    if x.ndim != 1 or np.isnan(x).any() or np.isposinf(x).any() or not 0 <= target < 1:
        raise ValueError('Invalid calibration values or FAR target')
    if not len(x):
        return None
    allowed = int(np.floor(target*len(x)))
    threshold = float(np.sort(x)[::-1][allowed])
    assert int(np.sum(x > threshold)) <= allowed
    return threshold


def alarms(scores, boundaries, observed_lengths, threshold):
    scores, positions = np.asarray(scores, float), np.asarray(boundaries, int)
    length = np.asarray(observed_lengths, int)
    if threshold is None:
        raise ValueError('No correct calibration responses; cannot invent a threshold')
    if scores.shape != (len(length), len(positions)) or not np.all(np.diff(positions) > 0):
        raise ValueError('Invalid fixed boundary matrix')
    if np.isnan(scores).any() or np.isposinf(scores).any() or not np.all(length > 0):
        raise ValueError('Invalid response data')
    if np.any(np.isfinite(scores) & (length[:, None] <= positions[None, :])):
        raise ValueError('A risk score uses a boundary after response termination')
    flags = scores > threshold
    flagged = flags.any(1)
    first = np.full(len(length), -1, dtype=int)
    first[flagged] = positions[np.argmax(flags[flagged], axis=1)]
    saved = np.where(flagged, length-first, 0)
    assert np.all(saved >= 0)
    return {'alarmed': flagged, 'first_boundary': first, 'potential_saved_tokens': saved}

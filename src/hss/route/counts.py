"""Cross-fitted route counts. No correctness argument or correctness tuning."""
import hashlib
import numpy as np
from scipy.special import xlogy


def group_folds(groups, n_folds=5):
    return np.array([int(hashlib.sha256(('route-initial-v1:' + str(g)).encode()).hexdigest(), 16) % n_folds for g in groups])


def canonical_states(states):
    a = np.asarray(states)
    if a.ndim != 2 or a.shape[1] < 2 or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError('Expected nonnegative integer states, samples by layers')
    vocab = [np.unique(col) for col in a.T]
    return np.column_stack([np.searchsorted(v, a[:, l]) for l, v in enumerate(vocab)]), vocab


def js_rows(p, q):
    m = (p + q) / 2
    return ((xlogy(p, p) - xlogy(p, m) + xlogy(q, q) - xlogy(q, m)).sum(axis=-1) / (2 * np.log(2)))


def crossfit_scores(states, folds, probabilities=None, alpha=1., second_order_strength=10.):
    z, vocab = canonical_states(states)
    n, depth = z.shape; sizes = [len(v) for v in vocab]
    if len(folds) != n or len(np.unique(folds)) < 2:
        raise ValueError('Need at least two held-out groups of folds')
    layers = {k: np.zeros((n, depth - 1)) for k in ['edge', 'context', 'soft_js']}
    node = np.zeros((n, depth)); second = np.zeros((n, depth - 2))
    if probabilities is not None:
        pp = []
        for p, v in zip(probabilities, vocab):
            p = np.asarray(p, float)[:, v]
            if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(1) <= 0):
                raise ValueError('Invalid posterior probabilities')
            pp.append(p / p.sum(1, keepdims=True))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        marginals = []
        for l in range(depth):
            c = np.bincount(z[train, l], minlength=sizes[l]).astype(float) + alpha
            marginals.append(c / c.sum())
            node[test, l] = -np.log(marginals[-1][z[test, l]])
        for l in range(depth - 1):
            ka, kb = sizes[l:l+2]
            c = np.bincount(z[train, l] * kb + z[train, l+1], minlength=ka * kb).reshape(ka, kb).astype(float) + alpha
            t = c / c.sum(1, keepdims=True)
            edge = t[z[test, l], z[test, l+1]]
            layers['edge'][test, l] = -np.log(edge)
            layers['context'][test, l] = -np.log(edge / marginals[l+1][z[test, l+1]])
            if probabilities is not None:
                soft = pp[l][train].T @ pp[l+1][train] + alpha
                soft /= soft.sum(1, keepdims=True)
                layers['soft_js'][test, l] = js_rows(pp[l][test] @ soft, pp[l+1][test])
            if l:
                kp = sizes[l-1]
                pairs = z[train, l-1] * ka + z[train, l]
                c2 = np.bincount(pairs * kb + z[train, l+1], minlength=kp*ka*kb).reshape(kp*ka, kb)
                idx = z[test, l-1] * ka + z[test, l]
                prob = (c2[idx, z[test, l+1]] + second_order_strength * edge) / (c2.sum(1)[idx] + second_order_strength)
                second[test, l-1] = -np.log(prob)
    scores = dict(node=node.mean(1), final_node=node[:, -1], edge=layers['edge'].mean(1), context=layers['context'].mean(1))
    if probabilities is not None: scores['soft_js'] = layers['soft_js'].mean(1)
    if depth > 2:
        scores['second_order_surprise'] = second.mean(1)
        scores['second_order_predictive_gain'] = (layers['edge'][:, 1:] - second).mean(1)
    return scores, layers

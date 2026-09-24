"""Frozen whole-response mean regions applied to an observed token window.

Region inference never receives future tokens. Mean-shift retains the full
centered token residuals and must NOT be described as eight-coordinate token
compression. Token-project is an explicitly separate, eight-coordinate decoder.
"""
import numpy as np


def posterior(x, decoder):
    x = np.asarray(x, dtype=np.float64)
    mu, var = decoder['centers'], decoder['variances']
    score = -.5 * (np.square(x[:, None] - mu[None]) / var[None]).sum(-1)
    score -= .5 * np.log(2 * np.pi * var).sum(-1)[None]
    score += np.log(decoder['weights'])[None]
    maximum = score.max(1, keepdims=True)
    probability = np.exp(score - maximum)
    normalizer = probability.sum(1, keepdims=True)
    return probability / normalizer, (maximum + np.log(normalizer))[:, 0]


def condition_names():
    names = ['identity', 'tokenmap_centroid', 'tokenmap_shared8', 'tokenmap_local8']
    for operator in ['mean_shift', 'token_project']:
        names.extend(operator + '_' + method for method in
                     ['centroid', 'shared8', 'local8', 'wrong8_42', 'wrong8_137', 'wrong8_271'])
    return names


def apply(x, mean_decoder, token_decoder, condition):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty token window')
    m = x.mean(0)
    probability, density = posterior(m[None], mean_decoder)
    region = int(probability.argmax(1)[0])
    meta = {'mean_region': region, 'mean_max_posterior': float(probability.max()),
            'mean_log_density': float(density[0])}
    if condition == 'identity':
        return x.copy(), meta
    if condition.startswith('tokenmap_'):
        centers = token_decoder['centers'].astype(np.float64)
        distance = ((x[:, None] - centers[None]) ** 2).sum(-1)
        labels = distance.argmin(1)
        z = centers[labels].copy()
        method = condition[len('tokenmap_'):]
        if method != 'centroid':
            for i, s in enumerate(labels):
                b = (token_decoder['local_basis'][s, :8] if method == 'local8'
                     else token_decoder['shared_basis'][:8]).astype(np.float64)
                r = x[i] - centers[s]
                z[i] += (r @ b.T) @ b
        meta['token_regions'] = labels.tolist()
        return z, meta
    operator = 'mean_shift' if condition.startswith('mean_shift_') else 'token_project'
    if not condition.startswith(operator + '_'):
        raise ValueError(condition)
    method = condition[len(operator) + 1:]
    mu = mean_decoder['centers'][region]
    source = m[None] if operator == 'mean_shift' else x
    if method == 'centroid':
        reconstructed = np.broadcast_to(mu, source.shape).copy()
    else:
        if method == 'local8':
            b = mean_decoder['local_basis'][region]
        elif method == 'shared8':
            b = mean_decoder['shared_basis']
        elif method.startswith('wrong8_'):
            permutation = mean_decoder['permutation_' + method.rsplit('_', 1)[1]]
            b = mean_decoder['local_basis'][permutation[region]]
        else:
            raise ValueError(method)
        residual = source - mu
        reconstructed = mu + (residual @ b.T) @ b
    z = x + (reconstructed[0] - m) if operator == 'mean_shift' else reconstructed
    return z, meta

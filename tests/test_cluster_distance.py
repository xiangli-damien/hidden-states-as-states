import numpy as np

from hss.analysis.cluster_distance import cluster_percentiles, factor_distances


def test_mfa_distance_matches_dense_inverse_and_rotations():
    rng = np.random.default_rng(9)
    X, mu, W = rng.normal(size=(25, 13)), rng.normal(size=13), rng.normal(size=(13, 4))
    psi = rng.uniform(.2, 2, size=13)
    result = factor_distances(X, mu, W, psi)
    delta = X - mu
    dense = np.einsum('ni,ij,nj->n', delta, np.linalg.inv(W @ W.T + np.diag(psi)), delta)
    np.testing.assert_allclose(result['mahalanobis'] ** 2, dense, rtol=1e-12)
    np.testing.assert_allclose(result['euclidean'] ** 2, result['parallel'] ** 2 + result['perpendicular'] ** 2)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = factor_distances(X, mu, W @ Q, psi)
    for key in result:
        np.testing.assert_allclose(result[key], rotated[key], rtol=1e-12)
    diagonal = factor_distances(X, mu, W[:, :0], psi)
    np.testing.assert_allclose(diagonal['mahalanobis'] ** 2, (delta * delta / psi).sum(1))


def test_percentiles_are_local_and_ties_are_not_artificially_split():
    p = cluster_percentiles([1, 1, 5, 100, 200], [0, 0, 0, 1, 1])
    np.testing.assert_allclose(p, [1/3, 1/3, 5/6, .25, .75])

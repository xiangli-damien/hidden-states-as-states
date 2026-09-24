import numpy as np
from sklearn.manifold import trustworthiness

from hss.analysis.cluster_profiles import (
    association, neighborhood_audit, participation_dimension, wilson_interval,
)


def test_dimension_matches_covariance_and_is_rotation_invariant():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(18, 7)) * np.arange(1, 8)
    cov = np.cov(X, rowvar=False)
    expected = np.trace(cov) ** 2 / np.square(cov).sum()
    Q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    np.testing.assert_allclose(participation_dimension(X), expected)
    np.testing.assert_allclose(participation_dimension(X @ Q), expected)
    assert participation_dimension(np.ones((3, 5))) == 0


def test_neighbor_audit_matches_sklearn_and_identity():
    rng = np.random.default_rng(46)
    X = rng.normal(size=(80, 6))
    Y = X[:, :2]
    audit = neighborhood_audit(X, Y, np.arange(len(X)), k=5)
    np.testing.assert_allclose(audit['trustworthiness'], trustworthiness(X, Y, n_neighbors=5))
    same = neighborhood_audit(X, X, np.arange(len(X)), k=5)
    assert same['recall'] == 1 and same['trustworthiness'] == 1


def test_descriptions_handle_label_permutation_and_extreme_accuracy():
    a = [0, 0, 1, 1, 2, 2]
    result = association(a, [2, 2, 0, 0, 1, 1])
    assert result['ari'] == result['ami'] == 1
    assert np.sum(result['counts']) == 6
    lo, hi = wilson_interval(0, 10)
    assert abs(lo) < 1e-15 and hi > .25
    lo, hi = wilson_interval(10, 10)
    assert lo < .75 and abs(hi - 1) < 1e-15

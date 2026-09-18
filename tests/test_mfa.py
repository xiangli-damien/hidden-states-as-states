import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from hss.cluster.mfa import MFAModel, fit_mfa
from hss.cluster.registry import rebuild_model
from hss.cluster.metrics import compute_icl


def test_low_rank_likelihood_equals_dense_gaussian():
    rng = np.random.default_rng(4)
    means, W = rng.normal(size=(3, 7)), rng.normal(size=(3, 7, 2))
    psi = rng.uniform(0.2, 1.0, size=(3, 7))
    model = MFAModel(np.array([0.2, 0.3, 0.5]), means, W, psi, chunk_size=3)
    X = rng.normal(size=(17, 7))
    dense = np.stack(
        [
            multivariate_normal.logpdf(X, means[k], W[k] @ W[k].T + np.diag(psi[k]))
            + np.log(model.weights_[k])
            for k in range(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        model.score_samples(X), logsumexp(dense, axis=1), atol=1e-11
    )
    np.testing.assert_allclose(
        model.predict_proba(X),
        np.exp(dense - logsumexp(dense, axis=1)[:, None]),
        atol=1e-11,
    )
    restored = rebuild_model(model.config(), model.state_arrays())
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_em_increases_likelihood_and_recovers_separated_components():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(15)
    y = np.repeat([0, 1], 160)
    X = rng.normal(size=(320, 2)) @ rng.normal(size=(2, 12)) + 0.2 * rng.normal(
        size=(320, 12)
    )
    X += y[:, None] * 10
    model = fit_mfa(X, 2, rank=2, n_init=1, max_iter=35, chunk_size=41)
    assert np.min(np.diff(model.history_)) > -1e-7
    assert model.history_[-1] > model.history_[0] + 1
    assert adjusted_rand_score(y, model.predict(X)) > 0.95
    assert np.isfinite(compute_icl(model, X, "icl"))
    assert model._n_parameters() == 1 + 2 * (24 + 24 - 1)


def test_rank_zero_is_diagonal_gaussian_and_chunking_is_invariant():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 5))
    a = fit_mfa(X, 1, rank=0, n_init=1, max_iter=10, chunk_size=11)
    b = fit_mfa(X, 1, rank=0, n_init=1, max_iter=10, chunk_size=60)
    np.testing.assert_allclose(a.means_[0], X.mean(0), atol=1e-10)
    np.testing.assert_allclose(a.noise_[0], X.var(0), atol=1e-10)
    np.testing.assert_allclose(a.score_samples(X), b.score_samples(X), atol=1e-10)


def test_continuation_reproduces_uninterrupted_fit_without_mutating_checkpoint():
    rng = np.random.default_rng(32)
    X = (
        rng.normal(size=(90, 2)) @ rng.normal(size=(2, 9))
        + rng.normal(size=(90, 9)) * 0.1
    )
    saved = []
    a = fit_mfa(
        X,
        2,
        rank=2,
        n_init=1,
        max_iter=7,
        tol=0,
        init_method="svd",
        checkpoint=lambda m, r: saved.append(m),
        checkpoint_interval=3,
    )
    before = a.means_.copy()
    b = fit_mfa(X, 2, rank=2, n_init=1, max_iter=13, tol=0, initial_model=a)
    direct = fit_mfa(X, 2, rank=2, n_init=1, max_iter=13, tol=0, init_method="svd")
    np.testing.assert_array_equal(a.means_, before)
    np.testing.assert_allclose(b.history_, direct.history_, atol=1e-10)
    np.testing.assert_allclose(b.score_samples(X), direct.score_samples(X), atol=1e-10)
    assert len(saved[-1].history_) == 7
    assert np.min(np.diff(direct.history_)) > -1e-8
    assert np.isclose(direct.score(X), direct.history_[-1])


def test_svd_initialization_preserves_full_input_and_finite_small_clusters():
    X = np.random.default_rng(31).normal(size=(20, 12))
    original = X.copy()
    m = fit_mfa(X, 8, rank=5, n_init=1, max_iter=4, init_method="svd")
    np.testing.assert_array_equal(X, original)
    assert m.means_.shape == (8, 12)
    assert np.isfinite(m.score(X))

import numpy as np
from hss.analysis.cluster_structure import empirical_spectrum, factor_structure, subspace_overlap, variance_decomposition


def test_empirical_spectrum_and_variance_accounting():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(50, 7)) * np.arange(1, 8)
    spec, basis = empirical_spectrum(X, 7)
    covariance = np.cov(X, rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    np.testing.assert_allclose(spec['eigenvalues'], eigenvalues, rtol=1e-12)
    np.testing.assert_allclose(spec['trace'], np.trace(covariance), rtol=1e-12)
    np.testing.assert_allclose(spec['top_fraction'][-1], 1, atol=1e-12)
    np.testing.assert_allclose(basis @ basis.T, np.eye(7), atol=1e-12)
    parts = variance_decomposition(X, np.arange(50) % 3)
    assert parts['identity_error'] < 1e-10
    singleton, _ = empirical_spectrum(X[:1])
    assert singleton['trace'] == 0 and singleton['eigenvalues'] == []


def test_factor_diagnostics_do_not_depend_on_latent_rotation():
    rng = np.random.default_rng(9)
    W = rng.normal(size=(11, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    psi = rng.uniform(.1, 1, 11)
    a, ba = factor_structure(W, psi)
    b, bb = factor_structure(W @ Q, psi)
    np.testing.assert_allclose(a['factor_eigenvalues'], b['factor_eigenvalues'], rtol=1e-12)
    np.testing.assert_allclose(a['total_trace'], np.trace(W @ W.T + np.diag(psi)))
    np.testing.assert_allclose(subspace_overlap(ba, bb), 1, atol=1e-12)

import importlib.util
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Optional torch not installed"
)


def test_cuda_mfa_and_gmm_match_cpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from hss.cluster.mfa import fit_mfa
    from hss.cluster.gmm import _fit_gmm

    X = np.random.default_rng(81).normal(size=(80, 8))
    X[40:] += 4
    args = dict(rank=2, n_init=1, max_iter=8, chunk_size=23, tol=0.0)
    cpu = fit_mfa(X, 2, **args)
    gpu = fit_mfa(X, 2, backend="gpu", device="cuda:0", **args)
    assert gpu.backend_ == "gpu"
    np.testing.assert_allclose(
        cpu.score_samples(X), gpu.score_samples(X), rtol=1e-6, atol=1e-6
    )
    args = dict(
        covariance_type="diag",
        adaptive_reg=False,
        reg_covar=1e-5,
        n_init=1,
        max_iter=8,
        tol=0.0,
        chunk_size=23,
    )
    a = _fit_gmm(X, 2, 42, backend="cpu", **args)
    b = _fit_gmm(X, 2, 42, backend="gpu", device="cuda:0", **args)
    np.testing.assert_allclose(
        a.score_samples(X), b.score_samples(X), rtol=1e-4, atol=1e-4
    )

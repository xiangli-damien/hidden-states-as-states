import numpy as np
import pytest
from hss.cluster.mfa import fit_mfa


@pytest.mark.parametrize('rank', [0, 3])
def test_gpu_continues_cpu_checkpoint_with_same_em_updates(rank):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(91)
    X = rng.normal(size=(120, 15)) * np.linspace(.2, 2, 15)
    X[:60] += 3
    checkpoint = fit_mfa(X, 2, rank=rank, n_init=1, max_iter=4, tol=0, init_method='svd')
    cpu = fit_mfa(X, 2, rank=rank, n_init=1, max_iter=10, tol=0, initial_model=checkpoint)
    gpu = fit_mfa(X, 2, rank=rank, n_init=1, max_iter=10, tol=0, initial_model=checkpoint, backend='gpu')
    np.testing.assert_allclose(cpu.history_, gpu.history_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cpu.score_samples(X), gpu.score_samples(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(cpu.predict(X), gpu.predict(X))

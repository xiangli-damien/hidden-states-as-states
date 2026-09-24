from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_fixed_fa_keeps_common_anchor_and_matches_dense_gaussian_likelihood(tmp_path):
    pytest.importorskip('torch')
    from scipy.stats import multivariate_normal
    from fit_revision_fixed_factors import fit_one
    rng=np.random.default_rng(42);mean=rng.normal(size=7);w=rng.normal(size=(7,2))
    x=rng.normal(size=(500,2))@w.T+mean+rng.normal(size=(500,7))*.4
    staging=tmp_path/'staging';staging.mkdir();actual_mean=x.mean(0)
    np.save(staging/'training.npy',x);np.save(staging/'assignment.npy',np.zeros(len(x),int))
    np.save(staging/'empirical_means.npy',actual_mean[None])
    cfg={'staging':str(staging),'output':str(tmp_path/'out'),'blas_threads':1,'seed':42,
        'reg_covar':1e-5,'max_em_steps':2000,'em_tolerance':1e-5,'cluster_deadline_seconds':30,'accelerator':'squarem'}
    info=fit_one((cfg,2,0));assert info['converged']
    with np.load(tmp_path/'out/rank_2/component_000/model.npz') as f:
        mu=f['means'][0];W=f['loadings'][0];noise=f['noise'][0]
        np.testing.assert_allclose(mu,actual_mean,atol=1e-10)
        np.testing.assert_array_equal(f['common_anchor'],actual_mean)
        dense=multivariate_normal.logpdf(x,mean=mu,cov=W@W.T+np.diag(noise)).mean()
        assert dense==pytest.approx(info['mean_log_likelihood'],abs=1e-8)
    # Completed fits are immutable, checked through their saved receipts.
    assert fit_one((cfg,2,0))==info

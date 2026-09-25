import numpy as np
import pytest
from hss.cluster.gmm import _make_diag_model


def engine(x):
    pytest.importorskip('torch')
    from hss.cluster.gmm_resident import ResidentDiagonalEM
    return ResidentDiagonalEM(x,device='cpu',chunk_size=19)


def test_density_entropy_and_em_against_independent_numpy():
    rng=np.random.default_rng(13);x=rng.normal(size=(87,5))+np.array([1e5,4,-2,9,0])
    weights=np.array([.3,.7]);mu=x[[2,60]];var=rng.uniform(.3,2,size=(2,5))
    model=_make_diag_model(weights,mu,var,reg_covar=1e-6)
    e=engine(x);p=e.parameters(model);ll,new,ent,prob=e.evaluate(p,probabilities=True)
    reference=model.predict_proba(x)
    np.testing.assert_allclose(prob,reference,atol=4e-8)
    np.testing.assert_allclose(ll,model.score_samples(x).mean(),atol=1e-9)
    np.testing.assert_allclose(ent,-np.sum(prob*np.log(prob)),atol=1e-9)
    # Explicit centered residual covariance is independent of moment expansion.
    counts=prob.sum(0);new_mu=prob.T@x/counts[:,None]
    new_var=np.stack([(prob[:,j,None]*(x-new_mu[j])**2).sum(0)/counts[j]+1e-6 for j in range(2)])
    rebuilt=e.model(new,False,1)
    np.testing.assert_allclose(rebuilt.means_,new_mu,atol=1e-9)
    np.testing.assert_allclose(rebuilt.covariances_,new_var,atol=1e-9)


def test_single_gaussian_icland_convergence_not_iteration_limit():
    x=np.random.default_rng(5).normal(size=(100,3));e=engine(x)
    m,info,p=e.fit(1,42)
    assert m.converged_ and info['converged']
    np.testing.assert_allclose(m.means_[0],x.mean(0),atol=1e-10)
    np.testing.assert_allclose(m.covariances_[0],x.var(0)+1e-6,atol=1e-10)
    np.testing.assert_allclose(info['icl'],info['bic'],atol=1e-9)
    _,info,_=e.fit(3,42,max_iter=1,tol=0)
    assert not info['converged']


def test_translation_preserves_raw_geometry():
    x=np.random.default_rng(9).normal(size=(150,4))
    a,ia,_=engine(x).fit(3,4,max_iter=30)
    b,ib,_=engine(x+10000).fit(3,4,max_iter=30)
    np.testing.assert_allclose(a.means_+10000,b.means_,atol=1e-8)
    np.testing.assert_allclose(a.covariances_,b.covariances_,atol=1e-8)
    np.testing.assert_allclose(ia['icl'],ib['icl'],atol=1e-6)

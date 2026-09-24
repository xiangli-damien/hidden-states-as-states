from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.special import softmax,logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_factor_common import FactorDecoder,conditions,key,budget


def parameters():
    rng=np.random.default_rng(42)
    return (np.array([.2,.5,.3]),rng.normal(size=(3,7)),rng.normal(size=(3,7,2)),
            np.exp(rng.normal(size=(3,7))),rng.normal(size=(13,7)))


def test_dense_covariance_density_posterior_and_conditional_mean():
    weights,means,w,noise,x=parameters(); decoder=FactorDecoder(weights,means,w,noise)
    log=[];conditional=[]
    for k in range(3):
        covariance=w[k]@w[k].T+np.diag(noise[k])
        log.append(multivariate_normal.logpdf(x,mean=means[k],cov=covariance))
        conditional.append(means[k]+(w[k]@w[k].T@np.linalg.solve(covariance,(x-means[k]).T)).T)
    joint=np.stack(log,axis=1)+np.log(weights)
    probability=softmax(joint,axis=1);map_id=joint.argmax(1)
    expected_hard=np.stack([conditional[k][i] for i,k in enumerate(map_id)])
    expected_soft=np.einsum('nk,knd->nd',probability,np.stack(conditional))
    got_probability,density=decoder.posterior(x)
    np.testing.assert_allclose(got_probability,probability,atol=2e-14)
    np.testing.assert_allclose(density,logsumexp(joint,axis=1),atol=2e-13)
    for soft,expected in [(False,expected_hard),(True,expected_soft)]:
        actual,code=decoder.reconstruct(x,soft=soft)
        np.testing.assert_allclose(actual,expected,atol=1e-13)
        assert code['component']==map_id.tolist()


def test_posterior_shrinkage_is_not_orthogonal_projection():
    decoder=FactorDecoder([1.],[[0.,0.]],[[[2.],[0.]]],[[1.,1.]])
    x=np.array([[10.,3.]])
    np.testing.assert_allclose(decoder.assigned(x,[0]),[[8.,0.]])
    np.testing.assert_allclose(decoder.assigned(x,[0],orthogonal=True),[[10.,0.]])


def test_rotation_and_component_permutation_leave_soft_reconstruction_invariant():
    weights,means,w,noise,x=parameters();reference=FactorDecoder(weights,means,w,noise)
    rotation=np.linalg.qr(np.random.default_rng(4).normal(size=(2,2)))[0]
    permutation=np.array([2,0,1])
    changed=FactorDecoder(weights[permutation],means[permutation],(w@rotation)[permutation],noise[permutation])
    np.testing.assert_allclose(changed.reconstruct(x,soft=True)[0],reference.reconstruct(x,soft=True)[0],atol=2e-13)
    np.testing.assert_allclose(changed.posterior(x)[1],reference.posterior(x)[1],atol=2e-13)


def test_rank_zero_and_rank_deficient_loadings_do_not_invent_directions():
    x=np.array([[4.,5.,6.]])
    for w in [np.zeros((1,3,0)),np.zeros((1,3,2))]:
        decoder=FactorDecoder([1.],[[1.,2.,3.]],w,[[1.,1.,1.]])
        np.testing.assert_allclose(decoder.assigned(x,[0]),[[1.,2.,3.]])
        np.testing.assert_allclose(decoder.assigned(x,[0],orthogonal=True),[[1.,2.,3.]])
        assert decoder.column_bases[0].shape==(0,3)


def test_invalid_noise_is_not_silently_clipped():
    with pytest.raises(ValueError,match='positive noise'):
        FactorDecoder([1.],[[0.,0.]],np.zeros((1,2,1)),[[0.,1.]])


def test_soft_budget_and_unique_fixed_protocol():
    rows=conditions();assert len(rows)==19 and len({key(c) for c in rows})==19
    hard=budget({'method':'mfa_hard','rank':8},64,3584)
    soft=budget({'method':'mfa_soft','rank':8},64,3584)
    assert hard['continuous_values_per_token']==8 and hard['region_id']
    assert soft['continuous_values_per_token']==575 and not soft['region_id']
    assert hard['learned_scalars']==soft['learned_scalars']


def test_nonidentity_fa_patch_cached_losses_match_full_causal_forward():
    pytest.importorskip('torch');pytest.importorskip('transformers')
    from check_revision_factor_hook import run
    assert run()['passed']

import numpy as np
import pytest
from hss.analysis.change_clusters import residual_states, pair_positions, temporal_mean, gaussian_surrogate, aggregate_pairs


def test_depth_delta_commutes_with_token_average_and_pre_norm_replacement():
    rng=np.random.default_rng(3)
    h=rng.normal(size=(4,7,5,9))  # samples, tokens, layers, hidden
    pre=h[:,:,-1].copy(); post=h.copy(); post[:,:,-1]*=13
    mean=residual_states(post.mean(1),pre.mean(1))
    np.testing.assert_allclose(np.diff(mean,axis=1),np.diff(h,axis=2).mean(1),atol=2e-7)


def test_temporal_average_telescopes_and_never_crosses_questions():
    rng=np.random.default_rng(4);x=rng.normal(size=(19,7))
    np.testing.assert_allclose(np.diff(x,axis=0).mean(0),temporal_mean(x[0],x[-1],len(x)))
    for n in [1,2,7,99]:
        p=pair_positions(n,'question',8,921)
        assert len(p)==min(8,max(0,n-1)) and len(np.unique(p))==len(p)
        assert np.all(p>=0) and np.all(p+1<n)
        np.testing.assert_array_equal(p,pair_positions(n,'question',8,921))
    with pytest.raises(ValueError):temporal_mean(x[0],x[-1],1)


def test_gaussian_surrogate_preserves_correlations_not_mixture_labels():
    rng=np.random.default_rng(4);train=rng.normal(size=(200,3))@np.array([[3,1,0],[0,2,1],[1,1,2]])
    null=gaussian_surrogate(train,25000,17)
    np.testing.assert_allclose(np.cov(null,rowvar=False,bias=True),np.cov(train,rowvar=False,bias=True),rtol=.05,atol=.1)


def test_question_aggregation_gives_equal_question_weight():
    x=np.array([1,3,10]);owner=np.array([0,0,1])
    result=aggregate_pairs(x,owner,3)[:,0]
    np.testing.assert_allclose(result[:2],[2,10]);assert np.isnan(result[2])

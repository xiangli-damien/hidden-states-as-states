import numpy as np
import pytest
from hss.cluster.gmm import _make_diag_model
from scripts.run_qwen_reliability import (
    sample_indices, select, match_centers, uniform_baseline, symmetric_kl, geometry, variants,
)


def test_count_selection_is_not_fixed_and_excludes_unconverged():
    rows=[dict(k=2,icl=110.,converged=True),dict(k=3,icl=101.,converged=True),
          dict(k=4,icl=100.,converged=True),dict(k=5,icl=1.,converged=False)]
    assert select(rows,0,5)['k']==4
    assert select(rows,.02,5)['k']==3
    assert select(rows,0,3)['k']==3
    with pytest.raises(ValueError):select(rows,0,1)


def test_subset_is_unlabeled_nested_unique_and_both_protocols_included():
    a=sample_indices(100,.3,42);b=sample_indices(100,.8,42)
    assert len(a)==30 and len(b)==80 and set(a).issubset(b)
    np.testing.assert_array_equal(a,sample_indices(100,.3,42))
    cfg=dict(seeds=[42,43,44,45,46],primary_fractions=[.3,.5,.7,.9],figure_fractions=[.2,.4,.6,.8])
    assert len(variants(cfg))==13
    assert len({v['name'] for v in variants(cfg)})==13


def test_center_matching_ignores_number_permutation():
    x=np.array([[1.,2.,3.],[3.,-1.,2.],[2.,1.,-4.]])
    out=match_centers(x,x[[2,0,1]])
    assert abs(out['distance'])<1e-12 and out['unmatched_a']==0
    assert match_centers(x,x[:2])['unmatched_a']==1


def test_uniform_baseline_uses_real_sample_means():
    x=np.arange(42,dtype=float).reshape(14,3)+1;centers=x[[2,10]]
    rng=np.random.default_rng(7);labels=rng.integers(2,size=len(x))
    ref=np.stack([x[labels==j].mean(0) for j in range(2)])
    actual=uniform_baseline(x,centers,1,7)[0]
    np.testing.assert_allclose(actual['distance'],match_centers(centers,ref)['distance'],atol=1e-14)


def test_kl_matches_independent_direct_gaussian_formula_and_geometry_counts():
    ma=np.array([1.,2.]);mb=np.array([-1.,3.]);a=np.array([2.,4.]);b=np.array([1.,3.])
    kl=lambda mu,v,m,w:.5*np.sum(np.log(w/v)+(v+(mu-m)**2)/w-1)
    assert symmetric_kl(ma,a,ma,a)==0
    np.testing.assert_allclose(symmetric_kl(ma,a,mb,b),.5*(kl(ma,a,mb,b)+kl(mb,b,ma,a)))
    x=np.array([[1.,1.],[1.1,1.1],[4.,4.],[4.1,4.1]])
    m=_make_diag_model(np.array([.5,.5]),x[[0,2]],np.ones((2,2)),reg_covar=1e-6)
    cfg=dict(chunk_size=2,reg_covar=1e-6,random_repeats=2,subset_seed=42)
    out=geometry(x,m,cfg,0)
    assert out['counts']==[2,2] and len(out['between'])==1 and len(out['within'])==2
    assert np.isfinite(out['within_mean']) and np.isfinite(out['random_mean'])

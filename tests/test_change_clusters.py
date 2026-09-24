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


def test_fitting_needs_no_labels_and_saves_a_reusable_model(tmp_path):
    import json
    import joblib
    import pandas as pd
    from hss.analysis.change_clusters import fit_view
    out=tmp_path/'result';cache=tmp_path/'cache';out.mkdir();(cache/'views').mkdir(parents=True)
    rng=np.random.default_rng(8);x=np.r_[rng.normal(-3,.3,(80,4)),rng.normal(3,.3,(80,4))].astype(np.float32)
    np.save(cache/'views/mean_delta_L01.npy',x)
    split=np.tile(np.array(['train']*5+['validation']+['test']*2),20)
    pd.DataFrame(dict(sample_id=[str(i) for i in range(160)],split=split)).to_parquet(out/'splits.parquet')
    cfg=dict(output=str(out),cache=str(cache),k_grid=[1,2],seeds=[1,2],stability_seeds=[3,4],
             cpu_threads=1,max_iter=100,retry_max_iter=300,tol=.001,reg_covar=1e-5)
    result=fit_view(('mean_delta_L01',cfg))
    assert result['k']==2 and result['test_gain_nats_per_dimension']>0
    m=joblib.load(out/'fits/mean_delta_L01/selected_model.joblib')
    saved=np.load(out/'fits/mean_delta_L01/assignments.npz')
    np.testing.assert_array_equal(m.predict(x),saved['assignment'])
    assert len(result['candidates'])==4 and all(s['converged'] for s in result['stability'])


def test_controlled_permutation_keeps_group_composition():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from evaluate_change_clusters import permutation_test, bh
    y=np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)]
    h=np.eye(2)[y]
    assert permutation_test(h,y,y,99,1)['p']==1
    assert permutation_test(h,y,np.zeros(40),99,1)['p']==.01
    np.testing.assert_allclose(bh([.01,.04,.9]),[.03,.06,.9])

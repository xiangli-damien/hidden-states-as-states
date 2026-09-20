import numpy as np

from hss.route.counts import crossfit_scores, group_folds
from hss.route.nulls import ConditionalRouting, occupancy_null, shuffle_within, shuffle_suffixes


def test_group_split_and_scores_are_id_invariant():
    rng=np.random.default_rng(5)
    z=rng.integers(0,4,size=(160,5)); folds=group_folds(np.arange(160))
    assert group_folds(['same','same'])[0]==group_folds(['same','same'])[1]
    changed=np.column_stack([rng.permutation(4)[z[:,l]]+100*l for l in range(5)])
    a,_=crossfit_scores(z,folds); b,_=crossfit_scores(changed,folds)
    for k in a: np.testing.assert_allclose(a[k],b[k],atol=1e-12)


def test_scores_do_not_count_heldout_rows():
    z=np.array([[0,0],[0,1],[1,0],[1,1],[0,0],[1,0]])
    folds=np.array([0,0,0,1,1,1])
    scores,_=crossfit_scores(z,folds)
    # Fold0 train has origin0->0 once: P(0|0)=(1+1)/(1+2).
    np.testing.assert_allclose(scores['edge'][0],-np.log(2/3))
    changed=z.copy();changed[1,-1]=0
    other,_=crossfit_scores(changed,folds)
    np.testing.assert_allclose(scores['edge'][0],other['edge'][0])


def test_soft_scores_accept_probabilities_and_remain_finite():
    rng=np.random.default_rng(7); z=rng.integers(0,3,size=(100,4))
    p=[np.eye(3)[col] for col in z.T]
    s,_=crossfit_scores(z,group_folds(np.arange(100)),p)
    assert np.isfinite(s['soft_js']).all()
    assert np.all((s['soft_js']>=0)&(s['soft_js']<=1+1e-12))


def test_shuffle_preserves_every_group_column_histogram():
    rng=np.random.default_rng(4); a=rng.integers(0,8,(90,7)); g=np.arange(90)%9
    b=shuffle_within(a,g,rng)
    for group in np.unique(g):
        np.testing.assert_array_equal(np.sort(a[g==group],axis=0),np.sort(b[g==group],axis=0))


def test_route_signal_survives_occupancy_but_occupancy_only_does_not():
    rng=np.random.default_rng(3)
    a=np.tile([0,0,1,1],250); y=np.tile([0,1,0,1],250); c=np.zeros(len(y),int)
    # Every node marginal is class-independent, but transitions encode class.
    b=a^y; z=np.column_stack([a,b,a])
    test=ConditionalRouting(a,b,y,c).test(rng,199)
    assert test['js_bits']>.99 and test['p']==.005
    null=occupancy_null(z,y,c,rng,49)
    assert null['excess_bits']>.9 and null['p_greater']==.02
    # Destination equals class: purely a node effect, preserved in the null.
    pure=occupancy_null(np.column_stack([a,y]),y,c,rng,49)
    assert abs(pure['excess_bits'])<1e-12


def test_no_class_overlap_reports_zero_support():
    y=np.array([0,0,1,1]); a=np.zeros(4,int)
    r=ConditionalRouting(a,y,y,y).test(np.random.default_rng(2),19)
    assert r['support_n']==0 and r['p']==1 and np.isnan(r['js_bits'])


def test_suffix_null_preserves_all_edges_in_each_fold():
    rng=np.random.default_rng(29);z=rng.integers(0,4,(400,8));fold=np.arange(400)%5
    other=shuffle_suffixes(z,fold,rng)
    assert np.any(z!=other)
    for f in np.unique(fold):
        for l in range(7):
            a=np.sort(z[fold==f,l]*4+z[fold==f,l+1])
            b=np.sort(other[fold==f,l]*4+other[fold==f,l+1])
            np.testing.assert_array_equal(a,b)


def test_exact_js_bias_matches_monte_carlo():
    rng=np.random.default_rng(11);a=rng.integers(0,5,500);b=rng.integers(0,7,500);y=rng.integers(0,2,500)
    t=ConditionalRouting(a,b,y,np.zeros(500,int))
    sampled=t.test(rng,10000)['null_mean_bits']
    assert abs(t.exact_null_mean()-sampled)<.001

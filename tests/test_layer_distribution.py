import numpy as np

from hss.analysis.layer_distribution import covariance_profile, partition_profile, relative_icl_choices


def test_covariance_rotation_and_scale_invariants():
    rng=np.random.default_rng(29)
    x=rng.normal(size=(250,8))*np.arange(1,9)+np.arange(8)*2
    q,_=np.linalg.qr(rng.normal(size=(8,8)))
    lengths=np.arange(250)+1
    a,_=covariance_profile(x,lengths)
    b,_=covariance_profile(7*x@q,lengths)
    for key in ['effective_dimension','mean_energy_fraction','radius_fourth_ratio','log_length_variance_fraction']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-10)
    np.testing.assert_allclose(a['trace']*49,b['trace'],rtol=1e-10)


def test_partition_matches_direct_within_sum():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,5));y=np.arange(100)%4
    r=partition_profile(x,y)
    total=np.square(x-x.mean(0)).sum()
    within=sum(np.square(x[y==k]-x[y==k].mean(0)).sum() for k in range(4))
    np.testing.assert_allclose(r['between_variance_fraction'],1-within/total)


def test_delta_policy_ignores_score_origin_and_unconverged():
    rows=[dict(k=2,icl=100.,converged=True),dict(k=3,icl=80.,converged=True),dict(k=4,icl=-100.,converged=False)]
    a=relative_icl_choices(rows,[0.,.1,.2],10,10)
    shifted=[dict(r,icl=r['icl']-10000) for r in rows]
    assert a==relative_icl_choices(shifted,[0.,.1,.2],10,10)=={'0.0':3,'0.1':3,'0.2':2}

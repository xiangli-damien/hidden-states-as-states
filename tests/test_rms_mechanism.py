import numpy as np
from hss.analysis.rms_mechanism import token_factors


def test_norm_factorizations_and_gamma_not_unit_length():
    x=np.array([[2.,1.],[1.,3.]])
    gamma=np.array([2.,.5]);v=np.array([1.,0.])
    s,u,e,mx,my=token_factors(x,gamma,1e-6,v)
    np.testing.assert_allclose(my,gamma*u)
    np.testing.assert_allclose(s['post_mean_norm'],s['mean_post_token_norm']*s['post_coherence'])
    np.testing.assert_allclose(s['post_mean_norm'],s['u_mean_norm']*s['mean_gain'])
    assert s['mean_post_token_norm'] != np.sqrt(2)


def test_mean_can_shrink_with_equal_token_norms():
    g=np.ones(2);v=np.array([1.,0.])
    a=token_factors(np.array([[1.,0.],[1.,0.]]),g,1e-6,v)[0]
    b=token_factors(np.array([[1.,0.],[0.,1.]]),g,1e-6,v)[0]
    np.testing.assert_allclose(a['mean_post_token_norm'],b['mean_post_token_norm'])
    np.testing.assert_allclose(a['post_coherence'],1.)
    np.testing.assert_allclose(b['post_coherence'],1/np.sqrt(2))


def test_post_norm_depends_on_gamma_weighted_direction_not_raw_radius():
    x=np.array([[2.,1.],[1.,3.]])
    g=np.array([2.,.5]);v=np.array([1.,0.])
    a=token_factors(x,g,0.,v)
    b=token_factors(x*np.array([[3.],[7.]]),g,0.,v)
    np.testing.assert_allclose(a[1],b[1])
    np.testing.assert_allclose(a[4],b[4])
    expected=2*np.sum(x*x*g*g,axis=1)/np.sum(x*x,axis=1)
    actual=np.sum((x/np.sqrt(np.mean(x*x,axis=1,keepdims=True))*g)**2,axis=1)
    np.testing.assert_allclose(expected,actual)

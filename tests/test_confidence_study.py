import numpy as np
import pandas as pd
from scipy.special import softmax
from hss.analysis.confidence_data import distribution_scalars, head_geometry
from hss.analysis.confidence_study import fit_fixed, control_features, cosine


def test_full_vocabulary_entropy_margin_and_extreme_logits():
    x = np.eye(3,dtype=np.float32)
    w = np.array([[1000,0,3],[999,0,0],[-1000,0,-1],[997,0,2]],dtype=np.float32)
    result = distribution_scalars(x,w,batch=2)
    p = softmax(x.astype(float)@w.T,axis=1)
    expected = -(p*np.log(np.maximum(p,1e-300))).sum(1)
    np.testing.assert_allclose(result.entropy,expected,rtol=1e-6)
    np.testing.assert_allclose(result.logit_margin,[1,0,1])
    np.testing.assert_allclose(result.top1_probability,p.max(1),rtol=1e-6)


def test_softmax_null_direction_folds_gamma_and_removes_common_logit():
    w = np.array([[1,2,9],[-1,-2,9],[2,4,9]],float)
    gamma = np.array([2.,.5,3.])
    sv,b = head_geometry(w,gamma,chunk=2)
    a = (w-w.mean(0))*gamma
    assert sv[0]<1e-7
    np.testing.assert_allclose(a@b[:,0],0,atol=1e-6)
    np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-10)


def test_raw_probe_direction_matches_scaled_linear_score_and_ignores_test_labels():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(160,8))*np.arange(1,9)
    y=(x[:,0]+x[:,3]/4>0).astype(int)
    train=np.arange(100)
    fit=fit_fixed(x,y,train,.1)
    altered=y.copy();altered[100:]=1-altered[100:]
    other=fit_fixed(x,altered,train,.1)
    np.testing.assert_array_equal(fit['coef'],other['coef'])
    np.testing.assert_allclose(fit['score'],x@fit['coef']+fit['intercept'])
    assert cosine(fit['coef'],other['coef'])>.999999


def test_pregeneration_nuisance_does_not_consume_opening_tokens():
    rows=pd.DataFrame({'category':['a']*50,'level':[1]*50,'first_token':[3]*25+[4]*25,'n_prompt_tokens':10})
    x=np.ones((50,4));train=np.arange(50)
    _,names=control_features(rows,x,train,False)
    _,diagnostic=control_features(rows,x,train,True)
    assert not any('first_token' in s for s in names)
    assert any('first_token' in s for s in diagnostic)

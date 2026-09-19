import numpy as np
import pandas as pd
from scipy.special import softmax
from hss.analysis.confidence_data import distribution_scalars, head_geometry
from hss.analysis.confidence_study import fit_fixed, control_features, cosine
from hss.analysis.confidence_precision import penalize
from hss.analysis.confidence_bootstrap import auc_samples
from sklearn.metrics import roc_auc_score
from hss.analysis.confidence_readout_check import intervention


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


def test_repetition_penalty_uses_unique_prompt_ids_and_sign():
    z=np.array([[2.,-2.,4.],[3.,6.,-9.]])
    result=penalize(z,[[0,0,1],[2]],2.)
    np.testing.assert_array_equal(result,[[1.,-4.,4.],[3.,6.,-18.]])
    np.testing.assert_array_equal(z,[[2.,-2.,4.],[3.,6.,-9.]])


def test_fast_bootstrap_matches_literal_resampling_with_score_ties():
    y=np.array([0,0,1,0,1,1,0,1])
    s=np.array([.2,.2,.3,.4,.8,.1,.1,.8])
    fast=auc_samples(y,s,7,80)
    rng=np.random.default_rng(7);groups=[np.flatnonzero(y==k) for k in [0,1]]
    expected=[]
    for _ in range(80):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        expected.append(roc_auc_score(y[idx],s[idx]))
    np.testing.assert_allclose(fast,expected,atol=1e-14)


def test_readout_intervention_identity_and_exact_null_temperature_control():
    rng=np.random.default_rng(13);h=rng.normal(size=(7,3));v=np.array([0.,0.,1.])
    w=rng.normal(size=(12,3));w[:,2]=0;gamma=np.array([.4,2.,1.5])
    base,changed,temp,modified=intervention(h,v,v,0.,w,gamma,1e-5)
    direct=(modified/np.sqrt(np.mean(modified**2,axis=1,keepdims=True)+1e-5)*gamma)@w.T
    np.testing.assert_allclose(changed,direct,atol=1e-12)
    np.testing.assert_allclose(changed,temp,atol=1e-12)
    np.testing.assert_array_equal(base.argmax(1),changed.argmax(1))

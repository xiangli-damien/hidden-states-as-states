import numpy as np
from hss.analysis.component_data import raw_moments, centered_energy


def test_aggregation_order_is_not_interchangeable():
    x=np.array([[10.,0.],[-1.,1.]])
    m=raw_moments(x)
    np.testing.assert_allclose(m['q'],[.75,.25])
    q_mean=m['mean'][0]**2/np.sum(m['mean']**2)
    q_pooled=m['second'][0]/m['second'].sum()
    assert not np.isclose(q_mean,m['q'][0])
    assert not np.isclose(q_pooled,m['q'][0])
    np.testing.assert_allclose(m['top1'].sum(),1.)


def test_centered_moments_match_explicit_token_distances():
    x=np.array([[3.,1.,-2.],[1.,-5.,3.],[4.,7.,2.]])
    center=np.array([2.,-1.,5.]);m=raw_moments(x)
    expected=np.mean(np.sum((x-center)**2,axis=1))/3
    np.testing.assert_allclose(centered_energy(m['mean'],m['second'],center),expected)
    within=centered_energy(m['mean'],m['second'],m['mean'])
    np.testing.assert_allclose(expected,within+np.mean((m['mean']-center)**2))


def test_fraction_invariant_but_raw_energy_changes_with_token_scaling():
    x=np.array([[2.,3.],[4.,1.]])
    a=raw_moments(x);b=raw_moments(x*np.array([[7.],[2.]]))
    np.testing.assert_allclose(a['q'],b['q'])
    assert not np.allclose(a['second'],b['second'])


def test_layer_diagnostic_uses_pre_norm_and_paired_terminal_updates(tmp_path,monkeypatch):
    import pandas as pd
    from hss.analysis import component_reduction as study
    states=np.ones((4,3,2));states[:,-1]=999.  # Deliberately misleading post-norm slot.
    pre=np.array([[3.,2.],[3.,2.],[5.,2.],[5.,2.]])
    raw=states.copy();raw[:,-1]=pre
    class Data:
        rows=pd.DataFrame({'y':[0,0,1,1]})
        info={'model':{'n_layers':3}}
        def array(self,view):
            return pre if view.startswith('pre_') else states
    monkeypatch.setattr(study,'layer_array',lambda cfg,data,name,layer:raw[:,layer])
    cfg={'seed':2,'bootstrap':20}
    result=study.layer_statistics(cfg,{},Data(),'synthetic',0,np.array([],dtype=int),np.arange(4),tmp_path)
    for view in ['prompt_last','mean','t16']:
        v=result[view]['absolute_coordinate_update']
        assert v['correct_mean']==4. and v['incorrect_mean']==2.
        np.testing.assert_allclose(v['difference_ci'],[2.,2.])
    table=pd.read_parquet(tmp_path/'layers.parquet')
    assert table[(table.layer==2)&(table.metric=='absolute')]['mean'].max()==5.

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_completion_common import transform,target_regions,choose_candidate,ordered,test_conditions as conditions


def fixture():
    centers=np.array([[0.,0.,0.],[2.,0.,0.],[5.,0.,0.]])
    decoder={'centers':centers,'local_basis':np.tile(np.eye(3)[None,:1],(3,1,1)), 'shared_basis':np.eye(3)[:1]}
    support={'question_count':[40,40,19],'shrunk_correctness':[.2,.8,.95]}
    cfg={'rank':1,'nearest_candidates':3,'support_min_questions':20,'support_min_gain':.1,'validation_questions':64,'random_seeds':[42,137,271]}
    return decoder,support,cfg


def test_soft_projection_and_c2_mask():
    d,s,c=fixture();x=np.array([[.1,3.,2.],[2.,2.,3.]])
    z,meta=transform(x,d,s,c,{'family':'c1','alpha':.3},'a')
    np.testing.assert_allclose(z,[[.1,2.1,1.4],[2.,1.4,2.1]])
    z,meta=transform(x,d,s,c,{'family':'c2','alpha':1.},'a')
    assert meta['target']==[1,1] and meta['mask']==[True,False]
    np.testing.assert_array_equal(z[1],x[1])
    np.testing.assert_allclose(z[0],[.1,0.,0.])


def test_random_matches_selected_delta_and_mask():
    d,s,c=fixture();x=np.array([[.1,3.,2.],[2.,2.,3.]])
    z,m=transform(x,d,s,c,{'family':'c2','alpha':.3},'a')
    q,n=transform(x,d,s,c,{'family':'random','target_family':'c2','alpha':.3,'seed':42},'a')
    assert m==n
    np.testing.assert_allclose(np.linalg.norm(z-x,axis=1),np.linalg.norm(q-x,axis=1),atol=1e-12)
    np.testing.assert_allclose(np.sum(x*(z-x),axis=1),np.sum(x*(q-x),axis=1),atol=1e-12)
    np.testing.assert_array_equal(q[1],x[1])


def test_validation_selection_fails_closed():
    _,_,cfg=fixture()
    r={'split':'validation','n':64,'net_correct':0,'parse_failure_increase':0,'mean_actual_energy':1.,'condition':{'name':'c1_0.1','family':'c1','alpha':.1}}
    assert choose_candidate([r],cfg) is None
    assert choose_candidate([dict(r,net_correct=2)],cfg)==r['condition']
    assert choose_candidate([dict(r,net_correct=2,parse_failure_increase=3)],cfg) is None
    with pytest.raises(ValueError):choose_candidate([dict(r,split='test')],cfg)
    assert len(conditions({'name':'c2_0.1','family':'c2','alpha':.1},cfg))==7


def test_question_selection_independent_of_order():
    assert ordered(['b','a','c'],'fixed')==ordered(['c','b','a'],'fixed')

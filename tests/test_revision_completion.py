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


def test_raw_statistics_pipeline_and_tamper_detection(tmp_path):
    """Synthetic outcomes only: exercises selection, seed averaging and audit."""
    import json,time
    from revision_common import write_json,sha
    from summarize_revision_completion import select,summarize
    from audit_revision_completion_statistics import run as audit
    cfg={'validation_questions':64,'test_questions':4,'fallback_test_questions':2,
         'reserve_hours':6,'bootstrap_draws':60,'bootstrap_seed':42}
    candidates=[{'name':'c1_0.1','family':'c1','alpha':.1},{'name':'c2_0.1','family':'c2','alpha':.1}]
    write_json(tmp_path/'plan.json',{'config':cfg,'candidates':candidates,
        'deadline_unix':time.time()+48*3600,'validation_decision_unix':time.time()+22*3600})
    def stage(name,n,methods):
        paths={}
        for i in range(n):
            for method in methods:
                base=bool(i%2);correct=base
                if method=='c1_0.1' and i in (0,2):correct=True
                if method=='random_137' and i==0:correct=True
                record={'sample_id':f'q{i}','split':'test' if name=='test' else 'validation',
                    'condition':{'name':method},'correct':correct,'parse_failed':False,'length':40,
                    'finish_reason':'eos','geometry':{'energy':float(method!='baseline'),'modified_tokens':int(method!='baseline')},'seconds':1.}
                p=tmp_path/name/f'q{i}_{method}.json';write_json(p,record);paths[str(p.relative_to(tmp_path))]=sha(p)
        write_json(tmp_path/f'{name}_SUCCESS.json',{'records':paths})
        write_json(tmp_path/f'{name}_audit.json',{'complete':True,'stage_receipt_sha256':sha(tmp_path/f'{name}_SUCCESS.json')})
    stage('smoke',8,['baseline','c1_0.1','c2_0.1'])
    stage('validation',64,['baseline','c1_0.1','c2_0.1']);select(tmp_path)
    selection=json.loads((tmp_path/'selection.json').read_text())
    assert selection['selected']['name']=='c1_0.1' and selection['test_questions']==4
    stage('test',4,['baseline','c1_0.1','shared8','random_42','random_137','random_271'])
    summarize(tmp_path);audit(tmp_path)
    assert json.loads((tmp_path/'steering_statistics_audit.json').read_text())['complete']
    bad=json.loads((tmp_path/'steering_summary.json').read_text());bad['rows'][0]['accuracy']['estimate']+=.1
    write_json(tmp_path/'steering_summary.json',bad)
    with pytest.raises(AssertionError):audit(tmp_path)

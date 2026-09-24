"""Scientific contracts and synthetic reporting integrity, not model results."""
from pathlib import Path
import sys,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from projection_v2_common import project,catalog,derangement
from projection_v2_audit_math import independent_projection
from revision_completion_common import transform
from run_projection_v2_queue import choose_tier


def fixture():
    rng=np.random.default_rng(42)
    bases=np.stack([np.linalg.qr(rng.normal(size=(80,80)))[0].T for _ in range(3)])
    decoder=dict(centers=rng.normal(size=(3,80)),local_basis=bases,shared_basis=bases[0])
    return rng.normal(size=(16,80)),decoder,derangement(3,24)


def test_all_operators_independent_arithmetic_and_wrong_anchor():
    x,d,perm=fixture()
    for c in catalog():
        before=x[-c['width']:].copy();z,meta=project(before,d,c,perm)
        expected,labels,fallback=independent_projection(before,d['centers'],d['local_basis'],d['shared_basis'],c,perm,1e-12)
        np.testing.assert_allclose(z,expected,atol=1e-12)
        assert meta['current']==labels.tolist()
        if c['operator']=='wrong':assert np.all(np.asarray(meta['basis_region'])!=labels)
        np.testing.assert_array_equal(before,x[-c['width']:])


def test_original_operator_compatibility_and_sign():
    x,d,perm=fixture();c=dict(name='local8',prefix=16,width=4,rank=8,alpha=1.,operator='local')
    for alpha in [0.,.1,.3,1.,-.3]:
        z,_=project(x,d,dict(c,alpha=alpha),perm)
        old,_=transform(x,d,{},dict(rank=8),dict(family='c1',alpha=alpha),'q')
        np.testing.assert_array_equal(z,old)
    plus,_=project(x,d,dict(c,alpha=.3),perm);minus,_=project(x,d,dict(c,alpha=-.3),perm)
    np.testing.assert_allclose(plus-x,-(minus-x),atol=1e-14)


def test_common_residual_preserves_differences_norm_fallback():
    x,d,perm=fixture();cs={c['name']:c for c in catalog()}
    common,_=project(x[-4:],d,cs['D10'],perm)
    np.testing.assert_allclose(np.diff(common,axis=0),np.diff(x[-4:],axis=0),atol=1e-13)
    norm,_=project(x,d,cs['D09'],perm)
    np.testing.assert_allclose(np.linalg.norm(norm,axis=1),np.linalg.norm(x,axis=1),atol=1e-13)
    d=dict(centers=np.zeros((2,80)),local_basis=np.tile(np.eye(80)[None],(2,1,1)),shared_basis=np.eye(80))
    v=np.zeros((1,80));v[0,-1]=3
    z,m=project(v,d,cs['D09'],[1,0]);np.testing.assert_array_equal(z,v);assert m['norm_fallback']==[True]


def test_budget_has_no_outcome_inputs_and_no_partial_primary_reduction():
    plan=dict(no_new_batch_unix=100000,conditions=[dict(c,asset_available=True) for c in catalog()])
    assert choose_tier(plan,25,0)['expected_generations']==640
    assert choose_tier(plan,25,90000)['name']=='reduced'
    assert choose_tier(plan,25,99000)['expected_generations']==0
    assert np.all(derangement(64,9242026)!=np.arange(64))


def test_synthetic_summary_audit_report_and_tamper(tmp_path):
    from revision_common import write_json,sha
    from summarize_projection_v2 import run as summarize
    from audit_projection_v2_statistics import run as audit
    from report_projection_v2 import run as report
    main=tmp_path/'primary';root=tmp_path/'extension';root.mkdir()
    decoder=tmp_path/'decoder.npz'
    np.savez(decoder,centers=np.zeros((2,8)),local_basis=np.tile(np.eye(8)[None],(2,1,1)),shared_basis=np.eye(8))
    write_json(main/'support.json',dict(question_count=[30,30]))
    def record(folder,sid,split,name,correct):
        p=folder/(name+'.json');write_json(p,dict(sample_id=sid,split=split,condition=dict(name=name),correct=bool(correct),
            prompt_text='Synthetic question '+sid,ground_truth='1',response_text='Synthetic answer',parsed_answer='1',normalized_answer=str(int(correct)),
            length=30,parse_failed=False,finish_reason='eos',geometry=dict(energy=0.,modified_tokens=0),seconds=1.,files={}))
        write_json(p.with_suffix('.receipt.json'),dict(record_sha256=sha(p)));return p
    reuse={}
    for split in ['validation','test']:
        paths={}
        for i in range(4):
            sid=f'{split}_{i}';folder=main/'outputs'/sid
            rr={}
            for name in ['baseline','c1_1.0','c1_0.3']:
                p=record(folder,sid,split,name,(i%2==1) or (name=='c1_1.0' and i==0))
                paths[str(p.relative_to(main))]=sha(p);rr[name]=str(p)
            if split=='validation':reuse[sid]=rr
        write_json(main/f'{split}_SUCCESS.json',dict(records=paths))
    primary=dict(rows=[dict(stage='test',method='baseline',n=4)],paired_controls=[])
    write_json(main/'steering_summary.json',primary);write_json(main/'steering_statistics_audit.json',dict(complete=True,source_summary_sha256=sha(main/'steering_summary.json')))
    write_json(root/'plan.json',dict(config=dict(protected_primary=str(main),decoder=str(decoder)),conditions=[dict(c,asset_available=True) for c in catalog()]))
    write_json(root/'reuse.json',reuse);write_json(root/'tier.json',dict(name='synthetic',transfer_conditions=[]))
    paths={}
    for i,sid in enumerate(reuse):
        p=record(root/'outputs/D01'/sid,sid,'validation','D01',i%2==0);paths[str(p.relative_to(root))]=sha(p)
    write_json(root/'D01_SUCCESS.json',dict(records=paths));write_json(root/'D01_audit.json',dict(complete=True,source_receipt_sha256=sha(root/'D01_SUCCESS.json')))
    summarize(root);audit(root);report(root)
    receipt=json.loads((root/'statistics_audit.json').read_text());assert receipt['complete'] and receipt['all_flip_pairs_included']>0
    assert (root/'report/paired_changes.pdf').exists()
    bad=json.loads((root/'summary.json').read_text());bad['comparisons'][0]['delta_accuracy']['estimate']+=.1
    write_json(root/'summary.json',bad)
    with pytest.raises(AssertionError):audit(root)

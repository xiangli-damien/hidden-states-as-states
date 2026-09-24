"""Independent scalar-sum patch and question-paired statistics checks."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from revision_common import config,sha,write_json


def expected(x,d,name):
    s=min(range(len(d['centers'])),key=lambda j:sum((x[0]-d['centers'][j].astype(float))**2))
    mu=d['centers'][s].astype(float)
    if name=='identity':return x.copy(),s
    if name=='global_mean':return d['train_mean'][None].astype(float).copy(),s
    z=mu.copy()
    if name!='centroid':
        axes=d['shared_basis'] if name=='shared8' else d['local_basis'][s if name=='local8' else d['permutation_'+name.rsplit('_',1)[1]][s],:8]
        for axis in axes.astype(float):z+=sum((x[0]-mu)*axis)*axis
    return z[None],s


def audit(cfg,smoke=False):
    import torch
    root=Path(cfg['output']);dest=root/('smoke' if smoke else 'functional')
    plan=json.loads((root/'plan.json').read_text())
    for p,h in plan['files'].items():assert sha(p)==h,p
    cases=json.loads((root/'cases.json').read_text())
    if smoke:cases=[c for c in cases if c['sample_id'] in plan['smoke_ids']]
    receipt=json.loads((dest/'_SUCCESS.json').read_text());assert receipt['plan_sha256']==sha(root/'plan.json')
    assert len(list((dest/'samples').glob('*.json')))==len(cases)*8==receipt['conditions']
    with np.load(root/'decoder.npz') as z:d={k:z[k].copy() for k in z.files}
    largest=0.;checked=0
    for c in cases:
        sid=c['sample_id']
        with np.load(dest/'samples'/(sid+'__identity.npz')) as z:original=z['logp'].astype(float);clean=z['x'].astype(float)
        for name in plan['conditions']:
            p=dest/'samples'/(sid+'__'+name+'.json');r=json.loads(p.read_text())
            assert (r['sample_id'],r['dataset'],r['condition'])==(sid,c['dataset'],name)
            assert r['positions']==[len(c['prompt_ids'])-1] and r['plan_sha256']==sha(root/'plan.json')
            assert r['arrays_sha256']==sha(p.with_suffix('.npz'))
            with np.load(p.with_suffix('.npz')) as z:a={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(a['x'],clean)
            ideal,s=expected(clean,d,name);assert r['meta']['region']==s
            np.testing.assert_allclose(a['ideal'],ideal,rtol=1e-11,atol=1e-10)
            actual=torch.as_tensor(a['ideal']).to(torch.bfloat16).float().numpy()
            np.testing.assert_array_equal(actual,a['actual'])
            loss=a['reference_nll'];assert len(loss)==len(c['response_ids']) and np.all(loss>=0) and np.isfinite(loss).all()
            np.testing.assert_allclose(np.exp(a['logp'].astype(float)).sum(),1,atol=2e-6)
            metrics={'next_token_kl':sum(float(np.exp(u)*(u-v)) for u,v in zip(original,a['logp'])),
                'nll':float(np.mean(loss)),'nll_sum':float(sum(loss)),'first_token_nll':float(loss[0]),
                'first16_nll':float(np.mean(loss[:16])),'later_nll':float(np.mean(loss[16:])),
                'reconstruction_sse':float(np.sum((actual-clean)**2)),'token_mse':float(np.mean((actual-clean)**2)),
                'train_centered_energy':float(np.sum((clean-d['train_mean'])**2))}
            for k,v in metrics.items():
                np.testing.assert_allclose(r['metrics'][k],v,rtol=1e-9,atol=1e-10);largest=max(largest,abs(r['metrics'][k]-v))
            np.testing.assert_allclose(r['metrics']['actual_patch_energy'],metrics['reconstruction_sse'],rtol=2e-6,atol=1e-7)
            assert r['metrics']['patch_calls']==1 and r['metrics']['reference_tokens']==len(loss)
            assert r['metrics']['argmax_agreement']==bool(original.argmax()==a['logp'].argmax())
            if name=='identity':assert r['identity_plain_logits_exact'];np.testing.assert_array_equal(actual,clean)
            checked+=1
    write_json(dest/'audit.json',{'complete':True,'questions':len(cases),'records':checked,'max_metric_arithmetic_error':largest,
        'prepared_inputs_sha_verified':len(plan['files']),'independent_projection_sums':True,'bf16_exact':True,
        'scope':'Independent coverage/provenance/arithmetic audit; no new semantic scoring or independent full-model repetition of every intervention.'})
    print(json.dumps({'audited':checked,'max_error':largest}),flush=True)


def audit_report(cfg):
    root=Path(cfg['output']);report=root/'report';summary=json.loads((report/'summary.json').read_text())
    with (report/'per_question.csv').open() as f:rows=list(csv.DictReader(f))
    idx={(r['sample_id'],r['condition']):r for r in rows};cases=json.loads((root/'cases.json').read_text())
    with np.load(report/'bootstrap.npz') as z:boots={k:z[k] for k in z.files}
    def vals(ds,name,key):
        ids=[c['sample_id'] for c in cases if c['dataset']==ds]
        if name=='wrong8':return np.array([np.mean([float(idx[s,'wrong8_'+str(seed)][key]) for seed in cfg['seeds']]) for s in ids])
        return np.array([float(idx[s,name][key]) for s in ids])
    checks=0
    for r in summary['paired']:
        v=vals(r['dataset'],r['method'],r['metric'])-vals(r['dataset'],r['control'],r['metric'])
        draws=np.array([sum(v[ix])/len(ix) for ix in boots[r['dataset']]])
        np.testing.assert_allclose(r['estimate'],sum(v)/len(v),atol=1e-12)
        np.testing.assert_allclose(r['ci95'],np.percentile(draws,[2.5,97.5]),atol=1e-12);checks+=1
    for r in summary['conditions']:
        for k in ['next_token_kl','delta_nll','delta_first16_nll','delta_later_nll','reconstruction_sse','argmax_agreement']:
            np.testing.assert_allclose(r[k],vals(r['dataset'],r['condition'],k).mean(),atol=1e-12);checks+=1
        a=vals(r['dataset'],r['condition'],'reconstruction_sse');b=vals(r['dataset'],r['condition'],'train_centered_energy')
        np.testing.assert_allclose(r['nmse'],sum(a)/sum(b),atol=1e-12);checks+=1
    for row in rows:
        raw=json.loads((root/'functional/samples'/(row['sample_id']+'__'+row['condition']+'.json')).read_text())
        base=json.loads((root/'functional/samples'/(row['sample_id']+'__identity.json')).read_text())
        for k in ['next_token_kl','nll','reconstruction_sse','train_centered_energy']:
            assert float(row[k])==raw['metrics'][k]
        for k in ['nll','first16_nll','later_nll']:
            np.testing.assert_allclose(float(row['delta_'+k]),raw['metrics'][k]-base['metrics'][k],atol=1e-15)
    write_json(report/'statistics_audit.json',{'complete':True,'rows':len(rows),'summary_checks':checks,'summary_sha256':sha(report/'summary.json')})
    print(json.dumps({'statistics_audit':True,'rows':len(rows),'checks':checks}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();cfg=config(a.config)
    if a.report:audit_report(cfg)
    else:audit(cfg,a.smoke)

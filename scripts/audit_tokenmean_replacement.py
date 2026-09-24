"""Independent row-wise arithmetic and legacy-output audit, no operator imports."""
import argparse
import json
from pathlib import Path
import numpy as np
from revision_common import config, sha, write_json


def expected_patch(x, mean, token, name):
    m=x.mean(0)
    scores=[]
    for k in range(len(mean['weights'])):
        r=m-mean['centers'][k];v=mean['variances'][k]
        scores.append(float(np.log(mean['weights'][k])-.5*np.sum(np.log(2*np.pi*v)+r*r/v)))
    s=int(np.argmax(scores))
    if name=='identity':return x.copy(),s
    if name.startswith('tokenmap_'):
        out=[];method=name.split('_',1)[1]
        for row in x:
            j=min(range(len(token['centers'])),key=lambda j:np.sum((row-token['centers'][j].astype(float))**2))
            mu=token['centers'][j].astype(float);value=mu.copy()
            if method!='centroid':
                b=token['local_basis'][j,:8] if method=='local8' else token['shared_basis'][:8]
                for axis in b.astype(float):value+=np.dot(row-mu,axis)*axis
            out.append(value)
        return np.array(out),s
    shift=name.startswith('mean_shift_')
    method=name[len('mean_shift_' if shift else 'token_project_'):]
    mu=mean['centers'][s];source=[m] if shift else x
    if method=='centroid':axes=[]
    elif method=='shared8':axes=mean['shared_basis']
    elif method=='local8':axes=mean['local_basis'][s]
    else:axes=mean['local_basis'][mean['permutation_'+method.rsplit('_',1)[1]][s]]
    values=[]
    for row in source:
        value=mu.copy()
        for axis in axes:value+=np.dot(row-mu,axis)*axis
        values.append(value)
    return (x+values[0]-m if shift else np.array(values)),s


def audit(cfg,smoke=False):
    import torch
    root=Path(cfg['output']);dest=root/('smoke' if smoke else 'functional')
    plan=json.loads((root/'plan.json').read_text())
    for path,digest in plan['files'].items():assert sha(path)==digest,path
    receipt=json.loads((dest/'_SUCCESS.json').read_text())
    assert receipt['plan_sha256']==sha(root/'plan.json')
    cases=json.loads((root/'cases.json').read_text())
    if smoke:cases=[c for c in cases if c['sample_id'] in plan['smoke_ids']]
    with np.load(root/'mean_decoder.npz') as f:mean={k:f[k] for k in f.files}
    with np.load(root/'token_decoder.npz') as f:token={k:f[k] for k in f.files}
    with np.load(root/'captures.npz') as f:captures=dict(zip(f['sample_ids'].tolist(),f['x'].astype(float)))
    assert len(list((dest/'samples').glob('*.json')))==len(cases)*len(plan['conditions'])==receipt['conditions']
    largest_error=0.;legacy=0;checked=0
    legacy_names={'identity':'identity','tokenmap_centroid':'centroid',
                  'tokenmap_shared8':'shared_8','tokenmap_local8':'local_8'}
    for case in cases:
        sid=case['sample_id'];x=captures[sid]
        basefile=dest/'samples'/(sid+'__identity.npz')
        with np.load(basefile) as f:original=f['logp'].astype(float)
        prefixlen=len(case['prompt_ids'])+cfg['prefix_tokens']
        for name in plan['conditions']:
            path=dest/'samples'/(sid+'__'+name+'.json');r=json.loads(path.read_text())
            assert (r['sample_id'],r['dataset'],r['condition'])==(sid,case['dataset'],name)
            assert r['plan_sha256']==sha(root/'plan.json') and sha(path.with_suffix('.npz'))==r['arrays_sha256']
            assert r['positions']==list(range(prefixlen-16,prefixlen))
            with np.load(path.with_suffix('.npz')) as f:data={k:f[k] for k in f.files}
            z,s=expected_patch(x,mean,token,name)
            np.testing.assert_allclose(data['ideal'],z,rtol=1e-12,atol=1e-10)
            # Evaluate rounding on the saved ideal, after independent ideal verification.
            actual=torch.as_tensor(data['ideal']).to(torch.bfloat16).float().numpy()
            np.testing.assert_array_equal(actual,data['actual'])
            assert r['meta']['mean_region']==s and np.isfinite(data['logp']).all()
            np.testing.assert_allclose(np.exp(data['logp'].astype(float)).sum(),1,atol=2e-6)
            losses=data['reference_nll'];assert len(losses)==len(case['response_ids'])-16
            assert np.isfinite(losses).all() and np.all(losses>=0)
            m=r['metrics'];assert m['patch_calls']==1 and m['reference_tokens']==len(losses)
            kl=sum(float(np.exp(a)*(a-b)) for a,b in zip(original,data['logp']))
            centered_delta=(actual.astype(float)-actual.astype(float).mean(0))-(x-x.mean(0))
            expected={'next_token_kl':kl,'nll':float(losses.mean()),'nll_sum':float(losses.sum()),
                      'first16_nll':float(losses[:16].mean()),'first_token_nll':float(losses[0]),
                      'token_mse':float(((actual-x)**2).mean()),
                      'mean_mse':float(((actual.astype(float).mean(0)-x.mean(0))**2).mean()),
                      'centered_change_mse':float(np.square(centered_delta).mean())}
            for key,val in expected.items():
                np.testing.assert_allclose(m[key],val,rtol=1e-9,atol=1e-10)
                largest_error=max(largest_error,abs(m[key]-val))
            if name.startswith('mean_shift_'):
                np.testing.assert_allclose(z-z.mean(0),x-x.mean(0),rtol=1e-11,atol=1e-10)
            if name=='identity':np.testing.assert_array_equal(actual,x)
            if name in legacy_names:
                old=Path(case['legacy_samples'])/(sid+'_p16_l14_tokens_w16_'+legacy_names[name]+'.json')
                oldrow=json.loads(old.read_text())
                assert sha(old.with_suffix('.npz'))==oldrow['arrays_sha256']
                with np.load(old.with_suffix('.npz')) as f:
                    np.testing.assert_array_equal(data['logp'],f['logp'])
                    np.testing.assert_array_equal(losses,f['reference_nll'])
                legacy+=1
            checked+=1
    write_json(dest/'audit.json',{'complete':True,'conditions':checked,'questions':len(cases),
        'legacy_conditions_exact':legacy,'maximum_metric_arithmetic_error':largest_error,
        'frozen_files_verified':len(plan['files']),'code_sha256':sha(__file__),
        'operator': 'independent per-row Gaussian and projection sums; shift invariants; exact bf16; no operator helper imported',
        'scope':'Arithmetical/coverage/provenance checks plus exact legacy logprob/NLL replay; not semantic correctness or independent model re-execution of every new condition.'})
    print(json.dumps({'complete':True,'conditions':checked,'legacy_exact':legacy}),flush=True)


def audit_report(cfg):
    import csv
    root=Path(cfg['output']);report=root/'report'
    with (report/'per_question.csv').open() as f:rows=list(csv.DictReader(f))
    index={(r['dataset'],r['sample_id'],r['condition']):r for r in rows}
    summary=json.loads((report/'summary.json').read_text())
    cases=json.loads((root/'cases.json').read_text())
    with np.load(report/'bootstrap.npz') as f:boots={k:f[k] for k in f.files}
    checked=0
    def values(ds,name,metric):
        out=[]
        for c in cases:
            if c['dataset']!=ds:continue
            sid=c['sample_id']
            if name.endswith('_wrong8'):
                out.append(sum(float(index[ds,sid,name+'_'+str(seed)][metric]) for seed in cfg['seeds'])/3)
            else:out.append(float(index[ds,sid,name][metric]))
        return np.array(out)
    for row in summary['paired']:
        delta=values(row['dataset'],row['method'],row['metric'])-values(row['dataset'],row['control'],row['metric'])
        estimates=[]
        for ix in boots[row['dataset']]:estimates.append(sum(delta[int(i)] for i in ix)/len(ix))
        np.testing.assert_allclose(row['estimate'],sum(delta)/len(delta),atol=1e-12)
        np.testing.assert_allclose(row['ci95'],np.percentile(estimates,[2.5,97.5]),atol=1e-12)
        checked+=1
    for row in summary['conditions']:
        vals=values(row['dataset'],row['condition'],row['metric'])
        np.testing.assert_allclose(row['estimate'],sum(vals)/len(vals),atol=1e-12)
        checked+=1
    # Ensure exported per-question values are exactly sourced from audited raw records.
    for row in rows:
        record=json.loads((root/'functional/samples'/(row['sample_id']+'__'+row['condition']+'.json')).read_text())
        base=json.loads((root/'functional/samples'/(row['sample_id']+'__identity.json')).read_text())
        for key in ['next_token_kl','nll','token_mse','mean_mse','centered_change_mse']:
            assert float(row[key])==record['metrics'][key]
        np.testing.assert_allclose(float(row['delta_nll']),record['metrics']['nll']-base['metrics']['nll'],atol=1e-15)
    write_json(report/'statistics_audit.json',{'complete':True,'rows':len(rows),'summary_checks':checked,
        'summary_sha256':sha(report/'summary.json'),'scope':'Independent bootstrap arithmetic using saved question resamples; raw-to-table join verified.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report',action='store_true');a=p.parse_args();cfg=config(a.config)
    if a.report:audit_report(cfg)
    else:audit(cfg,a.smoke)

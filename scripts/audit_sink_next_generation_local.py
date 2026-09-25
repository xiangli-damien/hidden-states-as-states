"""Independent local count/exact-test audit after the full frozen GPU trial.

Never change primary scores. Exact paired-bootstrap distribution is computed
by convolution, independently of the frozen auditor's Monte Carlo bootstrap.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

from sink_direction_common import boxed_answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(left, right):
    delta=np.asarray(left,int)-np.asarray(right,int)
    n=len(delta);wins=int((delta>0).sum());losses=int((delta<0).sum());discordant=wins+losses
    p=min(1.,2*sum(comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    distribution=np.array([1.])
    masses=np.array([losses,n-discordant,wins],float)/n
    for _ in range(n):distribution=np.convolve(distribution,masses)
    assert np.isclose(distribution.sum(),1.)
    cdf=np.cumsum(distribution)
    interval=[(int(np.searchsorted(cdf,q))-n)/n for q in [.025,.975]]
    return dict(n=n,wins=wins,losses=losses,net=wins-losses,difference=(wins-losses)/n,
                p_two_sided_exact=p,exact_bootstrap_ci95=interval)


def run(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    audit=json.loads((root/'audit_SUCCESS.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==sha(root/'summary.json')
    assert audit['plan_sha256']==sha(root/'plan.json')==summary['plan_sha256']
    done=json.loads((root/'GPU_COMPLETE.json').read_text())
    assert done['records']==606 and done['table_sha256']==sha(root/'scores.parquet')
    cases=json.loads((root/'cases.json').read_text())
    assert len(cases)==202 and len({r['question_id'] for r in cases})==202
    grouped={};norm_steps=0;worst_energy=0.
    for case in cases:
        q=case['question_id'];grouped[q]={}
        for cond in plan['conditions']:
            path=root/'records'/q/(cond+'.json')
            r=json.loads(path.read_text());receipt=json.loads(path.with_suffix('.receipt.json').read_text())
            assert receipt['sha256']==sha(path) and receipt['plan_sha256']==sha(root/'plan.json')
            assert receipt['arrays_sha256']==r['arrays_sha256']==sha(path.with_suffix('.npz'))
            assert r['question_id']==q and r['condition']==cond and r['set']==case['set']
            assert r['n_tokens']==len(r['generated_ids'])<=2048
            assert r['boxed']==(boxed_answer(r['response_text']) is not None)
            assert r['primary_success']==(r['boxed'] and r['n_tokens']<2048)
            with np.load(path.with_suffix('.npz')) as arrays:
                assert np.isfinite(arrays['response_mean']).all()
                norms=arrays['update_norms_and_cosines']
                if cond=='zero':assert len(norms)==0
                else:
                    assert len(norms)==r['n_tokens']-1 and np.isfinite(norms).all()
                    target,actual=norms[:,0].astype(float),norms[:,1].astype(float)
                    assert np.array_equal(target>0,actual>0)
                    cfg=plan['config']
                    assert np.all(np.abs(actual-target)<=cfg['norm_match_relative_tolerance']*target+cfg['norm_match_absolute_tolerance']+1e-7)
                    energy=np.abs(actual**2-target**2)/np.maximum(target**2,1e-30)
                    assert energy.max(initial=0)<=cfg['relative_total_energy_tolerance']+1e-6
                    worst_energy=max(worst_energy,float(energy.max(initial=0)));norm_steps+=len(norms)
            grouped[q][cond]=r
    groups={}
    for group in ['sink','normal']:
        ids=sorted(q for q,arms in grouped.items() if arms['zero']['set']==group)
        expected=summary['groups'][group];assert len(ids)==expected['n']
        arms={};contrasts={}
        for cond in plan['conditions']:
            rows=[grouped[q][cond] for q in ids]
            arms[cond]=dict(n=len(ids),primary_count=sum(r['primary_success'] for r in rows),
                            correct=sum(r['correct'] for r in rows),boxed=sum(r['boxed'] for r in rows),
                            at_cap=sum(r['n_tokens']==2048 for r in rows),mean_tokens=float(np.mean([r['n_tokens'] for r in rows])),
                            sink_count=sum(r['sink'] for r in rows),
                            exit_vs_zero=sum(grouped[q]['zero']['sink'] and not grouped[q][cond]['sink'] for q in ids),
                            entry_vs_zero=sum(not grouped[q]['zero']['sink'] and grouped[q][cond]['sink'] for q in ids))
            for key in ['primary_count','correct','boxed','sink_count','exit_vs_zero','entry_vs_zero']:
                assert arms[cond][key]==expected['arms'][cond][key],(group,cond,key)
            assert np.isclose(arms[cond]['mean_tokens'],expected['arms'][cond]['mean_tokens'])
        for other in ['zero','ORTH_MAN_1']:
            primary=paired([grouped[q]['C1']['primary_success'] for q in ids],[grouped[q][other]['primary_success'] for q in ids])
            ref=expected['primary_comparisons'][other]
            for key in ['n','wins','losses','net','difference','p_two_sided_exact']:
                assert np.isclose(primary[key],ref[key],atol=1e-12),(group,other,key)
            assert np.max(np.abs(np.array(primary['exact_bootstrap_ci95'])-ref['ci95']))<=2/len(ids)
            contrasts[other]=dict(primary=primary,automatic_correctness=paired([grouped[q]['C1']['correct'] for q in ids],[grouped[q][other]['correct'] for q in ids]))
        if group=='sink':
            ordered=sorted(contrasts,key=lambda c:contrasts[c]['primary']['p_two_sided_exact'])
            low=min(1.,2*contrasts[ordered[0]]['primary']['p_two_sided_exact'])
            adj={ordered[0]:low,ordered[1]:max(low,contrasts[ordered[1]]['primary']['p_two_sided_exact'])}
            for cond,pvalue in adj.items():
                assert np.isclose(pvalue,expected['primary_comparisons'][cond]['holm_p'])
                contrasts[cond]['primary']['holm_p']=pvalue
        groups[group]=dict(n=len(ids),arms=arms,contrasts=contrasts)
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),records=606,questions=202,
                plan_sha256=sha(root/'plan.json'),summary_sha256=sha(root/'summary.json'),
                all_record_and_array_hashes_verified=True,all_primary_endpoints_reparsed=True,
                checked_online_steps=norm_steps,max_relative_step_energy_error=worst_energy,
                groups=groups,scope='Independent complete-cohort counts, exact McNemar/Holm, exact paired-bootstrap distribution; original automatic correctness unchanged. Original remote audit additionally replayed frozen GMM assignment.',
                semantic_review='Separate; this audit does not verify mathematical derivations or final-answer equivalence.')
    (root/'independent_local_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

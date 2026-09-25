"""Complete-cohort, frozen two-sided McNemar/Holm analysis for online C1."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from revision_common import sha,write_json
from report_sink_energy_matched import bootstrap,interval
from sink_direction_common import boxed_answer
from hss_followup_common import gmm_assign


def comparison(a,b,seed):
    d=np.asarray(a,int)-np.asarray(b,int)
    wins=int((d>0).sum());losses=int((d<0).sum())
    p=float(binomtest(wins,wins+losses,.5,alternative='two-sided').pvalue) if wins+losses else 1.
    bs=bootstrap(d[:,None],100000,seed)[:,0]
    return dict(n=len(d),wins=wins,losses=losses,net=wins-losses,difference=float(d.mean()),ci95=interval(bs,.95),p_two_sided_exact=p)


def holm(pvalues):
    order=np.argsort(pvalues);out=np.empty(len(order));last=0.
    for i,j in enumerate(order):
        last=max(last,min(1.,(len(order)-i)*pvalues[j]));out[j]=last
    return out.tolist()


def run(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    for p,h in plan['files'].items():assert sha(p)==h,p
    done=json.loads((root/'GPU_COMPLETE.json').read_text())
    assert done['records']==606 and done['plan_sha256']==sha(root/'plan.json') and done['table_sha256']==sha(root/'scores.parquet')
    cases=json.loads((root/'cases.json').read_text());rows=[]
    maps=dict(np.load(Path(cfg['source'])/'maps.npz'))
    for case in cases:
        for cond in plan['conditions']:
            p=root/'records'/case['question_id']/(cond+'.json');r=json.loads(p.read_text());rec=json.loads(p.with_suffix('.receipt.json').read_text())
            assert rec['sha256']==sha(p) and rec['plan_sha256']==sha(root/'plan.json') and rec['arrays_sha256']==r['arrays_sha256']==sha(p.with_suffix('.npz'))
            assert r['question_id']==case['question_id'] and r['condition']==cond and r['set']==case['set']
            assert len(r['generated_ids'])==r['n_tokens']<=2048
            assert r['boxed']==(boxed_answer(r['response_text']) is not None)
            assert r['primary_success']==(r['boxed'] and r['n_tokens']<2048)
            with np.load(p.with_suffix('.npz')) as z:
                k=int(gmm_assign(z['response_mean'],maps['l14_means'],maps['l14_covariances'],maps['l14_weights'])[0])
                assert r['cluster']==k and r['sink']==(k==plan['sink_local'])
                norms=z['update_norms_and_cosines']
                if cond!='zero':
                    assert len(norms)==r['n_tokens']-1
                    target,actual=norms[:,0].astype(float),norms[:,1].astype(float)
                    assert np.array_equal(target>0,actual>0)
                    assert np.all(abs(actual-target)<=cfg['norm_match_relative_tolerance']*target+cfg['norm_match_absolute_tolerance']+1e-7)
                    # Cached batch1 generation forwards one generated position at a time.
                    e=abs(actual**2-target**2)/np.maximum(target**2,1e-30)
                    assert e.max(initial=0)<=cfg['relative_total_energy_tolerance']+1e-6
                else:assert len(norms)==0
            rows.append(r)
    frame=pd.DataFrame(rows);assert len(frame)==606 and not frame.duplicated(['question_id','condition']).any()
    groups={}
    for g,sub in frame.groupby('set'):
        ids=sorted(sub.question_id.unique());assert len(ids)==(152 if g=='sink' else 50)
        indexed=sub.set_index(['question_id','condition'])
        arms={c:indexed.xs(c,level='condition').loc[ids] for c in plan['conditions']}
        contrasts={}
        for other in ['zero','ORTH_MAN_1']:
            contrasts[other]=comparison(arms['C1'].primary_success,arms[other].primary_success,plan['seed'])
        if g=='sink':
            adjusted=holm([contrasts[c]['p_two_sided_exact'] for c in ['zero','ORTH_MAN_1']])
            for c,p in zip(['zero','ORTH_MAN_1'],adjusted):contrasts[c]['holm_p']=p
        effects={}
        for c,a in arms.items():
            exits=int((arms['zero'].sink & ~a.sink).sum());entries=int((~arms['zero'].sink & a.sink).sum())
            effects[c]=dict(primary_count=int(a.primary_success.sum()),boxed=int(a.boxed.sum()),correct=int(a.correct.sum()),
                mean_tokens=float(a.n_tokens.mean()),sink_count=int(a.sink.sum()),exit_vs_zero=exits,entry_vs_zero=entries,net_exit=exits-entries,
                mean_update_norm=float(np.mean([x['mean_norm'] for x in a.online_audit])),mean_total_update_energy=float(np.mean([x['total_energy'] for x in a.online_audit])))
        strata={}
        if g=='sink':
            for origin in ['historical_test','S_B']:
                qq=arms['zero'].origin.eq(origin)
                strata[origin]=dict(n=int(qq.sum()),counts={c:int(a.loc[qq,'primary_success'].sum()) for c,a in arms.items()})
        groups[g]=dict(n=len(ids),arms=effects,primary_comparisons=contrasts,strata=strata)
    utc=datetime.now(timezone.utc).isoformat();primary=groups['sink']['primary_comparisons']
    summary=dict(completed_utc=utc,plan_sha256=sha(root/'plan.json'),records=606,groups=groups,
        submission_eligible=datetime.fromisoformat(utc)<=datetime.fromisoformat(plan['cutoff_utc']),
        decision=dict(improves_vs_both=all(c['difference']>0 and c['holm_p']<.05 for c in primary.values())),
        limitations=['Reused, previously inspected questions; exploratory.',
            'Online rule matching uses each arm current activation; realized paths and aggregate energy can differ.',
            'Boxed completion is not mathematical correctness; report both.',
            'Post-generation membership uses unhooked replay of the new text, not hooked online activations.'])
    write_json(root/'summary.json',summary);write_json(root/'audit_SUCCESS.json',dict(passed=True,utc=utc,summary_sha256=sha(root/'summary.json'),records=606,plan_sha256=sha(root/'plan.json')))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

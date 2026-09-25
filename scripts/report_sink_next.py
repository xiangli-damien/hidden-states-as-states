"""Audit raw generalization records, then compute question-level frozen endpoints."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import sha,write_json
from report_sink_energy_matched import bootstrap,interval


def run(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    for p,h in plan['files'].items():assert sha(p)==h,p
    done=json.loads((root/'GPU_COMPLETE.json').read_text())
    assert done['plan_sha256']==sha(root/'plan.json') and done['table_sha256']==sha(root/'scores.parquet')
    specs=json.loads((root/'specs.json').read_text());notes=json.loads((root/'notes.json').read_text())
    expected={(s['name'],c['question_id'],k) for s in specs for c in s['cases'] for k in s['conditions']}
    assert len(expected)==plan['expected_records']==done['records']
    rows=[];geometry=[];max_energy=0.;max_error=0.
    for spec in specs:
        for case in spec['cases']:
            base=root/'records'/spec['name']/case['question_id']
            with np.load(base/'C1.npz') as z:target=z['actual_norm'].astype(float);active=z['active']
            for cond in spec['conditions']:
                p=base/(cond+'.json');r=json.loads(p.read_text());receipt=json.loads(p.with_suffix('.receipt.json').read_text())
                assert receipt['plan_sha256']==sha(root/'plan.json') and receipt['sha256']==sha(p)
                assert receipt['arrays_sha256']==r['arrays_sha256']==sha(p.with_suffix('.npz'))
                with np.load(p.with_suffix('.npz')) as z:
                    lp=z['logprobs'];assert len(lp)==r['n_tokens'] and np.isfinite(lp).all()
                    nll=float(-lp[r['cut']:].mean(dtype=np.float64));assert abs(nll-r['nll'])<1e-12
                    if cond!='NONE':
                        actual=z['actual_norm'].astype(float);t=z['target_norm'].astype(float)
                        np.testing.assert_array_equal(t,target);np.testing.assert_array_equal(z['active'],active);np.testing.assert_array_equal(actual>0,active)
                        err=abs(actual-t)/(cfg['norm_match_relative_tolerance']*t+cfg['norm_match_absolute_tolerance'])
                        assert err.max()<=1.000001
                        energy=abs((actual**2).sum()-(t**2).sum())/max((t**2).sum(),1e-30)
                        assert energy<=cfg['relative_total_energy_tolerance']
                        max_energy=max(max_energy,float(energy));max_error=max(max_error,float(err.max()))
                        geometry.append(dict(spec=spec['name'],question_id=case['question_id'],set=case['set'],condition=cond,
                            active_fraction=float(active.mean()),mean_norm=float(actual.mean()),energy=float((actual**2).sum()),
                            intended_cosine_weighted=float((z['actual_cosine_intended']*actual**2).sum()/max((actual**2).sum(),1e-30)),
                            sink_leakage_rms=float(np.sqrt((z['actual_cosine_sink']**2*actual**2).sum()/max((actual**2).sum(),1e-30))),
                            exceedance_fraction=r['ideal_exceedance_fraction'],mean_positive_excess=r['mean_positive_excess']))
                rows.append(r)
    frame=pd.DataFrame(rows)
    assert set(zip(frame.spec,frame.question_id,frame.condition))==expected
    assert not frame.duplicated(['spec','question_id','condition']).any()
    tab=pd.read_parquet(root/'scores.parquet').set_index(['spec','question_id','condition'])
    for r in rows:assert abs(tab.loc[(r['spec'],r['question_id'],r['condition']),'nll']-r['nll'])<1e-12
    summaries={};boots_by={};values_by={}
    for j,spec in enumerate(specs):
        name=spec['name'];groups={}
        for g,sub in frame[frame.spec==name].groupby('set'):
            t=sub.pivot(index='question_id',columns='condition',values='nll').reindex(columns=spec['conditions'])
            assert t.notna().all().all()
            delta=t.subtract(t.NONE,axis=0)
            b=bootstrap(delta.to_numpy(),plan['bootstrap_replicates'],plan['seed']+j*13+(1 if g in ['out','normal','matched'] else 0))
            contrast=delta.C1 if len(t.columns)==2 else delta.C1-delta.iloc[:,2:].mean(axis=1)
            bc=b[:,1] if len(t.columns)==2 else b[:,1]-b[:,2:].mean(axis=1)
            level=plan['primary_ci'] if g in ['in','sink'] else .95
            groups[g]=dict(n=len(t),ci_level=level,c1_vs_none=dict(mean=float(delta.C1.mean()),ci=interval(b[:,1],level)),
                contrast=dict(mean=float(contrast.mean()),ci=interval(bc,level)),
                individual_deltas={c:dict(mean=float(delta[c].mean()),ci=interval(b[:,i],level)) for i,c in enumerate(t.columns)})
            if len(t.columns)>2:
                groups[g]['control_mean_vs_none']=dict(mean=float(delta.iloc[:,2:].to_numpy().mean()),ci=interval(b[:,2:].mean(axis=1),level))
                groups[g]['c1_minus_each_control']={c:dict(mean=float((delta.C1-delta[c]).mean()),ci=interval(b[:,1]-b[:,i],.995)) for i,c in enumerate(t.columns) if i>=2}
            boots_by[name,g]=bc;values_by[name,g]=contrast
        summaries[name]=dict(layer=spec['layer'],groups=groups)
        for key in ['cluster','training_count','training_correctness','training_types','cosine_axis14','excluded_centroids','control_pairs','control_centroid_ids']:
            if key in spec:summaries[name][key]=spec[key]
        if plan['experiment']==3:
            summaries[name]['in_minus_out']=dict(mean=float(values_by[name,'in'].mean()-values_by[name,'out'].mean()),ci=interval(boots_by[name,'in']-boots_by[name,'out'],.95),ci_level=.95,unit='Independent question resampling within in and out groups')
    if plan['experiment']==1:
        def paired(pairs):
            x=np.array([values_by['matched','sink'][p['sink']]-values_by['matched','matched'][p['matched']] for p in pairs])
            b=bootstrap(x[:,None],plan['bootstrap_replicates'],plan['seed'])[:,0]
            return dict(n=len(x),mean=float(x.mean()),ci=interval(b,.95),ci_level=.95)
        summaries['matched']['primary_paired']=paired(notes['pairs'])
        exact=[p for p in notes['pairs'] if p['level_gap']==0]
        summaries['matched']['exact_only']=paired(exact) if exact else None
        summaries['matched']['matching']=dict(exact=notes['exact'],relaxed=notes['relaxed'],unmatched=notes['unmatched'])
    geo=pd.DataFrame(geometry);geo.to_parquet(root/'geometry.parquet',index=False)
    geometric=[]
    for (s,g,c),d in geo.groupby(['spec','set','condition']):
        geometric.append(dict(spec=s,set=g,condition=c,n=len(d),**{k:float(d[k].mean()) for k in ['active_fraction','mean_norm','energy','intended_cosine_weighted','sink_leakage_rms']},
            exceedance_fraction=float(d.exceedance_fraction.mean()) if c=='C1' else None,mean_positive_excess=float(d.mean_positive_excess.mean()) if c=='C1' else None))
    utc=datetime.now(timezone.utc).isoformat()
    summary=dict(experiment=plan['experiment'],plan_sha256=sha(root/'plan.json'),completed_utc=utc,
        submission_eligible=datetime.fromisoformat(utc)<=datetime.fromisoformat(plan['cutoff_utc']),records=len(rows),
        summaries=summaries,geometry=geometric,audit=dict(passed=True,max_relative_energy_error=max_energy,max_token_tolerance_fraction=max_error),
        notes='Exploratory on reused questions. Matched metadata does not rule out all confounding; depth effects are not dose-normalized; selected common states do not represent every state.')
    write_json(root/'summary.json',summary);write_json(root/'audit_SUCCESS.json',dict(passed=True,utc=utc,summary_sha256=sha(root/'summary.json'),plan_sha256=sha(root/'plan.json'),records=len(rows)))
    print(json.dumps(dict(experiment=plan['experiment'],summaries=summaries,audit=summary['audit']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

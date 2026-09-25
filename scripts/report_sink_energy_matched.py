"""CPU audit and frozen analysis of energy-matched sink controls, from raw records."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_common import sha, write_json


def bootstrap(x, B, seed):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    out = np.empty((B, x.shape[1]))
    for i in range(0, B, 500):
        w = rng.multinomial(len(x), np.full(len(x), 1/len(x)), size=min(500, B-i))
        out[i:i+len(w)] = w @ x / len(x)
    return out


def interval(x, level):
    return np.quantile(x, [(1-level)/2, (1+level)/2]).tolist()


def run(root):
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text());cfg=plan['config']
    complete = json.loads((root/'GPU_COMPLETE.json').read_text())
    assert complete['plan_sha256']==sha(root/'plan.json')
    assert complete['table_sha256']==sha(root/'scores.parquet')
    for p,h in plan['files'].items():assert sha(p)==h,p
    cases=json.loads((root/'cases.json').read_text())
    saved=pd.read_parquet(root/'scores.parquet').set_index(['question_id','condition'])
    assert len(saved)==196*12 and saved.index.is_unique
    source=pd.read_parquet(Path(cfg['source'])/'step2_screen_long.parquet').set_index(['question_id','condition'])
    rows=[];diagnostics=[];norm_errors=[];energy_errors=[];max_replay=0.
    for case in cases:
        sid=case['question_id']
        with np.load(root/'records'/sid/'C1.npz') as z:
            target=z['actual_norm'].copy();mask=z['active'].copy()
        for cond in cfg['conditions']:
            p=root/'records'/sid/(cond+'.json')
            rec=json.loads(p.read_text());receipt=json.loads(p.with_suffix('.receipt.json').read_text())
            assert receipt['sha256']==sha(p) and receipt['plan_sha256']==sha(root/'plan.json')
            assert receipt['arrays_sha256']==rec['arrays_sha256']==sha(p.with_suffix('.npz'))
            assert rec['set']==case['set'] and rec['cut']==case['cut'] and rec['n_tokens']==case['n_tokens']
            with np.load(p.with_suffix('.npz')) as z:
                lp=z['logprobs'];assert len(lp)==case['n_tokens'] and np.isfinite(lp).all()
                nll=float(-lp[case['cut']:].mean(dtype=np.float64)) if case['set']=='sink' else float(-lp.mean(dtype=np.float64))
                assert abs(nll-rec['nll'])<1e-12
                assert abs(nll-float(saved.loc[(sid,cond),'nll']))<1e-12
                if cond in ['NONE','C1']:
                    col='m_loop' if case['set']=='sink' else 'm_col_nll'
                    err=abs(nll-float(source.loc[(sid,cond),col]));max_replay=max(max_replay,err)
                    assert err<=cfg['baseline_replay_nll_tolerance']
                if cond!='NONE':
                    actual=z['actual_norm'].astype(float);t=z['target_norm'].astype(float)
                    np.testing.assert_array_equal(t,target)
                    np.testing.assert_array_equal(z['active'],mask)
                    np.testing.assert_array_equal(actual>0,mask)
                    error=abs(actual-t);allowed=cfg['norm_match_relative_tolerance']*t+cfg['norm_match_absolute_tolerance']
                    assert (error<=allowed+1e-7).all()
                    e=abs(np.square(actual).sum()-np.square(t).sum())/max(np.square(t).sum(),1e-30)
                    assert e<=cfg['relative_total_energy_tolerance']
                    np.testing.assert_allclose(actual.mean(),rec['geometry']['mean_actual_delta_norm'],atol=5e-6,rtol=1e-6)
                    norm_errors.append(float((error/allowed).max()));energy_errors.append(float(e))
                    diagnostics.append(dict(question_id=sid,set=case['set'],condition=cond,
                        tokens=len(lp),active=int(mask.sum()),repaired=int(z['rounding_repaired'].sum()),
                        target_norm_sum=float(t.sum()),actual_norm_sum=float(actual.sum()),
                        update_energy=float(np.square(actual).sum()),
                        intended_cosine_energy_sum=float((z['actual_cosine_intended']*np.square(actual)).sum()),
                        sink_cosine_sq_energy_sum=float((z['actual_cosine_sink']**2*np.square(actual)).sum()),
                        max_absolute_cosine_sink=float(abs(z['actual_cosine_sink'][mask]).max()) if mask.any() else 0.,
                        max_tolerance_fraction=float((error/allowed).max()),relative_total_energy_error=float(e)))
            rows.append(dict(question_id=sid,set=case['set'],condition=cond,nll=nll,cut_rule=case['cut_rule']))
    assert len(rows)==plan['expected_records']==complete['records']
    frame=pd.DataFrame(rows);controls=cfg['conditions'][2:]
    groups={}
    for offset,group in enumerate(['sink','normal']):
        t=frame[frame['set']==group].pivot(index='question_id',columns='condition',values='nll').reindex(columns=cfg['conditions'])
        assert len(t)==cfg['expected_'+group]
        delta=t.subtract(t.NONE,axis=0)
        boots=bootstrap(delta.to_numpy(),cfg['bootstrap_replicates'],cfg['seed']+offset)
        estimates={}
        for j,c in enumerate(t.columns):
            estimates[c]=dict(mean=float(delta[c].mean()),ci=interval(boots[:,j],cfg['primary_ci_level']))
        vsmean=(delta.C1-delta[controls].mean(axis=1)).to_numpy()
        contrast_boot=boots[:,1]-boots[:,2:].mean(axis=1)
        pairwise={c:dict(mean=float((delta.C1-delta[c]).mean()),ci=interval(boots[:,1]-boots[:,j],cfg['individual_control_ci_level']))
                  for j,c in enumerate(controls,2)}
        groups[group]=dict(n=len(t),deltas_vs_NONE=estimates,
            control_mean_vs_NONE=dict(mean=float(delta[controls].to_numpy().mean()),ci=interval(boots[:,2:].mean(axis=1),cfg['primary_ci_level'])),
            c1_minus_control_mean=dict(mean=float(vsmean.mean()),ci=interval(contrast_boot,cfg['primary_ci_level'])),
            c1_minus_each_control=pairwise,
            individual_point_means_exceeded=sum(x['mean']>0 for x in pairwise.values()),
            individual_adjusted_intervals_positive=sum(x['ci'][0]>0 for x in pairwise.values()))
    dg=pd.DataFrame(diagnostics);geometry=[]
    for (group,cond),d in dg.groupby(['set','condition']):
        geometry.append(dict(set=group,condition=cond,n_questions=len(d),tokens=int(d.tokens.sum()),
            active=int(d.active.sum()),rounding_repaired=int(d.repaired.sum()),
            mean_actual_norm=float(d.actual_norm_sum.sum()/d.tokens.sum()),
            intended_cosine_energy_weighted=float(d.intended_cosine_energy_sum.sum()/d.update_energy.sum()),
            sink_cosine_energy_weighted_rms=float(np.sqrt(d.sink_cosine_sq_energy_sum.sum()/d.update_energy.sum())),
            max_absolute_cosine_sink=float(d.max_absolute_cosine_sink.max()),
            max_tolerance_fraction=float(d.max_tolerance_fraction.max()),
            max_relative_total_energy_error=float(d.relative_total_energy_error.max())))
    decision=dict(energy_audit_passed=True,
        P1_mean_control_CI_positive=groups['sink']['c1_minus_control_mean']['ci'][0]>0,
        P2_all10_point_means_exceeded=groups['sink']['individual_point_means_exceeded']==10,
        all10_individual_adjusted_CIs_positive=groups['sink']['individual_adjusted_intervals_positive']==10)
    decision['frozen_rule_passed']=decision['P1_mean_control_CI_positive'] and decision['P2_all10_point_means_exceeded']
    summary=dict(config=cfg,plan_sha256=sha(root/'plan.json'),record_count=len(rows),
        primary_ci_level=cfg['primary_ci_level'],individual_ci_level=cfg['individual_control_ci_level'],
        unit='question; mean NLL in nats/token; source suffix for sink versus source full response for normal',
        groups=groups,geometry=geometry,decision=decision,
        audit=dict(max_NONE_C1_replay_error=max_replay,max_norm_tolerance_fraction=max(norm_errors),
            max_relative_response_energy_error=max(energy_errors),all_record_hashes_checked=True,all_token_masks_and_recorded_norms_checked=True,
            no_original_full_activation_tensors_retained=True),
        limitations=['Same previously inspected questions, not fresh-sample confirmation.',
            'BF16 updates approximate the intended direction; report energy-weighted actual cosines and repaired-token counts.',
            'A likelihood effect does not establish free-generation escape or recovery.',
            'Normal references were selected automated-correct and boxed, not independently human-verified.'])
    write_json(root/'summary.json',summary)
    write_json(root/'audit_SUCCESS.json',dict(passed=True,summary_sha256=sha(root/'summary.json'),
        plan_sha256=sha(root/'plan.json'),table_sha256=sha(root/'scores.parquet'),records=len(rows)))
    dg.to_parquet(root/'energy_audit.parquet',index=False)
    print(json.dumps(dict(decision=decision,groups=groups,audit=summary['audit']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

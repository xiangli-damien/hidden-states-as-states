"""Finite validation selection and honest question-level steering statistics."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import sha,write_json
from revision_completion_common import choose_candidate


def records(root,stage):
    success=json.loads((root/f'{stage}_SUCCESS.json').read_text())
    audit=json.loads((root/f'{stage}_audit.json').read_text())
    assert audit['complete'] and audit['stage_receipt_sha256']==sha(root/f'{stage}_SUCCESS.json')
    result=[]
    for rel,h in success['records'].items():
        assert sha(root/rel)==h; r=json.loads((root/rel).read_text())
        result.append({'sample_id':r['sample_id'],'split':r['split'],'name':r['condition']['name'],
            'correct':int(r['correct']),'parse_failed':int(r['parse_failed']),
            'length':r['length'],'truncated':int(r['finish_reason']=='length'),
            'energy':r['geometry']['energy'],'modified':int(r['geometry']['modified_tokens']>0),'seconds':r['seconds']})
    return pd.DataFrame(result)


def interval(x,cfg):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(cfg['bootstrap_seed'])
    sample=x[rng.integers(len(x),size=(cfg['bootstrap_draws'],len(x)))].mean(1)
    return {'estimate':float(x.mean()),'ci95':np.quantile(sample,[.025,.975]).tolist(),'n':len(x)}


def select(root):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];df=records(root,'validation')
    base=df.loc[df.name.eq('baseline')].set_index('sample_id');rows=[]
    for c in plan['candidates']:
        p=df.loc[df.name.eq(c['name'])].set_index('sample_id').loc[base.index]
        assert len(p)==len(base)==cfg['validation_questions']
        rows.append({'condition':c,'split':'validation','n':len(p),
            'net_correct':int((p.correct-base.correct).sum()),
            'wrong_to_correct':int(((base.correct==0)&(p.correct==1)).sum()),
            'correct_to_wrong':int(((base.correct==1)&(p.correct==0)).sum()),
            'parse_failure_increase':int((p.parse_failed-base.parse_failed).sum()),
            'mean_actual_energy':float(p.energy.mean())})
    selected=choose_candidate(rows,cfg)
    if time.time()>plan['validation_decision_unix']:
        selected=None;reason='22-hour validation decision cutoff reached'
    else:reason='positive_validation_candidate' if selected else 'no_candidate_with_positive_net_gain_and_format_gate'
    # Freeze n using timing only, before opening any test condition.
    smoke=records(root,'smoke');seconds=max(1.,float(smoke.seconds.quantile(.9)))
    remaining=plan['deadline_unix']-time.time()-cfg['reserve_hours']*3600
    n=cfg['test_questions']
    if seconds*7*n>remaining:n=cfg['fallback_test_questions']
    run=selected is not None and seconds*7*n<=remaining
    if selected and not run:reason='insufficient_budget_after_preserving_audit_reserve'
    result={'complete':True,'selected':selected,'run_test':run,'reason':reason,'validation':rows,
        'test_questions':n if run else 0,'frozen_before_any_test_generation':True,
        'p90_smoke_condition_seconds':seconds,'conservative_test_seconds':seconds*7*n if run else 0,
        'validation_audit_sha256':sha(root/'validation_audit.json'),'plan_sha256':sha(root/'plan.json')}
    assert not (root/'selection.json').exists()
    write_json(root/'selection.json',result);print(json.dumps(result),flush=True)


def summarize(root):
    cfg=json.loads((root/'plan.json').read_text())['config'];selection=json.loads((root/'selection.json').read_text())
    out=[];paired=[]
    for stage in ['validation']+(['test'] if selection['run_test'] else []):
        df=records(root,stage);df.to_parquet(root/f'{stage}_predictions.parquet',index=False)
        base=df.loc[df.name.eq('baseline')].set_index('sample_id').sort_index()
        parts=[(name,p) for name,p in df.groupby('name') if not name.startswith('random_')]
        random=df.loc[df.name.str.startswith('random_')]
        if len(random):
            assert random.groupby('sample_id').size().eq(3).all()
            parts.append(('random_seed_average',random.groupby('sample_id',as_index=False).mean(numeric_only=True)))
        aligned={}
        for name,p in parts:
            p=p.set_index('sample_id').loc[base.index];aligned[name]=p
            delta=p.correct-base.correct
            out.append({'stage':stage,'method':name,'n':len(p),'accuracy':interval(p.correct,cfg),
                'delta_accuracy':interval(delta,cfg),'wrong_to_correct_expected':float(((1-base.correct)*p.correct).sum()),
                'correct_to_wrong_expected':float((base.correct*(1-p.correct)).sum()),
                'parse_failure_rate':float(p.parse_failed.mean()),'truncation_rate':float(p.truncated.mean()),
                'mean_length':float(p.length.mean()),'intervention_coverage':float(p.modified.mean())})
        if stage=='test':
            chosen=selection['selected']['name']
            for other in ['shared8','random_seed_average']+(['current_local8'] if 'current_local8' in aligned else []):
                paired.append({'stage':stage,'method':chosen,'control':other,
                    'accuracy_difference':interval(aligned[chosen].correct-aligned[other].correct,cfg)})
    result={'complete':True,'selection':selection,'rows':out,'paired_controls':paired,
        'unit':'Question; random3seeds averaged within question; pointwise intervals, no multiple-comparison correction',
        'limitation':'Historical MATH; policy held out only for this selection. Free generation, not teacher-forced fidelity.',
        'source_audits':{stage:sha(root/f'{stage}_audit.json') for stage in ['validation']+(['test'] if selection['run_test'] else [])}}
    write_json(root/'steering_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('mode',choices=['select','summarize']);a=p.parse_args()
    select(a.root) if a.mode=='select' else summarize(a.root)

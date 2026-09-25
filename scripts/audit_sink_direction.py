"""Read-only independent raw-output checks after the finite GPU queue completes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from revision_common import sha,write_json
from sink_direction_common import boxed_answer


def run(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    done=json.loads((root/'COMPLETE.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert done['summary_sha256']==sha(root/'summary.json') and done['report_sha256']==sha(root/'report.md')
    assert done['plan_sha256']==summary['plan_sha256']==sha(root/'plan.json')
    for p,h in plan['files'].items():assert sha(p)==h,p
    execution=json.loads((root/'execution_plan.json').read_text())
    for p,h in execution['files'].items():assert sha(p)==h,p
    cases=json.loads((root/'cases.json').read_text());selection=json.loads((root/'selection.json').read_text())
    for p,h in selection['dev_record_hashes'].items():assert sha(root/p)==h,p
    for p,h in summary['records'].items():assert sha(root/p)==h,p
    assert len({r['question_group'] for r in cases})==len(cases)
    with np.load(root/'directions.npz') as z:directions={k:z[k].copy() for k in z.files}
    np.testing.assert_allclose(np.linalg.norm(directions['hss']),np.linalg.norm(directions['random']),rtol=1e-6)
    # Recompute the direction from original train means, using an independent join.
    frames=pd.concat([pd.read_parquet(p) for p in Path(cfg['foundation']).glob('prefixes/shard_*/rows.parquet')])
    train_ids=set(frames.loc[frames.split.eq('train'),'sample_id'])
    meta=pd.read_parquet(Path(cfg['mean_cache'])/'rows.parquet')
    train=meta.sample_id.isin(train_ids).to_numpy()
    x=np.load(Path(cfg['mean_cache'])/f'layer_{cfg["layer"]}.npy').astype(float)
    with np.load(Path(cfg['fit'])/'assignments.npz') as z:member=z['posterior']==plan['audit']['candidate_cluster']
    target=x[train&~member].mean(0)-x[train&member].mean(0)
    np.testing.assert_allclose(target,directions['hss'],atol=3e-7,rtol=1e-6)
    assert not {r['sample_id'] for r in cases}&train_ids
    records={};count=0;tokens=0
    for case in cases:
        expected=['zero']+[f'hss_{a:g}' for a in cfg['alphas']] if case['split']=='dev' else ['zero','hss','random']
        current=[]
        for condition in expected:
            p=root/'outputs'/case['sample_id']/(condition+'.json')
            r=json.loads(p.read_text());receipt=json.loads(p.with_suffix('.receipt.json').read_text())
            assert sha(p)==receipt['record_sha256'] and sha(p.with_suffix('.npz'))==r['geometry_sha256']
            assert r['plan_sha256']==sha(root/'plan.json')
            assert r['generation_config_sha256']==sha(root/'generation_config.json')
            assert r['execution_plan_sha256']==sha(root/'execution_plan.json')
            assert r['sample_id']==case['sample_id'] and r['split']==case['split'] and r['condition']==condition
            assert r['prompt_text']==case['prompt_text'] and r['ground_truth']==case['ground_truth']
            assert r['n_tokens']==len(r['generated_ids'])<=cfg['max_new_tokens']
            assert r['shifted_forwarded_tokens']==r['n_tokens']-1
            assert r['complete_boxed']==(boxed_answer(r['response_text']) is not None)
            vector=np.zeros_like(directions['hss']) if condition=='zero' else directions['random' if condition=='random' else 'hss']*r['alpha']
            np.testing.assert_allclose(np.linalg.norm(vector),r['ideal_shift_norm'],rtol=1e-6,atol=1e-8)
            with np.load(p.with_suffix('.npz')) as z:
                stats=z['step_geometry'];assert stats.shape==(r['n_tokens']-1,5) and np.isfinite(stats).all()
                if len(stats):
                    assert np.all(stats[:,1:3]>=0)
                    np.testing.assert_allclose(stats[:,1].mean(),r['mean_actual_shift_norm'],rtol=1e-6,atol=1e-8)
                    # Difference of means equals the average actual applied shift;
                    # its deviation from ideal is bounded by observed rounding.
                    discrepancy=np.linalg.norm(z['mean_after']-z['mean_before']-vector)
                    assert discrepancy<=stats[:,2].mean()+.002,(p,discrepancy)
                    if condition=='zero':assert np.all(stats[:,1:3]==0) and np.array_equal(z['mean_after'],z['mean_before'])
            current.append(r);records[(case['sample_id'],condition)]=r;count+=1;tokens+=r['n_tokens']
        assert len({r['generated_ids'][0] for r in current})==1
    assert count==plan['expected_generations']
    dev=[r for r in cases if r['split']=='dev']
    totals={a:sum(records[(r['sample_id'],f'hss_{a:g}')]['complete_boxed'] for r in dev) for a in cfg['alphas']}
    assert selection['alpha']==min(cfg['alphas'],key=lambda a:(-totals[a],a))
    checks=0
    for split in ['sink_test','normal_test']:
        subset=[r for r in cases if r['split']==split]
        for cond in ['zero','hss','random']:
            rows=[records[(r['sample_id'],cond)] for r in subset];reported=summary['results'][split][cond]
            assert reported['n']==len(rows)
            assert reported['boxed_count']==sum(r['complete_boxed'] for r in rows)
            assert reported['correct_count']==sum(r['correct'] for r in rows)
        for endpoint in ['complete_boxed','correct']:
            for control in ['zero','random']:
                d=np.array([int(records[(r['sample_id'],'hss')][endpoint])-int(records[(r['sample_id'],control)][endpoint]) for r in subset])
                stat=summary['paired'][split][endpoint+'_hss_minus_'+control]
                wins=int((d>0).sum());losses=int((d<0).sum())
                assert stat['wins']==wins and stat['losses']==losses
                p=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.
                np.testing.assert_allclose(stat['one_sided_exact_p'],p,atol=1e-14)
                rng=np.random.default_rng(cfg['seed']);draw=d[rng.integers(len(d),size=(cfg['bootstrap'],len(d)))].mean(1)
                np.testing.assert_allclose(stat['ci95'],np.quantile(draw,[.025,.975]),atol=1e-14)
                assert stat['difference']==float(d.mean());checks+=1
    primary=summary['paired']['sink_test']['complete_boxed_hss_minus_random']
    zero=summary['paired']['sink_test']['complete_boxed_hss_minus_zero']
    assert summary['predeclared_success']==(primary['ci95'][0]>0 and primary['one_sided_exact_p']<.05 and zero['difference']>0)
    receipt={'complete':True,'records':count,'generated_tokens':tokens,'paired_checks':checks,
        'summary_sha256':sha(root/'summary.json'),'plan_sha256':sha(root/'plan.json'),
        'source_sha256':sha(__file__),'semantic_review_complete':False,
        'note':'Automatic outputs/provenance/arithmetic checked; changed-answer semantic inspection remains separate'}
    write_json(root/'audit_SUCCESS.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

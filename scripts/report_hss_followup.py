"""CPU-only, complete-cohort summaries and the frozen D6 decision. No fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from hss_followup_tools import PARAMS, paired_binary_summary, step2_decision, decision_table_markdown
from revision_common import sha, write_json
from hss_followup_common import verify_plan, verify_record


def step1(root,plan):
    if not (root/'step1_COMPLETE.json').exists():return
    entries=[json.loads(p.read_text()) for p in sorted((root/'step1').glob('*/*.json')) if not p.name.endswith('.receipt.json')]
    cases=json.loads((root/'test_cases.json').read_text());L=PARAMS['hook_layer'];late=plan['L_late'];sink=plan['sink_global']
    vectors=dict(np.load(root/'vectors_initial.npz'));by={(r['sample_id'],r['arm']):r for r in entries}
    groups={}
    for group in ['sink_test','normal_test']:
        ids=[r['sample_id'] for r in cases if r['split']==group]
        g={}
        for arm in ['historical','zero','hss','random']:
            rr=[by[s,arm] for s in ids]
            g[arm]=dict(n=len(rr),in_sink=sum(r['global_ids'][L]==sink for r in rr),
                correct=sum(r['correct'] for r in rr),boxed=sum(r['boxed'] for r in rr),
                unboxed=sum(not r['boxed'] for r in rr),forced_correct=sum(bool(r['forced'] and r['forced']['correct']) for r in rr),
                original_or_forced_correct=sum(r['original_or_forced_correct'] for r in rr))
        q0=[s for s in ids if by[s,'zero']['global_ids'][L]==sink]
        g['Q0_n']=len(q0)
        if q0:g['exit_hss_vs_random']=paired_binary_summary(
            [by[s,'hss']['global_ids'][L]!=sink for s in q0],[by[s,'random']['global_ids'][L]!=sink for s in q0])
        g['text_reproduction_rate']=float(np.mean([by[s,'zero']['historical_text_exact'] for s in ids]))
        g['reproduction_warning']=group=='normal_test' and g['text_reproduction_rate']<.9
        shifts={}
        for ll,d in [(L,vectors['d_run']),(late,vectors['d_late'])]:
            direction=d/np.linalg.norm(d);a=[];b=[]
            for sid in ids:
                means={arm:np.load(root/'step1'/sid/(arm+'.npz'))['means'][ll] for arm in ['zero','hss','random']}
                a.append(float((means['hss']-means['zero'])@direction));b.append(float((means['random']-means['zero'])@direction))
            diff=np.asarray(a)-b
            shifts[str(ll)]=dict(mean_hss=float(np.mean(a)),mean_random=float(np.mean(b)),
                wilcoxon_two_sided_p=float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.)
        g['direction_shift']=shifts;groups[group]=g
    destinations=[]
    for case in cases:
        sid=case['sample_id']
        for arm in ['hss','random']:
            r=by[sid,arm]
            destinations.append(dict(sample_id=sid,split=case['split'],arm=arm,
                zero_in_sink=by[sid,'zero']['global_ids'][L]==sink,state=r['global_ids'][L],type=r['type'],
                dominant_type=r['dominant_types'][L],type_matches=r['type']==r['dominant_types'][L]))
    pd.DataFrame(destinations).to_parquet(root/'step1_destinations.parquet',index=False)
    write_json(root/'step1_summary.json',dict(groups=groups,sink_global=sink,L_late=late,
        limitation='Descriptive re-encoding of the single-random-direction historical test cohort; forced suffixes are unhooked.'))


def step2(root,plan):
    marker=root/'step2_COMPLETE.json'
    if not marker.exists():return
    assert sha(root/'step2_screen_long.parquet')==json.loads(marker.read_text())['parquet_sha256']
    rows=[]
    for p in sorted((root/'step2').glob('*/*.json')):
        if p.name.endswith('.receipt.json'):continue
        verify_record(p,root);rows.append(json.loads(p.read_text()))
    assert len(rows)==plan['expected_step2']
    frame=pd.read_parquet(root/'step2_screen_long.parquet')
    raw=pd.DataFrame(rows)
    # Independently recompute the scores table from individually hashed rows.
    keys=['question_id','condition']
    for metric in ['m_ans','m_loop','m_col_ans','m_col_nll']:
        np.testing.assert_allclose(frame.sort_values(keys)[metric].to_numpy(dtype=float),
            raw.sort_values(keys)[metric].to_numpy(dtype=float),equal_nan=True)
    old=Path(plan['config']['previous']);cases=json.loads((root/'test_cases.json').read_text())
    counts={arm:0 for arm in ['hss','random']}
    for case in cases:
        if case['split']!='sink_test':continue
        for arm in counts:
            record=json.loads((old/'outputs'/case['sample_id']/(arm+'.json')).read_text())
            counts[arm]+=int(record['correct'] and record['complete_boxed'])
    decision=step2_decision(frame,c0_generation_beats_random=counts['hss']>counts['random'])
    write_json(root/'step2_decision.json',decision)
    (root/'step2_decision.md').write_text(decision_table_markdown(decision)+'\n')
    vectors=json.loads((root/'step2_vector_diagnostics.json').read_text())
    prep=json.loads((root/'preparation.json').read_text())
    geometries=[]
    for r in rows:
        geometries.append(dict(condition=r['condition'],set=r['set'],**r['geometry']))
    geo=pd.DataFrame(geometries)
    pooled=[]
    for (qset,condition),g in geo.groupby(['set','condition']):
        pooled.append(dict(set=qset,condition=condition,n_tokens=int(g.n.sum()),
            actual_changed_fraction=float(g.changed.sum()/g.n.sum()),
            mean_actual_delta_norm=float((g.mean_actual_delta_norm*g.n).sum()/g.n.sum())))
    diagnostics=json.loads((root/'step2_token_diagnostics.json').read_text())
    coverage={}
    for qset in ['sink','normal']:
        sub=[d for d in diagnostics if d['set']==qset];n=sum(d['n_tokens'] for d in sub)
        coverage[qset]=dict(n_tokens=n,coverage_exceed_rate=sum(d['coverage_exceed_count'] for d in sub)/n,
            mean_chart_projection_ratio=sum(d['chart_ratio_sum'] for d in sub)/n,
            ideal_clamped_fraction=sum(d['c1_ideal_clamped_fraction']*d['n_tokens'] for d in sub)/n)
    write_json(root/'step2_diagnostics.json',dict(vectors=vectors,preparation=prep,
        geometry=pooled,coverage=coverage,original_correct_and_boxed=counts,
        caveat='C0 A-half direction differs from the original full-train direction; failure of this conservative gate does not prove miscalibration.'))
    wide=frame[frame['set']=='sink'].pivot(index='question_id',columns='condition',values='m_ans')
    delta=wide.sub(wide['NONE'],axis=0)
    delta.mean().rename('mean_delta_answer_logprob').to_csv(root/'step2_all_conditions.csv')
    types=frame[frame.condition=='C2'].set_index('question_id')['type']
    pd.DataFrame({'type':types.reindex(delta.index),'delta_ans':delta.C2}).groupby('type').agg(['size','mean']).to_csv(root/'step2_C2_by_type.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,4),sharey=True)
    for ax,C in zip(axes,['C0','C1','C2','C3']):
        fam=decision['per_candidate'][C]['null_family']; names=[s for s in delta if s.startswith(fam)]
        ax.scatter(np.arange(len(names)),delta[names].mean(),color='#8899aa',label='10 controls')
        ax.axhline(float(delta[C].mean()),color='#c43b30',label=C)
        ax.axhline(0,color='black',lw=.5);ax.set_title(C+' vs '+fam);ax.set_xlabel('Frozen control index')
    axes[0].set_ylabel('Mean change in gold-answer log probability (nats)')
    fig.tight_layout();fig.savefig(root/'step2_screen.png',dpi=180);plt.close(fig)
    print(decision_table_markdown(decision),flush=True)


def step3(root,plan):
    if not (root/'step3A_COMPLETE.json').exists():return
    cases=json.loads((root/'test_cases.json').read_text());decision=json.loads((root/'step2_decision.json').read_text())
    arms=['zero',decision['chosen'],decision['matched_control_for_3A']];old=Path(plan['config']['previous'])
    groups={}
    for split in ['sink_test','normal_test']:
        ids=[r['sample_id'] for r in cases if r['split']==split];rows={}
        for arm in arms:
            rows[arm]=[]
            for sid in ids:
                path=root/'step3A'/sid/(arm+'.json')
                if not path.exists() and arm=='zero':path=old/'outputs'/sid/'zero.json'
                rows[arm].append(json.loads(path.read_text()))
        C=arms[1];R=arms[2]
        g={arm:dict(n=len(rr),correct=sum(r['correct'] for r in rr),boxed=sum(r['complete_boxed'] for r in rr),
            correct_and_boxed=sum(r['correct'] and r['complete_boxed'] for r in rr),
            mean_tokens=float(np.mean([r['n_tokens'] for r in rr])),truncated=sum(r['finish_reason']=='length' for r in rr)) for arm,rr in rows.items()}
        g['primary_candidate_vs_control']=paired_binary_summary([r['correct'] and r['complete_boxed'] for r in rows[C]],
            [r['correct'] and r['complete_boxed'] for r in rows[R]])
        g['candidate_vs_zero']=paired_binary_summary([r['correct'] and r['complete_boxed'] for r in rows[C]],
            [r['correct'] and r['complete_boxed'] for r in rows['zero']])
        groups[split]=g
    p=groups['sink_test']['primary_candidate_vs_control']
    write_json(root/'step3A_summary.json',dict(groups=groups,
        positive_paper_gate=p['a_success']>p['b_success'] and p['mcnemar_p']<.05,
        chosen=arms[1],control=arms[2],exploratory=True))


def run(root):
    root=Path(root);plan=verify_plan(root);step1(root,plan);step2(root,plan);step3(root,plan)
    parts=['# Sink follow-up — exploratory results','',
        'Original automatic scoring remains frozen. No claim of confirmatory evidence on a fresh test set.','']
    for name in ['step1_summary.json','step2_decision.md','step3A_summary.json','step3_status.json']:
        p=root/name
        if p.exists():parts+=['## '+name,'',p.read_text(),'']
    (root/'report.md').write_text('\n'.join(parts))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)

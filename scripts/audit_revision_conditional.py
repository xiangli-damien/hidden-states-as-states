"""Audit saved conditional-prediction results and render all predefined views."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from revision_statistics import paired_auc_ci


def run(root):
    cfg=json.loads((root/'plan.json').read_text())['config']
    expected={f'p{p}_l{l}_{v}' for p,l,v in cfg['jobs']}
    assert {p.parent.name for p in root.glob('p*/_SUCCESS.json')}==expected
    summaries=[];trials=0;converged=0;boundary=[];common_test=None
    for name in sorted(expected):
        directory=root/name
        marker=json.loads((directory/'_SUCCESS.json').read_text())
        for key,filename in [('summary_sha256','summary.json'),('predictions_sha256','predictions.parquet')]:
            assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==marker[key]
        result=json.loads((directory/'summary.json').read_text())
        frame=pd.read_parquet(directory/'predictions.parquet')
        assert not frame.sample_id.duplicated().any() and not frame.question_group.duplicated().any()
        assert set(frame.split)=={'train','validation','test'}
        test=frame.loc[frame.split.eq('test')].set_index('sample_id').sort_index()
        if common_test is not None:
            pd.testing.assert_series_equal(common_test, test.failure)
        common_test=test.failure
        for metric in result['metrics']:
            method=metric['method'];selection=json.loads((directory/method/'selection.json').read_text())
            eligible=[x for x in selection['trials'] if x['converged']]
            best=min(eligible,key=lambda x:x['validation_log_loss'])
            assert best==selection['selected'] and best['C']==metric['selected_C']
            assert sorted(x['C'] for x in selection['trials'])==sorted(cfg['C_grid'])
            trials+=len(selection['trials']);converged+=len(eligible)
            if best['C'] in (min(cfg['C_grid']),max(cfg['C_grid'])):
                boundary.append({'view':name,'method':method,'C':best['C']})
            validation=frame.loc[frame.split.eq('validation')]
            # Saved scores may be float32; sklearn/BLAS reduction ordering can
            # differ across hosts at float32 precision for loss metrics.
            np.testing.assert_allclose(log_loss(validation.failure,validation[method]),best['validation_log_loss'],rtol=0,atol=2e-7)
            for key,fun in [('test_auroc',roc_auc_score),('test_log_loss',log_loss),('test_brier',brier_score_loss)]:
                np.testing.assert_allclose(fun(test.failure,test[method]),metric[key],rtol=0,atol=1e-12 if key=='test_auroc' else 2e-7)
        for comparison in result['paired_comparisons']:
            # Preserve stored row ordering so the seeded bootstrap draws match.
            t=frame.loc[frame.split.eq('test')]
            recomputed=paired_auc_ci(t.failure,t[comparison['method']],t[comparison['baseline']])
            for key in ('auroc','baseline_auroc','delta','ci95'):
                np.testing.assert_allclose(recomputed[key],comparison[key],atol=1e-12)
        summaries.append(result)
    notes=[
        'Qwen2-7B-Instruct, raw pre-final-norm block outputs. Same 986 historical MATH test questions; exploratory.',
        'Prefix 16/64 means that many generated tokens have been consumed; no future response length is used.',
        'Train fits scalers and LR; validation log loss selects C. All nine C candidates and selected predictions are saved.',
        'No detected linear-readout increment does not establish conditional independence or absence of computational information.',
        'The state one-hot block shares a single regularization strength with thousands of scaled continuous coordinates. Its tiny increment is not an information upper bound.',
        'Some selections hit the smallest tested C. The finite readout family is a limitation; no test-guided grid expansion is included.',
        'TF-IDF is a question-only lexical baseline, not a semantic encoder. Its controls also include current-prefix confidence.',
        'Intervals are paired question bootstrap, 2,000 draws, pointwise and not multiplicity-adjusted.',
        'Geometric clustering remains unnormalized; train-only feature scaling here belongs to supervised prediction.'
    ]
    audit={'views':len(summaries),'readouts':sum(len(s['metrics']) for s in summaries),
           'candidate_fits':trials,'converged':converged,'test_questions':len(common_test),
           'checksum_metric_selection_and_CI_checks_passed':True,'C_boundary_selections':boundary,'notes':notes}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    dest=root/'report';dest.mkdir(exist_ok=True)
    labels=[];conditional=[];state=[];table=[]
    for s in summaries:
        label=f"B{s['block']} / {s['prefix']} tokens / {s['view']}"
        metrics={x['method']:x for x in s['metrics']}
        a=next(x for x in s['paired_comparisons'] if x['method']=='prompt_plus_current_plus_controls' and x['baseline']=='prompt_plus_controls')
        b=next(x for x in s['paired_comparisons'] if x['method']=='prompt_plus_state_plus_controls' and x['baseline']=='prompt_plus_controls')
        labels.append(label);conditional.append(a);state.append(b)
        table.append({'view':label,'prompt only':metrics['prompt_only']['test_auroc'],
            'current only':metrics['current_only']['test_auroc'],'prompt + controls':a['baseline_auroc'],
            '+ current vector':a['auroc'],'delta':a['delta'],'CI95':a['ci95']})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
    for axis,values,title in [(axes[0],conditional,'Add current continuous vector'),(axes[1],state,'Add current discrete state')]:
        means=np.array([x['delta'] for x in values]);low=np.array([x['ci95'][0] for x in values]);high=np.array([x['ci95'][1] for x in values])
        axis.errorbar(means,np.arange(len(labels)),xerr=[means-low,high-means],fmt='o',capsize=4,color='#007f86')
        axis.axvline(0,color='.4',ls='--');axis.set_yticks(range(len(labels)),labels)
        axis.set_xlabel('AUROC gain over prompt + current controls');axis.set_title(title,fontsize=11)
        axis.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Qwen2 MATH: generation-state increment beyond initial prompt representation',fontsize=12)
    fig.savefig(dest/'conditional_increment.png',dpi=180);fig.savefig(dest/'conditional_increment.pdf');plt.close(fig)
    parts=['<!doctype html><html><meta charset="utf-8"><title>Conditional information audit</title>',
        '<style>body{font:16px system-ui;margin:35px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}img{max-width:100%}</style>',
        '<h1>生成窗口是否超过最初的 prompt 表示？</h1>',
        '<p>当前读出实验没有检测到稳定正增益；这不证明生成表示没有新信息，也不是因果结论。</p>',
        '<img src="conditional_increment.png" alt="Paired conditional increments">',
        pd.DataFrame(table).to_html(index=False,float_format=lambda x:f'{x:.6f}'),
        '<h2>审计与边界</h2><ul>']
    parts.extend('<li>'+html.escape(x)+'</li>' for x in notes)
    parts.extend(['</ul>',f'<p>Audited: {len(summaries)} views, {audit["readouts"]} readouts, {trials} candidates, {converged} converged. Boundary selections: {len(boundary)}.</p>',
                  '<h2>全部结果</h2>'])
    for s in summaries:
        parts.extend([f'<h3>B{s["block"]} / prefix {s["prefix"]} / {s["view"]}</h3>',pd.DataFrame(s['metrics']).to_html(index=False)])
    parts.append('</html>');(dest/'index.html').write_text('\n'.join(parts))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('notes','C_boundary_selections')},indent=2))
    print('Boundary selections:',len(boundary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)

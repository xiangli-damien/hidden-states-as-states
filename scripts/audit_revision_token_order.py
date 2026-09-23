"""Audit every frozen token-order readout before rendering the full comparison."""
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
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss

from revision_common import write_json
from revision_statistics import paired_auc_ci


def run(root):
    cfg=json.loads((root/'plan.json').read_text())['config']
    expected={f'p{p}_l{l}_{v}' for p,l,v in cfg['views']}
    if {p.parent.name for p in root.glob('p*/_SUCCESS.json')}!=expected:
        raise ValueError('All predefined views must finish before this complete-stage audit')
    results=[];nfit=0;nconverged=0;boundary=[]
    for name in sorted(expected):
        d=root/name;s=json.loads((d/'summary.json').read_text());m=json.loads((d/'_SUCCESS.json').read_text())
        for filename,key in [('summary.json','summary_sha256'),('predictions.parquet','predictions_sha256')]:
            assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==m[key]
        frame=pd.read_parquet(d/'predictions.parquet')
        assert not frame.sample_id.duplicated().any() and not frame.question_group.duplicated().any()
        assert set(frame.split)=={'train','validation','test'}
        test=frame.loc[frame.split.eq('test')];val=frame.loc[frame.split.eq('validation')]
        for metric in s['metrics']:
            method=metric['method'];selection=json.loads((d/method/'selection.json').read_text())
            candidates=selection['trials'];eligible=[x for x in candidates if x['converged']]
            best=min(eligible,key=lambda x:x['validation_log_loss'])
            assert best==selection['selected'] and best['C']==metric['selected_C']
            assert sorted(x['C'] for x in candidates)==sorted(cfg['C_grid'])
            nfit+=len(candidates);nconverged+=len(eligible)
            if best['C'] in (min(cfg['C_grid']),max(cfg['C_grid'])):boundary.append({'view':name,'method':method,'C':best['C']})
            np.testing.assert_allclose(log_loss(val.failure,val[method]),best['validation_log_loss'],rtol=0,atol=2e-7)
            for key,fun in [('test_auroc',roc_auc_score),('test_log_loss',log_loss),('test_brier',brier_score_loss)]:
                np.testing.assert_allclose(fun(test.failure,test[method]),metric[key],rtol=0,atol=1e-12 if key=='test_auroc' else 2e-7)
        for c in s['paired_comparisons']:
            check=paired_auc_ci(test.failure,test[c['method']],test[c['baseline']])
            for key in ('auroc','baseline_auroc','delta','ci95'):np.testing.assert_allclose(check[key],c[key],rtol=0,atol=1e-12)
        results.append(s)
    audit={'views':len(results),'readouts':sum(len(s['metrics']) for s in results),
        'candidate_fits':nfit,'converged':nconverged,'boundary_selections':boundary,
        'predictions_selection_metrics_and_CI_valid':True,'primary':cfg['primary']}
    write_json(root/'audit.json',audit)
    table=[];comparisons=[]
    for s in results:
        row={'view':s['view'],'test_questions':s['split_counts']['test']}
        row.update({x['method']:x['test_auroc'] for x in s['metrics']});table.append(row)
        comparisons.extend([{'view':s['view'],**c} for c in s['paired_comparisons']])
    dest=root/'report';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained',sharey=True)
    pairs=[('state_ordered','state_occupancy','Order versus occupancy'),
           ('state_ordered','state_ordered_shuffled_42','Order versus questionwise shuffle (42)'),
           ('prompt_controls_plus_ordered','prompt_controls','Order added beyond prompt + controls')]
    for ax,(method,base,title) in zip(axes,pairs):
        rows=[next(c for c in s['paired_comparisons'] if c['method']==method and c['baseline']==base) for s in results]
        means=np.array([r['delta'] for r in rows]);low=np.array([r['ci95'][0] for r in rows]);high=np.array([r['ci95'][1] for r in rows])
        ax.errorbar(means,np.arange(len(rows)),xerr=[means-low,high-means],fmt='o',capsize=4,color='#007f86')
        ax.axvline(0,color='.4',ls='--');ax.set_title(title,fontsize=10);ax.set_xlabel('Paired AUROC difference')
        ax.set_yticks(np.arange(len(rows)),[s['view'].replace('question_tokens','question').replace('tokens','chat' if s['prefix']==0 else 'generated') for s in results]);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.suptitle('Qwen2 MATH: order of 16-token GMM states; historical test, exploratory',fontsize=12)
    fig.savefig(dest/'token_order.png',dpi=180);fig.savefig(dest/'token_order.pdf');plt.close(fig)
    notes=[
        'All six views and all 16 readout families reported; no test-guided choice of the best layer/role/seed.',
        'Main comparison was fixed to block28/generated16 ordered versus occupancy; shuffle42 is the primary order control, 137/271 are sensitivity checks.',
        'Raw pre-final-norm activations, frozen train-only K64 GMM, nearest assignments. No cluster refit in this stage.',
        'Mean/last/occupancy/order readouts have different feature dimensions. Predictive gains are not equal-capacity or compression-rate superiority claims.',
        'Ordered LR is additive in position. A transition histogram uses adjacent pair features; neither is a general nonlinear sequence model.',
        'Within-question shuffles preserve exact state counts and are used in both train and evaluation. Position marginals can change; this is not a causal intervention.',
        'Controls include category, difficulty, prompt length, representation norms and current entropy/margin, never future answer length.',
        'All scalers and vocabularies are fit on train; C chosen by validation log loss. Unknown token IDs have an explicit reserved feature.',
        'Conditional readouts share a single C after feature scaling; no information-theoretic upper bound is inferred from a null gain.',
        'Question-tail views have fewer valid questions than chat/generation; only within-view paired effects are evaluated.',
        'Intervals use 2,000 question bootstrap draws, pointwise without multiplicity correction; reused historical test is exploratory.'
    ]
    parts=['<!doctype html><html><meta charset="utf-8"><title>Token state order</title>',
        '<style>body{font:16px system-ui;margin:35px;max-width:1900px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}.wide{overflow:auto}img{max-width:100%}</style>',
        '<h1>同一16-token窗口：状态频率、顺序、均值与输入词身份</h1>',
        '<img src="token_order.png" alt="Paired order comparison">','<ul>']
    parts.extend('<li>'+html.escape(n)+'</li>' for n in notes)
    parts.extend(['</ul>',f'<p>Audited {audit["readouts"]} readouts; {nconverged}/{nfit} candidates converged; {len(boundary)} selected C at grid boundary.</p>',
        '<h2>全部 AUROC</h2><div class="wide">',pd.DataFrame(table).to_html(index=False,float_format=lambda x:f'{x:.6f}'),'</div>',
        '<h2>问题级配对比较</h2>',pd.DataFrame(comparisons).to_html(index=False,float_format=lambda x:f'{x:.6f}'),'</html>'])
    (dest/'index.html').write_text('\n'.join(parts));print(json.dumps({k:v for k,v in audit.items() if k!='boundary_selections'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)

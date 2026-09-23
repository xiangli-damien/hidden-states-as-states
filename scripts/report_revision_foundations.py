"""Render only observed results; partially finished stages remain visibly partial."""
import argparse
import html
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from revision_common import config, write_json, paired_ratio_ci


def run(cfg):
    root=Path(cfg['output']);dest=root/'report';dest.mkdir(parents=True,exist_ok=True)
    summaries=[json.loads(p.read_text()) for p in sorted((root/'geometry').glob('*/summary.json'))]
    rows=[]
    for s in summaries:
        for g in s['geometry']:
            rows.append({'view':s['view'],'prefix':s['prefix'],'block':s['layer'],'K':s['selected']['k'],
                         'K_at_upper_boundary':s['k_grid_boundary'],'method':g['method'].replace('local_pca_','local_residual_svd_'),
                         'test_questions':s['test_questions'],'NMSE':g['test_nmse']['estimate'],
                         'CI95':str(g['test_nmse']['ci95'])})
    tables={'geometry':pd.DataFrame(rows)}
    pred=[]
    for p in sorted((root/'geometry').glob('*/prediction_per_question.parquet')):
        data=pd.read_parquet(p);test=data.loc[data.split.eq('test')]
        delta=[];rng=np.random.default_rng(42)
        y=test.failure.to_numpy();a=test.nuisance_plus_state.to_numpy();b=test.nuisance.to_numpy()
        for _ in range(1000):
            ix=rng.integers(len(y),size=len(y))
            if len(np.unique(y[ix]))==2:
                delta.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
        for method in ('state_nb','linear_probe','nuisance','nuisance_plus_state'):
            pred.append({'view':p.parent.name,'method':method,'AUROC':roc_auc_score(y,test[method]),
                         'test_questions':len(test),'increment_over_nuisance_CI':str(np.quantile(delta,[.025,.975]).tolist()) if method=='nuisance_plus_state' else ''})
    tables['prediction']=pd.DataFrame(pred)
    matched=[]
    for view in ('last','mean4','mean16','all_mean'):
        for layer in cfg['layers']:
            frames={}
            for prefix in cfg['prefixes']:
                path=root/'geometry'/f'p{prefix}_l{layer}_{view}'/'prediction_per_question.parquet'
                if path.exists():
                    frame=pd.read_parquet(path);frames[prefix]=frame.loc[frame.split.eq('test')].set_index('sample_id')
            if len(frames)!=len(cfg['prefixes']):
                continue
            common=sorted(set.intersection(*(set(f.index) for f in frames.values())))
            for prefix,frame in frames.items():
                sub=frame.loc[common]
                for method in ('state_nb','linear_probe','nuisance','nuisance_plus_state'):
                    matched.append({'view':view,'block':layer,'prefix':prefix,'method':method,
                        'same_questions':len(common),'AUROC':float(roc_auc_score(sub.failure,sub[method]))})
    tables['prefix_comparison_same_questions']=pd.DataFrame(matched)
    for phase in ('functional','behavior'):
        records=[json.loads(p.read_text()) for p in (root/phase/'samples').glob('*.json')]
        if not records:
            continue
        frame=pd.DataFrame(records)
        keys=['sample_id','split','prefix_tokens','layer','role','width']
        baseline=frame.loc[frame.method.eq('identity')].set_index(keys)
        results=[]
        for group,sub in frame.groupby(['split','prefix_tokens','layer','role','width','method']):
            baseline_columns=['nll'] if phase=='functional' else ['correct','normalized_answer']
            aligned=sub.set_index(keys).join(baseline[baseline_columns],rsuffix='_baseline',how='inner')
            if not len(aligned):
                continue
            row=dict(zip(['split','prefix','block','role','width','method'],group));row['n']=len(aligned)
            if phase=='functional':
                delta=aligned.nll-aligned.nll_baseline
                row.update(mean_NLL=float(aligned.nll.mean()),mean_baseline_NLL=float(aligned.nll_baseline.mean()),
                           delta_NLL=paired_ratio_ci(delta,np.ones(len(delta))),mean_next_KL=float(aligned.next_token_kl.mean()),
                           mean_patch_energy=float(aligned.actual_patch_energy.mean()))
                zero=frame.loc[frame.method.eq('zero')].set_index(keys)
                paired=aligned.join(zero[['nll']].rename(columns={'nll':'nll_zero'}),how='inner')
                if len(paired):
                    den=(paired.nll_zero-paired.nll_baseline).to_numpy()
                    row.update(mean_zero_NLL=float(paired.nll_zero.mean()),
                               loss_recovered=None if den.mean()<=1e-8 else float((paired.nll_zero-paired.nll).sum()/den.sum()),
                               zero_not_worse_than_baseline_fraction=float((den<=0).mean()))
            else:
                before=aligned.correct_baseline.astype(bool);after=aligned.correct.astype(bool)
                delta=after.to_numpy(int)-before.to_numpy(int)
                row.update(accuracy=float(after.mean()),baseline_accuracy=float(before.mean()),
                           wrong_to_correct=int((~before&after).sum()),correct_to_wrong=int((before&~after).sum()),
                           net_accuracy=paired_ratio_ci(delta,np.ones(len(delta))),
                           answer_agreement=float((aligned.normalized_answer.fillna('')==aligned.normalized_answer_baseline.fillna('')).mean()),
                           parse_failed=float(aligned.parse_failed.mean()),truncated=float(aligned.finish_reason.eq('length').mean()))
            results.append(row)
        write_json(dest/f'{phase}_summary.json',results);tables[phase]=pd.DataFrame(results)
    notes=['Qwen2-7B MATH; raw pre-final-norm decoder block activations, blocks 7/14/21/28.',
           'MATH test was previously explored: all findings here are exploratory. No causal conclusion follows from AUROC or geometric NMSE alone.',
           'Prefix 0 is pre-generation; 16/64 consume only that many saved generated tokens. Full-answer mean is not substituted for an online state.',
           'Mean and token codebooks are distinct. Token fit uses 4 positions per training question; reconstruction tests all 16 positions.',
           'GMM primary assignment is nearest centroid. Implementation key local_pca means SVD of local train residuals about the FIXED GMM mean; it is not empirical-mean-centered local PCA, nor MFA. The exact empirical-mean PCA comparison remains pending.',
           'Across-prefix AUROC comparisons must use the common-question table; samples that terminate before a prefix are unavailable, not padded or repeated.',
           'Single-layer patch leaves other layers, positions and prompt context available. This does not establish full-model compression.',
           'All confidence intervals are pointwise question-level bootstrap. No multiple-comparison correction or confirmatory selection is implied.',
           'Selective steering, controlled counterfactuals and frozen-map GSM8K transfer evaluation are pending separate stages. Collection is not a transfer result.']
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>HSS revision foundations</title>',
           '<style>body{font:16px system-ui;margin:40px;max-width:1800px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}h2{margin-top:40px}.table{overflow:auto}</style>',
           '<h1>多 token / mean / reconstruction 前提检验</h1>',
           '<p>仅显示真实已写入的结果；阶段是否完成请看下面状态。结果不能预先保证为正。</p><ul>']
    parts.extend('<li>'+html.escape(n)+'</li>' for n in notes);parts.append('</ul>')
    for p in sorted(root.glob('*_status.json')):
        s=json.loads(p.read_text());parts.append('<h3>'+html.escape(p.stem)+'</h3><pre>'+html.escape(json.dumps(s,indent=2))+'</pre>')
    for name,frame in tables.items():
        parts.append('<h2>'+html.escape(name)+'</h2>')
        if len(frame):
            frame.to_parquet(dest/f'{name}_table.parquet',index=False)
            parts.append('<div class="table">'+frame.to_html(index=False,escape=True)+'</div>')
        else:
            parts.append('<p>尚无结果。</p>')
    parts.append('</html>');(dest/'index.html').write_text('\n'.join(parts))
    write_json(dest/'report_manifest.json',{'updated_unix':time.time(),'completed_geometry_views':len(summaries),
                                         'tables':{k:len(v) for k,v in tables.items()}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    run(config(p.parse_args().config))

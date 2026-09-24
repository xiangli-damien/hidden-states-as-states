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

from revision_common import write_json,paired_ratio_ci
from revision_statistics import paired_auc_ci
from revision_token_order import shuffle_questions


def code_identity_diagnostics(codes,token_ids,train,test,k):
    """Question-held-out readouts; descriptive token contingency, no token p-values."""
    t=codes.shape[1];position=np.broadcast_to(np.arange(t),codes.shape)
    counts=np.zeros((k,t),np.int64)
    np.add.at(counts,(codes[train].ravel(),position[train].ravel()),1)
    decoded=counts.argmax(1)[codes[test]]
    accuracy=(decoded==position[test]).mean(1)
    contingency=np.zeros((k,t),np.float64)
    np.add.at(contingency,(codes[test].ravel(),position[test].ravel()),1)
    joint=contingency/contingency.sum();state=joint.sum(1);slot=joint.sum(0)
    entropy=lambda p:float(-(p[p>0]*np.log2(p[p>0])).sum())
    information=entropy(state)+entropy(slot)-entropy(joint.ravel())
    vocabulary,inverse=np.unique(token_ids[train].ravel(),return_inverse=True)
    lexical=np.zeros((k,len(vocabulary)),np.int64)
    np.add.at(lexical,(codes[train].ravel(),inverse),1)
    fallback=int(np.bincount(inverse).argmax())
    token_of_state=lexical.argmax(1);token_of_state[lexical.sum(1)==0]=fallback
    token_accuracy=(vocabulary[token_of_state[codes[test]]]==token_ids[test]).mean(1)
    slot_tokens=np.array([np.unique(token_ids[train,j],return_counts=True)[0][np.unique(token_ids[train,j],return_counts=True)[1].argmax()] for j in range(t)])
    slot_accuracy=(token_ids[test]==slot_tokens).mean(1)
    return {'position_from_state_accuracy':paired_ratio_ci(accuracy,np.ones(len(accuracy))),
            'position_chance_accuracy':1/t,'state_entropy_bits':entropy(state),
            'position_entropy_bits':entropy(slot),'state_position_mutual_information_bits':information,
            'position_entropy_fraction_in_state':information/entropy(slot),
            'token_from_state_accuracy':paired_ratio_ci(token_accuracy,np.ones(len(token_accuracy))),
            'token_from_position_accuracy':paired_ratio_ci(slot_accuracy,np.ones(len(slot_accuracy))),
            'token_accuracy_state_minus_position':paired_ratio_ci(token_accuracy-slot_accuracy,np.ones(len(slot_accuracy))),
            'state_position_test_counts':contingency.astype(int).tolist(),
            'state_order_from_train':np.lexsort((-counts.sum(1),counts.argmax(1))).tolist(),
            'accuracy_bootstrap_draws':1000,'accuracy_bootstrap_seed':42,
            'scope':'Train majority decoder, held-out questions; entropy/MI are descriptive token contingencies, no independent-token significance claim'}


def run(root):
    cfg=json.loads((root/'plan.json').read_text())['config']
    expected={f'p{p}_l{l}_{v}' for p,l,v in cfg['views']}
    if {p.parent.name for p in root.glob('p*/_SUCCESS.json')}!=expected:
        raise ValueError('All predefined views must finish before this complete-stage audit')
    results=[];nfit=0;nconverged=0;boundary=[];identity=[]
    for name in sorted(expected):
        d=root/name;s=json.loads((d/'summary.json').read_text());m=json.loads((d/'_SUCCESS.json').read_text())
        for filename,key in [('summary.json','summary_sha256'),('predictions.parquet','predictions_sha256'),('question_codes.npz','question_codes_sha256')]:
            assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==m[key]
        frame=pd.read_parquet(d/'predictions.parquet')
        assert not frame.sample_id.duplicated().any() and not frame.question_group.duplicated().any()
        assert set(frame.split)=={'train','validation','test'}
        test=frame.loc[frame.split.eq('test')];val=frame.loc[frame.split.eq('validation')]
        with np.load(d/'question_codes.npz') as z:
            np.testing.assert_array_equal(frame.sample_id.to_numpy(str),z['sample_ids'])
            diag=code_identity_diagnostics(z['codes'],z['token_ids'],frame.split.eq('train').to_numpy(),frame.split.eq('test').to_numpy(),len(z['centers']))
            train=frame.split.eq('train').to_numpy();k=len(z['centers']);t=z['codes'].shape[1]
            def active_columns(codes):
                count=np.bincount((codes+np.arange(t)*k).ravel(),minlength=t*k)
                return int(((count>0)&(count<len(codes))).sum())
            diag['ordered_nonconstant_train_columns']=active_columns(z['codes'][train])
            shuffled=shuffle_questions(z['codes'],z['sample_ids'],42)
            diag['shuffle42_nonconstant_train_columns']=active_columns(shuffled[train])
        identity.append({'view':name,**diag})
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
    write_json(root/'code_identity_diagnostics.json',identity)
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
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for ax,d in zip(axes.ravel(),identity):
        counts=np.asarray(d['state_position_test_counts'],float)[d['state_order_from_train']]
        probability=np.divide(counts,counts.sum(1,keepdims=True),out=np.full_like(counts,np.nan),where=counts.sum(1,keepdims=True)>0)
        im=ax.imshow(probability,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        ax.set_title(d['view'],fontsize=10);ax.set_xticks([0,3,7,11,15],[1,4,8,12,16]);ax.set_xlabel('Window position')
        ax.set_ylabel('State (sorted using train only)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Held-out P(position | state)',shrink=.85)
    fig.suptitle('Do state IDs already encode window position? Gray/blank rows have no test tokens',fontsize=12)
    fig.savefig(dest/'state_position.png',dpi=180);fig.savefig(dest/'state_position.pdf');plt.close(fig)
    notes=[
        'All six views and all 16 readout families reported; no test-guided choice of the best layer/role/seed.',
        'Main comparison was fixed to block28/generated16 ordered versus occupancy; shuffle42 is the primary order control, 137/271 are sensitivity checks.',
        'Raw pre-final-norm activations, frozen train-only K64 GMM, nearest assignments. No cluster refit in this stage.',
        'Mean/last/occupancy/order readouts have different feature dimensions. Predictive gains are not equal-capacity or compression-rate superiority claims.',
        'Ordered LR is additive in position. A transition histogram uses adjacent pair features; neither is a general nonlinear sequence model.',
        'Within-question shuffles preserve exact state counts and are used in both train and evaluation. Position marginals can change; this is not a causal intervention.',
        'A state ID may itself reveal its position. In that case occupancy implicitly retains some order; a null order gain does not show order is unimportant. The separate position-decoding diagnostic measures this.',
        'Shuffling can greatly increase the number of nonconstant position-state features, even with the same nominal dimension. Reduced shuffled-readout performance alone is not evidence of lost task information.',
        'Controls include category, difficulty, prompt length, representation norms and current entropy/margin, never future answer length.',
        'All scalers and vocabularies are fit on train; C chosen by validation log loss. Unknown token IDs have an explicit reserved feature.',
        'Conditional readouts share a single C after feature scaling; no information-theoretic upper bound is inferred from a null gain.',
        'Question-tail views have fewer valid questions than chat/generation; only within-view paired effects are evaluated.',
        'AUROC intervals use 2,000 question bootstrap draws; identity-decoding accuracy intervals use 1,000. All are pointwise without multiplicity correction; reused historical test is exploratory.'
    ]
    parts=['<!doctype html><html><meta charset="utf-8"><title>Token state order</title>',
        '<style>body{font:16px system-ui;margin:35px;max-width:1900px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}.wide{overflow:auto}img{max-width:100%}</style>',
        '<h1>同一16-token窗口：状态频率、顺序、均值与输入词身份</h1>',
        '<img src="token_order.png" alt="Paired order comparison">','<ul>']
    parts.extend('<li>'+html.escape(n)+'</li>' for n in notes)
    parts.extend(['</ul>',f'<p>Audited {audit["readouts"]} readouts; {nconverged}/{nfit} candidates converged; {len(boundary)} selected C at grid boundary.</p>',
        '<h2>全部 AUROC</h2><div class="wide">',pd.DataFrame(table).to_html(index=False,float_format=lambda x:f'{x:.6f}'),'</div>',
        '<h2>簇编号本身透露多少位置／词身份？</h2><img src="state_position.png" alt="State position contingency">',pd.DataFrame([{
            'view':d['view'],'position_accuracy':d['position_from_state_accuracy']['estimate'],
            'position_chance':d['position_chance_accuracy'],'position_entropy_fraction_in_state':d['position_entropy_fraction_in_state'],
            'ordered_active_columns':d['ordered_nonconstant_train_columns'],'shuffled_active_columns':d['shuffle42_nonconstant_train_columns'],
            'token_from_state_accuracy':d['token_from_state_accuracy']['estimate'],
            'token_from_position_accuracy':d['token_from_position_accuracy']['estimate']} for d in identity]).to_html(index=False,float_format=lambda x:f'{x:.6f}'),
        '<h2>问题级配对比较</h2>',pd.DataFrame(comparisons).to_html(index=False,float_format=lambda x:f'{x:.6f}'),'</html>'])
    (dest/'index.html').write_text('\n'.join(parts));print(json.dumps({k:v for k,v in audit.items() if k!='boundary_selections'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)

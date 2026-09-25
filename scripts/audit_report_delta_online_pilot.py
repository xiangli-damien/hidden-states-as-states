"""Independent arithmetic, prefix availability and reporting checks."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from revision_common import config,sha,write_json,write_npz
from run_delta_online_pilot import verify
from delta_online_common import replacement,FUNCTIONAL_METHODS,prefix_controls


def audit_functional(cfg,smoke=False):
    import torch
    r=Path(cfg['output']);plan=verify(r);out=r/('smoke' if smoke else 'functional')
    receipt=json.loads((out/'_SUCCESS.json').read_text());assert receipt['plan_sha256']==sha(r/'plan.json')
    tasks=json.loads((r/'functional_tasks.json').read_text())['tasks']
    if smoke:tasks=[t for t in tasks if t['sample_id'] in plan['smoke_ids'] and t['prefix']==16 and t['width']==4]
    assert receipt['records']==len(tasks)
    d={};baseline={};records=[];maximum=0
    for rep in ['delta','state']:
        with np.load(r/(rep+'_decoder.npz')) as z:d[rep]={k:z[k].copy() for k in z.files}
    tokens=json.loads((r/'tokens.json').read_text())
    for task in tasks:
        path=out/(task['key']+'.json');a=json.loads(path.read_text());assert a['task']==task and a['plan_sha256']==sha(r/'plan.json')
        assert sha(path.with_suffix('.npz'))==a['arrays_sha256'];key=(task['sample_id'],task['prefix'],task['width'])
        with np.load(path.with_suffix('.npz')) as z:
            ideal=replacement(z['before'],z['after'],d['delta'],d['state'],task['method'])
            np.testing.assert_allclose(ideal,z['ideal'],rtol=1e-12,atol=1e-12)
            actual=torch.tensor(ideal).bfloat16().float().numpy();np.testing.assert_array_equal(actual,z['actual'])
            lp=z['logp'].astype(float);assert np.isfinite(lp).all();np.testing.assert_allclose(np.exp(lp).sum(),1,atol=2e-6)
            if task['method']=='identity':baseline[key]=(lp,z['before'].copy(),z['after'].copy(),z['reference_nll'].copy())
            base,before,after,base_nll=baseline[key];np.testing.assert_array_equal(z['before'],before);np.testing.assert_array_equal(z['after'],after)
            kl=float(np.dot(np.exp(base),base-lp));err=abs(kl-a['metrics']['kl']);maximum=max(maximum,err);assert err<1e-9
            loss=z['reference_nll'].copy();np.testing.assert_allclose(loss.mean(),a['metrics']['nll'],rtol=1e-12)
            prefix_len=len(tokens[task['sample_id']]['prompt_ids'])+task['prefix']
            assert a['positions']==list(range(prefix_len-task['width'],prefix_len)) and a['metrics']['patch_calls']==1
            assert len(loss)==min(cfg['reference_tokens'],len(tokens[task['sample_id']]['response_ids'])-task['prefix'])
            delta=after.astype(float)-before.astype(float);recovered=actual.astype(float)-before.astype(float)
            records.append({**task,'kl':kl,'reference_nll':float(loss.mean()),'delta_nll':float((loss-base_nll).mean()),
                            'update_relative_squared_error':float(np.square(recovered-delta).sum()/np.square(delta).sum()),
                            'state_relative_squared_error':float(np.square(actual.astype(float)-after).sum()/np.square(after.astype(float)).sum())})
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    write_json(out/'audit.json',{'complete':True,'records':len(records),'max_kl_error':maximum,'metrics_sha256':sha(out/'metrics.csv'),'plan_sha256':sha(r/'plan.json')})


def audit_risk(cfg):
    import joblib,pandas as pd
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    from hss.analysis.change_bayes import MonotonePlatt
    r=Path(cfg['output']);plan=verify(r);rows=pd.read_parquet(r/'rows.parquet');index={s:i for i,s in enumerate(rows.sample_id)}
    with np.load(r/'prefix_sequences.npz') as z:seqs={k:z[k].copy() for k in z.files}
    conf=np.full((5000,max(cfg['prefixes'])+1,4),np.nan,np.float32)
    for path in sorted((r/'confidence').glob('shard*.npz')):
        rec=json.loads(path.with_suffix('.json').read_text());assert sha(path)==rec['arrays_sha256'] and rec['plan_sha256']==sha(r/'plan.json')
        with np.load(path) as z:
            for sid,x in zip(z['sample_ids'],z['values']):conf[index[str(sid)]]=x
    checks=0;max_error=0
    for p in cfg['prefixes']:
        root=r/'risk'/f'p{p}';frozen=json.loads((root/'SCORES_FROZEN.json').read_text())
        for f,h in [('features.npz',frozen['features_sha256']),('scores.parquet',frozen['scores_sha256']),('selection.json',frozen['selection_sha256'])]:assert sha(root/f)==h
        with np.load(root/'features.npz') as z:features={k:z[k].copy() for k in z.files}
        ix=features['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(seqs['lengths']>p));frame=rows.iloc[ix].reset_index(drop=True)
        s=pd.read_parquet(root/'scores.parquet');prob=pd.read_parquet(root/'probabilities.parquet');sel=json.loads((root/'selection.json').read_text())
        np.testing.assert_array_equal(s.sample_id,frame.sample_id);np.testing.assert_array_equal(prob.sample_id,frame.sample_id)
        expected=np.stack([prefix_controls(conf[i],rows.n_prompt_tokens.iloc[i],seqs['kinds'][i],p) for i in ix])
        np.testing.assert_array_equal(expected,features['controls'])
        y=1-frame.label.to_numpy(int);tr=frame.split.eq('train').to_numpy();va=frame.split.eq('validation').to_numpy();te=frame.split.eq('test').to_numpy()
        # Audit every saved prefix table through direct categorical counts.
        for rep in ['delta','state']:
            for ri,i in enumerate(ix):
                a=seqs[rep][i,:p];ks=plan['cards'][rep];parts=[]
                for j,k in enumerate(ks):parts.append(np.array([(a[:,j]==c).sum()/p for c in range(k)]))
                occ=np.concatenate(parts);np.testing.assert_array_equal(occ,features[rep+'_occupancy'][ri]);checks+=1
                for j,(ka,kb) in enumerate(zip(ks,ks[1:])):
                    table=np.zeros((ka,kb));np.add.at(table,(a[:,j],a[:,j+1]),1/p);parts.append(table.ravel())
                depth=np.concatenate(parts);np.testing.assert_allclose(depth,features[rep+'_depth'][ri],atol=1e-15);checks+=1
                for j,k in enumerate(ks):
                    table=np.zeros((k,k));np.add.at(table,(a[:-1,j],a[1:,j]),1/(p-1));parts.append(table.ravel())
                np.testing.assert_allclose(np.concatenate(parts),features[rep+'_full'][ri],atol=1e-14);checks+=1
        with np.load(root/'folds.npz') as z:fold=z['fold'].copy();np.testing.assert_array_equal(z['row_index'],ix)
        with np.load(root/'train_oof.npz') as z:oofs={k:z[k].copy() for k in z.files}
        for rep in ['delta','state']:
            for kind in ['occupancy','depth','full']:
                name=rep+'_'+kind;x=features[name];m=joblib.load(root/'models'/name/'bayes.joblib')
                np.testing.assert_allclose(m.decision_function(x),s[name],atol=1e-10)
                offset=0;weights=[]
                for aa,bb in m.shapes:
                    v=np.stack([x[tr & (y==cl),offset:offset+aa*bb].sum(0).reshape(aa,bb)+cfg['alpha'] for cl in [0,1]])
                    v/=v.sum(-1,keepdims=True);weights.extend((np.log(v[1])-np.log(v[0])).ravel());offset+=aa*bb
                np.testing.assert_allclose(m.weights,weights,atol=1e-12)
                for f in range(5):
                    fm=joblib.load(root/'models'/name/f'fold{f}.joblib');hold=fold[tr]==f
                    np.testing.assert_allclose(fm.decision_function(x[tr][hold]),oofs[name][hold],atol=1e-10)
                lr=joblib.load(root/'models'/name/'adjusted/selected_spline.joblib')
                np.testing.assert_allclose(lr.decision_function(np.column_stack([features['controls'],s[name]])),s[name+'_adjusted'],atol=1e-10)
                if kind=='full':np.testing.assert_allclose(m.decision_function(features[rep+'_shuffled']),s[name+'_frozen_shuffle'],atol=1e-10)
        for name,file in [('controls','selected_spline.joblib'),('controls_hgb','hgb.joblib')]:
            m=joblib.load(root/'models/controls'/file);np.testing.assert_allclose(m.decision_function(features['controls']),s[name],atol=1e-10)
        with np.load(root/'bootstrap.npz') as z:boots=z['indices'].copy()
        with np.load(root/'bootstrap_auc.npz') as z:draws={k:z[k].copy() for k in z.files}
        metrics=json.loads((root/'metrics.json').read_text());tab={v['name']:v for v in metrics};yt=y[te]
        for metric in metrics:
            name=metric['name'];score=s[name].to_numpy();pr=prob[name].to_numpy();cal=MonotonePlatt();vars(cal).update(sel['calibrations'][name])
            np.testing.assert_allclose(cal.predict_proba(score),pr,atol=1e-12)
            observed=roc_auc_score(yt,score[te]);max_error=max(max_error,abs(observed-metric['auroc']));assert abs(observed-metric['auroc'])<1e-12
            np.testing.assert_allclose([average_precision_score(yt,score[te]),brier_score_loss(yt,pr[te])],[metric['auprc'],metric['brier']],atol=1e-12)
            threshold=sel['thresholds'][name];assert (score[va & (y==0)]>threshold).mean()<=cfg['far']+1e-12
            for b in range(0,cfg['bootstrap'],max(1,cfg['bootstrap']//20)):
                ii=boots[b];np.testing.assert_allclose(roc_auc_score(yt[ii],score[te][ii]),draws[name][b],atol=1e-12)
            np.testing.assert_allclose(np.quantile(draws[name],[.025,.975]),[metric['ci_low'],metric['ci_high']],atol=1e-12);checks+=1
        for c in json.loads((root/'contrasts.json').read_text()):
            np.testing.assert_allclose(tab[c['a']]['auroc']-tab[c['b']]['auroc'],c['delta'],atol=1e-12)
            np.testing.assert_allclose(np.quantile(draws[c['a']]-draws[c['b']],[.025,.975]),[c['ci_low'],c['ci_high']],atol=1e-12);checks+=1
        del features
    write_json(r/'risk/audit.json',{'complete':True,'checks':checks,'max_auc_error':max_error,'coverage':'Every prefix count table and control; train-only full Bayes weights; OOF score replay; classifier/calibration replay; point metrics and paired intervals; 20 independent bootstrap AUROCs per score.'})


def report(cfg):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=Path(cfg['output']);verify(r)
    assert json.loads((r/'risk/audit.json').read_text())['complete'] and json.loads((r/'functional/audit.json').read_text())['complete']
    out=r/'report';out.mkdir(exist_ok=True);m=pd.read_csv(r/'risk/metrics.csv');c=pd.read_csv(r/'risk/contrasts.csv');f=pd.read_csv(r/'functional/metrics.csv')
    coverage=json.loads((r/'risk/coverage.json').read_text());functional=[];comparisons=[];rng=np.random.default_rng(cfg['seed'])
    for (p,w),g in f.groupby(['prefix','width']):
        ids=sorted(g.sample_id.unique());boot=rng.integers(0,len(ids),(cfg['bootstrap'],len(ids)))
        for method in FUNCTIONAL_METHODS:
            v=g[g.method.eq(method)].set_index('sample_id').loc[ids]
            functional.append({'prefix':int(p),'width':int(w),'method':method,'questions':len(ids),**{col:float(v[col].mean()) for col in ['kl','delta_nll','update_relative_squared_error','state_relative_squared_error']}})
        for a,b in [('delta_local8','delta_shared8'),('delta_local8','zero_update'),('delta_local8','delta_wrong8'),('delta_centroid','global_update')]:
            aa=g[g.method.eq(a)].set_index('sample_id').loc[ids];bb=g[g.method.eq(b)].set_index('sample_id').loc[ids]
            for col in ['kl','reference_nll']:
                d=aa[col].to_numpy()-bb[col].to_numpy();draw=d[boot].mean(1)
                comparisons.append({'prefix':int(p),'width':int(w),'a':a,'b':b,'metric':col,'questions':len(ids),'delta':float(d.mean()),'ci_low':float(np.quantile(draw,.025)),'ci_high':float(np.quantile(draw,.975))})
        write_npz(out/f'functional_bootstrap_p{p}_w{w}.npz',sample_ids=np.array(ids),indices=boot)
    ft=pd.DataFrame(functional);fc=pd.DataFrame(comparisons);ft.to_csv(out/'functional_summary.csv',index=False);fc.to_csv(out/'functional_contrasts.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for name,color in [('controls','#8b8b8b'),('state_full_adjusted','#4977bd'),('delta_occupancy_adjusted','#b77642'),('delta_full_adjusted','#138271')]:
        q=m[m.name.eq(name)].sort_values('prefix');ax.errorbar(q.prefix,q.auroc,yerr=np.array([q.auroc-q.ci_low,q.ci_high-q.auroc]),label=name,color=color,marker='o',capsize=3)
    ax.set(xlabel='Observed response tokens (ongoing answers only)',ylabel='Failure AUROC',title='Frozen five-layer maps: prefix risk');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'risk.png',dpi=160);fig.savefig(out/'risk.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');q=ft[(ft.prefix==16)&(ft.width==4)]
    for ax,col in zip(axes,['kl','delta_nll']):
        ax.barh(q.method,q[col],color='#327c96');ax.set(xlabel=col,title='Block14 / prefix16 / last4 positions');ax.axvline(0,color='gray',lw=.7)
    fig.savefig(out/'functional.png',dpi=160);fig.savefig(out/'functional.pdf');plt.close(fig)
    primary={'risk':c[c.prefix.eq(cfg['primary_prefix'])].to_dict('records'),'functional':fc[(fc.prefix==16)&(fc.width==4)].to_dict('records')}
    summary={'scope':cfg['scope'],'coverage':coverage,'primary':primary,'risk':m.to_dict('records'),'functional':functional,'functional_contrasts':comparisons,
             'limits':['Historical automated correctness labels, previously explored data.','Conditional risk among ongoing answers, different cohorts per prefix.','No exact erroneous-step labels or free-generation correction claims.','Final-layer current entropy baseline requires completing the token forward pass; a mid-layer steering gate would require separate validation.','Fixed models; pointwise question bootstrap, no multiple-comparison adjustment.']}
    write_json(r/'summary.json',summary)
    css='<style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#192c3e}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced5db;padding:6px}th{background:#edf3f7}.scroll{overflow:auto}.note{padding:15px;background:#fff3d6}pre{white-space:pre-wrap}</style>'
    body='<h1>逐 token 层间更新：在线风险与功能检查</h1><p class="note">Qwen2 × MATH，冻结 0→1、6→7、13→14、20→21、27→28 的 GMM。只有已出现的前缀参与预测。目标是整题最终失败风险，不是证明某个 token／层“算错”。本轮没有新增自由生成 steering。</p>'
    body+='<h2>在线风险</h2><p>每个时间点仅纳入仍在继续的回答；长度固定为已观察的前缀长度，不使用最终回答长度。控制包含当前／已观察 token 分布的熵、margin、已观察 token NLL、prompt 长度和可见 token 类型比例。depth 使用同一 token 的跨层边，full 再加入同一层的时间边；标签只训练读出，GMM 不使用标签。</p><img src="risk.png">'
    body+='<h3>实际覆盖</h3>'+pd.DataFrame(coverage).to_html(index=False)+'<h3>预设主比较：prefix64</h3><div class="scroll">'+c[c.prefix.eq(cfg['primary_prefix'])].to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>'
    body+='<h2>功能检查</h2><p>固定32道历史测试题，block14，prefix16/64，修改最后1或4个生成位置。zero_update 放回本层输入；global_update 加训练平均更新；delta 方法重构更新后加回本层输入。state_local8 直接重构完整状态，信息预算不同。所有条件保持同一参考前缀和接下来最多32个参考 token；这是保真度，不是准确率。</p><img src="functional.png"><div class="scroll">'+fc[(fc.prefix==16)&(fc.width==4)].to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>'
    body+='<h2>逐题查看</h2><ul>';meta=pd.read_parquet(r/'rows.parquet').set_index('sample_id');seq=np.load(r/'prefix_sequences.npz');tokens=json.loads((r/'tokens.json').read_text())
    for sid in json.loads((r/'functional_tasks.json').read_text())['selected_ids']:
        row=meta.loc[sid];i=tokens[sid]['row_index'];pieces=[]
        for p in cfg['prefixes']:
            scores=pd.read_parquet(r/'risk'/f'p{p}'/'probabilities.parquet').set_index('sample_id')
            if sid in scores.index:pieces.append({'prefix':p,**{n:float(scores.loc[sid,n]) for n in ['controls','delta_full_adjusted','state_full_adjusted']}})
        case='<h1>'+sid+'</h1><p><a href="index.html">返回</a> · 原有自动评分：'+str(row.label)+'</p><h2>问题</h2><pre>'+html.escape(row.prompt_text)+'</pre><h2>可用前缀的校准失败概率</h2>'+pd.DataFrame(pieces).to_html(index=False,float_format=lambda x:f'{x:.4f}')
        case+='<h2>功能条件</h2><div class="scroll">'+f[f.sample_id.eq(sid)].to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div><h2>原回答</h2><pre>'+html.escape(row.response_text)+'</pre>'
        (out/(sid+'.html')).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+css+'<body>'+case+'</body></html>')
        write_json(out/(sid+'.json'),{'sample_id':sid,'prefix_risk':pieces,'delta_regions':seq['delta'][i].tolist(),'state_regions':seq['state'][i].tolist(),'layers':cfg['layers'],'response_ids':tokens[sid]['response_ids'][:128]})
        body+='<li><a href="'+sid+'.html">'+sid+'</a></li>'
    body+='</ul><h2>完整指标</h2><div class="scroll">'+m.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div><h2>结论边界</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limits'])+'</ul>'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Δh 在线风险初测</title>'+css+'<body>'+body+'</body></html>')
    # Independent SciPy/sklearn audits above cover risk; direct paired vector arithmetic checks functional summary.
    for row in comparisons:
        g=f[(f.prefix==row['prefix'])&(f.width==row['width'])];a=g[g.method.eq(row['a'])].set_index('sample_id');b=g[g.method.eq(row['b'])].set_index('sample_id')
        with np.load(out/f'functional_bootstrap_p{row["prefix"]}_w{row["width"]}.npz') as z:
            ids=z['sample_ids'];d=(a.loc[ids,row['metric']]-b.loc[ids,row['metric']]).to_numpy();bounds=np.quantile(d[z['indices']].mean(1),[.025,.975])
        np.testing.assert_allclose([d.mean(),*bounds],[row['delta'],row['ci_low'],row['ci_high']],atol=1e-12)
    write_json(r/'report_SUCCESS.json',{'summary_sha256':sha(r/'summary.json'),'functional_contrasts_checked':len(comparisons),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True,choices=['smoke','audit','risk','report']);a=p.parse_args();cfg=config(a.config)
    if a.stage=='risk':audit_risk(cfg)
    elif a.stage=='report':report(cfg)
    else:audit_functional(cfg,a.stage=='smoke')

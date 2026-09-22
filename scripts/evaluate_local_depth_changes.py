"""Question-level supervised readouts of frozen token/sentence depth clusters."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.change_bayes import bootstrap_auc, far_threshold, MonotonePlatt
from hss.analysis.local_depth import sequence_features, SequenceBayes
from hss.experiments.artifacts import save_json, file_digest, lock
from evaluate_change_bayes import fit_adjustment


def read_inputs(cfg):
    root=Path(cfg['output']);rows=pd.read_parquet(root/'rows.parquet');splits=pd.read_parquet(root/'splits.parquet')
    np.testing.assert_array_equal(rows.sample_id,splits.sample_id)
    assert rows.question_group.is_unique and rows.sample_id.is_unique
    if not (root/'ASSIGNED.json').exists():raise RuntimeError('Full assignment stage incomplete')
    prepared=json.loads((root/'PREPARED.json').read_text());n=len(rows)
    names=[f'{tag}_{g}_{r}' for tag in ['selected','matched'] for g in ['token','sentence'] for r in ['state','delta']]
    sequences={name:[None]*n for name in names};lexical=np.zeros((n,6));lengths=[None]*n
    manifests=[dict(path='assignments/means.npz',sha256=file_digest(root/'assignments'/'means.npz'))]
    manifests += [dict(path=str(p.relative_to(root)),sha256=file_digest(p))
                  for p in sorted((root/'fits').glob('*/selection.json'))]
    for rec in prepared['shards']:
        shard=rec['shard'];p=root/'assignments'/f'{shard}.npz'
        receipt=json.loads((root/'assignments'/f'{shard}.json').read_text())
        assert file_digest(p)==receipt['sha256']
        manifests.append(dict(path=str(p.relative_to(root)),sha256=receipt['sha256']))
        a=np.load(p);layout=np.load(Path(cfg['cache'])/'shards'/shard/'layout.npz')
        qi=a['question_index'];tp=a['token_ptr'];sp=a['sentence_ptr']
        seg=pd.read_parquet(Path(cfg['cache'])/'shards'/shard/'segments.parquet')
        for j,i in enumerate(qi):
            for name in names:
                ptr=tp if '_token_' in name else sp
                if sequences[name][i] is not None:raise ValueError('Duplicate question assignment')
                sequences[name][i]=a[name][ptr[j]:ptr[j+1]].copy()
            lexical[i]=np.bincount(layout['token_kind'][tp[j]:tp[j+1]],minlength=6)/(tp[j+1]-tp[j])
            lengths[i]=seg.iloc[sp[j]:sp[j+1]].n_tokens.to_numpy()
            assert tp[j+1]-tp[j]==rows.n_tokens.iloc[i]
    assert all(v is not None for v in sequences[names[0]])
    means=np.load(root/'assignments'/'means.npz')
    for tag in ['selected','matched']:
        for r in ['state','delta']:
            name=f'{tag}_mean_{r}';sequences[name]=[x[None,:] for x in means[name]]
    inputs={};cards={}
    for name,seq in sequences.items():
        tag,g,r=name.split('_')
        ks=[]
        for l in cfg['layers']:
            p=root/'fits'/f'{g}_{r}_L{l:02d}'/'selection.json';s=json.loads(p.read_text())
            ks.append(s['k'] if tag=='selected' else s['matched_k'])
        cards[name]=ks
        occ,edge=sequence_features(seq,ks,rows.sample_id.to_numpy(),cfg['seed'])
        inputs[name+'_occupancy']=(occ,ks,False)
        if g!='mean':
            inputs[name+'_ordered']=(np.column_stack([occ,edge]),ks,True)
            shuffled_occ,shuffled_edge=sequence_features(seq,ks,rows.sample_id.to_numpy(),cfg['seed'],shuffle=True)
            np.testing.assert_array_equal(shuffled_occ,occ)
            inputs[name+'_shuffled']=(np.column_stack([occ,shuffled_edge]),ks,True)
    basic=np.column_stack([np.log1p(rows.n_tokens),rows.entropy])
    extra=np.column_stack([np.log1p([len(x) for x in lengths]),
        np.log1p([np.std(x) for x in lengths]),lexical])
    assert np.isfinite(basic).all() and np.isfinite(extra).all()
    return rows,splits,inputs,basic,np.column_stack([basic,extra]),cards,sequences,manifests


def evaluate(cfg):
    root=Path(cfg['output']);out=root/'evaluation';out.mkdir(exist_ok=True)
    if (out/'_SUCCESS.json').exists():return
    started=time.monotonic()
    rows,splits,inputs,basic,extended,cards,sequences,manifests=read_inputs(cfg)
    tr=splits.split.to_numpy()=='train';va=splits.split.to_numpy()=='validation';te=splits.split.to_numpy()=='test'
    y=1-rows.label.to_numpy(int)
    assert set(y[tr])=={0,1} and set(y[va])=={0,1} and set(y[te])=={0,1}
    protocol=dict(config=cfg,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_protocol_sha256=file_digest(root/'protocol.json'),assignment_files=manifests,
        target='failure=1',split=splits.split.value_counts().to_dict(),
        fit='Train correctness labels only; fixed alpha=1, independent question cross-fitting for combination.',
        selection='GMM training ICL; downstream logistic C selected on validation AUROC. No score reversal using test.',
        calibration='Validation-only monotone Platt; validation success FAR target 10%.',
        controls='Basic: log1p response tokens, mean token entropy. Extended: plus log1p segment count, log1p segment length std, six token-type proportions.',
        sequence='Question-normalized occupancy + row-conditional adjacent-unit transitions; composite score. Train and evaluate shuffled control with the same within-question randomization rule.',
        limits=['Exploratory repeatedly used dataset.', 'Pointwise bootstrap confidence intervals, no multiplicity correction.',
                'Sentence segmentation is punctuation/newline, not ground-truth reasoning steps.',
                'Shuffle changes absolute position structure as well as local ordering.',
                'Clustering is unsupervised but Bayesian readout uses correctness labels.'])
    save_json(out/'protocol.json',protocol)
    models=out/'models';models.mkdir(exist_ok=True)
    scores={};selection={};oof_scores={}
    folds=list(StratifiedKFold(n_splits=cfg['crossfit_folds'],shuffle=True,random_state=cfg['seed']).split(np.zeros(tr.sum()),y[tr]))
    fold_ids=np.full(len(rows),-1)
    for j,(_,hold) in enumerate(folds):fold_ids[np.flatnonzero(tr)[hold]]=j
    pd.DataFrame(dict(sample_id=rows.sample_id,split=splits.split,fold=fold_ids)).to_parquet(out/'folds.parquet',index=False)
    for name,x in [('length_entropy',basic),('extended_controls',extended)]:
        folder=models/name;folder.mkdir(exist_ok=True)
        s,h,info=fit_adjustment(x[tr],x,y[tr],y[va],va,cfg,folder)
        scores[name]=s;scores[name+'_hgb']=h;selection[name]=info
    for name,(x,ks,transitions) in inputs.items():
        folder=models/name;folder.mkdir(exist_ok=True)
        model=SequenceBayes(ks,cfg['alpha'],transitions).fit(x[tr],y[tr])
        score=model.decision_function(x);scores[name]=score
        joblib.dump(model,folder/'bayes.joblib')
        np.testing.assert_allclose(joblib.load(folder/'bayes.joblib').decision_function(x),score)
        oof=np.full(tr.sum(),np.nan)
        for j,(fit,hold) in enumerate(folds):
            m=SequenceBayes(ks,cfg['alpha'],transitions).fit(x[tr][fit],y[tr][fit])
            oof[hold]=m.decision_function(x[tr][hold]);joblib.dump(m,folder/f'fold{j}.joblib')
        assert np.isfinite(oof).all();oof_scores[name]=oof
        np.save(folder/'train_oof.npy',oof)
        selection[name]=dict(cardinalities=ks,alpha=cfg['alpha'],transitions=transitions)
        for suffix,controls in [('adjusted',basic),('extended',extended)]:
            sub=folder/suffix;sub.mkdir(exist_ok=True)
            s,h,info=fit_adjustment(np.column_stack([controls[tr],oof]),np.column_stack([controls,score]),
                                    y[tr],y[va],va,cfg,sub)
            scores[name+'__'+suffix]=s
            selection[name][suffix]=info
        # Frozen ordered readout on shuffled sequences: sensitivity without refitting.
        if name.endswith('_ordered'):
            sx=inputs[name.removesuffix('_ordered')+'_shuffled'][0]
            scores[name+'_frozen_shuffle']=model.decision_function(sx)
        print(json.dumps(dict(readout=name,validation_auc=float(roc_auc_score(y[va],score[va])))),flush=True)
    # Does order add to occupancy after the same nuisance controls? OOF supervised features.
    for base in cards:
        if '_mean_' in base:continue
        occ=base+'_occupancy';order=base+'_ordered';folder=models/(base+'_incremental');folder.mkdir(exist_ok=True)
        s,h,info=fit_adjustment(np.column_stack([basic[tr],oof_scores[occ],oof_scores[order]]),
            np.column_stack([basic,scores[occ],scores[order]]),y[tr],y[va],va,cfg,folder)
        scores[base+'_occupancy_plus_order__adjusted']=s;selection[base+'_incremental']=info
    thresholds={};probabilities={};calibration={}
    for name,score in scores.items():
        assert np.isfinite(score).all()
        cal=MonotonePlatt().fit(score[va],y[va]);probabilities[name]=cal.predict_proba(score)
        calibration[name]=vars(cal);thresholds[name]=far_threshold(score[va & (y==0)],cfg['far'])
    pd.DataFrame(dict(sample_id=rows.sample_id,split=splits.split,**scores)).to_parquet(out/'scores.parquet',index=False)
    pd.DataFrame(dict(sample_id=rows.sample_id,**probabilities)).to_parquet(out/'probabilities.parquet',index=False)
    save_json(out/'selection.json',dict(models=selection,thresholds=thresholds,calibration=calibration))
    save_json(out/'_SCORES_FROZEN.json',dict(time=time.time(),scores_sha256=file_digest(out/'scores.parquet'),selection_sha256=file_digest(out/'selection.json')))
    # Test correctness first enters evaluation after all models and scores are frozen.
    rng=np.random.default_rng(cfg['seed']+1);yt=y[te]
    pos,neg=np.flatnonzero(yt==1),np.flatnonzero(yt==0)
    boot=np.column_stack([rng.choice(pos,(cfg['bootstrap'],len(pos))),rng.choice(neg,(cfg['bootstrap'],len(neg)))])
    np.save(out/'bootstrap_indices.npy',boot)
    metrics=[];draws={}
    for name,score in scores.items():
        test=score[te];d=bootstrap_auc(yt,test,boot);draws[name]=d;alarm=test>thresholds[name]
        metrics.append(dict(name=name,auroc=float(roc_auc_score(yt,test)),ci_low=float(np.quantile(d,.025)),ci_high=float(np.quantile(d,.975)),
            auprc=float(average_precision_score(yt,test)),test_far=float(alarm[yt==0].mean()),failure_recall=float(alarm[yt==1].mean()),
            validation_far=float((score[va & (y==0)]>thresholds[name]).mean()),brier=float(brier_score_loss(yt,probabilities[name][te]))))
    table=pd.DataFrame(metrics).set_index('name');table.reset_index().to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'bootstrap_auroc.npz',**draws)
    contrasts=[]
    def compare(description,a,b):
        delta=draws[a]-draws[b]
        contrasts.append(dict(comparison=description,a=a,b=b,delta=float(table.loc[a,'auroc']-table.loc[b,'auroc']),
            ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975))))
    for tag in ['selected','matched']:
        for g in ['token','sentence']:
            for r in ['delta','state']:
                base=f'{tag}_{g}_{r}';mean=f'{tag}_mean_{r}_occupancy';occ=base+'_occupancy';order=base+'_ordered';sh=base+'_shuffled'
                compare('Ordered minus occupancy',order,occ)
                compare('Ordered minus refitted shuffled',order,sh)
                compare('Ordered minus frozen-model shuffled',order,order+'_frozen_shuffle')
                compare('Order added beyond occupancy and controls',base+'_occupancy_plus_order__adjusted',occ+'__adjusted')
                for n in [occ,order]:
                    compare('Added beyond length/entropy',n+'__adjusted','length_entropy')
                    compare('Added beyond extended controls',n+'__extended','extended_controls')
                for suffix in ['', '__adjusted','__extended']:
                    compare('Local occupancy minus whole mean'+suffix,occ+suffix,mean+suffix)
            for method in ['occupancy','ordered']:
                for suffix in ['', '__adjusted','__extended']:
                    compare('Delta minus matched-granularity state'+suffix,
                            f'{tag}_{g}_delta_{method}'+suffix,f'{tag}_{g}_state_{method}'+suffix)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    save_json(out/'summary.json',dict(seconds=time.monotonic()-started,n_questions=len(rows),n_test=int(te.sum()),
        n_tokens=sum(len(v) for v in sequences['selected_token_delta']),n_sentences=sum(len(v) for v in sequences['selected_sentence_delta']),
        cards=cards,models=len(inputs),scores=len(scores),target='failure',bootstrap='2000 question-level stratified paired draws; pointwise 95% intervals'))
    # Retain text plus exact cluster paths for reviewing each held-out question.
    cases=out/'report'/'cases';cases.mkdir(parents=True,exist_ok=True);index=[]
    segments=pd.read_parquet(root/'segments.parquet')
    for i in np.flatnonzero(te):
        row=rows.iloc[i];sid=str(row.sample_id)
        record=dict(sample_id=sid,correct=int(row.label),question=str(row.prompt_text),response=str(row.response_text),
            answer=str(row.ground_truth),tokens=int(row.n_tokens),entropy=float(row.entropy),layers=cfg['layers'],
            sentences=segments[segments.question_index==i].to_dict('records'),
            sentence_delta=sequences['selected_sentence_delta'][i].tolist(),token_delta=sequences['selected_token_delta'][i].tolist(),
            scores={n:float(scores[n][i]) for n in ['selected_sentence_delta_occupancy','selected_sentence_delta_ordered','selected_token_delta_occupancy','selected_token_delta_ordered']})
        save_json(cases/f'{sid}.json',record);index.append(dict(sample_id=sid,correct=int(row.label),tokens=int(row.n_tokens),scores=record['scores']))
    save_json(out/'report'/'cases.json',index)
    render(cfg)
    frozen=json.loads((out/'_SCORES_FROZEN.json').read_text())
    assert file_digest(out/'scores.parquet')==frozen['scores_sha256']
    assert file_digest(out/'selection.json')==frozen['selection_sha256']
    assert (table.validation_far<=cfg['far']+1e-12).all()
    for m in manifests:assert file_digest(root/m['path'])==m['sha256']
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ['inventory.json','_SUCCESS.json']]
    save_json(out/'inventory.json',[dict(path=str(p.relative_to(out)),size=p.stat().st_size,sha256=file_digest(p)) for p in files])
    save_json(out/'_SUCCESS.json',dict(scores_frozen=True,assignment_hashes_verified=len(manifests),validation_far_verified=True,
        inventory_sha256=file_digest(out/'inventory.json'),seconds=time.monotonic()-started))
    print(table.loc[['length_entropy','extended_controls','selected_mean_delta_occupancy','selected_sentence_delta_occupancy',
                    'selected_sentence_delta_ordered','selected_token_delta_occupancy','selected_token_delta_ordered']].to_string(),flush=True)


def render(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output']);out=root/'evaluation';folder=out/'report';folder.mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'metrics.csv').set_index('name');contrasts=pd.read_csv(out/'contrasts.csv')
    summary=json.loads((out/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    names=['selected_mean_delta_occupancy','selected_sentence_delta_occupancy','selected_sentence_delta_ordered',
           'selected_sentence_delta_shuffled','selected_token_delta_occupancy','selected_token_delta_ordered','selected_token_delta_shuffled']
    labels=['Whole mean','Sentence occupancy','Sentence ordered','Sentence shuffled','Token occupancy','Token ordered','Token shuffled']
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    for ax,suffix,title in zip(axes,['','__adjusted'],['Cluster readout','Combined with length + entropy']):
        subset=metrics.loc[[n+suffix for n in names]]; yy=np.arange(len(names))
        ax.errorbar(subset.auroc,yy,xerr=np.maximum(0,np.vstack([subset.auroc-subset.ci_low,subset.ci_high-subset.auroc])),fmt='o',capsize=3)
        ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Held-out failure AUROC')
        ax.axvline(metrics.loc['length_entropy','auroc'],ls='--',color='#ad622c',label='Length + entropy');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Qwen2 MATH | Same-token adjacent-layer changes | 5 layer pairs');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    selections=[]
    for p in sorted((root/'fits').glob('*/selection.json')):
        s=json.loads(p.read_text());selections.append(dict(view=s['name'],K=s['k'],boundary=s['boundary'],matched_K=s['matched_k'],
            converged=sum(c['converged'] for c in s['candidates']),candidates=len(s['candidates'])))
    pd.DataFrame(selections).to_csv(out/'cluster_selection.csv',index=False)
    save_json(folder/'data.json',dict(summary=summary,metrics=metrics.reset_index().to_dict('records'),contrasts=contrasts.to_dict('records'),selection=selections))
    table=metrics.reset_index().round(4).to_html(index=False,escape=True)
    contrast_table=contrasts.round(4).to_html(index=False,escape=True)
    selection_table=pd.DataFrame(selections).to_html(index=False,escape=True)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐 token／逐句层间变化 · Qwen MATH</title><style>body{max-width:1380px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;background:#f5f7fa;color:#213047}section{background:white;padding:24px;margin:24px 0;border-radius:12px;border:1px solid #dbe2eb}h1{line-height:1.3}img{max-width:100%}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #dbe2eb;text-align:left}.scroll{overflow:auto;max-height:650px}.note{padding:18px;background:#fff5df;border-left:4px solid #c88c26}select,button,input{padding:8px;font:inherit;margin:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui;max-height:500px;overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}a{color:#225ca3}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>更细的层间变化，是否提供了额外信号？</h1>
<p>Qwen2-7B-Instruct × MATH 5,000 · 层对 0→1、6→7、13→14、20→21、27→28 · 原始 3,584 维 · 末层 pre RMSNorm</p>
<div class="note">探索性留出评估：训练 3,011／验证 1,003／测试 986。已反复研究过的数据；不是独立确认实验。聚类不使用正误标签，贝叶斯读出使用训练标签。完整回答评分不能解释成提前预测，也不能把整题标签当作局部步骤标签。</div>
<section><h2>比较框架</h2><p>整段平均 → 全部句子／token 的簇占比 → 占比与真实相邻转移 → 题内乱序。每个粒度均有原始 hidden state 对照及固定 K=8 对照。簇占比、转移均按题目归一；句子是标点／换行段落。乱序会同时破坏位置结构。</p><p>__COUNT__</p><img src="comparison.png"><p><a href="comparison.pdf">图 PDF</a> · <a href="../metrics.csv">指标 CSV</a> · <a href="../contrasts.csv">配对比较 CSV</a></p></section>
<section><h2>如何读结果</h2><p>首先看局部占比是否超过整段平均；其次看 ordered 是否超过 occupancy 和 shuffled；最后看加入相同长度／熵与扩展控制后的增益是否仍存在。扩展控制包含句数、句长离散度、token 类型比例。只有原始 AUROC 增加不能说明存在独立的推理信号。</p><p>selected 为各视图训练 ICL 选 K；matched 为统一 K=8。adjusted 加入长度／熵，extended 加入扩展控制。frozen_shuffle 使用原顺序分类器直接评分乱序，shuffled 则在乱序训练集上重训。所有区间为题目级配对 bootstrap 的逐项 95% CI，未做多重比较校正。</p></section>
<section><h2>全部指标</h2><div class="scroll">__TABLE__</div></section>
<section><h2>配对增益</h2><div class="scroll">__CONTRASTS__</div></section>
<section><h2>逐层聚类选择</h2><p>两种子，K∈{1,4,8,16,32}；边界命中表示还未定位最优 K。单位之间相关，ICL 独立样本假设仅为近似。</p><div class="scroll">__SELECTION__</div></section>
<section><h2>逐题查看句子更新簇</h2><select id="method"><option>selected_sentence_delta_ordered</option><option>selected_sentence_delta_occupancy</option><option>selected_token_delta_ordered</option><option>selected_token_delta_occupancy</option></select><select id="filter"><option value="all">全部</option><option value="0">错误回答</option><option value="1">正确回答</option></select><input id="query" placeholder="sample ID"><button id="prev">上一题</button><button id="next">下一题</button><p id="meta"></p><div class="grid"><div><h3>题目</h3><pre id="question"></pre><h3>参考答案</h3><pre id="answer"></pre></div><div><h3>回答</h3><pre id="response"></pre></div></div><h3>句子 × 层对的更新簇</h3><div id="segments" class="scroll"></div><p id="download"></p></section>
<script>let cases=[],filtered=[],cursor=0;const $=id=>document.getElementById(id);async function show(){if(!filtered.length){$('meta').textContent='没有匹配题目';return;}cursor=(cursor+filtered.length)%filtered.length;const c=await fetch('cases/'+encodeURIComponent(filtered[cursor].sample_id)+'.json').then(r=>r.json());$('meta').textContent=`${cursor+1}/${filtered.length} · ${c.sample_id} · ${c.correct?'正确':'错误'} · ${c.tokens} tokens · score ${c.scores[$('method').value].toFixed(3)}`;for(const k of ['question','answer','response'])$(k).textContent=c[k];const table=document.createElement('table');const head=table.insertRow();['文本段','token 范围',...c.layers.map(l=>(l-1)+'→'+l)].forEach(t=>{let th=document.createElement('th');th.textContent=t;head.appendChild(th)});c.sentences.forEach((s,i)=>{let tr=table.insertRow();[c.response.slice(Math.max(0,s.char_start),Math.max(s.char_start,s.char_end)),`${s.token_start}–${s.token_end}`,...c.sentence_delta[i]].forEach(t=>{let td=tr.insertCell();td.textContent=t;td.style.whiteSpace='pre-wrap';td.style.maxWidth='600px'})});$('segments').replaceChildren(table);const a=document.createElement('a');a.href='cases/'+encodeURIComponent(c.sample_id)+'.json';a.textContent='下载该题 JSON（含完整 token 路径）';$('download').replaceChildren(a)}function refresh(){filtered=cases.filter(c=>($('filter').value==='all'||String(c.correct)===$('filter').value)&&c.sample_id.includes($('query').value)).sort((a,b)=>b.scores[$('method').value]-a.scores[$('method').value]);cursor=0;show()}for(const id of ['method','filter','query'])$(id).onchange=refresh;$('prev').onclick=()=>{cursor--;show()};$('next').onclick=()=>{cursor++;show()};fetch('cases.json').then(r=>r.json()).then(v=>{cases=v;refresh()});</script></html>'''
    page=page.replace('__COUNT__',f"完整评分：{summary['n_tokens']:,} tokens，{summary['n_sentences']:,} 文本段。")
    for key,value in [('__TABLE__',table),('__CONTRASTS__',contrast_table),('__SELECTION__',selection_table)]:page=page.replace(key,value)
    (folder/'index.html').write_text(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/local-depth-changes.toml');p.add_argument('--render-only',action='store_true')
    p.add_argument('--wait-for-assignment',action='store_true',help='Wait for the already-running prepare/fit/assign process; fail if it exits incomplete')
    args=p.parse_args();cfg=load_config(args.config)
    if args.wait_for_assignment:
        print('Waiting for RUN.lock; evaluation begins only after ASSIGNED.json exists.',flush=True)
        with lock(Path(cfg['output'])/'RUN.lock'):
            if not (Path(cfg['output'])/'ASSIGNED.json').exists():raise RuntimeError('Upstream exited before completing assignments; inspect run.log')
    with threadpool_limits(cfg['cpu_threads']),lock(Path(cfg['output'])/'EVALUATION.lock'):
        if args.render_only:render(cfg)
        else:evaluate(cfg)

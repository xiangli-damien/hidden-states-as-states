"""Retrospective label evaluation of frozen, label-free route detectors.

No fitted detector or chosen parameter is changed here. The optional nuisance
regression predicts detector scores, not correctness; its fit uses validation
only. All reported performance is on detector-held-out test questions, with
the explicitly transductive state-map limitation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from hss.experiments.artifacts import file_digest, save_json
from run_route_study import fdr, conditional_auc


def percentile(reference, x):
    return np.searchsorted(np.sort(reference), x, side='right')/(len(reference)+1)


def auc_bootstrap(y, s, indices):
    # Rank-sum identity, with average ranks for ties; avoids thousands of
    # sklearn validation calls while preserving the exact bootstrap statistic.
    values=[]
    for ii in np.array_split(indices,max(1,int(np.ceil(len(indices)/200)))):
        yy=y[ii];n1=yy.sum(1);n0=yy.shape[1]-n1
        ranks=rankdata(s[ii],axis=1)
        values.append(((ranks*yy).sum(1)-n1*(n1+1)/2)/(n1*n0))
    return np.concatenate(values)


def run(args):
    root=Path(args.root);out=root/'evaluation';out.mkdir(exist_ok=True)
    frozen=json.loads((root/'_SCORES_FROZEN.json').read_text())
    if file_digest(root/'unlabeled_scores.parquet')!=frozen['scores_sha256']:raise ValueError('Scores changed')
    if file_digest(root/'selected.json')!=frozen['selected_sha256']:raise ValueError('Selection changed')
    scores=pd.read_parquet(root/'unlabeled_scores.parquet')
    meta=pd.read_parquet(Path(args.source)/'inputs/rows.parquet').set_index('sample_id').loc[scores.sample_id].reset_index()
    tr=scores.split.to_numpy()=='train';va=scores.split.to_numpy()=='validation';te=scores.split.to_numpy()=='test'
    y=1-meta.label.to_numpy(int);yt=y[te];rng=np.random.default_rng(2051921)
    protocol=json.loads((root/'protocol.json').read_text())
    boot=rng.integers(te.sum(),size=(protocol['config']['bootstrap'],te.sum()))
    length=np.log1p(meta.n_tokens.to_numpy(float));entropy=meta.entropy.to_numpy(float)
    if not np.isfinite(entropy).all():raise ValueError('Missing token entropy')
    nuisance=np.column_stack([length,entropy])
    strata=np.zeros(len(y),int)
    for values in [length,entropy]:
        strata=strata*4+np.searchsorted(np.unique(np.quantile(values[tr],[.25,.5,.75])),values,side='right')
    pl=percentile(length[va],length);pe=percentile(entropy[va],entropy)
    joint=(pl+pe)/2;baseline_auc=roc_auc_score(yt,joint[te]);joint_boot=auc_bootstrap(yt,joint[te],boot)
    rows=[];curves={};per_question=pd.DataFrame(dict(sample_id=meta.sample_id[te],failure=y[te],n_tokens=meta.n_tokens[te],entropy=entropy[te]))
    def evaluate(name,s,baseline=False):
        test=s[te];auc=float(roc_auc_score(yt,test));bs=auc_bootstrap(yt,test,boot)
        cond=conditional_auc(yt,test,strata[te]);order=np.argsort(test,kind='stable')
        risk=yt[order].cumsum()/np.arange(1,len(yt)+1)
        curves[name]=dict(coverage=np.linspace(1/len(yt),1,len(yt)).tolist(),risk=risk.tolist())
        threshold=np.quantile(s[va],.9)
        flag=test>threshold
        row=dict(name=name,map=name.split('__')[0],method=name.split('__')[1],score=name.split('__')[2] if '__' in name and len(name.split('__'))>2 else 'mean',
            failure_auroc=auc,ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),
            failure_auprc=float(average_precision_score(yt,test)),conditional_auc=cond['auc'],conditional_support=cond['support_n'],
            risk_at_50=float(risk[int(len(yt)*.5)-1]),flag_fraction=float(flag.mean()),
            observed_correct_far=float(flag[yt==0].mean()),observed_failure_recall=float(flag[yt==1].mean()),
            p_greater_chance=float(mannwhitneyu(test[yt==1],test[yt==0],alternative='greater').pvalue))
        if not baseline:
            ps=percentile(s[va],s);combined=(pl+pe+ps)/3
            comb_bs=auc_bootstrap(yt,combined[te],boot)
            row.update(combined_auc=float(roc_auc_score(yt,combined[te])),
                combined_delta=float(roc_auc_score(yt,combined[te])-baseline_auc),
                combined_delta_low=float(np.quantile(comb_bs-joint_boot,.025)),combined_delta_high=float(np.quantile(comb_bs-joint_boot,.975)))
            # Conditional typicality diagnostic: regress score on length/entropy,
            # with no correctness target and no test rows used in fitting.
            reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=10.,random_state=920).fit(nuisance[va],s[va])
            residual=s-reg.predict(nuisance)
            row['nuisance_residual_auc']=float(roc_auc_score(yt,residual[te]))
        rows.append(row);per_question[name]=test
    for name in scores:
        if name not in ['sample_id','split']:evaluate(name,scores[name].to_numpy())
    for name,s in [('baseline__length__mean',length),('baseline__entropy__mean',entropy),('baseline__length_entropy__mean',joint)]:evaluate(name,s,True)
    table=pd.DataFrame(rows);table['fdr_q']=np.nan
    primary=(table['map']!='baseline')&(table.score=='mean')
    table.loc[primary,'fdr_q']=fdr(table.loc[primary,'p_greater_chance'].to_numpy())
    table.to_csv(out/'leaderboard.csv',index=False);per_question.to_parquet(out/'test_scores.parquet',index=False)
    save_json(out/'risk_coverage.json',curves)
    predictive=[]
    for row in json.loads((root/'selected.json').read_text()):
        if row['method'] not in ['node','markov','mixture_markov','hmm','gru','causal_transformer']:continue
        loss=np.mean([np.load(root/name/'scores.npz')['losses'] for name in row['candidates']],axis=0)
        predictive.append(dict(map=row['map'],method=row['method'],paramkey=row['paramkey'],
            train_nll=float(loss[tr].mean()),validation_nll=float(loss[va].mean()),test_nll=float(loss[te].mean())))
    pd.DataFrame(predictive).to_csv(out/'structure_learning.csv',index=False)
    # Seeds/configurations remain exploratory diagnostics; never choose by these.
    all_candidates=[]
    for r in json.loads((root/'candidates.json').read_text()):
        s=np.load(root/r['path']/'scores.npz')['mean']
        training=json.loads((root/r['path']/'training.json').read_text())
        all_candidates.append(dict(map=r['map'],method=r['method'],paramkey=r['paramkey'],seed=r['seed'],
            validation=r['validation'],seconds=r['seconds'],failure_auroc=float(roc_auc_score(yt,s[te])),
            converged=training.get('converged'),selected_epoch=training.get('best_epoch'),path=r['path']))
    pd.DataFrame(all_candidates).to_csv(out/'candidate_diagnostics.csv',index=False)
    save_json(out/'summary.json',dict(test_n=int(te.sum()),test_correct=int((yt==0).sum()),test_incorrect=int(yt.sum()),
        train_n=int(tr.sum()),validation_n=int(va.sum()),candidates=len(all_candidates),primary_scores=int(primary.sum()),
        label_free_joint_baseline_auc=float(baseline_auc),
        training_scope='Only route fitting is held out. Original GMM/MFA maps used all samples; previously explored dataset.',
        increment='Equal-weight validation-percentile average of length, entropy and route versus length/entropy average. No correctness fit.',
        conditional='Pair-weighted AUROC inside training-defined length-quartile x entropy-quartile strata; incomplete confounder control.',
        nuisance_residual='Score-minus-predicted-score; regression uses only validation length/entropy and detector scores, no labels.',
        labels='Used only in this evaluation. No score sign, architecture, parameter, seed, ensemble weight or threshold chosen by correctness.',
        multiplicity='BH for greater-than-chance Mann-Whitney tests over all primary selected map/method mean scores. Bootstrap CIs/deltas are pointwise; ranking is exploratory.',
        checkpoints='All configurations and seeds saved, including unselected models.',
        observed_far='Threshold is validation 90th percentile; observed test FAR is reported descriptively, not guaranteed.',
        frozen_scores_sha256=frozen['scores_sha256']))
    render(root,table,meta.loc[te].copy(),per_question)
    print(table[primary].sort_values('failure_auroc',ascending=False)[['name','failure_auroc','conditional_auc','combined_delta']].head(15).to_string(index=False))


def render(root,table,meta,test_scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from html import escape
    out=root/'report';out.mkdir(exist_ok=True)
    primary=table[table.score=='mean'].copy();routes=primary[primary['map']!='baseline']
    matrix=routes.pivot(index='method',columns='map',values='failure_auroc')
    map_titles={'gmm':'GMM','gmm_matched':'GMM\nmatched K','mfa':'MFA'}
    labels=[m.replace('_',' ') for m in matrix.index]
    fig,ax=plt.subplots(figsize=(9,9));im=ax.imshow(matrix.values,norm=TwoSlopeNorm(vmin=.3,vcenter=.5,vmax=.8),cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(len(matrix.columns)),[map_titles[c] for c in matrix.columns]);ax.set_yticks(range(len(matrix)),labels)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',fontsize=10)
    ax.set_title('Unsupervised route scores | test failure AUROC\nFrozen whole-data maps; higher score fixed as failure')
    fig.colorbar(im,ax=ax,shrink=.65);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'auroc.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,sharex=True)
    methods=matrix.index.tolist()
    for ax,map_name in zip(axes,matrix.columns):
        a=routes[routes['map']==map_name].set_index('method').loc[methods]
        ax.hlines(np.arange(len(a)),a.combined_delta_low,a.combined_delta_high,color='steelblue')
        ax.plot(a.combined_delta,np.arange(len(a)),'o',color='steelblue')
        ax.axvline(0,color='black',lw=1);ax.set_title(map_titles[map_name].replace('\n',' '));ax.set_xlabel('AUROC change vs length + entropy')
        ax.set_yticks(np.arange(len(a)),labels);ax.grid(alpha=.2)
    axes[0].invert_yaxis();fig.suptitle('Fixed equal-weight percentile combination | pointwise 95% CI\nNo label-fitted weights; exploratory comparisons');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'increment.{ext}',dpi=180)
    plt.close(fig)
    records=table.replace({np.nan:None}).to_dict('records')
    selected=json.loads((root/'selected.json').read_text());summary=json.loads((root/'evaluation/summary.json').read_text())
    predictive=pd.read_csv(root/'evaluation/structure_learning.csv').to_dict('records')
    save_json(out/'data.json',dict(leaderboard=records,selected=selected,summary=summary,predictive=predictive))
    items=[]
    for _,row in meta.iterrows():
        i=test_scores[test_scores.sample_id==row.sample_id].iloc[0]
        items.append(dict(id=str(row.sample_id),correct=int(row.label),tokens=int(row.n_tokens),entropy=float(row.entropy),
            prompt=str(row.prompt_text),response=str(row.response_text),ground_truth=str(row.ground_truth),
            scores={k:float(i[k]) for k in test_scores.columns if '__' in k}))
    save_json(out/'samples.json',items)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>无标签路径学习 · 15 类算法</title>
<style>body{max-width:1250px;margin:40px auto;padding:0 24px;background:#f6f7fb;color:#17253a;font:16px/1.6 system-ui}h1{font-size:32px}section{background:white;border:1px solid #dce2ea;border-radius:12px;padding:24px;margin:24px 0}.note{border-left:4px solid #df9b29;padding:16px;background:#fff5df}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #e4e8ee;text-align:right}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:#e9eff7}img{max-width:100%}select,input,button{font:inherit;padding:8px;margin:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;font:14px/1.6 ui-monospace}a{color:#225bc6}.muted{color:#60728a}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>无标签路径学习：15 类算法 + 节点基线</h1><p>Qwen2-7B-Instruct × MATH · GMM / MFA / 同 K GMM · 完整回答的跨层状态序列</p>
<div class="note"><b>探索性结果：</b>检测器按训练／验证／测试划分，原聚类地图曾使用全部 5,000 题。异常分数不等于错误概率；所有方向、参数选择和组合权重均不根据正确性标签调整。</div>
<section><h2>本轮完成情况</h2><p id="summary">加载中…</p><p>15 类：高阶 Markov、路径混合、层相关 HMM、kNN、LOF、Isolation Forest、One-Class SVM、PCA 重构、去噪 AE、VAE、GRU、因果 Transformer、掩码 Transformer、Deep SVDD、后缀打乱对比学习。另报告节点罕见程度与固定等权集成。</p><p class="muted">神经方法为针对离散路径的实现；不宣称复现每篇原论文，也不宣称穷尽所有算法。PCA 作用于类别 one-hot，不改变原始 hidden state 聚类。</p></section>
<section><h2>测试表现</h2><p>AUROC 0.5 为随机排序；分数方向固定为越高越可能错误。主分数为逐层平均，top4 是预设的局部异常敏感性结果。</p><label>地图 <select id="map"><option>mfa</option><option>gmm</option><option>gmm_matched</option></select></label><label>汇总 <select id="agg"><option>mean</option><option>top4</option></select></label><div style="overflow:auto"><table><thead><tr><th>方法</th><th>AUROC [95% CI]</th><th>长度×熵组内</th><th>加入基线 Δ</th><th>50%覆盖错误率</th><th>观测 FAR</th></tr></thead><tbody id="table"></tbody></table></div><p class="muted">“加入基线”用验证集百分位固定等权合并长度、熵、路径，与长度+熵比较；不是监督拟合的最优分类器。区间为逐项区间，不能用挑出的最佳值声称独立验证成功。FAR 为验证集90%分位阈值下的测试观测值，无10%保证。</p></section>
<section><h2>跨地图比较</h2><img src="auroc.png" alt="各方法AUROC热图"><p><a href="auroc.pdf">下载 PDF</a></p><img src="increment.png" alt="加入长度和熵基线后的AUROC变化"><p><a href="increment.pdf">下载 PDF</a></p></section>
<section><h2>逐题查看：测试集</h2><select id="method"></select><select id="order"><option value="high">高分优先</option><option value="low">低分优先</option></select><input id="query" placeholder="搜索题目／sample ID"><button id="prev">上一题</button><button id="next">下一题</button><p id="sampleMeta"></p><div class="grid"><div><h3>题目</h3><pre id="prompt"></pre><h3>参考答案</h3><pre id="gold"></pre></div><div><h3>模型回答</h3><pre id="response"></pre></div></div></section>
<section><h2>复核与下载</h2><p><a href="../evaluation/leaderboard.csv">所有分数与指标</a> · <a href="../evaluation/candidate_diagnostics.csv">210 个候选诊断</a> · <a href="../evaluation/structure_learning.csv">预测状态的 NLL：结构是否学会</a> · <a href="../protocol.json">冻结协议</a> · <a href="../selected.json">无标签选参记录</a> · <a href="../evaluation/summary.json">限制与统计定义</a></p><p>训练文件只读取 sample_id、question_group 和 state sequence；正确性首次进入单独评估脚本。所有候选模型、初始化、逐层损失和逐题分数均保留。</p></section>
<script>let data,samples,shown=[],pos=0;const el=x=>document.getElementById(x),fmt=x=>x==null?'—':x.toFixed(3);function table(){const rows=data.leaderboard.filter(r=>(r.map===el('map').value&&r.score===el('agg').value)||r.map==='baseline').sort((a,b)=>b.failure_auroc-a.failure_auroc);el('table').replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const v of [r.method,`${fmt(r.failure_auroc)} [${fmt(r.ci_low)}, ${fmt(r.ci_high)}]`,fmt(r.conditional_auc),fmt(r.combined_delta),fmt(r.risk_at_50),fmt(r.observed_correct_far)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}el('table').append(tr)}}function filter(){const k=el('method').value,q=el('query').value.toLowerCase(),sign=el('order').value==='high'?-1:1;shown=samples.filter(s=>(s.id+' '+s.prompt).toLowerCase().includes(q)).sort((a,b)=>sign*(a.scores[k]-b.scores[k]));pos=0;show()}function show(){if(!shown.length){el('sampleMeta').textContent='没有匹配的题目';return}const s=shown[pos],k=el('method').value;el('sampleMeta').textContent=`${pos+1}/${shown.length} · ${s.id} · ${s.correct?'正确':'错误'} · ${s.tokens} tokens · 熵 ${fmt(s.entropy)} · 分数 ${fmt(s.scores[k])}`;el('prompt').textContent=s.prompt;el('gold').textContent=s.ground_truth;el('response').textContent=s.response}Promise.all([fetch('data.json').then(r=>r.json()),fetch('samples.json').then(r=>r.json())]).then(([d,s])=>{data=d;samples=s;const a=d.summary;el('summary').textContent=`训练 ${a.train_n} / 验证 ${a.validation_n} / 测试 ${a.test_n}；测试正确 ${a.test_correct}、错误 ${a.test_incorrect}；完成 ${a.candidates} 个候选，${a.primary_scores} 个主分数。`;for(const k of Object.keys(s[0].scores).filter(k=>k.endsWith('__mean'))){const o=document.createElement('option');o.value=k;o.textContent=k;el('method').append(o)}table();filter()}).catch(e=>el('summary').textContent='加载失败：'+e);['map','agg'].forEach(k=>el(k).onchange=table);['method','order'].forEach(k=>el(k).onchange=filter);el('query').oninput=filter;el('prev').onclick=()=>{pos=(pos-1+shown.length)%shown.length;show()};el('next').onclick=()=>{pos=(pos+1)%shown.length;show()};</script></html>'''
    (out/'index.html').write_text(html)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--render-only',action='store_true');a=p.parse_args()
    with threadpool_limits(2):
        if a.render_only:
            root=Path(a.root);s=pd.read_parquet(root/'unlabeled_scores.parquet')
            m=pd.read_parquet(Path(a.source)/'inputs/rows.parquet').set_index('sample_id').loc[s.sample_id].reset_index()
            render(root,pd.read_csv(root/'evaluation/leaderboard.csv'),m.loc[(s.split=='test').to_numpy()],pd.read_parquet(root/'evaluation/test_scores.parquet'))
        else:run(a)

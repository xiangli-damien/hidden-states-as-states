"""Evaluate frozen change-space mixtures and render an inspectable report.

Correctness first enters here, after all unsupervised fits/selection finish.
Question-level randomization avoids treating correlated token pairs as labels.
"""
import argparse
import html
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.change_clusters import aggregate_pairs
from hss.experiments.artifacts import file_digest, save_json


def js(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float)
    p=p/p.sum(-1,keepdims=True);q=q/q.sum(-1,keepdims=True);m=(p+q)/2
    return (xlogy(p,p)-xlogy(p,m)+xlogy(q,q)-xlogy(q,m)).sum(-1)/(2*np.log(2))


def bh(p):
    p=np.asarray(p);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(p);out[order]=np.minimum(q,1);return out


def permutation_test(hist,y,groups,repetitions,seed):
    """Keep per-stratum label counts, permute only held-out QUESTIONS."""
    observed=float(js(hist[y==0].mean(0),hist[y==1].mean(0)))
    group=[np.flatnonzero(groups==s) for s in np.unique(groups)]
    rng=np.random.default_rng(seed);sims=[]
    for _ in range(repetitions):
        yy=y.copy()
        for idx in group:yy[idx]=rng.permutation(yy[idx])
        sims.append(float(js(hist[yy==0].mean(0),hist[yy==1].mean(0))))
    return dict(js_bits=observed,p=float((1+np.sum(np.asarray(sims)>=observed-1e-15))/(repetitions+1)),
                null_mean=float(np.mean(sims)),excess=float(observed-np.mean(sims)),
                movable_fraction=float(sum(len(i) for i in group if len(np.unique(y[i]))>1)/len(y)))


def group_bins(rows,train,test,columns,quantiles=4):
    values=[]
    for col in columns:
        x=rows[col].to_numpy(float)
        cuts=np.unique(np.quantile(x[train],np.arange(1,quantiles)/quantiles))
        values.append(np.searchsorted(cuts,x[test],side='right').astype(str))
    return np.array(['|'.join(parts) for parts in zip(*values)])


def plot_view(root,name,summary,rows,splits,pairs):
    folder=root/'fits'/name;dest=root/'report'/'figures';dest.mkdir(parents=True,exist_ok=True)
    saved=np.load(folder/'assignments.npz');display=np.load(folder/'display.npz')
    owner=saved['question_index'];z=saved['assignment'];xy=display['xy'];k=summary['k']
    test=splits.split.to_numpy()[owner]=='test';sel=np.flatnonzero(test)
    if len(sel)>5000:sel=np.random.default_rng(921).choice(sel,5000,replace=False)
    y=rows.label.to_numpy()[owner];colors=plt.get_cmap('turbo',max(k,2))
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    axes[0].scatter(xy[sel,0],xy[sel,1],c=z[sel],cmap=colors,s=6,alpha=.55,rasterized=True)
    axes[0].scatter(display['center_xy'][:,0],display['center_xy'][:,1],c='black',s=20,marker='x')
    axes[0].set_title(f'GMM components, K={k}')
    axes[1].scatter(xy[sel,0],xy[sel,1],c=y[sel],cmap='coolwarm_r',vmin=0,vmax=1,s=6,alpha=.4,rasterized=True)
    axes[1].set_title('Correct (blue) / incorrect (red)')
    for ax in axes[:2]:
        ax.set_xlabel(f'PC1 ({display["explained"][0]:.1%})');ax.set_ylabel(f'PC2 ({display["explained"][1]:.1%})')
    h=aggregate_pairs(np.eye(k)[z],owner,len(rows));mask=splits.split.to_numpy()=='test'
    yy=rows.label.to_numpy()[mask];hh=h[mask]
    a=hh[yy==1].mean(0);b=hh[yy==0].mean(0);xx=np.arange(k)
    axes[2].bar(xx-.19,a,.38,label='Correct',color='#2563eb');axes[2].bar(xx+.19,b,.38,label='Incorrect',color='#dc5c60')
    axes[2].set(xlabel='Local component',ylabel='Mean question occupancy');axes[2].legend(frameon=False)
    fig.suptitle(f'{name} — held-out questions; PCA is display only',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def evaluate(cfg):
    root=Path(cfg['output']);cache=Path(cfg['cache']);dest=root/'evaluation';dest.mkdir(exist_ok=True)
    if not all((root/f'{kind}_FITTED.json').exists() for kind in ['DEPTH','TEMPORAL']):
        raise RuntimeError('All unsupervised candidates must finish before evaluating labels')
    manifests=sum([json.loads((root/f'{kind}_views.json').read_text()) for kind in ['depth','temporal']],[])
    rows=pd.read_parquet(root/'rows.parquet');splits=pd.read_parquet(root/'splits.parquet')
    if 'question' not in rows:
        rows['question']=rows.prompt_text
    np.testing.assert_array_equal(rows.sample_id,splits.sample_id)
    train=splits.split.to_numpy()=='train';test=splits.split.to_numpy()=='test';y=rows.label.to_numpy(int)
    pairs=pd.read_parquet(root/'pairs.parquet')
    np.testing.assert_array_equal(rows.sample_id.to_numpy()[pairs.question_index],pairs.sample_id)
    if not np.isfinite(rows[['n_tokens','entropy']].to_numpy(float)).all():raise ValueError('Missing nuisance values')
    groups=group_bins(rows,train,test,['n_tokens','entropy'])
    detailed=group_bins(rows,train,test,['n_tokens','entropy'],quantiles=2)
    detailed=np.array([f'{g}|{cat}|{level}' for g,cat,level in zip(detailed,rows.category.to_numpy()[test],rows.level.to_numpy()[test])])
    outcomes=[];profiles=[];examples={};all_hist={};fits=[];token_tables=[]
    for view in manifests:
        name=view['name'];folder=root/'fits'/name;summary=json.loads((folder/'selection.json').read_text());fits.append(summary)
        if view['kind']=='gaussian_null':continue
        saved=np.load(folder/'assignments.npz');z=saved['assignment'];owner=saved['question_index'];k=summary['k']
        h=aggregate_pairs(np.eye(k)[z],owner,len(rows));all_hist[name]=h.astype(np.float32)
        if not np.isfinite(h).all():raise ValueError('Question missing from occupancy')
        unconditional=permutation_test(h[test],y[test],np.zeros(test.sum()),cfg['permutations'],921)
        controlled=permutation_test(h[test],y[test],groups,cfg['permutations'],921)
        controlled_task=permutation_test(h[test],y[test],detailed,cfg['permutations'],921)
        out=dict(name=name,kind=view['kind'],layer=view['layer'],k=k,
                 js_bits=unconditional['js_bits'],p=unconditional['p'],
                 controlled_p=controlled['p'],controlled_excess_js=controlled['excess'],
                 task_controlled_p=controlled_task['p'],task_controlled_excess_js=controlled_task['excess'],
                 task_controlled_movable=controlled_task['movable_fraction'])
        outcomes.append(out)
        gm=joblib.load(folder/'selected_model.joblib')
        x=np.load(cache/'views'/f'{name}.npy',mmap_mode='r')
        temporal=name.startswith('token_delta_')
        if temporal:
            pt=pairs[splits.split.to_numpy()[pairs.question_index]=='test'].copy()
            pt['cluster']=z[splits.split.to_numpy()[owner]=='test']
            pt['position_quartile']=np.minimum((pt.relative_position*4).astype(int),3)
            out['token_kind_nmi']=float(normalized_mutual_info_score(pt.cluster,pt.kind_before+'→'+pt.kind_after))
            out['position_nmi']=float(normalized_mutual_info_score(pt.cluster,pt.position_quartile))
            tab=pt.groupby(['cluster','kind_before','kind_after']).size().reset_index(name='count');tab['view']=name;token_tables.append(tab)
        examples[name]={}
        for cluster in range(k):
            weights=h[test,cluster];total=weights.sum()
            if total:
                correct=float(weights@y[test]/total)
                mean_length=float(weights@rows.n_tokens.to_numpy()[test]/total)
                mean_entropy=float(weights@rows.entropy.to_numpy()[test]/total)
            else:correct=mean_length=mean_entropy=None
            profile=dict(view=name,cluster=cluster,test_question_mass=float(total),test_correct_rate=correct,
                         weighted_mean_length=mean_length,weighted_mean_entropy=mean_entropy,
                         center_norm=float(np.linalg.norm(gm.means_[cluster])),
                         diagonal_variance_sum=float(gm.covariances_[cluster].sum()))
            profiles.append(profile)
            eligible=np.flatnonzero((z==cluster)&(splits.split.to_numpy()[owner]=='test'))
            if len(eligible):
                d=((np.asarray(x[eligible],float)-gm.means_[cluster])**2/gm.covariances_[cluster]).sum(1)
                ranked=np.argsort(d);take=[];seen=set()
                for j in ranked:
                    i=int(eligible[j]);q=int(owner[i])
                    if q not in seen:take.append((i,q,float(d[j])));seen.add(q)
                    if len(take)==4:break
                examples[name][str(cluster)]=[dict(question_index=q,vector_index=i,mahalanobis_sq=dist,
                     **(dict(position=int(pairs.iloc[i].position), **{key:str(pairs.iloc[i][key]) for key in ['piece_before','piece_after','kind_before','kind_after']}) if temporal else {})) for i,q,dist in take]
        plot_view(root,name,summary,rows,splits,pairs)
        print(json.dumps(dict(evaluated=name,k=k,js=out['js_bits'],controlled_p=out['controlled_p'])),flush=True)
    for column in ['p','controlled_p','task_controlled_p']:
        q=bh([r[column] for r in outcomes])
        for row,value in zip(outcomes,q):row[column.replace('_p','_q') if column!='p' else 'q']=float(value)
    pd.DataFrame(outcomes).to_csv(dest/'outcome_association.csv',index=False)
    pd.DataFrame(profiles).to_csv(dest/'cluster_profiles.csv',index=False)
    if token_tables:pd.concat(token_tables).to_csv(dest/'token_kind_counts.csv',index=False)
    np.savez_compressed(dest/'question_occupancy.npz',**all_hist)
    save_json(dest/'examples.json',examples);save_json(dest/'fit_summary.json',fits)
    # All saved candidates and selections are hashed before any label-based summaries.
    save_json(dest/'provenance.json',dict(at=time.time(),protocol_sha256=file_digest(root/'protocol.json'),
        selections={r['name']:file_digest(root/'fits'/r['name']/'selection.json') for r in fits},
        test_questions=int(test.sum()),test_correct=int(y[test].sum()),permutations=cfg['permutations'],
        primary_endpoint='JS(correct, incorrect) of per-question component occupancy on held-out questions',
        corrections='BH separately over all real-data views for unconditional and each conditional test family',
        limitations=['Conditional randomization only controls coarse bins; not a causal test.',
                     'Token-type association is descriptive; correctness tests do not fully control token content.',
                     'Mixture K is component count, not an established number of reasoning modes.',
                     'Temporal inference covers uniformly sampled pairs, not complete generated trajectories.']))
    render(cfg)


def render(cfg):
    root=Path(cfg['output']);dest=root/'report';dest.mkdir(exist_ok=True)
    fits=json.loads((root/'evaluation/fit_summary.json').read_text())
    stats=pd.read_csv(root/'evaluation/outcome_association.csv').set_index('name')
    profiles=pd.read_csv(root/'evaluation/cluster_profiles.csv')
    rows=pd.read_parquet(root/'rows.parquet');splits=pd.read_parquet(root/'splits.parquet')
    if 'question' not in rows:
        rows['question']=rows.prompt_text
    examples=json.loads((root/'evaluation/examples.json').read_text())
    real=[r for r in fits if not r['name'].startswith('gaussian_')]
    depth=[r for r in real if r['name'].startswith('mean_delta_')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    layers=[int(r['name'].split('L')[-1]) for r in depth]
    axes[0,0].plot(layers,[r['k'] for r in depth],'o-',label='Training ICL')
    axes[0,0].plot(layers,[r['k_validation'] for r in depth],'s--',label='Validation likelihood')
    axes[0,0].set(ylabel='Selected component count',title='Raw mean block updates');axes[0,0].legend()
    gain=np.array([r['test_gain_nats_per_dimension'] for r in depth]);ci=np.array([r['test_gain_ci'] for r in depth])
    axes[0,1].errorbar(layers,gain,yerr=np.maximum(0,np.array([gain-ci[:,0],ci[:,1]-gain])),fmt='o-',capsize=2)
    axes[0,1].axhline(0,color='gray',lw=.7);axes[0,1].set(ylabel='Test log-likelihood gain / dimension',title='Mixture versus one diagonal Gaussian')
    for kind,label in [('initialization','Different initialization'),('80pct_questions','80% training questions')]:
        values=[np.mean([s['ari'] for s in r['stability'] if s['kind']==kind and s['converged'] and (kind!='initialization' or s['seed']!=r['seed'])]) for r in depth]
        axes[1,0].plot(layers,values,'o-',label=label)
    axes[1,0].set(ylabel='Held-out assignment ARI',ylim=(-.05,1.05),title='Assignment reproducibility');axes[1,0].legend()
    axes[1,1].plot(layers,[stats.loc[r['name'],'js_bits'] for r in depth],'o-',label='Raw JS')
    axes[1,1].plot(layers,[stats.loc[r['name'],'controlled_excess_js'] for r in depth],'s-',label='Excess over length + entropy null')
    axes[1,1].set(ylabel='JS divergence (bits)',title='Correct / incorrect question occupancy');axes[1,1].legend()
    for ax in axes.ravel():ax.set_xlabel('Decoder block');ax.grid(alpha=.15)
    for ext in ['png','pdf']:fig.savefig(dest/f'overview.{ext}',dpi=170)
    plt.close(fig)
    records=[]
    for r in real:
        stat={k:None if pd.isna(v) else v for k,v in stats.loc[r['name']].to_dict().items()}
        records.append(dict(**r,association=stat,profiles=profiles[profiles.view==r['name']].replace({np.nan:None}).to_dict('records'),examples=examples[r['name']]))
    save_json(dest/'data.json',dict(views=records,nulls=[r for r in fits if r['name'].startswith('gaussian_')]))
    keep=['sample_id','question','response_text','ground_truth','label','category','level','n_tokens','entropy']
    frame=rows[[k for k in keep if k in rows]].copy();frame['split']=splits.split
    (dest/'questions.json').write_text(frame.to_json(orient='records',force_ascii=False),encoding='utf-8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>变化空间的 GMM｜Qwen MATH</title><style>
body{font:16px/1.7 system-ui;margin:32px auto;max-width:1320px;padding:0 20px;color:#203044;background:#fafbfc}h1{line-height:1.25}h2{margin-top:36px}section,.card{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:18px 0}img{max-width:100%}select{padding:9px;font-size:16px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{background:#f1f5f9}small,.muted{color:#617085}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.7 system-ui}summary{cursor:pointer;font-weight:600}.warn{border-left:4px solid #df9e36;padding-left:16px}.scroll{overflow-x:auto}.badge{background:#eef3f8;padding:5px 10px;border-radius:6px;margin-right:8px;display:inline-block}a{color:#2362b0}</style>
<h1>变化本身，会形成簇吗？</h1><p>Qwen2-7B-Instruct × MATH 5,000 · 跨层更新与同层跨 token 差值 · 原始 3,584 维 · train-only GMM</p>
<section><strong>两种差值分别建模</strong><p>跨层：Δh<sub>l,t</sub> = h<sub>l,t</sub> − h<sub>l−1,t</sub>，对同一回答的全部生成 token 平均，覆盖 28 个 block。末层采用 pre-RMSNorm。Prompt 对照只取 prompt 最后一个 token。</p><p>跨 token：Δh<sub>t</sub> = h<sub>t+1</sub> − h<sub>t</sub>，末层 pre/post 各做一次；每题均匀随机抽取最多 8 对相邻生成 token。每题簇占比按题等权，不让长回答主导。</p><p class="warn">跨 token 的平均差值严格等于 (末 token − 首 token)/(T−1)，仅是端点基线。token 簇占比是抽样估计，本轮没有对全部 token 标注轨迹。GMM 成分不自动等于密度峰或计算操作。</p></section>
<section><h2>全层平均更新概览</h2><img src="overview.png"><p><a href="overview.pdf">导出 PDF</a> · 似然增益相对于单个<strong>对角</strong>高斯，不能单独排除相关协方差的解释。ARI 是换初始化／80%训练题目后的测试簇一致性。</p></section>
<section><h2>检查每一种表示</h2><select id="view"></select><div id="metrics"></div><img id="plot"><p id="note"></p><div class="scroll"><table id="profiles"></table></div><h3>距离簇中心较近的测试题目</h3><select id="cluster"></select><div id="examples"></div></section>
<section><h2>单高斯对照：相关性会不会被切成多个簇？</h2><div id="nulls"></div><p>每个预设层生成一份单高斯数据，保留真实训练更新量的均值及完整经验协方差；不使用标签，不经过 PCA。它没有预置离散簇。仅一份模拟是诊断，不能报告为统计显著性检验。</p></section>
<section><h2>协议与解释边界</h2><ul><li>按 question group 固定划分：3,011 训练、1,003 验证、986 测试。同一题的 token 永不跨集合。</li><li>K 首轮 ∈ {1,2,4,8,16,32}；训练 ICL 选到 32 时扩展 {48,64,80}。每个 K 两次初始化；主结果按训练 ICL 最小选择，另列已搜索候选中的验证似然最优 K。仅接受收敛候选，触顶明确标注。</li><li>不 normalize、whiten 或用 PCA 拟合 GMM。图中 PCA 仅用于展示；高维中的分簇不一定在两个主成分上可见。</li><li>正误标签只用于冻结聚类后的描述和 999 次测试题目置换。BH 校正覆盖全部真实数据表示，各控制家族分别校正。</li><li>受控置换保留长度×熵分箱中的正误数量；更强对照另加题型与难度。粗分箱不能完全排除混杂。token 类型关联另列，不声称已排除全部词汇解释。</li><li>当前 MATH 已被多次探索；本轮新聚类不使用测试题目，但不是全新确认数据。相关性不等于因果。</li></ul><p><a href="../evaluation/outcome_association.csv">正误关联 CSV</a> · <a href="../evaluation/cluster_profiles.csv">簇描述 CSV</a> · <a href="../evaluation/token_kind_counts.csv">token 类型 CSV</a> · <a href="../protocol.json">冻结协议</a> · <a href="https://scikit-learn.org/stable/modules/mixture.html">GMM 方法说明</a></p></section>
<script>
let DATA,QUESTIONS;const el=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=(x,n=3)=>x==null?'—':Number(x).toFixed(n);
function samples(){let v=DATA.views[el('view').value],c=el('cluster').value;el('examples').innerHTML=(v.examples[c]||[]).map(e=>{let q=QUESTIONS[e.question_index];return `<details class="card"><summary>${q.label?'正确':'错误'} · ${esc(q.category)} · ${q.n_tokens} tokens · ${esc(q.sample_id)}</summary><p>${e.position?`相邻生成 token ${e.position} → ${e.position+1}：<code>${esc(e.piece_before)}</code> → <code>${esc(e.piece_after)}</code>`:''}</p><p>Mahalanobis² = ${num(e.mahalanobis_sq)} · ${esc(q.split)}</p><strong>题目</strong><pre>${esc(q.question)}</pre><strong>模型回答</strong><pre>${esc(q.response_text)}</pre><strong>参考答案</strong><pre>${esc(q.ground_truth)}</pre></details>`}).join('')||'<p>该簇没有测试题目。</p>'}
function show(){let v=DATA.views[el('view').value],a=v.association;let ari=v.stability.filter(s=>s.kind==='80pct_questions'&&s.converged).map(s=>s.ari);el('metrics').innerHTML=`<p><span class="badge">ICL K=${v.k}${v.k_boundary?'（搜索边界）':''}</span><span class="badge">验证 LL K=${v.k_validation}</span><span class="badge">ΔLL/d=${num(v.test_gain_nats_per_dimension,4)}</span><span class="badge">子样本 ARI=${ari.map(x=>num(x)).join(' / ')}</span></p><p>正误占比 JS=${num(a.js_bits,4)} bits · q=${num(a.q)} · 长度+熵控制 q=${num(a.controlled_q)} · 再加题型+难度 q=${num(a.task_controlled_q)}</p>${a.token_kind_nmi==null?'':`<p>簇与 token 类型转换 NMI=${num(a.token_kind_nmi)}；与生成相对位置 NMI=${num(a.position_nmi)}</p>`}`;el('plot').src=`figures/${v.name}.png`;el('note').textContent=v.note+' Silhouette='+num(v.silhouette_test);el('profiles').innerHTML='<tr><th>簇</th><th>测试题目权重</th><th>加权正确率</th><th>回答长度</th><th>Token 熵</th><th>中心范数</th><th>方差总和</th></tr>'+v.profiles.map(p=>`<tr><td>${p.cluster}</td><td>${num(p.test_question_mass,1)}</td><td>${num(p.test_correct_rate==null?null:p.test_correct_rate*100,1)}%</td><td>${num(p.weighted_mean_length,0)}</td><td>${num(p.weighted_mean_entropy)}</td><td>${num(p.center_norm,1)}</td><td>${num(p.diagonal_variance_sum,1)}</td></tr>`).join('');el('cluster').innerHTML=v.profiles.map(p=>`<option value="${p.cluster}">Cluster ${p.cluster}</option>`).join('');samples()}
Promise.all([fetch('data.json').then(r=>r.json()),fetch('questions.json').then(r=>r.json())]).then(([d,q])=>{DATA=d;QUESTIONS=q;el('view').innerHTML=d.views.map((v,i)=>`<option value="${i}">${v.name}</option>`).join('');el('view').onchange=show;el('cluster').onchange=samples;el('nulls').innerHTML='<table><tr><th>层</th><th>真实更新 K</th><th>单高斯 K</th><th>真实 ΔLL/d</th><th>单高斯 ΔLL/d</th></tr>'+d.nulls.map(n=>{let r=d.views.find(v=>v.name===n.name.replace('gaussian','mean'));return `<tr><td>${n.name}</td><td>${r.k}</td><td>${n.k}</td><td>${num(r.test_gain_nats_per_dimension,4)}</td><td>${num(n.test_gain_nats_per_dimension,4)}</td></tr>`}).join('')+'</table>';show()}).catch(e=>el('metrics').textContent='加载失败：'+e);
</script></html>'''
    (dest/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/change-clusters.toml');p.add_argument('--render-only',action='store_true');a=p.parse_args()
    cfg=load_config(a.config)
    with threadpool_limits(cfg['cpu_threads']):
        render(cfg) if a.render_only else evaluate(cfg)

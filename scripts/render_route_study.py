"""Render frozen route results and a per-question reader; no fitting."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hss.experiments.artifacts import file_digest,save_json


COLORS={'gmm':'#326d9f','mfa':'#c16b30','gmm_matched':'#28836a'}
NAMES={'gmm':'GMM native K','mfa':'MFA rank16','gmm_matched':'GMM at MFA K'}


def plot_save(fig,root,name):
    fig.savefig(root/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def render(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    a=root/'analysis';c=root/'controls';inp=root/'inputs'
    summary=json.loads((a/'summary.json').read_text());ctrl=json.loads((c/'summary.json').read_text())
    meta=pd.read_parquet(inp/'rows.parquet');states=dict(np.load(inp/'states.npz'));states['gmm_matched']=np.load(c/'matched_states.npz')['states']
    geometry=dict(np.load(inp/'geometry.npz'))
    main=pd.read_csv(a/'conditional_routing.csv');extra=pd.read_csv(c/'conditional_controls.csv')
    metrics=pd.concat([pd.read_csv(a/'score_evaluation.csv'),pd.read_csv(c/'score_evaluation.csv').query("method=='gmm_matched'")],ignore_index=True)
    scores=pd.read_parquet(a/'unlabeled_scores.parquet');sc=pd.read_parquet(c/'unlabeled_scores.parquet')
    for col in sc.columns:
        if col.startswith('gmm_matched__'):scores[col]=sc[col]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharex=True)
    for method in ['gmm','mfa','gmm_matched']:
        for j,control in enumerate(['full','geometry']):
            p=extra[(extra.method==method)&(extra.control==control)].sort_values('layer_from')
            axes[0,j].plot(p.layer_from,p.excess_bits,color=COLORS[method],label=NAMES[method])
            axes[1,j].plot(p.layer_from,p.support_fraction,color=COLORS[method])
            sig=p.fdr_q<.05;axes[0,j].scatter(p.layer_from[sig],p.excess_bits[sig],s=30,color=COLORS[method])
        axes[0,0].legend(frameon=False,fontsize=9)
    for j,title in enumerate(['Task + length + prompt + finish','Task + length + RMS + Mahalanobis']):
        axes[0,j].set(title=title,ylabel='Conditional JS minus label-null mean (bits)');axes[0,j].axhline(0,color='#888',lw=.7)
        axes[1,j].set(xlabel='Origin decoder layer',ylabel='Fraction in common-support groups',ylim=(0,1))
    fig.suptitle('Frozen-map conditional routing | dots: BH q < .05 within each control family',fontsize=12)
    plot_save(fig,out,'conditional_routes')
    chosen=[('baseline','length'),('baseline','mean_token_entropy'),('mfa','final_rms'),('gmm','context'),('gmm_matched','context'),('mfa','context'),('mfa','node'),('mfa','soft_js')]
    labels=['Response length','Mean token entropy','Final hidden-mean RMS','GMM route context','Same-K GMM route context','MFA route context','MFA node rarity','MFA soft route JS']
    sub=pd.concat([metrics[(metrics.method==m)&(metrics.score==s)] for m,s in chosen])
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');yy=np.arange(len(sub))
    ax.errorbar(sub.failure_auroc,yy-.1,xerr=[sub.failure_auroc-sub.ci_low,sub.ci_high-sub.failure_auroc],fmt='o',color='#326d9f',capsize=3,label='Overall AUROC, conditional bootstrap 95% CI')
    ax.scatter(sub.conditional_auc,yy+.13,marker='x',s=45,color='#c16b30',label='Within full-control strata, descriptive AUROC')
    ax.axvline(.5,color='#777',ls='--');ax.set(yticks=yy,yticklabels=labels,xlabel='Failure AUROC | higher score predeclared as greater risk',xlim=(.35,.83));ax.invert_yaxis();ax.legend(frameon=False,fontsize=8,loc='lower right')
    plot_save(fig,out,'score_comparison')
    memory=ctrl['memory'];fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    names=['gmm','gmm_matched','mfa'];xx=np.arange(3)
    axes[0].bar(xx-.18,[memory[k]['observed_gain_nats'] for k in names],.34,label='Observed',color='#326d9f')
    axes[0].bar(xx+.18,[memory[k]['null_mean_nats'] for k in names],.34,label='Edge-preserving suffix null',color='#abb9c2')
    axes[0].set(xticks=xx,xticklabels=[NAMES[k] for k in names],ylabel='Next-state log-loss gain (nats)',title='Does the previous state add predictive information?');axes[0].legend(frameon=False,fontsize=8)
    nulls=ctrl['corrected_occupancy_nulls']
    axes[1].bar(xx-.18,[nulls[k]['bias_corrected']['observed_bits'] for k in names],.34,label='Observed',color='#326d9f')
    axes[1].bar(xx+.18,[nulls[k]['bias_corrected']['null_mean_bits'] for k in names],.34,label='Class/stratum occupancy null',color='#abb9c2')
    axes[1].set(xticks=xx,xticklabels=[NAMES[k] for k in names],ylabel='Bias-corrected conditional JS (bits)',title='Do real routes increase class separation?');axes[1].legend(frameon=False,fontsize=8)
    plot_save(fig,out,'structure_nulls')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    sensitivity=pd.read_csv(a/'sensitivity.csv')
    for method in ['mfa_seed1042','mfa_seed2042','mfa_nearest']:
        p=sensitivity[sensitivity.method==method].sort_values('layer');axes[0].plot(p.layer,p.ari_vs_reference,label=method)
    axes[0].set(xlabel='Decoder layer',ylabel='ARI vs frozen MFA',ylim=(0,1.02),title='Membership sensitivity');axes[0].legend(frameon=False,fontsize=8)
    for method in ['mfa','mfa_seed1042','mfa_seed2042','mfa_nearest','mfa_rank4_own_k','mfa_rank8_own_k']:
        p=metrics[(metrics.method==method)&metrics.score.isin(['node','context'])].set_index('score')
        axes[1].plot([0,1],[p.loc['node','failure_auroc'],p.loc['context','failure_auroc']],'o-',label=method)
    axes[1].axhline(.5,color='#999',ls='--');axes[1].set(xticks=[0,1],xticklabels=['Node rarity','Route context'],ylabel='Failure AUROC',title='Scores under seeds / assignment / rank-specific K');axes[1].legend(frameon=False,fontsize=7)
    plot_save(fig,out,'sensitivity')
    # A fixed layer pair (middle and final) avoids selecting a picture by labels.
    edges=pd.read_parquet(a/'edges.parquet');fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,layer in enumerate([14,27]):
        part=edges[(edges.method=='mfa')&(edges.layer_from==layer)]
        for col,key in enumerate(['p_correct','p_incorrect','delta']):
            mat=part.pivot(index='origin',columns='destination',values=key)
            im=axes[row,col].imshow(mat,vmin=-.5 if col==2 else 0,vmax=.5 if col==2 else 1,cmap='RdBu_r' if col==2 else 'Blues',aspect='auto')
            axes[row,col].set(title=f'MFA L{layer} → L{layer+1}: '+['correct','incorrect','incorrect − correct'][col],xlabel='Destination local cluster',ylabel='Origin local cluster');fig.colorbar(im,ax=axes[row,col],shrink=.7)
    plot_save(fig,out,'transition_examples')
    # JSON is separate: it is loaded once to inspect any of the 5000 questions.
    samples=[]
    for i,r in meta.iterrows():
        samples.append(dict(id=r.sample_id,correct=int(r.label),category=r.category,level=int(r.level),tokens=int(r.n_tokens),
            prompt=r.prompt_text,response=r.response_text,gold=str(r.ground_truth),
            states={m:states[m][i].astype(int).tolist() for m in ['gmm','mfa','gmm_matched']},
            global_ids={m:states[m+'_global'][i].astype(int).tolist() for m in ['gmm','mfa']},
            scores={m:{k:float(scores.loc[i,m+'__'+k]) for k in ['node','edge','context']} for m in ['gmm','mfa','gmm_matched']},
            geometry={m:{k:[float(geometry[f'{m}_{l}_{k}'][i]) for l in range(1,29)] for k in ['rms','mahal']} for m in ['gmm','mfa']}))
    (out/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,separators=(',',':')))
    e=edges.replace([np.inf,-np.inf],np.nan);(out/'edges.json').write_text(e.to_json(orient='records'))
    stat=extra.groupby(['method','control']).agg(mean_excess=('excess_bits','mean'),support=('support_fraction','mean'),significant=('fdr_q',lambda x:int((x<.05).sum())))
    stathtml=stat.reset_index().rename(columns={'method':'方法','control':'控制变量','mean_excess':'平均校正 JS','support':'可比较样本比例','significant':'q<.05 的相邻层数'}).to_html(index=False,float_format=lambda x:f'{x:.4f}',classes='numbers',border=0)
    scorehtml=sub[['method','score','failure_auroc','ci_low','ci_high','conditional_auc','risk_at_50']].rename(columns={'method':'方法','score':'分数','failure_auroc':'失败 AUROC','ci_low':'95% 下界','ci_high':'95% 上界','conditional_auc':'分层内 AUROC','risk_at_50':'保留一半的错误率'}).to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    facts=[]
    for m in names:
        v=memory[m];n=nulls[m]['bias_corrected']
        facts.append(f'<tr><td>{NAMES[m]}</td><td>{v["observed_gain_nats"]:.4f}</td><td>{v["null_mean_nats"]:.4f}</td><td>{v["p"]:.3f}</td><td>{n["observed_bits"]:.4f}</td><td>{n["null_mean_bits"]:.4f}</td><td>{n["p_greater"]:.3f}</td></tr>')
    payload=dict(summary=summary,controls_summary=ctrl,metrics=metrics.to_dict('records'),grouped_controls=stat.reset_index().to_dict('records'))
    save_json(out/'results.json',payload)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>状态路由：第一阶段结果</title>
<style>body{margin:0;background:#f2f5f7;color:#20313e;font:16px/1.7 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:auto;padding:30px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:24px;margin-top:0}section{background:white;padding:26px;border:1px solid #dbe3e8;border-radius:12px;margin:22px 0}p{max-width:100ch}.note{background:#fff4db;border-left:4px solid #c79031;padding:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{max-width:100%;height:auto}a{color:#24668f}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px 9px;border-bottom:1px solid #dce4e8;white-space:nowrap;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}select,input,button{font:inherit;padding:7px;border:1px solid #aebec8;border-radius:5px;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto;background:#f4f7f8;padding:16px;font:14px/1.65 ui-monospace,monospace}.path{display:flex;gap:4px;flex-wrap:wrap}.state{background:#e8f0f6;padding:5px 7px;font-size:12px;text-align:center;border-radius:4px}.state b{display:block;font-size:16px}small{color:#5a7182}nav a{margin-right:14px}.bar{height:9px;display:inline-block;background:#326d9f}.orange{background:#c16b30}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:15px}section{padding:16px}h1{font-size:28px}}</style>
<main><small>Qwen2-7B-Instruct · MATH 5,000 · 原始 hidden mean · 28 个 decoder 层</small><h1>状态路由：第一阶段结果</h1><p>2,378 个正确回答，2,622 个错误回答。保留已有 cosine/Hungarian η=0.6 matching；新增同 K 的 GMM 作为协方差家族对照。所有风险分数均未用正确性标签拟合。</p><nav><a href="#structure">结构检验</a><a href="#prediction">失败评分</a><a href="#controls">稳定性</a><a href="#flows">条件分流</a><a href="#reader">逐题原文与路径</a></nav>
<p class="note">探索性结果：聚类和 K/rank 此前使用了全部题目，仅转移计数做了五折隔离。完整回答的跨层路径不是推理步骤的时间轴。这份结果不证明因果机制、提前预警或独立测试泛化。</p>
<section><h2>先看结论</h2><ol><li><b>路径确实有高阶结构。</b>知道前一层的来路，可以改善下一层状态预测；这种增益超过保留全部相邻边计数的打乱对照。但它不自动代表正确性或推理记忆。</li><li><b>部分正确／错误路由差异仍在，但位置不稳定。</b>MFA 主种子有 7 个相邻层通过当前控制后的 FDR 检查；两个独立种子也各有 7 个，三者只有第 25→26 层共同出现。它只是层位置重复，并未证明同一条语义路径。</li><li><b>“异常路线预示错误”目前不成立。</b>路线 context AUROC：原 GMM 0.566、同 K GMM 0.416、MFA 0.437。仅回答长度为 0.775；回答平均 token 熵为 0.689。</li><li><b>聚类粒度影响很大。</b>同 K 后 GMM 分数方向接近 MFA。真实类别分流也没有超过保留类别占用的打乱对照，因此尚未建立独立的失败路由机制。</li></ol></section>
<section id="structure"><h2>1. 从同一区域出发，正确与错误的去向是否不同？</h2><p>比较同一起点及 nuisance 分层中的下一层分布。先减去标签置换产生的平均 JS，避免把有限样本偏差当成差异。每组至少需要两个正确和两个错误样本；覆盖不足的样本不参与该条件比较。</p><img src="conditional_routes.png" alt="条件转移差异与共同支持样本比例"><div class="scroll">__STAT__</div><p><small>full：题型×难度×回答长度四分位×prompt 长度三分位×结束原因；geometry：题型×难度×回答长度四分位×当前 RMS 二分位×当前 Mahalanobis 距离二分位；entropy：full 再加回答平均 token 熵二分位。分箱无法完全排除连续混杂。各行 significant 是该控制家族内 BH q&lt;.05 的相邻层数，仍属于探索性定位。</small></p></section>
<section><h2>2. 真实路径相对于两种打乱对照</h2><img src="structure_nulls.png" alt="二阶预测增益与占用保留对照"><div class="scroll"><table><tr><th>方法</th><th>二阶增益</th><th>保留边的 null</th><th>p</th><th>真实校正 JS</th><th>保留占用的 null</th><th>p（更大）</th></tr>__NULL__</table></div><p><b>左图：</b>在相同中间状态处交换整条后缀，逐折、逐层的相邻边计数完全不变。它检验“两步历史是否比当前状态更能预测下一层”，不用正确性标签。199 次打乱的最小可报告 p 为 0.005；三种方法的 Bonferroni 校正值均为 0.015。原 GMM 的 null 本身也有较大正增益，说明不能把未经对照的原始 log-loss 增益全部解释为高阶结构。即使有记忆，也可能来自未建模的题目属性，不能直接称为模型的推理记忆。</p><p><b>右图：</b>在正确性和 nuisance 组内分别打乱各层，保留每层类别占用，破坏跨层配对。原始值及每张打乱图都减去各自精确的标签置换期望，校正占用单元数量变化。真实分歧小于 null 与真实连接中的类别信息更冗余相符，不能据此断言路径没有任何信息。两种 null 的问题不同。</p></section>
<section id="prediction"><h2>3. 有结构，不等于异常分数能预测失败</h2><img src="score_comparison.png" alt="失败 AUROC 及控制分层后的 AUROC"><div class="scroll">__SCORES__</div><p>所有方向在评分前固定为“越大越可能失败”，低于 0.5 的结果没有反转。区间是固定模型/分数下 400 次题目 bootstrap，不包含重新聚类的不确定性。条件 AUROC 只比较 full 控制分层内部的正负样本，按可比较样本对数量加权；它不是重新训练的分类器，也不能与全体 AUROC 当作同一个估计目标。</p><p>risk_at_50：按该分数保留风险最低的一半回答，剩余错误率。baseline mean_token_entropy 是完整回答平均 token 熵，不是第一个 token 的熵。</p></section>
<section id="controls"><h2>4. 协方差、种子与末层归一化</h2><img src="sensitivity.png" alt="成员归属和风险分数的敏感性"><p>同 K 对照覆盖全部 28 层，每层 3 次 GMM 初始化，全部模型和初始化参数均已保存。它同时改变协方差家族和重新估计的中心，不能当作纯协方差因果干预。rank4/8 使用各自选出的 K，不是固定 K 的 rank 消融。</p><p>后验几乎确定不意味着模型解稳定；最近中心、初始化和协方差假设仍会改变归属。pre-RMSNorm 仅替换最后一个位置，是单独的敏感性结果。</p></section>
<section id="flows"><h2>5. 查看同一起点的下一步分流</h2><p>这里是原始描述性概率，没有做 nuisance 控制；统计控制结果见上方。局部编号只在该层内部有意义。</p><label>方法 <select id="flowMethod"><option value="mfa">MFA</option><option value="gmm">GMM</option></select></label> <label>起点层 <select id="flowLayer"></select></label> <label>起点簇 <select id="flowOrigin"></select></label><div class="scroll" id="flowTable">正在加载…</div><details><summary>固定中层与末层转移矩阵</summary><img src="transition_examples.png" alt="第14及27层的 MFA 条件转移矩阵"></details></section>
<section id="reader"><h2>6. 逐题原文、状态路径与几何量</h2><p>按预先固定的无标签 context 分数浏览所有题目；正确性只供事后检查。全局 ID 是当前几何 matching 的注释，不保证同一语义。</p><label>方法 <select id="sampleMethod"><option value="mfa">MFA</option><option value="gmm">GMM</option><option value="gmm_matched">同 K GMM</option></select></label> <label>排序 <select id="sort"><option value="id">题目顺序</option><option value="high">context 从高到低</option><option value="low">context 从低到高</option></select></label><p><input id="search" placeholder="搜索题号或题目文字"><select id="sample"></select></p><div id="sampleInfo"></div><div class="path" id="path"></div><details><summary>逐层局部 ID / 全局 ID / RMS / Mahalanobis</summary><div class="scroll" id="geometry"></div></details><div class="grid"><div><h3>原题与 prompt</h3><pre id="prompt"></pre></div><div><h3>模型回答</h3><pre id="response"></pre></div></div></section>
<section><h2>范围和复现</h2><p>已完成：P1 全层条件转移与占用保留对照；P2 全层同 K GMM、nearest/soft、两个现有种子、rank 各自 K、末层 pre/post、几何与熵分层、保留一阶边的高阶检验。</p><p>待完成：独立训练集建图的 P3、固定 K 的 MFA rank 消融、拟合样本重采样、Gaussian 参数模拟、跨任务/模型转移，以及真正的 prompt/prefix 提前预警和因果干预。</p><p><a href="results.json">结果与限制 JSON</a> · <a href="../analysis/conditional_routing.csv">逐层主检验</a> · <a href="../controls/conditional_controls.csv">逐层控制检验</a> · <a href="../analysis/score_evaluation.csv">完整评分表</a> · <a href="../analysis/protocol.json">P1 冻结协议</a> · <a href="../controls/protocol.json">P2 协议与偏差校正说明</a></p><p>图同时提供同名 PDF。改图不触发重新拟合。</p></section></main>
<script>
let S=[],E=[];const $=id=>document.getElementById(id),fmt=x=>x==null?'—':Number(x).toFixed(3);
function cell(text){const d=document.createElement('td');d.textContent=text;return d}
function table(head,rows){const t=document.createElement('table');const h=document.createElement('tr');head.forEach(v=>{let q=document.createElement('th');q.textContent=v;h.append(q)});t.append(h);rows.forEach(row=>{let tr=document.createElement('tr');row.forEach(v=>tr.append(cell(v)));t.append(tr)});return t}
function flows(reset=true){let m=$('flowMethod').value,l=+$('flowLayer').value;let e=E.filter(x=>x.method===m&&x.layer_from===l);if(reset){$('flowOrigin').replaceChildren();[...new Set(e.map(x=>x.origin))].sort((a,b)=>a-b).forEach(v=>$('flowOrigin').add(new Option(v,v)))}let a=+$('flowOrigin').value;e=e.filter(x=>x.origin===a);$('flowTable').replaceChildren(table(['目的簇','总人数','正确组转移人数/起点人数','错误组转移人数/起点人数','P(目的|起点,正确)','P(目的|起点,错误)','错误 − 正确'],e.map(x=>[x.destination,x.n,`${x.correct}/${x.origin_correct}`,`${x.incorrect}/${x.origin_incorrect}`,fmt(x.p_correct),fmt(x.p_incorrect),fmt(x.delta)])))}
function sampleList(){let m=$('sampleMethod').value,q=$('search').value.toLowerCase(),sort=$('sort').value;let arr=S.map((s,i)=>[s,i]).filter(([s])=>s.id.includes(q)||s.prompt.toLowerCase().includes(q));if(sort!=='id')arr.sort((a,b)=>(a[0].scores[m].context-b[0].scores[m].context)*(sort==='high'?-1:1));$('sample').replaceChildren();arr.forEach(([s,i])=>$('sample').add(new Option(`${s.id} · ${s.tokens} tokens · context ${fmt(s.scores[m].context)}`,i)));showSample()}
function showSample(){if(!$('sample').value)return;let s=S[+$('sample').value],m=$('sampleMethod').value;$('sampleInfo').textContent=`${s.id} · ${s.correct?'正确':'错误'} · ${s.category} / level ${s.level} · ${s.tokens} tokens · gold: ${s.gold} · node ${fmt(s.scores[m].node)} / edge ${fmt(s.scores[m].edge)} / context ${fmt(s.scores[m].context)}`;$('prompt').textContent=s.prompt;$('response').textContent=s.response;$('path').replaceChildren();s.states[m].forEach((c,l)=>{let d=document.createElement('div');d.className='state';let small=document.createElement('small');small.textContent='L'+(l+1);let b=document.createElement('b');b.textContent=c;d.append(small,b);$('path').append(d)});let g=s.geometry[m];$('geometry').replaceChildren(table(['层','局部 ID','全局 ID','RMS','Mahalanobis² / D'],s.states[m].map((c,l)=>[l+1,c,s.global_ids[m]?.[l]??'未做关联',g?fmt(g.rms[l]):'—',g?fmt(g.mahal[l]):'—'])))}
for(let l=1;l<28;l++)$('flowLayer').add(new Option(`${l} → ${l+1}`,l));$('flowLayer').value='14';$('flowMethod').onchange=()=>flows();$('flowLayer').onchange=()=>flows();$('flowOrigin').onchange=()=>flows(false);$('sampleMethod').onchange=sampleList;$('sort').onchange=sampleList;$('search').oninput=sampleList;$('sample').onchange=showSample;
Promise.all([fetch('samples.json').then(r=>r.json()),fetch('edges.json').then(r=>r.json())]).then(([s,e])=>{S=s;E=e;sampleList();flows()}).catch(e=>{$('flowTable').textContent='数据加载失败：'+e});
</script></html>'''
    (out/'index.html').write_text(page.replace('__STAT__',stathtml).replace('__NULL__',''.join(facts)).replace('__SCORES__',scorehtml))
    save_json(out/'render_manifest.json',dict(driver_sha256=file_digest(Path(__file__)),analysis_summary=file_digest(a/'summary.json'),controls_summary=file_digest(c/'summary.json'),files={f.name:file_digest(f) for f in out.iterdir() if f.is_file() and f.name!='render_manifest.json'}))
    print(out/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();render(a.root)

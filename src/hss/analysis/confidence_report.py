"""Export scientific figures and an inspectable Chinese confidence follow-up."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hss.analysis.channel_data import load_config
from hss.analysis.channel_report import STYLE,save,table_html
from hss.experiments.artifacts import save_json,file_digest


LABELS = {'entropy':'Entropy','logit_margin':'Top-1 / top-2 logit margin',
    'top1_probability':'Top-1 probability','rms':'Pre-RMS vector RMS',
    'v_min_projection':'Projection on weakest-readout v',
    'low_readout_fraction':'Bottom 1% subspace norm fraction',
    'v_min_absolute':'Absolute weakest-readout projection',
    'full_direction':'Full-dimensional L2 probe','channels16':'Historical 16 channels'}
DATA_NAMES = {'llama32_math':'Llama-3.2-1B · MATH','qwen2_math':'Qwen2-7B · MATH',
              'llama32_mmlu':'Llama-3.2-1B · MMLU'}


def forest(ax,records,labels,chance=.5,color='#287e93'):
    for i,r in enumerate(records):
        ax.plot(r['ci'],[i,i],color=color,lw=2)
        ax.scatter(r['auc'],i,color=color,s=32,zorder=3)
        ax.text(r['ci'][1]+.005,i,f"{r['auc']:.3f}",va='center',fontsize=8)
    ax.axvline(chance,color='#8993a0',ls='--',lw=1)
    ax.set(yticks=range(len(records)),yticklabels=labels,xlabel='Held-out AUROC (95% CI)')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.15)


def image(name,filename,caption):
    return f'<figure><a href="{name}/{filename}"><img loading="lazy" src="{name}/{filename}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def examples(cfg,ds,out):
    s=pd.read_parquet(out/'scalars.parquet'); rows=pd.read_parquet(out/'samples.parquet')
    p=pd.read_parquet(out/'predictions.parquet'); assert np.array_equal(s.sample_id,rows.sample_id)
    te=np.flatnonzero(rows.partition.eq('confirmation'))
    wanted=[]
    for label in [0,1]:
        take=te[rows.y.to_numpy()[te]==label]
        order=take[np.argsort(s.prompt_last_entropy.to_numpy()[take])]
        wanted.extend(order[:5]);wanted.extend(order[-5:])
    selected=rows.iloc[wanted].copy()
    for col in ['prompt_last_entropy','prompt_last_logit_margin','prompt_last_v_min_projection']:
        selected[col]=s[col].iloc[wanted].to_numpy()
    selected['full_direction_score']=p.pre_prompt_last__full_direction.iloc[wanted].to_numpy()
    lookup={}
    for shard in selected.shard.unique():
        frame=pd.read_parquet(Path(ds['path'])/shard/'data.parquet',columns=['sample_id','prompt_text','response_text'])
        lookup.update({r['sample_id']:r for r in frame.to_dict('records')})
    parts=['<h1>按首 token 熵选择的验证题实例</h1><p>每类选熵最低和最高的各五题；这是有意选出的极端案例，不能用其比例估计准确率。</p><a href="../index.html">返回报告</a>']
    for r in selected.to_dict('records'):
        text=lookup[r['sample_id']]
        title=f"{r['sample_id']} | {'正确' if r['y'] else '错误'} | H={r['prompt_last_entropy']:.3f} | margin={r['prompt_last_logit_margin']:.3f} | w·h+b={r['full_direction_score']:.3f}"
        parts.append(f'<details><summary>{html.escape(title)}</summary><pre>{html.escape(text["prompt_text"])}</pre><pre>{html.escape(text["response_text"])}</pre></details>')
    (out/'examples.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1050px;margin:35px auto;font:16px/1.7 system-ui}details{padding:15px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}summary{cursor:pointer}</style>'+''.join(parts))


def render(cfg):
    root=Path(cfg['output_root']);cc=load_config(cfg['channel_config'])
    parts=[];heads=[];files=[]
    with plt.rc_context(STYLE):
        for ds in cc['datasets']:
            name=ds['name'];out=root/name
            a=json.loads((out/'analysis.json').read_text()); meta=json.loads((out/'scalars.json').read_text())
            scalar=lambda metric:next(r for r in a['scalars'] if r['position']=='prompt_last' and r['metric']==metric)
            v=a['views']['pre_prompt_last']; models=v['models']
            delta=next(r for r in v['incremental'] if r['augmented']=='nuisance_entropy_channels')
            heads.append({'模型 × 数据':DATA_NAMES[name],'验证题':a['test_n'],'熵':scalar('entropy')['auc'],
                '16 通道':models['channels16']['auc'],'全维方向':models['full_direction']['auc'],
                'nuisance + 熵':models['nuisance_entropy']['auc'],
                '再加16通道增益':f"{delta['delta']:+.3f} [{delta['ci'][0]:+.3f}, {delta['ci'][1]:+.3f}]"})
            parts.append(f'<section id="{name}"><h2>{DATA_NAMES[name]}</h2><p>分析 {a["n"]:,} 题，发现 {a["train_n"]:,} / 验证 {a["test_n"]:,}。首 token FP32 重建 argmax 一致率 {meta["first_token_argmax_agreement"]:.2%}；不一致 {meta["mismatch_n"]} 题。<a href="{name}/scalars.json">精度与模型记录</a> · <a href="{name}/examples.html">查看题目和原始回答</a></p>')
            pp=out/'precision.json'
            if pp.exists():
                precision=json.loads(pp.read_text())
                entropy=next(r for r in precision['scalars'] if r['metric']=='policy_entropy')
                parts.append(f'<p><strong>精度／解码策略审计：</strong>采用 BF16 舍入及原生 repetition penalty={precision["effective_repetition_penalty"]:g} 后，首 token 一致率 {precision["agreement"]["policy_argmax"]:.2%}；相应熵 AUROC={entropy["auc"]:.3f}。Teacher-forced 状态与 generate prefill 仍有数值差异，原始生成首步 logits 未保存，不能声称完全重放原始分布。<a href="{name}/precision.json">完整敏感性结果</a></p>')
            fig,axs=plt.subplots(1,2,figsize=(13,4.6),layout='constrained',sharex=True)
            for ax,pos in zip(axs,['prompt_last','t1']):
                records=[r for metric in LABELS for r in a['scalars'] if r['position']==pos and r['metric']==metric]
                forest(ax,records,[LABELS[r['metric']] for r in records])
                ax.set_title('Prompt-last → first token' if pos=='prompt_last' else 'After token 1 → second token')
            filename=save(fig,out,'scalars');files.append(out/filename)
            parts.append(image(name,filename,'每个标量的符号仅由发现集确定；原始符号 AUROC 同时存于表格。首 token 熵与首个 token 读入后的熵是两个不同预测位置。v_min 只是新构造的候选。'))
            fig,axs=plt.subplots(1,2,figsize=(13,4.4),layout='constrained',sharex=True)
            for ax,view in zip(axs,['pre_prompt_last','pre_t1']):
                pairs=a['views'][view]['incremental']
                labels=[]
                for i,r in enumerate(pairs):
                    ax.plot(r['ci'],[i,i],color='#287e93',lw=2);ax.scatter(r['delta'],i,color='#287e93')
                    labels.append({'nuisance_entropy':'Add first-token entropy',
                        'nuisance_entropy_channels':'Then add 16 channels',
                        'diagnostic_nuisance_entropy_channels':'16 channels / diagnostic controls',
                        'nuisance_confidence_channels':'16 channels / all confidence scalars',
                        'nuisance_confidence_full':'Full residual / all confidence scalars',
                        'nuisance_current_entropy_channels':'16 channels / second-token entropy'}[r['augmented']])
                ax.axvspan(-cfg['equivalence_auc'],cfg['equivalence_auc'],color='#287e93',alpha=.08)
                ax.axvline(0,color='#888',ls='--');ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Paired AUROC increment (95% CI)',title=view)
                ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
            filename=save(fig,out,'incremental');files.append(out/filename)
            parts.append(image(name,filename,'浅色带为预定的 ±0.01 AUROC 实际等价范围。跨 0 但超出等价范围表示不确定。所有增量使用配对题目 bootstrap。诊断控制额外包含首 token 身份。'))
            geometry=np.load(out/'readout_geometry.npz');sv=geometry['singular_values']
            fig,axs=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
            axs[0].semilogy(np.arange(len(sv)),np.maximum(sv,1e-15),color='#287e93')
            axs[0].set(xlabel='Singular index (smallest first)',ylabel='Singular value',title='Centered, RMS-weight-folded head')
            axs[0].text(.03,.95,f'Minimum = {sv[0]:.3f}\nSecond / minimum = {sv[1]/sv[0]:.2f}',transform=axs[0].transAxes,va='top',fontsize=9)
            g=v['geometry'];keys=['cosine_v_min','cosine_entropy_regression']+(['cosine_difficulty'] if 'cosine_difficulty' in g else [])
            labels=['Weakest-readout candidate','Entropy regression']+(['Difficulty regression'] if 'cosine_difficulty' in g else [])
            axs[1].barh(range(len(keys)),[g[k] for k in keys],color='#287e93')
            axs[1].set(yticks=range(len(keys)),yticklabels=labels,xlim=(-1,1),xlabel='Cosine in raw pre-RMS coordinates',title='Correctness direction w')
            axs[1].axvline(0,color='#aaa');axs[1].invert_yaxis()
            filename=save(fig,out,'geometry');files.append(out/filename)
            parts.append(image(name,filename,f"难度方向以更高 level 为正；因此负余弦也可表示对齐。题型为多类，w 落在题型系数子空间的范数比例为 {g['category_subspace_w_fraction']:.3f}。谱上低值并不自动证明置信度功能。"))
            ap=out/'audit.json'
            if ap.exists():
                audit=json.loads(ap.read_text())
                correlations=audit['pre_prompt_last']['projection_correlations']['correctness_w']
                parts.append('<p>原始坐标角度与样本分布上的读出相关性不同。以下给出验证题的 Pearson 相关；它们仍是关联指标。</p>'+table_html(pd.DataFrame([correlations]))+f'<p><a href="{name}/audit.json">身份、方向重建及精度子集独立核验</a></p>')
                energy=audit['pre_prompt_last'].get('weakest_direction_energy')
                if energy:
                    parts.append(f'<p>v 承载 prompt-last RMS-only 状态总能量的 <strong>{energy["rms_only"]["total_energy_fraction"]:.2%}</strong>，承载中心化变化能量的 {energy["rms_only"]["centered_energy_fraction"]:.2%}。有一个弱方向，不等于本任务在该方向储存大量状态能量。</p>')
            s=pd.read_parquet(out/'scalars.parquet');rows=pd.read_parquet(out/'samples.parquet')
            tr=rows.partition.eq('discovery');te=rows.partition.eq('confirmation')
            fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
            boundaries=np.unique(np.quantile(s.loc[tr,'prompt_last_entropy'],np.linspace(0,1,11)))
            boundaries[0]=-np.inf;boundaries[-1]=np.inf
            bins=np.digitize(s.loc[te,'prompt_last_entropy'],boundaries[1:-1]); yy=rows.loc[te,'y'].to_numpy()
            centers=[];accuracy=[];error=[]
            for b in sorted(set(bins)):
                take=bins==b;rate=yy[take].mean();n=take.sum()
                centers.append(s.loc[te,'prompt_last_entropy'].to_numpy()[take].mean());accuracy.append(rate);error.append(1.96*np.sqrt(rate*(1-rate)/n))
            ax.errorbar(centers,accuracy,yerr=error,fmt='o-',capsize=3,color='#287e93')
            ax.set(xlabel='First-token entropy (nats)',ylabel='Answer accuracy',title='Discovery-defined entropy bins; held-out questions')
            filename=save(fig,out,'entropy_accuracy');files.append(out/filename)
            parts.append(image(name,filename,'分桶边界仅由发现集确定。该图检查熵与整题正确率的关系是否单调；首 token 的输出确定性不必等于答案确定性。'))
            lp=out/'layers.json'
            if lp.exists():
                lr=json.loads(lp.read_text());fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained',sharey=True)
                for ax,pos in zip(axs,['prompt_last','t16']):
                    for method,color in [('frozen_prompt_last','#287e93'),('frozen_t1','#cf744b'),('layer_specific_probe','#6954a1')]:
                        records=[r for r in lr['records'] if r['position']==pos and r['method']==method]
                        x=[r['layer'] for r in records];y=[r['auc'] for r in records]
                        ax.plot(x,y,'o-',ms=3,color=color,label={'frozen_prompt_last':'Frozen prompt-last probe','frozen_t1':'Frozen token-1 probe','layer_specific_probe':'Probe fitted at this layer'}[method])
                        ax.fill_between(x,[r['ci'][0] for r in records],[r['ci'][1] for r in records],color=color,alpha=.1)
                    ax.axhline(.5,color='#aaa',ls='--');ax.set(xlabel='Decoder block output (final = pre-RMS)',ylabel='Held-out AUROC',title='Prompt-last (predicts token 1)' if pos=='prompt_last' else 'After generated token 16');ax.legend(fontsize=8)
                filename=save(fig,out,'layers');files.append(out/filename)
                parts.append(image(name,filename,f"同一批 ≥16-token 验证题 n={lr['same_cohort_n']:,}；t16 指读入第16个生成 token 后的状态，预测的是第17个 token。冻结方向失效但同层拟合成功，支持表示改变；两者都失败仍不能证明信息消失。prompt 状态在 causal decoder 中保持不变，不能由此断言 attention 回取。"))
            else:
                parts.append('<p><strong>此数据的 token16 全层提取／分析尚在运行，本节图未完成。</strong></p>')
            rp=out/'readout_check.json'
            if rp.exists():
                rd=json.loads(rp.read_text());frame=pd.DataFrame(rd['records'])
                fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
                for ax,metric,label in zip(axs,['entropy_delta_mean','argmax_changed_fraction','kl_changed_to_matched_temperature_mean'],['Entropy change (nats)','Fraction with changed top-1','KL to matched-temperature control']):
                    vrows=frame[frame.direction.eq('weakest_v')].sort_values('alpha')
                    random=frame[frame.direction.ne('weakest_v')].groupby('alpha')[metric].agg(['mean','min','max'])
                    # Alpha=1 is the exact identity, not an interpolation across a gap.
                    identity=np.nan if metric=='kl_changed_to_matched_temperature_mean' else 0.
                    vrows=pd.concat([vrows,pd.DataFrame([{'alpha':1.,metric:identity}])]).sort_values('alpha')
                    random.loc[1.]=identity;random=random.sort_index()
                    ax.plot(vrows.alpha,vrows[metric],'o-',color='#287e93',label='Weakest-readout v')
                    ax.plot(random.index,random['mean'],'o-',color='#ce774c',label='Norm-matched random')
                    ax.fill_between(random.index,random['min'],random['max'],color='#ce774c',alpha=.15)
                    ax.axvline(1,color='#aaa',ls='--');ax.set(xlabel='Projection scaling alpha (1 = unchanged)',ylabel=label)
                    if metric=='kl_changed_to_matched_temperature_mean':
                        ax.set_yscale('log');ax.text(.5,.02,'KL(alpha=1) = 0 (exact)',transform=ax.transAxes,ha='center',fontsize=8)
                axs[0].legend(fontsize=8)
                filename=save(fig,out,'readout_check');files.append(out/filename)
                parts.append(image(name,filename,'256 个发现集样本的直接读出计算，原始 pre-RMS 状态上的投影缩放。α=1 为解析恒等对照；log-KL 图在零值处断开。随机方向逐题匹配扰动范数；阴影是三个随机方向的范围，不是置信区间。部分改动幅度很大，详见 JSON。没有上游 block 重算、采样生成或正确率干预。'))
                parts.append(f'<p><a href="{name}/readout_check.json">方向、幅度、匹配温度及样本 ID</a></p>')
            parts.append('<details><summary>展开完整数值、原始标量符号及模型选择</summary>'+table_html(pd.DataFrame(a['scalars']))+f'<p><a href="{name}/analysis.json">全部拟合、内部交叉验证与增量 CI</a> · <a href="{name}/predictions.parquet">逐题预测</a> · <a href="{name}/scalars.parquet">逐题标量</a></p></details></section>')
            examples(cfg,ds,out)
        transfer=json.loads((root/'transfer.json').read_text())
        records=[r for r in transfer['records'] if r['position']=='prompt_last' and r['metric'] in ['entropy','logit_margin','v_min_projection','low_readout_fraction','full_direction','channels16']]
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');forest(ax,records,[LABELS[r['metric']] for r in records]);ax.set_title('Frozen Llama MATH → Llama MMLU; no target-label fit')
        filename=save(fig,root,'transfer');files.append(root/filename)
    overview=pd.DataFrame(heads)
    overview.to_csv(root/'headline.csv',index=False)
    intro='<h1>输出置信度能否解释正确性信号？</h1><p class="lead">首 token 分布、全维方向、跨数据集与跨层读出。所有结果来自已保存的 OpenAct 激活，CPU 计算。</p>'
    intro+='<aside><strong>先明确证据边界</strong><ul><li>这是对已查看过的验证集的后续探索，不是新的独立确认。</li><li>v 沿用 terminal-readout 实际工作分支的最弱读出方向定义，在各模型上重算。原项目 Qwen2.5-0.5B 的观察不能自动推广到当前模型；置信度功能仍待验证。</li><li>熵之外没有增益，不等于信息完全相同；有增益，也不等于超越完整输出分布。</li><li>因果采样干预尚未运行；本报告不声称找到维护或写入机制。</li></ul></aside>'
    intro+='<h2>主要结果：prompt-last</h2>'+table_html(overview)+'<p>所有 AUROC 的正向只在发现集确定。新 probe 使用全部维度，不筛选中等幅度通道。nuisance 包含题型、难度、prompt 长度和 RMS。</p>'
    intro+='<p><strong>判读：</strong>单一首 token 熵不足以替代残差读出；在题型、难度等控制之上，目前尚未确认额外收益。这两点可以同时成立。冻结方向的时间衰减也不等于信息消失，见逐层重新拟合的对照。<a href="findings.md">阅读完整结论与下一步</a>。</p>'
    intro+=f'<h2>跨数据集：冻结源数据上的符号与模型</h2>{image(".",filename,"MMLU 已按完整 prompt 去重；没有用目标标签重新选择方向或符号。")}'
    nav=' · '.join(f'<a href="#{n}">{label}</a>' for n,label in DATA_NAMES.items())
    css='body{font:16px/1.65 system-ui;color:#203246;background:#f4f7fa;max-width:1320px;margin:36px auto;padding:0 24px}h1{font-size:36px}h2{margin-top:32px}.lead{font-size:19px;color:#526273}aside,section{background:white;border:1px solid #dce4eb;border-radius:12px;padding:24px;margin:22px 0}aside{border-left:5px solid #287e93}img{width:100%;height:auto}figure{margin:24px 0}figcaption{color:#516477;font-size:14px}a{color:#176b87}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4eb;white-space:nowrap}.table{overflow:auto}summary{cursor:pointer;font-weight:600}nav{position:sticky;top:0;padding:14px;background:#f4f7faf2;z-index:2}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>置信度与正确性方向 · OpenAct/HSS</title><style>'+css+'</style><nav>'+nav+'</nav>'+intro+''.join(parts)+'<footer><a href="protocol.md">完整协议与条件性因果方案</a> · <a href="analysis.json">分析来源</a> · <a href="transfer.json">全部迁移指标</a></footer></html>'
    (root/'index.html').write_text(page)
    (root/'protocol.md').write_text(Path('docs/confidence-study.zh-CN.md').read_text())
    (root/'findings.md').write_text(Path('docs/confidence-findings.zh-CN.md').read_text())
    save_json(root/'report_provenance.json',{'code_sha256':file_digest(__file__),
        'figures':{str(p.relative_to(root)):file_digest(p) for p in files},'n_figures':len(files)})

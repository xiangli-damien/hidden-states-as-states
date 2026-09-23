"""Question-level paired reporting for the audited fair factor experiment."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from revision_common import sha,write_json
from revision_factor_common import key

METRICS=['kl','delta_nll','delta_first16_nll','mse','region_retention']
GROUP=['dataset','split']


def interval(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid question-level metric')
    draw=np.random.default_rng(42).integers(0,len(x),(2000,len(x)))
    ci=np.quantile(x[draw].mean(1),[.025,.975])
    return {'estimate':float(x.mean()),'low':float(ci[0]),'high':float(ci[1]),'n':len(x)}


def statistics(frame):
    if frame.duplicated(GROUP+['sample_id','method']).any():raise ValueError('Question must be the independent unit')
    summaries=[];pairs=[]
    for group,part in frame.groupby(GROUP,sort=True):
        for method,rows in part.groupby('method',sort=True):
            rows=rows.sort_values('sample_id')
            for metric in METRICS:
                summaries.append(dict(zip(GROUP,group))|{'method':method,'metric':metric,**interval(rows[metric])})
        for rank in [8,4,16]:
            comparisons=[(f'fa_{rank}',f'pca_{rank}'),(f'fa_orthogonal_{rank}',f'pca_{rank}'),
                         (f'fa_{rank}',f'fa_orthogonal_{rank}'),(f'mfa_hard_{rank}',f'pca_{rank}'),
                         (f'mfa_soft_{rank}',f'mfa_hard_{rank}'),(f'mfa_soft_{rank}',f'pca_{rank}')]
            for a,b in comparisons:
                left=part.loc[part.method.eq(a)].set_index('sample_id')
                right=part.loc[part.method.eq(b)].set_index('sample_id')
                if not len(left) and not len(right):continue
                if set(left.index)!=set(right.index):raise ValueError('Incomplete paired question coverage')
                ids=sorted(left.index)
                for metric in METRICS:
                    delta=left.loc[ids,metric].to_numpy()-right.loc[ids,metric].to_numpy()
                    primary=a=='fa_8' and b=='pca_8' and metric in ['kl','delta_nll']
                    pairs.append(dict(zip(GROUP,group))|{'method_a':a,'method_b':b,'metric':metric,
                        'primary':primary,**interval(delta)})
    return summaries,pairs


def run(root):
    raw=root/'functional';dest=root/'report';dest.mkdir(exist_ok=True)
    audit=json.loads((raw/'audit.json').read_text());assert audit['complete']
    plan=json.loads((root/'plan.json').read_text());manifest=json.loads((raw/'audit_inputs.json').read_text())
    records=[]
    for name,digest in sorted(manifest.items()):
        path=raw/'samples'/name;assert sha(path)==digest;records.append(json.loads(path.read_text()))
    assert len(records)==audit['conditions']==plan['expected_conditions']
    baseline={(r['task']['dataset'],r['task']['sample_id']):r for r in records if r['task']['condition']['method']=='identity'}
    rows=[]
    for r in records:
        t=r['task'];original=baseline[t['dataset'],t['sample_id']]
        rows.append({'dataset':t['dataset'],'split':r['split'],'sample_id':t['sample_id'],'method':key(t['condition']),
            'kl':r['next_token_kl'],'delta_nll':r['nll']-original['nll'],
            'delta_first16_nll':r['first16_nll']-original['first16_nll'],
            'mse':float(np.mean(r['geometry']['actual']['token_delta_energy'])/3584),
            'region_retention':r['geometry']['actual']['state_retained_fraction'],
            'argmax_agreement':r['next_token_argmax_agreement']})
    frame=pd.DataFrame(rows)
    for (_,_,_),part in frame.groupby(GROUP+['sample_id']):
        assert set(part.method)=={key(c) for c in plan['conditions']}
    summaries,pairs=statistics(frame)
    frame.to_parquet(dest/'per_question.parquet',index=False)
    for name,value in [('summary.json',summaries),('paired.json',pairs),('primary.json',[r for r in pairs if r['primary']])]:
        write_json(dest/name,value)
    summary=pd.DataFrame(summaries);primary=pd.DataFrame([r for r in pairs if r['primary']])
    cohorts=[('math','validation'),('math','test'),('gsm8k','confirmation')]
    methods=['pca','fa','fa_orthogonal','mfa_hard','mfa_soft']
    labels={'pca':'Local PCA','fa':'Local FA posterior','fa_orthogonal':'FA subspace projection',
            'mfa_hard':'Joint MFA hard','mfa_soft':'Joint MFA soft'}
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    for col,(dataset,split) in enumerate(cohorts):
        for row,metric in enumerate(['kl','delta_nll']):
            ax=axes[row,col]
            for method in methods:
                values=summary.loc[summary.dataset.eq(dataset)&summary.split.eq(split)&summary.metric.eq(metric)].set_index('method')
                entries=values.loc[[f'{method}_{rank}' for rank in [4,8,16]]]
                ax.errorbar([4,8,16],entries.estimate,yerr=[entries.estimate-entries.low,entries.high-entries.estimate],
                    marker='o',markersize=4,capsize=2,label=labels[method],alpha=.9)
            ax.set_xticks([4,8,16]);ax.set_xlabel('Factor / PCA rank');ax.grid(alpha=.2)
            ax.set_ylabel('Next-token KL' if metric=='kl' else 'Full-reference delta NLL')
            ax.set_title(f'{dataset.upper()} / {split}');ax.axhline(0,color='black',lw=.5)
    handles,legends=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Fixed protocol; pointwise 95% paired-question bootstrap. Soft MFA has a larger code budget.')
    fig.savefig(dest/'rank_comparison.png',dpi=180);fig.savefig(dest/'rank_comparison.pdf');plt.close(fig)
    budgets=json.loads((root/'budgets.json').read_text());fits=json.loads((root/'fitting_summary.json').read_text())
    geometry=json.loads((root/'geometry/_SUCCESS.json').read_text());assert geometry['complete']
    for name,digest in geometry['files'].items():assert sha(root/'geometry'/name)==digest,name
    training_density=json.loads((root/'geometry/training_density.json').read_text())
    heldout_density=pd.read_parquet(root/'geometry/heldout_density.parquet')
    density_rows=[{'dataset':'math','split':'train',**r} for r in training_density]
    for (dataset,split,rank),part in heldout_density.groupby(['dataset','split','rank']):
        density_rows.append({'dataset':dataset,'split':split,'rank':int(rank),'tokens':int(part.tokens.sum()),
            **{name:float(np.average(part[name],weights=part.tokens))
               for name in ['fixed_mixture_log_density','joint_mixture_log_density']}})
    write_json(dest/'density.json',density_rows)
    functional=json.loads((raw/'_SUCCESS.json').read_text())
    all_converged=all(r['fixed_converged']==r['components'] and r['joint_converged'] for r in fits.values())
    status='全部拟合通过收敛检查。' if all_converged else '存在未收敛拟合；功能结果只能作为探索性结果。'
    style='<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#21313b}table{border-collapse:collapse;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}img{width:100%}h1,h2{color:#143c53}.note{background:#eef5fa;padding:16px}</style>'
    def table(data):return pd.DataFrame(data).to_html(index=False,escape=True,float_format=lambda x:f'{x:.5g}')
    paragraphs=[]
    for dataset,split in cohorts:
        sub=primary.loc[primary.dataset.eq(dataset)&primary.split.eq(split)]
        parts=[]
        for metric in ['kl','delta_nll']:
            r=sub.loc[sub.metric.eq(metric)].iloc[0]
            parts.append(f'{metric}: {r.estimate:+.5f} [{r.low:+.5f}, {r.high:+.5f}]')
        paragraphs.append(f'<li>{dataset.upper()} / {split}，FA8 − PCA8：'+html.escape('; '.join(parts))+'</li>')
    question_dir=dest/'questions';question_dir.mkdir(exist_ok=True);links=[]
    for dataset in plan['datasets']:
        metadata=pd.read_parquet(root/'inputs'/dataset/'rows.parquet').set_index('sample_id')
        for sid in plan['datasets'][dataset]['selected_ids']:
            source=metadata.loc[sid];sub=frame.loc[frame.dataset.eq(dataset)&frame.sample_id.eq(sid)]
            filename=f'{dataset}_{sid}.html'
            page=f'<!doctype html><meta charset="utf-8">{style}<a href="../index.html">返回</a><h1>{html.escape(dataset+" / "+sid)}</h1>'
            for title,field in [('原题与提示','prompt_text'),('原参考回答','response_text'),('标准答案','ground_truth'),('已有标签','label')]:
                page+=f'<h2>{title}</h2><pre>{html.escape(str(source.get(field,"未提供")))}</pre>'
            page+='<p>本页比较替换后的参考保真度；没有生成新的作答。</p>'+table(sub.to_dict('records'))
            (question_dir/filename).write_text(page)
            links.append(f'<li><a href="questions/{html.escape(filename)}">{html.escape(dataset+" / "+sid)}</a></li>')
    budget_rows=[{'method':m,**{k:v for k,v in b.items() if k!='accounting'}} for m,b in budgets.items()]
    page=f'<!doctype html><meta charset="utf-8">{style}<title>公平 PCA / FA / MFA 功能比较</title><h1>公平 PCA / FA / MFA 功能比较</h1>'
    page+=f'<p>{status} Qwen2-7B，block14，已生成16token后替换这16个位置；每个位置各自编码。</p>'
    page+='<div class="note">主比较固定为同分区、同经验中心的 FA8 − PCA8。区间为问题级、pointwise 95%；两个主指标均报告。MATH 历史测试与此前固定的 GSM8K 确认题分开。KL/NLL 衡量参考保真，不能解释为正确率或选择性 steering。</div>'
    page+='<h2>预定主比较</h2><ul>'+''.join(paragraphs)+'</ul>'+table(primary.to_dict('records'))
    page+='<h2>完整 rank 曲线</h2><p>rank4/8/16均保留，不按结果选rank；hard/soft共用同一联合MFA参数。</p><img src="rank_comparison.png">'
    page+='<h2>表示预算</h2><p>Soft MFA rank8需要575个连续值；hard MFA需要8个加component ID。存储量计入FA噪声及独立GMM编码中心，缓存分解和训练元数据不计入学习参数。</p>'+table(budget_rows)
    page+='<h2>训练与持出似然</h2><p>下表均为完整高斯混合密度。固定FA的混合权重来自训练分区计数；它与最近GMM中心编码器不同。PCA没有在本协议下定义概率密度。更高似然不预设更低输出KL。</p>'+table(density_rows)
    page+=f'<p>本轮功能前向总耗时 {functional["seconds"]:.1f} 秒；CPU几何及全训练密度计算 {geometry["seconds"]:.1f} 秒。训练参数的逐簇收敛记录保存在独立拟合目录。</p>'
    page+='<h2>完整指标与逐题原文</h2><p><a href="summary.json">全部汇总</a> · <a href="paired.json">全部配对比较</a> · <a href="rank_comparison.pdf">PDF</a></p><ul>'+''.join(links)+'</ul>'
    (dest/'index.html').write_text(page)
    outputs=[p for p in dest.rglob('*') if p.is_file() and p.name not in ['_SUCCESS.json','statistics_audit.json']]
    write_json(dest/'_SUCCESS.json',{'complete':True,'conditions':len(frame),'questions':len(baseline),
        'source_audit_sha256':sha(raw/'audit.json'),'primary':[r for r in pairs if r['primary']],
        'all_fits_converged':all_converged,'files':{str(p.relative_to(dest)):sha(p) for p in outputs}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)

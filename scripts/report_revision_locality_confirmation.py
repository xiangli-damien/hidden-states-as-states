"""Held-out target report: no target rank choice, refitting or subset selection."""
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
from report_revision_functional import paired_difference
from report_revision_locality import collect,tables,question_pages


def run(root):
    raw,per,audit=collect(root);summary,pairs=tables(per)
    assert set(per.split)=={'test'}
    main=per.loc[per.prefix_tokens.eq(16)&per.layer.eq(14)&per.width.eq(16)]
    extra=[]
    for metric in ['next_token_kl','delta_nll','delta_first16_nll','mse_per_coordinate']:
        extra.append({'split':'test','prefix_tokens':16,'layer':14,'role':'tokens','width':16,
            'method_a':'local_8','method_b':'shared_64','metric':metric,'primary':False,
            **paired_difference(main,'local_8','shared_64',metric)})
    pairs=pd.concat([pairs,pd.DataFrame(extra)],ignore_index=True)
    dest=root/'report';dest.mkdir(exist_ok=True)
    for name,table in [('individual_conditions',raw),('per_question_seed_average',per),('summary',summary),('paired_methods',pairs)]:
        table.to_parquet(dest/f'{name}.parquet',index=False)
        if name in ['summary','paired_methods']:write_json(dest/f'{name}.json',table.to_dict('records'))
    primary=pairs.loc[pairs.primary&pairs.metric.isin(['next_token_kl','delta_nll'])].to_dict('records')
    write_json(dest/'primary.json',primary)
    s=summary.loc[summary.width.eq(16)].set_index('method')
    methods=['centroid','shared_8','local_8','wrong_local_8','shared_64','shared_512']
    labels=['Center','Shared8','Local8','Wrong local8*','Shared64','Shared512']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['next_token_kl','delta_first16_nll'],['Next-token KL','First16 reference ΔNLL']):
        q=s.loc[methods];x=np.arange(len(q));ax.bar(x,q[metric+'_estimate'],color=['#8996aa','#d29b56','#269887','#b4779b','#d8b57f','#eed4b2'])
        ax.vlines(x,q[metric+'_low'],q[metric+'_high'],color='black',lw=1)
        ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylabel(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Frozen MATH decoder → new GSM8K questions · n={int(s.n.iloc[0])}\nBlock14 / generated16 / width16 · pointwise question bootstrap · *3 permutations averaged')
    for ext in ['png','pdf']:fig.savefig(dest/f'confirmation_reconstruction.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    controls=['centroid','centroid_random','centroid_radial_random','centroid_gram_random'];q=s.loc[controls]
    axes[0].bar(np.arange(4),q.next_token_kl_estimate,color=['#8996aa','#777','#d29b56','#269887'])
    axes[0].vlines(np.arange(4),q.next_token_kl_low,q.next_token_kl_high,color='black',lw=1)
    axes[0].set_xticks(np.arange(4),['Center','Energy random','Radial random','Gram random'],rotation=25,ha='right')
    for family,label,color in [('remove_local','Remove local u','#269887'),('remove_local_radial_random','Radial random','#777'),('complement_energy_matched','Complement matched','#d29b56')]:
        q=s.loc[s.family.eq(family)].sort_values('alpha');x=np.r_[0,q.alpha]
        axes[1].plot(x,np.r_[0,q.next_token_kl_estimate],'o-',label=label,color=color)
        axes[1].fill_between(x,np.r_[0,q.next_token_kl_low],np.r_[0,q.next_token_kl_high],alpha=.12,color=color)
    axes[1].set_xlabel('Clean ablation dose α');axes[1].set_xticks([0,.25,.5,1]);axes[1].legend(fontsize=8)
    for ax in axes:ax.set_ylabel('Next-token KL');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen geometric controls and clean-state ablation\nRadial and Gram controls preserve different properties; actual bf16 deviations audited')
    for ext in ['png','pdf']:fig.savefig(dest/f'confirmation_controls.{ext}',dpi=180)
    plt.close(fig)
    links=question_pages(root,raw,dest)
    all_cis=pairs.loc[pairs.width.eq(16)&pairs.metric.isin(['next_token_kl','delta_nll'])]
    primary_improved=all(r['high']<0 for r in primary)
    lead=('本批两个预定主指标的配对区间都小于0：同样每token保留8个坐标时，局部表示的功能保真优势迁移到了新题。'
          if primary_improved else '本批未在两个预定主指标上同时确认局部表示优于共享表示；保留全部区间。')
    compact=s.loc[methods,['n','next_token_kl_estimate','delta_nll_estimate','mse_per_coordinate_estimate']].copy()
    compact.columns=['问题数','下一token KL ↓','完整参考ΔNLL ↓','每坐标MSE ↓']
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结MATH表征的GSM8K确认</title><style>body{font:16px system-ui;margin:30px;max-width:1400px;line-height:1.6}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}.scroll{overflow:auto}</style>',
        '<h1>冻结MATH表征 → 新GSM8K问题：功能确认</h1>',
        '<h2>先看本批结果</h2><p>'+lead+'</p>',compact.to_html(float_format=lambda x:f'{x:.5f}'),
        '<p>局部8维与共享8维的MSE也不同，尚未单独隔离方向的功能专属性。共享64/512保留更多连续坐标；它们的优势不能被省略，也不能作为等编码预算比较。几何控制和clean消融见下方，均不是数学正确率或选择性steering证据。</p>',
        '<p>64道GSM8K train问题在任何本轮模型结果出现前按hash选定。不是之前使用的1319道GSM8K test。所有中心和方向由MATH训练集固定，没有target拟合、target验证或按结果补抽题。它测试跨数据迁移；不等于同域MATH复现，也不排除模型预训练见过GSM8K。</p>',
        '<p>主设置block14／已生成16token／替换16位置。每个token独立编码，模型其余上下文保留。主比较local8−shared8，KL与完整参考NLL都报告；区间以问题为单位，随机seed先在题内平均，pointwise 95%。宽度1/4为辅助条件。本轮参考回答未评分，没有干预后自由生成或正确率结论。</p>',
        '<h2>预先固定的两个主指标</h2>',pd.DataFrame(primary).to_html(index=False),
        '<p>shared64由旧MATH验证MSE选定；新GSM8K误差未必相近，不可称target误差匹配。shared512只与K64 local8的基矩阵存储量相同，每token保留512而非8坐标。错误基、随机控制、经验中心桥接、clean消融及所有不确定区间均保留。</p>',
        '<img src="confirmation_reconstruction.png"><p><a href="confirmation_reconstruction.pdf">PDF</a></p>',
        '<img src="confirmation_controls.png"><p><a href="confirmation_controls.pdf">PDF</a></p>',
        '<h2>width16全部KL／完整NLL配对比较</h2><div class="scroll">',all_cis.to_html(index=False),'</div>',
        '<details><summary>全部设置（含实际MSE、region保留率和argmax）</summary><div class="scroll">',summary.to_html(index=False),'</div></details>',
        '<details><summary>逐题原文、标准答案、原参考回答和全部条件</summary><ul>',*links,'</ul></details>',
        '<details><summary>执行审计、bf16误差与排除记录</summary><pre>',html.escape(json.dumps(audit,indent=2)),
        html.escape(json.dumps(json.loads((root/'functional/plan.json').read_text())['exclusions'],indent=2)),'</pre></details></html>']
    (dest/'index.html').write_text('\n'.join(parts))
    write_json(dest/'_SUCCESS.json',{'source_audit_sha256':sha(root/'functional/audit.json'),'primary':primary,
        'conditions':len(raw),'question_averaged_conditions':len(per),'matching_mode':'frozen_source',
        'source_mse_match_sha256':sha(root/'source_mse_match.json'),
        'report_code_sha256':sha(Path(__file__)),
        'dependencies_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['report_revision_locality.py','report_revision_functional.py','revision_common.py']},
        'files':{p.name:sha(p) for p in dest.glob('*.parquet')}})
    print(json.dumps({'conditions':len(raw),'primary':primary},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)

"""Question-level exploratory summaries and inspectable saved-output profiles."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_common import sha, write_json

METRICS = ['next_token_kl', 'total_variation', 'delta_nll_first_token', 'delta_nll_first16',
           'delta_nll_positions2_16', 'delta_nll_after16', 'delta_nll_full', 'mse_per_coordinate']
STYLE = '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap}img{width:100%}.scroll{overflow:auto}summary{cursor:pointer;font-weight:bold}h2{margin-top:35px}</style>'


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    boot = np.random.default_rng(42).integers(0, len(a), (2000, len(a)))
    low, high = np.percentile(a[boot].mean(1), [2.5, 97.5])
    positive = np.sort(np.maximum(a, 0))[::-1]
    k = max(1, int(np.ceil(.1*len(a))))
    return {'n': len(a), 'mean': float(a.mean()), 'low': float(low), 'high': float(high),
            'median': float(np.median(a)), 'fraction_positive': float((a>0).mean()),
            'top10pct_n': k,
            'top10pct_share_of_positive_sum': float(positive[:k].sum()/positive.sum()) if positive.sum()>0 else None,
            'mean_after_dropping_largest10pct': float(np.sort(a)[:-k].mean()) if len(a)>k else None}


def contrast(frame, method_a, width_a, method_b, width_b, metric):
    a = frame.loc[frame.method.eq(method_a)&frame.width.eq(width_a)].set_index('sample_id')
    b = frame.loc[frame.method.eq(method_b)&frame.width.eq(width_b)].set_index('sample_id')
    assert a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index) and len(a)
    return (a[metric]-b[metric]).sort_index()


def run(root):
    receipt = json.loads((root/'_SUCCESS.json').read_text())
    for name, digest in receipt['files'].items():
        assert sha(root/name) == digest
    frame = pd.read_parquet(root/'conditions.parquet')
    cats = pd.read_parquet(root/'categories.parquet')
    keys = ['cohort', 'sample_id', 'width', 'method']
    assert not frame.duplicated(keys).any()
    # Independent accounting check over ALL vocabulary categories, not selected top tokens.
    csum = cats.groupby(keys)[['kl_signed_sum', 'total_variation', 'probability_change']].sum()
    indexed = frame.set_index(keys).loc[csum.index]
    np.testing.assert_allclose(csum.kl_signed_sum, indexed.next_token_kl, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(csum.total_variation, indexed.total_variation, atol=1e-10, rtol=1e-10)
    assert csum.probability_change.abs().max() < 4e-6
    summary, pairs, individual = [], [], []
    comparisons = [('local_8',16,'shared_64',16), ('local_8',16,'shared_8',16)]
    comparisons += [(m,a,m,b) for m in ['centroid','local_8','shared_8'] for a,b in [(4,1),(16,4),(16,1)]]
    for (cohort,width,method), sub in frame.groupby(['cohort','width','method'], sort=True):
        for metric in METRICS:
            summary.append({'cohort':cohort,'width':int(width),'method':method,'metric':metric, **describe(sub[metric])})
    for cohort, sub in frame.groupby('cohort', sort=True):
        for a,wa,b,wb in comparisons:
            for metric in METRICS:
                delta = contrast(sub,a,wa,b,wb,metric)
                common = {'cohort':cohort,'method_a':a,'width_a':wa,'method_b':b,'width_b':wb,'metric':metric}
                pairs.append(common|describe(delta))
                individual.extend(common|{'sample_id':sid,'difference':float(value)} for sid,value in delta.items() if np.isfinite(value))
    dest = root/'report'; dest.mkdir(exist_ok=True); (dest/'questions').mkdir(exist_ok=True)
    summary, pairs = pd.DataFrame(summary), pd.DataFrame(pairs)
    summary.to_parquet(dest/'summary.parquet',index=False);pairs.to_parquet(dest/'paired.parquet',index=False)
    pd.DataFrame(individual).to_parquet(dest/'paired_questions.parquet',index=False)
    category_summary = cats.groupby(['cohort','width','method','category'],as_index=False).mean(numeric_only=True)
    category_summary.to_parquet(dest/'categories.parquet',index=False)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(cohort,sub) in zip(axes,frame.groupby('cohort',sort=True)):
        for other,color in [('shared_64','#9868ae'),('shared_8','#168887')]:
            delta = np.sort(contrast(sub,'local_8',16,other,16,'next_token_kl').to_numpy())
            ax.plot(delta,(np.arange(len(delta))+1)/len(delta),label='Local8 minus '+other,color=color)
        ax.axvline(0,c='gray',ls='--',lw=1);ax.set_title(cohort);ax.set_xlabel('Per-question next-token KL difference (nats)')
        ax.set_ylabel('Cumulative fraction of questions');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Positive = Local8 changes the original distribution more; each dot of the CDF is one question')
    for ext in ['png','pdf']: fig.savefig(dest/f'question_differences.{ext}',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    sections=['first_token','positions2_16','after16','full']
    for ax,(cohort,sub) in zip(axes,frame.groupby('cohort',sort=True)):
        for method,color in [('local_8','#168887'),('shared_64','#9868ae'),('shared_8','#d39640')]:
            table=summary.loc[summary.cohort.eq(cohort)&summary.width.eq(16)&summary.method.eq(method)].set_index('metric')
            selected=table.loc[['delta_nll_'+s for s in sections]]
            ax.plot(range(4),selected['mean'],'o-',c=color,label=method)
            ax.vlines(range(4),selected.low,selected.high,color=color,alpha=.5)
        ax.axhline(0,c='gray',lw=1);ax.set_xticks(range(4),['First','2–16','After16','Full']);ax.set_title(cohort)
        ax.set_yscale('symlog', linthresh=.01)
        ax.set_ylabel('Reference ΔNLL (symlog, linear within ±0.01)');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One fixed original continuation; question-weighted means and pointwise 95% bootstrap intervals')
    for ext in ['png','pdf']: fig.savefig(dest/f'loss_windows.{ext}',dpi=170)
    plt.close(fig)
    links=[]
    for path in sorted((root/'questions').glob('*.json')):
        q=json.loads(path.read_text());sid=q['sample_id']
        content=['<!doctype html><meta charset="utf-8">'+STYLE,'<a href="../index.html">返回报告</a>',
                 '<h1>'+html.escape(q['cohort']+' · '+sid)+'</h1>',
                 '<p>原模型参考续写；下面没有干预后的自由生成。fragment 分类仅描述字符形式。</p>']
        for title,field in [('题目','prompt'),('标准答案','ground_truth'),('已生成16 token','prefix_text'),('原模型参考回答','reference')]:
            content+=['<details><summary>'+title+'</summary><pre>'+html.escape(q[field])+'</pre></details>']
        for method,c in q['conditions'].items():
            loss=np.asarray(c['reference_nll']);base=np.asarray(q['conditions']['identity']['reference_nll'])
            assert len(loss)==len(base)==len(q['reference_ids'])
            np.testing.assert_allclose(loss-base,c['delta_reference_nll'],atol=1e-12,rtol=1e-12)
            delta=loss-base;worst=np.argsort(-np.abs(delta))[:15]
            positions=pd.DataFrame([{'reference_position':int(i+1),'response_position':int(i+17),
                'fragment':repr(q['reference_fragments'][i]),'original_nll':base[i],
                'changed_nll':loss[i],'delta_nll':delta[i]} for i in worst])
            table=pd.DataFrame(c['top_candidates']).sort_values('delta_probability',key=abs,ascending=False)
            table['fragment']=table.fragment.map(repr);table['selected_by']=table.selected_by.map(', '.join)
            content+=['<details><summary>'+html.escape(method)+'</summary><p>原始概率最高、概率变化绝对值最大和 KL 正负项最大的候选并集；不是完整词表。</p>',
                '<div class="scroll">'+table.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>',
                '<h3>参考续写中 |ΔNLL| 最大的15个位置</h3>',positions.to_html(index=False,float_format=lambda x:f'{x:.5g}'),'</details>']
        (dest/'questions'/f'{path.stem}.html').write_text('\n'.join(content))
        links.append('<li><a href="questions/'+path.stem+'.html">'+html.escape(q['cohort']+' '+sid)+'</a></li>')
    show = pairs.loc[pairs.width_a.eq(16)&pairs.width_b.eq(16)&pairs.metric.isin(['next_token_kl','delta_nll_full','delta_nll_after16'])]
    four=summary.loc[summary.width.eq(16)&summary.method.isin(['identity','centroid','local_8','remove_local_8_1.0'])&summary.metric.eq('next_token_kl')]
    window=pairs.loc[pairs.method_a.eq(pairs.method_b)&pairs.metric.isin(['next_token_kl','delta_nll_full'])]
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>输出差异画像</title>'+STYLE,
        '<h1>局部低秩表示到底改变了哪些输出？</h1>',
        '<p>Qwen2-7B-Instruct · block14 · 已生成16 token · 128道问题 · 已保存输出的 CPU 事后分析。MATH 32验证/32历史测试与 GSM8K 64确认题分别报告。</p>',
        '<p><b>先读边界：</b>KL 是下一 token 全分布偏差；参考 NLL 是原模型既定续写的损失，不是干预正确率。词表分类是字符规则，不是语义机制；有符号 KL 项可能为负。所有配对以问题为单位，置信区间为逐项区间。</p>',
        '<h2>逐题差异与少数大效应</h2><img src="question_differences.png">',
        '<p>正数表示 local8 的偏差更大。top10pct_share_of_positive_sum 表示最大的10%题占全部正差总和的比例；不是总净效应的比例。删去它们的均值只用于敏感性描述，不能换成新的主结论。</p>',
        '<div class="scroll">'+show.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>',
        '<h2>边界变化是否延续？</h2><img src="loss_windows.png">',
        '<p>First 是 patch 边界后的第1个参考 token（回答第17个），After16 是回答第33个及之后；不是 prompt 后的第1个 token。每题在该窗口内平均，再跨题平均。纵轴在 ±0.01 内线性，外侧对数，以显示边界与后续差异。后续只保存了参考 token 损失，不能计算后续完整词表 KL。</p>',
        '<h2>中心、局部部分、补空间、完整状态</h2><p>centroid=μ；local8=μ+u；remove_local_8_1.0=μ+v；identity=μ+u+v。KL 不可加，不能据此分配功能百分比。</p>',
        four.to_html(index=False,float_format=lambda x:f'{x:.5g}'),
        '<h2>现有窗口宽度比较</h2><p>width1/4/16 是替换最后1/4/16个位置；同时改变总扰动能量。shared64仅有width16，未补造其他宽度。</p>',
        '<div class="scroll">'+window.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>',
        '<h2>全词表字符类别统计</h2><p>以下来自全词表，非 top-k 采样。kl_signed_sum 是按原分布概率加权的有符号 KL 项；类别差异不能直接命名为算术功能。</p>',
        '<div class="scroll">'+category_summary.loc[category_summary.width.eq(16)&category_summary.method.isin(['local_8','shared_64'])].to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>',
        '<h2>逐题核对</h2><p>功能假设的候选发现只使用 MATH validation；其余用于描述性复查。未来反事实测试必须另定独立协议。</p><ul>',*links,'</ul></html>']
    (dest/'index.html').write_text('\n'.join(parts))
    files=[p for p in dest.rglob('*') if p.is_file() and p.name!='_SUCCESS.json']
    write_json(dest/'_SUCCESS.json',{'complete':True,'source_receipt_sha256':sha(root/'_SUCCESS.json'),
        'questions':receipt['questions'],'conditions':len(frame),'summary_rows':len(summary),'paired_rows':len(pairs),
        'checks':['unique question/condition keys','full-vocabulary category conservation','exact paired-question coverage','all reference-loss differences'],
        'files':{str(p.relative_to(dest)):sha(p) for p in files}})
    print(show.to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
    run(ap.parse_args().root)

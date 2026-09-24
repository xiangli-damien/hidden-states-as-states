"""Completed behavior pilot: paired repairs/damages and all raw continuations."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from revision_common import sha,write_json,paired_ratio_ci


def run(root):
    stage=root/'behavior';audit=json.loads((stage/'execution_audit.json').read_text())
    assert audit['stage_complete'] and audit['snapshot_conditions']==1008
    assert audit['plan_sha256']==sha(stage/'plan.json')
    frame=pd.read_parquet(stage/'per_question.parquet')
    assert len(frame)==1008 and frame.sample_id.nunique()==24
    rows=[];keys=['sample_id','split','prefix_tokens','layer','role','width']
    baseline=frame.loc[frame.method.eq('identity')].set_index(keys)
    for group,sub in frame.groupby(['split','prefix_tokens','width','method']):
        paired=sub.set_index(keys).join(baseline[['correct','normalized_answer']],rsuffix='_baseline',validate='one_to_one').sort_index()
        assert len(paired)==12
        before=paired.correct_baseline.to_numpy(bool);after=paired.correct.to_numpy(bool)
        repair=int((~before&after).sum());damage=int((before&~after).sum());n=len(paired)
        row=dict(zip(['split','prefix','width','method'],group))
        row.update(n=n,baseline_correct=int(before.sum()),after_correct=int(after.sum()),wrong_to_correct=repair,
                   correct_to_wrong=damage,net_correct=repair-damage,
                   paired_net_accuracy_bootstrap=paired_ratio_ci(after.astype(int)-before.astype(int),np.ones(n)),
                   exact_paired_p=1. if repair+damage==0 else float(binomtest(repair,repair+damage,.5).pvalue),
                   # Unlike the empirical bootstrap, zero observed changes must
                   # not imply a zero-width population equivalence interval.
                   changed_correctness_wilson95=list(binomtest(repair+damage,n).proportion_ci(method='wilson')),
                   answer_agreement=float((paired.normalized_answer.fillna('')==paired.normalized_answer_baseline.fillna('')).mean()),
                   parse_failed=int(paired.parse_failed.sum()),truncated=int(paired.finish_reason.eq('length').sum()))
        rows.append(row)
    previous=json.loads((root/'report/behavior_summary.json').read_text())
    previous={(r['split'],r['prefix'],r['width'],r['method']):r for r in previous}
    for r in rows:
        old=previous[r['split'],r['prefix'],r['width'],r['method']]
        for k in ['wrong_to_correct','correct_to_wrong','answer_agreement']:np.testing.assert_allclose(r[k],old[k],rtol=0,atol=1e-12)
        # The initial report read filesystem order. Finite bootstrap draws can
        # vary with row order; the estimate is invariant, the endpoints need not
        # be bitwise identical. This report freezes sorted question order.
        np.testing.assert_allclose(r['paired_net_accuracy_bootstrap']['estimate'],old['net_accuracy']['estimate'],rtol=0,atol=1e-12)
    dest=stage/'report';dest.mkdir(exist_ok=True);write_json(dest/'summary.json',rows)
    questions={}
    for marker in (root/'prefixes').glob('shard_*/_SUCCESS.json'):
        data=pd.read_parquet(marker.parent/'rows.parquet')
        for row in data.loc[data.sample_id.isin(frame.sample_id)].to_dict('records'):
            questions[row['sample_id']]=str(row.get('prompt_text',row.get('model_input_text','')))
    assert len(questions)==24
    links=[];(dest/'questions').mkdir(exist_ok=True)
    style='<style>body{font:16px system-ui;margin:32px;max-width:1400px;line-height:1.5}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}details{margin:8px 0}.scroll{overflow:auto}img{width:100%}</style>'
    for sid,sub in frame.groupby('sample_id'):
        title=f'{sid} · {sub.split.iloc[0]}'
        parts=['<!doctype html><html lang="zh"><meta charset="utf-8">'+style,'<h1>'+html.escape(title)+'</h1>',
               '<h2>题目与指令</h2><pre>'+html.escape(questions[sid])+'</pre>',
               '<h2>参考答案</h2><pre>'+html.escape(str(sub.ground_truth.iloc[0]))+'</pre>']
        for r in sub.sort_values(['prefix_tokens','width','method']).to_dict('records'):
            heading=f'prefix={r["prefix_tokens"]}, width={r["width"]}, {r["method"]}: correct={r["correct"]}, {r["finish_reason"]}'
            parts+=['<details><summary>'+html.escape(heading)+'</summary><pre>'+html.escape(r['response_text'])+'</pre></details>']
        parts+=['</html>'];(dest/'questions'/f'{sid}.html').write_text('\n'.join(parts))
        links.append(f'<li><a href="questions/{sid}.html">{html.escape(title)}</a></li>')
    methods=['position_mean','centroid','local_pca_8','empirical_pca_8','matched_random','centroid_energy1']
    names=['Slot mean','GMM center','Residual SVD 8','Local PCA 8','Random','Center, fixed E']
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    df=pd.DataFrame(rows)
    for i,split in enumerate(['validation','test']):
        for j,prefix in enumerate([0,16]):
            sub=df.loc[df.split.eq(split)&df.prefix.eq(prefix)&df.width.eq(16)].set_index('method').loc[methods]
            ax=axes[i,j];x=np.arange(len(methods))
            ax.bar(x-.17,sub.wrong_to_correct,.34,label='Wrong → correct',color='#23896a')
            ax.bar(x+.17,-sub.correct_to_wrong,.34,label='Correct → wrong',color='#bd5b5b')
            ax.axhline(0,color='#333',linewidth=.7);ax.set_ylim(-3.5,4.5);ax.set_yticks(range(-3,5))
            ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylabel('Number of questions')
            ax.set_title(f'{split} · prefix {prefix} · baseline {int(sub.baseline_correct.iloc[0])}/12')
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=9);fig.suptitle('Qwen2-7B MATH · block14 · 16-token replacement\n12 questions per split; descriptive pilot, not a selected correction policy',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(dest/f'behavior_repairs_damages.{ext}',dpi=180)
    plt.close(fig)
    manifest={'conditions':1008,'questions':24,'all_saved_outputs_independently_rescored':True,
        'prior_report_counts_and_paired_estimates_verified':True,'bootstrap_question_order':'sorted sample IDs',
        'no_equivalence_claim_from_zero_flips':True,
        'decoding':'do_sample=False; pinned checkpoint repetition_penalty=1.05 inherited; counterfactual greedy-v2 separately uses1.0',
        'plan_sha256':sha(stage/'plan.json'),'data_sha256':sha(stage/'per_question.parquet'),
        'execution_audit_sha256':sha(stage/'execution_audit.json'),'code_sha256':sha(Path(__file__))}
    write_json(dest/'audit.json',manifest)
    (dest/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Behavior reconstruction pilot</title>'+style+
        '<h1>重构之后，实际答案是否变化？</h1><p>24题，验证12/历史测试12；1008个生成条件。全部重新评分并通过identity、位置、token预算与解码审计。</p>'
        '<p>所有方法/宽度均报告；没有按历史测试最佳条件挑选纠错方法。每题才是统计单位，1008不是1008个独立问题。</p>'
        '<p>12题中没有观察到正确性翻转时，bootstrap可退化为[0,0]；这不是等效证明。另列Wilson区间与精确配对检验。所有区间/检验未校正多重比较，pilot不支持稳定准确率提升。</p>'
        '<p>基线是同prefix、同fresh-prefill的identity续写，不是历史标签。prefix16的验证基线5/12，prompt基线6/12；不混用两者。原回答可因重建前缀/浮点计算路径不同而改变。</p>'
        '<p>实际解码为do_sample=False并继承固定Qwen版本的repetition_penalty=1.05；新反事实gate另用显式1.0，两者不混称同一解码协议。</p>'
        '<img src="behavior_repairs_damages.png" alt="Paired repairs and damages"><p><a href="behavior_repairs_damages.pdf">PDF</a></p>'
        '<h2>全部条件</h2><div class="scroll">'+df.to_html(index=False)+'</div><h2>逐题原文与全部续写</h2><ul>'+''.join(links)+'</ul></html>')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)

"""Full frozen generation results plus original texts; semantic review kept separate."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(v):return html.escape(str(v))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def run(root,output,review):
    root,output,review=Path(root),Path(output),Path(review)
    s=json.loads((root/'summary.json').read_text());a=json.loads((root/'independent_local_audit.json').read_text())
    assert a['passed'] and a['summary_sha256']==sha(root/'summary.json')
    cases=json.loads((root/'cases.json').read_text());cohort={r['sample_id']:r for r in json.loads((root/'cohort.json').read_text())}
    notes=json.loads((review/'review_notes.json').read_text());snapshot=json.loads((review/'snapshot.json').read_text())
    assert notes['snapshot_sha256']==sha(review/'snapshot.json')
    old={(x['record']['question_id'],x['record']['condition']):x['record'] for x in snapshot['records']}
    reviewed={r['question_id']:r for r in notes['rows']}
    grouped={c['question_id']:{arm:json.loads((root/'records'/c['question_id']/(arm+'.json')).read_text()) for arm in ['zero','C1','ORTH_MAN_1']} for c in cases}
    assert len(grouped)==202 and sum(len(v) for v in grouped.values())==606
    for q in reviewed:
        for arm in ['zero','C1']:
            assert grouped[q][arm]['response_text']==old[(q,arm)]['response_text']
    output.mkdir(parents=True,exist_ok=True)
    names={'zero':'不干预','C1':'C1 截断','ORTH_MAN_1':'正交方向对照'}
    colors={'zero':'#8290a3','C1':'#147d92','ORTH_MAN_1':'#cb913c'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,metric,title in zip(axes,['primary_count','correct'],['Complete boxed and <2048 tokens','Frozen automatic correctness']):
        for j,arm in enumerate(names):
            vals=[s['groups'][g]['arms'][arm][metric]/s['groups'][g]['n']*100 for g in ['sink','normal']]
            xs=np.arange(2)+(j-1)*.23
            ax.bar(xs,vals,.22,color=colors[arm],label={'zero':'Zero','C1':'C1','ORTH_MAN_1':'Orthogonal control'}[arm])
            for x,v,g in zip(xs,vals,['sink','normal']):
                ax.text(x,v+1.2,f"{s['groups'][g]['arms'][arm][metric]}/{s['groups'][g]['n']}",ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['Historical candidate questions','Normal comparison questions'],ylim=(0,100),ylabel='Percent',title=title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9,loc='upper left')
    for ext in ['png','pdf']:fig.savefig(output/f'final_generation.{ext}',dpi=160)
    plt.close(fig)
    rows=[]
    for g in ['sink','normal']:
        n=s['groups'][g]['n']
        for arm in names:
            v=s['groups'][g]['arms'][arm];local=a['groups'][g]['arms'][arm]
            rows.append(['候选题' if g=='sink' else '正常对照题',names[arm],f"{v['primary_count']}/{n} ({v['primary_count']/n:.1%})",f"{v['correct']}/{n} ({v['correct']/n:.1%})",f"{v['mean_tokens']:.1f}",local['at_cap'],f"{v['exit_vs_zero']}/{v['entry_vs_zero']}",v['sink_count']])
    comparisons=[]
    for arm,c in s['groups']['sink']['primary_comparisons'].items():
        comparisons.append([f'C1 对 {names[arm]}',c['wins'],c['losses'],f"{100*c['difference']:+.2f} pp",f"[{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}]",f"{c['p_two_sided_exact']:.6g}",f"{c['holm_p']:.6g}"])
    correctrows=[]
    for group in ['sink','normal']:
        for arm in ['zero','ORTH_MAN_1']:
            c=a['groups'][group]['contrasts'][arm]['automatic_correctness']
            correctrows.append([group,f'C1 对 {names[arm]}',c['wins'],c['losses'],c['net'],f"{c['difference']:+.2%}"])
    decision='通过' if s['decision']['improves_vs_both'] else '未通过'
    title='在线 C1 截断：606 条完整生成结果'
    scope='Qwen2-7B-Instruct；hidden-state index14；原2% ICL容差 K33 的 response-mean 地图。152道历史候选题＋50道正常对照题，zero/C1/ORTH_MAN_1各一条。0%容差地图的ID58分析没有用于这轮干预。主终点是完整非空boxed且不足2048token。'
    body='<h1>'+title+'</h1><p class="note">'+scope+'</p><p><strong>606/606生成完成，原始记录审核及独立本地统计核验通过。预定“C1对zero、对正交对照均改善且显著”规则：'+decision+'。</strong></p>'
    body+='<img src="final_generation.png" alt="完整生成的主终点与自动正确率"><h2>全部结果</h2>'+table(['人群','组别','主终点成功','自动正确','平均tokens','达到2048','退出/进入原区域 vs zero','重编码仍在原区域'],rows)
    body+='<h2>预定主检验：152道候选题</h2>'+table(['比较','主终点改善题数','损伤题数','净变化','95%配对bootstrap区间（pp）','双侧精确p','Holm校正p'],comparisons)
    body+='<p>只有上述两项比较共同达到预定方向与校正p&lt;0.05，才算完整通过。95%区间逐项报告，主判据以Holm校正检验为准。不能按结果换终点、换地图或只报对zero。</p>'
    body+='<h2>自动正确率的修复与损伤（次要描述）</h2>'+table(['人群','比较','自动错→对','自动对→错','净题数','净比例'],correctrows)
    body+='<p>自动分数沿用冻结解析器。已有43题全文检查发现过“答案正确但无boxed、被解析器漏掉”以及“答案对了但推导错误”，所以这里不能把自动错→对都称为有效推理修复。</p>'
    body+='<h2>范围与解释</h2><ul><li>历史候选区域里抽出的题，新baseline不一定仍在该区域。退出/进入以同题新zero为参照。</li><li>生成后在无干预模型中重编码新文本；不是直接读取被干预时的在线状态。</li><li>正交对照在自己的当前激活上匹配C1规则的更新量；生成路径分叉后，三组累计能量并不相同。</li><li>沿用此前检查过的问题，属于探索性扩展。原无标签地图使用全部5000题，不能称独立新数据确认。</li><li>最终数学正确性和推导质量尚未全部逐题复核。当前仅43/202题有此前AI辅助、非盲的全文复核；其余159题明确列为待复核。</li></ul>'
    body+='<p>审核时间 '+esc(s['completed_utc'])+'；在用户指定冻结时间前：'+esc(s['submission_eligible'])+'。数值修正v2/v3保留历史失败与原门槛；没有删题或放宽审核。</p>'
    body+='<h2>202道题的三组完整原文</h2><label>人群 <select id="group"><option value="all">全部</option><option value="sink">候选题</option><option value="normal">正常题</option></select></label><label>变化 <select id="change"><option value="all">全部</option><option value="gain">C1自动错→对</option><option value="loss">C1自动对→错</option><option value="boxgain">C1新增主终点成功</option><option value="boxloss">C1丢失主终点成功</option></select></label><label>搜索 <input id="search" placeholder="题号 / 题目"></label><span id="count"></span>'
    for case in sorted(cases,key=lambda r:r['question_id']):
        q=case['question_id'];rs=grouped[q];z,c=rs['zero'],rs['C1'];meta=cohort[q]
        tags=[]
        if c['correct'] and not z['correct']:tags.append('gain')
        if z['correct'] and not c['correct']:tags.append('loss')
        if c['primary_success'] and not z['primary_success']:tags.append('boxgain')
        if z['primary_success'] and not c['primary_success']:tags.append('boxloss')
        note=reviewed[q]['note'] if q in reviewed else '尚未逐题语义复核；下列自动标签不能代替数学检查。'
        arms=''
        for arm in names:
            r=rs[arm]
            arms+='<section><h3>'+names[arm]+'</h3><p>'+esc(f"{r['n_tokens']} tokens · 主终点 {r['primary_success']} · 自动正确 {r['correct']} · 区域成员 {r['sink']}")+'</p><pre>'+esc(r['response_text'])+'</pre></section>'
        body+='<details class="case" id="'+q+'" data-group="'+case['set']+'" data-change="'+','.join(tags)+'" data-search="'+esc((q+' '+meta['prompt_text']).lower())+'"><summary>'+esc(f"{q} · {case['set']} · 自动正确 {z['correct']}→{c['correct']} · 主终点 {z['primary_success']}→{c['primary_success']}")+'</summary><p class="note">'+esc(note)+'</p><h3>题目</h3><pre>'+esc(meta['prompt_text'])+'</pre><h3>参考答案</h3><pre>'+esc(meta['ground_truth'])+'</pre><div class="arms">'+arms+'</div></details>'
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 20px;background:#f7f9fc;color:#17283a}h1{font-size:28px}h2{margin-top:35px}img{max-width:100%}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{border-bottom:1px solid #dfe5ec;padding:8px 12px;text-align:right}th{background:#e8edf4}.note{background:#eaf1f8;padding:15px;border-radius:8px}.arms{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}section{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}details{background:white;padding:12px;border:1px solid #dbe2eb;margin:10px 0;border-radius:7px}summary{cursor:pointer;font-weight:600}select,input{font:inherit;padding:5px;margin:8px}[hidden]{display:none!important}@media(max-width:900px){.arms{grid-template-columns:1fr}table{font-size:12px}}'
    js="const g=document.getElementById('group'),c=document.getElementById('change'),s=document.getElementById('search');function filter(){let n=0;for(const r of document.querySelectorAll('.case')){r.hidden=!((g.value==='all'||g.value===r.dataset.group)&&(c.value==='all'||r.dataset.change.split(',').includes(c.value))&&r.dataset.search.includes(s.value.toLowerCase()));if(!r.hidden)n++;}document.getElementById('count').textContent=n+'道题';}for(const e of [g,c,s])e.addEventListener('input',filter);filter();"
    (output/'steering_final.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style>'+body+'<script>'+js+'</script></html>')
    artifact=dict(records=606,questions=202,reviewed_pairs=43,pending_semantic_review=159,summary_sha256=sha(root/'summary.json'),independent_audit_sha256=sha(root/'independent_local_audit.json'),
                  files={n:sha(output/n) for n in ['steering_final.html','final_generation.png','final_generation.pdf']})
    (output/'final_report_manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps(artifact,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--review',default='results/sink-generation-review-20260925');a=p.parse_args();run(a.root,a.output,a.review)

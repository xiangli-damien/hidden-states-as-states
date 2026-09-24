"""Render audited projection results; preserve scope and pending human review."""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_common import sha,write_json
from report_revision_completion import STYLE,table


def page(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HSS projection v2</title><style>'+STYLE+'</style>'+body+'</html>')


def run(root):
    audit=json.loads((root/'statistics_audit.json').read_text());assert audit['complete'] and audit['summary_sha256']==sha(root/'summary.json')
    s=json.loads((root/'summary.json').read_text());plan=json.loads((root/'plan.json').read_text());dest=root/'report'
    if (dest/'_SUCCESS.json').exists():raise RuntimeError('Do not overwrite a completed report')
    dest.mkdir(exist_ok=True)
    parts=['<h1>HSS · 局部投影干预</h1><p>Qwen2-7B-Instruct，block14，第16个生成token后修改最后4个位置；原测试与新探索严格分开。</p>',
        '<div class="note">已完成原始数据与统计审计；图表人工检查和修复／损伤的盲语义复核仍待完成。自动解析／截断标记不能代替推理错误分析。</div>',
        '<h2>原冻结 MATH 256题确认</h2><p>沿用原统计结果。随机三seed按题内平均，未挑seed；下面区间均为逐项95%，不是同时覆盖区间。</p>',table(s['primary_rows']),table(s['primary_paired']),
        '<h2>新 GSM8K 冻结迁移</h2><p>官方train中新选项目内未使用问题，排除过去64题和历史test1319。沿用原MATH规则；不拟合目标地图。</p>']
    display=[]
    for c in s['comparisons']:
        if c['group']=='MATH_primary256_supplement':continue
        d={k:v for k,v in c.items() if k not in ['per_question','delta_accuracy']}
        d.update(delta=c['delta_accuracy']['estimate'],ci95=c['delta_accuracy']['ci95']);display.append(d)
    parts.append(table([r for r in display if r['group']=='GSM8K_frozen_transfer']))
    parts+=['<h2>旧验证题上的解释性消融</h2><p>32题来自已经使用过的64题验证集。这些变体没有独立确认，不能替换原256题主方法。第8/32token干预使用各自重算路径的基线；最多640次新增生成。</p>',
        table([r for r in display if r['group']=='MATH_exploratory32']),
        '<h3>运行与缺失项</h3>',table([dict(batch=c['name'],asset=c['asset_available'],completed=c['name'] in s['batches']) for c in plan['conditions']])]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,group,title in [(axes[0],'GSM8K_frozen_transfer','Frozen GSM8K transfer'),(axes[1],'MATH_exploratory32','MATH validation only (exploratory)')]:
        rows=[c for c in s['comparisons'] if c['group']==group and c['control'].startswith('baseline')]
        for i,c in enumerate(rows):
            ci=np.asarray(c['delta_accuracy']['ci95'])*100;v=c['delta_accuracy']['estimate']*100
            ax.plot(ci,[i,i],color='#346f86');ax.scatter([v],[i],color='#346f86')
        ax.set(yticks=np.arange(len(rows)),yticklabels=[c['method'] for c in rows],xlabel='Accuracy change (percentage points)',title=title)
        ax.axvline(0,color='gray',ls='--');ax.grid(axis='x',alpha=.2)
    fig.savefig(dest/'paired_changes.png',dpi=170,bbox_inches='tight');fig.savefig(dest/'paired_changes.pdf',bbox_inches='tight');plt.close(fig)
    parts+=['<img src="paired_changes.png"><p><a href="paired_changes.pdf">配对变化 PDF</a></p>',
            '<p><a href="../geometry_behavior.csv">逐题几何指标 CSV</a> · <a href="../summary.json">完整统计 JSON</a></p>']
    records={r['path']:json.loads(Path(r['path']).read_text()) for r in json.loads((root/'records.json').read_text())}
    by_sid={}
    for path,r in records.items():by_sid.setdefault(r['sample_id'],[]).append((path,r))
    links=[]
    for sid,items in sorted(by_sid.items()):
        body='<a href="../index.html">返回</a><h1>'+html.escape(sid)+'</h1><pre>'+html.escape(items[0][1]['prompt_text'])+'</pre>'
        body+='<p>参考答案：'+html.escape(str(items[0][1]['ground_truth']))+'</p>'
        for path,r in items:
            label=html.escape(r.get('batch',r['split'])+'/'+r['condition']['name'])
            body+=f'<details><summary>{label} · correct={r["correct"]} · {r["length"]} tokens</summary><pre>'+html.escape(r['response_text'])+'</pre></details>'
        page(dest/'questions'/(sid+'.html'),body);links.append(f'<a href="questions/{sid}.html">{html.escape(sid)}</a>')
    parts+=['<h2>逐题全部原文</h2>',' · '.join(links),'<h2>全部修复／损伤的盲核验材料</h2><p>下列页面隐藏方法与正确性标签，A/B顺序预定随机。语义类别仍为待核验；没有把格式修复自动称为推理提升。</p>']
    links=[]
    for case in json.loads((root/'case_review_key.json').read_text()):
        left=records[case['left_record']];right=records[case['right_record']]
        body='<h1>盲核验 '+case['case_id']+'</h1><pre>'+html.escape(left['prompt_text'])+'</pre><p>参考答案：'+html.escape(str(left['ground_truth']))+'</p>'
        for name,r in [('A',left),('B',right)]:body+='<h2>'+name+'</h2><pre>'+html.escape(r['response_text'])+'</pre>'
        page(dest/'blind_cases'/(case['case_id']+'.html'),body);links.append(f'<a href="blind_cases/{case["case_id"]}.html">{case["case_id"]}</a>')
    parts.append(' · '.join(links));parts+=['<h2>解释边界</h2>','<ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in s['limitations'])+'</ul>']
    page(dest/'index.html',''.join(parts))
    write_json(dest/'_SUCCESS.json',dict(complete=True,statistics_audit_sha256=sha(root/'statistics_audit.json'),
        visual_review_pending=True,semantic_case_review_pending=True,
        files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)

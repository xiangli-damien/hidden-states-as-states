"""Compact evidence report; never present teacher-forced KL as answer accuracy."""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_common import sha,write_json


STYLE='body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 22px;line-height:1.7;color:#192733}h1,h2{line-height:1.3}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 11px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}img{max-width:100%}.note{background:#edf4f7;border-left:4px solid #3d788c;padding:16px}details{margin:15px 0}pre{white-space:pre-wrap}a{color:#126d84}'


def table(rows):return '<div class="scroll">'+pd.DataFrame(rows).to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')+'</div>'


def run(root,mode):
    dest=root/('report_abd' if mode=='cached' else 'report');dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists():raise RuntimeError('Completed report; do not overwrite')
    parts=['<h1>重构、冻结迁移与真实答题干预</h1>'];sources={}
    def plot(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=165,bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
        parts.append(f'<img src="{name}.png"><p><a href="{name}.pdf">PDF</a></p>')
    if (root/'cached/audit.json').exists():
        audit=json.loads((root/'cached/audit.json').read_text());s=json.loads((root/'cached/summary.json').read_text())
        assert audit['complete'] and audit['summary_sha256']==sha(root/'cached/summary.json')
        sources['cached_audit']=sha(root/'cached/audit.json')
        parts+=['<h2>A · 固定中心＋局部坐标的功能重构</h2><p>复用已审计数据：block14、生成16token后、width16。KL/NLL衡量参考输出保真，不是自由答题正确率。每个token保留自己的8个坐标。</p>',table(s['reconstruction'])]
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,(name,rows) in zip(axes,pd.DataFrame(s['reconstruction']).groupby('dataset',sort=False)):
            sub=rows.set_index('method').loc[['centroid','shared_8','local_8','shared_64']];x=np.arange(4)
            ax.bar(x,sub.KL,color=['#9ba6ad','#c59a65','#278b80','#5c729d']);ax.vlines(x,sub.KL_low,sub.KL_high,color='black',lw=1)
            ax.set(xticks=x,xticklabels=['Center','Shared8','Local8','Shared64'],ylabel='Next-token KL (nats)',title=name)
        plot(fig,'reconstruction')
        parts+=['<h2>B · MATH prompt-last → GSM8K</h2><p>冻结四个block7/14/21/28，保留原536题确认集。全1319仅为冻结模型的描述性补充。Adapt128来自旧541题适配池，使用目标标签，不是零样本。线性基线固定block28；未比较或选择新的classifier。</p>',table(s['transfer'])]
        parts+=['<h3>支持范围</h3>',table([{k:v for k,v in r.items() if 'occupancy' not in k} for r in s['support']])]
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        rows=[r for r in s['transfer'] if r['scope']=='original_confirmation536'];x=np.arange(len(rows))
        axes[0].scatter(x,[r['auroc'] for r in rows],color='#278b80');axes[0].vlines(x,[r['auroc_ci95'][0] for r in rows],[r['auroc_ci95'][1] for r in rows])
        axes[0].axhline(.5,color='gray',ls='--');axes[0].set(xticks=x,xticklabels=['Frozen NB','Frozen linear28','Adapt128 NB'],ylabel='Failure AUROC',title='Original confirmation n=536')
        for field,label in [('source_test_coverage','MATH test'),('target_confirmation_coverage','GSM8K confirmation')]:
            axes[1].plot([r['block'] for r in s['support']],[r[field] for r in s['support']],'o-',label=label)
        axes[1].set(xlabel='Block output',ylabel='Fraction below source-val distance q95',ylim=(0,1.05),title='Support coverage; no sample exclusions');axes[1].legend()
        plot(fig,'transfer')
        parts+=['<h2>D · 训练几何匹配与持出样本流</h2><p>四个间隔block的prompt-last地图。匹配只依赖训练中心；机会水平使用持出区域边际占用。样本流的一致性不等于功能身份或因果迁移。</p>',table([{k:v for k,v in r.items() if k!='mapping_pairs'} for r in s['flow']])]
    else:parts.append('<p>A/B/D CPU汇总尚未完成。</p>')
    if mode=='cached':
        parts.append('<h2>C · 真实答题干预</h2><p>单独GPU队列正在进行；此页不提前展示或声称干预收益。</p>')
    else:
        audit=json.loads((root/'steering_statistics_audit.json').read_text());s=json.loads((root/'steering_summary.json').read_text())
        assert audit['complete'] and audit['source_summary_sha256']==sha(root/'steering_summary.json')
        sources['steering_statistics_audit']=sha(root/'steering_statistics_audit.json')
        parts+=['<h2>C · 真实MATH自由生成</h2><p>自然生成16token后，在block14最后4个生成位置只修改一次；每条件新cache，随后自由生成至结束。256或128题数量在测试前按预算冻结。测试随机三seed在题内平均。</p>',
                '<div class="note">'+html.escape(s['selection']['reason'])+'</div>',table(s['rows']),table(s['paired_controls'])]
        page_dir=dest/'questions';page_dir.mkdir(exist_ok=True)
        ids=[]
        for stage in ['validation']+(['test'] if s['selection']['run_test'] else []):
            rc=json.loads((root/f'{stage}_SUCCESS.json').read_text())
            for rel in rc['records']:
                r=json.loads((root/rel).read_text());sid=r['sample_id']
                if sid not in ids:ids.append(sid)
        links=[]
        for sid in ids:
            rows=[]
            for p in sorted((root/'outputs'/sid).glob('*.json')):
                if p.name=='prefix.json' or p.name.endswith('.receipt.json'):continue
                r=json.loads(p.read_text());rows.append(r)
            body='<a href="../index.html">返回报告</a><h1>'+html.escape(sid)+'</h1><pre>'+html.escape(rows[0]['prompt_text'])+'</pre>'
            body+='<p>Ground truth: '+html.escape(rows[0]['ground_truth'])+'</p>'
            for r in rows:
                body+='<details open><summary>'+html.escape(r['condition']['name'])+f' · correct={r["correct"]} · {r["length"]} tokens · {r["finish_reason"]}</summary><pre>'+html.escape(r['response_text'])+'</pre></details>'
            (page_dir/(sid+'.html')).write_text('<!doctype html><meta charset="utf-8"><style>'+STYLE+'</style>'+body)
            links.append(f'<a href="questions/{sid}.html">{sid}</a>')
        parts+=['<h3>逐题原文与全部条件</h3>',' · '.join(links)]
    parts.append('<p>区间以问题为单位，逐项95%，未做多重比较校正。保留所有阴性结果；没有达到纠错门槛时不扩大参数搜索。</p>')
    (dest/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HSS final experiments</title><style>'+STYLE+'</style>'+''.join(parts)+'</html>')
    write_json(dest/'_SUCCESS.json',{'complete':True,'sources':sources,'mode':mode,
        'visual_review_pending':True,'files':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--mode',choices=['cached','final'],required=True);a=p.parse_args();run(a.root,a.mode)

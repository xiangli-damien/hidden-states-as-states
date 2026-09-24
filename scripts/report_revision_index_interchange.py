"""Report audited free-generation outcomes, including capability failures."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from revision_common import sha,write_json


def interval(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n':0,'estimate':None,'low':None,'high':None}
    ix=np.random.default_rng(42).integers(len(x),size=(2000,len(x)))
    lo,hi=np.quantile(x[ix].mean(1),[.025,.975])
    return {'n':len(x),'estimate':float(x.mean()),'low':float(lo),'high':float(hi)}


def run(root):
    audit=json.loads((root/'validation_audit.json').read_text());assert audit['passed']
    assert audit['stage_receipt_sha256']==sha(root/'validation/_SUCCESS.json')
    dest=root/'report';dest.mkdir(exist_ok=True);(dest/'questions').mkdir(exist_ok=True)
    gate=audit['capability_gate'];records=[];rows=[];pairs=[]
    if gate['passed']:
        test=json.loads((root/'test_audit.json').read_text());assert test['passed']
        assert test['stage_receipt_sha256']==sha(root/'test/_SUCCESS.json')
        receipt=json.loads((root/'test/_SUCCESS.json').read_text())
        for name,digest in receipt['files'].items():assert sha(root/'test'/name)==digest
        records=[json.loads(p.read_text()) for p in sorted((root/'test/cases').glob('*.json'))]
        assert len(records)==48
        methods=['identity','full_donor','local8','shared8']
        for width in [16,4]:
            values={}
            for method in methods:
                sub={r['case_id']:next(p for p in r['patches'] if p['width']==width and p['method']==method)
                     for r in records if r['eligible']}
                values[method]=sub
                for metric in ['joint_success','target_counter_sequence','new_symbols_preserved','format_valid','ordinary_recipient_correct']:
                    x=[sub[cid]['score'][metric] for cid in sorted(sub)]
                    rows.append({'width':width,'method':method,'metric':metric,'all_candidate_pairs':48,
                        'eligible_pairs':len(sub),'coverage':len(sub)/48,
                        'coverage_adjusted_success':sum(x)/48,**interval(x)})
            for metric in ['joint_success','target_counter_sequence','new_symbols_preserved']:
                a,b=values['local8'],values['shared8'];assert set(a)==set(b)
                delta=[int(a[k]['score'][metric])-int(b[k]['score'][metric]) for k in sorted(a)]
                pairs.append({'width':width,'method_a':'local8','method_b':'shared8','metric':metric,
                    'primary':width==16 and metric=='joint_success',**interval(delta)})
    write_json(dest/'summary.json',rows);write_json(dest/'paired.json',pairs)
    style='<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}pre{white-space:pre-wrap}.scroll{overflow:auto}</style>'
    parts=['<!doctype html><meta charset="utf-8"><title>序列下标交换</title>'+style,
        '<h1>下标计数进度能否交换，同时保留字母？</h1>',
        '<p>Qwen2-7B-Instruct · block14 · 单一合成任务 · 固定MATH local/shared8方向 · greedy repetition_penalty=1.0。这里的8维是传入donor差值的维度，recipient完整状态仍保留；不是8维压缩重构。</p>',
        '<p>供给合法的18-token assistant前缀，在末尾16或4个位置干预后真实自由生成。目标是donor的第5–7项下标；非目标是新生成第6、7项的recipient字母。第5项字母已在供给前缀，不能计入保持率。</p>',
        '<h2>预先固定的能力门槛</h2><pre>'+html.escape(json.dumps(gate,indent=2))+'</pre>']
    if not gate['passed']:
        parts.append('<p><b>能力门槛未通过；测试集没有运行，不能得出方向是否可用的结论。</b>保留失败，不扩展模板、rank或样本。</p>')
    else:
        parts+=['<h2>主对比：width16 local8−shared8 联合成功率</h2>',
                pd.DataFrame([r for r in pairs if r['primary']]).to_html(index=False),
                '<h2>全部条件与覆盖率</h2><p>条件成功率只针对双方原任务能力通过且对齐的pair；coverage_adjusted_success使用全部48对作分母，不合格pair计为未证明成功。问题对bootstrap、逐项区间；单模板结果不能外推通用数学推理。</p>',
                '<div class="scroll">'+pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>']
    links=[]
    for stage in ['validation','test'] if gate['passed'] else ['validation']:
        for path in sorted((root/stage/'cases').glob('*.json')):
            r=json.loads(path.read_text());cid=r['case_id'];c=r['case']
            content=['<!doctype html><meta charset="utf-8">'+style,'<a href="../index.html">返回</a>',
                '<h1>'+html.escape(stage+' '+cid)+'</h1>','<pre>'+html.escape(json.dumps(c,indent=2))+'</pre>']
            for side in ['recipient','donor']:
                content+=['<h2>'+side+'</h2>']
                for kind in ['free','prefixed']:
                    content+=['<h3>'+kind+'</h3><pre>'+html.escape(r[side][kind]['text'])+'</pre>']
            for p in r['patches']:
                content+=['<h3>'+html.escape(f'width{p["width"]} {p["method"]}')+'</h3><pre>'+html.escape(p['output']['text'])+'</pre>',
                          '<pre>'+html.escape(json.dumps(p['score'],indent=2))+'</pre>']
            (dest/'questions'/f'{cid}.html').write_text('\n'.join(content))
            links.append('<li><a href="questions/'+cid+'.html">'+stage+' '+cid+'</a></li>')
    parts+=['<h2>原始自由生成与所有失败</h2><ul>',*links,'</ul>']
    (dest/'index.html').write_text('\n'.join(parts))
    files=[p for p in dest.rglob('*') if p.is_file() and p.name!='_SUCCESS.json']
    write_json(dest/'_SUCCESS.json',{'complete':True,'validation_gate_passed':gate['passed'],
        'validation_audit_sha256':sha(root/'validation_audit.json'),
        'test_audit_sha256':sha(root/'test_audit.json') if gate['passed'] else None,
        'primary':[r for r in pairs if r['primary']],
        'files':{str(p.relative_to(dest)):sha(p) for p in files}})
    print(json.dumps({'gate':gate,'primary':[r for r in pairs if r['primary']]},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)

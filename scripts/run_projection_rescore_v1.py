"""Versioned, all-record scoring sensitivity audit; CPU only, no generation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import projection_rescore_v1 as scoring
from revision_common import sha, write_json


def read(path): return json.loads(Path(path).read_text())


def location(remote, root, primary):
    p = Path(remote)
    for name, dest in [(root.name, root), (primary.name, primary)]:
        prefix = Path('/lambda/nfs/dami/hss') / name
        if p.is_relative_to(prefix): return dest / p.relative_to(prefix)
    if p.is_file(): return p
    raise ValueError('unknown source root: ' + remote)


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and len(a)
    delta = a-b
    rng = np.random.default_rng(9242026)
    boot = [float(delta[rng.integers(0, len(a), len(a))].mean()) for _ in range(2000)]
    return dict(n=len(a), method_accuracy=float(a.mean()), control_accuracy=float(b.mean()),
                delta=float(delta.mean()), ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                repair=int(((a==1)&(b==0)).sum()), damage=int(((a==0)&(b==1)).sum()))


def run(root, primary, out, expected):
    import sympy
    if out.exists(): raise RuntimeError('Output exists; preserve it and use another version namespace')
    manifest = read(root/'records.json')
    assert len(manifest) == expected and len({x['path'] for x in manifest}) == expected
    records = {}
    for item in manifest:
        p = location(item['path'], root, primary)
        assert sha(p) == item['sha256'], item['path']
        records[item['path']] = read(p)
    plan = dict(version=scoring.VERSION, started_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={str(Path(p).resolve()):sha(p) for p in [__file__,scoring.__file__]},
        input_manifest_sha256=sha(root/'records.json'), original_summary_sha256=sha(root/'summary.json'),
        expected_records=expected, sympy_version=sympy.__version__,numpy_version=np.__version__,
        python=sys.version, purpose='Post-hoc automatic scoring sensitivity; NOT full semantic adjudication',
        protocol=['All input records and all experimental conditions; no label-dependent extraction.',
                  'Balanced last box, then final answer marker, final conclusion, terminal math, legacy fallback.',
                  'Numeric rel_tol=0.001 abs_tol=1e-6; symbolic equivalence with one-second timeout.',
                  'Unknown/prose/timeout extraction cases explicitly flagged. No method/label used to choose a candidate.',
                  'No raw labels, selection, question split, scientific source or original report changes.',
                  'Review-inspired rules are post-hoc, not an independently validated new primary endpoint.',
                  'Intervals conditional on automatic labels; extraction uncertainty not included.'])
    out.mkdir(parents=True)
    write_json(out/'plan.json',plan)
    rows=[];cache={};groups=defaultdict(list)
    # Do not inspect aggregate outcomes until every source record has been processed.
    with (out/'scores.jsonl').open('x') as stream:
        for path, raw in records.items():
            task='gsm8k' if raw['split']=='transfer' else 'math'
            key=(raw['response_text'],str(raw['ground_truth']),task,raw['parsed_answer'])
            if key not in cache:
                extraction=scoring.extract(raw['response_text'],task,raw['parsed_answer'])
                correct,flags=scoring.match(extraction['candidate'],str(raw['ground_truth']),task)
                cache[key]=(extraction,correct,flags)
            extraction,correct,flags=cache[key]
            row=dict(path=path,source_sha256=sha(location(path,root,primary)),sample_id=raw['sample_id'],
                split=raw['split'],condition=raw['condition']['name'],task=task,
                original_correct=bool(raw['correct']),new_correct=bool(correct),
                original_extracted=raw['parsed_answer'],new_extracted=extraction['candidate'],
                rule=extraction['rule'],flags=extraction['flags']+flags,
                finish_reason=raw['finish_reason'],final_answer_only=True)
            if raw['finish_reason']=='length':row['flags'].append('truncated_response')
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            group=(raw['split'],str(Path(path).parent.parent.name),raw['condition']['name'])
            # Primary outputs are directly under outputs/sample, while v2 groups
            # include their own Dxx/transfer directory; keep numerical paths distinct.
            groups[group].append(row)
    bypath={r['path']:r for r in rows}
    comparisons=[]
    for c in read(root/'summary.json')['comparisons']:
        paths=[(p['method_record'],p['control_record']) for p in c['per_question']]
        comparisons.append(dict(group=c['group'],method=c['method'],control=c['control'],
            original=paired([bypath[a]['original_correct'] for a,b in paths],[bypath[b]['original_correct'] for a,b in paths]),
            rescored=paired([bypath[a]['new_correct'] for a,b in paths],[bypath[b]['new_correct'] for a,b in paths]),
            flagged_pairs=sum(bool(bypath[a]['flags'] or bypath[b]['flags']) for a,b in paths)))
    # Include the primary shared8 and three-seed-averaged random controls as well.
    test=defaultdict(dict)
    for r in rows:
        if r['split']=='test':test[r['sample_id']][r['condition']]=r
    for control in ['shared8','random_mean3']:
        a=[];b=[];old_a=[];old_b=[]; flagged=0
        for sid, members in sorted(test.items()):
            cs=[members[control]] if control=='shared8' else [members['random_'+str(s)] for s in [42,137,271]]
            a.append(members['c1_1.0']['new_correct']);old_a.append(members['c1_1.0']['original_correct'])
            b.append(np.mean([r['new_correct'] for r in cs]));old_b.append(np.mean([r['original_correct'] for r in cs]))
            flagged+=bool(members['c1_1.0']['flags'] or any(r['flags'] for r in cs))
        if a:comparisons.append(dict(group='MATH_primary256_supplement',method='local8',control=control,
            original=paired(old_a,old_b),rescored=paired(a,b),flagged_pairs=flagged,
            note='Random seeds averaged within question; binary repair/damage counts not defined for random mean.'))
        if a and control=='random_mean3':
            for k in ['original','rescored']:
                comparisons[-1][k].pop('repair');comparisons[-1][k].pop('damage')
    changed=[r for r in rows if r['original_correct']!=r['new_correct']]
    summary=dict(complete=True,scope='All frozen report manifest records',records=len(rows),
        posthoc_automatic_sensitivity_only=True,full_semantic_rescoring_complete=False,
        plan_sha256=sha(out/'plan.json'),scores_sha256=sha(out/'scores.jsonl'),
        changed_labels=len(changed),false_to_true=sum(r['new_correct'] for r in changed),
        true_to_false=sum(not r['new_correct'] for r in changed),
        flagged_records=sum(bool(r['flags']) for r in rows),flag_counts=dict(Counter(f for r in rows for f in r['flags'])),
        rules=dict(Counter(r['rule'] for r in rows)),comparisons=comparisons,
        strata=[dict(split=k[0],batch=k[1],condition=k[2],n=len(v),original_correct=sum(x['original_correct'] for x in v),new_correct=sum(x['new_correct'] for x in v),flagged=sum(bool(x['flags']) for x in v)) for k,v in groups.items()],
        limitations=plan['protocol'],input_manifest_sha256=plan['input_manifest_sha256'])
    write_json(out/'summary.json',summary)
    write_json(out/'changed_records.json',changed)
    # Every score change gets full original response/reference and both extractions.
    esc=lambda x:html.escape(str(x))
    links=[]
    for r in changed:
        raw=records[r['path']];cid=hashlib.sha256(r['path'].encode()).hexdigest()[:16]
        (out/'cases').mkdir(exist_ok=True)
        body='<a href="../index.html">返回</a><h1>'+esc(r['sample_id'])+'</h1>'
        body+='<p>新旧自动评分发生变化，语义复核待完成。</p><h2>题目</h2><pre>'+esc(raw['prompt_text'])+'</pre>'
        body+='<h2>参考</h2><pre>'+esc(raw['ground_truth'])+'</pre><h2>完整回答</h2><pre>'+esc(raw['response_text'])+'</pre>'
        body+='<details><summary>提取与方法记录</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
        page(out/'cases'/f'{cid}.html',body)
        links.append('<tr><td><a href="cases/'+cid+'.html">'+esc(r['sample_id'])+'</a></td><td>'+esc(r['condition'])+'</td><td>'+str(int(r['original_correct']))+' → '+str(int(r['new_correct']))+'</td><td>'+esc(r['original_extracted'])+'</td><td>'+esc(r['new_extracted'])+'</td><td>'+esc(', '.join(r['flags']))+'</td></tr>')
    body='<h1>统一提取规则：全量自动评分敏感性 v1</h1><div class="note">这是事后自动评分敏感性分析，不是已经完成的全量语义裁决。新标签可能仍有提取或匹配错误；原实验、分数、选择和报告不变。</div>'
    body+='<p>全部 '+str(len(rows))+' 条记录，改变 '+str(len(changed))+' 个自动标签；'+str(summary['flagged_records'])+' 条带提取／匹配／截断标记。标记不是确认的错误数。</p>'
    body+='<p><a href="summary.json">完整统计与限制</a> · <a href="scores.jsonl">全部新旧评分</a> · <a href="../semantic_review_v2/index.html">此前182对翻转的AI核验</a></p>'
    body+='<h2>同题差值（百分点）</h2><p>区间按题配对重采样，仅反映抽题不确定性，未包含评分不确定性。验证／消融仍是探索，不重新选优。</p><table><tr><th>范围</th><th>方法／对照</th><th>N</th><th>原Δ</th><th>v1 Δ [95%区间]</th><th>带标记的题对</th></tr>'
    for c in comparisons:
        z=c['rescored'];ci=z['ci95']
        body+='<tr><td>'+esc(c['group'])+'</td><td>'+esc(c['method']+' / '+c['control'])+'</td><td>'+str(z['n'])+'</td><td>'+f"{100*c['original']['delta']:+.2f}"+'</td><td>'+f"{100*z['delta']:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"+'</td><td>'+str(c['flagged_pairs'])+'</td></tr>'
    body+='</table><h2>全部改变标签的记录：待逐项复核</h2><table><tr><th>题目</th><th>条件</th><th>标签</th><th>原提取</th><th>新提取</th><th>标记</th></tr>'+''.join(links)+'</table>'
    page(out/'index.html',body)
    write_json(out/'_SUCCESS.json',dict(complete=True,semantic_review_pending=True,visual_review_pending=True,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({k:summary[k] for k in ['records','changed_labels','false_to_true','true_to_false','flagged_records','full_semantic_rescoring_complete']}))


def page(path,body):
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>评分敏感性 v1</title><style>body{font:16px system-ui;line-height:1.6;max-width:1400px;margin:30px auto;padding:20px;color:#21343f}.note{padding:18px;background:#fff2df;border-left:4px solid #ac7330}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126d84}</style>'+body+'</html>')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-records',type=int,default=2336)
    a=p.parse_args();run(a.root,a.primary_root,a.output,a.expected_records)

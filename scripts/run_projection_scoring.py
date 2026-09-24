"""All-record rescoring with frozen source versions and exact-text review reuse."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import projection_rescore_v1
import projection_rescore_v2 as scoring
import run_projection_rescore_v1 as helpers
import revision_common
from revision_common import sha, write_json


def read(p):return json.loads(Path(p).read_text())
def answer_id(raw):return hashlib.sha256((raw['prompt_text']+'\0'+str(raw['ground_truth'])+'\0'+raw['response_text']).encode()).hexdigest()


def render(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>评分敏感性 v2</title><style>body{font:16px system-ui;line-height:1.7;max-width:1200px;margin:30px auto;padding:20px;color:#20333f}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.note{padding:18px;background:#fff2df;border-left:4px solid #ac7330}a{color:#14677c}</style>'+body+'</html>')


def run(root,primary,out,expected=2336):
    import sympy
    if out.exists():raise RuntimeError('Preserve old output; choose a new version namespace')
    manifest=read(root/'records.json');assert len(manifest)==expected
    raw={}
    for item in manifest:
        p=helpers.location(item['path'],root,primary)
        assert sha(p)==item['sha256'];assert item['path'] not in raw
        raw[item['path']]=read(p)
    sources=[__file__,scoring.__file__,projection_rescore_v1.__file__,helpers.__file__,revision_common.__file__]
    out.mkdir(parents=True)
    plan=dict(version=scoring.VERSION,started_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={str(Path(p).resolve()):sha(p) for p in sources},input_manifest_sha256=sha(root/'records.json'),
        original_summary_sha256=sha(root/'summary.json'),expected_records=expected,
        sympy_version=sympy.__version__,numpy_version=np.__version__,python=sys.version,
        protocol=['Post-hoc automatic scoring sensitivity; original experiment and scores preserved.',
                  'No reference, method or label inputs to extraction.',
                  'Last balanced box; explicit final section; terminal paragraph with last answer-bearing predicate; explicit answer; frozen fallback.',
                  'Exact integer equality; noninteger numeric rel_tol.001 abs_tol1e-6; bounded symbolic equivalence.',
                  'Multiple final values are not reduced to one number; source names and periods in honorifics are not answers.',
                  'All2336 records/all30 comparisons; 2000 paired question bootstraps seed9242026.',
                  'Reused semantic readings require exact prompt/reference/response equality. New answer reading remains separate.',
                  'No new model generations, winner selection, or automatic semantic-correction claims.'])
    write_json(out/'plan.json',plan)
    rows=[];cache={}
    for path,r in raw.items():
        task='gsm8k' if r['split']=='transfer' else 'math';key=(answer_id(r),task,r['parsed_answer'])
        if key not in cache:
            ext=scoring.extract(r['response_text'],task,r['parsed_answer']);correct,flags=scoring.match(ext['candidate'],str(r['ground_truth']),task)
            cache[key]=(ext,correct,flags)
        ext,correct,flags=cache[key]
        rows.append(dict(path=path,source_sha256=sha(helpers.location(path,root,primary)),sample_id=r['sample_id'],split=r['split'],
            condition=r['condition']['name'],task=task,answer_id=answer_id(r),original_correct=bool(r['correct']),new_correct=bool(correct),
            original_extracted=r['parsed_answer'],new_extracted=ext['candidate'],rule=ext['rule'],
            flags=ext['flags']+flags+(['truncated_response'] if r['finish_reason']=='length' else []),finish_reason=r['finish_reason']))
    (out/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    bypath={r['path']:r for r in rows};contrasts=[]
    for c in read(root/'summary.json')['comparisons']:
        pairs=[(bypath[p['method_record']],[bypath[p['control_record']]]) for p in c['per_question']]
        contrasts.append((c['group'],c['method'],c['control'],pairs))
    test=defaultdict(dict)
    for r in rows:
        if r['split']=='test':test[r['sample_id']][r['condition']]=r
    for control,keys in [('shared8',['shared8']),('random_mean3',['random_42','random_137','random_271'])]:
        if test:contrasts.append(('MATH_primary256_supplement','local8',control,[(v['c1_1.0'],[v[k] for k in keys]) for _,v in sorted(test.items())]))
    comparisons=[]
    for group,method,control,pairs in contrasts:
        c=dict(group=group,method=method,control=control,flagged_pairs=sum(bool(a['flags'] or any(b['flags'] for b in bs)) for a,bs in pairs))
        for version,field in [('original','original_correct'),('rescored','new_correct')]:
            c[version]=helpers.paired([a[field] for a,bs in pairs],[np.mean([b[field] for b in bs]) for a,bs in pairs])
            if control=='random_mean3':
                c[version].pop('repair');c[version].pop('damage')
        comparisons.append(c)
    known={}
    review=read(root/'semantic_review_v2/summary.json')
    for note in review['rows']:
        for side,letter in [('left','A'),('right','B')]:
            p=note[side+'_record'];aid=answer_id(raw[p]);value=note['final_answer_correct_'+letter]
            assert aid not in known or known[aid]['correct']==value
            known[aid]=dict(correct=value,note=note['note'],case_id=note['case_id'])
    changed=[r for r in rows if r['original_correct']!=r['new_correct']]
    conflicts=[dict(r,prior_review=known[r['answer_id']]) for r in rows if r['answer_id'] in known and r['new_correct']!=known[r['answer_id']]['correct']]
    new_answers={};review_key=[]
    # All changed-label answers not already read, irrespective of method or effect.
    for r in changed:
        aid=r['answer_id']
        if aid in known:continue
        rr=raw[r['path']]
        new_answers[aid]=dict(answer_id=aid,prompt=rr['prompt_text'],reference=rr['ground_truth'],response=rr['response_text'])
        review_key.append(dict(answer_id=aid,path=r['path'],new_correct=r['new_correct'],original_correct=r['original_correct']))
    packet=dict(scope='All changed-label full answers without an exact prior reading; no methods or automatic labels',
        independent_human_review=False,answers=[new_answers[k] for k in sorted(new_answers)])
    write_json(out/'review_packet.json',packet);write_json(out/'review_key.json',review_key)
    write_json(out/'prior_review_conflicts.json',conflicts)
    write_json(out/'changed_records.json',changed)
    summary=dict(complete=True,scope='All frozen manifest records; separate post-hoc automatic scorer',records=len(rows),
        full_semantic_rescoring_complete=False,plan_sha256=sha(out/'plan.json'),scores_sha256=sha(out/'scores.jsonl'),
        changed_labels=len(changed),false_to_true=sum(r['new_correct'] for r in changed),true_to_false=sum(not r['new_correct'] for r in changed),
        flagged_records=sum(bool(r['flags']) for r in rows),flag_counts=dict(Counter(x for r in rows for x in r['flags'])),
        comparisons=comparisons,prior_review_conflict_records=len(conflicts),
        changed_records_prior_read=sum(r['answer_id'] in known for r in changed),new_full_answers_to_read=len(new_answers),
        prior_reading_sha256=sha(root/'semantic_review_v2/summary.json'),review_packet_sha256=sha(out/'review_packet.json'),
        limitations=plan['protocol'])
    write_json(out/'summary.json',summary)
    e=lambda x:html.escape(str(x))
    body='<h1>统一评分 v2：自动重算与逐题核验进度</h1><div class="note">全条件自动重算不等于全量语义裁决。原分数、实验选择与旧版规则全部保留。最终答案判分也不证明推导正确。</div>'
    body+='<p>'+str(len(rows))+'条记录；'+str(len(changed))+'个标签改变；与原封存阅读不一致的源记录'+str(len(conflicts))+'条。改变标签的回答中，仍有'+str(len(new_answers))+'份不同全文需要新阅读。</p>'
    body+='<p><a href="summary.json">统计与来源</a> · <a href="scores.jsonl">全部新旧评分</a> · <a href="review_packet.json">待核验全文（隐藏方法）</a> · <a href="prior_review_conflicts.json">与旧阅读的分歧</a></p>'
    body+='<h2>自动分数下的同题差值（百分点）</h2><p>95%区间仅为问题重采样的不确定性，不包含评分误差。原验证和消融仍为探索性。</p><table><tr><th>范围</th><th>方法／对照</th><th>N</th><th>原Δ</th><th>v2 Δ [95%区间]</th></tr>'
    for c in comparisons:
        z=c['rescored'];ci=z['ci95'];body+='<tr><td>'+e(c['group'])+'</td><td>'+e(c['method']+' / '+c['control'])+'</td><td>'+str(z['n'])+'</td><td>'+f"{100*c['original']['delta']:+.2f}"+'</td><td>'+f"{100*z['delta']:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"+'</td></tr>'
    body+='</table><h2>全部改变标签的回答</h2><ul>'
    for r in changed:
        cid=hashlib.sha256(r['path'].encode()).hexdigest()[:16];rr=raw[r['path']]
        detail='<a href="../index.html">返回</a><h1>'+e(r['sample_id'])+'</h1><pre>'+e(rr['prompt_text'])+'</pre><p>参考：'+e(rr['ground_truth'])+'</p><pre>'+e(rr['response_text'])+'</pre><details><summary>提取与方法记录</summary><pre>'+e(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
        render(out/'cases'/f'{cid}.html',detail)
        body+='<li><a href="cases/'+cid+'.html">'+e(r['sample_id']+' / '+r['condition'])+'</a></li>'
    render(out/'index.html',body+'</ul>')
    write_json(out/'_SUCCESS.json',dict(complete=True,semantic_review_pending=True,visual_review_pending=True,files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({k:summary[k] for k in ['records','changed_labels','prior_review_conflict_records','new_full_answers_to_read']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--primary-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root,a.primary_root,a.output)

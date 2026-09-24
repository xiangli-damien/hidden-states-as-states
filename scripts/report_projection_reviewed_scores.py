"""Join sealed AI readings uniformly by exact text; preserve automatic scores.

This produces a partial-review sensitivity analysis, not a semantic gold set.
It neither generates answers nor judges their mathematical correctness.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import run_projection_rescore_v1 as helpers


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(prompt, reference, response):
    return hashlib.sha256((prompt+'\0'+str(reference)+'\0'+response).encode()).hexdigest()


def add_note(known, aid, value, note, provenance):
    if not isinstance(value, bool):
        raise ValueError('Semantic verdict must be an explicit boolean')
    if aid in known and known[aid]['correct'] != value:
        raise ValueError('Contradictory readings for identical full text')
    item = known.setdefault(aid, dict(correct=value, notes=[], provenance=[]))
    if note not in item['notes']:
        item['notes'].append(note)
    if provenance not in item['provenance']:
        item['provenance'].append(provenance)


def validate_packet(packet, notes, packet_sha):
    assert notes['complete'] and notes['packet_sha256'] == packet_sha
    assert notes['independent_human_review'] is False
    answers = {a['answer_id']: a for a in packet['answers']}
    assert len(answers) == len(packet['answers'])
    assert set(answers) == set(notes['notes'])
    for i, a in enumerate(packet['answers']):
        assert identity(a['prompt'], a['reference'], a['response']) == a['answer_id']
        n = notes['notes'][a['answer_id']]
        assert n['full_answer_read'] and n['packet_index'] == i
        assert isinstance(n['final_answer_correct'], bool) and n['note'].strip()
    return answers


def render(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    style = 'body{font:16px system-ui;line-height:1.7;max-width:1220px;margin:30px auto;padding:20px;color:#20333f}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.note{padding:18px;background:#fff2df;border-left:4px solid #ac7330}a{color:#14677c}details{margin:14px 0}'
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>评分核验与敏感性分析</title><style>'+style+'</style>'+body+'</html>')


def run(root, primary, out):
    if out.exists():
        raise RuntimeError('Preserve existing versioned analysis')
    scored = root/'scoring_sensitivity_v2'
    rows = [json.loads(x) for x in (scored/'scores.jsonl').read_text().splitlines()]
    manifest = read(root/'records.json')
    assert len(rows) == len(manifest) == len({x['path'] for x in rows})
    raw = {}
    for m in manifest:
        p = helpers.location(m['path'], root, primary)
        assert sha(p) == m['sha256']
        raw[m['path']] = read(p)
    assert set(raw) == {x['path'] for x in rows}
    known = {}
    for n in read(root/'semantic_review_v2/summary.json')['rows']:
        for side, letter in [('left', 'A'), ('right', 'B')]:
            r = raw[n[side+'_record']]
            aid = identity(r['prompt_text'], r['ground_truth'], r['response_text'])
            add_note(known, aid, n['final_answer_correct_'+letter], n['note'], 'prior:'+n['case_id'])
    prior_ids = set(known)
    packet = read(scored/'review_packet.json')
    notes = read(scored/'semantic_notes_blinded.json')
    new = validate_packet(packet, notes, sha(scored/'review_packet.json'))
    changed = [r for r in rows if r['original_correct'] != r['new_correct']]
    expected = {(r['answer_id'], r['path'], r['new_correct'], r['original_correct']) for r in changed if r['answer_id'] not in prior_ids}
    key = read(scored/'review_key.json')
    assert expected == {(r['answer_id'], r['path'], r['new_correct'], r['original_correct']) for r in key}
    assert set(new) == {x[0] for x in expected}
    for aid, n in notes['notes'].items():
        add_note(known, aid, n['final_answer_correct'], n['note'], 'new_packet:'+str(n['packet_index']))
    v1 = {r['path']: r for r in map(json.loads, (root/'scoring_sensitivity_v1/scores.jsonl').read_text().splitlines())}
    assert set(v1) == set(raw)
    output = []
    for r in rows:
        rr = raw[r['path']]
        aid = identity(rr['prompt_text'], rr['ground_truth'], rr['response_text'])
        assert aid == r['answer_id'] and rr['correct'] == r['original_correct']
        assert r['source_sha256'] == sha(helpers.location(r['path'], root, primary))
        output.append(dict(r, v1_correct=v1[r['path']]['new_correct'],
                           reviewed_correct=known[aid]['correct'] if aid in known else r['new_correct'],
                           label_source='sealed_AI_full_answer_reading' if aid in known else 'automatic_v2_unreviewed'))
    assert all(r['answer_id'] in known for r in changed)
    bypath = {r['path']: r for r in output}
    contrasts = []
    for c in read(root/'summary.json')['comparisons']:
        contrasts.append((c['group'], c['method'], c['control'],
                          [(p['method_record'], [p['control_record']]) for p in c['per_question']]))
    test = defaultdict(dict)
    for r in output:
        if r['split'] == 'test':
            test[r['sample_id']][r['condition']] = r['path']
    for control, keys in [('shared8', ['shared8']), ('random_mean3', ['random_42', 'random_137', 'random_271'])]:
        contrasts.append(('MATH_primary256_supplement', 'local8', control,
                          [(v['c1_1.0'], [v[k] for k in keys]) for _, v in sorted(test.items())]))
    comparisons = []
    for group, method, control, pairs in contrasts:
        c = dict(group=group, method=method, control=control,
                 fully_AI_read_pairs=sum(all(bypath[p]['answer_id'] in known for p in [a]+bs) for a, bs in pairs),
                 pairs=[dict(method_record=a, control_records=bs) for a, bs in pairs])
        for version, field in [('original','original_correct'), ('v1','v1_correct'), ('v2','new_correct'), ('partial_AI_review','reviewed_correct')]:
            c[version] = helpers.paired([bypath[a][field] for a,bs in pairs],
                                       [sum(bypath[b][field] for b in bs)/len(bs) for a,bs in pairs])
            if control == 'random_mean3':
                c[version].pop('repair'); c[version].pop('damage')
        comparisons.append(c)
    reviewed = [r for r in output if r['answer_id'] in known]
    disagreement = [r for r in reviewed if r['new_correct'] != r['reviewed_correct']]
    inputs = [root/'records.json', root/'summary.json', root/'semantic_review_v2/summary.json',
              root/'scoring_sensitivity_v1/scores.jsonl']
    inputs += [scored/p for p in ['scores.jsonl','summary.json','statistics_audit.json','review_packet.json','review_key.json','semantic_notes_blinded.json']]
    summary = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc partial-AI-review sensitivity; unreviewed records retain automatic v2 labels',
        full_semantic_rescoring_complete=False, independent_human_review=False,
        input_hashes={str(p.resolve()):sha(p) for p in inputs},
        source_hashes={str(Path(p).resolve()):sha(p) for p in [__file__, helpers.__file__]},
        records=len(output), unique_full_answers=len({r['answer_id'] for r in output}),
        AI_reviewed_records=len(reviewed), AI_reviewed_unique_answers=len(known),
        new_full_answers_read=len(new), changed_automatic_records=len(changed),
        changed_automatic_records_with_reading=len(changed),
        v2_disagreement_records=len(disagreement), v2_disagreement_unique_answers=len({r['answer_id'] for r in disagreement}),
        unreviewed_records=len(output)-len(reviewed),
        labels_changed_vs_original=sum(r['reviewed_correct'] != r['original_correct'] for r in output),
        comparisons=comparisons,
        limitations=['AI reading is fallible and was targeted by automatic label changes, not a representative audit sample.',
                     'Methods/labels were hidden in the new packet; earlier experiment aggregates were known. No independent human review.',
                     'Labels assess final mathematical answer, not proof validity or formatting compliance.',
                     'Uniform exact-text overrides applied to all conditions; no method-specific edits, new selection or raw-score replacement.',
                     '2000 paired question bootstrap draws, seed9242026; pointwise intervals conditional on labels, not scoring uncertainty.',
                     'Exploratory32 and originalvalidation64 are historical selection data; not new confirmation.'])
    out.mkdir(parents=True)
    def dump(p, value):
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (out/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    dump(out/'readings.json', known)
    dump(out/'v2_disagreements.json', disagreement)
    summary['scores_sha256'] = sha(out/'scores.jsonl')
    summary['readings_sha256'] = sha(out/'readings.json')
    dump(out/'summary.json', summary)
    e = lambda x: html.escape(str(x))
    body = '<h1>全条件评分复算：部分全文核验敏感性分析</h1><div class="note">这不是全量人工正确率。原评分和实验配置完整保留；AI 已阅读的全文使用封存判断，其余仍用自动 v2。区间不包含评分误差，最终答案正确也不代表推导正确。</div>'
    body += f'<p>全部 {len(output):,} 条记录；{len(reviewed)} 条对应 {len(known)} 份已阅读全文；其余 {len(output)-len(reviewed)} 条尚未逐题核验。新增阅读 {len(new)} 份，覆盖全部 {len(changed)} 条自动标签变化；另有 {len(disagreement)} 条 v2 与阅读判断不一致。</p>'
    body += '<p><a href="summary.json">完整统计与来源</a> · <a href="scores.jsonl">逐记录标签及来源</a> · <a href="readings.json">全部阅读笔记</a> · <a href="../scoring_sensitivity_v2/index.html">保留的纯自动 v2</a></p>'
    body += '<h2>主要同题对照</h2><p>表内“部分核验”是诊断版本，不替换预先冻结的主指标，也不用于重新挑选配置。计数和百分点均按问题配对。</p>'
    body += '<table><tr><th>范围／方法对照</th><th>N</th><th>原 Δ</th><th>纯自动 v2 Δ</th><th>部分核验：方法／对照正确数</th><th>部分核验 Δ [95%区间]</th></tr>'
    for c in comparisons:
        if c['group'] == 'MATH_exploratory32': continue
        z=c['partial_AI_review']; ci=z['ci95']; count=lambda x:f'{x*z["n"]:.2f}'.rstrip('0').rstrip('.')
        body += '<tr><td>'+e(c['group']+' · '+c['method']+' / '+c['control'])+'</td><td>'+str(z['n'])+'</td><td>'+f'{100*c["original"]["delta"]:+.2f}'+'</td><td>'+f'{100*c["v2"]["delta"]:+.2f}'+'</td><td>'+count(z['method_accuracy'])+' / '+count(z['control_accuracy'])+'</td><td>'+f'{100*z["delta"]:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]'+'</td></tr>'
    body += '</table><p>random_mean3 的对照数是三个随机种子的平均值。全部30项比较及四个评分版本在完整统计文件中。</p><h2>全部已阅读全文</h2><p>以下每份全文只出现一次；完全相同的题目、参考和回答可复用于多个条件。旧笔记可能谈及成对回答，保留其原上下文与来源。</p><ul>'
    grouped=defaultdict(list)
    for r in reviewed: grouped[r['answer_id']].append(r)
    for aid, members in sorted(grouped.items()):
        rr=raw[members[0]['path']]; n=known[aid]
        detail='<a href="../index.html">返回</a><h1>'+e(members[0]['sample_id'])+'</h1><p>AI最终答案判断：'+e(n['correct'])+'（不等于推导验证）</p><pre>'+e(rr['prompt_text'])+'</pre><p>参考：'+e(rr['ground_truth'])+'</p><h2>阅读笔记</h2><pre>'+e('\n\n'.join(n['notes']))+'</pre><h2>完整回答</h2><pre>'+e(rr['response_text'])+'</pre><details><summary>源记录、条件和标签</summary><pre>'+e(json.dumps(members,ensure_ascii=False,indent=2))+'</pre></details>'
        render(out/'answers'/(aid+'.html'),detail)
        body+='<li><a href="answers/'+aid+'.html">'+e(members[0]['sample_id']+' · '+aid[:12])+'</a> · '+str(len(members))+' 条源记录</li>'
    render(out/'index.html',body+'</ul>')
    dump(out/'_SUCCESS.json',dict(complete=True,statistical_audit_pending=True,visual_review_pending=True,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({k:summary[k] for k in ['records','AI_reviewed_records','AI_reviewed_unique_answers','v2_disagreement_records','unreviewed_records']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.primary_root,a.output)

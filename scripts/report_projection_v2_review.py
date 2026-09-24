"""Join sealed AI readings of all auto-score flips; never rescore the full set.

The additional134 pairs were read through exact question/reference/answer text
deduplication. This program validates the mapping, not the semantic judgments.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

from revision_common import sha, write_json


def read(path):
    return json.loads(path.read_text())


def classify(method_correct, control_correct):
    if method_correct == control_correct:
        return 'both_correct' if method_correct else 'both_wrong'
    return 'repair' if method_correct else 'damage'


def raw_path(path, root, primary):
    path = Path(path)
    for remote, local in [(root.name, root), (primary.name, primary)]:
        prefix = Path('/lambda/nfs/dami/hss') / remote
        if path.is_relative_to(prefix):
            return local / path.relative_to(prefix)
    raise ValueError(f'Unrecognized raw path: {path}')


def render_page(path, body):
    style = ('body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 24px;line-height:1.65;color:#192733}'
             '.note{padding:16px;background:#edf4f7;border-left:4px solid #247c78;margin:18px 0}'
             'table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}'
             '.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126d84}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                    '<title>HSS v2 semantic review</title><style>' + style + '</style>' + body + '</html>')


def run(root, primary):
    dest = root / 'semantic_review_v2'
    if (dest / 'summary.json').exists():
        raise RuntimeError('Preserve sealed summary; use a new output namespace for revisions')
    packet = read(dest / 'blind_packet.json')
    new = read(dest / 'notes_blinded.json')
    assert new['complete'] and new['source_sha256'] == sha(dest / 'blind_packet.json')
    answers = {a['answer_id']: a for a in packet['answers']}
    for aid, a in answers.items():
        assert hashlib.sha256((a['question'] + '\0' + a['reference'] + '\0' + a['answer']).encode()).hexdigest()[:16] == aid
    by_pair = {p['case_id']: p for p in packet['pairs']}
    notes = {}
    for n in new['rows']:
        p = by_pair[n['case_id']]
        assert n['source_html_sha256'] == p['html_sha256'] == sha(root/'report/blind_cases'/f"{n['case_id']}.html")
        assert n['full_pair_read']
        for side in ['A', 'B']:
            aid = p[side]
            assert n['answer_id_' + side] == aid
            assert n['final_answer_correct_' + side] == new['answer_notes'][aid][0]
        notes[n['case_id']] = n
    assert len(notes) == len(by_pair) == 134
    provenance = [{'kind': 'additional134', 'path': str(dest/'notes_blinded.json'), 'sha256': sha(dest/'notes_blinded.json')}]
    for name, source in [('semantic_review_primary', primary/'paper_primary/blind_cases.json'),
                         ('semantic_review_validation', primary/'semantic_review_validation/blind_cases.json')]:
        path = primary/name/'notes_blinded.json'
        prior = read(path)
        assert prior['complete'] and prior['source_sha256'] == sha(source)
        # Confirm the old A/B orientation against the new rendered blind pages.
        old = {a['case_id']: a for a in read(source)}
        for n in prior['rows']:
            cid = n['case_id']
            assert cid not in notes and n['full_pair_read']
            pre = [html.unescape(x) for x in re.findall(r'<pre>(.*?)</pre>', (root/'report/blind_cases'/f'{cid}.html').read_text(), re.S)]
            assert pre[1:] == [old[cid]['A'], old[cid]['B']]
            notes[cid] = n
        provenance.append({'kind': name, 'path': str(path), 'sha256': sha(path)})
    key = read(root/'case_review_key.json')
    assert len(key) == len(notes) == 182 and {k['case_id'] for k in key} == set(notes)
    manifest = {a['path']: a['sha256'] for a in read(root/'records.json')}
    rows, groups = [], defaultdict(Counter)
    esc = lambda value: html.escape(str(value))
    for k in key:
        n = notes[k['case_id']]
        pre = [html.unescape(x) for x in re.findall(r'<pre>(.*?)</pre>', (root/'report/blind_cases'/f"{k['case_id']}.html").read_text(), re.S)]
        raw = []
        for i, side in enumerate(['left', 'right']):
            p = raw_path(k[side+'_record'], root, primary)
            assert sha(p) == manifest[k[side+'_record']]
            a = read(p)
            assert a['sample_id'] == k['sample_id'] and a['response_text'] == pre[i+1]
            raw.append(a)
        assert raw[0]['correct'] != raw[1]['correct']
        mi = 0 if k['method_side'] == 'left' else 1
        assert ('repair' if raw[mi]['correct'] else 'damage') == k['effect']
        verdict = classify(n['final_answer_correct_A' if mi == 0 else 'final_answer_correct_B'],
                           n['final_answer_correct_B' if mi == 0 else 'final_answer_correct_A'])
        if k['case_id'] in by_pair:
            pp = by_pair[k['case_id']]
            for i, side in enumerate(['A', 'B']):
                assert answers[pp[side]]['answer'] == raw[i]['response_text']
                assert answers[pp[side]]['question'] == pre[0]
                assert answers[pp[side]]['reference'] == str(raw[i]['ground_truth'])
        row = dict(n, **{x:y for x,y in k.items() if x not in ['notes', 'semantic_category', 'review_status']},
                   review_verdict=verdict, auto_extracted_A=raw[0]['normalized_answer'],
                   auto_extracted_B=raw[1]['normalized_answer'], auto_correct_A=raw[0]['correct'],
                   auto_correct_B=raw[1]['correct'], finish_A=raw[0]['finish_reason'], finish_B=raw[1]['finish_reason'])
        rows.append(row)
        groups[(k['group'], k['method'], k['control'])][verdict] += 1
        detail = '<a href="../index.html">返回</a><h1>'+esc(k['sample_id'])+'</h1><div class="note">'+esc(n['note'])+'</div>'
        detail += '<pre>'+esc(pre[0])+'</pre><p>参考答案：'+esc(raw[0]['ground_truth'])+'</p>'
        for i, side in enumerate(['A', 'B']):
            detail += '<h2>'+side+'</h2><pre>'+esc(pre[i+1])+'</pre>'
        detail += '<details><summary>核验记录与封存后揭示的方法</summary><pre>'+esc(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details>'
        render_page(dest/'cases'/f"{k['case_id']}.html", detail)
    total = Counter(r['review_verdict'] for r in rows)
    summary = {'complete': True, 'scope': 'All182 automatic-score flip comparisons, not full-data rescoring',
               'independent_human_review': False, 'full_dataset_rescored': False,
               'unique_new_answer_texts_read': len(answers), 'prior_pairs_reused':48,
               'review_provenance':provenance, 'source_sha256':sha(Path(__file__)),
               'original_statistics_sha256':sha(root/'statistics_audit.json'),
               'case_key_sha256':sha(root/'case_review_key.json'), 'counts':dict(total),
               'groups':[dict(group=g[0],method=g[1],control=g[2],counts=dict(c)) for g,c in groups.items()],
               'limitations':new['limitations'], 'rows':rows}
    write_json(dest/'summary.json',summary)
    body = '<h1>Projection v2：全部自动评分翻转的语义核验</h1>'
    body += '<div class="note"><strong>182/182 对已核验；这不是全数据再评分。</strong><br>Codex AI 阅读全部相关回答，新增134对通过146份完全相同题目／参考／回答文本的去重阅读完成；原48对复用已封存记录。方法与自动标签在新增阅读材料中隐藏，汇总结果此前已知。不是独立人类盲审。</div>'
    body += '<p><a href="../report/index.html">原冻结图表与全部回答</a> · <a href="summary.json">完整核验记录</a></p>'
    body += '<h2>需要怎样解读</h2><p>同一题可能出现在多项比较中，下面的对数不能当作独立样本量。仅检查自动标签翻转，会遗漏两种条件都被误判的情况，因此不能据此重算全256题或64题准确率和置信区间。</p>'
    body += '<p>新GSM8K中，Rosie的正确30分钟被提取成结尾的20英里；Susan的742颗被提取成10条项链；围栏的286被提取成显示数学起始符。真实损伤则包括把同时烹饪算成顺序烹饪，以及反复自我否定后循环截断。最终答对也可能伴随错误推导。</p>'
    body += '<p>例如三角函数回答的嵌套根式最终等于2/5，但前面的恒等式与判别式推导有误；这些情况须区分最终答案与推理质量。部分题目的严格阈值或端点计数还有数据集措辞歧义，已逐题注明。</p>'
    body += '<div class="note">后续仍需独立版本的全部条件评分修正。原生成、自动分数、选优和统计收据保持冻结。当前没有新增GPU实验。</div>'
    body += '<h2>按比较核验：只统计自动翻转的回答对</h2><div class="scroll"><table><tr><th>范围</th><th>方法／对照</th><th>最终答案修复</th><th>最终答案损伤</th><th>两份都正确</th><th>两份都错误</th></tr>'
    for g,c in groups.items():
        body += '<tr><td>'+esc(g[0])+'</td><td>'+esc(g[1]+' / '+g[2])+'</td>'+''.join('<td>'+str(c[x])+'</td>' for x in ['repair','damage','both_correct','both_wrong'])+'</tr>'
    body += '</table></div><h2>全部案例</h2><table><tr><th>题目／比较</th><th>自动 → 核验</th><th>详细记录</th></tr>'
    for r in rows:
        body += '<tr><td><a href="cases/'+r['case_id']+'.html">'+esc(r['sample_id'])+'</a><br>'+esc(r['method']+' / '+r['control'])+'</td><td>'+esc(r['effect']+' → '+r['review_verdict'])+'</td><td>'+esc(r['note'])+'</td></tr>'
    body += '</table>'
    render_page(dest/'index.html',body)
    write_json(dest/'_SUCCESS.json', {'complete':True,'visual_review_pending':True,'files':{str(p.relative_to(dest)):sha(p) for p in [dest/'summary.json',dest/'index.html',*sorted((dest/'cases').glob('*.html'))]}})
    print(json.dumps({'pairs':len(rows),'counts':dict(total),'groups':len(groups),'summary_sha256':sha(dest/'summary.json')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--primary-root',required=True,type=Path)
    a=p.parse_args();run(a.root,a.primary_root)

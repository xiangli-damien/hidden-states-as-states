"""Join already sealed method-masked AI reading notes to the primary case key.

This is a selected-case diagnostic, never a replacement full-test accuracy.
"""
import argparse
import collections
import html
import json
from pathlib import Path
from revision_common import sha, write_json


def run(root):
    dest = root / 'semantic_review_primary'
    if (dest / 'summary.json').exists():
        raise RuntimeError('Preserve sealed review outputs')
    review = json.loads((dest / 'notes_blinded.json').read_text())
    assert review['complete']
    assert review['source_sha256'] == sha(root / 'paper_primary/blind_cases.json')
    cases = {r['case_id']:r for r in json.loads((root / 'paper_primary/case_key.json').read_text())}
    blind = {r['case_id']:r for r in json.loads((root / 'paper_primary/blind_cases.json').read_text())}
    assert set(cases) == set(blind) == {r['case_id'] for r in review['rows']}
    counts = collections.Counter()
    results = []
    for note in review['rows']:
        assert note['full_pair_read']
        key = cases[note['case_id']]
        method = json.loads((root / key['method_record']).read_text())
        baseline = json.loads((root / key['baseline_record']).read_text())
        assert method['correct'] != baseline['correct']
        same_answer_verdict = note['final_answer_correct_A'] == note['final_answer_correct_B']
        counts[(note['category'], key['effect'])] += 1
        results.append(dict(note, **key, same_final_answer_verdict=same_answer_verdict,
                            auto_method_correct=method['correct'], auto_baseline_correct=baseline['correct'],
                            method_extracted=method['normalized_answer'], baseline_extracted=baseline['normalized_answer'],
                            method_record_sha256=sha(root/key['method_record']), baseline_record_sha256=sha(root/key['baseline_record'])))
    conflicts = [r for r in results if r['same_final_answer_verdict']]
    genuine = collections.Counter(r['effect'] for r in results if not r['same_final_answer_verdict'])
    summary = dict(complete=True, reviewer=review['reviewer'], blinding=review['blinding'],
                   reviewed_pairs=len(results), scoring_conflicts=len(conflicts),
                   conflict_auto_repairs=sum(r['effect']=='repair' for r in conflicts),
                   conflict_auto_damages=sum(r['effect']=='damage' for r in conflicts),
                   different_final_answer_repairs=genuine['repair'], different_final_answer_damages=genuine['damage'],
                   scope=review['scope'], all_256_rescored=False, independent_human_review=False,
                   limitation='Only the 33 automatic-label flips were read; unchanged-label pairs can also contain scoring errors. Do not infer corrected full-test accuracy or confidence intervals.',
                   notes_sha256=sha(dest/'notes_blinded.json'), source_sha256=sha(Path(__file__)),
                   primary_summary_sha256=sha(root/'steering_summary.json'), rows=results)
    write_json(dest / 'summary.json', summary)
    esc = lambda x: html.escape(str(x))
    body = '''<h1>Steering 主测试：结果与逐题核查</h1>
<p>Qwen2-7B-Instruct · MATH 256题 · 固定 block14 / t16 / width4 / local8 / alpha1。</p>
<div class="note"><strong>尚未确认稳定的纠错收益。</strong> 冻结自动评分为baseline 129/256、local8 130/256；17题修复、16题损伤，净+0.39个百分点，配对95%区间[-3.91,+4.69]。相对shared8及随机对照的区间也跨零。</div>
<img src="../paper_primary/primary_accuracy.png" alt="Frozen automatic-score primary confirmation">
<p><a href="../paper_primary/index.html">自动评分图表与方法</a> · <a href="../report/index.html">原始审计报告及全部320题</a> · <a href="summary.json">逐题核查 JSON</a></p>
<h2>33个自动评分翻转已全部阅读</h2>
<p>由Codex AI逐字阅读问题、参考答案和两份完整回答；阅读时A/B隐藏方法及自动标签，记录封存后才与方法键对应。不是独立人类复核。</p>
<div class="note"><strong>8/33个翻转是评分提取冲突：</strong> 两份回答实际上都给出正确最终答案，其中6个被自动计为修复、2个计为损伤。排除这8个后，所检查的翻转中有11个最终答案修复、14个损伤。<strong>不能据此改报全256题准确率：</strong> 自动标签未翻转的223对尚未语义复核，也可能有漏判。</div>
<p>提取失败标记为0，只表示提取器返回了非空字符串。它仍可能提取到错误的数字或文字，例如把672美元后的“10年”当答案，或把list/satisfying中的is当结论动词。另有正确最终答案伴随错误推导，不能统称推理修复。</p>
<h2>全部案例</h2><table><thead><tr><th>题目</th><th>原自动效果</th><th>核查类型</th><th>核查结论</th></tr></thead><tbody>'''
    for r in results:
        body += '<tr><td><a href="cases/'+r['case_id']+'.html">'+esc(r['sample_id'])+'</a></td><td>'+esc(r['effect'])+'</td><td>'+esc(r['category'])+'</td><td>'+esc(r['note'])+'</td></tr>'
        x = blind[r['case_id']]
        page = '<a href="../index.html">返回核查</a><h1>'+esc(r['sample_id'])+'</h1><p>'+esc(r['note'])+'</p><pre>'+esc(x['prompt'])+'</pre><p>参考答案：'+esc(x['ground_truth'])+'</p>'
        for side in ['A','B']:
            page += '<h2>'+side+'</h2><pre>'+esc(x[side])+'</pre>'
        page += '<details><summary>封存阅读后揭示的方法及提取结果</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
        write_page(dest/'cases'/(r['case_id']+'.html'),page)
    body += '''</tbody></table><h2>正在进行的后续工作</h2><p>GSM8K64×4迁移与原验证32题的10项解释消融按既定协议继续。原始评分、生成、选优和统计产物保持冻结。新语义核查独立保存；任何完整再评分必须单独版本化，覆盖全部条件并保留原分数，不能只修漂亮案例。</p>'''
    write_page(dest/'index.html',body)
    write_json(dest/'_SUCCESS.json',dict(complete=True, visual_review_pending=True, independent_human_review=False,
        files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))


def write_page(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    style='body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 24px;line-height:1.65;color:#192733}img{max-width:100%}.note{background:#edf4f7;border-left:4px solid #247c78;padding:18px;margin:15px 0}table{border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126d84}'
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HSS primary case review</title><style>'+style+'</style>'+body+'</html>')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)

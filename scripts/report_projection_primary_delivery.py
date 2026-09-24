"""Presentation-only supplement to the immutable, audited primary experiment.

Read frozen statistics without refitting or recomputing their confidence intervals.
Export all primary flips with the same blinded case IDs as projection v2.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from revision_common import sha, write_json


def run(root):
    audit = json.loads((root / 'steering_statistics_audit.json').read_text())
    assert audit['complete'] and audit['source_summary_sha256'] == sha(root / 'steering_summary.json')
    summary = json.loads((root / 'steering_summary.json').read_text())
    receipt = json.loads((root / 'test_SUCCESS.json').read_text())
    dest = root / 'paper_primary'
    if (dest / '_SUCCESS.json').exists():
        raise RuntimeError('Preserve completed presentation; use a new destination for revisions')
    dest.mkdir(exist_ok=True)
    rows = [r for r in summary['rows'] if r['stage'] == 'test']
    names = {'baseline': 'Baseline', 'c1_1.0': 'Local8', 'shared8': 'Shared8',
             'random_seed_average': 'Random (3-seed mean)'}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout='constrained')
    for i, row in enumerate(rows):
        ci = np.array(row['accuracy']['ci95']) * 100
        value = row['accuracy']['estimate'] * 100
        axes[0].plot(ci, [i, i], color='#247c78', lw=2)
        axes[0].scatter(value, i, color='#247c78', zorder=3)
        axes[0].annotate(f'{value:.2f}%', (value, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    axes[0].set(yticks=range(len(rows)), yticklabels=[names[r['method']] for r in rows],
                xlabel='Accuracy (%)', title='Frozen MATH confirmation (256 questions)', ylim=(-.6, 3.6))
    axes[0].invert_yaxis()
    comparisons = [('Local8 − baseline', next(r['delta_accuracy'] for r in rows if r['method'] == 'c1_1.0'))]
    comparisons += [('Local8 − ' + names[r['control']], r['accuracy_difference']) for r in summary['paired_controls']]
    for i, (_, metric) in enumerate(comparisons):
        lo, hi = np.array(metric['ci95']) * 100
        value = metric['estimate'] * 100
        axes[1].plot([lo, hi], [i, i], color='#346f86', lw=2)
        axes[1].scatter(value, i, color='#346f86', zorder=3)
        axes[1].annotate(f'{value:+.2f} pp [{lo:+.2f}, {hi:+.2f}]', (value, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    axes[1].axvline(0, color='gray', ls='--')
    axes[1].set(yticks=range(3), yticklabels=[x[0] for x in comparisons], xlabel='Paired accuracy change (percentage points)',
                title='Pointwise 95% question-bootstrap intervals', ylim=(-.6, 2.6))
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(dest / 'primary_accuracy.png', dpi=180, bbox_inches='tight')
    fig.savefig(dest / 'primary_accuracy.pdf', bbox_inches='tight')
    plt.close(fig)

    records = {}
    for rel, digest in receipt['records'].items():
        assert sha(root / rel) == digest
        r = json.loads((root / rel).read_text())
        records.setdefault(r['sample_id'], {})[r['condition']['name']] = (r, rel)
    assert len(records) == 256 and all(len(x) == 6 for x in records.values())
    blind = []
    key = []
    for sid, methods in sorted(records.items()):
        a, ap = methods['c1_1.0']; b, bp = methods['baseline']
        if a['correct'] == b['correct']:
            continue
        case_id = hashlib.sha256(f'MATH_primary256_supplement/local8/baseline/{sid}'.encode()).hexdigest()[:16]
        flip = int(case_id[-1], 16) % 2
        left, right = (a, b) if flip else (b, a)
        blind.append(dict(case_id=case_id, prompt=left['prompt_text'], ground_truth=left['ground_truth'],
                          A=left['response_text'], B=right['response_text']))
        key.append(dict(case_id=case_id, sample_id=sid, method_side='A' if flip else 'B',
                        effect='repair' if a['correct'] else 'damage', method_record=ap, baseline_record=bp))
    assert len(blind) == 33
    write_json(dest / 'blind_cases.json', blind)
    write_json(dest / 'case_key.json', key)
    body = '''<h1>局部投影：256 题确认结果</h1>
<p>Qwen2-7B-Instruct · MATH · block14 · 自然生成16 token后修改最后4个位置 · local8 · alpha=1。</p>
<div class="note"><strong>验证集提升没有在这批确认题上复现。</strong> Local8 修复17题、损伤16题，净增1/256（+0.39个百分点；95%区间 −3.91至+4.69）。目前不能据此声称稳定提高正确率，也不能把不显著解释为严格等效。</div>
<img src="primary_accuracy.png" alt="Frozen primary accuracy and paired differences">
<p><a href="primary_accuracy.pdf">论文图 PDF</a> · <a href="../report/index.html">原始审计报告及320题全部回答</a> · <a href="../steering_summary.json">冻结统计 JSON</a></p>
<table><thead><tr><th>条件</th><th>正确率</th><th>错→对</th><th>对→错</th><th>截断</th><th>平均长度</th></tr></thead><tbody>'''
    for r in rows:
        body += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in [
            names[r['method']], f"{100*r['accuracy']['estimate']:.2f}%",
            f"{r['wrong_to_correct_expected']:.2f}", f"{r['correct_to_wrong_expected']:.2f}",
            f"{100*r['truncation_rate']:.2f}%", f"{r['mean_length']:.1f}"]) + '</tr>'
    body += '''</tbody></table><p>随机对照为同一径向／能量匹配操作的三个seed，先在每题内平均；题数仍为256。小数修复／损伤数为三个seed的平均，不是额外独立样本。所有条件自动解析失败为0。</p>
<h2>结果范围</h2><p>统计原样读取通过独立复算的原始汇总，未重选方法、子组或置信区间。题目是历史MATH，本次方法选择留出；不宣称从未接触的全新基准。新GSM8K与旧验证题消融继续按事前冻结协议执行。</p>
<h2>全部33个主比较修复／损伤案例</h2><p>下面可查看原始题目和回答。另存方法匿名的A/B材料供语义复核；生成这些材料本身不等于完成语义审查。自动数学判分已独立复算，语义审查状态单独记录。</p>'''
    body += ' · '.join(f'<a href="../report/questions/{x["sample_id"]}.html">{x["sample_id"]}</a>' for x in key)
    style = 'body{font:16px system-ui;max-width:1180px;margin:36px auto;padding:0 24px;line-height:1.65;color:#192733}img{max-width:100%}.note{background:#edf4f7;padding:18px;border-left:4px solid #247c78}table{border-collapse:collapse}td,th{padding:9px 16px;border-bottom:1px solid #ddd;text-align:left}a{color:#126d84}'
    (dest / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HSS primary confirmation</title><style>' + style + '</style>' + body + '</html>')
    write_json(dest / '_SUCCESS.json', dict(complete=True, visual_review_pending=True, semantic_review_pending=True,
        primary_summary_sha256=sha(root / 'steering_summary.json'), statistics_audit_sha256=sha(root / 'steering_statistics_audit.json'),
        source_sha256=sha(Path(__file__)), files={str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)

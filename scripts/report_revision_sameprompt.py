"""Create a finite report only from completed, independently audited readouts."""
import argparse
import html
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from revision_common import sha, write_json, freeze, provenance
from revision_sameprompt_report_metrics import VIEWS, METHODS, coverage, view_metrics, monitoring

STYLE = '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.6;color:#182536}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d9e1e9;padding:7px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px}.scroll{overflow:auto}img{max-width:100%}a{color:#155d96}.note{background:#fff5d6;padding:16px}details{margin:12px 0}</style>'


def read(root):
    dest = root/'readouts'; receipt = json.loads((dest/'_SUCCESS.json').read_text())
    audit = json.loads((dest/'audit.json').read_text())
    assert receipt['complete'] and audit['complete']
    assert audit['readout_receipt_sha256'] == sha(dest/'_SUCCESS.json')
    for name, digest in receipt['files'].items():
        assert sha(dest/name) == digest, name
    raw = json.loads((root/'full/_SUCCESS.json').read_text())
    raw_audit = json.loads((root/'full/audit.json').read_text())
    assert raw_audit['complete'] and raw_audit['stage_receipt_sha256'] == sha(root/'full/_SUCCESS.json')
    plan = json.loads((root/'plan.json').read_text())
    assert sha(root/'inputs.json') == plan['inputs_sha256']
    inputs = json.loads((root/'inputs.json').read_text()); records = {}
    for name, digest in sorted(raw['records'].items()):
        path = root/'full/samples'/name; assert sha(path) == digest
        record = json.loads(path.read_text()); records[record['trajectory']['trajectory_id']] = record
    universe = pd.read_parquet(dest/'all_trajectories.parquet')
    assert len(universe) == len(records) == 896 and set(universe.trajectory_id) == set(records)
    assert universe.groupby('role').question_group.nunique().to_dict() == plan['config']['role_counts']
    views = {name: pd.read_parquet(dest/name/'predictions.parquet') for name in VIEWS}
    for name, frame in views.items():
        prefix = int(name.split('_')[0][1:])
        assert set(frame.trajectory_id) == {t for t, r in records.items() if r['prefixes'][str(prefix)]['valid']}
        assert not frame.trajectory_id.duplicated().any()
    return plan['config'], inputs, records, universe, views


def fmt(result):
    if result['estimate'] is None:
        return '不可估计'
    return f"{result['estimate']:.4f} [{result['low']:.4f}, {result['high']:.4f}]"


def figures(dest, summaries, monitor):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, view in zip(axes, VIEWS):
        ax.axvline(0, color='#9aa4b1', lw=1)
        methods = ['mean16', 'duplicate_current']+(['allmean'] if view == 'p64_l28' else [])
        for i, method in enumerate(methods):
            r = summaries[view]['within_question'][method]['paired_delta']
            if r['estimate'] is not None:
                ax.plot([r['low'], r['high']], [i, i], color='#246aa2', lw=2)
                ax.scatter(r['estimate'], i, color='#246aa2', s=45)
            else:
                ax.text(.05, i, 'Not estimable', transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(methods)), methods); ax.set_ylim(-.6, len(methods)-.4)
        ax.set_title(view+(' (primary)' if view == 'p16_l28' else ' (secondary)'))
        ax.set_xlabel('AUROC difference vs baseline')
    fig.suptitle('Fixed readouts | Question-bootstrap 95% intervals | No causal claim')
    fig.savefig(dest/'within_question.png', dpi=180); fig.savefig(dest/'within_question.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.axvline(.1, ls='--', color='#9aa4b1', label='Calibration target (not guaranteed test FAR)')
    for method, color in zip(METHODS, ['#687887', '#246aa2', '#a9622a']):
        test = monitor[method]['test']
        if test is None:
            continue
        x, y = test['far'], test['failure_detection']
        if x['estimate'] is None or y['estimate'] is None:
            continue
        ax.plot([x['low'], x['high']], [y['estimate']]*2, color=color)
        ax.plot([x['estimate']]*2, [y['low'], y['high']], color=color)
        ax.scatter(x['estimate'], y['estimate'], color=color, label=method, s=55)
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='Actual test false-alarm rate',
           ylabel='Failure detection rate', title='Two checks (16 / 64 tokens), one response-level threshold')
    ax.legend(fontsize=9, loc='upper left'); fig.savefig(dest/'monitoring.png', dpi=180)
    fig.savefig(dest/'monitoring.pdf'); plt.close(fig)


def page(dest, summary, inputs, records, views, alarm_rows, selected, coverage_all):
    parts = ['<!doctype html><meta charset="utf-8"><title>同题多次生成：mean 增量</title>'+STYLE,
             '<h1>同一道题，均值能否识别这一次解答的成败？</h1>',
             '<p>Qwen2-7B-Instruct · MATH 224 题 × 4 次随机回答 · temperature 0.7 · 原始 block 输出。</p>',
             '<p>数据范围：'+html.escape(summary['scope'])+'</p>',
             '<p class="note">这是带标签线性读出的观察实验，不是无监督检测或因果干预。使用历史上研究过的问题和新随机回答；不能称为全新问题确认。训练96题、调参32题、阈值校准32题、测试64题，按整道题隔离。</p>',
             '<h2>主结果：block28 / RMSNorm 前 / 第16个生成 token</h2>',
             '<p>基线 B = prompt-last + current-token + controls；controls 包含题型、难度、prompt长度、prompt/current/mean16的norm、当前熵与margin。各方法仅在训练集拟合标准化和权重，tuning选择三个C之一。最终长度不是特征。</p>']
    primary = summary['views']['p16_l28']
    parts += ['<p><b>mean16 − B 的题内 AUROC 增量：'+fmt(primary['within_question']['mean16']['paired_delta'])+'</b>；mean16 − duplicate-current：'+fmt(primary['mean16_minus_duplicate'])+'。</p>',
              '<p>可估计混合题 '+str(primary['within_question']['mean16']['paired_delta']['n_questions'])+'/64。每题权重相同，正确/错误轨迹对不是独立样本；没有混合题时明确不可估计。</p>',
              '<img src="within_question.png" alt="逐视图题内AUROC差值与问题bootstrap区间"><p><a href="within_question.pdf">导出 PDF</a></p>',
              '<h2>覆盖率与前缀多样性</h2>']
    coverage_rows = []
    for view in VIEWS:
        for key, value in summary['views'][view].items():
            if key in ['test_questions', 'valid_test_trajectories', 'full_outcomes', 'valid_outcomes']:
                coverage_rows.append({'view': view, 'item': key, 'value': str(value)})
    parts.append(pd.DataFrame(coverage_rows).to_html(index=False, escape=True))
    parts += ['<p>全对/全错基于完整四条回答；valid_outcomes仅基于可用前缀。相同前缀没有四份不同状态，独特前缀数量保存在逐题覆盖表。过短响应保留在完整分母中，不追加采样制造混合题。</p>',
              pd.DataFrame([{'role': role, **value} for role, value in summary['outcomes_by_role'].items()]).to_html(index=False),
              '<h2>辅助跨题指标</h2>']
    rows = []
    for view, metrics in summary['views'].items():
        for method, values in metrics['cross_question'].items():
            rows.append({'view': view, 'method': method, **{k: fmt(v) for k, v in values.items()}})
    parts += ['<div class="scroll">'+pd.DataFrame(rows).to_html(index=False)+'</div>',
              '<p>全部区间：2000次问题bootstrap、逐项95%，未校正多重比较；只反映固定训练模型下的测试问题不确定性。主视图不根据结果替换。common_prefix_cohort.json额外提供16/64均可用的同一批轨迹及问题结果；原覆盖不同的辅助结果不能直接解释为时间趋势。</p>',
              '<h2>两次检查，10% 校准 FAR 目标</h2>',
              '<p>只用最终正确的 calibration 回答，在16/64两个风险的最大值上校准一个阈值，采用严格 risk &gt; threshold。两处都不可用时最大值记为负无穷；没有正确校准回答则整个监测指标不可估计。allmean没有单独16步模型，未参与这项监测。</p>',
              '<img src="monitoring.png" alt="实际测试FAR与失败检测率"><p><a href="monitoring.pdf">导出 PDF</a></p>']
    rows = []
    for method, values in summary['monitoring'].items():
        test = values['test']
        rows.append({'method': method, 'threshold': str(values['threshold']), 'calibration_FAR': values['calibration_far'],
                     'test_FAR': fmt(test['far']) if test else '不可估计',
                     'test_failure_detection': fmt(test['failure_detection']) if test else '不可估计',
                     'potential_token_fraction': fmt(test['potential_token_fraction']) if test else '不可估计'})
    parts += ['<div class="scroll">'+pd.DataFrame(rows).to_html(index=False)+'</div>',
              '<p class="note">相同校准目标不保证测试FAR相同，因此不能把检测率差直接称为严格matched-test-FAR优势。区间固定了已拟合模型和校准阈值，不包含重训或校准的不确定性。token节约只是对已生成响应的假设停止计数，不是实际节省计算或纠正答案。完整summary另列正确回答受影响比例与首报位置。</p>',
              '<h2>复核与复用</h2><ul>',
              '<li><a href="summary.json">全部指标与阈值</a> · <a href="question_coverage.parquet">逐题覆盖/前缀数量</a> · <a href="within_question.parquet">逐题AUROC</a></li>',
              '<li><a href="predictions.parquet">全部角色/视图预测</a> · <a href="alarms.parquet">每条轨迹风险/首次报警</a> · <a href="selection.json">只按tuning选择的C</a></li>',
              '<li><a href="common_prefix_cohort.json">16/64共同可用轨迹的辅助结果</a> · <a href="source_manifest.json">数据与模型来源</a> · <a href="statistics_audit.json">独立统计审计（队列完成后生成）</a></li></ul>',
              '<p>可复用权重、所有C候选、训练变换和类别编码保留在同一根目录readouts/；上述下载保留轻量预测，不复制hidden states。</p>',
              '<h2>全部224题原文与四条回答</h2><ul>']
    by_sample = {}
    for tid, record in records.items():
        by_sample.setdefault(record['sample_id'], []).append((tid, record))
    pred = pd.concat([frame.assign(view=view) for view, frame in views.items()], ignore_index=True)
    for index, item in enumerate(inputs):
        filename = f'q{index:03}.html'; role = item['role']; sid = item['sample_id']
        title = f'{role} · {sid} · {item["category"]} · {item["level"]}'
        chunks = ['<!doctype html><meta charset="utf-8">'+STYLE, '<a href="../index.html">返回报告</a>',
                  '<h1>'+html.escape(title)+'</h1>', '<h2>原题</h2><pre>'+html.escape(item['prompt_text'])+'</pre>',
                  '<details><summary>完整模型输入</summary><pre>'+html.escape(item['model_input_text'])+'</pre></details>',
                  '<p>参考答案：'+html.escape(item['ground_truth'])+'</p>']
        for tid, record in sorted(by_sample[sid]):
            chunks += ['<h2>'+html.escape(tid)+(' · 正确' if record['correct'] else ' · 错误')+'</h2>',
                       '<p>'+html.escape(f"{record['length']} tokens · {record['finish_reason']} · parsed: {record['parsed_answer']}")+'</p>',
                       '<pre>'+html.escape(record['response_text'])+'</pre>',
                       '<div class="scroll">'+pred[pred.trajectory_id == tid].to_html(index=False, na_rep='不可用')+'</div>']
            a = alarm_rows[alarm_rows.trajectory_id == tid]
            if len(a):
                chunks.append('<div class="scroll">'+a.to_html(index=False, na_rep='不可用')+'</div>')
        (dest/'questions'/filename).write_text('\n'.join(chunks))
        parts.append('<li><a href="questions/'+filename+'">'+html.escape(title)+'</a></li>')
    parts.append('</ul>'); (dest/'index.html').write_text('\n'.join(parts))


def run(root):
    started = time.monotonic(); dest = root/'report'
    if dest.exists():
        raise RuntimeError('Report namespace exists; preserve partial evidence, inspect before recovery')
    cfg, inputs, records, universe, views = read(root)
    dest.mkdir(); (dest/'questions').mkdir()
    files = [root/'plan.json', root/'inputs.json', root/'full/_SUCCESS.json', root/'full/audit.json',
             root/'readouts/_SUCCESS.json', root/'readouts/audit.json', Path(__file__),
             Path(__file__).with_name('revision_sameprompt_report_metrics.py'),
             Path(__file__).with_name('revision_sameprompt_statistics.py')]
    freeze(dest/'plan.json', provenance(cfg, files)); draws = cfg['bootstrap_questions']
    summary = {'scope': cfg['scope'], 'bootstrap_draws': draws, 'bootstrap_seed': 42,
               'ci_scope': 'Pointwise, conditional on fixed readouts and fixed calibration thresholds; no multiplicity adjustment.',
               'primary_view': 'p16_l28', 'views': {}, 'outcomes_by_role': {}}
    all_coverage = []; question_rows = []; selected = []
    for role, frame in universe.groupby('role', sort=True):
        summary['outcomes_by_role'][role] = {'questions': frame.question_group.nunique(), 'responses': len(frame),
            'correct': int(frame.failure.eq(0).sum()), 'truncated': int(frame.finish_reason.eq('length').sum()),
            'truncation_rate': float(frame.finish_reason.eq('length').mean())}
    for name, frame in views.items():
        prefix = int(name.split('_')[0][1:]); cov = coverage(universe, frame, records, prefix)
        metrics, rows = view_metrics(frame, cov, draws); summary['views'][name] = metrics
        all_coverage.append(cov.assign(view=name)); question_rows += [{'view': name, **r} for r in rows]
        for method in metrics['within_question']:
            selection = json.loads((root/'readouts'/name/method/'selection.json').read_text())
            selected.append({'view': name, 'method': method, **selection})
    summary['monitoring'], alarm_rows = monitoring(universe, views, draws, cfg['far_target'])
    common = set(views['p16_l28'].trajectory_id) & set(views['p64_l28'].trajectory_id)
    common_results = {}
    for name in ['p16_l28', 'p64_l28']:
        frame = views[name][views[name].trajectory_id.isin(common)]
        cov = coverage(universe, frame, records, int(name.split('_')[0][1:]))
        common_results[name] = view_metrics(frame, cov, draws)[0]
    cov = pd.concat(all_coverage, ignore_index=True)
    cov.to_parquet(dest/'question_coverage.parquet', index=False)
    pd.DataFrame(question_rows, columns=['view', 'method', 'question_group', 'valid_trajectories', 'auc_first', 'auc_second', 'delta']).to_parquet(dest/'within_question.parquet', index=False)
    alarm_rows.to_parquet(dest/'alarms.parquet', index=False)
    pd.concat([f.assign(view=n) for n, f in views.items()], ignore_index=True).to_parquet(dest/'predictions.parquet', index=False)
    write_json(dest/'summary.json', summary); write_json(dest/'selection.json', selected)
    write_json(dest/'common_prefix_cohort.json', {'trajectory_ids': sorted(common), 'views': common_results})
    write_json(dest/'source_manifest.json', {'root': str(root), 'weights_and_transforms': str(root/'readouts'),
        'input_sha256': sha(root/'inputs.json'), 'raw_receipt_sha256': sha(root/'full/_SUCCESS.json'),
        'readout_receipt_sha256': sha(root/'readouts/_SUCCESS.json'), 'readout_audit_sha256': sha(root/'readouts/audit.json'),
        'question_order': [{'file': f'questions/q{i:03}.html', 'sample_id': r['sample_id']} for i, r in enumerate(inputs)]})
    figures(dest, summary['views'], summary['monitoring'])
    page(dest, summary, inputs, records, views, alarm_rows, selected, cov)
    files = [p for p in dest.rglob('*') if p.is_file()]
    write_json(dest/'_SUCCESS.json', {'complete': True, 'statistics_audit_pending': True, 'seconds': time.monotonic()-started,
        'plan_sha256': sha(dest/'plan.json'), 'files': {str(p.relative_to(dest)): sha(p) for p in files}})
    print(json.dumps({'report_built': True, 'statistics_audit_pending': True, 'seconds': time.monotonic()-started}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)

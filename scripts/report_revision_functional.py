"""Report a fully audited functional pilot, keeping questions as the unit.

This is descriptive analysis of a historical MATH pilot, not confirmation or
selective steering. It never reads the running free-generation outcome files.
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
import pandas as pd


KEYS = ['split', 'prefix_tokens', 'layer', 'role', 'width']
METRICS = ['next_token_kl', 'nll', 'next_token_argmax_agreement']
COMPARISONS = [('centroid', 'position_mean'), ('empirical_pca_8', 'centroid'),
               ('local_pca_8', 'centroid'), ('global_pca_8', 'centroid'),
               ('kmeans_centroid', 'centroid'), ('centroid', 'matched_random'),
               ('centroid_energy1', 'matched_random_energy1'),
               ('empirical_pca_8', 'global_pca_8')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def interval(values, boot=2000):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('A finite nonempty question-level vector is required')
    ix = np.random.default_rng(42).integers(len(x), size=(boot, len(x)))
    low, high = np.quantile(x[ix].mean(axis=1), [.025, .975])
    return {'estimate': float(x.mean()), 'low': float(low), 'high': float(high), 'n': len(x)}


def paired_difference(frame, method_a, method_b, metric):
    a = frame.loc[frame.method.eq(method_a)].set_index('sample_id')[metric].sort_index()
    b = frame.loc[frame.method.eq(method_b)].set_index('sample_id')[metric].sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError('Paired comparisons require exactly the same unique questions')
    return interval(a.to_numpy(float)-b.to_numpy(float))


def tables(frame):
    summaries, comparisons, widths = [], [], []
    for group, sub in frame.groupby(KEYS, sort=True):
        common = dict(zip(KEYS, group))
        baseline = sub.loc[sub.method.eq('identity')].set_index('sample_id').sort_index()
        zero = sub.loc[sub.method.eq('zero')].set_index('sample_id').sort_index()
        for method, rows in sub.groupby('method', sort=True):
            rows = rows.set_index('sample_id').sort_index()
            if not rows.index.equals(baseline.index) or not rows.index.equals(zero.index):
                raise ValueError('Unequal method coverage')
            row = {**common, 'method': method, 'n': len(rows)}
            for metric in METRICS+['actual_patch_energy']:
                for name, value in interval(rows[metric]).items():
                    if name != 'n': row[f'{metric}_{name}'] = value
            for name, value in interval(rows.nll-baseline.nll).items():
                if name != 'n': row[f'delta_nll_{name}'] = value
            row.update(mean_identity_nll=float(baseline.nll.mean()),
                       mean_zero_nll=float(zero.nll.mean()),
                       median_reference_tokens=float(rows.reference_tokens.median()))
            den = float((zero.nll-baseline.nll).sum())
            row['loss_recovered'] = float((zero.nll-rows.nll).sum()/den) if den > 1e-8 else None
            summaries.append(row)
        for a, b in COMPARISONS:
            for metric in METRICS:
                comparisons.append({**common, 'method_a': a, 'method_b': b,
                                    'metric': metric, **paired_difference(sub, a, b, metric)})
    for group, sub in frame.groupby(KEYS[:-1], sort=True):
        for method in ['centroid', 'centroid_energy1', 'matched_random', 'matched_random_energy1']:
            selected = sub.loc[sub.method.eq(method)].copy()
            selected['method'] = selected.width.map(lambda w: f'width{w}')
            for width in [4, 16]:
                for metric in METRICS:
                    widths.append({**dict(zip(KEYS[:-1], group)), 'method': method,
                                   'width_a': width, 'width_b': 1, 'metric': metric,
                                   **paired_difference(selected, f'width{width}', 'width1', metric)})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons), pd.DataFrame(widths)


def plot(s, dest):
    methods = ['position_mean', 'centroid', 'empirical_pca_8', 'global_pca_8', 'matched_random']
    labels = ['Slot mean', 'GMM center', 'Local PCA 8', 'Global PCA 8', 'Random*']
    colors = ['#9d7660', '#275dba', '#23896a', '#8169a8', '#777777']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
    for col, (prefix, role, title) in enumerate([(0, 'tokens', 'Chat tail'),
            (0, 'question_tokens', 'Question-content tail'), (16, 'tokens', 'Generated tokens 1–16')]):
        sub = s.loc[s.split.eq('test') & s.layer.eq(14) & s.width.eq(16)
                    & s.prefix_tokens.eq(prefix) & s.role.eq(role)].set_index('method').loc[methods]
        for i, metric in enumerate(['next_token_kl', 'delta_nll']):
            ax = axes[i, col]; y = sub[f'{metric}_estimate'].to_numpy()
            low, high = sub[f'{metric}_low'].to_numpy(), sub[f'{metric}_high'].to_numpy()
            ax.bar(np.arange(len(methods)), y, color=colors, width=.68)
            # CI endpoints are drawn directly (bootstrap intervals need not contain the point estimate).
            ax.vlines(np.arange(len(methods)), low, high, color='black', linewidth=1)
            ax.scatter(np.arange(len(methods)), low, marker='_', color='black', s=28)
            ax.scatter(np.arange(len(methods)), high, marker='_', color='black', s=28)
            ax.set_xticks(np.arange(len(methods)), labels, rotation=25, ha='right')
            if i == 0: ax.set_yscale('log'); ax.set_title(f'{title} · n={int(sub.n.iloc[0])}')
            else: ax.axhline(0, color='#555', linewidth=.6)
            ax.set_ylabel('Next-token KL (nats; log scale)' if i == 0 else 'Reference ΔNLL (nats/token)')
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Qwen2-7B · MATH · block 14 · replace 16 real tokens\nHistorical-test pilot; pointwise 95% question bootstrap intervals', fontsize=13)
    fig.supxlabel('*Random adds per-token energy-matched noise to the original activation; it is not a compressed decoder.', fontsize=9)
    for ext in ['png', 'pdf']: fig.savefig(dest/f'functional_roles.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for method, label, color in [('centroid', 'Full centroid replacement', '#275dba'),
                                ('centroid_energy1', 'Centroid direction, fixed total energy', '#23896a'),
                                ('matched_random_energy1', 'Random, same fixed energy', '#777777')]:
        sub = s.loc[s.split.eq('test') & s.layer.eq(14) & s.prefix_tokens.eq(16)
                    & s.role.eq('tokens') & s.method.eq(method)].sort_values('width')
        for ax, metric in zip(axes, ['next_token_kl', 'delta_nll']):
            ax.plot(sub.width, sub[f'{metric}_estimate'], 'o-', color=color, label=label)
            ax.fill_between(sub.width, sub[f'{metric}_low'], sub[f'{metric}_high'], color=color, alpha=.15)
            ax.set_xticks([1, 4, 16]); ax.set_xlabel('Patched tokens'); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Next-token KL (nats; log scale)')
    axes[1].set_ylabel('Reference ΔNLL (nats/token)'); axes[1].axhline(0, color='#555', linewidth=.6)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Generated-prefix 16 · block 14 · same 32 historical-test questions\nFixed energy means partial movement toward centers, not full reconstruction.', fontsize=12)
    for ext in ['png', 'pdf']: fig.savefig(dest/f'functional_width_energy.{ext}', dpi=180)
    plt.close(fig)


def run(root):
    stage = root/'functional'; audit = json.loads((stage/'execution_audit.json').read_text())
    if not audit['stage_complete'] or audit['snapshot_conditions'] != audit['expected_conditions']:
        raise ValueError('Full execution audit required')
    if audit['plan_sha256'] != digest(stage/'plan.json'):
        raise ValueError('Plan changed after execution audit')
    paths = sorted((stage/'samples').glob('*.json'))
    frame = pd.DataFrame([json.loads(p.read_text()) for p in paths])
    if len(frame) != audit['expected_conditions'] or frame.duplicated(KEYS+['method', 'sample_id']).any():
        raise ValueError('Coverage or duplicate condition failure')
    stored = pd.read_parquet(stage/'per_question.parquet')
    sort = KEYS+['method', 'sample_id']
    for column in METRICS+['nll_sum', 'reference_tokens', 'actual_patch_energy']:
        np.testing.assert_array_equal(frame.sort_values(sort)[column], stored.sort_values(sort)[column])
    summaries, comparisons, widths = tables(frame)
    dest = stage/'report'; dest.mkdir(exist_ok=True)
    for name, table in [('summary', summaries), ('paired_methods', comparisons), ('paired_widths', widths)]:
        table.to_parquet(dest/(name+'.parquet'), index=False)
        write_json(dest/(name+'.json'), table.astype(object).where(pd.notnull(table), None).to_dict('records'))
    # Per-position final block replacements have the same causal scope for
    # nested windows. Window-shaped PCA arithmetic can change rounding slightly.
    deterministic = ['identity', 'zero', 'mean', 'position_mean', 'centroid', 'kmeans_centroid',
                     'empirical_pca_8', 'local_pca_8', 'global_pca_8', 'beta_0.25', 'beta_0.5', 'beta_0.75']
    final = frame.loc[frame.layer.eq(28) & frame.role.eq('tokens') & frame.method.isin(deterministic)]
    final_spread = {metric: 0. for metric in METRICS}
    for _, sub in final.groupby(['sample_id', 'prefix_tokens', 'method']):
        assert len(sub) == 3
        for metric in METRICS:
            values = sub[metric].to_numpy(float)
            final_spread[metric] = max(final_spread[metric], float(np.ptp(values)))
            np.testing.assert_allclose(values, np.repeat(values[0], 3), rtol=0,
                                       atol=1e-7 if metric == 'next_token_kl' else 0)
    hashes = {p.name: digest(p) for p in paths}; write_json(dest/'input_hashes.json', hashes)
    manifest = {'conditions': len(frame), 'questions': int(frame.sample_id.nunique()),
        'method_rows': len(summaries), 'paired_method_rows': len(comparisons), 'paired_width_rows': len(widths),
        'bootstrap': '2000 paired question resamples, seed42; pointwise; validation and test reported separately',
        'scope': 'Exploratory completed functional pilot; no behavior or steering conclusion',
        'final_block_deterministic_window_invariance_passed': True,
        'final_block_max_width_spread': final_spread,
        'final_block_KL_arithmetic_tolerance': 1e-7,
        'plan_sha256': digest(stage/'plan.json'), 'execution_audit_sha256': digest(stage/'execution_audit.json'),
        'per_question_sha256': digest(stage/'per_question.parquet'), 'code_sha256': digest(Path(__file__)),
        'files': {name: digest(dest/name) for name in ['summary.parquet', 'paired_methods.parquet', 'paired_widths.parquet', 'input_hashes.json']}}
    write_json(dest/'audit.json', manifest); plot(summaries, dest)
    notes = [
        '完整执行：64题（验证32、历史测试32），17,010条件；正文尾部部分题不足16token，单列有效数量。',
        '每题等权；NLL先在该题完整保存的参考续写上求每token均值，再跨题平均。KL只针对当前next-token分布。',
        '参考续写可能本来就是错答案。保留其概率不等于正确率；自由生成正在独立测试。',
        '当前为单层一次干预，其他位置/层/上下文仍然保留；不能推导全模型压缩。',
        'Local PCA 8是经验均值中心化的局部PCA；local_pca_8键是固定GMM中心残差SVD，二者都不是MFA。PCA保留8个连续坐标，信息预算高于只保留簇ID。',
        'Random控制在原始激活上加扰动，保留原始信息，不是压缩方法。它只检验破坏是否由扰动范数单独解释。',
        '相同总能量的centroid_energy1是向中心部分移动。不能把其高保真称作多token完整中心重构成功。bf16实际能量审计单独保存，最大误差约0.83%。',
        '最后block的历史位置output无法改变后来token的KV。确定性中心/PCA替换的width1/4/16同效是架构预期；正文历史位置为null。',
        '因此末层完整参考平均NLL可稀释只影响一个预测位置的效应；必须一起读next-token KL，不能只报loss recovered。',
        '损失恢复率=(zero NLL−method NLL)/(zero NLL−identity NLL)。分母<=1e-8时不报告；原始三个loss始终保留。',
        '全部预设方法/位置/宽度及验证/历史测试分开报告。所有区间为pointwise，未做多重比较校正；本阶段叙述选择是探索性的。',
    ]
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Functional reconstruction pilot</title>',
             '<style>body{font:16px system-ui;margin:32px;max-width:1500px;line-height:1.6}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}summary{cursor:pointer}</style>',
             '<h1>多token重构：几何压缩保留多少真实输出功能？</h1>',
             '<p>Qwen2-7B-Instruct × MATH；已完整审计的功能实验，非自由生成正确率结果。</p><ul>']
    parts += ['<li>'+html.escape(n)+'</li>' for n in notes]; parts += ['</ul>']
    for name in ['functional_roles', 'functional_width_energy']:
        parts += [f'<img src="{name}.png" alt="{name}"><p><a href="{name}.pdf">PDF</a></p>']
    for name, table in [('全部方法', summaries), ('配对方法差值（A−B）', comparisons), ('配对范围差值（width−1）', widths)]:
        parts += [f'<details><summary>{name} · {len(table)}行</summary><div class="scroll">',
                  table.to_html(index=False, float_format=lambda x: f'{x:.6g}'), '</div></details>']
    parts += ['<p><a href="audit.json">审计与SHA</a> · <a href="summary.json">全部数值</a> · <a href="paired_methods.json">配对差值</a></p></html>']
    (dest/'index.html').write_text('\n'.join(parts)); print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)

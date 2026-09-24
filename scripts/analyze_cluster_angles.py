"""Separate token-length controls, vector norm and angle in frozen clusters."""

import argparse
import html
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from analyze_cluster_distances import conditional_models, summarize
from hss.analysis.cluster_distance import cluster_percentiles, radial_angular_distances
from hss.experiments.artifacts import file_digest, save_json
from hss.experiments.fitting import load_fitted


def paired_increment(y, prediction, baseline):
    rng = np.random.default_rng(43)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    deltas = []
    for _ in range(400):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(roc_auc_score(y[idx], prediction[idx])-roc_auc_score(y[idx], baseline[idx]))
    return dict(delta=float(roc_auc_score(y, prediction)-roc_auc_score(y, baseline)),
                interval=np.quantile(deltas, [.025, .975]).tolist())


def build(root, states, rows, fits_root=None):
    root = Path(root)
    source = json.loads((root/'distances.json').read_text())
    X = np.load(states, mmap_mode='r')
    meta = pd.read_parquet(rows)
    if file_digest(Path(states)) != source['states_sha256'] or not np.array_equal(meta.sample_id.astype(str), [s['id'] for s in source['samples']]):
        raise ValueError('Input identity differs from frozen distance analysis')
    import hss.analysis.cluster_distance as geometry_module
    import analyze_cluster_distances as controls_module
    configs = dict(composition=['length'], composition_norm=['length', 'norm'],
                   plus_euclidean=['length', 'norm', 'euclidean'],
                   length_angle=['length', 'angle'],
                   norm_angle=['length', 'norm', 'angle'],
                   norm_radial=['length', 'norm', 'radial'],
                   norm_angle_distance=['length', 'norm', 'angle', 'euclidean'])
    result = dict(created_at=time.time(), source_sha256=file_digest(root/'distances.json'),
                  driver_sha256=file_digest(Path(__file__)), geometry_sha256=file_digest(Path(geometry_module.__file__)),
                  controls_sha256=file_digest(Path(controls_module.__file__)), states_sha256=source['states_sha256'],
                  methods={}, controls=source['regression'], numeric_configs=configs,
                  note='Angle is between h and its saved component mean around the original origin. '
                  'Only diagnostic scores use unit directions; no normalized clustering or refitting. '
                  'All clusters were fit on all samples. OOF label checks and conditional bootstrap are exploratory. '
                  'Norm, angle and cluster identity mathematically determine Euclidean distance; incremental '
                  'distance gains can only reflect predictor approximation/interactions, not new geometric information.')
    table = meta.copy()
    norm_control = np.linalg.norm(X, axis=1)  # match the previous float32 baseline exactly
    y = meta.label.to_numpy()
    for name, old in source['methods'].items():
        path = Path(fits_root)/name if fits_root else Path(old['fit_path'])
        if file_digest(path/'model.npz') != old['model_sha256']:
            raise ValueError('Frozen model hash changed')
        model, _ = load_fitted(path)
        labels = np.asarray(old['assignment'])
        centers = np.asarray(model.means_ if hasattr(model, 'means_') else model.centers_, dtype=float)
        scores = radial_angular_distances(X, centers[labels])
        np.testing.assert_allclose(scores['euclidean'], old['values']['euclidean'], rtol=1e-7)
        checks, predictions, fold_ids = conditional_models(meta, {k:scores[k] for k in ['angle','euclidean','radial']}, labels, norm_control, configs)
        entry = dict(k=old['k'], rank=old['rank'], checks=checks,
                     distance_above_angle=paired_increment(y,predictions['norm_angle_distance'],predictions['norm_angle']),
                     angle_quantiles_degrees=np.quantile(np.degrees(scores['angle']), [.1,.5,.9]).tolist(),
                     angular_energy_fraction_quantiles=np.quantile(scores['angular_fraction'], [.1,.5,.9]).tolist(),
                     max_identity_error=float(scores['identity_error'].max()),
                     angle_euclidean_spearman=float(spearmanr(scores['angle'],scores['euclidean']).statistic),
                     metrics={}, predictions={k:p.tolist() for k,p in predictions.items()}, fold_ids=fold_ids.tolist())
        for key in ['euclidean','angle','chord','radial']:
            p = cluster_percentiles(scores[key], labels)
            entry['metrics'][key] = dict(values=scores[key].tolist(), percentiles=p.tolist(),
                auc_near_is_correct=float(roc_auc_score(y,-p)),
                all=summarize(scores[key],p,y,meta,predictions['composition_norm'],np.arange(len(X))),
                clusters={str(k):summarize(scores[key],p,y,meta,predictions['composition_norm'],np.flatnonzero(labels==k)) for k in np.unique(labels)})
            table[name+'_'+key] = scores[key]
        for key in ['norm','center_norm','angular_fraction']:
            table[name+'_'+key] = scores[key]
        entry['angular_fraction'] = scores['angular_fraction'].tolist()
        result['methods'][name] = entry
        print(json.dumps(dict(method=name, angular_fraction_median=entry['angular_energy_fraction_quantiles'][1], checks=checks, distance_above_angle=entry['distance_above_angle'])),flush=True)
    table.to_parquet(root/'angle_members.parquet',index=False)
    save_json(root/'angles.json',result)
    render(root)


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    d = json.loads((root/'angles.json').read_text())
    labels = dict(composition='题型 / 难度 / 簇 / 长度 / 截断',composition_norm='基线 + 向量 norm',
                  plus_euclidean='基线 + norm + 欧氏距离',length_angle='基线 + 角度',norm_angle='基线 + norm + 角度',
                  norm_angle_distance='基线 + norm + 角度 + 欧氏距离',norm_radial='基线 + norm + 去角度后的径向差')
    m = d['methods']['mfa_r8_k21']
    fig, axs = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax, cluster in zip(axs,['all','2']):
        for key,color in [('euclidean','#287e89'),('angle','#a677a6'),('radial','#bb7d3f')]:
            data=m['metrics'][key]['all' if cluster=='all' else 'clusters']
            if cluster!='all': data=data[cluster]
            bins=data['bins'];xs=[(b['bin']+.5)/5 for b in bins if b['n']];ys=[b['accuracy'] for b in bins if b['n']]
            cis=np.array([b['accuracy_ci'] for b in bins if b['n']]).T
            ax.errorbar(xs,ys,yerr=np.maximum(0,np.array([np.array(ys)-cis[0],cis[1]-np.array(ys)])),color=color,marker='o',capsize=3,label=key)
        ax.set(xlabel='Within-cluster score percentile: near → far',ylabel='Accuracy',ylim=(0,.7 if cluster=='all' else .4),title='MFA rank8 K21 / '+('all 5,000' if cluster=='all' else 'Cluster 2 (n=260)'))
        ax.grid(alpha=.18);ax.legend(frameon=False);ax.spines[['right','top']].set_visible(False)
    fig.savefig(root/'angle_comparison.png',dpi=180);fig.savefig(root/'angle_comparison.pdf');plt.close(fig)
    rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{v["angular_energy_fraction_quantiles"][1]:.1%}</td><td>{v["angle_quantiles_degrees"][1]:.2f}°</td><td>{v["metrics"]["euclidean"]["auc_near_is_correct"]:.3f}</td><td>{v["metrics"]["angle"]["auc_near_is_correct"]:.3f}</td><td>{v["metrics"]["radial"]["auc_near_is_correct"]:.3f}</td></tr>' for name,v in d['methods'].items())
    blocks=[]
    for name,v in d['methods'].items():
        table=''.join(f'<tr><td>{labels[k]}</td><td>{check["auc"]:.4f}</td><td>{check["delta_vs_norm"]:+.4f}</td><td>[{check["delta_ci"][0]:+.4f}, {check["delta_ci"][1]:+.4f}]</td></tr>' for k,check in v['checks'].items())
        gain=v['distance_above_angle']
        blocks.append(f'<h3>{name}</h3><div class="scroll"><table><tr><th>模型</th><th>OOF AUROC</th><th>相对 norm 基线增益</th><th>条件配对区间</th></tr>{table}</table></div><p>已经加入 norm 和角度后，再加距离：Δ AUROC {gain["delta"]:+.4f}，区间 [{gain["interval"][0]:+.4f}, {gain["interval"][1]:+.4f}]。</p>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>长度、norm 与角度补偿</title><style>body{font:16px/1.65 system-ui;background:#f3f6f6;color:#163747;max-width:1280px;margin:30px auto;padding:0 24px}section{background:white;border:1px solid #dae5e8;padding:24px;border-radius:12px;margin:22px 0}h1{font-size:30px}h2{font-size:23px}a{color:#287e89}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dde6ea;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{border-left:3px solid #d6a33e;background:#fcf6e9;padding:12px 16px}.formula{font-size:23px;font-family:serif;text-align:center;padding:14px}img{width:100%}</style><p><a href="distance.html">← 返回中心距离分析</a> · <a href="index.html">簇地图</a></p><h1>角度补偿：保留方向信号，还是把它解释掉？</h1>
<section><h2>先区分三个不同的量</h2><p>生成长度 T 是 token 数；向量 norm r=‖h‖ 是回答均值的幅度；θ 是 h 与所属簇保存中心 μ 的夹角。此前分析对 T 和 norm 做了统计控制，没有把聚类输入改成单位向量。</p><div class="formula">‖h − μ‖² = (r − s)² + 2rs(1 − cos θ)，s = ‖μ‖</div><p>左边是距离平方；右边依次为径向长度差和角度贡献。将距离中的角度项减掉，只剩 (r−s)²。若比较单位方向，距离则是 √[2(1−cos θ)]，即完全保留角度、去掉 norm 尺度。</p><p class="note">输入和聚类保持原始形式。角度、单位方向距离及径向项均为事后诊断。参考原点固定为原始表征的零点；改变原点会改变角度。</p></section>
<section><h2>距离中有多少来自角度？</h2><div class="scroll"><table><tr><th>固定划分</th><th>角度能量占比中位数</th><th>夹角中位数</th><th>欧氏 AUROC</th><th>角度 AUROC</th><th>径向 AUROC</th></tr>__ROWS__</table></div><p>能量占比为逐题角度项 / 距离平方，然后取中位数，不是正确性解释比例。单变量 AUROC 均先做簇内排名，并固定“越小越正确”。</p><img src="angle_comparison.png" alt="距离、角度和径向差异的正确率曲线"><p>图示为同批观察比例，误差棒为描述性 95% Wilson 区间；并非独立验证。</p></section>
<section><h2>控制长度后加入角度，有额外收益吗？</h2>__BLOCKS__<p class="note">给定 norm、角度和簇编号，欧氏距离已经被上面的恒等式决定。因此“再加距离”的收益只能反映有限预测模型更容易表达某些交互，不能证明存在新的独立几何信息。</p><p>固定 5 折交叉拟合正确性模型；聚类几何已看过全部数据。连续特征使用固定三次样条，C=1，无调参。区间为固定 OOF 预测的 400 次配对 bootstrap，未覆盖重新拟合/选模不确定性，亦未校正多项探索比较。</p></section>
<section><h2>如何使用“补偿”</h2><ul><li>想排除 norm 的影响、保留方向：单独看角度或单位方向距离，再控制生成长度。</li><li>想检验距离是否超出角度：先加入 norm 与角度，再观察模型是否仍有增益；数学上没有额外信息。</li><li>想解释模型在做什么：一个夹角仍丢失了围绕中心的方向信息，需要具体方向、原文和时序对应。</li></ul><p><a href="angles.json">全部结果、来源与折外预测</a> · <a href="angle_members.parquet">逐题角度与距离</a> · <a href="angle_comparison.pdf">图 PDF</a></p></section></html>'''
    (root/'angles.html').write_text(page.replace('__ROWS__',rows).replace('__BLOCKS__',''.join(blocks)))
    save_json(root/'angle_render_manifest.json',dict(data_sha256=file_digest(root/'angles.json'),renderer_sha256=file_digest(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--states');p.add_argument('--rows');p.add_argument('--fits-root');p.add_argument('--render-only',action='store_true')
    args=p.parse_args()
    if args.render_only: render(args.report)
    else:
        if not args.states or not args.rows:p.error('--states and --rows required')
        with threadpool_limits(2):build(args.report,args.states,args.rows,args.fits_root)

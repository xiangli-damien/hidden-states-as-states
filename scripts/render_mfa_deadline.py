"""Render a completed bounded MFA study using saved models, labels and texts only.

No activation loading or fitting. Output lives outside immutable HSS trials.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hss.analysis.tables import state_map, dynamics, trajectory_similarity
from hss.experiments.artifacts import file_digest, save_json
from hss.results import Result
from hss.viz import plots
from hss.viz.artifacts import FigureBundle


def load_results(root):
    results = {}
    for view, remote in json.loads((root / 'latest_exports.json').read_text()).items():
        local = root / 'trials' / Path(remote).name
        result = Result(local if local.exists() else remote)
        audit = result.validate(full=True)
        if not audit['valid']:
            raise ValueError(audit)
        results[view] = result
    return results


def parameter_figure(selected, candidates, stability):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    post = selected[selected.view == 'post'].sort_values('layer')
    axes[0, 0].plot(post.layer, post.k, 'o-', color='#236b76')
    axes[0, 0].set(title='Selected K at each position', xlabel='Layer position', ylabel='K')
    axes[0, 1].plot(post.layer, post['rank'], 'o-', color='#956438')
    axes[0, 1].set(title='Rank selection (search: 4, 8, 16)', xlabel='Layer position', ylabel='Rank', ylim=(0, 19), yticks=[4, 8, 16])
    axes[0, 1].text(.02, .05, 'Rank 16 is the search boundary, not a global optimum guarantee.', transform=axes[0, 1].transAxes, fontsize=8)
    final = candidates[(candidates.view == 'post') & (candidates.layer == post.layer.max()) & (candidates.seed == 42)]
    for rank, color in [(4, '#346b91'), (8, '#ad703c'), (16, '#39836b')]:
        part = final[final['rank'] == rank]
        strict = part.converged & (part.tol <= 1e-5) & (part.initializations_completed >= 3)
        axes[1, 0].scatter(part.loc[~strict, 'k'], part.loc[~strict, 'icl']/5000, facecolors='none', edgecolors=color, alpha=.5)
        axes[1, 0].scatter(part.loc[strict, 'k'], part.loc[strict, 'icl']/5000, color=color, label=f'rank {rank}')
    axes[1, 0].set(title='Final post-RMS position: evaluated K', xlabel='K (filled: strict; hollow: screening)', ylabel='ICL / sample; lower is better')
    axes[1, 0].legend()
    for seed, group in stability[stability.view == 'post'].groupby('seed'):
        group = group.sort_values('layer')
        axes[1, 1].plot(group.layer, group.ari, 'o-', ms=3, label=f'seed {seed}')
    axes[1, 1].set(title='Assignment agreement across initializations', xlabel='Layer position', ylabel='ARI vs selected fit', ylim=(0, 1.02))
    axes[1, 1].legend()
    return fig


def join_texts(path, post, pre):
    source = json.loads(path.read_text())
    by_id = {str(s['id']): s for s in source['samples']}
    ids = post.rows.sample_id.astype(str).tolist()
    if len(by_id) != 5000 or set(by_id) != set(ids) or pre.rows.sample_id.astype(str).tolist() != ids:
        raise ValueError('Text source and fitted results must have exactly the same 5000 sample IDs')
    local = np.load(post.path / 'local_states.npy')
    pre_local = np.load(pre.path / 'local_states.npy')[:, -1]
    samples = []
    for i, row in enumerate(post.rows.itertuples()):
        text = by_id[str(row.sample_id)]
        if bool(text['correct']) != bool(row.label) or int(text['tokens']) != int(row.n_tokens):
            raise ValueError(f'Text source label/token-count mismatch: {row.sample_id}')
        samples.append({**{k: text[k] for k in ['id', 'correct', 'category', 'tokens', 'question', 'response', 'level', 'finish_reason']},
                        'cluster': int(local[i, -1]), 'pre_cluster': int(pre_local[i]),
                        'local_states': local[i].tolist(), 'global_states': post.states[i].tolist()})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--samples', required=True, help='Existing Qwen-MATH exploration.json containing original texts')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    out = Path(args.output).resolve() if args.output else root / 'final_report'
    results = load_results(root)
    post, pre = results['post'], results['pre_final']
    delivery = json.loads((root / 'delivery.json').read_text())
    if delivery['status'] != 'complete':
        raise ValueError('This report requires a completed study; do not relabel an interim export')
    selected = pd.read_csv(root / 'selected_layers.csv')
    candidates = pd.read_csv(root / 'candidate_metrics.csv')
    stability = pd.read_csv(root / 'stability.csv')
    bundle = FigureBundle(out, [post, pre], dict(
        ranks=[4, 8, 16], coarse_k=[2, 8, 16, 32, 64, 80], trajectory_sample=1000, seed=42,
        normalization=False, selection='Minimum ICL among evaluated strict candidates, tolerance 0',
        script_sha256=file_digest(Path(__file__)), text_source_sha256=file_digest(Path(args.samples))))
    nodes, edges = state_map(post)
    bundle.table('nodes', nodes); bundle.table('edges', edges)
    bundle.figure('entropy_state_map', plots.state_graph(nodes, edges, 'entropy', 'Qwen2 MATH / MFA: transition entropy'))
    bundle.figure('correctness_state_map', plots.state_graph(nodes, edges, 'accuracy_delta', 'Qwen2 MATH / MFA: correctness deviation'))
    marginal, profile = dynamics(post)
    similarity, order = trajectory_similarity(post, max_rows=1000, seed=42)
    bundle.table('state_frequency', marginal); bundle.table('layer_profile', profile); bundle.table('trajectory_order', order)
    bundle.figure('geometry', plots.geometry(marginal, profile, similarity))
    assoc = post.table('associations.csv'); tags = post.table('state_tags.csv')
    bundle.table('associations', assoc); bundle.table('state_tags', tags)
    bundle.figure('associations', plots.associations(assoc))
    bundle.figure('correctness_bands', plots.correctness_bands(tags))
    bundle.figure('selection_and_stability', parameter_figure(selected, candidates, stability))
    for name, table in [('selected_layers', selected), ('stability', stability), ('rank_k_selection', pd.read_csv(root/'rank_k_selection.csv'))]:
        bundle.table(name, table)
    samples = join_texts(Path(args.samples), post, pre)
    save_json(out/'samples.json', samples); bundle.outputs.append(out/'samples.json')
    frame = pd.DataFrame(samples)
    clusters = frame.groupby('cluster').agg(samples=('id', 'size'), accuracy=('correct', 'mean'), median_tokens=('tokens', 'median')).reset_index()
    bundle.table('final_clusters', clusters)
    last = selected[(selected.view == 'post') & (selected.layer == max(post.layers))].iloc[0]
    prelast = selected[selected.view == 'pre_final'].iloc[0]
    median_ari = stability[stability.view == 'post'].ari.median()
    last_ari = stability[(stability.view == 'post') & (stability.layer == max(post.layers))].ari
    figure_titles = [('selection_and_stability', 'K/rank 搜索与种子稳定性'), ('entropy_state_map', '跨层状态图：转移熵'),
                     ('correctness_state_map', '跨层状态图：正确率相对全体的差异'), ('geometry', '状态占比、轨迹相似性与层间变化'),
                     ('associations', '题型、难度和正确性的关联'), ('correctness_bands', '各全局状态的正确率差异')]
    images = ''.join(f'<section><h2>{title}</h2><a href="{name}.svg"><img src="{name}.png" alt="{title}" loading="lazy"></a><p><a href="{name}.svg">SVG 矢量图</a> · <a href="{name}.png">PNG 图片</a></p></section>' for name, title in figure_titles)
    document = PAGE.replace('__FIGURES__', images).replace('__PARAMETERS__', selected[['view','layer','rank','k','icl','bic']].to_html(index=False, float_format=lambda v: f'{v:,.1f}'))
    document = document.replace('__CLUSTERS__', clusters.to_html(index=False, float_format=lambda v: f'{v:.3f}'))
    document = document.replace('__POST_K__', str(int(last.k))).replace('__PRE_K__', str(int(prelast.k))).replace('__RANK__', str(int(last['rank'])))
    document = document.replace('__MEDIAN_ARI__', f'{median_ari:.3f}').replace('__LAST_ARI__', f'{last_ari.min():.3f}–{last_ari.max():.3f}')
    (out/'index.html').write_text(document)
    bundle.outputs.append(out/'index.html')
    bundle.finish(status='complete', scope='Full-data descriptive MFA geometry and initialization sensitivity; no held-out prediction or exhaustive K/rank guarantee',
                  sample_join=dict(count=len(samples),ids_labels_token_counts_match=True), deadline_met=delivery['deadline_met'])
    print(json.dumps(dict(report=str(out/'index.html'),samples=len(samples),figures=len(figure_titles),post_k=int(last.k),pre_k=int(prelast.k),rank=int(last['rank']),median_ari=float(median_ari))))


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen MATH · MFA 最终结果</title><style>
:root{font-family:system-ui,sans-serif;color:#20343b;background:#f4f6f5}body{max-width:1440px;margin:0 auto;padding:32px}h1{font-size:36px;margin-bottom:8px}h2{font-size:24px}h3{font-size:18px}p,li{line-height:1.7}a{color:#176a79}section,.card{background:white;border:1px solid #dce4e4;border-radius:12px;padding:24px;margin:22px 0}img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px}th,td{padding:8px 12px;border-bottom:1px solid #e2e8e8;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.note{border-left:4px solid #a97932;padding:14px 20px;background:#fff8e9}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric{padding:20px;background:#e9f1ef;border-radius:10px}.metric b{font-size:26px;display:block}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}input,select,button{font:inherit;padding:9px;border:1px solid #bccdce;border-radius:5px;background:white}button{cursor:pointer}button.active{background:#dceee8}.explorer{display:grid;grid-template-columns:290px 1fr;gap:24px}#sample-list{max-height:690px;overflow:auto;display:flex;flex-direction:column;gap:4px}#sample-list button{text-align:left;font-size:13px}pre{font:14px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f5;padding:16px;border-radius:8px;max-height:620px;overflow:auto}.muted{color:#627980;font-size:14px}.error{color:#aa3028}@media(max-width:800px){body{padding:15px}.metrics{grid-template-columns:repeat(2,1fr)}.explorer{grid-template-columns:1fr}#sample-list{max-height:240px}}
</style><header><p class="muted">Qwen/Qwen2-7B-Instruct · MATH 5,000 · 2026-09-20</p><h1>MFA：独立 K/rank 搜索结果</h1><p>原始回答 hidden mean，3584 维；不额外归一化、PCA 或 whitening。29 个层位置（含 embedding）及末层 pre-RMS 对照。</p></header>
<div class="metrics"><div class="metric"><b>540 + 175</b>粗网格与局部加密</div><div class="metric"><b>180 / 180</b>严格候选（含 20 个缓存）</div><div class="metric"><b>60 / 60</b>种子检查完成并收敛</div><div class="metric"><b>__MEDIAN_ARI__</b>post 位置的种子 ARI 中位数</div></div>
<section><h2>先看结论与限制</h2><ul><li>末层 post-RMS：<strong>K=__POST_K__、rank=__RANK__</strong>；末层 pre-RMS：<strong>K=__PRE_K__、rank=__RANK__</strong>。</li><li>各层独立搜索后，本轮都选择 rank=16；post 各位置 K=4–9。rank=16 位于本轮上限，不能证明更高 rank 不会更好。</li><li>严格候选要求 tol≤1e-5、三次初始化完成，保存收敛解中似然最好者；其中一次使用普通 EM 作为控制。</li><li>种子 ARI 中位数 __MEDIAN_ARI__，末层 __LAST_ARI__。ARI 衡量分簇的一致性，1 表示完全一致；这些结果不能称为高度稳定。</li></ul><p class="note">最优仅指已验证候选中的最低 ICL。局部加密采用最佳 K 的两侧补点，并未逐整数穷尽 K=2…80。图中的正确率属于全数据描述性关联；没有独立测试集预测结论，也不能把低正确率簇直接解释为因果“错误状态”。</p><p><a href="selected_layers.csv">逐层 K/rank</a> · <a href="rank_k_selection.csv">每个 rank 的 K 搜索覆盖</a> · <a href="stability.csv">种子 ARI</a> · <a href="samples.json">逐题原文与状态轨迹</a> · <a href="manifest.json">图表与数据来源校验</a></p></section>
__FIGURES__
<section><h2>末层各簇与逐题阅读</h2><p class="muted">下表与筛选器使用末层 post-RMS 的局部簇 ID；跨层图使用关联后的全局状态 ID，两者不要混用。</p><div class="scroll">__CLUSTERS__</div><div class="controls"><select id="cluster"><option value="">所有末层簇</option></select><select id="correct"><option value="">全部正确性</option><option value="1">正确</option><option value="0">错误</option></select><input id="search" placeholder="搜索题号、题型或题目" aria-label="搜索题目"><button id="prev">上一页</button><button id="next">下一页</button></div><p id="count" class="muted">正在加载 5,000 道题的原文…</p><div class="explorer"><div id="sample-list"></div><article><h3 id="sample-title">选择一道题</h3><p id="sample-meta" class="muted"></p><h3>原始题目</h3><pre id="question"></pre><h3>模型原始回答</h3><pre id="response"></pre><h3>跨层轨迹：局部簇 ID（位置 0–28）</h3><pre id="local-states"></pre><h3>跨层轨迹：关联后全局状态 ID</h3><pre id="global-states"></pre></article></div></section>
<section><h2>全部位置的选定参数</h2><div class="scroll">__PARAMETERS__</div></section>
<script>
const $=id=>document.getElementById(id);let samples=[],filtered=[],page=0,current=null;const size=50;
function show(s){current=s.id;$('sample-title').textContent=s.id+' · 末层簇 '+s.cluster;$('sample-meta').textContent=(s.correct?'正确':'错误')+' · '+s.category+' · '+s.level+' · '+s.tokens+' tokens · pre-RMS 簇 '+s.pre_cluster+' · '+s.finish_reason;$('question').textContent=s.question;$('response').textContent=s.response;$('local-states').textContent=s.local_states.join(' → ');$('global-states').textContent=s.global_states.join(' → ');for(const b of $('sample-list').children)b.classList.toggle('active',b.dataset.id===current)}
function render(){const shown=filtered.slice(page*size,(page+1)*size);$('sample-list').replaceChildren();for(const s of shown){const b=document.createElement('button');b.dataset.id=s.id;b.textContent=s.id+' · 簇 '+s.cluster+' · '+(s.correct?'正确':'错误')+' · '+s.category;b.onclick=()=>show(s);$('sample-list').appendChild(b)}$('count').textContent=filtered.length+' / 5000 题 · 第 '+(filtered.length?page+1:0)+' / '+Math.ceil(filtered.length/size)+' 页';$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=filtered.length;if(shown.length)show(shown.find(s=>s.id===current)||shown[0]);else{for(const id of ['sample-title','sample-meta','question','response','local-states','global-states'])$(id).textContent=''}}
function filter(){const c=$('cluster').value,y=$('correct').value,q=$('search').value.toLowerCase().trim();filtered=samples.filter(s=>(c===''||s.cluster===Number(c))&&(y===''||Number(s.correct)===Number(y))&&(!q||(s.id+' '+s.category+' '+s.question).toLowerCase().includes(q)));page=0;render()}
for(const id of ['cluster','correct','search'])$(id).addEventListener('input',filter);$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};
fetch('samples.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{samples=data;for(const c of [...new Set(samples.map(s=>s.cluster))].sort((a,b)=>a-b)){const o=document.createElement('option');o.value=c;o.textContent='末层簇 '+c;$('cluster').appendChild(o)}filter()}).catch(e=>{$('count').className='error';$('count').textContent='数据加载失败，请通过本地 HTTP 报告链接打开：'+e.message});
</script></html>'''


if __name__ == '__main__':
    main()

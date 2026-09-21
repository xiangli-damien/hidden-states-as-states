"""Describe per-layer/per-cluster effective dimensions of frozen Qwen MFA fits.

Selected rank/K/models and assignments stay fixed. No correctness labels are
used. Empirical covariance describes hard assignments; MFA covariance describes
fitted soft components. PR is the primary definition throughout.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.effective_dimension import (
    covariance_dimensions, empirical_dimensions, matched_size_dimensions, weighted_quantile,
)
from hss.data.cache import CachedStates
from hss.experiments.artifacts import file_digest, runtime_versions, save_json
from hss.experiments.fitting import load_fitted
from hss.results.models import load_layer

METRICS = ['factor_pr', 'model_pr', 'empirical_pr', 'matched_pr_median']
TITLES = dict(factor_pr='因子部分有效维度', model_pr='完整 MFA 有效维度',
              empirical_pr='真实样本有效维度', matched_pr_median='每簇 100 题有效维度')


def summarize(frame, metrics):
    result = []
    for unit, part in frame.groupby('unit', sort=False):
        first = part.iloc[0]
        row = dict(unit=unit, position=int(first.position), view=first["view"],
            layer=int(first.layer), k=int(first.k), rank=int(first['rank']),
            n_clusters=len(part), n_questions=int(part.n.sum()),
            n_min=int(part.n.min()), n_max=int(part.n.max()))
        for key in metrics:
            valid = part[key].notna()
            p = part.loc[valid]
            row[key+'_coverage'] = float(p.n.sum()/part.n.sum())
            if not len(p): continue
            for q, value in zip([10, 25, 50, 75, 90], np.percentile(p[key], [10, 25, 50, 75, 90])):
                row[f'{key}_p{q}'] = float(value)
            row[key+'_min'], row[key+'_max'] = float(p[key].min()), float(p[key].max())
            row[key+'_weighted_median'] = float(weighted_quantile(p[key], p.n, [.5])[0])
            row[key+'_weighted_mean'] = float(np.average(p[key], weights=p.n))
        row['factor_fraction_median'] = float(part.factor_fraction.median())
        result.append(row)
    return pd.DataFrame(result).sort_values('position')


def run(cfg):
    start = time.monotonic(); study = Path(cfg['study']); out = Path(cfg['output'])
    if (out/'_SUCCESS.json').exists(): raise RuntimeError('Completed result exists; use --render-only')
    out.mkdir(parents=True, exist_ok=True); (out/'units').mkdir(exist_ok=True)
    selected = pd.read_csv(study/'selected_layers.csv')
    assert len(selected) == 30 and selected.converged.all() and (selected.initializations_completed == 3).all()
    exports = {k:Path(v) for k,v in json.loads((study/'latest_exports.json').read_text()).items()}
    source_hashes = {}
    def record(path):
        path = Path(path); source_hashes[str(path)] = file_digest(path)
    for name in ['selected_layers.csv', 'latest_exports.json', 'stability.csv']: record(study/name)
    caches, local, global_states, alignments = {}, {}, {}, {}
    ids = None
    for view in ['post', 'pre_final']:
        snapshot = study/'snapshots'/f'{view}.json'; record(snapshot)
        key = json.loads(snapshot.read_text())['key']; cache = CachedStates(Path(cfg['cache'])/key)
        caches[view] = cache
        assert cache.n_items() == cfg['expected_samples'] and cache.state_dim() == cfg['hidden_dim']
        meta = pd.read_parquet(exports[view]/'rows.parquet', columns=['sample_id'])
        np.testing.assert_array_equal(meta.sample_id, cache.meta.sample_id)
        assert meta.sample_id.is_unique
        if ids is None: ids = meta.sample_id.to_numpy()
        else: np.testing.assert_array_equal(ids, meta.sample_id)
        for name in ['rows.parquet', 'local_states.npy', 'states.npy', 'alignment.json', '_SUCCESS.json']:
            record(exports[view]/name)
        local[view] = np.load(exports[view]/'local_states.npy')
        global_states[view] = np.load(exports[view]/'states.npy')
        alignments[view] = json.loads((exports[view]/'alignment.json').read_text())
    pd.DataFrame({'sample_id':ids}).to_parquet(out/'sample_ids.parquet', index=False)
    protocol = dict(config=cfg, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime=runtime_versions(), model='Qwen/Qwen2-7B-Instruct', dataset='MATH5000',
        representation='Per-question generation-token mean; raw hidden states, no PCA/normalization.',
        selected='Frozen 24h MFA exports: 29 HF positions (0 embeddings, 1..27 residual outputs, 28 post final RMSNorm) plus separate pre-final-norm layer28. All selected ranks16.',
        primary='Participation ratio trace(C)^2 / trace(C^2). Not entropy rank and not an intrinsic-dimension estimator.',
        factor='C=W W.T. Formula uses eigenvalues of W.T W; PR <= configured rank.',
        model_covariance='C=W W.T+diag(psi). Exact trace identity, no dense D x D eigendecomposition; full PR may exceed rank.',
        empirical='Each hard-assigned cluster centered at its own empirical mean. Full finite-sample covariance; PR <= min(n-1,D).',
        equal_size='100 questions without replacement,20 seeded repeats per eligible cluster. Report median and10/90 percentiles of repeats, not confidence intervals. Clusters below100 excluded with coverage reported.',
        summaries='Per-layer unweighted cluster median/IQR/range and question-count-weighted summaries. They answer different questions.',
        seed_checks='Existing seeds1042/2042 with same per-layer rank/K. Only fitted covariance diagnostics, no new fits and no cluster-ID matching.',
        norm_control='Hold pre or post hard assignments fixed, calculate empirical PR in both stored mean representations. Separates representation change from reclustering; not RMS applied to the mean.',
        interpretation='Frozen descriptive maps fitted on all5000; no correctness labels, p-values or independent generalization claims. Rank16 was the search upper boundary.')
    save_json(out/'protocol.json', protocol)
    records, spectra, matched_values, inference = [], {}, {}, []
    assignments = {}
    for chosen in selected.itertuples(index=False):
        view, layer = chosen.view, int(chosen.layer)
        unit = 'L28_pre' if view == 'pre_final' else ('L28_post' if layer == 28 else f'L{layer:02d}')
        position = layer if unit != 'L28_post' else 29
        cache = caches[view]; x = cache.array(layer)
        record(cache.path/f'layer_{layer}.npy')
        model, projection, _ = load_layer(exports[view], layer)
        for f in ['model.npz', 'model.json', 'projection.npz', 'selection.json']:
            record(exports[view]/'models'/f'layer_{layer}'/f)
        assert model.converged_ and model.loadings_.shape == (chosen.k, cfg['hidden_dim'], chosen.rank)
        col = list(cache.layers()).index(layer); z = local[view][:,col]
        np.testing.assert_array_equal(z, np.load(Path(chosen.fit_path)/'assignments.npz')['posterior'])
        np.testing.assert_array_equal(np.asarray(alignments[view]['local_to_global'][col])[z], global_states[view][:,col])
        probe = np.sort(np.random.default_rng(cfg['seed']+position).choice(len(x),cfg['inference_audit_rows'],replace=False))
        np.testing.assert_array_equal(projection.transform(x[probe]), x[probe])
        np.testing.assert_array_equal(model.predict(x[probe]), z[probe])
        inference.append(dict(unit=unit, n_audited=len(probe), exact=True))
        assignments[unit] = z
        unit_records, unit_spectra = [], []
        for k in range(model.n_clusters()):
            idx = np.flatnonzero(z==k); assert len(idx) > 1
            part = np.asarray(x[idx], dtype=np.float64)
            values = covariance_dimensions(model.loadings_[k], model.noise_[k])
            unit_spectra.append(values.pop('factor_eigenvalues'))
            empirical = empirical_dimensions(part)
            repeated = matched_size_dimensions(part,cfg['matched_size'],cfg['matched_repeats'],cfg['seed']+1000*position+k)
            matched_values[f'{unit}_c{k}'] = repeated
            stats = dict(matched_pr_median=float(np.median(repeated)) if len(repeated) else None,
                         matched_pr_p10=float(np.quantile(repeated,.1)) if len(repeated) else None,
                         matched_pr_p90=float(np.quantile(repeated,.9)) if len(repeated) else None)
            entry = dict(unit=unit,position=position,view=view,layer=layer,k=int(chosen.k),cluster=k,
                n=len(idx),weight=float(model.weights_[k]),seed=int(chosen.seed),**values,**empirical,**stats)
            assert entry['factor_pr'] <= chosen.rank+1e-8
            assert entry['empirical_pr'] <= entry['empirical_rank_ceiling']+1e-7
            unit_records.append(entry)
        records.extend(unit_records); spectra[unit] = np.asarray(unit_spectra)
        save_json(out/'units'/f'{unit}.json',dict(clusters=unit_records,factor_eigenvalues=unit_spectra))
        print(json.dumps(dict(unit=unit,clusters=len(unit_records),factor_median=float(np.median([r['factor_pr'] for r in unit_records])),
            model_median=float(np.median([r['model_pr'] for r in unit_records])),empirical_median=float(np.median([r['empirical_pr'] for r in unit_records])),
            seconds=round(time.monotonic()-start,1))),flush=True)
    frame = pd.DataFrame(records).sort_values(['position','cluster'])
    frame.to_csv(out/'cluster_metrics.csv',index=False)
    summarize(frame,METRICS).to_csv(out/'layer_summary.csv',index=False)
    np.savez_compressed(out/'factor_spectra.npz',**spectra)
    np.savez_compressed(out/'matched_size_repeats.npz',**matched_values)
    np.savez_compressed(out/'assignments.npz',**assignments)
    seed_records=[]
    for row in pd.read_csv(study/'stability.csv').itertuples(index=False):
        unit='L28_pre' if row.view=='pre_final' else ('L28_post' if row.layer==28 else f'L{row.layer:02d}')
        path=Path(row.fit_path)
        for f in ['model.npz','fit.json','assignments.npz']:record(path/f)
        model,_=load_fitted(path); assert model.converged_
        z=np.load(path/'assignments.npz')['posterior'];assert len(z)==len(ids)
        for k in range(model.n_clusters()):
            v=covariance_dimensions(model.loadings_[k],model.noise_[k]);v.pop('factor_eigenvalues')
            seed_records.append(dict(unit=unit,position=29 if unit=='L28_post' else int(row.layer),view=row.view,
                layer=int(row.layer),k=int(row.k),cluster=k,n=int(np.sum(z==k)),seed=int(row.seed),**v))
    seeds=pd.DataFrame(seed_records);seeds.to_csv(out/'seed_cluster_metrics.csv',index=False)
    seed_summary=[]
    for seed,part in pd.concat([frame,seeds],ignore_index=True).groupby('seed'):
        summary=summarize(part,['factor_pr','model_pr']);summary['seed']=int(seed);seed_summary.append(summary)
    pd.concat(seed_summary,ignore_index=True).to_csv(out/'seed_layer_summary.csv',index=False)
    norm=[]; pre=caches['pre_final'].array(28);post=caches['post'].array(28)
    for owner in ['L28_pre','L28_post']:
        z=assignments[owner]
        for k in np.unique(z):
            idx=np.flatnonzero(z==k)
            a=empirical_dimensions(pre[idx])['empirical_pr'];b=empirical_dimensions(post[idx])['empirical_pr']
            norm.append(dict(assignment_source=owner,cluster=int(k),n=len(idx),pre_pr=a,post_pr=b,post_pre_ratio=b/a))
    pd.DataFrame(norm).to_csv(out/'norm_control.csv',index=False)
    save_json(out/'provenance.json',dict(source_hashes=source_hashes,inference_checks=inference,
        input_ids_sha256=file_digest(out/'sample_ids.parquet'),labels_used=False))
    summary=dict(units=len(frame.unit.unique()),clusters=len(frame),rank_values=sorted(frame['rank'].unique().astype(int).tolist()),
        hidden_dim=cfg['hidden_dim'],n_samples=len(ids),matched_size=cfg['matched_size'],matched_repeats=cfg['matched_repeats'],
        seed_fits=len(seeds[['unit','seed']].drop_duplicates()),seed_clusters=len(seeds),
        all_matched_eligible=bool(frame.matched_pr_median.notna().all()),minimum_cluster_n=int(frame.n.min()),seconds=time.monotonic()-start)
    save_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    render(out);audit(out)


def audit(out):
    frame=pd.read_csv(out/'cluster_metrics.csv');summary=pd.read_csv(out/'layer_summary.csv')
    assert len(summary)==30 and (summary.n_questions==5000).all()
    assert not frame.duplicated(['unit','cluster']).any()
    assert (frame.factor_pr<=frame['rank']+1e-8).all()
    assert np.isfinite(frame[['factor_pr','model_pr','empirical_pr']]).all().all()
    provenance=json.loads((out/'provenance.json').read_text())
    for path,digest in provenance['source_hashes'].items():assert file_digest(path)==digest,path
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ['inventory.json','_SUCCESS.json'])
    inventory=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=file_digest(p)) for p in files]
    save_json(out/'inventory.json',inventory)
    save_json(out/'_SUCCESS.json',dict(files=len(files),bytes=sum(r['bytes'] for r in inventory),
        inventory_sha256=file_digest(out/'inventory.json'),source_files_verified=len(provenance['source_hashes']),
        model_inference_audit_rows=sum(r['n_audited'] for r in provenance['inference_checks']),units=30,valid=True))


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.read_csv(out/'cluster_metrics.csv');layers=pd.read_csv(out/'layer_summary.csv')
    seeds=pd.read_csv(out/'seed_layer_summary.csv');norm=pd.read_csv(out/'norm_control.csv')
    summary=json.loads((out/'summary.json').read_text());spectra=np.load(out/'factor_spectra.npz')
    dest=out/'report';dest.mkdir(exist_ok=True)
    ticks=[0,4,8,12,16,20,24,27,28,29];labels=['0','4','8','12','16','20','24','27','28\npre','28\npost']
    names=['Factor covariance only','Full fitted MFA covariance','Empirical within-cluster covariance']
    fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True)
    for ax,key,title in zip(axes,METRICS[:3],names):
        for row in layers.itertuples():
            part=frame[frame.unit==row.unit].sort_values('cluster');x=row.position+np.linspace(-.24,.24,len(part))
            ax.scatter(x,part[key],s=10+50*part.n/part.n.max(),alpha=.65,color='#3576a4')
        ax.fill_between(layers.position,layers[key+'_p25'],layers[key+'_p75'],color='#3576a4',alpha=.12,label='Across-cluster IQR')
        ax.plot(layers.position,layers[key+'_p50'],color='#214d70',lw=1.7,label='Cluster-equal median')
        ax.plot(layers.position,layers[key+'_weighted_median'],color='#ba7236',ls='--',lw=1.4,label='Question-weighted median')
        ax.axhline(16,color='grey',ls=':',label='Configured factor rank=16')
        ax.axvspan(28.55,29.45,color='#eddfce',alpha=.5)
        ax.set_yscale('log',base=2);ax.set_ylabel('Participation dimension');ax.set_title(title,loc='left');ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.27))
    axes[-1].set_xticks(ticks,labels);axes[-1].set_xlabel('Layer0=embeddings; layers1..28=decoder residual outputs; final post-RMSNorm shown separately')
    fig.suptitle('Qwen2 MATH | fixed MFA rank16, independently selected K per layer\nEach dot is a cluster; dot size indicates sample count; bands are distributions, not confidence intervals',y=.995)
    fig.tight_layout(rect=[0,0,1,.97])
    for ext in ['png','pdf']:fig.savefig(dest/f'dimensions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for key,color,label in [('empirical_pr','#3576a4','All cluster observations'),('matched_pr_median','#ba7236','100 questions per eligible cluster (20 repeat medians)')]:
        axes[0].plot(layers.position,layers[key+'_p50'],label=label,color=color)
    axes[0].set_yscale('log',base=2);axes[0].set(xlabel='Layer position (29=final post norm)',ylabel='Cluster-median empirical PR',title='Sensitivity to cluster sample size');axes[0].legend(fontsize=8)
    for seed,part in seeds.groupby('seed'):
        part=part.sort_values('position');axes[1].plot(part.position,part.model_pr_p50,label=f'seed {seed}')
    axes[1].set_yscale('log',base=2);axes[1].set(xlabel='Layer position (29=final post norm)',ylabel='Cluster-median full model PR',title='Existing alternative initializations; no refitting');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'controls.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True)
    for ax,owner in zip(axes,['L28_pre','L28_post']):
        for r in norm[norm.assignment_source==owner].itertuples():
            ax.plot([0,1],[r.pre_pr,r.post_pr],'o-',alpha=.75,label=f'C{r.cluster} n={r.n}')
        ax.set_xticks([0,1],['Pre RMSNorm','Post RMSNorm']);ax.set_yscale('log',base=2)
        ax.set_title(f'Fixed {owner} cluster membership');ax.set_ylabel('Empirical participation dimension');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Final norm comparison with identical questions in each cluster');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'norm_control.{ext}',dpi=180)
    plt.close(fig)
    def clean(df):return df.replace({np.nan:None}).to_dict('records')
    data=dict(summary=summary,clusters=clean(frame),layers=clean(layers),seeds=clean(seeds),norm_control=clean(norm),
              spectra={k:spectra[k].tolist() for k in spectra.files})
    save_json(dest/'data.json',data)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MFA 逐层有效维度 · Qwen MATH</title>
<style>body{max-width:1320px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#203149;background:#f5f7fa}section{background:white;border:1px solid #dce3ec;border-radius:12px;padding:24px;margin:22px 0}h1{line-height:1.25}img{max-width:100%}.note{background:#fff4de;border-left:4px solid #d29a3b;padding:16px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf2f8}select{padding:8px;font:inherit;margin:5px}a{color:#245fac}svg{width:100%;max-width:760px;background:#fafcff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>rank 都是 16，簇的有效维度仍然不同吗？</h1><p>Qwen2-7B-Instruct × MATH 5,000 · 生成 token 均值 · 冻结全层 MFA · 原始 3,584 维</p>
<div class="note">本报告描述已拟合模型与簇内样本，不重新聚类，不用正确性标签。有效维度统一采用 participation ratio：tr(C)² / tr(C²)。它反映方差集中程度，不等于独立神经元数，也不是经过确认的真实内在维数。</div>
<section><h2>三个对象分别看</h2><p>① 因子部分 C=WWᵀ：有效维度 ≤ rank16。② 完整模型 C=WWᵀ+diag(ψ)：可以高于16。③ 真实样本：在每个硬分配簇内减去经验均值，计算完整经验协方差；不同于 MFA 的软成分协方差。</p><p id="summary"></p><img src="dimensions.png" alt="每层各簇因子、完整模型和经验有效维度分布"><p><a href="dimensions.pdf">导出 PDF</a></p><p>每点是一个簇，大小表示题目数；蓝线是簇等权中位数，橙线按每簇题目数加权。阴影为簇间四分位范围，不是置信区间。层0是 embedding；层28 pre/post 分开，避免把最终 RMSNorm 当成额外 decoder block。</p></section>
<section><h2>逐层汇总</h2><div class="scroll"><table><thead><tr><th>位置</th><th>K</th><th>每簇题数</th><th>因子 PR 中位数</th><th>完整模型 PR 中位数</th><th>经验 PR 中位数</th><th>100题对照中位数</th><th>覆盖题目</th></tr></thead><tbody id="layers"></tbody></table></div></section>
<section><h2>每层簇分布与因子谱</h2><label>位置<select id="unit"></select></label><label>维度<select id="metric"><option value="model_pr">完整模型 PR</option><option value="factor_pr">因子 PR</option><option value="empirical_pr">真实样本 PR</option><option value="matched_pr_median">100题对照 PR</option></select></label><div class="grid"><div><h3>簇等权经验累积分布</h3><svg id="cdf" viewBox="0 0 600 290" role="img" aria-label="每层簇有效维度累积分布"></svg><p>横轴为有效维度，纵轴为不超过该维度的簇比例。小样本簇同样占一个点；可结合题目加权汇总阅读。</p></div><div><h3>单簇相关因子的方差分配</h3><select id="cluster"></select><svg id="spectrum" viewBox="0 0 600 290" role="img" aria-label="因子协方差累计解释比例"></svg><p id="spectrumText"></p><p>这里的解释比例仅相对于 WWᵀ，不包含对角残余。因子旋转不会改变这条谱。</p></div></div><div class="scroll"><table><thead><tr><th>簇</th><th>题数</th><th>因子 PR</th><th>完整模型 PR</th><th>经验 PR</th><th>100题 PR</th><th>因子方差占比</th><th>因子90%方差维数</th></tr></thead><tbody id="clusters"></tbody></table></div></section>
<section><h2>样本量与初始化对照</h2><img src="controls.png" alt="每簇固定100题以及不同种子的对照"><p><a href="controls.pdf">导出 PDF</a></p><p>每簇均匀无放回抽100题，重复20次；不足100题的簇不进入这一对照，汇总表明确报告覆盖。重复分位数是抽样敏感性，不能当成置信区间。固定样本量能减少比较中的样本量差异，不能消除全部有限样本偏差。</p><p>另外使用已保存的1042／2042种子拟合（相同层的 K/rank 不变）检查层级分布敏感性，不重新拟合、不对应簇ID。曲线相近并不代表成员划分稳定。</p></section>
<section><h2>末层 RMSNorm：固定簇成员再比较</h2><img src="norm_control.png" alt="固定末层簇成员的pre和post经验维度比较"><p><a href="norm_control.pdf">导出 PDF</a></p><p>分别固定 pre 模型和 post 模型的簇成员，再在两种表征上计算相同问题集合的经验 PR。这样避免仅比较 K8 和 K7 引入重新划分的差异。这里使用分别保存的 pre/post token 均值，不是把 RMSNorm 直接作用于平均向量。</p></section>
<section><h2>复核与下载</h2><p><a href="../cluster_metrics.csv">逐簇指标 CSV</a> · <a href="../layer_summary.csv">逐层分布 CSV</a> · <a href="../seed_layer_summary.csv">种子对照</a> · <a href="../norm_control.csv">末层同成员对照</a> · <a href="../protocol.json">协议</a> · <a href="../provenance.json">来源与核验</a> · <a href="../inventory.json">文件清单</a></p><p>原 MFA 在全5,000题上拟合，本报告属于样本内描述。rank16是既有搜索上限，不代表真实维度为16。K随层独立选择、成员会变动；这里比较逐层分布，不声称追踪同一个簇的维度演化。历史 rank8/K21 的12.7中位数不是本次模型。</p></section>
<script>let data;const el=x=>document.getElementById(x),fmt=x=>x==null?'—':x.toFixed(2),ns='http://www.w3.org/2000/svg';function elem(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}function plot(id,points,xmax,ylabel){const svg=el(id);svg.replaceChildren();const sx=x=>55+510*x/xmax,sy=y=>245-210*y;svg.append(elem('path',{d:'M55 30V245H565',fill:'none',stroke:'#617287'}));for(let i=0;i<=4;i++){svg.append(elem('text',{x:48,y:sy(i/4)+4,'text-anchor':'end',fill:'#52657c','font-size':12},(i/4).toFixed(2)));svg.append(elem('text',{x:sx(xmax*i/4),y:266,'text-anchor':'middle',fill:'#52657c','font-size':12},fmt(xmax*i/4)))}svg.append(elem('text',{x:60,y:20,fill:'#52657c','font-size':13},ylabel));svg.append(elem('polyline',{points:points.map(p=>`${sx(p[0])},${sy(p[1])}`).join(' '),fill:'none',stroke:'#3275a2','stroke-width':2}));for(const p of points)svg.append(elem('circle',{cx:sx(p[0]),cy:sy(p[1]),r:2.5,fill:'#3275a2'}))}function spectrum(){const unit=el('unit').value,k=Number(el('cluster').value),s=data.spectra[unit][k],total=s.reduce((a,b)=>a+b,0);let sum=0;plot('spectrum',[[0,0],...s.map((v,i)=>[i+1,(sum+=v)/total])],s.length,'因子方差累计比例');const r=data.clusters.find(x=>x.unit===unit&&x.cluster===k);el('spectrumText').textContent=`C${k}：因子 PR ${fmt(r.factor_pr)}；第一方向贡献 ${(100*r.factor_top1_fraction).toFixed(1)}%；前 ${r.factor_d90} 个方向解释90%因子方差；因子部分占全部模型方差 ${(100*r.factor_fraction).toFixed(1)}%。`}function draw(){const p=data.clusters.filter(r=>r.unit===el('unit').value),key=el('metric').value,v=p.map(r=>r[key]).filter(v=>v!=null).sort((a,b)=>a-b);let points=[[0,0]];for(let i=0;i<v.length;i++)points.push([v[i],i/v.length],[v[i],(i+1)/v.length]);plot('cdf',points,Math.max(1,...v)*1.08,'累计簇比例');el('clusters').replaceChildren();el('cluster').replaceChildren();for(const r of p){const tr=document.createElement('tr');for(const value of [`C${r.cluster}`,r.n,fmt(r.factor_pr),fmt(r.model_pr),fmt(r.empirical_pr),fmt(r.matched_pr_median),(100*r.factor_fraction).toFixed(1)+'%',r.factor_d90]){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('clusters').append(tr);const o=document.createElement('option');o.value=r.cluster;o.textContent=`C${r.cluster} · ${r.n}题`;el('cluster').append(o)}spectrum()}fetch('data.json').then(r=>r.json()).then(d=>{data=d;const s=d.summary;el('summary').textContent=`已完成 ${s.units} 个表示位置、${s.clusters} 个簇，全部 rank=${s.rank_values.join('/')}；另检查 ${s.seed_fits} 个既有种子模型。最小簇 ${s.minimum_cluster_n} 题。`;for(const r of d.layers){const o=document.createElement('option');o.value=r.unit;o.textContent=r.unit;el('unit').append(o);const tr=document.createElement('tr');for(const v of [r.unit,r.k,`${r.n_min}–${r.n_max}`,fmt(r.factor_pr_p50),fmt(r.model_pr_p50),fmt(r.empirical_pr_p50),fmt(r.matched_pr_median_p50),(100*r.matched_pr_median_coverage).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=v;tr.append(td)}el('layers').append(tr)}el('unit').value='L28_pre';draw()}).catch(e=>el('summary').textContent='加载失败：'+e);el('unit').onchange=draw;el('metric').onchange=draw;el('cluster').onchange=spectrum;</script></html>'''
    (dest/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/mfa-effective-dimension.toml');p.add_argument('--output');p.add_argument('--render-only',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if a.output:cfg['output']=a.output
    with threadpool_limits(cfg['cpu_threads']):
        if a.render_only:render(Path(cfg['output']));audit(Path(cfg['output']))
        else:run(cfg)

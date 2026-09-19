"""Export an offline, per-cluster geometry report from saved Qwen MATH fits."""

import argparse
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hss.analysis.cluster_structure import empirical_spectrum, factor_structure, subspace_overlap, variance_decomposition
from hss.data import CachedStates
from hss.experiments.artifacts import file_digest, save_json
from hss.experiments.cluster_ablation import task_key
from hss.experiments.fitting import load_fitted


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snap = json.loads((source / "snapshots/post.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    cache_root = Path(protocol["identity"]["config"]["execution"]["cache_root"])
    data = CachedStates(cache_root / "data" / snap["key"])
    X = np.asarray(data.array(28), dtype=np.float64)
    if X.shape != (5000, 3584):
        raise ValueError("Expected complete Qwen2 MATH response means")
    meta = data.meta
    global_spec, axes = empirical_spectrum(X, 32)
    xy = (X - X.mean(0)) @ axes[:2].T
    np.savez(output / "display_projection.npz", coordinates=xy, axes=axes,
             center=X.mean(0), sample_ids=meta.sample_id.to_numpy(dtype=str))
    raw = {}
    for item in snap["identity"]["sources"]:
        rows = pd.read_parquet(Path(item["path"]) / "data.parquet",
                               columns=["sample_id", "problem", "response_text"])
        raw.update({str(r.sample_id): dict(question=str(r.problem), response=str(r.response_text))
                    for r in rows.itertuples()})
    methods = [("kmeans_k21", "kmeans", 0, 21), ("gmm_k21", "gmm", 0, 21),
               ("mfa_r8_k21", "mfa", 8, 21), ("kmeans_selected_k2", "kmeans", 0, 2),
               ("gmm_selected_k51", "gmm", 0, 51)]
    report = dict(source=str(source), created_at=time.time(), layer=28, view="post",
                  n_samples=len(X), hidden_dim=X.shape[1],
                  note="Raw full-dimensional response means. Shared PCA only for display; no refitting or normalization. All diagnostics are in-sample and descriptive.",
                  global_spectrum=global_spec, methods={}, driver_sha256=file_digest(Path(__file__)))
    members = meta.copy()
    all_summaries = []
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3), constrained_layout=True)
    for method_index, (name, method, rank, k) in enumerate(methods):
        task = dict(view="post", layer=28, method=method, rank=rank, k=k, seed=42)
        candidate_path = source / "candidates" / f"{task_key(task)}.json"
        candidate = json.loads(candidate_path.read_text())
        fit_path = Path(candidate["record"]["fit_path"])
        model, info = load_fitted(fit_path)
        if info["context"]["snapshot"] != snap["key"] or info["context"]["layer"] != 28:
            raise ValueError("Fit uses a different row snapshot or layer")
        if not info["model"].get("converged"):
            raise ValueError(f"Unconverged model: {name}")
        with np.load(fit_path / "assignments.npz") as f:
            y = f["posterior"].copy()
        if y.shape != (len(X),) or y.min() < 0 or y.max() >= k:
            raise ValueError("Invalid assignment identity/shape")
        members[name] = y
        entry = dict(k=k, rank=rank, fit_path=str(fit_path),
                     candidate_sha256=file_digest(candidate_path), scores=info["scores"],
                     decomposition=variance_decomposition(X, y), clusters=[])
        spectra, bases = [], []
        for cluster in range(k):
            idx = np.flatnonzero(y == cluster)
            if not len(idx):
                entry["clusters"].append(dict(cluster=cluster, n=0)); bases.append(None)
                continue
            spec, vectors = empirical_spectrum(X[idx], 32, seed=42 + cluster)
            center = X[idx].mean(0)
            dist = np.square(X[idx] - center).sum(1)
            example_ids = idx[np.argsort(dist)[:3]].tolist()
            counts = meta.iloc[idx].category.fillna("unknown").value_counts()
            item = dict(cluster=cluster, **spec,
                        pc1_fraction=spec["top_fraction"][0] if spec["top_fraction"] else 0,
                        top8_fraction=spec["top_fraction"][min(7, len(spec["top_fraction"]) - 1)] if spec["top_fraction"] else 0,
                        accuracy=float(meta.iloc[idx].label.mean()),
                        mean_tokens=float(meta.iloc[idx].n_tokens.mean()),
                        categories={str(a): int(b) for a,b in counts.items()},
                        examples=example_ids)
            if method == "mfa":
                factor, basis = factor_structure(model.loadings_[cluster], model.noise_[cluster])
                item["fitted_covariance"] = factor
                item["leading_factor_coordinates"] = np.argsort(np.abs(basis[:, 0]))[-10:][::-1].tolist() if basis.shape[1] else []
                bases.append(basis)
            else:
                bases.append(vectors[:min(rank or 8, len(vectors))].T)
            entry["clusters"].append(item)
            spectra.append((cluster, spec))
            all_summaries.append(dict(method=name, cluster=cluster, n=len(idx), trace=spec["trace"],
                                     pc1_fraction=item["pc1_fraction"], top8_fraction=item["top8_fraction"],
                                     accuracy=item["accuracy"], mean_tokens=item["mean_tokens"],
                                     factor_fraction=item.get("fitted_covariance", {}).get("factor_fraction")))
        report["methods"][name] = entry
        if method_index < 3:
            axs[method_index].scatter(xy[:,0], xy[:,1], c=y, cmap="turbo", s=4, alpha=.5, rasterized=True)
            axs[method_index].set(title=name.replace("_", " "), xlabel="Shared PC1", ylabel="Shared PC2")
        sf, sa = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
        for cluster,spec in spectra:
            sa[0].plot(np.arange(1,len(spec["eigenvalues"])+1), np.array(spec["eigenvalues"])/max(spec["trace"],1e-30), alpha=.55)
            sa[1].plot(np.arange(1,len(spec["top_fraction"])+1), spec["top_fraction"], alpha=.55)
        sa[0].set(xlabel="Within-cluster PC", ylabel="Variance / exact total trace", yscale="log", title=f"{name}: empirical spectra")
        sa[1].set(xlabel="Number of PCs", ylabel="Cumulative explained variance", ylim=(0,1.01), title="One curve per occupied cluster")
        sf.savefig(output/f"{name}_spectra.png",dpi=150);plt.close(sf)
        if method == "mfa":
            overlap = np.array([[subspace_overlap(a,b) if a is not None and b is not None else np.nan for b in bases] for a in bases])
            np.save(output/"mfa_factor_subspace_overlap.npy",overlap)
            ff, aa = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            occupied=[c for c in entry["clusters"] if c["n"]]
            aa[0].bar([c["cluster"] for c in occupied],[c["fitted_covariance"]["factor_fraction"] for c in occupied],color="#2a788e")
            aa[0].set(xlabel="Cluster",ylabel="trace(WWᵀ) / trace(Σ)",ylim=(0,1),title="Fitted MFA correlated variance fraction")
            im=aa[1].imshow(overlap,vmin=0,vmax=1,cmap="viridis");ff.colorbar(im,ax=aa[1])
            aa[1].set(xlabel="Cluster",ylabel="Cluster",title="Factor subspace overlap (rotation invariant)")
            ff.savefig(output/"mfa_factor_structure.png",dpi=150);plt.close(ff)
        print(json.dumps({'method_complete':name,'clusters':len(spectra)}),flush=True)
    axs[3].scatter(xy[:,0],xy[:,1],c=meta.label.to_numpy(),cmap="coolwarm",s=4,alpha=.5,vmin=0,vmax=1,rasterized=True)
    axs[3].set(title="Correctness (red=correct)",xlabel="Shared PC1",ylabel="Shared PC2")
    fig.suptitle(f"Qwen MATH · final post-RMS means · PCA display retains {100*global_spec['top_fraction'][1]:.1f}% variance")
    fig.savefig(output/"cluster_map.png",dpi=160);plt.close(fig)
    members.to_parquet(output/"members.parquet",index=False)
    pd.DataFrame(all_summaries).to_csv(output/"cluster_summary.csv",index=False)
    samples=[]
    for i,r in enumerate(meta.itertuples()):
        samples.append(dict(id=r.sample_id,correct=int(r.label),category=str(r.category),
                            tokens=int(r.n_tokens),**raw.get(str(r.sample_id),{})))
    report["samples"]=samples
    report["assignments"]={name:members[name].astype(int).tolist() for name,_,_,_ in methods}
    save_json(output/"structure.json",report)
    build_page(report,output)


def build_page(report, output):
    payload=json.dumps(report,ensure_ascii=False).replace('<','\\u003c')
    page=r'''<!doctype html><meta charset="utf-8"><title>Qwen MATH · Cluster structure</title>
<style>body{font:16px system-ui;background:#f5f7fa;color:#152b3c;max-width:1380px;margin:30px auto;padding:0 24px}h1{font-size:30px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}img{width:100%}select,input{font:inherit;padding:8px;margin:6px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;max-height:600px;overflow:auto}button{padding:8px;cursor:pointer}small{color:#526477}.warning{border-left:4px solid #e4a631;padding:12px;background:#fff8e9}</style>
<h1>Qwen2-7B-Instruct × MATH · 簇与方差结构</h1>
<p>5,000 条 · 最后一层 post-RMS · 原始 3,584 维生成均值 · 不重新聚类</p>
<p class="warning">PCA 只用于展示；聚类没有额外 normalization 或降维。图中重叠不代表高维无法区分。簇编号在不同方法之间不对应。正确率为同一批数据的描述统计，不能视为验证集预测性能。</p>
<section><h2>同一个二维视角，固定 K=21</h2><img src="cluster_map.png"><p id="projection"></p></section>
<section><h2>选择方法和簇</h2><select id="method"></select><select id="cluster"></select><div id="overview"></div><div id="stats"></div><img id="spectrum"><div id="factors"></div></section>
<section><h2>逐题查看</h2><p>当前簇所有样本；排序为原始数据顺序。</p><select id="sample"></select><pre id="example"></pre></section>
<section><h2>如何解读</h2><ul><li>trace 是簇内中心化方差总量；PC1 / top-8 是经验协方差的前 1 / 8 个方向解释的比例。谱来自 randomized SVD，分母使用精确总方差。</li><li>MFA 将拟合协方差分为 WWᵀ（相关方向）与 diag(ψ)（各坐标残余方差）；因子占比不等于“正确性解释比例”。</li><li>子空间重叠衡量不同簇是否沿相似方向变化。使用主夹角，避免把可旋转的 W 列误当作唯一 neuron 方向。</li><li>经验方差基于 hard assignment；MFA 协方差基于 soft mixture 拟合，两者应对照看，数值无需完全相等。</li><li>短回答、题型和难度均可能造成分簇；不能把簇正确率差异直接解释为因果机制。</li></ul><a href="cluster_summary.csv">簇摘要 CSV</a> · <a href="members.parquet">全部样本标签</a> · <a href="structure.json">完整数值与来源</a></section>
<script id="data" type="application/json">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);const pct=x=>(x*100).toFixed(1)+'%',num=x=>Number(x).toFixed(2);
Object.keys(d.methods).forEach(k=>$('method').add(new Option(k,k)));
$('projection').textContent='PC1 解释 '+pct(d.global_spectrum.top_fraction[0])+'，前两维合计 '+pct(d.global_spectrum.top_fraction[1])+'；剩余高维结构没有显示在散点图中。';
function changeMethod(){let m=d.methods[$('method').value];$('cluster').replaceChildren();m.clusters.forEach(c=>$('cluster').add(new Option('Cluster '+c.cluster+' · n='+c.n,c.cluster)));$('overview').textContent='总方差中，簇间均值差异占 '+pct(m.decomposition.between_fraction)+'；簇内剩余占 '+pct(1-m.decomposition.between_fraction)+'。';$('spectrum').src=$('method').value+'_spectra.png';$('factors').replaceChildren();if($('method').value.startsWith('mfa')){let img=new Image();img.src='mfa_factor_structure.png';$('factors').append(img)}changeCluster()}
function changeCluster(){let key=$('method').value,c=d.methods[key].clusters[Number($('cluster').value)];let s='n='+c.n;if(c.n){s+=' · 正确率 '+pct(c.accuracy)+' · 平均生成长度 '+num(c.mean_tokens)+' · 方差 trace '+num(c.trace)+' · PC1 '+pct(c.pc1_fraction)+' · top-8 '+pct(c.top8_fraction)+'\n题型：'+JSON.stringify(c.categories);if(c.fitted_covariance)s+='\nMFA 因子方差占比 '+pct(c.fitted_covariance.factor_fraction)}$('stats').textContent=s;$('stats').style.whiteSpace='pre-wrap';$('sample').replaceChildren();d.assignments[key].forEach((k,i)=>{if(k===c.cluster)$('sample').add(new Option(d.samples[i].id+' · '+(d.samples[i].correct?'正确':'错误'),i))});changeSample()}
function changeSample(){let x=d.samples[Number($('sample').value)];$('example').textContent=x?x.id+' · '+x.category+' · '+x.tokens+' tokens\n\n'+x.question+'\n\n模型回答：\n'+x.response:''}
$('method').onchange=changeMethod;$('cluster').onchange=changeCluster;$('sample').onchange=changeSample;changeMethod();</script>'''
    (output/'index.html').write_text(page.replace('PAYLOAD',payload))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(2):build(args.source,args.output)

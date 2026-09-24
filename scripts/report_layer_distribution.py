"""Render saved all-layer diagnostics; never loads raw activations or fits models."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hss.experiments.artifacts import file_digest, save_json


COLORS=['#1d4ed8','#0891b2','#e07a15','#7c3aed']


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)


def render(root):
    assert (root/'ANALYSIS_READY.json').exists()
    if (root/'_SUCCESS.json').exists():raise FileExistsError('Completed report is immutable')
    m=pd.read_csv(root/'layer_metrics.csv').sort_values('position')
    p=pd.read_csv(root/'partition_metrics.csv');c=pd.read_csv(root/'icl_curves.csv')
    meta=pd.read_parquet(root/'sample_ids.parquet')
    assert len(m)==30 and len(p)==120 and len(c)==2370 and len(meta)==5000
    report=root/'report';report.mkdir(exist_ok=True)
    x=m.position.to_numpy();ticks=[0,7,14,21,27,28,29];ticklabels=['0','7','14','21','27','28pre','28post']
    fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    ax=axs[0,0];ax.plot(x,m.norm_median,color=COLORS[0]);ax.fill_between(x,m.norm_q05,m.norm_q95,alpha=.16)
    ax.set(yscale='log',title='Vector norm: median and 5–95% range',ylabel='Norm of response mean')
    ax=axs[0,1];ax.plot(x,m.mean_energy_fraction,label='Global mean / total energy',color=COLORS[0]);ax.plot(x,m.raw_pair_cosine_mean,label='Mean cosine over all pairs',color=COLORS[2]);ax.legend(fontsize=9);ax.set(title='Shared offset and direction',ylim=(-.05,1.05))
    ax=axs[1,0];ax.plot(x,m.effective_dimension,label='Full covariance PR',color=COLORS[0]);ax.plot(x,m.length_residual_effective_dimension,label='After linear log-length removal',color=COLORS[2]);ax.legend(fontsize=9);ax.set(title='Global effective dimension',ylabel='Participation ratio')
    ax=axs[1,1];ax.plot(x,m.pc1_fraction,label='PC1',color=COLORS[2]);ax.plot(x,m.pc16_fraction,label='Top16',color=COLORS[0]);ax.legend();ax.set(title='Concentration of centered variance',ylim=(0,1))
    ax=axs[2,0];ax.plot(x,m.offdiagonal_covariance_fraction,color=COLORS[0]);ax.set(title='Covariance energy in off-diagonal entries',ylabel='Off-diagonal Frobenius fraction',ylim=(0,1))
    ax=axs[2,1];ax.plot(x,m.radius_q99_gaussian_ratio,label='Radius-squared q99 ratio',color=COLORS[0]);ax.plot(x,m.radius_fourth_ratio,label='Fourth-moment ratio',color=COLORS[2]);ax.axhline(1,color='gray',ls=':');ax.legend(fontsize=9);ax.set(title='Tail diagnostics vs fitted single Gaussian',ylabel='Observed / Gaussian reference')
    for ax in axs.flat:style(ax);ax.set_xticks(ticks,ticklabels,rotation=20);ax.set_xlabel('Layer / final norm side')
    fig.suptitle('Qwen2-7B-Instruct / MATH5000 — raw full-response means',fontsize=16)
    for ext in ['png','pdf']:fig.savefig(report/f'overview.{ext}',dpi=155)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for color,method in zip(COLORS,['gmm_0pct','gmm_2pct','gmm_5pct','mfa_rank16']):
        part=p[p.method==method].set_index('unit').loc[m.unit]
        axs[0,0].plot(x,part.between_variance_fraction,label=method,color=color)
        axs[0,1].plot(x,part.silhouette_mean,label=method,color=color)
        axs[1,0].plot(x,part.occupied_clusters,label=method,color=color)
    for budget,color in zip(['0.01','0.05','0.1'],COLORS):axs[1,1].plot(x,m[f'k_delta_{budget}'],label=f'ΔICL/(N D) ≤ {budget}',color=color)
    for ax,title in zip(axs.flat,['Between-cluster / total centered variance','Euclidean silhouette: same1024 questions','Frozen-map occupied components','K under common score-difference budgets']):
        ax.set(title=title);ax.legend(fontsize=8);ax.set_xticks(ticks,ticklabels,rotation=20);style(ax)
    axs[0,1].axhline(0,color='gray',ls=':')
    for ext in ['png','pdf']:fig.savefig(report/f'partitions.{ext}',dpi=155)
    plt.close(fig)
    checks=[]
    for row in m.to_dict('records'):
        unit=row['unit'];a=np.load(root/'units'/f'{unit}.npz');pc=a['pc_scores'];e=a['eigenvalues']
        assert e.shape==(3584,) and pc.shape==(5000,32) and np.isfinite(e).all()
        np.testing.assert_allclose(e.sum()**2/np.square(e).sum(),row['effective_dimension'],rtol=1e-9)
        np.testing.assert_allclose(a['radius_squared'].mean(),row['trace'],rtol=1e-9)
        fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
        ax=axs[0,0];ax.scatter(pc[:,0],pc[:,1],c=a['labels_mfa_rank16'],cmap='tab20',s=3,alpha=.45,rasterized=True)
        ax.set(title=f'PCA view, colored by frozen MFA (K={int(row["mfa_k"])})',xlabel='PC1',ylabel='PC2')
        ax=axs[0,1];pts=ax.scatter(pc[:,0],pc[:,1],c=np.log1p(meta.n_tokens),cmap='viridis',s=3,alpha=.5,rasterized=True);fig.colorbar(pts,ax=ax,label='log(1 + generation tokens)');ax.set(title='Same projection, colored by response length',xlabel='PC1',ylabel='PC2')
        ax=axs[0,2];ax.plot(np.arange(1,129),e[:128]/e.sum());ax.set(yscale='log',title='Variance spectrum: first128 of3584 directions',xlabel='Principal direction',ylabel='Fraction of total variance')
        ax=axs[1,0];observed=a['radius_squared']/row['trace'];gauss=a['gaussian_normalized_radius_squared'];lo=max(min(observed.min(),gauss.min()),1e-8);hi=max(observed.max(),gauss.max());bins=np.geomspace(lo,hi,65)
        ax.hist(observed,bins=bins,density=True,histtype='step',lw=1.6,label='Observed')
        for j,g in enumerate(gauss):ax.hist(g,bins=bins,density=True,histtype='step',color=COLORS[2],alpha=.55,label='Gaussian, same covariance' if j==0 else None)
        ax.set(xscale='log',title='Centered radial distribution (all observations)',xlabel='Squared radius / mean squared radius',ylabel='Density');ax.legend(fontsize=8)
        ax=axs[1,1];curve=c[c.unit==unit].sort_values('k');ax.plot(curve.k,curve.delta_per_dimension,color=COLORS[0]);ax.set(yscale='symlog',title='ICL excess relative to best candidate',xlabel='GMM components K',ylabel='ΔICL / (5000 × 3584)');ax.set_yscale('symlog',linthresh=.005)
        ax=axs[1,2]
        for key,label,color in [('raw_pair_cosines','Raw vectors',COLORS[0]),('centered_pair_cosines','After subtracting global mean',COLORS[2])]:ax.hist(a[key],bins=np.linspace(-1,1,61),density=True,histtype='step',label=label,color=color)
        ax.set(title='Same12000 question pairs',xlabel='Cosine similarity',ylabel='Density');ax.legend(fontsize=8)
        for ax in axs.flat:style(ax)
        fig.suptitle(f'{unit}: PR={row["effective_dimension"]:.2f}; first2 PCs explain {100*e[:2].sum()/e.sum():.1f}% — projection is not a clustering test',fontsize=15)
        fig.savefig(report/f'{unit}.png',dpi=130);plt.close(fig)
        checks.append(dict(unit=unit,full_spectrum=True,trace_and_pr_verified=True))
    selected=m[m.unit.isin(['L07','L14','L21','L22','L27','L28_pre','L28_post'])]
    columns=['unit','effective_dimension','diagonal_effective_dimension','mean_energy_fraction','offdiagonal_covariance_fraction','radius_q99_gaussian_ratio','log_length_variance_fraction','k_percent_0.0','k_percent_0.02','k_delta_0.05','mfa_k']
    save_json(root/'findings.json',dict(representative_layers=selected[columns].to_dict('records'),
              scope='Descriptive; no Gaussian-refit/rotation-refit control, no causal or correctness conclusion'))
    names={'unit':'位置','effective_dimension':'全局PR','diagonal_effective_dimension':'只保留对角项PR','mean_energy_fraction':'公共均值能量占比',
           'offdiagonal_covariance_fraction':'非对角协方差占比','radius_q99_gaussian_ratio':'q99径向比值','log_length_variance_fraction':'log长度线性解释比例',
           'k_percent_0.0':'GMM严格K','k_percent_0.02':'GMM2% K','k_delta_0.05':'共同Δ预算 K','mfa_k':'MFA K'}
    table=m[columns].rename(columns=names).to_html(index=False,float_format=lambda v:f'{v:.3f}',border=0)
    options=''.join(f'<option>{html.escape(u)}</option>' for u in m.unit)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen–MATH 全层向量分布</title>
<style>body{font:16px/1.7 system-ui,sans-serif;color:#172033;background:#f7f9fc;margin:0}main{max-width:1280px;margin:auto;padding:30px}h1{font-size:30px}h2{margin-top:36px}img{width:100%;background:white;border-radius:8px}.note{background:#e9f0fa;padding:18px;border-left:4px solid #2563eb}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#edf2fa}select{font-size:18px;padding:8px}a{color:#1d4ed8}.scroll{overflow:auto}li{margin:8px 0}</style><main>
<h1>Qwen–MATH：全层向量分布有什么不同？</h1>
<p>Qwen2-7B-Instruct · MATH全部5,000题 · 完整生成token均值 · 原始3,584维 · 30个表示位置</p>
<div class="note">本页区分：公共方向、题间方差、相关性、长尾和已有簇的分离程度。模型输入没有额外normalization。协方差中心化、余弦归一化及PCA只用于诊断。0为embedding，28pre和28post分别为最终RMSNorm前后。</div>
<h2>1. 全局点云：尺度、方差方向和长尾</h2><img src="overview.png" alt="全层分布概览"><p><a href="overview.pdf">下载PDF</a></p>
<ul><li><b>全局PR</b>反映全部题目的方差集中度，与“每个簇内部PR”不同，也不是已确认的内在维数。</li><li><b>非对角占比</b>是协方差矩阵平方元素中非对角项的份额，不是“相关性解释了多少语义信息”。它依赖坐标系。</li><li><b>径向比值</b>比较观察到的尾部与一个具有同样完整协方差的单高斯。大于1提示该径向统计更重尾；它不是显著性检验，也不能单独证明多峰。</li><li><b>长度调整</b>只移除log长度的一维线性投影，未排除所有非线性、题型或难度混杂。</li></ul>
<h2>2. 已有分区与ICL计分影响</h2><img src="partitions.png" alt="分区和共同ICL预算"><p><a href="partitions.pdf">下载PDF</a></p>
<p>共同预算使用 ΔICL/(N×D)，扣掉每层最优分数基线，避免按|ICL|百分比带来的不同容忍度。预算0.01/0.05/0.1是诊断设置，没有根据正确性选阈值。所有分区都是样本内诊断；silhouette受簇形状影响。</p>
<h2>3. 逐层查看</h2><label for="unit">选择位置：</label><select id="unit">'''+options+'''</select><img id="detail" src="L00.png" alt="逐层分布详情"><p>PCA散点图仅显示前两个方向；颜色沿用已有MFA分区或回答长度。图中的色块不构成离散状态的证据。</p>
<h2>4. 全部数值</h2><p>共同Δ预算K使用 ΔICL/(N×D)≤0.05。</p><div class="scroll">'''+table+'''</div>
<h2>5. 保存与边界</h2><p><a href="../layer_metrics.csv">逐层指标</a> · <a href="../partition_metrics.csv">分区指标</a> · <a href="../icl_curves.csv">完整ICL曲线</a> · <a href="../protocol.json">协议</a> · <a href="../analysis_audit.json">分析审计</a> · <a href="../report_audit.json">报告审计</a></p>
<p>本轮没有使用正确性标签，没有新拟合GMM/MFA，也没有给模拟单高斯重新拟合GMM。已有地图使用过全部5,000题。本报告刻画分布，不能据此宣称不同功能状态、因果机制或独立测试预测能力。</p>
<p>后续验证可在固定样本划分上比较真实点云、匹配协方差的单高斯和正交旋转后的点云，再独立拟合同预算GMM/MFA，以区分密度多峰与协方差近似。</p>
<script>document.getElementById('unit').onchange=e=>{document.getElementById('detail').src=e.target.value+'.png'};</script></main></html>'''
    (report/'index.html').write_text(page)
    save_json(root/'report_audit.json',dict(valid=True,units=checks,metrics_rows=30,partition_rows=120,
        source_analysis_audit_sha256=file_digest(root/'analysis_audit.json')))
    inventory={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=file_digest(f))
               for f in root.rglob('*') if f.is_file() and f.name not in ['.lock','_SUCCESS.json','inventory.json','status.json']}
    save_json(root/'inventory.json',inventory)
    save_json(root/'_SUCCESS.json',dict(status='complete',units=30,report='report/index.html',
              inventory_sha256=file_digest(root/'inventory.json')))
    print(json.dumps({'status':'complete','report':str(report/'index.html')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    render(parser.parse_args().root)

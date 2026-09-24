"""Exact full-dimensional geometry and standalone figures for native examples."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def angles(x, y):
    x = np.atleast_2d(x)
    return np.degrees(np.arccos(np.clip(x @ y / (np.linalg.norm(x, axis=1)*np.linalg.norm(y)), -1, 1)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--inline-path',type=Path); args=p.parse_args(); root=args.root
    meta=json.loads((root/'examples.json').read_text())
    font=Path('/System/Library/Fonts/STHeiti Light.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    report=root/'report'; report.mkdir(exist_ok=True)
    colors=['#2563eb','#d97706','#059669']; cases=[]; audits=[]
    overview, ax=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for j,r in enumerate(meta['cases']):
        f=root/r['array_file']; assert hashlib.sha256(f.read_bytes()).hexdigest()==r['array_sha256']
        with np.load(f) as z: x=z['x'].astype(np.float64)
        n,d=x.shape; h=x[:16]; m=h.mean(0); full=x.mean(0)
        norms=np.linalg.norm(x,axis=1); mnorm=float(np.linalg.norm(m)); fnorm=float(np.linalg.norm(full))
        a16=angles(x,m); af=angles(x,full); running=x.cumsum(0)/np.arange(1,n+1)[:,None]
        # Check norm decomposition and independent Gram computation of mean energy.
        residual=h-m; energy=float(np.mean(np.sum(h*h,axis=1)))
        np.testing.assert_allclose(energy,mnorm*mnorm+np.mean(np.sum(residual*residual,axis=1)),rtol=1e-12)
        np.testing.assert_allclose(np.sum(h@h.T)/16**2,mnorm*mnorm,rtol=1e-12)
        assert mnorm<=norms[:16].mean()+1e-10 and fnorm<=norms.mean()+1e-10
        directions=h/norms[:16,None]; pair=directions@directions.T
        combined=np.vstack([h,m,full]); unit=combined/np.linalg.norm(combined,axis=1)[:,None]
        c={k:r[k] for k in ['sample_id','problem','response_text','ground_truth','tokens','n_tokens','category']}
        c.update(norms=norms.tolist(), mean16_norm=mnorm, full_mean_norm=fnorm,
            mean16_angle=a16.tolist(),full_mean_angle=af.tolist(),
            prefix_full_angle=float(angles(m,full)[0]),
            prefix_norm_retention=float(mnorm/norms[:16].mean()),
            full_norm_retention=float(fnorm/norms.mean()),
            prefix_energy_retention=float(mnorm*mnorm/energy),
            mean16_average_token_norm=float(norms[:16].mean()),
            mean_full_average_token_norm=float(norms.mean()),
            prefix_pairwise_cosine=float(pair[np.triu_indices(16,1)].mean()),
            first16_cosine=(unit@unit.T).tolist(),
            running_norm=np.linalg.norm(running,axis=1).tolist(),
            running_angle_to_full=angles(running,full).tolist())
        cases.append(c); audits.append({'sample_id':r['sample_id'],'sha_verified':True,'norm_energy_and_gram_identity':True})
        for target in [ax[j]]:
            left,right=target; t=np.arange(1,17)
            left.bar(t,norms[:16],color=colors[0],alpha=.75,label='单个 token')
            left.axhline(mnorm,color=colors[1],lw=2,label='前16均值')
            left.axhline(fnorm,color=colors[2],lw=2,ls='--',label='整段均值')
            left.set(xlabel='生成 token 位置',ylabel='原始向量长度（L2 norm）',xticks=[1,4,8,12,16],ylim=(0,max(norms[:16])*1.42))
            left.set_title(f'{r["sample_id"]} · 完整回答 {n} tokens',loc='left')
            left.legend(fontsize=9,loc='upper right')
            right.plot(t,a16[:16],'-o',color=colors[1],label='与前16均值的夹角',ms=4)
            right.plot(t,af[:16],'--s',color=colors[2],label='与整段均值的夹角',ms=4)
            right.set(xlabel='生成 token 位置',ylabel='原始3584维夹角（度）',xticks=[1,4,8,12,16],ylim=(0,max(np.max(a16[:16]),np.max(af[:16]))*1.2))
            right.set_title(f'两个均值之间：{c["prefix_full_angle"]:.1f}°',loc='left')
            right.legend(fontsize=9)
        fig, axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
        im=axes[0].imshow(unit@unit.T,vmin=-1,vmax=1,cmap='coolwarm_r')
        axes[0].set(xticks=[0,3,7,11,15,16,17],xticklabels=['1','4','8','12','16','均16','全均'],
            yticks=[0,3,7,11,15,16,17],yticklabels=['1','4','8','12','16','均16','全均'],
            title='前16 token 与两个均值：方向余弦')
        plt.setp(axes[0].get_xticklabels(),rotation=50,ha='right')
        fig.colorbar(im,ax=axes[0],label='cosine；1同向，0垂直，−1反向')
        axes[1].plot(np.arange(1,n+1),angles(running,full),color=colors[2])
        axes[1].axvline(16,color=colors[1],ls='--',label='当前干预位置16')
        axes[1].set(xlabel='累计到第几个生成 token',ylabel='累计均值与完整回答均值的夹角（度）',title='同一道题：累计均值的方向如何变化')
        axes[1].legend(); fig.suptitle(r['sample_id']+' · 真实3584维测量；非降维投影')
        fig.savefig(report/(r['sample_id']+'.png'),dpi=150);fig.savefig(report/(r['sample_id']+'.pdf'));plt.close(fig)
    overview.suptitle('Qwen2-7B-Instruct · MATH · block14 原始激活\n三道预先固定例子：单个 token、前16均值、完整回答均值',fontsize=16)
    overview.savefig(report/'overview.png',dpi=160);overview.savefig(report/'overview.pdf');plt.close(overview)
    out={k:meta[k] for k in ['model','revision','layer','hidden_dim','selection','scope']};out['cases']=cases
    (root/'geometry.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    (root/'geometry_audit.json').write_text(json.dumps({'checks':audits,'rows':sum(c['n_tokens'] for c in cases),'source_metadata_sha256':hashlib.sha256((root/'examples.json').read_bytes()).hexdigest()},indent=2))
    compact={'cases':[{k:(c[k][:16] if k in ['tokens','norms','mean16_angle','full_mean_angle'] else c[k])
        for k in ['sample_id','n_tokens','tokens','norms','mean16_norm','full_mean_norm','mean16_angle','full_mean_angle','prefix_full_angle','prefix_norm_retention']} for c in cases]}
    template=Path(__file__).with_name('vector_examples_inline.html').read_text()
    fragment=template.replace('__GEOMETRY_DATA__',json.dumps(compact,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (report/'inline.html').write_text(fragment)
    if args.inline_path: args.inline_path.write_text(fragment)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实 token 与均值向量</title>',
        '<style>body{font:17px/1.65 system-ui;max-width:1080px;margin:32px auto;padding:0 20px;color:#17212b}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{padding:7px;text-align:left;border-bottom:1px solid #ccd3d9}a{color:#245dba}</style>',
        '<h1>三个真实例子：token 向量与均值</h1><p>Qwen2-7B-Instruct，MATH，原 block14 输出，3584维，没有 normalization。读取已有全回答 teacher-forced 激活，没有重新生成或拟合模型。选用旧实验名单最先三道MATH题，未按结果或正误筛选。</p>',
        '<p><a href="interactive.html">打开可切换 token 的精确双向量图</a> · <a href="overview.pdf">下载总览 PDF</a> · <a href="../geometry.json">完整数值</a></p>',
        '<h2>怎么读</h2><p>长度表示向量的总幅度；夹角表示朝向差异，0°同向、90°垂直、180°反向。它们不直接代表语义相似度或正确性。两根高维向量可以在其张成的平面中无损画出长度和夹角；交互图两个面板分别使用自己的平面，不能比较面板之间的朝向。</p>',
        '<p>蓝柱是单个token的长度，橙线是前16个token的均值长度，绿虚线是完整回答均值长度。先平均向量再取长度，与先取长度再平均不同；朝向不完全一致时，均值通常更短。长度比例不是“保留的信息百分比”。</p>',
        '<img src="overview.png" alt="三个例子的长度和原始高维夹角"><p>每题完整回答包括EOS，前16均值和全回答均值来自同一次原生前向；没有混用重新prefill的激活。三题共1329个token，不代表整个数据集的统计结论。</p>']
    for c in cases:
        parts.extend([f'<h2>{c["sample_id"]} · {c["n_tokens"]} tokens</h2>',f'<p>{html.escape(c["problem"])}</p>',
            f'<p>前16个token平均长度 {c["mean16_average_token_norm"]:.2f}；前16均值长度 {c["mean16_norm"]:.2f}；全回答均值长度 {c["full_mean_norm"]:.2f}；两个均值的夹角 {c["prefix_full_angle"]:.2f}°。</p>',
            '<table><tr><th>位置</th><th>token片段</th><th>长度</th><th>对前16均值角度</th><th>对全均值角度</th></tr>'])
        for i in range(16):
            parts.append(f'<tr><td>{i+1}</td><td><code>{html.escape(repr(c["tokens"][i]))}</code></td><td>{c["norms"][i]:.2f}</td><td>{c["mean16_angle"][i]:.2f}°</td><td>{c["full_mean_angle"][i]:.2f}°</td></tr>')
        parts.extend(['</table>',f'<img src="{c["sample_id"]}.png" alt="方向余弦矩阵和累计均值变化">',
            '<p>累计均值最后与全回答均值完全相等，是定义决定的；这条曲线不能单独证明模型在向某个目标收敛。</p>',
            f'<details><summary>原回答全文（未在本次重新判分）</summary><pre>{html.escape(c["response_text"])}</pre></details>'])
    parts.append('</html>');(report/'index.html').write_text('\n'.join(parts))
    print(json.dumps([{k:c[k] for k in ['sample_id','mean16_average_token_norm','mean16_norm','full_mean_norm','prefix_full_angle','prefix_norm_retention','full_norm_retention','prefix_pairwise_cosine']} for c in cases],indent=2))


if __name__=='__main__':main()

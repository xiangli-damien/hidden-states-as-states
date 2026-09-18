"""Exact geometry identities and approximate FP64 normalization counterfactuals.

Frozen responses, no decoder intervention: effects on geometric metrics do NOT
establish effects on answer correctness. Actual BF16 post means remain primary.
"""
from concurrent.futures import ThreadPoolExecutor
import html
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import roc_auc_score

from hss.analysis.channel_data import load_config, ChannelData
from hss.analysis.confidence_data import checkpoint, read_tensor
from hss.analysis.confidence_study import partitions
from hss.analysis.confidence_bootstrap import auc_samples
from hss.analysis.mean_geometry import trajectory_scores, within_length_auc, NAMES
from hss.experiments.artifacts import save_json, digest, file_digest, runtime_versions


def token_factors(x, gamma, eps, v):
    """One response. Lengths and coherence factorize the response mean exactly."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0) or not np.isfinite(x).all(): raise ValueError('Invalid token vector')
    rms = np.sqrt(np.mean(x*x, axis=1)+eps)
    u = x/rms[:, None]
    y = u*gamma
    mu, mx, my = u.mean(0), x.mean(0), y.mean(0)
    un, yn = np.linalg.norm(u, axis=1), np.linalg.norm(y, axis=1)
    p = x@v
    # Remove only the weak-readout component from the RMS denominator, holding
    # the numerator fixed. This isolates denominator arithmetic, not a decoder.
    residual_norm2 = np.maximum(norms**2-p**2, 0)
    r_without_v = np.sqrt(residual_norm2/x.shape[1]+eps)
    y_fixed_denominator = x/r_without_v[:, None]*gamma
    stats = dict(mean_pre_token_norm=float(norms.mean()), mean_rms=float(rms.mean()),
                 pre_mean_norm=float(np.linalg.norm(mx)),
                 pre_coherence=float(np.linalg.norm(mx)/norms.mean()),
                 mean_u_token_norm=float(un.mean()), u_mean_norm=float(np.linalg.norm(mu)),
                 u_coherence=float(np.linalg.norm(mu)/un.mean()),
                 mean_post_token_norm=float(yn.mean()), post_mean_norm=float(np.linalg.norm(my)),
                 post_coherence=float(np.linalg.norm(my)/yn.mean()),
                 mean_gain=float(np.linalg.norm(my)/np.linalg.norm(mu)),
                 v_token_energy_fraction=float(np.mean(p*p/(norms**2))),
                 v_mean_energy_fraction=float((mx@v)**2/(mx@mx)),
                 no_v_denominator_mean_norm=float(np.linalg.norm(y_fixed_denominator.mean(0))))
    np.testing.assert_allclose(stats['post_mean_norm'],stats['mean_post_token_norm']*stats['post_coherence'],rtol=1e-12)
    return stats, mu, np.mean(u*u, axis=0), mx, my


def extract_shard(job):
    cfg, ds, data_info, shard, gamma, eps, v = job
    name = ds['name']; source = Path(ds['path'])/shard
    dest = Path(cfg['cache_root'])/name/shard
    key = digest({'source_key':data_info['source_keys'][data_info['shards'].index(shard)],
                  'model':data_info['model'],'prefix':cfg['prefix_tokens'],
                  'gamma':gamma.tolist(),'v':v.tolist(),'eps':eps,
                  'extraction_code':digest({'factors':inspect.getsource(token_factors),'shard':inspect.getsource(extract_shard)})})
    if (dest/'_SUCCESS.json').exists():
        old=json.loads((dest/'_SUCCESS.json').read_text())
        if old['key'] != key: raise ValueError('Stale RMS cache: '+str(dest))
        return old
    start=time.monotonic()
    z=zarr.open_consolidated(str(source/'tensors.zarr'),mode='r')
    ptr=np.asarray(z['tokens/sample_ptr'][:],dtype=int)
    # Read only final pre-RMS, never the full token x layer tensor.
    pre=np.asarray(z['final_norm/pre/per_token'][:])
    if pre.shape[0] != ptr[-1]: raise ValueError('Token alignment mismatch')
    rows=pd.read_parquet(source/'data.parquet',columns=['sample_id'])
    post_means=np.asarray(z['final_norm/post/mean'][:])
    pre_means=np.asarray(z['final_norm/pre/mean'][:])
    stats=[]; means=[]; energies=[]; prefix_means=[]; checks=[]
    for j,(a,b) in enumerate(zip(ptr[:-1],ptr[1:])):
        s,mu,energy,mx,my=token_factors(pre[a:b],gamma,eps,v)
        s['sample_id']=str(rows.sample_id.iloc[j]);s['n_tokens']=int(b-a)
        s['ideal_vs_saved_post_relative_error']=float(np.linalg.norm(my-post_means[j])/np.linalg.norm(post_means[j]))
        s['raw_mean_max_error']=float(np.max(np.abs(mx-pre_means[j])))
        if s['ideal_vs_saved_post_relative_error'] > .02: raise ValueError('RMS formula fails saved-state check')
        np.testing.assert_allclose(mx,pre_means[j],rtol=1e-4,atol=1e-5)
        if b-a>=cfg['prefix_tokens']:
            ps,pm,_,_,_=token_factors(pre[a:a+cfg['prefix_tokens']],gamma,eps,v)
            s.update({'prefix_'+k:val for k,val in ps.items()})
        else: pm=np.full_like(mu,np.nan)
        stats.append(s);means.append(mu);energies.append(energy);prefix_means.append(pm)
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(stats).to_parquet(dest/'scalars.parquet',index=False)
    np.savez(dest/'means.npz',u=np.asarray(means),u2=np.asarray(energies),prefix_u=np.asarray(prefix_means))
    result={'key':key,'dataset':name,'shard':shard,'n':len(rows),'tokens':int(ptr[-1]),
            'seconds':time.monotonic()-start,'ideal_vs_saved_post_relative_error_max':max(s['ideal_vs_saved_post_relative_error'] for s in stats)}
    save_json(dest/'_SUCCESS.json',result)
    print(json.dumps(result),flush=True)
    return result


def extract(cfg, limit_shards=0):
    cc=load_config(cfg['channel_config']);fc=load_config(cfg['confidence_config'])
    for name in cfg['datasets']:
        ds=next(s for s in cc['datasets'] if s['name']==name);data=ChannelData(cc,name)
        path=checkpoint(fc,data.info['model']);mc=json.loads((path/'config.json').read_text())
        gamma=read_tensor(path,'model.norm.weight').astype(float)
        geo=np.load(Path(fc['output_root'])/name/'readout_geometry.npz')
        np.testing.assert_array_equal(gamma,geo['gamma'])
        shards=data.info['shards'][:limit_shards] if limit_shards else data.info['shards']
        jobs=[(cfg,ds,data.info,s,gamma,mc['rms_norm_eps'],geo['v_min']) for s in shards]
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: results=list(pool.map(extract_shard,jobs))
        if not limit_shards:
            save_json(Path(cfg['cache_root'])/name/'_SUCCESS.json',{'sources':data.info,'shards':shards,'results':results})


def boot_log_decomposition(frame,y,cfg):
    # Exact per-response identity: log ||mean y|| = log mean||y|| + log coherence.
    fields=['post_mean_norm','mean_post_token_norm','post_coherence','u_mean_norm','mean_gain']
    logs=np.log(frame[fields].to_numpy(float));groups=[np.flatnonzero(y==j) for j in [0,1]]
    rng=np.random.default_rng(cfg['seed']);rep=[]
    for _ in range(cfg['bootstrap']):
        a,b=[rng.choice(g,len(g),replace=True) for g in groups]
        rep.append(logs[b].mean(0)-logs[a].mean(0))
    rep=np.asarray(rep);point=logs[y==1].mean(0)-logs[y==0].mean(0)
    np.testing.assert_allclose(point[0],point[1]+point[2],atol=1e-12)
    np.testing.assert_allclose(point[0],point[3]+point[4],atol=1e-12)
    return {k:{'log_correct_minus_incorrect':float(point[i]),'ci':np.quantile(rep[:,i],[.025,.975]).tolist(),
               'geometric_correct_over_incorrect':float(np.exp(point[i]))} for i,k in enumerate(fields)}


def score_result(y,s,test,cfg,bins):
    draws=auc_samples(y[test],s[test],cfg['seed'],cfg['bootstrap'])
    return {'auc':float(roc_auc_score(y[test],s[test])),'ci':np.quantile(draws,[.025,.975]).tolist(),
            'n':len(test),'within_length_bin_auc':within_length_auc(y[test],s[test],bins[test])}


def analyse(cfg):
    cc=load_config(cfg['channel_config']);fc=load_config(cfg['confidence_config']);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    all_results={}
    for name in cfg['datasets']:
        data=ChannelData(cc,name);train,test,saved=partitions(cfg,name,data.rows);y=data.rows.y.to_numpy(int)
        cache=Path(cfg['cache_root'])/name
        identity=json.loads((cache/'_SUCCESS.json').read_text())
        if identity['shards'] != data.info['shards'] or identity['sources']['source_keys'] != data.info['source_keys']: raise ValueError('Cache snapshot mismatch')
        frames=[pd.read_parquet(cache/s/'scalars.parquet') for s in data.info['shards']]
        scalars=pd.concat(frames,ignore_index=True).iloc[data._indices].reset_index(drop=True)
        if not np.array_equal(scalars.sample_id,data.rows.sample_id): raise ValueError('Identity mismatch')
        u=np.concatenate([np.load(cache/s/'means.npz')['u'] for s in data.info['shards']])[data._indices]
        u2=np.concatenate([np.load(cache/s/'means.npz')['u2'] for s in data.info['shards']])[data._indices]
        geo=np.load(Path(fc['output_root'])/name/'readout_geometry.npz');gamma=geo['gamma'].astype(float);v=geo['v_min']
        means=data.array('mean');pre=data.array('pre_mean').astype(float)
        post=means[:,-1].astype(float);ideal=u*gamma
        length=data.rows.n_tokens.to_numpy();edges=np.unique(np.quantile(length[train],np.linspace(0,1,11)))[1:-1];bins=np.searchsorted(edges,length,side='right')
        samples=data.rows.copy();samples['partition']=saved.partition.to_numpy()
        for col in scalars:
            if col not in samples: samples[col]=scalars[col]
        conditions={'raw':pre,'gamma_only':pre*gamma,'rms_only':u,'rms_gamma_ideal':ideal,'rms_gamma_actual':post}
        result={'n':len(y),'test_n':len(test),'conditions':{},'scalars':{},'length_edges':edges.tolist()}
        # Keep every other layer fixed for NDR and CoE.
        for condition,endpoint in conditions.items():
            chunks=[]
            for start in range(0,len(y),cfg['batch_size']):
                x=means[start:start+cfg['batch_size']].astype(float);x[:,-1]=endpoint[start:start+len(x)]
                chunks.append(pd.DataFrame(trajectory_scores(x)))
            values=pd.concat(chunks,ignore_index=True)
            for metric in ['ndr','coe_r','coe_c','negative_final_norm']:
                key=condition+'_'+metric;samples[key]=values[metric].to_numpy()
                result['conditions'][key]=score_result(y,values[metric].to_numpy(),test,cfg,bins)
        scalar_fields=['mean_pre_token_norm','mean_rms','pre_coherence','u_coherence','mean_post_token_norm','post_coherence','mean_gain','v_token_energy_fraction','v_mean_energy_fraction','no_v_denominator_mean_norm']
        for col in scalar_fields:
            a=samples[col].to_numpy(float)
            # Diagnostic raw increasing orientation, never choose a sign using test labels.
            result['scalars'][col]={'correct_mean':float(a[y==1].mean()),'incorrect_mean':float(a[y==0].mean()),
                 'increasing_auc':score_result(y,a,test,cfg,bins)}
        result['log_decomposition_all']=boot_log_decomposition(samples,y,cfg)
        result['log_decomposition_test']=boot_log_decomposition(samples.iloc[test],y[test],cfg)
        eligible=samples.n_tokens.to_numpy()>=cfg['prefix_tokens'];pt=test[eligible[test]]
        pframe=samples.loc[eligible,["prefix_"+c for c in ['post_mean_norm','mean_post_token_norm','post_coherence','u_mean_norm','mean_gain']]].rename(columns=lambda c:c.removeprefix('prefix_'))
        result['prefix']={'tokens':cfg['prefix_tokens'],'n':int(eligible.sum()),'test_n':len(pt),
                          'log_decomposition':boot_log_decomposition(pframe,y[eligible],cfg),
                          'full_same_cohort_log_decomposition':boot_log_decomposition(samples.loc[eligible],y[eligible],cfg),
                          'negative_norm_auc':score_result(y,-samples.prefix_post_mean_norm.to_numpy(),pt,cfg,bins)}
        # How much does the coordinate placement of learned gamma matter?
        rng=np.random.default_rng(cfg['seed']);permutations=[]
        prevsum=np.linalg.norm(means[:,:-1].astype(float),axis=2).sum(1);layers=means.shape[1]
        for k in range(cfg['gamma_permutations']):
            g=gamma[rng.permutation(len(gamma))];norm=np.linalg.norm(u*g,axis=1)
            ndr=(prevsum/norm+1)/layers
            permutations.append({'permutation':k,'ndr_auc':float(roc_auc_score(y[test],ndr[test])),
                                 'negative_norm_auc':float(roc_auc_score(y[test],-norm[test]))})
        result['gamma_permutations']=permutations
        # Fixed low-readout direction, and coordinate contributions to mean norm^2.
        result['gamma_geometry']={'min':float(gamma.min()),'max':float(gamma.max()),'min_abs':float(abs(gamma).min()),
             'rms':float(np.sqrt(np.mean(gamma**2))),'gain_along_v':float(np.linalg.norm(gamma*v)),
             'gamma_sha256':file_digest(Path(fc['output_root'])/name/'readout_geometry.npz')}
        bands=[]
        # Checkpoint-defined bins, no label-selected coordinates.
        for k,indices in enumerate(np.array_split(np.argsort(abs(gamma)),10)):
            for label in [0,1]:
                mu2=u[y==label]**2
                mean_fraction=np.sum(mu2[:,indices],axis=1)/np.sum(mu2,axis=1)
                token_fraction=u2[y==label][:,indices].sum(1)/u2[y==label].sum(1)
                bands.append({'gamma_abs_decile':k+1,'correct':label,'coordinates':len(indices),
                    'gamma_abs_min':float(abs(gamma[indices]).min()),'gamma_abs_max':float(abs(gamma[indices]).max()),
                    'mean_direction_energy_fraction':float(mean_fraction.mean()),
                    'token_direction_energy_fraction':float(token_fraction.mean())})
        result['gamma_energy_bands']=bands
        signed_contribution=(ideal[y==1]**2).mean(0)-(ideal[y==0]**2).mean(0)
        coord=pd.DataFrame({'coordinate':np.arange(len(gamma)),'gamma':gamma,'v':v,'difference_mean_post_squared_component':signed_contribution,
           'correct_mean_u_squared_component':np.mean(u[y==1]**2,axis=0),'incorrect_mean_u_squared_component':np.mean(u[y==0]**2,axis=0),
           'correct_mean_token_u2':u2[y==1].mean(0),'incorrect_mean_token_u2':u2[y==0].mean(0)})
        out=root/name;out.mkdir(exist_ok=True);coord.to_parquet(out/'coordinate_decomposition.parquet',index=False)
        samples.to_parquet(out/'samples.parquet',index=False)
        result['validation']={'max_ideal_vs_saved_relative_error':float(samples.ideal_vs_saved_post_relative_error.max()),
                             'max_pre_mean_absolute_error':float(samples.raw_mean_max_error.max())}
        save_json(out/'analysis.json',result);all_results[name]=result
        print(json.dumps({'dataset':name,'decomposition':result['log_decomposition_all'],'validation':result['validation']}),flush=True)
        del means,pre,post,ideal,u,u2
    save_json(root/'analysis.json',all_results)
    save_json(root/'provenance.json',{'config':cfg,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'code_sha256':file_digest(__file__),'runtime':runtime_versions(),'cache_roots':cfg['cache_root'],
              'interpretation':'Fixed-response arithmetic counterfactuals; not decoder interventions or causal correctness effects.'})
    render(cfg)


def render(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output_root']);results=json.loads((root/'analysis.json').read_text())
    body='<h1>RMSNorm：单 token 幅度与跨 token 抵消</h1><p>基于已完成的全量响应；无需重新生成。gamma 是 checkpoint 固定权重。保留原始正误标签，不重判读反事实表示。</p>'
    body+='<p><b>身份：</b>u=x/RMS(x)，y=gamma⊙u。精确分解 ||mean(y)|| = mean(||y||) × C，其中 C=||mean(y)||/mean(||y||)，衡量跨 token 的方向一致性（越小抵消越强）。另一个分解为 ||mean(y)|| = ||mean(u)|| × G，其中 G 是 gamma 对均值方向的增益。</p>'
    body+='<p>下图比较正确减错误的 log 均值：负值意味着正确更小。总差等于两项之和，允许两项互相抵消；这不是因果贡献百分比。FP64 理想 RMS 由原始 pre token 重算并与真实 BF16 post mean 核对。实际 post 指标独立保留。</p>'
    body+='<p>AUROC 使用原冻结 60% 分区，已见验证集上的探索性后续。bootstrap 以问题为单位，固定分数方向。prefix16 只取至少16 token 的回答，控制 token 数量；仍不控制题目难度和内容。gamma 打乱仅为坐标权重敏感性，不能当作回答正确率实验。</p>'
    summary=[]
    for name,r in results.items():
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
        for ax,fields,title in [(axes[0],['post_mean_norm','mean_post_token_norm','post_coherence'],'Amplitude + token coherence'),
                                (axes[1],['post_mean_norm','u_mean_norm','mean_gain'],'RMS-only mean + gamma gain')]:
            for j,(scope,offset,color) in enumerate([('log_decomposition_all',-.1,'#297e77'),('prefix',.1,'#b86632')]):
                d=r['prefix']['full_same_cohort_log_decomposition'] if scope!='prefix' else r['prefix']['log_decomposition']
                a=np.array([d[k]['log_correct_minus_incorrect'] for k in fields]);ci=np.array([d[k]['ci'] for k in fields])
                ax.errorbar(np.arange(3)+offset,a,yerr=[a-ci[:,0],ci[:,1]-a],fmt='o',capsize=4,color=color,label='Full response' if j==0 else 'First 16 tokens')
            ax.set_xticks(np.arange(3),['Total']+(['Token amplitude','Coherence'] if ax==axes[0] else ['RMS-only mean','Gamma gain']))
            ax.axhline(0,color='gray',ls='--');ax.set_title(title);ax.set_ylabel('Mean log difference: correct - incorrect');ax.legend()
        fig.suptitle(NAMES[name]+f" | same {r['prefix']['n']:,} questions; 95% bootstrap intervals")
        for ext in ['png','svg']:fig.savefig(root/name/('decomposition.'+ext),dpi=160)
        plt.close(fig)
        bands=pd.DataFrame(r['gamma_energy_bands'])
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained',sharey=True)
        for ax,field,title in zip(axes,['token_direction_energy_fraction','mean_direction_energy_fraction'],
                                 ['Individual token directions','Response mean direction']):
            for label,color in [(1,'#287348'),(0,'#bd553d')]:
                group=bands[bands.correct==label]
                ax.plot(group.gamma_abs_decile,100*group[field],'o-',color=color,label='Correct' if label else 'Incorrect')
            ax.set(title=title,xlabel='Coordinate decile: low to high |gamma|',ylabel='Mean energy fraction (%)')
            ax.set_xticks(range(1,11));ax.legend()
        axes[0].set_ylim(0,108*bands[['token_direction_energy_fraction','mean_direction_energy_fraction']].to_numpy().max())
        fig.suptitle(NAMES[name]+' | RMS-only states; checkpoint-defined coordinate bins')
        for ext in ['png','svg']:fig.savefig(root/name/('gamma_energy.'+ext),dpi=160)
        plt.close(fig)
        records=[]
        for condition in ['raw','gamma_only','rms_only','rms_gamma_ideal','rms_gamma_actual']:
            record={'Condition':condition}
            for metric in ['ndr','coe_r','coe_c','negative_final_norm']:record[metric]=r['conditions'][condition+'_'+metric]['auc']
            records.append(record);summary.append({'dataset':name,**record})
        body+=f'<h2>{html.escape(NAMES[name])}</h2><img src="{name}/decomposition.png"><img src="{name}/gamma_energy.png">'+pd.DataFrame(records).to_html(index=False,float_format=lambda x:f'{x:.3f}')
        body+=f'<p>真实末层均值的理想公式重建：最大相对误差 {r["validation"]["max_ideal_vs_saved_relative_error"]:.4%}。</p>'
        body+=f'<p><a href="{name}/analysis.json">全部区间与对照</a> · <a href="{name}/samples.parquet">逐题数据</a> · <a href="{name}/coordinate_decomposition.parquet">坐标分解（描述性）</a></p>'
    pd.DataFrame(summary).to_csv(root/'conditions.csv',index=False)
    (root/'protocol.md').write_text(Path('docs/rms-mechanism.zh-CN.md').read_text())
    body+='<p><a href="protocol.md">机制推导、候选解释与复现协议</a></p>'
    findings=Path('docs/rms-mechanism-findings.zh-CN.md')
    if findings.exists():
        (root/'findings.md').write_text(findings.read_text())
        body+='<p><a href="findings.md">本轮解释：哪些机制得到支持、哪些仍未知</a></p>'
    body+='<h2>因果边界</h2><p>本实验可以说明 RMS 和 gamma 的数学操作如何改变固定响应的指标，不能说明改变哪个量会让答案变正确。题目难度、词汇和回答结构可能同时影响这些几何量与正确性。将 v 从分母贡献中去掉，也是固定分子的一项算术对照，不是可直接等同于模型真实运行的干预。</p><p><a href="provenance.json">配置与代码版本</a> · <a href="conditions.csv">指标表</a></p>'
    (root/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>RMS mechanism</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#18333c}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ccd}a{color:#176e68}</style>'+body+'</html>')
    save_json(root/'report_provenance.json',{'code_sha256':file_digest(__file__),'analysis_sha256':file_digest(root/'analysis.json'),
              'figures':{str(p.relative_to(root)):file_digest(p) for p in sorted(root.glob('*/*.png'))}})

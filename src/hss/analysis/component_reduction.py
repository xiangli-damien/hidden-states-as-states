"""Exploratory reduction of NDR to raw common-component energy.

All fits, centering vectors, and coordinate rankings use discovery questions.
Previously inspected holdout: these are follow-up tests, not fresh confirmation.
"""
import html
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from hss.analysis.channel_data import ChannelData, load_config
from hss.analysis.component_data import load_moments, centered_energy
from hss.analysis.confidence_study import partitions, tuned_fit, control_features
from hss.analysis.confidence_data import layer_array
from hss.analysis.confidence_bootstrap import auc_samples
from hss.analysis.mean_geometry import group_summary, within_length_auc, NAMES
from hss.experiments.artifacts import save_json, file_digest, runtime_versions


def scalar_summary(a,y,test,cfg,bins,sign=1):
    take=test[np.isfinite(a[test])]
    draws=auc_samples(y[take],sign*a[take],cfg['seed'],cfg['bootstrap'])
    return {'sign':sign,'auc':float(roc_auc_score(y[take],sign*a[take])),
            'ci':np.quantile(draws,[.025,.975]).tolist(), 'test_n':len(take),
            'within_length_auc':within_length_auc(y[take],sign*a[take],bins[take]),
            'groups_test':group_summary(a[take],y[take],cfg)}


def poly_log(a,fraction=False):
    """Fixed transform; standardization and regularization fitted inside CV."""
    z=np.log(np.clip(a,1e-10,None))
    if fraction: z-=np.log(np.clip(1-a,1e-10,None))
    return np.column_stack([z,z*z,z*z*z])


def prediction_tests(values,rows,pre,train,test,cfg,out):
    y=rows.y.to_numpy(int)
    # q_all contains raw-only fractions. RMS-mean q is a separate diagnostic.
    q=poly_log(values['q_token_mean'],True)
    n=poly_log(values['ndr'])
    centered=poly_log(values['global_centered_energy'])
    qr=np.column_stack([poly_log(values[k],True) for k in ['q_token_mean','q_raw_mean','q_energy_ratio']])
    controls,names=control_features(rows,pre,train)
    controls=np.column_stack([controls,np.log1p(rows.n_tokens.to_numpy(float))])
    names+=['log_response_length']
    designs={'q_token':q,'q_raw_family':qr,'ndr':n,'q_token+ndr':np.c_[q,n],
             'q_raw_family+ndr':np.c_[qr,n],
             'controls':controls,'controls+q_raw_family':np.c_[controls,qr],
             'controls+ndr':np.c_[controls,n],'controls+q_raw_family+ndr':np.c_[controls,qr,n],
             'centered':centered,'centered+ndr':np.c_[centered,n],
             'controls+centered':np.c_[controls,centered],
             'controls+centered+ndr':np.c_[controls,centered,n]}
    results={};draws={};pred=rows[['sample_id','y']].copy()
    for label,x in designs.items():
        fit=tuned_fit(x,y,train,cfg);score=fit['score'];pred[label]=score
        draws[label]=auc_samples(y[test],score[test],cfg['seed'],cfg['bootstrap'])
        results[label]={'auc':float(roc_auc_score(y[test],score[test])),
                        'ci':np.quantile(draws[label],[.025,.975]).tolist(),
                        'c':fit['c'],'cv':fit['cv'],'coef':fit['coef'].tolist(),
                        'intercept':fit['intercept'],'features':x.shape[1]}
    contrasts={}
    for base,full in [('q_token','q_token+ndr'),('ndr','q_token+ndr'),
                      ('q_raw_family','q_raw_family+ndr'),('ndr','q_raw_family+ndr'),
                      ('controls+q_raw_family','controls+q_raw_family+ndr'),
                      ('controls+ndr','controls+q_raw_family+ndr'),
                      ('centered','centered+ndr'),('ndr','centered+ndr'),
                      ('controls+centered','controls+centered+ndr'),
                      ('controls+ndr','controls+centered+ndr')]:
        ci=np.quantile(draws[full]-draws[base],[.025,.975]).tolist()
        contrasts[full+' minus '+base]={'delta':results[full]['auc']-results[base]['auc'],
                                      'ci':ci,'within_equivalence_band':bool(ci[0]>-cfg['equivalence_auc'] and ci[1]<cfg['equivalence_auc'])}
    pred.to_parquet(out/'predictions.parquet',index=False)
    return {'models':results,'contrasts':contrasts,'control_names':names,
            'function_class':'Cubic log/logit features with discovery-only CV L2 logistic fits; does not test all possible information.',
            'ci_scope':'Paired question bootstrap, frozen fits; does not include model-training uncertainty.'}


def layer_statistics(cfg,fc,data,name,k,train,test,out):
    y=data.rows.y.to_numpy(int);records=[];updates={}
    # Every displayed final layer is pre-RMS, including the cached t16.
    for view in ['prompt_last','mean','t16']:
        states=data.array(view) if view!='t16' else None
        layers=data.info['model']['n_layers']
        for layer in range(layers):
            if view=='t16': x=layer_array(fc,data,name,layer).astype(float)
            elif layer==layers-1: x=data.array('pre_'+view).astype(float)
            else: x=states[:,layer].astype(float)
            valid=np.isfinite(x).all(1);take=test[valid[test]]
            energy=np.sum(x*x,axis=1);q=x[:,k]**2/np.maximum(energy,1e-30)
            if layer==layers-2:
                previous_coordinate=x[:,k].copy();previous_q=q.copy()
            if layer==layers-1:
                updates[view]={metric:group_summary(a[take],y[take],cfg) for metric,a in {
                    'signed_coordinate_update':x[:,k]-previous_coordinate,
                    'absolute_coordinate_update':np.abs(x[:,k])-np.abs(previous_coordinate),
                    'q_update':q-previous_q}.items()}
            for label in [0,1]:
                ii=take[y[take]==label]
                for metric,a in [('absolute',np.abs(x[:,k])),('signed',x[:,k]),('q',q)]:
                    records.append({'view':view,'layer':layer,'coordinate':k,'correct':label,'metric':metric,
                                    'mean':float(a[ii].mean()),'se':float(a[ii].std(ddof=1)/np.sqrt(len(ii))),'n':len(ii)})
        del states
    frame=pd.DataFrame(records);frame.to_parquet(out/'layers.parquet',index=False)
    return updates


def analyse(cfg):
    cc=load_config(cfg['channel_config']);fc=load_config(cfg['confidence_config'])
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True);results={}
    for name in cfg['datasets']:
        data=ChannelData(cc,name);rows=data.rows;train,test,saved=partitions(cfg,name,rows)
        y=rows.y.to_numpy(int);out=root/name;out.mkdir(exist_ok=True)
        m=load_moments(cfg,data,name);pre=m['mean'];dim=pre.shape[1]
        np.testing.assert_allclose(pre,data.array('pre_mean'),rtol=1e-4,atol=1e-5)
        # 2570 follows the user's hypothesis; Llama counterpart selected WITHOUT labels.
        discovered=int(np.argmax(m['second'][train].mean(0)))
        k=2570 if name=='qwen2_math' else discovered
        center=pre[train].mean(0);np.save(out/'discovery_common_mean.npy',center)
        token_center=np.average(pre[train],axis=0,weights=rows.n_tokens.to_numpy()[train])
        np.save(out/'discovery_token_weighted_mean.npy',token_center)
        sq=np.sum(pre*pre,axis=1);e=m['second'].sum(1)
        v={'q_raw_mean':pre[:,k]**2/sq,'q_token_mean':m['q'][:,k],
           'q_energy_ratio':m['second'][:,k]/e,
           'abs_mean_coordinate':np.abs(pre[:,k]),'mean_abs_coordinate':m['absolute'][:,k],
           'rms_coordinate':np.sqrt(m['second'][:,k]),
           'rest_mean_norm':np.sqrt(np.maximum(sq-pre[:,k]**2,0)),
           'rest_token_energy':(e-m['second'][:,k])/(dim-1),
           'raw_token_energy':e/dim,
           'global_centered_energy':centered_energy(pre,m['second'],center),
           'token_weighted_centered_energy':centered_energy(pre,m['second'],token_center),
           'within_response_energy':(e-sq)/dim,
           'mean_centered_energy':np.mean((pre-center)**2,axis=1)}
        # Prefix excludes short responses; same train-only global mean is frozen.
        pm=m['prefix_mean'];ps=m['prefix_second'];pe=ps.sum(1)
        v.update({'prefix_q_token_mean':m['prefix_q'][:,k],
                  'prefix_abs_coordinate':m['prefix_absolute'][:,k],
                  'prefix_rest_energy':(pe-ps[:,k])/(dim-1),
                  'prefix_centered_energy':centered_energy(pm,ps,center)})
        np.testing.assert_allclose(v['global_centered_energy'],v['within_response_energy']+v['mean_centered_energy'],rtol=1e-10,atol=1e-10)
        if np.any(v['within_response_energy'] < -1e-8):raise ValueError('Negative variance')
        old=pd.read_parquet(Path(cfg['rms_root'])/name/'samples.parquet')
        if not np.array_equal(old.sample_id,rows.sample_id):raise ValueError('RMS sample mismatch')
        v['ndr']=old.rms_gamma_actual_ndr.to_numpy()
        v['post_mean_norm']=np.linalg.norm(data.array('mean',-1).astype(float),axis=1)
        cache=Path(cfg['rms_cache'])/name
        u=np.concatenate([np.load(cache/s/'means.npz')['u'] for s in data.info['shards']])[data._indices]
        un2=np.sum(u*u,axis=1);v['q_rms_mean']=u[:,k]**2/un2
        gamma=np.load(Path(fc['output_root'])/name/'readout_geometry.npz')['gamma'].astype(float)
        restgain=(np.sum((u*gamma)**2,axis=1)-u[:,k]**2*gamma[k]**2)/(un2-u[:,k]**2)
        reconstruction=np.sqrt(un2*(gamma[k]**2*v['q_rms_mean']+restgain*(1-v['q_rms_mean'])))
        np.testing.assert_allclose(reconstruction,old.post_mean_norm,rtol=1e-10,atol=1e-9)
        lengths=rows.n_tokens.to_numpy();edges=np.unique(np.quantile(lengths[train],np.linspace(0,1,11)))[1:-1];bins=np.searchsorted(edges,lengths,side='right')
        result={'dataset':name,'n':len(rows),'test_n':len(test),'coordinate':k,'discovery_top_raw_rms_coordinate':discovered,
                'scalar_results':{},'spearman_with_ndr':{},'length_edges':edges.tolist(),
                'adaptation':'Previously inspected frozen 40/60 split. Exploratory; intervals are unadjusted for multiplicity.'}
        for key,a in v.items():
            sign=1 if key.startswith('q_') or key.startswith('prefix_q') or key=='ndr' or 'coordinate' in key else -1
            result['scalar_results'][key]=scalar_summary(a,y,test,cfg,bins,sign)
            valid=test[np.isfinite(a[test])]
            result['spearman_with_ndr'][key]=float(spearmanr(a[valid],v['ndr'][valid]).statistic)
        ndr_draws=auc_samples(y[test],v['ndr'][test],cfg['seed'],cfg['bootstrap'])
        result['paired_scalar_minus_ndr']={}
        for key in ['q_token_mean','q_raw_mean','q_energy_ratio','q_rms_mean','global_centered_energy','within_response_energy']:
            sign=result['scalar_results'][key]['sign']
            delta=auc_samples(y[test],sign*v[key][test],cfg['seed'],cfg['bootstrap'])-ndr_draws
            result['paired_scalar_minus_ndr'][key]={'delta':result['scalar_results'][key]['auc']-result['scalar_results']['ndr']['auc'],
                                                   'ci':np.quantile(delta,[.025,.975]).tolist()}
        sample=rows.assign(partition=saved.partition.to_numpy(),**v)
        sample.to_parquet(out/'samples.parquet',index=False)
        ranking=pd.DataFrame({'coordinate':np.arange(dim),'gamma':gamma,
                              'discovery_mean_abs':m['absolute'][train].mean(0),
                              'discovery_rms':np.sqrt(m['second'][train].mean(0)),
                              'discovery_q':m['q'][train].mean(0),
                              'discovery_top1_fraction':m['top1'][train].mean(0),
                              'test_mean_abs':m['absolute'][test].mean(0),
                              'test_q':m['q'][test].mean(0),
                              'test_top1_fraction':m['top1'][test].mean(0)})
        ranking['discovery_token_weighted_rms']=np.sqrt(np.average(m['second'][train],axis=0,weights=lengths[train]))
        ranking['discovery_token_weighted_abs']=np.average(m['absolute'][train],axis=0,weights=lengths[train])
        ranking['test_token_weighted_top1_fraction']=np.average(m['top1'][test],axis=0,weights=lengths[test])
        ranking=ranking.sort_values('discovery_rms',ascending=False)
        ranking['discovery_rms_rank']=np.arange(1,dim+1)
        ranking.to_parquet(out/'coordinate_ranking.parquet',index=False)
        result['ranking_top10']=ranking.head(10).to_dict('records')
        result['target_ranking']=ranking.set_index('coordinate').loc[k].to_dict()
        result['target_over_median_rms']=float(np.sqrt(m['second'][train].mean(0))[k]/np.median(np.sqrt(m['second'][train].mean(0))))
        result['prediction_tests']=prediction_tests(v,rows,pre,train,test,cfg,out)
        # Fixed-position diagnostics; t1 is after consuming generated token 1.
        token=ChannelData(cc,name,'tokens');positions=[]
        for p,pos in enumerate(['prompt_last',1,2,4,8,16,32,64,'last']):
            x=(data.array('pre_prompt_last') if p==0 else token.array('pre',p-1)).astype(float)
            valid=np.isfinite(x).all(1);take=test[valid[test]]
            energy=np.sum(x*x,axis=1)
            fields={'q':x[:,k]**2/energy,'abs_coordinate':np.abs(x[:,k]),
                    'rest_energy':(energy-x[:,k]**2)/(dim-1),
                    'centered_energy':np.mean((x-center)**2,axis=1)}
            for metric,a in fields.items():
                sign=-1 if metric.endswith('energy') else 1
                for label in [0,1]:
                    ii=take[y[take]==label]
                    positions.append({'position':str(pos),'metric':metric,'correct':label,'n':len(ii),
                                      'mean':float(a[ii].mean()),'se':float(a[ii].std(ddof=1)/np.sqrt(len(ii))),
                                      'signed_auc':float(roc_auc_score(y[take],sign*a[take]))})
        pd.DataFrame(positions).to_parquet(out/'positions.parquet',index=False)
        print(json.dumps({'dataset':name,'stage':'scalar_tests_complete','coordinate':k,
                          'q_token_auc':result['scalar_results']['q_token_mean']['auc'],
                          'centered_auc':result['scalar_results']['global_centered_energy']['auc'],
                          'contrasts':result['prediction_tests']['contrasts']}),flush=True)
        result['terminal_block_updates']=layer_statistics(cfg,fc,data,name,k,train,test,out)
        save_json(out/'analysis.json',result);results[name]=result
        del m,u,pre,token
    save_json(root/'analysis.json',results)
    save_json(root/'provenance.json',{'config':cfg,'versions':runtime_versions(),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'code_sha256':file_digest(__file__),'config_sha256':file_digest('configs/component_reduction.toml'),
              'data_sources':{n:ChannelData(cc,n).info for n in cfg['datasets']}})
    render(cfg)


def render(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output_root']);results=json.loads((root/'analysis.json').read_text())
    body='<h1>公共分量能否解释 NDR？</h1><p>全量回答 · raw pre-RMS · 冻结 40/60 分区 · CPU 分析</p>'
    body+='<p>本轮是已见验证集上的探索性检验。正确性未重新评估，所有反事实均为固定回答的指标分析，不是改变生成行为的因果实验。正误组均值与 AUROC 均报告在原 60% 分区。误差区间按问题 bootstrap；曲线阴影为逐点 ±1.96 SE，未校正多重比较。</p>'
    body+='<h2>三个不同的 q</h2><p>q_raw_mean = mean(x)[k]² / ||mean(x)||²；q_token_mean = mean(x[k]² / ||x||²)；q_energy_ratio = mean(x[k]²) / mean(||x||²)。三者都不经过 RMSNorm 或 γ。此前 65.3%／59.7% 是第四种：q_rms_mean，先逐 token 做 RMS-only 再平均。</p>'
    body+='<p>NDR = (各层回答均值 norm 的平均值) / (末层回答均值 norm)，并非末层 norm 本身。较小的末层分母会提高 NDR，但其他层也参与计算。</p>'
    body+='<h2>中心化的定义</h2><p>μ 是 discovery 问题的 raw 回答均值之平均，每道题等权。统一指标为 mean_t ||x_t−μ||² / D。μ 固定后用于验证集及各位置，不使用正确性标签。另报告回答内方差与均值到 μ 的距离；“中心化剩余”不自动等于语义内容。</p>'
    summary=[]
    colors={1:'#267c65',0:'#b85043'}
    for name,r in results.items():
        out=root/name;s=pd.read_parquet(out/'samples.parquet');test=s.partition.eq('confirmation');ss=s.loc[test]
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        keys=['q_raw_mean','q_token_mean','q_energy_ratio','q_rms_mean','ndr','global_centered_energy','within_response_energy']
        for j,k in enumerate(keys):
            v=r['scalar_results'][k];ci=v['ci'];axes[0].errorbar(v['auc'],j,xerr=[[v['auc']-ci[0]],[ci[1]-v['auc']]],fmt='o',capsize=3,color='#226d77')
        axes[0].set_yticks(range(len(keys)),['q(raw mean)','mean q(token)','ratio of energies','q(RMS-only mean)','NDR','− global-centered E','− within-response E'])
        axes[0].axvline(.5,color='gray',ls='--');axes[0].set(xlabel='Correctness AUROC (95% bootstrap CI)',title='Fixed score directions');axes[0].invert_yaxis()
        for label in [0,1]:
            a=ss[ss.y==label];axes[1].scatter(a.q_token_mean,a.ndr,s=6,alpha=.18,color=colors[label],label='Correct' if label else 'Incorrect',rasterized=True)
        axes[1].set(xlabel='Mean raw token energy fraction',ylabel='NDR',title=f"Spearman ρ = {r['spearman_with_ndr']['q_token_mean']:.3f}");axes[1].legend()
        fig.suptitle(NAMES[name]+f" | coordinate {r['coordinate']}")
        for ext in ['png','svg']:fig.savefig(out/('reduction.'+ext),dpi=160)
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,key,title in zip(axes,['mean_abs_coordinate','rest_token_energy','global_centered_energy'],['Raw coordinate amplitude','Raw remaining token energy / (D−1)','Global-centered token energy / D']):
            for label in [0,1]:
                values=ss.loc[ss.y==label,key].to_numpy();ordered=np.sort(values)
                ax.plot(ordered,np.arange(1,len(values)+1)/len(values),color=colors[label],label='Correct' if label else 'Incorrect')
            ax.set(xlabel=title,ylabel='Empirical CDF');ax.legend();ax.set_xscale('log')
        fig.suptitle(NAMES[name]+' | Raw energies; no gamma or RMSNorm')
        for ext in ['png','svg']:fig.savefig(out/('raw_energy.'+ext),dpi=160)
        plt.close(fig)
        pos=pd.read_parquet(out/'positions.parquet');fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,metric,title in zip(axes,['centered_energy','q'],['Centered token energy / D','Coordinate energy fraction']):
            for label in [0,1]:
                a=pos[(pos.metric==metric)&(pos.correct==label)];xx=np.arange(len(a));yy=a['mean'].to_numpy();se=1.96*a.se.to_numpy()
                ax.plot(xx,yy,'o-',color=colors[label],label='Correct' if label else 'Incorrect');ax.fill_between(xx,yy-se,yy+se,color=colors[label],alpha=.13)
                ax.set_xticks(xx,a.position)
            ax.set(xlabel='Consumed generation position (prompt_last predicts token 1)',ylabel=title);ax.legend()
        fig.suptitle(NAMES[name]+' | Fixed discovery common mean; available-question cohorts')
        for ext in ['png','svg']:fig.savefig(out/('positions.'+ext),dpi=160)
        plt.close(fig)
        layers=pd.read_parquet(out/'layers.parquet');fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for col,view in enumerate(['prompt_last','mean','t16']):
            for row,metric in enumerate(['absolute','q']):
                ax=axes[row,col]
                for label in [0,1]:
                    a=layers[(layers.view==view)&(layers.metric==metric)&(layers.correct==label)]
                    xx=a.layer.to_numpy();yy=a['mean'].to_numpy();se=1.96*a.se.to_numpy()
                    ax.plot(xx,yy,color=colors[label],label='Correct' if label else 'Incorrect');ax.fill_between(xx,yy-se,yy+se,color=colors[label],alpha=.12)
                ax.set(title=view,xlabel='Hidden-state slot (0 = embedding)',ylabel='Absolute coordinate' if metric=='absolute' else 'Coordinate energy fraction');ax.legend()
        fig.suptitle(NAMES[name]+f" | coordinate {r['coordinate']}; final slot is RAW pre-RMS")
        for ext in ['png','svg']:fig.savefig(out/('layers.'+ext),dpi=160)
        plt.close(fig)
        body+=f'<h2>{html.escape(NAMES[name])}</h2><p>坐标 {r["coordinate"]}；按 discovery 的 raw RMS 排名第 {int(r["target_ranking"]["discovery_rms_rank"])}。Qwen 固定检验 2570；Llama 用 discovery 中 RMS 最大的坐标，未按正误选择。</p>'
        for stem in ['reduction','raw_energy','positions','layers']:body+=f'<img src="{name}/{stem}.png">'
        updates=[{'View':view,'Quantity':metric,'Correct':v['correct_mean'],'Incorrect':v['incorrect_mean'],
                  'Difference':v['difference'],'95% CI':str(np.round(v['difference_ci'],4))}
                 for view,metrics in r['terminal_block_updates'].items() for metric,v in metrics.items()]
        body+='<h3>最终 block 的逐题增量（RMSNorm 之前）</h3><p>先对每道题求末层减上一层，再比较正误。该区间保留层间配对，属于观测定位，不证明功能上的因果性。</p>'+pd.DataFrame(updates).to_html(index=False,float_format=lambda x:f'{x:.4f}')
        table=[]
        for k,v in r['scalar_results'].items():
            g=v['groups_test'];table.append({'Metric':k,'Sign':v['sign'],'AUROC':v['auc'],'CI':str(np.round(v['ci'],4)),
                                            'Correct':g['correct_mean'],'Incorrect':g['incorrect_mean'],'Within length AUC':v['within_length_auc']})
            summary.append({'dataset':name,**table[-1]})
        body+='<h3>标量结果</h3>'+pd.DataFrame(table).to_html(index=False,float_format=lambda x:f'{x:.4f}')
        contrasts=[{'Comparison':k,'Delta AUROC':v['delta'],'95% CI':str(np.round(v['ci'],4)),'Within ±.01':v['within_equivalence_band']} for k,v in r['prediction_tests']['contrasts'].items()]
        body+='<h3>是否提供补充信息</h3><p>在 discovery 内交叉验证 C；固定三次 log/logit 特征。controls 包括题型、难度、prompt 长度、raw mean RMS 和回答长度。长度是生成后的诊断控制，不是 prompt-only 预测。相近 AUROC 不代表信息相同；区间跨零也不代表等价。</p>'+pd.DataFrame(contrasts).to_html(index=False,float_format=lambda x:f'{x:.4f}')
        body+='<h3>激活坐标排名（discovery，标签无关）</h3>'+pd.DataFrame(r['ranking_top10'])[['coordinate','discovery_rms','discovery_q','discovery_top1_fraction','gamma']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
        body+=f'<p><a href="{name}/analysis.json">完整结果与区间</a> · <a href="{name}/samples.parquet">逐题数据</a> · <a href="{name}/predictions.parquet">冻结模型分数</a> · <a href="{name}/coordinate_ranking.parquet">全部坐标</a></p>'
    body+='<h2>2570 与文献</h2><p><a href="https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf">See What You Are Told（ICLR 2025），附录 A.1</a>报告 Qwen2-VL-7B 的 massive-activation 维度为 458、2570。这是相关多模态模型，不是本研究的精确 checkpoint。当前仅有 prompt 最后位置及生成位置，没有完整 prompt attention；不能据此把 2570 定性为 attention sink。</p>'
    body+='<p>逐层残差变化只能定位在哪个 block 前后出现，不能单独分离 attention / MLP 的贡献，也不能证明“确定时主动写入”。同样，中心化能量差不能单独证明犹豫、重复或语义内容变化。全部中心、排名、拟合参数来自 discovery；当前验证分区已用于先前研究。</p><p><a href="protocol.md">定义与复现</a> · <a href="findings.md">结果解读</a> · <a href="provenance.json">来源版本</a> · <a href="summary.csv">汇总表</a></p>'
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    (root/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Common component vs NDR</title><style>body{max-width:1300px;margin:40px auto;padding:0 25px;font:16px/1.7 system-ui;color:#19363c}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ccd}a{color:#176e68}</style>'+body+'</html>')
    protocol=Path('docs/component-reduction.zh-CN.md')
    if protocol.exists():(root/'protocol.md').write_text(protocol.read_text())
    findings=Path('docs/component-reduction-findings.zh-CN.md')
    if findings.exists():(root/'findings.md').write_text(findings.read_text())
    save_json(root/'report_provenance.json',{'analysis_sha256':file_digest(root/'analysis.json'),'code_sha256':file_digest(__file__),
              'figures':{str(p.relative_to(root)):file_digest(p) for p in root.glob('*/*.png')}})

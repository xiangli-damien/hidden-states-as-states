"""Recompute every full-vocabulary KL and render complete token trajectories."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from revision_common import config, sha, write_json
from promptlast_replacement_common import apply
from run_promptlast_tokenwise import verify


def audit(cfg, smoke=False):
    import torch
    root=Path(cfg['output']); plan=verify(root); dest=root/('smoke' if smoke else 'functional')
    completed=json.loads((dest/'_SUCCESS.json').read_text()); assert completed['plan_sha256']==sha(root/'plan.json')
    cases={c['sample_id']:c for c in json.loads((root/'cases.json').read_text())}
    with np.load(root/'decoder.npz') as z:d={k:z[k].copy() for k in z.files}
    total=0; errors=[]; order=plan['conditions']; allrows=[]
    for receipt in sorted(dest.glob('*/_SUCCESS.json')):
        meta=json.loads(receipt.read_text()); sid=meta['sample_id']; c=cases[sid]; positions=[]
        for f,h in meta['files'].items():assert sha(receipt.parent/f)==h
        assert meta['plan_sha256']==sha(root/'plan.json')
        for path in sorted(receipt.parent.glob('chunk_*.json')):
            saved=json.loads(path.read_text()); assert saved['plan_sha256']==sha(root/'plan.json') and sha(path.with_suffix('.npz'))==saved['arrays_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                assert z['logp'].shape==(len(saved['rows']),len(order),z['logp'].shape[-1])
                assert len(z['position'])==len(saved['rows'])
                for i,r in enumerate(saved['rows']):
                    pos=int(z['position'][i]); positions.append(pos); assert pos==r['position']
                    assert int(z['target_id'][i])==r['target_id']==c['response_ids'][pos]
                    assert int(z['input_id'][i])==r['input_id']==(c['prompt_ids'][-1] if pos==0 else c['response_ids'][pos-1])
                    x=z['x'][i].astype(float); actual=z['actual'][i].astype(float); lp=z['logp'][i].astype(float)
                    assert np.isfinite(lp).all() and np.isfinite(actual).all()
                    np.testing.assert_allclose(np.exp(lp).sum(-1),1,atol=2e-6)
                    np.testing.assert_array_equal(actual[0],x)
                    region=int(np.argmin(np.square(d['centers'].astype(float)-x).sum(-1)))
                    assert region==r['region']
                    dist=float(np.linalg.norm(x-d['centers'][region])); np.testing.assert_allclose(dist,r['distance'],atol=1e-10)
                    for j,name in enumerate(order[1:],1):
                        ideal,_=apply(x[None],d,name.split('_',1)[1])
                        rounded=torch.tensor(ideal).to(torch.bfloat16).float().numpy()[0]
                        np.testing.assert_array_equal(rounded,actual[j])
                    # Independent arithmetic from the saved full vocabulary distributions.
                    kl=np.array([np.dot(np.exp(lp[0]),lp[0]-value) for value in lp])
                    err=float(np.max(np.abs(kl-r['kl']))); errors.append(err); assert err<1e-9 and kl.min()>-1e-5
                    nll=-lp[:,r['target_id']]; np.testing.assert_array_equal(nll,r['reference_nll'])
                    np.testing.assert_allclose(np.square(actual-x).sum(-1)/np.square(x-d['train_mean']).sum(),r['nmse'],atol=1e-10)
                    np.testing.assert_array_equal(lp.argmax(-1),r['argmax'])
                    for j in range(1,4):
                        np.testing.assert_array_equal(actual[j],actual[j+3])
                        if pos==0:np.testing.assert_array_equal(lp[j],lp[j+3])
                    for j,name in enumerate(order):
                        allrows.append({'sample_id':sid,'dataset':c['dataset'],'position':pos,'condition':name,
                            'kl':float(kl[j]),'reference_nll':float(nll[j]),'delta_nll':float(nll[j]-nll[0]),
                            'nmse':r['nmse'][j],'argmax_agreement':int(lp[j].argmax()==lp[0].argmax()),
                            'region':region,'distance':dist,'x_norm':r['x_norm'],'input_id':r['input_id'],'target_id':r['target_id'],
                            'input_text':r['input_text'],'target_text':r['target_text']})
        assert positions==list(range(meta['predictions'])); total+=len(positions)
    assert total==completed['predictions'] and total==(8 if smoke else plan['predictions'])
    with (dest/'rows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    write_json(dest/'audit.json',{'complete':True,'predictions':total,'conditions':len(order),'records':len(allrows),
        'full_vocabulary_recomputed':True,'max_kl_arithmetic_error':max(errors), 'rows_sha256':sha(dest/'rows.csv'),
        'plan_sha256':sha(root/'plan.json')})


def report(cfg):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output']); plan=verify(root); audit=json.loads((root/'functional/audit.json').read_text())
    assert audit['complete'] and sha(root/'functional/rows.csv')==audit['rows_sha256']
    f=pd.read_csv(root/'functional/rows.csv'); cases=json.loads((root/'cases.json').read_text())
    out=root/'report';out.mkdir(exist_ok=True)
    style='<style>body{max-width:1120px;margin:36px auto;padding:0 20px;font:17px/1.65 system-ui;color:#172332}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #ccd3dc;text-align:left}th{background:#edf3f8}img{width:100%}code{white-space:pre-wrap}.note{background:#fff6dc;padding:14px}a{color:#176295}</style>'
    def page(title,body):return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+style+'<body><h1>'+title+'</h1>'+body+'</body></html>'
    summaries=[];links=[]
    for c in cases:
        sid=c['sample_id'];g=f[f.sample_id.eq(sid)];identity=g[g.condition.eq('identity')].sort_values('position')
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
        for ax,mode in zip(axes[:2],['independent','cumulative']):
            for method,color in zip(cfg['methods'],['#a74531','#5675c2','#087f74']):
                q=g[g.condition.eq(mode+'_'+method)].sort_values('position')
                ax.plot(q.position,q.kl,label=method,color=color,alpha=.85,lw=1)
            ax.set_ylabel('Next-token KL (nats)');ax.set_title(mode+' replacement');ax.legend(ncol=3);ax.grid(alpha=.2)
        axes[2].scatter(identity.position,identity.region,s=6,color='#6d4e98')
        axes[2].set_ylabel('Frozen region ID');axes[2].set_xlabel('Observed response tokens: 0 = prompt-last, predicts response token 1')
        axes[2].set_ylim(-1,32);axes[2].grid(alpha=.2);fig.suptitle(sid+' | Qwen2 block14 | same reference prefixes')
        fig.savefig(out/(sid+'.png'),dpi=150);fig.savefig(out/(sid+'.pdf'));plt.close(fig)
        for window,mask in [('prompt_last',g.position.eq(0)),('generation_1_16',g.position.between(1,16)),('generation_all',g.position.ge(1))]:
            for name in plan['conditions'][1:]:
                q=g[mask & g.condition.eq(name)]
                summaries.append({'sample_id':sid,'dataset':c['dataset'],'window':window,'condition':name,'positions':len(q),
                    **{col:float(q[col].mean()) for col in ['kl','delta_nll','nmse','argmax_agreement']}})
        wide=g.pivot(index='position',columns='condition',values='kl').reset_index()
        tokens=identity[['position','input_text','target_text','region','distance','x_norm']].merge(wide,on='position')
        tokens.to_csv(out/(sid+'_positions.csv'),index=False)
        body='<p><a href="index.html">返回总览</a> · <a href="'+sid+'.pdf">PDF</a> · <a href="'+sid+'_positions.csv">逐位置 CSV</a></p>'
        body+='<p>横坐标 p：已读入 p 个原回答 token。p=0 修改 prompt-last；p=1 修改原回答第 1 个 token 的状态，预测第 2 个 token。末点预测原记录的最后一个 token，不在 EOS 之后继续。</p><img src="'+sid+'.png">'
        body+='<details><summary>原始题目与原回答</summary><h3>题目</h3><pre style="white-space:pre-wrap">'+html.escape(c['prompt_text'])+'</pre><h3>原回答</h3><pre style="white-space:pre-wrap">'+html.escape(c['response_text'])+'</pre></details>'
        body+='<h2>每个位置的下一 token KL</h2><p>单 token 文本可能包含空格或分词碎片，不能当作独立词义。ID 只是区域编号，数字差不代表距离。</p><div style="overflow:auto">'+tokens.to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')+'</div>'
        (out/(sid+'.html')).write_text(page(sid,body))
        links.append('<li><a href="'+sid+'.html">'+sid+'</a>：'+str(len(identity))+' 个预测位置，生成状态进入 '+str(identity[identity.position.ge(1)].region.nunique())+' 个区域。</li>')
    summary=pd.DataFrame(summaries);summary.to_csv(out/'per_question_summary.csv',index=False)
    aggregate=summary.groupby(['dataset','window','condition'],sort=False)[['kl','delta_nll','nmse','argmax_agreement']].mean().reset_index()
    aggregate.to_csv(out/'summary.csv',index=False)
    # All questions have equal weight; no token-level bootstrap pseudo-replication.
    stats={'scope':cfg['scope'],'questions':6,'predictions':plan['predictions'],'condition_records':audit['records'],
           'aggregation':'Within-question position mean, then equal mean across three questions per dataset; descriptive, no inferential confidence intervals.',
           'table':aggregate.to_dict('records'),'per_question':summaries,'audit':audit}
    write_json(root/'summary.json',stats)
    body='<p class="note">6 个预先固定示例（MATH 3、GSM8K 3），共 '+str(plan['predictions'])+' 个预测位置。整段原回答逐位置检查；不是 96 题总体检验，也不是自由生成正确率实验。</p>'
    body+='<h2>这个实验在比较什么</h2><p>冻结在 MATH prompt-last 向量上学到的 K=32 地图和基。每个当前 token 的 block14 状态独立找最近中心，不锁定首个区域，不看后续 token。原模型与各替换条件始终读相同的原回答前缀。</p>'
    body+='<ul><li><b>independent：</b>只替换当前位置；历史 KV cache 全部来自未修改模型。</li><li><b>cumulative：</b>从 prompt-last 到当前位置持续替换，保留修改后的历史 cache。</li><li><b>centroid：</b>直接换成中心。<b>shared8 / local8：</b>同中心加共享／本区域的 8 个投影坐标。</li></ul>'
    body+='<p>KL = Σ P原始(token) log[P原始(token)/P替换(token)]，对整个词表计算，单位 nats；并非“高了几个准确率百分点”。低 KL 表示更接近原模型，原模型本身仍可能答错。固定前缀实验也不会证明实际自由生成的后续文字相同。</p>'
    body+='<h2>完整轨迹与逐 token 原文</h2><ul>'+''.join(links)+'</ul>'
    body+='<h2>按题等权的描述汇总</h2><p>prompt_last：预测第一个 token；generation_1_16：已读入 1–16 个回答 token；generation_all：所有生成状态（不含 prompt-last）。delta_nll 是同一个原回答下的参考 token NLL 差，负值不等于纠错。</p>'+aggregate.to_html(index=False,float_format=lambda x:f'{x:.5f}')
    body+='<p><a href="summary.csv">汇总 CSV</a> · <a href="per_question_summary.csv">每题汇总 CSV</a>。所有完整词表 log-probability、真实输入及替换向量、逐位置指标和 SHA 校验记录保存在 Dami 原始实验目录。'+str(audit['records'])+' 条记录已独立重算完整词表 KL。</p>'
    (out/'index.html').write_text(page('Prompt-last 地图能用于后续每个 token 吗？',body))
    write_json(root/'report_SUCCESS.json',{'summary_sha256':sha(root/'summary.json'),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();cfg=config(a.config)
    if a.report:report(cfg)
    else:audit(cfg,a.smoke)

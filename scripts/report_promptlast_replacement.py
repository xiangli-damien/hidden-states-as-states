"""Matched prompt-last reconstruction fidelity; no answer correctness claims."""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import config,sha,write_json,write_npz


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output']);dest=root/'report';dest.mkdir(exist_ok=True)
    assert json.loads((root/'functional/audit.json').read_text())['complete']
    cases=json.loads((root/'cases.json').read_text());fit=json.loads((root/'fit_audit.json').read_text())
    rows=[]
    for p in sorted((root/'functional/samples').glob('*.json')):
        r=json.loads(p.read_text());rows.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'condition':r['condition'],**r['metrics'],**r['meta']})
    frame=pd.DataFrame(rows);base=frame[frame.condition.eq('identity')].set_index('sample_id')
    frame['argmax_agreement']=frame.argmax_agreement.astype(int)
    for k in ['nll','first16_nll','later_nll']:frame['delta_'+k]=frame[k]-frame.sample_id.map(base[k])
    frame.to_csv(dest/'per_question.csv',index=False)
    ids={ds:[c['sample_id'] for c in cases if c['dataset']==ds] for ds in ['math','gsm8k']}
    rng=np.random.default_rng(cfg['bootstrap_seed']);boots={ds:rng.integers(len(v),size=(cfg['bootstrap'],len(v))) for ds,v in ids.items()}
    write_npz(dest/'bootstrap.npz',**boots)
    def values(ds,name,key):
        if name=='wrong8':return np.mean([values(ds,'wrong8_'+str(s),key) for s in cfg['seeds']],axis=0)
        return frame[(frame.dataset==ds)&frame.condition.eq(name)].set_index('sample_id').loc[ids[ds],key].to_numpy(float)
    names=['identity','global_mean','centroid','shared8','local8','wrong8'];conditions=[];paired=[];distribution={}
    metrics=['next_token_kl','delta_nll','delta_first16_nll','delta_later_nll','reconstruction_sse','argmax_agreement']
    for ds,sids in ids.items():
        for name in names:
            r={'dataset':ds,'condition':name,'n':len(sids)}
            for key in metrics:r[key]=float(values(ds,name,key).mean())
            r['nmse']=float(values(ds,name,'reconstruction_sse').sum()/values(ds,name,'train_centered_energy').sum());conditions.append(r)
        for method,control in [('local8','shared8'),('local8','centroid'),('local8','wrong8'),('centroid','global_mean')]:
            for key in metrics:
                diff=values(ds,method,key)-values(ds,control,key)
                paired.append({'dataset':ds,'method':method,'control':control,'metric':key,
                    'estimate':float(diff.mean()),'ci95':np.quantile(diff[boots[ds]].mean(1),[.025,.975]).tolist(),
                    'primary':ds=='gsm8k' and method=='local8' and control=='shared8' and key in ['next_token_kl','delta_nll']})
        b=base.loc[sids];counts=np.bincount(b.region.to_numpy(int),minlength=32)
        distribution[ds]={'occupied_regions':int((counts>0).sum()),'counts':counts.tolist(),'max_region_fraction':float(counts.max()/len(sids)),
            'distance_median':float(b.distance.median()),'source_q95_coverage':float(np.mean(b.distance<=fit['source_validation_distance_q95']))}
    summary={'config':cfg,'fit':fit,'conditions':conditions,'paired':paired,'distribution':distribution,
        'raw_audit_sha256':sha(root/'functional/audit.json'),'scope':cfg['scope'],'primary':cfg['primary'],
        'uncertainty':'Fixed maps/bases, paired question bootstrap, pointwise95%, no retraining uncertainty or multiple-comparison correction.'}
    write_json(dest/'summary.json',summary);pd.DataFrame(conditions).to_csv(dest/'condition_summary.csv',index=False);pd.DataFrame(paired).to_csv(dest/'paired_comparisons.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');plotnames=names[1:]
    for i,ds in enumerate(['math','gsm8k']):
        for j,key in enumerate(['next_token_kl','delta_nll','delta_later_nll']):
            vals=[float(values(ds,n,key).mean()) for n in plotnames];ax=axes[i,j]
            ax.bar(range(5),vals,color=['#a4acb8','#889cb1','#daaa64','#66a68a','#baa3bf']);ax.axhline(0,lw=.6,color='#333333')
            ax.set_xticks(range(5),['Global\nmean','GMM\ncenter','Shared8','Local8','Wrong8\navg'])
            ax.set_title(ds.upper()+': '+{'next_token_kl':'next-token KL','delta_nll':'whole reference delta NLL','delta_later_nll':'reference tokens 17+ delta NLL'}[key])
            ax.set_ylabel('nats; lower is better');ax.margins(y=.25)
            for k,v in enumerate(vals):ax.annotate(f'{v:.4f}',(k,v),xytext=(0,5 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('Frozen prompt-last GMM K32 | Qwen2-7B block14 | one prompt position replaced\nHistorical MATH32 / reused GSM64; original model continuation, not correctness',fontsize=13)
    fig.savefig(dest/'comparison.png',dpi=160);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    style='<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#213144}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #cbd3db}pre{white-space:pre-wrap;background:#f3f5f7;padding:15px}.scroll{overflow:auto}</style>'
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prompt-last reconstruction</title>',style,
        '<h1>Prompt-last 冻结地图：重构与实际替换</h1>',
        '<p>原 block14、raw3584维，MATH训练的GMM K32及局部方向冻结。仅替换完整prompt最后一个位置，然后继续原模型计算。其他prompt位置完整保留；不是整道题的压缩，更不是答案正确率实验。</p>',
        '<p>Reference NLL对应未干预模型的原回答，包括EOS，不是标准答案。正的ΔNLL表示原回答更难重放；负值也不代表正确率提高。单独首token KL可能受固定开场白影响，因此同时报告全回答和第17token以后的NLL。</p>',
        '<img src="comparison.png" alt="Prompt-last functional fidelity">','<h2>固定主比较：GSM64 local8−shared8</h2>',
        pd.DataFrame([r for r in paired if r['primary']]).to_html(index=False),
        '<h2>几何误差与功能保真</h2><div class="scroll">'+pd.DataFrame(conditions).to_html(index=False)+'</div>',
        '<p>NMSE：重构平方误差总和 / 相对MATH训练均值的平方误差总和；global mean参考为1，identity为0。Local8和shared8均保留同一个簇ID及8连续坐标，但局部字典需要更多总存储。</p>',
        '<h2>区域占用与源域覆盖</h2><pre>'+html.escape(json.dumps(distribution,ensure_ascii=False,indent=2))+'</pre>',
        '<p>源验证集q95距离仅作分布描述，不排除任何目标题。旧题探索比较，问题级配对区间，无多重比较校正。只改1个prompt位置，不能与旧实验改16个生成位置的KL直接比较来宣布地图更好。</p>',
        '<p><a href="summary.json">完整统计</a> · <a href="per_question.csv">逐题数值</a> · <a href="statistics_audit.json">统计核验</a> · <a href="comparison.pdf">图PDF</a></p><h2>逐题原文</h2><ul>']
    folder=dest/'cases';folder.mkdir(exist_ok=True)
    for c in cases:
        sid=c['sample_id'];page=['<!doctype html><meta charset="utf-8">',style,'<a href="../index.html">返回</a><h1>'+html.escape(sid)+'</h1>']
        for title,key in [('问题','prompt_text'),('标准终答案','ground_truth'),('原模型参考回答（不是新生成答案）','response_text')]:page+=['<h2>'+title+'</h2><pre>'+html.escape(c[key])+'</pre>']
        table=frame[frame.sample_id.eq(sid)][['condition','next_token_kl','delta_nll','delta_first16_nll','delta_later_nll','reconstruction_sse']]
        page.append(table.to_html(index=False));(folder/(sid+'.html')).write_text(''.join(page));parts.append('<li><a href="cases/'+sid+'.html">'+sid+'</a></li>')
    parts.append('</ul>');(dest/'index.html').write_text(''.join(parts))
    write_json(dest/'_SUCCESS.json',{'summary_sha256':sha(dest/'summary.json'),'questions':len(cases),'visual_review':'pending'})
    print(json.dumps({'primary':[r for r in paired if r['primary']],'distribution':distribution}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(config(a.config))

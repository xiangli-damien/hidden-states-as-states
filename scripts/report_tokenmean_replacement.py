"""Question-paired report with separate mean-shift and token-projection budgets."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from revision_common import config,sha,write_json,write_npz


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['output']);dest=root/'report';dest.mkdir(exist_ok=True)
    assert json.loads((root/'functional/audit.json').read_text())['complete']
    cases=json.loads((root/'cases.json').read_text())
    rows=[]
    for path in sorted((root/'functional/samples').glob('*.json')):
        r=json.loads(path.read_text())
        rows.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'condition':r['condition'],
                     **r['metrics'],**{k:v for k,v in r['meta'].items() if k!='token_regions'}})
    frame=pd.DataFrame(rows)
    base=frame[frame.condition.eq('identity')].set_index('sample_id')
    frame['delta_nll']=frame.nll-frame.sample_id.map(base.nll)
    frame.to_csv(dest/'per_question.csv',index=False)
    ds_ids={ds:[c['sample_id'] for c in cases if c['dataset']==ds] for ds in ['math','gsm8k']}
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    boots={ds:rng.integers(len(ids),size=(cfg['bootstrap'],len(ids))) for ds,ids in ds_ids.items()}
    write_npz(dest/'bootstrap.npz',**boots)
    def values(ds,name,metric):
        if name.endswith('_wrong8'):
            return np.mean([values(ds,name+'_'+str(s),metric) for s in cfg['seeds']],axis=0)
        return frame[(frame.dataset==ds)&(frame.condition==name)].set_index('sample_id').loc[ds_ids[ds],metric].to_numpy(float)
    names=['identity','tokenmap_centroid','tokenmap_shared8','tokenmap_local8']
    names += [op+'_'+m for op in ['mean_shift','token_project'] for m in ['centroid','shared8','local8','wrong8']]
    metrics=['next_token_kl','delta_nll','token_mse','mean_mse','centered_change_mse']
    means=[];paired=[]
    contrasts=[(op+'_'+m,op+'_'+c) for op in ['mean_shift','token_project']
               for m,c in [('local8','shared8'),('local8','wrong8'),('local8','centroid')]]
    contrasts += [('token_project_local8','tokenmap_local8'),('mean_shift_local8','token_project_local8'),
                  ('tokenmap_local8','tokenmap_shared8')]
    for ds in ds_ids:
        for name in names:
            for metric in metrics:
                v=values(ds,name,metric)
                means.append({'dataset':ds,'condition':name,'metric':metric,'estimate':float(v.mean()),'n':len(v)})
        for method,control in contrasts:
            for metric in metrics:
                v=values(ds,method,metric)-values(ds,control,metric)
                ci=np.quantile(v[boots[ds]].mean(1),[.025,.975]).tolist()
                paired.append({'dataset':ds,'method':method,'control':control,'metric':metric,
                    'estimate':float(v.mean()),'ci95':ci,'n':len(v),
                    'primary':ds=='gsm8k' and method=='mean_shift_local8' and control=='mean_shift_shared8'
                              and metric in ['next_token_kl','delta_nll']})
    pd.DataFrame(means).to_csv(dest/'condition_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_comparisons.csv',index=False)
    fit=json.loads((root/'fit_audit.json').read_text())
    distribution={}
    for ds,ids in ds_ids.items():
        b=base.loc[ids]
        counts=b.mean_region.value_counts()
        distribution[ds]={'prefix_mean_occupied_clusters':len(counts),'counts':{str(k):int(v) for k,v in counts.items()},
            'prefix_density_below_fullmean_train_q05':float((b.mean_log_density<fit['full_mean_train_density_quantiles'][1]).mean()),
            'prefix_log_density_median':float(b.mean_log_density.median())}
        if ds=='math':
            old={c['sample_id']:c['old_full_mean_region'] for c in cases if c['dataset']==ds}
            distribution[ds]['prefix_vs_full_mean_assignment_agreement']=float(np.mean([int(base.loc[s,'mean_region'])==old[s] for s in ids]))
    summary={'config':cfg,'fit':fit,'conditions':means,'paired':paired,'distribution':distribution,
             'raw_audit_sha256':sha(root/'functional/audit.json'),'scope':cfg['scope'],
             'primary':cfg['primary'],'uncertainty':'Fixed maps/bases, paired question bootstrap; pointwise95%; no training uncertainty or multiplicity correction.'}
    write_json(dest/'summary.json',summary)
    plt.rcParams.update({'font.size':10})
    labels=['Token\ncentroid','Token\nshared8','Token\nlocal8','Mean shift\ncentroid','Mean shift\nshared8','Mean shift\nlocal8','Mean shift\nwrong8','Token project\ncentroid','Token project\nshared8','Token project\nlocal8','Token project\nwrong8']
    figure_names=names[1:]
    fig,axes=plt.subplots(2,1,figsize=(14,9),layout='constrained')
    colors=['#8b9bad']*3+['#70a0be']*4+['#b9a1cd']*4
    for ax,ds in zip(axes,['math','gsm8k']):
        vals=[values(ds,n,'next_token_kl').mean() for n in figure_names]
        ax.bar(range(len(vals)),vals,color=colors)
        ax.set_xticks(range(len(vals)),labels,fontsize=8)
        ax.set_ylabel('Next-token KL (nats; lower is better)')
        ax.set_title(('Historical MATH32 (transductive mean map)' if ds=='math' else 'Reused GSM8K64 (frozen MATH maps)'))
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
        ax.margins(y=.2)
    fig.suptitle('Frozen whole-response mean GMM K33 vs token GMM K64\nMean shift retains all centered token residuals; budgets are different',fontsize=13)
    fig.savefig(dest/'comparison.png',dpi=170);plt.close(fig)
    style='<style>body{font:16px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#203040}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd;padding:7px}pre{white-space:pre-wrap;background:#f3f5f7;padding:15px}img{max-width:100%}.note{background:#fff3d0;padding:15px}</style>'
    main=['<!doctype html><meta charset="utf-8"><title>Frozen token-mean replacement</title>',style,
          '<h1>冻结 token-mean 地图：功能替换</h1>',
          '<p class="note">mean-shift 保留全部 token 间差异，不能称八维压缩。token-project 每 token 八维，但全窗口共用前缀均值选出的区域。K33/64 与分配规则不同；本轮为旧题上的探索对照，不是新的独立确认。NLL 针对原模型回答，不是标准答案；没有自由生成正确率结果。</p>',
          '<img src="comparison.png" alt="Mean-map and token-map output KL comparison">',
          '<h2>主比较及配对区间</h2>',pd.DataFrame([r for r in paired if r['metric'] in ['next_token_kl','delta_nll']]).to_html(index=False,escape=True),
          '<h2>前缀均值分布偏移</h2><pre>'+html.escape(json.dumps(distribution,ensure_ascii=False,indent=2))+'</pre>',
          '<p><a href="summary.json">完整 JSON</a> · <a href="per_question.csv">逐题指标</a> · <a href="paired_comparisons.csv">配对比较</a> · <a href="statistics_audit.json">统计审计</a></p><h2>逐题原文</h2><ul>']
    casefolder=dest/'cases';casefolder.mkdir(exist_ok=True)
    for c in cases:
        sid=c['sample_id'];path=casefolder/(sid+'.html')
        table=frame[frame.sample_id.eq(sid)][['condition','next_token_kl','delta_nll','token_mse','mean_mse','centered_change_mse']]
        content='<!doctype html><meta charset="utf-8">'+style+'<a href="../index.html">返回</a><h1>'+html.escape(sid)+'</h1>'
        for title,key in [('问题','prompt_text'),('标准终答案','ground_truth'),('未干预模型参考回答','response_text')]:
            content+='<h2>'+title+'</h2><pre>'+html.escape(c[key])+'</pre>'
        content+='<h2>功能指标（不是干预后新生成回答）</h2>'+table.to_html(index=False,escape=True)
        path.write_text(content)
        main.append('<li><a href="cases/'+html.escape(sid,quote=True)+'.html">'+html.escape(c['dataset']+' '+sid)+'</a></li>')
    main.append('</ul>');(dest/'index.html').write_text(''.join(main))
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_tokenmean_replacement.py')),
                    '--config',str(root/'report_config.json'),'--report'],check=True)
    write_json(dest/'_SUCCESS.json',{'summary_sha256':sha(dest/'summary.json'),'statistics_audit_sha256':sha(dest/'statistics_audit.json'),
        'questions':len(cases),'visual_review':'pending'})
    print(json.dumps({'primary':[r for r in paired if r['primary']],'distribution':distribution},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=config(a.config)
    write_json(Path(cfg['output'])/'report_config.json',cfg);run(cfg)

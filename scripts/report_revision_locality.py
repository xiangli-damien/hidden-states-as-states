"""Report all audited locality comparisons without selecting on pilot outcomes."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from revision_common import sha,write_json,freeze,provenance
from revision_locality_common import key
from report_revision_functional import interval,paired_difference

GROUP=['split','prefix_tokens','layer','role','width']
METRICS=['next_token_kl','delta_nll','delta_first16_nll','delta_first_token_nll',
         'next_token_argmax_agreement','actual_patch_energy','state_retained_fraction','mse_per_coordinate']


def collect(root):
    stage=root/'functional';audit=json.loads((stage/'audit.json').read_text())
    assert audit['complete'] and audit['conditions']==audit['expected']
    assert audit['plan_sha256']==sha(stage/'plan.json')
    manifest=json.loads((stage/'audit_inputs.json').read_text());raw=[]
    for path in sorted((stage/'samples').glob('*.json')):
        assert sha(path)==manifest[path.name]
        r=json.loads(path.read_text());t=r['task'];c=t['condition'];g=r['geometry']['actual']
        label=key({k:v for k,v in c.items() if k!='seed'})
        row={'sample_id':t['sample_id'],'split':r['split'],**{k:t[k] for k in GROUP if k!='split'},
            'method':label,'individual_method':key(c),'family':c['method'],'rank':c.get('rank',0),
            'alpha':c.get('alpha',0),'seed':c.get('seed',-1),'state_retained_fraction':g['state_retained_fraction'],
            'mse_per_coordinate':sum(g['token_delta_energy'])/(len(g['token_delta_energy'])*3584)}
        for k in ['next_token_kl','nll','first16_nll','first_token_nll','next_token_argmax_agreement','actual_patch_energy']:row[k]=r[k]
        raw.append(row)
    raw=pd.DataFrame(raw)
    keys=GROUP+['sample_id']
    baselines=raw.loc[raw.family.eq('identity'),keys+['nll','first16_nll','first_token_nll']]
    assert not baselines.duplicated(keys).any()
    raw=raw.merge(baselines,on=keys,suffixes=('','_identity'),validate='many_to_one')
    for m in ['nll','first16_nll','first_token_nll']:raw['delta_'+m]=raw[m]-raw[m+'_identity']
    # Random seeds are repeated conditions on the SAME question, never extra n.
    per=raw.groupby(keys+['method','family','rank','alpha'],as_index=False)[METRICS].mean()
    return raw,per,audit


def tables(per):
    summaries=[];pairs=[]
    for group,sub in per.groupby(GROUP,sort=True):
        base=dict(zip(GROUP,group))
        for method,rows in sub.groupby('method',sort=True):
            rows=rows.sort_values('sample_id');summary={**base,'method':method,'n':len(rows),
                'family':rows.family.iloc[0],'rank':int(rows['rank'].iloc[0]),'alpha':float(rows.alpha.iloc[0])}
            for metric in METRICS:
                for k,v in interval(rows[metric]).items():
                    if k!='n':summary[f'{metric}_{k}']=v
            summaries.append(summary)
        comparisons=[('local_8','shared_8'),('local_8','wrong_local_8'),
            ('empirical_local_8','empirical_shared_8'),('local_8','shared_512'),('local_4','shared_256'),
            ('centroid','centroid_random'),('centroid','centroid_radial_random'),('centroid','centroid_gram_random')]
        for a in [.25,.5,1.]:
            comparisons.extend([(f'remove_local_8_{a}',f'remove_local_radial_random_8_{a}'),
                (f'remove_local_8_{a}',f'complement_energy_matched_8_{a}')])
        methods=set(sub.method)
        for a,b in comparisons:
            if {a,b}<=methods:
                for metric in ['next_token_kl','delta_nll','delta_first16_nll','mse_per_coordinate','state_retained_fraction']:
                    pairs.append({**base,'method_a':a,'method_b':b,'metric':metric,
                        'primary':a=='local_8' and b=='shared_8' and base['prefix_tokens']==16 and base['layer']==14 and base['width']==16,
                        **paired_difference(sub,a,b,metric)})
    return pd.DataFrame(summaries),pd.DataFrame(pairs)


def match_mse(per,protocol):
    sub=per
    for field,value in protocol['view'].items():sub=sub.loc[sub[field].eq(value)]
    val=sub.loc[sub.split.eq('validation')];test=sub.loc[sub.split.eq('test')]
    result=[]
    for rank in protocol['local_ranks']:
        local=f'local_{rank}';mse=float(val.loc[val.method.eq(local),'mse_per_coordinate'].mean())
        candidates=[]
        for r in protocol['shared_ranks']:
            error=float(val.loc[val.method.eq(f'shared_{r}'),'mse_per_coordinate'].mean())
            assert mse>0 and error>0 and np.isfinite(error)
            candidates.append({'rank':r,'validation_mse':error,'selection_distance':abs(float(np.log(error/mse)))})
        chosen=min(candidates,key=lambda x:(x['selection_distance'],x['rank']));shared=f'shared_{chosen["rank"]}'
        gap=abs(chosen['validation_mse']/mse-1)
        row={'local_rank':rank,'selected_shared_rank':chosen['rank'],'validation_local_mse':mse,
            'validation_shared_mse':chosen['validation_mse'],'relative_validation_MSE_gap':gap,
            'within_predeclared_10pct_validation_gap':gap<=.1,'all_validation_candidates':candidates}
        for metric in ['next_token_kl','delta_nll','delta_first16_nll','mse_per_coordinate']:
            row['test_'+metric+'_local_minus_shared']=paired_difference(test,local,shared,metric)
        row['test_local_mse']=float(test.loc[test.method.eq(local),'mse_per_coordinate'].mean())
        row['test_shared_mse']=float(test.loc[test.method.eq(shared),'mse_per_coordinate'].mean())
        row['relative_test_MSE_gap']=abs(row['test_shared_mse']/row['test_local_mse']-1)
        result.append(row)
    return result


def draw(summary,per,dest):
    main=summary.loc[summary.split.eq('test')&summary.prefix_tokens.eq(16)&summary.layer.eq(14)&summary.width.eq(16)]
    selected=main.set_index('method')
    methods=['centroid','global_8','shared_8','local_8','wrong_local_8','empirical_shared_8','empirical_local_8']
    labels=['GMM center','Global8','Center+shared8','Center+local8','Wrong local8*','Emp.center+shared8','Emp.local PCA8']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric in zip(axes,['next_token_kl','delta_first16_nll']):
        sub=selected.loc[methods];positions=np.arange(len(methods));values=sub[f'{metric}_estimate']
        ax.bar(positions,values,color=['#9aa4b3','#8196c5','#cf9c5a','#269887','#ba7b9f','#edbf87','#78beae'])
        ax.vlines(positions,sub[f'{metric}_low'],sub[f'{metric}_high'],color='black',lw=1)
        ax.set_xticks(positions,labels,rotation=32,ha='right');ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Next-token KL (nats)' if metric=='next_token_kl' else 'First16 reference ΔNLL (nats/token)')
    fig.suptitle(f'Common-anchor locality · Qwen2 / MATH · block14 · generated16 · width16\nHistorical-test intervention pilot n={int(main.n.iloc[0])}; pointwise 95% question bootstrap; *3 permutations averaged within question')
    save(fig,dest,'locality_common_anchor')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),layout='constrained')
    for family,label,color in [('local','K64 local bases','#269887'),('shared','One shared basis','#cf9c5a')]:
        sub=main.loc[main.family.eq(family)].sort_values('rank')
        for ax,budget in zip(axes,['coordinate','parameter']):
            x=sub['rank'].to_numpy()* (64 if family=='local' and budget=='parameter' else 1)
            ax.plot(x,sub.next_token_kl_estimate,'o-',label=label,color=color)
            ax.fill_between(x,sub.next_token_kl_low,sub.next_token_kl_high,color=color,alpha=.15)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_ylabel('Next-token KL (log scale)')
            ax.set_xlabel('Continuous coordinates per token' if budget=='coordinate' else 'Stored basis scalars / hidden dimension')
            ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.suptitle('Two different budgets · common fixed GMM centers / encoder\nEqual parameter count can preserve different numbers of continuous coordinates')
    save(fig,dest,'locality_budgets')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    families=[('remove_local','Remove local u','#269887'),('remove_local_radial_random','Radial matched random*','#777777'),
        ('complement_energy_matched','Complement, same energy','#cf9c5a'),('remove_complement','Remove complement v','#986ac2')]
    for family,label,color in families:
        sub=main.loc[main.family.eq(family)].sort_values('alpha')
        for ax,metric in zip(axes,['next_token_kl','delta_first16_nll']):
            x=np.r_[0,sub.alpha];y=np.r_[0,sub[f'{metric}_estimate']]
            ax.plot(x,y,'o-',label=label,color=color)
            ax.fill_between(x,np.r_[0,sub[f'{metric}_low']],np.r_[0,sub[f'{metric}_high']],color=color,alpha=.12)
            ax.set_xlabel('Dose α');ax.set_xticks([0,.25,.5,1.]);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Next-token KL (nats)');axes[1].set_ylabel('First16 reference ΔNLL');axes[0].legend(fontsize=8)
    fig.suptitle('Ablation from original activations · baseline is clean h\nComplement removal has its own energy; separate matched-energy control included; *3 seeds averaged within question')
    save(fig,dest,'locality_clean_ablation')
    controls=['centroid','centroid_random','centroid_radial_random','centroid_gram_random']
    sub=selected.loc[controls];fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');x=np.arange(4)
    ax.bar(x,sub.next_token_kl_estimate,color=['#9aa4b3','#777','#cf9c5a','#269887'])
    ax.vlines(x,sub.next_token_kl_low,sub.next_token_kl_high,color='black',lw=1)
    ax.set_xticks(x,['Center replacement','Energy matched','Radial + energy matched','Error Gram preserved'],rotation=15,ha='right')
    ax.set_ylabel('Next-token KL');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Finite-amplitude corruption controls\nRandom controls retain original activation; radial and Gram controls preserve different properties')
    save(fig,dest,'locality_corruptions')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    q=per.loc[per.split.eq('test')&per.prefix_tokens.eq(16)&per.layer.eq(14)&per.width.eq(16)]
    for method,label,color in [('local_8','Local8','#269887'),('shared_8','Shared8','#cf9c5a'),('wrong_local_8','Wrong local8*','#ba7b9f')]:
        sub=q.loc[q.method.eq(method)];ax.scatter(sub.mse_per_coordinate,sub.next_token_kl,label=label,color=color,alpha=.7,s=24)
    ax.set_xlabel('Activation MSE per coordinate');ax.set_ylabel('Next-token KL');ax.set_xscale('log');ax.set_yscale('log');ax.legend()
    fig.suptitle('Descriptive geometry–function relation, one point per question\nNot an error-matched causal comparison; *average of three wrong-basis permutations')
    save(fig,dest,'locality_error_function')


def save(fig,dest,name):
    for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def run(root):
    protocol_path=Path(__file__).resolve().parents[1]/'configs/revision_locality_mse_match_20260923.json'
    protocol=json.loads(protocol_path.read_text())
    frozen=provenance(protocol,[protocol_path])
    if (root/'mse_match_plan.json').exists():
        old=json.loads((root/'mse_match_plan.json').read_text())
        assert old['config']==protocol and old['files']==frozen['files']
    else:freeze(root/'mse_match_plan.json',frozen)
    raw,per,audit=collect(root);summary,pairs=tables(per);dest=root/'report';dest.mkdir(exist_ok=True)
    matched=match_mse(per,protocol);write_json(dest/'validation_mse_match.json',matched)
    for name,table in [('individual_conditions',raw),('per_question_seed_average',per),('summary',summary),('paired_methods',pairs)]:
        table.to_parquet(dest/f'{name}.parquet',index=False)
        if name in ['summary','paired_methods']:write_json(dest/f'{name}.json',table.to_dict('records'))
    primary=pairs.loc[pairs.primary&pairs.metric.isin(['next_token_kl','delta_nll'])].to_dict('records')
    write_json(dest/'primary.json',primary);draw(summary,per,dest)
    notes=['主比较预先固定为block14、生成16、width16、同一GMM中心local8−shared8；原0.648经验中心PCA单列桥接。',
        '新干预题排除旧64题，但历史MATH测试已探索，不是全新确认集。验证／测试分别报告，所有区间为pointwise。',
        '每个token有自己的region ID和连续坐标；模型其余上下文保留。不是整段推理8维，也不是完整模型压缩。',
        'shared512与K64 local8只匹配basis存储标量数，不匹配每token连续编码维数。',
        '错误基中心不变，3个置换全部保存；图和配对先在题内平均seed，再bootstrap问题，不扩充样本量。',
        '径向匹配与Gram匹配是不同对照，理想不变量通过审计，实际bf16偏差见下表。随机变化保留原activation，不是压缩decoder。',
        '从clean移除局部部分，与从center加回局部部分回答不同问题；所有操作重新分配state并报告保留率。',
        'KL/NLL评价原模型输出保真，不等于正确率或选择性控制。MSE匹配只按验证均值选择rank，测试差距单列，不能说逐题误差相同。',
        '首token、前16token和完整参考NLL都保存；末层过去位置不影响未来KV，不能把辅助末层的宽窗口当多位置因果作用。']
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Locality controls</title><style>body{font:16px system-ui;margin:30px;max-width:1450px;line-height:1.6}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px}</style>',
        '<h1>局部方向为什么有效？同中心、共享／局部／错误基与clean-state消融</h1><ul>']
    parts+=['<li>'+html.escape(n)+'</li>' for n in notes];parts+=['</ul><h2>预定主比较</h2>',pd.DataFrame(primary).to_html(index=False),
        '<details><summary>仅验证MSE选择shared rank，再评价历史测试</summary><p>均值MSE接近不等于逐题匹配，10%门槛和测试MSE差异明确报告；不改变主比较。</p><pre>',
        html.escape(json.dumps(matched,indent=2,ensure_ascii=False)),'</pre></details>']
    for name in ['locality_common_anchor','locality_budgets','locality_clean_ablation','locality_corruptions','locality_error_function']:
        parts+=[f'<img src="{name}.png"><p><a href="{name}.pdf">PDF</a></p>']
    for title,table in [('全部设置',summary),('配对比较',pairs)]:
        parts+=[f'<details><summary>{title} · {len(table)}行</summary><div class="scroll">',table.to_html(index=False),'</div></details>']
    parts+=['<details><summary>执行审计和bf16控制偏差</summary><pre>',html.escape(json.dumps(audit,indent=2,ensure_ascii=False)),'</pre></details></html>']
    (dest/'index.html').write_text('\n'.join(parts))
    write_json(dest/'_SUCCESS.json',{'source_audit_sha256':sha(root/'functional/audit.json'),
        'primary':primary,'conditions':len(raw),'question_averaged_conditions':len(per),
        'report_code_sha256':sha(Path(__file__)),'files':{p.name:sha(p) for p in dest.glob('*.parquet')}})
    print(json.dumps({'conditions':len(raw),'primary':primary},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)

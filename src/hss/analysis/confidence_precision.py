"""BF16 readout and inherited generation-policy sensitivity, CPU only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hss.analysis.channel_data import ChannelData,load_config
from hss.analysis.confidence_data import head_weights
from hss.analysis.confidence_study import partitions,control_features,tuned_fit,summary_score
from hss.analysis.channel_study import paired_auc_delta_ci
from hss.experiments.artifacts import save_json,file_digest


def penalize(z,prompts,penalty):
    z=z.copy()
    for i,prompt in enumerate(prompts):
        ids=np.unique(prompt)
        selected=z[i,ids]
        z[i,ids]=np.where(selected<0,selected*penalty,selected/penalty)
    return z


def metrics(z):
    arg=z.argmax(1)
    top=np.partition(z,-2,axis=1)[:,-2:]
    hi,lo=top.max(1),top.min(1)
    z=z-hi[:,None];ex=np.exp(z);total=ex.sum(1,dtype=np.float64)
    entropy=np.log(total)-(ex*z).sum(1,dtype=np.float64)/total
    return pd.DataFrame({'entropy':entropy,'margin':hi-lo,'argmax':arg})


def run(cfg):
    import torch
    torch.set_num_threads(cfg['threads'])
    cc=load_config(cfg['channel_config']);root=Path(cfg['output_root'])
    for ds in cc['datasets']:
        name=ds['name'];data=ChannelData(cc,name);rows=data.rows;out=root/name
        w,gamma,mc,path,key=head_weights(cfg,data.info['model'])
        gc=json.loads((path/'generation_config.json').read_text())
        penalty=float(gc.get('repetition_penalty',1.))
        frames=[pd.read_parquet(Path(ds['path'])/s/'data.parquet',columns=['sample_id','prompt_token_ids_json']) for s in data.info['shards']]
        raw=pd.concat(frames,ignore_index=True).iloc[data._indices]
        assert np.array_equal(raw.sample_id,rows.sample_id)
        prompts=[json.loads(v) for v in raw.prompt_token_ids_json]
        x=data.array('prompt_last',data.info['model']['n_layers']-1)
        frames=[]
        for start in range(0,len(x),cfg['logit_batch']):
            logits=x[start:start+cfg['logit_batch']]@w.T
            rounded=torch.from_numpy(logits).bfloat16().float().numpy()
            native=metrics(rounded).add_prefix('bf16_')
            policy=metrics(penalize(rounded,prompts[start:start+len(logits)],penalty)).add_prefix('policy_')
            frames.append(pd.concat([native,policy],axis=1))
        sensitivity=pd.concat(frames,ignore_index=True)
        sensitivity.insert(0,'sample_id',rows.sample_id)
        sensitivity.to_parquet(out/'precision_scalars.parquet',index=False)
        train,test,_=partitions(cfg,name,rows);y=rows.y.to_numpy(int)
        records=[]
        for metric in ['bf16_entropy','policy_entropy','bf16_margin','policy_margin']:
            score=sensitivity[metric].to_numpy()
            sign=1 if roc_auc_score(y[train],score[train])>=.5 else -1
            records.append({'metric':metric,'sign':sign,**summary_score(y[test],sign*score[test],cfg)})
        prior=json.loads((Path(cfg['prior_root'])/name/'probes.json').read_text())
        comparisons=[]
        for pos in ['prompt_last','t1']:
            view='pre_'+pos
            x=data.array('pre_prompt_last') if pos=='prompt_last' else ChannelData(cc,name,'tokens').array('pre',0)
            ids=next(r['indices'] for r in prior[view] if r.get('kind')=='middle_channels' and r.get('k')==16)
            for diagnostic in [False,True]:
                c,_=control_features(rows,x,train,diagnostic)
                c=np.column_stack([c,sensitivity.policy_entropy])
                base=tuned_fit(c,y,train,cfg);aug=tuned_fit(np.column_stack([c,x[:,ids]]),y,train,cfg)
                base_summary=summary_score(y[test],base['score'][test],cfg)
                aug_summary=summary_score(y[test],aug['score'][test],cfg)
                comparisons.append({'view':view,'diagnostic':diagnostic,'base':base_summary,'augmented':aug_summary,
                    'delta':aug_summary['auc']-base_summary['auc'],
                    'ci':paired_auc_delta_ci(y[test],base['score'][test],aug['score'][test],cfg['seed'],cfg['bootstrap'])})
        original=pd.read_parquet(out/'scalars.parquet')
        agreement={k:float(np.mean(sensitivity[k].to_numpy()==rows.first_token.to_numpy())) for k in ['bf16_argmax','policy_argmax']}
        report={'checkpoint_generation_config':gc,'effective_repetition_penalty':penalty,
            'do_sample_effective':False,'temperature_top_p_top_k_inactive':True,
            'code_sha256':file_digest(__file__),'agreement':agreement,'scalars':records,'comparisons':comparisons,
            'fp32_bf16_entropy_spearman':float(original.prompt_last_entropy.corr(sensitivity.bf16_entropy,method='spearman')),
            'fp32_bf16_entropy_abs_difference_max':float(np.max(np.abs(original.prompt_last_entropy-sensitivity.bf16_entropy))),
            'caveat':'Approximate native logits by rounding FP32 head product to BF16. Teacher-forced states can differ numerically from generate prefill; original generation logits were not retained.'}
        save_json(out/'precision.json',report)
        print(json.dumps({'dataset':name,'agreement':agreement,'scalars':records}),flush=True)

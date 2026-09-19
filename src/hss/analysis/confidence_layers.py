"""Frozen-direction transfer versus same-layer linear accessibility."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from hss.analysis.channel_data import load_config,ChannelData
from hss.analysis.confidence_data import layer_array
from hss.analysis.confidence_study import partitions,fit_fixed
from hss.analysis.confidence_bootstrap import auc_ci
from hss.experiments.artifacts import save_json,file_digest


def score_summary(y,score,cfg):
    return {'auc':float(roc_auc_score(y,score)),'ci':auc_ci(y,score,cfg['seed'],cfg['bootstrap']),'n':len(y)}


def run(cfg):
    cc=load_config(cfg['channel_config']);root=Path(cfg['output_root'])
    for ds in cc['datasets']:
        name=ds['name']
        if name not in cfg['layer_datasets']:
            continue
        data=ChannelData(cc,name);rows=data.rows;y=rows.y.to_numpy(int)
        tr,te,_=partitions(cfg,name,rows)
        enough=rows.n_tokens.to_numpy()>=cfg['layer_position']
        train,test=tr[enough[tr]],te[enough[te]];out=root/name
        if (out/'layers.json').exists():
            old=json.loads((out/'layers.json').read_text())
            if old.get('code_sha256')==file_digest(__file__):
                continue
        frozen={pos:np.load(out/('pre_'+pos+'__full_direction.npz')) for pos in ['prompt_last','t1']}
        records=[];n_layers=data.info['model']['n_layers']
        for pos in ['prompt_last','t16']:
            for layer in range(1,n_layers):
                if pos=='prompt_last':
                    x=data.array('pre_prompt_last') if layer==n_layers-1 else data.array('prompt_last',layer)
                else:
                    x=layer_array(cfg,data,name,layer)
                for source,fit in frozen.items():
                    score=x[test]@fit['coef']+fit['intercept']
                    records.append({'position':pos,'layer':layer,'method':'frozen_'+source,
                                    **score_summary(y[test],score,cfg)})
                fit=fit_fixed(x.astype(float),y,train,float(frozen['prompt_last']['c']))
                records.append({'position':pos,'layer':layer,'method':'layer_specific_probe',
                    **score_summary(y[test],fit['score'][test],cfg),'train_n':len(train),'c':fit['c']})
                np.savez(out/f'{pos}_L{layer}_probe.npz',coef=fit['coef'],intercept=fit['intercept'],c=fit['c'],test_indices=test,test_scores=fit['score'][test])
                print(json.dumps({'dataset':name,'position':pos,'layer':layer,'auc':records[-1]['auc']}),flush=True)
        save_json(out/'layers.json',{'same_cohort_n':len(test),'train_n':len(train),'position':cfg['layer_position'],
            'final_layer':'pre_RMS','records':records,'code_sha256':file_digest(__file__),
            'bootstrap_code_sha256':file_digest(Path(__file__).with_name('confidence_bootstrap.py')),
            'limit':'Frozen failure does not prove erasure; same-layer fits auxiliary. Prompt states are unchanged after generation, which does not prove attention retrieval.'})

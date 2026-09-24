"""CPU algebraic readout sensitivity; NOT upstream causal generation evidence."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from hss.analysis.channel_data import load_config,ChannelData
from hss.analysis.confidence_data import head_weights
from hss.analysis.confidence_study import partitions
from hss.experiments.artifacts import save_json,file_digest


def intervention(h,v,direction,alpha,w,gamma,epsilon):
    coefficient=(alpha-1)*(h@v)
    modified=h+coefficient[:,None]*direction
    original_rms=np.sqrt(np.mean(h*h,axis=1)+epsilon)
    new_rms=np.sqrt(np.mean(modified*modified,axis=1)+epsilon)
    base=(h/original_rms[:,None]*gamma)@w.T
    readout_delta=w@(gamma*direction)
    scale=original_rms/new_rms
    exact=(base+coefficient[:,None]*readout_delta[None,:]/original_rms[:,None])*scale[:,None]
    temperature=base*scale[:,None]
    return base,exact,temperature,modified


def compare(base,changed,temperature):
    lb=base-logsumexp(base,axis=1,keepdims=True);pb=np.exp(lb)
    lc=changed-logsumexp(changed,axis=1,keepdims=True);pc=np.exp(lc)
    lt=temperature-logsumexp(temperature,axis=1,keepdims=True)
    entropy_b=-(pb*lb).sum(1);entropy_c=-(pc*lc).sum(1)
    return {'argmax_changed_fraction':float(np.mean(base.argmax(1)!=changed.argmax(1))),
        'temperature_argmax_changed_fraction':float(np.mean(base.argmax(1)!=temperature.argmax(1))),
        'entropy_delta_mean':float(np.mean(entropy_c-entropy_b)),
        'kl_native_to_changed_mean':float(np.mean((pb*(lb-lc)).sum(1))),
        'kl_changed_to_matched_temperature_mean':float(np.mean((pc*(lc-lt)).sum(1)))}


def run(cfg):
    cc=load_config(cfg['channel_config']);root=Path(cfg['output_root'])
    for ds in cc['datasets']:
        if ds['name']=='llama32_mmlu':
            continue
        name=ds['name'];data=ChannelData(cc,name);out=root/name
        train,_,_=partitions(cfg,name,data.rows)
        rng=np.random.default_rng(cfg['seed']);chosen=np.sort(rng.choice(train,min(256,len(train)),replace=False))
        h=data.array('pre_prompt_last')[chosen].astype(np.float64)
        w,gamma,mc,_,_=head_weights(cfg,data.info['model'])
        v=np.load(out/'readout_geometry.npz')['v_min']
        records=[]
        directions={'weakest_v':v}
        for i in range(3):
            d=rng.normal(size=h.shape[1]);directions[f'norm_matched_random_{i}']=d/np.linalg.norm(d)
        for kind,d in directions.items():
            for alpha in [0.,.5,1.5,2.]:
                base,changed,temp,modified=intervention(h,v,d,alpha,w,gamma,mc['rms_norm_eps'])
                item={'direction':kind,'alpha':alpha,**compare(base,changed,temp),
                    'relative_perturbation_norm_mean':float(np.mean(np.linalg.norm(modified-h,axis=1)/np.linalg.norm(h,axis=1)))}
                records.append(item)
                print(json.dumps({'dataset':name,**item}),flush=True)
        save_json(out/'readout_check.json',{'n':len(chosen),'partition':'discovery only',
            'sample_ids':data.rows.sample_id.iloc[chosen].tolist(),'records':records,'code_sha256':file_digest(__file__),
            'not_causal_generation':True,'accuracy_not_measured':True,
            'interpretation':'Direct final-readout diagnostic using ideal FP64 RMS algebra. No decoder rerun, no sampling, no answer correctness change measured. Candidate v and random controls have exactly matched per-question perturbation norm.'})

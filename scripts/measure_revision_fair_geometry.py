"""CPU-only geometry/density measurements on frozen comparison inputs.

No model logits or functional conclusions. All per-question/per-token summaries
and training/held-out likelihood definitions are saved without choosing ranks.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from revision_common import config,freeze,provenance,sha,write_json,status,nearest
from revision_factor_common import FairDecoders,key,budget
from revision_locality_common import geometry_metrics
from prepare_revision_fair_comparison import load_arrays


def verify_inputs(root):
    success=json.loads((root/'prepare_SUCCESS.json').read_text())
    assert sha(root/'plan.json')==success['plan_sha256']
    plan=json.loads((root/'plan.json').read_text())
    for filename,digest in plan['files'].items():assert sha(filename)==digest,filename
    return plan


def measure(root):
    plan=verify_inputs(root);cfg=plan['config'];started=time.monotonic()
    dest=root/'geometry';dest.mkdir(parents=True,exist_ok=True)
    files=[root/'plan.json',root/'prepare_SUCCESS.json',Path(__file__),
           Path(__file__).with_name('revision_factor_common.py'),Path(__file__).with_name('revision_locality_common.py')]
    frozen=provenance(cfg,files)
    if (dest/'_SUCCESS.json').exists():
        receipt=json.loads((dest/'_SUCCESS.json').read_text())
        assert sha(dest/'plan.json')==receipt['plan_sha256']
        for filename,digest in receipt['files'].items():assert sha(dest/filename)==digest,filename
        print('Completed geometry files verified.');return
    freeze(dest/'plan.json',frozen)
    decoders=FairDecoders(load_arrays(root/'decoders.npz'));rows=[];densities=[]
    training_ids=set(json.loads((Path(cfg['fit_root'])/'training_identity.json').read_text())['question_ids'])
    for dataset,spec in plan['datasets'].items():
        folder=root/'inputs'/dataset
        frame=pd.read_parquet(folder/'rows.parquet').set_index('sample_id')
        captured=load_arrays(folder/'captures.npz');mapping=dict(zip(captured['sample_ids'].tolist(),captured['x']))
        assert not set(spec['selected_ids'])&training_ids
        for sid in spec['selected_ids']:
            x=mapping[sid].astype(float);split=str(frame.loc[sid,'split'])
            assigned=nearest(x,decoders.arrays['gmm_centers'])
            for rank in cfg['ranks']:
                fa=decoders.fixed[rank];joint=decoders.joint[rank]
                component=fa.component_log_density(x)
                # Counts define a proper mixture density, distinct from nearest
                # GMM assignment. The assigned score alone isn't that density.
                from revision_factor_common import logsumexp
                fixed_ll=logsumexp(component+np.log(fa.weights),axis=1)
                _,joint_ll=joint.posterior(x)
                densities.append({'dataset':dataset,'split':split,'sample_id':sid,'rank':rank,
                    'fixed_mixture_log_density':float(fixed_ll.mean()),
                    'fixed_assigned_component_log_density':float(component[np.arange(len(x)),assigned].mean()),
                    'joint_mixture_log_density':float(joint_ll.mean()),'tokens':len(x)})
            for condition in plan['conditions']:
                z,coding=decoders.replace(x,condition)
                actual=torch.from_numpy(z).to(torch.bfloat16).float().numpy()
                ideal=geometry_metrics(x,z,decoders.arrays['gmm_centers'])
                rounded=geometry_metrics(x,actual,decoders.arrays['gmm_centers'])
                row={'dataset':dataset,'split':split,'sample_id':sid,'method':key(condition),
                     'rank':condition.get('rank',0),'ideal_mse':float(np.square(x-z).mean()),
                     'actual_bf16_mse':float(np.square(x-actual).mean()),
                     'actual_region_retention':rounded['state_retained_fraction'],
                     'actual_norm_ratio_mean':float(np.mean(np.linalg.norm(actual,axis=1)/np.linalg.norm(x,axis=1))),
                     'geometry_ideal':ideal,'geometry_bf16':rounded,'coding':coding}
                rows.append(row)
            status(root,'geometry',state='running',questions=len(rows)//len(plan['conditions']),expected_questions=128)
    assert len(rows)==plan['expected_conditions']
    write_json(dest/'per_question.json',rows);write_json(dest/'heldout_density.json',densities)
    simple=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
    pd.DataFrame(simple).to_parquet(dest/'per_question.parquet',index=False)
    pd.DataFrame(densities).to_parquet(dest/'heldout_density.parquet',index=False)
    # Full training mixture density (not a held-out score), streamed on CPU.
    fitplan=json.loads((Path(cfg['fit_root'])/'plan.json').read_text())
    training=Path(fitplan['config']['staging'])/'training.npy'
    assert sha(training)==fitplan['files'][str(training)]
    train=np.load(training,mmap_mode='r');train_rows=[]
    for rank in cfg['ranks']:
        total={'fixed':0.,'joint':0.};seen=0
        for start in range(0,len(train),1024):
            batch=np.asarray(train[start:start+1024])
            for kind,model in [('fixed',decoders.fixed[rank]),('joint',decoders.joint[rank])]:
                _,score=model.posterior(batch);total[kind]+=float(score.sum())
            seen+=len(batch)
            status(root,'geometry',state='training_density',rank=rank,completed_tokens=seen,expected_tokens=len(train))
        train_rows.append({'rank':rank,'tokens':seen,'fixed_mixture_log_density':total['fixed']/seen,
                           'joint_mixture_log_density':total['joint']/seen})
    write_json(dest/'training_density.json',train_rows)
    output_names=['per_question.json','per_question.parquet','heldout_density.json','heldout_density.parquet','training_density.json']
    write_json(dest/'_SUCCESS.json',{'complete':True,'questions':len(rows)//len(plan['conditions']),
        'conditions':len(rows),'scope':'CPU geometry and generative densities only; output KL/NLL not measured.',
        'plan_sha256':sha(dest/'plan.json'),'files':{name:sha(dest/name) for name in output_names},
        'seconds':time.monotonic()-started})
    status(root,'geometry',state='complete',questions=len(rows)//len(plan['conditions']),conditions=len(rows),seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):measure(a.root)

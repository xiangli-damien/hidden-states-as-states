"""Fresh-cache, once-hook functional comparison of frozen PCA/FA/MFA decoders."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from evaluate_revision_locality import measure
from extract_revision_prefixes import load_model
from revision_common import freeze,provenance,sha,write_json,write_npz,status
from revision_factor_common import FairDecoders,key
from revision_locality_common import geometry_metrics
from prepare_revision_fair_comparison import load_arrays
from measure_revision_fair_geometry import verify_inputs


def tasks(plan,smoke=False):
    result=[]
    for dataset,spec in plan['datasets'].items():
        ids=spec['selected_ids'][:1] if smoke else spec['selected_ids']
        for sid in ids:
            if '/' in sid or '\\' in sid:raise ValueError('Invalid sample ID for output filename')
            for condition in plan['conditions']:
                result.append({'dataset':dataset,'sample_id':sid,'condition':condition,
                               'name':f'{dataset}_{sid}_{key(condition)}'})
    return result


def inputs(root,plan):
    datasets={}
    for dataset in plan['datasets']:
        folder=root/'inputs'/dataset
        frame=pd.read_parquet(folder/'rows.parquet').set_index('sample_id')
        tokens={r['sample_id']:r for r in json.loads((folder/'tokens.json').read_text())}
        cap=load_arrays(folder/'captures.npz')
        datasets[dataset]=(frame,tokens,dict(zip(cap['sample_ids'].tolist(),cap['x'])))
    return datasets


def run(root,smoke=False):
    plan=verify_inputs(root);cfg=plan['config'];stage='smoke' if smoke else 'functional'
    dest=root/stage;dest.mkdir(parents=True,exist_ok=True)
    lock=(root/'functional.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    wait=Path(cfg['wait_for_collection'])/'queue_status.json'
    if json.loads(wait.read_text())['state']!='complete':
        raise RuntimeError('GSM8K collection queue has GPU priority until complete')
    expected=tasks(plan,smoke);data=inputs(root,plan)
    files=[root/'plan.json',root/'prepare_SUCCESS.json',Path(__file__),
        Path(__file__).with_name('revision_factor_common.py'),Path(__file__).with_name('revision_common.py'),
        Path(__file__).with_name('revision_locality_common.py'),Path(__file__).with_name('evaluate_revision_locality.py'),
        Path(__file__).with_name('extract_revision_prefixes.py'),Path(__file__).with_name('prepare_revision_fair_comparison.py'),
        Path(__file__).with_name('measure_revision_fair_geometry.py')]
    frozen=provenance(cfg,files);frozen.update(expected=expected,smoke=smoke,
        input_plan_sha256=sha(root/'plan.json'),scope='Raw forward KL and complete stored-reference NLL; no new generation or correctness claims.')
    if (dest/'plan.json').exists():
        previous=json.loads((dest/'plan.json').read_text())
        # A documentation-only commit may change HEAD; exact code/input SHA and
        # protocol, rather than a fresh git timestamp, govern safe resumption.
        assert {k:v for k,v in previous.items() if k!='git'}=={k:v for k,v in frozen.items() if k!='git'}
    else:freeze(dest/'plan.json',frozen)
    decoders=FairDecoders(load_arrays(root/'decoders.npz'))
    model,_=load_model(cfg)
    eos=model.generation_config.eos_token_id
    assert set(eos if isinstance(eos,list) else [eos])==set(cfg['eos_ids'])
    baseline={};done=0;started=time.monotonic()
    status(root,stage,state='running',completed=0,expected=len(expected))
    for task in expected:
        dataset,sid=task['dataset'],task['sample_id'];frame,tokens,captures=data[dataset]
        item=tokens[sid];prefix=item['prompt_ids']+item['response_ids'][:16];reference=item['response_ids'][16:]
        positions=list(range(len(prefix)-16,len(prefix)));saved=captures[sid]
        output=dest/'samples'/(task['name']+'.json');arraypath=output.with_suffix('.npz');ident=(dataset,sid)
        if output.exists():
            record=json.loads(output.read_text());assert record['task']==task and sha(arraypath)==record['arrays_sha256']
            if task['condition']['method']=='identity':baseline[ident]=load_arrays(arraypath)['logp']
            done+=1;continue
        details={}
        def transform(h):
            x=h.float().cpu().numpy();np.testing.assert_array_equal(x,saved)
            z,coding=decoders.replace(x,task['condition'])
            actual=torch.from_numpy(z).to(device=h.device,dtype=h.dtype)
            rounded=actual.float().cpu().numpy()
            details.update(ideal=geometry_metrics(x,z,decoders.arrays['gmm_centers']),
                actual=geometry_metrics(x,rounded,decoders.arrays['gmm_centers']),
                coding=coding,rounded=rounded)
            return actual
        metrics,logp,loss=measure(model,prefix,reference,cfg['layer'],positions,transform)
        if task['condition']['method']=='identity':baseline[ident]=logp.copy()
        original=baseline[ident].astype(float)
        metrics.update(next_token_kl=float((np.exp(original)*(original-logp)).sum()),
                       next_token_argmax_agreement=bool(original.argmax()==logp.argmax()))
        write_npz(arraypath,logp=logp,reference_nll=loss,replacement=details.pop('rounded'))
        write_json(output,{'task':task,'split':str(frame.loc[sid,'split']),'positions':positions,
            'capture_match_exact':True,**metrics,'geometry':details,'arrays_sha256':sha(arraypath)})
        done+=1
        if done%19==0:status(root,stage,state='running',completed=done,expected=len(expected),seconds=time.monotonic()-started)
    assert {p.stem for p in (dest/'samples').glob('*.json')}=={t['name'] for t in expected}
    write_json(dest/'_SUCCESS.json',{'complete':True,'conditions':done,'plan_sha256':sha(dest/'plan.json'),'seconds':time.monotonic()-started})
    status(root,stage,state='complete',completed=done,expected=len(expected),seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    try:
        with threadpool_limits(limits=4):run(a.root,a.smoke)
    except BaseException:
        status(a.root,'smoke' if a.smoke else 'functional',state='failed',traceback=traceback.format_exc());raise

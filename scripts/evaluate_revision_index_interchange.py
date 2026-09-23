"""Actual free generation for the frozen notation case, after fair FA/MFA.

Validation measures capability only. Test is forbidden until its independent
validation audit passes. No rank/position/template selection from outcomes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from extract_revision_prefixes import load_model
from revision_common import OncePatch,provenance,freeze,sha,write_json,write_npz,status,nearest
from revision_index_interchange_common import ordinary_correct,score_suffix,transfer,capability_gate


@torch.inference_mode()
def generate(model,tokenizer,ids,positions=None,transform=None,layer=14):
    patch=OncePatch(positions,len(ids),transform) if positions is not None else None
    handle=model.model.layers[layer-1].register_forward_hook(patch) if patch else None
    try:
        out=model.generate(input_ids=torch.tensor([ids],device=model.device),
            do_sample=False,repetition_penalty=1.0,max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        generated=out[0,len(ids):].tolist()
        text=tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        assert generated
        if patch:assert patch.calls==1
        return {'generated_ids':generated,'text':text,'finish_reason':'eos' if generated[-1] in eos else 'length',
                'patch_calls':patch.calls if patch else 0,'actual_patch_energy':patch.energy if patch else 0.0}
    finally:
        if handle:handle.remove()


@torch.inference_mode()
def capture(model,ids,positions,layer):
    holder={}
    def hook(module,args,out):
        x=out[0] if isinstance(out,tuple) else out
        holder['x']=x[0,positions].float().cpu().numpy()
    handle=model.model.layers[layer-1].register_forward_hook(hook)
    try:
        model(input_ids=torch.tensor([ids],device=model.device),use_cache=True,logits_to_keep=1)
    finally:handle.remove()
    return holder['x']


def eligible(record):
    return record['aligned'] and all(record[s]['free_correct'] and record[s]['prefixed_correct'] for s in ['recipient','donor'])


def run(root,stage):
    lock=(root/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    assert json.loads((root/'prepare_SUCCESS.json').read_text())['plan_sha256']==sha(root/'plan.json')
    for path,digest in plan['files'].items():assert sha(Path(path))==digest
    fair=Path(cfg['wait_for_fair_queue'])
    assert json.loads((fair/'queue_status.json').read_text())['state']=='complete'
    assert json.loads((fair/'report/statistics_audit.json').read_text())['complete']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'Another GPU process is active'
    if stage=='test':
        gate=json.loads((root/'validation_audit.json').read_text())
        assert gate['passed'] and gate['capability_gate']['passed']
        assert gate['stage_receipt_sha256']==sha(root/'validation/_SUCCESS.json')
    dest=root/stage;dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists():raise RuntimeError('Stage complete; audit or report it without rerunning')
    files=[root/'plan.json',Path(__file__),Path(__file__).with_name('revision_index_interchange_common.py'),
           Path(__file__).with_name('revision_common.py'),Path(__file__).with_name('extract_revision_prefixes.py')]
    identity=provenance(cfg,files)
    ep=root/'evaluation_plan.json'
    if ep.exists():
        old=json.loads(ep.read_text());assert old['config']==identity['config'] and old['files']==identity['files']
    else:freeze(ep,identity)
    model,tokenizer=load_model(cfg)
    generation={'model_defaults':model.generation_config.to_dict(),'overrides':cfg['decoding'],
                'attention':'sdpa','dtype':'bfloat16','model_revision':cfg['revision']}
    freeze(root/'generation_config.json',generation)
    with np.load(cfg['decoder'],allow_pickle=False) as f:
        centers=f['centers'].copy();local=f['local_basis'][:,:8].copy();shared=f['shared_basis'][:8].copy()
    cases=[r for r in plan['cases'] if r['case']['split']==stage]
    records=[];patch_count=0
    status(root,'interchange',phase=stage,state='running',completed=0,expected=len(cases))
    for item in cases:
        case=item['case'];cid=case['id'];path=dest/'cases'/f'{cid}.json'
        if path.exists():
            record=json.loads(path.read_text());assert record['case']==case;records.append(record)
            patch_count+=len(record['patches']);continue
        record={'case_id':cid,'case':case,'aligned':item['aligned'],'patches':[]}
        for side in ['recipient','donor']:
            value=item[side]
            free=generate(model,tokenizer,value['prompt_ids'])
            prefixed=generate(model,tokenizer,value['input_ids'])
            record[side]={'free':free,'prefixed':prefixed,
                'free_correct':ordinary_correct(free['text'],case[side]),
                'prefixed_correct':ordinary_correct(value['assistant_prefix']+prefixed['text'],case[side])}
        record['eligible']=eligible(record)
        if stage=='test' and record['eligible']:
            r,d=item['recipient'],item['donor']
            x=capture(model,r['input_ids'],r['positions16'],cfg['layer'])
            y=capture(model,d['input_ids'],d['positions16'],cfg['layer'])
            cap=dest/'captures'/f'{cid}.npz';write_npz(cap,recipient=x,donor=y)
            record['capture_file']=str(cap.relative_to(dest));record['capture_sha256']=sha(cap)
            for width in cfg['widths']:
                for method in cfg['methods']:
                    before,donor=x[-width:],y[-width:];saved={}
                    def transform(h):
                        actual_before=h.float().cpu().numpy()
                        np.testing.assert_array_equal(actual_before,before)
                        ideal=transfer(actual_before,donor,centers,local,shared,method)
                        z=torch.from_numpy(ideal).to(device=h.device,dtype=h.dtype)
                        saved['actual']=z.float().cpu().numpy()
                        saved['geometry']={'ideal_squared_displacement':float(np.square(ideal-actual_before).sum()),
                            'actual_squared_displacement_float64':float(np.square(saved['actual'].astype(np.float64)-actual_before).sum()),
                            'recipient_regions':nearest(actual_before,centers).tolist(),
                            'donor_regions':nearest(donor,centers).tolist(),
                            'actual_regions':nearest(saved['actual'],centers).tolist()}
                        return z
                    output=generate(model,tokenizer,r['input_ids'],r['positions16'][-width:],transform,cfg['layer'])
                    if method=='identity':assert output['generated_ids']==record['recipient']['prefixed']['generated_ids']
                    arr=dest/'patches'/f'{cid}_w{width}_{method}.npz';write_npz(arr,actual=saved['actual'])
                    result={'method':method,'width':width,'positions':r['positions16'][-width:],
                        'donor_positions':d['positions16'][-width:],'output':output,
                        'score':score_suffix(output['text'],case['recipient'],case['donor']),'geometry':saved['geometry'],
                        'array_file':str(arr.relative_to(dest)),'array_sha256':sha(arr)}
                    record['patches'].append(result);patch_count+=1
        write_json(path,record);records.append(record)
        status(root,'interchange',phase=stage,state='running',completed=len(records),expected=len(cases),patches=patch_count)
    result={'complete':True,'stage':stage,'pairs':len(records),'eligible_pairs':sum(r['eligible'] for r in records),
            'patch_conditions':patch_count,'seconds':time.time()-started,
            'plan_sha256':sha(root/'plan.json'),'evaluation_plan_sha256':sha(ep),
            'generation_config_sha256':sha(root/'generation_config.json')}
    if stage=='validation':result['capability_gate']=capability_gate(records)
    result['files']={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    write_json(dest/'_SUCCESS.json',result)
    status(root,'interchange',phase=stage,state='complete',completed=len(records),patches=patch_count)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2));lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--stage',required=True,choices=['validation','test']);a=ap.parse_args()
    try:run(a.root,a.stage)
    except BaseException:
        status(a.root,'interchange',phase=a.stage,state='failed',traceback=traceback.format_exc());raise

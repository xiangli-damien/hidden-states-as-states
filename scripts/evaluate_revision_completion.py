"""Real MATH greedy continuation after a one-time four-token block14 patch."""
import argparse,copy,fcntl,inspect,json,os,shutil,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from openact_eval.evaluators.registry import auto_select_evaluator
from extract_revision_prefixes import load_model
from revision_common import sha,freeze,provenance,write_json,write_npz,status
from revision_completion_common import transform,test_conditions


@torch.inference_mode()
def continuation(model, tokenizer, ids, budget, generation, layer=None, positions=None, operation=None):
    """HF builds a NEW complete prefix cache. Remove hook during first prefill."""
    saved={};handles=[];calls=[]
    if positions is not None:
        def hook(module,args,out):
            assert not calls
            h=out[0] if isinstance(out,tuple) else out
            assert h.shape[:2]==(1,len(ids))
            before=h[0,positions].detach().clone();x=before.float().cpu().numpy()
            z,meta=operation(x)
            after=torch.from_numpy(z).to(device=h.device,dtype=h.dtype)
            assert after.shape==before.shape and torch.isfinite(after).all()
            saved.update(before=x,ideal=z,actual=after.float().cpu().numpy(),meta=meta)
            calls.append(True)
            handles[0].remove()  # no hook survives into autoregressive decoding
            if np.array_equal(saved['before'],saved['actual']):return None
            replaced=h.clone();replaced[0,positions]=after
            return (replaced,*out[1:]) if isinstance(out,tuple) else replaced
        handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    try:
        inp=torch.tensor([ids],device=model.device)
        generated=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),
            generation_config=generation,max_new_tokens=budget,use_cache=True)
        output=generated[0,len(ids):].tolist()
        if positions is not None:assert len(calls)==1
        return output,saved
    finally:
        for handle in handles:handle.remove()


def verify_existing(path,plan_sha):
    r=json.loads(path.read_text());assert r['plan_sha256']==plan_sha
    receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    assert receipt['record_sha256']==sha(path)
    for p,h in r['files'].items():assert sha(path.parent/p)==h
    return r


def run(root,stage):
    root=Path(root); lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];plan_sha=sha(root/'plan.json')
    assert json.loads((root/'prepare_SUCCESS.json').read_text())['plan_sha256']==plan_sha
    for p,h in plan['files'].items():assert sha(p)==h,p
    if (root/f'{stage}_SUCCESS.json').exists():raise RuntimeError('Completed stage must not be rerun')
    if stage=='validation':assert json.loads((root/'smoke_audit.json').read_text())['complete']
    selected=None
    if stage=='test':
        selection=json.loads((root/'selection.json').read_text());assert selection['run_test']
        assert selection['validation_audit_sha256']==sha(root/'validation_audit.json')
        selected=selection['selected']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    evaluator=auto_select_evaluator('math')
    source=[root/'plan.json',Path(inspect.getfile(type(evaluator)))]
    source += [Path(__file__).with_name(n) for n in ['evaluate_revision_completion.py','revision_completion_common.py',
        'revision_common.py','revision_locality_common.py','extract_revision_prefixes.py']]
    identity=provenance(cfg,source)
    if (root/'execution_plan.json').exists():
        old=json.loads((root/'execution_plan.json').read_text());assert old['files']==identity['files'] and old['config']==cfg
    else:freeze(root/'execution_plan.json',identity)
    model,tok=load_model(cfg)
    generation=copy.deepcopy(model.generation_config)
    generation.do_sample=False;generation.repetition_penalty=1.;generation.num_beams=1
    generation.max_new_tokens=cfg['max_new_tokens']
    if generation.pad_token_id is None:generation.pad_token_id=tok.pad_token_id or tok.eos_token_id
    freeze(root/'generation_config.json',{'resolved':generation.to_dict(),'dtype':'bfloat16','attention':'sdpa',
        'prefix_flow':'natural16 then fresh full-prefix replay for every condition; no reference continuation'})
    eos=generation.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    support=json.loads((root/'support.json').read_text())
    with np.load(cfg['decoder']) as f:decoder={k:f[k].copy() for k in ['centers','local_basis','shared_basis']}
    cases=[r for r in json.loads((root/'inputs.json').read_text()) if r['split']==('test' if stage=='test' else 'validation')]
    if stage=='smoke':cases=cases[:cfg['smoke_questions']]
    if stage=='test':cases=cases[:selection['test_questions']]
    conditions=([{'name':'baseline','family':'baseline'},*plan['candidates']] if stage!='test' else test_conditions(selected,cfg))
    if stage=='smoke':conditions.insert(1,{'name':'identity','family':'identity'})
    done=0;records={};started=time.time()
    for row in cases:
        if time.time()>plan['deadline_unix']-cfg['reserve_hours']*3600:
            raise TimeoutError('Compute budget exhausted; preserving six-hour audit reserve')
        if shutil.disk_usage('/home/ubuntu').free<50*2**30:raise RuntimeError('SSD reserve under50GiB')
        sid=row['sample_id'];dest=root/'outputs'/sid;dest.mkdir(parents=True,exist_ok=True)
        prefix_path=dest/'prefix.json'
        if not prefix_path.exists():
            tick=time.monotonic();prefix,_=continuation(model,tok,row['prompt_ids'],cfg['prefix'],generation)
            freeze(prefix_path,{'sample_id':sid,'plan_sha256':plan_sha,'prompt_ids':row['prompt_ids'],'generated_prefix':prefix,
                'active':len(prefix)==cfg['prefix'] and not eos.intersection(prefix),'seconds':time.monotonic()-tick})
        pre=json.loads(prefix_path.read_text());assert pre['plan_sha256']==plan_sha and pre['prompt_ids']==row['prompt_ids']
        full_prefix=row['prompt_ids']+pre['generated_prefix']
        positions=list(range(len(full_prefix)-cfg['width'],len(full_prefix))) if pre['active'] else []
        for cond in conditions:
            if time.time()>min(plan['deadline_unix']-cfg['reserve_hours']*3600,
                               plan['validation_decision_unix'] if stage in ('smoke','validation') else plan['deadline_unix']):
                raise TimeoutError('Predeclared compute/validation deadline reached; do not expand the search')
            path=dest/(cond['name']+'.json')
            if path.exists():r=verify_existing(path,plan_sha)
            else:
                tick=time.monotonic();saved={}
                if pre['active']:
                    suffix,saved=continuation(model,tok,full_prefix,cfg['max_new_tokens']-cfg['prefix'],generation,
                        cfg['layer'],positions,lambda x:transform(x,decoder,support,cfg,cond,sid))
                else:suffix=[]
                ids=pre['generated_prefix']+suffix
                assert ids and not eos.intersection(ids[:-1])
                text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                score=evaluator.evaluate_sample(SimpleNamespace(sample_idx=row['source_sample_idx'],response_text=text,
                    ground_truth=row['ground_truth'],meta={'sample_id':sid}))
                assert score.is_correct is not None and not score.error
                files={'prefix.json':sha(prefix_path)};geometry={'energy':0.,'modified_tokens':0,'eligible_tokens':0}
                if saved:
                    arr=dest/(cond['name']+'.npz');write_npz(arr,before=saved['before'],ideal=saved['ideal'],actual=saved['actual'])
                    files[arr.name]=sha(arr)
                    delta=saved['actual'].astype(float)-saved['before']
                    geometry={'energy':float(np.square(delta).sum()),'modified_tokens':int(np.any(delta!=0,axis=1).sum()),
                        'eligible_tokens':sum(saved['meta']['mask']),'regions':saved['meta'],
                        'ideal_token_energy':np.square(saved['ideal']-saved['before']).sum(1).tolist(),
                        'actual_token_energy':np.square(delta).sum(1).tolist(),
                        'actual_radial_dot':np.sum(saved['before']*delta,axis=1).tolist()}
                r={'sample_id':sid,'split':row['split'],'condition':cond,'prompt_text':row['prompt_text'],
                    'generated_ids':ids,'response_text':text,'ground_truth':row['ground_truth'],
                    'correct':bool(score.is_correct),'parsed_answer':score.extracted_answer,'normalized_answer':score.normalized_answer,
                    'parse_failed':bool(score.meta['parse_failed']),'length':len(ids),
                    'finish_reason':'eos' if ids[-1] in eos else 'length','prefix_active':pre['active'],
                    'positions':positions,'hook_calls':int(pre['active']),'hook_removed_after_prefill':True,'fresh_cache':True,
                    'geometry':geometry,'seconds':time.monotonic()-tick,'files':files,
                    'plan_sha256':plan_sha,'execution_plan_sha256':sha(root/'execution_plan.json'),
                    'generation_config_sha256':sha(root/'generation_config.json')}
                write_json(path,r);write_json(path.with_suffix('.receipt.json'),{'record_sha256':sha(path)})
            assert r['condition']==cond
            if cond['family']=='identity':
                assert r['generated_ids']==json.loads((dest/'baseline.json').read_text())['generated_ids']
            records[str(path.relative_to(root))]=sha(path);done+=1
            status(root,'steering',state='running',phase=stage,completed=done,expected=len(cases)*len(conditions),
                   seconds=time.time()-started,last=sid+'/'+cond['name'])
    write_json(root/f'{stage}_SUCCESS.json',{'complete':True,'phase':stage,'questions':len(cases),'conditions':conditions,
        'records':records,'seconds':time.time()-started,'plan_sha256':plan_sha,
        'execution_plan_sha256':sha(root/'execution_plan.json'),'generation_config_sha256':sha(root/'generation_config.json')})
    status(root,'steering',state='complete',phase=stage,completed=done,expected=done,seconds=time.time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--stage',required=True,choices=['smoke','validation','test']);a=p.parse_args()
    try:run(a.root,a.stage)
    except BaseException:status(a.root,'steering',state='failed',phase=a.stage,traceback=traceback.format_exc());raise

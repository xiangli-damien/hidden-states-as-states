"""Finite GPU queue: identity audit -> dev alpha -> three paired test arms.

No old queue is changed or restarted. Every generation uses a fresh prompt KV
cache, greedy BF16 HF decoding and the same 2048-token budget.
"""
import argparse
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from openact_core.tasks.parsers.math_parser import MathParser
from openact_eval.evaluators.registry import auto_select_evaluator
from extract_revision_prefixes import load_model
from revision_common import freeze, write_json, write_npz, sha, provenance
from sink_direction_common import GeneratedTokenShift, boxed_answer, select_alpha, paired_binary


@torch.inference_mode()
def generate(model, tok, row, generation, layer, vector, budget):
    ids = torch.tensor([row['prompt_ids']], device=model.device)
    patch = None; handle = None
    if vector is not None:
        patch = GeneratedTokenShift(ids.shape[1], torch.as_tensor(vector,device=model.device))
        handle = model.model.layers[layer-1].register_forward_hook(patch)
    try:
        output = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            generation_config=generation,max_new_tokens=budget,use_cache=True)
    finally:
        if handle is not None: handle.remove()
    result = output[0,ids.shape[1]:].tolist()
    if patch is not None:
        assert patch.calls==len(result) and len(patch.steps)==len(result)-1
    return result, patch.arrays() if patch is not None else None


def read_record(path, plan_sha):
    r=json.loads(path.read_text());receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    assert r['plan_sha256']==plan_sha and sha(path)==receipt['record_sha256']
    assert sha(path.with_suffix('.npz'))==r['geometry_sha256']
    return r


def summarize(root, plan, selection):
    cfg=plan['config'];plan_sha=sha(root/'plan.json');results={};paired={}
    cases=json.loads((root/'cases.json').read_text())
    records={}; all_files={}
    for split in ['sink_test','normal_test']:
        selected=[r for r in cases if r['split']==split]
        groups={cond:[] for cond in ['zero','hss','random']}
        for row in selected:
            for cond in groups:
                path=root/'outputs'/row['sample_id']/(cond+'.json')
                r=read_record(path,plan_sha);groups[cond].append(r)
                all_files[str(path.relative_to(root))]=sha(path)
        results[split]={cond:{'n':len(rows),'boxed_count':sum(r['complete_boxed'] for r in rows),
            'boxed_rate':float(np.mean([r['complete_boxed'] for r in rows])),
            'correct_count':sum(r['correct'] for r in rows),
            'correct_rate':float(np.mean([r['correct'] for r in rows])),
            'permissive_parse_rate':float(np.mean([not r['parse_failed'] for r in rows])),
            'truncated':sum(r['finish_reason']=='length' for r in rows),
            'mean_tokens':float(np.mean([r['n_tokens'] for r in rows]))} for cond,rows in groups.items()}
        paired[split]={}
        for endpoint in ['complete_boxed','correct']:
            for control in ['random','zero']:
                paired[split][endpoint+'_hss_minus_'+control]=paired_binary(
                    [r[endpoint] for r in groups['hss']], [r[endpoint] for r in groups[control]],
                    cfg['seed'],cfg['bootstrap'])
        records[split]=groups
    primary=paired['sink_test']['complete_boxed_hss_minus_random']
    baseline=paired['sink_test']['complete_boxed_hss_minus_zero']
    success=primary['ci95'][0]>0 and primary['one_sided_exact_p']<.05 and baseline['difference']>0
    result={'complete':True,'selection':selection,'results':results,'paired':paired,
        'predeclared_success':success,'primary_endpoint':plan['audit']['primary_endpoint'],
        'scope':plan['audit']['scope'],'limitations':[plan['audit']['map_fit_scope'],
        'One frozen random direction only; no claim of superiority to all random directions',
        'Boxed compliance measures answer formatting; correctness is a separate secondary endpoint',
        'Cohort selected by historical baseline response; no online sink detector is being validated'],
        'plan_sha256':plan_sha,'records':all_files}
    write_json(root/'summary.json',result)
    lines=['# Response-mean sink-direction exploratory pilot','',
        '**Primary endpoint: complete nonempty boxed answer, not semantic correctness.**','',
        plan['audit']['scope'], '', f"Selected alpha: {selection['alpha']}; predeclared success: {success}", '',
        '| Cohort | Arm | Boxed | Correct | Mean tokens | Truncated |',
        '|---|---|---:|---:|---:|---:|']
    for split,groups in results.items():
        for cond,r in groups.items():
            lines.append(f"| {split} | {cond} | {r['boxed_count']}/{r['n']} | {r['correct_count']}/{r['n']} | {r['mean_tokens']:.1f} | {r['truncated']} |")
    lines.extend(['','## Paired comparisons','',json.dumps(paired,indent=2),'','## Limits','',*result['limitations']])
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'COMPLETE.json',{'summary_sha256':sha(root/'summary.json'),
        'report_sha256':sha(root/'report.md'),'completed_unix':time.time(),'plan_sha256':plan_sha})
    return result


def run(root, smoke_only=False):
    root=Path(root);lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():raise RuntimeError('Finished experiment: do not rerun')
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];plan_sha=sha(root/'plan.json')
    assert json.loads((root/'prepare_SUCCESS.json').read_text())['plan_sha256']==plan_sha
    for p,h in plan['files'].items():assert sha(p)==h,p
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    assert shutil.disk_usage('/home/ubuntu').free>50*2**30
    torch.set_num_threads(4)
    evaluator=auto_select_evaluator('math')
    execution=provenance(cfg,[root/'plan.json', Path(inspect.getfile(MathParser)),
        Path(inspect.getfile(type(evaluator))),Path(inspect.getfile(type(evaluator._matcher)))])
    freeze(root/'execution_plan.json',execution)
    cases=json.loads((root/'cases.json').read_text());dev=[r for r in cases if r['split']=='dev']
    with np.load(root/'directions.npz') as z: directions={k:z[k].copy() for k in ['hss','random']}
    np.testing.assert_allclose(np.linalg.norm(directions['hss']),np.linalg.norm(directions['random']),rtol=1e-6)
    model,tok=load_model(cfg);generation=copy.deepcopy(model.generation_config)
    generation.do_sample=False;generation.num_beams=1;generation.repetition_penalty=1.
    if generation.pad_token_id is None:generation.pad_token_id=tok.pad_token_id or tok.eos_token_id
    eos=generation.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    freeze(root/'generation_config.json',generation.to_dict())
    start=time.time();deadline=start+cfg['max_gpu_hours']*3600
    zero=np.zeros_like(directions['hss'])
    if not (root/'smoke_SUCCESS.json').exists():
        smoke=[]
        for row in dev[:2]:
            natural,_=generate(model,tok,row,generation,cfg['layer'],None,32)
            identity,z=generate(model,tok,row,generation,cfg['layer'],zero,32)
            assert identity==natural and np.all(z['step_geometry'][:,1:3]==0)
            shifted,geom=generate(model,tok,row,generation,cfg['layer'],directions['hss']*.3,32)
            random,rg=generate(model,tok,row,generation,cfg['layer'],directions['random']*.3,32)
            assert shifted[0]==identity[0]==random[0]
            assert np.all(geom['step_geometry'][:,1]>0) and np.all(rg['step_geometry'][:,1]>0)
            smoke.append({'sample_id':row['sample_id'],'identity_exact':True,
                'first_token_unchanged':True,'identity_ids':identity,'shift_ids':shifted,'random_ids':random,
                'shift_forwarded_tokens':len(geom['step_geometry']),
                'mean_shift_norm':float(geom['step_geometry'][:,1].mean()),
                'mean_random_norm':float(rg['step_geometry'][:,1].mean()),
                'ideal_norm':float(np.linalg.norm(directions['hss']*.3))})
        write_json(root/'smoke_SUCCESS.json',{'checks':smoke,'seconds':time.time()-start,
            'torch':torch.__version__,'plan_sha256':plan_sha,
            'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    if smoke_only:return
    clockpath=root/'queue_clock.json'
    if clockpath.exists():deadline=json.loads(clockpath.read_text())['deadline_unix']
    else:freeze(clockpath,{'started_unix':start,'deadline_unix':deadline,'max_gpu_hours':cfg['max_gpu_hours']})
    completed=0;total=plan['expected_generations'];stage_start=time.time();new_tokens=0

    def one(row, condition, vector, alpha):
        nonlocal completed,new_tokens
        dest=root/'outputs'/row['sample_id'];dest.mkdir(parents=True,exist_ok=True)
        path=dest/(condition+'.json')
        if path.exists():
            r=read_record(path,plan_sha)
        else:
            if time.time()>deadline:raise TimeoutError('Six-hour queue budget reached; do not extend silently')
            if shutil.disk_usage('/home/ubuntu').free<50*2**30:raise RuntimeError('Disk reserve under50GiB')
            tick=time.time();ids,arrays=generate(model,tok,row,generation,cfg['layer'],vector,cfg['max_new_tokens'])
            assert ids and not eos.intersection(ids[:-1]) and len(ids)<=cfg['max_new_tokens']
            text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            score=evaluator.evaluate_sample(SimpleNamespace(sample_idx=int(row['sample_id'].split('_')[-1]),
                response_text=text,ground_truth=row['ground_truth'],meta={'sample_id':row['sample_id']}))
            assert score.is_correct is not None and not score.error
            write_npz(path.with_suffix('.npz'),**arrays)
            r={'sample_id':row['sample_id'],'split':row['split'],'condition':condition,'alpha':alpha,
                'generated_ids':ids,'response_text':text,'prompt_text':row['prompt_text'],
                'ground_truth':row['ground_truth'],'complete_boxed':boxed_answer(text) is not None,
                'boxed_answer':boxed_answer(text),'correct':bool(score.is_correct),
                'parsed_answer':score.extracted_answer,'parse_failed':bool(score.meta['parse_failed']),
                'n_tokens':len(ids),'finish_reason':'eos' if ids[-1] in eos else 'length',
                'shifted_forwarded_tokens':len(arrays['step_geometry']),
                'ideal_shift_norm':float(np.linalg.norm(vector)),
                'mean_actual_shift_norm':float(arrays['step_geometry'][:,1].mean()) if len(ids)>1 else 0.,
                'fresh_cache':True,'prompt_unmodified':True,'seconds':time.time()-tick,
                'geometry_sha256':sha(path.with_suffix('.npz')),'plan_sha256':plan_sha,
                'execution_plan_sha256':sha(root/'execution_plan.json'),
                'generation_config_sha256':sha(root/'generation_config.json')}
            write_json(path,r);write_json(path.with_suffix('.receipt.json'),{'record_sha256':sha(path)})
            new_tokens+=len(ids)
        assert r['condition']==condition and r['split']==row['split'] and r['alpha']==alpha
        completed+=1
        status={'state':'running','completed':completed,'expected':total,'last':row['sample_id']+'/'+condition,
                'split':row['split'],'elapsed_seconds':time.time()-stage_start,
                'new_tokens':new_tokens,'updated_unix':time.time(),'pid':os.getpid(),
                'deadline_unix':deadline,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        write_json(root/'status.json',status);print(json.dumps(status),flush=True)
        return r

    dev_records=[]
    for row in dev:
        dev_records.append(one(row,'zero',zero,0.))
        for a in cfg['alphas']:dev_records.append(one(row,f'hss_{a:g}',directions['hss']*a,a))
    selection=select_alpha(dev_records,cfg['alphas'])
    selection['dev_record_hashes']={str(p.relative_to(root)):sha(p)
        for row in dev for p in (root/'outputs'/row['sample_id']).glob('*.json') if not p.name.endswith('.receipt.json')}
    freeze(root/'selection.json',selection)
    alpha=selection['alpha']
    # Normal and sink cohorts are interleaved; each question gets all three arms.
    tests=[r for r in cases if r['split']!='dev']
    tests.sort(key=lambda r:__import__('hashlib').sha256(r['sample_id'].encode()).hexdigest())
    for row in tests:
        for cond,vector,a in [('zero',zero,0.),('hss',directions['hss']*alpha,alpha),
                              ('random',directions['random']*alpha,alpha)]:one(row,cond,vector,a)
    result=summarize(root,plan,selection)
    write_json(root/'status.json',{'state':'complete','completed':completed,'expected':total,
        'seconds':time.time()-stage_start,'summary_sha256':sha(root/'summary.json'),
        'predeclared_success':result['predeclared_success'],'updated_unix':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    try:run(a.root,a.smoke_only)
    except BaseException:
        write_json(Path(a.root)/'failure.json',{'traceback':traceback.format_exc(),'updated_unix':time.time()})
        raise

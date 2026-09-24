"""Single-GPU v2 batches using the unchanged protected fresh-cache executor."""
import argparse, copy, fcntl, inspect, json, shutil, subprocess, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from openact_eval.evaluators.registry import auto_select_evaluator
from evaluate_revision_completion import continuation, verify_existing
from extract_revision_prefixes import load_model
from revision_common import freeze, provenance, sha, write_json, write_npz, status
from projection_v2_common import project, geometry


def batch_spec(plan, tier, batch):
    if batch=='transfer':
        return 'transfer',tier['transfer_n'],[c for c in plan['transfer_conditions'] if c['name'] in tier['transfer_conditions']]
    c=next(c for c in plan['conditions'] if c['name']==batch)
    assert batch in tier['validation_ids'] and c['asset_available']
    conditions=[c]
    if c['prefix']!=16:conditions.insert(0,dict(c,name='baseline',operator='baseline'))
    return 'validation',32,conditions


def run(root,batch):
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];plan_sha=sha(root/'plan.json')
    assert json.loads((root/'prepare_SUCCESS.json').read_text())['plan_sha256']==plan_sha
    assert not (root/f'{batch}_SUCCESS.json').exists(),'Completed batch must not be rerun'
    for p,h in plan['files'].items():assert sha(p)==h,p
    main=Path(cfg['protected_primary'])
    assert json.loads((main/'queue_status.json').read_text())['state']=='complete'
    assert json.loads((main/'steering_statistics_audit.json').read_text())['complete']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    tier=json.loads((root/'tier.json').read_text());cohort,n,conditions=batch_spec(plan,tier,batch)
    rows=[r for r in json.loads((root/'inputs.json').read_text()) if r['cohort']==cohort][:n]
    assert len(rows)==n
    evaluator=auto_select_evaluator('gsm8k' if cohort=='transfer' else 'math')
    source=[Path(__file__),Path(inspect.getfile(type(evaluator))),root/'plan.json',root/'tier.json']
    freeze(root/(batch+'_execution.json'),provenance(cfg,source))
    model,tok=load_model(cfg)
    generation=copy.deepcopy(model.generation_config)
    generation.do_sample=False;generation.repetition_penalty=1.;generation.num_beams=1;generation.max_new_tokens=cfg['max_new_tokens']
    if generation.pad_token_id is None:generation.pad_token_id=tok.pad_token_id or tok.eos_token_id
    assert generation.to_dict()==json.loads((main/'generation_config.json').read_text())['resolved']
    freeze(root/'generation_config.json',json.loads((main/'generation_config.json').read_text()))
    eos=generation.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    with np.load(cfg['decoder']) as f:decoder={k:f[k].copy() for k in ['centers','local_basis','shared_basis']}
    counts=json.loads((main/'support.json').read_text())['question_count']
    reuse=json.loads((root/'reuse.json').read_text());records={};started=time.time();done=0
    for row in rows:
        sid=row['sample_id'];dest=root/'outputs'/batch/sid;dest.mkdir(parents=True,exist_ok=True)
        prefixn=conditions[0]['prefix'];prefix_path=dest/'prefix.json'
        if not prefix_path.exists():
            if cohort=='validation' and prefixn==16:
                source_prefix=Path(reuse[sid]['baseline']).parent/'prefix.json'
                old=json.loads(source_prefix.read_text());prefix=old['generated_prefix'];prefix_seconds=0.
                origin=dict(path=str(source_prefix),sha256=sha(source_prefix))
            else:
                tick=time.monotonic();prefix,_=continuation(model,tok,row['prompt_ids'],prefixn,generation)
                prefix_seconds=time.monotonic()-tick;origin=None
            assert prefix and len(prefix)<=prefixn
            freeze(prefix_path,dict(sample_id=sid,plan_sha256=plan_sha,prompt_ids=row['prompt_ids'],generated_prefix=prefix,
                active=len(prefix)==prefixn and not eos.intersection(prefix),seconds=prefix_seconds,reused_source=origin))
        pre=json.loads(prefix_path.read_text());assert pre['plan_sha256']==plan_sha and pre['prompt_ids']==row['prompt_ids']
        full_prefix=row['prompt_ids']+pre['generated_prefix']
        for c in conditions:
            if time.time()>plan['deadline_unix']-2*3600:raise TimeoutError('Preserve final two-hour audit/report reserve; incomplete batch is not a result')
            if shutil.disk_usage('/home/ubuntu').free<50*2**30:raise RuntimeError('SSD reserve under50GiB')
            path=dest/(c['name']+'.json')
            if path.exists():r=verify_existing(path,plan_sha)
            else:
                tick=time.monotonic();saved={}
                positions=list(range(len(full_prefix)-c['width'],len(full_prefix))) if pre['active'] else []
                if pre['active']:
                    suffix,saved=continuation(model,tok,full_prefix,cfg['max_new_tokens']-prefixn,generation,
                        cfg['layer'],positions,lambda x:project(x,decoder,c,plan['permutation'],cfg['near_zero_norm']))
                else:suffix=[]
                ids=pre['generated_prefix']+suffix
                assert ids and not eos.intersection(ids[:-1])
                text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                score=evaluator.evaluate_sample(SimpleNamespace(sample_idx=row['source_sample_idx'],response_text=text,
                    ground_truth=row['ground_truth'],meta={'sample_id':sid}))
                assert score.is_correct is not None and not score.error
                files={'prefix.json':sha(prefix_path)};g=dict(energy=0.,modified_tokens=0,region_retained=[],offspace_fraction=[],relative_change=[])
                if saved:
                    arr=path.with_suffix('.npz');write_npz(arr,before=saved['before'],ideal=saved['ideal'],actual=saved['actual'])
                    files[arr.name]=sha(arr);g=geometry(saved['before'],saved['actual'],decoder,counts)
                    g['operator_metadata']=saved['meta']
                    # Exact compatibility against the protected baseline at shared positions.
                    if cohort=='validation' and prefixn==16:
                        with np.load(Path(reuse[sid]['baseline']).with_suffix('.npz')) as f:
                            overlap=min(4,c['width']);np.testing.assert_array_equal(saved['before'][-overlap:],f['before'][-overlap:])
                r=dict(sample_id=sid,split=row['split'],cohort=cohort,dataset=row['dataset'],batch=batch,condition=c,
                    prompt_text=row['prompt_text'],ground_truth=row['ground_truth'],generated_ids=ids,response_text=text,
                    correct=bool(score.is_correct),parsed_answer=score.extracted_answer,normalized_answer=score.normalized_answer,
                    parse_failed=bool(score.meta['parse_failed']),length=len(ids),finish_reason='eos' if ids[-1] in eos else 'length',
                    prefix_active=pre['active'],positions=positions,hook_calls=int(pre['active']),hook_removed_after_prefill=True,
                    fresh_cache=True,geometry=g,seconds=time.monotonic()-tick,files=files,plan_sha256=plan_sha,
                    execution_plan_sha256=sha(root/(batch+'_execution.json')),generation_config_sha256=sha(root/'generation_config.json'))
                write_json(path,r);write_json(path.with_suffix('.receipt.json'),dict(record_sha256=sha(path)))
            assert r['condition']==c
            records[str(path.relative_to(root))]=sha(path);done+=1
            status(root,'generation',state='running',batch=batch,completed=done,expected=len(rows)*len(conditions),seconds=time.time()-started)
    write_json(root/f'{batch}_SUCCESS.json',dict(complete=True,batch=batch,questions=len(rows),conditions=conditions,
        records=records,seconds=time.time()-started,plan_sha256=plan_sha,tier_sha256=sha(root/'tier.json'),
        execution_plan_sha256=sha(root/(batch+'_execution.json')),generation_config_sha256=sha(root/'generation_config.json')))
    status(root,'generation',state='complete',batch=batch,completed=done,expected=done,seconds=time.time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',required=True);a=p.parse_args()
    try:run(a.root,a.batch)
    except BaseException:status(a.root,'generation',state='failed',batch=a.batch,traceback=traceback.format_exc());raise

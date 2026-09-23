"""Freeze new GSM8K train questions, generate references and capture real prefixes.

This stage performs no fitting or scientific intervention analysis. It preserves
short/truncated responses and freezes questions before any model outcome exists.
Run in OpenAct's pinned environment, with only one GPU model process.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status


def normalized(value):
    return ' '.join(value.split())


def select(candidates,old_prompts,n,salt):
    seen=set();eligible=[];excluded=[]
    for row in candidates:
        question=normalized(row['question']);prompt=normalized(row['prompt_text'])
        if question in seen or prompt in old_prompts:
            excluded.append(row['sample_id']);continue
        seen.add(question)
        order=hashlib.sha256((salt+'\0'+question).encode()).hexdigest()
        eligible.append((order,row['sample_id'],row))
    chosen=[row for _,_,row in sorted(eligible)[:n]]
    assert len(chosen)==n
    return chosen,excluded


def prepare(cfg):
    from datasets import load_dataset
    from openact_core.tasks.templates import get_template
    from transformers import AutoTokenizer
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    old=root/'collection_plan.json'
    if old.exists():
        plan=json.loads(old.read_text());assert plan['config']==cfg
        for name,digest in plan['files'].items():assert sha(name)==digest,name
        return plan
    template=get_template('math','zot');assert template.hash==cfg['template_hash']
    dataset=load_dataset(cfg['dataset'],cfg['dataset_config'],split=cfg['dataset_split'],revision=cfg['dataset_revision'])
    assert len(dataset)==cfg['dataset_expected_rows']
    foundation=Path(cfg['source_foundation'])/'prefixes'
    metadata=[p.parent/'rows.parquet' for p in sorted(foundation.glob('shard_*/_SUCCESS.json'))]
    metadata.append(Path(cfg['previous_gsm8k']))
    old_prompts=set()
    for path in metadata:
        old_prompts.update(pd.read_parquet(path,columns=['prompt_text']).prompt_text.map(normalized))
    candidates=[]
    for i,row in enumerate(dataset):
        assert '####' in row['answer']
        candidates.append({'sample_id':f'gsm8k_train_{i}','dataset_index':i,'question':row['question'],
            'ground_truth':row['answer'].rsplit('####',1)[1].strip(),'dataset_answer':row['answer'],
            'prompt_text':template.format(problem=row['question'])})
    selected,excluded=select(candidates,old_prompts,cfg['questions'],cfg['selection_salt'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    for row in selected:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':row['prompt_text']}],tokenize=False,add_generation_prompt=True)
        row.update(model_input_text=rendered,prompt_ids=tokenizer(rendered,add_special_tokens=False).input_ids)
    source=Path(cfg['source_locality'])/'decoders/p16_l14_tokens'
    receipt=json.loads((source/'_SUCCESS.json').read_text())
    assert sha(source/'decoder.npz')==receipt['decoder_sha256']
    files=[Path(__file__),Path(__file__).with_name('revision_common.py'),
        Path(__file__).with_name('extract_revision_prefixes.py'),source/'decoder.npz',source/'_SUCCESS.json',
        Path(inspect.getfile(get_template)),*metadata]
    plan=provenance(cfg,files)
    plan.update(selected=selected,duplicate_or_prior_exact_prompt_exclusions=excluded,
        dataset_fingerprint=dataset._fingerprint,template=template.to_dict(),
        tokenizer_chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        frozen_decoder_sha256=receipt['decoder_sha256'],previous_prompt_count=len(old_prompts),
        question_selection='hash of normalized question; no output, label, difficulty or fitted score used')
    freeze(old,plan)
    status(root,'collection',state='prepared',completed=0,expected=len(selected))
    return plan


def run(cfg):
    import torch
    from extract_revision_prefixes import load_model,capture
    plan=prepare(cfg);root=Path(cfg['output']);started=time.monotonic()
    lock=(root/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'collection_SUCCESS.json').exists():
        audit(cfg);return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU process is running; do not start a second model')
    if shutil.disk_usage('/home/ubuntu').free<50*1024**3:raise RuntimeError('SSD reserve below50GiB')
    model,tokenizer=load_model(cfg);torch.manual_seed(42)
    generation=model.generation_config.to_dict();generation.update(cfg['generation'])
    freeze(root/'generation_config.json',generation)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    for number,row in enumerate(plan['selected'],1):
        dest=root/'collection';path=dest/(row['sample_id']+'.json');array=path.with_suffix('.npz')
        if path.exists():
            saved=json.loads(path.read_text());assert saved['source']==row
            assert sha(array)==saved['activations_sha256']
            continue
        prompt=row['prompt_ids']
        assert tokenizer(row['model_input_text'],add_special_tokens=False).input_ids==prompt
        with torch.inference_mode():
            result=model.generate(torch.tensor([prompt],device=model.device),**cfg['generation'],
                pad_token_id=model.generation_config.pad_token_id)
        assert result[0,:len(prompt)].tolist()==prompt
        response=result[0,len(prompt):].tolist();prefixn=cfg['prefix_tokens']
        valid=len(response)>prefixn and not eos.intersection(response[:prefixn])
        x=np.empty((0,model.config.hidden_size),dtype=np.float32)
        if valid:
            ids=prompt+response[:prefixn];positions=list(range(len(ids)-cfg['window'],len(ids)))
            x=capture(model,ids,[cfg['layer']],{'window':positions},positions)['window'][0]
            assert x.shape==(cfg['window'],model.config.hidden_size) and np.isfinite(x).all()
        write_npz(array,x=x)
        write_json(path,{'source':row,'response_ids':response,'response_text':tokenizer.decode(response,skip_special_tokens=True),
            'finish_reason':'eos' if response and response[-1] in eos else 'length','prefix_valid':valid,
            'layer':cfg['layer'],'prefix_tokens':prefixn,'activations_sha256':sha(array),
            'collection_plan_sha256':sha(root/'collection_plan.json')})
        status(root,'collection',state='running',completed=number,expected=len(plan['selected']),seconds=time.monotonic()-started)
    paths=sorted((root/'collection').glob('*.json'))
    assert {p.stem for p in paths}=={r['sample_id'] for r in plan['selected']}
    write_json(root/'collection_SUCCESS.json',{'questions':len(paths),'seconds':time.monotonic()-started,
        'plan_sha256':sha(root/'collection_plan.json'),'records':{p.name:sha(p) for p in paths}})
    status(root,'collection',state='complete',completed=len(paths),expected=len(paths),seconds=time.monotonic()-started)


def audit(cfg):
    from transformers import AutoTokenizer,GenerationConfig
    root=Path(cfg['output']);plan=prepare(cfg)
    receipt=json.loads((root/'collection_SUCCESS.json').read_text())
    assert receipt['plan_sha256']==sha(root/'collection_plan.json')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    generation=GenerationConfig.from_pretrained(cfg['model'],revision=cfg['revision']).to_dict()
    generation.update(cfg['generation'])
    assert generation==json.loads((root/'generation_config.json').read_text())
    eos=generation['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    short=[];truncated=[]
    for row in plan['selected']:
        path=root/'collection'/(row['sample_id']+'.json');r=json.loads(path.read_text())
        assert sha(path)==receipt['records'][path.name] and r['source']==row
        assert r['collection_plan_sha256']==receipt['plan_sha256']
        rendered=tokenizer.apply_chat_template([{'role':'user','content':row['prompt_text']}],tokenize=False,add_generation_prompt=True)
        assert rendered==row['model_input_text'] and tokenizer(rendered,add_special_tokens=False).input_ids==row['prompt_ids']
        ids=r['response_ids'];assert ids and len(ids)<=cfg['generation']['max_new_tokens']
        assert not eos.intersection(ids[:-1])
        assert tokenizer.decode(ids,skip_special_tokens=True)==r['response_text']
        finish='eos' if ids[-1] in eos else 'length';assert finish==r['finish_reason']
        if finish=='length':
            assert len(ids)==cfg['generation']['max_new_tokens'];truncated.append(row['sample_id'])
        valid=len(ids)>cfg['prefix_tokens'] and not eos.intersection(ids[:cfg['prefix_tokens']])
        assert valid==r['prefix_valid']
        if not valid:short.append(row['sample_id'])
        assert sha(path.with_suffix('.npz'))==r['activations_sha256']
        with np.load(path.with_suffix('.npz')) as f:
            assert f['x'].shape==((16,3584) if valid else (0,3584)) and np.isfinite(f['x']).all()
    out={'complete':True,'questions':len(plan['selected']),'prefix_unavailable':short,'truncated':truncated,
        'receipt_sha256':sha(root/'collection_SUCCESS.json'),'audit_code_sha256':sha(Path(__file__)),
        'scope':'Question/source/tokenizer/IDs/decode/EOS/length/capture-file integrity. Prefix values must also equal the fresh intervention hook in the subsequent functional execution audit. No scientific intervention outcome or correctness result yet.'}
    write_json(root/'collection_audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--audit',action='store_true')
    a=p.parse_args();cfg=config(a.config)
    try:
        if a.prepare_only:prepare(cfg)
        elif a.audit:audit(cfg)
        else:run(cfg)
    except BaseException:
        status(cfg['output'],'collection',state='failed',traceback=traceback.format_exc());raise

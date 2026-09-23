"""Independently validate every saved task, output, and train-only diagnostic count."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig
from revision_common import sha, write_json
from revision_counterfactual_gate import memory_split, parse
from revision_memory_capability import ROOT, encode_task, tasks


def run(root):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    for path,digest in plan['files'].items():
        assert sha(path)==digest, path
    assert cfg['tasks']==tasks() and cfg['repetition_penalty']==1.0 and cfg['do_sample'] is False
    marker=json.loads((root/'_SUCCESS.json').read_text())
    assert marker['conditions']==360 and marker['summary_sha256']==sha(root/'summary.json')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    gen=GenerationConfig.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    eos=set(gen.eos_token_id if isinstance(gen.eos_token_id,list) else [gen.eos_token_id])
    expected={t['id']:t for t in cfg['tasks']};paths=sorted((root/'samples').glob('*.json'))
    assert len(paths)==360 and {p.stem for p in paths}==set(expected)
    groups={};hashes={};errors=[]
    for path in paths:
        r=json.loads(path.read_text());t=r['task'];o=r['output']
        assert t==expected[path.stem]
        assert r['input_ids']==encode_task(tokenizer,t['text'])
        if '/' in t['group']:
            assert t['split']=='train' and memory_split(t['a'],t['b'])=='train'
            assert t['u']==(t['a']+t['b'])%10
        gold=t['u'] if t['group']=='delayed_copy' else (t['u']+t['c'])%10
        assert t['gold_digit']==gold and t['gold_tag'] in ('red','blue')
        ids=o['generated_ids']
        assert 0<len(ids)<=cfg['max_new_tokens'] and not eos.intersection(ids[:-1])
        assert len(ids)==cfg['max_new_tokens'] or ids[-1] in eos
        assert o['text']==tokenizer.decode(ids,skip_special_tokens=True)
        digit,tag=parse(o['text'])
        assert (digit,tag)==(o['digit'],o['tag']) and o['parse_failed']==(digit is None)
        correct=digit==gold and tag==t['gold_tag']
        assert r['correct']==correct and o['actual_patch_energy'] is None
        unique=groups.setdefault(t['group'],{})
        if t['text'] in unique:
            assert unique[t['text']]['output']['generated_ids']==ids
        unique[t['text']]=r
        if not correct:
            errors.append({'id':t['id'],'group':t['group'],'prompt':t['text'],
                'gold':f'{gold}|{t["gold_tag"]}','actual':o['text'],
                'returns_u_without_offset':digit==t['u'] and t.get('c',0)!=0})
        hashes[str(path)]=sha(path)
    summary=[]
    for group,unique in sorted(groups.items()):
        n=len(unique);correct=sum(r['correct'] for r in unique.values())
        summary.append({'group':group,'scheduled_cases':sum(t['group']==group for t in cfg['tasks']),
            'unique_prompts':n,'correct':correct,'accuracy':correct/n,
            'parse_failures':sum(r['output']['parse_failed'] for r in unique.values()),
            'meets_training_capability_threshold':correct/n>=.9})
    assert summary==json.loads((root/'summary.json').read_text())
    result={'conditions':360,'groups':summary,'plan_sha256':sha(root/'plan.json'),
        'task_definitions_train_split_tokenization_decode_gold_EOS_summary_verified':True,
        'audit_code_sha256':sha(Path(__file__)),
        'interpretation':'Train-only capability diagnosis. Passing a group does not validate a held-out task or establish a causal variable.'}
    write_json(root/'audit.json',result);write_json(root/'audit_input_hashes.json',hashes)
    write_json(root/'errors.json',errors)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    run(parser.parse_args().root)

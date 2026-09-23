"""Freeze all prospective notation cases using the real tokenizer on CPU only."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from revision_common import freeze, provenance, sha, write_json
from revision_index_interchange_common import case_pairs,prompt,assistant_prefix


def encode(tokenizer,item):
    text=prompt(item)
    rendered=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    prompt_ids=tokenizer(rendered,add_special_tokens=False).input_ids
    prefix=assistant_prefix(item)
    ids=tokenizer(prefix,add_special_tokens=False).input_ids
    full_ids=tokenizer(rendered+prefix,add_special_tokens=False).input_ids
    assert full_ids==prompt_ids+ids, 'Tokenization changed at assistant prefix boundary'
    fragments=tokenizer.batch_decode([[i] for i in ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    signature=[re.sub(r'\d','N',re.sub(r'[a-z]','V',s)) for s in fragments]
    return {'item':item,'prompt':text,'rendered_prompt':rendered,'prompt_ids':prompt_ids,
            'assistant_prefix':prefix,'prefix_ids':ids,'input_ids':full_ids,
            'prefix_fragments':fragments,'prefix_signature':signature,
            'positions16':list(range(len(full_ids)-16,len(full_ids)))}


def run(config_path):
    from transformers import AutoTokenizer
    cfg=json.loads(config_path.read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    decoder=Path(cfg['decoder']);assert sha(decoder)==cfg['decoder_sha256']
    with np.load(decoder,allow_pickle=False) as f:
        assert f['centers'].shape==(64,3584)
        assert f['local_basis'].shape[0]==64 and f['local_basis'].shape[1]>=8
        assert f['shared_basis'].shape[0]>=8
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    records=[]
    for case in case_pairs():
        r,d=[encode(tokenizer,case[s]) for s in ['recipient','donor']]
        aligned=(len(r['prefix_ids'])>=16 and len(r['prefix_ids'])==len(d['prefix_ids'])
                 and r['prefix_signature']==d['prefix_signature'])
        records.append({'case':case,'recipient':r,'donor':d,'aligned':bool(aligned),
            'alignment_reason':None if aligned else 'prefix length or token-shape mismatch; no replacement'})
    files=[config_path,Path(__file__),Path(__file__).with_name('revision_index_interchange_common.py'),decoder]
    freeze(root/'plan.json',provenance(cfg,files)|{'cases':records,'no_gpu_calls':True})
    result={'complete':True,'candidate_pairs':len(records),'aligned_pairs':sum(r['aligned'] for r in records),
            'validation_pairs':sum(r['case']['split']=='validation' for r in records),
            'test_pairs':sum(r['case']['split']=='test' for r in records),
            'prefix_token_lengths':sorted({len(r[s]['prefix_ids']) for r in records for s in ['recipient','donor']}),
            'plan_sha256':sha(root/'plan.json'),'functional_results_exist':False}
    write_json(root/'prepare_SUCCESS.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);run(ap.parse_args().config)

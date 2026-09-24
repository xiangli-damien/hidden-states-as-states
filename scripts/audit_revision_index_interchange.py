"""Independent CPU decoder, scoring, pairing and actual bf16 patch audit."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoTokenizer
from revision_common import sha,write_json


def terms(text):
    pieces=text.strip().split(',')
    found=[]
    for p in pieces:
        m=re.fullmatch(r'([a-z])_(\d+)',p.strip())
        if not m:return None
        found.append((m[1],int(m[2])))
    return found


def expected(item):
    return [(item['symbol'],item['start']+i) for i in range(7)]


def scores(suffix,item):
    r,d=item['case']['recipient'],item['case']['donor']
    actual=terms(item['recipient']['assistant_prefix']+suffix)
    valid=actual is not None and len(actual)==7 and actual[:4]==expected(r)[:4]
    target=bool(valid and [x[1] for x in actual[4:]]==[d['start']+4,d['start']+5,d['start']+6])
    preserved=bool(valid and [x[0] for x in actual[5:]]==[r['symbol'],r['symbol']])
    return {'format_valid':bool(valid),'target_counter_sequence':target,'new_symbols_preserved':preserved,
            'joint_success':target and preserved,'ordinary_recipient_correct':actual==expected(r)}


def run(root,stage):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];dest=root/stage
    receipt=json.loads((dest/'_SUCCESS.json').read_text());assert receipt['complete']
    assert receipt['plan_sha256']==sha(root/'plan.json')
    assert receipt['evaluation_plan_sha256']==sha(root/'evaluation_plan.json')
    assert receipt['generation_config_sha256']==sha(root/'generation_config.json')
    for filename,digest in receipt['files'].items():assert sha(dest/filename)==digest
    ep=json.loads((root/'evaluation_plan.json').read_text())
    for filename,digest in ep['files'].items():assert sha(Path(filename))==digest
    assert sha(Path(cfg['decoder']))==cfg['decoder_sha256']
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    gen=json.loads((root/'generation_config.json').read_text());eos=gen['model_defaults']['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    def decode(output):
        ids=output['generated_ids'];assert 1<=len(ids)<=64
        assert not set(ids[:-1]).intersection(eos)
        assert output['finish_reason']==('eos' if ids[-1] in eos else 'length')
        assert ids[-1] in eos or len(ids)==64
        text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert text==output['text'];return text
    with np.load(cfg['decoder'],allow_pickle=False) as f:
        centers=f['centers'].astype(np.float64);local=f['local_basis'][:,:8].astype(np.float64);shared=f['shared_basis'][:8].astype(np.float64)
    wanted={r['case']['id']:r for r in plan['cases'] if r['case']['split']==stage}
    assert {p.stem for p in (dest/'cases').glob('*.json')}==set(wanted)
    free=prefixed=eligible=patches=0;records=[]
    for cid,item in wanted.items():
        r=json.loads((dest/'cases'/f'{cid}.json').read_text());records.append(r)
        assert r['case']==item['case'] and r['case_id']==cid and r['aligned']==item['aligned']
        ok=True
        for side in ['recipient','donor']:
            a,b=r[side]['free'],r[side]['prefixed'];assert a['patch_calls']==b['patch_calls']==0
            f=terms(decode(a))==expected(item['case'][side])
            p=terms(item[side]['assistant_prefix']+decode(b))==expected(item['case'][side])
            assert f==r[side]['free_correct'] and p==r[side]['prefixed_correct']
            free+=f;prefixed+=p;ok=ok and f and p
        ok=bool(ok and item['aligned']);assert ok==r['eligible'];eligible+=ok
        wanted_patches={(w,m) for w in [16,4] for m in ['identity','full_donor','local8','shared8']} if ok and stage=='test' else set()
        assert {(p['width'],p['method']) for p in r['patches']}==wanted_patches
        assert len(r['patches'])==len(wanted_patches)
        if not wanted_patches:continue
        cap=dest/r['capture_file'];assert sha(cap)==r['capture_sha256']
        with np.load(cap,allow_pickle=False) as f:x=f['recipient'].astype(np.float64);y=f['donor'].astype(np.float64)
        assert x.shape==y.shape==(16,3584)
        for p in r['patches']:
            w=p['width'];method=p['method'];before=x[-w:];donor=y[-w:]
            assert p['positions']==item['recipient']['positions16'][-w:]
            assert p['donor_positions']==item['donor']['positions16'][-w:]
            assert min(p['positions'])>=len(item['recipient']['prompt_ids'])
            if method=='identity':z=before
            elif method=='full_donor':z=donor
            else:
                labels=np.square(before[:,None,:]-centers[None,:,:]).sum(2).argmin(1)
                matrices=local[labels] if method=='local8' else np.broadcast_to(shared,(w,*shared.shape))
                coordinates=np.einsum('td,trd->tr',donor-before,matrices)
                z=before+np.einsum('tr,trd->td',coordinates,matrices)
            expected_patch=torch.tensor(z,dtype=torch.bfloat16).float().numpy()
            path=dest/p['array_file'];assert sha(path)==p['array_sha256']
            with np.load(path,allow_pickle=False) as f:actual=f['actual']
            np.testing.assert_array_equal(actual,expected_patch)
            def labels_of(v):return np.square(v[:,None,:]-centers[None,:,:]).sum(2).argmin(1).tolist()
            geometry=p['geometry']
            assert labels_of(before)==geometry['recipient_regions'] and labels_of(donor)==geometry['donor_regions']
            assert labels_of(actual.astype(np.float64))==geometry['actual_regions']
            np.testing.assert_allclose(geometry['ideal_squared_displacement'],np.square(z-before).sum(),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(geometry['actual_squared_displacement_float64'],np.square(actual.astype(np.float64)-before).sum(),rtol=1e-12,atol=1e-10)
            energy=float(torch.from_numpy(actual-before.astype(np.float32)).square().sum())
            np.testing.assert_allclose(energy,p['output']['actual_patch_energy'],rtol=2e-6,atol=1e-7)
            assert p['output']['patch_calls']==1
            text=decode(p['output']);assert scores(text,item)==p['score']
            if method=='identity':assert p['output']['generated_ids']==r['recipient']['prefixed']['generated_ids']
            patches+=1
    assert patches==receipt['patch_conditions'] and eligible==receipt['eligible_pairs'] and len(wanted)==receipt['pairs']
    result={'passed':True,'stage':stage,'pairs':len(wanted),'eligible_pairs':eligible,'patch_conditions':patches,
        'free_correct':free,'prefixed_correct':prefixed,'stage_receipt_sha256':sha(dest/'_SUCCESS.json'),
        'plan_sha256':sha(root/'plan.json')}
    if stage=='validation':
        gate={'pairs':16,'clean_contexts':32,'free_correct':free,'prefixed_correct':prefixed,'eligible_pairs':eligible,
              'passed':free>=29 and prefixed>=29 and eligible>=12}
        assert gate==receipt['capability_gate'];result['capability_gate']=gate
    write_json(root/f'{stage}_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--stage',required=True,choices=['validation','test']);a=ap.parse_args();run(a.root,a.stage)

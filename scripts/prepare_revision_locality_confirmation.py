"""Adapt audited target captures to the existing, unchanged locality executor.

No fitting: copy the exact MATH decoder and frozen MATH-only rank choice.
The compatibility fit_SUCCESS marker explicitly records that no target fit ran.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status


def evaluation_config(cfg):
    assert cfg['layer']==14 and cfg['prefix_tokens']==16 and cfg['window']==16
    root=Path(cfg['output'])
    return {'output':str(root/'interventions'),'foundation':str(root/'prepared'),
        'model':cfg['model'],'revision':cfg['revision'],'views':[[16,14,'tokens']],
        'ranks':[8],'shared_ranks':[8,64,512],'seeds':[42,137,271],'alphas':[.25,.5,1.],
        'primary':cfg['primary'],'secondary':cfg['secondary'],
        'scope':cfg['scope'],'normalization':'none; exact frozen MATH decoder',
        'auxiliary_widths':'Existing executor also retains widths1/4 as explicitly secondary position-scope controls; primary width16 unchanged.',
        'split_semantics':'All rows named test for storage compatibility; all64 are new project-held-out GSM8K train confirmation questions. No target validation or training.'}


def run(cfg):
    base=Path(cfg['output']);root=base/'interventions'
    if (root/'plan.json').exists():
        old=json.loads((root/'plan.json').read_text());assert old['config']==evaluation_config(cfg)
        for name,digest in old['files'].items():assert sha(name)==digest,name
        assert (root/'fit_SUCCESS.json').exists()
        print('Existing immutable adapter inputs verified.');return
    audit=json.loads((base/'collection_audit.json').read_text());assert audit['complete'] and audit['questions']==cfg['questions']
    receipt=json.loads((base/'collection_SUCCESS.json').read_text())
    assert audit['receipt_sha256']==sha(base/'collection_SUCCESS.json')
    collection=json.loads((base/'collection_plan.json').read_text());assert collection['config']==cfg
    assert receipt['plan_sha256']==sha(base/'collection_plan.json')
    for path,digest in collection['files'].items():assert sha(path)==digest,path
    source=Path(cfg['source_locality']);original=source/'decoders/p16_l14_tokens/decoder.npz'
    assert sha(original)==collection['frozen_decoder_sha256']
    match=source/'report/validation_mse_match.json'
    choice=next(r for r in json.loads(match.read_text()) if r['local_rank']==8)
    assert choice['selected_shared_rank']==64 and choice['within_predeclared_10pct_validation_gap']
    rows=[];tokens=[];vectors=[];valid_ids=[];files=[]
    for selected in collection['selected']:
        sid=selected['sample_id'];path=base/'collection'/(sid+'.json');r=json.loads(path.read_text())
        assert sha(path)==receipt['records'][path.name] and r['source']==selected
        assert sha(path.with_suffix('.npz'))==r['activations_sha256']
        files.extend([path,path.with_suffix('.npz')])
        rows.append({'sample_id':sid,'split':'test','label':'本轮未评分','prompt_text':selected['prompt_text'],
            'ground_truth':selected['ground_truth'],'response_text':r['response_text'],
            'finish_reason':r['finish_reason'],'prefix_valid':r['prefix_valid']})
        tokens.append({'sample_id':sid,'prompt_ids':selected['prompt_ids'],'response_ids':r['response_ids'],
            'question_positions':[]})  # This protocol never requests question-tail patches.
        if r['prefix_valid']:
            with np.load(path.with_suffix('.npz')) as f:vectors.append(f['x'].copy())
            valid_ids.append(sid)
    shard=base/'prepared/prefixes/shard_00000';shard.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard/'rows.parquet',index=False);write_json(shard/'tokens.json',tokens)
    write_json(shard/'_SUCCESS.json',{'rows':len(rows),'collection_audit_sha256':sha(base/'collection_audit.json'),
        'sha256':{name:sha(shard/name) for name in ['rows.parquet','tokens.json']}})
    folder=root/'decoders/p16_l14_tokens';folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,folder/'decoder.npz');assert sha(folder/'decoder.npz')==sha(original)
    write_npz(folder/'pilot_activations.npz',sample_ids=np.array(valid_ids),x=np.stack(vectors))
    write_json(folder/'audit.json',{'target_fit_performed':False,'source_decoder_sha256':sha(original),
        'target_questions':len(rows),'valid_prefix_questions':len(valid_ids),'ranks':[8],
        'shared_ranks':[8,64,512],'source_validation_selected_rank':64,'normalization':'none'})
    write_json(folder/'_SUCCESS.json',{'decoder_sha256':sha(folder/'decoder.npz'),
        'pilot_activations_sha256':sha(folder/'pilot_activations.npz'),'audit_sha256':sha(folder/'audit.json')})
    shutil.copyfile(match,root/'source_mse_match.json')
    evcfg=evaluation_config(cfg);freeze(root/'evaluation_config.json',evcfg)
    files.extend([Path(__file__),Path(__file__).with_name('collect_revision_locality_confirmation.py'),
        base/'collection_audit.json',base/'collection_plan.json',base/'collection_SUCCESS.json',
        original,match,root/'source_mse_match.json',shard/'rows.parquet',shard/'tokens.json',
        folder/'decoder.npz',folder/'pilot_activations.npz',folder/'audit.json'])
    plan=provenance(evcfg,files)
    plan.update(selected_sample_ids=[r['sample_id'] for r in collection['selected']],
        excluded_prior_intervention_ids=json.loads((source/'plan.json').read_text())['selected_sample_ids'],
        target_fit_performed=False,source_mse_choice_sha256=sha(match),source_locality=str(source),
        original_confirmation_protocol_sha256=sha(base/'collection_plan.json'))
    freeze(root/'plan.json',plan)
    write_json(root/'fit_SUCCESS.json',{'views':1,'target_fit_performed':False,
        'meaning':'Compatibility receipt for reused frozen source decoder; no target model fitting.',
        'plan_sha256':sha(root/'plan.json')})
    status(base,'adapter',state='complete',questions=len(rows),valid_prefix_questions=len(valid_ids),target_fit_performed=False)
    print(json.dumps({'questions':len(rows),'valid':len(valid_ids),'target_fit_performed':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(config(p.parse_args().config))

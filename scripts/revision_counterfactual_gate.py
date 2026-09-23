"""Validation-only capability/interchange gate on a controlled memory task.

This is a custom positive-control benchmark, not MIB/RAVEL, not MATH steering,
and not proof of an internal arithmetic variable. All patch positions precede
the recipient's offset/tag, so donor future inputs cannot enter those states.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch
from extract_revision_prefixes import load_model
from revision_common import OncePatch,write_json,write_npz,freeze,provenance,status,sha


ROOT=Path('/lambda/nfs/dami/hss/revision-counterfactual-gate-20260923')
BUFFER=(' Keep the remembered value available for the next step. There is no other calculation in this sentence. '
        'Do not give an answer yet. Carefully retain the remembered value while reading the later input.')


def memory_split(a,b):
    u=(a+b)%10
    pairs=sorted([(i,(u-i)%10) for i in range(10)],key=lambda t:hashlib.sha256(f'memory-pair-v1/{t[0]}/{t[1]}'.encode()).hexdigest())
    index=pairs.index((a,b))
    return 'train' if index<6 else 'validation' if index<8 else 'test'


def cases():
    pairs=[(a,b) for a in range(10) for b in range(10) if memory_split(a,b)=='validation']
    result=[]
    for mode in ('explicit_digit','implicit_sum'):
        for i,(a,b) in enumerate(pairs):
            u=(a+b)%10;c=(3*a+7*b+1)%10;tag='red' if i%2 else 'blue'
            for kind in ('same_u','different_u'):
                # Every recipient appears under BOTH donor types, so recipient
                # identity, offset, color and original task difficulty are paired.
                possible=[p for p in pairs if p!=(a,b) and (((sum(p)%10)==u)==(kind=='same_u'))]
                da,db=possible[i%len(possible)]
                recipient={'a':a,'b':b,'u':u,'c':c,'tag':tag}
                donor={'a':da,'b':db,'u':(da+db)%10,'c':(c+3)%10,'tag':'blue' if tag=='red' else 'red'}
                rid=f'{mode}_{a}_{b}'
                result.append({'id':rid+'_'+kind,'recipient_id':rid,'mode':mode,'pair_type':kind,'split':'validation',
                               'recipient':recipient,'donor':donor,'cf_digit':(donor['u']+c)%10})
    return result


def task_text(item,mode):
    if mode=='explicit_digit':
        prefix=f'This is a memory and calculation task. Remember this digit: {item["u"]}.'
    elif mode=='implicit_sum':
        prefix=(f'This is a memory and calculation task. Add {item["a"]} and {item["b"]}, '
                'keeping only the last digit. Remember that resulting digit.')
    else:
        raise ValueError(mode)
    suffix=(f' Later input: the offset is {item["c"]}. The tag is {item["tag"]}. '
            'Add the remembered digit and the offset, keeping only the last digit. '
            'Return exactly the resulting digit, a vertical bar, and the tag. '
            'For example, the output format is 7|red. Give no explanation.')
    return prefix+BUFFER+suffix


def encode(tokenizer,item,mode,scope='tail'):
    text=task_text(item,mode)
    full=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    start=full.index(BUFFER);end=start+len(BUFFER)
    positions=[i for i,(a,b) in enumerate(encoded.offset_mapping) if a>=start and b<=end and b>a]
    if len(positions)<16:
        raise ValueError('Memory buffer must contain at least 16 real tokens')
    if scope=='tail':
        positions=positions[-16:]
    elif scope=='memory_region':
        memory_start=full.index(text)
        positions=[i for i,(a,b) in enumerate(encoded.offset_mapping) if a>=memory_start and b<=end and b>a]
    else:raise ValueError(scope)
    return encoded.input_ids,positions,full


def parse(text):
    match=re.fullmatch(r'\s*([0-9])\s*\|\s*(red|blue)\s*',text)
    return (int(match[1]),match[2]) if match else (None,None)


@torch.inference_mode()
def capture(model,ids,positions,layers=(7,14,28)):
    values={};handles=[]
    for layer in layers:
        def hook(module,args,out,layer=layer):
            x=out[0] if isinstance(out,tuple) else out
            values[layer]=x[0,positions].float().cpu().numpy()
        handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    try:
        out=model(input_ids=torch.tensor([ids],device=model.device),use_cache=False,logits_to_keep=1)
        logits=out.logits[0,-1].float().cpu().numpy()
    finally:
        for handle in handles:handle.remove()
    return values,logits


@torch.inference_mode()
def generate(model,tokenizer,ids,module=None,positions=None,transform=None):
    patch=OncePatch(positions,len(ids),transform) if module is not None else None
    handle=module.register_forward_hook(patch) if patch is not None else None
    try:
        out=model.generate(torch.tensor([ids],device=model.device),do_sample=False,max_new_tokens=16,
            pad_token_id=model.generation_config.pad_token_id)
        generated=out[0,len(ids):].tolist();text=tokenizer.decode(generated,skip_special_tokens=True)
        digit,tag=parse(text)
        if patch is not None and patch.calls!=1:
            raise AssertionError('Patch was not applied exactly once')
        return {'generated_ids':generated,'text':text,'digit':digit,'tag':tag,'parse_failed':digit is None,
                'actual_patch_energy':None if patch is None else patch.energy}
    finally:
        if handle is not None:handle.remove()


def full_transform(values):
    def apply(h):return torch.from_numpy(values).to(device=h.device,dtype=h.dtype)
    return apply


def random_transform(original,donor,seed):
    delta=donor-original;rng=np.random.default_rng(seed)
    direction=rng.standard_normal(delta.shape).astype(np.float32)
    random=direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-20)*np.linalg.norm(delta,axis=1,keepdims=True)
    return lambda h:h+torch.from_numpy(random).to(device=h.device,dtype=h.dtype)


def summarize(records):
    result={'capability':[],'interventions':[],
        'scope':'Validation-only custom memory task; not MIB/RAVEL or proof of a latent u variable',
        'gate':'Each mode needs >=90% unique original-prompt accuracy AND >=90% unique edited-input accuracy separately for each donor type before a state-codebook study',
        'uncertainty':'Small finite validation grid, repeated arithmetic values; descriptive counts, no independent-token significance test'}
    for mode in ('explicit_digit','implicit_sum'):
        selected=[r for r in records if r['case']['mode']==mode]
        originals={};edited={}
        for r in selected:
            if r['prompt'] in originals:
                if r['baseline']['generated_ids']!=originals[r['prompt']]['baseline']['generated_ids']:
                    raise AssertionError('Repeated recipient baseline is not identical')
            originals[r['prompt']]=r
            key=(r['case']['pair_type'],r['edited_prompt'])
            if key in edited and r['edited_baseline']['generated_ids']!=edited[key]['edited_baseline']['generated_ids']:
                raise AssertionError('Repeated edited baseline is not identical')
            edited[key]=r
        baseline=sum(r['baseline_correct'] for r in originals.values());edits=[]
        for kind in ('different_u','same_u'):
            values=[r for (label,_),r in edited.items() if label==kind]
            edits.append({'pair_type':kind,'unique_edited_prompts':len(values),
                          'edited_correct':sum(r['edited_correct'] for r in values)})
        result['capability'].append({'mode':mode,'case_pairs':len(selected),'unique_original_prompts':len(originals),
            'baseline_correct':baseline,'edited_by_type':edits,
            'passed':bool(baseline/len(originals)>=.9 and all(r['edited_correct']/r['unique_edited_prompts']>=.9 for r in edits))})
        for kind in ('different_u','same_u'):
            subset=[r for r in selected if r['case']['pair_type']==kind]
            names=sorted({p['condition'] for r in subset for p in r['patches']})
            for name in names:
                pairs=[(r,p) for r in subset for p in r['patches'] if p['condition']==name]
                eligible=[(r,p) for r,p in pairs if r['baseline_correct'] and r['edited_correct']]
                result['interventions'].append({'mode':mode,'pair_type':kind,'condition':name,
                    'all_n':len(pairs),'eligible_n':len(eligible),
                    'all_joint_cf_success':sum(p['digit']==r['case']['cf_digit'] and p['tag']==r['case']['recipient']['tag'] for r,p in pairs),
                    'eligible_joint_cf_success':sum(p['digit']==r['case']['cf_digit'] and p['tag']==r['case']['recipient']['tag'] for r,p in eligible),
                    'eligible_tag_preserved':sum(p['tag']==r['case']['recipient']['tag'] for r,p in eligible),
                    'format_failures':sum(p['parse_failed'] for r,p in pairs)})
    return result


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg={'model':'Qwen/Qwen2-7B-Instruct','revision':'f2826a00ceef68f0f2b946d945ecc0477ce4450c',
         'layers':[7,14,28],'widths':[1,4,16],'max_new_tokens':16,'cases':cases(),
         'additional_scope':'entire user memory prefix before later offset/tag, including memory inputs and buffer',
         'donor_types_paired_within_recipient':True,
         'scope_energy_note':'Random directions match each scope donor delta per token. Different scopes are not total-energy-matched; no window-size causal superiority claim.',
         'no_codebook_fit':True,'test_split_not_executed':True}
    freeze(ROOT/'plan.json',provenance(cfg,[Path(__file__),Path(__file__).with_name('revision_common.py')]))
    model,tokenizer=load_model(cfg);records=[]
    status(ROOT,'gate',state='running',completed=0,expected=len(cfg['cases']))
    for case in cfg['cases']:
        path=ROOT/'cases'/(case['id']+'.json')
        if path.exists():records.append(json.loads(path.read_text()));continue
        recipient,donor=case['recipient'],case['donor']
        edited={**recipient,'a':donor['a'],'b':donor['b'],'u':donor['u']}
        ids,positions,text=encode(tokenizer,recipient,case['mode'])
        dids,dpositions,dtext=encode(tokenizer,donor,case['mode'])
        eids,epositions,etext=encode(tokenizer,edited,case['mode'])
        if len(ids)!=len(dids) or len(ids)!=len(eids) or positions!=dpositions or positions!=epositions:
            raise ValueError('Counterfactual prefix shapes/positions must match exactly')
        _,memory_positions,_=encode(tokenizer,recipient,case['mode'],'memory_region')
        _,dmemory,_=encode(tokenizer,donor,case['mode'],'memory_region')
        _,ememory,_=encode(tokenizer,edited,case['mode'],'memory_region')
        if memory_positions!=dmemory or memory_positions!=ememory or memory_positions[-16:]!=positions:
            raise ValueError('Memory-region positions must match and contain the original tail')
        original_h,original_logits=capture(model,ids,memory_positions)
        donor_h,_=capture(model,dids,memory_positions)
        edited_h,_=capture(model,eids,memory_positions)
        for layer in cfg['layers']:
            # Same memory prefix, different FUTURE offset/tag, exact fixed shape.
            np.testing.assert_array_equal(donor_h[layer],edited_h[layer])
        baseline=generate(model,tokenizer,ids);counterfactual=generate(model,tokenizer,eids)
        # Embedding-swap positive control must exactly equal editing the real input.
        changed=[i for i,(a,b) in enumerate(zip(ids,eids)) if a!=b]
        if any(i>=positions[0] for i in changed):
            raise AssertionError('Edited input modified recipient offset/tag or buffer')
        if changed:
            with torch.inference_mode():
                embedding=model.model.embed_tokens(torch.tensor([eids[i] for i in changed],device=model.device)).float().cpu().numpy()
            embedding_control=generate(model,tokenizer,ids,model.model.embed_tokens,changed,full_transform(embedding))
            if embedding_control['generated_ids']!=counterfactual['generated_ids']:
                raise AssertionError('Embedding intervention not equal to true counterfactual input')
        else:
            embedding_control={**baseline,'note':'explicit same-u: no input token needs changing'}
        patches=[]
        for layer in (7,14):
            for width in cfg['widths']:
                for method in ('donor','matched_random'):
                    transform=(full_transform(donor_h[layer][-width:]) if method=='donor' else
                        random_transform(original_h[layer][-width:],donor_h[layer][-width:],
                            int(hashlib.sha256(f'{case["id"]}/{layer}/{width}'.encode()).hexdigest()[:8],16)))
                    output=generate(model,tokenizer,ids,model.model.layers[layer-1],positions[-width:],transform)
                    patches.append({'condition':f'{method}_l{layer}_w{width}',**output})
            for method in ('donor','matched_random'):
                transform=(full_transform(donor_h[layer]) if method=='donor' else
                    random_transform(original_h[layer],donor_h[layer],
                        int(hashlib.sha256(f'{case["id"]}/{layer}/memory_region'.encode()).hexdigest()[:8],16)))
                output=generate(model,tokenizer,ids,model.model.layers[layer-1],memory_positions,transform)
                patches.append({'condition':f'{method}_l{layer}_memory_region','positions':memory_positions,**output})
        # Past-position final-block outputs have no downstream block to carry the
        # change. KV at that block is produced BEFORE the output hook.
        null=generate(model,tokenizer,ids,model.model.layers[27],memory_positions,full_transform(donor_h[28]))
        if null['generated_ids']!=baseline['generated_ids']:
            raise AssertionError('Final-block past-position architectural null failed')
        identity=generate(model,tokenizer,ids,model.model.layers[13],positions,lambda h:h)
        if identity['generated_ids']!=baseline['generated_ids']:
            raise AssertionError('Identity generation mismatch')
        record={'case':case,'prompt':text,'donor_prompt':dtext,'edited_prompt':etext,'input_ids':ids,
            'donor_input_ids':dids,'edited_input_ids':eids,'positions':positions,'baseline':baseline,
            'memory_region_positions':memory_positions,'saved_activation_positions':'memory_region_positions',
            'final_block_null_scope':'memory_region',
            'edited_baseline':counterfactual,'embedding_control':embedding_control,'final_block_null':null,
            'identity':identity,'patches':patches,
            'baseline_correct':baseline['digit']==(recipient['u']+recipient['c'])%10 and baseline['tag']==recipient['tag'],
            'edited_correct':counterfactual['digit']==case['cf_digit'] and counterfactual['tag']==recipient['tag'],
            'future_causality_exact':True,'embedding_control_exact':True,'final_block_null_exact':True}
        write_npz(ROOT/'activations'/(case['id']+'.npz'),**{f'{name}_{l}':h[l] for name,h in [('recipient',original_h),('donor',donor_h)] for l in cfg['layers']})
        write_json(path,record);records.append(record)
        status(ROOT,'gate',state='running',completed=len(records),expected=len(cfg['cases']))
        print(json.dumps({'counterfactual_gate_cases':len(records)}),flush=True)
    if len(records)!=len(cfg['cases']) or {r['case']['id'] for r in records}!={c['id'] for c in cfg['cases']}:
        raise AssertionError('Incomplete gate coverage')
    result=summarize(records);write_json(ROOT/'summary.json',result)
    write_json(ROOT/'_SUCCESS.json',{'cases':len(records),'summary_sha256':sha(ROOT/'summary.json'),
        'interpretation':'Gate executed, not necessarily scientifically passed'})
    status(ROOT,'gate',state='complete',completed=len(records),expected=len(cfg['cases']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        run()
    except BaseException:
        status(ROOT,'gate',state='failed',traceback=traceback.format_exc());raise

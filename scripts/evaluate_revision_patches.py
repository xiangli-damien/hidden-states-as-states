"""Single-layer multi-token reconstruction at an explicitly fixed visible prefix.

Teacher-forced NLL uses the entire stored reference continuation, including EOS;
KL uses the SAME next-token distribution. Free-generation pilot is separate and
is scored by OpenAct's MATH evaluator. Every condition uses a fresh KV cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from revision_common import config, freeze, provenance, reconstruct, write_json, write_npz, status, OncePatch
from extract_revision_prefixes import load_model


def load_questions(cfg):
    rows, tokens = [], {}
    for marker in sorted((Path(cfg['output'])/'prefixes').glob('shard_*/_SUCCESS.json')):
        rows.append(pd.read_parquet(marker.parent/'rows.parquet'))
        for item in json.loads((marker.parent/'tokens.json').read_text()):
            tokens[item['sample_id']]=item
    return pd.concat(rows,ignore_index=True), tokens


def choose_questions(frame,n):
    out=[]
    for split in ('validation','test'):
        sub=frame.loc[frame.split.eq(split)].copy()
        sub['order']=sub.sample_id.map(lambda s: hashlib.sha256(('revision-pilot-v1/'+s).encode()).hexdigest())
        out.extend(sub.sort_values('order').head(n).sample_id)
    return out


def replacement(decoder,method,seed):
    def apply(h):
        if method=='identity':
            return h
        x=h.float().cpu().numpy()
        if method.startswith('beta_'):
            beta=float(method.split('_')[1]);z=(1-beta)*reconstruct(x,decoder,'centroid')+beta*x
        elif method in ('matched_random','centroid_energy1','matched_random_energy1'):
            delta=reconstruct(x,decoder,'centroid')-x
            if 'energy1' in method:
                # Last selected slot is shared across nested windows.
                delta *= np.linalg.norm(delta[-1])/max(np.linalg.norm(delta),1e-20)
            if method.startswith('matched_random'):
                direction=np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
                delta=direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-20)*np.linalg.norm(delta,axis=1,keepdims=True)
            z=x+delta
        else:
            z=reconstruct(x,decoder,method)
        return torch.from_numpy(z).to(device=h.device,dtype=h.dtype)
    return apply


@torch.inference_mode()
def teacher_force(model,prefix,reference,layer,positions,transform):
    if not reference:
        raise ValueError('Empty reference continuation')
    patch=OncePatch(positions,len(prefix),transform)
    handle=model.model.layers[layer-1].register_forward_hook(patch)
    try:
        out=model(input_ids=torch.tensor([prefix],device=model.device),use_cache=True,logits_to_keep=1)
        logp=out.logits[0,-1].float().log_softmax(-1)
        first=logp.cpu().numpy()
        total=-float(logp[reference[0]])
        cache=out.past_key_values
        del out,logp
        for start in range(0,len(reference)-1,128):
            end=min(start+128,len(reference)-1)
            inp=torch.tensor([reference[start:end]],device=model.device)
            out=model(input_ids=inp,past_key_values=cache,use_cache=True)
            logits=out.logits[0].float()
            targets=torch.tensor(reference[start+1:end+1],device=model.device)
            total+=float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
            cache=out.past_key_values
            del out,logits,targets
        if patch.calls!=1:
            raise AssertionError('Intervention did not apply exactly once')
        return {'nll_sum':total,'reference_tokens':len(reference),'nll':total/len(reference),
                'actual_patch_energy':patch.energy,'patch_calls':patch.calls},first
    finally:
        handle.remove()


@torch.inference_mode()
def generate(model,tokenizer,prefix,layer,positions,transform,budget):
    patch=OncePatch(positions,len(prefix),transform)
    handle=model.model.layers[layer-1].register_forward_hook(patch)
    try:
        value=model.generate(torch.tensor([prefix],device=model.device),max_new_tokens=budget,
            do_sample=False,pad_token_id=model.generation_config.pad_token_id)
        ids=value[0,len(prefix):].tolist()
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        if patch.calls!=1:
            raise AssertionError('Generation patch missing')
        return {'generated_ids':ids,'suffix':tokenizer.decode(ids,skip_special_tokens=True),
                'length':len(ids),'finish_reason':'eos' if ids[-1] in eos else 'length',
                'actual_patch_energy':patch.energy}
    finally:
        handle.remove()


def run(cfg,phase):
    root=Path(cfg['output']);dest=root/phase;dest.mkdir(parents=True,exist_ok=True)
    frame,tokens=load_questions(cfg);indexed=frame.set_index('sample_id')
    n=cfg['functional_pilot_per_split'] if phase=='functional' else 12
    ids=choose_questions(frame,n)
    methods=(['identity','zero','mean','centroid','kmeans_centroid','local_pca_8','global_pca_8',
              'beta_0.25','beta_0.5','beta_0.75','matched_random','centroid_energy1','matched_random_energy1']
             if phase=='functional' else ['identity','centroid','local_pca_8','matched_random','centroid_energy1'])
    layers=cfg['functional_layers'] if phase=='functional' else [14]
    prefixes=cfg['functional_prefixes']
    files=[Path(__file__),Path(__file__).with_name('revision_common.py'),root/'prefixes/plan.json']
    decoder_paths=[]
    for prefix in prefixes:
        for layer in layers:
            for role in (['tokens','question_tokens'] if prefix==0 and phase=='functional' else ['tokens']):
                path=root/'geometry'/f'p{prefix}_l{layer}_{role}'/'decoder.npz'
                if not (path.parent/'_SUCCESS.json').exists():
                    raise ValueError(f'Validated token decoder not ready: {path}')
                files.extend([path,path.parent/'_SUCCESS.json']);decoder_paths.append(str(path))
    plan=provenance(cfg,files)
    plan.update(phase=phase,sample_ids=ids,methods=methods,layers=layers,
                observation='pilot; historical MATH test has been explored',
                positions='nested real-token windows; question-tail separate from chat-tail',
                scope='single-layer intervention; all unpatched positions remain available')
    freeze(dest/'plan.json',plan)
    model,tokenizer=load_model(cfg)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    evaluator=None
    if phase=='behavior':
        from openact_eval.evaluators.registry import auto_select_evaluator
        evaluator=auto_select_evaluator('math')
    done,started=0,time.monotonic()
    for sid in ids:
        item=tokens[sid];row=indexed.loc[sid]
        for prefixn in prefixes:
            if len(item['response_ids'])<=prefixn or eos.intersection(item['response_ids'][:prefixn]):
                continue
            prefix=item['prompt_ids']+item['response_ids'][:prefixn]
            reference=item['response_ids'][prefixn:]
            for role in (['tokens','question_tokens'] if prefixn==0 and phase=='functional' else ['tokens']):
                eligible=item['question_positions'] if role=='question_tokens' else list(range(len(prefix)))
                if len(eligible)<16:
                    continue
                for layer in layers:
                    with np.load(root/'geometry'/f'p{prefixn}_l{layer}_{role}'/'decoder.npz') as d:
                        decoder={k:d[k].copy() for k in d.files}
                    baseline=None
                    for width in cfg['functional_widths']:
                        positions=eligible[-width:]
                        for method in methods:
                            filename=f'{sid}_p{prefixn}_l{layer}_{role}_w{width}_{method}.json'
                            path=dest/'samples'/filename
                            logpath=path.with_suffix('.npz')
                            if path.exists():
                                record=json.loads(path.read_text())
                                if phase=='functional' and method=='identity':
                                    with np.load(logpath) as v:
                                        baseline=v['logp'].copy()
                                done+=1;continue
                            seed=int(hashlib.sha256(f'{sid}/{prefixn}/{layer}/{role}/{width}'.encode()).hexdigest()[:8],16)
                            transform=replacement(decoder,method,seed)
                            base={'sample_id':sid,'split':row.split,'prefix_tokens':prefixn,'layer':layer,
                                  'role':role,'width':width,'positions':positions,'method':method,
                                  'original_correct':int(row.label),'category':str(row.category),'level':int(row.level)}
                            if phase=='functional':
                                metrics,logp=teacher_force(model,prefix,reference,layer,positions,transform)
                                if method=='identity':
                                    if baseline is not None and not np.array_equal(logp,baseline):
                                        raise AssertionError('Identity logits differ across intervention width')
                                    baseline=logp.copy();write_npz(logpath,logp=logp)
                                kl=float((np.exp(baseline.astype(np.float64))*(baseline.astype(np.float64)-logp)).sum())
                                # Tiny negative roundoff is reported, not used to suggest negative divergence.
                                metrics.update(next_token_kl=kl,next_token_argmax_agreement=bool(baseline.argmax()==logp.argmax()))
                                record={**base,**metrics}
                            else:
                                record={**base,**generate(model,tokenizer,prefix,layer,positions,transform,
                                        cfg['generation_budget']-prefixn)}
                                text=tokenizer.decode(item['response_ids'][:prefixn]+record['generated_ids'],skip_special_tokens=True)
                                score=evaluator.evaluate_sample(SimpleNamespace(sample_idx=int(row.sample_idx),
                                    response_text=text,ground_truth=str(row.ground_truth),meta={'sample_id':sid}))
                                if score.is_correct is None or score.error:
                                    raise ValueError('Behavior evaluation failed')
                                record.update(response_text=text,parsed_answer=score.extracted_answer,
                                              normalized_answer=score.normalized_answer,correct=bool(score.is_correct),
                                              ground_truth=str(row.ground_truth),parse_failed=bool(score.meta['parse_failed']))
                            write_json(path,record);done+=1
                            if done%20==0:
                                status(root,phase,state='running',completed_conditions=done,
                                       current_sample=sid,seconds=time.monotonic()-started)
                                print(json.dumps({'phase':phase,'conditions':done,'sample':sid}),flush=True)
    records=[json.loads(p.read_text()) for p in sorted((dest/'samples').glob('*.json'))]
    pd.DataFrame(records).to_parquet(dest/'per_question.parquet',index=False)
    status(root,phase,state='complete',completed_conditions=len(records),questions=len(ids),
           seconds=time.monotonic()-started)
    write_json(dest/'_SUCCESS.json',{'conditions':len(records),'selected_questions':len(ids),'completed_unix':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--phase',choices=['functional','behavior'],default='functional')
    a=p.parse_args();cfg=config(a.config)
    try:
        run(cfg,a.phase)
    except BaseException:
        status(cfg['output'],a.phase,state='failed',traceback=traceback.format_exc());raise

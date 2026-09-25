"""Reviewed follow-up: exact saved tokens, generated-position hooks, resumable receipts.

The CPU reporter makes D6 decisions. This GPU worker never chooses a candidate
from partial results. Stage 3B requires the original frozen detector artifacts.
"""
import argparse
import copy
from datetime import datetime
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
import pandas as pd
import torch
import zarr
from extract_revision_prefixes import load_model
from revision_common import sha, freeze, write_json, write_npz, status
from sink_direction_common import boxed_answer
from hss_followup_common import (sentence_boundaries, token_text, boxed_token_indices,
    gmm_assign, verify_plan, receipt, verify_record)
from hss_followup_tools import (PARAMS, check_torch_equivalence, make_torch_transform,
    cut_token_index, subsample_token_indices, extract_braced, control_condition_names)
from openact_eval.evaluators.registry import auto_select_evaluator
from openact_core.tasks.parsers.math_parser import MathParser


class GeneratedPositions:
    """Batch-one full TF or fresh-cache generation, with an explicit absolute offset.

    Prefill may include generated prefix tokens. They ARE patched. Prompt tokens
    and the prompt-last-token prediction are untouched. No padded inputs allowed.
    """
    def __init__(self, prompt_length, transform):
        self.prompt_length = prompt_length; self.transform = transform; self.seen = 0
        self.n = 0; self.changed = 0; self.norm_sum = 0.; self.delta_sum = 0.

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output,tuple) else output
        assert h.ndim == 3 and h.shape[0] == 1
        start = max(0,self.prompt_length-self.seen); self.seen += h.shape[1]
        if start >= h.shape[1]: return None
        before = h[0,start:]
        after = self.transform(before)
        assert after.shape == before.shape and torch.isfinite(after).all()
        delta = (after.float()-before.float()).norm(dim=-1)
        self.n += len(delta); self.changed += int((delta>0).sum())
        self.norm_sum += float(before.float().norm(dim=-1).sum()); self.delta_sum += float(delta.sum())
        if torch.equal(before,after): return None
        new = h.clone();new[0,start:] = after
        return (new,*output[1:]) if isinstance(output,tuple) else new

    def metrics(self):
        return dict(n=self.n,changed=self.changed,changed_fraction=self.changed/max(1,self.n),
            mean_actual_delta_norm=self.delta_sum/max(1,self.n),mean_token_norm=self.norm_sum/max(1,self.n))


@torch.inference_mode()
def forward(model, prompt, response, transform=None, output_hidden=False):
    ids = torch.tensor([prompt+response],device=model.device)
    hook = GeneratedPositions(len(prompt),transform) if transform is not None else None
    handle = model.model.layers[PARAMS['hook_layer']-1].register_forward_hook(hook) if hook else None
    try:
        result = model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
            output_hidden_states=output_hidden)
    finally:
        if handle:handle.remove()
    if hook: assert hook.n == len(response)
    return result,hook


@torch.inference_mode()
def logprobs(model,prompt,response,transform=None):
    """Every response token is scored by the previous position; head in chunks."""
    result,hook = forward(model,prompt,response,transform)
    positions = result.last_hidden_state[0,len(prompt)-1:len(prompt)+len(response)-1]
    targets=torch.tensor(response,device=model.device)
    values=[]
    for i in range(0,len(response),64):
        logits=model.lm_head(positions[i:i+64]).float()
        values.append((logits.gather(1,targets[i:i+64,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)).cpu())
    return torch.cat(values).numpy(), hook.metrics() if hook else None


@torch.inference_mode()
def layer_tokens(model,row,indices=None):
    found={}
    def capture(module,args,out):
        h=out[0] if isinstance(out,tuple) else out
        v=h[0,len(row['prompt_ids']):]
        if indices is not None:v=v[indices]
        found['x']=v.float().cpu().numpy()
    handle=model.model.layers[PARAMS['hook_layer']-1].register_forward_hook(capture)
    try: forward(model,row['prompt_ids'],row['response_ids'])
    finally: handle.remove()
    return found['x']


@torch.inference_mode()
def generate(model,tok,prompt,prefix,generation,budget,transform=None):
    ids=torch.tensor([prompt+prefix],device=model.device)
    hook=GeneratedPositions(len(prompt),transform) if transform is not None else None
    handle=model.model.layers[PARAMS['hook_layer']-1].register_forward_hook(hook) if hook else None
    try:
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=generation,
            max_new_tokens=budget,use_cache=True)
    finally:
        if handle:handle.remove()
    continuation=out[0,ids.shape[1]:].tolist()
    if hook:assert hook.n==len(prefix)+len(continuation)-1
    return continuation,hook.metrics() if hook else None


def score(evaluator,sid,text,gt):
    r=evaluator.evaluate_sample(SimpleNamespace(sample_idx=int(sid.split('_')[-1]),
        response_text=text,ground_truth=gt,meta={'sample_id':sid}))
    assert not r.error and r.is_correct is not None
    return bool(r.is_correct)


def preflight(root,model,tok,cohort,generation,evaluator):
    path=root/'preflight_SUCCESS.json'
    if path.exists():verify_record(path,root);return
    check=check_torch_equivalence(D=model.config.hidden_size,device=model.device)
    assert max(check.values())<1e-4,check
    # Complete stored sequence, not a shortened context whose matrix arithmetic
    # can differ from the collector. The source was a teacher-forced pass too.
    row=cohort[sorted(cohort)[0]]
    x=layer_tokens(model,row)
    store=zarr.open(str(Path(row['source'])/'tensors.zarr'),mode='r')
    ptr=store['tokens/sample_ptr']; a,b=int(ptr[row['sample_idx']]),int(ptr[row['sample_idx']+1])
    np.testing.assert_array_equal(store['tokens/ids'][a:b],row['response_ids'])
    reference=np.asarray(store['hidden_states/per_token'][a:b,PARAMS['hook_layer'],:])
    relative=float(np.linalg.norm(x-reference)/np.linalg.norm(reference))
    assert relative<1e-3,{'relative_activation_error':relative}
    response=row['response_ids'][:24];prompt=row['prompt_ids']
    base,_=logprobs(model,prompt,response)
    identity,_=logprobs(model,prompt,response,lambda h:h)
    np.testing.assert_array_equal(base,identity)
    # Independent full-head check validates causal shifting and natural-log sums.
    ids=torch.tensor([prompt+response],device=model.device)
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False).logits[0,len(prompt)-1:-1].float()
        direct=logits.log_softmax(-1).gather(1,torch.tensor(response,device=model.device)[:,None]).squeeze(1)
    np.testing.assert_allclose(base,direct.cpu().numpy(),atol=2e-5,rtol=1e-5)
    natural,_=generate(model,tok,prompt,[],generation,24)
    zero,_=generate(model,tok,prompt,[],generation,24,lambda h:h)
    assert natural==zero
    # Both cached and full-sequence hooks must patch the same positions.
    perturb=torch.ones(model.config.hidden_size,device=model.device)*.001
    transform=lambda h:(h.float()+perturb).to(h.dtype)
    h=GeneratedPositions(len(prompt),transform)
    sample=torch.randn(1,len(prompt)+len(response),model.config.hidden_size,device=model.device)
    full=h(None,(),sample)
    hc=GeneratedPositions(len(prompt),transform)
    assert hc(None,(),sample[:,:len(prompt)]) is None
    pieces=[sample[:,:len(prompt)]]+[hc(None,(),sample[:,j:j+1]) for j in range(len(prompt),sample.shape[1])]
    assert torch.equal(full,torch.cat(pieces,1))
    assert torch.equal(full[:,:len(prompt)],sample[:,:len(prompt)])
    write_json(path,dict(torch_equivalence=check,stored_activation_relative_error=relative,
        stored_activation_max_absolute_error=float(abs(x-reference).max()),
        identity_logprobs_exact=True,identity_generation_exact=True,causal_logprobs_checked=True,
        cached_full_hook_positions_equal=True,source_id=row['sample_id'],torch=torch.__version__,
        scorer_sources={inspect.getfile(MathParser):sha(inspect.getfile(MathParser)),
            inspect.getfile(type(evaluator)):sha(inspect.getfile(type(evaluator))),
            inspect.getfile(type(evaluator._matcher)):sha(inspect.getfile(type(evaluator._matcher)))},
        tokenizer_revision=tok.init_kwargs.get('_commit_hash'),gpu=torch.cuda.get_device_name()))
    receipt(path,root)


@torch.inference_mode()
def reencode(model,tok,prompt,ids,maps):
    text,offsets,core,rule=token_text(tok,ids)
    boundaries=sentence_boundaries(text,offsets,len(ids))
    result,_=forward(model,prompt,ids,output_hidden=True)
    means=[];prefixes=[];local=[];global_ids=[]
    for L,h in enumerate(result.hidden_states):
        x=h[0,len(prompt):].float()
        m=x.mean(0).cpu().numpy()
        prefix=x.cumsum(0)[torch.tensor(boundaries,device=x.device)-1]/torch.tensor(boundaries,device=x.device)[:,None]
        means.append(m);prefixes.append(prefix.cpu().numpy().astype('float16'))
        k=int(gmm_assign(m,maps[f'l{L}_means'],maps[f'l{L}_covariances'],maps[f'l{L}_weights'])[0])
        local.append(k);global_ids.append(int(maps[f'l{L}_global'][k]))
    return dict(means=np.stack(means),prefix_means=np.stack(prefixes,axis=1),
        boundaries=np.array(boundaries),local=np.array(local),global_ids=np.array(global_ids)),core,rule


def step1(root,plan,model,tok,generation,cohort,maps,evaluator):
    old=Path(plan['config']['previous']);cases=json.loads((root/'test_cases.json').read_text())
    records=[];start=time.time();n=0
    for case in cases:
        sid=case['sample_id'];historical=cohort[sid]
        for arm in ['historical','zero','hss','random']:
            dest=root/'step1'/sid;dest.mkdir(parents=True,exist_ok=True);path=dest/(arm+'.json')
            if path.exists():
                rec=verify_record(path,root);assert sha(path.with_suffix('.npz'))==rec['arrays_sha256']
                item=json.loads(path.read_text())
            else:
                original=(dict(generated_ids=historical['response_ids'],correct=historical['correct'],
                    response_text=historical['response_text']) if arm=='historical' else
                    json.loads((old/'outputs'/sid/(arm+'.json')).read_text()))
                ids=original['generated_ids'];tick=time.time()
                arrays,core,offset_rule=reencode(model,tok,case['prompt_ids'],ids,maps)
                text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                assert text==original['response_text']
                forced=None;forced_correct=False
                if boxed_answer(text) is None:
                    force_ids=tok(PARAMS['force_string'],add_special_tokens=False)['input_ids']
                    # Remove terminal EOS only when appending a continuation.
                    new,_=generate(model,tok,case['prompt_ids'],ids[:core]+force_ids,generation,PARAMS['force_max_new_tokens'])
                    ftext=tok.decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    answer=extract_braced(ftext)
                    if answer is not None and answer.strip():
                        forced_correct=score(evaluator,sid,'\\boxed{'+answer+'}',case['ground_truth'])
                    forced=dict(ids=new,text=ftext,answer=answer,correct=forced_correct,unhooked=True)
                gids=arrays['global_ids'];loc=arrays['local']
                item=dict(sample_id=sid,split=case['split'],arm=arm,type=historical['type'],
                    correct=original['correct'],boxed=boxed_answer(text) is not None,forced=forced,
                    original_or_forced_correct=bool(original['correct'] or forced_correct),
                    global_ids=gids.tolist(),local_ids=loc.tolist(),
                    dominant_types=[str(maps[f'l{L}_dominant_type'][k]) for L,k in enumerate(loc)],
                    n_tokens=len(ids),offset_rule=offset_rule,seconds=time.time()-tick)
                if arm=='zero':
                    oldids=historical['response_ids'];common=min(len(ids),len(oldids))
                    mismatch=next((i for i in range(common) if ids[i]!=oldids[i]),None)
                    if mismatch is None and len(ids)!=len(oldids):mismatch=common
                    item.update(historical_text_exact=text==historical['response_text'],first_different_token=mismatch)
                write_npz(path.with_suffix('.npz'),**arrays);write_json(path,item)
                receipt(path,root,dict(arrays_sha256=sha(path.with_suffix('.npz'))))
            records.append(item);n+=1
            status(root,'step1',completed=n,expected=plan['expected_step1'],elapsed_seconds=time.time()-start,last=sid+'/'+arm)
    flat=[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in records]
    pd.DataFrame(flat).to_parquet(root/'step1_reencode.parquet',index=False)
    # Prefix arrays remain individually checksummed shards, so resuming never
    # rewrites a multi-GB archive. The manifest is the complete NPZ collection.
    manifest={str(p.relative_to(root)):sha(p) for p in (root/'step1').glob('*/*.npz')}
    write_json(root/'step1_prefix_states.manifest.json',manifest)
    write_json(root/'step1_COMPLETE.json',dict(n=n,parquet_sha256=sha(root/'step1_reencode.parquet'),
        prefix_manifest_sha256=sha(root/'step1_prefix_states.manifest.json')))


def build_vectors(root,plan,model,cohort,initial):
    path=root/'step2_vectors.npz'
    if path.exists(): verify_record(path,root);return dict(np.load(path))
    split=json.loads((root/'split.json').read_text());rng=np.random.default_rng(PARAMS['seed']+5)
    samples=[];norms=[];sink_z=[]
    for group,ids in [('N_A',split['tau_ids']),('S_A',split['S_A'])]:
        for i,sid in enumerate(ids):
            dest=root/'vector_samples'/group/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            indices=subsample_token_indices(len(cohort[sid]['response_ids']),rng)
            if dest.exists():
                verify_record(dest,root)
                with np.load(dest) as z:x=z['x'];np.testing.assert_array_equal(indices,z['indices'])
            else:
                x=layer_tokens(model,cohort[sid],indices)
                write_npz(dest,x=x,indices=indices);receipt(dest,root)
            if group=='N_A': samples.append(x);norms.extend(np.linalg.norm(x,axis=1).tolist())
            else:sink_z.extend((x@initial['s_hat'].astype('float32')).tolist())
            status(root,'vectors',group=group,completed=i+1,expected=len(ids),last=sid)
    X=np.concatenate(samples).astype('float32');del samples
    axes=np.concatenate([initial['s_hat'][None],initial['controls']],axis=0).astype('float32')
    projections=X@axes.T;taus=np.percentile(projections,PARAMS['tau_percentile'],axis=0)
    # GPU covariance PCA of the prespecified N_A sample, centered, no normalization.
    xt=torch.as_tensor(X,device=model.device);xt=xt-xt.mean(0)
    cov=xt.T@xt/(len(xt)-1)
    _,eigvec=torch.linalg.eigh(cov)
    du=torch.as_tensor(initial['d']/np.linalg.norm(initial['d']),device=model.device,dtype=torch.float32)
    energy={str(r):float((eigvec[:,-r:].T@du).square().sum()) for r in [8,64,512]}
    del xt,cov,eigvec
    # Coverage reference: actual training first16 tokens used by this decoder.
    distances=[];train=set(split['S_A']+split['S_B']+split['N_A']+split['N_B'])
    centers=torch.as_tensor(initial['token_centers'],device=model.device,dtype=torch.float32)
    for shard in sorted((Path(plan['config']['foundation'])/'prefixes').glob('shard_*')):
        rows=pd.read_parquet(shard/'rows.parquet')
        ref=json.loads((shard/'_SUCCESS.json').read_text())
        assert sha(shard/'prefix_16.npz')==ref['sha256']['prefix_16.npz']
        with np.load(shard/'prefix_16.npz') as z:
            take=rows.sample_id.isin(train).to_numpy() & z['valid'].astype(bool)
            values=z['window'][take,1].reshape(-1,centers.shape[1]).astype('float32')
        for j in range(0,len(values),512):
            x=torch.as_tensor(values[j:j+512],device=model.device)
            d2=x.square().sum(1)[:,None]-2*x@centers.T+centers.square().sum(1)[None]
            distances.extend(d2.min(1).values.clamp_min(0).sqrt().cpu().tolist())
    output=dict(initial,tau=taus[0],control_taus=taus[1:],coverage_q99=np.percentile(distances,99),
        sampled_token_median_norm=np.median(norms),seeds=np.array([PARAMS['seed']+i for i in range(6)]))
    write_npz(path,**output);receipt(path,root)
    write_json(root/'step2_vector_diagnostics.json',dict(n_tau_tokens=len(X),tau=float(taus[0]),
        n_coverage_reference_tokens=len(distances),coverage_q99=float(output['coverage_q99']),
        sink_A_projection_quantiles=np.quantile(sink_z,[0,.25,.5,.75,1]).tolist(),
        nonsink_A_projection_quantiles=np.quantile(projections[:,0],[0,.25,.5,.75,1]).tolist(),
        pca_direction_energy=energy,amplitude_over_median_norm=float(PARAMS['alpha']*np.linalg.norm(initial['d'])/np.median(norms))))
    return output


def transform_for(condition,qtype,vectors,device):
    kw={}; a=float(PARAMS['alpha']*np.linalg.norm(vectors['d']))
    if condition=='NONE':kind='none'
    elif condition=='C0':kind='add';kw=dict(a=a,v_unit=vectors['d']/np.linalg.norm(vectors['d']))
    elif condition=='C1':kind='clamp';kw=dict(s_hat=vectors['s_hat'],tau=float(vectors['tau']))
    elif condition=='C2':
        i=list(vectors['type_names']).index(qtype)
        kind='add';kw=dict(a=a,v_unit=vectors['type_vectors'][i]/np.linalg.norm(vectors['d']))
    elif condition=='C3':kind='chart';kw=dict(a=a,chart_units=vectors['chart'],centroids_tok=vectors['token_centers'])
    else:
        family,dist,k=condition.rsplit('_',2)
        i=int(k)-1+(PARAMS['n_ctrl_iso'] if dist=='man' else 0)
        if family=='CTRL_ADD':kind='add';kw=dict(a=a,v_unit=vectors['controls'][i])
        elif family=='CTRL_CLAMP':kind='clamp';kw=dict(s_hat=vectors['controls'][i],tau=float(vectors['control_taus'][i]))
        elif family=='CTRL_PROJ':kind='chart';kw=dict(a=a,chart_units=vectors['chart_controls'][i],centroids_tok=vectors['token_centers'])
        else:raise ValueError(condition)
    return make_torch_transform(kind,device=device,**kw)


def step2(root,plan,model,tok,cohort,vectors):
    split=json.loads((root/'split.json').read_text())
    conditions=['NONE','C0','C1','C2','C3']+sum([control_condition_names(f) for f in ['CTRL_ADD','CTRL_CLAMP','CTRL_PROJ']],[])
    records=[];start=time.time();force_ids=tok(PARAMS['force_string'],add_special_tokens=False)['input_ids']
    for qset,ids in [('sink',split['S_B']),('normal',split['M_B'])]:
        for sid in ids:
            row=cohort[sid];r=row['response_ids'];prompt=row['prompt_ids']
            text,offsets,core,offset_rule=token_text(tok,r)
            # EOS is kept in the historical NLL, but not in the forced context.
            bounds=sentence_boundaries(text,offsets[:core],core)
            starts=[0]+bounds[:-1]
            sentences=[tok.decode(r[a:b],skip_special_tokens=True,clean_up_tokenization_spaces=False) for a,b in zip(starts,bounds)]
            cut,rule=cut_token_index(sentences,starts,core)
            if qset=='sink' and cut is None:raise ValueError('Empty screen response')
            ans_ids=tok(str(row['ground_truth'])+'}',add_special_tokens=False)['input_ids']
            forced=r[:cut]+force_ids+ans_ids if qset=='sink' else None
            boxed_ix=boxed_token_indices(text,offsets)
            if qset=='normal' and not len(boxed_ix):raise ValueError('Normal answer has no boxed token span')
            for cond in conditions:
                dest=root/'step2'/sid;dest.mkdir(parents=True,exist_ok=True);path=dest/(cond+'.json')
                if path.exists():verify_record(path,root);item=json.loads(path.read_text())
                else:
                    tick=time.time();transform=transform_for(cond,row['type'],vectors,model.device)
                    lp,geometry=logprobs(model,prompt,r,transform)
                    item=dict(question_id=sid,set=qset,condition=cond,
                        family=cond if cond in ['NONE','C0','C1','C2','C3'] else cond.rsplit('_',2)[0],
                        m_ans=None,m_loop=None,m_col_ans=None,m_col_nll=None,type=row['type'],
                        cut=cut,cut_rule=rule,offset_rule=offset_rule,geometry=geometry)
                    if qset=='sink':
                        fp,fg=logprobs(model,prompt,forced,transform)
                        item.update(m_ans=float(fp[-len(ans_ids):].sum(dtype=np.float64)),
                            m_loop=float(-lp[cut:].mean(dtype=np.float64)) if len(r)-cut>=PARAMS['min_loop_tokens'] else None,
                            forced_geometry=fg,answer_tokens=len(ans_ids))
                    else:item.update(m_col_ans=float(lp[boxed_ix].sum(dtype=np.float64)),m_col_nll=float(-lp.mean(dtype=np.float64)))
                    item['seconds']=time.time()-tick;write_json(path,item);receipt(path,root)
                records.append(item)
                elapsed=time.time()-start
                status(root,'step2',completed=len(records),expected=plan['expected_step2'],elapsed_seconds=elapsed,last=sid+'/'+cond,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    flat=[{k:v for k,v in r.items() if k not in ['geometry','forced_geometry']} for r in records]
    pd.DataFrame(flat).to_parquet(root/'step2_screen_long.parquet',index=False)
    # Independent unhooked token diagnostics, one pass per held-out response.
    diagpath=root/'step2_token_diagnostics.json';diags=[]
    for qset,ids in [('sink',split['S_B']),('normal',split['M_B'])]:
        for sid in ids:
            path=root/'token_diagnostics'/(sid+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():verify_record(path,root);diag=json.loads(path.read_text())
            else:
                x=torch.as_tensor(layer_tokens(model,cohort[sid]),device=model.device)
                c=torch.as_tensor(vectors['token_centers'],device=model.device)
                dist=(x.square().sum(1)[:,None]-2*x@c.T+c.square().sum(1)[None]).clamp_min(0)
                region=dist.argmin(1);ratio=vectors['chart_ratio'][region.cpu().numpy()]
                axis=torch.as_tensor(vectors['s_hat'],device=model.device,dtype=torch.float32)
                control_axes=torch.as_tensor(vectors['controls'],device=model.device,dtype=torch.float32)
                control_thresholds=torch.as_tensor(vectors['control_taus'],device=model.device,dtype=torch.float32)
                diag=dict(sample_id=sid,set=qset,n_tokens=len(x),
                    c1_ideal_clamped_fraction=float(((x@axis)>float(vectors['tau'])).float().mean()),
                    control_ideal_clamped_fractions=((x@control_axes.T)>control_thresholds).float().mean(0).cpu().tolist(),
                    coverage_exceed_count=int((dist.min(1).values.sqrt()>float(vectors['coverage_q99'])).sum()),
                    chart_ratio_quantiles=np.quantile(ratio,[0,.25,.5,.75,1]).tolist(),chart_ratio_sum=float(ratio.sum()))
                write_json(path,diag);receipt(path,root)
            diags.append(diag)
    write_json(diagpath,diags)
    write_json(root/'step2_COMPLETE.json',dict(n=len(records),seconds=time.time()-start,
        parquet_sha256=sha(root/'step2_screen_long.parquet')))



def stage3(root,plan,model,tok,generation,cohort,maps,vectors,evaluator):
    decision=json.loads((root/'step2_decision.json').read_text())
    if decision['step3']=='3B':
        write_json(root/'step3_status.json',dict(state='blocked_missing_artifacts',branch='3B',
            required=['original_Qwen_section5.2_prefix_map','frozen_HSS_NB_parameters','Table2_threshold'],
            reason='Original Qwen artifacts were not located; no substitute detector was trained or silently used.',
            decided_unix=time.time()))
        return
    cases=json.loads((root/'test_cases.json').read_text());old=Path(plan['config']['previous'])
    gate=root/'step3_time_gate.json'
    if not gate.exists():
        # Conservative full-budget estimate including the possible entire zero
        # replay; 25% headroom for conditional hooks. No reduced test cohort.
        old_clock=json.loads((old/'status.json').read_text())
        old_records=[json.loads(p.read_text()) for p in (old/'outputs').glob('*/*.json') if not p.name.endswith('.receipt.json')]
        seconds_per_token=sum(r['seconds'] for r in old_records)/sum(r['n_tokens'] for r in old_records)
        estimate=len(cases)*3*PARAMS['max_new_tokens']*seconds_per_token*1.25
        deadline=datetime.fromisoformat(PARAMS['freeze_utc']).timestamp()
        freeze(gate,dict(estimated_seconds=estimate,seconds_per_token=seconds_per_token,
            started_unix=time.time(),deadline_unix=deadline,
            allowed=time.time()+estimate+PARAMS['finish_reserve_seconds']<deadline,
            rule='All155 cases,2 new arms,allow full155 zero replay,max2048tokens,25percent headroom,1h reserve'))
    if not json.loads(gate.read_text())['allowed']:
        write_json(root/'step3_status.json',dict(state='deferred_by_time_gate',branch='3A',chosen=decision['chosen']))
        return
    def one(case,arm):
        sid=case['sample_id'];path=root/'step3A'/sid/(arm+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():verify_record(path,root);return json.loads(path.read_text())
        tick=time.time();condition='NONE' if arm=='zero' else arm
        transform=transform_for(condition,cohort[sid]['type'],vectors,model.device)
        ids,geom=generate(model,tok,case['prompt_ids'],[],generation,PARAMS['max_new_tokens'],transform)
        text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        # New answers are re-encoded without hooks, not assigned from states
        # measured during intervention. This matches the Step1 definition.
        arrays,core,rule=reencode(model,tok,case['prompt_ids'],ids,maps)
        write_npz(path.with_suffix('.npz'),**arrays)
        item=dict(sample_id=sid,split=case['split'],condition=arm,generated_ids=ids,response_text=text,
            correct=score(evaluator,sid,text,case['ground_truth']),complete_boxed=boxed_answer(text) is not None,
            n_tokens=len(ids),finish_reason='eos' if ids[-1] in eos else 'length',geometry=geom,
            global_ids=arrays['global_ids'].tolist(),seconds=time.time()-tick,offset_rule=rule)
        write_json(path,item);receipt(path,root,dict(arrays_sha256=sha(path.with_suffix('.npz'))))
        return item
    replay=sorted(cases,key=lambda r:__import__('hashlib').sha256((str(PARAMS['seed'])+r['sample_id']).encode()).hexdigest())[:PARAMS['identity_replay_questions']]
    matches=[]
    for case in replay:
        r=one(case,'zero');previous=json.loads((old/'outputs'/case['sample_id']/'zero.json').read_text())
        matches.append(dict(sample_id=case['sample_id'],text_exact=r['response_text']==previous['response_text'],ids_exact=r['generated_ids']==previous['generated_ids']))
    reuse=all(r['text_exact'] and r['ids_exact'] for r in matches)
    write_json(root/'step3_zero_replay.json',dict(reuse_original_zero=reuse,checks=matches))
    arms=[decision['chosen'],decision['matched_control_for_3A']]+([] if reuse else ['zero'])
    for i,case in enumerate(cases):
        for arm in arms:
            one(case,arm)
            status(root,'step3',state='running',branch='3A',completed_cases=i,expected_cases=len(cases),last=case['sample_id']+'/'+arm)
    write_json(root/'step3A_COMPLETE.json',dict(n=len(cases),arms=arms,reuse_original_zero=reuse,
        completed_unix=time.time(),before_submission_freeze=time.time()<datetime.fromisoformat(PARAMS['freeze_utc']).timestamp()))
    write_json(root/'step3_status.json',dict(state='complete',branch='3A',n=len(cases),chosen=decision['chosen']))


def cpu_report(root):
    subprocess.run([str(Path(__file__).resolve().parents[1]/'.venv/bin/python'),
        str(Path(__file__).with_name('report_hss_followup.py')),'--root',str(root)],check=True)


def run(root,stage):
    root=Path(root);lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=verify_plan(root)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU occupied'
    assert shutil.disk_usage('/home/ubuntu').free>50*2**30
    torch.set_num_threads(4)
    cohort={r['sample_id']:r for r in json.loads((root/'cohort.json').read_text())}
    maps=dict(np.load(root/'maps.npz'));initial=dict(np.load(root/'vectors_initial.npz'))
    model,tok=load_model(plan['config']);evaluator=auto_select_evaluator('math')
    generation=copy.deepcopy(model.generation_config);generation.do_sample=False;generation.num_beams=1;generation.repetition_penalty=1.
    if generation.pad_token_id is None:generation.pad_token_id=tok.pad_token_id or tok.eos_token_id
    freeze(root/'generation_config.json',generation.to_dict())
    preflight(root,model,tok,cohort,generation,evaluator)
    if stage=='preflight':return
    if stage in ['all','step1']:
        step1(root,plan,model,tok,generation,cohort,maps,evaluator)
        cpu_report(root)
    if stage in ['all','step2']:
        vectors=build_vectors(root,plan,model,cohort,initial)
        step2(root,plan,model,tok,cohort,vectors)
        cpu_report(root)
    if stage in ['all','step3']:
        vectors=dict(np.load(root/'step2_vectors.npz'))
        stage3(root,plan,model,tok,generation,cohort,maps,vectors,evaluator)
        cpu_report(root)
    status(root,'gpu',state='complete',stage_requested=stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['preflight','step1','step2','step3','all'],default='all');a=p.parse_args()
    try:run(a.root,a.stage)
    except BaseException:
        write_json(Path(a.root)/'failure.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()))
        raise

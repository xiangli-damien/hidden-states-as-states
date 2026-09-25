"""Separately frozen HSS generalization experiments; immutable v3 numerics."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import torch

from revision_common import sha, write_json, write_npz, freeze
from hss_followup_common import (verify_plan, verify_record, receipt, token_text,
                                 sentence_boundaries, gmm_assign)
from hss_followup_tools import (stratified_half_split, iso_controls, manifold_controls,
                                subsample_token_indices)
from run_sink_energy_matched_v3 import MatchedTransform, selftest
from run_hss_followup import GeneratedPositions
from extract_revision_prefixes import load_model

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/lambda/nfs/dami/hss/sink-followup-20260925')
V3 = Path('/lambda/nfs/dami/hss/sink-energy-matched-20260925-v3')
BASE = Path('/lambda/nfs/dami/hss/sink-next-20260925')
SEED = 20260925
LAYERS = [4, 8, 12, 16, 20, 24]


def now():
    return datetime.now(timezone.utc).isoformat()


def state(root, **kw):
    write_json(root/'status.json', dict(updated_utc=now(), **kw))


def inputs():
    old = verify_plan(SOURCE)
    cfg = json.loads((V3/'plan.json').read_text())['config']
    split = json.loads((SOURCE/'split.json').read_text())
    prep = json.loads((SOURCE/'preparation.json').read_text())
    files = [SOURCE/n for n in ['plan.json','split.json','preparation.json','maps.npz','step2_vectors.npz']]
    files += [V3/n for n in ['plan.json','cases.json','directions.npz','summary.json','audit_SUCCESS.json']]
    tokens, frames = {}, []
    for folder in sorted((Path(old['config']['foundation'])/'prefixes').glob('shard_*')):
        mark=json.loads((folder/'_SUCCESS.json').read_text())
        for n in ['rows.parquet','tokens.json']:
            assert sha(folder/n)==mark['sha256'][n]
            files.append(folder/n)
        frames.append(pd.read_parquet(folder/'rows.parquet'))
        tokens.update({r['sample_id']:r for r in json.loads((folder/'tokens.json').read_text())})
    meta=pd.read_parquet(Path(old['config']['mean_cache'])/'rows.parquet')
    frame=pd.concat(frames).set_index('sample_id').loc[meta.sample_id].reset_index()
    np.testing.assert_array_equal(frame.label,meta.label)
    frame['type']=frame.category.astype(str)
    assignment=Path(old['config']['fit'])/'assignments.npz'
    frame['cluster']=np.load(assignment)['posterior'];files.append(assignment)
    frame['correct']=frame.label.astype(bool)
    cohort={}
    for row in frame.itertuples():
        cohort[row.sample_id]=dict(sample_id=row.sample_id,type=row.type,level=int(row.level),
            correct=bool(row.correct),ground_truth=str(row.ground_truth),response_text=str(row.response_text),
            prompt_text=str(row.prompt_text),**{k:v for k,v in tokens[row.sample_id].items() if k!='sample_id'})
    return old,cfg,split,prep,frame,cohort,files


def orthogonal(axis, centers):
    man,pairs=manifold_controls(centers)
    original=np.concatenate([iso_controls(len(axis)),man])
    unit=axis/np.linalg.norm(axis)
    orth=original-(original@unit)[:,None]*unit
    norms=np.linalg.norm(orth,axis=1)
    assert (norms>1e-6).all()
    orth/=norms[:,None]
    assert abs(orth@unit).max()<1e-6
    return orth,pairs


def prepare(experiment):
    root=BASE/f'exp{experiment}';root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        for p,h in plan['files'].items():assert sha(p)==h,p
        return root,plan
    old,cfg,split,prep,frame,cohort,files=inputs()
    maps=dict(np.load(SOURCE/'maps.npz'));v3=dict(np.load(V3/'directions.npz'))
    sink=prep['sink_local'];idindex={s:i for i,s in enumerate(frame.sample_id)}
    cache=Path(old['config']['mean_cache']);specs=[];arrays={};notes={}
    basecases=json.loads((V3/'cases.json').read_text())
    def means(L,ids):
        p=cache/f'layer_{L}.npy'
        if p not in files:files.append(p)
        return np.load(p,mmap_mode='r')[[idindex[s] for s in ids]].astype(np.float64).mean(0)
    if experiment==1:
        sinks=sorted(c['question_id'] for c in basecases if c['set']=='sink')
        order=np.random.default_rng(SEED).permutation(sinks).tolist()
        remaining=set(split['N_B']);pending=list(order);pairs=[]
        rng=np.random.default_rng(SEED)
        for gap in [0,1]:
            for sid in list(pending):
                s=cohort[sid]
                pool=sorted(q for q in remaining if cohort[q]['type']==s['type'] and abs(cohort[q]['level']-s['level'])==gap)
                if not pool:continue
                q=str(rng.choice(pool));remaining.remove(q);pending.remove(sid)
                pairs.append(dict(sink=sid,matched=q,level_gap=gap,type=s['type'],sink_level=s['level'],matched_level=cohort[q]['level']))
        assert len({p['matched'] for p in pairs})==len(pairs)
        assert not ({p['matched'] for p in pairs}&set(split['N_A']+split['S_A']))
        cases=[dict(question_id=p[g],set=g,pair=p['sink'],half=True) for p in pairs for g in ['sink','matched']]
        specs=[dict(name='matched',layer=14,cases=cases,conditions=['NONE','C1'],tau_ids=[])]
        arrays.update(matched_axis=v3['axis'],matched_tau=v3['tau'],matched_controls=v3['orthogonal_controls'])
        notes=dict(pairs=pairs,unmatched=pending,exact=sum(p['level_gap']==0 for p in pairs),relaxed=sum(p['level_gap']==1 for p in pairs))
    elif experiment==2:
        for L in LAYERS:
            name=f'layer{L}'
            axis=means(L,split['S_A'])-means(L,split['N_A']);axis/=np.linalg.norm(axis)
            gids=maps[f'l{L}_global'];keep=np.flatnonzero(gids!=prep['sink_global'])
            ctrl,pairs=orthogonal(axis,maps[f'l{L}_means'][keep])
            arrays[name+'_axis']=axis;arrays[name+'_controls']=ctrl
            specs.append(dict(name=name,layer=L,cases=[dict(c,half=False) for c in basecases],conditions=cfg['conditions'],tau_ids=split['tau_ids'],reuse_tau_indices=True,
                cosine_axis14=float(axis@v3['axis']/np.linalg.norm(v3['axis'])),excluded_centroids=np.flatnonzero(gids==prep['sink_global']).tolist(),control_centroid_ids=keep.tolist(),control_pairs=pairs))
    elif experiment==3:
        train=frame[frame.split.eq('train')]
        counts=train[train.cluster.ne(sink)].groupby('cluster').size()
        chosen=sorted(counts.index,key=lambda k:(-counts[k],int(k)))[:4]
        notes['chosen_states']=list(map(int,chosen))
        for k in chosen:
            name=f'state{k}';inn=train[train.cluster.eq(k)];out=train[~train.cluster.isin([k,sink])]
            ia,ib=stratified_half_split(inn.sample_id,inn.type)
            oa,ob=stratified_half_split(out.sample_id,out.type)
            rng=np.random.default_rng(SEED)
            in_ids=sorted(rng.choice(sorted(ib),min(100,len(ib)),replace=False).tolist())
            out_ids=sorted(rng.choice(sorted(ob),min(100,len(ob)),replace=False).tolist())
            tau_ids=np.random.default_rng(SEED+4).choice(sorted(oa),min(300,len(oa)),replace=False).tolist()
            axis=means(14,sorted(ia))-means(14,sorted(oa));axis/=np.linalg.norm(axis)
            keep=[j for j in range(len(maps['l14_means'])) if j not in [k,sink]]
            ctrl,pairs=orthogonal(axis,maps['l14_means'][keep])
            arrays[name+'_axis']=axis;arrays[name+'_controls']=ctrl
            specs.append(dict(name=name,layer=14,cases=[dict(question_id=s,set=g,half=True) for g,ids in [('in',in_ids),('out',out_ids)] for s in ids],
                conditions=cfg['conditions'],tau_ids=tau_ids,reuse_tau_indices=False,split=dict(in_A=sorted(ia),in_B=sorted(ib),out_A=sorted(oa),out_B=sorted(ob)),
                cluster=int(k),training_count=len(inn),training_correctness=float(inn.correct.mean()),training_types=inn.type.value_counts().to_dict(),control_centroid_ids=keep,control_pairs=pairs))
    else:raise ValueError(experiment)
    needed={c['question_id'] for s in specs for c in s['cases']}|{q for s in specs for q in s['tau_ids']}
    write_json(root/'cohort.json',[cohort[q] for q in sorted(needed)])
    write_json(root/'specs.json',specs);write_json(root/'notes.json',notes);write_npz(root/'initial.npz',**arrays)
    code=['run_sink_next.py','run_sink_energy_matched_v3.py','run_hss_followup.py','hss_followup_tools.py','hss_followup_common.py','revision_common.py','extract_revision_prefixes.py','report_sink_next.py','report_sink_energy_matched.py']
    files += [REPO/'scripts'/n for n in code]+[REPO/'docs/sink-next-protocol-20260925.zh-CN.md']
    files += [root/n for n in ['cohort.json','specs.json','notes.json','initial.npz']]
    if experiment==2:
        for q in split['tau_ids']:
            p=SOURCE/'vector_samples/N_A'/(q+'.npz');verify_record(p,SOURCE);files.append(p)
    plan=dict(experiment=experiment,config=cfg,files={str(p):sha(p) for p in files},frozen_utc=now(),
        git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primary_ci={1:.95,2:1-.05/6,3:.9875}[experiment],bootstrap_replicates=100000,seed=SEED,
        cutoff_utc='2026-09-26T01:59:00+00:00',expected_records=sum(len(s['cases'])*len(s['conditions']) for s in specs),
        original_split_sha256=sha(SOURCE/'split.json'),interpretation='Exploratory; reused questions and pre-existing transductive maps.')
    freeze(root/'plan.json',plan);write_json(root/'FREEZE.json',dict(plan_sha256=sha(root/'plan.json'),frozen_utc=plan['frozen_utc']))
    return root,plan


@torch.inference_mode()
def forward(model,row,L,transform=None,capture=False,indices=None):
    ids=torch.tensor([row['prompt_ids']+row['response_ids']],device=model.device)
    n=len(row['prompt_ids']);found={}
    hook=GeneratedPositions(n,transform) if transform else None
    def grab(module,args,out):
        x=(out[0] if isinstance(out,tuple) else out)[0,n:]
        if indices is not None:x=x[indices]
        found['x']=x.float().cpu().numpy()
    handle=model.model.layers[L-1].register_forward_hook(grab if capture else hook) if (capture or hook) else None
    try:result=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:
        if handle:handle.remove()
    if capture:return found['x']
    h=result.last_hidden_state[0,n-1:-1];targets=torch.tensor(row['response_ids'],device=model.device);lp=[]
    for j in range(0,len(targets),64):
        logits=model.lm_head(h[j:j+64]).float()
        lp.append((logits.gather(1,targets[j:j+64,None]).squeeze(1)-logits.logsumexp(-1)).cpu().numpy())
    if hook:assert hook.n==len(targets)
    return np.concatenate(lp),hook.metrics() if hook else None


def half_cut(tok,row):
    text,offsets,core,rule=token_text(tok,row['response_ids'])
    assert core>=2,(row['sample_id'],'Empty/short response')
    bounds=sentence_boundaries(text,offsets[:core],core)
    inner=[b for b in bounds if 0<b<core]
    cut=min(inner,key=lambda x:(abs(x-core/2),x)) if inner else core//2
    return cut,'half_sentence' if inner else 'half_token'


def build_vectors(root,plan,model,cohort,specs,initial):
    path=root/'vectors.npz'
    if path.exists():verify_record(path,root);return dict(np.load(path))
    arrays=dict(initial)
    for spec in specs:
        name=spec['name']
        if name+'_tau' in arrays:continue
        rng=np.random.default_rng(SEED+5);projections=[]
        for i,q in enumerate(spec['tau_ids']):
            if spec['reuse_tau_indices']:
                with np.load(SOURCE/'vector_samples/N_A'/(q+'.npz')) as z:ix=z['indices'].copy()
            else:ix=subsample_token_indices(len(cohort[q]['response_ids']),rng)
            p=root/'tau_samples'/name/(q+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                verify_record(p,root)
                with np.load(p) as z:np.testing.assert_array_equal(ix,z['indices']);x=z['x'].copy()
            else:
                x=forward(model,cohort[q],spec['layer'],capture=True,indices=ix)
                write_npz(p,x=x,indices=ix);receipt(p,root)
            projections.extend((x@arrays[name+'_axis'].astype(np.float32)).tolist())
            state(root,phase='threshold',spec=name,completed=i+1,expected=len(spec['tau_ids']))
        arrays[name+'_tau']=np.percentile(projections,75)
    write_npz(path,**arrays);receipt(path,root)
    write_json(root/'vectors_FROZEN.json',dict(utc=now(),sha256=sha(path),plan_sha256=sha(root/'plan.json')))
    return arrays


def run(experiment):
    root,plan=prepare(experiment)
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
    cfg=plan['config'];torch.set_num_threads(4);start=time.time()
    state(root,phase='loading',expected=plan['expected_records'])
    model,tok=load_model(cfg);selftest(cfg,model.device)
    cohort={r['sample_id']:r for r in json.loads((root/'cohort.json').read_text())}
    specs=json.loads((root/'specs.json').read_text());initial=dict(np.load(root/'initial.npz'))
    vectors=build_vectors(root,plan,model,cohort,specs,initial)
    # Identity and actual index14 replay check precede new outcome comparisons.
    from run_hss_followup import logprobs
    probe=cohort[specs[0]['cases'][0]['question_id']]
    probe=dict(probe,response_ids=probe['response_ids'][:32])
    reference,_=logprobs(model,probe['prompt_ids'],probe['response_ids'])
    for L in sorted({s['layer'] for s in specs}):
        identity,_=forward(model,probe,L,lambda h:h)
        np.testing.assert_array_equal(identity,reference)
    write_json(root/'preflight_SUCCESS.json',dict(identity_logprobs_exact=True,layers=[s['layer'] for s in specs],torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    records=[]
    for spec in specs:
        name=spec['name'];axis=vectors[name+'_axis'];tau=float(vectors[name+'_tau']);orth=vectors[name+'_controls']
        for case in spec['cases']:
            sid=case['question_id'];row=cohort[sid]
            cut,rule=half_cut(tok,row) if case['half'] else (case['cut'] if case['set']=='sink' else 0,case['cut_rule'] if case['set']=='sink' else 'whole')
            for ci,cond in enumerate(spec['conditions']):
                p=root/'records'/name/sid/(cond+'.json');p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():
                    verify_record(p,root);item=json.loads(p.read_text());assert sha(p.with_suffix('.npz'))==item['arrays_sha256']
                else:
                    tick=time.time();op=None if cond=='NONE' else MatchedTransform(axis,tau,None if cond=='C1' else orth[ci-2],cfg,model.device)
                    lp,geometry=forward(model,row,spec['layer'],op)
                    data=dict(logprobs=lp)
                    if op:data.update(op.arrays)
                    write_npz(p.with_suffix('.npz'),**data)
                    item=dict(spec=name,layer=spec['layer'],question_id=sid,set=case['set'],condition=cond,
                        cut=int(cut),cut_rule=rule,n_tokens=len(lp),nll=float(-lp[cut:].mean(dtype=np.float64)),
                        seconds=time.time()-tick,geometry=geometry,energy_match=op.metrics if op else None,
                        ideal_exceedance_fraction=float((op.arrays['scale']>0).mean()) if cond=='C1' else None,
                        mean_positive_excess=float(op.arrays['scale'].mean()) if cond=='C1' else None,
                        arrays_sha256=sha(p.with_suffix('.npz')))
                    write_json(p,item);receipt(p,root,dict(arrays_sha256=item['arrays_sha256']))
                records.append(item)
                state(root,phase='scoring',completed=len(records),expected=plan['expected_records'],last=name+'/'+sid+'/'+cond,elapsed_seconds=time.time()-start)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['geometry','energy_match']} for r in records]).to_parquet(root/'scores.parquet',index=False)
    write_json(root/'GPU_COMPLETE.json',dict(utc=now(),records=len(records),seconds=time.time()-start,table_sha256=sha(root/'scores.parquet'),plan_sha256=sha(root/'plan.json')))
    state(root,phase='gpu_complete',completed=len(records),expected=plan['expected_records'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=int,choices=[1,2,3],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    try:
        if a.prepare_only:
            root,plan=prepare(a.experiment);print(json.dumps(dict(root=str(root),records=plan['expected_records'],frozen=plan['frozen_utc'])))
        else:run(a.experiment)
    except BaseException:
        root=BASE/f'exp{a.experiment}';root.mkdir(parents=True,exist_ok=True)
        write_json(root/'failure.json',dict(utc=now(),traceback=traceback.format_exc()));raise

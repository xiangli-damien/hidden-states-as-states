"""Resumable raw token/sentence adjacent-layer study: prepare, fit, assign."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import zarr
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.change_clusters import fit_one
from hss.analysis.local_depth import balanced_positions, segment_means, pack_bf16_exact, unpack_bf16
from hss.data.openact import sentence_boundaries
from hss.experiments.artifacts import file_digest, save_json, runtime_versions, lock


def slots(cfg):
    return sorted(set(cfg['layers'] + [l-1 for l in cfg['layers']]))


def initialize(cfg):
    root, cache = Path(cfg['output']), Path(cfg['cache'])
    root.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(Path(cfg['reference'])/'rows.parquet')
    split = pd.read_parquet(Path(cfg['reference'])/'splits.parquet')
    np.testing.assert_array_equal(rows.sample_id, split.sample_id)
    assert len(rows) == cfg['expected_samples'] and rows.sample_id.is_unique and rows.question_group.is_unique
    frozen = {k:v for k,v in cfg.items() if k not in ['read_workers','fit_workers','cpu_threads']}
    protocol = dict(config=frozen, rows_sha256=file_digest(Path(cfg['reference'])/'rows.parquet'),
        split_sha256=file_digest(Path(cfg['reference'])/'splits.parquet'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), runtime=runtime_versions(),
        normalization=False, target='Failure=1 for supervised readout only; clustering never uses outcome labels.',
        delta='h_l,t - h_l-1,t at the same response token; last decoder output is PRE final RMSNorm.',
        segmentation='Existing HSS punctuation/newline boundaries; exhaustive nonoverlapping segments including terminal EOS. These are textual segments, not annotated reasoning steps.',
        fit='Equal four sampled units per question, with replacement only for fewer than four units; train questions only. Minimum ICL over converged fits. Matched K=8 sensitivity.',
        scores='All 2690319 response tokens and all textual segments; question-normalized occupancy and transitions. No edges cross question boundaries.',
        cache='Losslessly pack exactly representable bf16 source floats into uint16; unpack to float32 before subtraction or averaging. Never quantize deltas.',
        cautions=['Exploratory reused Qwen MATH, not fresh confirmation.',
                  'ICL treats within-question sampled units as independent; effective sample-size assumptions are approximate.',
                  'Five selected layer pairs only; K<=32 is a pilot grid, not a claim of global optimum.',
                  'Whole-answer error labels cannot localize erroneous steps.',
                  'Full-response features are retrospective; no causal or early-warning claim.'])
    p = root/'protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ['config','rows_sha256','split_sha256']:
            if old[key]!=protocol[key]: raise ValueError('Protocol changed; use a new output directory')
    else:
        save_json(p,protocol);rows.to_parquet(root/'rows.parquet',index=False);split.to_parquet(root/'splits.parquet',index=False)
    return rows, split


def prepare_shard(task):
    shard,cfg,index=task
    folder=Path(cfg['cache'])/'shards'/shard;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'READY.json'
    if marker.exists():return json.loads(marker.read_text())
    start=time.monotonic(); source=Path(cfg['source'])/shard
    if not (source/'_SUCCESS').exists() or not (source/'_COPY_VERIFIED.json').exists():raise ValueError('Unverified source')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['model']['identifier']!=cfg['model']:raise ValueError('Wrong model')
    z=zarr.open_consolidated(str(source/'tensors.zarr'),mode='r')
    rows=pd.read_parquet(source/'data.parquet')
    ptr=np.asarray(z['tokens/sample_ptr'][:],dtype=np.int64)
    offsets=np.asarray(z['tokens/offsets'][:]);ids=np.asarray(z['tokens/ids'][:])
    np.testing.assert_array_equal(np.diff(ptr),rows.n_response_tokens)
    assert (rows.status==1).all() and (np.diff(ptr)>0).all()
    ss=slots(cfg);nonfinal=[l for l in ss if l!=cfg['last_layer']]
    packed=np.lib.format.open_memmap(folder/'states_bf16.npy',mode='w+',dtype=np.uint16,
                                     shape=(int(ptr[-1]),len(ss),cfg['hidden_dim']))
    for a in range(0,int(ptr[-1]),cfg['read_tokens']):
        b=min(int(ptr[-1]),a+cfg['read_tokens'])
        x=np.asarray(z['hidden_states/per_token'].oindex[a:b,nonfinal,:],dtype=np.float32)
        packed[a:b,:len(nonfinal)]=pack_bf16_exact(x)
        if cfg['last_layer'] in ss:
            packed[a:b,-1]=pack_bf16_exact(z['final_norm/pre/per_token'][a:b])
    packed.flush()
    pools={};sentence_arrays={};segments=[];sptr=[0];qidx=[];kind=np.zeros(len(ids),np.uint8)
    kind_names=['text','number','symbol','whitespace','newline','special']
    special=set(manifest['custom']['special_token_ids'])
    max_mean_error=0.;max_mean_relative_error=0.
    for i,row in enumerate(rows.itertuples()):
        qidx.append(index[row.sample_id]);lo,hi=map(int,ptr[i:i+2]);h=unpack_bf16(packed[lo:hi])
        ends=sentence_boundaries(row.response_text,offsets[lo:hi],hi-lo)
        sm=segment_means(h,ends);sptr.append(sptr[-1]+len(ends))
        for si,(a,b) in enumerate(zip([0]+ends[:-1],ends)):
            segments.append(dict(sample_id=row.sample_id,question_index=index[row.sample_id],segment=si,
                token_start=a,token_end=b,char_start=int(offsets[lo+a,0]),char_end=int(offsets[lo+b-1,1]),
                n_tokens=b-a))
        tp=balanced_positions(hi-lo,row.sample_id,cfg['fit_units_per_question'],cfg['seed'],'token_fit')
        sp=balanced_positions(len(ends),row.sample_id,cfg['fit_units_per_question'],cfg['seed'],'sentence_fit')
        for layer in cfg['layers']:
            before,after=ss.index(layer-1),ss.index(layer)
            for rep in ['state','delta']:
                token=h[:,after] if rep=='state' else h[:,after]-h[:,before]
                sent=sm[:,after] if rep=='state' else sm[:,after]-sm[:,before]
                pools.setdefault(f'token_{rep}_L{layer:02d}',[]).append(token[tp])
                pools.setdefault(f'sentence_{rep}_L{layer:02d}',[]).append(sent[sp])
                sentence_arrays.setdefault(f'{rep}_L{layer:02d}',[]).append(sent)
                old=np.load(Path(cfg['mean_cache'])/f'mean_{rep}_L{layer:02d}.npy',mmap_mode='r')[index[row.sample_id]]
                mean=token.mean(0,dtype=np.float64)
                err=float(np.max(np.abs(mean-old)));max_mean_error=max(max_mean_error,err)
                rel=float(np.linalg.norm(mean-old)/max(np.linalg.norm(old),1e-9))
                max_mean_relative_error=max(max_mean_relative_error,rel)
                # Original saved means reduce in float32; allow small numerical reduction error.
                if rel>2e-4:raise ValueError(f'Mean mismatch {row.sample_id} {rep} {layer}: {rel}')
        for t in range(lo,hi):
            a,b=offsets[t];piece=row.response_text[int(a):int(b)]
            k=('special' if int(ids[t]) in special else 'newline' if '\n' in piece else
               'whitespace' if not piece.strip() else 'number' if any(c.isdigit() for c in piece) else
               'text' if any(c.isalpha() for c in piece) else 'symbol')
            kind[t]=kind_names.index(k)
    np.savez(folder/'pool.npz',**{n:np.concatenate(v) for n,v in pools.items()})
    for n,v in sentence_arrays.items():np.save(folder/f'sentence_{n}.npy',np.concatenate(v))
    np.savez(folder/'layout.npz',token_ptr=ptr,sentence_ptr=np.array(sptr),question_index=np.array(qidx),
             token_kind=kind,token_ids=ids,token_offsets=offsets)
    pd.DataFrame(segments).to_parquet(folder/'segments.parquet',index=False)
    result=dict(shard=shard,n_questions=len(rows),n_tokens=int(ptr[-1]),n_segments=int(sptr[-1]),
        seconds=time.monotonic()-start,source_manifest_sha256=file_digest(source/'manifest.json'),
        source_receipt_sha256=file_digest(source/'_COPY_VERIFIED.json'),cache_shape=list(packed.shape),
        max_mean_error=max_mean_error,max_mean_relative_error=max_mean_relative_error)
    save_json(marker,result);print(json.dumps(result),flush=True);return result


def prepare(cfg,only_shard=None):
    rows,_=initialize(cfg);root=Path(cfg['output']);cache=Path(cfg['cache'])
    if (root/'PREPARED.json').exists():return
    info=json.loads((Path(cfg['summary_cache'])/'_SUCCESS.json').read_text());shards=info['shards']
    if only_shard:shards=[s for s in shards if s==only_shard]
    completed_tokens=sum(json.loads(p.read_text())['n_tokens'] for p in (cache/'shards').glob('*/READY.json'))
    remaining_tokens=max(0,int(rows.n_tokens.sum())-completed_tokens)
    required=remaining_tokens*len(slots(cfg))*cfg['hidden_dim']*2
    if shutil.disk_usage(cache).free<required+(cfg['min_free_gib']+20)*2**30:
        raise OSError('Insufficient SSD for lossless token cache plus sentence features and reserve')
    index={sid:i for i,sid in enumerate(rows.sample_id)}
    with ThreadPoolExecutor(cfg['read_workers']) as pool:
        records=list(pool.map(prepare_shard,[(s,cfg,index) for s in shards]))
    if only_shard:return
    views=cache/'views';views.mkdir(exist_ok=True)
    names=[f'{g}_{r}_L{l:02d}' for g in ['token','sentence'] for r in ['state','delta'] for l in cfg['layers']]
    for name in names:
        dst=np.lib.format.open_memmap(views/f'{name}.npy',mode='w+',dtype=np.float32,
            shape=(len(rows)*cfg['fit_units_per_question'],cfg['hidden_dim']))
        for s in shards:
            folder=cache/'shards'/s
            layout=np.load(folder/'layout.npz');qi=layout['question_index']
            ix=(qi[:,None]*cfg['fit_units_per_question']+np.arange(cfg['fit_units_per_question'])).ravel()
            with np.load(folder/'pool.npz') as pool:dst[ix]=pool[name]
        dst.flush()
    segments=pd.concat([pd.read_parquet(cache/'shards'/s/'segments.parquet') for s in shards],ignore_index=True)
    segments.to_parquet(root/'segments.parquet',index=False)
    save_json(root/'PREPARED.json',dict(shards=records,n_questions=len(rows),n_segments=len(segments),
                                       n_tokens=sum(r['n_tokens'] for r in records)))


def fit_view(task):
    name,cfg=task;root=Path(cfg['output']);folder=root/'fits'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    mean=name.startswith('mean_')
    path=(Path(cfg['mean_cache']) if mean else Path(cfg['cache'])/'views')/(name+'.npy')
    x=np.load(path,mmap_mode='r');split=pd.read_parquet(root/'splits.parquet').split.to_numpy()
    owners=np.arange(len(split)) if mean else np.repeat(np.arange(len(split)),cfg['fit_units_per_question'])
    tr=np.flatnonzero(split[owners]=='train');va=np.flatnonzero(split[owners]=='validation')
    train=np.asarray(x[tr],dtype=np.float64);validation=np.asarray(x[va],dtype=np.float64)
    candidates=[];models={}
    with threadpool_limits(cfg['cpu_threads']):
        for k in cfg['k_grid']:
            for seed in cfg['seeds']:
                gm,rec=fit_one(train,k,seed,cfg,folder/f'k{k}_s{seed}',tr)
                rec=dict(rec,validation_ll=float(gm.score(validation)))
                candidates.append(rec);models[k,seed]=gm
        viable=[r for r in candidates if r['converged']]
        if not viable:raise RuntimeError(f'No converged candidates for {name}')
        selected=min(viable,key=lambda r:(r['icl'],r['k'],r['seed']))
        fixed=[r for r in viable if r['k']==cfg['matched_k']]
        if not fixed:raise RuntimeError(f'No converged matched-K candidate for {name}')
        matched=min(fixed,key=lambda r:r['icl'])
        for tag,r in [('selected',selected),('matched',matched)]:joblib.dump(models[r['k'],r['seed']],folder/f'{tag}.joblib')
        result=dict(name=name,k=selected['k'],seed=selected['seed'],boundary=selected['k']==max(cfg['k_grid']),
            matched_k=matched['k'],matched_seed=matched['seed'],candidates=candidates,
            feature_sha256=file_digest(path),training_questions=int((split=='train').sum()),
            training_vectors=len(tr),normalization=False)
        save_json(folder/'selection.json',result)
    print(json.dumps(dict(fitted=name,k=result['k'],boundary=result['boundary'])),flush=True);return result


def fit(cfg):
    initialize(cfg);root=Path(cfg['output'])
    if not (root/'PREPARED.json').exists():raise RuntimeError('Prepare first')
    names=[f'{g}_{r}_L{l:02d}' for g in ['mean','sentence','token'] for r in ['state','delta'] for l in cfg['layers']]
    with ProcessPoolExecutor(cfg['fit_workers']) as pool:
        futures=[pool.submit(fit_view,(n,cfg)) for n in names]
        for f in as_completed(futures):f.result()
    save_json(root/'FITTED.json',dict(names=names,at=time.time()))


def assign_shard(task):
    shard,cfg=task;root=Path(cfg['output']);folder=root/'assignments';folder.mkdir(exist_ok=True)
    marker=folder/f'{shard}.json'
    if marker.exists():return json.loads(marker.read_text())
    start=time.monotonic();cache=Path(cfg['cache'])/'shards'/shard;layout=np.load(cache/'layout.npz')
    packed=np.load(cache/'states_bf16.npy',mmap_mode='r');ss=slots(cfg);n=len(packed)
    models={(g,r,l,tag):joblib.load(root/'fits'/f'{g}_{r}_L{l:02d}'/f'{tag}.joblib')
        for g in ['token','sentence'] for r in ['state','delta'] for l in cfg['layers'] for tag in ['selected','matched']}
    outputs={f'{tag}_token_{r}':np.empty((n,len(cfg['layers'])),np.int16)
             for tag in ['selected','matched'] for r in ['state','delta']}
    with threadpool_limits(cfg['cpu_threads']):
        for a in range(0,n,cfg['read_tokens']):
            b=min(n,a+cfg['read_tokens']);h=unpack_bf16(packed[a:b])
            for j,l in enumerate(cfg['layers']):
                for r in ['state','delta']:
                    x=h[:,ss.index(l)] if r=='state' else h[:,ss.index(l)]-h[:,ss.index(l-1)]
                    for tag in ['selected','matched']:
                        outputs[f'{tag}_token_{r}'][a:b,j]=models['token',r,l,tag].predict(x)
        for r in ['state','delta']:
            for tag in ['selected','matched']:
                outputs[f'{tag}_sentence_{r}']=np.column_stack([
                    models['sentence',r,l,tag].predict(np.load(cache/f'sentence_{r}_L{l:02d}.npy',mmap_mode='r'))
                    for l in cfg['layers']]).astype(np.int16)
    np.savez_compressed(folder/f'{shard}.npz',**outputs,token_ptr=layout['token_ptr'],sentence_ptr=layout['sentence_ptr'],
                        question_index=layout['question_index'])
    result=dict(shard=shard,n_tokens=n,seconds=time.monotonic()-start,
                sha256=file_digest(folder/f'{shard}.npz'))
    save_json(marker,result);print(json.dumps(dict(assigned=shard,**result)),flush=True);return result


def assign(cfg):
    initialize(cfg);root=Path(cfg['output'])
    if not (root/'FITTED.json').exists():raise RuntimeError('Fit first')
    info=json.loads((root/'PREPARED.json').read_text());shards=[r['shard'] for r in info['shards']]
    with ProcessPoolExecutor(cfg['read_workers']) as pool:
        records=list(pool.map(assign_shard,[(s,cfg) for s in shards]))
    with threadpool_limits(cfg['cpu_threads']):
        out={}
        for r in ['state','delta']:
            for tag in ['selected','matched']:
                out[f'{tag}_mean_{r}']=np.column_stack([
                    joblib.load(root/'fits'/f'mean_{r}_L{l:02d}'/f'{tag}.joblib').predict(
                        np.load(Path(cfg['mean_cache'])/f'mean_{r}_L{l:02d}.npy',mmap_mode='r')) for l in cfg['layers']])
        np.savez_compressed(root/'assignments'/'means.npz',**out)
    save_json(root/'ASSIGNED.json',dict(shards=records,at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/local-depth-changes.toml')
    p.add_argument('--stage',choices=['prepare','fit','assign','all'],required=True)
    p.add_argument('--only-shard',help='Smoke prepare one shard; not a complete dataset')
    args=p.parse_args();cfg=load_config(args.config)
    Path(cfg['output']).mkdir(parents=True,exist_ok=True)
    with lock(Path(cfg['output'])/'RUN.lock'):
        if args.stage in ['prepare','all']:prepare(cfg,args.only_shard)
        if args.stage in ['fit','all']:fit(cfg)
        if args.stage in ['assign','all']:assign(cfg)

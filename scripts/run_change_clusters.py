"""Prepare and fit label-free cross-layer and cross-token change clusters."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import zarr
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.change_clusters import residual_states, pair_positions, temporal_mean, gaussian_surrogate, fit_view
from hss.experiments.artifacts import file_digest, save_json, runtime_versions
from hss.route.counts import group_folds


def summary(cfg):
    root=Path(cfg['output']);cache=Path(cfg['cache']);views=cache/'views'
    root.mkdir(parents=True,exist_ok=True);views.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['summary_cache']);info=json.loads((source/'_SUCCESS.json').read_text())
    rows=pd.read_parquet(cfg['route_rows'])
    if len(rows)!=cfg['expected_samples'] or rows.sample_id.duplicated().any():raise ValueError('Wrong question coverage')
    channel=pd.read_parquet(source/'rows.parquet')
    np.testing.assert_array_equal(channel.sample_id,rows.sample_id)
    np.testing.assert_array_equal(channel.n_tokens,rows.n_tokens)
    folds=group_folds(rows.question_group)
    splits=pd.DataFrame(dict(sample_id=rows.sample_id,split=np.where(folds==0,'test',np.where(folds==1,'validation','train'))))
    cfg_scientific={k:v for k,v in cfg.items() if k not in ['fit_workers','cpu_threads','read_workers']}
    protocol=dict(config=cfg_scientific,source_rows_sha256=file_digest(Path(cfg['route_rows'])),
        summary_manifest_sha256=file_digest(source/'_SUCCESS.json'),
        model=info['model'],counts=splits.split.value_counts().to_dict(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime=runtime_versions(),normalization=False,
        depth='delta_l = mean_t(h_l,t - h_l-1,t); terminal h_L from final_norm/pre',
        temporal='delta_t = h_l,t+1 - h_l,t; final decoder pre and post RMSNorm separately; no prompt-to-response boundary',
        token_sampling='At most eight unique uniformly sampled adjacent response pairs per question; deterministic sample-id seed. No correctness-dependent sampling.',
        primary_selection='Train-only diagonal GMM, minimum ICL over converged candidates; validation-likelihood K also reported without replacing primary.',
        caution=['Previously explored dataset; exploratory result even though new GMMs fit only training questions.',
                 'Mixture components need not be distinct density modes or computational operations.',
                 'Temporal pairs are subsampled; per-question histograms are estimates, not full token paths.',
                 'One covariance-matched Gaussian realization at each prespecified layer is a diagnostic, not a Monte Carlo significance test.',
                 'Terminal pre/post temporal features are distinct representations; layer-delta last block excludes final RMSNorm.'])
    if (root/'protocol.json').exists():
        old=json.loads((root/'protocol.json').read_text())
        for key in ['config','source_rows_sha256','summary_manifest_sha256']:
            if old[key]!=protocol[key]:raise ValueError('Changed experiment protocol; use new directory')
    else:save_json(root/'protocol.json',protocol)
    if (root/'SUMMARY_READY.json').exists():return
    rows.to_parquet(root/'rows.parquet',index=False);splits.to_parquet(root/'splits.parquet',index=False)
    n,d,L=len(rows),cfg['hidden_dim'],cfg['last_layer']
    states={name:np.lib.format.open_memmap(cache/f'{name}_residual.npy',mode='w+',dtype=np.float32,shape=(n,L+1,d)) for name in ['mean','prompt_last']}
    start=0
    for shard in info['shards']:
        path=source/shard;meta=pd.read_parquet(path/'rows.parquet');end=start+len(meta)
        np.testing.assert_array_equal(meta.sample_id,rows.sample_id.iloc[start:end])
        for name in states:
            states[name][start:end]=residual_states(np.load(path/f'{name}.npy',mmap_mode='r'),np.load(path/f'pre_{name}.npy',mmap_mode='r'))
        start=end
    if start!=n:raise ValueError('Incomplete summary shards')
    for a in states.values():a.flush()
    manifest=[]
    for layer in range(1,L+1):
        name=f'mean_delta_L{layer:02d}'
        np.save(views/f'{name}.npy',states['mean'][:,layer]-states['mean'][:,layer-1]);manifest.append(dict(name=name,kind='depth_delta',layer=layer))
        if layer in cfg['probe_layers']:
            for prefix,values in [('prompt_delta',states['prompt_last'][:,layer]-states['prompt_last'][:,layer-1]),
                                  ('mean_state',states['mean'][:,layer]),('prompt_state',states['prompt_last'][:,layer])]:
                name=f'{prefix}_L{layer:02d}';np.save(views/f'{name}.npy',values);manifest.append(dict(name=name,kind=prefix,layer=layer))
    tr=splits.split.to_numpy()=='train'
    with threadpool_limits(cfg['cpu_threads']):
        for layer in cfg['gaussian_null_layers']:
            name=f'gaussian_delta_L{layer:02d}'
            x=np.load(views/f'mean_delta_L{layer:02d}.npy',mmap_mode='r')
            np.save(views/f'{name}.npy',gaussian_surrogate(x[tr],n,921+layer))
            manifest.append(dict(name=name,kind='gaussian_null',layer=layer))
    save_json(root/'depth_views.json',manifest)
    save_json(root/'SUMMARY_READY.json',dict(views=len(manifest),n=n,at=time.time()))
    print(json.dumps(dict(summary_ready=True,views=len(manifest))),flush=True)


def token_kind(piece, token_id, special):
    if int(token_id) in special:return 'special'
    if '\n' in piece:return 'newline'
    if not piece.strip():return 'whitespace'
    if any(c.isdigit() for c in piece):return 'number'
    if any(c.isalpha() for c in piece):return 'text'
    return 'symbol'


def temporal_shard(task):
    shard,cfg,index=task;cache=Path(cfg['cache'])/'temporal_shards'/shard;cache.mkdir(parents=True,exist_ok=True)
    if (cache/'complete.json').exists():return json.loads((cache/'complete.json').read_text())
    started=time.monotonic();source=Path(cfg['source'])/shard
    if not (source/'_SUCCESS').exists() or not (source/'_COPY_VERIFIED.json').exists():raise ValueError('Unverified source')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['model']['identifier']!=cfg['model']:raise ValueError('Wrong model')
    z=zarr.open_consolidated(str(source/'tensors.zarr'),mode='r')
    rows=pd.read_parquet(source/'data.parquet',columns=['sample_id','response_text','n_response_tokens','status'])
    ptr=np.asarray(z['tokens/sample_ptr'][:]);ids=np.asarray(z['tokens/ids'][:]);offsets=np.asarray(z['tokens/offsets'][:])
    np.testing.assert_array_equal(np.diff(ptr),rows.n_response_tokens)
    if not (rows.status==1).all() or np.any(np.diff(ptr)<2):raise ValueError('Incomplete/too-short response')
    special=set(manifest['custom']['special_token_ids']);pairs=[];starts=[]
    for i,row in enumerate(rows.itertuples()):
        for t in pair_positions(int(row.n_response_tokens),row.sample_id,cfg['token_pairs_per_question'],cfg['token_seed']):
            a=int(ptr[i]+t);starts.append(a)
            pieces=[]
            for ix in [a,a+1]:
                lo,hi=offsets[ix];pieces.append(row.response_text[int(lo):int(hi)])
            pairs.append(dict(question_index=index[row.sample_id],sample_id=row.sample_id,position=int(t+1),
                relative_position=float(t/max(row.n_response_tokens-2,1)),token_before=int(ids[a]),token_after=int(ids[a+1]),
                piece_before=pieces[0],piece_after=pieces[1],
                kind_before=token_kind(pieces[0],ids[a],special),kind_after=token_kind(pieces[1],ids[a+1],special)))
    starts=np.asarray(starts);ends=starts+1
    endpoints=np.r_[ptr[:-1],ptr[1:]-1]
    unique=np.unique(np.r_[starts,ends,endpoints])
    arrays={}
    for norm in ['pre','post']:
        arr=z[f'final_norm/{norm}/per_token'];sampled=np.asarray(arr.oindex[unique,:],dtype=np.float32)
        before=sampled[np.searchsorted(unique,starts)];after=sampled[np.searchsorted(unique,ends)]
        arrays[f'delta_{norm}']=after-before
        arrays[f'mean_{norm}']=temporal_mean(sampled[np.searchsorted(unique,ptr[:-1])],sampled[np.searchsorted(unique,ptr[1:]-1)],np.diff(ptr)).astype(np.float32)
    np.savez(cache/'features.npz',**arrays)
    pd.DataFrame(pairs).to_parquet(cache/'pairs.parquet',index=False)
    rows[['sample_id']].to_parquet(cache/'rows.parquet',index=False)
    result=dict(shard=shard,questions=len(rows),pairs=len(pairs),seconds=time.monotonic()-started,
        source_manifest_sha256=file_digest(source/'manifest.json'),copy_receipt_sha256=file_digest(source/'_COPY_VERIFIED.json'))
    save_json(cache/'complete.json',result)
    print(json.dumps(result),flush=True);return result


def temporal(cfg):
    root=Path(cfg['output']);cache=Path(cfg['cache']);views=cache/'views'
    if (root/'TEMPORAL_READY.json').exists():return
    rows=pd.read_parquet(root/'splits.parquet',columns=['sample_id']);index={sid:i for i,sid in enumerate(rows.sample_id)}
    info=json.loads((Path(cfg['summary_cache'])/'_SUCCESS.json').read_text())
    with ThreadPoolExecutor(cfg['read_workers']) as pool:
        records=list(pool.map(temporal_shard,[(s,cfg,index) for s in info['shards']]))
    paths=[cache/'temporal_shards'/s for s in info['shards']]
    pair=pd.concat([pd.read_parquet(p/'pairs.parquet') for p in paths],ignore_index=True)
    pair.to_parquet(cache/'pairs.parquet',index=False);pair.to_parquet(root/'pairs.parquet',index=False)
    identity=pd.concat([pd.read_parquet(p/'rows.parquet') for p in paths],ignore_index=True)
    np.testing.assert_array_equal(identity.sample_id,rows.sample_id)
    manifest=[]
    for norm in ['pre','post']:
        for typ,name in [('delta',f'token_delta_{norm}'),('mean',f'token_mean_{norm}')]:
            np.save(views/f'{name}.npy',np.concatenate([np.load(p/'features.npz')[f'{typ}_{norm}'] for p in paths]))
            manifest.append(dict(name=name,kind='temporal_pairs' if typ=='delta' else 'temporal_signed_mean',layer=cfg['last_layer'],norm=norm))
    save_json(root/'temporal_views.json',manifest)
    save_json(root/'TEMPORAL_READY.json',dict(shards=records,questions=len(rows),pairs=len(pair),at=time.time()))


def fits(cfg,kind,only):
    root=Path(cfg['output'])
    manifest=json.loads((root/f'{kind}_views.json').read_text())
    if only:manifest=[r for r in manifest if r['name'] in only]
    with ProcessPoolExecutor(cfg['fit_workers']) as pool:
        futures=[pool.submit(fit_view,(v['name'],cfg)) for v in manifest]
        for f in as_completed(futures):f.result()
    if not only:save_json(root/f'{kind.upper()}_FITTED.json',dict(names=[r['name'] for r in manifest],at=time.time()))


def validate_protocol(cfg):
    saved=json.loads((Path(cfg['output'])/'protocol.json').read_text())['config']
    now={k:v for k,v in cfg.items() if k not in ['fit_workers','cpu_threads','read_workers']}
    if saved!=now:raise ValueError('Configuration differs from frozen protocol; use a new output directory')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/change-clusters.toml')
    p.add_argument('--stage',required=True,choices=['prepare-summary','prepare-temporal','fit-depth','fit-temporal'])
    p.add_argument('--only',nargs='*')
    p.add_argument('--workers',type=int,help='Override fit worker count; does not change scientific protocol')
    p.add_argument('--threads',type=int,help='Override BLAS threads per fit worker')
    a=p.parse_args();cfg=load_config(a.config)
    for field,value in [('fit_workers',a.workers),('cpu_threads',a.threads)]:
        if value is not None:
            if value<1:raise ValueError('Worker/thread count must be positive')
            cfg[field]=value
    if a.stage=='prepare-summary':summary(cfg)
    else:
        validate_protocol(cfg)
        if a.stage=='prepare-temporal':temporal(cfg)
        else:fits(cfg,a.stage.split('-')[1],a.only)

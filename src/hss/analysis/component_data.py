"""Raw final-residual moments, independent of RMSNorm and gamma.

Caches contain no label-dependent transforms. Only final pre-RMS token states
are decompressed; collectors and all-layer token tensors are left alone.
"""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from hss.analysis.channel_data import ChannelData, load_config
from hss.experiments.artifacts import digest, save_json


def raw_moments(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid response states')
    x2 = x*x
    energy = x2.sum(1)
    if np.any(energy <= 0):
        raise ValueError('Zero token energy')
    return {'mean': x.mean(0), 'second': x2.mean(0),
            'absolute': np.abs(x).mean(0), 'q': (x2/energy[:, None]).mean(0),
            'top1': np.bincount(np.argmax(np.abs(x), axis=1), minlength=x.shape[1])/len(x)}


def centered_energy(mean, second, center):
    """Mean token squared distance per dimension; NOT distance of the mean."""
    return (second.sum(-1) - 2*mean@center + center@center)/len(center)


def extract_shard(job):
    cfg, ds, info, shard = job
    dest = Path(cfg['cache_root'])/ds['name']/shard
    key = digest({'source': info['source_keys'][info['shards'].index(shard)],
                  'model': info['model'], 'prefix': cfg['prefix_tokens'],
                  'code': inspect.getsource(raw_moments)+inspect.getsource(extract_shard)})
    marker = dest/'_SUCCESS.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['key'] != key:
            raise ValueError('Stale raw-moment cache: '+str(dest))
        return old
    started = time.monotonic()
    source = Path(ds['path'])/shard
    z = zarr.open_consolidated(str(source/'tensors.zarr'), mode='r')
    ptr = np.asarray(z['tokens/sample_ptr'][:], dtype=int)
    x = np.asarray(z['final_norm/pre/per_token'][:])
    rows = pd.read_parquet(source/'data.parquet', columns=['sample_id'])
    if len(ptr) != len(rows)+1 or ptr[0] != 0 or ptr[-1] != len(x) or np.any(np.diff(ptr)<=0):
        raise ValueError('Invalid response boundaries')
    saved = np.asarray(z['final_norm/pre/mean'][:])
    values = {}; errors = []
    for j, (a,b) in enumerate(zip(ptr[:-1],ptr[1:])):
        m = raw_moments(x[a:b])
        errors.append(float(np.max(np.abs(m['mean']-saved[j]))))
        np.testing.assert_allclose(m['mean'], saved[j], rtol=1e-4, atol=1e-5)
        prefix = raw_moments(x[a:a+cfg['prefix_tokens']]) if b-a>=cfg['prefix_tokens'] else {k: np.full_like(v,np.nan) for k,v in m.items()}
        for k,v in {**m, **{'prefix_'+k:v for k,v in prefix.items()}}.items():
            values.setdefault(k,[]).append(v)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest/'moments.npz', **{k:np.asarray(v) for k,v in values.items()})
    rows.assign(n_tokens=np.diff(ptr)).to_parquet(dest/'rows.parquet', index=False)
    result = {'key':key, 'shard':shard, 'n':len(rows), 'tokens':len(x),
              'mean_error_max':max(errors), 'seconds':time.monotonic()-started}
    save_json(marker,result)
    print(json.dumps({'dataset':ds['name'], **result}),flush=True)
    return result


def extract(cfg, limit_shards=0):
    cc=load_config(cfg['channel_config'])
    for name in cfg['datasets']:
        data=ChannelData(cc,name); ds=next(d for d in cc['datasets'] if d['name']==name)
        shards=data.info['shards'][:limit_shards] if limit_shards else data.info['shards']
        with ThreadPoolExecutor(cfg['workers']) as pool:
            results=list(pool.map(extract_shard, [(cfg,ds,data.info,s) for s in shards]))
        if not limit_shards:
            save_json(Path(cfg['cache_root'])/name/'_SUCCESS.json',{'sources':data.info,'shards':shards,'results':results})


def load_moments(cfg,data,name):
    root=Path(cfg['cache_root'])/name
    meta=json.loads((root/'_SUCCESS.json').read_text())
    if meta['shards']!=data.info['shards'] or meta['sources']['source_keys']!=data.info['source_keys']:
        raise ValueError('Raw-moment source mismatch')
    rows=pd.concat([pd.read_parquet(root/s/'rows.parquet') for s in data.info['shards']],ignore_index=True).iloc[data._indices]
    if not np.array_equal(rows.sample_id,data.rows.sample_id) or not np.array_equal(rows.n_tokens,data.rows.n_tokens):
        raise ValueError('Moment identity mismatch')
    output={}
    for s in data.info['shards']:
        with np.load(root/s/'moments.npz') as z:
            for k in z.files: output.setdefault(k,[]).append(z[k])
    return {k:np.concatenate(v)[data._indices] for k,v in output.items()}

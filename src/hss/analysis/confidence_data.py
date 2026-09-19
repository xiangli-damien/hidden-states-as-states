"""Reconstruct full-vocabulary scalars, and cache token-16 residuals on CPU."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
import zarr

from hss.analysis.channel_data import ChannelData, load_config
from hss.experiments.artifacts import save_json, file_digest, digest, runtime_versions


def checkpoint(cfg, model):
    path = Path(cfg['hf_cache'])/('models--'+model['identifier'].replace('/', '--'))/'snapshots'/model['revision']
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def read_tensor(path, name):
    from safetensors import safe_open
    index = path/'model.safetensors.index.json'
    filename = json.loads(index.read_text())['weight_map'][name] if index.exists() else 'model.safetensors'
    with safe_open(path/filename, framework='pt', device='cpu') as f:
        return f.get_tensor(name).float().numpy()


def head_weights(cfg, model):
    path = checkpoint(cfg, model)
    config = json.loads((path/'config.json').read_text())
    if config['model_type'] not in ['llama','qwen2']:
        raise ValueError('Unverified readout architecture')
    key = 'model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
    return read_tensor(path,key), read_tensor(path,'model.norm.weight'), config, path, key


def head_geometry(w, gamma, chunk=8192):
    """Float64 Gram of centered, gamma-folded head, with label-free sign."""
    mu = w.mean(0, dtype=np.float64)
    d = w.shape[1]
    gram = np.zeros((d,d), dtype=np.float64)
    for start in range(0,len(w),chunk):
        a = (w[start:start+chunk].astype(np.float64)-mu)*gamma
        gram += a.T@a
    values, vectors = eigh(gram, check_finite=False, driver='evd')
    if values[0] < -1e-7*values[-1]:
        raise ValueError('Non-PSD readout Gram')
    for j in range(vectors.shape[1]):
        if vectors[np.abs(vectors[:,j]).argmax(),j] < 0:
            vectors[:,j] *= -1
    return np.sqrt(np.maximum(values,0)), vectors


def distribution_scalars(x, w, batch=128):
    """Full-vocabulary float32 logits; stable float64 scalar reductions."""
    out = []
    for start in range(0,len(x),batch):
        z = np.asarray(x[start:start+batch],np.float32)@w.T
        top = np.partition(z,-2,axis=1)[:,-2:]
        hi,lo = top.max(1),top.min(1)
        # Shift before exp: entropy = log(sum(exp(s))) - E_p[s].
        z -= hi[:,None]
        exp = np.exp(z)
        total = exp.sum(1,dtype=np.float64)
        entropy = np.log(total)-(exp*z).sum(1,dtype=np.float64)/total
        out.append(pd.DataFrame({'entropy':entropy,'logit_margin':hi-lo,
            'probability_margin':(1-np.exp(lo-hi))/total,'top1_probability':1/total,
            'argmax':z.argmax(1),'logit_top1':hi}))
    return pd.concat(out,ignore_index=True)


def scalars(cfg):
    cc = load_config(cfg['channel_config'])
    root = Path(cfg['output_root']); root.mkdir(parents=True,exist_ok=True)
    save_json(root/'protocol.json', {'config':cfg,'source_config':cc,
        'code_sha256':file_digest(__file__),'versions':runtime_versions(),
        'interpretation':'Exploratory follow-up on previously inspected holdout; v_min is a new candidate, not an established confidence vector.'})
    last_model = None
    for ds in cc['datasets']:
        t0 = time.monotonic()
        data = ChannelData(cc,ds['name'])
        out = root/ds['name'];out.mkdir(exist_ok=True)
        key = digest({'sources':data.info['source_keys'],'config':cfg,'code':file_digest(__file__)})
        marker = out/'scalars.json'
        if marker.exists() and json.loads(marker.read_text())['key']==key:
            continue
        if data.info['model']['identifier'] != last_model:
            w,gamma,mc,path,head_key = head_weights(cfg,data.info['model'])
            last_model = data.info['model']['identifier']
            sv,basis = head_geometry(w,gamma,cfg['vocab_chunk'])
        k = max(1,w.shape[1]//100)
        np.savez(out/'readout_geometry.npz',singular_values=sv,v_min=basis[:,0],low_basis=basis[:,:k],gamma=gamma)
        result = data.rows[['sample_id','y']].copy()
        all_validation = []
        for position in ['prompt_last','t1']:
            if position == 'prompt_last':
                pre = data.array('pre_prompt_last')
                post = data.array('prompt_last', data.info['model']['n_layers']-1)
            else:
                tokens = ChannelData(cc,ds['name'],'tokens')
                assert np.array_equal(data.rows.sample_id,tokens.rows.sample_id)
                pre,post = tokens.array('pre',0),tokens.array('post',0)
            frame = distribution_scalars(post,w,cfg['logit_batch'])
            rms = np.sqrt(np.mean(pre.astype(np.float64)**2,axis=1))
            low = pre@basis[:,:k]
            frame['rms'] = rms
            frame['v_min_projection'] = low[:,0]
            frame['v_min_absolute'] = np.abs(low[:,0])
            frame['low_readout_fraction'] = np.linalg.norm(low,axis=1)/np.maximum(np.linalg.norm(pre,axis=1),1e-20)
            frame['v_min_unit_projection'] = low[:,0]/np.maximum(rms,1e-20)
            result = pd.concat([result,frame.add_prefix(position+'_')],axis=1)
            # RMSNorm computes in float32, then rounds normalization before weight multiply in BF16.
            ideal_post = pre/np.sqrt(np.mean(pre*pre,axis=1,keepdims=True)+mc['rms_norm_eps'])*gamma
            relative = np.linalg.norm(ideal_post-post,axis=1)/np.maximum(np.linalg.norm(post,axis=1),1e-20)
            all_validation.append({'position':position,'ideal_fp32_norm_relative_error_mean':float(relative.mean()),
                'ideal_fp32_norm_relative_error_max':float(relative.max())})
            print(json.dumps({'dataset':ds['name'],'position':position,'n':len(frame),'seconds':time.monotonic()-t0}),flush=True)
        result.to_parquet(out/'scalars.parquet',index=False)
        actual = data.rows.first_token.to_numpy(int)
        predicted = result.prompt_last_argmax.to_numpy(int)
        # Exact BF16 rounding of FP32 matrix product can create ties: explicit diagnostic, not silently replaced logits.
        z = data.array('prompt_last',data.info['model']['n_layers']-1)
        actual_logits = np.einsum('ij,ij->i',z,w[actual])
        gaps = result.prompt_last_logit_top1.to_numpy()-actual_logits
        pd.DataFrame({'sample_id':data.rows.sample_id,'first_token':actual,'argmax':predicted,'actual_to_top_logit_gap':gaps}).to_parquet(out/'alignment.parquet',index=False)
        save_json(marker,{'key':key,'n':len(result),'model':data.info['model'],'head_key':head_key,
            'model_config_sha256':file_digest(path/'config.json'),'geometry_k':k,
            'singular_min':float(sv[0]),'singular_second':float(sv[1]),'singular_max':float(sv[-1]),
            'min_to_median_ratio':float(sv[0]/np.median(sv)),
            'first_token_argmax_agreement':float(np.mean(actual==predicted)),
            'mismatch_n':int(np.sum(actual!=predicted)),
            'mismatch_gap_max':float(gaps[actual!=predicted].max()) if np.any(actual!=predicted) else 0.,
            'norm_checks':all_validation,'elapsed_seconds':time.monotonic()-t0,
            'precision':'Saved BF16 states and BF16 checkpoint weights, promoted to float32 CPU matmul; not native BF16 logits.'})


def layer_shard(job):
    ds,shard,root,position = job
    dest = root/ds['name']/shard
    marker = dest/'_SUCCESS.json'
    if marker.exists():
        return json.loads(marker.read_text())
    source = Path(ds['path'])/shard
    if not (source/'_COPY_VERIFIED.json').exists():
        raise ValueError('Unverified source')
    start = time.monotonic()
    z = zarr.open_consolidated(str(source/'tensors.zarr'),mode='r')
    ptr = z['tokens/sample_ptr'][:]
    valid = np.diff(ptr)>=position
    positions = ptr[:-1][valid]+position-1
    a = z['hidden_states/per_token']
    values = np.full((len(valid),a.shape[1],a.shape[2]),np.nan,dtype=np.float32)
    values[valid] = a.oindex[positions,:,:]
    values[valid,-1,:] = z['final_norm/pre/per_token'].oindex[positions,:]
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'layers.npy',values,allow_pickle=False)
    meta = {'position':position,'n':len(valid),'valid_n':int(valid.sum()),
        'manifest_sha256':file_digest(source/'manifest.json'),'data_sha256':file_digest(source/'data.parquet'),
        'seconds':time.monotonic()-start,'final_layer':'pre_RMS'}
    save_json(marker,meta)
    print(json.dumps({'dataset':ds['name'],'shard':shard,**meta}),flush=True)
    return meta


def layers(cfg):
    cc = load_config(cfg['channel_config']); root = Path(cfg['cache_root'])/'layers'
    for ds in cc['datasets']:
        if ds['name'] not in cfg['layer_datasets']:
            continue
        data = ChannelData(cc,ds['name'])
        jobs = [(ds,s,root,cfg['layer_position']) for s in data.info['shards']]
        with ThreadPoolExecutor(cfg['layer_workers']) as pool:
            metas = list(pool.map(layer_shard,jobs))
        save_json(root/ds['name']/'_SUCCESS.json',{'shards':data.info['shards'],'position':cfg['layer_position'],'meta':metas})


def layer_array(cfg,data,name,layer):
    root = Path(cfg['cache_root'])/'layers'/name
    if not (root/'_SUCCESS.json').exists():
        raise FileNotFoundError('All-layer extraction has not completed: '+name)
    return np.concatenate([np.asarray(np.load(root/s/'layers.npy',mmap_mode='r')[:,layer,:]) for s in data.info['shards']])[data._indices]

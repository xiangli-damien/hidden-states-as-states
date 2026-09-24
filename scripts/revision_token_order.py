"""Does within-window state order add beyond occupancy, mean, prompt and tokens?

All methods use the same 16 visible positions and question splits. Codebooks
remain frozen. LR standardization is train-only, separate from raw clustering.
Question-specific shuffles preserve occupancy exactly and are applied before
both training and evaluation; no test-time-only distribution shift is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from threadpoolctl import threadpool_limits

from fit_revision_geometry import read_view
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status,nearest
from revision_conditional_information import fit_readout
from revision_statistics import paired_auc_ci


def categorical_features(codes,k,kind):
    codes=np.asarray(codes,dtype=np.int64)
    if codes.ndim!=2 or codes.shape[1]<2 or not ((codes>=0)&(codes<k)).all():
        raise ValueError('Invalid fixed-window state codes')
    n,t=codes.shape
    if kind=='occupancy':
        rows=np.repeat(np.arange(n),t);cols=codes.ravel();dim=k;weight=1/t
    elif kind=='ordered':
        rows=np.repeat(np.arange(n),t);cols=(codes+np.arange(t)*k).ravel();dim=t*k;weight=1
    elif kind=='transitions':
        rows=np.repeat(np.arange(n),t-1);cols=(codes[:,:-1]*k+codes[:,1:]).ravel();dim=k*k;weight=1/(t-1)
    else:raise ValueError(kind)
    result=sparse.coo_matrix((np.full(len(rows),weight,np.float64),(rows,cols)),shape=(n,dim)).tocsr()
    result.sum_duplicates();return result


def shuffle_questions(codes,ids,seed):
    out=[]
    for sequence,sid in zip(codes,ids):
        key=int(hashlib.sha256(f'token-order-v1/{seed}/{sid}'.encode()).hexdigest()[:16],16)
        out.append(np.random.default_rng(key).permutation(sequence))
    return np.stack(out)


def encode_training_tokens(ids,train):
    vocab=np.unique(ids[train]);position=np.searchsorted(vocab,ids)
    known=(position<len(vocab))
    known &= vocab[np.minimum(position,len(vocab)-1)]==ids
    encoded=np.where(known,position,len(vocab))
    return encoded,vocab,known


def load_token_ids(source,frame,prefix,role):
    rows={}
    for f in (source/'prefixes').glob('shard_*/tokens.json'):
        for row in json.loads(f.read_text()):rows[row['sample_id']]=row
    result=[]
    for sid in frame.sample_id:
        row=rows[sid];ids=row['prompt_ids']+row['response_ids'][:prefix]
        positions=row['question_positions'][-16:] if role=='question_tokens' else range(len(ids)-16,len(ids))
        result.append([ids[i] for i in positions])
    values=np.asarray(result,np.int64)
    assert values.shape==(len(frame),16)
    return values


def run_one(job):
    with threadpool_limits(limits=job[0]['blas_threads']):return _run_one(job)


def _run_one(job):
    cfg,prefix,layer,role=job;source=Path(cfg['source_root']);name=f'p{prefix}_l{layer}_{role}'
    dest=Path(cfg['output'])/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'_SUCCESS.json').exists():return json.loads((dest/'summary.json').read_text())
    source_cfg=json.loads((source/'prefixes/plan.json').read_text())['config']
    frame,x=read_view(source_cfg,prefix,layer,role)
    original,prompt=read_view(source_cfg,0,layer,'last')
    index=original.set_index('sample_id').index.get_indexer(frame.sample_id)
    if np.any(index<0):raise ValueError('Prompt question missing')
    np.testing.assert_array_equal(original.iloc[index].label,frame.label)
    np.testing.assert_array_equal(original.iloc[index].split,frame.split)
    prompt=prompt[index]
    if frame.sample_id.duplicated().any() or frame.question_group.duplicated().any():
        raise ValueError('Question split duplication')
    train=frame.split.eq('train').to_numpy();val=frame.split.eq('validation').to_numpy();test=frame.split.eq('test').to_numpy()
    y=1-frame.label.to_numpy(int)
    decoder_path=Path(cfg['token_decoders'])/name/'decoder.npz'
    marker=json.loads((decoder_path.parent/'_SUCCESS.json').read_text())
    if sha(decoder_path)!=marker['decoder_sha256']:raise ValueError('Frozen codebook checksum mismatch')
    with np.load(decoder_path) as decoder:centers=decoder['centers'].copy()
    codes=nearest(x.reshape(-1,x.shape[-1]),centers).reshape(x.shape[:2]);k=len(centers)
    token_ids=load_token_ids(source,frame,prefix,role)
    encoded,vocab,known=encode_training_tokens(token_ids,train)
    mean=x.mean(1);last=x[:,-1].copy();del x
    write_npz(dest/'question_codes.npz',sample_ids=frame.sample_id.to_numpy(str),codes=codes,
              token_ids=token_ids,token_vocabulary=vocab,centers=centers)
    scalers={}
    def scale(name,data):
        scaler=StandardScaler(with_mean=not sparse.issparse(data)).fit(data[train])
        scalers[name]={'mean':scaler.mean_,'scale':scaler.scale_}
        return scaler.transform(data)
    confidence=pd.read_parquet(source/'confidence'/f'prefix_{prefix}.parquet').set_index('sample_id').loc[frame.sample_id]
    numeric=np.column_stack([np.log1p(frame.n_prompt_tokens.to_numpy(float)),np.linalg.norm(mean,axis=1),
        np.linalg.norm(last,axis=1),np.linalg.norm(prompt,axis=1),confidence.next_token_entropy,confidence.next_token_logit_margin])
    categories=frame[['category','level']].fillna('unknown').astype(str)
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True).fit(categories[train])
    controls=sparse.hstack([sparse.csr_matrix(scale('numeric_controls',numeric)),
                           scale('categorical_controls',encoder.transform(categories))],format='csr')
    prompt_scaled=sparse.csr_matrix(scale('prompt',prompt))
    occupancy=scale('occupancy',categorical_features(codes,k,'occupancy'))
    ordered=scale('ordered',categorical_features(codes,k,'ordered'))
    def joint(*args):return sparse.hstack(args,format='csr')
    families={
        'mean_continuous':scale('mean_continuous',mean),
        'last_continuous':scale('last_continuous',last),
        'state_occupancy':occupancy,'state_ordered':ordered,
        'state_transitions':scale('transitions',categorical_features(codes,k,'transitions')),
        'token_occupancy':scale('token_occupancy',categorical_features(encoded,len(vocab)+1,'occupancy')),
        'token_ordered':scale('token_ordered',categorical_features(encoded,len(vocab)+1,'ordered')),
        'controls':controls,'controls_plus_occupancy':joint(controls,occupancy),
        'controls_plus_ordered':joint(controls,ordered),
        'prompt_controls':joint(prompt_scaled,controls),
        'prompt_controls_plus_occupancy':joint(prompt_scaled,controls,occupancy),
        'prompt_controls_plus_ordered':joint(prompt_scaled,controls,ordered)}
    for seed in cfg['shuffle_seeds']:
        shuffled=shuffle_questions(codes,frame.sample_id,seed)
        np.testing.assert_array_equal(np.sort(shuffled,axis=1),np.sort(codes,axis=1))
        label=f'state_ordered_shuffled_{seed}'
        families[label]=scale(label,categorical_features(shuffled,k,'ordered'))
        write_npz(dest/f'shuffled_codes_{seed}.npz',codes=shuffled)
    write_npz(dest/'feature_scalers.npz',**{f'{name}_{key}':value for name,parts in scalers.items() for key,value in parts.items()})
    write_json(dest/'features.json',{'K':k,'positions':16,'assignment':'nearest raw GMM center',
        'categorical_levels':[c.tolist() for c in encoder.categories_],
        'controls':['log prompt length','window mean norm','window last norm','initial prompt norm',
                    'current next-token entropy','current margin','category','difficulty'],
        'no_future_completion_length':True,'raw_clustering_unchanged':True,
        'readout_scaling':'train-only per-column standard deviations; sparse indicators uncentered',
        'state_sequence_bits':16*float(np.log2(k)),
        'token_train_vocabulary':len(vocab),'test_token_coverage':float(known[test].mean()),
        'feature_dimensions':{key:value.shape[1] for key,value in families.items()},
        'ordering_limit':'Ordered categorical LR is additive in position; transition histogram adds adjacent pair features, not a general sequence model',
        'shuffle_limit':'Per-question shuffling changes position marginals while preserving counts. This is a readout order control, not a model intervention.',
        'budget_limit':'Ordered/token/continuous feature dimensions differ; no equal-capacity superiority claim'})
    predictions=pd.DataFrame({'sample_id':frame.sample_id,'question_group':frame.question_group,'split':frame.split,'failure':y})
    metrics=[]
    for name,features in families.items():
        probability,selected=fit_readout(features[train],y[train],features[val],y[val],features,dest/name,
                                        cfg['C_grid'],cfg['max_iter'],cfg['seed'])
        predictions[name]=probability
        metrics.append({'method':name,'test_auroc':float(roc_auc_score(y[test],probability[test])),
            'test_log_loss':float(log_loss(y[test],probability[test])),
            'test_brier':float(brier_score_loss(y[test],probability[test])),
            'selected_C':selected['C'],'n_features':features.shape[1]})
    comparisons=[]
    pairs=[('state_ordered','state_occupancy'),('state_ordered','mean_continuous'),
        ('state_transitions','state_occupancy'),('state_ordered','token_ordered'),
        ('controls_plus_ordered','controls'),('controls_plus_ordered','controls_plus_occupancy'),
        ('prompt_controls_plus_ordered','prompt_controls'),
        ('prompt_controls_plus_ordered','prompt_controls_plus_occupancy')]
    pairs.extend([('state_ordered',f'state_ordered_shuffled_{seed}') for seed in cfg['shuffle_seeds']])
    for method,base in pairs:
        comparisons.append({'method':method,'baseline':base,
            **paired_auc_ci(y[test],predictions[method].to_numpy()[test],predictions[base].to_numpy()[test])})
    predictions.to_parquet(dest/'predictions.parquet',index=False)
    result={'view':f'p{prefix}_l{layer}_{role}','prefix':prefix,'block':layer,'role':role,
            'split_counts':frame.split.value_counts().to_dict(),'metrics':metrics,'paired_comparisons':comparisons,
            'scope':cfg['scope'],'unavailable':'No general nonlinear order model or causal intervention in this stage'}
    write_json(dest/'summary.json',result);write_json(dest/'_SUCCESS.json',{
        'summary_sha256':sha(dest/'summary.json'),'predictions_sha256':sha(dest/'predictions.parquet'),
        'question_codes_sha256':sha(dest/'question_codes.npz')})
    print(json.dumps({'token_order_complete':result['view']}),flush=True)
    return result


def run(cfg):
    root=Path(cfg['output']);source=Path(cfg['source_root'])
    files=[Path(__file__),Path(__file__).with_name('revision_conditional_information.py'),
           Path(__file__).with_name('revision_statistics.py'),Path(__file__).with_name('fit_revision_geometry.py'),
           source/'prefixes/plan.json']
    for prefix,layer,role in cfg['views']:
        directory=Path(cfg['token_decoders'])/f'p{prefix}_l{layer}_{role}'
        files.extend([directory/'_SUCCESS.json',directory/'summary.json'])
    freeze(root/'plan.json',provenance(cfg,files));results=[]
    status(root,'fit',state='running',completed=0,expected=len(cfg['views']))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for future in as_completed([pool.submit(run_one,(cfg,*job)) for job in cfg['views']]):
            results.append(future.result());status(root,'fit',state='running',completed=len(results),expected=len(cfg['views']))
    write_json(root/'summary.json',results);write_json(root/'_SUCCESS.json',{'views':len(results)})
    status(root,'fit',state='complete',completed=len(results),expected=len(cfg['views']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    cfg=config(parser.parse_args().config)
    try:run(cfg)
    except BaseException:
        status(cfg['output'],'fit',state='failed',traceback=traceback.format_exc());raise

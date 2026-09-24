"""Fit route detectors without loading correctness or answer metadata.

Existing maps are transductive; this is a frozen-map exploratory benchmark.
Use evaluate_route_unsupervised.py only after this script writes _SCORES_FROZEN.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.svm import OneClassSVM
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from hss.experiments.artifacts import file_digest, save_json
from hss.route.counts import group_folds
from hss.route.structure import RouteEncoder, BackoffMarkov, MixtureMarkov, LayerHMM, StableRouteLOF, aggregate_losses


def fit_classical(kind, x, train, cfg, z=None, train_z=None):
    if kind=='knn':
        m=NearestNeighbors(n_neighbors=cfg['knn_neighbors'], metric='euclidean', n_jobs=cfg['cpu_threads']).fit(train)
        score=m.kneighbors(x)[0].mean(1)
    elif kind=='lof':
        m=StableRouteLOF(n_neighbors=cfg['lof_neighbors']).fit(train_z)
        score=-m.score_samples(z)
    elif kind=='isolation_forest':
        m=IsolationForest(n_estimators=cfg['isolation_trees'], max_samples=256, random_state=920, n_jobs=cfg['cpu_threads']).fit(train)
        score=-m.score_samples(x)
    elif kind=='one_class_svm':
        m=OneClassSVM(nu=.1, gamma='scale').fit(train)
        score=-m.score_samples(x)
    elif kind=='pca':
        m=PCA(n_components=min(cfg['pca_components'],train.shape[1]-1, len(train)-1), svd_solver='randomized', random_state=920).fit(train)
        score=np.mean((x-m.inverse_transform(m.transform(x)))**2,1)
    else: raise ValueError(kind)
    return m, score[:,None]


def run(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    source=Path(args.source); config=tomllib.loads(Path(args.config).read_text())
    if args.smoke:
        config.update(epochs=2,patience=2,seeds=[920],neural_widths=[32],latent_components=[2],maps=['mfa'])
    # Column-level I/O prevents accidental access to correctness/nuisance fields.
    meta=pd.read_parquet(source/'inputs/rows.parquet',columns=['sample_id','question_group'])
    folds=group_folds(meta.question_group); split=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
    if args.smoke:
        ii=np.r_[np.where(split=='train')[0][:120],np.where(split=='validation')[0][:60],np.where(split=='test')[0][:60]]
        meta=meta.iloc[ii].reset_index(drop=True);split=split[ii]
    else: ii=np.arange(len(meta))
    tr=split=='train'; va=split=='validation'
    arrays=np.load(source/'inputs/states.npz')
    states={k:arrays[k][ii] for k in config['maps'] if k!='gmm_matched'}
    if 'gmm_matched' in config['maps']:
        states['gmm_matched']=np.load(source/'controls/matched_states.npz')['states'][ii]
    signature=dict(config=config,inputs={f:file_digest(source/f) for f in ['inputs/states.npz','inputs/rows.parquet','controls/matched_states.npz']},
        split='SHA256 canonical question group fold 0 test, fold 1 validation, folds 2-4 train',
        counts={k:int((split==k).sum()) for k in ['train','validation','test']},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        module_hashes={str(p):file_digest(p) for p in [Path(__file__),Path('src/hss/route/structure.py'),Path('src/hss/route/neural.py')]},
        scope='Exploratory frozen whole-data maps. Detector fitting excludes validation and test; map fitting did not.',
        score_direction='Higher = hypothesized failure, never flipped after correctness evaluation',
        selection='Mean unlabeled validation objective across three seeds. Classical settings fixed. Deep SVDD fixed width 64 and final epoch.',
        thresholds='Validation 90th percentile is 10% anomaly flagging, NOT a 10% false-alarm guarantee.',
        future_input='Frozen map assigns each layer; training-only encoder maps unseen states to per-layer UNK.',
        limitations=['Current MATH labels were examined in previous studies; not a fresh confirmation set.',
          'Full-response depth paths cannot identify the first incorrect reasoning token.',
          'One-class methods fit the mixed distribution, with no assumption that its majority is correct.',
          'Neural models are transparent categorical-route adaptations, not official paper reproductions.'])
    marker=root/'protocol.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous!=signature: raise ValueError('Changed protocol; use new output directory')
    else: save_json(marker,signature)
    pd.DataFrame(dict(sample_id=meta.sample_id,split=split)).to_parquet(root/'splits.parquet',index=False)
    import torch
    from hss.route.neural import fit_neural, neural_losses
    torch.set_num_threads(config['cpu_threads'])
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    records=[]; errors=[]; started=time.monotonic()
    for map_name,zraw in states.items():
        encoder=RouteEncoder().fit(zraw[tr]);z=encoder.transform(zraw);x=encoder.onehot(z)
        mapdir=root/'models'/map_name;mapdir.mkdir(parents=True,exist_ok=True)
        joblib.dump(encoder,mapdir/'encoder.joblib')
        tasks=[]
        for order in config['markov_orders']:
            tasks.append(('node' if order==0 else 'markov',dict(order=order),0))
        for kind in ['mixture_markov','hmm']:
            for components in config['latent_components']:
                for seed in config['seeds']:tasks.append((kind,dict(components=components),seed))
        for kind in ['knn','lof','isolation_forest','one_class_svm','pca']:tasks.append((kind,{},920))
        for kind in config['neural_methods']:
            for width in config['neural_widths']:
                for seed in config['seeds']:tasks.append((kind,dict(width=width),seed))
        for number,(kind,params,seed) in enumerate(tasks):
            paramkey='_'.join(f'{k}{v}' for k,v in params.items()) or 'fixed'
            name=f'{kind}__{paramkey}__s{seed}'; out=mapdir/name;out.mkdir(exist_ok=True)
            if (out/'complete.json').exists():
                record=json.loads((out/'complete.json').read_text())
                for fn,digest in record['artifacts'].items():
                    if file_digest(out/fn)!=digest:raise ValueError('Corrupt resumed candidate')
                records.append(record);continue
            t=time.monotonic();diag={}
            try:
                if kind in ['node','markov','mixture_markov','hmm']:
                    if kind in ['node','markov']: model=BackoffMarkov(encoder.sizes,**params)
                    elif kind=='mixture_markov':model=MixtureMarkov(encoder.sizes,seed=seed,**params,max_iter=8 if args.smoke else 150)
                    else:model=LayerHMM(encoder.sizes,seed=seed,**params,max_iter=8 if args.smoke else 100)
                    model.fit(z[tr]);losses=model.losses(z)
                    diag=dict(trace=getattr(model,'trace',[]),converged=getattr(model,'converged',None))
                    joblib.dump(model,out/'model.joblib')
                    val=float(losses[va].mean())
                elif kind not in config['neural_methods']:
                    model,losses=fit_classical(kind,x,x[tr],config,z,z[tr])
                    joblib.dump(model,out/'model.joblib');val=None
                else:
                    model,diag=fit_neural(z[tr],z[va],encoder.sizes,kind,params['width'],seed,config,args.device)
                    losses=neural_losses(model,z,batch=config['batch_size'],device=args.device)
                    torch.save(dict(state_dict=model.cpu().state_dict(),sizes=encoder.sizes.tolist(),kind=kind,width=params['width']),out/'model.pt')
                    val=diag['best_validation']
                if not np.isfinite(losses).all():raise ValueError('Nonfinite scores')
                np.savez_compressed(out/'scores.npz',losses=losses,**aggregate_losses(losses))
                save_json(out/'training.json',diag)
                record=dict(map=map_name,method=kind,params=params,paramkey=paramkey,seed=seed,
                    path=str(out.relative_to(root)),validation=val,seconds=time.monotonic()-t,
                    artifacts={p.name:file_digest(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'},
                    max_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device.startswith('cuda') else 0.)
                save_json(out/'complete.json',record);records.append(record)
                print(json.dumps(dict(map=map_name,task=number+1,total=len(tasks),candidate=name,seconds=round(record['seconds'],2),validation=val)),flush=True)
                del model;gc.collect()
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
            except Exception as e:
                error=dict(map=map_name,candidate=name,error=repr(e),traceback=traceback.format_exc())
                errors.append(error);save_json(out/'error.json',error);print(json.dumps(error),flush=True)
        save_json(root/'candidates.json',records);save_json(root/'errors.json',errors)
    if errors:raise RuntimeError(f'{len(errors)} failed candidates; inspect errors.json')
    # Select without ever loading labels. Every candidate remains available.
    selected=[]; scores=pd.DataFrame(dict(sample_id=meta.sample_id,split=split));variants={}
    for map_name in config['maps']:
        methods=sorted(set(r['method'] for r in records if r['map']==map_name))
        for kind in methods:
            rr=[r for r in records if r['map']==map_name and r['method']==kind]
            options=[]
            for key in sorted(set(r['paramkey'] for r in rr)):
                group=[r for r in rr if r['paramkey']==key]
                value=np.mean([r['validation'] for r in group]) if group[0]['validation'] is not None else 0.
                if kind=='deep_svdd':value=group[0]['params']['width']
                options.append((value,key,group))
                variants[f'{map_name}__{kind}__{key}']=np.mean([np.load(root/r['path']/'scores.npz')['mean'] for r in group],axis=0)
            _,key,group=min(options,key=lambda a:(a[0],a[1]))
            name=f'{map_name}__{kind}'
            for agg in ['mean','top4']:
                scores[name+'__'+agg]=np.mean([np.load(root/r['path']/'scores.npz')[agg] for r in group],axis=0)
            selected.append(dict(name=name,method=kind,map=map_name,paramkey=key,candidates=[r['path'] for r in group],
                                 validation_objective=None if group[0]['validation'] is None else float(np.mean([r['validation'] for r in group]))))
        # Equal-weight rank ensemble; calibration distribution is validation only.
        cols=[c for c in scores if c.startswith(map_name+'__') and c.endswith('__mean') and '__node__' not in c]
        percentiles=[]
        for c in cols:
            ref=np.sort(scores.loc[va,c].to_numpy());percentiles.append(np.searchsorted(ref,scores[c],side='right')/(len(ref)+1))
        scores[map_name+'__ensemble__mean']=np.mean(percentiles,axis=0)
    scores.to_parquet(root/'unlabeled_scores.parquet',index=False)
    np.savez_compressed(root/'configuration_scores.npz',**variants)
    save_json(root/'selected.json',selected)
    save_json(root/'_SCORES_FROZEN.json',dict(at=time.time(),seconds=time.monotonic()-started,candidates=len(records),
        protocol_sha256=file_digest(marker),scores_sha256=file_digest(root/'unlabeled_scores.parquet'),
        selected_sha256=file_digest(root/'selected.json'),errors=errors))
    print(json.dumps(dict(phase='FROZEN',candidates=len(records),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/route-unsupervised.toml');p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    with threadpool_limits(2):run(args)

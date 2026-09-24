"""All-layer distribution diagnostics using frozen raw Qwen-MATH response means."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, pairwise_distances
from threadpoolctl import threadpool_limits

from hss.analysis.layer_distribution import covariance_profile, partition_profile, relative_icl_choices
from hss.data import CachedStates
from hss.experiments.artifacts import file_digest, save_json, save_npz, lock, runtime_versions
from hss.experiments.fitting import load_fitted


def pair_cosines(x, pairs):
    norm=np.linalg.norm(x,axis=1); out=[]
    for start in range(0,len(pairs),512):
        i,j=pairs[start:start+512].T
        out.extend(np.sum(x[i]*x[j],axis=1)/np.maximum(norm[i]*norm[j],1e-30))
    return np.asarray(out)


def radius_controls(eigen, n, repeats, seed):
    rng=np.random.default_rng(seed); output=[]
    for _ in range(repeats):
        values=[]
        for start in range(0,n,256):
            z=rng.standard_normal((min(256,n-start),len(eigen)))
            values.extend(np.square(z)@eigen/eigen.sum())
        output.append(values)
    return np.asarray(output)


def frozen_partition(fit, data, x, layer, view, record):
    fit=Path(fit)
    for name in ['fit.json','model.npz','assignments.npz']:record(fit/name)
    model,info=load_fitted(fit)
    if 'context' in info:
        if info['context']['snapshot']!=data.info['key'] or info['context']['layer']!=layer:
            raise ValueError('Frozen fit does not correspond to this data snapshot')
    elif info['task']['layer']!=layer or info['task']['view']!=view:
        raise ValueError('MFA task provenance mismatch')
    if not info['model']['converged']:
        raise ValueError('Unconverged frozen model')
    with np.load(fit/'assignments.npz') as z:labels=z['posterior']
    if labels.shape!=(len(x),):raise ValueError('Incomplete assignments')
    probe=np.linspace(0,len(x)-1,64,dtype=int)
    np.testing.assert_array_equal(model.predict(x[probe]),labels[probe])
    return labels


def run(cfg):
    out=Path(cfg['output']);source=Path(cfg['source']);mfa=Path(cfg['mfa_source'])
    if (out/'_SUCCESS.json').exists():raise FileExistsError('Completed study is immutable')
    provenance={}
    def record(p):
        p=Path(p);provenance[str(p)]=file_digest(p)
    for p in [source/'candidate_metrics.csv',source/'protocol.json',mfa/'selected_layers.csv',Path(__file__),
              Path(__file__).parents[1]/'src/hss/analysis/layer_distribution.py']:record(p)
    candidates=pd.read_csv(source/'candidate_metrics.csv')
    candidates=candidates[(candidates.method=='gmm')&(candidates.seed==42)&(candidates.status=='complete')]
    selected_mfa=pd.read_csv(mfa/'selected_layers.csv')
    record(mfa/'latest_exports.json')
    mfa_exports=json.loads((mfa/'latest_exports.json').read_text())
    snapshots={}
    for view in ['post','pre_final']:
        p=source/'snapshots'/f'{view}.json';record(p);info=json.loads(p.read_text())
        data=CachedStates(Path(cfg['cache'])/info['key'])
        if data.info!=info or data.n_items()!=5000 or data.state_dim()!=3584:raise ValueError('Snapshot mismatch')
        record(data.path/'rows.parquet')
        exported=Path(mfa_exports[view]);record(exported/'data_snapshot.json');record(exported/'rows.parquet')
        if json.loads((exported/'data_snapshot.json').read_text())['key']!=info['key']:
            raise ValueError('MFA and GMM used different snapshots')
        np.testing.assert_array_equal(pd.read_parquet(exported/'rows.parquet').sample_id,data.meta.sample_id)
        snapshots[view]=data
    np.testing.assert_array_equal(snapshots['post'].meta.sample_id,snapshots['pre_final'].meta.sample_id)
    # Retain only non-outcome metadata in the new analysis; labels never used.
    meta=snapshots['post'].meta[['sample_id','n_tokens']].copy()
    if not meta.sample_id.is_unique:raise ValueError('Duplicated question rows')
    meta.to_parquet(out/'sample_ids.parquet',index=False)
    rng=np.random.default_rng(cfg['seed']);n=len(meta)
    subset=np.sort(rng.choice(n,cfg['silhouette_samples'],replace=False))
    i=rng.integers(0,n,cfg['pair_samples']);j=(i+rng.integers(1,n,cfg['pair_samples']))%n
    pairs=np.column_stack([i,j]);save_npz(out/'sampling.npz',silhouette_indices=subset,pairs=pairs)
    protocol=dict(config=cfg,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime=runtime_versions(),representation='All5000 question-level full-generation means; 3584 raw dimensions',
        units='0=embedding;1..27=block output;28pre=block28 before final RMSNorm;28post=after final RMSNorm',
        scope='Descriptive reused MATH5000; no correctness labels, model fitting, or causal claims',
        covariance='Exact population empirical covariance on all5000; float64 complete eigendecomposition; centered only for covariance/PCA',
        normalization='No clustering input normalization. Cosine and normalized radii are diagnostic metrics only.',
        gaussian='Three radial simulations from a single zero-mean Gaussian with fitted full covariance eigenvalues; conditional reference, not a p-value or GMM refit',
        separation='Euclidean silhouette on same1024 questions for four frozen partitions; samples and centroids were used in original fits',
        length='Remove only the linear projection on centered standardized log(1+n_tokens) for covariance diagnostic; not a causal or complete length adjustment',
        delta_icl='(ICL(K)-minICL)/(N*D), same illustrative budgets for all layers; not a correctness-tuned threshold',
        finite_sample='Five512-question subsets: sensitivity ranges, not confidence intervals')
    save_json(out/'protocol.json',protocol)
    units=[('post',l,f'L{l:02d}') for l in range(28)]+[('pre_final',28,'L28_pre'),('post',28,'L28_post')]
    metrics=[];partitions=[];icl_curves=[];subsamples=[]
    for position,(view,layer,unit) in enumerate(units):
        t=time.time();data=snapshots[view];p=data.path/f'layer_{layer}.npy';record(p)
        x=np.asarray(data.array(layer),dtype=np.float64)
        row,arrays=covariance_profile(x,meta.n_tokens.to_numpy())
        row.update(unit=unit,view=view,layer=layer,position=position)
        xc=x-arrays['mean'];radius=arrays['radius_squared']/row['trace']
        gaussian=radius_controls(arrays['eigenvalues'],n,cfg['gaussian_radius_repeats'],cfg['seed']+position)
        arrays['gaussian_normalized_radius_squared']=gaussian
        arrays['raw_pair_cosines']=pair_cosines(x,pairs)
        arrays['centered_pair_cosines']=pair_cosines(xc,pairs)
        row['centered_cosine_std']=float(arrays['centered_pair_cosines'].std())
        row['radius_q99']=float(np.quantile(radius,.99))
        row['gaussian_radius_q99_median']=float(np.median(np.quantile(gaussian,.99,axis=1)))
        row['radius_q99_gaussian_ratio']=row['radius_q99']/row['gaussian_radius_q99_median']
        for rep in range(cfg['subsample_repeats']):
            ids=np.random.default_rng(cfg['seed']+rep).choice(n,cfg['subsample_size'],replace=False)
            a=xc[ids];a=a-a.mean(0);gram=a@a.T
            subsamples.append(dict(unit=unit,repeat=rep,n=len(ids),effective_dimension=float(np.trace(gram)**2/np.square(gram).sum())))
        records=candidates[(candidates.view==view)&(candidates.layer==layer)].to_dict('records')
        if len(records)!=79 or {int(r['k']) for r in records}!=set(range(2,81)):raise ValueError('Incomplete GMM scan')
        valid=[r for r in records if r['converged'] and np.isfinite(r['icl'])];best=min(r['icl'] for r in valid)
        delta_choices=relative_icl_choices(valid,cfg['delta_icl_per_dimension_budgets'],n,x.shape[1])
        for budget,k in delta_choices.items():row[f'k_delta_{budget}']=k
        for tol in [0,.005,.01,.02,.03,.04,.05]:
            choice=min([r for r in valid if r['icl']<=best+tol*max(abs(best),1)],key=lambda r:r['k'])
            row[f'k_percent_{tol}']=int(choice['k'])
        for r in records:
            icl_curves.append(dict(unit=unit,k=r['k'],icl=r['icl'],bic=r['bic'],entropy=r['entropy'],
                converged=r['converged'],delta_per_dimension=(r['icl']-best)/(n*x.shape[1])))
        row['icl_min']=best
        mr=selected_mfa[(selected_mfa.view==view)&(selected_mfa.layer==layer)].iloc[0]
        row.update(mfa_k=int(mr.k),mfa_rank=int(mr['rank']))
        # Distances are computed once in the full-dimensional raw space.
        distances=pairwise_distances(x[subset],metric='euclidean');np.fill_diagonal(distances,0.)
        methods={}
        for name,tol in [('gmm_0pct',0.),('gmm_2pct',.02),('gmm_5pct',.05)]:
            methods[name]=min([r for r in valid if r['icl']<=best+tol*max(abs(best),1)],key=lambda r:r['k'])['fit_path']
        methods['mfa_rank16']=mr.fit_path
        for name,fit in methods.items():
            labels=frozen_partition(fit,data,x,layer,view,record);arrays[f'labels_{name}']=labels
            st=partition_profile(x,labels);ys=labels[subset]
            sil=silhouette_samples(distances,ys,metric='precomputed')
            st.update(unit=unit,method=name,silhouette_mean=float(sil.mean()),silhouette_negative_fraction=float(np.mean(sil<0)),
                      fit_path=str(fit),inference_audit_rows=64)
            partitions.append(st)
        save_npz(out/'units'/f'{unit}.npz',**arrays)
        row['seconds']=time.time()-t;metrics.append(row)
        pd.DataFrame(metrics).to_csv(out/'layer_metrics.csv',index=False)
        pd.DataFrame(partitions).to_csv(out/'partition_metrics.csv',index=False)
        pd.DataFrame(icl_curves).to_csv(out/'icl_curves.csv',index=False)
        pd.DataFrame(subsamples).to_csv(out/'subsample_sensitivity.csv',index=False)
        save_json(out/'status.json',dict(state='running',completed=len(metrics),total=len(units),last_unit=unit))
        print(json.dumps({k:row[k] for k in ['unit','effective_dimension','offdiagonal_covariance_fraction','radius_q99_gaussian_ratio','seconds']}),flush=True)
    for p,sha in provenance.items():
        if file_digest(p)!=sha:raise ValueError(f'Source changed during analysis: {p}')
    save_json(out/'provenance.json',provenance)
    save_json(out/'analysis_audit.json',dict(valid=True,units=len(metrics),partitions=len(partitions),
              inference_rows=len(partitions)*64,source_files_checked=len(provenance),
              covariance_trace_and_frobenius_spectrum_checks=True,snapshot_row_order_verified=True))
    save_json(out/'ANALYSIS_READY.json',dict(status='complete',units=len(metrics)))
    save_json(out/'status.json',dict(state='analysis_complete',completed=len(metrics),total=len(units)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'.lock'),threadpool_limits(limits=cfg['threads']):
        try:run(cfg)
        except Exception as exc:
            save_json(out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':main()

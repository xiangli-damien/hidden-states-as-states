"""Train-only fixed-partition FA, reusing the tested exact EM/SQUAREM engine.

Empirical anchors are shared with genuine local PCA. This does not train a
joint MFA partition and does not yet constitute a functional evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from revision_common import config,freeze,provenance,sha,write_json,write_npz,nearest,status


def prepare(cfg):
    from fit_revision_geometry import read_view
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        p=json.loads((root/'plan.json').read_text());assert p['config']==cfg
        for name,digest in p['files'].items():assert sha(name)==digest,name
        return
    if shutil.disk_usage('/home/ubuntu').free<50*1024**3:raise RuntimeError('SSD reserve below50GiB')
    source=Path(cfg['source_locality']);old=json.loads((source/'plan.json').read_text())
    prefix=Path(cfg['foundation'])/'prefixes';files=[]
    for name,digest in old['verified_prefix_inputs'].items():
        if name.endswith('prefix_16.npz') or name.endswith('rows.parquet'):
            assert sha(name)==digest,name;files.append(Path(name))
    frame,x=read_view({**cfg,'prefix_root':str(prefix)},cfg['prefix_tokens'],cfg['layer'],'tokens')
    train=frame.split.eq('train').to_numpy();xf=x[train].reshape(-1,x.shape[-1]).astype(np.float64)
    path=source/'decoders/p16_l14_tokens/decoder.npz';receipt=json.loads((path.parent/'_SUCCESS.json').read_text())
    assert sha(path)==receipt['decoder_sha256']
    with np.load(path) as f:decoder={k:f[k].copy() for k in f.files}
    assignment=nearest(xf,decoder['centers']);counts=np.bincount(assignment,minlength=cfg['k'])
    assert len(counts)==cfg['k']==64 and counts.min()>max(cfg['ranks'])
    means=np.stack([xf[assignment==k].mean(0) for k in range(cfg['k'])])
    # The historical PCA decoder stored anchors as float32. Both new methods
    # share exact float64 empirical means; preserve and quantify that bridge.
    np.testing.assert_array_equal(means.astype(np.float32),decoder['local_empirical_centers'])
    staging=Path(cfg['staging']);staging.mkdir(parents=True,exist_ok=True)
    np.save(staging/'training.npy',xf);np.save(staging/'assignment.npy',assignment)
    np.save(staging/'empirical_means.npy',means)
    shutil.copyfile(path,root/'source_decoder.npz')
    info={'questions':int(train.sum()),'tokens':len(xf),'hidden_size':xf.shape[1],
        'question_ids':frame.loc[train,'sample_id'].tolist(),'counts':counts.tolist(),
        'all_16_training_positions':True,'historical_float32_anchor_identity_checked':True,
        'historical_anchor_rounding_max_abs':float(np.max(np.abs(means-decoder['local_empirical_centers']))),
        'comparison_anchors':'Both new local PCA and FA use identical float64 empirical means; old float32 PCA is retained separately.'}
    write_json(root/'training_identity.json',info)
    files.extend([path,root/'source_decoder.npz',root/'training_identity.json',staging/'training.npy',
        staging/'assignment.npy',staging/'empirical_means.npy',Path(__file__),Path(__file__).with_name('revision_common.py'),
        Path(__file__).with_name('fit_revision_geometry.py'),
        Path(__file__).resolve().parents[1]/'src/hss/cluster/mfa.py',
        Path(__file__).resolve().parents[1]/'src/hss/cluster/mfa_fast.py'])
    freeze(root/'plan.json',provenance(cfg,files))


def fit_one(job):
    cfg,rank,k=job
    import torch
    from hss.cluster.mfa import MFAModel
    from hss.cluster.mfa_fast import BatchedMFAEM
    from fit_revision_geometry import basis as fit_basis
    torch.set_num_threads(cfg['blas_threads'])
    with threadpool_limits(limits=cfg['blas_threads']):
        folder=Path(cfg['output'])/f'rank_{rank}'/f'component_{k:03d}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():
            receipt=json.loads((folder/'receipt.json').read_text())
            assert sha(folder/'model.npz')==receipt['model_sha256']
            assert sha(folder/'summary.json')==receipt['summary_sha256']
            return json.loads((folder/'summary.json').read_text())
        data=np.load(Path(cfg['staging'])/'training.npy',mmap_mode='r')
        assign=np.load(Path(cfg['staging'])/'assignment.npy',mmap_mode='r');x=np.asarray(data[assign==k])
        mean=np.load(Path(cfg['staging'])/'empirical_means.npy')[k]
        basis=fit_basis(x,rank,mean,cfg['seed']).astype(float)
        residual=x-mean;variance=np.square(residual).mean(0)
        latent_variance=np.square(residual@basis.T).mean(0)
        w=basis.T*np.sqrt(np.maximum(.95*latent_variance,1e-12))
        psi=np.maximum(variance-np.square(w).sum(1),cfg['reg_covar'])
        initial=MFAModel(np.ones(1),mean[None],w[None],psi[None],reg_covar=cfg['reg_covar'])
        engine=BatchedMFAEM(x,device='cpu',chunk_size=1024,component_batch=1,reg_covar=cfg['reg_covar'])
        started=time.monotonic()
        model,opt=engine.fit(initial,max_steps=cfg['max_em_steps'],tol=cfg['em_tolerance'],
            accelerator=cfg['accelerator'],deadline=time.time()+cfg['cluster_deadline_seconds'])
        # Single-component mean must equal the common empirical anchor. Do not
        # silently recenter the FA decoder to hide a model/implementation change.
        anchor_error=float(np.max(np.abs(model.means_[0]-mean)))
        assert anchor_error<1e-7,anchor_error
        actual_ll=model.score(x);np.testing.assert_allclose(actual_ll,model.history_[-1],rtol=1e-10,atol=1e-8)
        before,one_more,_=engine.evaluate(engine.parameters(model));after,_,_=engine.evaluate(one_more,update=False)
        increment=after-before
        assert increment>=-1e-6*max(1.,abs(before))
        stationarity=abs(increment)<cfg['em_tolerance']
        converged=bool(model.converged_ and stationarity)
        write_npz(folder/'model.npz',weights=model.weights_,means=model.means_,loadings=model.loadings_,noise=model.noise_,
            pca_basis=basis,common_anchor=mean)
        record={'rank':rank,'component':k,'tokens':len(x),'converged':converged,
            'engine_converged':bool(model.converged_),'ordinary_em_stationarity_verified':stationarity,
            'additional_em_increment':increment,'mean_log_likelihood':actual_ll,'anchor_max_abs_error':anchor_error,
            'min_noise':float(model.noise_.min()),'seconds':time.monotonic()-started,'optimizer':opt,
            'history':model.history_,'model':'Single-component FA inside fixed GMM nearest partition; empirical anchor shared with local PCA'}
        write_json(folder/'summary.json',record)
        write_json(folder/'receipt.json',{'model_sha256':sha(folder/'model.npz'),'summary_sha256':sha(folder/'summary.json')})
        return record


def run(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'fit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(cfg);completed=[];total=cfg['k']*len(cfg['ranks'])
    # Rank8 first; secondary ranks only use the same training data and budget.
    for rank in cfg['ranks']:
        status(root,'fit',state='running',completed=len(completed),expected=total,current_rank=rank)
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit_one,(cfg,rank,k)) for k in range(cfg['k'])]
            for f in as_completed(futures):
                completed.append(f.result())
                status(root,'fit',state='running',completed=len(completed),expected=total,
                    converged=sum(r['converged'] for r in completed),current_rank=rank)
        rows=[r for r in completed if r['rank']==rank]
        write_json(root/f'rank_{rank}/fit_summary.json',{'components':len(rows),'converged':sum(r['converged'] for r in rows),'rows':rows})
    write_json(root/'fit_summary.json',{'fits':total,'converged':sum(r['converged'] for r in completed),
        'scope':'Fixed-partition FA training; likelihood/model audit done per component. Functional tests and joint MFA not yet run.','rows':completed})
    status(root,'fit',state='complete',completed=total,expected=total,converged=sum(r['converged'] for r in completed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);cfg=config(p.parse_args().config)
    try:run(cfg)
    except BaseException:
        status(cfg['output'],'fit',state='failed',traceback=traceback.format_exc());raise

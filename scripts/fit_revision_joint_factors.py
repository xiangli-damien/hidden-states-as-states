"""Bounded joint MFA following frozen fixed-partition FA initialization.

One GPU fitting process, after the confirmation decoder and its audits finish.
Hard/soft decoding will use the same parameters; this stage only fits/audits.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status


def initialization(root,rank,counts):
    from hss.cluster.mfa import MFAModel
    values=[];files=[]
    for k in range(len(counts)):
        folder=root/f'rank_{rank}'/f'component_{k:03d}'
        receipt=json.loads((folder/'receipt.json').read_text())
        assert sha(folder/'model.npz')==receipt['model_sha256']
        assert sha(folder/'summary.json')==receipt['summary_sha256']
        summary=json.loads((folder/'summary.json').read_text())
        if not summary['converged']:raise ValueError(f'Fixed FA rank{rank} component{k} not converged; do not silently seed the main joint fit')
        with np.load(folder/'model.npz') as a:values.append({name:a[name].copy() for name in ['means','loadings','noise']})
        files.extend([folder/'model.npz',folder/'summary.json',folder/'receipt.json'])
    return MFAModel(np.asarray(counts,float)/sum(counts),
        np.concatenate([a['means'] for a in values]),np.concatenate([a['loadings'] for a in values]),
        np.concatenate([a['noise'] for a in values]),reg_covar=1e-5),files


def run(cfg,confirmation):
    root=Path(cfg['output']);dest=root/'joint';dest.mkdir(parents=True,exist_ok=True)
    lock=(root/'joint.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'_SUCCESS.json').exists():print('Joint stage complete; no restart.');return
    deadline=time.monotonic()+6*3600
    while True:
        fit=root/'fit_status.json';confirm=confirmation/'functional_queue_status.json'
        for p in [fit,confirm]:
            if p.exists() and json.loads(p.read_text())['state']=='failed':raise RuntimeError(f'Prerequisite failed: {p}')
        if (root/'fit_summary.json').exists() and (confirmation/'functional_queue_SUCCESS.json').exists():break
        if time.monotonic()>deadline:raise TimeoutError('Prerequisite not complete')
        status(root,'joint',state='waiting',reason='FA fit and audited confirmation');time.sleep(20)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if time.monotonic()>deadline:raise TimeoutError('GPU occupied')
        status(root,'joint',state='waiting',reason='GPU occupied');time.sleep(20)
    if shutil.disk_usage('/home/ubuntu').free<50*1024**3:raise RuntimeError('SSD reserve below50GiB')
    import torch
    from hss.cluster.mfa_fast import BatchedMFAEM
    from hss.cluster.mfa import MFAModel
    torch.set_num_threads(cfg['blas_threads'])
    counts=json.loads((root/'training_identity.json').read_text())['counts']
    data=np.load(Path(cfg['staging'])/'training.npy',mmap_mode='r')
    assert sha(Path(cfg['staging'])/'training.npy')==json.loads((root/'plan.json').read_text())['files'][str(Path(cfg['staging'])/'training.npy')]
    summary=[]
    for rank in cfg['ranks']:
        folder=dest/f'rank_{rank}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():
            receipt=json.loads((folder/'receipt.json').read_text());assert sha(folder/'model.npz')==receipt['model_sha256']
            assert sha(folder/'summary.json')==receipt['summary_sha256'];summary.append(json.loads((folder/'summary.json').read_text()));continue
        initial,files=initialization(root,rank,counts)
        protocol={**cfg,'rank':rank,'initialization':'fixed-partition FA, counts/N mixture weights; same initialization for hard/soft decoder',
            'joint_max_em_steps':2000,'joint_tolerance':1e-5,'joint_deadline_seconds':7200,'device':'cuda:0',
            'chunk_size':1024,'component_batch':8,'noise':'component-specific diagonal, floor1e-5',
            'primary_rank':8,'scope':'Joint training only; no functional selection, no extra initialization chosen by target data.'}
        frozen=provenance(protocol,[Path(__file__),root/'plan.json',root/'training_identity.json',
            Path(cfg['staging'])/'training.npy',Path(__file__).resolve().parents[1]/'src/hss/cluster/mfa.py',
            Path(__file__).resolve().parents[1]/'src/hss/cluster/mfa_fast.py',*files])
        if (folder/'plan.json').exists():
            old=json.loads((folder/'plan.json').read_text());assert old['config']==frozen['config'] and old['files']==frozen['files']
        else:freeze(folder/'plan.json',frozen)
        # Preserve interrupted fits; do not restart them with a new 2h budget.
        if (folder/'checkpoint.json').exists():raise RuntimeError('Interrupted joint checkpoint exists; inspect and explicitly resume without resetting budget')
        started=time.monotonic();started_wall=time.time()
        engine=BatchedMFAEM(data,device='cuda:0',chunk_size=1024,component_batch=8,reg_covar=1e-5)
        def checkpoint(model,opt):
            write_npz(folder/'checkpoint.npz',**model.state_arrays())
            write_json(folder/'checkpoint.json',{'optimizer':opt,'history':model.history_,
                'started_unix':started_wall,'seconds':time.monotonic()-started,
                'arrays_sha256':sha(folder/'checkpoint.npz')})
            status(root,'joint',state='running',rank=rank,completed_ranks=len(summary),steps=opt['steps'],seconds=time.monotonic()-started)
        status(root,'joint',state='running',rank=rank,completed_ranks=len(summary),steps=0)
        model,opt=engine.fit(initial,max_steps=2000,tol=1e-5,accelerator='squarem',
            deadline=started_wall+7200,checkpoint=checkpoint,checkpoint_seconds=60)
        before,one_more,_=engine.evaluate(engine.parameters(model));after,_,_=engine.evaluate(one_more,update=False)
        np.testing.assert_allclose(before,model.history_[-1],rtol=1e-10,atol=1e-7)
        increment=after-before;assert increment>=-1e-6*max(1.,abs(before))
        # Independent numpy density on a fixed training subset, never target data.
        ix=np.arange(0,len(data),max(1,len(data)//256))[:256];subset=np.asarray(data[ix])
        check_engine=BatchedMFAEM(subset,device='cuda:0',component_batch=8,reg_covar=1e-5)
        gpu_ll,_,_=check_engine.evaluate(check_engine.parameters(model),update=False)
        cpu_ll=model.score(subset);np.testing.assert_allclose(cpu_ll,gpu_ll,rtol=1e-9,atol=1e-6)
        converged=bool(model.converged_ and abs(increment)<1e-5)
        write_npz(folder/'model.npz',**model.state_arrays())
        record={'rank':rank,'K':cfg['k'],'training_tokens':len(data),'converged':converged,
            'engine_converged':bool(model.converged_),'additional_em_increment':increment,
            'mean_training_log_likelihood':before,'independent_density_subset':len(ix),'density_abs_error':abs(cpu_ll-gpu_ll),
            'min_weight':float(model.weights_.min()),'min_noise':float(model.noise_.min()),
            'history':model.history_,'optimizer':opt,'seconds':time.monotonic()-started,
            'scope':'Joint MFA parameter fitting only; hard/soft functional evaluation remains a separate stage.'}
        write_json(folder/'summary.json',record)
        write_json(folder/'receipt.json',{'model_sha256':sha(folder/'model.npz'),'summary_sha256':sha(folder/'summary.json'),
            'plan_sha256':sha(folder/'plan.json')})
        summary.append(record);del check_engine,engine,model;torch.cuda.empty_cache()
    write_json(dest/'_SUCCESS.json',{'ranks':len(summary),'converged':sum(r['converged'] for r in summary),'rows':summary})
    status(root,'joint',state='complete',completed_ranks=len(summary),converged=sum(r['converged'] for r in summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--confirmation',required=True,type=Path);a=p.parse_args();cfg=config(a.config)
    try:run(cfg,a.confirmation)
    except BaseException:
        status(cfg['output'],'joint',state='failed',traceback=traceback.format_exc());raise

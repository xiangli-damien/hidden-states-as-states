"""Frozen prompt-last map: prepare, real smoke, fixed functional evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status,nearest
from promptlast_replacement_common import apply,conditions


def prepare(cfg):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from fit_revision_geometry import basis
    from revision_locality_common import derangement
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'prepare_SUCCESS.json').exists():verify(root);return
    origin=Path(cfg['foundation'])/'geometry/p0_l14_last'
    rc=json.loads((origin/'_SUCCESS.json').read_text())
    assert sha(origin/'decoder.npz')==rc['decoder_sha256'] and sha(origin/'summary.json')==rc['summary_sha256']
    summary=json.loads((origin/'summary.json').read_text())
    assert summary['selected']['k']==cfg['expected_k']==32 and summary['selected']['converged']
    assert summary['view']=='last' and summary['prefix']==0 and summary['layer']==14 and summary['normalization']=='none'
    extraction=json.loads((Path(cfg['foundation'])/'prefixes/plan.json').read_text())
    assert extraction['config']['layers']==[7,14,21,28]
    with np.load(origin/'decoder.npz') as z:d={k:z[k].copy() for k in ['centers','variances','weights','train_mean','local_basis','local_counts']}
    frames=[];xs=[];sources={};start=time.monotonic()
    for marker in sorted((Path(cfg['foundation'])/'prefixes').glob('shard_*/_SUCCESS.json')):
        receipt=json.loads(marker.read_text());sources[str(marker)]=sha(marker)
        for filename in ['rows.parquet','prefix_0.npz']:
            p=marker.parent/filename;h=sha(p);assert h==receipt['sha256'][filename];sources[str(p)]=h
        frame=pd.read_parquet(marker.parent/'rows.parquet')
        with np.load(marker.parent/'prefix_0.npz') as z:
            assert z['valid'].all();x=z['window'][:,1,-1].copy()
        assert len(x)==len(frame);frames.append(frame);xs.append(x)
    frame=pd.concat(frames,ignore_index=True);x=np.concatenate(xs);del xs
    assert x.shape==(5000,3584) and not frame.sample_id.duplicated().any()
    train=frame.split.eq('train').to_numpy();val=frame.split.eq('validation').to_numpy()
    assert train.sum()==3011 and val.sum()==1003
    with threadpool_limits(limits=cfg['threads']):
        codes=nearest(x,d['centers'])
        with np.load(origin/'assignments.npz') as z:
            np.testing.assert_array_equal(z['sample_id'],frame.sample_id.to_numpy(str));np.testing.assert_array_equal(z['nearest'],codes)
        np.testing.assert_array_equal(np.bincount(codes[train],minlength=32),d['local_counts'])
        np.testing.assert_allclose(x[train].astype(float).mean(0),d['train_mean'],rtol=1e-12,atol=1e-12)
        residual=x[train].astype(float)-d['centers'][codes[train]].astype(float)
        d['shared_basis']=basis(residual,8,np.zeros(3584))
    for b in [d['shared_basis'],*d['local_basis'][:,:8]]:np.testing.assert_allclose(b@b.T,np.eye(8),atol=2e-6)
    for seed in cfg['seeds']:d['permutation_'+str(seed)]=derangement(32,seed)
    distances=np.linalg.norm(x.astype(float)-d['centers'][codes].astype(float),axis=1)
    d['source_validation_distance_q95']=np.quantile(distances[val],.95)
    write_npz(root/'decoder.npz',**d)
    cases=json.loads(Path(cfg['cases_source']).read_text())
    assert len(cases)==96 and sum(c['dataset']=='math' for c in cases)==32
    index={sid:i for i,sid in enumerate(frame.sample_id)};selected={}
    for c in cases:
        if c['dataset']=='math':
            i=index[c['sample_id']];assert frame.iloc[i].split=='test';selected[c['sample_id']]=x[i]
    write_npz(root/'source_captures.npz',sample_ids=np.array(list(selected)),x=np.stack(list(selected.values())))
    write_npz(root/'source_geometry.npz',sample_ids=frame.sample_id.to_numpy(str),split=frame.split.to_numpy(str),codes=codes,distances=distances)
    write_json(root/'cases.json',cases);write_json(root/'source_inputs.json',sources)
    write_json(root/'fit_audit.json',{'k':32,'k_selection':summary['k_selection'],'selected':summary['selected'],
        'local_counts':d['local_counts'].tolist(),'train_questions':3011,'shared_fit':'Same-center train residual SVD only',
        'source_decoder_sha256':sha(origin/'decoder.npz'),'source_files_verified_at_prepare':len(sources),
        'source_validation_distance_q95':float(d['source_validation_distance_q95']),
        'all5000_nearest_assignments_exact':True,'local_basis_frozen':True,'gmm_refit':False,
        'seconds':time.monotonic()-start})
    files=[origin/f for f in ['decoder.npz','summary.json','_SUCCESS.json','assignments.npz']]
    files.append(Path(cfg['foundation'])/'prefixes/plan.json')
    files.extend(root/f for f in ['decoder.npz','source_captures.npz','source_geometry.npz','cases.json','source_inputs.json','fit_audit.json'])
    files.append(Path(cfg['cases_source']))
    files.extend(Path(__file__).with_name(f) for f in ['run_promptlast_replacement.py','promptlast_replacement_common.py',
        'audit_promptlast_replacement.py','report_promptlast_replacement.py','revision_common.py','evaluate_revision_locality.py','extract_revision_prefixes.py','fit_revision_geometry.py','revision_locality_common.py'])
    plan=provenance(cfg,files);plan.update(conditions=conditions(),questions=96,expected_records=768,
        smoke_ids=[next(c['sample_id'] for c in cases if c['dataset']==ds) for ds in ['math','gsm8k']],
        source_verification='All source arrays checked against extraction receipts once at preparation; downstream stages verify the frozen prepared inputs and source decoder.')
    freeze(root/'plan.json',plan);write_json(root/'prepare_SUCCESS.json',{'plan_sha256':sha(root/'plan.json')})
    print(json.dumps({'prepared':True,'questions':96,'k':32,'seconds':time.monotonic()-start}),flush=True)


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for p,h in plan['files'].items():
        if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
    return plan


def evaluate(cfg,smoke=False):
    import fcntl,subprocess,torch
    from evaluate_revision_locality import measure
    from extract_revision_prefixes import load_model
    root=Path(cfg['output']);plan=verify(root);dest=root/('smoke' if smoke else 'functional');dest.mkdir(exist_ok=True)
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'_SUCCESS.json').exists():return
    if not smoke:assert json.loads((root/'smoke/audit.json').read_text())['complete']
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    assert shutil.disk_usage('/home/ubuntu').free>50*1024**3
    cases=json.loads((root/'cases.json').read_text())
    if smoke:cases=[c for c in cases if c['sample_id'] in plan['smoke_ids']]
    with np.load(root/'decoder.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(root/'source_captures.npz') as z:source=dict(zip(z['sample_ids'].tolist(),z['x']))
    torch.set_num_threads(cfg['threads']);model,_=load_model(cfg)
    start=time.monotonic();done=0
    for c in cases:
        sid=c['sample_id'];prefix=c['prompt_ids'];reference=c['response_ids'];positions=[len(prefix)-1]
        baseline=None;clean=None
        for name in plan['conditions']:
            path=dest/'samples'/(sid+'__'+name+'.json');array=path.with_suffix('.npz')
            if not smoke and not path.exists():
                old=root/'smoke/samples'/path.name
                if old.exists():
                    path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,path);shutil.copyfile(old.with_suffix('.npz'),array)
            if path.exists():
                rec=json.loads(path.read_text());assert rec['plan_sha256']==sha(root/'plan.json') and sha(array)==rec['arrays_sha256']
                if name=='identity':
                    with np.load(array) as z:baseline=z['logp'].astype(float);clean=z['x'].copy()
                done+=1;continue
            details={}
            def transform(h):
                x=h.float().cpu().numpy()
                assert x.shape==(1,3584)
                if sid in source:np.testing.assert_array_equal(x[0],source[sid])
                if clean is not None:np.testing.assert_array_equal(x,clean)
                ideal,meta=apply(x,d,name);actual=torch.as_tensor(ideal,device=h.device,dtype=h.dtype)
                details.update(x=x,ideal=ideal,actual=actual.float().cpu().numpy(),meta=meta)
                return actual
            metrics,logp,losses=measure(model,prefix,reference,cfg['layer'],positions,transform)
            if name=='identity':
                baseline=logp.astype(float);clean=details['x'].copy()
                with torch.inference_mode():
                    plain=model(input_ids=torch.tensor([prefix],device=model.device),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1).cpu().numpy()
                np.testing.assert_array_equal(plain,logp)
            assert len(losses)>16 and baseline is not None
            delta=details['actual'].astype(float)-details['x'].astype(float)
            metrics.update(next_token_kl=float(np.sum(np.exp(baseline)*(baseline-logp.astype(float)))),
                argmax_agreement=bool(baseline.argmax()==logp.argmax()),later_nll=float(losses[16:].mean()),
                reconstruction_sse=float(np.sum(delta**2)),token_mse=float(np.mean(delta**2)),
                train_centered_energy=float(np.sum((details['x'].astype(float)-d['train_mean'])**2)))
            write_npz(array,logp=logp,reference_nll=losses,x=details['x'],ideal=details['ideal'],actual=details['actual'])
            write_json(path,{'sample_id':sid,'dataset':c['dataset'],'condition':name,'positions':positions,
                'meta':details['meta'],'metrics':metrics,'identity_plain_logits_exact':name=='identity',
                'arrays_sha256':sha(array),'plan_sha256':sha(root/'plan.json')})
            done+=1
            status(root,'smoke' if smoke else 'functional',state='running',completed=done,expected=len(cases)*8,seconds=time.monotonic()-start)
        print(json.dumps({'completed':done,'expected':len(cases)*8,'seconds':time.monotonic()-start}),flush=True)
    write_json(dest/'_SUCCESS.json',{'questions':len(cases),'conditions':done,'seconds':time.monotonic()-start,'plan_sha256':sha(root/'plan.json')})
    status(root,'smoke' if smoke else 'functional',state='complete',completed=done,expected=len(cases)*8)


def queue(cfg,path):
    import fcntl,subprocess
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=Path(__file__).resolve().parents[1]/'.venv/bin/python';gpu=Path('/lambda/nfs/dami/openact/.venv/bin/python')
    jobs=[('prepare',cpu,'run_promptlast_replacement.py',['--stage','prepare']),
          ('smoke',gpu,'run_promptlast_replacement.py',['--stage','smoke']),
          ('smoke_audit',gpu,'audit_promptlast_replacement.py',['--smoke']),
          ('functional',gpu,'run_promptlast_replacement.py',['--stage','evaluate']),
          ('audit',gpu,'audit_promptlast_replacement.py',[]),
          ('report',cpu,'report_promptlast_replacement.py',[]),
          ('statistics_audit',cpu,'audit_promptlast_replacement.py',['--report'])]
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]=str(cfg['threads'])
    state={'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    for name,python,script,extra in jobs:
        state['phase']=name;write_json(root/'queue_status.json',state)
        with (root/(name+'.log')).open('a') as log:
            child=subprocess.Popen([str(python),str(Path(__file__).with_name(script)),'--config',str(Path(path).resolve()),*extra],env=env,stdout=log,stderr=log)
            state['stages'][name]={'state':'running','pid':child.pid,'started_unix':time.time()};write_json(root/'queue_status.json',state)
            rc=child.wait()
        state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time())
        if rc:
            state.update(state='failed');write_json(root/'queue_status.json',state);raise RuntimeError(name+' failed')
    state.update(state='complete',finished_unix=time.time(),delivery='visual_review_pending');write_json(root/'queue_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','smoke','evaluate','queue'],required=True)
    a=p.parse_args();cfg=config(a.config)
    if a.stage=='prepare':prepare(cfg)
    elif a.stage=='queue':queue(cfg,a.config)
    else:evaluate(cfg,a.stage=='smoke')

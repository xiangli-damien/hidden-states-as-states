"""Sequential Qwen/Llama MMLU diagonal-GMM fits after the authorized predecessor.

All fits are on raw full-response token means. This is descriptive full-data
geometry, not a held-out correctness prediction experiment. No MFA/KMeans model
is fitted. D² seeding initializes only the diagonal mixture.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from hss.data import prepare,CachedStates
from hss.data.spec import DataSpec
from hss.cluster.gmm import GMMModel
from hss.cluster.gmm_resident import ResidentDiagonalEM
from hss.align import align_layers
from hss.types import AlignSpec
from hss.experiments.config import load as load_experiment
from hss.experiments.artifacts import digest
from hss.results import Result
from hss.results.models import save_layer
from hss.results.store import seal_result
from hss.transform.projection import Projection
from revision_common import write_json,write_npz,sha,freeze,status


def select(rows,tolerance):
    rows=[r for r in rows if r['converged'] and np.isfinite(r['icl'])]
    if not rows:raise ValueError('No converged candidate')
    best=min(r['icl'] for r in rows)
    return min((r for r in rows if r['icl']<=best+tolerance*max(abs(best),1.)),key=lambda r:(r['k'],r['icl']))


def refinement(rows,tried,cfg):
    valid=sorted((r for r in rows if r['converged']),key=lambda r:r['icl'])
    if not valid:return []
    anchors={r['k'] for r in valid[:cfg['refine_top_icl']]}
    anchors.update(select(valid,t)['k'] for t in cfg['icl_tolerances'])
    tested=sorted(tried);wanted=set()
    for k in anchors:
        i=tested.index(k)
        lo=tested[max(0,i-1)];hi=tested[min(len(tested)-1,i+1)]
        wanted.update(range(lo,hi+1))
    return sorted(wanted-set(tried))


def needs_extension(rows,cfg):
    return select(rows,0)['k']>=max(cfg['initial_k'])-cfg['boundary_margin']


def freeze_protocol(root,cfg):
    source=Path(__file__).resolve();repo=source.parents[1]
    files=[source,repo/'src/hss/cluster/gmm_resident.py',repo/'src/hss/cluster/gmm.py',
        repo/'src/hss/data/openact.py',repo/'src/hss/data/spec.py',repo/'src/hss/align.py',
        repo/'configs/base.toml',source.with_name('revision_common.py')]
    identity=dict(config=cfg,source_sha256={str(p):sha(p) for p in files},
        method='diagonal Gaussian mixture',precision='float64',fit_scope='all14042 descriptive unlabeled fitting',
        selection='ICL=BIC+2*posterior entropy; only converged fits; smallest K in relative tolerance',
        search='shared coarse grid plus dense integer neighbors of three ICL leaders and all tolerance choices, two rounds; optional upper extension',
        search_limit='Adaptive evaluated-grid selection; not an exhaustive global optimum guarantee',
        init='Three independent single-trial D-squared kmeans++ seeds, hard initial moments; no separate KMeans fit',
        no_feature_normalization=True)
    p=root/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['identity']==identity,'Source/config changed; use new namespace'
    else:freeze(p,dict(identity=identity,frozen_utc=datetime.now(timezone.utc).isoformat(),
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))


def prepare_model(root,cfg,model):
    out=root/model['key'];out.mkdir(parents=True,exist_ok=True)
    snapshots={}
    for view in ['post','pre_final'] if cfg['include_pre_final'] else ['post']:
        spec=DataSpec(paths=[str(Path(cfg['source_root'])/model['key'])],dataset_id='mmlu',
            expected_samples=cfg['samples'],expected_model=model['identifier'],representation='mean',
            final_norm='post' if view=='post' else 'pre',layers=None if view=='post' else [model['last_layer']],
            require_copy_receipt=True,min_free_gib=cfg['min_disk_free_gib'])
        data=prepare(spec,cfg['cache_root'])
        assert data.n_items()==cfg['samples'] and data.state_dim()==model['dimension']
        assert data.meta.sample_id.nunique()==cfg['samples'] and data.meta.label.notna().all()
        record=dict(path=str(data.path),key=data.info['key'],layers=data.layers(),n=data.n_items(),d=data.state_dim(),
            arrays={str(L):sha(data.path/f'layer_{L}.npy') for L in data.layers()},rows_sha256=sha(data.path/'rows.parquet'))
        freeze(out/'snapshots'/f'{view}.json',record);snapshots[view]=data
        status(root,'queue',state='preparing',model=model['key'],view=view,n=data.n_items())
    return snapshots


def wait_predecessor(root,cfg):
    prev=Path(cfg['predecessor'])
    while True:
        if (prev/'failure.json').exists():
            raise RuntimeError('Predecessor reports failure; do not skip it or start GMM')
        end=prev/'gpu_status.json';s3=prev/'step3_status.json'
        terminal=(end.exists() and json.loads(end.read_text()).get('state')=='complete'
            and (prev/'step2_COMPLETE.json').exists() and s3.exists()
            and json.loads(s3.read_text()).get('state') in ['complete','blocked_missing_artifacts','deferred_by_time_gate'])
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if terminal and not gpu:
            freeze(root/'predecessor_receipt.json',dict(gpu_status_sha256=sha(end),step3_status_sha256=sha(s3),
                plan_sha256=sha(prev/'plan.json'),completed_before_fitting=True))
            return
        status(root,'queue',state='waiting_for_predecessor',predecessor=str(prev),active_gpu_pids=gpu)
        time.sleep(30)


def load_candidate(path):
    marker=json.loads((path/'complete.json').read_text())
    for f,digest_value in marker['files'].items():assert sha(path/f)==digest_value
    return json.loads((path/'fit.json').read_text())


def candidate(engine,root,cfg,model,view,layer,k,snapshot):
    path=root/model['key']/'candidates'/view/f'layer_{layer:02d}'/f'k_{k:03d}'
    path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return load_candidate(path)
    best=None;audits=[];tick=time.time()
    for restart in range(cfg['n_init']):
        seed=cfg['seed']+1009*layer+10007*restart
        dest=path/f'restart_{restart}.npz';auditpath=path/f'restart_{restart}.json'
        status(root,'queue',state='fitting',model=model['key'],view=view,layer=layer,k=k,restart=restart)
        if auditpath.exists():
            audit=json.loads(auditpath.read_text())
            fitted=None
            if 'model' in audit:
                assert sha(dest)==audit['model_sha256']
                fitted=GMMModel.from_state(audit['model'],dict(np.load(dest)))
        else:
            start=time.time()
            try:
                fitted,metrics,_=engine.fit(k,seed,max_iter=cfg['max_iter'],tol=cfg['tol'])
                write_npz(dest,**fitted.state_arrays())
                audit=dict(seed=seed,seconds=time.time()-start,**metrics,model=fitted.config(),model_sha256=sha(dest))
            except FloatingPointError as error:
                fitted=None;audit=dict(seed=seed,seconds=time.time()-start,converged=False,error=str(error))
            write_json(auditpath,audit)
        audits.append(audit)
        if fitted is not None and fitted.converged_ and (best is None or audit['log_likelihood']>best[1]['log_likelihood']):
            best=(fitted,audit)
    record=dict(k=k,layer=layer,view=view,snapshot=snapshot,fit_path=str(path),
        restarts=audits,converged=best is not None,seconds=time.time()-tick)
    if best is not None:
        fitted,audit=best
        # Verify the saved raw-coordinate model independently against stable
        # direct-residual NumPy density, on deterministic rows of the full data.
        p=engine.parameters(fitted);ll,_,ent,prob=engine.evaluate(p,update=False,probabilities=True)
        selected=np.linspace(0,engine.n-1,min(64,engine.n),dtype=int)
        X=(engine.X[selected]+engine.offset).cpu().numpy()
        np.testing.assert_allclose(prob[selected],fitted.predict_proba(X),atol=2e-6,rtol=2e-5)
        weights=fitted.weights_;means=fitted.means_;variance=fitted.covariances_
        # Same independent posterior plus normalized score protects export coordinates.
        npar=2*k*engine.d+k-1;bic=-2*ll*engine.n+npar*np.log(engine.n)
        record.update(icl=float(bic+2*ent),bic=float(bic),entropy=ent,log_likelihood=ll*engine.n,
            n_parameters=npar,model=fitted.config(),n_iter=fitted.n_iter_)
        labels=prob.argmax(1).astype('int32');near=[]
        t=engine.t;mu=t.as_tensor(means,device=engine.device,dtype=t.float64)-engine.offset
        for i in range(0,engine.n,engine.chunk):
            x=engine.X[i:i+engine.chunk]
            d=x.square().sum(1)[:,None]-2*x@mu.T+mu.square().sum(1)[None]
            near.extend(d.argmin(1).cpu().tolist())
        counts=np.bincount(labels,minlength=k)
        record.update(min_hard_occupancy=int(counts.min()),empty_hard_components=int((counts==0).sum()))
        write_npz(path/'model.npz',**fitted.state_arrays())
        write_npz(path/'assignments.npz',posterior=labels,nearest=np.asarray(near,dtype='int32'),
            posterior_probability=prob.astype('float32'))
    else:record.update(icl=None,bic=None,entropy=None,log_likelihood=None)
    write_json(path/'fit.json',record)
    files={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.name!='complete.json'}
    write_json(path/'complete.json',dict(files=files))
    return record


def write_tables(root,model,records,cfg):
    out=root/model['key'];flat=[{k:v for k,v in r.items() if k not in ['restarts','model']} for r in records]
    pd.DataFrame(flat).to_csv(out/'candidate_metrics.csv',index=False)
    selection=[]
    for view,layer in sorted({(r['view'],r['layer']) for r in records}):
        rows=[r for r in records if (r['view'],r['layer'])==(view,layer)]
        if not any(r['converged'] for r in rows):continue
        tested=sorted({r['k'] for r in rows})
        for tolerance in cfg['icl_tolerances']:
            chosen=select(rows,tolerance)
            selection.append(dict(view=view,layer=layer,tolerance=tolerance,k=chosen['k'],icl=chosen['icl'],
                fit_path=chosen['fit_path'],converged=True,k_evaluated=tested,k_max_evaluated=max(tested),
                raw_icl_best_k=select(rows,0)['k'],upper_boundary_warning=select(rows,0)['k']>=max(tested)-cfg['boundary_margin']))
    write_json(out/'selection.json',selection)
    pd.DataFrame(selection).to_csv(out/'selection.csv',index=False)
    return selection


def export(root,cfg,model,view,data,selection,records,tolerance):
    chosen=sorted([r for r in selection if r['view']==view and r['tolerance']==tolerance],key=lambda r:r['layer'])
    assert [r['layer'] for r in chosen]==data.layers()
    out=root/model['key']/'exports'/f'{view}_icl_{tolerance:g}'
    if (out/'_SUCCESS.json').exists():return str(out)
    out.mkdir(parents=True,exist_ok=True);models=[];local=[];scans=[]
    identity=Projection(np.zeros(data.state_dim()),np.ones(data.state_dim()),np.zeros(data.state_dim()),np.empty((0,data.state_dim())),np.empty(0))
    for selected in chosen:
        path=Path(selected['fit_path']);fit=load_candidate(path)
        fitted=GMMModel.from_state(fit['model'],dict(np.load(path/'model.npz')))
        models.append(fitted);local.append(np.load(path/'assignments.npz')['posterior'])
        scan=dict(selected=selected,candidates=[{k:r[k] for k in ['k','icl','converged','fit_path']} for r in records if (r['view'],r['layer'])==(view,selected['layer'])])
        scans.append(dict(layer=selected['layer'],**scan));save_layer(out,selected['layer'],fitted,identity,scan)
    aligned=align_layers([m.centers() for m in models],layers=data.layers(),spec=AlignSpec(similarity='cosine',method='hungarian',threshold=cfg['eta']))
    states=np.column_stack([g[a] for g,a in zip(aligned.local_to_global,local)])
    np.save(out/'states.npy',states);np.save(out/'local_states.npy',np.column_stack(local))
    data.meta.to_parquet(out/'rows.parquet',index=False);write_json(out/'data_snapshot.json',data.info)
    write_json(out/'alignment.json',dict(layers=data.layers(),local_to_global=[g.tolist() for g in aligned.local_to_global]))
    write_json(out/'selection.json',scans);write_json(out/'diagnostics.json',[])
    idx=np.arange(data.n_items());write_npz(out/'split.npz',train=idx,map_fit=idx,validation=np.array([],int),test=np.array([],int))
    config=load_experiment(Path(__file__).parents[1]/'configs/base.toml').to_dict()
    config['name']=f"{model['key']}_mmlu_diag_{view}_icl{tolerance:g}"
    config['data'].update(dataset_id='mmlu',paths=[str(Path(cfg['source_root'])/model['key'])],expected_model=model['identifier'],
        expected_samples=cfg['samples'],final_norm='pre' if view=='pre_final' else 'post',layers=data.layers())
    config['cluster'].update(method='gmm',covariance_type='diag',assignment='posterior',n_init=cfg['n_init'],
        max_iter=cfg['max_iter'],tol=cfg['tol'],k_min=1,k_max=max(r['k'] for r in records),parsimony_tolerance=tolerance,
        backend='gpu',device='cuda:0',require_convergence=True,save_candidate_assignments=True)
    config['evaluation']['mode']='geometry';write_json(out/'config.json',config)
    summary=dict(name=config['name'],trial_id=digest(scans),path=str(out),snapshot=data.info['key'],
        n_samples=data.n_items(),n_rows=data.n_items(),model=data.info['model'],n_global_states=aligned.n_global_states,
        profile=[dict(layer=r['layer'],k=r['k'],criterion=r['icl'],criterion_name='icl') for r in chosen],
        evaluation={'scope':'descriptive full-data geometry; correctness labels are metadata only'},seconds=0)
    write_json(out/'summary.json',summary);seal_result(out);write_json(out/'_SUCCESS.json',summary)
    check=Result(out).validate(full=True)
    if not check['valid']:raise ValueError(check)
    return str(out)


def run_model(root,cfg,model,snapshots):
    import torch
    out=root/model['key'];records=[];start=time.time()
    for view,data in snapshots.items():
        for layer in data.layers():
            info=json.loads((out/'snapshots'/f'{view}.json').read_text())
            assert sha(data.path/f'layer_{layer}.npy')==info['arrays'][str(layer)]
            if torch.cuda.mem_get_info()[0]/2**30<cfg['min_gpu_free_gib']+3:raise MemoryError('GPU reserve')
            engine=ResidentDiagonalEM(data.array(layer),reg_covar=cfg['reg_covar'],chunk_size=cfg['chunk_size'])
            rows=[];tried=set()
            def fit(ks):
                for k in ks:
                    if k in tried:continue
                    r=candidate(engine,root,cfg,model,view,layer,k,data.info['key'])
                    rows.append(r);records.append(r);tried.add(k)
                    write_tables(root,model,records,cfg)
            fit(cfg['initial_k'])
            if needs_extension(rows,cfg):fit(cfg['extension_k'])
            for _ in range(cfg['refine_rounds']):fit(refinement(rows,tried,cfg))
            selection=write_tables(root,model,records,cfg)
            write_json(out/'layers'/f'{view}_{layer}.json',dict(complete=True,tested_k=sorted(tried),
                min_icl_k=select(rows,0)['k'],primary_k=select(rows,cfg['primary_tolerance'])['k'],
                at_upper_boundary=select(rows,0)['k']>=max(tried)-cfg['boundary_margin']))
            del engine;torch.cuda.empty_cache()
    selection=write_tables(root,model,records,cfg);exports={}
    for view,data in snapshots.items():
        for tolerance in cfg['icl_tolerances']:
            exports[f'{view}/{tolerance:g}']=export(root,cfg,model,view,data,selection,records,tolerance)
    write_json(out/'exports.json',exports)
    # A fitting report, without launching prediction/steering or other models.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(selection);fig,ax=plt.subplots(figsize=(11,4))
    for tol,part in table[table['view']=='post'].groupby('tolerance'):
        part=part.sort_values('layer');ax.plot(part.layer,part.k,'o-',label=f'ICL {tol:.0%}')
    ax.set(xlabel='Stored layer index',ylabel='Selected GMM components',title=model['identifier']+' / MMLU raw token means');ax.legend()
    fig.tight_layout();fig.savefig(out/'k_by_layer.png',dpi=160);plt.close(fig)
    write_json(out/'COMPLETE.json',dict(model=model['identifier'],n_samples=cfg['samples'],n_candidates=len(records),
        converged_candidates=sum(r['converged'] for r in records),seconds=time.time()-start,exports=exports))


def run(config,prepare_only=False):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    handle=(root/'study.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert cfg['covariance_type']=='diag' and cfg['normalization'] is False and cfg['representation']=='mean'
    freeze_protocol(root,cfg)
    with threadpool_limits(cfg['threads']):
        snapshots={m['key']:prepare_model(root,cfg,m) for m in cfg['models']}
        if prepare_only:
            status(root,'queue',state='prepared');return
        wait_predecessor(root,cfg)
        # No CUDA model is allocated before the predecessor exits.
        gpu_lock=Path('/tmp/hss-gpu-cuda_0.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(cfg['threads'])
        for model in cfg['models']:
            if not (root/model['key']/'COMPLETE.json').exists():run_model(root,cfg,model,snapshots[model['key']])
        write_json(root/'COMPLETE.json',dict(models=[m['key'] for m in cfg['models']],completed_utc=datetime.now(timezone.utc).isoformat()))
        status(root,'queue',state='complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    try:run(a.config,a.prepare_only)
    except BaseException:
        c=json.loads(Path(a.config).read_text());write_json(Path(c['output'])/'failure.json',dict(traceback=traceback.format_exc(),at=time.time()))
        raise

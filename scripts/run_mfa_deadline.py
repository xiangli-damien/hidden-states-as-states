"""Bounded, restartable raw Qwen-MATH MFA study with reusable HSS exports.

Screening is explicitly provisional. Only converged strict-tolerance candidates
with three completed initialization attempts may enter the main HSS export.
No correctness labels are used for fitting or selection. Imported fits retain
their immutable paths and original optimizer provenance.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import datetime as dt
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hss.align import align_layers
from hss.cluster.mfa import MFAModel, fit_mfa
from hss.cluster.mfa_fast import BatchedMFAEM
from hss.data import CachedStates
from hss.experiments.artifacts import digest, file_digest, lock, save_json, save_npz
from hss.experiments.config import Experiment
from hss.experiments.evaluate import characterize
from hss.experiments.fitting import load_fitted
from hss.results import Result
from hss.results.models import save_layer
from hss.results.store import seal_result
from hss.transform.projection import Projection
from hss.types import AlignSpec


def initialize(path, k, rank, seed, output):
    """CPU initialization runs in two spawned workers while the GPU fits."""
    started=time.time();output=Path(output)
    if output.exists():return str(output)
    with threadpool_limits(limits=2):
        X=np.load(path,mmap_mode='r')
        model=fit_mfa(X,k,seed=seed,rank=rank,n_init=1,max_iter=1,tol=0,init_method='svd')
        save_npz(output,config_json=np.array(json.dumps(model.config())),**model.state_arrays())
        save_json(output.with_suffix('.json'),dict(seconds=time.time()-started,seed=seed,k=k,rank=rank))
    return str(output)


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as z:
        return MFAModel.from_state(json.loads(str(z['config_json'])),{k:z[k] for k in z.files if k!='config_json'})


def task_id(task):
    return digest({k:task[k] for k in ['view','layer','k','rank','seed','n_init','tol','max_steps','optimizer']})


def admissible(row):
    return (row.get('status')=='complete' and row.get('converged') is True
            and row.get('tol',1e-5)<=1e-5 and row.get('initializations_completed',0)>=3)


def select(rows, *, strict=True, tolerance=0., criterion='icl'):
    rows=[r for r in rows if (admissible(r) if strict else r.get('status')=='complete')
          and np.isfinite(r.get(criterion,np.nan))]
    if not rows:return None
    best=min(r[criterion] for r in rows)
    return min((r for r in rows if r[criterion]<=best+tolerance*max(abs(best),1.)),
               key=lambda r:(r['k'],r['rank'],r[criterion]))


class Study:
    def __init__(self,args):
        self.args=args;self.root=Path(args.directory);self.source=Path(args.source)
        self.root.mkdir(parents=True,exist_ok=True)
        self.deadline=dt.datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp()
        self.source_protocol=json.loads((self.source/'protocol.json').read_text())
        self.snapshots={};self.units=[];self.records={};self.failures=[];self.active=None
        for view in ['post','pre_final']:
            info=json.loads((self.source/'snapshots'/f'{view}.json').read_text())
            data=CachedStates(Path('/home/ubuntu/hss-cache/data')/info['key'])
            assert data.info['key']==info['key'] and data.n_items()==5000 and data.state_dim()==3584
            self.snapshots[view]=data;self.units.extend((view,l) for l in data.layers())
            save_json(self.root/'snapshots'/f'{view}.json',info)
            data.meta.to_parquet(self.root/'snapshots'/f'{view}_rows.parquet',index=False)
        selections=list(csv.DictReader((self.source/'selection_ablation.csv').open()))
        self.reference={(r['view'],int(r['layer'])):int(r['k']) for r in selections
                        if r['method']=='gmm' and r['criterion']=='icl' and float(r['tolerance'])==.02}
        self.protocol=dict(schema_version=1,source=str(self.source),deadline_utc=args.deadline,
            ranks=[4,8,16],screen_ranks_per_layer=2,raw=True,normalization=False,pca=False,
            dimension=3584,samples=5000,screen=dict(n_init=1,max_steps=400,tol=1e-4),
            final=dict(n_init=3,max_steps=2000,tol=1e-5),optimizer=args.optimizer,
            final_optimizer_control='Last initialization uses plain batched EM; others use the requested accelerator',
            execution=dict(gpu_workers=1,cpu_initialization_workers=2,cpu_threads_each=2,
                           float_precision='float64',gpu_input_resident=True,component_batch=8),
            phases=dict(coverage_cutoff_hours_before_deadline=18,screen_cutoff=7,refine_cutoff=5,
                        final_cutoff=2,stability_cutoff=.5),
            selection='ICL on converged strict fits; BIC and ICL tolerances also exported',
            limitations=['Adaptive bounded search, not exhaustive or a global optimum guarantee.',
                'All 5000 rows fit descriptive geometry; no held-out correctness claim.',
                'Rank can vary by layer; each layer uses one rank shared in count, not direction, across its components.',
                'Historical CPU/GPU EM fits are reused with provenance; accelerated fits may reach different local optima.'],
            driver_sha256=file_digest(Path(__file__)),
            optimizer_sha256=file_digest(Path(__file__).resolve().parents[1]/'src/hss/cluster/mfa_fast.py'))
        previous=self.root/'protocol.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            if old!=self.protocol:raise ValueError('Protocol changed: use a new study directory')
        save_json(previous,self.protocol)
        self.import_previous()
        for p in (self.root/'candidates').glob('*.json'):
            item=json.loads(p.read_text());self.records[item['key']]=item
        self.phase='prepared';self.started=time.time()

    def import_previous(self):
        for view,layer in self.units:
            for rank in [0,1,2,4,8,16,32]:
                task=dict(view=view,layer=layer,method='mfa',rank=rank,k=self.reference[view,layer],seed=42)
                p=self.source/'candidates'/f'{digest(task)}.json'
                if not p.exists():continue
                old=json.loads(p.read_text())
                if old['status']!='complete':continue
                fit=Path(old['record']['fit_path']);info=json.loads((fit/'fit.json').read_text())
                restarts=json.loads((fit/'restarts.json').read_text())
                row=dict(task,**old['record'],key='import_'+digest(str(p)),status='complete',
                         tol=1e-5,n_init=3,initializations_completed=len(restarts),
                         origin='imported',source_candidate=str(p),optimizer=info['model'].get('backend'),
                         purpose='historical',max_steps=2000)
                save_json(self.root/'candidates'/f"{row['key']}.json",row)

    def rows(self,unit=None,seed=42):
        return [r for r in self.records.values() if r.get('seed')==seed and
                (unit is None or (r['view'],r['layer'])==unit)]

    def status(self):
        import psutil
        strict=[r for r in self.records.values() if admissible(r)]
        selected={f'{v}/{l}':select([r for r in self.rows((v,l)) if r['rank'] in [4,8,16]]) for v,l in self.units}
        status=dict(at=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),status='running',
            phase=self.phase,active=self.active,deadline_utc=self.args.deadline,
            remaining_hours=max(0,(self.deadline-time.time())/3600),completed=len(self.records),
            strict_candidates=len(strict),covered_units=sum(x is not None for x in selected.values()),
            total_units=len(self.units),new_fits=sum(r.get('origin')!='imported' for r in self.records.values()),
            failed=len(self.failures),ram_available_gib=psutil.virtual_memory().available/1024**3)
        if self.phase=='finished':
            remaining=sum(len(json.loads(p.read_text()).get('remaining',[])) for p in (self.root/'phases').glob('*.json'))
            status.update(unrun_planned_tasks=remaining,deadline_met=time.time()<=self.deadline,
                status=('complete_with_budget_limits' if remaining or self.failures else 'complete')
                if status['covered_units']==len(self.units) else 'finished_with_missing_layers')
        save_json(self.root/'study.json',status)
        return status

    def report(self,export=False):
        status=self.status();frame=pd.DataFrame(self.records.values());frame.to_csv(self.root/'candidate_metrics.csv',index=False)
        chosen=[];policies=[]
        for unit in self.units:
            options=[r for r in self.rows(unit) if r['rank'] in [4,8,16]]
            row=select(options)
            if row:chosen.append(row)
            for criterion in ['icl','bic']:
                for tol in [0.,.005,.01,.02]:
                    value=select(options,criterion=criterion,tolerance=tol)
                    if value:policies.append(dict(view=unit[0],layer=unit[1],criterion=criterion,tolerance=tol,
                                                  k=value['k'],rank=value['rank'],score=value[criterion],fit_path=value['fit_path']))
        pd.DataFrame(chosen).to_csv(self.root/'selected_layers.csv',index=False)
        pd.DataFrame(policies).to_csv(self.root/'selection_ablation.csv',index=False)
        title='Qwen MATH MFA — 24-hour bounded study'
        body=f'<h1>{title}</h1><p>Raw 5000 × 3584 response means. No extra normalization or PCA. Descriptive fit, not held-out prediction.</p><pre>{html.escape(json.dumps(status,indent=2))}</pre>'
        body+='<p><a href="candidate_metrics.csv">All fits</a> · <a href="selected_layers.csv">Selected layers</a> · <a href="selection_ablation.csv">ICL/BIC sensitivity</a> · <a href="protocol.json">Protocol</a> · <a href="stability.csv">Seed stability</a> · <a href="figures.png">Figures</a></p>'
        if chosen:body+=pd.DataFrame(chosen)[['view','layer','rank','k','icl','converged','origin']].to_html(index=False)
        body+='<h2>Interpretation</h2><p>Only converged fits with all three initialization attempts enter this table. Screening fits are provisional. Historical candidates and unselected fits remain available. Per-layer rank selection does not imply that all clusters have identical effective dimension.</p>'
        if (self.root/'latest_exports.json').exists():body+='<pre>'+html.escape((self.root/'latest_exports.json').read_text())+'</pre>'
        (self.root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;margin:3em;max-width:1300px}td,th{padding:.4em}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'+body)
        if export:
            paths={}
            for view in self.snapshots:
                subset=[r for r in chosen if r['view']==view]
                if len(subset)==len(self.snapshots[view].layers()):paths[view]=self.export(view,subset)
            save_json(self.root/'latest_exports.json',paths)
            self.figures(frame,chosen)
            with (self.root/'index.html').open('a') as f:
                f.write('<h2>Reusable HSS results</h2><ul>'+''.join(
                    f'<li><a href="{Path(path).relative_to(self.root)}/">{view} HSS artifacts</a></li>'
                    for view,path in paths.items())+'</ul>')

    def figures(self,frame,chosen):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        if chosen:
            post=pd.DataFrame([r for r in chosen if r['view']=='post']).sort_values('layer')
            ax[0,0].plot(post.layer,post.k,'o-');ax[0,0].set(title='Selected K',xlabel='Layer')
            ax[0,1].plot(post.layer,post['rank'],'o-');ax[0,1].set(title='Selected factor rank',xlabel='Layer')
        final=frame[(frame.view=='post')&(frame.layer==28)&(frame.status=='complete')]
        for rank,part in final.groupby('rank'):
            part=part.sort_values('k');ax[1,0].scatter(part.k,part.icl/1e6,label=f'rank {rank}')
        ax[1,0].set(title='Final layer candidates (screening included)',xlabel='K',ylabel='ICL / 1e6');ax[1,0].legend(fontsize=8)
        stability=self.root/'stability.csv'
        if stability.exists():
            data=pd.read_csv(stability)
            for seed,p in data[data.view=='post'].groupby('seed'):ax[1,1].plot(p.layer,p.ari,'o-',label=str(seed))
            ax[1,1].legend();ax[1,1].set(title='Initialization sensitivity',xlabel='Layer',ylabel='ARI')
        else:ax[1,1].text(.1,.5,'Seed checks pending',transform=ax[1,1].transAxes)
        fig.savefig(self.root/'figures.png',dpi=150);fig.savefig(self.root/'figures.pdf');plt.close(fig)

    def export(self,view,chosen):
        data=self.snapshots[view];chosen=sorted(chosen,key=lambda r:r['layer'])
        key=digest({'fits':[r['key'] for r in chosen],'eta':.6,'assignment':'posterior'})
        root=self.root/'trials'/f'{view}_{key}'
        if (root/'_SUCCESS.json').exists():return str(root)
        root.mkdir(parents=True,exist_ok=True);models=[];local=[];scans=[]
        identity=Projection(np.zeros(data.state_dim()),np.ones(data.state_dim()),np.zeros(data.state_dim()),np.empty((0,data.state_dim())),np.empty(0))
        for row in chosen:
            model,_=load_fitted(row['fit_path']);models.append(model)
            with np.load(Path(row['fit_path'])/'assignments.npz') as z:local.append(z['posterior'])
            scan=dict(selected=row,candidates=[r for r in self.rows((view,row['layer'])) if r['rank'] in [4,8,16]])
            scans.append(dict(layer=row['layer'],**scan));save_layer(root,row['layer'],model,identity,scan)
        alignment=align_layers([m.centers() for m in models],layers=data.layers(),spec=AlignSpec(similarity='cosine',method='hungarian',threshold=.6,allow_negative=True))
        states=np.column_stack([v[y] for v,y in zip(alignment.local_to_global,local)])
        np.save(root/'states.npy',states,allow_pickle=False);np.save(root/'local_states.npy',np.column_stack(local),allow_pickle=False)
        data.meta.to_parquet(root/'rows.parquet',index=False);save_json(root/'data_snapshot.json',data.info)
        save_json(root/'alignment.json',dict(layers=data.layers(),local_to_global=[v.tolist() for v in alignment.local_to_global]))
        save_json(root/'selection.json',scans);save_json(root/'diagnostics.json',[])
        idx=np.arange(data.n_items());save_npz(root/'split.npz',train=idx,map_fit=idx,validation=np.array([],dtype=int),test=np.array([],dtype=int))
        config=dict(self.source_protocol['identity']['config']);config=json.loads(json.dumps(config))
        config['name']='qwen_math_mfa_24h_'+view;config['data']['final_norm']='pre' if view=='pre_final' else 'post'
        config['cluster'].update(method='mfa',assignment='posterior',rank=8,
            rank_by_layer={str(r['layer']):r['rank'] for r in chosen})
        config['evaluation']['mode']='geometry'
        Experiment.from_dict(config)
        save_json(root/'config.json',config)
        assoc,tags,trans=characterize(states,data.meta,data.layers())
        assoc.to_csv(root/'associations.csv',index=False);tags.to_csv(root/'state_tags.csv',index=False);trans.to_csv(root/'transitions.csv',index=False)
        summary=dict(name=config['name'],trial_id=key,path=str(root),snapshot=data.info['key'],n_samples=5000,n_rows=5000,
                     model=data.info['model'],n_global_states=int(states.max())+1,seconds=0.,
                     profile=[dict(layer=r['layer'],k=r['k'],rank=r['rank'],criterion=r['icl'],criterion_name='icl') for r in chosen],
                     evaluation={'scope':'descriptive full-data geometry'},protocol=str(self.root/'protocol.json'))
        save_json(root/'summary.json',summary);seal_result(root);save_json(root/'_SUCCESS.json',summary)
        audit=Result(root).validate(full=True)
        if not audit['valid']:raise ValueError(audit)
        save_json(root/'validation.json',audit)
        return str(root)

    def task(self,unit,rank,k,*,final=False,seed=42,purpose='screen'):
        return dict(view=unit[0],layer=unit[1],rank=rank,k=k,seed=seed,n_init=3 if final else 1,
                    tol=1e-5 if final or purpose=='stability' else 1e-4,
                    max_steps=2000 if final or purpose=='stability' else 400,
                    optimizer=self.args.optimizer,purpose=purpose)

    def initial_path(self,task,restart):
        seed=task['seed']+1009*task['layer']+restart
        key=digest(dict(snapshot=self.snapshots[task['view']].info['key'],layer=task['layer'],rank=task['rank'],k=task['k'],seed=seed))
        return self.root/'initializations'/f'{key}.npz'

    def fit(self,task,cutoff,initials):
        import torch
        key=task_id(task);path=self.root/'cache/fits'/key;path.mkdir(parents=True,exist_ok=True)
        X=self.snapshots[task['view']].array(task['layer']);engine=None;start=time.time();best=None;audits=[]
        self.active=dict(task,key=key,started_at=start);self.status()
        try:
            free,_=torch.cuda.mem_get_info()
            if free < 10*1024**3:raise MemoryError('Insufficient GPU reserve')
            engine=BatchedMFAEM(X,component_batch=8)
            for restart in range(task['n_init']):
                if time.time()>=cutoff:break
                checkpoint=path/f'restart_{restart:02d}.npz'
                if checkpoint.exists():initial=load_checkpoint(checkpoint)
                else:
                    warm=[r for r in self.rows((task['view'],task['layer']),seed=task['seed'])
                          if r['rank']==task['rank'] and r['k']==task['k'] and r.get('origin')=='batched_24h'
                          and r.get('purpose')=='screen' and restart==0]
                    if warm:
                        initial,_=load_fitted(max(warm,key=lambda r:r['log_likelihood'])['fit_path'])
                        initial=replace(initial,converged_=False)
                    else:initial=load_checkpoint(initials[restart].result())
                def save_checkpoint(model,audit):
                    save_npz(checkpoint,config_json=np.array(json.dumps(model.config())),**model.state_arrays())
                    save_json(path/'progress.json',dict(task,restart=restart,updated_at=time.time(),converged=model.converged_,**audit))
                    self.active.update(restart=restart,steps=audit.get('steps'));self.status()
                if initial.converged_:
                    model,audit=initial,dict(resumed_completed_restart=True)
                else:
                    accelerator='none' if task['n_init']==3 and restart==2 else task['optimizer']
                    model,audit=engine.fit(initial,max_steps=task['max_steps'],tol=task['tol'],accelerator=accelerator,
                        deadline=min(cutoff,start+(900 if task['n_init']==3 else 300)),checkpoint=save_checkpoint)
                audits.append(dict(restart=restart,converged=model.converged_,history_length=len(model.history_),**audit))
                if best is None or (model.converged_,model.history_[-1])>(best.converged_,best.history_[-1]):best=model
            if best is None:raise TimeoutError('No initialization completed before deadline')
            ll,_,probs=engine.evaluate(engine.parameters(best),update=False,probabilities=True)
            entropy=float(-(probs*np.log(np.maximum(probs,1e-300))).sum());bic=-2*len(X)*ll+best._n_parameters()*np.log(len(X))
            # Full score and assignment use the saved model; nearest is a separate control.
            mu=torch.as_tensor(best.means_,dtype=torch.float64,device='cuda:0')
            nearest=(engine.X2.sum(1)[:,None]+mu.square().sum(1)[None,:]-2*engine.X@mu.T).argmin(1).cpu().numpy().astype(np.int32)
            save_npz(path/'model.npz',**best.state_arrays())
            save_npz(path/'assignments.npz',posterior=probs.argmax(1).astype(np.int32),nearest=nearest)
            scores=dict(icl=float(bic+2*entropy),bic=float(bic),entropy=entropy,log_likelihood=ll*len(X),n_parameters=best._n_parameters())
            save_json(path/'fit.json',dict(key=key,k=task['k'],model=best.config(),scores=scores,seconds=time.time()-start,task=task))
            save_json(path/'restarts.json',audits)
            row=dict(task,key=key,origin='batched_24h',status='complete',fit_path=str(path),converged=best.converged_,
                     initializations_completed=len(audits),seconds=time.time()-start,**scores)
            self.records[key]=row;save_json(self.root/'candidates'/f'{key}.json',row)
            print(json.dumps({k:row[k] for k in ['view','layer','k','rank','purpose','converged','seconds','icl']}),flush=True)
        except Exception as exc:
            import traceback
            error=dict(task,key=key,error=repr(exc),traceback=traceback.format_exc(),at=time.time())
            self.failures.append(error);save_json(self.root/'failures'/f'{key}.json',error);print(json.dumps(error),flush=True)
        finally:
            del engine;torch.cuda.empty_cache();self.active=None;self.status()

    def run_queue(self,phase,tasks,cutoff):
        self.phase=phase;unique={task_id(t):t for t in tasks}
        pending=[]
        for key,t in unique.items():
            if key in self.records:continue
            if t['n_init']==3 and any(admissible(r) and r['rank']==t['rank'] and r['k']==t['k'] for r in self.rows((t['view'],t['layer']),seed=t['seed'])):continue
            pending.append(t)
        self.status();print(json.dumps(dict(phase=phase,pending=len(pending),cutoff=cutoff)),flush=True)
        manifest=dict(phase=phase,planned=len(unique),cached=len(unique)-len(pending),
                      pending_keys=[task_id(t) for t in pending],cutoff=cutoff)
        save_json(self.root/'phases'/f'{phase}.json',manifest)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            ready={}
            def submit(i):
                if i>=len(pending) or i in ready or time.time()>=cutoff:return
                t=pending[i];data=self.snapshots[t['view']];path=data.path/f"layer_{t['layer']}.npy"
                ready[i]=[pool.submit(initialize,str(path),t['k'],t['rank'],t['seed']+1009*t['layer']+j,str(self.initial_path(t,j))) for j in range(t['n_init'])]
            for i,t in enumerate(pending):
                if time.time()>=cutoff:break
                submit(i);submit(i+1)
                self.fit(t,cutoff,ready.pop(i))
                if (i+1)%5==0:self.report()
        manifest.update(completed_new=sum(task_id(t) in self.records for t in pending),
                        remaining=[t for t in pending if task_id(t) not in self.records],finished_at=time.time())
        save_json(self.root/'phases'/f'{phase}.json',manifest)
        self.report(export=True)

    def run(self):
        # First produce a complete, strict rank-8 map. Extra ranks follow.
        coverage=[self.task(u,r,self.reference[u],final=True,purpose='coverage') for r in [8,4,16] for u in self.units]
        self.run_queue('coverage_and_rank',coverage,self.deadline-18*3600)
        shortlist={}
        for u in self.units:
            ranked=[]
            for rank in [4,8,16]:
                row=select([r for r in self.rows(u) if r['rank']==rank])
                if row:ranked.append(row)
            shortlist[u]=[r['rank'] for r in sorted(ranked,key=lambda r:r['icl'])[:2]] or [8]
        tasks=[]
        for index in range(5):
            for u in self.units:
                k0=self.reference[u];grid=sorted(set([2,max(2,k0//2),k0,min(80,round(1.5*k0)),80]))
                if index>=len(grid):continue
                for rank in shortlist[u]:
                    k=grid[index]
                    if not any(r['rank']==rank and r['k']==k and r.get('status')=='complete' for r in self.rows(u)):
                        tasks.append(self.task(u,rank,k))
        self.run_queue('bounded_k_screen',tasks,self.deadline-7*3600)
        tasks=[]
        for u in self.units:
            for rank in shortlist[u]:
                row=select([r for r in self.rows(u) if r['rank']==rank],strict=False)
                if row:
                    step=max(1,row['k']//5)
                    for k in sorted(set([max(2,row['k']-step),min(80,row['k']+step)])):
                        if not any(r['rank']==rank and r['k']==k for r in self.rows(u)):tasks.append(self.task(u,rank,k))
        self.run_queue('local_k_refinement',tasks,self.deadline-5*3600)
        tasks=[]
        for u in self.units:
            options=sorted([r for r in self.rows(u) if r['rank'] in shortlist[u]],key=lambda r:r['icl'])
            seen=set()
            for row in options:
                pair=(row['rank'],row['k'])
                if pair in seen:continue
                seen.add(pair);tasks.append(self.task(u,*pair,final=True,purpose='final'))
                if len(seen)>=2:break
        self.run_queue('strict_finalists',tasks,self.deadline-2*3600)
        chosen={u:select([r for r in self.rows(u) if r['rank'] in [4,8,16]]) for u in self.units}
        tasks=[self.task(u,r['rank'],r['k'],seed=seed,purpose='stability') for seed in [1042,2042] for u,r in chosen.items() if r]
        self.run_queue('independent_seed_checks',tasks,self.deadline-.5*3600)
        from sklearn.metrics import adjusted_rand_score
        stability=[]
        for u,reference in chosen.items():
            if not reference:continue
            with np.load(Path(reference['fit_path'])/'assignments.npz') as z:a=z['posterior']
            for seed in [1042,2042]:
                for row in self.rows(u,seed=seed):
                    if (row['rank'],row['k'])!=(reference['rank'],reference['k']):continue
                    with np.load(Path(row['fit_path'])/'assignments.npz') as z:b=z['posterior']
                    stability.append(dict(view=u[0],layer=u[1],seed=seed,k=row['k'],rank=row['rank'],ari=adjusted_rand_score(a,b),converged=row['converged'],reference_fit=reference['fit_path'],fit_path=row['fit_path']))
        pd.DataFrame(stability,columns=['view','layer','seed','k','rank','ari','converged','reference_fit','fit_path']).to_csv(self.root/'stability.csv',index=False)
        self.phase='finished';self.report(export=True)
        s=self.status();remaining=sum(len(json.loads(p.read_text()).get('remaining',[])) for p in (self.root/'phases').glob('*.json'))
        s['unrun_planned_tasks']=remaining
        s['status']=('complete_with_budget_limits' if remaining or self.failures else 'complete') if s['covered_units']==len(self.units) else 'finished_with_missing_layers'
        s['stability_checks']=len(stability);s['deadline_met']=time.time()<=self.deadline
        save_json(self.root/'study.json',s);save_json(self.root/'delivery.json',dict(s,exports=json.loads((self.root/'latest_exports.json').read_text())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--deadline',required=True);p.add_argument('--optimizer',choices=['none','squarem'],default='none');p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    with lock(Path(args.directory)/'study.lock'),threadpool_limits(limits=2):
        study=Study(args)
        if args.report_only:study.report(export=True)
        else:study.run()


if __name__=='__main__':main()

"""Frozen Qwen MATH reliability replication; CPU preview then exclusive GPU fits.

No generation, labels, normalization, MFA, or new dataset collection. Historical
geometry is explicitly a preview; primary findings use newly fitted, converged
float64 diagonal mixtures with one protocol across all procedural perturbations.
"""
import argparse
from datetime import datetime, timezone
import errno
import fcntl
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from hss.cluster.gmm import GMMModel
from hss.cluster.gmm_resident import ResidentDiagonalEM
from hss.data import CachedStates

try:
    from scripts.revision_common import write_json, write_npz, sha, freeze, digest, nearest
except ModuleNotFoundError:
    from revision_common import write_json, write_npz, sha, freeze, digest, nearest

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    for attempt in range(10):
        try:
            return json.loads(Path(path).read_text())
        except OSError as exc:
            if exc.errno not in (errno.ESTALE, errno.ENOENT) or attempt == 9:
                raise
            time.sleep(.2)


def variants(cfg):
    result = [dict(name=f'seed_{s}', seed=s, fraction=1., group='seed') for s in cfg['seeds']]
    for f in sorted(set(cfg['primary_fractions'] + cfg['figure_fractions'])):
        result.append(dict(name=f'fraction_{int(round(100*f)):02d}', seed=cfg['seeds'][0],
            fraction=f, group='subsample', primary=f in cfg['primary_fractions']))
    return result


def sample_indices(n, fraction, seed):
    if fraction == 1:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[:int(round(n*fraction))])


def select(rows, tolerance, cap):
    valid = [r for r in rows if r['k'] <= cap and r['converged'] and np.isfinite(r['icl'])]
    if not valid:
        raise ValueError('No converged candidate inside the requested K range')
    best = min(r['icl'] for r in valid)
    return min((r for r in valid if r['icl'] <= best+tolerance*max(abs(best),1.)), key=lambda r:r['k'])


def match_centers(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a,axis=1), np.linalg.norm(b,axis=1)
    if np.any(an == 0) or np.any(bn == 0):
        raise ValueError('Zero center has undefined cosine direction')
    distance = 1-np.clip((a/an[:,None]) @ (b/bn[:,None]).T,-1,1)
    i,j = linear_sum_assignment(distance)
    return dict(distance=float(distance[i,j].mean()), unmatched_a=len(a)-len(i), unmatched_b=len(b)-len(j))


def uniform_baseline(X, centers, repeats, seed):
    rng=np.random.default_rng(seed);out=[];k=len(centers)
    for repeat in range(repeats):
        labels=rng.integers(k,size=len(X))
        counts=np.bincount(labels,minlength=k)
        # Randomly assign the REAL samples, not synthetic Gaussian centers.
        random_centers=np.stack([np.asarray(X[labels==j],dtype=np.float64).mean(0)
                                 for j in range(k) if counts[j]])
        out.append(dict(repeat=repeat, empty_random_components=int((counts==0).sum()),
                        **match_centers(centers,random_centers)))
    return out


def symmetric_kl(mu_a, va, mu_b, vb):
    va=np.maximum(va,1e-12);vb=np.maximum(vb,1e-12)
    return .25*float(np.sum(va/vb+vb/va+(mu_a-mu_b)**2*(1/va+1/vb)-2))


def geometry(X, model, cfg, layer):
    labels=nearest(X,model.means_,cfg['chunk_size'])
    within=[];between=[];counts=np.bincount(labels,minlength=model.n_clusters())
    for j in range(model.n_clusters()):
        if counts[j]>1:
            part=np.asarray(X[labels==j],dtype=np.float64);mu=part.mean(0);var=part.var(0)
            within.append(dict(component=j,n=int(counts[j]),
                kl=symmetric_kl(model.means_[j],model.covariances_[j],mu,var+cfg['reg_covar']),
                floor_only_kl=symmetric_kl(model.means_[j],model.covariances_[j],mu,var)))
        for i in range(j):
            between.append(dict(a=i,b=j,kl=symmetric_kl(model.means_[i],model.covariances_[i],
                                                       model.means_[j],model.covariances_[j])))
    random=uniform_baseline(X,model.means_,cfg['random_repeats'],cfg['subset_seed']+layer)
    return dict(layer=layer,k=model.n_clusters(),assignment='nearest',n=len(X),
        counts=counts.tolist(),empty_components=int((counts==0).sum()),singletons=int((counts==1).sum()),
        within_mean=float(np.mean([x['kl'] for x in within])) if within else None,
        between_mean=float(np.mean([x['kl'] for x in between])) if between else None,
        random_mean=float(np.mean([x['distance'] for x in random])),
        within=within,between=between,random=random)


def load_model(path, metadata=None):
    p=Path(path);metadata=metadata or read(p/'fit.json')
    with np.load(p/'model.npz',allow_pickle=False) as arrays:
        return GMMModel.from_state(metadata['model'],dict(arrays))


def verified_fit(path):
    r=read(path/'fit.json')
    marker=read(path/'complete.json')
    assert marker['fit_sha256']==sha(path/'fit.json')
    if r['converged']:
        assert marker['model_sha256']==sha(path/'model.npz')
    return r


def fit_candidate(engine, path, cfg, seed, layer, k, phase, progress):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():
        return verified_fit(path)
    best=None;restarts=[];started=time.monotonic()
    for restart in range(cfg['n_init']):
        init_seed=1000003*seed+1009*layer+10007*restart+(2000000000 if phase=='fixed' else 0)
        progress(phase=phase,layer=layer,k=k,restart=restart)
        try:
            fitted,metrics,_=engine.fit(k,init_seed,max_iter=cfg['max_iter'],tol=cfg['tol'])
            metrics.update(seed=init_seed)
            restarts.append(metrics)
            if fitted.converged_ and (best is None or metrics['log_likelihood']>best[1]['log_likelihood']):
                best=(fitted,metrics)
        except FloatingPointError as exc:
            restarts.append(dict(seed=init_seed,converged=False,error=str(exc)))
    row=dict(k=k,layer=layer,phase=phase,seed=seed,n=engine.n,converged=best is not None,
             restarts=restarts,path=str(path),seconds=time.monotonic()-started,icl=None)
    if best is not None:
        model,metrics=best
        idx=np.linspace(0,engine.n-1,min(64,engine.n),dtype=int)
        p=engine.parameters(model)
        _,_,_,prob=engine.evaluate(p,update=False,probabilities=True)
        X=(engine.X[idx]+engine.offset).cpu().numpy()
        np.testing.assert_allclose(prob[idx],model.predict_proba(X),atol=2e-6,rtol=2e-5)
        row.update({k:metrics[k] for k in ['icl','bic','entropy','log_likelihood','n_parameters','n_iter']})
        row['model']=model.config();write_npz(path/'model.npz',**model.state_arrays())
    write_json(path/'fit.json',row)
    write_json(path/'complete.json',dict(fit_sha256=sha(path/'fit.json'),
        model_sha256=sha(path/'model.npz') if best is not None else None))
    return row


def prepare(root,cfg):
    data=CachedStates(cfg['cache'])
    assert data.n_items()==cfg['samples'] and data.state_dim()==cfg['dimension'] and data.layers()==cfg['layers']
    assert data.info['model'][:2]==[cfg['model'],cfg['revision']]
    assert data.info['identity']['spec']['representation']=='mean'
    assert data.info['identity']['spec']['final_norm']=='post'
    assert data.meta.sample_id.nunique()==cfg['samples']
    for layer in data.layers():
        assert np.isfinite(data.array(layer)).all()
    files=[Path(__file__),REPO/'src/hss/cluster/gmm_resident.py',REPO/'src/hss/cluster/gmm.py',
           REPO/'scripts/revision_common.py']
    identity=dict(config=cfg,code_sha256={str(p):sha(p) for p in files},
        environment=dict(python=sys.version,packages={name:version(name) for name in
            ['numpy','scipy','scikit-learn','torch','pandas','matplotlib']}),
        data_sha256={str(data.path/f'layer_{L}.npy'):sha(data.path/f'layer_{L}.npy') for L in data.layers()},
        rows_sha256=sha(data.path/'rows.parquet'),cache_manifest_sha256=sha(data.path/'_SUCCESS.json'),
        legacy_selection_sha256=sha(cfg['legacy_selection']),
        scope='Figure4(a-c), Qwen MATH. No labels used. New estimates, not reproduction of exact published values.',
        definitions=dict(icl='BIC+2*posterior entropy; converged fits only',
            covariance='diagonal; empirical within-cluster variance ddof0 + same reg; floor-only sensitivity also saved',
            center_bands='procedural pair variation, not confidence intervals; refit pairs are dependent',
            subset='one nested unlabeled random permutation; primary 30/50/70/90%, figure 20/40/60/80%',
            fixed_k='fresh independent restart seeds at baseline primary K; never used for count trend',
            k_caps='restrict the identical per-K scan, requiring no additional fitting',
            assignment='Euclidean nearest center; primary ARI on same all5000 reference population',
            random='10 uniform sample-label assignments, empirical means, Hungarian cosine'))
    p=root/'protocol.json'
    if p.exists():
        assert read(p)['identity']==identity,'Frozen protocol changed'
    else:
        freeze(p,dict(identity=identity,frozen_at=now(),git=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    for v in variants(cfg):
        idx=sample_indices(data.n_items(),v['fraction'],cfg['subset_seed'])
        p=root/'subsets'/f'{v["name"]}.npz'
        if p.exists():
            with np.load(p) as f:np.testing.assert_array_equal(f['indices'],idx)
        else:
            write_npz(p,indices=idx,sample_ids=data.meta.sample_id.astype(str).to_numpy()[idx].astype(str))
    return data


def preview(root,cfg,data,progress):
    selections=read(cfg['legacy_selection'])
    if isinstance(selections,dict):
        selections=selections.get('layers',selections.get('selection',selections.get('selected',selections)))
    if not isinstance(selections,list):
        raise ValueError('Unexpected historical selection schema')
    rows=[]
    for entry in selections:
        layer=entry['layer'];chosen=entry.get('selected',entry);path=Path(chosen.get('fit_path',chosen.get('path','')))
        if layer not in cfg['layers']:continue
        dest=root/'historical_preview'/f'layer_{layer:02d}.json'
        if dest.exists():r=read(dest)
        else:
            progress(phase='historical_geometry',layer=layer)
            model=load_model(path);assert model.converged_
            r=geometry(data.array(layer),model,cfg,layer)
            r.update(source='historical map; not a new independent replication',fit_path=str(path),
                     fit_sha256=sha(path/'fit.json'),model_sha256=sha(path/'model.npz'))
            write_json(dest,r)
        rows.append(r)
    assert sorted(r['layer'] for r in rows)==cfg['layers']
    write_json(root/'historical_preview'/'summary.json',dict(rows=rows,scope='geometry/random-baseline preview only; no new refits'))
    render(root,cfg)
    return rows


def wait_for_collection(root,cfg,progress):
    prev=Path(cfg['collection_predecessor'])
    while True:
        status=read(prev/'job_status.json')
        if status['status'] in ('failed','running_with_failure'):
            raise RuntimeError('Collection predecessor failed; preserve priority and stop waiting queue')
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        if (prev/'_SUCCESS').exists() and status['status']=='complete' and status['completed_samples']==5400 and not pids:
            write_json(root/'predecessor_receipt.json',dict(at=now(),success_sha256=sha(prev/'_SUCCESS'),
                completed_samples=5400,gpu_empty=True))
            return
        progress(phase='waiting_for_collection',predecessor_completed=status['completed_samples'],active_gpu_pids=pids)
        time.sleep(30)


def gpu_check(root):
    X=np.random.default_rng(11).normal(size=(137,9))+10000
    cpu=ResidentDiagonalEM(X,device='cpu',chunk_size=31)
    p,_=cpu.initialize(4,42);initial=cpu.model(p,False,0)
    ll,updated,ent,prob=cpu.evaluate(p,probabilities=True)
    gpu=ResidentDiagonalEM(X,device='cuda:0',chunk_size=31)
    gl,gu,ge,gp=gpu.evaluate(gpu.parameters(initial),probabilities=True)
    np.testing.assert_allclose(gp,prob,atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose([ll,ent],[gl,ge],atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(cpu.model(updated,False,1).covariances_,gpu.model(gu,False,1).covariances_,atol=1e-8)
    write_json(root/'gpu_preflight.json',dict(passed=True,at=now(),posterior_max_error=float(abs(gp-prob).max())))


def run_gpu(root,cfg,data,progress):
    import torch
    gpu_check(root)
    selections=[];refits=[]
    baseline={}
    for v in variants(cfg):
        idx=sample_indices(data.n_items(),v['fraction'],cfg['subset_seed'])
        for layer in cfg['layers']:
            if shutil.disk_usage(root).free<cfg['minimum_disk_free_gib']*1024**3:
                raise OSError('Insufficient artifact disk space')
            def update(**kw):progress(variant=v['name'],**kw)
            engine=ResidentDiagonalEM(data.array(layer)[idx],device='cuda:0',
                chunk_size=cfg['chunk_size'],reg_covar=cfg['reg_covar'])
            rows=[fit_candidate(engine,root/'fits'/v['name']/f'L{layer:02d}'/f'k{k:03d}',
                    cfg,v['seed'],layer,k,'scan',update) for k in range(cfg['k_min'],cfg['k_max']+1)]
            for cap in cfg['k_caps']:
                for tol in cfg['icl_tolerances']:
                    best=select(rows,tol,cap);raw=select(rows,0,cap)
                    selections.append(dict(**v,layer=layer,cap=cap,tolerance=tol,k=best['k'],
                        icl=best['icl'],raw_best_k=raw['k'],best_on_upper_boundary=raw['k']==cap,
                        converged_candidates=sum(r['converged'] for r in rows if r['k']<=cap),
                        expected_candidates=cap-cfg['k_min']+1,fit_path=best['path']))
            chosen=select(rows,cfg['primary_tolerance'],cfg['k_max'])
            if v['name']==f'seed_{cfg["seeds"][0]}':
                baseline[layer]=chosen
                fixed=chosen
                model=load_model(chosen['path']);r=geometry(data.array(layer),model,cfg,layer)
                write_json(root/'geometry'/f'layer_{layer:02d}.json',r)
            else:
                k=baseline[layer]['k']
                fixed=fit_candidate(engine,root/'fixed'/v['name']/f'L{layer:02d}',
                    cfg,v['seed'],layer,k,'fixed',update)
                if not fixed['converged']:
                    raise RuntimeError(f'No converged fixed-K refit: {v["name"]} layer{layer}')
            model=load_model(fixed['path']);labels=nearest(data.array(layer),model.means_,cfg['chunk_size'])
            label_file=root/'fixed_labels'/v['name']/f'L{layer:02d}.npz'
            write_npz(label_file,nearest=labels)
            refits.append(dict(**v,layer=layer,k=fixed['k'],fit_path=fixed['path'],labels_path=str(label_file)))
            write_json(root/'selection.json',selections);write_json(root/'refits.json',refits)
            del engine;torch.cuda.empty_cache()
            render(root,cfg)
    comparisons=[]
    for layer in cfg['layers']:
        for a,b in itertools.combinations([r for r in refits if r['layer']==layer],2):
            base=f'seed_{cfg["seeds"][0]}'
            if a['group']==b['group']=='seed':group='seed'
            elif all(x['group']=='subsample' or x['name']==base for x in (a,b)):group='subsample'
            else:continue
            ma,mb=load_model(a['fit_path']),load_model(b['fit_path'])
            assert a['k']==b['k']
            with np.load(a['labels_path']) as f:la=f['nearest']
            with np.load(b['labels_path']) as f:lb=f['nearest']
            comparisons.append(dict(layer=layer,group=group,a=a['name'],b=b['name'],k=a['k'],
                ari=float(adjusted_rand_score(la,lb)),common_reference_rows=len(la),
                **match_centers(ma.means_,mb.means_)))
    write_json(root/'center_comparisons.json',comparisons)
    audit(root,cfg,data,selections,refits,comparisons)
    render(root,cfg)


def audit(root,cfg,data,selections,refits,comparisons):
    assert len(selections)==len(variants(cfg))*len(cfg['layers'])*len(cfg['k_caps'])*len(cfg['icl_tolerances'])
    assert len(refits)==len(variants(cfg))*len(cfg['layers'])
    for v in variants(cfg):
        for layer in cfg['layers']:
            rows=[verified_fit(root/'fits'/v['name']/f'L{layer:02d}'/f'k{k:03d}')
                  for k in range(cfg['k_min'],cfg['k_max']+1)]
            for r in rows:
                for x in r['restarts']:
                    if 'icl' in x:np.testing.assert_allclose(x['icl'],x['bic']+2*x['entropy'],rtol=1e-12)
            chosen=[r for r in selections if r['name']==v['name'] and r['layer']==layer]
            for c in chosen:assert c['k']==select(rows,c['tolerance'],c['cap'])['k']
    assert all(r['unmatched_a']==r['unmatched_b']==0 for r in comparisons)
    summaries=[]
    for tolerance in cfg['icl_tolerances']:
        full=[r for r in selections if r['cap']==cfg['k_max'] and r['tolerance']==tolerance and r['group']=='seed']
        curves=np.array([[next(r['k'] for r in full if r['seed']==s and r['layer']==L)
                          for L in cfg['layers']] for s in cfg['seeds']])
        summaries.append(dict(tolerance=tolerance,
            mean_absolute_deviation_from_seed_mean=float(abs(curves-curves.mean(0)).mean()),
            mean_absolute_deviation_from_seed42=float(abs(curves[1:]-curves[0]).mean()),
            seed_peak_layers={str(s):[cfg['layers'][i] for i in np.flatnonzero(row==row.max())]
                              for s,row in zip(cfg['seeds'],curves)},
            all_seed_peaks_include_paper_layer10=bool(all(row[cfg['layers'].index(10)]==row.max() for row in curves))))
    write_json(root/'trend_summary.json',summaries)
    center_summary={}
    for group in ['seed','subsample']:
        rows=[r for r in comparisons if r['group']==group]
        center_summary[group]={key:float(np.mean([r[key] for r in rows])) for key in ['distance','ari']}
    gs=[read(root/'geometry'/f'layer_{L:02d}.json') for L in cfg['layers']]
    center_summary['uniform_random_distance_mean']=float(np.mean([r['random_mean'] for r in gs]))
    center_summary['inter_exceeds_intra_layers']=[r['layer'] for r in gs
        if r['within_mean'] is not None and r['between_mean'] is not None and r['between_mean']>r['within_mean']]
    write_json(root/'geometry_summary.json',center_summary)
    write_json(root/'audit.json',dict(passed=True,at=now(),selections=len(selections),
        fixed_refits=len(refits),matched_pairs=len(comparisons),data_rows=data.n_items()))


def render(root,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=root/'report';report.mkdir(exist_ok=True)
    panels=[]
    for folder,label in [('historical_preview','Historical map: geometry preview'),('geometry','New baseline: geometry')]:
        files=sorted((root/folder).glob('layer_*.json'))
        if not files:continue
        rows=[read(p) for p in files]
        fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        ax[0].plot([r['layer'] for r in rows],[r['between_mean'] for r in rows],label='Between components')
        ax[0].plot([r['layer'] for r in rows],[r['within_mean'] for r in rows],label='Within-component fit')
        ax[0].set(yscale='log',ylabel='Symmetric KL',xlabel='Stored layer index');ax[0].legend()
        ax[1].plot([r['layer'] for r in rows],[r['random_mean'] for r in rows])
        ax[1].set(xlabel='Stored layer index',ylabel='Matched cosine distance',title='Uniform sample-assignment baseline')
        fig.suptitle(label+' | raw Qwen MATH means')
        fig.savefig(report/f'{folder}.png',dpi=170);fig.savefig(report/f'{folder}.svg');plt.close(fig)
        panels.append(f'<h2>{label}</h2><img src="{folder}.png" width="100%">')
    if (root/'selection.json').exists():
        rows=read(root/'selection.json');pd.DataFrame(rows).to_csv(report/'selection.csv',index=False)
        primary=[r for r in rows if r['cap']==cfg['k_max'] and r['tolerance']==cfg['primary_tolerance']]
        fig,ax=plt.subplots(1,3,figsize=(15,3.6),layout='constrained')
        for v in variants(cfg):
            rr=sorted([r for r in primary if r['name']==v['name']],key=lambda r:r['layer'])
            if rr:ax[int(v['group']=='subsample')].plot([r['layer'] for r in rr],[r['k'] for r in rr],label=v['name'])
        for cap in cfg['k_caps']:
            rr=sorted([r for r in rows if r['name']==f'seed_{cfg["seeds"][0]}' and
                r['cap']==cap and r['tolerance']==cfg['primary_tolerance']],key=lambda r:r['layer'])
            if rr:ax[2].plot([r['layer'] for r in rr],[r['k'] for r in rr],label=f'Kmax={cap}')
        for a,title in zip(ax,['Seed variation: reselect K','Subsamples: reselect K','Search-range sensitivity']):
            a.set(title=title,xlabel='Stored layer index',ylabel='Selected K');a.legend(fontsize=7)
        fig.savefig(report/'trends.png',dpi=170);fig.savefig(report/'trends.svg');plt.close(fig)
        panels.append('<h2>New fits: count trends (partial until COMPLETE)</h2><img src="trends.png" width="100%">')
    if (root/'center_comparisons.json').exists():
        rows=read(root/'center_comparisons.json');frame=pd.DataFrame(rows);frame.to_csv(report/'centers.csv',index=False)
        fig,ax=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
        for group,part in frame.groupby('group'):
            for a,metric in zip(ax,['distance','ari']):
                g=part.groupby('layer')[metric];m=g.mean();sd=g.std().fillna(0)
                a.plot(m.index,m,label=group);a.fill_between(m.index,m-sd,m+sd,alpha=.15)
        geo=[read(p) for p in sorted((root/'geometry').glob('layer_*.json'))]
        ax[0].plot([r['layer'] for r in geo],[r['random_mean'] for r in geo],ls='--',label='Uniform assignment')
        for a,title in zip(ax,['Matched-center cosine distance','Member consistency (ARI)']):
            a.set(title=title,xlabel='Stored layer index');a.legend()
        fig.savefig(report/'centers.png',dpi=170);fig.savefig(report/'centers.svg');plt.close(fig)
        panels.append('<h2>Fixed-K refits: centers and memberships</h2><img src="centers.png" width="100%">')
    (report/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen MATH reliability</title>'
        '<style>body{font:17px system-ui;max-width:1150px;margin:32px auto;padding:20px;line-height:1.6}img{border:1px solid #ddd}</style>'
        '<h1>Qwen2 × MATH: reliability replication</h1><p>All plots use actual results. Historical preview is not a new refit. '
        'Pending runs are not evidence of stability. Raw response means; no correctness labels used. '
        'Layer28 includes final RMSNorm. Complete only when COMPLETE.json and audit.json exist. '
        'Variation across refit pairs is not a confidence interval.</p>'+''.join(panels))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO/'configs/qwen_math_reliability_20260925.json')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();cfg=read(args.config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'job.lock').open('a') as lock, threadpool_limits(cfg['threads']):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='running',pid=os.getpid(),started_at=now())
        def progress(**kw):
            state.update(kw,updated_at=now());write_json(root/'job_status.json',state)
        try:
            progress(phase='preparing');data=prepare(root,cfg)
            preview(root,cfg,data,progress)
            progress(phase='cpu_preview_complete')
            if args.prepare_only:return
            wait_for_collection(root,cfg,progress)
            import torch
            torch.set_num_threads(cfg['threads'])
            run_gpu(root,cfg,data,progress)
            write_json(root/'COMPLETE.json',dict(at=now(),protocol_sha256=sha(root/'protocol.json'),audit_sha256=sha(root/'audit.json')))
            progress(status='complete',phase='complete')
        except BaseException:
            progress(status='failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()

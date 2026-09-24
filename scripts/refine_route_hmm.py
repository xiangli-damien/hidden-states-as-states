"""Continue unconverged HMM fits without labels; preserve initial benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from hss.experiments.artifacts import file_digest,save_json
from hss.route.structure import aggregate_losses


def refine(job):
    root,r,z,tr,va=job;t=time.monotonic();p=Path(root)/r['path']
    with threadpool_limits(2):
        model=joblib.load(p/'model.joblib');old=list(model.trace);model.max_iter=2000
        model.fit(z[tr],warm_start=True);loss=model.losses(z)
        joblib.dump(model,p/'model.joblib');np.savez_compressed(p/'scores.npz',losses=loss,**aggregate_losses(loss))
        save_json(p/'training.json',dict(trace=model.trace,initial_trace=old,converged=model.converged,
                                       max_additional_iterations=2000,stopping='absolute mean log-likelihood change < 1e-5',warm_start=True))
        r.update(validation=float(loss[va].mean()),initial_seconds=r['seconds'],seconds=r['seconds']+time.monotonic()-t,
                 refinement_seconds=time.monotonic()-t,artifacts={f.name:file_digest(f) for f in p.iterdir() if f.is_file() and f.name!='complete.json'})
        save_json(p/'complete.json',r)
    return r,model.converged,len(model.trace)


def run(a):
    start=time.monotonic();source=Path(a.source);root=Path(a.output);data=Path(a.data)
    if root.exists():raise ValueError('Refinement requires a new output directory')
    skip={'evaluation','report','inventory.json','_SUCCESS.json','local_verification.json','process.json','fit.log'}
    shutil.copytree(source,root,ignore=lambda p,n:[f for f in n if f in skip] if Path(p)==source else [])
    for name in ['protocol.json','_SCORES_FROZEN.json','selected.json','unlabeled_scores.parquet']:
        (root/name).rename(root/('initial_'+name.lstrip('_')))
    protocol=json.loads((root/'initial_protocol.json').read_text());records=json.loads((root/'candidates.json').read_text())
    scores=pd.read_parquet(root/'initial_unlabeled_scores.parquet');tr=(scores.split=='train').to_numpy();va=(scores.split=='validation').to_numpy()
    meta=pd.read_parquet(data/'inputs/rows.parquet',columns=['sample_id'])
    assert meta.sample_id.tolist()==scores.sample_id.tolist()
    for f,digest in protocol['inputs'].items():assert file_digest(data/f)==digest
    protocol.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        refinement=dict(reason='All selected HMM seeds reached initial 100-iteration cap; remedy chosen from convergence diagnostics only.',
                        source=str(source),additional_max_iter=2000,workers=4,threads_each=2,
                        selection='Minimum validation NLL among HMM component settings whose three seeds all converge.',
                        source_sha256=file_digest(Path(__file__)),structure_sha256=file_digest(Path('src/hss/route/structure.py'))))
    save_json(root/'protocol.json',protocol)
    raw=np.load(data/'inputs/states.npz');states={k:raw[k] for k in ['gmm','mfa']}
    states['gmm_matched']=np.load(data/'controls/matched_states.npz')['states']
    jobs=[]
    for r in records:
        if r['method']=='hmm' and not json.loads((root/r['path']/'training.json').read_text())['converged']:
            enc=joblib.load(root/'models'/r['map']/'encoder.joblib');z=enc.transform(states[r['map']])
            jobs.append((str(root),r,z,tr,va))
    changes={};status=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(refine,j) for j in jobs]):
            r,conv,iterations=f.result();changes[r['path']]=r
            status.append(dict(path=r['path'],converged=conv,additional_iterations=iterations,seconds=r['refinement_seconds']))
            save_json(root/'refinement.json',status)
            print(json.dumps(status[-1]),flush=True)
    records=[changes.get(r['path'],r) for r in records];save_json(root/'candidates.json',records)
    selected=json.loads((root/'initial_selected.json').read_text())
    for row in selected:
        if row['method']!='hmm':continue
        options=[]
        for paramkey in sorted(set(r['paramkey'] for r in records if r['map']==row['map'] and r['method']=='hmm')):
            group=[r for r in records if r['map']==row['map'] and r['method']=='hmm' and r['paramkey']==paramkey]
            if all(json.loads((root/r['path']/'training.json').read_text())['converged'] for r in group):
                options.append((np.mean([r['validation'] for r in group]),paramkey,group))
        if not options:raise ValueError('No fully converged HMM configuration')
        value,key,group=min(options,key=lambda x:x[0]);row.update(paramkey=key,candidates=[r['path'] for r in group],validation_objective=float(value))
        for agg in ['mean','top4']:scores[row['name']+'__'+agg]=np.mean([np.load(root/r['path']/'scores.npz')[agg] for r in group],axis=0)
    for map_name in protocol['config']['maps']:
        cols=[c for c in scores if c.startswith(map_name+'__') and c.endswith('__mean') and '__node__' not in c and '__ensemble__' not in c]
        scores[map_name+'__ensemble__mean']=np.mean([np.searchsorted(np.sort(scores.loc[va,c]),scores[c],side='right')/(va.sum()+1) for c in cols],axis=0)
    variants=dict(np.load(root/'configuration_scores.npz'))
    for map_name in protocol['config']['maps']:
        for paramkey in ['components2','components4','components8']:
            group=[r for r in records if r['map']==map_name and r['method']=='hmm' and r['paramkey']==paramkey]
            variants[map_name+'__hmm__'+paramkey]=np.mean([np.load(root/r['path']/'scores.npz')['mean'] for r in group],axis=0)
    np.savez_compressed(root/'configuration_scores.npz',**variants)
    scores.to_parquet(root/'unlabeled_scores.parquet',index=False);save_json(root/'selected.json',selected)
    original=json.loads((root/'initial_SCORES_FROZEN.json').read_text())
    save_json(root/'_SCORES_FROZEN.json',dict(at=time.time(),seconds=time.monotonic()-start,initial_fit_seconds=original['seconds'],
        candidates=len(records),continued_candidates=len(jobs),protocol_sha256=file_digest(root/'protocol.json'),
        scores_sha256=file_digest(root/'unlabeled_scores.parquet'),selected_sha256=file_digest(root/'selected.json'),errors=[]))
    print(json.dumps(dict(phase='REFINEMENT_FROZEN',continued=len(jobs),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    run(p.parse_args())

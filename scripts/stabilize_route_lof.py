"""Fix LOF tie handling in a new preserved benchmark version; no labels."""
import argparse
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
from hss.route.structure import StableRouteLOF,aggregate_losses


def run(a):
    start=time.monotonic();source=Path(a.source);root=Path(a.output);data=Path(a.data)
    if root.exists():raise ValueError('Use a new output directory')
    skip={'evaluation','report','inventory.json','_SUCCESS.json','local_verification.json','process.json','fit.log'}
    shutil.copytree(source,root,ignore=lambda p,n:[f for f in n if f in skip] if Path(p)==source else [])
    for name in ['protocol.json','_SCORES_FROZEN.json','selected.json','unlabeled_scores.parquet']:
        (root/name).rename(root/('pre_lof_'+name.lstrip('_')))
    protocol=json.loads((root/'pre_lof_protocol.json').read_text())
    protocol.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        deterministic_lof=dict(reason='Repeated discrete distances made sklearn LOF depend on query batching; discovered by checkpoint reload checks.',
            tie_break='Stable training-row index',distance='sqrt(2 * differing layer count), identical to one-hot Euclidean',
            source=str(source),script_sha256=file_digest(Path(__file__)),structure_sha256=file_digest(Path('src/hss/route/structure.py'))))
    save_json(root/'protocol.json',protocol)
    scores=pd.read_parquet(root/'pre_lof_unlabeled_scores.parquet');tr=(scores.split=='train').to_numpy();va=(scores.split=='validation').to_numpy()
    meta=pd.read_parquet(data/'inputs/rows.parquet',columns=['sample_id']);assert meta.sample_id.tolist()==scores.sample_id.tolist()
    for f,digest in protocol['inputs'].items():assert file_digest(data/f)==digest
    raw=np.load(data/'inputs/states.npz');states={k:raw[k] for k in ['gmm','mfa']};states['gmm_matched']=np.load(data/'controls/matched_states.npz')['states']
    records=json.loads((root/'candidates.json').read_text());variants=dict(np.load(root/'configuration_scores.npz'))
    for r in records:
        if r['method']!='lof':continue
        t=time.monotonic();p=root/r['path'];enc=joblib.load(root/'models'/r['map']/'encoder.joblib');z=enc.transform(states[r['map']])
        model=StableRouteLOF(protocol['config']['lof_neighbors']).fit(z[tr]);loss=-model.score_samples(z)[:,None]
        check=np.r_[np.where(~tr)[0][:9],np.where(~tr)[0][-5:]]
        np.testing.assert_array_equal(-model.score_samples(z[check]),loss[check,0])
        joblib.dump(model,p/'model.joblib');np.savez_compressed(p/'scores.npz',losses=loss,**aggregate_losses(loss))
        save_json(p/'training.json',dict(deterministic_ties=True,query_batch_invariance_checked=True))
        r.update(initial_seconds=r['seconds'],seconds=time.monotonic()-t,artifacts={f.name:file_digest(f) for f in p.iterdir() if f.is_file() and f.name!='complete.json'})
        save_json(p/'complete.json',r)
        for agg in ['mean','top4']:scores[r['map']+'__lof__'+agg]=loss[:,0]
        variants[r['map']+'__lof__fixed']=loss[:,0]
        print(json.dumps(dict(map=r['map'],seconds=r['seconds'],batch_invariant=True)),flush=True)
    for name in protocol['config']['maps']:
        cols=[c for c in scores if c.startswith(name+'__') and c.endswith('__mean') and '__node__' not in c and '__ensemble__' not in c]
        scores[name+'__ensemble__mean']=np.mean([np.searchsorted(np.sort(scores.loc[va,c]),scores[c],side='right')/(va.sum()+1) for c in cols],axis=0)
    scores.to_parquet(root/'unlabeled_scores.parquet',index=False);np.savez_compressed(root/'configuration_scores.npz',**variants)
    save_json(root/'candidates.json',records);shutil.copy2(root/'pre_lof_selected.json',root/'selected.json')
    save_json(root/'_SCORES_FROZEN.json',dict(at=time.time(),seconds=time.monotonic()-start,candidates=len(records),
        protocol_sha256=file_digest(root/'protocol.json'),scores_sha256=file_digest(root/'unlabeled_scores.parquet'),
        selected_sha256=file_digest(root/'selected.json'),errors=[]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    with threadpool_limits(2):run(p.parse_args())

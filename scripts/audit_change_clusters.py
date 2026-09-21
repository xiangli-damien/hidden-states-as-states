"""Verify change-cluster provenance, saved inference, coverage and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from hss.analysis.channel_data import load_config
from hss.experiments.artifacts import file_digest, save_json, runtime_versions


def audit(cfg):
    root=Path(cfg['output']);cache=Path(cfg['cache']);splits=pd.read_parquet(root/'splits.parquet')
    meta=pd.read_parquet(root/'rows.parquet');np.testing.assert_array_equal(meta.sample_id,splits.sample_id)
    assert len(meta)==cfg['expected_samples'] and not meta.sample_id.duplicated().any()
    pairs=pd.read_parquet(root/'pairs.parquet');np.testing.assert_array_equal(meta.sample_id.to_numpy()[pairs.question_index],pairs.sample_id)
    assert not pairs.duplicated(['sample_id','position']).any()
    assert (pairs.position.to_numpy()>=1).all()
    assert (pairs.position.to_numpy()<meta.n_tokens.to_numpy()[pairs.question_index]).all()
    assert pairs.groupby('sample_id').size().max()<=cfg['token_pairs_per_question']
    manifests=sum([json.loads((root/f'{kind}_views.json').read_text()) for kind in ['depth','temporal']],[])
    checks=[];candidate_count=0;converged=0;stability_count=0
    for spec in manifests:
        name=spec['name'];folder=root/'fits'/name;j=json.loads((folder/'selection.json').read_text())
        feature=cache/'views'/f'{name}.npy';assert file_digest(feature)==j['feature_sha256']
        saved=np.load(folder/'assignments.npz');owner=saved['question_index'];prob=saved['probability']
        expected=(pairs.question_index.to_numpy() if name.startswith('token_delta_') else np.arange(len(meta)))
        np.testing.assert_array_equal(owner,expected)
        np.testing.assert_allclose(prob.sum(1),1,atol=1e-6);assert np.isfinite(prob).all()
        np.testing.assert_array_equal(prob.argmax(1),saved['assignment'])
        train_idx=np.flatnonzero(splits.split.to_numpy()[owner]=='train')
        fingerprint=hashlib.sha256(np.asarray(train_idx,dtype=np.int64).tobytes()).hexdigest()
        for r in j['candidates']:
            assert r['request']['training_index_sha256']==fingerprint
            assert file_digest(root/r['path']/'model.joblib')==r['model_sha256']
            candidate_count+=1;converged+=r['converged']
        best=min([r for r in j['candidates'] if r['converged']],key=lambda r:(r['icl'],r['k'],r['seed']))
        assert (best['k'],best['seed'])==(j['k'],j['seed'])
        gm=joblib.load(folder/'selected_model.joblib');x=np.load(feature,mmap_mode='r')
        query=np.flatnonzero(splits.split.to_numpy()[owner]=='test')[:17]
        p=gm.predict_proba(np.asarray(x[query],dtype=float));error=float(np.max(np.abs(p-prob[query])))
        assert error<2e-5
        for r in j['stability']:
            if r['kind']!='80pct_questions':continue
            stability_count+=1
            ck=json.loads((folder/f'subsample_k{j["k"]}_s{r["seed"]}'/'complete.json').read_text())
            assert ck['converged']==r['converged']
        checks.append(dict(name=name,k=j['k'],vectors=len(x),reload_max_probability_error=error,
                           stability_all_converged=all(r['converged'] for r in j['stability'])))
    outcomes=pd.read_csv(root/'evaluation/outcome_association.csv')
    assert len(outcomes)==len([v for v in manifests if v['kind']!='gaussian_null'])
    assert np.isfinite(outcomes[['js_bits','p','controlled_p','q','controlled_q']]).all().all()
    assert len(json.loads((root/'report/questions.json').read_text()))==len(meta)
    for fn in ['index.html','data.json','overview.png','overview.pdf']:
        assert (root/'report'/fn).stat().st_size>0
    receipt=dict(at=time.time(),questions=len(meta),token_pairs=len(pairs),views=len(manifests),
        candidates=candidate_count,converged_candidates=converged,subsample_refits=stability_count,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime=runtime_versions(),checks=checks)
    save_json(root/'audit.json',receipt)
    inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in sorted(root.rglob('*'))
               if p.is_file() and p.name not in ['inventory.json','_SUCCESS.json'] and not p.name.endswith('.log')}
    save_json(root/'inventory.json',inventory)
    save_json(root/'_SUCCESS.json',dict(questions=len(meta),views=len(manifests),candidates=candidate_count,
        inventory_sha256=file_digest(root/'inventory.json'),files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values())))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','runtime']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/change-clusters.toml');a=p.parse_args();cfg=load_config(a.config)
    with threadpool_limits(cfg['cpu_threads']):audit(cfg)

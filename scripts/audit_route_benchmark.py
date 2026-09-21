"""Validate a completed route benchmark and write portable artifact receipts."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import sklearn
from hss.experiments.artifacts import file_digest,save_json


def run(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    freeze=json.loads((root/'_SCORES_FROZEN.json').read_text())
    records=json.loads((root/'candidates.json').read_text())
    selected=json.loads((root/'selected.json').read_text())
    scores=pd.read_parquet(root/'unlabeled_scores.parquet')
    assert scores.sample_id.is_unique
    assert len(records)==freeze['candidates'] and not json.loads((root/'errors.json').read_text())
    assert file_digest(root/'protocol.json')==freeze['protocol_sha256']
    assert file_digest(root/'unlabeled_scores.parquet')==freeze['scores_sha256']
    assert file_digest(root/'selected.json')==freeze['selected_sha256']
    assert all(int((scores.split==k).sum())==n for k,n in protocol['counts'].items())
    assert np.isfinite(scores.drop(columns=['sample_id','split']).to_numpy()).all()
    for r in records:
        p=root/r['path']
        for name,digest in r['artifacts'].items():assert file_digest(p/name)==digest,(p,name)
        a=np.load(p/'scores.npz');assert a['losses'].shape[0]==len(scores)
        assert np.isfinite(a['losses']).all()
    for row in selected:
        for agg in ['mean','top4']:
            s=np.mean([np.load(root/p/'scores.npz')[agg] for p in row['candidates']],axis=0)
            np.testing.assert_allclose(s,scores[row['name']+'__'+agg])
    evaluation=json.loads((root/'evaluation/summary.json').read_text())
    assert evaluation['test_n']==int((scores.split=='test').sum())
    assert len(json.loads((root/'report/samples.json').read_text()))==evaluation['test_n']
    provenance=dict(at=time.time(),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        fit_commit=protocol['code_commit'],evaluation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evaluation_source_sha256=file_digest(Path('scripts/evaluate_route_unsupervised.py')),
        audit_source_sha256=file_digest(Path(__file__)))
    save_json(root/'evaluation/provenance.json',provenance)
    exclude={'inventory.json','_SUCCESS.json','local_verification.json'}
    inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in exclude}
    save_json(root/'inventory.json',inventory)
    result=dict(complete=True,at=time.time(),candidates=len(records),selected_methods=len(selected),
        samples=len(scores),files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
        inventory_sha256=file_digest(root/'inventory.json'),source_commit=provenance['evaluation_commit'])
    save_json(root/'_SUCCESS.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    run(p.parse_args().root)

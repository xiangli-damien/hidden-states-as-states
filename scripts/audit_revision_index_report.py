"""Recompute all reported rates/paired intervals from independently audited cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from revision_common import sha,write_json


def check_ci(values,record):
    assert len(values)==record['n']
    if not values:
        assert all(record[k] is None for k in ['estimate','low','high']);return
    rng=np.random.default_rng(42);x=np.asarray(values,dtype=float)
    boot=[np.mean(x[rng.choice(len(x),size=len(x),replace=True)]) for _ in range(2000)]
    lo,hi=np.percentile(boot,[2.5,97.5])
    np.testing.assert_allclose([record['estimate'],record['low'],record['high']],[np.mean(x),lo,hi],atol=1e-12)


def run(root):
    report=root/'report';receipt=json.loads((report/'_SUCCESS.json').read_text())
    for filename,digest in receipt['files'].items():assert sha(report/filename)==digest
    assert receipt['validation_audit_sha256']==sha(root/'validation_audit.json')
    gate=json.loads((root/'validation_audit.json').read_text())['capability_gate']
    rows=json.loads((report/'summary.json').read_text());pairs=json.loads((report/'paired.json').read_text())
    if not gate['passed']:
        assert rows==pairs==receipt['primary']==[] and receipt['test_audit_sha256'] is None
        assert not (root/'test/_SUCCESS.json').exists()
    else:
        assert receipt['test_audit_sha256']==sha(root/'test_audit.json')
        cases=[json.loads(p.read_text()) for p in sorted((root/'test/cases').glob('*.json'))]
        raw={}
        for r in cases:
            for p in r['patches']:raw[r['case_id'],p['width'],p['method']]=p['score']
        ids=sorted(r['case_id'] for r in cases if r['eligible'])
        assert len(cases)==48 and len(raw)==len(ids)*8
        assert len(rows)==40 and len(pairs)==6
        for r in rows:
            values=[int(raw[sid,r['width'],r['method']][r['metric']]) for sid in ids]
            check_ci(values,r)
            assert r['all_candidate_pairs']==48 and r['eligible_pairs']==len(ids)
            assert r['coverage']==len(ids)/48 and r['coverage_adjusted_success']==sum(values)/48
        for r in pairs:
            delta=[int(raw[sid,r['width'],'local8'][r['metric']])-int(raw[sid,r['width'],'shared8'][r['metric']]) for sid in ids]
            check_ci(delta,r)
            assert r['primary']==(r['width']==16 and r['metric']=='joint_success')
        assert receipt['primary']==[r for r in pairs if r['primary']]
    result={'passed':True,'validation_gate_passed':gate['passed'],'summary_rows':len(rows),'paired_rows':len(pairs),
            'report_receipt_sha256':sha(report/'_SUCCESS.json')}
    write_json(root/'report_audit.json',result);print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)

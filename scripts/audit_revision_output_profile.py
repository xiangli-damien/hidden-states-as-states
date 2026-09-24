"""Independent statistics audit, without importing profile/report implementations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_stats(values, r):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    assert len(x)==r['n']
    if not len(x):return
    rng=np.random.default_rng(42)
    draws=np.asarray([x[rng.integers(len(x),size=len(x))].mean() for _ in range(2000)])
    lo,hi=np.quantile(draws,[.025,.975]);k=int(np.ceil(.1*len(x)))
    np.testing.assert_allclose([r['mean'],r['low'],r['high'],r['median'],r['fraction_positive']],
        [x.mean(),lo,hi,np.median(x),(x>0).mean()],rtol=1e-11,atol=1e-12)
    assert r['top10pct_n']==k
    positive=x[x>0]
    if len(positive):
        np.testing.assert_allclose(r['top10pct_share_of_positive_sum'],np.sort(positive)[-k:].sum()/positive.sum(),atol=1e-12)
    if len(x)>k:
        np.testing.assert_allclose(r['mean_after_dropping_largest10pct'],np.sort(x)[:-k].mean(),atol=1e-12)


def run(root):
    report=root/'report'
    receipt=json.loads((report/'_SUCCESS.json').read_text())
    assert receipt['source_receipt_sha256']==sha(root/'_SUCCESS.json')
    for name,digest in receipt['files'].items():assert sha(report/name)==digest
    source_receipt=json.loads((root/'_SUCCESS.json').read_text())
    for name,digest in source_receipt['files'].items():assert sha(root/name)==digest
    frame=pd.read_parquet(root/'conditions.parquet')
    manifest=json.loads((root/'input_hashes.json').read_text())
    # Rebuild the reported loss columns and KL directly from source audited JSON,
    # independently of the profiler's aggregation of saved arrays.
    sources={}
    for path,digest in manifest.items():
        if '/functional/samples/' in path and path.endswith('.json'):
            p=Path(path);assert sha(p)==digest
            r=json.loads(p.read_text());t=r['task'];c=t['condition']
            method='_'.join(str(c[k]) for k in ['method','rank','alpha','seed'] if k in c)
            key=(t['sample_id'],t['width'],method)
            assert key not in sources;sources[key]=r
    assert len(sources)==len(frame)==receipt['conditions']
    for r in frame.to_dict('records'):
        source=sources[(r['sample_id'],r['width'],r['method'])]
        baseline=sources[(r['sample_id'],r['width'],'identity')]
        np.testing.assert_allclose(r['next_token_kl'],source['next_token_kl'],atol=1e-10,rtol=1e-10)
        for target,field in [('first_token','first_token_nll'),('first16','first16_nll'),('full','nll')]:
            np.testing.assert_allclose(r['nll_'+target],source[field],atol=1e-12,rtol=1e-10)
            np.testing.assert_allclose(r['delta_nll_'+target],source[field]-baseline[field],atol=1e-12,rtol=1e-10)
    summary=pd.read_parquet(report/'summary.parquet');pairs=pd.read_parquet(report/'paired.parquet')
    for r in summary.to_dict('records'):
        sub=frame.loc[frame.cohort.eq(r['cohort'])&frame.width.eq(r['width'])&frame.method.eq(r['method'])]
        check_stats(sub[r['metric']],r)
    for r in pairs.to_dict('records'):
        sub=frame.loc[frame.cohort.eq(r['cohort'])]
        a=sub.loc[sub.width.eq(r['width_a'])&sub.method.eq(r['method_a'])].set_index('sample_id')
        b=sub.loc[sub.width.eq(r['width_b'])&sub.method.eq(r['method_b'])].set_index('sample_id')
        ids=sorted(a.index);assert ids==sorted(b.index)
        check_stats(a.loc[ids,r['metric']].to_numpy()-b.loc[ids,r['metric']].to_numpy(),r)
    result={'passed':True,'raw_conditions_checked':len(frame),'summary_rows':len(summary),'paired_rows':len(pairs),
            'report_receipt_sha256':sha(report/'_SUCCESS.json'),'source_receipt_sha256':sha(root/'_SUCCESS.json')}
    (root/'statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)

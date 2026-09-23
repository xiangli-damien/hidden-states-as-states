"""Independent raw-record -> paired question statistics audit; no producer import."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import sha,write_json


def ci(values):
    values=np.array(values);rng=np.random.default_rng(42)
    draws=[float(np.mean(values[rng.integers(len(values),size=len(values))])) for _ in range(2000)]
    return float(np.mean(values)),*np.percentile(draws,[2.5,97.5])


def run(root):
    report=root/'report';stage=root/'functional'
    receipt=json.loads((report/'_SUCCESS.json').read_text())
    audit=json.loads((stage/'audit.json').read_text());assert audit['complete']
    assert receipt['source_audit_sha256']==sha(stage/'audit.json')
    for name,digest in receipt['files'].items():assert sha(report/name)==digest,name
    input_hashes=json.loads((stage/'audit_inputs.json').read_text());records=[]
    for name,digest in input_hashes.items():
        path=stage/'samples'/name;assert sha(path)==digest;records.append(json.loads(path.read_text()))
    assert len(records)==receipt['conditions']==audit['conditions']
    originals={}
    for r in records:
        t=r['task']
        if t['condition']['method']=='identity':originals[t['dataset'],t['sample_id']]=r
    frame=pd.read_parquet(report/'per_question.parquet')
    assert len(frame)==len(records) and not frame.duplicated(['dataset','split','sample_id','method']).any()
    indexed=frame.set_index(['dataset','split','sample_id','method']);values=defaultdict(dict)
    metrics=['kl','delta_nll','delta_first16_nll','mse','region_retention']
    for r in records:
        t=r['task'];c=t['condition'];method=c['method']+(f'_{c["rank"]}' if 'rank' in c else '')
        identity=originals[t['dataset'],t['sample_id']]
        data=[r['next_token_kl'],r['nll']-identity['nll'],r['first16_nll']-identity['first16_nll'],
              sum(r['geometry']['actual']['token_delta_energy'])/(16*3584),r['geometry']['actual']['state_retained_fraction']]
        group=(t['dataset'],r['split'],method)
        np.testing.assert_allclose(indexed.loc[(group[0],group[1],t['sample_id'],method),metrics].to_numpy(float),data,rtol=1e-12,atol=1e-12)
        values[group][t['sample_id']]=dict(zip(metrics,data))
    summaries=json.loads((report/'summary.json').read_text());pairs=json.loads((report/'paired.json').read_text())
    assert len(summaries)==len(values)*len(metrics)
    seen=set()
    for r in summaries:
        group=(r['dataset'],r['split'],r['method']);metric=r['metric']
        assert (group,metric) not in seen;seen.add((group,metric))
        ordered=values[group];vals=[ordered[sid][metric] for sid in sorted(ordered)]
        assert r['n']==len(vals)
        np.testing.assert_allclose([r['estimate'],r['low'],r['high']],ci(vals),rtol=1e-11,atol=2e-12)
    groups={(g[0],g[1]) for g in values};expected_pairs=set()
    for dataset,split in groups:
        for rank in [8,4,16]:
            for a,b in [('fa','pca'),('fa_orthogonal','pca'),('fa','fa_orthogonal'),
                        ('mfa_hard','pca'),('mfa_soft','mfa_hard'),('mfa_soft','pca')]:
                for metric in metrics:expected_pairs.add((dataset,split,f'{a}_{rank}',f'{b}_{rank}',metric))
    assert len(pairs)==len(expected_pairs)
    for r in pairs:
        pair=(r['dataset'],r['split'],r['method_a'],r['method_b'],r['metric'])
        assert pair in expected_pairs;expected_pairs.remove(pair)
        left=values[pair[0],pair[1],pair[2]];right=values[pair[0],pair[1],pair[3]]
        assert set(left)==set(right) and r['n']==len(left)
        delta=[left[sid][pair[4]]-right[sid][pair[4]] for sid in sorted(left)]
        np.testing.assert_allclose([r['estimate'],r['low'],r['high']],ci(delta),rtol=1e-11,atol=2e-12)
        assert r['primary']==(pair[2]=='fa_8' and pair[3]=='pca_8' and pair[4] in ['kl','delta_nll'])
    primary=[r for r in pairs if r['primary']]
    assert primary==receipt['primary']==json.loads((report/'primary.json').read_text())
    assert {f'{d}_{sid}' for d,sid in originals}=={p.stem for p in (report/'questions').glob('*.html')}
    outcome={'complete':True,'conditions':len(records),'questions':len(originals),'summary_rows':len(summaries),
        'paired_rows':len(pairs),'primary_rows':len(primary),'raw_audit_sha256':sha(stage/'audit.json'),
        'report_receipt_sha256':sha(report/'_SUCCESS.json'),'audit_code_sha256':sha(Path(__file__)),
        'unit':'One question; token positions and methods are not independent observations. Pointwise 95% intervals.'}
    write_json(report/'statistics_audit.json',outcome);print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)

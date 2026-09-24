"""Independently verify all-record coverage, paired statistics and preserved inputs.

Does not import the extractor or infer semantic correctness from reproducibility.
"""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())


def run(root, out):
    summary=read(out/'summary.json');plan=read(out/'plan.json')
    assert summary['complete'] and not summary['full_semantic_rescoring_complete']
    assert plan['input_manifest_sha256']==sha(root/'records.json')
    assert plan['original_summary_sha256']==sha(root/'summary.json')
    assert summary['scores_sha256']==sha(out/'scores.jsonl')
    assert summary['plan_sha256']==sha(out/'plan.json')
    for p,h in plan['source_hashes'].items(): assert sha(p)==h,p
    rows=[json.loads(line) for line in (out/'scores.jsonl').read_text().splitlines()]
    assert len(rows)==summary['records']==plan['expected_records']
    bypath={r['path']:r for r in rows};assert len(bypath)==len(rows)
    manifest=read(root/'records.json');assert set(bypath)=={r['path'] for r in manifest}
    for m in manifest:
        assert sha(m['path'])==m['sha256']==bypath[m['path']]['source_sha256']
        raw=read(m['path']);r=bypath[m['path']]
        assert r['original_correct']==bool(raw['correct']) and r['original_extracted']==raw['parsed_answer']
        assert r['sample_id']==raw['sample_id'] and r['condition']==raw['condition']['name']
    changed=[r for r in rows if r['original_correct']!=r['new_correct']]
    assert changed==read(out/'changed_records.json') and len(changed)==summary['changed_labels']
    assert sum(r['new_correct'] for r in changed)==summary['false_to_true']
    assert sum(not r['new_correct'] for r in changed)==summary['true_to_false']
    assert sum(bool(r['flags']) for r in rows)==summary['flagged_records']
    original={(r['group'],r['method'],r['control']):r for r in read(root/'summary.json')['comparisons']}
    for c in summary['comparisons']:
        key=c['group'],c['method'],c['control']
        if key in original:
            pairs=[(bypath[p['method_record']],[bypath[p['control_record']]]) for p in original[key]['per_question']]
        else:
            test={}
            for r in rows:
                if r['split']=='test':test.setdefault(r['sample_id'],{})[r['condition']]=r
            controls=['shared8'] if c['control']=='shared8' else ['random_42','random_137','random_271']
            pairs=[(v['c1_1.0'],[v[k] for k in controls]) for _,v in sorted(test.items())]
        assert c['flagged_pairs']==sum(bool(a['flags'] or any(b['flags'] for b in bs)) for a,bs in pairs)
        for version,field in [('original','original_correct'),('rescored','new_correct')]:
            a=np.asarray([a[field] for a,bs in pairs],float);b=np.asarray([sum(x[field] for x in bs)/len(bs) for a,bs in pairs])
            z=c[version];delta=a-b;n=len(a)
            assert z['n']==n
            np.testing.assert_allclose([z['method_accuracy'],z['control_accuracy'],z['delta']],[a.mean(),b.mean(),delta.mean()],atol=1e-14,rtol=0)
            indices=np.random.default_rng(9242026).integers(0,n,(2000,n))
            interval=np.quantile(delta[indices].mean(axis=1),[.025,.975])
            np.testing.assert_allclose(interval,z['ci95'],atol=1e-14,rtol=0)
            if 'repair' in z:
                assert z['repair']==int(((a==1)&(b==0)).sum()) and z['damage']==int(((a==0)&(b==1)).sum())
    for p,h in read(out/'_SUCCESS.json')['files'].items():assert sha(out/p)==h,p
    receipt=dict(complete=True,records=len(rows),comparisons=len(summary['comparisons']),
        all_source_hashes_unchanged=True,independent_statistics_recomputed=True,
        semantic_correctness_verified=False,summary_sha256=sha(out/'summary.json'),audit_code_sha256=sha(__file__))
    path=out/'statistics_audit.json'
    if path.exists():raise RuntimeError('Preserve existing audit receipt')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root,a.output)

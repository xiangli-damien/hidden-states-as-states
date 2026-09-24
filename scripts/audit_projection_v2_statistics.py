"""Rebuild every paired comparison from immutable per-question output records."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import sha,write_json


def run(root):
    summary=json.loads((root/'summary.json').read_text());plan=json.loads((root/'plan.json').read_text())
    main=Path(plan['config']['protected_primary']);parent=json.loads((main/'steering_summary.json').read_text())
    assert summary['protected_primary_summary_sha256']==sha(main/'steering_summary.json')
    assert summary['primary_rows']==[r for r in parent['rows'] if r['stage']=='test'] and summary['primary_paired']==parent['paired_controls']
    records={}
    for item in json.loads((root/'records.json').read_text()):
        assert sha(item['path'])==item['sha256'];records[item['path']]=json.loads(Path(item['path']).read_text())
    cases=json.loads((root/'case_review_key.json').read_text());expected=set()
    for c in summary['comparisons']:
        pairs=c['per_question'];assert len(pairs)==c['n'] and len({p['sample_id'] for p in pairs})==c['n']
        a=[];b=[];selected=[]
        for pair in pairs:
            left=records[pair['method_record']];right=records[pair['control_record']]
            assert left['sample_id']==right['sample_id']==pair['sample_id']
            assert left['ground_truth']==right['ground_truth'] and left['prompt_text']==right['prompt_text']
            a.append(int(left['correct']));b.append(int(right['correct']));selected.append(left)
            if a[-1]!=b[-1]:expected.add((c['group'],c['method'],c['control'],pair['sample_id']))
        a=np.asarray(a);b=np.asarray(b);delta=a-b
        assert c['wrong_to_correct']==int(((a==1)&(b==0)).sum()) and c['correct_to_wrong']==int(((a==0)&(b==1)).sum())
        np.testing.assert_allclose([c['accuracy'],c['control_accuracy'],c['delta_accuracy']['estimate']],[a.mean(),b.mean(),delta.mean()],rtol=0,atol=1e-14)
        rng=np.random.default_rng(9242026);samples=[]
        for _ in range(2000):samples.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
        np.testing.assert_allclose(c['delta_accuracy']['ci95'],np.percentile(samples,[2.5,97.5]),atol=1e-14)
        assert c['delta_accuracy']['n']==len(a)
        np.testing.assert_allclose(c['mean_length'],np.mean([r['length'] for r in selected]))
        np.testing.assert_allclose(c['parse_failure_rate'],np.mean([r['parse_failed'] for r in selected]))
        np.testing.assert_allclose(c['truncation_rate'],np.mean([r['finish_reason']=='length' for r in selected]))
    assert {(c['group'],c['method'],c['control'],c['sample_id']) for c in cases}==expected
    assert len(cases)==len(expected)==summary['case_pairs']
    frame=pd.read_csv(root/'geometry_behavior.csv');assert len(frame)==len(records) and not frame.record_path.duplicated().any()
    assert np.isfinite(frame.energy).all() and (frame.energy>=0).all()
    write_json(root/'statistics_audit.json',dict(complete=True,comparisons=len(summary['comparisons']),records=len(records),
        all_flip_pairs_included=len(cases),summary_sha256=sha(root/'summary.json'),
        records_sha256=sha(root/'records.json'),geometry_sha256=sha(root/'geometry_behavior.csv'),
        case_key_sha256=sha(root/'case_review_key.json'),audit_code_sha256=sha(Path(__file__)),semantic_review_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)

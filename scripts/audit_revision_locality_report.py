"""Independently rebuild question statistics from execution-audited JSON records.

Does not import the report's aggregation, selection, or interval functions. Raw
logits and bf16 geometry are checked separately by audit_revision_locality.py.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = ['split', 'prefix_tokens', 'layer', 'role', 'width']
METRICS = ['next_token_kl', 'delta_nll', 'delta_first16_nll', 'delta_first_token_nll',
           'next_token_argmax_agreement', 'actual_patch_energy',
           'state_retained_fraction', 'mse_per_coordinate']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    draw = np.random.default_rng(42).integers(0, len(values), (2000, len(values)))
    lo, hi = np.percentile(np.mean(values[draw], axis=1), [2.5, 97.5])
    return {'estimate': values.mean(), 'low': lo, 'high': hi, 'n': len(values)}


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-12)


def run(root):
    stage = root/'functional'; report = root/'report'
    audit = json.loads((stage/'audit.json').read_text())
    receipt = json.loads((report/'_SUCCESS.json').read_text())
    assert audit['complete'] and receipt['source_audit_sha256'] == sha(stage/'audit.json')
    manifest = json.loads((stage/'audit_inputs.json').read_text())
    records = []
    for path in sorted((stage/'samples').glob('*.json')):
        assert sha(path) == manifest[path.name]
        records.append(json.loads(path.read_text()))
    assert len(records) == audit['expected'] == receipt['conditions']
    for name, digest in receipt['files'].items():
        assert sha(report/name) == digest
    frame = pd.read_parquet(report/'per_question_seed_average.parquet')
    raw_report = pd.read_parquet(report/'individual_conditions.parquet')
    assert len(raw_report) == len(records)
    raw_index = raw_report.set_index(GROUP+['sample_id','individual_method'])
    assert raw_index.index.is_unique
    baselines = {}
    for r in records:
        t = r['task']; c = t['condition']
        group = (r['split'], t['prefix_tokens'], t['layer'], t['role'], t['width'], t['sample_id'])
        if c['method'] == 'identity':
            assert group not in baselines
            baselines[group] = r
    independent = defaultdict(list)
    for r in records:
        t = r['task']; c = t['condition']
        group = (r['split'], t['prefix_tokens'], t['layer'], t['role'], t['width'], t['sample_id'])
        base = baselines[group]
        method = '_'.join(str(c[k]) for k in ['method','rank','alpha'] if k in c)
        individual = method + (f'_{c["seed"]}' if 'seed' in c else '')
        g = r['geometry']['actual']
        values = {m: r[m] for m in ['next_token_kl','next_token_argmax_agreement','actual_patch_energy']}
        for metric in ['nll','first16_nll','first_token_nll']:
            values['delta_'+metric] = r[metric]-base[metric]
        values['state_retained_fraction'] = g['state_retained_fraction']
        values['mse_per_coordinate'] = np.mean(g['token_delta_energy'])/3584
        close([values[m] for m in METRICS], raw_index.loc[group+(individual,), METRICS].to_numpy(float))
        independent[group+(method,)].append(values)
    assert len(independent) == len(frame) == receipt['question_averaged_conditions']
    result = {}
    for group, repeated in independent.items():
        result[group] = {m: np.mean([v[m] for v in repeated]) for m in METRICS}
    for r in frame.to_dict('records'):
        group = tuple(r[k] for k in GROUP+['sample_id','method'])
        close([r[m] for m in METRICS], [result[group][m] for m in METRICS])
    assert not frame.duplicated(GROUP+['sample_id','method']).any()
    vectors = defaultdict(dict)
    for group, values in result.items():
        vectors[group[:5]+(group[-1],)][group[5]] = values
    summary = json.loads((report/'summary.json').read_text())
    for row in summary:
        key = tuple(row[k] for k in GROUP+['method'])
        data = vectors[key]
        assert row['n'] == len(data)
        for metric in METRICS:
            actual = interval([data[sid][metric] for sid in sorted(data)])
            for stat in ['estimate','low','high']: close(row[f'{metric}_{stat}'], actual[stat])
    pairs = json.loads((report/'paired_methods.json').read_text())
    for row in pairs:
        key = tuple(row[k] for k in GROUP)
        a = vectors[key+(row['method_a'],)]; b = vectors[key+(row['method_b'],)]
        assert set(a) == set(b) and row['n'] == len(a)
        actual = interval([a[sid][row['metric']]-b[sid][row['metric']] for sid in sorted(a)])
        for stat in ['estimate','low','high']: close(row[stat], actual[stat])
    expected_primary = [r for r in pairs if r['primary'] and r['metric'] in ['next_token_kl','delta_nll']]
    assert expected_primary == json.loads((report/'primary.json').read_text()) == receipt['primary']
    protocol = json.loads((root/'mse_match_plan.json').read_text())['config']
    matches = json.loads((report/'validation_mse_match.json').read_text())
    vkey = tuple(protocol['view'][k] for k in GROUP[1:])
    def vector(split, method, metric):
        data = vectors[(split,)+vkey+(method,)]
        return np.array([data[sid][metric] for sid in sorted(data)])
    for row in matches:
        local = f'local_{row["local_rank"]}'
        val_local = vector('validation', local, 'mse_per_coordinate').mean()
        candidate_mse = {r: vector('validation', f'shared_{r}', 'mse_per_coordinate').mean()
                         for r in protocol['shared_ranks']}
        rank = min(candidate_mse, key=lambda r: (abs(np.log(candidate_mse[r]/val_local)), r))
        assert rank == row['selected_shared_rank']
        close(row['validation_local_mse'], val_local)
        close(row['validation_shared_mse'], candidate_mse[rank])
        val_gap = abs(candidate_mse[rank]/val_local-1)
        close(row['relative_validation_MSE_gap'], val_gap)
        assert bool(val_gap <= .1) == row['within_predeclared_10pct_validation_gap']
        for metric in ['next_token_kl','delta_nll','delta_first16_nll','mse_per_coordinate']:
            a = vector('test', local, metric); b = vector('test', f'shared_{rank}', metric)
            actual = interval(a-b)
            for stat in ['estimate','low','high','n']:
                close(row['test_'+metric+'_local_minus_shared'][stat], actual[stat])
        test_local = vector('test', local, 'mse_per_coordinate').mean()
        test_shared = vector('test', f'shared_{rank}', 'mse_per_coordinate').mean()
        close(row['test_local_mse'], test_local); close(row['test_shared_mse'], test_shared)
        close(row['relative_test_MSE_gap'], abs(test_shared/test_local-1))
    ids = {r['task']['sample_id'] for r in records}
    assert ids == {p.stem for p in (report/'questions').glob('*.html')}
    outcome = {'complete': True, 'conditions': len(records), 'questions': len(ids),
               'seed_averaged_rows': len(frame), 'summary_rows': len(summary), 'paired_rows': len(pairs),
               'checked': 'Raw JSON -> baseline differences -> within-question seed means -> paired question bootstrap; validation-only MSE choice; report SHA and question-page coverage.',
               'scope': 'Statistical report audit; real logits and geometry have a separate execution audit.',
               'raw_audit_sha256': sha(stage/'audit.json'), 'report_receipt_sha256': sha(report/'_SUCCESS.json'),
               'audit_code_sha256': sha(Path(__file__))}
    (report/'statistics_audit.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path)
    run(p.parse_args().root)

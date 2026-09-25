"""Independent CPU audit of exported follow-up tables; does not rerun models.

The runner separately checks raw-record receipts. This script checks the sealed
table hashes and recalculates the reported aggregates without importing the
experiment's decision/statistics helpers.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    hashes = {}
    for step, table in [(1, 'step1_reencode.parquet'), (2, 'step2_screen_long.parquet')]:
        hashes[table] = hashlib.sha256((root / table).read_bytes()).hexdigest()
        assert hashes[table] == read(f'step{step}_COMPLETE.json')['parquet_sha256']
    frame = pd.read_parquet(root / 'step2_screen_long.parquet')
    dec = read('step2_decision.json')
    params = dec['params']
    assert not frame.duplicated(['question_id', 'condition']).any()
    assert len(frame) == 6895
    assert frame.condition.nunique() == 35
    for group, n, fields in [('sink', 97, ['m_ans']),
                             ('normal', 100, ['m_col_ans', 'm_col_nll'])]:
        f = frame[frame['set'] == group]
        assert f.question_id.nunique() == n
        assert (f.groupby('question_id').size() == 35).all()
        assert np.isfinite(f[fields].to_numpy()).all()
    sink = frame[frame['set'] == 'sink'].pivot(index='question_id', columns='condition', values='m_ans')
    normal = frame[frame['set'] == 'normal'].pivot(index='question_id', columns='condition', values='m_col_ans')
    sd, nd = sink.subtract(sink.NONE, axis=0), normal.subtract(normal.NONE, axis=0)
    passes = []
    for c, fam in [('C0', 'CTRL_ADD'), ('C1', 'CTRL_CLAMP'), ('C2', 'CTRL_ADD'), ('C3', 'CTRL_PROJ')]:
        r = dec['per_candidate'][c]
        delta = sd[c].to_numpy()
        rng = np.random.default_rng(params['seed'])
        means = np.array([delta[rng.integers(len(delta), size=len(delta))].mean()
                          for _ in range(params['bootstrap_B'])])
        ci = np.quantile(means, [.025, .975])
        np.testing.assert_allclose(ci, r['ci95_d_ans'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(delta.mean(), r['mean_d_ans'], rtol=0, atol=1e-12)
        controls = [s for s in sd.columns if s.startswith(fam + '_')]
        assert len(controls) == 10
        exceeded = int((delta.mean() > sd[controls].mean()).sum())
        p1 = bool(ci[0] > 0)
        p2 = exceeded >= math.ceil(params['controls_fraction'] * len(controls))
        p3 = bool(nd[c].mean() >= max(nd[controls].mean().min(), params['collateral_abs_bound']))
        assert [p1, p2, p3] == [r['P1_ci_above_zero'], r['P2_beats_controls'], r['P3_collateral_ok']]
        assert exceeded == r['n_controls_exceeded']
        np.testing.assert_allclose(nd[c].mean(), r['mean_d_col_ans'], rtol=0, atol=1e-12)
        if p1 and p2 and p3:
            passes.append(c)
    assert passes == dec['pass_set'] == []
    assert dec['step3'] == '3B' and dec['chosen'] is None
    assert read('step3_status.json')['state'] == 'blocked_missing_artifacts'

    one = pd.read_parquet(root / 'step1_reencode.parquet')
    summary = read('step1_summary.json')
    assert len(one) == 620 and not one.duplicated(['sample_id', 'arm']).any()
    one['in_sink'] = one.global_ids.map(lambda s: json.loads(s)[14] == summary['sink_global'])
    for group, n in [('sink_test', 55), ('normal_test', 100)]:
        f = one[one['split'] == group]
        expected = summary['groups'][group]
        assert f.sample_id.nunique() == n
        for arm in ['historical', 'zero', 'hss', 'random']:
            sub = f[f.arm == arm]
            assert len(sub) == n
            for field in ['correct', 'boxed', 'original_or_forced_correct', 'in_sink']:
                assert int(sub[field].sum()) == expected[arm][field]
        z = f[f.arm == 'zero']
        assert float(z.historical_text_exact.mean()) == expected['text_reproduction_rate']
        ids = z.loc[z.in_sink, 'sample_id']
        assert len(ids) == expected['Q0_n']
        for arm, key in [('hss', 'a_success'), ('random', 'b_success')]:
            sub = f[(f.arm == arm) & f.sample_id.isin(ids)]
            assert int((~sub.in_sink).sum()) == expected['exit_hss_vs_random'][key]
    receipt = dict(passed=True, scope='Exported table hashes, complete grids, independent paired question bootstrap and D6 gates, Step1 counts; no new generation or raw-record audit.',
                   hashes=hashes, bootstrap_replicates_per_candidate=params['bootstrap_B'],
                   candidates=4, step1_rows=len(one), step2_rows=len(frame), pass_set=passes)
    (root / 'independent_table_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    audit(parser.parse_args().root)

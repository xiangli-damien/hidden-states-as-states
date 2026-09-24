"""Synthetic report tests, including degenerate coverage and corrupt statistics."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from revision_common import write_json, sha
from revision_sameprompt_report_metrics import cross_question, monitoring
from audit_revision_sameprompt_report import pooled, equal, check_monitor, run as audit
from report_revision_sameprompt import run as report


def test_question_bootstrap_with_ties_and_empty_questions():
    frame = pd.DataFrame({'question_group': ['a', 'a', 'b', 'c'], 'failure': [0, 1, 1, 0],
                          'baseline': [.5, .5, .8, .2]})
    equal(cross_question(frame, 'baseline', ['a', 'b', 'c', 'empty'], 100),
          pooled(frame, 'baseline', ['a', 'b', 'c', 'empty'], 100))
    single = frame[frame.failure == 0]
    got = cross_question(single, 'baseline', ['a', 'b', 'c', 'empty'], 100)
    assert got['auroc']['estimate'] is None and got['auroc']['valid_bootstraps'] == 0
    empty = cross_question(frame.iloc[:0], 'baseline', ['a', 'b', 'c', 'empty'], 100)
    assert all(r['estimate'] is None for r in empty.values())


def test_monitor_early_end_negative_infinity_and_unavailable():
    universe = pd.DataFrame({'trajectory_id': ['c0', 'c1', 't0', 't1'], 'question_group': ['c', 'c', 't', 't'],
                            'role': ['calibration', 'calibration', 'test', 'test'], 'failure': [0, 1, 0, 1],
                            'observed_length': [3, 100, 100, 40]})
    views = {}
    for name, ids in [('p16_l28', ['c1', 't0', 't1']), ('p64_l28', ['c1', 't0'])]:
        f = universe[universe.trajectory_id.isin(ids)].copy()
        for method in ['baseline', 'mean16', 'duplicate_current']:
            f[method] = .5
        views[name] = f
    results, table = monitoring(universe, views, 30)
    assert results['baseline']['threshold']['kind'] == 'negative_infinity'
    assert results['baseline']['calibration_far'] == 0
    assert results['baseline']['test']['far']['estimate'] == 1
    check_monitor(universe, views, results, table, 30, .1)
    universe.loc[universe.role == 'calibration', 'failure'] = 1
    results, table = monitoring(universe, views, 30)
    assert results['baseline']['test'] is None
    assert table.alarmed.isna().all()
    check_monitor(universe, views, results, table, 30, .1)


def synthetic_report_fixture(root):
    rng = np.random.default_rng(9)
    cfg = {'role_counts': {'train': 96, 'tuning': 32, 'calibration': 32, 'test': 64},
           'bootstrap_questions': 40, 'far_target': .1, 'scope': 'SYNTHETIC TEST ONLY'}
    inputs = []; rows = []; raw = {}; predictions = {n: [] for n in ['p16_l28', 'p16_l14', 'p64_l28']}
    for role, count in cfg['role_counts'].items():
        for _ in range(count):
            i = len(inputs); sid = f'synthetic_{i:03}'
            inputs.append(dict(sample_id=sid, question_group=sid, role=role, prompt_text=f'Synthetic question {i} <x>',
                model_input_text=f'User: question {i}', ground_truth='1', category='Synthetic', level='1'))
            for repeat in range(4):
                tid = f'{sid}_r{repeat}'; failure = int((i % 5) == 1 or (i % 5 > 1 and repeat % 2))
                length = 10 if i % 19 == 0 or (i % 7 == 0 and repeat == 0) else 40 if repeat == 1 else 100
                finish = 'length' if repeat == 3 else 'eos'
                meta = dict(trajectory_id=tid, sample_id=sid, question_group=sid, role=role,
                            failure=failure, observed_length=length, finish_reason=finish)
                rows.append(meta)
                record = dict(trajectory={'trajectory_id': tid}, sample_id=sid, question_group=sid,
                    role=role, correct=not failure, length=length, finish_reason=finish,
                    generated_ids=[i % 3]*length, response_text=f'Synthetic response {tid} <test>', parsed_answer='1',
                    prefixes={str(p): {'valid': length > p} for p in [16, 64]})
                path = root/'full/samples'/f'{tid}.json'; write_json(path, record); raw[path.name] = sha(path)
                for view in predictions:
                    prefix = int(view.split('_')[0][1:])
                    if length <= prefix:
                        continue
                    entry = dict(meta)
                    for method in ['baseline', 'mean16', 'duplicate_current']+(['allmean'] if prefix == 64 else []):
                        entry[method] = float(rng.uniform(.1, .9))
                    predictions[view].append(entry)
    write_json(root/'inputs.json', inputs)
    write_json(root/'plan.json', {'config': cfg, 'inputs_sha256': sha(root/'inputs.json')})
    write_json(root/'full/_SUCCESS.json', dict(complete=True, records=raw, synthetic_only=True))
    write_json(root/'full/audit.json', dict(complete=True, stage_receipt_sha256=sha(root/'full/_SUCCESS.json'), synthetic_only=True))
    dest = root/'readouts'; dest.mkdir(); pd.DataFrame(rows).to_parquet(dest/'all_trajectories.parquet', index=False)
    for view, items in predictions.items():
        folder = dest/view; folder.mkdir(); pd.DataFrame(items).to_parquet(folder/'predictions.parquet', index=False)
        for method in ['baseline', 'mean16', 'duplicate_current']+(['allmean'] if view == 'p64_l28' else []):
            write_json(folder/method/'selection.json', dict(selected={'C': .01}, synthetic_only=True))
    files = {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    write_json(dest/'_SUCCESS.json', dict(complete=True, files=files, synthetic_only=True))
    write_json(dest/'audit.json', dict(complete=True, readout_receipt_sha256=sha(dest/'_SUCCESS.json'), synthetic_only=True))


def test_full_synthetic_report_independent_audit_and_tamper(tmp_path):
    synthetic_report_fixture(tmp_path)
    report(tmp_path); audit(tmp_path)
    assert json.loads((tmp_path/'report/statistics_audit.json').read_text())['complete']
    assert len(list((tmp_path/'report/questions').glob('*.html'))) == 224
    with pytest.raises(RuntimeError, match='namespace exists'):
        report(tmp_path)
    # Re-seal a wrong summary: the independent arithmetic audit must still reject it.
    p = tmp_path/'report/summary.json'; summary = json.loads(p.read_text())
    summary['views']['p16_l28']['within_question']['mean16']['paired_delta']['estimate'] += .1
    write_json(p, summary)
    p = tmp_path/'report/_SUCCESS.json'; receipt = json.loads(p.read_text())
    receipt['files']['summary.json'] = sha(tmp_path/'report/summary.json'); write_json(p, receipt)
    with pytest.raises(AssertionError):
        audit(tmp_path)

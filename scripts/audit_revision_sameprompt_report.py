"""Independent report arithmetic from saved predictions and original answers.

No reporting/statistics implementation is imported. AUC uses ranks/sklearn;
monitor thresholds enumerate admissible values; bootstrap resamples questions.
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from revision_common import sha, write_json


def equal(got, expected, path='root'):
    if isinstance(expected, dict):
        assert set(got) == set(expected), path
        for key in expected:
            equal(got[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        assert len(got) == len(expected), path
        for i, (a, b) in enumerate(zip(got, expected)):
            equal(a, b, path+f'[{i}]')
    elif isinstance(expected, (float, np.floating)):
        assert got is not None and np.isclose(got, expected, rtol=0, atol=1e-10), (path, got, expected)
    else:
        assert got == expected, (path, got, expected)


def rank_auc(y, score):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y)-n1
    if not n1 or not n0:
        return None
    ranks = rankdata(score, method='average')
    return float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def mean_ci(x, draws):
    x = np.asarray(x, float); n = len(x)
    if not n:
        return dict(n_questions=0, estimate=None, low=None, high=None)
    rng = np.random.default_rng(42)
    values = np.array([np.mean(x[rng.integers(n, size=n)]) for _ in range(draws)])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return dict(n_questions=n, estimate=float(np.mean(x)), low=float(lo), high=float(hi))


def ci(estimate, values, questions, draws):
    values = [v for v in values if v is not None and np.isfinite(v)]
    limits = np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]
    return dict(estimate=estimate, low=limits[0], high=limits[1], n_questions=questions,
                valid_bootstraps=len(values), requested_bootstraps=draws)


def pooled(frame, method, questions, draws):
    y = frame.failure.to_numpy(int); p = frame[method].to_numpy(float)
    names = sorted(questions); mapping = {q: i for i, q in enumerate(names)}
    group = np.array([mapping[q] for q in frame.question_group], int)
    pclip = np.clip(p, np.finfo(float).eps, 1-np.finfo(float).eps)
    logloss = -(y*np.log(pclip)+(1-y)*np.log1p(-pclip)); brier = (y-p)**2
    def calculate(weights):
        if not weights.sum():
            return [None, None, None]
        both = weights[y == 0].sum() > 0 and weights[y == 1].sum() > 0
        return [float(roc_auc_score(y, p, sample_weight=weights)) if both else None,
                float(np.average(logloss, weights=weights)), float(np.average(brier, weights=weights))]
    actual = calculate(np.ones(len(y)))
    rng = np.random.default_rng(42); boot = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        boot.append(calculate(counts[group]))
    return {key: ci(actual[j], [r[j] for r in boot], len(names), draws)
            for j, key in enumerate(['auroc', 'log_loss', 'brier'])}


def classify(values):
    if not len(values):
        return 'none'
    if sum(values) == 0:
        return 'all_correct'
    return 'all_wrong' if sum(values) == len(values) else 'mixed'


def check_view(view, frame, universe, records, expected, draws):
    prefix = int(view.split('_')[0][1:]); covers = []; qs = []; all_q = universe[universe.role == 'test']
    for name, original in universe.groupby('question_group', sort=True):
        part = frame[frame.question_group == name]
        seq = {tuple(records[t]['generated_ids'][:prefix]) for t in part.trajectory_id}
        covers.append(dict(question_group=name, sample_id=original.sample_id.iloc[0], role=original.role.iloc[0],
            all_trajectories=len(original), valid_trajectories=len(part), invalid_trajectories=len(original)-len(part),
            full_outcome=classify(original.failure.tolist()), valid_outcome=classify(part.failure.tolist()),
            unique_valid_prefixes=len(seq), truncated_trajectories=int(original.finish_reason.eq('length').sum())))
    cov = pd.DataFrame(covers); test_cov = cov[cov.role == 'test']; test = frame[frame.role == 'test']
    methods = ['baseline', 'mean16', 'duplicate_current']+(['allmean'] if 'allmean' in frame else [])
    result = dict(test_questions=len(test_cov), valid_test_trajectories=len(test),
                  full_outcomes=test_cov.full_outcome.value_counts().to_dict(),
                  valid_outcomes=test_cov.valid_outcome.value_counts().to_dict(), within_question={}, cross_question={})
    values = {method: [] for method in methods}
    for name, part in test.groupby('question_group', sort=True):
        baseline = rank_auc(part.failure.to_numpy(int), part.baseline.to_numpy())
        if baseline is None:
            continue
        for method in methods:
            value = rank_auc(part.failure.to_numpy(int), part[method].to_numpy())
            values[method].append(value)
            qs.append(dict(view=view, method=method, question_group=name, valid_trajectories=len(part),
                           auc_first=value, auc_second=baseline, delta=value-baseline))
    for method in methods:
        result['within_question'][method] = dict(macro_first=mean_ci(values[method], draws),
            macro_second=mean_ci(values['baseline'], draws),
            paired_delta=mean_ci(np.array(values[method])-values['baseline'], draws))
        result['cross_question'][method] = pooled(test, method, sorted(set(all_q.question_group)), draws)
    result['mean16_minus_duplicate'] = mean_ci(np.array(values['mean16'])-values['duplicate_current'], draws)
    equal(expected, result, view)
    return cov.assign(view=view), qs


def monitoring_stats(table, draws):
    labels = ['far', 'failure_detection', 'any_alarm', 'potential_token_fraction',
              'wrong_answer_potential_token_fraction', 'correct_answer_potential_token_fraction']
    qnames = sorted(set(table.question_group)); mapping = {q: i for i, q in enumerate(qnames)}
    gi = np.array([mapping[q] for q in table.question_group]); n = len(qnames)
    y = table.failure.to_numpy(int); a = table.alarmed.to_numpy(bool)
    saved = table.potential_saved_tokens.to_numpy(int); length = table.observed_length.to_numpy(int)
    numerator = np.column_stack([a & (y == 0), a & (y == 1), a, saved, saved*(y == 1), saved*(y == 0)])
    denominator = np.column_stack([y == 0, y == 1, np.ones(len(y)), length, length*(y == 1), length*(y == 0)])
    def ratio(w):
        aa, bb = w@numerator, w@denominator
        return [float(x/z) if z else None for x, z in zip(aa, bb)]
    actual = ratio(np.ones(len(y))); rng = np.random.default_rng(42); boot = []
    for _ in range(draws):
        count = np.bincount(rng.integers(n, size=n), minlength=n); boot.append(ratio(count[gi]))
    output = {name: ci(actual[j], [r[j] for r in boot], n, draws) for j, name in enumerate(labels)}
    output['first_boundary_counts'] = {str(k): int(v) for k, v in table.first_boundary.value_counts().sort_index().items()}
    output['potential_saved_tokens'] = int(saved.sum())
    return output


def check_monitor(universe, views, reported, saved, draws, target):
    data = universe[universe.role.isin(['calibration', 'test'])].copy().reset_index(drop=True)
    cal = data.role.eq('calibration').to_numpy(); correct = data.failure.eq(0).to_numpy()
    for method in ['baseline', 'mean16', 'duplicate_current']:
        scores = np.full((len(data), 2), -np.inf)
        for j, prefix in enumerate([16, 64]):
            lookup = dict(zip(views[f'p{prefix}_l28'].trajectory_id, views[f'p{prefix}_l28'][method]))
            scores[:, j] = [lookup.get(t, -np.inf) for t in data.trajectory_id]
        calibration = scores.max(1)[cal & correct]
        if len(calibration):
            candidates = sorted(set([-np.inf, *calibration.tolist()]))
            allowed = int(np.floor(target*len(calibration)))
            threshold = next(t for t in candidates if sum(calibration > t) <= allowed)
            threshold_json = {'kind': 'negative_infinity', 'value': None} if np.isneginf(threshold) else {'kind': 'finite', 'value': threshold}
        else:
            threshold = None; threshold_json = {'kind': 'unavailable', 'value': None}
        meta = dict(threshold=threshold_json, target_far=target, calibration_questions=int(data.loc[cal, 'question_group'].nunique()),
            calibration_correct_responses=int((cal & correct).sum()), calibration_ids=data.loc[cal, 'trajectory_id'].tolist(),
            calibration_correct_ids=data.loc[cal & correct, 'trajectory_id'].tolist())
        table = saved[saved.method == method].reset_index(drop=True)
        pd.testing.assert_frame_equal(table[data.columns], data)
        for j, prefix in enumerate([16, 64]):
            np.testing.assert_array_equal(table[f'risk{prefix}'].fillna(-np.inf), scores[:, j])
        if threshold is None:
            assert table[['alarmed', 'first_boundary', 'potential_saved_tokens']].isna().all().all()
            meta.update(calibration_far=None, test=None, reason='No correct calibration response')
        else:
            flags = []; first = []; counts = []
            for i, length in enumerate(data.observed_length):
                assert all(length > p for p, s in zip([16, 64], scores[i]) if np.isfinite(s))
                triggered = [p for p, s in zip([16, 64], scores[i]) if s > threshold]
                flags.append(bool(triggered)); first.append(triggered[0] if triggered else -1)
                counts.append(length-triggered[0] if triggered else 0)
            np.testing.assert_array_equal(table.alarmed, flags)
            np.testing.assert_array_equal(table.first_boundary, first)
            np.testing.assert_array_equal(table.potential_saved_tokens, counts)
            meta['calibration_far'] = float(np.mean(np.array(flags)[cal & correct]))
            meta['test'] = monitoring_stats(table.loc[~cal], draws)
        equal(reported[method], meta, 'monitor.'+method)


def run(root):
    dest = root/'report'; receipt = json.loads((dest/'_SUCCESS.json').read_text())
    assert receipt['complete'] and receipt['plan_sha256'] == sha(dest/'plan.json')
    for name, digest in receipt['files'].items():
        assert sha(dest/name) == digest, name
    plan = json.loads((dest/'plan.json').read_text()); cfg = plan['config']
    for path, digest in plan['files'].items():
        assert sha(path) == digest, path
    readout = json.loads((root/'readouts/_SUCCESS.json').read_text())
    for name, digest in readout['files'].items():
        assert sha(root/'readouts'/name) == digest
    assert json.loads((root/'readouts/audit.json').read_text())['readout_receipt_sha256'] == sha(root/'readouts/_SUCCESS.json')
    raw = json.loads((root/'full/_SUCCESS.json').read_text()); records = {}
    for name, digest in raw['records'].items():
        path = root/'full/samples'/name; assert sha(path) == digest
        row = json.loads(path.read_text()); records[row['trajectory']['trajectory_id']] = row
    universe = pd.read_parquet(root/'readouts/all_trajectories.parquet')
    summary = json.loads((dest/'summary.json').read_text()); draws = cfg['bootstrap_questions']
    assert summary['primary_view'] == 'p16_l28' and summary['bootstrap_draws'] == draws and summary['bootstrap_seed'] == 42
    for role, frame in universe.groupby('role'):
        equal(summary['outcomes_by_role'][role], dict(questions=frame.question_group.nunique(), responses=len(frame),
              correct=int(frame.failure.eq(0).sum()), truncated=int(frame.finish_reason.eq('length').sum()),
              truncation_rate=float(frame.finish_reason.eq('length').mean())))
    views = {v: pd.read_parquet(root/'readouts'/v/'predictions.parquet') for v in ['p16_l28', 'p16_l14', 'p64_l28']}
    expected_preds = pd.concat([f.assign(view=n) for n, f in views.items()], ignore_index=True)
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'predictions.parquet'), expected_preds)
    covers = []; questions = []
    for view, frame in views.items():
        cov, rows = check_view(view, frame, universe, records, summary['views'][view], draws)
        covers.append(cov); questions += rows
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'question_coverage.parquet'), pd.concat(covers, ignore_index=True))
    qcols = ['view', 'method', 'question_group', 'valid_trajectories', 'auc_first', 'auc_second', 'delta']
    expected_q = pd.DataFrame(questions, columns=qcols)
    got_q = pd.read_parquet(dest/'within_question.parquet')
    order = ['view', 'method', 'question_group']
    pd.testing.assert_frame_equal(got_q.sort_values(order).reset_index(drop=True),
        expected_q.sort_values(order).reset_index(drop=True), check_dtype=False)
    common = json.loads((dest/'common_prefix_cohort.json').read_text())
    ids = set(views['p16_l28'].trajectory_id) & set(views['p64_l28'].trajectory_id)
    assert common['trajectory_ids'] == sorted(ids)
    for view in ['p16_l28', 'p64_l28']:
        frame = views[view][views[view].trajectory_id.isin(ids)]
        check_view(view, frame, universe, records, common['views'][view], draws)
    check_monitor(universe, views, summary['monitoring'], pd.read_parquet(dest/'alarms.parquet'), draws, cfg['far_target'])
    selections = json.loads((dest/'selection.json').read_text())
    assert len(selections) == 10
    for entry in selections:
        original = json.loads((root/'readouts'/entry['view']/entry['method']/'selection.json').read_text())
        equal({k: v for k, v in entry.items() if k not in ['view', 'method']}, original)
    inputs = json.loads((root/'inputs.json').read_text())
    equal(json.loads((dest/'source_manifest.json').read_text()), {
        'root': str(root), 'weights_and_transforms': str(root/'readouts'),
        'input_sha256': sha(root/'inputs.json'), 'raw_receipt_sha256': sha(root/'full/_SUCCESS.json'),
        'readout_receipt_sha256': sha(root/'readouts/_SUCCESS.json'),
        'readout_audit_sha256': sha(root/'readouts/audit.json'),
        'question_order': [{'file': f'questions/q{i:03}.html', 'sample_id': r['sample_id']} for i, r in enumerate(inputs)]})
    assert len(inputs) == len(list((dest/'questions').glob('*.html'))) == 224
    for i, item in enumerate(inputs):
        content = (dest/'questions'/f'q{i:03}.html').read_text()
        assert html.escape(item['prompt_text']) in content and html.escape(item['ground_truth']) in content
        answers = [r for r in records.values() if r['sample_id'] == item['sample_id']]
        assert len(answers) == 4
        for record in answers:
            assert html.escape(record['response_text']) in content
    result = dict(complete=True, report_receipt_sha256=sha(dest/'_SUCCESS.json'),
        audit_source_sha256=sha(Path(__file__)), questions=len(inputs), trajectories=len(universe),
        methods=10, bootstrap_draws=draws, independent_rank_auc_pooled_weighted_auc_and_thresholds=True,
        question_coverage_original_text_predictions_first_alarms_and_potential_tokens_verified=True,
        visual_review_pending=True)
    write_json(dest/'statistics_audit.json', result); print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)

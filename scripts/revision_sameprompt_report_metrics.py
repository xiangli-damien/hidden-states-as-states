"""Fixed question bootstrap and two-boundary monitoring; no model selection."""
import numpy as np
import pandas as pd
from revision_sameprompt_statistics import auc, within_question, far_threshold, alarms

SEED = 42
VIEWS = ('p16_l28', 'p16_l14', 'p64_l28')
METHODS = ('baseline', 'mean16', 'duplicate_current')


def interval(estimate, replicates, questions, draws):
    finite = np.asarray([x for x in replicates if x is not None and np.isfinite(x)], float)
    limits = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
    return dict(estimate=estimate, low=limits[0], high=limits[1], n_questions=questions,
                valid_bootstraps=len(finite), requested_bootstraps=draws)


def cross_question(frame, method, universe_questions, draws=2000):
    """Resample ALL test questions, including those with no available boundary.

    Weighted pair counts implement a pooled, trajectory-weighted AUROC; this is
    auxiliary, distinct from the equally weighted within-question primary AUC.
    """
    names = sorted(set(universe_questions)); lookup = {q: i for i, q in enumerate(names)}
    y = frame.failure.to_numpy(int); p = frame[method].to_numpy(float)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    g = np.array([lookup[q] for q in frame.question_group], int); n = len(names)
    counts = np.bincount(g, minlength=n)
    pos = np.bincount(g[y == 1], minlength=n); neg = np.bincount(g[y == 0], minlength=n)
    wins = np.zeros((n, n))
    for i in np.flatnonzero(y == 1):
        js = np.flatnonzero(y == 0)
        np.add.at(wins[g[i]], g[js], (p[i] > p[js])+.5*(p[i] == p[js]))
    clipped = np.clip(p, np.finfo(float).eps, 1-np.finfo(float).eps)
    losses = np.bincount(g, weights=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)), minlength=n)
    brier = np.bincount(g, weights=(y-p)**2, minlength=n)
    ix = np.random.default_rng(SEED).integers(n, size=(draws, n))
    weights = np.array([np.bincount(row, minlength=n) for row in ix])
    denominator = weights@counts
    auc_den = (weights@pos)*(weights@neg)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc_boot = np.einsum('bi,ij,bj->b', weights, wins, weights)/auc_den
        log_boot = (weights@losses)/denominator
        brier_boot = (weights@brier)/denominator
    return {'auroc': interval(auc(y, p), auc_boot, n, draws),
            'log_loss': interval(float(losses.sum()/len(y)) if len(y) else None, log_boot, n, draws),
            'brier': interval(float(brier.sum()/len(y)) if len(y) else None, brier_boot, n, draws)}


def outcome_kind(y):
    values = set(y)
    return 'none' if not values else 'all_correct' if values == {0} else 'all_wrong' if values == {1} else 'mixed'


def coverage(universe, predictions, records, prefix):
    rows = []
    for q, group in universe.groupby('question_group', sort=True):
        valid = predictions[predictions.question_group == q]
        sequences = [tuple(records[t]['generated_ids'][:prefix]) for t in valid.trajectory_id]
        rows.append({'question_group': q, 'sample_id': group.sample_id.iloc[0], 'role': group.role.iloc[0],
                     'all_trajectories': len(group), 'valid_trajectories': len(valid),
                     'invalid_trajectories': len(group)-len(valid),
                     'full_outcome': outcome_kind(group.failure), 'valid_outcome': outcome_kind(valid.failure),
                     'unique_valid_prefixes': len(set(sequences)),
                     'truncated_trajectories': int(group.finish_reason.eq('length').sum())})
    return pd.DataFrame(rows)


def view_metrics(frame, coverage_frame, draws=2000):
    test = frame[frame.role == 'test']; cover = coverage_frame[coverage_frame.role == 'test']
    methods = list(METHODS)+(['allmean'] if 'allmean' in frame else [])
    within = {}; cross = {}; question_rows = []
    for method in methods:
        result = within_question(test.question_group, test.failure, test[method], test.baseline, draws, SEED)
        within[method] = {k: v for k, v in result.items() if k != 'question_rows'}
        question_rows.extend([{'method': method, **r} for r in result['question_rows']])
        cross[method] = cross_question(test, method, cover.question_group, draws)
    duplicate = within_question(test.question_group, test.failure, test.mean16, test.duplicate_current, draws, SEED)
    return {'test_questions': len(cover), 'valid_test_trajectories': len(test),
            'full_outcomes': cover.full_outcome.value_counts().to_dict(),
            'valid_outcomes': cover.valid_outcome.value_counts().to_dict(),
            'within_question': within, 'mean16_minus_duplicate': duplicate['paired_delta'],
            'cross_question': cross}, question_rows


def encoded_threshold(value):
    if value is None:
        return {'kind': 'unavailable', 'value': None}
    if np.isneginf(value):
        return {'kind': 'negative_infinity', 'value': None}
    if not np.isfinite(value):
        raise ValueError('Unexpected threshold')
    return {'kind': 'finite', 'value': float(value)}


def monitor_metrics(table, draws=2000):
    names = sorted(set(table.question_group)); n = len(names)
    groups = [table[table.question_group == q] for q in names]
    # Six numerator/denominator pairs; responses are pooled, uncertainty by question.
    labels = ['far', 'failure_detection', 'any_alarm', 'potential_token_fraction',
              'wrong_answer_potential_token_fraction', 'correct_answer_potential_token_fraction']
    nums, dens = [], []
    for group in groups:
        y = group.failure.to_numpy(int); a = group.alarmed.to_numpy(bool)
        saved = group.potential_saved_tokens.to_numpy(int); lengths = group.observed_length.to_numpy(int)
        nums.append([(a & (y == 0)).sum(), (a & (y == 1)).sum(), a.sum(), saved.sum(),
                     saved[y == 1].sum(), saved[y == 0].sum()])
        dens.append([(y == 0).sum(), (y == 1).sum(), len(y), lengths.sum(),
                     lengths[y == 1].sum(), lengths[y == 0].sum()])
    nums, dens = np.asarray(nums), np.asarray(dens)
    ix = np.random.default_rng(SEED).integers(n, size=(draws, n))
    result = {}
    for j, name in enumerate(labels):
        a, b = nums[:, j], dens[:, j]; denominator = b[ix].sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = a[ix].sum(1)/denominator
        estimate = float(a.sum()/b.sum()) if b.sum() else None
        result[name] = interval(estimate, boot, n, draws)
    result['first_boundary_counts'] = {str(k): int(v) for k, v in table.first_boundary.value_counts().sort_index().items()}
    result['potential_saved_tokens'] = int(table.potential_saved_tokens.sum())
    return result


def monitoring(universe, views, draws=2000, target=.1):
    """Layer28 readouts at16/64; auxiliary allmean lacks a16 readout and is excluded."""
    data = universe[universe.role.isin(['calibration', 'test'])].copy().reset_index(drop=True)
    result = {}; tables = []
    for method in METHODS:
        scores = np.full((len(data), 2), -np.inf)
        for j, prefix in enumerate([16, 64]):
            values = views[f'p{prefix}_l28'].set_index('trajectory_id')[method]
            mapped = data.trajectory_id.map(values)
            scores[:, j] = mapped.fillna(-np.inf)
        maxima = scores.max(1); calibration = data.role.eq('calibration').to_numpy()
        correct = data.failure.eq(0).to_numpy()
        threshold = far_threshold(maxima[calibration & correct], target)
        meta = {'threshold': encoded_threshold(threshold), 'target_far': target,
                'calibration_questions': int(data.loc[calibration, 'question_group'].nunique()),
                'calibration_correct_responses': int((calibration & correct).sum()),
                'calibration_ids': data.loc[calibration, 'trajectory_id'].tolist(),
                'calibration_correct_ids': data.loc[calibration & correct, 'trajectory_id'].tolist()}
        table = data.copy(); table['method'] = method
        table['risk16'] = np.where(np.isfinite(scores[:, 0]), scores[:, 0], np.nan)
        table['risk64'] = np.where(np.isfinite(scores[:, 1]), scores[:, 1], np.nan)
        if threshold is None:
            table['alarmed'] = pd.Series([None]*len(data), dtype='boolean')
            table['first_boundary'] = pd.Series([None]*len(data), dtype='Int64')
            table['potential_saved_tokens'] = pd.Series([None]*len(data), dtype='Int64')
            meta.update(calibration_far=None, test=None, reason='No correct calibration response')
        else:
            out = alarms(scores, [16, 64], data.observed_length.to_numpy(), threshold)
            for key, value in out.items():
                table[key] = value
            meta['calibration_far'] = float(out['alarmed'][calibration & correct].mean())
            meta['test'] = monitor_metrics(table.loc[~calibration], draws)
        result[method] = meta; tables.append(table)
    return result, pd.concat(tables, ignore_index=True)

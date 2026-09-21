"""Fit supervised Bayesian detectors on frozen change-cluster assignments.

All supervised fitting uses train, calibration/regularization selection uses
validation, and test is evaluated only after selection/scores are frozen.
"""
import argparse
import html
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from hss.analysis.channel_data import load_config
from hss.analysis.change_bayes import (MonotonePlatt, bootstrap_auc, far_threshold,
                                      fit_bayes, log_odds, pair_counts)
from hss.experiments.artifacts import file_digest, runtime_versions, save_json

TITLES = {
    'delta28_nb': '跨层变化 · 28 层 NB',
    'delta28_markov': '跨层变化 · 28 层 Markov',
    'delta5_nb': '跨层变化 · 相同 5 层 NB',
    'state5_nb': '原始状态 · 相同 5 层 NB',
    'prompt_delta5_nb': 'Prompt 变化 · 5 层 NB',
    'prompt_state5_nb': 'Prompt 状态 · 5 层 NB',
    'token_pre_mnb': 'Token 变化计数 · pre RMSNorm',
    'token_post_mnb': 'Token 变化计数 · post RMSNorm',
    'length': '回答长度', 'entropy': '平均 token 熵',
    'length_entropy_spline': '长度 + 熵 · 样条逻辑回归',
    'length_entropy_hgb': '长度 + 熵 · 梯度提升树',
}


def load_inputs(source, cfg):
    rows = pd.read_parquet(source / 'rows.parquet')
    splits = pd.read_parquet(source / 'splits.parquet')
    assert rows.sample_id.equals(splits.sample_id) and rows.sample_id.is_unique
    assert rows.question_group.is_unique
    assert set(rows.label.unique()) == {0, 1}
    assert set(splits.split.unique()) == {'train', 'validation', 'test'}
    n = len(rows)
    inputs, files = {}, [source / 'rows.parquet', source / 'splits.parquet']
    def sequence(prefix, layers):
        columns, ks = [], []
        for layer in layers:
            folder = source / 'fits' / f'{prefix}_L{layer:02d}'
            files.extend([folder / 'assignments.npz', folder / 'selection.json'])
            with np.load(folder / 'assignments.npz') as a:
                np.testing.assert_array_equal(a['question_index'], np.arange(n))
                columns.append(a['assignment'].copy())
            selected = json.loads((folder / 'selection.json').read_text())
            ks.append(selected['k'])
        return np.column_stack(columns), ks
    d, k = sequence('mean_delta', range(1, cfg['last_layer'] + 1))
    inputs['delta28_nb'] = ('categorical', d, k)
    inputs['delta28_markov'] = ('markov', d, k)
    for name, prefix in [('delta5_nb', 'mean_delta'), ('state5_nb', 'mean_state'),
                         ('prompt_delta5_nb', 'prompt_delta'), ('prompt_state5_nb', 'prompt_state')]:
        x, ks = sequence(prefix, cfg['probe_layers'])
        inputs[name] = ('categorical', x, ks)
    for norm in ['pre', 'post']:
        folder = source / 'fits' / f'token_delta_{norm}'
        files.extend([folder / 'assignments.npz', folder / 'selection.json'])
        k = json.loads((folder / 'selection.json').read_text())['k']
        with np.load(folder / 'assignments.npz') as a:
            x = pair_counts(a['assignment'], a['question_index'], n, k)
        np.testing.assert_array_equal(x.sum(1), 8)
        inputs[f'token_{norm}_mnb'] = ('multinomial', x, [k])
    return rows, splits, inputs, {str(f.relative_to(source)): file_digest(f) for f in sorted(set(files))}


def spline_model(c):
    transform = ColumnTransformer([
        ('nuisance_splines', SplineTransformer(n_knots=4, degree=3,
                                             include_bias=False), [0, 1])
    ], remainder='passthrough')
    return make_pipeline(transform, StandardScaler(),
                         LogisticRegression(C=c, max_iter=2000, solver='lbfgs'))


def fit_adjustment(x_fit, x_predict, y_fit, y_val, val, cfg, folder):
    candidates = []
    for c in cfg['logistic_c']:
        m = spline_model(c).fit(x_fit, y_fit)
        if int(m[-1].n_iter_.max()) >= m[-1].max_iter:
            raise RuntimeError('Logistic regression did not converge')
        score = m.decision_function(x_predict)
        auc = float(roc_auc_score(y_val, score[val]))
        path = folder / f'spline_c{c:g}.joblib'
        joblib.dump(m, path)
        candidates.append((auc, -c, m, score, str(path.name)))
    best = max(candidates, key=lambda item: item[:2])
    hgb = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
        min_samples_leaf=30, l2_regularization=10., early_stopping=False,
        random_state=cfg['seed']).fit(x_fit, y_fit)
    hscore = hgb.decision_function(x_predict)
    joblib.dump(hgb, folder / 'hgb.joblib')
    joblib.dump(best[2], folder / 'selected_spline.joblib')
    info = dict(selected_c=-best[1], validation_auc=best[0],
                candidates=[dict(c=-a[1], validation_auc=a[0], path=a[4]) for a in candidates],
                hgb_validation_auc=float(roc_auc_score(y_val, hscore[val])))
    return best[3], hscore, info


def run(cfg, source, out):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if (out / '_SUCCESS.json').exists():
        raise RuntimeError('Completed result exists; choose a new output directory or --render-only')
    rows, splits, inputs, hashes = load_inputs(source, cfg)
    tr = splits.split.to_numpy() == 'train'
    va = splits.split.to_numpy() == 'validation'
    te = splits.split.to_numpy() == 'test'
    y = 1 - rows.label.to_numpy(int)
    length = np.log1p(rows.n_tokens.to_numpy(float))
    entropy = rows.entropy.to_numpy(float)
    nuisance = np.column_stack([length, entropy])
    assert np.isfinite(nuisance).all()
    protocol = dict(config=cfg, source=str(source), source_hashes=hashes,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        runtime=runtime_versions(), split=splits.split.value_counts().to_dict(),
        target='failure=1, correctness=0; never flip score using test labels',
        selection='alpha=1 fixed; logistic C selected by validation AUROC, smaller C breaks ties',
        calibration='Monotone Platt uses validation only; cannot reverse raw score direction',
        thresholds='Strict score > validation-success quantile; target empirical validation FAR 10%',
        combination='Train-only five-fold out-of-fold Bayesian log odds plus length/entropy; full-train Bayes scores on validation/test. Frozen GMM maps use full train, no labels.',
        nuisance='Spline logistic: log1p length and entropy, 4 knots/degree3, then downstream feature scaling. HGB sensitivity:100 iterations,7 leaves,L2=10. Raw GMM inputs unchanged.',
        limitations=['Previously explored MATH; not fresh confirmation.',
            'Full-response features are retrospective, not early warnings.',
            'Static versus update comparison is matched at layers 1,7,14,21,28 only.',
            'Token count vectors use 8 sampled adjacent pairs per question, not ordered transitions.',
            'Pointwise paired bootstrap CIs condition on trained models; no retraining or multiplicity correction.',
            'Uses existing 60/20/20 split, not exact paper 40/60 reproduction.',
            'Nuisance adjustment is predictive comparison, not causal identification.'])
    save_json(out / 'protocol.json', protocol)
    print('Protocol saved; fitting on train and selecting/calibrating on validation.', flush=True)
    models = out / 'models'; models.mkdir(exist_ok=True)
    scores, adjustments, selection, calibration, thresholds = {}, {}, {}, {}, {}
    feature_archive = {}
    y_train = y[tr]
    fold = StratifiedKFold(n_splits=cfg['crossfit_folds'], shuffle=True, random_state=cfg['seed'])
    folds = list(fold.split(np.zeros(len(y_train)), y_train))
    fold_id = np.full(len(y), -1, int)
    train_indices = np.flatnonzero(tr)
    for i, (_, held) in enumerate(folds): fold_id[train_indices[held]] = i
    pd.DataFrame(dict(sample_id=rows.sample_id, split=splits.split, crossfit_fold=fold_id)).to_parquet(out / 'split.parquet', index=False)
    (models / 'nuisance').mkdir(exist_ok=True)
    bs, bh, info = fit_adjustment(nuisance[tr], nuisance, y_train, y[va], va, cfg, models / 'nuisance')
    scores.update(length=length, entropy=entropy,
                  length_entropy_spline=bs, length_entropy_hgb=bh)
    selection['nuisance'] = info
    for name, (kind, x, ks) in inputs.items():
        folder = models / name; folder.mkdir(exist_ok=True)
        model = fit_bayes(kind, x[tr], y_train, ks, cfg['alpha'])
        score = log_odds(model, x)
        joblib.dump(model, folder / 'bayes.joblib')
        np.testing.assert_allclose(log_odds(joblib.load(folder / 'bayes.joblib'), x), score)
        oof = np.full(tr.sum(), np.nan)
        for i, (fit, hold) in enumerate(folds):
            fitted = fit_bayes(kind, x[tr][fit], y_train[fit], ks, cfg['alpha'])
            oof[hold] = log_odds(fitted, x[tr][hold])
            joblib.dump(fitted, folder / f'crossfit_{i}.joblib')
        assert np.isfinite(oof).all()
        fit_x = np.column_stack([nuisance[tr], oof])
        pred_x = np.column_stack([nuisance, score])
        cs, ch, info = fit_adjustment(fit_x, pred_x, y_train, y[va], va, cfg, folder)
        scores[name] = score
        scores[name + '__spline'] = cs
        scores[name + '__hgb'] = ch
        adjustments[name] = info
        selection[name] = dict(kind=kind, cardinalities=ks, alpha=cfg['alpha'],
                               validation_auc=float(roc_auc_score(y[va], score[va])))
        feature_archive[name] = x
        np.save(folder / 'training_oof_log_odds.npy', oof)
        print(f'Fitted {name}: validation AUROC {selection[name]["validation_auc"]:.4f}', flush=True)
    probabilities = {}
    for name, score in scores.items():
        assert np.isfinite(score).all()
        cal = MonotonePlatt().fit(score[va], y[va])
        probabilities[name] = cal.predict_proba(score)
        calibration[name] = vars(cal)
        thresholds[name] = far_threshold(score[va & (y == 0)], cfg['far'])
    np.savez_compressed(out / 'input_cluster_features.npz', **feature_archive)
    score_frame = pd.DataFrame(dict(sample_id=rows.sample_id, split=splits.split, **scores))
    score_frame.to_parquet(out / 'scores.parquet', index=False)
    pd.DataFrame(dict(sample_id=rows.sample_id, **probabilities)).to_parquet(out / 'calibrated_probabilities.parquet', index=False)
    save_json(out / 'selection.json', dict(bayes=selection, adjustments=adjustments, calibration=calibration, thresholds=thresholds))
    save_json(out / '_SCORES_FROZEN.json', dict(scores_sha256=file_digest(out / 'scores.parquet'),
        selection_sha256=file_digest(out / 'selection.json'), time=time.time()))
    # No test correctness used above this boundary for fit/selection/calibration.
    print('Models, scores, thresholds and selection frozen. Evaluating test questions.', flush=True)
    rng = np.random.default_rng(cfg['seed'] + 1)
    positive, negative = np.flatnonzero(y[te] == 1), np.flatnonzero(y[te] == 0)
    boot = np.column_stack([rng.choice(positive, (cfg['bootstrap'], len(positive)), replace=True),
                            rng.choice(negative, (cfg['bootstrap'], len(negative)), replace=True)])
    np.save(out / 'test_bootstrap_indices.npy', boot)
    metrics, bootstrap = [], {}
    for name, score in scores.items():
        test = score[te]; yy = y[te]; threshold = thresholds[name]
        auc = float(roc_auc_score(yy, test)); b = bootstrap_auc(yy, test, boot)
        bootstrap[name] = b
        p = probabilities[name][te]; alert = test > threshold
        row = dict(name=name, auroc=auc, ci_low=float(np.quantile(b, .025)), ci_high=float(np.quantile(b, .975)),
            auprc=float(average_precision_score(yy, test)),
            validation_auc=float(roc_auc_score(y[va], score[va])),
            validation_far=float(np.mean(score[va & (y == 0)] > threshold)),
            test_far=float(alert[yy == 0].mean()), failure_recall=float(alert[yy == 1].mean()),
            threshold=threshold, calibrated_brier=float(brier_score_loss(yy, p)),
            calibrated_logloss=float(log_loss(yy, p, labels=[0, 1])))
        if name in inputs:
            raw = expit(test)
            row.update(raw_brier=float(brier_score_loss(yy, raw)), raw_logloss=float(log_loss(yy, raw, labels=[0, 1])))
        metrics.append(row)
    table = pd.DataFrame(metrics).set_index('name')
    contrasts = []
    def contrast(label, a, b):
        delta = bootstrap[a] - bootstrap[b]
        contrasts.append(dict(comparison=label, a=a, b=b,
            delta=float(table.loc[a, 'auroc'] - table.loc[b, 'auroc']),
            ci_low=float(np.quantile(delta, .025)), ci_high=float(np.quantile(delta, .975))))
    contrast('Markov minus NB, same 28 update layers', 'delta28_markov', 'delta28_nb')
    contrast('Update minus state, same five layers', 'delta5_nb', 'state5_nb')
    contrast('Prompt update minus prompt state, same five layers', 'prompt_delta5_nb', 'prompt_state5_nb')
    for name in inputs:
        for adjustment in ['spline', 'hgb']:
            contrast(f'{name}: add to {adjustment} length+entropy', name + '__' + adjustment, 'length_entropy_' + adjustment)
    table.reset_index().to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'contrasts.csv', index=False)
    np.savez_compressed(out / 'bootstrap_auroc.npz', **bootstrap)
    test_rows = rows.loc[te].copy()
    for name, score in scores.items(): test_rows[name] = score[te]
    test_rows.to_parquet(out / 'test_predictions.parquet', index=False)
    summary = dict(train_n=int(tr.sum()), validation_n=int(va.sum()), test_n=int(te.sum()),
                   test_failure=int(y[te].sum()), test_success=int((y[te] == 0).sum()),
                   models=list(inputs), seconds=time.time() - started,
                   bootstrap='2000 stratified question bootstrap draws; paired across all scores',
                   probability_warning='Raw NB may be overconfident; calibration uses validation only.',
                   comparisons='Main length/entropy baseline: spline logistic; nonlinear HGB sensitivity also reported.',
                   strongest_by_validation=max(inputs, key=lambda n: selection[n]['validation_auc']))
    save_json(out / 'summary.json', summary)
    render(out, source)
    audit(out, source)
    print(table.loc[list(inputs) + ['length_entropy_spline', 'length_entropy_hgb'],
                    ['auroc', 'ci_low', 'ci_high', 'test_far', 'failure_recall']].to_string(), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


def audit(out, source):
    frozen = json.loads((out / '_SCORES_FROZEN.json').read_text())
    assert frozen['scores_sha256'] == file_digest(out / 'scores.parquet')
    assert frozen['selection_sha256'] == file_digest(out / 'selection.json')
    p = json.loads((out / 'protocol.json').read_text())
    for f, digest in p['source_hashes'].items():
        assert file_digest(source / f) == digest, f
    metrics = pd.read_csv(out / 'metrics.csv')
    assert (metrics.validation_far <= p['config']['far'] + 1e-12).all()
    assert np.isfinite(metrics[['auroc', 'ci_low', 'ci_high', 'calibrated_brier']]).all().all()
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name not in ['inventory.json', '_SUCCESS.json'])
    inventory = [dict(path=str(f.relative_to(out)), size=f.stat().st_size, sha256=file_digest(f)) for f in files]
    save_json(out / 'inventory.json', inventory)
    save_json(out / '_SUCCESS.json', dict(files=len(inventory), bytes=sum(f['size'] for f in inventory),
        inventory_sha256=file_digest(out / 'inventory.json'), scores_frozen=True,
        source_hashes_verified=len(p['source_hashes']), validation_far_verified=True))


def supplemental_comparisons(out):
    """Post-hoc interpretation checks; reuse frozen scores, never refit/select."""
    table = pd.read_csv(out / 'metrics.csv').set_index('name')
    bootstrap = np.load(out / 'bootstrap_auroc.npz')
    pairs = [('28-layer NB minus five-layer NB', 'delta28_nb', 'delta5_nb')]
    for adjustment in ['spline', 'hgb']:
        pairs.extend([
            (f'Update minus state after {adjustment} adjustment, same five layers',
             'delta5_nb__' + adjustment, 'state5_nb__' + adjustment),
            (f'Markov minus NB after {adjustment} adjustment, same 28 layers',
             'delta28_markov__' + adjustment, 'delta28_nb__' + adjustment),
        ])
    records = []
    for title, a, b in pairs:
        delta = bootstrap[a] - bootstrap[b]
        records.append(dict(comparison=title, a=a, b=b,
            delta=float(table.loc[a, 'auroc'] - table.loc[b, 'auroc']),
            ci_low=float(np.quantile(delta, .025)), ci_high=float(np.quantile(delta, .975))))
    pd.DataFrame(records).to_csv(out / 'supplemental_contrasts.csv', index=False)
    return records


def render(out, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.read_csv(out / 'metrics.csv').set_index('name')
    contrasts = pd.read_csv(out / 'contrasts.csv')
    summary = json.loads((out / 'summary.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    selection = json.loads((out / 'selection.json').read_text())
    supplemental = supplemental_comparisons(out)
    folder = out / 'report'; folder.mkdir(exist_ok=True)
    names = summary['models']
    english = ['Update NB (28 layers)', 'Update Markov (28 layers)', 'Update NB (5 layers)',
               'State NB (5 layers)', 'Prompt update NB (5 layers)', 'Prompt state NB (5 layers)',
               'Token update counts (pre)', 'Token update counts (post)']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1.1, 1]})
    a = table.loc[names]
    axes[0].errorbar(a.auroc, np.arange(len(a)), xerr=np.vstack([a.auroc-a.ci_low, a.ci_high-a.auroc]), fmt='o', capsize=3)
    axes[0].set_yticks(np.arange(len(a)), english); axes[0].invert_yaxis()
    axes[0].axvline(.5, color='grey', ls=':')
    axes[0].axvline(table.loc['length_entropy_spline', 'auroc'], color='#b74c24', ls='--', label='Length + entropy (spline)')
    axes[0].axvline(table.loc['length_entropy_hgb', 'auroc'], color='#538641', ls='--', label='Length + entropy (HGB)')
    axes[0].set_xlabel('Failure AUROC (test n=986)'); axes[0].legend(fontsize=8, loc='upper left')
    for offset, adjustment, color in [(-.13, 'spline', '#2766a1'), (.13, 'hgb', '#538641')]:
        d = contrasts.set_index('a').loc[[n+'__'+adjustment for n in names]]
        axes[1].errorbar(d.delta, np.arange(len(d))+offset,
            xerr=np.vstack([d.delta-d.ci_low, d.ci_high-d.delta]), fmt='o', capsize=3, color=color, label=adjustment)
    axes[1].set_yticks(np.arange(len(a)), []); axes[1].invert_yaxis(); axes[1].axvline(0, color='grey', ls=':')
    axes[1].set_xlabel('AUROC gain when added to length + entropy'); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Bayesian readouts of change clusters | Qwen2 MATH\nTrain-only GMM; held-out prediction; pointwise 95% paired bootstrap CI')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(folder / f'bayes.{ext}', dpi=180)
    plt.close(fig)
    scores = pd.read_parquet(out / 'scores.parquet')
    rows = pd.read_parquet(source / 'rows.parquet'); test = scores.split.to_numpy() == 'test'; y = 1 - rows.label.to_numpy(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ['delta28_nb', 'delta28_markov', 'token_pre_mnb', 'token_post_mnb', 'length_entropy_spline', 'delta28_nb__spline']:
        fpr, tpr, _ = roc_curve(y[test], scores.loc[test, name])
        ax.plot(fpr, tpr, label=f'{name}: {table.loc[name,"auroc"]:.3f}')
    ax.plot([0, 1], [0, 1], color='grey', ls=':'); ax.set(xlabel='False alarm rate among correct answers', ylabel='Failure recall', title='Retrospective failure detection')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(folder / f'roc.{ext}', dpi=180)
    plt.close(fig)
    def fmt(value): return f'{value:.3f}'
    body = []
    for name in names:
        r = table.loc[name]
        ds = contrasts[contrasts.a == name+'__spline'].iloc[0]
        dh = contrasts[contrasts.a == name+'__hgb'].iloc[0]
        body.append('<tr>' + ''.join('<td>'+str(s)+'</td>' for s in [TITLES[name],
            f'{r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]',
            f'{ds.delta:+.3f} [{ds.ci_low:+.3f}, {ds.ci_high:+.3f}]',
            f'{dh.delta:+.3f} [{dh.ci_low:+.3f}, {dh.ci_high:+.3f}]',
            f'{r.test_far:.1%}', f'{r.failure_recall:.1%}']) + '</tr>')
    baseline_rows = ''.join(f'<li>{TITLES[n]}：AUROC <b>{table.loc[n,"auroc"]:.3f}</b> [{table.loc[n,"ci_low"]:.3f}, {table.loc[n,"ci_high"]:.3f}]</li>' for n in ['length', 'entropy', 'length_entropy_spline', 'length_entropy_hgb'])
    comparison_rows = ''.join('<li>'+html.escape(r.comparison)+f'：Δ AUROC {r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]</li>' for r in contrasts.head(3).itertuples())
    supplemental_rows = ''.join('<li>'+html.escape(r['comparison'])+f'：Δ AUROC {r["delta"]:+.3f} [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]</li>' for r in supplemental)
    probability_rows = ''.join(f'<tr><td>{TITLES[n]}</td><td>{table.loc[n,"raw_brier"]:.3f}</td><td>{table.loc[n,"calibrated_brier"]:.3f}</td><td>{table.loc[n,"raw_logloss"]:.3f}</td><td>{table.loc[n,"calibrated_logloss"]:.3f}</td></tr>' for n in names)
    probs = pd.read_parquet(out / 'calibrated_probabilities.parquet')
    examples = []
    for i in np.flatnonzero(test):
        r = rows.iloc[i]
        examples.append(dict(sample_id=r.sample_id, correct=int(r.label), question=str(r.prompt_text),
            response=str(r.response_text), answer=str(r.ground_truth), tokens=int(r.n_tokens), entropy=float(r.entropy),
            scores={n:float(scores.iloc[i][n]) for n in names},
            probabilities={n:float(probs.iloc[i][n]) for n in names}))
    save_json(folder / 'samples.json', examples)
    save_json(folder / 'data.json', dict(summary=summary, metrics=table.reset_index().replace({np.nan:None}).to_dict('records'),
                                       contrasts=contrasts.to_dict('records'), supplemental=supplemental, selection=selection))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>变化簇的贝叶斯检测 · Qwen MATH</title><style>
body{max-width:1300px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui;color:#213047;background:#f5f7fa}h1{line-height:1.3}section{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce2eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f8}img{max-width:100%}.note{border-left:4px solid #d99936;padding:16px;background:#fff6e5}.scroll{overflow:auto}select,input,button{font:inherit;padding:8px;margin:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto;font:14px/1.7 system-ui}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}a{color:#2164af}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
<h1>变化簇能否用贝叶斯判断正误？</h1><p>Qwen2-7B-Instruct × MATH 5,000 · 原始 GMM 不额外归一化 · <a href="../../report/index.html">返回变化簇报告</a></p>
<div class="note">训练 3,011／验证 1,003／测试 986（正确 458、错误 528）。使用标签训练贝叶斯；测试题只用状态／变化簇预测。当前 MATH 已反复探索，结果是探索性留出评估，不是全新确认实验。生成特征用于回答完成后的检测；Prompt 两项才是生成前预测。</div>
<section><h2>测试表现</h2><p>正类＝错误。AUROC 越高越好；区间为 2,000 次题目级分层配对 bootstrap 的逐项 95% CI。增益是将贝叶斯分数加入同一长度＋熵基线后的变化。</p><div class="scroll"><table><thead><tr><th>表示／模型</th><th>AUROC [95% CI]</th><th>加入样条基线 Δ</th><th>加入树基线 Δ</th><th>测试 FAR</th><th>错误召回</th></tr></thead><tbody>__ROWS__</tbody></table></div><p>阈值仅用验证集正确回答校准到 FAR ≤10%；测试 FAR 是实际观测值，不保证等于 10%。Prompt 加入长度／熵后的结果使用了生成后的信息，因此只是诊断对照，不能称为生成前预测。</p><h3>相同测试集的基线</h3><ul>__BASELINES__</ul><h3>配对比较</h3><ul>__COMPARISONS__</ul></section>
<section><h2>效果与增益</h2><img src="bayes.png" alt="贝叶斯分类表现和长度熵之外的增益"><p><a href="bayes.pdf">下载 PDF</a></p><img src="roc.png" alt="测试 ROC 曲线"><p><a href="roc.pdf">下载 ROC PDF</a></p></section>
<section><h2>补充比较：收益来自转移关系吗？</h2><p>以下比较在看到初步结果后补充，只复用冻结分数，没有重新拟合或选择模型；属于事后探索。即使变化簇能预测正误，也要检查它是否胜过相同层数的原始状态，以及 Markov 是否确实胜过 NB。</p><ul>__SUPPLEMENTAL__</ul><p><a href="../supplemental_contrasts.csv">补充配对比较 CSV</a></p></section>
<section><h2>概率是否可信</h2><p>不同层存在依赖，朴素贝叶斯容易过度自信。单调 Platt 校准只用验证集，不反转分数方向。下列 Brier 和 log loss 均在测试集计算，越小越好。</p><div class="scroll"><table><thead><tr><th>模型</th><th>原始 Brier</th><th>校准 Brier</th><th>原始 log loss</th><th>校准 log loss</th></tr></thead><tbody>__PROBABILITY__</tbody></table></div></section>
<section><h2>逐题检查：测试集</h2><select id="method"></select><select id="order"><option value="high">预测错误分数从高到低</option><option value="low">从低到高</option></select><input id="query" placeholder="题目或 ID"><button id="prev">上一题</button><button id="next">下一题</button><p id="meta"></p><div class="grid"><div><h3>题目</h3><pre id="question"></pre><h3>参考答案</h3><pre id="answer"></pre></div><div><h3>模型回答</h3><pre id="response"></pre></div></div></section>
<section><h2>方法与边界</h2><ul><li>GMM 沿用已冻结、仅训练题拟合的结果。NB 使用各层独立簇词表、经验类别先验和 α=1 的 Laplace 平滑，无需 Hungarian matching。</li><li>Markov 使用类别条件的首层概率和逐层转移概率，是论文式 NB 的扩展。28 层变化取完整回答均值，末层用 pre-RMSNorm。</li><li>原始状态与变化的公平对照仅使用共同的 1、7、14、21、28 层。没有全层原始状态分类器，也没有本轮 MFA 分类器。</li><li>Token 多项式 NB 输入每题 8 个均匀随机采样相邻 token 对的簇计数；采样对不构成连续轨迹。</li><li>组合分类器在训练题的五折 out-of-fold 贝叶斯分数上拟合，验证集只选逻辑回归正则 C。基线与组合使用相同样条配置，并报告固定梯度提升树敏感性对照。下游分类器的缩放不改变 GMM 特征。</li><li>单个对比的区间不校正多重比较，也没有重采样重训 GMM／分类器。超越长度和熵仍不代表机制或因果；token 类型、题型等仍可能解释信号。</li><li>沿用 60%／20%／20% 划分，不是论文 40%／60% 的严格复现。测试结果不用于模型、阈值或符号选择。</li></ul><p><a href="../metrics.csv">所有指标</a> · <a href="../contrasts.csv">配对差异</a> · <a href="../protocol.json">协议</a> · <a href="../selection.json">模型选择／校准</a> · <a href="../test_predictions.parquet">逐题分数</a> · <a href="../inventory.json">结果清单</a></p></section>
<script>const titles=__TITLES__;let samples,shown=[],position=0;const el=x=>document.getElementById(x);function show(){if(!shown.length){el('meta').textContent='没有匹配题目';for(const k of ['question','answer','response'])el(k).textContent='';return}const s=shown[position],m=el('method').value;el('meta').textContent=`${position+1}/${shown.length} · ${s.sample_id} · 实际${s.correct?'正确':'错误'} · ${s.tokens} tokens · 熵 ${s.entropy.toFixed(3)} · 贝叶斯 log odds ${s.scores[m].toFixed(3)} · 校准错误概率 ${(100*s.probabilities[m]).toFixed(1)}%`;for(const k of ['question','answer','response'])el(k).textContent=s[k]}function filter(){const query=el('query').value.toLowerCase(),m=el('method').value,sign=el('order').value==='high'?-1:1;shown=samples.filter(s=>(s.sample_id+' '+s.question).toLowerCase().includes(query)).sort((a,b)=>sign*(a.scores[m]-b.scores[m]));position=0;show()}fetch('samples.json').then(r=>r.json()).then(s=>{samples=s;for(const m of Object.keys(s[0].scores)){const o=document.createElement('option');o.value=m;o.textContent=titles[m];el('method').append(o)}filter()}).catch(e=>el('meta').textContent='加载失败：'+e);el('method').onchange=filter;el('order').onchange=filter;el('query').oninput=filter;el('prev').onclick=()=>{if(shown.length){position=(position-1+shown.length)%shown.length;show()}};el('next').onclick=()=>{if(shown.length){position=(position+1)%shown.length;show()}};</script></html>'''
    for key, value in {'__ROWS__': ''.join(body), '__BASELINES__': baseline_rows,
                        '__COMPARISONS__': comparison_rows, '__PROBABILITY__': probability_rows,
                        '__SUPPLEMENTAL__': supplemental_rows,
                        '__TITLES__': json.dumps(TITLES, ensure_ascii=False)}.items(): page = page.replace(key, value)
    (folder / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/change-bayes.toml')
    parser.add_argument('--source'); parser.add_argument('--output'); parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(); cfg = load_config(args.config)
    source = Path(args.source or cfg['source']); out = Path(args.output or cfg['output'])
    with threadpool_limits(cfg['cpu_threads']):
        if args.render_only: render(out, source); audit(out, source)
        else: run(cfg, source, out)

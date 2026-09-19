"""Describe center distances, correctness and nuisance associations of frozen fits."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

from hss.analysis.cluster_distance import cluster_percentiles, factor_distances
from hss.analysis.cluster_profiles import wilson_interval
from hss.experiments.artifacts import file_digest, save_json
from hss.experiments.fitting import load_fitted


def correlation(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else None


def conditional_models(meta, scores, labels, norm):
    """Cross-fit labels only; clusters themselves were fitted on all samples.

    These are conditional/transductive checks, NOT a held-out cluster pipeline.
    Splines model nonlinear associations. C=1 fixed, no tuning/feature selection.
    """
    frame = pd.DataFrame(dict(category=meta.category.astype(str), level=meta.level.astype(str),
                              cluster=labels.astype(str), length=np.log1p(meta.n_tokens),
                              capped=(meta.finish_reason == 'length').astype(str),
                              norm=np.log1p(norm)))
    for key, values in scores.items():
        frame[key] = np.log1p(values)
    configs = dict(composition=['length'], composition_norm=['length', 'norm'],
                   plus_euclidean=['length', 'norm', 'euclidean'])
    if 'mahalanobis' in scores:
        configs['plus_mahalanobis'] = ['length', 'norm', 'mahalanobis']
    if 'parallel' in scores:
        configs['plus_factor_parts'] = ['length', 'norm', 'parallel', 'perpendicular']
    y = meta.label.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(frame, y))
    fold_ids = np.zeros(len(y), dtype=int)
    predictions = {}
    for key, features in configs.items():
        pred = np.zeros(len(y))
        for fold, (train, test) in enumerate(folds):
            transform = ColumnTransformer([
                ('category', OneHotEncoder(handle_unknown='ignore'), ['category', 'level', 'cluster', 'capped']),
                ('numeric', SplineTransformer(n_knots=5, degree=3, include_bias=False), features),
            ])
            model = make_pipeline(transform, LogisticRegression(C=1, solver='lbfgs', max_iter=2000))
            model.fit(frame.iloc[train], y[train])
            pred[test] = model.predict_proba(frame.iloc[test])[:, 1]
            fold_ids[test] = fold
        predictions[key] = pred
    result = {}
    rng = np.random.default_rng(43)
    strata = [np.flatnonzero(y == c) for c in [0, 1]]
    bootstrap = [np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in strata]) for _ in range(400)]
    baseline = predictions['composition_norm']
    for key, pred in predictions.items():
        delta = [roc_auc_score(y[idx], pred[idx]) - roc_auc_score(y[idx], baseline[idx]) for idx in bootstrap]
        result[key] = dict(auc=float(roc_auc_score(y, pred)),
                          delta_vs_norm=float(roc_auc_score(y, pred) - roc_auc_score(y, baseline)),
                          delta_ci=np.quantile(delta, [.025, .975]).tolist(), features=configs[key])
    return result, predictions, fold_ids


def summarize(values, percentiles, y, meta, baseline, idx):
    bins = []
    for b in range(5):
        part = idx[np.minimum((percentiles[idx] * 5).astype(int), 4) == b]
        if not len(part):
            bins.append(dict(bin=b, n=0)); continue
        successes = int(y[part].sum())
        bins.append(dict(bin=b, n=len(part), accuracy=float(y[part].mean()),
                         accuracy_ci=wilson_interval(successes, len(part)),
                         mean_distance=float(values[part].mean()),
                         median_tokens=float(meta.iloc[part].n_tokens.median()),
                         expected_accuracy=float(baseline[part].mean()),
                         observed_minus_expected=float((y[part]-baseline[part]).mean())))
    subset_y = y[idx]
    auc = float(roc_auc_score(subset_y, -values[idx])) if len(np.unique(subset_y)) == 2 else None
    return dict(n=len(idx), auc_near_is_correct=auc, bins=bins,
                length_spearman=correlation(values[idx], meta.iloc[idx].n_tokens),
                level_spearman=correlation(values[idx], meta.iloc[idx].level),
                norm_spearman=None)


def build(root, states, rows, fits_root=None):
    root = Path(root)
    old = json.loads((root / 'exploration.json').read_text())
    X = np.load(states, mmap_mode='r')
    meta = pd.read_parquet(rows)
    if not np.array_equal(meta.sample_id.astype(str), [s['id'] for s in old['samples']]):
        raise ValueError('Row IDs differ from frozen report')
    if file_digest(Path(states)) != old['exploration']['states_sha256']:
        raise ValueError('Raw matrix differs from explored data')
    import hss.analysis.cluster_distance as metrics_module
    result = dict(created_at=time.time(), samples=old['samples'], methods={},
                  source_sha256=file_digest(root/'exploration.json'),
                  states_sha256=file_digest(Path(states)), rows_sha256=file_digest(Path(rows)),
                  driver_sha256=file_digest(Path(__file__)), metrics_sha256=file_digest(Path(metrics_module.__file__)),
                  analysis_note='Frozen full-batch clusters; descriptive distances. Label models use 5-fold OOF predictions, '
                  'but cluster geometry saw every sample. Paired bootstrap holds geometry and OOF predictions fixed; '
                  'intervals do not cover clustering/training/model-selection uncertainty. No causal or prospective claims.',
                  regression=dict(folds=5, seed=42, C=1, spline_knots=5, spline_degree=3,
                                  bootstrap=400, bootstrap_seed=43,
                                  controls=['category', 'level', 'cluster', 'length', 'length_cap', 'hidden_mean_norm']))
    norm = np.linalg.norm(X, axis=1)
    y = meta.label.to_numpy()
    table = meta.copy()
    table['hidden_mean_norm'] = norm
    for name, fit in old['methods'].items():
        path = Path(fits_root)/name if fits_root else Path(fit['fit_path'])
        model, info = load_fitted(path)
        if not info['model'].get('converged'):
            raise ValueError('Unconverged fit')
        labels = np.asarray(old['assignments'][name])
        if not np.array_equal(model.predict(X), labels):
            raise ValueError('Model does not reproduce frozen assignments: '+name)
        means = np.asarray(model.means_ if hasattr(model, 'means_') else model.centers_, dtype=float)
        keys = ['euclidean']
        is_mfa = hasattr(model, 'loadings_')
        is_gmm = hasattr(model, 'covariances_')
        if is_mfa: keys += ['mahalanobis', 'parallel', 'perpendicular', 'latent_map_norm', 'residual_psi_norm']
        elif is_gmm: keys += ['mahalanobis']
        values = {key: np.empty(len(X)) for key in keys}
        center_shifts = {}
        for k in np.unique(labels):
            idx = np.flatnonzero(labels == k)
            part = np.asarray(X[idx], dtype=float)
            center_shifts[str(k)] = float(np.linalg.norm(means[k] - part.mean(0)))
            if is_mfa:
                scores = factor_distances(part, means[k], model.loadings_[k], model.noise_[k])
            else:
                delta = part - means[k]
                scores = dict(euclidean=np.linalg.norm(delta, axis=1))
                if is_gmm:
                    if model.covariance_type != 'diag': raise ValueError('Expected diagonal GMM')
                    scores['mahalanobis'] = np.sqrt((delta**2/model.covariances_[k]).sum(1))
            for key, value in scores.items(): values[key][idx] = value
        percentiles = {key: cluster_percentiles(value, labels) for key, value in values.items()}
        diagnostic_keys = [key for key in keys if key not in ['latent_map_norm', 'residual_psi_norm']]
        checks, predictions, fold_ids = conditional_models(meta, {k:values[k] for k in diagnostic_keys}, labels, norm)
        baseline = predictions['composition_norm']
        entry = dict(k=fit['k'], rank=fit['rank'], fit_path=fit['fit_path'],
                     fit_sha256=file_digest(path/'fit.json'), model_sha256=file_digest(path/'model.npz'),
                     assignment=labels.tolist(), center_definition='saved fitted component mean',
                     fitted_vs_empirical_center_shift=center_shifts,
                     values={key:value.tolist() for key,value in values.items()},
                     percentiles={key:value.tolist() for key,value in percentiles.items()},
                     checks=checks, predictions={key:p.tolist() for key,p in predictions.items()},
                     fold_ids=fold_ids.tolist(), summary={})
        for key in diagnostic_keys:
            summary = {}
            for cluster in ['all'] + [str(k) for k in np.unique(labels)]:
                idx = np.arange(len(X)) if cluster == 'all' else np.flatnonzero(labels == int(cluster))
                summary[cluster] = summarize(values[key], percentiles[key], y, meta, baseline, idx)
                summary[cluster]['norm_spearman'] = correlation(values[key][idx], norm[idx])
                if cluster == 'all':
                    summary[cluster]['auc_near_is_correct'] = float(roc_auc_score(y, -percentiles[key]))
            entry['summary'][key] = summary
        result['methods'][name] = entry
        table[name+'_cluster'] = labels
        for key,value in values.items():
            table[name+'_'+key] = value
            table[name+'_'+key+'_percentile'] = percentiles[key]
        print(json.dumps(dict(method=name, checks=checks)), flush=True)
    table.to_parquet(root/'distance_members.parquet', index=False)
    save_json(root/'distances.json', result)
    render(root)


def render(root):
    root = Path(root)
    data = json.loads((root/'distances.json').read_text())
    template = Path(__file__).parents[1]/'src/hss/reporting/templates/cluster_distances.html'
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (root/'distance.html').write_text(template.read_text().replace('__DATA__', payload))
    save_json(root/'distance_render_manifest.json', dict(
        data_sha256=file_digest(root/'distances.json'), template_sha256=file_digest(template),
        renderer_sha256=file_digest(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--states'); parser.add_argument('--rows'); parser.add_argument('--fits-root')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only: render(args.report)
    else:
        if not args.states or not args.rows: parser.error('--states and --rows required')
        with threadpool_limits(2): build(args.report, args.states, args.rows, args.fits_root)

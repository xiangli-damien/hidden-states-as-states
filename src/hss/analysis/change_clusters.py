"""Label-free change features and GMM fitting, separated from outcome evaluation.

Depth means are means of per-token block updates, not temporal differences.
The terminal decoder output must be pre-RMSNorm. Temporal pairs never cross
response boundaries. All mixture fitting and selection uses training groups.
"""
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from hss.experiments.artifacts import file_digest, save_json


def residual_states(states, pre_final):
    """Replace HF's terminal post-norm slot before subtracting decoder layers."""
    states = np.asarray(states, dtype=np.float32).copy()
    pre_final = np.asarray(pre_final, dtype=np.float32)
    if states.ndim != 3 or pre_final.shape != (len(states), states.shape[-1]):
        raise ValueError('Inconsistent residual-state shapes')
    states[:, -1] = pre_final
    if not np.isfinite(states).all():
        raise ValueError('Nonfinite residual states')
    return states


def pair_positions(n_tokens, sample_id, count, seed):
    """Unique, uniformly sampled adjacent pairs, seeded by question identity."""
    if n_tokens < 2:
        return np.empty(0, dtype=np.int64)
    key = hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
    return np.sort(rng.choice(n_tokens - 1, min(count, n_tokens - 1), replace=False))


def temporal_mean(first, last, n_tokens):
    """Signed average over ALL adjacent temporal differences telescopes."""
    if np.any(np.asarray(n_tokens) < 2):
        raise ValueError('A temporal difference requires at least two tokens')
    return (np.asarray(last) - np.asarray(first)) / (np.asarray(n_tokens) - 1)[..., None]


def gaussian_surrogate(train, n, seed, block=128):
    """One (possibly singular) Gaussian with exact empirical training covariance.

    G @ centered_train / sqrt(n_train) has population covariance X.T@X/n_train.
    No PCA, whitening, independent-coordinate null, or correctness is involved.
    """
    train = np.asarray(train, dtype=np.float64)
    center = train.mean(0)
    centered = train - center
    out = np.empty((n, train.shape[1]), np.float32)
    rng = np.random.default_rng(seed)
    for a in range(0, n, block):
        g = rng.standard_normal((min(block, n-a), len(train))) / np.sqrt(len(train))
        out[a:a+len(g)] = g @ centered + center
    return out


def aggregate_pairs(values, question_index, n_questions):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    total = np.zeros((n_questions, values.shape[1]), dtype=float)
    np.add.at(total, question_index, values)
    count = np.bincount(question_index, minlength=n_questions)
    total /= np.maximum(count[:, None], 1)
    total[count == 0] = np.nan
    return total


def posterior_statistics(gm, x):
    prob = gm.predict_proba(x)
    ll = gm.score_samples(x)
    entropy = -xlogy(prob, prob).sum(1)
    return prob, ll, entropy


def fit_one(x, k, seed, cfg, out, train_index, tag='grid'):
    """Persist every fitted model, including capped/nonconverged candidates."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    marker = out/'complete.json'
    request = dict(k=int(k), seed=int(seed), tag=tag,
                   training_index_sha256=hashlib.sha256(np.asarray(train_index, dtype=np.int64).tobytes()).hexdigest(),
                   tol=cfg['tol'], reg_covar=cfg['reg_covar'], max_iter=cfg['max_iter'],
                   retry_max_iter=cfg['retry_max_iter'])
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['request'] != request or file_digest(out/'model.joblib') != old['model_sha256']:
            raise ValueError('Stale/corrupt mixture candidate')
        return joblib.load(out/'model.joblib'), old
    start = time.monotonic()
    gm = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=cfg['reg_covar'],
        n_init=1, init_params='k-means++', random_state=seed, tol=cfg['tol'],
        max_iter=cfg['max_iter'], warm_start=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        gm.fit(x)
        first_iterations = gm.n_iter_
        if not gm.converged_:
            gm.max_iter = cfg['retry_max_iter']; gm.fit(x)
            iterations = first_iterations + gm.n_iter_
        else:
            iterations = first_iterations
    prob, ll, ent = posterior_statistics(gm, x)
    params = 2*k*x.shape[1] + k-1
    bic = -2*ll.sum() + params*np.log(len(x))
    icl = bic + 2*ent.sum()
    record = dict(request=request, k=int(k), seed=int(seed), converged=bool(gm.converged_),
        iterations=int(iterations), train_ll=float(ll.mean()), bic=float(bic), icl=float(icl),
        min_train_occupancy=int(np.bincount(prob.argmax(1), minlength=k).min()),
        seconds=time.monotonic()-start, warnings=[str(w.message) for w in caught])
    joblib.dump(gm, out/'model.joblib')
    np.savez_compressed(out/'parameters.npz', means=gm.means_, variances=gm.covariances_, weights=gm.weights_)
    record['model_sha256'] = file_digest(out/'model.joblib')
    save_json(marker, record)
    return gm, record


def fit_view(task):
    """Fit one representation independently; no outcome metadata is read."""
    name, cfg = task
    root = Path(cfg['output']); cache = Path(cfg['cache'])
    folder = root/'fits'/name; folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'selection.json'
    source = cache/'views'/f'{name}.npy'
    source_sha = file_digest(source)
    search_signature = dict(grid=cfg['k_grid'], extension=cfg.get('k_extension', []))
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['feature_sha256'] != source_sha:
            raise ValueError('Changed source features')
        if old.get('search_signature') == search_signature:
            return old
        # Preserve the unlabeled pilot before extending a boundary-selected grid.
        import shutil
        for filename in ['selection.json','assignments.npz','selected_model.joblib','display.npz','display_pca.joblib']:
            p=folder/filename
            if p.exists() and not (folder/('pilot_'+filename)).exists():shutil.copy2(p,folder/('pilot_'+filename))
    started = time.monotonic()
    meta = pd.read_parquet(root/'splits.parquet', columns=['sample_id','split'])
    temporal = name.startswith('token_delta_')
    owners = (pd.read_parquet(cache/'pairs.parquet', columns=['question_index']).question_index.to_numpy()
              if temporal else np.arange(len(meta)))
    split = meta.split.to_numpy()[owners]
    train_index = np.flatnonzero(split == 'train')
    va = split == 'validation'; te = split == 'test'
    x = np.asarray(np.load(source, mmap_mode='r'), dtype=np.float64)
    if len(x) != len(owners) or not np.isfinite(x).all():
        raise ValueError('Invalid feature rows')
    rows = []; models = {}
    with threadpool_limits(cfg['cpu_threads']):
        search_grid=list(cfg['k_grid'])
        for k in search_grid:
            for seed in cfg['seeds']:
                path = folder/f'k{k}_s{seed}'
                gm, rec = fit_one(x[train_index], k, seed, cfg, path, train_index)
                rec = dict(rec, path=str(path.relative_to(root)), validation_ll=float(gm.score(x[va])))
                rows.append(rec); models[k, seed] = gm
            # Adaptive extension is decided by training ICL, never by labels.
            if k==max(cfg['k_grid']) and cfg.get('k_extension'):
                current=[r for r in rows if r['converged']]
                if current and min(current,key=lambda r:r['icl'])['k']==k:
                    search_grid.extend(cfg['k_extension'])
        viable = [r for r in rows if r['converged']]
        if not viable or not any(r['k']==1 for r in viable):
            raise RuntimeError('No converged candidates or K=1 baseline')
        best = min(viable, key=lambda r: (r['icl'], r['k'], r['seed']))
        best_val = max(viable, key=lambda r: r['validation_ll'])
        gm = models[best['k'], best['seed']]
        null_row = min((r for r in viable if r['k']==1), key=lambda r:r['icl'])
        gm1 = models[1, null_row['seed']]
        prob, ll, entropy = posterior_statistics(gm, x)
        ll1 = gm1.score_samples(x); assigned = prob.argmax(1)
        np.savez_compressed(folder/'assignments.npz', assignment=assigned, probability=prob.astype(np.float32),
            log_likelihood=ll, baseline_log_likelihood=ll1, entropy=entropy, question_index=owners)
        joblib.dump(gm, folder/'selected_model.joblib')
        stable = []
        for seed in cfg['seeds']:
            alt = models[best['k'], seed]
            stable.append(dict(kind='initialization', seed=int(seed), converged=bool(alt.converged_),
                ari=float(adjusted_rand_score(assigned[te], alt.predict(x[te])))))
        for seed in cfg['stability_seeds']:
            # Subsample whole QUESTIONS, never individual temporal pairs across splits.
            rng = np.random.default_rng(seed)
            train_questions = np.flatnonzero(meta.split.to_numpy() == 'train')
            chosen = rng.choice(train_questions, int(.8*len(train_questions)), replace=False)
            idx = np.flatnonzero(np.isin(owners, chosen))
            alt, diag = fit_one(x[idx], best['k'], seed, cfg, folder/f'subsample_k{best["k"]}_s{seed}', idx, '80pct_questions')
            stable.append(dict(kind='80pct_questions', seed=int(seed), converged=diag['converged'],
                ari=float(adjusted_rand_score(assigned[te], alt.predict(x[te])))))
        # Visual projection ONLY, fitted on training examples. No fitted feature scaling.
        pca = PCA(n_components=2, svd_solver='randomized', random_state=921).fit(x[train_index])
        np.savez_compressed(folder/'display.npz', xy=pca.transform(x).astype(np.float32),
                            center_xy=pca.transform(gm.means_).astype(np.float32), explained=pca.explained_variance_ratio_)
        joblib.dump(pca, folder/'display_pca.joblib')
        test_idx = np.flatnonzero(te)
        if len(test_idx)>1500: test_idx=np.random.default_rng(921).choice(test_idx,1500,replace=False)
        z = assigned[test_idx]
        silhouette = float(silhouette_score(x[test_idx], z)) if 1<len(np.unique(z))<len(z) else None
        qgain = aggregate_pairs((ll-ll1)/x.shape[1], owners, len(meta))[:,0]
        qtest = qgain[meta.split.to_numpy()=='test']
        rng=np.random.default_rng(921)
        boot=np.array([np.nanmean(qtest[rng.integers(len(qtest),size=len(qtest))]) for _ in range(1000)])
        summary = dict(name=name, k=best['k'], seed=best['seed'], k_validation=best_val['k'],
            k_boundary=best['k']==max(search_grid), feature_sha256=source_sha,
            search_signature=search_signature, searched_k=search_grid,
            selection='minimum training ICL = BIC + 2 posterior entropy; converged fits only',
            normalization=False, dimension=x.shape[1], n_vectors=len(x),
            n_train_vectors=len(train_index), n_test_questions=int((meta.split=='test').sum()),
            selected=best, candidates=rows, stability=stable, silhouette_test=silhouette,
            test_gain_nats_per_dimension=float(np.nanmean(qtest)), test_gain_ci=np.quantile(boot,[.025,.975]).tolist(),
            test_mean_max_posterior=float(prob[te].max(1).mean()), test_occupancy=np.bincount(assigned[te],minlength=best['k']).tolist(),
            visual_explained=pca.explained_variance_ratio_.tolist(), seconds=time.monotonic()-started,
            note=('Temporal BIC/ICL treats sampled pairs as independent; within-question dependence remains. Test gain CI resamples questions.'
                  if temporal else 'Independent question-group fitting. Current dataset was explored previously; not a fresh confirmation set.'))
        save_json(marker, summary)
    print(json.dumps(dict(done=name,k=best['k'],seconds=round(time.monotonic()-started,1))),flush=True)
    return summary

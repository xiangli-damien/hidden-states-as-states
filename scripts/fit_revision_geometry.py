"""Train-only raw GMM codebooks and held-out reconstruction/prediction audits.

Mean-vector codebooks and actual-token codebooks are separate artifacts. The
question is the resampling unit, including for token reconstruction. ICL chooses
K using training likelihood/entropy; test labels never choose partitions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from revision_common import (config, digest, freeze, provenance, sha, write_json,
    write_npz, nearest, reconstruct, nmse_rows, paired_ratio_ci, status)


def read_view(cfg, prefix, layer, view):
    frames, xs = [], []
    li = cfg['layers'].index(layer)
    prefix_root=Path(cfg.get('prefix_root',str(Path(cfg['output'])/'prefixes')))
    for marker in sorted(prefix_root.glob('shard_*/_SUCCESS.json')):
        frame = pd.read_parquet(marker.parent/'rows.parquet')
        with np.load(marker.parent/f'prefix_{prefix}.npz') as saved:
            valid = saved['valid'].copy()
            if view == 'question_tokens':
                valid &= saved['question_valid']
                x = saved['question_window'][:, li].copy()
            elif view == 'tokens':
                x = saved['window'][:, li].copy()
            elif view == 'last':
                x = saved['window'][:, li, -1].copy()
            elif view in ('mean4', 'mean16'):
                n = int(view[4:])
                x = saved['window'][:, li, -n:].mean(1)
            elif view == 'all_mean':
                x = saved['all_mean'][:, li].copy()
            else:
                raise ValueError(view)
        if not np.isfinite(x[valid]).all():
            raise ValueError('Nonfinite valid activations')
        xs.append(x[valid]); frames.append(frame.loc[valid])
    return pd.concat(frames, ignore_index=True), np.concatenate(xs)


def basis(x, rank, center, seed=42):
    # Uncentered SVD of residual relative to explicit decoder center. Ordinary
    # PCA would add an extra hidden mean if used with a GMM mean anchor.
    from sklearn.utils.extmath import randomized_svd
    r = min(rank, len(x)-1, x.shape[-1])
    out = np.zeros((rank, x.shape[-1]), np.float32)
    if r > 0:
        _, _, v = randomized_svd(x-center, n_components=r, n_iter=5, random_state=seed)
        out[:r] = v
    return out


def prediction(frame, x, codes, centers, confidence):
    """Exploratory question labels: state-only NB and independently regularized LR.

    This is not a causal criterion and not an online confidence claim. Historical
    MATH test labels were explored before this study, explicitly noted in report.
    """
    train = frame.split.eq('train').to_numpy()
    val = frame.split.eq('validation').to_numpy()
    test = frame.split.eq('test').to_numpy()
    y = 1-frame.label.to_numpy(int)  # source correctness=1; prediction failure=1
    if len(np.unique(y[train])) != 2 or not val.any() or not test.any():
        raise ValueError('Need both classes and the fixed train/val/test splits')
    counts = np.array([np.bincount(codes[train & (y == c)], minlength=len(centers)) for c in (0, 1)]) + 1.
    likelihood = counts/counts.sum(1, keepdims=True)
    prior = np.bincount(y[train], minlength=2)+1.
    prior /= prior.sum()
    logp = np.log(likelihood[:, codes].T) + np.log(prior)
    probability = 1/(1+np.exp(logp[:, 0]-logp[:, 1]))
    results = {'state_nb': probability}
    candidates = []
    scaler = StandardScaler().fit(x[train])
    standardized = scaler.transform(x)  # only the READOUT, not the clustering
    for c in (.001, .01, .1, 1):
        model = LogisticRegression(C=c, max_iter=2000, tol=1e-5).fit(standardized[train], y[train])
        if model.n_iter_.max() >= 2000:
            raise RuntimeError('Linear readout did not converge')
        pred = model.predict_proba(standardized)[:, 1]
        candidates.append((log_loss(y[val], pred[val]), c, pred, model))
    _, c, results['linear_probe'], selected = min(candidates, key=lambda v: v[0])
    numeric=np.column_stack([np.log1p(frame.n_prompt_tokens.to_numpy(float)),
        np.linalg.norm(x,axis=1),confidence.next_token_entropy.to_numpy(float),
        confidence.next_token_logit_margin.to_numpy(float)])
    control_scaler=StandardScaler().fit(numeric[train])
    categorical=frame[['category','level']].fillna('unknown').astype(str)
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(categorical[train])
    nuisance=np.column_stack([control_scaler.transform(numeric),encoder.transform(categorical)])
    state=np.eye(len(centers),dtype=np.float32)[codes]
    for name,features in [('nuisance',nuisance),('nuisance_plus_state',np.column_stack([nuisance,state]))]:
        trials=[]
        for strength in (.001,.01,.1,1):
            clf=LogisticRegression(C=strength,max_iter=2000,tol=1e-5).fit(features[train],y[train])
            if clf.n_iter_.max()>=2000:
                raise RuntimeError('Nuisance readout did not converge')
            score=clf.predict_proba(features)[:,1]
            trials.append((log_loss(y[val],score[val]),strength,score))
        _,_,results[name]=min(trials,key=lambda v:v[0])
    report, out = {}, {'sample_id': frame.sample_id.to_numpy(), 'split': frame.split.to_numpy(), 'failure': y}
    for name, p in results.items():
        out[name] = p
        report[name] = {'test_auroc': float(roc_auc_score(y[test], p[test])),
                       'test_log_loss': float(log_loss(y[test], p[test])),
                       'test_brier': float(brier_score_loss(y[test], p[test]))}
    report['linear_probe']['selected_C'] = c
    report['nuisance_definition']='train-fit category, difficulty, log prompt length, representation norm, current next-token entropy and margin; no future answer length'
    readout = {'nb_likelihood': likelihood, 'nb_prior': prior,
               'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
               'linear_coef': selected.coef_, 'linear_intercept': selected.intercept_}
    return report, pd.DataFrame(out), readout


def fit_one(job):
    cfg, prefix, layer, view = job
    with threadpool_limits(limits=cfg['blas_threads']):
        return _fit_one(cfg, prefix, layer, view)


def _fit_one(cfg, prefix, layer, view):
    started = time.monotonic()
    name = f'p{prefix}_l{layer}_{view}'
    dest = Path(cfg['output'])/'geometry'/name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'_SUCCESS.json').exists():
        return json.loads((dest/'summary.json').read_text())
    frame, x = read_view(cfg, prefix, layer, view)
    train = frame.split.eq('train').to_numpy()
    val = frame.split.eq('validation').to_numpy()
    test = frame.split.eq('test').to_numpy()
    tokens = x.ndim == 3
    # Equal fitting contribution per question (four fixed positions), whereas
    # reconstruction below evaluates ALL 16 actual held-out token vectors.
    fit_positions=cfg.get('train_token_positions',[0,5,10,15])
    if tokens and (not fit_positions or len(set(fit_positions))!=len(fit_positions)
                   or min(fit_positions)<0 or max(fit_positions)>=x.shape[1]):
        raise ValueError('Invalid training token positions')
    xf = x[train][:, fit_positions].reshape(-1, x.shape[-1]) if tokens else x[train]
    # Raw values are unchanged. Float64 avoids cancellation in variance updates
    # on Qwen's large shared coordinates; this is numerical precision, not scaling.
    xf = xf.astype(np.float64)
    if len(xf) < 100 or not test.any():
        raise ValueError('Insufficient fixed-split data')
    trials, best = [], None
    for k in cfg['k_grid']:
        for seed in cfg['seeds']:
            file = dest/f'gmm_k{k}_s{seed}.npz'
            info = dest/f'gmm_k{k}_s{seed}.json'
            if info.exists():
                row = json.loads(info.read_text())
                if row['converged']:
                    with np.load(file) as saved:
                        arrays = {n: saved[n].copy() for n in saved.files}
                else:
                    arrays = None
            else:
                start = time.monotonic()
                model = GaussianMixture(k, covariance_type='diag', n_init=1, random_state=seed,
                    tol=1e-4, max_iter=300, reg_covar=1e-5)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    model.fit(xf)
                    first_iterations = model.n_iter_
                    if not model.converged_:
                        model.set_params(max_iter=600, warm_start=True).fit(xf)
                posterior = model.predict_proba(xf).astype(np.float64)
                entropy = float(-(posterior*np.log(np.maximum(posterior, 1e-300))).sum())
                bic = float(model.bic(xf))
                row = {'k': k, 'seed': seed, 'bic': bic, 'entropy': entropy, 'icl': bic+2*entropy,
                       'converged': bool(model.converged_), 'iterations_last_fit': model.n_iter_,
                       'iterations_initial_fit': first_iterations,
                       'seconds': time.monotonic()-start, 'validation_log_likelihood': float(model.score(x[val].reshape(-1,x.shape[-1])))}
                arrays = {'centers': model.means_.astype(np.float32), 'variances': model.covariances_.astype(np.float32),
                          'weights': model.weights_.astype(np.float32)}
                write_npz(file, **arrays)
                write_json(info, row)
            trials.append(row)
            if row['converged'] and (best is None or row['icl'] < best[0]['icl']):
                best = (row, arrays)
    if best is None:
        raise RuntimeError('No converged GMM: refusing to select an invalid fit')
    selected, decoder = best
    decoder['train_mean'] = xf.mean(0)
    k = selected['k']
    km = KMeans(k, n_init=10, random_state=42).fit(xf)
    decoder['kmeans_centers'] = km.cluster_centers_.astype(np.float32)
    assignments = nearest(xf, decoder['centers'])
    rank = max(cfg['pca_ranks'])
    decoder['global_basis'] = basis(xf, rank, decoder['train_mean'])
    decoder['local_basis'] = np.stack([basis(xf[assignments == j], rank, decoder['centers'][j]) for j in range(k)])
    # Separate true affine local PCA from the existing fixed-GMM-center residual
    # SVD. Both use the SAME nearest-GMM partition; neither is an MFA decoder.
    decoder['local_empirical_centers']=np.stack([xf[assignments==j].mean(0) if np.any(assignments==j)
                                               else decoder['centers'][j] for j in range(k)]).astype(np.float32)
    decoder['local_empirical_basis']=np.stack([basis(xf[assignments==j],rank,decoder['local_empirical_centers'][j]) for j in range(k)])
    decoder['local_counts'] = np.bincount(assignments, minlength=k)
    write_npz(dest/'decoder.npz', **decoder)
    flat = x.reshape(-1,x.shape[-1])
    codes = nearest(flat, decoder['centers']).reshape(x.shape[:-1])
    probabilities = []
    # Explicit GMM posterior assignments as an audit, not silently substituted
    # for the nearest-center definition in reconstruction and state-only NB.
    for start in range(0, len(flat), 256):
        b = flat[start:start+256].astype(np.float64)
        v = decoder['variances'].astype(np.float64)
        logp = -.5*(np.square(b[:,None]-decoder['centers'][None])/v[None]).sum(-1)
        logp -= .5*np.log(v).sum(-1)[None]
        logp += np.log(decoder['weights'])[None]
        probabilities.append(logp.argmax(1))
    posterior_codes = np.concatenate(probabilities).reshape(codes.shape)
    write_npz(dest/'assignments.npz', sample_id=frame.sample_id.to_numpy(str),
              nearest=codes, posterior=posterior_codes)
    methods = ['mean', 'centroid', 'kmeans_centroid'] + [f'{kind}_pca_{r}' for kind in ('global','local','empirical') for r in cfg['pca_ranks']]
    rows, metrics = [], []
    for method in methods:
        recovered = reconstruct(flat, decoder, method).reshape(x.shape)
        error, den = nmse_rows(x, recovered, decoder['train_mean'])
        if tokens:
            error, den = error.sum(1), den.sum(1)
        result = paired_ratio_ci(error[test], den[test])
        metrics.append({'method': method, 'test_nmse': result,
                        'test_explained_variance': 1-result['estimate'],
                        'continuous_coordinates_per_vector': int(method.rsplit('_',1)[1]) if 'pca' in method else 0,
                        'state_id_bits_per_vector': float(np.log2(k)) if method in ('centroid','kmeans_centroid') or method.startswith(('local','empirical')) else 0})
        rows.append(pd.DataFrame({'sample_id': frame.sample_id, 'split': frame.split, 'method': method,
                                   'squared_error': error, 'train_centered_energy': den}))
    pd.concat(rows).to_parquet(dest/'reconstruction_per_question.parquet',index=False)
    result = {'name': name, 'prefix': prefix, 'layer': layer, 'view': view,
              'train_questions': int(train.sum()), 'val_questions': int(val.sum()), 'test_questions': int(test.sum()),
              'fit_vectors': len(xf), 'normalization': 'none', 'k_selection': 'min training ICL over converged candidates',
              'selected': selected, 'k_grid_boundary': k == max(cfg['k_grid']), 'trials': trials,
              'nearest_posterior_agreement': float((codes == posterior_codes).mean()),
              'geometry': metrics, 'decoder_parameters': {n: int(a.size) for n,a in decoder.items()},
              'test_scope': 'exploratory reused MATH test; question-level pointwise bootstrap',
              'tokens_per_question_fit': len(fit_positions) if tokens else 1,
              'train_token_positions':fit_positions if tokens else [],
              'local_residual_svd_definition':'uncentered train residual SVD about fixed GMM mean',
              'empirical_pca_definition':'PCA about nearest-assigned training sample mean, same GMM partition'}
    if not tokens:
        confidence=pd.read_parquet(Path(cfg['output'])/'confidence'/f'prefix_{prefix}.parquet').set_index('sample_id').loc[frame.sample_id]
        report, predictions, readout = prediction(frame, x, codes, decoder['centers'],confidence)
        result['prediction'] = report
        predictions.to_parquet(dest/'prediction_per_question.parquet',index=False)
        write_npz(dest/'readout.npz', **readout)
    result['seconds'] = time.monotonic()-started
    write_json(dest/'summary.json', result)
    write_json(dest/'_SUCCESS.json', {'summary_sha256': sha(dest/'summary.json'), 'decoder_sha256': sha(dest/'decoder.npz')})
    print(json.dumps({'complete': name, 'k': k, 'seconds': result['seconds']}), flush=True)
    return result


def alignment(cfg, view='last', prefix=0):
    root = Path(cfg['output'])/'geometry'
    result = []
    for a,b in zip(cfg['layers'][:-1], cfg['layers'][1:]):
        da, db = root/f'p{prefix}_l{a}_{view}', root/f'p{prefix}_l{b}_{view}'
        aa, bb = np.load(da/'assignments.npz'), np.load(db/'assignments.npz')
        if not np.array_equal(aa['sample_id'], bb['sample_id']):
            raise ValueError('Cross-layer alignment identity mismatch')
        ca, cb = np.load(da/'decoder.npz')['centers'], np.load(db/'decoder.npz')['centers']
        frame, _ = read_view(cfg, prefix, a, view)
        train, test = frame.split.eq('train').to_numpy(), frame.split.eq('test').to_numpy()
        xa, xb = aa['nearest'], bb['nearest']
        flow = np.zeros((len(ca),len(cb)))
        np.add.at(flow, (xa[train],xb[train]),1)
        cosine = ca@cb.T/np.maximum(np.linalg.norm(ca,axis=1)[:,None]*np.linalg.norm(cb,axis=1)[None],1e-20)
        pa = np.bincount(xa[test],minlength=len(ca))/test.sum()
        pb = np.bincount(xb[test],minlength=len(cb))/test.sum()
        for name, matrix in [('cosine',cosine),('train_membership',flow)]:
            ia, ib = linear_sum_assignment(matrix,maximize=True)
            mapping = dict(zip(ia.tolist(),ib.tolist()))
            matched = np.array([mapping.get(int(x),-1)==int(y) for x,y in zip(xa[test],xb[test])])
            chance = float(np.sum(pa[ia]*pb[ib]))
            stats = paired_ratio_ci(matched,np.ones(len(matched)))
            result.append({'from':a,'to':b,'mapping':name,'mapping_pairs':list(zip(ia.tolist(),ib.tolist())),
                'heldout_flow':stats,'heldout_independent_marginal_chance':chance,
                'excess_over_chance':float(matched.mean()-chance),
                'interpretation':'Partition correspondence only; not causal state identity.'})
    write_json(Path(cfg['output'])/'alignment.json',result)


def run(cfg):
    root = Path(cfg['output'])
    if not (root/'prefixes/_SUCCESS.json').exists():
        raise ValueError('Extraction must complete and audit before fitting')
    if not (root/'confidence/_SUCCESS.json').exists():
        raise ValueError('Current-prefix confidence controls must be computed before fitting')
    freeze(root/'geometry_plan.json',provenance(cfg,[Path(__file__),Path(__file__).with_name('revision_common.py'),root/'prefixes/plan.json']))
    jobs=[]
    # Actual-token decoders first so functional GPU work can start without waiting
    # for all question-level representation comparisons.
    for view in ('tokens','question_tokens','last','mean4','mean16','all_mean'):
        for prefix in cfg['prefixes']:
            if view == 'question_tokens' and prefix:
                continue
            for layer in cfg['layers']:
                jobs.append((cfg,prefix,layer,view))
    completed=[]
    status(root,'geometry',state='running',completed=0,expected=len(jobs))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        fs = [pool.submit(fit_one,j) for j in jobs]
        for f in as_completed(fs):
            result=f.result();completed.append(result)
            status(root,'geometry',state='running',completed=len(completed),expected=len(jobs),last=result['name'])
    alignment(cfg)
    write_json(root/'geometry_summary.json',completed)
    status(root,'geometry',state='complete',completed=len(jobs),expected=len(jobs))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    args=p.parse_args();cfg=config(args.config)
    try:
        run(cfg)
    except BaseException:
        status(cfg['output'],'geometry',state='failed',traceback=traceback.format_exc());raise

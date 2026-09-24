"""Read frozen Qwen MATH models/caches into a small CPU route-analysis bundle.

No fitting, normalization or GPU work. Correctness is copied as evaluation
metadata, never passed to mixture inference. Source Result files are immutable.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from hss.data.cache import CachedStates
from hss.experiments.artifacts import file_digest, save_json
from hss.experiments.fitting import load_fitted
from hss.results.models import load_layer
from hss.cluster.gmm import _log_resp


def diagnostics(model, x):
    x = np.asarray(x, dtype=np.float64)
    if hasattr(model, 'loadings_'):
        log = model._log_joint(x)
    else:
        log = _log_resp(x, model.weights_, model.means_, model.precisions_cholesky_, model.covariance_type)
    p = np.exp(log - logsumexp(log, axis=1, keepdims=True))
    assigned = p.argmax(1)
    mu = model.means_[assigned]
    delta = x - mu
    mahal = np.zeros(len(x))
    for k in range(model.n_clusters()):
        idx = assigned == k
        d = delta[idx]
        if hasattr(model, 'loadings_'):
            w, psi = model.loadings_[k], model.noise_[k]
            chol = np.linalg.cholesky(np.eye(w.shape[1]) + w.T @ (w / psi[:, None]))
            low = np.linalg.solve(chol, ((d / psi) @ w).T)
            mahal[idx] = (d * d / psi).sum(1) - (low * low).sum(0)
        else:
            if model.covariance_type != 'diag':
                raise ValueError('This bundle currently requires diagonal GMM')
            mahal[idx] = (d * d / model.covariances_[k]).sum(1)
    norm = np.linalg.norm(x, axis=1)
    distance = ((x * x).sum(1)[:, None] - 2 * x @ model.means_.T
                + (model.means_ * model.means_).sum(1)[None, :])
    ranked = np.sort(p, axis=1)
    return p, dict(posterior=assigned, nearest=distance.argmin(1),
                   rms=norm / np.sqrt(x.shape[1]), mahal=np.maximum(0, mahal) / x.shape[1],
                   margin=ranked[:, -1] - ranked[:, -2],
                   confidence=ranked[:, -1], euclidean=np.linalg.norm(delta, axis=1),
                   angle=np.arccos(np.clip((x * mu).sum(1) / np.maximum(norm * np.linalg.norm(mu, axis=1), 1e-30), -1, 1)),
                   nll=-logsumexp(log, axis=1) / x.shape[1])


def prepare(args):
    start = time.monotonic()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    study = Path(args.study)
    exports = json.loads((study / 'latest_exports.json').read_text())
    paths = {'gmm': Path(args.gmm), 'mfa': Path(exports['post'])}
    meta = pd.read_parquet(paths['mfa'] / 'rows.parquet')
    rows = pd.concat([pd.read_parquet(p / 'data.parquet') for p in sorted(Path(args.collection).glob('shard_*'))
                      if (p / '_COPY_VERIFIED.json').is_file() and (p / '_SUCCESS').is_file()], ignore_index=True)
    if rows.sample_id.duplicated().any():
        raise ValueError('Duplicate source questions')
    rows = rows.set_index('sample_id').loc[meta.sample_id].reset_index()
    if not np.array_equal(rows.n_response_tokens, meta.n_tokens):
        raise ValueError('Token counts differ')
    for col in ['question', 'prompt_text', 'response_text', 'ground_truth', 'n_prompt_tokens', 'entropy', 'perplexity', 'max_probability']:
        if col in rows:
            meta[col] = rows[col].to_numpy()
    # Preserve duplicate prompt groups across folds without inspecting correctness.
    import hashlib
    canonical = rows.prompt_text.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    meta['question_group'] = canonical.map(lambda x: hashlib.sha256(x.encode()).hexdigest()).to_numpy()
    meta.to_parquet(root / 'rows.parquet', index=False)
    snapshots = {v: CachedStates(Path(args.cache) / json.loads((study / 'snapshots' / (v + '.json')).read_text())['key']) for v in ['post', 'pre_final']}
    for data in snapshots.values():
        assert np.array_equal(data.meta.sample_id, meta.sample_id)
    states, probabilities, geometry, provenance = {}, {}, {}, {}
    for method, path in paths.items():
        m = pd.read_parquet(path / 'rows.parquet')
        assert np.array_equal(m.sample_id, meta.sample_id) and np.array_equal(m.label, meta.label)
        states[method + '_global'] = np.load(path / 'states.npy')[:, 1:]
        local, nearest = [], []
        alignment = json.loads((path / 'alignment.json').read_text())['local_to_global']
        provenance[method] = dict(path=str(path), files={f: file_digest(path / f) for f in ['states.npy', 'rows.parquet', 'selection.json', 'alignment.json']}, models={})
        for layer in range(1, 29):
            model, projection, _ = load_layer(path, layer)
            x = snapshots['post'].array(layer)
            np.testing.assert_allclose(projection.transform(x[:3]), x[:3], rtol=0, atol=0)
            p, d = diagnostics(model, x)
            np.testing.assert_array_equal(np.array(alignment[layer])[d['posterior']], states[method + '_global'][:, layer - 1])
            local.append(d.pop('posterior')); nearest.append(d.pop('nearest'))
            probabilities[f'{method}_{layer}'] = p.astype(np.float32)
            for key, value in d.items(): geometry[f'{method}_{layer}_{key}'] = value.astype(np.float32)
            provenance[method]['models'][str(layer)] = file_digest(path / 'models' / f'layer_{layer}' / 'model.npz')
            print(json.dumps(dict(method=method, layer=layer, seconds=round(time.monotonic() - start, 1))), flush=True)
        states[method] = np.column_stack(local); states[method + '_nearest'] = np.column_stack(nearest)
    stability = pd.read_csv(study / 'stability.csv')
    for seed in sorted(stability.seed.unique()):
        part = stability[(stability.view == 'post') & (stability.seed == seed)].sort_values('layer')
        states[f'mfa_seed{seed}'] = np.column_stack([np.load(Path(r.fit_path) / 'assignments.npz')['posterior'] for r in part.itertuples() if r.layer > 0])
    ranks = pd.read_csv(study / 'rank_k_selection.csv')
    for rank in [4, 8]:
        part = ranks[(ranks.view == 'post') & (ranks['rank'] == rank)].sort_values('layer')
        states[f'mfa_rank{rank}_own_k'] = np.column_stack([np.load(Path(r.strict_fit_path) / 'assignments.npz')['posterior'] for r in part.itertuples() if r.layer > 0])
    # A separate pre-norm last-layer view, never mixed into the main map.
    model, _, _ = load_layer(exports['pre_final'], 28)
    p, d = diagnostics(model, snapshots['pre_final'].array(28))
    states['mfa_pre_final'] = states['mfa'].copy(); states['mfa_pre_final'][:, -1] = d.pop('posterior')
    probabilities['mfa_pre_28'] = p.astype(np.float32)
    for key, value in d.items(): geometry[f'mfa_pre_28_{key}'] = value.astype(np.float32)
    np.savez_compressed(root / 'states.npz', **states)
    np.savez_compressed(root / 'probabilities.npz', **probabilities)
    np.savez_compressed(root / 'geometry.npz', **geometry)
    stability.to_csv(root / 'source_stability.csv', index=False)
    save_json(root / 'manifest.json', dict(samples=len(meta), unique_question_groups=meta.question_group.nunique(), layers=list(range(1,29)),
        methods=list(states), sources=provenance, seconds=time.monotonic()-start, source_driver_sha256=file_digest(Path(__file__)),
        raw=True, fitting=False, gpu_used=False, normalization=False,
        note='Frozen whole-dataset descriptive maps. Auxiliary seeds and rank-specific K are sensitivity controls, not an independent test or fixed-K rank ablation.',
        files={f:file_digest(root/f) for f in ['rows.parquet','states.npz','probabilities.npz','geometry.npz']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', default='/lambda/nfs/dami/hss/qwen-math-mfa-24h-20260920')
    p.add_argument('--gmm', default='/lambda/nfs/dami/hss/qwen-math-ablation-20260918/trials/cb427148921dfddc5130c884')
    p.add_argument('--collection', default='/lambda/nfs/dami/openact/runs/math_full_20260916/qwen2')
    p.add_argument('--cache', default='/home/ubuntu/hss-cache/data')
    p.add_argument('--output', required=True)
    a=p.parse_args()
    with threadpool_limits(2): prepare(a)

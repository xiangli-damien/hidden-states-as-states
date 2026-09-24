"""Bounded frozen-mean-map reconstruction experiment; no generation or GMM fit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from revision_common import config, freeze, provenance, sha, write_json, write_npz, status
from tokenmean_replacement_common import apply, condition_names, posterior


def prepare(cfg):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from fit_revision_geometry import basis
    from revision_locality_common import derangement
    from hss.experiments.fitting import load_fitted
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'prepare_SUCCESS.json').exists():
        verify_inputs(root)
        return
    source = Path(cfg['mean_source']); locality = Path(cfg['locality'])
    table = pd.read_csv(source/'selection_ablation.csv')
    choice = table[(table.view == 'post') & (table.layer == cfg['layer']) &
                   (table.method == 'gmm') & (table.criterion == 'icl') &
                   np.isclose(table.tolerance, cfg['mean_icl_tolerance'])]
    assert len(choice) == 1
    selected = choice.iloc[0]; fit = Path(selected.fit_path)
    model, fit_info = load_fitted(fit)
    assert model.n_clusters() == cfg['expected_mean_k'] == int(selected.k)
    assert fit_info['model']['converged'] and fit_info['context']['layer'] == cfg['layer']
    transform = source/'cache/transforms'/fit_info['context']['transform_key']/'projection.npz'
    with np.load(transform) as z:
        assert np.all(z['mean']==0) and np.all(z['scale']==1)
        assert np.all(z['pca_mean']==0) and z['components'].size==z['whitening'].size==0
    files = [source/'selection_ablation.csv', source/'snapshots/post.json', fit/'model.npz',
             fit/'fit.json', fit/'assignments.npz', transform]
    snapshot = json.loads((source/'snapshots/post.json').read_text())
    cache = Path(cfg['mean_cache'])
    assert json.loads((cache/'_SUCCESS.json').read_text()) == snapshot
    assert fit_info['context']['snapshot'] == snapshot['key']
    assert snapshot['identity']['spec']['representation'] == 'mean'
    assert snapshot['hidden_dim'] == 3584 and snapshot['n_rows'] == 5000
    X = np.load(cache/f'layer_{cfg["layer"]}.npy').astype(np.float64)
    meta = pd.read_parquet(cache/'rows.parquet')
    files.extend([cache/'_SUCCESS.json', cache/'rows.parquet', cache/f'layer_{cfg["layer"]}.npy'])
    frames, token_records = [], {}
    for marker in sorted((Path(cfg['foundation'])/'prefixes').glob('shard_*/_SUCCESS.json')):
        frames.append(pd.read_parquet(marker.parent/'rows.parquet'))
        for row in json.loads((marker.parent/'tokens.json').read_text()):
            token_records[row['sample_id']] = row
        files.extend([marker, marker.parent/'rows.parquet', marker.parent/'tokens.json'])
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 5000 and not frame.sample_id.duplicated().any()
    train_ids = set(frame.loc[frame.split.eq('train'), 'sample_id'])
    train = meta.sample_id.isin(train_ids).to_numpy()
    assert train.sum() == len(train_ids) == 3011
    with np.load(fit/'model.npz') as z:
        decoder = {'centers': z['means'].copy(), 'variances': z['covariances'].copy(),
                   'weights': z['weights'].copy()}
    with threadpool_limits(limits=cfg['threads']):
        labels, densities = [], []
        for start in range(0, len(X), 128):
            block = X[start:start+128]
            p, ld = posterior(block, decoder)
            np.testing.assert_allclose(p, model.predict_proba(block), atol=1e-8, rtol=1e-7)
            labels.extend(p.argmax(1)); densities.extend(ld)
        labels = np.asarray(labels)
        with np.load(fit/'assignments.npz') as z:
            np.testing.assert_array_equal(labels, z['posterior'])
        Xt = X[train]; st = labels[train]
        counts = np.bincount(st, minlength=len(decoder['centers']))
        if np.any(counts <= cfg['rank']):
            raise ValueError('Insufficient training means for an eight-dimensional basis: '+str(counts))
        decoder['local_basis'] = np.stack([basis(Xt[st == j], cfg['rank'], mu).astype(np.float64)
                                          for j, mu in enumerate(decoder['centers'])])
        decoder['shared_basis'] = basis(Xt-decoder['centers'][st], cfg['rank'],
                                        np.zeros(X.shape[1])).astype(np.float64)
    for b in list(decoder['local_basis']) + [decoder['shared_basis']]:
        np.testing.assert_allclose(b @ b.T, np.eye(cfg['rank']), atol=2e-6)
    for seed in cfg['seeds']:
        decoder['permutation_'+str(seed)] = derangement(len(counts), seed)
    decoder['train_counts'] = counts
    write_npz(root/'mean_decoder.npz', **decoder)
    original_decoder = locality/'decoders/p16_l14_tokens/decoder.npz'
    original_receipt = json.loads((original_decoder.parent/'_SUCCESS.json').read_text())
    assert sha(original_decoder) == original_receipt['decoder_sha256']
    shutil.copyfile(original_decoder, root/'token_decoder.npz')
    files.extend([original_decoder, original_decoder.parent/'_SUCCESS.json'])
    oldplan = locality/'plan.json'; files.append(oldplan)
    old_selected = json.loads(oldplan.read_text())['selected_sample_ids']
    test_ids = set(frame.loc[frame.split.eq('test'), 'sample_id'])
    selected_ids = [sid for sid in old_selected if sid in test_ids]
    assert len(selected_ids) == 32
    capfile = original_decoder.parent/'pilot_activations.npz'; files.append(capfile)
    with np.load(capfile) as z: captures = dict(zip(z['sample_ids'].tolist(), z['x']))
    rows = frame.set_index('sample_id'); old_index = dict(zip(meta.sample_id, range(len(meta))))
    cases, arrays = [], {}
    for sid in selected_ids:
        r = rows.loc[sid]; item = token_records[sid]
        cases.append({'sample_id': sid, 'dataset': 'math', 'prompt_ids': item['prompt_ids'],
                      'response_ids': item['response_ids'], 'prompt_text': r.prompt_text,
                      'response_text': r.response_text, 'ground_truth': str(r.ground_truth),
                      'old_full_mean_region': int(labels[old_index[sid]]),
                      'legacy_samples': str(locality/'functional/samples')})
        arrays[sid] = captures[sid]
    gsm = Path(cfg['gsm'])
    receipt = json.loads((gsm/'collection_SUCCESS.json').read_text())
    files.extend([gsm/'collection_SUCCESS.json', gsm/'collection_audit.json', gsm/'collection_plan.json'])
    for row in json.loads((gsm/'collection_plan.json').read_text())['selected']:
        sid = row['sample_id']; path = gsm/'collection'/(sid+'.json')
        r = json.loads(path.read_text())
        assert sha(path) == receipt['records'][path.name] and r['source'] == row and r['prefix_valid']
        assert sha(path.with_suffix('.npz')) == r['activations_sha256']
        files.extend([path, path.with_suffix('.npz')])
        with np.load(path.with_suffix('.npz')) as z: arrays[sid] = z['x'].copy()
        cases.append({'sample_id': sid, 'dataset': 'gsm8k', 'prompt_ids': row['prompt_ids'],
                      'response_ids': r['response_ids'], 'prompt_text': row['prompt_text'],
                      'response_text': r['response_text'], 'ground_truth': row['ground_truth'],
                      'legacy_samples': str(gsm/'interventions/functional/samples')})
    assert len(cases) == len(arrays) == 96
    write_json(root/'cases.json', cases)
    write_npz(root/'captures.npz', sample_ids=np.array([c['sample_id'] for c in cases]),
              x=np.stack([arrays[c['sample_id']] for c in cases]))
    train_ld = np.asarray(densities)[train]
    write_json(root/'fit_audit.json', {'mean_k': len(counts), 'mean_tolerance': float(selected.tolerance),
        'full_mean_map_fit_questions': 5000, 'basis_fit_questions': int(train.sum()),
        'basis_fit_representation': 'whole-response mean, posterior MAP, fixed GMM anchors',
        'train_counts': counts.tolist(), 'posterior_replay_all5000_exact': True,
        'full_mean_train_density_quantiles': np.quantile(train_ld, [.01,.05,.5,.95,.99]).tolist(),
        'rank': cfg['rank'], 'gmm_refit': False, 'target_fit': False,
        'selected_fit': str(fit), 'source_map_sha256': sha(fit/'model.npz')})
    files.extend([root/'mean_decoder.npz', root/'token_decoder.npz', root/'cases.json',
                  root/'captures.npz', root/'fit_audit.json'])
    files.extend(Path(__file__).with_name(n) for n in ['run_tokenmean_replacement.py',
        'tokenmean_replacement_common.py', 'audit_tokenmean_replacement.py',
        'report_tokenmean_replacement.py', 'revision_common.py', 'evaluate_revision_locality.py',
        'extract_revision_prefixes.py', 'fit_revision_geometry.py'])
    identity = provenance(cfg, sorted(set(files)))
    identity.update(conditions=condition_names(), questions=96, expected_records=1536,
                    smoke_ids=[next(c['sample_id'] for c in cases if c['dataset']==ds) for ds in ['math','gsm8k']])
    freeze(root/'plan.json', identity)
    write_json(root/'prepare_SUCCESS.json', {'plan_sha256': sha(root/'plan.json'),
        'questions': 96, 'conditions_per_question': 16, 'mean_k': len(counts)})
    print(json.dumps({'prepared': True, 'k': len(counts), 'min_train_cluster': int(counts.min()),
                      'questions': 96, 'records': 1536}), flush=True)


def verify_inputs(root):
    plan = json.loads((root/'plan.json').read_text())
    for p, h in plan['files'].items():
        if sha(p) != h: raise ValueError('Frozen input changed: '+p)
    return plan


def evaluate(cfg, smoke=False):
    import fcntl
    import subprocess
    import torch
    from evaluate_revision_locality import measure
    from extract_revision_prefixes import load_model
    root = Path(cfg['output']); plan = verify_inputs(root)
    lock = (root/'gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest = root/('smoke' if smoke else 'functional'); dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists(): return
    if not smoke:
        assert json.loads((root/'smoke/audit.json').read_text())['complete']
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; refusing second model')
    assert shutil.disk_usage('/home/ubuntu').free > 50*1024**3
    cases = json.loads((root/'cases.json').read_text())
    if smoke: cases = [c for c in cases if c['sample_id'] in plan['smoke_ids']]
    with np.load(root/'captures.npz') as z: captures = dict(zip(z['sample_ids'].tolist(), z['x']))
    with np.load(root/'mean_decoder.npz') as z: mean = {k:z[k].copy() for k in z.files}
    with np.load(root/'token_decoder.npz') as z: token = {k:z[k].copy() for k in z.files}
    torch.set_num_threads(cfg['threads'])
    model, _ = load_model(cfg)
    started = time.monotonic(); done = 0; expected = len(cases)*len(plan['conditions'])
    for case in cases:
        sid = case['sample_id']; x_saved = captures[sid]
        prefix = case['prompt_ids']+case['response_ids'][:cfg['prefix_tokens']]
        reference = case['response_ids'][cfg['prefix_tokens']:]
        positions = list(range(len(prefix)-cfg['window'], len(prefix)))
        assert len(reference) and x_saved.shape == (16,3584)
        baseline = None
        for name in plan['conditions']:
            path = dest/'samples'/(sid+'__'+name+'.json'); array = path.with_suffix('.npz')
            if not smoke and not path.exists():
                old = root/'smoke/samples'/path.name
                if old.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(old,path); shutil.copyfile(old.with_suffix('.npz'),array)
            if path.exists():
                row = json.loads(path.read_text()); assert sha(array)==row['arrays_sha256']
                if name == 'identity':
                    with np.load(array) as z: baseline=z['logp'].astype(np.float64)
                done += 1; continue
            details = {}
            def transform(h):
                x = h.float().cpu().numpy()
                np.testing.assert_array_equal(x, x_saved)
                z, info = apply(x, mean, token, name)
                actual = torch.as_tensor(z, device=h.device, dtype=h.dtype)
                details.update(meta=info, ideal=z, actual=actual.float().cpu().numpy())
                return actual
            metrics, logp, losses = measure(model, prefix, reference, cfg['layer'], positions, transform)
            if name == 'identity': baseline = logp.astype(np.float64)
            assert baseline is not None
            kl = float((np.exp(baseline)*(baseline-logp.astype(np.float64))).sum())
            actual = details['actual'].astype(np.float64)
            metrics.update(next_token_kl=kl, argmax_agreement=bool(baseline.argmax()==logp.argmax()),
                           token_mse=float(np.square(actual-x_saved).mean()),
                           mean_mse=float(np.square(actual.mean(0)-x_saved.astype(np.float64).mean(0)).mean()),
                           centered_change_mse=float(np.square((actual-actual.mean(0))-
                               (x_saved.astype(np.float64)-x_saved.astype(np.float64).mean(0))).mean()))
            write_npz(array, logp=logp, reference_nll=losses, ideal=details['ideal'], actual=details['actual'])
            write_json(path, {'sample_id':sid, 'dataset':case['dataset'], 'condition':name,
                'positions':positions, 'meta':details['meta'], 'metrics':metrics,
                'arrays_sha256':sha(array), 'plan_sha256':sha(root/'plan.json')})
            done += 1
            status(root, 'smoke' if smoke else 'functional', state='running', completed=done,
                   expected=expected, seconds=time.monotonic()-started)
            if done % 16 == 0: print(json.dumps({'completed':done,'expected':expected,'seconds':time.monotonic()-started}),flush=True)
    write_json(dest/'_SUCCESS.json', {'conditions':done,'questions':len(cases),
        'seconds':time.monotonic()-started,'plan_sha256':sha(root/'plan.json')})
    status(root, 'smoke' if smoke else 'functional', state='complete', completed=done, expected=expected)


def queue(cfg, config_path):
    import fcntl
    import subprocess
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=Path(__file__).resolve().parents[1]/'.venv/bin/python'
    gpu=Path('/lambda/nfs/dami/openact/.venv/bin/python')
    jobs=[('prepare',cpu,'run_tokenmean_replacement.py',['--stage','prepare']),
          ('smoke',gpu,'run_tokenmean_replacement.py',['--stage','smoke']),
          ('smoke_audit',gpu,'audit_tokenmean_replacement.py',['--smoke']),
          ('functional',gpu,'run_tokenmean_replacement.py',['--stage','evaluate']),
          ('audit',gpu,'audit_tokenmean_replacement.py',[]),
          ('report',cpu,'report_tokenmean_replacement.py',[])]
    state={'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(cfg['threads'])
    for stage,python,script,args in jobs:
        state['phase']=stage;write_json(root/'queue_status.json',state)
        with (root/(stage+'.log')).open('a') as log:
            child=subprocess.Popen([str(python),str(Path(__file__).with_name(script)),
                                    '--config',str(Path(config_path).resolve()),*args],env=env,stdout=log,stderr=log)
            state['stages'][stage]={'state':'running','pid':child.pid,'started_unix':time.time()}
            write_json(root/'queue_status.json',state)
            rc=child.wait()
        state['stages'][stage].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time())
        if rc:
            state['state']='failed';write_json(root/'queue_status.json',state);raise RuntimeError(stage+' failed')
    state.update(state='complete',finished_unix=time.time(),delivery='visual_review_pending')
    write_json(root/'queue_status.json',state)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['prepare','smoke','evaluate','queue'],required=True)
    a=p.parse_args();cfg=config(a.config)
    if a.stage=='queue':queue(cfg,a.config)
    elif a.stage=='prepare':prepare(cfg)
    else:evaluate(cfg,smoke=a.stage=='smoke')

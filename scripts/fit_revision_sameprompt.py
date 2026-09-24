"""Finite grouped readouts; tuning labels select C, calibration/test labels do not."""
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits
from revision_common import freeze, provenance, sha, write_json, write_npz, status
from revision_sameprompt_io import read_inputs, verify_record_files


VIEWS = [(16, 28), (16, 14), (64, 28)]


def candidates(x_train, y_train, x_tune, y_tune, x_all, grid, iterations):
    """No calibration/test label argument; retain every convergence outcome."""
    if set(np.unique(y_train)) != {0, 1} or not len(y_tune):
        raise ValueError('Training needs both classes and tuning needs valid trajectories')
    trials = []; fitted = {}
    for c in grid:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=iterations, tol=1e-5, random_state=42)
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x_train, y_train)
            initial = int(model.n_iter_.max()); used = initial
            initial_warning_count = len(seen)
            if initial >= iterations:
                model.set_params(warm_start=True, max_iter=iterations).fit(x_train, y_train)
                used += int(model.n_iter_.max())
            last_warnings = seen[initial_warning_count:] if initial >= iterations else seen
        converged = int(model.n_iter_.max()) < iterations and not any(
            issubclass(w.category, ConvergenceWarning) for w in last_warnings)
        validation_loss = float(log_loss(y_tune, model.predict_proba(x_tune)[:, 1].astype(np.float64), labels=[0, 1]))
        trials.append({'C': c, 'converged': converged, 'iterations_total': used,
            'tuning_log_loss': validation_loss, 'seconds': time.monotonic()-started,
            'convergence_warnings': sum(issubclass(w.category, ConvergenceWarning) for w in seen)})
        fitted[c] = {'coef': model.coef_.copy(), 'intercept': model.intercept_.copy(),
                     'probability': model.predict_proba(x_all)[:, 1].astype(np.float64)}
    valid = [r for r in trials if r['converged']]
    if not valid:
        raise RuntimeError('All C candidates failed convergence; no scientific comparison')
    selected = min(valid, key=lambda r: (r['tuning_log_loss'], r['C']))
    return selected, trials, fitted


def load_views(root):
    plan, inputs = read_inputs(root); cfg = plan['config']; folder = root/'full'
    receipt = json.loads((folder/'_SUCCESS.json').read_text())
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['complete'] and audit['stage_receipt_sha256'] == sha(folder/'_SUCCESS.json')
    by_id = {r['sample_id']: r for r in inputs}; prompts = {}; all_rows = []; views = {}
    for prefix, layer in VIEWS:
        views[prefix, layer] = {'metadata': [], 'prompt': [], 'current': [], 'mean16': [], 'allmean': [], 'numeric': [], 'categories': []}
    for name, digest in sorted(receipt['records'].items()):
        path = folder/'samples'/name; assert sha(path) == digest
        r = json.loads(path.read_text()); verify_record_files(folder, r); source = by_id[r['sample_id']]
        meta = {'trajectory_id': r['trajectory']['trajectory_id'], 'sample_id': r['sample_id'],
            'question_group': r['question_group'], 'role': r['role'], 'failure': int(not r['correct']),
            'observed_length': r['length'], 'finish_reason': r['finish_reason']}
        all_rows.append(meta)
        if r['sample_id'] not in prompts:
            with np.load(folder/r['prompt_file']) as z:
                prompts[r['sample_id']] = z['current'].copy()
        for prefix in cfg['prefixes']:
            entry = r['prefixes'][str(prefix)]
            if not entry['valid']:
                continue
            with np.load(folder/entry['file']) as z:
                current, mean, whole = z['current'], z['mean_window'], z['mean_all']
            for p, layer in VIEWS:
                if p != prefix:
                    continue
                target = views[p, layer]; index = cfg['layers'].index(layer)
                prompt = prompts[r['sample_id']][index]
                target['metadata'].append(meta)
                for key, value in [('prompt', prompt), ('current', current[index]), ('mean16', mean[index]), ('allmean', whole[index])]:
                    target[key].append(value)
                target['numeric'].append([np.log1p(len(source['prompt_ids'])), np.linalg.norm(prompt.astype(float)),
                    np.linalg.norm(current[index].astype(float)), np.linalg.norm(mean[index].astype(float)),
                    entry['entropy'], entry['margin']])
                target['categories'].append([source['category'], source['level']])
    assert len(all_rows) == 896 and len({r['trajectory_id'] for r in all_rows}) == 896
    return cfg, pd.DataFrame(all_rows), views


def transformed(view, directory):
    frame = pd.DataFrame(view['metadata']); train = frame.role.eq('train').to_numpy()
    if not train.any():
        raise ValueError('No valid training prefix')
    arrays, transforms = {}, {}
    for key in ['prompt', 'current', 'mean16', 'allmean', 'numeric']:
        raw = np.asarray(view[key], np.float64); scaler = StandardScaler().fit(raw[train])
        arrays[key] = scaler.transform(raw).astype(np.float32)
        transforms[key+'_mean'] = scaler.mean_; transforms[key+'_scale'] = scaler.scale_
    categorical = np.asarray(view['categories'], str)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(categorical[train])
    arrays['controls'] = np.column_stack([arrays.pop('numeric'), encoder.transform(categorical)]).astype(np.float32)
    write_npz(directory/'scaled_features.npz', **arrays)
    write_npz(directory/'feature_transforms.npz', **transforms)
    frame.to_parquet(directory/'rows.parquet', index=False)
    write_json(directory/'feature_schema.json', {'categories': [x.tolist() for x in encoder.categories_],
        'numeric': ['log1p_prompt_length', 'prompt_norm', 'current_norm', 'mean16_norm', 'raw_entropy', 'raw_margin'],
        'vector_blocks': ['prompt', 'current', 'mean16', 'allmean'],
        'transform_fit_role': 'train', 'future_length_or_label_features': False})
    return frame, arrays


def design(arrays, method):
    blocks = [arrays['prompt'], arrays['current'], arrays['controls']]
    extra = {'mean16': 'mean16', 'duplicate_current': 'current', 'allmean': 'allmean'}
    if method != 'baseline':
        blocks.append(arrays[extra[method]])
    return np.column_stack(blocks)


def run(root):
    dest = root/'readouts'; dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists():
        raise RuntimeError('Readouts complete; audit and report without refitting')
    if (dest/'plan.json').exists():
        raise RuntimeError('Readout attempt already exists; inspect rather than overwrite partial fitting')
    cfg, all_rows, views = load_views(root)
    files = [root/'plan.json', root/'full/_SUCCESS.json', root/'full/audit.json', Path(__file__),
             Path(__file__).with_name('revision_sameprompt_io.py'), Path(__file__).with_name('revision_common.py')]
    freeze(dest/'plan.json', provenance(cfg, files))
    all_rows.to_parquet(dest/'all_trajectories.parquet', index=False)
    identities = all_rows.groupby('question_group').role.nunique()
    assert identities.eq(1).all() and all_rows.groupby('question_group').size().eq(4).all()
    completed = []; start = time.monotonic()
    with threadpool_limits(limits=4):
        for prefix, layer in VIEWS:
            name = f'p{prefix}_l{layer}'; folder = dest/name; folder.mkdir(exist_ok=True)
            frame, arrays = transformed(views[prefix, layer], folder)
            y = frame.failure.to_numpy(int); train = frame.role.eq('train').to_numpy(); tune = frame.role.eq('tuning').to_numpy()
            predictions = frame.copy(); methods = ['baseline', 'mean16', 'duplicate_current']
            if prefix == 64:
                methods.append('allmean')
            for method in methods:
                status(root, 'readout', state='running', view=name, method=method, completed_methods=len(completed))
                x = design(arrays, method); target = folder/method; target.mkdir(exist_ok=True)
                selected, trials, fitted = candidates(x[train], y[train], x[tune], y[tune], x,
                                                     cfg['readout_C_grid'], cfg['max_iter'])
                for c, model in fitted.items():
                    write_npz(target/f'C_{c:g}.npz', **model)
                write_json(target/'selection.json', {'selected': selected, 'trials': trials,
                    'train_ids': frame.loc[train, 'trajectory_id'].tolist(), 'tuning_ids': frame.loc[tune, 'trajectory_id'].tolist(),
                    'criterion': 'tuning log loss; smaller C on exact tie', 'n_features': x.shape[1]})
                predictions[method] = fitted[selected['C']]['probability']
                completed.append({'view': name, 'method': method, 'selected_C': selected['C']})
            predictions.to_parquet(folder/'predictions.parquet', index=False)
    files = [p for p in dest.rglob('*') if p.is_file()]
    write_json(dest/'_SUCCESS.json', {'complete': True, 'methods': completed, 'seconds': time.monotonic()-start,
        'files': {str(p.relative_to(dest)): sha(p) for p in files}, 'plan_sha256': sha(dest/'plan.json')})
    status(root, 'readout', state='complete', completed_methods=len(completed), seconds=time.monotonic()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)

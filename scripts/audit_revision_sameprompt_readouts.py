"""Rebuild feature transforms and predictions, check C selection and question roles."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from revision_common import sha, write_json


def run(root):
    dest = root/'readouts'; success = json.loads((dest/'_SUCCESS.json').read_text())
    assert success['complete'] and success['plan_sha256'] == sha(dest/'plan.json')
    for name, digest in success['files'].items():
        assert sha(dest/name) == digest, name
    plan = json.loads((dest/'plan.json').read_text()); cfg = plan['config']
    for name, digest in plan['files'].items():
        assert sha(name) == digest, name
    inputs = {r['sample_id']: r for r in json.loads((root/'inputs.json').read_text())}
    source = root/'full'; receipt = json.loads((source/'_SUCCESS.json').read_text())
    assert json.loads((source/'audit.json').read_text())['stage_receipt_sha256'] == sha(source/'_SUCCESS.json')
    originals = {}
    for name, digest in receipt['records'].items():
        path = source/'samples'/name; assert sha(path) == digest
        originals[path.stem] = json.loads(path.read_text())
    universe = pd.read_parquet(dest/'all_trajectories.parquet')
    assert len(universe) == len(originals) == 896 and not universe.trajectory_id.duplicated().any()
    for row in universe.itertuples():
        original = originals[row.trajectory_id]
        assert row.sample_id == original['sample_id'] and row.question_group == original['question_group']
        assert row.role == original['role'] and row.failure == int(not original['correct'])
        assert row.observed_length == original['length'] and row.finish_reason == original['finish_reason']
    assert universe.groupby('question_group').role.nunique().eq(1).all()
    assert universe.groupby('question_group').size().eq(4).all()
    assert universe.groupby('role').question_group.nunique().to_dict() == cfg['role_counts']
    checked = []; worst_prediction_error = 0.
    for prefix, layer in [(16, 28), (16, 14), (64, 28)]:
        folder = dest/f'p{prefix}_l{layer}'; rows = pd.read_parquet(folder/'rows.parquet')
        predictions = pd.read_parquet(folder/'predictions.parquet')
        assert predictions[rows.columns].equals(rows)
        expected_ids = {tid for tid, r in originals.items() if r['prefixes'][str(prefix)]['valid']}
        assert set(rows.trajectory_id) == expected_ids and not rows.trajectory_id.duplicated().any()
        train = rows.role.eq('train').to_numpy(); tuning = rows.role.eq('tuning').to_numpy()
        raw = {k: [] for k in ['prompt', 'current', 'mean16', 'allmean', 'numeric']}; categories = []
        index = cfg['layers'].index(layer)
        for row in rows.itertuples():
            original = originals[row.trajectory_id]; item = inputs[original['sample_id']]
            assert row.failure == int(not original['correct']) and row.role == item['role']
            assert row.question_group == item['question_group']
            with np.load(source/original['prompt_file']) as z:
                prompt = z['raw'][index, -1].copy()
            entry = original['prefixes'][str(prefix)]
            with np.load(source/entry['file']) as z:
                values = z['raw'][index].astype(float)
                current = values[-1]; mean = values[-16:].mean(0).astype(np.float32)
                whole = values.mean(0).astype(np.float32)
                lp = z['logprobs'].astype(float); logits = np.sort(z['logits'].astype(float))
            for key, value in [('prompt', prompt), ('current', current), ('mean16', mean), ('allmean', whole)]:
                raw[key].append(value)
            raw['numeric'].append([np.log1p(len(item['prompt_ids'])), np.linalg.norm(prompt.astype(float)),
                np.linalg.norm(current), np.linalg.norm(mean.astype(float)),
                float(-(np.exp(lp)*lp).sum()), float(logits[-1]-logits[-2])])
            categories.append([item['category'], item['level']])
        with np.load(folder/'scaled_features.npz') as z:
            saved = {k: z[k] for k in z.files}
        with np.load(folder/'feature_transforms.npz') as z:
            transforms = {k: z[k] for k in z.files}
        rebuilt = {}
        for key in raw:
            x = np.asarray(raw[key], float); mean = x[train].mean(0); var = x[train].var(0)
            n = int(train.sum()); eps = np.finfo(float).eps
            constant = var <= n*eps*var+(n*mean*eps)**2
            scale = np.sqrt(var); scale[constant] = 1.
            np.testing.assert_allclose(transforms[key+'_mean'], mean, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(transforms[key+'_scale'], scale, atol=1e-10, rtol=1e-10)
            rebuilt[key] = ((x-mean)/scale).astype(np.float32)
        schema = json.loads((folder/'feature_schema.json').read_text()); cat = np.asarray(categories, str)
        levels = [sorted(set(cat[train, j])) for j in range(2)]
        assert levels == schema['categories'] and schema['future_length_or_label_features'] is False
        categorical = np.column_stack([(cat[:, j] == value).astype(float) for j in range(2) for value in levels[j]])
        rebuilt['controls'] = np.column_stack([rebuilt.pop('numeric'), categorical]).astype(np.float32)
        assert set(rebuilt) == set(saved)
        for name in saved:
            np.testing.assert_allclose(saved[name], rebuilt[name], rtol=1e-5, atol=2e-6)
        families = ['baseline', 'mean16', 'duplicate_current']+(['allmean'] if prefix == 64 else [])
        for method in families:
            blocks = [saved[k] for k in ['prompt', 'current', 'controls']]
            if method != 'baseline':
                blocks.append(saved['current' if method == 'duplicate_current' else method])
            x = np.column_stack(blocks)
            selection = json.loads((folder/method/'selection.json').read_text())
            assert selection['train_ids'] == rows.loc[train, 'trajectory_id'].tolist()
            assert selection['tuning_ids'] == rows.loc[tuning, 'trajectory_id'].tolist()
            assert selection['n_features'] == x.shape[1]
            assert [r['C'] for r in selection['trials']] == cfg['readout_C_grid']
            losses = []
            for trial in selection['trials']:
                with np.load(folder/method/f'C_{trial["C"]:g}.npz') as z:
                    probability = expit(x@z['coef'][0]+z['intercept'][0]).astype(np.float64)
                    np.testing.assert_allclose(probability, z['probability'], rtol=0, atol=1e-10)
                p = np.clip(probability[tuning], np.finfo(float).eps, 1-np.finfo(float).eps)
                y = rows.failure.to_numpy()[tuning]
                loss = float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
                np.testing.assert_allclose(loss, trial['tuning_log_loss'], rtol=0, atol=1e-10)
                if trial['converged']:
                    losses.append((loss, trial['C']))
                assert trial['iterations_total'] <= 2*cfg['max_iter']
                if trial['C'] == selection['selected']['C']:
                    error = float(np.max(np.abs(probability-predictions[method].to_numpy())))
                    assert error < 1e-10; worst_prediction_error = max(worst_prediction_error, error)
            assert min(losses)[1] == selection['selected']['C']
            checked.append({'view': folder.name, 'method': method})
    assert len(checked) == len(success['methods']) == 10
    result = {'complete': True, 'readout_receipt_sha256': sha(dest/'_SUCCESS.json'),
        'views': 3, 'methods': len(checked), 'candidates': 30,
        'question_roles_raw_features_train_transforms_tuning_selection_and_weights_verified': True,
        'worst_prediction_error': worst_prediction_error, 'audit_source_sha256': sha(Path(__file__))}
    write_json(dest/'audit.json', result); print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)

"""CPU-only receipt, selection and export audit of a completed MMLU model.

No fitting, extra search or correctness prediction. All candidate scalar records
and restart metadata are checked; numerical arrays are checked for the primary
selection, and every portable export gets full Result validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from hss.results import Result, load_layer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, key):
    start = time.time()
    root = Path(root)
    out = root / key
    read = lambda path: json.loads(Path(path).read_text())
    protocol = read(root / 'protocol.json')
    cfg = protocol['identity']['config']
    model = next(m for m in cfg['models'] if m['key'] == key)
    complete = read(out / 'COMPLETE.json')
    n, d = cfg['samples'], model['dimension']
    metrics = pd.read_csv(out / 'candidate_metrics.csv')
    selections = read(out / 'selection.json')
    assert len(metrics) == complete['n_candidates']
    assert int(metrics.converged.sum()) == complete['converged_candidates']
    assert not metrics.duplicated(['view', 'layer', 'k']).any()
    expected_layers = {('post', i) for i in range(model['last_layer'] + 1)}
    if cfg['include_pre_final']:
        expected_layers.add(('pre_final', model['last_layer']))
    assert set(zip(metrics['view'], metrics.layer)) == expected_layers
    assert len(selections) == len(expected_layers) * len(cfg['icl_tolerances'])
    restart_converged = restart_total = 0
    for row in metrics.to_dict('records'):
        path = Path(row['fit_path'])
        marker = read(path / 'complete.json')['files']
        assert sha(path / 'fit.json') == marker['fit.json']
        fit = read(path / 'fit.json')
        assert len(fit['restarts']) == cfg['n_init']
        valid = []
        for i, restart in enumerate(fit['restarts']):
            name = f'restart_{i}.json'
            assert sha(path / name) == marker[name]
            assert read(path / name) == restart
            assert restart['seed'] == cfg['seed'] + 1009 * row['layer'] + 10007 * i
            restart_total += 1
            if 'model' in restart:
                assert (path / f'restart_{i}.npz').exists()
                assert marker[f'restart_{i}.npz'] == restart['model_sha256']
            if restart['converged']:
                restart_converged += 1
                valid.append(restart)
        assert bool(valid) == bool(row['converged'])
        if valid:
            k = int(row['k'])
            assert row['n_parameters'] == 2 * k * d + k - 1
            np.testing.assert_allclose(row['log_likelihood'], max(v['log_likelihood'] for v in valid), atol=1e-5, rtol=1e-10)
            bic = -2 * row['log_likelihood'] + row['n_parameters'] * np.log(n)
            np.testing.assert_allclose([row['bic'], row['icl']], [bic, bic + 2 * row['entropy']], atol=1e-5, rtol=1e-10)
    for view, layer in expected_layers:
        rows = metrics[(metrics['view'] == view) & (metrics.layer == layer)]
        valid = rows[rows.converged & np.isfinite(rows.icl)]
        best = float(valid.icl.min())
        marker = read(out / 'layers' / f'{view}_{layer}.json')
        assert marker['complete'] and marker['tested_k'] == sorted(rows.k.tolist())
        for tolerance in cfg['icl_tolerances']:
            chosen = valid[valid.icl <= best + tolerance * max(abs(best), 1)].sort_values(['k', 'icl']).iloc[0]
            selection = next(s for s in selections if (s['view'], s['layer'], s['tolerance']) == (view, layer, tolerance))
            assert selection['k'] == chosen.k and selection['fit_path'] == chosen.fit_path
            assert selection['k_evaluated'] == marker['tested_k']
            assert selection['upper_boundary_warning'] == marker['at_upper_boundary']
    exports = read(out / 'exports.json')
    assert exports == complete['exports']
    export_checks = {}
    for name, path in exports.items():
        result = Result(path)
        check = result.validate(full=True)
        assert check['valid'], check
        export_checks[name] = check
    primary_checks = 0
    for selection in selections:
        if selection['tolerance'] != cfg['primary_tolerance']:
            continue
        path = Path(selection['fit_path'])
        marker = read(path / 'complete.json')['files']
        for name in ['model.npz', 'assignments.npz']:
            assert sha(path / name) == marker[name]
        export = Path(exports[f"{selection['view']}/{selection['tolerance']:g}"])
        layer = selection['layer']
        fitted, projection, _ = load_layer(export, layer)
        with np.load(path / 'assignments.npz') as z:
            prob, assigned = z['posterior_probability'], z['posterior']
            assert prob.shape == (n, selection['k'])
            assert np.isfinite(prob).all() and (prob >= 0).all()
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=2e-6)
            assert np.array_equal(prob.argmax(axis=1), assigned)
            columns = read(export / 'alignment.json')['layers']
            assert np.array_equal(np.load(export / 'local_states.npy')[:, columns.index(layer)], assigned)
        assert fitted.covariance_type == 'diag'
        assert np.isfinite(fitted.means_).all() and (fitted.covariances_ > 0).all()
        data_path = Path(read(out / 'snapshots' / f"{selection['view']}.json")['path'])
        x = np.load(data_path / f'layer_{layer}.npy', mmap_mode='r')[np.linspace(0, n - 1, 16, dtype=int)]
        np.testing.assert_array_equal(projection.transform(x), x)
        primary_checks += 1
    receipt = dict(passed=True, model=model['identifier'], n_samples=n,
        candidates=len(metrics), converged_candidates=int(metrics.converged.sum()),
        restarts=restart_total, converged_restarts=restart_converged,
        layer_views=len(expected_layers), selections=len(selections), primary_array_checks=primary_checks,
        exports=export_checks, seconds=time.time()-start,
        hashes={name: sha(out/name) for name in ['COMPLETE.json','candidate_metrics.csv','selection.json','exports.json']},
        scope='All candidate/restart metadata and ICL selections, all portable exports with full hash validation, primary selected probability/model arrays and identity projections. Nonselected restart array hashes were not re-read; no new fitting.')
    (out / 'independent_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'exports'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    with threadpool_limits(2):
        audit(a.root, a.model)

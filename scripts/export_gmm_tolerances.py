"""Export complete HSS maps from frozen GMM candidates, without fitting.

Selection is label-free. Original fits remain immutable; every export contains
portable parameters, both assignment policies, posterior probabilities, and
the full per-layer candidate scan. CPU replay checks all saved assignments.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hss.align import align_layers
from hss.cluster.assignment import assign
from hss.data import CachedStates
from hss.experiments.artifacts import digest, file_digest, lock, save_json, save_npz
from hss.experiments.config import Experiment
from hss.experiments.evaluate import characterize
from hss.experiments.fitting import load_fitted, select_candidate
from hss.results import Result
from hss.results.models import load_layer, save_layer
from hss.results.store import seal_result
from hss.transform.projection import Projection
from hss.types import AlignSpec


def select_complete(records, cluster, tolerance):
    expected = set(range(cluster.k_min, cluster.k_max + 1))
    if len(records) != len(expected) or {r['k'] for r in records} != expected:
        raise ValueError('Incomplete or duplicated source K scan; refusing export')
    cfg = replace(cluster, selection_criterion='icl', parsimony_tolerance=tolerance,
                  require_convergence=True)
    return select_candidate(deepcopy(records), cfg)


def export_view(source, output, view, tolerances, base, all_records, cache_root):
    snapshot_file = source / 'snapshots' / f'{view}.json'
    snapshot = json.loads(snapshot_file.read_text())
    data = CachedStates(cache_root / 'data' / snapshot['key'])
    if data.info != snapshot:
        raise ValueError('Source snapshot identity changed')
    if data.n_items() != 5000 or data.state_dim() != 3584:
        raise ValueError('Expected complete Qwen-MATH 5000 x 3584 snapshot')
    meta = pd.read_parquet(source / 'snapshots' / f'{view}_rows.parquet')
    pd.testing.assert_frame_equal(meta, data.meta)
    if data.info['identity']['spec']['representation'] != 'mean':
        raise ValueError('This export requires response token means')
    dim = data.state_dim()
    projection = Projection(np.zeros(dim), np.ones(dim), np.zeros(dim),
                            np.empty((0, dim)), np.empty(0))
    roots = {t: output / f'icl_{round(t * 100):02d}pct' / view for t in tolerances}
    scans = {t: [] for t in tolerances}
    centers = {t: [] for t in tolerances}
    local = {t: [] for t in tolerances}
    nearest = {t: [] for t in tolerances}
    table, replay = [], []
    for layer in data.layers():
        candidates = sorted([r['record'] for r in all_records
                             if r['view'] == view and r['layer'] == layer], key=lambda r: r['k'])
        choices = {t: select_complete(candidates, base.cluster, t) for t in tolerances}
        # Float64 identity projection matches the original fitting input.
        X = projection.transform(data.array(layer))
        loaded = {}
        for t, selected in choices.items():
            fit = Path(selected['fit_path'])
            if fit not in loaded:
                model, info = load_fitted(fit)
                if (info['context']['snapshot'] != snapshot['key']
                        or info['context']['layer'] != layer
                        or model.n_clusters() != selected['k']
                        or not info['model']['converged']):
                    raise ValueError('Selected fit provenance/shape/convergence mismatch')
                with np.load(fit / 'assignments.npz', allow_pickle=False) as f:
                    saved = {k: f[k] for k in ['posterior', 'nearest']}
                probs, near = [], []
                for start in range(0, len(X), 256):
                    block = X[start:start + 256]
                    probs.append(model.predict_proba(block))
                    near.append(assign(model, block, 'nearest'))
                probs = np.concatenate(probs)
                near = np.concatenate(near)
                if not np.isfinite(probs).all():
                    raise ValueError('Nonfinite probabilities')
                np.testing.assert_allclose(probs.sum(1), 1., atol=1e-10)
                np.testing.assert_array_equal(probs.argmax(1), saved['posterior'])
                np.testing.assert_array_equal(near, saved['nearest'])
                hashes = {f: file_digest(fit / f) for f in ['model.npz', 'fit.json', 'assignments.npz']}
                loaded[fit] = model, saved, probs, hashes
                replay.append(dict(view=view, layer=layer, k=selected['k'], rows=len(X),
                                   fit_path=str(fit), posterior_exact=True, nearest_exact=True,
                                   source_sha256=hashes))
            model, saved, probs, hashes = loaded[fit]
            root = roots[t]
            best = min(r['icl'] for r in candidates if r['converged'])
            scan = dict(selected=selected, candidates=candidates,
                        requested_k=list(range(base.cluster.k_min, base.cluster.k_max + 1)),
                        selection_criterion='icl', parsimony_tolerance=t,
                        best_icl=best, admissible_icl_max=best + t * max(abs(best), 1.),
                        excluded_unconverged=[r['k'] for r in candidates if not r['converged']],
                        source_sha256=hashes, refitted=False)
            save_layer(root, layer, model, projection, scan)
            lp = root / 'models' / f'layer_{layer}'
            shutil.copyfile(fit / 'fit.json', lp / 'source_fit.json')
            save_npz(lp / 'assignments.npz', **saved)
            save_npz(lp / 'posterior_probabilities.npz', probabilities=probs.astype(np.float32))
            restored, restored_projection, _ = load_layer(root, layer)
            for key, value in model.state_arrays().items():
                np.testing.assert_array_equal(restored.state_arrays()[key], value)
            np.testing.assert_array_equal(restored_projection.transform(X[:2]), X[:2])
            scans[t].append(dict(layer=layer, **scan))
            centers[t].append(model.centers())
            local[t].append(saved['posterior'])
            nearest[t].append(saved['nearest'])
            table.append(dict(view=view, layer=layer, tolerance=t, k=selected['k'],
                              icl=selected['icl'], best_icl=best,
                              icl_gap_fraction=(selected['icl']-best)/max(abs(best), 1.),
                              source_fit=str(fit), result=str(root)))
        print(json.dumps(dict(view=view, layer=layer, k={str(t): c['k'] for t, c in choices.items()})), flush=True)
    exports = []
    for t, root in roots.items():
        started = time.time()
        alignment = align_layers(centers[t], layers=data.layers(),
                                 spec=AlignSpec(**base.alignment.__dict__, allow_negative=True))
        labels = np.column_stack(local[t])
        states = np.column_stack([v[y] for v, y in zip(alignment.local_to_global, local[t])])
        np.save(root / 'local_states.npy', labels, allow_pickle=False)
        np.save(root / 'nearest_local_states.npy', np.column_stack(nearest[t]), allow_pickle=False)
        np.save(root / 'states.npy', states, allow_pickle=False)
        meta.to_parquet(root / 'rows.parquet', index=False)
        save_json(root / 'data_snapshot.json', data.info)
        save_json(root / 'selection.json', scans[t])
        save_json(root / 'alignment.json', dict(layers=data.layers(),
                  local_to_global=[v.tolist() for v in alignment.local_to_global]))
        save_json(root / 'diagnostics.json', [])
        idx = np.arange(len(meta)); empty = np.array([], dtype=int)
        save_npz(root / 'split.npz', train=idx, map_fit=idx, validation=empty, test=empty)
        cfg = base.to_dict()
        cfg['name'] = f'qwen_math_mean_gmm_icl_{round(t*100):02d}pct_{view}'
        cfg['data'] = deepcopy(snapshot['identity']['spec'])
        cfg['cluster'].update(method='gmm', rank=0, assignment='posterior',
                              selection_criterion='icl', parsimony_tolerance=t)
        cfg['evaluation'].update(mode='geometry', fixed_map=None, fixed_k_map=None)
        cfg['execution']['output_root'] = str(output)
        Experiment.from_dict(cfg).validate()
        save_json(root / 'config.json', cfg)
        assoc, tags, trans = characterize(states, meta, data.layers())
        for name, frame in [('associations', assoc), ('state_tags', tags), ('transitions', trans)]:
            frame.to_csv(root / f'{name}.csv', index=False)
        key = digest(dict(snapshot=snapshot['key'], view=view, tolerance=t,
                          fits=[s['source_sha256'] for s in scans[t]], alignment=cfg['alignment']))
        summary = dict(name=cfg['name'], trial_id=key, path=str(root), snapshot=snapshot['key'],
                       n_samples=meta.sample_id.nunique(), n_rows=len(meta), model=snapshot['model'],
                       n_global_states=int(states.max())+1, seconds=time.time()-started,
                       profile=[dict(layer=s['layer'], k=s['selected']['k'],
                                     criterion=s['selected']['icl'], criterion_name='icl') for s in scans[t]],
                       evaluation={'scope': 'descriptive full-data geometry'}, refitted=False)
        save_json(root / 'summary.json', summary)
        save_json(root / 'replay_audit.json', dict(valid=True, checks=[r for r in replay
                  if any(s['layer']==r['layer'] and s['selected']['k']==r['k'] for s in scans[t])]))
        seal_result(root)
        save_json(root / '_SUCCESS.json', summary)
        audit = Result(root).validate(full=True)
        if not audit['valid']:
            raise ValueError(audit)
        exports.append(dict(view=view, tolerance=t, path=str(root), audit=audit))
    return table, exports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/home/ubuntu/hss-cache'))
    p.add_argument('--tolerances', type=float, nargs='+', default=[.03, .04, .05])
    args = p.parse_args()
    ts = sorted(set(args.tolerances))
    if not ts or any(not 0 < t < 1 or abs(t*100-round(t*100)) > 1e-9 for t in ts):
        p.error('Use distinct whole-percentage tolerances between 0 and 100 percent')
    output = args.output.resolve(); source = args.source.resolve()
    with lock(output / '.lock'), threadpool_limits(limits=2):
        if (output / 'delivery.json').exists():
            raise FileExistsError('Completed export is immutable; use a new output directory')
        protocol = json.loads((source / 'protocol.json').read_text())
        base = Experiment.from_dict(protocol['identity']['config'])
        if base.transform.standardize or base.transform.whiten or base.transform.pca_components:
            raise ValueError('Only raw, untransformed source fits are supported')
        records = []
        for path in (source / 'candidates').glob('*.json'):
            r = json.loads(path.read_text())
            if r['method']=='gmm' and r['rank']==0 and r['seed']==base.seed and r['status']=='complete':
                records.append(r)
        identity = dict(source=str(source), source_protocol_sha256=file_digest(source/'protocol.json'),
                        tolerances=ts, driver_sha256=file_digest(__file__), refitted=False,
                        criterion='ICL <= min_ICL + tolerance * max(abs(min_ICL), 1); smallest K',
                        scope='all 5000 Qwen2-MATH response means; descriptive geometry',
                        normalization=False, correctness_used_for_selection=False)
        if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != identity:
            raise ValueError('Changed export protocol')
        save_json(output/'protocol.json', identity)
        table, exports = [], []
        for view in ['post', 'pre_final']:
            a, b = export_view(source, output, view, ts, base, records, args.cache_root)
            table.extend(a); exports.extend(b)
        frame = pd.DataFrame(table)
        frame.to_csv(output/'selected_layers.csv', index=False)
        wide = frame.pivot(index=['view','layer'], columns='tolerance', values='k').sort_index()
        wide.to_csv(output/'cluster_counts.csv')
        save_json(output/'delivery.json', dict(status='complete', exports=exports,
                  selected_layers=len(table),
                  selected_layers_sha256=file_digest(output/'selected_layers.csv')))
        print(wide.to_string(), flush=True)


if __name__ == '__main__':
    main()

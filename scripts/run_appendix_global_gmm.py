"""User-requested appendix control: one raw GMM across all Qwen MATH layers.

Uses the frozen reliability EM implementation and identical ICL selection,
without changing or interrupting that experiment. Reports MAP and nearest
assignments separately, including sample-weighted (not cluster-weighted) purity.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from hss.cluster.gmm_resident import ResidentDiagonalEM
from hss.data import CachedStates
from scripts.revision_common import freeze, sha, write_json, write_npz, nearest
from scripts.run_qwen_reliability import fit_candidate, load_model, select


def purity(labels, n_layers, n_per_layer, k):
    labels = np.asarray(labels, dtype=int)
    assert labels.shape == (n_layers * n_per_layer,)
    assert labels.min() >= 0 and labels.max() < k
    counts = np.zeros((k, n_layers), dtype=np.int64)
    np.add.at(counts, (labels, np.repeat(np.arange(n_layers), n_per_layer)), 1)
    sizes = counts.sum(1)
    per_cluster = np.divide(counts.max(1), sizes, out=np.zeros(k), where=sizes > 0)
    return dict(sample_weighted_purity=float(counts.max(1).sum() / sizes.sum()),
                occupied_clusters=int((sizes > 0).sum()), empty_clusters=int((sizes == 0).sum()),
                cluster_sizes=sizes.tolist(), layer_purity=per_cluster.tolist(),
                cluster_layer_counts=counts.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = dict(n_init=3, max_iter=2000, tol=1e-5, reg_covar=1e-6,
               k_values=list(range(2,81)), seed=42, chunk_size=8192,
               tolerances=[0.0, 0.02], primary_tolerance=0.02)
    data = CachedStates(args.cache)
    assert data.n_items() == 5000 and data.state_dim() == 3584 and data.layers() == list(range(29))
    assert data.info['model'][0] == 'Qwen/Qwen2-7B-Instruct'
    assert data.info['identity']['spec']['representation'] == 'mean'
    assert data.info['identity']['spec']['final_norm'] == 'post'
    sources = [Path(args.cache)/f'layer_{l}.npy' for l in range(29)]
    sources += [Path(args.cache)/'rows.parquet', Path(__file__),
                Path(__file__).resolve().parents[1]/'src/hss/cluster/gmm_resident.py',
                Path(__file__).with_name('run_qwen_reliability.py')]
    protocol = dict(config=cfg, shape=[145000,3584], cache=args.cache,
                    initialization='single-trial D-squared kmeans++ seeding plus hard moments',
                    scaling='none; numerical translation only',
                    selection='smallest converged K within relative ICL tolerance; BIC+2H',
                    sources={str(p):sha(p) for p in sources})
    freeze(root/'protocol.json', protocol)
    started = time.monotonic()
    def progress(**kwargs):
        write_json(root/'status.json', dict(status='running',pid=os.getpid(),
                    updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs))
    progress(phase='loading')
    x = np.concatenate([data.array(l) for l in range(29)])
    import torch
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < 16*1024**3:
        raise RuntimeError('Insufficient spare GPU memory; existing jobs must not be interrupted')
    with threadpool_limits(limits=2):
        engine = ResidentDiagonalEM(x, device='cuda:0', chunk_size=cfg['chunk_size'], reg_covar=cfg['reg_covar'])
        rows=[]
        for k in cfg['k_values']:
            row=fit_candidate(engine,root/'fits'/f'k{k:03d}',cfg,cfg['seed'],0,k,'pooled',progress)
            rows.append(row)
            write_json(root/'candidates.json',rows)
        results=[]
        for tolerance in cfg['tolerances']:
            selected=select(rows,tolerance,80)
            model=load_model(selected['path'])
            _,_,_,p=engine.evaluate(engine.parameters(model),update=False,probabilities=True)
            assignments={'posterior':p.argmax(1),'nearest':nearest(x,model.means_,4096)}
            report=dict(tolerance=tolerance,k=selected['k'],selected_path=selected['path'],
                        **{name:purity(a,29,5000,selected['k']) for name,a in assignments.items()})
            write_npz(root/f'assignments_t{tolerance}.npz',**assignments)
            results.append(report)
        summary=dict(results=results,seconds=time.monotonic()-started,
                     failed_k=[r['k'] for r in rows if not r['converged']],
                     best_at_upper_bound=select(rows,0,80)['k']==80,
                     completed_candidates=len(rows),protocol_sha256=sha(root/'protocol.json'))
        write_json(root/'summary.json',summary)
        write_json(root/'COMPLETE.json',dict(summary_sha256=sha(root/'summary.json'),
                                           utc=datetime.now(timezone.utc).isoformat()))
        write_json(root/'status.json',dict(status='complete',pid=os.getpid()))


if __name__ == '__main__':
    main()

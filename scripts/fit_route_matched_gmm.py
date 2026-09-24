"""Same-K diagonal GMM control at the frozen MFA K, on raw cached means."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from hss.cluster.gmm import _fit_gmm
from hss.experiments.artifacts import file_digest, save_json, save_npz
from prepare_route_study import diagnostics


def fit_layer(task):
    layer,k,cache,output=task; root=Path(output)/f'layer_{layer}';root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return json.loads((root/'complete.json').read_text())
    start=time.monotonic();x=np.load(Path(cache)/f'layer_{layer}.npy',mmap_mode='r');best=None;best_ll=-np.inf;audit=[]
    with threadpool_limits(2):
        for restart in range(3):
            seed=42+1009*layer+restart
            model=_fit_gmm(x,k,seed,covariance_type='diag',reg_covar=1e-6,n_init=1,max_iter=2000,tol=1e-5,adaptive_reg=False,backend='cpu',init_method='kmeans++')
            score=float(model.score_samples(x).mean())
            save_npz(root/f'restart_{restart}.npz',**model.state_arrays())
            save_json(root/f'restart_{restart}.json',model.config())
            audit.append(dict(restart=restart,seed=seed,converged=model.converged_,iterations=model.n_iter_,mean_loglik=score))
            if model.converged_ and score>best_ll:best,best_ll=model,score
        if best is None:raise RuntimeError(f'No converged GMM for layer {layer}; all checkpoints retained')
        p,d=diagnostics(best,x)
    save_npz(root/'model.npz',**best.state_arrays());save_json(root/'model.json',best.config())
    save_npz(root/'diagnostics.npz',probability=p.astype(np.float32),**d)
    result=dict(layer=layer,k=k,seconds=time.monotonic()-start,restarts=audit,converged=True,n_init_completed=3,
                model_sha256=file_digest(root/'model.npz'),raw=True,normalization=False,backend='cpu')
    save_json(root/'complete.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='/lambda/nfs/dami/hss/qwen-math-mfa-24h-20260920')
    p.add_argument('--output',required=True);p.add_argument('--layers',type=int,nargs='+',default=[14,28]);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();s=Path(a.study);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    exports=json.loads((s/'latest_exports.json').read_text());selection=json.loads((Path(exports['post'])/'selection.json').read_text())
    ks={r['layer']:r['selected']['k'] for r in selection};snapshot=json.loads((s/'snapshots/post.json').read_text())
    cache=str(Path('/home/ubuntu/hss-cache/data')/snapshot['key'])
    save_json(out/('request_'+ '_'.join(map(str,a.layers))+'.json'),dict(layers=a.layers,k={str(l):ks[l] for l in a.layers},driver_sha256=file_digest(Path(__file__)),snapshot=snapshot['key'],reason='Same-K covariance-family control; no correctness labels used; fixed selected MFA K, not GMM optimum'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(fit_layer,(l,ks[l],cache,str(out))) for l in a.layers]):
            print(json.dumps(f.result()),flush=True)

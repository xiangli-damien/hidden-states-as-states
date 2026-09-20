"""Read-only optimizer comparison on an existing full-dimensional data matrix."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from hss.cluster.mfa import fit_mfa
from hss.cluster.mfa_fast import BatchedMFAEM
from hss.experiments.fitting import load_fitted
from hss.experiments.artifacts import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',required=True);p.add_argument('--fit',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--optimizer-steps',type=int,default=100);p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args();X=np.load(a.states,mmap_mode='r');model,_=load_fitted(a.fit)
    model=replace(model,history_=[model.score(X)],converged_=False)
    out={'shape':list(X.shape),'k':model.n_clusters(),'rank':model.loadings_.shape[2],'benchmarks':[]}
    with threadpool_limits(limits=2):
        start=time.perf_counter()
        reference=fit_mfa(X,model.n_clusters(),rank=model.loadings_.shape[2],n_init=1,
                          max_iter=a.steps+1,tol=0,initial_model=model,
                          backend='cpu' if a.cpu_only else 'gpu')
        out['reference_seconds']=time.perf_counter()-start
        for tile in ([8] if a.cpu_only else [4,8,16]):
            torch.cuda.reset_peak_memory_stats() if not a.cpu_only else None
            start=time.perf_counter();engine=BatchedMFAEM(X,device='cpu' if a.cpu_only else 'cuda:0',component_batch=tile)
            result,audit=engine.fit(model,max_steps=a.steps,tol=0)
            seconds=time.perf_counter()-start
            error=max(np.linalg.norm(result.state_arrays()[k]-reference.state_arrays()[k])/max(np.linalg.norm(reference.state_arrays()[k]),1e-20) for k in result.state_arrays())
            item={'tile':tile,'seconds':seconds,'speedup':out['reference_seconds']/seconds,
                  'max_parameter_relative_error':float(error),
                  'likelihood_difference':result.history_[-1]-reference.history_[-1],
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3 if not a.cpu_only else None}
            assert error<1e-7 and abs(item['likelihood_difference'])<1e-6,item
            out['benchmarks'].append(item);print(json.dumps(item),flush=True)
            del engine,result
            if not a.cpu_only:torch.cuda.empty_cache()
        if not a.cpu_only:
            initial=fit_mfa(X,model.n_clusters(),rank=model.loadings_.shape[2],seed=42,n_init=1,max_iter=1,tol=0,init_method='svd')
            out['optimizers']=[]
            for method in ['none','squarem']:
                start=time.perf_counter();engine=BatchedMFAEM(X)
                result,audit=engine.fit(initial,max_steps=a.optimizer_steps,tol=1e-5,accelerator=method)
                item={'method':method,'seconds':time.perf_counter()-start,'log_likelihood':result.history_[-1],
                      'converged':result.converged_,**audit}
                out['optimizers'].append(item);print(json.dumps(item),flush=True)
                del engine,result;torch.cuda.empty_cache()
    save_json(a.output,out)


if __name__=='__main__':main()

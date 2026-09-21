"""Score new local-ID depth routes using frozen unsupervised detectors."""
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from hss.route.inference import score_new_routes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True);p.add_argument('--map',required=True)
    p.add_argument('--states',required=True,help='N x L .npy local IDs from original fitted map')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--methods',nargs='+')
    a=p.parse_args()
    try:
        import torch
        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    with threadpool_limits(a.threads):
        pd.DataFrame(score_new_routes(a.benchmark,a.map,np.load(a.states),a.device,a.methods)).to_parquet(a.output,index=False)

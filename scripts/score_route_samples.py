"""Score new local-ID depth routes using frozen unsupervised detectors."""
import argparse
import numpy as np
import pandas as pd
from hss.route.inference import score_new_routes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True);p.add_argument('--map',required=True)
    p.add_argument('--states',required=True,help='N x L .npy local IDs from original fitted map')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    pd.DataFrame(score_new_routes(a.benchmark,a.map,np.load(a.states),a.device)).to_parquet(a.output,index=False)

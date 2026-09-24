"""One actual prompt-last vector encoded with a frozen nearest-GMM map."""
import numpy as np
from revision_common import nearest


def conditions():
    return ['identity','global_mean','centroid','shared8','local8','wrong8_42','wrong8_137','wrong8_271']


def apply(x, decoder, name):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)!=1 or not np.isfinite(x).all():
        raise ValueError('Require one finite prompt-last vector')
    if name not in conditions():raise ValueError(name)
    s=int(nearest(x,decoder['centers'])[0]);mu=decoder['centers'][s].astype(np.float64)
    if name=='identity':z=x.copy()
    elif name=='global_mean':z=decoder['train_mean'][None].astype(np.float64).copy()
    elif name=='centroid':z=mu[None].copy()
    else:
        if name=='shared8':b=decoder['shared_basis']
        elif name=='local8':b=decoder['local_basis'][s,:8]
        else:b=decoder['local_basis'][decoder['permutation_'+name.rsplit('_',1)[1]][s],:8]
        b=b.astype(np.float64);z=mu+(x-mu)@b.T@b
    return z,{'region':s,'distance':float(np.linalg.norm(x-mu))}

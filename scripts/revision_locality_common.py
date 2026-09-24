"""Explicit common-anchor decoders and finite-amplitude geometric controls.

This module is numpy-only and shared by fitting, GPU evaluation and auditing.
It never fits on the vectors passed to an intervention.
"""
import hashlib
import numpy as np
from revision_common import nearest, reconstruct


def stable_seed(*items):
    return int(hashlib.sha256('/'.join(map(str,items)).encode()).hexdigest()[:8],16)


def derangement(k,seed):
    if k<2:raise ValueError('Wrong-region control needs at least two regions')
    rng=np.random.default_rng(seed)
    while True:
        p=rng.permutation(k)
        if np.all(p!=np.arange(k)):return p


def project(residual,bases,assignment):
    out=np.empty_like(residual,dtype=np.float64)
    if bases.ndim==2:return (residual@bases.T)@bases
    for k in np.unique(assignment):
        ix=assignment==k;b=bases[k]
        out[ix]=(residual[ix]@b.T)@b
    return out


def radial_control(x,delta,seed):
    """Match each token's ||delta||, x.dot(delta), and final norm in float64."""
    rng=np.random.default_rng(seed)
    energy=np.square(x).sum(1,keepdims=True)
    parallel=x*(np.sum(x*delta,1,keepdims=True)/np.maximum(energy,1e-300))
    tangent=delta-parallel
    random=rng.standard_normal(x.shape)
    random-=x*(np.sum(x*random,1,keepdims=True)/np.maximum(energy,1e-300))
    norms=np.linalg.norm(random,axis=1,keepdims=True)
    if np.any(norms<1e-20):raise FloatingPointError('Degenerate random tangent')
    return parallel+random/norms*np.linalg.norm(tangent,axis=1,keepdims=True)


def gram_control(delta,seed):
    """Uniform random orientation of the entire error row space, same Gram.

    Equivalent in distribution to a common Haar orthogonal transform of all
    token errors, without allocating a d-by-d matrix. This does not also match
    each activation's radial component.
    """
    n,d=delta.shape
    if n>d:raise ValueError('This thin-frame control expects window<=hidden dimension')
    eigen,vectors=np.linalg.eigh(delta@delta.T)
    q,r=np.linalg.qr(np.random.default_rng(seed).standard_normal((d,n)),mode='reduced')
    q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
    return (vectors*np.sqrt(np.maximum(eigen,0))[None,:])@q.T


def conditions(cfg,full):
    out=[{'method':'identity'},{'method':'centroid'},{'method':'empirical_centroid'},
         {'method':'global','rank':8},{'method':'empirical_local','rank':8},
         {'method':'empirical_shared','rank':8}]
    for rank in cfg['ranks'] if full else [8]:
        out.extend([{'method':'local','rank':rank},{'method':'shared','rank':rank}])
    if full:
        out.extend({'method':'shared','rank':r} for r in cfg['shared_ranks'] if r not in cfg['ranks'])
    for seed in cfg['seeds']:
        out.extend([{'method':'wrong_local','rank':8,'seed':seed},
                    {'method':'centroid_radial_random','seed':seed}])
        if full:
            out.extend([{'method':'centroid_random','seed':seed},
                        {'method':'centroid_gram_random','seed':seed}])
    if full:
        for alpha in cfg['alphas']:
            out.extend([{'method':'remove_local','rank':8,'alpha':alpha},
                        {'method':'remove_complement','rank':8,'alpha':alpha},
                        {'method':'complement_energy_matched','rank':8,'alpha':alpha}])
            out.extend({'method':'remove_local_radial_random','rank':8,'alpha':alpha,'seed':seed}
                       for seed in cfg['seeds'])
    return out


def key(condition):
    return '_'.join(str(condition[k]) for k in ('method','rank','alpha','seed') if k in condition)


def replacement(x,decoder,condition,context):
    x=np.asarray(x,dtype=np.float64);name=condition['method'];rank=condition.get('rank',8)
    centers=np.asarray(decoder['centers'],dtype=np.float64)
    assignment=nearest(x,centers);mu=centers[assignment];residual=x-mu
    local=project(residual,np.asarray(decoder['local_basis'][:,:rank],dtype=np.float64),assignment)
    seed=stable_seed(context,condition.get('seed',42),name)
    # Alpha is deliberately excluded from seeds: each dose uses the same direction.
    if name=='identity':z=x.copy()
    elif name=='centroid':z=mu
    elif name=='empirical_centroid':z=np.asarray(decoder['local_empirical_centers'],dtype=np.float64)[assignment]
    elif name=='global':z=reconstruct(x.astype(np.float32),decoder,f'global_pca_{rank}').astype(np.float64)
    elif name=='local':z=mu+local
    elif name=='shared':z=mu+project(residual,decoder['shared_basis'][:rank].astype(np.float64),assignment)
    elif name=='wrong_local':
        permutation=decoder[f'permutation_{condition["seed"]}']
        z=mu+project(residual,decoder['local_basis'][permutation,:rank].astype(np.float64),assignment)
    elif name in ('empirical_local','empirical_shared'):
        anchor=decoder['local_empirical_centers'].astype(np.float64)[assignment]
        basis=(decoder['local_empirical_basis'][:,:rank] if name=='empirical_local' else decoder['shared_empirical_basis'][:rank]).astype(np.float64)
        z=anchor+project(x-anchor,basis,assignment)
    elif name.startswith('centroid_'):
        delta=mu-x
        if name=='centroid_radial_random':change=radial_control(x,delta,seed)
        elif name=='centroid_gram_random':change=gram_control(delta,seed)
        elif name=='centroid_random':
            rnd=np.random.default_rng(seed).standard_normal(x.shape)
            change=rnd/np.linalg.norm(rnd,axis=1,keepdims=True)*np.linalg.norm(delta,axis=1,keepdims=True)
        else:raise ValueError(name)
        z=x+change
    else:
        alpha=condition['alpha'];other=residual-local
        if name=='remove_local':change=-alpha*local
        elif name=='remove_complement':change=-alpha*other
        elif name=='complement_energy_matched':
            norm=np.linalg.norm(other,axis=1,keepdims=True)
            if np.any((norm<1e-20)&(np.linalg.norm(local,axis=1,keepdims=True)>1e-20)):
                raise ValueError('No complement direction for energy-matched control')
            change=-alpha*other/np.maximum(norm,1e-300)*np.linalg.norm(local,axis=1,keepdims=True)
        elif name=='remove_local_radial_random':change=radial_control(x,-alpha*local,seed)
        else:raise ValueError(name)
        z=x+change
    return z,assignment


def geometry_metrics(x,z,centers):
    x=np.asarray(x,dtype=np.float64);z=np.asarray(z,dtype=np.float64);delta=z-x
    before=nearest(x,centers);after=nearest(z,centers)
    return {'token_delta_energy':np.square(delta).sum(1).tolist(),
        'token_radial_dot':np.sum(x*delta,1).tolist(),
        'token_original_norm':np.linalg.norm(x,axis=1).tolist(),
        'token_final_norm':np.linalg.norm(z,axis=1).tolist(),
        'error_gram':(delta@delta.T).tolist(),
        'state_before':before.tolist(),'state_after':after.tolist(),
        'state_retained_fraction':float(np.mean(before==after))}

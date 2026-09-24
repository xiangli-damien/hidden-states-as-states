"""Finite real-task steering; raw token vectors and fixed training-only decoders."""
import hashlib
import numpy as np
from revision_common import nearest
from revision_locality_common import radial_control, stable_seed


def ordered(ids, namespace):
    return sorted(ids, key=lambda x: (hashlib.sha256((namespace+'/'+x).encode()).hexdigest(), x))


def candidates(cfg, c2=True):
    out=[{'name':f'c1_{a}', 'family':'c1', 'alpha':a} for a in cfg['c1_alphas']]
    if c2: out += [{'name':f'c2_{a}', 'family':'c2', 'alpha':a} for a in cfg['c2_alphas']]
    return out


def target_regions(x, centers, support, cfg, family):
    x=np.asarray(x,dtype=np.float64); c=np.asarray(centers,dtype=np.float64)
    distance=np.square(x[:,None,:]-c[None,:,:]).sum(-1)
    ranked=np.argsort(distance,axis=1,kind='stable')
    current=ranked[:,0]; target=current.copy(); mask=np.ones(len(x),dtype=bool)
    if family=='c2':
        count=np.asarray(support['question_count']); p=np.asarray(support['shrunk_correctness'])
        mask[:]=False
        for i,choices in enumerate(ranked[:,:cfg['nearest_candidates']]):
            good=[k for k in choices if count[k]>=cfg['support_min_questions']
                  and p[k]-p[current[i]]>=cfg['support_min_gain']]
            if good: target[i]=good[0]; mask[i]=True
    elif family!='c1': raise ValueError(family)
    return current,target,mask


def transform(x, decoder, support, cfg, condition, sid):
    x=np.asarray(x,dtype=np.float64)
    centers=np.asarray(decoder['centers'],dtype=np.float64)
    if condition['family'] in ('baseline','identity'):
        current=nearest(x,centers)
        return x.copy(), {'current':current.tolist(),'target':current.tolist(),'mask':[False]*len(x)}
    family=condition.get('target_family',condition['family'])
    current,target,mask=target_regions(x,centers,support,cfg,family)
    local=decoder['local_basis'][:,:cfg['rank']].astype(np.float64)
    shared=decoder['shared_basis'][:cfg['rank']].astype(np.float64)
    residual=x-centers[target]; z=x.copy()
    for k in np.unique(target[mask]):
        ix=mask & (target==k)
        b=shared if condition['family']=='shared' else local[k]
        z[ix]=centers[k]+(residual[ix]@b.T)@b
    delta=condition['alpha']*(z-x)
    if condition['family']=='random':
        delta=radial_control(x,delta,stable_seed('completion-random',sid,condition['seed']))
        delta[~mask]=0
    return x+delta, {'current':current.tolist(),'target':target.tolist(),'mask':mask.tolist()}


def test_conditions(selected, cfg):
    a=selected['alpha']; family=selected['family']
    out=[{'name':'baseline','family':'baseline'},selected,
         {'name':'shared8','family':'shared','target_family':family,'alpha':a}]
    out += [{'name':f'random_{seed}','family':'random','target_family':family,'alpha':a,'seed':seed}
            for seed in cfg['random_seeds']]
    if family=='c2': out.append({'name':'current_local8','family':'c1','alpha':a})
    return out


def choose_candidate(rows, cfg):
    """Consumes validation aggregate records only; never target/test records."""
    if not rows or any(r['split']!='validation' or r['n']!=cfg['validation_questions'] for r in rows):
        raise ValueError('Only complete validation aggregates may select a policy')
    eligible=[r for r in rows if r['net_correct']>0 and r['parse_failure_increase']<=2]
    if not eligible:return None
    return min(eligible,key=lambda r:(-r['net_correct'],r['mean_actual_energy'],r['condition']['alpha'],
                                      r['condition']['family']!='c1',r['condition']['name']))['condition']

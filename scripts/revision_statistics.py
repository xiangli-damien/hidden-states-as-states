"""Question-level paired AUROC bootstrap with exact handling of tied scores."""
import numpy as np


def weighted_auc(y,score,counts):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    counts=np.atleast_2d(counts).astype(np.float64)
    if not np.isin(y,[0,1]).all() or len(score)!=len(y) or counts.shape[1]!=len(y):
        raise ValueError('Invalid binary question arrays')
    order=np.argsort(score,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
    positive=np.add.reduceat(counts[:,order]*y[order],starts,axis=1)
    negative=np.add.reduceat(counts[:,order]*(1-y[order]),starts,axis=1)
    wins=(positive*(np.cumsum(negative,axis=1)-.5*negative)).sum(1)
    denominator=positive.sum(1)*negative.sum(1)
    return np.divide(wins,denominator,out=np.full(len(counts),np.nan),where=denominator>0)


def paired_auc_ci(y,score,baseline,boot=2000,seed=42):
    y=np.asarray(y,dtype=int)
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=boot)
    delta=weighted_auc(y,score,counts)-weighted_auc(y,baseline,counts)
    a=float(weighted_auc(y,score,np.ones(len(y)))[0])
    b=float(weighted_auc(y,baseline,np.ones(len(y)))[0])
    return {'n':len(y),'auroc':a,'baseline_auroc':b,'delta':a-b,
            'ci95':np.nanquantile(delta,[.025,.975]).tolist(),
            'bootstrap':'paired question bootstrap, pointwise; exact score ties',
            'bootstrap_draws':boot,'seed':seed}

"""Evaluation-only conditional label and class-occupancy nulls.

Labels enter this module to TEST structure, never to fit a detector.
Permutation p-values condition on the already fitted, whole-data maps.
"""
import numpy as np
from functools import lru_cache
from scipy.special import xlogy
from scipy.stats import hypergeom
from .counts import js_rows


@lru_cache(maxsize=300000)
def expected_js_cell(n, n1, m):
    """Exact label-permutation JS expectation for one destination cell."""
    x=np.arange(max(0,m-(n-n1)),min(m,n1)+1)
    p=x/n1; q=(m-x)/(n-n1); middle=(p+q)/2
    term=(xlogy(p,p)-xlogy(p,middle)+xlogy(q,q)-xlogy(q,middle))/(2*np.log(2))
    return float(hypergeom.pmf(x,n,m,n1) @ term)


def codes(*columns):
    return np.unique(np.column_stack(columns), axis=0, return_inverse=True)[1]


class ConditionalRouting:
    """Equal-class JS, weighted by common pooled group prevalence.

    Groups are origin cluster x nuisance stratum. Only groups with >=2
    examples in EACH class enter; this target population is reported explicitly.
    """
    def __init__(self, origin, destination, y, nuisance, min_class=2):
        self.y = np.asarray(y, int)
        if not np.isin(self.y, [0, 1]).all(): raise ValueError('Binary evaluation labels required')
        self.g = codes(origin, nuisance)
        self.b = np.unique(destination, return_inverse=True)[1]
        self.k = int(self.b.max()) + 1; self.ng = int(self.g.max()) + 1
        self.total = np.bincount(self.g * self.k + self.b, minlength=self.ng * self.k).reshape(self.ng, self.k)
        self.one = np.bincount(self.g * self.k + self.b, weights=self.y, minlength=self.ng * self.k).reshape(self.ng, self.k).astype(int)
        self.n = self.total.sum(1); self.n1 = self.one.sum(1); self.n0 = self.n - self.n1
        self.eligible = (self.n1 >= min_class) & (self.n0 >= min_class)
        self.support = int(self.n[self.eligible].sum())
        self.weights = self.n[self.eligible] / max(1, self.support)

    def statistic(self, destination=None):
        if self.support == 0: return float('nan')
        if destination is None: total, one = self.total, self.one
        else:
            b = np.asarray(destination, int)
            total = np.bincount(self.g * self.k + b, minlength=self.ng * self.k).reshape(self.ng, self.k)
            one = np.bincount(self.g * self.k + b, weights=self.y, minlength=self.ng * self.k).reshape(self.ng, self.k)
        e=self.eligible
        return float(self.weights @ js_rows(one[e]/self.n1[e,None], (total[e]-one[e])/self.n0[e,None]))

    def test(self, rng, permutations=4999):
        observed = self.statistic()
        null = np.zeros(permutations)
        for g, weight in zip(np.flatnonzero(self.eligible), self.weights):
            counts = self.total[g]; counts = counts[counts > 0]
            draw = rng.multivariate_hypergeometric(counts, int(self.n1[g]), size=permutations)
            null += weight * js_rows(draw/self.n1[g], (counts[None,:]-draw)/self.n0[g])
        return dict(js_bits=observed, null_mean_bits=float(null.mean()), excess_bits=float(observed-null.mean()),
                    null_95=np.quantile(null,[.025,.975]).tolist(),
                    p=float((1+(null>=observed-1e-15).sum())/(permutations+1)) if self.support else 1.,
                    support_n=self.support, support_fraction=self.support/len(self.y), eligible_groups=int(self.eligible.sum()),
                    all_groups=self.ng, permutations=permutations)

    def exact_null_mean(self):
        if not self.support:return float('nan')
        return float(sum(weight*sum(expected_js_cell(int(self.n[g]),int(self.n1[g]),int(m))
                                    for m in self.total[g] if m)
                         for g,weight in zip(np.flatnonzero(self.eligible),self.weights)))


def shuffle_within(values, groups, rng):
    """Independently shuffle columns within groups, preserving all marginals."""
    a=np.asarray(values); one=a.ndim==1
    if one: a=a[:,None]
    g=np.unique(groups,return_inverse=True)[1]
    ordered=np.argsort(g,kind='stable')
    source=np.argsort(g[:,None]*2. + rng.random(a.shape),axis=0)
    out=np.empty_like(a); out[ordered]=np.take_along_axis(a,source,axis=0)
    return out[:,0] if one else out


def occupancy_null(states, y, nuisance, rng, permutations=499, bias_correct=False):
    """Keep every (class,nuisance,layer) occupancy; destroy cross-layer pairing."""
    z=np.column_stack([np.unique(col,return_inverse=True)[1] for col in states.T])
    tests=[ConditionalRouting(z[:,l],z[:,l+1],y,nuisance) for l in range(z.shape[1]-1)]
    observed=np.array([t.statistic() for t in tests]); null=np.empty((permutations,len(tests)))
    bias=np.array([t.exact_null_mean() for t in tests]) if bias_correct else np.zeros(len(tests))
    null_bias=np.zeros_like(null)
    groups=codes(y,nuisance)
    for b in range(permutations):
        shuffled=shuffle_within(z,groups,rng)
        # BOTH endpoints are shuffled. Rebuild origins; class counts and common
        # support remain fixed under this class/nuisance-preserving shuffle.
        others=[ConditionalRouting(shuffled[:,l],shuffled[:,l+1],y,nuisance) for l in range(z.shape[1]-1)]
        null[b]=[t.statistic() for t in others]
        if bias_correct:null_bias[b]=[t.exact_null_mean() for t in others]
    obs=float(np.nanmean(observed)); means=np.nanmean(null,axis=1)
    result=dict(observed_mean_js_bits=obs,null_mean_bits=float(means.mean()),
                excess_bits=float(obs-means.mean()),null_95=np.quantile(means,[.025,.975]).tolist(),
                p_greater=float((1+(means>=obs).sum())/(permutations+1)),
                layer_observed=observed.tolist(),layer_null_mean=np.nanmean(null,axis=0).tolist(),
                permutations=permutations,scope='Class/nuisance occupancy preserved; original edge counts not preserved; evaluation-only null')
    if bias_correct:
        corrected=np.nanmean(observed-bias); corrected_null=np.nanmean(null-null_bias,axis=1)
        result['bias_corrected']=dict(observed_bits=float(corrected),null_mean_bits=float(corrected_null.mean()),
            excess_bits=float(corrected-corrected_null.mean()),null_95=np.quantile(corrected_null,[.025,.975]).tolist(),
            p_greater=float((1+(corrected_null>=corrected-1e-15).sum())/(permutations+1)),
            correction='Exact conditional label-permutation expectation, accounting for occupied-cell sparsity of EACH shuffled map')
    return result


def shuffle_suffixes(states, folds, rng):
    """Swap whole suffixes at a shared middle state within a count-fitting fold.

    Every layer's edge counts AND fold membership are exactly preserved.
    Longer prefix/suffix associations are randomized without using correctness.
    """
    out=np.asarray(states).copy()
    for layer in range(1,out.shape[1]-1):
        order=shuffle_within(np.arange(len(out)),codes(out[:,layer],folds),rng)
        out[:,layer+1:]=out[order,layer+1:]
    return out

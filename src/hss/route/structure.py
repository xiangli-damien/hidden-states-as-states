"""Unsupervised categorical depth-route models. No correctness inputs.

State vocabularies are layer specific. Fit vocabularies on training routes;
reserve an explicit unknown symbol at every layer for new observations.
"""
from collections import Counter, defaultdict
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist


class RouteEncoder:
    def fit(self, z):
        self.vocab = [np.unique(a) for a in np.asarray(z).T]
        self.sizes = np.array([len(v) + 1 for v in self.vocab])
        self.offsets = np.r_[0, np.cumsum(self.sizes)[:-1]]
        return self

    def transform(self, z):
        z = np.asarray(z)
        if z.shape[1] != len(self.vocab):
            raise ValueError('Incorrect route depth')
        out = np.empty_like(z, dtype=np.int64)
        for l, v in enumerate(self.vocab):
            lookup = {int(a): i for i, a in enumerate(v)}
            out[:, l] = [lookup.get(int(a), len(v)) for a in z[:, l]]
        return out

    def onehot(self, z):
        out = np.zeros((len(z), self.sizes.sum()), dtype=np.float32)
        out[np.arange(len(z))[:, None], z + self.offsets] = 1.
        return out


class BackoffMarkov:
    """Interpolated order-m conditionals, with strength-count backoff."""
    def __init__(self, sizes, order=2, strength=10., alpha=1.):
        self.sizes, self.order, self.strength, self.alpha = sizes, order, strength, alpha

    def fit(self, z):
        self.marginals = []
        self.counts = {}
        for l, k in enumerate(self.sizes):
            c = np.bincount(z[:, l], minlength=k) + self.alpha
            self.marginals.append(c / c.sum())
            for depth in range(1, min(l, self.order) + 1):
                table = defaultdict(Counter)
                for row in z:
                    table[tuple(row[l-depth:l])][int(row[l])] += 1
                self.counts[l, depth] = dict(table)
        return self

    def losses(self, z):
        p = np.column_stack([a[z[:, l]] for l, a in enumerate(self.marginals)])
        for (l, depth), table in self.counts.items():
            for i, row in enumerate(z):
                c = table.get(tuple(row[l-depth:l]), {})
                p[i, l] = (c.get(int(row[l]), 0) + self.strength * p[i, l]) / (sum(c.values()) + self.strength)
        return -np.log(np.maximum(p, 1e-30))


class MixtureMarkov:
    """EM mixture of layer-specific first-order chains; MAP-smoothed EM."""
    def __init__(self, sizes, components=4, seed=0, max_iter=150, alpha=.1):
        self.sizes, self.components, self.seed = sizes, components, seed
        self.max_iter, self.alpha = max_iter, alpha

    def _mstep(self, z, r):
        self.weights = (r.sum(0) + self.alpha) / (r.sum() + self.components*self.alpha)
        self.initial = np.stack([np.bincount(z[:, 0], weights=r[:, j], minlength=self.sizes[0]) + self.alpha for j in range(self.components)])
        self.initial /= self.initial.sum(1, keepdims=True)
        self.transitions = []
        for l in range(1, z.shape[1]):
            ka, kb = self.sizes[l-1:l+1]
            c = np.stack([np.bincount(z[:, l-1]*kb+z[:, l], weights=r[:, j], minlength=ka*kb).reshape(ka, kb) + self.alpha for j in range(self.components)])
            self.transitions.append(c / c.sum(2, keepdims=True))

    def component_logprob(self, z):
        out = np.log(self.initial[:, z[:, 0]].T)
        for l, t in enumerate(self.transitions):
            out += np.log(t[:, z[:, l], z[:, l+1]].T)
        return out

    def fit(self, z):
        rng = np.random.default_rng(self.seed)
        r = rng.dirichlet(np.ones(self.components)*.3, len(z))
        self.trace = []
        self.converged = False
        for _ in range(self.max_iter):
            self._mstep(z, r)
            lp = self.component_logprob(z) + np.log(self.weights)
            ll = logsumexp(lp, axis=1)
            self.trace.append(float(ll.mean()))
            r = np.exp(lp-ll[:, None])
            if len(self.trace) > 2 and abs(self.trace[-1]-self.trace[-2]) < 1e-5:
                self.converged = True
                break
        return self

    def losses(self, z):
        logw = np.broadcast_to(np.log(self.weights), (len(z), self.components)).copy()
        losses = []
        probs = [self.initial[:, z[:, 0]].T]
        probs += [t[:, z[:, l], z[:, l+1]].T for l, t in enumerate(self.transitions)]
        for p in probs:
            lp = logw + np.log(p)
            norm = logsumexp(lp, axis=1)
            losses.append(-norm)
            logw = lp - norm[:, None]
        return np.column_stack(losses)


class LayerHMM:
    """Latent chain with layer-specific categorical emissions/transitions."""
    def __init__(self, sizes, components=4, seed=0, max_iter=100, alpha=.1):
        self.sizes, self.components, self.seed = sizes, components, seed
        self.max_iter, self.alpha = max_iter, alpha

    def _forward(self, z):
        em = [e[:, z[:, l]].T for l, e in enumerate(self.emissions)]
        f = []; scales = []
        a = em[0] * self.initial
        c = a.sum(1); f.append(a/c[:, None]); scales.append(c)
        for l in range(1, len(em)):
            a = (f[-1] @ self.transitions[l-1]) * em[l]
            c = a.sum(1); f.append(a/c[:, None]); scales.append(c)
        return em, f, scales

    def fit(self, z, warm_start=False):
        rng = np.random.default_rng(self.seed); m = self.components; depth = z.shape[1]
        if not warm_start:
            self.initial = np.ones(m)/m
            self.transitions = np.stack([rng.dirichlet(np.ones(m), m) for _ in range(depth-1)])
            self.emissions = [rng.dirichlet(np.ones(k), m) for k in self.sizes]
        self.trace = []; self.converged = False
        for _ in range(self.max_iter):
            em, f, scales = self._forward(z)
            ll = float(np.log(scales).sum(0).mean()); self.trace.append(ll)
            if len(self.trace)>2 and abs(self.trace[-1]-self.trace[-2])<1e-5:
                self.converged = True
                break
            b = [None]*depth; b[-1] = np.ones_like(f[-1])
            for l in range(depth-2, -1, -1):
                b[l] = ((em[l+1]*b[l+1]) @ self.transitions[l].T) / scales[l+1][:, None]
            gamma = [a*c/(a*c).sum(1, keepdims=True) for a,c in zip(f,b)]
            self.initial = gamma[0].sum(0)+self.alpha; self.initial /= self.initial.sum()
            for l in range(depth-1):
                t = np.einsum('ni,ij,nj->ij', f[l], self.transitions[l], em[l+1]*b[l+1]/scales[l+1][:, None]) + self.alpha
                self.transitions[l] = t/t.sum(1, keepdims=True)
            self.emissions = []
            for l, k in enumerate(self.sizes):
                c = np.stack([np.bincount(z[:,l], weights=gamma[l][:,j], minlength=k)+self.alpha for j in range(m)])
                self.emissions.append(c/c.sum(1, keepdims=True))
        return self

    def losses(self, z):
        return -np.log(np.column_stack(self._forward(z)[2]))


def aggregate_losses(losses):
    """Frozen scoring directions; no optimization against correctness."""
    return dict(mean=losses.mean(1), top4=np.sort(losses, axis=1)[:, -min(4, losses.shape[1]):].mean(1))


class StableRouteLOF:
    """Novelty LOF with deterministic training-index tie breaks.

    Euclidean distance on layer-wise one-hot equals sqrt(2 * Hamming count).
    Categorical paths have many exact ties; resolving them by training row
    index makes a single-query score agree with arbitrary query batches.
    """
    input_encoding = 'local_state_ids'

    def __init__(self, n_neighbors=35):
        self.n_neighbors=n_neighbors

    def _neighbors(self, z, exclude_self=False):
        all_idx=[];all_d=[]
        for start in range(0,len(z),256):
            d=np.sqrt(2*z.shape[1]*cdist(z[start:start+256],self.train,metric='hamming'))
            if exclude_self:d[np.arange(len(d)),start+np.arange(len(d))]=np.inf
            idx=np.argsort(d,axis=1,kind='stable')[:,:self.k]
            all_idx.append(idx);all_d.append(np.take_along_axis(d,idx,axis=1))
        return np.concatenate(all_idx),np.concatenate(all_d)

    def fit(self,z):
        self.train=np.asarray(z).copy();self.k=min(self.n_neighbors,len(z)-1)
        if self.k<1:raise ValueError('LOF needs at least two training paths')
        idx,d=self._neighbors(self.train,exclude_self=True)
        self.k_distance=d[:,-1]
        self.lrd=1/(np.maximum(d,self.k_distance[idx]).mean(1)+1e-10)
        return self

    def score_samples(self,z):
        idx,d=self._neighbors(np.asarray(z))
        lrd=1/(np.maximum(d,self.k_distance[idx]).mean(1)+1e-10)
        return -(self.lrd[idx]/lrd[:,None]).mean(1)

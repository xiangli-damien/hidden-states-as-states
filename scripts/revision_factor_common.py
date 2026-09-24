"""Numpy-only affine PCA/FA/MFA decoders; no fitting or feature normalization.

FA reconstructs E[mu + W z | x, component], not the noisy observation x and
not an orthogonal projection. Hard/soft MFA share every fitted parameter.
"""
import numpy as np
from revision_common import nearest


def logsumexp(x, axis=-1, keepdims=False):
    maximum = np.max(x, axis=axis, keepdims=True)
    value = maximum + np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))
    return value if keepdims else np.squeeze(value, axis=axis)


class FactorDecoder:
    def __init__(self, weights, means, loadings, noise):
        self.weights, self.means, self.loadings, self.noise = (
            np.array(a, dtype=np.float64, copy=True) for a in (weights, means, loadings, noise))
        if self.loadings.ndim != 3:
            raise ValueError('Loadings must have shape K,D,rank')
        self.k, self.d, self.rank = self.loadings.shape
        if (self.means.shape != (self.k,self.d) or self.noise.shape != self.means.shape
                or self.weights.shape != (self.k,) or not self.k or not self.d):
            raise ValueError('Inconsistent factor model dimensions')
        if (any(not np.isfinite(a).all() for a in (self.weights,self.means,self.loadings,self.noise))
                or np.any(self.weights <= 0) or np.any(self.noise <= 0)
                or not np.isclose(self.weights.sum(),1,rtol=0,atol=1e-10)):
            raise ValueError('Finite parameters, positive noise/weights and normalized weights required')
        self.posterior_maps = []
        self.cholesky = []
        self.logdet = []
        self.column_bases = []
        for w, psi in zip(self.loadings, self.noise):
            weighted = w/psi[:,None]
            matrix = np.eye(self.rank)+w.T@weighted
            chol = np.linalg.cholesky(matrix)
            # z = (x-mu) B; B = Psi^-1 W (I+W'Psi^-1 W)^-1.
            mapping = np.linalg.solve(chol.T,np.linalg.solve(chol,weighted.T)).T
            self.posterior_maps.append(mapping)
            self.cholesky.append(chol)
            self.logdet.append(np.log(psi).sum()+2*np.log(chol.diagonal()).sum())
            if self.rank:
                u,s,_ = np.linalg.svd(w,full_matrices=False)
                tolerance = max(w.shape)*np.finfo(float).eps*s[0]
                self.column_bases.append(u[:,s>tolerance].T)
            else:
                self.column_bases.append(np.zeros((0,self.d)))
        self.posterior_maps = np.stack(self.posterior_maps)
        self.logdet = np.array(self.logdet)
        for a in (self.weights,self.means,self.loadings,self.noise,self.posterior_maps):
            a.setflags(write=False)

    def _x(self, x):
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d or not np.isfinite(x).all():
            raise ValueError('Expected finite N,D observations')
        return x

    def component_log_density(self, x):
        x=self._x(x); out=np.empty((len(x),self.k))
        for k in range(self.k):
            residual=x-self.means[k]
            low=(residual/self.noise[k])@self.loadings[k]
            whitened=np.linalg.solve(self.cholesky[k],low.T)
            quad=np.square(residual).dot(1/self.noise[k])-np.square(whitened).sum(0)
            out[:,k]=-.5*(self.d*np.log(2*np.pi)+self.logdet[k]+quad)
        return out

    def posterior(self, x):
        joint=self.component_log_density(x)+np.log(self.weights)[None,:]
        density=logsumexp(joint,axis=1)
        return np.exp(joint-density[:,None]),density

    def assigned(self, x, assignment, orthogonal=False):
        x=self._x(x); assignment=np.asarray(assignment)
        if (assignment.shape != (len(x),) or not np.issubdtype(assignment.dtype,np.integer)
                or np.any(assignment<0) or np.any(assignment>=self.k)):
            raise ValueError('Invalid component IDs')
        out=np.empty_like(x)
        for k in np.unique(assignment):
            ix=assignment==k; residual=x[ix]-self.means[k]
            if orthogonal:
                basis=self.column_bases[k]
                out[ix]=self.means[k]+(residual@basis.T)@basis
            else:
                out[ix]=self.means[k]+(residual@self.posterior_maps[k])@self.loadings[k].T
        return out

    def reconstruct(self, x, soft=False):
        x=self._x(x); probabilities,density=self.posterior(x)
        assignment=probabilities.argmax(1)
        if not soft:
            out=self.assigned(x,assignment)
        else:
            out=np.zeros_like(x)
            # No N,K,D allocation; every component contributes, no top-k pruning.
            for k in range(self.k):
                local=self.means[k]+((x-self.means[k])@self.posterior_maps[k])@self.loadings[k].T
                out+=probabilities[:,k,None]*local
        return out,{'component':assignment.tolist(),
                    'posterior_max':probabilities.max(1).tolist(),
                    'posterior_entropy':(-probabilities*np.log(np.maximum(probabilities,1e-300))).sum(1).tolist(),
                    'mixture_log_density':density.tolist()}


def affine_pca(x, means, basis, assignment):
    x=np.asarray(x,dtype=np.float64); out=np.empty_like(x)
    for k in np.unique(assignment):
        ix=assignment==k; b=basis[k]
        out[ix]=means[k]+((x[ix]-means[k])@b.T)@b
    return out


def conditions():
    result=[{'method':'identity'},{'method':'gmm_center'},
            {'method':'common_center'},{'method':'old_pca','rank':8}]
    for rank in [8,4,16]:
        result.extend({'method':method,'rank':rank}
                      for method in ['pca','fa','fa_orthogonal','mfa_hard','mfa_soft'])
    return result


def key(condition):
    return condition['method']+(f'_{condition["rank"]}' if 'rank' in condition else '')


class FairDecoders:
    def __init__(self, arrays):
        self.arrays=arrays
        self.fixed={}; self.joint={}
        for rank in [8,4,16]:
            self.fixed[rank]=FactorDecoder(arrays['fixed_weights'],arrays['common_anchor'],
                arrays[f'fixed_w_{rank}'],arrays[f'fixed_noise_{rank}'])
            self.joint[rank]=FactorDecoder(*(arrays[f'joint_{name}_{rank}']
                for name in ['weights','means','loadings','noise']))

    def replace(self, x, condition):
        x=np.asarray(x,dtype=np.float64); method=condition['method']; rank=condition.get('rank')
        a=self.arrays; assignment=nearest(x,a['gmm_centers'])
        coding={'gmm_region':assignment.tolist()}
        if method=='identity': out=x.copy()
        elif method=='gmm_center': out=a['gmm_centers'][assignment]
        elif method=='common_center': out=a['common_anchor'][assignment]
        elif method=='old_pca': out=affine_pca(x,a['old_anchor'],a['old_pca'][:,:rank],assignment)
        elif method=='pca': out=affine_pca(x,a['common_anchor'],a[f'pca_{rank}'],assignment)
        elif method in ['fa','fa_orthogonal']:
            out=self.fixed[rank].assigned(x,assignment,orthogonal=method=='fa_orthogonal')
        elif method in ['mfa_hard','mfa_soft']:
            out,extra=self.joint[rank].reconstruct(x,soft=method=='mfa_soft');coding.update(extra)
        else: raise ValueError(method)
        return out,coding


def budget(condition,k,d):
    method=condition['method']; r=condition.get('rank',0)
    if method=='identity':return {'continuous_values_per_token':d,'region_id':False,'learned_scalars':0}
    continuous=k*r+k-1 if method=='mfa_soft' else r
    if method=='gmm_center': count=k*d
    elif method=='common_center': count=2*k*d
    elif method in ['old_pca','pca']: count=k*d*(r+2)
    elif method in ['fa','fa_orthogonal']: count=k*d*(r+3)
    else: count=k*(d*(r+2)+1)
    return {'continuous_values_per_token':continuous,'region_id':method!='mfa_soft',
            'learned_scalars':count,
            'accounting':'Means, bases/loadings, FA noise and any separate GMM encoder centers; joint mixture weights. Cached factorizations and training metadata excluded. FA-orthogonal includes full fitted FA parameters; its minimal projection decoder can omit noise.'}

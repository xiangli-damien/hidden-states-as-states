"""Float64 diagonal GMM EM with resident input and auditable convergence.

Internal translation improves moment arithmetic; fitted means are returned in
original coordinates. No feature scaling, PCA, whitening or dimension reduction.
The covariance regularizer is additive, matching HSS's diagonal-GMM convention.
"""
from dataclasses import replace
import numpy as np
from .gmm import _make_diag_model


class ResidentDiagonalEM:
    def __init__(self, X, *, device='cuda:0', chunk_size=2048, reg_covar=1e-6):
        import torch
        self.t=torch;self.device=device;self.reg=float(reg_covar);self.chunk=int(chunk_size)
        raw=torch.as_tensor(np.asarray(X).copy(),device=device,dtype=torch.float64)
        if raw.ndim!=2 or min(raw.shape)<1 or not bool(torch.isfinite(raw).all()):
            raise ValueError('Finite nonempty N by D input required')
        if self.reg<=0 or self.chunk<1:raise ValueError('Invalid regularizer/chunk')
        self.offset=raw.mean(0);self.X=raw-self.offset;self.X2=self.X.square()
        self.n,self.d=self.X.shape;self.sq=self.X2.sum(1)

    def initialize(self,k,seed):
        """Single-trial D² kmeans++ seeding, followed by hard-assignment moments.

        Not a separate KMeans fit. Random choices are from a CPU NumPy RNG;
        distances and sufficient statistics use float64 on the selected device.
        """
        if not 1<=k<=self.n:raise ValueError('Invalid K')
        t=self.t;rng=np.random.default_rng(seed);indices=[int(rng.integers(self.n))]
        closest=t.full((self.n,),float('inf'),device=self.device,dtype=t.float64)
        for j in range(k):
            c=self.X[indices[-1]]
            distance=(self.sq-2*(self.X@c)+c.square().sum()).clamp_min(0)
            closest=t.minimum(closest,distance)
            if j+1<k:
                prob=closest.cpu().numpy();prob[indices]=0;total=prob.sum()
                if total>0:new=int(rng.choice(self.n,p=prob/total))
                else:new=int(rng.choice(np.setdiff1d(np.arange(self.n),indices)))
                indices.append(new)
        centers=self.X[indices]
        labels=[]
        for i in range(0,self.n,self.chunk):
            x=self.X[i:i+self.chunk]
            d=self.sq[i:i+len(x),None]-2*x@centers.T+centers.square().sum(1)[None]
            labels.append(d.argmin(1))
        labels=t.cat(labels)
        labels[t.tensor(indices,device=self.device)]=t.arange(k,device=self.device)
        weights=t.bincount(labels,minlength=k).to(t.float64)
        sums=t.zeros((k,self.d),device=self.device,dtype=t.float64)
        squares=t.zeros_like(sums)
        sums.index_add_(0,labels,self.X);squares.index_add_(0,labels,self.X2)
        means=sums/weights[:,None]
        variance=(squares/weights[:,None]-means.square()).clamp_min(0)+self.reg
        return (weights/self.n,means,variance),indices

    def evaluate(self,p,*,update=True,probabilities=False):
        t=self.t;w,mu,var=p;k=len(w);inv=var.reciprocal()
        const=w.log()-.5*(self.d*np.log(2*np.pi)+var.log().sum(1)+(mu.square()*inv).sum(1))
        counts=t.zeros(k,device=self.device,dtype=t.float64)
        sums=t.zeros((k,self.d),device=self.device,dtype=t.float64);sq=t.zeros_like(sums)
        total=t.zeros((),device=self.device,dtype=t.float64);entropy=t.zeros_like(total);rows=[]
        for i in range(0,self.n,self.chunk):
            x=self.X[i:i+self.chunk];x2=self.X2[i:i+len(x)]
            joint=const[None]+x@(mu*inv).T-.5*x2@inv.T
            norm=t.logsumexp(joint,1);logr=joint-norm[:,None];resp=logr.exp()
            total+=norm.sum();entropy-=(resp*logr).sum()
            if update:
                counts+=resp.sum(0);sums+=resp.T@x;sq+=resp.T@x2
            if probabilities:rows.append(resp.cpu().numpy())
        ll=float(total);ent=float(entropy)
        if not np.isfinite([ll,ent]).all():raise FloatingPointError('Nonfinite objective')
        new=None
        if update:
            if bool((counts<1e-8).any()):raise FloatingPointError('Collapsed component; restart retained as failed')
            mean=sums/counts[:,None]
            new=(counts/counts.sum(),mean,(sq/counts[:,None]-mean.square()).clamp_min(0)+self.reg)
        return ll/self.n,new,ent,np.concatenate(rows) if probabilities else None

    def model(self,p,converged,iterations):
        w,mu,var=p
        model=_make_diag_model(w.cpu().numpy(),(mu+self.offset).cpu().numpy(),var.cpu().numpy(),reg_covar=self.reg)
        return replace(model,converged_=bool(converged),n_iter_=int(iterations))

    def parameters(self,model):
        t=self.t
        return (t.as_tensor(model.weights_,device=self.device,dtype=t.float64),
            t.as_tensor(model.means_,device=self.device,dtype=t.float64)-self.offset,
            t.as_tensor(model.covariances_,device=self.device,dtype=t.float64))

    def fit(self,k,seed,*,max_iter=2000,tol=1e-5,initial=None):
        p,indices=self.initialize(k,seed) if initial is None else (self.parameters(initial),[])
        ll,new,_,_=self.evaluate(p);history=[ll];converged=False
        for iteration in range(1,max_iter+1):
            p=new;now,new,_,_=self.evaluate(p)
            history.append(now)
            if abs(now-ll)<=tol:converged=True;break
            ll=now
        # Metrics are for the returned parameters, not the next EM update.
        ll,_,entropy,prob=self.evaluate(p,update=False,probabilities=True)
        params=k*(2*self.d)+(k-1);bic=-2*ll*self.n+params*np.log(self.n)
        metrics=dict(log_likelihood=ll*self.n,bic=float(bic),entropy=entropy,icl=float(bic+2*entropy),
            n_parameters=params,converged=converged,n_iter=iteration,history=history,
            initial_row_indices=indices,precision='float64',backend=self.device,
            internal_translation_only=True,raw_coordinates_preserved=True)
        return self.model(p,converged,iteration),metrics,prob

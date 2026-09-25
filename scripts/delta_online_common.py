"""Prefix-only route features and raw GMM update decoders."""
import numpy as np


FUNCTIONAL_METHODS=['identity','zero_update','global_update','delta_centroid','delta_shared8','delta_local8','delta_wrong8','state_local8']


def route_features(sequence, ks, depth=False, temporal=False):
    """Each table has unit mass per question; layer vocabularies remain local."""
    s=np.asarray(sequence,dtype=int)
    if s.ndim!=2 or len(s)==0 or s.shape[1]!=len(ks):raise ValueError('Invalid prefix sequence')
    blocks=[]; shapes=[]
    for j,k in enumerate(ks):
        if (s[:,j]<0).any() or (s[:,j]>=k).any():raise ValueError('Unknown region')
        blocks.append(np.bincount(s[:,j],minlength=k)/len(s));shapes.append((1,k))
    if depth:
        for j,(a,b) in enumerate(zip(ks,ks[1:])):
            blocks.append(np.bincount(s[:,j]*b+s[:,j+1],minlength=a*b)/len(s));shapes.append((a,b))
    if temporal:
        for j,k in enumerate(ks):
            blocks.append(np.bincount(s[:-1,j]*k+s[1:,j],minlength=k*k)/max(1,len(s)-1));shapes.append((k,k))
    return np.concatenate(blocks),shapes


class RouteBayes:
    """Class-conditional, row-normalized composite evidence; not exact likelihood."""
    def __init__(self,shapes,alpha=1.):self.shapes=shapes;self.alpha=alpha
    def fit(self,x,y):
        y=np.asarray(y);assert set(y)=={0,1} and self.alpha>0
        self.prior=float(np.log(np.sum(y==1)/np.sum(y==0)));weights=[];offset=0
        for a,b in self.shapes:
            v=np.stack([x[y==c,offset:offset+a*b].sum(0).reshape(a,b)+self.alpha for c in [0,1]])
            v/=v.sum(-1,keepdims=True);weights.extend((np.log(v[1])-np.log(v[0])).ravel());offset+=a*b
        assert offset==x.shape[1];self.weights=np.array(weights);return self
    def decision_function(self,x):return np.asarray(x)@self.weights+self.prior


def posterior_region(x,d):
    """Diagonal Gaussian posterior MAP, matching the original sklearn GMM."""
    x=np.asarray(x,dtype=np.float64);c=d['centers'].astype(float);v=d['variances'].astype(float)
    scores=-.5*((x[:,None]-c[None])**2/v[None]+np.log(v)[None]).sum(-1)+np.log(d['weights'])[None]
    return scores.argmax(1)


def decode(x,d,kind):
    x=np.asarray(x,dtype=np.float64);s=posterior_region(x,d);mu=d['centers'][s].astype(float)
    if kind=='centroid':return mu,s
    b=d['shared_basis'] if kind=='shared8' else None
    if b is not None:return mu+(x-mu)@b.astype(float).T@b.astype(float),s
    result=mu.copy()
    for i,k in enumerate(s):
        b=d['local_basis'][int(d['permutation'][k]) if kind=='wrong8' else k].astype(float)
        result[i]+=(x[i]-mu[i])@b.T@b
    return result,s


def replacement(before,after,delta_decoder,state_decoder,name):
    before=np.asarray(before,dtype=np.float64);after=np.asarray(after,dtype=np.float64)
    delta=after-before
    if name=='identity':return after.copy()
    if name=='zero_update':return before.copy()
    if name=='global_update':return before+delta_decoder['train_mean']
    if name=='state_local8':return decode(after,state_decoder,'local8')[0]
    if name not in FUNCTIONAL_METHODS:raise ValueError(name)
    return before+decode(delta,delta_decoder,name.removeprefix('delta_'))[0]


def prefix_controls(confidence,prompt_length,token_kinds,p):
    """Rows0..p are distributions after 0..p observed response tokens.

    Entropy/NLL for the observed prefix use rows[:p]; row p predicts the NEXT token.
    Future reference NLL at row p is deliberately excluded.
    """
    c=np.asarray(confidence);kinds=np.asarray(token_kinds)
    assert len(c)>p and len(kinds)>=p and p>0
    return np.r_[c[p,0],c[:p,0].mean(),c[p,1],c[:p,3].mean(),np.log1p(prompt_length),
                 np.bincount(kinds[:p],minlength=6)/p]

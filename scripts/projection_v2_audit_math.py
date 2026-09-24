"""Independent row-wise audit arithmetic; never imported by the generator."""
import numpy as np


def independent_projection(x,centers,local,shared,c,permutation,epsilon):
    x=np.asarray(x,dtype=float);labels=np.square(x[:,None]-centers[None]).sum(-1).argmin(1)
    expected=x.copy();fallback=np.zeros(len(x),bool)
    if c['operator']!='baseline':
        for i,k in enumerate(labels):
            donor=permutation[k] if c['operator']=='wrong' else k
            basis=shared[:c['rank']] if c['operator']=='shared' else local[donor,:c['rank']]
            # Separate row-wise matrix products from the grouped executor.
            expected[i]=centers[k]+basis.T@(basis@(x[i]-centers[k]))
        if c['operator']=='norm':
            for i in range(len(x)):
                norm=np.linalg.norm(expected[i])
                if norm<=epsilon:expected[i]=x[i];fallback[i]=True
                else:expected[i]*=np.linalg.norm(x[i])/norm
        elif c['operator']=='common':expected=x+np.sum(expected-x,axis=0)/len(x)
    return x+c['alpha']*(expected-x),labels,fallback


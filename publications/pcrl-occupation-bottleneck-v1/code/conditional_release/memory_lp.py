"""Exact low-memory actual-code LP aggregation; shared mature solver unchanged."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .lp import LPProgram,solve_program


def encoded_pair(emitted1,emitted2):
    a,b=np.asarray(emitted1),np.asarray(emitted2)
    if a.dtype!=np.int64 or b.dtype!=np.int64 or a.shape!=b.shape or np.any(a<0) or np.any(b<0):
        raise ValueError('pair encoding needs aligned nonnegative global IDs')
    width=int(b.max())+1
    if int(a.max())*width+int(b.max())>np.iinfo(np.int64).max:
        raise OverflowError('pair encoding exceeds int64')
    # Numeric ascending order is precisely lexicographic tuple order.
    return a*width+b


def protection_program(law,sensitive,constraints):
    sex=np.asarray(sensitive)
    if sex.dtype!=np.int64 or sex.shape!=(len(law.base),) or not np.isin(sex,(0,1)).all():
        raise ValueError('protected fitting labels differ')
    if constraints.matrix.shape[1]!=law.parameters:raise ValueError('constraint parameter count differs')
    projections=[];offsets=[];epigraph_weights=[];names=[];blocks=[];cursor=0
    for view,weight in (('pair',1.),('r1',.5),('r2',.5)):
        values=(encoded_pair(law.emitted1,law.emitted2).ravel() if view=='pair' else
                getattr(law,'emitted'+view[-1]).ravel())
        unique,inv=np.unique(values,return_inverse=True);z=len(unique)
        del unique,values
        for s in (0,1):
            weighted=np.repeat(law.weights*(sex==s),law.base.shape[1])
            incidence=sparse.coo_matrix((weighted,(inv,np.arange(law.base.size))),shape=(z,law.base.size)).tocsr()
            projections.append(incidence@law.delta)
            offsets.append(np.asarray(incidence@law.base.ravel()).ravel())
            names.extend(f'epigraph/{view}/sex{s}/output{k}' for k in range(z))
            del weighted,incidence
        blocks.append((z,cursor));cursor+=z;epigraph_weights.extend([weight]*z)
        del inv
    epigraph=[]
    for z,start in blocks:
        neg=sparse.coo_matrix((-np.ones(z),(np.arange(z),start+np.arange(z))),shape=(z,cursor)).tocsr()
        epigraph.extend((neg,neg))
    a=constraints.matrix
    matrix=sparse.vstack((sparse.hstack((a,sparse.csr_matrix((a.shape[0],cursor)))),
                         sparse.hstack((sparse.vstack(projections),sparse.vstack(epigraph)))),format='csr')
    eq=sparse.hstack((law.equality_matrix,sparse.csr_matrix((len(law.equality_rhs),cursor))),format='csr')
    return LPProgram(np.r_[np.zeros(law.parameters),epigraph_weights],matrix,np.r_[constraints.rhs,-np.concatenate(offsets)],
                     eq,law.equality_rhs,np.vstack((law.bounds,np.tile([0.,np.inf],(cursor,1)))),
                     list(constraints.names)+names,law.parameters)


def solve_protection(law,sensitive,constraints,**kwargs):return solve_program(protection_program(law,sensitive,constraints),**kwargs)


def direct_bayes(law,theta,sensitive):
    q=law.probabilities(theta);sex=np.asarray(sensitive);answers={}
    for view in ('pair','r1','r2'):
        ids=encoded_pair(law.emitted1,law.emitted2).ravel() if view=='pair' else getattr(law,'emitted'+view[-1]).ravel()
        unique,inv=np.unique(ids,return_inverse=True);width=len(unique)
        del unique,ids
        mass=np.stack([np.bincount(inv,weights=(q*(law.weights*(sex==s))[:,None]).ravel(),minlength=width) for s in (0,1)])
        answers[view]=float(mass.max(0).sum());del inv,mass
    return answers['pair']+.5*(answers['r1']+answers['r2']),answers


class LazyTaskCosts:
    """Constant-sized code/class tables; one action-cost matrix per access."""
    def __init__(self,space,prepared,labels):
        from .preparation import probability_costs
        self.codes={r:prepared[f'code_r{r}'] for r in (1,2)};self.labels=dict(enumerate(labels,1));self.tables={}
        for r in (1,2):
            bank=getattr(space,f'probability{r}')
            lookup={metric:np.empty((len(bank),bank.shape[1]),np.float64) for metric in ('accuracy','nll','brier')}
            for label in range(bank.shape[1]):
                loss=probability_costs(bank,np.full(len(bank),label,np.int64))
                for metric,value in loss.items():lookup[metric][:,label]=value
            for metric,table in lookup.items():self.tables[f'r{r}_{metric}']=table
    def __getitem__(self,key):
        r=int(key[1]);return self.tables[key][self.codes[r],self.labels[r][:,None]]
    def items(self):
        for key in self.tables:yield key,self[key]
    def __len__(self):return len(self.tables)


def expected_rows(prepared,mass,space,labels,references):
    from .preparation import reference_costs
    costs=LazyTaskCosts(space,prepared,labels);refs=reference_costs(references,labels)
    expected={key:(mass*value).sum(1) for key,value in costs.items()}
    return {'expected':expected,'reference':refs,'excess':{ref:{key:expected[key]-value for key,value in values.items()} for ref,values in refs.items()}}

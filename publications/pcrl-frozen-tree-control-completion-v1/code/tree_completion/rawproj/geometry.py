"""C1 frozen float64 own-head compression. No loading or fitting side effects.

The only row representation transmitted is factorized Pz. q is internal and
every transmitted coordinate depends solely on q; the original residual is
never added to a rounded matrix projection. Outputs use the actual current
head on the transmitted Pz, including its original affine arithmetic order.
"""
from __future__ import annotations
import numpy as np

RULE={
    'arithmetic':'float64', 'relative_rank_cutoff':1e-12,
    'epsilon_multiplier':32., 'ambiguous_cutoff_factors':[.5,2.],
    'algebra_atol':1e-10, 'algebra_rtol':1e-10,
    'logit_atol':1e-10, 'logit_rtol':1e-10,
    'probability_atol':1e-10, 'probability_rtol':0.,
    'near_tie_absolute_logit_gap':1e-10,
    'decision_rule':'argmax probabilities, first class in fixed source order',
    'decision_requirement':'exact equality on every checked row, including near ties',
}


def _matrix(value,name):
    x=np.asarray(value,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError(name+' must be a finite matrix')
    return x


def build_geometry(weight,bias,class_order=None):
    """Build from admitted current affine-head tensors, never older source heads.

    A = W - class_mean(W); P = A^+ A = B.T B. Numerically ambiguous
    singular values fail admission rather than silently changing the release.
    Dimensions/rank are measured; income rank one is not assumed.
    """
    w=_matrix(weight,'weight');k,d=w.shape;b=np.asarray(bias,dtype=np.float64)
    if k<2 or d<1 or b.shape!=(k,) or not np.isfinite(b).all():raise ValueError('head shape differs')
    order=np.arange(k,dtype=np.int64) if class_order is None else np.asarray(class_order)
    if order.shape!=(k,) or order.dtype.kind not in 'iu' or len(np.unique(order))!=k:raise ValueError('class order differs')
    a=w-w.mean(0,keepdims=True);c=b-b.mean()
    _,s,vt=np.linalg.svd(a,full_matrices=False)
    top=float(s[0]) if len(s) else 0.
    cutoff=max(RULE['relative_rank_cutoff'],RULE['epsilon_multiplier']*max(k,d)*np.finfo(np.float64).eps)*top
    ambiguous=(s>=.5*cutoff)&(s<=2.*cutoff)&(s>0)
    if ambiguous.any():raise ValueError('singular value at frozen rank boundary; package unavailable')
    rank=int(np.sum(s>cutoff)) if top>0 else 0
    if rank>min(k-1,d):raise ValueError('class-centered head rank exceeds theoretical bound')
    basis=vt[:rank].copy()
    for row in basis:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    p=basis.T@basis
    residual=float(np.max(np.abs(a@p-a)))
    algebra=bool(np.allclose(a@p,a,atol=RULE['algebra_atol'],rtol=RULE['algebra_rtol']))
    if not algebra:raise ValueError('A P = A check failed')
    return {'schema':'OB1-head-row-space-v1','rule':dict(RULE),'weight':w.copy(),'bias':b.copy(),
            'class_order':order.astype(np.int64),'A':a,'c':c,'B':basis,'P':p,
            'rank':rank,'classes':k,'source_dimension':d,'singular_values':s,
            'rank_cutoff':cutoff,'AP_max_absolute_residual':residual,
            'rank_theoretical_maximum':min(k-1,d),'AP_check_passed':algebra,
            'release_contract':'factorized_Pz,actual_centered_logits,actual_probabilities,actual_decision'}


def _head(z,g):
    logits=z@g['weight'].T+g['bias'];logits-=logits.mean(1,keepdims=True)
    e=np.exp(logits-logits.max(1,keepdims=True));p=e/e.sum(1,keepdims=True)
    decision=g['class_order'][p.argmax(1)]
    return logits,p,decision


def projected_release(z,geometry):
    z=_matrix(z,'representation');g=geometry
    if z.shape[1]!=g['source_dimension']:raise ValueError('source representation dimension differs')
    # Never evaluate z @ rounded P: residual-dependent rounding can become
    # visible after reader standardization of tiny coordinate variation.
    coordinates=z@g['B'].T
    protected=coordinates@g['B']
    l,p,d=_head(protected,g)
    return {'representation':protected,'centered_logits':l,'probabilities':p,'decision':d,
            'full':np.column_stack((protected,l,p,d))}


def audit_geometry(z,geometry):
    """Authorized real features only. No labels required or accepted."""
    z=_matrix(z,'representation')
    if not len(z):raise ValueError('nonempty real-row audit required')
    g=geometry;released=projected_release(z,g);l,p,d=_head(z,g)
    projected_l=released['centered_logits'];projected_p=released['probabilities']
    logits_match=bool(np.allclose(l,projected_l,atol=RULE['logit_atol'],rtol=RULE['logit_rtol']))
    probability_match=bool(np.allclose(p,projected_p,atol=RULE['probability_atol'],rtol=RULE['probability_rtol']))
    same=d==released['decision'];ordered=np.sort(l,axis=1)
    gaps=ordered[:,-1]-ordered[:,-2];near=gaps<=RULE['near_tie_absolute_logit_gap']
    return {'rows':len(z),'rank':g['rank'],'theoretical_maximum':g['rank_theoretical_maximum'],
            'AP_check_passed':g['AP_check_passed'],'centered_logits_match':logits_match,
            'probabilities_match':probability_match,'decision_preserving':bool(same.all()),
            'decision_mismatch_rows':int((~same).sum()),'near_tie_rows':int(near.sum()),
            'near_tie_mismatch_rows':int((near&~same).sum()),
            'max_absolute_logit_difference':float(np.max(np.abs(l-projected_l))),
            'max_absolute_probability_difference':float(np.max(np.abs(p-projected_p))),
            'minimum_top_two_logit_gap':float(gaps.min()),
            'passed':bool(g['AP_check_passed'] and logits_match and probability_match and same.all()),
            'raw_z_or_teacher_outputs_co_released':False}


def public_geometry(geometry):
    """Fixed head/basis parameters are public to readers; no per-row q."""
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in geometry.items()}

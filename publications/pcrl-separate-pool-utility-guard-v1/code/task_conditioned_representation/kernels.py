"""Pinned MMD-B-Fair core and named full-wire/scale adaptations.

Official source: namratadeka/mmd-b-fair@280a178abc86802cef58f99ab3eff4d2365ddacb.
No scientific job is launched on import.
"""
from dataclasses import dataclass
import math
import torch

OFFICIAL_SHA = "280a178abc86802cef58f99ab3eff4d2365ddacb"
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_DRAW = 32
N_BLOCKS = 4
BLOCK_SIZE = 8
VAR_FLOOR = 1e-6
STD_ABSOLUTE_FLOOR = 1e-12
STD_INITIAL_RELATIVE_FLOOR = 1e-8
DISTANCE_FLOOR = 1e-12
ALPHA = 0.05

def squared_distances(x, y):
    return ((x*x).sum(-1)[:, None] + (y*y).sum(-1)[None, :] - 2*x@y.T).clamp_min(0)

def official_core(kx, ky, kxy, n_blocks=N_BLOCKS, variance_floor=VAR_FLOOR):
    """Literal official biased-variance branch, including its missing /B.

    Inputs have shape [bandwidth,n,n]. The cross diagonal is removed for
    the one-sample U MMD; the variance includes all H entries as official.
    This estimates hypothetical block-test power and does not physically
    split the estimator minibatch (paper section 3.1 explicitly says so).
    """
    n = kx.shape[-1]
    if n < 2 or kx.shape != ky.shape or kx.shape != kxy.shape:
        raise ValueError("equal square kernel matrices with n>=2 required")
    def offdiag_mean(k):
        return (k.sum((-2,-1)) - k.diagonal(dim1=-2,dim2=-1).sum(-1))/(n*(n-1))
    mmd2 = offdiag_mean(kx) + offdiag_mean(ky) - 2*offdiag_mean(kxy)
    h = kx + ky - kxy - kxy.transpose(-1,-2)
    v1 = (h.sum(-1)/n).square().sum(-1)/n
    v2 = h.sum((-2,-1))/(n*n)
    variance = 4*(v1-v2.square())
    std = variance.clamp_min(variance_floor).sqrt()
    # Same alpha=.05 and standard normal CDF/ICDF as the official trainer.
    normal = torch.distributions.Normal(mmd2.new_tensor(0.),mmd2.new_tensor(1.))
    threshold = normal.icdf(mmd2.new_tensor(1-ALPHA))
    argument = math.sqrt(n_blocks)*mmd2/std-threshold
    return {"power":normal.cdf(argument), "mmd2":mmd2,
            "variance":variance,"argument":argument,"variance_floor_active":variance<variance_floor}

@dataclass
class DiagonalReference:
    mean: torch.Tensor
    scale: torch.Tensor
    initial_scale: torch.Tensor
    distance: torch.Tensor
    floor_count: int

    @classmethod
    def fit(cls, wire, previous=None):
        # All reference quantities are detached; retained release is untouched.
        w=wire.detach().to(torch.float64)
        if w.ndim!=2 or len(w)<2 or not torch.isfinite(w).all():
            raise ValueError("finite two-dimensional reference with >=2 rows required")
        mean=w.mean(0); raw=w.std(0,correction=0)
        initial=raw if previous is None else previous.initial_scale.to(w)
        floor=torch.maximum(initial*STD_INITIAL_RELATIVE_FLOOR,raw.new_full(raw.shape,STD_ABSOLUTE_FLOOR))
        scale=torch.maximum(raw,floor)
        norm=(w-mean)/scale
        distances=squared_distances(norm,norm)
        upper=distances[torch.triu(torch.ones_like(distances,dtype=torch.bool),diagonal=1)]
        distance=upper.median().clamp_min(DISTANCE_FLOOR)
        return cls(mean,scale,initial.detach().clone(),distance,int((raw<floor).sum()))

    def normalize(self,wire):
        return (wire.to(torch.float64)-self.mean.to(wire))/self.scale.to(wire)

    def snapshot(self):
        return {"mean":self.mean.cpu(),"scale":self.scale.cpu(),"initial_scale":self.initial_scale.cpu(),
                "distance":float(self.distance),"floor_count":self.floor_count}

def full_wire(z, logits):
    """Features plus actual centered own-head logits: sufficient for outputs.

    Probabilities/decision are deterministic functions of these logits. They
    remain on the actual release and in attacks; excluding their duplicate
    transforms from the fitting kernel is a declared sufficient-wire adapter.
    """
    if z.ndim!=2 or logits.ndim!=2 or z.shape[0]!=logits.shape[0]:
        raise ValueError("aligned feature/logit matrices required")
    return torch.cat((z.to(torch.float64),logits.to(torch.float64)-logits.to(torch.float64).mean(-1,keepdim=True)),-1)

def kernel_triplet(left_wires,right_wires,references):
    """Product RBF over aligned actual recipient wires; detects XOR laws.

    A single recipient is the same code with a one-element sequence.
    Shared relative bandwidth index across recipients and weighted strata.
    """
    if len(left_wires)!=len(right_wires) or len(left_wires)!=len(references):
        raise ValueError("wire/reference count mismatch")
    dxx=dyy=dxy=None
    for x,y,ref in zip(left_wires,right_wires,references):
        xn=ref.normalize(x); yn=ref.normalize(y)
        a=squared_distances(xn,xn)/ref.distance
        b=squared_distances(yn,yn)/ref.distance
        c=squared_distances(xn,yn)/ref.distance
        dxx=a if dxx is None else dxx+a
        dyy=b if dyy is None else dyy+b
        dxy=c if dxy is None else dxy+c
    bandwidth=dxx.new_tensor(BANDWIDTH_MULTIPLIERS)[:,None,None].square()
    return torch.exp(-dxx[None]/bandwidth),torch.exp(-dyy[None]/bandwidth),torch.exp(-dxy[None]/bandwidth)

def require_alignment(row_ids0,row_ids1):
    if list(row_ids0)!=list(row_ids1):
        raise ValueError("coalition recipient row alignment mismatch")

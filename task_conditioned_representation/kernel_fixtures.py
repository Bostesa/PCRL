"""Data-free pre-optimizer engineering fixtures, executed only by root queue.

The harness compares selected functions compiled from pinned official source;
no private data, optimizers, checkpoints or historical results are read.
"""
import ast
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from .kernels import (official_core,DiagonalReference,full_wire,kernel_triplet,
                      require_alignment,N_DRAW,N_BLOCKS,BLOCK_SIZE)
from .objectives import StrataRegistry

def run():
    torch.set_num_threads(1)
    torch.manual_seed(79031)
    results={}
    # Compile the exact source functions, retaining all their numeric expressions.
    source=Path(__file__).with_name("official_reference_mmd.py").read_text()
    names={"mmd_variance","h1_mean_var_gram","block_approx_test_power"}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    official={"torch":torch,"np":np,"zero_diag":lambda k:k-torch.diag_embed(k.diagonal(dim1=-2,dim2=-1))}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned_official_mmd.py","exec"),official)
    grid=torch.arange(64*3,dtype=torch.float64).reshape(64,3)
    features=torch.sin(grid*.117)+.1*torch.cos(grid*.031)
    widths=torch.tensor((.0625,.25,1.,4.,16.),dtype=torch.float64)[:,None,None]
    def d(a,b):return (a[:,None]-b[None,:]).square().sum(-1)
    x,y=features[:32],features[32:]
    kx,ky,kxy=(torch.exp(-v[None]/widths) for v in (d(x,x),d(y,y),d(x,y)))
    actual=official_core(kx,ky,kxy)
    om,ov,_=official["h1_mean_var_gram"](kx,ky,kxy,BLOCK_SIZE,True,True,False)
    normal=torch.distributions.Normal(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
    op,oa=official["block_approx_test_power"](om,ov.clamp_min(1e-6).sqrt(),N_BLOCKS,normal.icdf(torch.tensor(.95,dtype=torch.float64)),normal)
    errors={"mmd2":float((actual["mmd2"]-om).abs().max()),"variance":float((actual["variance"]-ov).abs().max()),"power":float((actual["power"]-op).abs().max()),"argument":float((actual["argument"]-oa).abs().max())}
    assert max(errors.values())<1e-12,errors
    results["official_core_parity"]=errors

    # Exactly label-mediated signal is equal within task labels, while marginal
    # groups may differ. No finite-sample method-win requirement is asserted.
    const=torch.ones((32,2),dtype=torch.float64)
    reference=DiagonalReference.fit(torch.cat((const,const*2)))
    equal=official_core(*kernel_triplet([const],[const],[reference]))["power"]
    signal=official_core(*kernel_triplet([const],[const*2],[reference]))["power"]
    assert float(equal.max())<=.05+1e-12
    assert float(signal.max())>.99
    results["label_mediated_and_extra_signal"]={"equal_within_label_max":float(equal.max()),"extra_within_label_max":float(signal.max())}

    # Complementary views with exactly identical local marginals, aligned XOR.
    u=torch.tensor([-1.,1.]*16,dtype=torch.float64)[:,None]
    xr=[u,u]; yr=[u,-u]
    refs=[DiagonalReference.fit(torch.cat((a,b))) for a,b in zip(xr,yr)]
    locals_=[float(official_core(*kernel_triplet([a],[b],[r]))["power"].max()) for a,b,r in zip(xr,yr,refs)]
    pair=float(official_core(*kernel_triplet(xr,yr,refs))["power"].max())
    assert max(locals_)<=.05+1e-12 and pair>.99,(locals_,pair)
    results["aligned_xor"]={"local_powers":locals_,"pair_power":pair}
    try:require_alignment(range(64),list(range(1,64))+[0])
    except ValueError:results["shuffled_alignment_rejected"]=True
    else:raise AssertionError("shuffled alignment accepted")

    # Reciprocal feature coordinate scales + inverse head coefficients leave
    # probabilities and information fixed. Refresh fitting reference each time.
    z=torch.sin(torch.arange(64*16,dtype=torch.float64).reshape(64,16)*.053)
    head=torch.cos(torch.arange(6*16,dtype=torch.float64).reshape(6,16)*.072)
    logits=z@head.T
    w=full_wire(z,logits); ref=DiagonalReference.fit(w)
    baseline=official_core(*kernel_triplet([w[:32]],[w[32:]],[ref]))["power"]
    baseline_idx=int(baseline.argmax())
    scale_errors=[]
    for magnitude in (1e-3,1e-2,1.,1e2,1e3):
        scales=torch.linspace(magnitude,1/magnitude,16,dtype=torch.float64)
        scaled_z=z*scales; scaled_head=head/scales
        scaled_logits=scaled_z@scaled_head.T
        scaled_w=full_wire(scaled_z,scaled_logits); scaled_ref=DiagonalReference.fit(scaled_w)
        power=official_core(*kernel_triplet([scaled_w[:32]],[scaled_w[32:]],[scaled_ref]))["power"]
        e=float((power-baseline).abs().max())
        assert e<1e-10 and int(power.argmax())==baseline_idx
        assert torch.allclose(torch.softmax(scaled_logits,-1),torch.softmax(logits,-1),atol=1e-12,rtol=1e-12)
        scale_errors.append({"magnitude":magnitude,"power_max_error":e,"kernel_index":int(power.argmax()),"floor_count":scaled_ref.floor_count})
    results["reciprocal_scale_inverse_head"]=scale_errors

    live=w.clone().requires_grad_(True)
    penalty=official_core(*kernel_triplet([live[:32]],[live[32:]],[ref]))["power"].max()
    gradient=torch.autograd.grad(penalty,live)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm())>1e-12
    assert not ref.mean.requires_grad and not ref.scale.requires_grad and not ref.distance.requires_grad
    results["finite_nonzero_gradient"]={"norm":float(gradient.norm()),"reference_detached":True}

    # Registry laws: supported vs uncovered mass, no replacement, badgroups.
    yi=[];yo=[];sex=[];groups=[]
    for c in range(2):
        for s in range(2):
            count=32 if c==0 else (31 if s==0 else 32)
            for j in range(count):yi.append(c);yo.append(c);sex.append(s);groups.append(f"{c}_{s}_{j}")
    registry=StrataRegistry(yi,yo,sex,groups,class_counts=(2,2))
    for draw in registry.draw(random.Random(11)):
        assert len(draw.left)==len(draw.right)==N_DRAW
        assert len({groups[i] for i in draw.left+draw.right})==2*N_DRAW
    c0=next(r for r in registry.coverage if r["kind"]=="local0" and r["key"]==[0])
    c1=next(r for r in registry.coverage if r["kind"]=="local0" and r["key"]==[1])
    assert c0["supported"] and not c1["supported"] and c0["weight"]<1.
    assert math.isclose(c0["weight"]+c1["weight"],1.)
    results["support_weights_without_replacement"]={"covered_mass":c0["weight"],"uncovered_mass":c1["weight"],"minimum":32}
    bad_groups=groups.copy();bad_groups[-1]=groups[0]
    try:StrataRegistry(yi,yo,sex,bad_groups,class_counts=(2,2))
    except ValueError:results["contradictory_duplicate_group_rejected"]=True
    else:raise AssertionError("contradictory duplicate groups accepted")
    results["status"]="pass"
    return results

if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))

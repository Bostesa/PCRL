"""Conditional strata registry and operational full-wire kernel objective."""
from dataclasses import dataclass
from collections import Counter,defaultdict
from itertools import combinations
import torch
from .kernels import N_DRAW,DiagonalReference,full_wire,kernel_triplet,official_core,require_alignment

@dataclass(frozen=True)
class Draw:
    kind: str
    key: tuple
    left: tuple
    right: tuple
    weight: float

class StrataRegistry:
    """Only called with admitted FIT optimization rows, never reporting rows.

    Sensitive class/pair weights use row frequencies among *all* FIT rows;
    unsupported mass is not redistributed. Distinct groups are drawn uniformly,
    then one of their rows uniformly: no duplicates within an arm or a draw.
    Task CE must use a separate uniformly sampled row minibatch.
    """
    def __init__(self,y_income,y_occ,sex,groups,class_counts=(2,6)):
        yi=list(map(int,y_income)); yo=list(map(int,y_occ)); s=list(map(int,sex)); g=list(groups)
        n=len(yi)
        if not n or not(len(yo)==len(s)==len(g)==n): raise ValueError("aligned FIT labels/groups required")
        if any(v not in (0,1) for v in s): raise ValueError("SEX must be binary")
        if any(not 0<=v<class_counts[0] for v in yi) or any(not 0<=v<class_counts[1] for v in yo):
            raise ValueError("declared task class order mismatch")
        # Duplicate groups must have consistent labels: no cross-stratum person reuse.
        group_law={}
        for a,b,c,d in zip(yi,yo,s,g):
            law=(a,b,c)
            if d in group_law and group_law[d]!=law: raise ValueError("duplicate group has contradictory labels")
            group_law[d]=law
        self.specs=[]; self.coverage=[]; self.n_rows=n; self.class_counts=class_counts
        for kind,labels in (("local0",yi),("local1",yo),("pair",list(zip(yi,yo)))):
            frequencies=Counter(labels)
            possible=range(class_counts[0]) if kind=="local0" else range(class_counts[1]) if kind=="local1" else [(a,b) for a in range(class_counts[0]) for b in range(class_counts[1])]
            for c in possible:
                pools=[defaultdict(list),defaultdict(list)]
                for i,(label,arm,group) in enumerate(zip(labels,s,g)):
                    if label==c: pools[arm][group].append(i)
                key=c if isinstance(c,tuple) else (c,)
                weight=frequencies[c]/n
                supported=all(len(p)>=N_DRAW for p in pools)
                self.coverage.append({"kind":kind,"key":list(key),"weight":weight,"groups_by_sex":[len(p) for p in pools],"supported":supported})
                if supported: self.specs.append((kind,key,pools,weight))
        # Marginal sensitive pools for unconditional control.
        pools=[defaultdict(list),defaultdict(list)]
        for i,(arm,group) in enumerate(zip(s,g)): pools[arm][group].append(i)
        if all(len(p)>=N_DRAW for p in pools): self.specs.append(("unconditional",(),pools,1.))
        for r,labels in enumerate((yi,yo)):
            counts=Counter(labels); pairs=list(combinations(range(class_counts[r]),2))
            denom=sum(counts[a]*counts[b] for a,b in pairs)
            for a,b in pairs:
                pools=[defaultdict(list),defaultdict(list)]
                for i,(c,group) in enumerate(zip(labels,g)):
                    if c==a: pools[0][group].append(i)
                    elif c==b: pools[1][group].append(i)
                supported=all(len(p)>=N_DRAW for p in pools)
                weight=counts[a]*counts[b]/denom if denom else 0.
                self.coverage.append({"kind":f"target{r}","key":[a,b],"weight":weight,"groups_by_class":[len(p) for p in pools],"supported":supported})
                if supported:self.specs.append((f"target{r}",(a,b),pools,weight))

    def draw(self,rng,family=None):
        draws=[]
        for kind,key,pools,weight in self.specs:
            if family=="U_TASKPOWER" and not kind.startswith("target"):continue
            if family=="JOINT_UNCONDITIONAL" and kind in ("local0","local1","pair"):continue
            if family!="JOINT_UNCONDITIONAL" and kind=="unconditional":continue
            arms=[]
            for pool in pools:
                sampled=rng.sample(list(pool),N_DRAW)
                arms.append(tuple(rng.choice(pool[group]) for group in sampled))
            draws.append(Draw(kind,key,arms[0],arms[1],weight))
        return draws

class KernelObjective:
    def __init__(self): self.references=None

    def refresh_reference(self,zs,logits):
        prior=self.references or (None,None)
        self.references=tuple(DiagonalReference.fit(full_wire(z,l),p) for z,l,p in zip(zs,logits,prior))
        if len(self.references)!=2:raise ValueError("two recipient references required")
        return [r.snapshot() for r in self.references]

    def refresh_model_reference(self,model,x_fixed):
        with torch.no_grad():
            zs=tuple(model.encode(r,x_fixed) for r in range(2))
            logits=tuple(model.head[r](zs[r].to(torch.float64)) for r in range(2))
        return self.refresh_reference(zs,logits)

    def evaluate(self,zs,logits,row_indices,draws,family,active=(0,1)):
        if self.references is None:raise RuntimeError("refresh fixed FIT reference before objective evaluation")
        indices=list(map(int,row_indices)); lookup={v:i for i,v in enumerate(indices)}
        if len(lookup)!=len(indices):raise ValueError("forward row union must be unique")
        wires=tuple(full_wire(z,l) for z,l in zip(zs,logits))
        require_alignment(indices,indices)
        zero=sum(w.sum()*0 for w in wires)
        values={k:zero.repeat(5) for k in ("local0","local1","pair","target0","target1")}
        detail=[]
        for draw in draws:
            left=torch.tensor([lookup[i] for i in draw.left],device=wires[0].device)
            right=torch.tensor([lookup[i] for i in draw.right],device=wires[0].device)
            if draw.kind=="unconditional": kinds=("local0","local1","pair")
            else:kinds=(draw.kind,)
            for kind in kinds:
                recipients=(0,1) if kind=="pair" else (int(kind[-1]),)
                # During sequential training inactive local and target terms are constants;
                # keep them for logging, pair still reads the frozen first actual wire.
                core=official_core(*kernel_triplet([wires[r][left] for r in recipients],[wires[r][right] for r in recipients],[self.references[r] for r in recipients]))
                values[kind]=values[kind]+draw.weight*core["power"]
                detail.append({"kind":kind,"key":list(draw.key),"weight":draw.weight,"powers":core["power"].detach().tolist(),"variance_floor_count":int(core["variance_floor_active"].sum())})
        terms={k:v.max() for k,v in values.items()}
        terms["target_loss"]=-sum(terms[f"target{r}"] for r in active)
        local=sum(terms[f"local{r}"] for r in active)
        terms["sensitive"]=local+(terms["pair"] if family in ("JOINT_CONDITIONAL","JOINT_UNCONDITIONAL","SEQ12","SEQ21") else zero)
        if family in ("U_NEW","U_TASKPOWER"):terms["sensitive"]=zero
        if family=="U_NEW":terms["target_loss"]=zero
        terms["diagnostics"]={"draws":detail,"kernel_indices":{k:int(v.argmax()) for k,v in values.items()},"reference":[r.snapshot() for r in self.references]}
        return terms

    def loss(self,model,x,draws,family,active=(0,1)):
        indices=sorted({i for d in draws for i in d.left+d.right})
        if not indices:raise ValueError("empty registered objective draw")
        batch=x[indices]
        zs=tuple(model.encode(r,batch) for r in range(2))
        logits=tuple(model.head[r](zs[r].to(torch.float64)) for r in range(2))
        return self.evaluate(zs,logits,indices,draws,family,active)

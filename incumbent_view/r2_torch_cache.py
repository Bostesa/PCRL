"""Cache frozen reader/cost tensors once per round, preserving exact formulas."""
import numpy as np
from .r2_space import CAPS,SAFETY


class FrozenRound:
    def __init__(self,cost,fold,readers,*,torch,task_views=(1,2),guard_views=("r1","r2")):
        self.torch=torch;self.masks=tuple(torch.as_tensor(x) for x in
            (np.ones(len(fold),bool),*(fold==k for k in range(3))))
        self.cost=[]
        for key,c in cost.items():
            if int(key[1]) not in task_views:continue
            sign=-1 if key.endswith("accuracy") else 1
            self.cost.append((key,torch.as_tensor(np.ascontiguousarray((c-c[:,0,None])*sign)),
                              CAPS[key.split("_")[-1]]-SAFETY))
        self.losses={};self.guards={}
        for view in ("r1","r2","pair"):
            matching=[r for r in readers if r.view==view]
            if matching:
                self.losses[view]=torch.as_tensor(np.stack([r.loss for r in matching]))
                if view in guard_views:
                    self.guards[view]=self.losses[view]-self.losses[view][:,:,0,None]

    def objective(self,q,arm,*,stage=None):
        views=("r1","r2") if arm=="V_L" else (("r1" if arm=="V_S12" else "r2",) if stage==1 else ("pair",))
        answer=self.torch.zeros((),dtype=self.torch.float64)
        for view in views:
            values=self.torch.einsum("na,mna->m",q,self.losses[view])/len(q)
            answer-=values.min()
        return answer

    def constraints(self,q):
        out=[]
        for mask in self.masks:
            qm=q[mask]
            for _,diff,cap in self.cost:
                out.append((qm*diff[mask]).sum(1).mean()-cap)
            for view in ("r1","r2"):
                if view in self.guards:
                    vals=self.torch.einsum("na,mna->m",qm,self.guards[view][:,mask,:])/len(qm)
                    out.extend([-v-SAFETY for v in vals])
        return out

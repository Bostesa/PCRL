"""Independent FIT selection diagnostic from registered conditional laws."""
from __future__ import annotations
import hashlib,random
from itertools import combinations
import numpy as np
from .science import squared_distance,core_power,own_release

def seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str,items)).encode()).hexdigest()[:16],16)%2**63

def specs(labels,sex,groups):
    yi,yo=labels;result=[]
    for kind,law,possible in [('local0',yi,range(2)),('local1',yo,range(6)),('pair',list(zip(yi,yo)),list((a,b) for a in range(2) for b in range(6)))]:
        for label in possible:
            pools=[{},{}]
            rows=0
            for i,(value,s,g) in enumerate(zip(law,sex,groups)):
                if value==label:
                    rows+=1;pools[int(s)].setdefault(g,[]).append(i)
            if min(map(len,pools))>=32:
                result.append((kind,label if isinstance(label,tuple) else (int(label),),pools,rows/len(groups)))
    pools=[{},{}]
    for i,(s,g) in enumerate(zip(sex,groups)):pools[int(s)].setdefault(g,[]).append(i)
    if min(map(len,pools))>=32:result.append(('unconditional',(),pools,1.))
    for r,law in enumerate(labels):
        counts=[int((law==c).sum()) for c in range((2,6)[r])]
        pairs=list(combinations(range((2,6)[r]),2));denom=sum(counts[a]*counts[b] for a,b in pairs)
        for a,b in pairs:
            pools=[{},{}]
            for i,(c,g) in enumerate(zip(law,groups)):
                if c==a:pools[0].setdefault(g,[]).append(i)
                if c==b:pools[1].setdefault(g,[]).append(i)
            if min(map(len,pools))>=32:result.append((f'target{r}',(a,b),pools,counts[a]*counts[b]/denom))
    return result

def draw_selection(training,selection,family,seed):
    weights={(kind,key):weight for kind,key,pools,weight in training}
    rng=random.Random(seed_for('TCR1selectiondraw',seed));draws=[]
    for kind,key,pools,unused in selection:
        if (kind,key) not in weights:continue
        if family=='U_TASKPOWER' and not kind.startswith('target'):continue
        if family=='JOINT_UNCONDITIONAL' and kind in ('local0','local1','pair'):continue
        if family!='JOINT_UNCONDITIONAL' and kind=='unconditional':continue
        arms=[]
        for pool in pools:
            sampled=rng.sample(list(pool),32)
            arms.append(np.array([rng.choice(pool[g]) for g in sampled]))
        draws.append((kind,key,arms[0],arms[1],weights[(kind,key)]))
    return draws

def fixed_gram(left,right,reference):
    def array(v):return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
    mean,scale=array(reference['mean']),array(reference['scale'])
    x,y=(left-mean)/scale,(right-mean)/scale
    denom=float(reference['distance'])*np.array([.25,.5,1,2,4])**2
    return tuple(np.exp(-squared_distance(a,b)[None,:,:]/denom[:,None,None]) for a,b in ((x,x),(y,y),(x,y)))

def diagnostic(payload,x,draws,active=(0,1)):
    family=payload['family'];release=own_release(payload['model'],x)
    views=[np.column_stack((release[f'r{r+1}'],release[f'c{r+1}'])) for r in (0,1)]
    value={kind:np.zeros(5) for kind in ('local0','local1','pair','target0','target1')}
    for kind,key,left,right,weight in draws:
        kinds=('local0','local1','pair') if kind=='unconditional' else (kind,)
        for name in kinds:
            recipients=(0,1) if name=='pair' else (int(name[-1]),)
            grams=[fixed_gram(views[r][left],views[r][right],payload['reference'][r]) for r in recipients]
            combined=tuple(np.prod([g[k] for g in grams],axis=0) for k in range(3))
            value[name]+=weight*core_power(*combined)[0]
    terms={name:float(power.max()) for name,power in value.items()}
    local=sum(terms[f'local{r}'] for r in active)
    sensitive=local+(terms['pair'] if family in ('JOINT_CONDITIONAL','JOINT_UNCONDITIONAL','SEQ12','SEQ21') else 0)
    if family in ('U_NEW','U_TASKPOWER'):sensitive=0.
    terms['sensitive']=sensitive
    return terms

def raw_diagnostic(payload,x,sex,seed):
    import torch
    import torch.nn.functional as F
    rng=random.Random(seed_for('TCR1selectiondraw',seed))
    for _ in range(5):rng.sample(range(len(sex)),min(256,len(sex)))
    index=rng.sample(range(len(sex)),min(256,len(sex)))
    release=own_release(payload['model'],x[index]);views=[]
    from .science import wire
    for r in (1,2):views.append(wire(release,r))
    views.append(np.column_stack(views))
    objective=payload['objective_state'];entropy=objective['entropy'];terms=[]
    target=torch.as_tensor(sex[index],dtype=torch.long)
    with torch.no_grad():
        for view,ref,bank in zip(views,payload['reference'],objective['critics']):
            mean=ref['mean'].numpy();scale=ref['scale'].numpy()
            v=torch.from_numpy((view-mean)/scale)
            risks=[]
            for state in bank:
                if 'weight' in state:
                    out=F.linear(v,state['weight'],state['bias'])
                else:
                    out=v
                    layers=sorted(int(k.split('.')[0]) for k in state if k.endswith('.weight'))
                    for j,layer in enumerate(layers):
                        out=F.linear(out,state[f'{layer}.weight'],state[f'{layer}.bias'])
                        if j<len(layers)-1:out=F.relu(out)
                risks.append(float(F.cross_entropy(out,target)))
            terms.append(1-min(risks+[entropy])/entropy)
    return sum(terms)

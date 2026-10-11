"""Independent serialized policy laws; no optimization/deployment imports."""
from __future__ import annotations
import numpy as np
from .reconstruction import source_joint_law,source_seq21_law,validate_support
from .contexts import context_cells


def law_from_numbers(source,c1,c2,kind,receipt=None,package=None):
    c1,c2=np.asarray(c1,np.int64),np.asarray(c2,np.int64);n=len(c1)
    if kind=='P0':return c1[:,None],c2[:,None],np.ones((n,1))
    if kind in ('V_J','V_S21'):
        if kind=='V_J':return source_joint_law(source,c1,c2,receipt['eta'])
        return source_seq21_law(source,c1,c2,receipt['eta1'],receipt['eta2'])
    p1,p2=source['probability1'],source['probability2']
    ctx=None if 'context_mapping' not in package else package['context_mapping'][context_cells(p1[c1],p2[c2])]
    actions1,actions2=source['actions1'][c1],source['actions2'][c2]
    if kind=='ALLOCATION':
        a,b,q=source_joint_law(source,c1,c2,1.)
        coef=np.where(ctx<0,0.,package['theta'][np.zeros(n,int) if receipt['mode']=='scalar' else np.maximum(ctx,0)])
        q*=coef[:,None];q[:,0]+=1-coef
        return a,b,q
    iswide=receipt.get('support')=='WIDE' or receipt.get('metadata',{}).get('support')=='WIDE'
    if kind=='TEMPLATE' or iswide:
        metadata=receipt.get('metadata');controlkind=receipt.get('kind')
        if kind=='TEMPLATE':width=metadata['maximum_actions_per_row']
        elif controlkind=='LOCAL':
            cls1,cls2=p1.argmax(1),p2.argmax(1)
            width=(1+max(np.sum(cls1==k) for k in np.unique(cls1)))*(1+max(np.sum(cls2==k) for k in np.unique(cls2)))
        else:width=metadata['maximum_actions_per_row']
        a=np.repeat(c1[:,None],width,1);b=np.repeat(c2[:,None],width,1);q=np.zeros((n,width));q[:,0]=1
        if controlkind=='LOCAL':
            desc=[]
            for k1 in np.unique(p1.argmax(1)):
                for k2 in np.unique(p2.argmax(1)):
                    desc.append({'class_pair':[k1,k2],'recipient1_templates':[-1]+np.flatnonzero(p1.argmax(1)==k1).tolist(),'recipient2_templates':[-1]+np.flatnonzero(p2.argmax(1)==k2).tolist()})
        else:desc=metadata['contexts']
        lookup=None if 'second_keys' not in package else {tuple(map(int,key)):j for j,key in enumerate(package['second_keys'])}
        for d in desc:
            g=d.get('context');selected=np.flatnonzero((p1[c1].argmax(1)==d['class_pair'][0])&(p2[c2].argmax(1)==d['class_pair'][1])) if controlkind=='LOCAL' else np.flatnonzero(ctx==g)
            if not len(selected):continue
            l=np.asarray(d['recipient1_templates'],int);r=np.asarray(d['recipient2_templates'],int);count=len(l)*len(r)
            ii=np.repeat(l,len(r));jj=np.tile(r,len(l))
            if metadata is not None and metadata['support']=='NARROW':
                aa=actions1[selected][:,ii];bb=actions2[selected][:,jj]
                executable=source['valid1'][c1[selected]][:,ii]&source['valid2'][c2[selected]][:,jj]
            else:
                aa=np.broadcast_to(ii,(len(selected),count)).copy();bb=np.broadcast_to(jj,(len(selected),count)).copy();aa[:,ii==-1]=c1[selected,None];bb[:,jj==-1]=c2[selected,None];executable=np.ones(aa.shape,bool)
            if kind=='TEMPLATE':mass=np.broadcast_to(package['theta'][d['parameter_offset']:d['parameter_offset']+count],aa.shape)
            elif controlkind=='DETERMINISTIC':mass=np.broadcast_to(package['first_weights'][d['parameter_offset']:d['parameter_offset']+count],aa.shape)
            elif controlkind in ('LOCAL','PRODUCT'):
                rows1=c1[selected] if controlkind=='LOCAL' else np.repeat(g,len(selected));rows2=c2[selected] if controlkind=='LOCAL' else rows1
                mass=(package['first_weights'][rows1,:len(l),None]*package['second_weights'][rows2,None,:len(r)]).reshape(len(selected),count)
            else:
                first=receipt['first_recipient'];axis=l if first==1 else r;secondwidth=len(r) if first==1 else len(l)
                firstcodes=np.broadcast_to(axis,(len(selected),len(axis))).copy();firstcodes[:,axis==-1]=(c1 if first==1 else c2)[selected,None]
                secondgroups=np.array([[lookup.get((g,int(code)),lookup[(g,-1)]) for code in row] for row in firstcodes])
                z=package['first_weights'][g,:len(axis)][None,:,None]*package['second_weights'][secondgroups,:secondwidth]
                mass=(z if first==1 else z.transpose(0,2,1)).reshape(len(selected),count)
            okay=~np.any((mass>0)&~executable,axis=1);selected=selected[okay]
            a[selected,:count]=aa[okay];b[selected,:count]=bb[okay];q[selected]=0;q[selected,:count]=mass[okay]
        return a,b,q
    # Original four-neighbour controls.
    a=np.repeat(actions1,4,1);b=np.tile(actions2,(1,4));q=np.zeros((n,16));q[:,0]=1
    controlkind=receipt['kind'];first=package['first_weights'];second=package.get('second_weights')
    if controlkind=='LOCAL':q=(first[c1,:,None]*second[c2,None,:]).reshape(n,16)
    else:
        sel=np.flatnonzero(ctx>=0);g=ctx[sel]
        if controlkind=='DETERMINISTIC':q[sel]=first[g]
        elif controlkind=='PRODUCT':q[sel]=(first[g,:,None]*second[g,None,:]).reshape(len(sel),16)
        else:
            rec=receipt['first_recipient'];codes=actions1[sel] if rec==1 else actions2[sel]
            lookup={tuple(map(int,key)):j for j,key in enumerate(package['second_keys'])}
            groups=np.array([[lookup[(int(group),int(code))] for code in row] for group,row in zip(g,codes)])
            z=first[g,:,None]*second[groups]
            q[sel]=(z if rec==1 else z.transpose(0,2,1)).reshape(len(sel),16)
    return a,b,q


def checked_law(source,c1,c2,*args,**kwargs):
    a,b,q=law_from_numbers(source,c1,c2,*args,**kwargs)
    if np.min(q,initial=0)<0 or np.max(np.abs(q.sum(1)-1),initial=0)>1e-8:raise ValueError('serialized policy mass')
    for p,old,out in ((source['probability1'],c1,a),(source['probability2'],c2,b)):
        if np.any((p[out].argmax(2)!=p[old].argmax(1)[:,None])&(q>0)):raise ValueError('serialized policy class preservation')
    return a,b,q

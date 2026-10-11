"""Independent cached-release HMAC domains and actual emitted-tuple CDFs."""
import hashlib,hmac,math


def uniform(secret,kind,policy_sha,source_binding,seed,key,recipient=None):
    domain=('CBR1|'+kind+'|'+policy_sha+'|'+source_binding+'|'+str(seed)).encode()
    cachekey=hmac.new(secret,domain+b'|key|'+str(key).encode(),hashlib.sha256).hexdigest()
    sub=b'joint' if recipient is None else ('recipient'+str(recipient)).encode()
    token=hmac.new(secret,domain+b'|draw|'+sub+b'|'+cachekey.encode(),hashlib.sha256).digest()
    return (int.from_bytes(token[:8],'big')//2048)/(2**53)


def draw_ordered(values,weights,u):
    if not 0<=u<1:raise ValueError('uniform outside [0,1)')
    cumulative=0.;last=None
    for value,weight in zip(values,weights):
        if weight<0:raise ValueError('negative sampling weight')
        cumulative+=float(weight)
        if weight>0:last=value
        if u<cumulative:return value
    if last is None:raise ValueError('empty sampling support')
    return last


def draw_marginal(law,recipient,u):
    masses={}
    for pair,mass in law.items():masses[pair[recipient]]=masses.get(pair[recipient],0.)+mass
    total=math.fsum(masses.values());values=sorted(masses)
    return draw_ordered(values,[masses[v]/total for v in values],u)


def sampled_pair(a,b,mass,kind,u_joint,u1,u2):
    if kind=='joint':return draw_ordered(list(zip(map(int,a),map(int,b))),mass,u_joint)
    law={}
    for x,y,w in zip(a,b,mass):
        if w>0:law[int(x),int(y)]=law.get((int(x),int(y)),0.)+float(w)
    if kind=='deterministic':
        if len(law)!=1:raise ValueError('nondeterministic law')
        return next(iter(law))
    if kind=='product':
        x=draw_marginal(law,0,u1);y=draw_marginal(law,1,u2)
        p1={};p2={}
        for (aa,bb),w in law.items():p1[aa]=p1.get(aa,0.)+w;p2[bb]=p2.get(bb,0.)+w
        if any(abs(law.get((aa,bb),0)-v1*v2)>1e-10 for aa,v1 in p1.items() for bb,v2 in p2.items()):raise ValueError('law not product')
        return x,y
    if kind=='seq12':
        x=draw_marginal(law,0,u1);subset={pair:w for pair,w in law.items() if pair[0]==x}
        return x,draw_marginal(subset,1,u2)
    if kind=='seq21':
        y=draw_marginal(law,1,u2);subset={pair:w for pair,w in law.items() if pair[1]==y}
        return draw_marginal(subset,0,u1),y
    raise ValueError('unknown sampler')

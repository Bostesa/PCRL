"""Raw83 to original recipient wires, with one private cached draw per person.

Person cache keys do not enter the mechanism's inputs or emitted metadata.
Distinct cache keys use independently domain-separated pseudorandom uniforms.
"""
from __future__ import annotations
import hashlib,hmac,json,os,sqlite3
from pathlib import Path
import numpy as np
from .source import load_manifest,load_bank,pinned


def load_p0_service(manifest,seed):
    from incumbent_view.r1_deploy import P0Service
    record=manifest['seeds'][str(seed)]
    for item in record['pins'].values(): pinned(item)
    return P0Service(record['pins'],load_bank(manifest,seed))


def validate_law(bank,c1,c2,out1,out2,mass):
    c1,c2=np.asarray(c1),np.asarray(c2)
    a,b,q=np.asarray(out1),np.asarray(out2),np.asarray(mass)
    if (a.dtype!=np.int64 or b.dtype!=np.int64 or q.dtype!=np.float64 or
        a.ndim!=2 or b.shape!=a.shape or q.shape!=a.shape or len(a)!=len(c1) or
        not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1,rtol=0,atol=1e-12)):
        raise ValueError('invalid finite release law')
    for r,code,old in ((1,a,c1),(2,b,c2)):
        p=np.array(bank['recipients'][str(r)]['probability'],dtype=np.float64)
        if np.any(code<0) or np.any(code>=len(p)): raise ValueError('code outside bank')
        if np.any((p[code].argmax(2)!=p[old].argmax(1)[:,None]) & (q>0)):
            raise ValueError('release law changes decision')
    return a,b,q


class CachedRelease:
    """Provider-only persistent single-query cache. Never include keys in wire."""
    def __init__(self,service,law_codes,*,policy_sha256,cache_path,secret):
        if not isinstance(secret,bytes) or len(secret)<32: raise ValueError('private 256-bit sampling root required')
        self.service=service;self.law_codes=law_codes;self.secret=secret
        self.domain=('CBR1|'+policy_sha256+'|'+service.bank['source_binding_sha256']+'|'+str(service.seed)).encode()
        cache_path=Path(cache_path)
        cache_path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        if not cache_path.exists():
            fd=os.open(cache_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
        if cache_path.is_symlink() or cache_path.stat().st_mode&0o077:
            raise PermissionError('private cache permission required')
        self.db=sqlite3.connect(cache_path)
        self.db.execute('CREATE TABLE IF NOT EXISTS releases (key TEXT PRIMARY KEY, input_binding TEXT NOT NULL, code1 INTEGER NOT NULL, code2 INTEGER NOT NULL)')
        self.db.commit()

    def close(self):self.db.close()

    def _key(self,key):
        if not isinstance(key,(str,int,np.integer)):raise ValueError('provider-only canonical person key required')
        return hmac.new(self.secret,self.domain+b'|key|'+str(key).encode(),hashlib.sha256).hexdigest()

    def uniform(self,key):
        token=hmac.new(self.secret,self.domain+b'|draw|'+self._key(key).encode(),hashlib.sha256).digest()
        return (int.from_bytes(token[:8],'big')>>11)/float(2**53)

    def release(self,raw,names,person_keys):
        from incumbent_view.r1_bank import wires_for_codes
        x=np.asarray(raw)
        if len(person_keys)!=len(x):raise ValueError('cache keys must align with input rows')
        outputs,_=self.service.route_raw(x,names)
        c1,c2=outputs[0][0],outputs[1][0]
        a,b,q=validate_law(self.service.bank,c1,c2,*self.law_codes(c1,c2))
        e1=np.empty(len(x),np.int64);e2=np.empty(len(x),np.int64)
        # A transaction prevents two requests for one person creating two draws.
        with self.db:
            for i,key in enumerate(person_keys):
                cachekey=self._key(key)
                binding=hashlib.sha256(np.ascontiguousarray(x[i],dtype=np.float32).tobytes()).hexdigest()
                saved=self.db.execute('SELECT input_binding,code1,code2 FROM releases WHERE key=?',(cachekey,)).fetchone()
                if saved is not None:
                    if saved[0]!=binding:raise ValueError('same person key supplied with a different raw input')
                    e1[i],e2[i]=saved[1:];continue
                u=self.uniform(key)
                j=int(np.searchsorted(np.cumsum(q[i]),u,side='right'))
                if j>=len(q[i]) or q[i,j]<=0:j=int(np.flatnonzero(q[i]>0)[-1])
                e1[i],e2[i]=a[i,j],b[i,j]
                self.db.execute('INSERT INTO releases VALUES (?,?,?,?)',(cachekey,binding,int(e1[i]),int(e2[i])))
        return wires_for_codes(self.service.bank,1,e1),wires_for_codes(self.service.bank,2,e2)


def identity_law(c1,c2):
    return np.asarray(c1,dtype=np.int64)[:,None],np.asarray(c2,dtype=np.int64)[:,None],np.ones((len(c1),1),np.float64)


def historical_law(policy):
    def law(c1,c2):
        p=policy.space.prepare(c1,c2)
        return p['code_r1'],p['code_r2'],policy.law_prepared(p)
    return law

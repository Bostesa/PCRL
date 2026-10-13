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


def sample_tuple(a,b,q,u_joint,u1,u2,kind):
    """Sample actual tuples; product uses two independent recipient coins."""
    a,b,q=np.asarray(a),np.asarray(b),np.asarray(q)
    if kind=='joint':
        j=int(np.searchsorted(np.cumsum(q),u_joint,side='right'))
        if j>=len(q) or q[j]<=0:j=int(np.flatnonzero(q>0)[-1])
        return int(a[j]),int(b[j])
    pairs,inv=np.unique(np.c_[a,b],axis=0,return_inverse=True)
    weights=np.bincount(inv,weights=q,minlength=len(pairs));active=weights>0
    pairs,weights=pairs[active],weights[active]
    if kind=='deterministic':
        if len(pairs)!=1:raise ValueError('deterministic sampler given a nonconstant law')
        return int(pairs[0,0]),int(pairs[0,1])
    def draw(ids,mass,u):
        values,idx=np.unique(ids,return_inverse=True);m=np.bincount(idx,weights=mass,minlength=len(values));m=m/m.sum()
        slot=int(np.searchsorted(np.cumsum(m),u,side='right'))
        if slot>=len(m) or m[slot]<=0:slot=int(np.flatnonzero(m>0)[-1])
        return int(values[slot])
    if kind=='product':
        v1,i1=np.unique(pairs[:,0],return_inverse=True);v2,i2=np.unique(pairs[:,1],return_inverse=True)
        marginal1=np.bincount(i1,weights=weights);marginal2=np.bincount(i2,weights=weights)
        expected=marginal1[i1]*marginal2[i2]
        # Check every Cartesian emitted tuple, including omitted zero tuples.
        actual=np.zeros((len(v1),len(v2)));actual[i1,i2]=weights
        if not np.allclose(actual,marginal1[:,None]*marginal2[None,:],rtol=0,atol=1e-10):
            raise ValueError('independent-coin sampler requires an actual product law')
        return draw(pairs[:,0],weights,u1),draw(pairs[:,1],weights,u2)
    if kind=='seq12':
        first=draw(pairs[:,0],weights,u1);mask=pairs[:,0]==first
        return first,draw(pairs[mask,1],weights[mask],u2)
    if kind=='seq21':
        first=draw(pairs[:,1],weights,u2);mask=pairs[:,1]==first
        return draw(pairs[mask,0],weights[mask],u1),first
    raise ValueError('unknown sampler')


class CachedRelease:
    """Provider-only persistent single-query cache. Never include keys in wire."""
    def __init__(self,service,law_codes,*,policy_sha256,cache_path,secret,sampling_kind='joint'):
        if not isinstance(secret,bytes) or len(secret)<32: raise ValueError('private 256-bit sampling root required')
        if sampling_kind not in ('joint','product','seq12','seq21','deterministic'):raise ValueError('unregistered sampler')
        self.sampling_kind=sampling_kind
        self.service=service;self.law_codes=law_codes;self.secret=secret
        self.domain=('CBR1|'+sampling_kind+'|' +policy_sha256+'|'+service.bank['source_binding_sha256']+'|'+str(service.seed)).encode()
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

    def uniform(self,key,recipient=None):
        domain=b'joint' if recipient is None else ('recipient'+str(recipient)).encode()
        token=hmac.new(self.secret,self.domain+b'|draw|'+domain+b'|'+self._key(key).encode(),hashlib.sha256).digest()
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
                e1[i],e2[i]=sample_tuple(a[i],b[i],q[i],self.uniform(key),self.uniform(key,1),self.uniform(key,2),self.sampling_kind)
                self.db.execute('INSERT INTO releases VALUES (?,?,?,?)',(cachekey,binding,int(e1[i]),int(e2[i])))
        return wires_for_codes(self.service.bank,1,e1),wires_for_codes(self.service.bank,2,e2)


def identity_law(c1,c2):
    return np.asarray(c1,dtype=np.int64)[:,None],np.asarray(c2,dtype=np.int64)[:,None],np.ones((len(c1),1),np.float64)


def historical_law(policy):
    def law(c1,c2):
        p=policy.space.prepare(c1,c2)
        return p['code_r1'],p['code_r2'],policy.law_prepared(p)
    return law


def load_control_package(directory,manifest,receipt_sha256):
    from .source import sha_file
    receipt=Path(directory)/'POLICY_RECEIPT.json'
    if sha_file(receipt)!=receipt_sha256:raise PermissionError('control receipt differs before dispatch')
    schema=json.loads(receipt.read_text())['schema']
    if schema=='CBR1-control-policy-v1':
        from .controls import load_control_policy
        return load_control_policy(directory,manifest,expected_receipt_sha256=receipt_sha256)
    if schema=='CBR1-wide-control-policy-v1':
        from .control_wide_policy import load_wide_control_policy
        return load_wide_control_policy(directory,manifest,expected_receipt_sha256=receipt_sha256)
    raise PermissionError('unregistered control package schema')


def main(argv=None):
    import argparse
    from qpc.deploy import load_permitted_input
    from .source import sha_file,load_historical_policy
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--seed',type=int,choices=(0,1),required=True)
    parser.add_argument('--kind',choices=('P0','ALLOCATION','CONTROL','TEMPLATE','V_J','V_S21'),required=True)
    parser.add_argument('--policy-dir')
    parser.add_argument('--policy-receipt-sha256')
    parser.add_argument('--input',required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--person-keys',required=True)
    parser.add_argument('--person-keys-sha256',required=True)
    parser.add_argument('--sampling-root',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output-r1',required=True)
    parser.add_argument('--output-r2',required=True)
    args=parser.parse_args(argv)
    if args.output_r1==args.output_r2:raise ValueError('recipient outputs must differ')
    if any(Path(x).exists() for x in (args.output_r1,args.output_r2)):raise FileExistsError('new output paths required')
    if sha_file(args.input)!=args.input_sha256:raise PermissionError('raw input binding differs')
    if sha_file(args.person_keys)!=args.person_keys_sha256:raise PermissionError('provider cache keys binding differs')
    manifest=load_manifest(args.source_manifest,args.source_manifest_sha256)
    service=load_p0_service(manifest,args.seed)
    sampling_kind='joint'
    if args.kind in ('ALLOCATION','CONTROL','TEMPLATE'):
        if not args.policy_dir or not args.policy_receipt_sha256:raise ValueError('allocation requires bound policy')
        if args.kind=='ALLOCATION':
            from .fit import load_allocation_policy
            policy=load_allocation_policy(args.policy_dir,manifest,expected_receipt_sha256=args.policy_receipt_sha256)
        elif args.kind=='TEMPLATE':
            from .secondary import load_template_policy
            policy=load_template_policy(args.policy_dir,manifest,expected_receipt_sha256=args.policy_receipt_sha256)
        else:
            policy=load_control_package(args.policy_dir,manifest,args.policy_receipt_sha256)
            sampling_kind={'LOCAL':'product','PRODUCT':'product','DETERMINISTIC':'deterministic','SEQUENTIAL':'seq12' if policy.first_recipient==1 else 'seq21'}[policy.kind]
        receipt=json.loads((Path(args.policy_dir)/'POLICY_RECEIPT.json').read_text())
        if receipt['seed']!=args.seed:raise PermissionError('allocation source seed differs')
        law=policy.law_codes;binding=args.policy_receipt_sha256
    elif args.kind=='P0':
        if args.policy_dir or args.policy_receipt_sha256:raise ValueError('P0 cannot accept policy arguments')
        sampling_kind='deterministic'
        law=identity_law;binding=manifest['seeds'][str(args.seed)]['public_bank']['sha256']
    else:
        if args.policy_dir or args.policy_receipt_sha256:raise ValueError('historical policy paths come only from admitted manifest')
        sampling_kind='seq21' if args.kind=='V_S21' else 'joint'
        law=historical_law(load_historical_policy(manifest,args.seed,args.kind))
        binding=manifest['seeds'][str(args.seed)]['policies'][args.kind]['receipt']['sha256']
    raw=load_permitted_input(args.input,list(service.names))
    keys=json.loads(Path(args.person_keys).read_text())
    if not isinstance(keys,list):raise ValueError('provider person keys must be JSON list')
    root=Path(args.sampling_root)
    if root.is_symlink() or root.stat().st_mode&0o077:raise PermissionError('private sampling root permission required')
    secret=root.read_bytes()
    if len(secret)!=32:raise ValueError('private root must be exactly 32 bytes')
    api=CachedRelease(service,law,policy_sha256=binding,cache_path=args.cache,secret=secret,sampling_kind=sampling_kind)
    try:wires=api.release(raw,service.names,keys)
    finally:api.close()
    hashes=[]
    for path,view in zip((args.output_r1,args.output_r2),wires):
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as stream:
            for wire in view:stream.write(json.dumps(wire,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        hashes.append(sha_file(path))
    print(json.dumps({'schema':'CBR1-cached-raw83-deployment-receipt-v1','kind':args.kind,'seed':args.seed,
                      'rows':len(wires[0]),'policy_binding_sha256':binding,'wire_sha256':hashes,'sampling_kind':sampling_kind,
                      'contract':'one cached draw per provider person key; no transmitted cache key'},sort_keys=True))

if __name__=='__main__':main()

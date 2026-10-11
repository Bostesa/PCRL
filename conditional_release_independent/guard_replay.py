"""Independent frozen OOF reader predictions on every original code."""
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import admitted_arrays,digest


def source_guard_tables(source,seed):
    record=source['seeds'][str(seed)]['fixed_readers']
    if digest(record['index']['path'])!=record['index']['sha256']:raise PermissionError('guard index binding')
    index=json.loads(Path(record['index']['path']).read_text())
    with ZipFile(record['arrays']['path']) as z:names={n[:-4] for n in z.namelist()}
    arrays=admitted_arrays(record['arrays'],names);result={}
    bankentry=source['seeds'][str(seed)]['public_bank'];bank=json.loads(Path(bankentry['path']).read_text())
    if digest(bankentry['path'])!=bankentry['sha256']:raise PermissionError('guard bank binding')
    for i,reader in enumerate(index['readers']):
        if reader['view'] not in ('r1','r2'):continue
        recipient=int(reader['view'][-1]);prob=np.asarray(bank['recipients'][str(recipient)]['probability'],float);n,k=prob.shape;width=(n-1).bit_length();codes=np.arange(n)
        x=np.c_[np.eye(n),prob,np.eye(k)[prob.argmax(1)],((codes[:,None]>>np.arange(width))&1).astype(float)]
        family=reader['name'].rsplit('/',1)[1];table=np.empty((3,2,n))
        for fold in range(3):
            prefix=f'r{i}_f{fold}'
            if family=='prior':p=np.repeat(arrays[prefix][0],n)
            elif family=='categorical':
                parent=reader['name'].rsplit('/',1)[0]
                prioridx=next(j for j,r in enumerate(index['readers']) if r['name']==parent+'/prior')
                count=arrays[prefix];total=count.sum(1);p=np.repeat(arrays[f'r{prioridx}_f{fold}'][0],n);seen=total>0;p[seen]=(count[seen,1]+1)/(total[seen]+2)
            else:
                h=(x-arrays[prefix+'_mean'])/arrays[prefix+'_scale']
                for a,b in ((0,1),(2,3)):h=np.maximum(h@arrays[prefix+'_w'+str(a)]+arrays[prefix+'_w'+str(b)],0)
                z=(h@arrays[prefix+'_w4']+arrays[prefix+'_w5']).ravel();p=np.empty(n);pos=z>=0;p[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);p[~pos]=e/(1+e)
            p=np.clip(p,1e-12,1-1e-12);table[fold,0]=-np.log1p(-p);table[fold,1]=-np.log(p)
        result[reader['name']]={'view':reader['view'],'table':table,'source_loss':arrays[f'r{i}_loss']}
    return result

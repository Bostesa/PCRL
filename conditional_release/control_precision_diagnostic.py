"""Queued inspection of existing solver archives only; never fits or loads people."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from .scheduler import assert_active_from_env
from .source import sha_file,write_once


def run(config):
    active=assert_active_from_env(stage='verification');cfg=json.loads(Path(config).read_text())
    if active['unit_id']!=cfg['unit_id']:raise PermissionError('diagnostic unit differs')
    results=[]
    for record in cfg['archives']:
        if sha_file(record['path'])!=record['sha256']:raise PermissionError('existing archive changed')
        with np.load(record['path'],allow_pickle=False) as data:
            theta=data['theta'];full=data['primal'];eq=sparse.csr_matrix((data['equality_data'],data['equality_indices'],data['equality_indptr']),shape=tuple(data['equality_shape']))
            margin=eq@full-data['equality_rhs']
            result={'archive':Path(record['path']).name,'archive_sha256':record['sha256'],'parameters':len(theta),
                    'minimum_theta':float(theta.min()),'maximum_equality_error':float(np.abs(margin).max(initial=0)),
                    'primal_objective':float(data['objective']@full),
                    'negative_parameters':int((theta<0).sum()),'minimum_lower_bound_margin':float((theta-data['bounds'][:len(theta),0]).min())}
            results.append(result)
    write_once(cfg['public_output'],{'schema':'CBR1-control-precision-diagnostic-v1','status':'COMPLETE','unit_id':cfg['unit_id'],
               'results':results,'person_arrays_read':False,'fits':0,'scope':'Existing fixed solver primal/simplex residuals only'},False)
    print(json.dumps({'unit_id':cfg['unit_id'],'status':'COMPLETE'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)

"""Queue-only task utility on exact role-only archives; no SEX reads."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .r3_family import ARMS
from .r3_utility import (TASKS, METRICS, costs, evaluate_law, evaluate_realization,
                         decimal_fit_check, point_clause, stable_mean)
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

LAW_KEYS={"row_id","unit","code_r1","code_r2","mass","valid","b0_code_r1","b0_code_r2"}
WIRE_KEYS={"row_id","unit","code_r1","code_r2","prob_r1","prob_r2","decision_r1","decision_r2"}
DRAW_KEYS={"row_id","unit","code_r1","code_r2"}
REF_KEYS={"row_id","unit","U0_1","U0_2","Ucal_1","Ucal_2","d_1","d_2"}
CSV_FIELDS=("role","arm_id","source_seed","estimand","realization","task","reference","metric","candidate_value","reference_value","margin","cap","primary_gate","point_status")


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sha_array(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256();digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def read_exact(path, keys):
    # Check names before deserialization; no post-load masking.
    with ZipFile(path) as z:
        if len(z.infolist())!=len(keys) or {x.filename for x in z.infolist()}!={k+'.npy' for k in keys}:
            raise PermissionError('role-only member allowlist differs')
    with np.load(path,allow_pickle=False) as a:return {k:np.array(a[k],copy=True) for k in keys}


def same_people(item,row,unit):
    if not np.array_equal(item['row_id'],row) or not np.array_equal(item['unit'],unit):raise PermissionError('exact row/group order differs')


def identity_law(wire):
    n=len(wire['row_id'])
    return {'row_id':wire['row_id'],'unit':wire['unit'],
            'code_r1':wire['code_r1'][:,None],'code_r2':wire['code_r2'][:,None],
            'b0_code_r1':wire['code_r1'],'b0_code_r2':wire['code_r2'],
            'mass':np.ones((n,1),np.float64),'valid':np.ones((n,1),bool)}


def wire_check(wire,tables):
    for j in (1,2):
        code=wire[f'code_r{j}']
        if code.dtype!=np.int64 or np.any(code<0) or np.any(code>=len(tables[j-1])):
            raise PermissionError('P0 global ID differs')
        if not np.array_equal(wire[f'prob_r{j}'],tables[j-1][code]) or not np.array_equal(wire[f'decision_r{j}'],tables[j-1][code].argmax(1)):
            raise PermissionError('P0 wire code/decoder/decision differs')


def score(law,tables,labels,refs,draws,role,arm,seed,old_b0=None):
    evaluated=evaluate_law(law,tables,labels)
    numeric=decimal_fit_check(law,tables,labels,evaluated) if role=='OSF_DEFENSE_FIT' else None
    rows=[];vectors={}
    realization_cost=[evaluate_realization((v['code_r1'],v['code_r2']),tables,labels) for v in draws]
    for task,j,width in TASKS:
        reference_cost={'P0':evaluated[task]['p0']}
        for name in ('U0','Ucal'):
            p=refs[f'{name}_{j+1}']
            if p.dtype!=np.float64 or p.shape!=(len(labels[j]),width) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,rtol=0,atol=1e-8):
                raise PermissionError('reference probability schema differs')
            reference_cost[name]=costs(p,labels[j])
        if old_b0 is not None:reference_cost['OLD_B0']=costs(old_b0[f'prob{j}'],labels[j])
        for name,value in {'candidate':evaluated[task]['expected'],**{k:v for k,v in reference_cost.items() if k!='OLD_B0'}}.items():
            for m in METRICS:vectors[f'{name}_{task}_{m}']=value[m]
        estimands=[('exact_expected',-1,evaluated[task]['expected'])]+[('actual_single_draw',d,v[task]) for d,v in enumerate(realization_cost)]
        for estimand,draw,candidate in estimands:
            for reference,comparator in reference_cost.items():
                primary=(estimand=='exact_expected' and (reference=='P0' if role=='OSF_DEFENSE_FIT' else reference in ('U0','Ucal')))
                regime='fit' if role=='OSF_DEFENSE_FIT' and reference in ('P0','OLD_B0') else 'inner'
                for metric in METRICS:
                    difference=stable_mean(candidate[metric]-comparator[metric]);clause=point_clause(metric,difference,regime)
                    rows.append(dict(role=role,arm_id=arm,source_seed=seed,estimand=estimand,realization=draw,task=task,reference=reference,metric=metric,candidate_value=stable_mean(candidate[metric]),reference_value=stable_mean(comparator[metric]),margin=difference,cap=clause['cap'],primary_gate=primary,point_status=clause['status']))
    return rows,vectors,numeric


def run(config_path):
    active=assert_active_from_env()
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    sources={Path(v['path']).resolve():v['sha256'] for v in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(record):
        p=Path(record['path'])
        if p.is_symlink():raise PermissionError('source symlink refused')
        p=p.resolve(strict=True)
        if sources.get(p)!=record['sha256'] or sha(p)!=record['sha256']:raise PermissionError('active queue byte pin differs')
        return p
    config_path=Path(config_path).resolve(strict=True)
    if sources.get(config_path)!=sha(config_path):raise PermissionError('config unpinned')
    cfg=json.loads(config_path.read_text())
    arm,role,seed=cfg['arm_id'],cfg['role'],cfg['source_seed']
    if (cfg.get('schema')!='pcrl-incumbent-r3-utility-config-v1' or cfg['unit_id']!=active['unit_id'] or cfg['stage']!=active['stage']
        or arm not in ARMS or seed not in (0,1) or role not in ('OSF_DEFENSE_FIT','INNER_SELECTION')
        or cfg['stage'] not in ('witness','protection') or (cfg['stage']=='witness' and arm!='P0')):
        raise PermissionError('utility unit identity differs')
    paths={name:pinned(item) for name,item in cfg['pins'].items()}
    if cfg['pins']['seal']['sha256']!=active['seal_sha256']:raise PermissionError('seal differs')
    bank=json.loads(paths['public_bank'].read_text())
    if bank['schema']!='pcrl-incumbent-original-public-bank-v1' or bank['seed']!=seed:raise PermissionError('P0 bank differs')
    binding=bank['source_binding_sha256'];tables=tuple(np.array(bank['recipients'][str(j)]['probability'],np.float64) for j in (1,2))
    admission=json.loads(paths['p0_admission_receipt'].read_text())
    if admission['seed']!=seed or admission['status']!='PASS' or admission['public_bank_sha256']!=sha(paths['public_bank']):raise PermissionError('P0 source not admitted')
    release_receipt=None
    if arm!='P0' or 'law' in paths:
        release_receipt=json.loads(paths['release_receipt'].read_text())
        policy=json.loads(paths['policy_receipt'].read_text())
        if (policy.get('schema')!='IVR1-selected-policy-v1' or policy.get('arm_id')!=arm or policy.get('source_seed')!=seed
            or policy.get('public_bank_sha256')!=sha(paths['public_bank']) or policy.get('source_lock_sha256')!=active['seal_sha256']
            or release_receipt.get('schema')!='IVR1-selected-role-release-v1' or release_receipt.get('realization')!=0):
            raise PermissionError('selected policy/release identity differs')
        if (release_receipt['arm_id']!=arm or release_receipt['source_seed']!=seed or release_receipt['role']!=role
            or release_receipt['law_file_sha256']!=sha(paths['law']) or release_receipt['policy_receipt_sha256']!=sha(paths['policy_receipt'])
            or release_receipt['source_binding_sha256']!=binding or policy['source_binding_sha256']!=binding):
            raise PermissionError('selected policy/law/source provenance differs')
        if 'policy_package' in paths and policy.get('policy_package_sha256')!=sha(paths['policy_package']):raise PermissionError('policy package differs')
    # Only after source and policy checks, deserialize declared task-only archives.
    income=read_exact(paths['income'],{'row_id','unit','income'})
    occupation=read_exact(paths['occupation'],{'row_id','unit','occupation'})
    refs=read_exact(paths['references'],REF_KEYS)
    row,unit=income['row_id'],income['unit'];same_people(occupation,row,unit);same_people(refs,row,unit)
    if row.dtype!=np.int64 or unit.dtype!=np.int64 or len(row)!=cfg['rows'] or np.unique(unit).size!=cfg['groups']:raise PermissionError('role identity differs')
    pair_hash=sha_array(np.c_[row,unit])
    if pair_hash!=cfg['row_unit_pair_sha256']:raise PermissionError('role person order hash differs')
    wire=read_exact(paths['p0_wire'],WIRE_KEYS);same_people(wire,row,unit);wire_check(wire,tables)
    if sha(paths['p0_wire'])!=admission['roles'][role]['release_sha256']:raise PermissionError('P0 release differs from admission')
    law=read_exact(paths['law'],LAW_KEYS) if 'law' in paths else identity_law(wire)
    same_people(law,row,unit)
    if release_receipt is not None:
        content=hashlib.sha256(''.join(k+sha_array(law[k]) for k in sorted(law)).encode()).hexdigest()
        if content!=release_receipt['law_content_sha256'] or release_receipt['row_unit_pair_sha256']!=pair_hash:
            raise PermissionError('law content/person binding differs from R2 release')
    for j in (1,2):
        if not np.array_equal(law[f'b0_code_r{j}'],wire[f'code_r{j}']):raise PermissionError('law reference is not exact P0')
    draws=[]
    for d in range(3):
        key=f'draw{d}'
        draw=read_exact(paths[key],DRAW_KEYS) if key in paths else {k:wire[k] for k in DRAW_KEYS}
        same_people(draw,row,unit)
        if release_receipt is not None:
            dr=json.loads(paths[f'{key}_receipt'].read_text())
            if (dr.get('schema')!='IVR1-selected-role-release-v1' or dr['arm_id']!=arm or dr['source_seed']!=seed or dr['role']!=role
                or dr['realization']!=d or dr['policy_receipt_sha256']!=sha(paths['policy_receipt'])
                or dr['source_binding_sha256']!=binding or dr['codes_file_sha256']!=sha(paths[key])
                or dr['law_content_sha256']!=release_receipt['law_content_sha256'] or dr['row_unit_pair_sha256']!=pair_hash):
                raise PermissionError('cached actual release provenance differs')
            positive=((law['code_r1']==draw['code_r1'][:,None])&(law['code_r2']==draw['code_r2'][:,None])&(law['mass']>0)).any(1)
            if not positive.all():raise PermissionError('cached release outside selected positive actual law')
        draws.append(draw)
    old=read_exact(paths['old_b0_wire'],{'row_id','unit','code0','code1','prob0','prob1','decision0','decision1'}) if 'old_b0_wire' in paths else None
    if old is not None:same_people(old,row,unit)
    labels=(income['income'],occupation['occupation'])
    rows,vectors,numeric=score(law,tables,labels,refs,draws,role,arm,seed,old)
    if numeric is not None and any(v['numerical_admission']!='PASS' for v in numeric.values()):
        raise ArithmeticError('independent FIT numerical error exceeds sealed envelope')
    old_numeric=None
    if old is not None:
        old_doc=json.loads(paths['old_b0_table'].read_text())
        if old_doc['seed']!=seed or old_doc['bit_count']!=10:raise PermissionError('old B0 table identity differs')
        old_tables=tuple(np.asarray(old_doc['families']['W_TSTAR'][f'table{j}'],np.float64) for j in (0,1))
        ids=[]
        for j in (0,1):
            bits=old[f'code{j}']
            if bits.dtype!=np.uint8 or bits.shape!=(len(row),10) or not np.isin(bits,(0,1)).all():raise PermissionError('old B0 bits differ')
            code=np.sum(bits.astype(np.int64)*(1<<np.arange(10))[None,:],axis=1)
            if np.any(code>=len(old_tables[j])) or not np.array_equal(old[f'prob{j}'],old_tables[j][code]) or not np.array_equal(old[f'decision{j}'],old_tables[j][code].argmax(1)):raise PermissionError('old B0 code/table/decision differs')
            ids.append(code+len(tables[j]))
        diagnostic={**identity_law(wire)}
        for j in (1,2):
            diagnostic[f'code_r{j}']=np.c_[wire[f'code_r{j}'],ids[j-1]]
            diagnostic[f'b0_code_r{j}']=ids[j-1]
        diagnostic['mass']=np.tile(np.array([1.,0.]),(len(row),1))
        diagnostic['valid']=np.ones((len(row),2),bool)
        joined=tuple(np.concatenate((tables[j],old_tables[j]),axis=0) for j in (0,1))
        old_numeric=decimal_fit_check(diagnostic,joined,labels,evaluate_law(diagnostic,joined,labels))
    output=cfg['outputs'];owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    for key,value in output.items():
        p=Path(value).resolve();expected=owner/'.private' if key=='private_vectors' else owner
        if p.parent!=expected or p.exists() or p.is_symlink():raise PermissionError('owner output must be new')
    with os.fdopen(os.open(output['csv'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),'w') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS);writer.writeheader();writer.writerows(rows)
    vectors.update(row_id=row,unit=unit)
    with os.fdopen(os.open(output['private_vectors'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:np.savez_compressed(f,**vectors)
    primary=[v for v in rows if v['primary_gate']]
    receipt={'schema':'pcrl-incumbent-r3-utility-receipt-v1','unit_id':cfg['unit_id'],'arm_id':arm,'source_seed':seed,'role':role,
             'source_binding_sha256':binding,'source_unit_id':cfg['source_unit_id'],'policy_receipt_sha256':sha(paths['policy_receipt']) if 'policy_receipt' in paths else None,
             'law_file_sha256':sha(paths['law']) if 'law' in paths else None,'law_content_sha256':release_receipt.get('law_content_sha256') if release_receipt else None,
             'row_unit_pair_sha256':pair_hash,'csv_sha256':sha(output['csv']),'private_vectors_sha256':sha(output['private_vectors']),
             'source_pins':{name:item['sha256'] for name,item in cfg['pins'].items()},'primary_clauses':{'required':len(primary),'passed':sum(v['point_status']=='PASS' for v in primary)},
             'fit_numeric_diagnostics':numeric,'old_p0_vs_b0_fit_numeric_diagnostics':old_numeric,'queue':active,'sensitive_labels_read':False,'old_assessment_read':False,'new_calibration_fits':0}
    with os.fdopen(os.open(output['receipt'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),'w') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();r=run(a.config)
    print(json.dumps({'unit_id':r['unit_id'],'primary_clauses':r['primary_clauses']}))

if __name__=='__main__':main()

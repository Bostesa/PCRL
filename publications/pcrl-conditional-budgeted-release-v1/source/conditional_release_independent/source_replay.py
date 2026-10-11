"""Queued independent admission replay of role-only source numbers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .reconstruction import digest,admitted_arrays,folds,source_joint_law,source_seq21_law,validate_support
from .raw_forward import raw_p0

ROLE_FIELDS={'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'}


def run(config):
    # Shared semaphore is infrastructure, not scientific runner arithmetic.
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={i['path']:i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(entry):
        if registered.get(str(Path(entry['path']).absolute()))!=entry['sha256'] or digest(entry['path'])!=entry['sha256']:
            raise PermissionError('independent input not queued at exact hash')
        return entry
    if str(Path(config).absolute()) not in registered or digest(config)!=registered[str(Path(config).absolute())]:
        raise PermissionError('independent replay config unqueued')
    cfg=json.loads(Path(config).read_text())
    manifest=json.loads(Path(pinned(cfg['manifest'])['path']).read_text())
    if manifest['schema']!='CBR1-role-safe-private-source-v1':raise ValueError('role-only source schema')
    def check_entries(v):
        if isinstance(v,dict):
            if 'path' in v and 'sha256' in v:pinned(v)
            else:
                for item in v.values():check_entries(item)
    check_entries(manifest)
    role_sets={};reports=[]
    for seed in (0,1):
        record=manifest['seeds'][str(seed)]
        bank=json.loads(Path(record['public_bank']['path']).read_text())
        policies={}
        for name in ('V_J','V_S21'):
            receipt=json.loads(Path(record['policies'][name]['receipt']['path']).read_text())
            arrays=admitted_arrays(record['policies'][name]['package'],set(receipt['array_sha256']))
            for key,array in arrays.items():
                value=np.ascontiguousarray(array)
                arrayhash=__import__('hashlib').sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
                if arrayhash!=receipt['array_sha256'][key]:raise ValueError('source number identity differs')
            policies[name]=(receipt,arrays)
        for role,expected in (('OSF_DEFENSE_FIT',(15434,15428)),('ATTACK_FIT_NEW',(4064,4061)),('INNER_SELECTION',(2235,2234))):
            wire=admitted_arrays(record['roles'][role]['release'],ROLE_FIELDS)
            if (len(wire['row_id']),len(np.unique(wire['unit'])))!=expected:raise ValueError('role count differs')
            if role in role_sets:
                if not all(np.array_equal(wire[key],role_sets[role][key]) for key in ('row_id','unit')):raise ValueError('source seeds not paired')
            else:role_sets[role]={key:wire[key].copy() for key in ('row_id','unit')}
            features_=admitted_arrays(manifest['roles'][role]['features'],{'row_id','unit','X','feature_names'})
            if not all(np.array_equal(features_[key],wire[key]) for key in ('row_id','unit')):raise ValueError('raw feature pairing differs')
            output,scores=raw_p0(record['pins'],bank,features_['X'][:32],features_['feature_names'])
            for recipient in (1,2):
                p=output[recipient-1]
                if not np.array_equal(p['codes'],wire[f'code_r{recipient}'][:32]):raise ValueError('independent raw codes differ')
                if not np.array_equal(p['probabilities'],wire[f'prob_r{recipient}'][:32]):raise ValueError('independent decoded probability differs')
                if not np.array_equal(p['decisions'],wire[f'decision_r{recipient}'][:32]):raise ValueError('independent decisions differ')
            vj,arrays=policies['V_J']
            c1,c2,donor=source_joint_law(arrays,wire['code_r1'],wire['code_r2'],1.)
            support=validate_support(arrays,c1,c2,donor)
            seq,seqarrays=policies['V_S21']
            _,_,seqmass=source_seq21_law(seqarrays,wire['code_r1'],wire['code_r2'],seq['eta1'],seq['eta2'])
            validate_support(seqarrays,c1,c2,seqmass)
            fold=folds(wire['unit']) if role=='OSF_DEFENSE_FIT' else None
            item={'seed':seed,'role':role,'rows':expected[0],'groups':expected[1],
                  'raw83_rows_independently_replayed':32,'codes_probability_decisions':'EXACT',
                  'donor_mass_support':support,'historical_sequential_mass':'PASS',
                  'fold_counts':None if fold is None else [int(np.sum(fold==k)) for k in range(3)]}
            if role!='ATTACK_FIT_NEW':
                references=admitted_arrays(manifest['roles'][role][f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
                if not all(np.array_equal(references[key],wire[key]) for key in ('row_id','unit')):raise ValueError('reference pairing differs')
                item['raw_U0_maximum_gap']=[float(np.max(np.abs(scores[r-1]-references[f'U0_{r}'][:32]))) for r in (1,2)]
                if max(item['raw_U0_maximum_gap'])>1e-12:raise ValueError('raw U0 reference mismatch')
            reports.append(item)
    for a in role_sets:
        for b in role_sets:
            if a<b:
                for key in ('row_id','unit'):
                    if np.intersect1d(role_sets[a][key],role_sets[b][key]).size:raise ValueError('admitted roles overlap')
    report={'schema':'CBR1-independent-source-replay-v1','status':'PASS','unit_id':active['unit_id'],
            'source_execution_sha':manifest['source_execution_sha'],'source_publication_sha':manifest['source_publication_sha'],
            'source_manifest_sha256':cfg['manifest']['sha256'],'role_reports':reports,'role_disjointness':'PASS',
            'SEX_arrays_read':False,'task_label_arrays_read':False,'all_role_archive_read_or_hashed':False,
            'new_models_fitted':0,'scope':'Independent frozen raw83/codes on 32 rows per role/seed; full-role archived donor and historical sequential law normalization, source/fold pairing. Not full population or raw83 all-row replay.'}
    output=Path(cfg['output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','reports':len(reports),'raw83_rows':192,'output_sha256':digest(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)

"""Optional registered finer-conditioning branch with all primary laws retained.

The trigger can inspect accepted validation aggregates, never evaluation. The
target input is always Trisk, dictionary17; it is not selected from the trigger's
best-performing arm. Four-cell service partitions use only teacher/codebook
rows and preserve the original B median. No primary or branch-A file is edited.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .config import INPUTS, PRIMARY, configuration, digest, mechanism_id
from .data import array_hash
from .encoding import ServicePartitions
from .finite import ACCEPTANCE_TOLERANCES


BRANCH='C'
PRIORITY_BUDGETS=(.002,.0005,.01)
CONDITIONING_FAMILY='primary_intersect_fineC4'


def _run():
    from . import run
    return run


def _source_sha():return _run().sha(Path(__file__))


def robustness_specs():
    maps=[]
    for budget in PRIORITY_BUDGETS:
        for policy in ('L','C'):
            for anchor in configuration()['seeds']:
                name=mechanism_id('Trisk',policy,budget)+'_fineC'
                maps.append({'id':f'{name}/anchor_{anchor}','configuration':name,
                    'anchor':anchor,'input':'Trisk','policy':policy,'budget':budget,
                    'max_actions':17,'branch':BRANCH,'conditioning_family':CONDITIONING_FAMILY,
                    'matched_frontier':'Trisk_fineC_a17',
                    'paired_local_configuration':mechanism_id('Trisk','L',budget)+'_fineC',
                    'paired_coalition_configuration':mechanism_id('Trisk','C',budget)+'_fineC',
                    'original_constraints_retained':True,'coarse_witness':'none'})
    return maps


def _mandatory_t0():
    run=_run();receipts={}
    for budget in PRIORITY_BUDGETS:
        for policy in ('L','C'):
            for anchor in configuration()['seeds']:
                name=mechanism_id('T0',policy,budget)
                receipts[f'{name}/anchor_{anchor}']=run._audit_receipt(anchor,name)['registry_sha256']
    return receipts


def robustness_trigger():
    """Flag any accepted primary channel's full-H validation recovery breach.

    Each flag is one anchor/role/weighting, at its registered numerical budget.
    Original H-only selected models define the comparator. Zero-budget primary
    channels are eligible; unconstrained references have no privacy budget and
    cannot trigger this branch. The branch's input/family remains predetermined.
    """
    run=_run();mandatory=_mandatory_t0();violations=[];checked=[];unavailable=[]
    h_cache={}
    for spec in configuration()['maps']:
        if spec['policy'] not in ('L','C') or spec['budget'] is None:continue
        anchor,name,budget=spec['anchor'],spec['configuration'],float(spec['budget'])
        base=run.anchor_dir(anchor)
        if not ((base/'maps'/name/'ACCEPTED.json').exists()
                and (base/'audits'/name/'COMPLETE.json').exists()):
            unavailable.append(spec['id']);continue
        map_receipt=run._map_receipt(anchor,name)
        audit_receipt=run._audit_receipt(anchor,name)
        metadata=json.loads((base/'maps'/name/'metadata.json').read_text())
        if not metadata.get('feasible'):continue
        required=[f'{view}/{target}/{weight}' for view in (('A','AB') if spec['policy']=='C' else ('A',))
                  for target in ('SEX','RAC1P') for weight in ('U','W')]
        if metadata.get('budget')!=budget or sorted(metadata.get('constrained_roles',[]))!=sorted(required):
            raise ValueError('Accepted primary channel privacy schema differs from its trigger specification')
        information=metadata['independent_cmi']
        if any(not np.isfinite(information[key]) or information[key]>budget+ACCEPTANCE_TOLERANCES['cmi'] for key in required):
            continue
        if anchor not in h_cache:
            h_receipt=run._audit_receipt(anchor,'H')
            h_cache[anchor]=(json.loads((base/'audits/H/summary.json').read_text()),h_receipt['registry_sha256'])
        h,h_sha=h_cache[anchor]
        current=json.loads((base/'audits'/name/'summary.json').read_text())
        threshold=max(.003,2*budget)
        checked.append({'id':spec['id'],'map_sha256':map_receipt['sha256'],
                        'audit_registry_sha256':audit_receipt['registry_sha256']})
        for role in PRIMARY:
            for weighting in ('unweighted','weighted'):
                h_loss=h['attack:'+role]['validation'][weighting]
                loss=current['attack:'+role]['validation'][weighting]
                if not np.isfinite(h_loss) or not np.isfinite(loss):
                    raise ValueError('Nonfinite full-H validation trigger score')
                recovery=float(h_loss-loss)
                if recovery>threshold:
                    violations.append({'configuration':name,'anchor':anchor,'role':role,
                        'weighting':weighting,'budget':budget,'threshold':threshold,
                        'recovery':recovery,'H_registry_sha256':h_sha,
                        'audit_registry_sha256':audit_receipt['registry_sha256'],
                        'map_sha256':map_receipt['sha256']})
    return {'triggered':bool(violations),'violations':violations,
            'predetermined_input':'Trisk','predetermined_actions':17,
            'threshold_rule':'max(.003,2*budget)',
            'scope':'any accepted primary anchor/primary role/report weighting, validation only',
            'mandatory_T0_audit_registry_sha256':mandatory,
            'checked_channels':checked,'unavailable_channels':unavailable,'evaluation_data_used':False}


def _registration():
    run=_run();record=json.loads((run.OUT/'ROBUSTNESS_CONFIGS.json').read_text())
    expected={'schema':1,'branch':BRANCH,'registered':True,'primary_config_hash':digest(configuration()),
              'scientific_source_hashes':run.source_fingerprint(),'robustness_source_sha256':_source_sha(),
              'maps':robustness_specs(),'controls':[]}
    for key,value in expected.items():
        if record.get(key)!=value:raise ValueError(f'Frozen robustness registration changed: {key}')
    if record.get('registration_payload_hash')!=digest({k:v for k,v in record.items() if k!='registration_payload_hash'}):
        raise ValueError('Robustness registration hash mismatch')
    if not record.get('trigger',{}).get('triggered'):raise ValueError('Missing robustness trigger')
    return record


def register_robustness():
    run=_run();path=run.OUT/'ROBUSTNESS_CONFIGS.json'
    with run.unit_lock('branch/C/registration'):
        if path.exists():return _registration()
        schedule=json.loads((run.OUT/'RESOURCE_SCHEDULE.json').read_text())
        if (not schedule.get('frozen_before_comparative_outcomes')
                or schedule.get('config_hash')!=digest(configuration())):
            raise ValueError('A frozen primary resource schedule is required')
        trigger=robustness_trigger()
        if not trigger['triggered']:return {'registered':False,'branch':BRANCH,'trigger':trigger}
        for spec in robustness_specs():
            base=run.anchor_dir(spec['anchor'])
            if (base/'maps'/spec['configuration']).exists() or (base/'audits'/spec['configuration']).exists():
                raise ValueError('Robustness registration must precede affected attempts')
        record={'schema':1,'branch':BRANCH,'registered':True,'created_utc':run.now(),
            'primary_config_hash':digest(configuration()),'scientific_source_hashes':run.source_fingerprint(),
            'robustness_source_sha256':_source_sha(),'trigger':trigger,'maps':robustness_specs(),'controls':[],
            'nominal_extra_units':18,'original_maps_retained':True,
            'reused_controls':'primary17 U/withholding/RR/constant/independent/direct-code/teacher/H; same input and action alphabet',
            'conditioning':{'local_cells':4,'coalition_cells':8,'features':'H_A[:,[1,3]]',
                'fit_rows':'teacher_fit union teacher_internal_validation; no mechanism rows',
                'KMeans_n_init':10,'KMeans_random_state':'202609210+anchor',
                'center_order':'lexicographic','B_split':'original frozen H_B[:,1] median',
                'constraint_set':'intersection of original8 and fine8 empirical laws'},
            'primary_resource_schedule_sha256':run.sha(run.OUT/'RESOURCE_SCHEDULE.json')}
        record['registration_payload_hash']=digest(record);run.atomic(path,record)
        return record


def lookup_spec(name,anchor):
    run=_run();path=run.OUT/'ROBUSTNESS_CONFIGS.json'
    if not name.endswith('_fineC') or not path.exists():return None
    registration=_registration()
    raw=next((s for s in registration['maps'] if s['configuration']==name and s['anchor']==anchor),None)
    return None if raw is None else {**raw,'robustness_registration_sha256':run.sha(path),
                                    'robustness_source_sha256':registration['robustness_source_sha256']}


def freeze_robustness_schedule(unit_ids=None,*,resource_decision):
    run=_run();registration=_registration();known={s['id']:s for s in registration['maps']}
    ids=list(known) if unit_ids is None else list(unit_ids)
    if not ids or len(set(ids))!=len(ids) or not set(ids).issubset(known):
        raise ValueError('Unknown or duplicate robustness unit')
    budgets={known[unit]['budget'] for unit in ids}
    prefix=set(PRIORITY_BUDGETS[:len(budgets)])
    expected={s['id'] for s in registration['maps'] if s['budget'] in budgets}
    if budgets!=prefix or set(ids)!=expected:
        raise ValueError('Robustness schedule requires matched L/C three-anchor budget blocks in priority order')
    if not isinstance(resource_decision,dict) or not resource_decision:raise ValueError('Explicit resource decision required')
    path=run.OUT/'ROBUSTNESS_RESOURCE_SCHEDULE.json'
    with run.unit_lock('branch/C/resource_schedule'):
        if path.exists():raise FileExistsError('Robustness resource schedule is already frozen')
        for spec in registration['maps']:
            base=run.anchor_dir(spec['anchor'])
            if (base/'maps'/spec['configuration']).exists() or (base/'audits'/spec['configuration']).exists():
                raise ValueError('Robustness resource schedule must precede affected attempts')
        record={'schema':1,'branch':BRANCH,'created_utc':run.now(),
            'robustness_registration_sha256':run.sha(run.OUT/'ROBUSTNESS_CONFIGS.json'),
            'primary_resource_schedule_sha256':run.sha(run.OUT/'RESOURCE_SCHEDULE.json'),
            'frozen_before_affected_outcomes':True,'unit_ids':ids,'resource_decision':resource_decision}
        record['schedule_payload_hash']=digest(record);run.atomic(path,record)
        return record


def require_scheduled(spec):
    run=_run();path=run.OUT/'ROBUSTNESS_RESOURCE_SCHEDULE.json'
    record=json.loads(path.read_text())
    if (record.get('schedule_payload_hash')!=digest({k:v for k,v in record.items() if k!='schedule_payload_hash'})
            or not record.get('frozen_before_affected_outcomes')
            or record.get('robustness_registration_sha256')!=spec['robustness_registration_sha256']
            or record.get('primary_resource_schedule_sha256')!=run.sha(run.OUT/'RESOURCE_SCHEDULE.json')
            or spec['id'] not in record.get('unit_ids',[])):
        raise ValueError('Unit is not covered by the frozen robustness resource schedule')
    return run.sha(path)


def prepared_for_spec(anchor,prepared,spec,*,allow_fit=True):
    """Copy the encoder's service partition and augment, never replace, laws."""
    run=_run()
    if spec!=lookup_spec(spec['configuration'],anchor):raise ValueError('Frozen robustness specification required')
    if 'test' in prepared['ctx']['pools']:raise ValueError('Cannot fit conditioning on evaluation/test context')
    if prepared['ctx']['anchor']!=anchor:raise ValueError('Robustness preparation anchor mismatch')
    expected={'schema':1,'anchor':anchor,'branch':BRANCH,
        'prepared_cache_sha256':run._prepared_receipt(anchor)['cache_sha256'],
        'robustness_registration_sha256':spec['robustness_registration_sha256'],
        'robustness_source_sha256':spec['robustness_source_sha256'],
        'robustness_resource_schedule_sha256':require_scheduled(spec)}
    base=run.anchor_dir(anchor)/'branches/fineC';marker=base/'TABLES.json';cache=base/'tables.joblib'

    def load():
        record=json.loads(marker.read_text())
        for key,value in expected.items():
            if record.get(key)!=value:raise ValueError(f'Robustness table dependency changed: {key}')
        run._verify_artifacts(base,record);payload=joblib.load(cache)
        encoder=copy.copy(prepared['encoder']);encoder.partitions=payload['partitions']
        return {**prepared,'encoder':encoder,'tables':payload['tables']}

    if marker.exists():return load()
    if not allow_fit:raise FileNotFoundError('Frozen finer-conditioning tables are missing')
    with run.unit_lock(f'branch/C/tables/{anchor}'):
        if marker.exists():return load()
        run._quarantine([base],f'branch-C-tables-{anchor}');start=time.perf_counter()
        rf=prepared['ctx']['pools']['representation_fit'];roles=prepared['roles']
        rows=np.sort(np.r_[roles['teacher_fit'],roles['teacher_internal_validation']])
        if len(rows)<4 or len(np.unique(rows))!=len(rows):raise ValueError('Four-cell fit needs distinct teacher/codebook rows')
        if np.intersect1d(rows,roles['mechanism']).size:raise ValueError('Conditioning fit overlaps mechanism-estimation rows')
        partition=ServicePartitions.fit(rf['ha'][rows],rf['hb'][rows],anchor,n_local=4)
        if partition.b_median!=prepared['encoder'].partitions.b_median:
            raise ValueError('Finer conditioning changed the frozen B median')
        encoder=copy.copy(prepared['encoder']);encoder.partitions=partition
        from .mechanisms import estimate_tables
        fine=estimate_tables(prepared['ctx'],encoder,prepared['encoded'],roles,actions=17)
        tables={};cost_errors={}
        for family in INPUTS:
            original=prepared['tables'][family]
            if original['actions']!=17:raise ValueError('Finer conditioning requires unchanged dictionary17')
            for field in ('cost_U','cost_W','cost'):
                error=float(np.max(np.abs(original[field]-fine[family][field]),initial=0))
                cost_errors[family+'/'+field]=error
                if error>1e-12:raise AssertionError('Finer conditioning changed empirical utility costs')
            if not np.array_equal(original['state_mass'],fine[family]['state_mass']):
                raise AssertionError('Finer conditioning changed mechanism people')
            laws={**original['roles'],**{key+'/fineC':p for key,p in fine[family]['roles'].items()}}
            support={**original['support'],**{key+'/fineC':p for key,p in fine[family]['support'].items()}}
            tables[family]={**original,'roles':laws,'support':support,
                'aggregation':{'maximum_error':max(original['aggregation']['maximum_error'],fine[family]['aggregation']['maximum_error']),
                               'original':original['aggregation'],'fineC':fine[family]['aggregation']}}
        base.mkdir(parents=True,exist_ok=False)
        run.atomic_joblib(cache,{'partitions':partition,'tables':tables})
        run.atomic(marker,{**expected,'created_utc':run.now(),'seconds':time.perf_counter()-start,
            'teacher_codebook_rows':len(rows),'teacher_codebook_rows_sha256':array_hash(np.asarray(rf['raw_rows'])[rows]),
            'centers':partition.centers,'B_median':partition.b_median,
            'cost_maximum_error':max(cost_errors.values()),'cost_errors':cost_errors,
            'original_constraints_retained':True,'joint_laws_per_input':16,
            'artifact_hashes':run._artifact_hashes(base,[cache])})
        return {**prepared,'encoder':encoder,'tables':tables}

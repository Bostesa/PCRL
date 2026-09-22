"""Registered branch A without changing the frozen primary scientific modules.

The 33-action dictionary was frozen with the primary encoder. This module only
derives its empirical costs from the same mechanism rows after the registered
validation trigger and a separate resource schedule have both been frozen.
It never reads evaluation data or edits the primary preparation/configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np

from .config import INPUTS, configuration, digest, mechanism_id


BRANCH='A'
ACTION_COUNT=33
GAP_THRESHOLD=.001
PRIORITY_BUDGETS=(.002,.0005,.01)


def _run():
    from . import run
    return run


def _source_sha():
    return _run().sha(Path(__file__))


def action33_specs():
    """Full matched branch: 54 positive L/C maps and nine U references."""
    specs=[]
    for budget in PRIORITY_BUDGETS:
        for policy in ('L','C'):
            for family in INPUTS:
                for anchor in configuration()['seeds']:
                    name=mechanism_id(family,policy,budget,ACTION_COUNT)
                    specs.append({'id':f'{name}/anchor_{anchor}','configuration':name,
                        'anchor':anchor,'input':family,'policy':policy,'budget':budget,
                        'max_actions':ACTION_COUNT,'branch':BRANCH})
    for family in INPUTS:
        for anchor in configuration()['seeds']:
            name=mechanism_id(family,'U',None,ACTION_COUNT)
            specs.append({'id':f'{name}/anchor_{anchor}','configuration':name,
                'anchor':anchor,'input':family,'policy':'U','budget':None,
                'max_actions':ACTION_COUNT,'branch':BRANCH})
    return specs


def action33_controls():
    """Matched mixtures use the same larger unprotected action alphabet."""
    controls=[];cfg=configuration()
    for family in INPUTS:
        for mode,key in (('withhold','withholding_publish_probability'),
                         ('rr','randomized_response_deterministic_mix')):
            for rate in cfg[key]:
                canonical=f'{family}_{mode}_{rate:g}'
                for anchor in cfg['seeds']:
                    name=canonical+'_a33'
                    controls.append({'id':f'{name}/anchor_{anchor}','configuration':name,
                         'canonical_release':canonical,'anchor':anchor,'input':family,
                         'required_map':mechanism_id(family,'U',None,ACTION_COUNT),
                         'max_actions':ACTION_COUNT,'branch':BRANCH,'kind':'control',
                         'label_matched':True,'baseline_family':mode})
    for canonical in ('constant_best','independent_token'):
        for anchor in cfg['seeds']:
            name=canonical+'_a33'
            controls.append({'id':f'{name}/anchor_{anchor}','configuration':name,
                 'canonical_release':canonical,'anchor':anchor,
                 'required_map':mechanism_id('T0','U',None,ACTION_COUNT) if canonical=='constant_best' else None,
                 'max_actions':ACTION_COUNT,'branch':BRANCH,'kind':'control',
                 'label_matched':True,'baseline_family':canonical})
    return controls


def _mandatory_t0():
    run=_run();receipts={}
    for budget in PRIORITY_BUDGETS:
        for policy in ('L','C'):
            for anchor in configuration()['seeds']:
                name=mechanism_id('T0',policy,budget)
                record=run._audit_receipt(anchor,name)
                receipts[f'{name}/anchor_{anchor}']=record['registry_sha256']
    return receipts


def _fixed_decoder_score(anchor,name):
    run=_run();receipt=run._audit_receipt(anchor,name)
    path=run.anchor_dir(anchor)/'audits'/name/'summary.json'
    summary=json.loads(path.read_text())
    score=summary['utility:A/same_residence']['fixed_decoder']['balanced']
    if isinstance(score,bool) or not np.isfinite(score):
        raise ValueError('Branch trigger requires a finite direct-decoder balanced CE')
    return float(score),receipt['registry_sha256']


def action33_trigger():
    """Read accepted validation aggregates only, after all mandatory T0 audits.

    Every available family uses all three anchors equally. Incomplete families
    are disclosed and cannot trigger from a selected single anchor. The direct
    frozen teacher/decoder losses are used, never selected independent probes.
    """
    run=_run();mandatory=_mandatory_t0();anchors=configuration()['seeds']
    teacher={anchor:_fixed_decoder_score(anchor,'continuous_task') for anchor in anchors}
    families={};unavailable={}
    for family in INPUTS:
        name=mechanism_id(family,'U',None)
        missing=[anchor for anchor in anchors
                 if not (run.anchor_dir(anchor)/'audits'/name/'COMPLETE.json').exists()]
        if missing:
            unavailable[family]={'missing_anchors':missing};continue
        scores={anchor:_fixed_decoder_score(anchor,name) for anchor in anchors}
        by_anchor={str(anchor):{'unprotected_fixed_balanced_ce':scores[anchor][0],
                    'teacher_direct_balanced_ce':teacher[anchor][0],
                    'balanced_gap':scores[anchor][0]-teacher[anchor][0],
                    'unprotected_registry_sha256':scores[anchor][1],
                    'teacher_registry_sha256':teacher[anchor][1]} for anchor in anchors}
        gap=float(np.mean([row['balanced_gap'] for row in by_anchor.values()]))
        families[family]={'by_anchor':by_anchor,'mean_balanced_gap':gap}
    if not families:raise ValueError('No complete three-anchor unprotected family is available')
    triggering=sorted(family for family,value in families.items()
                      if value['mean_balanced_gap']>GAP_THRESHOLD)
    return {'triggered':bool(triggering),'triggering_families':triggering,
            'threshold':GAP_THRESHOLD,'comparison':'unprotected fixed decoder minus direct frozen teacher',
            'aggregation':'equal mean of three anchor balanced validation CE differences',
            'family_gaps':families,'unavailable_families':unavailable,
            'mandatory_T0_audit_registry_sha256':mandatory,'evaluation_data_used':False}


def _registration():
    run=_run();path=run.OUT/'EXTRA_CONFIGS.json'
    record=json.loads(path.read_text())
    checks={'schema':1,'branch':BRANCH,'registered':True,
            'primary_config_hash':digest(configuration()),
            'scientific_source_hashes':run.source_fingerprint(),
            'branch_source_sha256':_source_sha(),'maps':action33_specs(),'controls':action33_controls()}
    for key,value in checks.items():
        if record.get(key)!=value:raise ValueError(f'Frozen branch registration changed: {key}')
    payload={k:v for k,v in record.items() if k!='registration_payload_hash'}
    if record.get('registration_payload_hash')!=digest(payload):
        raise ValueError('Frozen branch registration payload hash mismatch')
    if not record.get('trigger',{}).get('triggered'):
        raise ValueError('Registered branch lacks its validation trigger')
    return record


def register_action33():
    """Atomically freeze branch A after mandatory completion and its trigger.

    Calling again reads the identical frozen registration. False triggers are
    returned to the caller without creating an enabled extra configuration.
    """
    run=_run();path=run.OUT/'EXTRA_CONFIGS.json'
    with run.unit_lock('branch/A/registration'):
        if path.exists():return _registration()
        schedule_path=run.OUT/'RESOURCE_SCHEDULE.json'
        schedule=json.loads(schedule_path.read_text())
        if (not schedule.get('frozen_before_comparative_outcomes')
                or schedule.get('config_hash')!=digest(configuration())):
            raise ValueError('Primary resource schedule must already be frozen')
        trigger=action33_trigger()
        if not trigger['triggered']:return {'registered':False,'branch':BRANCH,'trigger':trigger}
        # A pre-existing extra result is never retrospectively registered.
        for spec in action33_specs()+action33_controls():
            base=run.anchor_dir(spec['anchor'])
            if (base/'maps'/spec['configuration']).exists() or (base/'audits'/spec['configuration']).exists():
                raise ValueError('Extra registration must precede affected Q/audit attempts')
        record={'schema':1,'branch':BRANCH,'registered':True,'created_utc':run.now(),
                'primary_config_hash':digest(configuration()),
                'scientific_source_hashes':run.source_fingerprint(),
                'branch_source_sha256':_source_sha(),
                'primary_resource_schedule_sha256':run.sha(schedule_path),
                'trigger':trigger,'maps':action33_specs(),'controls':action33_controls(),
                'nominal_extra_maps':63,'nominal_extra_controls':60,'nominal_extra_units':123,
                'original_maps_retained':True,'primary_preparation_mutated':False,
                'dictionary_source':'reserved33 frozen on primary teacher/code construction rows',
                'priority_budgets':list(PRIORITY_BUDGETS)}
        record['registration_payload_hash']=digest(record)
        run.atomic(path,record)
        return record


def lookup_spec(name,anchor):
    spec=lookup_release(name,anchor)
    return spec if spec is not None and spec.get('kind')!='control' else None


def lookup_release(name,anchor):
    run=_run();path=run.OUT/'EXTRA_CONFIGS.json'
    # Primary/reference requests never acquire a dependency on extra branches.
    if not name.endswith('_a33') or not path.exists():return None
    registration=_registration()
    raw=next((spec for spec in registration['maps']+registration['controls']
              if spec['configuration']==name and spec['anchor']==anchor),None)
    if raw is None:return None
    return {**raw,'extra_registration_sha256':run.sha(path),
            'branch_source_sha256':registration['branch_source_sha256']}


def freeze_action33_schedule(unit_ids,*,resource_decision):
    """Freeze explicit extra units/capacity before any affected channel attempt."""
    run=_run();registration=_registration();path=run.OUT/'EXTRA_RESOURCE_SCHEDULE.json'
    ids=list(unit_ids);known={spec['id']:spec for spec in registration['maps']+registration['controls']}
    if not ids or len(set(ids))!=len(ids) or not set(ids).issubset(known):
        raise ValueError('Extra schedule requires unique registered unit IDs')
    if not isinstance(resource_decision,dict) or not resource_decision:
        raise ValueError('An explicit resource decision is required')
    for index,unit in enumerate(ids):
        spec=known[unit]
        if spec.get('kind')=='control':
            required=spec['required_map']
            if required is not None and required+f"/anchor_{spec['anchor']}" not in ids[:index]:
                raise ValueError('Each matched control requires its U33 map earlier in the schedule')
        elif spec['input']!='T0':
            coarse=mechanism_id('T0',spec['policy'],spec['budget'],ACTION_COUNT)+f"/anchor_{spec['anchor']}"
            if coarse not in ids[:index]:raise ValueError('Each refined unit requires its coarse witness earlier in the schedule')
    with run.unit_lock('branch/A/resource_schedule'):
        if path.exists():raise FileExistsError('Extra resource schedule is already frozen')
        for spec in registration['maps']+registration['controls']:
            base=run.anchor_dir(spec['anchor'])
            if (base/'maps'/spec['configuration']).exists() or (base/'audits'/spec['configuration']).exists():
                raise ValueError('Extra schedule must precede affected Q attempts')
        record={'schema':1,'branch':BRANCH,'created_utc':run.now(),
                'extra_registration_sha256':run.sha(run.OUT/'EXTRA_CONFIGS.json'),
                'frozen_before_affected_outcomes':True,'unit_ids':ids,
                'resource_decision':resource_decision,
                'primary_resource_schedule_sha256':run.sha(run.OUT/'RESOURCE_SCHEDULE.json')}
        record['schedule_payload_hash']=digest(record)
        run.atomic(path,record)
        return record


def require_scheduled(spec):
    run=_run();path=run.OUT/'EXTRA_RESOURCE_SCHEDULE.json'
    record=json.loads(path.read_text())
    payload={k:v for k,v in record.items() if k!='schedule_payload_hash'}
    if (record.get('schedule_payload_hash')!=digest(payload)
            or not record.get('frozen_before_affected_outcomes')
            or record.get('extra_registration_sha256')!=spec['extra_registration_sha256']
            or record.get('primary_resource_schedule_sha256')!=run.sha(run.OUT/'RESOURCE_SCHEDULE.json')
            or spec['id'] not in record.get('unit_ids',[])):
        raise ValueError('Branch unit is not covered by the frozen extra resource schedule')
    return run.sha(path)


def prepared_for_spec(anchor,prepared,spec,*,allow_fit=True):
    """Return a shallow context copy with separate 33-action empirical tables."""
    run=_run();expected=lookup_spec(spec['configuration'],anchor)
    if spec!=expected:raise ValueError('Only the frozen registered branch specification is accepted')
    if 'test' in prepared['ctx']['pools']:
        raise ValueError('Branch tables cannot be built from evaluation/test context')
    if prepared['ctx']['anchor']!=anchor:raise ValueError('Branch preparation anchor mismatch')
    schedule_sha=require_scheduled(spec)
    primary=run._prepared_receipt(anchor)
    base=run.anchor_dir(anchor)/'branches/action33';marker=base/'TABLES.json';cache=base/'tables.joblib'
    expected_metadata={'schema':1,'branch':BRANCH,'anchor':anchor,
             'prepared_cache_sha256':primary['cache_sha256'],
             'extra_registration_sha256':spec['extra_registration_sha256'],
             'branch_source_sha256':spec['branch_source_sha256'],
             'extra_resource_schedule_sha256':schedule_sha}

    def load():
        record=json.loads(marker.read_text())
        for key,value in expected_metadata.items():
            if record.get(key)!=value:raise ValueError(f'Branch table dependency changed: {key}')
        run._verify_artifacts(base,record)
        return {**prepared,'tables':joblib.load(cache)}

    if marker.exists():return load()
    if not allow_fit:raise FileNotFoundError('Frozen branch tables are missing')
    with run.unit_lock(f'branch/A/tables/{anchor}'):
        if marker.exists():return load()
        if ACTION_COUNT not in prepared['encoder'].dictionaries:
            raise ValueError('Primary encoder did not reserve the 33-action dictionary')
        run._quarantine([base],f'branch-A-tables-{anchor}')
        from .mechanisms import estimate_tables
        start=time.perf_counter()
        tables=estimate_tables(prepared['ctx'],prepared['encoder'],prepared['encoded'],
                               prepared['roles'],actions=ACTION_COUNT)
        errors={}
        for family in INPUTS:
            original=prepared['tables'][family];expanded=tables[family]
            if original['actions']!=17:raise ValueError('Branch derivation requires original primary tables')
            if not np.array_equal(original['state_mass'],expanded['state_mass']):
                raise AssertionError('Branch changed original mechanism people')
            for role,law in original['roles'].items():
                error=float(np.max(np.abs(law-expanded['roles'][role]),initial=0))
                errors[family+'/'+role]=error
                if error>1e-12:raise AssertionError('Action expansion changed an empirical privacy law')
        base.mkdir(parents=True,exist_ok=False)
        run.atomic_joblib(cache,tables)
        run.atomic(marker,{**expected_metadata,'created_utc':run.now(),
                   'seconds':time.perf_counter()-start,'privacy_law_maximum_error':max(errors.values()),
                   'privacy_law_errors':errors,'artifact_hashes':run._artifact_hashes(base,[cache])})
        return {**prepared,'tables':tables}

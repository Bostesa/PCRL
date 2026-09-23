"""Rebuild unique-unit status from frozen receipts and aggregate registers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


SUPPLEMENTARY = (
    'exchange_r01', 'exchange_r01_audit',
    'exchange_r01_deterministic', 'exchange_r01_deterministic_audit',
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def complete_receipt(directory: Path) -> dict:
    receipt_path = directory / 'COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get('unit_id') or not receipt.get('queue_sha256') or not receipt.get('artifacts'):
        raise ValueError(f'incomplete supplementary receipt: {directory}')
    for name, expected in receipt['artifacts'].items():
        artifact = (directory / name).resolve()
        if not artifact.is_relative_to(directory.resolve()) or not artifact.is_file() or sha(artifact) != expected:
            raise ValueError(f'supplementary artifact mismatch: {directory}/{name}')
    return {'unit_id':receipt['unit_id'], 'complete_sha256':sha(receipt_path),
            'artifact_count':len(receipt['artifacts'])}


def generate(root: Path, output: Path) -> dict:
    root = root.resolve(); output = output.resolve()
    sys.path.insert(0, str(root))
    from experiments.pcrl_task_aligned_cuts_v1 import scheduler
    base = root / 'results/pcrl_task_aligned_cuts_v1'
    existing = json.loads(output.read_text())
    core_path = base / 'CORE_FACTORIAL.csv'
    queue_path = base / 'RUN_QUEUE.json'
    core = list(csv.DictReader(core_path.open()))
    units = {}
    for row in core:
        key = row['configuration']
        if key in units:
            prior = units[key]
            for field in ('fit_status','audit_status','registered_feasible','fallback','release_sha256'):
                if row[field] != prior[field]:
                    raise ValueError(f'inconsistent core configuration rows: {key}')
        else:
            units[key] = row
    if len(core) != 360 or len(units) != 36 or any(sum(x['configuration']==key for x in core)!=10 for key in units):
        raise ValueError('core endpoint enumeration is incomplete')
    completed = [u for u in units.values() if u['audit_status']=='complete']
    untriggered = [u for u in units.values() if u['audit_status']=='not_triggered']
    counts = {
        'nominal_configurations':len(units),
        'fit_complete':len(completed),
        'audit_complete':len(completed),
        'registered_feasible':sum(u['registered_feasible']=='True' for u in completed),
        'infeasible_fallback':sum(u['fallback']=='True' for u in completed),
        'not_triggered':len(untriggered),
        'audit_running':0, 'audit_failed':0,
        'exact_aliases':sum(bool(u['alias_of']) for u in completed),
    }
    if counts != existing['counts'] or sha(core_path) != existing['core_factorial_sha256']:
        raise ValueError('core report changed from frozen status')
    queue = json.loads(queue_path.read_text())
    queue_sha = sha(queue_path)
    state = json.loads((base/'private/QUEUE_STATUS.json').read_text())
    if queue_sha != existing['queue_sha256'] or queue_sha != state['queue_sha256']:
        raise ValueError('main queue hash mismatch')
    declarations = {unit['id']:unit for unit in queue['units']}
    if len(declarations)!=len(queue['units']) or set(state['units'])-set(declarations):
        raise ValueError('main queue unit identities invalid')
    for ident, unit_state in state['units'].items():
        marker = scheduler._complete_marker(root, declarations[ident], queue_sha)
        if (unit_state.get('status') != 'complete' or marker is None
                or marker['marker_sha256'] != unit_state['completion']['marker_sha256']):
            raise ValueError(f'main queue unit not fully verified: {ident}')
    unselected = sorted(set(declarations)-set(state['units']))
    if any(declarations[ident]['tier'] != 'B' for ident in unselected):
        raise ValueError('unselected main units include mandatory Tier A work')
    sidecar_path = base/'agents/audit_review/CONTROL_AUDIT_SIDECAR.json'
    sidecar = json.loads(sidecar_path.read_text()); sidecar_sha = sha(sidecar_path)
    side_state = json.loads((base/'private/SIDECAR_QUEUE_STATUS.json').read_text())
    if side_state['sidecar_sha256'] != sidecar_sha:
        raise ValueError('control sidecar hash mismatch')
    all_registered = {**sidecar['simple_map_audit_units'], **sidecar['fitted_control_audits']}
    distinct = {record['release_id']:record for record in all_registered.values()
                if record['status']=='registered_unrun'}
    aliases = [record for record in all_registered.values()
               if record['status']=='aliased_to_registered_unit']
    if len(side_state['units']) != len(distinct):
        raise ValueError('control receipt count differs from distinct registrations')
    for ident, record in distinct.items():
        verified = scheduler._control_sidecar_marker(
            root, {'id':ident, 'output_dir':record['output_relative_dir'],
                   'source_file_sha256':record['source_file_sha256'],
                   'source_array_sha256':record['source_array_sha256']},
            sidecar, sidecar_sha)
        state_unit = side_state['units'].get(ident)
        if (not verified or not state_unit or state_unit.get('status')!='complete'
                or verified['marker_sha256'] != state_unit['completion']['marker_sha256']):
            raise ValueError(f'control receipt incomplete or changed: {ident}')
    supplementary = {name:complete_receipt(base/'private'/name) for name in SUPPLEMENTARY}
    c2_path = base/'C2_NOT_TRIGGERED.json'; c2=json.loads(c2_path.read_text())
    if (c2['candidate_round_units_executed'] != 0
            or len(c2['candidate_round_units_not_triggered']) != 9
            or len(set(c2['candidate_round_units_not_triggered'])) != 9):
        raise ValueError('C2 contingency counts changed')
    inference_path = base/'INFERENCE.json'; inference=json.loads(inference_path.read_text())
    if (inference['outer_assessment_opened'] is not False
            or any(inference[name] for name in ('outer_assessment_fit_units',
                  'outer_assessment_score_units','outer_assessment_endpoints','distinct_outer_comparisons'))):
        raise ValueError('outer assessment status changed')
    incident_path = base/'agents/release_math/EXCHANGED_MILP_INCIDENT.json'
    incident=json.loads(incident_path.read_text()); attempts=incident['attempts']
    if (len(attempts)!=2 or attempts[0].get('optimizer_started') is not False
            or attempts[0].get('output_created') is not False
            or attempts[0].get('exit_code')==0 or attempts[1].get('exit_code')!=0
            or attempts[1].get('optimizer_started') is not True):
        raise ValueError('MILP technical incident does not match recorded safe retry')
    package_path=base/'private/packages/PACKAGE_RECEIPT.json'
    package=json.loads(package_path.read_text())
    if package['schema']!='pcrl-packaged-channels-v1' or len(package['packages'])!=5:
        raise ValueError('final package receipt incomplete')
    for record in package['packages']:
        qpath=base/record['package_channel_relative_path']/'Q.npz'
        if sha(qpath)!=record['package_Q_file_sha256']:
            raise ValueError(f'package Q changed: {record["name"]}')
        with np.load(qpath,allow_pickle=False) as arrays:
            q=np.ascontiguousarray(arrays['Q'],dtype=np.float64)
        if q.shape!=(32,17) or sha(base/record['package_channel_relative_path']/'manifest.json')!=record['package_manifest_sha256']:
            raise ValueError(f'package shape/manifest mismatch: {record["name"]}')
        if hashlib.sha256(q.tobytes()).hexdigest()!=record['package_array_sha256'] or record['array_byte_identical'] is not True:
            raise ValueError(f'package array identity mismatch: {record["name"]}')
    updated={**existing,
        'schema':'pcrl-study-run-status-v2',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'core_endpoint_rows':len(core),
        'counting_note':'360 core CSV endpoint rows represent 36 configurations, not 360 fitted units',
        'main_queue':{'declared_units':len(declarations),'verified_complete':len(state['units']),
                      'not_triggered':len(unselected),'failed':0,'queue_sha256':queue_sha},
        'control_sidecar':{'declared_labels':len(all_registered),'verified_distinct_complete':len(distinct),
                           'exact_aliases':len(aliases),'failed':0,'sidecar_sha256':sidecar_sha},
        'supplementary_units':supplementary,
        'supplementary_verified_complete':len(supplementary),
        'c2':{'candidate_rounds_executed':0,'candidate_rounds_not_triggered':len(c2['candidate_round_units_not_triggered']),
              'matched_control_rounds_executed':c2['matched_c2_control_units_executed'],
              'source_sha256':sha(c2_path)},
        'outer':{'fit_units':0,'score_units':0,'endpoints':0,'comparisons':0,
                 'opened':False,'source_sha256':sha(inference_path)},
        'technical_incident':{'kind':'pre-import Python path failure; optimizer not started',
                              'attempts':len(attempts),'same_configuration_retry_complete':True,
                              'source_sha256':sha(incident_path)},
        'packages':{'modes':len(package['packages']),'experimental_no_advantage':sum(
            record['status']=='EXPERIMENTAL_NO_ADVANTAGE' for record in package['packages']),
            'baseline_control':sum(record['status']=='BASELINE_CONTROL' for record in package['packages']),
            'receipt_sha256':sha(package_path)},
    }
    temp=output.with_name(output.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(updated,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(temp,output)
    return {'output_sha256':sha(output),'main_complete':len(state['units']),
            'control_complete':len(distinct),'supplementary_complete':len(supplementary),
            'outer_score_units':0}


if __name__=='__main__':
    root=Path.cwd(); path=root/'results/pcrl_task_aligned_cuts_v1/RUN_STATUS.json'
    print(json.dumps(generate(root,path),sort_keys=True))

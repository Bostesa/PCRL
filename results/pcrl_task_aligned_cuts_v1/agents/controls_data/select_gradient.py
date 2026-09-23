"""Apply the committed inner-selection-only matched-gradient rule.

The control audit report contains inner-pilot aggregates too, but this selector
indexes only ``candidate_selection`` fields. No pilot or outer loss determines
eligibility, ranking, tie breaking, or the chosen comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROLES = ('utility:A/same_residence', 'attack:A/SEX', 'attack:A/RAC1P',
         'attack:AB/SEX', 'attack:AB/RAC1P')
PROTECTED = ROLES[1:]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_scores(report: dict, role: str) -> dict:
    route = report['roles'][role]
    selection = route['candidate_selection']
    name = selection['selected']
    scores = selection['scores'][name]
    answer = {weighting: float(scores[weighting]) for weighting in ('U','PWGTP','balanced')}
    if (not all(math.isfinite(value) for value in answer.values())
            or abs(answer['balanced'] - (answer['U'] + answer['PWGTP'])/2) > 1e-10):
        raise ValueError(f'invalid balanced inner-selection loss: {role}')
    return {'selected_route': name,
            'selected_model_sha256': route['candidate_selected_model_sha256'],
            'exact_expected_ce': answer}


def select(root: Path, output: Path) -> dict:
    root = root.resolve(); output = output.resolve()
    if not output.is_relative_to(root):
        raise ValueError('selection output must remain in owned worktree')
    sys.path.insert(0, str(root))
    from experiments.pcrl_task_aligned_cuts_v1 import controls, scheduler, solver

    base = root / 'results/pcrl_task_aligned_cuts_v1'
    rule_path = base / 'CONTROL_SELECTION_RULE.json'
    rule = json.loads(rule_path.read_text())
    if (rule.get('schema') != 'pcrl-gradient-selection-v1'
            or 'inner_selection' not in rule.get('selection_pool','')
            or rule.get('task_score') !=
            'selected frozen utility:A/same_residence audit route exact expected-token CE, balanced = 0.5 U + 0.5 PWGTP, on inner_selection'):
        raise ValueError('gradient selection rule changed')
    sidecar_path = base / 'agents/audit_review/CONTROL_AUDIT_SIDECAR.json'
    sidecar = json.loads(sidecar_path.read_text())
    sidecar_sha = sha(sidecar_path)
    if sidecar.get('outer_pool_opened') is not False:
        raise ValueError('outer pool already opened')
    control_dir = base / 'private/controls/a0_u1p1_z000_initial_bank'
    control_path = control_dir / 'CONTROLS.json'
    if sha(control_path) != sidecar['fitted_control_source_sha256']:
        raise ValueError('saved gradient grid changed')
    fitted = json.loads(control_path.read_text())
    cost, cuts, bank_hash = controls.load_saved_bank(
        base / 'private/run/a0_reference_bank', cost_dir=base / 'private/run/a0_u1p1_z000')
    if bank_hash != fitted['bank_sha256']:
        raise ValueError('gradient fixed bank differs from saved control report')
    registrations = sidecar['fitted_control_audits']
    if {f'GRADIENT_{i:03d}' for i in range(12)} - set(registrations):
        raise ValueError('registered 12-gradient selection family incomplete')
    records = []
    eligible_distinct = []
    for i in range(12):
        label = f'GRADIENT_{i:03d}'
        registration = registrations[label]
        canonical = registration['canonical_release_id']
        canonical_record = next((item for item in registrations.values()
                                 if item['release_id'] == canonical), None)
        if canonical_record is None:
            raise ValueError(f'gradient alias lacks canonical record: {label}')
        source = root / registration['source_relative_path']
        if sha(source) != registration['source_file_sha256']:
            raise ValueError(f'gradient source file changed: {label}')
        with np.load(source, allow_pickle=False) as arrays:
            q = np.asarray(arrays['Q'], dtype=np.float64)
        if controls._array_sha(q) != registration['source_array_sha256']:
            raise ValueError(f'gradient Q array changed: {label}')
        independent = solver.replay_p1(q, cost, cuts)
        saved = fitted['gradient'][i]['replay']
        if (abs(independent['objective'] - saved['objective']) > 1e-10
                or abs(independent['maximum_cut_violation']
                       - saved['maximum_cut_violation']) > 1e-10):
            raise ValueError(f'gradient frozen-bank replay mismatch: {label}')
        valid_simplex = (independent['simplex_residual'] <= 1e-8
                         and independent['minimum_entry'] >= -1e-10)
        feasible = valid_simplex and independent['maximum_cut_violation'] <= 1e-5
        output_dir = root / canonical_record['output_relative_dir']
        receipt = scheduler._control_sidecar_marker(
            root, {'id':canonical, 'output_dir':canonical_record['output_relative_dir'],
                   'source_file_sha256':canonical_record['source_file_sha256'],
                   'source_array_sha256':canonical_record['source_array_sha256']},
            sidecar, sidecar_sha)
        if receipt is None:
            raise ValueError(f'incomplete or invalid gradient audit receipt: {label}')
        report_path = output_dir / 'INNER_PANEL.json'
        report = json.loads(report_path.read_text())
        if (report['release_id'] != canonical or report['slate'] != sidecar['slate']
                or report['split_assignment_sha256'] != sidecar['inner_split_sha256']
                or report['selection_pool'] != sidecar['selection_pool']
                or report['outer_pool_opened'] is not False):
            raise ValueError(f'gradient audit selection scope differs: {label}')
        # Only frozen-route inner-selection scores enter this record and rule.
        task = selected_scores(report, ROLES[0])
        protected = {role:selected_scores(report, role) for role in PROTECTED}
        record = {
            'label':label, 'unit_id':registration['release_id'],
            'canonical_unit_id':canonical,
            'exact_Q_alias':registration['release_id'] != canonical,
            'source_array_sha256':registration['source_array_sha256'],
            'settings':{name:fitted['gradient'][i][name] for name in (
                'initialization','seed','penalty','learning_rate','dual_learning_rate',
                'steps','initialization_smoothing')},
            'fixed_bank_objective':independent['objective'],
            'fixed_bank_max_cut_violation':independent['maximum_cut_violation'],
            'fixed_bank_simplex_residual':independent['simplex_residual'],
            'eligible':bool(feasible),
            'eligibility_reason':('feasible_same_frozen_bank' if feasible else
                                  'invalid_simplex_or_cut_violation'),
            'task_inner_selection':task,
            'protected_inner_selection':protected,
            'audit_receipt_sha256':receipt['marker_sha256'],
            'inner_panel_sha256':sha(report_path),
        }
        records.append(record)
        if feasible and not record['exact_Q_alias']:
            eligible_distinct.append(record)
    chosen = min(eligible_distinct,
                 key=lambda row:(row['task_inner_selection']['exact_expected_ce']['balanced'],
                                 row['unit_id'])) if eligible_distinct else None
    result = {
        'schema':'pcrl-gradient-inner-selection-v1',
        'scope':'2018 used-data development; comparator selection only, not confirmation',
        'rule_sha256':sha(rule_path),
        'sidecar_sha256':sidecar_sha,
        'controls_sha256':sha(control_path),
        'frozen_bank_sha256':bank_hash,
        'outer_pool_opened':False,
        'gradient_labels':len(records),
        'eligible_distinct_count':len(eligible_distinct),
        'selected_unit_id':chosen['unit_id'] if chosen else None,
        'selected_source_array_sha256':chosen['source_array_sha256'] if chosen else None,
        'selection_score':chosen['task_inner_selection']['exact_expected_ce']['balanced'] if chosen else None,
        'full_curve':records,
        'limitation':rule['limitation'],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_name(output.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(temp,output)
    return {'selected_unit_id':result['selected_unit_id'],
            'eligible_distinct_count':result['eligible_distinct_count'],
            'gradient_labels':len(records), 'output_sha256':sha(output)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,
                        default=Path('results/pcrl_task_aligned_cuts_v1/agents/controls_data/GRADIENT_SELECTION.json'))
    args=parser.parse_args()
    print(json.dumps(select(args.root,args.output),sort_keys=True))

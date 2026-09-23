"""Build aggregate-only 2018 inner-pilot matched-control rows from receipts.

Run from the owned worktree.  This reads no person-level contributions or outer
assessment labels.  Missing controls retain their registered rows/statuses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROLES = ('utility:A/same_residence', 'attack:A/SEX', 'attack:A/RAC1P',
         'attack:AB/SEX', 'attack:AB/RAC1P')
WEIGHTINGS = ('U', 'PWGTP')
FIELDS = ('anchor', 'control_label', 'release_id', 'category', 'audit_status',
          'alias_of', 'duplicate_core_release_id', 'role', 'weighting', 'candidate_loss', 'H_loss',
          'H_minus_candidate', 'n_people', 'n_households',
          'fixed_bank_feasible', 'fixed_bank_objective',
          'source_array_sha256', 'sidecar_sha256', 'inner_panel_sha256',
          'comparison_scope')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def generate(root: Path, sidecar_path: Path, output: Path) -> dict:
    root = root.resolve(); sidecar_path = sidecar_path.resolve()
    sys.path.insert(0, str(root))
    from experiments.pcrl_task_aligned_cuts_v1 import scheduler
    if not sidecar_path.is_relative_to(root):
        raise ValueError('sidecar is outside this worktree')
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get('schema') != 'pcrl-control-audit-sidecar-v1' or sidecar.get('outer_pool_opened') is not False:
        raise ValueError('wrong sidecar schema or outer pool opened')
    sidecar_hash = sha(sidecar_path)
    fitted_source = root / sidecar['fitted_control_source_relative_path']
    if sha(fitted_source) != sidecar['fitted_control_source_sha256']:
        raise ValueError('fixed-bank controls report changed')
    fitted = json.loads(fitted_source.read_text())
    core_q_path = root / 'results/pcrl_task_aligned_cuts_v1/private/run/a0_u0p0_b010/channel/Q.npz'
    core_report_path = root / 'results/pcrl_task_aligned_cuts_v1/private/run/a0_u0p0_b010_audit/INNER_PANEL.json'
    core_report = json.loads(core_report_path.read_text())
    if (core_report['channel_sha256'] != 'dc22f943c06cb1bd1ecde8e2f57dd679c1fef8d3cee5aa289c1089abc3a41ee5'
            or core_report['release_id'] != 'a0_u0p0_b010'
            or core_report['split_assignment_sha256'] != sidecar['inner_split_sha256']
            or core_report['slate'] != sidecar['slate']
            or core_report['fit_pool'] != sidecar['fit_pool']
            or core_report['selection_pool'] != sidecar['selection_pool']
            or core_report['score_pool'] != sidecar['scoring_pool']):
        raise ValueError('historical Q core audit has changed or uses a different route')
    with np.load(core_q_path, allow_pickle=False) as store:
        core_channel = store['Q']
    core_raw_sha = hashlib.sha256(np.ascontiguousarray(core_channel, dtype=np.float64).tobytes()).hexdigest()
    if core_raw_sha != core_report['channel_sha256']:
        raise ValueError('historical Q core channel/report mismatch')
    controls = {}
    for family in ('simple_map_audit_units', 'fitted_control_audits'):
        for label, registration in sidecar[family].items():
            controls[label] = (family, registration)
    rows = []
    status_counts = {}
    for label, (family, registration) in sorted(controls.items()):
        canonical = registration['canonical_release_id']
        alias = registration['status'] == 'aliased_to_registered_unit'
        canonical_record = next((record for _, record in controls.values()
                                 if record['release_id'] == canonical), None)
        if canonical_record is None:
            raise ValueError(f'missing canonical control registration: {label}')
        if alias and registration['source_array_sha256'] != canonical_record['source_array_sha256']:
            raise ValueError(f'exact alias has different channel array: {label}')
        duplicate_core_release_id = ''
        if label == 'Q_historical':
            source_q = root / registration['source_relative_path']
            with np.load(source_q, allow_pickle=False) as store:
                if not np.array_equal(core_channel, store['Q']):
                    raise ValueError('sidecar historical Q differs from already audited core Q')
            duplicate_core_release_id = 'a0_u0p0_b010'
        output_dir = (root / canonical_record['output_relative_dir']).resolve()
        if not output_dir.is_relative_to(root) or 'private' not in output_dir.parts:
            raise ValueError('unsafe private control audit path')
        receipt_path = output_dir / 'SIDECAR_COMPLETE.json'
        report_path = output_dir / 'INNER_PANEL.json'
        report = None
        report_sha = ''
        if receipt_path.is_file():
            marker = scheduler._control_sidecar_marker(
                root, {'id': canonical, 'output_dir': canonical_record['output_relative_dir'],
                       'source_file_sha256': canonical_record['source_file_sha256'],
                       'source_array_sha256': canonical_record['source_array_sha256']},
                sidecar, sidecar_hash)
            if marker is None:
                raise ValueError(f'control receipt missing during build: {label}')
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('schema') != 'pcrl-control-audit-complete-v1' or
                    receipt.get('sidecar_sha256') != sidecar_hash or
                    receipt.get('release_id') != canonical or
                    receipt.get('source_file_sha256') != canonical_record['source_file_sha256'] or
                    receipt.get('source_array_sha256') != canonical_record['source_array_sha256'] or
                    not report_path.is_file() or
                    receipt.get('artifacts', {}).get('INNER_PANEL.json') != sha(report_path)):
                raise ValueError(f'control audit receipt/report mismatch: {label}')
            report = json.loads(report_path.read_text())
            if (report.get('release_id') != canonical or
                    report.get('anchor') != sidecar['anchor'] or
                    report.get('outer_pool_opened') is not False or
                    report.get('split_assignment_sha256') != sidecar['inner_split_sha256']):
                raise ValueError(f'control audit scope mismatch: {label}')
            report_sha = sha(report_path)
            status = 'alias_complete' if alias else 'complete'
        elif output_dir.exists() and any(output_dir.iterdir()):
            status = 'alias_partial' if alias else 'partial_retained'
        else:
            status = 'alias_unrun' if alias else 'registered_unrun'
        status_counts[status] = status_counts.get(status, 0) + 1
        if family == 'fitted_control_audits':
            if label == 'MILP':
                control_result = fitted['milp']['replay']
            elif label == 'HEURISTIC':
                control_result = fitted['heuristic']['replay']
            else:
                control_result = fitted['gradient'][int(label.split('_')[1])]['replay']
            feasible = str(bool(control_result['feasible'])).lower()
            fixed_objective = repr(float(control_result['objective']))
            category = ('deterministic_same_bank' if label in ('MILP', 'HEURISTIC')
                        else 'gradient_same_bank')
        else:
            feasible = ''
            fixed_objective = ''
            category = ('simple_randomization' if '_RR_' in label or '_withhold_' in label
                        else 'historical_or_unconstrained_reference')
        for role in ROLES:
            for weighting in WEIGHTINGS:
                cell = report['roles'][role] if report else None
                rows.append({
                    'anchor': sidecar['anchor'], 'control_label': label,
                    'release_id': canonical, 'category': category,
                    'audit_status': status, 'alias_of': canonical if alias else '',
                    'duplicate_core_release_id': duplicate_core_release_id,
                    'role': role, 'weighting': weighting,
                    'candidate_loss': repr(cell['candidate'][weighting]) if cell else '',
                    'H_loss': repr(cell['H'][weighting]) if cell else '',
                    'H_minus_candidate': repr(cell['H_minus_candidate'][weighting]) if cell else '',
                    'n_people': cell['n_scored_people'] if cell else '',
                    'n_households': cell['n_scored_households'] if cell else '',
                    'fixed_bank_feasible': feasible,
                    'fixed_bank_objective': fixed_objective,
                    'source_array_sha256': registration['source_array_sha256'],
                    'sidecar_sha256': sidecar_hash,
                    'inner_panel_sha256': report_sha,
                    'comparison_scope': '2018 used-data independent inner-development pilot; not confirmation',
                })
    if len(rows) != len(controls) * len(ROLES) * len(WEIGHTINGS):
        raise AssertionError('control endpoint enumeration incomplete')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, output)
    return {'schema': 'pcrl-matched-controls-inner-pilot-v1',
            'sidecar_sha256': sidecar_hash, 'control_labels': len(controls),
            'endpoint_rows': len(rows), 'status_counts': status_counts,
            'output_sha256': sha(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--sidecar', type=Path,
                        default=Path('results/pcrl_task_aligned_cuts_v1/agents/audit_review/CONTROL_AUDIT_SIDECAR.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/pcrl_task_aligned_cuts_v1/MATCHED_CONTROLS.csv'))
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.sidecar, args.output), sort_keys=True))

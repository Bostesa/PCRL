#!/usr/bin/env python3
"""Verifier, independent code: restore integrity of .private/pcrl_structural_mechanism_pilot_v1/source.

Hashes BYTES only (`shasum -a 256`); no array is loaded and no file content under .private is parsed except the text
RESTORE_RECEIPT.json. Compares every restored file with the receipt and with the public records the receipt cites:
  - exports  -> results/pcrl_acs_protected_tree_replication_v1/EXTRACTION_SUMMARY.json  export_file_sha256[role][field]
  - bank     -> results/pcrl_acs_frozen_hybrids_v1/FROZEN_BANK_MANIFEST.json            bank_files_sha256[S0|S1]
  - runs     -> results/pcrl_acs_protected_tree_replication_v1/SELECTION_LOCK.json      output_file_sha256 (where listed)
  and, as further public anchors for the run files: TRAINING_COMPLETE_LOCK.json frozen_files_sha256 and MODEL_MANIFEST.json (U0).
Also checks that no DEV_REPORT truth/weights export, no weights file and no SEALED file was copied, including under another
name (by hash).

Run from the worktree root:
  <repository>/.venv/bin/python -B results/pcrl_structural_mechanism_pilot_v1/roles/verifier/v_restore_integrity.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIV = ROOT / '.private' / 'pcrl_structural_mechanism_pilot_v1'
SRC = PRIV / 'source'
PUB_SRC = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PUB_HYB = ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
# expected admitted fields, transcribed from structural_pilot/common.py ADMITTED (commit 2d96c471e)
ADMITTED = {'MODEL_FIT': ('features', 'income', 'occupation', 'sex', 'raw_inputs'), 'CALIBRATION': ('features', 'income', 'occupation'),
            'MODEL_SELECT': ('features', 'income', 'occupation', 'sex'), 'ATTACK_FIT': ('features', 'sex'),
            'ATTACK_VALIDATION': ('features', 'sex'), 'DEV_REPORT': ('features', 'raw_inputs')}


def shasum(p):
    out = subprocess.run(['shasum', '-a', '256', str(p)], capture_output=True, text=True, check=True).stdout
    return out.split()[0]


def main():
    receipt = json.loads((PRIV / 'RESTORE_RECEIPT.json').read_text())
    extraction = json.loads((PUB_SRC / 'EXTRACTION_SUMMARY.json').read_text())
    export_sha = extraction['export_file_sha256']
    bank_sha = json.loads((PUB_HYB / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    sel_sha = json.loads((PUB_SRC / 'SELECTION_LOCK.json').read_text())['output_file_sha256']
    tcl_sha = json.loads((PUB_SRC / 'TRAINING_COMPLETE_LOCK.json').read_text())['frozen_files_sha256']
    u0_sha = {f'runs/STAGE_A_{s}/U0_MODEL.pt': json.loads((PUB_SRC / 'MODEL_MANIFEST.json').read_text())['models'][f'U0_S{s}']['sha256'] for s in (0, 1)}

    on_disk = sorted(str(Path(dp, f).relative_to(SRC)) for dp, _, fs in os.walk(SRC) for f in fs)
    listed = sorted(receipt['files'])
    rows, problems = [], []
    for rel in on_disk:
        p = SRC / rel
        h = shasum(p)
        rec = receipt['files'].get(rel)
        row = {'file': rel, 'sha256': h, 'bytes': p.stat().st_size, 'receipt_sha_match': bool(rec and rec['sha256'] == h),
               'receipt_bytes_match': bool(rec and rec['bytes'] == p.stat().st_size), 'public_checks': {}}
        parts = rel.split('/')
        if parts[0] == 'exports':
            role, field = parts[1], parts[2].removesuffix('.npz')
            row['public_checks']['EXTRACTION_SUMMARY.export_file_sha256'] = export_sha[role][field] == h
            row['admitted_field'] = field in ADMITTED.get(role, ())
        elif parts[0] == 'bank':
            row['public_checks']['FROZEN_BANK_MANIFEST.bank_files_sha256'] = bank_sha[parts[1].removeprefix('FROZEN_BANK_').removesuffix('.npz')] == h
        elif parts[0] == 'runs':
            if rel in sel_sha:
                row['public_checks']['SELECTION_LOCK.output_file_sha256'] = sel_sha[rel] == h
            key = rel.removeprefix('runs/')
            if key in tcl_sha:
                row['public_checks']['TRAINING_COMPLETE_LOCK.frozen_files_sha256'] = tcl_sha[key] == h
            if rel in u0_sha:
                row['public_checks']['MODEL_MANIFEST.models.U0'] = u0_sha[rel] == h
        if not (row['receipt_sha_match'] and row['receipt_bytes_match'] and row['public_checks'] and all(row['public_checks'].values())
                and row.get('admitted_field', True)):
            problems.append(rel)
        rows.append(row)

    # forbidden content: DEV_REPORT truth or weights of any role, under any name (by hash) or by path; any SEALED path
    forbidden_hashes = {export_sha['DEV_REPORT'][f]: f'DEV_REPORT/{f}' for f in ('income', 'occupation', 'sex', 'weights')}
    forbidden_hashes.update({export_sha[r]['weights']: f'{r}/weights' for r in export_sha})
    all_priv = sorted(str(Path(dp, f).relative_to(PRIV)) for dp, _, fs in os.walk(PRIV) for f in fs)
    hash_hits = [r['file'] for r in rows if r['sha256'] in forbidden_hashes]
    path_hits = [f for f in on_disk if f.startswith('exports/DEV_REPORT/') and Path(f).stem in ('income', 'occupation', 'sex', 'weights')]
    weights_paths = [f for f in all_priv if 'weights' in f.lower()]
    sealed_paths = [f for f in all_priv if 'sealed' in f.lower()]
    roles_restored = {}
    for f in on_disk:
        if f.startswith('exports/'):
            _, role, name = f.split('/')
            roles_restored.setdefault(role, []).append(name.removesuffix('.npz'))
    fields_equal_admitted = {r: sorted(roles_restored.get(r, [])) == sorted(ADMITTED[r]) for r in ADMITTED}
    out = {'receipt_n_files': receipt['n_files'], 'receipt_listed': len(listed), 'files_on_disk': len(on_disk),
           'disk_equals_receipt_set': on_disk == listed, 'files_only_on_disk': sorted(set(on_disk) - set(listed)),
           'files_only_in_receipt': sorted(set(listed) - set(on_disk)), 'problems': problems,
           'public_check_counts': {k: sum(1 for r in rows if k in r['public_checks'] and r['public_checks'][k]) for k in
                                   ('EXTRACTION_SUMMARY.export_file_sha256', 'FROZEN_BANK_MANIFEST.bank_files_sha256',
                                    'SELECTION_LOCK.output_file_sha256', 'TRAINING_COMPLETE_LOCK.frozen_files_sha256', 'MODEL_MANIFEST.models.U0')},
           'restored_fields_equal_ADMITTED': fields_equal_admitted, 'roles_restored': sorted(roles_restored),
           'forbidden': {'files_matching_DEV_REPORT_truth_or_any_weights_hash': hash_hits, 'DEV_REPORT_truth_paths': path_hits,
                         'paths_containing_weights': weights_paths, 'paths_containing_sealed': sealed_paths,
                         'public_sealed_assessment_record': extraction['sealed_assessment']},
           'receipt_not_copied': receipt['not_copied'], 'files': rows}
    out['pass'] = bool(out['disk_equals_receipt_set'] and not problems and all(fields_equal_admitted.values()) and not hash_hits
                       and not path_hits and not weights_paths and not sealed_paths and receipt['n_files'] == len(on_disk))
    (HERE / 'V_RESTORE_INTEGRITY.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=1))


if __name__ == '__main__':
    main()

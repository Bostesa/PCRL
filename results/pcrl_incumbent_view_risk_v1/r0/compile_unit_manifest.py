"""Compile public unit custody metadata without opening scientific arrays."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from .compute_queue import Queue

FIELDS = ('unit_id', 'stage', 'status', 'scientific_status', 'arm_id',
          'source_seed', 'realization', 'charged_cpu_s', 'wall_s',
          'spec_file_sha256', 'receipt_sha256', 'note')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    queue = Queue()
    state = queue.read()
    rows = {}
    for uid, record in state['units'].items():
        spec = record['spec']
        row = dict.fromkeys(FIELDS, '')
        row.update(unit_id=uid, stage=spec['stage'], status=record['status'],
                   spec_file_sha256=record['source_spec_sha256'])
        receipt = queue.root / 'units' / uid / 'RECEIPT.json'
        if receipt.is_file():
            value = json.loads(receipt.read_text())
            row.update(charged_cpu_s=value['charged_cpu_s'], wall_s=value['wall_s'],
                       receipt_sha256=digest(receipt))
        # Public unit receipts have aggregates, never per-person payloads.
        for owner in ('r1', 'r2', 'r3', 'r4', 'r5'):
            candidates = (root / owner).glob(f'{uid}*RECEIPT.json')
            for path in candidates:
                value = json.loads(path.read_text())
                for field in ('arm_id', 'source_seed', 'realization'):
                    if field in value:
                        row[field] = value[field]
                row['scientific_status'] = value.get('status', row['scientific_status'])
        if record.get('successor'):
            row['note'] = 'Superseded before run by ' + record['successor']
        rows[uid] = row
    index = root / 'r2/PROTECTION_SOURCE_INDEX_V2.json'
    for uid in json.loads(index.read_text())['unit_ids']:
        rows.setdefault(uid, dict(unit_id=uid, stage='protection', status='planned_not_run'))
    attack_plan = json.loads((root / 'r4/PLANNED_UNITS.json').read_text())
    for unit in attack_plan['units']:
        uid = unit['unit_id']
        row = rows.setdefault(uid, dict(unit_id=uid, stage='attacks', status='planned_not_run'))
        for field in ('arm_id', 'source_seed', 'realization'):
            row.setdefault(field, unit[field])
    # Fixed protected scoring roster remains present even if a fit is incomplete.
    for arm in ('O_J', 'V_J', 'V_P', 'V_D', 'V_L', 'V_S12', 'V_S21'):
        for seed in (0, 1):
            for role in ('FIT', 'INNER'):
                uid = f'R3_{arm}_S{seed}_{role}_PROTECTED_V1'
                rows.setdefault(uid, dict(unit_id=uid, stage='witness', status='planned_not_run',
                                         arm_id=arm, source_seed=seed))
    # Owner-only unregistered specs are preserved engineering attempts, not fits.
    for path in root.glob('r*/.private/**/*.json'):
        try:
            value = json.loads(path.read_text())
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        uid = value.get('unit_id')
        if uid and uid not in rows and 'stage' in value and 'argv' in value:
            rows[uid] = dict(unit_id=uid, stage=value['stage'], status='unregistered_not_run',
                             spec_file_sha256=digest(path), note='Preserved unregistered engineering spec')
    with (root / 'UNIT_MANIFEST.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows[uid] for uid in sorted(rows))
    print(json.dumps({'units': len(rows), 'queue_units': len(state['units']),
                      'charged_cpu_s': state['used_cpu_s']}))


if __name__ == '__main__':
    main()

"""Append the verifier's check outputs to roles/verifier/CHECKS.json (label-free; append-only: an earlier record is never removed,
rewritten or relaxed; a new output file adds new records).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/compile_checks.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

VER = REPO / 'results' / 'pcrl_direct_recovery_frontier_v1' / 'roles' / 'verifier'
OUT = VER / 'out'
CHECKS = VER / 'CHECKS.json'
TITLES = {'check1': '1. finite-law objective of the DR fits (out-of-fold fine cells, merged emitted law, AUC*, plug-in MI, 8 spends) '
                    'and deployment parity on MODEL_SELECT',
          'check2': '2. decisive MODEL_SELECT task metrics (28 conditions, G1, paired costs, plan distinct inputs and aliases)',
          'check3': '3. reader comparisons (aggregate paired screen, recipient gates, D-labels by the verifier\'s reading)',
          'check4': '4. reader refits of winning cells with the admitted recipes',
          'check5': '5. integrity (order of work, specification count, label-access policy, disclosures)'}


def records_of(path):
    o = json.loads(path.read_text())
    kind = next((k for k in TITLES if f'_{k}' in path.name or path.name.startswith(k)), 'other')
    base = {'source_file': str(path.relative_to(REPO)), 'source_sha256': V.sha256_file(path), 'check': kind,
            'title': TITLES.get(kind, kind), 'unit_id': (o.get('info') or {}).get('unit_id'),
            'utc': o.get('utc_end') or o.get('utc') or (o.get('info') or {}).get('utc_start')}
    out = []
    for g, s in (o.get('summary') or {}).items():
        out.append({**base, 'id': f'{path.stem}:{g}', 'group': g, 'result': s['result'], 'n_items': s['n'], 'n_fail': s['n_fail'],
                    'failed_items': s['failed']})
    out.append({**base, 'id': f'{path.stem}:OVERALL', 'group': 'OVERALL', 'result': o['overall']['result'],
                'n_items': o['overall']['n'], 'n_fail': o['overall']['n_fail'], 'failed_items': o['overall']['failed']})
    return out


def main():
    cur = json.loads(CHECKS.read_text()) if CHECKS.is_file() else {'schema': 'DRF1-verifier-checks-v1', 'records': []}
    have = {r['id'] for r in cur['records']}
    added = 0
    for p in sorted(OUT.glob('*.json')):
        for r in records_of(p):
            if r['id'] in have:
                continue
            r['compiled_utc'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
            cur['records'].append(r)
            have.add(r['id'])
            added += 1
    cur['note'] = ('append-only: every record of every verifier output is kept (failed checks included); records are never edited or '
                   'removed; a re-run gets a new unit id and adds new records')
    tmp = CHECKS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(V.jsonable(cur), indent=1, sort_keys=True) + '\n')
    tmp.replace(CHECKS)
    print(json.dumps({'added': added, 'records': len(cur['records'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

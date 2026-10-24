"""Investigator A, R2 design: the men's / women's guard values of every release in the committed FA_R1 decision record (an aggregate
MODEL_SELECT record named by the coordinator; nothing is recomputed from MODEL_SELECT; disclosed as adaptive).
Run: python3 results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_read_guards.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
D = json.loads((ROOT / 'results' / 'pcrl_strong_anchor_release_v1' / 'claims' / 'FA_R1' / 'DECISIONS.json').read_text())
print('top keys:', list(D))
print('decision keys:', list(D.get('decisions', {})))
found = []


def walk(o, path):
    if isinstance(o, dict):
        if 'guard|sex0' in o and isinstance(o['guard|sex0'], dict) and 'value' in o['guard|sex0']:
            found.append((path, {k: o[k].get('value') for k in ('guard|sex0', 'guard|sex1') if k in o},
                          {k: o[k].get('se') for k in ('guard|sex0', 'guard|sex1') if k in o},
                          o.get('occupation|U0|nll_excess', {}).get('value'), o.get('occupation|U0|brier_excess', {}).get('value')))
        for k, v in o.items():
            walk(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f'{path}[{i}]')


walk(D, '')
for path, val, se, nll, brier in found:
    print(path, '| guards', {k: round(v, 5) for k, v in val.items()}, '| se', {k: round(v, 5) for k, v in se.items() if v is not None},
          '| occ nll', None if nll is None else round(nll, 5), '| occ brier', None if brier is None else round(brier, 5))

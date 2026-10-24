"""Investigator A, R1 design: SUCC's per-version utility clause values and margins from the committed anchor table (label-free record).
Run: python3 results/pcrl_strong_anchor_release_v1/roles/investigator_a/r1_succ_margins.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
T = json.loads((ROOT / 'results' / 'pcrl_strong_anchor_release_v1' / 'anchors' / 'ANCHOR_TABLE.json').read_text())
print('top keys:', list(T)[:20])
succ = None
for k in ('releases', 'anchors', 'rows', 'table'):
    if isinstance(T.get(k), dict) and 'SUCC' in T[k]:
        succ = T[k]['SUCC']; break
    if isinstance(T.get(k), list):
        for r in T[k]:
            if isinstance(r, dict) and r.get('release') == 'SUCC' or (isinstance(r, dict) and r.get('name') == 'SUCC'):
                succ = r; break
if succ is None and 'SUCC' in T:
    succ = T['SUCC']


def show(o, pre='', depth=0):
    if depth > 6:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                show(v, f'{pre}{k}.', depth + 1)
            else:
                print(f'{pre}{k} = {v}')
    elif isinstance(o, list):
        for i, v in enumerate(o[:40]):
            show(v, f'{pre}[{i}].', depth + 1)


show(succ.get('utility', succ) if isinstance(succ, dict) else succ)

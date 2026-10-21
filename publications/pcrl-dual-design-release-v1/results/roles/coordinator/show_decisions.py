"""Coordinator's summary of a decision record (claims/<reading>/DECISIONS.json).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/show_decisions.py DA_REG [DB_REG ...]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]


def short(x, depth=0):
    if isinstance(x, dict):
        return {k: short(v, depth + 1) for k, v in x.items() if k not in ('frontier', 'secondary', 'aggregate_detail', 'cells')}
    if isinstance(x, float):
        return round(x, 5)
    return x


for name in sys.argv[1:]:
    d = json.loads((R / 'claims' / name / 'DECISIONS.json').read_text())
    dec = d['decisions']
    print(f'===== {name}  grid validity: {d["grid_validity"]["label"]} errors {d["grid_validity"]["errors"][:3]}')
    print('registration:', d.get('registration_status'))
    for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'outcome_class'):
        v = dec.get(k)
        if isinstance(v, dict):
            print(f'{k}: {v.get("label")}  met={v.get("met")}')
        else:
            print(k, v)
    d2 = dec.get('D2') or {}
    print('D2 passed:', d2.get('passed'), 'flags:', d2.get('flags'))
    for c, x in (d2.get('comparisons') or {}).items():
        fr = x.get('frontier') or {}
        print(f'  {c}: pass={x.get("pass")} label={x.get("label")} reason={x.get("reason")} point={x.get("point")} '
              f'benefit_mean={fr.get("benefit_mean")} se={fr.get("benefit_se")} more={fr.get("secondary_more_spending")} '
              f'targeting={x.get("targeting_pass")} recip={x.get("recipient_gate_pass")}')
        for kk in ('benefit', 'recipient_gate', 'cheaper_check'):
            if kk in x:
                print('     ', kk, short(x[kk]))
    for k in ('D3', 'D4'):
        v = dec.get(k) or {}
        print(k, short({kk: vv for kk, vv in v.items() if kk in ('label', 'failing', 'level', 'not_credited', 'missing')}))

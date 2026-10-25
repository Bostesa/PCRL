"""Verifier check 3d (DRF1; label-free): the D1 confidence conjunction and the two-SE reserve (reported, never gating) of the
candidate, OBJ_MI, TWIN and START, recomputed from my check-2 values and household-clustered SEs, against DECISIONS.json.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check3d_confidence_reserve.py --plan PLAN_DRF_R1 \
        --check2 OUT2 --out FILE

My reading: confidence (RULES.md 2): per condition PASS iff value + 2 se <= cap, VIOLATION iff value - 2 se > cap, UNRESOLVED
otherwise; CONFIDENCE_ALL_PASS iff all 28 pass, CONFIDENCE_VIOLATION iff any violation, else CONFIDENCE_UNRESOLVED. Reserve (NRM1
RULES.md 4): se_used = max(own SE, H_LJ's SE of the same condition and version x sqrt(5306 / G)); ratio = (cap - value) / se_used;
RESERVE_MET in a version iff every ratio >= 2; qualified iff met in both versions; the binding condition is the minimum ratio.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES = REPO / 'results' / 'pcrl_direct_recovery_frontier_v1'
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}


def conditions(per, sid, v):
    d = per[sid][str(v)]
    out = []
    for key, (val, se) in d['clauses'].items():
        out.append((key, val, se, CAPS[key.split('|')[2]]))
    for g, (val, se) in d['guards'].items():
        out.append((f'guard|{g}', val, se, 0.005))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='PLAN_DRF_R1')
    ap.add_argument('--check2', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    plan = json.loads((RES / 'claims' / f'{a.plan}.json').read_text())
    c2 = json.loads(Path(a.check2).read_text())
    per = c2['per_subject']
    G = int(c2['info']['households'])
    subjects = {'cand': plan['cand'], 'OBJ_MI': plan['obj_mi']['config_id'], 'TWIN': plan['twin']['id'], 'START': 'REF:SUCC'}
    mine = {}
    for role, sid in subjects.items():
        conf, res_by_v = [], {}
        for v in (0, 1):
            floor = {k: se for k, _, se, _ in conditions(per, 'REF:H_LJ', v)}
            ratios = {}
            for key, val, se, cap in conditions(per, sid, v):
                conf.append((f'v{v}|{key}', 'PASS' if val + 2 * se <= cap else 'VIOLATION' if val - 2 * se > cap else 'UNRESOLVED'))
                su = max(se, floor[key] * math.sqrt(5306 / G))
                ratios[key] = (cap - val) / su
            b = min(ratios, key=ratios.get)
            res_by_v[v] = {'min_ratio': ratios[b], 'binding': f's{v}|{b}', 'met': all(r >= 2.0 for r in ratios.values())}
        unresolved = [k for k, s in conf if s == 'UNRESOLVED']
        viol = [k for k, s in conf if s == 'VIOLATION']
        lab = 'CONFIDENCE_VIOLATION' if viol else 'CONFIDENCE_UNRESOLVED' if unresolved else 'CONFIDENCE_ALL_PASS'
        mine[role] = {'confidence': {'label': lab, 'n_pass': sum(s == 'PASS' for _, s in conf), 'unresolved': unresolved,
                                     'violations': viol},
                      'reserve': {'by_version': res_by_v, 'qualified': all(x['met'] for x in res_by_v.values())}}
    dec = json.loads((RES / 'claims' / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']['D1']
    items = []
    for role in subjects:
        rc, rr = dec['confidence'][role], dec['reserve'][role]
        m = mine[role]
        items.append(V.cmp_eq(f'{role}:confidence_label', m['confidence']['label'], rc['label']))
        items.append(V.cmp_eq(f'{role}:confidence_n_pass', m['confidence']['n_pass'], rc.get('n_pass')))
        items.append(V.cmp_eq(f'{role}:confidence_unresolved', sorted(m['confidence']['unresolved']), sorted(rc.get('unresolved') or [])))
        items.append(V.cmp_eq(f'{role}:confidence_violations', sorted(m['confidence']['violations']), sorted(rc.get('violations') or [])))
        items.append(V.cmp_eq(f'{role}:reserve_qualified', m['reserve']['qualified'], rr.get('qualified')))
        for v in (0, 1):
            items.append(V.cmp_num(f'{role}:reserve_min_ratio_v{v}', m['reserve']['by_version'][v]['min_ratio'],
                                   rr['min_ratio_by_version'][str(v)], 1e-8))
        b = min(m['reserve']['by_version'].values(), key=lambda x: x['min_ratio'])['binding']
        items.append(V.cmp_eq(f'{role}:reserve_binding', b, rr.get('binding')))
    out = {'schema': 'DRF1-verifier-check3d-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'info': {'plan': a.plan, 'check2': a.check2, 'households': G}, 'mine': mine, 'overall': V.summarize(items),
           'summary': {'D1': V.summarize(items)}, 'items': items}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'mine': {r: {'confidence': m['confidence']['label'],
                                                                        'reserve_qualified': m['reserve']['qualified']}
                                                                    for r, m in mine.items()}}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

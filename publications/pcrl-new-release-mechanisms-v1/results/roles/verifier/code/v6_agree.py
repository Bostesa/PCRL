"""V6 agreement on INVENTED inputs between the verifier's own rule implementation (v6_select) and the reviewer's nrm.rules: random
grid entries (values, SEs, MC SEs, aliases, k-duplicates, missing cells) and random ledgers. No real data, no label. A disagreement
is adjudicated against the RULES.md text before any real replay (recorded, never silently patched).

    python3 .../v6_agree.py --tag V1
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import v6_select as S  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import nrm.rules as NR  # noqa: E402  (reviewer code, compared on invented inputs only)


def rand_entry(rng):
    kind = rng.integers(0, 6)
    vals = rng.normal(rng.choice([0.0, 0.005, 0.01, 0.025]), rng.choice([0.001, 0.004, 0.01]), 8)
    ses = np.abs(rng.normal(0.004, 0.002, 8))
    if kind == 0:
        vals[:] = 0.0; ses[:] = 0.0                      # exact alias
    elif kind == 1:
        vals[1::2] = vals[0::2]                          # seed-deterministic duplicates
    elif kind == 2:
        ses[:] = 1e-5                                    # floor binds
    e = {'cells': {c: float(x) for c, x in zip(S.REG, vals)}, 'cell_se': {c: float(s) for c, s in zip(S.REG, ses)}}
    if kind == 3:
        e['cell_se_mc'] = {c: float(abs(rng.normal(0.002, 0.001))) for c in S.REG}
    if kind == 4 and rng.random() < 0.5:
        e['cells'].pop(S.REG[int(rng.integers(0, 8))])  # incomplete
    return e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    rng = np.random.default_rng(20261020)
    dis = []
    n = 3000
    for i in range(n):
        e = rand_entry(rng)
        b1, b2 = S.my_benefit(e), NR.benefit_screen(e)
        if b1['label'] != b2['label']:
            dis.append({'i': i, 'what': 'benefit', 'mine': b1['label'], 'theirs': b2['label']})
        if b1['agg'] and b2['aggregate'].get('complete'):
            if abs(b1['agg']['tot']['mean'] - b2['aggregate']['mean']) > 1e-15 or abs(b1['agg']['tot']['se'] - b2['aggregate']['se_agg']) > 1e-15:
                dis.append({'i': i, 'what': 'aggregate numbers'})
            for v in (0, 1):
                if abs(b1['agg']['per'][v]['se'] - b2['aggregate']['per_version'][v]['se_agg']) > 1e-15:
                    dis.append({'i': i, 'what': 'per-version se'})
        t1, t2 = S.my_target(e)['label'], NR.target_screen(e)['label']
        if t1 != t2:
            dis.append({'i': i, 'what': 'target', 'mine': t1, 'theirs': t2})
        r1, r2 = S.my_recipient(e), NR.recipient_screen(e)
        if r1['label'] != r2['label'] or r1['gate'] != r2['gate_pass']:
            dis.append({'i': i, 'what': 'recipient', 'mine': r1['label'], 'theirs': r2['label']})
        o1 = S.old_stable(e)
        o2 = (b2.get('old_every_cell_rule') or {}).get('stable')
        if o1 is not None and bool(o1) != bool(o2):
            dis.append({'i': i, 'what': 'old stable', 'mine': o1, 'theirs': o2})
    # random ledgers through claims and selection
    refw = {r: {v: {'income': f'i{r}', 'occupation': f'o{r}'} for v in (0, 1)} for r in S.BENCH}
    nsel = 0
    for j in range(150):
        ledger = []
        for c in range(int(rng.integers(1, 6))):
            def strong(mu):
                return {'cells': {x: float(rng.normal(mu, 0.002)) for x in S.REG}, 'cell_se': {x: 0.002 for x in S.REG}}
            if rng.random() < 0.7:                          # mostly competitive invented candidates (exercise tiers and ties)
                good = {'pair': strong(0.012), 'income': strong(0.0), 'occupation': strong(0.0)}
                grid = {r: {'pair': strong(float(rng.choice([0.01, 0.012, 0.03]))), 'income': good['income'],
                            'occupation': good['occupation']} for r in S.BENCH}
            else:
                good = {'pair': rand_entry(rng), 'income': rand_entry(rng), 'occupation': rand_entry(rng)}
                grid = {r: {'pair': rand_entry(rng), 'income': good['income'], 'occupation': good['occupation']} for r in S.BENCH}
            grid.update({'TW': good, 'ALT': good})
            val = float(rng.choice([0.0, 0.0, 0.004, 0.02]))
            ev = {v: {'utility': {'clauses': [{'seed': v, 'task': t, 'reference': r, 'metric': m, 'value': val * rng.random(),
                                               'cap': V.CAPS[m], 'se': 0.001 + 0.002 * rng.random(),
                                               **({'se_mc': 0.001} if c % 2 else {})}
                                              for t in V.TASKS for r in V.REFS for m in V.METRICS],
                                  'guards': [{'seed': v, 'group': g, 'value': 0.0, 'cap': 0.005, 'se': 0.001} for g in ('sex0', 'sex1')],
                                  'households': 5306}} for v in (0, 1)}
            w = {v: {'income': f'i{j}_{c}', 'occupation': f'o{j}_{c}'} for v in (0, 1)}
            if rng.random() < 0.1:
                w = refw['W']
            ledger.append({'config_id': f'C{j}_{c}', 'family': 'X', 'kind': 'candidate' if c % 3 else 'control', 'entry': c,
                           'n_fitted_components': int(rng.integers(1, 4)), 'wire_sha256_eval': w,
                           'wire_sha256_grid': w if rng.random() > 0.05 else {}, 'evals': ev, 'grid': grid,
                           'matched': {'twin': 'TW', 'alternative': 'ALT'}, 'grid_valid': bool(rng.random() > 0.05),
                           'frontier': {'label': str(rng.choice(['DOMINATES_CONTROL_FRONTIER', 'NOT_SEPARATED_FROM_TASK_SPENDING']))}})
        m1 = S.my_select(ledger, refw)
        m2 = NR.select_candidate(json.loads(json.dumps(ledger)), refw)
        nsel += m1['outcome'] == 'DEVELOPMENT_CANDIDATE_SELECTED'
        mm = S.compare_selection(m1, m2)
        if mm:
            dis.append({'j': j, 'what': 'selection', 'mismatches': mm[:5]})
    res = {'n_entries': n, 'n_ledgers': 150, 'n_ledgers_with_selection': nsel, 'disagreements': dis[:30], 'n_disagreements': len(dis)}
    chk = {'id': f'V6.RULES_AGREEMENT_INVENTED__{a.tag}', 'what': 'agreement of the verifier rule implementation with nrm.rules on invented '
           'grid entries (3,000: benefit, target, recipient labels, aggregate numbers, old every-cell flag) and 150 invented ledgers '
           '(claims, tiers, scores, statuses, selection); invented data only',
           'inputs': V.inputs_sha([Path(__file__), V.CODE_DIR / 'v6_select.py', V.ROOT / 'nrm' / 'rules.py', S.RULES]),
           'result': res, 'numerical_status': 'PASS' if not dis else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(res, indent=1, default=V._jd)[:3000])
    return 0 if not dis else 1


if __name__ == '__main__':
    sys.exit(main())

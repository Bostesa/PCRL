"""Write RESULTS_TABLES.md from the frozen records (aggregates only): MODEL_SELECT utility bank, capacity, final-bank recovery
and DEV_REPORT utility, primary endpoints with statuses, controls and coverage. Missing comparisons are shown explicitly."""
from __future__ import annotations
import json

from . import common as C


def f(x, d=4):
    return 'n/a' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def main():
    R = C.RESULTS
    sel = json.loads((R / 'SELECTION_LOCK.json').read_text()); fit = json.loads((R / 'FIT_SUMMARY.json').read_text())
    rep = json.loads((R / 'report' / 'DEV_REPORT_RESULTS.json').read_text()); rd = json.loads((R / 'READERS_SUMMARY.json').read_text())
    L = ['# Results tables (exploratory development evidence; all values from frozen records)', '']
    L += ['## 1. Utility bank on MODEL_SELECT (24 held-out point clauses per arm; worst normalised clause = max value/cap)', '',
          '| Arm | Cells s0 / s1 | Point-feasible (both seeds) | Worst normalised s0 / s1 | Cross-fitted pair proxy s0 / s1 | Men occ. Brier excess vs U0 s0 / s1 |', '|---|---|---|---|---|---|']
    for arm in ['C0', 'SCM_b0.5', 'SCM_b1.0', 'SCM_b1.5', 'TWIN_b0.5', 'TWIN_b1.0', 'TWIN_b1.5', 'IBSI_b1.0_g0.5', 'IBSI_b1.0_g1.0', 'IBSI_b1.0_g2.0',
                'SHUF_b1.0_r0', 'SHUF_b1.0_r1', 'SHUF_b1.0_r2', 'SHUF_b1.0_r3', 'SHUF_b1.0_r4']:
        P = [sel['per_seed'][s][arm] for s in ('0', '1')]
        cells = [sel['final_cells'][s][arm] for s in ('0', '1')]
        L.append(f"| {arm} | {cells[0]} / {cells[1]} | {all(p['point_feasible_24'] for p in P)} | {f(P[0]['max_normalized'], 3)} / {f(P[1]['max_normalized'], 3)} | "
                 f"{f(P[0]['proxy']['arm_proxy'])} / {f(P[1]['proxy']['arm_proxy'])} | {f(P[0]['subgroup_occupation_brier_excess_vs_U0']['sex0'])} / {f(P[1]['subgroup_occupation_brier_excess_vs_U0']['sex0'])} |")
    for name in ('FROZEN_NEURAL_RAW', 'FROZEN_H_LJ'):
        P = [sel['per_seed'][s][name] for s in ('0', '1')]
        feas = all(c['point_pass'] for p in P for c in p['clauses'])
        wn = [max(c['normalized'] for c in p['clauses']) for p in P]
        L.append(f"| {name} (frozen) | n/a | {feas} | {f(wn[0], 3)} / {f(wn[1], 3)} | {f(P[0]['proxy']['arm_proxy'])} / {f(P[1]['proxy']['arm_proxy'])} | not recorded |")
    L += ['', f"Nominee {sel['nominee']['arm']}; twin {sel['twin']['arm']} (matched capacity: {sel['twin_matched_capacity']}); IBSI {sel['ibsi']['arm']} "
          f"(eligible gammas {sel['ibsi']['eligible_gammas']}); SHUF {sel['shuf']['arm']}. Registered F1 check on the proxy (TWIN - SCM): "
          f"{f(sel['falsifier_F1_check']['0'])} / {f(sel['falsifier_F1_check']['1'])}.", '']
    L += ['## 2. Final bank on DEV_REPORT: recovery (selected reader AUC, 8-reader bank incl. LOOKUP) and point utility', '',
          '| Seed | Release | Pair AUC (se) | Income AUC | Occupation AUC | Pair AUC, 7-reader slate | Pair winner | DEV point-feasible, this seed (12 clauses) |', '|---|---|---|---|---|---|---|---|']
    for s in ('S0', 'S1'):
        for svc, v in rep['per_seed'][s]['services'].items():
            rc = v['recovery']
            L.append(f"| {s} | {svc} | {f(rc['pair']['auc'])} ({f(rc['pair']['auc_se'])}) | {f(rc['income']['auc'])} | {f(rc['occupation']['auc'])} | "
                     f"{f(v['recovery_slate7']['pair']['auc'])} | {rc['pair']['winner']} | {v['utility_point_feasible_24_this_seed']} |")
    L += ['', '## 3. Primary endpoints (benefit = comparator AUC - SCM AUC; worsening = SCM - TWIN; clause one-sided alpha/3; family K = 42)', '',
          'Status is each endpoint\'s own family-interval status (K = 42) against its margin. It is not a claim status. The nominee fails the DEV utility gate, so every RECIPE, ATTRIBUTION and COMPETITIVE endpoint is descriptive; the claim labels are in section 4. Clause bound is the one-sided alpha/3 lower bound (L1) for benefit endpoints, and the upper bound (U1) for worsening, utility and subgroup endpoints.', '',
          '| Endpoint | Point | Clause bound | Family interval | Status |', '|---|---|---|---|---|']
    for k, e in rep['endpoints_primary'].items():
        if e.get('status') == 'MISSING':
            L.append(f'| {k} | MISSING | | | comparator absent |'); continue
        cb = e['clause_L1'] if e['kind'] == 'benefit' else e['clause_U1']
        L.append(f"| {k} | {f(e['point'])} | {f(cb)} | [{f(e['family_lower'])}, {f(e['family_upper'])}] | {e['status']} |")
    L += ['', 'Secondary endpoints (SECONDARY_ENDPOINTS.csv) are descriptive differences: use their intervals and difference status only.', '', '## 4. Claims', '']
    for k, v in rep['claims'].items():
        L.append(f"- **{k}:** {v['decision'] if isinstance(v, dict) else v}")
    L += ['', f"Separate status fields: {json.dumps(rep['separate_status_fields'])}", '', f"Controls: {json.dumps(rep['controls'])}", '',
          '## 5. Coverage on the attack roles (pair view; diagnostic)', '', '| Seed | Release | Distinct pair vectors ATTACK_FIT | Median people per vector | Share of ATTACK_VALIDATION people whose pair vector occurs in ATTACK_FIT |', '|---|---|---|---|---|']
    for s in ('S0', 'S1'):
        for svc, cov in rd['per_seed'][s]['coverage'].items():
            c = cov['pair']
            L.append(f"| {s} | {svc} | {c['distinct_vectors_ATTACK_FIT']} | {f(c['median_ATTACK_FIT_people_per_vector'], 1)} | {f(c['share_ATTACK_VALIDATION_people_whose_vector_occurs_in_ATTACK_FIT'], 3)} |")
    (R / 'RESULTS_TABLES.md').write_text('\n'.join(L) + '\n')
    print('written', len(L), 'lines')


if __name__ == '__main__':
    main()

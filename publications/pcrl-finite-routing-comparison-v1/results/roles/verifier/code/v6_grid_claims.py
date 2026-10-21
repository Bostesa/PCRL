"""V6 registered-grid replay and the D1-D6 decisions under ER1 (label-free), own code.

Inputs: runs/GRID/<bank>/S<v>/GRID.json, BOOT.npz (400 household-bootstrap SCORE AUC draws of each cell / member / view winner),
the committed plans claims/PLAN_MAIN.json and PLAN_R1.json, READER_SPLITS.json (seeds), the P1 / R1 selection records, the
evaluation records (MODEL_SELECT paired costs; recomputed by V4) and the qualification decision. Recomputed with my own code:
- per cell, member pair and view: pair benefit = AUC(comparator) - AUC(candidate), single views AUC(candidate) - AUC(comparator),
  household-bootstrap SE = SD (ddof 1) of the paired draw differences; against the grid's 'paired' block;
- the aggregate screen (blocks = version x repetition, block means over reader seeds, x = mean of block means; se_hh = RMS of cell
  SEs with the x weights, se_rs = sqrt(mean (m_b - x)^2), se_agg = max(sqrt(se_hh^2 + se_rs^2), 0.001), 0 for an exact alias;
  both version means), the benefit label, the recipient screens (limit 0.010), the target screen (0.020);
- the frontier comparisons (pooled MODEL_SELECT total-NLL costs, bracket = the cheapest control spending at least as much, benefit
  at the bracket, eight secondary metrics), F4 (diff - 2 se > 0 both versions), the reader-targeting point conditions (non-table
  winner per-cell benefits from the grid extras, both version means > 0, and the plan's nested LOOKUP benefits > 0, which are
  cross-checked against the selection records), the reference reader-budget credit;
- D2 (both readings), D3 (G4 over the nine references with credit), D4 (G5 levels), the outcome class; against DECISIONS.json;
- the ER1 grid validity from the grid extras' score fingerprints: k0 vs k1 per recipe / view / member / version / repetition,
  seeds (+700 registered, 300 / 400 / 500 / 600 forbidden), (ii) / (iii) with the ER1 explanation (a) another seed-dependent
  recipe differs, (b) the same recipe differs on another view, (c) the view's distinct-input count recorded (my own label-free
  counts on MODEL_SELECT).
The reader-budget flags themselves (frc.readerbudget.contrast) are taken from the plan (not recomputed here).

    python3 .../v6_grid_claims.py --tag V1 [--bank FRC1_MAIN]
    python3 .../v6_grid_claims.py --selftest
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

VERS = (0, 1)
REG = tuple(f'S{v}{r}k{k}' for v in VERS for r in ('R1', 'R2') for k in (0, 1))
VIEWS = ('pair', 'income', 'occupation')
SINGLE = ('income', 'occupation')
Z = 2.0
FLOOR = 0.001
RECIP = 0.010
TARGET = 0.020
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
BENCH = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SEED_INV = ('OUT_LR', 'OUT_HGB', 'LOG_LR', 'DEC_LR', 'CONF_HGB', 'LOOKUP', 'HGB15', 'HGB31', 'CATI01', 'CATI1')
SEED_BASE = 2026101205
D2_ORDER = ('TWIN', 'SHUF_FRONTIER', 'HQ_FRONTIER', 'ORD_FRONTIER')
TOL = 1e-12


def parse(cell):
    return int(cell[1]), cell[2:4], int(cell[5:])


# ---------------------------------------------------------------- grid inputs
def load_bank(bank):
    out = {'cells': {}, 'paired': {}, 'boot': {}, 'members': {}}
    for v in VERS:
        d = V.RUNS / 'GRID' / bank / f'S{v}'
        g = json.loads((d / 'GRID.json').read_text())
        out['members'][v] = g.get('members') or {}
        out['cells'].update(g['cells'])
        for cand, comps in g['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    x = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    x['cells'].update(e['cells'])
                    x['cell_se'].update(e['cell_se'])
        with np.load(d / 'BOOT.npz', allow_pickle=False) as z:
            for k in z.files:
                out['boot'][k] = z[k].copy()
    return out


def my_paired(bank, cand, comp):
    out = {vw: {'cells': {}, 'cell_se': {}} for vw in VIEWS}
    for cell, cres in bank['cells'].items():
        if cand not in cres or comp not in cres:
            continue
        for vw in VIEWS:
            a, b = cres[cand][vw]['score_auc'], cres[comp][vw]['score_auc']
            ba, bb = bank['boot'][f'{cell}__{cand}__{vw}'], bank['boot'][f'{cell}__{comp}__{vw}']
            d, db = (b - a, bb - ba) if vw == 'pair' else (a - b, ba - bb)
            out[vw]['cells'][cell] = float(d)
            out[vw]['cell_se'][cell] = float(np.std(db, ddof=1))
    return out


def nolookup(bank, cand, comp):
    cells = {}
    for cell in REG:
        try:
            a = bank['cells'][cell][cand]['pair']['extras']['nolookup']['score_auc']
            b = bank['cells'][cell][comp]['pair']['extras']['nolookup']['score_auc']
        except KeyError:
            return None
        cells[cell] = float(b) - float(a)
    return cells


# ---------------------------------------------------------------- screens (own)
def aggregate(entry, cells=REG):
    vals, ses = entry['cells'], entry['cell_se']
    if not all(c in vals and c in ses for c in cells):
        return None
    blocks = {}
    for c in cells:
        v, r, k = parse(c)
        blocks.setdefault((v, r), []).append((float(vals[c]), float(ses[c])))

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b]) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        se_hh = math.sqrt(sum(t[1] ** 2 / (nb * len(blocks[b])) for b in keys for t in blocks[b]))
        se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'se_hh': se_hh, 'se_rs': se_rs, 'raw': math.sqrt(se_hh ** 2 + se_rs ** 2)}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in VERS}
    alias = all(t[0] == 0.0 and t[1] == 0.0 for b in keys for t in blocks[b])
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], FLOOR)
        s['lower'], s['upper'] = s['mean'] - Z * s['se_agg'], s['mean'] + Z * s['se_agg']
    return {**tot, 'alias': alias, 'per_version': per}


def benefit(entry, threshold=0.0):
    a = aggregate(entry) if entry else None
    if a is None:
        return {'label': 'INCOMPLETE', 'resolved': False}
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['lower'] > threshold and all(a['per_version'][v]['mean'] > threshold for v in VERS):
        lab = 'AGG_POSITIVE' if threshold == 0 else 'TARGET_AGG_RESOLVED'
    elif a['mean'] > threshold:
        lab = 'AGG_POSITIVE_POINT_ONLY' if threshold == 0 else 'TARGET_AGG_POINT_ONLY'
    else:
        lab = 'AGG_NOT_POSITIVE' if threshold == 0 else 'TARGET_AGG_NOT_MET'
    return {'label': lab, 'resolved': lab in ('AGG_POSITIVE', 'TARGET_AGG_RESOLVED'), 'mean': a['mean'], 'se_agg': a['se_agg'],
            'version_means': {v: a['per_version'][v]['mean'] for v in VERS}}


def recipient(entry):
    a = aggregate(entry) if entry else None
    if a is None:
        return {'label': 'INCOMPLETE', 'gate_pass': False}
    pv = a['per_version']
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif any(pv[v]['lower'] > RECIP for v in VERS):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= RECIP for v in VERS):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= RECIP for v in VERS):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return {'label': lab, 'gate_pass': lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')}


def comparison(entry):
    b = benefit(entry.get('pair'))
    r = {vw: recipient(entry.get(vw)) for vw in SINGLE}
    return {'benefit': b, 'recipient': r, 'benefit_positive': b['resolved'], 'recipient_gate_pass': all(x['gate_pass'] for x in r.values())}


def target(entry):
    b = benefit(entry.get('pair'), TARGET)
    if b['label'] in ('INCOMPLETE', 'ALIAS_ZERO'):
        b['label'] = 'TARGET_AGG_NOT_MET'
    return b


def version_means(cells):
    if cells is None or not all(c in cells for c in REG):
        return None
    out = {}
    for v in VERS:
        bl = {}
        for c in REG:
            vv, r, _ = parse(c)
            if vv == v:
                bl.setdefault(r, []).append(cells[c])
        out[v] = sum(sum(x) / len(x) for x in bl.values()) / len(bl)
    return out


def pooled(per):
    vals = [float(per[v]['value']) for v in VERS]
    ses = [math.sqrt(float(per[v]['se']) ** 2 + float(per[v].get('se_mc') or 0.0) ** 2) for v in VERS]
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2}


def frontier(costs_vs, grid_vs, path):
    prim = {q: pooled({v: costs_vs[q][v]['total|nll'] for v in VERS}) for q in path}
    above = sorted((q for q in path if prim[q]['value'] <= 0), key=lambda q: -prim[q]['value'])
    if not above:
        return {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT', 'point': None,
                'primary': prim}
    qs = above[0]
    ben = benefit(grid_vs[qs]['pair'])
    sec = {m: pooled({v: costs_vs[qs][v][m] for v in VERS}) for m in SECONDARY}
    more = sorted(m for m, x in sec.items() if x['value'] - Z * x['se'] > 0)
    if not ben['resolved']:
        lab, why = 'NOT_SEPARATED_FROM_TASK_SPENDING', 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
    elif more:
        lab, why = 'SEPARATED_ON_PRIMARY_COST_ONLY', None
    else:
        lab, why = 'DOMINATES_CONTROL_FRONTIER', None
    return {'label': lab, 'reason': why, 'point': qs, 'primary': prim, 'benefit_at_bracket': ben, 'secondary_more_spending': more}


# ---------------------------------------------------------------- decisions (own assembly from my numbers)
def decide(plan, bank, costs, qdec, g1_feasible):
    s2 = plan['prc_s2']
    grid = {}
    for m in list(plan['members']) + list(BENCH):
        if m != s2 and m in bank['cells'][REG[0]]:
            grid[m] = my_paired(bank, s2, m)
    comps, flags = {}, []
    ct = comparison(grid['TWIN'])
    comps['TWIN'] = {'benefit': ct['benefit']['label'], 'recipient_gate_pass': ct['recipient_gate_pass'], 'point': 'TWIN',
                     'pass': ct['benefit_positive'] and ct['recipient_gate_pass'], 'mean': ct['benefit']['mean'],
                     'se_agg': ct['benefit']['se_agg']}
    for name, key in (('SHUF_FRONTIER', 'SHUF'), ('HQ_FRONTIER', 'HQ'), ('ORD_FRONTIER', 'ORD')):
        ids = []
        for i in plan['paths'].get(key) or []:
            j = (plan.get('alias_of') or {}).get(i, i)
            if j and j not in ids:
                ids.append(j)
        if not ids:
            ids = ['TWIN']
            flags.append(f'{key}_AT_IDENTITY')
        fr = frontier(costs, grid, ids)
        comps[name] = {'label': fr['label'], 'reason': fr.get('reason'), 'path': ids, 'point': fr.get('point'),
                       'pass': fr['label'] == 'DOMINATES_CONTROL_FRONTIER', 'primary': fr['primary'],
                       'benefit_at_bracket': fr.get('benefit_at_bracket')}
    f4 = {v: plan['f4']['version'][str(v)] for v in VERS}
    f4p = {v: f4[v]['diff'] - Z * f4[v]['se'] > 0 for v in VERS}
    tg = {}
    for q in dict.fromkeys(c['point'] for c in comps.values() if c.get('point')):
        vm = version_means(nolookup(bank, s2, q))
        nl = (plan.get('nested_lookup') or {}).get(q)
        nl = {v: float(nl[str(v)]) for v in VERS} if nl else None
        tg[q] = {'pass': bool(vm and nl and all(vm[v] > 0 for v in VERS) and all(nl[v] > 0 for v in VERS)), 'nolookup_vm': vm,
                 'nested_lookup': nl}
    passed = [n for n in D2_ORDER if comps[n]['pass']]
    for n in D2_ORDER:
        comps[n]['targeting_pass'] = tg[comps[n]['point']]['pass'] if comps[n].get('point') in tg else None
    targeted = not all(x['pass'] for x in tg.values())
    if not all(f4p.values()):
        lab = 'NOT_MET_F4'
    elif len(passed) == 4 and targeted:
        lab = 'READER_TARGETED'
    elif len(passed) == 4:
        lab = 'COMPONENT_BEYOND_MATCHED_CONTROLS' if g1_feasible else 'MET_WITHOUT_TASK_FEASIBILITY'
    elif not passed:
        lab = 'NO_REDUCTION_BEYOND_ANY_CONTROL'
    else:
        lab = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    d2 = {'label': lab, 'met': lab == 'COMPONENT_BEYOND_MATCHED_CONTROLS', 'passed': passed, 'f4_pass': f4p, 'targeting': tg,
          'comparisons': comps, 'flags': sorted(set(flags))}
    # D3 (G4 + reader-budget credit) and D4 (G5 + credit)
    rc = {r: comparison(grid[r]) for r in BENCH if r in grid}
    missing = [r for r in BENCH if r not in rc]
    failing = [r for r in BENCH if r in rc and not (rc[r]['benefit_positive'] and rc[r]['recipient_gate_pass'])]
    g4 = 'INCOMPLETE' if missing else ('NOT_SUPERIOR' if failing else
                                       ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if g1_feasible else 'MET_WITHOUT_TASK_FEASIBILITY'))
    credit = {}
    for r in BENCH:
        if (plan.get('reader_budget') or {}).get(r) == 'READER_BUDGET_ASYMMETRIC':
            vm = version_means(nolookup(bank, s2, r))
            credit[r] = 'INCOMPLETE' if vm is None else ('CREDITED' if all(vm[v] > 0 for v in VERS) else 'NOT_CREDITED')
    nc = sorted(r for r, x in credit.items() if x == 'NOT_CREDITED')
    inc = sorted(r for r, x in credit.items() if x == 'INCOMPLETE')
    d3l = g4
    if g4 in ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'MET_WITHOUT_TASK_FEASIBILITY'):
        d3l = 'NOT_SUPERIOR' if nc else ('INCOMPLETE' if inc else g4)
    d3 = {'label': d3l, 'met': d3l == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'failing': sorted(set(failing) | set(nc)),
          'reader_budget_credit': credit, 'not_credited': nc}
    tl = {r: target(grid[r])['label'] for r in TARGET_REFS}
    rgate = all(rc[r]['recipient_gate_pass'] for r in TARGET_REFS)
    if all(x == 'TARGET_AGG_RESOLVED' for x in tl.values()):
        level = 'MATERIAL_0020_RESOLVED' if rgate else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in tl.values()):
        level = 'MATERIAL_0020_POINT_ONLY'
    elif all(rc[r]['benefit_positive'] for r in TARGET_REFS):
        level = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        level = 'NOT_MET'
    cr4 = {r: credit[r] for r in TARGET_REFS if r in credit}
    if any(x == 'NOT_CREDITED' for x in cr4.values()):
        level = 'NOT_MET'
    elif any(x == 'INCOMPLETE' for x in cr4.values()) and level != 'NOT_MET':
        level = 'INCOMPLETE'
    d4l = level if (g1_feasible or level in ('NOT_MET', 'INCOMPLETE')) else 'MET_WITHOUT_TASK_FEASIBILITY'
    d4 = {'label': d4l, 'level': level, 'target_labels': tl}
    d1 = (qdec.get('d1') or {}).get('label')
    if d2['met'] and d3['met']:
        oc = 'CANDIDATE'
    elif d2['met']:
        oc = 'COMPONENT'
    elif d1 == 'REPAIRED_TASK_ONLY_RELEASE':
        oc = 'ENGINEERING_REPAIR'
    else:
        oc = 'FINITE_NEGATIVE_RESULT'
    return {'D1': d1, 'D2': d2, 'D3': d3, 'D4': d4, 'outcome': oc,
            'reference_comparisons': {r: {'benefit': rc[r]['benefit']['label'], 'recipient_gate': rc[r]['recipient_gate_pass'],
                                          'mean': rc[r]['benefit']['mean'], 'se_agg': rc[r]['benefit']['se_agg']} for r in rc}}, grid


def compare_decisions(mine, rec):
    mm = []
    dec = rec['decisions']
    if mine['D1'] != dec['D1']['label']:
        mm.append(f"D1 mine {mine['D1']} recorded {dec['D1']['label']}")
    d2r = dec['D2']
    if mine['D2']['label'] != d2r['label']:
        mm.append(f"D2 mine {mine['D2']['label']} recorded {d2r['label']}")
    if sorted(mine['D2']['passed']) != sorted(d2r.get('passed') or []):
        mm.append(f"D2 passed mine {mine['D2']['passed']} recorded {d2r.get('passed')}")
    for n in D2_ORDER:
        a, b = mine['D2']['comparisons'][n], (d2r.get('comparisons') or {}).get(n) or {}
        for k in ('pass', 'point', 'targeting_pass'):
            if a.get(k) != b.get(k):
                mm.append(f'D2 {n} {k} mine {a.get(k)} recorded {b.get(k)}')
        if n == 'TWIN':
            if a['benefit'] != b.get('benefit') or a['recipient_gate_pass'] != b.get('recipient_gate_pass'):
                mm.append(f'D2 TWIN benefit / gate mine {a["benefit"]}/{a["recipient_gate_pass"]} recorded {b.get("benefit")}/'
                          f'{b.get("recipient_gate_pass")}')
        else:
            if a['label'] != b.get('label') or a.get('reason') != b.get('reason'):
                mm.append(f'D2 {n} label mine {a["label"]}/{a.get("reason")} recorded {b.get("label")}/{b.get("reason")}')
            fr = b.get('frontier') or {}
            for q, x in (a.get('primary') or {}).items():
                r = (fr.get('primary_cost_differences') or {}).get(q)
                if r is None or abs(r['value'] - x['value']) > TOL or abs(r['se'] - x['se']) > TOL:
                    mm.append(f'D2 {n} primary cost vs {q} mine {x} recorded {r}')
            if a.get('benefit_at_bracket') and fr.get('benefit_mean') is not None:
                if abs(fr['benefit_mean'] - a['benefit_at_bracket']['mean']) > TOL or \
                        abs(fr['benefit_se'] - a['benefit_at_bracket']['se_agg']) > TOL:
                    mm.append(f'D2 {n} benefit at bracket mine {a["benefit_at_bracket"]["mean"]} recorded {fr["benefit_mean"]}')
    f4r = (d2r.get('f4') or {}).get('per_version') or {}
    for v in VERS:
        if mine['D2']['f4_pass'][v] != (f4r.get(str(v)) or {}).get('pass'):
            mm.append(f'F4 V{v} mine {mine["D2"]["f4_pass"][v]} recorded {(f4r.get(str(v)) or {}).get("pass")}')
    for q, t in mine['D2']['targeting'].items():
        r = (d2r.get('targeting') or {}).get(q) or {}
        if t['pass'] != r.get('pass'):
            mm.append(f'targeting {q} mine {t["pass"]} recorded {r.get("pass")}')
        for v in VERS:
            rv = (r.get('nolookup_version_means') or {}).get(str(v))
            if t['nolookup_vm'] is None or rv is None or abs(rv - t['nolookup_vm'][v]) > TOL:
                mm.append(f'targeting {q} nolookup V{v} mine {(t["nolookup_vm"] or {}).get(v)} recorded {rv}')
    d3r = dec['D3']
    if mine['D3']['label'] != d3r['label'] or sorted(mine['D3']['failing']) != sorted(d3r.get('failing') or []):
        mm.append(f"D3 mine {mine['D3']['label']} {mine['D3']['failing']} recorded {d3r['label']} {d3r.get('failing')}")
    if mine['D3']['reader_budget_credit'] != (d3r.get('reader_budget_credit') or {}):
        mm.append(f"D3 credit mine {mine['D3']['reader_budget_credit']} recorded {d3r.get('reader_budget_credit')}")
    d4r = dec['D4']
    if mine['D4']['label'] != d4r['label'] or mine['D4']['level'] != d4r.get('level'):
        mm.append(f"D4 mine {mine['D4']['label']}/{mine['D4']['level']} recorded {d4r['label']}/{d4r.get('level')}")
    if mine['outcome'] != dec['outcome_class']['label']:
        mm.append(f"outcome mine {mine['outcome']} recorded {dec['outcome_class']['label']}")
    return mm


# ---------------------------------------------------------------- ER1 grid validity (own)
def validity(bank, minimum_set, distinct):
    man = json.loads((V.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': man['reader_seeds'][f'V{v}k{k}'] for v in VERS for k in (0, 1)}
    errors = []
    forb = {SEED_BASE + off + 10 * v + k for off in (300, 400, 500, 600) for v in VERS for k in range(10)}
    sv = list(seeds.values())
    if len(set(sv)) != len(sv) or any(not isinstance(x, int) for x in sv):
        errors.append('seeds not distinct integers')
    if set(sv) & forb:
        errors.append('forbidden seed')
    if any(seeds[f'S{v}k{k}'] != SEED_BASE + 700 + 10 * v + k for v in VERS for k in (0, 1)):
        errors.append('seed differs from the registered +700 value')
    checks = []
    for cell, cres in bank['cells'].items():
        if not cell.endswith('k0'):
            continue
        v, rep, _ = parse(cell)
        other = cell[:-1] + '1'
        for member, ent in cres.items():
            for vw, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((bank['cells'].get(other) or {}).get(member) or {}).get(vw, {}).get('extras') or {}).get('score_sha256') or {}
                for rcp in SEED_DEP + SEED_INV:
                    if rcp in f0 and rcp in f1:
                        checks.append((rcp, vw, member, v, rep, f0[rcp] == f1[rcp]))
    eq = [c for c in checks if c[0] in SEED_DEP and c[5]]
    diff = {(c[2], c[1], c[3], c[4], c[0]) for c in checks if c[0] in SEED_DEP and not c[5]}

    def explained(c):
        rcp, vw, w, v, rep, _ = c
        a = any((w, vw, v, rep, r) in diff for r in SEED_DEP if r != rcp)
        b = any((w, x, v, rep, rcp) in diff for x in VIEWS if x != vw)
        n = (distinct.get(f'{w}|V{v}') or {}).get(vw)
        return a and b and isinstance(n, int)
    on_min = [c for c in eq if c[2] in minimum_set]
    unexpl = [c for c in eq if not explained(c)]
    unexpl_min = [c for c in unexpl if c[2] in minimum_set]
    nondet = [c for c in checks if c[0] in SEED_INV and not c[5]]
    if unexpl:
        errors.append(f'{len(unexpl)} unexplained seed-dependent equalities ({len(unexpl_min)} on the minimum set)')
    if [r for r in SEED_DEP if r not in {c[0] for c in checks}]:
        errors.append('seed-dependent recipes not checked')
    return {'effective': not errors, 'errors': errors, 'n_checks': len(checks), 'seed_dependent_equalities': len(eq),
            'minimum_set_equalities': len(on_min), 'explained_minimum_set': len(on_min) - len(unexpl_min),
            'explained_total': len(eq) - len(unexpl), 'nondeterministic_invariant_recipes': len(nondet),
            'minimum_set_items': sorted({f'{c[2]}|{c[1]}|{c[0]}' for c in on_min})}


def distinct_counts(bank_members):
    """Own label-free distinct-input counts on MODEL_SELECT per member / version / view (income, occupation, pair)."""
    import v4_utility as U
    out = {}
    names = sorted(bank_members[0])
    for name in names:
        for v in VERS:
            m = bank_members[v].get(name) or {}
            rel = member_release(name, m, v, U)
            if rel is None:
                continue
            ii, ni = V.row_ids(rel['income'])
            io, no = V.row_ids(rel['occupation'])
            out[f'{name}|V{v}'] = {'income': ni, 'occupation': no, 'pair': int(len(np.unique(ii * (no + 1) + io)))}
    return out


REF_SPEC = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
            'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'U0': 'REF:U0',
            'Ucal': 'REF:Ucal'}


def member_release(name, m, v, U):
    if name in REF_SPEC:
        return U.comparator(REF_SPEC[name], v, None)
    nm = m.get('nrm_member') or {}
    if nm.get('type') == 'nrm':
        return U.comparator(f'NRM:{nm["family"]}/{nm["config"]}', v, None)
    fm = m.get('frc_member') or m.get('member') or {}
    cfg = fm.get('config') or ({'TWIN': 'TWB_C64'}.get(name, name))
    p = V.RUNS / 'FA' / cfg / f'V{v}' / 'RELEASE.json'
    return U.comparator(f'FA/{cfg}', v, None) if p.is_file() else None


# ---------------------------------------------------------------- run
def run(tag, bank_id):
    bank = load_bank(bank_id)
    qdec = json.loads((V.RESULTS / 'decisions' / 'QUAL_DECISION.json').read_text())
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'bank': bank_id}
    # 1. every paired entry of the grid from the bootstrap draws
    worst_d, worst_s, n = 0.0, 0.0, 0
    for cand, comps in bank['paired'].items():
        for comp, rec in comps.items():
            mine = my_paired(bank, cand, comp)
            for vw in VIEWS:
                for c, x in rec[vw]['cells'].items():
                    worst_d = max(worst_d, abs(x - mine[vw]['cells'][c]))
                    worst_s = max(worst_s, abs(rec[vw]['cell_se'][c] - mine[vw]['cell_se'][c]))
                    n += 1
    record['paired'] = {'n_cell_values': n, 'max_abs_value_diff': worst_d, 'max_abs_se_diff': worst_s}
    checks.append({'id': f'V6.GRID_PAIRED.{bank_id}__{tag}',
                   'what': f'{bank_id}: every paired cell value (pair benefit / recipient difference) and household-bootstrap SE (SD of '
                           f'the paired draw differences, ddof 1) recomputed from the cell SCORE AUCs and BOOT.npz',
                   'inputs': V.inputs_sha([V.RUNS / 'GRID' / bank_id / f'S{v}' / f for v in VERS for f in ('GRID.json', 'BOOT.npz')]),
                   'result': record['paired'], 'numerical_status': 'PASS' if (worst_d <= TOL and worst_s <= TOL and n) else 'FAIL',
                   'procedural_status': 'NOT_APPLICABLE'})
    # 2. grid validity under ER1
    plan_main = json.loads((V.RESULTS / 'claims' / 'PLAN_MAIN.json').read_text())
    dist = distinct_counts(bank['members'])
    gv = validity(bank, set(plan_main.get('minimum_set') or []), dist)
    record['validity'] = gv
    recs = {}
    for pid, out in (('MAIN', 'MAIN_ER1'), ('R1', 'R1_ER1')):
        p = V.RESULTS / 'claims' / out / 'DECISIONS.json'
        recs[pid] = json.loads(p.read_text()) if p.is_file() else None
    rgv = (recs['MAIN'] or {}).get('grid_validity') or {}
    gv_mm = []
    if gv['effective'] != rgv.get('effective'):
        gv_mm.append(f'effective mine {gv["effective"]} recorded {rgv.get("effective")}')
    if gv['explained_minimum_set'] != ((rgv.get('explained_minimum_set_equalities') or {}).get('count')):
        gv_mm.append(f'explained minimum-set equalities mine {gv["explained_minimum_set"]} recorded '
                     f'{(rgv.get("explained_minimum_set_equalities") or {}).get("count")}')
    if gv['explained_total'] != rgv.get('explained_equalities') + (rgv.get('explained_minimum_set_equalities') or {}).get('count', 0) \
            and gv['explained_total'] != rgv.get('explained_equalities'):
        gv_mm.append(f'explained equalities mine {gv["explained_total"]} recorded {rgv.get("explained_equalities")}')
    checks.append({'id': f'V6.GRID_VALIDITY_ER1.{bank_id}__{tag}',
                   'what': f'{bank_id}: ER1 grid validity from the score fingerprints with own (i) seeds, (ii)-(iii) and the ER1 '
                           f'explanation (a) + (b) + (c) using my own distinct-input counts on MODEL_SELECT',
                   'inputs': V.inputs_sha([V.RESULTS / 'READER_SPLITS.json']), 'result': {**gv, 'mismatches': gv_mm,
                                                                                           'recorded_checks': (recs['MAIN'] or {}).get('grid_validity_checks')},
                   'numerical_status': 'PASS' if not gv_mm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
    # 3. decisions, both readings
    for pid in ('MAIN', 'R1'):
        rec = recs[pid]
        if rec is None:
            continue
        plan = json.loads((V.RESULTS / 'claims' / f'PLAN_{pid}.json').read_text())
        s2 = plan['prc_s2']
        ev = json.loads((V.RUNS / 'FA' / s2 / 'EVAL_MS.json').read_text())
        costs = {}
        for m, spec in plan['members'].items():
            e = spec['eval']
            if m == s2 or e.startswith('ref:'):
                continue
            key = e.split(':', 1)[1] if e.startswith('frc:') else e.replace('nrm:', 'NRM:')
            per = {v: ((ev['versions'][str(v)].get('paired_costs') or {}).get(key) or {}).get('costs') for v in VERS}
            if all(per.values()):
                costs[m] = per
        g1 = all(ev['versions'][str(v)]['point']['point_pass_12_plus_2'] for v in VERS)
        mine, grid = decide(plan, bank, costs, qdec, g1)
        mm = compare_decisions(mine, rec)
        # nested LOOKUP benefits of the plan against the selection records (AUC_control - AUC_prc)
        nl_mm = []
        for m, per in (plan.get('nested_lookup') or {}).items():
            for v in VERS:
                sel = json.loads((V.RESULTS / 'fa' / f'SELECTION_V{v}.json').read_text())
                prc_auc = sel['nested_lookup_pair'][sel['selection']['PRC']['S2']]['auc']
                if m == 'TWIN':
                    ctrl = sel['nested_lookup_pair']['PRC_TWIN']['auc']
                elif m == 'UNT_BR':
                    u = json.loads((V.RESULTS / 'fa' / f'UNT_SELECTION_V{v}.json').read_text())
                    ctrl = u['nested_lookup_pair'][u['bracket']]['auc']
                else:
                    rj = V.RUNS / 'FA' / m / f'V{v}' / 'RELEASE.json'
                    if not rj.is_file():
                        continue
                    ctrl = sel['nested_lookup_pair'][json.loads(rj.read_text())['selected']]['auc']
                if abs((ctrl - prc_auc) - float(per[str(v)])) > TOL:
                    nl_mm.append(f'nested_lookup {m} V{v} plan {per[str(v)]} selection {ctrl - prc_auc}')
        record[pid] = {'mine': mine, 'mismatches': mm, 'nested_lookup_mismatches': nl_mm}
        checks.append({'id': f'V6.DECISIONS.{pid}_ER1__{tag}',
                       'what': f'{pid} reading under ER1: D2 conjuncts (TWIN comparison, SHUF / HQ / ORD frontiers with pooled '
                               f'total-NLL brackets and secondary metrics, F4, reader targeting), D3 (G4 + reader-budget credit), '
                               f'D4 (G5), outcome class, all from my own grid replay and the V4-verified costs; the plan\'s nested '
                               f'LOOKUP benefits against the selection records',
                       'inputs': V.inputs_sha([V.RESULTS / 'claims' / f'PLAN_{pid}.json',
                                               V.RESULTS / 'claims' / f'{pid}_ER1' / 'DECISIONS.json']),
                       'result': {'D2': mine['D2']['label'], 'D3': mine['D3']['label'], 'D4': mine['D4']['label'],
                                  'outcome': mine['outcome'], 'TWIN': mine['D2']['comparisons']['TWIN'],
                                  'frontiers': {n: {k: mine['D2']['comparisons'][n].get(k) for k in ('label', 'reason', 'point', 'path')}
                                                for n in D2_ORDER[1:]},
                                  'targeting': mine['D2']['targeting'], 'f4_pass': mine['D2']['f4_pass'],
                                  'reference_comparisons': mine['reference_comparisons'], 'D4_detail': mine['D4'],
                                  'mismatches': mm, 'nested_lookup_mismatches': nl_mm},
                       'numerical_status': 'PASS' if not (mm or nl_mm) else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V6_GRID_CLAIMS__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], (c['result'].get('mismatches') or [])[:6], c['result'].get('nested_lookup_mismatches'))
    return 0


# ---------------------------------------------------------------- invented-data selftest
def selftest():
    ok = {}
    e = {'cells': {c: 0.01 for c in REG}, 'cell_se': {c: 0.002 for c in REG}}
    a = aggregate(e)
    ok['constant_cells'] = abs(a['mean'] - 0.01) < 1e-15 and abs(a['se_hh'] - 0.002) < 1e-15 and a['se_rs'] == 0.0 and \
        abs(a['se_agg'] - 0.002) < 1e-15
    ok['positive_resolved'] = benefit(e)['label'] == 'AGG_POSITIVE'
    e2 = {'cells': {c: (0.01 if c.startswith('S0') else -0.002) for c in REG}, 'cell_se': {c: 0.002 for c in REG}}
    ok['point_only_when_a_version_negative'] = benefit(e2)['label'] == 'AGG_POSITIVE_POINT_ONLY'
    z = {'cells': {c: 0.0 for c in REG}, 'cell_se': {c: 0.0 for c in REG}}
    ok['alias_zero'] = benefit(z)['label'] == 'ALIAS_ZERO' and recipient(z)['gate_pass']
    f = {'cells': {c: 0.0001 for c in REG}, 'cell_se': {c: 0.0001 for c in REG}}
    ok['floor_binds'] = abs(aggregate(f)['se_agg'] - FLOOR) < 1e-15
    w = {'cells': {c: 0.02 for c in REG}, 'cell_se': {c: 0.001 for c in REG}}
    ok['recipient_worse'] = recipient(w)['label'] == 'RECIPIENT_AGG_RESOLVED_WORSE'
    ok['recipient_edge'] = recipient({'cells': {c: 0.009 for c in REG}, 'cell_se': {c: 0.001 for c in REG}})['label'] == 'RECIPIENT_AGG_EDGE'
    ok['target_point_only'] = target({'pair': {'cells': {c: 0.021 for c in REG}, 'cell_se': {c: 0.002 for c in REG}}})['label'] == \
        'TARGET_AGG_POINT_ONLY'
    # block structure: se_rs from block means
    e3 = {'cells': {c: (0.0 if 'R1' in c else 0.02) for c in REG}, 'cell_se': {c: 0.0 for c in REG}}
    ok['se_rs_block_means'] = abs(aggregate(e3)['se_rs'] - 0.01) < 1e-15
    # frontier: no control spends as much -> not separated; bracket is the cheapest one spending at least as much
    costs = {q: {v: {'total|nll': {'value': val, 'se': 0.001}, **{m: {'value': 0.0, 'se': 0.001} for m in SECONDARY}} for v in VERS}
             for q, val in (('A', 0.002), ('B', -0.001), ('C', -0.004))}
    grid = {q: {'pair': {'cells': {c: 0.01 for c in REG}, 'cell_se': {c: 0.002 for c in REG}}} for q in costs}
    fr = frontier(costs, grid, ['A', 'B', 'C'])
    ok['bracket_cheapest_spending_at_least_as_much'] = fr['point'] == 'B' and fr['label'] == 'DOMINATES_CONTROL_FRONTIER'
    ok['spends_more_than_every_control'] = frontier(costs, grid, ['A'])['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    ok['version_means_blocks'] = version_means({c: (1.0 if c.endswith('k0') else 0.0) for c in REG}) == {0: 0.5, 1: 0.5}
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {k: bool(x) for k, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--bank', default='FRC1_MAIN')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag, a.bank)


if __name__ == '__main__':
    sys.exit(main())

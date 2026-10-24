"""Verifier (label-free): selections (check 2), the per-comparison aggregates and labels (check 3), the decisions of each reading
(check 5) and the label-access log (check 6), recomputed with the verifier's own code from the frozen fit records, plans, decision
records, the verifier's own MODEL_SELECT recomputation (work/ms_recomputed.json, written by the queue unit v_ms.py) and the
access / queue logs. No label is read here; nothing from sar.rules, sar.claims, nrm.rules or sar.devreaders is imported.

    python3 results/pcrl_strong_anchor_release_v1/roles/verifier/code/v_decide.py --out results/.../roles/verifier/work/decide.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES, RUNS = V.RES, V.RUNS
TOL = 1e-9
CHECKS = []


def load(p):
    return json.loads(Path(p).read_text())


def rel(p):
    p = Path(p)
    return str(p.relative_to(V.ROOT)) if p.is_relative_to(V.ROOT) else str(p)


def ins(*paths):
    return {rel(p): V.fsha(p) for p in paths}


def add(cid, what, inputs, recomputed, record, ok, tol=None, note=None):
    CHECKS.append({'id': cid, 'what': what, 'inputs_sha256': inputs, 'recomputed': recomputed, 'record': record,
                   'status': 'PASS' if ok else 'FAIL', 'tolerance': tol, **({'note': note} if note else {})})
    return ok


def num_eq(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# ====================================================================== check 2: selections
FA_V1_BUDGET = {'occ_nll': 0.0040, 'occ_brier': 0.0025, 'occ_acc': 0.0050, 'inc_nll': 0.0060, 'inc_brier': 0.0030, 'inc_acc': 0.0035}
FA_V1_GUARD = {'occ_brier_men': 0.0005, 'occ_brier_women': 0.0050}
FB_V1_BUDGET = {'occ_nll': 0.0060, 'inc_nll': 0.0050, 'occ_brier': 0.0025, 'inc_brier': 0.0025, 'occ_brier_men': 0.0028,
                'occ_brier_women': 0.0028, 'occ_acc': 0.0060, 'inc_acc': 0.0060}
FB_R1_BUDGET = {'occ_nll': 0.0060, 'occ_brier': 0.0025, 'occ_brier_men': 0.0028, 'occ_brier_women': 0.0028, 'occ_acc': 0.0060}


def meets(row, budget):
    return {c: bool(row[c]['estimate'] + 2.0 * row[c]['se'] <= budget[c]) for c in budget}


def fa_reg():
    for v in (0, 1):
        p = RES / 'fa' / f'FIT_V{v}.json'
        card = RES / 'cards' / 'FA_TFR_v1.md'
        f = load(p)
        s = f['selection']
        i = ins(p, card)
        add(f'S2_FA_REG_v{v}_budgets', f'FA_REG v{v}: the record\'s selection budgets equal card FA_TFR_v1 section 6', i,
            {**FA_V1_BUDGET, **FA_V1_GUARD}, {**s['budget_cal'], **s['guard_did']},
            s['budget_cal'] == FA_V1_BUDGET and s['guard_did'] == FA_V1_GUARD, 0)
        rho_star, table, first_fail = None, {}, None
        for rho in s['rho_grid']:
            row = s['per_rho'].get(str(float(rho)))
            if row is None:
                break
            a = meets(row['cal_spend_over_SUCC'], FA_V1_BUDGET)
            b = meets(row['mf_guard_spend_over_reference'], FA_V1_GUARD)
            table[str(rho)] = {'part_a': a, 'part_b': b,
                               'part_a_bounds': {c: row['cal_spend_over_SUCC'][c]['estimate'] + 2 * row['cal_spend_over_SUCC'][c]['se'] for c in FA_V1_BUDGET},
                               'part_b_bounds': {c: row['mf_guard_spend_over_reference'][c]['estimate'] + 2 * row['mf_guard_spend_over_reference'][c]['se'] for c in FA_V1_GUARD},
                               'guard_reference_s_equals_bracket_lo': row.get('guard_reference_s') == row.get('bracket_lo')}
            add(f'S2_FA_REG_v{v}_rho{rho}_flags', f'FA_REG v{v} rho {rho}: parts (a) and (b) re-applied to the recorded spends', i,
                {'meets_cal': all(a.values()), 'meets_guard': all(b.values())},
                {'meets_cal': row['meets_cal'], 'meets_guard': row['meets_guard']},
                all(a.values()) == row['meets_cal'] and all(b.values()) == row['meets_guard'], 0)
            if all(a.values()) and all(b.values()):
                rho_star = rho
            else:
                first_fail = rho
                break
        add(f'S2_FA_REG_v{v}_rho_star', f'FA_REG v{v}: rho* re-selected (largest rho with every smaller rho passing (a) and (b))', i,
            {'rho_star': rho_star, 'status': 'NO_FEASIBLE_STRENGTH' if rho_star is None else 'FITTED', 'table': table},
            {'rho_star': s['rho_star'], 'status': f['cand']['status']},
            rho_star == s['rho_star'] and (f['cand']['status'] == ('NO_FEASIBLE_STRENGTH' if rho_star is None else 'FITTED')), 0)
        t = table.get('0.3', {})
        add(f'S2_FA_REG_v{v}_rho0p3_fails_a_and_b', f'FA_REG v{v}: rho 0.3 fails both part (a) and part (b) (coordinator brief)', i,
            {'a_failed': sorted(c for c, ok in t.get('part_a', {}).items() if not ok),
             'b_failed': sorted(c for c, ok in t.get('part_b', {}).items() if not ok)}, 'fails (a) and (b)',
            first_fail == 0.3 and not all(t['part_a'].values()) and not all(t['part_b'].values()), 0)


def fb_reg():
    for v in (0, 1):
        p = RES / 'fb' / f'FIT_PFD_V{v}.json'
        card = RES / 'cards' / 'FB_PFD_v1.md'
        f = load(p)
        s = f['selection']
        i = ins(p, card)
        add(f'S2_FB_REG_v{v}_budget', f'FB_REG v{v}: the record\'s budget equals card FB_PFD_v1 item 6', i, FB_V1_BUDGET, s['budget'],
            s['budget'] == FB_V1_BUDGET, 0)
        k_star, fail = None, None
        for k in range(1, 11):
            m = meets(s['path'][f'PFD_E{k}'], FB_V1_BUDGET)
            if all(m.values()):
                k_star = k
            else:
                fail = {'checkpoint': f'E{k}', 'failed': sorted(c for c, ok in m.items() if not ok),
                        'bounds': {c: s['path'][f'PFD_E{k}'][c]['estimate'] + 2 * s['path'][f'PFD_E{k}'][c]['se'] for c in sorted(c for c, ok in m.items() if not ok)}}
                break
        add(f'S2_FB_REG_v{v}_k_star', f'FB_REG v{v}: k* re-selected (largest E_k with E_1..E_k within budget at estimate + 2 SE)', i,
            {'k_star': k_star, 'first_failure': fail}, {'k_star': s['k_star'], 'status': f['cand']['status']},
            k_star == s['k_star'] and f['cand']['status'] == ('NO_FEASIBLE_STRENGTH' if k_star is None else 'FITTED'), 0)
        if v == 1:
            add('S2_FB_REG_v1_E1_fails', 'FB_REG v1: E1 fails the budget (coordinator brief: version 1 fails E1)', i, fail,
                'E1 fails', fail is not None and fail['checkpoint'] == 'E1', 0)


def reg_decisions():
    for fam, plan, card in (('FA', 'FA_REG', 'FA_TFR_v1.md'), ('FB', 'FB_REG', 'FB_PFD_v1.md')):
        dp, pp = RES / 'claims' / plan / 'DECISIONS.json', RES / 'claims' / f'PLAN_{plan}.json'
        d, pl = load(dp)['decisions'], load(pp)
        i = ins(dp, pp)
        exp = {'D1': 'NO_FEASIBLE_STRENGTH', 'UM': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D2': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH',
               'D3': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D4': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH',
               'D5': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D6': 'NOT_CLAIMED', 'D2R': 'NOT_RUN', 'class': 'TASK_INFEASIBLE'}
        got = {k: d[k]['label'] for k in ('D1', 'UM', 'D2', 'D3', 'D4', 'D5', 'D6', 'D2R')}
        got['class'] = d['outcome_class']['label']
        stat = pl['cand_status_by_version']
        no_cand = not all(stat[str(v)] == 'FITTED' for v in (0, 1))
        add(f'S5_{plan}_labels', f'{plan}: every decision label recomputed (no candidate in both versions -> NO_FEASIBLE_STRENGTH; '
            'class TASK_INFEASIBLE; registered, unprefixed)', i, exp, got, no_cand and got == exp and d['registered'] is True
            and 'NO_STRENGTH_MEETS_MODEL_FIT_BUDGET' in d['outcome_class']['flags'], 0)


def fb_r1_selection():
    plan = load(RES / 'claims' / 'PLAN_FB_R1.json')
    for v in (0, 1):
        p = RES / 'fb' / f'FIT_PFO_V{v}.json'
        card = RES / 'cards' / 'FB_PFO_R1.md'
        f = load(p)
        s = f['selection']
        i = ins(p, card, RES / 'claims' / 'PLAN_FB_R1.json')
        add(f'S2_FB_R1_v{v}_budget', f'FB_R1 v{v}: the record\'s budget equals card FB_PFO_R1 item 6', i, FB_R1_BUDGET, s['budget'],
            s['budget'] == FB_R1_BUDGET, 0)
        steps = sorted(int(k.split('_S')[1]) for k in s['path'] if k.startswith('PFO_S') and int(k.split('_S')[1]) > 0)
        k_star, fail = None, None
        for st in steps:
            m = meets(s['path'][f'PFO_S{st}'], FB_R1_BUDGET)
            if all(m.values()):
                k_star = st
            else:
                bad = sorted(c for c, ok in m.items() if not ok)
                fail = {'checkpoint': f'S{st}', 'failed': bad,
                        'bounds': {c: s['path'][f'PFO_S{st}'][c]['estimate'] + 2 * s['path'][f'PFO_S{st}'][c]['se'] for c in bad}}
                break
        add(f'S2_FB_R1_v{v}_k_star', f'FB_R1 v{v}: k* re-selected on the 10-step path (S10..S400) from the recorded out-of-fold spends', i,
            {'k_star': k_star, 'first_failure': fail},
            {'k_star': s['k_star'], 'cand_strength': f['cand']['strength'], 'plan_strength': plan['strengths'][0]['strength'][str(v)]},
            k_star == s['k_star'] == f['cand']['strength'] == plan['strengths'][0]['strength'][str(v)] == 50, 0)
        inc = {c: s['path'][f'PFO_S{k_star}'][c] for c in ('inc_acc', 'inc_brier', 'inc_nll')} if k_star else None
        add(f'S2_FB_R1_v{v}_income_spend_zero_over_start', f'FB_R1 v{v}: CAND\'s income part of the spend over START is 0 and the '
            'fit records the income vectors as START\'s (family float64 law)', i,
            {'income_part': f['cand']['spend_over_start']['income_part'], 'flag': s.get('income_vectors_equal_START_bitwise')},
            'income part 0', f['cand']['spend_over_start']['income_part'] == 0.0 and bool(s.get('income_vectors_equal_START_bitwise')), 0,
            note=f'income spends over the fold start at k*: {inc}')
        for key, cfg, keyfun in (('CTRL', 'FB_PFO_DQ_HI', float), ('ALT', 'FB_PFO_RAWJ_HI', int)):
            pr = f['paths'][key]
            sc = pr['cand_spend_over_start']['value']
            order = sorted(pr['grid'], key=keyfun)
            rows = {g: pr['grid'][g] for g in order}
            below = [g for g in order if rows[g]['spend_over_start'] <= sc]
            lo = max(below, key=lambda g: rows[g]['spend_over_start']) if below else None
            above = [g for g in order if rows[g]['spend_over_start'] >= sc + 2 * rows[g]['se_diff_vs_cand']]
            hi = min(above, key=lambda g: rows[g]['spend_over_start']) if above else max(order, key=lambda g: rows[g]['spend_over_start'])
            flags = [] if above else ['CONTROL_PATH_MAY_NOT_BRACKET']
            role = plan['roles'][f'FB_R1_{key}_HI'][str(v)]
            add(f'S2_FB_R1_v{v}_{key}_bracket', f'FB_R1 v{v} {key}: LO / HI re-selected on the MODEL_FIT out-of-fold grid (LO largest '
                'spend <= CAND\'s; HI smallest spend >= CAND\'s + 2 SE of the point\'s paired difference vs CAND); HI saved as the plan role',
                i, {'cand_spend': sc, 'lo': lo, 'hi': hi, 'hi_spend': rows[hi]['spend_over_start'], 'flags': flags, 'role': f'FB/{cfg}'},
                {'lo': pr['lo'], 'hi': pr['hi'], 'hi_config': pr['hi_config'], 'flags': pr['flags'], 'role': role},
                sc > 0 and keyfun(lo) == pr['lo'] and keyfun(hi) == pr['hi'] and pr['hi_config'] == cfg and role == f'FB/{cfg}'
                and not flags and pr['selection_data'] == 'MODEL_FIT_OOF', 0)
            # fineness on the declared selection data (RULES 4.4; lower saved point = the origin, LO not saved)
            fin = plan['fineness_inputs'][key][str(v)]
            need = sc + 2 * rows[hi]['se_diff_vs_cand']
            lows = [0.0] + [rows[g]['spend_over_start'] for g in order if rows[g]['spend_over_start'] <= sc]
            ups = [rows[g]['spend_over_start'] for g in order if rows[g]['spend_over_start'] >= sc + 2 * rows[g]['se_diff_vs_cand']]
            lower, upper = 0.0, rows[hi]['spend_over_start']
            exact = abs(max(lows) - lower) <= 1e-9 and (not ups or abs(min(ups) - upper) <= 1e-9)
            if upper < need:
                lab = 'CONTROL_PATH_MAY_NOT_BRACKET'
            elif lower > sc or not exact:
                lab = 'SAVED_POINT_RULE_NOT_FOLLOWED'
            else:
                lo_ok, hi_ok = lower >= 0.75 * sc, upper <= need + 0.25 * sc
                lab = 'FINE' if lo_ok and hi_ok else ('COARSE_BELOW' if hi_ok else ('COARSE_ABOVE' if lo_ok else 'COARSE_BOTH'))
            plan_grid_ok = (sorted((a['spend'], a['se']) for a in fin['grid']) ==
                            sorted((rows[g]['spend_over_start'], rows[g]['se_diff_vs_cand']) for g in order))   # order-free (a list)
            add(f'S2_FB_R1_v{v}_{key}_fineness', f'FB_R1 v{v} {key}: fineness label on the declared data (lower = origin since LO is not '
                'saved); the plan\'s fineness inputs equal the fit record', i,
                {'label': lab, 'largest_qualifying_lower': max(lows), 'saved_lower': lower, 'upper': upper, 'need': need},
                {'plan_inputs': {k: fin[k] for k in ('cand', 'lower', 'upper', 'upper_se')}, 'decision_flag': f'CONTROL_POINT_RULE_NOT_FOLLOWED_{key}'},
                plan_grid_ok and num_eq(fin['cand'], sc, 0) and num_eq(fin['upper'], upper, 0) and fin['lower'] == 0.0
                and lab == 'SAVED_POINT_RULE_NOT_FOLLOWED', 0,
                note='reported, never gating: with only HI saved (specification budget) the origin is the lower point, and the full '
                     'recorded grid holds a larger qualifying lower point, so the frozen fineness rule reads SAVED_POINT_RULE_NOT_FOLLOWED '
                     '(PROTOCOL 10 anticipated COARSE_BELOW for this case)')


def fa_r1_selection(ms):
    plan = load(RES / 'claims' / 'PLAN_FA_R1.json')
    for v in (0, 1):
        p = RES / 'fa' / f'FIT_R1_V{v}.json'
        card = RES / 'cards' / 'FA_TFRO_R1.md'
        f = load(p)
        s = f['selection']
        i = ins(p, card, RES / 'claims' / 'PLAN_FA_R1.json', V.WORK / 'ms_recomputed.json')
        u = ms['utility']['SUCC'][str(v)]
        mine = {'occ_nll': 0.010 - max(u['occupation|U0|nll_excess']['value'], u['occupation|Ucal|nll_excess']['value']),
                'occ_brier': 0.005 - max(u['occupation|U0|brier_excess']['value'], u['occupation|Ucal|brier_excess']['value']),
                'occ_acc': 0.010 - max(u['occupation|U0|accuracy_loss']['value'], u['occupation|Ucal|accuracy_loss']['value']),
                'men': 0.005 - u['guard|sex0']['value'], 'women': 0.005 - u['guard|sex1']['value']}
        add(f'S2_FA_R1_v{v}_margins', f'FA_R1 v{v}: the budgets (SUCC\'s margins to the caps) recomputed from SUCC\'s MODEL_SELECT utility '
            '(verifier recomputation)', i, mine, s['margins'], all(num_eq(mine[k], s['margins'][k], 1e-9) for k in mine), 1e-9)
        M = s['margins']
        rho_star, table = None, {}
        for rho in s['rho_grid']:
            row = s['per_rho'].get(str(float(rho)))
            if row is None:
                break
            cal = row['cal_spend_over_SUCC']
            a = {c: bool(cal[c]['estimate'] + 2 * cal[c]['se'] <= M[c]) for c in ('occ_nll', 'occ_brier', 'occ_acc')}
            inc0 = all(cal[c]['estimate'] == 0.0 and cal[c]['se'] == 0.0 for c in ('inc_acc', 'inc_brier', 'inc_nll'))
            ge = row['guard_estimate']
            L, sL, Cc, sC, pm = cal['occ_brier']['estimate'], cal['occ_brier']['se'], ge['mf_men'] - ge['mf_women'], ge['se_C'], ge['p_men']
            dm, sm = L + (1 - pm) * Cc, math.sqrt(sL ** 2 + (1 - pm) ** 2 * sC ** 2)
            dw, sw = L - pm * Cc, math.sqrt(sL ** 2 + pm ** 2 * sC ** 2)
            b = {'men': bool(dm + 2 * sm <= M['men']), 'women': bool(dw + 2 * sw <= M['women'])}
            consistent = (num_eq(ge['L_cal'], L, 0) and num_eq(ge['se_L'], sL, 0) and num_eq(ge['C_mf'], Cc, 1e-15)
                          and num_eq(ge['men'][0], dm, 1e-15) and num_eq(ge['men'][1], sm, 1e-15)
                          and num_eq(ge['women'][0], dw, 1e-15) and num_eq(ge['women'][1], sw, 1e-15))
            passes = all(a.values()) and all(b.values()) and inc0
            table[str(rho)] = {'part_a': a, 'part_b': b, 'income_spends_zero': inc0, 'delta_men': [dm, sm], 'delta_women': [dw, sw],
                               'bound_men': dm + 2 * sm, 'bound_women': dw + 2 * sw}
            add(f'S2_FA_R1_v{v}_rho{rho}', f'FA_R1 v{v} rho {rho}: part (a) and the guard estimate Delta = L +/- weight x C with its SE '
                're-derived from the recorded L, C, p, SE_L, SE_C, and the pass flags', i,
                {'part_a': a, 'part_b': b, 'passes': passes, 'delta_men': [dm, sm], 'delta_women': [dw, sw]},
                {'part_a': row['part_a'], 'part_b': row['part_b'], 'passes': row['passes'], 'men': ge['men'], 'women': ge['women']},
                consistent and a == row['part_a'] and b == row['part_b'] and passes == row['passes'], 1e-15)
            if passes:
                rho_star = rho
            else:
                break
        add(f'S2_FA_R1_v{v}_rho_star', f'FA_R1 v{v}: rho* re-selected (scan stops at the first failing rho)', i,
            {'rho_star': rho_star, 'table': table},
            {'rho_star': s['rho_star'], 'cand': f['cand']['rho_star'], 'plan': plan['strengths'][0]['strength'][str(v)]},
            rho_star == s['rho_star'] == f['cand']['rho_star'] == plan['strengths'][0]['strength'][str(v)] == {0: 1.0, 1: 0.6}[v], 0)
        # CTRL bracket on CALIBRATION paired total NLL (ties to the smaller s)
        c = f['ctrl']
        sc = f['cand']['cal_spend_over_SUCC']['spend_over_start']
        order = sorted(c['path'], key=float)
        rows = c['path']
        below = [g for g in order if rows[g]['spend_over_start'] <= sc]
        lo = max(below, key=lambda g: rows[g]['spend_over_start']) if below else None
        above = [g for g in order if rows[g]['spend_over_start'] >= sc + 2 * rows[g]['se_diff_vs_cand']]
        hi = min(above, key=lambda g: rows[g]['spend_over_start']) if above else None
        flags = ([] if above else ['CONTROL_PATH_MAY_NOT_BRACKET']) + ([] if below else ['CONTROL_PATH_NO_LO_POINT'])
        alias = lambda g: None if g is None else ('START' if float(g) == 0.0 else ('FA_TFRO_R1_CONT' if float(g) == 1.0 else None))
        role = plan['roles']['FA_R1_CTRL_LO'][str(v)]
        add(f'S2_FA_R1_v{v}_CTRL_bracket', f'FA_R1 v{v} CTRL: LO / HI re-selected on the CALIBRATION path (spend values; ties to the '
            'smaller s; endpoints aliased)', i,
            {'cand_spend': sc, 'lo': None if lo is None else float(lo), 'hi': None if hi is None else float(hi), 'hi_alias': alias(hi),
             'flags': flags}, {'lo': c['lo'], 'hi': c['hi'], 'flags': c['flags'], 'saved_HI': c['saved']['HI'], 'role_LO': role},
            (lo is not None and float(lo) == c['lo']) and ((hi is None and c['hi'] is None) or (hi is not None and float(hi) == c['hi']))
            and sorted(flags) == sorted(c['flags']) and role == 'FA/FA_TFRO_R1_TASK_LO'
            and (c['saved']['HI'].get('alias') == alias(hi)), 0)


# ====================================================================== checks 3 and 5: aggregates and decisions
def merged(ms, bank):
    out = {}
    for v in ('0', '1'):
        for comp, views in ms['grid'][bank][v]['paired'].items():
            for view, e in views.items():
                o = out.setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                o['cells'].update(e['cells'])
                o['cell_se'].update(e['cell_se'])
    return out


def nl_merged(ms, bank, key):
    out = {}
    for v in ('0', '1'):
        for comp, cells in ms['grid'][bank][v][key].items():
            out.setdefault(comp, {}).update(cells)
    return out


def comparison(entry):
    ab = V.aggregate(entry['pair']['cells'], entry['pair']['cell_se'])
    rec = {'benefit': V.benefit_label(ab), 'benefit_mean': ab['mean'], 'benefit_se': ab['se_agg'], 'agg': ab, 'recipient': {}}
    gate = True
    for w in ('income', 'occupation'):
        ar = V.aggregate(entry[w]['cells'], entry[w]['cell_se'])
        lab, g = V.recipient_label(ar)
        rec['recipient'][w] = lab
        rec['recipient_point_by_version_' + w] = {v: ar['per_version'][v]['mean'] for v in (0, 1)}
        gate = gate and g
    rec['gate'] = gate
    rec['pass'] = rec['benefit'] == 'AGG_POSITIVE' and gate
    return rec


def version_means(cells):
    return {v: sum(x for c, x in cells.items() if c[1] == str(v)) / len([c for c in cells if c[1] == str(v)]) for v in (0, 1)}


METRIC_OF = {'guard|sex0': 'occupation|men_brier', 'guard|sex1': 'occupation|women_brier'}


def cond_metric(c):
    if c in METRIC_OF:
        return METRIC_OF[c]
    t, _, m = c.split('|')
    return f"{t}|{ {'accuracy_loss': 'acc_loss', 'nll_excess': 'nll', 'brier_excess': 'brier'}[m] }"


def floors():
    r = load(V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json')
    return r['reserve']['floor']['se'], int(r['reserve']['floor']['households']), float(r['reserve']['multiplier_k'])


def d1_parts(U, name, households):
    fl, G0, k = floors()
    failed, conf, ratios = [], {}, {}
    for v in (0, 1):
        for c in V.CONDS:
            x = U[name][str(v)][c]
            if not x['value'] <= x['cap']:
                failed.append(f'v{v}|{c}')
            st = 'PASS' if x['value'] + 2 * x['se'] <= x['cap'] else ('VIOLATION' if x['value'] - 2 * x['se'] > x['cap'] else 'UNRESOLVED')
            conf[f'v{v}|{c}'] = st
            fse = fl[str(v)][c] * math.sqrt(G0 / households)
            ratios[f's{v}|{c}'] = (x['cap'] - x['value']) / max(x['se'], fse)
    clab = ('CONFIDENCE_VIOLATION' if any(s == 'VIOLATION' for s in conf.values()) else
            'CONFIDENCE_ALL_PASS' if all(s == 'PASS' for s in conf.values()) else 'CONFIDENCE_UNRESOLVED')
    per = {v: min(r for c, r in ratios.items() if c.startswith(f's{v}|')) for v in (0, 1)}
    met = [v for v in (0, 1) if per[v] >= k]
    status = 'RESERVE_MET' if len(met) == 2 else ('RESERVE_ONE_SEED_ONLY' if met else 'RESERVE_NOT_MET')
    binding_v = {v: min((c for c in ratios if c.startswith(f's{v}|')), key=lambda c: ratios[c]).split('|', 1)[1] for v in (0, 1)}
    return {'point': 'POINT_ELIGIBLE' if not failed else 'POINT_INELIGIBLE', 'failed': failed, 'confidence': clab,
            'n_pass': sum(s == 'PASS' for s in conf.values()), 'unresolved': sorted(c for c, s in conf.items() if s == 'UNRESOLVED'),
            'violations': sorted(c for c, s in conf.items() if s == 'VIOLATION'), 'reserve_min_ratio': per, 'reserve_status': status,
            'reserve_binding': min(ratios, key=lambda c: ratios[c]), 'binding_by_version': binding_v,
            'point_margin': {v: min(((U[name][str(v)][c]['cap'] - U[name][str(v)][c]['value']) / U[name][str(v)][c]['cap'], c) for c in V.CONDS) for v in (0, 1)}}


def check_d1(tag, U, rec_d1, cand, start, cont, households, i, prefix):
    P = {r: d1_parts(U, n, households) for r, n in (('cand', cand), ('start', start), ('cont', cont))}
    ok = True
    mine, record = {}, {}
    for r in ('cand', 'start', 'cont'):
        mine[r] = {'point': P[r]['point'], 'confidence': P[r]['confidence'], 'n_pass': P[r]['n_pass'], 'unresolved': P[r]['unresolved'],
                   'violations': P[r]['violations'], 'reserve_min_ratio': P[r]['reserve_min_ratio'],
                   'reserve_status': P[r]['reserve_status'], 'reserve_binding': P[r]['reserve_binding']}
        rc = rec_d1['confidence'][r]
        rr = rec_d1['reserve'][r]
        record[r] = {'point': rec_d1[r], 'confidence': rc['label'], 'n_pass': rc['n_pass'], 'unresolved': sorted(rc['unresolved']),
                     'violations': sorted(rc['violations']), 'reserve_min_ratio': rr['min_ratio_by_version'],
                     'reserve_status': rr['status'], 'reserve_binding': rr['binding']}
        ok = ok and mine[r]['point'] == record[r]['point'] and mine[r]['confidence'] == record[r]['confidence'] \
            and mine[r]['n_pass'] == record[r]['n_pass'] and mine[r]['unresolved'] == record[r]['unresolved'] \
            and mine[r]['violations'] == record[r]['violations'] and mine[r]['reserve_status'] == record[r]['reserve_status'] \
            and mine[r]['reserve_binding'] == record[r]['reserve_binding'] \
            and all(num_eq(mine[r]['reserve_min_ratio'][v], record[r]['reserve_min_ratio'][str(v)], 1e-6) for v in (0, 1))
    lab = prefix + ('TASK_FEASIBLE' if P['cand']['point'] == 'POINT_ELIGIBLE' else 'CAND_INFEASIBLE')
    ok = ok and lab == rec_d1['label'] and P['cand']['failed'] == rec_d1['cand_failed']
    add(f'S5_{tag}_D1', f'{tag} D1: G1 at CAND (28 conditions), START / CONT point eligibility, the confidence conjunction and the '
        'two-SE reserve, from the verifier\'s utility recomputation', i,
        {'label': lab, 'cand_failed': P['cand']['failed'], **mine}, {'label': rec_d1['label'], 'cand_failed': rec_d1['cand_failed'], **record},
        ok, 1e-6, note='reserve ratios compared at 1e-6 (they divide by SEs); floors from the inherited OHN1 RULES.json')
    return P


def fb_r1_decisions(ms):
    pfx = 'ADAPTIVE_R1__'
    dp, pp = RES / 'claims' / 'FB_R1' / 'DECISIONS.json', RES / 'claims' / 'PLAN_FB_R1.json'
    D, plan = load(dp), load(pp)
    d = D['decisions']
    U, PC = ms['utility'], ms['paired_costs']['FB_PFO']
    G = merged(ms, 'SAR1_FB_R1')
    NL = nl_merged(ms, 'SAR1_FB_R1', 'nolookup_pair_benefit')
    EN = nl_merged(ms, 'SAR1_FB_R1', 'enhanced_pair_benefit_from_record')
    i = ins(dp, pp, V.WORK / 'ms_recomputed.json')
    P = check_d1('FB_R1', U, d['D1'], 'FB_PFO', 'NEURAL_RAW', 'FB_PFO_TASK', ms['n_households'], i, pfx)

    # ---- UM
    binding = P['start']['binding_by_version']
    dv = {v: PC['NEURAL_RAW'][str(v)][cond_metric(binding[v])] for v in (0, 1)}
    pooled = V.pooled({v: dv[v] for v in (0, 1)})
    resolved = pooled['value'] + 2 * pooled['se'] < 0 and all(dv[v]['value'] < 0 for v in (0, 1))
    worse = []
    for m in ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|men_brier',
              'occupation|nll', 'occupation|women_brier'):
        q = V.pooled({v: PC['NEURAL_RAW'][str(v)][m] for v in (0, 1)})
        if q['value'] - 2 * q['se'] > 0 and all(PC['NEURAL_RAW'][str(v)][m]['value'] > 0 for v in (0, 1)):
            worse.append(m)
    pm_c, pm_s = P['cand']['point_margin'], P['start']['point_margin']
    not_below = all(pm_c[v][0] >= pm_s[v][0] for v in (0, 1))
    um_label = pfx + ('MARGIN_NOT_RESOLVED' if not resolved else 'NEEDS_PROFILE')
    rum = d['UM']
    add('S5_FB_R1_UM', 'FB_R1 UM: START\'s binding condition per version (smallest reserve ratio), the pooled paired difference on it, '
        'the resolved-worse conditions and the whole-release point margin', i,
        {'binding': binding, 'by_version': {v: dv[v]['value'] for v in (0, 1)}, 'pooled': pooled['value'], 'se': pooled['se'],
         'upper': pooled['value'] + 2 * pooled['se'], 'resolved': resolved, 'resolved_worse': worse,
         'point_margin_cand': {v: pm_c[v][0] for v in (0, 1)}, 'point_margin_start': {v: pm_s[v][0] for v in (0, 1)},
         'not_below_start': not_below, 'label': um_label},
        {'binding': rum['binding'], 'by_version': rum['binding_difference']['by_version'], 'pooled': rum['binding_difference']['pooled'],
         'se': rum['binding_difference']['se'], 'upper': rum['binding_difference']['upper'],
         'resolved': rum['binding_difference']['resolved_improvement'], 'resolved_worse': rum['whole_release']['resolved_worse'],
         'not_below_start': rum['whole_release']['point_margin']['not_below_start'], 'label': rum['label']},
        {str(v): binding[v] for v in (0, 1)} == rum['binding'] == plan['start']['binding']
        and all(num_eq(dv[v]['value'], rum['binding_difference']['by_version'][str(v)]) for v in (0, 1))
        and num_eq(pooled['value'], rum['binding_difference']['pooled']) and num_eq(pooled['se'], rum['binding_difference']['se'])
        and resolved == rum['binding_difference']['resolved_improvement'] and sorted(worse) == sorted(rum['whole_release']['resolved_worse'])
        and not_below == rum['whole_release']['point_margin']['not_below_start'] and um_label == rum['label'], TOL)

    # ---- D2 simple comparisons
    rec2 = d['D2']['comparisons']
    comps = {}
    for name, q in (('START', 'NEURAL_RAW'), ('CONT', 'FB_PFO_TASK')):
        c = comparison(G[q])
        comps[name] = c
        r = rec2[name]
        add(f'S3_FB_R1_D2_{name}', f'FB_R1 D2 ({"b" if name == "START" else "b prime"}) {name} = {q}: aggregate pair benefit (block means over '
            'version x repetition; se_agg) and the recipient screens, from the verifier\'s recomputed paired cells', i,
            {'benefit': c['benefit'], 'mean': c['benefit_mean'], 'se_agg': c['benefit_se'], 'recipient': c['recipient'], 'pass': c['pass']},
            {'benefit': r['benefit'], 'mean': r['benefit_mean'], 'se_agg': r['benefit_se'], 'recipient': r['recipient'], 'pass': r['pass']},
            c['benefit'] == r['benefit'] and num_eq(c['benefit_mean'], r['benefit_mean'], 1e-12) and num_eq(c['benefit_se'], r['benefit_se'], 1e-12)
            and c['recipient'] == r['recipient'] and c['pass'] == r['pass'], 1e-12)

    # ---- D2 frontiers
    def frontier(key, path):
        prim = {q: V.pooled({v: PC[q][str(v)]['total|nll'] for v in (0, 1)}) for q in path}
        above = sorted((q for q in path if prim[q]['value'] <= 0), key=lambda q: -prim[q]['value'])
        res = {'primary': {q: prim[q]['value'] for q in path}, 'primary_se': {q: prim[q]['se'] for q in path}}
        dominated = [q for q in path if prim[q]['value'] > 0 and _resolved_negative(G[q])]
        if not above:
            res.update({'label': 'NOT_SEPARATED', 'reason': 'NO_BRACKET', 'point': None, 'pass': False, 'dominated': dominated})
            return res
        qs = above[0]
        c = comparison(G[qs])
        sec = {}
        for m in V.SECONDARY:
            x = V.pooled({v: PC[qs][str(v)][m] for v in (0, 1)})
            sec[m] = x['value'] - 2 * x['se'] > 0
        more = sorted(m for m, x in sec.items() if x)
        fl = ('NOT_SEPARATED_FROM_TASK_SPENDING' if c['benefit'] != 'AGG_POSITIVE' else
              'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER')
        below = [q for q in path if prim[q]['value'] > 0]
        interp = None
        if below:
            lo = min(below, key=lambda q: prim[q]['value'])
            dlo, dhi = prim[lo]['value'], prim[qs]['value']
            lam = dlo / (dlo - dhi) if dlo != dhi else 0.0
            cl = comparison(G[lo])
            interp = {'lower': lo, 'lambda': lam, 'benefit': (1 - lam) * cl['benefit_mean'] + lam * c['benefit_mean'],
                      'se': (1 - lam) * cl['benefit_se'] + lam * c['benefit_se']}
        ok = fl == 'DOMINATES_CONTROL_FRONTIER' and c['gate'] and not dominated
        res.update({'label': fl, 'reason': None if fl != 'NOT_SEPARATED_FROM_TASK_SPENDING' else 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL',
                    'point': qs, 'benefit': c['benefit'], 'benefit_mean': c['benefit_mean'], 'benefit_se': c['benefit_se'],
                    'recipient_vs_bracket': c['recipient'], 'gate': c['gate'], 'secondary_more_spending': more,
                    'interpolation': interp, 'dominated': dominated, 'pass': ok})
        return res

    for key, name in (('CTRL', 'CTRL_FRONTIER'), ('ALT', 'ALT_FRONTIER')):
        path = ['NEURAL_RAW'] + plan['paths'][key]
        fr = frontier(key, path)
        comps[name] = fr
        r = rec2[name]
        rp = r['frontier'].get('primary_cost_differences', {})
        ok = (fr['label'] == r['label'] and fr['point'] == r['point'] and fr['pass'] == r['pass'] and r['path'] == path
              and all(num_eq(fr['primary'][q], rp[q]['value'], 1e-9) and num_eq(fr['primary_se'][q], rp[q]['se'], 1e-9) for q in path)
              and fr['dominated'] == r['dominated_by_cheaper'])
        rec_view = {'label': r['label'], 'point': r['point'], 'pass': r['pass'], 'primary': {q: rp[q]['value'] for q in rp},
                    'reason': r['reason'], 'dominated': r['dominated_by_cheaper']}
        if fr['point'] is not None:
            rf = r['frontier']
            ok = ok and num_eq(fr['benefit_mean'], rf['benefit_mean'], 1e-12) and num_eq(fr['benefit_se'], rf['benefit_se'], 1e-12) \
                and fr['benefit'] == rf['benefit_vs_bracket']['label'] and fr['secondary_more_spending'] == rf['secondary_more_spending'] \
                and fr['recipient_vs_bracket'] == r['recipient_vs_bracket'] and fr['gate'] == r['recipient_gate_vs_bracket'] \
                and num_eq(fr['interpolation']['lambda'], rf['interpolation']['lambda'], 1e-9) \
                and num_eq(fr['interpolation']['benefit'], rf['interpolation']['benefit'], 1e-9)
            rec_view.update({'benefit': rf['benefit_vs_bracket']['label'], 'benefit_mean': rf['benefit_mean'], 'benefit_se': rf['benefit_se'],
                             'secondary_more_spending': rf['secondary_more_spending'], 'recipient_vs_bracket': r['recipient_vs_bracket'],
                             'interpolation': rf['interpolation']})
        else:
            ok = ok and r['reason'] == 'NO_BRACKET'
        add(f'S3_FB_R1_D2_{name}', f'FB_R1 D2 (c) {name}: Q* by the frontier rule on MODEL_SELECT pooled paired total-NLL costs '
            '(verifier recomputation), the benefit vs Q*, the secondary metrics, (c2) the recipient gate, (c3) cheaper points', i,
            fr, rec_view, ok, 1e-9)

    # ---- (d) targeting with the non-table winner, enhanced flags, the label
    tg = {}
    for q in [comps['START'] and 'NEURAL_RAW', 'FB_PFO_TASK'] + [comps[n]['point'] for n in ('CTRL_FRONTIER', 'ALT_FRONTIER') if comps[n]['point']]:
        vm = version_means(NL[q])
        tg[q] = {'version_means': vm, 'pass': all(vm[v] > 0 for v in (0, 1))}
    en = {q: {'version_means': version_means(EN[q]), 'pass': all(x > 0 for x in version_means(EN[q]).values())} for q in tg}
    passed = [n for n in ('START', 'CONT', 'CTRL_FRONTIER', 'ALT_FRONTIER') if comps[n]['pass']]
    gating = ['START', 'CONT', 'CTRL_FRONTIER', 'ALT_FRONTIER']
    targeted = not all(x['pass'] for x in tg.values())
    if len(passed) == len(gating):
        lab = 'READER_TARGETED' if targeted else 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'
    elif not passed:
        lab = 'NOT_SEPARATED'
    else:
        lab = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    rt = d['D2']['targeting']
    flags_mine = sorted({'NO_BRACKET_CTRL' if comps['CTRL_FRONTIER']['reason'] == 'NO_BRACKET' else '',
                         'NO_BRACKET_ALT' if comps['ALT_FRONTIER']['reason'] == 'NO_BRACKET' else ''} - {''}
                        | {f'READER_SENSITIVE_ENHANCED_{q}' for q, x in en.items() if not x['pass']}
                        | {f'READER_TARGETED_{n}' for n in passed if not tg[{'START': 'NEURAL_RAW', 'CONT': 'FB_PFO_TASK'}[n] if n in ('START', 'CONT') else comps[n]['point']]['pass']})
    rflags = sorted(f for f in d['D2']['flags'] if not f.startswith('CONTROL_POINT_RULE_NOT_FOLLOWED'))
    add('S5_FB_R1_D2', 'FB_R1 D2: (d) reader targeting with the non-table winner (version means of the per-cell point benefits), the '
        'enhanced-winner flags, and the D2 label from the recomputed comparisons', i,
        {'label': pfx + lab, 'passed': passed, 'targeting': tg, 'enhanced': en, 'flags_excluding_fineness': flags_mine},
        {'label': d['D2']['label'], 'passed': d['D2']['passed'], 'targeting': rt, 'enhanced': d['D2']['enhanced_reported_only'],
         'flags_excluding_fineness': rflags},
        pfx + lab == d['D2']['label'] and passed == d['D2']['passed'] and flags_mine == rflags
        and all(tg[q]['pass'] == rt[q]['pass'] and all(num_eq(tg[q]['version_means'][v], rt[q]['version_means'][str(v)], 1e-12) for v in (0, 1)) for q in tg)
        and all(en[q]['pass'] == d['D2']['enhanced_reported_only'][q]['pass'] for q in en), 1e-12)

    # ---- D3 and D4
    refs = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
    elig = {r: d1_parts(U, r, ms['n_households'])['point'] for r in refs}
    per, passed3 = {}, []
    for r in refs:
        c = comparison(G[r])
        per[r] = {'benefit': c['benefit'], 'mean': c['benefit_mean'], 'se_agg': c['benefit_se'], 'recipient_gate_pass': c['gate']}
        if c['pass']:
            passed3.append(r)
    rd3 = d['D3']['per_reference']
    ok3 = all(per[r]['benefit'] == rd3[r]['benefit'] and per[r]['recipient_gate_pass'] == rd3[r]['recipient_gate_pass']
              and num_eq(per[r]['mean'], rd3[r]['mean'], 1e-12) and num_eq(per[r]['se_agg'], rd3[r]['se_agg'], 1e-12) for r in refs)
    lab3 = pfx + ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if all(r in passed3 for r in refs if elig[r] == 'POINT_ELIGIBLE') else 'NOT_SUPERIOR')
    add('S5_FB_R1_D3', 'FB_R1 D3: per strong reference, resolved pair benefit with the recipient gates (verifier aggregates); the '
        'references\' point eligibility from the verifier\'s utility recomputation', i,
        {'label': lab3, 'passed': passed3, 'per_reference': per, 'reference_point_eligibility': elig},
        {'label': d['D3']['label'], 'passed': d['D3']['passed'], 'per_reference': rd3, 'reader_budget': D['reader_budget']},
        ok3 and lab3 == d['D3']['label'] and passed3 == d['D3']['passed'] and all(e == 'POINT_ELIGIBLE' for e in elig.values()), 1e-12,
        note='reader-budget flags taken from the record (all READER_BUDGET_COMPARABLE); not recomputed')
    t4 = {}
    for r in ('NEURAL_RAW', 'H_LJ'):
        ab = V.aggregate(G[r]['pair']['cells'], G[r]['pair']['cell_se'])
        c = comparison(G[r])
        t4[r] = {'label': V.benefit_label(ab, threshold=V.TARGET), 'gate': c['gate']}
    labs4 = [t4[r]['label'] for r in t4]
    if all(x == 'TARGET_AGG_RESOLVED' for x in labs4):
        lev = 'MATERIAL_0020_RESOLVED' if all(t4[r]['gate'] for r in t4) else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in labs4):
        lev = 'MATERIAL_0020_POINT_ONLY'
    elif all(comparison(G[r])['benefit'] == 'AGG_POSITIVE' for r in t4):
        lev = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        lev = 'NOT_MET'
    lab4 = pfx + lev
    add('S5_FB_R1_D4', 'FB_R1 D4: the 0.020 target screen against NEURAL_RAW and H_LJ (verifier aggregates)', i,
        {'label': lab4, 'per_reference': t4}, {'label': d['D4']['label'], 'per_reference': d['D4']['per_reference']},
        lab4 == d['D4']['label'] and all(t4[r]['label'] == d['D4']['per_reference'][r]['label'] for r in t4), 0)

    # ---- outcome class and the remaining labels
    d1_ok = P['cand']['point'] == 'POINT_ELIGIBLE'
    d2_met = lab == 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'
    d3_met = lab3.endswith('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES')
    cls = ('TASK_INFEASIBLE' if not d1_ok else 'METHOD_CANDIDATE' if d2_met and d3_met else 'COMPONENT_GAIN_NOT_COMPETITIVE' if d2_met
           else 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' if d3_met else 'USEFULNESS_MARGIN_ONLY' if resolved else 'FEASIBLE_NO_RELIABLE_GAIN')
    reg = load(V.PRIV / 'access' / 'LABEL_ACCESS.jsonl') if False else None
    add('S5_FB_R1_class', 'FB_R1 outcome class and the D5 / D6 / D2R labels; adaptive tag (card committed after the first MODEL_SELECT '
        'read of the campaign)', i,
        {'class': pfx + cls, 'D5': pfx + 'NOT_APPLICABLE_NOT_JOINT', 'D6': pfx + 'NOT_CLAIMED', 'D2R': 'NOT_RUN', 'registered': False},
        {'class': d['outcome_class']['label'], 'D5': d['D5']['label'], 'D6': d['D6']['label'], 'D2R': d['D2R']['label'],
         'registered': d['registered']},
        pfx + cls == d['outcome_class']['label'] and d['D5']['label'] == pfx + 'NOT_APPLICABLE_NOT_JOINT'
        and d['D6']['label'] == pfx + 'NOT_CLAIMED' and d['D2R']['label'] == 'NOT_RUN' and d['registered'] is False, 0,
        note='D2R reads NOT_RUN in this record (the robustness bank SAR1_FB_R1_ROB was still running at the verifier\'s launch)')
    # grid seeds and bootstrap seeds of the record
    seeds_ok = D['seeds'] == {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)} and \
        D['boot_seeds'] == {f'S{v}{r}': V.BOOT_SEED + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2')}
    add('S3_FB_R1_seeds', 'FB_R1: the recorded reader seeds and bootstrap seeds equal MODEL_SEED_BASE + 1000 + 10 v + k and '
        '2026102400 + 100 v + repetition index', i, {'reader': {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)}},
        {'reader': D['seeds'], 'boot': D['boot_seeds']}, seeds_ok, 0)
    # the plan's hash in the decision record (PROTOCOL 6: "its hash is recorded in the reading's decision record")
    psha = V.fsha(pp)
    add('S6_FB_R1_plan_hash_recorded', 'FB_R1: the plan file\'s SHA-256 is recorded in the decision record (PROTOCOL 6)', i,
        {'plan_sha256': psha}, {'found_in_record': psha in json.dumps(D)}, psha in json.dumps(D), 0)
    return {'D2_label': lab, 'class': cls}


def fa_r1_decisions(ms):
    pfx = 'ADAPTIVE_R1__'
    dp, pp = RES / 'claims' / 'FA_R1' / 'DECISIONS.json', RES / 'claims' / 'PLAN_FA_R1.json'
    D = load(dp)
    d = D['decisions']
    i = ins(dp, pp, V.WORK / 'ms_recomputed.json')
    U = ms['utility']
    P = check_d1('FA_R1', U, d['D1'], 'FA_TFRO_R1', 'SUCC', 'FA_TFRO_R1_CONT', ms['n_households'], i, pfx)
    g = U['FA_TFRO_R1']['1']['guard|sex0']
    add('S1_FA_R1_v1_men_guard', 'FA_R1: version-1 men\'s occupation guard of FA_TFRO_R1 on MODEL_SELECT against 0.005 (verifier metric '
        'code)', i, {'value': g['value'], 'se': g['se'], 'fails': g['value'] > 0.005},
        {'value': d['D1']['confidence_detail_cand']['per_version']['1']['guard|sex0']['value'], 'brief': '+0.0099 against 0.005'},
        g['value'] > 0.005 and num_eq(g['value'], d['D1']['confidence_detail_cand']['per_version']['1']['guard|sex0']['value']), TOL)
    v0 = [c for c in V.CONDS if U['FA_TFRO_R1']['0'][c]['value'] > U['FA_TFRO_R1']['0'][c]['cap']]
    add('S1_FA_R1_v0_point_feasible', 'FA_R1: version 0 of FA_TFRO_R1 meets all 14 conditions at the point (descriptive)', i,
        {'failed_v0': v0}, 'version 0 point-feasible', not v0, 0)
    exp = {k: pfx + 'NOT_ASSESSED_TASK_INFEASIBLE' for k in ('UM', 'D2', 'D3', 'D4', 'D5')}
    got = {k: d[k]['label'] for k in exp}
    cls = pfx + ('TASK_INFEASIBLE' if P['cand']['point'] != 'POINT_ELIGIBLE' else 'CHECK')
    add('S5_FA_R1_labels', 'FA_R1: UM / D2-D5 NOT_ASSESSED_TASK_INFEASIBLE; class TASK_INFEASIBLE (no other fitted nonzero strength); '
        'flag CAND_STRENGTH_DIFFERS_BY_VERSION (rho* 1.0 / 0.6)', i, {**exp, 'class': cls},
        {**got, 'class': d['outcome_class']['label'], 'flags': d['outcome_class']['flags'],
         'fitted_strengths': d['D1']['fitted_strengths']},
        got == exp and cls == d['outcome_class']['label'] and 'CAND_STRENGTH_DIFFERS_BY_VERSION' in d['outcome_class']['flags']
        and list(d['D1']['fitted_strengths']) == ['FA_TFRO_R1'], 0)
    psha = V.fsha(pp)
    add('S6_FA_R1_plan_hash_recorded', 'FA_R1: the plan file\'s SHA-256 is recorded in the decision record (PROTOCOL 6)', i,
        {'plan_sha256': psha}, {'found_in_record': psha in json.dumps(D)}, psha in json.dumps(D), 0)


def fb_income_alias(ms):
    x = ms['fb_income_vs_neural_raw']
    add('S1_FB_R1_income_bitwise_START', 'FB_R1 card item 1: CAND\'s income vector equals START\'s released income vector bitwise '
        '(START = the frozen NEURAL_RAW reference read by the grid and the evaluator) on MODEL_SELECT', ins(V.WORK / 'ms_recomputed.json'),
        x, 'bitwise equal', all(x[v]['bitwise_equal'] for v in ('0', '1')), 0,
        note='CAND\'s income is NEURAL_RAW\'s E2 income network under the family\'s float64 forward law; the frozen reference release '
             'differs from it by at most the recorded max_abs_diff with identical decisions; the income paired utility differences are '
             '< 1e-9 and every income-view grid cell vs NEURAL_RAW is exactly 0 (ALIAS_ZERO), so no label depends on it')


# ====================================================================== check 6: access
def access():
    la = V.PRIV / 'access' / 'LABEL_ACCESS.jsonl'
    rc = V.PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json'
    lg = V.PRIV / 'compute' / 'ledger.jsonl'
    rl = Path('<repository>/.git/worktrees/pcrl-strong-anchor-release-v1/logs/HEAD')
    ev = [json.loads(x) for x in la.read_text().splitlines() if x.strip()]
    led = [json.loads(x) for x in lg.read_text().splitlines() if x.strip()]
    receipt = load(rc)
    i = {**ins(la, rc, lg), str(rl): V.fsha(rl)}
    t = lambda s: dt.datetime.fromisoformat(s)
    reflog = {}
    for line in rl.read_text().splitlines():
        head, _, msg = line.partition('\t')
        parts = head.split()
        reflog[parts[1]] = {'utc': dt.datetime.fromtimestamp(int(parts[-2]), dt.timezone.utc), 'msg': msg[:120]}
    first = min(t(e['utc']) for e in ev)
    rv = t(receipt['verified_utc'])
    reg_commit = reflog.get(receipt['research_commit'] if len(receipt['research_commit']) == 40 else
                            next((k for k in reflog if k.startswith(receipt['research_commit'])), ''), {})
    add('S6_receipt_before_first_read', 'no label read before the verified registration receipt (receipt verified_utc and the '
        'registration commit time from the worktree reflog, read as a file)', i,
        {'first_read_utc': first.isoformat(), 'receipt_verified_utc': rv.isoformat(),
         'registration_commit_utc': reg_commit.get('utc').isoformat() if reg_commit else None},
        {'remote_sha_verified': receipt['remote_sha_verified'], 'receipt_commit': receipt['receipt_commit']},
        first > rv and receipt['remote_sha_verified'] and bool(reg_commit) and reg_commit['utc'] < rv, 0)
    bad_card, bad_ms, bad_closed, bad_unit, bad_cal, n_fit = [], [], [], [], [], 0
    starts = {e['unit_id']: t(e['utc']) for e in led if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: t(e['utc']) for e in led if e.get('event') == 'unit_end'}
    for e in ev:
        ut = t(e['utc'])
        if e['operation'] in ('fit', 'calibrate'):
            n_fit += 1
            cp = V.ROOT / e['card']
            rcm = reflog.get(e.get('card_commit'))
            if not (rcm and rcm['utc'] <= ut and t(e['card_commit_utc']) <= ut and V.fsha(cp) == e['card_sha256']):
                bad_card.append(e['utc'] + ' ' + e['unit_id'])
        if e['role'] == 'MODEL_SELECT':
            chain = [e['entry_point'].split(':')[0]] + list(e.get('module_chain') or [])
            if any(m.split('.')[0] == 'sar' and len(m.split('.')) > 1 and m.split('.')[1] in ('fa', 'fb', 'ctl', 'fs') for m in chain):
                bad_ms.append(e['utc'] + ' ' + e['unit_id'])
        if e['role'] in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT'):
            bad_closed.append(e['utc'] + ' ' + e['role'])
        if e['role'] == 'CALIBRATION' and 'sex' in e['fields']:
            bad_cal.append(e['utc'])
        u = e.get('unit_id')
        if not u or u not in starts or starts[u] > ut or (u in ends and ends[u] < ut):
            bad_unit.append(e['utc'] + ' ' + str(u))
    add('S6_fitting_reads_after_card_commit', 'every fitting read (fit / calibrate) after its card\'s commit (commit hash found in the '
        'worktree reflog with an earlier time) and the card unchanged since (current SHA-256 = logged SHA-256)', i,
        {'fitting_reads': n_fit, 'violations': bad_card}, 'no violation', not bad_card, 0)
    add('S6_no_model_select_in_fitting_module', 'no MODEL_SELECT label read with a fitting module (sar.fa / sar.fb / sar.ctl / sar.fs) '
        'on the logged entry point or module chain', i, {'violations': bad_ms, 'model_select_reads': sum(e['role'] == 'MODEL_SELECT' for e in ev)},
        'no violation', not bad_ms, 0)
    add('S6_no_closed_role', 'no ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT / sealed label read; no SEX read on CALIBRATION', i,
        {'closed_reads': bad_closed, 'calibration_sex_reads': bad_cal, 'roles_seen': sorted({e['role'] for e in ev})}, 'no violation',
        not bad_closed and not bad_cal, 0)
    add('S6_reads_inside_queue_units', 'every label read carries a queue unit id whose unit_start precedes it and whose unit_end (if any) '
        'follows it in the queue ledger', i, {'violations': bad_unit, 'reads': len(ev)}, 'no violation', not bad_unit, 0)
    # plan commits before each reading's first MODEL_SELECT read (reflog commit messages name the plan / bank commits)
    for plan, commit_prefix, units in (('PLAN_FB_R1', '21b25d188', ('SAR_EVAL_FB_R1', 'SAR_GRID_SAR1_FB_R1')),
                                       ('PLAN_FA_R1', '107f7f23f', ('SAR_EVAL_FA_R1', 'SAR_GRID_SAR1_FA_R1'))):
        c = next((v for k, v in reflog.items() if k.startswith(commit_prefix)), None)
        reads = [t(e['utc']) for e in ev if e['role'] == 'MODEL_SELECT' and any(str(e.get('unit_id', '')).startswith(u) for u in units)]
        pw = t(load(RES / 'claims' / f'{plan}.json')['written_utc'])
        add(f'S6_{plan}_before_first_MS', f'{plan} written and committed before the reading\'s first MODEL_SELECT utility / reader read', i,
            {'plan_written_utc': pw.isoformat(), 'commit_utc': c['utc'].isoformat() if c else None, 'commit_msg': c['msg'] if c else None,
             'first_reading_ms_read_utc': min(reads).isoformat() if reads else None},
            'plan before the first read', bool(c and reads and pw <= c['utc'] < min(reads)), 0,
            note='the commit is identified by the hash prefix the coordinator\'s commit log gives for the plan / bank commit; the '
                 'verifier did not run git (forbidden) and did not open the commit tree')
    # registration status of the readings (first MODEL_SELECT read of the campaign vs each card's commit)
    first_ms = min(t(e['utc']) for e in ev if e['role'] == 'MODEL_SELECT')
    cards = {}
    for e in ev:
        if e['operation'] == 'fit':
            cards.setdefault(e['card'], t(e['card_commit_utc']))
    exp = {'results/pcrl_strong_anchor_release_v1/cards/FA_TFR_v1.md': True, 'results/pcrl_strong_anchor_release_v1/cards/FB_PFD_v1.md': True,
           'results/pcrl_strong_anchor_release_v1/cards/FB_PFO_R1.md': False, 'results/pcrl_strong_anchor_release_v1/cards/FA_TFRO_R1.md': False}
    mine = {c: w < first_ms for c, w in cards.items() if c in exp}
    later = {c: w.isoformat() for c, w in cards.items() if c not in exp}
    add('S6_registration_status', 'registered iff revision 0 and the card was committed before the first MODEL_SELECT read of the '
        'campaign; FA_REG / FB_REG registered, FB_R1 / FA_R1 ADAPTIVE_R1', i,
        {'first_model_select_read': first_ms.isoformat(), 'card_before_first_ms': mine}, exp, mine == exp, 0,
        note=f'cards of later readings seen in the access log (outside this wave): {later}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ms = load(V.WORK / 'ms_recomputed.json')
    fa_reg()
    fb_reg()
    reg_decisions()
    fb_r1_selection()
    fa_r1_selection(ms)
    fb_r1_decisions(ms)
    fa_r1_decisions(ms)
    fb_income_alias(ms)
    access()
    out = {'checks': CHECKS, 'n': len(CHECKS), 'fail': [c['id'] for c in CHECKS if c['status'] != 'PASS'],
           'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    V.jdump(Path(a.out), out)
    print(json.dumps({'n': out['n'], 'fail': out['fail']}, indent=1))


def _resolved_negative(entry):
    ab = V.aggregate(entry['pair']['cells'], entry['pair']['cell_se'])
    return ab['upper'] < 0 and all(ab['per_version'][v]['mean'] < 0 for v in (0, 1))


if __name__ == '__main__':
    main()

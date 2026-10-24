"""Verifier wave 2 (label-free): FA_R2 selections, brackets and decisions; FS_CPI selection and absence of any MODEL_SELECT
computation; FB_R1 D2R (and FA_R2 D2R when its records exist) from the verifier's recomputed robustness cells; plan-hash sidecars and
the file-level tracking check of claims/FA_REG and claims/FB_REG; the access log re-checked over all reads. Reuses the wave-1 helpers
of v_decide (the verifier's own code). No label read; nothing from sar.rules / sar.claims / nrm.rules / sar.devreaders.

    python3 .../roles/verifier/code/v2_decide.py --out .../work/decide2.json [--fa-r2-rob work/ms2_fa_r2_rob.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
import v_decide as D  # noqa: E402   (wave-1 helpers; its CHECKS list collects the wave-2 entries here)

RES, RUNS = V.RES, V.RUNS
add, ins, load, num_eq = D.add, D.ins, D.load, D.num_eq


# ====================================================================== FA_R2 selections (check 2)
def fa_r2_selection(ms1):
    plan = load(RES / 'claims' / 'PLAN_FA_R2.json')
    for v in (0, 1):
        p = RES / 'fa' / f'FIT_R2_V{v}.json'
        f = load(p)
        s = f['selection']
        i = ins(p, RES / 'cards' / 'FA_TFRB_R2.md', RES / 'claims' / 'PLAN_FA_R2.json', V.WORK / 'ms_recomputed.json')
        u = ms1['utility']['SUCC'][str(v)]
        mine = {'occ_nll': 0.010 - max(u['occupation|U0|nll_excess']['value'], u['occupation|Ucal|nll_excess']['value']),
                'occ_brier': 0.005 - max(u['occupation|U0|brier_excess']['value'], u['occupation|Ucal|brier_excess']['value']),
                'occ_acc': 0.010 - max(u['occupation|U0|accuracy_loss']['value'], u['occupation|Ucal|accuracy_loss']['value']),
                'men': 0.005 - u['guard|sex0']['value'], 'women': 0.005 - u['guard|sex1']['value']}
        add(f'W2_S2_FA_R2_v{v}_margins', f'FA_R2 v{v}: budgets = SUCC\'s cap margins recomputed from SUCC\'s MODEL_SELECT utility', i,
            mine, s['margins'], all(num_eq(mine[k], s['margins'][k], 1e-9) for k in mine), 1e-9)
        M = s['margins']
        rho_star, table = None, {}
        for rho in s['rho_grid']:
            row = s['per_rho'].get(str(float(rho)))
            if row is None:
                break
            cal, ge = row['cal_spend_over_SUCC'], row['guard_estimate']
            a = {c: bool(cal[c]['estimate'] + 2 * cal[c]['se'] <= M[c]) for c in ('occ_nll', 'occ_brier', 'occ_acc')}
            inc0 = all(cal[c]['estimate'] == 0.0 and cal[c]['se'] == 0.0 for c in ('inc_acc', 'inc_brier', 'inc_nll'))
            L, sL, pm = cal['occ_brier']['estimate'], cal['occ_brier']['se'], ge['p_men']
            Cc = ge['mf_men'] - ge['mf_women']
            cf = ge['C_per_fold']
            mean_cf = sum(cf) / len(cf)
            se_between = math.sqrt(sum((x - mean_cf) ** 2 for x in cf) / (len(cf) - 1)) / math.sqrt(5)
            se_C = max(ge['se_C_household'], se_between)
            dm, sm = L + (1 - pm) * Cc, math.sqrt(sL ** 2 + (1 - pm) ** 2 * se_C ** 2)
            dw, sw = L - pm * Cc, math.sqrt(sL ** 2 + pm ** 2 * se_C ** 2)
            b = {'men': bool(dm + 2 * sm <= M['men']), 'women': bool(dw + 2 * sw <= M['women'])}
            passes = all(a.values()) and all(b.values()) and inc0
            cons = (num_eq(ge['L_cal'], L, 0) and num_eq(ge['se_L'], sL, 0) and num_eq(ge['C_mf'], Cc, 1e-15)
                    and num_eq(ge['se_C_between'], se_between, 1e-15) and num_eq(ge['se_C'], se_C, 1e-15)
                    and num_eq(ge['men'][0], dm, 1e-15) and num_eq(ge['men'][1], sm, 1e-15)
                    and num_eq(ge['women'][0], dw, 1e-15) and num_eq(ge['women'][1], sw, 1e-15))
            table[str(rho)] = {'part_a': a, 'part_b': b, 'income_zero': inc0, 'se_C_between': se_between, 'se_C': se_C,
                               'delta_men': [dm, sm], 'delta_women': [dw, sw], 'bound_men': dm + 2 * sm, 'bound_women': dw + 2 * sw}
            add(f'W2_S2_FA_R2_v{v}_rho{rho}', f'FA_R2 v{v} rho {rho}: part (a); SE_C = max(household SE, SD(five per-fold contrasts, '
                'ddof 1)/sqrt 5) re-derived; Delta = L +/- weight x C with SE^2 = SE_L^2 + weight^2 SE_C^2; pass flags', i,
                table[str(rho)], {'part_a': row['part_a'], 'part_b': row['part_b'], 'passes': row['passes'], 'se_C_between': ge['se_C_between'],
                                  'se_C': ge['se_C'], 'men': ge['men'], 'women': ge['women']},
                cons and a == row['part_a'] and b == row['part_b'] and passes == row['passes'], 1e-15)
            if passes:
                rho_star = rho
            else:
                break
        exp = {0: 0.6, 1: 1.0}[v]
        add(f'W2_S2_FA_R2_v{v}_rho_star', f'FA_R2 v{v}: rho* re-selected (largest rho with every smaller rho passing; scan stops at '
            'the first failure)', i, {'rho_star': rho_star, 'stopped_at': next((r for r, t in table.items() if not (all(t['part_a'].values()) and all(t['part_b'].values()))), None)},
            {'rho_star': s['rho_star'], 'cand': f['cand']['rho_star'], 'plan': plan['strengths'][0]['strength'][str(v)]},
            rho_star == s['rho_star'] == f['cand']['rho_star'] == plan['strengths'][0]['strength'][str(v)] == exp, 0)
        # CTRL bracket on R1's lambda = 0 path against this CAND (CALIBRATION total NLL; ties to the smaller s)
        c = f['ctrl']
        sc = f['cand']['cal_spend_over_SUCC']['spend_over_start']
        order = sorted(c['path'], key=float)
        rows = c['path']
        below = [g for g in order if rows[g]['spend_over_start'] <= sc]
        lo = max(below, key=lambda g: rows[g]['spend_over_start']) if below else None
        above = [g for g in order if rows[g]['spend_over_start'] >= sc + 2 * rows[g]['se_diff_vs_cand']]
        hi = min(above, key=lambda g: rows[g]['spend_over_start']) if above else None
        flags = ([] if above else ['CONTROL_PATH_MAY_NOT_BRACKET']) + ([] if below else ['CONTROL_PATH_NO_LO_POINT'])
        r1 = {float(k): x for k, x in c['r1_points'].items()}

        def cfg_of(g, side):
            if g is None:
                return None
            x = float(g)
            return 'START' if x == 0.0 else r1.get(x, f'FA_TFRB_R2_TASK_{side}')
        roles = plan['roles']
        role_exp = {'FA_R2_CTRL_LO': 'FA/' + cfg_of(lo, 'LO') if cfg_of(lo, 'LO') not in ('START', None) else ('REF:SUCC' if cfg_of(lo, 'LO') == 'START' else None),
                    'FA_R2_CTRL_HI': 'FA/' + cfg_of(hi, 'HI') if cfg_of(hi, 'HI') not in ('START', None) else ('REF:SUCC' if cfg_of(hi, 'HI') == 'START' else None)}
        saved = c['saved']
        rec_cfg = {'LO': saved['LO'].get('config') or saved['LO'].get('alias'), 'HI': saved['HI'].get('config') or saved['HI'].get('alias')}
        ok = ((lo is not None and float(lo) == c['lo']) and (hi is not None and float(hi) == c['hi']) and sorted(flags) == sorted(c['flags'])
              and rec_cfg == {'LO': cfg_of(lo, 'LO'), 'HI': cfg_of(hi, 'HI')}
              and roles['FA_R2_CTRL_LO'][str(v)] == role_exp['FA_R2_CTRL_LO'] and roles['FA_R2_CTRL_HI'][str(v)] == role_exp['FA_R2_CTRL_HI']
              and all(c['reuse_check'].values()) and c['selection_data'] == 'CALIBRATION')
        add(f'W2_S2_FA_R2_v{v}_CTRL_bracket', f'FA_R2 v{v} CTRL: LO / HI re-selected on R1\'s lambda = 0 path against this CAND (spend values, '
            'ties to the smaller s), reuse aliases (s = 0 START; R1 points; s = 1 CONT) and the plan roles', i,
            {'cand_spend': sc, 'lo': float(lo) if lo is not None else None, 'hi': float(hi) if hi is not None else None, 'flags': flags,
             'configs': {'LO': cfg_of(lo, 'LO'), 'HI': cfg_of(hi, 'HI')}, 'roles': role_exp},
            {'lo': c['lo'], 'hi': c['hi'], 'flags': c['flags'], 'configs': rec_cfg,
             'roles': {k: roles[k][str(v)] for k in roles}, 'reuse_check': c['reuse_check']}, ok, 0)
        # fineness (RULES 4.4): CAND's spend over START <= 0 on the declared data -> the origin is the bracket
        add(f'W2_S2_FA_R2_v{v}_fineness', f'FA_R2 v{v}: fineness label on the declared CALIBRATION data', i,
            {'label': 'NOT_APPLICABLE_CAND_NOT_ABOVE_START' if sc <= 0 else 'APPLICABLE', 'cand_spend': sc},
            {'plan_cand': plan['fineness_inputs']['CTRL'][str(v)]['cand'], 'decision_flags': 'no CONTROL_GRID / POINT_RULE flag in D2'},
            sc <= 0 and num_eq(plan['fineness_inputs']['CTRL'][str(v)]['cand'], sc, 0), 0,
            note='reported, never gating; the bracket is still taken on spend values (cards FA_TFR_v1 / FA_TFRB_R2 section 7)')


# ====================================================================== FA_R2 decisions (checks 1, 3, 5)
def fa_r2_decisions(ms2):
    pfx = 'ADAPTIVE_R2__'
    dp, pp = RES / 'claims' / 'FA_R2' / 'DECISIONS.json', RES / 'claims' / 'PLAN_FA_R2.json'
    DD, plan = load(dp), load(pp)
    d = DD['decisions']
    U, PC = ms2['utility'], ms2['paired_costs']['FA_TFRB_R2']
    G = D.merged(ms2, 'SAR1_FA_R2')
    NL = D.nl_merged(ms2, 'SAR1_FA_R2', 'nolookup_pair_benefit')
    EN = D.nl_merged(ms2, 'SAR1_FA_R2', 'enhanced_pair_benefit_from_record')
    i = ins(dp, pp, V.WORK / 'ms2_fa_r2_fb_rob.json')
    P = D.check_d1('W2_FA_R2', U, d['D1'], 'FA_TFRB_R2', 'SUCC', 'FA_TFRO_R1_CONT', ms2['n_households'], i, pfx)
    add('W2_S1_FA_R2_ctrl_hi_v1_is_succ', 'FA_R2: the v1 CTRL_HI role is the frozen SUCC release (origin as the HI point)', i,
        ms2['ctrl_hi_v1_is_succ'], True, ms2['ctrl_hi_v1_is_succ'] is True, 0)
    add('W2_S1_FA_R2_income_bitwise_SUCC', 'FA_R2 card section 1: CAND\'s income equals SUCC\'s H3 income bitwise on MODEL_SELECT', i,
        ms2['fa_r2_income_vs_succ'], 'bitwise', all(x['bitwise_equal'] for x in ms2['fa_r2_income_vs_succ'].values()), 0)
    # ---- UM
    binding = P['start']['binding_by_version']
    dv = {v: PC['SUCC'][str(v)][D.cond_metric(binding[v])] for v in (0, 1)}
    pooled = V.pooled(dv)
    resolved = pooled['value'] + 2 * pooled['se'] < 0 and all(dv[v]['value'] < 0 for v in (0, 1))
    cs = D.comparison(G['SUCC'])
    pair_cs = {v: -cs['agg']['per_version'][v]['mean'] for v in (0, 1)}          # CAND - START pair AUC per version
    views = {w: cs['recipient_point_by_version_' + w] for w in ('income', 'occupation')}
    within = all(pair_cs[v] <= 0.010 for v in (0, 1)) and all(views[w][v] <= 0.010 for w in views for v in (0, 1))
    worsening = cs['agg']['upper'] < 0                     # RULES 3.3: aggregate upper bound of the pair benefit vs START below 0
    retained = within and not worsening
    pm_c, pm_s = P['cand']['point_margin'], P['start']['point_margin']
    not_below = all(pm_c[v][0] >= pm_s[v][0] for v in (0, 1))
    um = pfx + ('MARGIN_NOT_RESOLVED' if not resolved else 'MARGIN_GAIN_PROFILE_NOT_RETAINED' if not retained
                else 'MARGIN_GAIN_SHIFTED_ELSEWHERE' if not not_below else 'MARGIN_RECOVERED')
    r = d['UM']
    add('W2_S5_FA_R2_UM', 'FA_R2 UM: binding (men\'s guard, both versions), the pooled paired difference on it (verifier paired costs), the '
        'recovery profile from the verifier\'s grid aggregates, the whole-release point margin, and the label', i,
        {'binding': binding, 'by_version': {v: dv[v]['value'] for v in (0, 1)}, 'pooled': pooled['value'], 'se': pooled['se'],
         'resolved': resolved, 'pair_cand_minus_start': pair_cs, 'view_points': views, 'retained': retained, 'not_below_start': not_below,
         'label': um},
        {'binding': r['binding'], 'by_version': r['binding_difference']['by_version'], 'pooled': r['binding_difference']['pooled'],
         'se': r['binding_difference']['se'], 'resolved': r['binding_difference']['resolved_improvement'],
         'pair_cand_minus_start': r['recovery_profile']['pair']['cand_minus_start_by_version'],
         'occupation_points': r['recovery_profile']['single_views']['occupation']['point_by_version'],
         'retained': r['recovery_profile']['retained'], 'not_below_start': r['whole_release']['point_margin']['not_below_start'], 'label': r['label']},
        {str(v): binding[v] for v in (0, 1)} == r['binding'] == plan['start']['binding']
        and all(num_eq(dv[v]['value'], r['binding_difference']['by_version'][str(v)]) for v in (0, 1))
        and num_eq(pooled['value'], r['binding_difference']['pooled']) and num_eq(pooled['se'], r['binding_difference']['se'])
        and resolved == r['binding_difference']['resolved_improvement']
        and all(num_eq(pair_cs[v], r['recovery_profile']['pair']['cand_minus_start_by_version'][str(v)], 1e-12) for v in (0, 1))
        and all(num_eq(views['occupation'][v], r['recovery_profile']['single_views']['occupation']['point_by_version'][str(v)], 1e-12) for v in (0, 1))
        and retained == r['recovery_profile']['retained'] and not_below == r['whole_release']['point_margin']['not_below_start']
        and um == r['label'], 1e-12)
    # ---- D2
    rec2 = d['D2']['comparisons']
    comps = {}
    for name, q in (('START', 'SUCC'), ('CONT', 'FA_TFRO_R1_CONT')):
        c = D.comparison(G[q])
        comps[name] = c
        rr = rec2[name]
        add(f'W2_S3_FA_R2_D2_{name}', f'FA_R2 D2 {name} = {q}: aggregate pair benefit and recipient screens (verifier cells)', i,
            {'benefit': c['benefit'], 'mean': c['benefit_mean'], 'se_agg': c['benefit_se'], 'recipient': c['recipient'], 'pass': c['pass']},
            {'benefit': rr['benefit'], 'mean': rr['benefit_mean'], 'se_agg': rr['benefit_se'], 'recipient': rr['recipient'], 'pass': rr['pass']},
            c['benefit'] == rr['benefit'] and num_eq(c['benefit_mean'], rr['benefit_mean'], 1e-12) and num_eq(c['benefit_se'], rr['benefit_se'], 1e-12)
            and c['recipient'] == rr['recipient'] and c['pass'] == rr['pass'], 1e-12)
    path = ['SUCC'] + plan['paths']['CTRL']
    prim = {q: V.pooled({v: PC[q][str(v)]['total|nll'] for v in (0, 1)}) for q in path}
    above = sorted((q for q in path if prim[q]['value'] <= 0), key=lambda q: -prim[q]['value'])
    fr = {'primary': {q: prim[q]['value'] for q in path}, 'primary_se': {q: prim[q]['se'] for q in path}}
    if not above:
        fr.update({'label': 'NOT_SEPARATED', 'reason': 'NO_BRACKET', 'point': None, 'pass': False})
    else:
        qs = above[0]
        c = D.comparison(G[qs])
        more = sorted(m for m in V.SECONDARY if (lambda x: x['value'] - 2 * x['se'] > 0)(V.pooled({v: PC[qs][str(v)][m] for v in (0, 1)})))
        lab = ('NOT_SEPARATED_FROM_TASK_SPENDING' if c['benefit'] != 'AGG_POSITIVE' else
               'SEPARATED_ON_PRIMARY_COST_ONLY' if more else 'DOMINATES_CONTROL_FRONTIER')
        below = [q for q in path if prim[q]['value'] > 0]
        lo_q = min(below, key=lambda q: prim[q]['value']) if below else None
        lam = (prim[lo_q]['value'] / (prim[lo_q]['value'] - prim[qs]['value'])) if lo_q else None
        dominated = [q for q in below if D._resolved_negative(G[q])]
        fr.update({'label': lab, 'point': qs, 'benefit': c['benefit'], 'benefit_mean': c['benefit_mean'], 'benefit_se': c['benefit_se'],
                   'recipient_vs_bracket': c['recipient'], 'gate': c['gate'], 'secondary_more_spending': more,
                   'interpolation_lower': lo_q, 'lambda': lam, 'dominated': dominated,
                   'pass': lab == 'DOMINATES_CONTROL_FRONTIER' and c['gate'] and not dominated})
    comps['CTRL_FRONTIER'] = fr
    rr = rec2['CTRL_FRONTIER']
    rp = rr['frontier']['primary_cost_differences']
    okf = (fr['label'] == rr['label'] and fr['point'] == rr['point'] and fr['pass'] == rr['pass'] and rr['path'] == path
           and all(num_eq(fr['primary'][q], rp[q]['value'], 1e-9) and num_eq(fr['primary_se'][q], rp[q]['se'], 1e-9) for q in path))
    if fr['point']:
        rf = rr['frontier']
        okf = okf and num_eq(fr['benefit_mean'], rf['benefit_mean'], 1e-12) and num_eq(fr['benefit_se'], rf['benefit_se'], 1e-12) \
            and fr['benefit'] == rf['benefit_vs_bracket']['label'] and fr['secondary_more_spending'] == rf['secondary_more_spending'] \
            and fr['recipient_vs_bracket'] == rr['recipient_vs_bracket'] and fr['gate'] == rr['recipient_gate_vs_bracket'] \
            and num_eq(fr['lambda'], rf['interpolation']['lambda'], 1e-9) and fr['dominated'] == rr['dominated_by_cheaper']
    add('W2_S3_FA_R2_D2_CTRL_FRONTIER', 'FA_R2 D2 (c) CTRL_FRONTIER: Q* on MODEL_SELECT pooled total-NLL costs over the path [SUCC, '
        'CTRL_LO, CTRL_HI], the benefit vs Q*, secondary metrics, recipient gate, cheaper points, interpolation weight', i, fr,
        {'label': rr['label'], 'point': rr['point'], 'pass': rr['pass'], 'primary': {q: rp[q]['value'] for q in rp},
         'benefit_mean': rr['frontier'].get('benefit_mean'), 'benefit_se': rr['frontier'].get('benefit_se'),
         'recipient_vs_bracket': rr['recipient_vs_bracket'], 'lambda': (rr['frontier'].get('interpolation') or {}).get('lambda')}, okf, 1e-9)
    pts = {'START': 'SUCC', 'CONT': 'FA_TFRO_R1_CONT', 'CTRL_FRONTIER': fr['point']}
    tg = {q: {'version_means': D.version_means(NL[q]), 'pass': all(x > 0 for x in D.version_means(NL[q]).values())} for q in pts.values() if q}
    en = {q: {'version_means': D.version_means(EN[q]), 'pass': all(x > 0 for x in D.version_means(EN[q]).values())} for q in tg}
    passed = [n for n in ('START', 'CONT', 'CTRL_FRONTIER') if comps[n]['pass']]
    gating = ['START', 'CONT', 'CTRL_FRONTIER']
    if len(passed) == len(gating):
        lab2 = 'READER_TARGETED' if not all(x['pass'] for x in tg.values()) else 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'
    elif not passed:
        lab2 = 'NOT_SEPARATED'
    else:
        lab2 = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    flags = sorted({f'READER_SENSITIVE_ENHANCED_{q}' for q, x in en.items() if not x['pass']}
                   | {f'READER_TARGETED_{n}' for n in passed if not tg[pts[n]]['pass']} | {'SELECTION_DATA_CALIBRATION_CTRL'}
                   | ({'NO_BRACKET_CTRL'} if fr.get('reason') == 'NO_BRACKET' else set()))
    rt = d['D2']['targeting']
    add('W2_S5_FA_R2_D2', 'FA_R2 D2: reader targeting (non-table winner version means), enhanced flags, the D2 label', i,
        {'label': pfx + lab2, 'passed': passed, 'targeting': tg, 'flags': flags},
        {'label': d['D2']['label'], 'passed': d['D2']['passed'], 'targeting': rt, 'flags': sorted(d['D2']['flags'])},
        pfx + lab2 == d['D2']['label'] and passed == d['D2']['passed'] and flags == sorted(d['D2']['flags'])
        and all(tg[q]['pass'] == rt[q]['pass'] and all(num_eq(tg[q]['version_means'][v], rt[q]['version_means'][str(v)], 1e-12) for v in (0, 1)) for q in tg),
        1e-12)
    # ---- D3 / D4
    refs = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
    per, passed3 = {}, []
    for q in refs:
        c = D.comparison(G[q])
        per[q] = {'benefit': c['benefit'], 'mean': c['benefit_mean'], 'se_agg': c['benefit_se'], 'recipient_gate_pass': c['gate']}
        if c['pass']:
            passed3.append(q)
    r3 = d['D3']['per_reference']
    lab3 = pfx + ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if len(passed3) == len(refs) else 'NOT_SUPERIOR')
    add('W2_S5_FA_R2_D3', 'FA_R2 D3 per strong reference (verifier aggregates)', i, {'label': lab3, 'passed': passed3, 'per_reference': per},
        {'label': d['D3']['label'], 'passed': d['D3']['passed'], 'per_reference': r3},
        lab3 == d['D3']['label'] and sorted(passed3) == sorted(d['D3']['passed'])
        and all(per[q]['benefit'] == r3[q]['benefit'] and per[q]['recipient_gate_pass'] == r3[q]['recipient_gate_pass']
                and num_eq(per[q]['mean'], r3[q]['mean'], 1e-12) and num_eq(per[q]['se_agg'], r3[q]['se_agg'], 1e-12) for q in refs), 1e-12)
    t4 = {q: V.benefit_label(V.aggregate(G[q]['pair']['cells'], G[q]['pair']['cell_se']), threshold=V.TARGET) for q in ('NEURAL_RAW', 'H_LJ')}
    gates = {q: D.comparison(G[q])['gate'] for q in t4}
    if all(x == 'TARGET_AGG_RESOLVED' for x in t4.values()):
        lev = 'MATERIAL_0020_RESOLVED' if all(gates.values()) else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in t4.values()):
        lev = 'MATERIAL_0020_POINT_ONLY'
    elif all(D.comparison(G[q])['benefit'] == 'AGG_POSITIVE' for q in t4):
        lev = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        lev = 'NOT_MET'
    add('W2_S5_FA_R2_D4', 'FA_R2 D4: the 0.020 screen vs NEURAL_RAW and H_LJ (verifier aggregates) and the level', i,
        {'label': pfx + lev, 'per_reference': t4, 'means': {q: V.aggregate(G[q]['pair']['cells'], G[q]['pair']['cell_se'])['mean'] for q in t4}},
        {'label': d['D4']['label'], 'per_reference': {q: d['D4']['per_reference'][q]['label'] for q in t4}},
        pfx + lev == d['D4']['label'] and all(t4[q] == d['D4']['per_reference'][q]['label'] for q in t4), 0)
    cls = ('TASK_INFEASIBLE' if P['cand']['point'] != 'POINT_ELIGIBLE' else 'METHOD_CANDIDATE' if lab2.startswith('COMPONENT') and lab3.endswith('REFERENCES')
           else 'COMPONENT_GAIN_NOT_COMPETITIVE' if lab2.startswith('COMPONENT') else 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' if lab3.endswith('REFERENCES')
           else 'USEFULNESS_MARGIN_ONLY' if um.endswith('MARGIN_RECOVERED') else 'FEASIBLE_NO_RELIABLE_GAIN')
    add('W2_S5_FA_R2_class', 'FA_R2 outcome class, D5 / D6 / D2R labels in the record, the CAND_STRENGTH_DIFFERS_BY_VERSION flag', i,
        {'class': pfx + cls, 'D5': pfx + 'NOT_APPLICABLE_NOT_JOINT', 'D6': pfx + 'NOT_CLAIMED', 'D2R': 'NOT_RUN'},
        {'class': d['outcome_class']['label'], 'D5': d['D5']['label'], 'D6': d['D6']['label'], 'D2R': d['D2R']['label'],
         'flags': d['outcome_class']['flags']},
        pfx + cls == d['outcome_class']['label'] and d['D5']['label'] == pfx + 'NOT_APPLICABLE_NOT_JOINT' and d['D6']['label'] == pfx + 'NOT_CLAIMED'
        and 'CAND_STRENGTH_DIFFERS_BY_VERSION' in d['outcome_class']['flags'], 0)
    psha = V.fsha(pp)
    add('W2_S6_FA_R2_plan_sha256', 'FA_R2: plan_sha256 in the decision record equals the plan file\'s SHA-256', i,
        {'plan_sha256': psha}, {'plan_sha256': DD.get('plan_sha256')}, DD.get('plan_sha256') == psha, 0)
    seeds_ok = DD['seeds'] == {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)} and \
        DD['boot_seeds'] == {f'S{v}{r}': V.BOOT_SEED + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2')}
    add('W2_S3_FA_R2_seeds', 'FA_R2: reader and bootstrap seeds as registered', i, {'ok': seeds_ok}, {'seeds': DD['seeds'], 'boot': DD['boot_seeds']},
        seeds_ok, 0)


# ====================================================================== FS_CPI (selection, no candidate, no MODEL_SELECT computation)
FS_BUDGET = {'inc_nll': 0.0065, 'inc_brier': 0.0035, 'inc_acc': 0.0040}
FS_GRID = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def fs_cpi():
    dp, pp = RES / 'claims' / 'FS_CPI' / 'DECISIONS.json', RES / 'claims' / 'PLAN_FS_CPI.json'
    DD, plan = load(dp), load(pp)
    status = {}
    for v in (0, 1):
        p = RES / 'fs' / f'FIT_CPI_V{v}.json'
        f = load(p)
        s = f['selection']
        i = ins(p, RES / 'cards' / 'FS_CPI_v1.md')
        lam_star, fail = None, None
        for lam in FS_GRID:
            row = s['per_lambda'][str(lam)]
            m = {c: bool(row[c]['estimate'] + 2 * row[c]['se'] <= FS_BUDGET[c]) for c in FS_BUDGET}
            if all(m.values()):
                lam_star = lam
            else:
                bad = sorted(c for c, ok in m.items() if not ok)
                fail = {'lambda': lam, 'failed': bad, 'bounds': {c: row[c]['estimate'] + 2 * row[c]['se'] for c in bad}}
                break
        status[v] = 'NO_FEASIBLE_STRENGTH' if lam_star is None else 'FITTED'
        add(f'W2_S2_FS_CPI_v{v}_lambda_star', f'FS_CPI v{v}: lambda* re-selected from the recorded CALIBRATION income spends over START_CF '
            '(budget = card item 6: NLL 0.0065, Brier 0.0035, accuracy 0.0040)', i,
            {'lambda_star': lam_star, 'first_failure': fail, 'budget_equals_card': s['budget'] == FS_BUDGET},
            {'lambda_star': s['lambda_star'], 'status': f['cand']['status'], 'strength': f['cand']['strength']},
            lam_star == s['lambda_star'] and f['cand']['status'] == status[v] and s['budget'] == FS_BUDGET
            and (f['cand']['strength'] == lam_star), 0)
    i = ins(dp, pp)
    d = DD['decisions']
    pf = 'ADAPTIVE_FS__'
    exp = {'D1': pf + 'NO_FEASIBLE_STRENGTH', 'UM': pf + 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D2': pf + 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH',
           'D3': pf + 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D4': pf + 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH',
           'D5': pf + 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'D6': pf + 'NOT_CLAIMED', 'class': pf + 'TASK_INFEASIBLE'}
    got = {k: d[k]['label'] for k in ('D1', 'UM', 'D2', 'D3', 'D4', 'D5', 'D6')}
    got['class'] = d['outcome_class']['label']
    add('W2_S5_FS_CPI_labels', 'FS_CPI: no candidate in both versions (v0 NO_FEASIBLE_STRENGTH) -> the reading ends at D1; every label '
        'recomputed; adaptive (replacement family) prefix', i, {**exp, 'status_by_version': status},
        {**got, 'plan_status': plan['cand_status_by_version']},
        got == exp and plan['cand_status_by_version'] == {str(v): status[v] for v in (0, 1)} and DD['plan'] == 'FS_CPI', 0)
    # v0 saved nothing; no MODEL_SELECT computation of any FS_CPI release
    fs = RUNS / 'FS'
    v0_saved = sorted(str(x.relative_to(RUNS)) for x in fs.glob('*/V0'))
    evals = sorted(str(x.relative_to(RUNS)) for x in fs.glob('**/EVAL_MS*.json'))
    grids = sorted(str(x.relative_to(V.RES)) for x in (V.RES / 'grid_specs').glob('*.json') if 'FS_CPI' in x.read_text())
    gdirs = sorted(x.name for x in (RUNS / 'GRID').iterdir() if 'FS_CPI' in x.name or x.name.startswith('SAR1_FS'))
    roles = sorted(x.name for x in (RUNS / 'ROLES').iterdir() if x.name.startswith('FS_'))
    ev = [json.loads(x) for x in (V.PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
    ms_fs = [e['utc'] + ' ' + str(e.get('unit_id')) for e in ev if e['role'] == 'MODEL_SELECT'
             and ('_FS_' in str(e.get('unit_id', '')) or 'FS_CPI' in str(e.get('unit_id', '')) or 'FS_CPI' in e.get('purpose', ''))]
    led = [json.loads(x) for x in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if x.strip()]
    units_fs = sorted({e['unit_id'] for e in led if e.get('event') == 'unit_start' and ('_FS_' in e['unit_id'] or 'FS_CPI' in e['unit_id'])})
    ui = {**ins(V.PRIV / 'access' / 'LABEL_ACCESS.jsonl', V.PRIV / 'compute' / 'ledger.jsonl')}
    add('W2_S6_FS_CPI_no_model_select', 'FS_CPI: version 0 saved nothing; no MODEL_SELECT evaluation record, grid bank, grid output, role '
        'member or MODEL_SELECT label read concerns any FS_CPI release', ui,
        {'v0_saved_dirs': v0_saved, 'eval_records': evals, 'grid_specs_naming_FS': grids, 'grid_dirs': gdirs, 'roles': roles,
         'model_select_reads_naming_FS': ms_fs, 'queue_units_naming_FS': units_fs},
        'none', not v0_saved and not evals and not grids and not gdirs and not roles and not ms_fs, 0,
        note='queue units naming FS are the fit units (MODEL_FIT / CALIBRATION only) listed for transparency')


# ====================================================================== D2R from the verifier's robustness cells
def d2r(tag, ms, bank, rec_path, decisive_names):
    R = load(rec_path)
    x = R['d2r']
    G = D.merged(ms, bank)
    i = ins(rec_path, V.WORK / ('ms2_fa_r2_fb_rob.json' if 'FB' in bank else 'ms2_fa_r2_rob.json'))
    per, sign = {}, []
    for comp, e in G.items():
        cells, ses = e['pair']['cells'], e['pair']['cell_se']
        reg = {c: v for c, v in cells.items() if c in V.REG_CELLS}
        rob = {c: v for c, v in cells.items() if c not in V.REG_CELLS}
        a_reg, a_rob, a_all = (V.aggregate(sel, {c: ses[c] for c in sel}) for sel in (reg, rob, cells))
        lab_rob, lab_all = V.benefit_label(a_rob), V.benefit_label(a_all)
        conf = lab_rob == 'AGG_POSITIVE' and lab_all == 'AGG_POSITIVE'
        blocks = {}
        for c, val in rob.items():
            vv, rr, kk = V.parse_cell(c)
            blocks.setdefault((vv, rr), []).append(val)
        n_eff = sum(len(set(b)) for b in blocks.values())
        kd = [len(set(b)) > 1 for b in blocks.values() if len(b) > 1]
        qual = ('READER_SEEDS_EFFECTIVE_ALL_BLOCKS' if kd and all(kd) else 'READER_SEED_REPLICATION_ABSENT' if not any(kd)
                else 'READER_SEEDS_PARTLY_EFFECTIVE')
        sc = (a_reg['mean'] > 0) != (a_rob['mean'] > 0)
        if sc:
            sign.append(comp)
        per[comp] = {'registered_mean': a_reg['mean'], 'robustness_mean': a_rob['mean'], 'robustness_se': a_rob['se_agg'],
                     'robustness_label': lab_rob, 'all_cells_label': lab_all, 'label': 'ROBUSTNESS_CONFIRMED' if conf else 'ROBUSTNESS_NOT_CONFIRMED',
                     'sign_change': sc, 'n_eff_robustness_cells': n_eff, 'reader_seed_qualifier': qual,
                     'n_cells': {'registered': len(reg), 'robustness': len(rob)}}
    rp = x['per_comparator']
    ok = True
    for comp, m in per.items():
        r = rp.get(comp, {})
        ok = ok and r.get('label') == m['label'] and r.get('all_cells_label') == m['all_cells_label'] \
            and num_eq(r.get('registered_mean'), m['registered_mean'], 1e-12) and num_eq(r.get('robustness_mean'), m['robustness_mean'], 1e-12) \
            and num_eq(r.get('robustness_se'), m['robustness_se'], 1e-12) and r.get('sign_change') == m['sign_change'] \
            and r.get('n_eff_robustness_cells') == m['n_eff_robustness_cells'] and r.get('reader_seed_qualifier') == m['reader_seed_qualifier']
    fam = 'ROBUSTNESS_CONFIRMED' if all(per[c]['label'] == 'ROBUSTNESS_CONFIRMED' for c in decisive_names) else 'ROBUSTNESS_NOT_CONFIRMED'
    add(f'W2_S5_{tag}_D2R', f'{tag} D2R: per comparator the registered-cell, robustness-cell (R3 / R4 and k2 / k3) and all-cell aggregates '
        'from the verifier\'s recomputed cells; ROBUSTNESS_CONFIRMED iff AGG_POSITIVE on the robustness cells and on all cells; sign changes; '
        'the label over the decisive set', i,
        {'label': fam, 'decisive': decisive_names, 'sign_changes': sorted(sign), 'per_comparator': per},
        {'label': x['label'], 'decisive': x['decisive'], 'sign_changes': sorted(x['sign_changes']), 'refit': x.get('refit'),
         'per_comparator': {c: rp[c] for c in per if c in rp}},
        ok and fam == x['label'] and sorted(decisive_names) == sorted(x['decisive']) and sorted(sign) == sorted(x['sign_changes']), 1e-12)
    # robustness splits differ from each other and from R1 / R2; bootstrap seeds as declared
    return per


def fa_r2_folds_are_r1():
    """Card FA_TFRB_R2: the bag's fold models are R1's fold fits (R1's recipe and fold seed keys), so the out-of-fold MODEL_FIT quantities
    of the guard estimate must equal R1's at every rho both scans reached (a record-level consistency check, not a recomputation)."""
    for v in (0, 1):
        a = load(RES / 'fa' / f'FIT_R2_V{v}.json')['selection']['per_rho']
        b = load(RES / 'fa' / f'FIT_R1_V{v}.json')['selection']['per_rho']
        common = sorted(set(a) & set(b), key=float)
        diffs = {}
        for r in common:
            ga, gb = a[r]['guard_estimate'], b[r]['guard_estimate']
            for k in ('mf_men', 'mf_women', 'C_mf', 'p_men'):
                if ga[k] != gb[k]:
                    diffs[f'{r}|{k}'] = (ga[k], gb[k])
            if ga['se_C_household'] != gb['se_C']:
                diffs[f'{r}|se_C_household'] = (ga['se_C_household'], gb['se_C'])
            if a[r]['lambda_per_fold'] is None:
                diffs[f'{r}|lambda'] = None
        add(f'W2_S2_FA_R2_v{v}_fold_models_are_R1', f'FA_R2 v{v}: the out-of-fold MODEL_FIT contrast inputs (men / women Brier spends, '
            'C, p, household SE of C) equal FA_R1\'s at every common rho, bitwise (the bag reuses R1\'s fold fits)',
            ins(RES / 'fa' / f'FIT_R2_V{v}.json', RES / 'fa' / f'FIT_R1_V{v}.json'), {'common_rho': common, 'differences': diffs},
            'identical', bool(common) and not diffs, 0)


def rob_registered_equal_main(ms_rob, bank_rob, ms_main, bank_main, tag):
    """The robustness bank's registered cells (R1 / R2 x k0 / k1) equal the main bank's (cache reuse): the verifier's recomputed paired
    cells and SEs of both banks, for every comparator they share."""
    a, b = D.merged(ms_rob, bank_rob), D.merged(ms_main, bank_main)
    diffs, n = [], 0
    for comp in a:
        if comp not in b:
            continue
        for view in ('pair', 'income', 'occupation'):
            for c in V.REG_CELLS:
                n += 1
                da = abs(a[comp][view]['cells'][c] - b[comp][view]['cells'][c])
                ds = abs(a[comp][view]['cell_se'][c] - b[comp][view]['cell_se'][c])
                if da > 0 or ds > 0:
                    diffs.append((comp, view, c, da, ds))
    add(f'W2_S3_{tag}_rob_registered_equal_main', f'{tag}: the robustness bank\'s registered cells equal the main bank\'s (verifier '
        'recomputations of both banks; every shared comparator, view and registered cell)', ins(V.WORK / 'ms2_fa_r2_fb_rob.json'),
        {'compared': n, 'differences': diffs[:10]}, 'identical', n > 0 and not diffs, 0)


def splits_and_seeds(rec_path, tag):
    R = load(rec_path)
    exp = {f'S{v}{r}': V.BOOT_SEED + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2', 'R3', 'R4')}
    seeds = {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in range(4)}
    add(f'W2_S3_{tag}_D2R_seeds', f'{tag} D2R: bootstrap seeds for R1-R4 and reader seeds k0-k3 as registered', ins(rec_path),
        {'boot': exp, 'reader': seeds}, {'boot': R['boot_seeds'], 'reader': R['seeds']}, R['boot_seeds'] == exp and R['seeds'] == seeds, 0)


# ====================================================================== housekeeping: sidecars, index tracking, access over all reads
def git_index_entries():
    """Parse this worktree's own git index file (read as a file; git is not run): {path: blob sha1}."""
    gitfile = (V.ROOT / '.git').read_text().strip()
    idx = Path(gitfile.split(':', 1)[1].strip()) / 'index'
    b = idx.read_bytes()
    if b[:4] != b'DIRC':
        raise ValueError('not a git index')
    ver, n = int.from_bytes(b[4:8], 'big'), int.from_bytes(b[8:12], 'big')
    pos, out, prev = 12, {}, b''
    for _ in range(n):
        start = pos
        sha = b[pos + 40:pos + 60].hex()
        flags = int.from_bytes(b[pos + 60:pos + 62], 'big')
        pos += 62
        if ver >= 3 and flags & 0x4000:
            pos += 2
        if ver == 4:
            # varint (offset encoding) of the number of bytes to strip from the previous path
            k, c = 0, b[pos]
            pos += 1
            k = c & 0x7F
            while c & 0x80:
                c = b[pos]
                pos += 1
                k = ((k + 1) << 7) | (c & 0x7F)
            end = b.index(b'\0', pos)
            path = prev[:len(prev) - k] + b[pos:end]
            pos = end + 1
        else:
            end = b.index(b'\0', pos)
            path = b[pos:end]
            pos = start + ((62 + (2 if ver >= 3 and flags & 0x4000 else 0) + len(path) + 8) // 8) * 8
        out[path.decode()] = sha
        prev = path
    return out, ver, str(idx)


def blob_sha1(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def housekeeping():
    try:
        idx, ver, ipath = git_index_entries()
    except Exception as ex:                                  # recorded, never hidden
        idx, ver, ipath = None, None, f'unreadable: {type(ex).__name__}: {ex}'
    rel_c = 'results/pcrl_strong_anchor_release_v1/claims/'
    for reading in ('FA_REG', 'FB_REG', 'FB_R1', 'FA_R1', 'FS_CPI'):
        sc = RES / 'claims' / reading / 'PLAN_SHA256.json'
        s = load(sc)
        plan = V.ROOT / ('results/pcrl_strong_anchor_release_v1/' + s['plan'])
        dec = RES / 'claims' / reading / 'DECISIONS.json'
        mine = {'plan_sha256': V.fsha(plan), 'decisions_sha256': V.fsha(dec)}
        tracked = {}
        if idx is not None:
            for f in (plan, dec, sc):
                k = str(Path(f).relative_to(V.ROOT))
                tracked[k] = {'in_index': k in idx, 'index_blob_equals_file': idx.get(k) == blob_sha1(f)}
        ok = s['plan_sha256'] == mine['plan_sha256'] and s['decisions_sha256'] == mine['decisions_sha256'] and \
            (idx is not None and all(t['in_index'] and t['index_blob_equals_file'] for t in tracked.values()))
        add(f'W2_S6_{reading}_plan_sidecar', f'{reading}: PLAN_SHA256.json equals the SHA-256 of the plan file and of DECISIONS.json; the three '
            'files are tracked in this worktree\'s git index with identical content (file-level; git not run)', {**ins(sc, plan, dec), 'git_index': ipath},
            {**mine, 'index_version': ver, 'tracked': tracked}, {'plan_sha256': s['plan_sha256'], 'decisions_sha256': s['decisions_sha256']}, ok, 0,
            note='adjudication entry for wave-1 S6_FB_R1 / S6_FA_R1 plan-hash failures (those failures stand; deviation 9 adds the sidecars)'
            if reading in ('FB_R1', 'FA_R1') else None)
    for reading in ('FA_REG', 'FB_REG'):
        files = sorted((RES / 'claims' / reading).iterdir())
        tr = {str(f.relative_to(V.ROOT)): {'in_index': idx is not None and str(f.relative_to(V.ROOT)) in idx,
                                           'index_blob_equals_file': idx is not None and idx.get(str(f.relative_to(V.ROOT))) == blob_sha1(f)}
              for f in files if f.is_file()}
        add(f'W2_S6_{reading}_tracked', f'claims/{reading}/ files are tracked in the worktree index with identical content (wave-1 snapshot '
            'showed them untracked); file-level check, git not run', {'git_index': ipath},
            tr, 'tracked and identical', bool(tr) and all(t['in_index'] and t['index_blob_equals_file'] for t in tr.values()), 0,
            note='the index records staged content; that the stage was committed is supported by the worktree reflog commits after 06:21Z '
                 'but HEAD\'s tree was not read')
    # access over every read logged so far (wave-1 rules)
    la, lg = V.PRIV / 'access' / 'LABEL_ACCESS.jsonl', V.PRIV / 'compute' / 'ledger.jsonl'
    rl = Path((V.ROOT / '.git').read_text().strip().split(':', 1)[1].strip()) / 'logs' / 'HEAD'
    ev = [json.loads(x) for x in la.read_text().splitlines() if x.strip()]
    led = [json.loads(x) for x in lg.read_text().splitlines() if x.strip()]
    t = lambda s: dt.datetime.fromisoformat(s)
    reflog = {}
    for line in rl.read_text().splitlines():
        head, _, msg = line.partition('\t')
        parts = head.split()
        reflog[parts[1]] = {'utc': dt.datetime.fromtimestamp(int(parts[-2]), dt.timezone.utc), 'msg': msg[:140]}
    starts = {e['unit_id']: t(e['utc']) for e in led if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: t(e['utc']) for e in led if e.get('event') == 'unit_end'}
    bad = {'card': [], 'ms_fit': [], 'closed': [], 'cal_sex': [], 'unit': []}
    for e in ev:
        ut = t(e['utc'])
        if e['operation'] in ('fit', 'calibrate'):
            rc = reflog.get(e.get('card_commit'))
            if not (rc and rc['utc'] <= ut and V.fsha(V.ROOT / e['card']) == e['card_sha256']):
                bad['card'].append(e['utc'] + ' ' + e['unit_id'])
        if e['role'] == 'MODEL_SELECT':
            chain = [e['entry_point'].split(':')[0]] + list(e.get('module_chain') or [])
            if any(m.startswith(('sar.fa', 'sar.fb', 'sar.ctl', 'sar.fs')) for m in chain):
                bad['ms_fit'].append(e['utc'] + ' ' + e['unit_id'])
        if e['role'] in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT'):
            bad['closed'].append(e['utc'])
        if e['role'] == 'CALIBRATION' and 'sex' in e['fields']:
            bad['cal_sex'].append(e['utc'])
        u = e.get('unit_id')
        if not u or u not in starts or starts[u] > ut or (u in ends and ends[u] < ut):
            bad['unit'].append(e['utc'] + ' ' + str(u))
    rcpt = load(V.PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json')
    first = min(t(e['utc']) for e in ev)
    i = {**ins(la, lg), str(rl): V.fsha(rl)}
    add('W2_S6_access_all_reads', 'access over every label read logged so far: after the verified receipt; fitting reads after their card\'s '
        'commit (reflog) with the card unchanged; no MODEL_SELECT read in a fitting module; no closed role; no SEX on CALIBRATION; every read '
        'inside its queue unit', i, {'reads': len(ev), 'first_read': first.isoformat(), 'violations': bad,
                                     'fitting_cards': sorted({e['card'] for e in ev if e['operation'] in ('fit', 'calibrate')})},
        'no violation', first > t(rcpt['verified_utc']) and not any(bad.values()), 0)
    # FA_R2 and FS_CPI ordering: card commits before fitting reads, plan commit before the first MODEL_SELECT read of FA_R2
    card_c = {'FA_TFRB_R2': 'd3b453d1d', 'FS_CPI_v1': '3a8ee012c'}
    found = {k: next(({'commit': h, 'utc': r['utc'].isoformat(), 'msg': r['msg']} for h, r in reflog.items() if h.startswith(c)), None)
             for k, c in card_c.items()}
    fit_reads = {k: sorted(e['utc'] for e in ev if e['operation'] in ('fit', 'calibrate') and k in e.get('card', '')) for k in card_c}
    ok = all(found[k] and fit_reads[k] and t(found[k]['utc']) <= t(fit_reads[k][0]) for k in card_c)
    add('W2_S6_card_commits_before_fits', 'cards FA_TFRB_R2 (d3b453d1d) and FS_CPI_v1 (3a8ee012c) committed (worktree reflog) before their '
        'first fitting reads', i, {'commits': found, 'first_fit_reads': {k: (v[0] if v else None) for k, v in fit_reads.items()}},
        'commit before first fitting read', ok, 0)
    pc = next(({'commit': h, 'utc': r['utc'].isoformat(), 'msg': r['msg']} for h, r in reflog.items() if h.startswith('a85bd035a')), None)
    reads = sorted(e['utc'] for e in ev if e['role'] == 'MODEL_SELECT' and str(e.get('unit_id', '')).startswith(('SAR_EVAL_FA_R2', 'SAR_GRID_SAR1_FA_R2')))
    pw = load(RES / 'claims' / 'PLAN_FA_R2.json')['written_utc']
    add('W2_S6_PLAN_FA_R2_before_first_MS', 'PLAN_FA_R2 written and committed (a85bd035a, worktree reflog) before the reading\'s first MODEL_SELECT '
        'utility / reader read', i, {'plan_written_utc': pw, 'commit': pc, 'first_reading_ms_read_utc': reads[0] if reads else None},
        'plan before the first read', bool(pc and reads and t(pw) <= t(pc['utc']) < t(reads[0])), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--fa-r2-rob', default=None)
    a = ap.parse_args()
    ms1 = load(V.WORK / 'ms_recomputed.json')
    ms2 = load(V.WORK / 'ms2_fa_r2_fb_rob.json')
    fa_r2_selection(ms1)
    fa_r2_folds_are_r1()
    fa_r2_decisions(ms2)
    fs_cpi()
    d2r('FB_R1', ms2, 'SAR1_FB_R1_ROB', RES / 'claims' / 'FB_R1_D2R' / 'D2R.json', ['NEURAL_RAW', 'FB_PFO_TASK', 'FB_R1_ALT_HI'])
    splits_and_seeds(RES / 'claims' / 'FB_R1_D2R' / 'D2R.json', 'FB_R1')
    rob_registered_equal_main(ms2, 'SAR1_FB_R1_ROB', ms1, 'SAR1_FB_R1', 'FB_R1')
    if a.fa_r2_rob:
        ms3 = load(Path(a.fa_r2_rob))
        rec = sorted((RES / 'claims' / 'FA_R2_D2R').glob('*.json'))
        d2r('FA_R2', ms3, 'SAR1_FA_R2_ROB', rec[0], ['SUCC', 'FA_TFRO_R1_CONT', 'FA_R2_CTRL_HI'])
        splits_and_seeds(rec[0], 'FA_R2')
        rob_registered_equal_main(ms3, 'SAR1_FA_R2_ROB', ms2, 'SAR1_FA_R2', 'FA_R2')
    housekeeping()
    out = {'checks': D.CHECKS, 'n': len(D.CHECKS), 'fail': [c['id'] for c in D.CHECKS if c['status'] != 'PASS'],
           'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    V.jdump(Path(a.out), out)
    print(json.dumps({'n': out['n'], 'fail': out['fail']}, indent=1))


if __name__ == '__main__':
    main()

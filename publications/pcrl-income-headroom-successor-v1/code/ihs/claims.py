"""Claim conjunctions of the frozen claims draft (roles/statistical_reviewer/CLAIMS_DRAFT.md, registered at 74376912c),
encoded verbatim for one scorecard's DEV_REPORT endpoints. Used by ihs.report only for a promoted finalist.

Endpoint keys (built by ihs.locks.endpoint_roster from generic roles):
  UTILITY|S|<task>|<ref>|<metric>|s<s>   no_excess, cap            (12 per seed; reader-free)
  SUBGROUP|S|sex<g>|s<s>                 no_excess, 0.005          (2 per seed; reader-free)
  BEN|<X>|s<s>                           benefit = AUC_pair(X) - AUC_pair(S), margin 0, refs (0.020,)   X in REF, TW, CC
  NW|<X>|<view>|s<s>                     no_excess, AUC_view(S) - AUC_view(X), margin 0.010             X in REF, TW, CC
  UTILITY|H_LJ|income|... (family D)     H_LJ's own income clauses (the R2 SE floor)
Tests: clause bound L1 = point - c1 SE (benefit), U1 = point + c1 SE (no_excess); family bounds at c_P.
"""
from __future__ import annotations

R2_FACTOR = 1.7267108            # 2 x sqrt(3955 / 5306): the registered MODEL_SELECT reserve at DEV_REPORT's household scale
REFS = ('NEURAL_RAW', 'H_LJ')
SEEDS = (0, 1)
TASK_METRICS = [(t, r, m) for t in ('income', 'occupation') for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]


def _u(eps, t, r, m, s, who='S'):
    return eps[f'UTILITY|{who}|{t}|{r}|{m}|s{s}']


def evaluate(eps, CV, declared):
    """declared: {'TW': True, 'CC': bool, 'occupation_inherited': bool (bitwise inherited occupation output on DEV, both seeds),
    'control_designated': bool}. Returns {claim: record} for E, R, P, N, M, C, T, O, COMPETITIVE_SUCCESSOR."""
    out = {}
    # ---- E: development point eligibility
    ug = [k for k, e in eps.items() if k.startswith('UTILITY|S|') and not e['point'] <= e['margin']]
    sg = [k for k, e in eps.items() if k.startswith('SUBGROUP|S|') and not e['point'] <= e['margin']]
    rg = [f'NW|{x}|{v}|s{s}' for x in REFS for v in ('income', 'occupation') for s in SEEDS if not eps[f'NW|{x}|{v}|s{s}']['point'] <= 0.010]
    e_point = not (ug or sg or rg)
    cond28 = [k for k in eps if k.startswith('UTILITY|S|') or k.startswith('SUBGROUP|S|')]
    viol = [k for k in cond28 if eps[k]['family_lower'] > eps[k]['margin']]
    unres = [k for k in cond28 if eps[k]['family_upper'] > eps[k]['margin']]
    out['E'] = {'label': 'DEV_POINT_ELIGIBLE' if e_point else 'DEV_POINT_INELIGIBLE', 'holds': e_point,
                'failed_in_order': ug + sg + rg,
                'utility_confidence': ('ESTABLISHED_UTILITY_VIOLATION' if viol else 'CONFIDENCE_FEASIBLE' if not unres
                                       else f'CONFIDENCE_UNRESOLVED ({len(unres)} of 28)'),
                'recipient_confidence': {'clause_pass': sum(eps[f'NW|{x}|{v}|s{s}']['clause_pass'] for x in REFS for v in ('income', 'occupation') for s in SEEDS),
                                         'of': 8}}
    # ---- R: utility reserve on DEV_REPORT (R2), income confidence (R3), planning projection (R4)
    r2 = {}
    for s in SEEDS:
        for (t, r, m) in TASK_METRICS:
            if t != 'income':
                continue
            e, h = _u(eps, t, r, m, s), _u(eps, t, r, m, s, 'H_LJ')
            se = max(e['se'], h['se']); margin = e['margin'] - e['point']
            r2[f'{t}|{r}|{m}|s{s}'] = {'margin': margin, 'se_used': se, 'required': R2_FACTOR * se, 'met': margin >= R2_FACTOR * se,
                                       'margin_se_units': margin / se if se > 0 else float('inf'),
                                       'projected_17510_se_units': margin / (e['se'] * (3955 / 17510) ** 0.5) if e['se'] > 0 else float('inf')}
    if all(v['met'] for v in r2.values()):
        r2_lab = 'RESERVE_REPLICATED'
    elif all(v['margin'] >= 0 for v in r2.values()):
        r2_lab = 'RESERVE_ERODED'
    else:
        r2_lab = 'CAP_EXCEEDED_AT_POINT'
    inc = [k for k in eps if k.startswith('UTILITY|S|income|')]
    r3 = ('ESTABLISHED_INCOME_VIOLATION' if any(eps[k]['family_lower'] > eps[k]['margin'] for k in inc) else
          'INCOME_CONFIDENCE_RESOLVED' if all(eps[k]['family_upper'] <= eps[k]['margin'] for k in inc) else
          f"INCOME_CONFIDENCE_UNRESOLVED ({sum(eps[k]['family_upper'] > eps[k]['margin'] for k in inc)} of 12)")
    out['R'] = {'label': f'{r2_lab}; {r3}', 'R1': 'RESERVE_MET (MODEL_SELECT, both seeds; the nomination gate)', 'R2': r2_lab,
                'R2_detail': r2, 'R3': r3, 'holds': r2_lab == 'RESERVE_REPLICATED'}
    # ---- N and P
    nw = {f'{x}|{v}|s{s}': eps[f'NW|{x}|{v}|s{s}'] for x in REFS for v in ('income', 'occupation') for s in SEEDS}
    if all(e['clause_pass'] for e in nw.values()):
        n_lab = 'NON_WORSENING_ESTABLISHED'
    elif all(e['point'] <= 0.010 for e in nw.values()):
        n_lab = 'NON_WORSENING_UNRESOLVED'
    else:
        n_lab = 'NON_WORSENING_VIOLATED_AT_POINT'
    out['N'] = {'label': n_lab, 'holds': n_lab == 'NON_WORSENING_ESTABLISHED', 'max_point': max(e['point'] for e in nw.values()),
                'clauses': {k: {'point': e['point'], 'U1': e['clause_U1']} for k, e in nw.items()}}
    ben = {f'{x}|s{s}': eps[f'BEN|{x}|s{s}'] for x in REFS for s in SEEDS}
    p_ok = CV and e_point and all(e['clause_pass'] for e in ben.values()) and out['N']['holds']
    if p_ok:
        p_lab = 'SUPERIOR_TO_ALL_ELIGIBLE'
    elif not CV:
        p_lab = 'CONTROL_FAILURE'
    elif ug:
        p_lab = 'FAILS_DEV_UTILITY_GATE'
    elif sg:
        p_lab = 'FAILS_DEV_SUBGROUP_POINT_GUARD'
    elif rg:
        p_lab = 'FAILS_DEV_RECIPIENT_POINT_GATE'
    else:
        p_lab = 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
    out['P'] = {'label': p_lab, 'holds': p_ok, 'benefits': {k: {'point': e['point'], 'L1': e['clause_L1'], 'se': e['se']} for k, e in ben.items()},
                'condition_vector': {'CV': CV, 'E_POINT': e_point, **{f'BEN_{k}': e['clause_pass'] for k, e in ben.items()},
                                     **{f'NW_{k}': e['clause_pass'] for k, e in nw.items()}}}
    # ---- T: 0.020 target (nested in P)
    if p_ok and all(e['clause_L1'] > 0.020 for e in ben.values()):
        t_lab = 'MATERIAL_TARGET_ESTABLISHED'
    elif all(e['point'] > 0.020 for e in ben.values()) and any(e['clause_L1'] <= 0.020 for e in ben.values()):   # reviewer C1
        t_lab = 'TARGET_POINT_ONLY'
    else:
        t_lab = 'TARGET_NOT_ESTABLISHED'
    out['T'] = {'label': t_lab, 'holds': t_lab == 'MATERIAL_TARGET_ESTABLISHED'}

    # ---- M: twin first, then P (strict fixed sequence)
    def attribution(x):
        bens = [eps[f'BEN|{x}|s{s}'] for s in SEEDS]
        nws = [eps[f'NW|{x}|{v}|s{s}'] for v in ('income', 'occupation') for s in SEEDS]
        ok = CV and e_point and all(b['clause_pass'] for b in bens) and all(n['clause_pass'] for n in nws)
        return ok, {f'BEN|{x}|s{s}': {'point': eps[f'BEN|{x}|s{s}']['point'], 'L1': eps[f'BEN|{x}|s{s}']['clause_L1']} for s in SEEDS}
    if declared.get('TW'):
        step1, det = attribution('TW')
        m_lab = ('MECHANISM_ATTRIBUTION_ESTABLISHED' if step1 and p_ok else 'TASK_ONLY_ATTRIBUTION_ONLY' if step1 else 'NOT_ESTABLISHED')
        out['M'] = {'label': m_lab, 'holds': m_lab == 'MECHANISM_ATTRIBUTION_ESTABLISHED', 'step1_twin': step1, 'step1_detail': det,
                    'step2': p_lab if step1 else 'NOT_TESTED_SEQUENCE_STOPPED',
                    'control_designated_finalist': bool(declared.get('control_designated'))}
    else:
        out['M'] = {'label': 'NOT_CLAIMED (no matched twin in the bank)', 'holds': None}
    if declared.get('CC'):
        ok, det = attribution('CC')
        out['C'] = {'label': 'COMPONENT_ATTRIBUTION_ESTABLISHED' if ok else 'NOT_ESTABLISHED', 'holds': ok, 'detail': det}
    else:
        out['C'] = {'label': 'NOT_CLAIMED', 'holds': None}
    # ---- O: inherited occupation constraints
    occ = {}
    for s in SEEDS:
        for (t, r, m) in TASK_METRICS:
            if t == 'occupation':
                e, h = _u(eps, t, r, m, s), _u(eps, t, r, m, s, 'H_LJ')
                occ[f'{t}|{r}|{m}|s{s}'] = (e, h)
        for g in (0, 1):
            occ[f'guard|sex{g}|s{s}'] = (eps[f'SUBGROUP|S|sex{g}|s{s}'], eps[f'SUBGROUP|H_LJ|sex{g}|s{s}'])
    det = {}
    for k, (e, h) in occ.items():
        se = max(e['se'], h['se']); m_s = e['margin'] - e['point']; m_h = h['margin'] - h['point']
        det[k] = {'margin': m_s, 'inherited_margin': m_h, 'se_used': se, 'meets_R2_rule': m_s >= R2_FACTOR * se, 'worse_than_inherited': m_s < m_h - 1e-15}
    if declared.get('occupation_inherited'):
        o_lab = 'INHERITED_CONSTRAINTS_BIND' if any(not d['meets_R2_rule'] for d in det.values()) else 'INHERITED_CONSTRAINTS_HAVE_RESERVE'
    else:
        o_lab = ('OCCUPATION_CONSTRAINTS_RELIEVED' if all(d['meets_R2_rule'] for d in det.values()) else
                 'OCCUPATION_IMPROVED_BUT_BINDING' if not any(d['worse_than_inherited'] for d in det.values()) else 'OCCUPATION_WORSENED')
    out['O'] = {'label': o_lab, 'detail': det, 'below_rule': sorted(k for k, d in det.items() if not d['meets_R2_rule'])}
    # ---- composite
    if e_point and r2_lab == 'RESERVE_REPLICATED' and p_ok:
        comp = 'COMPETITIVE_SUCCESSOR'
    elif p_ok:
        comp = 'SUPERIOR_WITHOUT_RESERVE'
    else:
        comp = 'NOT_COMPETITIVE'
    out['COMPOSITE'] = {'label': comp, 'with': {'O': o_lab, 'R3': r3}}
    # ---- D: descriptive (marginal 95% intervals only; no alpha)
    b2 = [eps.get(f'DESC|BEN|B2TWIN|s{s}') for s in SEEDS]
    gain = bool(all(e is not None and e['point'] - MARGINAL * e['se'] > 0 for e in b2))
    out['D'] = {'label': 'GAIN_OVER_B2TWIN_DESCRIPTIVE' if gain else 'NO_DESCRIPTIVE_GAIN_OVER_B2TWIN', 'holds': None,
                'GAIN_OVER_B2TWIN_DESCRIPTIVE': gain,
                'pair_difference_vs_B2TWIN': {f's{s}': None if e is None else {'point': e['point'], 'marginal95': [e['point'] - MARGINAL * e['se'], e['point'] + MARGINAL * e['se']]}
                                              for s, e in zip(SEEDS, b2)}}
    return out


MARGINAL = 1.9605641


def condition_values(c):
    """Every key of the P, M (step 1) and C conjunctions with its truth value (for the key-by-key bank comparison)."""
    out = dict(c['P']['condition_vector'])
    for name in ('M', 'C'):
        det = c[name].get('step1_detail') or c[name].get('detail') or {}
        for k, v in det.items():
            out[f'{name}:{k}'] = v['L1'] is not None and v['L1'] > 0
    return out


def reader_sensitive(enh, orig, auc_changes=None):
    """Reviewer C2: compare the P, M and C conjunctions key by key between the ENHANCED and ORIGINAL scorecards; for each
    differing key give the side under the validation's paired reader-class change rule (which release's winning reader read
    differently between the banks), and mark the affected ENHANCED labels."""
    a, b = condition_values(enh), condition_values(orig)
    keys = [k for k in a if a[k] != b.get(k)]
    detail = []
    for k in keys:
        side = None
        if auc_changes is not None:
            side = auc_changes.get(k)
        detail.append({'key': k, 'ENHANCED': a[k], 'ORIGINAL': b.get(k), 'side': side})
    affected = sorted({('P' if not k.startswith(('M:', 'C:')) else k[0]) for k in keys})
    return {'keys': detail, 'affected_claims': affected}

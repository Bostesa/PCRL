"""Verifier check 3 (DRF1): reader comparisons and decision labels of a reading, recomputed with separate code (label-free).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check3_readers.py --plan PLAN_DRF_R1 --bank DRF1_R1 \
        --check2 results/pcrl_direct_recovery_frontier_v1/roles/verifier/out/DRF_VER_C2_V1_check2.json --out FILE

Inputs (records only; no label): the grid's per-version GRID.json (per-cell winners' SCORE AUCs, recorded paired cell SEs, the
non-table winners' SCORE AUCs), the committed reading plan, my check-2 output (G1 per subject and my recomputed MODEL_SELECT paired
costs; the recorded paired costs are compared there), and the reading's DECISIONS.json when it exists (compared, never used).
Recomputed (vlib.py; my reading of RULES.md and NRM1 RULES.md 5 and 7):
  - cell values from the per-cell SCORE AUCs (pair benefit = comparator - candidate; recipient difference = candidate - comparator);
  - the aggregate paired screen per comparison and view (block means, x, version means, se_hh, se_rs, se_agg with the 0.001 floor),
    the pair label, the 0.020 target label, the recipient gates;
  - D2: the CTRL frontier (path TWIN + saved CTRL points; Q* = the cheapest point whose pooled MODEL_SELECT paired total-NLL cost is
    >= the candidate's; AGG_POSITIVE vs Q*; no secondary metric resolved higher vs Q*; recipient gate vs Q*; no cheaper point
    resolved less recoverable), OBJ (pair benefit, recipient gate, secondary metrics, the single-point spend rule), reader targeting
    (non-table winner's point benefit > 0 in both version means vs Q* and vs OBJ_MI), the D2 label;
  - FC (vs SUCC), TWIN beside, D3 over the point-eligible references, D4 (0.020 against NEURAL_RAW and H_LJ), the outcome class.
The SE of each cell is the grid's recorded household-bootstrap SE (it is recomputed from the cached reader scores in check 4).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
REF_SUBJECT = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
               'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE'}
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
VERSIONS = (0, 1)


def load_grid(bank):
    g = {'cells': {}, 'paired': {}}
    for v in VERSIONS:
        p = PRIV / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json'
        x = json.loads(p.read_text())
        g.setdefault('files', {})[v] = {'path': str(p.relative_to(REPO)), 'sha256': V.sha256_file(p), 'unit_id': x.get('unit_id')}
        g['cells'].update(x['cells'])
        for cand, comps in x['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = g['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
        g.setdefault('exact_aliases', {})[v] = x.get('exact_aliases')
    return g


def my_cells(grid, cand, comp, view, items, key=None):
    """Cell values recomputed from the per-cell SCORE AUCs (key: None = the registered winner; 'nolookup' = the non-table winner)."""
    out = {}
    for c in V.REGISTERED:
        a = grid['cells'][c][cand][view]
        b = grid['cells'][c][comp][view]
        if key is not None:
            a, b = a['extras'][key], b['extras'][key]
        out[c] = (b['score_auc'] - a['score_auc']) if view == 'pair' else (a['score_auc'] - b['score_auc'])
    if key is None:
        rec = grid['paired'].get(cand, {}).get(comp, {}).get(view, {}).get('cells', {})
        for c in V.REGISTERED:
            items.append(V.cmp_num(f'A:{cand}_vs_{comp}:{view}:{c}:cell_value', out[c], rec.get(c), 1e-15))
    return out


def pooled(costs, key):
    """NR's pooling across versions: d = mean of the two version differences; SE = mean of the two SEs."""
    vals = [costs[v][key] for v in VERSIONS]
    return {'d': sum(x['value'] for x in vals) / 2.0, 'se': sum(x['se'] for x in vals) / 2.0,
            'by_version': {v: [costs[v][key]['value'], costs[v][key]['se']] for v in VERSIONS}}


def costs_from_check2(c2, sid, spec):
    """My recomputed MODEL_SELECT paired costs loss(sid) - loss(spec) per version and metric, from the check-2 items."""
    out = {v: {} for v in VERSIONS}
    for it in c2['items']:
        n = it['name']
        for v in VERSIONS:
            pre = f'C:{sid}:V{v}:vs_{spec}:'
            if n.startswith(pre) and (n.endswith(':value') or n.endswith(':se')):
                key, fld = n[len(pre):].rsplit(':', 1)
                out[v].setdefault(key, {})[fld] = it['verifier']
    for v in VERSIONS:
        if not out[v] or any('value' not in x or 'se' not in x for x in out[v].values()):
            return None
    return out


def optimistic(plan, items):
    """RULES.md 5.1 (my reading), from the recorded OPT_L / OPT_S values (their mappings are not saved, so their values are taken
    as recorded) and the CTRL path values of the fit record named by the plan (verified in check 1 for the in-sample mode). A
    construction X is USEFUL in a mode iff in both versions: within all 8 budgets (its in-sample spends), and against B = the
    cheapest path point (TWIN first, spend 0) whose total-NLL spend over the twin is >= X's (ties in path order): AUC*_pair(B) -
    AUC*_pair(X) >= 0.020 and neither single view's AUC* exceeds B's by more than 0.010."""
    tag = plan.get('optimistic_ctrl_fit_tag', '')
    per = {}
    for x in ('OPT_S', 'OPT_L'):
        per[x] = {}
        for mode in ('in_sample', 'nested'):
            by = {}
            for v in VERSIONS:
                fit = json.loads((RES / 'dr' / f'FIT_{tag}V{v}.json').read_text())
                op = json.loads((RES / 'dr' / f'OPTIMISTIC_V{v}.json').read_text())
                ent = op['entries'][x]['insample']['empirical']
                bud = op['budgets']
                within = all(ent['spend'][k] <= bud[k] + 1e-12 for k in V.METRICS8)
                items.append(V.cmp_eq(f'E:{x}:V{v}:recorded_feasible_equals_spend_within_budgets', bool(ent['feasible']), within))
                if mode == 'in_sample':
                    sx = ent['spend']['inc_nll'] + ent['spend']['occ_nll']
                    ob = ent['observable_merged_wires']
                    ax = {'pair': ob['auc_pair'], 'income': ob['auc_inc'], 'occupation': ob['auc_occ']}
                    path = []
                    for p in fit['path_full']:
                        e = p['insample']['empirical']
                        o2 = e['observable_merged_wires']
                        path.append((int(p['step']), e['spend']['inc_nll'] + e['spend']['occ_nll'],
                                     {'pair': o2['auc_pair'], 'income': o2['auc_inc'], 'occupation': o2['auc_occ']}))
                else:
                    ne = op['nested']['entries'][x]
                    sx = ne['spend_over_identity']['total_nll'][0]
                    ex = ne['heldout_exact_auc_mean']
                    ax = {'pair': ex['auc_pair'], 'income': ex['auc_inc'], 'occupation': ex['auc_occ']}
                    npth = fit['nested']['path']
                    path = [(int(s), npth['spend_over_identity'][s]['total_nll'][0],
                             {'pair': npth['heldout_exact_auc_mean'][s]['auc_pair'], 'income': npth['heldout_exact_auc_mean'][s]['auc_inc'],
                              'occupation': npth['heldout_exact_auc_mean'][s]['auc_occ']}) for s in sorted(npth['spend_over_identity'], key=int)]
                qual = [p for p in path if p[1] >= sx]
                if not qual:
                    by[v] = {'label': 'NOT_EVALUABLE_NO_BRACKET', 'spend': sx}
                    continue
                b = min(qual, key=lambda p: (p[1], path.index(p)))
                gain = b[2]['pair'] - ax['pair']
                exc = {vw: ax[vw] - b[2][vw] for vw in ('income', 'occupation')}
                ok = bool(within and gain >= 0.020 and all(e_ <= 0.010 for e_ in exc.values()))
                by[v] = {'label': 'USEFUL' if ok else 'NOT_USEFUL', 'bracket': f'PATH_STEP_{b[0]}', 'bracket_spend': b[1], 'spend': sx,
                         'gain_pair': gain, 'single_view_excess': exc, 'within_budget': within}
            labs = [by[v]['label'] for v in VERSIONS]
            per[x][mode] = {'by_version': by, 'label': ('USEFUL' if all(l_ == 'USEFUL' for l_ in labs) else
                                                       'NOT_EVALUABLE_NO_BRACKET' if any(l_ == 'NOT_EVALUABLE_NO_BRACKET' for l_ in labs)
                                                       else 'NOT_USEFUL')}
    ins = [per[x]['in_sample']['label'] for x in per]
    if 'USEFUL' in ins:
        lab = 'USEFUL_OPTIMISTIC_POINT'
    elif all(l_ == 'NOT_USEFUL' for l_ in ins):
        lab = 'NO_USEFUL_OPTIMISTIC_POINT'
    else:
        lab = 'UNDETERMINED'
    return {'label': lab, 'per_construction': per, 'held_out_survives': [x for x in per if per[x]['nested']['label'] == 'USEFUL'],
            'ctrl_fit_tag': tag}


def failure_stage(d1, d2, d3, cand_alias_of_twin, opt_label):
    if d1 != 'TASK_FEASIBLE' or cand_alias_of_twin:
        legal = 'L2'
    elif d3 == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES':
        legal = 'L5'
    elif d2 == 'COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS':
        legal = 'L4'
    else:
        legal = 'L3'
    if legal == 'L4':
        return legal, 'STAGE_4_BEATS_CONTROLS_MISSES_STRONG_REFERENCES'
    if legal == 'L5':
        return legal, 'STAGE_5_COMPETITIVE_CONDITIONS_PASS'
    if opt_label == 'NO_USEFUL_OPTIMISTIC_POINT':
        return legal, 'STAGE_1_NO_USEFUL_OPTIMISTIC_POINT'
    if opt_label == 'USEFUL_OPTIMISTIC_POINT':
        return legal, ('STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS' if legal == 'L2' else 'STAGE_3_KEEPS_UTILITY_TIES_CONTROLS')
    return legal, 'UNDETERMINED_OPTIMISTIC_DIAGNOSTIC'


def comparison(grid, cand, comp):
    pair = my_cells(grid, cand, comp, 'pair', [])
    se = grid['paired'][cand][comp]['pair']['cell_se']
    ag = V.aggregate(pair, se)
    rec = {'pair': {'aggregate': ag, 'label': V.benefit_label(ag), 'target_label': V.target_label(ag)}}
    for view in ('income', 'occupation'):
        cells = my_cells(grid, cand, comp, view, [])
        rec[view] = V.recipient(cells, grid['paired'][cand][comp][view]['cell_se'])
        rec[view].pop('per_version', None)
    rec['gates'] = bool(rec['income']['gate'] and rec['occupation']['gate'])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='PLAN_DRF_R1')
    ap.add_argument('--bank', default='DRF1_R1')
    ap.add_argument('--check2', required=True)
    ap.add_argument('--decisions', default=None)
    ap.add_argument('--stopped', action='store_true', help='a reading whose reader stage is stopped (candidate fails G1): no grid')
    ap.add_argument('--no-record', action='store_true', help='never open the reading\'s DECISIONS.json (labels computed before it is read; '
                                                              'no reader-budget flags, no comparison) [added 04:29Z]')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    plan_p = RES / 'claims' / f'{a.plan}.json'
    plan = json.loads(plan_p.read_text())
    info['plan'] = {'path': str(plan_p.relative_to(REPO)), 'sha256': V.sha256_file(plan_p)}
    reg_tag, reg_inputs = registration_tag(plan)
    info['registration'] = {'tag': reg_tag, **reg_inputs}
    c2p = Path(a.check2)
    c2 = json.loads(c2p.read_text())
    info['check2'] = {'path': str(c2p.relative_to(REPO)) if c2p.is_relative_to(REPO) else str(c2p), 'sha256': V.sha256_file(c2p),
                      'overall': c2['overall']['result']}
    cand, twin, obj = plan['cand'], plan['twin']['id'], plan['obj_mi']['config_id']
    ctrl = list(plan['paths'].get('CTRL') or [])
    specs = plan['compare_specs']
    # ---- G1 (check 2)
    g1 = c2['g1']
    d1 = {'cand': g1.get(cand), 'label': 'TASK_FEASIBLE' if g1.get(cand) == 'G1_PASS_ALL_28' else
          ('INCOMPLETE' if g1.get(cand) in (None, 'INCOMPLETE') else 'CAND_INFEASIBLE'),
          'start_point_eligible': g1.get('REF:SUCC') == 'G1_PASS_ALL_28', 'twin_point_eligible': g1.get('TWB_C64') == 'G1_PASS_ALL_28',
          'obj_mi_point_eligible': g1.get(obj) == 'G1_PASS_ALL_28', 'ctrl_point_eligible': {c: g1.get(c) == 'G1_PASS_ALL_28' for c in ctrl}}
    opt = optimistic(plan, items)
    alias_twin = bool((plan.get('alias_records') or {}).get('cand_alias_of_twin', False))
    dec_path = Path(a.decisions) if a.decisions else RES / 'claims' / plan['plan_id'] / 'DECISIONS.json'
    if a.no_record:
        dec_path = Path('/nonexistent/DECISIONS.json')   # never opened: labels before the record
    info['record_opened'] = not a.no_record
    if a.stopped:
        # RULES.md 2: a usefulness failure may stop the competitive reader stage; D2-D5 (and FC / TWIN, which carry D2's gate
        # label) read NOT_ASSESSED_TASK_INFEASIBLE; the class is TASK_INFEASIBLE; the stage follows section 5.2 (legal stage L2).
        items.append(V.cmp_eq('S:stop_allowed_only_when_candidate_fails_G1', d1['label'], 'CAND_INFEASIBLE'))
        na = 'NOT_ASSESSED_TASK_INFEASIBLE'
        legal, stage = failure_stage(d1['label'], na, na, alias_twin, opt['label'])
        labels = {'D1': d1['label'], 'D2': na, 'FC': na, 'TWIN': na, 'D3': na, 'D4': na, 'D5': na, 'D6': 'NOT_CLAIMED',
                  'optimistic': opt['label'], 'legal_stage': legal, 'failure_stage': stage, 'class': 'TASK_INFEASIBLE'}
        compared = compare_with_record(dec_path, labels, items, info, reg_tag)
        out = {'schema': 'DRF1-verifier-check3-v1', 'mode': 'stopped_reader_stage', 'info': info, 'labels': labels, 'd1': d1,
               'optimistic': opt, 'compared_with_decisions': compared}
        return finish(out, items, a.out)
    grid = load_grid(a.bank)
    info['grid_files'] = grid['files']
    # ---- A: every recorded cell value of the candidate's comparisons, recomputed from the per-cell AUCs
    for comp in sorted(grid['paired'].get(cand, {})):
        for view in ('pair', 'income', 'occupation'):
            my_cells(grid, cand, comp, view, items)
    # ---- B: aggregates and labels
    comps = {c: comparison(grid, cand, c) for c in dict.fromkeys([twin, obj] + ctrl + list(REFS))}
    # ---- costs (my check-2 recomputation)
    costs = {}
    for k in [twin, obj] + ctrl + ['SUCC']:
        spec = specs.get(k) or ('FRC:FA/TWB_C64' if k == twin else f'DR/{k}')
        costs[k] = costs_from_check2(c2, cand, spec)
        items.append(V.cmp_eq(f'C:costs_available_vs_{k}', costs[k] is not None, True))
    # ---- D2 (c): the CTRL frontier
    alias_of = plan.get('alias_of') or {}
    path = [twin] + [c for c in ctrl if alias_of.get(c) is None]
    pts = {}
    for p in path:
        if costs.get(p) is None:
            continue
        tot = pooled(costs[p], 'total|nll')
        pts[p] = {'pooled_total_nll_cost_diff_cand_minus_point': tot['d'], 'se': tot['se'], 'by_version': tot['by_version']}
    qual = {p: x for p, x in pts.items() if x['pooled_total_nll_cost_diff_cand_minus_point'] <= 0}
    q = max(qual, key=lambda p: (qual[p]['pooled_total_nll_cost_diff_cand_minus_point'], -path.index(p))) if qual else None
    fr = {'path': path, 'points': pts, 'bracketing_control': q}
    if len(pts) < len(path):
        fr['label'] = 'INCOMPLETE'
    elif q is None:
        fr['label'], fr['reason'] = 'NOT_SEPARATED_FROM_TASK_SPENDING', 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    else:
        sec = {m: pooled(costs[q], m) for m in SECONDARY}
        higher = [m for m, s in sec.items() if s['d'] - 2 * s['se'] > 0]
        fr['secondary_vs_Q'] = sec
        fr['secondary_resolved_higher'] = higher
        lab = comps[q]['pair']['label']
        fr['pair_label_vs_Q'] = lab
        if lab != 'AGG_POSITIVE':
            fr['label'], fr['reason'] = 'NOT_SEPARATED_FROM_TASK_SPENDING', 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
        elif higher:
            fr['label'] = 'SEPARATED_ON_PRIMARY_COST_ONLY'
        else:
            fr['label'] = 'DOMINATES_CONTROL_FRONTIER'
    c1 = fr.get('label') == 'DOMINATES_CONTROL_FRONTIER'
    c2_gate = bool(q is not None and comps[q]['gates'])
    dominated = []
    for p, x in pts.items():
        if x['pooled_total_nll_cost_diff_cand_minus_point'] > 0:          # p is cheaper than the candidate
            ag = comps[p]['pair']['aggregate']
            if ag.get('complete') and ag['x'] + 2 * ag['se_agg'] < 0 and all(m < 0 for m in ag['version_means'].values()):
                dominated.append(p)
    c3 = not dominated
    ctrl_pass = bool(c1 and c2_gate and c3)
    # ---- D2 (o): OBJ_MI
    o = comps[obj]
    tot = pooled(costs[obj], 'total|nll') if costs.get(obj) else None
    if tot is None:
        spend_state = 'INCOMPLETE'
    elif tot['d'] <= 0:
        spend_state = 'OBJ_MI_SPENDS_AT_LEAST_AS_MUCH'
    elif tot['d'] - 2 * tot['se'] <= 0:
        spend_state = 'OBJ_MI_SPENDS_LESS_UNRESOLVED'
    else:
        spend_state = 'OBJ_MI_SPENDS_LESS_RESOLVED'
    sec_o = {m: pooled(costs[obj], m) for m in SECONDARY} if costs.get(obj) else {}
    higher_o = [m for m, s in sec_o.items() if s['d'] - 2 * s['se'] > 0]
    obj_parts = {'pair_label': o['pair']['label'], 'recipient_gates': o['gates'], 'secondary_resolved_higher': higher_o,
                 'spend_state': spend_state, 'spend_d': tot}
    obj_pass = bool(o['pair']['label'] == 'AGG_POSITIVE' and o['gates'] and not higher_o and spend_state != 'OBJ_MI_SPENDS_LESS_RESOLVED')
    # ---- D2 (d): reader targeting with the non-table winner
    tgt = {}
    for k in ([q] if q else []) + [obj]:
        cells = my_cells(grid, cand, k, 'pair', [], key='nolookup')
        vm = {v: sum(cells[c] for c in V.REGISTERED if c.startswith(f'S{v}')) / 4.0 for v in VERSIONS}
        tgt[k] = {'version_means': vm, 'positive_both': bool(all(x > 0 for x in vm.values()))}
    d_pass = bool(q is not None and tgt.get(q, {}).get('positive_both') and tgt[obj]['positive_both'])
    if d1['label'] != 'TASK_FEASIBLE':
        d2 = 'NOT_ASSESSED_TASK_INFEASIBLE'
    elif ctrl_pass and obj_pass and d_pass:
        d2 = 'COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS'
    elif ctrl_pass and obj_pass:
        d2 = 'READER_TARGETED'
    elif not ctrl_pass and not obj_pass:
        d2 = 'NOT_SEPARATED'
    else:
        d2 = 'PARTIAL_CTRL_FRONTIER_ONLY' if ctrl_pass else 'PARTIAL_OBJ_ONLY'
    # ---- FC, TWIN beside
    s = comps['SUCC']
    fc = 'FAMILY_CHANGE_BELOW_START_RESOLVED' if (s['pair']['label'] == 'AGG_POSITIVE' and s['gates']) else 'FAMILY_CHANGE_NOT_RESOLVED'
    t_ = comps[twin]
    twin_lab = 'BELOW_TWIN_RESOLVED' if t_['pair']['label'] == 'AGG_POSITIVE' else 'NOT_RESOLVED_VS_TWIN'
    twin_with_gates = 'BELOW_TWIN_RESOLVED' if (t_['pair']['label'] == 'AGG_POSITIVE' and t_['gates']) else 'NOT_RESOLVED_VS_TWIN'
    # ---- D3, D4
    elig = {r: g1.get(REF_SUBJECT[r]) == 'G1_PASS_ALL_28' for r in REFS}
    nolook = {}
    for r in REFS:                                   # the reader-budget credit rule's input: non-table winner's point benefit
        cells = my_cells(grid, cand, r, 'pair', [], key='nolookup')
        vm = {v: sum(cells[c] for c in V.REGISTERED if c.startswith(f'S{v}')) / 4.0 for v in VERSIONS}
        nolook[r] = {'version_means': vm, 'positive_both': bool(all(x > 0 for x in vm.values()))}
    budget_flags = None
    dec_p0 = dec_path
    if dec_p0.is_file():
        budget_flags = (json.loads(dec_p0.read_text()).get('reader_budget') or None)
    asym = {r: bool(budget_flags and 'ASYMMETRIC' in str(budget_flags.get(r, ''))) for r in REFS}
    passed = [r for r in REFS if elig[r] and comps[r]['pair']['label'] == 'AGG_POSITIVE' and comps[r]['gates']
              and (not asym[r] or nolook[r]['positive_both'])]
    failed = [r for r in REFS if elig[r] and r not in passed]
    d3 = 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if (d1['label'] == 'TASK_FEASIBLE' and not failed and passed) else 'NOT_SUPERIOR'
    tl = {r: comps[r]['pair']['target_label'] for r in ('NEURAL_RAW', 'H_LJ')}
    gates4 = all(comps[r]['gates'] for r in ('NEURAL_RAW', 'H_LJ'))
    if all(x == 'TARGET_AGG_RESOLVED' for x in tl.values()):
        d4 = 'MATERIAL_0020_RESOLVED' if gates4 else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(comps[r]['pair']['aggregate']['x'] > 0.020 for r in ('NEURAL_RAW', 'H_LJ')):
        d4 = 'MATERIAL_0020_POINT_ONLY'
    elif all(comps[r]['pair']['label'] == 'AGG_POSITIVE' for r in ('NEURAL_RAW', 'H_LJ')):
        d4 = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        d4 = 'NOT_MET'
    d2_met = d2 == 'COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS'
    d3_met = d3 == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES'
    if d1['label'] == 'INCOMPLETE':
        cls = 'INCOMPLETE'
    elif d1['label'] != 'TASK_FEASIBLE':
        cls = 'TASK_INFEASIBLE'
    elif d2_met and d3_met:
        cls = 'METHOD_CANDIDATE'
    elif d2_met:
        cls = 'COMPONENT_GAIN_NOT_COMPETITIVE'
    elif d3_met:
        cls = 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED'
    else:
        cls = 'FEASIBLE_NO_RELIABLE_GAIN'
    legal, stage = failure_stage(d1['label'], d2, d3, alias_twin, opt['label'])
    labels = {'D1': d1['label'], 'D2': d2, 'CTRL_FRONTIER': fr.get('label'), 'bracketing_control': q, 'CTRL_pass': ctrl_pass,
              'c1': c1, 'c2_recipient_gate_vs_Q': c2_gate, 'c3_no_cheaper_dominates': c3, 'dominated_by_cheaper': dominated,
              'OBJ_pass': obj_pass, 'OBJ_spend_state': spend_state, 'reader_targeting_pass': d_pass, 'FC': fc, 'TWIN': twin_lab,
              'TWIN_with_recipient_gates': twin_with_gates, 'D3': d3, 'D3_passed': passed, 'D3_failed': failed, 'D3_eligible': elig,
              'D3_reader_budget_flags_used': budget_flags, 'D3_nontable_winner_benefit': nolook,
              'D4': d4, 'D4_target_labels': tl, 'D5': 'NOT_APPLICABLE_NOT_JOINT', 'D6': 'NOT_CLAIMED', 'optimistic': opt['label'],
              'legal_stage': legal, 'failure_stage': stage, 'class': cls}
    compared = compare_with_record(dec_path, labels, items, info, reg_tag)
    if compared is not None:
        dec = json.loads(dec_path.read_text())['decisions']
        rf = (((dec.get('D2') or {}).get('comparisons') or {}).get('CTRL_FRONTIER') or {}).get('frontier') or {}
        items.append(V.cmp_eq('D:bracketing_control', q, rf.get('bracketing_control')))
        rd3 = dec.get('D3') or {}
        items.append(V.cmp_eq('D:D3_passed_list', sorted(passed), sorted(rd3.get('passed') or [])))
        items.append(V.cmp_eq('D:D3_failing_list', sorted(failed), sorted(rd3.get('failing') or [])))
    out = {'schema': 'DRF1-verifier-check3-v1', 'mode': 'reader_stage_run', 'info': info, 'labels': labels, 'frontier': fr,
           'obj': obj_parts, 'targeting': tgt, 'optimistic': opt,
           'd1': d1, 'comparisons': {k: {'pair': {'x': v_['pair']['aggregate'].get('x'),
                                                  'version_means': v_['pair']['aggregate'].get('version_means'),
                                                  'se_hh': v_['pair']['aggregate'].get('se_hh'), 'se_rs': v_['pair']['aggregate'].get('se_rs'),
                                                  'se_agg': v_['pair']['aggregate'].get('se_agg'), 'label': v_['pair']['label'],
                                                  'target_label': v_['pair']['target_label']},
                                         'income': {k2: v_['income'].get(k2) for k2 in ('label', 'gate', 'r_v', 'se_v')},
                                         'occupation': {k2: v_['occupation'].get(k2) for k2 in ('label', 'gate', 'r_v', 'se_v')}}
                                     for k, v_ in comps.items()},
           'compared_with_decisions': compared}
    return finish(out, items, a.out)


def registration_tag(plan):
    """RULES.md 7 (my reading): a reading is registered iff its card was committed before any MODEL_SELECT outcome of the campaign
    (the first MODEL_SELECT read of the access log) and, for revision >= 1, its prospective note names the observed failure, the
    changed operation and a MODEL_FIT or SYNTHETIC trigger; otherwise ADAPTIVE_R0_LATE_CARD, ADAPTIVE_R<n>_NO_NOTE or ADAPTIVE_R<n>.
    Returns (tag or None, inputs). Added 04:25Z for the adaptive R2 reading (prefixed labels)."""
    import subprocess
    card = plan['card']
    out = subprocess.run(['git', '-C', str(REPO), 'log', '--diff-filter=A', '--format=%cI', '--', card], capture_output=True, text=True)
    when = dt.datetime.fromisoformat(out.stdout.strip().splitlines()[-1]).astimezone(dt.timezone.utc) if out.stdout.strip() else None
    reads = [json.loads(x) for x in (PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
    first_ms = min((dt.datetime.fromisoformat(e['utc']) for e in reads if e['role'] == 'MODEL_SELECT'), default=None)
    before = bool(when and first_ms and when < first_ms)
    rev = int(plan.get('revision', 0))
    note = plan.get('revision_note') or {}
    complete = all(note.get(k) for k in ('observed_failure', 'changed_operation', 'trigger'))
    if rev == 0:
        tag = None if before else 'ADAPTIVE_R0_LATE_CARD'
    elif not complete:
        tag = f'ADAPTIVE_R{rev}_NO_NOTE'
    elif not before or note.get('trigger') not in ('MODEL_FIT', 'SYNTHETIC'):
        tag = f'ADAPTIVE_R{rev}'
    else:
        tag = None
    return tag, {'card_first_commit_utc': when.isoformat() if when else None,
                 'first_model_select_read_utc': first_ms.isoformat() if first_ms else None, 'revision': rev,
                 'trigger': note.get('trigger'), 'note_complete': complete}


def compare_with_record(dec_path, labels, items, info, tag=None):
    """The verifier's labels against the reading's DECISIONS.json (when it exists; never used to compute a label). A non-registered
    reading's labels carry the tag prefix '<tag>__' (RULES.md 7): compared both prefixed and unprefixed."""
    if not dec_path.is_file():
        return None
    dec = json.loads(dec_path.read_text())['decisions']
    info['decisions'] = {'path': str(dec_path.relative_to(REPO)), 'sha256': V.sha256_file(dec_path)}
    compared = {}
    items.append(V.cmp_eq('D:registered', tag is None, dec.get('registered')))
    items.append(V.cmp_eq('D:adaptive_tag', tag, dec.get('adaptive_tag')))
    for key, rkey in (('D1', 'D1'), ('D2', 'D2'), ('FC', 'FC'), ('TWIN', 'TWIN'), ('D3', 'D3'), ('D4', 'D4'), ('D5', 'D5'),
                      ('D6', 'D6'), ('optimistic', 'optimistic'), ('failure_stage', 'failure_stage'), ('class', 'outcome_class')):
        rec = dec.get(rkey)
        lab = rec.get('label') if isinstance(rec, dict) else rec
        mine = labels.get(key)
        if tag is not None and key != 'optimistic':
            unpref = rec.get('label_unprefixed') if isinstance(rec, dict) else None
            items.append(V.cmp_eq(f'D:label_{key}_unprefixed', mine, unpref))
            mine = f'{tag}__{mine}'
        compared[key] = {'verifier': mine, 'record': lab}
        items.append(V.cmp_eq(f'D:label_{key}', mine, lab))
    fs = dec.get('failure_stage') or {}
    if 'legal_stage' in fs:
        items.append(V.cmp_eq('D:legal_stage', labels.get('legal_stage'), str(fs['legal_stage']).split('_')[0]))
    return compared


def finish(out, items, path):
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out['summary'] = {g: V.summarize(x) for g, x in groups.items()}
    out['overall'] = V.summarize(items)
    out['items'] = items
    if path:
        V.dump(path, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'summary': out['summary'], 'labels': out['labels'],
                                 'compared': out.get('compared_with_decisions')}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Repair-stage comparison, reserve-qualified frontier, original benchmark and claim ladder of the PCRL correction-transfer campaign
(label-free: reads saved selection, ladder, bank-evaluation and reader-grid records; RULES.md sections 7, 9 and 11).

    python3 -m cts.claims all          results/<campaign>/repair/RULE_COMPARISON.csv / .json, MATCHED_CONTROLS.csv / .json,
                                       RESERVE_QUALIFIED_FRONTIER.csv / .json, ORIGINAL_BENCHMARK.csv / .json, CLAIM_LADDER.json

Conventions. A repair "pair" joins R0 (seed-0 references and reader cells) with R1 (seed-1). Pair benefit = comparator SCORE AUC minus
candidate SCORE AUC (positive favours the candidate); recipient views = candidate minus comparator. Stability: cts.rules
stability_summary (8 cells = 2 partitions x 2 household repetitions x 2 reader seeds). The formal claim ladder applies to the
selected recipe only; the recipe selection gave NO_RECIPE, so every rung is NOT_MET. The per-strength ladder evaluated for the
coalition-aware direction (REFIT_PAIR, full deployment) is DESCRIPTIVE and never a claim.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys

from . import common as C
from . import rules as RL
from . import trackf as TF

R_PARTS = ('R0', 'R1')
STRENGTHS = (0.1, 0.2, 0.3, 0.5, 1.0)
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
RULE_IDS = ('OLD-B80', 'OLD-B50', 'SIMPLE', 'UNCERTAINTY')
# Whole-release reserve ratios of the required references as the source recorded them with the same frozen rule (registered in
# RULES.md section 11 before any fit): per-seed minimum ratio.
SOURCE_REFERENCE_RATIOS = {'SUCC': (0.321, 0.624), 'H_LJ': (0.321, 0.624), 'B2TWIN': (0.321, 0.624), 'TWJ': (0.321, 0.624),
                           'RETIRED_B3CTL': (0.321, 0.349), 'NEURAL_RAW': (1.947, 1.558)}
OUT = C.RESULTS / 'repair'


def _grid(bank):
    out = {}
    for s in (0, 1):
        f = C.RUNS / 'GRID' / bank / f'S{s}' / 'GRID.json'
        if f.is_file():
            out[s] = json.loads(f.read_text())
    return out


def _paired(g, cand, comp, view='pair', sign=1.0):
    cells, ses = {}, {}
    for s, gs in g.items():
        p = gs['paired'].get(cand, {}).get(comp, {}).get(view)
        if p is None:
            continue
        for c, v in p['cells'].items():
            if '@' not in c:
                cells[c] = sign * v; ses[c] = p['cell_se'][c]
    return cells, ses


def _auc(g, name):
    out = {}
    for s, gs in g.items():
        v = [cres[name]['pair']['score_auc'] for c, cres in gs['cells'].items() if '@' not in c and name in cres]
        if v:
            out[s] = sum(v) / len(v)
    return out


def _bank_eval(bank):
    f = C.RUNS / 'PAIRS' / f'BANK_EVAL__{bank}.json'
    return json.loads(f.read_text())['members'] if f.is_file() else {}


def _ladder(P, did):
    f = TF.pdir(P) / 'LADDER_CHECK' / f'{did}.json'
    return json.loads(f.read_text())['estimators']


def _baseline(P, did='REFIT_PAIR'):
    return json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())['baseline']


def _summ(cells, ses, threshold=0.0):
    if len(cells) != 8:
        n = len(cells)
        return {'label': f'NOT_EVALUABLE_{n}_CELLS', 'n_cells': n, 'mean': (sum(cells.values()) / n) if n else None}
    st = RL.stability_summary(cells, ses, threshold=threshold)
    return {'label': st['label'], 'n_cells': 8, 'mean': st['mean'], 'per_seed_mean': st['per_seed_mean'], 'sd_total': st['sd_total'],
            'min_lower': st['min_lower'], 'min_cell': min(cells.values()), 'max_cell': max(cells.values())}


def _write_csv(stem, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / f'{stem}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- rules on the repair panels (RULES.md section 9)
def rule_comparison():
    rows, comps = [], []
    for P in R_PARTS:
        for did in ('REFIT_PAIR', 'REFIT_LOC', 'REFIT_CONT', 'REFIT_PAIR_TR'):
            sf = TF.pdir(P) / 'SELECT' / f'{did}.json'
            rf = TF.pdir(P) / 'SELECT' / f'{did}__REALIZED.json'
            if not (sf.is_file() and rf.is_file()):
                continue
            sel = json.loads(sf.read_text())['selected']
            real = json.loads(rf.read_text())['realized']
            for rule, s in sel.items():
                r = {'partition': P, 'direction': did, 'rule': rule, 'selected': s}
                for where in ('CHECK', 'MS'):
                    o = real[where][rule]['outcome']
                    r[f'{where}_reserve_crossing'] = o['reserve_crossing']
                    r[f'{where}_point_cap_crossing'] = o['point_cap_crossing']
                    r[f'{where}_max_slack_use'] = o['max_slack_use']
                rows.append(r)
    idx = {(r['partition'], r['direction'], r['rule']): r for r in rows}
    for did in ('REFIT_PAIR', 'REFIT_LOC', 'REFIT_PAIR_TR'):
        for dep in ('full', 'ens'):
            for a in ('SIMPLE', 'UNCERTAINTY'):
                for b in ('OLD-B80', 'OLD-B50'):
                    per = {}
                    for P in R_PARTS:
                        ra, rb = idx.get((P, did, f'{a}|{dep}')), idx.get((P, did, f'{b}|{dep}'))
                        if not (ra and rb):
                            continue
                        res = RL.compare_rules(
                            {'rule': a, 'selected': ra['selected'], 'reserve_crossing': ra['CHECK_reserve_crossing'] or ra['MS_reserve_crossing'],
                             'pair_auc': None},
                            {'rule': b, 'selected': rb['selected'], 'reserve_crossing': rb['CHECK_reserve_crossing'] or rb['MS_reserve_crossing'],
                             'pair_auc': None})
                        per[P] = {'a_selected': ra['selected'], 'b_selected': rb['selected'], **res}
                    if len(per) == 2:
                        comps.append({'direction': did, 'deployment': dep, 'rule_a': a, 'rule_b': b,
                                      'R0': per['R0']['category'], 'R1': per['R1']['category'],
                                      'a_improves_on_both': per['R0']['a_improves'] and per['R1']['a_improves'], 'detail': per})
    _write_csv('RULE_OUTCOMES_REPAIR', rows)
    C.write_json(OUT / 'RULE_COMPARISON.json', {
        'schema': 'CTS1-rule-comparison-v1', 'utc': C.now(), 'outcomes': rows, 'comparisons': comps,
        'note': ('compare_rules on the same fitted directions (RULES.md section 9): identity never improves; a smaller strength counts '
                 'only if it avoids a realized crossing (CHECK panel or MODEL_SELECT) that the other rule incurs; "improves" needs the '
                 'category on both R partitions. Rule-selected strengths were not reader-evaluated (protection change not available).')})
    return rows, comps


# ---------------------------------------------------------------- matched controls at common strengths (rung 3, descriptive)
def matched_controls():
    gF, gC, gT = _grid('R_FRONTIER'), _grid('R_COMP0'), _grid('R_TRACKR')
    rows = []
    for s in STRENGTHS:
        cand = f'REFIT_PAIR@s{s}'
        for comp in ('W', f'REFIT_CONT@s{s}', f'REFIT_LOC@s{s}') + REQUIRED:
            cells, ses = _paired(gF, cand, comp)
            rows.append({'candidate': cand, 'comparator': comp, 'bank': 'R_FRONTIER', **_summ(cells, ses)})
        cells, ses = _paired(gC, f'COMP:REFIT_PAIR@s{s}', cand, sign=-1.0)      # benefit of PAIR over its compression twin
        rows.append({'candidate': cand, 'comparator': f'COMP:REFIT_PAIR@s{s} (R0 only)', 'bank': 'R_COMP0', **_summ(cells, ses)})
        for ctl in (f'REFIT_CONT@s{s}', f'REFIT_LOC@s{s}'):                       # each control against the witness
            cells, ses = _paired(gF, ctl, 'W')
            rows.append({'candidate': ctl, 'comparator': 'W', 'bank': 'R_FRONTIER', **_summ(cells, ses)})
    for cand, comp in (('H3R:REFIT_PAIR_TR@s1.0', 'H3R:W'), ('H3R:REFIT_PAIR_TR@s0.5', 'H3R:W'), ('H3R:REFIT_PAIR_TR@s1.0', 'REFIT_PAIR@s1.0'),
                       ('H3R:REFIT_PAIR_TR@s0.5', 'REFIT_PAIR@s0.5'), ('H3R:W', 'W'), ('HCR:W', 'W'), ('H3R:W', 'SUCC'),
                       ('H3R:REFIT_PAIR_TR@s1.0', 'SUCC'), ('H3R:REFIT_PAIR_TR@s1.0', 'NEURAL_RAW'), ('H3R:REFIT_PAIR_TR@s1.0', 'H_LJ')):
        cells, ses = _paired(gT, cand, comp)
        rows.append({'candidate': cand, 'comparator': comp, 'bank': 'R_TRACKR', **_summ(cells, ses)})
    _write_csv('MATCHED_CONTROLS', [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows])
    C.write_json(OUT / 'MATCHED_CONTROLS.json', {
        'schema': 'CTS1-matched-controls-v1', 'utc': C.now(), 'rows': rows,
        'note': ('pair benefit = comparator AUC - candidate AUC (positive favours the candidate); 8 cells = R0 / R1 x 2 repetitions x 2 '
                 'reader seeds; the compression twins exist on R0 only (R1 men\'s reserve slack is negative, so a matched GUARD '
                 'utilization is undefined) and give 4 cells, which the 8-cell stability rule cannot label')})
    return rows


# ---------------------------------------------------------------- reserve-qualified frontier and original benchmark (section 11)
def frontier():
    gF, gT, gC = _grid('R_FRONTIER'), _grid('R_TRACKR'), _grid('R_COMP0')
    evF, evT, evC = _bank_eval('R_FRONTIER'), _bank_eval('R_TRACKR'), _bank_eval('R_COMP0')
    points, single = [], []

    def kind(n):
        if n.startswith('REFIT_PAIR@') or n.startswith('H3R:REFIT_PAIR_TR@'):
            return 'candidate'
        return 'reference' if n in REQUIRED else 'control'
    for g, ev in ((gF, evF), (gT, evT)):
        for n, e in ev.items():
            if n in [p['id'] for p in points]:
                continue
            auc = _auc(g, n)
            if sorted(e['per_seed']) == ['0', '1'] and sorted(auc) == [0, 1]:
                points.append({'id': n, 'kind': kind(n), 'min_ratio': {0: e['per_seed']['0']['min_ratio'], 1: e['per_seed']['1']['min_ratio']},
                               'pair_auc': {0: auc[0], 1: auc[1]}, 'point_pass_24_4': e['point_pass_all'],
                               'binding': {s: e['per_seed'][s]['binding'] for s in e['per_seed']}})
    for n, e in evC.items():
        if not n.startswith('COMP:'):                  # the bank's other members are R0 copies of the R_FRONTIER points
            continue
        auc = _auc(gC, n)
        single.append({'id': n, 'kind': 'control (R0 only)', 'min_ratio_seed0': e['per_seed']['0']['min_ratio'],
                       'pair_auc_seed0': auc.get(0), 'point_pass_12_2_seed0': e['per_seed']['0']['point_pass_12'] and e['per_seed']['0']['guard_pass']})
    for r in REQUIRED:
        auc = _auc(gF, r)
        points.append({'id': r, 'kind': 'reference', 'min_ratio': {0: SOURCE_REFERENCE_RATIOS[r][0], 1: SOURCE_REFERENCE_RATIOS[r][1]},
                       'pair_auc': {0: auc[0], 1: auc[1]}, 'point_pass_24_4': True, 'binding': 'source record (RULES.md section 11)'})
    rq = RL.reserve_qualified_frontier([{k: p[k] for k in ('id', 'kind', 'min_ratio', 'pair_auc')} for p in points])
    q = {r['id']: r for r in rq['points']}
    rows = [{'release': p['id'], 'kind': p['kind'], 'min_ratio_seed0': p['min_ratio'][0], 'min_ratio_seed1': p['min_ratio'][1],
             'reserve_qualified': q[p['id']]['qualified'], 'on_qualified_frontier': p['id'] in rq['frontier'],
             'pair_auc_seed0': p['pair_auc'][0], 'pair_auc_seed1': p['pair_auc'][1], 'pair_auc_mean': q[p['id']]['pair_auc_mean'],
             'point_pass_24_4': p['point_pass_24_4']} for p in points]
    _write_csv('RESERVE_QUALIFIED_FRONTIER', rows + [{'release': s['id'], 'kind': s['kind'], 'min_ratio_seed0': s['min_ratio_seed0'],
                                                      'pair_auc_seed0': s['pair_auc_seed0']} for s in single])
    C.write_json(OUT / 'RESERVE_QUALIFIED_FRONTIER.json', {
        'schema': 'CTS1-reserve-qualified-frontier-v1', 'utc': C.now(), 'points': points, 'single_seed_points': single, 'result': rq,
        'qualified_protected_points': [r['release'] for r in rows if r['reserve_qualified'] and r['kind'] != 'reference'],
        'note': ('the same whole-release reserve (28 conditions, k = 2, H_LJ floor; ohn.rules.reserve) for every point; qualified iff '
                 'the per-seed minimum ratio is >= 2 on both seeds; reference ratios are the source records registered in RULES.md '
                 'section 11; a positive reserve-qualified result is not superiority over every original reference')})
    bench = []
    for s in STRENGTHS:
        for cand in (f'REFIT_PAIR@s{s}',) + ((f'H3R:REFIT_PAIR_TR@s{s}',) if s in (0.5, 1.0) else ()):
            g = gF if cand.startswith('REFIT_') else gT
            e = (evF if cand.startswith('REFIT_') else evT).get(cand)
            rec = {'candidate': cand, 'point_pass_24_4': e['point_pass_all'] if e else None}
            gates, sup, tgt = {}, {}, {}
            phi = RL.load_rules()['screens']['aggregate_paired_rule']['phi_prior']
            for r in REQUIRED:
                for view in ('income', 'occupation'):
                    cells, ses = _paired(g, cand, r, view)
                    if len(cells) == 8:
                        og = RL.per_cell_gate(cells)
                        ag = RL.aggregate_paired_screen(cells, ses, phi[view], alias=all(v == 0.0 for v in cells.values()))
                        gates[f'{r}|{view}'] = {'old_gate': og['label'], 'old_gate_pass': og['pass'], 'agg1': ag['label'],
                                                'agg1_pass': ag['pass'], 'agg1_U': ag.get('U')}
                cells, ses = _paired(g, cand, r)
                sup[r] = _summ(cells, ses)['label']
                if r in ('NEURAL_RAW', 'H_LJ'):
                    t = _summ(cells, ses, threshold=RL.TARGET)
                    tgt[r] = ('DEV_GRID_TARGET_STABLE' if t['label'] == 'STABLE_POSITIVE' else
                              'DEV_GRID_TARGET_POINT_ONLY' if (t.get('mean') or -1) > RL.TARGET else 'DEV_GRID_TARGET_NOT_MET')
            rec['old_gate_pass_all'] = len(gates) == 12 and all(v['old_gate_pass'] for v in gates.values())
            rec['agg1_pass_all'] = len(gates) == 12 and all(v['agg1_pass'] for v in gates.values())
            rec['superiority_all_required'] = all(v == 'STABLE_POSITIVE' for v in sup.values())
            rec['target_label'] = ('DEV_GRID_TARGET_STABLE' if all(v == 'DEV_GRID_TARGET_STABLE' for v in tgt.values()) else
                                   'DEV_GRID_TARGET_POINT_ONLY' if all(v in ('DEV_GRID_TARGET_STABLE', 'DEV_GRID_TARGET_POINT_ONLY') for v in tgt.values())
                                   else 'DEV_GRID_TARGET_NOT_MET')
            rec.update({'gates': gates, 'superiority': sup, 'target_per_reference': tgt})
            bench.append(rec)
    _write_csv('ORIGINAL_BENCHMARK', [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in bench])
    C.write_json(OUT / 'ORIGINAL_BENCHMARK.json', {
        'schema': 'CTS1-original-benchmark-v1', 'utc': C.now(), 'records': bench,
        'note': ('original-benchmark comparison (RULES.md section 11, unchanged): 24 clauses + 4 guards at the point, the old per-cell '
                 'recipient gate against every required comparator (income and occupation views; AGG1 beside it), STABLE_POSITIVE '
                 'pair benefit against all six required comparators, and the separate 0.020 target label against NEURAL_RAW and H_LJ')})
    return rows, bench


# ---------------------------------------------------------------- claim ladder (section 7)
MS_HOUSEHOLDS = 5306


def _seed_reserve(ps, seed, min_ratio_recorded):
    """Per-condition records of the frozen reserve rule (ohn.rules.reserve, unchanged) for one seed of a bank-evaluation member,
    rebuilt from its stored values and own SEs with the original caps (MODEL_SELECT: 5306 households); the rebuilt minimum ratio
    must equal the recorded one."""
    from ohn import rules as OR
    u = {'households': MS_HOUSEHOLDS,
         'clauses': [{'task': t, 'reference': r, 'metric': m, 'value': v['value'], 'cap': OR.CAPS[m], 'se': v['se']}
                     for k, v in ps['clauses'].items() for t, r, m in [k.split('|')]],
         'guards': [{'group': g, 'value': v['value'], 'cap': OR.GUARD_CAP, 'se': v['se']} for g, v in ps['guards'].items()]}
    conds = OR.reserve({seed: u})['per_seed'][seed]['conditions']
    mn = min(OR._ratio(v) for v in conds.values())
    if not math.isclose(mn, min_ratio_recorded, rel_tol=0, abs_tol=1e-9):
        raise AssertionError(f'rebuilt reserve differs from the recorded one ({mn} vs {min_ratio_recorded})')
    return conds


def _u(P, did, s):
    st = _ladder(P, did)['MS'].get(str(s))
    return None if st is None else RL.utilization(st['stats'], RL.stat_slack(_baseline(P, did)))


def ladder():
    sel = json.loads((C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').read_text())
    outcome = sel.get('outcome') or sel.get('result', {}).get('outcome')
    formal = RL.claim_ladder({})
    qa = json.loads((C.RESULTS / 'diagnosis' / 'QUESTION_A_LABELS.json').read_text())
    pl = {}
    for P in ('D0', 'D1', 'R0', 'R1'):
        tag = f'{P}|REFIT_PAIR|full|OOF'
        pl[P] = 'PREDICTION_IMPROVED' if tag in qa['prediction_improved_primary'] else 'NOT_IMPROVED'
    evF = _bank_eval('R_FRONTIER')
    mc = {(r['candidate'], r['comparator']): r for r in json.loads((OUT / 'MATCHED_CONTROLS.json').read_text())['rows']}
    gF = _grid('R_FRONTIER')
    # Rung 4 needs, on each R partition, at least two reader-evaluated strengths with realized MODEL_SELECT utilization u <= 1
    # before any frontier interpolation; the condition is checked first and the interpolation is only needed where it holds.
    fr = {P: {str(s): {'u': _u(P, 'REFIT_PAIR', s), 'pair_auc_seed': _auc(gF, f'REFIT_PAIR@s{s}').get(TF.seed_of(P))} for s in STRENGTHS}
          for P in R_PARTS}
    frontier_ok = {}
    for P in R_PARTS:
        n_ok = sum(1 for v in fr[P].values() if v['u'] is not None and v['u'] <= 1.0)
        if n_ok >= 2:
            raise NotImplementedError(f'{P}: {n_ok} strengths with u <= 1; the frontier interpolation of rung 4 must be evaluated')
        frontier_ok[P] = False
        fr[P]['strengths_with_u_le_1'] = n_ok
    desc = {}
    for s in STRENGTHS:
        cand = f'REFIT_PAIR@s{s}'
        e = evF.get(cand)
        util = {}
        from ohn import rules as OR
        for P in R_PARTS:
            seed = TF.seed_of(P)
            ps = e['per_seed'][str(seed)] if e else None
            base = RL.stat_slack(_baseline(P))
            chk = _ladder(P, 'REFIT_PAIR')['CHECK'][str(s)]['stats']
            conds = _seed_reserve(ps, seed, ps['min_ratio']) if ps else None
            u_check = RL.utilization(chk, base)
            util[P] = {'nonzero': True,
                       'ms_reserve_kept': None if conds is None else all(conds[n]['met'] for n in OR.OCC_SIDE),
                       'min_occupation_side_ratio': None if conds is None else min(OR._ratio(conds[n]) for n in OR.OCC_SIDE),
                       'check_no_crossing': u_check <= 1.0, 'check_utilization_u': u_check,
                       'point_pass': None if ps is None else (ps['point_pass_12'] and ps['guard_pass']),
                       'ms_utilization_u': _u(P, 'REFIT_PAIR', s)}
        controls = {'CONT': mc.get((cand, f'REFIT_CONT@s{s}'), {}).get('label'), 'LOC': mc.get((cand, f'REFIT_LOC@s{s}'), {}).get('label'),
                    'COMP': mc.get((cand, f'COMP:REFIT_PAIR@s{s} (R0 only)'), {}).get('label'), 'W': mc.get((cand, 'W'), {}).get('label')}
        bm = next((b for b in json.loads((OUT / 'ORIGINAL_BENCHMARK.json').read_text())['records'] if b['candidate'] == cand), {})
        ev = {'arm': 'PAIR', 'prediction_labels': pl, 'utility': util, 'matched_controls': controls,
              'frontier': {P: frontier_ok[P] for P in R_PARTS},
              'benchmark': {'point_and_gate': bool(bm.get('point_pass_24_4')) and bool(bm.get('old_gate_pass_all')),
                            'superiority_all_required': bool(bm.get('superiority_all_required')),
                            'target_label': bm.get('target_label', 'DEV_GRID_TARGET_NOT_MET')}}
        desc[str(s)] = {'evaluation': ev, 'ladder': RL.claim_ladder(ev)}
    rec = {'schema': 'CTS1-claim-ladder-v1', 'utc': C.now(), 'recipe_selection_outcome': outcome,
           'formal_ladder': formal, 'formal_note': ('the claim ladder applies to the selected development recipe; recipe selection gave '
                                                    f'{outcome}, so no rung is met and the highest claimable rung is 0'),
           'descriptive_REFIT_PAIR_full': desc, 'frontier_utilization': fr,
           'descriptive_note': ('DESCRIPTIVE, NOT A CLAIM: the registered rung conditions evaluated for the coalition-aware direction at '
                                'the reader-evaluated common strengths; rung 4 is recorded as not met because no R partition has two '
                                'reader-evaluated strengths with realized MODEL_SELECT utilization u <= 1 (see frontier_utilization); '
                                'rung 5 inputs are in ORIGINAL_BENCHMARK.json')}
    C.write_json(OUT / 'CLAIM_LADDER.json', rec)
    return rec


def drivers():
    """What drives the frontier (fallback programme): per R partition, arm and reader-evaluated strength, the binding reserve
    condition and minimum ratio of the complete release (its own seed), the realized MODEL_SELECT incremental costs (men's / women's /
    all summed Brier, NLL, accuracy loss) and the mean pair SCORE AUC of the partition's 4 cells."""
    evF, gF = _bank_eval('R_FRONTIER'), _grid('R_FRONTIER')
    rows = []
    for P in R_PARTS:
        seed = TF.seed_of(P)
        for arm in ('PAIR', 'LOC', 'CONT'):
            did = f'REFIT_{arm}'
            ms = _ladder(P, did)['MS']
            for s in (0.0,) + STRENGTHS:
                name = 'W' if s == 0.0 else f'{did}@s{s}'
                e = evF.get(name)
                ps = e['per_seed'][str(seed)] if e else None
                st = ms[str(s)]['stats'] if str(s) in ms else None
                rows.append({'partition': P, 'arm': arm, 'strength': s, 'release': name,
                             'binding_condition': ps['binding'] if ps else None, 'min_ratio': ps['min_ratio'] if ps else None,
                             'point_pass_12_2': (ps['point_pass_12'] and ps['guard_pass']) if ps else None,
                             **({f'ms_{k}': st[k] for k in RL.STATS} if st else {}),
                             'pair_auc_mean_4_cells': _auc(gF, name).get(seed)})
    _write_csv('FRONTIER_DRIVERS', rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('all', 'rules', 'controls', 'frontier', 'ladder', 'drivers'))
    a = ap.parse_args(argv)
    if a.cmd in ('all', 'drivers'):
        rows = drivers()
        print(json.dumps([(r['partition'], r['release'], r['binding_condition'], None if r['min_ratio'] is None else round(r['min_ratio'], 2))
                          for r in rows]))
    if a.cmd in ('all', 'rules'):
        rows, comps = rule_comparison()
        print(json.dumps({'rule_rows': len(rows), 'comparisons': [(c['direction'], c['deployment'], c['rule_a'], c['rule_b'], c['R0'], c['R1'])
                                                                   for c in comps]}))
    if a.cmd in ('all', 'controls'):
        rows = matched_controls()
        print(json.dumps([(r['candidate'], r['comparator'], r['label'], None if r.get('mean') is None else round(r['mean'], 4))
                          for r in rows]))
    if a.cmd in ('all', 'frontier'):
        rows, bench = frontier()
        print(json.dumps({'qualified': [r['release'] for r in rows if r['reserve_qualified']],
                          'bench': [(b['candidate'], b['point_pass_24_4'], b['old_gate_pass_all'], b['superiority_all_required'],
                                     b['target_label']) for b in bench]}))
    if a.cmd in ('all', 'ladder'):
        rec = ladder()
        print(json.dumps({'formal_highest': rec['formal_ladder']['highest_claimable_rung'],
                          'descriptive': {s: v['ladder']['highest_claimable_rung'] for s, v in rec['descriptive_REFIT_PAIR_full'].items()}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

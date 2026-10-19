"""Reader-stage labels, benchmark tables and claim labels of the conditional stage (no label is read; RULES.md sections 9-11).

    python3 -m tsz.claims run

Inputs (all saved records): the reader grids runs/GRID/COND/S<v>/GRID.json (both versions), readers/READER_SEED_CHECK.json,
conditional/COND_UTILITY.json, conditional/SELECTION_V<v>.json, size/SIZE_RESULTS.json, trigger/TRIGGER.json.
Outputs (results/<campaign>/conditional/): CLAIMS.json, MATCHED_CONTROLS.csv (every registered comparison: 8 cells, mean, label),
ORIGINAL_BENCHMARK.csv (candidates against the six point-eligible strong references, reserve reported beside, never merged),
EQUAL_RESERVE.csv (one reserve for every method, Pareto frontier of qualified points).
Every stability label is computed only if the reader-seed check is READER_SEEDS_EFFECTIVE (RULES.md section 9).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

from . import common as C
from . import rules as R

OUT = C.RESULTS / 'conditional'
REFERENCES = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
CANDIDATES = ('LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR', 'W')
REGISTERED = (('LOC_SEL', 'W', 'C3w LOC'), ('LOC_SEL', 'CONT_AT_LOC', 'C3 LOC'), ('PAIR_SEL', 'W', 'C3w PAIR'),
              ('PAIR_SEL', 'CONT_AT_PAIR', 'C3 PAIR'), ('PAIR_SEL', 'LOC_SEL', 'C4 PAIR vs LOC at its own selected strength'),
              ('PAIR_SEL', 'LOC_AT_PAIR', 'descriptive: PAIR vs LOC at the same strength'),
              ('CONT_AT_LOC', 'W', 'descriptive: continuation vs witness'), ('CONT_AT_PAIR', 'W', 'descriptive: continuation vs witness'),
              ('W', 'TR100', 'descriptive: witness vs the equally exposed task-only control'))


def _j(p):
    return json.loads(Path(p).read_text())


def grids():
    g = {}
    for v in C.VERSIONS:
        g[v] = _j(C.RUNS / 'GRID' / 'COND' / f'S{v}' / 'GRID.json')
    return g


def cells_for(g, cand, comp, view):
    """8 registered cells (both versions) of one paired comparison and one view, with their paired bootstrap SEs."""
    vals, ses = {}, {}
    for v in C.VERSIONS:
        p = g[v]['paired'].get(cand, {}).get(comp)
        if p is None:
            return None, None
        vals.update(p[view]['cells']); ses.update(p[view]['cell_se'])
    return vals, ses


def pair_auc(g, name, v):
    cs = [c for c in g[v]['cells'] if '@' not in c]
    return sum(g[v]['cells'][c][name]['pair']['score_auc'] for c in cs) / len(cs)


SD_RECIPES = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
MINIMUM_SET = ('W', 'LOC_SEL', 'PAIR_SEL')
COARSE_REFS = ('B2TWIN', 'RETIRED_B3CTL')


def a_sc1_validity(seed, card):
    """Grid validity under the statistical reviewer's adjudication A-SC1 (ADJUDICATION_SEEDCHECK.md), which replaces only the scope
    of RULES.md section 9 item 2: (i) item 1 (seeds distinct, recorded, not OHN1 / CTS1 seeds); (ii) no equality of a
    seed-dependent recipe on the minimum set W100_v / LOC_SEL / PAIR_SEL; (iii) every other equality explained: (a) another
    seed-dependent recipe differs in the same (wire, view, v, R), (b) the same recipe differs on another view of the same wire in the
    same (v, R), (c) the view's distinct-input count is recorded label-free; (iv) seed-invariant recipes as item 3. Every checked
    (member, view, v, R, recipe) exists (missing_entries empty), so absence from the equality list means 'differs'."""
    from . import rules as R
    eq = [c for c in seed['rule']['ineffective_checks'] if c.get('recipe') in SD_RECIPES]
    eqset = {(c['wire'], c['view'], c['v'], c['R'], c['recipe']) for c in eq}
    sv = [seed['seeds'].get(f'S{v}k{k}') for v in C.VERSIONS for k in (0, 1)]
    item1 = (all(isinstance(x, int) and not isinstance(x, bool) for x in sv) and len(set(sv)) == 4
             and not set(sv) & R.forbidden_seeds())
    complete = not seed.get('missing_entries')
    ii = not [c for c in eq if c['wire'] in MINIMUM_SET]
    unexplained, explained = [], []
    for c in eq:
        a = any((c['wire'], c['view'], c['v'], c['R'], r) not in eqset for r in SD_RECIPES if r != c['recipe'])
        b = any((c['wire'], w, c['v'], c['R'], c['recipe']) not in eqset for w in ('income', 'occupation', 'pair') if w != c['view'])
        n = (card['distinct_input_vectors'].get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
        rec = {**c, 'iii_a_other_recipe_differs': a, 'iii_b_other_view_differs': b, 'iii_c_distinct_input_vectors': n,
               'property': 'SEED_INVARIANT_BY_INPUT'}
        (explained if (a and b and isinstance(n, int)) else unexplained).append(rec)
    effective = bool(item1 and complete and ii and not unexplained)
    return {'adjudication': 'A-SC1 (statistical reviewer, roles/statistical_reviewer/ADJUDICATION_SEEDCHECK.md)',
            'literal_registered': seed['rule']['label'], 'amended_A-SC1': 'READER_SEEDS_EFFECTIVE' if effective else 'READER_SEEDS_NOT_EFFECTIVE',
            'effective': effective, 'item1': item1, 'all_checks_present': complete, 'minimum_set_no_equality': ii,
            'explained_equalities': explained, 'unexplained_equalities': unexplained,
            'nondeterministic_flags': seed['rule']['flags'],
            'disclosure': ('The registered effective-reader-seed check, read literally over all 14 checked bank members, returned '
                           'READER_SEEDS_NOT_EFFECTIVE: ExtraTrees readers (ET5, ET25) are seed-invariant on the income views of B2TWIN '
                           'and RETIRED_B3CTL, which take 6 distinct values. On the registered minimum set (W100_v and every selected '
                           'release) there was no equality. Amendment A-SC1, made after this check\'s outcome and before any '
                           'reader-stability label was computed or read, limited grid invalidation to unexplained equalities. The grid '
                           'passed the amended check. No stability formula changed.')}


def coarse_flag(g, comp, view):
    """COARSE_REFERENCE_INCOME_WIRE (A-SC1 section 4; never gates, never changes a label): income-view recipient gates against
    B2TWIN / RETIRED_B3CTL, and pair-view comparisons against them whose comparator pair winner is an income-view reader in some
    cell."""
    if comp not in COARSE_REFS:
        return False
    if view == 'income':
        return True
    if view == 'pair':
        return any(str(g[v]['cells'][c][comp]['pair']['winner']).startswith('IGNORE_occupation__')
                   for v in C.VERSIONS for c in g[v]['cells'] if '@' not in c)
    return False


def run():
    out_json = OUT / 'CLAIMS.json'
    if out_json.exists():
        raise PermissionError('CLAIMS.json exists (write-once)')
    g = grids()
    seed = _j(C.RESULTS / 'readers' / 'READER_SEED_CHECK.json')
    card = _j(C.RESULTS / 'readers' / 'WIRE_CARDINALITY.json')
    validity = a_sc1_validity(seed, card)
    seeds_ok = validity['effective']
    cu = _j(OUT / 'COND_UTILITY.json')
    size = _j(C.RESULTS / 'size' / 'SIZE_RESULTS.json')
    trig = _j(C.RESULTS / 'trigger' / 'TRIGGER.json')
    sel = {v: _j(OUT / f'SELECTION_V{v}.json')['selected'] for v in C.VERSIONS}
    # 1. every registered and descriptive comparison (pair view) + recipient differences (single views)
    comps = {}
    rows = []
    pairs = list(REGISTERED) + [(c, r, f'vs reference {r}') for c in ('LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'W')
                                for r in REFERENCES]
    for cand, comp, what in pairs:
        vals, ses = cells_for(g, cand, comp, 'pair')
        if vals is None:
            comps[f'{cand}|{comp}'] = {'what': what, 'label': 'NOT_IN_GRID'}
            continue
        stab = R.stability_label(vals, ses) if seeds_ok else {'label': 'NOT_COMPUTED_READER_SEEDS_NOT_EFFECTIVE'}
        gates = {}
        for view in ('income', 'occupation'):
            rv, _ = cells_for(g, cand, comp, view)
            gates[view] = R.recipient_gate(rv)
        flags = {view: coarse_flag(g, comp, view) for view in ('income', 'pair')}
        rec = {'what': what, 'pair': stab, 'recipient_gate': gates,
               'flags': ['COARSE_REFERENCE_INCOME_WIRE'] if any(flags.values()) else [], 'flag_views': flags}
        if comp in TARGET_REFS:
            rec['target'] = R.target_label(vals, ses) if seeds_ok else {'label': 'NOT_COMPUTED'}
        comps[f'{cand}|{comp}'] = rec
        rows.append({'candidate': cand, 'comparator': comp, 'what': what, 'label': stab.get('label'), 'mean': stab.get('mean'),
                     'min': stab.get('min'), 'max': stab.get('max'), 'cells_at_or_below_0': stab.get('cells_at_or_below'),
                     'reader_seed_qualifier': stab.get('reader_seed_qualifier'),
                     'income_gate': gates['income'].get('label'), 'occupation_gate': gates['occupation'].get('label'),
                     **{f'cell_{c}': (stab.get('cells') or {}).get(c) for c in R.CELLS}})
    # 2. per-arm inputs of the claim labels
    rel = cu['releases']

    def util(name):
        return {v: {'point_pass': rel[name][str(v)]['point_pass'], 'occ_reserve_met': rel[name][str(v)]['occ_reserve_met']}
                for v in C.VERSIONS}

    def lab(key):
        return (comps.get(key) or {}).get('pair', {}).get('label')

    def competitive(name):
        point_all = all(rel[name][str(v)]['point_pass'] for v in C.VERSIONS)
        gate_all = all((comps.get(f'{name}|{r}') or {}).get('recipient_gate', {}).get(view, {}).get('pass') is True
                       for r in REFERENCES for view in ('income', 'occupation'))
        stable_all = all(lab(f'{name}|{r}') == 'STABLE_POSITIVE' for r in REFERENCES)
        return {'point_all': point_all, 'recipient_gate_all': gate_all, 'stable_vs_all_required': stable_all}

    def target(name):
        ls = [(comps.get(f'{name}|{r}') or {}).get('target', {}).get('label') for r in TARGET_REFS]
        order = ('DEV_GRID_TARGET_NOT_MET', 'DEV_GRID_TARGET_POINT_ONLY', 'DEV_GRID_TARGET_STABLE')
        return min((x if x in order else 'DEV_GRID_TARGET_NOT_MET' for x in ls), key=order.index)
    arms = {'LOC': {'selected': {v: sel[v]['LOC'] for v in C.VERSIONS}, 'utility': util('LOC_SEL'), 'vs_CONT': lab('LOC_SEL|CONT_AT_LOC'),
                    'vs_W': lab('LOC_SEL|W'), 'competitive': competitive('LOC_SEL'), 'target': target('LOC_SEL')},
            'PAIR': {'selected': {v: sel[v]['PAIR'] for v in C.VERSIONS}, 'utility': util('PAIR_SEL'),
                     'vs_CONT': lab('PAIR_SEL|CONT_AT_PAIR'), 'vs_W': lab('PAIR_SEL|W'), 'vs_LOC': lab('PAIR_SEL|LOC_SEL'),
                     'competitive': competitive('PAIR_SEL'), 'target': target('PAIR_SEL')}}
    ev = {'trigger_fired': trig['trigger']['fires'], 'size_explanation': size['interpretation']['size_explanation'], 'arms': arms}
    claims = R.claim_labels(ev)
    # 3. original benchmark table (references by point feasibility; reserve reported beside, never merged)
    bench = []
    for name in ('LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'W'):
        for r in REFERENCES:
            c = comps.get(f'{name}|{r}') or {}
            bench.append({'candidate': name, 'reference': r,
                          'candidate_point_both_versions': all(rel[name][str(v)]['point_pass'] for v in C.VERSIONS),
                          'candidate_occ_reserve_both_versions': all(rel[name][str(v)]['occ_reserve_met'] for v in C.VERSIONS),
                          'candidate_whole_reserve_both_versions': all(rel[name][str(v)]['whole_release_reserve_met'] for v in C.VERSIONS),
                          'reference_point_both_versions': all(cu['references'][r][str(v)]['point_pass_12_plus_2'] for v in C.VERSIONS),
                          'reference_whole_reserve_both_versions': all(cu['references'][r][str(v)]['whole_release_reserve_met']
                                                                       for v in C.VERSIONS),
                          'pair_benefit_label': (c.get('pair') or {}).get('label'), 'pair_benefit_mean': (c.get('pair') or {}).get('mean'),
                          'income_gate': (c.get('recipient_gate') or {}).get('income', {}).get('label'),
                          'occupation_gate': (c.get('recipient_gate') or {}).get('occupation', {}).get('label'),
                          'target_0020': (c.get('target') or {}).get('label')})
    # 4. equal-reserve comparison (whole-release 28 conditions and, separately, the 16 occupation-side conditions)
    sres = size['occupation_reserve']

    def ctl_ratio(key, v, whole):
        o = sres[f'{key}|S100|V{v}']
        return min(o['min_ratio'], o['income']['min_income_ratio']) if whole else o['min_ratio']
    eq = {}
    for whole in (True, False):
        pts = []
        for name in ('LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR', 'W'):
            kind = 'control' if name in ('W', 'CONT_AT_LOC', 'CONT_AT_PAIR') else 'candidate'
            mr = {v: (rel[name][str(v)]['min_whole_ratio'] if whole else rel[name][str(v)]['min_occ_ratio']) for v in C.VERSIONS}
            pts.append({'id': name, 'kind': kind, 'min_ratio': mr, 'pair_auc': {v: pair_auc(g, name, v) for v in C.VERSIONS}})
        for name, key in (('TR100', 'TR'), ('TT100', 'TT')):
            pts.append({'id': name, 'kind': 'control', 'min_ratio': {v: ctl_ratio(key, v, whole) for v in C.VERSIONS},
                        'pair_auc': {v: pair_auc(g, name, v) for v in C.VERSIONS}})
        for r in REFERENCES:
            mr = {v: (cu['references'][r][str(v)]['min_whole_ratio'] if whole else cu['references'][r][str(v)]['min_occ_ratio'])
                  for v in C.VERSIONS}
            pts.append({'id': r, 'kind': 'reference', 'min_ratio': mr, 'pair_auc': {v: pair_auc(g, r, v) for v in C.VERSIONS}})
        eq['whole_release_28' if whole else 'occupation_side_16'] = R.equal_reserve_table(pts)
    rec = {'schema': 'TSZ1-claims-v1', 'utc': C.now(), 'reader_seed_check': validity, 'selected': sel,
           'comparisons': comps, 'claim_inputs': ev, 'claims': claims, 'original_benchmark': bench, 'equal_reserve': eq,
           'rules_sha256': R.load_rules()['_sha256'],
           'scope': ['declared development readers on adaptively used MODEL_SELECT; not arbitrary-reader protection',
                     'reader recipes and overlapping splits are not independent datasets',
                     'occupation-only optimization is not a new two-output joint method']}
    C.write_json_x(out_json, rec)
    for fname, table in (('MATCHED_CONTROLS.csv', rows), ('ORIGINAL_BENCHMARK.csv', bench)):
        if table:
            with open(OUT / fname, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    with open(OUT / 'EQUAL_RESERVE.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['reserve', 'id', 'kind', 'min_ratio_both_versions', 'pair_auc_mean', 'qualified', 'on_frontier'])
        for k, t in eq.items():
            for p in t['points']:
                w.writerow([k, p['id'], p['kind'], p['min_ratio'], p['pair_auc_mean'], p['qualified'], p['id'] in t['frontier']])
    print(json.dumps({'outcome': claims.get('outcome'), 'arms': {a: {k: (x['label'] if isinstance(x, dict) and 'label' in x else x)
                                                                    for k, x in claims['arms'][a].items() if k in ('C2', 'C2p', 'C3', 'C3w', 'C4', 'C5', 'qualifies')}
                                                                for a in claims.get('arms', {})},
                      'reader_seed_check': validity['amended_A-SC1'], 'literal': validity['literal_registered']}, default=str))
    return rec


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise SystemExit(__doc__)
    run()

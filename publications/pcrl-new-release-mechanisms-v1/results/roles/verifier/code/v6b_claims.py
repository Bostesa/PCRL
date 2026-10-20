"""V6b claims and selection replay from the RECORDED SOURCES (not from the coordinator's in-memory ledger): the committed claims plan,
the evaluation records (EVAL_MS*.json; replayed separately by v2_utility), the grid records (GRID.json 'paired' and 'cells';
replayed separately by v4_grid), the release records (fitted-component counts), the reference wires, and this campaign's reader
cache (k0 / k1 score equalities for the A-SC1 grid-validity rule). Everything the rules need is rebuilt with the verifier's own
code: Monte Carlo grid cells (mean over the replicate members, household SE = RMS of their SEs, MC SE = SD / sqrt(D)), frontier
records per declared path, the joint-design record, grid validity, claims G1-G7 with the G2 conjunction over alternatives, the
multi-path G3 rule, the ledger, and the selection (v6_select). Compared with CLAIMS.json (records and selection).

Labels: MODEL_SELECT SEX only (to locate the cache entries by the registered key), via nrm.common.load(operation 'verify') called
from THIS script (logged).

    python3 .../v6b_claims.py --tag V1 --plan results/<campaign>/claims/PLAN_X.json --bank NRM1_MAIN --claims results/<campaign>/claims/X/CLAIMS.json
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
import v4_grid as G  # noqa: E402

SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SEED_INVARIANT = ('OUT_LR', 'OUT_HGB', 'LOG_LR', 'DEC_LR', 'CONF_HGB', 'LOOKUP', 'HGB15', 'HGB31', 'CATI01', 'CATI1')
FORBIDDEN = {V.MODEL_SEED_BASE + off + 10 * v + k for off in (300, 500) for v in (0, 1) for k in range(10)}
REF_SPEC = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
            'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE'}


def load_sex(purpose):
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    Sx = C.load('MODEL_SELECT', ('features', 'sex'), 'verify', purpose)
    return np.asarray(Sx['sex']), np.asarray(Sx['unit'])


def merged_grid(bank):
    out = {'paired': {}, 'cells': {}, 'members': {}, 'spec': None}
    for v in (0, 1):
        g = json.loads((V.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())
        out['spec'] = g['spec']
        out['members'][v] = g.get('members') or {}
        out['cells'].update(g.get('cells') or {})
        for cand, comps in (g.get('paired') or {}).items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells']); d['cell_se'].update(e['cell_se'])
    return out


ENG2 = {'on': False}      # wave-2 definitions (735cde202): costs in time order, every path member, failing alternative, F1 on every
                          # declared path, Monte Carlo pooled pair AUC in the reserve frontier


def eval_base(d):
    """The evaluation record governing a configuration (nrm.claims docstring, 87810b4da): utility and wires from the most recent
    EVAL_MS*.json carrying wire_sha256_eval for both versions; paired costs merged from all records in name order (wave 2: in
    time order, later records win)."""
    recs = [(p, json.loads(p.read_text())) for p in sorted(d.glob('EVAL_MS*.json'))]
    withw = [r for _, r in recs if all('wire_sha256_eval' in (r['versions'].get(str(v)) or {}) for v in (0, 1))]
    if not withw:
        raise KeyError(f'{d}: no evaluation record carries wire_sha256_eval')
    base = max(withw, key=lambda r: r.get('utc', ''))
    costs = {v: {} for v in (0, 1)}
    if ENG2['on']:
        recs = sorted(recs, key=lambda pr: pr[1].get('utc', ''))
    for _, r in recs:
        for v in (0, 1):
            for spec, c in ((r['versions'].get(str(v)) or {}).get('paired_costs') or {}).items():
                costs[v][spec] = c['costs']
    return base, costs


def mc_cell(vals, ses):
    d = len(vals)
    m = sum(vals) / d
    return {'value': m, 'se': math.sqrt(sum(s * s for s in ses) / d),
            'se_mc': math.sqrt(sum((x - m) ** 2 for x in vals) / (d - 1) / d) if d > 1 else None}


def entry(grid, cand, comp, reps):
    base = grid['paired'].get(cand, {}).get(comp)
    if base is None:
        return None
    rc, rp = reps.get(cand, []), reps.get(comp, [])
    if not rc and not rp:
        return base
    out = {}
    n = max(len(rc), len(rp))
    for view, e in base.items():
        cells, ses, mcs = {}, {}, {}
        for cell in e['cells']:
            vals, ss = [e['cells'][cell]], [e['cell_se'][cell]]
            for i in range(n):
                c2 = rc[i] if i < len(rc) else cand
                k2 = rp[i] if i < len(rp) else comp
                x = grid['paired'].get(c2, {}).get(k2, {}).get(view)
                if x is None or cell not in x['cells']:
                    return {'error': f'replicate pairing {c2} vs {k2} missing ({view} {cell})'}
                vals.append(x['cells'][cell]); ss.append(x['cell_se'][cell])
            m = mc_cell(vals, ss)
            cells[cell], ses[cell], mcs[cell] = m['value'], m['se'], m['se_mc']
        out[view] = {'cells': cells, 'cell_se': ses, 'cell_se_mc': mcs}
    return out


def seed_checks(grid, sex, unit):
    """k0 vs k1 bitwise equality per (member, view, version, repetition, recipe) from the cached SCORE scores."""
    sx = V.arr_sha(sex)
    spec = grid['spec']
    checks, distinct = [], {}
    for v in (0, 1):
        for name, m in spec['members'].items():
            rel, prov = G.member_arrays(m, v)
            rel = {t: np.ascontiguousarray(np.asarray(rel[t], dtype=np.float64)) for t in V.TASKS}
            rules = prov.get('draw_rule') if prov.get('randomized') else None
            i0 = {t: (rel[t][G.draw_index(rules, 'R1')] if rel[t].ndim == 3 else rel[t]) for t in V.TASKS}
            distinct[f'{name}|V{v}'] = {'income': V.distinct_rows(i0['income']), 'occupation': V.distinct_rows(i0['occupation']),
                                        'pair': V.distinct_rows(np.column_stack((i0['income'], i0['occupation'])))}
            for rep in V.REG_REPS:
                if f'S{v}{rep}k0' not in grid['cells'] or name not in grid['cells'][f'S{v}{rep}k0']:
                    continue
                di = G.draw_index(rules, rep) if rules else None
                inc = rel['income'][di] if rel['income'].ndim == 3 else rel['income']
                occ = rel['occupation'][di] if rel['occupation'].ndim == 3 else rel['occupation']
                n_sc = int(V.reader_roles(unit, rep)['SCORE'].sum())
                for view in V.VIEWS:
                    e0, _ = G.read_cache(view, inc, occ, rep, V.reader_seed(v, 0), sx, n_sc)
                    e1, _ = G.read_cache(view, inc, occ, rep, V.reader_seed(v, 1), sx, n_sc)
                    if e0 is None or e1 is None:
                        continue
                    for r in SEED_DEPENDENT + SEED_INVARIANT:
                        checks.append({'recipe': r, 'view': view, 'wire': name, 'v': v, 'R': rep,
                                       'equal_k0_k1': bool(np.array_equal(e0['scores'][r], e1['scores'][r]))})
    return checks, distinct


def validity(seeds, checks, minimum_set, distinct):
    errors = []
    keys = [f'S{v}k{k}' for v in (0, 1) for k in (0, 1)]
    sv = [seeds.get(k) for k in keys]
    if any(not isinstance(x, int) for x in sv) or len(set(sv)) != len(sv):
        errors.append('seeds')
    elif set(sv) & FORBIDDEN:
        errors.append('forbidden seeds')
    seen = {c['recipe'] for c in checks}
    if any(r not in seen for r in SEED_DEPENDENT):
        errors.append('seed-dependent recipes not checked')
    eq = [c for c in checks if c['recipe'] in SEED_DEPENDENT and c['equal_k0_k1'] is not False]
    diff = {(c['wire'], c['view'], c['v'], c['R'], c['recipe']) for c in checks if c['recipe'] in SEED_DEPENDENT and c['equal_k0_k1'] is False}
    on_min = [c for c in eq if c['wire'] in set(minimum_set)]
    if on_min:
        errors.append(f'{len(on_min)} seed-dependent equalities on the minimum set')
    unexpl, listing = [], []
    for c in eq:
        a = any((c['wire'], c['view'], c['v'], c['R'], r) in diff for r in SEED_DEPENDENT if r != c['recipe'])
        b = any((c['wire'], w, c['v'], c['R'], c['recipe']) in diff for w in V.VIEWS if w != c['view'])
        n = (distinct.get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
        listing.append({**c, 'other_seed_recipe_differs': a, 'same_recipe_differs_other_view': b, 'distinct_inputs': n,
                        'on_minimum_set': c['wire'] in set(minimum_set)})
        if c['wire'] in set(minimum_set):
            continue
        if not (a and b and isinstance(n, int)):
            unexpl.append(c)
    if unexpl:
        errors.append(f'{len(unexpl)} unexplained seed-dependent equalities')
    nondet = [c for c in checks if c['recipe'] in SEED_INVARIANT and c['equal_k0_k1'] is False]
    return {'effective': not errors, 'errors': errors, 'n_equal_seed_dependent': len(eq), 'n_on_minimum_set': len(on_min),
            'n_unexplained': len(unexpl), 'n_explained': len(eq) - len(on_min) - len(unexpl),
            'nondeterministic_recipe': bool(nondet), 'n_nondeterministic_checks': len(nondet), 'n_checks': len(checks),
            'equalities': listing}


FAMILY_ROWS = ('FB2_BASE', 'FB2_SEX_S3', 'FB2_SEX_S5', 'FB2_MINF_S3', 'FB2_MINF_S5', 'CTL_REM_0p50', 'CTL_REM_1p00', 'FB_TWIN',
               'FB_SEX_S2', 'FB_MINF_S2', 'PRC_TWIN')
DOM = 'DOMINATES_CONTROL_FRONTIER'


def reserve_frontier(plan, grid, reps, mine, refs_doc):
    """RULES.md section 4: points = the final plan's members (TABLES rows) and the nine references; qualified = reserve with the MC
    term on both versions; pair AUC = mean of the 8 registered cells; Pareto frontier of qualified points (lower pair AUC, higher
    minimum ratio); a qualified candidate improves iff AGG_POSITIVE against every other qualified control and reference."""
    def pair_auc(m):
        reps_m = [m, f'{m}__d2', f'{m}__d4']
        if ENG2['on'] and all(r in grid['cells'].get(S.REG[0], {}) for r in reps_m[1:]):     # Monte Carlo pooled (ENG2-08)
            xs = [sum(grid['cells'][c][r]['pair']['score_auc'] for r in reps_m) / 3 for c in S.REG]
            return sum(xs) / len(xs)
        xs = [grid['cells'][c][m]['pair']['score_auc'] for c in S.REG if m in grid['cells'].get(c, {})]
        return sum(xs) / len(xs) if len(xs) == len(S.REG) else None
    pts = []
    for m in FAMILY_ROWS + S.BENCH:
        if m in S.BENCH:
            res = S.my_reserve({v: refs_doc['references'][REF_SPEC[m]][str(v)]['utility'] for v in (0, 1)})
            kind = 'reference'
        else:
            if m not in mine:
                continue
            res = mine[m]['reserve']
            kind = 'candidate' if plan['configs'][m].get('kind') == 'candidate' else 'control'
        mr = res.get('min_by_version') or {}
        pts.append({'id': m, 'kind': kind, 'reserve_qualified': bool(res.get('qualified')), 'pair_auc': pair_auc(m),
                    'min_ratio': min(mr.values()) if mr else None})
    q = [p for p in pts if p['reserve_qualified'] and p['pair_auc'] is not None]

    def r_of(p):
        return p['min_ratio'] if p['min_ratio'] is not None and math.isfinite(p['min_ratio']) else 0.0
    for p in pts:
        p['on_reserve_pareto_frontier'] = p in q and not any(o['pair_auc'] <= p['pair_auc'] and r_of(o) >= r_of(p) and
                                                              (o['pair_auc'] < p['pair_auc'] or r_of(o) > r_of(p)) for o in q)
        if p['kind'] == 'candidate' and p['reserve_qualified']:
            others = [o['id'] for o in q if o['id'] != p['id'] and o['kind'] in ('control', 'reference')]
            labs = {}
            for o in others:
                e = entry(grid, p['id'], o, reps)
                labs[o] = S.my_benefit((e or {}).get('pair'))['label'] if e and 'error' not in e else 'INCOMPLETE'
            p['paired_vs_qualified'] = labs
            p['label'] = 'RESERVE_FRONTIER_IMPROVEMENT' if others and all(x == 'AGG_POSITIVE' for x in labs.values()) \
                else 'NO_RESERVE_FRONTIER_IMPROVEMENT'
    return pts


def compare_frontier(mine_pts, rec):
    mm = []
    theirs = {p['id']: p for p in rec.get('points', [])}
    if set(theirs) != {p['id'] for p in mine_pts}:
        mm.append({'what': 'reserve frontier point set', 'mine': sorted(p['id'] for p in mine_pts), 'recorded': sorted(theirs)})
    for p in mine_pts:
        t = theirs.get(p['id'])
        if t is None:
            continue
        for k in ('reserve_qualified', 'on_reserve_pareto_frontier', 'label'):
            if p.get(k) != t.get(k):
                mm.append({'what': f"{p['id']} {k}", 'mine': p.get(k), 'recorded': t.get(k)})
        for k, tol in (('min_ratio', 1e-9), ('pair_auc', 1e-12)):
            a_, b_ = p.get(k), t.get(k)
            if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - float(b_)) > tol):
                mm.append({'what': f"{p['id']} {k}", 'mine': a_, 'recorded': b_})
        if (p.get('paired_vs_qualified') or {}) != (t.get('paired_vs_qualified') or {}):
            mm.append({'what': f"{p['id']} paired_vs_qualified", 'mine': p.get('paired_vs_qualified'), 'recorded': t.get('paired_vs_qualified')})
    return mm


def decisive(kind, path, mine, grid, reps):
    """Own replay of nrm.decisive rules from my own claims replay; returns (mine, mismatches)."""
    rec = json.loads(Path(path).read_text())
    mm = []
    if kind == 'f1':
        def verdict_of(labels):
            if not labels or any(x in ('INCOMPLETE', 'MISSING', None) for x in labels):
                return 'NOT_READ_INCOMPLETE'
            return 'F1_DOES_NOT_HOLD' if all(x == DOM for x in labels) else 'F1_HOLDS'
        lab = {j: ((mine.get(f'FB2_SEX_{j}') or {}).get('frontier_per_path', {}).get('MINF') or {}).get('label', 'MISSING')
               for j in ('S3', 'S5')}
        if ENG2['on']:              # every declared path, with the MINF-path reading beside (ENG2-04)
            paths = {j: {p: f.get('label') for p, f in ((mine.get(f'FB2_SEX_{j}') or {}).get('frontier_per_path') or {}).items()}
                     for j in ('S3', 'S5')}
            verdict = verdict_of([x for j in paths for x in paths[j].values()] + [lab[j] for j in lab if lab[j] == 'MISSING'])
            out = {'path_labels': paths, 'minf_path_labels': lab, 'F1_minf_path_reading': verdict_of(list(lab.values())), 'F1': verdict}
            keys = ('F1', 'minf_path_labels', 'path_labels', 'F1_minf_path_reading')
        else:
            verdict = verdict_of(list(lab.values()))
            out = {'minf_path_labels': lab, 'F1': verdict}
            keys = ('F1', 'minf_path_labels')
        if any(rec.get(k) != out[k] for k in keys):
            mm.append({'what': 'F1', 'mine': out, 'recorded': {k: rec.get(k) for k in keys}})
        return out, mm
    per = {}
    for j in ('S5', 'S3'):
        sex = f'FB2_SEX_{j}'
        g1 = (mine.get(sex) or {}).get('summary', {}).get('G1', 'NOT_IN_CLAIMS')
        if kind == 'mean':
            paths = {p: f['label'] for p, f in ((mine.get(sex) or {}).get('frontier_per_path') or {}).items()}
            missing = [p for p in ('MINF', 'CTL') if p not in paths]
            inc = [p for p, x in paths.items() if x == 'INCOMPLETE']
            per[j] = {'G1': g1, 'paths': paths, 'qualifies': not missing and not inc and g1 == 'TASK_FEASIBLE' and all(paths[p] == DOM for p in ('MINF', 'CTL'))}
            t = rec['per_strength'][j]
            if t.get('paths') != paths or t.get('G1') != g1 or bool(t.get('qualifies')) != per[j]['qualifies']:
                mm.append({'what': f'MEAN {j}', 'mine': per[j], 'recorded': {k: t.get(k) for k in ('G1', 'paths', 'qualifies')}})
        else:                                                      # fb2p: G1 and AGG_POSITIVE vs the same-strength MINF (MC cells)
            e = entry(grid, sex, f'FB2_MINF_{j}', reps)
            b = S.my_benefit((e or {}).get('pair'))
            ag = b['agg']
            scr = {'label': b['label'], 'mean': ag['tot']['mean'] if ag else None, 'se_agg': ag['tot']['se'] if ag else None,
                   'per_version_means': {str(v): ag['per'][v]['mean'] for v in (0, 1)} if ag else None}
            per[j] = {'G1': g1, 'benefit': scr, 'qualifies': g1 == 'TASK_FEASIBLE' and b['label'] == 'AGG_POSITIVE'}
            t = rec['per_strength'][j]
            tb = t['benefit_vs_same_strength_minf']
            bad = t.get('G1') != g1 or tb['label'] != scr['label'] or bool(t.get('qualifies')) != per[j]['qualifies'] or \
                abs(float(tb['mean']) - scr['mean']) > 1e-12 or abs(float(tb['se_agg']) - scr['se_agg']) > 1e-12 or \
                any(abs(float(tb['per_version_means'][v]) - scr['per_version_means'][v]) > 1e-12 for v in ('0', '1'))
            if bad:
                mm.append({'what': f'FB2P {j}', 'mine': per[j], 'recorded': t})
    chosen = next((j for j in ('S5', 'S3') if per[j]['qualifies']), None)
    if rec.get('chosen_strength') != chosen:
        mm.append({'what': f'{kind} chosen strength', 'mine': chosen, 'recorded': rec.get('chosen_strength')})
    return {'per_strength': per, 'chosen_strength': chosen}, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--plan', required=True); ap.add_argument('--bank', required=True)
    ap.add_argument('--claims', required=True); ap.add_argument('--tables', default=None)
    ap.add_argument('--decisive', action='append', default=[], help='KIND=PATH with KIND in fb2p, f1, mean')
    ap.add_argument('--eng2', action='store_true', help='wave-2 definitions of nrm.claims / nrm.decisive / nrm.tables (735cde202)')
    a = ap.parse_args()
    ENG2['on'] = bool(a.eng2)
    plan = json.loads(Path(a.plan).read_text())
    claims_doc = json.loads(Path(a.claims).read_text())
    grid = merged_grid(a.bank)
    manifest = json.loads((V.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': manifest['reader_seeds'][f'V{v}k{k}'] for v in (0, 1) for k in (0, 1)}
    sex, unit = load_sex(f'verify: independent claims and selection replay of {a.bank} (cache k0/k1 equalities need the SEX hash)')
    checks, distinct = seed_checks(grid, sex, unit)
    reps = {m: c.get('replicates', []) for m, c in plan['configs'].items()}
    rw = {}
    for p in plan['references_eval']:
        r = json.loads((V.ROOT / p).read_text())
        for name, spec in REF_SPEC.items():
            if spec in r['references']:
                rw[name] = {v: r['references'][spec][str(v)]['wire_sha256'] for v in (0, 1)}
    ledger, mine, mm = [], {}, []
    for entry_no, (member, cfg) in enumerate(plan['configs'].items()):
        d = V.RUNS / cfg['family'] / cfg['config']
        base, costs = eval_base(d)
        evals = {v: base['versions'][str(v)]['utility'] for v in (0, 1)}
        comps = list(S.BENCH) + [x for x in [cfg.get('twin')] + list(cfg.get('alternatives') or []) if x] + \
            [q for path in (cfg.get('paths') or {}).values() for q in path]
        gentries = {}
        for comp in dict.fromkeys(comps):
            e = entry(grid, member, comp, reps)
            if e is not None and 'error' not in e:
                gentries[comp] = e
            elif e is not None:
                mm.append({'what': f'{member} vs {comp}: {e["error"]}'})
        minimum = [member] + [x for x in [cfg.get('twin')] + list(cfg.get('alternatives') or []) if x]
        gv = validity(seeds, checks, minimum, distinct)
        fr_all = {}
        for pid, path in (cfg.get('paths') or {}).items():
            pts = []
            for q in path:
                pa = [grid['cells'][c][q]['pair']['score_auc'] for c in S.REG if q in grid['cells'].get(c, {})]
                pts.append({'id': q, 'pair_auc': sum(pa) / len(pa) if pa else None})
            spec_of = {q: f"{plan['configs'][q]['family']}/{plan['configs'][q]['config']}" if q in plan['configs'] else q for q in path}
            if ENG2['on']:          # every declared path member: a missing paired cost gives INCOMPLETE (ENG2-01)
                point = {'costs_vs': {q: {v: costs[v][spec_of[q]] for v in (0, 1) if costs[v].get(spec_of[q])} for q in path},
                         'grid_vs': {q: gentries.get(q) for q in path if gentries.get(q)}}
                fr_all[pid] = S.my_frontier(point, pts)
            else:
                point = {'costs_vs': {q: {v: costs[v].get(spec_of[q]) for v in (0, 1)} for q in path
                                      if all(costs[v].get(spec_of[q]) for v in (0, 1))},
                         'grid_vs': {q: gentries.get(q) for q in path if gentries.get(q)}}
                fr_all[pid] = S.my_frontier(point, [p for p in pts if p['id'] in point['costs_vs']])
        fr_pass = bool(fr_all) and all(f['label'] == 'DOMINATES_CONTROL_FRONTIER' for f in fr_all.values())
        fr_main = next((f for f in fr_all.values() if f['label'] != 'DOMINATES_CONTROL_FRONTIER'), None) or \
            (next(iter(fr_all.values())) if fr_all else None)
        joint = None
        if cfg.get('twin') in plan['configs']:
            tw = plan['configs'][cfg['twin']]
            tb, _ = eval_base(V.RUNS / tw['family'] / tw['config'])
            changed = [t for t in V.TASKS if any(base['versions'][str(v)]['wire_sha256_eval'][t] != tb['versions'][str(v)]['wire_sha256_eval'][t]
                                                 for v in (0, 1))]
            joint = {'outputs_changed': changed, 'jointly_designed': bool(cfg.get('outputs_jointly_designed'))}
        per_alt = {}
        for alt in (cfg.get('alternatives') or [None]):
            rec = {'evals': evals, 'grid': gentries, 'matched': {'twin': cfg.get('twin'), 'alternative': alt}, 'joint': joint,
                   'novelty': cfg.get('novelty', 'NOT_CLAIMED'), 'grid_valid': gv['effective']}
            per_alt[alt or 'NONE'] = S.my_claims(rec, frontier_override=fr_main or {})
        cl = next(iter(per_alt.values()))
        g2 = 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' if all(x['G2'] == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' for x in per_alt.values()) \
            else next((x['G2'] for x in per_alt.values() if x['G2'] != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS'), 'NO_MATCHED_CONTROLS')
        summ = {k: cl[k] for k in ('G1', 'G3', 'G4', 'G5', 'G6', 'G7')}
        summ['G2'] = g2
        summ['G3_all_paths'] = {pid: f['label'] for pid, f in fr_all.items()}
        if not fr_pass and summ['G3'] == 'STABLE_FRONTIER_IMPROVEMENT':
            summ['G3'] = 'NOT_DOMINATING_EVERY_DECLARED_PATH'
        nfit = {}
        for v in (0, 1):
            rj = json.loads((d / f'V{v}' / 'RELEASE.json').read_text())
            nfit[v] = int(rj.get('n_model_fits', 0)) + int(rj.get('n_calibrator_fits', 0)) + int(rj.get('n_critic_fits', 0))
        row = {'config_id': member, 'family': cfg['family'], 'kind': cfg.get('kind'), 'entry': entry_no, 'n_fitted_components': nfit,
               'wire_sha256_eval': {v: base['versions'][str(v)]['wire_sha256_eval'] for v in (0, 1)},
               'wire_sha256_grid': {v: (grid['members'][v].get(member) or {}).get('wire_sha256') for v in (0, 1)},
               'evals': evals, 'grid': gentries, 'matched': {'twin': cfg.get('twin'), 'alternative': (cfg.get('alternatives') or [None])[0]},
               'frontier': fr_main, 'joint': joint, 'novelty': cfg.get('novelty', 'NOT_CLAIMED'), 'grid_valid': gv['effective']}
        if len(cfg.get('alternatives') or []) > 1 and g2 != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS':
            failing = next(a_ for a_, x in per_alt.items() if x['G2'] != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS')
            row['matched'] = {'twin': cfg.get('twin'), 'alternative': failing if ENG2['on'] else None}   # ENG2-02
        ledger.append(row)
        mine[member] = {'reserve': cl['reserve'], 'summary': summ, 'grid_validity': gv, 'reserve_qualified': cl['reserve']['qualified'], 'score': cl['score'],
                        'tier': cl['tier'], 'frontier_per_path': fr_all}
        rec = (claims_doc.get('records') or {}).get(member)
        if rec is None:
            mm.append({'what': f'{member} missing in CLAIMS.json'}); continue
        for k, x in summ.items():
            if rec['summary'].get(k) != x:
                mm.append({'what': f'{member} {k}', 'mine': x, 'recorded': rec['summary'].get(k)})
        rgv = rec.get('grid_validity') or {}
        if bool(rgv.get('effective')) != gv['effective'] or list(rgv.get('errors') or []) != gv['errors'] or \
                (rgv.get('explained_equalities') is not None and int(rgv['explained_equalities']) != gv['n_explained']):
            mm.append({'what': f'{member} grid validity', 'mine': {k: gv[k] for k in ('effective', 'errors', 'n_explained')},
                       'recorded': {k: rgv.get(k) for k in ('effective', 'errors', 'explained_equalities')}})
        if rec.get('grid_validity_checks') is not None and int(rec['grid_validity_checks']) != gv['n_checks']:
            mm.append({'what': f'{member} number of k0/k1 checks', 'mine': gv['n_checks'], 'recorded': rec['grid_validity_checks']})
        if bool((rec.get('reserve') or {}).get('qualified')) != bool(cl['reserve']['qualified']):
            mm.append({'what': f'{member} reserve qualified', 'mine': cl['reserve']['qualified'], 'recorded': (rec.get('reserve') or {}).get('qualified')})
    sel = S.my_select(ledger, rw) if all(r in rw for r in S.BENCH) else {'outcome': 'NOT_COMPUTED_REFERENCE_WIRES_MISSING', 'records': [], 'near': []}
    mm += S.compare_selection(sel, claims_doc.get('selection') or {})
    gv_short = {m: {k: x['grid_validity'][k] for k in x['grid_validity'] if k != 'equalities'} for m, x in mine.items()}
    any_m = next(iter(mine.values()))['grid_validity']['equalities'] if mine else []
    eq_list = [{k: c[k] for k in ('wire', 'view', 'v', 'R', 'recipe', 'other_seed_recipe_differs', 'same_recipe_differs_other_view',
                                  'distinct_inputs')} for c in any_m]
    summary = {'members': {m: x['summary'] for m, x in mine.items()}, 'grid_validity': gv_short,
               'seed_dependent_equalities': eq_list,
               'scores': {m: x['score'] for m, x in mine.items()}, 'tiers': {m: x['tier'] for m, x in mine.items()},
               'reserve_qualified': {m: x['reserve_qualified'] for m, x in mine.items()},
               'selection': {k: sel.get(k) for k in ('outcome', 'selected', 'tier', 'best', 'near')},
               'statuses': {r['config_id']: r['status'] for r in sel.get('records', [])}, 'n_seed_checks': len(checks),
               'mismatches': mm}
    V.write_record(f'V6B_CLAIMS__{a.bank}__{a.tag}.json', {**summary, 'frontiers': {m: x['frontier_per_path'] for m, x in mine.items()}})
    chk = {'id': f'V6B.CLAIMS_SELECTION.{a.bank}__{a.tag}',
           'what': 'claims and selection rebuilt from the recorded sources with own code (MC grid cells, frontier per declared path, joint '
                   'record, A-SC1 grid validity from cached k0/k1 scores, G2 conjunction, multi-path G3, ledger, selection) vs CLAIMS.json',
           'inputs': V.inputs_sha([a.plan, a.claims, V.RUNS / 'GRID' / a.bank / 'S0' / 'GRID.json', V.RUNS / 'GRID' / a.bank / 'S1' / 'GRID.json']),
           'result': summary, 'numerical_status': 'PASS' if not mm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    extra = []
    if a.tables:
        refs_doc = json.loads((V.ROOT / plan['references_eval'][0]).read_text())
        pts = reserve_frontier(plan, grid, reps, mine, refs_doc)
        tdoc = json.loads(Path(a.tables).read_text())
        fmm = compare_frontier(pts, tdoc.get('reserve_frontier') or {})
        extra.append({'id': f'V6C.RESERVE_FRONTIER.{a.bank}__{a.tag}',
                      'what': 'own reserve-qualified frontier (RULES.md section 4: reserve with the MC term, registered-cell pair AUC, '
                              'Pareto set, candidate labels against every other qualified control and reference) vs TABLES.json',
                      'inputs': V.inputs_sha([a.tables, a.plan, a.claims]),
                      'result': {'points': pts, 'mismatches': fmm},
                      'numerical_status': 'PASS' if not fmm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
    for spec in a.decisive:
        kind, path = spec.split('=', 1)
        out, dmm = decisive(kind, path, mine, grid, reps)
        extra.append({'id': f'V6D.DECISIVE.{kind.upper()}.{Path(path).stem}__{a.tag}',
                      'what': f'own replay of the decisive rule {kind} (nrm.decisive, frozen rulings) from the own claims replay vs {Path(path).name}',
                      'inputs': V.inputs_sha([path, a.claims, a.plan]), 'result': {'mine': out, 'mismatches': dmm},
                      'numerical_status': 'PASS' if not dmm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
    print(json.dumps(V.append_checks([chk] + extra), indent=1))
    for c in extra:
        print(c['id'], c['numerical_status'], json.dumps(c['result'].get('mismatches'), default=V._jd)[:600])
    print(json.dumps({k: summary[k] for k in ('members', 'selection', 'scores', 'tiers')}, indent=1, default=V._jd)[:6000])
    for m in mm[:30]:
        print('  MISMATCH', json.dumps(m, default=V._jd)[:400])
    return 0


if __name__ == '__main__':
    sys.exit(main())

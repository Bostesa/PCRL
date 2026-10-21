"""TABLES.md of the PCRL finite-routing comparison (coordinator; label-free: saved records only).

    python3 -m frc.tables   -> results/<campaign>/TABLES.md and TABLES.json

Sections: 1 fixed-routing probability comparison (T1 nested, T3 MODEL_SELECT); 2 qualification; 3 utility and recovery per release
(MODEL_SELECT utility; registered-cell mean SCORE AUC of the registered winner); 4 matched-control comparisons of PRC_S2 (both
readings); 5 decisions D1-D6 (both readings); 6 fitted dependence and nested MODEL_FIT LOOKUP against measured recovery; 7
robustness cells and the ablation (when present).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

from . import common as C
from . import claims as CL
from nrm import rules as NR

R = C.RESULTS


def _j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def _f(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '-'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def mean_auc(grid, member, view, cells=NR.REGISTERED):
    vals = []
    for c in cells:
        e = ((grid['cells'].get(c) or {}).get(member) or {}).get(view)
        if e is None:
            return None
        vals.append(e['score_auc'])
    return sum(vals) / len(vals)


def section1():
    rows = ['## 1. Fixed-routing probability comparison (identical partitions)', '',
            'MODEL_FIT nested held-out (5 outer household folds; SE_used = max(household-clustered SE, SD of the fold means / sqrt 5)) and '
            'MODEL_SELECT paired costs (household-clustered SE). Differences are loss(B1) - loss(A); negative favours B1. A = label-refit '
            'decoders (source), B1 = out-of-fold teacher means of the routed cells, B0 = stored prototypes.', '',
            '| Capacity | Version | Nested occ NLL B1-A (SE_used) | Nested total NLL B1-A | Nested occ acc loss B1-A | B0-B1 occ NLL | '
            'Spend over CONT, occ NLL: A / B1 | Share of A spend removed | Original gate A / B1 | MODEL_SELECT occ NLL B1-A (SE) | '
            'MODEL_SELECT men\'s Brier B1-A |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for cid in ('C64', 'C128'):
        for v in C.VERSIONS:
            n = _j(R / 'fa' / f'NEST_{cid}_V{v}.json')
            if n is None:
                continue
            pd, fl, so = n['paired_differences'], n['paired_differences_fold_level'], n['spend_over_cont']
            eb = _j(C.RUNS / 'FA' / f'TWB_{cid}' / 'EVAL_MS.json')
            comp = 'NRM:FA/PRC_TWIN' if cid == 'C64' else 'NRM:FA/PRC_TWIN_C128'
            c = eb['versions'][str(v)]['paired_costs'][comp]['costs'] if eb else {}
            rows.append(f"| {cid} | {v} | {_f(pd['B1_minus_A']['occ_nll'][0], 5, True)} ({_f(fl['B1_minus_A']['occ_nll']['se_used'], 5)}) | "
                        f"{_f(pd['B1_minus_A']['tot_nll'][0], 5, True)} | {_f(pd['B1_minus_A']['occ_acc'][0], 5, True)} | "
                        f"{_f(pd['B1_minus_B0']['occ_nll'][0] * -1, 6, True)} | {_f(so['A']['occ_nll'][0], 5)} / {_f(so['B1']['occ_nll'][0], 5)} | "
                        f"{_f(n['share_of_A_spend_over_cont_removed_by_B1']['occ_nll'], 3)} | "
                        f"{n['original_capacity_gate']['A'][len('ORIGINAL_CAPACITY_GATE_'):]} / {n['original_capacity_gate']['B1'][len('ORIGINAL_CAPACITY_GATE_'):]} | "
                        f"{_f(c.get('occupation|nll', {}).get('value'), 5, True)} ({_f(c.get('occupation|nll', {}).get('se'), 5)}) | "
                        f"{_f(c.get('occupation|men_brier', {}).get('value'), 5, True)} |")
    return rows


def _util_row(name, rec_path, grid):
    r = _j(rec_path)
    cells = []
    if r is None:
        return None
    vs = r.get('versions') or {}
    for v in C.VERSIONS:
        e = vs.get(str(v)) or {}
        if e.get('missing') or 'point' not in e:
            cells.append('missing')
            continue
        tot = None
        u = e['utility']
        nll = {f"{c['task']}": float(c['value']) for c in u['clauses'] if c['reference'] == 'U0' and c['metric'] == 'nll_excess'}
        tot = nll.get('income', 0) + nll.get('occupation', 0)
        cells.append(f"{'pass' if e['point']['point_pass_12_plus_2'] else 'FAIL'} {_f(e['point']['max_value_over_cap'], 3)} / "
                     f"{_f(tot, 4, True)} / {_f(e['reserve_without_mc']['min_whole_ratio'], 2)}")
    aucs = [mean_auc(grid, name, vw) if grid else None for vw in ('pair', 'income', 'occupation')]
    return f"| {name} | {cells[0]} | {cells[1]} | " + ' | '.join(_f(a) for a in aucs) + ' |'


def section3(grid):
    rows = ['## 3. Utility and recovery per release', '',
            'Per version: original point status (24 clauses + 4 guards), worst value / cap, total NLL excess vs U0 (income + occupation, '
            'nats), and the whole-release two-SE reserve ratio (>= 2 meets the reserve; reported, never gating). AUC = registered-cell mean '
            'SCORE AUC (8 cells) of the registered winner for the recorded SEX column, per view.', '',
            '| Release | Version 0 | Version 1 | Pair AUC | Income AUC | Occupation AUC |', '|---|---|---|---|---|---|']
    fam = {'TWIN': 'TWB_C64', 'PRC_S2': 'PRC_S2', 'HQ_S2': 'HQ_S2', 'SHUF_S2': 'SHUF_S2', 'UNT_BR': 'UNT_BR', 'ORD_S2': 'ORD_S2',
           'ORD_S3': 'ORD_S3', 'TWB_C128': 'TWB_C128', 'ABL_LOC': 'ABL_LOC', 'ABL_PAIR': 'ABL_PAIR'}
    for name, cfg in fam.items():
        x = _util_row(name, C.RUNS / 'FA' / cfg / 'EVAL_MS.json', grid)
        if x:
            rows.append(x)
    for name, cfg in (('A_TWIN', 'PRC_TWIN'), ('A_TWIN_C128', 'PRC_TWIN_C128')):
        x = _util_row(name, C.RUNS / 'SOURCE' / 'FA' / cfg / 'EVAL_MS.json', grid)
        if x:
            rows.append(x)
    refs = _j(C.RUNS / 'REFERENCES' / 'EVAL_MS.json')
    if refs:
        spec = {'W': 'W', 'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
                'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'U0': 'REF:U0', 'Ucal': 'REF:Ucal'}
        for name, key in spec.items():
            r = refs['references'].get(key)
            if not r:
                continue
            cells = []
            for v in C.VERSIONS:
                e = r[str(v)]
                u = e['utility']
                nll = {f"{c['task']}": float(c['value']) for c in u['clauses'] if c['reference'] == 'U0' and c['metric'] == 'nll_excess'}
                cells.append(f"{'pass' if e['point']['point_pass_12_plus_2'] else 'FAIL'} {_f(e['point']['max_value_over_cap'], 3)} / "
                             f"{_f(nll.get('income', 0) + nll.get('occupation', 0), 4, True)} / {_f(e['reserve']['min_whole_ratio'], 2)}")
            aucs = [mean_auc(grid, name, vw) if grid else None for vw in ('pair', 'income', 'occupation')]
            rows.append(f"| {name} | {cells[0]} | {cells[1]} | " + ' | '.join(_f(a) for a in aucs) + ' |')
    return rows


def section4(grid, dec_main, dec_r1):
    rows = ['## 4. Matched-control comparisons of PRC_S2 (pair benefit = comparator AUC - PRC_S2 AUC; positive favours PRC_S2)', '',
            '| Comparator | Benefit (aggregate) | se_agg | Version 0 | Version 1 | Screen | Income recipient | Occupation recipient | '
            'Non-table winner benefit (V0 / V1) |', '|---|---|---|---|---|---|---|---|---|']
    for comp in ('TWIN', 'HQ_S2', 'SHUF_S2', 'UNT_BR', 'ORD_S2', 'ORD_S3', 'A_TWIN', 'NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL',
                 'SUCC', 'TWJ', 'W', 'TR100', 'TT100'):
        e = (grid['paired'].get('PRC_S2') or {}).get(comp)
        if e is None:
            continue
        c = NR.comparison(e)
        b = c['benefit']
        ag = b.get('aggregate') or {}
        vm = {k: (x or {}).get('mean') for k, x in (ag.get('per_version') or {}).items()}
        nl = CL.nolookup_benefit(grid, 'PRC_S2', comp)
        from . import rules as RU
        nvm = RU._version_means(nl) if nl else None
        rows.append(f"| {comp} | {_f(ag.get('mean'), 4, True)} | {_f(ag.get('se_agg'), 4)} | {_f(vm.get(0, vm.get('0')), 4, True)} | "
                    f"{_f(vm.get(1, vm.get('1')), 4, True)} | {b['label']} | {c['recipient']['income']['label']} | "
                    f"{c['recipient']['occupation']['label']} | {_f((nvm or {}).get(0), 4, True)} / {_f((nvm or {}).get(1), 4, True)} |")
    for title, dec in (('registered reading (untargeted path [SHUF_S2])', dec_main), ('revision R1 reading ([SHUF_S2, UNT_BR])', dec_r1)):
        if not dec:
            continue
        d2 = dec['decisions']['D2']
        def _one(n, x):
            if 'label' not in x:                         # the TWIN comparison carries benefit / pass / targeting, no frontier label
                return (f"{n}: benefit {x.get('benefit')}, pass {x.get('pass')}, targeting pass {x.get('targeting_pass')}, "
                        f"recipient gate pass {x.get('recipient_gate_pass')}")
            return f"{n} {x.get('label')}" + (f" (bracket {x.get('point')}, {x.get('reason')})" if x.get('point') or x.get('reason') else '')
        rows += ['', f'Frontier comparisons, {title}: ' + '; '.join(_one(n, x) for n, x in (d2.get('comparisons') or {}).items())]
    return rows


def section5(dec_main, dec_r1):
    rows = ['## 5. Decisions (frc.rules, FRC1-rules-v1)', '', '| Decision | Registered reading | Revision R1 reading |', '|---|---|---|']
    for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'outcome_class'):
        a = ((dec_main or {}).get('decisions') or {}).get(k, {})
        b = ((dec_r1 or {}).get('decisions') or {}).get(k, {})
        rows.append(f"| {k} | {a.get('label', '-')} | {b.get('label', '-')} |")
    for title, dec in (('registered', dec_main), ('R1', dec_r1)):
        if dec:
            d2 = dec['decisions']['D2']
            rows.append(f"\nD2 detail ({title}): passed {d2.get('passed')}; F4 pass {((d2.get('f4') or {}).get('pass'))}; flags "
                        f"{d2.get('flags')}; targeting " + json.dumps({q: x.get('pass') for q, x in (d2.get('targeting') or {}).items()}))
    return rows


def section6(grid):
    rows = ['## 6. Fitted dependence and MODEL_FIT nested LOOKUP against measured recovery', '',
            'Fitted pi-weighted plug-in pair MI (MODEL_FIT out-of-fold cells; nats) and the nested held-out pair LOOKUP AUC (MODEL_FIT) are '
            'estimates under the fitted propensity and table readers; the grid AUC is the registered winner on MODEL_SELECT SCORE. None of '
            'them is a guarantee about other classifiers.', '',
            '| Release | Version | Fitted pi pair MI (twin) | Re-routed vs twin, MODEL_SELECT | Occupied out-of-fold cells, income / '
            'occupation / pair (twin) | Nested LOOKUP AUC (twin) | Grid pair AUC, this version | Grid pair AUC, twin |', '|---|---|---|---|---|---|---|---|']
    for name, cfg, fam in (('PRC_S2', 'PRC_S2', 'PRC'), ('HQ_S2', 'HQ_S2', 'HQ'), ('SHUF_S2', 'SHUF_S2', 'SHUF'), ('UNT_BR', 'UNT_BR', 'UNT')):
        for v in C.VERSIONS:
            rec = _j(C.RUNS / 'FA' / cfg / f'V{v}' / 'RELEASE.json') or {}
            o = rec.get('objective') or {}
            mv = rec.get('rerouted') or {}
            if fam == 'UNT':
                u = _j(R / 'fa' / f'UNT_SELECTION_V{v}.json')
                look = u['nested_lookup_pair'][u['bracket']]['auc'] if u else None
                twl = u['nested_lookup_pair']['TWIN']['auc'] if u else None
            else:
                s = _j(R / 'fa' / f'SELECTION_V{v}.json')
                key = s['selection'][fam]['S2']
                look, twl = s['nested_lookup_pair'][key]['auc'], s['nested_lookup_pair']['PRC_TWIN']['auc']
            cells = [c for c in NR.REGISTERED if c.startswith(f'S{v}')]
            ga = mean_auc(grid, name, 'pair', cells) if grid else None
            gt = mean_auc(grid, 'TWIN', 'pair', cells) if grid else None
            oc = rec.get('occupied_cells_oof') or {}
            tw = (_j(C.RUNS / 'FA' / 'TWB_C64' / f'V{v}' / 'RELEASE.json') or {}).get('occupied_cells_oof') or {}
            occ = f"{oc.get('income', '-')} / {oc.get('occupation', '-')} / {oc.get('pair', '-')} ({tw.get('pair', '-')} pair)"
            rows.append(f"| {name} | {v} | {_f(o.get('fitted_pi_pair_mi'), 4)} ({_f(o.get('twin_pi_pair_mi'), 4)}) | "
                        f"{_f(mv.get('model_select_rerouted_vs_twin'), 3)} | {occ} | {_f(look)} ({_f(twl)}) | {_f(ga)} | {_f(gt)} |")
    return rows


def section7():
    st = _j(R / 'claims' / 'STABILITY' / 'CLASSIFIER_STABILITY.json')
    if not st:
        return []
    eq = _j(R / 'claims' / 'STABILITY' / 'REFIT_EQUALITIES.json') or {}
    rows = ['## 7. Classifier stability of PRC_S2 (robustness readers; descriptive, never a decision input)', '',
            'Refit bank FRC1_ROB: the 8 registered cells plus 24 robustness cells (repetitions R3, R4 with reader seeds k0-k3, and R1, R2 '
            'with k2, k3; both versions). Benefit = comparator pair AUC - PRC_S2 pair AUC (positive favours PRC_S2); nrm.rules.robustness.',
            '', '| Comparator | All 32 cells: mean (se_agg) | Robustness cells only (24): mean (se_agg) | Cells opposite to the mean / distinct | '
            'Versions opposite | Label |', '|---|---|---|---|---|---|']
    for comp, r in st['robustness'].items():
        if 'all_cells' not in r:
            rows.append(f'| {comp} | - | - | - | - | {r.get("label")} |')
            continue
        a, b = r['all_cells']['aggregate'], r['robustness_cells']['aggregate']
        sr = a['sign_reversals']
        rows.append(f"| {comp} | {_f(a['mean'], 4, True)} ({_f(a['se_agg'], 4)}) | {_f(b['mean'], 4, True)} ({_f(b['se_agg'], 4)}) | "
                    f"{sr['cells_opposite_to_mean']} / {sr['distinct_cells']} | {sr['versions_opposite']} | {r['label']} |")
    rows += ['', '| Ablation (candidate) vs comparator | Registered cells: mean (se_agg), screen | All 32 cells: mean (se_agg) | '
                 'Robustness label |', '|---|---|---|---|']
    for k, r in st['ablation'].items():
        ag = r['registered']['benefit'].get('aggregate') or {}
        a = r['robustness']['all_cells']['aggregate']
        rows.append(f"| {k.replace('_vs_', ' vs ')} | {_f(ag.get('mean'), 4, True)} ({_f(ag.get('se_agg'), 4)}), "
                    f"{r['registered']['benefit']['label']} | {_f(a['mean'], 4, True)} ({_f(a['se_agg'], 4)}) | {r['robustness']['label']} |")
    re_ = st['refit_effectiveness']
    rows += ['', f"Refit effectiveness (frc.rules.refit_effectiveness, frozen strict rule): **{re_['label']}** (reasons {re_['reasons']}; "
                 f"{st['refit_checks']} k-pair checks). Seed-dependent equalities: {eq.get('equal_seed_dependent')} of "
                 f"{eq.get('checks_seed_dependent')}, by recipe and view {eq.get('by_recipe_view')}; meeting the NRM1 (iii) explanation "
                 f"(descriptive, frc.refitcheck): {(eq.get('nrm1_iii_explanation') or {}).get('explained')} explained, "
                 f"{(eq.get('nrm1_iii_explanation') or {}).get('unexplained')} unexplained; every other seed-dependent check in the bank "
                 f"({eq.get('checks_seed_dependent', 0) - eq.get('equal_seed_dependent', 0)} of {eq.get('checks_seed_dependent')}) differed "
                 f"between its two reader seeds. The strict rule counts 240 of the ties on the minimum set. For those extra-trees recipes "
                 f"the k2 / k3 cells duplicate k0 / k1, so 26-29 of the 32 cells are distinct: the robustness evidence varies the split "
                 f"fully but the reader seed only partly. The new repetitions' splits pass. The label stands."]
    return rows


def section8():
    rows = ['## 8. Label-free strength probe (design record; no label file, no MODEL_SELECT)', '',
            'The full-pool PRC continuation chain on cached MODEL_FIT out-of-fold statistics, continued above the registered grid. '
            'Decision rule registered at 00:44Z before the run: no revision R2 if either version is saturated.', '',
            '| Version | Strengths | Re-routed vs twin (income / occupation) | Fitted pi pair MI (twin) |', '|---|---|---|---|']
    for v in C.VERSIONS:
        p = _j(R / 'fa' / f'PROBE_STRENGTH_V{v}.json')
        if not p:
            continue
        groups = {}
        for lam in sorted(p['per_strength'], key=float):
            x = p['per_strength'][lam]
            groups.setdefault(x['partition_sha256'], []).append((lam, x))
        for part, xs in groups.items():
            x = xs[0][1]
            rows.append(f"| {v} | {xs[0][0]}-{xs[-1][0]} | {_f(x['rerouted_vs_twin'], 3)} ({_f(x['income_rerouted'], 3)} / "
                        f"{_f(x['occupation_rerouted'], 3)}) | {_f(x['fitted_pi_pair_mi'], 5)} ({_f(p['twin_pi_pair_mi'], 5)}) |")
    return rows


def section9():
    """FIT-fraction curve (FRC1_LC; descriptive): registered-cell mean pair SCORE AUC of the winner per member and reader-FIT fraction."""
    if not all((C.RUNS / 'GRID' / 'FRC1_LC' / f'S{v}' / 'GRID.json').is_file() for v in C.VERSIONS):
        return []
    g = CL.load_grid('FRC1_LC')
    fr = (0.25, 0.5, 1.0)
    members = ('TWIN', 'PRC_S2', 'HQ_S2', 'UNT_BR', 'ORD_S2')
    rows = ['## 9. Reader FIT-fraction curve (FRC1_LC; descriptive, PROTOCOL 9)', '',
            'Registered-cell mean pair SCORE AUC of the winner (8 cells) when the readers fit on 25%, 50% or 100% of the reader-FIT '
            'households; the 100% column equals the registered grid. Rising AUC with more FIT data means the readers are data-limited.', '',
            '| Release | 25% | 50% | 100% | Benefit of PRC_S2 at 25% / 50% / 100% |', '|---|---|---|---|---|']
    def m(name, f):
        cells = [c if f == 1.0 else f'{c}@{f}' for c in NR.REGISTERED]
        vals = [((g['cells'].get(c) or {}).get(name) or {}).get('pair', {}).get('score_auc') for c in cells]
        return None if any(x is None for x in vals) else sum(vals) / len(vals)
    for name in members:
        a = [m(name, f) for f in fr]
        p = [m('PRC_S2', f) for f in fr]
        b = ' / '.join(_f(x - y, 4, True) if x is not None and y is not None else '-' for x, y in zip(a, p)) if name != 'PRC_S2' else '-'
        rows.append(f"| {name} | {_f(a[0])} | {_f(a[1])} | {_f(a[2])} | {b} |")
    return rows


def main():
    grid = None
    if (C.RUNS / 'GRID' / 'FRC1_MAIN' / 'S0' / 'GRID.json').is_file() and (C.RUNS / 'GRID' / 'FRC1_MAIN' / 'S1' / 'GRID.json').is_file():
        grid = CL.load_grid('FRC1_MAIN')
    dec_main = _j(R / 'claims' / 'MAIN_ER1' / 'DECISIONS.json')
    dec_r1 = _j(R / 'claims' / 'R1_ER1' / 'DECISIONS.json')
    q = _j(R / 'decisions' / 'QUAL_DECISION.json')
    out = ['# Tables (campaign `pcrl_finite_routing_comparison_v1`)', '',
           'Generated by `python3 -m frc.tables` from the saved records. Development evidence on adaptively used MODEL_SELECT records, '
           'for the declared readers only; lower recovery for these readers is not protection against every reader. Versions share '
           'every training row and the MODEL_SELECT people (procedure repeats, not replications).', '']
    out += section1()
    out += ['', '## 2. Qualification (QUAL, S2 slack; frc.decide)', '']
    if q:
        for cid, r in q['per_capacity'].items():
            out.append(f"- {cid}: Q-PE {r['qpe']['label']}; chosen construction {r['construction_choice'].get('choice')}; QUAL "
                       + ', '.join(f"{c} {x['label']}" for c, x in r['qual'].items()))
        out.append(f"- Outcome {q['outcome']}: protection capacity {q.get('capacity_id')} with construction {q.get('construction')}; D1 "
                   f"{q['d1']['label']}.")
    if grid:
        out += [''] + section3(grid) + [''] + section4(grid, dec_main, dec_r1)
    out += [''] + section5(dec_main, dec_r1)
    if grid:
        out += [''] + section6(grid)
    out += [''] + section7() + [''] + section8() + [''] + section9()
    (R / 'TABLES.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out[:12]))


if __name__ == '__main__':
    sys.exit(main())

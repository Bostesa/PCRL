#!/usr/bin/env python3
"""RFA1 verifier, wave 3, task 1 (one queue unit): the R1 decision record, the half-budget grid validity, the family-change reading
FC_R1, the rebuilt v1 record and the v1 second split (D2R), from the saved outputs with the verifier's own code.

1. R1 usefulness: the 28 conditions of RF1_* from RELEASE.npz + MODEL_SELECT labels (vlib) vs runs/FAM/RF1_*/EVAL_MS.json; paired
   costs (incl. vs FAM/RF_FULL); income identity; the plan's bitwise-identity disclosure (RF1_TASK / RF1_RAWJ_HI vs v1).
2. Grids RFA1_R1_MAIN, RFA1_R1_LC, RFA1_LC, RFA1_ROB (both versions, every cell incl. '@0.5'): winners, SCORE AUCs, bootstrap draws,
   per-recipe and non-table AUCs, paired entries of the candidate vs every comparator (verifier code; compared with GRID.json /
   BOOT.npz); reader-seed equalities per (rep, fraction) for seed pairs (0,1) and (2,3).
3. Verifier labels: R1 D1-D4 (vagg), FC_R1 (RULES_ADDENDUM_1_R1 section 3, vagg / own pooling), v1 D2R per comparator (vagg2).
4. Frozen-rules legs (rfa.rules unchanged): the R1 and rebuilt-v1 records rebuilt by rfa.rfclaims.build and decided (a) as recorded
   (replay) and (b) with the verifier's grid / non-table / half-budget entries substituted; rfa.rfclaims.family_change on the
   verifier's entry; rfa.rules.d2r on the verifier's merged MAIN + ROB entries. Reader-budget flags and the grid-validity /
   refit-effectiveness labels are taken from the records (their inputs are label-free fingerprints).
SEX and task labels through rfa.common.load(..., 'verify') from this __main__, in a queue unit.

    python3 -m rfa.jobqueue run --unit RFA_VER_W3_MAIN_v1 --stage verify --rss-gib 3 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v3_main.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vagg as A  # noqa: E402
import vagg2 as B  # noqa: E402
import vcore  # noqa: E402
import vlib as V  # noqa: E402
import v2_grid as G2  # noqa: E402
import v2_usefulness as U2  # noqa: E402

OUT = HERE.parent / 'out' / 'V3_MAIN.json'
RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
RUNS = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1' / 'runs'
R1 = ('RF1_FULL', 'RF1_TASK', 'RF1_STOP_LO', 'RF1_STOP_HI', 'RF1_ALTC_HI', 'RF1_RAWJ_HI')
V1 = ('RF_FULL', 'RF_TASK', 'RF_RAWJ_HI')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
TOL = 1e-12


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_ms():
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'verify: wave 3 independent recomputation of the R1 usefulness, the R1 / LC / ROB reader grids, the R1 decision record, '
               'the family-change reading, the rebuilt v1 record and the v1 second split (verifier wave 3)')
    S.pop('X', None)
    return S


def recompute(bank, cand, version, sex, unit, row_id):
    """Generalized v2_grid.recompute_version: every cell of the bank's S<version> incl. learning-curve '@<frac>' cells."""
    gdir = RUNS / 'GRID' / bank / f'S{version}'
    grid = json.loads((gdir / 'GRID.json').read_text())
    with np.load(gdir / 'BOOT.npz') as z:
        boot = {k: z[k].copy() for k in z.files}
    members, prov_ok = G2.members_of(grid, version, row_id)
    sx = V.arr_sha(np.asarray(sex))
    worst = {k: 0.0 for k in ('val_auc', 'score_auc', 'boot', 'per_recipe_score_auc', 'nontable_score_auc', 'paired_d', 'paired_se')}
    mism, mine, cache = [], {}, {}
    for cell, cres in grid['cells'].items():
        vv, rep, k, frac = vcore.parse_cell(cell)
        if vv != version:
            mism.append(f'{cell}: wrong version')
            continue
        roles = V.reader_roles(unit, rep, frac)
        y = np.asarray(sex)[roles['SCORE']]
        W = V.boot_weights(np.asarray(unit)[roles['SCORE']], version, rep)
        rs = V.reader_seed(version, k)
        for name, ent in cres.items():
            rel = members[name]
            E = {}
            for view in V.VIEWS:
                p = RUNS / 'devgrid_cache' / f"{V.cache_key(rep, rs, frac, V.view_hash(view, rel['income'], rel['occupation']), sx)}.npz"
                if p.is_file():
                    E[view] = V.read_cache_entry(p)
                    cache[(version, rep, k, frac, name, view)] = E[view]
            if len(E) != 3:
                mism.append(f'{cell}/{name}: cache entry missing')
                continue
            info = {vw: {n: E[vw][n][1] for n in V.RECIPES} for vw in V.VIEWS}
            w, wn = V.winners(info), G2.nontable_winner(info)
            for view in V.VIEWS:
                g = ent[view]
                vw, rname = w[view]['reader_key'].split('__', 1)
                s = E[vw][rname][0]
                a = V.auc(y, s)
                bd = V.weighted_auc(y, s, W)
                if w[view]['winner'] != g['winner'] or w[view]['near_tie'] != g['near_tie'] or w[view]['gap_to_runner_up'] != g['gap_to_runner_up']:
                    mism.append(f'{cell}/{name}/{view}: winner {g["winner"]} vs verifier {w[view]["winner"]}')
                worst['val_auc'] = max(worst['val_auc'], abs(w[view]['val_auc'] - g['val_auc']))
                worst['score_auc'] = max(worst['score_auc'], abs(a - g['score_auc']))
                key = f'{cell}__{name}__{view}'
                worst['boot'] = max(worst['boot'], float(np.max(np.abs(bd - boot[key]))) if key in boot else float('inf'))
                if 'per_recipe_score_auc' in g:
                    worst['per_recipe_score_auc'] = max(worst['per_recipe_score_auc'],
                                                        max(abs(A.auc_fast(y, E[view][n][0]) - g['per_recipe_score_auc'][n]) for n in V.RECIPES))
                nvw, nname = wn[view]['reader_key'].split('__', 1)
                na = V.auc(y, E[nvw][nname][0])
                gx = (g.get('extras') or {}).get('nolookup') or {}
                if gx and gx.get('winner') != wn[view]['winner']:
                    mism.append(f'{cell}/{name}/{view}: non-table winner differs')
                if gx:
                    worst['nontable_score_auc'] = max(worst['nontable_score_auc'], abs(na - gx['score_auc']))
                mine[(cell, name, view)] = {'auc': a, 'boot': bd, 'nontable_auc': na}
    seed_eq = {}
    for (v_, rep, k, frac, name, view), E in cache.items():
        for a_, b_ in ((0, 1), (2, 3)):
            if k == a_ and (v_, rep, b_, frac, name, view) in cache:
                E1 = cache[(v_, rep, b_, frac, name, view)]
                for n in SEED_DEP:
                    key = f'{n}|frac{frac}'
                    seed_eq.setdefault(key, [0, 0])
                    seed_eq[key][0] += int(np.array_equal(E[n][0], E1[n][0]))
                    seed_eq[key][1] += 1
    paired, nontable = {}, {}
    for comp in (grid.get('paired') or {}).get(cand, {}):
        for view in V.VIEWS:
            e = {'cells': {}, 'cell_se': {}}
            for cell in grid['cells']:
                if (cell, cand, view) in mine and (cell, comp, view) in mine:
                    a_, b_ = mine[(cell, cand, view)], mine[(cell, comp, view)]
                    d, se = V.paired_cell(view, a_['auc'], b_['auc'], a_['boot'], b_['boot'])
                    e['cells'][cell], e['cell_se'][cell] = d, se
                    gp = grid['paired'][cand][comp][view]
                    worst['paired_d'] = max(worst['paired_d'], abs(d - gp['cells'][cell]))
                    worst['paired_se'] = max(worst['paired_se'], abs(se - gp['cell_se'][cell]))
            paired.setdefault(comp, {})[view] = e
        nontable[comp] = {'cells': {cell: mine[(cell, comp, 'pair')]['nontable_auc'] - mine[(cell, cand, 'pair')]['nontable_auc']
                                    for cell in grid['cells'] if (cell, comp, 'pair') in mine and (cell, cand, 'pair') in mine}}
    return {'n_cells': len(grid['cells']), 'worst': worst, 'mismatches': mism, 'provenance_ok': all(prov_ok.values()),
            'seed_equalities': seed_eq, 'paired': paired, 'nontable': nontable}


def merge(per_v, key='paired'):
    out = {}
    for v in (0, 1):
        for comp, views in per_v[v][key].items():
            if key == 'paired':
                for view, e in views.items():
                    d = out.setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells']); d['cell_se'].update(e['cell_se'])
            else:
                out.setdefault(comp, {'cells': {}})['cells'].update(views['cells'])
    return out


def usefulness_r1(labels, sex, unit, row_id, chk, res):
    bank, rels = {}, {}
    for v in (0, 1):
        bank[v], _ = U2.own_bank(v, row_id)
        for c in R1 + V1:
            rels[(c, v)], prov = U2.own_release(c, v)
            chk.setdefault('usefulness|release_files_and_arrays_hash_ok', True)
            chk['usefulness|release_files_and_arrays_hash_ok'] &= prov['file_hash_ok'] and prov['all_12_arrays_hash_ok']
    refs = {v: {'U0': bank[v]['U0'], 'Ucal': bank[v]['Ucal']} for v in (0, 1)}
    out = {}
    for c in R1:
        ev = json.loads((RUNS / 'FAM' / c / 'EVAL_MS.json').read_text())
        per = {v: V.usefulness_version(labels, rels[(c, v)], refs[v], sex, unit) for v in (0, 1)}
        g = V.g1(per)
        cmp_ = {v: V.compare_with_campaign_utility(per[v], ev['versions'][str(v)]['utility']) for v in (0, 1)}
        camp = all(bool(ev['versions'][str(v)]['point']['point_pass_12_plus_2']) for v in (0, 1))
        out[c] = {'G1': g['G1'], 'failed': g['failed'], 'agree': all(x['agree'] for x in cmp_.values()), 'campaign_G1': camp,
                  'max_abs_diff': max(x['max_abs_diff'] for x in cmp_.values()),
                  'confidence': {str(v): U2.confidence(per[v]) for v in (0, 1)},
                  'worst_over_cap': {str(v): max([x['value'] / x['cap'] for x in per[v]['clauses'].values()] +
                                                 [x['value'] / x['cap'] for x in per[v]['guards'].values()]) for v in (0, 1)}}
        chk[f'usefulness|{c}|28_conditions_match_EVAL_MS'] = out[c]['agree']
        chk[f'usefulness|{c}|G1_agrees'] = g['G1'] == camp
    res['usefulness_R1'] = out
    # paired costs of every RF1 configuration vs its EVAL_MS comparators
    losses = {(c, v): U2.per_person_losses(labels, rels[(c, v)]) for c in R1 + V1 for v in (0, 1)}
    for v in (0, 1):
        losses[('REF:NEURAL_RAW', v)] = U2.per_person_losses(labels, bank[v]['NEURAL_RAW'])
        losses[('REF:H_LJ', v)] = U2.per_person_losses(labels, bank[v]['H_LJ'])
    worst, skipped = 0.0, []
    for c in R1:
        ev = json.loads((RUNS / 'FAM' / c / 'EVAL_MS.json').read_text())
        for v in (0, 1):
            for spec, rec in ev['versions'][str(v)]['paired_costs'].items():
                if rec.get('missing'):
                    continue
                key = (spec.split('/')[1], v) if spec.startswith('FAM/') else (spec, v)
                if key not in losses:
                    skipped.append(f'{c}|V{v}|{spec}')
                    continue
                mine = U2.paired_costs(losses[(c, v)], losses[key], sex, unit)
                worst = max(worst, max(max(abs(mine[m]['value'] - rec['costs'][m]['value']), abs(mine[m]['se'] - rec['costs'][m]['se'])) for m in mine))
    res['paired_costs_R1'] = {'worst_abs_diff': worst, 'skipped': skipped}
    chk['usefulness|paired_costs_match (abs <= 1e-12)'] = worst <= TOL and not skipped
    # income identity and the plan's bitwise-identity disclosure
    plan = json.loads((RES / 'claims' / 'PLAN_RFA_R1.json').read_text())
    ident = {}
    for v in (0, 1):
        base = rels[('RF1_FULL', v)]['income']
        ident[str(v)] = {'RF1_occupation_only_income_identical': all(np.array_equal(rels[(c, v)]['income'], base) for c in R1 if c != 'RF1_RAWJ_HI'),
                         'RF1_FULL_income_equals_RF_FULL_income': bool(np.array_equal(base, rels[('RF_FULL', v)]['income'])),
                         'RF1_TASK_equals_RF_TASK_bitwise': all(np.array_equal(rels[('RF1_TASK', v)][t], rels[('RF_TASK', v)][t]) for t in V.TASKS),
                         'RF1_RAWJ_HI_equals_RF_RAWJ_HI_bitwise': all(np.array_equal(rels[('RF1_RAWJ_HI', v)][t], rels[('RF_RAWJ_HI', v)][t]) for t in V.TASKS)}
    disc = plan['bitwise_identity_with_v1']
    res['identity'] = ident
    chk['usefulness|RF1_income_identity'] = all(x['RF1_occupation_only_income_identical'] and x['RF1_FULL_income_equals_RF_FULL_income'] for x in ident.values())
    chk['usefulness|plan_bitwise_identity_disclosure_matches'] = all(
        disc['RF1_TASK']['bitwise_equal_by_version'][str(v)] == ident[str(v)]['RF1_TASK_equals_RF_TASK_bitwise'] and
        disc['RF1_RAWJ_HI']['bitwise_equal_by_version'][str(v)] == ident[str(v)]['RF1_RAWJ_HI_equals_RF_RAWJ_HI_bitwise'] for v in (0, 1))
    return out


def own_labels_r1(ent, nt, d1, chk, res):
    """R1 D2 / D3 / D4 with vagg on the verifier's R1 MAIN entries; costs from RF1_FULL's EVAL_MS (reproduced above)."""
    ev = json.loads((RUNS / 'FAM' / 'RF1_FULL' / 'EVAL_MS.json').read_text())
    costs = lambda spec: {v: ev['versions'][str(v)]['paired_costs'][spec]['costs'] for v in (0, 1)}
    paths = {'STOPFIT': ['RF1_STOP_LO', 'RF1_STOP_HI'], 'ALTCRIT': ['RF1_ALTC_HI'], 'RAWJ': ['RF1_RAWJ_HI']}
    cv = {'NEURAL_RAW': costs('REF:NEURAL_RAW'), **{m: costs(f'FAM/{m}') for ms in paths.values() for m in ms}}
    cmp_all = {c: A.comparison(ent[c]) for c in ent}
    fr = {k: A.frontier(cv, ['NEURAL_RAW'] + ids, ent) for k, ids in paths.items()}
    targets = ['NEURAL_RAW', 'RF1_TASK'] + [f['bracket'] for f in fr.values() if f.get('bracket')]
    tg = {t: A.point_positive(nt.get(t)) for t in dict.fromkeys(targets)}
    gating = {'START': cmp_all['NEURAL_RAW']['passes'], 'CONT': cmp_all['RF1_TASK']['passes'], **{k: bool(f.get('passes')) for k, f in fr.items()}}
    order = ('START', 'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ')
    tpass = all(x is not None and x['pass'] for x in tg.values())
    d2 = ('NOT_ASSESSED_TASK_INFEASIBLE' if d1 != 'TASK_FEASIBLE' else
          ('COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS' if tpass else 'READER_TARGETED') if all(gating.values()) else
          'NOT_SEPARATED' if not any(gating.values()) else 'PARTIAL_' + '+'.join(k for k in order if gating[k]) + '_ONLY')
    refs = [r for r in A.BENCHMARK if r in cmp_all]
    fail = [r for r in refs if not cmp_all[r]['passes']]
    d3 = 'INCOMPLETE' if len(refs) != 9 else ('NOT_SUPERIOR' if fail else 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES')
    tsc = {r: A.target_screen(ent[r]['pair']) for r in A.TARGET_REFS}
    labs = [tsc[r]['label'] for r in A.TARGET_REFS]
    rg = all(cmp_all[r]['recipient_gate_pass'] for r in A.TARGET_REFS)
    lvl = ('MATERIAL_0020_RESOLVED' if rg else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED') if all(x == 'TARGET_AGG_RESOLVED' for x in labs) else \
        'MATERIAL_0020_POINT_ONLY' if all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in labs) else \
        'POSITIVE_BELOW_MATERIAL_TARGET' if all(cmp_all[r]['benefit_positive'] for r in A.TARGET_REFS) else 'NOT_MET'
    # FC_R1 (addendum section 3) with the verifier's entry and RF1_FULL vs FAM/RF_FULL costs
    c = cmp_all['RF_FULL']
    pc = costs('FAM/RF_FULL')
    d = A.pooled({v: pc[v]['total|nll'] for v in (0, 1)})
    sec = {m: A.pooled({v: pc[v][m] for v in (0, 1)}) for m in A.SECONDARY}
    higher = sorted(m for m, x in sec.items() if x['value'] - 2 * x['se'] > 0)
    state = 'R1_SPENDS_NO_MORE' if d['value'] <= 0 else ('R1_SPENDS_MORE_RESOLVED' if d['value'] - 2 * d['se'] > 0 else 'R1_SPENDS_MORE_UNRESOLVED')
    b = c['benefit']
    fc = ('FAMILY_CHANGE_RESOLVED' if b['resolved'] and c['recipient_gate_pass'] and state != 'R1_SPENDS_MORE_RESOLVED' and not higher else
          'FAMILY_CHANGE_WITH_HIGHER_SPEND' if b['resolved'] and c['recipient_gate_pass'] else
          'FAMILY_CHANGE_RECIPIENT_GATE_FAILED' if b['resolved'] else
          'FAMILY_CHANGE_POINT_ONLY' if b['aggregate']['mean'] > 0 else 'FAMILY_CHANGE_NOT_POSITIVE')
    res['own_labels_R1'] = {'D1': d1, 'D2': d2, 'D2_gating': gating, 'D3': d3, 'D3_failing': fail, 'D4': lvl,
                            'comparisons': {k: G2.summarize_comparison(x) for k, x in cmp_all.items()}, 'frontiers': fr, 'targeting': tg,
                            'FC_R1': {'label': fc, 'mean': b['aggregate']['mean'], 'se_agg': b['aggregate']['se_agg'],
                                      'version_means': {str(v): b['aggregate']['per_version'][v]['mean'] for v in (0, 1)},
                                      'recipient': {w: c['recipient'][w]['label'] for w in ('income', 'occupation')},
                                      'primary_d': d, 'spend_state': state, 'secondary_higher': higher}}
    return res['own_labels_R1']


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    S = load_ms()
    labels = {t: S[t] for t in V.TASKS}
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    res = {'schema': 'RFA1-verifier-wave3-main-v1', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'checks': {}}
    chk = res['checks']
    u = usefulness_r1(labels, sex, unit, row_id, chk, res)
    d1 = 'TASK_FEASIBLE' if u['RF1_FULL']['G1'] else 'CAND_INFEASIBLE'
    grids = {}
    for bank, cand in (('RFA1_R1_MAIN', 'RF1_FULL'), ('RFA1_R1_LC', 'RF1_FULL'), ('RFA1_LC', 'RF_FULL'), ('RFA1_ROB', 'RF_FULL')):
        pv = {v: recompute(bank, cand, v, sex, unit, row_id) for v in (0, 1)}
        grids[bank] = pv
        for v in (0, 1):
            chk[f'grid|{bank}|S{v}|reproduced_exactly'] = (not pv[v]['mismatches'] and pv[v]['provenance_ok'] and
                                                         all(x <= TOL for x in pv[v]['worst'].values()))
        res.setdefault('grids', {})[bank] = {str(v): {k: x for k, x in pv[v].items() if k not in ('paired', 'nontable')} for v in (0, 1)}
    ent = {b: merge(grids[b]) for b in grids}
    nt = {b: merge(grids[b], 'nontable') for b in grids}
    res['inputs'] = {'entries': ent, 'nontable': nt}
    own = own_labels_r1(ent['RFA1_R1_MAIN'], nt['RFA1_R1_MAIN'], d1, chk, res)
    # ---- compare with the R1 decision record and FC_R1
    D = json.loads((RES / 'claims' / 'RFA_R1' / 'DECISIONS.json').read_text())
    dd = D['decisions']
    pre = 'ADAPTIVE_R1__'
    for k in ('D1', 'D2', 'D3', 'D4'):
        chk[f'R1|{k}_label_equals_verifier'] = dd[k]['label'] == pre + own[k]
    chk['R1|class_BOUNDED_NEGATIVE'] = dd['outcome_class']['label'] == pre + 'BOUNDED_NEGATIVE_EXPERIMENT' and \
        not own['D2'].startswith('COMPONENT_GAIN') and own['D3'] != 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES'
    c2 = dd['D2']['comparisons']
    ok = True
    for name, comp in (('START', 'NEURAL_RAW'), ('CONT', 'RF1_TASK')):
        m = own['comparisons'][comp]
        ok &= c2[name]['label'].replace(pre, '') == m['benefit_label'] and abs(c2[name]['benefit_mean'] - m['mean']) <= TOL and \
            abs(c2[name]['benefit_se'] - m['se_agg']) <= TOL
    for key in ('STOPFIT', 'ALTCRIT', 'RAWJ'):
        f = own['frontiers'][key]
        lab = 'NOT_SEPARATED' if f.get('reason') == 'NO_BRACKET' else f['label']
        ok &= c2[key]['label'].replace(pre, '') == lab and c2[key]['point'] == f.get('bracket')
        if f.get('bracket'):
            ok &= abs(c2[key]['benefit_mean'] - f['benefit_mean']) <= TOL and abs(c2[key]['benefit_se'] - f['benefit_se_agg']) <= TOL
    chk['R1|D2_comparisons_equal_verifier'] = bool(ok)
    pr = dd['D3']['per_reference']
    chk['R1|D3_per_reference_equal_verifier'] = all(
        x['benefit'].replace(pre, '') == own['comparisons'][r]['benefit_label'] and abs(x['mean'] - own['comparisons'][r]['mean']) <= TOL and
        x['recipient_gate_pass'] == own['comparisons'][r]['recipient_gate_pass'] for r, x in pr.items())
    fcj = json.loads((RES / 'claims' / 'RFA_R1' / 'FAMILY_CHANGE_vs_RF_FULL_corrected.json').read_text())['reading']
    fc0 = json.loads((RES / 'claims' / 'RFA_R1' / 'FAMILY_CHANGE_vs_RF_FULL.json').read_text())['reading']
    fo = own['FC_R1']
    chk['FC_R1|corrected_label_mean_state_equal_verifier'] = (fcj['label'] == pre + fo['label'] and abs(fcj['benefit_mean'] - fo['mean']) <= TOL and
                                                             fcj['spend_state'] == fo['spend_state'] and
                                                             sorted(fcj['secondary_resolved_higher']) == fo['secondary_higher'])
    res['FC_R1_first_file'] = {'label': fc0.get('label'), 'benefit_mean': fc0.get('benefit_mean'), 'note': 'deviation 9: mean read from the wrong field'}
    # ---- half-budget grid validity: the verifier's seed-equality counts at fraction 0.5
    fd = json.loads((RES / 'claims' / 'RFA_R1' / 'FALSIFIER_DETAIL.json').read_text())
    res['lc_validity'] = {'recorded_R1': {k: fd['lc_grid_validity'].get(k) for k in ('label', 'effective', 'n_checks')},
                          'verifier_seed_equalities_R1_LC': {str(v): grids['RFA1_R1_LC'][v]['seed_equalities'] for v in (0, 1)},
                          'verifier_seed_equalities_v1_LC': {str(v): grids['RFA1_LC'][v]['seed_equalities'] for v in (0, 1)}}
    half_eq = sum(x[0] for v in (0, 1) for kk, x in grids['RFA1_R1_LC'][v]['seed_equalities'].items() if kk.endswith('frac0.5'))
    chk['LC|R1_half_budget_seed_dependent_recipes_all_differ (0 equalities)'] = half_eq == 0 and fd['lc_grid_validity']['label'] == 'READER_SEEDS_EFFECTIVE'
    # ---- v1 D2R with the verifier's merged MAIN (wave-2 entries) + ROB entries
    d2r = json.loads((RES / 'claims' / 'RFA_v1_D2R' / 'D2R.json').read_text())
    g2 = json.loads((HERE.parent / 'out' / 'V2_GRID.json').read_text())['inputs']['entries']
    rob = {}
    for comp, views in ent['RFA1_ROB'].items():
        rob[comp] = {}
        for view, e in views.items():
            reg = (g2.get(comp) or {}).get(view) or {'cells': {}, 'cell_se': {}}
            rob[comp][view] = {'cells': {**reg['cells'], **e['cells']}, 'cell_se': {**reg['cell_se'], **e['cell_se']}}
    refit_ok = (d2r['refit_effectiveness'] or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    own_d2r = {c: B.d2r_comparator(rob[c], refit_ok) for c in rob}
    res['own_D2R'] = own_d2r
    per = (d2r['d2r'] or {}).get('per_comparator') or (d2r['d2r'] or {}).get('comparators') or {}
    agree = {c: (per[c].get('label') == own_d2r[c]['label']) for c in per if c in own_d2r}
    res['D2R_compare'] = {'recorded_label': d2r['d2r'].get('label'), 'per_comparator_agree': agree,
                          'recorded_comparators': d2r.get('comparators'), 'refit_label': (d2r['refit_effectiveness'] or {}).get('label')}
    chk['D2R|per_comparator_labels_equal_verifier'] = bool(agree) and all(agree.values())
    sc_rec, rf_rec = sorted(d2r['d2r'].get('sign_changes') or []), sorted(d2r['d2r'].get('recipient_failures') or [])
    sc_me = sorted(c for c in per if own_d2r.get(c, {}).get('sign_change'))
    rf_me = sorted(c for c in per if own_d2r.get(c, {}).get('recipient_failed'))
    res['D2R_compare'].update({'sign_changes': {'recorded': sc_rec, 'verifier': sc_me}, 'recipient_failures': {'recorded': rf_rec, 'verifier': rf_me},
                               'robustness_means': {c: (per[c].get('robustness_mean'), own_d2r[c].get('robustness_mean')) for c in per if c in own_d2r}})
    chk['D2R|sign_changes_and_recipient_failures_equal_verifier'] = sc_rec == sc_me and rf_rec == rf_me
    chk['D2R|robustness_means_equal_verifier (abs <= 1e-12)'] = all(
        per[c].get('robustness_mean') is not None and abs(per[c]['robustness_mean'] - own_d2r[c]['robustness_mean']) <= TOL
        for c in per if c in own_d2r and own_d2r[c]['label'] != 'NOT_RUN')
    res['_rob_entries'] = rob
    # ---- frozen-rules legs (rfa.rules unchanged) on the verifier's substituted inputs
    from rfa import claims as CL
    from rfa import rfclaims as RFC
    from rfa import rules as RU
    labels_of = lambda x: {k: (x.get(k) or {}).get('label') for k in ('D1', 'D2', 'D3', 'D4', 'LC')} | {'class': x['outcome_class']['label']}
    frozen = {}
    for name, plan_file, bank, lc_bank, dec_dir in (('R1', 'PLAN_RFA_R1.json', 'RFA1_R1_MAIN', 'RFA1_R1_LC', 'RFA_R1'),
                                                    ('v1_rebuilt', 'PLAN_RFA_v1.json', 'RFA1_MAIN', 'RFA1_LC', 'RFA_v1_rebuilt')):
        Dx = json.loads((RES / 'claims' / dec_dir / 'DECISIONS.json').read_text())
        plan = json.loads((RES / 'claims' / plan_file).read_text())
        CL.reader_budget_flags = (lambda flags: (lambda bank_, cand_, refs=CL.REFS: dict(flags)))(Dx['reader_budget'])
        grid = CL.load_grid(bank)
        lc_grid = CL.load_grid(lc_bank)
        gv_lc = (Dx.get('lc_grid_validity') or {}).get('effective')
        rec = RFC.build(plan, grid, bank, lc_grid=lc_grid, lc_bank=lc_bank, gv_effective=gv_lc)
        rec['grid_valid'] = Dx['grid_validity']['effective']
        rep = RU.decisions(copy.deepcopy(rec))
        sub = copy.deepcopy(rec)
        main_ent = ent['RFA1_R1_MAIN'] if name == 'R1' else g2
        main_nt = nt['RFA1_R1_MAIN'] if name == 'R1' else json.loads((HERE.parent / 'out' / 'V2_GRID.json').read_text())['inputs']['nontable']
        sub['grid'] = {k: main_ent[k] for k in rec['grid']}
        sub['nolookup'] = {k: (main_nt[k] if k in main_nt else rec['nolookup'][k]) for k in rec['nolookup']}
        lce = ent['RFA1_R1_LC'] if name == 'R1' else ent['RFA1_LC']
        if (sub.get('learning_curve') or {}).get('half'):
            sub['learning_curve']['half'] = {c: {view: {'cells': {k.split('@')[0]: x for k, x in lce[c][view]['cells'].items() if k.endswith('@0.5')},
                                                       'cell_se': {k.split('@')[0]: x for k, x in lce[c][view]['cell_se'].items() if k.endswith('@0.5')}}
                                                for view in lce[c]} for c in rec['learning_curve']['half'] if c in lce}
        mine_dec = RU.decisions(sub)
        frozen[name] = {'recorded': labels_of(Dx['decisions']), 'replay': labels_of(rep), 'on_verifier_inputs': labels_of(mine_dec)}
        chk[f'frozen|{name}|replay_equals_record'] = labels_of(rep) == labels_of(Dx['decisions'])
        chk[f'frozen|{name}|verifier_inputs_give_recorded_labels'] = labels_of(mine_dec) == labels_of(Dx['decisions'])
    v1o = json.loads((RES / 'claims' / 'RFA_v1' / 'DECISIONS.json').read_text())['decisions']
    v1r = json.loads((RES / 'claims' / 'RFA_v1_rebuilt' / 'DECISIONS.json').read_text())
    chk['v1_rebuilt|labels_equal_first_record'] = labels_of(v1o) == labels_of(v1r['decisions'])
    chk['v1_rebuilt|lc_validity_is_half_budget_READER_SEEDS_EFFECTIVE'] = (v1r.get('lc_grid_validity') or {}).get('label') == 'READER_SEEDS_EFFECTIVE'
    res['v1_rebuilt'] = {'lc_grid_validity': {k: (v1r.get('lc_grid_validity') or {}).get(k) for k in ('label', 'effective', 'n_checks')},
                         'labels': labels_of(v1r['decisions'])}
    # FC_R1 through the campaign function on the verifier's entry
    fc_frozen = RFC.family_change({'paired': {'RF1_FULL': {'RF_FULL': ent['RFA1_R1_MAIN']['RF_FULL']}}}, 'RF1_FULL', 'RF_FULL',
                                  CL.costs_between('FAM/RF1_FULL', 'FAM/RF_FULL'))
    chk['frozen|FC_R1_on_verifier_entry_equals_corrected_file'] = fc_frozen['label'] == fcj['label'] and abs(fc_frozen['benefit_mean'] - fcj['benefit_mean']) <= TOL
    # D2R through the frozen rules on the verifier's merged entries
    d2r_rec = {'grid_valid': True, 'release': 'RF_FULL', 'refit': d2r['refit_effectiveness'], 'comparators': d2r['comparators'],
               'rob': {k: v for k, v in rob.items()}}
    d2r_mine = RU.d2r(d2r_rec)
    frozen['D2R'] = {'recorded': d2r['d2r'].get('label'), 'on_verifier_inputs': d2r_mine.get('label')}
    chk['frozen|D2R_on_verifier_inputs_equals_record'] = d2r_mine.get('label') == d2r['d2r'].get('label')
    res['frozen_rules'] = frozen
    del res['_rob_entries']
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v3_main.py', 'vlib.py', 'vcore.py', 'vagg.py', 'vagg2.py', 'v2_grid.py', 'v2_usefulness.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'cpu_s': res['cpu_s'],
                      'own_R1': {k: own[k] for k in ('D1', 'D2', 'D3', 'D4')}, 'FC_R1': own['FC_R1']['label'], 'frozen': frozen}, indent=1, default=str))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

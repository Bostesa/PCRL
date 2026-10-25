#!/usr/bin/env python3
"""RFA1 independent verifier, wave 1, check 3: tests of the prepared later-wave code (vlib) on INVENTED data only.

(a) the 28 usefulness conditions: vlib vs the campaign evaluator path rfa.evaluate.utility_of (-> ohn.evaluate.complete_eval ->
    ihs.evaluate.utility) on invented MODEL_SELECT-like labels, invented U0 / Ucal references for two versions, a deterministic
    release, a randomized (3-draw) release, an edge-case release (argmax ties, zero probability on the true class) and a release
    that fails exactly known clauses.
(b) the reader grid: the campaign's rfa.devreaders.run_bank on invented releases (one cell S0R1k0, two members, a temporary cache
    and output directory under roles/verifier/out/invented_grid); then vlib recomputes reader roles, cache keys, winners, SCORE
    AUCs (winners and all 16 recipes), household-bootstrap draws and the paired per-cell differences / SEs from the saved files.
(c) bounded refit: vlib.refit of every (member, view) winner and of all 16 recipes on one view, compared with the cached scores
    and the grid's SCORE AUCs.

Run:  <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v1_prepare_test.py
Writes roles/verifier/out/V1_PREPARE_TEST.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vguard  # noqa: E402

vguard.install()
sys.path.insert(1, str(ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

T0 = time.process_time()
OUT = HERE.parent / 'out' / 'V1_PREPARE_TEST.json'
GRID_DIR = HERE.parent / 'out' / 'invented_grid'


def dirichlet_rows(rng, n, k, conc, bias_col=None, bias=None):
    a = np.ones((n, k)) * conc
    if bias_col is not None:
        a[np.arange(n), bias_col] += bias
    g = rng.gamma(a)
    return g / g.sum(1, keepdims=True)


def test_usefulness():
    from rfa import evaluate as RE
    rng = np.random.default_rng(1001)
    H = 420
    sizes = rng.integers(1, 4, H)
    unit = np.repeat(np.arange(H) * 3 + 11, sizes)
    n = len(unit)
    S = {'row_id': np.arange(n) + 500, 'unit': unit, 'sex': rng.integers(0, 2, n), 'income': rng.integers(0, 2, n),
         'occupation': rng.integers(0, 5, n)}
    labels = {t: S[t] for t in V.TASKS}
    res, per_v_mine, per_v_camp = {}, {}, {}
    for v in (0, 1):
        r2 = np.random.default_rng(2000 + v)
        mix = lambda a, b, w: (1 - w) * a + w * b
        u0 = {'income': dirichlet_rows(r2, n, 2, 1.0, S['income'], 2.0), 'occupation': dirichlet_rows(r2, n, 5, 0.7, S['occupation'], 2.5)}
        refs = {'U0': u0, 'Ucal': {t: mix(u0[t], dirichlet_rows(r2, n, V.K[t], 1.0), 0.01) for t in V.TASKS}}
        others = {nm: dirichlet_rows(r2, n, 5, 1.0) for nm in ('T_TREE', 'TASK_REFIT', 'JOINT')}
        rel = {'income': mix(refs['U0']['income'], dirichlet_rows(r2, n, 2, 1.0), 0.004),
               'occupation': mix(refs['U0']['occupation'], dirichlet_rows(r2, n, 5, 1.0), 0.004)}
        stack = {'income': np.stack([mix(rel['income'], dirichlet_rows(r2, n, 2, 1.0), 0.03) for _ in range(3)]),
                 'occupation': np.stack([mix(rel['occupation'], dirichlet_rows(r2, n, 5, 1.0), 0.03) for _ in range(3)])}
        edge = {'income': rel['income'].copy(), 'occupation': rel['occupation'].copy()}
        edge['occupation'][:15] = 0.2                                   # exact argmax ties (first index wins)
        z = edge['income'][15:30]; z[np.arange(15), S['income'][15:30]] = 0.0; z[np.arange(15), 1 - S['income'][15:30]] = 1.0
        edge['income'][15:30] = z                                       # zero probability on the true class (NLL floor 1e-12)
        bad = {'income': mix(rel['income'], 1 - rel['income'], 0.35), 'occupation': rel['occupation']}
        out_v = {}
        for name, R in (('deterministic', rel), ('randomized_3_draws', stack), ('edge_ties_and_zero_prob', edge), ('known_failures', bad)):
            u = RE.utility_of(v, R, S, (refs, others, None))[0]['utility']
            ps = RE.point_status(u)
            mine = V.usefulness_version(labels, R, refs, S['sex'], S['unit'])
            cmpr = V.compare_with_campaign_utility(mine, u)
            same_fail = sorted(mine['failed']) == sorted(ps['failed'])
            out_v[name] = {**cmpr, 'campaign_point_pass_12_plus_2': ps['point_pass_12_plus_2'], 'verifier_all_pass': mine['all_pass'],
                           'failed_agree': same_fail, 'verifier_failed': mine['failed'], 'draws': mine['draws'],
                           'max_abs_se_mc_diff': max(abs(float(c.get('se_mc', 0.0)) - mine['clauses'][f"{c['task']}|{c['reference']}|{c['metric']}"]['se_mc'])
                                                     for c in u['clauses']) if name == 'randomized_3_draws' else 0.0}
            if name == 'deterministic':
                per_v_mine[v] = mine
                per_v_camp[v] = ps['point_pass_12_plus_2']
        res[f'V{v}'] = out_v
    gg = V.g1(per_v_mine)
    res['G1'] = {**gg, 'campaign_G1': bool(all(per_v_camp.values())), 'agree': gg['G1'] == bool(all(per_v_camp.values()))}
    ok = all(x['agree'] and x['failed_agree'] and x['max_abs_se_mc_diff'] < 1e-12 for v in ('V0', 'V1') for x in res[v].values()) and \
        res['G1']['agree'] and gg['n_conditions'] == 28 and gg['G1'] and \
        all(res[v]['known_failures']['verifier_failed'] for v in ('V0', 'V1'))
    res['paired_mean_vs_difference_of_means_max_abs'] = max(abs(c['paired_mean_minus_value']) for pv in per_v_mine.values()
                                                             for c in pv['clauses'].values()) if all('paired_mean_minus_value' in c for pv in per_v_mine.values() for c in pv['clauses'].values()) else None
    return ok, res


def invented_grid_inputs(n_house=300, seed=4242):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_house) * 5 + 7, 2)
    n = len(unit)
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * sex; z[:, 3] -= 0.3 * sex
    occA = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = z.copy(); z2[:, 1] -= 0.6 * sex
    occB = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-(rng.normal(size=n) + 0.3 * sex)))
    incA = np.column_stack((1 - t, t))
    levels = np.array([[0.89, 0.11], [0.76, 0.24], [0.57, 0.43], [0.45, 0.55], [0.31, 0.69], [0.13, 0.87]])
    code = np.clip(np.round(2.5 + rng.normal(size=n) + 0.4 * sex), 0, 5).astype(int)
    incB = levels[code]                                                # a finite (6-level) income wire: CATI categorical path
    return unit, sex, {'A': {'income': incA, 'occupation': occA}, 'B': {'income': incB, 'occupation': occB}}


def test_grid_and_refit():
    from rfa import devreaders as DR
    from ohn import common as OC
    unit, sex, bank = invented_grid_inputs()
    if GRID_DIR.exists():
        shutil.rmtree(GRID_DIR)
    GRID_DIR.mkdir(parents=True)
    caches = {'own': GRID_DIR / 'cache', 'source': None, 'source_r3': None}
    spec = {'bank_id': 'INVENTED', 'members': {'A': {}, 'B': {}}, 'candidates': ['B'], 'comparators': ['A'], 'reps': ['R1'],
            'reader_seeds': [0], 'fractions': [1.0]}
    t1 = time.process_time()
    rec = DR.run_bank(bank, sex, unit, spec, 0, GRID_DIR / 'S0', caches=caches)
    grid_cpu = time.process_time() - t1
    out = {'campaign_run_bank_cpu_s': grid_cpu, 'campaign_reader_fits': rec['stats']['reader_fits']}
    ok = {}
    # roles, folds, seeds
    for tag in ('RFA1_devgrid_R1', 'RFA1_devgrid_R2', 'RFA1_devgrid_LC'):
        k = 4 if tag.endswith('LC') else 10
        ok[f'household_folds_equal_campaign_{tag}'] = bool(np.array_equal(V.household_folds(unit, k, tag), OC.household_folds(unit, k, tag)))
    mine_roles = V.reader_roles(unit, 'R1')
    camp_roles = DR.roles_of(unit, 'R1')
    ok['reader_roles_equal_campaign'] = all(np.array_equal(mine_roles[r], camp_roles[r]) for r in mine_roles)
    ok['lc_half_equal_campaign'] = bool(np.array_equal(V.reader_roles(unit, 'R1', 0.5)['FIT'], DR.lc_fit(unit, camp_roles['FIT'], 0.5)))
    ok['reader_seed_equal_campaign'] = all(V.reader_seed(v, k) == DR.reader_seed(v, k) for v in (0, 1) for k in range(4))
    # saved files
    grid = json.loads((GRID_DIR / 'S0' / 'GRID.json').read_text())
    with np.load(GRID_DIR / 'S0' / 'BOOT.npz') as z:
        boot = {k: z[k].copy() for k in z.files}
    sc = mine_roles['SCORE']; y = sex[sc]
    W = V.boot_weights(unit[sc], 0, 'R1')
    rs = V.reader_seed(0, 0)
    sx = V.arr_sha(np.asarray(sex))
    cell = V.cell_name(0, 'R1', 0)
    entries, rec_cmp = {}, {}
    win_ok = auc_ok = boot_ok = perrec_ok = key_ok = True
    worst = {'score_auc': 0.0, 'boot': 0.0, 'per_recipe': 0.0}
    for name, rel in bank.items():
        ent = {}
        for view in V.VIEWS:
            p = caches['own'] / f"{V.cache_key('R1', rs, 1.0, V.view_hash(view, rel['income'], rel['occupation']), sx)}.npz"
            key_ok &= p.is_file()
            ent[view] = V.read_cache_entry(p)
        entries[name] = ent
        w = V.winners({view: {n: ent[view][n][1] for n in V.RECIPES} for view in V.VIEWS})
        for view in V.VIEWS:
            g = grid['cells'][cell][name][view]
            vw, rname = w[view]['reader_key'].split('__', 1)
            s = ent[vw][rname][0]
            a = V.auc(y, s)
            bd = V.weighted_auc(y, s, W)
            win_ok &= (w[view]['winner'] == g['winner'] and abs(w[view]['val_auc'] - g['val_auc']) == 0.0 and
                       w[view]['near_tie'] == g['near_tie'] and (w[view]['gap_to_runner_up'] == g['gap_to_runner_up']))
            worst['score_auc'] = max(worst['score_auc'], abs(a - g['score_auc']))
            worst['boot'] = max(worst['boot'], float(np.max(np.abs(bd - boot[f'{cell}__{name}__{view}']))))
            pr = {n: V.auc(y, ent[view][n][0]) for n in V.RECIPES}
            worst['per_recipe'] = max(worst['per_recipe'], max(abs(pr[n] - g['per_recipe_score_auc'][n]) for n in V.RECIPES))
            rec_cmp[f'{name}/{view}'] = {'winner': g['winner'], 'verifier_winner': w[view]['winner'], 'score_auc_grid': g['score_auc'],
                                         'score_auc_verifier': a, 'near_tie': g['near_tie']}
    ok['cache_keys_reproduced'] = bool(key_ok)
    ok['winners_reproduced_from_stored_validation_aucs'] = bool(win_ok)
    ok['winner_score_auc_reproduced (abs < 1e-12)'] = worst['score_auc'] < 1e-12
    ok['bootstrap_draws_reproduced (abs < 1e-12)'] = worst['boot'] < 1e-12
    ok['per_recipe_score_auc_reproduced (abs < 1e-12)'] = worst['per_recipe'] < 1e-12
    out['worst_abs_diffs'] = worst
    out['cells'] = rec_cmp
    # paired per-cell differences and SEs (candidate B vs comparator A)
    pd_worst = 0.0
    for view in V.VIEWS:
        ga, gb = grid['cells'][cell]['B'][view], grid['cells'][cell]['A'][view]
        wB = V.winners({vv: {n: entries['B'][vv][n][1] for n in V.RECIPES} for vv in V.VIEWS})[view]
        wA = V.winners({vv: {n: entries['A'][vv][n][1] for n in V.RECIPES} for vv in V.VIEWS})[view]
        sB = entries['B'][wB['reader_key'].split('__')[0]][wB['reader_key'].split('__', 1)[1]][0]
        sA = entries['A'][wA['reader_key'].split('__')[0]][wA['reader_key'].split('__', 1)[1]][0]
        d, se = V.paired_cell(view, V.auc(y, sB), V.auc(y, sA), V.weighted_auc(y, sB, W), V.weighted_auc(y, sA, W))
        gp = grid['paired']['B']['A'][view]
        pd_worst = max(pd_worst, abs(d - gp['cells'][cell]), abs(se - gp['cell_se'][cell]))
        out.setdefault('paired', {})[view] = {'grid_d': gp['cells'][cell], 'verifier_d': d, 'grid_se': gp['cell_se'][cell], 'verifier_se': se}
        del ga, gb
    ok['paired_differences_and_SEs_reproduced (abs < 1e-12)'] = pd_worst < 1e-12
    out['paired_worst_abs_diff'] = pd_worst
    # (c) bounded refit with sklearn: every (member, view) winner, plus all 16 recipes on member B's pair view
    val = mine_roles['VALIDATION']; fit = mine_roles['FIT']
    refits, rf_worst = {}, {'scores': 0.0, 'val_auc': 0.0, 'score_auc': 0.0}
    t2 = time.process_time()
    todo = []
    for name in bank:
        w = V.winners({vv: {n: entries[name][vv][n][1] for n in V.RECIPES} for vv in V.VIEWS})
        for view in V.VIEWS:
            vw, rname = w[view]['reader_key'].split('__', 1)
            todo.append((name, vw, rname, f'winner {name}/{view}'))
    todo += [('B', 'pair', n, 'all16 B/pair') for n in V.RECIPES]
    seen = set()
    for name, vw, rname, why in todo:
        if (name, vw, rname) in seen:
            continue
        seen.add((name, vw, rname))
        s, va, _ = V.refit(rname, vw, bank[name]['income'], bank[name]['occupation'], sex, fit, val, rs)
        stored, info = entries[name][vw][rname]
        d_s = float(np.max(np.abs(s[sc] - stored)))
        d_v = abs(va - info['val_auc'])
        d_a = abs(V.auc(y, s[sc]) - V.auc(y, stored))
        rf_worst = {'scores': max(rf_worst['scores'], d_s), 'val_auc': max(rf_worst['val_auc'], d_v), 'score_auc': max(rf_worst['score_auc'], d_a)}
        refits[f'{name}/{vw}/{rname}'] = {'why': why, 'max_abs_score_diff': d_s, 'val_auc_diff': d_v, 'score_auc_diff': d_a,
                                          'bitwise_scores': bool(np.array_equal(s[sc], stored))}
    out['refit_cpu_s'] = time.process_time() - t2
    out['refits'] = refits
    out['refit_worst'] = rf_worst
    ok['refit_scores_reproduce_cached_scores (abs < 1e-9)'] = rf_worst['scores'] < 1e-9
    ok['refit_validation_auc_reproduced (abs < 1e-12)'] = rf_worst['val_auc'] < 1e-12
    out['refit_bitwise_count'] = sum(r['bitwise_scores'] for r in refits.values())
    out['refit_count'] = len(refits)
    return all(ok.values()), {**out, 'checks': ok}


def main():
    res = {'schema': 'RFA1-verifier-prepare-test-v1', 'checks': {}, 'files_sha256': {}}
    for f in ('rfa/evaluate.py', 'rfa/devreaders.py', 'rfa/readerextra.py', 'ohn/evaluate.py', 'ihs/evaluate.py', 'scm_bench/selection.py',
              'scm_bench/common.py', 'scm_bench/readers.py', 'ffv/readers.py', 'ffv/features.py', 'task_conditioned_representation/attacks.py',
              'task_conditioned_representation/inference.py', 'results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/vlib.py'):
        res['files_sha256'][f] = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
    t = time.process_time()
    ok_a, res['usefulness'] = test_usefulness()
    res['usefulness_cpu_s'] = time.process_time() - t
    res['checks']['a_usefulness_28_conditions_match_campaign_evaluator'] = ok_a
    ok_bc, res['grid'] = test_grid_and_refit()
    res['checks']['b_c_grid_recompute_and_bounded_refit'] = ok_bc
    res['guard_refused_opens'] = list(vguard.REFUSED)
    res['all_pass'] = all(res['checks'].values())
    res['cpu_s'] = time.process_time() - T0
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'checks': res['checks'], 'grid_checks': res['grid']['checks'], 'cpu_s': res['cpu_s'],
                      'refit_worst': res['grid']['refit_worst'], 'worst': res['grid']['worst_abs_diffs'],
                      'refit_bitwise': [res['grid']['refit_bitwise_count'], res['grid']['refit_count']],
                      'usefulness': {v: {k: {kk: x[kk] for kk in ('agree', 'max_abs_diff', 'failed_agree', 'verifier_failed')}
                                         for k, x in res['usefulness'][v].items()} for v in ('V0', 'V1')}}, indent=1, default=str))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

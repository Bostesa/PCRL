"""Verifier check 3: MODEL_SELECT utility clauses, subgroup guards, expenditures, cross-fitted proxies, nomination (own code).

For EVERY effective release of both seeds (RELEASE_TABLES.npz, verified bitwise by check 2; routing verified by check 1) and the
frozen NEURAL_RAW / H_LJ outputs: the 12 original point clauses against U0 / Ucal, the within-SEX occupation Brier guards against
U0, occupation NLL and its expenditure against C0, by-SEX summaries, and the cross-fitted pair-reader proxy with OWN readers
(sklearn LogisticRegression / HistGradientBoostingClassifier with the source settings, own finite scaler, own exact-vector
lookup, own household split from the registered seed, own Mann-Whitney AUC). Then the conservative nomination rule and its
fallback are re-applied. No campaign module is imported. MODEL_SELECT labels and SEX are read (permitted for checks 3-4).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

CAPS = (('accuracy_loss', 0.010), ('nll_excess', 0.010), ('brier_excess', 0.005))
MODEL_SEED_BASE = 2026101205       # source reader seed base (task_conditioned_representation.attacks, read, not imported)
NOM_OCC = 0.008; GUARD = 0.005
NLL_GRID = tuple(i / 1000 for i in range(11))


class Scaler:
    def fit(self, x):
        self.mu = x.mean(axis=0); sd = x.std(axis=0); self.sd = np.where(sd > 0, sd, 1.0); return self

    def tr(self, x):
        return (x - self.mu) / self.sd


def reader_fit(x, y, name, seed):
    sc = Scaler().fit(x)
    if name == 'OUT_HGB':
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed)
    else:
        m = LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(sc.tr(x), y)
    assert list(m.classes_) == [0, 1]
    return sc, m


def reader_score(sc_m, x):
    sc, m = sc_m
    return m.predict_proba(sc.tr(x))[:, 1].astype(np.float64)


def lookup_fit(keys, y):
    tab = {}
    for kk, v in zip(keys, y):
        a, b = tab.get(kk, (0, 0)); tab[kk] = (a + int(v), b + 1)
    return float(np.mean(y)), tab


def lookup_score(model, keys):
    pi, tab = model
    return np.array([(tab[kk][0] + 2.0 * pi) / (tab[kk][1] + 2.0) if kk in tab else pi for kk in keys])


def out_kind(p):
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, np.argmax(p, axis=1).astype(np.float64)))


def proxy(inc, occ, sex, halves, seed):
    X = np.column_stack((out_kind(inc), out_kind(occ)))
    keys = [r.tobytes() for r in np.ascontiguousarray(np.column_stack((inc, occ)), dtype=np.float64)]
    keys = np.array(keys, dtype=object)
    res = {}
    for name in ('OUT_LR', 'OUT_HGB', 'LOOKUP'):
        aucs = []
        for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
            if name == 'LOOKUP':
                s = lookup_score(lookup_fit(list(keys[tr]), sex[tr]), list(keys[te]))
            else:
                s = reader_score(reader_fit(X[tr], sex[tr], name, MODEL_SEED_BASE + seed), X[te])
            aucs.append(V.auc_mw(sex[te], s))
        res[name] = {'directions': aucs, 'mean': float(np.mean(aucs))}
    res['arm_proxy'] = max(res[n]['mean'] for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP'))
    return res


def measure(inc, occ, lab, sex, refs, c0_nll, halves, seed):
    own = {'income': V.task_vectors(lab['income'], inc, 2), 'occupation': V.task_vectors(lab['occupation'], occ, 5)}
    cl = {}
    for t in V.TASKS:
        for ref in ('U0', 'Ucal'):
            r = V.task_vectors(lab[t], refs[ref][t], V.KCLS[t])
            vals = {'accuracy_loss': r['accuracy'].mean() - own[t]['accuracy'].mean(), 'nll_excess': own[t]['nll'].mean() - r['nll'].mean(),
                    'brier_excess': own[t]['brier'].mean() - r['brier'].mean()}
            for met, cap in CAPS:
                cl[f'{t}|{ref}|{met}'] = {'value': float(vals[met]), 'cap': cap, 'pass': bool(vals[met] <= cap)}
    u0 = V.task_vectors(lab['occupation'], refs['U0']['occupation'], 5)
    sub = {f'sex{g}': float((own['occupation']['brier'] - u0['brier'])[sex == g].mean()) for g in (0, 1)}
    occ_nll = float(own['occupation']['nll'].mean())
    return {'clauses': cl, 'point_pass_12': all(c['pass'] for c in cl.values()), 'max_normalized_12': max(c['value'] / c['cap'] for c in cl.values()),
            'subgroup': sub, 'occupation_nll': occ_nll, 'expenditure': (occ_nll - c0_nll) if c0_nll is not None else 0.0,
            'by_sex': {f'sex{g}': {'occupation_nll': float(own['occupation']['nll'][sex == g].mean()),
                                   'occupation_brier': float(own['occupation']['brier'][sex == g].mean())} for g in (0, 1)},
            'task_summary': {t: {m: float(own[t][m].mean()) for m in ('accuracy', 'nll', 'brier')} for t in V.TASKS},
            'proxy': proxy(inc, occ, sex, halves, seed)}


def compare(mine, rec):
    """Field-by-field comparison with the SELECTION_LOCK record; returns (n_exact, n_total, max_abs_diff, max_proxy_diff, problems)."""
    pairs = []
    rc = {f"{c['task']}|{c['reference']}|{c['metric']}": c for c in rec['clauses']}
    probs = []
    if set(rc) != set(mine['clauses']):
        probs.append('clause key sets differ')
    for kk, c in mine['clauses'].items():
        pairs.append((c['value'], rc[kk]['value']))
        if c['pass'] != rc[kk]['point_pass']:
            probs.append(f'clause pass differs {kk}')
    for g in (0, 1):
        pairs.append((mine['subgroup'][f'sex{g}'], rec['subgroup_occupation_brier_excess_vs_U0'][f'sex{g}']))
        for m in ('occupation_nll', 'occupation_brier'):
            pairs.append((mine['by_sex'][f'sex{g}'][m], rec['by_sex'][f'sex{g}'][m]))
    pairs.append((mine['occupation_nll'], rec['occupation_nll']))
    pairs.append((mine['expenditure'], rec['occupation_nll_expenditure_vs_C0']))
    pairs.append((mine['max_normalized_12'], rec['max_normalized_12']))
    for t in V.TASKS:
        for m in ('accuracy', 'nll', 'brier'):
            pairs.append((mine['task_summary'][t][m], rec['task_summary'][t][m]))
    if mine['point_pass_12'] != rec['point_pass_12']:
        probs.append('point_pass_12 differs')
    pdiff = 0.0
    for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP'):
        for a, b in zip(mine['proxy'][n]['directions'], rec['proxy'][n]['directions']):
            pdiff = max(pdiff, abs(a - b))
    pdiff = max(pdiff, abs(mine['proxy']['arm_proxy'] - rec['proxy']['arm_proxy']))
    exact = sum(a == b for a, b in pairs)
    mad = max(abs(a - b) for a, b in pairs)
    return exact, len(pairs), mad, pdiff, probs


def nomination_conditions(m):
    out = [(k_, c['value'], c['cap']) for k_, c in m['clauses'].items()]
    for ref in ('U0', 'Ucal'):
        out.append((f'occupation|{ref}|nll_excess<=0.008', m['clauses'][f'occupation|{ref}|nll_excess']['value'], NOM_OCC))
    for g in (0, 1):
        out.append((f'subgroup|sex{g}|occupation_brier_excess_vs_U0', m['subgroup'][f'sex{g}'], GUARD))
    return out


def main():
    t0 = time.process_time()
    sel = V.jload(V.RES / 'SELECTION_LOCK.json')
    tl = V.jload(V.RES / 'TRAJECTORY_COMPLETE_LOCK.json')
    fbm = V.jload(V.HYB_RES / 'FROZEN_BANK_MANIFEST.json')['bank_files_sha256']
    F = V.export('MODEL_SELECT', 'features'); lab = {t: V.export('MODEL_SELECT', t)[t].astype(np.int64) for t in V.TASKS}
    sx = V.export('MODEL_SELECT', 'sex')
    for t in V.TASKS:
        z = V.export('MODEL_SELECT', t); assert np.array_equal(z['row_id'], F['row_id']) and np.array_equal(z['unit'], F['unit'])
    assert np.array_equal(sx['row_id'], F['row_id']) and np.array_equal(sx['unit'], F['unit'])
    sex = sx['sex'].astype(np.int64); unit = F['unit']
    groups = np.unique(unit); perm = np.random.default_rng(V.seed_for('SCM1proxysplit')).permutation(len(groups))
    hA = np.isin(unit, groups[perm[:len(groups) // 2]]); halves = (hA, ~hA)
    out = {'check': 3, 'title': 'MODEL_SELECT utility, guards, expenditures, proxies and nomination', 'per_seed': {},
           'split': {'households': int(len(groups)), 'half_A_persons': int(hA.sum()), 'half_B_persons': int((~hA).sum()),
                     'seed': V.seed_for('SCM1proxysplit')}, 'discrepancies': []}
    mine_all = {}
    for s in V.SEEDS:
        tdir = V.PRIV / 'runs' / f'TRAJ_{s}'
        for f in ('TRAJ_SEED.json', 'RELEASE_TABLES.npz'):
            assert V.sha_file(tdir / f) == tl['files_sha256'][f'S{s}'][f]
        TS = V.jload(tdir / 'TRAJ_SEED.json'); RTAB = V.npz(tdir / 'RELEASE_TABLES.npz')
        RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz'); assert np.array_equal(RT['row_id__MODEL_SELECT'], F['row_id'])
        bp = V.SRC / 'bank' / f'FROZEN_BANK_S{s}.npz'; assert V.sha_file(bp) == fbm[f'S{s}']
        BK = V.npz(bp); assert np.array_equal(BK['row_id__MODEL_SELECT'], F['row_id'])
        refs = {r: {t: BK[f'{r}__MODEL_SELECT__{t}'] for t in V.TASKS} for r in ('U0', 'Ucal')}
        init = RT['init__MODEL_SELECT']
        rids = sorted({k_.split('__')[0] for k_ in RTAB})

        def outs(rid):
            return RTAB[f'{rid}__v1'][RTAB[f'{rid}__group_of_initial'][init]], RTAB[f'{rid}__v2'][RTAB[f'{rid}__cell_of_initial'][init]]
        c0 = TS['checkpoints']['SCM']['C512']['release']
        c0_nll = float(V.task_vectors(lab['occupation'], outs(c0)[1], 5)['nll'].mean())
        mine = {}
        for rid in rids:
            inc, occ = outs(rid)
            mine[rid] = measure(inc, occ, lab, sex, refs, c0_nll, halves, s)
        for name in ('NEURAL_RAW', 'H_LJ'):
            mine[f'FROZEN_{name}'] = measure(BK[f'{name}__MODEL_SELECT__income'], BK[f'{name}__MODEL_SELECT__occupation'], lab, sex, refs, c0_nll, halves, s)
        mine_all[s] = mine
        # compare
        rec = sel['per_seed_measurements'][str(s)]
        stats = {'releases_own': len(rids), 'releases_recorded': len(rec), 'same_release_sets': set(rids) == set(rec),
                 'values_exact': 0, 'values_total': 0, 'max_abs_value_diff': 0.0, 'max_abs_proxy_diff': 0.0, 'problems': {}}
        for rid in rids + ['FROZEN_NEURAL_RAW', 'FROZEN_H_LJ']:
            r = rec[rid] if not rid.startswith('FROZEN_') else sel['frozen_measurements'][str(s)][rid[7:]]
            ex, tot, mad, pdiff, probs = compare(mine[rid], r)
            stats['values_exact'] += ex; stats['values_total'] += tot
            stats['max_abs_value_diff'] = max(stats['max_abs_value_diff'], mad); stats['max_abs_proxy_diff'] = max(stats['max_abs_proxy_diff'], pdiff)
            if probs:
                stats['problems'][rid] = probs
        # proxy exactness detail
        stats['proxy_direction_values_exact'] = int(sum(a == b for rid in rids for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP')
                                                        for a, b in zip(mine[rid]['proxy'][n]['directions'], rec[rid]['proxy'][n]['directions'])))
        stats['proxy_direction_values_total'] = len(rids) * 6
        stats['arm_proxy_argmax_reader_agrees'] = int(sum(max(('OUT_LR', 'OUT_HGB', 'LOOKUP'), key=lambda n: mine[rid]['proxy'][n]['mean'])
                                                          == max(('OUT_LR', 'OUT_HGB', 'LOOKUP'), key=lambda n: rec[rid]['proxy'][n]['mean']) for rid in rids))
        stats['c0_occupation_nll'] = c0_nll
        out['per_seed'][f'S{s}'] = stats
    # ---------------- nomination re-applied with own numbers
    cps = {s: V.jload(V.PRIV / 'runs' / f'TRAJ_{s}' / 'TRAJ_SEED.json')['checkpoints'] for s in V.SEEDS}
    grid = []
    for t in NLL_GRID:
        lab_t = f'N{t:.3f}'
        rid = {s: cps[s]['SCM'][lab_t]['release'] for s in V.SEEDS}
        conds = {s: nomination_conditions(mine_all[s][rid[s]]) for s in V.SEEDS}
        grid.append({'t': t, 'releases': rid, 'eligible': all(v <= lim for s in V.SEEDS for (_, v, lim) in conds[s]),
                     'worst_normalized': max(v / lim for s in V.SEEDS for (_, v, lim) in conds[s]),
                     'failed': {s: [n for (n, v, lim) in conds[s] if v > lim] for s in V.SEEDS},
                     'mean_proxy': float(np.mean([mine_all[s][rid[s]]['proxy']['arm_proxy'] for s in V.SEEDS])),
                     'expenditure': {s: mine_all[s][rid[s]]['expenditure'] for s in V.SEEDS},
                     'min_margin_to_limit': min(lim - v for s in V.SEEDS for (_, v, lim) in conds[s])})
    elig = [g for g in grid if g['eligible']]
    nom = min(elig, key=lambda g: (g['mean_proxy'], g['t'])) if elig else min(grid, key=lambda g: (g['worst_normalized'], g['t']))
    status = 'NOMINEE_SELECTED' if elig else 'INELIGIBLE_DIAGNOSTIC_FALLBACK'
    rg = sel['nomination']['grid']
    grid_cmp = []
    for a, b in zip(grid, rg):
        grid_cmp.append({'t': a['t'], 'eligible_equal': a['eligible'] == b['eligible'], 'worst_normalized_diff': abs(a['worst_normalized'] - b['worst_normalized']),
                         'mean_proxy_diff': abs(a['mean_proxy'] - b['mean_proxy']), 'releases_equal': [a['releases'][s] for s in V.SEEDS] == [b['releases'][str(s)] for s in V.SEEDS],
                         'failed_equal': [a['failed'][s] for s in V.SEEDS] == [b['failed_conditions'][str(s)] for s in V.SEEDS],
                         'expenditure_diff': max(abs(a['expenditure'][s] - b['expenditure'][str(s)]) for s in V.SEEDS)})
    # sorted order of eligible thresholds by mean proxy (how robust is the argmin)
    ranking = sorted(((g['mean_proxy'], g['t']) for g in elig))
    out['nomination'] = {'own_status': status, 'own_t_star': nom['t'], 'recorded_t_star': sel['nomination']['t_star'], 'recorded_status': sel['status'],
                         'own_eligible': [g['t'] for g in elig], 'recorded_eligible': sel['nomination']['eligible_thresholds'],
                         'own_K_star': {s: 512 - cps[s]['SCM'][f"N{nom['t']:.3f}"]['m'] for s in V.SEEDS}, 'recorded_K_star': sel['nomination']['K_star'],
                         'own_e_star': {s: mine_all[s][nom['releases'][s]]['expenditure'] for s in V.SEEDS}, 'recorded_e_star': sel['nomination']['e_star'],
                         'grid_comparison': grid_cmp, 'own_grid': grid,
                         'eligible_by_mean_proxy': ranking,
                         'runner_up_gap': (ranking[1][0] - ranking[0][0]) if len(ranking) > 1 else None}
    # what-if diagnostics (descriptive only): the historical nominee P1 and the margin of the binding guard
    out['bets_selection_side'] = {
        'P1_t0.010_not_nominated': nom['t'] != 0.010,
        'P2_ORD_NLLM_available_both_seeds': all(sel['matching']['ORD_NLLM_available'].values()),
        'P3_proxy_ORD_NLLM_minus_SCM_ge_0.020_both_seeds': {s: sel['roles'][str(s)]['ORD_NLLM']['proxy'] - sel['roles'][str(s)]['SCM']['proxy'] for s in V.SEEDS}}
    # per-release aggregates for check 4 (no person-level values)
    out['own_measurements'] = {str(s): {rid: {'expenditure': m['expenditure'], 'point_pass_12': m['point_pass_12'], 'proxy': m['proxy']['arm_proxy'],
                                              'max_normalized_12': m['max_normalized_12'], 'task_summary': m['task_summary'],
                                              'subgroup': m['subgroup'], 'proxy_detail': {n: m['proxy'][n]['directions'] for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP')}}
                                        for rid, m in mine_all[s].items()} for s in V.SEEDS}
    ok = all(st['same_release_sets'] and not st['problems'] and st['max_abs_value_diff'] <= 1e-12 and st['max_abs_proxy_diff'] <= 1e-9
             for st in out['per_seed'].values())
    ok &= (nom['t'] == sel['nomination']['t_star'] and status == sel['status'] and [g['t'] for g in elig] == sel['nomination']['eligible_thresholds']
           and all(g['eligible_equal'] and g['releases_equal'] and g['failed_equal'] for g in grid_cmp))
    out['status'] = 'PASS' if ok else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['code_under_test_imported'] = []
    V.write_json('check3_model_select.json', out)
    print(json.dumps({'status': out['status'], 'cpu_s': out['cpu_s'], 'per_seed': {k_: {kk: v for kk, v in st.items() if kk != 'problems'} | {'n_problem_releases': len(st['problems'])}
                                                                                   for k_, st in out['per_seed'].items()},
                      'nomination': {k_: v for k_, v in out['nomination'].items() if k_ not in ('grid_comparison', 'own_grid')},
                      'bets': out['bets_selection_side']}, indent=1, default=str))


if __name__ == '__main__':
    main()

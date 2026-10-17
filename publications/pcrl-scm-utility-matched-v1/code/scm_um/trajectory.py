"""Unit UM_TRAJ_S{s}: the fixed master trajectories of one model seed, their registered checkpoints and release tables
(PROTOCOL.md sections 3-4). Requires the committed ANALYSIS_LOCK.

Trajectories (one cached run per configuration): SCM (beta 1.0), ORD (the source task-only twin rule at beta 1.0, no target),
IBSI_g{0.5,1.0,2.0} (PUB_IBSI to 32 cells) and SHUF_r{0..4} (source shuffled-signal SCM, beta 1.0). `run_path_recorded`
mirrors `scm_bench.engine.run_path` line for line and additionally records every merge and the engine's cumulative spending.

Historical reproduction gate (before any additional prefix is admitted): (a) the recorded run's final merge map, step count,
stop reason and spending equal `engine.run_path` with the same arguments; (b) the prefix at the historical cell count
reproduces the historical ARM_TABLES bitwise; (c) its outputs equal the historical ARM_OUTPUTS on the four output roles.

Access: MODEL_FIT task labels and SEX enter only through the admitted source statistics (within-SEX guard law, out-of-fold
propensity), exactly as in the source fit. No MODEL_SELECT, ATTACK or DEV_REPORT label is read. Outputs (private):
TRAJECTORIES.npz (merge sequences and spending), RELEASE_TABLES.npz (one table set per effective release), TRAJ_SEED.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from acs_replication import trees
from scm_bench import engine as E
from . import common as C

BETA = 1.0
GAMMAS = (0.5, 1.0, 2.0)
N_SHUF = 5
NLL_GRID = tuple(i / 1000 for i in range(11))     # t = k/1000
CNT_GRID = (512, 448, 384, 320, 256, 224, 192, 160, 128, 96, 64, 32)
SLACK = 1e-15
TRAJ_NAMES = ('SCM', 'ORD') + tuple(f'IBSI_g{g}' for g in GAMMAS) + tuple(f'SHUF_r{r}' for r in range(N_SHUF))
HIST_ARM = {'SCM': 'SCM_b1.0', 'ORD': 'TWIN_b1.0', **{f'IBSI_g{g}': f'IBSI_b1.0_g{g}' for g in GAMMAS},
            **{f'SHUF_r{r}': f'SHUF_b1.0_r{r}' for r in range(N_SHUF)}}


def nll_label(t):
    return f'N{t:.3f}'


def run_path_recorded(st0, kind, beta=None, gamma=None, target=None, max_steps=None):
    """`scm_bench.engine.run_path`, unchanged, plus a record of every accepted merge and of `spent` after it."""
    st = st0.copy(); k = st.k; alive = st.n > 0; parent = np.arange(k)
    capped = kind in ('SCM', 'TWIN', 'SHUF')
    B = E.budgets(beta) if capped else None
    inc0, _ = E.grouped_income_costs(st, alive); occ0 = E.occupation_abs_costs(st, alive); g0 = E.guard_abs(st, alive)
    spent = {key: 0.0 for key in E.BUDGET_KEYS}
    steps, reason = 0, 'exhausted'
    merges, spends = [], [[spent[key] for key in E.BUDGET_KEYS]]
    while True:
        live = np.flatnonzero(alive)
        if target is not None and len(live) <= target:
            reason = 'target_reached'; break
        if max_steps is not None and steps >= max_steps:
            reason = 'max_steps'; break
        if len(live) < 2:
            reason = 'single_cell'; break
        I, J = np.triu_indices(len(live), 1); I, J = live[I], live[J]
        d = E.pair_deltas(st, I, J)
        if capped:
            R = {key: max(B[key] - spent[key], 1e-15) for key in E.BUDGET_KEYS}
            norm = sum(np.maximum(d[key], 0.0) / R[key] for key in E.BUDGET_KEYS)
            pri = (d['sexvar'] / (norm + 1e-12)) if kind in ('SCM', 'SHUF') else -norm
            order = np.lexsort((J, I, -pri))
        else:
            score = d['nll1'] + d['nll2'] - gamma * d['sexinfo']
            order = np.lexsort((J, I, score))
        accepted = None
        if capped:
            ok = np.ones(len(I), dtype=bool)
            for key in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
                ok &= spent[key] + d[key] <= B[key] + 1e-15
            order = order[ok[order]]
        for o in order:
            if capped:
                a, b = int(I[o]), int(J[o])
                trial = st.copy(); E.merge_into(trial, a, b); al = alive.copy(); al[b] = False
                inc, _ = E.grouped_income_costs(trial, al)
                dinc = inc - inc0
                if dinc[0] > B['brier1'] + 1e-15 or dinc[1] > B['nll1'] + 1e-15 or dinc[2] > B['acc1'] + 1e-15:
                    continue
                accepted = (o, a, b, dinc); break
            else:
                accepted = (o, int(I[o]), int(J[o]), None); break
        if accepted is None:
            reason = 'no_admissible_pair'; break
        o, a, b, dinc = accepted
        for key in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
            spent[key] += float(d[key][o])
        E.merge_into(st, a, b); alive[b] = False; parent[b] = a; steps += 1
        if capped:
            spent['brier1'], spent['nll1'], spent['acc1'] = (float(x) for x in dinc)
        merges.append((a, b)); spends.append([spent[key] for key in E.BUDGET_KEYS])

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    final = np.array([find(i) for i in range(k)], dtype=np.int64)
    return {'final': final, 'steps': steps, 'stop_reason': reason, 'final_cells': int(alive.sum()),
            'merges': np.array(merges, dtype=np.int64).reshape(-1, 2), 'spent': np.array(spends, dtype=np.float64),
            'spent_final': {key: float(v) for key, v in spent.items()}}


def replay(st0, merges, wanted_steps):
    """Replay a recorded merge sequence once; return {m: (final, grp, v1, v2, plug-in costs vs C0)} at the wanted step counts."""
    st = st0.copy(); k = st.k; alive = st.n > 0; parent = np.arange(k)
    inc0, _ = E.grouped_income_costs(st, alive); occ0 = E.occupation_abs_costs(st, alive); g0 = E.guard_abs(st, alive)
    wanted = sorted(set(wanted_steps)); out = {}

    def snap():
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        final = np.array([find(i) for i in range(k)], dtype=np.int64)
        inc, grp = E.grouped_income_costs(st, alive); occ = E.occupation_abs_costs(st, alive); gd = E.guard_abs(st, alive)
        tab = E.tables(st, alive, final, grp)
        costs = {'income_grouped_brier_nll_acc': [float(x) for x in inc - inc0], 'occupation_brier_nll_acc': [float(x) for x in occ - occ0],
                 'guard0_guard1': [float(x) for x in gd - g0], 'live_cells': int(alive.sum())}
        return tab, costs
    m = 0
    if 0 in wanted:
        out[0] = snap()
    for a, b in merges:
        E.merge_into(st, int(a), int(b)); alive[int(b)] = False; parent[int(b)] = int(a); m += 1
        if m in wanted:
            out[m] = snap()
        if m >= wanted[-1]:
            break
    return out


def release_key(tab):
    """Effective-release identity: bit-identical per-initial-cell tables (income vector, occupation vector), float64."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tab['v1'][tab['group_of_initial']], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(tab['v2'][tab['cell_of_initial']], dtype=np.float64).tobytes())
    return h.hexdigest()


def checkpoints(spent_nll2, M, k, scm_nll_steps=None, hist_steps=None):
    """Registered checkpoint labels -> (merge count m or None, reached flag, note). spent_nll2[m] = F_m (F_0 = 0)."""
    F = np.asarray(spent_nll2); out = {}
    for t in NLL_GRID:
        over = np.flatnonzero(F > t + SLACK)          # first crossing: the last prefix before F first exceeds t (+ slack)
        m = int(over[0]) - 1 if len(over) else M
        reached = bool(len(over) > 0 or F[M] >= t - SLACK)
        out[nll_label(t)] = (m, reached, None if reached else f'unreached: trajectory stopped at F = {F[M]:.6g} < {t}; alias of TERM')
    for c in CNT_GRID:
        m = k - c
        out[f'C{c}'] = (m, True, None) if 0 <= m <= M else (None, False, f'unreached: terminal count {k - M} > {c} (missing coverage)')
    out['TERM'] = (M, True, None)
    if scm_nll_steps is not None:
        for t, ms in scm_nll_steps.items():
            out[f'S{t:.3f}'] = (ms, True, None) if ms <= M else (None, False, f'unreached: terminal count {k - M} > SCM count {k - ms} (missing coverage)')
    if hist_steps is not None:
        out['HIST'] = (hist_steps, True, None) if hist_steps <= M else (None, False, 'unreached historical count')
    return out


def run(seed, out_dir):
    if not C.committed(C.RESULTS / 'ANALYSIS_LOCK.json'):
        raise PermissionError('trajectories require the committed ANALYSIS_LOCK')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh trajectory directory required')
    out.mkdir(parents=True, mode=0o700)
    t_start = time.process_time()
    P, T, fs = C.source_fit(seed)
    hist_out, _ = C.source_arm_outputs(seed, arms=set(HIST_ARM.values()) | {'C0'})
    route = C.routing(seed)
    F = C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    if not np.array_equal(P['fine_assign_MODEL_FIT'].shape[0], len(F['row_id'])):
        raise AssertionError('MODEL_FIT size differs from the admitted partition')
    k = int(P['cent1'].shape[0])
    st0 = E.Stats(P['fine_assign_MODEL_FIT'], k, P['oof_income'], P['oof_occupation'], P['oof_sex_propensity'], F['sex'], F['occupation'])
    rec = {'schema': 'SCMUM1-trajectories-seed-v1', 'seed': seed, 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'),
           'initial_cells': k, 'trajectories': {}, 'gate': {}, 'checkpoints': {}, 'releases': {}, 'cpu_s': {},
           'MODEL_SELECT_labels_opened': False, 'DEV_REPORT_labels_opened': False, 'ATTACK_labels_used': False}
    # C0 identity: the reconstructed statistics reproduce the historical C0 tables bitwise
    alive0 = st0.n > 0; _, grp0 = E.grouped_income_costs(st0, alive0)
    tab0 = E.tables(st0, alive0, np.arange(k), grp0)
    rec['gate']['C0_tables_bitwise'] = all(np.array_equal(tab0[x], T[f'C0__{x}']) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
    K_hist = fs['arms']['SCM_b1.0']['final_cells']
    configs = {'SCM': (st0, dict(kind='SCM', beta=BETA)), 'ORD': (st0, dict(kind='TWIN', beta=BETA))}
    for g in GAMMAS:
        configs[f'IBSI_g{g}'] = (st0, dict(kind='IBSI', gamma=g, target=32))
    for r in range(N_SHUF):
        sh = st0.copy(); perm = np.random.default_rng(trees.seed_for('SCM1shuffle', seed, r)).permutation(k)
        rate = st0.SP / st0.n; sh.SP = st0.n * rate[perm]
        configs[f'SHUF_r{r}'] = (sh, dict(kind='SHUF', beta=BETA))
    paths, arrays, refs = {}, {}, {}
    for name in TRAJ_NAMES:
        base, kw = configs[name]
        t0 = time.process_time()
        p = run_path_recorded(base, **kw)
        t1 = time.process_time()
        ref = E.run_path(base, **kw)                     # (a) the source engine with the same arguments
        rec['cpu_s'][name] = {'recorded': round(t1 - t0, 2), 'engine_check': round(time.process_time() - t1, 2)}
        same = (np.array_equal(p['final'], ref[0]) and p['steps'] == ref[4]['steps'] and p['stop_reason'] == ref[4]['stop_reason']
                and p['final_cells'] == ref[4]['final_cells']
                and all(p['spent_final'][key] == ref[4]['spent_plugin_vs_C0'][key] for key in E.BUDGET_KEYS))
        rec['gate'][f'{name}|engine_run_path_identical'] = bool(same)
        paths[name] = p; refs[name] = ref
        arrays[f'{name}__merges'] = p['merges']; arrays[f'{name}__spent'] = p['spent']
        rec['trajectories'][name] = {'kind': kw['kind'], 'beta': kw.get('beta'), 'gamma': kw.get('gamma'), 'target': kw.get('target'),
                                     'steps': p['steps'], 'stop_reason': p['stop_reason'], 'terminal_cells': p['final_cells'],
                                     'terminal_spent_plugin_vs_C0': p['spent_final'], 'historical_arm': HIST_ARM[name]}
    # ---- checkpoints
    nll_col = list(E.BUDGET_KEYS).index('nll2')
    scm_cp = checkpoints(paths['SCM']['spent'][:, nll_col], paths['SCM']['steps'], k, None, k - K_hist)
    scm_nll_steps = {t: scm_cp[nll_label(t)][0] for t in NLL_GRID}
    cps = {'SCM': scm_cp}
    for name in TRAJ_NAMES[1:]:
        cps[name] = checkpoints(paths[name]['spent'][:, nll_col], paths[name]['steps'], k, scm_nll_steps, k - K_hist)
    # ---- replay once per trajectory; effective releases; historical gate (b), (c)
    rel_tabs, rel_meta = {}, {}
    for name in TRAJ_NAMES:
        steps = sorted({m for (m, reached, note) in cps[name].values() if m is not None})
        snaps = replay(configs[name][0], paths[name]['merges'], steps)
        rec['gate'][f'{name}|replayed_terminal_map_equal_recorded'] = bool(np.array_equal(snaps[paths[name]['steps']][0]['cell_of_initial'], paths[name]['final']))
        ref = refs[name]                                               # engine's own final state, tables compared bitwise
        rtab = E.tables(ref[2], ref[3], ref[0], ref[1])
        rec['gate'][f'{name}|replayed_terminal_tables_equal_engine'] = all(np.array_equal(snaps[paths[name]['steps']][0][x], rtab[x])
                                                                          for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
        hm = k - K_hist; arm = HIST_ARM[name]
        if hm in snaps:
            htab = snaps[hm][0]
            rec['gate'][f'{name}|HIST_tables_bitwise'] = all(np.array_equal(htab[x], T[f'{arm}__{x}']) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
            rec['gate'][f'{name}|HIST_outputs_bitwise_4_roles'] = all(np.array_equal(E.release(htab, route[f'init__{r}'])[t], hist_out[arm][r][t])
                                                                     for r in C.OUT_ROLES for t in C.TASKS)
        else:
            rec['gate'][f'{name}|HIST_tables_bitwise'] = False; rec['gate'][f'{name}|HIST_outputs_bitwise_4_roles'] = False
        rec['checkpoints'][name] = {}
        for label, (m, reached, note) in cps[name].items():
            if m is None:
                rec['checkpoints'][name][label] = {'m': None, 'reached': False, 'note': note, 'release': None}
                continue
            tab, costs = snaps[m]
            key = release_key(tab)
            if key not in rel_tabs:
                rel_tabs[key] = tab
                rel_meta[key] = {'first_label': f'{name}|{label}', 'live_cells': costs['live_cells'], 'plugin_costs_vs_C0': costs,
                                 'fitted_occupation_nll_F': float(paths[name]['spent'][m, nll_col]), 'aliases': []}
            rel_meta[key]['aliases'].append(f'{name}|{label}')
            rec['checkpoints'][name][label] = {'m': int(m), 'cells': k - int(m), 'reached': reached, 'note': note, 'release': key[:16],
                                               'F': float(paths[name]['spent'][m, nll_col])}
    gate_ok = all(rec['gate'].values())
    rec['gate']['ALL_PASS'] = bool(gate_ok)
    if not gate_ok:
        (out / 'TRAJ_SEED_FAILED_GATE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float))
        raise SystemExit('HALT: historical reproduction gate failed; no checkpoint admitted')
    # ---- capacity per effective release (label-free)
    def distinct(a):
        return int(len(np.unique(np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))))
    tab_arrays = {}
    for key, tab in rel_tabs.items():
        rid = key[:16]
        if any(m_['first_label'] for k2, m_ in rel_meta.items() if k2 != key and k2[:16] == rid):
            raise AssertionError('release id prefix collision')
        cap = {'nominal_cells': rel_meta[key]['live_cells'], 'income_groups_used': int(len(np.unique(tab['group_of_initial'])))}
        wires = {}
        for r in C.OUT_ROLES:
            o = E.release(tab, route[f'init__{r}'])
            cap[r] = {'occupied_cells': int(len(np.unique(tab['cell_of_initial'][route[f'init__{r}']]))),
                      'distinct_income_vectors': distinct(o['income']), 'distinct_occupation_vectors': distinct(o['occupation']),
                      'distinct_pair_vectors': distinct(np.column_stack((o['income'], o['occupation'])))}
            wires[r] = {t: C.wire_sha(o[t]) for t in C.TASKS}
        rec['releases'][rid] = {**{k_: v for k_, v in rel_meta[key].items()}, 'release_key_sha256': key, 'capacity': cap, 'wire_sha256': wires}
        for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'):
            tab_arrays[f'{rid}__{x}'] = tab[x]
    with open(out / 'TRAJECTORIES.npz', 'xb') as fh:
        np.savez(fh, **arrays)
    with open(out / 'RELEASE_TABLES.npz', 'xb') as fh:
        np.savez(fh, **tab_arrays)
    rec['n_effective_releases'] = len(rel_tabs)
    rec['cpu_s_total'] = round(time.process_time() - t_start, 2)
    for p in out.iterdir():
        p.chmod(0o600)
    rec['files_sha256'] = {p.name: C.sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    with open(out / 'TRAJ_SEED.json', 'x') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=float)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'gate_all_pass': r['gate']['ALL_PASS'], 'effective_releases': r['n_effective_releases'],
                      'trajectories': {k: (v['steps'], v['stop_reason'], v['terminal_cells'], round(v['terminal_spent_plugin_vs_C0']['nll2'], 5))
                                       for k, v in r['trajectories'].items()}, 'cpu_s_total': r['cpu_s_total']}, indent=1))

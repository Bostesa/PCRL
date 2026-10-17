"""Verifier check 2: trajectory checkpoints and aliases (own code: vengine.py; no campaign module imported).

Per seed and trajectory (SCM, ORD, IBSI_g{0.5,1,2}, SHUF_r{0..4}):
- greedy certificate along the RECORDED merge sequence: at every step all live pairs are re-scored with own costs; the
  recorded pair must be the first admissible pair in priority order (ties to the lowest (a, b)); the capped trajectories must
  end with no admissible pair and IBSI exactly at 32 cells. Divergences are classified by their relative priority gap.
- own running costs (spent) against the recorded spending matrix; F_m (occupation-NLL running sum) by two routes (per-merge
  identities and difference of absolute plug-in costs) against the recorded F_m.
- checkpoint labels from own F with the registered first-crossing rule (1e-15 slack), reached flags, counts, TERM, S{t}, HIST;
  the inventory's 'max' wording evaluated for comparison.
- own replay tables at every label: bitwise against RELEASE_TABLES.npz, own release keys, alias partition, capacity, wire hashes.
- historical endpoints at 261 / 266 cells against the source ARM_TABLES / ARM_OUTPUTS and the source FIT_SEED spending.
MODEL_FIT SEX and occupation labels enter only through the fixed fine-cell guard law, as in the campaign. No MODEL_SELECT,
ATTACK or DEV_REPORT label is read.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import vengine as G  # noqa: E402

NLL_GRID = tuple(i / 1000 for i in range(11))
CNT_GRID = (512, 448, 384, 320, 256, 224, 192, 160, 128, 96, 64, 32)
NAMES = ('SCM', 'ORD', 'IBSI_g0.5', 'IBSI_g1.0', 'IBSI_g2.0', 'SHUF_r0', 'SHUF_r1', 'SHUF_r2', 'SHUF_r3', 'SHUF_r4')
HIST = {'SCM': 'SCM_b1.0', 'ORD': 'TWIN_b1.0', 'IBSI_g0.5': 'IBSI_b1.0_g0.5', 'IBSI_g1.0': 'IBSI_b1.0_g1.0', 'IBSI_g2.0': 'IBSI_b1.0_g2.0',
        **{f'SHUF_r{r}': f'SHUF_b1.0_r{r}' for r in range(5)}}
REL_TIE = 1e-9          # relative priority gap below which a divergence is classified as a float-level near-tie


def pair_index(live, a, b):
    L = len(live); i = int(np.searchsorted(live, a)); j = int(np.searchsorted(live, b))
    assert live[i] == a and live[j] == b and i < j
    return i * (L - 1) - i * (i - 1) // 2 + (j - i - 1)


def income_after(st, a, b):
    n = st.n.copy(); S1 = st.S1.copy(); al = st.alive.copy()
    n[a] += n[b]; S1[a] += S1[b]; n[b] = 0.0; S1[b] = 0.0; al[b] = False
    return G.grouped_income_abs(st, n, S1, al)[0]


def certify(st0, kind, merges, gamma=None, target=None):
    st = st0.clone(); capped = kind in ('SCM', 'TWIN', 'SHUF')
    B = G.BUDGET
    inc0, _ = G.grouped_income_abs(st); occ0 = G.occupation_abs(st); g0 = G.guard_abs(st)
    spent = {k: 0.0 for k in G.KEYS}
    rows = [[spent[k] for k in G.KEYS]]; F_abs = [0.0]; occ_abs_rows = [np.zeros(3)]; guard_abs_rows = [np.zeros(2)]
    M = len(merges)
    res = {'steps_checked': 0, 'agree': 0, 'near_tie_divergences': [], 'divergences': [], 'recorded_pair_inadmissible': [],
           'income_checks': 0, 'terminal': None, 'min_dF': None}
    for m in range(M + 1):
        live = np.flatnonzero(st.alive)
        if not capped:
            if m == M:
                res['terminal'] = {'live_cells': int(len(live)), 'target': target, 'ok': bool(len(live) <= target)}
                break
            if len(live) <= target:
                res['divergences'].append({'m': m, 'what': 'recorded merge after target reached'}); break
        I, J = np.triu_indices(len(live), 1); I = live[I]; J = live[J]
        d = G.pair_costs(st, I, J)
        rec = None
        if m < M:
            a, b = int(merges[m][0]), int(merges[m][1]); rec = pair_index(live, a, b)
        if capped:
            R = {k: max(B[k] - spent[k], 1e-15) for k in G.KEYS}
            norm = sum(np.maximum(d[k], 0.0) / R[k] for k in G.KEYS)
            pri = d['sexvar'] / (norm + 1e-12) if kind in ('SCM', 'SHUF') else -norm
            ok = np.ones(len(I), dtype=bool)
            for k in G.CELL_KEYS:
                ok &= spent[k] + d[k] <= B[k] + G.SLACK
            order = np.lexsort((J, I, -pri))
            order = order[ok[order]]
            chosen = None
            for o in order:
                res['income_checks'] += 1
                dinc = income_after(st, int(I[o]), int(J[o])) - inc0
                if dinc[0] <= B['brier1'] + G.SLACK and dinc[1] <= B['nll1'] + G.SLACK and dinc[2] <= B['acc1'] + G.SLACK:
                    chosen = int(o); break
            if m == M:
                term = {'admissible_pair_found': chosen is not None, 'pairs_passing_cell_prefilter': int(len(order))}
                if chosen is not None:
                    dinc = income_after(st, int(I[chosen]), int(J[chosen])) - inc0
                    term['pair'] = [int(I[chosen]), int(J[chosen])]; term['dinc_minus_budget'] = [float(dinc[0] - B['brier1']), float(dinc[1] - B['nll1']), float(dinc[2] - B['acc1'])]
                res['terminal'] = term
                break
            # recorded pair under own computation
            rec_ok_cell = bool(all(spent[k] + d[k][rec] <= B[k] + G.SLACK for k in G.CELL_KEYS))
            dinc_rec = income_after(st, a, b) - inc0
            rec_ok_inc = bool(dinc_rec[0] <= B['brier1'] + G.SLACK and dinc_rec[1] <= B['nll1'] + G.SLACK and dinc_rec[2] <= B['acc1'] + G.SLACK)
            if not (rec_ok_cell and rec_ok_inc):
                res['recorded_pair_inadmissible'].append({'m': m, 'cell_ok': rec_ok_cell, 'income_ok': rec_ok_inc,
                                                          'cell_margins': {k: float(spent[k] + d[k][rec] - B[k]) for k in G.CELL_KEYS},
                                                          'income_margins': [float(x) for x in (dinc_rec - np.array([B['brier1'], B['nll1'], B['acc1']]))]})
            score_c = pri[chosen] if chosen is not None else None; score_r = pri[rec]
        else:
            score = d['nll1'] + d['nll2'] - gamma * d['sexinfo']
            order = np.lexsort((J, I, score)); chosen = int(order[0])
            score_c, score_r = -score[chosen], -score[rec]
            dinc_rec = None
        res['steps_checked'] += 1
        if chosen == rec:
            res['agree'] += 1
        else:
            gap = (score_c - score_r) if chosen is not None else None
            rel = abs(gap) / max(abs(score_r), 1e-300) if gap is not None else None
            item = {'m': m, 'recorded': [a, b], 'own_choice': [int(I[chosen]), int(J[chosen])] if chosen is not None else None,
                    'priority_gap': None if gap is None else float(gap), 'relative_gap': None if rel is None else float(rel)}
            (res['near_tie_divergences'] if (rel is not None and rel <= REL_TIE) else res['divergences']).append(item)
        # follow the recorded path
        for k in G.CELL_KEYS:
            spent[k] += float(d[k][rec])
        if capped:
            spent['brier1'], spent['nll1'], spent['acc1'] = (float(x) for x in dinc_rec)
        st.merge(a, b)
        rows.append([spent[k] for k in G.KEYS])
        oa = G.occupation_abs(st) - occ0; ga = G.guard_abs(st) - g0
        F_abs.append(float(oa[1])); occ_abs_rows.append(oa); guard_abs_rows.append(ga)
    rows = np.array(rows)
    dF = np.diff(rows[:, G.KEYS.index('nll2')])
    res['min_dF'] = float(dF.min()) if len(dF) else None
    res['n_negative_dF'] = int((dF < 0).sum())
    return res, rows, np.array(F_abs), np.array(occ_abs_rows), np.array(guard_abs_rows), st


def labels_from_F(F, M, k, scm_steps=None, hist_steps=None):
    out = {}
    for t in NLL_GRID:
        over = np.flatnonzero(F > t + G.SLACK)
        m = int(over[0]) - 1 if len(over) else M
        out[f'N{t:.3f}'] = (m, bool(len(over) > 0 or F[M] >= t - G.SLACK))
    for c in CNT_GRID:
        m = k - c
        out[f'C{c}'] = (m, True) if 0 <= m <= M else (None, False)
    out['TERM'] = (M, True)
    if scm_steps is not None:
        for t, ms in scm_steps.items():
            out[f'S{t:.3f}'] = (ms, True) if ms <= M else (None, False)
    if hist_steps is not None:
        out['HIST'] = (hist_steps, True) if hist_steps <= M else (None, False)
    return out


def max_rule(F, M):
    return {f'N{t:.3f}': int(np.flatnonzero(F <= t + G.SLACK).max()) for t in NLL_GRID}


def replay(st0, merges, wanted):
    st = st0.clone(); out = {}
    if 0 in wanted:
        out[0] = G.tables(st)
    for m, (a, b) in enumerate(merges, start=1):
        st.merge(int(a), int(b))
        if m in wanted:
            out[m] = G.tables(st)
        if m >= max(wanted):
            break
    return out


def key_of(tab):
    inc = np.ascontiguousarray(tab['v1'][tab['group_of_initial']], dtype=np.float64)
    occ = np.ascontiguousarray(tab['v2'][tab['cell_of_initial']], dtype=np.float64)
    raw = inc.tobytes() + occ.tobytes()
    return hashlib.sha256(raw).hexdigest(), raw


def n_distinct(a):
    a = np.ascontiguousarray(a)
    return int(len({r.tobytes() for r in a}))


def wire_hash(p):
    p = np.ascontiguousarray(p, dtype=np.float64)
    return hashlib.sha256(p.tobytes() + np.ascontiguousarray(np.argmax(p, axis=1).astype(np.int64)).tobytes()).hexdigest()


def run_seed(s):
    t0 = time.process_time()
    P = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'PARTITION.npz')
    FS = V.jload(V.SRC / 'fit' / f'FIT_{s}' / 'FIT_SEED.json')
    TH = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'ARM_TABLES.npz')
    AO = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'ARM_OUTPUTS.npz')
    RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz')
    tdir = V.PRIV / 'runs' / f'TRAJ_{s}'
    TR = V.npz(tdir / 'TRAJECTORIES.npz'); RTAB = V.npz(tdir / 'RELEASE_TABLES.npz'); TS = V.jload(tdir / 'TRAJ_SEED.json')
    sex = V.export('MODEL_FIT', 'sex')['sex'].astype(np.int64); occ = V.export('MODEL_FIT', 'occupation')['occupation'].astype(np.int64)
    k = P['cent1'].shape[0]
    base = G.St(P['fine_assign_MODEL_FIT'], P['oof_income'], P['oof_occupation'], P['oof_sex_propensity'], sex, occ, k)
    K_hist = FS['arms']['SCM_b1.0']['final_cells']
    cfg = {'SCM': (base, 'SCM', None, None), 'ORD': (base, 'TWIN', None, None)}
    for g in (0.5, 1.0, 2.0):
        cfg[f'IBSI_g{g}'] = (base, 'IBSI', g, 32)
    for r in range(5):
        sh = base.clone(); perm = np.random.default_rng(V.seed_for('SCM1shuffle', s, r)).permutation(k)
        rate = base.SP / base.n; sh.SP = base.n * rate[perm]
        cfg[f'SHUF_r{r}'] = (sh, 'SHUF', None, None)
    out = {'seed': s, 'K_hist': K_hist, 'trajectories': {}, 'labels': {}, 'discrepancies': []}
    myF, M_of, rows_of = {}, {}, {}
    for name in NAMES:
        st0, kind, gamma, target = cfg[name]
        merges = TR[f'{name}__merges']; spent_rec = TR[f'{name}__spent']
        res, rows, F_abs, occ_abs, guard_abs, st_end = certify(st0, kind, merges, gamma, target)
        M = len(merges); M_of[name] = M; myF[name] = rows[:, G.KEYS.index('nll2')]; rows_of[name] = rows
        diff = np.abs(rows - spent_rec)
        res['spent_shape_equal'] = rows.shape == spent_rec.shape
        res['spent_max_abs_diff_by_key'] = {k: float(diff[:, i].max()) for i, k in enumerate(G.KEYS)}
        res['spent_bitwise_equal_by_key'] = {k: bool(np.array_equal(rows[:, i], spent_rec[:, i])) for i, k in enumerate(G.KEYS)}
        Frec = spent_rec[:, G.KEYS.index('nll2')]
        res['F_max_abs_diff_own_delta_route_vs_recorded'] = float(np.abs(myF[name] - Frec).max())
        res['F_max_abs_diff_absolute_route_vs_recorded'] = float(np.abs(F_abs - Frec).max())
        res['occupation_brier_acc_absolute_route_max_abs_diff'] = [float(np.abs(occ_abs[:, 0] - spent_rec[:, G.KEYS.index('brier2')]).max()),
                                                                     float(np.abs(occ_abs[:, 2] - spent_rec[:, G.KEYS.index('acc2')]).max())]
        res['guard_absolute_route_max_abs_diff'] = [float(np.abs(guard_abs[:, gg] - spent_rec[:, G.KEYS.index(f'guard{gg}')]).max()) for gg in (0, 1)]
        tr_rec = TS['trajectories'][name]
        res['recorded'] = {'steps': tr_rec['steps'], 'stop_reason': tr_rec['stop_reason'], 'terminal_cells': tr_rec['terminal_cells']}
        res['own_terminal_cells'] = int(st_end.alive.sum())
        res['terminal_spent_equal_recorded_summary'] = all(rows[-1][i] == tr_rec['terminal_spent_plugin_vs_C0'][kk] for i, kk in enumerate(G.KEYS))
        res['terminal_spent_max_abs_diff_recorded_summary'] = max(abs(rows[-1][i] - tr_rec['terminal_spent_plugin_vs_C0'][kk]) for i, kk in enumerate(G.KEYS))
        # historical source spending at K_hist (the source ran the same rule to K_hist cells)
        hm = k - K_hist
        if hm <= M:
            src = FS['arms'][HIST[name]]['spent_plugin_vs_C0']
            res['source_FIT_SEED_spent_at_K_hist'] = {'recorded_bitwise': all(spent_rec[hm, i] == src[kk] for i, kk in enumerate(G.KEYS)),
                                                      'own_max_abs_diff': max(abs(rows[hm, i] - src[kk]) for i, kk in enumerate(G.KEYS)),
                                                      'source_final_cells': FS['arms'][HIST[name]]['final_cells']}
        out['trajectories'][name] = res
    # ---- labels
    scm_lab = labels_from_F(myF['SCM'], M_of['SCM'], k, None, k - K_hist)
    scm_steps = {t: scm_lab[f'N{t:.3f}'][0] for t in NLL_GRID}
    labs = {'SCM': scm_lab}
    for name in NAMES[1:]:
        labs[name] = labels_from_F(myF[name], M_of[name], k, scm_steps, k - K_hist)
    lab_cmp = {}
    max_vs_first = {}
    for name in NAMES:
        rec = TS['checkpoints'][name]; mine = labs[name]
        mism = []
        if set(rec) != set(mine):
            mism.append({'label_sets_differ': sorted(set(rec) ^ set(mine))})
        for lab, (m, reached) in mine.items():
            r = rec.get(lab)
            if r is None:
                continue
            if r['m'] != m or bool(r['reached']) != reached:
                mism.append({'label': lab, 'own': [m, reached], 'recorded': [r['m'], r['reached']]})
            if m is not None and r.get('F') is not None and r['F'] != float(myF[name][m]):
                mism.append({'label': lab, 'F_own': float(myF[name][m]), 'F_recorded': r['F']})
        lab_cmp[name] = mism
        mr = max_rule(myF[name], M_of[name])
        max_vs_first[name] = {lab: [mr[lab], mine[lab][0]] for lab in mr if mr[lab] != mine[lab][0]}
    out['labels'] = {'mismatches': lab_cmp, 'all_match': all(not v for v in lab_cmp.values()),
                     'inventory_max_rule_differs_from_first_crossing': max_vs_first,
                     'unreached_N_labels': {name: {lab: round(lab_t - float(myF[name][M_of[name]]), 12) for lab, (m, rch) in labs[name].items()
                                                   if lab.startswith('N') and not rch for lab_t in [float(lab[1:])]} for name in NAMES}}
    # ---- replay tables, release keys, aliases, capacity, wires
    own_rel = {}            # key -> {'raw': bytes, 'aliases': [...], 'tab': tab, 'live': int}
    label_key = {}
    tab_cmp = {'bitwise_equal_release_tables': 0, 'not_equal': []}
    for name in NAMES:
        st0 = cfg[name][0]
        wanted = sorted({m for (m, rch) in labs[name].values() if m is not None})
        snaps = replay(st0, TR[f'{name}__merges'], wanted)
        for lab, (m, rch) in labs[name].items():
            if m is None:
                continue
            tab = snaps[m]
            key, raw = key_of(tab)
            label_key[f'{name}|{lab}'] = key
            e = own_rel.setdefault(key, {'raw': raw, 'aliases': [], 'tab': tab, 'live': int(len(np.unique(tab['cell_of_initial'])))})
            if e['raw'] != raw:
                raise AssertionError('sha256 collision?')
            e['aliases'].append(f'{name}|{lab}')
            rid = TS['checkpoints'][name][lab]['release']
            same = rid is not None and all(np.array_equal(tab[x], RTAB[f'{rid}__{x}']) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
            same_dtype = rid is not None and all(tab[x].dtype == RTAB[f'{rid}__{x}'].dtype for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
            if same and same_dtype and key[:16] == rid:
                tab_cmp['bitwise_equal_release_tables'] += 1
            else:
                tab_cmp['not_equal'].append({'label': f'{name}|{lab}', 'recorded_rid': rid, 'own_rid': key[:16], 'tables_equal': bool(same)})
    rec_rel = TS['releases']
    own_part = {k_[:16]: sorted(v['aliases']) for k_, v in own_rel.items()}
    rec_part = {rid: sorted(v['aliases']) for rid, v in rec_rel.items()}
    part_equal = own_part == rec_part
    # dedupe exactly when bit-identical: every pair of distinct own releases differs in bytes (by construction) and no
    # recorded release merges labels with different bytes
    cap_mism, wire_mism = [], []
    for key, e in own_rel.items():
        rid = key[:16]; tab = e['tab']; rc = rec_rel.get(rid)
        if rc is None:
            continue
        cap = {'nominal_cells': int((np.bincount(tab['cell_of_initial'], minlength=k) > 0).sum()),
               'income_groups_used': int(len(np.unique(tab['group_of_initial'])))}
        for r in V.OUT_ROLES:
            init = RT[f'init__{r}']
            inc = tab['v1'][tab['group_of_initial'][init]]; oc = tab['v2'][tab['cell_of_initial'][init]]
            cap[r] = {'occupied_cells': int(len(np.unique(tab['cell_of_initial'][init]))), 'distinct_income_vectors': n_distinct(inc),
                      'distinct_occupation_vectors': n_distinct(oc), 'distinct_pair_vectors': n_distinct(np.column_stack((inc, oc)))}
            for t, p in (('income', inc), ('occupation', oc)):
                if rc['wire_sha256'][r][t] != wire_hash(p):
                    wire_mism.append(f'{rid}|{r}|{t}')
        rcap = rc['capacity']
        for kk, v in cap.items():
            if rcap.get(kk) != v:
                cap_mism.append({'release': rid, 'field': kk, 'own': v, 'recorded': rcap.get(kk)})
        if rc['live_cells'] != cap['nominal_cells']:
            cap_mism.append({'release': rid, 'field': 'live_cells', 'own': cap['nominal_cells'], 'recorded': rc['live_cells']})
    out['releases'] = {'own_effective_releases': len(own_rel), 'recorded_effective_releases': TS['n_effective_releases'],
                       'alias_partition_equal': part_equal, 'tables': tab_cmp, 'capacity_mismatches': cap_mism, 'wire_hash_mismatches': wire_mism,
                       'aliases_across_trajectories': sorted([v for v in own_part.values() if len({a.split('|')[0] for a in v}) > 1], key=len)[:50],
                       'n_aliases_across_trajectories': sum(1 for v in own_part.values() if len({a.split('|')[0] for a in v}) > 1)}
    if not part_equal:
        out['releases']['partition_diff'] = {'only_own': {k_: v for k_, v in own_part.items() if rec_part.get(k_) != v},
                                             'only_recorded': {k_: v for k_, v in rec_part.items() if own_part.get(k_) != v}}
    # ---- historical gate at K_hist with own tables and routing
    hist = {}
    for name in NAMES:
        m = k - K_hist; arm = HIST[name]
        tab = replay(cfg[name][0], TR[f'{name}__merges'], [m])[m]
        t_ok = all(np.array_equal(tab[x], TH[f'{arm}__{x}']) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
        o_ok = all(np.array_equal(tab['v1'][tab['group_of_initial'][RT[f'init__{r}']]], AO[f'{arm}__{r}__income'])
                   and np.array_equal(tab['v2'][tab['cell_of_initial'][RT[f'init__{r}']]], AO[f'{arm}__{r}__occupation']) for r in V.OUT_ROLES)
        hist[name] = {'arm': arm, 'cells': int(len(np.unique(tab['cell_of_initial']))), 'tables_bitwise': bool(t_ok), 'outputs_bitwise_4_roles': bool(o_ok),
                      'HIST_label_m': labs[name]['HIST'][0]}
    c0 = replay(base, TR['SCM__merges'], [0])[0]
    hist['C0'] = {'tables_bitwise': all(np.array_equal(c0[x], TH[f'C0__{x}']) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))}
    out['historical'] = hist
    # ---- public summary equals the private record
    pubsum = V.jload(V.RES / 'TRAJECTORY_SUMMARY.json')['per_seed'][f'S{s}']
    out['public_summary_consistent'] = {
        'checkpoints': pubsum['checkpoints'] == TS['checkpoints'],
        'trajectories': pubsum['trajectories'] == TS['trajectories'],
        'releases_aliases_capacity': all(pubsum['releases'][rid]['aliases'] == TS['releases'][rid]['aliases'] and pubsum['releases'][rid]['capacity'] == TS['releases'][rid]['capacity']
                                         for rid in TS['releases']) and set(pubsum['releases']) == set(TS['releases']),
        'n_effective_releases': pubsum['n_effective_releases'] == TS['n_effective_releases'], 'gate': pubsum['gate'] == TS['gate']}
    out['recorded_gate_all_pass'] = TS['gate'].get('ALL_PASS')
    out['cpu_s'] = round(time.process_time() - t0, 2)
    return out


if __name__ == '__main__':
    t0 = time.process_time()
    res = {'check': 2, 'title': 'trajectory checkpoints and aliases', 'per_seed': {}}
    for s in V.SEEDS:
        res['per_seed'][f'S{s}'] = run_seed(s)
        V.write_json('check2_trajectories.json', res)
    ok = True
    for s, r in res['per_seed'].items():
        for name, t in r['trajectories'].items():
            ok &= (not t['divergences']) and (not t['recorded_pair_inadmissible'])
            ok &= (t['terminal'].get('ok', True) if 'ok' in t['terminal'] else not t['terminal']['admissible_pair_found'])
            ok &= t['own_terminal_cells'] == t['recorded']['terminal_cells']
        ok &= r['labels']['all_match'] and r['releases']['alias_partition_equal'] and not r['releases']['tables']['not_equal']
        ok &= not r['releases']['capacity_mismatches'] and not r['releases']['wire_hash_mismatches']
        ok &= all(h['tables_bitwise'] and h.get('outputs_bitwise_4_roles', True) for h in r['historical'].values())
        ok &= all(r['public_summary_consistent'].values())
    res['status'] = 'PASS' if ok else 'FAIL_OR_REVIEW'
    res['cpu_s'] = round(time.process_time() - t0, 2)
    V.write_json('check2_trajectories.json', res)
    brief = {s: {'traj': {n: (t['agree'], t['steps_checked'], len(t['near_tie_divergences']), len(t['divergences']), len(t['recorded_pair_inadmissible']),
                              t['terminal'], t['F_max_abs_diff_own_delta_route_vs_recorded'], t['F_max_abs_diff_absolute_route_vs_recorded'])
                          for n, t in r['trajectories'].items()},
                 'labels_match': r['labels']['all_match'], 'partition_equal': r['releases']['alias_partition_equal'],
                 'tables_equal': r['releases']['tables']['bitwise_equal_release_tables'], 'tables_not_equal': len(r['releases']['tables']['not_equal']),
                 'cap_mism': len(r['releases']['capacity_mismatches']), 'wire_mism': len(r['releases']['wire_hash_mismatches']),
                 'hist': {n: (h['tables_bitwise'], h.get('outputs_bitwise_4_roles')) for n, h in r['historical'].items()},
                 'pub': r['public_summary_consistent'], 'max_vs_first': r['labels']['inventory_max_rule_differs_from_first_crossing']}
             for s, r in res['per_seed'].items()}
    print(json.dumps({'status': res['status'], 'cpu_s': res['cpu_s'], 'brief': brief}, indent=1, default=str))

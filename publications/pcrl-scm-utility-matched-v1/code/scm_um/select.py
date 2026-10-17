"""Unit UM_SELECT (MODEL_SELECT only; PROTOCOL.md sections 5-7). Writes SELECTION.json (committed as SELECTION_LOCK.json).

For every effective release of both seeds: the 12 original point clauses per seed against the frozen U0 / Ucal MODEL_SELECT
outputs, the within-SEX occupation Brier guard against U0, the occupation-NLL expenditure against the frozen C0 release and the
source cross-fitted pair-reader proxy (`scm_bench.select_scm.proxy`, unchanged). Then, in this order and with these rules only:
the conservative SCM nomination over the fixed NLL grid (one common threshold), explicit occupation-NLL matching (0.001 nats per
seed) with strongest-proxy comparator selection for ORD (per seed), PUB_IBSI (common gamma) and shuffled (common r), nearest
diagnostics when a match is unavailable, and the same-cell-count ORD and PUB_IBSI roles. No ATTACK, DEV_REPORT or sealed record
is read.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from scm_bench import engine as E
from scm_bench.select_scm import proxy
from scm_bench.selection import clauses
from . import common as C
from .trajectory import NLL_GRID, TRAJ_NAMES, GAMMAS, N_SHUF, nll_label

NOMINATION_OCC_NLL = 0.008
SUBGROUP_GUARD = 0.005
MATCH_TOL = 0.001


def load_traj(seed, tl):
    d = C.PRIV / 'runs' / f'TRAJ_{seed}'
    for name in ('TRAJ_SEED.json', 'RELEASE_TABLES.npz'):
        if C.sha(d / name) != tl['files_sha256'][f'S{seed}'][name]:
            raise PermissionError(f'{name} differs from TRAJECTORY_COMPLETE_LOCK')
    rec = json.loads((d / 'TRAJ_SEED.json').read_text())
    with np.load(d / 'RELEASE_TABLES.npz') as z:
        tabs = {}
        for key in z.files:
            rid, x = key.split('__')
            tabs.setdefault(rid, {})[x] = z[key].copy()
    return rec, tabs


def measure(outs, labels, sex, unit, refs, c0_nll, seed):
    cl = clauses(labels, outs, refs, seed)
    own = C.task_vectors(labels['occupation'], outs['occupation'], 5)
    u0 = C.task_vectors(labels['occupation'], refs['U0']['occupation'], 5)
    sub = {f'sex{g}': float((own['brier'] - u0['brier'])[sex == g].mean()) for g in (0, 1)}
    occ = float(own['nll'].mean())
    by_sex = {f'sex{g}': {'occupation_nll': float(own['nll'][sex == g].mean()), 'occupation_brier': float(own['brier'][sex == g].mean())} for g in (0, 1)}
    summ = {t: {m: float(v.mean()) for m, v in C.task_vectors(labels[t], outs[t], C.K[t]).items()} for t in C.TASKS}
    return {'clauses': cl, 'point_pass_12': all(c['point_pass'] for c in cl), 'max_normalized_12': max(c['normalized'] for c in cl),
            'subgroup_occupation_brier_excess_vs_U0': sub, 'occupation_nll': occ,
            'occupation_nll_expenditure_vs_C0': (occ - c0_nll) if c0_nll is not None else 0.0,
            'by_sex': by_sex, 'task_summary': summ, 'proxy': proxy(outs, sex, unit, seed)}


def nomination_conditions(m):
    """The 16 per-seed nomination conditions as (name, value, limit)."""
    out = [(f"{c['task']}|{c['reference']}|{c['metric']}", c['value'], c['cap']) for c in m['clauses']]
    for c in m['clauses']:
        if c['task'] == 'occupation' and c['metric'] == 'nll_excess':
            out.append((f"occupation|{c['reference']}|nll_excess<=0.008", c['value'], NOMINATION_OCC_NLL))
    for g, v in m['subgroup_occupation_brier_excess_vs_U0'].items():
        out.append((f'subgroup|{g}|occupation_brier_excess_vs_U0', v, SUBGROUP_GUARD))
    return out


def run(out_path):
    for name in ('ANALYSIS_LOCK.json', 'TRAJECTORY_COMPLETE_LOCK.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'selection requires the committed {name}')
    t_start = time.process_time()
    tl = json.loads((C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json').read_text())
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'))
    labels = {t: S[t] for t in C.TASKS}
    per, traj, frozen = {}, {}, {}
    for s in C.SEEDS:
        rec, tabs = load_traj(s, tl)
        traj[s] = rec
        route = C.routing(s)
        if not np.array_equal(route['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('MODEL_SELECT row order differs')
        bank, ids = C.frozen_bank(s, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
        c0_rid = rec['checkpoints']['SCM']['C512']['release']
        c0_out = E.release(tabs[c0_rid], route['init__MODEL_SELECT'])
        c0_nll = float(C.task_vectors(labels['occupation'], c0_out['occupation'], 5)['nll'].mean())
        per[s] = {}
        for rid in sorted(tabs):
            outs = E.release(tabs[rid], route['init__MODEL_SELECT'])
            per[s][rid] = measure(outs, labels, S['sex'], S['unit'], refs, c0_nll, s)
        frozen[s] = {}
        for name in ('NEURAL_RAW', 'H_LJ'):
            o = {t: bank[name]['MODEL_SELECT'][t] for t in C.TASKS}
            frozen[s][name] = measure(o, labels, S['sex'], S['unit'], refs, c0_nll, s)
    cps = {s: traj[s]['checkpoints'] for s in C.SEEDS}

    # ---------------- SCM nomination (common threshold over the NLL grid)
    def scm_rid(s, t):
        return cps[s]['SCM'][nll_label(t)]['release']
    grid = []
    for t in NLL_GRID:
        conds = {s: nomination_conditions(per[s][scm_rid(s, t)]) for s in C.SEEDS}
        ok = all(v <= lim for s in C.SEEDS for (_, v, lim) in conds[s])
        worst = max(v / lim for s in C.SEEDS for (_, v, lim) in conds[s])
        grid.append({'t': t, 'releases': {str(s): scm_rid(s, t) for s in C.SEEDS},
                     'cells': {str(s): cps[s]['SCM'][nll_label(t)]['cells'] for s in C.SEEDS},
                     'reached': {str(s): cps[s]['SCM'][nll_label(t)]['reached'] for s in C.SEEDS},
                     'eligible': bool(ok), 'worst_normalized': float(worst),
                     'failed_conditions': {str(s): [n for (n, v, lim) in conds[s] if v > lim] for s in C.SEEDS},
                     'mean_proxy': float(np.mean([per[s][scm_rid(s, t)]['proxy']['arm_proxy'] for s in C.SEEDS])),
                     'expenditure': {str(s): per[s][scm_rid(s, t)]['occupation_nll_expenditure_vs_C0'] for s in C.SEEDS}})
    elig = [g for g in grid if g['eligible']]
    if elig:
        nom = min(elig, key=lambda g: (g['mean_proxy'], g['t'])); status = 'NOMINEE_SELECTED'
    else:
        nom = min(grid, key=lambda g: (g['worst_normalized'], g['t'])); status = 'INELIGIBLE_DIAGNOSTIC_FALLBACK'
    t_star = nom['t']
    scm = {s: scm_rid(s, t_star) for s in C.SEEDS}
    e_star = {s: per[s][scm[s]]['occupation_nll_expenditure_vs_C0'] for s in C.SEEDS}
    m_star = {s: cps[s]['SCM'][nll_label(t_star)]['m'] for s in C.SEEDS}

    # ---------------- candidate sets and matching
    def candidates(s, name):
        rows = {}
        for label, cp in cps[s][name].items():
            if cp['release'] is None:
                continue
            r = rows.setdefault(cp['release'], {'release': cp['release'], 'labels': [], 'm': cp['m']})
            r['labels'].append(label); r['m'] = min(r['m'], cp['m'])
        out = []
        for rid, r in rows.items():
            mm = per[s][rid]
            d = mm['occupation_nll_expenditure_vs_C0'] - e_star[s]
            out.append({**r, 'trajectory': name, 'proxy': mm['proxy']['arm_proxy'], 'delta': float(d), 'abs_delta': float(abs(d)),
                        'match': bool(abs(d) <= MATCH_TOL), 'point_pass_12': mm['point_pass_12'], 'cells': cps[s][name][r['labels'][0]]['cells']})
        return sorted(out, key=lambda c: c['m'])

    def best_matched(cands):
        el = [c for c in cands if c['match'] and c['point_pass_12']]
        return min(el, key=lambda c: (c['proxy'], c['abs_delta'], c['m'])) if el else None

    def nearest(cands):
        el = [c for c in cands if c['point_pass_12']]
        return min(el, key=lambda c: (c['abs_delta'], c['proxy'], c['m'])) if el else None

    effort = {}
    roles = {s: {} for s in C.SEEDS}
    # ORD, per seed
    ordc = {s: candidates(s, 'ORD') for s in C.SEEDS}
    effort['ORD_NLLM'] = {str(s): len(ordc[s]) for s in C.SEEDS}
    ord_avail = {}
    for s in C.SEEDS:
        b = best_matched(ordc[s])
        ord_avail[s] = b is not None
        if b is not None:
            roles[s]['ORD_NLLM'] = {**b, 'status': 'MATCHED'}
        else:
            n = nearest(ordc[s])
            roles[s]['ORD_NLLM'] = ({**n, 'status': 'NEAREST_DIAGNOSTIC_NOT_MATCHED'} if n else {'release': None, 'status': 'MISSING'})

    def common_choice(names, label):
        cands = {name: {s: candidates(s, name) for s in C.SEEDS} for name in names}
        effort[label] = {name: {str(s): len(cands[name][s]) for s in C.SEEDS} for name in names}
        adm = []
        for i, name in enumerate(names):
            picks = {s: best_matched(cands[name][s]) for s in C.SEEDS}
            if all(p is not None for p in picks.values()):
                adm.append((float(np.mean([picks[s]['proxy'] for s in C.SEEDS])), i, name, picks))
        if adm:
            score, i, name, picks = min(adm, key=lambda x: (x[0], x[1]))
            return name, {s: {**picks[s], 'status': 'MATCHED'} for s in C.SEEDS}, True, [a[2] for a in adm]
        near = []
        for i, name in enumerate(names):
            picks = {s: nearest(cands[name][s]) for s in C.SEEDS}
            if all(p is not None for p in picks.values()):
                near.append((max(picks[s]['abs_delta'] for s in C.SEEDS), float(np.mean([picks[s]['proxy'] for s in C.SEEDS])), i, name, picks))
        if near:
            _, _, i, name, picks = min(near, key=lambda x: (x[0], x[1], x[2]))
            return name, {s: {**picks[s], 'status': 'NEAREST_DIAGNOSTIC_NOT_MATCHED'} for s in C.SEEDS}, False, []
        return None, {s: {'release': None, 'status': 'MISSING'} for s in C.SEEDS}, False, []
    ib_names = [f'IBSI_g{g}' for g in GAMMAS]; sh_names = [f'SHUF_r{r}' for r in range(N_SHUF)]
    ib_name, ib_picks, ib_avail, ib_adm = common_choice(ib_names, 'IBSI_NLLM')
    sh_name, sh_picks, sh_avail, sh_adm = common_choice(sh_names, 'SHUF_NLLM')
    for s in C.SEEDS:
        roles[s]['IBSI_NLLM'] = {**ib_picks[s], 'trajectory': ib_name}
        roles[s]['SHUF_NLLM'] = {**sh_picks[s], 'trajectory': sh_name}

    # ---------------- same cell count (the S{t*} companions)
    lab = f'S{t_star:.3f}'
    ord_cnt_avail = all(cps[s]['ORD'][lab]['release'] is not None for s in C.SEEDS)
    for s in C.SEEDS:
        cp = cps[s]['ORD'][lab]
        roles[s]['ORD_CNT'] = ({'release': cp['release'], 'm': cp['m'], 'cells': cp['cells'], 'trajectory': 'ORD', 'labels': [lab],
                                'point_pass_12': per[s][cp['release']]['point_pass_12'], 'proxy': per[s][cp['release']]['proxy']['arm_proxy'],
                                'delta': per[s][cp['release']]['occupation_nll_expenditure_vs_C0'] - e_star[s], 'status': 'SAME_COUNT'}
                               if cp['release'] is not None else {'release': None, 'status': 'UNMATCHED_CAPACITY'})
    ib_cnt = []
    for i, name in enumerate(ib_names):
        rids = {s: cps[s][name][lab]['release'] for s in C.SEEDS}
        el = all(per[s][rids[s]]['point_pass_12'] for s in C.SEEDS)
        ib_cnt.append({'trajectory': name, 'i': i, 'releases': rids, 'eligible_24': el,
                       'mean_proxy': float(np.mean([per[s][rids[s]]['proxy']['arm_proxy'] for s in C.SEEDS])),
                       'worst_normalized_24': max(per[s][rids[s]]['max_normalized_12'] for s in C.SEEDS)})
    effort['IBSI_CNT'] = len(ib_cnt)
    el = [x for x in ib_cnt if x['eligible_24']]
    pick = min(el, key=lambda x: (x['mean_proxy'], x['i'])) if el else min(ib_cnt, key=lambda x: (x['worst_normalized_24'], x['i']))
    for s in C.SEEDS:
        rid = pick['releases'][s]
        roles[s]['IBSI_CNT'] = {'release': rid, 'trajectory': pick['trajectory'], 'labels': [lab], 'cells': cps[s][pick['trajectory']][lab]['cells'],
                                'm': cps[s][pick['trajectory']][lab]['m'], 'point_pass_12': per[s][rid]['point_pass_12'],
                                'proxy': per[s][rid]['proxy']['arm_proxy'], 'delta': per[s][rid]['occupation_nll_expenditure_vs_C0'] - e_star[s],
                                'status': 'SAME_COUNT' if pick['eligible_24'] else 'SAME_COUNT_INELIGIBLE_FALLBACK'}
    # ---------------- remaining roles
    for s in C.SEEDS:
        roles[s]['SCM'] = {'release': scm[s], 'trajectory': 'SCM', 'labels': [nll_label(t_star)], 'm': m_star[s], 'cells': cps[s]['SCM'][nll_label(t_star)]['cells'],
                           'proxy': per[s][scm[s]]['proxy']['arm_proxy'], 'delta': 0.0, 'status': status}
        roles[s]['C0'] = {'release': cps[s]['SCM']['C512']['release'], 'trajectory': 'SCM', 'labels': ['C512'], 'm': 0, 'cells': cps[s]['SCM']['C512']['cells'],
                          'status': 'REFERENCE'}
        roles[s]['HIST_SCM'] = {'release': cps[s]['SCM']['TERM']['release'], 'trajectory': 'SCM', 'labels': ['TERM', 'HIST'], 'm': cps[s]['SCM']['TERM']['m'],
                                'cells': cps[s]['SCM']['TERM']['cells'], 'status': 'HISTORICALLY_SELECTED_DIAGNOSTIC'}
        for name in ('NEURAL_RAW', 'H_LJ'):
            roles[s][name] = {'release': f'FROZEN_{name}', 'status': 'FROZEN_REFERENCE', 'proxy': frozen[s][name]['proxy']['arm_proxy'],
                              'delta': frozen[s][name]['occupation_nll_expenditure_vs_C0'] - e_star[s], 'point_pass_12': frozen[s][name]['point_pass_12']}
        for rname, r in roles[s].items():
            if r.get('release') and not str(r['release']).startswith('FROZEN_'):
                mm = per[s][r['release']]
                r['expenditure'] = mm['occupation_nll_expenditure_vs_C0']; r['point_pass_12'] = mm['point_pass_12']
                r['proxy'] = mm['proxy']['arm_proxy']; r['delta'] = mm['occupation_nll_expenditure_vs_C0'] - e_star[s]
                sc = per[s][scm[s]]['task_summary']
                r['SCM_minus_role_MODEL_SELECT'] = {f'{t}_{m}': sc[t][m] - mm['task_summary'][t][m] for t in C.TASKS for m in ('nll', 'brier', 'accuracy')}
                r['income_expenditure_qualifier'] = ('INCOME_EXPENDITURE_NOT_MATCHED' if r['SCM_minus_role_MODEL_SELECT']['income_nll'] > MATCH_TOL
                                                     else 'income NLL within 0.001 or lower for SCM')
    ROLE_ORDER = ('SCM', 'ORD_NLLM', 'ORD_CNT', 'IBSI_NLLM', 'IBSI_CNT', 'SHUF_NLLM', 'NEURAL_RAW', 'H_LJ', 'C0', 'HIST_SCM')
    services = {}
    for s in C.SEEDS:                                   # deduplicate roles that share an effective release
        by_rel = {}
        for rn in ROLE_ORDER:
            rid = roles[s][rn].get('release')
            if rid is None:
                continue
            by_rel.setdefault(rid, []).append(rn)
        services[str(s)] = {'service_of_role': {rn: names[0] for names in by_rel.values() for rn in names},
                            'aliases': {names[0]: names for names in by_rel.values()},
                            'missing_roles': [rn for rn in ROLE_ORDER if roles[s][rn].get('release') is None]}
    sel = {'schema': 'SCMUM1-selection-v1', 'utc': C.now(), 'status': status,
           'nomination': {'t_star': t_star, 'rule': 'eligible: 24 original point clauses + occupation NLL excess <= 0.008 vs U0 and Ucal + both '
                          'within-SEX occupation Brier guards <= 0.005 (both seeds); min mean proxy; ties to smaller t',
                          'grid': grid, 'eligible_thresholds': [g['t'] for g in elig], 'K_star': {str(s): traj[s]['initial_cells'] - m_star[s] for s in C.SEEDS},
                          'e_star': {str(s): e_star[s] for s in C.SEEDS}},
           'matching': {'tolerance': MATCH_TOL, 'ORD_NLLM_available': {str(s): ord_avail[s] for s in C.SEEDS},
                        'IBSI_NLLM': {'gamma_trajectory': ib_name, 'available_both_seeds': ib_avail, 'admissible': ib_adm},
                        'SHUF_NLLM': {'r_trajectory': sh_name, 'available_both_seeds': sh_avail, 'admissible': sh_adm},
                        'ORD_CNT_available_both_seeds': ord_cnt_avail, 'IBSI_CNT': {k_: v for k_, v in pick.items() if k_ != 'i'},
                        'candidates': {name: {str(s): candidates(s, name) for s in C.SEEDS} for name in TRAJ_NAMES[1:]}},
           'matching_coverage': {name: {str(s): (lambda xs: {'n_candidates': len(xs), 'expenditures_sorted': xs,
                                                             'largest_gap_bracketing_e_star': max([b_ - a_ for a_, b_ in zip(xs, xs[1:]) if a_ <= e_star[s] <= b_], default=None),
                                                             'max_candidate_expenditure': max(xs) if xs else None})(
                                       sorted(per[s][c['release']]['occupation_nll_expenditure_vs_C0'] for c in candidates(s, name)))
                                       for s in C.SEEDS} for name in TRAJ_NAMES[1:]},
           'selection_effort': effort, 'roles': {str(s): roles[s] for s in C.SEEDS}, 'services': services, 'role_order': list(ROLE_ORDER),
           'per_seed_measurements': {str(s): per[s] for s in C.SEEDS}, 'frozen_measurements': {str(s): frozen[s] for s in C.SEEDS},
           'trajectory_lock_sha256': C.sha(C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json'),
           'cpu_s': round(time.process_time() - t_start, 2), 'DEV_REPORT_labels_opened': False, 'ATTACK_labels_used': False}
    with open(out_path, 'x') as f:
        json.dump(sel, f, indent=1, sort_keys=True, default=float)
    return sel


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); r = run(a.out)
    print(json.dumps({'status': r['status'], 't_star': r['nomination']['t_star'], 'eligible': r['nomination']['eligible_thresholds'],
                      'K_star': r['nomination']['K_star'], 'e_star': r['nomination']['e_star'], 'matching': {k: v for k, v in r['matching'].items() if k != 'candidates'},
                      'roles': {s: {rn: (v.get('release'), v.get('status'), v.get('cells'), round(v['delta'], 5) if v.get('delta') is not None else None,
                                         round(v['proxy'], 4) if v.get('proxy') is not None else None) for rn, v in rr.items()} for s, rr in r['roles'].items()},
                      'cpu_s': r['cpu_s']}, indent=1, default=float))

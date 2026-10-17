"""Fit unit (one model seed): teachers, fine partition, the 31-release utility bank and its outputs (PROTOCOL.md sections 2-6).

Requires the committed ANALYSIS_LOCK. Access: MODEL_FIT features, task labels and SEX (SEX only for the out-of-fold
propensity that drives SCM merges and for the within-SEX guard); features of MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and
DEV_REPORT to compute releases (no labels, no SEX). Writes private files under the given directory:
TEACHERS/ (fold teacher payloads), PARTITION.npz, ARM_TABLES.npz, ARM_OUTPUTS.npz and FIT_SEED.json (diagnostics).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from acs_replication import trees
from . import common as C
from . import engine as E
from .teachers import fit_teacher, household_folds, teacher_probs

OUT_ROLES = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
BETAS = (0.5, 1.0, 1.5)
GAMMAS = (0.5, 1.0, 2.0)
N_SHUF = 5
M0 = 512


def run(seed, out_dir):
    if not C.committed(C.RESULTS / 'ANALYSIS_LOCK.json'):
        raise PermissionError('fits require the committed ANALYSIS_LOCK')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh fit directory required')
    (out / 'TEACHERS').mkdir(parents=True, mode=0o700)
    t_start = time.process_time()
    F = C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    X, names, unit = F['X'], F['feature_names'], F['unit']
    roles = {r: C.load_role(r, ('features',)) for r in OUT_ROLES}
    rec = {'schema': 'SCM1-fit-seed-v1', 'seed': seed, 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'),
           'teachers': {}, 'propensity': {}, 'partition': {}, 'arms': {}, 'capacity': {}, 'diagnostics': {}}
    # ---- teachers and propensity: five household folds, out-of-fold on MODEL_FIT, fold-average at runtime
    fold = household_folds(unit, 5, trees.seed_for('SCM1|folds', seed))
    oof = {'income': np.zeros((len(X), 2)), 'occupation': np.zeros((len(X), 5)), 'sex': np.zeros(len(X))}
    runtime = {r: {'income': 0.0, 'occupation': 0.0} for r in OUT_ROLES + ('MODEL_FIT',)}
    for f in range(5):
        tr, te = fold != f, fold == f
        for task in ('income', 'occupation'):
            t, info = fit_teacher(task, X[tr], F[task][tr], unit[tr], names, 100 * seed + f, 'SCM1teacher')
            (out / 'TEACHERS' / f'fold{f}_{task}.json').write_text(json.dumps(t.payload(), sort_keys=True))
            rec['teachers'][f'fold{f}_{task}'] = info
            oof[task][te] = teacher_probs(t, X[te], names)
            for r in runtime:
                Xr = X if r == 'MODEL_FIT' else roles[r]['X']
                runtime[r][task] = runtime[r][task] + teacher_probs(t, Xr, names) / 5.0
        tp, pinfo = fit_teacher('sex', X[tr], F['sex'][tr], unit[tr], names, 100 * seed + f, 'SCM1propensity')
        oof['sex'][te] = teacher_probs(tp, X[te], names)[:, 1]
        rec['propensity'][f'fold{f}'] = {k: v for k, v in pinfo.items() if k != 'parity_on_fit_rows'}
    # ---- fine partition C0 (KL-Bregman k-means on OOF task probabilities)
    mu1, mu2, fine, kinfo = E.kl_kmeans(oof['income'], oof['occupation'], M0, trees.seed_for('SCM1kmeans', seed))
    k = kinfo['occupied_cells']; rec['partition'] = kinfo
    st0 = E.Stats(fine, k, oof['income'], oof['occupation'], oof['sex'], F['sex'], F['occupation'])
    cent1, cent2 = st0.S1 / st0.n[:, None], st0.S2 / st0.n[:, None]
    init = {r: E.kl_assign(runtime[r]['income'], runtime[r]['occupation'], cent1, cent2) for r in runtime}
    # OOF-vs-runtime mismatch diagnostics
    occ_fit = np.bincount(fine, minlength=k) / len(fine); occ_sel = np.bincount(init['MODEL_SELECT'], minlength=k) / len(init['MODEL_SELECT'])
    rec['diagnostics']['tv_distance_MODEL_FIT_oof_vs_MODEL_SELECT_runtime_occupancy'] = float(0.5 * np.abs(occ_fit - occ_sel).sum())
    rec['diagnostics']['share_MODEL_FIT_runtime_cell_equals_oof_cell'] = float(np.mean(init['MODEL_FIT'] == fine))
    rec['diagnostics']['mean_KL_oof_vs_runtime_MODEL_FIT'] = {t: float((E.xlogy(oof[t], oof[t]) - E.xlogy(oof[t], runtime['MODEL_FIT'][t])).sum(1).mean())
                                                             for t in ('income', 'occupation')}
    # ---- the utility bank
    paths = {}
    final0 = np.arange(k); alive0 = st0.n > 0; _, grp0 = E.grouped_income_costs(st0, alive0)
    paths['C0'] = (final0, grp0, st0, alive0, {'kind': 'C0', 'final_cells': int(k), 'steps': 0, 'stop_reason': 'unmerged'})
    for b in BETAS:
        t0 = time.process_time()
        paths[f'SCM_b{b}'] = E.run_path(st0, 'SCM', beta=b)
        K = paths[f'SCM_b{b}'][4]['final_cells']
        paths[f'TWIN_b{b}'] = E.run_path(st0, 'TWIN', beta=b, target=K)
        for g in GAMMAS:
            paths[f'IBSI_b{b}_g{g}'] = E.run_path(st0, 'IBSI', gamma=g, target=K)
        for r in range(N_SHUF):
            sh = st0.copy(); perm = np.random.default_rng(trees.seed_for('SCM1shuffle', seed, r)).permutation(k)
            rate = st0.SP / st0.n; sh.SP = st0.n * rate[perm]
            paths[f'SHUF_b{b}_r{r}'] = E.run_path(sh, 'SHUF', beta=b, target=K)
        rec['diagnostics'][f'cpu_s_beta_{b}'] = round(time.process_time() - t0, 2)
    tabs, arrays = {}, {}
    for r in OUT_ROLES:
        arrays[f'row_id__{r}'] = roles[r]['row_id']; arrays[f'unit__{r}'] = roles[r]['unit']
    for arm, (final, grp, st, alive, prec) in paths.items():
        tab = E.tables(st, alive, final, grp); tabs[arm] = tab
        cap = {'nominal_cells': int(prec['final_cells']), 'income_groups_used': int(len(np.unique(tab['group_of_initial'])))}
        for r in OUT_ROLES:
            rel = E.release(tab, init[r])
            arrays[f'{arm}__{r}__income'] = rel['income']; arrays[f'{arm}__{r}__occupation'] = rel['occupation']
            def distinct(a):
                return int(len(np.unique(np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))))
            cap[r] = {'occupied_cells': int(len(np.unique(tab['cell_of_initial'][init[r]]))), 'distinct_income_vectors': distinct(rel['income']),
                      'distinct_occupation_vectors': distinct(rel['occupation']),
                      'distinct_pair_vectors': distinct(np.column_stack((rel['income'], rel['occupation'])))}
        rec['arms'][arm] = prec; rec['capacity'][arm] = cap
    np.savez(out / 'PARTITION.npz', fine_mu1=mu1, fine_mu2=mu2, cent1=cent1, cent2=cent2, fine_assign_MODEL_FIT=fine,
             oof_income=oof['income'], oof_occupation=oof['occupation'], oof_sex_propensity=oof['sex'], fold=fold)
    tab_arrays = {f'{arm}__{key}': v for arm, t in tabs.items() for key, v in t.items()}
    np.savez(out / 'ARM_TABLES.npz', **tab_arrays)
    with open(out / 'ARM_OUTPUTS.npz', 'xb') as fh:
        np.savez(fh, **arrays)
    for p in out.rglob('*'):
        if p.is_file():
            p.chmod(0o600)
    rec['files_sha256'] = {str(p.relative_to(out)): C.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    rec['cpu_s_total'] = round(time.process_time() - t_start, 2)
    rec['DEV_REPORT_labels_opened'] = False; rec['MODEL_SELECT_labels_opened'] = False
    with open(out / 'FIT_SEED.json', 'x') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=float)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'cpu_s': r['cpu_s_total'], 'partition': r['partition'],
                      'final_cells': {k: v['final_cells'] for k, v in r['arms'].items() if not k.startswith('SHUF')},
                      'stop': {k: v['stop_reason'] for k, v in r['arms'].items() if k.startswith(('SCM', 'TWIN'))}}, indent=1))

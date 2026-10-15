"""Stage NOMINATION: replay the RAW_J and FARE nomination rules (COMPLETION_LOCK text) from the FIT2 tables and
recompute the FIT2 coalition proxy (two-fold group cross-fit, max of LR_C1 / HGB_15x200 pooled OOF AUC) for the
two nominees from their frozen artifacts with own code.
"""
from __future__ import annotations

import argparse
import hashlib
import warnings

import numpy as np

from . import core
from .stage_wires import forward, load_state

RHOS = ('1.0', '3.0', '10.0')
EPOCHS = (2, 5, 10, 20)
TOL = 1e-12


def fold_of(unit):
    return np.array([int(hashlib.sha256(('THP1|FIT2-proxy|' + str(u)).encode()).hexdigest(), 16) % 2 for u in unit])


def view_tables(p1, d1, p2, d2):
    a = np.column_stack((p1, d1.astype(np.float64)))
    b = np.column_stack((p2, d2.astype(np.float64)))
    return {'income': a, 'occupation': b, 'pair': np.column_stack((a, b))}


def crossfit_auc(x, y, folds, reader, scaling):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    oof = np.empty(len(y))
    xg = StandardScaler().fit_transform(x) if scaling == 'global' else None
    for f in (0, 1):
        tr, te = folds != f, folds == f
        if scaling == 'fold':
            sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        else:
            xtr, xte = xg[tr], xg[te]
        if reader == 'LR_C1':
            m = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
        else:
            m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False,
                                               random_state=2026101502)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(xtr, y[tr])
        oof[te] = m.predict_proba(xte)[:, 1]
    return core.auc(y, oof)


def replay(tables, keys, order_of):
    rows = []
    for key in keys:
        feas = []; prox = []; nll = []; viol = []; consistent = True
        for s in ('0', '1'):
            t = tables[s][key]
            ok = len(t['internal']) == 6 and all(c['point_pass'] == (c['value'] <= c['cap']) for c in t['internal'])
            f = all(c['point_pass'] for c in t['internal'])
            consistent &= ok and f == t['internal_point_feasible']
            feas.append(f); prox.append(t['proxy']['pair']['value']); nll.append(t['summed_nll']); viol.append(t['max_normalized_violation'])
        rows.append({'key': key, 'feasible': all(feas), 'worst_pair_proxy': max(prox), 'summed_nll': nll[0] + nll[1],
                     'worst_violation': max(viol), 'order': order_of(key), 'table_consistent': consistent})
    feasible = [r for r in rows if r['feasible']]
    if feasible:
        nom = min(feasible, key=lambda r: (r['worst_pair_proxy'], r['summed_nll'], *r['order']))
        status = 'POINT_FEASIBLE_NOMINEE'
    else:
        nom = min(rows, key=lambda r: (r['worst_violation'], r['summed_nll'], *r['order']))
        status = 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC'
    return rows, nom, status


def run(spec):
    import torch
    torch.set_num_threads(1)
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification')
    R = core.Receipt('NOMINATION', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    J = lambda it: core.load_json(verify, it)
    tables = {s: J(cfg['fit2_tables'][s])['table'] for s in ('0', '1')}
    sel = J(cfg['selection']); slock = J(core.lock_item(cfg, 'selection_lock')); clock = J(cfg['completion_lock'])
    for s in ('0', '1'):
        R.check(f'FIT2_table_s{s}_sha_equals_selection_lock', cfg['fit2_tables'][s]['sha256'] == slock['fit2_sha256'][s])
        R.check(f'results_COMPLETION_FIT2_TABLE_S{s}_is_byte_copy', core.file_sha(verify, cfg['fit2_results_copy'][s]) == cfg['fit2_tables'][s]['sha256'])
        R.check(f'FIT2_s{s}_ATT_INNER_closed', J(cfg['fit2_tables'][s]).get('ATT_opened') is False and J(cfg['fit2_tables'][s]).get('INNER_opened') is False)
    R.check('SELECTION_sha_equals_selection_lock', cfg['selection']['sha256'] == slock['selection_output_sha256'])
    R.check('results_COMPLETION_SELECTION_DETAIL_is_byte_copy', core.file_sha(verify, cfg['selection_detail']) == cfg['selection']['sha256'])
    R.numbers['lock_rules'] = {'RAW_J': clock['added_controls']['RAW_J_NEQ']['nomination'], 'FARE': clock['added_controls']['FARE_TC']['nomination'],
                               'required_status_rule': clock['required_status_rule']}

    # ---------------------------------------------------------------- RAW_J rule replay
    rk = [f'RAW_J|{r}|E{e}' for r in RHOS for e in EPOCHS]
    rows, nom, status = replay(tables, rk, lambda k: (float(k.split('|')[1]), int(k.split('|')[2][1:])))
    R.check('RAW_J_tables_internally_consistent', all(r['table_consistent'] for r in rows))
    R.check('RAW_J_nominee_is_rho1_E2', nom['key'] == 'RAW_J|1.0|E2' and status == 'POINT_FEASIBLE_NOMINEE', nominee=nom['key'], status=status)
    mine_feasible = [r['key'] for r in rows if r['feasible']]
    R.check('RAW_J_feasible_set_equals_selection_lock', mine_feasible == slock['RAW_J']['feasible_candidates'], mine=mine_feasible)
    selc = {c['key']: c for c in sel['RAW_J']['candidates']}
    diff = max(max(abs(r[f] - selc[r['key']][f]) for f in ('worst_pair_proxy', 'summed_nll', 'worst_violation')) for r in rows)
    R.check('RAW_J_candidate_summaries_equal_SELECTION', diff <= TOL and all(r['feasible'] == selc[r['key']]['feasible'] for r in rows), max_abs=diff)
    R.check('RAW_J_selection_lock_nominee_status', slock['RAW_J']['nominee']['key'] == nom['key'] and slock['RAW_J']['status'] == status
            and sel['RAW_J']['nominee']['key'] == nom['key'])

    # ---------------------------------------------------------------- FARE rule replay
    fk = [f'FARE|F{i}' for i in (1, 2, 3)]
    frows, fnom, fstatus = replay(tables, fk, lambda k: (int(k.split('|')[1][1:]),))
    R.check('FARE_tables_internally_consistent', all(r['table_consistent'] for r in frows))
    R.check('FARE_nominee_F1_fallback', fnom['key'] == 'FARE|F1' and fstatus == 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC', nominee=fnom['key'], status=fstatus)
    self_f = {c['key']: c for c in sel['FARE']['candidates']}
    fdiff = max(max(abs(r[f] - self_f[r['key']][f]) for f in ('worst_pair_proxy', 'summed_nll', 'worst_violation')) for r in frows)
    R.check('FARE_candidate_summaries_equal_SELECTION', fdiff <= TOL and all(r['feasible'] == self_f[r['key']]['feasible'] for r in frows), max_abs=fdiff)
    R.check('FARE_selection_lock_nominee_status', slock['FARE']['nominee']['key'] == 'FARE|F1' and slock['FARE']['status'] == fstatus
            and slock['FARE']['feasible_candidates'] == [r['key'] for r in frows if r['feasible']])
    alt = min(frows, key=lambda r: (r['worst_pair_proxy'], r['summed_nll'], *r['order']))['key']
    R.notes.append(f'FARE: no setting is point-feasible on the 12 FIT2 clauses; the nominee follows the minimum worst-seed normalized '
                   f'violation fallback (same fallback text as RAW_J). A lowest-proxy fallback would instead pick {alt}; the lock text for '
                   f'FARE says only "else descriptive fallback".')
    R.numbers['RAW_J_candidates'] = rows; R.numbers['FARE_candidates'] = frows

    # ---------------------------------------------------------------- roster, required statuses, matched controls
    ro = slock['attack_roster']
    R.check('roster_matches_rules', ro['RAW_J_NEQ'] == nom['key'] and ro['FARE_TC'] == fnom['key'] and ro['FARE_TC_Z'] == 'FARE|Z1'
            and ro['T_REFIT_E20'] == 'T_REFIT|none|E20' and ro['T_TREE'] == 'T_TREE')
    R.check('T_NEURAL_matched_to_nominee_epoch_E2', ro['T_NEURAL'] == 'T_NEURAL|none|' + nom['key'].split('|')[2])
    rs = slock['required_status']
    R.check('required_status_RAW_J_NEQ_required', rs['RAW_J_NEQ'] == {'internal_point_feasible': True, 'required_comparator_for_L_TREE': True})
    R.check('required_status_FARE_TC_not_required', rs['FARE_TC'] == {'internal_point_feasible': False, 'required_comparator_for_L_TREE': False})

    # ---------------------------------------------------------------- FIT2 data
    guard()
    fx = core.load_npz(verify, cfg['fit2_features'], ['row_id', 'unit', 'X', 'feature_names'])
    sx = core.load_npz(verify, cfg['fit2_sex'], ['row_id', 'unit', 'sex'])
    yi = core.load_npz(verify, cfg['fit2_income'], ['row_id', 'unit', 'income'])
    yo = core.load_npz(verify, cfg['fit2_occupation'], ['row_id', 'unit', 'occupation'])
    R.check('FIT2_rows_aligned', all(np.array_equal(z['row_id'], fx['row_id']) and np.array_equal(z['unit'], fx['unit']) for z in (sx, yi, yo)),
            rows=int(len(fx['row_id'])), groups=int(len(np.unique(fx['unit']))))
    X = fx['X']; y = sx['sex'].astype(np.int64); folds = fold_of(fx['unit'])
    R.numbers['FIT2_fold_sizes_rows'] = [int((folds == 0).sum()), int((folds == 1).sum())]
    labels = {'income': yi['income'].astype(np.int64), 'occupation': yo['occupation'].astype(np.int64)}

    proxy_rows = []; max_proxy = {'fold': 0.0, 'global': 0.0}; metric_diff = 0.0; internal_diff = 0.0
    for s in ('0', '1'):
        guard()
        neq = forward(load_state(verify, cfg['neq'][s]), X)
        st = load_state(verify, cfg['neural'][s]['RAW_J'])
        R.check(f'RAW_J_s{s}_checkpoint_sha_equals_selection_lock', cfg['neural'][s]['RAW_J']['sha256'] ==
                slock['attack_roster_files'][s]['RAW_J_NEQ']['files_sha256']['RAW_J__rho1.0__E2.pt'])
        pr = forward(st, X)
        f = core.load_npz(verify, cfg['fare'][s]['F1'], ['row_id', 'q1', 'hard1', 'q2', 'hard2'])
        pos = {int(r): i for i, r in enumerate(f['row_id'])}
        idx = np.array([pos[int(r)] for r in fx['row_id']])
        outs = {'RAW_J|1.0|E2': (pr['income'], pr['income'].argmax(1), pr['occupation'], pr['occupation'].argmax(1)),
                'FARE|F1': (f['q1'][idx], f['hard1'][idx], f['q2'][idx], f['hard2'][idx])}
        for key, (p1, d1, p2, d2) in outs.items():
            t = tables[s][key]
            # task metrics and internal clauses vs N_EQ on FIT2 (labels opened for selection already)
            for ti, task in enumerate(('income', 'occupation')):
                k = 2 if task == 'income' else 6; p = p1 if task == 'income' else p2
                own = core.task_rows(labels[task], p, k); ref = core.task_rows(labels[task], neq[task], k)
                for m in ('accuracy', 'nll', 'brier'):
                    metric_diff = max(metric_diff, abs(float(np.mean(own[m])) - t['metrics'][task][m]))
                vals = {'accuracy_loss': float(np.mean(ref['accuracy'])) - float(np.mean(own['accuracy'])),
                        'nll_excess': float(np.mean(own['nll'])) - float(np.mean(ref['nll'])),
                        'brier_excess': float(np.mean(own['brier'])) - float(np.mean(ref['brier']))}
                for c in t['internal']:
                    if c['task'] == task:
                        internal_diff = max(internal_diff, abs(vals[c['metric']] - c['value']))
            tabs = view_tables(p1, d1, p2, d2)
            for view, x in tabs.items():
                rec = {'seed': s, 'key': key, 'view': view}
                for reader in ('LR_C1', 'HGB_15x200'):
                    for scaling in ('fold', 'global'):
                        a = crossfit_auc(x, y, folds, reader, scaling)
                        rec[f'{reader}|{scaling}'] = a
                        max_proxy[scaling] = max(max_proxy[scaling], abs(a - t['proxy'][view]['readers'][reader]))
                    rec[f'{reader}|table'] = t['proxy'][view]['readers'][reader]
                rec['value_table'] = t['proxy'][view]['value']
                rec['value_mine_fold'] = max(rec['LR_C1|fold'], rec['HGB_15x200|fold'])
                proxy_rows.append(rec)
    best = min(max_proxy, key=max_proxy.get)
    R.check('FIT2_proxy_recomputed_exactly_for_both_nominees', max_proxy[best] <= TOL, scaling_matching=best, max_abs=max_proxy)
    R.check('FIT2_proxy_values_equal_max_of_readers', all(abs(r['value_table'] - max(r['LR_C1|table'], r['HGB_15x200|table'])) == 0 for r in proxy_rows))
    R.check('FIT2_task_metrics_recomputed_for_nominees', metric_diff <= TOL, max_abs=metric_diff)
    R.check('FIT2_internal_clauses_vs_N_EQ_recomputed_for_nominees', internal_diff <= TOL, max_abs=internal_diff)
    R.numbers['proxy_recompute'] = proxy_rows
    R.numbers['proxy_max_abs_diff_by_scaling'] = max_proxy
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)

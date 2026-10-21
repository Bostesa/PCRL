"""V2 fixed-routing constructions (T2 TWB releases), own code.

Per capacity (C64 = PRC_TWIN 16 x 64, C128 = PRC_TWIN_C128 24 x 128) and version v:
 a. inputs re-hashed against their own records: source MODEL.pkl (MODEL.json), source RELEASE.npz (RELEASE.json), TASK.pkl
    (TASK.sha256), TWB MODEL.pkl (MODEL.json), TWB RELEASE.npz (RELEASE.json); TASK row-set hash = the admitted MODEL_FIT rows;
 b. out-of-fold routing recomputed: cell = argmin_k -sum_c q_c log C_kc (explicit per-class sums, first index on ties) for the
    cached out-of-fold teacher predictions and the saved source codewords; the best / second-best gap is reported (how far any
    arithmetic-order difference is from changing an assignment);
 c. B1 tables recomputed (sequential cell sums / counts, unoccupied cells keep the codeword, clip at 1e-6 and renormalize) and
    compared BITWISE with the TWB model's stored tables ('dec'); unoccupied counts and occupied cells against the TWB record;
 d. B0 = the stored codewords: the TWB book equals the source book (bitwise), the TWB teachers equal the source teachers (pickled
    bytes), construction / capacity / T* recorded as expected; full-pool max |B1 - A| and |B1 - B0| against the T1 record;
 e. every role: the TWB release and the source A release induce the same partition of the role's people (income, occupation and
    linked-pair views); the cell of every person is read off the A release (A table rows must be pairwise distinct) and the TWB
    vector must equal B1[cell] bitwise; distinct TWB vectors = occupied cells; people routed into out-of-fold-empty cells counted
    and compared with the TWB record;
 f. (--with-labels; MODEL_FIT income and occupation through frc.common.load, operation 'verify') the saved source A decoders are
    recomputed from the labels on the recomputed partition and compared bitwise (strong evidence that the recomputed partition is
    the one the source counted), and A and B1 count the same people;
 g. (--deployed [--deployed-roles R1,R2,...]) deployed routing replay (default MODEL_SELECT and MODEL_FIT; any admitted role's
    label-free features): the saved full-pool teachers' decision functions on the
    admitted compact coding (ohn.fam_a.compact_po, the admitted A1_PO_16 coding; disclosed) -> own probability transform ->
    own routing, against the cell read off the saved A release.

    python3 .../v2_routing.py --tag V1 [--with-labels] [--deployed]
    python3 .../v2_routing.py --selftest
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

CAPS = ('C64', 'C128')


def load_pickle_checked(pkl, meta_json, key='model_sha256'):
    buf = Path(pkl).read_bytes()
    meta = json.loads(Path(meta_json).read_text())
    ok = V.bytes_sha(buf) == meta.get(key)
    return pickle.loads(buf), {'sha256': V.bytes_sha(buf), 'record_agrees': ok}


def load_release_checked(d):
    rec = json.loads((Path(d) / 'RELEASE.json').read_text())
    h = V.file_sha(Path(d) / 'RELEASE.npz')
    out = {}
    with np.load(Path(d) / 'RELEASE.npz', allow_pickle=False) as z:
        for k in z.files:
            role, task = k.split('__')
            out.setdefault(role, {})[task] = z[k].copy()
    return out, rec, {'sha256': h, 'record_agrees': h == rec.get('release_sha256')}


def load_task(v):
    d = V.SRC_FA_CACHE / f'V{v}'
    buf = (d / 'TASK.pkl').read_bytes()
    ok = V.bytes_sha(buf) == (d / 'TASK.sha256').read_text().strip()
    return pickle.loads(buf), {'sha256': V.bytes_sha(buf), 'record_agrees': ok}


def probs_from_raw(raw):
    """Own probability transform of an HGB decision function: binary (1-D raw): p1 = 1 / (1 + exp(-raw)), columns (1 - p1, p1);
    multiclass: softmax after subtracting the row maximum; then clip at 1e-6 and renormalize."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-raw))
        P = np.column_stack([1.0 - p1, p1])
    else:
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        P = e / e.sum(axis=1, keepdims=True)
    return V.clip_norm(P)


def compare_tables(mine, stored):
    mine, stored = np.asarray(mine), np.asarray(stored)
    same_shape = mine.shape == stored.shape
    return {'bitwise_equal': bool(same_shape and np.array_equal(mine, stored)),
            'max_abs_diff': (V.max_abs(mine, stored) if same_shape else None), 'shape_mine': list(mine.shape),
            'shape_stored': list(stored.shape)}


def construction_core(oof_inc, oof_occ, CI, CO):
    """Own OOF routing and B1 tables. Returns a dict of arrays and summaries."""
    k0, gi = V.route_nearest(oof_inc, CI)
    l0, go = V.route_nearest(oof_occ, CO)
    TI, nI, uI = V.b1_table(k0, oof_inc, CI)
    TO, nO, uO = V.b1_table(l0, oof_occ, CO)
    return {'k0': k0, 'l0': l0, 'gap_income': gi, 'gap_occupation': go, 'B1': {'I': TI, 'O': TO}, 'n': {'I': nI, 'O': nO},
            'unoccupied': {'I': uI, 'O': uO}}


def role_checks(twb_rel, a_rel, a_dec, b1, n_oof):
    """Per role: partition identity of the TWB and A releases (three views), the cell read off the A release, TWB = B1[cell]
    bitwise, distinct vectors = occupied cells, people in out-of-fold-empty cells."""
    ids_b, nb = V.view_ids(twb_rel)
    ids_a, na = V.view_ids(a_rel)
    ident = {vw: V.same_partition(ids_b[vw], ids_a[vw]) for vw in ('income', 'occupation', 'pair')}
    k = V.table_index(a_rel['income'], a_dec['I'])
    l = V.table_index(a_rel['occupation'], a_dec['O'])
    found = bool((k >= 0).all() and (l >= 0).all())
    eq = bool(found and np.array_equal(twb_rel['income'], b1['I'][k]) and np.array_equal(twb_rel['occupation'], b1['O'][l]))
    occ_i, occ_o = len(np.unique(k)), len(np.unique(l))
    pair_cells = len(np.unique(k * 100000 + l))
    dist = {'income': nb['income'] == occ_i, 'occupation': nb['occupation'] == occ_o,
            'pair': len(np.unique(ids_b['pair'])) == pair_cells}
    empty = {'income': int((n_oof['I'][k] == 0).sum()), 'occupation': int((n_oof['O'][l] == 0).sum()), 'people': int(len(k))}
    return {'partition_identity': ident, 'cells_read_off_A_release': found, 'twb_equals_B1_of_cell_bitwise': eq,
            'distinct_vectors_equal_occupied_cells': dist, 'occupied_cells': {'income': occ_i, 'occupation': occ_o, 'pair': pair_cells},
            'distinct_twb_vectors': {'income': nb['income'], 'occupation': nb['occupation'],
                                     'pair': int(len(np.unique(ids_b['pair'])))},
            'people_routed_into_oof_empty_cells': empty}, (k, l)


def run(tag, with_labels, deployed, deployed_roles=('MODEL_SELECT', 'MODEL_FIT')):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    F_mf = C.features('MODEL_FIT')
    mf_sha = V.arr_sha(np.asarray(F_mf['row_id']))
    labels = None
    if with_labels:
        D = C.load('MODEL_FIT', ('features', 'income', 'occupation'), 'verify',
                   'verify: independent recomputation of the saved label-refit decoders A of PRC_TWIN / PRC_TWIN_C128 on the '
                   'recomputed out-of-fold partition (V2, separate code)')
        if not np.array_equal(np.asarray(D['row_id']), np.asarray(F_mf['row_id'])):
            raise AssertionError('MODEL_FIT label rows differ from the admitted feature rows')
        labels = {'income': np.asarray(D['income'], dtype=np.int64), 'occupation': np.asarray(D['occupation'], dtype=np.int64)}
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'with_labels': with_labels, 'deployed': deployed, 'per': {}}
    for v in (0, 1):
        task, task_h = load_task(v)
        oof_i = np.asarray(task['oof']['inc'], dtype=np.float64)
        oof_o = np.asarray(task['oof']['occ'], dtype=np.float64)
        for cid in CAPS:
            src = V.SOURCE_TWIN[cid]
            sd = V.SRC_RUNS / 'FA' / src / f'V{v}'
            td = V.RUNS / 'FA' / f'TWB_{cid}' / f'V{v}'
            key = f'{cid}_V{v}'
            res, probs = {}, []
            if not (td / 'MODEL.pkl').is_file():
                checks.append({'id': f'V2.TWB.{key}__{tag}', 'what': f'TWB_{cid} V{v} construction replay',
                               'inputs': {}, 'result': {'missing': str(V.rel(td))}, 'numerical_status': 'NOT_RUN',
                               'procedural_status': 'NOT_RUN'})
                continue
            M, mh = load_pickle_checked(sd / 'MODEL.pkl', sd / 'MODEL.json')
            TW, th = load_pickle_checked(td / 'MODEL.pkl', td / 'MODEL.json')
            a_rel, a_rec, ah = load_release_checked(sd)
            b_rel, b_rec, bh = load_release_checked(td)
            res['hashes'] = {'source_model': mh, 'twb_model': th, 'source_release': ah, 'twb_release': bh, 'task_cache': task_h,
                             'task_cache_equals_source_release_record': task_h['sha256'] == a_rec.get('task_cache_sha256'),
                             'task_cache_equals_twb_record': task_h['sha256'] == b_rec.get('task_cache_sha256'),
                             'task_rows_are_admitted_model_fit': task.get('row_id_sha256') == mf_sha}
            for nm, ok in (('source_model', mh['record_agrees']), ('twb_model', th['record_agrees']),
                           ('source_release', ah['record_agrees']), ('twb_release', bh['record_agrees']),
                           ('task_cache', task_h['record_agrees'])):
                if not ok:
                    probs.append(f'{nm} hash differs from its record')
            for nm in ('task_cache_equals_source_release_record', 'task_cache_equals_twb_record', 'task_rows_are_admitted_model_fit'):
                if not res['hashes'][nm]:
                    probs.append(nm)
            CI, CO = np.asarray(M['book']['CI']), np.asarray(M['book']['CO'])
            core = construction_core(oof_i, oof_o, CI, CO)
            res['oof_routing'] = {'min_gap_income': float(core['gap_income'].min()), 'min_gap_occupation': float(core['gap_occupation'].min()),
                                  'people_gap_below_1e-9': {'income': int((core['gap_income'] < 1e-9).sum()),
                                                            'occupation': int((core['gap_occupation'] < 1e-9).sum())},
                                  'occupied': {'income': int((core['n']['I'] > 0).sum()), 'occupation': int((core['n']['O'] > 0).sum()),
                                               'pair': int(len(np.unique(core['k0'] * 100000 + core['l0'])))},
                                  'unoccupied_cells': core['unoccupied']}
            # c. B1 tables bitwise
            res['B1_tables_vs_TWB_dec'] = {t: compare_tables(core['B1'][t], TW['dec'][t]) for t in ('I', 'O')}
            ti = b_rec.get('tables_info') or {}
            res['unoccupied_vs_record'] = {'I': [core['unoccupied']['I'], ti.get('unoccupied_prototype_fallback_I')],
                                           'O': [core['unoccupied']['O'], ti.get('unoccupied_prototype_fallback_O')]}
            res['occupied_vs_record'] = [res['oof_routing']['occupied'], b_rec.get('occupied_cells_oof')]
            num_ok = all(r['bitwise_equal'] for r in res['B1_tables_vs_TWB_dec'].values())
            num_ok &= all(a == b for a, b in res['unoccupied_vs_record'].values())
            num_ok &= res['occupied_vs_record'][0] == res['occupied_vs_record'][1]
            # rows of every table pairwise distinct (cells identifiable from vectors)
            res['table_rows_distinct'] = {f'B1_{t}': V.row_ids(core['B1'][t])[1] == core['B1'][t].shape[0] for t in ('I', 'O')}
            res['table_rows_distinct'].update({f'A_{t}': V.row_ids(M['dec'][t])[1] == np.asarray(M['dec'][t]).shape[0] for t in ('I', 'O')})
            num_ok &= all(res['table_rows_distinct'].values())
            # d. B0, teachers, metadata, full-pool differences
            bk, sb = TW['book'], M['book']
            book_eq = {k: (bool(np.array_equal(np.asarray(bk[k]), np.asarray(sb[k]))) if isinstance(sb[k], np.ndarray)
                           else bk.get(k) == sb[k]) for k in sb}
            book_eq['same_keys'] = set(bk) == set(sb)
            res['twb_book_equals_source_book'] = book_eq
            teach = {nm: pickle.dumps(TW['teachers'][nm], protocol=5) == pickle.dumps(M['teachers'][nm], protocol=5)
                     for nm in M['teachers']}
            teach['same_names'] = set(TW['teachers']) == set(M['teachers'])
            res['twb_teachers_equal_source_pickled_bytes'] = teach
            res['metadata'] = {'construction': TW.get('construction'), 'capacity': TW.get('capacity'), 'T': TW.get('T'),
                               'source_T': M.get('T'), 'feature_names_equal': tuple(TW['feature_names']) == tuple(M['feature_names'])}
            meta_ok = (TW.get('construction') == 'B1' and list(TW.get('capacity')) == list(V.CAPACITIES[cid])
                       and TW.get('T') == M.get('T') and res['metadata']['feature_names_equal'])
            if not (all(book_eq.values()) and all(teach.values()) and meta_ok):
                probs.append('TWB book / teachers / metadata differ from the source twin')
            fp = {'B1_minus_A': {t: V.max_abs(core['B1'][t], M['dec'][t]) for t in ('I', 'O')},
                  'B1_minus_B0': {t: V.max_abs(core['B1'][t], M['book']['C' + t]) for t in ('I', 'O')}}
            res['full_pool_max_abs'] = fp
            nest_p = V.RESULTS / 'fa' / f'NEST_{cid}_V{v}.json'
            if nest_p.is_file():
                sr = json.loads(nest_p.read_text()).get('source_reproduction', {})
                res['full_pool_vs_T1_record'] = {
                    'B1_minus_A': {t: [fp['B1_minus_A'][t], (sr.get('full_pool_max_abs_B1_minus_A') or {}).get(t)] for t in ('I', 'O')},
                    'B1_minus_B0': {t: [fp['B1_minus_B0'][t], (sr.get('full_pool_max_abs_B1_minus_B0') or {}).get(t)] for t in ('I', 'O')},
                    'fallbacks': [core['unoccupied'], sr.get('full_pool_b1_fallbacks')]}
                num_ok &= all(a == b for d in ('B1_minus_A', 'B1_minus_B0') for a, b in res['full_pool_vs_T1_record'][d].values())
            # e. every role
            per_role, rr_ok = {}, True
            emp_rec = b_rec.get('people_routed_into_oof_empty_cells') or {}
            pid_rec = b_rec.get('partition_identity_with_saved_A_release') or {}
            for role in V.ROLES6:
                rc, _ = role_checks(b_rel[role], a_rel[role], M['dec'], core['B1'], core['n'])
                rc['empty_vs_record'] = [rc['people_routed_into_oof_empty_cells'], emp_rec.get(role)]
                rc['record_partition_identity'] = pid_rec.get(role)
                per_role[role] = rc
                rr_ok &= (all(rc['partition_identity'].values()) and rc['twb_equals_B1_of_cell_bitwise'] and
                          all(rc['distinct_vectors_equal_occupied_cells'].values()) and
                          rc['people_routed_into_oof_empty_cells'] == emp_rec.get(role))
            res['roles'] = per_role
            num_ok &= rr_ok
            # f. A decoders from labels
            if labels is not None:
                AI = V.a_decoders(core['k0'], labels['income'], CI)
                AO = V.a_decoders(core['l0'], labels['occupation'], CO)
                res['A_decoders_from_labels_vs_source_dec'] = {'I': compare_tables(AI, M['dec']['I']),
                                                               'O': compare_tables(AO, M['dec']['O'])}
                cnt_i = np.bincount(core['k0'], minlength=CI.shape[0])
                cnt_o = np.bincount(core['l0'], minlength=CO.shape[0])
                res['A_and_B1_count_same_people'] = bool(np.array_equal(cnt_i, core['n']['I']) and np.array_equal(cnt_o, core['n']['O'])
                                                         and int(cnt_i.sum()) == len(labels['income']))
                num_ok &= all(r['bitwise_equal'] for r in res['A_decoders_from_labels_vs_source_dec'].values())
                num_ok &= res['A_and_B1_count_same_people']
            # g. deployed routing replay
            if deployed:
                from ohn import fam_a as FA                        # the admitted A1_PO_16 compact coding (data transform)
                dep = {}
                for role in deployed_roles:
                    F = C.features(role)
                    Z = FA.compact_po(np.asarray(F['X'], dtype=np.float64), tuple(map(str, F['feature_names'])))
                    qi = probs_from_raw(M['teachers']['inc'].decision_function(Z))
                    qo = probs_from_raw(M['teachers']['occ'].decision_function(Z))
                    kd, gdi = V.route_nearest(qi, CI)
                    ld, gdo = V.route_nearest(qo, CO)
                    ka = V.table_index(a_rel[role]['income'], M['dec']['I'])
                    la = V.table_index(a_rel[role]['occupation'], M['dec']['O'])
                    dep[role] = {'income_cells_equal': int((kd == ka).sum()), 'occupation_cells_equal': int((ld == la).sum()),
                                 'people': int(len(kd)), 'min_gap': [float(gdi.min()), float(gdo.min())],
                                 'A_release_bitwise_from_own_routing': bool(np.array_equal(np.asarray(M['dec']['I'])[kd], a_rel[role]['income'])
                                                                            and np.array_equal(np.asarray(M['dec']['O'])[ld], a_rel[role]['occupation'])),
                                 'TWB_release_bitwise_from_own_routing': bool(np.array_equal(core['B1']['I'][kd], b_rel[role]['income'])
                                                                              and np.array_equal(core['B1']['O'][ld], b_rel[role]['occupation'])),
                                 'oof_vs_deployed_agreement_model_fit': (float(((kd == core['k0']) & (ld == core['l0'])).mean())
                                                                         if role == 'MODEL_FIT' else None)}
                res['deployed_routing'] = dep
                num_ok &= all(d['A_release_bitwise_from_own_routing'] and d['TWB_release_bitwise_from_own_routing'] for d in dep.values())
            record['per'][key] = {k: x for k, x in res.items()}
            checks.append({'id': f'V2.TWB.{key}__{tag}',
                           'what': f'TWB_{cid} V{v}: own out-of-fold routing and B1 tables (bitwise vs the TWB model), B0 = source '
                                   f'book, teachers unchanged, partition identity with the source A release on all six roles, TWB '
                                   f'= B1[cell] bitwise, distinct vectors = occupied cells, empty-cell counts'
                                   + ('; A decoders from MODEL_FIT labels bitwise' if labels is not None else '')
                                   + ('; deployed routing replay' if deployed else ''),
                           'inputs': {'source_model': mh['sha256'], 'twb_model': th['sha256'], 'source_release': ah['sha256'],
                                      'twb_release': bh['sha256'], 'task_cache': task_h['sha256']},
                           'result': res, 'numerical_status': 'PASS' if num_ok else 'FAIL',
                           'procedural_status': 'PASS' if not probs else 'FAIL', 'procedural_findings': probs})
    V.write_record(f'V2_ROUTING__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], c.get('procedural_findings'))
    return 0


# ---------------------------------------------------------------- invented-data selftest
def selftest():
    rng = np.random.default_rng(20261021)
    ok = {}
    n, K1, K2 = 3000, 6, 16
    Qi = V.clip_norm(rng.dirichlet(np.ones(2) * 0.7, n))
    Qo = V.clip_norm(rng.dirichlet(np.ones(5) * 0.5, n))
    CI = V.clip_norm(Qi[rng.choice(n, K1, replace=False)])
    CO = V.clip_norm(Qo[rng.choice(n, K2, replace=False)])
    core = construction_core(Qi, Qo, CI, CO)
    # routing equals a matrix-product routing (alternative arithmetic) except possibly at exact ties
    alt = (-(Qo @ np.log(CO).T)).argmin(1)
    ok['routing_matches_matrix_form'] = bool((alt == core['l0']).mean() == 1.0)
    # sequential cell sums equal np.add.at in row order bitwise; B1 = alternative implementation bitwise
    S = np.zeros((K2, 5))
    np.add.at(S, core['l0'], Qo)
    cnt = np.bincount(core['l0'], minlength=K2)
    T = CO.copy()
    T[cnt > 0] = S[cnt > 0] / cnt[cnt > 0, None]
    T = np.clip(T, 1e-6, 1.0)
    T = T / T.sum(1, keepdims=True)
    ok['B1_bitwise_equals_add_at_form'] = bool(np.array_equal(T, core['B1']['O']))
    # an unoccupied cell keeps (clip-normalized) its codeword
    C3 = np.vstack([CO, V.clip_norm(np.array([[0.2, 0.2, 0.2, 0.2, 0.2]]))])
    T3, n3, u3 = V.b1_table(core['l0'], Qo, C3)
    ok['unoccupied_cell_keeps_codeword'] = u3 >= 1 and bool(np.array_equal(T3[-1], V.clip_norm(C3[-1:])[0]))
    # compare_tables flags a one-ulp perturbation
    P = core['B1']['O'].copy()
    P[3, 2] = np.nextafter(P[3, 2], 1.0)
    ok['compare_flags_one_ulp'] = not compare_tables(P, core['B1']['O'])['bitwise_equal']
    # role checks: a release from the same partition passes; a changed partition fails; TWB != B1[cell] fails
    y_i, y_o = rng.integers(0, 2, n), rng.integers(0, 5, n)
    A = {'I': V.a_decoders(core['k0'], y_i, CI), 'O': V.a_decoders(core['l0'], y_o, CO)}
    a_rel = {'income': A['I'][core['k0']], 'occupation': A['O'][core['l0']]}
    b_rel = {'income': core['B1']['I'][core['k0']], 'occupation': core['B1']['O'][core['l0']]}
    rc, _ = role_checks(b_rel, a_rel, A, core['B1'], core['n'])
    ok['role_checks_pass_on_consistent_release'] = (all(rc['partition_identity'].values()) and rc['twb_equals_B1_of_cell_bitwise']
                                                    and all(rc['distinct_vectors_equal_occupied_cells'].values()))
    bad = dict(b_rel)
    perm = core['l0'].copy()
    sel = np.flatnonzero(perm == perm[0])[:3]
    perm[sel] = (perm[sel] + 1) % K2
    bad['occupation'] = core['B1']['O'][perm]
    rc2, _ = role_checks(bad, a_rel, A, core['B1'], core['n'])
    ok['role_checks_flag_changed_partition'] = not rc2['partition_identity']['occupation'] and not rc2['twb_equals_B1_of_cell_bitwise']
    bad2 = dict(b_rel)
    bad2['income'] = A['I'][core['k0']]
    rc3, _ = role_checks(bad2, a_rel, A, core['B1'], core['n'])
    ok['role_checks_flag_wrong_table'] = not rc3['twb_equals_B1_of_cell_bitwise']
    # table_index refuses duplicate table rows
    try:
        V.table_index(a_rel['income'], np.vstack([A['I'], A['I'][:1]]))
        ok['duplicate_rows_refused'] = False
    except ValueError:
        ok['duplicate_rows_refused'] = True
    # A decoders: counts + 20 codeword pseudo-counts, alternative bincount form bitwise
    cnt2 = np.bincount(core['l0'] * 5 + y_o, minlength=K2 * 5).reshape(K2, 5).astype(np.float64)
    A2 = (cnt2 + 20.0 * CO) / (cnt2.sum(1, keepdims=True) + 20.0)
    A2 = np.clip(A2, 1e-6, 1.0)
    A2 = A2 / A2.sum(1, keepdims=True)
    ok['A_decoders_bitwise_equal_bincount_form'] = bool(np.array_equal(A2, A['O']))
    # probability transform: binary and multiclass rows sum to one and are clipped
    pb = probs_from_raw(rng.normal(size=50) * 30)
    pm = probs_from_raw(rng.normal(size=(50, 5)) * 30)
    ok['probability_transform_valid'] = bool(np.allclose(pb.sum(1), 1) and np.allclose(pm.sum(1), 1) and pb.min() > 0 and pm.min() > 0)
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {k: bool(x) for k, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--with-labels', action='store_true')
    ap.add_argument('--deployed', action='store_true')
    ap.add_argument('--deployed-roles', default='MODEL_SELECT,MODEL_FIT', help='comma list of roles for the deployed replay')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag, a.with_labels, a.deployed, tuple(r.strip() for r in a.deployed_roles.split(',') if r.strip()))


if __name__ == '__main__':
    sys.exit(main())

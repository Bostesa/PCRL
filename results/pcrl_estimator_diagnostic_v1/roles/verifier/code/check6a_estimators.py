"""CHECK6 (part a): estimator behaviour - C equals the DRF1 exact AUC and is the maximum over rankings (so C_same >= B_same on
every sample); C_nested, D_nested and D_bank re-implemented exactly; training / orientation never see test labels (call-argument
recording and label perturbation); sample_rep partitions by unit.

All samples are own synthetic draws (own sampler, own seeds); sample_rep is exercised only at repetition index 777777 (not a grid,
bank or smoke index)."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import estimators as E       # noqa: E402
from edg import simulate as SIM       # noqa: E402
from drf import finite_law as FL      # noqa: E402
from drf import solver as SV          # noqa: E402

T0 = V.utc_now()
VSEED = 202610251840
TOL = 1e-12
rec = {'check': 6, 'name': 'estimator_behaviour_part_a'}
fails = []


def exact_post_by_study_id(law, name):
    Vw = law['views'][name]
    wid, K, m1, m0 = V.study_law_exact_view(law, name)
    own2st = np.full(K, -1)
    own2st[wid] = Vw['inv']
    pe = V.posterior_exact(m1, m0)
    out_p, out_m1, out_m0 = [None] * K, [None] * K, [None] * K
    for w in range(K):
        out_p[own2st[w]], out_m1[own2st[w]], out_m0[own2st[w]] = pe[w], m1[w], m0[w]
    return out_p, out_m1, out_m0


# ------------------------------------------------------------------ (a) + (b): C_same == drf.solver / drf.finite_law / own exact;
#                                                                     C_same >= B_same (exact) on every sample
ab = {'samples': 0, 'max_abs_c_same_minus_exact': 0.0, 'max_abs_solver_minus_exact': 0.0, 'max_abs_finite_law_minus_exact': 0.0,
      'violations_C_lt_B_float_exact': 0, 'violations_C_lt_B_exacttie_exact': 0, 'violations_study_float_C_minus_B_lt_-1e-12': 0,
      'min_C_minus_B_float': 1.0, 'equalities_C_eq_B_exacttie': 0}
LAWS = {}
for fi, fam in enumerate(L.FAMILIES):
    for ip, pi in enumerate(L.PIS):
        for io, occ in enumerate(L.OCCS):
            law = L.build(fam, pi, occ)
            LAWS[(fam, pi, occ)] = law
            for iv, (name, Vw) in enumerate(law['views'].items()):
                pe, _, _ = exact_post_by_study_id(law, name)
                post = np.asarray(Vw['post'])
                for n, R in ((200, 6), (2000, 6), (20000, 2)):
                    for r in range(R):
                        rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(1, fi, ip, io, iv, n, r)))
                        c, y = V.draw_people(law['p'], law['q'], n, rng)
                        if y.min() == y.max():
                            continue
                        ids, K = Vw['inv'][c], Vw['K']
                        n1w = np.bincount(ids, weights=y, minlength=K).astype(np.int64)
                        n0w = np.bincount(ids, minlength=K).astype(np.int64) - n1w
                        occd = np.flatnonzero(n1w + n0w > 0)
                        a1, a0 = n1w[occd], n0w[occd]
                        C_ex = V.sample_auc_by_wire([Fr(int(x), int(x + z)) for x, z in zip(a1, a0)], a1, a0)
                        B_fl = V.sample_auc_by_wire([float(post[w]) for w in occd], a1, a0)
                        B_ex = V.sample_auc_by_wire([pe[w] for w in occd], a1, a0)
                        cs, _ = E.c_same(ids, y, K)
                        sv = float(SV.auc_star(n1w.astype(float), n0w.astype(float)))
                        fl = float(FL.auc_star(n1w.astype(float), n0w.astype(float)))
                        bk, _ = E.b_known(ids, y, post)
                        ab['samples'] += 1
                        ab['max_abs_c_same_minus_exact'] = max(ab['max_abs_c_same_minus_exact'], abs(cs - float(C_ex)))
                        ab['max_abs_solver_minus_exact'] = max(ab['max_abs_solver_minus_exact'], abs(sv - float(C_ex)))
                        ab['max_abs_finite_law_minus_exact'] = max(ab['max_abs_finite_law_minus_exact'], abs(fl - float(C_ex)))
                        ab['violations_C_lt_B_float_exact'] += int(C_ex < B_fl)
                        ab['violations_C_lt_B_exacttie_exact'] += int(C_ex < B_ex)
                        ab['violations_study_float_C_minus_B_lt_-1e-12'] += int(cs - bk < -1e-12)
                        ab['min_C_minus_B_float'] = min(ab['min_C_minus_B_float'], float(C_ex - B_fl))
                        ab['equalities_C_eq_B_exacttie'] += int(C_ex == B_ex)
rec['a_b_c_same_and_inequality'] = ab
if (ab['max_abs_c_same_minus_exact'] > TOL or ab['max_abs_solver_minus_exact'] > TOL or ab['max_abs_finite_law_minus_exact'] > TOL
        or ab['violations_C_lt_B_float_exact'] or ab['violations_C_lt_B_exacttie_exact']
        or ab['violations_study_float_C_minus_B_lt_-1e-12']):
    fails.append('a/b C_same equality or C >= B')
print('a/b', ab)

# ------------------------------------------------------------------ (c) C_same is the maximum over ALL rankings (tiny tables)
rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(3,)))
cm = {'tables': 0, 'max_abs_diff': 0.0, 'k6_tables': 0}
for t in range(104):
    k = 6 if t >= 100 else int(rng.integers(1, 6))
    n1w = [int(x) for x in rng.integers(0, 5, k)]
    n0w = [int(x) for x in rng.integers(0, 5, k)]
    if t % 3 == 0 and k > 1:
        n1w[1], n0w[1] = 2 * n1w[0], 2 * n0w[0]                 # proportional counts: an exactly tied empirical posterior
    if sum(n1w) == 0 or sum(n0w) == 0:
        continue
    ids = np.concatenate([np.repeat(np.arange(k), n1w), np.repeat(np.arange(k), n0w)])
    y = np.concatenate([np.ones(sum(n1w), dtype=np.int64), np.zeros(sum(n0w), dtype=np.int64)])
    cs, _ = E.c_same(ids, y, k)
    mx = V.max_auc_by_enumeration(n1w, n0w)
    cm['tables'] += 1
    cm['k6_tables'] += int(k == 6)
    cm['max_abs_diff'] = max(cm['max_abs_diff'], abs(cs - float(mx)))
rec['c_enumeration_max_over_rankings'] = cm
if cm['max_abs_diff'] > TOL:
    fails.append('c enumeration')
print('c', cm)

# ------------------------------------------------------------------ (d) C_nested, (e) D_nested, (f) D_bank: exact re-implementations
#                                                                     + call recording + label perturbation (n = 2,000)
orig_lookup = E.PRC.lookup_scores
CALLS = []


def recording_lookup(train_cells, train_sex, test_cells, a=1.0):
    CALLS.append((np.array(train_cells, copy=True), np.array(train_sex, copy=True), np.array(test_cells, copy=True), a))
    return orig_lookup(train_cells, train_sex, test_cells, a=a)


def exact_table(train_w, train_y, K, a):
    rbar = Fr(int(np.sum(train_y)), len(train_y))
    n = np.bincount(train_w, minlength=K)
    n1 = np.bincount(train_w, weights=train_y, minlength=K).astype(np.int64)
    return [(int(n1[w]) + a * rbar) / (int(n[w]) + a) if n[w] > 0 else rbar for w in range(K)]


dfx = {'views': 0, 'max_abs': {'C_nested': 0.0, 'D_nested_value': 0.0, 'D_nested_pop_fold_mean': 0.0, 'D_nested_pop_pooled': 0.0,
                               'D_bank_value': 0.0, 'D_bank_val_auc': 0.0, 'D_bank_pop': 0.0},
       'pop_pooled_exact_checked_views': 0, 'unseen_share_mismatch': 0, 'sign_mismatch': 0, 'call_argument_violations': 0,
       'score_label_perturbation_changed_fit_or_orientation': 0, 'score_label_perturbation_changed_value': 0,
       'fold_label_perturbation_changed_fold_scores': 0}
E.PRC.lookup_scores = recording_lookup
try:
    for fi, fam in enumerate(L.FAMILIES):
        for ip, pi in enumerate(L.PIS):
            for io, occ in enumerate(L.OCCS):
                law = LAWS[(fam, pi, occ)]
                for iv, (name, Vw) in enumerate(law['views'].items()):
                    rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(4, fi, ip, io, iv)))
                    n = 2000
                    c, y = V.draw_people(law['p'], law['q'], n, rng)
                    ids, K = Vw['inv'][c], Vw['K']
                    folds = rng.permutation(n) % 5
                    parts = rng.permutation(n) % 10
                    _, m1x, m0x = exact_post_by_study_id(law, name)
                    dfx['views'] += 1
                    # (d) C_nested
                    cn, _ = E.c_nested(ids, y, folds, K)
                    num = Fr(0)
                    for f in range(5):
                        m = folds == f
                        a1 = np.bincount(ids[m], weights=y[m], minlength=K).astype(np.int64)
                        a0 = np.bincount(ids[m], minlength=K).astype(np.int64) - a1
                        o = np.flatnonzero(a1 + a0 > 0)
                        num += int(m.sum()) * V.sample_auc_by_wire([Fr(int(a1[w]), int(a1[w] + a0[w])) for w in o], a1[o], a0[o])
                    dfx['max_abs']['C_nested'] = max(dfx['max_abs']['C_nested'], abs(cn - float(num / n)))
                    # (e) D_nested with call recording
                    CALLS.clear()
                    dn = E.d_nested(ids, y, folds, K, Vw['m1'], Vw['m0'])
                    viol = 0
                    if len(CALLS) != 10:
                        viol += 1
                    for f in range(5):
                        tr, te = folds != f, folds == f
                        for j, test_expect in ((2 * f, ids[te]), (2 * f + 1, np.arange(K))):
                            trc, trs, tst, a_ = CALLS[j]
                            if not (np.array_equal(trc, ids[tr]) and np.array_equal(trs, y[tr]) and np.array_equal(tst, test_expect)
                                    and a_ == 1.0):
                                viol += 1
                    dfx['call_argument_violations'] += viol
                    keys_p, unseen = [None] * n, 0
                    tabs, shares = [], []
                    for f in range(5):
                        tr, te = folds != f, folds == f
                        tab = exact_table(ids[tr], y[tr], K, Fr(1))
                        tabs.append(tab)
                        shares.append(Fr(int(te.sum()), n))
                        seen = np.bincount(ids[tr], minlength=K) > 0
                        unseen += int((~seen[ids[te]]).sum())
                        for i in np.flatnonzero(te):
                            keys_p[i] = tab[ids[i]]
                    val_ex = V.sample_auc_brute(y, exact_keys=keys_p)
                    pfm = sum(s_ * V.pop_auc_brute(t_, m1x, m0x) for s_, t_ in zip(shares, tabs))
                    dfx['max_abs']['D_nested_value'] = max(dfx['max_abs']['D_nested_value'], abs(dn['value'] - float(val_ex)))
                    dfx['max_abs']['D_nested_pop_fold_mean'] = max(dfx['max_abs']['D_nested_pop_fold_mean'],
                                                                   abs(dn['pop_fold_mean'] - float(pfm)))
                    if K <= 256:
                        pp = V.pop_auc_brute([t_[w] for t_ in tabs for w in range(K)],
                                             [s_ * m1x[w] for s_ in shares for w in range(K)],
                                             [s_ * m0x[w] for s_ in shares for w in range(K)])
                        dfx['max_abs']['D_nested_pop_pooled'] = max(dfx['max_abs']['D_nested_pop_pooled'], abs(dn['pop_pooled'] - float(pp)))
                        dfx['pop_pooled_exact_checked_views'] += 1
                    if unseen / n != dn['unseen_test_share']:
                        dfx['unseen_share_mismatch'] += 1
                    # fold-label perturbation: relabel fold 0's people; fold 0's held-out scores must not change
                    y3 = y.copy()
                    f0 = folds == 0
                    y3[f0] = rng.integers(0, 2, int(f0.sum()))
                    CALLS.clear()
                    E.d_nested(ids, y3, folds, K, Vw['m1'], Vw['m0'])
                    s_new = orig_lookup(CALLS[0][0], CALLS[0][1], CALLS[0][2])
                    s_old = orig_lookup(ids[folds != 0], y[folds != 0], ids[f0])
                    dfx['fold_label_perturbation_changed_fold_scores'] += int(not np.array_equal(s_new, s_old))
                    # (f) D_bank exact re-implementation
                    db = E.d_bank(ids, y, parts, K, Vw['m1'], Vw['m0'])
                    fit, val, scm = parts <= 4, (parts == 5) | (parts == 6), parts >= 7
                    tab = exact_table(ids[fit], y[fit], K, Fr(2))
                    va = V.sample_auc_brute(y[val], exact_keys=[tab[w] for w in ids[val]])
                    sign = 1 if va >= Fr(1, 2) else -1
                    otab = tab if sign == 1 else [1 - x for x in tab]
                    dfx['sign_mismatch'] += int(sign != db['sign'])
                    v_ex = V.sample_auc_brute(y[scm], exact_keys=[otab[w] for w in ids[scm]])
                    va_o = V.sample_auc_brute(y[val], exact_keys=[otab[w] for w in ids[val]])
                    pop_ex = V.pop_auc_brute(otab, m1x, m0x)
                    for k_, a_, b_ in (('D_bank_value', db['value'], v_ex), ('D_bank_val_auc', db['val_auc'], va_o),
                                       ('D_bank_pop', db['pop'], pop_ex)):
                        dfx['max_abs'][k_] = max(dfx['max_abs'][k_], abs(a_ - float(b_)))
                    nw_fit = np.bincount(ids[fit], minlength=K)
                    if float(np.mean(nw_fit[ids[scm]] == 0)) != db['unseen_score_share']:
                        dfx['unseen_share_mismatch'] += 1
                    # SCORE-label perturbation: fit, orientation, VALIDATION AUC and pop must be bitwise unchanged
                    y2 = y.copy()
                    y2[scm] = rng.integers(0, 2, int(scm.sum()))
                    db2 = E.d_bank(ids, y2, parts, K, Vw['m1'], Vw['m0'])
                    same = (db2['pop'] == db['pop'] and db2['val_auc'] == db['val_auc'] and db2['sign'] == db['sign']
                            and db2['unseen_score_share'] == db['unseen_score_share'])
                    dfx['score_label_perturbation_changed_fit_or_orientation'] += int(not same)
                    dfx['score_label_perturbation_changed_value'] += int(db2['value'] != db['value'])
finally:
    E.PRC.lookup_scores = orig_lookup
rec['d_e_f_nested_and_bank'] = dfx
if (max(dfx['max_abs'].values()) > TOL or dfx['unseen_share_mismatch'] or dfx['sign_mismatch'] or dfx['call_argument_violations']
        or dfx['score_label_perturbation_changed_fit_or_orientation'] or dfx['fold_label_perturbation_changed_fold_scores']):
    fails.append('d/e/f')
print('d/e/f', dfx)

# ------------------------------------------------------------------ (g) sample_rep partitions by unit (rep index 777777 only)
gs = {}
for cell in (5, 48):
    sc = SIM.scenarios()[cell]
    law = L.build(sc['family'], sc['pi'], sc['occ'])
    c, s, unit, ct, st, unit_t, folds, parts = SIM.sample_rep(sc, law, 777777)
    G = int(unit.max()) + 1
    one_fold = all(len(set(folds[unit == g])) == 1 for g in range(0, G, max(1, G // 2000)))
    one_part = all(len(set(parts[unit == g])) == 1 for g in range(0, G, max(1, G // 2000)))
    uf = np.unique(np.column_stack((unit, folds)), axis=0)
    up = np.unique(np.column_stack((unit, parts)), axis=0)
    one_fold = one_fold and len(uf) == G          # exhaustive: exactly one (unit, fold) pair per unit
    one_part = one_part and len(up) == G
    fu = np.bincount(uf[:, 1], minlength=5) / G
    pu = np.bincount(up[:, 1], minlength=10) / G
    gs[str(cell)] = {'scenario': sc, 'n': int(len(c)), 'n_test': int(len(ct)), 'units': G,
                     'household_never_split_across_folds_(sampled_units)': one_fold,
                     'household_never_split_across_parts_(sampled_units)': one_part,
                     'unit_share_per_fold': fu.tolist(), 'unit_share_FIT_VAL_SCORE': [float(pu[:5].sum()), float(pu[5:7].sum()),
                                                                                      float(pu[7:].sum())],
                     'test_sample_is_separate_draw': bool(ct is not c and (len(ct) != len(c) or not np.array_equal(ct, c)))}
    if not (one_fold and one_part and gs[str(cell)]['test_sample_is_separate_draw']):
        fails.append(f'g sample_rep cell {cell}')
rec['g_sample_rep'] = gs
print('g', gs)
rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
print(rec['status'], fails)
V.write_out('CHECK6a_estimators.json', rec, started_utc=T0)

"""V4: objective audit and source check with the verifier's own objective and routing (campaign `pcrl_finite_solver_adequacy_v1`).

Inputs (read-only, hash-checked): the cached MODEL_FIT out-of-fold statistics as the stages read them (NRM1 task cache TASK.pkl:
qI, qO; FRC1 SEXSIDE.pkl: pi = clip(P(S=1), 0.01, 0.99), qO0, qO1; both derived from MODEL_FIT training labels), the saved books
and partitions of the source stage (SOURCE_BOOKS_V<v>.pkl, SOURCE_PARTITIONS_V<v>.npz), the saved FRC1 releases (TWB_C64,
PRC_S2, HQ_S2 MODEL.pkl) and the records source/SOURCE_CHECK_V<v>.json and audit/OBJECTIVE_AUDIT_V<v>.json.
Own code (no fsa / frc / nrm function):
- objective J = mean CE(qI, CI[k]) + mean CE(qO, CO[l]) + lam (w_pair I(S; pair) + w_loc (I(S; k) + I(S; l))) + lam kappa
  mean[(1 - pi) / p0 |qO0 - CO[l]|^2 + pi / p1 |qO1 - CO[l]|^2] (plug-in information pi-weighted, rates clipped at 1e-9);
- the routing rule (FA_PRC_v1 section 1): joint argmin of CE + t [w_loc (pi a_I + b_I) + w_loc (pi a_O + b_O) + w_pair (pi a_P +
  b_P) + kappa guard] on the 20-pseudo-count tables (a, b from the clipped cell rates), tilt t = the book's own; HQ reads pi, qO0,
  qO1 as the twin-cell means stored in its book; per person the gap between the best and second-best joint cost is recorded;
- B1 tables (cell means of the out-of-fold teacher predictions, unoccupied cells keep the prototype, clipped at 1e-6);
- the partition hash sha256(dtype + shape + bytes of k, then of l).
Checks per version: every saved (family, solver, strength) state's objective, partition hash, rule consistency (own route of its
saved book reproduces its saved partition), emitted / prototype cross-entropies, fitted information, guard and re-routing against
SOURCE_CHECK; the audit's numbers for the saved FRC1 releases; the route S 'fitted' inputs from own numbers (written to
roles/verifier/ROUTE_S_FITTED_V<v>.json for V6).

    python3 .../v4_objective.py --selftest
    python3 .../v4_objective.py --version 0 --tag V1          (queue unit, stage verify)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/v4_objective.py'
GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FAMS = ('PRC', 'HQ')
SOLVERS = ('original', 'SV1', 'SV1_first')
CLIP = 1e-6
TOL = 1e-9


# ---------------------------------------------------------------- own primitives
def clipn(P):
    P = np.clip(np.asarray(P, dtype=np.float64), CLIP, 1.0)
    return P / P.sum(1, keepdims=True)


def xent(Q, C):
    return -(Q @ np.log(C).T)


def dep_ab(r, rbar):
    r = np.clip(np.asarray(r, dtype=np.float64), 0.01, 0.99)
    b = np.log((1.0 - r) / (1.0 - rbar))
    return np.log(r / rbar) - b, b


def plugin_mi(cells, w):
    cells = np.asarray(cells)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(cells, kind='stable')
    c = cells[order]
    starts = np.flatnonzero(np.concatenate(([True], c[1:] != c[:-1])))
    n = np.diff(np.concatenate((starts, [len(c)]))).astype(np.float64)
    r = np.clip(np.add.reduceat(w[order], starts) / n, 1e-9, 1 - 1e-9)
    rb = float(w.mean())
    return float(np.sum(n / len(c) * (r * np.log(r / rb) + (1 - r) * np.log((1 - r) / (1 - rb)))))


def partition_sha(k, l):
    h = hashlib.sha256()
    for a in (k, l):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def restricted(E, book):
    """HQ: pi, qO0, qO1 replaced by the stored twin-cell means of the twin's joint task cell of x."""
    if book.get('restrict') != 'twin_cells':
        if book.get('restrict') not in (None,):
            raise ValueError(f'restriction {book.get("restrict")} not handled')
        return E['pi'], E['qO0'], E['qO1']
    tw = book['twin']
    j = xent(E['qI'], tw['CI']).argmin(1) * len(tw['CO']) + xent(E['qO'], tw['CO']).argmin(1)
    return book['hq_pi'][j], book['hq_qO0'][j], book['hq_qO1'][j]


def route(E, book, chunk=1024):
    """Own implementation of the routing rule; returns (k, l, gap) with gap = second-best minus best joint cost."""
    CI, CO = book['CI'], book['CO']
    AI, AO = xent(E['qI'], CI), xent(E['qO'], CO)
    t = float(book.get('tilt', 0.0))
    n = len(AI)
    if t == 0.0:
        sI, sO = np.sort(AI, 1), np.sort(AO, 1)
        gap = np.minimum(sI[:, 1] - sI[:, 0], sO[:, 1] - sO[:, 0])
        return AI.argmin(1), AO.argmin(1), gap
    pi, q0, q1 = restricted(E, book)
    rbar = float(book['rbar'])
    aI, bI = dep_ab(book['rI'], rbar)
    aO, bO = dep_ab(book['rO'], rbar)
    aP, bP = dep_ab(book['rP'], rbar)
    AI = AI + t * book['w_loc'] * (pi[:, None] * aI[None, :] + bI[None, :])
    AO = AO + t * book['w_loc'] * (pi[:, None] * aO[None, :] + bO[None, :])
    if book['kappa'] > 0:
        g = ((1 - pi) / book['p0'])[:, None] * ((q0[:, None, :] - CO[None, :, :]) ** 2).sum(2) \
            + (pi / book['p1'])[:, None] * ((q1[:, None, :] - CO[None, :, :]) ** 2).sum(2)
        AO = AO + t * book['kappa'] * g
    KO = CO.shape[0]
    k = np.empty(n, dtype=np.int64)
    l = np.empty(n, dtype=np.int64)
    gap = np.empty(n)
    for a in range(0, n, chunk):
        s = slice(a, a + chunk)
        cost = AI[s, :, None] + AO[s, None, :] + t * book['w_pair'] * (pi[s, None, None] * aP[None] + bP[None])
        f = cost.reshape(cost.shape[0], -1)
        best = f.argmin(1)
        k[s], l[s] = best // KO, best % KO
        part = np.partition(f, 1, axis=1)
        gap[s] = part[:, 1] - part[:, 0]
    return k, l, gap


def objective(E, book, k, l):
    n = len(k)
    J = float(-(E['qI'] * np.log(book['CI'][k])).sum() / n - (E['qO'] * np.log(book['CO'][l])).sum() / n)
    lam = float(book['lam'])
    if lam > 0:
        pi = E['pi']
        KO = book['CO'].shape[0]
        J += lam * (book['w_pair'] * plugin_mi(k * KO + l, pi) + book['w_loc'] * (plugin_mi(k, pi) + plugin_mi(l, pi)))
        if book['kappa'] > 0:
            g = (1 - pi) / book['p0'] * ((E['qO0'] - book['CO'][l]) ** 2).sum(1) + pi / book['p1'] * ((E['qO1'] - book['CO'][l]) ** 2).sum(1)
            J += lam * book['kappa'] * float(np.mean(g))
    return J


def b1_tables(book, k, l, E):
    out = {}
    for key, cells, Q, Cw in (('I', k, E['qI'], book['CI']), ('O', l, E['qO'], book['CO'])):
        K = Cw.shape[0]
        T = np.array(Cw, dtype=np.float64, copy=True)
        for c in np.unique(cells):
            T[c] = Q[cells == c].mean(0)
        out[key] = clipn(T)
    return out


def summary(E, book, k, l, k0, l0, tables=None):
    n = len(k)
    KO = book['CO'].shape[0]
    out = {'objective': objective(E, book, k, l), 'partition_sha256': partition_sha(k, l),
           'rerouted_vs_twin': float(((k != k0) | (l != l0)).mean()), 'income_rerouted': float((k != k0).mean()),
           'occupation_rerouted': float((l != l0).mean()), 'tilt': float(book.get('tilt', 0.0)),
           'occupied_pair_cells': int(len(np.unique(k * KO + l))),
           'fitted_pi_pair_mi': plugin_mi(k * KO + l, E['pi']), 'fitted_pi_income_mi': plugin_mi(k, E['pi']),
           'fitted_pi_occupation_mi': plugin_mi(l, E['pi']),
           'proto_ce_income': float(-(E['qI'] * np.log(book['CI'][k])).sum() / n),
           'proto_ce_occupation': float(-(E['qO'] * np.log(book['CO'][l])).sum() / n)}
    if tables is not None:
        out['emitted_ce_income'] = float(-(E['qI'] * np.log(tables['I'][k])).sum() / n)
        out['emitted_ce_occupation'] = float(-(E['qO'] * np.log(tables['O'][l])).sum() / n)
        out['emitted_minus_proto_max_abs'] = float(max(np.abs(tables['I'] - book['CI']).max(), np.abs(tables['O'] - book['CO']).max()))
        if 'p0' in book:
            for nm, CO in (('proto', book['CO']), ('emitted', tables['O'])):
                g = (1 - E['pi']) / book['p0'] * ((E['qO0'] - CO[l]) ** 2).sum(1) + E['pi'] / book['p1'] * ((E['qO1'] - CO[l]) ** 2).sum(1)
                out[f'guard_{nm}'] = float(np.mean(g))
    return out


def compare_summary(mine, rec, tol=TOL):
    bad = []
    for key, rv in rec.items():
        if key not in mine:
            continue
        mv = mine[key]
        if isinstance(rv, str) or isinstance(mv, str):
            if rv != mv:
                bad.append(f'{key}: mine {mv} recorded {rv}')
        elif isinstance(rv, float) and np.isnan(rv):
            continue
        elif abs(float(mv) - float(rv)) > tol:
            bad.append(f'{key}: mine {mv} recorded {rv}')
    return bad


# ---------------------------------------------------------------- real run
def load_statistics(v, adm, row_id):
    tp = V.NRM_PRIV / 'fa_cache' / f'V{v}' / 'TASK.pkl'
    sp = V.FRC_PRIV / 'fa_cache' / f'V{v}' / 'SEXSIDE.pkl'
    pins = {'TASK': adm['source_private_artefacts'][f'fa_cache/V{v}/TASK.pkl']['sha256'],
            'SEXSIDE': adm['frc_private_artefacts'][f'frc/fa_cache/V{v}/SEXSIDE.pkl']['sha256']}
    got = {'TASK': V.sha_file(tp), 'SEXSIDE': V.sha_file(sp)}
    if got != pins:
        raise PermissionError(f'cached statistics differ from their pins: {got} vs {pins}')
    task = pickle.loads(tp.read_bytes())
    ss = pickle.loads(sp.read_bytes())
    rs = V.arr_sha(np.asarray(row_id))
    if task['row_id_sha256'] != rs or ss['row_id_sha256'] != rs:
        raise PermissionError('a cache was fitted on other rows')
    E = {'qI': np.asarray(task['oof']['inc'], dtype=np.float64), 'qO': np.asarray(task['oof']['occ'], dtype=np.float64),
         'pi': np.clip(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], 0.01, 0.99),
         'qO0': np.asarray(ss['oof']['occ_s'][0], dtype=np.float64), 'qO1': np.asarray(ss['oof']['occ_s'][1], dtype=np.float64)}
    return E, got


def frc_model(adm, cfg, v):
    d = V.FRC_PRIV / 'runs' / 'FA' / cfg / f'V{v}'
    if V.sha_file(d / 'MODEL.pkl') != adm['frc_private_artefacts'][f'frc/runs/FA/{cfg}/V{v}']['MODEL.pkl']:
        raise PermissionError(f'{cfg} V{v} MODEL.pkl differs from its pin')
    return pickle.loads((d / 'MODEL.pkl').read_bytes())


def main_real(v, tag):
    adm = V.load_json(V.RES / 'SOURCE_ADMISSION.json')
    rsi = V.load_json(V.RES / 'decisions' / 'ROUTE_S_INPUT.json')
    C = V.admitted_loaders()
    feats = C.features('MODEL_FIT')
    E, cache_sha = load_statistics(v, adm, feats['row_id'])
    bp = V.PRIV / 'solver' / f'SOURCE_BOOKS_V{v}.pkl'
    pp = V.PRIV / 'solver' / f'SOURCE_PARTITIONS_V{v}.npz'
    files_ok = {bp.name: V.sha_file(bp) == rsi['private_file_sha256'].get(bp.name),
                pp.name: V.sha_file(pp) == rsi['private_file_sha256'].get(pp.name)}
    books = pickle.loads(bp.read_bytes())
    with np.load(pp) as z:
        parts = {key: z[key].copy() for key in z.files}
    src = V.load_json(V.RES / 'source' / f'SOURCE_CHECK_V{v}.json')
    aud = V.load_json(V.RES / 'audit' / f'OBJECTIVE_AUDIT_V{v}.json')
    tw = frc_model(adm, 'TWB_C64', v)
    k0, l0, g0 = route(E, tw['book'])
    checks, rows, bad_all, min_gap = [], {}, [], {}
    # ---- source states
    for fam in FAMS:
        for solver in SOLVERS:
            for lam in GRID:
                key = f'{fam}|{solver}|{lam}'
                b = books[key]
                k, l = parts[f'{key}|k'], parts[f'{key}|l']
                kr, lr, gap = route(E, b)
                tabs = b1_tables(b, k, l, E)
                m = summary(E, b, k, l, k0, l0, tabs)
                m['rule_consistent'] = bool(np.array_equal(kr, k) and np.array_equal(lr, l))
                m['n_route_mismatch'] = int(((kr != k) | (lr != l)).sum())
                m['min_route_gap'] = float(gap.min())
                rec = src['runs'][f'{fam}|{solver}']['per_strength'][str(lam)]
                bad = compare_summary(m, {kk: vv for kk, vv in rec.items() if kk in m and kk != 'rule_consistent'})
                if bool(rec.get('rule_consistent')) != m['rule_consistent']:
                    bad.append(f'rule_consistent mine {m["rule_consistent"]} recorded {rec.get("rule_consistent")}')
                if float(b['lam']) != lam:
                    bad.append(f'book lam {b["lam"]} differs from {lam}')
                rows[key] = {'mine': m, 'mismatches': bad}
                bad_all += [f'{key}: {x}' for x in bad]
                min_gap[key] = m['min_route_gap']
    not_consistent = [k for k, r in rows.items() if not r['mine']['rule_consistent']]
    sv1_bad = [k for k in not_consistent if '|SV1|' in k]
    n_obj = len(rows)
    mx = max(abs(r['mine']['objective'] - src['runs'][f'{k.split("|")[0]}|{k.split("|")[1]}']['per_strength'][k.split('|')[2]]['objective'])
             for k, r in rows.items())
    checks.append({'id': f'V4.SOURCE_STATES__V{v}__{tag}', 'block': 'V4', 'kind': 'numerical', 'status': 'PASS' if not bad_all else 'FAIL',
                   'summary': f'{n_obj} saved source states (PRC / HQ x original / SV1 / SV1_first x 6 strengths): own objective, '
                              f'partition hash, B1 emitted and prototype cross-entropies, fitted information, guard and re-routing vs '
                              f'SOURCE_CHECK_V{v} (max |d objective| {mx:.2e}); own route of each saved book reproduces its partition '
                              f'in {n_obj - len(not_consistent)}/{n_obj} (SV1 not consistent: {sv1_bad}); smallest best-vs-second '
                              f'routing cost gap {min(min_gap.values()):.2e}; {len(bad_all)} mismatches'
                              + (f': {bad_all[:5]}' if bad_all else ''),
                   'detail': {'n_states': n_obj, 'max_abs_objective_diff': mx, 'rule_consistent': n_obj - len(not_consistent),
                              'not_rule_consistent': not_consistent, 'min_route_gap': min(min_gap.values()), 'n_mismatch': len(bad_all),
                              'private_files_match_route_s_record': files_ok, 'cache_sha256': cache_sha},
                   'record': rows, 'script': SCRIPT})
    # ---- SV1 against the original (the paired summary) and the route S 'fitted' inputs from own numbers
    cmp_bad, fitted = [], {}
    for fam in FAMS:
        for lam in GRID:
            o, s = rows[f'{fam}|original|{lam}']['mine'], rows[f'{fam}|SV1|{lam}']['mine']
            mine = {'objective_sv1_minus_original': s['objective'] - o['objective'],
                    'same_partition': s['partition_sha256'] == o['partition_sha256'],
                    'fitted_pair_mi_sv1_minus_original': s['fitted_pi_pair_mi'] - o['fitted_pi_pair_mi'],
                    'emitted_ce_total_sv1_minus_original': (s['emitted_ce_income'] + s['emitted_ce_occupation'])
                    - (o['emitted_ce_income'] + o['emitted_ce_occupation']),
                    'rerouted_sv1': s['rerouted_vs_twin'], 'rerouted_original': o['rerouted_vs_twin']}
            recd = src['sv1_vs_original'][f'{fam}|{lam}']
            for kk, vv in recd.items():
                if isinstance(vv, bool) or isinstance(mine[kk], bool):
                    if vv != mine[kk]:
                        cmp_bad.append(f'{fam}|{lam} {kk}')
                elif abs(vv - mine[kk]) > TOL:
                    cmp_bad.append(f'{fam}|{lam} {kk}: mine {mine[kk]} recorded {vv}')
            if fam == 'PRC':
                fitted[str(lam)] = {'original': o['objective'], 'revised': s['objective'], 'revised_deployable': s['rule_consistent'],
                                    'distinct_from_original': s['partition_sha256'] != o['partition_sha256'],
                                    'distinct_from_twin': s['rerouted_vs_twin'] > 0,
                                    'original_rerouted_vs_twin': o['rerouted_vs_twin'], 'revised_rerouted_vs_twin': s['rerouted_vs_twin']}
    rf = rsi['fitted'][str(v)]
    fit_bad = []
    for lam, e in fitted.items():
        for kk, vv in e.items():
            r = rf[lam][kk]
            if (isinstance(vv, bool) and vv != r) or (not isinstance(vv, bool) and abs(vv - r) > TOL):
                fit_bad.append(f'{lam} {kk}: mine {vv} recorded {r}')
    improves = {lam: e['revised'] < e['original'] - 1e-9 for lam, e in fitted.items()}
    V.write_once_json(V.VER / f'ROUTE_S_FITTED_V{v}.json', {'version': v, 'fitted': fitted, 'by': 'independent verifier (v4_objective.py)',
                                                            'utc': V.utc_now(), 'unit_id': os.environ.get('FSA_UNIT_ID')})
    checks.append({'id': f'V4.SV1_VS_ORIGINAL__V{v}__{tag}', 'block': 'V4', 'kind': 'numerical',
                   'status': 'PASS' if not cmp_bad and not fit_bad else 'FAIL',
                   'summary': f'SV1 minus original per family and strength (objective, same partition, fitted pair information, emitted '
                              f'cross-entropy, re-routing) vs SOURCE_CHECK sv1_vs_original, and the route S fitted inputs (PRC) vs '
                              f'decisions/ROUTE_S_INPUT.json: {len(cmp_bad)} + {len(fit_bad)} mismatches; own reading: SV1 lowers the '
                              f'fitted objective beyond 1e-9 at {sorted(k for k, x in improves.items() if x)} '
                              f'(d J {[round(fitted[k]["revised"] - fitted[k]["original"], 6) for k in sorted(fitted, key=float)]})',
                   'detail': {'mismatches': (cmp_bad + fit_bad)[:10], 'sv1_improves': improves,
                              'd_objective': {k: fitted[k]['revised'] - fitted[k]['original'] for k in fitted}},
                   'record': {'fitted_mine': fitted, 'mismatches': cmp_bad + fit_bad}, 'script': SCRIPT})
    # ---- the audit's numbers for the saved FRC1 releases and the original chain
    a_bad, a_rows = [], {}
    for fam, cfg in (('PRC', 'PRC_S2'), ('HQ', 'HQ_S2'), ('TWIN', 'TWB_C64')):
        M = frc_model(adm, cfg, v)
        k, l, gap = route(E, M['book'])
        m = summary(E, M['book'], k, l, k0, l0, M['dec'])
        rec = aud[f'{fam}_emitted_vs_proto']
        bad = compare_summary(m, {kk: vv for kk, vv in rec.items() if kk in m})
        if fam != 'TWIN':
            last = rows[f'{fam}|original|8.0']['mine']
            if last['partition_sha256'] != m['partition_sha256']:
                bad.append('saved FRC1 partition differs from the source original chain at strength 8')
            ch = aud[f'{fam}_chain']['per_strength']
            for lam in GRID:
                d = abs(ch[str(lam)]['objective'] - rows[f'{fam}|original|{lam}']['mine']['objective'])
                if d > TOL:
                    bad.append(f'audit chain objective at {lam} differs from own objective of the source original state by {d:.2e}')
            if abs(aud[f'{fam}_tilt_line_at_8']['current_objective'] - m['objective']) > TOL:
                bad.append('tilt-line current objective differs')
        a_rows[cfg] = {'mine': m, 'min_route_gap': float(gap.min()), 'mismatches': bad}
        a_bad += [f'{cfg}: {x}' for x in bad]
    checks.append({'id': f'V4.AUDIT__V{v}__{tag}', 'block': 'V4', 'kind': 'numerical', 'status': 'PASS' if not a_bad else 'FAIL',
                   'summary': f'OBJECTIVE_AUDIT_V{v}: the saved FRC1 PRC_S2 / HQ_S2 / TWB_C64 releases routed by own code on the cached '
                              f'statistics (objective, partition hash, emitted tables vs prototypes, guard, information, re-routing), the '
                              f'original chain objectives per strength and the tilt-line current objective: {len(a_bad)} mismatches'
                              + (f': {a_bad[:4]}' if a_bad else ''),
                   'detail': {'n_mismatch': len(a_bad), 'min_route_gap': {c: r['min_route_gap'] for c, r in a_rows.items()}},
                   'record': a_rows, 'script': SCRIPT})
    V.Checks().append(checks)
    print(json.dumps({c['id']: (c['status'], c['summary'][:500]) for c in checks}, indent=1))
    return 0


# ---------------------------------------------------------------- selftest (invented statistics and books)
def _fake(rng, n=500, KI=3, KO=4):
    def simplex(m, K, a=2.0):
        return rng.dirichlet(np.ones(K) * a, m)
    E = {'qI': clipn(simplex(n, 2)), 'qO': clipn(simplex(n, 5)), 'pi': np.clip(rng.random(n), 0.01, 0.99)}
    E['qO0'], E['qO1'] = clipn(simplex(n, 5)), clipn(simplex(n, 5))
    book = {'CI': clipn(simplex(KI, 2)), 'CO': clipn(simplex(KO, 5)), 'lam': 2.0, 'tilt': 0.7, 'restrict': None, 'w_pair': 1.0,
            'w_loc': 1.0, 'kappa': 1.0, 'm_shrink': 20.0, 'rbar': float(E['pi'].mean()), 'p0': float(1 - E['pi'].mean()),
            'p1': float(E['pi'].mean()), 'rI': rng.uniform(0.2, 0.8, KI), 'rO': rng.uniform(0.2, 0.8, KO),
            'rP': rng.uniform(0.2, 0.8, (KI, KO))}
    return E, book


def selftest():
    rng = np.random.default_rng(11)
    res = []
    E, book = _fake(rng)
    k, l, gap = route(E, book)
    # brute-force cost for every person and joint cell from the rule's definition
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    rb = book['rbar']

    def ab(r):
        r = min(max(r, 0.01), 0.99)
        return np.log(r / rb) - np.log((1 - r) / (1 - rb)), np.log((1 - r) / (1 - rb))
    ok_route, t = True, book['tilt']
    for i in range(0, 500, 37):
        best, arg = np.inf, None
        for a in range(KI):
            for b in range(KO):
                ce = -sum(E['qI'][i] * np.log(book['CI'][a])) - sum(E['qO'][i] * np.log(book['CO'][b]))
                p = E['pi'][i]
                aI, bI = ab(book['rI'][a]); aO, bO = ab(book['rO'][b]); aP, bP = ab(book['rP'][a, b])
                gd = (1 - p) / book['p0'] * sum((E['qO0'][i] - book['CO'][b]) ** 2) + p / book['p1'] * sum((E['qO1'][i] - book['CO'][b]) ** 2)
                c = ce + t * (p * aI + bI + p * aO + bO + p * aP + bP + gd)
                if c < best - 1e-15:
                    best, arg = c, (a, b)
        ok_route &= (int(k[i]), int(l[i])) == arg
    res.append(('route_vs_bruteforce', ok_route, None))
    b0 = dict(book, tilt=0.0)
    k0, l0, _ = route(E, b0)
    res.append(('route_tilt0_nearest', np.array_equal(k0, xent(E['qI'], book['CI']).argmin(1)) and np.array_equal(l0, xent(E['qO'], book['CO']).argmin(1)), None))
    # objective against a loop over persons
    J = objective(E, book, k, l)
    n = len(k)
    ce = np.mean([-sum(E['qI'][i] * np.log(book['CI'][k[i]])) - sum(E['qO'][i] * np.log(book['CO'][l[i]])) for i in range(n)])

    def mi(cells):
        out = 0.0
        rbm = E['pi'].mean()
        for c in set(cells.tolist()):
            m_ = cells == c
            r = min(max(E['pi'][m_].mean(), 1e-9), 1 - 1e-9)
            out += m_.mean() * (r * np.log(r / rbm) + (1 - r) * np.log((1 - r) / (1 - rbm)))
        return out
    gd = np.mean([(1 - E['pi'][i]) / book['p0'] * sum((E['qO0'][i] - book['CO'][l[i]]) ** 2)
                  + E['pi'][i] / book['p1'] * sum((E['qO1'][i] - book['CO'][l[i]]) ** 2) for i in range(n)])
    Jb = ce + 2.0 * (mi(k * KO + l) + mi(k) + mi(l)) + 2.0 * gd
    res.append(('objective_vs_loops', abs(J - Jb) < 1e-12, (J, Jb)))
    # HQ restriction: pi, qO0, qO1 replaced by the stored twin-cell tables
    hb = dict(book, restrict='twin_cells', twin={'CI': book['CI'], 'CO': book['CO']}, hq_pi=rng.uniform(0.1, 0.9, KI * KO),
              hq_qO0=clipn(rng.dirichlet(np.ones(5), KI * KO)), hq_qO1=clipn(rng.dirichlet(np.ones(5), KI * KO)))
    pi_r, q0r, q1r = restricted(E, hb)
    jj = k0 * KO + l0
    res.append(('hq_restriction', np.array_equal(pi_r, hb['hq_pi'][jj]) and np.array_equal(q0r, hb['hq_qO0'][jj]), None))
    kh, lh, _ = route(E, hb)
    Eh = dict(E, pi=pi_r, qO0=q0r, qO1=q1r)
    kh2, lh2, _ = route(Eh, dict(hb, restrict=None))
    res.append(('hq_route_equals_route_on_restricted_statistics', np.array_equal(kh, kh2) and np.array_equal(lh, lh2), None))
    # B1 tables: occupied cells are member means; unoccupied keep the prototype
    tb = b1_tables(book, k, l, E)
    occ_ok = all(np.allclose(tb['O'][c], clipn(E['qO'][l == c].mean(0)[None])[0], atol=1e-15) for c in np.unique(l))
    un = [c for c in range(KO) if c not in set(l.tolist())]
    res.append(('b1_tables', occ_ok and all(np.allclose(tb['O'][c], clipn(book['CO'][c][None])[0]) for c in un), un))
    res.append(('partition_sha', partition_sha(k, l) == partition_sha(k.copy(), l.copy()) and partition_sha(k, l) != partition_sha(l, k), None))
    res.append(('plugin_mi', abs(plugin_mi(k * KO + l, E['pi']) - mi(k * KO + l)) < 1e-14, None))
    res.append(('route_gap_nonnegative', bool((gap >= 0).all()), float(gap.min())))
    n_ok = sum(r[1] for r in res)
    for name, ok, info in res:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  [{info}]'))
    print(f'v4_objective selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--version', type=int, choices=(0, 1))
    ap.add_argument('--tag', default='V1')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.version is None:
        ap.error('--version is required')
    if not os.environ.get('FSA_UNIT_ID'):
        sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
    sys.exit(main_real(a.version, a.tag))

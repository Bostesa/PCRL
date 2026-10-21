"""V5b B1 tables of the saved protected-stage configurations on their OWN out-of-fold partitions (label-free), own code.

For every saved FA configuration other than the twins (PRC_*, HQ_*, SHUF_*, ORD_*, the R1 control UNT_BR; per version):
- MODEL.pkl against MODEL.json; RELEASE.npz against RELEASE.json; SEX-side cache against its .sha256;
- the configuration's out-of-fold statistics: qI, qO = the cached out-of-fold task teachers (source TASK.pkl); pi = clip(out-of-fold
  P(SEX = 1), 0.01, 0.99) and qO0 / qO1 from the P1 SEX-side cache (this campaign's fa_cache/V<v>/SEXSIDE.pkl);
- my own routing of those statistics through the saved book:
    tilt 0 (twin, ORD): nearest codeword per task (cross-entropy, first index on ties);
    UNT (R1): nearest codeword, except people whose keyed hash SHA-256(tag | row bytes of the compact features), first 8 bytes /
      2^64, is below f_O (tag + '|O') / f_I (tag + '|I') take the second-nearest codeword (stable order);
    tilt t > 0 (PRC, HQ, SHUF): argmin over joint cells (k, l) of CE(qI, cI_k) + CE(qO, cO_l) + t w_loc [pi a_I,k + b_I,k] +
      t w_loc [pi a_O,l + b_O,l] + t kappa guard_l + t w_pair [pi a_P,kl + b_P,kl], with a = log(r / rbar) - b,
      b = log((1 - r) / (1 - rbar)), r clipped to [0.01, 0.99]; guard_l = (1 - pi) / p0 |qO0 - cO_l|^2 + pi / p1 |qO1 - cO_l|^2;
      HQ: pi, qO0, qO1 replaced by the stored twin-cell tables at the twin's nearest-codeword cell k_tw * KO + l_tw;
      SHUF (this campaign's mode): pi replaced by the stored bank at donor SHA-256(tag | row bytes) first 8 bytes mod bank size,
      and qO0 = qO1 = qO;
  the joint argmin takes the first flat index (k major) on ties; the best / second-best gap is reported;
- the B1 tables recomputed on that partition (sequential cell sums, unoccupied cells keep the codeword, clip-renormalize) and
  compared BITWISE with the stored 'dec'; occupied / unoccupied counts against the record;
- every role: each released vector is a row of the stored tables (cells readable), and distinct vectors = occupied cells.
The compact coding of MODEL_FIT (for the keyed hashes) uses the admitted A1_PO_16 coding ohn.fam_a.compact_po (data transform,
disclosed).

    python3 .../v5b_protected_tables.py --tag V1
    python3 .../v5b_protected_tables.py --selftest
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

SKIP = ('TWB_C64', 'TWB_C128', 'TWB_C256', 'TWA_C256')
SHUF_MODE = 'shuf_pi_own_guard'


def keyed_unit(Z, tag):
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float64))
    t = tag.encode()
    return np.array([int.from_bytes(hashlib.sha256(t + Z[i].tobytes()).digest()[:8], 'big') / 2.0 ** 64 for i in range(len(Z))])


def donor(Z, tag, n_bank):
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float64))
    t = tag.encode()
    return np.array([int.from_bytes(hashlib.sha256(t + Z[i].tobytes()).digest()[:8], 'big') % n_bank for i in range(len(Z))],
                    dtype=np.int64)


def dep_ab(r, rbar):
    r = np.minimum(np.maximum(np.asarray(r, dtype=np.float64), 0.01), 0.99)
    b = np.log((1 - r) / (1 - rbar))
    return np.log(r / rbar) - b, b


def embedding(E, book):
    """Statistics the book's rule reads: HQ twin-cell tables, the SHUF donor copy, or E itself."""
    R = dict(E)
    if book.get('frc_embed') == SHUF_MODE:
        R['pi'] = np.asarray(book['shuf_bank_pi'], dtype=np.float64)[donor(E['Z'], book['shuf_tag'], len(book['shuf_bank_pi']))]
        R['qO0'] = E['qO']
        R['qO1'] = E['qO']
    elif book.get('restrict') == 'twin_cells':
        tw = book['twin']
        kt, _ = V.route_nearest(E['qI'], tw['CI'])
        lt, _ = V.route_nearest(E['qO'], tw['CO'])
        j = kt * len(tw['CO']) + lt
        R['pi'] = np.asarray(book['hq_pi'])[j]
        R['qO0'] = np.asarray(book['hq_qO0'])[j]
        R['qO1'] = np.asarray(book['hq_qO1'])[j]
    elif book.get('restrict') not in (None,):
        raise ValueError(f'unknown restriction {book.get("restrict")!r}')
    return R


def route_book(E, book, chunk=1024):
    """(k, l, min gap) of statistics E under a saved book (see the module docstring)."""
    CI, CO = np.asarray(book['CI']), np.asarray(book['CO'])
    if book.get('frc_route') == 'unt':
        AI, AO = V.xent_costs(E['qI'], CI), V.xent_costs(E['qO'], CO)
        oi, oo = np.argsort(AI, axis=1, kind='stable'), np.argsort(AO, axis=1, kind='stable')
        uI, uO = keyed_unit(E['Z'], book['unt_tag'] + '|I'), keyed_unit(E['Z'], book['unt_tag'] + '|O')
        k = np.where(uI < book['unt_f_I'], oi[:, 1], oi[:, 0])
        l = np.where(uO < book['unt_f_O'], oo[:, 1], oo[:, 0])
        r = np.arange(len(k))
        gap = min(float((AI[r, oi[:, 2]] - AI[r, oi[:, 1]]).min()) if CI.shape[0] > 2 else np.inf,
                  float((AI[r, oi[:, 1]] - AI[r, oi[:, 0]]).min()), float((AO[r, oo[:, 1]] - AO[r, oo[:, 0]]).min()),
                  float((AO[r, oo[:, 2]] - AO[r, oo[:, 1]]).min()))
        return k, l, gap
    t = float(book.get('tilt', 0.0))
    if t == 0.0:
        k, gi = V.route_nearest(E['qI'], CI)
        l, go = V.route_nearest(E['qO'], CO)
        return k, l, float(min(gi.min(), go.min()))
    R = embedding(E, book)
    pi, rbar = R['pi'], float(book['rbar'])
    aI, bI = dep_ab(book['rI'], rbar)
    aO, bO = dep_ab(book['rO'], rbar)
    aP, bP = dep_ab(book['rP'], rbar)
    wl, wp = t * float(book['w_loc']), t * float(book['w_pair'])
    AI = V.xent_costs(E['qI'], CI) + wl * (pi[:, None] * aI[None] + bI[None])
    AO = V.xent_costs(E['qO'], CO) + wl * (pi[:, None] * aO[None] + bO[None])
    if float(book['kappa']) > 0:
        g = np.zeros_like(AO)
        for key, w in (('qO0', (1 - pi) / float(book['p0'])), ('qO1', pi / float(book['p1']))):
            Q = R[key]
            sq = np.zeros_like(AO)
            for c in range(Q.shape[1]):
                sq += (Q[:, c][:, None] - CO[:, c][None, :]) ** 2
            g += w[:, None] * sq
        AO = AO + t * float(book['kappa']) * g
    KI, KO = CI.shape[0], CO.shape[0]
    k, l = np.empty(len(pi), dtype=np.int64), np.empty(len(pi), dtype=np.int64)
    gap = np.inf
    for a in range(0, len(pi), chunk):
        sl = slice(a, a + chunk)
        cost = AI[sl, :, None] + AO[sl, None, :]
        if wp != 0.0:
            cost = cost + wp * (pi[sl, None, None] * aP[None] + bP[None])
        flat = cost.reshape(cost.shape[0], -1)
        f = flat.argmin(axis=1)
        best = flat[np.arange(len(f)), f]
        flat2 = flat.copy()
        flat2[np.arange(len(f)), f] = np.inf
        gap = min(gap, float((flat2.min(axis=1) - best).min()))
        k[sl], l[sl] = f // KO, f % KO
    return k, l, gap


def run(tag):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    from ohn import fam_a as FA
    F = C.features('MODEL_FIT')
    Z = FA.compact_po(np.asarray(F['X'], dtype=np.float64), tuple(map(str, F['feature_names'])))
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    for v in (0, 1):
        task = pickle.loads((V.SRC_FA_CACHE / f'V{v}' / 'TASK.pkl').read_bytes())
        sxp = V.FA_CACHE / f'V{v}' / 'SEXSIDE.pkl'
        sx_buf = sxp.read_bytes()
        sx_ok = V.bytes_sha(sx_buf) == (V.FA_CACHE / f'V{v}' / 'SEXSIDE.sha256').read_text().strip()
        ss = pickle.loads(sx_buf)
        E = {'qI': np.asarray(task['oof']['inc'], dtype=np.float64), 'qO': np.asarray(task['oof']['occ'], dtype=np.float64), 'Z': Z,
             'pi': np.minimum(np.maximum(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], 0.01), 0.99),
             'qO0': np.asarray(ss['oof']['occ_s'][0], dtype=np.float64), 'qO1': np.asarray(ss['oof']['occ_s'][1], dtype=np.float64)}
        for d in sorted((V.RUNS / 'FA').iterdir()):
            cfg = d.name
            if cfg in SKIP or not (d / f'V{v}' / 'MODEL.pkl').is_file():
                continue
            md = d / f'V{v}'
            buf = (md / 'MODEL.pkl').read_bytes()
            meta = json.loads((md / 'MODEL.json').read_text())
            relrec = json.loads((md / 'RELEASE.json').read_text())
            proc = []
            if V.bytes_sha(buf) != meta.get('model_sha256'):
                proc.append('MODEL.pkl differs from MODEL.json')
            if V.file_sha(md / 'RELEASE.npz') != relrec.get('release_sha256'):
                proc.append('RELEASE.npz differs from RELEASE.json')
            if not sx_ok:
                proc.append('SEX-side cache differs from its hash')
            M = pickle.loads(buf)
            book = M['book']
            k, l, gap = route_book(E, book)
            TI, nI, uI = V.b1_table(k, E['qI'], book['CI'])
            TO, nO, uO = V.b1_table(l, E['qO'], book['CO'])
            res = {'construction': M.get('construction'), 'selected': M.get('selected'), 'tilt': float(book.get('tilt', 0.0)),
                   'route': book.get('frc_route') or ('shuf' if book.get('frc_embed') == SHUF_MODE else book.get('restrict') or 'rule'),
                   'min_gap': gap, 'B1_vs_dec': {'I': {'bitwise_equal': bool(np.array_equal(TI, M['dec']['I'])),
                                                       'max_abs_diff': V.max_abs(TI, M['dec']['I'])},
                                                 'O': {'bitwise_equal': bool(np.array_equal(TO, M['dec']['O'])),
                                                       'max_abs_diff': V.max_abs(TO, M['dec']['O'])}},
                   'occupied_mine': {'income': int((nI > 0).sum()), 'occupation': int((nO > 0).sum()),
                                     'pair': int(len(np.unique(k * 100000 + l)))},
                   'occupied_recorded': relrec.get('occupied_cells_oof'), 'unoccupied_mine': {'I': uI, 'O': uO},
                   'unoccupied_recorded': {kk: (relrec.get('tables_info') or {}).get(f'unoccupied_prototype_fallback_{kk}')
                                           for kk in ('I', 'O')}}
            ok = (M.get('construction') == 'B1' and res['B1_vs_dec']['I']['bitwise_equal'] and res['B1_vs_dec']['O']['bitwise_equal']
                  and res['occupied_mine'] == res['occupied_recorded'] and res['unoccupied_mine'] == res['unoccupied_recorded'])
            roles = {}
            with np.load(md / 'RELEASE.npz', allow_pickle=False) as z:
                for role in V.ROLES6:
                    try:
                        ki = V.table_index(z[f'{role}__income'], M['dec']['I'])
                        li = V.table_index(z[f'{role}__occupation'], M['dec']['O'])
                        ids, nd = V.view_ids({'income': z[f'{role}__income'], 'occupation': z[f'{role}__occupation']})
                        rr = {'rows_of_tables': bool((ki >= 0).all() and (li >= 0).all()),
                              'distinct_equal_cells': nd['income'] == len(np.unique(ki)) and nd['occupation'] == len(np.unique(li))}
                    except ValueError as ex:
                        rr = {'rows_of_tables': False, 'error': str(ex)}
                    roles[role] = rr
                    ok &= all(x for x in rr.values() if isinstance(x, bool))
            res['roles'] = roles
            record['per'][f'{cfg}_V{v}'] = res
            checks.append({'id': f'V5B.TABLES.{cfg}_V{v}__{tag}',
                           'what': f'{cfg} V{v}: own routing of the out-of-fold statistics through the saved book ({res["route"]}), B1 '
                                   f'tables recomputed on that partition vs the stored tables (bitwise), occupancy vs the record, every '
                                   f'role\'s vectors are rows of the stored tables with distinct vectors = occupied cells',
                           'inputs': {'model': V.bytes_sha(buf), 'release': relrec.get('release_sha256'), 'sexside': V.bytes_sha(sx_buf)},
                           'result': res, 'numerical_status': 'PASS' if ok else 'FAIL',
                           'procedural_status': 'PASS' if not proc else 'FAIL', 'procedural_findings': proc})
    V.write_record(f'V5B_TABLES__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        r = c['result']
        print(c['id'], c['numerical_status'], c['procedural_status'], r['route'], r['tilt'], '%.2e' % r['min_gap'],
              r['B1_vs_dec']['I']['max_abs_diff'], r['B1_vs_dec']['O']['max_abs_diff'])
    return 0


def selftest():
    rng = np.random.default_rng(3)
    ok = {}
    n, KI, KO = 1200, 4, 8
    E = {'qI': V.clip_norm(rng.dirichlet(np.ones(2), n)), 'qO': V.clip_norm(rng.dirichlet(np.ones(5) * 0.6, n)),
         'pi': rng.uniform(0.05, 0.95, n), 'Z': rng.normal(size=(n, 6))}
    E['qO0'] = V.clip_norm(E['qO'] + rng.uniform(0, 0.1, (n, 5)))
    E['qO1'] = V.clip_norm(E['qO'] + rng.uniform(0, 0.1, (n, 5)))
    CI = V.clip_norm(E['qI'][rng.choice(n, KI, replace=False)])
    CO = V.clip_norm(E['qO'][rng.choice(n, KO, replace=False)])
    base = {'CI': CI, 'CO': CO, 'tilt': 0.0, 'lam': 0.0, 'restrict': None, 'w_pair': 1.0, 'w_loc': 1.0, 'kappa': 1.0}
    k0, l0, _ = route_book(E, base)
    ok['tilt0_is_nearest_codeword'] = bool(np.array_equal(k0, V.route_nearest(E['qI'], CI)[0]) and
                                           np.array_equal(l0, V.route_nearest(E['qO'], CO)[0]))
    prot = dict(base, tilt=2.0, lam=2.0, rbar=float(E['pi'].mean()), p0=float(1 - E['pi'].mean()), p1=float(E['pi'].mean()),
                rI=rng.uniform(0.2, 0.8, KI), rO=rng.uniform(0.2, 0.8, KO), rP=rng.uniform(0.2, 0.8, (KI, KO)))
    k1, l1, g1 = route_book(E, prot)
    ok['tilt_changes_partition'] = bool((k1 != k0).any() or (l1 != l0).any())
    # brute force joint argmin for a few people
    aI, bI = dep_ab(prot['rI'], prot['rbar'])
    aO, bO = dep_ab(prot['rO'], prot['rbar'])
    aP, bP = dep_ab(prot['rP'], prot['rbar'])
    good = True
    for i in range(40):
        best, arg = np.inf, None
        for kk in range(KI):
            for ll in range(KO):
                c = (-(E['qI'][i] * np.log(CI[kk])).sum() - (E['qO'][i] * np.log(CO[ll])).sum()
                     + 2.0 * (E['pi'][i] * aI[kk] + bI[kk]) + 2.0 * (E['pi'][i] * aO[ll] + bO[ll])
                     + 2.0 * ((1 - E['pi'][i]) / prot['p0'] * ((E['qO0'][i] - CO[ll]) ** 2).sum()
                              + E['pi'][i] / prot['p1'] * ((E['qO1'][i] - CO[ll]) ** 2).sum())
                     + 2.0 * (E['pi'][i] * aP[kk, ll] + bP[kk, ll]))
                if c < best - 1e-12:
                    best, arg = c, (kk, ll)
        good &= arg == (int(k1[i]), int(l1[i]))
    ok['joint_argmin_matches_brute_force'] = bool(good)
    # SHUF: person-level pi / qO0 / qO1 permutation does not change the routing (only the stored bank is read)
    sh = dict(prot, frc_embed=SHUF_MODE, shuf_bank_pi=E['pi'].copy(), shuf_tag='T')
    kS, lS, _ = route_book(E, sh)
    perm = rng.permutation(n)
    Ep = dict(E, pi=E['pi'][perm], qO0=E['qO0'][perm], qO1=E['qO1'][perm])
    kP, lP, _ = route_book(Ep, sh)
    ok['shuf_ignores_person_sex_statistics'] = bool(np.array_equal(kS, kP) and np.array_equal(lS, lP))
    # HQ: routing depends on pi only through the twin-cell tables
    hq = dict(prot, restrict='twin_cells', twin={'CI': CI, 'CO': CO}, hq_pi=rng.uniform(0.3, 0.7, KI * KO),
              hq_qO0=V.clip_norm(rng.dirichlet(np.ones(5), KI * KO)), hq_qO1=V.clip_norm(rng.dirichlet(np.ones(5), KI * KO)))
    kH, lH, _ = route_book(E, hq)
    kH2, lH2, _ = route_book(Ep, hq)
    ok['hq_ignores_person_sex_statistics'] = bool(np.array_equal(kH, kH2) and np.array_equal(lH, lH2))
    # UNT: share of moved people close to f; a moved person takes the second-nearest codeword
    un = dict(base, frc_route='unt', unt_f_O=0.3, unt_f_I=0.1, unt_tag='U')
    kU, lU, _ = route_book(E, un)
    moved_o = (lU != l0).mean()
    ok['unt_moves_about_f'] = bool(0.22 < moved_o < 0.38 and 0.05 < (kU != k0).mean() < 0.16)
    AO = V.xent_costs(E['qO'], CO)
    second = np.argsort(AO, axis=1, kind='stable')[:, 1]
    ok['unt_moved_take_second_nearest'] = bool(np.array_equal(lU[lU != l0], second[lU != l0]))
    # B1 recomputation detects a changed table
    T, _, _ = V.b1_table(l1, E['qO'], CO)
    T2 = T.copy()
    T2[0, 0] = np.nextafter(T2[0, 0], 0)
    ok['b1_change_detected'] = not np.array_equal(T, T2)
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {kk: bool(x) for kk, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())

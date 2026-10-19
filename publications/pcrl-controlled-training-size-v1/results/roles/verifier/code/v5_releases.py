"""Verifier stage V5c (label-free): the corrected releases recomputed with the verifier's own code, and the reader bank members
checked against the selected strengths.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Per version v (runs/CORRECT/V<v>):
  - W__<role> in RELEASES.npz == W100_v's evaluated release (runs/SIZE/A1__S100__V<v>/RELEASE.npz), bitwise, six roles;
  - own routing of every role (own FIT-only dictionary application of BASE/DICT.json, own oblivious-tree evaluation of
    ROUTING/TEACHER_occupation.json) and own correction sum Delta = sum_t theta[t, leaf_t] / T from THETA.npz, vs DELTA.npz;
  - own release r_s = softmax(centred log p_W + s Delta) for every arm (CONT, LOC, PAIR) and registered strength, vs RELEASES.npz;
  - if the selections and the reader bank spec exist: the bank's LOC_SEL / PAIR_SEL / CONT_AT_LOC / CONT_AT_PAIR (/ LOC_AT_PAIR)
    occupation arrays equal RELEASES.npz at exactly the selected strengths of SELECTION_V<v>.json, W equals W100_v, and every
    member's income equals the admitted H3 S<v> wire.

    python3 .../v5_releases.py --tag V1 [--bank-spec results/pcrl_controlled_training_size_v1/grid_specs/COND.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (label-free admitted features only)

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
CORRECT = PRIV / 'runs' / 'CORRECT'
SIZE = PRIV / 'runs' / 'SIZE'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ARMS = ('CONT', 'LOC', 'PAIR')


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def skey(s):
    return repr(float(s))


def my_release(pw, dn, s):
    if float(s) == 0.0:
        return np.array(pw, dtype=np.float64, copy=True)
    lp = np.log(np.maximum(np.asarray(pw, dtype=np.float64), 1e-300))
    lp = lp - lp.mean(axis=1, keepdims=True)
    return V.softmax(lp + float(s) * np.asarray(dn, dtype=np.float64))


def oblivious_leaves(payload, X):
    f = np.asarray(payload['split_features'], dtype=np.int64)
    b = np.asarray(payload['split_borders'], dtype=np.float32)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    T, depth = f.shape
    leaf = np.zeros((len(X32), T), dtype=np.int64)
    for i in range(depth):
        leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
    return leaf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--bank-spec', default=None)
    a = ap.parse_args()
    feats = {r: C.features(r) for r in ROLES6}
    names = [str(x) for x in feats['MODEL_FIT']['feature_names']]
    checks = []
    for v in (0, 1):
        root = CORRECT / f'V{v}'
        if not (root / 'RELEASES.json').is_file():
            continue
        rec = json.loads((root / 'RELEASES.json').read_text())
        if V.file_sha(root / 'RELEASES.npz') != rec['npz_sha256']:
            raise PermissionError('RELEASES.npz differs from its record')
        Z = npz(root / 'RELEASES.npz')
        W = npz(SIZE / f'A1__S100__V{v}' / 'RELEASE.npz')
        dic = json.loads((root / 'BASE' / 'DICT.json').read_text())
        payload = json.loads((root / 'ROUTING' / 'TEACHER_occupation.json').read_text())
        T = len(payload['split_features']); L = 2 ** len(payload['split_features'][0])
        w_ok = {r: bool(np.array_equal(Z[f'W__{r}'], W[r])) for r in ROLES6}
        leaves = {r: oblivious_leaves(payload, V.apply_dictionary(np.asarray(feats[r]['X']), names, dic['collapsed'])) for r in ROLES6}
        dmax, rmax, rbit, n_arr = {}, 0.0, True, 0
        for arm in ARMS:
            drec = json.loads((root / 'DIR' / arm / 'DIR.json').read_text())
            for nme, h in drec['files_sha256'].items():
                if V.file_sha(root / 'DIR' / arm / nme) != h:
                    raise PermissionError(f'DIR {arm} {nme} differs from its record')
            th = npz(root / 'DIR' / arm / 'THETA.npz')['theta'].reshape(T, L, 5)
            D = npz(root / 'DIR' / arm / 'DELTA.npz')
            dm = {}
            for r in ROLES6:
                mine = th[np.arange(T)[None, :], leaves[r]].sum(1) / T
                dm[r] = float(np.max(np.abs(mine - D[r])))
                for s in rec['strengths']:
                    p = my_release(W[r], D[r], s)
                    q = Z[f'{arm}__s{skey(s)}__{r}']
                    rmax = max(rmax, float(np.max(np.abs(p - q))))
                    rbit = rbit and bool(np.array_equal(p, q))
                    n_arr += 1
            dmax[arm] = max(dm.values())
        ok = all(w_ok.values()) and max(dmax.values()) <= 1e-12 and rmax <= 1e-12
        checks.append({'id': f'V5.RELEASES.V{v}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
                       'what': f'version {v}: corrected releases recomputed label-free with own code: W__<role> == W100_{v} release (bitwise); own '
                               'routing (own dictionary application + own oblivious-tree evaluation of the saved routing teacher) and own '
                               'correction sum from THETA.npz vs DELTA.npz; own r_s = softmax(centred log p_W + s Delta) for CONT / LOC / PAIR '
                               'at every registered strength on six roles vs RELEASES.npz (tolerance 1e-12)',
                       'inputs': V.inputs_sha([root / 'RELEASES.json', root / 'RELEASES.npz', SIZE / f'A1__S100__V{v}' / 'RELEASE.npz',
                                               root / 'ROUTING' / 'TEACHER_occupation.json', root / 'BASE' / 'DICT.json'] +
                                              [root / 'DIR' / arm / f for arm in ARMS for f in ('THETA.npz', 'DELTA.npz', 'DIR.json')]),
                       'expected': 'agreement', 'observed': {'W_bitwise': w_ok, 'delta_max_abs_diff_per_arm': dmax, 'release_max_abs_diff': rmax,
                                                             'releases_all_bitwise': rbit, 'n_arrays': n_arr, 'strengths': rec['strengths'],
                                                             'routing_T_L': [T, L]},
                       'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)})
    if a.bank_spec and Path(a.bank_spec).is_file():
        from ohn import devrun as OD
        spec_p = Path(a.bank_spec)
        spec = json.loads(spec_p.read_text())
        res, bad = {}, []
        for v in (0, 1):
            sel_p = V.RESULTS / 'conditional' / f'SELECTION_V{v}.json'
            sel = json.loads(sel_p.read_text())
            s_loc, s_pair = float(sel['selected']['LOC']), float(sel['selected']['PAIR'])
            want = {'LOC_SEL': ('LOC', s_loc), 'PAIR_SEL': ('PAIR', s_pair), 'CONT_AT_LOC': ('CONT', s_loc), 'CONT_AT_PAIR': ('CONT', s_pair),
                    'LOC_AT_PAIR': ('LOC', s_pair)}
            Z = npz(CORRECT / f'V{v}' / 'RELEASES.npz')
            W = npz(SIZE / f'A1__S100__V{v}' / 'RELEASE.npz')
            inc = np.asarray(OD.anchor_income('H3', v, 'MODEL_SELECT'), dtype=np.float64)
            res[v] = {'selected': {'LOC': s_loc, 'PAIR': s_pair}, 'selection_sha256_in_spec':
                      (spec.get('selection_provenance') or {}).get(str(v), {}).get('selection_sha256') == V.file_sha(sel_p)}
            if not res[v]['selection_sha256_in_spec']:
                bad.append(f'V{v} selection hash in the bank spec differs from SELECTION_V{v}.json')
            for name, m in spec['members'].items():
                if m.get('type') != 'arrays':
                    res[v][name] = {'type': m.get('type')}
                    continue
                per = m['seeds'][str(v)]
                ent = {'income_anchor': per['income'].get('frozen_anchor') if isinstance(per['income'], dict) else None}
                x = per['occupation']
                p = Path(x['path']); p = p if p.is_absolute() else ROOT / p
                ent['file_sha_ok'] = V.file_sha(p) == x.get('sha256')
                with np.load(p, allow_pickle=False) as z:
                    occ = z[x['key']].copy()
                if name in want:
                    arm, s = want[name]
                    exp = Z['W__MODEL_SELECT'] if s == 0.0 else Z[f'{arm}__s{skey(s)}__MODEL_SELECT']
                    ent['equals_selected_release'] = bool(np.array_equal(occ, exp))
                    ent['strength'] = s
                    if not ent['equals_selected_release']:
                        bad.append(f'V{v} {name}')
                elif name == 'W':
                    ent['equals_W100'] = bool(np.array_equal(occ, W['MODEL_SELECT']))
                    if not ent['equals_W100']:
                        bad.append(f'V{v} W')
                elif name in ('TR100', 'TT100'):
                    key = 'TASK_REFIT' if name == 'TR100' else 'T_TREE'
                    from ohn import common as OC
                    bank, _ = OC.frozen_bank(v, roles=('MODEL_SELECT',))
                    ent['equals_frozen_bank'] = bool(np.array_equal(occ, bank[key]['MODEL_SELECT']['occupation']))
                    if not ent['equals_frozen_bank']:
                        bad.append(f'V{v} {name}')
                if ent['income_anchor'] != 'H3' or not ent['file_sha_ok']:
                    bad.append(f'V{v} {name} income anchor / file hash')
                res[v][name] = ent
        checks.append({'id': f'V5.READER_BANK_MEMBERS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
                       'what': 'the reader bank members given to the development readers are exactly the selected releases: LOC_SEL / PAIR_SEL at '
                               's*_LOC / s*_PAIR of SELECTION_V<v>.json, CONT at the matched strengths, W = W100_v, income = admitted H3 S<v>',
                       'inputs': V.inputs_sha([spec_p] + [V.RESULTS / 'conditional' / f'SELECTION_V{v}.json' for v in (0, 1)]),
                       'expected': 'all equal', 'observed': {'per_version': res, 'problems': bad},
                       'status': 'PASS' if not bad else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)})
    import os
    V.write_record(f'V5_RELEASES__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: [c['status'], {k: c['observed'][k] for k in c['observed'] if k in ('release_max_abs_diff', 'releases_all_bitwise', 'delta_max_abs_diff_per_arm', 'problems')}] for c in checks}, default=str))


if __name__ == '__main__':
    main()

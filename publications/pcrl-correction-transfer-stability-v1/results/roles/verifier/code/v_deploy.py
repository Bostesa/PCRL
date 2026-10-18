"""Verifier check 7: restored deployment. One or more packages are replayed from PACKAGE.json and the stored wire with the verifier's
own code: dependency hashes; the raw-field encoder (from the package's encoder spec) against the admitted encoding; the occupation
release from raw fields (FIT-only dictionary of DICT.json -> compact_po -> the saved witness model -> own oblivious-tree routing -> own
correction sum -> softmax(centred log p_W + s Delta)) against the stored DEV_REPORT wire; the package's own entry points (api, csv,
underlying) against the verifier's pipeline; and at least one refusal at each entry point. Label-free: DEV_REPORT features and raw
inputs only (deployment parity); no label of any role is read.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 .../v_deploy.py --packages R0__REFIT_PAIR_AT_s1.0,R1__W --tag V1
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v_refit import my_compact_po, oblivious_raw  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402   (label-free admitted features and raw inputs)

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
DEPLOY = CAMP / 'deploy'
RUNS = CAMP / 'runs'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')


def resolve(key):
    if key.startswith('CTS::'):
        return RUNS / key[5:]
    if key.startswith('CODE::'):
        return ROOT / (key[6:].replace('.', '/') + '.py')
    if key.startswith('OHN::'):
        return ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs' / key[5:]
    if key.startswith('REPO::'):
        return ROOT / key[6:]
    return MIRROR / key


def my_encode(raw, enc):
    names = list(enc['feature_names']); col = {n: i for i, n in enumerate(names)}
    n = len(raw['AGEP'])
    X = np.zeros((n, len(names)), dtype=np.float64)
    for c in ('AGEP', 'WKHP'):
        v = np.asarray(raw[c], dtype=np.float64)
        X[:, col[c]] = (v - enc['numeric'][c]['mean']) / enc['numeric'][c]['sd']
    for c in BLOCKS:
        vals = [str(x).strip() for x in raw[c]]
        for i, v in enumerate(vals):
            key = f'{c}={v}'
            X[i, col[key] if key in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32), names


def apply_dict(X, names, collapsed):
    X = np.array(X, dtype=np.float64, copy=True)
    for b in BLOCKS:
        other = names.index(f'{b}=__OTHER__')
        for lvl in collapsed[b]:
            i = names.index(lvl)
            on = X[:, i] == 1
            X[on, i] = 0.0
            X[on, other] = 1.0
    return X


def my_occupation(P, name, X, names):
    d = RUNS / 'TRACKF' / P
    dic = json.loads((d / 'DICT.json').read_text())
    Xr = apply_dict(X, names, dic['collapsed'])
    with open(d / 'WITNESS' / 'MODEL.pkl', 'rb') as f:              # the saved deployed witness (campaign artifact, hash-checked)
        m = pickle.load(f)
    raw = np.asarray(m.decision_function(my_compact_po(Xr, names)), dtype=np.float64)
    z = raw - raw.max(1, keepdims=True); e = np.exp(z); p_w = e / e.sum(1, keepdims=True)
    if name == 'W':
        return p_w
    mm = re.match(r'^([A-Z_]+)@s([0-9.]+)$', name)
    did, s = mm.group(1), float(mm.group(2))
    if not did.startswith('REFIT'):
        raise SystemExit('only REFIT-routed packages are replayed here')
    payload = json.loads((d / 'ROUTING' / 'TEACHER_occupation.json').read_text())
    _, leaf = oblivious_raw(payload, Xr)
    with np.load(d / 'DIR' / did / 'TRAIN.npz') as zz:
        th = zz['theta_full'].astype(np.float64)
    T, L = leaf.shape[1], 2 ** len(payload['split_features'][0])
    th = th.reshape(T, L, 5)
    delta = th[np.arange(T)[None, :], leaf].sum(1) / T
    return V.release_full(p_w, delta, s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--packages', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    from cts import deploy as CD                                  # the package's own entry points (refusal and parity tests)
    R = C.features('DEV_REPORT', raw=True)
    raw = R['raw_inputs']
    out = []
    for pk in [x for x in a.packages.split(',') if x]:
        pdir = DEPLOY / pk
        meta = json.loads((pdir / 'PACKAGE.json').read_text())
        dep_bad = [k for k, h in meta['depends_sha256'].items() if not resolve(k).is_file() or V.file_sha(resolve(k)) != h]
        X_mine, names = my_encode(raw, meta['encoder'])
        enc_equal = bool(np.array_equal(X_mine.astype(np.float64), np.asarray(R['X'], dtype=np.float64))) and \
            list(R['feature_names']) == names
        P, name = meta['partition'], meta['name']
        with np.load(RUNS / 'STORED' / P / (name.replace('@', '_AT_') + '.npz')) as z:
            stored_occ = z['occ__DEV_REPORT'].copy()
        num = {'partition': P, 'name': name, 'dependencies': len(meta['depends_sha256']), 'dependency_hash_problems': dep_bad,
               'encoder_equals_admitted_encoding': enc_equal}
        if name != 'ANCHOR':
            occ = my_occupation(P, name, X_mine, names)
            num['own_pipeline_vs_stored_occupation_max_abs_diff'] = float(np.max(np.abs(occ - stored_occ)))
            num['own_pipeline_bitwise_equal_stored'] = bool(np.array_equal(occ, stored_occ))
        rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in FIELDS} for i in range(len(R['row_id']))]
        api = CD.release(str(pdir), rows)
        from ohn import devrun as OD
        inc_ref = np.asarray(OD.anchor_income('H3', meta['seed'], 'DEV_REPORT'), dtype=np.float64)
        num['api_exported_keys'] = {t: sorted(api[t]) for t in api}
        num['api_occupation_bitwise_equal_stored'] = bool(np.array_equal(api['occupation']['probability'], stored_occ))
        num['api_income_bitwise_equal_admitted_H3'] = bool(np.array_equal(api['income']['probability'], inc_ref))
        if name != 'ANCHOR':
            num['api_occupation_equal_own_pipeline_max_abs_diff'] = float(np.max(np.abs(api['occupation']['probability'] - occ)))
        ref = {}
        try:
            CD.release(str(pdir), [{**rows[0], 'SEX': 1}]); ref['api_SEX_field'] = 'NOT_REFUSED'
        except (PermissionError, ValueError) as e:
            ref['api_SEX_field'] = f'refused: {type(e).__name__}'
        try:
            CD.release(str(pdir), rows[:2], requested=('probability', 'leaf_ids')); ref['api_hidden_output'] = 'NOT_REFUSED'
        except (PermissionError, ValueError) as e:
            ref['api_hidden_output'] = f'refused: {type(e).__name__}'
        tmp = Path(tempfile.mkdtemp(dir=CAMP))
        try:
            q = tmp / 'bad.csv'
            with open(q, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(list(FIELDS) + ['SEX']); w.writerow([rows[0][c] for c in FIELDS] + [2])
            try:
                CD.release_csv(str(pdir), q); ref['csv_SEX_column'] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                ref['csv_SEX_column'] = f'refused: {type(e).__name__}'
            q2 = tmp / 'reordered.csv'
            with open(q2, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(list(FIELDS)[::-1]); w.writerow([rows[0][c] for c in list(FIELDS)[::-1]])
            try:
                CD.release_csv(str(pdir), q2); ref['csv_reordered_columns'] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                ref['csv_reordered_columns'] = f'refused: {type(e).__name__}'
        finally:
            for f in tmp.glob('*'):
                f.unlink()
            tmp.rmdir()
        X3 = np.asarray(R['X'][:3], dtype=np.float64)
        try:
            CD.underlying(str(pdir), np.column_stack((X3, np.ones(3))), names + ['SEX']); ref['underlying_SEX_column'] = 'NOT_REFUSED'
        except (PermissionError, ValueError) as e:
            ref['underlying_SEX_column'] = f'refused: {type(e).__name__}'
        try:
            CD.underlying(str(pdir), X3[:, ::-1], names[::-1]); ref['underlying_reordered_columns'] = 'NOT_REFUSED'
        except (PermissionError, ValueError) as e:
            ref['underlying_reordered_columns'] = f'refused: {type(e).__name__}'
        num['refusals'] = ref
        ok = (not dep_bad and enc_equal and num['api_occupation_bitwise_equal_stored'] and num['api_income_bitwise_equal_admitted_H3']
              and all(v.startswith('refused') for v in ref.values()) and
              all(set(v) <= {'probability', 'decision'} for v in num['api_exported_keys'].values()) and
              (name == 'ANCHOR' or (num['own_pipeline_bitwise_equal_stored'] or num['own_pipeline_vs_stored_occupation_max_abs_diff'] <= 1e-12)))
        chk = {'id': f'C7.DEPLOY.{pk}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
               'what': f'deployment package {pk}: dependency hashes; raw-field encoding; the occupation release from the six raw fields by '
                       'the verifier\'s own pipeline vs the stored DEV_REPORT wire; the package entry points vs the stored wires (income: '
                       'the admitted frozen H3 DEV_REPORT release); refusals at the api, csv and underlying entry points; exported keys',
               'method': 'own encoder from the package encoder spec, own dictionary application, own compact_po, the saved witness model, '
                         'own oblivious-tree routing and correction sum, own softmax; cts.deploy entry points called with SEX / '
                         'hidden-output / reordered inputs (must raise); label-free (DEV_REPORT features and raw inputs only)',
               'status': 'PASS' if ok else 'FAIL', 'numbers': num, 'code_sha256': V.code_hashes(__file__, V.__file__, HERE / 'v_refit.py')}
        out.append(chk)
    V.write_record(f'V_DEPLOY__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: [c['status'], {k: v for k, v in c['numbers'].items() if k != 'refusals'}, c['numbers']['refusals']]
                      for c in out}, indent=1, default=str)[:4000])


if __name__ == '__main__':
    main()

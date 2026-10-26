#!/usr/bin/env python3
"""RFA1 verifier, wave 3, task 3: independent re-run of one deployment package per version (RF_FULL_V0, RF1_FULL_V1) from raw inputs.

Per package (no label is read; admitted features / raw inputs only):
- the package's pinned dependencies re-hashed by the verifier (every depends_sha256 entry), MODEL.pkl also against MODEL.json;
- the verifier's own encoder (written from the package's encoder spec and acs_replication.schema: AGEP / WKHP z-scored in float64
  then float32; COW / SCHL / MAR / POBP one-hot '<field>=<value>' with the '__OTHER__' fallback) on 50 MODEL_FIT raw rows, compared
  with the admitted MODEL_FIT features bitwise;
- the verifier's own float64 forward pass of the pinned weights on (a) those 50 encoded rows and (b) the attacked MODEL_SELECT full
  batch (admitted features), compared with the evaluated RELEASE.npz rows (bitwise flag and max abs difference; tolerance 1e-15);
- the package entry points as black boxes: rfa.deploy.release (50 raw rows), release_csv, underlying (MODEL_SELECT full batch and
  request sizes 1 / 7 / 50 / 333): validity (shape, finite, non-negative, rows sum to 1 within 1e-12, decision = argmax, int64), the
  exported keys exactly {probability, decision}, equality to the verifier's outputs and to RELEASE.npz;
- refusals (own probe set): SEX, PINCP, OCCP, SERIALNO, SPORDER, ROW_ID, HOUSEHOLD, PUMA, an unknown extra field, reordered and
  missing fields, a non-finite AGEP, hidden outputs (logits, raw_score, embedding), draw keys on a deterministic package, the CSV
  path with SEX / PINCP / extra columns, the encoded path with SEX / extra / missing / reordered columns and NaN.

    python3 -m rfa.jobqueue run --unit RFA_VER_W3_DEPLOY_v1 --stage verify --rss-gib 2 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v3_deploy.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

OUT = HERE.parent / 'out' / 'V3_DEPLOY.json'
PRIV = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1'
RUNS = PRIV / 'runs'
PACKAGES = ('RF_FULL_V0', 'RF1_FULL_V1')
FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
NUMERIC, CATEG = ('AGEP', 'WKHP'), ('COW', 'SCHL', 'MAR', 'POBP')
TASKS = ('income', 'occupation')
TOL = 1e-15


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(key):
    if key.startswith('RFA::'):
        return RUNS / key[5:]
    if key.startswith('REPO::'):
        return ROOT / key[6:]
    if key.startswith('CODE::'):
        m = key[6:].replace('.', '/')
        p = ROOT / f'{m}.py'
        return p if p.is_file() else ROOT / m / '__init__.py'
    raise KeyError(key)


def my_encode(rows, enc):
    names = list(enc['feature_names'])
    col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, r in enumerate(rows):
        for c in NUMERIC:
            st = enc['numeric'][c]
            X[i, col[c]] = (float(r[c]) - float(st['mean'])) / float(st['sd'])
        for c in CATEG:
            k = f'{c}={str(r[c]).strip()}'
            X[i, col.get(k, col[f'{c}=__OTHER__'])] = 1.0
    return X.astype(np.float32)


def my_forward(layers, X):
    h = np.asarray(X, dtype=np.float64)
    for j, (w, b) in enumerate(layers):
        h = np.matmul(h, np.asarray(w, dtype=np.float64).T) + np.asarray(b, dtype=np.float64)
        if j < 2:
            h = np.where(h > 0.0, h, 0.0)
    z = h - h.mean(axis=1, keepdims=True)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def valid(out, n):
    ok = True
    for t, k in (('income', 2), ('occupation', 5)):
        p, d = out[t]['probability'], out[t]['decision']
        ok &= (p.shape == (n, k) and np.isfinite(p).all() and (p >= 0).all() and float(np.max(np.abs(p.sum(1) - 1))) <= 1e-12
               and d.dtype == np.int64 and np.array_equal(d, p.argmax(1)) and sorted(out[t]) == ['decision', 'probability'])
    return bool(ok and sorted(out) == list(TASKS))


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: run inside an rfa.jobqueue unit')
    t0 = time.process_time()
    from rfa import common as C
    from rfa import deploy as DP
    R = C.features('MODEL_FIT', raw=True)                      # label-free: features and raw inputs only
    raw = R['raw_inputs']
    rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in FIELDS} for i in range(50)]
    MS = C.features('MODEL_SELECT')
    res = {'schema': 'RFA1-verifier-deploy-v3', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'labels_read': False, 'packages': {},
           'environment': {'python': sys.version.split()[0], 'numpy': np.__version__, 'interpreter': sys.executable}}
    tmp = HERE.parent / 'out' / 'invented_deploy'           # CSV files hold INVENTED rows only (never real raw rows)
    tmp.mkdir(parents=True, exist_ok=True)
    for pk in PACKAGES:
        pkg = PRIV / 'deploy' / pk
        meta = json.loads((pkg / 'PACKAGE.json').read_text())
        name, v = meta['name'], int(meta['version'])
        fam, cfg = name.split('/')
        chk, num = {}, {}
        deps_bad = [k for k, h in meta['depends_sha256'].items() if sha256(resolve(k)) != h]
        chk['all_dependency_hashes_verified'] = not deps_bad
        num['n_dependencies'] = len(meta['depends_sha256'])
        mdir = RUNS / fam / cfg / f'V{v}'
        mj = json.loads((mdir / 'MODEL.json').read_text())
        buf = (mdir / 'MODEL.pkl').read_bytes()
        chk['model_pkl_matches_MODEL_json_and_package_pin'] = (hashlib.sha256(buf).hexdigest() == mj['model_sha256'] ==
                                                               meta['depends_sha256'][f'RFA::{fam}/{cfg}/V{v}/MODEL.pkl'])
        P = pickle.loads(buf)
        W = P['weights']
        rel = {}
        with np.load(mdir / 'RELEASE.npz', allow_pickle=False) as z:
            for role in ('MODEL_FIT', 'MODEL_SELECT'):
                for t in TASKS:
                    rel[(role, t)] = np.asarray(z[f'{role}__{t}'], dtype=np.float64).copy()
        chk['release_npz_is_the_evaluated_release'] = sha256(mdir / 'RELEASE.npz') == meta['release_sha256']
        # own encoder vs the admitted encoding (50 MODEL_FIT raw rows)
        Xme = my_encode(rows, meta['encoder'])
        chk['own_encoder_equals_admitted_features_bitwise'] = bool(np.array_equal(Xme, np.asarray(R['X'][:50], dtype=np.float32)))
        chk['encoder_column_order_equals_admitted'] = tuple(meta['encoder']['feature_names']) == tuple(R['feature_names'])
        # own forward pass vs the evaluated release
        own50 = {t: my_forward(W[t], Xme) for t in TASKS}
        ownMS = {t: my_forward(W[t], MS['X']) for t in TASKS}
        d50 = {t: float(np.max(np.abs(own50[t] - rel[('MODEL_FIT', t)][:50]))) for t in TASKS}
        dMS = {t: float(np.max(np.abs(ownMS[t] - rel[('MODEL_SELECT', t)]))) for t in TASKS}
        num['own_forward_max_abs_diff'] = {'MODEL_FIT_50': d50, 'MODEL_SELECT_full': dMS}
        num['own_forward_bitwise'] = {'MODEL_FIT_50': {t: bool(np.array_equal(own50[t], rel[('MODEL_FIT', t)][:50])) for t in TASKS},
                                      'MODEL_SELECT_full': {t: bool(np.array_equal(ownMS[t], rel[('MODEL_SELECT', t)])) for t in TASKS}}
        chk['own_forward_equals_evaluated_release (abs <= 1e-15)'] = all(x <= TOL for x in list(d50.values()) + list(dMS.values()))
        chk['own_forward_decisions_equal_release'] = all(np.array_equal(own50[t].argmax(1), rel[('MODEL_FIT', t)][:50].argmax(1)) and
                                                         np.array_equal(ownMS[t].argmax(1), rel[('MODEL_SELECT', t)].argmax(1)) for t in TASKS)
        # package entry points
        a = DP.release(pkg, rows)
        chk['release_output_valid_and_only_probability_decision'] = valid(a, 50)
        pa = {t: float(np.max(np.abs(a[t]['probability'] - rel[('MODEL_FIT', t)][:50]))) for t in TASKS}
        chk['package_release_equals_evaluated_release (abs <= 1e-15)'] = all(x <= TOL for x in pa.values())
        num['package_release_max_abs_diff_50'] = pa
        num['package_release_bitwise_50'] = {t: bool(np.array_equal(a[t]['probability'], rel[('MODEL_FIT', t)][:50])) for t in TASKS}
        enc = meta['encoder']
        inv = [{'AGEP': 25 + 7 * i, 'WKHP': 20 + 5 * i, 'COW': 1 + i % 3, 'SCHL': 16 + i % 6, 'MAR': 1 + i % 5, 'POBP': 12} for i in range(5)]
        q = tmp / f'{pk}_invented.csv'
        with open(q, 'w', newline='') as f:
            w_ = csv.DictWriter(f, fieldnames=list(FIELDS)); w_.writeheader(); w_.writerows(inv)
        b, b2 = DP.release_csv(pkg, q), DP.release(pkg, [{k: str(x) for k, x in r.items()} for r in inv])
        chk['release_csv_equals_release_api_bitwise (invented rows)'] = all(np.array_equal(b[t]['probability'], b2[t]['probability']) for t in TASKS)
        mine_inv = {t: my_forward(W[t], my_encode([{k: str(x) for k, x in r.items()} for r in inv], enc)) for t in TASKS}
        num['invented_rows_package_vs_own_max_abs_diff'] = {t: float(np.max(np.abs(b[t]['probability'] - mine_inv[t]))) for t in TASKS}
        chk['invented_rows_package_equals_own_encoder_and_forward (abs <= 1e-15)'] = all(x <= TOL for x in num['invented_rows_package_vs_own_max_abs_diff'].values())
        u = DP.underlying(pkg, MS['X'], MS['feature_names'])
        um = {t: float(np.max(np.abs(u[t] - rel[('MODEL_SELECT', t)]))) for t in TASKS}
        num['package_underlying_MODEL_SELECT_max_abs_diff'] = um
        num['package_underlying_MODEL_SELECT_bitwise'] = {t: bool(np.array_equal(u[t], rel[('MODEL_SELECT', t)])) for t in TASKS}
        chk['package_attacked_full_batch_equals_release (abs <= 1e-15)'] = all(x <= TOL for x in um.values())
        sizes = {}
        n = len(MS['row_id'])
        cap = {1: 50, 7: 700, 50: 2000, 333: n}                 # every call re-hashes the 20 pinned dependencies: bounded rows
        for size in (1, 7, 50, 333):
            parts = {t: [] for t in TASKS}
            for s0 in range(0, cap[size], size):
                r_ = DP.underlying(pkg, MS['X'][s0:s0 + size], MS['feature_names'])
                for t in TASKS:
                    parts[t].append(r_[t])
            cat = {t: np.vstack(parts[t]) for t in TASKS}
            m_ = min(len(cat['income']), cap[size])
            cat = {t: cat[t][:m_] for t in TASKS}
            sizes[str(size)] = {'rows': m_, 'max_abs_diff': {t: float(np.max(np.abs(cat[t] - rel[('MODEL_SELECT', t)][:m_]))) for t in TASKS},
                                'decisions_equal': all(np.array_equal(cat[t].argmax(1), rel[('MODEL_SELECT', t)][:m_].argmax(1)) for t in TASKS)}
        num['request_sizes'] = sizes
        chk['request_sizes_decisions_equal_and_abs_diff <= 1e-15'] = all(x['decisions_equal'] and max(x['max_abs_diff'].values()) <= TOL
                                                                          for x in sizes.values())
        # refusals (own probe set)
        r0 = rows[0]
        probes = {}

        def probe(label, fn):
            try:
                fn()
                probes[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                probes[label] = f'refused: {type(e).__name__}'
        for fld, val in (('SEX', 1), ('PINCP', 50000), ('OCCP', '0010'), ('SERIALNO', '2019HU0001'), ('SPORDER', 1), ('ROW_ID', 5),
                         ('HOUSEHOLD', 7), ('PUMA', 101), ('NOTE', 'x')):
            probe(f'release_with_{fld}', lambda fld=fld, val=val: DP.release(pkg, [{**r0, fld: val}]))
        probe('release_reordered', lambda: DP.release(pkg, [dict(reversed(list(r0.items())))]))
        probe('release_missing_POBP', lambda: DP.release(pkg, [{k: x for k, x in r0.items() if k != 'POBP'}]))
        probe('release_nonfinite_AGEP', lambda: DP.release(pkg, [{**r0, 'AGEP': float('nan')}]))
        for hid in ('logits', 'raw_score', 'embedding'):
            probe(f'release_hidden_output_{hid}', lambda hid=hid: DP.release(pkg, rows[:3], requested=('probability', hid)))
        probe('release_deterministic_with_draw_keys', lambda: DP.release(pkg, rows[:3], keys={'role': 'MODEL_FIT', 'row_id': np.arange(3)}))
        for fld in ('SEX', 'PINCP', 'EXTRA'):
            qq = tmp / f'{pk}_{fld}.csv'
            with open(qq, 'w', newline='') as f:
                cw = csv.writer(f); cw.writerow(list(FIELDS) + [fld]); cw.writerow([inv[0][c] for c in FIELDS] + [1])
            probe(f'csv_with_{fld}', lambda qq=qq: DP.release_csv(pkg, qq))
        names = list(MS['feature_names']); X3 = MS['X'][:3]
        probe('underlying_SEX_column', lambda: DP.underlying(pkg, np.column_stack((X3, np.ones(3))), names + ['SEX']))
        probe('underlying_extra_column', lambda: DP.underlying(pkg, np.column_stack((X3, np.ones(3))), names + ['EXTRA']))
        probe('underlying_missing_column', lambda: DP.underlying(pkg, X3[:, 1:], names[1:]))
        probe('underlying_reordered_columns', lambda: DP.underlying(pkg, X3[:, ::-1], names[::-1]))
        Xn = np.array(X3, dtype=np.float64); Xn[0, 0] = np.nan
        probe('underlying_nan', lambda: DP.underlying(pkg, Xn, names))
        chk['every_forbidden_input_or_output_refused'] = all(x.startswith('refused') for x in probes.values())
        num['refusals'] = probes
        res['packages'][pk] = {'name': name, 'version': v, 'checks': chk, 'numbers': num, 'all_pass': all(bool(x) for x in chk.values())}
    for f_ in tmp.glob('*.csv'):
        f_.unlink()
    res['all_pass'] = all(p['all_pass'] for p in res['packages'].values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {'v3_deploy.py': sha256(HERE / 'v3_deploy.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': {pk: [k for k, x in p['checks'].items() if not x] for pk, p in res['packages'].items()},
                      'cpu_s': res['cpu_s']}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

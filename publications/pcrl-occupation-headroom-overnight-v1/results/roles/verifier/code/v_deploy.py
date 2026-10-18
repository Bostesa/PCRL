"""VER check 8: raw-field deployment (label-free), optionally from a RESTORED copy of the campaign (backup alone).

    python3 v_deploy.py --root <campaign root (worktree or restored copy)> [--packages DIR ...] --out <record json>

Run with the root's own code (sys.path[0] = root): every package's dependency files re-hashed with own code against PACKAGE.json;
own raw-field encoder (z-score / one-hot with OTHER, from the package encoder spec) against the admitted DEV_REPORT encoding; the
documented wrapper `ohn.deploy.release` on the DEV_REPORT raw rows, the CSV entry point and the underlying deployment function on
both the admitted and the own encoding, each compared bitwise with the stored evaluated DEV_REPORT release; an own refusal battery
at both entry points (SEX and target / identifier fields or columns, extra, missing, reordered, duplicated fields, non-finite input,
hidden-output requests). Only released probability vectors and decisions are compared; no label is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import os

TRACE = {'on': False, 'opened': set()}


def _audit(event, args):
    # records every path opened while TRACE['on'] (runtime reads of the deployment calls); never blocks anything
    if TRACE['on'] and event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            TRACE['opened'].add(os.path.realpath(os.fsdecode(args[0])))
        except Exception:
            pass


sys.addaudithook(_audit)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def own_encode(rows, enc):
    names = list(enc['feature_names'])
    idx = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for c in enc['numeric']:
            st = enc['numeric'][c]
            X[i, idx[c]] = (float(r[c]) - st['mean']) / st['sd']
        for c in enc['categorical']:
            k = f"{c}={str(r[c]).strip()}"
            X[i, idx.get(k, idx[f'{c}=__OTHER__'])] = 1.0
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--packages', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--subdir', default=None, help='only packages under runs/<subdir>/*/PACKAGE.json')
    ap.add_argument('--forbid-prefix', nargs='*', default=[], help='runtime reads under these paths fail the check')
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root))
    from ohn import common as C
    from ohn import deploy as DP
    from ohn import devrun as D
    assert Path(C.__file__).resolve().is_relative_to(root), f'ohn imported from {C.__file__}, not from {root}'
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    raw = R['raw_inputs']
    permitted = list(DP.PERMITTED)
    rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in permitted} for i in range(len(R['row_id']))]
    if a.subdir:
        pkgs = sorted(p.parent for p in (C.RUNS / a.subdir).glob('*/PACKAGE.json'))
    else:
        pkgs = [Path(p) for p in a.packages] or sorted(p.parent for p in C.RUNS.glob('DEPLOY*/*/PACKAGE.json'))
    forbid = [os.path.realpath(x) + os.sep for x in a.forbid_prefix]
    tmp = Path(tempfile.mkdtemp(prefix='ver_deploy_'))
    csv_p = tmp / 'raw.csv'
    with open(csv_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=permitted)
        w.writeheader()
        w.writerows(rows)
    out = {'check': 'VER_DEPLOY', 'root': str(root), 'interpreter': sys.executable, 'numpy': np.__version__, 'packages': {},
           'subdir': a.subdir, 'forbid_prefix': a.forbid_prefix, 'ohn_imported_from': str(Path(C.__file__).resolve()),
           'labels_read': False}
    for pkg in pkgs:
        t0 = time.process_time()
        meta = json.loads((pkg / 'PACKAGE.json').read_text())
        dep_bad, manifest_paths = [], set()
        for rel, h in meta['depends_sha256'].items():
            if rel.startswith('OHN::'):
                p = C.RUNS / rel[5:]
            elif rel.startswith('CODE::'):
                p = C.ROOT / (rel[6:].replace('.', '/') + '.py')
            elif rel.startswith('REPO::'):
                p = C.ROOT / rel[6:]
            else:
                p = C.MIRROR / rel
            manifest_paths.add(os.path.realpath(p))
            if not p.is_file() or sha(p) != h:
                dep_bad.append(rel)
        enc = meta['encoder']
        X_own = own_encode(rows, enc)
        names = tuple(enc['feature_names'])
        enc_equal = bool(np.array_equal(X_own, R['X'])) and names == tuple(R['feature_names'])
        ref = DP.stored_dev(meta)
        TRACE['on'] = True
        TRACE['opened'] = set()
        api = DP.release(str(pkg), rows)
        via_csv = DP.release_csv(str(pkg), csv_p)
        und_adm = DP.underlying(str(pkg), R['X'], R['feature_names'])
        und_own = DP.underlying(str(pkg), X_own, names)
        TRACE['on'] = False
        root_real = os.path.realpath(C.ROOT)
        opened = {q for q in TRACE['opened'] if q.startswith(root_real + os.sep) and os.path.isfile(q)}
        pkg_json = os.path.realpath(pkg / 'PACKAGE.json')
        data_opened = sorted(q for q in opened if not q.endswith(('.py', '.pyc')) and q != pkg_json and not q.startswith(str(tmp)))
        code_opened = sorted(q for q in opened if q.endswith('.py'))
        unmanifested = [os.path.relpath(q, root_real) for q in data_opened if q not in manifest_paths]
        forbidden_reads = sorted(q for q in TRACE['opened'] if any(q.startswith(f) for f in forbid))
        code_unmanifested = [os.path.relpath(q, root_real) for q in code_opened if q not in manifest_paths]
        bit = {}
        for name, o in (('wrapper_api', {t: api[t]['probability'] for t in ('income', 'occupation')}),
                        ('wrapper_csv', {t: via_csv[t]['probability'] for t in ('income', 'occupation')}),
                        ('underlying_admitted_X', und_adm), ('underlying_own_encoding', und_own)):
            bit[name] = {t: bool(np.array_equal(np.asarray(o[t]), ref[t])) for t in ('income', 'occupation')}
        dec_ok = {t: bool(np.array_equal(api[t]['decision'], np.argmax(ref[t], axis=1))) and api[t]['decision'].dtype == np.int64
                  for t in ('income', 'occupation')}
        exported = {t: sorted(api[t]) for t in ('income', 'occupation')}
        r0 = rows[0]
        bad_rows = {
            'sex_lowercase_field': [{**r0, 'sex': 2}], 'SEX_field_first': [{'SEX': 1, **r0}], 'target_PINCP': [{**r0, 'PINCP': 60000}],
            'target_OCCP': [{**r0, 'OCCP': '4720'}], 'identifier_SERIALNO': [{**r0, 'SERIALNO': '2019HU0000001'}],
            'extra_ST': [{**r0, 'ST': 12}], 'missing_MAR': [{k: v for k, v in r0.items() if k != 'MAR'}],
            'reordered_AGEP_WKHP': [{'WKHP': r0['WKHP'], 'AGEP': r0['AGEP'], **{k: v for k, v in r0.items() if k not in ('AGEP', 'WKHP')}}],
            'empty_row': [{}]}
        refusals = {}
        for label, br in bad_rows.items():
            try:
                DP.release(str(pkg), br)
                refusals['wrapper_' + label] = 'NOT_REFUSED'
            except (PermissionError, ValueError, KeyError) as e:
                refusals['wrapper_' + label] = f'refused: {type(e).__name__}'
        for label, req in (('hidden_logits', ('probability', 'logits')), ('hidden_scores', ('decision', 'scores')), ('hidden_only', ('state',))):
            try:
                DP.release(str(pkg), rows[:2], requested=req)
                refusals['wrapper_' + label] = 'NOT_REFUSED'
            except PermissionError as e:
                refusals['wrapper_' + label] = f'refused: {type(e).__name__}'
        bad_csv = tmp / 'bad.csv'
        with open(bad_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=permitted + ['SEX'])
            w.writeheader()
            w.writerow({**r0, 'SEX': 1})
        try:
            DP.release_csv(str(pkg), bad_csv)
            refusals['csv_SEX_column'] = 'NOT_REFUSED'
        except (PermissionError, ValueError) as e:
            refusals['csv_SEX_column'] = f'refused: {type(e).__name__}'
        X3 = R['X'][:4]
        nl = list(names)
        for label, (Xb, nb) in (('sex_column', (np.column_stack((X3, np.zeros(4))), nl + ['SEX=2'])),
                                ('target_column', (np.column_stack((X3, np.zeros(4))), nl + ['PINCP'])),
                                ('extra_column', (np.column_stack((X3, np.zeros(4))), nl + ['ST=12'])),
                                ('missing_column', (X3[:, :-1], nl[:-1])),
                                ('swapped_columns', (X3[:, [1, 0] + list(range(2, X3.shape[1]))], [nl[1], nl[0]] + nl[2:])),
                                ('nonfinite_value', (np.where(np.arange(X3.shape[1]) == 0, np.nan, X3), nl)),
                                ('one_dimensional', (X3[0], nl))):
            try:
                DP.underlying(str(pkg), Xb, nb)
                refusals['underlying_' + label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals['underlying_' + label] = f'refused: {type(e).__name__}'
        out['packages'][pkg.name] = {
            'anchor': meta.get('anchor'), 'occupation': meta.get('occupation'), 'seed': meta.get('seed'),
            'n_dependencies': len(meta['depends_sha256']), 'dependency_mismatches': dep_bad, 'own_encoder_equals_admitted_X': enc_equal,
            'bitwise_equal_to_stored_DEV_REPORT_release': bit, 'decisions_argmax_int64': dec_ok, 'exported_keys': exported,
            'refusals': refusals, 'all_refused': all(v.startswith('refused') for v in refusals.values()),
            'runtime_data_files_opened': len(data_opened),
            'runtime_reads_under_forbidden_prefix': forbidden_reads,
            'runtime_data_files_not_in_manifest': unmanifested,
            'runtime_code_files_opened_not_in_manifest': code_unmanifested,
            'cpu_s': round(time.process_time() - t0, 2)}
        print(pkg.name, 'deps_ok', not dep_bad, 'enc', enc_equal, 'bitwise', all(all(v.values()) for v in bit.values()),
              'refused', out['packages'][pkg.name]['all_refused'], 'data_opened', len(data_opened),
              'unmanifested_data', unmanifested, 'unmanifested_code', code_unmanifested, 'forbidden_reads', len(forbidden_reads), flush=True)
    out['all_ok'] = all((not v['dependency_mismatches']) and v['own_encoder_equals_admitted_X'] and
                        not v['runtime_data_files_not_in_manifest'] and not v['runtime_reads_under_forbidden_prefix'] and
                        all(all(x.values()) for x in v['bitwise_equal_to_stored_DEV_REPORT_release'].values()) and
                        all(v['decisions_argmax_int64'].values()) and v['all_refused'] and
                        v['exported_keys'] == {'income': ['decision', 'probability'], 'occupation': ['decision', 'probability']}
                        for v in out['packages'].values()) if out['packages'] else None
    text = json.dumps(out, indent=1, sort_keys=True) + '\n'
    Path(a.out).write_text(text)
    hist = Path(a.out).parent / 'history'                 # write-once copy of every run (verifier commitment, 07:24Z)
    if hist.is_dir():
        import datetime as _dt
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with open(hist / f'{Path(a.out).stem}__{stamp}.json', 'x') as f:
            f.write(text)
    print(json.dumps({'all_ok': out['all_ok'], 'n_packages': len(out['packages'])}))


if __name__ == '__main__':
    main()

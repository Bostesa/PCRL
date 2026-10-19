"""Raw-field deployment of the PCRL controlled training-size campaign's packages (six permitted ACS fields -> each recipient's
probability vector and argmax decision; nothing else is exported). Adapted from cts.deploy / ohn.deploy (same wrapper, encoder and
refusal checks; no label is read anywhere in this module).

    python3 -m tsz.deploy store --name NAME --version V          label-free: the release of NAME on all six roles from admitted
                                                                 features (the 'stored wire' a package is checked against)
    python3 -m tsz.deploy build --name NAME --version V --out DIR
    python3 -m tsz.deploy check --packages DIR1 DIR2 ... --report OUT.json     bitwise parity on DEV_REPORT raw inputs + refusals

NAME (income is always the frozen H3 income S<v>):
  'A1:<fit id>'                  occupation = an A1_PO_16 size fit (tsz.sizefit.predict), e.g. A1:A1__S100__V0
  'CTRL:<fit id>'                occupation = the equally exposed TASK_REFIT control (tsz.control.predict_occupation)
  'CORR:<ARM>@s<x>'              occupation = W100_v corrected by arm ARM (CONT / LOC / PAIR) of version v at strength x
                                 (tsz.correct.predict_occupation)
Entry points, both enforcing the schema:
- `release(pkg, rows, requested)` / `release_csv(pkg, path, requested)`: every row must carry exactly AGEP, WKHP, COW, SCHL, MAR, POBP
  in that order; SEX, task targets (PINCP, OCCP, ...), identifiers and every other field are refused; only 'probability' and
  'decision' may be requested.
- `underlying(pkg, X, names)`: the deployment function on the encoded features; it refuses any column list other than the encoder's
  exact 145 names in order (missing, extra, reordered, target or SEX columns are rejected) and non-finite or mis-shaped input.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from ohn import deploy as OD
from . import common as C

PERMITTED = OD.PERMITTED
ALLOWED = OD.ALLOWED
RELEASE_CODE = ('tsz.common', 'tsz.deploy', 'tsz.sizefit', 'tsz.control', 'tsz.correct', 'cts.trackf', 'cts.common', 'ohn.common',
                'ohn.deploy', 'ohn.devrun', 'ohn.fam_a', 'ohn.protect', 'ihs.common', 'ihs.bases', 'ihs.fam_h', 'acs_replication.deploy',
                'acs_replication.schema', 'acs_replication.trees', 'scm_bench.common')


def store_dir():
    return C.RUNS / 'STORED'


def _fname(name, v):
    return name.replace(':', '__').replace('@', '_AT_') + f'__V{v}.npz'


def occupation_from_features(name, v, X, names):
    """The occupation release of NAME (version v) from admitted (encoded) features alone."""
    kind, rest = name.split(':', 1)
    if kind == 'A1':
        from . import sizefit
        return sizefit.predict(rest, X, names)
    if kind == 'CTRL':
        from . import control
        sid, vv = rest.rsplit('__V', 1)
        return control.predict(sid, int(vv), X, names)['occupation']
    if kind == 'CORR':
        from . import correct
        arm, s = rest.split('@s')
        return correct.predict(v, arm, float(s), X, names)
    raise ValueError(f'unknown package name {name}')


def _version_of(name, v):
    kind, rest = name.split(':', 1)
    if kind in ('A1', 'CTRL') and not rest.endswith(f'__V{v}'):
        raise ValueError(f'{name}: the fit id must belong to version {v}')
    return v


def store(name, v, check=None):
    """check: optional {role: array} of evaluated release arrays that the stored release must equal bitwise."""
    t0 = time.process_time()
    _version_of(name, v)
    out = store_dir() / _fname(name, v)
    arrs = {}
    for r in C.ROLES6:
        F = C.features(r)
        arrs[f'occ__{r}'] = occupation_from_features(name, v, F['X'], F['feature_names'])
    for r, ref in (check or {}).items():
        if not np.array_equal(np.asarray(ref), arrs[f'occ__{r}']):
            raise AssertionError(f'{r}: release from features differs from the evaluated release array')
    sha = C.save_npz_x(out, **arrs)
    print(json.dumps({'name': name, 'version': v, 'stored': str(out.relative_to(C.ROOT)), 'sha256': sha,
                      'cpu_s': round(time.process_time() - t0, 1)}))
    return out


def _deps(name, v):
    deps = {}
    kind, rest = name.split(':', 1)
    dirs = []
    if kind == 'A1':
        dirs.append(C.RUNS / 'SIZE' / rest)
    elif kind == 'CTRL':
        dirs.append(C.RUNS / 'CONTROL' / rest)
    elif kind == 'CORR':
        dirs += [C.RUNS / 'SIZE' / f'A1__S100__V{v}', C.RUNS / 'CORRECT' / f'V{v}', C.RUNS / 'CONTROL' / f'S100__V{v}']
    for d in dirs:
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix in ('.pkl', '.json', '.npz') and 'NESTED' not in p.parts:
                deps[f'TSZ::{p.relative_to(C.RUNS)}'] = C.sha(p)
    st = store_dir() / _fname(name, v)
    deps[f'TSZ::{st.relative_to(C.RUNS)}'] = C.sha(st)
    return deps


def build(name, v, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    meta = OD.build('H3', 'SOURCE:JOINT', v, out / '_anchor')          # the source anchor package (H3 income, encoder, its deps)
    shutil.rmtree(out / '_anchor')
    deps = dict(meta['depends_sha256'])
    deps.update(_deps(name, v))
    deps.update({f'CODE::{m}': C.sha(C.ROOT / (m.replace('.', '/') + '.py')) for m in RELEASE_CODE})
    rec = {'schema': 'TSZ1-deploy-package-v1', 'version': v, 'seed': v, 'name': name, 'income': f'H3_STRONG_LJDIR_PB S{v} (frozen anchor)',
           'encoder': meta['encoder'], 'depends_sha256': deps, 'release_contract': 'probability and decision per recipient; nothing else',
           'permitted_inputs': PERMITTED, 'environment_at_build': OD._environment(), 'utc': C.now()}
    (out / 'PACKAGE.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    return rec


def _resolve_dep(key):
    if key.startswith('TSZ::'):
        return C.RUNS / key[5:]
    if key.startswith('CODE::'):
        return C.ROOT / (key[6:].replace('.', '/') + '.py')
    if key.startswith('OHN::'):
        from ohn import common as OC
        return OC.RUNS / key[5:]
    if key.startswith('REPO::'):
        return C.ROOT / key[6:]
    from ohn import common as OC
    return OC.MIRROR / key


def _load(pkg):
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for k, h in meta['depends_sha256'].items():
        if C.sha(_resolve_dep(k)) != h:
            raise PermissionError(f'package dependency hash differs: {k}')
    return meta


def underlying(pkg, X, names):
    meta = _load(pkg)
    OD.check_encoded(meta, X, names)
    from ihs import fam_h
    src = fam_h.predict(meta['seed'], 'H3_STRONG_LJDIR_PB', X, names)
    occ = occupation_from_features(meta['name'], meta['version'], X, names)
    return {'income': np.asarray(src['income'], dtype=np.float64), 'occupation': np.asarray(occ, dtype=np.float64)}


def release(pkg, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    OD.check_rows(rows)
    meta = _load(pkg)
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = underlying(pkg, X, names)
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested}
            for t in C.TASKS}


def release_csv(pkg, path, requested=('probability', 'decision')):
    with open(path, newline='') as f:
        header = csv.DictReader(f).fieldnames or []
    bad = [h for h in header if h.upper() in OD.FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present (never accepted as inputs): {bad}')
    return release(pkg, SRC_DEPLOY.read_raw_csv(path), requested)


def stored(meta):
    from ohn import devrun as ODR
    inc = ODR.anchor_income('H3', meta['seed'], 'DEV_REPORT')
    with np.load(store_dir() / _fname(meta['name'], meta['version'])) as z:
        occ = z['occ__DEV_REPORT'].copy()
    return {'income': inc, 'occupation': occ}


def check(packages, report):
    R = C.features('DEV_REPORT', raw=True)                       # label-free: DEV_REPORT features and raw inputs only
    raw = R['raw_inputs']
    rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in PERMITTED} for i in range(len(R['row_id']))]
    tmp = Path(tempfile.mkdtemp(dir=C.PRIV)); p = tmp / 'raw.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PERMITTED); w.writeheader(); w.writerows(rows)
    res = {}
    for pkg in packages:
        t0 = time.process_time()
        meta = _load(pkg)
        ref = stored(meta)
        a = release(pkg, rows); b = release_csv(pkg, p); u = underlying(pkg, R['X'], R['feature_names'])
        same = {nm: {t: bool(np.array_equal(o[t]['probability'], ref[t])) for t in C.TASKS}
                for nm, o in (('wrapper_api', a), ('wrapper_csv', b), ('underlying', {t: {'probability': u[t]} for t in C.TASKS}))}
        refusals = {}
        for label, bad in (('wrapper_SEX_field', [{**rows[0], 'SEX': 1}]), ('wrapper_target_OCCP_field', [{**rows[0], 'OCCP': '0010'}]),
                           ('wrapper_target_PINCP_field', [{**rows[0], 'PINCP': 50000}]), ('wrapper_extra_field', [{**rows[0], 'PUMA': 1}]),
                           ('wrapper_reordered_fields', [dict(reversed(list(rows[0].items())))]),
                           ('wrapper_missing_field', [{k: v for k, v in rows[0].items() if k != 'POBP'}])):
            try:
                release(pkg, bad); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        for label, hdr in (('csv_SEX_column', PERMITTED + ['SEX']), ('csv_target_PINCP_column', PERMITTED + ['PINCP']),
                           ('csv_target_OCCP_column', PERMITTED + ['OCCP']), ('csv_extra_column', PERMITTED + ['PUMA']),
                           ('csv_reordered_columns', PERMITTED[::-1]), ('csv_missing_column', PERMITTED[:-1])):
            q = tmp / f'{label}.csv'
            with open(q, 'w', newline='') as f:
                cw = csv.writer(f); cw.writerow(hdr); cw.writerow([rows[0].get(h, 1) for h in hdr])
            try:
                release_csv(pkg, q); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        names = list(R['feature_names']); X3 = R['X'][:3]
        for label, (Xb, nb) in (('underlying_SEX_column', (np.column_stack((X3, np.ones(3))), names + ['SEX'])),
                                ('underlying_target_column', (np.column_stack((X3, np.ones(3))), names + ['OCCP'])),
                                ('underlying_extra_column', (np.column_stack((X3, np.ones(3))), names + ['EXTRA'])),
                                ('underlying_missing_column', (X3[:, 1:], names[1:])),
                                ('underlying_reordered_columns', (X3[:, ::-1], names[::-1]))):
            try:
                underlying(pkg, Xb, nb); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        try:
            release(pkg, rows[:3], requested=('probability', 'raw_score')); refusals['hidden_output_request'] = 'NOT_REFUSED'
        except PermissionError:
            refusals['hidden_output_request'] = 'refused: PermissionError'
        res[Path(pkg).name] = {'name': meta['name'], 'version': meta['version'],
                               'bitwise_equal_to_stored_DEV_REPORT_wire': same,
                               'decisions_equal': {t: bool(np.array_equal(a[t]['decision'], ref[t].argmax(1))) for t in C.TASKS},
                               'refusals': refusals, 'exported_keys': {t: sorted(a[t]) for t in C.TASKS},
                               'cpu_s': round(time.process_time() - t0, 2), 'n_dependencies': len(meta['depends_sha256'])}
    shutil.rmtree(tmp, ignore_errors=True)
    rec = {'schema': 'TSZ1-deployment-check-v1', 'utc': C.now(), 'packages': res, 'labels_read': False, 'environment': OD._environment()}
    rec['all_bitwise'] = all(all(all(v.values()) for v in r['bitwise_equal_to_stored_DEV_REPORT_wire'].values()) for r in res.values())
    rec['all_refused'] = all(v.startswith('refused') for r in res.values() for v in r['refusals'].values())
    C.write_json_x(report, rec)
    print(json.dumps({'all_bitwise': rec['all_bitwise'], 'all_refused': rec['all_refused']}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('store'); s.add_argument('--name', required=True); s.add_argument('--version', type=int, required=True)
    b = sub.add_parser('build'); b.add_argument('--name', required=True); b.add_argument('--version', type=int, required=True)
    b.add_argument('--out', required=True)
    c = sub.add_parser('check'); c.add_argument('--packages', nargs='+', required=True); c.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'store':
        kind, rest = a.name.split(':', 1)
        check = None
        if kind == 'A1':                       # the stored wire must equal the fit's saved (evaluated) release on every role
            from . import sizefit
            rel, _ = sizefit.release(rest)
            check = {r: rel[r] for r in C.ROLES6}
        elif kind == 'CTRL':
            from . import evaluate as EV
            sid, vv = rest.rsplit('__V', 1)
            check = {r: EV.control_release(sid, int(vv), 'TR', r)[0] for r in C.ROLES6}
        elif kind == 'CORR':
            from . import correct
            arm, s = rest.split('@s')
            check = {r: correct.load_release(a.version, arm, float(s), r) for r in C.ROLES6}
        store(a.name, a.version, check=check)
    elif a.cmd == 'build':
        print(json.dumps(sorted(build(a.name, a.version, a.out)['depends_sha256'])[:5]))
    else:
        rec = check(a.packages, a.report)
        if not (rec['all_bitwise'] and rec['all_refused']):
            raise SystemExit(1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

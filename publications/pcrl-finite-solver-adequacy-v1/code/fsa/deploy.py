"""Raw-field deployment of the PCRL finite-output solver-adequacy campaign's packages (six permitted ACS fields -> each recipient's
probability vector and argmax decision; nothing else is exported). Adapted from tsz.deploy / ohn.deploy (same encoder, row and column
checks and refusal probes; derived from nrm.deploy; no label is read anywhere in this module).

    python3 -m fsa.deploy store --name FAMILY/CONFIG --version V      label-free: the release recomputed from admitted features on all six
                                                                     roles; must equal the saved, evaluated RELEASE.npz bitwise
    python3 -m fsa.deploy build --name FAMILY/CONFIG --version V --out DIR
    python3 -m fsa.deploy check --packages DIR1 DIR2 ... --report OUT.json   bitwise parity on DEV_REPORT raw inputs + refusals

Family interface (each family package fsa.<family lower case> provides it in its `release` module or package root):
    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
        deterministic given its inputs; reads only its own model files. A RANDOMIZED release draws instance `draw` from per-request
        keys {'role': <stream name>, 'row_id': (n,) request identifiers}, as its card states (keyed one-draw-per-person emission);
        the keys are supplied out-of-band by the service operator, never as a Census field, and are refused by deterministic
        packages;
    model_paths(config_id, version) -> list of paths the prediction reads (hash-pinned in the package).
Entry points, both enforcing the schema:
- `release(pkg, rows, requested)` / `release_csv(pkg, path, requested)`: every row must carry exactly AGEP, WKHP, COW, SCHL, MAR, POBP
  in that order; SEX, task targets, identifiers and every other field are refused; only 'probability' and 'decision' may be requested.
- `underlying(pkg, X, names)`: the deployment function on the encoded features; it refuses any column list other than the encoder's
  exact 145 names in order and non-finite or mis-shaped input.
"""
from __future__ import annotations
import argparse
import csv
import importlib
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
from . import release as RL

PERMITTED = OD.PERMITTED
ALLOWED = OD.ALLOWED
CODE_BASE = ('fsa.common', 'fsa.deploy', 'fsa.release', 'ohn.common', 'ohn.deploy', 'acs_replication.deploy', 'acs_replication.schema',
             'scm_bench.common')


def family_module(family):
    mod = importlib.import_module(f'fsa.{family.lower()}')
    if not hasattr(mod, 'predict'):
        mod = importlib.import_module(f'fsa.{family.lower()}.release')
    return mod


def _split(name):
    family, config = name.split('/')
    return family, config


def store_dir():
    return C.RUNS / 'STORED'


def _fname(name, v):
    return name.replace('/', '__') + f'__V{v}.npz'


def from_features(name, v, X, names, draw=0, keys=None):
    family, config = _split(name)
    mod = family_module(family)
    out = mod.predict(config, v, X, names, draw=draw, keys=keys) if keys is not None else mod.predict(config, v, X, names, draw=draw)
    return {t: np.ascontiguousarray(np.asarray(out[t], dtype=np.float64)) for t in C.TASKS}


def _keys_for(rec, role, row_id):
    """Per-request draw keys of the evaluated wire (randomized releases only): the role stream and the admitted row ids."""
    return {'role': role, 'row_id': np.asarray(row_id)} if rec.get('randomized') else None


def store(name, v):
    """Recompute the release from admitted features on all six roles (draw 0 for a randomized release) and require bitwise equality
    with the saved, evaluated RELEASE.npz."""
    t0 = time.process_time()
    family, config = _split(name)
    saved, rec = RL.load(family, config, v)
    arrs, eq = {}, {}
    for r in C.ROLES6:
        F = C.features(r)
        rel = from_features(name, v, F['X'], F['feature_names'], draw=0, keys=_keys_for(rec, r, F['row_id']))
        for t in C.TASKS:
            ref = saved[r][t][0] if saved[r][t].ndim == 3 else saved[r][t]
            eq[f'{r}|{t}'] = bool(np.array_equal(ref, rel[t]))
            arrs[f'{t}__{r}'] = rel[t]
    if not all(eq.values()):
        raise AssertionError(f'release from features differs from the evaluated release: {[k for k, v in eq.items() if not v]}')
    sha = C.save_npz_x(store_dir() / _fname(name, v), **arrs)
    res = {'name': name, 'version': v, 'stored_sha256': sha, 'bitwise_equal_to_evaluated_release': eq,
           'randomized': bool(rec.get('randomized')), 'draw': 0, 'cpu_s': round(time.process_time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != 'bitwise_equal_to_evaluated_release'}))
    return res


def build(name, v, out):
    family, config = _split(name)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    enc = json.loads(OD.C.admitted(f'admitted/packages/T_TREE_S{v}/PACKAGE.json').read_text())['encoder']
    mod = family_module(family)
    deps = {}
    for p in mod.model_paths(config, v):              # campaign runs keyed FRC::; other repository files (frozen bases) keyed REPO::
        rp = Path(p).resolve()
        if rp.is_relative_to(C.RUNS.resolve()):
            deps[f'FRC::{rp.relative_to(C.RUNS.resolve())}'] = C.sha(rp)
        elif rp.is_relative_to(C.ROOT.resolve()):
            deps[f'REPO::{rp.relative_to(C.ROOT.resolve())}'] = C.sha(rp)
        else:
            raise PermissionError(f'{p}: package dependencies must lie inside the repository')
    st = store_dir() / _fname(name, v)
    deps[f'FRC::{st.relative_to(C.RUNS)}'] = C.sha(st)
    code = set(CODE_BASE) | {mod.__name__} | {m for m in getattr(mod, 'CODE_MODULES', ())}
    deps.update({f'CODE::{m}': C.sha(_code_path(m)) for m in sorted(code)})
    _, rec = RL.load(family, config, v, roles=('DEV_REPORT',))
    meta = {'schema': 'FSA1-deploy-package-v1', 'name': name, 'version': v, 'encoder': enc, 'depends_sha256': deps,
            'release_sha256': rec['release_sha256'], 'randomized': bool(rec.get('randomized')),
            'release_contract': 'probability and decision per recipient; nothing else', 'permitted_inputs': PERMITTED,
            'environment_at_build': OD._environment(), 'utc': C.now()}
    (out / 'PACKAGE.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return meta


def _code_path(module):
    p = C.ROOT / (module.replace('.', '/') + '.py')
    return p if p.is_file() else C.ROOT / module.replace('.', '/') / '__init__.py'


def _resolve_dep(key):
    if key.startswith('FRC::'):
        return C.RUNS / key[5:]
    if key.startswith('REPO::'):
        return C.ROOT / key[6:]
    if key.startswith('CODE::'):
        return _code_path(key[6:])
    raise KeyError(key)


def _load(pkg):
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for k, h in meta['depends_sha256'].items():
        if C.sha(_resolve_dep(k)) != h:
            raise PermissionError(f'package dependency hash differs: {k}')
    return meta


def _check_keys(meta, keys, n):
    if meta.get('randomized'):
        if keys is None or 'row_id' not in keys or len(keys['row_id']) != n:
            raise PermissionError('a randomized package needs out-of-band per-request draw keys (one per row)')
    elif keys is not None:
        raise PermissionError('draw keys are refused by a deterministic package')


def underlying(pkg, X, names, draw=0, keys=None):
    meta = _load(pkg)
    OD.check_encoded(meta, X, names)
    _check_keys(meta, keys, len(X))
    return from_features(meta['name'], meta['version'], X, names, draw=draw, keys=keys)


def release(pkg, rows, requested=('probability', 'decision'), keys=None):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    OD.check_rows(rows)
    meta = _load(pkg)
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = underlying(pkg, X, names, keys=keys)
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested}
            for t in C.TASKS}


def release_csv(pkg, path, requested=('probability', 'decision'), keys=None):
    with open(path, newline='') as f:
        header = csv.DictReader(f).fieldnames or []
    bad = [h for h in header if h.upper() in OD.FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present (never accepted as inputs): {bad}')
    return release(pkg, SRC_DEPLOY.read_raw_csv(path), requested, keys=keys)


def stored(meta):
    with np.load(store_dir() / _fname(meta['name'], meta['version'])) as z:
        return {t: z[f'{t}__DEV_REPORT'].copy() for t in C.TASKS}


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
        keys = {'role': 'DEV_REPORT', 'row_id': np.asarray(R['row_id'])} if meta.get('randomized') else None
        a = release(pkg, rows, keys=keys); b = release_csv(pkg, p, keys=keys); u = underlying(pkg, R['X'], R['feature_names'], keys=keys)
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
        try:                                       # randomized: keys are required; deterministic: keys are refused
            if meta.get('randomized'):
                release(pkg, rows[:3]); refusals['randomized_without_draw_keys'] = 'NOT_REFUSED'
            else:
                release(pkg, rows[:3], keys={'role': 'DEV_REPORT', 'row_id': np.arange(3)})
                refusals['deterministic_with_draw_keys'] = 'NOT_REFUSED'
        except PermissionError:
            refusals['randomized_without_draw_keys' if meta.get('randomized') else 'deterministic_with_draw_keys'] = 'refused: PermissionError'
        res[Path(pkg).name] = {'name': meta['name'], 'version': meta['version'], 'randomized': meta.get('randomized'),
                               'bitwise_equal_to_stored_DEV_REPORT_wire': same,
                               'decisions_equal': {t: bool(np.array_equal(a[t]['decision'], ref[t].argmax(1))) for t in C.TASKS},
                               'refusals': refusals, 'exported_keys': {t: sorted(a[t]) for t in C.TASKS},
                               'cpu_s': round(time.process_time() - t0, 2), 'n_dependencies': len(meta['depends_sha256'])}
    shutil.rmtree(tmp, ignore_errors=True)
    rec = {'schema': 'FSA1-deployment-check-v1', 'utc': C.now(), 'packages': res, 'labels_read': False, 'environment': OD._environment()}
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
        store(a.name, a.version)
    elif a.cmd == 'build':
        print(json.dumps(sorted(build(a.name, a.version, a.out)['depends_sha256'])[:5]))
    else:
        rec = check(a.packages, a.report)
        if not (rec['all_bitwise'] and rec['all_refused']):
            raise SystemExit(1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

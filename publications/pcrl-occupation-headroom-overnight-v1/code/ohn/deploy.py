"""Raw-field deployment of complete releases of the overnight occupation-headroom campaign (six permitted ACS fields -> each
recipient's probability vector and argmax decision; nothing else is exported).

    python -m ohn.deploy build --anchor H3 --occ FAMILY:CONFIG --seed s --out DIR    # FAMILY:CONFIG or SOURCE:JOINT
    python -m ohn.deploy check --packages DIR1 DIR2 ... --report OUT.json            # bitwise parity + refusals (no label read)

A package (private) holds PACKAGE.json: the income anchor (H3 = the source's H3_STRONG_LJDIR_PB income, TW = HC_STRONG income,
both computed by the source's unmodified `ihs.fam_h.predict` from the admitted mirror), the occupation source (a family module's
`predict`, or the inherited JOINT_TREE output for the unmodified anchor package), the frozen 145-column encoder of the replication,
and the SHA-256 of every file the release depends on (anchor state, step-1 learner, teacher and correction files, the occupation
run directory, and the family module source). Entry points, both enforcing the schema:
- `release(pkg, rows, requested)` / `release_csv(pkg, path, requested)`: the documented wrapper. Every row must carry exactly AGEP,
  WKHP, COW, SCHL, MAR, POBP in that order; SEX, task targets (PINCP, OCCP, ...), identifiers and every other field are refused;
  only 'probability' and 'decision' may be requested.
- `underlying(pkg, X, names)`: the deployment function on the encoded features. It refuses any column list other than the
  package encoder's exact 145 names in order (so a missing, extra, reordered, target or SEX column is rejected here too), and
  non-finite or mis-shaped input. Intermediate quantities are never returned.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import schema
from . import common as C
from . import devrun as D

PERMITTED = list(SRC_DEPLOY.PERMITTED)
ALLOWED = {'probability', 'decision'}
FORBIDDEN = set(schema.FORBIDDEN) | {'INCOME', 'OCCUPATION', 'Y', 'LABEL', 'TARGET', 'UNIT', 'ROW_ID', 'HOUSEHOLD'}
FAMILY_MODULE = {'A': 'ohn.fam_a', 'B': 'ohn.fam_b', 'C': 'ohn.fam_c', 'PROT': 'ohn.protect', 'INC': 'ohn.fam_inc'}
RELEASE_CODE = ('ohn.common', 'ohn.deploy', 'ohn.devrun', 'ihs.common', 'ihs.bases', 'ihs.fam_h', 'acs_replication.deploy',
                'acs_replication.schema', 'acs_replication.trees', 'scm_bench.common')       # recorded code closure, reported by check
GOVERNANCE = ('results/pcrl_income_headroom_successor_v1/SOURCE_ADMISSION.json',               # read at runtime by ihs admitted()
              'results/pcrl_occupation_headroom_overnight_v1/SOURCE_ADMISSION.json')           # read at runtime by ohn admitted()


def _run_deps(rd):
    return {f'OHN::{p.relative_to(C.RUNS)}': C.sha(p) for p in sorted(Path(rd).rglob('*'))
            if p.is_file() and not p.name.startswith(('EVAL_', 'VERIFY'))}            # mutable records excluded


def _environment():
    out = {'interpreter': sys.executable, 'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform(),
           'machine': platform.machine()}
    for m in ('sklearn', 'scipy'):
        try:
            out[m] = importlib.import_module(m).__version__
        except ImportError:
            out[m] = None
    return out


def _records():
    """Shared by every package: governance files read at runtime (hash-checked), build environment, code closure (recorded)."""
    return ({f'REPO::{g}': C.sha(C.ROOT / g) for g in GOVERNANCE},
            {'environment_at_build': _environment(),
             'code_closure_sha256': {m: C.sha(C.ROOT / (m.replace('.', '/') + '.py')) for m in RELEASE_CODE}})


def _code_dep(module):
    return {f'CODE::{module}': C.sha(C.ROOT / (module.replace('.', '/') + '.py'))}


def _upstream_deps(fam, cfg, seed):
    """Files and modules a family's predict reads beyond its own run directory (each also hash-checked inside predict):
    a protection configuration's witness run and module; an income-branch release's repaired Step-1 learner, its occupation witness,
    and family A (imported by ohn.fam_inc)."""
    deps = {}
    if fam == 'B':                                   # the base learners STATE.json points to (outside the run directory)
        st = json.loads((D.occ_dir(fam, cfg, seed) / 'STATE.json').read_text())
        for lid in st['base_model_sha256']:
            for f in ('BASE.json', 'MODEL.pkl'):
                p = C.RUNS / 'B' / 'BASES' / lid / f'S{seed}' / f
                deps[f'OHN::{p.relative_to(C.RUNS)}'] = C.sha(p)
        frozen = st['params'].get('frozen')            # B3: frozen checkpoint read by fam_b.frozen_logits (verifier finding)
        if frozen and frozen != 'T_TREE':
            rel = f'admitted/stage_b/S{seed}/{frozen}__E20.npz'
            deps[rel] = C.sha(C.MIRROR / rel)
    elif fam == 'PROT':
        P = json.loads((D.occ_dir(fam, cfg, seed) / 'PARAMS.json').read_text())
        wf, wc = P['witness']['family'], P['witness']['config']
        deps.update(_run_deps(D.occ_dir(wf, wc, seed))); deps.update(_code_dep(f'ohn.fam_{wf.lower()}'))
        deps.update(_upstream_deps(wf, wc, seed))
    elif fam == 'INC':
        st = json.loads((D.occ_dir(fam, cfg, seed) / 'STATE.json').read_text())
        wf, wc = st['witness']
        deps.update(_run_deps(C.RUNS / 'INC' / 'STEP1' / f'S{seed}'))
        deps.update(_run_deps(D.occ_dir(wf, wc, seed))); deps.update(_code_dep(f'ohn.fam_{wf.lower()}')); deps.update(_code_dep('ohn.fam_a'))
        deps.update(_upstream_deps(wf, wc, seed))
        for rel in (f'admitted/stage_a/S{seed}/TREE_TEACHER_income.json', f'admitted/stage_b/S{seed}/JOINT_TREE__E20.npz',
                    f'admitted/stage_b/S{seed}/LOCAL_TREE__E20.npz'):     # stored income scores (ihs.fam_h.stored_scores)
            deps[rel] = C.sha(C.MIRROR / rel)
    return deps


def build(anchor, occ, seed, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    enc = json.loads(C.admitted(f'admitted/packages/T_TREE_S{seed}/PACKAGE.json').read_text())['encoder']
    if anchor == 'COMPLETE':                         # income-branch complete release: one module predicts both tasks
        fam, cfg = occ.split(':')
        rd = D.occ_dir(fam, cfg, seed)
        deps = _run_deps(rd)
        module = FAMILY_MODULE[fam]
        deps.update(_code_dep(module)); deps.update(_upstream_deps(fam, cfg, seed))
        for fam_a, cfg_a in D.ANCHORS.values():     # the source anchor states and Step-1 learner its income step reuses (mirror)
            for d in (C.MIRROR / 'runs' / fam_a / cfg_a / f'S{seed}', C.MIRROR / 'runs' / 'H' / 'STEP1' / f'S{seed}'):
                deps.update({str(p.relative_to(C.MIRROR)): C.sha(p) for p in sorted(d.iterdir()) if p.is_file()})
        gov, recs = _records()
        deps.update(gov)
        meta = {'schema': 'OHN1-deploy-package-v1', 'anchor': 'COMPLETE', 'anchor_config': None, 'occupation': occ,
                'occupation_module': module, 'seed': seed, 'encoder': enc, 'depends_sha256': deps,
                'release_contract': 'probability and decision per recipient; nothing else', 'permitted_inputs': PERMITTED, 'utc': C.now(),
                'note': 'the module predict returns both outputs; its own records list every upstream file it reads (hash-checked there)',
                **recs}
        (out / 'PACKAGE.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
        return meta
    fam_a, cfg_a = D.ANCHORS[anchor]
    deps = {}
    for d in (C.MIRROR / 'runs' / fam_a / cfg_a / f'S{seed}', C.MIRROR / 'runs' / 'H' / 'STEP1' / f'S{seed}'):
        deps.update({str(p.relative_to(C.MIRROR)): C.sha(p) for p in sorted(d.iterdir()) if p.is_file()})
    for rel in (f'admitted/stage_a/S{seed}/TREE_TEACHER_income.json', f'admitted/stage_a/S{seed}/TREE_TEACHER_occupation.json',
                f'admitted/stage_b/S{seed}/JOINT_TREE__E20.npz', f'admitted/stage_b/S{seed}/LOCAL_TREE__E20.npz'):
        deps[rel] = C.sha(C.MIRROR / rel)
    occ_family, occ_config = occ.split(':')
    module = None
    if occ_family != 'SOURCE':
        deps.update(_run_deps(D.occ_dir(occ_family, occ_config, seed)))
        module = FAMILY_MODULE[occ_family]
        deps.update(_code_dep(module)); deps.update(_upstream_deps(occ_family, occ_config, seed))
    elif occ_config != 'JOINT':
        raise ValueError('SOURCE occupation must be JOINT (the inherited output of the anchor package)')
    gov, recs = _records()
    deps.update(gov)
    meta = {'schema': 'OHN1-deploy-package-v1', 'anchor': anchor, 'anchor_config': cfg_a, 'occupation': occ, 'occupation_module': module,
            'seed': seed, 'encoder': enc, 'depends_sha256': deps, 'release_contract': 'probability and decision per recipient; nothing else',
            'permitted_inputs': PERMITTED, 'utc': C.now(), **recs}
    (out / 'PACKAGE.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return meta


def _load(pkg):
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for rel, h in meta['depends_sha256'].items():
        if rel.startswith('OHN::'):
            p = C.RUNS / rel[5:]
        elif rel.startswith('CODE::'):
            p = C.ROOT / (rel[6:].replace('.', '/') + '.py')
        elif rel.startswith('REPO::'):
            p = C.ROOT / rel[6:]
        else:
            p = C.MIRROR / rel
        if C.sha(p) != h:
            raise PermissionError(f'package dependency hash differs: {rel}')
    return meta


def check_rows(rows):
    for r in rows:
        keys = list(r.keys())
        bad = [k for k in keys if str(k).upper() in FORBIDDEN]
        if bad:
            raise PermissionError(f'forbidden fields present (never accepted as inputs): {bad}')
        if keys != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {keys}')


def check_encoded(meta, X, names):
    names = tuple(map(str, names))
    bad = [n for n in names if n.split('=')[0].upper() in FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present in the encoded input: {bad}')
    if names != tuple(meta['encoder']['feature_names']):
        raise ValueError('encoded input columns differ from the package encoder (missing, extra or reordered columns)')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError('encoded input has the wrong shape or non-finite values')


def underlying(pkg, X, names):
    meta = _load(pkg)
    check_encoded(meta, X, names)
    if meta['anchor'] == 'COMPLETE':
        fam, cfg = meta['occupation'].split(':')
        o = importlib.import_module(meta['occupation_module']).predict(meta['seed'], cfg, X, names)
        return {'income': np.asarray(o['income'], dtype=np.float64), 'occupation': np.asarray(o['occupation'], dtype=np.float64)}
    from ihs import fam_h
    src = fam_h.predict(meta['seed'], meta['anchor_config'], X, names)
    if meta['occupation'] == 'SOURCE:JOINT':
        occ = src['occupation']
    else:
        fam, cfg = meta['occupation'].split(':')
        o = importlib.import_module(meta['occupation_module']).predict(meta['seed'], cfg, X, names)
        occ = o['occupation'] if isinstance(o, dict) else o
    return {'income': np.asarray(src['income'], dtype=np.float64), 'occupation': np.asarray(occ, dtype=np.float64)}


def release(pkg, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    check_rows(rows)
    meta = _load(pkg)
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = underlying(pkg, X, names)
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested} for t in C.TASKS}


def release_csv(pkg, path, requested=('probability', 'decision')):
    with open(path, newline='') as f:
        header = csv.DictReader(f).fieldnames or []
    bad = [h for h in header if h.upper() in FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present (never accepted as inputs): {bad}')
    return release(pkg, SRC_DEPLOY.read_raw_csv(path), requested)


def stored_dev(meta):
    seed = meta['seed']
    if meta['anchor'] == 'COMPLETE':
        fam, cfg = meta['occupation'].split(':')
        return D.load_complete(fam, cfg, seed, roles=('DEV_REPORT',))[0]['DEV_REPORT']
    inc = D.anchor_income(meta['anchor'], seed, 'DEV_REPORT')
    if meta['occupation'] == 'SOURCE:JOINT':
        occ = D.source_occupation('JOINT', seed, 'DEV_REPORT')
    else:
        fam, cfg = meta['occupation'].split(':')
        occ = D.load_occupation(fam, cfg, seed, roles=('DEV_REPORT',))[0]['DEV_REPORT']
    return {'income': inc, 'occupation': occ}


def check(packages, report):
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    raw = R['raw_inputs']
    rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in PERMITTED} for i in range(len(R['row_id']))]
    tmp = Path(tempfile.mkdtemp(dir=C.PRIV)); p = tmp / 'raw.csv'     # person-level raw inputs stay under .private/ (removed below)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PERMITTED); w.writeheader(); w.writerows(rows)
    res = {}
    for pkg in packages:
        t0 = time.process_time()
        meta = _load(pkg)
        ref = stored_dev(meta)
        a = release(pkg, rows); b = release_csv(pkg, p); u = underlying(pkg, R['X'], R['feature_names'])
        same = {name: {t: bool(np.array_equal(o[t]['probability'], ref[t])) for t in C.TASKS}
                for name, o in (('wrapper_api', a), ('wrapper_csv', b), ('underlying', {t: {'probability': u[t]} for t in C.TASKS}))}
        ref_dec = {t: bool(np.array_equal(a[t]['decision'], ref[t].argmax(1))) for t in C.TASKS}
        refusals = {}
        for label, bad in (('wrapper_SEX_field', [{**rows[0], 'SEX': 1}]), ('wrapper_target_field', [{**rows[0], 'OCCP': '0010'}]),
                           ('wrapper_target_PINCP_field', [{**rows[0], 'PINCP': 50000}]),
                           ('wrapper_extra_field', [{**rows[0], 'PUMA': 1}]), ('wrapper_reordered_fields', [dict(reversed(list(rows[0].items())))]),
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
        small = release(pkg, rows[:7])
        res[Path(pkg).name] = {
            'anchor': meta['anchor'], 'occupation': meta['occupation'], 'seed': meta['seed'],
            'bitwise_equal_to_evaluated_DEV_REPORT_release': same, 'decisions_equal': ref_dec, 'refusals': refusals,
            'exported_keys': {t: sorted(a[t]) for t in C.TASKS},
            'small_batch_max_abs_diff': {t: float(np.max(np.abs(small[t]['probability'] - ref[t][:7]))) for t in C.TASKS},
            'cpu_s': round(time.process_time() - t0, 2), 'n_dependencies': len(meta['depends_sha256']),
            'environment_at_build': meta.get('environment_at_build'),
            'code_closure_unchanged_since_build': ({m: C.sha(C.ROOT / (m.replace('.', '/') + '.py')) for m in meta['code_closure_sha256']}
                                                   == meta['code_closure_sha256'] if meta.get('code_closure_sha256') else None)}
    shutil.rmtree(tmp, ignore_errors=True)
    rec = {'schema': 'OHN1-deployment-check-v1', 'utc': C.now(), 'packages': res, 'labels_read': False, 'environment': _environment()}
    rec['all_bitwise'] = all(all(all(v.values()) for v in r['bitwise_equal_to_evaluated_DEV_REPORT_release'].values()) for r in res.values())
    rec['all_refused'] = all(v.startswith('refused') for r in res.values() for v in r['refusals'].values())
    C.write_json(report, rec)
    print(json.dumps({'all_bitwise': rec['all_bitwise'], 'all_refused': rec['all_refused']}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--anchor', required=True, choices=tuple(D.ANCHORS) + ('COMPLETE',))
    b.add_argument('--occ', required=True); b.add_argument('--seed', type=int, required=True); b.add_argument('--out', required=True)
    c = sub.add_parser('check'); c.add_argument('--packages', nargs='+', required=True); c.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        print(json.dumps(build(a.anchor, a.occ, a.seed, a.out)['depends_sha256'], indent=1))
    else:
        rec = check(a.packages, a.report)
        if not (rec['all_bitwise'] and rec['all_refused']):
            raise SystemExit(1)


if __name__ == '__main__':
    main()

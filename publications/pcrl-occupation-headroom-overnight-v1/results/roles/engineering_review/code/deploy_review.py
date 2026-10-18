"""Engineering and deployment review of the overnight occupation-headroom campaign: extended raw-field deployment checks.

LABEL-FREE. Reads only admitted features / raw inputs (DEV_REPORT raw inputs for the wrapper tests; the features of every role for
the deployment-parity replay) and the saved release arrays. Never reads income / occupation / SEX labels, never writes into a
release run directory (the family `verify` commands overwrite VERIFY.json, which deployment packages hash), and keeps every
person-level temporary file under .private/ (deleted at the end).

    <python> results/pcrl_occupation_headroom_overnight_v1/roles/engineering_review/code/deploy_review.py \
        --packages DIR [DIR ...] --out OUT.json [--standard-report CHECK.json]

Per package (built with `python3 -m ohn.deploy build`):
  standard     ohn.deploy.check itself (bitwise parity at the API / CSV / underlying entry points, its refusal set) when
               --standard-report is given;
  refusals     extra refusal cases at all three entry points: SEX / sex / Sex, PINCP, OCCP, extra, reordered, missing and
               duplicated fields; CSV header variants (the CSV path is not exercised by ohn.deploy.check's refusal set), a data
               row longer than the header; renamed encoded columns; non-finite and mis-shaped encoded input; hidden-output
               requests;
  values       behaviour on value-level problems (empty / None / unseen categorical value, empty numeric value) - recorded,
               not a pass criterion;
  contract     exported keys, shapes, dtypes, probability simplex, decision = argmax;
  invariance   a seeded permutation of all DEV_REPORT rows, single-row calls and duplicated rows reproduce the stored outputs
               (the output of a row depends on that row's six fields only);
  roles6       `underlying` from each role's features alone against the saved release arrays of all six roles (bitwise);
  audit        a fresh interpreter per package with sys.addaudithook: every file opened, directory listed, subprocess started
               or socket connected during one API release of all DEV_REPORT rows; files not listed in the package's
               depends_sha256 (runtime dependencies missing from the manifest), label-bearing files, code modules loaded;
  env          interpreter, library versions, BLAS, machine.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ohn import common as C  # noqa: E402
from ohn import deploy as DP  # noqa: E402
from ohn import devrun as D  # noqa: E402

TMP = C.RUNS / 'DEPLOY_REVIEW' / 'tmp'
DEPLOY_MODULE_FILE = None


def use_deploy_module(path):
    """Use a pinned or proposed version of ohn.deploy (loaded as a sibling module of the ohn package) instead of ohn.deploy;
    audit children inherit it."""
    global DP, DEPLOY_MODULE_FILE
    import importlib.util
    name = f'ohn._review_deploy_{C.sha(path)[:12]}'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    DP, DEPLOY_MODULE_FILE = mod, str(path)
    return mod
LABEL_FILE = re.compile(r'/exports/[^/]+/(income|occupation|sex|weights)\.npz$|/extraction/|EXTRACTION_RECEIPT|TRUTH', re.I)


def env():
    out = {'executable': sys.executable, 'python': sys.version.split()[0], 'numpy': np.__version__, 'machine': platform.machine(),
           'platform': platform.platform()}
    for mod in ('sklearn', 'scipy', 'torch', 'xgboost'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        cfg = np.show_config(mode='dicts')
        out['numpy_blas'] = cfg['Build Dependencies']['blas'].get('name')
        out['numpy_lapack'] = cfg['Build Dependencies']['lapack'].get('name')
    except Exception as e:                                   # recorded, not hidden
        out['numpy_blas'] = f'unknown ({type(e).__name__})'
    try:
        out['cpu'] = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    except Exception:
        out['cpu'] = None
    return out


def dev_rows():
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    raw = R['raw_inputs']
    rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in DP.PERMITTED} for i in range(len(R['row_id']))]
    return R, rows


def outcome(fn):
    try:
        fn()
        return 'NOT_REFUSED'
    except (PermissionError, ValueError) as e:
        return f'refused: {type(e).__name__}'
    except Exception as e:                                   # refused, but not by the documented exception types
        return f'refused_other: {type(e).__name__}'


def write_csv(path, header, rows_values):
    with open(path, 'w', newline='') as f:
        f.write(','.join(header) + '\n')
        for vals in rows_values:
            f.write(','.join(str(v) for v in vals) + '\n')
    return path


def refusals(pkg, rows, X, names):
    r0 = rows[0]
    P = list(DP.PERMITTED)
    vals0 = [r0[c] for c in P]
    out = {}
    # documented API entry point (`release`)
    api = {
        'api_SEX_appended': [{**r0, 'SEX': 1}], 'api_sex_lowercase': [{**r0, 'sex': 1}], 'api_Sex_mixedcase_first': [{'Sex': 2, **r0}],
        'api_PINCP_target': [{**r0, 'PINCP': 50000}], 'api_OCCP_target': [{**r0, 'OCCP': '0010'}],
        'api_extra_PUMA': [{**r0, 'PUMA': 1}], 'api_extra_unlisted_FOO': [{**r0, 'FOO': 1}],
        'api_reordered_swap': [{**{k: r0[k] for k in ['WKHP', 'AGEP'] + P[2:]}}],
        'api_reversed': [dict(reversed(list(r0.items())))],
        'api_missing_POBP': [{k: v for k, v in r0.items() if k != 'POBP'}], 'api_missing_AGEP': [{k: v for k, v in r0.items() if k != 'AGEP'}],
        'api_second_row_has_SEX': [r0, {**rows[1], 'SEX': 2}], 'api_second_row_missing_MAR': [r0, {k: v for k, v in rows[1].items() if k != 'MAR'}],
        'api_empty_numeric_AGEP': [{**r0, 'AGEP': ''}], 'api_nonfinite_WKHP': [{**r0, 'WKHP': float('nan')}],
    }
    for label, bad in api.items():
        out[label] = outcome(lambda bad=bad: DP.release(pkg, bad))
    out['api_hidden_output_raw_score'] = outcome(lambda: DP.release(pkg, rows[:3], requested=('probability', 'raw_score')))
    out['api_hidden_output_string_request'] = outcome(lambda: DP.release(pkg, rows[:3], requested='probability'))
    # documented CSV entry point (`release_csv`)
    TMP.mkdir(parents=True, exist_ok=True, mode=0o700)
    csvs = {
        'csv_SEX_column': (P + ['SEX'], [vals0 + [1]]), 'csv_sex_lowercase': (P + ['sex'], [vals0 + [1]]),
        'csv_PINCP_target': (P + ['PINCP'], [vals0 + [50000]]), 'csv_OCCP_target': (P + ['OCCP'], [vals0 + ['0010']]),
        'csv_extra_PUMA': (P + ['PUMA'], [vals0 + [1]]), 'csv_extra_unlisted_FOO': (P + ['FOO'], [vals0 + [1]]),
        'csv_reordered_header': (['WKHP', 'AGEP'] + P[2:], [[vals0[1], vals0[0]] + vals0[2:]]),
        'csv_missing_POBP': (P[:-1], [vals0[:-1]]), 'csv_duplicate_AGEP': (P + ['AGEP'], [vals0 + [vals0[0]]]),
        'csv_row_longer_than_header': (P, [vals0 + [1]]), 'csv_header_with_space': ([' AGEP'] + P[1:], [vals0]),
        'csv_SEX_first_column': (['SEX'] + P, [[1] + vals0]),
    }
    for label, (hdr, body) in csvs.items():
        p = write_csv(TMP / f'{label}.csv', hdr, body)
        out[label] = outcome(lambda p=p: DP.release_csv(pkg, p))
        p.unlink(missing_ok=True)
    # underlying deployment function (`underlying`, encoded features)
    names = list(names); X3 = np.asarray(X[:3])
    ones = np.ones((3, 1))
    und = {
        'und_SEX_appended': (np.hstack((X3, ones)), names + ['SEX']), 'und_sex_lowercase_appended': (np.hstack((X3, ones)), names + ['sex']),
        'und_PINCP_appended': (np.hstack((X3, ones)), names + ['PINCP']), 'und_OCCP_appended': (np.hstack((X3, ones)), names + ['OCCP']),
        'und_extra_appended': (np.hstack((X3, ones)), names + ['EXTRA']), 'und_missing_first': (X3[:, 1:], names[1:]),
        'und_missing_POBP_OTHER': (X3[:, :-1], names[:-1]),
        'und_reordered_swap01': (X3[:, [1, 0] + list(range(2, X3.shape[1]))], [names[1], names[0]] + names[2:]),
        'und_reversed': (X3[:, ::-1], names[::-1]),
        'und_renamed_AGEP_to_SEX': (X3, ['SEX'] + names[1:]), 'und_renamed_COW1_to_SEX=1': (X3, [names[0], names[1], 'SEX=1'] + names[3:]),
        'und_nan_value': (np.where(np.arange(X3.size).reshape(X3.shape) == 0, np.nan, X3), names),
        'und_1d_input': (X3[0], names),
    }
    for label, (Xb, nb) in und.items():
        out[label] = outcome(lambda Xb=Xb, nb=nb: DP.underlying(pkg, Xb, nb))
    return out


def value_behaviour(pkg, rows, ref):
    """Value-level problems (not schema problems): recorded behaviour only."""
    r0 = rows[0]
    out = {}
    for label, row in (('categorical_empty_string_COW', {**r0, 'COW': ''}), ('categorical_None_MAR', {**r0, 'MAR': None}),
                       ('categorical_unseen_code_COW_99', {**r0, 'COW': 99}), ('numeric_None_AGEP', {**r0, 'AGEP': None}),
                       ('numeric_out_of_population_AGEP_7', {**r0, 'AGEP': 7})):
        try:
            o = DP.release(pkg, [row])
            out[label] = {'accepted': True, 'same_as_row0_stored': {t: bool(np.array_equal(o[t]['probability'][0], ref[t][0])) for t in C.TASKS}}
        except Exception as e:
            out[label] = {'accepted': False, 'exception': type(e).__name__}
    return out


def contract(pkg, rows):
    o = DP.release(pkg, rows[:11])
    u = DP.underlying(pkg, DP.SRC_DEPLOY.encode_raw(rows[:11], DP._load(pkg)['encoder']), DP._load(pkg)['encoder']['feature_names'])
    res = {'release_tasks': sorted(o), 'release_keys': {t: sorted(o[t]) for t in o}, 'underlying_keys': sorted(u)}
    ok = res['release_tasks'] == ['income', 'occupation'] and all(res['release_keys'][t] == ['decision', 'probability'] for t in o) \
        and res['underlying_keys'] == ['income', 'occupation']
    for t, k in (('income', 2), ('occupation', 5)):
        p, d = o[t]['probability'], o[t]['decision']
        good = (p.dtype == np.float64 and p.shape == (11, k) and d.dtype == np.int64 and d.shape == (11,) and np.isfinite(p).all()
                and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-12, rtol=0) and np.array_equal(d, p.argmax(1))
                and u[t].dtype == np.float64 and np.array_equal(u[t], p))
        res[f'{t}_ok'] = bool(good); ok &= bool(good)
    res['ok'] = bool(ok)
    return res


def invariance(pkg, rows, ref):
    n = len(rows)
    perm = np.random.default_rng(20261018).permutation(n)
    o = DP.release(pkg, [rows[i] for i in perm])
    res = {'permutation_all_rows': {t: {'bitwise': bool(np.array_equal(o[t]['probability'], ref[t][perm])),
                                        'max_abs_diff': float(np.max(np.abs(o[t]['probability'] - ref[t][perm])))} for t in C.TASKS}}
    idx = np.random.default_rng(7).choice(n, 25, replace=False)
    single = {t: [] for t in C.TASKS}
    for i in idx:
        oi = DP.release(pkg, [rows[i]])
        for t in C.TASKS:
            single[t].append(float(np.max(np.abs(oi[t]['probability'][0] - ref[t][i]))))
    res['single_row_calls_25'] = {t: {'max_abs_diff': max(v), 'bitwise_all': bool(max(v) == 0.0)} for t, v in single.items()}
    i, j = int(idx[0]), int(idx[1])
    od = DP.release(pkg, [rows[i], rows[j], rows[i], rows[i]])
    res['duplicate_rows_identical'] = bool(all(np.array_equal(od[t]['probability'][0], od[t]['probability'][k]) for t in C.TASKS for k in (2, 3)))
    res['ok_1e12'] = bool(all(v['max_abs_diff'] <= 1e-12 for v in res['permutation_all_rows'].values())
                          and all(v['max_abs_diff'] <= 1e-12 for v in res['single_row_calls_25'].values()) and res['duplicate_rows_identical'])
    return res


def stored_role(meta, role):
    seed = meta['seed']
    if meta['anchor'] == 'COMPLETE':
        fam, cfg = meta['occupation'].split(':')
        return D.load_complete(fam, cfg, seed, roles=(role,))[0][role]
    inc = D.anchor_income(meta['anchor'], seed, role)
    if meta['occupation'] == 'SOURCE:JOINT':
        occ = D.source_occupation('JOINT', seed, role)
    else:
        fam, cfg = meta['occupation'].split(':')
        occ = D.load_occupation(fam, cfg, seed, roles=(role,))[0][role]
    return {'income': inc, 'occupation': occ}


def roles6(pkg):
    meta = DP._load(pkg)
    out = {}
    for role in C.ROLES6:
        F = C.load_role(role, ('features',))
        u = DP.underlying(pkg, F['X'], F['feature_names'])
        ref = stored_role(meta, role)
        out[role] = {t: {'bitwise': bool(np.array_equal(u[t], ref[t])), 'max_abs_diff': float(np.max(np.abs(u[t] - ref[t])))} for t in C.TASKS}
    out['all_bitwise'] = bool(all(v[t]['bitwise'] for k, v in out.items() if k in C.ROLES6 for t in C.TASKS))
    return out


def load_module(fam, module_file=None):
    """The family module, or (module_file given) a pinned source version of it loaded as a sibling module of the ohn package
    (relative imports resolve to ohn.*), e.g. a committed blob extracted with `git show <commit>:ohn/protect.py`."""
    import importlib
    import importlib.util
    if not module_file:
        return importlib.import_module(DP.FAMILY_MODULE[fam]), C.sha(C.ROOT / (DP.FAMILY_MODULE[fam].replace('.', '/') + '.py'))
    h = C.sha(module_file)
    name = f"ohn._review_{fam.lower()}_{h[:12]}"
    spec = importlib.util.spec_from_file_location(name, module_file)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'ohn'
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, h


def module_parity(spec):
    """FAM:CONFIG:SEED[:MODULE_FILE]: the module's predict() from each role's features alone against the saved release (bitwise)."""
    parts = spec.split(':')
    fam, cfg, seed = parts[0], parts[1], int(parts[2])
    mod, h = load_module(fam, parts[3] if len(parts) > 3 else None)
    out = {'module_sha256': h, 'module_file_given': len(parts) > 3}
    for role in C.ROLES6:
        F = C.load_role(role, ('features',))
        p = mod.predict(seed, cfg, F['X'], F['feature_names'])
        if fam == 'INC':
            ref = D.load_complete(fam, cfg, seed, roles=(role,))[0][role]
            out[role] = {t: {'bitwise': bool(np.array_equal(p[t], ref[t])), 'max_abs_diff': float(np.max(np.abs(p[t] - ref[t])))} for t in C.TASKS}
        else:
            ref = D.load_occupation(fam, cfg, seed, roles=(role,))[0][role]
            out[role] = {'occupation': {'bitwise': bool(np.array_equal(p, ref)), 'max_abs_diff': float(np.max(np.abs(p - ref)))}}
    out['all_bitwise'] = bool(all(v[t]['bitwise'] for k, v in out.items() if k in C.ROLES6 for t in v))
    return out


def grid_replay():
    """Label-free: the development reader grid's household partition recomputed with the current code (ohn.devreaders.manifest:
    ohn.common.household_folds, which lies outside the DEVGRID_CODE_LOCK file list) against the registered DEV_GRID.json."""
    from ohn import devreaders as DR
    reg = json.loads((C.RESULTS / 'DEV_GRID.json').read_text())
    cur = DR.manifest()
    lock = json.loads((C.RESULTS / 'DEVGRID_CODE_LOCK.json').read_text())
    return {'repetitions_equal': cur['repetitions'] == reg['repetitions'],
            'overlap_equal': cur['household_overlap_between_repetitions'] == reg['household_overlap_between_repetitions'],
            'cells_equal': cur['cells'] == reg['cells'], 'reader_seeds_equal': cur['reader_seeds'] == reg['reader_seeds'],
            'lock_dev_grid_sha256_equal': lock['dev_grid_sha256'] == C.sha(C.RESULTS / 'DEV_GRID.json'),
            'lock_files_equal': all(C.sha(C.ROOT / f) == h for f, h in lock['files_sha256'].items()),
            'common_py_sha256': C.sha(C.ROOT / 'ohn' / 'common.py')}


# ---------------------------------------------------------------- runtime audit (fresh interpreter per package)
def audit_child(pkg):
    """Runs in its own interpreter: one API release of all DEV_REPORT rows under an audit hook."""
    R, rows = dev_rows()
    before = set(sys.modules)
    events = {'open': [], 'listdir': [], 'subprocess': [], 'socket': []}

    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            events['open'].append(os.fsdecode(args[0]))
        elif event in ('os.listdir', 'os.scandir') and args:
            events['listdir'].append(os.fsdecode(args[0]) if isinstance(args[0], (str, bytes, os.PathLike)) else str(args[0]))
        elif event == 'subprocess.Popen':
            events['subprocess'].append(str(args[1] if len(args) > 1 else args)[:200])
        elif event.startswith('socket.connect'):
            events['socket'].append(str(args)[:200])
    sys.addaudithook(hook)
    t0 = time.process_time()
    DP.release(pkg, rows)
    cpu = time.process_time() - t0
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    listed = set()
    for rel in meta['depends_sha256']:
        if rel.startswith('OHN::'):
            listed.add(str((C.RUNS / rel[5:]).resolve()))
        elif rel.startswith('CODE::'):
            listed.add(str((C.ROOT / (rel[6:].replace('.', '/') + '.py')).resolve()))
        elif rel.startswith('REPO::'):
            listed.add(str((C.ROOT / rel[6:]).resolve()))
        else:
            listed.add(str((C.MIRROR / rel).resolve()))
    root = str(C.ROOT.resolve())
    opened = sorted({str(Path(p).resolve()) for p in events['open'] if not p.endswith('.pyc')})
    data_files = [p for p in opened if p.startswith(root) and not p.endswith('.py') and Path(p).is_file()]
    unlisted_data = sorted(p[len(root) + 1:] for p in data_files if p not in listed and p != str((Path(pkg) / 'PACKAGE.json').resolve()))
    label_files = sorted(p for p in opened if LABEL_FILE.search(p))
    new_mods = sorted(set(sys.modules) - before)
    repo_code = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and str(Path(f).resolve()).startswith(root) and f.endswith('.py'):
            repo_code[str(Path(f).resolve())[len(root) + 1:]] = C.sha(f)
    code_listed = {str((C.ROOT / (rel[6:].replace('.', '/') + '.py')).resolve())[len(root) + 1:] for rel in meta['depends_sha256'] if rel.startswith('CODE::')}
    code_listed |= {m.replace('.', '/') + '.py' for m in meta.get('code_closure_sha256', {})}      # recorded code closure (proposed patch)
    libs = {}
    for name in ('numpy', 'sklearn', 'scipy', 'torch', 'xgboost'):
        if name in sys.modules:
            libs[name] = getattr(sys.modules[name], '__version__', None)
    return {'cpu_s_release_all_dev_rows': round(cpu, 3), 'n_files_opened': len(opened), 'unlisted_runtime_data_files': unlisted_data,
            'label_bearing_files_opened': label_files, 'subprocesses': events['subprocess'], 'sockets': events['socket'],
            'directories_listed': sorted(set(events['listdir']))[:20],
            'repo_code_modules_loaded_sha256': dict(sorted(repo_code.items())),
            'repo_code_modules_not_in_manifest': sorted(set(repo_code) - code_listed),
            'third_party_libraries_loaded': libs, 'modules_newly_imported_during_release': len(new_mods)}


def audit(pkg):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--audit-child', str(pkg)]
    if DEPLOY_MODULE_FILE:
        cmd += ['--deploy-module', DEPLOY_MODULE_FILE]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.ROOT))
    if r.returncode != 0:
        return {'error': r.stderr[-1500:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--packages', nargs='*', default=[]); ap.add_argument('--out')
    ap.add_argument('--standard-report'); ap.add_argument('--audit-child')
    ap.add_argument('--module-parity', nargs='*', default=[], help='FAM:CONFIG:SEED[:MODULE_FILE]')
    ap.add_argument('--grid-replay', action='store_true')
    ap.add_argument('--deploy-module', help='a pinned or proposed ohn/deploy.py version to use instead of ohn.deploy')
    a = ap.parse_args(argv)
    if a.deploy_module:
        use_deploy_module(a.deploy_module)
    if a.audit_child:
        print(json.dumps(audit_child(a.audit_child), default=str))
        return 0
    t0 = time.process_time(); w0 = time.time()
    rec = {'schema': 'OHN1-engineering-deploy-review-v1', 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'env': env(),
           'labels_read': False, 'code_sha256': C.sha(__file__), 'packages': {}}
    if a.standard_report:
        import tempfile                                       # ohn.deploy.check writes a person-level raw CSV with mkdtemp() and
        TMP.mkdir(parents=True, exist_ok=True, mode=0o700)    # never deletes it: keep it under .private/ (removed at the end)
        tempfile.tempdir = str(TMP)
        std = DP.check(a.packages, a.standard_report)
        rec['standard_check'] = {'report': a.standard_report, 'all_bitwise': std['all_bitwise'], 'all_refused': std['all_refused'],
                                 'per_package': {k: {'bitwise': v['bitwise_equal_to_evaluated_DEV_REPORT_release'], 'decisions_equal': v['decisions_equal'],
                                                     'refusals': v['refusals'], 'exported_keys': v['exported_keys'],
                                                     'small_batch_max_abs_diff': v['small_batch_max_abs_diff']} for k, v in std['packages'].items()}}
    R, rows = dev_rows()
    for pkg in a.packages:
        meta = DP._load(pkg)
        ref = DP.stored_dev(meta)
        res = {'anchor': meta['anchor'], 'occupation': meta['occupation'], 'seed': meta['seed'], 'occupation_module': meta['occupation_module'],
               'n_dependencies': len(meta['depends_sha256']),
               'package_records': {k: (k in meta) for k in ('interpreter', 'python', 'numpy', 'sklearn', 'cpu', 'code_sha256', 'versions')}}
        res['refusals'] = refusals(pkg, rows, R['X'], R['feature_names'])
        res['all_refused_cleanly'] = all(v.startswith('refused: ') for v in res['refusals'].values())
        res['all_refused'] = all(v.startswith('refused') for v in res['refusals'].values())
        res['value_level_behaviour'] = value_behaviour(pkg, rows, ref)
        res['contract'] = contract(pkg, rows)
        res['invariance'] = invariance(pkg, rows, ref)
        res['roles6'] = roles6(pkg)
        res['audit'] = audit(pkg)
        rec['packages'][Path(pkg).name] = res
        print(json.dumps({Path(pkg).name: {'refused': res['all_refused'], 'contract': res['contract']['ok'], 'invariance': res['invariance']['ok_1e12'],
                                           'roles6_bitwise': res['roles6']['all_bitwise'],
                                           'unlisted': len(res['audit'].get('unlisted_runtime_data_files', [])),
                                           'label_files': res['audit'].get('label_bearing_files_opened')}}), flush=True)
    rec['module_parity'] = {}
    for spec in a.module_parity:
        try:
            rec['module_parity'][spec.split(':')[0] + ':' + spec.split(':')[1] + ':' + spec.split(':')[2]] = module_parity(spec)
        except Exception as e:                                # recorded, not hidden
            rec['module_parity'][spec] = {'error': f'{type(e).__name__}: {e}'[:500]}
        print(json.dumps({'module_parity': spec.split(':')[:3], 'all_bitwise': rec['module_parity'].get(':'.join(spec.split(':')[:3]), {}).get('all_bitwise')}), flush=True)
    if a.grid_replay:
        rec['grid_replay'] = grid_replay()
        print(json.dumps({'grid_replay': rec['grid_replay']}), flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    rec['cpu_s'] = round(time.process_time() - t0, 2); rec['wall_s'] = round(time.time() - w0, 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

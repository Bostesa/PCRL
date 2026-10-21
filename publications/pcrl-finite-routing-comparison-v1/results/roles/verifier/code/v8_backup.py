"""V8 independent check of the same-device backup (label-free), own code and own fresh restore.

    python3 .../v8_backup.py manifest --backup DIR --tip SHA --tag V1
    python3 .../v8_backup.py extract  --backup DIR --restore NEWDIR --scratch-root ROOT --tag V1
    python3 .../v8_backup.py deploy   --restore NEWDIR --tag V1        (python3; restored code first on sys.path)
    <torch venv python> .../v8_backup.py custody --restore NEWDIR --tag V1
    python3 .../v8_backup.py cleanup  --restore NEWDIR --scratch-root ROOT --tag V1
    python3 .../v8_backup.py selftest                                   (invented files in temporary directories only)

The scratch root (extract, cleanup) comes from --scratch-root, else the FRC_VER_SCRATCH_ROOT environment variable; there is no
default (no machine-specific path in this file).

manifest: SHA-256 of TIP_COMMIT, branch_since_base.bundle, code.tar and private.tar against MANIFEST.sha256; `git bundle verify`
  (prerequisite base in this repository) and `git bundle list-heads`; TIP_COMMIT equals the stated tip.
extract: tar -xf of code.tar and private.tar into a NEW directory under my scratchpad (refused if it exists).
deploy: in the restored copy, for FA/PRC_S2, FA/TWB_C64, FA/HQ_S2 and both versions: the restored DEV_REPORT raw inputs encoded by
  the package's own encoder (acs_replication.deploy.encode_raw from the RESTORED code; data transform) must equal the restored
  admitted DEV_REPORT features bitwise; then my own replay of the deployment function on those features: the restored MODEL.pkl
  teachers' decision functions on the admitted compact coding (restored ohn.fam_a.compact_po), my own probability transform and my
  own routing (v5b route_book), the stored tables; compared bitwise with the stored DEV_REPORT wires of the deployment store
  (runs/STORED) and of RELEASE.npz. Every module used is checked to come from the restored root. Refusal probe: the restored
  frc.fa.route.predict refuses per-request keys, a draw other than 0 and reordered feature columns.
custody: the restored runs/READER_CUSTODY/*.pkl re-loaded from the restored copy (the pickled predict function must resolve to the
  restored code); predict(model, X_score) must equal score_raw bitwise; the pickle hashes against the custody records.
cleanup: the restore directory is deleted and its absence confirmed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import vlib as V  # noqa: E402

FILES = ('TIP_COMMIT', 'branch_since_base.bundle', 'code.tar', 'private.tar')
CONFIGS = ('PRC_S2', 'TWB_C64', 'HQ_S2')
SCRATCH_ENV = 'FRC_VER_SCRATCH_ROOT'


def scratch_root(arg):
    """The scratch root that a restore must lie under (extract) and that cleanup may delete inside: the --scratch-root argument,
    else the FRC_VER_SCRATCH_ROOT environment variable; there is no default."""
    import os
    root = arg or os.environ.get(SCRATCH_ENV)
    if not root:
        raise SystemExit(f'a scratch root is required (--scratch-root or {SCRATCH_ENV}); there is no default')
    return Path(root)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(backup, tip, tag):
    b = Path(backup)
    man = {}
    for line in (b / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, name = line.split()
            man[name] = h
    hashes = {f: sha(b / f) for f in FILES}
    eq = {f: hashes[f] == man.get(f) for f in FILES}
    r = subprocess.run(['git', '-C', str(V.ROOT), 'bundle', 'verify', str(b / 'branch_since_base.bundle')], capture_output=True, text=True)
    heads = subprocess.run(['git', '-C', str(V.ROOT), 'bundle', 'list-heads', str(b / 'branch_since_base.bundle')], capture_output=True,
                           text=True).stdout.split()
    tip_file = (b / 'TIP_COMMIT').read_text().strip()
    res = {'manifest': man, 'mine': hashes, 'equal': eq, 'bundle_verify_rc': r.returncode,
           'bundle_verify_output': (r.stdout + r.stderr).strip()[-600:], 'bundle_heads': heads,
           'tip_file': tip_file, 'tip_equals_expected': tip_file == tip, 'tip_in_bundle_heads': tip in heads,
           'manifest_lists_exactly_these_files': sorted(man) == sorted(FILES)}
    ok = all(eq.values()) and r.returncode == 0 and res['tip_equals_expected'] and res['tip_in_bundle_heads'] and \
        res['manifest_lists_exactly_these_files']
    return [{'id': f'V8.BACKUP_MANIFEST__{tag}', 'what': 'backup files re-hashed against MANIFEST.sha256, git bundle verify, '
             'bundle heads and TIP_COMMIT equal to the stated tip', 'inputs': {'backup': str(b)}, 'result': res,
             'numerical_status': 'PASS' if all(eq.values()) else 'FAIL', 'procedural_status': 'PASS' if ok else 'FAIL'}]


def extract(backup, restore, tag, scratch):
    b, d = Path(backup), Path(restore)
    if d.exists():
        raise PermissionError(f'{d} exists: the restore must be fresh')
    if not str(d.resolve()).startswith(str(scratch)):
        raise PermissionError('the restore must be under the scratchpad')
    d.mkdir(parents=True)
    out = {}
    for t in ('code.tar', 'private.tar'):
        r = subprocess.run(['tar', '-xf', str(b / t), '-C', str(d)], capture_output=True, text=True)
        out[t] = {'rc': r.returncode, 'stderr': r.stderr[-300:]}
    n = sum(1 for p in d.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
    res = {'restore': str(d), 'tar': out, 'files': n, 'bytes': size,
           'has': {k: (d / k).exists() for k in ('frc/fa/route.py', 'ohn/fam_a.py', 'acs_replication/deploy.py',
                                                  '.private/pcrl_finite_routing_comparison_v1/runs/FA/PRC_S2/V0/MODEL.pkl',
                                                  '.private/pcrl_finite_routing_comparison_v1/runs/READER_CUSTODY',
                                                  '.private/pcrl_income_headroom_successor_v1/admitted')}}
    ok = all(x['rc'] == 0 for x in out.values()) and all(res['has'].values())
    return [{'id': f'V8.BACKUP_EXTRACT__{tag}', 'what': 'fresh extraction of code.tar and private.tar into my own scratchpad directory',
             'inputs': {'backup': str(b)}, 'result': res, 'numerical_status': 'NOT_APPLICABLE',
             'procedural_status': 'PASS' if ok else 'FAIL'}]


def _restored_import(restore):
    root = str(Path(restore).resolve())
    sys.path.insert(0, root)
    for m in [m for m in list(sys.modules) if m.split('.')[0] in ('frc', 'nrm', 'ohn', 'acs_replication', 'scm_bench', 'tsz', 'ihs',
                                                                  'task_conditioned_representation', 'ffv', 'cts')]:
        del sys.modules[m]
    return root


def _from_root(mod, root):
    f = getattr(mod, '__file__', None)
    return bool(f) and str(Path(f).resolve()).startswith(root)


def deploy(restore, tag):
    import numpy as np
    root = _restored_import(restore)
    import frc.common as C
    from acs_replication import deploy as SD
    from ohn import fam_a as FA
    import v2_routing as R2
    import v5b_protected_tables as T
    mods = {'frc.common': _from_root(C, root), 'acs_replication.deploy': _from_root(SD, root), 'ohn.fam_a': _from_root(FA, root)}
    R = C.features('DEV_REPORT', raw=True)
    raw = R['raw_inputs']
    names = tuple(map(str, R['feature_names']))
    priv = Path(root) / '.private' / 'pcrl_finite_routing_comparison_v1'
    res, ok = {}, all(mods.values())
    for cfg in CONFIGS:
        for v in (0, 1):
            key = f'{cfg}_V{v}'
            meta = json.loads((priv / 'deploy' / key / 'PACKAGE.json').read_text())
            enc = meta['encoder']
            perm = list(meta.get('permitted_inputs') or ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP'])
            rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in perm} for i in range(len(R['row_id']))]
            X = SD.encode_raw(rows, enc)
            enc_eq = bool(np.array_equal(np.asarray(X, dtype=np.float64), np.asarray(R['X'], dtype=np.float64))) and \
                tuple(enc['feature_names']) == names
            md = priv / 'runs' / 'FA' / cfg / f'V{v}'
            buf = (md / 'MODEL.pkl').read_bytes()
            model_ok = V.bytes_sha(buf) == json.loads((md / 'MODEL.json').read_text())['model_sha256']
            M = pickle.loads(buf)
            Z = FA.compact_po(np.asarray(X, dtype=np.float64), names)
            E = {'qI': R2.probs_from_raw(M['teachers']['inc'].decision_function(Z)),
                 'qO': R2.probs_from_raw(M['teachers']['occ'].decision_function(Z)), 'Z': Z}
            if 'sex' in M['teachers']:
                E['pi'] = np.minimum(np.maximum(R2.probs_from_raw(M['teachers']['sex'].decision_function(Z))[:, 1], 0.01), 0.99)
                E['qO0'] = R2.probs_from_raw(M['teachers']['occ_s'].decision_function(np.column_stack([Z, np.zeros(len(Z))])))
                E['qO1'] = R2.probs_from_raw(M['teachers']['occ_s'].decision_function(np.column_stack([Z, np.ones(len(Z))])))
            k, l, gap = T.route_book(E, M['book'])
            mine = {'income': np.asarray(M['dec']['I'])[k], 'occupation': np.asarray(M['dec']['O'])[l]}
            with np.load(priv / 'runs' / 'STORED' / f'FA__{cfg}__V{v}.npz') as z:
                stored = {t: z[f'{t}__DEV_REPORT'].copy() for t in V.TASKS}
            rel = json.loads((md / 'RELEASE.json').read_text())
            rel_ok = sha(md / 'RELEASE.npz') == rel['release_sha256']
            with np.load(md / 'RELEASE.npz') as z:
                saved = {t: z[f'DEV_REPORT__{t}'].copy() for t in V.TASKS}
            r = {'encoder_equals_admitted_features': enc_eq, 'model_hash_ok': model_ok, 'release_hash_ok': rel_ok, 'min_gap': gap,
                 'equal_stored_wire': {t: bool(np.array_equal(mine[t], stored[t])) for t in V.TASKS},
                 'equal_release_npz': {t: bool(np.array_equal(mine[t], saved[t])) for t in V.TASKS},
                 'people': int(len(k)), 'teachers': sorted(M['teachers'])}
            res[key] = r
            ok &= enc_eq and model_ok and rel_ok and all(r['equal_stored_wire'].values()) and all(r['equal_release_npz'].values())
    # refusal probe of the restored deployment function
    import frc.fa.route as FR
    mods['frc.fa.route'] = _from_root(FR, root)
    ref = {}
    X3 = np.asarray(R['X'], dtype=np.float64)[:3]
    for label, kw in (('keys', {'keys': {'role': 'DEV_REPORT', 'row_id': np.arange(3)}}), ('draw_1', {'draw': 1})):
        try:
            FR.predict('PRC_S2', 0, X3, names, **kw)
            ref[label] = 'NOT_REFUSED'
        except PermissionError:
            ref[label] = 'refused'
    try:
        FR.predict('PRC_S2', 0, X3[:, ::-1], names[::-1])
        ref['reordered_columns'] = 'NOT_REFUSED'
    except PermissionError:
        ref['reordered_columns'] = 'refused'
    ok &= all(x == 'refused' for x in ref.values()) and all(mods.values())
    return [{'id': f'V8.BACKUP_DEPLOY__{tag}', 'what': 'own replay of the deployment of FA/PRC_S2, FA/TWB_C64, FA/HQ_S2 (both versions) in my '
             'fresh restore from the restored DEV_REPORT raw inputs (package encoder), vs the stored DEV_REPORT wires and RELEASE.npz '
             '(bitwise); modules from the restored root; refusal probe', 'inputs': {'restore': root},
             'result': {'modules_from_restored_root': mods, 'per_config': res, 'refusals': ref},
             'numerical_status': 'PASS' if all(all(r['equal_stored_wire'].values()) and all(r['equal_release_npz'].values())
                                                 and r['encoder_equals_admitted_features'] for r in res.values()) else 'FAIL',
             'procedural_status': 'PASS' if ok else 'FAIL'}]


def custody(restore, tag):
    import numpy as np
    root = _restored_import(restore)
    import frc.common as C
    priv = Path(root) / '.private' / 'pcrl_finite_routing_comparison_v1'
    recs = {}
    for p in sorted((Path(root) / 'results' / 'pcrl_finite_routing_comparison_v1' / 'custody').glob('READER_CUSTODY_*.json')):
        for x in json.loads(p.read_text()).get('readers', []):
            recs[x['file']] = x['sha256']
    res, ok = {}, _from_root(C, root)
    files = sorted((priv / 'runs' / 'READER_CUSTODY').glob('*.pkl'))
    for p in files:
        h = sha(p)
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        pred = obj['predict']
        mod = sys.modules.get(getattr(pred, '__module__', ''), None)
        out = pred(obj['model'], obj['X_score'])
        eq = bool(np.array_equal(np.asarray(out), np.asarray(obj['score_raw'])))
        r = {'hash_equals_record': recs.get(p.name) == h, 'predict_module': getattr(pred, '__module__', None),
             'predict_from_restored_root': bool(mod is not None and _from_root(mod, root)), 'rescore_bitwise': eq,
             'recipe': obj.get('recipe'), 'n_score': int(len(obj['score_raw']))}
        res[p.name] = r
        ok &= all(v for k, v in r.items() if isinstance(v, bool))
    return [{'id': f'V8.BACKUP_CUSTODY__{tag}', 'what': 'the restored reader-custody pickles re-loaded from my fresh restore under the torch '
             'venv: predict(model, X_score) equals score_raw bitwise, predict resolves to the restored code, hashes equal the custody '
             'records', 'inputs': {'restore': root, 'python': sys.version.split()[0]},
             'result': {'n_pickles': len(files), 'per_pickle': res},
             'numerical_status': 'PASS' if (files and all(r['rescore_bitwise'] for r in res.values())) else 'FAIL',
             'procedural_status': 'PASS' if (ok and files) else 'FAIL'}]


def cleanup(restore, tag, scratch):
    import shutil
    d = Path(restore)
    if not str(d.resolve()).startswith(str(scratch)):
        raise PermissionError('refusing to delete outside the scratchpad')
    shutil.rmtree(d, ignore_errors=True)
    return [{'id': f'V8.BACKUP_CLEANUP__{tag}', 'what': 'my restore directory deleted', 'inputs': {'restore': str(d)},
             'result': {'exists_after': d.exists()}, 'numerical_status': 'NOT_APPLICABLE',
             'procedural_status': 'PASS' if not d.exists() else 'FAIL'}]


def selftest():
    """Invented files in temporary directories only (no real path is touched): the scratch root is required (no default), extract
    refuses an existing directory and a restore outside the scratch root and extracts invented tars inside it, cleanup refuses
    outside the scratch root and deletes inside it, and the manifest hashes flag a changed file."""
    import os
    import tarfile
    import tempfile
    ok = {}
    saved = os.environ.pop(SCRATCH_ENV, None)
    try:
        try:
            scratch_root(None)
            ok['no_default_scratch_root'] = False
        except SystemExit:
            ok['no_default_scratch_root'] = True
        os.environ[SCRATCH_ENV] = '/nonexistent/root'
        ok['environment_variable_used'] = str(scratch_root(None)) == '/nonexistent/root'
        ok['argument_wins_over_environment'] = str(scratch_root('/a/b')) == '/a/b'
    finally:
        os.environ.pop(SCRATCH_ENV, None)
        if saved is not None:
            os.environ[SCRATCH_ENV] = saved
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        root, other, b = tmp / 'scratch', tmp / 'elsewhere', tmp / 'backup'
        for d in (root, other, b):
            d.mkdir()
        src = tmp / 'src'
        (src / 'frc').mkdir(parents=True)
        (src / 'frc' / 'x.py').write_text('x = 1\n')
        with tarfile.open(b / 'code.tar', 'w') as t:
            t.add(src / 'frc', arcname='frc')
        with tarfile.open(b / 'private.tar', 'w') as t:
            t.add(src / 'frc' / 'x.py', arcname='.private/y.py')
        try:
            extract(b, other / 'restore', 'T', root)
            ok['extract_refuses_outside_scratch'] = False
        except PermissionError:
            ok['extract_refuses_outside_scratch'] = True
        try:
            extract(b, root, 'T', root)
            ok['extract_refuses_existing_directory'] = False
        except PermissionError:
            ok['extract_refuses_existing_directory'] = True
        r = extract(b, root / 'restore', 'T', root)[0]['result']
        ok['extract_inside_scratch'] = all(x['rc'] == 0 for x in r['tar'].values()) and r['files'] == 2 and \
            (root / 'restore' / 'frc' / 'x.py').is_file() and (root / 'restore' / '.private' / 'y.py').is_file()
        try:
            cleanup(other, 'T', root)
            ok['cleanup_refuses_outside_scratch'] = False
        except PermissionError:
            ok['cleanup_refuses_outside_scratch'] = True
        ok['cleanup_deletes_inside_scratch'] = cleanup(root / 'restore', 'T', root)[0]['result']['exists_after'] is False and \
            other.is_dir()
        f = b / 'code.tar'
        h0 = sha(f)
        with open(f, 'ab') as fh:
            fh.write(b'\0')
        ok['manifest_hash_detects_change'] = sha(f) != h0
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('manifest', 'extract', 'deploy', 'custody', 'cleanup', 'selftest'))
    ap.add_argument('--backup')
    ap.add_argument('--restore')
    ap.add_argument('--tip')
    ap.add_argument('--scratch-root', help=f'root under which the restore must lie (extract, cleanup); else ${SCRATCH_ENV}; no default')
    ap.add_argument('--tag')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    if not a.tag:
        raise SystemExit('--tag is required')
    fn = {'manifest': lambda: manifest(a.backup, a.tip, a.tag),
          'extract': lambda: extract(a.backup, a.restore, a.tag, scratch_root(a.scratch_root)),
          'deploy': lambda: deploy(a.restore, a.tag), 'custody': lambda: custody(a.restore, a.tag),
          'cleanup': lambda: cleanup(a.restore, a.tag, scratch_root(a.scratch_root))}[a.cmd]
    checks = fn()
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'])
        print(json.dumps(c['result'], default=str)[:3000])
    return 0


if __name__ == '__main__':
    sys.exit(main())

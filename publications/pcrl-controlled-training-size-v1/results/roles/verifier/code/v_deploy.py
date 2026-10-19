"""Verifier deployment check (stages V1 and V6): a package directory is replayed label-free from the six raw fields.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Inputs: DEV_REPORT and MODEL_FIT raw inputs and admitted features (tsz.common.features(role, raw=True); no label of any role is
read), the package's PACKAGE.json, the stored wire .private/<campaign>/runs/STORED/<name>__V<v>.npz, the evaluated release arrays.

Per package (version v):
  dependencies   every depends_sha256 entry re-hashed by the verifier's own resolver
  encoder        own raw-field encoder (from the package encoder spec) == the admitted 145-column encoding, bitwise (DEV_REPORT,
                 MODEL_FIT: the two roles with admitted raw inputs)
  stored         the stored wire == the evaluated release arrays (A1: runs/SIZE/<fit id>/RELEASE.npz) on every role, bitwise
  wrapper        tsz.deploy.release(pkg, rows) and tsz.deploy.release_csv(pkg, own CSV of the six fields) on the raw rows of
                 DEV_REPORT and MODEL_FIT: occupation probability == stored wire bitwise; decision == argmax, int64; income ==
                 the admitted H3 S<v> wire bitwise; exported keys exactly probability and decision
  underlying     tsz.deploy.underlying(pkg, X, names): own raw encoding (DEV_REPORT, MODEL_FIT) and the admitted encoding of the
                 four roles without raw inputs (CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION) == stored wire bitwise
  own pipeline   (A1 packages) own encoder -> own FIT-only dictionary application (DICT.json) -> own compact_po -> the saved
                 MODEL.pkl decision function -> own softmax, vs the stored wire (bitwise / max abs difference)
  refusals       wrapper: SEX, sex, OCCP, PINCP, extra (PUMA), missing (POBP), reordered fields, SEX on a later row, hidden output
                 requests; csv: SEX, OCCP, PINCP, extra, missing, reordered columns, hidden output request; underlying: SEX, OCCP,
                 PINCP, extra columns appended, SEX renamed in place of AGEP, missing and reordered columns. A refusal passes iff the
                 call raises PermissionError or ValueError (no output returned).
`bitwise` (per version) = every occupation comparison above that uses the package (wrapper api, wrapper csv, underlying; all six
roles) and stored == evaluated, all bitwise. `refusals_all_pass` = every refusal raised.

    python3 .../v_deploy.py --packages DIR1,DIR2 --tag V1 [--w100]
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (label-free admitted features and raw inputs only)

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
RUNS = PRIV / 'runs'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
RAW_ROLES = ('DEV_REPORT', 'MODEL_FIT')
TASKS = ('income', 'occupation')
DEPLOY_W100 = V.VER_DIR / 'DEPLOY_W100.json'


def resolve(key):
    if key.startswith('TSZ::'):
        return RUNS / key[5:]
    if key.startswith('CODE::'):
        return ROOT / (key[6:].replace('.', '/') + '.py')
    if key.startswith('OHN::'):
        return OHN_RUNS / key[5:]
    if key.startswith('REPO::'):
        return ROOT / key[6:]
    return MIRROR / key


def stored_name(name, v):
    return name.replace(':', '__').replace('@', '_AT_') + f'__V{v}.npz'


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def evaluated_release(name, v=None):
    kind, rest = name.split(':', 1)
    if kind == 'A1':
        d = RUNS / 'SIZE' / rest
        rec = json.loads((d / 'FIT.json').read_text())
        if V.file_sha(d / 'RELEASE.npz') != rec['files_sha256']['RELEASE.npz']:
            raise PermissionError('RELEASE.npz differs from FIT.json')
        return npz(d / 'RELEASE.npz'), [d / 'RELEASE.npz', d / 'FIT.json']
    if kind == 'CTRL':
        rest = rest[4:] if rest.startswith('TR__') else rest
        d = RUNS / 'CONTROL' / rest / 'STAGE_B'
        rec = json.loads((d / 'STAGE_B.json').read_text())
        if V.file_sha(d / 'STAGE_B_OUTPUTS_FINAL.npz') != rec['files_sha256']['STAGE_B_OUTPUTS_FINAL.npz']:
            raise PermissionError('STAGE_B_OUTPUTS_FINAL.npz differs from STAGE_B.json')
        z = npz(d / 'STAGE_B_OUTPUTS_FINAL.npz')
        return {r: z[f'TASK_REFIT__{r}__occupation'] for r in ROLES6}, [d / 'STAGE_B_OUTPUTS_FINAL.npz', d / 'STAGE_B.json']
    if kind == 'CORR':
        arm, s = rest.split('@s')
        d = RUNS / 'CORRECT' / f'V{v}'
        rec = json.loads((d / 'RELEASES.json').read_text())
        if V.file_sha(d / 'RELEASES.npz') != rec['npz_sha256']:
            raise PermissionError('RELEASES.npz differs from RELEASES.json')
        z = npz(d / 'RELEASES.npz')
        key = (lambda r: f'W__{r}') if float(s) == 0.0 else (lambda r: f'{arm}__s{repr(float(s))}__{r}')
        return {r: z[key(r)] for r in ROLES6}, [d / 'RELEASES.npz', d / 'RELEASES.json']
    return None, []


def own_pipeline_a1(fid, X, names):
    d = RUNS / 'SIZE' / fid
    rec = json.loads((d / 'FIT.json').read_text())
    for f in ('MODEL.pkl', 'DICT.json'):
        if V.file_sha(d / f) != rec['files_sha256'][f]:
            raise PermissionError(f'{fid}/{f} differs from FIT.json')
    dic = json.loads((d / 'DICT.json').read_text())
    with open(d / 'MODEL.pkl', 'rb') as f:
        m = pickle.load(f)                                     # the saved deployed model (campaign artifact, hash-checked)
    Xr = V.apply_dictionary(X, names, dic['collapsed'])
    Zc = V.compact_po(Xr, names)
    return V.softmax(np.asarray(m.decision_function(Zc), dtype=np.float64)), {'collapsed': {b: len(x) for b, x in dic['collapsed'].items()},
                                                                               'n_iter': int(m.n_iter_)}


def _oblivious(payload, X):
    f = np.asarray(payload['split_features'], dtype=np.int64)
    b = np.asarray(payload['split_borders'], dtype=np.float32)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    T, depth = f.shape
    leaf = np.zeros((len(X32), T), dtype=np.int64)
    for i in range(depth):
        leaf |= (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) << i
    return leaf


def own_pipeline_ctrl(fid, X, names):
    """TASK_REFIT occupation of a control fit 'TR__<SID>__V<v>' (or '<SID>__V<v>'): own FIT-only dictionary application (control
    DICT.json), own oblivious routing of the T_TREE occupation teacher, raw = scale x sum leaf values + bias + sum theta / T, own
    centred softmax."""
    rest = fid[4:] if fid.startswith('TR__') else fid
    d = RUNS / 'CONTROL' / rest
    dic = json.loads((d / 'TEACHERS' / 'DICT.json').read_text())
    payload = json.loads((d / 'TEACHERS' / 'TEACHER_occupation.json').read_text())
    th = npz(d / 'STAGE_B' / 'TASK_REFIT__E20.npz')['theta_occupation']
    Xr = V.apply_dictionary(X, names, dic['collapsed'])
    leaf = _oblivious(payload, Xr)
    T = leaf.shape[1]; L = 2 ** len(payload['split_features'][0])
    lv = np.asarray(payload['leaf_values'], dtype=np.float64)
    t = np.arange(T)[None, :]
    raw = float(payload['scale']) * lv[t, leaf].sum(1) + np.asarray(payload['bias'], dtype=np.float64)
    raw = raw + np.asarray(th, dtype=np.float64).reshape(T, L, 5)[t, leaf].sum(1) / T
    return V.softmax(raw - raw.mean(1, keepdims=True)), {'T': T}


def own_pipeline_corr(v, arm, s, X, names):
    """Corrected release: W100_v by the own A1 pipeline, own routing of the S100 routing teacher (BASE DICT.json), own correction sum
    from THETA.npz, own r_s = softmax(centred log p_W + s Delta)."""
    p_w, _ = own_pipeline_a1(f'A1__S100__V{v}', X, names)
    root = RUNS / 'CORRECT' / f'V{v}'
    dic = json.loads((root / 'BASE' / 'DICT.json').read_text())
    payload = json.loads((root / 'ROUTING' / 'TEACHER_occupation.json').read_text())
    leaf = _oblivious(payload, V.apply_dictionary(X, names, dic['collapsed']))
    T = leaf.shape[1]; L = 2 ** len(payload['split_features'][0])
    th = npz(root / 'DIR' / arm / 'THETA.npz')['theta'].reshape(T, L, 5)
    delta = th[np.arange(T)[None, :], leaf].sum(1) / T
    if float(s) == 0.0:
        return p_w, {'T': T}
    lp = np.log(np.maximum(p_w, 1e-300)); lp = lp - lp.mean(1, keepdims=True)
    return V.softmax(lp + float(s) * delta), {'T': T}


def refusal(fn):
    """A refusal passes iff the call raises PermissionError or ValueError for the bad input itself; a refusal caused by a broken
    package (dependency hash differs) is recorded as NOT a pass (the positive controls must also succeed, checked by the caller)."""
    try:
        fn()
        return {'refused': False, 'error': None}
    except (PermissionError, ValueError) as e:
        msg = f'{type(e).__name__}: {str(e)[:160]}'
        if 'dependency hash differs' in str(e):
            return {'refused': False, 'error': 'BROKEN_PACKAGE ' + msg}
        return {'refused': True, 'error': msg}
    except Exception as e:                                     # any other exception is NOT counted as a registered refusal
        return {'refused': False, 'error': f'UNEXPECTED {type(e).__name__}: {str(e)[:160]}'}


def attempt(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:200]}'


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def env():
    out = {'python': sys.version.split()[0], 'executable': sys.executable}
    for p in ('numpy', 'scipy', 'scikit-learn'):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    out['threads'] = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    r = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    out['git_head'] = r.stdout.strip()
    r = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'tsz', 'ohn', 'cts', 'ihs', 'acs_replication', 'scm_bench'],
                       capture_output=True, text=True)
    out['uncommitted_code_paths'] = [l[3:] for l in r.stdout.splitlines()]
    return out


def check_package(pdir, tmp):
    from tsz import deploy as TD                                # the package's documented entry points
    from ohn import devrun as ODR
    t0 = time.process_time()
    pdir = Path(pdir)
    meta = json.loads((pdir / 'PACKAGE.json').read_text())
    name, v = meta['name'], int(meta['version'])
    res = {'package_dir': str(pdir.relative_to(ROOT)) if str(pdir).startswith(str(ROOT)) else str(pdir), 'name': name, 'version': v,
           'package_json_sha256': V.file_sha(pdir / 'PACKAGE.json')}
    dep_bad = [k for k, h in meta['depends_sha256'].items() if not resolve(k).is_file() or V.file_sha(resolve(k)) != h]
    res['dependencies'] = {'n': len(meta['depends_sha256']), 'problems': dep_bad}
    st_path = RUNS / 'STORED' / stored_name(name, v)
    stored = npz(st_path)
    res['stored_wire'] = {'path': str(st_path.relative_to(ROOT)), 'sha256': V.file_sha(st_path),
                          'in_package_dependencies': meta['depends_sha256'].get(f'TSZ::STORED/{stored_name(name, v)}') == V.file_sha(st_path),
                          'roles': sorted(k.split('__', 1)[1] for k in stored)}
    occ_cmp = {}                                               # every occupation comparison that defines `bitwise`
    # stored wire vs evaluated release
    ev, ev_files = evaluated_release(name, v)
    if ev is not None:
        for r in ROLES6:
            occ_cmp[f'stored_equals_evaluated_release|{r}'] = bool(np.array_equal(stored[f'occ__{r}'], ev[r]) and stored[f'occ__{r}'].dtype == np.float64)
        res['evaluated_release_files'] = V.inputs_sha(ev_files)
    feats, raws, rows, Xmine = {}, {}, {}, {}
    enc_equal = {}
    for r in ROLES6:
        feats[r] = C.features(r, raw=r in RAW_ROLES)
        if np.asarray(feats[r]['X']).shape[0] != stored[f'occ__{r}'].shape[0]:
            raise AssertionError(f'{r}: stored wire row count differs')
    for r in RAW_ROLES:
        raw = feats[r]['raw_inputs']
        raws[r] = raw
        n = len(feats[r]['row_id'])
        rows[r] = [{c: str(raw[c][i]) for c in V.FIELDS6} for i in range(n)]
        Xm, nm = V.encode_raw(raw, meta['encoder'])
        Xmine[r] = Xm
        enc_equal[r] = bool(nm == list(map(str, feats[r]['feature_names'])) and Xm.dtype == np.asarray(feats[r]['X']).dtype
                            and np.array_equal(Xm, np.asarray(feats[r]['X'])))
    res['own_encoder_equals_admitted_encoding'] = enc_equal
    inc_ref = {r: np.asarray(ODR.anchor_income('H3', v, r), dtype=np.float64) for r in ROLES6}
    inc_cmp, keys_ok, dec_ok, errors = {}, {}, {}, {}
    names = tuple(map(str, meta['encoder']['feature_names']))
    for r in RAW_ROLES:
        p_csv = tmp / f'{r}.csv'
        write_csv(p_csv, list(V.FIELDS6), [[rw[c] for c in V.FIELDS6] for rw in rows[r]])
        api, errors[f'wrapper_api|{r}'] = attempt(lambda: TD.release(str(pdir), rows[r]))
        viacsv, errors[f'wrapper_csv|{r}'] = attempt(lambda: TD.release_csv(str(pdir), p_csv))
        und, errors[f'underlying_own_raw_encoding|{r}'] = attempt(lambda: TD.underlying(str(pdir), Xmine[r], names))
        for lab, o in (('wrapper_api', api), ('wrapper_csv', viacsv)):
            if o is None:
                occ_cmp[f'{lab}_from_raw|{r}'] = False; inc_cmp[f'{lab}_from_raw|{r}'] = False
                keys_ok[f'{lab}|{r}'] = False; dec_ok[f'{lab}|{r}'] = False
                continue
            occ_cmp[f'{lab}_from_raw|{r}'] = bool(np.array_equal(o['occupation']['probability'], stored[f'occ__{r}']))
            inc_cmp[f'{lab}_from_raw|{r}'] = bool(np.array_equal(o['income']['probability'], inc_ref[r]))
            keys_ok[f'{lab}|{r}'] = bool(sorted(o) == ['income', 'occupation'] and all(sorted(o[t]) == ['decision', 'probability'] for t in TASKS))
            dec_ok[f'{lab}|{r}'] = bool(all(o[t]['decision'].dtype == np.int64 and np.array_equal(o[t]['decision'], o[t]['probability'].argmax(1))
                                            for t in TASKS) and np.array_equal(o['occupation']['decision'], stored[f'occ__{r}'].argmax(1)))
        if und is None:
            occ_cmp[f'underlying_own_raw_encoding|{r}'] = False; inc_cmp[f'underlying_own_raw_encoding|{r}'] = False
            keys_ok[f'underlying|{r}'] = False
        else:
            occ_cmp[f'underlying_own_raw_encoding|{r}'] = bool(np.array_equal(und['occupation'], stored[f'occ__{r}']))
            inc_cmp[f'underlying_own_raw_encoding|{r}'] = bool(np.array_equal(und['income'], inc_ref[r]))
            keys_ok[f'underlying|{r}'] = sorted(und) == ['income', 'occupation']
    for r in ROLES6:
        if r in RAW_ROLES:
            continue
        und, errors[f'underlying_admitted_encoding|{r}'] = attempt(lambda: TD.underlying(str(pdir), np.asarray(feats[r]['X']), names))
        occ_cmp[f'underlying_admitted_encoding|{r}'] = und is not None and bool(np.array_equal(und['occupation'], stored[f'occ__{r}']))
        inc_cmp[f'underlying_admitted_encoding|{r}'] = und is not None and bool(np.array_equal(und['income'], inc_ref[r]))
    res['entry_point_errors'] = {k: e for k, e in errors.items() if e}
    res['positive_controls_succeeded'] = not res['entry_point_errors']
    res['occupation_bitwise'] = occ_cmp
    res['income_bitwise_equal_admitted_H3'] = inc_cmp
    res['exported_keys_only_probability_and_decision'] = keys_ok
    res['decision_is_int64_argmax'] = dec_ok
    # own pipeline (A1 packages)
    kind, rest = name.split(':', 1)
    own = {}
    if kind in ('A1', 'CTRL', 'CORR'):
        for r in ROLES6:
            X = Xmine[r] if r in RAW_ROLES else np.asarray(feats[r]['X'])
            if kind == 'A1':
                p, info = own_pipeline_a1(rest, X, list(names))
            elif kind == 'CTRL':
                p, info = own_pipeline_ctrl(rest, X, list(names))
            else:
                arm, s = rest.split('@s')
                p, info = own_pipeline_corr(v, arm, float(s), X, list(names))
            own[r] = {'bitwise': bool(np.array_equal(p, stored[f'occ__{r}'])), 'max_abs_diff': float(np.max(np.abs(p - stored[f'occ__{r}']))),
                      'decisions_identical': bool(np.array_equal(p.argmax(1), stored[f'occ__{r}'].argmax(1))),
                      'input': 'own raw encoding' if r in RAW_ROLES else 'admitted encoding'}
        own['_model'] = info
    res['own_pipeline_vs_stored'] = own
    # refusals
    r0 = rows['DEV_REPORT'][0]; r1 = rows['DEV_REPORT'][1]
    ref = {}
    ref['wrapper_SEX_field'] = refusal(lambda: TD.release(str(pdir), [{**r0, 'SEX': '1'}]))
    ref['wrapper_sex_lowercase_field'] = refusal(lambda: TD.release(str(pdir), [{**r0, 'sex': '2'}]))
    ref['wrapper_SEX_on_second_row'] = refusal(lambda: TD.release(str(pdir), [r0, {**r1, 'SEX': '2'}]))
    ref['wrapper_target_OCCP_field'] = refusal(lambda: TD.release(str(pdir), [{**r0, 'OCCP': '0010'}]))
    ref['wrapper_target_PINCP_field'] = refusal(lambda: TD.release(str(pdir), [{**r0, 'PINCP': '50000'}]))
    ref['wrapper_extra_field'] = refusal(lambda: TD.release(str(pdir), [{**r0, 'PUMA': '00101'}]))
    ref['wrapper_missing_field'] = refusal(lambda: TD.release(str(pdir), [{k: x for k, x in r0.items() if k != 'POBP'}]))
    ref['wrapper_reordered_fields'] = refusal(lambda: TD.release(str(pdir), [dict(reversed(list(r0.items())))]))
    ref['wrapper_hidden_output_raw_score'] = refusal(lambda: TD.release(str(pdir), [r0], requested=('probability', 'raw_score')))
    ref['wrapper_hidden_output_leaf_ids'] = refusal(lambda: TD.release(str(pdir), [r0], requested=('decision', 'leaf_ids')))
    base = [r0[c] for c in V.FIELDS6]
    for lab, hdr, vals in (('csv_SEX_column', list(V.FIELDS6) + ['SEX'], base + ['2']),
                           ('csv_target_OCCP_column', list(V.FIELDS6) + ['OCCP'], base + ['0010']),
                           ('csv_target_PINCP_column', list(V.FIELDS6) + ['PINCP'], base + ['50000']),
                           ('csv_extra_column', list(V.FIELDS6) + ['PUMA'], base + ['00101']),
                           ('csv_missing_column', list(V.FIELDS6)[:-1], base[:-1]),
                           ('csv_reordered_columns', list(V.FIELDS6)[::-1], base[::-1])):
        q = tmp / f'{lab}.csv'
        write_csv(q, hdr, [vals])
        ref[lab] = refusal(lambda q=q: TD.release_csv(str(pdir), q))
    q = tmp / 'ok_one_row.csv'
    write_csv(q, list(V.FIELDS6), [base])
    ref['csv_hidden_output_request'] = refusal(lambda: TD.release_csv(str(pdir), q, requested=('probability', 'logits')))
    X3 = np.asarray(feats['DEV_REPORT']['X'][:3]); nl = list(names)
    ref['underlying_SEX_column'] = refusal(lambda: TD.underlying(str(pdir), np.column_stack((X3, np.ones(3, np.float32))), nl + ['SEX']))
    ref['underlying_target_OCCP_column'] = refusal(lambda: TD.underlying(str(pdir), np.column_stack((X3, np.ones(3, np.float32))), nl + ['OCCP']))
    ref['underlying_target_PINCP_column'] = refusal(lambda: TD.underlying(str(pdir), np.column_stack((X3, np.ones(3, np.float32))), nl + ['PINCP']))
    ref['underlying_extra_column'] = refusal(lambda: TD.underlying(str(pdir), np.column_stack((X3, np.ones(3, np.float32))), nl + ['EXTRA']))
    ref['underlying_SEX_in_place_of_AGEP'] = refusal(lambda: TD.underlying(str(pdir), X3, ['SEX'] + nl[1:]))
    ref['underlying_missing_column'] = refusal(lambda: TD.underlying(str(pdir), X3[:, 1:], nl[1:]))
    ref['underlying_reordered_columns'] = refusal(lambda: TD.underlying(str(pdir), X3[:, ::-1], nl[::-1]))
    res['refusals'] = ref
    res['bitwise'] = bool(occ_cmp and all(occ_cmp.values()))
    # a refusal set passes only if every bad input raised AND the same entry points accepted the valid inputs (positive controls)
    res['refusals_all_pass'] = bool(all(x['refused'] for x in ref.values()) and res['positive_controls_succeeded'])
    res['income_all_bitwise'] = bool(inc_cmp and all(inc_cmp.values()))
    res['own_pipeline_all_bitwise'] = bool(own and all(x['bitwise'] for k, x in own.items() if not k.startswith('_')))
    res['own_pipeline_within_1e-12'] = bool(own and all(x['bitwise'] or (x['max_abs_diff'] <= 1e-12 and x['decisions_identical'])
                                                         for k, x in own.items() if not k.startswith('_')))
    res['bitwise_definition'] = ('occupation probability arrays: wrapper api and wrapper csv from the raw fields and the underlying function on '
                                 'the verifier\'s own raw encoding (DEV_REPORT, MODEL_FIT: the roles with admitted raw inputs), the underlying '
                                 'function on the admitted encoding (CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION: no admitted raw '
                                 'inputs; encoder parity is shown on the raw roles), and stored wire == evaluated release on all six roles')
    res['cpu_s'] = round(time.process_time() - t0, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packages', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--w100', action='store_true')
    ap.add_argument('--dry', action='store_true', help='code test only: print, write no record')
    a = ap.parse_args()
    if a.dry:
        tmp = Path(tempfile.mkdtemp(prefix='verifier_deploy_dry_', dir=PRIV))
        try:
            for x in a.packages.split(','):
                r = check_package(Path(x) if os.path.isabs(x) else ROOT / x, tmp)
                print(json.dumps({k: r[k] for k in ('name', 'bitwise', 'refusals_all_pass', 'income_all_bitwise', 'own_pipeline_all_bitwise',
                                                    'entry_point_errors', 'dependencies', 'own_encoder_equals_admitted_encoding', 'cpu_s')},
                                 default=str)[:3000])
                print(json.dumps({k: x['refused'] for k, x in r['refusals'].items()}))
                print(json.dumps(r['own_pipeline_vs_stored'], default=str)[:1500])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    pk = [Path(x) if os.path.isabs(x) else ROOT / x for x in a.packages.split(',') if x]
    tmp = Path(tempfile.mkdtemp(prefix='verifier_deploy_', dir=PRIV))
    try:
        results = {}
        for p in pk:
            results[p.name] = check_package(p, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    e = env()
    code = V.code_hashes(__file__, V.__file__)
    code['tsz/deploy.py'] = V.file_sha(ROOT / 'tsz' / 'deploy.py')
    code['tsz/sizefit.py'] = V.file_sha(ROOT / 'tsz' / 'sizefit.py')
    checks = []
    for nm, r in results.items():
        ok = (r['bitwise'] and r['refusals_all_pass'] and not r['dependencies']['problems'] and all(r['own_encoder_equals_admitted_encoding'].values())
              and all(r['exported_keys_only_probability_and_decision'].values()) and all(r['decision_is_int64_argmax'].values())
              and r['stored_wire']['in_package_dependencies'])
        attention = ok and not (r['income_all_bitwise'] and (r['own_pipeline_within_1e-12'] or not r['own_pipeline_vs_stored']))
        checks.append({'id': f'V1.DEPLOY.{nm}__{a.tag}' if a.w100 else f'V6.DEPLOY.{nm}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
                       'what': f'deployment of package {nm} ({r["name"]}, version {r["version"]}) from the six raw fields: dependency hashes, own '
                               'encoder == admitted encoding, stored wire == evaluated release, wrapper api / csv and underlying outputs == stored '
                               'wire bitwise on all six roles, income == admitted H3 wire, exported keys, own pipeline replay, and every '
                               'registered refusal at the wrapper (api, csv) and the underlying function',
                       'inputs': {r['package_dir'] + '/PACKAGE.json': r['package_json_sha256'], r['stored_wire']['path']: r['stored_wire']['sha256'],
                                  **r.get('evaluated_release_files', {})},
                       'expected': 'bitwise true, refusals_all_pass true, no dependency problem, encoder parity, only probability and decision exported',
                       'observed': {k: r[k] for k in ('bitwise', 'refusals_all_pass', 'income_all_bitwise', 'own_pipeline_all_bitwise',
                                                       'own_pipeline_within_1e-12', 'dependencies', 'own_encoder_equals_admitted_encoding',
                                                       'entry_point_errors')},
                       'status': 'PASS' if ok and not attention else ('ATTENTION' if ok else 'FAIL'),
                       'code_sha256': code, 'labels_read': False})
    rec = {'schema': 'TSZ1-verifier-deployment-v1', 'utc': V.utc(), 'labels_read': False, 'packages': results, 'environment': e,
           'code_sha256': code, 'checks': [c['id'] for c in checks]}
    V.write_record(f'V_DEPLOY__{a.tag}.json', rec)
    if a.w100:
        out = {'schema': 'TSZ1-verifier-DEPLOY_W100-v1', 'utc': V.utc(), 'labels_read': False,
               'what': 'independent verifier deployment record for the trigger part T3 (RULES.md section 7): per version, `bitwise` = the '
                       'package computed from the six raw fields reproduces the stored W100_v occupation release bitwise on every role '
                       '(definition in `bitwise_definition`), `refusals_all_pass` = every registered refusal check raised',
               'record': f'records/V_DEPLOY__{a.tag}.json', 'check_ids': [c['id'] for c in checks], 'environment': e, 'code_sha256': code}
        for nm, r in results.items():
            out[str(r['version'])] = {'bitwise': r['bitwise'], 'refusals_all_pass': r['refusals_all_pass'], 'name': r['name'],
                                      'entry_point_errors': r['entry_point_errors'],
                                      'package_dir': r['package_dir'], 'package_json_sha256': r['package_json_sha256'],
                                      'stored_wire_sha256': r['stored_wire']['sha256'], 'income_all_bitwise': r['income_all_bitwise'],
                                      'own_pipeline_all_bitwise': r['own_pipeline_all_bitwise'],
                                      'dependency_problems': r['dependencies']['problems'],
                                      'own_encoder_equals_admitted_encoding': r['own_encoder_equals_admitted_encoding'],
                                      'occupation_bitwise': r['occupation_bitwise'], 'refusals': r['refusals'],
                                      'bitwise_definition': r['bitwise_definition']}
        if set(k for k in out if k in ('0', '1')) != {'0', '1'}:
            raise SystemExit('DEPLOY_W100 needs exactly the two versions 0 and 1')
        if DEPLOY_W100.is_file():
            V.HISTORY.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEPLOY_W100, V.HISTORY / f'DEPLOY_W100__replaced_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json')
        DEPLOY_W100.write_text(json.dumps(out, indent=1, sort_keys=True, default=V._json_default) + '\n')
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: [c['status'], c['observed']['bitwise'], c['observed']['refusals_all_pass'], c['observed']['income_all_bitwise'],
                                c['observed']['own_pipeline_all_bitwise']] for c in checks}, indent=1))


if __name__ == '__main__':
    main()

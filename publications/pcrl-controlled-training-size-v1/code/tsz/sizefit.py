"""A1_PO_16 on a registered training subset (controlled training-size test, PCRL `pcrl_controlled_training_size_v1`; EXPLORATORY
DEVELOPMENT on previously used records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records (housing-unit residents aged 25-64,
civilian employed). The occupation recipient (5 official broad civilian groups) receives its own float64 probability vector and
argmax decision, computed from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP) through the admitted 145-column encoding. This
module is TASK-ONLY: it reads MODEL_FIT occupation labels of one registered subset and CALIBRATION occupation labels (diagnostic
temperature, recorded, never applied); never SEX, never income, never a MODEL_SELECT / ATTACK / DEV_REPORT label.

Recipe (ohn.fam_a.CONFIGS['A1_PO_16'] and the source fit, reused by import; DESIGN_BRIEF section 3):
  dictionary  FIT-only: cts.trackf.fit_dictionary on the subset's X (a COW / SCHL / MAR / POBP level with < 10 subset people moves to
              its block's __OTHER__ column), applied to every role with cts.trackf.recode; encoding sensitivity (fit id suffix
              __ADM, S50(o) and v = 0 only): cts.trackf.admitted_dictionary (no level re-collapsed).
  coding      ohn.fam_a.rep_matrix('cmp_po', ...) on the recoded X.
  T*          ohn.fam_a.cv_select over ohn.common.household_folds(ALL MODEL_FIT units, 5, 'OHN_A_cv_S<v>') restricted to the subset
              rows (the source's sub-pipeline convention, ohn.fam_a.run_pipeline); first argmin of the pooled inner-CV NLL.
  deployed    ohn.fam_a.fit_model(cfg, Xr, y, T*, v) on the subset; release softmax(raw) on all six roles from features alone; no
              calibrator applied; ohn.fam_a.fit_temperature on the deployed raw CALIBRATION scores recorded only.
  support     fixed, no repair: all five occupation classes must be present in the subset and in every inner training set; otherwise
              SUPPORT_FAILURE.json is written (terminal: the cell is reported missing, never redrawn) and the unit exits with code 3.
At S100 every step equals the source fit (the FIT-only dictionary of all MODEL_FIT equals the admitted dictionary), so the S100
release must reproduce the source A1_PO_16 S<v> release bitwise (`repro`).

Commands (real-data ones only through `python3 -m tsz.jobqueue run ...`):
  selftest                                              invented data only
  timing                                                invented data of the real size (label-free; CPU estimate)
  fit --subset SID --version V [--dictionary fit_only|admitted]
  verify --fit-id ID                                    label-free: predict() from features alone == saved release (bitwise, six
                                                        roles) + input refusals
  repro --version V                                     label-free: S100 fit of version v vs the source A1_PO_16 S<v> release
Outputs: .private/<campaign>/runs/SIZE/<fit id>/ (write-once): MODEL.pkl, RELEASE.npz (key = role), DICT.json, CURVE.npz, FIT.json.
`predict(fit_id, X, names)` recomputes a saved fit's occupation release from admitted features alone.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C

RECIPE = 'A1_PO_16'
STAGE_DIR = 'SIZE'
SCHEMA = 'TSZ1-sizefit-v1'
K_OCC = 5
ROLES = C.ROLES6
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
DICT_KINDS = ('fit_only', 'admitted')
N_INNER = 5
SUPPORT_EXIT = 3
DEVIATIONS = (
    'none at S100 (the source fit procedure; the FIT-only dictionary of all MODEL_FIT collapses no level, so recode() is the identity '
    'on every role, checked label-free and in `repro`)',
    'subsets: the dictionary is re-collapsed on the subset (DESIGN_BRIEF section 3); inner folds are the source seed-specific '
    'household folds of ALL MODEL_FIT units restricted to the subset rows (the source sub-pipeline convention), so the inner folds '
    'of a subset are unbalanced by up to the subset fraction',
    'no household cross-fitting here (the source fit also ran 5 cross-fitting pipelines for its out-of-fold diagnostics; none of them '
    'enters the deployed release); the conditional correction stage owns the cross-fitting base of W100',
    'CALIBRATION: only the full-CALIBRATION diagnostic temperature (ohn.fam_a.fit_temperature) is recorded; the source also recorded a '
    'household cross-fitted temperature effect and an OOF temperature, which need the cross-fitting arrays',
)


class SupportFailure(RuntimeError):
    """An occupation class is absent from the subset or from an inner training set (fixed rule: no repair, cell missing)."""


# ------------------------------------------------------------------ data access (hooks; the selftest replaces them with fakes)
_NO_REAL_ACCESS = False          # set by selftest / timing: the real label hook refuses (invented data only)


def _load(role, fields, operation, purpose, subset=None):
    if _NO_REAL_ACCESS:
        raise PermissionError('selftest / timing: no real label read')
    return C.load(role, fields, operation, purpose, subset=subset)


def _features(role):
    return C.features(role)


def _source_release(v):
    """The source A1_PO_16 S<v> release (ohn.devrun.load_occupation: OCC.npz checked against OCC.json) and its record files."""
    from ohn import devrun as OD
    occ, rec, cf = OD.load_occupation('A', RECIPE, v)
    d = OD.occ_dir('A', RECIPE, v)
    files = {n: C.sha(d / n) for n in ('OCC.npz', 'OCC.json', 'STATE.json', 'MODEL.pkl') if (d / n).is_file()}
    return occ, rec, files


# ------------------------------------------------------------------ identifiers and paths
def fit_id(sid, v, dictionary='fit_only'):
    if sid not in C.subset_ids():
        raise ValueError(f'unknown subset {sid}')
    if int(v) not in C.VERSIONS:
        raise ValueError(f'unknown version {v}')
    if dictionary == 'admitted':
        if not (int(v) == 0 and sid.startswith('S50_')):
            raise ValueError('the encoding sensitivity is registered for S50(o), version 0 only')
        return f'A1__{sid}__V{int(v)}__ADM'
    if dictionary != 'fit_only':
        raise ValueError(f'unknown dictionary kind {dictionary}')
    return f'A1__{sid}__V{int(v)}'


def parse_fit_id(fid):
    parts = fid.split('__')
    if parts[0] != 'A1' or len(parts) not in (3, 4) or not parts[2].startswith('V'):
        raise ValueError(f'not a size-fit id: {fid}')
    sid, v = parts[1], int(parts[2][1:])
    kind = 'admitted' if len(parts) == 4 and parts[3] == 'ADM' else 'fit_only'
    if len(parts) == 4 and parts[3] != 'ADM':
        raise ValueError(f'not a size-fit id: {fid}')
    if fit_id(sid, v, kind) != fid:
        raise ValueError(f'not a size-fit id: {fid}')
    return sid, v, kind


def fit_dir(fid, root=None):
    return (Path(root) if root else C.RUNS / STAGE_DIR) / fid


def _fresh_dir(d, record_name):
    """Write-once fit directory: a completed fit (record present) or a recorded support failure is never redone; an incomplete
    attempt is moved aside (kept, named) so that a retry unit starts from an empty directory."""
    d = Path(d)
    if (d / record_name).exists() or (d / 'SUPPORT_FAILURE.json').exists():
        raise PermissionError(f'{d.name} is complete or recorded missing (append-only)')
    moved = None
    if d.exists():
        moved = d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}')
        d.rename(moved)
    d.mkdir(parents=True, mode=0o700)
    return None if moved is None else moved.name


def _save_npz(path, **arrs):
    path = Path(path)
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return C.sha(path)


def _write_once(path, obj):
    C.write_json_x(path, obj)
    return C.sha(path)


def _interp():
    import importlib.metadata as m
    out = {'python': sys.version.split()[0], 'executable': sys.executable}
    for p in ('numpy', 'scipy', 'scikit-learn'):
        try:
            out[p] = m.version(p)
        except Exception:
            out[p] = None
    return out


# ------------------------------------------------------------------ dictionary and coding (shared with tsz.control)
def subset_dictionary(X_sub, names, kind='fit_only'):
    """The training subset's feature dictionary: FIT-only re-collapse (primary) or the admitted dictionary (encoding sensitivity)."""
    from cts import trackf as TF
    if kind == 'fit_only':
        return TF.fit_dictionary(X_sub, tuple(map(str, names)))
    if kind == 'admitted':
        return TF.admitted_dictionary(tuple(map(str, names)))
    raise ValueError(kind)


def recode(X, names, dic):
    from cts import trackf as TF
    return TF.recode(X, tuple(map(str, names)), dic)


def rep(X, names, dic):
    from ohn import fam_a as FA
    names = tuple(map(str, names))
    return FA.rep_matrix(FA.CONFIGS[RECIPE]['rep'], recode(X, names, dic), names)


def dictionary_diagnostics(X_sub, X_ms, names, dic):
    """Per block: collapsed levels; levels absent from the subset; MODEL_SELECT share with a collapsed level and with a level absent
    from the subset ("unknown"); label-free."""
    names = tuple(map(str, names)); X_sub = np.asarray(X_sub); X_ms = np.asarray(X_ms)
    out, any_c, any_u = {}, np.zeros(len(X_ms), dtype=bool), np.zeros(len(X_ms), dtype=bool)
    for b in BLOCKS:
        cols = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        cnt = X_sub[:, cols].sum(0)
        act = X_ms[:, cols].argmax(1)
        coll = np.array([names[c] in set(dic['collapsed'][b]) for c in cols])
        unk = cnt == 0
        any_c |= coll[act]; any_u |= unk[act]
        out[b] = {'levels': len(cols), 'n_collapsed': int(coll.sum()), 'levels_absent_from_subset': int(unk.sum()),
                  'MODEL_SELECT_people_collapsed': int(coll[act].sum()), 'MODEL_SELECT_share_collapsed': float(coll[act].mean()),
                  'MODEL_SELECT_people_unknown': int(unk[act].sum()), 'MODEL_SELECT_share_unknown': float(unk[act].mean())}
    out['any_block'] = {'MODEL_SELECT_share_collapsed': float(any_c.mean()), 'MODEL_SELECT_share_unknown': float(any_u.mean())}
    return out


def class_support(y, inner):
    """Occupation class counts in the subset and in every inner training set (fixed rule: all five classes everywhere)."""
    y = np.asarray(y, dtype=np.int64)
    sub = np.bincount(y, minlength=K_OCC)
    per = {}
    for j in np.unique(inner):
        per[str(int(j))] = [int(c) for c in np.bincount(y[inner != j], minlength=K_OCC)]
    ok = bool(len(sub) == K_OCC and (sub > 0).all() and all(min(c) > 0 for c in per.values()) and y.min() >= 0 and y.max() < K_OCC)
    return {'ok': ok, 'subset_class_counts': [int(c) for c in sub], 'subset_min_class_count': int(sub.min()),
            'inner_training_class_counts': per, 'inner_training_min_class_count': int(min(min(c) for c in per.values()))}


# ------------------------------------------------------------------ fit
def fit(sid, v, dictionary='fit_only', root=None, cfg_override=None, log=print):
    from ohn import fam_a as FA
    from ohn import common as OC
    t0 = time.process_time(); w0 = time.time()
    v = int(v)
    fid = fit_id(sid, v, dictionary)
    if cfg_override is not None and root is None:
        raise PermissionError('configuration overrides are for the invented-data selftest only')
    cfg = dict(FA.CONFIGS[RECIPE]) if cfg_override is None else {**FA.CONFIGS[RECIPE], **cfg_override}
    d = fit_dir(fid, root)
    if (d / 'FIT.json').exists() or (d / 'SUPPORT_FAILURE.json').exists():
        raise PermissionError(f'{fid} is complete or recorded missing (append-only)')
    F = _load('MODEL_FIT', ('features', 'occupation'), 'fit',
              f'TSZ1 size fit {fid}: A1_PO_16 occupation task model on registered subset {sid} (inner household CV over the source '
              f'folds OHN_A_cv_S{v} restricted to the subset, refit at T*); no SEX, no income', subset=sid)
    names = tuple(map(str, F['feature_names'])); y = np.asarray(F['occupation'], dtype=np.int64); unit = F['unit']; rows = F['rows']
    Fall = _features('MODEL_FIT')
    if not (np.array_equal(Fall['unit'][rows], unit) and np.array_equal(Fall['row_id'][rows], F['row_id'])):
        raise AssertionError('subset rows differ from the MODEL_FIT feature rows')
    inner = OC.household_folds(Fall['unit'], N_INNER, FA.inner_tag(v))[rows]
    support = class_support(y, inner)
    if not support['ok']:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        rec = {'schema': SCHEMA + '-support-failure', 'fit_id': fid, 'subset': sid, 'version': v, 'support': support,
               'rule': 'fixed before drawing: no repair; the cell is reported missing; no redraw', 'utc': C.now(),
               'unit_id': os.environ.get(C.UNIT_ENV)}
        _write_once(d / 'SUPPORT_FAILURE.json', rec)
        raise SupportFailure(json.dumps(support))
    dic = subset_dictionary(F['X'], names, dictionary)
    feats = {r: _features(r) for r in ROLES}
    for r in ROLES:
        if tuple(map(str, feats[r]['feature_names'])) != names:
            raise AssertionError(f'{r}: feature schema differs')
    Xr = rep(F['X'], names, dic)
    allr = np.ones(len(y), dtype=bool)
    T_star, curve, inner_models = FA.cv_select(cfg, v, Xr, y, inner, allr)
    dep = FA.fit_model(cfg, Xr, y, T_star, v)
    reps = {r: rep(feats[r]['X'], names, dic) for r in ROLES}
    rel = {r: FA.release_prob([dep], reps[r]) for r in ROLES}
    Cal = _load('CALIBRATION', ('features', 'occupation'), 'calibrate',
                f'TSZ1 size fit {fid}: A1_PO_16 diagnostic temperature on CALIBRATION occupation (recorded, never applied, as in the source)')
    if not np.array_equal(Cal['row_id'], feats['CALIBRATION']['row_id']):
        raise AssertionError('CALIBRATION row order differs')
    tau, tinfo = FA.fit_temperature(FA.raw_full(dep, reps['CALIBRATION']), Cal['occupation'])
    log(json.dumps({'fit_id': fid, 'T_star': T_star, 'cpu_s': round(time.process_time() - t0, 1)}))
    incomplete = _fresh_dir(d, 'FIT.json')
    files = {}
    buf = io.BytesIO(); pickle.dump(dep, buf, protocol=5)
    with open(d / 'MODEL.pkl', 'xb') as f:
        f.write(buf.getvalue())
    files['MODEL.pkl'] = C.sha(d / 'MODEL.pkl')
    files['RELEASE.npz'] = _save_npz(d / 'RELEASE.npz', **rel)
    files['CURVE.npz'] = _save_npz(d / 'CURVE.npz', curve=curve)
    files['DICT.json'] = _write_once(d / 'DICT.json', dic)
    flags = []
    if T_star >= int(cfg['max_iter']):
        flags.append('T_STAR_AT_T_MAX')
    inner_sizes = {str(int(j)): {'people': int((inner == j).sum()), 'households': int(len(np.unique(unit[inner == j])))}
                   for j in np.unique(inner)}
    rec = {'schema': SCHEMA, 'fit_id': fid, 'subset': sid, 'version': v, 'dictionary_kind': dictionary, 'recipe': RECIPE,
           'config': cfg, 'config_override': cfg_override, 'aliases': list(C.ORDERINGS) if sid == 'S100' else [sid.split('_')[1]],
           'T_star': int(T_star), 'T_max': int(cfg['max_iter']), 'flags': flags,
           'inner_cv': {'tag': FA.inner_tag(v), 'folds_over': 'all MODEL_FIT units, restricted to the subset rows', 'k': N_INNER,
                        'fold_sizes': inner_sizes, 'n_inner_models': len(inner_models)},
           'curve_summary': FA._curve_summary(curve),
           'curve_at_T_star': {'nll': float(curve[0, T_star - 1]), 'brier': float(curve[1, T_star - 1]),
                               'accuracy': float(curve[2, T_star - 1]), 'nll_at_T_max': float(curve[0, -1])},
           'n_people': int(len(y)), 'n_households': int(len(np.unique(unit))), 'rows_sha256': C.arr_sha(np.asarray(F['row_id'])),
           'class_support': support,
           'dictionary': {'kind': dictionary, 'min_count': dic['min_count'], 'n_collapsed': dic['n_collapsed'],
                          'diagnostics': dictionary_diagnostics(F['X'], feats['MODEL_SELECT']['X'], names, dic)},
           'temperature_diagnostic_CALIBRATION': {**tinfo, 'applied': False},
           'release_sha256': {r: C.arr_sha(rel[r]) for r in ROLES},
           'role_summary': {r: {'n': int(len(p)), 'mean_prob': [float(x) for x in p.mean(0)],
                                'decision_share': [float(x) for x in np.bincount(p.argmax(1), minlength=K_OCC) / len(p)]}
                            for r, p in rel.items()},
           'row_id_sha256': {r: C.arr_sha(np.asarray(feats[r]['row_id'])) for r in ROLES},
           'feature_names': list(names), 'files_sha256': files, 'previous_incomplete_attempt': incomplete,
           'n_model_fits': len(inner_models) + 1, 'n_calibrator_fits': 1,
           'labels_read': f'MODEL_FIT {sid}: occupation (fit); CALIBRATION: occupation (diagnostic temperature only); no SEX, no income',
           'deviations': list(DEVIATIONS), 'code_sha256': {'tsz/sizefit.py': C.sha(Path(__file__)),
                                                           'ohn/fam_a.py': C.sha(Path(FA.__file__))},
           'interpreter': _interp(), 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'FIT.json', rec)
    C.unit_stats(n_model_fits=rec['n_model_fits'], calibrator_fits=1)
    log(json.dumps({'fit_id': fid, 'T_star': T_star, 'n_people': rec['n_people'], 'min_class': support['subset_min_class_count'],
                    'n_collapsed': dic['n_collapsed'], 'tau_cal': tinfo['tau'], 'cpu_s': round(rec['cpu_s'], 1)}))
    return rec


# ------------------------------------------------------------------ deployment
def load_fit(fid, root=None):
    d = fit_dir(fid, root)
    rec = json.loads((d / 'FIT.json').read_text())
    for n in ('MODEL.pkl', 'DICT.json', 'RELEASE.npz'):
        if C.sha(d / n) != rec['files_sha256'][n]:
            raise PermissionError(f'{fid}/{n} differs from its record')
    dic = json.loads((d / 'DICT.json').read_text())
    with open(d / 'MODEL.pkl', 'rb') as f:
        m = pickle.load(f)
    return m, dic, rec


def release(fid, root=None):
    d = fit_dir(fid, root)
    rec = json.loads((d / 'FIT.json').read_text())
    if C.sha(d / 'RELEASE.npz') != rec['files_sha256']['RELEASE.npz']:
        raise PermissionError(f'{fid}/RELEASE.npz differs from its record')
    return C.load_npz(d / 'RELEASE.npz'), rec


def _check_inputs(X, names, expected):
    if tuple(map(str, names)) != tuple(expected):
        raise ValueError('wrong or reordered input schema')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(expected) or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ValueError('finite input matrix of the registered width required')
    return X


def predict(fid, X, names, root=None):
    """(n, 5) float64 occupation release of a saved size fit, recomputed from admitted features alone (deterministic)."""
    from ohn import fam_a as FA
    m, dic, rec = load_fit(fid, root)
    X = _check_inputs(X, names, rec['feature_names'])
    return FA.release_prob([m], rep(X, names, dic))


def _refusal_probes(fid, X, names, root=None):
    names = tuple(map(str, names)); X = np.asarray(X)
    i_cow = [i for i, n in enumerate(names) if n.startswith('COW=')]
    probes = {}
    sw = list(names); sw[i_cow[0]], sw[i_cow[1]] = sw[i_cow[1]], sw[i_cow[0]]
    bad_block = X[:3].copy(); bad_block[0, i_cow] = 0.0
    nan = X[:3].astype(np.float64).copy(); nan[0, 0] = np.nan
    cases = {'reordered_names': (X[:3], tuple(sw)), 'incomplete_one_hot_block': (bad_block, names), 'non_finite': (nan, names),
             'wrong_width': (X[:3, :-1], names[:-1]), 'integer_matrix': (X[:3].astype(np.int64), names)}
    for k, (Xp, nm) in cases.items():
        try:
            predict(fid, Xp, nm, root); probes[k] = {'refused': False}
        except (ValueError, KeyError) as e:
            probes[k] = {'refused': True, 'error': repr(e)[:160]}
    return probes


def verify(fid, root=None):
    parse_fit_id(fid)
    saved, rec = release(fid, root)
    res = {}
    for r in ROLES:
        Fr = _features(r)
        if C.arr_sha(np.asarray(Fr['row_id'])) != rec['row_id_sha256'][r]:
            raise AssertionError(f'{r}: row order differs from the saved release')
        p = predict(fid, Fr['X'], Fr['feature_names'], root)
        res[r] = {'bitwise': bool(np.array_equal(p, saved[r])), 'max_abs_diff': float(np.max(np.abs(p - saved[r]))),
                  'decisions_identical': bool(np.array_equal(p.argmax(1), saved[r].argmax(1)))}
    Fm = _features('MODEL_SELECT')
    probes = _refusal_probes(fid, Fm['X'], Fm['feature_names'], root)
    ok = all(v['bitwise'] for v in res.values()) and all(v['refused'] for v in probes.values())
    out = {'schema': SCHEMA + '-verify', 'fit_id': fid, 'ok': ok, 'roles': res, 'refusals': probes, 'files_sha256': rec['files_sha256'],
           'interpreter': _interp(), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(fit_dir(fid, root) / 'VERIFY.json', out)
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'fit_id': fid, 'roles': {r: v['bitwise'] for r, v in res.items()},
                      'refusals': {k: v['refused'] for k, v in probes.items()}}))
    return ok


def repro(v, root=None):
    """S100 fit of version v against the source A1_PO_16 S<v> release (label-free; every difference is reported, never tuned)."""
    v = int(v)
    fid = fit_id('S100', v)
    mine, rec = release(fid, root)
    src, srec, sfiles = _source_release(v)
    roles = {}
    for r in ROLES:
        a, b = mine[r], src[r]
        same_shape = a.shape == b.shape
        roles[r] = {'bitwise': bool(same_shape and np.array_equal(a, b)),
                    'max_abs_diff': float(np.max(np.abs(a - b))) if same_shape else None,
                    'decision_agreement': float((a.argmax(1) == b.argmax(1)).mean()) if same_shape else None,
                    'mine_sha256': C.arr_sha(a), 'source_sha256': C.arr_sha(b)}
    t_src = srec.get('T_star')
    tau_src = (srec.get('temperature_diagnostics') or {}).get('CALIBRATION_full', {}).get('tau')
    tau_mine = rec['temperature_diagnostic_CALIBRATION']['tau']
    curve_src = srec.get('curve_main_oof')
    out = {'schema': SCHEMA + '-repro', 'version': v, 'fit_id': fid, 'roles': roles,
           'all_roles_bitwise': all(x['bitwise'] for x in roles.values()),
           'T_star': {'mine': rec['T_star'], 'source': t_src, 'equal': rec['T_star'] == t_src},
           'inner_curve_summary_equal': (curve_src == rec['curve_summary']) if curve_src is not None else None,
           'diagnostic_temperature': {'mine': tau_mine, 'source': tau_src, 'equal': tau_mine == tau_src,
                                      'abs_diff': None if tau_src is None else abs(tau_mine - tau_src)},
           'dictionary_n_collapsed': rec['dictionary']['n_collapsed'],
           'model_pkl_sha256': {'mine': rec['files_sha256']['MODEL.pkl'], 'source': sfiles.get('MODEL.pkl'),
                                'equal': rec['files_sha256']['MODEL.pkl'] == sfiles.get('MODEL.pkl')},
           'source_files_sha256': sfiles, 'source_record': {k: srec.get(k) for k in ('unit_id', 'utc', 'sklearn', 'numpy', 'python', 'occ_sha256')},
           'mine_interpreter': rec['interpreter'],
           'not_compared': 'the source MODEL_FIT out-of-fold array (CROSSFIT.npz) comes from 5 cross-fitting pipelines that this fit '
                           'does not run (they never enter the deployed release)',
           'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    out['reproduced'] = bool(out['all_roles_bitwise'] and out['T_star']['equal'])
    adm_p = C.RESULTS / 'SOURCE_ADMISSION.json'
    if root is None and adm_p.is_file():          # tsz.admit record of the source release this comparison used
        a = json.loads(adm_p.read_text()).get('releases_used', {}).get(f'A/{RECIPE}/S{v}', {})
        out['source_admission_check'] = {'occ_npz_sha256_equal': a.get('occ_npz_sha256') == sfiles.get('OCC.npz'),
                                         'T_star_equal': a.get('T_star') == t_src, 'admission_sha256': C.sha(adm_p)}
    else:
        out['source_admission_check'] = None
    _write_once(fit_dir(fid, root) / 'REPRO.json', out)
    print(json.dumps({'repro': 'REPRODUCED' if out['reproduced'] else 'NOT_REPRODUCED', 'version': v,
                      'roles_bitwise': {r: x['bitwise'] for r, x in roles.items()}, 'T_star': out['T_star'],
                      'tau': out['diagnostic_temperature']}, default=str))
    return out


# ------------------------------------------------------------------ invented data (selftest, timing)
def _names():
    from ohn import common as OC
    return tuple(OC.teacher(0, 'occupation').feature_names)        # schema only (stored teacher payload; no person data)


def _fake_world(rng, names, sizes, rare_count=14):
    """Invented roles: every block uses a few common levels plus 1-2 rare levels with exactly `rare_count` MODEL_FIT people (kept at
    S100, collapsed in most halves); all other admitted levels are unused, so at S100 recode() is the identity on every role."""
    from ohn import fam_a as FA
    plan = {}
    for b, (nc, nr) in {'COW': (4, 1), 'SCHL': (5, 1), 'MAR': (3, 0), 'POBP': (5, 2)}.items():
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=') and not n.endswith('__OTHER__')]
        pick = rng.choice(len(idx), nc + nr, replace=False)
        plan[b] = ([idx[i] for i in pick[:nc]], [idx[i] for i in pick[nc:]])
    world, start = {}, 0
    for role, n in sizes.items():
        X = np.zeros((n, len(names)), dtype=np.float32)
        X[:, 0] = np.round(rng.normal(size=n) * 8) / 8; X[:, 1] = np.round(rng.normal(size=n) * 8) / 8
        for b, (common, rare) in plan.items():
            col = np.asarray(common)[rng.choice(len(common), n, p=rng.dirichlet(np.ones(len(common)) * 2))]
            if role == 'MODEL_FIT':
                pos = rng.choice(n, rare_count * len(rare), replace=False)
                for k, c in enumerate(rare):
                    col[pos[k * rare_count:(k + 1) * rare_count]] = c
            elif rare:
                m = rng.random(n) < 0.01
                col[m] = np.asarray(rare)[rng.integers(0, len(rare), m.sum())]
            X[np.arange(n), col] = 1.0
        hh = rng.integers(1, 4, size=n)
        unit = np.repeat(np.arange(start, start + n), hh)[:n]; start += n
        Xc = FA.compact_po(X, names)
        s = np.column_stack([0.9 * Xc[:, 1] + 0.1 * (Xc[:, 3] - 16), -0.7 * Xc[:, 1] + 0.3 * (Xc[:, 2] % 3), 0.4 * Xc[:, 0],
                             -0.12 * (Xc[:, 3] - 16) + 0.4 * (Xc[:, 5] < 60), 0.4 * (Xc[:, 4] == Xc[:, 4].min()) - 0.3 * Xc[:, 0] * Xc[:, 1]])
        p = FA.softmax(s)
        occ = (rng.random(n)[:, None] > np.cumsum(p, 1)).sum(1).astype(np.int64)
        inc = (rng.random(n) < 0.4).astype(np.int64)
        sex = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 * Xc[:, 1] + 1.0 * (occ == 3) - 1.0 * (occ == 0) + 0.5 * inc)))).astype(np.int64)
        world[role] = {'X': X, 'row_id': np.arange(n, dtype=np.int64) + 10 ** 6 * (len(world) + 1), 'unit': unit, 'occupation': occ,
                       'income': inc, 'sex': sex}
    return world


def _fake_access(world, names, calls):
    def load(role, fields, operation, purpose, subset=None):
        labels = [f for f in fields if f in C.LABELS]
        calls.append({'role': role, 'labels': labels, 'operation': operation, 'subset': subset, 'purpose': purpose})
        W = world[role]
        rows = np.ones(len(W['row_id']), dtype=bool)
        if labels:
            if role == 'MODEL_FIT':
                if operation != 'fit' or subset not in C.subset_ids():
                    raise PermissionError('selftest: MODEL_FIT labels only for a registered subset under fit')
                rows = C.subset_mask(W['unit'], subset)
            elif role == 'CALIBRATION':
                if operation != 'calibrate' or 'sex' in labels:
                    raise PermissionError('selftest: CALIBRATION labels only under calibrate')
            else:
                raise PermissionError(f'selftest: {role} labels requested')
        out = {'X': W['X'][rows], 'feature_names': tuple(names), 'row_id': W['row_id'][rows], 'unit': W['unit'][rows], 'rows': rows}
        for f in labels:
            out[f] = W[f][rows]
        return out

    def features(role):
        W = world[role]
        return {'X': W['X'].copy(), 'feature_names': tuple(names), 'row_id': W['row_id'].copy(), 'unit': W['unit'].copy()}
    return load, features


class _Fakes:
    def __init__(self, world, names, calls, source=None):
        self.load, self.features = _fake_access(world, names, calls)
        self.source = source

    def __enter__(self):
        mod = sys.modules[__name__]
        self.saved = (mod._load, mod._features, mod._source_release)
        mod._load, mod._features = self.load, self.features
        if self.source is not None:
            mod._source_release = self.source
        return self

    def __exit__(self, *a):
        mod = sys.modules[__name__]
        mod._load, mod._features, mod._source_release = self.saved
        return False


SELFTEST_CFG = {'lr': 0.1, 'max_iter': 60, 'leaves': 8, 'msl': 20}


def selftest():
    from ohn import fam_a as FA
    from ohn import common as OC
    from cts import trackf as TF
    global _NO_REAL_ACCESS
    _NO_REAL_ACCESS = True
    t0 = time.process_time()
    rng = np.random.default_rng(20261019)
    names = _names()
    sizes = {'MODEL_FIT': 1500, 'CALIBRATION': 500, 'MODEL_SELECT': 400, 'ATTACK_FIT': 120, 'ATTACK_VALIDATION': 110, 'DEV_REPORT': 130}
    A = _fake_world(rng, names, sizes)
    checks, info = {}, {}
    cfg = {**FA.CONFIGS[RECIPE], **SELFTEST_CFG}
    W = A['MODEL_FIT']
    # No call into the real access layer here (invented data only; every load below goes through the fake loader, and the real
    # label hook refuses while _NO_REAL_ACCESS is set). The former probe "real label read refused before the registration" is not
    # applicable after the verified receipt (it would READ labels then), so it is recorded, never executed.
    info['real_access_probe'] = ('not applicable after the receipt (registered); never executed by the selftest' if C.registered()
                                 else 'not executed: the selftest never calls the real access layer')
    # dictionary: identity at S100 on every role (no level below 10 among used ones), collapses rare levels in a half
    dic_all = TF.fit_dictionary(W['X'], names)
    checks['s100_recode_identity_all_roles'] = all(np.array_equal(recode(A[r]['X'], names, dic_all), A[r]['X'].astype(np.float64))
                                                   for r in ROLES)
    # source pipeline (the S100 reference): ohn.fam_a.run_pipeline on the admitted X with the source folds, deployed part
    inner_all = OC.household_folds(W['unit'], 5, FA.inner_tag(0))
    pipe = FA.run_pipeline(cfg, 0, FA.rep_matrix('cmp_po', W['X'], names), W['occupation'], OC.crossfit_folds(W['unit']), inner_all,
                           log=lambda s: None)
    src_rel = {r: FA.release_prob(pipe['deployed'], FA.rep_matrix('cmp_po', A[r]['X'], names)) for r in ROLES}
    src_tau = FA.fit_temperature(FA.raw_full(pipe['deployed'][0], FA.rep_matrix('cmp_po', A['CALIBRATION']['X'], names)),
                                 A['CALIBRATION']['occupation'])[1]['tau']
    src_rec = {'T_star': pipe['T_star'], 'temperature_diagnostics': {'CALIBRATION_full': {'tau': src_tau}},
               'curve_main_oof': FA._curve_summary(pipe['main_curve']), 'unit_id': 'INVENTED'}
    calls = []
    with tempfile.TemporaryDirectory(prefix='tsz_sizefit_selftest_') as tmp:
        root = Path(tmp)
        src_hook = lambda v: (src_rel, src_rec, {'OCC.npz': 'invented'})
        with _Fakes(A, names, calls, source=src_hook):
            r100 = fit('S100', 0, root=root, cfg_override=SELFTEST_CFG, log=lambda s: None)
            rel100, _ = release(r100['fit_id'], root)
            checks['s100_equals_source_pipeline_bitwise_all_roles'] = all(np.array_equal(rel100[r], src_rel[r]) for r in ROLES)
            checks['s100_T_star_equals_source'] = r100['T_star'] == pipe['T_star']
            coll_cols = [names.index(n) for b in BLOCKS for n in dic_all['collapsed'][b]]
            checks['s100_collapsed_levels_have_no_people'] = not any(np.any(A[r]['X'][:, coll_cols]) for r in ROLES)
            rp = repro(0, root)
            checks['repro_reports_reproduced'] = bool(rp['reproduced'] and rp['diagnostic_temperature']['equal'] and rp['inner_curve_summary_equal'])
            # a perturbed source is reported, not hidden
            bad = {r: p.copy() for r, p in src_rel.items()}; bad['DEV_REPORT'][0, 0] += 1e-15; bad['DEV_REPORT'][0, 1] -= 1e-15
            with _Fakes(A, names, calls, source=lambda v: (bad, src_rec, {})):
                (fit_dir(r100['fit_id'], root) / 'REPRO.json').rename(fit_dir(r100['fit_id'], root) / 'REPRO_first.json')
                rp2 = repro(0, root)
            checks['repro_detects_difference'] = bool(not rp2['reproduced'] and not rp2['roles']['DEV_REPORT']['bitwise']
                                                      and rp2['roles']['MODEL_SELECT']['bitwise'])
            # a half: FIT-only dictionary, inner folds of ALL units restricted to the subset, brute-force T*
            r50 = fit('S50_O0', 0, root=root, cfg_override=SELFTEST_CFG, log=lambda s: None)
            m50 = C.subset_mask(W['unit'], 'S50_O0')
            dic50 = TF.fit_dictionary(W['X'][m50], names)
            checks['s50_dictionary_is_fit_only'] = json.loads((fit_dir(r50['fit_id'], root) / 'DICT.json').read_text()) == dic50
            checks['s50_collapses_rare_levels'] = sum(dic50['n_collapsed'].values()) > sum(dic_all['n_collapsed'].values())
            info['s50_n_collapsed'] = dic50['n_collapsed']; info['s50_dictionary_diagnostics'] = r50['dictionary']['diagnostics']['any_block']
            Xr = FA.rep_matrix('cmp_po', TF.recode(W['X'][m50], names, dic50), names); y = W['occupation'][m50]
            inner = inner_all[m50]
            cur = np.zeros(cfg['max_iter'])
            for j in range(5):
                tr, va = inner != j, inner == j
                mj = FA.fit_model(cfg, Xr[tr], y[tr], cfg['max_iter'], 0)
                for t in range(1, cfg['max_iter'] + 1):
                    cur[t - 1] += FA.metric_sums(y[va], FA.softmax(FA.raw_at(mj, Xr[va], t)))[0]
            tb = int(np.argmin(cur)) + 1
            checks['s50_T_star_equals_bruteforce_subset_restricted_folds'] = r50['T_star'] == tb
            own = OC.household_folds(W['unit'][m50], 5, FA.inner_tag(0))
            info['restricted_vs_own_subset_folds_differ'] = bool(not np.array_equal(own, inner))
            mfresh = FA.fit_model(cfg, Xr, y, tb, 0)
            rel50, _ = release(r50['fit_id'], root)
            ms_rep = FA.rep_matrix('cmp_po', TF.recode(A['MODEL_SELECT']['X'], names, dic50), names)
            checks['s50_deployed_equals_fresh_refit'] = bool(np.array_equal(rel50['MODEL_SELECT'], FA.release_prob([mfresh], ms_rep)))
            checks['s50_model_fit_release_covers_all_people'] = rel50['MODEL_FIT'].shape == (sizes['MODEL_FIT'], K_OCC)
            # verify (features alone, bitwise, refusals) and write-once
            checks['verify_bitwise_and_refusals'] = bool(verify(r50['fit_id'], root) and verify(r100['fit_id'], root))
            try:
                fit('S50_O0', 0, root=root, cfg_override=SELFTEST_CFG, log=lambda s: None); checks['write_once'] = False
            except PermissionError:
                checks['write_once'] = True
            # encoding sensitivity: admitted dictionary, guards
            radm = fit('S50_O0', 0, dictionary='admitted', root=root, cfg_override=SELFTEST_CFG, log=lambda s: None)
            checks['adm_no_collapse_same_rows'] = bool(all(v == 0 for v in radm['dictionary']['n_collapsed'].values())
                                                       and radm['rows_sha256'] == r50['rows_sha256'])
            g = []
            for args in (('S70_O0', 0), ('S50_O0', 1), ('S100', 0)):
                try:
                    fit_id(*args, dictionary='admitted'); g.append(False)
                except ValueError:
                    g.append(True)
            checks['adm_guards'] = all(g)
            checks['fit_id_roundtrip'] = all(parse_fit_id(fit_id(s, v, k)) == (s, v, k) for s in C.subset_ids() for v in C.VERSIONS
                                             for k in DICT_KINDS if k == 'fit_only' or (v == 0 and s.startswith('S50_')))
        # label reads: MODEL_FIT under 'fit' with the subset, CALIBRATION under 'calibrate'; never SEX, income or other roles
        lab = [c for c in calls if c['labels']]
        checks['label_reads_policy'] = bool(lab) and all(
            (c['role'] == 'MODEL_FIT' and c['operation'] == 'fit' and c['subset'] in C.subset_ids() and c['labels'] == ['occupation']) or
            (c['role'] == 'CALIBRATION' and c['operation'] == 'calibrate' and c['labels'] == ['occupation']) for c in lab)
        # class support: an occupation class held by one household only -> SUPPORT_FAILURE recorded, no model, refused afterwards
        B = {r: dict(v) for r, v in A.items()}
        yb = np.minimum(A['MODEL_FIT']['occupation'], 3).copy(); yb[0] = 4
        B['MODEL_FIT'] = {**A['MODEL_FIT'], 'occupation': yb}
        with _Fakes(B, names, []):
            try:
                fit('S100', 1, root=root / 'B', cfg_override=SELFTEST_CFG, log=lambda s: None); sf = False
            except SupportFailure:
                d = fit_dir(fit_id('S100', 1), root / 'B')
                sf = (d / 'SUPPORT_FAILURE.json').is_file() and not (d / 'MODEL.pkl').exists()
            try:
                fit('S100', 1, root=root / 'B', cfg_override=SELFTEST_CFG, log=lambda s: None); sf2 = False
            except PermissionError:
                sf2 = True
        checks['support_failure_recorded_and_terminal'] = bool(sf and sf2)
    try:
        fit('S100', 0, cfg_override=SELFTEST_CFG); checks['override_refused_on_real_root'] = False
    except PermissionError:
        checks['override_refused_on_real_root'] = True
    checks['recipe_constants'] = bool(FA.CONFIGS[RECIPE] == {**FA.CONFIGS[RECIPE], 'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.025, 'leaves': 16,
                                                              'msl': 100, 'l2': 3.0, 'max_iter': 1000, 'calibrator': None, 'deploy': 'refit'})
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info, 'cpu_s': round(time.process_time() - t0, 1),
                      'interpreter': _interp()}, default=str))
    return ok


def timing():
    """Invented data of the real size (20,000 MODEL_FIT people, real feature names, feature-dependent invented labels): CPU and wall
    time of the real A1_PO_16 fit path (cfg unchanged) at S100, S70 and S50. Label-free."""
    global _NO_REAL_ACCESS
    _NO_REAL_ACCESS = True
    rng = np.random.default_rng(7)
    names = _names()
    sizes = {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001, 'ATTACK_FIT': 4000, 'ATTACK_VALIDATION': 4000, 'DEV_REPORT': 4000}
    A = _fake_world(rng, names, sizes, rare_count=40)
    out = {}
    with tempfile.TemporaryDirectory(prefix='tsz_sizefit_timing_') as tmp:
        with _Fakes(A, names, []):
            for sid in ('S100', 'S70_O0', 'S50_O0'):
                t0 = time.process_time(); w0 = time.time()
                r = fit(sid, 0, root=Path(tmp), cfg_override={}, log=lambda s: None)
                out[sid] = {'cpu_s': round(time.process_time() - t0, 1), 'wall_s': round(time.time() - w0, 1), 'T_star': r['T_star'],
                            'n_people': r['n_people']}
                print(json.dumps({sid: out[sid]}), flush=True)
            t0 = time.process_time()
            verify(fit_id('S100', 0), Path(tmp))
            out['verify_S100_cpu_s'] = round(time.process_time() - t0, 1)
    out['threads'] = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    print(json.dumps({'timing': out, 'interpreter': _interp()}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest'); sub.add_parser('timing')
    f = sub.add_parser('fit'); f.add_argument('--subset', required=True, choices=C.subset_ids())
    f.add_argument('--version', type=int, required=True, choices=C.VERSIONS); f.add_argument('--dictionary', default='fit_only', choices=DICT_KINDS)
    v = sub.add_parser('verify'); v.add_argument('--fit-id', required=True)
    r = sub.add_parser('repro'); r.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'timing':
        timing(); return 0
    if a.cmd == 'fit':
        try:
            fit(a.subset, a.version, a.dictionary)
        except SupportFailure as e:
            print(json.dumps({'SUPPORT_FAILURE': a.subset, 'version': a.version, 'support': json.loads(str(e))}), flush=True)
            return SUPPORT_EXIT
        return 0
    if a.cmd == 'verify':
        return 0 if verify(a.fit_id) else 1
    repro(a.version)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Conditional correction stage, full pool (PCRL controlled training-size campaign `pcrl_controlled_training_size_v1`;
EXPLORATORY DEVELOPMENT on previously used records, not confirmation). Runs ONLY if the registered task-stage trigger fires
(DESIGN_BRIEF section 6; the statistical reviewer's RULES.json fixes the trigger, the strength set and the selection rule).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). Occupation group (5 official broad civilian groups) is predicted from AGEP, WKHP, COW, SCHL, MAR, POBP through
the admitted 145-column encoding; each recipient receives only its own float64 probability vector and argmax decision. SEX (the
recorded demographic column) is a training label of the correction critics on MODEL_FIT only; never a runtime input or output.

Per source version v in (0, 1), on all 20,000 MODEL_FIT people (registered subset S100, operation 'fit'):
  base       python3   W100_v's 5-fold household cross-fitting on MODEL_FIT exactly as the source fit: ohn.fam_a.run_pipeline with
                       the A1_PO_16 configuration and seed v, outer folds ohn.common.crossfit_folds (OHN1_crossfit), inner folds
                       household_folds(MODEL_FIT units, 5, 'OHN_A_cv_S<v>') restricted to each pipeline's rows, refit at T*_k. The
                       same call refits the deployed model: its six-role release must equal W100_v (tsz.sizefit) bitwise, and the
                       out-of-fold array must equal the admitted source A1_PO_16 S<v> CROSSFIT.npz bitwise (checked and recorded;
                       a difference is reported and blocks the later steps, never tuned away).
  direction  torch     ohn.protect.train_correction (source P2 recipe, guard 'none'), ONE full pipeline per arm (seed 'TSZ1_V<v>',
                       fold 'full'): CONT (zero-critic continuation), LOC (occupation critic; PRIMARY), PAIR (occupation and linked
                       pair critics reading the frozen H3 income S<v> wire on MODEL_FIT). Identical routing, epochs, task loss and
                       effort except the critic term. Routing = the S100 T_TREE occupation teacher of seed v (tsz.control; expected
                       to equal the admitted teacher). Delta on all six roles.
  releases   python3   label-free: r_s = softmax(log p_W + s Delta_arm) (cts.trackf.release_at) for every strength of the frozen set,
                       every arm, all six roles; write-once npz with SHA-256 record; then predict() from features alone is re-run
                       and must reproduce every array bitwise.
  verify     python3   label-free: predict() from features alone against RELEASES.npz (bitwise), written to VERIFY_<utc>.json.
Deployment from features alone: `witness_predict`, `correction_delta`, `predict(v, arm, s, X, names)`.
Outputs: .private/<campaign>/runs/CORRECT/V<v>/ (write-once). Real-data commands run only through `python3 -m tsz.jobqueue run`.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C
from cts import trackf as TF

K_OCC = 5
ROLES = C.ROLES6
ARMS = ('CONT', 'LOC', 'PAIR')
PRIMARY = 'LOC'
W_RECIPE = 'A1_PO_16'
DEFAULT_STRENGTHS = (0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0)       # DESIGN_BRIEF proposal (label-free fallback only)
TARGETS = {'men': 0.005, 'women': 0.005}                         # RULES.json correction.guard_none_targets_constant (the guard cap);
#                                                                  under guard 'none' it only enters logged slack weights (theta is
#                                                                  bitwise invariant to it: selftest check); no MODEL_SELECT quantity
W100_CELL = 'A1__S100__V{v}'                                     # tsz.sizefit.fit_id('S100', v) (task-refit investigator)
CONTROL_CELL = 'S100__V{v}'                                      # tsz.control output (task-refit investigator)
RULES_JSON = C.RESULTS / 'roles' / 'statistical_reviewer' / 'RULES.json'
H3_NOTE = ('H3 income (H/H3_STRONG_LJDIR_PB, source seed v) is FROZEN: fitted by its source campaign on all MODEL_FIT, the '
           "service's fixed income anchor; read on MODEL_FIT only as the PAIR critic's income wire during training (in-sample for "
           'H3), never in a cost, a selection statistic or a release of this stage')


def cdir(v):
    return C.RUNS / 'CORRECT' / f'V{int(v)}'


def seed_str(v):
    return f'TSZ1_V{int(v)}'


def skey(s):
    return repr(float(s))


def release(p_w, delta, s):
    """r_s = softmax(log p_W + s Delta) (cts.trackf.release_at); the identity s = 0 is W itself, exactly (RULES 8.3: at s* = 0 every
    arm equals W exactly), so increments against s = 0 are increments against the witness and vanish exactly at s = 0."""
    if float(s) == 0.0:
        return np.array(p_w, dtype=np.float64, copy=True)
    return TF.release_at(p_w, delta, s)


def sizes(unit, mask=None):
    unit = np.asarray(unit)
    u = unit if mask is None else unit[np.asarray(mask, dtype=bool)]
    return {'people': int(len(u)), 'households': int(len(np.unique(u)))}


# ------------------------------------------------------------------ the frozen strength set
def _find_lists(obj, names, path=()):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                out.append(('/'.join(path + (k,)), tuple(float(x) for x in v)))
            out += _find_lists(v, names, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _find_lists(v, names, path + (str(i),))
    return out


def check_strengths(S):
    S = tuple(float(s) for s in S)
    if not S or S[0] != 0.0 or list(S) != sorted(set(S)) or not all(np.isfinite(S)) or min(S) < 0:
        raise ValueError(f'strength set must be strictly increasing, start at 0 and be finite: {S}')
    return S


def _rules_obj():
    return json.loads(RULES_JSON.read_text()) if RULES_JSON.is_file() else None


def strength_set(strict=True, rules_obj=None, rules_module=None):
    """(strengths, source) from the statistical reviewer's RULES.json: correction.strengths (else the unique list under a key
    'strengths' / 'strength_set' anywhere in it), cross-checked against tsz.rules.STRENGTHS when that module exists (a mismatch
    refuses). strict=True (real-data commands) refuses the design-brief default; strict=False falls back to it (selftest, benchmark)."""
    RU = rules_module
    if RU is None and rules_obj is None:
        try:
            from . import rules as RU                            # noqa: F811 (written by the statistical reviewer)
        except ImportError:
            RU = None
    if rules_obj is None:
        rules_obj = _rules_obj()
    found, src = None, None
    if rules_obj is not None:
        explicit = (rules_obj.get('correction') or {}).get('strengths') if isinstance(rules_obj, dict) else None
        if explicit:
            found, src = check_strengths(explicit), 'RULES.json:correction/strengths'
        else:
            lists = _find_lists(rules_obj, ('strengths', 'strength_set'))
            vals = {x for _, x in lists}
            if len(vals) > 1:
                raise ValueError(f'RULES.json holds several different strength sets: {lists}')
            if vals:
                found, src = check_strengths(vals.pop()), 'RULES.json:' + ','.join(p for p, _ in lists)
    mod = getattr(RU, 'STRENGTHS', None) if RU is not None else None
    if found is not None:
        if mod is not None and check_strengths(mod) != found:
            raise ValueError(f'tsz.rules.STRENGTHS {tuple(mod)} differs from {src} {found}')
        return found, src + (' (= tsz.rules.STRENGTHS)' if mod is not None else '')
    if mod is not None and not strict:
        return check_strengths(mod), 'tsz.rules.STRENGTHS (no RULES.json)'
    if strict:
        raise PermissionError('no frozen strength set found in RULES.json (correction.strengths); real-data steps refuse the '
                              'design-brief default')
    return DEFAULT_STRENGTHS, 'DESIGN_BRIEF proposal (no frozen set found; label-free use only)'


def guard_targets(strict=True):
    """The fixed guard-'none' targets constant, checked against RULES.json correction.guard_none_targets_constant."""
    R = _rules_obj()
    reg = ((R or {}).get('correction') or {}).get('guard_none_targets_constant')
    if reg is None:
        if strict:
            raise PermissionError('RULES.json correction.guard_none_targets_constant missing')
        return dict(TARGETS), 'tsz.correct.TARGETS (no RULES.json)'
    if {k: float(x) for k, x in reg.items()} != TARGETS:
        raise ValueError(f'RULES.json guard_none_targets_constant {reg} differs from tsz.correct.TARGETS {TARGETS}')
    return dict(TARGETS), 'RULES.json:correction/guard_none_targets_constant'


# ------------------------------------------------------------------ witness representation and the cross-fitted base
def witness_rep(X, names, dic):
    from ohn import fam_a as FA
    return FA.rep_matrix(FA.CONFIGS[W_RECIPE]['rep'], TF.recode(X, names, dic), names)


def base_pipeline(cfg, v, Xr, y, unit_all, rows, log=None):
    """A1_PO_16 pipeline with its own household cross-fitting on the training rows `rows` (boolean mask over ALL MODEL_FIT people):
    outer folds ohn.common.crossfit_folds and inner folds household_folds(.., 5, 'OHN_A_cv_S<v>'), both computed on ALL MODEL_FIT
    units and restricted to `rows` (the source's sub-pipeline convention). Xr / y are the training rows only."""
    from ohn import fam_a as FA, common as OC
    rows = np.asarray(rows, dtype=bool)
    unit_all = np.asarray(unit_all)
    if len(Xr) != int(rows.sum()) or len(y) != int(rows.sum()):
        raise ValueError('training arrays must hold exactly the training rows')
    folds = OC.crossfit_folds(unit_all)[rows]
    inner = OC.household_folds(unit_all, 5, FA.inner_tag(v))[rows]
    if sorted(np.unique(folds)) != list(range(OC.CROSSFIT_K)):
        raise AssertionError('a cross-fitting fold is empty inside the training rows')
    for c in range(K_OCC):
        if not (np.asarray(y) == c).any():
            raise AssertionError(f'occupation class {c} absent from the training rows (no support repair for A1_PO_16)')
    pipe = FA.run_pipeline(cfg, v, Xr, y, folds, inner, log=log or (lambda m: None))
    oof = np.full((len(y), K_OCC), np.nan)
    for k in range(pipe['K']):
        oof[folds == k] = FA.combine(pipe['oof_raw'][k])
    if not np.isfinite(oof).all():
        raise AssertionError('out-of-fold base incomplete')
    ur = unit_all[rows]
    return {'pipe': pipe, 'oof': oof, 'folds': folds, 'inner': inner,
            'crossfit_fold_sizes': {str(k): int((folds == k).sum()) for k in range(pipe['K'])},
            'submodel_training_sizes': {'deployed_witness': sizes(ur),
                                        **{f'crossfit_pipeline_{k}': sizes(ur, folds != k) for k in range(pipe['K'])}}}


# ------------------------------------------------------------------ adapters to the size-fit and control outputs
def _role_arrays(npz_path):
    with np.load(npz_path) as z:
        keys = set(z.files)
        for pat in ('{r}__occupation', 'dep__{r}', 'W__{r}', 'occupation__{r}', '{r}'):
            ks = [pat.format(r=r) for r in ROLES]
            if all(k in keys for k in ks):
                return {r: np.asarray(z[pat.format(r=r)], dtype=np.float64).copy() for r in ROLES}, pat
    return None, None


def w100_release(v, root=None):
    """ADAPTER POINT (tsz.sizefit, task-refit investigator): W100_v's stored six-role occupation release.
    Production: tsz.sizefit.release(tsz.sizefit.fit_id('S100', v)) -> ({role: (n, 5)}, FIT.json record; RELEASE.npz hash-checked).
    Fallback (selftest, or a module without `release`): the unique npz under `root` holding all six roles as '<ROLE>__occupation',
    'dep__<ROLE>', 'W__<ROLE>' or '<ROLE>'."""
    cell = W100_CELL.format(v=int(v))
    if root is None:
        from . import sizefit as SF
        if SF.fit_id('S100', int(v)) != cell:
            raise AssertionError('tsz.sizefit fit id convention differs')
        arrs, rec = SF.release(cell)
        missing = [r for r in ROLES if r not in arrs]
        if missing:
            raise KeyError(f'W100 release misses roles {missing}')
        return ({r: np.asarray(arrs[r], dtype=np.float64) for r in ROLES},
                {'source': f'tsz.sizefit.release({cell!r})', 'release_npz_sha256': rec['files_sha256']['RELEASE.npz'],
                 'model_sha256': rec['files_sha256'].get('MODEL.pkl'), 'T_star': rec.get('T_star')})
    root = Path(root)
    hits = []
    for p in sorted(root.rglob('*.npz')) if root.is_dir() else []:
        arrs, pat = _role_arrays(p)
        if arrs is not None:
            hits.append((p, arrs, pat))
    if len(hits) != 1:
        raise FileNotFoundError(f'W100 release of {cell}: {len(hits)} candidate npz files under {root} (need exactly one, or '
                                'tsz.sizefit.load_release)')
    p, arrs, pat = hits[0]
    return arrs, {'source': 'file scan', 'file': str(p), 'file_sha256': C.sha(p), 'key_pattern': pat}


def control_teacher(v, root=None):
    """The T_TREE occupation teacher refitted on S100 with seed v. Production: tsz.control.load_teachers('S100', v) (TEACHERS/
    TEACHER_occupation.json, hash-checked against TEACHERS.json). With `root` (selftest): the unique JSON payload with task
    'occupation' under root (acs_replication.trees payload)."""
    from acs_replication import trees
    if root is None:
        from . import control as CT
        ts, rec = CT.load_teachers('S100', int(v))
        te = ts['occupation']
        p = CT.cell_dir('S100', int(v)) / 'TEACHERS' / 'TEACHER_occupation.json'
        return te, {'source': f"tsz.control.load_teachers('S100', {int(v)})", 'file': str(p.relative_to(C.ROOT)),
                    'file_sha256': rec['files_sha256']['TEACHER_occupation.json'], 'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L}
    root = Path(root)
    hits = []
    for p in sorted(root.rglob('*.json')) if root.is_dir() else []:
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict) and obj.get('task') == 'occupation' and 'leaf_values' in obj and 'split_features' in obj:
            hits.append((p, obj))
    if len(hits) != 1:
        raise FileNotFoundError(f'control S100 occupation teacher V{v}: {len(hits)} candidate payloads under {root}')
    p, obj = hits[0]
    te = trees.TreeTeacher(obj)
    return te, {'source': 'file scan', 'file': str(p), 'file_sha256': C.sha(p), 'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L}


# ------------------------------------------------------------------ saved state (hash-checked loaders)
def _write_npz_x(path, **arrs):
    return C.save_npz_x(path, **arrs)


def load_base(v, d=None):
    d = Path(d) if d is not None else cdir(v) / 'BASE'
    rec = json.loads((d / 'BASE.json').read_text())
    for n, h in rec['files_sha256'].items():
        if C.sha(d / n) != h:
            raise PermissionError(f'BASE {n} differs from its record')
    return C.load_npz(d / 'BASE.npz'), rec, json.loads((d / 'DICT.json').read_text())


def routing_teacher(v, d=None):
    from acs_replication import trees
    d = Path(d) if d is not None else cdir(v) / 'ROUTING'
    rec = json.loads((d / 'ROUTING.json').read_text())
    if C.sha(d / 'TEACHER_occupation.json') != rec['teacher_file_sha256']:
        raise PermissionError('routing teacher file differs from its record')
    te = trees.TreeTeacher(json.loads((d / 'TEACHER_occupation.json').read_text()))
    if te.sha() != rec['teacher_sha256']:
        raise PermissionError('routing teacher payload differs from its record')
    return te, rec


def load_direction(v, arm, d=None):
    d = Path(d) if d is not None else cdir(v) / 'DIR' / arm
    rec = json.loads((d / 'DIR.json').read_text())
    for n, h in rec['files_sha256'].items():
        if C.sha(d / n) != h:
            raise PermissionError(f'DIR {arm} {n} differs from its record')
    return rec, C.load_npz(d / 'THETA.npz')['theta'], C.load_npz(d / 'DELTA.npz')


# ------------------------------------------------------------------ deployment from features alone
def witness_predict(v, X, names, d=None):
    """W100_v's deployed occupation probabilities from admitted features alone: production = W100_v's own package
    (tsz.sizefit.predict); with an explicit BASE directory `d` (selftest) = the BASE refit, which the base step requires to equal
    W100_v bitwise on all six roles."""
    from ohn import fam_a as FA
    if d is None:
        from . import sizefit as SF
        return SF.predict(W100_CELL.format(v=int(v)), X, names)
    d = Path(d)
    rec = json.loads((d / 'BASE.json').read_text())
    if C.sha(d / 'MODEL.pkl') != rec['files_sha256']['MODEL.pkl'] or C.sha(d / 'DICT.json') != rec['files_sha256']['DICT.json']:
        raise PermissionError('witness model or dictionary differs from its record')
    dic = json.loads((d / 'DICT.json').read_text())
    with open(d / 'MODEL.pkl', 'rb') as f:
        m = pickle.load(f)
    return FA.release_prob([m], witness_rep(X, names, dic))


def correction_delta(v, arm, X, names, d=None):
    """Delta_arm(x) on the frozen routing (T_TREE S100 teacher of seed v) from admitted features alone."""
    from ohn import protect as PR
    root = Path(d) if d is not None else cdir(v)
    te, _ = routing_teacher(v, root / 'ROUTING')
    _, theta, _ = load_direction(v, arm, root / 'DIR' / arm)
    brec = json.loads((root / 'BASE' / 'BASE.json').read_text())
    if C.sha(root / 'BASE' / 'DICT.json') != brec['files_sha256']['DICT.json']:
        raise PermissionError('BASE dictionary differs from its record')
    dic = json.loads((root / 'BASE' / 'DICT.json').read_text())
    return PR.correction(theta, te.route(TF.recode(X, names, dic), names), te.T, te.L)


def predict(v, arm, s, X, names, d=None):
    """(n, 5) float64 corrected occupation release r_s = softmax(log p_W + s Delta_arm) from the six raw fields' admitted encoding
    (production: W100_v's tsz.sizefit package + the installed routing + theta; d = a selftest tree)."""
    return release(witness_predict(v, X, names, None if d is None else Path(d) / 'BASE'), correction_delta(v, arm, X, names, d), s)


def predict_many(v, X, names, arms, strengths, d=None):
    pw = witness_predict(v, X, names, None if d is None else Path(d) / 'BASE')
    out = {}
    for arm in arms:
        dl = correction_delta(v, arm, X, names, d)
        for s in strengths:
            out[(arm, skey(s))] = release(pw, dl, s)
    return pw, out


# ------------------------------------------------------------------ base (python3)
def base(v, dry=None):
    """Real data unless `dry` = dict(X, names, y, unit, role_X, d, cfg, source) (selftest)."""
    from ohn import fam_a as FA, devrun as OD
    t0 = time.process_time(); w0 = time.time()
    d = Path(dry['d']) if dry else cdir(v) / 'BASE'
    if (d / 'BASE.json').exists():
        raise PermissionError(f'{d} exists (write-once)')
    if dry:
        X, names, y, unit, rows, role_X = dry['X'], dry['names'], dry['y'], dry['unit'], np.ones(len(dry['y']), bool), dry['role_X']
        row_id = np.arange(len(y))
    else:
        F = C.load('MODEL_FIT', ('features', 'occupation'), 'fit',
                   f'conditional correction V{v}: correction base = W100_V{v} 5-fold household cross-fitting on all MODEL_FIT '
                   f'(A1_PO_16, OHN1_crossfit outer folds, inner CV OHN_A_cv_S{v}); occupation only, no SEX', subset='S100')
        X, names, y, unit, rows, row_id = F['X'], F['feature_names'], F['occupation'], F['unit'], F['rows'], F['row_id']
        if not rows.all():
            raise AssertionError('S100 must hold every MODEL_FIT person')
        role_X = {}
        for r in ROLES:
            Fr = C.features(r)
            if tuple(Fr['feature_names']) != tuple(names):
                raise AssertionError(f'{r}: feature schema differs')
            role_X[r] = F['X'] if r == 'MODEL_FIT' else Fr['X']
    cfg = dict(dry['cfg']) if dry else dict(FA.CONFIGS[W_RECIPE])
    dic = TF.fit_dictionary(X, names)
    Xrec = TF.recode(X, names, dic)
    identity = bool(np.array_equal(Xrec, np.asarray(X, dtype=np.float64)))
    Xr = FA.rep_matrix(cfg['rep'], Xrec, names)
    res = base_pipeline(cfg, v, Xr, y, unit, rows, log=None if dry else (lambda m: print(m, flush=True)))
    pipe = res['pipe']
    p_dep = {r: FA.release_prob(pipe['deployed'], witness_rep(role_X[r], names, dic)) for r in ROLES}
    checks = {'fit_only_dictionary_is_identity_at_S100': identity}
    # source A1_PO_16 S<v> (admitted copy; OCC.npz / CROSSFIT.npz hash-checked against OCC.json by ohn.devrun.load_occupation)
    if dry:
        occ_src, rec_src, cf_src = dry['source']
    else:
        occ_src, rec_src, cf_src = OD.load_occupation('A', W_RECIPE, v)
    checks['source_deployed_bitwise'] = {r: bool(np.array_equal(p_dep[r], occ_src[r])) for r in ROLES}
    checks['source_T_star_equal'] = bool(pipe['T_star'] == rec_src.get('T_star'))
    if cf_src is not None:
        checks['source_crossfit_folds_equal'] = bool(np.array_equal(res['folds'], cf_src['fold']))
        checks['source_oof_bitwise'] = bool(np.array_equal(res['oof'], cf_src['MODEL_FIT__oof']))
        checks['source_oof_max_abs_diff'] = float(np.max(np.abs(res['oof'] - cf_src['MODEL_FIT__oof'])))
    else:
        checks['source_oof_bitwise'] = None
    try:
        w100, wsrc = (dry['w100'] if dry else w100_release(v))
        checks['w100_bitwise'] = {r: bool(np.array_equal(p_dep[r], w100[r])) for r in ROLES}
        checks['w100_source'] = wsrc
    except (FileNotFoundError, KeyError) as e:
        checks['w100_bitwise'] = None
        checks['w100_source'] = {'error': repr(e)[:400]}
    # Gating: the base must be W100_v's own cross-fitting (deployed refit bitwise equal to W100_v on all six roles, S100 dictionary =
    # admitted). The source comparison is reported (RULES section 5: a non-reproduced W100_v is used regardless); it gates only when
    # the deployed release equals the source bitwise, in which case the out-of-fold array must equal the source CROSSFIT.npz too.
    w100_ok = checks['w100_bitwise'] is not None and all(checks['w100_bitwise'].values())
    src_dep_ok = all(checks['source_deployed_bitwise'].values())
    required = bool(identity and w100_ok and (not src_dep_ok or checks['source_oof_bitwise'] is not False))
    checks['gating'] = ('identity_at_S100 AND w100_bitwise (all six roles) AND (source OOF bitwise whenever the deployed release equals '
                        'the source bitwise)')
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    files = {}
    with open(d / 'MODEL.pkl', 'xb') as f:
        pickle.dump(pipe['deployed'][0], f, protocol=5)
    files['MODEL.pkl'] = C.sha(d / 'MODEL.pkl')
    for k, m in pipe['outer_refits'].items():
        with open(d / f'FOLD_{k}.pkl', 'xb') as f:
            pickle.dump(m, f, protocol=5)
        files[f'FOLD_{k}.pkl'] = C.sha(d / f'FOLD_{k}.pkl')
    C.write_json_x(d / 'DICT.json', dic)
    files['DICT.json'] = C.sha(d / 'DICT.json')
    files['BASE.npz'] = _write_npz_x(d / 'BASE.npz', oof=res['oof'], fold=res['folds'], inner=res['inner'], row_id=np.asarray(row_id),
                                     **{f'dep__{r}': p_dep[r] for r in ROLES})
    own = C.task_vectors(y, res['oof'], K_OCC)
    rec = {'schema': 'TSZ1-correct-base-v1', 'version': int(v), 'recipe': W_RECIPE, 'config': cfg, 'T_star': pipe['T_star'],
           'T_k': {str(k): t for k, t in pipe['T_k'].items()}, 'training': sizes(unit),
           'crossfit': {'outer': 'ohn.common.crossfit_folds (OHN1_crossfit, 5 household folds of all MODEL_FIT units)',
                        'inner': f'household_folds(all MODEL_FIT units, 5, {FA.inner_tag(v)!r}) restricted to each pipeline',
                        'fold_sizes': res['crossfit_fold_sizes'], 'submodel_training_sizes': res['submodel_training_sizes']},
           'dictionary': {'n_collapsed': dic['n_collapsed'], 'identity_at_S100': identity},
           'oof_summary': {m: float(x.mean()) for m, x in own.items()}, 'checks': checks, 'checks_pass': bool(required),
           'release_sha256': {r: C.arr_sha(p_dep[r]) for r in ROLES}, 'files_sha256': files,
           'n_physical_fits': pipe['n_physical_fits'], 'labels_used': 'MODEL_FIT occupation (S100); no SEX, no income label',
           'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(d / 'BASE.json', rec)
    C.unit_stats(n_model_fits=len(pipe['deployed_inner_models']) + len(pipe['deployed']),
                 n_crossfit_fits=pipe['n_physical_fits'] - len(pipe['deployed_inner_models']) - len(pipe['deployed']))
    print(json.dumps({'version': v, 'T_star': pipe['T_star'], 'checks_pass': required, 'cpu_s': round(rec['cpu_s'], 1)}), flush=True)
    return rec


# ------------------------------------------------------------------ direction (torch venv)
def _install_routing(v, te, src, d):
    """Copy the routing teacher into CORRECT/V<v>/ROUTING (write-once; a later arm verifies the identical teacher)."""
    from ohn import common as OC
    d = Path(d)
    if (d / 'ROUTING.json').exists():
        te0, rec0 = routing_teacher(v, d)
        if te0.sha() != te.sha():
            raise PermissionError('routing teacher differs from the one installed for an earlier arm')
        return te0, rec0
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(d / 'TEACHER_occupation.json', 'x') as f:
        json.dump(te.payload(), f, sort_keys=True)
    try:
        adm = OC.teacher(int(v), 'occupation').sha()
    except Exception as e:                                         # recorded, not hidden (selftest has no admitted teacher of v)
        adm = f'unavailable: {e!r}'[:200]
    rec = {'schema': 'TSZ1-correct-routing-v1', 'version': int(v), 'source': src, 'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L,
           'teacher_file_sha256': C.sha(d / 'TEACHER_occupation.json'), 'admitted_teacher_sha256': adm,
           'equals_admitted_teacher': te.sha() == adm, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(d / 'ROUTING.json', rec)
    return te, rec


def direction(v, arm, dry=None, epochs=None, steps=None):
    from ohn import protect as PR, devrun as OD
    if arm not in ARMS:
        raise ValueError(arm)
    t0 = time.process_time(); w0 = time.time()
    root = Path(dry['root']) if dry else cdir(v)
    d = root / 'DIR' / arm
    if (d / 'DIR.json').exists():
        raise PermissionError(f'{d} exists (write-once)')
    B, brec, dic = load_base(v, root / 'BASE')
    if not brec['checks_pass']:
        raise PermissionError('the correction base failed its reproduction checks (BASE.json); no direction is trained on it')
    if dry:
        X, names, y, sex, unit, role_X = dry['X'], dry['names'], dry['y'], dry['sex'], dry['unit'], dry['role_X']
        te_src, src = dry['teacher'], {'source': 'invented'}
        inc = dry['income']
    else:
        F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'fit',
                   f'conditional correction V{v}: {arm} direction (task CE + SEX critics, P2 recipe guard none) on all MODEL_FIT from '
                   'the W100 cross-fitted base', subset='S100')
        if not np.array_equal(F['row_id'], B['row_id']):
            raise AssertionError('MODEL_FIT rows differ from the base')
        X, names, y, sex, unit = F['X'], F['feature_names'], F['occupation'], F['sex'], F['unit']
        role_X = {r: (X if r == 'MODEL_FIT' else C.features(r)['X']) for r in ROLES}
        te_src, src = control_teacher(v)
        from . import control as CT
        if CT.load_dict('S100', int(v)) != dic:                   # the routing teacher saw the same (identity) S100 dictionary
            raise AssertionError('control S100 dictionary differs from the base dictionary')
        src['control_dictionary_equals_base'] = True
        inc = OD.anchor_income('H3', int(v), 'MODEL_FIT')[F['rows']]
    targets, tsrc = guard_targets(strict=not dry)
    te, rrec = _install_routing(v, te_src, src, root / 'ROUTING')
    leaves = {r: te.route(TF.recode(role_X[r], names, dic), names) for r in ROLES}
    inc_wire = PR.income_wire(inc)
    r = PR.train_correction(TF.logp(B['oof']), leaves['MODEL_FIT'], y, sex, inc_wire, te.T, te.L, arm, seed_str(v), 'full', targets,
                            epochs=epochs, steps=steps, guard='none')
    delta = {rr: PR.correction(r['theta'], leaves[rr], te.T, te.L) for rr in ROLES}
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    files = {'THETA.npz': _write_npz_x(d / 'THETA.npz', theta=r['theta']),
             'DELTA.npz': _write_npz_x(d / 'DELTA.npz', **delta)}
    rec = {'schema': 'TSZ1-correct-direction-v1', 'version': int(v), 'arm': arm, 'primary': arm == PRIMARY, 'seed': seed_str(v),
           'fold': 'full', 'guard': 'none', 'targets': targets, 'targets_source': tsrc,
           'recipe': PR.RECIPE, 'recipe_source': 'ohn.protect.train_correction (source P2 recipe)',
           'epochs': r['epochs'], 'steps_per_epoch': r['steps_per_epoch'], 'lambda': r['lambda'], 'rho': r['rho'],
           'calibration': r['calibration'], 'training': {**sizes(unit), 'summary': PR.fold_summary(r)}, 'epochs_log': r['epochs_log'],
           'start_in_sample': r['start_in_sample'], 'routing': {'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L, 'record': rrec},
           'base_files_sha256': brec['files_sha256'], 'h3_income': {'array_sha256': C.arr_sha(inc), 'note': H3_NOTE,
                                                                     'used_by_this_arm': arm == 'PAIR'},
           'theta_sha256': C.arr_sha(r['theta']), 'delta_sha256': {rr: C.arr_sha(x) for rr, x in delta.items()}, 'files_sha256': files,
           'labels_used': 'MODEL_FIT occupation (task CE) and SEX (critic training label), S100', 'interpreter': sys.executable,
           'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'n_fits': {'corrections': 1, 'critics': 9, 'note': 'CONT: critics updated, unused (lambda = 0)'}}
    C.write_json_x(d / 'DIR.json', rec)
    C.unit_stats(n_correction_fits=1, n_critic_fits=9)
    print(json.dumps({'version': v, 'arm': arm, 'lambda': r['lambda'], 'cpu_s': round(rec['cpu_s'], 1)}), flush=True)
    return rec


# ------------------------------------------------------------------ releases (python3; label-free)
def releases(v, dry=None):
    t0 = time.process_time()
    root = Path(dry['root']) if dry else cdir(v)
    if (root / 'RELEASES.json').exists() or (root / 'RELEASES.npz').exists():
        raise PermissionError(f'{root}/RELEASES exists (write-once)')
    S, ssrc = dry['strengths'] if dry else strength_set(strict=True)
    B, brec, dic = load_base(v, root / 'BASE')
    if not brec['checks_pass']:
        raise PermissionError('the correction base failed its reproduction checks')
    if dry:
        p_w, wsrc = {r: B[f'dep__{r}'] for r in ROLES}, {'source': 'selftest BASE refit'}
    else:                                                             # p_W = W100_v's own stored arrays (equal to the base refit)
        p_w, wsrc = w100_release(v)
        if not all(np.array_equal(B[f'dep__{r}'], p_w[r]) for r in ROLES):
            raise PermissionError('W100 release differs from the base refit')
    from ohn import protect as PR
    names = dry['names'] if dry else C.features('MODEL_FIT')['feature_names']
    role_X = {r: (dry['role_X'][r] if dry else C.features(r)['X']) for r in ROLES}
    te, _ = routing_teacher(v, root / 'ROUTING')
    leaves = {r: te.route(TF.recode(role_X[r], names, dic), names) for r in ROLES}
    arrays, summ, drecs = {f'W__{r}': p_w[r] for r in ROLES}, {}, {}
    for arm in ARMS:
        drec, theta, D = load_direction(v, arm, root / 'DIR' / arm)
        # Delta recomputed in THIS interpreter on the deployed path (theta + routing from features); compared with the
        # training-time Delta of the direction step (torch venv), recorded
        delta = {r: PR.correction(theta, leaves[r], te.T, te.L) for r in ROLES}
        drecs[arm] = {'files_sha256': drec['files_sha256'], 'lambda': drec['lambda'], 'theta_sha256': drec['theta_sha256'],
                      'delta_bitwise_vs_direction_step': {r: bool(np.array_equal(delta[r], D[r])) for r in ROLES},
                      'delta_max_abs_diff_vs_direction_step': {r: float(np.max(np.abs(delta[r] - D[r]))) for r in ROLES}}
        for s in S:
            for r in ROLES:
                p = release(p_w[r], delta[r], s)
                arrays[f'{arm}__s{skey(s)}__{r}'] = p
                summ[f'{arm}__s{skey(s)}__{r}'] = {'sha256': C.arr_sha(p), 'mean_abs_prob_change': float(np.abs(p - p_w[r]).sum(1).mean()),
                                                   'flip_rate': float((p.argmax(1) != p_w[r].argmax(1)).mean())}
    npz_sha = _write_npz_x(root / 'RELEASES.npz', **arrays)
    # deployment from features alone (production: W100_v's tsz.sizefit package + installed routing + theta), bitwise
    dep = {}
    for r in ROLES:
        pw, rel = predict_many(v, role_X[r], names, ARMS, S, root if dry else None)
        dep[r] = {'witness_bitwise': bool(np.array_equal(pw, p_w[r])),
                  'all_releases_bitwise': bool(all(np.array_equal(rel[(a, skey(s))], arrays[f'{a}__s{skey(s)}__{r}'])
                                                   for a in ARMS for s in S))}
    ok = all(x['witness_bitwise'] and x['all_releases_bitwise'] for x in dep.values())
    rec = {'schema': 'TSZ1-correct-releases-v1', 'version': int(v), 'strengths': list(S), 'strength_source': ssrc, 'arms': list(ARMS),
           'operation': 'r_s = softmax(log p_W + s Delta_arm) (cts.trackf.release_at); p_W = W100_v deployed output; decision = argmax',
           'npz_sha256': npz_sha, 'arrays': summ, 'directions': drecs, 'base_files_sha256': brec['files_sha256'], 'witness_source': wsrc,
           'deployment_from_features_bitwise': dep, 'deployment_ok': ok, 'labels_read': False, 'interpreter': sys.executable,
           'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(root / 'RELEASES.json', rec)
    print(json.dumps({'version': v, 'arrays': len(arrays), 'deployment_ok': ok, 'cpu_s': round(rec['cpu_s'], 1)}), flush=True)
    if not ok:
        raise AssertionError(f'predict() from features differs from the saved releases: {dep}')
    return rec


def load_release(v, arm, s, role, d=None):
    root = Path(d) if d is not None else cdir(v)
    rec = json.loads((root / 'RELEASES.json').read_text())
    if C.sha(root / 'RELEASES.npz') != rec['npz_sha256']:
        raise PermissionError('RELEASES.npz differs from its record')
    with np.load(root / 'RELEASES.npz') as z:
        return z[f'{arm}__s{skey(s)}__{role}'].copy() if arm != 'W' else z[f'W__{role}'].copy()


def realized(v, dry=None):
    """AFTER the committed selection (SELECTION_V<v>.json exists): realized MODEL_SELECT incremental costs of every arm at every
    strength against its own s = 0 release (cts.trackf.cost_summary: means + household-clustered SEs), and realized minus nested
    predicted per statistic. Recorded, never gating, never a strength-rule input (RULES 8.3). Operation 'benchmark'."""
    from . import select as SEL
    from . import nested as NS
    t0 = time.process_time()
    root = Path(dry['root']) if dry else cdir(v)
    outp = root / 'REALIZED_MS.json'
    if outp.exists():
        raise PermissionError(f'{outp} exists (write-once)')
    selp = Path(dry['selection']) if dry else SEL.SEL_DIR / f'SELECTION_V{int(v)}.json'
    if not selp.is_file():
        raise PermissionError('realized MODEL_SELECT costs are computed only after the selection record exists')
    sel = json.loads(selp.read_text())
    rel = json.loads((root / 'RELEASES.json').read_text())
    if C.sha(root / 'RELEASES.npz') != rel['npz_sha256']:
        raise PermissionError('RELEASES.npz differs from its record')
    Z = C.load_npz(root / 'RELEASES.npz')
    if dry:
        y, sex, unit = dry['y'], dry['sex'], dry['unit']
    else:
        S = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'benchmark',
                   f'conditional correction V{v}: realized MODEL_SELECT incremental costs of the CONT / LOC / PAIR corrected releases '
                   'at every registered strength, after the committed selection (recorded, never a strength-rule input)')
        y, sex, unit = S['occupation'], S['sex'], S['unit']
    out, diff = {}, {}
    for arm in ARMS:
        r0 = Z[f'{arm}__s{skey(0.0)}__MODEL_SELECT']
        out[arm] = {skey(s): TF.cost_summary(Z[f'{arm}__s{skey(s)}__MODEL_SELECT'], r0, y, sex, unit) for s in rel['strengths']}
        nroot = Path(dry['nested_root']) if dry else NS.ndir(v)
        if (nroot / f'NESTED_{arm}.json').is_file():
            pred = NS.load_nested(v, arm, nroot)['per_strength']
            diff[arm] = {k: {j: out[arm][k]['stats'][j] - pred[k]['stats'][j] for j in NS.STATS} for k in out[arm] if k in pred}
    rec = {'schema': 'TSZ1-correct-realized-ms-v1', 'version': int(v), 'selection_sha256': C.sha(selp), 'selected': sel.get('selected'),
           'cont_matched': sel.get('cont_matched'), 'realized_MODEL_SELECT': out, 'realized_minus_nested_predicted': diff,
           'releases_npz_sha256': rel['npz_sha256'], 'note': 'recorded, never gating; MODEL_SELECT is never a strength-rule input',
           'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(outp, rec)
    print(json.dumps({'version': v, 'selected': sel.get('selected'), 'cpu_s': round(rec['cpu_s'], 2)}), flush=True)
    return rec


def verify(v):
    """Label-free re-check: predict() from features alone against RELEASES.npz, every arm / strength / role (bitwise)."""
    root = cdir(v)
    rec = json.loads((root / 'RELEASES.json').read_text())
    if C.sha(root / 'RELEASES.npz') != rec['npz_sha256']:
        raise PermissionError('RELEASES.npz differs from its record')
    Z = C.load_npz(root / 'RELEASES.npz')
    names = C.features('MODEL_FIT')['feature_names']
    res = {}
    for r in ROLES:
        pw, rel = predict_many(v, C.features(r)['X'], names, ARMS, rec['strengths'], None)   # W100_v's own package (tsz.sizefit)
        res[r] = bool(np.array_equal(pw, Z[f'W__{r}']) and all(np.array_equal(rel[(a, skey(s))], Z[f'{a}__s{skey(s)}__{r}'])
                                                              for a in ARMS for s in rec['strengths']))
    out = {'schema': 'TSZ1-correct-verify-v1', 'version': int(v), 'roles_bitwise': res, 'ok': all(res.values()), 'utc': C.now(),
           'interpreter': sys.executable, 'code_sha256': C.sha(__file__), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(root / f'VERIFY_{C.now().replace(":", "")}.json', out)
    print(json.dumps(out))
    return 0 if out['ok'] else 1


# ------------------------------------------------------------------ selftest (invented data only)
def invented(rng, n=1600, T=12, nfold_house=2):
    """Invented features in the admitted 145-column schema (column names from a stored teacher payload, no person data), invented
    households, occupation, income probabilities and SEX."""
    from ohn import fam_a as FA, common as OC, protect as PR
    names = list(OC.teacher(0, 'occupation').feature_names)
    X = FA._fake_X(n, names, rng).astype(np.float64)
    X = TF.recode(X, names, TF.fit_dictionary(X, names))      # like the admitted encoding: every kept level has >= 10 people
    Xc = FA.compact(X, names)
    y = FA._fake_labels(Xc, rng)
    unit = np.repeat(np.arange((n + nfold_house - 1) // nfold_house), nfold_house)[:n] + 50_000
    sex = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * Xc[:, 1] - 0.5 * (y == 2) + 0.5 * (y == 4))))).astype(np.int64)
    p1 = 1 / (1 + np.exp(-(0.6 * Xc[:, 0] + 0.3 * Xc[:, 1])))
    inc = np.column_stack((1 - p1, p1))
    te = PR._fake_teacher(rng, T, tuple(names))
    return {'names': tuple(names), 'X': X, 'y': y, 'unit': unit, 'sex': sex, 'income': inc, 'teacher': te}


def selftest():
    from ohn import fam_a as FA, protect as PR
    t0 = time.process_time()
    rng = np.random.default_rng(20261019)
    checks = {}
    # 1. strength set resolution (invented rules objects)
    S, src = strength_set(strict=False, rules_obj={'x': 1})
    checks['strengths_default_when_absent'] = S == DEFAULT_STRENGTHS and 'DESIGN_BRIEF' in src
    S, src = strength_set(strict=False, rules_obj={'conditional': {'correction': {'strengths': [0, 0.05, 0.2, 1]}}})
    checks['strengths_from_rules_json'] = S == (0.0, 0.05, 0.2, 1.0) and 'strengths' in src
    try:
        strength_set(strict=True, rules_obj={'x': 1}); checks['strengths_strict_refuses_default'] = False
    except PermissionError:
        checks['strengths_strict_refuses_default'] = True
    try:
        strength_set(strict=False, rules_obj={'a': {'strengths': [0, 0.1]}, 'b': {'strengths': [0, 0.2]}}); checks['strengths_ambiguous_refused'] = False
    except ValueError:
        checks['strengths_ambiguous_refused'] = True
    try:
        check_strengths((0.1, 0.3)); checks['strengths_must_start_at_zero'] = False
    except ValueError:
        checks['strengths_must_start_at_zero'] = True
    if RULES_JSON.is_file():                                     # the registered constants (label-free file read)
        S, src = strength_set(strict=True)
        tg, tsrc = guard_targets(strict=True)
        checks['registered_strengths_and_targets'] = {'strengths': S, 'source': src, 'targets': tg, 'targets_source': tsrc,
                                                      'pass': len(S) >= 2 and S[0] == 0.0 and tg == TARGETS}
    # 2. base pipeline (small invented configuration) on a household-restricted row set: folds computed on all units then restricted
    D = invented(rng)
    names, X, y, unit = D['names'], D['X'], D['y'], D['unit']
    cfg = {'learner': 'hgb', 'rep': 'cmp_po', 'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 25, 'calibrator': None,
           'deploy': 'refit', 'kind': 'task_model'}
    from ohn import common as OC
    rows = OC.household_folds(unit, 10, 'TSZ1_selftest_ncv') != 3
    dic = TF.fit_dictionary(X[rows], names)
    Xr = FA.rep_matrix(cfg['rep'], TF.recode(X[rows], names, dic), names)
    res = base_pipeline(cfg, 0, Xr, y[rows], unit, rows)
    cf = OC.crossfit_folds(unit)[rows]; inn = OC.household_folds(unit, 5, FA.inner_tag(0))[rows]
    pipe2 = FA.run_pipeline(cfg, 0, Xr, y[rows], cf, inn, log=lambda m: None)
    checks['base_folds_restricted_from_all_units'] = bool(np.array_equal(res['folds'], cf) and np.array_equal(res['inner'], inn))
    oof_ok = True
    for k in range(5):
        m = FA.fit_model(cfg, Xr[cf != k], y[rows][cf != k], pipe2['T_k'][k], 0)
        oof_ok &= bool(np.array_equal(FA.softmax(FA.raw_full(m, Xr[cf == k])), res['oof'][cf == k]))
    checks['base_oof_equals_fresh_fold_refits_bitwise'] = oof_ok
    checks['base_deterministic'] = bool(np.array_equal(FA.release_prob(res['pipe']['deployed'], Xr), FA.release_prob(pipe2['deployed'], Xr)))
    # 3. adapters (temporary directories)
    tmp = Path(tempfile.mkdtemp(prefix='tsz_correct_selftest_'))
    arrs = {r: rng.dirichlet(np.ones(K_OCC), size=7) for r in ROLES}
    (tmp / 'SIZE').mkdir()
    np.savez(tmp / 'SIZE' / 'OCC.npz', **{f'{r}__occupation': a for r, a in arrs.items()})
    got, info = w100_release(0, root=tmp / 'SIZE')
    checks['w100_adapter_file_scan'] = bool(all(np.array_equal(got[r], arrs[r]) for r in ROLES) and info['key_pattern'] == '{r}__occupation')
    np.savez(tmp / 'SIZE' / 'OTHER.npz', **{f'dep__{r}': a for r, a in arrs.items()})
    try:
        w100_release(0, root=tmp / 'SIZE'); checks['w100_adapter_ambiguous_refused'] = False
    except FileNotFoundError:
        checks['w100_adapter_ambiguous_refused'] = True
    (tmp / 'CONTROL').mkdir()
    (tmp / 'CONTROL' / 'TEACHER_occupation.json').write_text(json.dumps(D['teacher'].payload()))
    (tmp / 'CONTROL' / 'RECORD.json').write_text(json.dumps({'task': 'occupation', 'note': 'no payload'}))
    te2, tinfo = control_teacher(0, root=tmp / 'CONTROL')
    checks['control_adapter_file_scan'] = te2.sha() == D['teacher'].sha()
    # 4. full dry run of base -> direction (torch only) -> releases -> deployment, invented data in a temporary tree
    role_X = {r: (X if r == 'MODEL_FIT' else FA._fake_X(300 + 10 * i, list(names), rng).astype(np.float64)) for i, r in enumerate(ROLES)}
    dtree = tmp / 'CORRECT' / 'V0'
    dic_all = TF.fit_dictionary(X, names)
    Xr_all = FA.rep_matrix(cfg['rep'], TF.recode(X, names, dic_all), names)
    src_pipe = base_pipeline(cfg, 0, Xr_all, y, unit, np.ones(len(y), bool))      # stands in for the source fit
    src_occ = {r: FA.release_prob(src_pipe['pipe']['deployed'], witness_rep(role_X[r], names, dic_all)) for r in ROLES}
    src = (src_occ, {'T_star': src_pipe['pipe']['T_star']}, {'MODEL_FIT__oof': src_pipe['oof'], 'fold': src_pipe['folds']})
    brec = base(0, dry={'X': X, 'names': names, 'y': y, 'unit': unit, 'role_X': role_X, 'd': dtree / 'BASE', 'cfg': cfg, 'source': src,
                        'w100': (src_occ, {'source': 'invented'})})
    checks['dry_base_reproduces_source_and_w100'] = bool(brec['checks_pass'] and brec['checks']['source_oof_bitwise'])
    bad = {r: a.copy() for r, a in src_occ.items()}
    bad['DEV_REPORT'][0] = bad['DEV_REPORT'][0][::-1]                            # one person's W100 vector differs
    brec_bad = base(0, dry={'X': X, 'names': names, 'y': y, 'unit': unit, 'role_X': role_X, 'd': tmp / 'BAD' / 'BASE', 'cfg': cfg,
                            'source': src, 'w100': (bad, {'source': 'invented, perturbed'})})
    checks['dry_base_fails_when_w100_differs'] = bool(not brec_bad['checks_pass'] and not brec_bad['checks']['w100_bitwise']['DEV_REPORT'])
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    if have_torch:
        dry = {'root': dtree, 'X': X, 'names': names, 'y': y, 'sex': D['sex'], 'unit': unit, 'role_X': role_X, 'teacher': D['teacher'],
               'income': D['income']}
        try:
            direction(0, 'LOC', dry={**dry, 'root': tmp / 'BAD'}, epochs=1, steps=2); checks['dry_direction_refuses_failed_base'] = False
        except PermissionError:
            checks['dry_direction_refuses_failed_base'] = True
        recs = {arm: direction(0, arm, dry=dry, epochs=2, steps=4) for arm in ARMS}
        checks['dry_cont_lambda_zero_loc_pair_positive'] = bool(recs['CONT']['lambda'] == 0.0 and recs['LOC']['lambda'] > 0 and recs['PAIR']['lambda'] > 0)
        checks['dry_arms_differ'] = len({recs[a]['theta_sha256'] for a in ARMS}) == 3
        r2 = PR.train_correction(TF.logp(load_base(0, dtree / 'BASE')[0]['oof']), D['teacher'].route(TF.recode(X, names, dic_all), names), y,
                                 D['sex'], PR.income_wire(D['income']), D['teacher'].T, D['teacher'].L, 'PAIR', seed_str(0), 'full', TARGETS,
                                 epochs=2, steps=4, guard='none')
        checks['dry_direction_deterministic_bitwise'] = C.arr_sha(r2['theta']) == recs['PAIR']['theta_sha256']
        # RULES 8.1: under guard 'none' theta is bitwise invariant to `targets` (no MODEL_SELECT quantity can enter); every arm
        lv_all = D['teacher'].route(TF.recode(X, names, dic_all), names)
        zb = TF.logp(load_base(0, dtree / 'BASE')[0]['oof'])
        inv = {}
        for arm in ARMS:
            th = [PR.train_correction(zb, lv_all, y, D['sex'], PR.income_wire(D['income']), D['teacher'].T, D['teacher'].L, arm, seed_str(0),
                                      'inv', tg, epochs=2, steps=4, guard='none')['theta']
                  for tg in (TARGETS, {'men': 1.0, 'women': 1.0}, {'men': 1e-6, 'women': 0.3})]
            inv[arm] = bool(np.array_equal(th[0], th[1]) and np.array_equal(th[0], th[2]))
        checks['theta_bitwise_invariant_to_targets_under_guard_none'] = {'per_arm': inv, 'targets_tried': [TARGETS, {'men': 1.0, 'women': 1.0},
                                                                          {'men': 1e-6, 'women': 0.3}], 'pass': all(inv.values())}
        S = (0.0, 0.01, 0.1, 1.0)
        rrec = releases(0, dry={'root': dtree, 'strengths': (S, 'selftest'), 'names': names, 'role_X': role_X})
        checks['dry_releases_deploy_bitwise'] = bool(rrec['deployment_ok'])
        p0 = load_release(0, 'LOC', 0.0, 'MODEL_SELECT', dtree); pw = load_release(0, 'W', 0.0, 'MODEL_SELECT', dtree)
        checks['dry_identity_is_W_exactly'] = bool(np.array_equal(p0, pw)) and all(
            np.array_equal(load_release(0, a, 0.0, r, dtree), load_release(0, 'W', 0.0, r, dtree)) for a in ARMS for r in ROLES)
        p1 = predict(0, 'PAIR', 0.1, role_X['DEV_REPORT'], names, dtree)
        checks['dry_predict_single'] = bool(np.array_equal(p1, load_release(0, 'PAIR', 0.1, 'DEV_REPORT', dtree)))
        try:
            releases(0, dry={'root': dtree, 'strengths': (S, 'selftest'), 'names': names, 'role_X': role_X}); checks['dry_releases_write_once'] = False
        except PermissionError:
            checks['dry_releases_write_once'] = True
        n_ms = len(role_X['MODEL_SELECT'])
        drr = {'root': dtree, 'selection': tmp / 'SELECTION_V0.json', 'y': rng.integers(0, K_OCC, n_ms), 'sex': rng.integers(0, 2, n_ms),
               'unit': np.arange(n_ms) // 2, 'nested_root': tmp / 'no_nested'}
        try:
            realized(0, dry=drr); checks['dry_realized_refuses_before_selection'] = False
        except PermissionError:
            checks['dry_realized_refuses_before_selection'] = True
        (tmp / 'SELECTION_V0.json').write_text(json.dumps({'selected': {'LOC': 0.1, 'PAIR': 0.0}}))
        rz = realized(0, dry=drr)
        checks['dry_realized_zero_at_identity'] = all(x == 0.0 for a in ARMS for x in rz['realized_MODEL_SELECT'][a]['0.0']['stats'].values())
    else:
        checks['torch_part'] = 'SKIPPED (no torch in this interpreter; run the selftest under tsz.common.TORCH_PY too)'
    ok = all((v['pass'] if isinstance(v, dict) else v) for v in checks.values() if not isinstance(v, str))
    out = {'selftest': 'PASS' if ok else 'FAIL', 'module': 'tsz.correct', 'interpreter': sys.executable, 'torch': have_torch,
           'checks': checks, 'data': 'invented only (numpy default_rng(20261019)); no real data or label read',
           'cpu_s': round(time.process_time() - t0, 1)}
    print(json.dumps(out, default=str))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    for c in ('base', 'releases', 'verify', 'realized'):
        x = sub.add_parser(c); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    x = sub.add_parser('direction'); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    x.add_argument('--arm', required=True, choices=ARMS)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    if a.cmd == 'base':
        return 0 if base(a.version)['checks_pass'] else 1
    if a.cmd == 'direction':
        direction(a.version, a.arm)
        return 0
    if a.cmd == 'releases':
        releases(a.version)
        return 0
    if a.cmd == 'realized':
        realized(a.version)
        return 0
    return verify(a.version)


if __name__ == '__main__':
    sys.exit(main())

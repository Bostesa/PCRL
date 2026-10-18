"""Track R of the PCRL correction-transfer campaign: whole-service refit of the source income recipe on the FIT households of a
repair partition (EXPLORATORY DEVELOPMENT on previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). The service predicts an income class (2) and an occupation group (5 official broad civilian groups) from six
raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding). Each recipient receives only its own float64
probability vector and argmax decision. SEX (the recorded demographic column) is a training-time label of the Stage B critics on
FIT households only; never a runtime input or output.

A recipe test on a different training subset (DESIGN_BRIEF section 6), not a confirmation of the frozen H3 package. Per repair
partition P in (R0, R1), associated replication seed s (R0 -> 0, R1 -> 1). Labels are read only for the MODEL_FIT rows of (P, FIT)
(cts.common.load operation 'fit') and for CALIBRATION (operation 'calibrate'); GUARD, CHECK (C0 / C1), MODEL_SELECT, ATTACK and
DEV_REPORT labels are never read here. Every role's features are used label-free to compute the release.

  dictionary  cts.trackf.fit_dictionary on FIT and cts.trackf.recode for every role (the replication's >= 10-people level rule applied
              to the training pool; AGEP / WKHP keep the admitted MODEL_FIT z-map, to which the tree learners are order-invariant).
  teachers    (CatBoost venv, cts.common.CATBOOST_PY) the replication Stage A T_TREE loop (acs_replication.stage_a.run) for income
              and occupation on FIT: CatBoost 1.2.8, depth 4, lr 0.05, l2 3, <= 800 trees, random_seed =
              trees.seed_for('ACSR1catboost', s, task, 4) (unreduced, as stored by the replication); tree count = first argmin of the
              loss on the 15% household internal split of FIT (acs_replication.data.internal_split, both tasks' support repair) + 1;
              refit on FIT; numpy teacher (trees.from_catboost) checked with trees.verify_against_catboost on all six roles.
  stage_b     (torch venv, cts.common.TORCH_PY) the replication Stage B loop (acs_replication.stage_b.run, leafcorr.Trainer /
              run_arm) on FIT: zero-correction identity on all six roles, warm critics + 16-draw FIT calibration, LOCAL_TREE
              (L_TREE) and JOINT_TREE (J_TREE) at rho = 1 (TASK_REFIT on request), checkpoints E0, E2, E5, E10, E20; probe and
              reference rows by the replication's label-free hash order within FIT.
  income      (torch venv) the H3 recipe of ihs.fam_h with its own functions: Step 1 (HGB_GRID, household 5-fold CV on FIT with
              the source tag, refit on FIT), STRONG stack, the precision-aware B1 beta budget on household cross-fitted CALIBRATION
              and the Platt map on CALIBRATION, with the refitted T_TREE income and LOCAL / JOINT E20 income corrections:
              H3R (H3_STRONG_LJDIR_PB recipe) and HCR (HC_STRONG recipe, the task-only twin) income releases for all six roles;
              predict() from features alone, checked bitwise.

Commands (real-data ones only through `python3 -m cts.jobqueue run ...`):
  estimate                         CPU estimate per partition from the replication's / source campaign's own records (no data)
  selftest                         invented data only; the real command paths run against a fake loader (CatBoost or torch parts
                                   are mocked, and recorded as mocked, when the interpreter lacks them)
  teachers --partition P           CatBoost venv
  stage_b  --partition P [--arms LOCAL_TREE,JOINT_TREE]       torch venv
  income   --partition P           torch venv (also runs verify)
  verify   --partition P           label-free deployment parity (predict == saved release, bitwise, all six roles)
  crosscheck --partition P         label-free: dictionary and occupation teacher SHA-256 against Track F's REFIT routing of P
Outputs: .private/<campaign>/runs/TRACK_R/<P>/ (write-once).
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from . import common as C

TRACK = 'TRACK_R'
PARTS = ('R0', 'R1')
TASKS = ('income', 'occupation')
ROLES = C.ROLES6
DEFAULT_ARMS = ('LOCAL_TREE', 'JOINT_TREE')
INCOME_CONFIGS = {'H3R': 'H3_STRONG_LJDIR_PB', 'HCR': 'HC_STRONG'}
LABEL_FIELDS = ('income', 'occupation', 'sex')
DEVIATIONS = (
    'dictionary: the replication rule (a COW / SCHL / MAR / POBP level is kept iff >= 10 training-pool people have it, else the '
    "block's OTHER column) is applied to FIT through cts.trackf.fit_dictionary / recode; AGEP and WKHP keep the admitted MODEL_FIT "
    'z-map, which is strictly increasing and affine; CatBoost and HistGradientBoosting bin by value order, so the fitted trees do not '
    'depend on it (up to floating-point rounding of bin edges)',
    'training pool: the FIT households of the partition instead of all 20,000 MODEL_FIT people; the 15% internal split, the Step-1 '
    'household folds (same tags) and the Stage B probe / reference rows (same label-free hash order) are formed within FIT',
    'Stage B: the recipe fixes 20 epochs x 40 steps x 256 rows per recipient, so each epoch covers 10,240 of about 10,500 FIT people '
    '(about 97%; recorded exactly) instead of 10,240 of 20,000: with the same number of updates each FIT person is used about twice '
    'as often',
    'Stage B arms: LOCAL_TREE and JOINT_TREE (H3 uses only their E20 income corrections); TASK_REFIT only on request. The '
    'zero-correction identity is checked against the teacher outputs computed in the CatBoost environment, not against stored '
    'replication outputs',
    'H3 Step 1: the ExtraTrees comparison model and the CALIBRATION comparison table of ihs.fam_h.step1 are not refitted (neither '
    'enters the H3 or HC release)',
    'budget: the B1 values are the source values (they were set from MODEL_SELECT reference values); CALIBRATION is the same admitted '
    'role as for the frozen H3, shared by R0 and R1, so it is not an independent holdout',
    'not refitted: U0 and Ucal (the utility references stay frozen), NEURAL_RAW and every other reference; the occupation side of a '
    'complete Track R release is the Track F pipeline (not part of this module)',
)


# ------------------------------------------------------------------ data access (hooks; the selftest replaces them with fakes)
def _load(role, fields, operation, purpose, partition=None, prole=None):
    return C.load(role, fields, operation, purpose, partition=partition, prole=prole)


def _features(role):
    return C.features(role)


def pdir(P, root=None):
    return (Path(root) if root else C.RUNS / TRACK) / P


def seed_of(P):
    if P not in PARTS:
        raise ValueError('Track R runs on the repair partitions R0 / R1 only')
    return int(C.PARTITIONS[P]['seed'])


def _load_fit(P, fields, purpose):
    seed_of(P)
    return _load('MODEL_FIT', ('features',) + tuple(fields), 'fit', purpose, partition=P, prole='FIT')


def _load_cal(fields, purpose):
    if 'sex' in fields:
        raise PermissionError('CALIBRATION has no SEX')
    return _load('CALIBRATION', ('features',) + tuple(fields), 'calibrate', purpose)


def role_features(dic):
    from . import trackf as TF
    out = {}
    for r in ROLES:
        F = _features(r)
        out[r] = {'X': TF.recode(F['X'], F['feature_names'], dic), 'names': tuple(map(str, F['feature_names'])),
                  'row_id': F['row_id'], 'unit': F['unit']}
    return out


def _save_npz(path, **arrs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return C.sha(path)


def _load_npz(path, sha=None):
    if sha is not None and C.sha(path) != sha:
        raise PermissionError(f'{Path(path).name} differs from its record')
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _write_once(path, obj):
    C.write_json_x(path, obj)
    return C.sha(path)


def _fresh_dir(d, record_name):
    """Write-once step directory. A completed step (its record exists) is never redone; an incomplete one (a failed attempt) is
    moved aside, kept and named, so that the retry unit (a new unit id) starts from an empty directory."""
    d = Path(d)
    if (d / record_name).exists():
        raise PermissionError(f'{d} is complete (append-only)')
    moved = None
    if d.exists():
        moved = d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}')
        d.rename(moved)
    d.mkdir(parents=True, mode=0o700)
    return None if moved is None else moved.name


def _interp():
    import importlib.metadata as m
    out = {'python': sys.version.split()[0], 'executable': sys.executable}
    for p in ('numpy', 'scipy', 'scikit-learn', 'torch', 'catboost'):
        try:
            out[p] = m.version(p)
        except Exception:
            out[p] = None
    return out


# ------------------------------------------------------------------ teachers (replication Stage A loop; CatBoost venv)
def fit_teachers_arrays(X, names, y, unit, seed, max_trees=None, log=print):
    """acs_replication.stage_a.run's T_TREE loop on the given training rows (X recoded, y = {'income', 'occupation'})."""
    from acs_replication import data as AD, trees
    fit_rows, val_rows, moved = AD.internal_split(unit, y, seed)
    split = {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows)), 'moved_households': len(moved),
             'household_overlap': int(len(np.intersect1d(unit[fit_rows], unit[val_rows])))}
    out = {}
    for task in TASKS:
        depth = trees.REGISTRY['depth'][0]
        rs = trees.seed_for('ACSR1catboost', seed, task, depth)
        t0 = time.process_time()
        m = trees.make_classifier(task, depth, max_trees or trees.REGISTRY['max_trees'], rs)
        m.fit(X[fit_rows], y[task][fit_rows], eval_set=(X[val_rows], y[task][val_rows]))
        curve = m.get_evals_result()['validation'][trees.LOSS[task]]
        count = int(np.argmin(curve)) + 1
        final = trees.make_classifier(task, depth, count, rs)
        final.fit(X, y[task])
        teacher = trees.from_catboost(final, task, depth, names)
        out[task] = {'teacher': teacher, 'model': final,
                     'info': {'depth': depth, 'catboost_seed': rs, 'tree_count': count, 'max_trees': int(max_trees or trees.REGISTRY['max_trees']),
                              'at_cap': bool(count >= int(max_trees or trees.REGISTRY['max_trees'])), 'internal_curve_len': len(curve),
                              'internal_best_loss': float(curve[count - 1]), 'internal_curve_every_50': [float(c) for c in curve[::50]],
                              'cpu_s': time.process_time() - t0,
                              'params': {k: (v.item() if isinstance(v, (np.floating, np.integer)) else v) for k, v in final.get_params().items()}}}
        log(json.dumps({'task': task, 'tree_count': count, 'cpu_s': round(out[task]['info']['cpu_s'], 1)}))
    return out, split


def _parity_ok(v):
    return v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12


def teachers(P, root=None, max_trees=None, fitter=None):
    """Fit and save the FIT-only T_TREE teachers of partition P (write-once)."""
    from acs_replication import trees
    from . import trackf as TF
    t0 = time.process_time(); w0 = time.time()
    d = pdir(P, root) / 'TEACHERS'
    if (d / 'TEACHERS.json').exists():
        raise PermissionError(f'{P} Track R teachers exist (append-only)')
    s = seed_of(P)
    F = _load_fit(P, ('income', 'occupation'), f'Track R {P}: replication T_TREE recipe (CatBoost 1.2.8) for income and occupation '
                  'on FIT (15% household internal split of FIT for the tree counts, refit on FIT); no SEX')
    names = tuple(map(str, F['feature_names']))
    dic = TF.fit_dictionary(F['X'], names)
    X = TF.recode(F['X'], names, dic)
    y = {t: np.asarray(F[t], dtype=np.int64) for t in TASKS}
    mocked = fitter is not None
    fitted, split = (fitter or fit_teachers_arrays)(X, names, y, F['unit'], s, max_trees=max_trees)
    roles = role_features(dic)
    incomplete = _fresh_dir(d, 'TEACHERS.json')
    files = {'DICT.json': _write_once(d / 'DICT.json', dic)}
    outputs, rec_t = {}, {}
    for task in TASKS:
        teacher, model = fitted[task]['teacher'], fitted[task]['model']
        parity = None
        if model is not None:
            parity = {r: trees.verify_against_catboost(teacher, model, roles[r]['X'], names) for r in ROLES}
            if not all(_parity_ok(v) for v in parity.values()):
                raise AssertionError(f'teacher conversion parity failed: {task}: {parity}')
            model.save_model(str(d / f'CATBOOST_{task}.cbm'))
            files[f'CATBOOST_{task}.cbm'] = C.sha(d / f'CATBOOST_{task}.cbm')
        with open(d / f'TEACHER_{task}.json', 'x') as f:
            json.dump(teacher.payload(), f, sort_keys=True)
        files[f'TEACHER_{task}.json'] = C.sha(d / f'TEACHER_{task}.json')
        for r in ROLES:
            outputs[f'T_TREE__{r}__{task}'] = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(roles[r]['X'], names)))
        rec_t[task] = {**fitted[task]['info'], 'T': teacher.T, 'L': teacher.L, 'teacher_sha256': teacher.sha(), 'parity': parity}
    files['TEACHER_OUTPUTS.npz'] = _save_npz(d / 'TEACHER_OUTPUTS.npz', **outputs, **{f'row_id__{r}': roles[r]['row_id'] for r in ROLES})
    rec = {'schema': 'CTS1-trackr-teachers-v1', 'partition': P, 'seed': s, 'mocked': mocked,
           'recipe': 'acs_replication.stage_a T_TREE loop on FIT (recoded with the FIT-only dictionary); unreduced CatBoost seeds',
           'catboost_version': trees.CATBOOST_VERSION, 'registry': trees.REGISTRY, 'internal_split': split,
           'n_fit_people': int(len(F['row_id'])), 'n_fit_households': int(len(np.unique(F['unit']))),
           'fit_rows_sha256': C.arr_sha(np.asarray(F['row_id'])), 'dictionary': {'min_count': dic['min_count'], 'n_collapsed': dic['n_collapsed']},
           'teachers': rec_t, 'files_sha256': files, 'previous_incomplete_attempt': incomplete,
           'labels_read': 'MODEL_FIT (P, FIT): income, occupation; no SEX',
           'interpreter': _interp(), 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'TEACHERS.json', rec)
    C.unit_stats(n_teacher_fits=4)
    if root is None:
        try:
            rec['crosscheck'] = crosscheck(P)
        except FileNotFoundError:
            pass
    print(json.dumps({'partition': P, 'tree_counts': {t: rec_t[t]['tree_count'] for t in TASKS}, 'split': split,
                      'cpu_s': round(rec['cpu_s'], 1)}))
    return rec


def load_dict(P, root=None):
    rec = json.loads((pdir(P, root) / 'TEACHERS' / 'TEACHERS.json').read_text())
    p = pdir(P, root) / 'TEACHERS' / 'DICT.json'
    if C.sha(p) != rec['files_sha256']['DICT.json']:
        raise PermissionError('DICT.json differs from its record')
    return json.loads(p.read_text())


def load_teachers(P, root=None):
    from acs_replication import trees
    d = pdir(P, root) / 'TEACHERS'
    rec = json.loads((d / 'TEACHERS.json').read_text())
    out = {}
    for task in TASKS:
        p = d / f'TEACHER_{task}.json'
        if C.sha(p) != rec['files_sha256'][f'TEACHER_{task}.json']:
            raise PermissionError(f'TEACHER_{task}.json differs from its record')
        t = trees.TreeTeacher(json.loads(p.read_text()))
        if t.sha() != rec['teachers'][task]['teacher_sha256']:
            raise PermissionError(f'{task} teacher payload differs')
        out[task] = t
    return out, rec


def crosscheck(P, root=None):
    """Label-free: the Track R occupation teacher and dictionary against Track F's REFIT routing of the same partition."""
    from . import trackf as TF
    tf_rec = TF.pdir(P) / 'ROUTING' / 'ROUTING.json'
    if not tf_rec.is_file():
        raise FileNotFoundError('Track F REFIT routing not fitted yet')
    rf = json.loads(tf_rec.read_text())
    _, rec = load_teachers(P, root)
    mine = load_dict(P, root)
    theirs = json.loads((TF.pdir(P) / 'DICT.json').read_text())
    out = {'schema': 'CTS1-trackr-crosscheck-v1', 'partition': P, 'trackr_occupation_teacher_sha256': rec['teachers']['occupation']['teacher_sha256'],
           'trackf_refit_routing_teacher_sha256': rf.get('teacher_sha256'),
           'identical_occupation_teacher': rec['teachers']['occupation']['teacher_sha256'] == rf.get('teacher_sha256'),
           'identical_dictionary': mine == theirs, 'trackr_tree_count': rec['teachers']['occupation']['tree_count'],
           'trackf_tree_count': rf.get('tree_count'), 'utc': C.now()}
    C.write_json(pdir(P, root) / 'CROSSCHECK.json', out)
    print(json.dumps(out))
    return out


# ------------------------------------------------------------------ Stage B (replication loop; torch venv)
@contextlib.contextmanager
def _recipe(overrides):
    """Temporarily override leafcorr.RECIPE entries (selftest only; the real run uses none and asserts the final label E20)."""
    from acs_replication import leafcorr as lc
    old = {k: lc.RECIPE[k] for k in (overrides or {})}
    try:
        lc.RECIPE.update(overrides or {})
        yield lc.RECIPE
    finally:
        lc.RECIPE.update(old)


def run_stage_b_arrays(seed, tch, X, names, y_inc, y_occ, sex, row_id, arms, on_checkpoint, recipe_overrides=None, log=print):
    """acs_replication.stage_b.run's training sequence on the given rows; returns (record, {arm: final thetas})."""
    from acs_replication import data as AD, leafcorr as lc
    from acs_replication.stage_b import ARMS
    with _recipe(recipe_overrides):
        n = len(y_inc); B = lc.RECIPE['batch_size']; steps = lc.RECIPE['steps_per_epoch']
        if n < steps * B:
            raise ValueError(f'every task batch must be full: n = {n} < steps_per_epoch x batch_size = {steps * B}')
        tl = [tch[t] for t in TASKS]
        leaves = [tl[i].route(X, names) for i in range(2)]
        upd = np.arange(n)
        probe = AD.order_rows(row_id, 'ACSR1-probe', 512); reference = AD.order_rows(row_id, 'ACSR1-reference', 256)
        y = [np.ascontiguousarray(y_inc, dtype=np.int64), np.ascontiguousarray(y_occ, dtype=np.int64)]
        sex = np.ascontiguousarray(sex, dtype=np.int64)

        def make():
            rec = [lc.Recipient(t, tl[i], leaves[i], tl[i].raw_from_leaves(leaves[i])) for i, t in enumerate(TASKS)]
            return lc.Trainer(seed, rec, y, upd, sex, probe, reference)
        tr0 = make(); critics = tr0.warm(); calib = tr0.calibrate(critics); warm_sha = critics.state_sha()
        final = f"E{lc.RECIPE['epochs_per_active_recipient']}"
        record = {'seed': seed, 'calibration': calib, 'warm_critic_sha256': warm_sha, 'runs': {}, 'checkpoints': {},
                  'probe_rows_sha256': C.arr_sha(probe), 'reference_rows_sha256': C.arr_sha(reference), 'final_label': final,
                  'n': int(n), 'epoch_coverage': steps * B / n, 'recipe': {k: lc.RECIPE[k] for k in (
                      'batch_size', 'steps_per_epoch', 'epochs_per_active_recipient', 'checkpoint_epochs', 'warm_critic_rounds',
                      'critic_rounds_per_recipient_update', 'learning_rate', 'weight_decay')}, 'recipe_overrides': recipe_overrides or {}}
        finals, logs = {}, {}
        for name in arms:
            saved = {}

            def on_ck(label, thetas, info, name=name, saved=saved):
                saved[label] = on_checkpoint(name, label, thetas, info)
                if label == final:
                    finals[name] = [np.array(t, dtype=np.float64, copy=True) for t in thetas]
            tr = make()
            t0 = time.process_time()
            res = lc.run_arm(tr, ARMS[name], 1.0, calib['coefficients'], on_ck, counterfactual_steps=8 if name == 'LOCAL_TREE' else 0)
            if res['warm_critic_sha256'] != warm_sha:
                raise AssertionError('warm critics differ across runs')
            record['runs'][name] = {**{k: v for k, v in res.items() if k != 'logs'}, 'cpu_s': time.process_time() - t0}
            record['checkpoints'][name] = saved
            logs[name] = res['logs']
            log(json.dumps({'arm': name, 'lambda': res['lambda'], 'steps': res['steps'], 'cpu_s': round(record['runs'][name]['cpu_s'], 1)}))
        if set(finals) != set(arms):
            raise AssertionError('a final checkpoint is missing')
    return record, finals, logs


def stage_b(P, root=None, arms=DEFAULT_ARMS, recipe_overrides=None, trainer=None):
    """Train and save the FIT-only Stage B corrections of partition P on the Track R teachers (write-once)."""
    from acs_replication import trees
    from . import trackf as TF
    t0 = time.process_time(); w0 = time.time()
    arms = tuple(arms)
    if not set(arms) <= {'TASK_REFIT', 'LOCAL_TREE', 'JOINT_TREE'} or not {'LOCAL_TREE', 'JOINT_TREE'} <= set(arms):
        raise ValueError('arms must include LOCAL_TREE and JOINT_TREE (TASK_REFIT optional)')
    d = pdir(P, root) / 'STAGE_B'
    if (d / 'STAGE_B.json').exists():
        raise PermissionError(f'{P} Track R Stage B exists (append-only)')
    s = seed_of(P)
    tch, trec = load_teachers(P, root)
    dic = load_dict(P, root)
    stored = _load_npz(pdir(P, root) / 'TEACHERS' / 'TEACHER_OUTPUTS.npz', trec['files_sha256']['TEACHER_OUTPUTS.npz'])
    roles = role_features(dic)
    identity = {}
    for t in TASKS:
        te = tch[t]
        for r in ROLES:
            if not np.array_equal(stored[f'row_id__{r}'], roles[r]['row_id']):
                raise AssertionError(f'{r}: row order differs from the teacher outputs')
            zero = trees.probabilities(t, te.raw_from_leaves(te.route(roles[r]['X'], roles[r]['names']), np.zeros((te.T * te.L, te.dim))))
            ref = stored[f'T_TREE__{r}__{t}']
            identity[f'{t}|{r}'] = {'max_abs_prob_diff': float(np.max(np.abs(zero - ref))),
                                    'decisions_identical': bool(np.array_equal(zero.argmax(1), ref.argmax(1)))}
    if not all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in identity.values()):
        raise AssertionError(f'zero-correction identity failed: {identity}')
    F = _load_fit(P, ('income', 'occupation', 'sex'), f'Track R {P}: replication Stage B recipe on FIT (LOCAL_TREE and JOINT_TREE '
                  'leaf-score corrections on the Track R teachers; task losses and online SEX critics on FIT households)')
    names = tuple(map(str, F['feature_names']))
    if not np.array_equal(F['row_id'], roles['MODEL_FIT']['row_id'][F['rows']]):
        raise AssertionError('FIT rows differ from the MODEL_FIT feature rows')
    X = TF.recode(F['X'], names, dic)
    incomplete = _fresh_dir(d, 'STAGE_B.json')
    files = {}

    def on_checkpoint(name, label, thetas, info):
        p = d / f'{name}__{label}.npz'
        h = _save_npz(p, theta_income=thetas[0], theta_occupation=thetas[1])
        files[p.name] = h
        return {'file': p.name, 'sha256': h, **info}
    mocked = trainer is not None
    record, finals, logs = (trainer or run_stage_b_arrays)(s, tch, X, names, F['income'], F['occupation'], F['sex'], F['row_id'], arms,
                                                           on_checkpoint, recipe_overrides=recipe_overrides)
    if root is None and (record['final_label'] != 'E20' or recipe_overrides):
        raise AssertionError('real Track R runs use the unchanged Stage B recipe (final checkpoint E20)')
    outputs = {}
    for name, th in finals.items():
        for r in ROLES:
            for i, t in enumerate(TASKS):
                te = tch[t]
                outputs[f'{name}__{r}__{t}'] = te.probabilities_from_raw(te.raw_from_leaves(te.route(roles[r]['X'], roles[r]['names']), th[i]))
    files['STAGE_B_OUTPUTS_FINAL.npz'] = _save_npz(d / 'STAGE_B_OUTPUTS_FINAL.npz', **outputs)
    with open(d / 'LOGS.json', 'x') as f:
        json.dump(logs, f, default=float)
    files['LOGS.json'] = C.sha(d / 'LOGS.json')
    rec = {'schema': 'CTS1-trackr-stage-b-v1', 'partition': P, 'seed': s, 'mocked': mocked, 'arms': list(arms), 'zero_identity': identity,
           **record, 'teachers_sha256': {t: trec['teachers'][t]['teacher_sha256'] for t in TASKS},
           'fit_rows_sha256': C.arr_sha(np.asarray(F['row_id'])), 'files_sha256': files, 'previous_incomplete_attempt': incomplete,
           'labels_read': 'MODEL_FIT (P, FIT): income, occupation, SEX (training-time critics)', 'interpreter': _interp(),
           'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    if rec['fit_rows_sha256'] != trec['fit_rows_sha256']:
        raise AssertionError('Stage B FIT rows differ from the teachers\' FIT rows')
    _write_once(d / 'STAGE_B.json', rec)
    C.unit_stats(n_correction_fits=len(arms), n_critic_fits=9 * (len(arms) + 1))
    print(json.dumps({'partition': P, 'arms': list(arms), 'lambda': {k: v.get('lambda') for k, v in record['runs'].items()},
                      'epoch_coverage': record['epoch_coverage'], 'cpu_s': round(rec['cpu_s'], 1)}, default=float))
    return rec


def load_thetas(P, root=None, arms=('LOCAL_TREE', 'JOINT_TREE')):
    """Final-checkpoint thetas {arm: [theta_income, theta_occupation]} (hash-checked)."""
    d = pdir(P, root) / 'STAGE_B'
    rec = json.loads((d / 'STAGE_B.json').read_text())
    out = {}
    for a in arms:
        ck = rec['checkpoints'][a][rec['final_label']]
        z = _load_npz(d / ck['file'], ck['sha256'])
        out[a] = [z['theta_income'], z['theta_occupation']]
    return out, rec


# ------------------------------------------------------------------ income (ihs.fam_h H3 / HC recipe on the refitted components)
def components(tch, thetas, hgb, params, X, names):
    """ihs.fam_h.components with the Track R teachers and corrections: T_TREE, LOCAL (s_L), D_L = s_L - s_T, D_J = s_J - s_T, HGB."""
    from ihs import fam_h as FH
    te = tch['income']
    L = te.route(X, names)
    s_T = te.raw_from_leaves(L).reshape(-1)
    s_L = te.raw_from_leaves(L, thetas['LOCAL_TREE'][0]).reshape(-1)
    s_J = te.raw_from_leaves(L, thetas['JOINT_TREE'][0]).reshape(-1)
    out = {'T_TREE': s_T, 'LOCAL': s_L, 'D_J': s_J - s_T, 'D_L': s_L - s_T}
    if hgb is not None:
        out['HGB'] = hgb.decision_function(FH.rep_matrix(params['rep'], X, names)).reshape(-1)
    return out


def fit_income_arrays(seed, Xf, names, yf, uf, tch, thetas, Xc, yc, uc, grid=None, beta_grid=None, log=print):
    """ihs.fam_h.step1 (HGB part) on FIT and ihs.fam_h.fit (H3_STRONG_LJDIR_PB, HC_STRONG) on CALIBRATION, with Track R components."""
    from ihs import common as IC
    from ihs import fam_h as FH
    grid = FH.HGB_GRID if grid is None else tuple(grid)
    beta_grid = FH.BETA_GRID if beta_grid is None else tuple(beta_grid)
    t0 = time.process_time()
    yf = np.asarray(yf, dtype=np.int64); yc = np.asarray(yc, dtype=np.int64)
    folds = IC.household_folds(uf, FH.N_FOLDS, f'IHS_H_step1_cv_S{seed}')
    reps = {r: FH.rep_matrix(r, Xf, names) for r in sorted({g['rep'] for g in grid})}
    rows, fits = [], 0
    for params in grid:
        curves, nf = FH.cv_curves(reps[params['rep']], yf, folds, params, seed); fits += nf
        i = int(np.argmin(curves['brier']))
        rows.append({'params': params, 'best_iter': i + 1, 'oof_brier': float(curves['brier'][i]), 'oof_nll': float(curves['nll'][i]),
                     'oof_accuracy': float(curves['accuracy'][i]), 'oof_nll_min': float(curves['nll'].min()),
                     'oof_brier_at_max': float(curves['brier'][-1]), 'curve_brier_every_50': [float(c) for c in curves['brier'][49::50]]})
        log(json.dumps({'hgb_setting': params, 'best_iter': i + 1, 'oof_brier': round(rows[-1]['oof_brier'], 6)}))
    best = min(rows, key=lambda g: g['oof_brier'])
    params = {**best['params'], 'max_iter': best['best_iter']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hgb = FH.make_hgb(params, seed).fit(reps[params['rep']], yf)
    fits += 1
    step1 = {'folds': FH.N_FOLDS, 'tag': f'IHS_H_step1_cv_S{seed}', 'grid': rows, 'selected': params, 'n_model_fits': fits,
             'cpu_s': time.process_time() - t0}
    cc = components(tch, thetas, hgb, params, Xc, names)
    cfolds = IC.household_folds(uc, FH.N_FOLDS, f'IHS_H_budget_S{seed}')
    states = {}
    for tag, cid in INCOME_CONFIGS.items():
        cfg = FH.CONFIGS[cid]
        Zb = FH.base_design(cc, cfg['base'])
        table, nfits, note = None, 0, 'no budgeted direction'
        if cfg.get('direction') is None:
            beta = 0.0
        else:
            beta, table, nf = FH.budget_select(Zb, cc, cfg, yc, uc, cfolds, FH.BUDGETS[cfg['budget']], grid=beta_grid); nfits += nf
            note = 'largest grid beta inside the budget'
            if beta is None:
                beta, note = 0.0, 'OUTSIDE BUDGET even at beta = 0 (the fixed residual alone exceeds it); released at beta = 0'
        pres = bool(cfg.get('preserve_decision'))
        v0 = FH.person_metrics(yc, FH.crossfit(Zb, np.zeros(len(yc)), yc, cfolds)); nfits += 2 * FH.N_FOLDS
        v1 = FH.person_metrics(yc, FH.crossfit(Zb, FH.offset(cc, cfg, beta), yc, cfolds, pres)); nfits += 2 * FH.N_FOLDS
        cal_cost = {'brier': float((v1['brier'] - v0['brier']).mean()), 'nll': float((v1['nll'] - v0['nll']).mean()),
                    'accuracy': float((v0['accuracy'] - v1['accuracy']).mean()), 'reference': 'cross-fitted task-only base map',
                    'base_reference': {m: float(v0[m].mean()) for m in ('brier', 'nll', 'accuracy')}}
        wb, wp = FH.fit_maps(Zb, FH.offset(cc, cfg, beta), yc); nfits += 2
        states[tag] = {'tag': tag, 'config_id': cid, 'config': cfg, 'base_columns': list(FH.BASE_COLS[cfg['base']]),
                       'stack_w': [float(x) for x in wb], 'platt_w': [float(x) for x in wp], 'beta': float(beta), 'budget_note': note,
                       'budget': FH.BUDGETS.get(cfg['budget']) if cfg['budget'] else None, 'budget_table': table,
                       'calibration_cost_vs_base': cal_cost, 'hgb_params': params, 'n_calibrator_fits': nfits,
                       'budget_tag': f'IHS_H_budget_S{seed}'}
    return hgb, params, step1, states


def income_release(state, tch, thetas, hgb, X, names):
    """Released income probability vectors (n, 2) of a saved H3R / HCR state from recoded features (ihs.fam_h.predict path)."""
    from ihs import fam_h as FH
    cfg = state['config']
    comp = components(tch, thetas, hgb, state['hgb_params'], X, names)
    d = FH.release_logodds(FH.base_design(comp, cfg['base']), FH.offset(comp, cfg, state['beta']),
                           np.asarray(state['stack_w'], dtype=np.float64), np.asarray(state['platt_w'], dtype=np.float64),
                           bool(cfg.get('preserve_decision')))
    return FH.inc_prob(d)


def income(P, root=None, grid=None, beta_grid=None):
    """Fit and save the H3R / HCR income releases of partition P (write-once); then verify deployment parity."""
    from . import trackf as TF
    t0 = time.process_time(); w0 = time.time()
    d = pdir(P, root) / 'INCOME'
    if (d / 'INCOME.json').exists():
        raise PermissionError(f'{P} Track R income exists (append-only)')
    s = seed_of(P)
    tch, trec = load_teachers(P, root)
    dic = load_dict(P, root)
    thetas, brec = load_thetas(P, root)
    if brec['teachers_sha256'] != {t: trec['teachers'][t]['teacher_sha256'] for t in TASKS}:
        raise AssertionError('Stage B corrections were trained on other teachers')
    F = _load_fit(P, ('income',), f'Track R {P}: H3 recipe Step 1 (HistGradientBoosting income grid, household 5-fold CV and refit on '
                  'FIT); no SEX')
    Cal = _load_cal(('income',), f'Track R {P}: H3 recipe CALIBRATION stack, B1 beta budget (household cross-fitted) and Platt map '
                    'with the Track R teachers and corrections')
    names = tuple(map(str, F['feature_names']))
    if C.arr_sha(np.asarray(F['row_id'])) != trec['fit_rows_sha256']:
        raise AssertionError('income FIT rows differ from the teachers\' FIT rows')
    Xf = TF.recode(F['X'], names, dic); Xc = TF.recode(Cal['X'], names, dic)
    hgb, params, step1, states = fit_income_arrays(s, Xf, names, F['income'], F['unit'], tch, thetas, Xc, Cal['income'], Cal['unit'],
                                                   grid=grid, beta_grid=beta_grid)
    incomplete = _fresh_dir(d, 'INCOME.json')
    files = {}
    with open(d / 'HGB.pkl', 'xb') as f:
        pickle.dump(hgb, f, protocol=5)
    files['HGB.pkl'] = C.sha(d / 'HGB.pkl')
    for tag, st in states.items():
        files[f'STATE_{tag}.json'] = _write_once(d / f'STATE_{tag}.json', st)
    roles = role_features(dic)
    arrays, summary = {}, {}
    for tag in states:
        st = json.loads((d / f'STATE_{tag}.json').read_text())          # the saved (JSON round-tripped) state, as predict() reads it
        for r in ROLES:
            p = income_release(st, tch, thetas, hgb, roles[r]['X'], roles[r]['names'])
            arrays[f'{tag}__{r}__income'] = p
            summary[f'{tag}|{r}'] = {'n': int(len(p)), 'mean_p1': float(p[:, 1].mean()), 'share_class1': float((p.argmax(1) == 1).mean()),
                                     'release_sha256': C.arr_sha(p)}
    files['RELEASE.npz'] = _save_npz(d / 'RELEASE.npz', **arrays, **{f'row_id__{r}': roles[r]['row_id'] for r in ROLES})
    import sklearn
    rec = {'schema': 'CTS1-trackr-income-v1', 'partition': P, 'seed': s, 'configs': INCOME_CONFIGS,
           'mechanism': {'H3R': 'd = a (s_B + D_L + beta D_J) + b; s_B = CALIBRATION stack of the FIT-refitted HGB income log-odds and the '
                                'Track R T_TREE income log-odds; D_L / D_J = Track R LOCAL / JOINT E20 income corrections; beta by the '
                                'precision-aware B1 rule on household cross-fitted CALIBRATION; (a, b) Platt map on CALIBRATION',
                         'HCR': 'the same Step 1 and stack without the residual (task-only twin)'},
           'step1': step1, 'states': {t: {k: v for k, v in st.items() if k not in ('budget_table', 'config')} for t, st in states.items()},
           'teachers_sha256': brec['teachers_sha256'], 'stage_b_final_label': brec['final_label'], 'fit_rows_sha256': trec['fit_rows_sha256'],
           'calibration_rows_sha256': C.arr_sha(np.asarray(Cal['row_id'])), 'role_summary': summary, 'files_sha256': files,
           'previous_incomplete_attempt': incomplete,
           'deviations': list(DEVIATIONS), 'labels_read': 'MODEL_FIT (P, FIT): income; CALIBRATION: income; no SEX in this step',
           'sklearn': sklearn.__version__, 'interpreter': _interp(), 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0,
           'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'INCOME.json', rec)
    C.unit_stats(n_model_fits=step1['n_model_fits'], n_calibrator_fits=sum(st['n_calibrator_fits'] for st in states.values()))
    ok = verify(P, root)
    print(json.dumps({'partition': P, 'hgb': params, 'beta': {t: st['beta'] for t, st in states.items()},
                      'stack_w': {t: [round(x, 4) for x in st['stack_w']] for t, st in states.items()},
                      'platt_w': {t: [round(x, 4) for x in st['platt_w']] for t, st in states.items()}, 'verify': ok,
                      'cpu_s': round(rec['cpu_s'], 1)}))
    if not ok:
        raise AssertionError('deployment parity failed')
    return rec


def load_income(P, tag, root=None):
    d = pdir(P, root) / 'INCOME'
    rec = json.loads((d / 'INCOME.json').read_text())
    if C.sha(d / 'HGB.pkl') != rec['files_sha256']['HGB.pkl'] or C.sha(d / f'STATE_{tag}.json') != rec['files_sha256'][f'STATE_{tag}.json']:
        raise PermissionError('Track R income files differ from their record')
    with open(d / 'HGB.pkl', 'rb') as f:
        hgb = pickle.load(f)
    return hgb, json.loads((d / f'STATE_{tag}.json').read_text()), rec


def predict(P, tag, X, names, root=None):
    """(n, 2) float64 income probability vectors of a saved Track R release (H3R or HCR), recomputed from admitted features alone."""
    from . import trackf as TF
    names = tuple(map(str, names))
    tch, _ = load_teachers(P, root)
    th, _ = load_thetas(P, root)
    hgb, st, _ = load_income(P, tag, root)
    return income_release(st, tch, th, hgb, TF.recode(X, names, load_dict(P, root)), names)


def release_arrays(P, tag, role, root=None):
    """The saved income release of one role (hash-checked), e.g. for the coordinator's benchmark evaluator (MODEL_SELECT)."""
    d = pdir(P, root) / 'INCOME'
    rec = json.loads((d / 'INCOME.json').read_text())
    z = _load_npz(d / 'RELEASE.npz', rec['files_sha256']['RELEASE.npz'])
    return z[f'{tag}__{role}__income'], z[f'row_id__{role}']


def verify(P, root=None):
    """Deployment parity: predict() from each role's features alone against the saved release (bitwise; label-free)."""
    d = pdir(P, root) / 'INCOME'
    rec = json.loads((d / 'INCOME.json').read_text())
    z = _load_npz(d / 'RELEASE.npz', rec['files_sha256']['RELEASE.npz'])
    res = {}
    for tag in INCOME_CONFIGS:
        for r in ROLES:
            F = _features(r)
            if not np.array_equal(F['row_id'], z[f'row_id__{r}']):
                raise AssertionError(f'{r}: row order differs from the saved release')
            p = predict(P, tag, F['X'], F['feature_names'], root)
            ref = z[f'{tag}__{r}__income']
            res[f'{tag}|{r}'] = {'bitwise': bool(np.array_equal(p, ref)), 'max_abs_diff': float(np.max(np.abs(p - ref))),
                                 'decisions_identical': bool(np.array_equal(p.argmax(1), ref.argmax(1)))}
    ok = all(v['bitwise'] for v in res.values())
    C.write_json(d / 'VERIFY.json', {'ok': ok, 'roles': res, 'utc': C.now(), 'interpreter': _interp(), 'unit_id': os.environ.get(C.UNIT_ENV)})
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'partition': P}))
    return ok


# ------------------------------------------------------------------ CPU estimate (from the replication's / source's own records)
def estimate():
    """Upper CPU estimate per partition from the recorded source costs (label-free records only)."""
    from ihs import common as IC
    from ohn import common as OC
    rep = json.loads((C.ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'RESOURCE_INVENTORY.json').read_text())
    units = {u['unit_id']: u['charged_cpu_s'] for u in rep['units']}
    tree_cpu, step1_cpu, fit_cpu = [], [], []
    for s in C.SEEDS:
        a = json.loads((OC.ADM / 'stage_a' / f'S{s}' / 'STAGE_A_SEED.json').read_text())
        tree_cpu.append(sum(a['T_TREE'][t]['cpu_s'] for t in TASKS))
        st1 = json.loads((IC.RUNS / 'H' / 'STEP1' / f'S{s}' / 'STEP1.json').read_text())
        step1_cpu.append(st1['cpu_s'])
        fit_cpu.append(json.loads((IC.RUNS / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{s}' / 'RELEASE.json').read_text())['fit_cpu_s'])
    stage_b = max(units['STAGE_B_S0_V1'], units['STAGE_B_S1_V1'])
    part = json.loads((C.RESULTS / 'PARTITIONS.json').read_text())['summary']
    frac = max(part[P]['FIT']['people'] for P in PARTS) / 20000.0
    est = {'teachers_s': max(tree_cpu) * frac + 15.0,                       # CatBoost loop scales ~ with rows; + parity / IO allowance
           'stage_b_s': stage_b * 1.0 + 15.0,                                # 3 arms in the replication unit; per-step cost ~ row-free
           'income_s': max(step1_cpu) * frac + 2 * max(fit_cpu) + 20.0}      # HGB grid scales ~ with rows; ET and CAL table skipped
    est['per_partition_s'] = sum(est.values())
    est['both_partitions_s'] = 2 * est['per_partition_s']
    est['safety_factor_3_cpu_h'] = 3 * est['both_partitions_s'] / 3600
    out = {'records': {'stage_a_tree_cpu_s': tree_cpu, 'stage_b_unit_cpu_s': [units['STAGE_B_S0_V1'], units['STAGE_B_S1_V1']],
                       'ihs_step1_cpu_s': step1_cpu, 'ihs_h3_fit_cpu_s': fit_cpu, 'fit_fraction': frac},
           'estimate': est, 'track_r_cap_cpu_h': 2.5, 'fits': est['safety_factor_3_cpu_h'] <= 2.5}
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ invented-data self-test (no real data)
def _fake_X(n, names, rng):
    X = np.zeros((n, len(names)), dtype=np.float32)
    X[:, 0] = np.round(rng.normal(size=n) * 4) / 4; X[:, 1] = np.round(rng.normal(size=n) * 4) / 4
    for b in ('COW', 'SCHL', 'MAR', 'POBP'):
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        p = rng.dirichlet(np.ones(len(idx)) * 0.3)                     # skewed: many levels are rare, so the re-collapse acts
        X[np.arange(n), rng.choice(idx, size=n, p=p)] = 1.0
    return X


def _fake_world(rng, names, sizes):
    """Invented roles with households, labels depending on invented features, and an R-like FIT / GUARD / CHECK split."""
    world = {}
    start = 0
    for role, n in sizes.items():
        X = _fake_X(n, names, rng)
        sizes_hh = rng.integers(1, 4, size=n)
        unit = np.repeat(np.arange(start, start + n), sizes_hh)[:n]; start += n
        a, w = X[:, 0].astype(np.float64), X[:, 1].astype(np.float64)
        sch = X[:, [i for i, nm in enumerate(names) if nm.startswith('SCHL=')]].argmax(1).astype(np.float64)
        cow = X[:, [i for i, nm in enumerate(names) if nm.startswith('COW=')]].argmax(1)
        inc = (rng.random(n) < 1 / (1 + np.exp(-(1.1 * w + 0.12 * (sch - 12) + 0.3 * a - 0.4)))).astype(np.int64)
        s = np.column_stack([0.8 * w + 0.1 * sch, -0.6 * w + 0.3 * (cow % 3), 0.3 * a, -0.1 * sch + 0.2 * (cow == 1), 0.4 * a * w])
        pz = np.exp(s - s.max(1, keepdims=True)); pz /= pz.sum(1, keepdims=True)
        occ = (rng.random(n)[:, None] > np.cumsum(pz, 1)).sum(1).astype(np.int64)
        sex = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 * w + 1.0 * (occ == 3) - 1.0 * (occ == 0))))).astype(np.int64)
        world[role] = {'X': X, 'row_id': np.arange(n, dtype=np.int64) + 10 ** 6 * len(world), 'unit': unit,
                       'income': inc, 'occupation': occ, 'sex': sex}
    part = C.household_folds(world['MODEL_FIT']['unit'], 6, 'CTS1_trackr_selftest')
    world['MODEL_FIT']['FIT'] = part <= 3
    world['MODEL_FIT']['GUARD'] = part == 4
    world['MODEL_FIT']['CHECK'] = part == 5
    return world


def _fake_access(world, names, P, calls):
    def load(role, fields, operation, purpose, partition=None, prole=None):
        labels = [f for f in fields if f in LABEL_FIELDS]
        calls.append({'role': role, 'labels': labels, 'operation': operation, 'partition': partition, 'prole': prole})
        W = world[role]
        rows = np.ones(len(W['row_id']), dtype=bool)
        if labels:
            if role == 'MODEL_FIT':
                if not (operation == 'fit' and partition == P and prole == 'FIT'):
                    raise PermissionError(f'selftest: MODEL_FIT labels requested outside (P, FIT): {operation} {partition} {prole}')
                rows = W['FIT'].copy()
            elif role == 'CALIBRATION':
                if operation != 'calibrate' or 'sex' in labels:
                    raise PermissionError('selftest: CALIBRATION labels outside the calibrate operation')
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


def mock_teacher(X, y, task, names, T=8, depth=4, seed=0):
    """A tiny random oblivious-tree payload: random split columns, label-free borders (FIT quantiles), leaf values = smoothed per-leaf
    class log-frequencies of the given (FIT) rows / T. Stands in for CatBoost when the interpreter has none."""
    from acs_replication import trees
    rng = np.random.default_rng(seed)
    K = trees.CLASSES[task]; dim = 1 if task == 'income' else K; L = 2 ** depth
    X32 = np.asarray(X, dtype=np.float32)
    feats = np.where(rng.random((T, depth)) < 0.5, rng.integers(0, 2, size=(T, depth)), rng.integers(0, X32.shape[1], size=(T, depth)))
    borders = np.zeros((T, depth), dtype=np.float32)
    for t in range(T):
        for i in range(depth):
            col = X32[:, feats[t, i]]
            borders[t, i] = np.float32(np.quantile(col, rng.uniform(0.3, 0.7))) if col.max() > col.min() else np.float32(0.5)
    base = {'task': task, 'depth': depth, 'split_features': feats.tolist(), 'split_borders': borders.tolist(),
            'leaf_values': np.zeros((T, L, dim)).tolist(), 'scale': 1.0, 'bias': [0.0] * dim, 'n_features': len(names),
            'feature_names': list(names)}
    leaves = trees.TreeTeacher(base).route(X, names)
    lv = np.zeros((T, L, dim))
    for t in range(T):
        for l in range(L):
            m = leaves[:, t] == l
            cnt = np.bincount(y[m], minlength=K) + 1.0
            lp = np.log(cnt / cnt.sum())
            lv[t, l] = (lp[1] - lp[0]) / T if task == 'income' else (lp - lp.mean()) / T
    return trees.TreeTeacher({**base, 'leaf_values': lv.tolist()})


def _mock_teacher_fitter(X, names, y, unit, seed, max_trees=None):
    out = {t: {'teacher': mock_teacher(X, y[t], t, names, seed=seed + 11 * i), 'model': None,
               'info': {'mock': True, 'tree_count': 8, 'catboost_seed': None}} for i, t in enumerate(TASKS)}
    return out, {'mock': True}


def _mock_stage_b(seed, tch, X, names, y_inc, y_occ, sex, row_id, arms, on_checkpoint, recipe_overrides=None, log=print):
    """Deterministic invented corrections from FIT rows only (per-leaf SEX contrasts), in the Stage B file format (no torch)."""
    finals, rec = {}, {'seed': seed, 'runs': {}, 'checkpoints': {}, 'final_label': 'E20', 'n': int(len(y_inc)),
                       'epoch_coverage': None, 'mock': True}
    sx = np.asarray(sex, dtype=np.float64) - np.mean(sex)
    for a_i, name in enumerate(arms):
        th = []
        for t in TASKS:
            te = tch[t]
            lv = te.route(X, names)
            g = np.zeros((te.T * te.L, te.dim))
            for k in range(te.T):
                cnt = np.bincount(lv[:, k], minlength=te.L).astype(np.float64)
                tot = np.bincount(lv[:, k], weights=sx, minlength=te.L)
                g[k * te.L:(k + 1) * te.L, :] = (0.3 * (a_i + 1) * tot / np.maximum(cnt, 1.0))[:, None]
            th.append(g)
        rec['checkpoints'][name] = {'E20': on_checkpoint(name, 'E20', th, {'mock': True})}
        finals[name] = th
    return rec, finals, {name: [] for name in arms}


SELFTEST_STAGE_B = {'batch_size': 128, 'steps_per_epoch': 4, 'epochs_per_active_recipient': 2, 'checkpoint_epochs': [0, 2]}
SELFTEST_GRID = ({'rep': 'cmp', 'lr': 0.1, 'max_iter': 30, 'leaves': 8, 'msl': 20, 'l2': 1.0},
                 {'rep': 'ohe', 'lr': 0.1, 'max_iter': 30, 'leaves': 8, 'msl': 20, 'l2': 1.0})
SELFTEST_BETA = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


def _chain(world, names, P, root, have_cb, have_torch):
    """Run the real command functions (teachers, stage_b, income, verify) on an invented world through the fake loader."""
    mod = sys.modules[__name__]
    calls = []
    saved = (mod._load, mod._features)
    mod._load, mod._features = _fake_access(world, names, P, calls)
    try:
        teachers(P, root=root, max_trees=40, fitter=None if have_cb else _mock_teacher_fitter)
        stage_b(P, root=root, recipe_overrides=SELFTEST_STAGE_B, trainer=None if have_torch else _mock_stage_b)
        income(P, root=root, grid=SELFTEST_GRID, beta_grid=SELFTEST_BETA)
        ok = verify(P, root)
    finally:
        mod._load, mod._features = saved
    return calls, ok


def _artifacts(P, root, world):
    """Everything fitted (and the release on rows whose features are not perturbed), as hashes."""
    tch, trec = load_teachers(P, root)
    th, brec = load_thetas(P, root)
    out = {f'teacher_{t}': tch[t].sha() for t in TASKS}
    out.update({f'theta_{a}_{i}': C.arr_sha(np.ascontiguousarray(v[i])) for a, v in th.items() for i in range(2)})
    out['dictionary'] = C.sha(pdir(P, root) / 'TEACHERS' / 'DICT.json')
    for tag in INCOME_CONFIGS:
        _, st, _ = load_income(P, tag, root)
        out[f'state_{tag}'] = json.dumps({k: st[k] for k in ('stack_w', 'platt_w', 'beta', 'hgb_params')}, sort_keys=True)
        for r in ROLES:
            p, _ = release_arrays(P, tag, r, root)
            if r == 'MODEL_FIT':
                p = p[world['MODEL_FIT']['FIT']]
            out[f'release_{tag}_{r}'] = C.arr_sha(np.ascontiguousarray(p))
    hgb, st, _ = load_income(P, 'H3R', root)
    from ihs import fam_h as FH
    from . import trackf as TF
    Xc = TF.recode(world['CALIBRATION']['X'], names_of(world), load_dict(P, root))
    out['hgb_cal_scores'] = C.arr_sha(np.ascontiguousarray(hgb.decision_function(FH.rep_matrix(st['hgb_params']['rep'], Xc, names_of(world)))))
    return out


def names_of(world):
    return world['_names']


def selftest():
    import importlib.util
    from acs_replication import trees
    from ihs import fam_h as FH
    from ohn import common as OC
    from . import trackf as TF
    t0 = time.process_time()
    have_cb = importlib.util.find_spec('catboost') is not None
    have_torch = importlib.util.find_spec('torch') is not None
    names = tuple(OC.teacher(0, 'income').feature_names)          # schema only (stored teacher payload; no person data)
    rng = np.random.default_rng(20261018)
    sizes = {'MODEL_FIT': 1500, 'CALIBRATION': 700, 'MODEL_SELECT': 400, 'ATTACK_FIT': 150, 'ATTACK_VALIDATION': 120, 'DEV_REPORT': 130}
    A = _fake_world(rng, names, sizes); A['_names'] = names
    P = 'R0'
    checks, info = {}, {'catboost_available': have_cb, 'torch_available': have_torch,
                        'teacher_part': 'real CatBoost (40 trees)' if have_cb else 'MOCKED (random oblivious payload fitted on FIT)',
                        'stage_b_part': 'real leafcorr loop (reduced recipe)' if have_torch else 'MOCKED (per-leaf FIT SEX contrasts)'}
    # world B: GUARD / CHECK labels permuted and their features altered; FIT, CALIBRATION and the other roles unchanged
    B = {r: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in W.items()} for r, W in A.items() if r != '_names'}
    B['_names'] = names
    gc = B['MODEL_FIT']['GUARD'] | B['MODEL_FIT']['CHECK']
    for f in LABEL_FIELDS:
        v = B['MODEL_FIT'][f]; v[gc] = np.random.default_rng(7).permutation(1 - v[gc] if f != 'occupation' else (v[gc] + 1) % 5)
    Xb = B['MODEL_FIT']['X']
    Xb[gc, 0] = Xb[gc, 0] + 1.5
    cow = [i for i, nm in enumerate(names) if nm.startswith('COW=')]
    rare = min(cow, key=lambda i: A['MODEL_FIT']['X'][A['MODEL_FIT']['FIT'], i].sum())
    Xb[np.ix_(gc, cow)] = 0.0; Xb[gc, rare] = 1.0                      # a FIT-rare level becomes common in GUARD / CHECK
    info['guard_check_people'] = int(gc.sum()); info['fit_people'] = int(A['MODEL_FIT']['FIT'].sum())
    with tempfile.TemporaryDirectory(prefix='cts_trackr_selftest_') as tmp:
        ra, rb = Path(tmp) / 'A', Path(tmp) / 'B'
        calls, ok_a = _chain(A, names, P, ra, have_cb, have_torch)
        checks['predict_parity_bitwise_all_roles'] = bool(ok_a)
        lab = [c for c in calls if c['labels']]
        checks['label_reads_fit_or_calibration_only'] = bool(lab) and all(
            (c['role'] == 'MODEL_FIT' and c['operation'] == 'fit' and c['prole'] == 'FIT' and c['partition'] == P) or
            (c['role'] == 'CALIBRATION' and c['operation'] == 'calibrate' and 'sex' not in c['labels']) for c in lab)
        checks['sex_read_only_by_stage_b'] = sum('sex' in c['labels'] for c in lab) == 1
        info['label_reads'] = lab
        # zero-correction identity: teacher at theta = 0 (numpy) and the Stage B recipient at theta = 0 (torch, when available)
        tch, trec = load_teachers(P, ra)
        Xf = TF.recode(A['MODEL_FIT']['X'], names, load_dict(P, ra))
        zi = []
        for t in TASKS:
            te = tch[t]; Lv = te.route(Xf, names)
            p0 = te.probabilities_from_raw(te.raw_from_leaves(Lv)); pz = trees.probabilities(t, te.raw_from_leaves(Lv, np.zeros((te.T * te.L, te.dim))))
            zi.append(bool(np.array_equal(p0, pz)))
            if have_torch:
                import torch
                from acs_replication import leafcorr as lc
                rc = lc.Recipient(t, te, Lv, te.raw_from_leaves(Lv))
                with torch.no_grad():
                    pt = torch.softmax(rc.logits(torch.zeros(te.T * te.L, te.dim, dtype=torch.float64), torch.arange(len(Xf))), 1).numpy()
                zi.append(bool(np.max(np.abs(pt - p0)) <= 1e-12))
        checks['zero_correction_identity'] = all(zi)
        brec = json.loads((pdir(P, ra) / 'STAGE_B' / 'STAGE_B.json').read_text())
        checks['stage_b_identity_vs_teacher_outputs'] = bool(not brec.get('zero_identity') is None and all(
            v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in brec['zero_identity'].values()))
        # teacher JSON round trip
        checks['teacher_json_roundtrip'] = all(trees.TreeTeacher(tch[t].payload()).sha() == tch[t].sha() for t in TASKS)
        if have_cb:
            checks['catboost_parity_all_roles'] = all(_parity_ok(v) for t in TASKS for v in trec['teachers'][t]['parity'].values())
        # H3R == HCR when the corrections are zero (the residual is the only difference between the two recipes)
        th, _ = load_thetas(P, ra)
        zero = {a: [np.zeros_like(v[0]), np.zeros_like(v[1])] for a, v in th.items()}
        Fa = A['MODEL_FIT']; fit = Fa['FIT']
        Xc = TF.recode(A['CALIBRATION']['X'], names, load_dict(P, ra))
        hgb0, _, _, st0 = fit_income_arrays(0, Xf[fit], names, Fa['income'][fit], Fa['unit'][fit], tch, zero, Xc, A['CALIBRATION']['income'],
                                            A['CALIBRATION']['unit'], grid=SELFTEST_GRID, beta_grid=SELFTEST_BETA, log=lambda m: None)
        s3 = json.loads(json.dumps(st0['H3R'])); sc = json.loads(json.dumps(st0['HCR']))
        checks['h3r_equals_hcr_at_zero_corrections'] = bool(np.array_equal(income_release(s3, tch, zero, hgb0, Xc, names),
                                                                           income_release(sc, tch, zero, hgb0, Xc, names)))
        # GUARD / CHECK never enter any fit: world B (their labels permuted, features altered) gives identical fitted artifacts
        _, ok_b = _chain(B, names, P, rb, have_cb, have_torch)
        fa, fb = _artifacts(P, ra, A), _artifacts(P, rb, B)
        diff = sorted(k for k in fa if fa[k] != fb.get(k))
        checks['guard_check_never_fit'] = bool(ok_b) and not diff
        info['artifacts_compared'] = len(fa); info['artifacts_differing'] = diff
        # the GUARD / CHECK release does respond to their (altered) features: the comparison is not vacuous
        pa, _ = release_arrays(P, 'H3R', 'MODEL_FIT', ra); pb, _ = release_arrays(P, 'H3R', 'MODEL_FIT', rb)
        checks['altered_guard_check_features_change_their_release'] = bool(not np.array_equal(pa[gc], pb[gc]))
        # retry safety: an incomplete step directory (failed attempt) is moved aside and kept; a completed one is refused
        tdir = Path(tmp) / 'retry' / 'STEP'; tdir.mkdir(parents=True); (tdir / 'partial.npz').write_bytes(b'x')
        moved = _fresh_dir(tdir, 'STEP.json')
        (tdir / 'STEP.json').write_text('{}')
        try:
            _fresh_dir(tdir, 'STEP.json'); refused = False
        except PermissionError:
            refused = True
        checks['retry_moves_incomplete_refuses_complete'] = bool(moved and (tdir.parent / moved / 'partial.npz').is_file() and refused)
        # the recipe's fixed constants are the source ones
        checks['source_recipe_constants'] = bool(FH.N_FOLDS == 5 and FH.BUDGETS['B1'] == {'brier': 0.0045, 'nll': 0.0080, 'accuracy': 0.0035}
                                                 and len(FH.HGB_GRID) == 7 and FH.BETA_GRID[-1] == 2.0 and trees.REGISTRY['max_trees'] == 800)
        st = json.loads((pdir(P, ra) / 'INCOME' / 'STATE_H3R.json').read_text())
        info['invented_beta'] = st['beta']; info['invented_stack_w'] = st['stack_w']
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info, 'cpu_s': round(time.process_time() - t0, 1)},
                     default=str))
    return ok, checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('estimate')
    for name in ('teachers', 'income', 'verify', 'crosscheck'):
        x = sub.add_parser(name); x.add_argument('--partition', required=True, choices=PARTS)
    x = sub.add_parser('stage_b'); x.add_argument('--partition', required=True, choices=PARTS)
    x.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        ok, _ = selftest(); return 0 if ok else 1
    if a.cmd == 'estimate':
        estimate(); return 0
    if a.cmd == 'teachers':
        teachers(a.partition); return 0
    if a.cmd == 'stage_b':
        stage_b(a.partition, arms=tuple(x for x in a.arms.split(',') if x)); return 0
    if a.cmd == 'income':
        income(a.partition); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.partition) else 1
    crosscheck(a.partition)
    return 0


if __name__ == '__main__':
    sys.exit(main())

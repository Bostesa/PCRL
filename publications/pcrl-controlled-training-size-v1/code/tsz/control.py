"""Equally exposed task-only control on a registered training subset: the protected-tree replication's T_TREE teachers followed by
its Stage B TASK_REFIT arm (controlled training-size test, PCRL `pcrl_controlled_training_size_v1`; EXPLORATORY DEVELOPMENT on
previously used records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records (housing-unit residents aged 25-64,
civilian employed). Income (2 classes) and occupation (5 official broad civilian groups) recipients each receive only their own
float64 probability vector and argmax decision, computed from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP) through the
admitted 145-column encoding. SEX is read only as the training label of Stage B's online critics on the subset (as in the source
loop; with lambda = 0 the critics never reach the released parameters); never a runtime input or output.

Per (subset SID, source version v) - DESIGN_BRIEF section 3:
  dictionary  tsz.sizefit.subset_dictionary (the FIT-only re-collapse; identical function and input as the A1_PO_16 fit of the cell)
              applied to every role with cts.trackf.recode.
  teachers    (CatBoost venv) the replication Stage A T_TREE loop through cts.track_r.fit_teachers_arrays, for income AND occupation
              on the subset: CatBoost 1.2.8, depth 4, lr 0.05, l2 3, <= 800 trees, random_seed = trees.seed_for('ACSR1catboost', v,
              task, 4) (unreduced), tree count = first argmin of the loss on the 15% household internal split of the subset
              (acs_replication.data.internal_split, two-task class-support repair, seed v) + 1, refit on the subset; numpy teacher
              checked with trees.verify_against_catboost on all six roles; outputs on all six roles.
  stage_b     (torch venv) the replication Stage B loop (acs_replication.stage_b.run / leafcorr) with the TASK_REFIT arm only
              (source arm T_REFIT, lambda 0; 20 epochs x 40 updates x 256 rows per recipient; AdamW lr 0.01 / wd 0.01; warm critics
              and the 16-draw calibration as in the source loop; probe / reference rows by the replication's label-free hash order
              within the subset). Running TASK_REFIT alone reproduces the replication's multi-arm run (arm order JOINT_TREE,
              LOCAL_TREE, TASK_REFIT): every arm starts from a fresh Trainer whose streams are seeded from v only (checked bitwise
              in the selftest against cts.track_r.run_stage_b_arrays with the replication's three arms).
              Subsets with fewer than 40 x 256 = 10,240 people (every S50 subset: 9,934-10,046 people): the source epoch order
              order[b*256:(b+1)*256] of one permutation would leave the last batch short (S50_O2: 62 rows, S50_O3: 30) or, for
              S50_O0 / S50_O1, one short batch (206 / 229 rows) and one EMPTY batch per epoch (NaN task loss, zero task gradient:
              a momentum-and-weight-decay-only AdamW step); the epoch order is the same permutation (same random stream) extended
              cyclically to 10,240 positions, so
              every update has exactly 256 rows and every subset person is used once before any is reused. For n >= 10,240 the
              unchanged leafcorr.run_arm runs (the cyclic transcription equals it bitwise there; selftest).
  repro       (label-free) S100 teachers vs the admitted teachers ohn.common.teacher(v, task), S100 TASK_REFIT vs the admitted
              Stage B theta and outputs and the frozen bank (TASK_REFIT, T_TREE) on MODEL_SELECT.

Commands (real-data ones only through `python3 -m tsz.jobqueue run ...`):
  selftest                         invented data only (the CatBoost part is real under the CatBoost venv and mocked elsewhere; the
                                   Stage B part is real under the torch venv and mocked elsewhere; recorded)
  timing                           invented data of the real size (label-free), the part this interpreter can run
  teachers --subset SID --version V      CatBoost venv
  stage_b  --subset SID --version V      torch venv
  verify   --subset SID --version V      label-free: predict() from features alone == saved outputs (bitwise, six roles) + refusals
  repro    --version V                   label-free: S100 control of version v vs the admitted source objects
Outputs: .private/<campaign>/runs/CONTROL/<SID>__V<v>/ (write-once).
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C

TASKS = ('income', 'occupation')
ROLES = C.ROLES6
STAGE_DIR = 'CONTROL'
SCHEMA = 'TSZ1-control-v1'
ARM = 'TASK_REFIT'
REPLICATION_ARMS = ('JOINT_TREE', 'LOCAL_TREE', 'TASK_REFIT')     # arm order recorded in the replication's STAGE_B_SEED.json
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
DEVIATIONS = (
    'none at S100 (the replication Stage A T_TREE loop and Stage B TASK_REFIT loop on all MODEL_FIT; the FIT-only dictionary is the '
    'admitted one; checked by `repro`)',
    'subsets: dictionary re-collapsed on the subset; the 15% internal split (with its class-support repair) and the Stage B probe / '
    'reference rows are formed within the subset; each Stage B epoch covers min(10,240, n) / n of the subset (S70: about 73%, S100: '
    '51% as in the replication)',
    'S50 subsets (n < 10,240): cyclic extension of the epoch permutation (see the module docstring); the source slice would give a '
    'short or empty last batch',
    'Stage B arms: TASK_REFIT only (the replication also ran JOINT_TREE and LOCAL_TREE first; each arm is an independent fresh trainer, '
    'so TASK_REFIT is unchanged; selftest)',
    'teachers: CatBoost receives the recoded X as float32 (the replication passed the admitted float32 X; recoded values are exact '
    'float32 values), numpy routing as in the source',
)


# ------------------------------------------------------------------ data access (hooks; the selftest replaces them with fakes)
_NO_REAL_ACCESS = False          # set by selftest / timing: the real label hook refuses (invented data only)


def _load(role, fields, operation, purpose, subset=None):
    if _NO_REAL_ACCESS:
        raise PermissionError('selftest / timing: no real label read')
    return C.load(role, fields, operation, purpose, subset=subset)


def _features(role):
    return C.features(role)


def cell_id(sid, v):
    if sid not in C.subset_ids() or int(v) not in C.VERSIONS:
        raise ValueError(f'unknown cell {sid} v{v}')
    return f'{sid}__V{int(v)}'


def cell_dir(sid, v, root=None):
    return (Path(root) if root else C.RUNS / STAGE_DIR) / cell_id(sid, v)


def _fresh_dir(d, record_name):
    d = Path(d)
    if (d / record_name).exists():
        raise PermissionError(f'{d} is complete (append-only)')
    moved = None
    if d.exists():
        moved = d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}')
        d.rename(moved)
    d.mkdir(parents=True, mode=0o700)
    return None if moved is None else moved.name


def _save_npz(path, **arrs):
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return C.sha(path)


def _load_npz(path, sha=None):
    if sha is not None and C.sha(path) != sha:
        raise PermissionError(f'{Path(path).name} differs from its record')
    return C.load_npz(path)


def _write_once(path, obj):
    C.write_json_x(path, obj)
    return C.sha(path)


def _interp():
    import importlib.metadata as m
    out = {'python': sys.version.split()[0], 'executable': sys.executable}
    for p in ('numpy', 'scipy', 'scikit-learn', 'torch', 'catboost'):
        try:
            out[p] = m.version(p)
        except Exception:
            out[p] = None
    return out


def role_features(dic):
    from . import sizefit as SF
    out = {}
    for r in ROLES:
        F = _features(r)
        names = tuple(map(str, F['feature_names']))
        out[r] = {'X': SF.recode(F['X'], names, dic), 'names': names, 'row_id': F['row_id'], 'unit': F['unit']}
    return out


# ------------------------------------------------------------------ teachers (CatBoost venv)
def _parity_ok(v):
    return v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12


def teachers(sid, v, root=None, max_trees=None, fitter=None, log=print):
    from acs_replication import trees
    from cts import track_r as TR
    from . import sizefit as SF
    t0 = time.process_time(); w0 = time.time()
    v = int(v)
    d = cell_dir(sid, v, root) / 'TEACHERS'
    if (d / 'TEACHERS.json').exists():
        raise PermissionError(f'{cell_id(sid, v)} teachers exist (append-only)')
    if max_trees is not None and root is None:
        raise PermissionError('a reduced tree cap is for the invented-data selftest only')
    F = _load('MODEL_FIT', ('features', 'income', 'occupation'), 'fit',
              f'TSZ1 control {cell_id(sid, v)}: replication T_TREE recipe (CatBoost 1.2.8) for income and occupation on subset {sid} '
              '(15% household internal split of the subset with the two-task class-support repair, refit on the subset); no SEX',
              subset=sid)
    names = tuple(map(str, F['feature_names']))
    dic = SF.subset_dictionary(F['X'], names, 'fit_only')
    X64 = SF.recode(F['X'], names, dic)
    X = X64.astype(np.float32)
    if not np.array_equal(X.astype(np.float64), X64):
        raise AssertionError('recoded features are not exact float32 values')
    y = {t: np.asarray(F[t], dtype=np.int64) for t in TASKS}
    mocked = fitter is not None
    fitted, split = (fitter or TR.fit_teachers_arrays)(X, names, y, F['unit'], v, max_trees=max_trees)
    roles = role_features(dic)
    incomplete = _fresh_dir(d, 'TEACHERS.json')
    files = {'DICT.json': _write_once(d / 'DICT.json', dic)}
    outputs, rec_t = {}, {}
    for task in TASKS:
        teacher, model = fitted[task]['teacher'], fitted[task]['model']
        parity = None
        if model is not None:
            parity = {r: trees.verify_against_catboost(teacher, model, roles[r]['X'], names) for r in ROLES}
            if not all(_parity_ok(x) for x in parity.values()):
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
    rec = {'schema': SCHEMA + '-teachers', 'cell': cell_id(sid, v), 'subset': sid, 'version': v, 'mocked': mocked,
           'recipe': 'acs_replication.stage_a T_TREE loop (via cts.track_r.fit_teachers_arrays) on the subset, FIT-only dictionary; '
                     'unreduced CatBoost seeds', 'catboost_version': trees.CATBOOST_VERSION, 'registry': trees.REGISTRY,
           'internal_split': split, 'n_people': int(len(F['row_id'])), 'n_households': int(len(np.unique(F['unit']))),
           'rows_sha256': C.arr_sha(np.asarray(F['row_id'])),
           'class_counts': {t: [int(c) for c in np.bincount(y[t], minlength=trees.CLASSES[t])] for t in TASKS},
           'dictionary': {'kind': 'fit_only', 'min_count': dic['min_count'], 'n_collapsed': dic['n_collapsed']},
           'teachers': rec_t, 'files_sha256': files, 'previous_incomplete_attempt': incomplete, 'deviations': list(DEVIATIONS),
           'labels_read': f'MODEL_FIT {sid}: income, occupation (fit); no SEX', 'interpreter': _interp(),
           'code_sha256': {'tsz/control.py': C.sha(Path(__file__)), 'cts/track_r.py': C.sha(Path(TR.__file__))},
           'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'TEACHERS.json', rec)
    C.unit_stats(n_model_fits=4, n_teacher_fits=4)          # 2 tasks x (internal-split fit + refit)
    log(json.dumps({'cell': cell_id(sid, v), 'tree_counts': {t: rec_t[t].get('tree_count') for t in TASKS}, 'split': split,
                    'cpu_s': round(rec['cpu_s'], 1)}))
    return rec


def load_dict(sid, v, root=None):
    d = cell_dir(sid, v, root) / 'TEACHERS'
    rec = json.loads((d / 'TEACHERS.json').read_text())
    if C.sha(d / 'DICT.json') != rec['files_sha256']['DICT.json']:
        raise PermissionError('DICT.json differs from its record')
    return json.loads((d / 'DICT.json').read_text())


def load_teachers(sid, v, root=None):
    from acs_replication import trees
    d = cell_dir(sid, v, root) / 'TEACHERS'
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


# ------------------------------------------------------------------ Stage B (torch venv)
def cyclic_order(perm, length):
    """The epoch's permutation extended cyclically to `length` positions (identity on the first min(n, length) positions)."""
    perm = np.asarray(perm)
    return perm[:length].copy() if len(perm) >= length else np.resize(perm, length)


def run_arm_cyclic(trainer, arm, rho, coefficients, on_checkpoint, log_every=20, counterfactual_steps=0):
    """acs_replication.leafcorr.run_arm transcribed line by line; ONE change (marked): the epoch order is the permutation extended
    cyclically to steps_per_epoch x batch_size positions, so that every update has a full batch when the pool is smaller than that.
    For pools of at least steps_per_epoch x batch_size people it equals leafcorr.run_arm bitwise (selftest)."""
    import torch
    from acs_replication import leafcorr as lc
    RECIPE = lc.RECIPE
    rule = RECIPE['arms'][arm]
    critics = trainer.warm(); warm_sha = critics.state_sha()
    lam = [0.0, 0.0] if rule['privacy'] is None else coefficients[str(rho)][rule['coefficient']]
    phases = [[0, 1]] if rule['schedule'] == 'simultaneous' else [[rule['order'][0]], [rule['order'][1]]]
    opts = [torch.optim.AdamW([t], lr=RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=RECIPE['weight_decay'])
            for t in trainer.theta]
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = lc.random.Random(lc.seed_for('THP1privacy', trainer.seed)); critic_rng = lc.random.Random(lc.seed_for('THP1critic-rounds', trainer.seed))
    logs, counterfactuals, step, updates = [], [], 0, [0, 0]
    for pi, active in enumerate(phases):
        prefix = '' if len(phases) == 1 else f'P{pi + 1}'
        frozen = {i: trainer.theta[i].detach().clone() for i in range(2) if i not in active}
        for epoch in range(RECIPE['epochs_per_active_recipient'] + 1):
            trainer.refresh(critics)
            if epoch in RECIPE['checkpoint_epochs']:
                on_checkpoint(f'{prefix}E{epoch}', trainer.snapshot(), {'updates_per_recipient': updates[:], 'critic_rounds': critics.rounds,
                                                                       'phase_active': active, 'step': step})
            if epoch == RECIPE['epochs_per_active_recipient']:
                break
            order = cyclic_order(task_rng.permutation(trainer.n), RECIPE['steps_per_epoch'] * RECIPE['batch_size'])   # TSZ1-CHANGE
            for b in range(RECIPE['steps_per_epoch']):
                task_rows = torch.from_numpy(order[b * RECIPE['batch_size']:(b + 1) * RECIPE['batch_size']])
                pr, ps = trainer.privacy_batch(priv_rng)
                grads, info = {}, {}
                if rule['privacy'] is not None:
                    w = trainer.wires(pr); terms = critics.terms(w[0], w[1], ps); P = lc.objective(terms, rule['privacy'])
                    pg = torch.autograd.grad(P, [trainer.theta[i] for i in active], allow_unused=True)
                    pg = [torch.zeros_like(trainer.theta[i]) if g is None else g for g, i in zip(pg, active)]
                    info['privacy_objective'] = float(P.detach()); info['risk'] = {v: float(terms[v].detach()) for v in lc.VIEWS}
                else:
                    pg = [torch.zeros_like(trainer.theta[i]) for i in active]
                for g_p, i in zip(pg, active):
                    g_t = torch.autograd.grad(trainer.ce(i, task_rows), trainer.theta[i])[0]
                    grads[i] = (g_t, g_p)
                if step % log_every == 0:
                    logs.append({'step': step, 'phase': pi, 'active': active, **info,
                                 'task_grad_norm': {i: float(grads[i][0].norm()) for i in active},
                                 'privacy_grad_norm': {i: float(grads[i][1].norm()) for i in active},
                                 'weighted_privacy_grad_norm': {i: float(lam[i] * grads[i][1].norm()) for i in active},
                                 'effective_ratio': {i: (float(lam[i] * grads[i][1].norm()) / max(float(grads[i][0].norm()), 1e-300)) for i in active}})
                record_cf = counterfactual_steps and rule['privacy'] is not None and step < counterfactual_steps
                if record_cf:
                    before = {i: (trainer.theta[i].detach().clone(), copy.deepcopy(opts[i].state_dict())) for i in active}
                for i in active:
                    g_t, g_p = grads[i]
                    opts[i].zero_grad(set_to_none=True)
                    trainer.theta[i].grad = g_t + lam[i] * g_p
                    opts[i].step(); updates[i] += 1
                if record_cf:
                    counterfactuals.append(lc._counterfactual(trainer, active, before, grads, lam, step))
                for i, t in frozen.items():
                    if not torch.equal(trainer.theta[i].detach(), t):
                        raise AssertionError('inactive recipient changed')
                for _ in range(RECIPE['critic_rounds_per_recipient_update'] * len(active)):
                    trainer.critic_round(critics, critic_rng)
                step += 1
    return {'logs': logs, 'counterfactuals': counterfactuals, 'warm_critic_sha256': warm_sha, 'critic_rounds': critics.rounds,
            'final_critic_sha256': critics.state_sha(), 'lambda': lam, 'steps': step, 'updates_per_recipient': updates}


def run_stage_b_arrays(seed, tch, X, names, y_inc, y_occ, sex, row_id, arms, on_checkpoint, recipe_overrides=None, log=print):
    """cts.track_r.run_stage_b_arrays (= acs_replication.stage_b.run's sequence) without its full-batch guard: pools with
    n >= steps x batch use the unchanged leafcorr.run_arm, smaller pools run_arm_cyclic."""
    from acs_replication import data as AD, leafcorr as lc
    from acs_replication.stage_b import ARMS
    from cts import track_r as TR
    with TR._recipe(recipe_overrides):
        n = len(y_inc); B = lc.RECIPE['batch_size']; steps = lc.RECIPE['steps_per_epoch']
        runner = lc.run_arm if n >= steps * B else run_arm_cyclic
        tl = [tch[t] for t in TASKS]
        leaves = [tl[i].route(X, names) for i in range(2)]
        upd = np.arange(n)
        probe = AD.order_rows(row_id, 'ACSR1-probe', 512); reference = AD.order_rows(row_id, 'ACSR1-reference', 256)
        y = [np.ascontiguousarray(y_inc, dtype=np.int64), np.ascontiguousarray(y_occ, dtype=np.int64)]
        sex = np.ascontiguousarray(sex, dtype=np.int64)

        def make():
            rec = [lc.Recipient(t, tl[i], leaves[i], tl[i].raw_from_leaves(leaves[i])) for i, t in enumerate(TASKS)]
            return lc.Trainer(seed, rec, y, upd, sex, probe, reference)
        tr0 = make(); critics = tr0.warm(); warm_sha = critics.state_sha()
        try:
            calib = tr0.calibrate(critics)
        except (ZeroDivisionError, FloatingPointError) as e:
            # the 16-draw calibration only sets the privacy coefficients, which TASK_REFIT (lambda 0) never reads; the source loop
            # would stop here. Recorded, and allowed only for a TASK_REFIT-only run.
            if tuple(arms) != (ARM,):
                raise
            calib = {'failed': repr(e)[:200], 'coefficients': None,
                     'note': 'calibration failed (a privacy-gradient median was zero or a coefficient left the registered range); '
                             'TASK_REFIT does not use the coefficients'}
        final = f"E{lc.RECIPE['epochs_per_active_recipient']}"
        slice_sizes = [int(max(0, min(n, (b + 1) * B) - b * B)) for b in range(steps)]
        record = {'seed': seed, 'calibration': calib, 'warm_critic_sha256': warm_sha, 'runs': {}, 'checkpoints': {},
                  'probe_rows_sha256': C.arr_sha(probe), 'reference_rows_sha256': C.arr_sha(reference), 'final_label': final,
                  'n': int(n), 'epoch_coverage': min(steps * B, n) / n,
                  'epoch_order': 'source slice of one permutation (leafcorr.run_arm unchanged)' if runner is lc.run_arm else
                                 'cyclic extension of the epoch permutation (run_arm_cyclic)',
                  'source_slice_batches_not_full': {str(b): s for b, s in enumerate(slice_sizes) if s < B},
                  'source_slice_empty_batches': int(sum(s == 0 for s in slice_sizes)),
                  'people_used_twice_per_epoch': int(max(steps * B - n, 0)),
                  'recipe': {k: lc.RECIPE[k] for k in ('batch_size', 'steps_per_epoch', 'epochs_per_active_recipient', 'checkpoint_epochs',
                                                       'warm_critic_rounds', 'critic_rounds_per_recipient_update', 'learning_rate',
                                                       'weight_decay')}, 'recipe_overrides': recipe_overrides or {}}
        finals, logs = {}, {}
        for name in arms:
            saved = {}

            def on_ck(label, thetas, info, name=name, saved=saved):
                saved[label] = on_checkpoint(name, label, thetas, info)
                if label == final:
                    finals[name] = [np.array(t, dtype=np.float64, copy=True) for t in thetas]
            tr = make()
            t0 = time.process_time()
            res = runner(tr, ARMS[name], 1.0, calib['coefficients'], on_ck, counterfactual_steps=8 if name == 'LOCAL_TREE' else 0)
            if res['warm_critic_sha256'] != warm_sha:
                raise AssertionError('warm critics differ across runs')
            record['runs'][name] = {**{k: v for k, v in res.items() if k != 'logs'}, 'cpu_s': time.process_time() - t0}
            record['checkpoints'][name] = saved
            logs[name] = res['logs']
            log(json.dumps({'arm': name, 'lambda': res['lambda'], 'steps': res['steps'], 'cpu_s': round(record['runs'][name]['cpu_s'], 1)}))
        if set(finals) != set(arms):
            raise AssertionError('a final checkpoint is missing')
    return record, finals, logs


def stage_b(sid, v, root=None, recipe_overrides=None, trainer=None, log=print):
    from acs_replication import trees
    from . import sizefit as SF
    t0 = time.process_time(); w0 = time.time()
    v = int(v)
    d = cell_dir(sid, v, root) / 'STAGE_B'
    if (d / 'STAGE_B.json').exists():
        raise PermissionError(f'{cell_id(sid, v)} Stage B exists (append-only)')
    if recipe_overrides and root is None:
        raise PermissionError('recipe overrides are for the invented-data selftest only')
    tch, trec = load_teachers(sid, v, root)
    dic = load_dict(sid, v, root)
    stored = _load_npz(cell_dir(sid, v, root) / 'TEACHERS' / 'TEACHER_OUTPUTS.npz', trec['files_sha256']['TEACHER_OUTPUTS.npz'])
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
    if not all(x['max_abs_prob_diff'] <= 1e-12 and x['decisions_identical'] for x in identity.values()):
        raise AssertionError(f'zero-correction identity failed: {identity}')
    F = _load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit',
              f'TSZ1 control {cell_id(sid, v)}: replication Stage B TASK_REFIT arm (lambda 0) on the subset teachers; task losses on '
              f'subset {sid}; SEX only for the source loop\'s warm / online critics (never reach the released parameters)', subset=sid)
    names = tuple(map(str, F['feature_names']))
    if not np.array_equal(F['row_id'], roles['MODEL_FIT']['row_id'][F['rows']]):
        raise AssertionError('subset rows differ from the MODEL_FIT feature rows')
    if C.arr_sha(np.asarray(F['row_id'])) != trec['rows_sha256']:
        raise AssertionError("Stage B rows differ from the teachers' rows")
    X = SF.recode(F['X'], names, dic)
    incomplete = _fresh_dir(d, 'STAGE_B.json')
    files = {}

    def on_checkpoint(name, label, thetas, info):
        h = _save_npz(d / f'{name}__{label}.npz', theta_income=thetas[0], theta_occupation=thetas[1])
        files[f'{name}__{label}.npz'] = h
        return {'file': f'{name}__{label}.npz', 'sha256': h, **info}
    mocked = trainer is not None
    record, finals, logs = (trainer or run_stage_b_arrays)(v, tch, X, names, F['income'], F['occupation'], F['sex'], F['row_id'], (ARM,),
                                                           on_checkpoint, recipe_overrides=recipe_overrides)
    if root is None and (record['final_label'] != 'E20' or recipe_overrides):
        raise AssertionError('real control runs use the unchanged Stage B recipe (final checkpoint E20)')
    outputs = {}
    th = finals[ARM]
    for r in ROLES:
        for i, t in enumerate(TASKS):
            te = tch[t]
            outputs[f'{ARM}__{r}__{t}'] = te.probabilities_from_raw(te.raw_from_leaves(te.route(roles[r]['X'], roles[r]['names']), th[i]))
    files['STAGE_B_OUTPUTS_FINAL.npz'] = _save_npz(d / 'STAGE_B_OUTPUTS_FINAL.npz', **outputs, **{f'row_id__{r}': roles[r]['row_id'] for r in ROLES})
    with open(d / 'LOGS.json', 'x') as f:
        json.dump(logs, f, default=float)
    files['LOGS.json'] = C.sha(d / 'LOGS.json')
    rec = {'schema': SCHEMA + '-stage-b', 'cell': cell_id(sid, v), 'subset': sid, 'version': v, 'mocked': mocked, 'arms': [ARM],
           'zero_identity': identity, **record, 'teachers_sha256': {t: trec['teachers'][t]['teacher_sha256'] for t in TASKS},
           'rows_sha256': C.arr_sha(np.asarray(F['row_id'])), 'files_sha256': files, 'previous_incomplete_attempt': incomplete,
           'release_sha256': {k: C.arr_sha(a) for k, a in outputs.items()}, 'deviations': list(DEVIATIONS),
           'labels_read': f'MODEL_FIT {sid}: income, occupation, SEX (training-time critics of the source loop; lambda 0)',
           'interpreter': _interp(), 'code_sha256': {'tsz/control.py': C.sha(Path(__file__))},
           'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'STAGE_B.json', rec)
    C.unit_stats(n_model_fits=1, n_correction_fits=1, n_critic_fits=18)   # TASK_REFIT; 9 calibration-warm + 9 arm critics
    log(json.dumps({'cell': cell_id(sid, v), 'n': record['n'], 'epoch_order': record['epoch_order'], 'cpu_s': round(rec['cpu_s'], 1)}, default=float))
    return rec


def load_theta(sid, v, root=None):
    d = cell_dir(sid, v, root) / 'STAGE_B'
    rec = json.loads((d / 'STAGE_B.json').read_text())
    ck = rec['checkpoints'][ARM][rec['final_label']]
    z = _load_npz(d / ck['file'], ck['sha256'])
    return [z['theta_income'], z['theta_occupation']], rec


# ------------------------------------------------------------------ deployment (numpy only)
def _check_inputs(X, names, expected):
    names = tuple(map(str, names))
    if names != tuple(expected):
        raise ValueError('wrong or reordered input schema')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ValueError('finite input matrix of the registered width required')
    for b in BLOCKS:
        Z = X[:, [i for i, n in enumerate(names) if n.startswith(b + '=')]]
        if not (np.all((Z == 0) | (Z == 1)) and np.all(Z.sum(1) == 1)):
            raise ValueError(f'{b}: one-hot block is not complete')
    return X


def predict(sid, v, X, names, root=None, arm=ARM):
    """{'income': (n, 2), 'occupation': (n, 5)} float64 releases of the control (TASK_REFIT, or T_TREE with arm='T_TREE') from
    admitted features alone."""
    from . import sizefit as SF
    tch, trec = load_teachers(sid, v, root)
    dic = load_dict(sid, v, root)
    X = _check_inputs(X, names, tch['occupation'].feature_names)
    Xr = SF.recode(X, names, dic)
    th = load_theta(sid, v, root)[0] if arm == ARM else [None, None]
    return {t: tch[t].probabilities_from_raw(tch[t].raw_from_leaves(tch[t].route(Xr, names), th[i])) for i, t in enumerate(TASKS)}


def verify(sid, v, root=None):
    d = cell_dir(sid, v, root)
    _, trec = load_teachers(sid, v, root)
    _, brec = load_theta(sid, v, root)
    tout = _load_npz(d / 'TEACHERS' / 'TEACHER_OUTPUTS.npz', trec['files_sha256']['TEACHER_OUTPUTS.npz'])
    bout = _load_npz(d / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz', brec['files_sha256']['STAGE_B_OUTPUTS_FINAL.npz'])
    res = {}
    for r in ROLES:
        F = _features(r)
        if not np.array_equal(F['row_id'], bout[f'row_id__{r}']):
            raise AssertionError(f'{r}: row order differs from the saved outputs')
        for arm, store, key in ((ARM, bout, ARM), ('T_TREE', tout, 'T_TREE')):
            p = predict(sid, v, F['X'], F['feature_names'], root, arm=arm)
            for t in TASKS:
                ref = store[f'{key}__{r}__{t}']
                res[f'{arm}|{r}|{t}'] = {'bitwise': bool(np.array_equal(p[t], ref)), 'max_abs_diff': float(np.max(np.abs(p[t] - ref)))}
    Fm = _features('MODEL_SELECT'); names = tuple(map(str, Fm['feature_names'])); X = Fm['X'][:3]
    i_cow = [i for i, n in enumerate(names) if n.startswith('COW=')]
    sw = list(names); sw[i_cow[0]], sw[i_cow[1]] = sw[i_cow[1]], sw[i_cow[0]]
    bad = X.copy(); bad[0, i_cow] = 0.0
    nan = X.astype(np.float64).copy(); nan[0, 0] = np.nan
    probes = {}
    for k, (Xp, nm) in {'reordered_names': (X, tuple(sw)), 'incomplete_one_hot_block': (bad, names), 'non_finite': (nan, names),
                        'wrong_width': (X[:, :-1], names[:-1])}.items():
        try:
            predict(sid, v, Xp, nm, root); probes[k] = {'refused': False}
        except ValueError as e:
            probes[k] = {'refused': True, 'error': repr(e)[:160]}
    size_dict = C.RUNS / 'SIZE' / f'A1__{sid}__V{int(v)}' / 'DICT.json' if root is None else Path(root) / 'SIZE' / f'A1__{sid}__V{int(v)}' / 'DICT.json'
    cross = {'size_fit_dict_present': size_dict.is_file()}
    if size_dict.is_file():
        cross['identical_dictionary_file'] = C.sha(size_dict) == trec['files_sha256']['DICT.json']
    ok = all(x['bitwise'] for x in res.values()) and all(x['refused'] for x in probes.values()) and cross.get('identical_dictionary_file', True)
    out = {'schema': SCHEMA + '-verify', 'cell': cell_id(sid, v), 'ok': ok, 'outputs': res, 'refusals': probes, 'dictionary_crosscheck': cross,
           'interpreter': _interp(), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    _write_once(d / 'VERIFY.json', out)
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'cell': cell_id(sid, v), 'all_outputs_bitwise': all(x['bitwise'] for x in res.values()),
                      'refusals': {k: x['refused'] for k, x in probes.items()}, 'dictionary_crosscheck': cross}))
    return ok


# ------------------------------------------------------------------ source reproduction (label-free, numpy only)
def _cmp(a, b):
    same = a.shape == b.shape
    return {'bitwise': bool(same and np.array_equal(a, b)), 'max_abs_diff': float(np.max(np.abs(a - b))) if same else None,
            'decision_agreement': float((a.argmax(1) == b.argmax(1)).mean()) if same else None}


def _source_objects(v):
    from ohn import common as OC
    bank, _ = OC.frozen_bank(v, roles=('MODEL_SELECT',))
    return {'teachers': {t: OC.teacher(v, t) for t in TASKS}, 'stage_a': OC.stage_a_outputs(v), 'stage_b': OC.stage_b_outputs(v),
            'theta': OC.stage_b_theta(v, 'TASK_REFIT'), 'bank': bank}


def repro(v, root=None, source=None):
    v = int(v)
    sid = 'S100'
    tch, trec = load_teachers(sid, v, root)
    th, brec = load_theta(sid, v, root)
    d = cell_dir(sid, v, root)
    tout = _load_npz(d / 'TEACHERS' / 'TEACHER_OUTPUTS.npz', trec['files_sha256']['TEACHER_OUTPUTS.npz'])
    bout = _load_npz(d / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz', brec['files_sha256']['STAGE_B_OUTPUTS_FINAL.npz'])
    S = (source or _source_objects)(v)
    out = {'schema': SCHEMA + '-repro', 'version': v, 'teachers': {}, 'stage_a_outputs': {}, 'stage_b_outputs': {}, 'bank_MODEL_SELECT': {}}
    for t in TASKS:
        mine, adm = tch[t], S['teachers'][t]
        e = {'mine_sha256': mine.sha(), 'admitted_sha256': adm.sha(), 'identical_payload': mine.sha() == adm.sha(),
             'tree_count': {'mine': mine.T, 'admitted': adm.T}, 'routing_identical': {}}
        for r in ROLES:
            F = _features(r)
            e['routing_identical'][r] = bool(mine.T == adm.T and np.array_equal(
                mine.route(load_recoded(F, sid, v, root), F['feature_names']), adm.route(F['X'], F['feature_names'])))
            key = f'T_TREE__{r}__{t}'
            if key in S['stage_a']:
                out['stage_a_outputs'][f'{r}|{t}'] = _cmp(tout[key], S['stage_a'][key])
            key = f'TASK_REFIT__{r}__{t}'
            if key in S['stage_b']:
                out['stage_b_outputs'][f'{r}|{t}'] = _cmp(bout[key], S['stage_b'][key])
        out['teachers'][t] = e
    out['theta_E20'] = {t: {'bitwise': bool(th[i].shape == S['theta'][i].shape and np.array_equal(th[i], S['theta'][i])),
                            'max_abs_diff': float(np.max(np.abs(th[i] - S['theta'][i]))) if th[i].shape == S['theta'][i].shape else None}
                        for i, t in enumerate(TASKS)}
    for svc in ('TASK_REFIT', 'T_TREE'):
        store, key = (bout, 'TASK_REFIT') if svc == 'TASK_REFIT' else (tout, 'T_TREE')
        for t in TASKS:
            ref = S['bank'].get(svc, {}).get('MODEL_SELECT', {}).get(t)
            if ref is not None:
                out['bank_MODEL_SELECT'][f'{svc}|{t}'] = _cmp(store[f'{key}__MODEL_SELECT__{t}'], ref)
    out['primary'] = {'TASK_REFIT_occupation_MODEL_SELECT_vs_bank': out['bank_MODEL_SELECT'].get('TASK_REFIT|occupation'),
                      'T_TREE_occupation_MODEL_SELECT_vs_bank': out['bank_MODEL_SELECT'].get('T_TREE|occupation')}
    out['reproduced'] = bool(all(e['identical_payload'] for e in out['teachers'].values())
                             and all(x['bitwise'] for x in out['theta_E20'].values())
                             and all(x['bitwise'] for x in out['bank_MODEL_SELECT'].values()) and len(out['bank_MODEL_SELECT']) >= 2)
    adm_p = C.RESULTS / 'SOURCE_ADMISSION.json'
    if source is None and adm_p.is_file():        # tsz.admit record of the admitted teachers this comparison used
        ru = json.loads(adm_p.read_text()).get('releases_used', {})
        out['source_admission_check'] = {t: ru.get(f'T_TREE/{t}/S{v}', {}).get('teacher_sha256') == S['teachers'][t].sha() for t in TASKS}
        out['source_admission_check']['admission_sha256'] = C.sha(adm_p)
    else:
        out['source_admission_check'] = None
    out.update({'stage_b_epoch_order': brec.get('epoch_order'), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'interpreter': _interp()})
    _write_once(d / 'REPRO.json', out)
    print(json.dumps({'repro': 'REPRODUCED' if out['reproduced'] else 'NOT_REPRODUCED', 'version': v,
                      'teachers_identical': {t: e['identical_payload'] for t, e in out['teachers'].items()},
                      'theta_E20_bitwise': {t: x['bitwise'] for t, x in out['theta_E20'].items()}, 'bank': out['bank_MODEL_SELECT']}))
    return out


def load_recoded(F, sid, v, root=None):
    from . import sizefit as SF
    return SF.recode(F['X'], F['feature_names'], load_dict(sid, v, root))


# ------------------------------------------------------------------ invented data (selftest, timing)
def _fake(world, names, calls):
    from . import sizefit as SF
    return SF._fake_access(world, names, calls)


@contextlib.contextmanager
def _fakes(world, names, calls):
    load, features = _fake(world, names, calls)
    mod = sys.modules[__name__]
    from . import sizefit as SF
    saved = (mod._load, mod._features, SF._features)
    mod._load, mod._features, SF._features = load, features, features
    try:
        yield
    finally:
        mod._load, mod._features, SF._features = saved


SELFTEST_STAGE_B = {'batch_size': 128, 'steps_per_epoch': 4, 'epochs_per_active_recipient': 2, 'checkpoint_epochs': [0, 2]}


def _mock_trainer(seed, tch, X, names, y_inc, y_occ, sex, row_id, arms, on_checkpoint, recipe_overrides=None, log=print):
    """Deterministic invented TASK_REFIT-shaped corrections (no torch): per-leaf occupation / income label contrasts on the subset."""
    finals, rec = {}, {'seed': seed, 'runs': {}, 'checkpoints': {}, 'final_label': 'E20', 'n': int(len(y_inc)), 'epoch_coverage': None,
                       'epoch_order': 'MOCK', 'mock': True}
    ys = {'income': np.asarray(y_inc, dtype=np.float64), 'occupation': np.asarray(y_occ, dtype=np.float64)}
    for name in arms:
        th = []
        for t in TASKS:
            te = tch[t]; lv = te.route(X, names); g = np.zeros((te.T * te.L, te.dim))
            for k in range(te.T):
                cnt = np.bincount(lv[:, k], minlength=te.L).astype(np.float64)
                tot = np.bincount(lv[:, k], weights=ys[t] - ys[t].mean(), minlength=te.L)
                g[k * te.L:(k + 1) * te.L, :] = (0.2 * tot / np.maximum(cnt, 1.0))[:, None]
            th.append(g)
        rec['checkpoints'][name] = {'E20': on_checkpoint(name, 'E20', th, {'mock': True})}
        finals[name] = th
    return rec, finals, {name: [] for name in arms}


def selftest():
    from acs_replication import trees, data as AD
    from cts import track_r as TR
    from . import sizefit as SF
    global _NO_REAL_ACCESS
    _NO_REAL_ACCESS = True                 # invented data only: the real label hook refuses (every load goes through the fakes)
    SF._NO_REAL_ACCESS = True
    t0 = time.process_time()
    have_cb = importlib.util.find_spec('catboost') is not None
    have_torch = importlib.util.find_spec('torch') is not None
    names = SF._names()
    rng = np.random.default_rng(20261019)
    sizes = {'MODEL_FIT': 1500, 'CALIBRATION': 300, 'MODEL_SELECT': 400, 'ATTACK_FIT': 120, 'ATTACK_VALIDATION': 110, 'DEV_REPORT': 130}
    A = SF._fake_world(rng, names, sizes)
    W = A['MODEL_FIT']
    checks, info = {}, {'catboost_available': have_cb, 'torch_available': have_torch,
                        'teacher_part': 'real CatBoost 1.2.8 (40-tree cap)' if have_cb else 'MOCKED (cts.track_r.mock_teacher, T = 8)',
                        'stage_b_part': 'real leafcorr loop (reduced recipe 2 x 4 x 128)' if have_torch else 'MOCKED (per-leaf label contrasts)'}
    fitter = None if have_cb else TR._mock_teacher_fitter
    trainer = None if have_torch else _mock_trainer
    calls = []
    with tempfile.TemporaryDirectory(prefix='tsz_control_selftest_') as tmp:
        root = Path(tmp)
        with _fakes(A, names, calls):
            for sid in ('S100', 'S50_O0'):
                teachers(sid, 0, root=root, max_trees=40, fitter=fitter, log=lambda s: None)
                stage_b(sid, 0, root=root, recipe_overrides=SELFTEST_STAGE_B, trainer=trainer, log=lambda s: None)
                checks[f'verify_bitwise_and_refusals_{sid}'] = bool(verify(sid, 0, root))
            m50 = C.subset_mask(W['unit'], 'S50_O0')
            checks['dictionary_identical_to_sizefit_function'] = load_dict('S50_O0', 0, root) == SF.subset_dictionary(W['X'][m50], names)
            tch100, trec100 = load_teachers('S100', 0, root)
            if have_cb:
                checks['catboost_parity_all_roles'] = all(_parity_ok(x) for t in TASKS for x in trec100['teachers'][t]['parity'].values())
                # S100 equals the replication Stage A T_TREE loop transcribed from acs_replication.stage_a.run on the admitted X
                y = {t: W[t] for t in TASKS}
                fr, vr, _ = AD.internal_split(W['unit'], y, 0)
                same = []
                for t in TASKS:
                    rs = trees.seed_for('ACSR1catboost', 0, t, 4)
                    m = trees.make_classifier(t, 4, 40, rs)
                    m.fit(W['X'][fr], y[t][fr], eval_set=(W['X'][vr], y[t][vr]))
                    cnt = int(np.argmin(m.get_evals_result()['validation'][trees.LOSS[t]])) + 1
                    fin = trees.make_classifier(t, 4, cnt, rs); fin.fit(W['X'], y[t])
                    same.append(trees.from_catboost(fin, t, 4, names).sha() == tch100[t].sha())
                checks['s100_teachers_equal_stage_a_loop'] = all(same)
            checks['label_reads_policy'] = all(
                c['role'] == 'MODEL_FIT' and c['operation'] == 'fit' and c['subset'] in C.subset_ids() and 'sex' not in c['labels'] or
                c['role'] == 'MODEL_FIT' and c['operation'] == 'fit' and set(c['labels']) == {'income', 'occupation', 'sex'}
                for c in calls if c['labels'])
            checks['sex_read_only_by_stage_b'] = sum('sex' in c['labels'] for c in calls) == 2
            # repro logic against invented "admitted" objects built from the S100 control itself (and one perturbed copy)
            th, _ = load_theta('S100', 0, root)
            tout = C.load_npz(cell_dir('S100', 0, root) / 'TEACHERS' / 'TEACHER_OUTPUTS.npz')
            bout = C.load_npz(cell_dir('S100', 0, root) / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz')
            bank = {svc: {'MODEL_SELECT': {t: (bout if svc == 'TASK_REFIT' else tout)[f'{svc}__MODEL_SELECT__{t}'] for t in TASKS}}
                    for svc in ('TASK_REFIT', 'T_TREE')}
            src = {'teachers': tch100, 'stage_a': tout, 'stage_b': bout, 'theta': th, 'bank': bank}
            rp = repro(0, root, source=lambda v: src)
            bad = copy.deepcopy(bank); bad['TASK_REFIT']['MODEL_SELECT']['occupation'][0] = bad['TASK_REFIT']['MODEL_SELECT']['occupation'][0][::-1]
            (cell_dir('S100', 0, root) / 'REPRO.json').rename(cell_dir('S100', 0, root) / 'REPRO_first.json')
            rp2 = repro(0, root, source=lambda v: {**src, 'bank': bad})
            checks['repro_reports_reproduced_and_detects_difference'] = bool(rp['reproduced'] and not rp2['reproduced'])
            try:
                teachers('S50_O0', 0, root=root, max_trees=40, fitter=fitter, log=lambda s: None); checks['write_once'] = False
            except PermissionError:
                checks['write_once'] = True
        if have_torch:
            import torch  # noqa: F401
            from acs_replication import leafcorr as lc
            tch = {t: TR.mock_teacher(W['X'].astype(np.float64), W[t], t, names, T=6, seed=3 + i) for i, t in enumerate(TASKS)}
            Xr = W['X'].astype(np.float64)

            def run(fn, arms, n=None, overrides=SELFTEST_STAGE_B):
                sel = slice(None) if n is None else slice(0, n)
                cks = {}

                def on_ck(name, label, thetas, info):
                    cks[(name, label)] = [np.array(t, copy=True) for t in thetas]
                    return {'label': label}
                rec, fin, _ = fn(0, tch, Xr[sel], names, W['income'][sel], W['occupation'][sel], W['sex'][sel], W['row_id'][sel], arms, on_ck,
                                 recipe_overrides=overrides, log=lambda s: None)
                return rec, fin, cks
            _, fa, ca = run(TR.run_stage_b_arrays, REPLICATION_ARMS)
            _, fb, cb = run(TR.run_stage_b_arrays, (ARM,))
            _, fc, cc = run(run_stage_b_arrays, (ARM,))
            eq = lambda x, y: all(np.array_equal(p, q) for p, q in zip(x, y))
            checks['task_refit_alone_equals_replication_multi_arm_run'] = bool(
                eq(fa[ARM], fb[ARM]) and all(eq(ca[(ARM, lab)], cb[(ARM, lab)]) for lab in ('E0', 'E2')))
            checks['control_stage_b_equals_track_r_path'] = bool(eq(fb[ARM], fc[ARM]))
            # a failed 16-draw calibration (coefficients unused by TASK_REFIT) leaves TASK_REFIT unchanged; other arms still stop
            orig = lc.Trainer.calibrate

            def boom(self, critics):
                raise ZeroDivisionError('invented calibration failure')
            lc.Trainer.calibrate = boom
            try:
                rec_f, fin_f, _ = run(run_stage_b_arrays, (ARM,))
                try:
                    run(run_stage_b_arrays, ('LOCAL_TREE', ARM)); multi_raised = False
                except ZeroDivisionError:
                    multi_raised = True
            finally:
                lc.Trainer.calibrate = orig
            checks['calibration_failure_recorded_task_refit_unchanged'] = bool(rec_f['calibration'].get('failed') and eq(fin_f[ARM], fc[ARM])
                                                                               and multi_raised)
            info['task_refit_theta_nonzero'] = bool(any(np.abs(t).max() > 0 for t in fc[ARM]))
            # the cyclic transcription equals leafcorr.run_arm when the pool fills every batch (T_REFIT and L_TREE)
            with TR._recipe(SELFTEST_STAGE_B):
                ok_t = []
                for arm in ('T_REFIT', 'L_TREE'):
                    outs = []
                    for fn in (lc.run_arm, run_arm_cyclic):
                        leaves = [tch[t].route(Xr, names) for t in TASKS]
                        rec = [lc.Recipient(t, tch[t], leaves[i], tch[t].raw_from_leaves(leaves[i])) for i, t in enumerate(TASKS)]
                        tr = lc.Trainer(0, rec, [W['income'], W['occupation']], np.arange(len(Xr)), W['sex'],
                                        AD.order_rows(W['row_id'], 'ACSR1-probe', 512), AD.order_rows(W['row_id'], 'ACSR1-reference', 256))
                        calib = lc.Trainer(0, rec, [W['income'], W['occupation']], np.arange(len(Xr)), W['sex'],
                                           AD.order_rows(W['row_id'], 'ACSR1-probe', 512), AD.order_rows(W['row_id'], 'ACSR1-reference', 256))
                        coef = calib.calibrate(calib.warm())['coefficients']
                        snaps = []
                        res = fn(tr, arm, 1.0, coef, lambda lab, th, inf: snaps.append([np.array(x, copy=True) for x in th]),
                                 counterfactual_steps=2 if arm == 'L_TREE' else 0)
                        outs.append((snaps, res['final_critic_sha256']))
                    ok_t.append(all(eq(a, b) for a, b in zip(outs[0][0], outs[1][0])) and outs[0][1] == outs[1][1])
            checks['cyclic_transcription_equals_leafcorr_when_batches_full'] = all(ok_t)
            # a pool smaller than (steps - 1) x batch: the source slice leaves one short and one EMPTY batch per epoch; the cyclic
            # order fills every batch
            n_small = 3 * 128 - 20
            rec_s, fin_s, _ = run(run_stage_b_arrays, (ARM,), n=n_small)
            checks['small_pool_cyclic_full_batches_finite'] = bool(rec_s['epoch_order'].startswith('cyclic') and rec_s['source_slice_empty_batches'] == 1
                                                                   and all(np.isfinite(t).all() for t in fin_s[ARM]))
            perm = np.random.default_rng(5).permutation(n_small)
            co = cyclic_order(perm, 512)
            checks['cyclic_order_prefix_is_the_permutation'] = bool(np.array_equal(co[:n_small], perm) and len(co) == 512
                                                                    and np.array_equal(cyclic_order(np.arange(600), 512), np.arange(512)))
            try:
                run(TR.run_stage_b_arrays, (ARM,), n=n_small); info['track_r_guard'] = 'not raised'
            except ValueError as e:
                info['track_r_guard'] = str(e)[:100]
            try:
                with TR._recipe(SELFTEST_STAGE_B):
                    leaves = [tch[t].route(Xr[:n_small], names) for t in TASKS]
                    rec = [lc.Recipient(t, tch[t], leaves[i], tch[t].raw_from_leaves(leaves[i])) for i, t in enumerate(TASKS)]
                    tr = lc.Trainer(0, rec, [W['income'][:n_small], W['occupation'][:n_small]], np.arange(n_small), W['sex'][:n_small],
                                    AD.order_rows(W['row_id'][:n_small], 'ACSR1-probe', 512), AD.order_rows(W['row_id'][:n_small], 'ACSR1-reference', 256))
                    snaps = []
                    lc.run_arm(tr, 'T_REFIT', 1.0, None, lambda lab, th, inf: snaps.append(th))
                info['source_slice_on_small_pool_final_theta_finite'] = bool(all(np.isfinite(t).all() for t in snaps[-1]))
            except Exception as e:
                info['source_slice_on_small_pool_error'] = repr(e)[:120]
    ok = all(checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks, 'info': info, 'cpu_s': round(time.process_time() - t0, 1),
                      'interpreter': _interp()}, default=str))
    return ok


def timing():
    """Invented data of the real size (label-free): the CatBoost teacher loop (CatBoost venv) or the Stage B TASK_REFIT loop with the
    full recipe on mock teachers of the replication's tree counts (torch venv), at S100 and S50 sizes."""
    from cts import track_r as TR
    from . import sizefit as SF
    global _NO_REAL_ACCESS
    _NO_REAL_ACCESS = True
    SF._NO_REAL_ACCESS = True
    rng = np.random.default_rng(11)
    names = SF._names()
    sizes = {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001, 'ATTACK_FIT': 4000, 'ATTACK_VALIDATION': 4000, 'DEV_REPORT': 4000}
    A = SF._fake_world(rng, names, sizes, rare_count=40)
    out = {}
    with tempfile.TemporaryDirectory(prefix='tsz_control_timing_') as tmp:
        with _fakes(A, names, []):
            if importlib.util.find_spec('catboost') is not None:
                for sid in ('S100', 'S50_O0'):
                    t0 = time.process_time(); w0 = time.time()
                    r = teachers(sid, 0, root=Path(tmp), log=lambda s: None)
                    out[f'teachers_{sid}'] = {'cpu_s': round(time.process_time() - t0, 1), 'wall_s': round(time.time() - w0, 1),
                                              'tree_counts': {t: r['teachers'][t]['tree_count'] for t in TASKS}}
                    print(json.dumps(out), flush=True)
            if importlib.util.find_spec('torch') is not None:
                W = A['MODEL_FIT']
                for sid, T in (('S100', (587, 800)), ('S50_O0', (587, 800))):
                    m = C.subset_mask(W['unit'], sid)
                    Xs = W['X'][m].astype(np.float64)
                    tch = {t: TR.mock_teacher(Xs, W[t][m], t, names, T=T[i], seed=i) for i, t in enumerate(TASKS)}
                    t0 = time.process_time(); w0 = time.time()
                    rec, _, _ = run_stage_b_arrays(0, tch, Xs, names, W['income'][m], W['occupation'][m], W['sex'][m], W['row_id'][m], (ARM,),
                                                   lambda *a: {}, log=lambda s: None)
                    out[f'stage_b_{sid}'] = {'cpu_s': round(time.process_time() - t0, 1), 'wall_s': round(time.time() - w0, 1), 'n': rec['n'],
                                             'epoch_order': rec['epoch_order'], 'tree_counts': T}
                    print(json.dumps(out), flush=True)
    print(json.dumps({'timing': out, 'interpreter': _interp()}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest'); sub.add_parser('timing')
    for c in ('teachers', 'stage_b', 'verify'):
        x = sub.add_parser(c); x.add_argument('--subset', required=True, choices=C.subset_ids())
        x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    r = sub.add_parser('repro'); r.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'timing':
        timing(); return 0
    if a.cmd == 'teachers':
        teachers(a.subset, a.version); return 0
    if a.cmd == 'stage_b':
        stage_b(a.subset, a.version); return 0
    if a.cmd == 'verify':
        return 0 if verify(a.subset, a.version) else 1
    repro(a.version)
    return 0


if __name__ == '__main__':
    sys.exit(main())

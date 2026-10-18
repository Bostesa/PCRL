"""Track F of the PCRL correction-transfer campaign: fixed H3 income, newly fitted occupation pipeline per household partition
(EXPLORATORY DEVELOPMENT on previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed). Occupation group (5 official broad civilian groups) is predicted from AGEP, WKHP, COW, SCHL, MAR, POBP through
the admitted 145-column encoding; each recipient receives only its own float64 probability vector and argmax decision. SEX (the
recorded demographic column) is a training-time label for the correction critics on FIT households only and an evaluation label
for subgroup costs; never a runtime input or output.

Per partition P (cts.common.PARTITIONS; FIT / GUARD / CHECK; associated source seed s):
  dictionary   the admitted encoding re-collapsed on FIT: a COW / SCHL / MAR / POBP level with fewer than 10 FIT people moves to its
               block's __OTHER__ column (equal to a dictionary built from FIT raw codes alone; see PROTOCOL.md).
  witness      the A1_PO_16 recipe (ohn.fam_a: HGB, compact_po coding, 16 leaves, min leaf 100, lr 0.025, l2 3, T* <= 1000 by
               household 5-fold inner CV on FIT, refit on FIT) + household cross-fitting on FIT (5 outer folds, nested inner CV):
               out-of-fold logits on FIT (correction base) and the 5 fold pipelines (deployed-ensemble variant). No SEX, no calibrator.
  routing      REFIT: the replication's T_TREE occupation recipe (CatBoost 1.2.8, acs_replication.stage_a) on FIT (recoded X);
               FROZEN: the admitted T_TREE of seed s (fitted on all MODEL_FIT).
  direction    ohn.protect.crossfit_train (source P2 recipe, guard 'none') on FIT: arms PAIR / LOC / CONT, 5 fold pipelines on the
               witness's outer folds + the full-FIT pipeline; the PAIR critic reads H3's frozen income wire of seed s.
  ladder       S = 0, 0.1, ..., 1.0. Deployed r_s = softmax(log p_W + s Delta_full); deployed-ensemble e_s = mean_k softmax(log p_Wk +
               s Delta_k). Incremental costs against s = 0 on the same people: OOF (fold pipeline k on fold k of FIT, the source
               estimator), GUARD, CHECK (D partitions now; R partitions only after the committed REPAIR_FREEZE), MODEL_SELECT.

Commands (real-data ones only through `python3 -m cts.jobqueue run ...`):
  selftest                                   invented data only
  witness   --partition P                    python3
  routing   --partition P                    CatBoost venv (cts.common.CATBOOST_PY)
  routing_repro --seed s                     CatBoost venv; source replay: the replication recipe on ALL MODEL_FIT vs the admitted teacher
  direction --partition P --routing REFIT|FROZEN --arm PAIR|LOC|CONT     torch venv (cts.common.TORCH_PY)
  ladder    --partition P --direction ID     python3 (labels: FIT 'fit', GUARD 'select', CHECK 'check_diag', MODEL_SELECT 'benchmark')
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

K_OCC = 5
LADDER = tuple(round(0.1 * i, 1) for i in range(11))
ROLES = C.ROLES6
DICT_MIN = 10
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
W_RECIPE = 'A1_PO_16'
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')


def pdir(P):
    return C.RUNS / 'TRACKF' / P


def seed_of(P):
    return C.PARTITIONS[P]['seed']


# ------------------------------------------------------------------ FIT-only dictionary
def fit_dictionary(X_fit, names, min_count=DICT_MIN):
    names = tuple(map(str, names)); X_fit = np.asarray(X_fit)
    kept, collapsed = {}, {}
    for b in BLOCKS:
        cols = [i for i, n in enumerate(names) if n.startswith(b + '=') and not n.endswith('__OTHER__')]
        cnt = X_fit[:, cols].sum(0)
        kept[b] = [names[i] for i, c in zip(cols, cnt) if c >= min_count]
        collapsed[b] = [names[i] for i, c in zip(cols, cnt) if c < min_count]
    return {'min_count': int(min_count), 'kept': kept, 'collapsed': collapsed,
            'n_collapsed': {b: len(v) for b, v in collapsed.items()}}


def recode(X, names, dic):
    """Move every collapsed level to its block's __OTHER__ column (rows stay complete one-hot)."""
    names = tuple(map(str, names)); X = np.array(X, dtype=np.float64, copy=True)
    for b in BLOCKS:
        other = names.index(f'{b}=__OTHER__')
        for n in dic['collapsed'][b]:
            i = names.index(n)
            on = X[:, i] == 1
            X[on, i] = 0.0
            X[on, other] = 1.0
    return X


# ------------------------------------------------------------------ small numerics
def probs(z):
    from acs_replication import trees
    return trees.probabilities('occupation', np.asarray(z, dtype=np.float64))


def logp(p):
    z = np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-300))
    return z - z.mean(1, keepdims=True)


def cost_vectors(p, base, y):
    """Per-person incremental costs of p against base on the same people (corrected minus uncorrected)."""
    a, b = C.task_vectors(y, p, K_OCC), C.task_vectors(y, base, K_OCC)
    return {'brier': a['brier'] - b['brier'], 'nll': a['nll'] - b['nll'], 'acc_loss': b['accuracy'] - a['accuracy']}


def cost_summary(p, base, y, sex, unit):
    from ihs.evaluate import cluster_se, subgroup_se
    v = cost_vectors(p, base, y)
    men, women = sex == 0, sex == 1
    out = {'men_brier': float(v['brier'][men].mean()), 'women_brier': float(v['brier'][women].mean()), 'brier': float(v['brier'].mean()),
           'nll': float(v['nll'].mean()), 'acc_loss': float(v['acc_loss'].mean())}
    se = {'men_brier': subgroup_se(v['brier'], men, unit), 'women_brier': subgroup_se(v['brier'], women, unit),
          'brier': cluster_se(v['brier'], unit), 'nll': cluster_se(v['nll'], unit), 'acc_loss': cluster_se(v['acc_loss'], unit)}
    mag = {'mean_abs_logit_change': float(np.abs(logp(p) - logp(base)).sum(1).mean()),
           'flip_rate': float((p.argmax(1) != base.argmax(1)).mean()), 'max_abs_prob_change': float(np.abs(p - base).max())}
    return {'stats': out, 'se': se, 'n': int(len(y)), 'n_men': int(men.sum()), 'n_women': int(women.sum()),
            'households': int(len(np.unique(unit))), 'magnitude': mag}


def _save_npz(path, **arrs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return C.sha(path)


def _load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _role_features(dic=None):
    out = {}
    for r in ROLES:
        F = C.features(r)
        X = F['X'] if dic is None else recode(F['X'], F['feature_names'], dic)
        out[r] = {'X': X, 'names': F['feature_names'], 'row_id': F['row_id'], 'unit': F['unit']}
    return out


# ------------------------------------------------------------------ witness (python3)
def fit_witness(P):
    if C.PARTITIONS[P].get('witness') == 'B2_SINGLE':          # bounded sensitivity witness (cts.sensitivity), same file layout
        from . import sensitivity
        return sensitivity.fit_witness_b2(P)
    from ohn import fam_a as FA
    t0 = time.process_time(); w0 = time.time()
    d = pdir(P) / 'WITNESS'
    if (d / 'WITNESS.json').exists():
        raise PermissionError(f'{P} witness exists (append-only)')
    s = seed_of(P)
    F = C.load('MODEL_FIT', ('features', 'occupation'), 'fit', f'Track F {P}: occupation witness (A1_PO_16 recipe) fit, inner CV and '
               'household cross-fitting on FIT; no SEX', partition=P, prole='FIT')
    names = F['feature_names']; y = F['occupation']; unit = F['unit']
    dic = fit_dictionary(F['X'], names)
    roles = _role_features(dic)
    cfg = dict(FA.CONFIGS[W_RECIPE])
    Xr = FA.rep_matrix(cfg['rep'], recode(F['X'], names, dic), names)
    folds = C.household_folds(unit, 5, f'CTS1_cf_{P}')
    inner = C.household_folds(unit, 5, f'CTS1_inner_{P}')
    reps = {r: FA.rep_matrix(cfg['rep'], roles[r]['X'], names) for r in ROLES}
    pipe = FA.run_pipeline(cfg, s, Xr, y, folds, inner, extra={r: reps[r] for r in ('MODEL_FIT', 'MODEL_SELECT')},
                           log=lambda m: print(m, flush=True))
    dep = pipe['deployed']
    p_dep = {r: FA.release_prob(dep, reps[r]) for r in ROLES}
    oof = np.zeros((len(y), K_OCC))
    for k in range(pipe['K']):
        oof[folds == k] = FA.combine(pipe['oof_raw'][k])
    p_fold = {r: np.stack([FA.combine(pipe['extra_raw'][r][k]) for k in range(pipe['K'])]) for r in ('MODEL_FIT', 'MODEL_SELECT')}
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    files = {}
    with open(d / 'MODEL.pkl', 'xb') as f:
        pickle.dump(dep[0], f, protocol=5)
    files['MODEL.pkl'] = C.sha(d / 'MODEL.pkl')
    for k, m in pipe['outer_refits'].items():
        with open(d / f'FOLD_{k}.pkl', 'xb') as f:
            pickle.dump(m, f, protocol=5)
        files[f'FOLD_{k}.pkl'] = C.sha(d / f'FOLD_{k}.pkl')
    files['WITNESS.npz'] = _save_npz(d / 'WITNESS.npz', **{f'dep__{r}': p_dep[r] for r in ROLES},
                                     **{f'fold__{r}': p_fold[r] for r in p_fold})
    fit_rows = F['rows']
    files['OOF.npz'] = _save_npz(d / 'OOF.npz', oof=oof, fold=folds, fit_rows=fit_rows, row_id=F['row_id'])
    C.write_json(pdir(P) / 'DICT.json', dic)
    own = C.task_vectors(y, oof, K_OCC)
    rec = {'schema': 'CTS1-witness-v1', 'partition': P, 'seed': s, 'recipe': W_RECIPE, 'config': cfg, 'T_star': pipe['T_star'],
           'T_k': {str(k): v for k, v in pipe['T_k'].items()}, 'n_fit_people': int(len(y)), 'n_fit_households': int(len(np.unique(unit))),
           'dictionary': {'min_count': dic['min_count'], 'n_collapsed': dic['n_collapsed'], 'dict_sha256': C.sha(pdir(P) / 'DICT.json')},
           'outer_fold_tag': f'CTS1_cf_{P}', 'inner_fold_tag': f'CTS1_inner_{P}',
           'oof_fit_summary': {m: float(v.mean()) for m, v in own.items()},
           'n_physical_fits': pipe['n_physical_fits'], 'n_logical_fits': pipe['n_logical_fits'], 'files_sha256': files,
           'release_sha256': {r: C.arr_sha(p_dep[r]) for r in ROLES}, 'labels_used': 'FIT occupation only (no SEX, no income)',
           'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'wall_s': time.time() - w0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(d / 'WITNESS.json', rec)
    C.unit_stats(n_model_fits=len(pipe['deployed_inner_models']) + len(dep), n_crossfit_fits=pipe['n_physical_fits'] - len(pipe['deployed_inner_models']) - len(dep))
    print(json.dumps({k: rec[k] for k in ('partition', 'T_star', 'T_k', 'oof_fit_summary', 'cpu_s')}))
    return rec


def witness_predict(P, X, names):
    """Deployed witness probabilities from admitted features alone (dictionary + compact_po + the saved model)."""
    if C.PARTITIONS[P].get('witness') == 'B2_SINGLE':
        from . import sensitivity
        return sensitivity.predict_b2(P, X, names)
    from ohn import fam_a as FA
    d = pdir(P) / 'WITNESS'
    rec = json.loads((d / 'WITNESS.json').read_text())
    if C.sha(d / 'MODEL.pkl') != rec['files_sha256']['MODEL.pkl']:
        raise PermissionError('witness model differs from its record')
    dic = json.loads((pdir(P) / 'DICT.json').read_text())
    with open(d / 'MODEL.pkl', 'rb') as f:
        m = pickle.load(f)
    return FA.release_prob([m], FA.rep_matrix(FA.CONFIGS[W_RECIPE]['rep'], recode(X, names, dic), names))


def load_witness(P):
    d = pdir(P) / 'WITNESS'
    rec = json.loads((d / 'WITNESS.json').read_text())
    for n in ('WITNESS.npz', 'OOF.npz'):
        if C.sha(d / n) != rec['files_sha256'][n]:
            raise PermissionError(f'{n} differs from its record')
    return rec, _load_npz(d / 'WITNESS.npz'), _load_npz(d / 'OOF.npz')


# ------------------------------------------------------------------ routing (CatBoost venv)
def _catboost_teacher(X, y, unit, s, names, rs_items, y_income=None):
    """The replication's Stage A T_TREE occupation recipe: internal split with the two-task class-support repair (as Stage A),
    the UNREDUCED CatBoost seed seed_for('ACSR1catboost', s, 'occupation', 4) (as Stage A), tree count, refit."""
    from acs_replication import trees, data as AD
    labels = {'occupation': y} if y_income is None else {'income': y_income, 'occupation': y}
    fit_rows, val_rows, moved = AD.internal_split(unit, labels, s)
    depth = trees.REGISTRY['depth'][0]
    rs = trees.seed_for(*rs_items)
    m = trees.make_classifier('occupation', depth, trees.REGISTRY['max_trees'], rs)
    m.fit(X[fit_rows], y[fit_rows], eval_set=(X[val_rows], y[val_rows]))
    curve = m.get_evals_result()['validation'][trees.LOSS['occupation']]
    count = int(np.argmin(curve)) + 1
    final = trees.make_classifier('occupation', depth, count, rs)
    final.fit(X, y)
    teacher = trees.from_catboost(final, 'occupation', depth, names)
    info = {'catboost_seed': rs, 'tree_count': count, 'internal_fit_rows': int(len(fit_rows)), 'internal_validation_rows': int(len(val_rows)),
            'moved_households': len(moved), 'internal_best_loss': float(curve[count - 1])}
    return teacher, final, info


def fit_routing(P):
    from acs_replication import trees
    t0 = time.process_time()
    d = pdir(P) / 'ROUTING'
    if (d / 'ROUTING.json').exists():
        raise PermissionError(f'{P} routing exists (append-only)')
    s = seed_of(P)
    F = C.load('MODEL_FIT', ('features', 'income', 'occupation'), 'fit', f'Track F {P}: REFIT routing = replication T_TREE occupation '
               'recipe on FIT (CatBoost 1.2.8; income read only by the two-task class-support repair of the internal split); no SEX',
               partition=P, prole='FIT')
    dic = json.loads((pdir(P) / 'DICT.json').read_text())
    names = F['feature_names']
    X = recode(F['X'], names, dic)
    teacher, model, info = _catboost_teacher(X, F['occupation'], F['unit'], s, names, ('ACSR1catboost', s, 'occupation', 4),
                                             y_income=F['income'])
    roles = _role_features(dic)
    parity = {r: trees.verify_against_catboost(teacher, model, roles[r]['X'], names) for r in ('MODEL_FIT', 'MODEL_SELECT')}
    ok = all(v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 for v in parity.values())
    if not ok:
        raise AssertionError(f'teacher conversion parity failed: {parity}')
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(d / 'TEACHER_occupation.json', 'x') as f:
        json.dump(teacher.payload(), f, sort_keys=True)
    rec = {'schema': 'CTS1-routing-v1', 'partition': P, 'seed': s, 'recipe': 'acs_replication.stage_a T_TREE occupation recipe on FIT '
           '(recoded with the FIT-only dictionary)', **info, 'T': teacher.T, 'L': teacher.L, 'teacher_sha256': teacher.sha(),
           'teacher_file_sha256': C.sha(d / 'TEACHER_occupation.json'), 'parity': parity, 'catboost_version': trees.CATBOOST_VERSION,
           'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(d / 'ROUTING.json', rec)
    C.unit_stats(n_routing_fits=2)
    print(json.dumps({k: rec[k] for k in ('partition', 'tree_count', 'T', 'cpu_s')}))
    return rec


def routing_repro(seed):
    """Source replay: the replication's T_TREE occupation recipe on ALL MODEL_FIT (admitted X) with the installed CatBoost 1.2.8,
    compared with the admitted teacher (exact payload / routing / probabilities). The refitted object is never used in a release."""
    from acs_replication import trees, data as AD
    t0 = time.process_time()
    out = C.RESULTS / 'preflight' / f'ROUTING_REPRO_S{seed}.json'
    F = C.load('MODEL_FIT', ('features', 'income', 'occupation'), 'source_replay',
               f'source replay: refit the replication T_TREE occupation recipe of seed {seed} on all MODEL_FIT to check that the '
               'installed CatBoost reproduces the admitted teacher (never used in a release)')
    names = F['feature_names']
    fit_rows, val_rows, moved = AD.internal_split(F['unit'], {'income': F['income'], 'occupation': F['occupation']}, seed)
    depth = trees.REGISTRY['depth'][0]
    rs = trees.seed_for('ACSR1catboost', seed, 'occupation', depth)
    m = trees.make_classifier('occupation', depth, trees.REGISTRY['max_trees'], rs)
    m.fit(F['X'][fit_rows], F['occupation'][fit_rows], eval_set=(F['X'][val_rows], F['occupation'][val_rows]))
    curve = m.get_evals_result()['validation'][trees.LOSS['occupation']]
    count = int(np.argmin(curve)) + 1
    final = trees.make_classifier('occupation', depth, count, rs); final.fit(F['X'], F['occupation'])
    teacher = trees.from_catboost(final, 'occupation', depth, names)
    from ohn import common as OC
    adm = OC.teacher(seed, 'occupation')
    S = C.features('MODEL_SELECT')
    p_new = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(S['X'], S['feature_names'])))
    p_adm = adm.probabilities_from_raw(adm.raw_from_leaves(adm.route(S['X'], S['feature_names'])))
    rec = {'schema': 'CTS1-routing-repro-v1', 'seed': seed, 'tree_count': count, 'admitted_tree_count': adm.T,
           'teacher_sha256': teacher.sha(), 'admitted_teacher_sha256': adm.sha(), 'identical_teacher': teacher.sha() == adm.sha(),
           'routing_identical_MODEL_SELECT': bool(np.array_equal(teacher.route(S['X'], S['feature_names']), adm.route(S['X'], S['feature_names'])))
           if teacher.T == adm.T else False,
           'max_abs_prob_diff_MODEL_SELECT': float(np.abs(p_new - p_adm).max()) if p_new.shape == p_adm.shape else None,
           'decision_agreement_MODEL_SELECT': float((p_new.argmax(1) == p_adm.argmax(1)).mean()),
           'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows)), 'moved_households': len(moved)},
           'catboost_version': trees.CATBOOST_VERSION, 'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(out, rec)
    C.unit_stats(n_routing_fits=2)
    print(json.dumps(rec))
    return rec


def load_routing(P, kind):
    from acs_replication import trees
    from ohn import common as OC
    if kind == 'FROZEN':
        return OC.teacher(seed_of(P), 'occupation'), None
    d = pdir(P) / 'ROUTING'
    rec = json.loads((d / 'ROUTING.json').read_text())
    if C.sha(d / 'TEACHER_occupation.json') != rec['teacher_file_sha256']:
        raise PermissionError('routing teacher differs from its record')
    t = trees.TreeTeacher(json.loads((d / 'TEACHER_occupation.json').read_text()))
    if t.sha() != rec['teacher_sha256']:
        raise PermissionError('routing payload differs')
    return t, json.loads((pdir(P) / 'DICT.json').read_text())


def route(P, kind, X, names):
    t, dic = load_routing(P, kind)
    return t.route(X if dic is None else recode(X, names, dic), names), t


# ------------------------------------------------------------------ directions (torch venv)
def direction_id(routing, arm):
    return f'{routing}_{arm}'


def fit_direction(P, routing, arm):
    from ohn import protect as PR, devrun as OD
    t0 = time.process_time()
    did = direction_id(routing, arm)
    d = pdir(P) / 'DIR' / did
    if (d / 'DIR.json').exists():
        raise PermissionError(f'{P}/{did} exists (append-only)')
    s = seed_of(P)
    wrec, W, O = load_witness(P)
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'fit', f'Track F {P}: correction direction {did} (task CE + SEX critics '
               'on FIT; cross-fitted on the witness outer folds)', partition=P, prole='FIT')
    fit_rows = F['rows']
    if not np.array_equal(fit_rows, O['fit_rows']) or not np.array_equal(F['row_id'], O['row_id']):
        raise AssertionError('witness FIT rows differ')
    names = F['feature_names']
    Fall = C.features('MODEL_FIT')
    leaves_all, teacher = route(P, routing, Fall['X'], names)
    S = C.features('MODEL_SELECT')
    leaves_ms, _ = route(P, routing, S['X'], S['feature_names'])
    z_oof = logp(O['oof'])
    inc_all = OD.anchor_income('H3', s, 'MODEL_FIT')
    if inc_all.shape[0] != len(Fall['row_id']):
        raise AssertionError('H3 MODEL_FIT income rows differ')
    inc_wire = PR.income_wire(inc_all[fit_rows])
    folds = O['fold']
    full, fold_res, delta_oof = PR.crossfit_train(z_oof, leaves_all[fit_rows], F['occupation'], F['sex'], inc_wire, folds, teacher.T,
                                                  teacher.L, arm, P, {'men': 1.0, 'women': 1.0}, guard='none', verbose=False)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    thetas = {'theta_full': full['theta'], **{f'theta_fold{k}': r['theta'] for k, r in fold_res.items()}}
    files = {'TRAIN.npz': _save_npz(d / 'TRAIN.npz', delta_oof=delta_oof, **thetas)}
    delta = {}
    for r, lv in (('MODEL_FIT', leaves_all), ('MODEL_SELECT', leaves_ms)):
        delta[f'full__{r}'] = PR.correction(full['theta'], lv, teacher.T, teacher.L)
        delta[f'folds__{r}'] = np.stack([PR.correction(fold_res[k]['theta'], lv, teacher.T, teacher.L) for k in sorted(fold_res)])
    files['DELTA.npz'] = _save_npz(d / 'DELTA.npz', **delta)
    rec = {'schema': 'CTS1-direction-v1', 'partition': P, 'seed': s, 'direction': did, 'routing': routing, 'arm': arm,
           'recipe': 'ohn.protect.crossfit_train, source P2 recipe (guard none), RECIPE as ohn.protect.RECIPE',
           'routing_teacher_sha256': teacher.sha(), 'T': teacher.T, 'L': teacher.L, 'witness_files_sha256': wrec['files_sha256'],
           'income_wire': f'H3_STRONG_LJDIR_PB seed {s} MODEL_FIT release restricted to FIT rows (in-sample for H3)',
           'training_full': PR.fold_summary(full), 'training_folds': {str(k): PR.fold_summary(r) for k, r in fold_res.items()},
           'lambda_full': full['lambda'], 'files_sha256': files, 'interpreter': sys.executable, 'cpu_s': time.process_time() - t0,
           'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'n_fits': {'corrections': 6, 'critics': 6 * 9, 'note': '1 full + 5 fold pipelines, each a correction and 9 online critics '
                                                                  '(CONT: updated, unused)'}}
    C.write_json(d / 'DIR.json', rec)
    C.unit_stats(n_model_fits=10, n_crossfit_fits=50)
    print(json.dumps({'partition': P, 'direction': did, 'lambda_full': full['lambda'], 'cpu_s': round(rec['cpu_s'], 1)}))
    return rec


def load_direction(P, did):
    d = pdir(P) / 'DIR' / did
    rec = json.loads((d / 'DIR.json').read_text())
    for n in ('TRAIN.npz', 'DELTA.npz'):
        if C.sha(d / n) != rec['files_sha256'][n]:
            raise PermissionError(f'{n} differs from its record')
    return rec, _load_npz(d / 'TRAIN.npz'), _load_npz(d / 'DELTA.npz')


def release_at(p_w, delta, s):
    return probs(logp(p_w) + float(s) * np.asarray(delta, dtype=np.float64))


def ensemble_at(p_folds, delta_folds, s):
    return np.mean([release_at(p_folds[k], delta_folds[k], s) for k in range(len(p_folds))], axis=0)


# ------------------------------------------------------------------ ladder (python3)
def ladder(P, did, include_check=None):
    t0 = time.process_time()
    out_p = pdir(P) / 'LADDER' / f'{did}.json'
    kind = C.PARTITIONS[P]['kind']
    include_check = (kind == 'diagnosis') if include_check is None else include_check
    if include_check and kind == 'repair' and not C.repair_frozen():
        raise PermissionError('repair CHECK stays closed until the committed REPAIR_FREEZE')
    stem = 'LADDER' if not (kind == 'repair' and include_check) else 'LADDER_CHECK'
    out_p = pdir(P) / stem / f'{did}.json'
    if out_p.exists():
        raise PermissionError(f'{out_p.name} exists (append-only)')
    wrec, W, O = load_witness(P)
    drec, T, D = load_direction(P, did)
    Fit = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'fit', f'Track F {P}: OOF (source estimator) incremental costs of '
                 f'{did} on FIT', partition=P, prole='FIT')
    res = {'schema': 'CTS1-ladder-v1', 'partition': P, 'direction': did, 'ladder': LADDER, 'estimators': {}}
    z_oof = logp(O['oof'])
    base_oof = probs(z_oof)
    res['estimators']['OOF'] = {str(s): cost_summary(probs(z_oof + s * T['delta_oof']), base_oof, Fit['occupation'], Fit['sex'], Fit['unit'])
                                for s in LADDER}
    roles = [('GUARD', 'select', P, 'GUARD')]
    if include_check:
        roles.append(('CHECK', 'check_diag' if kind == 'diagnosis' else 'check_repair', P, 'CHECK'))
    for name, op, part, prole in roles:
        L = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), op, f'Track F {P}: {name} incremental costs of {did} on the fixed '
                   'ladder (deployed and deployed-ensemble)', partition=part, prole=prole)
        rows = L['rows']
        pw = W['dep__MODEL_FIT'][rows]; dl = D['full__MODEL_FIT'][rows]
        pf = W['fold__MODEL_FIT'][:, rows]; df = D['folds__MODEL_FIT'][:, rows]
        res['estimators'][name] = {str(s): cost_summary(release_at(pw, dl, s), pw, L['occupation'], L['sex'], L['unit']) for s in LADDER}
        e0 = ensemble_at(pf, df, 0.0)
        res['estimators'][name + '_ENS'] = {str(s): cost_summary(ensemble_at(pf, df, s), e0, L['occupation'], L['sex'], L['unit']) for s in LADDER}
    if kind == 'diagnosis' or include_check:            # repair partitions: MODEL_SELECT costs only after the REPAIR_FREEZE
        S = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'benchmark', f'Track F {P}: realized MODEL_SELECT incremental costs '
                   f'of {did} on the fixed ladder (diagnosis; never used by a strength rule)')
        pw = W['dep__MODEL_SELECT']; dl = D['full__MODEL_SELECT']
        res['estimators']['MS'] = {str(s): cost_summary(release_at(pw, dl, s), pw, S['occupation'], S['sex'], S['unit']) for s in LADDER}
        pf, df = W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT']
        e0 = ensemble_at(pf, df, 0.0)
        res['estimators']['MS_ENS'] = {str(s): cost_summary(ensemble_at(pf, df, s), e0, S['occupation'], S['sex'], S['unit']) for s in LADDER}
    res.update({'witness_files_sha256': wrec['files_sha256'], 'direction_files_sha256': drec['files_sha256'], 'cpu_s': time.process_time() - t0,
                'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)})
    C.write_json(out_p, res)
    print(json.dumps({'partition': P, 'direction': did, 'estimators': list(res['estimators']), 'cpu_s': round(res['cpu_s'], 1)}))
    return res


# ------------------------------------------------------------------ selftest (invented data only)
def selftest():
    rng = np.random.default_rng(0)
    names = ('AGEP', 'WKHP') + tuple(f'COW={i}' for i in (1, 2)) + ('COW=__OTHER__',) + tuple(f'SCHL={i}' for i in (16, 21)) + \
        ('SCHL=__OTHER__',) + tuple(f'MAR={i}' for i in (1, 5)) + ('MAR=__OTHER__',) + tuple(f'POBP={i}' for i in (12, 36, 303)) + ('POBP=__OTHER__',)
    n = 400
    X = np.zeros((n, len(names))); X[:, 0] = rng.normal(size=n); X[:, 1] = rng.normal(size=n)
    for b, k in (('COW', 3), ('SCHL', 3), ('MAR', 3), ('POBP', 4)):
        cols = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        pick = rng.choice(len(cols), n, p=np.r_[np.full(len(cols) - 1, 0.9 / (len(cols) - 1)), 0.1] if b != 'POBP' else [0.6, 0.38, 0.01, 0.01])
        X[np.arange(n), np.asarray(cols)[pick]] = 1
    dic = fit_dictionary(X, names, min_count=10)
    Xr = recode(X, names, dic)
    checks = {'collapsed_rare_pobp': 'POBP=303' in dic['collapsed']['POBP'],
              'one_hot_preserved': all(np.all(Xr[:, [i for i, nm in enumerate(names) if nm.startswith(b + '=')]].sum(1) == 1) for b in BLOCKS),
              'kept_unchanged': bool(np.array_equal(Xr[:, names.index('POBP=12')], X[:, names.index('POBP=12')]))}
    p = rng.dirichlet(np.ones(K_OCC), size=50); dlt = rng.normal(size=(50, K_OCC))
    checks['identity_at_zero'] = bool(np.allclose(release_at(p, dlt, 0.0), p, atol=1e-12))
    pf = np.stack([p, p]); df = np.stack([dlt, dlt])
    checks['ensemble_of_equal_members'] = bool(np.allclose(ensemble_at(pf, df, 0.7), release_at(p, dlt, 0.7), atol=1e-12))
    y = rng.integers(0, K_OCC, 50); sex = rng.integers(0, 2, 50); unit = np.arange(50) // 2
    cs = cost_summary(release_at(p, dlt, 0.0), p, y, sex, unit)
    checks['zero_cost_at_identity'] = all(abs(v) < 1e-12 for v in cs['stats'].values())
    ok = all(checks.values())
    print(json.dumps({'ok': ok, **checks}))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    for c in ('witness', 'routing'):
        x = sub.add_parser(c); x.add_argument('--partition', required=True, choices=tuple(C.PARTITIONS))
    x = sub.add_parser('routing_repro'); x.add_argument('--seed', type=int, required=True)
    x = sub.add_parser('direction'); x.add_argument('--partition', required=True, choices=tuple(C.PARTITIONS))
    x.add_argument('--routing', required=True, choices=('REFIT', 'FROZEN')); x.add_argument('--arm', required=True, choices=('PAIR', 'LOC', 'CONT'))
    x = sub.add_parser('ladder'); x.add_argument('--partition', required=True, choices=tuple(C.PARTITIONS)); x.add_argument('--direction', required=True)
    x.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    if a.cmd == 'witness':
        fit_witness(a.partition)
    elif a.cmd == 'routing':
        fit_routing(a.partition)
    elif a.cmd == 'routing_repro':
        routing_repro(a.seed)
    elif a.cmd == 'direction':
        fit_direction(a.partition, a.routing, a.arm)
    elif a.cmd == 'ladder':
        ladder(a.partition, a.direction, include_check=True if a.check else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())

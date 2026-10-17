"""Family C: SEX-mixture marginalization (MIX), PCRL autonomous method-development campaign (EXPLORATORY DEVELOPMENT; not
confirmation).

Census context: public-use Florida ACS 2019 person records (development roles of the protected-tree replication). The service
predicts each person's income class (2 classes) and occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL,
MAR, POBP; admitted 145-column encoding) and gives each recipient only its own float64 probability vector and argmax decision.
Evaluation readers try to infer the dataset's recorded SEX column (0 = male, SEX code 1; 1 = female, SEX code 2) from those
released outputs.

Mechanism (Pope & Sydnor 2011 style): fit a SEX-aware task model P(y | x, s) on MODEL_FIT with SEX as an extra input column at
training time only, and release r(x) = sum_s w_s P(y | x, s), averaging the model's two SEX-specific predictions with weights w
that never use the person's SEX. The release is computed from the 145-column admitted encoding of the six permitted inputs only.

Model: the replication's T_TREE CatBoost recipe (acs_replication/stage_a.py), unchanged except for the SEX column: depth 4,
learning rate 0.05, at most 800 trees, l2_leaf_reg 3, thread_count 1, the same CatBoost random seed
(seed_for('ACSR1catboost', seed, task, 4)), tree count = first argmin of the loss on the same label-free 15% household-disjoint
internal split (acs_replication.data.internal_split, seed + 17001) + 1, then a refit on all MODEL_FIT people with that count.
The fitted model is converted to the explicit numpy oblivious-tree teacher (acs_replication.trees) and verified bitwise against
CatBoost (routing) and to 1e-12 (probabilities) on every role, with SEX set to 0 and to 1.

Data access (PROTOCOL.md section 3): MODEL_FIT features, income, occupation and SEX (fitting); features only for CALIBRATION,
MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT (release computation; no label or SEX of those roles is read here).
MODEL_SELECT labels are read only by `amd.devrun score`.

    python -m amd.fam_c fit --config C1_MIX_FULL --seed 0 [--control-check]

writes runs/C/<config>/S<seed>/{RELEASE.npz, RELEASE.json, FIT_DIAG.json, TEACHER_SEXAWARE_<task>.json}.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np

from . import common as C
from . import devrun as DR

FAMILY = 'C'
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
DEPTH = 4
SEX_COL = 'SEX_TRAINING_ONLY'

# Configuration registry. Appended as revisions occur (MECHANISM_CARDS.md / roles/investigator_c/CARD_REVISIONS.md).
CONFIGS = {
    'C1_MIX_FULL': {
        'version': 'C1',
        'model': 'single',            # one CatBoost per task with SEX appended as the last input column
        'weights': 'constant',        # w = MODEL_FIT SEX shares (full marginalization)
        'marginalize': ('income', 'occupation'),
        'description': 'T_TREE CatBoost recipe with SEX appended as an input column at training time only; release '
                       'w0*P(y|x,SEX=0) + w1*P(y|x,SEX=1) with w = MODEL_FIT SEX shares, for income and occupation, on every role',
    },
    # Control for C1 (not a revision): matched-cost non-SEX shrinkage of the SEX-blind teacher.
    'C1_CTRL_TEMP': {
        'version': 'C1',
        'model': 'teacher_temperature',
        'match_to': 'C1_MIX_FULL',
        'description': 'SEX-blind T_TREE teacher (stored; identical-recipe refit reproduced it bitwise) with one temperature per task '
                       'on its raw scores, each chosen label-free so that the mean KL(T_TREE || release) on MODEL_FIT features equals '
                       'that of the C1_MIX_FULL release of the same seed (matched-cost "less confident" control, no SEX used)',
    },
    # Revision 1 (C2): marginalize income only; occupation = frozen eligible JOINT_TREE E20 occupation (the occupation output of H_LJ).
    'C2_MIXINC_JOCC': {
        'version': 'C2',
        'model': 'hybrid_mixincome_jointocc',
        'weights': 'constant',
        'marginalize': ('income',),
        'description': 'income: the C1 SEX-aware CatBoost income model (T_TREE recipe + SEX input at training time only) released as '
                       'w0*P(income|x,SEX=0) + w1*P(income|x,SEX=1), w = MODEL_FIT SEX shares; occupation: the frozen JOINT_TREE E20 '
                       'occupation output (T_TREE routing + stored stage B leaf-score correction theta; identical to the occupation '
                       'output of the eligible reference H_LJ); no new occupation fit',
    },
    # Diagnostic (counts 0; fits nothing new; never a candidate): attributes C2's pair result to the income output.
    'CDIAG_TTINC_JOCC': {
        'version': 'C2',
        'model': 'hybrid_mixincome_jointocc',
        'income_source': 'teacher',
        'marginalize': (),
        'description': 'DIAGNOSTIC ONLY: income = the stored SEX-blind T_TREE income output; occupation = the frozen JOINT_TREE E20 '
                       'occupation output (as in H_LJ and C2). Fits nothing; places the SEX-marginalized income (C2) between the '
                       'SEX-blind teacher income and the LOCAL income (H_LJ) at the linked-pair level',
    },
}


def _trees():
    from acs_replication import trees
    return trees


def kl_rows(p, q):
    """Row-wise KL(p || q) in nats (float64)."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), 0.0)
    return t.sum(1)


def fit_recipe(X, y, task, seed, fit_rows, val_rows, names):
    """The stage A T_TREE recipe on the given input matrix (internal early stopping, refit on all rows, numpy conversion)."""
    trees = _trees()
    rs = trees.seed_for('ACSR1catboost', seed, task, DEPTH)
    t0 = time.process_time()
    m = trees.make_classifier(task, DEPTH, trees.REGISTRY['max_trees'], rs)
    m.fit(X[fit_rows], y[fit_rows], eval_set=(X[val_rows], y[val_rows]))
    curve = m.get_evals_result()['validation'][trees.LOSS[task]]
    count = int(np.argmin(curve)) + 1
    final = trees.make_classifier(task, DEPTH, count, rs)
    final.fit(X, y)
    teacher = trees.from_catboost(final, task, DEPTH, names)
    info = {'catboost_seed': rs, 'tree_count': count, 'internal_curve_len': len(curve), 'internal_best_loss': float(curve[count - 1]),
            'internal_curve_every_50': [float(c) for c in curve[::50]], 'cpu_s': time.process_time() - t0}
    return final, teacher, info


def with_sex(X, s):
    return np.ascontiguousarray(np.column_stack((X, np.full(len(X), float(s), dtype=np.float32))), dtype=np.float32)


def sex_specific(teacher, X, names_aug):
    """P(y | x, SEX=s) for s = 0, 1 from the numpy SEX-aware teacher (the person's own SEX is never used)."""
    return [teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(with_sex(X, s), names_aug))) for s in (0, 1)]


def sex_split_stats(teacher, sex_index):
    used = teacher.features == sex_index                         # (T, depth); padding splits use feature 0 with a never-true border
    level = np.flatnonzero(used.any(0)).tolist()
    return {'trees': int(teacher.T), 'trees_with_sex_split': int(used.any(1).sum()),
            'sex_split_levels_histogram': {str(i): int(used[:, i].sum()) for i in range(teacher.depth)}, 'levels_used': level}


def load_data():
    F = C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    feats = {'MODEL_FIT': F}
    for r in ROLES[1:]:
        feats[r] = C.load_role(r, ('features',))                 # features only: no label or SEX of these roles is read
    return F, feats


def control_check(seed, F, feats, fit_rows, val_rows):
    """Refit the SEX-blind T_TREE recipe and compare with the stored teacher and its stored outputs on every role."""
    stored = C.stage_a_outputs(seed)
    out = {}
    names = F['feature_names']
    for task in C.TASKS:
        final, teacher, info = fit_recipe(F['X'], F[task], task, seed, fit_rows, val_rows, names)
        ref = C.teacher(seed, task)
        diffs = {}
        for role, R in feats.items():
            if not np.array_equal(stored[f'row_id__{role}'], R['row_id']):
                raise AssertionError(f'stored stage A row order differs on {role}')
            p = teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(R['X'], names)))
            diffs[role] = float(np.max(np.abs(p - stored[f'T_TREE__{role}__{task}'])))
        out[task] = {**info, 'teacher_sha_equal': teacher.sha() == ref.sha(), 'stored_tree_count': int(ref.T),
                     'max_abs_prob_diff_vs_stored_by_role': diffs, 'bitwise_all_roles': all(v == 0.0 for v in diffs.values())}
    return out


def run_fit(config_id, seed, do_control):
    t_unit = time.process_time()
    cfg = CONFIGS[config_id]
    trees = _trees()
    from acs_replication.data import internal_split
    F, feats = load_data()
    names = tuple(F['feature_names'])
    names_aug = names + (SEX_COL,)
    sex = F['sex'].astype(np.int64)
    if set(np.unique(sex)) != {0, 1}:
        raise ValueError('SEX coding differs from {0, 1}')
    y = {t: F[t] for t in C.TASKS}
    fit_rows, val_rows, moved = internal_split(F['unit'], y, seed)
    diag = {'config': cfg, 'seed': seed, 'internal_split': {'fit_rows': int(len(fit_rows)), 'validation_rows': int(len(val_rows)),
                                                            'moved_households': len(moved)}}
    if do_control:
        diag['control_check_T_TREE_refit'] = control_check(seed, F, feats, fit_rows, val_rows)
    stored = C.stage_a_outputs(seed)
    stage_a_rec = json.loads(C.admitted(f'stage_a/S{seed}/STAGE_A_SEED.json').read_text())
    w = np.array([np.mean(sex == 0), np.mean(sex == 1)], dtype=np.float64)
    diag['weights_model_fit_sex_shares'] = w.tolist()
    X_aug = with_sex(F['X'], 0); X_aug[:, -1] = sex.astype(np.float32)
    outs = {r: {} for r in ROLES}
    d = DR.release_dir(FAMILY, config_id, seed); d.mkdir(parents=True, exist_ok=True, mode=0o700)
    fit_cpu = 0.0
    for task in C.TASKS:
        final, teacher, info = fit_recipe(X_aug, y[task], task, seed, fit_rows, val_rows, names_aug)
        fit_cpu += info['cpu_s']
        tp = d / f'TEACHER_SEXAWARE_{task}.json'
        tp.write_text(json.dumps(teacher.payload(), sort_keys=True))
        parity = {}
        for role, R in feats.items():
            for s in (0, 1):
                v = trees.verify_against_catboost(teacher, final, with_sex(R['X'], s), names_aug)
                parity[f'{role}|SEX={s}'] = v
        ok = all(v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12
                 for v in parity.values())
        if not ok:
            raise AssertionError(f'SEX-aware teacher conversion parity failed: {task}')
        imp = final.get_feature_importance()
        tinfo = {**info, 'teacher_sha256': teacher.sha(), 'parity_all_roles_both_sex_values': ok,
                 'sex_feature_importance': float(imp[-1]), 'sex_importance_rank': int((imp > imp[-1]).sum()) + 1,
                 'sex_splits': sex_split_stats(teacher, len(names)),
                 'T_TREE_internal_best_loss': stage_a_rec['T_TREE'][task]['internal_best_loss'],
                 'T_TREE_tree_count': stage_a_rec['T_TREE'][task]['tree_count']}
        mt = {}
        for role, R in feats.items():
            p0, p1 = sex_specific(teacher, R['X'], names_aug)
            if task in cfg['marginalize']:
                r = w[0] * p0 + w[1] * p1
            else:
                r = stored[f'T_TREE__{role}__{task}']
            r = r / r.sum(1, keepdims=True)
            outs[role][task] = r
            tt = stored[f'T_TREE__{role}__{task}']
            gap = np.abs(p1 - p0)
            mt[role] = {'mean_KL_T_TREE_to_release': float(kl_rows(tt, r).mean()), 'p90_KL_T_TREE_to_release': float(np.quantile(kl_rows(tt, r), 0.9)),
                        'mean_abs_sex_gap_by_class': gap.mean(0).tolist(), 'mean_signed_gap_female_minus_male': (p1 - p0).mean(0).tolist(),
                        'distinct_release_rows': int(np.unique(r, axis=0).shape[0]), 'rows': int(len(r))}
            if role == 'MODEL_FIT':
                yy = y[task]; idx = np.arange(len(yy))
                own = lambda p: -np.log(np.maximum(p[idx, yy], 1e-12))
                brier = lambda p: ((p - (yy[:, None] == np.arange(p.shape[1])[None, :])) ** 2).sum(1)
                p_own = np.where(sex[:, None] == 1, p1, p0)       # diagnostic only (uses the person's SEX); never released
                mt[role]['in_sample_by_sex'] = {
                    f'sex{g}': {'release_nll': float(own(r)[sex == g].mean()), 'T_TREE_nll': float(own(tt)[sex == g].mean()),
                                'own_sex_model_nll': float(own(p_own)[sex == g].mean()),
                                'release_brier': float(brier(r)[sex == g].mean()), 'T_TREE_brier': float(brier(tt)[sex == g].mean()),
                                'own_sex_model_brier': float(brier(p_own)[sex == g].mean())} for g in (0, 1)}
        tinfo['by_role'] = mt
        diag[task] = tinfo
    rec = {'mechanism': cfg['description'], 'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           'training_labels_used': 'MODEL_FIT income, occupation and SEX (SEX as an input column of the task model at training time only)',
           'runtime_inputs': 'the 145-column admitted encoding of AGEP, WKHP, COW, SCHL, MAR, POBP; SEX is never a runtime input: the '
                             'release averages the SEX-aware model over SEX=0 and SEX=1 with fixed weights',
           'weights': w.tolist(), 'internal_split': diag['internal_split'],
           'tree_counts': {t: diag[t]['tree_count'] for t in C.TASKS}, 'catboost_seeds': {t: diag[t]['catboost_seed'] for t in C.TASKS},
           'teacher_sha256': {t: diag[t]['teacher_sha256'] for t in C.TASKS}, 'catboost_version': trees.CATBOOST_VERSION,
           'mechanism_fit_cpu_s': fit_cpu, 'fit_cpu_s': time.process_time() - t_unit,
           'roles_read': 'MODEL_FIT (features, income, occupation, SEX); features only for CALIBRATION, MODEL_SELECT, ATTACK_FIT, '
                         'ATTACK_VALIDATION, DEV_REPORT'}
    DR.save_release(FAMILY, config_id, seed, outs, rec)
    diag['unit_cpu_s'] = time.process_time() - t_unit
    (d / 'FIT_DIAG.json').write_text(json.dumps(diag, indent=1, sort_keys=True, default=float) + '\n')
    summary = {'config': config_id, 'seed': seed, 'tree_counts': rec['tree_counts'], 'weights': rec['weights'],
               'sex_importance': {t: (round(diag[t]['sex_feature_importance'], 3), diag[t]['sex_importance_rank'],
                                      diag[t]['sex_splits']['trees_with_sex_split']) for t in C.TASKS},
               'internal_best_loss': {t: (round(diag[t]['internal_best_loss'], 5), round(diag[t]['T_TREE_internal_best_loss'], 5)) for t in C.TASKS},
               'mean_KL_T_TREE_to_release_MODEL_FIT': {t: round(diag[t]['by_role']['MODEL_FIT']['mean_KL_T_TREE_to_release'], 5) for t in C.TASKS},
               'unit_cpu_s': round(diag['unit_cpu_s'], 1)}
    if do_control:
        summary['control'] = {t: (diag['control_check_T_TREE_refit'][t]['teacher_sha_equal'], diag['control_check_T_TREE_refit'][t]['bitwise_all_roles'],
                                  diag['control_check_T_TREE_refit'][t]['tree_count'], round(diag['control_check_T_TREE_refit'][t]['internal_best_loss'], 5))
                              for t in C.TASKS}
    print(json.dumps(summary, default=float))


def teacher_raw(seed, task, X, names):
    t = C.teacher(seed, task)
    return t, t.raw_from_leaves(t.route(X, names))


def run_temperature_control(config_id, seed):
    """Matched-cost non-SEX shrinkage control: T_TREE raw scores / temperature, temperature matched label-free on MODEL_FIT."""
    t_unit = time.process_time()
    cfg = CONFIGS[config_id]
    trees = _trees()
    F, feats = load_data()
    names = tuple(F['feature_names'])
    stored = C.stage_a_outputs(seed)
    ref_outs, ref_rec = DR.load_release(FAMILY, cfg['match_to'], seed, roles=('MODEL_FIT',))
    temps, diag, outs = {}, {'config': cfg, 'seed': seed, 'matched_release_sha256': ref_rec['release_sha256']}, {r: {} for r in ROLES}
    for task in C.TASKS:
        _, raw_fit = teacher_raw(seed, task, F['X'], names)
        tt = trees.probabilities(task, raw_fit)
        if np.max(np.abs(tt - stored[f'T_TREE__MODEL_FIT__{task}'])) > 1e-12:
            raise AssertionError('teacher raw scores do not reproduce the stored T_TREE outputs')
        target = float(kl_rows(tt, ref_outs['MODEL_FIT'][task]).mean())
        f = lambda T: float(kl_rows(tt, trees.probabilities(task, raw_fit / T)).mean())
        lo, hi = 1.0, 2.0
        while f(hi) < target:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        T = 0.5 * (lo + hi); temps[task] = T
        diag[task] = {'temperature': T, 'target_mean_KL_MODEL_FIT': target, 'achieved_mean_KL_MODEL_FIT': f(T)}
        for role, R in feats.items():
            _, raw = teacher_raw(seed, task, R['X'], names)
            r = trees.probabilities(task, raw / T)
            outs[role][task] = r
            diag[task][role] = {'mean_KL_T_TREE_to_release': float(kl_rows(stored[f'T_TREE__{role}__{task}'], r).mean()),
                                'distinct_release_rows': int(np.unique(r, axis=0).shape[0])}
    rec = {'mechanism': cfg['description'], 'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           'training_labels_used': 'none new (the stored T_TREE teacher was fitted on MODEL_FIT income/occupation without SEX); the '
                                   'temperatures are matched label-free on MODEL_FIT features to the C1_MIX_FULL release',
           'runtime_inputs': 'the 145-column admitted encoding of the six permitted inputs; SEX never used',
           'temperatures': temps, 'fit_cpu_s': time.process_time() - t_unit,
           'roles_read': 'MODEL_FIT features (+ the stored C1_MIX_FULL MODEL_FIT release); features only for the other roles'}
    DR.save_release(FAMILY, config_id, seed, outs, rec)
    d = DR.release_dir(FAMILY, config_id, seed)
    diag['unit_cpu_s'] = time.process_time() - t_unit
    (d / 'FIT_DIAG.json').write_text(json.dumps(diag, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'config': config_id, 'seed': seed, 'temperatures': temps,
                      'KL': {t: (round(diag[t]['target_mean_KL_MODEL_FIT'], 6), round(diag[t]['achieved_mean_KL_MODEL_FIT'], 6)) for t in C.TASKS},
                      'KL_MODEL_SELECT': {t: round(diag[t]['MODEL_SELECT']['mean_KL_T_TREE_to_release'], 6) for t in C.TASKS}}))


def joint_occupation(seed, X, names):
    """Frozen JOINT_TREE E20 occupation output (stage B wire: teacher routing + stored theta)."""
    t = C.teacher(seed, 'occupation')
    theta = C.stage_b_theta(seed, 'JOINT_TREE', 'E20')[1]
    return t.probabilities_from_raw(t.raw_from_leaves(t.route(X, names), theta))


def run_hybrid(config_id, seed):
    """C2: SEX-marginalized income (C1 income model; loaded from the C1 run of this seed when present, else fitted with the
    identical recipe) + the frozen JOINT_TREE E20 occupation output."""
    t_unit = time.process_time()
    cfg = CONFIGS[config_id]
    trees = _trees()
    from acs_replication.data import internal_split
    F, feats = load_data()
    names = tuple(F['feature_names']); names_aug = names + (SEX_COL,)
    sex = F['sex'].astype(np.int64)
    w = np.array([np.mean(sex == 0), np.mean(sex == 1)], dtype=np.float64)
    c1 = DR.release_dir(FAMILY, 'C1_MIX_FULL', seed) / 'TEACHER_SEXAWARE_income.json'
    diag = {'config': cfg, 'seed': seed, 'weights': w.tolist()}
    fit_cpu = 0.0
    if c1.is_file():
        teacher = trees.TreeTeacher(json.loads(c1.read_text()))
        c1rec = json.loads((DR.release_dir(FAMILY, 'C1_MIX_FULL', seed) / 'RELEASE.json').read_text())
        if teacher.sha() != c1rec['teacher_sha256']['income']:
            raise AssertionError('stored SEX-aware income teacher differs from the C1 record')
        diag['income_model'] = {'source': 'C1_MIX_FULL SEX-aware income teacher (no refit)', 'teacher_sha256': teacher.sha(),
                                'tree_count': int(teacher.T)}
    else:
        y = {t: F[t] for t in C.TASKS}
        fit_rows, val_rows, moved = internal_split(F['unit'], y, seed)
        X_aug = with_sex(F['X'], 0); X_aug[:, -1] = sex.astype(np.float32)
        final, teacher, info = fit_recipe(X_aug, y['income'], 'income', seed, fit_rows, val_rows, names_aug)
        fit_cpu += info['cpu_s']
        for role, R in feats.items():
            for s in (0, 1):
                v = trees.verify_against_catboost(teacher, final, with_sex(R['X'], s), names_aug)
                if not (v['routing_identical'] and v['decision_identical'] and v['raw_max_abs_diff'] <= 1e-9 and v['probability_max_abs_diff'] <= 1e-12):
                    raise AssertionError('SEX-aware income teacher parity failed')
        d0 = DR.release_dir(FAMILY, config_id, seed); d0.mkdir(parents=True, exist_ok=True, mode=0o700)
        (d0 / 'TEACHER_SEXAWARE_income.json').write_text(json.dumps(teacher.payload(), sort_keys=True))
        diag['income_model'] = {'source': 'fitted here with the C1 recipe', **info, 'teacher_sha256': teacher.sha()}
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
    stored = C.stage_a_outputs(seed)
    outs = {r: {} for r in ROLES}
    teacher_income = cfg.get('income_source') == 'teacher'
    if teacher_income:
        diag['income_model'] = {'source': 'stored SEX-blind T_TREE income output (diagnostic; the SEX-aware model is not used)',
                                'teacher_sha256': C.teacher(seed, 'income').sha()}
    for role, R in feats.items():
        if teacher_income:
            outs[role]['income'] = stored[f'T_TREE__{role}__income']
        else:
            p0, p1 = sex_specific(teacher, R['X'], names_aug)
            inc = w[0] * p0 + w[1] * p1
            outs[role]['income'] = inc / inc.sum(1, keepdims=True)
        occ = joint_occupation(seed, R['X'], names)
        outs[role]['occupation'] = occ
        rr = {'income_mean_KL_T_TREE_to_release': float(kl_rows(stored[f'T_TREE__{role}__income'], outs[role]['income']).mean()),
              'distinct_income': int(np.unique(outs[role]['income'], axis=0).shape[0]),
              'distinct_pair': int(np.unique(np.column_stack((outs[role]['income'], occ)), axis=0).shape[0])}
        if role in bank.get('H_LJ', {}):
            if not np.array_equal(ids[f'row_id__{role}'], R['row_id']):
                raise AssertionError(f'frozen bank row order differs on {role}')
            rr['occupation_max_abs_diff_vs_frozen_H_LJ'] = float(np.max(np.abs(occ - bank['H_LJ'][role]['occupation'])))
            if rr['occupation_max_abs_diff_vs_frozen_H_LJ'] > 1e-12:
                raise AssertionError(f'JOINT_TREE occupation does not reproduce the frozen H_LJ occupation on {role}')
        diag[role] = rr
    inc_text = ('income: none new (stored SEX-blind T_TREE income output, fitted in the replication on MODEL_FIT income without SEX)'
                if teacher_income else 'income: MODEL_FIT income and SEX (SEX as an input column at training time only)')
    rec = {'mechanism': cfg['description'], 'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           'training_labels_used': inc_text + '; occupation: none new (frozen JOINT_TREE E20 leaf-score correction, fitted in the '
                                   'replication on MODEL_FIT with its SEX critics)',
           'runtime_inputs': 'the 145-column admitted encoding of the six permitted inputs; SEX never a runtime input',
           'weights': None if teacher_income else w.tolist(),
           'income_teacher_sha256': C.teacher(seed, 'income').sha() if teacher_income else teacher.sha(), 'mechanism_fit_cpu_s': fit_cpu,
           'fit_cpu_s': time.process_time() - t_unit,
           'roles_read': 'MODEL_FIT (features, SEX; income if refitted); features only for the other roles; frozen bank outputs '
                         '(H_LJ occupation) for the reproduction check'}
    DR.save_release(FAMILY, config_id, seed, outs, rec)
    d = DR.release_dir(FAMILY, config_id, seed)
    diag['unit_cpu_s'] = time.process_time() - t_unit
    (d / 'FIT_DIAG.json').write_text(json.dumps(diag, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'config': config_id, 'seed': seed, 'income_model': diag['income_model']['source'],
                      'occ_vs_H_LJ': {r: diag[r].get('occupation_max_abs_diff_vs_frozen_H_LJ') for r in ROLES},
                      'distinct_pair_MS': diag['MODEL_SELECT']['distinct_pair'], 'unit_cpu_s': round(diag['unit_cpu_s'], 1)}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--config', required=True, choices=sorted(CONFIGS)); f.add_argument('--seed', type=int, required=True)
    f.add_argument('--control-check', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'fit':
        model = CONFIGS[a.config]['model']
        if model == 'teacher_temperature':
            run_temperature_control(a.config, a.seed)
        elif model == 'hybrid_mixincome_jointocc':
            run_hybrid(a.config, a.seed)
        else:
            run_fit(a.config, a.seed, a.control_check)


if __name__ == '__main__':
    main()

"""Task-only CatBoost teachers and household-fold cross-fitting for the shared-cell mechanism (T_TREE registry).

Registry (acs_replication.trees / stage_a): depth 4, learning rate 0.05, at most 800 trees, l2_leaf_reg 3, one thread;
tree count = first argmin of the internal household-disjoint 15% validation loss + 1 (acs_replication.data.internal_split,
seed + 17001, class-support repair), then a refit on all fitting rows with that count. Every teacher is exported to an
explicit numpy TreeTeacher and verified against CatBoost (bitwise routing, raw-score parity), so deployment needs numpy only.
Seeds are campaign-specific (seed_for('SMP1...', ...)), so these are fresh fits, not the stored T_TREE.
"""
from __future__ import annotations

import numpy as np

from acs_replication import trees
from acs_replication.data import internal_split


def fit_teacher(task, X, y, unit, names, seed, tag):
    labels = {task: y} if task in trees.CLASSES else {'income': y}
    fit_rows, val_rows, moved = internal_split(unit, labels, seed)
    depth = trees.REGISTRY['depth'][0]; rs = trees.seed_for(tag, seed, task, depth)
    kind = task if task in trees.CLASSES else 'income'
    m = trees.make_classifier(kind, depth, trees.REGISTRY['max_trees'], rs)
    m.fit(X[fit_rows], y[fit_rows], eval_set=(X[val_rows], y[val_rows]))
    curve = m.get_evals_result()['validation'][trees.LOSS[kind]]
    count = int(np.argmin(curve)) + 1
    final = trees.make_classifier(kind, depth, count, rs); final.fit(X, y)
    teacher = trees.from_catboost(final, kind, depth, names)
    parity = trees.verify_against_catboost(teacher, final, X, names)
    if not (parity['routing_identical'] and parity['decision_identical'] and parity['raw_max_abs_diff'] <= 1e-9
            and parity['probability_max_abs_diff'] <= 1e-12):
        raise AssertionError(f'teacher conversion parity failed: {task} {parity}')
    return teacher, {'catboost_seed': rs, 'tree_count': count, 'internal_best_loss': float(curve[count - 1]),
                     'moved_households': len(moved), 'parity_on_fit_rows': parity, 'teacher_sha256': teacher.sha()}


def teacher_probs(teacher, X, names):
    return teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(X, names)))


def household_folds(unit, n_folds, seed):
    groups = np.unique(unit); perm = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {g: i % n_folds for i, g in enumerate(groups[perm])}
    return np.array([fold_of[u] for u in unit], dtype=np.int64)


def cross_fit(task, X, y, unit, names, seed, tag, n_folds):
    """Out-of-fold probabilities (each row predicted by the teacher not trained on its household) and the fold teachers."""
    fold = household_folds(unit, n_folds, trees.seed_for(tag + '|folds', seed))
    k = trees.CLASSES[task] if task in trees.CLASSES else 2
    oof = np.zeros((len(X), k)); teachers, infos = [], []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        t, info = fit_teacher(task, X[tr], y[tr], unit[tr], names, seed * 100 + f, tag)
        oof[te] = teacher_probs(t, X[te], names); teachers.append(t); infos.append(info)
    return oof, teachers, infos, fold

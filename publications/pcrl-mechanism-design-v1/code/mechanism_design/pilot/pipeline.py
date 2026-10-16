"""Execution path of the prepared MMP pilot: fit arms, Stage-0 gates, Stage-1 readers, Stage-2 report.

The same functions serve the synthetic rehearsal (this campaign) and a future authorised real run. Real comparator
outputs would come from the admitted frozen bank; synthetic comparators are stand-ins with the same shapes.
Utility clauses follow acs_replication.data.clauses exactly; readers follow the ACS slate exactly
(task_conditioned_representation.attacks.fit_reader / predict_reader, MODEL_SEED_BASE + s); the endpoint and status
rules are copied from acs_hybrids.report (crit, status_at, endpoint) so the pilot does not import that campaign's
private-path module.
"""
from __future__ import annotations

import numpy as np

SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}


def kinds(p):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); one = np.eye(p.shape[1])[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def auc(y, s):
    from task_conditioned_representation.inference import auc as _auc
    return float(_auc(np.asarray(y), np.asarray(s, dtype=np.float64)))


# ------------------------------------------------------------------ fitting
def fit_catboost(task, X, y, unit, seed, max_trees):
    """T_TREE registry with household internal early stopping (acs_replication.stage_a), then refit on all rows."""
    from acs_replication import trees
    from task_headroom.data import group_disjoint_split
    fit_rows, val_rows = group_disjoint_split(unit, seed + 17001, 0.15)
    rs = trees.seed_for('MD1catboost', seed, task, X.shape[1])
    m = trees.make_classifier(task if task in K else 'income', 4, max_trees, rs)
    m.fit(X[fit_rows], y[fit_rows], eval_set=(X[val_rows], y[val_rows]))
    loss = trees.LOSS[task if task in K else 'income']
    curve = m.get_evals_result()['validation'][loss]
    count = int(np.argmin(curve)) + 1
    final = trees.make_classifier(task if task in K else 'income', 4, count, rs)
    final.fit(X, y)
    return final, {'tree_count': count, 'internal_best_loss': float(curve[count - 1]), 'seed': rs}


def with_sex(X, s):
    return np.column_stack((X, np.full(len(X), float(s), dtype=np.float32) if np.isscalar(s) else s.astype(np.float32)))


def fit_mmp_arms(src, seed, max_trees):
    """MODEL_FIT only: SEX-aware teachers per task, SEX-blind teachers (for the SEX-gain gate) and the cross-fitted propensity."""
    mf = src.load_role('MODEL_FIT', ('features', 'sex', 'income', 'occupation'))
    X, sex, unit = mf['X'], mf['sex'], mf['unit']
    pibar = float(sex.mean())
    aware, blind, info = {}, {}, {'pibar': pibar}
    for task in TASKS:
        aware[task], ia = fit_catboost(task, with_sex(X, sex), mf[task], unit, seed, max_trees)
        blind[task], ib = fit_catboost(task, X, mf[task], unit, seed, max_trees)
        info[task] = {'sex_aware': ia, 'sex_blind': ib, 'internal_nll_gain_from_sex': ib['internal_best_loss'] - ia['internal_best_loss']}
    groups = np.unique(unit); perm = np.random.default_rng(seed + 9101).permutation(len(groups))
    fold_of_group = {g: i % 5 for i, g in enumerate(groups[perm])}
    fold = np.array([fold_of_group[u] for u in unit])
    prop = []
    for f in range(5):
        tr = fold != f
        model, _ = fit_catboost('income', X[tr], sex[tr], unit[tr], seed * 10 + f, max_trees)   # binary Logloss registry
        prop.append(model)
    info['propensity_folds'] = 5
    return {'aware': aware, 'blind': blind, 'propensity': prop, 'pibar': pibar, 'info': info}


def arm_outputs(arms, X):
    """Release vectors for MMP and its twin on encoded inputs X (no SEX read: the SEX slot is set to 0 and 1)."""
    out = {'MMP': {}, 'MMP_TWIN': {}, 'T_BLIND': {}}
    logits = np.mean([m.predict(X, prediction_type='RawFormulaVal') for m in arms['propensity']], axis=0)
    pihat = 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64).reshape(-1)))
    for task in TASKS:
        q0 = arms['aware'][task].predict_proba(with_sex(X, 0)); q1 = arms['aware'][task].predict_proba(with_sex(X, 1))
        out['MMP'][task] = (1 - arms['pibar']) * q0 + arms['pibar'] * q1
        out['MMP_TWIN'][task] = (1 - pihat)[:, None] * q0 + pihat[:, None] * q1
        out['T_BLIND'][task] = arms['blind'][task].predict_proba(X)
    return out, pihat


# ------------------------------------------------------------------ stage 0
def utility_clauses(labels, probs, refs, seed):
    from acs_replication.data import clauses
    return clauses(labels, [probs['income'], probs['occupation']], {r: [refs[r]['income'], refs[r]['occupation']] for r in refs}, seed)


def proxy_auc(reader, fit_p, fit_sex, sel_p, sel_sex, seed):
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    kf = np.column_stack((kinds(fit_p['income'])['out'], kinds(fit_p['occupation'])['out']))
    ks = np.column_stack((kinds(sel_p['income'])['out'], kinds(sel_p['occupation'])['out']))
    rec = fit_reader(kf, fit_sex, reader, MODEL_SEED_BASE + seed)
    return auc(sel_sex, predict_reader(rec, ks))


def merge_diagnostic(X, p):
    xi = len(np.unique(np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))))
    rel = np.column_stack((p['income'], p['occupation'])).astype(np.float64)
    ri = len(np.unique(np.ascontiguousarray(rel).view(np.dtype((np.void, rel.dtype.itemsize * rel.shape[1])))))
    return {'distinct_encoded_inputs': int(xi), 'distinct_release_pairs': int(ri), 'release_pairs_per_input': ri / max(xi, 1)}


def stage0(src, seeds, max_trees, references):
    res = {'per_seed': {}, 'gates': {}}
    sel = src.load_role('MODEL_SELECT', ('features', 'sex', 'income', 'occupation'))
    mf = src.load_role('MODEL_FIT', ('features', 'sex'))
    fitted = {}
    for s in seeds:
        arms = fit_mmp_arms(src, s, max_trees)
        out_sel, _ = arm_outputs(arms, sel['X']); out_fit, _ = arm_outputs(arms, mf['X'])
        refs = references(s, 'MODEL_SELECT')
        cl = utility_clauses({t: sel[t] for t in TASKS}, out_sel['MMP'], refs, s)
        prox = {arm: {r: proxy_auc(r, out_fit[arm], mf['sex'], out_sel[arm], sel['sex'], s) for r in ('OUT_LR', 'OUT_HGB')}
                for arm in ('MMP', 'MMP_TWIN', 'T_BLIND')}
        res['per_seed'][str(s)] = {'fit_info': arms['info'], 'mmp_clauses': cl, 'mmp_failed_clauses': [c for c in cl if not c['point_pass']],
                                   'proxies': prox, 'merge_diagnostic_mmp': merge_diagnostic(sel['X'], out_sel['MMP']),
                                   'merge_diagnostic_twin': merge_diagnostic(sel['X'], out_sel['MMP_TWIN'])}
        fitted[s] = arms
    ps = res['per_seed']
    res['gates'] = {
        'G0_sex_gain': all(max(ps[str(s)]['fit_info'][t]['internal_nll_gain_from_sex'] for t in TASKS) > 0 for s in seeds),
        'G1_utility_F3': all(not ps[str(s)]['mmp_failed_clauses'] for s in seeds),
        'G2_premise_F1': all(ps[str(s)]['proxies']['MMP_TWIN']['OUT_LR'] - ps[str(s)]['proxies']['MMP']['OUT_LR'] >= 0.02 for s in seeds),
        'G3_nonlinear_premise': all(ps[str(s)]['proxies']['MMP_TWIN']['OUT_HGB'] - ps[str(s)]['proxies']['MMP']['OUT_HGB'] >= 0.01 for s in seeds),
    }
    res['continue_to_stage1'] = all(res['gates'].values())
    return res, fitted


# ------------------------------------------------------------------ stage 1
def fit_view_bank(fit_k, fit_sex, val_k, val_sex, rep_k, seed):
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    out = []
    for name, kind in SLATE:
        rec = fit_reader(fit_k[kind], fit_sex, name, MODEL_SEED_BASE + seed)
        sv, sr = predict_reader(rec, val_k[kind]), predict_reader(rec, rep_k[kind])
        a = auc(val_sex, sv); sign = 1 if a >= 0.5 else -1
        out.append({'name': name, 'val_auc': max(a, 1 - a), 'sign': sign, 'dev_scores': sr if sign == 1 else 1 - sr})
    return out


def readers_for_arm(p_fit, p_val, p_dev, fit_sex, val_sex, seed):
    views = {}
    for role, p in (('fit', p_fit), ('val', p_val), ('dev', p_dev)):
        ki, ko = kinds(p['income']), kinds(p['occupation'])
        views[role] = {'income': ki, 'occupation': ko, 'pair': {k: np.column_stack((ki[k], ko[k])) for k in ki}}
    banks = {v: fit_view_bank(views['fit'][v], fit_sex, views['val'][v], val_sex, views['dev'][v], seed) for v in ('income', 'occupation', 'pair')}
    pair_cands = banks['pair'] + [{**c, 'name': f'IGNORE_occupation__{c["name"]}'} for c in banks['income']] + \
                 [{**c, 'name': f'IGNORE_income__{c["name"]}'} for c in banks['occupation']]
    win = {v: max(banks[v], key=lambda c: c['val_auc']) for v in ('income', 'occupation')}
    win['pair'] = max(pair_cands, key=lambda c: c['val_auc'])
    return {v: {'winner': w['name'], 'val_auc': w['val_auc'], 'dev_scores': w['dev_scores']} for v, w in win.items()}, len(pair_cands)


def stage1(src, seeds, fitted, comparators, fractions=(0.25, 0.5, 1.0)):
    att = src.load_role('ATTACK_FIT', ('features', 'sex')); val = src.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    dev = src.load_role('DEV_REPORT', ('features',))
    res = {}
    for s in seeds:
        outs = {r: arm_outputs(fitted[s], R['X'])[0] for r, R in (('fit', att), ('val', val), ('dev', dev))}
        arms = {a: {r: outs[r][a] for r in outs} for a in ('MMP', 'MMP_TWIN')}
        arms.update({name: {r: comparators(s, role) for r, role in (('fit', 'ATTACK_FIT'), ('val', 'ATTACK_VALIDATION'), ('dev', 'DEV_REPORT'))}
                     for name in ('NEURAL_RAW',)} if comparators else {})
        rs = {}
        for a, p in arms.items():
            rs[a], n_pair = readers_for_arm(p['fit'], p['val'], p['dev'], att['sex'], val['sex'], s)
        # learning curve on nested household subsets of ATTACK_FIT (pair view only)
        groups = np.unique(att['unit']); perm = np.random.default_rng(4242 + s).permutation(len(groups))
        lc = {}
        for frac in fractions:
            keep = np.isin(att['unit'], groups[perm[:max(2, int(round(frac * len(groups))))]])
            lc[str(frac)] = {}
            for a, p in arms.items():
                sub = {t: p['fit'][t][keep] for t in TASKS}
                r, _ = readers_for_arm(sub, p['val'], p['dev'], att['sex'][keep], val['sex'], s)
                lc[str(frac)][a] = {'pair_winner': r['pair']['winner'], 'pair_val_auc': r['pair']['val_auc']}
        # controls on the MMP wire (pair view)
        from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
        kf = np.column_stack((kinds(arms['MMP']['fit']['income'])['out'], kinds(arms['MMP']['fit']['occupation'])['out']))
        kv = np.column_stack((kinds(arms['MMP']['val']['income'])['out'], kinds(arms['MMP']['val']['occupation'])['out']))
        shuf = np.random.default_rng(2026101602 + s).permutation(att['sex'])
        null = auc(val['sex'], predict_reader(fit_reader(kf, shuf, 'OUT_HGB', MODEL_SEED_BASE + s), kv))
        planted = auc(val['sex'], predict_reader(fit_reader(np.column_stack((kf, att['sex'])), att['sex'], 'OUT_HGB', MODEL_SEED_BASE + s),
                                                 np.column_stack((kv, val['sex']))))
        res[str(s)] = {'readers': rs, 'pair_candidates_per_arm': n_pair, 'learning_curve_val_auc': lc,
                       'controls': {'shuffled_null_val_auc': null, 'planted_bit_val_auc': planted,
                                    'valid': bool(0.45 <= null <= 0.55 and planted >= 0.99)}}
    return res


# ------------------------------------------------------------------ stage 2
def crit(G, level_one_sided):
    from scipy import stats
    return float(stats.t.ppf(1 - level_one_sided, G - 1))


def status_at(kind, L, U, m):
    if kind == 'benefit':
        return 'ESTABLISHED' if L > m else ('ESTABLISHED_SHORTFALL' if U < m else 'UNRESOLVED')
    return 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')


def endpoint(point, draws, kind, margin, c1, cK):
    se = float(np.std(np.asarray(draws, dtype=np.float64), ddof=1))
    L, U = point - cK * se, point + cK * se
    clause = (point - c1 * se > margin) if kind == 'benefit' else (point + c1 * se <= margin)
    return {'point': float(point), 'se': se, 'family_lower': float(L), 'family_upper': float(U), 'clause_pass': bool(clause),
            'status': status_at(kind, L, U, margin)}


def stage2(src, seeds, s1, references, K_primary=36, B=2048, boot_seed=2026101701):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import bootstrap_auc
    truth = src.dev_report_truth()
    dev = src.load_role('DEV_REPORT', ('features',))
    w, G = group_weights(dev['unit'], seed=boot_seed, draws=B)
    c1, cK = crit(G, 0.05 / 3), crit(G, 0.05 / (2 * K_primary))
    y = truth['sex']
    ends = {}
    for s in seeds:
        R = s1[str(s)]['readers']
        def a_and_draws(arm, view):
            sc = R[arm][view]['dev_scores']
            return auc(y, sc), np.asarray(bootstrap_auc(y, sc, w))
        mmp_pair, mmp_d = a_and_draws('MMP', 'pair'); tw_pair, tw_d = a_and_draws('MMP_TWIN', 'pair')
        ends[f'RECIPE|coalition_benefit_vs_MMP_TWIN|s{s}'] = endpoint(tw_pair - mmp_pair, tw_d - mmp_d, 'benefit', 0.020, c1, cK)
        if 'NEURAL_RAW' in R:
            nr_pair, nr_d = a_and_draws('NEURAL_RAW', 'pair')
            ends[f'COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s{s}'] = endpoint(nr_pair - mmp_pair, nr_d - mmp_d, 'benefit', 0.0, c1, cK)
            ends[f'COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s{s}']['nested_material_0.020'] = status_at(
                'benefit', ends[f'COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s{s}']['family_lower'],
                ends[f'COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s{s}']['family_upper'], 0.020)
        for view in ('income', 'occupation'):
            m, md = a_and_draws('MMP', view); t, td = a_and_draws('MMP_TWIN', view)
            ends[f'RECIPE|{view}_recipient_worsening_vs_MMP_TWIN|s{s}'] = endpoint(m - t, md - td, 'no_excess', 0.010, c1, cK)
    return {'endpoints': ends, 'G': G, 'B': B, 'critical_clause': c1, 'critical_family': cK, 'weights': w, 'truth': truth}


def dev_utility(src, seeds, fitted, references, w, c1, cK, truth):
    """24 DEV_REPORT utility clauses (acs_replication.data task vectors) and the 4 within-SEX occupation Brier guards,
    each with paired household-bootstrap endpoints."""
    from acs_replication.data import task_vectors, CAPS
    dev = src.load_role('DEV_REPORT', ('features',))
    ends = {}
    wsum = w.sum(1)
    for s in seeds:
        mmp = arm_outputs(fitted[s], dev['X'])[0]['MMP']
        refs = references(s, 'DEV_REPORT')
        for task in TASKS:
            own = task_vectors(truth[task], mmp[task], K[task])
            for ref_name, ref in refs.items():
                rv = task_vectors(truth[task], ref[task], K[task])
                for metric, cap in CAPS.items():
                    key = {'accuracy_loss': 'accuracy', 'nll_excess': 'nll', 'brier_excess': 'brier'}[metric]
                    diff = (rv[key] - own[key]) if metric == 'accuracy_loss' else (own[key] - rv[key])
                    draws = (w * diff[None, :]).sum(1) / wsum
                    ends[f'UTILITY|{task}|{ref_name}|{metric}|s{s}'] = endpoint(float(diff.mean()), draws, 'no_excess', cap, c1, cK)
        own = task_vectors(truth['occupation'], mmp['occupation'], K['occupation'])['brier']
        rv = task_vectors(truth['occupation'], refs['U0']['occupation'], K['occupation'])['brier']
        for g in (0, 1):
            m = (truth['sex'] == g).astype(np.float64)
            diff = (own - rv) * m
            draws = (w * diff[None, :]).sum(1) / (w * m[None, :]).sum(1)
            ends[f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s}'] = endpoint(float(diff.sum() / m.sum()), draws, 'no_excess', 0.005, c1, cK)
    return ends

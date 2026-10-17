"""Family A: constrained coalition-aware leaf correction (CCLC). Exploratory; see MECHANISM_CARDS.md.

What the code does (one seed, one configuration):
- Routing: the replication's frozen CatBoost teachers T_TREE (income, occupation) route every person to one leaf per tree;
  the released raw score is teacher_raw(x) + theta[leaves(x)].sum / T (acs_replication.leafcorr parametrization; theta = 0
  reproduces the teacher exactly, checked against the stored teacher outputs before training).
- Data: MODEL_FIT only (features, both task labels, SEX). A label-free, household-separated 20% internal holdout
  (amd.common.household_folds, 5 folds, fold 0) never enters any gradient; the corrections train on the other 80%.
- Objective: the transported JOINT_TREE objective. Task cross-entropy plus lambda_r * kappa_r * privacy risk, where the
  privacy risk is the online SEX critic bank's (linear / MLP32 / MLP64) normalized cross-entropy deficit on the income,
  occupation and linked-pair wires (`objective='joint'`) or on the pair wire only (`objective='pair'`). lambda_r is the
  recipe's FIT-only gradient-norm calibration at strength rho.
- NEW, the utility controller: after every epoch the release is scored on the internal holdout against the teacher: income
  NLL, summed Brier and accuracy; occupation NLL and summed Brier; occupation summed Brier within each SEX group. Each
  recipient's budget is the teacher's published MODEL_SELECT headroom over the worse of U0/Ucal (STAGE_A_HEADROOM.json; for
  the SEX groups, the smaller of that and the holdout's in-sample per-SEX headroom) plus `frac` x the original cap. If a
  recipient exceeds any of its budgets, its theta and optimizer state roll back to its last within-budget epoch and its
  privacy multiplier kappa_r is multiplied by `down`; otherwise kappa_r grows by `up` (capped at kappa_max).
- Output: the final (always within holdout budget) corrections; probabilities for every role from the permitted inputs.
Training-time SEX labels are used only by the critics and the per-SEX holdout budget; SEX is never a runtime input.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import time

import numpy as np

from acs_replication import leafcorr as lc
from acs_replication import data as AD
from . import common as C
from . import devrun as DR

ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
CONFIGS = {
    # A1: initial version (JOINT_TREE objective at the recipe strength, controller at 60% of each cap)
    'CCLC_a1': {'objective': 'joint', 'rho': 1.0, 'frac': 0.6, 'up': 1.2, 'down': 0.5, 'kappa0': 1.0, 'kappa_max': 4.0, 'epochs': 30},
    # A0: matched task-only control (same loop, same 80% fitting people, privacy off)
    # A2 (revision 1, after A1): per-SEX budgets from the teacher's measured MODEL_SELECT per-SEX headroom (A1's in-sample
    # holdout estimate made the men's budget 0.003 and throttled occupation); income budget at 20% of each cap (A1's income
    # Brier drifted +0.004 from the internal holdout to MODEL_SELECT) and income pressure capped at the recipe strength.
    'CCLC_a2': {'objective': 'joint', 'rho': 1.0, 'frac': 0.6, 'frac_income': 0.2, 'up': 1.2, 'down': 0.5, 'kappa0': 1.0,
                'kappa_max': 4.0, 'kappa_max_income': 1.0, 'epochs': 30, 'per_sex_headroom': 'model_select_baseline'},
    # A3 (revision 2, after A2): A2 met every MODEL_SELECT clause and guard with margin but its occupation output read 0.597
    # (> H_LJ + 0.010); the holdout log shows the men's occupation Brier budget throttling occupation protection. Explicit
    # subgroup-aware task loss: occupation cross-entropy weighted men 2 : women 1 (training-time SEX only), per-SEX budgets at
    # 80% of the guard, occupation pressure up to 4x.
    'CCLC_a3': {'objective': 'joint', 'rho': 1.0, 'frac': 0.6, 'frac_income': 0.2, 'frac_guard': 0.8, 'up': 1.2, 'down': 0.5,
                'kappa0': 1.0, 'kappa_max': 4.0, 'kappa_max_income': 1.0, 'epochs': 30, 'per_sex_headroom': 'model_select_baseline',
                'occupation_sex_weights': [2.0, 1.0]},
    'TASKREFIT80': {'objective': None, 'rho': 0.0, 'frac': 0.6, 'up': 1.2, 'down': 0.5, 'kappa0': 0.0, 'kappa_max': 0.0, 'epochs': 30},
}


def budgets(seed, hold_sex, teacher_ho, u0_ho, cfg):
    h = json.loads((C.REP_RESULTS / 'STAGE_A_HEADROOM.json').read_text())['per_seed'][str(seed)]
    frac = cfg['frac']
    out = {}
    for t, cap_map in (('income', ('nll', 'brier', 'accuracy')), ('occupation', ('nll', 'brier'))):
        out[t] = {}
        f = cfg.get('frac_income', frac) if t == 'income' else frac
        for m in cap_map:
            head = min(h[t]['headroom_vs_U0'][m], h[t]['headroom_vs_Ucal'][m])
            cap = {'nll': C.CAPS['nll_excess'], 'brier': C.CAPS['brier_excess'], 'accuracy': C.CAPS['accuracy_loss']}[m]
            out[t][m] = max(0.0, head) + f * cap
    head_b = min(h['occupation']['headroom_vs_U0']['brier'], h['occupation']['headroom_vs_Ucal']['brier'])
    out['occupation_by_sex'] = {}
    if cfg.get('per_sex_headroom') == 'model_select_baseline':
        # the teacher's MODEL_SELECT within-SEX occupation Brier headroom over U0 (amd.baselines, logged MODEL_SELECT use)
        base = json.loads((C.RUNS / 'BASELINES' / f'BASELINES_MS_S{seed}.json').read_text())['T_TREE'][str(seed)]
        for g in (0, 1):
            out['occupation_by_sex'][f'sex{g}'] = max(0.0, -base['guard_occupation_brier_excess_vs_U0'][f'sex{g}']) + cfg.get('frac_guard', frac) * C.GUARD
        return out
    for g in (0, 1):
        ins = float((u0_ho[g] - teacher_ho[g]))          # in-sample per-SEX Brier headroom of the teacher over U0 on the holdout
        out['occupation_by_sex'][f'sex{g}'] = max(0.0, min(head_b, ins)) + frac * C.GUARD
    return out


def metrics(p_inc, p_occ, y_inc, y_occ, sex):
    vi, vo = C.task_vectors(y_inc, p_inc, 2), C.task_vectors(y_occ, p_occ, 5)
    return {'income': {'nll': float(vi['nll'].mean()), 'brier': float(vi['brier'].mean()), 'accuracy': float(vi['accuracy'].mean())},
            'occupation': {'nll': float(vo['nll'].mean()), 'brier': float(vo['brier'].mean()), 'accuracy': float(vo['accuracy'].mean())},
            'occupation_by_sex': {f'sex{g}': float(vo['brier'][sex == g].mean()) for g in (0, 1)}}


def excess(cur, ref):
    out = {t: {m: (cur[t][m] - ref[t][m]) if m != 'accuracy' else (ref[t][m] - cur[t][m]) for m in cur[t]} for t in ('income', 'occupation')}
    out['occupation_by_sex'] = {g: cur['occupation_by_sex'][g] - ref['occupation_by_sex'][g] for g in cur['occupation_by_sex']}
    return out


def violations(ex, bud):
    v = {0: [], 1: []}
    for m, b in bud['income'].items():
        if ex['income'][m] > b:
            v[0].append(f'income_{m}')
    for m, b in bud['occupation'].items():
        if ex['occupation'][m] > b:
            v[1].append(f'occupation_{m}')
    for g, b in bud['occupation_by_sex'].items():
        if ex['occupation_by_sex'][g] > b:
            v[1].append(f'occupation_brier_{g}')
    return v


def run(seed, config_id):
    import torch
    cfg = CONFIGS[config_id]
    t_cpu = time.process_time()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    F = C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex'))
    names, X = F['feature_names'], F['X']
    fold = C.household_folds(F['unit'], 5, f'AMD-A-holdout-S{seed}')
    tr_idx, ho_idx = np.where(fold != 0)[0], np.where(fold == 0)[0]
    teachers = [C.teacher(seed, t) for t in C.TASKS]
    leaves = [tch.route(X, names) for tch in teachers]
    raws = [tch.raw_from_leaves(L) for tch, L in zip(teachers, leaves)]
    stored = C.stage_a_outputs(seed)
    if not np.array_equal(stored['row_id__MODEL_FIT'], F['row_id']):
        raise AssertionError('MODEL_FIT row order differs from the stored teacher outputs')
    for i, t in enumerate(C.TASKS):
        if np.max(np.abs(teachers[i].probabilities_from_raw(raws[i]) - stored[f'T_TREE__MODEL_FIT__{t}'])) > 1e-12:
            raise AssertionError('zero-correction identity failed')
    y = [F['income'], F['occupation']]; sex = F['sex']
    rec_tr = [lc.Recipient(t, teachers[i], leaves[i][tr_idx], raws[i][tr_idx]) for i, t in enumerate(C.TASKS)]
    rec_ho = [lc.Recipient(t, teachers[i], leaves[i][ho_idx], raws[i][ho_idx]) for i, t in enumerate(C.TASKS)]
    rid_tr = F['row_id'][tr_idx]
    probe = AD.order_rows(rid_tr, 'AMD-A-probe', 512); reference = AD.order_rows(rid_tr, 'AMD-A-reference', 256)
    trainer = lc.Trainer(seed, rec_tr, [y[0][tr_idx], y[1][tr_idx]], np.arange(len(tr_idx)), sex[tr_idx], probe, reference)
    if cfg.get('occupation_sex_weights'):
        import torch.nn.functional as Fn
        wv = torch.from_numpy(np.where(sex[tr_idx] == 0, cfg['occupation_sex_weights'][0], cfg['occupation_sex_weights'][1]).astype(np.float64))

        def weighted_ce(i, rows, _tr=trainer):
            lg = _tr.rec[i].logits(_tr.theta[i], rows)
            if i == 0:
                return Fn.cross_entropy(lg, _tr.labels[i][rows])
            w = wv[rows]
            return (Fn.cross_entropy(lg, _tr.labels[i][rows], reduction='none') * w).sum() / w.sum()
        trainer.ce = weighted_ce                              # SEX-weighted occupation task loss (training time only)
    ho_all = torch.arange(len(ho_idx))
    y_ho = [y[0][ho_idx], y[1][ho_idx]]; sex_ho = sex[ho_idx]

    def holdout_metrics():
        with torch.no_grad():
            p = [torch.softmax(r.logits(th, ho_all), 1).numpy() for r, th in zip(rec_ho, trainer.theta)]
        return metrics(p[0], p[1], y_ho[0], y_ho[1], sex_ho)
    ref = holdout_metrics()                                   # theta = 0: the teacher on the holdout
    u0b = C.task_vectors(y_ho[1], stored['U0__MODEL_FIT__occupation'][ho_idx], 5)['brier']
    teb = C.task_vectors(y_ho[1], stored['T_TREE__MODEL_FIT__occupation'][ho_idx], 5)['brier']
    bud = budgets(seed, sex_ho, {g: float(teb[sex_ho == g].mean()) for g in (0, 1)}, {g: float(u0b[sex_ho == g].mean()) for g in (0, 1)}, cfg)

    critics = trainer.warm(); calib = trainer.calibrate(critics)
    lam = [0.0, 0.0] if cfg['objective'] is None else calib['coefficients'][str(cfg['rho'])]['lambda_J'] if str(cfg['rho']) in calib['coefficients'] else \
        [cfg['rho'] * calib['medians']['task'][i] / calib['medians']['joint'][i] for i in range(2)]
    kappa = [cfg['kappa0'], cfg['kappa0']]
    opts = [torch.optim.AdamW([t], lr=lc.RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=lc.RECIPE['weight_decay'])
            for t in trainer.theta]
    good = [(trainer.theta[i].detach().clone(), copy.deepcopy(opts[i].state_dict())) for i in range(2)]
    task_rng = np.random.default_rng(lc.seed_for('AMD-A-task', seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('AMD-A-privacy', seed)); critic_rng = random.Random(lc.seed_for('AMD-A-critic', seed))
    log = []
    B, S = lc.RECIPE['batch_size'], lc.RECIPE['steps_per_epoch']
    for epoch in range(cfg['epochs']):
        trainer.refresh(critics)
        order = task_rng.permutation(trainer.n)
        risks = []
        for b in range(S):
            task_rows = torch.from_numpy(order[b * B:(b + 1) * B])
            pr, ps = trainer.privacy_batch(priv_rng)
            if cfg['objective'] is not None:
                w = trainer.wires(pr); terms = critics.terms(w[0], w[1], ps)
                P = terms['pair'] if cfg['objective'] == 'pair' else lc.objective(terms, 'joint')
                pg = torch.autograd.grad(P, trainer.theta, allow_unused=True)
                pg = [torch.zeros_like(trainer.theta[i]) if g is None else g for i, g in enumerate(pg)]
                risks.append({v: float(terms[v].detach()) for v in lc.VIEWS})
            else:
                pg = [torch.zeros_like(t) for t in trainer.theta]
            for i in range(2):
                g_t = torch.autograd.grad(trainer.ce(i, task_rows), trainer.theta[i])[0]
                opts[i].zero_grad(set_to_none=True)
                trainer.theta[i].grad = g_t + lam[i] * kappa[i] * pg[i]
                opts[i].step()
            for _ in range(lc.RECIPE['critic_rounds_per_recipient_update'] * 2):
                trainer.critic_round(critics, critic_rng)
        ex = excess(holdout_metrics(), ref)
        viol = violations(ex, bud)
        actions = {}
        for i in range(2):
            if viol[i]:
                with torch.no_grad():
                    trainer.theta[i].copy_(good[i][0])
                opts[i].load_state_dict(copy.deepcopy(good[i][1]))
                kappa[i] *= cfg['down']; actions[i] = 'rollback'
            else:
                good[i] = (trainer.theta[i].detach().clone(), copy.deepcopy(opts[i].state_dict()))
                kmax = cfg.get('kappa_max_income', cfg['kappa_max']) if i == 0 else cfg['kappa_max']
                kappa[i] = min(kmax, kappa[i] * cfg['up']) if cfg['objective'] is not None else 0.0; actions[i] = 'accept'
        mean_risk = {v: float(np.mean([r[v] for r in risks])) for v in lc.VIEWS} if risks else {}
        log.append({'epoch': epoch + 1, 'holdout_excess_vs_teacher': ex, 'violations': viol, 'actions': actions, 'kappa': list(kappa),
                    'mean_train_critic_risk': mean_risk})
        print(json.dumps({'epoch': epoch + 1, 'inc_nll': round(ex['income']['nll'], 4), 'inc_brier': round(ex['income']['brier'], 4),
                          'occ_nll': round(ex['occupation']['nll'], 4), 'men': round(ex['occupation_by_sex']['sex0'], 4),
                          'women': round(ex['occupation_by_sex']['sex1'], 4), 'viol': viol, 'kappa': [round(k, 3) for k in kappa],
                          'risk': {k: round(v, 3) for k, v in mean_risk.items()}}), flush=True)
    final = [good[i][0].numpy() for i in range(2)]
    outs = {}
    for role in ROLES:
        XX = X if role == 'MODEL_FIT' else C.load_role(role, ('features',))['X']
        outs[role] = {t: teachers[i].probabilities_from_raw(teachers[i].raw_from_leaves(teachers[i].route(XX, names), final[i]))
                      for i, t in enumerate(C.TASKS)}
    d = DR.save_release('A', config_id, seed, outs, {
        'mechanism': 'CCLC: leaf-score corrections on frozen T_TREE routing; JOINT_TREE critic objective; holdout utility controller',
        'config': cfg, 'lambda': lam, 'calibration_medians': calib['medians'], 'budgets_vs_teacher_on_holdout': bud,
        'teacher_holdout_metrics': ref, 'final_holdout_excess': log[-1]['holdout_excess_vs_teacher'] if log else None,
        'holdout': {'rule': 'household_folds(unit, 5, AMD-A-holdout-S<seed>) == 0', 'persons': int(len(ho_idx)), 'train_persons': int(len(tr_idx))},
        'labels_used_at_training': 'MODEL_FIT income, occupation (task loss); MODEL_FIT SEX (critics, per-SEX holdout budget)',
        'runtime_inputs': 'admitted 145-column encoding of AGEP, WKHP, COW, SCHL, MAR, POBP only', 'fit_cpu_s': time.process_time() - t_cpu,
        'epochs_log': log})
    np.savez(d / 'THETA.npz', theta_income=final[0], theta_occupation=final[1])
    print(json.dumps({'saved': str(d.name), 'cpu_s': round(time.process_time() - t_cpu, 1)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--config', required=True)
    a = ap.parse_args(); run(a.seed, a.config)

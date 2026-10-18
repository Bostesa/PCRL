"""Family P: pair-aware learned income output with the occupation output frozen (PAIO). EXPLORATORY DEVELOPMENT on previously
used Florida ACS 2019 records; see results/pcrl_income_headroom_successor_v1/roles/investigator_p/CARD.md.

Census context: public-use ACS 2019 1-Year PUMS person records for Florida (development roles of the protected-tree
replication). The service predicts income class (2 classes) and occupation group (5 classes) from the six permitted raw fields
(AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X). Each recipient receives only its own float64 probability
vector and argmax decision. SEX is the dataset's recorded demographic column; readers estimate it from released vectors.

The release of one configuration (one seed):
- occupation: the accepted, frozen JOINT_TREE E20 occupation output (T_TREE occupation routing + the stored Stage B JOINT_TREE
  E20 occupation leaf corrections; identical to ihs.bases.joint_occupation). Never trained here.
- income: the frozen T_TREE income routing (CatBoost teacher) plus learned income leaf-score corrections theta
  (acs_replication.leafcorr parametrization: raw = teacher_raw + theta[leaves].sum / T; theta = 0 is the teacher exactly),
  optionally followed by a 2-parameter Platt map fitted on CALIBRATION income labels (config 'platt').
  Released probability = acs_replication.trees.probabilities('income', raw); decision = argmax.
Training (MODEL_FIT only; SEX a training-time label only):
- task loss: income cross-entropy on MODEL_FIT minibatches (the leafcorr recipe: AdamW lr 0.01, wd 0.01, batch 256,
  40 steps per epoch);
- privacy term (lambda > 0 only): online SEX critic bank (linear / MLP32 / MLP64 per view, acs_replication.leafcorr.Critics)
  on the income wire and on the linked pair wire (this income wire + the frozen occupation wire): P = w_inc * risk(income) +
  w_pair * risk(pair). lambda = rho * median||grad CE|| / median||grad P|| at the exact starting point with warm critics
  (16 fixed MODEL_FIT draws, leafcorr calibration rule);
- explicit income-utility budget (label-free): after every epoch theta is scaled toward the teacher (theta <- s * theta, the
  largest s in [0, 1] by bisection) so that on the MODEL_FIT people
    BRIER: 2 * mean((p1 - p1_T)^2)            <= B_brier   (expected summed-Brier increase over the teacher if the teacher's
                                                            probabilities were calibrated: the cross term then vanishes),
    ACC:   mean(flip * |2 p1_T - 1|)           <= B_acc     (expected accuracy loss under the same assumption),
    KL:    mean(KL(p_T || p))                  <= B_kl      (expected NLL increase under the same assumption).
  The budgets are measured against the frozen teacher (T_TREE income), never against labels, so the in-sample position of
  MODEL_FIT people in the teacher routing does not enter them; their relation to MODEL_SELECT costs is checked only through
  `ihs.devrun score` and the published reference values (adaptive, logged).
- matched task-only twin: the identical loop (people, batches, epochs, critics updated, projection) with lambda = 0.
- temperature control: the twin's income log-odds divided by tau, tau chosen label-free on MODEL_FIT so that its BRIER
  distortion against the teacher equals a matched configuration's final value (ordinary shrinkage at matched cost).
Runtime inputs: the admitted 145-column encoding of the six raw fields only. The income output does not use the occupation
output or any hidden quantity of the occupation recipient at runtime (it is a function of X through the income routing).

    python -m ihs.fam_p fit --seed 0 --config P1_PAIR_B40
    python -m ihs.fam_p selftest          (invented data only; no real data is read)
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from acs_replication import leafcorr as lc
from acs_replication import trees

FAMILY = 'P'
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')

_BASE = {'kind': 'train', 'init': 'teacher', 'rho': 3.0, 'weights': {'income': 1.0, 'pair': 1.0}, 'epochs': 30,
         'budget': {'brier': 0.004, 'acc': 0.003, 'kl': 0.008}, 'platt': False}
CONFIGS = {
    # P1 (initial version): pair-aware income corrections from the teacher, privacy = income-view + pair critics at 3x the
    # leafcorr calibrated strength, explicit label-free budget BRIER 0.004 / ACC 0.003 / KL 0.008 against the teacher.
    'P1_PAIR_B40': {**_BASE},
    # P1 at the upper budget level (same mechanism; second point of the initial version's two-level budget):
    # BRIER 0.005 / ACC 0.0035 / KL 0.009 (reserve arithmetic: teacher MODEL_SELECT Brier excess vs U0 -0.0041 / -0.0045,
    # reserve threshold about +0.0018 / +0.0016 for B2TWIN-like SEs).
    'P1_PAIR_B50': {**_BASE, 'budget': {'brier': 0.005, 'acc': 0.0035, 'kl': 0.009}},
    # matched task-only twin of P1: identical loop, people, batches, epochs, critic updates and projection; lambda = 0.
    # (Its distortion stays far below both budgets, so the projection never binds and it is the twin of both P1 levels.)
    'PC_TWIN_B40': {**_BASE, 'rho': 0.0, 'twin_of': 'P1_PAIR_B40'},
    # ordinary comparator at the same budget (no training): the stored JOINT_TREE E20 income corrections scaled toward the
    # teacher (s * theta, the largest s meeting the same label-free budget).
    'PC_JSCALE_B40': {'kind': 'scaled', 'arm': 'JOINT_TREE', 'label': 'E20', 'budget': dict(_BASE['budget']), 'platt': False,
                      'matched_twin': 'PC_TSCALE_B40'},
    # matched task-only control of PC_JSCALE_B40: the stored TASK_REFIT E20 income corrections (Stage B's identical loop with
    # lambda = 0) under the same scaling rule (the budget does not bind for them), with the same frozen occupation output.
    'PC_TSCALE_B40': {'kind': 'scaled', 'arm': 'TASK_REFIT', 'label': 'E20', 'budget': dict(_BASE['budget']), 'platt': False},
    # matched-cost shrinkage control: the twin's income log-odds / tau with P1's final BRIER distortion (label-free).
    'PC_TEMP_B40': {'kind': 'temperature', 'base': 'PC_TWIN_B40', 'match': 'P1_PAIR_B40', 'platt': False},
    # P2 (revision 1, about 02:42Z, after the seed-0 first wave): P1 was dominated by PC_JSCALE_B40 at the same budget (pair 0.6180
    # vs 0.6150; MODEL_SELECT Brier increment 0.0041 = 1.37 x its label-free distortion vs 0.0026 = 1.0 x). Test whether the
    # inefficiency is P1's training dynamics: JOINT_TREE's own strength (rho = 1) against the frozen accepted occupation, no
    # per-epoch projection, one projection at the end to the same budget. Same loop, people, batches and epochs as P1, so its
    # exact matched twin is PC_TWIN_B40 (whose projection never binds).
    'P2_JR_B40': {**_BASE, 'rho': 1.0, 'project': 'end', 'matched_twin': 'PC_TWIN_B40'},
    # matched-cost shrinkage control for P2: the twin's income log-odds / tau, tau chosen label-free on MODEL_FIT so that its
    # BRIER distortion against the teacher equals P2's final value ("less confident" vs "less SEX-structured").
    'PC_TEMP_P2': {'kind': 'temperature', 'base': 'PC_TWIN_B40', 'match': 'P2_JR_B40', 'platt': False},
    # P3 (revision 2, 02:46Z (ledger entry 46; its text says "about 02:57Z" in error), after P2 on both seeds): P2 met (a), (b), c1, c2, c3 on both seeds with wide utility
    # margins (min income ratio 4.15 / 3.82; MODEL_SELECT Brier increment 0.82 / 0.69 x its label-free distortion; accuracy
    # better than the teacher on both seeds) but a thin c3 margin on seed 1 (0.0011 in the B cells). P3 spends more of the
    # reserve along the same learned direction: identical training (deterministic; end projection), larger label-free budget
    # BRIER 0.006 / ACC 0.0045 / KL 0.010. Pure budget effect; matched twin PC_TWIN_B40.
    'P3_JR_B60': {**_BASE, 'rho': 1.0, 'project': 'end', 'budget': {'brier': 0.006, 'acc': 0.0045, 'kl': 0.010},
                  'matched_twin': 'PC_TWIN_B40'},
    # matched-cost shrinkage control for P3 (requested by the coordinator, allotment 12 -> 14): the twin's income log-odds /
    # tau with P3's final label-free BRIER distortion.
    'PC_TEMP_P3': {'kind': 'temperature', 'base': 'PC_TWIN_B40', 'match': 'P3_JR_B60', 'platt': False},
    # ordinary comparator at P3's budget (no training): stored JOINT_TREE E20 income corrections scaled to the same budget.
    'PC_JSCALE_B60': {'kind': 'scaled', 'arm': 'JOINT_TREE', 'label': 'E20', 'budget': {'brier': 0.006, 'acc': 0.0045, 'kl': 0.010},
                      'platt': False, 'matched_twin': 'PC_TSCALE_B40'},
}
for _k in ('P1_PAIR_B40', 'P1_PAIR_B50'):
    CONFIGS[_k]['matched_twin'] = 'PC_TWIN_B40'


def config(config_id):
    if config_id not in CONFIGS:
        raise KeyError(f'unknown family-P configuration {config_id}')
    return CONFIGS[config_id]


# ---------------------------------------------------------------- label-free budget statistics (numpy only)
def _logsig(z):
    return -np.logaddexp(0.0, -z)


def budget_stats(raw_t, contrib, s=1.0):
    """Distortion of the income output raw_t + s * contrib against the teacher raw_t (both log-odds, shape (n,))."""
    raw_t = np.asarray(raw_t, dtype=np.float64).reshape(-1); z = raw_t + s * np.asarray(contrib, dtype=np.float64).reshape(-1)
    pt, p = np.exp(_logsig(raw_t)), np.exp(_logsig(z))
    flip = (z > 0) != (raw_t > 0)
    kl = pt * (_logsig(raw_t) - _logsig(z)) + (1 - pt) * (_logsig(-raw_t) - _logsig(-z))
    return {'brier': float(2 * np.mean((p - pt) ** 2)), 'acc': float(np.mean(flip * np.abs(2 * pt - 1))),
            'kl': float(np.mean(kl)), 'flip_rate': float(np.mean(flip)), 'mean_abs_dp': float(np.mean(np.abs(p - pt)))}


def within(st, budget):
    return all(st[k] <= budget[k] for k in budget)


def project_scale(raw_t, contrib, budget, iters=50):
    """Largest s in [0, 1] (bisection) with the budget met; s = 1 when already within budget."""
    st = budget_stats(raw_t, contrib, 1.0)
    if within(st, budget):
        return 1.0, st
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if within(budget_stats(raw_t, contrib, mid), budget):
            lo = mid
        else:
            hi = mid
    st = budget_stats(raw_t, contrib, lo)
    if not within(st, budget):
        raise AssertionError('projection failed to meet the budget')
    return lo, st


def contribution(teacher, leaves, theta):
    """theta[leaves].sum / T for a dim-1 (income) recipient, shape (n,)."""
    th = np.asarray(theta, dtype=np.float64).reshape(teacher.T, teacher.L)
    return th[np.arange(teacher.T)[None, :], leaves].sum(1) / teacher.T


def temperature_for(raw_t, raw_base, target_brier, lo=1.0, hi=64.0, iters=60):
    """tau >= 1 with BRIER distortion of raw_base / tau against the teacher equal to target (bisection; monotone for tau >= 1
    when the base sits close to the teacher)."""
    f = lambda tau: budget_stats(raw_t, raw_base / tau - raw_t, 1.0)['brier']
    if f(lo) >= target_brier:
        return lo
    if f(hi) < target_brier:
        raise ValueError('target distortion not reachable by temperature')
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target_brier:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- training core (array inputs; used by fit and selftest)
def _privacy(terms, w):
    return w['income'] * terms['income'] + w['pair'] * terms['pair']


def calibrate(tr, critics, w, seed):
    """leafcorr rule restricted to the income recipient: median ||grad CE|| and median ||grad P|| over 16 fixed draws."""
    import torch
    rng = random.Random(lc.seed_for('IHS-P-calibration', seed))
    task_rng = np.random.default_rng(lc.seed_for('IHS-P-calibration-task', seed) % 2 ** 32)
    gt, gp = [], []
    for _ in range(16):
        task_rows = torch.from_numpy(task_rng.choice(tr.n, lc.RECIPE['batch_size'], replace=False))
        pr, ps = tr.privacy_batch(rng)
        wires = tr.wires(pr); P = _privacy(critics.terms(wires[0], wires[1], ps), w)
        g = torch.autograd.grad(P, tr.theta[0], allow_unused=True)[0]
        gp.append(0.0 if g is None else float(g.norm()))
        gt.append(float(torch.autograd.grad(tr.ce(0, task_rows), tr.theta[0])[0].norm()))
    return {'task': statistics.median(gt), 'privacy': statistics.median(gp), 'draws_task': gt, 'draws_privacy': gp}


def train_income(seed, cfg, te_inc, te_occ, X, names, y_inc, y_occ, sex, row_id, theta_occ, theta_init=None, verbose=True):
    """Train the income corrections on the given (MODEL_FIT) people with the occupation corrections frozen at theta_occ."""
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    n = len(y_inc)
    L_inc, L_occ = te_inc.route(X, names), te_occ.route(X, names)
    raw_inc, raw_occ = te_inc.raw_from_leaves(L_inc), te_occ.raw_from_leaves(L_occ)
    recs = [lc.Recipient('income', te_inc, L_inc, raw_inc), lc.Recipient('occupation', te_occ, L_occ, raw_occ)]
    probe = _order_rows(row_id, 'IHS-P-probe', min(512, n)); reference = _order_rows(row_id, 'IHS-P-reference', min(256, n))
    tr = lc.Trainer(seed, recs, [np.asarray(y_inc, dtype=np.int64), np.asarray(y_occ, dtype=np.int64)], np.arange(n),
                    np.asarray(sex, dtype=np.int64), probe, reference)
    with torch.no_grad():
        tr.theta[1].copy_(torch.from_numpy(np.asarray(theta_occ, dtype=np.float64).reshape(tuple(tr.theta[1].shape))))
        if theta_init is not None:
            tr.theta[0].copy_(torch.from_numpy(np.asarray(theta_init, dtype=np.float64).reshape(tuple(tr.theta[0].shape))))
        occ_trainer = np.concatenate([torch.softmax(recs[1].logits(tr.theta[1], torch.arange(i, min(n, i + 2000))), 1).numpy()
                                      for i in range(0, n, 2000)])
    occ_frozen = te_occ.probabilities_from_raw(te_occ.raw_from_leaves(L_occ, theta_occ))
    occ_gap = float(np.max(np.abs(occ_trainer - occ_frozen)))
    if occ_gap > 1e-10:
        raise AssertionError(f'trainer occupation wire differs from the frozen occupation output ({occ_gap})')
    theta_occ_t = tr.theta[1].detach().clone()
    raw_t = raw_inc[:, 0]
    w, budget = cfg['weights'], cfg['budget']
    critics = tr.warm()
    cal = calibrate(tr, critics, w, seed)
    lam = 0.0 if cfg['rho'] == 0 else cfg['rho'] * cal['task'] / max(cal['privacy'], 1e-300)
    if not np.isfinite(lam) or lam < 0 or lam > 1e6:
        raise FloatingPointError(f'lambda outside range: {lam}')
    opt = torch.optim.AdamW([tr.theta[0]], lr=lc.RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=lc.RECIPE['weight_decay'])
    task_rng = np.random.default_rng(lc.seed_for('IHS-P-task', seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('IHS-P-privacy', seed)); critic_rng = random.Random(lc.seed_for('IHS-P-critic', seed))
    Bsz, S = lc.RECIPE['batch_size'], lc.RECIPE['steps_per_epoch']
    s0, st0 = project_scale(raw_t, contribution(te_inc, L_inc, tr.theta[0].detach().numpy()), budget)
    if s0 < 1.0:
        with torch.no_grad():
            tr.theta[0].mul_(s0)
    log = [{'epoch': 0, 'scale': s0, 'budget_stats': st0}]
    pen = cfg.get('penalty')                 # optional Lagrangian distortion penalty (label-free), dual-updated each epoch
    mu = float(pen['mu0']) if pen else 0.0
    pT = torch.from_numpy(1.0 / (1.0 + np.exp(-raw_t)))
    for epoch in range(cfg['epochs']):
        tr.refresh(critics)
        order = task_rng.permutation(n)
        risks = []
        for b in range(S):
            task_rows = torch.from_numpy(order[(b * Bsz) % n:(b * Bsz) % n + Bsz]) if n >= Bsz * S else \
                torch.from_numpy(order[np.arange(b * Bsz, (b + 1) * Bsz) % n])
            pr, ps = tr.privacy_batch(priv_rng)
            if lam > 0:
                wires = tr.wires(pr); terms = critics.terms(wires[0], wires[1], ps); P = _privacy(terms, w)
                g = torch.autograd.grad(P, tr.theta[0], allow_unused=True)[0]
                pg = torch.zeros_like(tr.theta[0]) if g is None else g
                risks.append({v: float(terms[v].detach()) for v in lc.VIEWS})
            else:
                pg = torch.zeros_like(tr.theta[0])
            g_t = torch.autograd.grad(tr.ce(0, task_rows), tr.theta[0])[0]
            if pen:
                p1 = torch.softmax(recs[0].logits(tr.theta[0], task_rows), 1)[:, 1]
                g_d = torch.autograd.grad(2 * torch.mean((p1 - pT[task_rows]) ** 2), tr.theta[0])[0]
            else:
                g_d = 0.0
            opt.zero_grad(set_to_none=True)
            tr.theta[0].grad = g_t + lam * pg + mu * g_d
            opt.step()
            for _ in range(lc.RECIPE['critic_rounds_per_recipient_update']):
                tr.critic_round(critics, critic_rng)
        pre = budget_stats(raw_t, contribution(te_inc, L_inc, tr.theta[0].detach().numpy()), 1.0)
        if pen:                              # dual update on the binding-type budget ratio (largest of the three)
            ratio = max(pre[k] / budget[k] for k in budget)
            mu = float(min(pen.get('mu_max', 1e6), max(pen.get('mu_min', 1e-3), mu * np.exp(pen['eta'] * (ratio - 1.0)))))
        if cfg.get('project', 'every_epoch') == 'every_epoch' or epoch == cfg['epochs'] - 1:
            s, st = project_scale(raw_t, contribution(te_inc, L_inc, tr.theta[0].detach().numpy()), budget)
        else:                                # 'end': no projection until the last epoch
            s, st = 1.0, pre
        if s < 1.0:
            with torch.no_grad():
                tr.theta[0].mul_(s)
        if not torch.equal(tr.theta[1].detach(), theta_occ_t):
            raise AssertionError('frozen occupation corrections changed')
        with torch.no_grad():
            ce_in = float(tr.ce(0, torch.arange(n)))
        mean_risk = {v: float(np.mean([r[v] for r in risks])) for v in lc.VIEWS} if risks else {}
        log.append({'epoch': epoch + 1, 'pre_projection': pre, 'scale': s, 'budget_stats': st, 'mean_train_critic_risk': mean_risk,
                    'in_sample_income_ce_model_fit': ce_in, 'mu': mu})
        if verbose:
            print(json.dumps({'epoch': epoch + 1, 'scale': round(s, 4), 'pre_brier': round(pre['brier'], 5),
                              'brier': round(st['brier'], 5), 'acc': round(st['acc'], 5), 'kl': round(st['kl'], 5),
                              'flip': round(st['flip_rate'], 4), 'risk': {k: round(v, 4) for k, v in mean_risk.items()},
                              'ce_in': round(ce_in, 5), 'mu': round(mu, 4)}), flush=True)
    return {'theta_income': tr.theta[0].detach().numpy().copy(), 'lambda': lam, 'calibration': cal, 'epochs_log': log,
            'occupation_wire_check_max_abs': occ_gap, 'critic_rounds': critics.rounds, 'final_critic_sha256': critics.state_sha()}


def _order_rows(row_id, salt, n):
    import hashlib
    ranked = sorted(range(len(row_id)), key=lambda i: hashlib.sha256(f'{salt}|{int(row_id[i])}'.encode()).hexdigest())
    return np.asarray(ranked[:n], dtype=np.int64)


# ---------------------------------------------------------------- Platt map (CALIBRATION income labels only)
def fit_platt(raw_cal, y_cal):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=10000, tol=1e-12, solver='lbfgs').fit(np.asarray(raw_cal, dtype=np.float64).reshape(-1, 1), y_cal)
    return {'a': float(m.coef_[0, 0]), 'b': float(m.intercept_[0])}


# ---------------------------------------------------------------- release from features alone
def income_raw(seed, config_id, X, names, _cache=None):
    """Released income log-odds of a configuration, from the admitted features only."""
    from . import common as C
    from . import devrun as DR
    cfg = config(config_id)
    te = C.teacher(seed, 'income')
    leaves = te.route(X, names)
    d = DR.release_dir(FAMILY, config_id, seed)
    if cfg['kind'] == 'train':
        with np.load(d / 'THETA.npz') as z:
            th = z['theta_income'].copy()
        raw = te.raw_from_leaves(leaves, th)[:, 0]
        rec = json.loads((d / 'RELEASE.json').read_text())
        if rec.get('platt'):
            raw = rec['platt']['a'] * raw + rec['platt']['b']
        return raw
    if cfg['kind'] == 'temperature':
        rec = json.loads((d / 'RELEASE.json').read_text())
        return income_raw(seed, cfg['base'], X, names) / rec['tau']
    if cfg['kind'] == 'scaled':
        rec = json.loads((d / 'RELEASE.json').read_text())
        th = C.stage_b_theta(seed, cfg['arm'], cfg['label'])[0]
        return te.raw_from_leaves(leaves)[:, 0] + rec['scale'] * contribution(te, leaves, th)
    raise ValueError(cfg['kind'])


def predict(seed, config_id, X, names):
    """{'income': (n, 2), 'occupation': (n, 5)} float64 probability vectors recomputed from the admitted features alone."""
    from . import common as C
    te_o = C.teacher(seed, 'occupation')
    th_occ = C.stage_b_theta(seed, 'JOINT_TREE', 'E20')[1]
    return {'income': trees.probabilities('income', income_raw(seed, config_id, X, names)),
            'occupation': te_o.probabilities_from_raw(te_o.raw_from_leaves(te_o.route(X, names), th_occ))}


# ---------------------------------------------------------------- real-data fit (MODEL_FIT; run through ihs.jobqueue)
def reference_distortions(seed, te, leaves, raw_t):
    """Label-free distortion of the stored LOCAL_TREE / JOINT_TREE income outputs against the teacher on the same people."""
    from . import common as C
    out = {}
    for arm in ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT'):
        for lab in ('E5', 'E10', 'E20'):
            th = C.stage_b_theta(seed, arm, lab)[0]
            out[f'{arm}_{lab}'] = budget_stats(raw_t, contribution(te, leaves, th), 1.0)
    return out


def refdist(seeds=(0, 1)):
    """Label-free pre-fit diagnostic: the budget statistics of stored income outputs against the teacher, on MODEL_FIT and
    CALIBRATION features (no label of any role is read). Aggregates only."""
    from . import common as C
    out = {'what': 'label-free income distortion against the T_TREE income teacher (2*mean dp^2, expected accuracy loss, KL) of '
                   'stored outputs; features only; for checking the family-P budget against published MODEL_SELECT increments',
           'utc': C.now(), 'per_seed': {}}
    for seed in seeds:
        te = C.teacher(seed, 'income')
        res = {}
        for role in ('MODEL_FIT', 'CALIBRATION'):
            Fx = C.load_role(role, ('features',))
            L = te.route(Fx['X'], Fx['feature_names']); raw_t = te.raw_from_leaves(L)[:, 0]
            r = {}
            for arm in ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT'):
                for lab in ('E2', 'E5', 'E10', 'E20'):
                    r[f'{arm}_{lab}'] = budget_stats(raw_t, contribution(te, L, C.stage_b_theta(seed, arm, lab)[0]), 1.0)
            for name, cfgid in (('RETIRED_B3CTL', 'B3CTL_UNCOUPLED_MIX50_C6A'), ('B2TWIN', 'B2CTL_UNCOUPLED_C6A')):
                p1 = C.amd_release(cfgid, seed, roles=(role,))[role]['income'][:, 1]
                p1 = np.clip(p1, 1e-300, 1 - 1e-16)
                r[name] = budget_stats(raw_t, np.log(p1) - np.log1p(-p1) - raw_t, 1.0)
            res[role] = r
        out['per_seed'][str(seed)] = res
    C.write_json(C.RESULTS / 'roles' / 'investigator_p' / 'REFERENCE_DISTORTIONS.json', out)
    print(json.dumps({s: {k: round(v['brier'], 5) for k, v in out['per_seed'][s]['MODEL_FIT'].items()} for s in out['per_seed']}))


def fit(seed, config_id):
    from . import common as C
    from . import bases as B
    from . import devrun as DR
    cfg = config(config_id)
    t_cpu = time.process_time()
    F = C.load_role('MODEL_FIT', ('features', 'income', 'occupation', 'sex') if cfg['kind'] == 'train' else ('features',))
    names, X = F['feature_names'], F['X']
    te_i, te_o = C.teacher(seed, 'income'), C.teacher(seed, 'occupation')
    stored = C.stage_a_outputs(seed)
    if not np.array_equal(stored['row_id__MODEL_FIT'], F['row_id']):
        raise AssertionError('MODEL_FIT row order differs from the stored teacher outputs')
    L_i = te_i.route(X, names); raw_t = te_i.raw_from_leaves(L_i)[:, 0]
    if np.max(np.abs(trees.probabilities('income', raw_t) - stored['T_TREE__MODEL_FIT__income'])) > 1e-12:
        raise AssertionError('zero-correction identity failed')
    th_joint = C.stage_b_theta(seed, 'JOINT_TREE', 'E20')
    roles_X = {r: (X if r == 'MODEL_FIT' else C.load_role(r, ('features',))['X']) for r in ROLES}
    occ_ref = B.joint_occupation(seed, ROLES)               # accepted occupation output, recomputed and checked vs storage
    refd = reference_distortions(seed, te_i, L_i, raw_t)
    rec = {'config': cfg, 'matched_twin': cfg.get('matched_twin')}
    if cfg['kind'] == 'train':
        theta_init = None
        if cfg['init'] != 'teacher':
            theta_init = C.stage_b_theta(seed, cfg['init'], 'E20')[0]
        res = train_income(seed, cfg, te_i, te_o, X, names, F['income'], F['occupation'], F['sex'], F['row_id'], th_joint[1],
                           theta_init)
        th = res['theta_income']
        raw_of = {r: te_i.raw_from_leaves(te_i.route(roles_X[r], names), th)[:, 0] for r in ROLES}
        platt = None
        if cfg['platt']:
            Cal = C.load_role('CALIBRATION', ('features', 'income'))
            platt = fit_platt(raw_of['CALIBRATION'], Cal['income'])
            raw_of = {r: platt['a'] * v + platt['b'] for r, v in raw_of.items()}
        rec.update({k: res[k] for k in ('lambda', 'calibration', 'epochs_log', 'occupation_wire_check_max_abs', 'critic_rounds',
                                         'final_critic_sha256')})
        rec['platt'] = platt
        n_fits = 1 + 9 + (1 if platt else 0)
        mech = ('PAIO: income leaf-score corrections on the frozen T_TREE income routing trained on MODEL_FIT with online SEX '
                'critics on the income wire and the linked pair wire (frozen JOINT_TREE E20 occupation wire), explicit label-free '
                'distortion budget against the teacher by projection; occupation = accepted JOINT_TREE E20 output (frozen)')
        if cfg['rho'] == 0:
            mech = 'PAIO matched task-only twin (identical loop, lambda = 0; critics updated but unused); occupation = JOINT_TREE E20 (frozen)'
        labels = 'MODEL_FIT income (task loss); MODEL_FIT SEX (online critics only)' + ('; CALIBRATION income (Platt map)' if platt else '')
        roles_read = ('MODEL_FIT (features, income, occupation, SEX); features of CALIBRATION, MODEL_SELECT, ATTACK_FIT, '
                      'ATTACK_VALIDATION, DEV_REPORT (label-free, release computation)' + ('; CALIBRATION income (Platt map)' if platt else ''))
    elif cfg['kind'] == 'temperature':
        bd = DR.release_dir(FAMILY, cfg['base'], seed); md = DR.release_dir(FAMILY, cfg['match'], seed)
        with np.load(bd / 'THETA.npz') as z:
            th_base = z['theta_income'].copy()
        target = json.loads((md / 'RELEASE.json').read_text())['final_budget_stats_model_fit']['brier']
        raw_base = te_i.raw_from_leaves(L_i, th_base)[:, 0]
        tau = temperature_for(raw_t, raw_base, target)
        raw_of = {r: te_i.raw_from_leaves(te_i.route(roles_X[r], names), th_base)[:, 0] / tau for r in ROLES}
        rec.update({'tau': tau, 'target_brier_distortion': target, 'base_theta_sha256': C.arr_sha(th_base)})
        th = None
        n_fits = 1
        mech = f"matched-cost temperature control: {cfg['base']} income log-odds / tau, tau chosen label-free on MODEL_FIT features to match {cfg['match']}'s BRIER distortion against the teacher; occupation = JOINT_TREE E20 (frozen)"
        labels = 'none (tau from label-free MODEL_FIT features; the base twin used MODEL_FIT income and SEX-free task loss)'
        roles_read = 'MODEL_FIT (features); features of every role (release computation); stored family-P twin and matched-config records'
    elif cfg['kind'] == 'scaled':
        th_arm = C.stage_b_theta(seed, cfg['arm'], cfg['label'])[0]
        scale, st = project_scale(raw_t, contribution(te_i, L_i, th_arm), cfg['budget'])
        raw_of = {}
        for r in ROLES:
            Lr = te_i.route(roles_X[r], names)
            raw_of[r] = te_i.raw_from_leaves(Lr)[:, 0] + scale * contribution(te_i, Lr, th_arm)
        rec.update({'scale': scale, 'scaled_budget_stats_model_fit': st, 'arm_theta_sha256': C.arr_sha(th_arm)})
        th = None
        n_fits = 1
        mech = f"ordinary comparator: stored {cfg['arm']} {cfg['label']} income corrections scaled toward the teacher to the same label-free budget (no training); occupation = JOINT_TREE E20 (frozen)"
        labels = 'none (scale from label-free MODEL_FIT features; the stored corrections were trained in Stage B on MODEL_FIT)'
        roles_read = 'MODEL_FIT (features); features of every role (release computation); stored Stage B corrections'
    else:
        raise ValueError(cfg['kind'])
    outs = {r: {'income': trees.probabilities('income', raw_of[r]), 'occupation': occ_ref[r]} for r in ROLES}
    final_mf = budget_stats(raw_t, raw_of['MODEL_FIT'] - raw_t, 1.0)
    L_cal = te_i.route(roles_X['CALIBRATION'], names); raw_t_cal = te_i.raw_from_leaves(L_cal)[:, 0]
    final_cal = budget_stats(raw_t_cal, raw_of['CALIBRATION'] - raw_t_cal, 1.0)
    rec.update({
        'mechanism': mech, 'inputs_used': 'admitted 145-column encoding of AGEP, WKHP, COW, SCHL, MAR, POBP only (income via the '
        'T_TREE income routing; occupation via the T_TREE occupation routing + stored JOINT_TREE E20 corrections)',
        'roles_read': roles_read, 'labels_used_at_training': labels, 'n_model_fits': n_fits,
        'n_model_fits_note': 'income corrections 1 + online critics 9 (3 views x linear/MLP32/MLP64; occupation-view critics have no gradient path to income) + Platt 1 if used; temperature control: 1 (tau)',
        'final_budget_stats_model_fit': final_mf, 'final_budget_stats_calibration_features': final_cal,
        'reference_distortions_model_fit': refd,
        'rebuild': 'ihs.fam_p.predict(seed, config_id, X, names): T_TREE routing + THETA.npz theta_income (+ platt a, b) [+ / tau]; '
                   'occupation = T_TREE occupation routing + stage_b_theta(seed, JOINT_TREE, E20)[1]',
        'fit_cpu_s': time.process_time() - t_cpu})
    d = DR.save_release(FAMILY, config_id, seed, outs, rec)
    if th is not None:
        np.savez(d / 'THETA.npz', theta_income=th)
    # deployment parity: recompute two roles from features alone through predict() and compare bitwise
    parity = {}
    for r in ('MODEL_SELECT', 'DEV_REPORT'):
        p = predict(seed, config_id, roles_X[r], names)
        parity[r] = {t: bool(np.array_equal(p[t], outs[r][t])) for t in ('income', 'occupation')}
    if not all(all(v.values()) for v in parity.values()):
        raise AssertionError(f'predict() parity failed: {parity}')
    print(json.dumps({'saved': str(d), 'final_mf': final_mf, 'final_cal_features': final_cal, 'parity': parity,
                      'cpu_s': round(time.process_time() - t_cpu, 1)}))


# ---------------------------------------------------------------- invented-data self test (no real data)
def _fake_teacher(task, rng, T, n_features, names):
    dim = 1 if task == 'income' else 5
    payload = {'task': task, 'depth': 4, 'split_features': rng.integers(0, n_features, size=(T, 4)).tolist(),
               'split_borders': rng.normal(size=(T, 4)).astype(np.float32).tolist(),
               'leaf_values': (0.3 * rng.normal(size=(T, 16, dim))).tolist(), 'scale': 1.0, 'bias': [0.0] * dim,
               'n_features': n_features, 'feature_names': list(names)}
    return trees.TreeTeacher(payload)


def selftest():
    rng = np.random.default_rng(7)
    n, nf = 1200, 10
    names = tuple(f'f{i}' for i in range(nf))
    sex = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, nf)) + 0.8 * sex[:, None] * (np.arange(nf) < 3)       # invented SEX-associated features
    te_i, te_o = _fake_teacher('income', rng, 30, nf, names), _fake_teacher('occupation', rng, 30, nf, names)
    raw_i = te_i.raw_from_leaves(te_i.route(X, names))[:, 0]
    y_inc = (rng.random(n) < 1 / (1 + np.exp(-raw_i))).astype(np.int64)
    y_occ = rng.integers(0, 5, size=n)
    theta_occ = 0.5 * rng.normal(size=(te_o.T * te_o.L, 5))
    row_id = np.arange(10_000, 10_000 + n)
    checks = {}
    # 1. budget statistics: zero at the teacher, monotone along the scale path, projection meets the budget
    c = 0.8 * rng.normal(size=n)
    z = budget_stats(raw_i, np.zeros(n)); checks['zero_distortion_at_teacher'] = all(abs(v) < 1e-15 for v in z.values())
    path = [budget_stats(raw_i, c, s)['brier'] for s in np.linspace(0, 1, 11)]
    checks['brier_monotone_on_scale_path'] = bool(np.all(np.diff(path) >= -1e-15))
    bud = {'brier': 0.5 * path[-1], 'acc': 1.0, 'kl': 1.0}
    s, st = project_scale(raw_i, c, bud); checks['projection_meets_budget'] = bool(st['brier'] <= bud['brier'] and 0 < s < 1)
    checks['projection_is_tight'] = bool(abs(st['brier'] - bud['brier']) < 1e-6)
    s1, _ = project_scale(raw_i, 0.001 * c, bud); checks['projection_inactive_when_within'] = s1 == 1.0
    # 2. Brier identity: with calibrated labels, the summed-Brier increase approaches 2 * mean(dp^2) (large invented sample)
    m = 400_000; rt = rng.normal(size=m); cc = 0.5 * rng.normal(size=m); yy = (rng.random(m) < 1 / (1 + np.exp(-rt)))
    pt, pn = 1 / (1 + np.exp(-rt)), 1 / (1 + np.exp(-(rt + cc)))
    inc = np.mean(2 * (pn - yy) ** 2) - np.mean(2 * (pt - yy) ** 2)
    checks['brier_identity_relative_error'] = float(abs(inc - budget_stats(rt, cc)['brier']) / budget_stats(rt, cc)['brier'])
    checks['brier_identity_ok'] = checks['brier_identity_relative_error'] < 0.02
    # 3. temperature solver hits its target
    tau = temperature_for(raw_i, raw_i + 0.05 * c, 0.01)
    checks['temperature_hits_target'] = bool(abs(budget_stats(raw_i, (raw_i + 0.05 * c) / tau - raw_i)['brier'] - 0.01) < 1e-6)
    # 4. a short training run: occupation stays frozen, the budget holds every epoch, the twin differs only by lambda
    cfg = {**_BASE, 'epochs': 3, 'budget': {'brier': 0.01, 'acc': 0.01, 'kl': 0.02}}
    r = train_income(0, cfg, te_i, te_o, X, names, y_inc, y_occ, sex, row_id, theta_occ, verbose=False)
    tw = train_income(0, {**cfg, 'rho': 0.0}, te_i, te_o, X, names, y_inc, y_occ, sex, row_id, theta_occ, verbose=False)
    checks['occupation_wire_frozen'] = r['occupation_wire_check_max_abs'] <= 1e-10
    checks['budget_held_every_epoch'] = all(within(e['budget_stats'], cfg['budget']) for e in r['epochs_log'])
    checks['lambda_positive'] = r['lambda'] > 0; checks['twin_lambda_zero'] = tw['lambda'] == 0.0
    checks['twin_differs_from_p'] = bool(not np.array_equal(r['theta_income'], tw['theta_income']))
    r2 = train_income(0, cfg, te_i, te_o, X, names, y_inc, y_occ, sex, row_id, theta_occ, verbose=False)
    checks['deterministic_rerun'] = bool(np.array_equal(r['theta_income'], r2['theta_income']))
    checks['critic_risk_logged'] = all('pair' in e['mean_train_critic_risk'] for e in r['epochs_log'][1:])
    rp = train_income(0, {**cfg, 'penalty': {'mu0': 1.0, 'eta': 1.0}}, te_i, te_o, X, names, y_inc, y_occ, sex, row_id,
                      theta_occ, verbose=False)
    checks['penalty_budget_held'] = all(within(e['budget_stats'], cfg['budget']) for e in rp['epochs_log'])
    checks['penalty_dual_updates'] = len({round(e.get('mu', -1), 12) for e in rp['epochs_log'][1:]}) > 1
    checks['penalty_changes_solution'] = bool(not np.array_equal(rp['theta_income'], r['theta_income']))
    re_ = train_income(0, {**cfg, 'rho': 1.0, 'project': 'end'}, te_i, te_o, X, names, y_inc, y_occ, sex, row_id, theta_occ,
                       verbose=False)
    checks['end_projection_final_within_budget'] = within(re_['epochs_log'][-1]['budget_stats'], cfg['budget'])
    checks['end_projection_inactive_before_last'] = all(e['scale'] == 1.0 for e in re_['epochs_log'][1:-1])
    # 5. release contract on invented features
    raw = te_i.raw_from_leaves(te_i.route(X, names), r['theta_income'])[:, 0]
    p = trees.probabilities('income', raw)
    checks['probability_contract'] = bool(np.allclose(p.sum(1), 1, atol=1e-12, rtol=0) and (p >= 0).all())
    cal = fit_platt(raw, y_inc); checks['platt_finite'] = bool(np.isfinite([cal['a'], cal['b']]).all())
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'checks': checks}, default=float))
    return ok, checks


def summary(config_ids, seeds=(0, 1)):
    """Compact view of the aggregate EVAL_MS.json files of family-P configurations (no data access)."""
    from . import devrun as DR
    rows = []
    for cid in config_ids:
        for s in seeds:
            p = DR.release_dir(FAMILY, cid, s) / 'EVAL_MS.json'
            if not p.is_file():
                continue
            m = json.loads(p.read_text()); u = m['utility']
            inc = {f"{c['reference']}|{c['metric']}": [round(c['value'], 5), round(c['se'], 5), round(c['margin_se_units'], 2)]
                   for c in u['clauses'] if c['task'] == 'income'}
            px = m['proxy']['summary']
            cmp = m.get('comparisons', {})
            rec = json.loads((DR.release_dir(FAMILY, cid, s) / 'RELEASE.json').read_text())
            rows.append({'config': cid, 'seed': s, 'point_pass_12': u['point_pass_12'], 'guard_pass': u['guard_pass'],
                         'income[value,se,margin_se]': inc,
                         'guards': {g['group']: [round(g['value'], 5), round(g['se'], 5)] for g in u['guards']},
                         'proxy': {v: {'mean': round(px[v]['mean'], 4), 'min': round(px[v]['min'], 4), 'max': round(px[v]['max'], 4)} for v in px},
                         'pair_cells': {c: round(v, 4) for c, v in px['pair']['cells'].items()},
                         'benefit': {r: {'pair_mean': round(b['pair']['mean'], 4), 'pair_min': round(b['pair']['min'], 4),
                                         'inc_recipient_max': round(b['income']['max'], 4), 'occ_recipient_max': round(b['occupation']['max'], 4)}
                                     for r, b in cmp.items()},
                         'reserve': {k: m['reserve'][k] for k in ('status', 'failed') if k in m.get('reserve', {})},
                         'pair_readers_cell_A0': {k: round(v, 4) for k, v in m['proxy']['cells']['A0']['pair'].items()},
                         'income_readers_cell_A0': {k: round(v, 4) for k, v in m['proxy']['cells']['A0']['income'].items()},
                         'distinct': m['distinct_outputs'],
                         'label_free_final_mf': {k: round(v, 5) for k, v in rec.get('final_budget_stats_model_fit', {}).items()},
                         'fit_cpu_s': round(rec.get('fit_cpu_s', 0), 1)})
    print(json.dumps(rows, indent=1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    sub.add_parser('selftest'); sub.add_parser('refdist')
    sm = sub.add_parser('summary'); sm.add_argument('configs', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'fit':
        fit(a.seed, a.config)
    elif a.cmd == 'refdist':
        refdist()
    elif a.cmd == 'summary':
        summary(a.configs)
    else:
        ok, _ = selftest()
        raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

"""Protection stage (privacy stage, PRIV24 / CTRL12) of the PCRL overnight occupation-headroom campaign (protection investigator).
EXPLORATORY DEVELOPMENT on previously used records; not confirmation.

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column
encoding X). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded
demographic column (sex0 = men, sex1 = women); readers are separate evaluation classifiers that estimate SEX from the released
vectors. SEX is a training-time label on MODEL_FIT only; never a runtime input or output.
Mechanism card: results/pcrl_occupation_headroom_overnight_v1/roles/investigator_prot/CARD.md.

Release of one configuration (seed s), computed from X alone (the income output is H3's frozen output, not part of this module):
  CONT / LOC / PAIR   p'(x) = softmax(log p_W(x) + scale * Delta_theta(x)),  Delta_theta(x) = sum_t theta[t, leaf_t(x), :] / T on the
                      frozen T_TREE occupation routing of seed s (acs_replication.leafcorr parametrization; theta = 0 gives W);
                      p_W = the task-only witness's own deployed output (ohn.fam_<f>.predict).
  COMP                p'(x) = the nearest K-means centroid of p_W(x) (label-free code fitted on the witness's cross-fitted MODEL_FIT
                      outputs).
  decision = argmax p'.
Training of theta (MODEL_FIT only): base = the witness's household-cross-fitted MODEL_FIT output (CROSSFIT.npz); task CE on all
people + per-SEX summed-Brier Lagrangian (dual ascent per epoch toward per-SEX targets; SEX a training label) + lambda * critic risk
(LOC: occupation wire; PAIR: occupation wire and the linked pair wire with H3's frozen income wire; CONT: lambda = 0, critics updated
but unused). The pipeline is cross-fitted over the 5 OHN1_crossfit household folds; the deployed correction (trained on all
MODEL_FIT people) is scaled by the largest s whose out-of-fold costs meet the budget (f x the witness's MODEL_SELECT reserve slack
of each occupation-side condition).

Commands (numerical ones through `python3 -m ohn.jobqueue run ...`; the trained arms need <repository>/.venv/bin/python):
  selftest                                                            invented data only
  dryrun --seed s                                                     real FEATURES, invented labels and SEX; nothing saved
  fit --witness F:C --arm CONT|LOC|PAIR --seed s --config STEM [--levels 0.5,0.8]
  fit --witness F:C --arm COMP --seed s --config STEM --match-stem PAIR_STEM [--levels 0.5,0.8]
  verify --seed s --config ID                                         predict() == saved release, bitwise, all six roles
  gridspec --bank ID --seed s --witness F:C --configs A,B [--extra-comparators A,B]
`predict(seed, config_id, X, names)` recomputes a saved configuration's occupation release from features alone.
MODEL_SELECT labels are never read here (only `python3 -m ohn.devrun score` and `ohn.devreaders` read them).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from acs_replication import trees
from . import common as C

FAMILY = 'PROT'
TRAIN_FAMILY = 'PROT_TRAIN'                       # training artifacts (fold thetas, logs); not releases
K_OCC = 5
ROLES = C.ROLES6
TRAINED = ('CONT', 'LOC', 'PAIR')
ARMS = TRAINED + ('COMP',)
VIEWS_OF = {'CONT': (), 'LOC': ('occupation',), 'PAIR': ('occupation', 'pair')}
KIND_OF = {'CONT': 'control', 'LOC': 'privacy', 'PAIR': 'privacy', 'COMP': 'control'}
BUDGET_CLASS = {'CONT': 'CTRL12', 'LOC': 'PRIV24', 'PAIR': 'PRIV24', 'COMP': 'CTRL12'}
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
RECIPE = {
    'schema': 'OHN1-protection-recipe-v1',
    'parametrization': 'p = softmax(log p_W + scale * theta[leaves].sum / T) on the frozen T_TREE occupation routing; theta = 0 is W',
    'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01,
    'batch_size': 256, 'steps_per_epoch': 40, 'epochs': 20, 'rho': 1.0, 'calibration_draws': 16,
    'warm_critic_rounds': 80, 'critic_rounds_per_update': 5, 'critics': 'acs_replication.leafcorr.Critics (linear, mlp32, mlp64 per view)',
    'guard_rule': ('equal-utilization Lagrangian (card addendum B): u_g = in-sample Inc_g / S_g; tau_g = max(abs_floor, S_g x max(u_other, 0)); '
                   'mu_g <- min(mu_max, max(0, mu_g + eta x (Inc_g / tau_g - 1))) once per epoch; S_g = witness reserve slack of guard g'),
    'guard_eta': 1.0, 'guard_mu_max': 5.0, 'guard_abs_floor': 5e-4,
    'levels': (0.5, 0.8), 'projection_grid': 101, 'projection_bisection': 30,
    'reference_rows': 256, 'eval_rows': 1024, 'chunk': 2048,
    'crossfit': 'ohn.common.crossfit_folds (OHN1_crossfit, 5 household folds); 5 fold pipelines + 1 full pipeline',
}
COMP_LADDER = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024,
               1536, 2048)
SLACK_OF = {'men_brier': ('guard|sex0',), 'women_brier': ('guard|sex1',),
            'brier': ('occupation|U0|brier_excess', 'occupation|Ucal|brier_excess'),
            'nll': ('occupation|U0|nll_excess', 'occupation|Ucal|nll_excess'),
            'acc_loss': ('occupation|U0|accuracy_loss', 'occupation|Ucal|accuracy_loss')}


# ---------------------------------------------------------------- small numerics (numpy)
def stream_seed(*items):
    return int(hashlib.sha256('|'.join(map(str, ('OHN-PROT',) + tuple(items))).encode()).hexdigest()[:15], 16)


def center(z):
    z = np.asarray(z, dtype=np.float64)
    return z - z.mean(1, keepdims=True)


def logp(p):
    return center(np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-300)))


def probs(z):
    return trees.probabilities('occupation', z)


def correction(theta, leaves, T, L, chunk=RECIPE['chunk']):
    """Delta_theta(x) = theta[t, leaf_t(x)].sum_t / T, (n, 5) float64 (the leafcorr correction term; fixed chunking)."""
    th = np.asarray(theta, dtype=np.float64).reshape(T, L, K_OCC)
    leaves = np.asarray(leaves, dtype=np.int64)
    out = np.empty((len(leaves), K_OCC), dtype=np.float64)
    t = np.arange(T)[None, :]
    for i in range(0, len(leaves), chunk):
        out[i:i + chunk] = th[t, leaves[i:i + chunk]].sum(1) / T
    return out


def release_from(p_w, delta, scale):
    """The trained arms' released occupation vectors (the one code path used by fit and predict)."""
    return probs(logp(p_w) + float(scale) * np.asarray(delta, dtype=np.float64))


def assign(P, cents, chunk=512):
    """Nearest centroid (squared Euclidean in the simplex), elementwise arithmetic only (no BLAS), ties to the first."""
    P = np.asarray(P, dtype=np.float64); cents = np.asarray(cents, dtype=np.float64)
    out = np.empty(len(P), dtype=np.int64)
    for i in range(0, len(P), chunk):
        d = ((P[i:i + chunk, None, :] - cents[None, :, :]) ** 2).sum(2)
        out[i:i + chunk] = d.argmin(1)
    return out


def code_release(p_w, cents):
    return np.asarray(cents, dtype=np.float64)[assign(p_w, cents)]


def income_wire(inc):
    inc = np.asarray(inc, dtype=np.float64)
    return np.column_stack((inc, inc.argmax(1).astype(np.float64)))


def cost_stats(p, base, y, sex):
    """Protected minus base, per person then averaged: men's / women's / all summed-Brier increase, NLL increase, accuracy loss."""
    v = C.task_vectors(y, p, K_OCC)
    db, dn, da = v['brier'] - base['brier'], v['nll'] - base['nll'], base['accuracy'] - v['accuracy']
    men, women = sex == 0, sex == 1
    return {'men_brier': float(db[men].mean()), 'women_brier': float(db[women].mean()), 'brier': float(db.mean()),
            'nll': float(dn.mean()), 'acc_loss': float(da.mean())}


def cost_stats_se(p, base, y, sex, unit):
    from ihs.evaluate import cluster_se, subgroup_se
    v = C.task_vectors(y, p, K_OCC)
    db, dn, da = v['brier'] - base['brier'], v['nll'] - base['nll'], base['accuracy'] - v['accuracy']
    return {'men_brier': subgroup_se(db, sex == 0, unit), 'women_brier': subgroup_se(db, sex == 1, unit), 'brier': cluster_se(db, unit),
            'nll': cluster_se(dn, unit), 'acc_loss': cluster_se(da, unit)}


def feasible(st, budget):
    return all(st[k] <= budget[k] for k in STATS)


def utilization(st, budget):
    u = 0.0
    for k in STATS:
        b = budget[k]
        u = max(u, st[k] / b if b > 0 else (math.inf if st[k] > b else 0.0))
    return u


def distortion(p0, p1):
    """Label-free distortion of p1 against p0: summed squared change, KL(p0 || p1), decision flip rate."""
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    kl = (p0 * (np.log(np.maximum(p0, 1e-300)) - np.log(np.maximum(p1, 1e-300)))).sum(1)
    return {'sq': float(((p1 - p0) ** 2).sum(1).mean()), 'kl': float(kl.mean()), 'flip': float((p0.argmax(1) != p1.argmax(1)).mean()),
            'max_abs': float(np.abs(p1 - p0).max())}


# ---------------------------------------------------------------- end projection (out-of-fold costs)
def project(z0, delta, y, sex, budget, grid=RECIPE['projection_grid'], iters=RECIPE['projection_bisection']):
    """Largest s in [0, 1] such that every grid point 0, 1/(grid-1), ..., up to s is feasible, refined by bisection toward the first
    infeasible grid point. Costs are measured against the base at s = 0 (exactly zero there)."""
    base = C.task_vectors(y, probs(z0), K_OCC)

    def at(s):
        return cost_stats(probs(z0 + s * delta), base, y, sex)
    curve, lo, hi = [], 0.0, None
    for i in range(grid):
        s = i / (grid - 1)
        st = at(s)
        curve.append({'s': s, **st, 'utilization': utilization(st, budget), 'feasible': feasible(st, budget)})
    for c in curve:
        if not c['feasible']:
            hi = c['s']
            break
        lo = c['s']
    if hi is not None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if feasible(at(mid), budget):
                lo = mid
            else:
                hi = mid
    st = at(lo)
    return lo, st, curve, base


# ---------------------------------------------------------------- training core (torch; arrays in, arrays out)
class _Net:
    def __init__(self, torch, base_z, leaves, T, L):
        self.torch, self.T = torch, T
        self.idx = torch.from_numpy((np.arange(T)[None, :] * L + np.asarray(leaves, dtype=np.int64)).astype(np.int64))
        self.base = torch.from_numpy(np.ascontiguousarray(base_z, dtype=np.float64))
        self.theta = torch.zeros(T * L, K_OCC, dtype=torch.float64, requires_grad=True)

    def logits(self, rows):
        r = self.base[rows] + self.theta[self.idx[rows]].sum(1) / self.T
        return r - r.mean(1, keepdim=True)

    def wire(self, rows):
        p = self.torch.softmax(self.logits(rows), 1)
        return self.torch.cat((p, p.argmax(1, keepdim=True).to(p.dtype).detach()), 1)


def train_correction(base_z, leaves, y, sex, inc_wire, T, L, arm, seed, fold, targets, epochs=None, steps=None, rho=None,
                     verbose=False, guard='equal'):
    """Train theta for one arm on the given people (MODEL_FIT, or a fold-excluded subset). Returns theta and the training log.
    guard = 'equal' (P1: equal-utilization per-SEX Brier Lagrangian) or 'none' (revision 1, P2: no in-training guard term)."""
    if guard not in ('equal', 'none', 'static'):
        raise ValueError(guard)
    import torch
    import torch.nn.functional as F
    from acs_replication import leafcorr as lc
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if arm not in TRAINED:
        raise ValueError(arm)
    R = RECIPE
    for mine, theirs in (('warm_critic_rounds', 'warm_critic_rounds'), ('critic_rounds_per_update', 'critic_rounds_per_recipient_update')):
        if lc.RECIPE[theirs] != R[mine]:
            raise AssertionError(f'critic recipe differs from leafcorr: {mine}')
    epochs = R['epochs'] if epochs is None else int(epochs)
    steps = R['steps_per_epoch'] if steps is None else int(steps)
    rho = R['rho'] if rho is None else float(rho)
    views = VIEWS_OF[arm]
    n, B = len(y), R['batch_size']
    t0 = time.process_time()
    net = _Net(torch, base_z, leaves, T, L)
    Y = torch.from_numpy(np.asarray(y, dtype=np.int64)); S = torch.from_numpy(np.asarray(sex, dtype=np.int64))
    INC = torch.from_numpy(np.ascontiguousarray(inc_wire, dtype=np.float64))
    onehot = F.one_hot(Y, K_OCC).to(torch.float64)
    with torch.no_grad():
        b0 = net.base - net.base.mean(1, keepdim=True)
        base_brier = ((torch.softmax(b0, 1) - onehot) ** 2).sum(1)
        base_ce = F.cross_entropy(b0, Y, reduction='none')
    men, women = S == 0, S == 1
    slack_w = {'men': max(float(targets['men']), 1e-6), 'women': max(float(targets['women']), 1e-6)}   # guard slack weights S_g
    if guard == 'static':                                  # revision 2 (P3): fixed per-person CE weights proportional to 1 / S_g
        W = torch.where(men, 1.0 / slack_w['men'], 1.0 / slack_w['women']).to(torch.float64)
        W = W / W.mean()
        sex_weights = {'men': float(W[men][0]) if bool(men.any()) else None, 'women': float(W[women][0]) if bool(women.any()) else None}
    else:
        W, sex_weights = None, None

    def task_ce(lg, rows):
        if W is None:
            return F.cross_entropy(lg, Y[rows])
        return (F.cross_entropy(lg, Y[rows], reduction='none') * W[rows]).mean()
    ref_rows = torch.from_numpy(np.sort(np.random.default_rng(stream_seed(seed, fold, 'reference') % 2 ** 32).choice(n, min(n, R['reference_rows']), replace=False)))
    eval_rows = torch.from_numpy(np.sort(np.random.default_rng(stream_seed(seed, fold, 'evaluation') % 2 ** 32).choice(n, min(n, R['eval_rows']), replace=False)))
    critics = lc.Critics(stream_seed(seed, fold, 'critics') % 2 ** 31, S)

    def privacy_batch(rng):
        rows = torch.tensor(rng.sample(range(n), B), dtype=torch.int64)
        return rows, S[rows]

    def critic_round(rng):
        rows, s_ = privacy_batch(rng)
        with torch.no_grad():
            w = net.wire(rows)
        critics.update(INC[rows], w, s_)

    def refresh():
        with torch.no_grad():
            critics.refresh(INC[ref_rows], net.wire(ref_rows))

    def in_sample():
        ce = torch.empty(n, dtype=torch.float64); br = torch.empty(n, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, n, R['chunk']):
                rows = torch.arange(i, min(n, i + R['chunk']))
                lg = net.logits(rows)
                ce[rows] = F.cross_entropy(lg, Y[rows], reduction='none')
                br[rows] = ((torch.softmax(lg, 1) - onehot[rows]) ** 2).sum(1)
        d = br - base_brier
        return {'ce': float(ce.mean()), 'ce_increase': float((ce - base_ce).mean()), 'brier_increase': float(d.mean()),
                'men_brier_increase': float(d[men].mean()), 'women_brier_increase': float(d[women].mean())}

    refresh()
    warm_rng = random.Random(stream_seed(seed, fold, 'warm'))
    for _ in range(R['warm_critic_rounds']):
        critic_round(warm_rng)
    warm_sha = critics.state_sha()
    # strength calibration at the exact start (theta = 0) with warm critics (leafcorr rule, own objective)
    cal_rng = random.Random(stream_seed(seed, fold, 'calibration'))
    cal_task = np.random.default_rng(stream_seed(seed, fold, 'calibration-task') % 2 ** 32)
    gt, gp = [], []
    for _ in range(R['calibration_draws']):
        trows = torch.from_numpy(cal_task.choice(n, min(n, B), replace=False))
        prow, ps = privacy_batch(cal_rng)
        g = torch.autograd.grad(task_ce(net.logits(trows), trows), net.theta)[0]
        gt.append(float(g.norm()))
        if views:
            terms = critics.terms(INC[prow], net.wire(prow), ps)
            g = torch.autograd.grad(sum(terms[v] for v in views), net.theta, allow_unused=True)[0]
            gp.append(0.0 if g is None else float(g.norm()))
    lam = 0.0 if not views else rho * statistics.median(gt) / max(statistics.median(gp), 1e-300)
    if not (math.isfinite(lam) and 0.0 <= lam < 1e6):
        raise FloatingPointError(f'lambda outside the registered range: {lam}')
    opt = torch.optim.AdamW([net.theta], lr=R['lr'], betas=R['betas'], eps=R['eps'], weight_decay=R['weight_decay'])
    task_rng = np.random.default_rng(stream_seed(seed, fold, 'task') % 2 ** 32)
    priv_rng = random.Random(stream_seed(seed, fold, 'privacy'))
    critic_rng = random.Random(stream_seed(seed, fold, 'critic-rounds'))
    mu = {'men': 0.0, 'women': 0.0}
    start = in_sample()
    log = []
    for epoch in range(epochs):
        refresh()
        order = task_rng.permutation(n)
        risks, ratios = [], []
        mu_used = dict(mu)
        for b in range(steps):
            trows = torch.from_numpy(order[np.arange(b * B, (b + 1) * B) % n])
            prow, ps = privacy_batch(priv_rng)             # drawn in every arm (identical streams; unused in CONT)
            g_p = None
            if views:
                terms = critics.terms(INC[prow], net.wire(prow), ps)
                g = torch.autograd.grad(sum(terms[v] for v in views), net.theta, allow_unused=True)[0]
                g_p = torch.zeros_like(net.theta) if g is None else g
                risks.append({v: float(terms[v].detach()) for v in ('income', 'occupation', 'pair')})
            lg = net.logits(trows)
            loss = task_ce(lg, trows)
            if mu['men'] > 0 or mu['women'] > 0:
                br = ((torch.softmax(lg, 1) - onehot[trows]) ** 2).sum(1)
                sb = S[trows]
                for name, code in (('men', 0), ('women', 1)):
                    m = sb == code
                    if mu[name] > 0 and bool(m.any()):
                        loss = loss + mu[name] * br[m].mean()
            g_t = torch.autograd.grad(loss, net.theta)[0]
            opt.zero_grad(set_to_none=True)
            net.theta.grad = g_t if g_p is None else g_t + lam * g_p
            if g_p is not None:
                ratios.append(float(lam * g_p.norm()) / max(float(g_t.norm()), 1e-300))
            opt.step()
            for _ in range(R['critic_rounds_per_update']):
                critic_round(critic_rng)
        st = in_sample()
        with torch.no_grad():
            ev = critics.terms(INC[eval_rows], net.wire(eval_rows), S[eval_rows])
        u = {g: st[f'{g}_brier_increase'] / slack_w[g] for g in ('men', 'women')}
        tau = {g: max(R['guard_abs_floor'], slack_w[g] * max(u[h], 0.0)) for g, h in (('men', 'women'), ('women', 'men'))}
        for name in ('men', 'women'):
            if guard == 'equal':
                mu[name] = min(R['guard_mu_max'], max(0.0, mu[name] + R['guard_eta'] * (st[f'{name}_brier_increase'] / tau[name] - 1.0)))
        rec = {'epoch': epoch + 1, 'mu_used': mu_used, 'mu_next': dict(mu), 'tau': tau, 'utilization_in_sample': u, 'in_sample': st,
               'train_critic_risk': {v: float(np.mean([r[v] for r in risks])) for v in ('income', 'occupation', 'pair')} if risks else {},
               'eval_critic_risk': {v: float(ev[v]) for v in ('income', 'occupation', 'pair')},
               'eval_critic_ces': {v: ev[v + '_ces'] for v in ('income', 'occupation', 'pair')},
               'effective_ratio_median': float(np.median(ratios)) if ratios else 0.0,
               'theta_norm': float(net.theta.detach().norm())}
        log.append(rec)
        if verbose:
            print(json.dumps({'fold': fold, 'epoch': epoch + 1, 'mu': {k: round(v, 3) for k, v in mu.items()},
                              'inc_men': round(st['men_brier_increase'], 5), 'inc_women': round(st['women_brier_increase'], 5),
                              'ce_inc': round(st['ce_increase'], 5), 'risk': {k: round(v, 4) for k, v in rec['eval_critic_risk'].items()},
                              'ratio': round(rec['effective_ratio_median'], 3)}), flush=True)
    return {'theta': net.theta.detach().numpy().copy(), 'lambda': lam, 'rho': rho if views else 0.0,
            'calibration': {'median_task_grad': statistics.median(gt), 'median_privacy_grad': statistics.median(gp) if gp else None,
                            'task_draws': gt, 'privacy_draws': gp},
            'targets_tau': {'guard_slack_weights': slack_w}, 'start_in_sample': start, 'epochs_log': log, 'warm_critic_sha256': warm_sha,
            'final_critic_sha256': critics.state_sha(), 'critic_rounds': critics.rounds, 'n': n, 'fold': fold, 'arm': arm,
            'epochs': epochs, 'steps_per_epoch': steps, 'guard': guard, 'sex_weights': sex_weights, 'cpu_s': time.process_time() - t0}


def crossfit_train(base_z, leaves, y, sex, inc_wire, folds, T, L, arm, seed, targets, k=None, verbose=False, **kw):
    """The entire training pipeline on MODEL_FIT minus each fold (predicting that fold) and on all MODEL_FIT (deployed)."""
    k = C.CROSSFIT_K if k is None else k
    n = len(y)
    delta_oof = np.full((n, K_OCC), np.nan)
    fold_res = {}
    for f in range(k):
        tr, te_ = folds != f, folds == f
        r = train_correction(base_z[tr], leaves[tr], y[tr], sex[tr], inc_wire[tr], T, L, arm, seed, f'fold{f}', targets,
                             verbose=verbose, **kw)
        delta_oof[te_] = correction(r['theta'], leaves[te_], T, L)
        fold_res[f] = r
    if not np.isfinite(delta_oof).all():
        raise AssertionError('out-of-fold correction incomplete')
    full = train_correction(base_z, leaves, y, sex, inc_wire, T, L, arm, seed, 'full', targets, verbose=verbose, **kw)
    return full, fold_res, delta_oof


def fold_summary(r):
    last = r['epochs_log'][-1] if r['epochs_log'] else {}
    return {'fold': r['fold'], 'n': r['n'], 'lambda': r['lambda'], 'cpu_s': r['cpu_s'], 'final_mu': last.get('mu_next'),
            'final_in_sample': last.get('in_sample'), 'final_eval_critic_risk': last.get('eval_critic_risk'),
            'mu_trajectory': [e['mu_next'] for e in r['epochs_log']], 'warm_critic_sha256': r['warm_critic_sha256'],
            'final_critic_sha256': r['final_critic_sha256'], 'theta_sha256': C.arr_sha(r['theta'])}


# ---------------------------------------------------------------- COMP: label-free K-means code
def kmeans_code(P, K, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(K), init='k-means++', n_init=1, max_iter=300, tol=1e-6,
                random_state=stream_seed(seed, 'comp', int(K)) % 2 ** 31, algorithm='lloyd')
    km.fit(np.asarray(P, dtype=np.float64))
    cents = np.asarray(km.cluster_centers_, dtype=np.float64)
    if (cents <= 0).any() or not np.isfinite(cents).all():
        raise FloatingPointError('centroid outside the open simplex')
    cents = cents / cents.sum(1, keepdims=True)
    return cents, {'K': int(K), 'inertia': float(km.inertia_), 'n_iter': int(km.n_iter_)}


def comp_select(P, y, sex, budget, target_u, seed, ladder=COMP_LADDER):
    """Smallest ladder K whose out-of-fold utilization is <= target_u (binary search assuming utilization falls with K; every
    evaluated K recorded). If even the largest K exceeds target_u, the largest K is returned and flagged."""
    base = C.task_vectors(y, np.asarray(P, dtype=np.float64), K_OCC)
    seen = {}

    def ev(i):
        if i not in seen:
            cents, info = kmeans_code(P, ladder[i], seed)
            st = cost_stats(code_release(P, cents), base, y, sex)
            seen[i] = {'K': ladder[i], 'stats': st, 'utilization': utilization(st, budget), 'info': info, 'cents': cents}
        return seen[i]
    lo, hi = 0, len(ladder) - 1
    if ev(hi)['utilization'] > target_u:
        best, flag = hi, 'TARGET_NOT_REACHED_LARGEST_K'
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if ev(mid)['utilization'] <= target_u:
                hi = mid
            else:
                lo = mid + 1
        best, flag = lo, 'MATCHED'
    path = [{'K': v['K'], 'utilization': v['utilization'], **{k: v['stats'][k] for k in STATS}} for _, v in sorted(seen.items())]
    return seen[best], path, flag, base


# ---------------------------------------------------------------- internal MODEL_FIT recovery diagnostic (never a selection rule)
def recovery_diag(outputs, inc_wire, sex, folds):
    """Fresh SEX readers on out-of-fold MODEL_FIT outputs: fitted on folds 0-2, scored on folds 3-4 (AUC)."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    fit, sc = folds <= 2, folds >= 3
    inc = np.asarray(inc_wire, dtype=np.float64)
    inc_feat = np.column_stack((np.log(np.maximum(inc[:, 1], 1e-300)) - np.log(np.maximum(inc[:, 0], 1e-300)), inc[:, 2]))
    res = {}

    def readers(Xv):
        out = {}
        for rname, model in (('LR', LogisticRegression(C=1.0, max_iter=5000)),
                             ('HGB', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0))):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(Xv[fit], sex[fit])
            out[rname] = float(roc_auc_score(sex[sc], model.predict_proba(Xv[sc])[:, 1]))
        return out
    res['income_view_H3_in_sample'] = readers(inc_feat)
    for name, p in outputs.items():
        occ_feat = np.column_stack((logp(p), np.eye(K_OCC)[np.asarray(p).argmax(1)]))
        res[name] = {'occupation': readers(occ_feat), 'pair': readers(np.column_stack((occ_feat, inc_feat)))}
    res['note'] = ('fresh logistic / HGB readers on out-of-fold MODEL_FIT outputs (fit folds 0-2, score folds 3-4); H3 income is '
                   'in-sample on MODEL_FIT; diagnostic only (the development reader grid on MODEL_SELECT is the evidence)')
    return res


# ---------------------------------------------------------------- witness interface and budgets (real data)
def parse_witness(w):
    f, c = w.split(':', 1)
    if f not in ('A', 'B', 'C'):
        raise ValueError('witness family must be A, B or C')
    return f, c


def witness_predict(family, config_id, seed, X, names):
    return importlib.import_module(f'ohn.fam_{family.lower()}').predict(seed, config_id, X, names)


def witness_slack(m, seed):
    """Per occupation-side condition of the witness (frozen reserve rule, one seed): slack_c = margin_c - 2 x se_used_c (reserve
    levels) and the point margin_c = cap - value (point-cap levels)."""
    from . import rules as RU
    res = RU.reserve({seed: m})
    conds = res['per_seed'][seed]['conditions']
    k = float(res['multiplier_k'])
    names = sorted({c for v in SLACK_OF.values() for c in v})
    return ({n: float(conds[n]['margin'] - k * conds[n]['se_used']) for n in names},
            {n: float(conds[n]['margin']) for n in names}, res)


def budget_at(slack, f):
    return {stat: float(f) * min(slack[c] for c in conds) for stat, conds in SLACK_OF.items()}


def parse_levels(spec):
    """'0.5,0.8,m0.9' -> [('B50', 'reserve', 0.5), ('B80', 'reserve', 0.8), ('M90', 'margin', 0.9)]: a reserve level spends f x the
    reserve slack (margin - 2 se_used); a margin level spends f x the point margin (cap - value; a frontier point beyond the reserve)."""
    out = []
    for tok in (spec.split(',') if isinstance(spec, str) else spec):
        tok = str(tok).strip()
        kind, f = ('margin', float(tok[1:])) if tok.startswith('m') else ('reserve', float(tok))
        out.append((('M' if kind == 'margin' else 'B') + f'{int(round(100 * f))}', kind, f))
    return out


def train_dir(stem, seed):
    return C.RUNS / TRAIN_FAMILY / stem / f'S{seed}'


def _roles_X(F):
    out = {}
    for r in ROLES:
        out[r] = F['X'] if r == 'MODEL_FIT' else C.load_role(r, ('features',))['X']
    return out


def _operation(arm, witness, scale=None, K=None):
    wf, wc = witness
    if arm == 'COMP':
        return (f'COMP: occupation vector = nearest of K = {K} label-free K-means centroids (fitted on the cross-fitted MODEL_FIT outputs '
                f'of witness {wf}/{wc}) of the witness deployed vector; decision = argmax; income = H3 (frozen, not in this release)')
    what = {'CONT': 'task-only continuation twin (lambda = 0; critics updated but unused)',
            'LOC': 'local occupation recovery penalty (online SEX critics on the occupation wire)',
            'PAIR': 'coalition-aware occupation correction (online SEX critics on the occupation wire and on the linked pair wire with '
                    "H3's frozen income wire)"}[arm]
    return (f'{arm}: occupation vector = softmax(log p_W(x) + {scale:.6f} * Delta_theta(x)); p_W = witness {wf}/{wc} deployed output; '
            f'Delta_theta = leaf-score correction on the frozen T_TREE occupation routing trained on MODEL_FIT ({what}; CE + per-SEX '
            'Brier Lagrangian; base = the witness cross-fitted MODEL_FIT output); scale = end projection on cross-fitted costs; '
            'decision = argmax; income = H3 (frozen, not in this release)')


def _exposure(arm, wrec):
    s = ('MODEL_FIT occupation labels and SEX (training-time labels: task loss, per-SEX guard Lagrangian, critics, cross-fitted '
         'projection); the witness saw its own fitting roles (see its OCC.json exposure); the T_TREE routing is a fixed partition fitted '
         'on all MODEL_FIT; H3 income on MODEL_FIT is in-sample for H3 (PAIR critics only); out of sample on MODEL_SELECT and later '
         'roles for every layer fitted here')
    if arm == 'COMP':
        s = ('label-free code (no label or SEX in the K-means fit); K chosen by out-of-fold MODEL_FIT costs with MODEL_FIT occupation '
             'labels and SEX (matched-utilization rule); the witness saw its own fitting roles')
    return {'summary': s, 'witness_exposure': wrec.get('exposure')}


def fit(witness, arm, seed, stem, levels='0.5,0.8', match_stem=None, twin=None, verbose=True, guard='equal'):
    from . import devrun as D
    t_cpu = time.process_time()
    if arm not in ARMS:
        raise ValueError(arm)
    wf, wc = parse_witness(witness)
    wocc, wrec, wcf = D.load_occupation(wf, wc, seed)
    if wcf is None or 'MODEL_FIT__oof' not in wcf:
        raise PermissionError('the witness has no cross-fitted MODEL_FIT output (CROSSFIT.npz)')
    if wrec.get('kind') not in ('task_model', 'control'):
        raise PermissionError('the witness must be a task-only configuration')
    m = json.loads((D.occ_dir(wf, wc, seed) / 'EVAL_MS_H3.json').read_text())
    slack, margin, wres = witness_slack(m, seed)
    lv = parse_levels(levels)
    order = [t for t, _, _ in lv]
    budgets = {t: budget_at(slack if kind == 'reserve' else margin, f) for t, kind, f in lv}
    targets = {'men': slack['guard|sex0'], 'women': slack['guard|sex1']}          # guard slack weights of the equal-utilization rule
    F = C.load_role('MODEL_FIT', ('features', 'occupation', 'sex'),
                    purpose=f'protection stage {arm} {stem} seed {seed}: task loss, per-SEX guard Lagrangian, SEX critics, cross-fitted projection')
    names = F['feature_names']
    folds = C.crossfit_folds(F['unit'])
    if not np.array_equal(folds, wcf['fold']):
        raise AssertionError('witness cross-fitting folds differ from the registered OHN1_crossfit folds')
    y, sex, unit = F['occupation'], F['sex'], F['unit']
    p_oof = np.asarray(wcf['MODEL_FIT__oof'], dtype=np.float64)
    z_oof = logp(p_oof)
    te = C.teacher(seed, 'occupation')
    roles_X = _roles_X(F)
    leaves = {r: te.route(roles_X[r], names) for r in ROLES}
    p_w = {r: witness_predict(wf, wc, seed, roles_X[r], names) for r in ROLES}
    parity = {r: {'max_abs_diff': float(np.max(np.abs(p_w[r] - wocc[r]))), 'bitwise': bool(np.array_equal(p_w[r], wocc[r]))} for r in ROLES}
    if max(v['max_abs_diff'] for v in parity.values()) > 1e-9:
        raise AssertionError(f'witness predict() under this interpreter differs from its saved release: {parity}')
    inc_wire = income_wire(D.anchor_income('H3', seed, 'MODEL_FIT'))
    common = {'arm': arm, 'stem': stem, 'witness': f'{wf}/{wc}', 'witness_detail': {'family': wf, 'config': wc, 'occ_sha256': wrec['occ_sha256'],
                                                     'crossfit_sha256': wrec.get('crossfit_sha256'), 'interpreter': wrec.get('interpreter')},
              'witness_predict_parity': parity, 'witness_slack': slack, 'witness_margin': margin, 'witness_reserve_status': wres['status'],
              'witness_min_occupation_side_ratio': wres['min_occupation_side_ratio'],
              'levels': [{'tag': t, 'kind': kind, 'fraction': f} for t, kind, f in lv], 'budgets': budgets,
              'budget_provenance': ('MODEL_SELECT-GUIDED DESIGN CHOICE: each budget level is f x the witness\'s MODEL_SELECT reserve slack '
                                    '(margin - 2 x se_used, frozen reserve rule) or f x its point margin (M levels), read from the logged '
                                    'development evaluation EVAL_MS_H3.json of the witness; an adaptive choice from a logged development '
                                    'quantity, not fitting or calibration on MODEL_SELECT (no MODEL_SELECT label enters any fit here)'),
              'guard_targets_tau': targets, 'guard': guard,
              'guard_note': ('equal: in-training equal-utilization per-SEX Brier Lagrangian (P1); none: no in-training guard term (revision 1, '
                             'P2); static: fixed per-person task-CE weights proportional to 1 / S_g, normalized to mean 1 (revision 2, P3); '
                             'the per-SEX constraints of the out-of-fold projection apply in all three'),
              'recipe': RECIPE, 'routing': {'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L},
              'inputs_used': 'admitted 145-column encoding X of AGEP, WKHP, COW, SCHL, MAR, POBP only (witness predict + T_TREE routing)'}
    tdir = train_dir(stem, seed)
    if (tdir / 'TRAIN.json').exists():
        raise PermissionError(f'training record {tdir} exists (append-only; use a new stem)')
    tdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    saved = []
    if arm in TRAINED:
        full, fold_res, delta_oof = crossfit_train(z_oof, leaves['MODEL_FIT'], y, sex, inc_wire, folds, te.T, te.L, arm, seed, targets,
                                                   verbose=verbose, guard=guard)
        theta = full['theta']
        np.savez(tdir / 'TRAIN.npz', theta_full=theta, delta_oof=delta_oof, **{f'theta_fold{f}': r['theta'] for f, r in fold_res.items()})
        delta_role = {r: correction(theta, leaves[r], te.T, te.L) for r in ROLES}
        base_oof = C.task_vectors(y, probs(z_oof), K_OCC)
        proj = {}
        for t in order:
            s, st, curve, base = project(z_oof, delta_oof, y, sex, budgets[t])
            proj[t] = {'scale': s, 'oof_stats': st, 'oof_utilization': utilization(st, budgets[t]),
                       'oof_stats_se': cost_stats_se(probs(z_oof + s * delta_oof), base, y, sex, unit),
                       'oof_stats_at_scale_1': cost_stats(probs(z_oof + delta_oof), base, y, sex),
                       'in_sample_full_correction': cost_stats(probs(z_oof + s * delta_role['MODEL_FIT']), base_oof, y, sex),
                       'curve': [c for c in curve if abs(round(c['s'] * 20) - c['s'] * 20) < 1e-9]}
        # releases: one per distinct scale (identical scales give identical releases; the later level is recorded as an alias)
        by_scale = {}
        for t in order:
            by_scale.setdefault(proj[t]['scale'], []).append(t)
        n_cf = 5 * (1 + 9)
        for s, ts in sorted(by_scale.items(), key=lambda kv: order.index(kv[1][0])):
            if s <= 0.0:
                for t in ts:
                    proj[t]['config_id'] = None
                    proj[t]['note'] = 'NO_RELEASE: the budget admits no protection (scale 0 = the witness)'
                continue
            cid = stem if (arm == 'CONT' and len(by_scale) == 1) else f'{stem}_{ts[0]}'
            occ = {r: release_from(p_w[r], delta_role[r], s) for r in ROLES}
            oof_rel = probs(z_oof + s * delta_oof)
            transfer = {'MODEL_FIT_oof': distortion(p_oof, oof_rel),
                        **{r: distortion(p_w[r], occ[r]) for r in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT')}}
            rdiag = recovery_diag({'W_oof': p_oof, f'{arm}_oof': oof_rel}, inc_wire, sex, folds)
            record = {**common, 'kind': KIND_OF[arm], 'budget_class': BUDGET_CLASS[arm], 'level': ts[0], 'levels_aliased': ts[1:],
                      'scale': s, 'operation': _operation(arm, (wf, wc), scale=s),
                      'roles_read': ('MODEL_FIT (features, occupation, SEX: training-time labels); features of CALIBRATION, MODEL_SELECT, '
                                     'ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT (label-free release computation); witness release and '
                                     'CROSSFIT.npz; witness EVAL_MS_H3.json aggregates (budget slack, adaptive); H3 income on MODEL_FIT'),
                      'exposure': _exposure(arm, wrec), 'n_model_fits': 1 + 9, 'n_calibrator_fits': 1, 'n_crossfit_fits': n_cf,
                      'n_fits_note': 'deployed correction 1 + its 9 online critics (CONT: updated, unused); projection scale 1 (calibrator); '
                                     'cross-fitting: 5 fold pipelines x (1 correction + 9 critics)',
                      'projection': proj,
                      'training_full': {k: full[k] for k in ('lambda', 'rho', 'calibration', 'targets_tau', 'start_in_sample', 'epochs_log',
                                                            'warm_critic_sha256', 'final_critic_sha256', 'critic_rounds', 'cpu_s', 'n')},
                      'training_folds': {str(f): fold_summary(r) for f, r in fold_res.items()},
                      'transfer_label_free': transfer, 'recovery_diagnostic_model_fit': rdiag,
                      'matched_twin': twin if arm != 'CONT' else None,
                      'train_dir': str(tdir.relative_to(C.ROOT)), 'train_npz_sha256': C.sha(tdir / 'TRAIN.npz'),
                      'fit_cpu_s': time.process_time() - t_cpu}
            d = D.save_occupation(FAMILY, cid, seed, occ, record)
            np.savez(d / 'THETA.npz', theta=theta)
            params = {'schema': 'OHN1-prot-params-v1', 'config_id': cid, 'seed': seed, 'arm': arm, 'scale': s, 'level': ts[0],
                      'witness': {'family': wf, 'config': wc, 'occ_sha256': wrec['occ_sha256']},
                      'routing': {'teacher_sha256': te.sha(), 'T': te.T, 'L': te.L}, 'theta_sha256': C.arr_sha(theta),
                      'theta_file_sha256': C.sha(d / 'THETA.npz')}
            (d / 'PARAMS.json').write_text(json.dumps(params, indent=1, sort_keys=True) + '\n')
            for t in ts:
                proj[t]['config_id'] = cid
            saved.append(cid)
        trec = {**common, 'proj': {t: {k: v for k, v in p.items() if k != 'curve'} for t, p in proj.items()},
                'training_full': fold_summary(full), 'training_folds': {str(f): fold_summary(r) for f, r in fold_res.items()},
                'saved_configs': saved, 'cpu_s': time.process_time() - t_cpu, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
        C.write_json(tdir / 'TRAIN.json', trec)
        D.unit_stats(n_model_fits=10 * len(saved), n_calibrator_fits=len(saved), n_crossfit_fits=n_cf)
    else:                                                                # COMP, matched to a PAIR training's utilization per level
        if not match_stem:
            raise ValueError('COMP needs --match-stem (the PAIR training it is matched to)')
        mrec = json.loads((train_dir(match_stem, seed) / 'TRAIN.json').read_text())
        if mrec['witness_detail']['occ_sha256'] != wrec['occ_sha256']:
            raise AssertionError('COMP must use the same witness as its matched configuration')
        n_km = 0
        trec = {**common, 'match_stem': match_stem, 'levels_done': {}}
        for t in order:
            target = mrec['proj'].get(t)
            if target is None or target.get('config_id') is None:
                trec['levels_done'][t] = {'note': 'matched configuration has no release at this level'}
                continue
            if mrec['budgets'][t] != budgets[t]:
                raise AssertionError(f'{t}: COMP budget differs from the matched configuration budget')
            u_t = float(target['oof_utilization'])
            best, path, flag, base = comp_select(p_oof, y, sex, budgets[t], u_t, seed)
            n_km += len(path)
            cents = best['cents']
            cid = f'{stem}_{t}'
            dup = [c for c, v in trec['levels_done'].items() if v.get('K') == best['K'] and v.get('config_id')]
            if dup:
                trec['levels_done'][t] = {'K': best['K'], 'alias_of': trec['levels_done'][dup[0]]['config_id'], 'flag': flag}
                continue
            occ = {r: code_release(p_w[r], cents) for r in ROLES}
            oof_rel = code_release(p_oof, cents)
            st_se = cost_stats_se(oof_rel, base, y, sex, unit)
            transfer = {'MODEL_FIT_oof': distortion(p_oof, oof_rel), **{r: distortion(p_w[r], occ[r]) for r in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT')}}
            rdiag = recovery_diag({'W_oof': p_oof, 'COMP_oof': oof_rel}, inc_wire, sex, folds)
            record = {**common, 'kind': 'control', 'budget_class': 'CTRL12', 'level': t, 'K': best['K'], 'match_flag': flag,
                      'matched_to': {'stem': match_stem, 'config_id': target['config_id'], 'target_utilization': u_t,
                                     'target_oof_stats': target['oof_stats']},
                      'oof_stats': best['stats'], 'oof_stats_se': st_se, 'oof_utilization': best['utilization'], 'ladder_path': path,
                      'kmeans_info': best['info'], 'operation': _operation('COMP', (wf, wc), K=best['K']),
                      'roles_read': ('MODEL_FIT (features; occupation and SEX only to evaluate the out-of-fold matched cost); features of '
                                     'every role (label-free release computation); witness release and CROSSFIT.npz; witness '
                                     'EVAL_MS_H3.json aggregates (budget slack, adaptive); matched PAIR training record'),
                      'exposure': _exposure('COMP', wrec), 'n_model_fits': 0, 'n_calibrator_fits': len(path), 'n_crossfit_fits': 0,
                      'n_fits_note': 'K-means fits of the ladder search (label-free, counted as calibrator-type fits)',
                      'transfer_label_free': transfer, 'recovery_diagnostic_model_fit': rdiag, 'fit_cpu_s': time.process_time() - t_cpu}
            d = D.save_occupation(FAMILY, cid, seed, occ, record)
            np.savez(d / 'CODEBOOK.npz', centroids=cents)
            params = {'schema': 'OHN1-prot-params-v1', 'config_id': cid, 'seed': seed, 'arm': 'COMP', 'level': t, 'K': best['K'],
                      'witness': {'family': wf, 'config': wc, 'occ_sha256': wrec['occ_sha256']}, 'codebook_sha256': C.arr_sha(cents),
                      'codebook_file_sha256': C.sha(d / 'CODEBOOK.npz')}
            (d / 'PARAMS.json').write_text(json.dumps(params, indent=1, sort_keys=True) + '\n')
            trec['levels_done'][t] = {'K': best['K'], 'config_id': cid, 'flag': flag, 'utilization': best['utilization'],
                                      'target_utilization': u_t, 'stats': best['stats']}
            saved.append(cid)
        trec.update({'saved_configs': saved, 'cpu_s': time.process_time() - t_cpu, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)})
        C.write_json(tdir / 'TRAIN.json', trec)
        D.unit_stats(n_calibrator_fits=n_km)
    for cid in saved:
        verify(seed, cid, roles_X=roles_X, names=names)
    out = {'saved': saved, 'cpu_s': round(time.process_time() - t_cpu, 1)}
    if arm in TRAINED:
        out['projection'] = {t: {'scale': round(p['scale'], 4), 'config': p.get('config_id'),
                                 'oof': {k: round(v, 5) for k, v in p['oof_stats'].items()},
                                 'utilization': round(p['oof_utilization'], 3)} for t, p in proj.items()}
        out['lambda_full'] = full['lambda']
    else:
        out['levels'] = trec['levels_done']
    print(json.dumps(out, default=float))
    return out


# ---------------------------------------------------------------- deployment from features
def _params(seed, config_id):
    from . import devrun as D
    d = D.occ_dir(FAMILY, config_id, seed)
    return d, json.loads((d / 'PARAMS.json').read_text())


def predict(seed, config_id, X, names):
    """(n, 5) float64 occupation probability vectors of a saved configuration, recomputed from the admitted features alone."""
    d, P = _params(seed, config_id)
    pw = witness_predict(P['witness']['family'], P['witness']['config'], seed, X, names)
    if P['arm'] in TRAINED:
        te = C.teacher(seed, 'occupation')
        if te.sha() != P['routing']['teacher_sha256'] or te.T != P['routing']['T'] or te.L != P['routing']['L']:
            raise PermissionError('routing teacher differs from the one recorded at fit time')
        if C.sha(d / 'THETA.npz') != P['theta_file_sha256']:
            raise PermissionError('THETA.npz differs from PARAMS.json')
        with np.load(d / 'THETA.npz') as z:
            theta = z['theta'].copy()
        return release_from(pw, correction(theta, te.route(X, names), te.T, te.L), P['scale'])
    if P['arm'] == 'COMP':
        if C.sha(d / 'CODEBOOK.npz') != P['codebook_file_sha256']:
            raise PermissionError('CODEBOOK.npz differs from PARAMS.json')
        with np.load(d / 'CODEBOOK.npz') as z:
            cents = z['centroids'].copy()
        return code_release(pw, cents)
    raise ValueError(P['arm'])


def verify(seed, config_id, roles_X=None, names=None):
    from . import devrun as D
    occ, rec, _ = D.load_occupation(FAMILY, config_id, seed)
    res = {}
    for r in ROLES:
        if roles_X is None:
            Fr = C.load_role(r, ('features',))
            X, nm = Fr['X'], Fr['feature_names']
        else:
            X, nm = roles_X[r], names
        p = predict(seed, config_id, X, nm)
        res[r] = {'bitwise': bool(np.array_equal(p, occ[r])), 'max_abs_diff': float(np.max(np.abs(p - occ[r])))}
    ok = all(v['bitwise'] for v in res.values())
    C.write_json(D.occ_dir(FAMILY, config_id, seed) / 'VERIFY.json',
                 {'ok': ok, 'roles': res, 'utc': C.now(), 'occ_sha256': rec['occ_sha256'], 'code_sha256': C.sha(__file__),
                  'interpreter': sys.executable, 'unit_id': os.environ.get(C.UNIT_ENV)})
    print(json.dumps({'verify': 'PASS' if ok else 'FAIL', 'config': config_id, 'seed': seed, 'roles': res}))
    if not ok:
        raise AssertionError(f'predict() differs from the saved release: {res}')
    return ok


def parity(witness, seed):
    """Label-free: the witness's predict() under THIS interpreter against its saved release, all six roles (max abs diff, bitwise)."""
    from . import devrun as D
    wf, wc = parse_witness(witness)
    wocc, wrec, wcf = D.load_occupation(wf, wc, seed)
    res = {}
    for r in ROLES:
        Fr = C.load_role(r, ('features',))
        p = witness_predict(wf, wc, seed, Fr['X'], Fr['feature_names'])
        res[r] = {'max_abs_diff': float(np.max(np.abs(p - wocc[r]))), 'bitwise': bool(np.array_equal(p, wocc[r]))}
    out = {'witness': witness, 'seed': seed, 'interpreter': sys.executable, 'witness_interpreter': wrec.get('interpreter'),
           'has_crossfit': wcf is not None, 'roles': res}
    print(json.dumps(out))
    return out


# ---------------------------------------------------------------- grid bank specification (coordinator's naming convention)
def gridspec(bank, seed, witness, configs, extra_comparators=()):
    wf, wc = parse_witness(witness)
    wname = f'{wf}/{wc}@H3'
    members = {n: {'type': 'reference', 'name': n} for n in REQUIRED}
    members[wname] = {'type': 'occ', 'family': wf, 'config': wc, 'anchor': 'H3'}
    for c in list(configs) + list(extra_comparators):
        members[f'{FAMILY}/{c}@H3'] = {'type': 'occ', 'family': FAMILY, 'config': c, 'anchor': 'H3'}
    spec = {'bank_id': bank, 'purpose': f'protection-stage candidates (seed {seed}) beside the six required comparators, the matched '
                                        'task-only witness (matched component) and the matched CONT / COMP controls',
            'members': members, 'candidates': [f'{FAMILY}/{c}@H3' for c in configs] + [wname],
            'comparators': list(REQUIRED) + [wname] + [f'{FAMILY}/{c}@H3' for c in extra_comparators], 'fractions': [1.0]}
    p = C.RESULTS / 'grid_specs' / f'{bank}_S{seed}.json'
    if p.exists():
        raise PermissionError(f'{p} exists (append-only)')
    C.write_json(p, spec)
    print(json.dumps({'spec': str(p.relative_to(C.ROOT)), 'members': len(members)}))
    return p


# ---------------------------------------------------------------- invented-data self-test (no real data)
def _fake_teacher(rng, T, names):
    payload = {'task': 'occupation', 'depth': 4, 'split_features': rng.integers(0, len(names), size=(T, 4)).tolist(),
               'split_borders': rng.normal(size=(T, 4)).astype(np.float32).tolist(),
               'leaf_values': (0.3 * rng.normal(size=(T, 16, K_OCC))).tolist(), 'scale': 1.0, 'bias': [0.0] * K_OCC,
               'n_features': len(names), 'feature_names': list(names)}
    return trees.TreeTeacher(payload)


def _sample(P, rng):
    u = rng.uniform(size=len(P))
    return np.minimum((u[:, None] > np.cumsum(P, axis=1)).sum(1), P.shape[1] - 1).astype(np.int64)


def _invented(rng, n=1500, nf=10, T=30, strength=1.2):
    names = tuple(f'f{i}' for i in range(nf))
    unit = np.repeat(np.arange((n + 1) // 2), 2)[:n] + 10_000
    sex = (rng.random(n) < 0.47).astype(np.int64)
    X = rng.normal(size=(n, nf)); X[:, :3] += strength * sex[:, None]
    W = 0.6 * rng.normal(size=(nf, K_OCC)); W[:3, 1] += 0.8; W[:3, 3] -= 0.8
    z = center(X @ W)
    y = _sample(probs(z + 0.4 * rng.normal(size=z.shape)), rng)
    p1 = 1 / (1 + np.exp(-(0.6 * X[:, 0] + 0.3 * X[:, 4])))
    inc = np.column_stack((1 - p1, p1))
    te = _fake_teacher(rng, T, names)
    folds = C.household_folds(unit, 5, 'OHN_PROT_selftest')
    return {'names': names, 'unit': unit, 'sex': sex, 'X': X, 'z': z, 'p': probs(z), 'y': y, 'inc_wire': income_wire(inc), 'te': te,
            'leaves': te.route(X, names), 'folds': folds}


def selftest():
    t0 = time.process_time()
    rng = np.random.default_rng(20261018)
    d = _invented(rng)
    te, leaves, y, sex, z, inc_w, folds = d['te'], d['leaves'], d['y'], d['sex'], d['z'], d['inc_wire'], d['folds']
    T, L = te.T, te.L
    checks = {}
    # 1. correction term: zero at theta = 0, equals the leafcorr / TreeTeacher correction, chunking invariant
    th = 0.7 * rng.normal(size=(T * L, K_OCC))
    c1 = correction(th, leaves, T, L)
    ref = te.raw_from_leaves(leaves, th) - te.raw_from_leaves(leaves)
    checks['correction_zero_at_theta0'] = bool(np.array_equal(correction(np.zeros_like(th), leaves, T, L), np.zeros((len(leaves), K_OCC))))
    checks['correction_matches_teacher_parametrization'] = bool(np.max(np.abs(c1 - ref)) < 1e-12)
    checks['correction_chunk_invariant_bitwise'] = bool(np.array_equal(c1, correction(th, leaves, T, L, chunk=7)))
    checks['logp_probs_round_trip'] = bool(np.max(np.abs(probs(logp(d['p'])) - d['p'])) < 1e-12)
    # 2. cost statistics equal direct computation
    p2 = probs(z + 0.5 * c1)
    base = C.task_vectors(y, d['p'], K_OCC); v = C.task_vectors(y, p2, K_OCC)
    st = cost_stats(p2, base, y, sex)
    direct = (((p2 - np.eye(K_OCC)[y]) ** 2).sum(1) - ((d['p'] - np.eye(K_OCC)[y]) ** 2).sum(1))
    checks['cost_stats_direct'] = bool(abs(st['men_brier'] - direct[sex == 0].mean()) < 1e-12 and abs(st['brier'] - direct.mean()) < 1e-12
                                       and abs(st['nll'] - (v['nll'] - base['nll']).mean()) < 1e-12)
    # 3. short trainings: CONT lambda 0, LOC / PAIR lambda > 0, deterministic, distinct; logs carry critic risk for all views
    tg = {'men': 0.05, 'women': 0.05}
    kw = dict(epochs=3, steps=10)
    r_c = train_correction(z, leaves, y, sex, inc_w, T, L, 'CONT', 0, 'st', tg, **kw)
    r_l = train_correction(z, leaves, y, sex, inc_w, T, L, 'LOC', 0, 'st', tg, **kw)
    r_p = train_correction(z, leaves, y, sex, inc_w, T, L, 'PAIR', 0, 'st', tg, **kw)
    r_p2 = train_correction(z, leaves, y, sex, inc_w, T, L, 'PAIR', 0, 'st', tg, **kw)
    checks['cont_lambda_zero'] = r_c['lambda'] == 0.0
    checks['loc_pair_lambda_positive'] = bool(r_l['lambda'] > 0 and r_p['lambda'] > 0)
    checks['deterministic_rerun_bitwise'] = bool(np.array_equal(r_p['theta'], r_p2['theta']))
    checks['arms_differ'] = bool(not np.array_equal(r_c['theta'], r_l['theta']) and not np.array_equal(r_l['theta'], r_p['theta']))
    checks['critic_risk_logged_all_arms'] = all(set(e['eval_critic_risk']) == {'income', 'occupation', 'pair'}
                                               for r in (r_c, r_l, r_p) for e in r['epochs_log'])
    checks['cont_guard_inactive'] = all(e['mu_next']['men'] == 0.0 and e['mu_next']['women'] == 0.0 for e in r_c['epochs_log'])
    # 4. equal-utilization guard Lagrangian: the men's multiplier rises when men use their (small) slack faster than women
    y_b = _sample(d['p'], rng)                               # labels drawn from the base itself: the base is calibrated
    S_g = {'men': 0.001, 'women': 1.0}
    r_g = train_correction(z, leaves, y_b, sex, inc_w, T, L, 'PAIR', 0, 'st', S_g, epochs=4, steps=10, rho=50.0)
    incs = [e['in_sample']['men_brier_increase'] for e in r_g['epochs_log']]
    checks['guard_dual_activates'] = {'men_increase_by_epoch': incs, 'mu_men_by_epoch': [e['mu_next']['men'] for e in r_g['epochs_log']],
                                      'pass': bool(any(i > 5e-4 for i in incs) and any(e['mu_next']['men'] > 0 for e in r_g['epochs_log']))}
    rule_ok = True
    for e in r_g['epochs_log']:
        inc = {g: e['in_sample'][f'{g}_brier_increase'] for g in ('men', 'women')}
        u = {g: inc[g] / S_g[g] for g in inc}
        tau = {'men': max(5e-4, S_g['men'] * max(u['women'], 0.0)), 'women': max(5e-4, S_g['women'] * max(u['men'], 0.0))}
        for g in ('men', 'women'):
            want = min(5.0, max(0.0, e['mu_used'][g] + (inc[g] / tau[g] - 1.0)))
            rule_ok &= abs(e['mu_next'][g] - want) < 1e-12 and abs(e['tau'][g] - tau[g]) < 1e-15
    checks['guard_dual_rule'] = bool(rule_ok)
    r_gn = train_correction(z, leaves, y_b, sex, inc_w, T, L, 'PAIR', 0, 'st', S_g, epochs=4, steps=10, rho=50.0, guard='none')
    checks['guard_none_inactive'] = bool(all(e['mu_next'] == {'men': 0.0, 'women': 0.0} for e in r_gn['epochs_log'])
                                         and not np.array_equal(r_gn['theta'], r_g['theta']) and r_gn['guard'] == 'none')
    S2 = {'men': 0.002, 'women': 0.004}
    r_s = train_correction(z, leaves, y_b, sex, inc_w, T, L, 'PAIR', 0, 'st', S2, epochs=2, steps=10, guard='static')
    r_s2 = train_correction(z, leaves, y_b, sex, inc_w, T, L, 'PAIR', 0, 'st', S2, epochs=2, steps=10, guard='static')
    r_n2 = train_correction(z, leaves, y_b, sex, inc_w, T, L, 'PAIR', 0, 'st', S2, epochs=2, steps=10, guard='none')
    sw = r_s['sex_weights']
    mean_w = (sw['men'] * (sex == 0).sum() + sw['women'] * (sex == 1).sum()) / len(sex)
    checks['guard_static_weights'] = {'weights': sw, 'pass': bool(abs(sw['men'] / sw['women'] - 2.0) < 1e-12 and abs(mean_w - 1.0) < 1e-12
                                                                  and all(e['mu_next'] == {'men': 0.0, 'women': 0.0} for e in r_s['epochs_log'])
                                                                  and np.array_equal(r_s['theta'], r_s2['theta'])
                                                                  and not np.array_equal(r_s['theta'], r_n2['theta']))}
    # 5. projection: tight when binding, s = 1 when within budget, s = 0 when nothing is affordable
    delta = correction(r_g['theta'], leaves, T, L) + 0.8 * rng.normal(size=(len(y), K_OCC))
    full_cost = cost_stats(probs(z + delta), C.task_vectors(y, probs(z), K_OCC), y, sex)
    bud = {k: 0.3 * abs(full_cost[k]) + 1e-6 for k in STATS}
    bud['acc_loss'] = 10.0                                   # accuracy is a step function of s; tightness is checked on the smooth ones
    s, stp, curve, _ = project(z, delta, y, sex, bud)
    checks['projection_feasible'] = feasible(stp, bud)
    checks['projection_tight_or_full'] = bool(s == 1.0 or utilization(stp, bud) > 0.999)
    s1, _, _, _ = project(z, delta, y, sex, {k: 10.0 for k in STATS})
    checks['projection_full_when_within'] = s1 == 1.0
    harm = -5.0 * np.eye(K_OCC)[y]                           # pushes every person away from the true group: every cost rises
    s0, st0, _, _ = project(z, harm, y, sex, {k: 0.0 for k in STATS})
    checks['projection_zero_budget'] = bool(s0 == 0.0 and feasible(st0, {k: 0.0 for k in STATS}))
    # 6. cross-fitted pipeline: every person gets an out-of-fold correction
    full, fres, d_oof = crossfit_train(z, leaves, y, sex, inc_w, folds, T, L, 'LOC', 0, tg, epochs=1, steps=4)
    checks['crossfit_complete'] = bool(np.isfinite(d_oof).all() and len(fres) == 5 and full['theta'].shape == (T * L, K_OCC))
    checks['crossfit_oof_differs_from_full'] = bool(not np.allclose(d_oof, correction(full['theta'], leaves, T, L)))
    # 7. COMP: deterministic K-means, brute-force assignment, matched selection, probability contract
    P = d['p']
    c_a, _ = kmeans_code(P, 16, 0); c_b, _ = kmeans_code(P, 16, 0)
    checks['kmeans_deterministic'] = bool(np.array_equal(c_a, c_b))
    brute = np.argmin(((P[:, None, :] - c_a[None]) ** 2).sum(2), axis=1)
    checks['assign_matches_brute_force'] = bool(np.array_equal(assign(P, c_a), brute))
    rel = code_release(P, c_a)
    checks['comp_contract'] = bool(np.allclose(rel.sum(1), 1, atol=1e-12, rtol=0) and (rel > 0).all() and len(np.unique(rel, axis=0)) <= 16)
    bud_c = {k: 0.01 for k in STATS}
    c32, _ = kmeans_code(P, 32, 0)
    u32 = utilization(cost_stats(code_release(P, c32), C.task_vectors(y, P, K_OCC), y, sex), bud_c)   # a reachable target
    best, path, flag, _ = comp_select(P, y, sex, bud_c, u32, 0, ladder=(2, 4, 8, 16, 32, 64, 128))
    checks['comp_select_meets_target'] = {'target_u': u32, 'K': best['K'], 'u': best['utilization'], 'flag': flag,
                                          'pass': bool(flag == 'MATCHED' and best['utilization'] <= u32 and best['K'] <= 32)}
    smaller = [q for q in path if q['K'] < best['K']]
    checks['comp_select_smallest_evaluated'] = all(q['utilization'] > u32 for q in smaller)
    # 8. recovery: a longer LOC training lowers held-out SEX recovery from the occupation vector relative to CONT (planted SEX)
    kw8 = dict(epochs=8, steps=20)
    tr_rows = folds <= 2
    rc = train_correction(z[tr_rows], leaves[tr_rows], y[tr_rows], sex[tr_rows], inc_w[tr_rows], T, L, 'CONT', 1, 'rec', tg, **kw8)
    rl = train_correction(z[tr_rows], leaves[tr_rows], y[tr_rows], sex[tr_rows], inc_w[tr_rows], T, L, 'LOC', 1, 'rec', {'men': 1.0, 'women': 1.0}, rho=3.0, **kw8)
    out_c = probs(z + correction(rc['theta'], leaves, T, L)); out_l = probs(z + correction(rl['theta'], leaves, T, L))
    diag = recovery_diag({'CONT': out_c, 'LOC': out_l}, inc_w, sex, folds)
    checks['recovery_loc_below_cont'] = {'CONT_LR': diag['CONT']['occupation']['LR'], 'LOC_LR': diag['LOC']['occupation']['LR'],
                                         'pass': bool(diag['LOC']['occupation']['LR'] < diag['CONT']['occupation']['LR'])}
    # 9. deployment round trip through saved parameters (temporary directory; JSON scale round trip; bitwise)
    tmp = Path(tempfile.mkdtemp(prefix='ohn_prot_selftest_'))
    sc = float(np.float64(0.7123456789012345))
    np.savez(tmp / 'THETA.npz', theta=r_p['theta'])
    (tmp / 'PARAMS.json').write_text(json.dumps({'scale': sc}))
    rel_fit = release_from(d['p'], correction(r_p['theta'], leaves, T, L), sc)
    with np.load(tmp / 'THETA.npz') as zz:
        th2 = zz['theta'].copy()
    sc2 = json.loads((tmp / 'PARAMS.json').read_text())['scale']
    rel_dep = release_from(d['p'], correction(th2, te.route(d['X'], d['names']), T, L), sc2)
    checks['deploy_round_trip_bitwise'] = bool(np.array_equal(rel_fit, rel_dep))
    # 10. level parsing and budget arithmetic (invented slack values)
    checks['parse_levels'] = parse_levels('0.5,0.8,m0.9') == [('B50', 'reserve', 0.5), ('B80', 'reserve', 0.8), ('M90', 'margin', 0.9)]
    sl = {'guard|sex0': 0.002, 'guard|sex1': 0.01, 'occupation|U0|brier_excess': 0.007, 'occupation|Ucal|brier_excess': 0.006,
          'occupation|U0|nll_excess': 0.016, 'occupation|Ucal|nll_excess': 0.015, 'occupation|U0|accuracy_loss': 0.009,
          'occupation|Ucal|accuracy_loss': 0.009}
    bb = budget_at(sl, 0.5)
    checks['budget_at'] = bool(abs(bb['men_brier'] - 0.001) < 1e-15 and abs(bb['brier'] - 0.003) < 1e-15 and abs(bb['nll'] - 0.0075) < 1e-15
                               and abs(bb['women_brier'] - 0.005) < 1e-15 and abs(bb['acc_loss'] - 0.0045) < 1e-15)
    checks['release_contract'] = bool(np.isfinite(rel_fit).all() and (rel_fit >= 0).all() and np.allclose(rel_fit.sum(1), 1, atol=1e-12, rtol=0))
    ok = all((v['pass'] if isinstance(v, dict) else bool(v)) for v in checks.values())
    res = {'schema': 'OHN1-prot-selftest-v1', 'data': 'invented only (numpy default_rng(20261018)); no real data read', 'checks': checks,
           'all_pass': ok, 'lambda': {'LOC': r_l['lambda'], 'PAIR': r_p['lambda']}, 'cpu_s': time.process_time() - t0,
           'code_sha256': C.sha(__file__), 'interpreter': sys.executable}
    C.write_json(C.RESULTS / 'checks' / 'PROT_SELFTEST.json', res)
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'n_checks': len(checks),
                      'failed': [k for k, v in checks.items() if not (v['pass'] if isinstance(v, dict) else bool(v))],
                      'cpu_s': round(res['cpu_s'], 1)}, default=float))
    return ok


# ---------------------------------------------------------------- label-free dry run on real features (invented labels and SEX)
def dryrun(seed, arm='PAIR', folds_to_run=5):
    """Timing and mechanics on real MODEL_FIT features with INVENTED occupation labels and SEX; the stand-in witness is the frozen
    T_TREE occupation output recomputed from the routing (label-free). Nothing is saved as a release; no label of any role is read."""
    from . import devrun as D
    t0 = time.process_time(); w0 = time.time()
    F = C.load_role('MODEL_FIT', ('features',))
    names = F['feature_names']
    te = C.teacher(seed, 'occupation')
    lv = te.route(F['X'], names)
    p_t = te.probabilities_from_raw(te.raw_from_leaves(lv))
    rng = np.random.default_rng(stream_seed(seed, 'dryrun') % 2 ** 32)
    y = _sample(p_t, rng)                                               # invented labels
    wk = F['X'][:, list(names).index('WKHP')]
    sex = (rng.random(len(y)) < 1 / (1 + np.exp(-(-0.8 * wk + 3.0 * (p_t[:, 1] + p_t[:, 2] - p_t[:, 3] - p_t[:, 4]))))).astype(np.int64)
    folds = C.crossfit_folds(F['unit'])
    inc_w = income_wire(D.anchor_income('H3', seed, 'MODEL_FIT'))
    z = logp(p_t)
    timing = {'setup_cpu_s': time.process_time() - t0}
    tg = {'men': 0.002, 'women': 0.008}
    per = []
    delta_oof = np.full((len(y), K_OCC), np.nan)
    last_theta = None
    for f in range(folds_to_run):
        tr = folds != f
        r = train_correction(z[tr], lv[tr], y[tr], sex[tr], inc_w[tr], te.T, te.L, arm, seed, f'dry{f}', tg, verbose=True)
        delta_oof[folds == f] = correction(r['theta'], lv[folds == f], te.T, te.L)
        last_theta = r['theta']
        per.append({'fold': f, 'cpu_s': r['cpu_s'], 'lambda': r['lambda'], 'final': r['epochs_log'][-1]['in_sample'],
                    'risk': r['epochs_log'][-1]['eval_critic_risk'], 'mu': r['epochs_log'][-1]['mu_next']})
    out = {'arm': arm, 'seed': seed, 'per_fold': per, 'timing': timing}
    if folds_to_run == C.CROSSFIT_K:
        bud = {'men_brier': 0.002, 'women_brier': 0.008, 'brier': 0.004, 'nll': 0.008, 'acc_loss': 0.005}
        s, st, curve, _ = project(z, delta_oof, y, sex, bud)
        out['projection'] = {'scale': s, 'stats': st, 'utilization': utilization(st, bud)}
        tr0 = time.process_time()
        rel = {}
        for r in ROLES:                                                  # release path on every role (last fold's theta as stand-in)
            Xr = F['X'] if r == 'MODEL_FIT' else C.load_role(r, ('features',))['X']
            lr = te.route(Xr, names)
            pw = te.probabilities_from_raw(te.raw_from_leaves(lr))
            rel[r] = distortion(pw, release_from(pw, correction(last_theta, lr, te.T, te.L), s))
        out['release_path_distortion'] = rel
        out['release_path_cpu_s'] = time.process_time() - tr0
        out['diag'] = recovery_diag({'W': p_t, f'{arm}_oof': probs(z + s * delta_oof)}, inc_w, sex, folds)
    out['cpu_s'] = time.process_time() - t0; out['wall_s'] = time.time() - w0
    print(json.dumps(out, default=float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    dr = sub.add_parser('dryrun'); dr.add_argument('--seed', type=int, default=0); dr.add_argument('--arm', default='PAIR')
    dr.add_argument('--folds', type=int, default=5)
    f = sub.add_parser('fit'); f.add_argument('--witness', required=True); f.add_argument('--arm', required=True, choices=ARMS)
    f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    f.add_argument('--levels', default='0.5,0.8', help="reserve fractions, and m<f> for point-margin fractions, e.g. 0.5,0.8,m0.9")
    f.add_argument('--match-stem', default=None); f.add_argument('--twin', default=None, help='matched CONT configuration id (recorded)')
    f.add_argument('--guard', default='equal', choices=('equal', 'none', 'static'),
                   help='in-training guard: equal (P1 Lagrangian), none (revision 1, P2), static (revision 2, P3: CE weights 1 / S_g)')
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    pa = sub.add_parser('parity'); pa.add_argument('--witness', required=True); pa.add_argument('--seed', type=int, required=True)
    g = sub.add_parser('gridspec'); g.add_argument('--bank', required=True); g.add_argument('--seed', type=int, required=True)
    g.add_argument('--witness', required=True); g.add_argument('--configs', required=True); g.add_argument('--extra-comparators', default='')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'dryrun':
        dryrun(a.seed, a.arm, a.folds)
        return 0
    if a.cmd == 'fit':
        fit(a.witness, a.arm, a.seed, a.config, a.levels, a.match_stem, a.twin, guard=a.guard)
        return 0
    if a.cmd == 'verify':
        verify(a.seed, a.config)
        return 0
    if a.cmd == 'parity':
        parity(a.witness, a.seed)
        return 0
    gridspec(a.bank, a.seed, a.witness, [c for c in a.configs.split(',') if c], [c for c in a.extra_comparators.split(',') if c])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

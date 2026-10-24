"""FB: pair Fisher-discriminant training of the two U0 task networks (PFD; card FB_PFD_v1, draft by investigator B). PCRL campaign
`pcrl_strong_anchor_release_v1`; EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida
person records; not confirmation.

Starting release: NEURAL_RAW (the U0 networks d-64-64-16-K per task, continued from U0 with the online-critic joint objective, rho 1,
checkpoint E2). Its pair recovery is its occupation view (0.6253 = 0.6253), and along its own path the INCOME clauses bind first
(rho 1 E5, rho 3 E2 and rho 10 E2 fail income Brier first) although the income-view critics read almost nothing (risk(income) ~ 0 on
most logged steps) while the income network's weighted privacy gradient is 1-2x its task gradient (NEURAL_RAW LOGS.json).

Operation (one): both networks start from U0 and are trained with task cross-entropy plus ONE coalition coefficient lambda times
    P = sum over v in (income view, occupation view, linked pair) of [Phi(D_v / sqrt 2) - 1/2],
    D_v^2 = Delta_v' (S_v + eps_v I)^-1 Delta_v
the exact in-batch best response of the quadratic discriminant readers of each view: phi_v = degree-2 polynomial features of the
view's logit differences (income: z1 - z0 -> 2 features; occupation: z_k - z0, k = 1..4 -> 14; pair: 5 -> 20), Delta_v the women-minus-
men mean of phi_v on a 2,048-row privacy batch, S_v the pooled within-SEX covariance, eps_v = 1e-3 tr(S_v) / p_v. Phi(D / sqrt 2) is the
AUC of the Fisher discriminant under Gaussian class conditionals, so P is a recovery (AUC-scale) objective, not a cross-entropy one.
Differences from NEURAL_RAW's objective (acs_replication.neural_raw) and JOINT_TREE's pair critic (acs_replication.leafcorr):
  * no adversary: the reader is re-solved exactly on every privacy batch (no critic lag, no min-with-entropy clamp at 0);
  * full-covariance whitening: the pair discriminant weight S^-1 Delta gives each recipient's coordinates their PARTIAL (incremental)
    weight beyond the other recipient's, where early-stopped, diagonally normalized SGD critics weigh correlated coordinates by
    their marginal mean difference (the income network is pushed for occupation-carried signal);
  * one coalition coefficient for both recipients (NEURAL_RAW calibrates lambda per recipient, which gives income ~2x the occupation
    multiplier: 1.57 / 0.88 and 1.82 / 0.77);
  * D is affine-invariant in the logit differences: no temperature or bias move can lower P.
Strength: rho = 1 (lambda = rho * median ||grad of both task CEs|| / median ||grad P|| over 16 fixed draws at the start); the path
is the epoch checkpoint E1..E10 (40 steps of 256 task rows per epoch; NEURAL_RAW's optimizer: Adam 1e-4, clip 5 per network; NEURAL_RAW's
task-row stream THP1task). Selection (MODEL_FIT only): k* = the largest E_k such that E_1..E_k all keep estimate + 2 household-
clustered SE <= BUDGET on every component of the out-of-fold paired spend over the fold's own start (famrun folds SAR_xf_FB_S<v>; each
fold restarts from the U0 recipe refitted on its training rows).
Configurations (at most 12 = 6 x 2 versions; coordinator adjudication of 2026-10-24):
  FB_PFD                 CAND: PFD at E_k*
  FB_PFD_TASK            CONT (gating): the identical PFD loop with lambda = 0 at E_k* (SEX not used)
  FB_PFD_DQ_LO / _HI     CTRL: SEX-blind subtractive-dithered quantization (sar.ctlkit.dq_fit / dq_release, steps DQ_GRID_FINE) of
                         NEURAL_RAW's release; basis fitted on the cross-fitted RAWJ-E2 MODEL_FIT outputs; deployed on NEURAL_RAW
  FB_PFD_RAWJ_LO / _HI   ALT: NEURAL_RAW's own recipe (online critics, per-recipient lambda_J at rho 1, `run_arm('RAW_J')` step for
                         step) continued beyond E2 (grid E3..E10)
  Both bracket paths have origin START = NEURAL_RAW (out of fold: the cross-fitted RAWJ E2; deployed: NEURAL_RAW rho 1 E2) and use the
  rule of `bracket` on MODEL_FIT out-of-fold total-NLL spends over START (LO <= CAND's spend; HI >= CAND's + 2 SE).
Deployment: `predict` is a float64 numpy forward pass of the saved weights, followed for CTRL points by the shared DQ law
(python3, no torch); training needs the torch venv.

Commands (real data only through the campaign queue after the receipt and the committed card):
    <repository>/.venv/bin/python -m sar.fb.pfd selftest | selftest_stage
    <repository>/.venv/bin/python -m sar.fb.pfd fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FB_PFD_v1.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK

FAMILY = 'FB'
TASKS = ('income', 'occupation')
KK = {'income': 2, 'occupation': 5}
STAGE_A_ROOT = Path('<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/runs')
NEURAL_RAW_CKPT = 'NEURAL_RAW_{v}/NEURAL_RAW__rho1.0__E{e}.pt'
RECIPE = {
    'schema': 'SAR1-FB-PFD-recipe-v1',
    'start': 'deployed fit: U0_S<v> (acs_replication STAGE_A_<v>/U0_MODEL.pt); fold k: the U0 recipe (Recipe A refit from scratch, '
             'admitted epoch count and seed, Adam 1e-3, batch 512) on the fold training rows',
    'learning_rate': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'clip': 5.0,
    'task_batch': 256, 'steps_per_epoch': 40, 'epochs': 10, 'checkpoints': list(range(0, 11)),
    'privacy_batch': 2048, 'ridge': 1e-3, 'rho': 1.0, 'calibration_draws': 16,
    'objective': 'sum_v [Phi(D_v / sqrt 2) - 1/2], v in (income, occupation, pair); D_v^2 = Delta_v^T (S_v + eps_v I)^-1 Delta_v on '
                 'degree-2 polynomial features of the standardized logit differences; pooled within-SEX covariance',
    'task_rows': 'NEURAL_RAW task stream (seed_for("THP1task", seed)); privacy rows: random.Random(seed_for("SAR1pfd-privacy", seed))',
}
STRENGTHS = tuple(range(1, RECIPE['epochs'] + 1))
# component budgets (MODEL_FIT out-of-fold paired spend over the fold start, estimate + 2 SE <= limit); about 0.5-0.6 of each
# benchmark cap (0.010 / 0.010 / 0.005; guards 0.005), above NEURAL_RAW's own MS spend on every component except income Brier (0.0028
# in version 1), which the operation is designed to spend less of
BUDGET = {'occ_nll': 0.0060, 'inc_nll': 0.0050, 'occ_brier': 0.0025, 'inc_brier': 0.0025, 'occ_brier_men': 0.0028,
          'occ_brier_women': 0.0028, 'occ_acc': 0.0060, 'inc_acc': 0.0060}
ORIGIN_EPOCH = 2                       # START = NEURAL_RAW = the RAWJ path at E2 (rho 1)
DQ_TAG = 'SAR_DQ_FB_S{v}'              # keyed dither of the CTRL path (SHA-256 of the six compact fields)
CONFIGS = {
    'FB_PFD': {'kind': 'candidate', 'path': 'CAND',
               'what': 'both U0 task networks trained with the exact in-batch quadratic-discriminant AUC of the linked pair and both '
                       'views (one coalition coefficient, rho 1) at the MODEL_FIT-selected checkpoint E_k*'},
    'FB_PFD_TASK': {'kind': 'twin', 'path': 'CONT',
                    'what': 'zero-protection continuation (gating): the identical PFD loop with lambda = 0 (task cross-entropy only, no '
                            'SEX) at the candidate checkpoint E_k*'},
    'FB_PFD_DQ_LO': {'kind': 'control', 'path': 'CTRL',
                     'what': "CTRL lower bracket point: SEX-blind subtractive-dithered quantization of NEURAL_RAW's release (basis on the "
                             'cross-fitted RAWJ-E2 MODEL_FIT outputs) at the largest out-of-fold spend over START <= CAND\'s'},
    'FB_PFD_DQ_HI': {'kind': 'control', 'path': 'CTRL',
                     'what': "CTRL upper bracket point: SEX-blind subtractive-dithered quantization of NEURAL_RAW's release at the "
                             "smallest out-of-fold spend over START >= CAND's + 2 SE"},
    'FB_PFD_RAWJ_LO': {'kind': 'control', 'path': 'ALT',
                       'what': "ALT lower bracket point: NEURAL_RAW's own recipe continued beyond E2 (online SEX critics, per-recipient "
                               "lambda_J, rho 1) at the largest out-of-fold spend over START <= CAND's"},
    'FB_PFD_RAWJ_HI': {'kind': 'control', 'path': 'ALT',
                       'what': "ALT upper bracket point: NEURAL_RAW's own recipe continued beyond E2 at the smallest out-of-fold spend "
                               "over START >= CAND's + 2 SE"},
}
PURPOSE = ('FB PFD: train the U0 task networks with the closed-form pair Fisher-discriminant objective, its zero-protection '
           'continuation, the NEURAL_RAW-recipe ALT path and the SEX-blind DQ control path of NEURAL_RAW on MODEL_FIT (SEX as a training '
           'label of the protection objectives and of the within-SEX guard components of the strength rule; the DQ path never reads SEX)')


# ---------------------------------------------------------------- deployment (numpy only)
def np_forward(layers, X):
    """Probability vectors of one U0-architecture network from its saved float32 weights, in float64 (the release law)."""
    h = np.asarray(X, dtype=np.float64)
    for j, (w, b) in enumerate(layers):
        h = h @ np.asarray(w, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
        if j < 2:
            h = np.maximum(h, 0.0)
    z = h - h.mean(1, keepdims=True)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def layers_of(state):
    """[(W, b)] float32 numpy copies of the four Linear layers of a U0-architecture state dict (Sequential indices 0, 2, 4, 5)."""
    return [(np.asarray(state[f'{i}.weight'].detach().cpu().numpy() if hasattr(state[f'{i}.weight'], 'detach') else state[f'{i}.weight'],
                        dtype=np.float32).copy(),
             np.asarray(state[f'{i}.bias'].detach().cpu().numpy() if hasattr(state[f'{i}.bias'], 'detach') else state[f'{i}.bias'],
                        dtype=np.float32).copy()) for i in (0, 2, 4, 5)]


def release_weights(W, X):
    return np_forward(W['income'], X), np_forward(W['occupation'], X)


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (sar.deploy contract); deterministic; numpy only (python3)."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    pi, po = release_weights(P['weights'], X)
    if P.get('op', 'net') == 'dq':                       # CTRL: the shared DQ law applied to the deployed NEURAL_RAW vectors
        from .. import ctlkit as CK
        pi, po = CK.dq_release(P['dq_basis'], pi, po, FK.compact(X, names), float(P['delta']), P['dq_tag'])
    elif P.get('op', 'net') != 'net':
        raise ValueError(f'unknown FB payload op {P.get("op")!r}')
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- the closed-form pair objective (torch)
def view_features(z_inc, z_occ):
    import torch
    li = (z_inc[:, 1] - z_inc[:, 0])[:, None]
    lo = z_occ[:, 1:] - z_occ[:, :1]
    return {'income': li, 'occupation': lo, 'pair': torch.cat([li, lo], 1)}


def poly2(u):
    """Degree-2 polynomial features of u (n, d), standardized with detached batch moments (D is affine-invariant in u)."""
    import torch
    s = (u - u.mean(0, keepdim=True).detach()) / (u.std(0, keepdim=True).detach() + 1e-9)
    d = s.shape[1]
    iu = torch.triu_indices(d, d)
    return torch.cat([s, (s[:, :, None] * s[:, None, :])[:, iu[0], iu[1]]], 1)


def fisher_d2(phi, sex, ridge=RECIPE['ridge']):
    """Squared Mahalanobis distance between the women's and men's means of phi under the pooled within-SEX covariance (+ ridge)."""
    import torch
    m = sex.bool()
    a, b = phi[~m], phi[m]
    delta = b.mean(0) - a.mean(0)
    ca, cb = a - a.mean(0), b - b.mean(0)
    S = (ca.T @ ca + cb.T @ cb) / max(len(phi) - 2, 1)
    p = S.shape[0]
    S = S + ridge * torch.trace(S).detach() / p * torch.eye(p, dtype=S.dtype)
    return delta @ torch.linalg.solve(S, delta)


def auc_proxy(d2):
    import torch
    return 0.5 * torch.erf(torch.sqrt(d2 + 1e-8) / 2.0)       # Phi(D / sqrt 2) - 1/2


def privacy_terms(z_inc, z_occ, sex):
    f = view_features(z_inc, z_occ)
    d2 = {v: fisher_d2(poly2(f[v]), sex) for v in ('income', 'occupation', 'pair')}
    return {v: auc_proxy(d2[v]) for v in d2}, d2


def objective(terms):
    return terms['income'] + terms['occupation'] + terms['pair']


# ---------------------------------------------------------------- networks
def network(task, d):
    from acs_replication import neural
    return neural.network(KK[task], d)


def model_from_layers(task, layers):
    import torch
    m = network(task, layers[0][0].shape[1])
    sd = {}
    for i, (w, b) in zip((0, 2, 4, 5), layers):
        sd[f'{i}.weight'] = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
        sd[f'{i}.bias'] = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    m.load_state_dict(sd)
    return m


def u0_admitted(version):
    """U0_S<v> state dicts, its admitted epoch counts and the file hashes (label-free files of the replication's Stage A)."""
    import torch
    d = STAGE_A_ROOT / f'STAGE_A_{int(version)}'
    payload = torch.load(d / 'U0_MODEL.pt', map_location='cpu', weights_only=True)
    sa = json.loads((d / 'STAGE_A_SEED.json').read_text())
    if int(sa['seed']) != int(version):
        raise PermissionError('Stage A seed differs from the version')
    return payload, {t: int(sa['U0'][t]['best_epoch']) for t in TASKS}, {str(d / 'U0_MODEL.pt'): C.sha(d / 'U0_MODEL.pt'),
                                                                         str(d / 'STAGE_A_SEED.json'): C.sha(d / 'STAGE_A_SEED.json')}


def neural_raw_admitted(version, names=None):
    """NEURAL_RAW_S<v> = rho 1 E2 weights ({task: layers}) and the file's SHA-256 (the START of both bracket paths)."""
    import torch
    p = STAGE_A_ROOT / NEURAL_RAW_CKPT.format(v=int(version), e=ORIGIN_EPOCH)
    ref = torch.load(p, map_location='cpu', weights_only=True)
    if names is not None and tuple(ref['feature_names']) != tuple(names):
        raise PermissionError('NEURAL_RAW feature columns differ from the admitted encoding')
    return {t: layers_of(ref[t]) for t in TASKS}, {str(p): C.sha(p)}


def same_layers(a, b):
    return all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def bracket(path_L, cand_L, start_L, unit, grid):
    """Bracket rule of the CTRL and ALT paths (coordinator adjudication, 2026-10-24), on MODEL_FIT out-of-fold total NLL with origin
    START: LO = the grid point with the largest spend over START that is <= CAND's spend over START; HI = the grid point with the
    smallest spend >= CAND's + 2 SE, SE = the household-clustered SE of that point's paired difference against CAND (ties: the first in
    grid order). No point reaches it: HI = the largest-spend point, flag CONTROL_PATH_MAY_NOT_BRACKET. No point <= CAND's: LO is the
    origin (not saved again), flag LO_IS_ORIGIN. CAND's spend over START <= 0: the origin is the bracket, no point is saved, flag
    CTRL_BRACKET_IS_ORIGIN."""
    dc = cand_L['tot_nll'] - start_L['tot_nll']
    c, c_se = float(dc.mean()), FK.cluster_se(dc, unit)
    rows = {}
    for g in grid:
        d = path_L[g]['tot_nll'] - start_L['tot_nll']
        diff = path_L[g]['tot_nll'] - cand_L['tot_nll']
        rows[g] = {'spend_over_start': float(d.mean()), 'se_spend_over_start': FK.cluster_se(d, unit),
                   'diff_vs_cand': float(diff.mean()), 'se_diff_vs_cand': FK.cluster_se(diff, unit)}
    out = {'origin': 'START', 'selection_data': 'MODEL_FIT_OOF', 'cand_spend_over_start': {'value': c, 'se': c_se},
           'grid': {str(g): r for g, r in rows.items()}, 'lo': None, 'hi': None, 'flags': []}
    if c <= 0:
        out['flags'].append('CTRL_BRACKET_IS_ORIGIN')
        return out
    below = [g for g in grid if rows[g]['spend_over_start'] <= c]
    lo = max(below, key=lambda g: rows[g]['spend_over_start']) if below else None
    above = [g for g in grid if rows[g]['spend_over_start'] >= c + 2.0 * rows[g]['se_diff_vs_cand']]
    if above:
        hi = min(above, key=lambda g: rows[g]['spend_over_start'])
    else:
        hi = max(grid, key=lambda g: rows[g]['spend_over_start'])
        out['flags'].append('CONTROL_PATH_MAY_NOT_BRACKET')
    if lo is None:
        out['flags'].append('LO_IS_ORIGIN')
    elif lo == hi:
        lo = None
        out['flags'].append('LO_EQUALS_HI')
    sp = [r['spend_over_start'] for r in rows.values()]
    out['fineness'] = {'below': any(0.75 * c <= s <= c for s in sp), 'above': any(c <= s <= 1.25 * c for s in sp)}
    out['lo'], out['hi'] = lo, hi
    return out


def u0_recipe(X, y, task, version, rows, epochs):
    """The U0 recipe (Recipe A refit from scratch with the admitted epoch count and Stage A's seed) on `rows` of X (positions)."""
    import torch
    from acs_replication import neural, trees
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    seed = trees.seed_for('ACSR1neural', int(version), task) % 2 ** 31
    Xf = np.ascontiguousarray(X, dtype=np.float32); yy = np.ascontiguousarray(y, dtype=np.int64)
    torch.manual_seed(seed)
    m = network(task, Xf.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for e in range(1, int(epochs) + 1):
        neural._epoch(m, opt, Xf, yy, np.asarray(rows), seed + 30000 + e)
    m.eval()
    return m


# ---------------------------------------------------------------- the PFD loop (and its lambda = 0 continuation)
def _batch(order, b, B):
    """Task rows of step b: order[b B:(b + 1) B] (NEURAL_RAW's slicing; 40 x 256 < 16,000 on every real fold); wraps around only
    when a pool is smaller than one epoch of batches (invented-data checks)."""
    if (b + 1) * B <= len(order):
        return order[b * B:(b + 1) * B]
    return order[np.arange(b * B, (b + 1) * B) % len(order)]


class PFDTrainer:
    def __init__(self, seed, models, X, y_inc, y_occ, sex):
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        self.seed = int(seed); self.models = models
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = [torch.from_numpy(np.asarray(y_inc, dtype=np.int64)), torch.from_numpy(np.asarray(y_occ, dtype=np.int64))]
        self.sex = None if sex is None else torch.from_numpy(np.asarray(sex, dtype=np.int64))
        self.n = len(y_inc)
        for m in models:
            for p in m.parameters():
                p.requires_grad_(True)

    def params(self, i):
        return list(self.models[i].parameters())

    def logits(self, i, rows):
        z = self.models[i](self.X[rows]).double()
        return z - z.mean(1, keepdim=True)

    def ce(self, i, rows):
        import torch.nn.functional as F
        return F.cross_entropy(self.logits(i, rows), self.y[i][rows])

    def privacy_batch(self, rng):
        import torch
        rows = torch.from_numpy(np.asarray(rng.sample(range(self.n), min(RECIPE['privacy_batch'], self.n)), dtype=np.int64))
        return rows, self.sex[rows]

    def privacy(self, rows, s):
        terms, d2 = privacy_terms(self.logits(0, rows), self.logits(1, rows), s)
        return objective(terms), terms, d2

    def calibrate(self, rho=RECIPE['rho']):
        import torch
        from acs_replication import leafcorr as lc
        rng = random.Random(lc.seed_for('SAR1pfd-calibration', self.seed))
        task_rng = np.random.default_rng(lc.seed_for('SAR1pfd-calibration-task', self.seed) % 2 ** 32)
        draws = []
        allp = self.params(0) + self.params(1)
        for _ in range(RECIPE['calibration_draws']):
            tr = torch.from_numpy(task_rng.choice(self.n, RECIPE['task_batch'], replace=False))
            pr, ps = self.privacy_batch(rng)
            gt = []
            for i in range(2):
                gt += list(torch.autograd.grad(self.ce(i, tr), self.params(i)))
            P, terms, _ = self.privacy(pr, ps)
            gp = torch.autograd.grad(P, allp)
            draws.append({'task': float(torch.sqrt(sum((g.double() ** 2).sum() for g in gt))),
                          'privacy': float(torch.sqrt(sum((g.double() ** 2).sum() for g in gp))),
                          'terms': {v: float(t.detach()) for v, t in terms.items()}})
        med = {k: statistics.median(d[k] for d in draws) for k in ('task', 'privacy')}
        lam = float(rho) * med['task'] / med['privacy']
        if not (math.isfinite(lam) and 0 < lam < 1e6):
            raise FloatingPointError('PFD calibration coefficient outside the registered range')
        return {'draws': draws, 'medians': med, 'rho': float(rho), 'lambda': lam}

    def snapshot(self):
        return {t: layers_of(m.state_dict()) for t, m in zip(TASKS, self.models)}


def run_pfd(trainer, lam, epochs, on_ck, log_every=40):
    """lam > 0: PFD; lam == 0: the zero-protection continuation (identical task rows; no SEX read by the update)."""
    import torch
    from acs_replication import leafcorr as lc
    opts = [torch.optim.Adam(trainer.params(i), lr=RECIPE['learning_rate'], betas=RECIPE['betas'], eps=RECIPE['eps'], weight_decay=0.0)
            for i in range(2)]
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('SAR1pfd-privacy', trainer.seed))
    allp = trainer.params(0) + trainer.params(1)
    n0 = len(trainer.params(0))
    logs, step = [], 0
    B = RECIPE['task_batch']
    for epoch in range(epochs + 1):
        on_ck(epoch, trainer.snapshot())
        if epoch == epochs:
            break
        order = task_rng.permutation(trainer.n)
        for b in range(RECIPE['steps_per_epoch']):
            rows = torch.from_numpy(_batch(order, b, B))
            info = {}
            if lam > 0:
                pr, ps = trainer.privacy_batch(priv_rng)
                P, terms, d2 = trainer.privacy(pr, ps)
                gp = torch.autograd.grad(P, allp)
                pgs = [list(gp[:n0]), list(gp[n0:])]
                info = {'P': float(P.detach()), 'terms': {v: float(t.detach()) for v, t in terms.items()},
                        'd2': {v: float(t.detach()) for v, t in d2.items()}}
            else:
                pgs = [[torch.zeros_like(p) for p in trainer.params(i)] for i in range(2)]
            gts = [torch.autograd.grad(trainer.ce(i, rows), trainer.params(i)) for i in range(2)]
            if step % log_every == 0:
                nrm = lambda gs: float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs)))
                logs.append({'step': step, **info, 'task_grad_norm': [nrm(gts[i]) for i in range(2)],
                             'weighted_privacy_grad_norm': [lam * nrm(pgs[i]) for i in range(2)]})
            for i in range(2):
                opts[i].zero_grad(set_to_none=True)
                for p, a, c in zip(trainer.params(i), gts[i], pgs[i]):
                    p.grad = a + lam * c
                torch.nn.utils.clip_grad_norm_(trainer.params(i), RECIPE['clip'])
                opts[i].step()
            step += 1
    return {'logs': logs, 'steps': step, 'lambda': float(lam)}


# ---------------------------------------------------------------- NEURAL_RAW's own recipe with per-epoch checkpoints (the control)
def rawj_trainer(seed, models, X, y_inc, y_occ, sex, row_id):
    from acs_replication import data as D
    from acs_replication import neural_raw as nr
    rec = [nr.NeuralRecipient(t, m, X) for t, m in zip(TASKS, models)]
    probe = D.order_rows(row_id, 'ACSR1-probe', min(512, len(row_id)))
    reference = D.order_rows(row_id, 'ACSR1-reference', min(256, len(row_id)))
    return nr.NeuralTrainer(int(seed), rec, [np.asarray(y_inc, dtype=np.int64), np.asarray(y_occ, dtype=np.int64)],
                            np.arange(len(row_id)), np.asarray(sex, dtype=np.int64), probe, reference)


def run_rawj(trainer, lam, epochs, on_ck):
    """acs_replication.neural_raw.run_arm('RAW_J') step for step (same RNG streams, refresh, critic rounds, update and clip), with a
    checkpoint at every epoch 0..epochs instead of {0, 2, 5, 10, 20}; on all MODEL_FIT rows its E2 is NEURAL_RAW rho 1 E2."""
    import torch
    from acs_replication import leafcorr as lc
    from acs_replication import neural_raw as nr
    critics = trainer.warm(); warm_sha = critics.state_sha()
    opts = [torch.optim.Adam(r.params(), lr=nr.RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) for r in trainer.rec]
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('THP1privacy', trainer.seed)); critic_rng = random.Random(lc.seed_for('THP1critic-rounds', trainer.seed))
    B, step = nr.RECIPE['batch_size'], 0
    for epoch in range(epochs + 1):
        trainer.refresh(critics)
        on_ck(epoch, {t: layers_of(r.model.state_dict()) for t, r in zip(TASKS, trainer.rec)})
        if epoch == epochs:
            break
        order = task_rng.permutation(trainer.n)
        for b in range(nr.RECIPE['steps_per_epoch']):
            task_rows = torch.from_numpy(_batch(order, b, B))
            pr, ps = trainer.privacy_batch(priv_rng)
            w = trainer.wires(pr); terms = critics.terms(w[0], w[1], ps); P = lc.objective(terms, 'joint')
            pgs = []
            for i in range(2):
                g = torch.autograd.grad(P, trainer.rec[i].params(), retain_graph=True, allow_unused=True)
                pgs.append([torch.zeros_like(p) if x is None else x for x, p in zip(g, trainer.rec[i].params())])
            grads = [(torch.autograd.grad(trainer.ce(i, task_rows), trainer.rec[i].params()), pgs[i]) for i in range(2)]
            for i, r in enumerate(trainer.rec):
                gt, gp = grads[i]
                opts[i].zero_grad(set_to_none=True)
                for p, a, c in zip(r.params(), gt, gp):
                    p.grad = a + lam[i] * c
                torch.nn.utils.clip_grad_norm_(r.params(), nr.RECIPE['clip'])
                opts[i].step()
            for _ in range(lc.RECIPE['critic_rounds_per_recipient_update'] * 2):
                trainer.critic_round(critics, critic_rng)
            step += 1
    return {'steps': step, 'lambda': [float(x) for x in lam], 'warm_critic_sha256': warm_sha, 'critic_rounds': critics.rounds}


# ---------------------------------------------------------------- one training unit (a fold or the deployed fit)
def train_paths(X, y_inc, y_occ, sex, row_id, version, starts, epochs=RECIPE['epochs'], arms=('PFD', 'TASK', 'RAWJ'), log=None,
                rho=RECIPE['rho']):
    """Train the requested arms from the given start state dicts; return {arm: {epoch: weights}} and the run records."""
    import copy
    paths, recs = {}, {}
    for arm in arms:
        models = []
        for t in TASKS:
            m = network(t, X.shape[1]); m.load_state_dict(copy.deepcopy(starts[t])); m.train(); models.append(m)
        ck = {}
        t0 = time.process_time()
        if arm in ('PFD', 'TASK'):
            tr = PFDTrainer(version, models, X, y_inc, y_occ, sex if arm == 'PFD' else None)
            if arm == 'PFD':
                cal = tr.calibrate(rho)
                res = run_pfd(tr, cal['lambda'], epochs, lambda e, s: ck.__setitem__(e, s))
                res['calibration'] = {k: v for k, v in cal.items() if k != 'draws'}
            else:
                res = run_pfd(tr, 0.0, epochs, lambda e, s: ck.__setitem__(e, s))
        else:
            t_cal = rawj_trainer(version, [network(t, X.shape[1]) for t in TASKS], X, y_inc, y_occ, sex, row_id)
            for m, t in zip(t_cal.rec, TASKS):
                m.model.load_state_dict(copy.deepcopy(starts[t]))
            critics = t_cal.warm(); calib = t_cal.calibrate(critics)
            lam = calib['coefficients']['1.0']['lambda_J']
            tr = rawj_trainer(version, models, X, y_inc, y_occ, sex, row_id)
            res = run_rawj(tr, lam, epochs, lambda e, s: ck.__setitem__(e, s))
            res['calibration_medians'] = calib['medians']
        res['cpu_s'] = time.process_time() - t0
        paths[arm], recs[arm] = ck, res
        if log:
            log(f'{arm}: {res["steps"]} steps, lambda {res["lambda"]}, {res["cpu_s"]:.1f} CPU-s')
    return paths, recs


def start_states_recipe(X, y_inc, y_occ, version, rows, epochs_adm):
    return {t: u0_recipe(X, y, t, version, rows, epochs_adm[t]).state_dict() for t, y in zip(TASKS, (y_inc, y_occ))}


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, epochs=RECIPE['epochs'], budget=None, u0=None,
              neural_raw=None, check_neural_raw=True, reader_check=True, rho=RECIPE['rho']):
    """One version: cross-fitting of the three trained arms, the CAND rule, the CTRL (DQ) and ALT (RAWJ) brackets over START, the
    deployed fits and the saved configurations (at most 6 per version). `u0` / `neural_raw` replace the admitted files only in the
    invented-data check (neural_raw = (None, {}) there: the DQ base is then the deployed RAWJ E2)."""
    from .. import ctlkit as CK
    from .. import famrun as FR
    t0 = time.process_time()
    X = np.asarray(data['X'], dtype=np.float32)
    names = tuple(map(str, data['feature_names']))
    yI, yO, sex = (np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex'))
    unit, row_id = np.asarray(data['unit']), np.asarray(data['row_id'])
    budget = budget or BUDGET
    strengths = tuple(range(1, epochs + 1))
    alt_grid = tuple(range(ORIGIN_EPOCH + 1, epochs + 1))
    payload, epochs_adm, pins = u0 if u0 is not None else u0_admitted(version)
    if tuple(payload['feature_names']) != names:
        raise PermissionError('U0 feature columns differ from the admitted encoding')
    nr_weights, nr_pins = neural_raw if neural_raw is not None else neural_raw_admitted(version, names)
    pins = {**pins, **nr_pins}
    starts_full = {t: payload[t] for t in TASKS}
    # 1. cross-fitting: every fold restarts from the U0 recipe on its training rows; three arms on identical task rows
    f = FR.folds(unit, FAMILY, version)
    n = len(unit)
    oof = {(a, e): (np.zeros((n, 2)), np.zeros((n, 5))) for a in ('PFD', 'TASK', 'RAWJ') for e in range(epochs + 1)}
    fold_recs = []
    for j in range(FR.N_XF):
        tr, te = np.flatnonzero(f != j), np.flatnonzero(f == j)
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both parts of a cross-fitting fold')
        st = start_states_recipe(X, yI, yO, version, tr, epochs_adm)
        paths, recs = train_paths(X[tr], yI[tr], yO[tr], sex[tr], row_id[tr], version, st, epochs, log=lambda s: log(f'fold {j} {s}'),
                                  rho=rho)
        for a, ck in paths.items():
            for e, W in ck.items():
                pi, po = release_weights(W, X[te])
                oof[(a, e)][0][te], oof[(a, e)][1][te] = pi, po
        fold_recs.append({a: {k: v for k, v in r.items() if k != 'logs'} for a, r in recs.items()})
        log(json.dumps({'fold': j, 'cpu_s': round(time.process_time() - t0, 1)}))
    # 2. CAND: paired spends over the fold start (E0 of every arm is the same start) and the predeclared rule
    L = {k: FK.person_losses(*v, yI, yO) for k, v in oof.items()}
    base = L[('PFD', 0)]
    sp = {k: FK.spend(L[k], base, unit, sex) for k in L}
    k_star = FK.select_strength({e: sp[('PFD', e)] for e in strengths}, strengths, budget)
    sel = {'rule': 'k* = the largest checkpoint E_k (k = 1..10) such that E_1..E_k all keep estimate + 2 household-clustered SE <= '
                   'BUDGET on every component of the MODEL_FIT out-of-fold paired spend over the fold start (folds SAR_xf_FB_S<v>)',
           'budget': budget, 'k_star': k_star,
           'path': {f'{a}_E{e}': FK.jsonable_spend(sp[(a, e)]) for a in ('PFD', 'TASK', 'RAWJ') for e in range(epochs + 1)}}
    cand = {'config': 'FB_PFD', 'strength': k_star, 'checkpoint': None if k_star is None else f'E{k_star}',
            'status': 'NO_FEASIBLE_STRENGTH' if k_star is None else 'FITTED', 'spend_over_start': None}
    out = {'version': int(version), 'recipe': RECIPE, 'pins_sha256': pins, 'epochs_admitted': epochs_adm, 'selection': sel,
           'cand': cand, 'cont': {'config': 'FB_PFD_TASK', 'strength': k_star, 'gating': True}, 'paths': {},
           'fold_records': fold_recs}
    if k_star is None:
        log(json.dumps({'stop': 'E1 fails the budget: no candidate, continuation or control point is released'}))
        out['cont']['config'] = None
        out.update({'stop': 'NO_FEASIBLE_STRENGTH', 'saved': {}, 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fb') / f'FIT_PFD_V{int(version)}.json', out)
        return out
    # 3. CTRL (DQ of NEURAL_RAW) and ALT (RAWJ beyond E2): brackets over START = the cross-fitted RAWJ E2 (MODEL_FIT out of fold)
    start_oof, L_start, L_cand = oof[('RAWJ', ORIGIN_EPOCH)], L[('RAWJ', ORIGIN_EPOCH)], L[('PFD', k_star)]
    dc = L_cand['tot_nll'] - L_start['tot_nll']
    cand['spend_over_start'] = {'value': float(dc.mean()), 'se': FK.cluster_se(dc, unit), 'data': 'MODEL_FIT_OOF',
                                'start': f'cross-fitted RAWJ E{ORIGIN_EPOCH}'}
    Z = FK.compact(X, names)
    tag = DQ_TAG.format(v=int(version))
    basis = CK.dq_fit(*start_oof)
    dq_grid = tuple(CK.DQ_GRID_FINE)
    dq_oof = {g: CK.dq_release(basis, *start_oof, Z, g, tag) for g in dq_grid}
    L_dq = {g: FK.person_losses(*dq_oof[g], yI, yO) for g in dq_grid}
    br = {'CTRL': bracket(L_dq, L_cand, L_start, unit, dq_grid),
          'ALT': bracket({e: L[('RAWJ', e)] for e in alt_grid}, L_cand, L_start, unit, alt_grid)}
    ids = {'CTRL': 'FB_PFD_DQ', 'ALT': 'FB_PFD_RAWJ'}
    for key, b in br.items():
        out['paths'][key] = {
            'operation': ('SEX-blind subtractive-dithered quantization of NEURAL_RAW (sar.ctlkit.dq_fit / dq_release, DQ_GRID_FINE, tag '
                          f'{tag}); basis on the cross-fitted RAWJ-E2 outputs' if key == 'CTRL' else
                          f"NEURAL_RAW's own recipe continued beyond E{ORIGIN_EPOCH} (RAW_J, rho 1), grid E{alt_grid[0] if alt_grid else '-'}"
                          f"..E{epochs}"),
            'origin': f'START = NEURAL_RAW (out of fold: cross-fitted RAWJ E{ORIGIN_EPOCH}; deployed: NEURAL_RAW rho 1 E{ORIGIN_EPOCH})',
            **b, 'lo_config': None if b['lo'] is None else f'{ids[key]}_LO', 'hi_config': None if b['hi'] is None else f'{ids[key]}_HI',
            'lo_spend_over_start': None if b['lo'] is None else b['grid'][str(b['lo'])]['spend_over_start'],
            'hi_spend_over_start': None if b['hi'] is None else b['grid'][str(b['hi'])]['spend_over_start']}
        log(json.dumps({key: {'lo': b['lo'], 'hi': b['hi'], 'flags': b['flags']}}))
    sel['dq_basis_fitted_on'] = f'cross-fitted RAWJ E{ORIGIN_EPOCH} MODEL_FIT outputs (SEX never read)'
    # attribution diagnostic (descriptive): income share of the NLL spend over the fold start and over START
    def _share(Lx):
        a, b = FK.spend(Lx, base, unit), FK.spend(Lx, L_start, unit)
        return {'over_fold_start': a['inc_nll'][0] / max(a['tot_nll'][0], 1e-12), 'over_start': b['inc_nll'][0] / max(b['tot_nll'][0], 1e-12)}
    sel['income_share_of_nll_spend'] = {'CAND': _share(L_cand)}
    for key, Lp in (('CTRL', L_dq), ('ALT', {e: L[('RAWJ', e)] for e in alt_grid})):
        for side in ('lo', 'hi'):
            g = br[key][side]
            if g is not None:
                sel['income_share_of_nll_spend'][f'{key}_{side.upper()}'] = _share(Lp[g])
    log(json.dumps({'k_star': k_star, 'cand_spend_over_start': cand['spend_over_start'], 'cpu_s': round(time.process_time() - t0, 1)}))
    # 4. the deployed fit on all MODEL_FIT people from U0 itself
    full, full_recs = train_paths(X, yI, yO, sex, row_id, version, starts_full, epochs, log=lambda s: log(f'full {s}'), rho=rho)
    checks = {}
    if check_neural_raw:
        checks['rawj_E2_equals_NEURAL_RAW_rho1_E2'] = all(same_layers(full['RAWJ'][ORIGIN_EPOCH][t], nr_weights[t]) for t in TASKS)
        rec0 = start_states_recipe(X, yI, yO, version, np.arange(n), epochs_adm)
        checks['u0_recipe_reproduces_U0_bitwise'] = all(same_layers(layers_of(rec0[t]), layers_of(payload[t])) for t in TASKS)
    dq_base, dq_base_name = (nr_weights, 'NEURAL_RAW__rho1.0__E2.pt (deployed NEURAL_RAW)') if nr_weights is not None else \
        (full['RAWJ'][ORIGIN_EPOCH], f'deployed RAWJ E{ORIGIN_EPOCH} (invented-data check only)')
    # 5. saved configurations (CAND, CONT, and the bracket points that exist)
    roles = roles or FK.role_features()
    people, hh = n, len(np.unique(unit))
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the trained networks (SEX only through '
                'the privacy objectives: PFD and RAWJ; never for TASK or DQ), the CAND rule (incl. the within-SEX guard components) and '
                'the bracket selections (total NLL only); through U0 and NEURAL_RAW (inherited) every start and the DQ base; no '
                'CALIBRATION or MODEL_SELECT label')
    plan = [('FB_PFD', {'op': 'net', 'arm': 'PFD', 'checkpoint': f'E{k_star}', 'weights': full['PFD'][k_star]}, k_star,
             L_cand, oof[('PFD', k_star)], (1, FR.N_XF, 0, 0)),
            ('FB_PFD_TASK', {'op': 'net', 'arm': 'TASK', 'checkpoint': f'E{k_star}', 'weights': full['TASK'][k_star]}, k_star,
             L[('TASK', k_star)], oof[('TASK', k_star)], (1, FR.N_XF, 0, 0))]
    for side in ('lo', 'hi'):
        g = br['CTRL'][side]
        if g is not None:
            plan.append((f'FB_PFD_DQ_{side.upper()}', {'op': 'dq', 'arm': 'DQ', 'weights': dq_base, 'dq_base': dq_base_name,
                                                       'dq_basis': basis, 'delta': float(g), 'dq_tag': tag}, float(g), L_dq[g], dq_oof[g],
                         (0, 0, 0, 0)))
        e = br['ALT'][side]
        if e is not None:
            plan.append((f'FB_PFD_RAWJ_{side.upper()}', {'op': 'net', 'arm': 'RAWJ', 'checkpoint': f'E{e}', 'weights': full['RAWJ'][e]},
                         e, L[('RAWJ', e)], oof[('RAWJ', e)], (1, FR.N_XF, 0, FR.N_XF + 1)))
    saved, views = {}, {'FOLD_START': oof[('PFD', 0)], 'START': start_oof}
    for cfg, pl, strength, Lx, oof_pair, fits in plan:
        meta = FK.write_model(FAMILY, cfg, version, {'feature_names': names, 'recipe': RECIPE, 'pins_sha256': pins, **pl})
        arrays = FK.release_all(predict, cfg, version, roles)
        d = Lx['tot_nll'] - L_start['tot_nll']
        key = CONFIGS[cfg]['path']
        extra = {'path': key, 'arm': pl['arm'], 'checkpoint': pl.get('checkpoint'), 'strength': strength, 'selection_k_star': k_star,
                 'model_sha256': meta['model_sha256'], 'pins_sha256': pins, 'checks': checks,
                 'oof_spend_over_fold_start': FK.jsonable_spend(FK.spend(Lx, base, unit, sex)),
                 'oof_spend_over_start': {'value': float(d.mean()), 'se': FK.cluster_se(d, unit)},
                 'bracket_flags': br[key]['flags'] if key in br else None,
                 'n_start_refits': 2 * FR.N_XF + 2 if pl['arm'] != 'DQ' else 0}
        if pl['arm'] in full_recs:
            extra['lambda'] = full_recs[pl['arm']]['lambda']
        if pl['arm'] == 'DQ':
            extra.update({'delta': float(pl['delta']), 'dq_tag': tag, 'dq_base': dq_base_name})
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure, people, hh,
                        time.process_time() - t0, extra)
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = extra
        views[cfg] = oof_pair
    if reader_check:
        t1 = time.process_time()
        out['mf_reader_check'] = {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input',
                                  'auc': FK.mf_reader_check(views, sex, unit, f'SAR_mfcheck_FB_S{int(version)}'), 'cpu_s': time.process_time() - t1}
    out.update({'saved': saved, 'n_saved': len(saved), 'checks': checks,
                'full_records': {a: {k: v for k, v in r.items() if k != 'logs'} for a, r in full_recs.items()},
                'full_logs_PFD': full_recs['PFD']['logs'], 'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fb') / f'FIT_PFD_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks (invented data; no label file)
def _invented(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    sex = rng.integers(0, 2, n)
    s = 0.8 * (sex - 0.5)
    yI = ((X[:, 0] + 0.6 * X[:, 1] + s + rng.normal(0, 1, n)) > 0).astype(np.int64)
    yO = np.clip(np.floor(2.5 + X[:, 2] + 0.8 * X[:, 3] - 1.2 * s + rng.normal(0, 1, n)), 0, 4).astype(np.int64)
    X[:, 4] += 1.0 * s                                     # a SEX-predictive input
    return X, yI, yO, sex


def selftest():
    import torch
    torch.set_num_threads(1)
    ok, info = {}, {}
    rng = np.random.default_rng(7)
    # 1. deployment forward = torch forward (float32 network) to float32 rounding, deterministic, valid vectors
    d = 145
    m = network('occupation', d)
    X = rng.normal(size=(500, d)).astype(np.float32)
    W = layers_of(m.state_dict())
    p_np = np_forward(W, X)
    with torch.no_grad():
        z = m(torch.from_numpy(X)).double(); p_t = torch.softmax(z - z.mean(1, keepdim=True), 1).numpy()
    ok['numpy_forward_matches_torch'] = float(np.abs(p_np - p_t).max()) < 1e-6
    ok['numpy_forward_deterministic'] = bool(np.array_equal(p_np, np_forward(W, X)))
    ok['valid_vectors'] = bool(np.allclose(p_np.sum(1), 1, atol=1e-12, rtol=0) and (p_np >= 0).all())
    m2 = model_from_layers('occupation', W)
    ok['layers_roundtrip'] = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(W, layers_of(m2.state_dict())))
    # 2. the closed-form statistic: Gaussian classes with known D; Phi(D / sqrt 2) = AUC of the Fisher score
    nn_ = 40000
    sx = torch.from_numpy(rng.integers(0, 2, nn_))
    Dtrue = 0.6
    u = torch.from_numpy(rng.normal(size=(nn_, 3))) + (sx[:, None].double() * torch.tensor([Dtrue, 0.0, 0.0], dtype=torch.float64))
    d2 = fisher_d2(u, sx, ridge=0.0)
    ok['fisher_d2_estimates_truth'] = abs(float(d2) - Dtrue ** 2) < 0.02
    score = u[:, 0].numpy()
    emp_auc = FK.auc(score, sx.numpy())
    ok['auc_proxy_matches_empirical_auc'] = abs(float(auc_proxy(d2)) + 0.5 - emp_auc) < 0.01
    info['fisher'] = {'d2': float(d2), 'proxy_auc': float(auc_proxy(d2)) + 0.5, 'empirical_auc': emp_auc}
    # 3. affine invariance (temperature / bias moves cannot lower D) on the polynomial features
    zi = torch.from_numpy(rng.normal(size=(4000, 2))); zo = torch.from_numpy(rng.normal(size=(4000, 5)))
    s4 = torch.from_numpy(rng.integers(0, 2, 4000))
    zo = zo + 0.5 * s4[:, None].double() * torch.tensor([0, 1.0, -0.5, 0.3, 0.0], dtype=torch.float64)
    t1, d2a = privacy_terms(zi, zo, s4)
    t2, d2b = privacy_terms(2.7 * zi + 1.3, 0.4 * zo - 2.0, s4)
    ok['affine_invariant'] = all(abs(float(d2a[v]) - float(d2b[v])) < 1e-3 * max(float(d2a[v]), 1e-3) for v in d2a)
    ok['pair_bounds_views'] = all(float(d2a['pair']) >= float(d2a[v]) * (1 - 1e-3) for v in ('income', 'occupation'))
    # 4. the attribution property: income logit difference = noisy copy of an occupation coordinate (no incremental SEX signal):
    #    the whitened pair gradient on the income coordinate is ~0, the marginal (diagonal) discriminant's is not
    n5 = 20000
    s5 = torch.from_numpy(rng.integers(0, 2, n5))
    lo = torch.from_numpy(rng.normal(size=(n5, 4))) + 0.5 * s5[:, None].double() * torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    li = (lo[:, :1] + 0.3 * torch.from_numpy(rng.normal(size=(n5, 1)))).clone().requires_grad_(True)
    lo = lo.clone().requires_grad_(True)
    D2 = fisher_d2(torch.cat([li, lo], 1), s5)
    g_i, g_o = torch.autograd.grad(D2, [li, lo])
    a = torch.cat([li, lo], 1).detach()
    mdiff = a[s5.bool()].mean(0) - a[~s5.bool()].mean(0)
    w_marg = mdiff / a.var(0)                                  # diagonal normalization (the online critics' first-order weight)
    info['attribution'] = {'whitened_income_grad_share': float(g_i.abs().sum() / (g_i.abs().sum() + g_o.abs().sum())),
                           'marginal_income_weight_share': float(w_marg[0].abs() / w_marg.abs().sum())}
    ok['whitened_attribution_partial'] = info['attribution']['whitened_income_grad_share'] < 0.1 < info['attribution']['marginal_income_weight_share']
    # 5. a short end-to-end run of the three arms on invented data (shapes, checkpoints, lambda, protection direction)
    Xs, yI, yO, sx_ = _invented(3000, 12, 11)
    rid = np.arange(3000) * 7 + 3
    torch.manual_seed(0)
    st = {t: network(t, 12).state_dict() for t in TASKS}
    t0 = time.process_time()
    paths, recs = train_paths(Xs, yI, yO, sx_, rid, 0, st, epochs=3, log=None)
    info['e2e_cpu_s'] = round(time.process_time() - t0, 2)
    ok['three_arms_checkpoints'] = all(sorted(paths[a]) == [0, 1, 2, 3] for a in ('PFD', 'TASK', 'RAWJ'))
    ok['common_start'] = all(np.array_equal(paths['PFD'][0][t][0][0], paths[a][0][t][0][0]) for a in ('TASK', 'RAWJ') for t in TASKS)
    ok['pfd_lambda_positive'] = recs['PFD']['lambda'] > 0 and recs['TASK']['lambda'] == 0.0
    ok['rawj_two_lambdas'] = len(recs['RAWJ']['lambda']) == 2
    pf = release_weights(paths['PFD'][3], Xs); tk = release_weights(paths['TASK'][3], Xs)
    ok['arms_differ'] = not np.array_equal(pf[1], tk[1])
    tk2, _ = train_paths(Xs, yI, yO, 1 - sx_, rid, 0, st, epochs=1, arms=('TASK',))
    p_a = release_weights(paths['TASK'][1], Xs); p_b = release_weights(tk2['TASK'][1], Xs)
    ok['task_arm_ignores_sex'] = bool(np.array_equal(p_a[0], p_b[0]) and np.array_equal(p_a[1], p_b[1]))

    def pairD(P):
        zi_ = torch.from_numpy(np.log(P[0])); zo_ = torch.from_numpy(np.log(P[1]))
        return float(privacy_terms(zi_, zo_, torch.from_numpy(sx_))[1]['pair'])
    info['invented_pair_d2'] = {'TASK_E3': pairD(tk), 'PFD_E3': pairD(pf)}
    ok['pfd_lowers_invented_pair_separation'] = info['invented_pair_d2']['PFD_E3'] < info['invented_pair_d2']['TASK_E3']
    # 6. rules on invented spends: the CAND selection stops at the first failure; the adjudicated bracket rule and its flags
    z = {k: (0.0, 0.0) for k in BUDGET}
    big = dict(z, occ_nll=(0.01, 0.0))
    ok['rule_stops_at_first_failure'] = FK.select_strength({1: z, 2: big, 3: z}, (1, 2, 3), BUDGET) == 1
    nb = 4000
    ub = np.arange(nb) // 2
    noise = np.random.default_rng(2).normal(0, 0.05, nb)

    def Lmk(v):
        return {'tot_nll': np.full(nb, 1.0) + v + noise * 0.1}
    start_L, cand_L = Lmk(0.0), Lmk(0.004)
    grid = (1, 2, 3, 4, 5)
    pathL = {g: Lmk(0.0015 * g) for g in grid}              # spends 0.0015 .. 0.0075 over START
    b1 = bracket(pathL, cand_L, start_L, ub, grid)
    ok['bracket_lo_hi'] = b1['lo'] == 2 and b1['hi'] == 3 and b1['flags'] == []
    b2 = bracket({g: Lmk(0.0002 * g) for g in grid}, cand_L, start_L, ub, grid)
    ok['bracket_may_not_bracket'] = b2['hi'] == 5 and 'CONTROL_PATH_MAY_NOT_BRACKET' in b2['flags'] and b2['lo'] is None
    b3 = bracket({g: Lmk(0.005 + 0.001 * g) for g in grid}, cand_L, start_L, ub, grid)
    ok['bracket_lo_is_origin'] = b3['lo'] is None and b3['hi'] == 1 and 'LO_IS_ORIGIN' in b3['flags']
    b4 = bracket(pathL, Lmk(-0.001), start_L, ub, grid)
    ok['bracket_is_origin'] = b4['lo'] is None and b4['hi'] is None and b4['flags'] == ['CTRL_BRACKET_IS_ORIGIN']
    ok['bracket_record_fields'] = all(set(r) >= {'spend_over_start', 'se_diff_vs_cand'} for r in b1['grid'].values()) and \
        b1['origin'] == 'START' and b1['selection_data'] == 'MODEL_FIT_OOF'
    # 7. the CTRL deployment law: predict on a saved 'dq' payload = sar.ctlkit.dq_release of the base network's vectors, bitwise
    import tempfile
    from .. import ctlkit as CK
    F = C.features('MODEL_FIT')
    Xr = F['X'][:400]
    torch.manual_seed(3)
    Wn = {t: layers_of(network(t, Xr.shape[1]).state_dict()) for t in TASKS}
    bI, bO = release_weights(Wn, np.asarray(Xr, dtype=np.float64))
    basis = CK.dq_fit(bI, bO)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fb_pfd_dq_selftest_'))
    try:
        C.RUNS = tmp
        FK.write_model(FAMILY, 'FB_PFD_DQ_HI', 0, {'feature_names': tuple(F['feature_names']), 'op': 'dq', 'weights': Wn,
                                                   'dq_basis': basis, 'delta': 0.5, 'dq_tag': 'TEST'})
        FK.write_model(FAMILY, 'FB_PFD', 0, {'feature_names': tuple(F['feature_names']), 'op': 'net', 'weights': Wn})
        got = predict('FB_PFD_DQ_HI', 0, Xr, F['feature_names'])
        net = predict('FB_PFD', 0, Xr, F['feature_names'])
    finally:
        C.RUNS = old
    want = CK.dq_release(basis, bI, bO, FK.compact(np.asarray(Xr, dtype=np.float64), F['feature_names']), 0.5, 'TEST')
    ok['dq_predict_is_shared_law_bitwise'] = bool(np.array_equal(got['income'], want[0]) and np.array_equal(got['occupation'], want[1]))
    ok['net_predict_is_forward_bitwise'] = bool(np.array_equal(net['income'], bI) and np.array_equal(net['occupation'], bO))
    ok['dq_changes_release'] = not np.array_equal(got['occupation'], bO)
    # 8. CPU per step at the real input width (privacy batch 2,048)
    Xb, yIb, yOb, sxb = _invented(6000, 145, 3)
    mods = [network(t, 145) for t in TASKS]
    trb = PFDTrainer(0, mods, Xb, yIb, yOb, sxb)
    t0 = time.process_time()
    run_pfd(trb, 1.0, 1, lambda e, s: None)
    info['pfd_cpu_s_per_step_d145'] = round((time.process_time() - t0) / RECIPE['steps_per_epoch'], 4)
    ok['all_ok'] = all(ok.values())
    return {**ok, 'info': info}


def selftest_stage(n=3000, seed=5, epochs=4, rho=20.0):
    """stage_fit end to end on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read): U0 and
    NEURAL_RAW are replaced by invented start weights (the DQ base is then the deployed RAWJ E2), models go to a temporary directory,
    nothing is saved to the campaign's runs. rho = 20 makes CAND spend more than START, so the bracket branch is exercised."""
    import tempfile
    import torch
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    X = F['X'][idx].astype(np.float32)
    sex = rng.integers(0, 2, n)
    s = sex - 0.5
    data = {'X': X, 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': ((X[:, 1] + 0.5 * s + rng.normal(0, 1, n)) > 0.3).astype(np.int64),
            'occupation': np.clip(np.floor(2 + X[:, 0] - 1.0 * s + rng.normal(0, 1, n)), 0, 4).astype(np.int64), 'sex': sex}
    torch.manual_seed(1)
    payload = {'feature_names': list(F['feature_names']), **{t: network(t, X.shape[1]).state_dict() for t in TASKS}}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fb_pfd_selftest_'))
    try:
        C.RUNS = tmp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s_: None, record_dir=False, epochs=epochs,
                        budget={k: 100.0 for k in BUDGET}, u0=(payload, {'income': 1, 'occupation': 1}, {'invented-u0': 'none'}),
                        neural_raw=(None, {'invented-neural-raw': 'none'}), check_neural_raw=False, reader_check=True, rho=rho)
        re = {c: predict(c, 0, data['X'], data['feature_names']) for c in saved}
    finally:
        C.RUNS = old
    c_sp = out['cand']['spend_over_start']['value']
    expect = {'FB_PFD', 'FB_PFD_TASK'}
    for key, pre in (('CTRL', 'FB_PFD_DQ'), ('ALT', 'FB_PFD_RAWJ')):
        for side in ('lo', 'hi'):
            if out['paths'][key][side] is not None:
                expect.add(f'{pre}_{side.upper()}')
    paths_ok = all(set(out['paths'][k]) >= {'origin', 'selection_data', 'grid', 'lo_config', 'hi_config', 'lo_spend_over_start',
                                             'hi_spend_over_start', 'flags'} and out['paths'][k]['selection_data'] == 'MODEL_FIT_OOF'
                   and all(set(r) >= {'spend_over_start', 'se_diff_vs_cand'} for r in out['paths'][k]['grid'].values())
                   for k in ('CTRL', 'ALT'))
    rule_ok = True
    for k in ('CTRL', 'ALT'):
        P_ = out['paths'][k]
        g = {kk: v['spend_over_start'] for kk, v in P_['grid'].items()}
        if c_sp <= 0:
            rule_ok &= P_['flags'] == ['CTRL_BRACKET_IS_ORIGIN'] and P_['lo'] is None and P_['hi'] is None
            continue
        if P_['lo'] is not None:
            rule_ok &= g[str(P_['lo'])] <= c_sp and g[str(P_['lo'])] == max(v for v in g.values() if v <= c_sp)
        if 'CONTROL_PATH_MAY_NOT_BRACKET' not in P_['flags']:
            rule_ok &= g[str(P_['hi'])] >= c_sp + 2 * P_['grid'][str(P_['hi'])]['se_diff_vs_cand']
    ok = {'saved_set_matches_brackets': set(saved) == expect,
          'cand_and_cont_saved': {'FB_PFD', 'FB_PFD_TASK'} <= set(saved),
          'bracket_branch_exercised': c_sp > 0 and any(c.startswith('FB_PFD_DQ_') for c in saved),
          'k_star_is_last_checkpoint': out['cand']['strength'] == epochs and out['cand']['status'] == 'FITTED',
          'fit_record_fields': paths_ok and out['cont']['config'] == 'FB_PFD_TASK' and set(out['cand']['spend_over_start']) >= {'value', 'se'},
          'bracket_rule_holds': bool(rule_ok),
          'records_valid': all(r['kind'] == CONFIGS[c]['kind'] and r['path'] == CONFIGS[c]['path'] for c, (_, r) in saved.items()),
          'release_is_predict_bitwise': all(np.array_equal(saved[c][0]['MODEL_FIT'][t], re[c][t]) for c in saved for t in TASKS),
          'valid_vectors': all(np.allclose(a['MODEL_FIT'][t].sum(1), 1, atol=1e-9) for a, _ in saved.values() for t in TASKS),
          'reader_check_ran': 'mf_reader_check' in out,
          'at_most_six_per_version': len(saved) <= 6}
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['cand_spend_over_start'] = out['cand']['spend_over_start']
    ok['brackets'] = {k: {'lo': out['paths'][k]['lo'], 'hi': out['paths'][k]['hi'], 'flags': out['paths'][k]['flags']} for k in ('CTRL', 'ALT')}
    ok['cpu_s'] = round(out['cpu_s'], 1)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fb.pfd')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd in ('selftest', 'selftest_stage'):
        r = {'selftest': selftest, 'selftest_stage': selftest_stage}[a.cmd]()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'k_star': out['selection']['k_star'], 'status': out['cand']['status'],
                      'cand_spend_over_start': out['cand']['spend_over_start'],
                      'brackets': {k: {'lo': v['lo'], 'hi': v['hi'], 'flags': v['flags']} for k, v in out['paths'].items()},
                      'saved': sorted(out['saved']), 'checks': out.get('checks'), 'cpu_s': round(out['cpu_s'], 1)}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

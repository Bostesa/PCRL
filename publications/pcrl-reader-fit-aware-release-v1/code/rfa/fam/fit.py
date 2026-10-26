"""Family FAM: the reader-fit-aware release fit stage (card results/pcrl_reader_fit_aware_release_v1/cards/RF_<name>_v1.md). PCRL
campaign `pcrl_reader_fit_aware_release_v1`; EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year
PUMS Florida person records; not confirmation.

Adapted from SAR1's `sar.fb.pfo` (card FB_PFO_R1): the same START (NEURAL_RAW: the U0 networks continued with RAW-J, rho 1, 80 steps =
E2), the same occupation-only loop (income frozen bitwise at START's released income: FB_PFD_v1's recorded diagnosis, income absorbed
the objective and bound first; NEURAL_RAW's pair recovery equals its occupation view), optimizer (Adam 1e-4, clip 5), task-row stream
(NEURAL_RAW's), 10-step checkpoints, the occupation budget rule over the fold start, the RAW-J ALT path and the bracket rule. The
closed-form Fisher objective is replaced by the episode objective of `rfa.fam.rfit` in three modes that share every episode,
reader family, loss, lambda and schedule:
    FULL     derivative of the held-out reader surrogate through the released E outputs AND through the reader fit on A (CAND)
    STOPFIT  the same, with the fitted reader parameters detached from their fitting path (direct path only)
    ALTCRIT  an ordinary alternating persistent critic of the same family and loss (lagging; direct path only)
    TASK     lambda = 0 (zero-protection continuation; no episode, no SEX)
Fold k: the U0 recipe refitted on the fold's training rows, its RAW-J path (fold START = its step 80) and the four arms from fold START.
Deployed: the RAW-J path from U0 (checked bitwise against NEURAL_RAW at step 80) and the arms from the pinned NEURAL_RAW weights.

Commands (real data only through the campaign queue after the receipt and the committed card):
    <repository>/.venv/bin/python -m rfa.fam.fit selftest_stage
    <repository>/.venv/bin/python -m rfa.fam.fit fit --version V --card results/pcrl_reader_fit_aware_release_v1/cards/<card>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK
from . import net as NET

FAMILY = NET.FAMILY
TASKS = NET.TASKS
REVISION = 'v1'
EVERY = 10
TOTAL_STEPS = 400                                  # arm steps from START (10 epochs of 40 task batches)
ORIGIN_STEP = 80                                   # START = the RAW-J path after 80 steps (E2)
RAWJ_TOTAL = ORIGIN_STEP + TOTAL_STEPS             # the ALT path spans the same 400 steps beyond START
STEPS = tuple(range(EVERY, TOTAL_STEPS + 1, EVERY))
ALT_STEPS = tuple(range(ORIGIN_STEP + EVERY, RAWJ_TOTAL + 1, EVERY))
ARMS = ('FULL', 'STOPFIT', 'ALTCRIT', 'TASK')
# occupation components of SAR1 FB_PFD_v1's budgets (as used unchanged by FB_PFO_R1), over the fold start (the U0 recipe):
# estimate + 2 household-clustered SE <= limit on the MODEL_FIT out-of-fold paired spend
BUDGET_OCC = {'occ_nll': 0.0060, 'occ_brier': 0.0025, 'occ_brier_men': 0.0028, 'occ_brier_women': 0.0028, 'occ_acc': 0.0060}
RECIPE = {
    'schema': 'RFA1-RF-fit-recipe-v1', 'revision': REVISION,
    'start': 'deployed: NEURAL_RAW_S<v> (rho 1 E2) weights; fold k: the U0 recipe refitted on the fold training rows, then RAW-J 80 '
             'steps (sar.fb.pfo.rawj_path)',
    'income': "START's income network, frozen bitwise (a constant input of the pair view)",
    'occupation': "START's occupation network, trained with CE_occ + lambda S (S from rfa.fam.rfit.EpisodeObjective)",
    'learning_rate': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'clip': 5.0, 'task_batch': 256, 'steps_per_epoch': 40,
    'total_steps': TOTAL_STEPS, 'checkpoint_every': EVERY, 'rawj_total_steps': RAWJ_TOTAL,
    'rho': 1.0, 'calibration_draws': 16,
    'lambda': 'calibrated once per training unit on the FULL objective at START (rho x median ||grad CE_occ|| / median ||grad S||, '
              '16 fixed draws) and reused unchanged by STOPFIT and ALTCRIT',
    'task_rows': 'NEURAL_RAW task stream (np.random.default_rng(seed_for("THP1task", v)) epoch permutations)',
    'episodes': 'household-separated A / E row sets per step from np.random.default_rng(seed_for("RFA1-episodes", v)); identical in '
                'FULL, STOPFIT and ALTCRIT',
}
CONFIGS = {
    'RF_FULL': {'kind': 'candidate', 'path': 'CAND',
                'what': 'occupation network trained against the held-out reader surrogate with the derivative through the reader fit '
                        '(FULL), income frozen at START, at the MODEL_FIT-selected checkpoint S_k*'},
    'RF_TASK': {'kind': 'twin', 'path': 'CONT',
                'what': 'zero-protection continuation (gating): the identical occupation-only loop with lambda = 0 at S_k*'},
    'RF_STOP_LO': {'kind': 'control', 'path': 'STOPFIT',
                   'what': 'stopped-fit control lower bracket point: identical episodes, readers, loss, lambda and schedule with the fitted '
                           "reader parameters detached from their fitting path, at the largest out-of-fold spend over START <= CAND's"},
    'RF_STOP_HI': {'kind': 'control', 'path': 'STOPFIT',
                   'what': 'stopped-fit control upper bracket point at the smallest out-of-fold spend over START >= CAND\'s + 2 SE'},
    'RF_ALTC_HI': {'kind': 'control', 'path': 'ALTCRIT',
                   'what': 'ordinary alternating persistent critic of the same family and loss (lagging; direct path), upper bracket '
                           "point at the smallest out-of-fold spend over START >= CAND's + 2 SE"},
    'RF_RAWJ_HI': {'kind': 'control', 'path': 'ALT',
                   'what': "NEURAL_RAW's own RAW-J recipe continued beyond E2 (both networks), upper bracket point at the smallest "
                           "out-of-fold spend over START >= CAND's + 2 SE"},
}
PURPOSE = ('RF v1: train the occupation network with the reader-fit-aware episode objective (FULL), its stopped-fit and alternating-'
           'critic controls, its zero-protection continuation and the RAW-J ALT path on MODEL_FIT (SEX as a training label of the '
           'protection objectives and of the within-SEX guard components of the strength rule; the TASK arm never reads SEX)')


def _pd():
    from sar.fb import pfd as PD                     # admitted SAR1 training utilities (U0 recipe, networks, brackets), unchanged
    return PD


def _pfo():
    from sar.fb import pfo as PFO                    # admitted SAR1 fine RAW-J path, unchanged
    return PFO


def seed_for(*items):
    from acs_replication import leafcorr as lc
    return lc.seed_for(*items)


# ---------------------------------------------------------------- the occupation-only trainer
class OccTrainer:
    def __init__(self, seed, inc_layers, occ_layers, X, y_occ, sex, unit, recipe_rf=None):
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        PD = _pd()
        self.seed = int(seed)
        self.inc_layers = inc_layers
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        inc = PD.model_from_layers('income', inc_layers); inc.eval()
        with torch.no_grad():
            z = inc(self.X).double()
            self.z_inc = z - z.mean(1, keepdim=True)           # the released income logits (constant)
        self.occ = PD.model_from_layers('occupation', occ_layers); self.occ.train()
        for p in self.occ.parameters():
            p.requires_grad_(True)
        self.y = torch.from_numpy(np.asarray(y_occ, dtype=np.int64))
        self.sex = None if sex is None else torch.from_numpy(np.asarray(sex, dtype=np.int64))
        self.n = len(y_occ)
        _, self.hh_of_row = np.unique(np.asarray(unit), return_inverse=True)
        self.hh_size = np.bincount(self.hh_of_row)
        self.recipe_rf = recipe_rf
        self.rf_module = None                                   # the objective module (None = rfa.fam.rfit; a revision sets its own)

    def rf(self):
        from . import rfit
        return self.rf_module or rfit

    def params(self):
        return list(self.occ.parameters())

    def logits(self, rows):
        z = self.occ(self.X[rows]).double()
        return z - z.mean(1, keepdim=True)

    def ce(self, rows):
        import torch.nn.functional as F
        return F.cross_entropy(self.logits(rows), self.y[rows])

    def episode(self, rng, n_a, n_e):
        """Household-separated A / E rows: households in a random order; A = the first households reaching n_a rows, E = the next
        households reaching n_e rows (whole households; sizes slightly above the targets)."""
        import torch
        H = len(self.hh_size)
        if int(self.hh_size.sum()) < int(n_a) + int(n_e) + 2 * int(self.hh_size.max()):
            raise ValueError('the unit pool is too small for the registered episode sizes')
        order = rng.permutation(H)
        cum = np.cumsum(self.hh_size[order])
        ka = int(np.searchsorted(cum, n_a) + 1)
        ke = int(np.searchsorted(cum, cum[ka - 1] + n_e) + 1)
        lab = np.full(H, 2, dtype=np.int8)
        lab[order[:ka]] = 0
        lab[order[ka:ke]] = 1
        rl = lab[self.hh_of_row]
        return torch.from_numpy(np.flatnonzero(rl == 0)), torch.from_numpy(np.flatnonzero(rl == 1))

    def snapshot(self):
        return {'income': self.inc_layers, 'occupation': NET.layers_of(self.occ.state_dict())}


def _objective_call(obj, tr, A, E):
    """The episode objective on A (auxiliary training) and E (auxiliary evaluation) rows: income logits are START's constants,
    occupation logits carry the graph to the trainable network."""
    return obj.objective(tr.z_inc[A], tr.logits(A), tr.sex[A], tr.z_inc[E], tr.logits(E), tr.sex[E])


def calibrate(tr, rho, n_draws=16):
    """lambda = rho x median ||grad CE_occ|| / median ||grad S_FULL|| over fixed draws at the trainer's current weights."""
    import torch
    RF = tr.rf()
    rr = tr.recipe_rf or RF.RECIPE_RF
    ep_rng = np.random.default_rng(seed_for('RFA1-calibration-episodes', tr.seed) % 2 ** 32)
    task_rng = np.random.default_rng(seed_for('RFA1-calibration-task', tr.seed) % 2 ** 32)
    obj = RF.EpisodeObjective('FULL', tr.seed, recipe=rr)
    draws = []
    nrm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs)))
    for _ in range(n_draws):
        rows = torch.from_numpy(task_rng.choice(tr.n, RECIPE['task_batch'], replace=False))
        A, E = tr.episode(ep_rng, rr['n_A'], rr['n_E'])
        gt = torch.autograd.grad(tr.ce(rows), tr.params())
        S, info = _objective_call(obj, tr, A, E)
        gp = torch.autograd.grad(S, tr.params())
        draws.append({'task': nrm(gt), 'privacy': nrm(gp), 'S': float(S.detach())})
    med = {k: statistics.median(d[k] for d in draws) for k in ('task', 'privacy')}
    lam = float(rho) * med['task'] / med['privacy']
    if not (math.isfinite(lam) and 0 < lam < 1e8):
        raise FloatingPointError('RF calibration coefficient outside the registered range')
    return {'draws': draws, 'medians': med, 'rho': float(rho), 'lambda': lam}


def _flat(gs):
    import torch
    return torch.cat([g.reshape(-1).double() for g in gs])


def mechanism_stats(g_dir, g_fit, groups=8):
    """Systematic fit-path statistics from per-episode parameter-space gradients (lists of flat float64 tensors, same episodes):
    group means over `groups` equal consecutive groups; unbiased cross-group estimates (pairs of different groups) of |E g_dir|^2,
    |E g_fit|^2, E g_fit . E g_dir and |E g_tot|^2; systematic share |E g_fit| / |E g_dir|; orthogonal share (the part of E g_fit
    orthogonal to E g_dir, over |E g_dir|); cos(E g_tot, E g_dir). The challenger's F0 / F1 statistics (HANDBACK_1 section 4)."""
    import torch
    m = min(len(g_dir), len(g_fit))
    k = int(groups)
    if m < 2 * k:
        return {'n': m, 'groups': k, 'status': 'TOO_FEW_EPISODES'}
    per = m // k
    D = torch.stack([torch.stack(g_dir[i * per:(i + 1) * per]).mean(0) for i in range(k)])
    F = torch.stack([torch.stack(g_fit[i * per:(i + 1) * per]).mean(0) for i in range(k)])
    T = D + F

    def cross(X, Y):
        s = (X @ Y.T)
        return float((s.sum() - torch.diagonal(s).sum()) / (k * (k - 1)))
    dd, ff, fd, tt, td = cross(D, D), cross(F, F), cross(F, D), cross(T, T), cross(T, D)
    sys_share = math.sqrt(max(ff, 0.0)) / math.sqrt(dd) if dd > 0 else None
    orth2 = ff - (fd ** 2 / dd if dd > 0 else 0.0)
    orth = math.sqrt(max(orth2, 0.0)) / math.sqrt(dd) if dd > 0 else None
    cos_td = td / math.sqrt(tt * dd) if tt > 0 and dd > 0 else None
    return {'n': per * k, 'groups': k, 'dir_sq': dd, 'fit_sq': ff, 'fit_dot_dir': fd, 'tot_sq': tt, 'systematic_share': sys_share,
            'orthogonal_share': orth, 'cos_tot_dir': cos_td, 'parallel_coefficient': fd / dd if dd > 0 else None,
            'status': 'OK' if dd > 0 else 'DIRECT_PATH_NOT_ESTIMABLE'}


GATE = {'episodes': 64, 'groups': 8, 'orthogonal_below': 0.05, 'cos_above': 0.995,
        'rule': 'MECHANISM_INERT iff the orthogonal share < 0.05 and cos(E g_tot, E g_dir) > 0.995 (unbiased cross-group estimates over '
                '64 episodes in 8 groups at START, MODEL_FIT only, before any arm is fitted); otherwise MECHANISM_ACTIVE'}


def mechanism_gate(tr, n_episodes=GATE['episodes'], groups=GATE['groups']):
    """F0 at the trainer's current weights (START): g_dir and g_fit of the FULL objective on fixed gate episodes (no update)."""
    import torch
    RF = tr.rf()
    rr = tr.recipe_rf or RF.RECIPE_RF
    ep_rng = np.random.default_rng(seed_for('RFA1-gate-episodes', tr.seed) % 2 ** 32)
    obj = RF.EpisodeObjective('FULL', tr.seed, recipe=rr)
    gd, gf = [], []
    t0 = time.process_time()
    for _ in range(int(n_episodes)):
        A, E = tr.episode(ep_rng, rr['n_A'], rr['n_E'])
        S_dir, S_fit, info = obj.objective_parts(tr.z_inc[A], tr.logits(A), tr.sex[A], tr.z_inc[E], tr.logits(E), tr.sex[E])
        gd.append(_flat(torch.autograd.grad(S_dir, tr.params(), retain_graph=True)))
        gf.append(_flat(torch.autograd.grad(S_fit, tr.params(), allow_unused=True, materialize_grads=True)))
    st = mechanism_stats(gd, gf, groups)
    inert = (st.get('orthogonal_share') is not None and st['orthogonal_share'] < GATE['orthogonal_below']
             and st.get('cos_tot_dir') is not None and st['cos_tot_dir'] > GATE['cos_above'])
    return {**st, 'label': 'MECHANISM_INERT' if inert else 'MECHANISM_ACTIVE', 'gate': GATE, 'cpu_s': time.process_time() - t0}


def run_arm(tr, mode, lam, total_steps, every, on_ck, log_every=10, stop_at=None, monitor_every=40):
    """One arm from the trainer's current weights. mode in ARMS; TASK = lambda 0, no episode, no SEX. `stop_at` ends the loop early
    at that step (the trajectory up to it is identical: deterministic, checkpoint prefixes). FULL takes its direct and fit-path parts
    separately (the update is their sum) and logs the systematic statistics per 40-step epoch (F1); every arm with episodes logs the
    held-out degree-2 Fisher monitor and the A-row SD of the logit differences every `monitor_every` steps (diagnostics only)."""
    import torch
    RF = tr.rf()
    rr = tr.recipe_rf or RF.RECIPE_RF
    opt = torch.optim.Adam(tr.params(), lr=RECIPE['learning_rate'], betas=RECIPE['betas'], eps=RECIPE['eps'], weight_decay=0.0)
    task_rng = np.random.default_rng(seed_for('THP1task', tr.seed) % 2 ** 32)
    ep_rng = np.random.default_rng(seed_for('RFA1-episodes', tr.seed) % 2 ** 32)
    obj = None if mode == 'TASK' else RF.EpisodeObjective(mode, tr.seed, recipe=rr)
    PD = _pd()
    S_, B = RECIPE['steps_per_epoch'], RECIPE['task_batch']
    end = total_steps if stop_at is None else min(int(stop_at), total_steps)
    logs, mech, monitors, step, n_episodes = [], [], [], 0, 0
    ep_dir, ep_fit = [], []
    nrm = lambda gs: float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs)))
    t0 = time.process_time()
    on_ck(0, tr.snapshot())
    while step < end:
        order = task_rng.permutation(tr.n)
        for b in range(S_):
            rows = torch.from_numpy(PD._batch(order, b, B))
            info = {}
            if mode != 'TASK':
                A, E = tr.episode(ep_rng, rr['n_A'], rr['n_E'])
                zA, zE = tr.logits(A), tr.logits(E)
                if mode == 'FULL':
                    S_dir, S_fit, info = obj.objective_parts(tr.z_inc[A], zA, tr.sex[A], tr.z_inc[E], zE, tr.sex[E])
                    S = S_dir + S_fit
                    if not torch.isfinite(S):
                        raise FloatingPointError(f'non-finite episode objective at step {step} ({mode})')
                    g_d = torch.autograd.grad(S_dir, tr.params(), retain_graph=True)
                    g_f = torch.autograd.grad(S_fit, tr.params(), allow_unused=True, materialize_grads=True)
                    gp = [a + c for a, c in zip(g_d, g_f)]
                    ep_dir.append(_flat(g_d)); ep_fit.append(_flat(g_f))
                else:
                    S, info = obj.objective(tr.z_inc[A], zA, tr.sex[A], tr.z_inc[E], zE, tr.sex[E])
                    if not torch.isfinite(S):
                        raise FloatingPointError(f'non-finite episode objective at step {step} ({mode})')
                    gp = torch.autograd.grad(S, tr.params())
                n_episodes += 1
                info = {'S': float(S.detach()), 'n_A': int(len(A)), 'n_E': int(len(E)), **info}
                if step % monitor_every == 0 and hasattr(RF, 'monitor_fisher2'):
                    with torch.no_grad():
                        monitors.append({'step': step, **RF.monitor_fisher2(tr.z_inc[A], zA.detach(), tr.sex[A], tr.z_inc[E],
                                                                             zE.detach(), tr.sex[E])})
            else:
                gp = [torch.zeros_like(p) for p in tr.params()]
            gt = torch.autograd.grad(tr.ce(rows), tr.params())
            if step % log_every == 0:
                logs.append({'step': step, **info, 'task_grad_norm': nrm(gt), 'weighted_privacy_grad_norm': lam * nrm(gp)})
            opt.zero_grad(set_to_none=True)
            for p, a, c in zip(tr.params(), gt, gp):
                p.grad = a + lam * c
            torch.nn.utils.clip_grad_norm_(tr.params(), RECIPE['clip'])
            opt.step()
            step += 1
            if mode == 'FULL' and step % S_ == 0:
                mech.append({'epoch_end_step': step, **mechanism_stats(ep_dir, ep_fit, 8)})
                ep_dir, ep_fit = [], []
            if step % every == 0:
                on_ck(step, tr.snapshot())
            if step >= end:
                break
    if mode == 'FULL' and ep_dir:
        mech.append({'epoch_end_step': step, 'partial_epoch': True, **mechanism_stats(ep_dir, ep_fit, 8)})
    return {'logs': logs, 'mechanism_by_epoch': mech, 'monitors': monitors, 'steps': step, 'lambda': float(lam), 'mode': mode,
            'n_episodes': n_episodes, 'cpu_s': time.process_time() - t0,
            'inner_reader_fits': None if obj is None else n_episodes * len(rr['views']) * len(rr['families']),
            'inner_fits_note': None if obj is None else ('one exact Newton fit per view and family per episode (FULL, STOPFIT); '
                                                         'ALTCRIT: its Adam rounds per view and family per episode plus one exact '
                                                         'diagnostic fit for the lag only')}


def arm_paths(X, yO, sex, unit, version, start_w, total_steps, every, rho, arms=ARMS, stop_at=None, log=None, recipe_rf=None,
              rf_module=None):
    """The requested arms from START's weights; lambda calibrated once on FULL and shared. {arm: {step: weights}}, records."""
    paths, recs = {}, {}
    cal_tr = OccTrainer(version, start_w['income'], start_w['occupation'], X, yO, sex, unit, recipe_rf)
    cal_tr.rf_module = rf_module
    cal = calibrate(cal_tr, rho) if any(a != 'TASK' for a in arms) else {'lambda': 0.0, 'medians': None}
    del cal_tr
    for arm in arms:
        tr = OccTrainer(version, start_w['income'], start_w['occupation'], X, yO, None if arm == 'TASK' else sex, unit, recipe_rf)
        tr.rf_module = rf_module
        ck = {}
        lam = 0.0 if arm == 'TASK' else cal['lambda']
        res = run_arm(tr, arm, lam, total_steps, every, lambda s, w: ck.__setitem__(s, w),
                      stop_at=None if stop_at is None else stop_at.get(arm))
        res['calibration'] = {k: v for k, v in cal.items() if k != 'draws'}
        paths[arm], recs[arm] = ck, res
        if log:
            log(f'{arm}: {res["steps"]} steps, lambda {res["lambda"]:.4g}, {res["cpu_s"]:.1f} CPU-s')
    return paths, recs


def _median_of(logs, key):
    vals = [l.get(key) for l in logs if isinstance(l.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def fit_path_share(logs):
    """Median over logged FULL steps of the fit-path share reported by the objective (None if the objective does not report it)."""
    vals = [((l.get('output_grad_norms') or {}).get('fit_share')) for l in logs]       # rfit info: output-space proxy (per step)
    vals = [x for x in vals if isinstance(x, (int, float))]
    return {'median': statistics.median(vals), 'n': len(vals), 'min': min(vals), 'max': max(vals)} if vals else None


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, total_steps=TOTAL_STEPS, every=EVERY, budget=None,
              u0=None, neural_raw=None, check_neural_raw=True, reader_check=True, rho=RECIPE['rho'], recipe_rf=None,
              configs=None, record_name=None, rf_module=None, alias_check=None, revision=None):
    """One version's fit unit. A revision passes its own objective module (`rf_module`), recipe, configuration ids (`configs[base]['id']`),
    record name and an `alias_check(base_cfg, arm, step, weights) -> existing config id | None` that reuses a bitwise-identical saved
    configuration instead of saving a new one."""
    PD, PFO = _pd(), _pfo()
    from .. import famrun as FR
    t0 = time.process_time()
    configs = configs or CONFIGS
    X = np.asarray(data['X'], dtype=np.float32)
    names = tuple(map(str, data['feature_names']))
    yI, yO, sex = (np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex'))
    unit, row_id = np.asarray(data['unit']), np.asarray(data['row_id'])
    budget = budget or BUDGET_OCC
    steps = tuple(range(every, total_steps + 1, every))
    origin = ORIGIN_STEP
    rawj_total = origin + total_steps
    alt_steps = tuple(range(origin + every, rawj_total + 1, every))
    payload, epochs_adm, pins = u0 if u0 is not None else PD.u0_admitted(version)
    if tuple(payload['feature_names']) != names:
        raise PermissionError('U0 feature columns differ from the admitted encoding')
    nr_weights, nr_pins = neural_raw if neural_raw is not None else PD.neural_raw_admitted(version, names)
    pins = {**pins, **nr_pins}
    n = len(unit)
    from . import rfit as RF0
    RF = rf_module or RF0
    rev = revision or REVISION
    rname = record_name or f'FIT_RF_V{int(version)}.json'
    # 0. F0, the mechanism gate at START (all MODEL_FIT rows; no update): an inert fit path stops the unit before any arm is fitted
    gate_w = nr_weights if nr_weights is not None else {t: PD.layers_of(payload[t]) for t in TASKS}
    gtr = OccTrainer(version, gate_w['income'], gate_w['occupation'], X, yO, sex, unit, recipe_rf)
    gtr.rf_module = rf_module
    gate = mechanism_gate(gtr)
    del gtr
    log(json.dumps({'gate': {k: gate[k] for k in ('label', 'systematic_share', 'orthogonal_share', 'cos_tot_dir', 'n')}}))
    if gate['label'] == 'MECHANISM_INERT':
        out = {'version': int(version), 'revision': rev, 'recipe': RECIPE, 'recipe_rf': recipe_rf or RF.RECIPE_RF,
               'pins_sha256': pins, 'mechanism_gate': gate, 'stop': 'MECHANISM_INERT', 'saved': {}, 'cpu_s': time.process_time() - t0}
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fam') / rname, out)
        return out
    # 1. cross-fitting: fold start = the U0 recipe; fold START = its RAW-J path at 80 steps; the four arms from fold START
    f = FR.folds(unit, FAMILY, version)
    z2 = lambda: (np.zeros((n, 2)), np.zeros((n, 5)))
    oof = {('RAWJ', s): z2() for s in (0,) + alt_steps + (origin,)}
    oof.update({(a, s): z2() for a in ARMS for s in (0,) + steps})
    fold_recs, income_bitwise, fps = [], True, []
    for j in range(FR.N_XF):
        tr, te = np.flatnonzero(f != j), np.flatnonzero(f == j)
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both parts of a cross-fitting fold')
        st = PD.start_states_recipe(X, yI, yO, version, tr, epochs_adm)
        rj, rj_rec = PFO.rawj_path(X[tr], yI[tr], yO[tr], sex[tr], row_id[tr], version, st, rawj_total, every,
                                   log=lambda s: log(f'fold {j} {s}'))
        start_w = rj[origin]
        ap, ap_recs = arm_paths(X[tr], yO[tr], sex[tr], unit[tr], version, start_w, total_steps, every, rho, rf_module=rf_module,
                                log=lambda s: log(f'fold {j} {s}'), recipe_rf=recipe_rf)
        for key, ck in [('RAWJ', rj)] + [(a, ap[a]) for a in ARMS]:
            for s, W in ck.items():
                if (key, s) in oof:
                    pi, po = NET.release_weights(W, X[te])
                    oof[(key, s)][0][te], oof[(key, s)][1][te] = pi, po
        s_inc = NET.release_weights(start_w, X[te])[0]
        income_bitwise &= all(np.array_equal(oof[(a, s)][0][te], s_inc) for a in ARMS for s in (0,) + steps)
        fps.append(fit_path_share(ap_recs['FULL']['logs']))
        fold_recs.append({'RAWJ': {k: v for k, v in rj_rec.items() if k != 'logs'},
                          **{a: {k: v for k, v in r.items() if k != 'logs'} for a, r in ap_recs.items()},
                          'FULL_logs_every_10': ap_recs['FULL']['logs'][::4]})
        log(json.dumps({'fold': j, 'cpu_s': round(time.process_time() - t0, 1)}))
    if not income_bitwise:
        raise AssertionError("arm income vectors differ from START's (the operation must not touch the income output)")
    # 2. CAND: occupation components of the out-of-fold spend over the fold start (the U0 recipe) and the predeclared rule
    L = {k: FK.person_losses(*v, yI, yO) for k, v in oof.items()}
    fold_start, L_start = L[('RAWJ', 0)], L[('RAWJ', origin)]
    sp = {k: FK.spend(L[k], fold_start, unit, sex) for k in L}
    k_star = FK.select_strength({s: sp[('FULL', s)] for s in steps}, steps, budget)
    sel = {'rule': f'k* = the largest checkpoint S_k (every {every} steps from START) such that S_{every}..S_k all keep estimate + 2 '
                   'household-clustered SE <= BUDGET_OCC on the five occupation components of the MODEL_FIT out-of-fold paired spend '
                   f'over the fold start (the U0 recipe; folds RFA_xf_{FAMILY}_S<v>); the income vector is START\'s',
           'budget': budget, 'k_star': k_star, 'income_vectors_equal_START_bitwise': bool(income_bitwise),
           'path': {f'{a}_S{s}': FK.jsonable_spend(sp[(a, s)]) for (a, s) in sorted(sp, key=lambda k: (k[0], k[1]))},
           'fit_path_share_by_fold': fps}
    cand = {'config': 'RF_FULL', 'strength': k_star, 'checkpoint': None if k_star is None else f'S{k_star}',
            'status': 'NO_FEASIBLE_STRENGTH' if k_star is None else 'FITTED', 'spend_over_start': None}
    out = {'version': int(version), 'revision': rev, 'recipe': RECIPE, 'pins_sha256': pins, 'epochs_admitted': epochs_adm,
           'selection': sel, 'cand': cand, 'cont': {'config': 'RF_TASK', 'strength': k_star, 'gating': True}, 'paths': {},
           'fold_records': fold_recs, 'start_spend_over_fold_start': FK.jsonable_spend(sp[('RAWJ', origin)]),
           'mechanism_gate': gate}
    out['recipe_rf'] = recipe_rf or RF.RECIPE_RF
    if k_star is None:
        log(json.dumps({'stop': f'S{every} fails the occupation budget: no candidate, continuation or control point is released'}))
        out['cont']['config'] = None
        out.update({'stop': 'NO_FEASIBLE_STRENGTH', 'saved': {}, 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fam') / rname, out)
        return out
    # 3. brackets over START (MODEL_FIT out of fold, total NLL): STOPFIT (LO + HI saved), ALTCRIT and ALT (HI saved)
    start_oof, L_cand = oof[('RAWJ', origin)], L[('FULL', k_star)]
    dc = L_cand['tot_nll'] - L_start['tot_nll']
    cand['spend_over_start'] = {'value': float(dc.mean()), 'se': FK.cluster_se(dc, unit), 'data': 'MODEL_FIT_OOF',
                                'start': f'cross-fitted RAWJ S{origin} (E2)'}
    br = {'STOPFIT': PD.bracket({s: L[('STOPFIT', s)] for s in steps}, L_cand, L_start, unit, steps),
          'ALTCRIT': PD.bracket({s: L[('ALTCRIT', s)] for s in steps}, L_cand, L_start, unit, steps),
          'ALT': PD.bracket({s: L[('RAWJ', s)] for s in alt_steps}, L_cand, L_start, unit, alt_steps)}
    for key in ('ALTCRIT', 'ALT'):
        if br[key]['lo'] is not None:
            br[key]['flags'] = br[key]['flags'] + ['LO_NOT_SAVED_SPEC_BUDGET']
    for key, b in br.items():
        out['paths'][key] = {**b, 'lo_config': None, 'hi_config': None,
                             'lo_spend_over_start': None if b['lo'] is None else b['grid'][str(b['lo'])]['spend_over_start'],
                             'hi_spend_over_start': None if b['hi'] is None else b['grid'][str(b['hi'])]['spend_over_start']}
        log(json.dumps({key: {'lo': b['lo'], 'hi': b['hi'], 'flags': b['flags']}}))
    same = {a: FK.jsonable_spend(FK.spend(L[(a, k_star)], L_cand, unit, sex)) for a in ('STOPFIT', 'ALTCRIT', 'TASK')}
    sel['same_checkpoint_spend_vs_cand'] = same              # residual cost differences at the identical schedule point S_k*
    log(json.dumps({'k_star': k_star, 'cand_spend_over_start': cand['spend_over_start'], 'cpu_s': round(time.process_time() - t0, 1)}))
    # 4. deployed fits: the RAW-J path from U0 (ALT, and the START check); the arms from the pinned START up to their saved steps
    starts_full = {t: payload[t] for t in TASKS}
    need_rj = max([origin] + [br['ALT']['hi']] if br['ALT']['hi'] is not None else [origin])
    full_rj, full_rj_rec = PFO.rawj_path(X, yI, yO, sex, row_id, version, starts_full, max(need_rj, origin), every,
                                         log=lambda s: log(f'full {s}'))
    start_full = nr_weights if nr_weights is not None else full_rj[origin]
    stop_at = {'FULL': k_star, 'TASK': k_star,
               'STOPFIT': max([s for s in (br['STOPFIT']['lo'], br['STOPFIT']['hi']) if s is not None] or [every]),
               'ALTCRIT': br['ALTCRIT']['hi'] or every}
    full_ap, full_ap_rec = arm_paths(X, yO, sex, unit, version, start_full, total_steps, every, rho, stop_at=stop_at, rf_module=rf_module,
                                     log=lambda s: log(f'full {s}'), recipe_rf=recipe_rf)
    checks = {'deployed_start': 'NEURAL_RAW__rho1.0__E2.pt' if nr_weights is not None else f'deployed RAWJ S{origin} (invented-data check only)'}
    if check_neural_raw:
        checks['rawj_S80_equals_NEURAL_RAW_rho1_E2'] = all(PD.same_layers(full_rj[origin][t], nr_weights[t]) for t in TASKS)
        rec0 = PD.start_states_recipe(X, yI, yO, version, np.arange(n), epochs_adm)
        checks['u0_recipe_reproduces_U0_bitwise'] = all(PD.same_layers(PD.layers_of(rec0[t]), PD.layers_of(payload[t])) for t in TASKS)
    checks['cand_income_layers_are_START'] = PD.same_layers(full_ap['FULL'][k_star]['income'], start_full['income'])
    # 5. saved configurations
    roles = roles or FK.role_features()
    people, hh = n, len(np.unique(unit))
    exposure = ('MODEL_FIT income, occupation and SEX labels of all MODEL_FIT people reach the trained networks (SEX only through the '
                'episode objectives FULL / STOPFIT / ALTCRIT and RAW-J; never for TASK), the CAND rule (occupation components, incl. the '
                'within-SEX guards) and the bracket selections (total NLL only); every inner episode row is MODEL_FIT training data; '
                'through U0 and NEURAL_RAW (inherited) every start and the frozen income output; no CALIBRATION or MODEL_SELECT label')
    plan = [('RF_FULL', 'FULL', k_star), ('RF_TASK', 'TASK', k_star)]
    for side in ('lo', 'hi'):
        if br['STOPFIT'][side] is not None:
            plan.append((f'RF_STOP_{side.upper()}', 'STOPFIT', br['STOPFIT'][side]))
    if br['ALTCRIT']['hi'] is not None:
        plan.append(('RF_ALTC_HI', 'ALTCRIT', br['ALTCRIT']['hi']))
    if br['ALT']['hi'] is not None:
        plan.append(('RF_RAWJ_HI', 'RAWJ', br['ALT']['hi']))
    saved, views = {}, {'FOLD_START': oof[('RAWJ', 0)], 'START': start_oof}
    out['aliases'] = {}
    for base_cfg, arm, s in plan:
        cfg = configs[base_cfg]['id'] if 'id' in configs[base_cfg] else base_cfg
        W = full_rj[s] if arm == 'RAWJ' else full_ap[arm][s]
        hit = alias_check(base_cfg, arm, s, W) if alias_check is not None else None
        if hit is not None:                              # a bitwise-identical saved configuration: reused, not saved again
            out['aliases'][cfg] = {'alias_of': hit, 'arm': arm, 'checkpoint': f'S{s}'}
            views[cfg] = oof[(arm, s)]
            log(json.dumps({'alias': {cfg: hit}}))
            continue
        pl = {'op': 'net', 'arm': arm, 'checkpoint': f'S{s}', 'weights': W}
        meta = FK.write_model(FAMILY, cfg, version, {'feature_names': names, 'recipe': RECIPE, 'recipe_rf': out['recipe_rf'],
                                                     'pins_sha256': pins, 'revision': rev, **pl})
        arrays = FK.release_all(NET.predict, cfg, version, roles)
        Lx = L[(arm, s)]
        d = Lx['tot_nll'] - L_start['tot_nll']
        key = configs[base_cfg]['path']
        extra = {'revision': rev, 'path': key, 'arm': arm, 'checkpoint': f'S{s}', 'strength': s, 'selection_k_star': k_star,
                 'model_sha256': meta['model_sha256'], 'pins_sha256': pins, 'checks': checks,
                 'oof_spend_over_fold_start': FK.jsonable_spend(FK.spend(Lx, fold_start, unit, sex)),
                 'oof_spend_over_start': {'value': float(d.mean()), 'se': FK.cluster_se(d, unit)},
                 'bracket_flags': br[key]['flags'] if key in br else None,
                 'lambda': full_rj_rec['lambda'] if arm == 'RAWJ' else full_ap_rec[arm]['lambda'],
                 'n_episodes_deployed_fit': None if arm == 'RAWJ' else full_ap_rec[arm]['n_episodes']}
        rec = FK.record(card, configs[base_cfg]['kind'], configs[base_cfg]['what'], (1, FR.N_XF, 0, 0 if arm == 'TASK' else FR.N_XF + 1),
                        'MODEL_FIT: income, occupation, sex', exposure, people, hh, time.process_time() - t0, extra)
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = extra
        views[cfg] = oof[(arm, s)]
    if reader_check:
        t1 = time.process_time()
        out['mf_reader_check'] = {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input',
                                  'auc': FK.mf_reader_check(views, sex, unit, f'RFA_mfcheck_RF_{rev}_S{int(version)}'),
                                  'cpu_s': time.process_time() - t1}
    out.update({'saved': saved, 'n_saved_new': len(saved), 'checks': checks,
                'full_records': {'RAWJ': {k: v for k, v in full_rj_rec.items() if k != 'logs'},
                                 **{a: {k: v for k, v in r.items() if k != 'logs'} for a, r in full_ap_rec.items()}},
                'full_logs_FULL': full_ap_rec['FULL']['logs'], 'full_logs_STOPFIT': full_ap_rec['STOPFIT']['logs'],
                'full_logs_ALTCRIT': full_ap_rec['ALTCRIT']['logs'],
                'altcrit_rel_lag_median_deployed': _median_of(full_ap_rec['ALTCRIT']['logs'], 'critic_rel_lag_median'),
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fam') / rname, out)
    return out


# ---------------------------------------------------------------- checks (invented labels; no label file)
def selftest_stage(n=3000, seed=5, total_steps=60, rho=1.0):
    """stage_fit end to end on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read): U0 is
    replaced by the U0 recipe on the invented labels and the deployed START by the deployed RAW-J replica; models go to a temporary
    directory; nothing is saved to the campaign's runs."""
    import tempfile
    from pathlib import Path
    PD = _pd()
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    X = F['X'][idx].astype(np.float32)
    sex = ((X[:, 1] + rng.normal(0, 0.7, n)) > np.median(X[:, 1])).astype(np.int64)
    s = sex - 0.5
    data = {'X': X, 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': ((X[:, 1] + 0.5 * s + rng.normal(0, 0.3, n)) > 0.0).astype(np.int64),
            'occupation': np.clip(np.floor(2 + 1.5 * X[:, 1] - 1.0 * s + rng.normal(0, 0.3, n)), 0, 4).astype(np.int64), 'sex': sex}
    adm = {'income': 15, 'occupation': 15}
    st = PD.start_states_recipe(X, data['income'], data['occupation'], 0, np.arange(n), adm)
    payload = {'feature_names': list(F['feature_names']), **st}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='rfa_fam_stage_'))
    t0 = time.process_time()
    try:
        C.RUNS = tmp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        from . import rfit as RF
        small = {**RF.RECIPE_RF, 'n_A': 768, 'n_E': 512}           # the invented pool (n = 3000) cannot hold registered episodes
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s_: None, record_dir=False, total_steps=total_steps,
                        every=EVERY, budget={k: 100.0 for k in BUDGET_OCC}, u0=(payload, adm, {'invented-u0': 'none'}),
                        neural_raw=(None, {'invented': 'none'}), check_neural_raw=False, reader_check=True, rho=rho, recipe_rf=small)
        re = {c: NET.predict(c, 0, data['X'], data['feature_names']) for c in saved}
    finally:
        C.RUNS = old
    ok = {'cand_and_cont_saved': {'RF_FULL', 'RF_TASK'} <= set(saved),
          'at_most_six_per_version': len(saved) <= 6,
          'k_star_is_last_checkpoint': out['cand']['strength'] == total_steps and out['cand']['status'] == 'FITTED',
          'income_bitwise_start': all(np.array_equal(saved[c][0]['MODEL_FIT']['income'], saved['RF_FULL'][0]['MODEL_FIT']['income'])
                                      for c in saved if c != 'RF_RAWJ_HI'),
          'records_valid': all(r['kind'] == CONFIGS[c]['kind'] and r['path'] == CONFIGS[c]['path'] for c, (_, r) in saved.items()),
          'release_is_predict_bitwise': all(np.array_equal(saved[c][0]['MODEL_FIT'][t], re[c][t]) for c in saved for t in TASKS),
          'valid_vectors': all(np.allclose(a['MODEL_FIT'][t].sum(1), 1, atol=1e-9) for a, _ in saved.values() for t in TASKS),
          'reader_check_ran': 'mf_reader_check' in out,
          'checks_income_layers': out['checks'].get('cand_income_layers_are_START') is True,
          'full_differs_from_task': not np.array_equal(saved['RF_FULL'][0]['MODEL_FIT']['occupation'],
                                                       saved['RF_TASK'][0]['MODEL_FIT']['occupation'])}
    if 'RF_STOP_HI' in saved:
        ok['full_differs_from_stopfit'] = not np.array_equal(saved['RF_FULL'][0]['MODEL_FIT']['occupation'],
                                                             saved['RF_STOP_HI'][0]['MODEL_FIT']['occupation'])
    ok['gate_recorded'] = out.get('mechanism_gate', {}).get('label') in ('MECHANISM_ACTIVE', 'MECHANISM_INERT')
    ok['mechanism_logged_along_full'] = bool(out['fold_records'][0]['FULL'].get('mechanism_by_epoch'))
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['gate'] = {k: out['mechanism_gate'].get(k) for k in ('label', 'systematic_share', 'orthogonal_share', 'cos_tot_dir')}
    ok['cand_spend_over_start'] = out['cand']['spend_over_start']
    ok['brackets'] = {k: {'lo': v['lo'], 'hi': v['hi'], 'flags': v['flags']} for k, v in out['paths'].items()}
    ok['fit_path_share_by_fold'] = out['selection']['fit_path_share_by_fold']
    ok['cpu_s'] = round(time.process_time() - t0, 1)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rfa.fam.fit')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest_stage':
        r = selftest_stage()
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

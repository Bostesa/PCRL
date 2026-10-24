"""FB revision R1: the coalition Fisher-discriminant objective carried by the OCCUPATION network alone, with the income output frozen at
START's released income (PFO; card FB_PFO_R1, draft by investigator B). PCRL campaign `pcrl_strong_anchor_release_v1`; EXPLORATORY
DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation. A revision
designed from the registered FB_PFD_v1 MODEL_FIT records only (no MODEL_SELECT outcome): label REVISED_.

Diagnosis (results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V<v>.json, MODEL_FIT only; roles/investigator_b/CARD_DRAFT_R1.md):
the binding condition of FB_PFD_v1 was income utility at its first checkpoints (v0: k* = E1, E2 fails income Brier and NLL; v1: E1
fails income Brier), while every occupation component stayed inside its budget; the income share of CAND's NLL spend was ~1.0 over
the fold start (0.40-0.53 for the controls), and in the logged batches the income-view D^2 fell 88% in the first 40 steps while the
pair's fell 5%: the AUC-scale view term (gradient ~ 1/D in D^2) and the one-coordinate income output made the income network absorb
the objective. The attribution hypothesis of FB_PFD_v1 is withdrawn.

Operation (one structural change, plus a finer path as part of it):
  income      START's income vector (NEURAL_RAW rho 1 E2, bitwise; in fold k the fold's cross-fitted RAW-J E2 income network): not
              trained, enters the pair term as released (a constant input of the objective)
  occupation  START's occupation network (NEURAL_RAW E2; fold k: the fold's RAW-J E2) trained with CE_occ + lambda P_O,
              P_O = [Phi(D_occ / sqrt 2) - 1/2] + [Phi(D_pair / sqrt 2) - 1/2] (FB_PFD_v1's closed-form whitened quadratic best
              response; the income-view term is dropped because the income view is constant), D_pair on (frozen income logit
              difference, trainable occupation logit differences); lambda = rho x median ||grad CE_occ|| / median ||grad P_O|| over
              16 fixed draws at START (occupation network only), rho = 1; Adam 1e-4, clip 5; NEURAL_RAW's task-row stream
  path        checkpoints every 10 steps, S10 .. S400 (the 40-step first checkpoint of FB_PFD_v1 was too coarse)
  rule        k* = the largest S_k such that S_10..S_k all keep estimate + 2 household-clustered SE <= BUDGET_OCC on the five
              occupation components of the MODEL_FIT out-of-fold spend over the fold start (FB_PFD_v1's budgets); the income
              components are not spent by the operation (CAND's income vector is START's, asserted bitwise) and are recorded only
Configurations (at most 8 new = 4 x 2 versions; ALT may alias an already saved FB_PFD_RAWJ point):
  FB_PFO            CAND at S_k*
  FB_PFO_TASK       CONT (gating): the identical loop with lambda = 0 at S_k* (income frozen, SEX not used)
  FB_PFO_DQ_HI      CTRL upper bracket point: SEX-blind DQ of NEURAL_RAW's OCCUPATION output (income passed through bitwise),
                    FB_PFD_v1's DQ basis and dither tag (its occupation coordinates equal FB_PFD_v1's CTRL path at the same step)
  FB_PFO_RAWJ_HI    ALT upper bracket point: NEURAL_RAW's own RAW-J recipe continued beyond E2, re-selected on the same path with
                    checkpoints every 10 steps (S90 .. S400); aliases FB_PFD_RAWJ_<side> when the selected checkpoint equals it bitwise
  The lower bracket points are computed and recorded but not saved (specification budget), flag LO_NOT_SAVED_SPEC_BUDGET.
Deployment: `predict` (dispatched from sar.fb for FB_PFO* configurations): float64 numpy forward pass; for the CTRL point the shared
DQ law on the occupation output with the income vector passed through bitwise (python3, no torch); training needs the torch venv.

Commands (real data only through the campaign queue after the receipt and the committed card):
    <repository>/.venv/bin/python -m sar.fb.pfo selftest | selftest_stage
    <repository>/.venv/bin/python -m sar.fb.pfo fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FB_PFO_R1.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK
from . import pfd as PD

FAMILY = PD.FAMILY
TASKS = PD.TASKS
REVISION = 'R1'
EVERY = 10
TOTAL_STEPS = 400
ORIGIN_STEP = PD.ORIGIN_EPOCH * 40                 # START = the RAW-J path after 80 steps (E2)
STEPS = tuple(range(EVERY, TOTAL_STEPS + 1, EVERY))
ALT_STEPS = tuple(s for s in STEPS if s > ORIGIN_STEP)
RECIPE = {
    'schema': 'SAR1-FB-PFO-R1-recipe-v1', 'revision': REVISION,
    'income': "START's income vector, frozen bitwise (deployed: NEURAL_RAW rho 1 E2; fold k: the fold's cross-fitted RAW-J E2)",
    'occupation_start': "START's occupation network (deployed: NEURAL_RAW rho 1 E2; fold k: the fold's cross-fitted RAW-J E2)",
    'objective': 'CE_occ + lambda [Phi(D_occ / sqrt 2) - 1/2 + Phi(D_pair / sqrt 2) - 1/2]; D_v^2 = Delta_v^T (S_v + eps_v I)^-1 Delta_v '
                 'on degree-2 polynomial features of the standardized logit differences (FB_PFD_v1 functions); the frozen income logit '
                 'difference enters D_pair as released',
    'learning_rate': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'clip': 5.0, 'task_batch': 256, 'steps_per_epoch': 40,
    'privacy_batch': 2048, 'ridge': PD.RECIPE['ridge'], 'rho': 1.0, 'calibration_draws': 16,
    'checkpoints': f'every {EVERY} steps, S0..S{TOTAL_STEPS}', 'task_rows': 'NEURAL_RAW task stream (seed_for("THP1task", v))',
    'privacy_rows': 'random.Random(seed_for("SAR1pfo-privacy", v))',
}
BUDGET_OCC = {k: PD.BUDGET[k] for k in ('occ_nll', 'occ_brier', 'occ_brier_men', 'occ_brier_women', 'occ_acc')}
CONFIGS = {
    'FB_PFO': {'kind': 'candidate', 'path': 'CAND',
               'what': "occupation network alone trained with the closed-form coalition (pair + occupation view) Fisher-discriminant "
                       "objective, income frozen at START's released income, at the MODEL_FIT-selected checkpoint S_k*"},
    'FB_PFO_TASK': {'kind': 'twin', 'path': 'CONT',
                    'what': "zero-protection continuation (gating): the identical occupation-only loop with lambda = 0 at S_k* (income "
                            "frozen at START's, SEX not used)"},
    'FB_PFO_DQ_HI': {'kind': 'control', 'path': 'CTRL',
                     'what': "CTRL upper bracket point: SEX-blind subtractive-dithered quantization of NEURAL_RAW's occupation output "
                             "(income passed through bitwise) at the smallest out-of-fold spend over START >= CAND's + 2 SE"},
    'FB_PFO_RAWJ_HI': {'kind': 'control', 'path': 'ALT',
                       'what': "ALT upper bracket point: NEURAL_RAW's own RAW-J recipe continued beyond E2 (checkpoints every 10 steps) "
                               "at the smallest out-of-fold spend over START >= CAND's + 2 SE"},
}
PURPOSE = ('FB revision R1 PFO: train the occupation network alone with the closed-form coalition Fisher-discriminant objective '
           "(income frozen at START's), its zero-protection continuation, the RAW-J ALT path and the occupation-restricted SEX-blind DQ "
           'control path on MODEL_FIT (SEX as a training label of the protection objectives and of the within-SEX guard components of '
           'the strength rule; the DQ path never reads SEX)')


# ---------------------------------------------------------------- deployment (numpy only)
def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function of the FB_PFO* configurations (sar.deploy contract via sar.fb); deterministic; numpy only."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    pi, po = PD.release_weights(P['weights'], X)
    op = P.get('op', 'net')
    if op == 'dq_occ':                                   # CTRL: the shared DQ law on the occupation output; income passed through
        from .. import ctlkit as CK
        po = CK.dq_release(P['dq_basis'], pi, po, FK.compact(X, names), float(P['delta']), P['dq_tag'])[1]
    elif op != 'net':
        raise ValueError(f'unknown FB_PFO payload op {op!r}')
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- the occupation-only objective and loop (torch)
def privacy_terms_occ(z_inc_const, z_occ, sex):
    """P_O terms: the occupation view and the pair with the income logit difference as a constant (released) input."""
    f = PD.view_features(z_inc_const, z_occ)
    d2 = {v: PD.fisher_d2(PD.poly2(f[v]), sex) for v in ('occupation', 'pair')}
    return {v: PD.auc_proxy(d2[v]) for v in d2}, d2


class PFOTrainer:
    def __init__(self, seed, inc_layers, occ_layers, X, y_occ, sex):
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        self.seed = int(seed)
        self.inc_layers = inc_layers
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        inc = PD.model_from_layers('income', inc_layers); inc.eval()
        with torch.no_grad():
            z = inc(self.X).double()
            self.z_inc = z - z.mean(1, keepdim=True)            # the released income logits (constant)
        self.occ = PD.model_from_layers('occupation', occ_layers); self.occ.train()
        for p in self.occ.parameters():
            p.requires_grad_(True)
        self.y = torch.from_numpy(np.asarray(y_occ, dtype=np.int64))
        self.sex = None if sex is None else torch.from_numpy(np.asarray(sex, dtype=np.int64))
        self.n = len(y_occ)

    def params(self):
        return list(self.occ.parameters())

    def logits(self, rows):
        z = self.occ(self.X[rows]).double()
        return z - z.mean(1, keepdim=True)

    def ce(self, rows):
        import torch.nn.functional as F
        return F.cross_entropy(self.logits(rows), self.y[rows])

    def privacy_batch(self, rng):
        import torch
        rows = torch.from_numpy(np.asarray(rng.sample(range(self.n), min(RECIPE['privacy_batch'], self.n)), dtype=np.int64))
        return rows, self.sex[rows]

    def privacy(self, rows, s):
        terms, d2 = privacy_terms_occ(self.z_inc[rows], self.logits(rows), s)
        return terms['occupation'] + terms['pair'], terms, d2

    def calibrate(self, rho=RECIPE['rho']):
        import torch
        from acs_replication import leafcorr as lc
        rng = random.Random(lc.seed_for('SAR1pfo-calibration', self.seed))
        task_rng = np.random.default_rng(lc.seed_for('SAR1pfo-calibration-task', self.seed) % 2 ** 32)
        draws = []
        for _ in range(RECIPE['calibration_draws']):
            tr = torch.from_numpy(task_rng.choice(self.n, RECIPE['task_batch'], replace=False))
            pr, ps = self.privacy_batch(rng)
            gt = torch.autograd.grad(self.ce(tr), self.params())
            P, terms, _ = self.privacy(pr, ps)
            gp = torch.autograd.grad(P, self.params())
            draws.append({'task': float(torch.sqrt(sum((g.double() ** 2).sum() for g in gt))),
                          'privacy': float(torch.sqrt(sum((g.double() ** 2).sum() for g in gp))),
                          'terms': {v: float(t.detach()) for v, t in terms.items()}})
        med = {k: statistics.median(d[k] for d in draws) for k in ('task', 'privacy')}
        lam = float(rho) * med['task'] / med['privacy']
        if not (math.isfinite(lam) and 0 < lam < 1e6):
            raise FloatingPointError('PFO calibration coefficient outside the registered range')
        return {'draws': draws, 'medians': med, 'rho': float(rho), 'lambda': lam}

    def snapshot(self):
        return {'income': self.inc_layers, 'occupation': PD.layers_of(self.occ.state_dict())}


def run_pfo(trainer, lam, total_steps, every, on_ck, log_every=40):
    """lam > 0: CAND path; lam == 0: CONT path (identical task rows; no SEX read by the update). Only the occupation network moves."""
    import torch
    from acs_replication import leafcorr as lc
    opt = torch.optim.Adam(trainer.params(), lr=RECIPE['learning_rate'], betas=RECIPE['betas'], eps=RECIPE['eps'], weight_decay=0.0)
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('SAR1pfo-privacy', trainer.seed))
    S, B = RECIPE['steps_per_epoch'], RECIPE['task_batch']
    logs, step = [], 0
    on_ck(0, trainer.snapshot())
    while step < total_steps:
        order = task_rng.permutation(trainer.n)
        for b in range(S):
            rows = torch.from_numpy(PD._batch(order, b, B))
            info = {}
            if lam > 0:
                pr, ps = trainer.privacy_batch(priv_rng)
                P, terms, d2 = trainer.privacy(pr, ps)
                gp = torch.autograd.grad(P, trainer.params())
                info = {'P': float(P.detach()), 'terms': {v: float(t.detach()) for v, t in terms.items()},
                        'd2': {v: float(t.detach()) for v, t in d2.items()}}
            else:
                gp = [torch.zeros_like(p) for p in trainer.params()]
            gt = torch.autograd.grad(trainer.ce(rows), trainer.params())
            if step % log_every == 0:
                nrm = lambda gs: float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs)))
                logs.append({'step': step, **info, 'task_grad_norm': nrm(gt), 'weighted_privacy_grad_norm': lam * nrm(gp)})
            opt.zero_grad(set_to_none=True)
            for p, a, c in zip(trainer.params(), gt, gp):
                p.grad = a + lam * c
            torch.nn.utils.clip_grad_norm_(trainer.params(), RECIPE['clip'])
            opt.step()
            step += 1
            if step % every == 0:
                on_ck(step, trainer.snapshot())
            if step >= total_steps:
                break
    return {'logs': logs, 'steps': step, 'lambda': float(lam)}


def run_rawj_fine(trainer, lam, total_steps, every, on_ck):
    """acs_replication.neural_raw.run_arm('RAW_J') step for step (= sar.fb.pfd.run_rawj: same RNG streams, epoch-start refresh, critic
    rounds, update and clip), with a weight snapshot every `every` steps; the snapshot at 80 steps is the E2 of run_rawj."""
    import torch
    from acs_replication import leafcorr as lc
    from acs_replication import neural_raw as nr
    critics = trainer.warm(); warm_sha = critics.state_sha()
    opts = [torch.optim.Adam(r.params(), lr=nr.RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) for r in trainer.rec]
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('THP1privacy', trainer.seed)); critic_rng = random.Random(lc.seed_for('THP1critic-rounds', trainer.seed))
    B, S, step = nr.RECIPE['batch_size'], nr.RECIPE['steps_per_epoch'], 0
    snap = lambda: {t: PD.layers_of(r.model.state_dict()) for t, r in zip(TASKS, trainer.rec)}
    on_ck(0, snap())
    n_epochs = math.ceil(total_steps / S)
    for epoch in range(n_epochs + 1):
        trainer.refresh(critics)
        if epoch == n_epochs:
            break
        order = task_rng.permutation(trainer.n)
        for b in range(S):
            task_rows = torch.from_numpy(PD._batch(order, b, B))
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
            if step % every == 0:
                on_ck(step, snap())
    return {'steps': step, 'lambda': [float(x) for x in lam], 'warm_critic_sha256': warm_sha, 'critic_rounds': critics.rounds}


def rawj_path(X, yI, yO, sex, row_id, version, starts, total_steps, every, log=None):
    """The RAW-J path (calibration on a separate trainer at the start, as the NEURAL_RAW worker does) with fine snapshots."""
    import copy
    t0 = time.process_time()
    t_cal = PD.rawj_trainer(version, [PD.network(t, X.shape[1]) for t in TASKS], X, yI, yO, sex, row_id)
    for m, t in zip(t_cal.rec, TASKS):
        m.model.load_state_dict(copy.deepcopy(starts[t]))
    critics = t_cal.warm(); calib = t_cal.calibrate(critics)
    lam = calib['coefficients']['1.0']['lambda_J']
    models = []
    for t in TASKS:
        m = PD.network(t, X.shape[1]); m.load_state_dict(copy.deepcopy(starts[t])); m.train(); models.append(m)
    tr = PD.rawj_trainer(version, models, X, yI, yO, sex, row_id)
    ck = {}
    res = run_rawj_fine(tr, lam, total_steps, every, lambda s, w: ck.__setitem__(s, w))
    res.update({'calibration_medians': calib['medians'], 'cpu_s': time.process_time() - t0})
    if log:
        log(f'RAWJ: {res["steps"]} steps, lambda {res["lambda"]}, {res["cpu_s"]:.1f} CPU-s')
    return ck, res


def pfo_paths(X, yO, sex, version, start_weights, total_steps, every, rho=RECIPE['rho'], log=None):
    """CAND (lambda calibrated, rho) and CONT (lambda 0) occupation-only paths from START's weights; income frozen."""
    out, recs = {}, {}
    for arm in ('PFO', 'TASK'):
        t0 = time.process_time()
        tr = PFOTrainer(version, start_weights['income'], start_weights['occupation'], X, yO, sex if arm == 'PFO' else None)
        ck = {}
        if arm == 'PFO':
            cal = tr.calibrate(rho)
            res = run_pfo(tr, cal['lambda'], total_steps, every, lambda s, w: ck.__setitem__(s, w))
            res['calibration'] = {k: v for k, v in cal.items() if k != 'draws'}
        else:
            res = run_pfo(tr, 0.0, total_steps, every, lambda s, w: ck.__setitem__(s, w))
        res['cpu_s'] = time.process_time() - t0
        out[arm], recs[arm] = ck, res
        if log:
            log(f'{arm}: {res["steps"]} steps, lambda {res["lambda"]:.4g}, {res["cpu_s"]:.1f} CPU-s')
    return out, recs


def existing_alt_points(version):
    """Already saved ALT points of FB_PFD_v1 (FB_PFD_RAWJ_LO / _HI) as {step: (config_id, weights)} (label-free files)."""
    from .. import release as RL
    out = {}
    for cfg in ('FB_PFD_RAWJ_LO', 'FB_PFD_RAWJ_HI'):
        if RL.exists(FAMILY, cfg, version):
            P = FK.load_model(FAMILY, cfg, version)
            e = int(str(P['checkpoint']).lstrip('E'))
            out[e * 40] = (cfg, P['weights'])
    return out


def alt_alias(prior, step, weights):
    """The already saved FB_PFD_RAWJ config id whose checkpoint is `step` and whose weights equal `weights` bitwise, else None."""
    hit = prior.get(int(step))
    if hit is not None and all(PD.same_layers(hit[1][t], weights[t]) for t in TASKS):
        return hit[0]
    return None


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, total_steps=TOTAL_STEPS, every=EVERY, budget=None,
              u0=None, neural_raw=None, check_neural_raw=True, reader_check=True, rho=RECIPE['rho'], existing_alt=None):
    from .. import ctlkit as CK
    from .. import famrun as FR
    t0 = time.process_time()
    X = np.asarray(data['X'], dtype=np.float32)
    names = tuple(map(str, data['feature_names']))
    yI, yO, sex = (np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex'))
    unit, row_id = np.asarray(data['unit']), np.asarray(data['row_id'])
    budget = budget or BUDGET_OCC
    steps = tuple(range(every, total_steps + 1, every))
    origin = PD.ORIGIN_EPOCH * 40
    alt_steps = tuple(s for s in steps if s > origin)
    payload, epochs_adm, pins = u0 if u0 is not None else PD.u0_admitted(version)
    if tuple(payload['feature_names']) != names:
        raise PermissionError('U0 feature columns differ from the admitted encoding')
    nr_weights, nr_pins = neural_raw if neural_raw is not None else PD.neural_raw_admitted(version, names)
    pins = {**pins, **nr_pins}
    n = len(unit)
    # 1. cross-fitting: fold start = the U0 recipe; fold START = the fold's RAW-J path at 80 steps; CAND / CONT from fold START
    f = FR.folds(unit, FAMILY, version)
    z2 = lambda: (np.zeros((n, 2)), np.zeros((n, 5)))
    oof = {('RAWJ', s): z2() for s in (0,) + steps}
    oof.update({(a, s): z2() for a in ('PFO', 'TASK') for s in (0,) + steps})
    fold_recs, income_bitwise = [], True
    for j in range(FR.N_XF):
        tr, te = np.flatnonzero(f != j), np.flatnonzero(f == j)
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both parts of a cross-fitting fold')
        st = PD.start_states_recipe(X, yI, yO, version, tr, epochs_adm)
        rj, rj_rec = rawj_path(X[tr], yI[tr], yO[tr], sex[tr], row_id[tr], version, st, total_steps, every,
                               log=lambda s: log(f'fold {j} {s}'))
        start_w = rj[origin]
        po_paths, po_recs = pfo_paths(X[tr], yO[tr], sex[tr], version, start_w, total_steps, every, rho=rho,
                                      log=lambda s: log(f'fold {j} {s}'))
        for key, ck in (('RAWJ', rj), ('PFO', po_paths['PFO']), ('TASK', po_paths['TASK'])):
            for s, W in ck.items():
                pi, po = PD.release_weights(W, X[te])
                oof[(key, s)][0][te], oof[(key, s)][1][te] = pi, po
        s_inc = PD.release_weights(start_w, X[te])[0]
        income_bitwise &= all(np.array_equal(oof[(a, s)][0][te], s_inc) for a in ('PFO', 'TASK') for s in (0,) + steps)
        fold_recs.append({'RAWJ': {k: v for k, v in rj_rec.items() if k != 'logs'},
                          **{a: {k: v for k, v in r.items() if k != 'logs'} for a, r in po_recs.items()}})
        log(json.dumps({'fold': j, 'cpu_s': round(time.process_time() - t0, 1)}))
    # 2. CAND: occupation components of the out-of-fold spend over the fold start (the U0 recipe) and the predeclared rule
    L = {k: FK.person_losses(*v, yI, yO) for k, v in oof.items()}
    fold_start, L_start = L[('RAWJ', 0)], L[('RAWJ', origin)]
    sp = {k: FK.spend(L[k], fold_start, unit, sex) for k in L}
    k_star = FK.select_strength({s: sp[('PFO', s)] for s in steps}, steps, budget)
    sel = {'rule': f'k* = the largest checkpoint S_k (every {every} steps) such that S_{every}..S_k all keep estimate + 2 household-'
                   'clustered SE <= BUDGET_OCC on the five occupation components of the MODEL_FIT out-of-fold paired spend over the fold '
                   'start (the U0 recipe; folds SAR_xf_FB_S<v>); the income vector is START\'s (not spent by the operation)',
           'budget': budget, 'k_star': k_star, 'income_vectors_equal_START_bitwise': bool(income_bitwise),
           'path': {f'{a}_S{s}': FK.jsonable_spend(sp[(a, s)]) for a in ('PFO', 'TASK', 'RAWJ') for s in (0,) + steps
                    if (a, s) in sp}}
    cand = {'config': 'FB_PFO', 'strength': k_star, 'checkpoint': None if k_star is None else f'S{k_star}',
            'status': 'NO_FEASIBLE_STRENGTH' if k_star is None else 'FITTED', 'spend_over_start': None}
    out = {'version': int(version), 'revision': REVISION, 'label_prefix': 'REVISED_', 'recipe': RECIPE, 'pins_sha256': pins,
           'epochs_admitted': epochs_adm, 'selection': sel, 'cand': cand,
           'cont': {'config': 'FB_PFO_TASK', 'strength': k_star, 'gating': True}, 'paths': {}, 'fold_records': fold_recs,
           'start_spend_over_fold_start': FK.jsonable_spend(sp[('RAWJ', origin)])}
    if not income_bitwise:
        raise AssertionError("CAND / CONT income vectors differ from START's (the operation must not touch the income output)")
    if k_star is None:
        log(json.dumps({'stop': 'S10 fails the occupation budget: no candidate, continuation or control point is released'}))
        out['cont']['config'] = None
        out.update({'stop': 'NO_FEASIBLE_STRENGTH', 'saved': {}, 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fb') / f'FIT_PFO_V{int(version)}.json', out)
        return out
    # 3. CTRL (occupation-restricted DQ of START) and ALT (RAW-J beyond 80 steps): brackets over START (MODEL_FIT out of fold)
    start_oof, L_cand = oof[('RAWJ', origin)], L[('PFO', k_star)]
    dc = L_cand['tot_nll'] - L_start['tot_nll']
    di = L_cand['inc_nll'] - L_start['inc_nll']
    cand['spend_over_start'] = {'value': float(dc.mean()), 'se': FK.cluster_se(dc, unit), 'data': 'MODEL_FIT_OOF',
                                'start': f'cross-fitted RAWJ S{origin} (E2)', 'income_part': float(di.mean())}
    Z = FK.compact(X, names)
    tag = PD.DQ_TAG.format(v=int(version))
    basis = CK.dq_fit(*start_oof)
    dq_grid = tuple(CK.DQ_GRID_FINE)
    dq_oof = {g: (start_oof[0], CK.dq_release(basis, *start_oof, Z, g, tag)[1]) for g in dq_grid}
    L_dq = {g: FK.person_losses(*dq_oof[g], yI, yO) for g in dq_grid}
    br = {'CTRL': PD.bracket(L_dq, L_cand, L_start, unit, dq_grid),
          'ALT': PD.bracket({s: L[('RAWJ', s)] for s in alt_steps}, L_cand, L_start, unit, alt_steps)}
    for key, b in br.items():
        if b['lo'] is not None:
            b['flags'] = b['flags'] + ['LO_NOT_SAVED_SPEC_BUDGET']
        out['paths'][key] = {
            'operation': (f"SEX-blind subtractive-dithered quantization of NEURAL_RAW's occupation output (income passed through "
                          f"bitwise; sar.ctlkit DQ_GRID_FINE, tag {tag}; basis on the cross-fitted RAWJ E2 outputs)" if key == 'CTRL' else
                          f"NEURAL_RAW's own recipe continued beyond {origin} steps (RAW_J, rho 1), checkpoints every {every} steps "
                          f"S{alt_steps[0] if alt_steps else '-'}..S{total_steps}"),
            'origin': f'START = NEURAL_RAW (out of fold: cross-fitted RAWJ S{origin}; deployed: NEURAL_RAW rho 1 E2)', **b,
            'lo_config': None, 'lo_spend_over_start': None if b['lo'] is None else b['grid'][str(b['lo'])]['spend_over_start'],
            'hi_config': None, 'hi_spend_over_start': None if b['hi'] is None else b['grid'][str(b['hi'])]['spend_over_start'],
            'alias_of': None}
        log(json.dumps({key: {'lo': b['lo'], 'hi': b['hi'], 'flags': b['flags']}}))

    def _share(Lx):
        a, b = FK.spend(Lx, fold_start, unit), FK.spend(Lx, L_start, unit)
        return {'over_fold_start': a['inc_nll'][0] / max(a['tot_nll'][0], 1e-12), 'over_start': b['inc_nll'][0] / max(b['tot_nll'][0], 1e-12)}
    sel['income_share_of_nll_spend'] = {'CAND': _share(L_cand)}
    for key, Lp in (('CTRL', L_dq), ('ALT', {s: L[('RAWJ', s)] for s in alt_steps})):
        for side in ('lo', 'hi'):
            g = br[key][side]
            if g is not None:
                sel['income_share_of_nll_spend'][f'{key}_{side.upper()}'] = _share(Lp[g])
    log(json.dumps({'k_star': k_star, 'cand_spend_over_start': cand['spend_over_start'], 'cpu_s': round(time.process_time() - t0, 1)}))
    # 4. deployed fits: CAND / CONT from the pinned START (NEURAL_RAW E2); the RAW-J path from U0 (ALT, and the START check)
    starts_full = {t: payload[t] for t in TASKS}
    full_rj, full_rj_rec = rawj_path(X, yI, yO, sex, row_id, version, starts_full, total_steps, every, log=lambda s: log(f'full {s}'))
    start_full = nr_weights if nr_weights is not None else full_rj[origin]
    full_po, full_po_rec = pfo_paths(X, yO, sex, version, start_full, total_steps, every, rho=rho, log=lambda s: log(f'full {s}'))
    checks = {'deployed_start': 'NEURAL_RAW__rho1.0__E2.pt' if nr_weights is not None else f'deployed RAWJ S{origin} (invented-data check only)'}
    if check_neural_raw:
        checks['rawj_S80_equals_NEURAL_RAW_rho1_E2'] = all(PD.same_layers(full_rj[origin][t], nr_weights[t]) for t in TASKS)
        rec0 = PD.start_states_recipe(X, yI, yO, version, np.arange(n), epochs_adm)
        checks['u0_recipe_reproduces_U0_bitwise'] = all(PD.same_layers(PD.layers_of(rec0[t]), PD.layers_of(payload[t])) for t in TASKS)
    checks['cand_income_layers_are_START'] = PD.same_layers(full_po['PFO'][k_star]['income'], start_full['income'])
    # 5. saved configurations: CAND, CONT, CTRL HI, ALT HI (or its alias of an already saved FB_PFD_RAWJ point)
    roles = roles or FK.role_features()
    people, hh = n, len(np.unique(unit))
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the trained networks (SEX only through '
                'the privacy objectives: PFO and RAWJ; never for TASK or DQ), the CAND rule (occupation components, incl. the within-SEX '
                'guards) and the bracket selections (total NLL only); through U0 and NEURAL_RAW (inherited) every start, the frozen '
                'income output and the DQ base; no CALIBRATION or MODEL_SELECT label')
    plan = [('FB_PFO', {'op': 'net', 'arm': 'PFO', 'checkpoint': f'S{k_star}', 'weights': full_po['PFO'][k_star]}, k_star, L_cand,
             oof[('PFO', k_star)], (1, FR.N_XF, 0, 0)),
            ('FB_PFO_TASK', {'op': 'net', 'arm': 'TASK', 'checkpoint': f'S{k_star}', 'weights': full_po['TASK'][k_star]}, k_star,
             L[('TASK', k_star)], oof[('TASK', k_star)], (1, FR.N_XF, 0, 0))]
    g = br['CTRL']['hi']
    if g is not None:
        plan.append(('FB_PFO_DQ_HI', {'op': 'dq_occ', 'arm': 'DQ', 'weights': start_full, 'dq_basis': basis, 'delta': float(g),
                                      'dq_tag': tag, 'dq_base': checks['deployed_start']}, float(g), L_dq[g], dq_oof[g], (0, 0, 0, 0)))
        out['paths']['CTRL']['hi_config'] = 'FB_PFO_DQ_HI'
    s_hi = br['ALT']['hi']
    if s_hi is not None:
        prior = existing_alt_points(version) if existing_alt is None else existing_alt
        hit = alt_alias(prior, s_hi, full_rj[s_hi])
        if hit is not None:
            out['paths']['ALT'].update({'hi_config': hit, 'alias_of': hit})
        else:
            plan.append(('FB_PFO_RAWJ_HI', {'op': 'net', 'arm': 'RAWJ', 'checkpoint': f'S{s_hi}', 'weights': full_rj[s_hi]}, s_hi,
                         L[('RAWJ', s_hi)], oof[('RAWJ', s_hi)], (1, FR.N_XF, 0, FR.N_XF + 1)))
            out['paths']['ALT']['hi_config'] = 'FB_PFO_RAWJ_HI'
    saved, views = {}, {'FOLD_START': oof[('RAWJ', 0)], 'START': start_oof}
    for cfg, pl, strength, Lx, oof_pair, fits in plan:
        meta = FK.write_model(FAMILY, cfg, version, {'feature_names': names, 'recipe': RECIPE, 'pins_sha256': pins, 'revision': REVISION,
                                                     **pl})
        arrays = FK.release_all(predict, cfg, version, roles)
        d = Lx['tot_nll'] - L_start['tot_nll']
        key = CONFIGS[cfg]['path']
        extra = {'revision': REVISION, 'label_prefix': 'REVISED_', 'path': key, 'arm': pl['arm'], 'checkpoint': pl.get('checkpoint'),
                 'strength': strength, 'selection_k_star': k_star, 'model_sha256': meta['model_sha256'], 'pins_sha256': pins,
                 'checks': checks, 'oof_spend_over_fold_start': FK.jsonable_spend(FK.spend(Lx, fold_start, unit, sex)),
                 'oof_spend_over_start': {'value': float(d.mean()), 'se': FK.cluster_se(d, unit)},
                 'bracket_flags': br[key]['flags'] if key in br else None, 'n_start_refits': 2 * FR.N_XF + 2 if pl['arm'] != 'DQ' else 0}
        if pl['arm'] == 'DQ':
            extra.update({'delta': float(pl['delta']), 'dq_tag': tag, 'dq_base': pl['dq_base'], 'outputs': 'occupation'})
        elif pl['arm'] == 'RAWJ':
            extra['lambda'] = full_rj_rec['lambda']
        else:
            extra['lambda'] = full_po_rec[pl['arm']]['lambda']
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure, people, hh,
                        time.process_time() - t0, extra)
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = extra
        views[cfg] = oof_pair
    if reader_check:
        t1 = time.process_time()
        out['mf_reader_check'] = {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input',
                                  'auc': FK.mf_reader_check(views, sex, unit, f'SAR_mfcheck_FB_R1_S{int(version)}'), 'cpu_s': time.process_time() - t1}
    out.update({'saved': saved, 'n_saved_new': len(saved), 'checks': checks,
                'full_records': {'RAWJ': {k: v for k, v in full_rj_rec.items() if k != 'logs'},
                                 **{a: {k: v for k, v in r.items() if k != 'logs'} for a, r in full_po_rec.items()}},
                'full_logs_PFO': full_po_rec['PFO']['logs'], 'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fb') / f'FIT_PFO_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks (invented data; no label file)
def selftest():
    import tempfile
    import torch
    from pathlib import Path
    from .. import ctlkit as CK
    torch.set_num_threads(1)
    ok, info = {}, {}
    # 1. the occupation-only loop never touches the income network; the pair term sees the income as a constant input
    Xs, yI, yO, sx = PD._invented(3000, 12, 11)
    torch.manual_seed(0)
    W0 = {t: PD.layers_of(PD.network(t, 12).state_dict()) for t in TASKS}
    ck, recs = pfo_paths(Xs, yO, sx, 0, W0, 60, EVERY)
    ok['checkpoints_every_10'] = sorted(ck['PFO']) == list(range(0, 61, 10)) and sorted(ck['TASK']) == list(range(0, 61, 10))
    ok['income_layers_frozen'] = all(PD.same_layers(ck[a][s]['income'], W0['income']) for a in ('PFO', 'TASK') for s in ck[a])
    ok['occupation_moves'] = not PD.same_layers(ck['PFO'][60]['occupation'], W0['occupation'])
    ok['cand_lambda_positive_cont_zero'] = recs['PFO']['lambda'] > 0 and recs['TASK']['lambda'] == 0.0
    tr = PFOTrainer(0, W0['income'], W0['occupation'], Xs, yO, sx)
    rows = torch.arange(0, 2048)
    zi = tr.z_inc[rows].clone().requires_grad_(True)
    terms, _ = privacy_terms_occ(zi, tr.logits(rows), torch.from_numpy(sx[:2048]))
    g_inc = torch.autograd.grad(terms['occupation'] + terms['pair'], zi, allow_unused=True)[0]
    ok['pair_term_depends_on_income_input'] = g_inc is not None and float(g_inc.abs().sum()) > 0   # income enters P_O ...
    ok['income_input_is_constant'] = not tr.z_inc.requires_grad
    pf = PD.release_weights(ck['PFO'][60], Xs); tk = PD.release_weights(ck['TASK'][60], Xs); s0 = PD.release_weights(W0, Xs)
    ok['income_output_bitwise_start'] = bool(np.array_equal(pf[0], s0[0]) and np.array_equal(tk[0], s0[0]))
    ok['cont_ignores_sex'] = PD.same_layers(pfo_paths(Xs, yO, 1 - sx, 0, W0, 20, EVERY)[0]['TASK'][20]['occupation'], ck['TASK'][20]['occupation'])

    def pairD(P):
        return float(privacy_terms_occ(torch.from_numpy(np.log(P[0])), torch.from_numpy(np.log(P[1])), torch.from_numpy(sx))[1]['pair'])
    info['invented_pair_d2'] = {'START': pairD(s0), 'CONT_S60': pairD(tk), 'CAND_S60': pairD(pf)}
    ok['cand_lowers_invented_pair_separation'] = info['invented_pair_d2']['CAND_S60'] < info['invented_pair_d2']['CONT_S60']
    # 2. the fine RAW-J loop reproduces sar.fb.pfd.run_rawj at every epoch boundary, bitwise
    rid = np.arange(3000) * 7 + 3
    st = {t: PD.network(t, 12).state_dict() for t in TASKS}
    coarse, _ = PD.train_paths(Xs, yI, yO, sx, rid, 0, st, epochs=3, arms=('RAWJ',))
    fine, _ = rawj_path(Xs, yI, yO, sx, rid, 0, st, 120, EVERY)
    ok['rawj_fine_equals_run_rawj_at_epochs'] = all(PD.same_layers(fine[40 * e][t], coarse['RAWJ'][e][t]) for e in range(4) for t in TASKS)
    ok['rawj_fine_has_mid_epoch_points'] = sorted(fine) == list(range(0, 121, 10))
    # 3. the R1 rule reads the occupation components only; the income components cannot stop it
    z = {k: (0.0, 0.0) for k in PD.BUDGET}
    inc_bad = dict(z, inc_brier=(0.01, 0.0), inc_nll=(0.02, 0.0))
    occ_bad = dict(z, occ_brier_women=(0.01, 0.0))
    ok['rule_ignores_income_components'] = FK.select_strength({10: inc_bad, 20: inc_bad}, (10, 20), BUDGET_OCC) == 20
    ok['rule_stops_on_occupation_guard'] = FK.select_strength({10: z, 20: occ_bad, 30: z}, (10, 20, 30), BUDGET_OCC) == 10
    # 4. the CTRL law: occupation-restricted DQ = the shared DQ law's occupation output, income passed through bitwise
    F = C.features('MODEL_FIT')
    Xr = F['X'][:400]
    torch.manual_seed(3)
    Wn = {t: PD.layers_of(PD.network(t, Xr.shape[1]).state_dict()) for t in TASKS}
    bI, bO = PD.release_weights(Wn, np.asarray(Xr, dtype=np.float64))
    basis = CK.dq_fit(bI, bO)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fb_pfo_selftest_'))
    try:
        C.RUNS = tmp
        FK.write_model(FAMILY, 'FB_PFO_DQ_HI', 0, {'feature_names': tuple(F['feature_names']), 'op': 'dq_occ', 'weights': Wn,
                                                   'dq_basis': basis, 'delta': 0.5, 'dq_tag': 'TEST'})
        got = predict('FB_PFO_DQ_HI', 0, Xr, F['feature_names'])
        from . import predict as pkg_predict
        via_pkg = pkg_predict('FB_PFO_DQ_HI', 0, Xr, F['feature_names'])
    finally:
        C.RUNS = old
    full = CK.dq_release(basis, bI, bO, FK.compact(np.asarray(Xr, dtype=np.float64), F['feature_names']), 0.5, 'TEST')
    ok['ctrl_income_passed_through_bitwise'] = bool(np.array_equal(got['income'], bI))
    ok['ctrl_occupation_is_shared_law_bitwise'] = bool(np.array_equal(got['occupation'], full[1]))
    ok['package_dispatch_routes_pfo'] = bool(np.array_equal(via_pkg['occupation'], got['occupation']))
    # 5. alias logic: an ALT HI equal to an already saved FB_PFD_RAWJ point is aliased, not saved again
    Wb = {t: [(w + 1.0, b) for w, b in Wn[t]] for t in TASKS}
    ok['alt_alias_bitwise_hit'] = alt_alias({120: ('FB_PFD_RAWJ_HI', Wn)}, 120, Wn) == 'FB_PFD_RAWJ_HI'
    ok['alt_alias_other_weights_or_step'] = alt_alias({120: ('FB_PFD_RAWJ_HI', Wn)}, 120, Wb) is None and \
        alt_alias({120: ('FB_PFD_RAWJ_HI', Wn)}, 130, Wn) is None
    ok['all_ok'] = all(ok.values())
    return {**ok, 'info': info}


def selftest_stage(n=3000, seed=5, total_steps=160, rho=50.0):
    """stage_fit end to end on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read): U0 is
    replaced by invented weights and the deployed START by the deployed RAW-J replica; models go to a temporary directory; nothing
    is saved to the campaign's runs. A fake already-saved FB_PFD_RAWJ point exercises the alias branch."""
    import tempfile
    import torch
    from pathlib import Path
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    X = F['X'][idx].astype(np.float32)
    sex = ((X[:, 1] + rng.normal(0, 0.7, n)) > np.median(X[:, 1])).astype(np.int64)   # invented SEX, predictable from x
    s = sex - 0.5
    data = {'X': X, 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': ((X[:, 1] + 0.5 * s + rng.normal(0, 0.3, n)) > 0.0).astype(np.int64),
            'occupation': np.clip(np.floor(2 + 1.5 * X[:, 1] - 1.0 * s + rng.normal(0, 0.3, n)), 0, 4).astype(np.int64), 'sex': sex}
    adm = {'income': 15, 'occupation': 15}                 # an invented 'U0': the U0 recipe on the invented labels (so START is fitted
    st = PD.start_states_recipe(X, data['income'], data['occupation'], 0, np.arange(n), adm)   # and CAND spends over it)
    payload = {'feature_names': list(F['feature_names']), **st}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fb_pfo_stage_'))
    try:
        C.RUNS = tmp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s_: None, record_dir=False, total_steps=total_steps,
                        every=EVERY, budget={k: 100.0 for k in BUDGET_OCC},
                        u0=(payload, adm, {'invented-u0': 'none'}), neural_raw=(None, {'invented': 'none'}),
                        check_neural_raw=False, reader_check=True, rho=rho, existing_alt={})
        re = {c: predict(c, 0, data['X'], data['feature_names']) for c in saved}
    finally:
        C.RUNS = old
    c_sp = out['cand']['spend_over_start']['value']
    ok = {'cand_and_cont_saved': {'FB_PFO', 'FB_PFO_TASK'} <= set(saved),
          'at_most_four_new_per_version': len(saved) <= 4,
          'k_star_is_last_checkpoint': out['cand']['strength'] == total_steps and out['cand']['status'] == 'FITTED',
          'income_bitwise_start_all_saved_trained_configs': all(np.array_equal(saved[c][0]['MODEL_FIT']['income'],
                                                                               saved['FB_PFO'][0]['MODEL_FIT']['income'])
                                                              for c in saved if c in ('FB_PFO_TASK', 'FB_PFO_DQ_HI')),
          'cand_income_spend_over_start_zero': out['cand']['spend_over_start']['income_part'] == 0.0,
          'fit_record_fields': all(set(out['paths'][k]) >= {'origin', 'selection_data', 'grid', 'lo', 'hi', 'hi_config', 'flags'}
                                   for k in ('CTRL', 'ALT')) and out['cont']['config'] == 'FB_PFO_TASK',
          'records_valid': all(r['kind'] == CONFIGS[c]['kind'] and r['path'] == CONFIGS[c]['path'] for c, (_, r) in saved.items()),
          'release_is_predict_bitwise': all(np.array_equal(saved[c][0]['MODEL_FIT'][t], re[c][t]) for c in saved for t in TASKS),
          'valid_vectors': all(np.allclose(a['MODEL_FIT'][t].sum(1), 1, atol=1e-9) for a, _ in saved.values() for t in TASKS),
          'reader_check_ran': 'mf_reader_check' in out,
          'checks_income_layers': out['checks'].get('cand_income_layers_are_START') is True}
    ok['bracket_branch_exercised'] = c_sp > 0 and 'FB_PFO_DQ_HI' in saved and out['paths']['ALT']['hi_config'] is not None
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['cand_spend_over_start'] = out['cand']['spend_over_start']
    ok['brackets'] = {k: {'lo': out['paths'][k]['lo'], 'hi': out['paths'][k]['hi'], 'hi_config': out['paths'][k]['hi_config'],
                          'flags': out['paths'][k]['flags']} for k in ('CTRL', 'ALT')}
    ok['cpu_s'] = round(out['cpu_s'], 1)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fb.pfo')
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
                      'brackets': {k: {'lo': v['lo'], 'hi': v['hi'], 'hi_config': v.get('hi_config'), 'flags': v['flags']}
                                   for k, v in out['paths'].items()},
                      'saved': sorted(out['saved']), 'checks': out.get('checks'), 'cpu_s': round(out['cpu_s'], 1)}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

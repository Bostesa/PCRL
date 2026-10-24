"""FA: SUCC-anchored joint teacher-feature residual (TFR; card draft roles/investigator_a/CARD_DRAFT.md, to be committed as
cards/FA_TFR_v1.md). PCRL campaign `pcrl_strong_anchor_release_v1`; EXPLORATORY DEVELOPMENT on previously used public-use U.S.
Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is a
MODEL_FIT training label of the critics and of the guard part of the selection only; never a runtime input or output.

Operation (one, both recipients at once):
  START     SUCC_S<v> = H3_STRONG_LJDIR_PB income log-odds d_S(x) + JOINT_TREE E20 occupation logits z_S(x) (frozen; ihs.fam_h.predict,
            checked bitwise against the pinned MODEL_SELECT wires by the investigator's label-free check)
  features  phi(x) = [standardized compact fields (AGEP, WKHP, SCHL, POBP code; COW and MAR one-hot), SUCC's own logits (d_S, z_S),
            the admitted HGB task teachers' logits (d_H, z_H; nrm fa_cache TASK.pkl, A1_PO_16 settings): their household out-of-fold
            values on the MODEL_FIT training rows, the deployed models on every other row]
  residual  r_theta(phi) = (r_I, r_O) in R x R^5: a 64-64 ReLU network whose output layer starts at zero (theta = 0 is SUCC)
  release   income = softmax(-(d_S + s r_I)/2, (d_S + s r_I)/2); occupation = softmax(centre(z_S + s r_O)); decisions = argmax; s = 1
            for CAND; s = 0 releases START exactly (SUCC's own vectors)
  training  task term: cross-entropy of both released vectors to the HGB teachers' MODEL_FIT out-of-fold probabilities (distillation
            toward the strong teacher; no task label is read by the fit itself, so SUCC's in-sample MODEL_FIT fit cannot be mistaken
            for task quality) + lambda * [risk(income) + risk(occupation) + risk(pair)] with the lineage's online SEX critics
            (acs_replication.leafcorr.Critics, unchanged); lambda = rho * median ||grad task|| / median ||grad joint risk|| at
            theta = 0 (16 draws); AdamW lr 1e-3, weight decay 0.01, clip 5, 20 epochs x 40 steps x 256 rows, kept at E20
  strength  rho* = the largest rho on RHO_GRID such that it and every smaller rho meet (a) the CALIBRATION paired spend over SUCC
            (estimate + 2 household-clustered SE <= BUDGET_CAL) and (b) the MODEL_FIT household out-of-fold men's / women's occupation
            Brier spend over the lambda = 0 path's LO point for that rho (estimate + 2 SE <= GUARD_DID)
CALIBRATION exposure (coordinator correction, 2026-10-24): the occupation side of SUCC (JOINT_TREE on the T_TREE routing), the HGB
teachers and the residual are out of sample on CALIBRATION; SUCC's income output H3 is NOT (its STRONG stack, its beta budget and its
Platt map were fitted on all 2,982 CALIBRATION people), and Ucal's temperatures are in-sample there too. Income-side CALIBRATION spends
over SUCC are therefore biased in SUCC's favour: conservative for CAND's selection, not honest estimates.
Controls (the same residual network, features, rows, folds, optimizer and schedule with lambda = 0: plain learned distillation from
START toward the teachers):
  CTRL  FA_TFR_TASK_LO / _HI, two points of the lambda = 0 path (origin START at s = 0) on the declared selection data, CALIBRATION paired
        total-NLL spend over SUCC: LO = the S_PATH point with the largest spend <= CAND's spend; HI = the point with the smallest spend
        >= CAND's spend + 2 SE (SE = the household-clustered CALIBRATION SE of that point's paired difference against CAND); flag
        CONTROL_PATH_MAY_NOT_BRACKET if no point reaches HI (CONTROL_PATH_NO_LO_POINT if none is <= CAND's spend). Comparisons are on
        spend values, never on s. A point at s = 0 is START itself and a point at s = 1 is CONT: recorded as aliases, not saved twice.
  CONT  FA_TFR_CONT, the lambda = 0 residual at s = 1 (zero-protection continuation, about the HGB teachers); gating.
Specifications: CAND 2 + CTRL 4 + CONT 2 = 8 at most (fewer when a CTRL point is an alias or a version has no feasible rho).

Commands (real data only through the campaign queue after the registration receipt and the committed card):
    <repository>/.venv/bin/python -m sar.fa.tfr selftest
    <repository>/.venv/bin/python -m sar.fa.tfr preflight --version V          (label-free pin check)
    <repository>/.venv/bin/python -m sar.fa.tfr fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FA_TFR_v1.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK
from .. import famrun as FR

FAMILY = 'FA'
SUCC_CONFIG = 'H3_STRONG_LJDIR_PB'          # ihs family H configuration; ihs.fam_h.predict(seed, config, X, names) = SUCC_S<seed>
RHO_GRID = (0.3, 1.0, 3.0, 10.0)
S_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
S_PATH = (0.0,) + S_GRID                    # the CTRL path; s = 0 is START (SUCC) exactly
ALIAS_ENDPOINTS = True                      # a CTRL point at s = 0 (START) or s = 1 (CONT) is recorded as an alias, not saved twice
MATCH_KEY = 'tot_nll'
# (a) task components: paired CALIBRATION spend over SUCC, estimate + 2 household-clustered SE <= limit (0.40-0.60 of SUCC's
#     MODEL_SELECT point margins; card section 6). The income components are biased in SUCC's favour (H3 is in-sample on CALIBRATION).
BUDGET_CAL = {'occ_nll': 0.0040, 'occ_brier': 0.0025, 'occ_acc': 0.0050, 'inc_nll': 0.0060, 'inc_brier': 0.0030, 'inc_acc': 0.0035}
# (b) the two SEX guards: MODEL_FIT out-of-fold occupation summed-Brier spend of the candidate over the lambda = 0 path's LO point
GUARD_DID = {'occ_brier_men': 0.0005, 'occ_brier_women': 0.0050}
RECIPE = {'hidden': (64, 64), 'lr': 1e-3, 'weight_decay': 0.01, 'clip': 5.0, 'batch': 256, 'steps_per_epoch': 40, 'epochs': 20,
          'checkpoints': (0, 2, 5, 10, 20), 'warm_rounds': 80, 'critic_rounds_per_update': 5, 'calibration_draws': 16,
          'reference_rows': 256}
CONFIGS = {
    'FA_TFR': {'kind': 'candidate', 'role': 'CAND',
               'what': "SUCC-anchored joint teacher-feature residual: both outputs corrected from SUCC's own logits by a network on the "
                       "compact fields, SUCC's logits and the HGB task teachers' logits, trained by distillation toward the HGB teachers "
                       "with the joint (income, occupation, pair) SEX critics at the selected rho*"},
    'FA_TFR_TASK_LO': {'kind': 'control', 'role': 'CTRL',
                       'what': 'CTRL lower bracket point: the lambda = 0 residual (plain distillation from START toward the HGB teachers) '
                               "at the path point with the largest CALIBRATION paired total-NLL spend over SUCC that is <= CAND's"},
    'FA_TFR_TASK_HI': {'kind': 'control', 'role': 'CTRL',
                       'what': 'CTRL upper bracket point: the lambda = 0 residual at the path point with the smallest CALIBRATION paired '
                               "total-NLL spend over SUCC that is >= CAND's + 2 SE of its paired difference against CAND"},
    'FA_TFR_CONT': {'kind': 'control', 'role': 'CONT', 'gating': True,
                    'what': 'zero-protection continuation (gating): the lambda = 0 residual at full scale s = 1'},
}
PURPOSE = ('FA TFR: SUCC-anchored joint residual of both outputs with the joint SEX critics (MODEL_FIT SEX), the guard part of its '
           'strength selection and its lambda = 0 control path on MODEL_FIT')
PURPOSE_CAL = ('FA TFR: CALIBRATION income / occupation for the task part of the strength selection and the CTRL bracket only (paired '
               'spend over SUCC; no fit on CALIBRATION; identical exposure for the controls)')
CODE_MODULES = ('sar.fa.tfr', 'sar.famkit', 'sar.famrun', 'acs_replication.leafcorr', 'acs_replication.trees', 'ihs.fam_h',
                'nrm.fa.prc', 'ohn.fam_a')
CLIPP = 1e-300


# ---------------------------------------------------------------- admission pins (this campaign's SOURCE_ADMISSION.json)
def _admission():
    return json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())


def admission_pin(rel, adm=None):
    """The sha256 pinned for a frozen file in this campaign's SOURCE_ADMISSION.json: any entry whose key ends with the file's last three
    path components and holds {'sha256': ...} (or the hex string itself). None if absent; PermissionError if pins disagree."""
    tail = '/'.join(str(rel).split('/')[-3:])
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and k.endswith(tail):
                    h = v.get('sha256') if isinstance(v, dict) else v
                    if isinstance(h, str) and len(h) == 64:
                        found.add(h)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(_admission() if adm is None else adm)
    if len(found) > 1:
        raise PermissionError(f'{tail}: conflicting pins in SOURCE_ADMISSION.json')
    return next(iter(found)) if found else None


# ---------------------------------------------------------------- frozen teachers (label-free: outputs of frozen functions)
def _succ(version, X, names):
    """SUCC_S<v> probabilities from admitted features (H3 income + JOINT_TREE E20 occupation; the ihs deployment function)."""
    from ihs import fam_h
    return fam_h.predict(int(version), SUCC_CONFIG, X, tuple(map(str, names)))


def task_cache(version):
    """The admitted HGB task teachers (nrm fa_cache TASK.pkl), hash-checked against this campaign's SOURCE_ADMISSION.json pin."""
    import pickle
    rel = f'fa_cache/V{int(version)}/TASK.pkl'
    want = admission_pin(rel)
    if want is None:
        raise PermissionError(f'{rel}: no pin in this campaign\'s SOURCE_ADMISSION.json')
    b = (C.NRM_FA_CACHE / f'V{int(version)}' / 'TASK.pkl').read_bytes()
    if FK.sha_bytes(b) != want:
        raise PermissionError('the cached task teachers differ from their admission pin')
    return pickle.loads(b), want


def frozen_inputs(version):
    """The frozen files predict() reads besides its own model files (SUCC through ihs.fam_h.predict; hash-pinned)."""
    m = C.ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
    v = int(version)
    return [m / 'runs' / 'H' / SUCC_CONFIG / f'S{v}' / 'STATE.json', m / 'runs' / 'H' / 'STEP1' / f'S{v}' / 'STEP1.json',
            m / 'runs' / 'H' / 'STEP1' / f'S{v}' / 'HGB.pkl',
            m / 'admitted' / 'stage_a' / f'S{v}' / 'TREE_TEACHER_income.json',
            m / 'admitted' / 'stage_a' / f'S{v}' / 'TREE_TEACHER_occupation.json',
            m / 'admitted' / 'stage_b' / f'S{v}' / 'LOCAL_TREE__E20.npz', m / 'admitted' / 'stage_b' / f'S{v}' / 'JOINT_TREE__E20.npz']


def frozen_sha(version):
    return {str(p.relative_to(C.ROOT)): C.sha(p) for p in frozen_inputs(version)}


def preflight(version):
    """Label-free check before any label read: TASK.pkl and every SUCC frozen file match this campaign's admission pins."""
    adm = _admission()
    out = {'TASK.pkl': None, 'frozen': {}, 'missing': [], 'mismatched': []}
    try:
        out['TASK.pkl'] = task_cache(version)[1]
    except PermissionError as e:
        out['missing' if 'no pin' in str(e) else 'mismatched'].append(f'fa_cache/V{int(version)}/TASK.pkl')
    for rel, h in frozen_sha(version).items():
        pin = admission_pin(rel, adm)
        out['frozen'][rel] = {'sha256': h, 'pin': pin}
        if pin is None:
            out['missing'].append(rel)
        elif pin != h:
            out['mismatched'].append(rel)
    out['ok'] = not out['missing'] and not out['mismatched']
    return out


def logodds(Pi):
    Pi = np.asarray(Pi, np.float64)
    return np.log(np.clip(Pi[:, 1], CLIPP, None)) - np.log(np.clip(Pi[:, 0], CLIPP, None))


def clogits(Po):
    L = np.log(np.clip(np.asarray(Po, np.float64), CLIPP, None))
    return L - L.mean(1, keepdims=True)


def blocks(Zc, succ, hgb):
    """Packed label-free array per person: [d_S, z_S (5) | d_H, z_H (5) | compact fields (6)] (n, 18)."""
    return np.column_stack([logodds(succ[0]), clogits(succ[1]), logodds(hgb[0]), clogits(hgb[1]), np.asarray(Zc, np.float64)])


def phi_fit(A):
    """Feature-map parameters from the training rows' packed array (label-free statistics)."""
    Zc = A[:, 12:18]
    num = Zc[:, [0, 1, 3, 5]]
    tb = A[:, 0:12]
    return {'num_mu': num.mean(0), 'num_sd': num.std(0) + 1e-9, 'lg_mu': tb.mean(0), 'lg_sd': tb.std(0) + 1e-9,
            'levels_COW': np.unique(Zc[:, 2]), 'levels_MAR': np.unique(Zc[:, 4])}


def phi_apply(P, A):
    Zc = A[:, 12:18]
    num = (Zc[:, [0, 1, 3, 5]] - P['num_mu']) / P['num_sd']
    oh = [(Zc[:, 2][:, None] == P['levels_COW'][None, :]).astype(np.float64),
          (Zc[:, 4][:, None] == P['levels_MAR'][None, :]).astype(np.float64)]
    lg = (A[:, 0:12] - P['lg_mu']) / P['lg_sd']
    return np.column_stack([num] + oh + [lg])


# ---------------------------------------------------------------- the residual network (torch for training, numpy for deployment)
def _net(d, seed):
    import torch
    torch.manual_seed(int(seed) % 2 ** 31)
    h1, h2 = RECIPE['hidden']
    m = torch.nn.Sequential(torch.nn.Linear(d, h1, dtype=torch.float64), torch.nn.ReLU(),
                            torch.nn.Linear(h1, h2, dtype=torch.float64), torch.nn.ReLU(),
                            torch.nn.Linear(h2, 6, dtype=torch.float64))
    with torch.no_grad():
        m[4].weight.zero_(); m[4].bias.zero_()           # theta = 0 reproduces SUCC
    return m


def net_state(m):
    return [p.detach().cpu().numpy().copy() for p in m.parameters()]


def net_numpy(W, F):
    h = np.maximum(F @ W[0].T + W[1], 0.0)
    h = np.maximum(h @ W[2].T + W[3], 0.0)
    return h @ W[4].T + W[5]


def release_arrays(A, out, s):
    """Released vectors from the packed array A (SUCC's logits in columns 0-5) and the residual output `out` at scale s > 0."""
    from acs_replication import trees
    d = A[:, 0] + float(s) * out[:, 0]
    z = A[:, 1:6] + float(s) * out[:, 1:6]
    return trees.probabilities('income', d), trees.probabilities('occupation', z)


def release_state(state, A, s=1.0, label=None):
    W = state['checkpoints'][state['keep'] if label is None else label]
    return release_arrays(A, net_numpy(W, phi_apply(state['phi'], A)), s)


def train(A, sex, rho, seed_key, epochs=None, log=None):
    """One training run on the given rows. The task term is label-free (the HGB teachers' probabilities in A); rho = 0: no critics
    (SEX unused)."""
    import random
    import statistics
    import torch
    import torch.nn.functional as F
    from acs_replication import leafcorr as lc
    from acs_replication import trees
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    t0 = time.process_time()
    E = int(epochs or RECIPE['epochs'])
    ph = phi_fit(A)
    Xf = torch.from_numpy(np.ascontiguousarray(phi_apply(ph, A)))
    B = torch.from_numpy(np.ascontiguousarray(A[:, 0:6]))
    TI = torch.from_numpy(np.ascontiguousarray(trees.probabilities('income', A[:, 6])))
    TO = torch.from_numpy(np.ascontiguousarray(trees.probabilities('occupation', A[:, 7:12])))
    n = len(A)
    seed = lc.seed_for('SAR_FA_TFR', seed_key) % 2 ** 31
    net = _net(Xf.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])

    def logits(rows):
        o = net(Xf[rows])
        d = B[rows, 0] + o[:, 0]
        z = B[rows, 1:6] + o[:, 1:6]
        return torch.stack((-d / 2, d / 2), 1), z - z.mean(1, keepdim=True)

    def wires(rows):
        out = []
        for l_ in logits(rows):
            p = torch.softmax(l_, 1)
            out.append(torch.cat((p, p.argmax(1, keepdim=True).to(p.dtype).detach()), 1))
        return out

    def task_loss(rows):                     # distillation toward the HGB task teachers (no task label is read)
        li, lo = logits(rows)
        return -(TI[rows] * F.log_softmax(li, 1)).sum(1).mean() - (TO[rows] * F.log_softmax(lo, 1)).sum(1).mean()

    ck = {0: net_state(net)}
    lam, critics, info = 0.0, None, {'rho': float(rho)}
    priv_rng = random.Random(lc.seed_for('SAR_FA_TFR_priv', seed_key))
    crit_rng = random.Random(lc.seed_for('SAR_FA_TFR_crit', seed_key))
    task_rng = np.random.default_rng(lc.seed_for('SAR_FA_TFR_task', seed_key) % 2 ** 32)
    if rho > 0:
        S = torch.from_numpy(np.asarray(sex, np.int64))
        ref = torch.from_numpy(np.sort(np.random.default_rng(lc.seed_for('SAR_FA_TFR_ref', seed_key) % 2 ** 32)
                                       .choice(n, min(RECIPE['reference_rows'], n), replace=False)))
        critics = lc.Critics(seed, S)

        def pbatch(rng):
            pos = torch.from_numpy(np.asarray(rng.sample(range(n), RECIPE['batch']), np.int64))
            return pos, S[pos]

        def refresh():
            with torch.no_grad():
                w = wires(ref)
            critics.refresh(w[0], w[1])

        def critic_round(rng):
            r, s_ = pbatch(rng)
            with torch.no_grad():
                w = wires(r)
            critics.update(w[0], w[1], s_)
        refresh()
        for _ in range(RECIPE['warm_rounds']):
            critic_round(crit_rng)
        cal_rng = random.Random(lc.seed_for('SAR_FA_TFR_cal', seed_key))
        cal_task = np.random.default_rng(lc.seed_for('SAR_FA_TFR_caltask', seed_key) % 2 ** 32)
        gt, gp = [], []
        params = list(net.parameters())
        for _ in range(RECIPE['calibration_draws']):
            tr = torch.from_numpy(cal_task.choice(n, RECIPE['batch'], replace=False))
            g = torch.autograd.grad(task_loss(tr), params)
            gt.append(float(torch.sqrt(sum((x ** 2).sum() for x in g))))
            r, s_ = pbatch(cal_rng)
            w = wires(r)
            P = lc.objective(critics.terms(w[0], w[1], s_), 'joint')
            g = torch.autograd.grad(P, params, allow_unused=True)
            gp.append(float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None))))
        lam = float(rho) * statistics.median(gt) / max(statistics.median(gp), 1e-300)
        if not (np.isfinite(lam) and 0 < lam < 1e6):
            raise FloatingPointError('calibration coefficient outside the registered range')
        info.update({'median_task_grad': statistics.median(gt), 'median_joint_grad': statistics.median(gp)})
    info['lambda'] = lam
    trace = []
    for ep in range(E + 1):
        if critics is not None:
            refresh()
        if ep > 0 and (ep in RECIPE['checkpoints'] or ep == E):
            ck[ep] = net_state(net)
        if ep == E:
            break
        order = task_rng.permutation(n)
        for b in range(RECIPE['steps_per_epoch']):
            rows = torch.from_numpy(order[(b * RECIPE['batch']) % n:][:RECIPE['batch']])
            loss = task_loss(rows)
            if critics is not None:
                r, s_ = pbatch(priv_rng)
                w = wires(r)
                loss = loss + lam * lc.objective(critics.terms(w[0], w[1], s_), 'joint')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE['clip'])
            opt.step()
            if critics is not None:
                for _ in range(RECIPE['critic_rounds_per_update']):
                    critic_round(crit_rng)
        trace.append(float(loss.detach()))
        if log:
            log(f'epoch {ep + 1} loss {trace[-1]:.5f} ({time.process_time() - t0:.1f} CPU-s)')
    return {'phi': ph, 'checkpoints': ck, 'keep': E, 'info': {**info, 'trace': trace, 'cpu_s': time.process_time() - t0,
                                                               'critic_sha256': None if critics is None else critics.state_sha()}}


# ---------------------------------------------------------------- famrun interface (Z = packed array; labels arrive as arguments)
def _key(version, unit, tag):
    return f'V{int(version)}|{tag}|{int(len(unit))}|{int(np.asarray(unit).sum() % 1000003)}'


def fit_rho(A, y_inc, y_occ, sex, unit, version, rho, epochs=None):
    return train(A, sex, rho, _key(version, unit, f'rho{rho:g}'), epochs=epochs)


def release_rho(state, A):
    return release_state(state, A, 1.0)


def fit_task(A, y_inc, y_occ, sex, unit, version, epochs=None):
    return train(A, None, 0.0, _key(version, unit, 'task'), epochs=epochs)


def release_scaled(state, A, s):
    return release_state(state, A, s)


def path_release(state0, A, s, start_pair):
    """A point of the lambda = 0 path: s = 0 is START (SUCC's own vectors, exactly)."""
    if float(s) == 0.0:
        return np.array(start_pair[0], np.float64, copy=True), np.array(start_pair[1], np.float64, copy=True)
    return release_state(state0, A, s)


# ---------------------------------------------------------------- the CTRL bracket (coordinator rule, 2026-10-24)
def bracket(path_L, cand_L, start_L, unit, grid=S_PATH, key=MATCH_KEY):
    """CTRL points on the declared selection data (CALIBRATION paired `key` spend over START). LO = the path point with the largest
    spend <= CAND's spend; HI = the point with the smallest spend >= CAND's spend + 2 SE, SE = the household-clustered SE of that
    point's paired difference against CAND. Comparisons are on spend values, never on s; ties go to the smaller s."""
    dc0 = cand_L[key] - start_L[key]
    cand = {'spend_over_start': float(dc0.mean()), 'se': FK.cluster_se(dc0, unit)}
    rows = {}
    for s in grid:
        d0 = path_L[s][key] - start_L[key]
        dc = path_L[s][key] - cand_L[key]
        rows[str(s)] = {'s': float(s), 'spend_over_start': float(d0.mean()), 'se_over_start': FK.cluster_se(d0, unit),
                        'diff_vs_cand': float(dc.mean()), 'se_diff_vs_cand': FK.cluster_se(dc, unit)}
    cs = cand['spend_over_start']
    lo_c = [s for s in grid if rows[str(s)]['spend_over_start'] <= cs]
    hi_c = [s for s in grid if rows[str(s)]['spend_over_start'] >= cs + 2.0 * rows[str(s)]['se_diff_vs_cand']]
    lo = max(lo_c, key=lambda s: (rows[str(s)]['spend_over_start'], -s)) if lo_c else None
    hi = min(hi_c, key=lambda s: (rows[str(s)]['spend_over_start'], s)) if hi_c else None
    flags = (['CONTROL_PATH_MAY_NOT_BRACKET'] if hi is None else []) + (['CONTROL_PATH_NO_LO_POINT'] if lo is None else [])
    return {'origin': 'START (s = 0: SUCC exactly)', 'selection_data': 'CALIBRATION', 'key': key, 'cand': cand, 'path': rows,
            'lo': lo, 'hi': hi, 'flags': flags,
            'rule': 'LO = largest spend <= CAND spend; HI = smallest spend >= CAND spend + 2 SE (SE of the point\'s paired difference '
                    'against CAND); spends are CALIBRATION paired total NLL over START; comparisons on spend values'}


def ctrl_ids(br):
    """Config ids (or aliases) of the two CTRL points."""
    out = {}
    for side, cfg in (('LO', 'FA_TFR_TASK_LO'), ('HI', 'FA_TFR_TASK_HI')):
        s = br[side.lower()]
        if s is None:
            out[side] = {'config': None, 's': None, 'alias': None}
        elif ALIAS_ENDPOINTS and float(s) == 0.0:
            out[side] = {'config': None, 's': 0.0, 'alias': 'START'}
        elif ALIAS_ENDPOINTS and float(s) == 1.0:
            out[side] = {'config': None, 's': 1.0, 'alias': 'FA_TFR_CONT'}
        else:
            out[side] = {'config': cfg, 's': float(s), 'alias': None}
        if s is not None:
            out[side]['spend_over_start'] = br['path'][str(s)]['spend_over_start']
            out[side]['se_diff_vs_cand'] = br['path'][str(s)]['se_diff_vs_cand']
    return out


# ---------------------------------------------------------------- deployment (sar.deploy contract)
def teacher_arrays(P, X, names):
    Zc = FK.compact(X, names)
    succ = _succ(P['version'], X, names)
    from nrm.fa import prc
    return succ, blocks(Zc, (succ['income'], succ['occupation']), (prc._proba(P['hgb_inc'], Zc), prc._proba(P['hgb_occ'], Zc)))


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function: the configuration's vectors from admitted features alone (deterministic). SUCC's frozen files must hash to
    the values recorded at fit time; scale 0 releases START (SUCC's own vectors) exactly."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    if P.get('frozen_sha256'):
        now = frozen_sha(version)
        if now != P['frozen_sha256']:
            raise PermissionError("SUCC's frozen files differ from those recorded at fit time")
    succ, A = teacher_arrays(P, X, names)
    if float(P['scale']) == 0.0:
        return {'income': np.array(succ['income'], np.float64, copy=True), 'occupation': np.array(succ['occupation'], np.float64, copy=True)}
    pi, po = release_arrays(A, net_numpy(P['net'], phi_apply(P['phi'], A)), P['scale'])
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    """This configuration's own model files (MODEL.pkl holds the residual, the feature map and the HGB teacher models); predict()
    also reads SUCC's frozen files through ihs.fam_h.predict (frozen_inputs, hash-checked against the fit-time record)."""
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- the fit stage (one unit per version)
def _cal_arrays(cal, version, T, sfn):
    names = tuple(map(str, cal['feature_names']))
    X = np.asarray(cal['X'], np.float64)
    Zc = FK.compact(X, names)
    s = sfn(version, X, names)
    from nrm.fa import prc
    A = blocks(Zc, (s['income'], s['occupation']), (prc._proba(T['models']['inc'], Zc), prc._proba(T['models']['occ'], Zc)))
    return A, (np.asarray(s['income'], np.float64), np.asarray(s['occupation'], np.float64))


def select_rho(cal_L, path_L, L_start, ucal, mf_oof, mf_oof0, lab, unit, rho_grid=RHO_GRID, budget_cal=None, guard=None):
    """The registered rule. For each rho (increasing): (a) its CALIBRATION paired spend over SUCC meets BUDGET_CAL; (b) its MODEL_FIT
    out-of-fold men's / women's occupation Brier spend over the lambda = 0 path's LO point for that rho (the bracket rule; s = 1 when no
    LO point exists) meets GUARD_DID (estimate + 2 household-clustered SE). rho* = the last rho such that it and every smaller rho
    pass; None if the first fails."""
    budget_cal, guard = budget_cal or BUDGET_CAL, guard or GUARD_DID
    rows, pick = {}, None
    for r in rho_grid:
        sp = FK.spend(cal_L[r], L_start, ucal)
        br = bracket(path_L, cal_L[r], L_start, ucal)
        ref = br['lo'] if br['lo'] is not None else max(S_PATH)
        did = FK.spend(FK.person_losses(*mf_oof[r], lab['income'], lab['occupation']),
                       FK.person_losses(*mf_oof0[ref], lab['income'], lab['occupation']), unit, lab['sex'])
        ok_cal, ok_g = FK.meets(sp, budget_cal), FK.meets(did, guard)
        rows[str(r)] = {'cal_spend_over_SUCC': FK.jsonable_spend(sp), 'guard_reference_s': ref, 'bracket_lo': br['lo'],
                        'bracket_hi': br['hi'], 'bracket_flags': br['flags'],
                        'mf_guard_spend_over_reference': {k: {'estimate': did[k][0], 'se': did[k][1]} for k in guard},
                        'meets_cal': ok_cal, 'meets_guard': ok_g}
        if ok_cal and ok_g:
            pick = r
        else:
            break
    return pick, rows


def stage_fit(version, data, cal, card, save, roles=None, log=print, record_dir=None, epochs=None, rho_grid=RHO_GRID,
              budget_cal=None, guard=None, succ_fn=None, cache=None, frozen=None):
    """data: MODEL_FIT (features, income, occupation, sex); cal: CALIBRATION (features, income, occupation; no SEX)."""
    t0 = time.process_time()
    names = tuple(map(str, data['feature_names']))
    X = np.asarray(data['X'], np.float64)
    unit = np.asarray(data['unit'])
    lab = {k: np.asarray(data[k], np.int64) for k in ('income', 'occupation', 'sex')}
    T, tsha = cache if cache is not None else task_cache(version)
    if T.get('row_id_sha256') is not None and 'row_id' in data:
        from nrm.fa import prc
        if T['row_id_sha256'] != prc.array_sha(np.asarray(data['row_id'])):
            raise PermissionError('the cached task teachers were fitted on other rows')
    frozen = frozen_sha(version) if frozen is None else frozen
    sfn = succ_fn or (lambda v, X_, n_: _succ(v, X_, n_))
    succ = sfn(version, X, names)
    A = blocks(FK.compact(X, names), (succ['income'], succ['occupation']), (T['oof']['inc'], T['oof']['occ']))
    succ_pair = (np.asarray(succ['income'], np.float64), np.asarray(succ['occupation'], np.float64))
    A_cal, succ_cal = _cal_arrays(cal, version, T, sfn)
    ucal = np.asarray(cal['unit'])
    yIc, yOc = np.asarray(cal['income'], np.int64), np.asarray(cal['occupation'], np.int64)
    L_cal_succ = FK.person_losses(*succ_cal, yIc, yOc)
    # 1. household out-of-fold MODEL_FIT releases (equal effort): every rho, and the lambda = 0 path at every scale (one fit per fold)
    oof, xf = FR.crossfit(lambda *a: fit_rho(*a[:6], a[6], epochs=epochs), release_rho, A, lab, unit, version, FAMILY,
                          tuple(rho_grid), use_sex=True, log=lambda s_: log(s_))
    oof0, xf0 = FR.crossfit(None, None, A, lab, unit, version, FAMILY, S_GRID, use_sex=False,          # the candidate's folds
                            shared=(lambda *a: fit_task(*a[:6], epochs=epochs), release_scaled), log=lambda s_: log(s_))
    oof0[0.0] = succ_pair                                                                               # START, exactly
    # 2. full MODEL_FIT fits and their CALIBRATION losses (the residual never sees CALIBRATION)
    full = {r: fit_rho(A, lab['income'], lab['occupation'], lab['sex'], unit, version, r, epochs=epochs) for r in rho_grid}
    full0 = fit_task(A, lab['income'], lab['occupation'], None, unit, version, epochs=epochs)
    cal_L = {r: FK.person_losses(*release_state(full[r], A_cal), yIc, yOc) for r in rho_grid}
    path_L = {s: FK.person_losses(*path_release(full0, A_cal, s, succ_cal), yIc, yOc) for s in S_PATH}
    rho_star, rows = select_rho(cal_L, path_L, L_cal_succ, ucal, oof, oof0, lab, unit, rho_grid, budget_cal, guard)
    mf_sp = FR.path_spends({**oof, 'SUCC': succ_pair}, 'SUCC', lab, unit)
    selection = {'rho_grid': list(rho_grid), 'budget_cal': budget_cal or BUDGET_CAL, 'guard_did': guard or GUARD_DID,
                 'rho_star': rho_star, 'per_rho': rows, 'crossfit': xf, 'task_crossfit': xf0,
                 'mf_spend_over_SUCC_descriptive_biased': {str(r): FK.jsonable_spend(mf_sp[r]) for r in rho_grid},
                 'calibration_exposure': 'occupation side of SUCC, HGB teachers and residual out of sample on CALIBRATION; H3 (SUCC '
                                         'income) and Ucal in-sample there: income-side CALIBRATION spends over SUCC are biased in '
                                         "SUCC's favour (conservative for CAND's selection)",
                 'rule': 'per version: the largest rho such that it and every smaller rho meet (a) estimate + 2 household-clustered SE '
                         'of the CALIBRATION paired spend over SUCC <= BUDGET_CAL and (b) estimate + 2 SE of the MODEL_FIT out-of-fold '
                         "men's / women's occupation Brier spend over the lambda = 0 path's LO point for that rho <= GUARD_DID"}
    log(json.dumps({'rho_star': rho_star, 'cpu_s': round(time.process_time() - t0, 1)}))
    out = {'version': int(version), 'family': FAMILY, 'task_cache_sha256': tsha, 'frozen_sha256': frozen, 'selection': selection,
           'cand': {'config': 'FA_TFR', 'rho_star': rho_star, 'status': 'FITTED' if rho_star is not None else 'NO_FEASIBLE_STRENGTH'},
           'ctrl': None, 'cont': {'config': 'FA_TFR_CONT', 'gating': True, 'scale': 1.0, 'rho': 0.0, 'status': None}}
    roles = roles or FK.role_features()
    people, hh = len(X), len(np.unique(unit))
    exposure = ('MODEL_FIT SEX of all 20,000 MODEL_FIT people reaches the residual through the joint critics; no task label reaches the '
                'residual directly (its task term is distillation toward the admitted HGB teachers, whose MODEL_FIT out-of-fold values '
                'carry MODEL_FIT task labels); MODEL_FIT task labels and SEX reach the guard part of the selection; CALIBRATION income '
                'and occupation labels (never SEX) reach only the task part of the selection and the CTRL bracket, identically for the '
                'controls; through SUCC, the frozen H3 / JOINT_TREE lineage (MODEL_FIT task and SEX labels in-sample; H3 also '
                'in-sample on CALIBRATION); no MODEL_SELECT label')
    base = {'feature_names': names, 'hgb_inc': T['models']['inc'], 'hgb_occ': T['models']['occ'], 'task_cache_sha256': tsha,
            'succ_config': SUCC_CONFIG, 'version': int(version), 'frozen_sha256': frozen}
    saved = {}

    def put(cfg, state, scale, extra, oof_pair, cal_pair, fits):
        payload = {**base, 'net': state['checkpoints'][state['keep']], 'phi': state['phi'], 'scale': float(scale),
                   'train_info': {k: v for k, v in state['info'].items() if k != 'trace'}}
        meta = FK.write_model(FAMILY, cfg, version, payload)
        arrays = FK.release_all(predict, cfg, version, roles)
        spo = FK.spend(FK.person_losses(*oof_pair, lab['income'], lab['occupation']),
                       FK.person_losses(*succ_pair, lab['income'], lab['occupation']), unit, lab['sex'])
        csp = FK.spend(FK.person_losses(*cal_pair, yIc, yOc), L_cal_succ, ucal)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits,
                        'MODEL_FIT: income, occupation, sex; CALIBRATION: income, occupation', exposure, people, hh,
                        time.process_time() - t0, {**extra, 'role': CONFIGS[cfg]['role'], 'gating': bool(CONFIGS[cfg].get('gating')),
                                                   'model_sha256': meta['model_sha256'], 'selection_rho_star': rho_star,
                                                   'cal_spend_over_SUCC': FK.jsonable_spend(csp),
                                                   'mf_oof_spend_over_SUCC_biased': FK.jsonable_spend(spo)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'cal_spend_over_SUCC': FK.jsonable_spend(csp), **extra}

    if rho_star is None:
        log(json.dumps({'stop': 'no rho meets the rule: no candidate and no control is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_V{int(version)}.json', out)
        return out
    cand_cal = release_state(full[rho_star], A_cal)
    br = bracket(path_L, cal_L[rho_star], L_cal_succ, ucal)
    ids = ctrl_ids(br)
    out['cand']['cal_spend_over_SUCC'] = {**br['cand'], 'all': FK.jsonable_spend(FK.spend(cal_L[rho_star], L_cal_succ, ucal))}
    put('FA_TFR', full[rho_star], 1.0, {'op': 'tfr', 'rho': rho_star, 'lambda': full[rho_star]['info']['lambda'], 'scale': 1.0,
                                        'strength': rho_star}, oof[rho_star], cand_cal, (1, xf['n_crossfit_fits'], 0, 1))
    for side in ('LO', 'HI'):
        if ids[side]['config'] is not None:
            s = ids[side]['s']
            put(ids[side]['config'], full0, s, {'op': 'tfr', 'rho': 0.0, 'scale': s, 'strength': s, 'bracket_side': side,
                                                'bracket_flags': br['flags']},
                oof0[s], path_release(full0, A_cal, s, succ_cal), (1, xf0['n_crossfit_fits'], 0, 0))
    put('FA_TFR_CONT', full0, 1.0, {'op': 'tfr', 'rho': 0.0, 'scale': 1.0, 'strength': 1.0}, oof0[1.0],
        path_release(full0, A_cal, 1.0, succ_cal), (0, 0, 0, 0))
    out['ctrl'] = {**{k: br[k] for k in ('origin', 'selection_data', 'key', 'path', 'lo', 'hi', 'flags', 'rule')},
                   'saved': ids, 'alias_endpoints': ALIAS_ENDPOINTS}
    out['cont']['status'] = 'SAVED'
    t1 = time.process_time()
    views = {'SUCC': succ_pair, 'FA_TFR': oof[rho_star], 'FA_TFR_CONT': oof0[1.0]}
    for side in ('LO', 'HI'):
        if ids[side]['config'] is not None:
            views[ids[side]['config']] = oof0[ids[side]['s']]
    check = FK.mf_reader_check(views, lab['sex'], unit, f'SAR_mfcheck_FA_S{int(version)}')
    out.update({'saved': saved,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input; SUCC is in-sample on '
                                            'MODEL_FIT, so its MF readability is optimistic', 'auc': check,
                                    'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks on invented data (no label file, no real teacher)
def _invented(n, seed):
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    Zc = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.6 * sex, rng.integers(0, 8, n), rng.integers(1, 25, n),
                          rng.integers(0, 5, n), rng.choice([1., 12., 36., 72., 110., 303., 999.], n)])
    li = 0.8 * Zc[:, 1] + 0.1 * Zc[:, 3] - 1.0 + 0.4 * sex
    yi = (li + rng.logistic(size=n) > 0).astype(np.int64)
    lo = np.column_stack([np.zeros(n), 0.5 * Zc[:, 0], 0.8 * sex - 0.4, 0.05 * Zc[:, 3], -0.3 * Zc[:, 1]])
    yo = np.array([rng.choice(5, p=np.exp(r) / np.exp(r).sum()) for r in lo], np.int64)
    sI = 1 / (1 + np.exp(-(0.7 * Zc[:, 1] + 0.1 * Zc[:, 3] - 1.0)))          # invented weak-in-SEX "protected start"
    so = np.exp(lo - 0.9 * np.column_stack([np.zeros(n), np.zeros(n), 0.8 * sex - 0.4, np.zeros(n), np.zeros(n)]))
    succ = (np.column_stack([1 - sI, sI]), so / so.sum(1, keepdims=True))
    hI = 1 / (1 + np.exp(-li))                                                # invented strong "task teacher"
    ho = np.exp(lo)
    hgb = (np.column_stack([1 - hI, hI]), ho / ho.sum(1, keepdims=True))
    return Zc, yi, yo, sex, np.arange(n) // 2, succ, hgb


def _bracket_cases():
    """Invented spend tables for the bracket rule (no model): returns {case: ok}."""
    rng = np.random.default_rng(4)
    n = 2000
    unit = np.arange(n) // 2
    base = rng.normal(1.0, 0.3, n)
    start = {MATCH_KEY: base}

    def L(shift, noise=0.0):
        return {MATCH_KEY: base + shift + noise * rng.normal(size=n)}
    shifts = {s: -0.02 * s for s in S_PATH}                       # monotone path: s = 1 spends -0.02
    path = {s: L(shifts[s], 0.01 if s > 0 else 0.0) for s in S_PATH}
    ok = {}
    b = bracket(path, L(-0.010, 0.01), start, unit)              # CAND improves (spend < 0)
    ok['improving_cand_brackets'] = (b['lo'] is not None and b['hi'] is not None and b['flags'] == []
                                     and b['path'][str(b['lo'])]['spend_over_start'] <= b['cand']['spend_over_start']
                                     and b['path'][str(b['hi'])]['spend_over_start'] >= b['cand']['spend_over_start']
                                     + 2 * b['path'][str(b['hi'])]['se_diff_vs_cand'])
    lo_ok = all(not (b['path'][str(s)]['spend_over_start'] <= b['cand']['spend_over_start']
                     and b['path'][str(s)]['spend_over_start'] > b['path'][str(b['lo'])]['spend_over_start']) for s in S_PATH)
    ok['lo_is_largest_below'] = lo_ok
    b2 = bracket(path, L(+0.005, 0.01), start, unit)             # CAND spends more than START: LO = START, no HI
    ok['spending_cand_flags'] = b2['lo'] == 0.0 and b2['hi'] is None and 'CONTROL_PATH_MAY_NOT_BRACKET' in b2['flags']
    b3 = bracket(path, L(-0.05, 0.01), start, unit)              # CAND better than the whole path: no LO
    ok['no_lo_flag'] = b3['lo'] is None and 'CONTROL_PATH_NO_LO_POINT' in b3['flags']
    shuffled = dict(zip(S_PATH, [path[s] for s in reversed(S_PATH)]))   # non-monotone in s: the rule must use spend values only
    b4 = bracket(shuffled, L(-0.010, 0.01), start, unit)
    ok['uses_spend_not_s'] = (b4['lo'] is not None and b4['path'][str(b4['lo'])]['spend_over_start'] ==
                              b['path'][str(b['lo'])]['spend_over_start'])
    ids = ctrl_ids(b2)
    ok['start_alias'] = ids['LO']['alias'] == 'START' and ids['LO']['config'] is None and ids['HI']['config'] is None
    return ok


def selftest(n=1600, epochs=3):
    import tempfile
    from pathlib import Path as _P
    import torch
    t0 = time.process_time()
    Zc, yi, yo, sex, unit, succ, hgb = _invented(n, 11)
    A = blocks(Zc, succ, hgb)
    ok = {}
    st0 = train(A, None, 0.0, 'selftest-task', epochs=epochs)
    pi0, po0 = release_state(st0, A, 1.0, label=0)
    ok['theta0_is_start'] = bool(np.max(np.abs(pi0 - succ[0])) < 1e-10 and np.max(np.abs(po0 - succ[1])) < 1e-10)
    z0 = path_release(st0, A, 0.0, succ)
    ok['s0_is_start_exactly'] = bool(np.array_equal(z0[0], succ[0]) and np.array_equal(z0[1], succ[1]))
    W = st0['checkpoints'][st0['keep']]
    F = phi_apply(st0['phi'], A)
    m = _net(F.shape[1], 0)
    with torch.no_grad():
        for p, w in zip(m.parameters(), W):
            p.copy_(torch.from_numpy(w))
        tout = m(torch.from_numpy(F)).numpy()
    ok['numpy_forward_equals_torch'] = bool(np.max(np.abs(tout - net_numpy(W, F))) < 1e-10)
    st1 = train(A, sex, 3.0, 'selftest-crit', epochs=epochs)
    st1b = train(A, sex, 3.0, 'selftest-crit', epochs=epochs)
    ok['deterministic'] = all(np.array_equal(a, b) for a, b in zip(st1['checkpoints'][st1['keep']], st1b['checkpoints'][st1b['keep']]))
    ok['lambda_positive'] = st1['info']['lambda'] > 0
    pi1, po1 = release_state(st1, A)
    ok['valid_vectors'] = bool(np.allclose(pi1.sum(1), 1) and np.allclose(po1.sum(1), 1) and (pi1 > 0).all() and (po1 > 0).all())
    kl = lambda P, Q: float(np.mean(np.sum(P * (np.log(P) - np.log(Q)), 1)))
    pt = release_state(st0, A)
    ok['task_term_moves_toward_teacher'] = (kl(hgb[1], pt[1]) < kl(hgb[1], succ[1])) and (kl(hgb[0], pt[0]) < kl(hgb[0], succ[0]))
    ok.update({f'bracket_{k}': v for k, v in _bracket_cases().items()})
    # the rule stops at the first failing rho
    lab = {'income': yi, 'occupation': yo, 'sex': sex}
    L_s = FK.person_losses(*succ, yi, yo)
    path_L = {s: FK.person_losses(*path_release(st0, A, s, succ), yi, yo) for s in S_PATH}
    cal_L = {1.0: FK.person_losses(*release_state(st1, A), yi, yo), 3.0: FK.person_losses(*release_state(st1, A), yi, yo)}
    oo = {r: release_state(st1, A) for r in (1.0, 3.0)}
    o0 = {s: path_release(st0, A, s, succ) for s in S_PATH}
    big = {k: 100.0 for k in GUARD_DID}
    pk, _ = select_rho(cal_L, path_L, L_s, unit, oo, o0, lab, unit, (1.0, 3.0), {k: 100.0 for k in BUDGET_CAL}, big)
    cal_L2 = {1.0: cal_L[1.0], 3.0: {k: v + 10.0 for k, v in cal_L[3.0].items()}}
    pk2, _ = select_rho(cal_L2, path_L, L_s, unit, oo, o0, lab, unit, (1.0, 3.0), {'occ_nll': 0.5}, big)
    ok['rule_monotone_stop'] = pk == 3.0 and pk2 == 1.0
    # end-to-end stage on invented data: fake teachers, temporary model directory, fake save
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    X = np.column_stack([Zc, np.zeros((n, 2))])
    names = tuple(f'f{i}' for i in range(X.shape[1]))
    data = {'X': X, 'feature_names': names, 'unit': unit, 'income': yi, 'occupation': yo, 'sex': sex}
    Zc2, yi2, yo2, _, unit2, succ2, _ = _invented(600, 12)
    X2 = np.column_stack([Zc2, np.zeros((600, 2))])
    cal = {'X': X2, 'feature_names': names, 'unit': unit2, 'income': yi2, 'occupation': yo2}
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from nrm.fa import prc
    mi = HGB(max_iter=10).fit(Zc, yi); mo = HGB(max_iter=10).fit(Zc, yo)
    T = {'models': {'inc': mi, 'occ': mo}, 'oof': {'inc': prc._proba(mi, Zc), 'occ': prc._proba(mo, Zc)}}
    table = {X.tobytes(): succ, X2.tobytes(): succ2}
    old = (C.RUNS, FK.compact, globals()['_succ'])
    tmp = _P(tempfile.mkdtemp(prefix='sar_fa_tfr_selftest_'))
    try:
        C.RUNS = tmp
        FK.compact = lambda X_, n_: np.asarray(X_, np.float64)[:, :6]
        globals()['_succ'] = lambda v, X_, n_: dict(zip(('income', 'occupation'), table[np.asarray(X_, np.float64).tobytes()]))
        res = stage_fit(0, data, cal, 'invented-card', fake_save, roles={'MODEL_FIT': {'X': X, 'feature_names': names}},
                        log=lambda s_: None, record_dir=False, epochs=epochs, rho_grid=(1.0, 3.0),
                        budget_cal={k: 100.0 for k in BUDGET_CAL}, guard=big, cache=(T, 'invented'), frozen={})
        again = predict('FA_TFR', 0, X, names)
        ok['deploy_bitwise'] = all(np.array_equal(again[t], saved['FA_TFR'][0]['MODEL_FIT'][t]) for t in ('income', 'occupation'))
        ok['model_paths'] = all(p.is_file() for p in model_paths('FA_TFR', 0))
        try:
            predict('FA_TFR', 0, X, names, keys=[1]); ok['refuses_keys'] = False
        except PermissionError:
            ok['refuses_keys'] = True
        P0 = FK.load_model(FAMILY, 'FA_TFR', 0)
        FK.write_model(FAMILY, 'SELFTEST_S0', 0, {k: v for k, v in P0.items() if k not in ('schema', 'family', 'config_id', 'version')}
                       | {'scale': 0.0, 'version': 0})
        z = predict('SELFTEST_S0', 0, X, names)
        ok['scale0_predict_is_start_exactly'] = bool(np.array_equal(z['income'], succ[0]) and np.array_equal(z['occupation'], succ[1]))
        FK.write_model(FAMILY, 'SELFTEST_PIN', 0, {k: v for k, v in P0.items() if k not in ('schema', 'family', 'config_id', 'version')}
                       | {'frozen_sha256': {'x': 'y'}, 'version': 0})
        try:
            globals()['frozen_sha'], old_fs = (lambda v: {'x': 'z'}), globals()['frozen_sha']
            predict('SELFTEST_PIN', 0, X, names); ok['refuses_changed_frozen_files'] = False
        except PermissionError:
            ok['refuses_changed_frozen_files'] = True
        finally:
            globals()['frozen_sha'] = old_fs
    finally:
        C.RUNS, FK.compact = old[0], old[1]
        globals()['_succ'] = old[2]
    ids = res['ctrl']['saved']
    want = {'FA_TFR', 'FA_TFR_CONT'} | {ids[s]['config'] for s in ('LO', 'HI') if ids[s]['config']}
    ok['stage_saved_declared'] = set(saved) == want and len(saved) <= 4
    ok['record_fields'] = (res['cand']['status'] == 'FITTED' and 'tot_nll' not in res['cand']['cal_spend_over_SUCC']
                           and {'spend_over_start', 'se'} <= set(res['cand']['cal_spend_over_SUCC'])
                           and res['ctrl']['selection_data'] == 'CALIBRATION' and res['ctrl']['origin'].startswith('START')
                           and all({'spend_over_start', 'se_diff_vs_cand'} <= set(r) for r in res['ctrl']['path'].values())
                           and res['cont']['gating'] is True and set(res['selection']['per_rho']) == {'1.0', '3.0'})
    ok['rho_star_largest_feasible'] = res['selection']['rho_star'] == 3.0
    ok['records_kinds'] = all(r['kind'] == CONFIGS[c]['kind'] and r['role'] == CONFIGS[c]['role'] for c, (_, r) in saved.items())
    ok['cont_gating_task_only'] = saved['FA_TFR_CONT'][1]['gating'] is True and saved['FA_TFR_CONT'][1]['rho'] == 0.0
    ok['all_ok'] = all(ok.values())
    ok['cpu_s'] = round(time.process_time() - t0, 1)
    ok['train_cpu_s_per_epoch_with_critics'] = round(st1['info']['cpu_s'] / epochs, 2)
    ok['ctrl_points_in_stage_test'] = {s: ids[s] for s in ('LO', 'HI')}
    ok['ctrl_flags_in_stage_test'] = res['ctrl']['flags']
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fa.tfr')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    p = sub.add_parser('preflight')
    p.add_argument('--version', type=int, required=True)
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    if a.cmd == 'preflight':
        r = preflight(a.version)
        print(json.dumps(r, indent=1, default=str))
        return 0 if r['ok'] else 1
    pf = preflight(a.version)                                   # label-free: refuse before any label read if a pin is missing
    if not pf['ok']:
        raise PermissionError(f'frozen inputs not pinned / changed: missing {pf["missing"]}, mismatched {pf["mismatched"]}')
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    cal = C.load('CALIBRATION', ('features', 'income', 'occupation'), 'calibrate', PURPOSE_CAL, card=a.card)
    out = stage_fit(a.version, data, cal, a.card, RL.save)
    print(json.dumps({'rho_star': out['selection']['rho_star'], 'saved': sorted(out['saved']), 'ctrl_flags': (out['ctrl'] or {}).get('flags'),
                      'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

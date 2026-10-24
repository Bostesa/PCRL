"""FA revision R1: SUCC-anchored teacher-feature residual on the OCCUPATION output only (TFRO; card draft
roles/investigator_a/CARD_DRAFT_R1.md, to be committed as cards/FA_TFRO_R1.md). PCRL campaign `pcrl_strong_anchor_release_v1`;
EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.
REVISED_ (designed from the FA_TFR_v1 MODEL_FIT / CALIBRATION fit records only; no MODEL_SELECT read for FA).

Diagnosis (results/pcrl_strong_anchor_release_v1/fa/FIT_V0.json, FIT_V1.json; rho = 0.3, both versions NO_FEASIBLE_STRENGTH):
  D1  part (a) failed on components whose CALIBRATION estimates are not usable at the v1 budgets: income Brier (+0.0007 / +0.0019,
      se 0.0018: bound 0.0043 / 0.0056 > 0.0030) and, in v1, income NLL and occupation accuracy (bound 0.0056 > 0.0050), while every
      point estimate was favourable or near zero (occupation NLL -0.016 / -0.014, income accuracy -0.008 / -0.005). The income-side
      CALIBRATION spends are biased in SUCC's favour (H3 is in-sample on CALIBRATION), and the budgets (0.4-0.6 of SUCC's margins) sat
      inside CALIBRATION's 2-SE noise (2 SE of income Brier 0.0036 > 0.0030; of occupation accuracy about 0.011 > 0.0050).
  D2  part (b) is not a valid estimator of the cap-relevant quantity: on CALIBRATION (occupation side out of sample) CAND improves the
      occupation summed Brier over SUCC by -0.0063 / -0.0053 (resolved), while on MODEL_FIT (SUCC in-sample) the same CAND reads
      +0.0148 / +0.0165 worse than SUCC (descriptive): a sign flip of about 0.02, SUCC's in-sample offset. The MODEL_FIT difference
      against the lambda = 0 LO point (+0.0073 / +0.0144 men, +0.0139 / +0.0110 women, se about 0.0006) is of the same order as that
      offset and cannot be separated from it on MODEL_FIT: two releases that depart from SUCC differently forfeit different amounts of
      SUCC's in-sample fit, so the first-order cancellation the v1 rule assumed does not hold (LO's CALIBRATION occupation Brier was
      not recorded; its total NLL is at least as good as CAND's by construction).
  D3  within one release, the MODEL_FIT SEX contrast is far less affected: the lambda = 0 LO point's men-minus-women Brier spend over
      SUCC on MODEL_FIT is -0.0005 / +0.0007 (both groups about +0.004, the shared in-sample term), whereas CAND's is -0.0072 / +0.0042
      at rho = 0.3 and the descriptive path turns men-heavy with rho (+0.0042 / +0.0061 at rho 3; +0.0156 / +0.0201 at rho 10).
Mechanism change (structural): the residual may move the OCCUPATION output only; the income recipient receives SUCC's H3 vector
bitwise. Every component that CALIBRATION cannot measure without SUCC's in-sample bias (income) is removed from the operation, so
the task part of the rule is out of sample for every fitted layer it measures (JOINT_TREE, the HGB teachers, the residual).
  release   income = SUCC's income vector (bitwise); occupation = softmax(centre(z_S + s r_O(phi))); r_O: 29 -> 64 -> 64 -> 5, zero-
            initialized output layer; s = 1 for CAND, s = 0 is START exactly
  training  as FA_TFR_v1 (distillation of the occupation output toward the HGB teacher's out-of-fold probabilities + lambda x joint
            critic risk; the income-view risk is constant because income is frozen); same critics, calibration, optimizer, schedule
Rule change (prospective; budgets are SUCC's own MODEL_SELECT margins to the caps, read from the committed anchor table):
  (a) CALIBRATION paired occupation spends over SUCC (NLL, summed Brier, accuracy loss): estimate + 2 household-clustered SE <= SUCC's
      margin to the cap (cap - the larger of its U0 / Ucal clause values) per version; the income spends are exactly 0
  (b) the guard change Delta_g of each SEX group is estimated as Delta_g = L_CAL + (share of the other group) x (+/-) C_MF, where
      L_CAL = the CALIBRATION paired occupation Brier spend over SUCC (the population-average change; out of sample) and C_MF = the
      MODEL_FIT household out-of-fold men-minus-women occupation Brier spend contrast of the same release over SUCC (SEX only exists
      on MODEL_FIT; SUCC's in-sample term enters both groups and cancels to the extent it is SEX-symmetric, D3). Then
      Delta_men = L + (1 - p_men) C and Delta_women = L - p_men C, with SE^2 = SE_L^2 + weight^2 SE_C^2 (disjoint samples);
      estimate + 2 SE <= SUCC's men's / women's guard margin (0.005 - SUCC's guard value) per version
  rho* = the largest rho on RHO_GRID such that it and every smaller rho pass (a) and (b); rho values are fitted in increasing order and
  the scan stops at the first failure
Controls under the frozen SAR1 rules (identical exposure; the same network restricted to occupation, lambda = 0):
  CTRL  FA_TFRO_R1_TASK_LO / _HI: the two-point bracket on CALIBRATION paired total-NLL spend over SUCC (origin START, s = 0 exactly;
        sar.fa.tfr.bracket unchanged); endpoint aliases START / FA_TFRO_R1_CONT not saved twice
  CONT  FA_TFRO_R1_CONT: lambda = 0 at s = 1 (H3 income + the distilled teacher occupation, about W); gating
Specifications: CAND 2 + CTRL 4 + CONT 2 = 8 at most.

Commands:
    <repository>/.venv/bin/python -m sar.fa.tfro selftest
    <repository>/.venv/bin/python -m sar.fa.tfro preflight --version V
    <repository>/.venv/bin/python -m sar.fa.tfro fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FA_TFRO_R1.md
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
from . import tfr as V1                      # FA_TFR_v1 helpers reused unchanged (teachers, features, pins, bracket)

FAMILY = 'FA'
REVISION = 'R1'
PREFIX = 'FA_TFRO_R1'
RHO_GRID = (0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0)
S_PATH = V1.S_PATH
ALIAS_ENDPOINTS = True
RECIPE = dict(V1.RECIPE)                     # identical training recipe (frozen copy)
CONFIGS = {
    PREFIX: {'kind': 'candidate', 'role': 'CAND',
             'what': "occupation-only SUCC-anchored teacher-feature residual: SUCC's H3 income bitwise; the occupation output corrected "
                     "from SUCC's JOINT logits by a network on the compact fields and the SUCC / HGB-teacher logits, trained by "
                     'distillation toward the HGB occupation teacher with the joint SEX critics at the selected rho*'},
    PREFIX + '_TASK_LO': {'kind': 'control', 'role': 'CTRL',
                          'what': 'CTRL lower bracket point: the occupation-only lambda = 0 residual at the path point with the largest '
                                  "CALIBRATION paired total-NLL spend over SUCC that is <= CAND's (income bitwise SUCC)"},
    PREFIX + '_TASK_HI': {'kind': 'control', 'role': 'CTRL',
                          'what': 'CTRL upper bracket point: the occupation-only lambda = 0 residual at the path point with the smallest '
                                  "CALIBRATION paired total-NLL spend over SUCC that is >= CAND's + 2 SE (income bitwise SUCC)"},
    PREFIX + '_CONT': {'kind': 'control', 'role': 'CONT', 'gating': True,
                       'what': 'zero-protection continuation (gating): the occupation-only lambda = 0 residual at s = 1 (income bitwise '
                               'SUCC)'},
}
PURPOSE = ('FA TFRO R1: occupation-only SUCC-anchored residual with the joint SEX critics (MODEL_FIT SEX), the SEX-contrast part of '
           'its guard estimate and its lambda = 0 control path on MODEL_FIT')
PURPOSE_CAL = ('FA TFRO R1: CALIBRATION occupation (and income, unchanged by construction) for the task part of the strength '
               'selection, the population-average part of the guard estimate and the CTRL bracket (paired spends over SUCC; no fit '
               'on CALIBRATION; identical exposure for the controls)')


# ---------------------------------------------------------------- budgets: SUCC's own MODEL_SELECT margins to the caps (committed)
def succ_margins(version, table=None):
    """Per-version margins of SUCC to the original caps, from the committed anchor table (results/<campaign>/anchors/ANCHOR_TABLE.json):
    occupation NLL / summed Brier / accuracy loss: cap - max(U0 clause value, Ucal clause value); guards: cap - SUCC's guard value."""
    T = table or json.loads((C.RESULTS / 'anchors' / 'ANCHOR_TABLE.json').read_text())
    rows = T['rows']
    if isinstance(rows, dict):
        s = rows['SUCC']
    else:
        s = next(r for r in rows if 'SUCC' in (r.get('release'), r.get('name'), r.get('id')))
    u = s.get('utility', s)[str(int(version))]
    cl = u['clauses']
    out = {}
    for key, name in (('occ_nll', 'nll_excess'), ('occ_brier', 'brier_excess'), ('occ_acc', 'accuracy_loss')):
        vals = [cl[f'occupation|{ref}|{name}'] for ref in ('U0', 'Ucal')]
        out[key] = float(vals[0]['cap']) - max(float(v['value']) for v in vals)
    for g, k in (('sex0', 'men'), ('sex1', 'women')):
        out[k] = float(u['guards'][g]['cap']) - float(u['guards'][g]['value'])
    return out


# ---------------------------------------------------------------- the occupation-only residual
def _net(d, seed):
    import torch
    torch.manual_seed(int(seed) % 2 ** 31)
    h1, h2 = RECIPE['hidden']
    m = torch.nn.Sequential(torch.nn.Linear(d, h1, dtype=torch.float64), torch.nn.ReLU(),
                            torch.nn.Linear(h1, h2, dtype=torch.float64), torch.nn.ReLU(),
                            torch.nn.Linear(h2, 5, dtype=torch.float64))
    with torch.no_grad():
        m[4].weight.zero_(); m[4].bias.zero_()           # theta = 0 reproduces SUCC
    return m


def release_occ(A, out, s):
    from acs_replication import trees
    return trees.probabilities('occupation', A[:, 1:6] + float(s) * out)


def release_state(state, A, succ_inc, s=1.0, start_occ=None):
    """(income, occupation) of a state at scale s: income is SUCC's vector bitwise; s = 0 returns START's occupation exactly."""
    inc = np.array(succ_inc, np.float64, copy=True)
    if float(s) == 0.0:
        if start_occ is None:
            raise ValueError('s = 0 needs START\'s occupation vectors')
        return inc, np.array(start_occ, np.float64, copy=True)
    W = state['checkpoints'][state['keep']]
    return inc, release_occ(A, V1.net_numpy(W, V1.phi_apply(state['phi'], A)), s)


def train(A, sex, rho, seed_key, epochs=None, log=None):
    """One training run (labels never read: the task term is the HGB occupation teacher's probabilities in A). Income is frozen at
    SUCC's log-odds (column 0 of A); rho = 0: no critics."""
    import random
    import statistics
    import torch
    import torch.nn.functional as F
    from acs_replication import leafcorr as lc
    from acs_replication import trees
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    t0 = time.process_time()
    E = int(epochs or RECIPE['epochs'])
    ph = V1.phi_fit(A)
    Xf = torch.from_numpy(np.ascontiguousarray(V1.phi_apply(ph, A)))
    dS = torch.from_numpy(np.ascontiguousarray(A[:, 0]))
    zS = torch.from_numpy(np.ascontiguousarray(A[:, 1:6]))
    TO = torch.from_numpy(np.ascontiguousarray(trees.probabilities('occupation', A[:, 7:12])))
    n = len(A)
    seed = lc.seed_for('SAR_FA_TFRO_R1', seed_key) % 2 ** 31
    net = _net(Xf.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])

    def occ_logits(rows):
        z = zS[rows] + net(Xf[rows])
        return z - z.mean(1, keepdim=True)

    def wires(rows):
        d = dS[rows]
        pi = torch.softmax(torch.stack((-d / 2, d / 2), 1), 1)
        po = torch.softmax(occ_logits(rows), 1)
        return [torch.cat((p, p.argmax(1, keepdim=True).to(p.dtype).detach()), 1) for p in (pi, po)]

    def task_loss(rows):
        return -(TO[rows] * F.log_softmax(occ_logits(rows), 1)).sum(1).mean()

    ck = {0: V1.net_state(net)}
    lam, critics, info = 0.0, None, {'rho': float(rho)}
    priv_rng = random.Random(lc.seed_for('SAR_FA_TFRO_priv', seed_key))
    crit_rng = random.Random(lc.seed_for('SAR_FA_TFRO_crit', seed_key))
    task_rng = np.random.default_rng(lc.seed_for('SAR_FA_TFRO_task', seed_key) % 2 ** 32)
    if rho > 0:
        S = torch.from_numpy(np.asarray(sex, np.int64))
        ref = torch.from_numpy(np.sort(np.random.default_rng(lc.seed_for('SAR_FA_TFRO_ref', seed_key) % 2 ** 32)
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
        cal_rng = random.Random(lc.seed_for('SAR_FA_TFRO_cal', seed_key))
        cal_task = np.random.default_rng(lc.seed_for('SAR_FA_TFRO_caltask', seed_key) % 2 ** 32)
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
            ck[ep] = V1.net_state(net)
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


# ---------------------------------------------------------------- the guard estimate (prospective rule part (b))
def guard_estimate(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit):
    """Delta_men = L + (1 - p) C, Delta_women = L - p C (p = MODEL_FIT share of men): L = CALIBRATION paired occupation Brier spend over
    SUCC (population average, out of sample), C = MODEL_FIT out-of-fold men-minus-women occupation Brier spend contrast over SUCC
    (household-clustered, linearized); SE^2 = SE_L^2 + weight^2 SE_C^2 (CALIBRATION and MODEL_FIT are disjoint people)."""
    dl = np.asarray(cal_c['occ_brier']) - np.asarray(cal_s['occ_brier'])
    L, se_L = float(dl.mean()), FK.cluster_se(dl, ucal)
    d = np.asarray(mf_c['occ_brier']) - np.asarray(mf_s['occ_brier'])
    men = np.asarray(sex) == 0
    p = float(men.mean())
    mm, mw = float(d[men].mean()), float(d[~men].mean())
    phi = men / p * (d - mm) - (~men) / (1 - p) * (d - mw)
    Cc, se_C = mm - mw, FK.cluster_se(phi, unit)
    out = {'L_cal': L, 'se_L': se_L, 'C_mf': Cc, 'se_C': se_C, 'p_men': p, 'mf_men': mm, 'mf_women': mw}
    out['men'] = (L + (1 - p) * Cc, float(np.sqrt(se_L ** 2 + ((1 - p) * se_C) ** 2)))
    out['women'] = (L - p * Cc, float(np.sqrt(se_L ** 2 + (p * se_C) ** 2)))
    return out


def evaluate_rho(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit, margins):
    """Rule parts (a) and (b) for one rho; returns (passes, row)."""
    sp = FK.spend(cal_c, cal_s, ucal)
    if abs(sp['inc_nll'][0]) > 0 or abs(sp['inc_brier'][0]) > 0 or abs(sp['inc_acc'][0]) > 0:
        raise AssertionError('the income output must be SUCC\'s bitwise')
    a = {k: sp[k][0] + 2.0 * sp[k][1] <= margins[k] for k in ('occ_nll', 'occ_brier', 'occ_acc')}
    g = guard_estimate(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit)
    b = {k: g[k][0] + 2.0 * g[k][1] <= margins[k] for k in ('men', 'women')}
    row = {'cal_spend_over_SUCC': FK.jsonable_spend(sp), 'part_a': a, 'guard_estimate': {k: (list(v) if isinstance(v, tuple) else v)
                                                                                         for k, v in g.items()},
           'part_b': b, 'margins': margins}
    return all(a.values()) and all(b.values()), row


# ---------------------------------------------------------------- deployment (sar.deploy contract, routed by sar.fa)
def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function: income = SUCC's vector bitwise; occupation = the residual on SUCC's JOINT logits (scale 0: START exactly);
    SUCC's frozen files must hash to the values recorded at fit time."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    if P.get('frozen_sha256') and V1.frozen_sha(version) != P['frozen_sha256']:
        raise PermissionError("SUCC's frozen files differ from those recorded at fit time")
    succ, A = V1.teacher_arrays(P, X, names)
    pi, po = release_state({'checkpoints': {0: P['net']}, 'keep': 0, 'phi': P['phi']}, A, succ['income'], P['scale'], succ['occupation'])
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- the fit stage (one unit per version)
def _key(version, unit, tag):
    return f'R1|V{int(version)}|{tag}|{int(len(unit))}|{int(np.asarray(unit).sum() % 1000003)}'


def stage_fit(version, data, cal, card, save, roles=None, log=print, record_dir=None, epochs=None, rho_grid=RHO_GRID, margins=None,
              succ_fn=None, cache=None, frozen=None):
    t0 = time.process_time()
    names = tuple(map(str, data['feature_names']))
    X = np.asarray(data['X'], np.float64)
    unit = np.asarray(data['unit'])
    lab = {k: np.asarray(data[k], np.int64) for k in ('income', 'occupation', 'sex')}
    T, tsha = cache if cache is not None else V1.task_cache(version)
    if T.get('row_id_sha256') is not None and 'row_id' in data:
        from nrm.fa import prc
        if T['row_id_sha256'] != prc.array_sha(np.asarray(data['row_id'])):
            raise PermissionError('the cached task teachers were fitted on other rows')
    margins = margins or succ_margins(version)
    frozen = V1.frozen_sha(version) if frozen is None else frozen
    sfn = succ_fn or (lambda v, X_, n_: V1._succ(v, X_, n_))
    succ = sfn(version, X, names)
    A = V1.blocks(FK.compact(X, names), (succ['income'], succ['occupation']), (T['oof']['inc'], T['oof']['occ']))
    succ_pair = (np.asarray(succ['income'], np.float64), np.asarray(succ['occupation'], np.float64))
    A_cal, succ_cal = V1._cal_arrays(cal, version, T, sfn)
    ucal = np.asarray(cal['unit'])
    yIc, yOc = np.asarray(cal['income'], np.int64), np.asarray(cal['occupation'], np.int64)
    L_cal_s = FK.person_losses(*succ_cal, yIc, yOc)
    L_mf_s = FK.person_losses(*succ_pair, lab['income'], lab['occupation'])
    folds = FR.folds(unit, FAMILY, version)                       # the v1 folds (SAR_xf_FA_S<v>)
    n = len(unit)

    def oof_fit(rho):
        """Household out-of-fold occupation (one fit per fold); the full fit; CPU."""
        occ = np.zeros((n, 5))
        for j in range(FR.N_XF):
            tr, te = folds != j, folds == j
            st = train(A[tr], None if rho == 0 else lab['sex'][tr], rho, _key(version, unit[tr], f'rho{rho:g}|f{j}'), epochs=epochs)
            if rho == 0:
                occ_s = {s: release_state(st, A[te], succ_pair[0][te], s, succ_pair[1][te])[1] for s in S_PATH}
                if j == 0:
                    paths = {s: np.zeros((n, 5)) for s in S_PATH}
                for s in S_PATH:
                    paths[s][te] = occ_s[s]
            else:
                occ[te] = release_state(st, A[te], succ_pair[0][te])[1]
        full = train(A, None if rho == 0 else lab['sex'], rho, _key(version, unit, f'rho{rho:g}|full'), epochs=epochs)
        return (paths if rho == 0 else occ), full

    # 1. rho values in increasing order; the scan stops at the first failure
    rows, rho_star, cand = {}, None, {}
    for r in rho_grid:
        occ_oof, full = oof_fit(r)
        cal_c = FK.person_losses(*release_state(full, A_cal, succ_cal[0]), yIc, yOc)
        mf_c = FK.person_losses(succ_pair[0], occ_oof, lab['income'], lab['occupation'])
        ok, row = evaluate_rho(cal_c, L_cal_s, ucal, mf_c, L_mf_s, lab['sex'], unit, margins)
        row.update({'lambda': full['info']['lambda'], 'mf_oof_spend_over_SUCC_biased': FK.jsonable_spend(FK.spend(mf_c, L_mf_s, unit, lab['sex'])),
                    'passes': ok})
        rows[str(r)] = row
        log(json.dumps({'rho': r, 'passes': ok, 'part_a': row['part_a'], 'part_b': row['part_b'],
                        'cpu_s': round(time.process_time() - t0, 1)}))
        if not ok:
            break
        rho_star, cand = r, {'full': full, 'occ_oof': occ_oof, 'cal_L': cal_c}
    selection = {'rho_grid': list(rho_grid), 'rho_star': rho_star, 'per_rho': rows, 'margins': margins,
                 'rule': 'per version: rho values in increasing order; the largest rho such that it and every smaller rho pass (a) '
                         'CALIBRATION paired occupation spends over SUCC (NLL, summed Brier, accuracy loss) estimate + 2 SE <= SUCC\'s '
                         'margins to the caps and (b) the guard estimate Delta_g = L_CAL +/- weight x C_MF, estimate + 2 SE <= SUCC\'s '
                         'guard margins; the scan stops at the first failing rho',
                 'calibration_exposure': 'income is SUCC\'s bitwise (income spends exactly 0); the occupation side of SUCC (JOINT_TREE), '
                                         'the HGB teachers and the residual are out of sample on CALIBRATION'}
    out = {'version': int(version), 'family': FAMILY, 'revision': REVISION, 'task_cache_sha256': tsha, 'frozen_sha256': frozen,
           'selection': selection,
           'cand': {'config': PREFIX, 'rho_star': rho_star, 'status': 'FITTED' if rho_star is not None else 'NO_FEASIBLE_STRENGTH'},
           'ctrl': None, 'cont': {'config': PREFIX + '_CONT', 'gating': True, 'scale': 1.0, 'rho': 0.0, 'status': None}}
    roles = roles or FK.role_features()
    people, hh = len(X), len(np.unique(unit))
    exposure = ('MODEL_FIT SEX of all 20,000 MODEL_FIT people reaches the occupation residual through the joint critics and the SEX '
                'contrast of the guard estimate; no task label reaches the residual (distillation toward the admitted HGB occupation '
                'teacher, whose MODEL_FIT out-of-fold values carry MODEL_FIT occupation labels); MODEL_FIT task labels reach the guard '
                'contrast; CALIBRATION income and occupation labels (never SEX) reach only the task part of the selection, the '
                'population-average part of the guard estimate and the CTRL bracket, identically for the controls; the income output is '
                'SUCC\'s H3 vector bitwise; no MODEL_SELECT label')
    base = {'feature_names': names, 'hgb_inc': T['models']['inc'], 'hgb_occ': T['models']['occ'], 'task_cache_sha256': tsha,
            'succ_config': V1.SUCC_CONFIG, 'version': int(version), 'frozen_sha256': frozen, 'revision': REVISION, 'outputs': 'occupation'}
    saved = {}

    def put(cfg, state, scale, extra, mf_occ, cal_L, fits):
        payload = {**base, 'net': state['checkpoints'][state['keep']], 'phi': state['phi'], 'scale': float(scale),
                   'train_info': {k: v for k, v in state['info'].items() if k != 'trace'}}
        meta = FK.write_model(FAMILY, cfg, version, payload)
        arrays = FK.release_all(predict, cfg, version, roles)
        spo = FK.spend(FK.person_losses(succ_pair[0], mf_occ, lab['income'], lab['occupation']), L_mf_s, unit, lab['sex'])
        csp = FK.spend(cal_L, L_cal_s, ucal)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits,
                        'MODEL_FIT: income, occupation, sex; CALIBRATION: income, occupation', exposure, people, hh,
                        time.process_time() - t0, {**extra, 'revision': REVISION, 'role': CONFIGS[cfg]['role'],
                                                   'gating': bool(CONFIGS[cfg].get('gating')), 'model_sha256': meta['model_sha256'],
                                                   'selection_rho_star': rho_star, 'cal_spend_over_SUCC': FK.jsonable_spend(csp),
                                                   'mf_oof_spend_over_SUCC_biased': FK.jsonable_spend(spo)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'cal_spend_over_SUCC': FK.jsonable_spend(csp), **extra}

    if rho_star is None:
        log(json.dumps({'stop': 'rho = 0.3 fails: no candidate and no control is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_R1_V{int(version)}.json', out)
        return out
    # 2. the lambda = 0 path (CTRL / CONT): one fit per fold serves every scale; START at s = 0 exactly
    paths, full0 = oof_fit(0.0)
    path_L = {s: FK.person_losses(*release_state(full0, A_cal, succ_cal[0], s, succ_cal[1]), yIc, yOc) for s in S_PATH}
    br = V1.bracket(path_L, cand['cal_L'], L_cal_s, ucal)
    ids = {}
    for side in ('LO', 'HI'):
        s = br[side.lower()]
        if s is None:
            ids[side] = {'config': None, 's': None, 'alias': None}
        elif ALIAS_ENDPOINTS and float(s) == 0.0:
            ids[side] = {'config': None, 's': 0.0, 'alias': 'START'}
        elif ALIAS_ENDPOINTS and float(s) == 1.0:
            ids[side] = {'config': None, 's': 1.0, 'alias': PREFIX + '_CONT'}
        else:
            ids[side] = {'config': f'{PREFIX}_TASK_{side}', 's': float(s), 'alias': None}
        if s is not None:
            ids[side].update({k: br['path'][str(s)][k] for k in ('spend_over_start', 'se_diff_vs_cand')})
    out['cand']['cal_spend_over_SUCC'] = {**br['cand'], 'all': FK.jsonable_spend(FK.spend(cand['cal_L'], L_cal_s, ucal))}
    put(PREFIX, cand['full'], 1.0, {'op': 'tfro', 'rho': rho_star, 'lambda': cand['full']['info']['lambda'], 'scale': 1.0,
                                    'strength': rho_star}, cand['occ_oof'], cand['cal_L'], (1, FR.N_XF, 0, 1))
    for side in ('LO', 'HI'):
        if ids[side]['config'] is not None:
            s = ids[side]['s']
            put(ids[side]['config'], full0, s, {'op': 'tfro', 'rho': 0.0, 'scale': s, 'strength': s, 'bracket_side': side,
                                                'bracket_flags': br['flags']}, paths[s], path_L[s], (1, FR.N_XF, 0, 0))
    put(PREFIX + '_CONT', full0, 1.0, {'op': 'tfro', 'rho': 0.0, 'scale': 1.0, 'strength': 1.0}, paths[1.0], path_L[1.0], (0, 0, 0, 0))
    out['ctrl'] = {**{k: br[k] for k in ('origin', 'selection_data', 'key', 'path', 'lo', 'hi', 'flags', 'rule')}, 'saved': ids,
                   'alias_endpoints': ALIAS_ENDPOINTS}
    out['cont']['status'] = 'SAVED'
    t1 = time.process_time()
    views = {'SUCC': succ_pair, PREFIX: (succ_pair[0], cand['occ_oof']), PREFIX + '_CONT': (succ_pair[0], paths[1.0])}
    for side in ('LO', 'HI'):
        if ids[side]['config'] is not None:
            views[ids[side]['config']] = (succ_pair[0], paths[ids[side]['s']])
    check = FK.mf_reader_check(views, lab['sex'], unit, f'SAR_mfcheck_FA_R1_S{int(version)}')
    out.update({'saved': saved, 'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input; SUCC is '
                                                            'in-sample on MODEL_FIT', 'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_R1_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks on invented data (no label file, no real teacher)
def selftest(n=1600, epochs=3):
    import tempfile
    from pathlib import Path as _P
    import torch
    t0 = time.process_time()
    Zc, yi, yo, sex, unit, succ, hgb = V1._invented(n, 11)
    A = V1.blocks(Zc, succ, hgb)
    ok = {}
    st0 = train(A, None, 0.0, 'r1-task', epochs=epochs)
    z = release_state(st0, A, succ[0], 0.0, succ[1])
    ok['s0_is_start_exactly'] = bool(np.array_equal(z[0], succ[0]) and np.array_equal(z[1], succ[1]))
    p1 = release_state(st0, A, succ[0], 1.0)
    ok['income_bitwise_succ'] = bool(np.array_equal(p1[0], succ[0]))
    zero = release_state({'checkpoints': {0: st0['checkpoints'][0]}, 'keep': 0, 'phi': st0['phi']}, A, succ[0], 1.0)
    ok['theta0_is_start'] = bool(np.max(np.abs(zero[1] - succ[1])) < 1e-10)
    W = st0['checkpoints'][st0['keep']]
    F = V1.phi_apply(st0['phi'], A)
    m = _net(F.shape[1], 0)
    with torch.no_grad():
        for p, w in zip(m.parameters(), W):
            p.copy_(torch.from_numpy(w))
        tout = m(torch.from_numpy(F)).numpy()
    ok['numpy_forward_equals_torch'] = bool(np.max(np.abs(tout - V1.net_numpy(W, F))) < 1e-10)
    st1 = train(A, sex, 3.0, 'r1-crit', epochs=epochs)
    st1b = train(A, sex, 3.0, 'r1-crit', epochs=epochs)
    ok['deterministic'] = all(np.array_equal(a, b) for a, b in zip(st1['checkpoints'][st1['keep']], st1b['checkpoints'][st1b['keep']]))
    ok['lambda_positive'] = st1['info']['lambda'] > 0
    kl = lambda P, Q: float(np.mean(np.sum(P * (np.log(P) - np.log(Q)), 1)))
    ok['task_term_moves_occupation_toward_teacher'] = kl(hgb[1], p1[1]) < kl(hgb[1], succ[1])
    # the guard estimate recovers each group's change exactly when the CALIBRATION average and the MODEL_FIT contrast are consistent
    rng = np.random.default_rng(2)
    nm = 3000
    s_ = rng.integers(0, 2, nm)
    u_ = np.arange(nm) // 2
    base = rng.normal(0.6, 0.1, nm)
    dm, dw = 0.004, -0.002
    mf_c = {'occ_brier': base + np.where(s_ == 0, dm, dw)}
    mf_s = {'occ_brier': base}
    p = float((s_ == 0).mean())
    lvl = p * dm + (1 - p) * dw
    cal_s = {'occ_brier': base[:1000]}
    cal_c = {'occ_brier': base[:1000] + lvl}
    g = guard_estimate(cal_c, cal_s, u_[:1000], mf_c, mf_s, s_, u_)
    ok['guard_recovers_groups'] = abs(g['men'][0] - dm) < 1e-12 and abs(g['women'][0] - dw) < 1e-12
    ok['guard_se_positive'] = g['men'][1] >= 0 and g['women'][1] >= 0
    # the cap-derived margins are read from the committed anchor table (label-free record)
    mg = succ_margins(0), succ_margins(1)
    ok['margins_from_anchor_table'] = (abs(mg[0]['men'] - (0.005 - 0.004417365683711137)) < 1e-12 and
                                       abs(mg[1]['women'] - (0.005 + 0.003038696569772226)) < 1e-12 and
                                       abs(mg[0]['occ_nll'] - (0.01 - 0.003406049134907274)) < 1e-12)
    # end-to-end stage on invented data: fake teachers, temporary model directory, fake save
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    X = np.column_stack([Zc, np.zeros((n, 2))])
    names = tuple(f'f{i}' for i in range(X.shape[1]))
    data = {'X': X, 'feature_names': names, 'unit': unit, 'income': yi, 'occupation': yo, 'sex': sex}
    Zc2, yi2, yo2, _, unit2, succ2, _ = V1._invented(600, 12)
    X2 = np.column_stack([Zc2, np.zeros((600, 2))])
    cal = {'X': X2, 'feature_names': names, 'unit': unit2, 'income': yi2, 'occupation': yo2}
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from nrm.fa import prc
    mi = HGB(max_iter=10).fit(Zc, yi); mo = HGB(max_iter=10).fit(Zc, yo)
    Tc = {'models': {'inc': mi, 'occ': mo}, 'oof': {'inc': prc._proba(mi, Zc), 'occ': prc._proba(mo, Zc)}}
    table = {X.tobytes(): succ, X2.tobytes(): succ2}
    old = (C.RUNS, FK.compact, V1._succ)
    tmp = _P(tempfile.mkdtemp(prefix='sar_fa_tfro_selftest_'))
    big = {k: 100.0 for k in ('occ_nll', 'occ_brier', 'occ_acc', 'men', 'women')}
    try:
        C.RUNS = tmp
        FK.compact = lambda X_, n_: np.asarray(X_, np.float64)[:, :6]
        V1._succ = lambda v, X_, n_: dict(zip(('income', 'occupation'), table[np.asarray(X_, np.float64).tobytes()]))
        res = stage_fit(0, data, cal, 'invented-card', fake_save, roles={'MODEL_FIT': {'X': X, 'feature_names': names}},
                        log=lambda s_: None, record_dir=False, epochs=epochs, rho_grid=(0.3, 1.0), margins=big, cache=(Tc, 'invented'),
                        frozen={})
        again = predict(PREFIX, 0, X, names)
        ok['deploy_bitwise'] = all(np.array_equal(again[t], saved[PREFIX][0]['MODEL_FIT'][t]) for t in ('income', 'occupation'))
        ok['saved_income_bitwise_succ'] = all(np.array_equal(a['MODEL_FIT']['income'], succ[0]) for a, _ in saved.values())
        ok['model_paths'] = all(p_.is_file() for p_ in model_paths(PREFIX, 0))
        from . import predict as route                                   # the package router sends R1 ids here
        ok['package_routes_r1'] = all(np.array_equal(route(PREFIX, 0, X, names)[t], again[t]) for t in ('income', 'occupation'))
        strict = dict(big, men=-1.0)                                      # a guard nobody can meet: nothing is released
        saved_before = set(saved)
        res2 = stage_fit(0, data, cal, 'invented-card', fake_save, roles={'MODEL_FIT': {'X': X, 'feature_names': names}},
                         log=lambda s_: None, record_dir=False, epochs=epochs, rho_grid=(0.3, 1.0), margins=strict,
                         cache=(Tc, 'invented'), frozen={})
        ok['first_failure_stops'] = (res2['cand']['status'] == 'NO_FEASIBLE_STRENGTH' and list(res2['selection']['per_rho']) == ['0.3']
                                     and set(saved) == saved_before)
    finally:
        C.RUNS, FK.compact, V1._succ = old
    ids = res['ctrl']['saved']
    want = {PREFIX, PREFIX + '_CONT'} | {ids[s]['config'] for s in ('LO', 'HI') if ids[s]['config']}
    ok['stage_saved_declared'] = set(saved) == want and len(saved) <= 4
    ok['rho_star_largest_feasible'] = res['selection']['rho_star'] == 1.0 and list(res['selection']['per_rho']) == ['0.3', '1.0']
    ok['record_fields'] = (res['cand']['status'] == 'FITTED' and {'spend_over_start', 'se'} <= set(res['cand']['cal_spend_over_SUCC'])
                           and res['ctrl']['selection_data'] == 'CALIBRATION' and res['ctrl']['origin'].startswith('START')
                           and res['cont']['gating'] is True
                           and all({'part_a', 'part_b', 'guard_estimate'} <= set(r) for r in res['selection']['per_rho'].values()))
    ok['records_kinds'] = all(r['kind'] == CONFIGS[c]['kind'] and r['role'] == CONFIGS[c]['role'] for c, (_, r) in saved.items())
    ok['all_ok'] = all(ok.values())
    ok['cpu_s'] = round(time.process_time() - t0, 1)
    ok['ctrl_points_in_stage_test'] = {s: ids[s] for s in ('LO', 'HI')}
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fa.tfro')
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
        r = V1.preflight(a.version)
        r['margins'] = succ_margins(a.version)
        print(json.dumps(r, indent=1, default=str))
        return 0 if r['ok'] else 1
    pf = V1.preflight(a.version)
    if not pf['ok']:
        raise PermissionError(f'frozen inputs not pinned / changed: missing {pf["missing"]}, mismatched {pf["mismatched"]}')
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    cal = C.load('CALIBRATION', ('features', 'income', 'occupation'), 'calibrate', PURPOSE_CAL, card=a.card)
    out = stage_fit(a.version, data, cal, a.card, RL.save)
    print(json.dumps({'rho_star': out['selection']['rho_star'], 'saved': sorted(out['saved']),
                      'ctrl_flags': (out['ctrl'] or {}).get('flags'), 'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

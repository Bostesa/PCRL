#!/usr/bin/env python3
"""RFA1 independent verifier, wave 1, check 2: graph wiring of the training loop rfa/fam/fit.py on INVENTED data (no label file,
no feature file, no model file; the guard of vguard refuses any .private path).

rfa.fam.fit.arm_paths is run for 3 steps per arm on invented rows (U0-architecture float32 networks with random weights, invented
households, occupation labels and SEX) with the registered RECIPE_RF except smaller episodes (n_A 600, n_E 400; the pool has
~3,000 rows). torch.optim.Adam and torch.nn.utils.clip_grad_norm_ are wrapped for the duration of the call to record the gradient
that reaches the optimizer (pre- and post-clip), the episode draw and the objective's inputs are recorded, and everything is then
recomputed independently: the task rows and episodes from the verifier's own replication of the seeded streams, the CE gradient on
a fresh copy of the network, the episode-objective gradient by a fresh EpisodeObjective sequence on the recorded inputs, the
calibration lambda from its own 16 draws, and the clip.

Run:  <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v1_fit_wiring_probe.py
Writes roles/verifier/out/V1_FIT_WIRING.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vguard  # noqa: E402

vguard.install()
sys.path.insert(1, str(ROOT))

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

T0 = time.process_time()
OUT = HERE.parent / 'out' / 'V2_FIT_WIRING_R1.json'
FILES = ('rfa/fam/fit.py', 'rfa/fam/rfit.py', 'rfa/fam/rfit_r1.py', 'rfa/fam/net.py')


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def v_seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:15], 16)


def v_episode(unit_idx_sizes, rng, n_a, n_e):
    """Verifier's replication of the documented rule: households (np.unique order) permuted by rng; A = the first households whose
    cumulative size reaches n_a; E = the next households reaching n_e more rows."""
    hh_of_row, sizes = unit_idx_sizes
    order = rng.permutation(len(sizes))
    A_h, E_h, na, ne = [], [], 0, 0
    for h in order:
        if na < n_a:
            A_h.append(h); na += sizes[h]
        elif ne < n_e:
            E_h.append(h); ne += sizes[h]
        else:
            break
    A = np.flatnonzero(np.isin(hh_of_row, A_h)); E = np.flatnonzero(np.isin(hh_of_row, E_h))
    return A, E


def main():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from acs_replication import neural
    from rfa.fam import fit as FIT
    from rfa.fam import rfit_r1 as RF          # the R1 objective module, passed to fit.arm_paths as rf_module
    from rfa.fam import net as NET
    res = {'schema': 'RFA1-verifier-fit-wiring-R1-v2', 'files_sha256_at_start': {f: sha(f) for f in FILES}, 'checks': {}, 'numbers': {}}
    chk, num = res['checks'], res['numbers']
    # ---- invented pool
    rng = np.random.default_rng(2026)
    H = 1250
    sizes = rng.integers(1, 5, H)
    unit = np.repeat(np.arange(H) * 7 + 3, sizes)               # non-contiguous household ids
    n = len(unit)
    sex = rng.integers(0, 2, n).astype(np.int64)
    X = rng.normal(size=(n, 145)).astype(np.float32)
    X[:, :10] += (0.4 * (2 * sex[:, None] - 1)).astype(np.float32)
    yO = np.clip(np.floor(2 + 1.2 * X[:, 20] + 0.6 * (2 * sex - 1) + rng.normal(0, 0.5, n)), 0, 4).astype(np.int64)
    torch.manual_seed(77)
    inc_net, occ_net = neural.network(2, 145), neural.network(5, 145)
    start_w = {'income': NET.layers_of(inc_net.state_dict()), 'occupation': NET.layers_of(occ_net.state_dict())}
    rr = copy.deepcopy(RF.RECIPE_RF_R1)
    rr['n_A'], rr['n_E'] = 600, 400
    rr['episode'] = dict(rr['episode'], n_A=600, n_E=400)
    num['pool'] = {'people': int(n), 'households': int(H), 'recipe_rf_changes': {'n_A': 600, 'n_E': 400},
                   'registered_views': rr['views'], 'registered_families': rr['families']}
    version = 0
    STEPS = 3
    # ---- instrumentation
    rec = {'adam_steps': [], 'clip_calls': [], 'episodes': [], 'obj_calls': [], 'trainers': []}
    orig_adam, orig_clip = torch.optim.Adam, torch.nn.utils.clip_grad_norm_
    orig_parts, orig_episode, orig_init = RF.EpisodeObjective.objective_parts, FIT.OccTrainer.episode, FIT.OccTrainer.__init__
    orig_fit_reader = RF.fit_reader
    rec['fit_reader_calls'] = []

    def rec_fit_reader(*a, **k):
        rec['fit_reader_calls'].append(state['arm'])
        return orig_fit_reader(*a, **k)
    state = {'arm': None}

    class RecAdam(orig_adam):
        def step(self, closure=None):
            ps = [p for g in self.param_groups for p in g['params']]
            rec['adam_steps'].append({'arm': state['arm'], 'n_tensors': len(ps), 'ids': [id(p) for p in ps],
                                      'grads': [p.grad.detach().clone() for p in ps]})
            return super().step(closure)

    def rec_clip(params, max_norm, *a, **k):
        params = list(params)
        rec['clip_calls'].append({'arm': state['arm'], 'max_norm': float(max_norm), 'pre': [p.grad.detach().clone() for p in params]})
        return orig_clip(params, max_norm, *a, **k)

    def rec_parts(self, zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments=None, diagnostics=True):
        rec['obj_calls'].append({'arm': state['arm'], 'mode': self.mode, 'cls': type(self).__module__, 'families': list(self.families), 'rg': [bool(z.requires_grad) for z in (zA_inc, zA_occ, zE_inc, zE_occ)],
                                 'zA_inc': zA_inc.detach().clone(), 'zA_occ': zA_occ.detach().clone(), 'zE_inc': zE_inc.detach().clone(),
                                 'zE_occ': zE_occ.detach().clone(), 'sA': sA.clone(), 'sE': sE.clone()})
        return orig_parts(self, zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments, diagnostics)

    def rec_episode(self, rng_, n_a, n_e):
        A, E = orig_episode(self, rng_, n_a, n_e)
        rec['episodes'].append({'arm': state['arm'], 'A': A.numpy().copy(), 'E': E.numpy().copy()})
        return A, E

    def rec_init(self, *a, **k):
        orig_init(self, *a, **k)
        rec['trainers'].append(self)

    # run_arm sets state['arm'] through a wrapper of run_arm (the arm name is its second argument)
    orig_run_arm = FIT.run_arm

    def rec_run_arm(tr, mode, *a, **k):
        state['arm'] = mode
        try:
            return orig_run_arm(tr, mode, *a, **k)
        finally:
            state['arm'] = 'between'
    orig_cal = FIT.calibrate

    def rec_cal(tr, rho, n_draws=16):
        state['arm'] = 'CALIBRATION'
        try:
            return orig_cal(tr, rho, n_draws)
        finally:
            state['arm'] = 'between'
    ck_store = {}
    t1 = time.process_time()
    try:
        torch.optim.Adam = RecAdam
        torch.nn.utils.clip_grad_norm_ = rec_clip
        RF.EpisodeObjective.objective_parts = rec_parts
        FIT.OccTrainer.episode = rec_episode
        FIT.OccTrainer.__init__ = rec_init
        RF.fit_reader = rec_fit_reader
        FIT.run_arm = rec_run_arm
        FIT.calibrate = rec_cal
        paths, recs = FIT.arm_paths(X, yO, sex, unit, version, start_w, STEPS, 1, 1.0, recipe_rf=rr, rf_module=RF)
    finally:
        torch.optim.Adam, torch.nn.utils.clip_grad_norm_ = orig_adam, orig_clip
        RF.EpisodeObjective.objective_parts, FIT.OccTrainer.episode, FIT.OccTrainer.__init__ = orig_parts, orig_episode, orig_init
        FIT.run_arm, FIT.calibrate = orig_run_arm, orig_cal
        RF.fit_reader = orig_fit_reader
    num['arm_paths_cpu_s'] = time.process_time() - t1
    arms = list(FIT.ARMS)
    chk['every_objective_call_is_the_R1_class_with_softbin'] = bool(rec['obj_calls']) and all(
        c['cls'] == 'rfa.fam.rfit_r1' and c['families'] == ['simple', 'rff', 'softbin'] for c in rec['obj_calls'])
    num['objective_call_classes'] = sorted({(c['cls'], tuple(c['families'])) for c in rec['obj_calls']})
    # ---- (a) lambda: one calibration on FULL, shared by FULL / STOPFIT / ALTCRIT; TASK 0
    lam = {a: recs[a]['lambda'] for a in arms}
    num['lambda'] = lam
    chk['lambda_shared_by_FULL_STOPFIT_ALTCRIT_and_TASK_zero'] = (lam['FULL'] == lam['STOPFIT'] == lam['ALTCRIT'] > 0 and lam['TASK'] == 0.0)
    cal_calls = [c for c in rec['obj_calls'] if c['arm'] == 'CALIBRATION']
    chk['calibration_uses_FULL_objective_16_draws'] = len(cal_calls) == 16 and all(c['mode'] == 'FULL' for c in cal_calls)
    # verifier's own lambda: 16 draws, fresh START copy, own task-row and episode streams
    PD = FIT._pd()
    occ0 = PD.model_from_layers('occupation', start_w['occupation'])
    inc0 = PD.model_from_layers('income', start_w['income'])
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        zi_all = inc0(Xt).double(); zi_all = zi_all - zi_all.mean(1, keepdim=True)
    hh_of_row = np.unique(unit, return_inverse=True)[1]
    sizes_u = np.bincount(hh_of_row)
    ep_c = np.random.default_rng(v_seed_for('RFA1-calibration-episodes', version) % 2 ** 32)
    tk_c = np.random.default_rng(v_seed_for('RFA1-calibration-task', version) % 2 ** 32)
    objc = RF.EpisodeObjective('FULL', version, recipe=rr)
    yt = torch.from_numpy(yO)
    st_t = torch.from_numpy(sex)
    tn, pn = [], []

    def logits(model, rows):
        z = model(Xt[rows]).double()
        return z - z.mean(1, keepdim=True)
    import torch.nn.functional as Fnn
    ep_match = True
    cal_eps = [e for e in rec['episodes'] if e['arm'] == 'CALIBRATION']
    ep_match &= len(cal_eps) == 16
    for d in range(16):
        rows = torch.from_numpy(tk_c.choice(n, 256, replace=False))
        A, E = v_episode((hh_of_row, sizes_u), ep_c, rr['n_A'], rr['n_E'])
        ep_match &= bool(np.array_equal(cal_eps[d]['A'], A) and np.array_equal(cal_eps[d]['E'], E))
        ep_match &= bool(np.array_equal(cal_calls[d]['sA'].numpy(), sex[A]) and np.array_equal(cal_calls[d]['sE'].numpy(), sex[E]))
        gt = torch.autograd.grad(Fnn.cross_entropy(logits(occ0, rows), yt[rows]), list(occ0.parameters()))
        S, _ = objc.objective(zi_all[A], logits(occ0, A), st_t[A], zi_all[E], logits(occ0, E), st_t[E])
        gp = torch.autograd.grad(S, list(occ0.parameters()))
        tn.append(float(torch.sqrt(sum((g.double() ** 2).sum() for g in gt))))
        pn.append(float(torch.sqrt(sum((g.double() ** 2).sum() for g in gp))))
    lam_v = statistics.median(tn) / statistics.median(pn)
    num['lambda_verifier'] = lam_v
    num['lambda_rel_diff'] = abs(lam_v - lam['FULL']) / lam['FULL']
    chk['calibration_lambda_reproduced (rel < 1e-9)'] = num['lambda_rel_diff'] < 1e-9
    chk['calibration_episodes_follow_documented_rule'] = ep_match
    # ---- (b) episodes: identical sequences in FULL / STOPFIT / ALTCRIT; household-separated; the documented rule; TASK none
    eps = {a: [e for e in rec['episodes'] if e['arm'] == a] for a in arms}
    num['episodes_per_arm'] = {a: len(eps[a]) for a in arms}
    same = all(np.array_equal(x['A'], y['A']) and np.array_equal(x['E'], y['E'])
               for a in ('STOPFIT', 'ALTCRIT') for x, y in zip(eps['FULL'], eps[a])) and \
        len(eps['FULL']) == len(eps['STOPFIT']) == len(eps['ALTCRIT']) == STEPS
    chk['same_episodes_in_FULL_STOPFIT_ALTCRIT'] = same
    chk['TASK_draws_no_episode_and_no_objective_call'] = len(eps['TASK']) == 0 and not [c for c in rec['obj_calls'] if c['arm'] == 'TASK']
    sep = all(not np.intersect1d(unit[e['A']], unit[e['E']]).size for a in arms for e in eps[a])
    chk['episodes_household_separated'] = sep
    ep_r = np.random.default_rng(v_seed_for('RFA1-episodes', version) % 2 ** 32)
    rule = True
    for e in eps['FULL']:
        A, E = v_episode((hh_of_row, sizes_u), ep_r, rr['n_A'], rr['n_E'])
        rule &= bool(np.array_equal(A, e['A']) and np.array_equal(E, e['E']))
    chk['training_episodes_follow_documented_rule_and_seed'] = rule
    num['episode_sizes'] = [(int(len(e['A'])), int(len(e['E']))) for e in eps['FULL']]
    # ---- (c) trainers: TASK has no SEX; income logits constant (no graph); only occupation parameters reach the optimizer
    trs = rec['trainers']
    num['n_trainers'] = len(trs)
    arm_tr = {a: trs[1 + i] for i, a in enumerate(arms)}          # trainer 0 = calibration
    chk['TASK_trainer_has_no_SEX'] = arm_tr['TASK'].sex is None and all(arm_tr[a].sex is not None for a in ('FULL', 'STOPFIT', 'ALTCRIT'))
    chk['income_logits_constant_no_graph'] = all(not t.z_inc.requires_grad and t.z_inc.grad_fn is None for t in trs)
    calls = [c for c in rec['obj_calls'] if c['arm'] in ('FULL', 'STOPFIT', 'ALTCRIT')]
    chk['objective_inputs_income_no_grad_occupation_with_grad'] = all(c['rg'] == [False, True, False, True] for c in calls)
    sexok = True
    for a in ('FULL', 'STOPFIT', 'ALTCRIT'):
        cs = [c for c in calls if c['arm'] == a]
        for c, e in zip(cs, eps[a]):
            sexok &= bool(np.array_equal(c['sA'].numpy(), sex[e['A']]) and np.array_equal(c['sE'].numpy(), sex[e['E']]))
            sexok &= bool(torch.allclose(c['zA_inc'], zi_all[e['A']], atol=0, rtol=0) and torch.allclose(c['zE_inc'], zi_all[e['E']], atol=0, rtol=0))
    chk['objective_SEX_and_income_rows_aligned_with_episode_rows'] = sexok
    adam = {a: [s_ for s_ in rec['adam_steps'] if s_['arm'] == a and s_['n_tensors'] == 8] for a in arms}
    critic_adam = [s_ for s_ in rec['adam_steps'] if s_['n_tensors'] == 1]
    occ_ids = {a: [id(p) for p in arm_tr[a].occ.parameters()] for a in arms}
    chk['optimizer_holds_exactly_the_occupation_parameters'] = all(len(adam[a]) == STEPS and all(s_['ids'] == occ_ids[a] for s_ in adam[a])
                                                                  for a in arms)
    num['alternating_critic_adam_steps_recorded'] = len(critic_adam)
    inc_bitwise = all(all(np.array_equal(w0, w1) for (w0, _), (w1, _) in zip(start_w['income'], paths[a][s]['income'])) and
                      all(np.array_equal(b0, b1) for (_, b0), (_, b1) in zip(start_w['income'], paths[a][s]['income']))
                      for a in arms for s in paths[a])
    occ_moved = all(not all(np.array_equal(w0, w1) for (w0, _), (w1, _) in zip(start_w['occupation'], paths[a][STEPS]['occupation']))
                    for a in arms)
    chk['income_layers_bitwise_START_at_every_checkpoint'] = inc_bitwise
    chk['occupation_layers_move_in_every_arm'] = occ_moved
    # ---- (d) the derivative passed to the optimizer: pre-clip = grad CE(task rows) + lambda x grad S; post-clip = clip(pre, 5)
    tk = np.random.default_rng(v_seed_for('THP1task', version) % 2 ** 32)
    order = tk.permutation(n)
    rows_k = [order[k * 256:(k + 1) * 256] for k in range(STEPS)]
    deriv = {}
    for a in arms:
        clips = [c for c in rec['clip_calls'] if c['arm'] == a]
        objv = None if a == 'TASK' else RF.EpisodeObjective(a, version, recipe=rr)
        errs, clip_errs = [], []
        for k in range(STEPS):
            W = paths[a][k]['occupation']                       # weights before step k (checkpoint k)
            m = PD.model_from_layers('occupation', W)
            gt = torch.autograd.grad(Fnn.cross_entropy(logits(m, torch.from_numpy(rows_k[k])), yt[torch.from_numpy(rows_k[k])]),
                                     list(m.parameters()))
            if a == 'TASK':
                gs = [torch.zeros_like(x) for x in gt]
            else:
                e = eps[a][k]
                At, Et = torch.from_numpy(e['A']), torch.from_numpy(e['E'])
                S, _ = objv.objective(zi_all[At], logits(m, At), st_t[At], zi_all[Et], logits(m, Et), st_t[Et])
                gs = torch.autograd.grad(S, list(m.parameters()))
            want = [x + lam[a] * y for x, y in zip(gt, gs)]
            got = clips[k]['pre']
            num_ = math.sqrt(sum(float(((x.double() - y.double()) ** 2).sum()) for x, y in zip(got, want)))
            den = math.sqrt(sum(float((y.double() ** 2).sum()) for y in want))
            errs.append(num_ / den)
            tot = math.sqrt(sum(float((x.double() ** 2).sum()) for x in got))
            scale = min(1.0, 5.0 / (tot + 1e-6))
            post = adam[a][k]['grads']
            clip_errs.append(max(float((p.double() - scale * q.double()).abs().max()) for p, q in zip(post, got)) /
                             max(float((scale * q.double()).abs().max()) for q in got))
        deriv[a] = {'rel_err_pre_clip_vs_gradCE_plus_lambda_gradS': errs, 'rel_err_post_clip_vs_verifier_clip': clip_errs,
                    'clip_max_norm': [c['max_norm'] for c in clips]}
    num['derivative_passed'] = deriv
    chk['derivative_passed_is_gradCE_plus_lambda_gradS (rel < 1e-5, float32)'] = all(max(d['rel_err_pre_clip_vs_gradCE_plus_lambda_gradS']) < 1e-5
                                                                                     for d in deriv.values())
    chk['clip_is_total_norm_5'] = all(max(d['rel_err_post_clip_vs_verifier_clip']) < 1e-5 and all(x == 5.0 for x in d['clip_max_norm']) for d in deriv.values())
    # ---- (e) FULL's separate direct / fit parts sum to the gradient of S (checked above through S = S_dir + S_fit); report shares
    full_logs = recs['FULL']['logs']
    num['FULL_logged_fit_share'] = [l.get('output_grad_norms', {}).get('fit_share') for l in full_logs]
    fps = FIT.fit_path_share(full_logs)
    num['fit_path_share_helper_returns'] = fps
    logged = [x for x in num['FULL_logged_fit_share'] if isinstance(x, (int, float))]
    chk['DEFECT_PROBE fit_path_share helper finds the logged share'] = fps is not None and abs(fps['median'] - statistics.median(logged)) < 1e-15
    calls = {a: rec['fit_reader_calls'].count(a) for a in arms + ['CALIBRATION']}
    crit = {a: len([s_ for s_ in rec['adam_steps'] if s_['arm'] == a and s_['n_tensors'] == 1]) for a in arms}
    num['inner_fit_accounting'] = {a: {'recorded_inner_reader_fits': recs[a].get('inner_reader_fits'), 'actual_exact_fit_reader_calls': calls[a],
                                       'actual_critic_adam_steps': crit[a]} for a in arms}
    num['inner_fit_accounting']['CALIBRATION_actual_exact_fit_reader_calls'] = calls['CALIBRATION']
    chk['DEFECT_PROBE inner_reader_fits equals the actual exact fits (FULL, STOPFIT)'] = all(
        recs[a].get('inner_reader_fits') == calls[a] for a in ('FULL', 'STOPFIT'))
    chk['DEFECT_PROBE inner_reader_fits equals the actual exact fits (ALTCRIT)'] = recs['ALTCRIT'].get('inner_reader_fits') == calls['ALTCRIT']
    num['mechanism_by_epoch_FULL'] = recs['FULL'].get('mechanism_by_epoch')
    num['monitors_logged'] = {a: len(recs[a].get('monitors', [])) for a in arms}
    # ---- (f) mechanism_stats vs the verifier's own cross-group estimator on invented vectors
    rg = np.random.default_rng(5)
    p, m_ = 40, 64
    base_d, base_f = rg.normal(size=p), rg.normal(size=p) * 0.3
    gd = [torch.from_numpy(base_d + rg.normal(size=p) * 2.0) for _ in range(m_)]
    gf = [torch.from_numpy(base_f + 0.5 * base_d + rg.normal(size=p) * 2.0) for _ in range(m_)]
    ms = FIT.mechanism_stats(gd, gf, 8)
    D = np.stack([np.mean([x.numpy() for x in gd[i * 8:(i + 1) * 8]], 0) for i in range(8)])
    Fm = np.stack([np.mean([x.numpy() for x in gf[i * 8:(i + 1) * 8]], 0) for i in range(8)])
    T = D + Fm

    def cross(Xa, Ya):
        tot_ = 0.0
        for i in range(8):
            for j in range(8):
                if i != j:
                    tot_ += float(Xa[i] @ Ya[j])
        return tot_ / 56.0
    dd, ff, fdd, tt, td = cross(D, D), cross(Fm, Fm), cross(Fm, D), cross(T, T), cross(T, D)
    mine = {'systematic_share': math.sqrt(max(ff, 0)) / math.sqrt(dd), 'orthogonal_share': math.sqrt(max(ff - fdd ** 2 / dd, 0)) / math.sqrt(dd),
            'cos_tot_dir': td / math.sqrt(tt * dd), 'parallel_coefficient': fdd / dd}
    num['mechanism_stats_compare'] = {'module': {k: ms[k] for k in mine}, 'verifier': mine,
                                      'truth_from_construction': {'parallel_coefficient': 0.5 + float(base_f @ base_d / (base_d @ base_d))}}
    chk['mechanism_stats_match_verifier_cross_group_estimator'] = all(abs(ms[k] - mine[k]) < 1e-10 for k in mine)
    # gate leaves the weights untouched
    gtr = FIT.OccTrainer(version, start_w['income'], start_w['occupation'], X, yO, sex, unit, rr)
    gtr.rf_module = RF                                 # as stage_fit does for a revision (fit.py: gtr.rf_module = rf_module)
    before = [p.detach().clone() for p in gtr.params()]
    gate = FIT.mechanism_gate(gtr, n_episodes=16, groups=4)
    chk['mechanism_gate_does_not_update_weights'] = all(torch.equal(a_, b_) for a_, b_ in zip(before, gtr.params()))
    num['mechanism_gate_invented'] = {k: gate[k] for k in ('label', 'systematic_share', 'orthogonal_share', 'cos_tot_dir', 'n', 'status')}
    res['files_sha256_at_end'] = {f: sha(f) for f in FILES}
    res['files_unchanged_during_probe'] = res['files_sha256_at_end'] == res['files_sha256_at_start']
    res['guard_refused_opens'] = list(vguard.REFUSED)
    res['defect_probes'] = {k: v for k, v in chk.items() if k.startswith('DEFECT_PROBE')}
    core = {k: v for k, v in chk.items() if not k.startswith('DEFECT_PROBE')}
    res['all_core_pass'] = all(bool(v) for v in core.values())
    res['cpu_s'] = time.process_time() - T0
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
    print(json.dumps({'all_core_pass': res['all_core_pass'], 'checks': chk, 'cpu_s': res['cpu_s'], 'lambda': lam,
                      'derivative_passed': deriv}, indent=1, default=str))
    return 0 if res['all_core_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

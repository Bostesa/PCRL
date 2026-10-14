"""Independent verification Stage A (+A4) for one source seed; queue-only worker.

Order: A3 (counterfactual re-solves), A1 (forward/witness), A2 (pool metrics & budget audit),
A4 (FIT2 point clauses). Each part's result is written as soon as it finishes.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import core as C

ARMS = ('J_TRUST', 'J_FREE', 'L_TRUST', 'L_FREE', 'TASK_TRUST')
EPOCHS = (0, 1, 3, 5, 10)
TRUST = {'J_TRUST': True, 'J_FREE': False, 'L_TRUST': True, 'L_FREE': False, 'TASK_TRUST': True}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def maxabs(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main(spec_path):
    from minimal_task_wire_adapter.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    torch.set_num_threads(1)
    t0 = time.time()
    V = lambda it: verify(it['path'], it['sha256'])
    seed = int(cfg['seed'])
    out_path = Path(cfg['output'])
    res = {'unit_id': active['unit_id'], 'seed': seed, 'parts': {}, 'errors': {}}

    def dump():
        res['elapsed_s'] = time.time() - t0
        out_path.write_text(json.dumps(jsonable(res), indent=1, sort_keys=True))

    # ------------------------------------------------------------ common setup
    payload = C.tload(V(cfg['checkpoint']))
    src = C.Source(payload['model'])
    geom = json.loads(V(cfg['geometry']).read_text())
    binding = json.loads(V(cfg['binding']).read_text())
    admission = json.loads(V(cfg['admission']).read_text())
    s = float(binding['scale'])
    Bs = []
    for i in (0, 1):
        h = geom['heads'][i]
        Bs.append(torch.from_numpy(np.asarray(h['B'], np.float64).reshape(int(h['rank']), 16).copy()))
    Bi, B = Bs
    rank = B.shape[0]
    W1, b1 = src.head(1); W0, b0 = src.head(0)
    G = {'B': B, 'W': W1, 'bias': b1}

    def role_feats(item):
        f = C.npz(V(item))
        return f
    fU = role_feats(cfg['roles']['UPDATE']['features']); fG = role_feats(cfg['roles']['GUARD']['features'])
    fV = role_feats(cfg['roles']['VALIDATION']['features'])
    oU = C.npz(V(cfg['roles']['UPDATE']['occupation'])); oG = C.npz(V(cfg['roles']['GUARD']['occupation']))
    sxU = C.npz(V(cfg['roles']['UPDATE']['sex']))
    for a, b_ in ((fU, oU), (fU, sxU), (fG, oG)):
        assert np.array_equal(a['row_id'], b_['row_id'])

    def frozen(X):
        h = src.hidden(1, X); z0 = src.final(1, h)
        zi = src.final(0, src.hidden(0, X))
        return {'h': h, 'z0': z0, 'hbar': h.to(torch.float64) / s, 'q0': z0.to(torch.float64) @ B.T,
                'inc': C.raw_proj_wire(zi, Bi, W0, b0)}
    FU, FG, FV = frozen(fU['X']), frozen(fG['X']), frozen(fV['X'])
    yU = torch.from_numpy(oU['occupation'].astype(np.int64)); yG = torch.from_numpy(oG['occupation'].astype(np.int64))
    sU = torch.from_numpy(sxU['sex'].astype(np.int64))
    hP = torch.cat((FU['hbar'], FG['hbar'])); qP = torch.cat((FU['q0'], FG['q0'])); yP = torch.cat((yU, yG))
    zeroR = torch.zeros(rank, 64, dtype=torch.float64); zerob = torch.zeros(rank, dtype=torch.float64)
    with torch.no_grad():
        lp0U = torch.log_softmax(C.occ_wire(FU['hbar'], FU['q0'], zeroR, zerob, B, W1, b1)['centered_logits'], 1)
        lp0G = torch.log_softmax(C.occ_wire(FG['hbar'], FG['q0'], zeroR, zerob, B, W1, b1)['centered_logits'], 1)
    lp0P = torch.cat((lp0U, lp0G))
    npool = {'P': (hP, qP, yP, lp0P), 'U': (FU['hbar'], FU['q0'], yU, lp0U), 'G': (FG['hbar'], FG['q0'], yG, lp0G)}

    def metrics(vec, which='P'):
        R, b = C.unflatten(vec, rank)
        h, q, y, lp = npool[which]
        with torch.no_grad():
            cl = C.occ_wire(h, q, R, b, B, W1, b1)['centered_logits'].numpy()
        return C.np_metrics(cl, y.numpy(), lp.numpy())
    start = metrics(np.zeros(rank * 64 + rank))
    limits = {'nll': start['nll'] + C.BUDGET['nll'], 'brier': start['brier'] + C.BUDGET['brier'],
              'correct': start['correct'] - int(C.BUDGET['acc_frac'] * start['rows'] + 1e-9), 'kl': C.BUDGET['kl']}
    res['my_start_pool'] = start; res['my_limits'] = limits
    probe = C.probe_positions(list(fU['row_id']))
    import hashlib
    res['probe_position_sha256_matches_split'] = hashlib.sha256(json.dumps(probe).encode()).hexdigest() == cfg['probe_position_sha256']

    # checkpoints (R,b + metadata)
    ck = {}
    for arm in ARMS:
        for e in EPOCHS:
            ck[(arm, e)] = C.tload(V(cfg['training'][arm]['epochs'][str(e)]))
    summ = {arm: json.loads(V(cfg['training'][arm]['summary']).read_text()) for arm in ARMS}
    traces = {arm: np.load(V(cfg['training'][arm]['state_trace']), allow_pickle=False) for arm in ARMS}

    def ckvec(arm, e):
        p = ck[(arm, e)]
        return C.flatten(p['R'].to(torch.float64), p['b_res'].to(torch.float64))

    # ------------------------------------------------------------ A3 counterfactual re-solves
    def part_a3():
        guard()
        states = C.tload(V(cfg['training']['J_TRUST']['counterfactual_states']))
        scan = {r['update_number']: r for r in summ['J_TRUST']['counterfactual_scan']}
        tr = traces['J_TRUST']
        rows_out = []
        hG_, qG_, yG_ = FG['hbar'], FG['q0'], yG
        for st in states:
            n = int(st['update_number'])
            rec = {'update_number': n, 'epoch': int(st['epoch'])}
            pre = C.flatten(st['pre_R'].to(torch.float64), st['pre_b'].to(torch.float64))
            rec['pre_equals_trace_n_minus_1'] = bool(np.array_equal(pre, tr[n - 1]))
            Rt, bt = C.unflatten(pre, rank)
            cur = metrics(pre)
            rec['current_vs_stored'] = {k: abs(cur[k] - st['pool_current'][k]) for k in ('nll', 'brier', 'kl')}
            rec['current_correct_equal'] = int(cur['correct']) == int(st['pool_current']['correct'])
            pg = C.pool_gradients(G, Rt, bt, hP, qP, yP, lp0P)
            rec['pool_gradient_maxabs'] = {k: maxabs(pg[k], st['pool_gradients'][k].numpy()) for k in pg}
            rec['pool_gradient_norm'] = {k: float(np.linalg.norm(pg[k])) for k in pg}
            _, gguard = C.grad_of(lambda R, b: C.my_ce(C.occ_wire(hG_, qG_, R, b, B, W1, b1)['centered_logits'], yG_), Rt, bt)
            rec['guard_ce_gradient_maxabs'] = maxabs(gguard, st['guard_ce_gradient'].numpy())
            rows = list(map(int, st['rows']))
            hb, qb, yb, ib, sb = FU['hbar'][rows], FU['q0'][rows], yU[rows], FU['inc']['full'][rows], sU[rows]
            _, gmb = C.grad_of(lambda R, b: C.my_ce(C.occ_wire(hb, qb, R, b, B, W1, b1)['centered_logits'], yb), Rt, bt)
            nU, nG = len(yU), len(yG)
            g_task = (nU / (nU + nG)) * gmb + (nG / (nU + nG)) * gguard
            rec['g_task_maxabs'] = maxabs(g_task, st['g_task'].numpy()); rec['g_task_norm'] = float(np.linalg.norm(g_task))
            cs = st['critic_state']
            _, g_priv = C.grad_of(lambda R, b: C.privacy_risk(G, cs, R, b, hb, qb, ib, sb, 'joint')[0], Rt, bt)
            with torch.no_grad():
                risk, ces = C.privacy_risk(G, cs, Rt, bt, hb, qb, ib, sb, 'joint')
            rec['g_privacy_maxabs'] = maxabs(g_priv, st['g_privacy'].numpy()); rec['g_privacy_norm'] = float(np.linalg.norm(g_priv))
            rec['critic_ces'] = ces; rec['joint_risk'] = float(risk)
            coef = float(st['coefficient'])
            rec['coefficient_matches_summary'] = coef == float(summ['J_TRUST']['coefficient'])
            prop = -C.LR * (g_task + coef * g_priv)
            rec['proposal_maxabs'] = maxabs(prop, st['proposal'].numpy())
            raw = [limits['nll'] - cur['nll'], limits['brier'] - cur['brier'], limits['kl'] - cur['kl']]
            rec['raw_slacks'] = raw
            if min(raw) < -C.TOL:
                rec['status'] = 'INFEASIBLE'; rows_out.append(rec); continue
            slack = np.maximum(0.0, raw)
            grads = np.vstack([pg['nll'], pg['brier'], pg['kl']])
            out = {}
            for label, p in (('privacy', prop), ('counterfactual', -C.LR * g_task)):
                d, kkt, (A, sn) = C.project(p, grads, slack)
                ds, ok = C.slsqp_project(p, A, sn)
                actual, frac, attempts, m, vec = C.ladder(pre, d, metrics, limits, True, A, sn)
                out[label] = {'d': d, 'actual': actual, 'vec': vec, 'fraction': frac}
                rec[label] = {'kkt': kkt, 'slsqp_maxabs_vs_exact': maxabs(ds, d), 'slsqp_success': ok,
                              'fraction': frac, 'status': 'ACCEPTED' if frac is not None else 'REJECTED_ZERO_STEP',
                              'post_metrics': m, 'attempts': [{k: a[k] for k in ('fraction', 'actual_norm', 'linearized', 'inside')} for a in attempts],
                              'actual_norm': float(np.linalg.norm(actual))}
            rec['privacy']['actual_maxabs_vs_stored'] = maxabs(out['privacy']['actual'], st['actual_privacy_displacement'].numpy())
            rec['privacy']['actual_maxabs_vs_trace'] = maxabs(out['privacy']['actual'], tr[n] - tr[n - 1])
            rec['privacy']['deployed_state_equals_trace'] = bool(np.array_equal(out['privacy']['vec'], tr[n]))
            rec['counterfactual']['actual_maxabs_vs_stored'] = maxabs(out['counterfactual']['actual'], st['actual_counterfactual_displacement'].numpy())
            rec['counterfactual']['status_matches'] = rec['counterfactual']['status'] == st['counterfactual_status']
            rec['counterfactual']['fraction_matches'] = out['counterfactual']['fraction'] == st['counterfactual_fraction']
            diff = out['privacy']['actual'] - out['counterfactual']['actual']
            rec['privacy_minus_cf_maxabs'] = float(np.max(np.abs(diff)))
            rec['privacy_minus_cf_norm'] = float(np.linalg.norm(diff))
            rec['privacy_changes_displacement_beyond_1e-10'] = bool(np.max(np.abs(diff)) > 1e-10)
            hp, qp_ = FU['hbar'][probe], FU['q0'][probe]
            with torch.no_grad():
                wp = C.np_wire(C.occ_wire(hp, qp_, *C.unflatten(out['privacy']['vec'], rank), B, W1, b1))
                wc = C.np_wire(C.occ_wire(hp, qp_, *C.unflatten(out['counterfactual']['vec'], rank), B, W1, b1))
            wd = C.wire_diff(wp, wc)
            rec['probe_wire_privacy_vs_cf'] = wd
            stored = scan[n]['privacy_vs_counterfactual_probe_wire_difference']
            rec['probe_wire_vs_stored_maxabs'] = {k: abs(wd[k] - stored[k]) for k in ('representation', 'centered_logits', 'probabilities')}
            rec['probe_decision_changes_match'] = wd['decision_changes'] == stored['decision_changes']
            rec['privacy_changes_probe_wire_beyond_1e-10'] = bool(max(wd[k] for k in ('representation', 'centered_logits', 'probabilities')) > 1e-10 or wd['decision_changes'] > 0)
            rec['stored_scan_minus_norm_diff'] = abs(scan[n]['privacy_minus_counterfactual_norm'] - rec['privacy_minus_cf_norm'])
            rows_out.append(rec)
        return {'states': rows_out, 'count': len(rows_out)}

    # ------------------------------------------------------------ A1 forward / witness
    def part_a1():
        guard()
        out = {}
        for name, i in (('income', 0), ('occupation', 1)):
            W, _ = src.head(i)
            Bm, sv, r_, A = C.own_rowspace(W.numpy())
            Bb = Bs[i].numpy()
            out[name] = {'my_rank': r_, 'bound_rank': int(Bb.shape[0]), 'my_singular_values': sv.tolist(),
                         'projector_maxabs_diff': maxabs(Bm.T @ Bm, Bb.T @ Bb),
                         'bound_rows_orthonormal_maxabs': maxabs(Bb @ Bb.T, np.eye(Bb.shape[0])),
                         'A_BtB_minus_A_maxabs': maxabs(A @ Bb.T @ Bb, A),
                         'rank_gap_ratio': float(sv[r_ - 1] / sv[r_]) if r_ < len(sv) and sv[r_] > 0 else 'inf'}
        adm = admission['seeds'][str(seed)]
        out['admission_sv_maxabs'] = maxabs(out['occupation']['my_singular_values'][:5], adm['head_norms']['occupation_centered_singular_values'][:5])
        hpool = torch.cat((FU['h'], FG['h'])).to(torch.float64).numpy()
        msq = float(np.mean((hpool ** 2).sum(1)))
        norms = np.sqrt((hpool ** 2).sum(1))
        out['hidden'] = {'mean_squared_norm': msq, 'scale': max(1.0, float(np.sqrt(msq / 64))), 'median_norm': float(np.median(norms)),
                         'min_norm': float(norms.min()), 'max_norm': float(norms.max()), 'fraction_zero': float(np.mean(hpool == 0))}
        hn = adm['hidden_norms']
        out['hidden_vs_admission_maxabs'] = max(abs(out['hidden']['mean_squared_norm'] - hn['mean_squared_norm']),
                                                abs(out['hidden']['median_norm'] - hn['median_norm']),
                                                abs(out['hidden']['min_norm'] - hn['min_norm']), abs(out['hidden']['max_norm'] - hn['max_norm']),
                                                abs(out['hidden']['fraction_zero'] - hn['fraction_zero_coordinates']))
        out['scale_matches_binding'] = out['hidden']['scale'] == s
        out['trainable_parameters'] = rank * 64 + rank
        out['trainable_matches_binding'] = out['trainable_parameters'] == binding['trainable_parameters'] and rank == binding['rank']
        # RAW_PROJ direct path vs zero-residual adapter path
        roles = {'UPDATE': FU, 'GUARD': FG, 'VALIDATION': FV}
        raw = {}
        wit = {}
        with torch.no_grad():
            for rn, F in roles.items():
                direct = C.np_wire(C.raw_proj_wire(F['z0'], B, W1, b1))
                zero = C.np_wire(C.occ_wire(F['hbar'], F['q0'], zeroR, zerob, B, W1, b1))
                raw[rn] = direct
                wit[rn] = {'bitwise': bool(np.array_equal(direct['full'], zero['full'])), 'maxabs': maxabs(direct['full'], zero['full'])}
        out['zero_residual_witness'] = wit
        # every checkpoint
        cks = {}
        ref_keys = None
        incV = FV['inc']['full'].numpy()
        for arm in ARMS:
            arm_rec = {}
            for e in EPOCHS:
                p = ck[(arm, e)]
                ob = p['occupation_basis'].to(torch.float64); ib = p['income_basis'].to(torch.float64)
                r = {'occupation_basis_bitwise': bool(torch.equal(ob, B)), 'income_basis_bitwise': bool(torch.equal(ib, Bi)),
                     'scale': float(p['scale']), 'rank': int(p['rank']), 'source_sha_ok': p['source_checkpoint_sha256'] == cfg['checkpoint']['sha256'],
                     'arm_epoch_ok': p['arm'] == arm and int(p['epoch']) == e, 'R_shape': list(p['R'].shape), 'R_dtype': str(p['R'].dtype)}
                with torch.no_grad():
                    zi = src.final(0, src.hidden(0, fV['X']))
                    iw = C.raw_proj_wire(zi, ib, W0, b0)['full'].numpy()
                r['income_wire_validation_bitwise'] = bool(np.array_equal(iw, incV))
                if e == 0:
                    r['R_b_all_zero'] = bool(not p['R'].any() and not p['b_res'].any())
                    with torch.no_grad():
                        diffs = {}
                        for rn, F in roles.items():
                            w = C.np_wire(C.occ_wire(F['hbar'], F['q0'], p['R'].to(torch.float64), p['b_res'].to(torch.float64), ob, W1, b1))
                            diffs[rn] = {'bitwise': bool(np.array_equal(w['full'], raw[rn]['full'])), **C.wire_diff(w, raw[rn])}
                    r['epoch0_vs_my_RAW_PROJ'] = diffs
                arm_rec[str(e)] = r
            # which payload keys differ across epochs
            base = ck[(arm, 0)]; differ = set()
            for e in EPOCHS[1:]:
                p = ck[(arm, e)]
                if set(p) != set(base):
                    differ.add('KEYSET')
                for k in base:
                    a, b_ = base[k], p.get(k)
                    if torch.is_tensor(a):
                        if not (torch.is_tensor(b_) and a.shape == b_.shape and torch.equal(a, b_)):
                            differ.add(k)
                    elif C.canonical(jsonable_safe(a)) != C.canonical(jsonable_safe(b_)):
                        differ.add(k)
            arm_rec['keys_differing_from_epoch0'] = sorted(differ)
            cks[arm] = arm_rec
        out['checkpoints'] = cks
        return out

    # ------------------------------------------------------------ A2 metrics & budget audit
    def part_a2():
        guard()
        out = {}
        seed_summary = json.loads(V(cfg['seed_summary']).read_text())
        sp = seed_summary['start_pool']
        out['start_vs_seed_summary_maxabs'] = max(abs(start[k] - sp[k]) for k in ('nll', 'brier', 'kl', 'ce'))
        out['start_correct_equal'] = start['correct'] == sp['correct']
        for arm in ARMS:
            S = summ[arm]; tr = traces[arm]; trust = TRUST[arm]
            log = [json.loads(l) for l in V(cfg['training'][arm]['update_log']).read_text().splitlines() if l.strip()]
            budget = json.loads(V(cfg['training'][arm]['budget_trace']).read_text())
            a = {'trace_shape': list(tr.shape), 'trace0_zero': bool(not tr[0].any())}
            # checkpoint metrics vs SUMMARY diagnostics
            diag = {d['epoch']: d for d in S['label_free_and_fitting_pool_diagnostics']}
            md = 0.0; cm = 0; trace_ck = True
            for e in EPOCHS:
                v = ckvec(arm, e)
                trace_ck &= bool(np.array_equal(v, tr[32 * e]))
                if e == 0:
                    st = S['anchor']['start']
                    md = max(md, abs(start['nll'] - st['nll']), abs(start['brier'] - st['brier']))
                    cm += int(start['correct'] != st['correct'])
                    continue
                for which, key in (('P', 'TASK_POOL'), ('U', 'UPDATE'), ('G', 'GUARD')):
                    m = metrics(v, which); ref = diag[e]['fitting_pool'][key]
                    md = max(md, *(abs(m[k] - ref[k]) for k in ('nll', 'brier', 'kl', 'ce', 'accuracy')))
                    cm += int(m['correct'] != ref['correct'])
            a['checkpoint_metric_maxabs_vs_summary'] = md; a['checkpoint_correct_mismatches'] = cm
            a['checkpoints_equal_trace'] = trace_ck
            # anchor
            an = S['anchor']
            a['anchor_vs_mine_maxabs'] = max(abs(an['nll_limit'] - limits['nll']), abs(an['brier_limit'] - limits['brier']))
            a['anchor_correct_floor_equal'] = an['correct_floor'] == limits['correct']
            a['anchor_kl_limit'] = an['kl_limit']; a['anchor_trust'] = an['trust']
            consts = {(b['nll_limit'], b['brier_limit'], b['correct_floor'], b['kl_limit']) for b in budget}
            a['budget_trace_limits_constant'] = len(consts) == 1
            a['budget_trace_limits'] = list(consts)[0] if len(consts) == 1 else None
            # every state
            mets = [metrics(tr[n]) for n in range(len(tr))]
            statuses = {}; cur_d = 0.0; post_d = 0.0; cur_c = 0; post_c = 0; viol = []; norm_viol = 0; nonacc_moved = 0
            norm_d = 0.0; max_kl = 0.0; max_nll = -1; max_brier = -1; min_correct = 10 ** 9; slack_len = set()
            for rec in log:
                n = rec['update_number']; r = rec['result']; stt = rec['status']
                statuses[stt] = statuses.get(stt, 0) + 1
                cur = r['current']; mine = mets[n - 1]
                cur_d = max(cur_d, abs(cur['nll'] - mine['nll']), abs(cur['brier'] - mine['brier']),
                            abs((cur['kl'] or 0) - mine['kl']) if cur.get('kl') is not None else 0.0)
                cur_c += int(cur['correct'] != mine['correct'])
                slack_len.add(len(r['raw_slacks']))
                step = tr[n] - tr[n - 1]
                if stt == 'ACCEPTED':
                    m = mets[n]; pm = r['post_metrics']
                    post_d = max(post_d, *(abs(pm[k] - m[k]) for k in ('nll', 'brier', 'kl')))
                    post_c += int(pm['correct'] != m['correct'])
                    ok = m['nll'] <= limits['nll'] + C.TOL and m['brier'] <= limits['brier'] + C.TOL and m['correct'] >= limits['correct']
                    if trust:
                        ok = ok and m['kl'] <= limits['kl'] + C.TOL
                    if not ok:
                        viol.append(n)
                    sn = float(np.linalg.norm(step))
                    norm_viol += int(sn > C.MAX_NORM)
                    norm_d = max(norm_d, abs(sn - rec['actual_norm']))
                    max_kl = max(max_kl, m['kl']); max_nll = max(max_nll, m['nll']); max_brier = max(max_brier, m['brier'])
                    min_correct = min(min_correct, m['correct'])
                else:
                    nonacc_moved += int(np.any(step))
            a.update({'status_counts_mine': statuses, 'status_counts_summary': S['update_status_counts'],
                      'status_counts_match': statuses == S['update_status_counts'], 'updates': len(log),
                      'current_metric_maxabs_vs_log': cur_d, 'current_correct_mismatches': cur_c,
                      'post_metric_maxabs_vs_log': post_d, 'post_correct_mismatches': post_c,
                      'accepted_budget_violations_mine': viol, 'accepted_step_norm_over_0.01': norm_viol,
                      'actual_norm_maxabs_vs_log': norm_d, 'nonaccepted_steps_that_moved': nonacc_moved,
                      'max_accepted_kl': max_kl, 'max_accepted_nll': max_nll, 'max_accepted_brier': max_brier,
                      'min_accepted_correct': min_correct, 'raw_slack_lengths': sorted(slack_len),
                      'final_state_metrics': mets[-1]})
            out[arm] = a
        return out

    # ------------------------------------------------------------ A4 FIT2 point clauses
    def part_a4():
        guard()
        inc = C.npz(V(cfg['fit2']['income'])); occ = C.npz(V(cfg['fit2']['occupation'])); refs = C.npz(V(cfg['fit2']['references']))
        for z in (inc, occ, refs):
            assert np.array_equal(z['row_id'], fV['row_id'])
        y = {'income': inc['income'].astype(np.int64), 'occupation': occ['occupation'].astype(np.int64)}
        fit2 = json.loads(V(cfg['fit2']['campaign_result']).read_text())
        caps = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
        refm = {(t, r): C.prob_metrics(refs[f'{r}_{i + 1}'], y[t]) for i, t in enumerate(('income', 'occupation')) for r in ('U0', 'Ucal')}
        pinc = FV['inc']['probabilities'].numpy()
        table = {}; maxd = 0.0; flag_mis = 0

        def clauses(pocc):
            own = {'income': C.prob_metrics(pinc, y['income']), 'occupation': C.prob_metrics(pocc, y['occupation'])}
            cl = []
            for t in ('income', 'occupation'):
                for r in ('U0', 'Ucal'):
                    rm = refm[(t, r)]
                    vals = {'accuracy_loss': rm['accuracy'] - own[t]['accuracy'], 'nll_excess': own[t]['nll'] - rm['nll'],
                            'brier_excess': own[t]['brier'] - rm['brier']}
                    for mtr, cap in caps.items():
                        cl.append({'task': t, 'reference': r, 'metric': mtr, 'value': vals[mtr], 'pass': bool(vals[mtr] <= cap)})
            return cl
        services = [('RAW_PROJ', 0, None)] + [(arm, e, arm) for arm in ARMS for e in EPOCHS]
        for name, e, arm in services:
            with torch.no_grad():
                if arm is None:
                    pocc = C.raw_proj_wire(FV['z0'], B, W1, b1)['probabilities'].numpy()
                else:
                    p = ck[(arm, e)]
                    pocc = C.occ_wire(FV['hbar'], FV['q0'], p['R'].to(torch.float64), p['b_res'].to(torch.float64), B, W1, b1)['probabilities'].numpy()
            cl = clauses(pocc)
            key = f'{seed}|{name}|{e}'
            camp = fit2['table'][key]['clauses']
            cmap = {(c['task'], c['reference'], c['metric']): c for c in camp}
            for c in cl:
                cc = cmap[(c['task'], c['reference'], c['metric'])]
                c['campaign_value'] = cc['value']; c['abs_diff'] = abs(cc['value'] - c['value'])
                maxd = max(maxd, c['abs_diff']); flag_mis += int(cc['point_pass'] != c['pass'])
            table[key] = {'all_pass': all(c['pass'] for c in cl), 'campaign_all_pass': fit2['table'][key]['all_point_pass'],
                          'n_clauses': len(cl), 'failed': [c for c in cl if not c['pass']],
                          'min_headroom': min({'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}[c['metric']] - c['value'] for c in cl)}
        return {'table': table, 'clause_value_maxabs_vs_campaign': maxd, 'pass_flag_mismatches': flag_mis,
                'campaign_selection': fit2['selection']}

    for name, fn in (('A3', part_a3), ('A1', part_a1), ('A2', part_a2), ('A4', part_a4)):
        if name not in cfg['parts']:
            continue
        try:
            res['parts'][name] = fn()
        except Exception:
            res['errors'][name] = traceback.format_exc()
        dump()
    dump()


def jsonable_safe(x):
    if torch.is_tensor(x):
        return {'tensor': C.canonical(x.detach().numpy().tolist())}
    if isinstance(x, dict):
        return {str(k): jsonable_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable_safe(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    main(ap.parse_args().spec)

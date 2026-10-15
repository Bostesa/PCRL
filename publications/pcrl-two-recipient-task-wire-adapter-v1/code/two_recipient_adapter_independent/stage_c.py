"""Stage C: J_BOTH counterfactual re-solve with the independent projection solver and own finite ladder."""
from __future__ import annotations
import argparse
import json

from .common import bind, Source, bases, write_out
from .pool import load_pool, pool_metrics, teacher, split, probe_wire
from .solver import solve

R_MAX = 0.01
LADDER = (1.0, 0.5, 0.25, 0.125)
TOL = 1e-10
NAMES = ('income', 'occupation')


def ladder(src, B, i, f, lp0, vec0, d, An, bn, anchor, rank):
    import numpy as np
    for frac in LADDER:
        scale = 1.0
        for att in range(16):
            vec = vec0 + frac * scale * d
            actual = vec - vec0
            an = float(np.linalg.norm(actual))
            if an <= R_MAX or not np.isfinite(an):
                break
            scale *= min(1.0, R_MAX / an) * (1 - 1e-6)
        bounded = bool(np.isfinite(an) and an <= R_MAX)
        lin = (not len(An)) or bool(np.max(An @ actual - bn) <= TOL)
        R, b = split(vec, rank)
        m = pool_metrics(src, B, i, f, R, b, lp0)
        inside = (m['nll'] <= anchor['nll_limit'] + TOL and m['brier'] <= anchor['brier_limit'] + TOL
                  and m['correct'] >= anchor['correct_floor'] and m['kl'] <= anchor['kl_limit'] + TOL)
        if bounded and lin and inside:
            return actual, frac, m
    return np.zeros_like(vec0), None, None


def run(spec):
    import numpy as np
    import torch
    import torch.nn.functional as F
    active, cfg, get, guard = bind(spec)
    res = {'schema': 'TRA1-independent-stage-C-v1', 'unit_id': active['unit_id'], 'seeds': {}}
    adm = json.loads(get(cfg['admission']).read_text())
    wtol = adm['wire_tolerance']
    ok_all = True
    for seed in ('0', '1'):
        guard()
        S = cfg['seeds'][seed]
        src = Source(get(S['checkpoint']))
        B, _ = bases(get(S['geometry']))
        binding = json.loads(get(S['adapter_binding']).read_text())
        scales = [float(x) for x in binding['scales']]
        tcfg = json.loads(get(S['train_config']).read_text())
        feats, probe, data = load_pool(tcfg, get, src, B, scales)
        split_meta = json.loads(get(S['role_split']).read_text())
        import hashlib
        probe_ok = hashlib.sha256(json.dumps(probe).encode()).hexdigest() == split_meta['probe_position_sha256']
        lp0 = [teacher(src, B, i, feats[i]) for i in (0, 1)]
        ranks = [int(b.shape[0]) for b in B]
        states = torch.load(get(S['counterfactual_states']), map_location='cpu', weights_only=False)
        summ = json.loads(get(S['summary']).read_text())
        trace = dict(np.load(get(S['state_trace'])))
        log = [json.loads(l) for l in get(S['update_log']).read_text().splitlines() if l.strip()]
        sel_label = cfg['nominated_label']; sel_epoch = int(sel_label.lstrip('E'))
        nU, nG = 8180, 2043; wU, wG = nU / (nU + nG), nG / (nU + nG)
        out_states = []; fails = []
        for k, st in enumerate(states):
            guard()
            i = int(st['recipient']); name = NAMES[i]; f = feats[i]; rank = ranks[i]
            vec0 = np.concatenate([st['pre_R'].reshape(-1).numpy(), st['pre_b'].numpy()])
            R0, b0 = split(vec0, rank)
            rec = {'index': k, 'recipient': name, 'opportunity': st['opportunity'], 'epoch': st['epoch'], 'phase': st['phase'],
                   'scan_kinds': st['scan_kinds']}
            # trace consistency of the saved pre-state (both recipients)
            rec['pre_state_matches_trace'] = bool(np.array_equal(vec0, trace[name][st['opportunity'] - 1])
                                                  and np.array_equal(st['other_pre'].numpy(), trace[NAMES[1 - i]][st['opportunity'] - 1]))
            # own pool metrics/gradients at the pre-state
            cur = pool_metrics(src, B, i, f, R0, b0, lp0[i], grads=True)
            pc = st['pool_current']
            rec['pool_metric_max_abs_diff'] = max(abs(cur['nll'] - pc['nll']), abs(cur['brier'] - pc['brier']), abs(cur['kl'] - pc['kl']))
            rec['pool_correct_equal'] = cur['correct'] == pc['correct']
            gd = {}
            for key in ('nll', 'brier', 'kl'):
                gs = st['pool_gradients'][key].numpy()
                gd[key] = float(np.max(np.abs(cur['gradients'][key] - gs)) / max(1e-300, float(np.max(np.abs(gs))))) if np.any(gs) else float(np.max(np.abs(cur['gradients'][key])))
            rec['pool_gradient_rel_diff'] = gd
            # own task gradient (UPDATE minibatch CE + whole-GUARD CE, population weights)
            rows = list(st['rows'])
            Rg = R0.clone().requires_grad_(True); bg = b0.clone().requires_grad_(True)
            from .pool import logits_of
            lg, _ = logits_of(src, B, i, f['hU'][rows], f['qU'][rows], Rg, bg)
            ce = -F.log_softmax(lg, 1)[torch.arange(len(rows)), f['yU'][rows]].mean()
            gmb = torch.autograd.grad(ce, (Rg, bg)); gmb = torch.cat([gmb[0].reshape(-1), gmb[1].reshape(-1)]).numpy()
            Rg = R0.clone().requires_grad_(True); bg = b0.clone().requires_grad_(True)
            gsum = None
            for s0 in range(0, nG, 1024):
                sl = slice(s0, s0 + 1024)
                lG, _ = logits_of(src, B, i, f['hG'][sl], f['qG'][sl], Rg, bg)
                c = -F.log_softmax(lG, 1)[torch.arange(lG.shape[0]), f['yG'][sl]].sum()
                gg = torch.autograd.grad(c, (Rg, bg)); gg = torch.cat([gg[0].reshape(-1), gg[1].reshape(-1)]).numpy()
                gsum = gg if gsum is None else gsum + gg
            gguard = gsum / nG
            g_task = wU * gmb + wG * gguard
            gts = st['g_task'].numpy()
            rec['task_gradient_max_abs_diff'] = float(np.max(np.abs(g_task - gts)))
            rec['guard_ce_gradient_max_abs_diff'] = float(np.max(np.abs(gguard - st['guard_ce_gradient'].numpy())))
            prop = st['proposal'].numpy()
            rec['proposal_formula_max_abs'] = float(np.max(np.abs(prop - (-0.01 * (gts + st['coefficient'] * st['g_privacy'].numpy())))))
            noise = 0.01 * rec['task_gradient_max_abs_diff']
            ctol = max(1e-10, 10 * noise)
            # slacks from the anchor limits minus the recorded pool state; gradients as recorded
            an = st['anchor']
            raw = [an['nll_limit'] - pc['nll'], an['brier_limit'] - pc['brier'], an['kl_limit'] - pc['kl']]
            if min(raw) < -TOL:
                rec['infeasible_current'] = True
            slack = np.array([max(0.0, v) for v in raw])
            A = np.vstack([st['pool_gradients'][kk].numpy() for kk in ('nll', 'brier', 'kl')])
            out = {}
            for tag, p in (('privacy', prop), ('counterfactual', -0.01 * gts)):
                d, info = solve(p, A, slack, R_MAX)
                An = info.get('normalized_rows', np.zeros((0, len(p)))); bn = info.get('normalized_slacks', np.zeros(0))
                actual, frac, m = ladder(src, B, i, f, lp0[i], vec0, d, An, bn, an, rank)
                out[tag] = {'actual': actual, 'fraction': frac, 'post': m, 'active_set': info['active_set'], 'ball_active': info.get('ball_active'),
                            'projected_norm': float(np.linalg.norm(d)), 'proposal_norm': float(np.linalg.norm(p))}
            ap = st['actual_privacy_displacement'].numpy(); ac = st['actual_counterfactual_displacement'].numpy()
            rec['privacy_actual_max_abs_vs_stored'] = float(np.max(np.abs(out['privacy']['actual'] - ap)))
            rec['counterfactual_actual_max_abs_vs_stored'] = float(np.max(np.abs(out['counterfactual']['actual'] - ac)))
            rec['privacy_fraction_own'] = out['privacy']['fraction']; rec['privacy_fraction_stored'] = st.get('privacy_accepted_fraction')
            rec['counterfactual_fraction_own'] = out['counterfactual']['fraction']; rec['counterfactual_fraction_stored'] = st.get('counterfactual_fraction')
            rec['privacy_active_set_own'] = out['privacy']['active_set']; rec['privacy_active_set_stored'] = st.get('active_set')
            rec['counterfactual_active_set_own'] = out['counterfactual']['active_set']
            rec['privacy_ball_active_own'] = out['privacy']['ball_active']; rec['counterfactual_ball_active_own'] = out['counterfactual']['ball_active']
            rec['active_halfspace_or_shortened_own'] = bool(out['privacy']['active_set'] or (out['privacy']['fraction'] or 1) < 1)
            rec['stored_active_budget_case'] = bool(st.get('active_budget_case'))
            # log record agreement for the privacy step
            lr = next((x for x in log if x['macro_step'] == st['macro_step']), None)
            if lr is not None:
                e = lr['recipients'][name]
                pm = e['result'].get('post_metrics') or {}
                if out['privacy']['post'] is not None and pm:
                    rec['privacy_post_metrics_max_abs_vs_log'] = max(abs(out['privacy']['post'][kk] - pm[kk]) for kk in ('nll', 'brier', 'kl'))
                    rec['privacy_post_correct_equal_log'] = out['privacy']['post']['correct'] == pm['correct']
            diff = out['privacy']['actual'] - out['counterfactual']['actual']
            rec['privacy_minus_counterfactual_max_abs'] = float(np.max(np.abs(diff)))
            rec['contribution_tolerance_own'] = ctol
            rec['displacement_changed_beyond_noise'] = bool(np.max(np.abs(diff)) > ctol)
            Rp, bp = split(vec0 + out['privacy']['actual'], rank); Rc, bc = split(vec0 + out['counterfactual']['actual'], rank)
            wp = probe_wire(src, B, i, f, probe, Rp, bp); wc = probe_wire(src, B, i, f, probe, Rc, bc)
            wd = {kk: float(np.max(np.abs(wp[kk] - wc[kk]))) for kk in ('representation', 'centered_logits', 'probabilities')}
            wd['decision_changes'] = int((wp['decision'] != wc['decision']).sum())
            rec['probe_wire_difference'] = wd
            rec['probe_wire_changed_beyond_tolerance'] = bool(any(wd[kk] > wtol[i][kk] for kk in ('representation', 'centered_logits', 'probabilities')) or wd['decision_changes'] > 0)
            rec['at_or_before_nominated_checkpoint'] = bool(st['phase'] == 1 and st['epoch'] <= sel_epoch)
            rec['qualifies'] = bool(rec['displacement_changed_beyond_noise'] and rec['probe_wire_changed_beyond_tolerance'] and rec['at_or_before_nominated_checkpoint'])
            # campaign scan record agreement
            scans = summ['counterfactual_scan'][name] + summ['active_budget_scan'][name]
            sc = next((x for x in scans if x['opportunity'] == st['opportunity']), None)
            if sc is not None:
                rec['campaign_record_agrees'] = bool(sc['privacy_changes_deployed_wire_beyond_tolerance'] == rec['probe_wire_changed_beyond_tolerance']
                                                     and sc['privacy_contribution_beyond_noise'] == rec['displacement_changed_beyond_noise'])
            ok = (rec['pre_state_matches_trace'] and rec['pool_metric_max_abs_diff'] <= 1e-10 and rec['pool_correct_equal']
                  and max(gd.values()) <= 1e-8 and rec['task_gradient_max_abs_diff'] <= 1e-10 and rec['proposal_formula_max_abs'] <= 1e-15
                  and rec['privacy_actual_max_abs_vs_stored'] <= 1e-12 and rec['counterfactual_actual_max_abs_vs_stored'] <= 1e-12
                  and rec['privacy_fraction_own'] == rec['privacy_fraction_stored'] and rec['counterfactual_fraction_own'] == rec['counterfactual_fraction_stored'])
            rec['reproduced'] = bool(ok)
            if not ok:
                fails.append(f'S{seed} state {k} ({name} opp {st["opportunity"]}) not reproduced')
            out_states.append(rec)
        movement = {}
        for i, name in enumerate(NAMES):
            first8 = [r for r in out_states if r['recipient'] == name and 'first_eight' in r['scan_kinds']]
            act = [r for r in out_states if r['recipient'] == name and 'active_budget' in r['scan_kinds']]
            movement[name] = {'first_eight_states': len(first8), 'qualifying': sum(r['qualifies'] for r in first8),
                              'meets_6_of_8': bool(len(first8) == 8 and sum(r['qualifies'] for r in first8) >= 6),
                              'epochs': [r['epoch'] for r in first8],
                              'active_budget_states_saved': len(act),
                              'states_with_active_halfspace_or_shortened_own': sum(r['active_halfspace_or_shortened_own'] for r in out_states if r['recipient'] == name),
                              'counterfactual_states_with_active_halfspace_own': sum(bool(r['counterfactual_active_set_own']) or (r['counterfactual_fraction_own'] or 1) < 1 for r in out_states if r['recipient'] == name),
                              'active_budget_states_qualifying_privacy_moves_wire': sum(r['probe_wire_changed_beyond_tolerance'] and r['displacement_changed_beyond_noise'] for r in act),
                              'all_states_at_or_before_nominated': all(r['at_or_before_nominated_checkpoint'] for r in out_states if r['recipient'] == name),
                              'summary_active_budget_cases': summ['active_budget_cases'][name]}
        sr = {'probe_positions_match_role_split': probe_ok, 'n_states': len(out_states), 'movement': movement, 'states': out_states,
              'failures': fails + ([] if probe_ok else ['probe positions differ'])}
        sr['pass'] = not sr['failures']
        ok_all &= sr['pass']
        res['seeds'][seed] = sr
    res['movement_ok_all'] = all(res['seeds'][s]['movement'][n]['meets_6_of_8'] for s in ('0', '1') for n in NAMES)
    res['pass'] = bool(ok_all)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))

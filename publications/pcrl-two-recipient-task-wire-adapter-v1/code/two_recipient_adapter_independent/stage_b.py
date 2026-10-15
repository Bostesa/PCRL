"""Stage B: training traces. Budgets (from BUDGET_TRACE and from own full-pool recomputation of every recorded state),
anchor constancy, sequential freezing, I_ONLY occupation zero, opportunities and critic rounds."""
from __future__ import annotations
import argparse
import json

from .common import bind, Source, bases, write_out, tsha
from .pool import load_pool, pool_metrics, teacher, split

TOL = 1e-10
ARMS = ('T_BOTH', 'L_BOTH', 'J_BOTH', 'S12', 'S21', 'I_ONLY')
PHASES = {'T_BOTH': [[0, 1]], 'L_BOTH': [[0, 1]], 'J_BOTH': [[0, 1]], 'S12': [[0], [1]], 'S21': [[1], [0]], 'I_ONLY': [[0]]}
ROUNDS = {'T_BOTH': 3280, 'L_BOTH': 3280, 'J_BOTH': 3280, 'S12': 3280, 'S21': 3280, 'I_ONLY': 1680}
NAMES = ('income', 'occupation')


def run(spec):
    import numpy as np
    import torch
    active, cfg, get, guard = bind(spec)
    res = {'schema': 'TRA1-independent-stage-B-v1', 'unit_id': active['unit_id'], 'seeds': {}}
    ok_all = True
    for seed in ('0', '1'):
        guard()
        S = cfg['seeds'][seed]; fails = []
        src = Source(get(S['checkpoint']))
        B, _ = bases(get(S['geometry']))
        binding = json.loads(get(S['adapter_binding']).read_text())
        scales = [float(x) for x in binding['scales']]
        tcfg = json.loads(get(S['train_config']).read_text())
        feats, probe, _ = load_pool(tcfg, get, src, B, scales)
        lp0 = [teacher(src, B, i, feats[i]) for i in (0, 1)]
        ranks = [int(b.shape[0]) for b in B]
        start = [pool_metrics(src, B, i, feats[i], torch.zeros(ranks[i], 64, dtype=torch.float64), torch.zeros(ranks[i], dtype=torch.float64), lp0[i]) for i in (0, 1)]
        seedrec = {'own_start_metrics': start, 'arms': {}}
        for arm in ARMS:
            guard()
            A = S['arms'][arm]
            summ = json.loads(get(A['summary']).read_text())
            budget = json.loads(get(A['budget_trace']).read_text())
            trace = dict(np.load(get(A['state_trace'])))
            log = [json.loads(l) for l in get(A['update_log']).read_text().splitlines() if l.strip()]
            cks = {lab: torch.load(get(it), map_location='cpu', weights_only=False) for lab, it in A['checkpoints'].items()}
            r = {'failures': []}
            # anchors: constant per recipient, equal to own start + budgets
            for i, n in enumerate(NAMES):
                rows = [x for x in budget if x['recipient'] == n]
                lim = {(x['nll_limit'], x['brier_limit'], x['correct_floor'], x['kl_limit']) for x in rows}
                act = sum(i in ph for ph in PHASES[arm])
                exp_lim = (start[i]['nll'] + 0.004, start[i]['brier'] + 0.002, start[i]['correct'] - int(0.005 * 10223 + 1e-9), 0.001)
                r[f'{n}_rows'] = len(rows)
                if act:
                    if len(lim) != 1:
                        r['failures'].append(f'{n} anchor not constant ({len(lim)} distinct)')
                    else:
                        l0 = next(iter(lim))
                        d = [abs(l0[0] - exp_lim[0]), abs(l0[1] - exp_lim[1]), abs(l0[2] - exp_lim[2]), abs(l0[3] - exp_lim[3])]
                        r[f'{n}_anchor_vs_own_start_abs_diff'] = d
                        if max(d) > 1e-12:
                            r['failures'].append(f'{n} anchor differs from own zero-residual start + budgets')
                if len(rows) != 320 * act:
                    r['failures'].append(f'{n} opportunities {len(rows)} != {320 * act}')
                bad = 0; acc = 0
                for x in rows:
                    if x['status'] == 'ACCEPTED':
                        acc += 1
                        if not (x['post_nll'] <= x['nll_limit'] + TOL and x['post_brier'] <= x['brier_limit'] + TOL
                                and x['post_correct'] >= x['correct_floor'] and x['post_kl'] <= x['kl_limit'] + TOL):
                            bad += 1
                    elif x['actual_norm'] != 0.0:
                        bad += 1
                    if x['actual_norm'] > 0.01:
                        bad += 1
                r[f'{n}_accepted'] = acc; r[f'{n}_trace_budget_violations'] = bad
                if bad:
                    r['failures'].append(f'{n} {bad} recorded post-step budget violations')
                # continuity: next current == previous post (accepted) / previous current (rejected)
                cont = 0
                for a, b in zip(rows, rows[1:]):
                    prevm = (a['post_nll'], a['post_brier'], a['post_correct'], a['post_kl']) if a['status'] == 'ACCEPTED' else (a['current_nll'], a['current_brier'], a['current_correct'], a['current_kl'])
                    curm = (b['current_nll'], b['current_brier'], b['current_correct'], b['current_kl'])
                    if b['phase'] == a['phase'] and (abs(prevm[0] - curm[0]) > 1e-12 or abs(prevm[1] - curm[1]) > 1e-12 or prevm[2] != curm[2] or abs(prevm[3] - curm[3]) > 1e-12):
                        cont += 1
                r[f'{n}_continuity_breaks'] = cont
                if cont:
                    r['failures'].append(f'{n} metric continuity breaks {cont}')
                # own recomputation of every state in the trace
                tr = trace[n]
                r[f'{n}_trace_states'] = int(len(tr))
                if len(tr) != 1 + 320 * act:
                    r['failures'].append(f'{n} trace length {len(tr)}')
                worst = {'nll': -1e9, 'brier': -1e9, 'kl': -1e9, 'correct_margin': 1e9}
                own_bad = 0; maxdiff = 0.0
                post_by_opp = {x['opportunity']: x for x in rows}
                for k, v in enumerate(tr):
                    R, b = split(v, ranks[i])
                    m = pool_metrics(src, B, i, feats[i], R, b, lp0[i])
                    worst['nll'] = max(worst['nll'], m['nll'] - exp_lim[0]); worst['brier'] = max(worst['brier'], m['brier'] - exp_lim[1])
                    worst['kl'] = max(worst['kl'], m['kl'] - 0.001); worst['correct_margin'] = min(worst['correct_margin'], m['correct'] - exp_lim[2])
                    if m['nll'] > exp_lim[0] + TOL or m['brier'] > exp_lim[1] + TOL or m['kl'] > 0.001 + TOL or m['correct'] < exp_lim[2]:
                        own_bad += 1
                    if k > 0 and post_by_opp.get(k, {}).get('status') == 'ACCEPTED':
                        x = post_by_opp[k]
                        maxdiff = max(maxdiff, abs(m['nll'] - x['post_nll']), abs(m['brier'] - x['post_brier']), abs(m['kl'] - x['post_kl']), abs(m['correct'] - x['post_correct']))
                    if k > 0 and post_by_opp.get(k, {}).get('status') != 'ACCEPTED' and not np.array_equal(tr[k], tr[k - 1]):
                        own_bad += 1
                r[f'{n}_own_recomputed_worst_margin'] = worst
                r[f'{n}_own_recomputed_violations'] = own_bad
                r[f'{n}_own_vs_recorded_post_max_abs'] = maxdiff
                if own_bad:
                    r['failures'].append(f'{n} own recomputation finds {own_bad} out-of-budget/changed-on-reject states')
                if maxdiff > 1e-9:
                    r['failures'].append(f'{n} own recomputation differs from recorded post metrics by {maxdiff:.2e}')
            # checkpoints vs trace / freezing
            def st(lab, i):
                p = cks[lab]['states'][i]
                return np.concatenate([p['R'].reshape(-1).numpy(), p['b'].numpy()])
            seq = arm in ('S12', 'S21')
            for i, n in enumerate(NAMES):
                tr = trace[n]
                if arm == 'I_ONLY' and i == 1:
                    allz = all(not np.any(st(l, 1)) for l in cks) and len(tr) == 1 and not np.any(tr[0])
                    r['I_ONLY_occupation_zero_everywhere'] = bool(allz)
                    if not allz:
                        r['failures'].append('I_ONLY occupation not zero')
                    continue
                if not seq:
                    for e in (0, 1, 3, 5, 10):
                        if not np.array_equal(st(f'E{e}', i), tr[32 * e]):
                            r['failures'].append(f'{n} E{e} checkpoint != trace[{32 * e}]')
                else:
                    first = PHASES[arm][0][0]
                    if i == first:
                        for e in (0, 1, 3, 5, 10):
                            if not np.array_equal(st(f'P1E{e}', i), tr[32 * e]):
                                r['failures'].append(f'{n} P1E{e} != trace')
                        frozen = all(np.array_equal(st(f'P2E{e}', i), tr[-1]) for e in (0, 1, 3, 5, 10))
                        r[f'first_recipient_{n}_bitwise_frozen_phase2'] = bool(frozen)
                        if not frozen:
                            r['failures'].append(f'first recipient {n} changed in phase 2')
                    else:
                        zero1 = all(not np.any(st(f'P1E{e}', i)) for e in (0, 1, 3, 5, 10)) and not np.any(st('P2E0', i)) and not np.any(tr[0])
                        r[f'second_recipient_{n}_zero_phase1'] = bool(zero1)
                        if not zero1:
                            r['failures'].append(f'second recipient {n} nonzero in phase 1')
                        for e in (0, 1, 3, 5, 10):
                            if not np.array_equal(st(f'P2E{e}', i), tr[32 * e]):
                                r['failures'].append(f'{n} P2E{e} != trace')
            # log-level freezing (inactive recipients never logged in a phase) and critic rounds
            for rec in log:
                ph = PHASES[arm][rec['phase'] - 1]
                if set(rec['recipients']) != {NAMES[i] for i in ph}:
                    r['failures'].append('log lists an inactive recipient'); break
            r['critic_rounds'] = summ['critic_rounds']; r['critic_optimizer_steps'] = summ['critic_optimizer_steps']
            r['log_last_critic_rounds'] = log[-1]['critic_rounds'] if log else None
            r['macro_steps'] = len(log)
            if summ['critic_rounds'] != ROUNDS[arm] or summ['critic_optimizer_steps'] != 9 * ROUNDS[arm] or log[-1]['critic_rounds'] != ROUNDS[arm]:
                r['failures'].append(f'critic rounds {summ["critic_rounds"]} != {ROUNDS[arm]}')
            if len(log) != 320 * len(PHASES[arm]):
                r['failures'].append('macro steps differ')
            opp = summ['opportunities']
            r['opportunities'] = opp
            for i, n in enumerate(NAMES):
                if opp[n] != 320 * sum(i in ph for ph in PHASES[arm]):
                    r['failures'].append(f'{n} opportunities {opp[n]}')
            r['status_counts'] = summ['update_status_counts']
            r['active_budget_cases'] = summ['active_budget_cases']
            r['pass'] = not r['failures']
            fails += [f'{arm}: {x}' for x in r['failures']]
            seedrec['arms'][arm] = r
        seedrec['pass'] = not fails
        ok_all &= seedrec['pass']
        res['seeds'][seed] = seedrec
    res['pass'] = bool(ok_all)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))

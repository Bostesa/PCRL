"""Unit IHS_REPORT: the ONE logged coordinator follow-up evaluation of the already-exposed DEV_REPORT truth for this campaign's
frozen successor (after the committed and published REPORTING_LOCK), both scorecards in one run. `--replay` is the
independent verifier's separately logged replay path (its own event name; the verifier also recomputes with its own code).

Census context: public-use ACS 2019 Florida person records; readers estimate the recorded SEX column from the released income
and occupation predictions (each output alone and the linked pair). Exploratory development evidence, not confirmation.

- ENHANCED scorecard (the final standard): enhanced winners (source 8 + new 8 readers; ATTACK_VALIDATION selection).
- ORIGINAL scorecard (historical comparison): the source-slate winners fitted in this campaign.
- Reader-free utility (24 clauses) and within-SEX guard (4) endpoints are identical in both scorecards.
- Household bootstrap (B and seed from ANALYSIS_LOCK; G = 3955) shared across releases, endpoints, scorecards and seeds;
  paired differences within replicate; Wald bounds with the registered critical values (amd.report.crit / endpoint).
- Endpoints and claim conjunctions come only from the committed ANALYSIS_LOCK (written with generic method roles before the
  finalist was frozen); nothing here chooses a claim after seeing results.
- Private outputs: per-replicate endpoint draws and per-household utility contributions (for the feasibility reassessment).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from amd.report import crit, endpoint
from task_headroom.inference import group_weights
from task_conditioned_representation.inference import auc, bootstrap_auc
from . import common as C
from . import final_readers as FR

VIEWS = ('income', 'occupation', 'pair')
SCORECARDS = ('ENHANCED', 'ORIGINAL')


def _scores(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def claims_engine(lock_claims, eps, CV):
    """Generic conjunction engine: a claim holds when every listed condition holds; 'sequence' claims stop at the first
    failing step; conditions may reference earlier claims. Tests: clause (registered clause bound), point_le (point <= margin),
    clause_at_0020 (clause lower bound > 0.020), cv (controls inside their bands)."""
    def ev(c, done):
        if 'claim_ref' in c:
            ok = bool(done[c['claim_ref']]['holds'])
            return {'condition': f"claim {c['claim_ref']}", 'holds': ok, 'category': 'PASS' if ok else 'POINT_FAILURE'}
        if c['test'] == 'cv':
            return {'condition': 'CV (planted and shuffled controls inside their bands, every view and seed)', 'holds': CV,
                    'category': 'PASS' if CV else 'CONTROL_FAILURE'}
        ep = eps[c['endpoint']]
        if c['test'] == 'point_le':
            ok = bool(ep['point'] <= ep['margin'])
        elif c['test'] == 'clause':
            ok = bool(ep['clause_pass'])
        elif c['test'] == 'clause_at_0020':
            ok = ep['clause_L1'] is not None and bool(ep['clause_L1'] > 0.020)
        else:
            raise ValueError('unregistered condition test')
        return {'condition': c['endpoint'], 'test': c['test'], 'gate': c.get('gate'), 'holds': ok, 'category': 'PASS' if ok else ep['category'],
                'point': ep['point'], 'se': ep['se'], 'clause_L1': ep.get('clause_L1'), 'clause_U1': ep.get('clause_U1'),
                'family': [ep.get('family_lower'), ep.get('family_upper')]}

    done = {}
    for cl in lock_claims:
        name = cl['name']
        if cl['kind'] in ('not_claimed', 'descriptive'):
            done[name] = {'label': cl['label'], 'holds': None}
            continue
        if cl['kind'] == 'sequence':
            steps, stop = [], False
            for st in cl['steps']:
                if stop:
                    steps.append({'name': st['name'], 'status': 'NOT_TESTED_SEQUENCE_STOPPED', 'holds': False}); continue
                conds = [ev(c, done) for c in st['conditions']]
                ok = all(c['holds'] for c in conds)
                steps.append({'name': st['name'], 'holds': ok, 'conditions': conds, 'first_failing': next((c for c in conds if not c['holds']), None)})
                stop = not ok
            n_ok = sum(1 for s_ in steps if s_['holds'])
            done[name] = {'label': cl['labels'][str(n_ok)], 'holds': n_ok == len(steps), 'steps': steps}
            continue
        conds = [ev(c, done) for c in cl['conditions']]
        ok = all(c['holds'] for c in conds)
        failed = [c for c in conds if not c['holds']]
        lab = cl['success_label'] if ok else cl['failure_label']
        if not ok and cl.get('failure_labels_by_gate'):
            order = cl.get('failure_order', [])
            failed.sort(key=lambda c: order.index(c.get('gate')) if c.get('gate') in order else len(order))
            lab = cl['failure_labels_by_gate'].get(failed[0].get('gate'), lab)
        done[name] = {'label': lab, 'holds': ok, 'conditions': conds, 'failed_in_order': [c['condition'] for c in failed]}
    return done


def run(out_dir, truth=None, priv_dir=None, event_name='DEV_REPORT_TRUTH_REREAD_BY_IHS_COORDINATOR', purpose=None):
    names = ('FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json', 'REPORTING_LOCK.json')
    if truth is None:
        for n in names:
            if not C.committed(C.RESULTS / n):
                raise PermissionError(f'report requires the committed {n}')
    out = Path(out_dir); priv = Path(priv_dir) if priv_dir else C.RUNS / 'REPORT'
    if (out / 'DEV_REPORT_RESULTS.json').exists() or (priv / 'ENDPOINT_DRAWS.npz').exists():
        raise PermissionError('report results exist (one evaluation)')
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    rlock = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text()) if truth is None else None
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    inf = lock['inference']
    T = truth if truth is not None else C.dev_report_truth(purpose or
        'income-headroom successor campaign: the one coordinator evaluation of the frozen successor, bank and readers', event_name)
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    if G != 3955 or len(sex) != 5943:
        raise SystemExit(f'HALT: DEV_REPORT has {G} households and {len(sex)} persons, not 3955 / 5943')
    Wsum = W.sum(1)
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cK = {fam: crit(G, 0.05 / (2 * k)) for fam, k in inf['K'].items()}; cm = crit(G, 0.025)
    for nm, val in (('c1', c1), *((f'c_{f}', v) for f, v in cK.items()), ('marginal', cm)):
        exp = inf['critical_values_expected'].get(nm)
        if exp is not None and abs(val - exp) > 1e-6:
            raise AssertionError(f'critical value {nm} {val} differs from the registered {exp}')
    res = {'schema': 'IHS1-dev-report-v1', 'utc': C.now(), 'event': event_name,
           'status': 'EXPLORATORY DEVELOPMENT EVIDENCE (already-exposed DEV_REPORT people; previously used records; adaptive development history not covered)',
           'households': int(G), 'persons': int(len(sex)), 'subgroup_households_per_sex': {int(g): int(len(np.unique(T['unit'][sex == g]))) for g in (0, 1)},
           'critical': {'c1': c1, 'n_claims': inf['n_claims'], 'c_family': cK, 'K': inf['K'], 't_marginal_95': cm},
           'locks_sha256': {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}, 'per_seed': {}}
    auc_pt, auc_dr, util, per_vec, wires, CV_ok = {}, {}, {}, {}, {}, True
    for s in C.SEEDS:
        bank, ids = FR.load_bank(s)
        frozen, fids = C.frozen_bank(s, roles=('DEV_REPORT',))
        if not (np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']) and np.array_equal(fids['row_id__DEV_REPORT'], T['row_id'])):
            raise AssertionError('bank / DEV_REPORT row order differs')
        rd = C.RUNS / f'READERS_S{s}'
        if rlock is not None and C.sha(rd / 'DEV_REPORT_SCORES.npz') != rlock['readers'][f'S{s}']['dev_report_scores_sha256']:
            raise PermissionError('reader scores differ from REPORTING_LOCK')
        rr = json.loads((rd / 'READERS_SEED.json').read_text()); scores = _scores(rd / 'DEV_REPORT_SCORES.npz')
        if not np.array_equal(scores['row_id'], T['row_id']):
            raise AssertionError('score / DEV_REPORT row order differs')
        srec = {'services': {}, 'controls': {}}
        refs = {r: {t: frozen[r]['DEV_REPORT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
        u0 = {t: C.task_vectors(y[t], refs['U0'][t], C.K[t]) for t in C.TASKS}
        wires[s] = {svc: {t: C.wire_sha(bank[svc]['DEV_REPORT'][t]) for t in C.TASKS} for svc in bank}
        for svc in sorted(bank):
            v = {'recovery': {}, 'utility': {}, 'subgroup': {}, 'task_summary': {}}
            for sc_name, wk in (('ENHANCED', 'enhanced'), ('ORIGINAL', 'original')):
                v['recovery'][sc_name] = {}
                for view in VIEWS:
                    w = rr['winners'][svc][wk][view]; sc = scores[w['reader_key']]
                    k = (svc, s, sc_name, view)
                    auc_pt[k] = auc(sex, sc); auc_dr[k] = bootstrap_auc(sex, sc, W)
                    se = float(np.std(auc_dr[k], ddof=1))
                    v['recovery'][sc_name][view] = {'winner': w['candidate'], 'origin': w['origin'], 'attack_validation_auc': w['attack_validation_auc'],
                                                    'AV_NEAR_TIE': w['AV_NEAR_TIE'], 'auc': auc_pt[k], 'auc_se': se,
                                                    'marginal95': [auc_pt[k] - cm * se, auc_pt[k] + cm * se]}
            v['pair_reader_slate_auc_descriptive'] = {k.split('__', 2)[2]: auc(sex, a) for k, a in scores.items() if k.startswith(f'PAIR__{svc}__')}
            for t in C.TASKS:
                vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], C.K[t]); per_vec[(svc, s, t)] = vv
                v['task_summary'][t] = {m_: float(vv[m_].mean()) for m_ in vv}
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], refs[ref][t], C.K[t])
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / Wsum
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr, d)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric],
                                                               'point_pass': bool(d.mean() <= C.CAPS[metric])}
            vv = per_vec[(svc, s, 'occupation')]
            for g in (0, 1):
                m_ = sex == g; d = vv['brier'] - u0['occupation']['brier']; Wg = W[:, m_]
                dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'guard', g)] = (float(d[m_].mean()), dr, None)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m_].mean()), 'se': float(np.std(dr, ddof=1))}
            v['utility_point_feasible_12_this_seed'] = all(x['point_pass'] for x in v['utility'].values())
            v['guard_point_pass_this_seed'] = all(x['occupation_brier_excess_vs_U0'] <= C.GUARD for x in v['subgroup'].values())
            srec['services'][svc] = v
        for view in VIEWS:
            crec = {}
            for name, c in rr['controls'][view].items():
                key = f'CONTROL__{view}__{name}'
                pl = np.where(sex == 1, scores[f'{key}__PLANTED_1'], scores[f'{key}__PLANTED_0'])
                sh, pa = auc(sex, scores[f'{key}__SHUFFLED']), auc(sex, pl)
                band = lock['controls_bands']
                ok = band['shuffled'][0] <= sh <= band['shuffled'][1] and pa >= band['planted_min']
                CV_ok = CV_ok and ok
                crec[name] = {'wire': c['wire'], 'shuffled_null_auc': sh, 'planted_bit_auc': pa, 'inside_bands': ok,
                              'shuffled_attack_validation_auc': c['shuffled_attack_validation_auc'], 'planted_attack_validation_auc': c['planted_attack_validation_auc']}
            srec['controls'][view] = crec
        srec['refit_roster'] = {svc: {'variation': {j: auc(sex, scores[f'VAR__{svc}__{j}']) for j in rr['variation'][svc]['refits']},
                                      'aux_sample': {j: auc(sex, scores[f'AUX__{svc}__{j}']) for j in rr['aux_sample_variation'][svc]['refits']},
                                      'winner': rr['variation'][svc]['winner']} for svc in rr['variation']}
        res['per_seed'][f'S{s}'] = srec
    res['CV'] = CV_ok

    # ---------------- endpoints and claims of the frozen roster, for each scorecard
    res['scorecards'] = {}
    draws_out = {}
    rows = []
    for sc_name in SCORECARDS:
        eps = {}
        for e in lock['endpoints']:
            s = e['seed']; fam = e['family']; identical = False
            if e['type'] == 'auc':
                a, b = e['a'], e['b']
                point = auc_pt[(a, s, sc_name, e['view'])] - auc_pt[(b, s, sc_name, e['view'])]
                draws = auc_dr[(a, s, sc_name, e['view'])] - auc_dr[(b, s, sc_name, e['view'])]
                identical = a == b or (e['view'] in C.TASKS and wires[s][a][e['view']] == wires[s][b][e['view']])
            elif e['type'] == 'utility':
                point, draws, _ = util[(e['service'], s, e['task'], e['ref'], e['metric'])]
            elif e['type'] == 'guard':
                point, draws, _ = util[(e['service'], s, 'guard', e['group'])]
            elif e['type'] == 'loss_diff':
                a, b = e['a'], e['b']
                d = per_vec[(a, s, e['task'])][e['metric']] - per_vec[(b, s, e['task'])][e['metric']]
                point, draws = float(d.mean()), (W * d[None, :]).sum(1) / Wsum
                identical = wires[s][a][e['task']] == wires[s][b][e['task']]
            else:
                raise ValueError('unregistered endpoint type')
            if identical:
                rec = {'point': 0.0, 'se': 0.0, 'kind': e['kind'], 'margin': e['margin'], 'family_lower': 0.0, 'family_upper': 0.0,
                       'clause_L1': 0.0, 'clause_U1': 0.0, 'status': 'IDENTICAL_RELEASE', 'category': 'PASS' if e['kind'] == 'no_excess' else 'IDENTICAL_RELEASE',
                       'point_pass': e['kind'] == 'no_excess', 'clause_pass': e['kind'] == 'no_excess', 'at': {}}
            else:
                rec = endpoint(point, draws, e['kind'], e['margin'], c1, cK[fam], e.get('refs', ()))
                draws_out[f'{sc_name}|{e["key"]}'] = np.asarray(draws, dtype=np.float64)
            rec.update({'family': fam, 'claim': e.get('claim'), 'flags': (['IDENTICAL_RELEASE'] if identical else []), 'spec': e})
            eps[e['key']] = rec
            rows.append({'scorecard': sc_name, 'key': e['key'], 'family': fam, 'claim': e.get('claim', ''), 'seed': s, 'kind': e['kind'],
                         'margin': e['margin'], 'point': rec['point'], 'se': rec['se'], 'clause_L1': rec.get('clause_L1'), 'clause_U1': rec.get('clause_U1'),
                         'family_lower': rec['family_lower'], 'family_upper': rec['family_upper'], 'category': rec['category'], 'status': rec['status'],
                         'flags': ';'.join(rec['flags'])})
        res['scorecards'][sc_name] = {'endpoints': eps, 'claims': claims_engine(lock['claims'], eps, CV_ok)}
    out.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DEV_REPORT_RESULTS.json', res)
    with open(out / 'PRIMARY_ENDPOINTS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (priv / 'ENDPOINT_DRAWS.npz').open('xb') as f:
        np.savez(f, **draws_out)
    hh = {}
    _, inv = np.unique(T['unit'], return_inverse=True)
    for (svc, s, t, ref, metric), (_, _, d) in ((k, v) for k, v in util.items() if k[2] != 'guard'):
        hh[f'{svc}|{s}|{t}|{ref}|{metric}'] = np.bincount(inv, weights=d)
    hh['household_size'] = np.bincount(inv)
    with (priv / 'HOUSEHOLD_CONTRIBUTIONS.npz').open('xb') as f:
        np.savez(f, **hh)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--priv', default=None)
    ap.add_argument('--replay', action='store_true')
    a = ap.parse_args()
    ev = 'DEV_REPORT_TRUTH_REREAD_BY_IHS_INDEPENDENT_VERIFIER' if a.replay else 'DEV_REPORT_TRUTH_REREAD_BY_IHS_COORDINATOR'
    r = run(a.out, priv_dir=a.priv, event_name=ev)
    print(json.dumps({sc: {k: v['label'] for k, v in r['scorecards'][sc]['claims'].items()} for sc in r['scorecards']}, indent=1))

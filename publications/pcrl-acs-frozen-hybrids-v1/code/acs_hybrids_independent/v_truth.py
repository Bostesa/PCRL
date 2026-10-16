"""V3 Utility, V4 Subgroups, V7 Household resampling, V8 Endpoints and multiplicity, V9 Decisions and labels.

One DEV_REPORT truth re-read per real run, through acs_hybrids.common.dev_report_truth (logged automatically). Every metric,
AUC, bootstrap, endpoint, critical value, status and decision below is computed by the verifier's own code (vcommon).
task_headroom.inference.group_weights is called once, only to cross-check the verifier's own household resampler.

--synthetic: rehearsal with random labels (no truth read); outputs go to <scratch>/synthetic and are never reported.
Inputs: results/.../independent/V5_reader_selection.json (winners re-derived by the verifier) and
<scratch>/v9_stage1_controls.json (verifier's control refits).
"""
from __future__ import annotations

import argparse
import csv
import json
import re

import numpy as np
from scipy import stats

from acs_hybrids import common as C   # hash-checked data loading and the logged truth re-read only

from . import vcommon as V

TOL = 1e-12
PURPOSE = ('independent verification: V3 utility, V4 subgroups, V7 household bootstrap, V8 endpoints, V9 decisions '
           '(verifier own metric and inference code; single re-read)')


def cmp_num(mine, theirs, tol=TOL):
    if theirs is None or mine is None:
        return {'mine': mine, 'theirs': theirs, 'abs_diff': None, 'bitwise': False, 'ok': False}
    d = abs(float(mine) - float(theirs))
    return {'abs_diff': d, 'bitwise': float(mine) == float(theirs), 'ok': d <= tol}


class Tally:
    def __init__(self):
        self.n = 0
        self.bit = 0
        self.max = 0.0
        self.bad = []

    def add(self, label, mine, theirs, tol=TOL):
        self.n += 1
        if mine is None and theirs is None:
            self.bit += 1
            return
        if theirs is None or mine is None or not np.isfinite(mine) or not np.isfinite(theirs):
            self.bad.append(label)
            return
        d = abs(float(mine) - float(theirs))
        self.max = max(self.max, d)
        self.bit += int(float(mine) == float(theirs))
        if d > tol:
            self.bad.append(label)

    def eq(self, label, mine, theirs):
        self.n += 1
        if mine == theirs:
            self.bit += 1
        else:
            self.bad.append(f'{label}: mine={mine} theirs={theirs}')

    def out(self):
        return {'compared': self.n, 'bitwise_or_equal': self.bit, 'max_abs_diff': self.max, 'failures': self.bad[:50],
                'n_failures': len(self.bad), 'ok': not self.bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--synthetic', action='store_true')
    a = ap.parse_args()
    scratch = V.scratch_dir(a.scratch)
    outdir = (scratch / 'synthetic') if a.synthetic else V.OUT
    L = V.analysis_lock()
    RL = V.reporting_lock()
    inf = L['inference']
    camp = V.jload(V.RES / 'DEV_REPORT_RESULTS.json')
    srcres = V.jload(V.SRC_RES / 'DEV_REPORT_RESULTS.json')
    comps = V.components()
    v5 = V.jload(V.OUT / 'V5_reader_selection.json')
    stage1 = V.jload(scratch / 'v9_stage1_controls.json')
    R = C.load_role('DEV_REPORT', ('features',))
    n = len(R['row_id'])
    if a.synthetic:
        rng = np.random.default_rng(99)
        T = {'row_id': R['row_id'], 'unit': R['unit'], 'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n),
             'sex': rng.integers(0, 2, n), 'weights': rng.integers(1, 200, n)}
        rereads = 0
    else:
        T = C.dev_report_truth(purpose=PURPOSE)
        rereads = 1
    if not (np.array_equal(T['row_id'], R['row_id']) and np.array_equal(T['unit'], R['unit'])):
        raise SystemExit('truth rows differ from the DEV_REPORT export')
    y = {'income': np.asarray(T['income'], dtype=np.int64), 'occupation': np.asarray(T['occupation'], dtype=np.int64)}
    sex = np.asarray(T['sex'], dtype=np.int64)
    pw = np.asarray(T['weights'], dtype=np.float64)
    unit = np.asarray(T['unit'])
    common_meta = {'truth_rereads_in_this_job': rereads, 'synthetic_rehearsal': bool(a.synthetic)}

    # =============================================================== V7 household bootstrap
    W, counts, G = V.household_weights(unit, V.BOOT_B, V.BOOT_SEED)
    from task_headroom.inference import group_weights   # cross-check of the verifier's resampler only
    W_x, G_x = group_weights(unit, seed=V.BOOT_SEED, draws=V.BOOT_B)
    banks, scores = {}, {}
    for s in V.SEEDS:
        b, ids = C.load_bank(s, roles=('DEV_REPORT',))
        if not (np.array_equal(ids['row_id__DEV_REPORT'], R['row_id']) and np.array_equal(ids['unit__DEV_REPORT'], unit)):
            raise SystemExit('bank rows differ from DEV_REPORT')
        banks[s] = b
        with np.load(V.PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz', allow_pickle=False) as z:
            scores[s] = {k: z[k].copy() for k in z.files}
        if not np.array_equal(scores[s]['row_id'], R['row_id']):
            raise SystemExit('score rows differ from DEV_REPORT')
    Wsum = W.sum(1)
    v7 = {'check': 'V7 Household resampling', **common_meta,
          'construction': 'households in sorted unit order; default_rng(2026101603).integers(0, G, size=(2048, G)); pick counts '
                          'accumulated with np.add.at; person weight = household count (verifier implementation)',
          'B': int(W.shape[0]), 'G': int(G), 'persons': int(n),
          'draw_rows_sum_to_G_households': bool(np.all(counts.sum(1) == G)),
          'cross_check_task_headroom_group_weights': {'bitwise_equal': bool(np.array_equal(W, W_x)), 'G_equal': int(G_x) == int(G),
                                                      'max_abs_diff': V.maxabs(W, W_x)},
          'bank_units_identical_both_seeds_and_equal_truth_units': True,
          'registered': {'B': inf['bootstrap_B'], 'seed': inf['bootstrap_seed'], 'G_text': inf['bootstrap_unit']}}
    del W_x

    # =============================================================== V3 utility
    vec = {}
    for s in V.SEEDS:
        for svc in V.ROSTER + V.REFS:
            for t in V.TASKS:
                vec[(svc, s, t)] = V.metric_vectors(y[t], banks[s][svc]['DEV_REPORT'][t], V.NCLASS[t])
    util = {}
    tu_c, tu_s, tu_cs, tu_cs_se, tu_pw = Tally(), Tally(), Tally(), Tally(), Tally()
    detail_hlj = {}
    for s in V.SEEDS:
        for svc in V.ROSTER:
            for ti, t in enumerate(V.TASKS):
                vv = vec[(svc, s, t)]
                for ref in V.REFS:
                    rv = vec[(ref, s, t)]
                    ds = {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'],
                          'brier_excess': vv['brier'] - rv['brier']}
                    for metric, d in ds.items():
                        key = f'{t}|{ref}|{metric}'
                        point = float(np.mean(d))
                        draws = V.boot_mean(W, d)
                        se = V.sd(draws)
                        util[(svc, s, t, ref, metric)] = (point, draws)
                        c = camp['per_seed'][f'S{s}']['services'][svc]['utility'][key]
                        sp = srcres['per_seed'][str(s)]['services'][comps[svc][ti]]['utility'][key]
                        lab = f'S{s}|{svc}|{key}'
                        tu_c.add(lab + '|point', point, c['point'])
                        tu_c.add(lab + '|se', se, c['se'])
                        tu_c.eq(lab + '|point_pass', point <= V.CAPS[metric], c['point_pass'])
                        tu_s.add(lab + '|point', point, sp['point'])
                        tu_s.add(lab + '|se', se, sp['se'])
                        tu_cs.add(lab + '|point', c['point'], sp['point'], tol=0.0)
                        tu_cs_se.add(lab + '|se', c['se'], sp['se'], tol=TOL)
                        tu_pw.add(lab + '|pwgtp', float(np.sum(pw * d) / pw.sum()), c['pwgtp_weighted_point'], tol=1e-12)
                        if svc == 'H_LJ':
                            detail_hlj[f'RECIPE|H_LJ|utility|{t}|{ref}|{metric}|s{s}'] = {
                                'point': point, 'se': se, 'cap': V.CAPS[metric], 'point_pass': point <= V.CAPS[metric],
                                'campaign_point': c['point'], 'campaign_se': c['se'], 'source_component': comps[svc][ti],
                                'source_published_point': sp['point'], 'source_published_se': sp['se'],
                                'campaign_equals_source_exactly': c['point'] == sp['point'] and c['se'] == sp['se']}
    inh = {}
    for s in V.SEEDS:
        for h in V.HYBRIDS:
            for ti, t in enumerate(V.TASKS):
                ref_svc = 'H_LL' if comps[h][ti] == 'LOCAL_TREE' else 'H_JJ'
                inh[f'S{s}|{h}|{t}=={comps[h][ti]}'] = all(np.array_equal(vec[(h, s, t)][m], vec[(ref_svc, s, t)][m]) for m in ('accuracy', 'nll', 'brier'))
    gate24 = all(v['point_pass'] for v in detail_hlj.values())
    # MODEL_SELECT 24-clause point feasibility (already-exposed MODEL_SELECT labels; recomputed for every service)
    ms = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation'))
    adm = V.jload(V.RES / 'SOURCE_ADMISSION.json')['model_select_point_feasibility']
    det = V.jload(V.PRIV / 'admission' / 'RECOMPUTE_DETAIL.json')['model_select_clauses']
    msf, ms_t = {}, Tally()
    bank_ms = {}
    for s in V.SEEDS:
        bms, ids = C.load_bank(s, roles=('MODEL_SELECT',))
        if not (np.array_equal(ids['row_id__MODEL_SELECT'], ms['row_id']) and np.array_equal(ids['unit__MODEL_SELECT'], ms['unit'])):
            raise SystemExit('MODEL_SELECT rows differ')
        bank_ms[s] = bms
    for svc in V.ROSTER:
        failed, clauses = [], []
        for s in V.SEEDS:
            bms = bank_ms[s]
            for t in V.TASKS:
                vv = V.metric_vectors(ms[t], bms[svc]['MODEL_SELECT'][t], V.NCLASS[t])
                for ref in V.REFS:
                    rv = V.metric_vectors(ms[t], bms[ref]['MODEL_SELECT'][t], V.NCLASS[t])
                    vals = {'accuracy_loss': float(np.mean(rv['accuracy']) - np.mean(vv['accuracy'])),
                            'nll_excess': float(np.mean(vv['nll']) - np.mean(rv['nll'])),
                            'brier_excess': float(np.mean(vv['brier']) - np.mean(rv['brier']))}
                    for metric in V.METRICS:
                        clauses.append((s, t, ref, metric, vals[metric]))
                        if not vals[metric] <= V.CAPS[metric]:
                            failed.append(f's{s}|{t}|{ref}|{metric}')
        msf[svc] = {'n_clauses': len(clauses), 'point_feasible_24': not failed and len(clauses) == 24, 'failed_clauses': failed,
                    'campaign_point_feasible_24': adm[svc]['point_feasible_24'], 'campaign_failed_clauses': adm[svc]['failed_clauses']}
        msf[svc]['agrees'] = bool(msf[svc]['point_feasible_24'] == adm[svc]['point_feasible_24'] and sorted(failed) == sorted(adm[svc]['failed_clauses']))
        theirs = {(c['seed'], c['task'], c['reference'], c['metric']): c['value'] for c in det[svc]}
        for (s, t, ref, metric, val) in clauses:
            ms_t.add(f'{svc}|s{s}|{t}|{ref}|{metric}', val, theirs.get((s, t, ref, metric)))
        if svc == 'H_LJ':
            msf[svc]['clause_values'] = {f's{s}|{t}|{ref}|{metric}': val for (s, t, ref, metric, val) in clauses}
    ms_ok_hlj = msf['H_LJ']['point_feasible_24']
    v3 = {'check': 'V3 Utility', **common_meta,
          'definitions': 'accuracy_loss = acc(ref) - acc(svc); nll_excess = nll(svc) - nll(ref) with p clipped at 1e-12; '
                         'brier_excess = summed Brier(svc) - summed Brier(ref); point = unweighted mean; SE = sd(ddof=1) of the '
                         'household-bootstrap weighted means (W @ d / W.sum)',
          'min_true_class_probability': min(v['min_true_class_prob'] for v in vec.values()),
          'H_LJ_24_clauses': detail_hlj, 'H_LJ_dev_report_point_gate_24': gate24,
          'H_LJ_points_passing': sum(v['point_pass'] for v in detail_hlj.values()),
          'all_services_vs_campaign': tu_c.out(), 'all_services_vs_source_published': tu_s.out(),
          'campaign_vs_source_published_points_exact': tu_cs.out(), 'campaign_vs_source_published_se': tu_cs_se.out(),
          'pwgtp_points_vs_campaign': tu_pw.out(),
          'per_person_inheritance_equalities': {'n': len(inh), 'all': all(inh.values()), 'failures': [k for k, v in inh.items() if not v]},
          'model_select_feasibility': msf, 'model_select_clause_values_vs_admission_detail': ms_t.out(),
          'H_LJ_model_select_point_feasible_24': ms_ok_hlj}
    v3['status'] = 'PASS' if (tu_c.out()['ok'] and tu_s.out()['ok'] and tu_cs.out()['ok'] and tu_cs_se.out()['ok'] and tu_pw.out()['ok'] and all(inh.values())
                             and all(v['agrees'] for v in msf.values()) and ms_t.out()['ok'] and len(detail_hlj) == 24) else 'FAIL'

    # =============================================================== V4 subgroups
    sub, sub_t, sub_cs, sub_cs_se = {}, Tally(), Tally(), Tally()
    groups = {}
    for g in (0, 1):
        m = sex == g
        groups[g] = {'mask': m, 'W': W[:, m], 'hh': int(len(np.unique(unit[m]))), 'persons': int(m.sum())}
    hlj_guards = {}
    for s in V.SEEDS:
        rv = vec[('U0', s, 'occupation')]
        for svc in V.ROSTER:
            vv = vec[(svc, s, 'occupation')]
            p = banks[s][svc]['DEV_REPORT']['occupation']
            for g in (0, 1):
                m = groups[g]['mask']
                d = vv['brier'] - rv['brier']
                point = float(np.mean(d[m]))
                draws = (groups[g]['W'] @ d[m]) / groups[g]['W'].sum(1)
                se = V.sd(draws)
                sub[(svc, s, g)] = (point, draws)
                dec = np.argmax(p[m], axis=1)
                ym = y['occupation'][m]
                recall = [float(np.mean(dec[ym == c] == c)) if np.any(ym == c) else None for c in range(5)]
                support = [int(np.sum(ym == c)) for c in range(5)]
                c = camp['per_seed'][f'S{s}']['services'][svc]['subgroup'][f'sex{g}']
                sp = srcres['per_seed'][str(s)]['services'][comps[svc][1]]['subgroup'][f'sex{g}']
                lab = f'S{s}|{svc}|sex{g}'
                sub_t.add(lab + '|point', point, c['occupation_brier_excess_vs_U0'])
                sub_t.add(lab + '|se', se, c['se'])
                sub_t.add(lab + '|accuracy', float(np.mean(vv['accuracy'][m])), c['group_accuracy'])
                sub_t.add(lab + '|nll', float(np.mean(vv['nll'][m])), c['group_nll'])
                sub_t.add(lab + '|acc_loss', float(np.mean((rv['accuracy'] - vv['accuracy'])[m])), c['group_accuracy_loss_vs_U0'])
                sub_t.add(lab + '|nll_excess', float(np.mean((vv['nll'] - rv['nll'])[m])), c['group_nll_excess_vs_U0'])
                sub_t.eq(lab + '|households', groups[g]['hh'], c['group_households'])
                sub_t.eq(lab + '|persons', groups[g]['persons'], c['group_persons'])
                sub_t.eq(lab + '|support', support, c['group_class_support'])
                for ci in range(5):
                    sub_t.add(lab + f'|recall{ci}', recall[ci], c['group_per_class_recall'][ci])
                sub_cs.add(lab + '|point', c['occupation_brier_excess_vs_U0'], sp['point'], tol=0.0)
                sub_cs_se.add(lab + '|se', c['se'], sp['se'], tol=TOL)
                sub_cs.add(lab + '|accuracy', c['group_accuracy'], sp['group_accuracy'], tol=0.0)
                sub_cs.eq(lab + '|households', c['group_households'], sp['group_households'])
                if svc == 'H_LJ':
                    hlj_guards[f'SUBGROUP|H_LJ|occupation_brier_excess_vs_U0|sex{g}|s{s}'] = {
                        'point': point, 'se': se, 'households': groups[g]['hh'], 'persons': groups[g]['persons'],
                        'accuracy': float(np.mean(vv['accuracy'][m])), 'nll': float(np.mean(vv['nll'][m])),
                        'per_class_recall': recall, 'class_support': support}

    # =============================================================== V8 AUCs and endpoints
    G1 = G - 1
    c1 = V.tcrit(inf['alpha_family_one_sided'] / inf['n_claims'], G1)
    c38 = V.tcrit(0.05 / (2 * inf['K_primary']), G1)
    c22 = V.tcrit(0.05 / (2 * inf['K_secondary']), G1)
    cm = V.tcrit(0.025, G1)
    c40 = V.tcrit(0.05 / 80, G1)
    reg = inf['critical_values']
    crit = {'df': G1, 'clause_one_sided_alpha_over_3': c1, 'primary_K38': c38, 'secondary_K22': c22, 'marginal_95': cm, 'source_K40_reference': c40}
    crit_check = {k: {'value': crit[k], 'rounded_4': round(crit[k], 4), 'registered': reg[k], 'equal': round(crit[k], 4) == reg[k]}
                  for k in ('clause_one_sided_alpha_over_3', 'primary_K38', 'secondary_K22', 'marginal_95', 'source_K40_reference')}
    crit_check['df'] = {'value': G1, 'registered': reg['df'], 'equal': G1 == reg['df']}
    crit_check['brief_values'] = {'2.3950': round(c1, 4) == 2.3950, '3.2148': round(c38, 4) == 3.2148, '3.0541': round(c22, 4) == 3.0541,
                                  '1.9606': round(cm, 4) == 1.9606}
    cc = camp['critical']
    crit_check['campaign_report_values_equal'] = {'t_one_sided_alpha_over_3': cc['t_one_sided_alpha_over_3'] == c1, 't_simultaneous_primary': cc['t_simultaneous_primary'] == c38,
                                                  't_simultaneous_secondary': cc['t_simultaneous_secondary'] == c22, 't_marginal_95': cc['t_marginal_95'] == cm}
    winners = v5['derived_winner_reader_ids']
    win_vs_lock = all(winners[f'S{s}'][svc][v] == RL['readers'][f'S{s}']['winners'][svc][v]['reader_id'] for s in V.SEEDS for svc in V.ROSTER for v in V.VIEWS)
    cache, auc_pt, auc_dr = {}, {}, {}
    rec_t, rec_src, rec_pw = Tally(), Tally(), Tally()
    for s in V.SEEDS:
        for svc in V.ROSTER:
            for view in V.VIEWS:
                rid = winners[f'S{s}'][svc][view]
                if (s, rid) not in cache:
                    sc = scores[s][rid]
                    cache[(s, rid)] = (V.auc(sex, sc), V.auc_weighted_draws(sex, sc, W), float(V.auc_weighted_draws(sex, sc, pw[None, :])[0]))
                pt, dr, pwa = cache[(s, rid)]
                auc_pt[(svc, s, view)] = pt
                auc_dr[(svc, s, view)] = dr
                c = camp['per_seed'][f'S{s}']['services'][svc]['recovery'][view]
                lab = f'S{s}|{svc}|{view}'
                rec_t.add(lab + '|auc', pt, c['auc'])
                rec_t.add(lab + '|auc_se', V.sd(dr), c['auc_se'])
                rec_t.eq(lab + '|reader_id', rid, c['reader_id'])
                rec_t.add(lab + '|marginal95_lo', pt - cm * V.sd(dr), c['marginal95'][0])
                rec_t.add(lab + '|marginal95_hi', pt + cm * V.sd(dr), c['marginal95'][1])
                rec_pw.add(lab + '|pwgtp_auc', pwa, c['pwgtp_weighted_auc_point'])
                srclab = {'H_LL': 'LOCAL_TREE', 'H_JJ': 'JOINT_TREE', 'NEURAL_RAW': 'NEURAL_RAW_rho1.0_E2', 'TASK_REFIT': 'TASK_REFIT', 'T_TREE': 'T_TREE'}.get(svc)
                if srclab is not None:
                    sr = srcres['per_seed'][str(s)]['services'][srclab]['recovery'][view]
                    rec_src.add(lab + '|auc_vs_source', pt, sr['auc'], tol=0.0)
                    rec_src.add(lab + '|auc_se_vs_source', V.sd(dr), sr['auc_se'], tol=TOL)
    # descriptive pair slate (every slate reader, pair view) for H_LJ, H_LL, TASK_REFIT, NEURAL_RAW
    desc_t = Tally()
    for s in V.SEEDS:
        for svc in ('H_LJ', 'H_LL', 'TASK_REFIT', 'NEURAL_RAW'):
            by = camp['descriptive']['pair_slate_envelope_selected_on_report_data'][f'S{s}|{svc}']['by_reader']
            for nm in V.SLATE:
                desc_t.add(f'S{s}|{svc}|{nm}', V.auc(sex, scores[s][f'PAIR__{svc}__{nm}']), by[nm])

    alpha_clause = inf['alpha_family_one_sided'] / inf['n_claims']

    def evaluate(spec):
        e = spec['estimand']
        s = spec['seed']
        if e['type'] == 'auc_diff':
            return auc_pt[(e['plus'], s, e['view'])] - auc_pt[(e['minus'], s, e['view'])], auc_dr[(e['plus'], s, e['view'])] - auc_dr[(e['minus'], s, e['view'])]
        if e['type'] == 'auc_interaction':
            f = lambda x: (auc_pt[(x, s, e['view'])], auc_dr[(x, s, e['view'])])
            (ll, dll), (lj, dlj), (jl, djl), (jj, djj) = f('H_LL'), f('H_LJ'), f('H_JL'), f('H_JJ')
            return (ll - lj) - (jl - jj), (dll - dlj) - (djl - djj)
        if e['type'] == 'utility':
            return util[(e['service'], s, e['task'], e['reference'], e['metric'])]
        if e['type'] == 'subgroup_occupation_brier_vs_U0':
            return sub[(e['service'], s, e['group'])]
        raise ValueError(e)

    def endpoint(spec, cK):
        point, draws = evaluate(spec)
        point = float(point)
        se = V.sd(draws)
        kind = spec['kind']
        Lo, Up = point - cK * se, point + cK * se
        clause = spec.get('claim') in ('A', 'B', 'C')
        r = {'point': point, 'se': se, 'L': Lo, 'U': Up, 'kind': kind, 'at': {},
             'skew': float(stats.skew(draws, bias=True))}
        for mref in spec['refs']:
            row = {'point_minus_m': point - mref, 'lower_minus_m': Lo - mref, 'upper_minus_m': Up - mref, 'status_at_m': V.status_at(kind, Lo, Up, mref)}
            if clause:
                row['clause_bound_minus_m'] = (point - c1 * se - mref) if kind == 'benefit' else (point + c1 * se - mref)
            r['at'][str(mref)] = row
        if clause:
            mg = spec['margin']
            r['L1'], r['U1'] = point - c1 * se, point + c1 * se
            r['clause_pass'] = (r['L1'] > mg) if kind == 'benefit' else (r['U1'] <= mg)
            r['point_pass'] = (point > mg) if kind == 'benefit' else (point <= mg)
            r['status'] = V.status_at(kind, Lo, Up, mg)
            q = float(np.quantile(draws, alpha_clause if kind == 'benefit' else 1 - alpha_clause))
            r['sens_q'] = q
            r['sens_pass'] = (q > mg) if kind == 'benefit' else (q <= mg)
        else:
            r['status'] = V.status_at(kind, Lo, Up, spec['refs'][0]) if kind == 'difference' else {str(mr): r['at'][str(mr)]['status_at_m'] for mr in spec['refs']}
        return r

    E = {sp['key']: endpoint(sp, c38) for sp in L['primary_roster']}
    E2 = {sp['key']: endpoint(sp, c22) for sp in L['secondary_roster']}
    ep_t = Tally()
    for fam, roster, cE in (('primary', L['primary_roster'], camp['endpoints_primary']), ('secondary', L['secondary_roster'], camp['endpoints_secondary'])):
        mine = E if fam == 'primary' else E2
        for sp in roster:
            k = sp['key']
            mi, th = mine[k], cE.get(k)
            if th is None:
                ep_t.bad.append(f'{k}: missing in campaign')
                continue
            ep_t.add(k + '|point', mi['point'], th['point'])
            ep_t.add(k + '|se', mi['se'], th['se'])
            ep_t.add(k + '|L', mi['L'], th['family_lower'])
            ep_t.add(k + '|U', mi['U'], th['family_upper'])
            ep_t.eq(k + '|status', mi['status'], th['status'])
            ep_t.add(k + '|skew', mi['skew'], th['bootstrap_skewness'], tol=1e-9)
            for mref, row in mi['at'].items():
                trow = th['at'].get(mref)
                if trow is None:
                    ep_t.bad.append(f'{k}|m={mref}: missing distance row in campaign')
                    continue
                for col in ('point_minus_m', 'lower_minus_m', 'upper_minus_m'):
                    ep_t.add(f'{k}|m={mref}|{col}', row[col], trow[col])
                ep_t.eq(f'{k}|m={mref}|status', row['status_at_m'], trow['status_at_m'])
                if 'clause_bound_minus_m' in row or 'clause_bound_minus_m' in trow:
                    ep_t.add(f'{k}|m={mref}|clause_bound', row.get('clause_bound_minus_m'), trow.get('clause_bound_minus_m'))
            if set(mi['at']) != set(th['at']):
                ep_t.bad.append(f'{k}: reference values differ {sorted(mi["at"])} vs {sorted(th["at"])}')
            if 'L1' in mi:
                ep_t.add(k + '|L1', mi['L1'], th.get('clause_L1'))
                ep_t.add(k + '|U1', mi['U1'], th.get('clause_U1'))
                ep_t.eq(k + '|clause_pass', mi['clause_pass'], th.get('clause_pass'))
                ep_t.eq(k + '|point_pass', mi['point_pass'], th.get('point_pass'))
                ep_t.add(k + '|sens_q', mi['sens_q'], th.get('sensitivity_percentile_bound'))
                ep_t.eq(k + '|sens_pass', mi['sens_pass'], th.get('sensitivity_pass'))
            elif 'clause_L1' in th:
                ep_t.bad.append(f'{k}: campaign has clause bounds for a non-claim key')
    roster_meta = {'primary_keys': len(E), 'secondary_keys': len(E2), 'all_finite': all(np.isfinite(v['point']) and np.isfinite(v['se']) for v in list(E.values()) + list(E2.values())),
                   'campaign_primary_keys': len(camp['endpoints_primary']), 'campaign_secondary_keys': len(camp['endpoints_secondary']),
                   'key_sets_equal': set(E) == set(camp['endpoints_primary']) and set(E2) == set(camp['endpoints_secondary'])}
    # roster definition consistency (signs, views, margins) read against the key names and definition strings
    defs_bad = []
    wire_src = {('income', 'LOCAL'): {'H_LL', 'H_LJ'}, ('income', 'JOINT'): {'H_JL', 'H_JJ'},
                ('occupation', 'LOCAL'): {'H_LL', 'H_JL'}, ('occupation', 'JOINT'): {'H_LJ', 'H_JJ'}}
    for sp in L['primary_roster'] + L['secondary_roster']:
        e, k = sp['estimand'], sp['key']
        if f"|s{sp['seed']}" != k[-3:]:
            defs_bad.append(f'{k}: seed suffix')
        dfn = sp.get('definition') or ''
        mm = re.match(r'AUC_(\w+)\((\w+)\) - AUC_(\w+)\((\w+)\)', dfn)
        if e['type'] == 'auc_diff':
            if not mm:
                defs_bad.append(f'{k}: no parsable definition')
                continue
            v1, p1, v2, p2 = mm.groups()
            ok_p = (p1 == e['plus']) or (e['plus'] in wire_src.get((e['view'], p1), set()))
            ok_m = (p2 == e['minus']) or (e['minus'] in wire_src.get((e['view'], p2), set()))
            if not (v1 == v2 == e['view'] and ok_p and ok_m):
                defs_bad.append(f'{k}: definition {dfn} vs estimand {e}')
            if 'coalition_benefit_vs_' in k:
                comp_ = k.split('coalition_benefit_vs_')[1].split('|')[0]
                svc_ = k.split('|')[1]
                if not (e['plus'] == comp_ and e['minus'] == svc_ and e['view'] == 'pair' and sp['kind'] == 'benefit'):
                    defs_bad.append(f'{k}: benefit sign')
            if 'recipient_worsening' in k and not (e['minus'] == 'TASK_REFIT' and sp['kind'] == 'no_excess'):
                defs_bad.append(f'{k}: worsening sign')
        elif e['type'] == 'utility':
            parts = k.split('|')
            if not (parts[1] == e['service'] and parts[3] == e['task'] and parts[4] == e['reference'] and parts[5] == e['metric']
                    and sp['margin'] == V.CAPS[e['metric']] and sp['refs'] == [V.CAPS[e['metric']]] and sp['kind'] == 'no_excess'):
                defs_bad.append(f'{k}: utility spec')
        elif e['type'] == 'subgroup_occupation_brier_vs_U0':
            if not (f"sex{e['group']}" in k and sp['margin'] == 0.005 and sp['refs'] == [0.005] and sp['kind'] == 'no_excess'):
                defs_bad.append(f'{k}: subgroup spec')
    margins = {'coalition_benefit_vs_TASK_REFIT (claim A)': all(sp['margin'] == 0.02 for sp in L['primary_roster'] if 'coalition_benefit_vs_TASK_REFIT' in sp['key']),
               'recipient_worsening (claim A)': all(sp['margin'] == 0.01 for sp in L['primary_roster'] if 'worsening' in sp['key']),
               'COMPETITIVE (claim B)': all(sp['margin'] == 0.0 for sp in L['primary_roster'] if sp['key'].startswith('COMPETITIVE'))}
    # distance rule: reference sets
    dist_rule_bad = []
    for sp in L['primary_roster'] + L['secondary_roster']:
        k, refs = sp['key'], sp['refs']
        if 'coalition_benefit' in k or 'coalition_gain' in k:
            exp = [0.0, 0.02]
        elif 'worsening' in k:
            exp = [0.0, 0.01]
        elif 'INTERACTION' in k:
            exp = [0.0]
        elif '|utility|' in k:
            exp = [V.CAPS[k.split('|')[5]]]
        elif k.startswith('SUBGROUP'):
            exp = [0.005]
        else:
            exp = None
        if refs != exp:
            dist_rule_bad.append(f'{k}: refs {refs} vs rule {exp}')
    # CSVs
    csv_t = Tally()
    for fname, mine in (('PRIMARY_ENDPOINTS.csv', E), ('SECONDARY_ENDPOINTS.csv', E2)):
        with open(V.RES / fname, newline='') as f:
            rows = list(csv.DictReader(f))
        csv_t.eq(fname + '|rows', len(rows), len(mine))
        for row in rows:
            k = row['key']
            mi = mine.get(k)
            if mi is None:
                csv_t.bad.append(f'{fname}: unknown key {k}')
                continue
            for col, mk in (('point', 'point'), ('se', 'se'), ('family_lower', 'L'), ('family_upper', 'U')):
                csv_t.add(f'{fname}|{k}|{col}', mi[mk], float(row[col]))
            st = row['status']
            st_m = json.dumps(mi['status']) if isinstance(mi['status'], dict) else mi['status']
            if isinstance(mi['status'], dict):
                csv_t.eq(f'{fname}|{k}|status', mi['status'], json.loads(st))
            else:
                csv_t.eq(f'{fname}|{k}|status', st_m, st)
            if 'L1' in mi:
                csv_t.add(f'{fname}|{k}|clause_L1', mi['L1'], float(row['clause_L1']))
                csv_t.add(f'{fname}|{k}|clause_U1', mi['U1'], float(row['clause_U1']))
                csv_t.eq(f'{fname}|{k}|clause_pass', str(mi['clause_pass']), row['clause_pass'])
    dist_t = Tally()
    with open(V.RES / 'DISTANCES.csv', newline='') as f:
        drows = list(csv.DictReader(f))
    exp_rows = sum(len(v['at']) for v in list(E.values()) + list(E2.values()))
    dist_t.eq('DISTANCES.csv|rows', len(drows), exp_rows)
    seen = set()
    for row in drows:
        k, mref = row['key'], row['m']
        mi = E.get(k) or E2.get(k)
        if mi is None or mref not in mi['at']:
            dist_t.bad.append(f'unknown distance row {k} m={mref}')
            continue
        seen.add((k, mref))
        fam_expected = 'primary (K = 38)' if k in E else 'secondary descriptive (K = 22)'
        dist_t.eq(f'{k}|{mref}|family', row['family'], fam_expected)
        r = mi['at'][mref]
        for col in ('point_minus_m', 'lower_minus_m', 'upper_minus_m', 'clause_bound_minus_m'):
            if col in r:
                dist_t.eq(f'{k}|{mref}|{col}', round(r[col], 4), float(row[col]) if row[col] != '' else None)
            else:
                dist_t.eq(f'{k}|{mref}|{col}(absent)', '', row[col])
        dist_t.eq(f'{k}|{mref}|status', r['status_at_m'], row['status_at_m'])
    dist_t.eq('DISTANCES.csv|all endpoint-reference rows present', len(seen), exp_rows)

    # settled-before-readers and value_source P/p endpoints vs published values
    settled = L['settled_before_readers']
    pub_t = Tally()
    for k, v in settled['per_key'].items():
        if 'published_se' in v:
            pub_t.add(k + '|point', E[k]['point'], v['published_point'])
            pub_t.add(k + '|se', E[k]['se'], v['published_se'])
            pub_t.eq(k + '|K38_status', E[k]['status'], v['K38_status'])
            pub_t.eq(k + '|clause_pass', E[k]['clause_pass'], v['clause_pass'])
        else:
            pub_t.add(k + '|point', E[k]['point'], v['published_point'], tol=1e-12)
    for sp in L['primary_roster'] + L['secondary_roster']:
        cp = sp.get('source_counterpart')
        if cp and cp.get('type') == 'endpoint':
            sr = srcres['endpoints'][cp['key']]
            mine_ = E.get(sp['key']) or E2[sp['key']]
            pub_t.add(sp['key'] + '|src_endpoint_point', mine_['point'], sr['point'], tol=0.0)
            pub_t.add(sp['key'] + '|src_endpoint_se', mine_['se'], sr['se'])
    v8_core = ep_t.out()
    v8 = {'check': 'V8 Endpoints and multiplicity', **common_meta,
          'critical_values': crit_check,
          'winners_used': 'derived by the verifier in V5 from saved models and ATTACK_VALIDATION only',
          'winners_equal_REPORTING_LOCK': bool(win_vs_lock),
          'recovery_auc_vs_campaign': rec_t.out(), 'recovery_auc_vs_source_published': rec_src.out(),
          'pwgtp_weighted_auc_points_vs_campaign': rec_pw.out(), 'descriptive_pair_slate_aucs_vs_campaign': desc_t.out(),
          'roster': roster_meta, 'roster_definition_consistency_failures': defs_bad, 'registered_margins': margins,
          'distance_rule_reference_sets_failures': dist_rule_bad,
          'endpoints_vs_DEV_REPORT_RESULTS': v8_core, 'endpoints_vs_csv_tables': csv_t.out(), 'DISTANCES_csv': dist_t.out(),
          'settled_and_published_values': pub_t.out(),
          'primary': {k: {kk: v[kk] for kk in ('point', 'se', 'L', 'U', 'status') if kk in v} | ({'L1': v['L1'], 'U1': v['U1'], 'clause_pass': v['clause_pass']} if 'L1' in v else {})
                      for k, v in E.items()},
          'secondary': {k: {kk: v[kk] for kk in ('point', 'se', 'L', 'U', 'status')} for k, v in E2.items()}}
    v8['status'] = 'PASS' if (all(x['equal'] for x in crit_check.values() if isinstance(x, dict) and 'equal' in x)
                             and all(crit_check['brief_values'].values()) and all(crit_check['campaign_report_values_equal'].values())
                             and win_vs_lock and rec_t.out()['ok'] and rec_src.out()['ok'] and rec_pw.out()['ok'] and desc_t.out()['ok']
                             and roster_meta['primary_keys'] == 38 and roster_meta['secondary_keys'] == 22 and roster_meta['all_finite']
                             and roster_meta['key_sets_equal'] and not defs_bad and all(margins.values()) and not dist_rule_bad
                             and v8_core['ok'] and csv_t.out()['ok'] and dist_t.out()['ok'] and pub_t.out()['ok']) else 'FAIL'

    # =============================================================== V4 finish (guards need the clause critical value)
    guards = {}
    for k, gv in hlj_guards.items():
        U1 = gv['point'] + c1 * gv['se']
        Lo, Up = gv['point'] - c38 * gv['se'], gv['point'] + c38 * gv['se']
        guards[k] = {**gv, 'cap': 0.005, 'clause_U1': U1, 'clause_pass': U1 <= 0.005, 'K38_interval': [Lo, Up],
                     'K38_status': V.status_at('no_excess', Lo, Up, 0.005), 'point_pass': gv['point'] <= 0.005,
                     'campaign_K38_status': camp['claims']['C_SUBGROUP']['guards_K38_status'][k],
                     'campaign_clause_pass': camp['claims']['C_SUBGROUP']['guards_clause_pass'][k]}
        guards[k]['agrees'] = guards[k]['K38_status'] == guards[k]['campaign_K38_status'] and guards[k]['clause_pass'] == guards[k]['campaign_clause_pass']
    hh = {g: groups[g]['hh'] for g in (0, 1)}
    v4 = {'check': 'V4 Subgroups', **common_meta, 'H_LJ_guards': guards,
          'households_per_sex': hh, 'campaign_households_per_sex': camp['subgroup_households_per_sex'],
          'all_services_vs_campaign': sub_t.out(), 'campaign_vs_source_published_exact': sub_cs.out(),
          'campaign_vs_source_published_se': sub_cs_se.out(),
          'note': 'H_LJ occupation output is JOINT_TREE occupation, so the guards are the source JOINT_TREE guards'}
    v4['status'] = 'PASS' if (sub_t.out()['ok'] and sub_cs.out()['ok'] and sub_cs_se.out()['ok'] and all(g['agrees'] for g in guards.values()) and len(guards) == 4
                             and hh == {int(k): v for k, v in camp['subgroup_households_per_sex'].items()}) else 'FAIL'

    # =============================================================== V7 finish: SE reproduction and service independence
    se_all = Tally()
    for k_ in list(E) + list(E2):
        th = camp['endpoints_primary'].get(k_) or camp['endpoints_secondary'].get(k_)
        se_all.add(k_, (E.get(k_) or E2.get(k_))['se'], th['se'])
    for s in V.SEEDS:
        for svc in V.ROSTER:
            for (t, ref, metric) in [(t, r, m_) for t in V.TASKS for r in V.REFS for m_ in V.METRICS]:
                se_all.add(f'S{s}|{svc}|{t}|{ref}|{metric}', V.sd(util[(svc, s, t, ref, metric)][1]),
                           camp['per_seed'][f'S{s}']['services'][svc]['utility'][f'{t}|{ref}|{metric}']['se'])
            for g in (0, 1):
                se_all.add(f'S{s}|{svc}|sex{g}', V.sd(sub[(svc, s, g)][1]), camp['per_seed'][f'S{s}']['services'][svc]['subgroup'][f'sex{g}']['se'])
            for view in V.VIEWS:
                se_all.add(f'S{s}|{svc}|{view}|auc', V.sd(auc_dr[(svc, s, view)]), camp['per_seed'][f'S{s}']['services'][svc]['recovery'][view]['auc_se'])
    # power check: a different seed or a service-specific redraw would not reproduce the SEs
    W_alt, _, _ = V.household_weights(unit, V.BOOT_B, V.BOOT_SEED + 1)
    k0 = 'RECIPE|H_LJ|coalition_benefit_vs_TASK_REFIT|s0'
    sp0 = next(sp for sp in L['primary_roster'] if sp['key'] == k0)
    e0 = sp0['estimand']
    alt = V.auc_weighted_draws(sex, scores[0][winners['S0'][e0['plus']]['pair']], W_alt) - V.auc_weighted_draws(sex, scores[0][winners['S0'][e0['minus']]['pair']], W_alt)
    d_u = vec[('H_LJ', 0, 'occupation')]['brier'] - vec[('U0', 0, 'occupation')]['brier']
    alt_u = V.sd(V.boot_mean(W_alt, d_u))
    del W_alt
    v7['se_reproduction_all_campaign_ses'] = se_all.out()
    v7['power_check_seed_plus_1'] = {
        k0: {'se_registered_draws': E[k0]['se'], 'se_seed_plus_1': V.sd(alt), 'relative_change': abs(V.sd(alt) - E[k0]['se']) / E[k0]['se']},
        'H_LJ|occupation|U0|brier_excess|s0': {'se_registered_draws': V.sd(util[('H_LJ', 0, 'occupation', 'U0', 'brier_excess')][1]), 'se_seed_plus_1': alt_u}}
    v7['service_independence'] = {
        'construction_inputs': ['DEV_REPORT household ids (unit)', 'B', 'seed'],
        'one_matrix_reproduces_all_services_and_both_seeds': se_all.out()['ok'],
        'services_covered': list(V.ROSTER), 'seeds_covered': list(V.SEEDS),
        'note': 'every campaign SE (all services, endpoints, utilities, subgroups and recovery AUCs, both seeds) is reproduced with the '
                'single verifier matrix; a redraw with any other seed moves SEs by ~1-5 percent, so a service- or seed-specific matrix '
                'would have been detected'}
    v7['status'] = 'PASS' if (v7['cross_check_task_headroom_group_weights']['bitwise_equal'] and G == 3955 and v7['draw_rows_sum_to_G_households']
                             and se_all.out()['ok'] and v7['power_check_seed_plus_1'][k0]['relative_change'] > 1e-6) else 'FAIL'

    # =============================================================== V9 decisions
    cr = L['controls']
    lo, hi = cr['null_band']
    wlo, whi = cr['null_wide_band_disclosure_only']
    st2, ctl_t = {}, Tally()
    for s in V.SEEDS:
        for view in V.VIEWS:
            sc = scores[s]
            pl = np.where(sex == 1, sc[f'CONTROL__{view}__PLANTED_1'], sc[f'CONTROL__{view}__PLANTED_0'])
            nul = sc[f'CONTROL__{view}__SHUFFLED']
            pa, na = V.auc(sex, pl), V.auc(sex, nul)
            nse = V.sd(V.auc_weighted_draws(sex, nul, W))
            s1 = stage1['per_view'][f'S{s}|{view}']
            st2[f'S{s}|{view}'] = {'planted_dev_report_auc': pa, 'null_dev_report_auc': na, 'null_se': nse, 'null_z': (na - 0.5) / nse,
                                   'NULL_WIDE': bool(not (wlo <= na <= whi) and lo <= na <= hi),
                                   'planted_attack_validation_auc_refit': s1['refit_planted_attack_validation_auc'],
                                   'null_attack_validation_auc_refit': s1['refit_null_attack_validation_auc']}
            c = camp['per_seed'][f'S{s}']['controls'][view]
            ctl_t.add(f'S{s}|{view}|planted', pa, c['planted_bit_auc'])
            ctl_t.add(f'S{s}|{view}|null', na, c['shuffled_null_auc'])
            ctl_t.add(f'S{s}|{view}|null_se', nse, c['shuffled_null_auc_se'])
            ctl_t.eq(f'S{s}|{view}|NULL_WIDE', st2[f'S{s}|{view}']['NULL_WIDE'], c['NULL_WIDE'])
            ctl_t.add(f'S{s}|{view}|planted_av', s1['refit_planted_attack_validation_auc'], c['planted_attack_validation_auc'])
            ctl_t.add(f'S{s}|{view}|null_av', s1['refit_null_attack_validation_auc'], c['shuffled_attack_validation_auc'])
    planted_vals = [v['planted_dev_report_auc'] for v in st2.values()] + [v['planted_attack_validation_auc_refit'] for v in st2.values()]
    null_vals = [v['null_dev_report_auc'] for v in st2.values()] + [v['null_attack_validation_auc_refit'] for v in st2.values()]
    PLANTED = len(planted_vals) == 12 and all(np.isfinite(x) and x >= cr['planted_min_auc'] for x in planted_vals)
    NULL = len(null_vals) == 12 and all(np.isfinite(x) and lo <= x <= hi for x in null_vals)
    valid = bool(PLANTED and NULL)
    controls = {'stage_1_refit_summary': stage1['stage_1'], 'stage_2': st2, 'PLANTED_PASS': PLANTED, 'NULL_PASS': NULL, 'CONTROLS_VALID': valid,
                'null_wide_flags': [k for k, v in st2.items() if v['NULL_WIDE']],
                'campaign': camp['controls'], 'values_vs_campaign': ctl_t.out()}
    controls['agrees'] = bool(PLANTED == camp['controls']['PLANTED_PASS'] and NULL == camp['controls']['NULL_PASS'] and valid == camp['controls']['CONTROLS_VALID']
                              and controls['null_wide_flags'] == camp['controls']['null_wide_flags'] and ctl_t.out()['ok']
                              and stage1['stage_1']['CONTROLS_STAGE1_VALID_recomputed'] == RL['stage_1_controls']['CONTROLS_STAGE1_VALID'])
    ck = L['claim_keys']
    # claim keys consistent with the roster
    keys_ok = (set(ck['A_utility']) == {sp['key'] for sp in L['primary_roster'] if sp['estimand']['type'] == 'utility'}
               and set(ck['C_subgroup']) == {sp['key'] for sp in L['primary_roster'] if sp['estimand']['type'].startswith('subgroup')}
               and set(ck['B_competitive']) == {sp['key'] for sp in L['primary_roster'] if sp.get('claim') == 'B'}
               and set(ck['A_coalition_benefit']) | set(ck['A_recipient_worsening']) | set(ck['A_utility']) == {sp['key'] for sp in L['primary_roster'] if sp.get('claim') == 'A'}
               and set(ck['increment']) == {sp['key'] for sp in L['primary_roster'] if sp.get('claim') is None})
    labA = L['claims']['A']
    gate = all(E[k]['point_pass'] for k in ck['A_utility'])
    ben = {k: E[k]['L1'] > 0.020 for k in ck['A_coalition_benefit']}
    wor = {k: E[k]['U1'] <= 0.010 for k in ck['A_recipient_worsening']}
    estA = all(ben.values()) and all(wor.values()) and gate
    dA = labA['invalid_label'] if not valid else (labA['labels'][0] if estA else labA['labels'][1])
    margin_of = {sp['key']: sp['margin'] for sp in L['primary_roster']}
    nestA = labA['invalid_label'] if not valid else (labA['nested_labels'][0] if (estA and all(E[k]['U1'] <= margin_of[k] for k in ck['A_utility'])) else labA['nested_labels'][1])
    failedA = [k for k in ck['A_coalition_benefit'] if not ben[k]] + [k for k in ck['A_recipient_worsening'] if not wor[k]] + [k for k in ck['A_utility'] if not E[k]['point_pass']]
    cA = camp['claims']['A_RECIPE_BENEFIT']
    A_ = {'coalition_benefit_L1': {k: E[k]['L1'] for k in ck['A_coalition_benefit']}, 'coalition_benefit_clause_pass': ben,
          'recipient_worsening_U1': {k: E[k]['U1'] for k in ck['A_recipient_worsening']}, 'recipient_worsening_clause_pass': wor,
          'utility_point_gate_24': gate, 'failed_clauses': failedA, 'decision': dA, 'nested_decision_utility_bounds': nestA,
          'campaign_decision': cA['decision'], 'campaign_nested': cA['nested_decision_utility_bounds'], 'campaign_failed_clauses': cA['failed_clauses']}
    A_['agrees'] = bool(dA == cA['decision'] and nestA == cA['nested_decision_utility_bounds'] and failedA == cA['failed_clauses']
                        and ben == cA['coalition_benefit_clause_pass'] and wor == cA['recipient_worsening_clause_pass'] and gate == cA['utility_point_gate_24'])
    # claim B with the registered precedence
    labB = L['claims']['B']['labels']
    sel = V.jload(V.SRC_RES / 'SELECTION_LOCK.json')
    nr_ids = {s: winners[f'S{s}']['NEURAL_RAW']['pair'] for s in V.SEEDS}
    nr_complete = all(nr_ids[s] in scores[s] and np.isfinite(scores[s][nr_ids[s]]).all() for s in V.SEEDS)
    nr_eligible = sel['frozen']['NEURAL_RAW']['status'] == 'ELIGIBLE'
    comp_keys = ck['B_competitive']
    sup = all(E[k]['L1'] > 0 for k in comp_keys)
    mat = all(E[k]['L1'] > 0.020 for k in comp_keys)
    trace = []
    if not nr_complete:
        dB = labB['incomplete']; trace.append('rule 1 INCOMPLETE applies')
    else:
        trace.append('rule 1 INCOMPLETE: NEURAL_RAW pair scores present and finite for both seeds -> not applicable')
        if not valid:
            dB = labB['invalid']; trace.append('rule 2 INVALID_CONTROL_FAILURE applies')
        else:
            trace.append('rule 2: controls valid -> not applicable')
            if not nr_eligible:
                dB = labB['no_eligible']; trace.append('rule 3 NO_ELIGIBLE_BASELINE applies')
            else:
                trace.append('rule 3: SELECTION_LOCK.frozen.NEURAL_RAW.status == ELIGIBLE -> not applicable')
                if not (ms_ok_hlj and gate):
                    dB = labB['fails_gate']; trace.append('rule 4 H_LJ_FAILS_SAME_UTILITY_REQUIREMENTS applies')
                else:
                    trace.append('rule 4: H_LJ MODEL_SELECT point feasible (24) and DEV_REPORT point gate (24) hold -> not applicable')
                    if sup:
                        dB = labB['superior']; trace.append('rule 5 SUPERIOR applies (both L1 > 0)')
                    else:
                        dB = labB['not_superior']; trace.append('rule 5 fails (not both L1 > 0) -> rule 6 NOT_SUPERIOR')
    nestB = L['claims']['B']['nested_labels'][0] if (dB == labB['superior'] and mat) else L['claims']['B']['nested_labels'][1]
    cB = camp['claims']['B_COMPETITIVE']
    # static check: order of the precedence branches in the campaign report code (text inspection only, not imported)
    src_txt = (V.ROOT / 'acs_hybrids' / 'report.py').read_text()
    blk = src_txt[src_txt.index('if not nr_complete:'):]
    blk = blk[:blk.index('Bc = {')]
    pos = [blk.find(f"labB['{x}']") for x in ('incomplete', 'invalid', 'no_eligible', 'fails_gate', 'superior')]
    B_ = {'eligible_set': ['NEURAL_RAW'], 'neural_raw_pair_scores_complete_finite': nr_complete, 'neural_raw_eligible_in_SELECTION_LOCK': nr_eligible,
          'H_LJ_model_select_point_feasible_24': ms_ok_hlj, 'H_LJ_dev_report_point_gate_24': gate,
          'L1': {k: E[k]['L1'] for k in comp_keys}, 'clause_pass_vs_0': {k: E[k]['L1'] > 0 for k in comp_keys},
          'L1_gt_0.020': {k: E[k]['L1'] > 0.020 for k in comp_keys},
          'K38_status_at_0_and_0.020': {k: {m_: E[k]['at'][m_]['status_at_m'] for m_ in E[k]['at']} for k in comp_keys},
          'precedence_trace': trace, 'decision': dB, 'nested_material_superiority_0.020': nestB,
          'campaign_decision': cB['decision'], 'campaign_nested': cB['nested_material_superiority_0.020'],
          'campaign_report_code_branch_order_incomplete_invalid_noeligible_failsgate_superior': bool(all(p >= 0 for p in pos) and pos == sorted(pos))}
    B_['agrees'] = bool(dB == cB['decision'] and nestB == cB['nested_material_superiority_0.020']
                        and B_['clause_pass_vs_0'] == cB['clause_pass_vs_0'] and B_['L1_gt_0.020'] == cB['clause_L1_gt_0.020']
                        and B_['K38_status_at_0_and_0.020'] == cB['K38_status_at_0_and_0.020']
                        and cB['h_lj_same_utility_requirements'] == {'model_select_point_feasible_24': ms_ok_hlj, 'dev_report_point_feasible_24': gate}
                        and B_['campaign_report_code_branch_order_incomplete_invalid_noeligible_failsgate_superior'])
    # claim C
    labC = L['claims']['C']['labels']
    estimable = all(v >= inf['subgroup_min_households_per_sex'] for v in hh.values())
    sg = ck['C_subgroup']
    dC = (labC['not_estimable'] if not estimable else labC['qualified'] if all(E[k]['U1'] <= 0.005 for k in sg)
          else labC['harm'] if any(E[k]['L'] > 0.005 for k in sg) else labC['unresolved'])
    cC = camp['claims']['C_SUBGROUP']
    C_ = {'estimable': estimable, 'households_per_sex': hh, 'guards_U1': {k: E[k]['U1'] for k in sg}, 'guards_clause_pass': {k: E[k]['U1'] <= 0.005 for k in sg},
          'guards_K38_status': {k: E[k]['status'] for k in sg}, 'harm_guards': [k for k in sg if E[k]['L'] > 0.005], 'decision': dC, 'campaign_decision': cC['decision']}
    C_['agrees'] = bool(dC == cC['decision'] and C_['guards_clause_pass'] == cC['guards_clause_pass'] and C_['guards_K38_status'] == cC['guards_K38_status']
                        and C_['harm_guards'] == cC['harm_guards'] and estimable == cC['estimable'])
    # separate status fields
    ut = {k: E[k]['status'] for k in ck['A_utility']}
    sep = {'overall_point_feasibility': {'DEV_REPORT': 'POINT_FEASIBLE_24' if gate else 'POINT_INFEASIBLE',
                                         'DEV_REPORT_failed_keys': [k for k in ck['A_utility'] if not E[k]['point_pass']],
                                         'MODEL_SELECT': 'POINT_FEASIBLE_24' if ms_ok_hlj else 'POINT_INFEASIBLE'},
           'overall_confidence_status': {'status': 'CONFIDENCE_FEASIBLE_24' if all(v == 'PASS' for v in ut.values()) else (
               'ESTABLISHED_UTILITY_VIOLATION' if any(v == 'ESTABLISHED_VIOLATION' for v in ut.values()) else 'CONFIDENCE_UNRESOLVED'),
               'violation_keys': [k for k, v in ut.items() if v == 'ESTABLISHED_VIOLATION'],
               'counts': {x: sum(v == x for v in ut.values()) for x in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
               'clause_bounds_passing_of_24': int(sum(E[k]['U1'] <= margin_of[k] for k in ck['A_utility']))},
           'subgroup_status': {'decision': dC, 'guards_K38_status': C_['guards_K38_status']}}
    sep_agrees = sep == camp['separate_status_fields']
    # conjunction
    unmet = []
    if not valid:
        unmet.append('CONTROL_FAILURE')
    if not gate:
        unmet.append('DEV_REPORT utility point gate: ' + ', '.join(sep['overall_point_feasibility']['DEV_REPORT_failed_keys']))
    if not ms_ok_hlj:
        unmet.append('MODEL_SELECT feasibility')
    unmet += [f'coalition benefit clause {k}' for k in ck['A_coalition_benefit'] if not ben[k]]
    unmet += [f'recipient worsening clause {k}' for k in ck['A_recipient_worsening'] if not wor[k]]
    if dB in (labB['incomplete'], labB['no_eligible']):
        unmet.append(f'claim B {dB}')
    unmet += [f'COMPETITIVE benefit clause {k}' for k in comp_keys if not E[k]['L1'] > 0]
    supported = valid and dA == labA['labels'][0] and dB == labB['superior']
    overall = L['central_question']['labels'][0] if supported else L['central_question']['labels'][1]
    cq = camp['central_question']
    conj = {'overall': overall, 'unmet_components_in_order': unmet, 'campaign_overall': cq['overall'], 'campaign_unmet': cq['unmet_components_in_order'],
            'Q1_point_utility_inherited': gate, 'Q2_material_improvement_over_TASK_REFIT': dA == labA['labels'][0], 'Q3_beats_neural_control': dB == labB['superior']}
    conj['agrees'] = bool(overall == cq['overall'] and unmet == cq['unmet_components_in_order'] and conj['Q1_point_utility_inherited'] == cq['Q1_point_utility_inherited']
                          and conj['Q2_material_improvement_over_TASK_REFIT'] == cq['Q2_material_improvement_over_TASK_REFIT']
                          and conj['Q3_beats_neural_control'] == cq['Q3_beats_neural_control'])
    inc = {k: {'point': E[k]['point'], 'se': E[k]['se'], 'K38': [E[k]['L'], E[k]['U']], 'status_at_0_and_0.020': E[k]['status'],
               'campaign': camp['endpoints_primary'][k]['status']} for k in ck['increment']}
    inc_ok = all(v['status_at_0_and_0.020'] == v['campaign'] for v in inc.values())
    settled_cmp = {'utility_point_gate_24': [settled['dev_report_utility_point_gate_24'], gate],
                   'utility_clause_bounds_passing_of_24': [settled['utility_clause_bounds_passing_of_24'], sep['overall_confidence_status']['clause_bounds_passing_of_24']],
                   'claim_C': [settled['claim_C_outcome_implied'], dC],
                   'income_worsening_clause_passes': [settled['income_worsening_clause_passes'], {k: wor[k] for k in ck['A_recipient_worsening'] if 'income' in k}],
                   'utility_K38_status_counts': [settled['utility_K38_status_counts'], sep['overall_confidence_status']['counts']],
                   'subgroup_K38_statuses': [settled['subgroup_K38_statuses'], C_['guards_K38_status']]}
    settled_ok = all(v[0] == v[1] for v in settled_cmp.values())
    v9 = {'check': 'V9 Decisions and labels', **common_meta, 'claim_keys_consistent_with_roster': keys_ok,
          'controls': controls, 'claim_A': A_, 'claim_B': B_, 'claim_C': C_, 'separate_status_fields': sep,
          'separate_status_fields_equal_campaign': sep_agrees, 'increment': inc, 'conjunction': conj,
          'settled_before_readers_vs_recomputed': settled_cmp, 'settled_agree': settled_ok,
          'scope_note': 'Stage-1 control AUCs come from the verifier\'s 12 control REFITS (v9_controls); Stage-2 values replay the stored '
                        'DEV_REPORT control scores with truth SEX; all claim logic is the verifier\'s own reading of ANALYSIS_LOCK'}
    v9['status'] = 'PASS' if (keys_ok and controls['agrees'] and A_['agrees'] and B_['agrees'] and C_['agrees'] and sep_agrees and conj['agrees']
                             and inc_ok and settled_ok) else 'FAIL'

    for name, obj in (('V3_utility.json', v3), ('V4_subgroups.json', v4), ('V7_household_bootstrap.json', v7),
                      ('V8_endpoints_and_multiplicity.json', v8), ('V9_decisions_and_labels.json', v9)):
        V.write_json(outdir / name, obj)
    print({k: o['status'] for k, o in (('V3', v3), ('V4', v4), ('V7', v7), ('V8', v8), ('V9', v9))}, 'synthetic' if a.synthetic else 'REAL')


if __name__ == '__main__':
    main()

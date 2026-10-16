"""Open DEV_REPORT once (after the committed REPORTING_LOCK) and evaluate the frozen roster.

Inference (SCIENCE_LOCK 'inference'): one nonparametric household (cluster) bootstrap (B = 2048, seed 2026101603) over DEV_REPORT
households, shared across every service, endpoint and both fitted seeds (paired). Every bound is Wald-type: point +/- t(G - 1) x
bootstrap SE (G = DEV_REPORT households); this is not a studentized bootstrap-t.
Decision family (positive direction only): three claims, each an intersection-union test (IUT) whose bound-based clauses are tested at
one-sided level alpha/3 (alpha = 0.025; Bonferroni across the three claims). Hence P(at least one of the three positive claim decisions is
false) <= 0.025 ONE-SIDED (each claim <= 0.0083 one-sided), approximately (Wald clause tests), conditional on the fitted models and readers.
RECIPE_EFFECT = 6 bound-based clauses plus a 24-clause utility POINT-feasibility gate (a deterministic condition on the estimates, with no
error control); its nested decision that also requires the 24 utility clauses to pass at their alpha/3 bounds shares the clause tests
and needs no further alpha. COMPETITIVE_COMPARISON = coalition benefit > 0 vs every eligible required baseline, both seeds, gated by
LOCAL_TREE meeting the same utility requirements. SUBGROUP_SUITABILITY = four guards. The IUT is NOT a simultaneous coverage statement:
per-clause alpha/3 bounds are marginal and can neither establish violations, shortfalls or harm nor be counted after a failed conjunction.
Reporting family: two-sided 95% intervals over the frozen K = 40 primary roster (Bonferroni t), with approximate simultaneous coverage
conditional on the fitted models and readers; every failing-direction label (ESTABLISHED_VIOLATION, ESTABLISHED_SHORTFALL,
ESTABLISHED_SUBGROUP_HARM) is read from these simultaneous intervals only.
Scope: the two seeds share people and are not independent samples; each seed is its own clause, so 'both seeds' is a two-fit
replication, not an estimate of between-fit variance. The household bootstrap treats DEV_REPORT households as a simple random sample
from a household superpopulation (unweighted ACS-sample composition); it is not ACS survey-design variance (no strata, PUMA clustering
or PWGTP1-80 replicate weights). PWGTP-weighted values are point summaries only.
Pre-registered sensitivity (reported, never decisive): one-sided percentile bounds from the same draws and bootstrap skewness per
roster endpoint; every disagreement with the primary Wald decision is disclosed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import registry

BOOT_SEED = 2026101603
B = 2048
ALPHA = 0.025
N_CLAIMS = 3
K_PRIMARY = 40
RECIPE_MARGIN = 0.020
WORSENING_CAP = 0.010
SUBGROUP_CAP = 0.005
REQUIRED_BASELINES = ('NEURAL_RAW', 'FAGTB', 'FARE')
ACCESS_LOG = D.PRIV / 'access' / 'DEV_REPORT_ACCESS.jsonl'


def crit_one_sided(G):
    from scipy import stats
    return float(stats.t.ppf(1 - ALPHA / N_CLAIMS, G - 1))


def crit_simultaneous(G):
    from scipy import stats
    return float(stats.t.ppf(1 - 0.05 / (2 * K_PRIMARY), G - 1))


def endpoint(point, draws, kind, margin, c1, cs, in_primary=True):
    """Wald-type bounds: point +/- t(G-1) x paired household-bootstrap SE (not a studentized bootstrap-t).
    iut_*: marginal one-sided (1 - alpha/3) bounds, used ONLY for positive-direction claim clauses (iut_pass).
    sim95_*: K = 40 Bonferroni two-sided bounds; the ONLY source of failing-direction labels (sim_status)."""
    draws = np.asarray(draws, dtype=np.float64)
    se = float(np.std(draws, ddof=1))
    a = ALPHA / N_CLAIMS
    rec = {'point': float(point), 'se': se, 'kind': kind, 'margin': margin, 'in_primary_family': bool(in_primary),
           'iut_lower': float(point - c1 * se), 'iut_upper': float(point + c1 * se),
           'sim95_lower': float(point - cs * se), 'sim95_upper': float(point + cs * se),
           'construction': 'point +/- t(G-1) x paired household-bootstrap SE; conditional on fitted models and readers',
           'sensitivity_percentile_lower': float(np.quantile(draws, a)), 'sensitivity_percentile_upper': float(np.quantile(draws, 1 - a)),
           'bootstrap_skewness': float(((draws - draws.mean()) ** 3).mean() / max(draws.std() ** 3, 1e-300))}
    if kind == 'benefit':
        rec['iut_pass'] = bool(rec['iut_lower'] > margin)
        rec['sensitivity_pass'] = bool(rec['sensitivity_percentile_lower'] > margin)
        rec['sim_status'] = 'ESTABLISHED' if rec['sim95_lower'] > margin else ('ESTABLISHED_SHORTFALL' if rec['sim95_upper'] < margin else 'UNRESOLVED')
    else:
        rec['iut_pass'] = bool(rec['iut_upper'] <= margin)
        rec['sensitivity_pass'] = bool(rec['sensitivity_percentile_upper'] <= margin)
        rec['sim_status'] = 'PASS' if rec['sim95_upper'] <= margin else ('ESTABLISHED_VIOLATION' if rec['sim95_lower'] > margin else 'UNRESOLVED')
    rec['point_pass'] = bool(point > margin) if kind == 'benefit' else bool(point <= margin)
    rec['sensitivity_disagrees_with_primary'] = rec['sensitivity_pass'] != rec['iut_pass']
    return rec


def weighted_means(v, W):
    return (W * v[None, :]).sum(1) / W.sum(1)


def run(out_dir):
    from task_headroom.inference import group_weights
    from task_conditioned_representation.inference import auc, bootstrap_auc
    if not D.lock_committed('REPORTING_LOCK.json'):
        raise PermissionError('DEV_REPORT opens only after the committed REPORTING_LOCK')
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if ACCESS_LOG.exists() and any(json.loads(l).get('event') == 'DEV_REPORT_TRUTH_OPENED' for l in ACCESS_LOG.read_text().splitlines() if l.strip()):
        raise PermissionError('DEV_REPORT truth was already opened once; no second opening in this campaign')
    with open(ACCESS_LOG, 'a') as f:
        f.write(json.dumps({'event': 'DEV_REPORT_TRUTH_OPENED', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                            'reporting_lock_sha256': registry.sha(D.RESULTS / 'REPORTING_LOCK.json')}) + '\n')
    sel = json.loads((D.RESULTS / 'SELECTION_LOCK.json').read_text())
    R = D.load_role('DEV_REPORT', ('features', 'income', 'occupation', 'sex', 'weights'))
    y = {t: R[t] for t in D.TASKS}; sex = R['sex']; pw = R['weights'].astype(np.float64)
    W, G = group_weights(R['unit'], seed=BOOT_SEED, draws=B)
    c1, cs = crit_one_sided(G), crit_simultaneous(G)
    hh_per_sex = {int(g): int(len(np.unique(R['unit'][sex == g]))) for g in (0, 1)}
    from . import schema
    estimable_subgroups = all(v >= schema.SUPPORT['subgroup_households_min_per_sex_DEV_REPORT'] for v in hh_per_sex.values())
    roster = sel['attack_roster']
    res = {'schema': 'ACSR1-dev-report-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'households': int(G),
           'persons': int(len(sex)), 'bootstrap': {'B': B, 'seed': BOOT_SEED, 'unit': 'DEV_REPORT household', 'shared_across': 'services, endpoints, seeds'},
           'critical': {'alpha_family_one_sided': ALPHA, 'one_sided_iut_alpha_per_clause': ALPHA / N_CLAIMS, 't_one_sided': c1, 't_simultaneous_K40': cs},
           'subgroup_households_per_sex': hh_per_sex, 'subgroup_estimable': estimable_subgroups, 'per_seed': {}, 'endpoints': {}, 'claims': {},
           'descriptive': {}}
    auc_pt, auc_dr, util = {}, {}, {}
    reader_matrix = {}
    for seed in (0, 1):
        outs, hashes, missing = registry.load(seed, roles=('DEV_REPORT',))
        for fam, pat in registry.FILES.items():   # Rc11: service output rows align with DEV_REPORT rows where recorded
            p = registry.RUNS / pat.format(s=seed)
            if p.is_file():
                with np.load(p) as z:
                    if 'row_id__DEV_REPORT' in z.files and not np.array_equal(z['row_id__DEV_REPORT'], R['row_id']):
                        raise AssertionError(f'{fam} DEV_REPORT row alignment differs')
        sc_dir = D.PRIV / 'runs' / f'READERS_{seed}'
        rr = json.loads((sc_dir / 'READERS_SEED.json').read_text())
        with np.load(sc_dir / 'DEV_REPORT_SCORES.npz') as z:
            scores = {k: z[k].copy() for k in z.files}
        if not np.array_equal(scores['row_id'], R['row_id']):
            raise AssertionError('DEV_REPORT score alignment differs')
        refs = {r: {t: outs[r]['DEV_REPORT'][t] for t in D.TASKS} for r in ('U0', 'Ucal')}
        seed_rec = {'services': {}, 'controls': {}, 'output_file_sha256': hashes}
        for svc in roster:
            srec = {'recovery': {}, 'utility': {}, 'task_summary': {}, 'subgroup': {}}
            for view in ('income', 'occupation', 'pair'):
                win = rr['services'][svc]['winner'][view]['reader']
                s = scores[f'{svc}__{view}__{win}']
                auc_pt[(svc, seed, view)] = auc(sex, s); auc_dr[(svc, seed, view)] = bootstrap_auc(sex, s, W)
                srec['recovery'][view] = {'reader': win, 'auc': auc_pt[(svc, seed, view)], 'auc_se': float(np.std(auc_dr[(svc, seed, view)], ddof=1)),
                                          'pwgtp_weighted_auc_point': auc(sex, s, pw)}
                if svc in ('LOCAL_TREE', 'TASK_REFIT'):   # Rc5: full slate matrix, descriptive (selected on report data if maximised)
                    reader_matrix[(svc, seed, view)] = {k.split('__', 2)[2]: auc(sex, v) for k, v in scores.items()
                                                        if k.startswith(f'{svc}__{view}__')}
            for t in D.TASKS:
                k = D.K[t]; p = outs[svc]['DEV_REPORT'][t]
                v = D.task_vectors(y[t], p, k)
                srec['task_summary'][t] = {'unweighted': D.summary(y[t], p, k), 'pwgtp_weighted_point': D.summary(y[t], p, k, pw)}
                for ref in ('U0', 'Ucal'):
                    rv = D.task_vectors(y[t], refs[ref][t], k)
                    diffs = {'accuracy_loss': rv['accuracy'] - v['accuracy'], 'nll_excess': v['nll'] - rv['nll'], 'brier_excess': v['brier'] - rv['brier']}
                    for metric, d in diffs.items():
                        ep = endpoint(d.mean(), weighted_means(d, W), 'no_excess', D.CAPS[metric], c1, cs, in_primary=(svc == 'LOCAL_TREE'))
                        ep['pwgtp_weighted_point'] = float(np.sum(pw * d) / pw.sum())
                        util[(svc, seed, t, ref, metric)] = ep
                        srec['utility'][f'{t}|{ref}|{metric}'] = ep
            rv = D.task_vectors(y['occupation'], refs['U0']['occupation'], 5)
            v = D.task_vectors(y['occupation'], outs[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m = sex == g; d = v['brier'] - rv['brier']
                Wg = W[:, m]
                ep = endpoint(d[m].mean(), (Wg * d[m][None, :]).sum(1) / Wg.sum(1), 'no_excess', SUBGROUP_CAP, c1, cs, in_primary=(svc == 'LOCAL_TREE'))
                ep['group_households'] = hh_per_sex[g]; ep['estimable'] = estimable_subgroups
                ep['group_nll_excess_vs_U0'] = float((v['nll'] - rv['nll'])[m].mean())
                sm = D.summary(y['occupation'][m], outs[svc]['DEV_REPORT']['occupation'][m], 5)
                ep['group_accuracy'] = sm['accuracy']; ep['group_per_class_recall'] = sm['per_class_recall']
                ep['group_accuracy_loss_vs_U0'] = float((rv['accuracy'] - v['accuracy'])[m].mean())
                srec['subgroup'][f'sex{g}'] = ep
            seed_rec['services'][svc] = srec
        for view in ('income', 'occupation', 'pair'):
            seed_rec['controls'][view] = {'shuffled_null_auc': auc(sex, scores[f'CONTROL__{view}__SHUFFLED']),
                                          'planted_bit_auc': auc(sex, np.where(sex == 1, scores[f'CONTROL__{view}__PLANTED_1'], scores[f'CONTROL__{view}__PLANTED_0']))}
        res['per_seed'][str(seed)] = seed_rec
    res['descriptive']['reader_slate_auc_matrix'] = {f'{s}|s{sd}|{v}': m for (s, sd, v), m in reader_matrix.items()}
    # ---------------- primary endpoints (K = 40 minus absent baselines)
    E = res['endpoints']
    for seed in (0, 1):
        d = auc_dr[('TASK_REFIT', seed, 'pair')] - auc_dr[('LOCAL_TREE', seed, 'pair')]
        E[f'RECIPE|coalition_benefit_vs_TASK_REFIT|s{seed}'] = endpoint(auc_pt[('TASK_REFIT', seed, 'pair')] - auc_pt[('LOCAL_TREE', seed, 'pair')], d, 'benefit', RECIPE_MARGIN, c1, cs)
        for view in ('income', 'occupation'):
            d = auc_dr[('LOCAL_TREE', seed, view)] - auc_dr[('TASK_REFIT', seed, view)]
            E[f'RECIPE|{view}_recipient_worsening_vs_TASK_REFIT|s{seed}'] = endpoint(auc_pt[('LOCAL_TREE', seed, view)] - auc_pt[('TASK_REFIT', seed, view)], d, 'no_excess', WORSENING_CAP, c1, cs)
        for t in D.TASKS:
            for ref in ('U0', 'Ucal'):
                for metric in D.CAPS:
                    E[f'RECIPE|utility|{t}|{ref}|{metric}|s{seed}'] = util[('LOCAL_TREE', seed, t, ref, metric)]
    comp = {}
    for fam in REQUIRED_BASELINES:
        fz = sel['frozen'][fam]; svc = fz.get('configuration')
        comp[fam] = {'selection_status': fz['status'], 'service': svc}
        if not svc or svc not in roster:
            comp[fam]['service'] = None
            continue
        for seed in (0, 1):
            d = auc_dr[(svc, seed, 'pair')] - auc_dr[('LOCAL_TREE', seed, 'pair')]
            ep = endpoint(auc_pt[(svc, seed, 'pair')] - auc_pt[('LOCAL_TREE', seed, 'pair')], d, 'benefit', 0.0, c1, cs)
            ep['material_margin_0.020_iut_pass'] = bool(ep['iut_lower'] > 0.020)
            ep['material_margin_0.020_sim_status'] = 'ESTABLISHED' if ep['sim95_lower'] > 0.020 else ('ESTABLISHED_SHORTFALL' if ep['sim95_upper'] < 0.020 else 'UNRESOLVED')
            for view in ('income', 'occupation'):   # Rc9 descriptive per-recipient differences (outside K)
                ep[f'descriptive_{view}_recipient_auc_difference_baseline_minus_local'] = auc_pt[(svc, seed, view)] - auc_pt[('LOCAL_TREE', seed, view)]
            E[f'COMPETITIVE|coalition_benefit_vs_{fam}|s{seed}'] = ep
    for seed in (0, 1):
        for g in (0, 1):
            E[f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{seed}'] = res['per_seed'][str(seed)]['services']['LOCAL_TREE']['subgroup'][f'sex{g}']
    absent = sum(1 for f in REQUIRED_BASELINES if not comp[f]['service'])
    if len(E) != K_PRIMARY - 2 * absent:
        raise AssertionError(f'primary roster size {len(E)} differs from {K_PRIMARY - 2 * absent}')

    # ---------------- claims (decisions use iut_pass only; failing-direction labels use sim_status only)
    def members(pred, n):
        eps = [v for k, v in E.items() if pred(k)]
        if len(eps) != n:  # all([]) is True: never let a missing endpoint pass a conjunction
            raise AssertionError(f'claim roster mismatch: {len(eps)} endpoints, expected {n}')
        return eps
    ben = members(lambda k: k.startswith('RECIPE|coalition_benefit'), 2)
    wor = members(lambda k: k.startswith('RECIPE|') and '_recipient_worsening_' in k, 4)
    utl = members(lambda k: k.startswith('RECIPE|utility|'), 24)
    sg = members(lambda k: k.startswith('SUBGROUP|'), 4)
    rc = res['claims']['RECIPE_EFFECT'] = {
        'coalition_benefit_iut_pass_both_seeds': all(v['iut_pass'] for v in ben),
        'recipient_non_worsening_iut_pass_all': all(v['iut_pass'] for v in wor),
        'utility_point_feasible_24': all(v['point_pass'] for v in utl),
        'utility_iut_bound_pass_count_of_24': int(sum(v['iut_pass'] for v in utl)),
        'utility_simultaneous_status_counts': {s: sum(v['sim_status'] == s for v in utl) for s in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')}}
    six = rc['coalition_benefit_iut_pass_both_seeds'] and rc['recipient_non_worsening_iut_pass_all']
    rc['decision'] = 'RECIPE_EFFECT_ESTABLISHED' if six and rc['utility_point_feasible_24'] else 'RECIPE_EFFECT_NOT_ESTABLISHED'
    rc['nested_decision_utility_bounds'] = ('RECIPE_EFFECT_AND_UTILITY_BOUNDS_ESTABLISHED' if six and rc['utility_iut_bound_pass_count_of_24'] == 24
                                            else 'RECIPE_EFFECT_AND_UTILITY_BOUNDS_NOT_ESTABLISHED')
    rc['decision_scope'] = ('error-controlled (IUT, one-sided alpha/3 per clause): coalition benefit > 0.020 and each recipient worsening <= 0.010 '
                            'for both fitted seed pairs, conditional on the fitted models and frozen readers; utility: point-feasibility gate only '
                            '(no error control); utility bounds: nested decision (same clause tests, no extra alpha)')
    lt_select_feasible = bool(sel['fixed_services'].get('LOCAL_TREE', {}).get('point_feasible_24', False))
    same_requirements = lt_select_feasible and rc['utility_point_feasible_24']
    eligible = [f for f in REQUIRED_BASELINES if comp[f]['selection_status'] == 'ELIGIBLE' and comp[f]['service']]
    complete = all(comp[f]['service'] for f in REQUIRED_BASELINES) and bool(sel['frozen']['FAGTB'].get('two_task_valid', False))
    beats = {f: all(E[f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}']['iut_pass'] for s in (0, 1)) for f in REQUIRED_BASELINES if comp[f]['service']}
    material = {f: all(E[f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}']['material_margin_0.020_iut_pass'] for s in (0, 1)) for f in REQUIRED_BASELINES if comp[f]['service']}
    decision = ('INCOMPLETE' if not complete else
                'LOCAL_TREE_FAILS_SAME_UTILITY_REQUIREMENTS' if not same_requirements else
                'NO_ELIGIBLE_BASELINE' if not eligible else
                'SUPERIOR_TO_ALL_ELIGIBLE' if all(beats[f] for f in eligible) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')
    res['claims']['COMPETITIVE_COMPARISON'] = {
        'baselines': comp, 'eligible_required_baselines': eligible, 'complete': bool(complete),
        'local_tree_same_utility_requirements': {'model_select_point_feasible_24': lt_select_feasible,
                                                 'dev_report_point_feasible_24': rc['utility_point_feasible_24']},
        'superiority_iut_pass_vs': beats, 'decision': decision,
        'nested_material_superiority_0.020': ('MATERIALLY_SUPERIOR_TO_ALL_ELIGIBLE' if decision == 'SUPERIOR_TO_ALL_ELIGIBLE' and all(material[f] for f in eligible)
                                              else 'NOT_ESTABLISHED'),
        'decision_scope': 'coalition (pair) recovery only; margin 0; one-sided alpha/3 per clause; both seeds; versus exactly the ELIGIBLE baselines listed'}
    res['claims']['SUBGROUP_SUITABILITY'] = {
        'estimable': estimable_subgroups,
        'guards_iut_pass': {k: v['iut_pass'] for k, v in E.items() if k.startswith('SUBGROUP|')},
        'guards_simultaneous_status': {k: v['sim_status'] for k, v in E.items() if k.startswith('SUBGROUP|')},
        'decision': ('NOT_ESTIMABLE' if not estimable_subgroups else
                     'SUBGROUP_QUALIFIED' if all(v['iut_pass'] for v in sg) else
                     'ESTABLISHED_SUBGROUP_HARM' if any(v['sim_status'] == 'ESTABLISHED_VIOLATION' for v in sg) else
                     'SUBGROUP_UNRESOLVED'),
        'decision_scope': 'QUALIFIED: IUT over the 4 guards at one-sided alpha/3; HARM: K = 40 simultaneous lower bound > 0.005 only; crossing intervals are unresolved'}
    res['primary_roster_size'] = len(E)
    res['sensitivity_disagreements'] = [k for k, v in E.items() if v['sensitivity_disagrees_with_primary']]
    # Rc10: LOCAL_TREE minus TASK_REFIT utility differences (descriptive)
    res['descriptive']['local_minus_task_refit_utility'] = {
        f's{s}|{t}|{m}': util[('LOCAL_TREE', s, t, 'U0', m)]['point'] - util[('TASK_REFIT', s, t, 'U0', m)]['point'] for s in (0, 1) for t in D.TASKS for m in D.CAPS}
    # ---------------- planning for a later sealed assessment (PLANNING ONLY; not inference, not evidence)
    from scipy import stats
    zc, zp = stats.norm.ppf(1 - ALPHA / N_CLAIMS), stats.norm.ppf(0.8)
    slack = lambda v: (v['point'] - v['margin']) if v['kind'] == 'benefit' else (v['margin'] - v['point'])

    def power(v, n, shrink):
        s = slack(v)
        if s <= 0:
            return 0.0
        return 1.0 if v['se'] <= 0 else float(stats.norm.cdf(shrink * s * np.sqrt(n) / (v['se'] * np.sqrt(G)) - zc))
    per_clause = {}
    for k, v in E.items():
        sd_h = v['se'] * np.sqrt(G)
        per_clause[k] = {'household_sd': sd_h, 'slack_at_development_point': slack(v)}
        if slack(v) <= 0:
            per_clause[k]['status'] = 'FUTILE_AT_DEVELOPMENT_POINT'
        else:
            for lab, sh in (('at_point', 1.0), ('at_half_slack', 0.5)):
                per_clause[k][f'households_single_clause_80pct_{lab}'] = float(np.ceil(((zc + zp) * sd_h / (sh * slack(v))) ** 2))
    claim_keys = {'RECIPE_EFFECT': [k for k in E if k.startswith('RECIPE|') and '|utility|' not in k],
                  'RECIPE_EFFECT_AND_UTILITY_BOUNDS': [k for k in E if k.startswith('RECIPE|')],
                  'COMPETITIVE_COMPARISON': [f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}' for f in eligible for s in (0, 1)],
                  'SUBGROUP_SUITABILITY': [k for k in E if k.startswith('SUBGROUP|')]}
    per_claim = {}
    for name, keys in claim_keys.items():
        for mult in (1, 2, 4, 8, 16, 32):
            for lab, sh in (('at_point', 1.0), ('at_half_slack', 0.5)):
                pw_ = [power(E[k], mult * G, sh) for k in keys]
                if pw_:
                    per_claim.setdefault(name, {})[f'{mult}xG|{lab}'] = {
                        'households': int(mult * G), 'min_clause_power': min(pw_),
                        'conjunction_power_lower_bound_any_dependence': max(0.0, 1 - sum(1 - p for p in pw_))}
    res['planning'] = {'note': ('PLANNING ONLY. Development (DEV_REPORT) points and household SDs are used as assumed true values for the SAME frozen '
                                'models and readers scored on new households. Not uncertainty about the true effect, not evidence that any clause holds, '
                                'and optimistic if a favourable development result triggers the sealed assessment. An IUT claim needs every clause: '
                                'conjunction power lies between conjunction_power_lower_bound_any_dependence and min_clause_power. '
                                'FUTILE clauses cannot be rescued by more households. Refitting models would add variance not represented here.'),
                       'alpha_per_clause_one_sided': ALPHA / N_CLAIMS, 'per_clause': per_clause, 'per_claim': per_claim}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps(r['claims'], indent=1, default=str))

"""Builds RESERVE_RULE.json (statistical reviewer, PCRL income-headroom successor campaign) from the committed clause-level table
REFERENCE_MS_TABLE.csv only (no label, no person-level array, no private file), and screens every historical reference with
ihs.reserve on both seeds (with the registered floor and, as a diagnostic, with own SEs only).

    OMP_NUM_THREADS=1 python3 results/pcrl_income_headroom_successor_v1/roles/statistical_reviewer/code/make_reserve_rule.py
"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from ihs import reserve as R  # noqa: E402

RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
TABLE = RES / 'REFERENCE_MS_TABLE.csv'
OUT = HERE.parent / 'RESERVE_RULE.json'
K = 2.0
G = {'MODEL_SELECT': 5306, 'DEV_REPORT': 3955, 'SAME_POPULATION_UNTESTED_FL2019': 9853 + 7657}
FLOOR_REF = 'H_LJ'


def cname(cond):
    return 'guard|' + cond.split('|')[-1] if cond.startswith('guard|') else cond


def table():
    rows = list(csv.DictReader(open(TABLE)))
    out = {}
    for r in rows:
        if r['condition'].startswith('proxy|'):
            continue
        out.setdefault(r['reference'], {}).setdefault(r['seed'], {})[cname(r['condition'])] = {
            'value': float(r['value']), 'cap': float(r['cap']), 'se': float(r['se']), 'households': int(r['households'])}
    return out


def as_measure(conds, seed):
    cl, gu = [], []
    for n, c in conds.items():
        if n.startswith('guard|'):
            gu.append({'seed': seed, 'group': n.split('|')[1], 'value': c['value'], 'cap': c['cap'], 'se': c['se']})
        else:
            t, ref, met = n.split('|')
            cl.append({'seed': seed, 'task': t, 'reference': ref, 'metric': met, 'value': c['value'], 'cap': c['cap'], 'se': c['se']})
    hh = {c['households'] for c in conds.values()}
    assert hh == {G['MODEL_SELECT']}, hh
    return {'utility': {'clauses': cl, 'guards': gu, 'households': G['MODEL_SELECT']}}


def derivation():
    N = NormalDist()
    r17 = math.sqrt(G['MODEL_SELECT'] / G['SAME_POPULATION_UNTESTED_FL2019'])
    rdev = math.sqrt(G['MODEL_SELECT'] / G['DEV_REPORT'])
    z95, z80 = N.inv_cdf(0.95), N.inv_cdf(0.80)
    plug = (z95 + z80) * r17
    return {
        'multiplier_k_at_MODEL_SELECT': K,
        'equivalent_multiplier_at_17510_households': K / r17,
        'equivalent_multiplier_at_DEV_REPORT_3955_households': K / rdev,
        'sqrt_5306_over_17510': r17,
        'sqrt_5306_over_3955': rdev,
        'z_0.95': z95, 'z_0.80': z80,
        'plugin_requirement_one_sided_5pct_80pct_power_at_17510_in_MS_SE': plug,
        'development_allowance_in_MS_SE': K - plug,
        'flat_prior_P_true_margin_meets_17510_requirement_at_k': N.cdf(K - plug),
        'flat_prior_P_true_margin_positive_at_k': N.cdf(K),
        'predictive_P_point_pass_same_population_DEV_size_at_k': N.cdf(K / math.sqrt(1 + rdev ** 2)),
        'predictive_P_point_pass_same_population_17510_at_k': N.cdf(K / math.sqrt(1 + r17 ** 2)),
        'predictive_P_one_sided_5pct_bound_below_cap_at_17510_at_k': N.cdf((K - z95 * r17) / math.sqrt(1 + r17 ** 2)),
        'note': ('per condition, normal approximation, iid households from the same population, unweighted, flat prior, no '
                 'selection adjustment; the income clauses are strongly correlated, so joint probabilities are lower when '
                 'several sit at the threshold. Planning arithmetic, not confidence statements.')}


def build():
    tab = table()
    floors, inh_vals, inh_marg = {}, {}, {}
    names = R.INCOME + R.OCCUPATION + R.GUARDS
    for s in ('0', '1'):
        h = tab[FLOOR_REF][s]
        floors[s] = {n: h[n]['se'] for n in names}
        inh_vals[s] = {n: h[n]['value'] for n in R.OCCUPATION + R.GUARDS}
        inh_marg[s] = {n: h[n]['cap'] - h[n]['value'] for n in R.OCCUPATION + R.GUARDS}
    rule = {
        'schema': R.SCHEMA,
        'campaign': 'pcrl_income_headroom_successor_v1',
        'author_role': 'baseline and statistical reviewer',
        'what': ('utility-reserve screen: a development selection heuristic on adaptively used MODEL_SELECT data; not a '
                 'confidence guarantee; not a changed cap (the original caps and the point-eligibility gate are unchanged)'),
        'original_caps_unchanged': {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005, 'within_sex_occupation_guard': 0.005},
        'multiplier_k': K,
        'se_rule': 'se_used = max(own household-clustered paired SE on the evaluated role, floor SE of the same condition and seed)',
        'se_scale': {'role': 'MODEL_SELECT', 'households': G['MODEL_SELECT'],
                     'projection': ('none: SEs are used at the 5,306-household scale at which they are measured; the 17,510-household '
                                    'same-population scale enters through the multiplier derivation (k = 2 at MODEL_SELECT equals '
                                    '3.633 SEs at 17,510 households)')},
        'floor': {'source': f'{FLOOR_REF} MODEL_SELECT household-clustered paired SE of the same condition and seed '
                            '(REFERENCE_MS_TABLE.csv); rescaled by sqrt(5306 / households) for an evaluation with another household count',
                  'reference': FLOOR_REF, 'households': G['MODEL_SELECT'], 'se': floors,
                  'absolute_minimum_income_margin_k_times_floor': {s: {n: K * floors[s][n] for n in R.INCOME} for s in ('0', '1')}},
        'income_conditions': list(R.INCOME),
        'occupation': {
            'conditions': list(R.OCCUPATION + R.GUARDS),
            'inherited_output': 'H_LJ occupation output (JOINT_TREE occupation, E20 corrections on the frozen T_TREE routing)',
            'inherited_values': inh_vals, 'inherited_margins': inh_marg,
            'identity_tolerance': 1e-12, 'inherited_wire_sha256': None,
            'rule_if_inherited': 'not screened; returned as the inherited-constraint report (INHERITED_CONSTRAINTS_BIND when any condition is below the reserve)',
            'rule_if_changed': ('each of the 6 occupation clauses and 2 guards must meet the reserve (margin >= k * max(own SE, H_LJ SE)) '
                                'or keep at least the inherited margin (non-worsening); otherwise it is a failed condition')},
        'status_per_seed': {'RESERVE_MET': 'all 6 income conditions met and no occupation condition worsened',
                            'RESERVE_NOT_MET': 'any income condition below k * se_used (or non-finite), or a changed occupation condition below both the reserve and the inherited margin'},
        'status_two_seeds': {'RESERVE_MET': 'seeds 0 and 1 both RESERVE_MET (the only label that meets the reserve)',
                             'RESERVE_ONE_SEED_ONLY': 'both seeds screened, exactly one met',
                             'RESERVE_NOT_MET': 'no screened seed met',
                             'RESERVE_PENDING_SECOND_SEED': 'one seed screened and met; the second seed has not been evaluated'},
        'ranking_quantity': 'min_income_ratio = minimum over the 12 income conditions of margin / se_used (floored SE units)',
        'derivation': derivation(),
        'source_table': {'path': 'results/pcrl_income_headroom_successor_v1/REFERENCE_MS_TABLE.csv',
                         'sha256': hashlib.sha256(TABLE.read_bytes()).hexdigest()},
        'implementation': 'ihs/reserve.py (screen, two_seed_status); invented-data tests ihs/test_reserve.py',
    }
    own_only = copy.deepcopy(rule)
    own_only['floor']['se'] = {s: {n: 0.0 for n in names} for s in ('0', '1')}
    res = {}
    for ref in tab:
        per, per_own = [], []
        for s in ('0', '1'):
            m = as_measure(tab[ref][s], int(s))
            per.append(R.screen(m, int(s), rule=rule)); per_own.append(R.screen(m, int(s), rule=own_only))
        two, two_own = R.two_seed_status(per), R.two_seed_status(per_own)
        res[ref] = {'two_seed_status': two['status'], 'two_seed_status_own_se_only': two_own['status'],
                    'min_income_ratio': two['min_income_ratio'], 'min_income_ratio_own_se_only': two_own['min_income_ratio'],
                    'per_seed': {str(sc['seed']): {'status': sc['status'], 'failed': sc['failed'], 'min_income_ratio': sc['min_income_ratio'],
                                                   'occupation_side': sc['occupation_side'],
                                                   'inherited_report': (sc['inherited_constraint_report'] or {}).get('status'),
                                                   'inherited_below_reserve': (sc['inherited_constraint_report'] or {}).get('below_reserve'),
                                                   'income_ratios': {r['condition']: r['ratio'] for r in sc['income']},
                                                   'income_se_source': {r['condition']: r['se_source'] for r in sc['income']},
                                                   'status_own_se_only': po['status'], 'failed_own_se_only': po['failed']}
                                 for sc, po in zip(per, per_own)}}
    rule['reference_screen'] = res
    return rule


def strict(x):
    """Strict JSON: non-finite floats become strings ('inf', '-inf', 'nan')."""
    if isinstance(x, float) and not math.isfinite(x):
        return 'nan' if math.isnan(x) else ('inf' if x > 0 else '-inf')
    if isinstance(x, dict):
        return {k: strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strict(v) for v in x]
    return x


if __name__ == '__main__':
    rule = build()
    OUT.write_text(json.dumps(strict(rule), indent=1, sort_keys=True, allow_nan=False) + '\n')
    for ref, r in rule['reference_screen'].items():
        print(f"{ref:14s} {r['two_seed_status']:22s} own-SE-only {r['two_seed_status_own_se_only']:22s} min ratio {r['min_income_ratio']:.4f}"
              f" (own {r['min_income_ratio_own_se_only']:.4f}) | " + ' | '.join(f"s{s}: {v['status']} {v['failed']} occ={v['occupation_side']}"
                                                                           for s, v in r['per_seed'].items()))
    print(json.dumps(rule['derivation'], indent=1))

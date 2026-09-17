"""Independent regeneration of the locked 2017 transport study's primary evidence.

Produces, under results/pcrl_evidence_review_v1/:

  INDEPENDENT_FAMILIES.csv    every F1-F4 endpoint recomputed from stored predictions
  FAMILY_AGREEMENT.json       row-by-row agreement against the study's FAMILIES.csv
  MODEB_TRANSPORT_TABLE.csv   absolute and additional recovery per interface
  SELECTION_RECHECK.json      selection re-derived from stored validation losses
  EQUIVALENCE_F1.csv          retrospective noninferiority / equivalence at margin .001

Run: python -m experiments.pcrl_evidence_review_v1.reanalyse_transport --study <dir>
Single process, one BLAS thread; per-interface arrays are released after use.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import independent_scorer as S

FAMILY_ENDPOINTS = ('utility/same_residence', 'recovery/A/SEX', 'recovery/AB/SEX',
                    'recovery/A/RAC1P', 'recovery/AB/RAC1P')
WEIGHTS = ('unweighted', 'person_weighted')
RESIDENCE_MARGIN = 0.001


def endpoint_vector(study, seed, mode, condition, scope, budget, endpoint, cache):
    """Per-person vector for one endpoint: utility loss, or prior-minus-attack recovery."""
    key = (seed, mode, condition, scope, budget, endpoint)
    if key in cache:
        return cache[key]
    kind, rest = endpoint.split('/', 1)
    if kind == 'utility':
        vec, _ = study.utility_loss(seed, mode, condition, rest)
        target = rest
    else:
        target = rest.split('/')[1]
        attack, _ = study.attack_loss(seed, mode, condition, scope, budget, rest)
        prior = prior_cache(study, seed, mode, target, cache)
        vec = prior - attack
    cache[key] = (target, vec)
    return cache[key]


def prior_cache(study, seed, mode, target, cache):
    key = ('prior', seed, mode, target)
    if key not in cache:
        cache[key] = study.prior_loss(seed, mode, target)
    return cache[key]


def contrast(study, boot, seeds, mode, scope, budget, left, right, endpoint, weight, cache):
    """Seed-averaged paired difference: bootstrap replicates, point, per-seed points."""
    reps = np.zeros(boot.replicates)
    point = 0.0
    per_seed = {}
    for seed in seeds:
        target, a = endpoint_vector(study, seed, mode, left, scope, budget, endpoint, cache)
        _, b = endpoint_vector(study, seed, mode, right, scope, budget, endpoint, cache)
        ra, pa = boot.ratio(target, weight, a)
        rb, pb = boot.ratio(target, weight, b)
        reps += (ra - rb) / len(seeds)
        point += (pa - pb) / len(seeds)
        per_seed[seed] = float(pa - pb)
    return reps, float(point), per_seed


def run_family(study, boot, seeds, name, spec, cache):
    mode, scope, budget = spec['mode'], spec['scope'], spec['budget']
    rows, reps = [], []
    for left, right in spec['contrasts']:
        for endpoint in spec['endpoints']:
            for weight in WEIGHTS:
                r, point, per_seed = contrast(study, boot, seeds, mode, scope, budget,
                                              left, right, endpoint, weight, cache)
                rows.append({
                    'family': name, 'left': left, 'right': right, 'endpoint': endpoint,
                    'weight': weight, 'estimate': point,
                    **{f'seed_{s}': v for s, v in per_seed.items()},
                    'se': float(r.std(ddof=1)),
                    'unadjusted_low': float(np.quantile(r, .025)),
                    'unadjusted_high': float(np.quantile(r, .975)),
                    'one_sided_upper_95_percentile': float(np.quantile(r, .95, method='higher')),
                })
                reps.append(r)
    R = np.column_stack(reps)
    est = np.array([row['estimate'] for row in rows])
    low, high, se, crit, live = S.max_t_intervals(R, est)
    one_sided_crit = S.one_sided_studentised_crit(R, est)
    for row, lo, hi, s_ in zip(rows, low, high, se):
        row.update(degenerate=not (s_ > 1e-15), adjusted_low=float(lo), adjusted_high=float(hi),
                   critical_value=crit, one_sided_critical_value=one_sided_crit,
                   one_sided_upper_95_simultaneous=float(row['estimate'] + one_sided_crit * s_))
    return rows, crit, R, est


def compare_to_study(rows, study_families_csv):
    """Row-by-row agreement with the published FAMILIES.csv."""
    published = {}
    with open(study_families_csv) as fh:
        for r in csv.DictReader(fh):
            published[(r['family'], r['left'], r['right'], r['endpoint'], r['weight'])] = r
    report = {'rows_compared': 0, 'max_abs_difference': {}, 'missing': [], 'worst': []}
    fields = ['estimate', 'se', 'unadjusted_low', 'unadjusted_high', 'adjusted_low', 'adjusted_high']
    worst = {f: 0.0 for f in fields}
    for row in rows:
        key = (row['family'], row['left'], row['right'], row['endpoint'], row['weight'])
        if key not in published:
            report['missing'].append(list(key))
            continue
        report['rows_compared'] += 1
        p = published[key]
        for f in fields:
            d = abs(float(p[f]) - float(row[f]))
            if d > worst[f]:
                worst[f] = d
                if f == 'estimate':
                    report['worst'] = [list(key), float(p[f]), float(row[f])]
    report['max_abs_difference'] = worst
    return report


def equivalence_table(rows, margin=RESIDENCE_MARGIN):
    """Retrospective assessment of the F1 residence endpoint against the .001 margin.

    The registered rule is a one-sided constraint on the *seed-mean point* difference.
    These rows add what the rule does not assert: whether the interval evidence would
    support noninferiority (upper bound < margin) or two-sided equivalence
    (whole interval inside +/- margin). Both are retrospective.
    """
    out = []
    for row in rows:
        if row['endpoint'] != 'utility/same_residence' or row['family'] not in ('F1_primary', 'F4_frozen_transfer'):
            continue
        out.append({
            'family': row['family'], 'contrast': f"{row['left']} - {row['right']}",
            'weight': row['weight'], 'estimate': row['estimate'],
            'registered_rule_point_le_margin': bool(row['estimate'] <= margin + 1e-12),
            'se': row['se'],
            'pointwise_upper_95_percentile': row['one_sided_upper_95_percentile'],
            'noninferior_pointwise': bool(row['one_sided_upper_95_percentile'] < margin),
            'simultaneous_upper_95_studentised': row['one_sided_upper_95_simultaneous'],
            'noninferior_simultaneous': bool(row['one_sided_upper_95_simultaneous'] < margin),
            'adjusted_low': row['adjusted_low'], 'adjusted_high': row['adjusted_high'],
            'equivalent_two_sided_simultaneous': bool(row['adjusted_low'] > -margin and row['adjusted_high'] < margin),
            'unadjusted_low': row['unadjusted_low'], 'unadjusted_high': row['unadjusted_high'],
            'equivalent_two_sided_pointwise': bool(row['unadjusted_low'] > -margin and row['unadjusted_high'] < margin),
        })
    return out


def modeb_table(study, boot, seeds, mode, scope, budget, cache):
    """Per-interface absolute recovery, H-baseline recovery and additional recovery."""
    endpoints = ['recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P',
                 'recovery/B/SEX', 'recovery/B/RAC1P']
    out = []
    for weight in WEIGHTS:
        # residence utility gain over H
        for condition in S.INTERFACES:
            row = {'mode': mode, 'scope': scope, 'budget': budget, 'weight': weight,
                   'interface': condition}
            r, point, per_seed = contrast(study, boot, seeds, mode, scope, budget,
                                          'H', condition, 'utility/same_residence', weight, cache)
            row['residence_gain_vs_H'] = point
            row['residence_gain_low'] = float(np.quantile(r, .025))
            row['residence_gain_high'] = float(np.quantile(r, .975))
            for e in endpoints:
                # absolute recovery of the interface, averaged over seeds
                abs_vals, h_vals = [], []
                for seed in seeds:
                    target, v = endpoint_vector(study, seed, mode, condition, scope, budget, e, cache)
                    _, hv = endpoint_vector(study, seed, mode, 'H', scope, budget, e, cache)
                    _, pa = boot.ratio(target, weight, v)
                    _, ph = boot.ratio(target, weight, hv)
                    abs_vals.append(pa)
                    h_vals.append(ph)
                name = e.split('/', 1)[1]
                row[f'absolute/{name}'] = float(np.mean(abs_vals))
                row[f'H_baseline/{name}'] = float(np.mean(h_vals))
                row[f'additional/{name}'] = float(np.mean(abs_vals) - np.mean(h_vals))
            out.append(row)
    return out


def selection_recheck(study, seeds, mode, scope, budget):
    """Re-derive every family-relevant selection from the stored validation losses."""
    result = {'checked': 0, 'mismatches': [], 'mode': mode, 'scope': scope, 'budget': budget}
    for seed in seeds:
        for condition in S.INTERFACES:
            rows = study.metrics(seed, condition, mode)
            for endpoint in S.FORBIDDEN:
                view, target = endpoint.split('/')
                stored = S.stored_selection(rows, 'audit', view, target, budget, scope)
                derived = S.reselect(rows, 'audit', view, target, budget, scope)
                result['checked'] += 1
                if derived is None or derived['candidate_id'] != stored['candidate_id']:
                    result['mismatches'].append({
                        'seed': seed, 'interface': condition, 'endpoint': endpoint,
                        'stored': stored['candidate_id'],
                        'derived': None if derived is None else derived['candidate_id'],
                        'stored_validation': stored.get('validation_log_loss'),
                        'derived_validation': None if derived is None else derived.get('validation_log_loss'),
                    })
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, help='locked study directory (read-only)')
    ap.add_argument('--published', required=True, help='published study directory with FAMILIES.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(','))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study = S.Study(args.study, seeds)
    spec = S.read_json(Path(args.published) / 'COMPARISONS.json')

    boot = S.HouseholdBootstrap(study.serialno, study.weights, study.labels)
    cache = {}

    all_rows, crits = [], {}
    for name, fspec in spec['families'].items():
        rows, crit, _, _ = run_family(study, boot, seeds, name, fspec, cache)
        all_rows.extend(rows)
        crits[name] = crit
        print(f'{name}: {len(rows)} endpoints, critical value {crit:.6f}', flush=True)

    write_csv(out / 'INDEPENDENT_FAMILIES.csv', all_rows)
    agreement = compare_to_study(all_rows, Path(args.published) / 'FAMILIES.csv')
    agreement['critical_values'] = crits
    (out / 'FAMILY_AGREEMENT.json').write_text(json.dumps(agreement, indent=2) + '\n')
    print('agreement:', json.dumps(agreement['max_abs_difference']), flush=True)

    write_csv(out / 'EQUIVALENCE_F1.csv', equivalence_table(all_rows))

    table = modeb_table(study, boot, seeds, 'B', 'transport_all', 360, cache)
    table += modeb_table(study, boot, seeds, 'A', 'kernel_expanded_catchup', 360, cache)
    write_csv(out / 'MODEB_TRANSPORT_TABLE.csv', table)

    rechecks = [selection_recheck(study, seeds, 'B', s, 360)
                for s in ('common_fresh', 'transport_all')]
    (out / 'SELECTION_RECHECK.json').write_text(json.dumps(rechecks, indent=2) + '\n')
    print('selection mismatches:', [len(r['mismatches']) for r in rechecks], flush=True)


if __name__ == '__main__':
    main()

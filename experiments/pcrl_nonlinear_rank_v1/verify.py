"""Independent verification of the comparison arithmetic in the published tables.

Recomputes, from the published CSVs alone, the identities the report layer relies
on. This does not re-derive the losses (that is the score replay and the
independent prediction replay); it checks that the arithmetic built on top of
them is what it claims to be.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .inputs import OUT, read_json, write_json


def _csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def run(out: Path = OUT) -> dict:
    out = Path(out)
    per_seed = _csv(out / 'PER_SEED.csv')
    intervals = _csv(out / 'PAIRED_INTERVALS.csv')
    report = read_json(out / 'DEVELOPMENT_2018.json')
    checks = {}

    values = {}
    for r in per_seed:
        if r['value'] == '' or r['value'] is None:
            continue
        values[(int(r['seed']), r['condition'], r['split'], r['weight'], r['budget'],
                r['scope'], r['kind'], r['endpoint'])] = float(r['value'])

    # 1. additional_recovery == absolute_recovery - H's absolute_recovery, exactly.
    worst, n = 0.0, 0
    for key, v in values.items():
        seed, c, split, w, b, scope, kind, endpoint = key
        if kind != 'additional_recovery':
            continue
        own = values.get((seed, c, split, w, b, scope, 'absolute_recovery', endpoint))
        h = values.get((seed, 'H', split, w, b, scope, 'absolute_recovery', endpoint))
        if own is None or h is None:
            continue
        worst = max(worst, abs(v - (own - h)))
        n += 1
    checks['additional_recovery_identity'] = {
        'comparisons': n, 'max_abs_error': worst, 'clean': bool(worst < 1e-12),
        'identity': 'additional = absolute - H absolute, matched on seed/split/weight/budget/scope'}

    # 2. absolute_recovery == prior loss - attack loss, so recovery + loss is constant per target.
    spread, n2 = 0.0, 0
    by_target = defaultdict(list)
    for key, v in values.items():
        seed, c, split, w, b, scope, kind, endpoint = key
        if kind != 'absolute_recovery':
            continue
        loss = values.get((seed, c, split, w, b, scope, 'attack_loss', endpoint))
        if loss is None:
            continue
        by_target[(seed, split, w, b, scope, endpoint)].append(v + loss)
    for group in by_target.values():
        if len(group) > 1:
            spread = max(spread, float(np.max(group) - np.min(group)))
            n2 += len(group)
    checks['prior_cancellation'] = {
        'comparisons': n2, 'max_spread': spread, 'clean': bool(spread < 1e-12),
        'identity': ('recovery + attack loss equals the same prior loss for every condition on a '
                     'given seed/target/weighting, i.e. one prior per mode as specified')}

    # 3. H's own gain over itself is exactly zero on every endpoint.
    worst3 = max((abs(v) for key, v in values.items()
                  if key[1] == 'H' and key[6] in ('additional_recovery', 'utility_gain_vs_H')),
                 default=0.0)
    checks['H_self_reference_zero'] = {'max_abs': worst3, 'clean': bool(worst3 == 0.0)}

    # 4. Interval estimate equals the seed-mean of the per-seed paired differences.
    worst4, n4 = 0.0, 0
    for row in intervals:
        endpoint = row['endpoint']
        kind, rest = endpoint.split('/', 1)
        left, right, w = row['left'], row['right'], row['weight']
        diffs = []
        for seed in (0, 1, 2):
            if kind == 'utility':
                a = values.get((seed, left, 'test', w, '360', report['scope'],
                                'utility_loss', rest))
                b = values.get((seed, right, 'test', w, '360', report['scope'],
                                'utility_loss', rest))
                if a is None or b is None:
                    diffs = None
                    break
                diffs.append(a - b)
            else:
                a = values.get((seed, left, 'test', w, '360', report['scope'],
                                'absolute_recovery', rest))
                b = values.get((seed, right, 'test', w, '360', report['scope'],
                                'absolute_recovery', rest))
                if a is None or b is None:
                    diffs = None
                    break
                # recovery difference: the prior cancels within a paired contrast
                diffs.append(a - b)
        if not diffs:
            continue
        # Both kinds use the convention that lower is better, so the seed mean of the
        # paired differences is the estimate in either case.
        estimate = float(np.mean(diffs))
        worst4 = max(worst4, abs(estimate - float(row['estimate'])))
        n4 += 1
    checks['interval_estimate_is_seed_mean_of_paired_differences'] = {
        'comparisons': n4, 'max_abs_error': worst4, 'clean': bool(worst4 < 1e-9),
        'note': ('recovery differences are reported with the sign convention that lower is better '
                 'for both utility loss and recovery; the prior cancels within a paired contrast')}

    # 5. Adjusted intervals bracket the estimate and are wider than the unadjusted ones.
    bad_bracket = [r['endpoint'] for r in intervals
                   if not (float(r['adjusted_low']) - 1e-12 <= float(r['estimate'])
                           <= float(r['adjusted_high']) + 1e-12)]
    narrower = [r['endpoint'] for r in intervals if r['degenerate'] != 'True'
                and (float(r['adjusted_high']) - float(r['adjusted_low']))
                < (float(r['unadjusted_high']) - float(r['unadjusted_low'])) - 1e-12]
    checks['interval_shape'] = {
        'rows': len(intervals), 'estimate_outside_adjusted_interval': len(bad_bracket),
        'adjusted_narrower_than_unadjusted': len(narrower),
        'clean': bool(not bad_bracket and not narrower),
        'note': 'a single-step max-|t| adjustment can never be narrower than the pointwise interval'}

    # 6. Withholding endpoints: p=0 must reproduce H and p=1 the source, exactly.
    wh = _csv(out / 'WITHHOLDING.csv')
    wv = {(r['source'], r['p'], int(r['seed']), r['weight'], r['kind'], r['endpoint']):
          float(r['value']) for r in wh}
    worst6, n6 = 0.0, 0
    for (src, p, seed, w, kind, endpoint), v in wv.items():
        target_kind = 'utility_loss' if kind == 'utility_loss' else 'absolute_recovery'
        if float(p) == 0.0:
            ref = values.get((seed, 'H', 'test', w, '360', report['scope'], target_kind, endpoint))
        elif float(p) == 1.0:
            ref = values.get((seed, src, 'test', w, '360', report['scope'], target_kind, endpoint))
        else:
            continue
        if ref is None:
            continue
        worst6 = max(worst6, abs(v - ref))
        n6 += 1
    checks['withholding_endpoints'] = {
        'comparisons': n6, 'max_abs_error': worst6, 'clean': bool(worst6 < 1e-12),
        'identity': 'p=0 reproduces H exactly and p=1 reproduces the source exactly'}

    record = {'checks': checks,
              'all_clean': bool(all(c.get('clean', True) for c in checks.values())),
              'scope': ('verifies the arithmetic built on the published losses; the losses '
                        'themselves are covered by the score replay and the independent '
                        'prediction replay')}
    write_json(out / 'COMPARISON_VERIFICATION.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    r = run(a.out)
    for name, c in r['checks'].items():
        print(('PASS' if c.get('clean') else 'FAIL'), name,
              {k: v for k, v in c.items() if k in ('comparisons', 'max_abs_error', 'max_spread',
                                                   'max_abs', 'rows',
                                                   'estimate_outside_adjusted_interval',
                                                   'adjusted_narrower_than_unadjusted')})
    print('ALL CLEAN' if r['all_clean'] else 'NOT ALL CLEAN')


if __name__ == '__main__':
    main()

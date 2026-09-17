"""Phase 4: build the 2018 development tables, intervals and decisions.

Fits nothing and selects nothing. Recomputes every selected loss from the stored
per-person predictions (score replay) so that the reported numbers are verified
against the cached arrays, not merely copied from them.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.run_acs_residual_spectral import load_labels

from .inputs import HIST_INTERFACES, HIST_ROOT, HIST_SPECTRAL, OUT, Registry, read_json, write_json
from .maps import HISTORICAL_ALIAS, condition_name
from .objective import POLICIES
from .report import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, ClusterBootstrap, FAMILY_ENDPOINTS,
                     FORBIDDEN, MAIN_BUDGET, MAIN_SCOPE, MAIN_SPLIT, ROUNDOFF, SENSITIVE,
                     UTILITY_DELTA, WEIGHTS, advantage, condition_dir, coordination, criteria_rows,
                     flat_rows, load_points, noninferiority, simultaneous)
from .run_fit import limit_threads, machine_state

NEW_CONDITIONS = tuple(condition_name(f, r, p) for r in (16, 8)
                       for f in ('original', 'nonlinear') for p in POLICIES)
REFERENCE_CONDITIONS = ('H', 'E', 'A0', 'L025', 'L20', 'J',
                        'spectral_L1', 'spectral_L2', 'spectral_C1', 'spectral_S0')


def predeclared_contrasts() -> list:
    """PROTOCOL.md section 6, in order. Void-by-alias comparisons are recorded, not run."""
    out = []
    for policy in POLICIES:
        out.append((condition_name('nonlinear', 16, policy), condition_name('original', 16, policy),
                    'C1_nonlinear_vs_original_rank16'))
        out.append((condition_name('nonlinear', 8, policy), condition_name('original', 8, policy),
                    'C6_nonlinear_vs_original_rank8'))
        for family in ('original', 'nonlinear'):
            out.append((condition_name(family, 8, policy), condition_name(family, 16, policy),
                        'C3_rank8_vs_rank16'))
    for family, rank in itertools.product(('original', 'nonlinear'), (16, 8)):
        for comparator in ('L1', 'L2'):
            out.append((condition_name(family, rank, 'C1'), condition_name(family, rank, comparator),
                        'C4_C1_vs_local_control'))
    for name in NEW_CONDITIONS:
        for reference in ('J', 'H', 'E', 'A0'):
            out.append((name, reference, 'C5_candidate_vs_reference'))
    seen, unique = set(), []
    for left, right, label in out:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        unique.append({'left': left, 'right': right, 'family': label})
    return unique


def build_bootstrap(seeds, registry: Registry):
    """Household clusters over the cohort; see ClusterBootstrap for why not per seed."""
    frames = {}
    for seed in seeds:
        frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
        frames[seed] = (frame, pools, labels, weights)
    cohort = frames[seeds[0]][0].SERIALNO.to_numpy()
    boot = ClusterBootstrap(cohort, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    targets = sorted({e.split('/')[1] for e in FORBIDDEN} | set(
        t for t in frames[seeds[0]][2]['test']))
    for seed in seeds:
        frame, pools, labels, weights = frames[seed]
        rows = np.asarray(pools['test'])
        for target in targets:
            y = labels['test'][target]
            boot.register(seed, target, rows, y >= 0, weights['test'])
    return boot, frames


def person_loss_vectors(out: Path, seeds, conditions, points, frames, registry: Registry):
    """Per-person test-pool losses of the SELECTED candidate for every endpoint."""
    vectors, replay = {}, []
    for seed in seeds:
        _frame, _pools, labels, _weights = frames[seed]
        for condition in conditions:
            path = condition_dir(out, seed, condition) / 'predictions.npz'
            registry.add(path)
            with np.load(path) as store:
                for weight in WEIGHTS:
                    point = points[seed, condition, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                    for task, row in ((t, point['selected']['utility/' + t]) for t in point['utility']):
                        key = f"utility/{row['view']}/{task}/None/{row['candidate_id']}/test"
                        y = labels['test'][task]
                        losses, _ = _losses(store[key], y)
                        vectors[seed, condition, weight, 'utility/' + task] = (task, losses)
                        replay.append(_replay_row(seed, condition, weight, 'utility/' + task,
                                                  losses, point['utility'][task], _weights_for(
                                                      frames[seed], y, weight)))
                    for endpoint in FORBIDDEN:
                        row = point['selected']['audit/' + endpoint]
                        target = endpoint.split('/')[1]
                        key = (f"audit/{row['view']}/{target}/{row['audit_budget']}/"
                               f"{row['candidate_id']}/test")
                        y = labels['test'][target]
                        losses, _ = _losses(store[key], y)
                        vectors[seed, condition, weight, 'recovery/' + endpoint] = (target, losses)
                        replay.append(_replay_row(seed, condition, weight, 'loss/' + endpoint,
                                                  losses, point['losses'][endpoint],
                                                  _weights_for(frames[seed], y, weight)))
    return vectors, replay


def _losses(predictions, y):
    y = np.asarray(y)
    valid = y >= 0
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2 or len(p) != int(valid.sum()):
        raise ValueError('prediction rows must match valid-label rows')
    if not np.isfinite(p).all():
        raise ValueError('nonfinite predictions')
    yy = y[valid].astype(int)
    clipped = np.clip(p, 1e-12, 1.)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return -np.log(clipped[np.arange(len(yy)), yy]), valid


def _weights_for(frame_bundle, y, weight):
    _frame, _pools, _labels, weights = frame_bundle
    valid = np.asarray(y) >= 0
    w = np.asarray(weights['test'])[valid]
    return w if weight == 'person_weighted' else np.ones_like(w)


def _replay_row(seed, condition, weight, endpoint, losses, stored, w):
    recomputed = float((w * losses).sum() / w.sum())
    return {'seed': seed, 'condition': condition, 'weight': weight, 'endpoint': endpoint,
            'stored': stored, 'recomputed': recomputed,
            'abs_difference': abs(recomputed - stored)}


def contrast_intervals(boot, vectors, seeds, contrasts, families):
    """Paired replicate differences, seed-averaged, with within-family max-|t| adjustment."""
    rows = []
    for spec in contrasts:
        left, right = spec['left'], spec['right']
        for weight in WEIGHTS:
            endpoints, estimates, replicate_columns = [], [], []
            for endpoint in FAMILY_ENDPOINTS:
                key_l = (seeds[0], left, weight, endpoint)
                if key_l not in vectors:
                    continue
                total_rep = np.zeros(boot.replicates)
                total_est = 0.0
                for seed in seeds:
                    target, lv = vectors[seed, left, weight, endpoint]
                    _, rv = vectors[seed, right, weight, endpoint]
                    rep_l, est_l = boot.means(seed, target, lv, weight)
                    rep_r, est_r = boot.means(seed, target, rv, weight)
                    sign = 1.0 if endpoint.startswith('utility/') else -1.0
                    # utility: candidate minus comparator loss (lower better)
                    # recovery: prior cancels, so recovery difference = -(loss difference)
                    total_rep += sign * (rep_l - rep_r) / len(seeds)
                    total_est += sign * (est_l - est_r) / len(seeds)
                endpoints.append(endpoint)
                estimates.append(total_est)
                replicate_columns.append(total_rep)
            if not endpoints:
                continue
            reps = np.column_stack(replicate_columns)
            est = np.asarray(estimates)
            low, high, se, crit, degenerate = simultaneous(reps, est)
            unadjusted_low = np.quantile(reps, .025, axis=0)
            unadjusted_high = np.quantile(reps, .975, axis=0)
            for i, endpoint in enumerate(endpoints):
                rows.append({
                    'left': left, 'right': right, 'comparison_family': spec['family'],
                    'weight': weight, 'endpoint': endpoint, 'estimate': float(est[i]),
                    'bootstrap_se': float(se[i]), 'critical_value': crit,
                    'adjusted_low': float(low[i]), 'adjusted_high': float(high[i]),
                    'unadjusted_low': float(unadjusted_low[i]),
                    'unadjusted_high': float(unadjusted_high[i]),
                    'degenerate': bool(degenerate[i]),
                    'significantly_better': bool(high[i] < 0),
                    'significantly_worse': bool(low[i] > 0)})
    return rows


def csvout(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    tick = time.perf_counter()
    registry = Registry.new()
    conditions = tuple(dict.fromkeys(REFERENCE_CONDITIONS + NEW_CONDITIONS))
    points, raw, _prior = load_points(out, seeds, conditions, registry)
    per_seed = flat_rows(points, seeds, conditions)
    criteria_per_seed, criteria_summary = criteria_rows(out, points, seeds, conditions, registry)

    boot, frames = build_bootstrap(seeds, registry)
    vectors, replay = person_loss_vectors(out, seeds, conditions, points, frames, registry)
    max_replay = max(r['abs_difference'] for r in replay)

    contrasts = predeclared_contrasts()
    contrasts = [c for c in contrasts if c['left'] in conditions and c['right'] in conditions]
    intervals = contrast_intervals(boot, vectors, seeds, contrasts, None)

    decisions = {}
    by_pair = defaultdict(list)
    for row in intervals:
        by_pair[row['left'], row['right'], row['weight']].append(row)
    for (left, right, weight), rows in by_pair.items():
        decisions[f'{left} vs {right} / {weight}'] = {
            **advantage(rows), 'noninferiority': noninferiority(rows)}

    registered = {}
    for family, rank in itertools.product(('original', 'nonlinear'), (16, 8)):
        candidate = condition_name(family, rank, 'C1')
        registered[f'coordination_{family}_rank{rank}_C1_vs_local'] = coordination(
            decisions, candidate, (condition_name(family, rank, 'L1'),
                                   condition_name(family, rank, 'L2')))
        registered[f'coordination_{family}_rank{rank}_C1_vs_J'] = coordination(
            decisions, candidate, ('J',))

    report = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
        'year': 2018, 'evaluation_status': 'DEVELOPMENT EVALUATION (repeatedly used pools)',
        'split': MAIN_SPLIT, 'scope': MAIN_SCOPE, 'budget': MAIN_BUDGET,
        'seeds': list(seeds), 'conditions': list(conditions),
        'bootstrap': {'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                      'clusters': int(boot.n_groups), 'unit': 'cohort SERIALNO household',
                      'adjustment': 'single-step studentized max-|t|, quantile(.95, higher)',
                      'note': ClusterBootstrap.__doc__.strip()},
        'score_replay': {'checked': len(replay), 'max_abs_difference': max_replay,
                         'clean': bool(max_replay < 1e-9)},
        'criteria_summary': criteria_summary,
        'registered_decisions': registered,
        'decisions': decisions,
        'runtime_seconds': time.perf_counter() - tick,
        'machine': machine_state(),
        'inputs': registry.dump(),
    }
    write_json(out / 'DEVELOPMENT_2018.json', report)
    csvout(out / 'PER_SEED.csv', per_seed)
    csvout(out / 'CRITERIA_PER_SEED.csv', criteria_per_seed)
    csvout(out / 'CRITERIA_SUMMARY.csv', criteria_summary)
    csvout(out / 'PAIRED_INTERVALS.csv', intervals)
    csvout(out / 'SCORE_REPLAY.csv', replay)
    print('REPORT done, replay max abs diff %.3e over %d checks' % (max_replay, len(replay)))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()

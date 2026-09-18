"""Stage 8: 2018 development endpoints, uncertainty and decisions.

Fits nothing and selects nothing. The bootstrap, the studentized max-|t| adjustment,
the advantage rule, the point-estimate coordination rule and every endpoint definition
are **imported unchanged**. New here: the contrast slate of `PROTOCOL.md` §6 and the
candidate-wide simultaneous family of `PROTOCOL.md` §5, which is *added to* the reused
within-contrast adjustment rather than replacing it.

All uncertainty is **descriptive development uncertainty conditional on fitted
systems**. It does not undo repeated use of the 2018 pools and it is neither Census
replicate-weight variance nor retraining variability. Every seed and both weightings
are reported, never only pooled means.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from collections import defaultdict

from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT, Registry, write_json
from experiments.pcrl_nonlinear_rank_v1.run_report import (_losses, _replay_row, _weights_for,
                                                           csvout)
from experiments.run_acs_residual_spectral import load_labels

from .report import (ALPHA, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, BETAS, FAMILY_ENDPOINTS,
                     FORBIDDEN, HISTORICAL_ERASURE, HISTORICAL_EXTERNAL, HISTORICAL_OPTNET,
                     HISTORICAL_SPECTRAL, MAIN, MAIN_BUDGET, MAIN_SCOPE, MAIN_SPLIT, NEW_ARMS,
                     NEW_ERASURE, NO_PROTECTION, OUT, POLICIES, REPEAT, SIMPLE, SINGLE, WEIGHTS,
                     WIDTHS, ClusterBootstrap, advantage, available, candidate_wide,
                     condition_file, coordination, flat_rows, load_points, main_arm,
                     noninferiority, simultaneous)
from .run_fit import REPEAT_SEEDS, SINGLE_CELLS, limit_threads, machine_state

ALL_CONDITIONS = SIMPLE + HISTORICAL_SPECTRAL + HISTORICAL_EXTERNAL + NEW_ARMS


def contrasts() -> list:
    """The predeclared comparison list of `PROTOCOL.md` §6. Nothing is added later."""
    out = []
    # 1. coalition conditioning against both local controls, at matched width and beta
    for width in WIDTHS:
        for beta in BETAS:
            c1 = main_arm(width, 'C1', beta)
            for policy in ('L1', 'L2'):
                out.append({'left': c1, 'right': main_arm(width, policy, beta),
                            'family': 'coalition_vs_local'})
    # 2. ensemble against single-attacker, at each ablation cell
    for width, policy, beta in SINGLE_CELLS:
        arm = main_arm(width, policy, beta)
        out.append({'left': arm, 'right': f'{arm}_single', 'family': 'ensemble_vs_single'})
    # 3. every main C1 against J, the historical erasure/OptNet comparators, and the
    #    matched-width NEW erasure control
    for width in WIDTHS:
        for beta in BETAS:
            c1 = main_arm(width, 'C1', beta)
            for right in ('J', 'leace_A0', 'splince_A0', 'optnet16_C1',
                          f'leace_dax{width}_none'):
                out.append({'left': c1, 'right': right, 'family': 'frontier_vs_comparator'})
    # 4. width 8 against width 16 at matched policy and beta
    for policy in POLICIES:
        for beta in BETAS:
            out.append({'left': main_arm(8, policy, beta), 'right': main_arm(16, policy, beta),
                        'family': 'width8_vs_width16'})
    # 5. optimisation stability: each repeat against its anchor
    for width, policy, beta in SINGLE_CELLS:
        arm = main_arm(width, policy, beta)
        for repeat in REPEAT_SEEDS:
            out.append({'left': f'{arm}_r{repeat}', 'right': arm,
                        'family': 'optimizer_stability'})
    # 6. the no-protection continuation and the new erasure controls against their source
    for arm in NO_PROTECTION:
        out.append({'left': arm, 'right': 'A0', 'family': 'no_protection_vs_A0'})
    for arm in NEW_ERASURE:
        width = 8 if 'dax8' in arm else 16
        out.append({'left': arm, 'right': f'dax{width}_none',
                    'family': 'erasure_vs_its_own_source'})
    return out


def build_bootstrap(seeds, registry: Registry):
    frames = {seed: load_labels(HIST_ROOT, seed) for seed in seeds}
    cohort = frames[seeds[0]][0].SERIALNO.to_numpy()
    boot = ClusterBootstrap(cohort, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    targets = sorted({e.split('/')[1] for e in FORBIDDEN} | set(frames[seeds[0]][2]['test']))
    for seed in seeds:
        _frame, pools, labels, weights = frames[seed]
        rows = np.asarray(pools['test'])
        for target in targets:
            boot.register(seed, target, rows, labels['test'][target] >= 0, weights['test'])
    return boot, frames


def person_loss_vectors(out: Path, seeds, conditions, points, frames, registry: Registry):
    """Per-person test-pool losses of the SELECTED candidate, streamed one file at a time."""
    vectors, replay = {}, []
    for seed in seeds:
        _frame, _pools, labels, _weights = frames[seed]
        for condition in conditions:
            path = condition_file(out, seed, condition, 'predictions.npz')
            registry.add(path)
            with np.load(path) as store:
                for weight in WEIGHTS:
                    point = points[seed, condition, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                    for task in point['utility']:
                        row = point['selected']['utility/' + task]
                        key = f"utility/{row['view']}/{task}/None/{row['candidate_id']}/test"
                        y = labels['test'][task]
                        losses, _ = _losses(store[key], y)
                        vectors[seed, condition, weight, 'utility/' + task] = (task, losses)
                        replay.append(_replay_row(seed, condition, weight, 'utility/' + task,
                                                  losses, point['utility'][task],
                                                  _weights_for(frames[seed], y, weight)))
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


def contrast_intervals(boot, vectors, seeds, specs):
    """Paired replicate differences, seed-averaged, with both adjustments computed."""
    rows, replicates = [], {}
    for spec in specs:
        left, right = spec['left'], spec['right']
        for weight in WEIGHTS:
            endpoints, estimates, columns = [], [], []
            for endpoint in FAMILY_ENDPOINTS:
                if (seeds[0], left, weight, endpoint) not in vectors:
                    continue
                if (seeds[0], right, weight, endpoint) not in vectors:
                    continue
                total_rep = np.zeros(boot.replicates)
                total_est = 0.0
                for seed in seeds:
                    target, lv = vectors[seed, left, weight, endpoint]
                    _, rv = vectors[seed, right, weight, endpoint]
                    rep_l, est_l = boot.means(seed, target, lv, weight)
                    rep_r, est_r = boot.means(seed, target, rv, weight)
                    sign = 1.0 if endpoint.startswith('utility/') else -1.0
                    total_rep += sign * (rep_l - rep_r) / len(seeds)
                    total_est += sign * (est_l - est_r) / len(seeds)
                endpoints.append(endpoint)
                estimates.append(total_est)
                columns.append(total_rep)
            if not endpoints:
                continue
            reps = np.column_stack(columns)
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
                replicates[(left, right, weight, endpoint)] = reps[:, i]
    return rows, replicates


def per_seed_signs(boot, vectors, seeds, specs) -> list:
    """Per-seed effect sizes and signs. Three anchors do not establish robustness."""
    rows = []
    for spec in specs:
        left, right = spec['left'], spec['right']
        for weight in WEIGHTS:
            for endpoint in FAMILY_ENDPOINTS:
                if (seeds[0], left, weight, endpoint) not in vectors:
                    continue
                if (seeds[0], right, weight, endpoint) not in vectors:
                    continue
                for seed in seeds:
                    target, lv = vectors[seed, left, weight, endpoint]
                    _, rv = vectors[seed, right, weight, endpoint]
                    _, est_l = boot.means(seed, target, lv, weight)
                    _, est_r = boot.means(seed, target, rv, weight)
                    sign = 1.0 if endpoint.startswith('utility/') else -1.0
                    value = sign * (est_l - est_r)
                    rows.append({'left': left, 'right': right, 'family': spec['family'],
                                 'weight': weight, 'endpoint': endpoint, 'seed': seed,
                                 'estimate': float(value),
                                 'sign': int(np.sign(value))})
    return rows


def registered_expectations(signs, intervals, decisions, seeds) -> dict:
    """Evaluate the five directional expectations of `PROTOCOL.md` §3, mechanically.

    Registered before any fit. Being wrong about one is a reportable outcome, not a
    failure, and the evaluation is generated rather than read off by eye so that it
    cannot drift toward whatever happened.
    """
    by_contrast = defaultdict(list)
    for row in signs:
        by_contrast[row['left'], row['right'], row['endpoint'], row['weight']].append(row)

    # A contrast is DEGENERATE when both arms released the same channel in every seed, so
    # every per-seed difference is exactly zero. At beta 0.1 and 0.3 the preregistered
    # checkpoint rule returns the unmoved channel, which makes C1, L1 and L2 the same
    # object there. Such a cell can neither confirm nor refute a directional expectation,
    # and counting it as a refutation would be an artifact. Degenerate cells are reported
    # separately, never silently dropped.
    degenerate = set()
    for (left, right, _endpoint, weight), rows in by_contrast.items():
        pair = (left, right, weight)
        if pair in degenerate:
            continue
        if all(r['estimate'] == 0.0
               for e in FAMILY_ENDPOINTS
               for r in by_contrast.get((left, right, e, weight), [])):
            degenerate.add(pair)

    def seed_wins(left, right, endpoint, weight, better=True):
        rows = by_contrast.get((left, right, endpoint, weight), [])
        # recovery endpoints: negative estimate means `left` leaks LESS
        return sum(1 for r in rows if (r['estimate'] < 0) == better), len(rows)

    out = {'degenerate_contrasts': sorted(f'{a} vs {b} / {w}' for a, b, w in degenerate),
           'degeneracy_note': ('a contrast whose two arms released the SAME channel in every '
                               'seed. It can neither confirm nor refute a directional '
                               'expectation, so it is reported separately rather than counted '
                               'as either outcome.')}

    # D1: C1 reduces AB/* recovery relative to L1 at matched width and beta, >=2 of 3 seeds
    cells, met, live = [], 0, 0
    for width in WIDTHS:
        for beta in BETAS:
            left, right = main_arm(width, 'C1', beta), main_arm(width, 'L1', beta)
            is_degenerate = (left, right, 'unweighted') in degenerate
            for endpoint in ('recovery/AB/SEX', 'recovery/AB/RAC1P'):
                wins, total = seed_wins(left, right, endpoint, 'unweighted')
                if not total:
                    continue
                ok = wins >= 2
                cells.append({'width': width, 'beta': beta, 'endpoint': endpoint,
                              'seeds_favouring_C1': wins, 'seeds': total, 'met': ok,
                              'degenerate': is_degenerate})
                if not is_degenerate:
                    live += 1
                    met += bool(ok)
    out['D1'] = {'statement': ('C1 reduces AB/* recovery relative to L1 at matched width and '
                               'beta in at least 2 of 3 seeds'),
                 'cells_evaluated': len(cells),
                 'degenerate_cells': sum(1 for c in cells if c['degenerate']),
                 'informative_cells': live, 'informative_cells_met': met, 'detail': cells,
                 'outcome': ('CONFIRMED' if live and met > live / 2
                             else 'REFUTED' if live else 'UNASSESSABLE'),
                 'outcome_note': ('evaluated on INFORMATIVE cells only. At beta 0.1 and 0.3 the '
                                  'checkpoint rule returned the unmoved channel, so C1 and L1 '
                                  'are the same object there and every difference is exactly '
                                  'zero; those cells are counted and reported but cannot bear '
                                  'on a directional expectation.')}

    # D2: at least one C1 arm shows a local race cost relative to L2
    costs = []
    for width in WIDTHS:
        for beta in BETAS:
            rows = by_contrast.get((main_arm(width, 'C1', beta), main_arm(width, 'L2', beta),
                                    'recovery/A/RAC1P', 'unweighted'), [])
            for row in rows:
                if row['estimate'] > 0:
                    costs.append({'width': width, 'beta': beta, 'seed': row['seed'],
                                  'excess_A_RAC1P_recovery': row['estimate']})
    out['D2'] = {'statement': ('at least one C1 arm shows a LOCAL RACE COST relative to L2 on '
                               'recovery/A/RAC1P'),
                 'instances': len(costs), 'detail': costs[:20],
                 'outcome': 'CONFIRMED' if costs else 'REFUTED'}

    # D3: ensemble recovery is NOT lower than single-attacker recovery in most cells
    cells, not_lower, live = [], 0, 0
    for width, policy, beta in SINGLE_CELLS:
        arm = main_arm(width, policy, beta)
        is_degenerate = (arm, f'{arm}_single', 'unweighted') in degenerate
        for endpoint in FAMILY_ENDPOINTS:
            if endpoint.startswith('utility/'):
                continue
            rows = by_contrast.get((arm, f'{arm}_single', endpoint, 'unweighted'), [])
            for row in rows:
                cells.append({'arm': arm, 'endpoint': endpoint, 'seed': row['seed'],
                              'ensemble_minus_single': row['estimate'],
                              'degenerate': is_degenerate})
                if not is_degenerate:
                    live += 1
                    not_lower += bool(row['estimate'] >= 0)
    out['D3'] = {'statement': ('the ensemble arm\'s measured recovery is NOT lower than the '
                               'single-attacker arm\'s in the majority of cells'),
                 'cells': len(cells),
                 'degenerate_cells': sum(1 for c in cells if c['degenerate']),
                 'informative_cells': live, 'informative_cells_not_lower': not_lower,
                 'detail': cells[:24],
                 'outcome': ('CONFIRMED' if live and not_lower > live / 2
                             else 'REFUTED' if live else 'UNASSESSABLE'),
                 'outcome_note': 'evaluated on INFORMATIVE cells only, as for D1'}

    # D5: width 8 shows lower recovery AND lower residence gain than width 16
    both, cells = 0, []
    for policy in POLICIES:
        for beta in BETAS:
            rec = [r['estimate'] for e in FAMILY_ENDPOINTS if e.startswith('recovery/')
                   for r in by_contrast.get((main_arm(8, policy, beta),
                                             main_arm(16, policy, beta), e, 'unweighted'), [])]
            res = [r['estimate'] for r in by_contrast.get(
                (main_arm(8, policy, beta), main_arm(16, policy, beta),
                 'utility/same_residence', 'unweighted'), [])]
            if not rec or not res:
                continue
            lower_recovery = float(np.mean(rec)) < 0
            # utility rows are candidate-minus-comparator LOSS, so positive means width 8
            # gives up more residence capability
            lower_residence = float(np.mean(res)) > 0
            both += bool(lower_recovery and lower_residence)
            cells.append({'policy': policy, 'beta': beta,
                          'mean_recovery_difference': float(np.mean(rec)),
                          'mean_residence_loss_difference': float(np.mean(res)),
                          'lower_recovery': lower_recovery, 'lower_residence': lower_residence})
    out['D5'] = {'statement': ('width 8 shows lower recovery AND lower residence gain than '
                               'width 16 at matched policy and beta'),
                 'cells': len(cells), 'cells_with_both': both, 'detail': cells,
                 'outcome': ('CONFIRMED' if cells and both > len(cells) / 2
                             else 'REFUTED' if cells else 'UNASSESSABLE')}
    return out


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    registry = Registry.new()

    conditions = available(out, ALL_CONDITIONS, seeds)
    missing = [c for c in ALL_CONDITIONS if c not in conditions]
    points, _raw, _prior = load_points(out, seeds, conditions, registry)
    print('POINTS loaded', len(conditions), 'conditions', flush=True)

    specs = [s for s in contrasts() if s['left'] in conditions and s['right'] in conditions]
    dropped = [s for s in contrasts() if s not in specs]
    involved = sorted({s['left'] for s in specs} | {s['right'] for s in specs} | {'H'})

    boot, frames = build_bootstrap(seeds, registry)
    vectors, replay = person_loss_vectors(out, seeds, involved, points, frames, registry)
    print('VECTORS loaded', len(involved), 'conditions', flush=True)

    rows = flat_rows(points, seeds, conditions)
    intervals, replicates = contrast_intervals(boot, vectors, seeds, specs)
    intervals = candidate_wide(intervals, replicates, ALPHA, BOOTSTRAP_REPLICATES)
    signs = per_seed_signs(boot, vectors, seeds, specs)

    decisions, grouped = {}, {}
    for row in intervals:
        grouped.setdefault((row['left'], row['right'], row['weight']), []).append(row)
    for (left, right, weight), group in grouped.items():
        decisions[f'{left} vs {right} / {weight}'] = {
            **advantage(group),
            'candidate_wide_better': [r['endpoint'] for r in group if r['candidate_wide_better']],
            'candidate_wide_worse': [r['endpoint'] for r in group if r['candidate_wide_worse']],
            'noninferiority_and_equivalence': noninferiority(group)}

    registered = {}
    for width in WIDTHS:
        for beta in BETAS:
            c1 = main_arm(width, 'C1', beta)
            if c1 not in conditions:
                continue
            registered[f'coordination_{c1}_vs_local'] = coordination(
                decisions, c1, [main_arm(width, 'L1', beta), main_arm(width, 'L2', beta)])
            registered[f'coordination_{c1}_vs_J'] = coordination(decisions, c1, ['J'])

    max_replay = max((r['abs_difference'] for r in replay), default=0.0)
    record = {
        'seeds': list(seeds), 'conditions': list(conditions),
        'missing_conditions': missing,
        'contrasts_planned': len(contrasts()), 'contrasts_run': len(specs),
        'contrasts_dropped_for_missing_arms': dropped,
        'new_arms_scored_here': [c for c in NEW_ARMS if c in conditions],
        'reused_never_rescored': [c for c in conditions if c not in NEW_ARMS],
        'bootstrap': {'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED,
                      'clusters': int(boot.n_groups), 'unit': 'cohort SERIALNO household',
                      'within_contrast_adjustment':
                          'single-step studentized max-|t| across the five family endpoints',
                      'candidate_wide_adjustment':
                          ('studentized Bonferroni over the WHOLE searched family: every '
                           'contrast x five family endpoints x two weightings'),
                      'note': ClusterBootstrap.__doc__.strip()},
        'score_replay': {'checks': len(replay), 'max_abs_difference': max_replay},
        'registered_decisions': registered,
        'registered_expectations': registered_expectations(signs, intervals, decisions, seeds),
        'decisions': decisions,
        'runtime_seconds': time.perf_counter() - tick, 'machine': machine_state(),
        'inputs': registry.dump(),
        'uncertainty_scope': (
            'DEVELOPMENT uncertainty conditional on fitted systems. The paired household '
            'bootstrap quantifies sampling variability for fixed fitted systems; it cannot undo '
            'repeated use of the 2018 pools, and it is neither Census replicate-weight variance '
            'nor retraining variability. Three anchor seeds do not establish broad '
            'training-population robustness.'),
        'coordination_rule_scope': (
            'The coordination rule is a one-sided POINT-ESTIMATE rule reused for continuity. A '
            'pass is NOT statistical equivalence and NOT noninferiority within .001. '
            'Equivalence and noninferiority are reported separately per contrast; absence of '
            'significance is neither.'),
        'negative_increment_note': (
            'Selection on validation can yield a NEGATIVE evaluation increment over H. Observed '
            'negative increments are reported as measured and are never truncated to zero.'),
    }
    csvout(out / 'PER_SEED.csv', rows)
    csvout(out / 'PAIRED_INTERVALS.csv', intervals)
    csvout(out / 'PER_SEED_SIGNS.csv', signs)
    csvout(out / 'SCORE_REPLAY.csv', replay)
    write_json(out / 'DEVELOPMENT_2018.json', record)
    print('REPORT done', round(record['runtime_seconds'], 1), 's; replay max abs diff',
          f'{max_replay:.2e}', flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()

"""Phase 3: endpoints, uncertainty and decisions. Fits nothing, selects nothing.

Every endpoint definition, selection rule and criterion is imported or restated
from the completed studies so the numbers stay comparable. Rules reused for
comparability are labelled as reused, never re-justified.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acs_coalition_strength_comparisons import AUDIT_ROLES, UTILITY_TASKS, finite
from scripts.report_acs_residual_spectral import person_losses, point_from_records, score_key
from scripts.summarize_acs_pca16_init import utility_criteria

from .inputs import (DEV_NAME, HIST_INTERFACES, HIST_ROOT, HIST_SPECTRAL, OUT, Registry, read_json,
                     resolve)
from .maps import HISTORICAL_ALIAS

WEIGHTS = ('unweighted', 'person_weighted')
MAIN_SCOPE = 'kernel_expanded_catchup'          # the 2018 development primary scope
MAIN_BUDGET = 360
MAIN_SPLIT = 'test'
FORBIDDEN = tuple(v + '/' + t for v, ts in AUDIT_ROLES.items() for t in ts)
SENSITIVE = tuple(v + '/' + t for v in ('A', 'B', 'AB') for t in ('SEX', 'RAC1P'))
FAMILY_SENSITIVE = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
FAMILY_ENDPOINTS = ('utility/same_residence', *('recovery/' + e for e in FAMILY_SENSITIVE))

# Reused thresholds (transport study COMPARISONS.json "rules")
RESIDENCE_REFERENCE = .01
UTILITY_DELTA = .001
ROUNDOFF = 1e-12
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260918

PRIOR_PATH = 'results/redesign_20260908_acs_coalition_v1/seed_{seed}/controls/metrics.json'
PROTECTION_PATH = 'results/redesign_20260908_acs_protection_v1/seed_{seed}/metrics.json'


# ------------------------------------------------------------------ points
def condition_dir(out: Path, seed: int, condition: str) -> Path:
    """New conditions live in this study; historical ones are read from the frozen study.

    A condition in ``HISTORICAL_ALIAS`` has an objective identical to a historical
    arm, so it is never refitted or re-audited: its frozen unit is read directly.
    That keeps the ledger honest (nine reused, not nine new) and guarantees the
    reused column is bit-for-bit the published one.
    """
    alias = HISTORICAL_ALIAS.get(condition)
    if alias is not None:
        return resolve(f'results/{DEV_NAME}/seed_{seed}/{alias}')
    local = Path(out) / f'seed_{seed}' / condition
    if (local / 'metrics.json').exists():
        return local
    return resolve(f'results/{DEV_NAME}/seed_{seed}/{condition}')


def load_points(out: Path, seeds, conditions, registry: Registry):
    points, raw = {}, {}
    for seed in seeds:
        priors_path = registry.resolve(PRIOR_PATH.format(seed=seed))
        prior = {r['target']: r['scores'] for r in read_json(priors_path)['raw_metrics']
                 if r['condition'] == 'prior'}
        for condition in conditions:
            path = condition_dir(out, seed, condition) / 'metrics.json'
            registry.add(path)
            records = read_json(path)['raw_metrics']
            raw[seed, condition] = records
            for split, weight, budget, scope in itertools.product(
                    ('validation', 'test'), WEIGHTS, (120, 360),
                    ('standard_independent', 'expanded_independent', 'expanded_catchup',
                     'kernel_standard_independent', 'kernel_expanded_independent',
                     'kernel_expanded_catchup')):
                p = point_from_records(records, prior, split, weight, budget, scope)
                if set(p['utility']) != set(UTILITY_TASKS) or set(p['gains']) != set(FORBIDDEN):
                    raise ValueError(f'incomplete selected endpoints {seed}/{condition}/{scope}/{budget}')
                points[seed, condition, split, weight, budget, scope] = p
    return points, raw, prior


def flat_rows(points, seeds, conditions):
    """Per-seed table with absolute recovery, additional recovery and gain over H."""
    rows = []
    for (seed, c, split, w, b, scope), point in points.items():
        h = points[seed, 'H', split, w, b, scope]
        common = {'seed': seed, 'condition': c, 'split': split, 'weight': w,
                  'budget': b, 'scope': scope}
        for task, value in point['utility'].items():
            rows.append({**common, 'kind': 'utility_loss', 'endpoint': task, 'value': value})
            rows.append({**common, 'kind': 'utility_gain_vs_H', 'endpoint': task,
                         'value': h['utility'][task] - value})
        for endpoint, value in point['gains'].items():
            rows.append({**common, 'kind': 'absolute_recovery', 'endpoint': endpoint, 'value': value})
            rows.append({**common, 'kind': 'additional_recovery', 'endpoint': endpoint,
                         'value': value - h['gains'][endpoint]})
            rows.append({**common, 'kind': 'attack_loss', 'endpoint': endpoint,
                         'value': point['losses'][endpoint]})
            rows.append({**common, 'kind': 'H_absolute_recovery', 'endpoint': endpoint,
                         'value': h['gains'][endpoint]})
            rows.append({**common, 'kind': 'selected_candidate', 'endpoint': endpoint,
                         'value': None,
                         'candidate_id': point['selected']['audit/' + endpoint]['candidate_id']})
    return rows


# ------------------------------------------------------------------ criteria
def criteria_rows(out: Path, points, seeds, conditions, registry: Registry):
    """0.01 residence reference, original half-headroom and legacy source allowance."""
    per_seed, summary = [], []
    for seed in seeds:
        path = registry.resolve(PROTECTION_PATH.format(seed=seed))
        records = read_json(path)['raw_metrics']
        selected = {(r['release'], r['target']): r for r in records
                    if r['role'] == 'transfer' and r['selected']}
        for weight in WEIGHTS:
            key = score_key(MAIN_SPLIT, weight)
            pca = {t: selected['E_pca', t][key]['log_loss'] for t in UTILITY_TASKS}
            banks = {c: selected[c, 'same_residence'][key]['log_loss']
                     for c in ('B_rich_bank', 'C_tree_bank')}
            h = points[seed, 'H', MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
            for condition in conditions:
                point = points[seed, condition, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                legacy = utility_criteria(point['utility'], pca, banks)
                gain = h['utility']['same_residence'] - point['utility']['same_residence']
                # These two legacy criteria are TRI-STATE: the historical helpers return
                # None when the headroom ratio or a source loss is undefined. Coercing
                # None to False would silently report a failure where the criterion does
                # not apply, so the raw value and the definedness flag are both kept.
                retention = legacy['residential_retention']
                preservation = legacy['source_preservation']
                per_seed.append({
                    'seed': seed, 'condition': condition, 'weight': weight,
                    'residence_loss': point['utility']['same_residence'],
                    'residence_gain_vs_H': gain,
                    'residence_gain_at_least_01': bool(gain >= RESIDENCE_REFERENCE - ROUNDOFF),
                    'half_headroom_pass': retention.get('pass'),
                    'half_headroom_ratio_defined': retention.get('ratio_defined'),
                    'half_headroom_retained_fraction': retention.get('retained_fraction'),
                    'source_allowance_pass': preservation.get('pass'),
                    'pca32_residence': pca['same_residence'],
                    'better_bank_residence': min(banks.values()),
                    'half_headroom_threshold': (pca['same_residence'] + min(banks.values())) / 2,
                })
    by = defaultdict(list)
    for row in per_seed:
        by[row['condition'], row['weight']].append(row)
    for (condition, weight), rows in sorted(by.items()):
        gains = [r['residence_gain_vs_H'] for r in rows]
        summary.append({
            'condition': condition, 'weight': weight,
            'residence_gain_mean': float(np.mean(gains)),
            'residence_gain_min': float(np.min(gains)),
            'residence_gain_max': float(np.max(gains)),
            'residence_01_all_seeds_and_mean': bool(
                np.mean(gains) >= RESIDENCE_REFERENCE - ROUNDOFF
                and np.min(gains) >= RESIDENCE_REFERENCE - ROUNDOFF),
            'half_headroom_pass_seeds': sum(1 for r in rows if r['half_headroom_pass'] is True),
            'half_headroom_fail_seeds': sum(1 for r in rows if r['half_headroom_pass'] is False),
            'half_headroom_undefined_seeds': sum(1 for r in rows if r['half_headroom_pass'] is None),
            'source_allowance_pass_seeds': sum(1 for r in rows if r['source_allowance_pass'] is True),
            'source_allowance_fail_seeds': sum(1 for r in rows if r['source_allowance_pass'] is False),
            'source_allowance_undefined_seeds': sum(1 for r in rows if r['source_allowance_pass'] is None),
            'seeds': len(rows),
        })
    return per_seed, summary


# ------------------------------------------------------------------ bootstrap
class ClusterBootstrap:
    """Paired household-cluster bootstrap over the 2018 development test pools.

    Deviation from the transport study, recorded deliberately: the 2017 study had
    ONE household partition shared by all three seeds, so it could cluster on the
    final partition's SERIALNO groups directly. The 2018 pools are split per seed
    (``split_households(frame, seed)``), so the three seed test pools overlap in
    people. Independent per-seed resampling would understate that dependence.
    Clusters are therefore the households of the underlying cohort, drawn ONCE per
    replicate and applied to every seed, so a drawn household contributes all of
    its rows in whichever seed test pools contain it. Weighted means remain ratio
    estimators recomputed in each replicate.
    """

    def __init__(self, cohort_households, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
        self.groups, self.inverse = np.unique(cohort_households, return_inverse=True)
        self.n_groups = len(self.groups)
        rng = np.random.default_rng(seed)
        # Draw one replicate at a time: materialising all (replicates x n_groups)
        # draws at once would add a transient array as large as `counts` itself,
        # and this machine is running with swap at capacity.
        # float64 so the ratio matmuls need no dtype-promotion temporary; the counts
        # are small non-negative integers and are represented exactly.
        self.counts = np.zeros((replicates, self.n_groups))
        for b in range(replicates):
            draw = rng.integers(0, self.n_groups, size=self.n_groups)
            self.counts[b] = np.bincount(draw, minlength=self.n_groups)
        self.replicates = replicates
        self.targets = {}

    def register(self, seed, target, rows, valid, weights):
        """``rows`` are cohort row positions of this seed's test pool."""
        household = self.inverse[np.asarray(rows)[valid]]
        self.targets[seed, target] = (household, np.asarray(weights)[valid])

    def means(self, seed, target, vector, weight):
        household, pweights = self.targets[seed, target]
        w = pweights if weight == 'person_weighted' else np.ones_like(pweights)
        numerator = np.bincount(household, weights=w * np.asarray(vector, float),
                                minlength=self.n_groups)
        denominator = np.bincount(household, weights=w, minlength=self.n_groups)
        reps = (self.counts @ numerator) / (self.counts @ denominator)
        return reps, float(numerator.sum() / denominator.sum())


def simultaneous(replicates: np.ndarray, estimate: np.ndarray, level=.95):
    """Single-step studentized max-|t|, exactly the transport study's procedure."""
    se = np.std(replicates, axis=0, ddof=1)
    live = se > 1e-15
    if not live.any():
        return estimate.copy(), estimate.copy(), se, float('nan'), ~live
    t = np.max(np.abs(replicates[:, live] - estimate[live]) / se[live], axis=1)
    crit = float(np.quantile(t, level, method='higher'))
    return (np.where(live, estimate - crit * se, estimate),
            np.where(live, estimate + crit * se, estimate), se, crit, ~live)


def advantage(rows):
    """At least one family sensitive endpoint significantly better, none significantly worse."""
    sensitive = [r for r in rows if r['endpoint'].startswith('recovery/')]
    better = [r['endpoint'] for r in sensitive if r['adjusted_high'] < 0]
    worse = [r['endpoint'] for r in sensitive if r['adjusted_low'] > 0]
    residence = next(r for r in rows if r['endpoint'] == 'utility/same_residence')
    return {'advantage': bool(better) and not worse,
            'better_endpoints': better, 'worse_endpoints': worse,
            'residence_difference': residence['estimate'],
            'residence_adjusted': [residence['adjusted_low'], residence['adjusted_high']],
            'local_race_significantly_worse': 'recovery/A/RAC1P' in worse}


def coordination(decisions, left, rights):
    """Primary rule: advantage over every comparator under both weightings, residence within .001,
    and A/RAC1P never significantly worse. A pass is NOT an equivalence result."""
    keys = [f'{left} vs {r} / {w}' for r in rights for w in WEIGHTS]
    missing = [k for k in keys if k not in decisions]
    if missing:
        return {'supported': False, 'missing': missing}
    ok = all(decisions[k]['advantage'] for k in keys)
    residence = all(decisions[k]['residence_difference'] <= UTILITY_DELTA + ROUNDOFF for k in keys)
    race = any(decisions[k]['local_race_significantly_worse'] for k in keys)
    return {'supported': bool(ok and residence and not race), 'advantage_all': bool(ok),
            'residence_within_001_all': bool(residence),
            'local_race_significantly_worse_any': bool(race),
            'comparators': list(rights), 'candidate': left,
            'note': 'point-difference residence rule reused for comparability; not equivalence'}


def noninferiority(rows, margin=UTILITY_DELTA):
    """Reported SEPARATELY: a one-sided interval-inside-margin condition, not the point rule."""
    out = {}
    for r in rows:
        out[r['endpoint']] = {
            'estimate': r['estimate'], 'adjusted_high': r['adjusted_high'],
            'margin': margin,
            'noninferior_upper_bound_inside_margin': bool(r['adjusted_high'] <= margin),
            'equivalence_interval_inside_margin': bool(
                r['adjusted_low'] >= -margin and r['adjusted_high'] <= margin),
        }
    return out

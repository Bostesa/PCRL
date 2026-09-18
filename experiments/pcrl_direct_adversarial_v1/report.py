"""Stage 8a: the comparison slate and path resolution. Fits nothing, selects nothing.

Every endpoint definition, selection rule, criterion, bootstrap and within-contrast
adjustment is **imported unchanged** from the completed studies so the numbers stay
comparable. Only the slate, the path resolution and the candidate-wide adjustment are
new, and the candidate-wide adjustment is *added to*, never substituted for, the
reused one.

Four sources of scored arms:

* this study's 126 new wires, under `results/pcrl_direct_adversarial_v1/`;
* the invariant study's `leace_A0`, `splince_A0` and `optnet16_*`;
* the historical `H`, `E`, `A0`, `L025`, `L20`, `J` and `spectral_*` arms;
* nothing is written to any completed study and no already-scored arm is rescored.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_acs_residual_spectral import point_from_records
from scripts.acs_coalition_strength_comparisons import UTILITY_TASKS

from experiments.pcrl_nonlinear_rank_v1.report import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
                                                       FAMILY_ENDPOINTS, FAMILY_SENSITIVE,
                                                       FORBIDDEN, MAIN_BUDGET, MAIN_SCOPE,
                                                       MAIN_SPLIT, PRIOR_PATH, RESIDENCE_REFERENCE,
                                                       ROUNDOFF, SENSITIVE, UTILITY_DELTA, WEIGHTS,
                                                       ClusterBootstrap, advantage, coordination,
                                                       flat_rows, noninferiority, simultaneous)
from experiments.pcrl_nonlinear_rank_v1.inputs import DEV_NAME, Registry, read_json, resolve

from .inputs import OUT
from .run_fit import WIDTHS, arm_plan, beta_tag, main_arm
from .train import BETAS, POLICIES

INVARIANT_NAME = 'pcrl_invariant_baselines_v1'

SIMPLE = ('H', 'E', 'A0', 'L025', 'L20', 'J')
HISTORICAL_SPECTRAL = tuple('spectral_' + a for a in
                            ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2'))
HISTORICAL_ERASURE = ('leace_A0', 'splince_A0')
HISTORICAL_OPTNET = tuple(f'optnet16_{p}' for p in POLICIES)
HISTORICAL_EXTERNAL = HISTORICAL_ERASURE + HISTORICAL_OPTNET

_PLAN = arm_plan()
MAIN = tuple(a for a, _ in _PLAN['main'])
NO_PROTECTION = tuple(a for a, _ in _PLAN['no_protection'])
SINGLE = tuple(a for a, _ in _PLAN['single_attacker'])
REPEAT = tuple(a for a, _ in _PLAN['optimizer_repeat'])
NEW_ERASURE = tuple(f'{m}_dax{w}_none' for w in WIDTHS for m in ('leace', 'splince'))
NEW_ARMS = MAIN + NO_PROTECTION + SINGLE + REPEAT + NEW_ERASURE

# The reduced 2018 scope grid: test split only, both weightings, both budgets, all six
# attack scopes (independent versus expanded pooled, with and without the kernel family).
SCOPES = ('standard_independent', 'expanded_independent', 'expanded_catchup',
          'kernel_standard_independent', 'kernel_expanded_independent',
          'kernel_expanded_catchup')
BUDGETS = (120, 360)

ALPHA = 0.05


def available(out: Path, conditions, seeds) -> tuple:
    """Only conditions whose metrics exist for EVERY seed enter any table."""
    keep = []
    for condition in conditions:
        try:
            ok = all(condition_file(out, seed, condition, 'metrics.json').exists()
                     for seed in seeds)
        except FileNotFoundError:
            ok = False
        if ok:
            keep.append(condition)
    return tuple(keep)


def condition_file(out: Path, seed: int, condition: str, name: str) -> Path:
    """Resolve one artifact of a condition, by FILE and not by directory."""
    local = Path(out) / f'seed_{seed}' / condition / name
    if local.exists():
        return local
    if condition in HISTORICAL_EXTERNAL:
        return resolve(f'results/{INVARIANT_NAME}/seed_{seed}/{condition}/{name}')
    return resolve(f'results/{DEV_NAME}/seed_{seed}/{condition}/{name}')


def load_points(out: Path, seeds, conditions, registry: Registry):
    points, raw = {}, {}
    prior = None
    for seed in seeds:
        priors_path = registry.resolve(PRIOR_PATH.format(seed=seed))
        prior = {r['target']: r['scores'] for r in read_json(priors_path)['raw_metrics']
                 if r['condition'] == 'prior'}
        for condition in conditions:
            path = condition_file(out, seed, condition, 'metrics.json')
            registry.add(path)
            records = read_json(path)['raw_metrics']
            raw[seed, condition] = records
            for weight, budget, scope in itertools.product(WEIGHTS, BUDGETS, SCOPES):
                p = point_from_records(records, prior, MAIN_SPLIT, weight, budget, scope)
                if set(p['utility']) != set(UTILITY_TASKS) or set(p['gains']) != set(FORBIDDEN):
                    raise ValueError(
                        f'incomplete selected endpoints {seed}/{condition}/{scope}/{budget}')
                points[seed, condition, MAIN_SPLIT, weight, budget, scope] = p
    return points, raw, prior


def candidate_wide(rows: list, replicates: dict, alpha: float = ALPHA,
                   replicate_count: int = BOOTSTRAP_REPLICATES) -> list:
    """The candidate-wide simultaneous family declared in `PROTOCOL.md` §5.

    The family is EVERY contrast searched for a winning claim, times the five family
    endpoints, times both weightings -- not the five endpoints of the eventual winner.

    `PROTOCOL.md` §5 registered "per-comparison two-sided bootstrap quantiles at level
    alpha/m". With the realised family size that level is far below `1 / replicates`, so
    the registered percentile is **not estimable** from the resample distribution: the
    required tail quantile sits beyond the most extreme replicate. The conservative
    correction the protocol also authorises is therefore used in its place -- a
    studentized Bonferroni bound, `estimate +/- z_{1 - alpha/(2m)} * bootstrap_SE` --
    and the registered percentile is reported alongside with an explicit
    `percentile_estimable` flag so the substitution is visible rather than silent
    (`RUN_STATUS.md` amendment 3).

    The reused within-contrast max-|t| bounds stay in the same table under their own
    names, so the predecessor's columns remain directly readable.
    """
    from scipy.stats import norm

    m = len(rows)
    if not m:
        return rows
    level = alpha / m
    critical = float(norm.ppf(1.0 - level / 2.0))
    estimable = bool(level / 2.0 >= 1.0 / replicate_count)
    for row in rows:
        reps = replicates[(row['left'], row['right'], row['weight'], row['endpoint'])]
        se = float(np.std(reps, ddof=1))
        estimate = row['estimate']
        row['family_size'] = m
        row['candidate_wide_alpha'] = alpha
        row['candidate_wide_level'] = level
        row['candidate_wide_critical_value'] = critical
        row['candidate_wide_low'] = estimate - critical * se
        row['candidate_wide_high'] = estimate + critical * se
        row['candidate_wide_better'] = bool(estimate + critical * se < 0)
        row['candidate_wide_worse'] = bool(estimate - critical * se > 0)
        row['percentile_estimable'] = estimable
        if estimable:
            row['candidate_wide_percentile_low'] = float(np.quantile(reps, level / 2.0))
            row['candidate_wide_percentile_high'] = float(np.quantile(reps, 1.0 - level / 2.0))
    return rows


__all__ = ['ALPHA', 'BOOTSTRAP_REPLICATES', 'BOOTSTRAP_SEED', 'BUDGETS', 'ClusterBootstrap',
           'FAMILY_ENDPOINTS', 'FAMILY_SENSITIVE', 'FORBIDDEN', 'HISTORICAL_ERASURE',
           'HISTORICAL_EXTERNAL', 'HISTORICAL_OPTNET', 'HISTORICAL_SPECTRAL', 'MAIN',
           'MAIN_BUDGET', 'MAIN_SCOPE', 'MAIN_SPLIT', 'NEW_ARMS', 'NEW_ERASURE',
           'NO_PROTECTION', 'OUT', 'REPEAT', 'RESIDENCE_REFERENCE', 'ROUNDOFF', 'SCOPES',
           'SENSITIVE', 'SIMPLE', 'SINGLE', 'UTILITY_DELTA', 'UTILITY_TASKS', 'WEIGHTS',
           'advantage', 'available', 'candidate_wide', 'condition_file', 'coordination',
           'flat_rows', 'load_points', 'noninferiority', 'simultaneous', 'main_arm', 'beta_tag',
           'BETAS', 'POLICIES', 'WIDTHS']

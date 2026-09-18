"""Stage 8: endpoints, uncertainty and decisions. Fits nothing, selects nothing.

Every endpoint definition, selection rule, criterion, bootstrap and adjustment is
**imported** from the completed study so the numbers stay comparable. Rules reused for
comparability are labelled as reused, never re-justified. Only the path resolution and
the comparison slate are new, because this study has three sources of scored arms:

* its own new wires, under ``results/pcrl_invariant_baselines_v1/``;
* the predecessor's ``spectral_lin8_*`` and ``spectral_nlr{8,16}_*`` controls;
* the historical ``H``, ``E``, ``A0``, ``L025``, ``L20``, ``J`` and ``spectral_*`` arms.

Nothing in the completed studies is written to, and no already-scored arm is rescored.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_acs_residual_spectral import point_from_records

# Endpoint definitions, criteria, bootstrap and decision rules: all REUSED unchanged.
from experiments.pcrl_nonlinear_rank_v1.report import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
                                                       FAMILY_ENDPOINTS, FAMILY_SENSITIVE,
                                                       FORBIDDEN, MAIN_BUDGET, MAIN_SCOPE,
                                                       MAIN_SPLIT, PRIOR_PATH, RESIDENCE_REFERENCE,
                                                       ROUNDOFF, SENSITIVE, UTILITY_DELTA, WEIGHTS,
                                                       ClusterBootstrap, advantage, coordination,
                                                       flat_rows, noninferiority, simultaneous)
from experiments.pcrl_nonlinear_rank_v1.inputs import DEV_NAME, Registry, read_json, resolve
from scripts.acs_coalition_strength_comparisons import UTILITY_TASKS

from .run_fit import OUT, PREDECESSOR_ROOTS

# ------------------------------------------------------------------ the comparison slate
SIMPLE = ('H', 'E', 'A0', 'L025', 'L20', 'J')
HISTORICAL_SPECTRAL = tuple('spectral_' + a for a in
                            ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2'))
PREDECESSOR = (tuple(f'spectral_lin8_{p}' for p in ('L1', 'L2', 'C1'))
               + tuple(f'spectral_nlr{r}_{p}' for r in (16, 8) for p in ('L1', 'L2', 'C1')))
REPAIRED = tuple(f'spectral_riv{r}_{p}' for r in (16, 8) for p in ('L1', 'L2', 'C1'))
ERASURE = ('leace_A0', 'splince_A0')
OPTNET = tuple(f'optnet16_{p}' for p in ('L1', 'L2', 'C1'))
NEW_ARMS = REPAIRED + ERASURE + OPTNET

# The rank-16 original-moment arms ARE the historical spectral arms, verified bitwise at
# fit time. They are read from the historical unit, never refitted or rescored.
RANK16_ORIGINAL_ALIAS = {f'spectral_lin16_{p}': f'spectral_{p}' for p in ('L1', 'L2', 'C1')}

ALL_CONDITIONS = SIMPLE + HISTORICAL_SPECTRAL + PREDECESSOR + NEW_ARMS


def condition_file(out: Path, seed: int, condition: str, name: str) -> Path:
    """Resolve one artifact of a condition, by FILE and not by directory.

    Resolving on a directory is wrong here: the tracked result directories of the
    completed studies exist in this fresh worktree but their ignored contents
    (``metrics.json``, ``predictions.npz``) do not, so a directory-level match points at
    an empty folder. Every lookup therefore resolves the file itself.
    """
    alias = RANK16_ORIGINAL_ALIAS.get(condition)
    if alias is not None:
        return resolve(f'results/{DEV_NAME}/seed_{seed}/{alias}/{name}')
    local = Path(out) / f'seed_{seed}' / condition / name
    if local.exists():
        return local
    if condition in PREDECESSOR:
        for root in PREDECESSOR_ROOTS:
            candidate = root / f'results/pcrl_nonlinear_rank_v1/seed_{seed}/{condition}/{name}'
            if candidate.exists():
                return candidate
    return resolve(f'results/{DEV_NAME}/seed_{seed}/{condition}/{name}')


def load_points(out: Path, seeds, conditions, registry: Registry):
    """Selected endpoints for every condition, split, weighting, budget and scope."""
    points, raw = {}, {}
    for seed in seeds:
        priors_path = registry.resolve(PRIOR_PATH.format(seed=seed))
        prior = {r['target']: r['scores'] for r in read_json(priors_path)['raw_metrics']
                 if r['condition'] == 'prior'}
        for condition in conditions:
            path = condition_file(out, seed, condition, 'metrics.json')
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
                    raise ValueError(
                        f'incomplete selected endpoints {seed}/{condition}/{scope}/{budget}')
                points[seed, condition, split, weight, budget, scope] = p
    return points, raw, prior


__all__ = ['ALL_CONDITIONS', 'BOOTSTRAP_REPLICATES', 'BOOTSTRAP_SEED', 'ClusterBootstrap',
           'ERASURE', 'FAMILY_ENDPOINTS', 'FAMILY_SENSITIVE', 'FORBIDDEN',
           'HISTORICAL_SPECTRAL', 'MAIN_BUDGET', 'MAIN_SCOPE', 'MAIN_SPLIT', 'NEW_ARMS',
           'OPTNET', 'OUT', 'PREDECESSOR', 'RANK16_ORIGINAL_ALIAS', 'REPAIRED',
           'RESIDENCE_REFERENCE', 'ROUNDOFF', 'SENSITIVE', 'SIMPLE', 'UTILITY_DELTA',
           'UTILITY_TASKS', 'WEIGHTS', 'advantage', 'condition_file', 'coordination',
           'flat_rows', 'load_points', 'noninferiority', 'simultaneous']

"""Read audited endpoints for this study's units and for historical comparators.

A unit's metrics are resolved through the exact-identity table: a duplicate release
reads its canonical unit's audit. Historical comparators (`H`, `A0`, `J`, `leace_A0`,
`splince_A0`, `optnet16_*`) are read through the predecessor's own resolver, in the
matched-exposure scope. Points are built with the predecessor's `point_from_records`
for any split, so validation-only panel selection and test-split reporting use one
code path.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1 import report as rep

from .common import ADVERSARIAL_RESULTS, OUT, read_json

HISTORICAL = ('H', 'E', 'A0', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1', 'optnet16_L1',
              'optnet16_L2')
SENSITIVE4 = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
MAIN_BUDGET = 360
MAIN_SCOPE = 'kernel_expanded_independent'


def canonical(seed: int, unit: str) -> str:
    path = OUT / f'seed_{seed}' / 'audit_identity.json'
    if path.exists():
        entry = read_json(path).get(unit)
        if entry:
            return entry['audited_as']
    return unit


def metrics_path(seed: int, unit: str) -> Path:
    if unit in HISTORICAL:
        return rep.condition_file(ADVERSARIAL_RESULTS, seed, unit, 'metrics.json')
    return OUT / f'seed_{seed}' / canonical(seed, unit) / 'metrics.json'


def audited(seed: int, unit: str) -> bool:
    if unit in HISTORICAL:
        return True
    return (OUT / f'seed_{seed}' / canonical(seed, unit) / 'complete.json').exists()


@lru_cache(maxsize=None)
def prior(seed: int) -> dict:
    registry = dax.Registry.new()
    return {r['target']: r['scores'] for r in
            read_json(registry.resolve(rep.PRIOR_PATH.format(seed=seed)))['raw_metrics']
            if r['condition'] == 'prior'}


@lru_cache(maxsize=4096)
def _records(seed: int, unit: str):
    return tuple(read_json(metrics_path(seed, unit))['raw_metrics'])


def point(seed: int, unit: str, split: str, weight: str, budget: int = MAIN_BUDGET,
          scope: str = MAIN_SCOPE) -> dict:
    return rep.point_from_records(list(_records(seed, unit)), prior(seed), split, weight,
                                  budget, scope)


@lru_cache(maxsize=None)
def epca_source(seed: int, split: str, weight: str) -> dict:
    """The frozen registry's `E_pca` parent source losses (historical allowance reference)."""
    from experiments.pcrl_nonlinear_rank_v1.report import PROTECTION_PATH
    from scripts.report_acs_residual_spectral import score_key
    registry = dax.Registry.new()
    records = read_json(registry.resolve(PROTECTION_PATH.format(seed=seed)))['raw_metrics']
    selected = {(r['release'], r['target']): r for r in records
                if r['role'] == 'transfer' and r['selected']}
    key = score_key(split, weight)
    return {t: selected['E_pca', t][key]['log_loss']
            for t in ('income_binary', 'civilian_at_work', 'public_coverage')}


@lru_cache(maxsize=None)
def training_entropy(seed: int) -> dict:
    registry = dax.Registry.new()
    training = read_json(registry.resolve(dax.fixed_path(seed, 'training/training.json')))
    return {name: float(training['priors'][name]['entropy']) for name in ('SEX', 'RAC1P')}


def increments(seed: int, unit: str, split: str, weight: str) -> dict:
    """Gain over the H view for every forbidden endpoint (negative values are kept)."""
    p = point(seed, unit, split, weight)
    h = point(seed, 'H', split, weight)
    return {e: p['gains'][e] - h['gains'][e] for e in p['gains']}

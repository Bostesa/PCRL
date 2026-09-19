"""Incident repair (RUN_STATUS incident 1): regenerate 2018 attacker weights for 2017.

The 2017 Mode B stage also scores the 2018-fitted attackers on 2017, so it needs their
`.pt` weights, which audit compaction had deleted. The audit is seeded and
deterministic: each needed canonical unit is re-audited into a fresh directory WITHOUT
compaction, and its `metrics.json` raw records must equal the compacted run's exactly,
and every kept prediction array must be bitwise equal. Only then is the regenerated
directory used; the compacted one is kept under `_compacted/`.
"""
from __future__ import annotations

import shutil
import sys

import numpy as np

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1 import run_dev_2018 as dev

from .common import OUT, limit_threads, read_json, utcnow, write_json_atomic


def regen(seed: int, unit: str) -> dict:
    live = OUT / f'seed_{seed}' / unit
    archive = OUT / f'seed_{seed}' / '_compacted' / unit
    if (live / 'regenerated.json').exists():
        return read_json(live / 'regenerated.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.move(str(live), str(archive))
    dev.evaluate_seed(OUT, seed, [unit], dax.Registry.new())
    old = read_json(archive / 'metrics.json')['raw_metrics']
    new = read_json(live / 'metrics.json')['raw_metrics']
    same_metrics = old == new
    with np.load(archive / 'predictions.npz') as a, np.load(live / 'predictions.npz') as b:
        same_predictions = all(np.array_equal(a[k], b[k]) for k in a.files)
        kept = len(a.files)
    if not (same_metrics and same_predictions):
        raise AssertionError(f'regenerated audit differs for {seed}/{unit}')
    record = {'seed': seed, 'unit': unit, 'utc': utcnow(), 'metrics_identical': same_metrics,
              'compacted_predictions_bitwise_identical': same_predictions,
              'compared_prediction_arrays': kept}
    write_json_atomic(live / 'regenerated.json', record)
    return record


def main(units):
    limit_threads()
    out = []
    for seed in (0, 1, 2):
        for unit in units:
            out.append(regen(seed, unit))
            print('REGEN', seed, unit, out[-1]['metrics_identical'], flush=True)
    write_json_atomic(OUT / 'REGEN_FOR_2017.json', {'units': out, 'utc': utcnow()})


if __name__ == '__main__':
    main(sys.argv[1:])

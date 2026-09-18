"""Stage 7b: exploratory cross-year report. Fits nothing, selects nothing.

**Every number here is EXPLORATORY DEVELOPMENT.** The 2017 locked evaluation is
finished and its final partition is spent. The original frozen transport result keeps
its historical status and is neither restated nor overwritten. 2017 is **not** a new
test merely because these methods had not been scored there. There are no intervals:
the transport study's uncertainty machinery is not re-applied to a spent partition.

Selection logic, priors and scope definitions are **imported unchanged** from the
predecessor's cross-year report so the two studies' 2017 tables are comparable.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.pcrl_nonlinear_rank_v1.inputs import Registry, read_json, resolve, write_json
from experiments.pcrl_nonlinear_rank_v1.report_2017 import (BUDGET, SCOPES, TRANSPORT_NAME,
                                                            point, priors)
from experiments.pcrl_nonlinear_rank_v1.report import FORBIDDEN, WEIGHTS
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads

from .report import REPAIRED
from .run_fit import OUT

# Frozen comparators that already have 2017 Mode B units in the transport study.
REFERENCES = ('H', 'J', 'E', 'A0', 'spectral_S0', 'spectral_C1', 'spectral_L1', 'spectral_L2')
FAMILY = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
MAIN_SCOPE = 'common_fresh'


def metrics_path(out: Path, seed: int, condition: str) -> Path:
    """This study's exploratory unit, or the transport study's frozen Mode B unit."""
    local = Path(out) / 'exploratory_2017' / f'seed_{seed}' / condition / 'mode_B' / 'metrics.json'
    if local.exists():
        return local
    return resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/{condition}/mode_B/metrics.json')


def build(out: Path, seeds):
    registry = Registry.new()
    conditions = tuple(dict.fromkeys(REFERENCES + tuple(REPAIRED)))
    points, missing = {}, []
    for seed in seeds:
        prior = priors(seed, registry)
        for condition in conditions:
            try:
                path = metrics_path(out, seed, condition)
            except FileNotFoundError:
                missing.append((seed, condition))
                continue
            if not Path(path).exists():
                missing.append((seed, condition))
                continue
            registry.add(path)
            records = read_json(path)['raw_metrics']
            for weight, scope in itertools.product(WEIGHTS, SCOPES):
                try:
                    p = point(records, prior, weight, scope)
                except ValueError:
                    continue
                if set(p['gains']) == set(FORBIDDEN) and 'same_residence' in p['utility']:
                    points[seed, condition, weight, scope] = p
    return points, registry, missing, conditions


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    points, registry, missing, conditions = build(out, seeds)

    rows = []
    for (seed, condition, weight, scope), p in points.items():
        h = points.get((seed, 'H', weight, scope))
        if h is None:
            continue
        for endpoint, value in p['gains'].items():
            rows.append({'year': 2017, 'seed': seed, 'condition': condition, 'weight': weight,
                         'scope': scope, 'kind': 'additional_recovery', 'endpoint': endpoint,
                         'value': value - h['gains'][endpoint]})
        rows.append({'year': 2017, 'seed': seed, 'condition': condition, 'weight': weight,
                     'scope': scope, 'kind': 'residence_gain_vs_H',
                     'endpoint': 'utility/same_residence',
                     'value': h['utility']['same_residence'] - p['utility']['same_residence']})

    # seed means at the main scope
    means = {}
    for condition in conditions:
        for weight in WEIGHTS:
            for endpoint in FAMILY:
                vals = [r['value'] for r in rows if r['condition'] == condition
                        and r['weight'] == weight and r['scope'] == MAIN_SCOPE
                        and r['endpoint'] == endpoint and r['kind'] == 'additional_recovery']
                if vals:
                    means[condition, weight, 'recovery/' + endpoint] = float(np.mean(vals))
            vals = [r['value'] for r in rows if r['condition'] == condition
                    and r['weight'] == weight and r['scope'] == MAIN_SCOPE
                    and r['kind'] == 'residence_gain_vs_H']
            if vals:
                means[condition, weight, 'residence_gain_vs_H'] = float(np.mean(vals))

    record = {
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT (2017 seal already spent)',
        'scope': MAIN_SCOPE, 'budget': BUDGET, 'seeds': list(seeds),
        'conditions_present': [c for c in conditions
                               if any(k[1] == c for k in points)],
        'missing': [list(m) for m in missing],
        'not_transported': {
            'leace_A0': 'acts on the frozen A0 neural auxiliary channel; its 2017 construction '
                        'belongs to a different pipeline',
            'splince_A0': 'same as leace_A0',
            'optnet16_*': 'encoders were not persisted by this run\'s fit stage; persistence was '
                          'added afterwards and the stage is deterministic, so the resume command '
                          'reproduces them',
        },
        'seed_means': {f'{c}|{w}|{e}': v for (c, w, e), v in sorted(means.items())},
        'interval_scope': 'NONE. No intervals are computed on a spent partition.',
        'reading_rule': ('additional recovery is relative to H on the same year; a LOWER value '
                         'means less disclosure. Years are reported separately and rows are '
                         'never pooled across years.'),
        'inputs': registry.dump(),
    }
    with open(out / 'EXPLORATORY_2017.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'EXPLORATORY_2017.json', record)
    print('2017 report:', len(rows), 'rows,',
          len(record['conditions_present']), 'conditions present', flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()

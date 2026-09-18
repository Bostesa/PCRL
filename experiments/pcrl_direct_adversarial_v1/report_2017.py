"""Stage 8b: the exploratory cross-year report. Fits nothing, selects nothing.

**Every number here is EXPLORATORY DEVELOPMENT.** The 2017 locked evaluation is
finished and its final partition is spent. The original frozen transport result keeps
its historical status and is neither restated nor overwritten. 2017 is **not** a new
test merely because these methods had not been scored there.

Selection logic, prior construction and scope definitions are imported unchanged from
the predecessors' cross-year reports so all three studies' 2017 tables are comparable.
Panel membership was fixed in `PROTOCOL.md` §10 **before** any 2017 number was read.

Intervals, where reported, are **conditional descriptive sampling uncertainty**. They
do not restore independence after repeated development. They are not forbidden on a
used dataset either; they simply mean less than they would on a fresh one.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.pcrl_nonlinear_rank_v1.inputs import Registry, read_json, resolve, write_json
from experiments.pcrl_nonlinear_rank_v1.report_2017 import BUDGET, SCOPES, TRANSPORT_NAME, point, priors
from experiments.pcrl_nonlinear_rank_v1.report import FORBIDDEN, WEIGHTS
from experiments.pcrl_nonlinear_rank_v1.run_report import csvout

from .freeze import panel_2017
from .inputs import OUT
from .run_fit import limit_threads

REFERENCES = ('H', 'J', 'E', 'A0', 'spectral_C1', 'spectral_L2')
FAMILY = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
MAIN_SCOPE = 'common_fresh'


def metrics_path(out: Path, seed: int, condition: str) -> Path:
    """This study's exploratory unit, or the transport study's frozen Mode B unit."""
    local = Path(out) / 'exploratory_2017' / f'seed_{seed}' / condition / 'mode_B' / 'metrics.json'
    if local.exists():
        return local
    return resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/{condition}/mode_B/metrics.json')


def panel_conditions() -> tuple:
    panel = panel_2017()
    ordered = (tuple(REFERENCES) + tuple(panel['main']) + tuple(panel['no_protection'])
               + tuple(panel['new_erasure']) + tuple(panel['historical_baselines']))
    return tuple(dict.fromkeys(ordered))


def build(out: Path, seeds):
    registry = Registry.new()
    conditions = panel_conditions()
    points, missing = {}, []
    for seed in seeds:
        prior = priors(seed, registry)
        for condition in conditions:
            try:
                path = metrics_path(out, seed, condition)
            except FileNotFoundError:
                missing.append({'seed': seed, 'condition': condition})
                continue
            if not Path(path).exists():
                missing.append({'seed': seed, 'condition': condition})
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
                         'value': value - h['gains'][endpoint],
                         'absolute_recovery': value,
                         'selected_candidate': p['selected'][endpoint]})
        for task, value in p['utility'].items():
            rows.append({'year': 2017, 'seed': seed, 'condition': condition, 'weight': weight,
                         'scope': scope, 'kind': 'utility_gain_vs_H', 'endpoint': task,
                         'value': h['utility'][task] - value, 'absolute_recovery': value,
                         'selected_candidate': None})

    means = defaultdict(list)
    for row in rows:
        if row['scope'] != MAIN_SCOPE:
            continue
        means[row['condition'], row['weight'], row['kind'], row['endpoint']].append(row['value'])
    summary = [{'condition': c, 'weight': w, 'kind': k, 'endpoint': e,
                'seed_mean': float(np.mean(v)), 'seed_min': float(np.min(v)),
                'seed_max': float(np.max(v)), 'seeds': len(v)}
               for (c, w, k, e), v in sorted(means.items())]

    record = {
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT',
        'panel_fixed_in_protocol_before_any_2017_number_was_read': True,
        'panel': panel_2017(),
        'conditions_present': sorted({c for (_s, c, _w, _sc) in points}),
        'missing_units': missing,
        'scope': MAIN_SCOPE, 'budget': BUDGET,
        'summary': summary,
        'interval_policy': ('No intervals are computed on the spent partition. Were any '
                            'computed, they would be conditional descriptive sampling '
                            'uncertainty and would not restore independence after repeated '
                            'development.'),
        'year_separation': 'Years are reported separately; rows are never pooled across years.',
        'inputs': registry.dump(),
    }
    csvout(out / 'EXPLORATORY_2017.csv', rows)
    write_json(out / 'EXPLORATORY_2017.json', record)
    print('REPORT_2017 done:', len(record['conditions_present']), 'conditions,',
          len(missing), 'missing units', flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()

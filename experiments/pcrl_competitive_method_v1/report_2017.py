"""EXPLORATORY 2017 summary: increments over H in the predecessor's `common_fresh` scope.

Transported Track E units come from this study's tree; `leace_A0`, `splince_A0` and
`optnet16_C1` from the direct-adversarial study's completed 2017 panel; `H`, `A0` and `J`
from the transport study. The neural nominees are bitwise J and inherit J's rows.
Seed means, per-anchor values and signs; no interval is claimed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.pcrl_direct_adversarial_v1.inputs import Registry, resolve
from experiments.pcrl_nonlinear_rank_v1.report_2017 import TRANSPORT_NAME, point, priors

from .common import ADVERSARIAL_RESULTS, OUT, read_json, sha_file, utcnow, write_json_atomic

SEEDS = (0, 1, 2)
SCOPE = 'common_fresh'
ENDS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
EXTERNAL = ('leace_A0', 'splince_A0', 'optnet16_C1')


def metrics_path(seed: int, condition: str) -> Path:
    mine = OUT / 'exploratory_2017' / f'seed_{seed}' / condition / 'mode_B' / 'metrics.json'
    if mine.exists():
        return mine
    if condition in EXTERNAL:
        return ADVERSARIAL_RESULTS / 'exploratory_2017' / f'seed_{seed}' / condition / 'mode_B' / 'metrics.json'
    return resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/{condition}/mode_B/metrics.json')


def run(units) -> dict:
    registry = Registry.new()
    conditions = ('H', 'A0', 'J') + EXTERNAL + tuple(units)
    rows, missing, sources = [], [], {}
    for seed in SEEDS:
        prior = priors(seed, registry)
        pts = {}
        for c in conditions:
            try:
                path = metrics_path(seed, c)
            except Exception:
                path = Path('/nonexistent')
            if not Path(path).exists():
                missing.append({'seed': seed, 'condition': c})
                continue
            sources[f'{seed}/{c}'] = sha_file(path)
            for w in ('unweighted', 'person_weighted'):
                pts[c, w] = point(read_json(path)['raw_metrics'], prior, w, SCOPE)
        for (c, w), p in pts.items():
            h = pts.get(('H', w))
            if h is None or c == 'H':
                continue
            row = {'seed': seed, 'condition': c, 'weight': w,
                   'residence_gain_vs_H': h['utility']['same_residence'] - p['utility']['same_residence']}
            for e in ENDS:
                row['increment/' + e] = p['gains'][e] - h['gains'][e]
            rows.append(row)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['condition'], r['weight']].append(r)
    means = {f'{c}|{w}': {k: float(np.mean([g[k] for g in v])) for k in v[0]
                          if k.startswith(('increment', 'residence'))} | {'anchors': len(v)}
             for (c, w), v in grouped.items()}
    record = {'generated_utc': utcnow(), 'status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT (2017 used repeatedly)',
              'scope': SCOPE, 'rows': rows, 'seed_means': means, 'missing': missing,
              'sources': sources,
              'neural_nominees': 'N_J_g000_C1_b100 / _b300 and their L1/L2 controls are bitwise J: see J'}
    write_json_atomic(OUT / 'EXPLORATORY_2017.json', record)
    return record


if __name__ == '__main__':
    import sys
    r = run(sys.argv[1:])
    for k, v in r['seed_means'].items():
        if k.endswith('unweighted'):
            print('%-14s' % k.split('|')[0], ' '.join('%+.4f' % v['increment/' + e] for e in ENDS),
                  ' res %+.4f n=%d' % (v['residence_gain_vs_H'], v['anchors']))
    print('missing', r['missing'])

"""Summarise the prespecified stronger attack: increments over the same-recipe H attack.

Per-person losses were not stored for this stage, so no bootstrap interval is claimed:
seed means, per-anchor values and per-anchor signs of each difference only.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import evidence as ev
from .common import OUT, read_json, sha_file, utcnow, write_json_atomic

SEEDS = (0, 1, 2)
ENDS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')


def run() -> dict:
    gain = {}
    conditions = sorted(p.name for p in (OUT / 'stress' / 'seed_0').iterdir())
    for seed in SEEDS:
        for c in conditions:
            for row in read_json(OUT / 'stress' / f'seed_{seed}' / c / 'stress_complete.json')['rows']:
                gain[seed, c, row['weight'], row['endpoint']] = row['gain_over_prior']
    table = {}
    for c in conditions:
        if c == 'H':
            continue
        for w in ('unweighted', 'person_weighted'):
            per = {e: [gain[s, c, w, e] - gain[s, 'H', w, e] for s in SEEDS] for e in ENDS}
            std = {}
            for e in ENDS:
                name = c if not c.startswith('ref_') else c
                try:
                    std[e] = float(np.mean([ev.increments(s, name, 'test', w)[e] for s in SEEDS]))
                except Exception:
                    std[e] = None
            table[f'{c}|{w}'] = {'stress_increment_seed_mean': {e: float(np.mean(v)) for e, v in per.items()},
                                 'stress_increment_per_anchor': per,
                                 'standard_audit_increment_seed_mean': std}
    panel = read_json(OUT / 'PANEL.json')['panel']
    diffs = []
    for p in panel:
        for right in ['ref_J', 'leace_A0', 'splince_A0', 'optnet16_C1'] + p['controls']:
            for w in ('unweighted', 'person_weighted'):
                for e in ENDS:
                    vals = [(gain[s, p['unit'], w, e] - gain[s, right, w, e]) for s in SEEDS]
                    diffs.append({'left': p['unit'], 'right': right, 'weight': w, 'endpoint': e,
                                  'seed_mean_recovery_difference': float(np.mean(vals)),
                                  'per_anchor': vals,
                                  'anchors_left_lower': int(sum(v < 0 for v in vals))})
    record = {'generated_utc': utcnow(), 'recipe': 'PROTOCOL section 7; MLP[256,256,128], 2 inits, 240 epochs',
              'conditions': table, 'panel_differences': diffs,
              'note': 'no interval: per-person stress losses were not stored; effect sizes and signs only'}
    write_json_atomic(OUT / 'STRESS.json', record)
    return record


if __name__ == '__main__':
    r = run()
    for k, v in r['conditions'].items():
        if k.endswith('unweighted'):
            print('%-20s' % k.split('|')[0], ' '.join('%+.4f' % v['stress_increment_seed_mean'][e] for e in ENDS),
                  ' | std', ' '.join('%+.4f' % (v['standard_audit_increment_seed_mean'][e] or 0) for e in ENDS))

"""Nominate the candidate panel (PROTOCOL §4) from VALIDATION-split evidence only.

Reads: validation-split audit and source-probe losses, the frozen `E_pca` source
reference, fixed training class entropies, and training-only distortion. Never reads a
test-split number, residence, or commute. The output is committed before any
test-split table of the panel is produced.
"""
from __future__ import annotations

import json

import numpy as np

from . import evidence as ev
from .common import OUT, read_json, utcnow, write_json_atomic
from .run_fit_n import BETAS, GAMMAS, INITS, unit_name

SEEDS = (0, 1, 2)
WEIGHTS = ('unweighted', 'person_weighted')
SOURCE = ('income_binary', 'civilian_at_work', 'public_coverage')
ALLOWANCE = 0.01
FALLBACK = {'neural': 'N_J_g010_C1_b100', 'projection': 'E_J_C_k2'}


def neural_configs() -> dict:
    return {unit_name(i, g, 'C1', b): {'family': 'neural', 'init': i, 'gamma': g, 'beta': b,
                                       'controls': [unit_name(i, g, p, b) for p in ('L1', 'L2')]}
            for i in INITS for g in GAMMAS for b in BETAS}


def projection_configs() -> dict:
    return {f'E_{c}_C_k{k}': {'family': 'projection', 'channel': c, 'k': k,
                              'controls': [f'E_{c}_L_k{k}', f'E_{c}_LX_k{k}']}
            for c in ('A0', 'J') for k in (1, 2, 4, 6, 8)}


def distortion(seed: int, unit: str, family: str) -> float:
    if family == 'neural':
        record = read_json(OUT / f'seed_{seed}' / 'fits' / unit / 'fit_record.json')
        return float(record['monitor_scores'][record['selected_index']]['distortion'])
    fit = read_json(OUT / f'seed_{seed}' / 'track_e_fit.json')
    return float(next(u['distortion_normalised'] for u in fit['units'] if u['unit'] == unit))


def assess(unit: str, spec: dict) -> dict:
    missing = [s for s in SEEDS if not ev.audited(s, unit)]
    if missing:
        return {'unit': unit, **spec, 'status': 'not audited', 'missing_seeds': missing}
    source_rows, eligible = [], True
    for seed in SEEDS:
        for weight in WEIGHTS:
            parent = ev.epca_source(seed, 'validation', weight)
            utility = ev.point(seed, unit, 'validation', weight)['utility']
            for task in SOURCE:
                diff = utility[task] - parent[task]
                ok = bool(diff <= ALLOWANCE + 1e-12)
                eligible &= ok
                source_rows.append({'seed': seed, 'weight': weight, 'task': task,
                                    'difference_vs_E_pca': diff, 'pass': ok})
    per_endpoint = {}
    for endpoint in ev.SENSITIVE4:
        target = endpoint.split('/')[1]
        values = [ev.increments(seed, unit, 'validation', 'unweighted')[endpoint]
                  / ev.training_entropy(seed)[target] for seed in SEEDS]
        per_endpoint[endpoint] = float(np.mean(values))
    worst = max(max(0.0, v) for v in per_endpoint.values())
    dist = float(np.mean([distortion(s, unit, spec['family']) for s in SEEDS]))
    return {'unit': unit, **spec, 'status': 'assessed', 'eligible': bool(eligible),
            'source_checks': source_rows,
            'normalised_increment_seed_mean': per_endpoint,
            'worst_normalised_positive_increment': worst,
            'training_distortion_mean': dist}


def nominate(configs: dict) -> dict:
    rows = [assess(u, s) for u, s in sorted(configs.items())]
    ranked = sorted((r for r in rows if r.get('eligible')),
                    key=lambda r: (r['worst_normalised_positive_increment'],
                                   r['training_distortion_mean'], r['unit']))
    return {'assessed': rows, 'eligible_count': len(ranked),
            'nominees': [r['unit'] for r in ranked[:2]],
            'ranking': [r['unit'] for r in ranked]}


def run() -> dict:
    neural = nominate(neural_configs())
    projection = nominate(projection_configs())
    panel = []
    for family, result, configs in (('neural', neural, neural_configs()),
                                    ('projection', projection, projection_configs())):
        if result['nominees']:
            for unit in result['nominees']:
                panel.append({'unit': unit, 'family': family, 'status': 'NOMINATED',
                              'controls': configs[unit]['controls']})
        else:
            unit = FALLBACK[family]
            panel.append({'unit': unit, 'family': family,
                          'status': 'FALLBACK - INELIGIBLE, diagnostic transport only',
                          'controls': configs[unit]['controls']})
    record = {'generated_utc': utcnow(),
              'rule': 'PROTOCOL.md section 4; validation split only; no residence, commute or '
                      'test-split value read',
              'panel': panel, 'neural': neural, 'projection': projection}
    write_json_atomic(OUT / 'PANEL.json', record)
    return record


if __name__ == '__main__':
    result = run()
    print(json.dumps(result['panel'], indent=1))
    for fam in ('neural', 'projection'):
        print(fam, 'eligible', result[fam]['eligible_count'], 'ranking', result[fam]['ranking'][:6])

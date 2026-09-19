"""2018 development report: points, paired intervals, families P and X, decision levels.

Fits nothing and selects nothing. Bootstrap, simultaneous and candidate-wide machinery
are the predecessor's, imported unchanged. New here: the family definitions of
PROTOCOL §5 and the decision levels of PROTOCOL §6.

Sign convention (inherited): for `recovery/*` endpoints the estimate is the
left-minus-right RECOVERY difference (negative = left leaks less); for
`utility/same_residence` it is the left-minus-right LOSS difference (positive = left
retains less residence capability). "Better" is `adjusted_high < 0` for both.
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.pcrl_direct_adversarial_v1 import report as rep
from experiments.pcrl_direct_adversarial_v1.run_report import (build_bootstrap,
                                                               contrast_intervals)
from experiments.pcrl_nonlinear_rank_v1.run_report import _losses

from . import evidence as ev
from .common import OUT, read_json, sha_file, utcnow, write_json_atomic
from .run_fit_n import BETAS, GAMMAS, INITS, POLICIES as NPOL, unit_name

SEEDS = (0, 1, 2)
HISTORICAL = ('H', 'A0', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1')
EXTERNAL = ('leace_A0', 'splince_A0', 'optnet16_C1')
MARGIN = 0.001
FAMILY_ENDPOINTS = rep.FAMILY_ENDPOINTS


def all_units() -> list:
    matrix = read_json(OUT / 'MATRIX.json')
    names = [s['unit'] for s in matrix['track_N']['slots'] + matrix['track_E']['slots']]
    return ['ref_A0', 'ref_J'] + names


def available_units(units) -> list:
    return [u for u in units if all(ev.audited(s, u) for s in SEEDS)]


def link_duplicates(units):
    for seed in SEEDS:
        for unit in units:
            canonical = ev.canonical(seed, unit)
            link = OUT / f'seed_{seed}' / unit
            if canonical != unit and not link.exists():
                link.symlink_to(OUT / f'seed_{seed}' / canonical)


def loss_vectors(conditions, points, frames, registry):
    """Per-person TEST losses of the selected candidate, family endpoints only."""
    vectors = {}
    for seed in SEEDS:
        labels = frames[seed][2]
        for condition in conditions:
            path = rep.condition_file(OUT, seed, condition, 'predictions.npz')
            registry.add(path)
            cache = {}
            with np.load(path) as store:
                for weight in rep.WEIGHTS:
                    point = points[seed, condition, rep.MAIN_SPLIT, weight, rep.MAIN_BUDGET,
                                   rep.MAIN_SCOPE]
                    for endpoint in FAMILY_ENDPOINTS:
                        kind, rest = endpoint.split('/', 1)
                        if kind == 'utility':
                            row = point['selected']['utility/' + rest]
                            target = rest
                            key = f"utility/{row['view']}/{rest}/None/{row['candidate_id']}/test"
                        else:
                            row = point['selected']['audit/' + rest]
                            target = rest.split('/')[1]
                            key = (f"audit/{row['view']}/{target}/{row['audit_budget']}/"
                                   f"{row['candidate_id']}/test")
                        if key not in cache:
                            cache[key] = _losses(store[key], labels['test'][target])[0]
                        vectors[seed, condition, weight, endpoint] = (target, cache[key])
    return vectors


# ------------------------------------------------------------------ families
def family_x(units) -> list:
    """Exploratory whole-grid family (PROTOCOL §5). Never used to nominate."""
    have = set(units)
    specs = []

    def add(left, right, family):
        if left in have and right in have and left != right:
            specs.append({'left': left, 'right': right, 'family': family})

    for init in INITS:
        for gamma in GAMMAS:
            for beta in BETAS:
                for policy in NPOL:
                    unit = unit_name(init, gamma, policy, beta)
                    if gamma != 1.0:
                        add(unit, unit_name(init, 1.0, policy, beta), 'N_gamma_vs_1')
                    if init == 'J':
                        add(unit, unit_name('A0', gamma, policy, beta), 'N_J_vs_A0_init')
                c1 = unit_name(init, gamma, 'C1', beta)
                add(c1, unit_name(init, gamma, 'L1', beta), 'N_C1_vs_L1')
                add(c1, unit_name(init, gamma, 'L2', beta), 'N_C1_vs_L2')
    for channel in ('A0', 'J'):
        for k in (1, 2, 4, 6, 8):
            c = f'E_{channel}_C_k{k}'
            for right, fam in ((f'E_{channel}_L_k{k}', 'E_C_vs_L'),
                               (f'E_{channel}_LX_k{k}', 'E_C_vs_LX'),
                               (f'E_{channel}_marginal_k{k}', 'E_C_vs_marginal'),
                               (f'E_{channel}_pca_k{k}', 'E_C_vs_pca'),
                               (f'E_{channel}_rand_k{k}', 'E_C_vs_random')):
                add(c, right, fam)
    for unit in units:
        if unit.startswith(('N_', 'E_')) or unit in ('leace_J', 'splince_J'):
            add(unit, 'J', 'grid_vs_J')
            add(unit, 'leace_A0', 'grid_vs_leace_A0')
    return specs


def family_p(panel: list, units) -> list:
    have = set(units)
    specs = []

    def add(left, right, family):
        if left in have and right in have and left != right:
            specs.append({'left': left, 'right': right, 'family': family})

    for entry in panel:
        unit = entry['unit']
        for right in ('J',) + EXTERNAL:
            add(unit, right, 'P_vs_external')
        for control in entry['controls']:
            add(unit, control, 'P_vs_local_control')
        if entry['family'] == 'projection':
            channel, k = unit.split('_')[1], unit.split('_k')[-1]
            for control in ('marginal', 'pca', 'rand'):
                add(unit, f'E_{channel}_{control}_k{k}', 'P_vs_compression_or_erasure')
            add(unit, 'leace_J', 'P_vs_compression_or_erasure')
            add(unit, 'ref_A0' if channel == 'A0' else 'ref_J', 'P_vs_starting_channel')
        else:
            parts = unit.split('_')
            init, gtag, beta = parts[1], parts[2], parts[-1]
            add(unit, f'N_{init}_g100_C1_{beta}', 'P_vs_gamma1')
            other = 'A0' if init == 'J' else 'J'
            add(unit, f'N_{other}_{gtag}_C1_{beta}', 'P_vs_other_init')
    return specs


# ------------------------------------------------------------------ point tables
def point_rows(conditions, points) -> list:
    rows = []
    for seed in SEEDS:
        for weight in rep.WEIGHTS:
            h = points[seed, 'H', rep.MAIN_SPLIT, weight, rep.MAIN_BUDGET, rep.MAIN_SCOPE]
            parent = ev.epca_source(seed, 'test', weight)
            for condition in conditions:
                p = points[seed, condition, rep.MAIN_SPLIT, weight, rep.MAIN_BUDGET,
                           rep.MAIN_SCOPE]
                row = {'seed': seed, 'condition': condition, 'weight': weight,
                       'residence_loss': p['utility']['same_residence'],
                       'residence_gain_vs_H': h['utility']['same_residence']
                                              - p['utility']['same_residence'],
                       'commute_loss': p['utility']['commute_over20']}
                for endpoint in p['gains']:
                    row['increment/' + endpoint] = p['gains'][endpoint] - h['gains'][endpoint]
                    row['recovery/' + endpoint] = p['gains'][endpoint]
                for task in ('income_binary', 'civilian_at_work', 'public_coverage'):
                    row['source_minus_Epca/' + task] = p['utility'][task] - parent[task]
                row['source_allowance_pass'] = all(
                    row['source_minus_Epca/' + t] <= 0.01 + 1e-12
                    for t in ('income_binary', 'civilian_at_work', 'public_coverage'))
                row['residence_reference_01_pass'] = row['residence_gain_vs_H'] >= 0.01 - 1e-12
                rows.append(row)
    return rows


def seed_means(rows) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['condition'], row['weight']].append(row)
    out = {}
    for (condition, weight), group in grouped.items():
        numeric = {k: float(np.mean([g[k] for g in group])) for k in group[0]
                   if isinstance(group[0][k], float)}
        numeric['source_allowance_pass_seeds'] = sum(g['source_allowance_pass'] for g in group)
        out[f'{condition}|{weight}'] = numeric
    return out


# ------------------------------------------------------------------ decisions
def decisions(panel, p_rows) -> list:
    index = {(r['left'], r['right'], r['weight'], r['endpoint']): r for r in p_rows}
    sensitive = [e for e in FAMILY_ENDPOINTS if e.startswith('recovery/')]
    out = []
    for entry in panel:
        unit = entry['unit']
        per_comparator = {}
        for comparator in ('J',) + EXTERNAL + tuple(entry['controls']):
            cell = {}
            for weight in rep.WEIGHTS:
                rows = {e: index.get((unit, comparator, weight, e)) for e in FAMILY_ENDPOINTS}
                if any(v is None for v in rows.values()):
                    cell[weight] = None
                    continue
                better = [e for e in sensitive if rows[e]['candidate_wide_high'] < 0]
                residence_ni = rows['utility/same_residence']['candidate_wide_high'] <= MARGIN
                others_ni = {e: rows[e]['candidate_wide_high'] <= MARGIN for e in sensitive}
                others_ni_zero = {e: rows[e]['candidate_wide_high'] <= 0 for e in sensitive}
                cell[weight] = {
                    'superior_endpoints': better,
                    'residence_noninferior_001': bool(residence_ni),
                    'residence_adjusted_high': rows['utility/same_residence']['candidate_wide_high'],
                    'sensitive_noninferior_001': others_ni,
                    'sensitive_noninferior_zero_margin': others_ni_zero,
                    'qualifies': bool(better and residence_ni
                                      and all(others_ni[e] for e in sensitive if e not in better))}
            per_comparator[comparator] = cell
        both = lambda c: all(per_comparator.get(c, {}).get(w) and
                             per_comparator[c][w]['qualifies'] for w in rep.WEIGHTS)
        competitive = both('J') and any(both(c) for c in EXTERNAL)
        coalition = competitive and all(both(c) for c in entry['controls'])
        out.append({'unit': unit, 'family': entry['family'], 'status': entry['status'],
                    'per_comparator': per_comparator,
                    'competitive_development_tradeoff_both_weightings': bool(competitive),
                    'coalition_specific_benefit': bool(coalition),
                    'qualifies_vs_J': {w: (per_comparator.get('J', {}).get(w) or {}).get('qualifies')
                                       for w in rep.WEIGHTS}})
    return out


def write_csv(path: Path, rows: list):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(panel_path: Path = OUT / 'PANEL.json') -> dict:
    from experiments.pcrl_direct_adversarial_v1.inputs import Registry
    registry = Registry.new()
    units = available_units(all_units())
    link_duplicates(units)
    conditions = tuple(HISTORICAL) + tuple(units)
    points, _raw, _prior = rep.load_points(OUT, SEEDS, conditions, registry)
    boot, frames = build_bootstrap(SEEDS, registry)
    vectors = loss_vectors(conditions, points, frames, registry)
    panel = read_json(panel_path)['panel'] if panel_path.exists() else []

    results = {}
    for name, specs in (('P', family_p(panel, units)), ('X', family_x(units))):
        rows, replicates = contrast_intervals(boot, vectors, SEEDS, specs)
        rows = rep.candidate_wide(rows, replicates)
        write_csv(OUT / f'INTERVALS_{name}.csv', rows)
        results[name] = {'contrasts': len(specs), 'rows': rows}
    table = point_rows(conditions, points)
    write_csv(OUT / 'POINTS_2018.csv', table)
    summary = {
        'generated_utc': utcnow(), 'units_available': len(units),
        'units_expected': len(all_units()), 'conditions': list(conditions),
        'family_sizes': {k: len(v['rows']) for k, v in results.items()},
        'seed_means': seed_means(table),
        'decisions': decisions(panel, results['P']['rows']) if panel else [],
        'sources': {name: sha_file(OUT / name) for name in
                    ('POINTS_2018.csv', 'INTERVALS_P.csv', 'INTERVALS_X.csv')
                    if (OUT / name).exists()},
        'registry': registry.dump()}
    write_json_atomic(OUT / 'DEVELOPMENT_2018.json', summary)
    return summary


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('units_available', 'units_expected',
                                             'family_sizes')}, indent=1))

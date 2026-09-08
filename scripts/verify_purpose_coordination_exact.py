"""Exact finite sign-channel coordination; Python standard library only.

Freeze source/config/protocol with --freeze before the first calculation.
All probabilistic arithmetic and comparisons use Fraction, not tolerances.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from math import comb, factorial
from pathlib import Path
import time

SIGNS = (-1, 1)
PAIRS = tuple(product(SIGNS, repeat=2))
ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'purpose_coordination_exact_config.json'
PROTOCOL = 'PURPOSE_COORDINATION_EXACT_PROTOCOL.md'
FREEZE = 'purpose_coordination_exact_freeze.json'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def csv_write(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def probability(value):
    return {'fraction': str(value), 'numerator': value.numerator,
            'denominator': value.denominator, 'decimal': float(value)}


def sources(out):
    paths = [Path(__file__), ROOT/'tests/test_purpose_coordination_exact.py', out/CONFIG, out/PROTOCOL]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def freeze(out):
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'phase': 'before exact enumeration', 'source_config_protocol_sha256': sources(out)}
    path = out/FREEZE
    if path.exists():
        old = json.loads(path.read_text())
        if old['source_config_protocol_sha256'] != record['source_config_protocol_sha256']:
            raise ValueError('Existing exact-study freeze differs; preserve and document it rather than overwrite')
        return old
    json_write(path, record)
    return record


def validate_law(law):
    assert set(law) == set(PAIRS)
    assert all(isinstance(p, F) and p >= 0 for p in law.values())
    assert sum(law.values(), F(0)) == 1


def forward_joint(law, k, policy):
    """Push each latent/noise outcome forward to its complete output history."""
    validate_law(law)
    if k not in (1, 2, 4) or policy not in ('fresh', 'cached'):
        raise ValueError('Outside the frozen history matrix')
    result = defaultdict(F)
    for u, v in PAIRS:
        for noises in product(PAIRS, repeat=k if policy == 'fresh' else 1):
            mass = F(1, 4)
            for noise in noises:
                mass *= law[noise]
            if not mass:
                continue
            sequence = noises if policy == 'fresh' else noises*k
            history = tuple((u*n1, v*n2) for n1, n2 in sequence)
            result[u, v, history] += mass
    assert sum(result.values(), F(0)) == 1
    return dict(result)


def inverse_joint(law, k, policy):
    """Evaluate likelihood of every possible observed history for each U,V.

    This enumerates outputs, not the forward algorithm's sampled-noise states.
    Zero-probability histories are retained explicitly.
    """
    result = {}
    for history in product(PAIRS, repeat=k):
        for u, v in PAIRS:
            if policy == 'cached':
                likelihood = law[u*history[0][0], v*history[0][1]] if len(set(history)) == 1 else F(0)
            else:
                likelihood = F(1)
                for r1, r2 in history:
                    likelihood *= law[u*r1, v*r2]
            result[u, v, history] = likelihood/4
    return result


def marginals(joint):
    result = {name: defaultdict(F) for name in ('coalition', 'own1', 'own2', 'individual1_S', 'individual2_S')}
    for (u, v, history), mass in joint.items():
        h1, h2 = tuple(pair[0] for pair in history), tuple(pair[1] for pair in history)
        result['coalition'][u*v, history] += mass
        result['own1'][u, h1] += mass
        result['own2'][v, h2] += mass
        result['individual1_S'][u*v, h1] += mass
        result['individual2_S'][u*v, h2] += mass
    return {name: dict(table) for name, table in result.items()}


def bayes_accuracy(joint):
    groups = defaultdict(lambda: {s: F(0) for s in SIGNS})
    for (target, observation), mass in joint.items():
        groups[observation][target] += mass
    return sum((max(values.values()) for values in groups.values()), F(0))


def independent_full_joint(joint):
    targets, observations = defaultdict(F), defaultdict(F)
    for (target, observation), mass in joint.items():
        targets[target] += mass
        observations[observation] += mass
    residuals = [joint.get((target, observation), F(0))-pt*po
                 for target, pt in targets.items() for observation, po in observations.items()]
    return all(value == 0 for value in residuals), max(map(abs, residuals), default=F(0))


def binomial_own_accuracy(k, q):
    """Majority decoder with half credit for ties under a symmetric channel."""
    return sum((F(comb(k, errors))*q**errors*(1-q)**(k-errors)*
                (F(1) if 2*errors < k else F(1, 2) if 2*errors == k else F(0))
                for errors in range(k+1)), F(0))


def count_vector_bayes(law, k):
    """Third check: unordered output counts and multinomial multiplicities."""
    rows, total = [], F(0)
    for counts in product(range(k+1), repeat=4):
        if sum(counts) != k:
            continue
        multiplicity = factorial(k)
        for count in counts:
            multiplicity //= factorial(count)
        masses = {s: F(0) for s in SIGNS}
        for u, v in PAIRS:
            mass = F(1, 4)
            for (r1, r2), count in zip(PAIRS, counts):
                mass *= law[u*r1, v*r2]**count
            masses[u*v] += mass
        contribution = multiplicity*max(masses.values())
        total += contribution
        rows.append({'counts_order': '--,-+,+-,++', 'counts': json.dumps(counts),
                     'ordered_history_multiplicity': multiplicity,
                     'P_Sminus_one_history': str(masses[-1]), 'P_Splus_one_history': str(masses[1]),
                     'bayes_accuracy_contribution': str(contribution)})
    return total, rows


def evaluate(law, k, policy):
    forward = forward_joint(law, k, policy)
    inverse = inverse_joint(law, k, policy)
    assert {key: mass for key, mass in inverse.items() if mass} == forward
    tables = marginals(inverse)
    accuracy = {name: bayes_accuracy(table) for name, table in tables.items()}
    independence = {name: independent_full_joint(tables[name]) for name in
                    ('coalition', 'individual1_S', 'individual2_S')}
    assert independence['individual1_S'][0] and independence['individual2_S'][0]
    q1, q2 = 1-accuracy['own1'], 1-accuracy['own2']
    lower_bound = max(F(1, 2), 1-q1-q2)
    assert accuracy['coalition'] >= lower_bound
    count_rows = []
    if policy == 'fresh':
        count_accuracy, count_rows = count_vector_bayes(law, k)
        assert count_accuracy == accuracy['coalition']
    return {'joint': inverse, 'tables': tables, 'accuracy': accuracy,
            'independence': independence, 'decoder_error_bound': lower_bound, 'count_rows': count_rows}


def history_text(history):
    return '/'.join(''.join('+' if value == 1 else '-' for value in pair) for pair in history)


def own_history_text(history):
    return ''.join('+' if value == 1 else '-' for value in history)


def exact_rows(case, result):
    latent, coalition, individual, decoders = [], [], [], []
    for (u, v, history), mass in sorted(result['joint'].items()):
        latent.append({'case': case, 'U': u, 'V': v, 'S': u*v, 'history': history_text(history),
                       'probability': str(mass), 'numerator': mass.numerator, 'denominator': mass.denominator})
    for (s, history), mass in sorted(result['tables']['coalition'].items()):
        coalition.append({'case': case, 'S': s, 'history': history_text(history),
                          'probability': str(mass), 'numerator': mass.numerator, 'denominator': mass.denominator})
    for purpose in (1, 2):
        table = result['tables'][f'own{purpose}']
        for (s, history), mass in sorted(result['tables'][f'individual{purpose}_S'].items()):
            individual.append({'case': case, 'purpose': purpose, 'S': s, 'own_history': own_history_text(history),
                               'probability': str(mass)})
        for history in sorted({key[1] for key in table}):
            minus, plus = table.get((-1, history), F(0)), table.get((1, history), F(0))
            if not minus+plus:
                continue
            decoders.append({'case': case, 'purpose': purpose, 'target': 'U' if purpose == 1 else 'V',
                'own_history': own_history_text(history), 'prediction': 1 if plus >= minus else -1,
                'tie': plus == minus, 'joint_target_minus': str(minus), 'joint_target_plus': str(plus),
                'posterior_prediction_correct': str(max(minus, plus)/(minus+plus)),
                'other_purpose_history_used': False})
    return latent, coalition, individual, decoders


def plot_svg(out, rows):
    """Small deterministic scientific plot; no numerical-library imports."""
    colors = {'independent': '#1769aa', 'coordinated': '#be3f35', 'noiseless': '#666666'}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="470" viewBox="0 0 960 470">',
        '<rect width="960" height="470" fill="white"/>',
        '<g font-family="sans-serif" fill="#222"><text x="480" y="26" text-anchor="middle" font-size="18">Exact coalition Bayes accuracy from the complete paired history</text>',
        '<text x="480" y="49" text-anchor="middle" font-size="12">Same U,V; public channel laws; no task truth, noise realization, or other side information</text>']
    for panel, policy in enumerate(('fresh', 'cached')):
        left, right, top, bottom = 65+470*panel, 435+470*panel, 100, 342
        x = lambda k: left+(k-1)*(right-left)/3
        y = lambda a: bottom-(float(a)-.5)*(bottom-top)/.5
        parts.append(f'<text x="{(left+right)/2}" y="79" text-anchor="middle" font-size="16">{policy.capitalize()} paired releases</text>')
        for value in (F(1,2), F(5,8), F(3,4), F(7,8), F(1)):
            parts.append(f'<line x1="{left}" y1="{y(value)}" x2="{right}" y2="{y(value)}" stroke="#ddd"/><text x="{left-9}" y="{y(value)+4}" text-anchor="end" font-size="11">{value}</text>')
        parts.append(f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#333"/>')
        for k in (1,2,4):
            parts.append(f'<text x="{x(k)}" y="{bottom+20}" text-anchor="middle" font-size="12">{k}</text>')
        for law in colors:
            chosen = [row for row in rows if row['law'] == law and row['policy'] == policy]
            coordinates = ' '.join(f'{x(row["k"])},{y(F(row["coalition_accuracy"]))}' for row in chosen)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[law]}" stroke-width="2"/>')
            for row in chosen:
                xx, yy = x(row['k']), y(F(row['coalition_accuracy']))
                offset = 15 if law == 'independent' else -10
                parts.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{colors[law]}"/><text x="{xx}" y="{yy+offset}" text-anchor="middle" fill="{colors[law]}" font-size="11">{row["coalition_accuracy"]}</text>')
        utility = '3/4, 3/4, 27/32' if policy == 'fresh' else '3/4 at every k'
        parts.append(f'<text x="{(left+right)/2}" y="{bottom+45}" text-anchor="middle" font-size="12">Number of paired issues k</text><text x="{(left+right)/2}" y="{bottom+68}" text-anchor="middle" font-size="12">Noisy own-history task accuracy: {utility}</text>')
    for index, (law, color) in enumerate(colors.items()):
        xx = 195+235*index
        parts.append(f'<line x1="{xx}" y1="444" x2="{xx+25}" y2="444" stroke="{color}" stroke-width="3"/><text x="{xx+33}" y="449" font-size="13">{law}</text>')
    parts.append('</g></svg>')
    (out/'PURPOSE_COORDINATION_EXACT.svg').write_text('\n'.join(parts)+'\n')


def run(out):
    start = time.perf_counter()
    frozen = json.loads((out/FREEZE).read_text())
    if frozen['source_config_protocol_sha256'] != sources(out):
        raise ValueError('Source/config/protocol differs from pre-enumeration freeze')
    if (out/'PURPOSE_COORDINATION_EXACT.json').exists():
        raise FileExistsError('Completed exact result exists; preserve it and use the focused tests to verify')
    config = json.loads((out/CONFIG).read_text())
    laws = {name: {tuple(pair): F(p) for pair, p in zip(config['noise_order'], values)}
            for name, values in config['laws'].items()}
    results, summary, full, coalition, individual, decoders, count_rows = {}, [], [], [], [], [], []
    for name, law in laws.items():
        for policy in config['policies']:
            for index, k in enumerate(config['history_lengths']):
                case = f'{name}/{policy}/k{k}'
                result = evaluate(law, k, policy)
                expected_index = index if policy == 'fresh' else 0
                assert result['accuracy']['coalition'] == F(config['expected_fresh_coalition'][name][expected_index])
                expected_own = F(config['expected_fresh_own_task'][name][expected_index])
                assert result['accuracy']['own1'] == result['accuracy']['own2'] == expected_own
                if name != 'noiseless':
                    assert expected_own == binomial_own_accuracy(k if policy == 'fresh' else 1, F(1,4))
                if name == 'independent':
                    assert result['accuracy']['coalition'] == (1+(2*expected_own-1)**2)/2
                results[name, policy, k] = result
                summary.append({'case': case, 'law': name, 'policy': policy, 'k': k,
                    **{key+'_accuracy': str(value) for key, value in result['accuracy'].items()},
                    **{key+'_independent': value[0] for key, value in result['independence'].items()},
                    'coalition_independence_max_residual': str(result['independence']['coalition'][1]),
                    'decoder_error_bound': str(result['decoder_error_bound']), 'forward_inverse_exact': True,
                    'count_vector_bayes_exact': policy == 'fresh',
                    'latent_positive_cells': sum(p > 0 for p in result['joint'].values()),
                    'latent_all_cells': len(result['joint'])})
                for target, rows in zip((full, coalition, individual, decoders), exact_rows(case, result)):
                    target.extend(rows)
                count_rows.extend({'case': case, **row} for row in result['count_rows'])
    for policy in config['policies']:
        for k in config['history_lengths']:
            a, b = results['independent', policy, k], results['coordinated', policy, k]
            assert a['tables']['own1'] == b['tables']['own1']
            assert a['tables']['own2'] == b['tables']['own2']
    assert results['coordinated','fresh',1]['independence']['coalition'][0]
    grid_rows, grid_joint = [], []
    for q1, q2 in product(map(F, config['grid']), repeat=2):
        if q1+q2 > F(config['grid_sum_ceiling']):
            continue
        law = {(1,1): 1-q1-q2, (-1,1): q1, (1,-1): q2, (-1,-1): F(0)}
        result = evaluate(law, 1, 'fresh')
        expected = max(F(1,2), 1-q1-q2)
        assert result['accuracy']['own1'] == 1-q1 and result['accuracy']['own2'] == 1-q2
        assert result['accuracy']['coalition'] == expected
        case = f'disjoint/q1={q1}/q2={q2}'
        grid_joint.extend(exact_rows(case, result)[0])
        grid_rows.append({'case': case, 'q1': str(q1), 'q2': str(q2), 'bound': str(expected),
                          'coalition_accuracy': str(result['accuracy']['coalition']), 'exact_equality': True})
    assert len(grid_rows) == 15 and len(summary) == 18
    calculation_seconds = time.perf_counter()-start
    artifacts = {'PURPOSE_COORDINATION_EXACT.csv': summary, 'PURPOSE_COORDINATION_LATENT_JOINT.csv': full,
        'PURPOSE_COORDINATION_COALITION_JOINT.csv': coalition, 'PURPOSE_COORDINATION_INDIVIDUAL_JOINT.csv': individual,
        'PURPOSE_COORDINATION_DECODERS.csv': decoders, 'PURPOSE_COORDINATION_COUNT_CHECK.csv': count_rows,
        'PURPOSE_COORDINATION_BOUND_GRID.csv': grid_rows, 'PURPOSE_COORDINATION_GRID_JOINT.csv': grid_joint}
    for filename, rows in artifacts.items():
        csv_write(out/filename, rows)
    plot_svg(out, summary)
    record = {'status': 'all exact assertions passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'freeze_sha256': digest(out/FREEZE), 'source_config_protocol_sha256': sources(out),
        'arithmetic': config['arithmetic'], 'law_order': config['noise_order'], 'laws': config['laws'],
        'history_cases': summary, 'grid': grid_rows, 'checks': {
            'history_cases': 18, 'forward_inverse_full_joint_equal': 18, 'individual_history_independence_checks': 36,
            'matching_full_marginal_task_channels': 12, 'fresh_count_vector_checks': 9,
            'grid_bound_equalities': 15, 'grid_forward_inverse_checks': 15,
            'expected_values_verified_not_substituted': True, 'all_zero_cells_preserved': True},
        'calculation_seconds': calculation_seconds, 'total_calculation_and_export_seconds': time.perf_counter()-start,
        'no_fitting_no_sampling_no_blas': True,
        'artifacts': {name: {'sha256': digest(out/name), 'bytes': (out/name).stat().st_size, 'rows': len(rows)}
                      for name, rows in artifacts.items()},
        'plot': {'path': 'PURPOSE_COORDINATION_EXACT.svg', 'sha256': digest(out/'PURPOSE_COORDINATION_EXACT.svg')}}
    json_write(out/'PURPOSE_COORDINATION_EXACT.json', record)
    print(json.dumps({'status': record['status'], 'calculation_seconds': calculation_seconds,
                      'total_calculation_and_export_seconds': record['total_calculation_and_export_seconds']}))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    output = args.out.resolve()
    if args.freeze:
        print(json.dumps(freeze(output)))
    else:
        run(output)

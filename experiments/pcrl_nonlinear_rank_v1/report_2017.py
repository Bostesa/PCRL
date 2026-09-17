"""Phase 6: the EXPLORATORY cross-year table. Deliberately lean.

No simultaneous intervals are computed here and none should be. The 2017 seal is
spent, so an interval on this partition would invite exactly the confirmatory
reading the evidence cannot support. What a cross-year development table can
honestly do is show whether the 2018 attribution reproduces in direction, so
that is what it reports: seed-mean and per-seed point estimates, and a
sign-consistency check against the 2018 result.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from experiments.acs_transfer_heads import load_candidate

from .inputs import OUT, Registry, TRANSPORT_NAME, read_json, resolve, write_json
from .maps import HISTORICAL_ALIAS, condition_name
from .objective import POLICIES
from .report import FORBIDDEN, WEIGHTS
from .run_fit import limit_threads, machine_state

SCOPES = ('common_fresh', 'transport_all')
BUDGET = 360
UTILITY_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage', 'same_residence',
                 'commute_over20')
REFERENCES = ('H', 'J', 'E', 'A0', 'spectral_S0', 'spectral_C1', 'spectral_L1', 'spectral_L2')
NEW = tuple(condition_name(f, r, p) for r in (16, 8)
            for f in ('original', 'nonlinear') for p in POLICIES)


def metrics_path(out: Path, seed: int, condition: str) -> Path:
    """This study's exploratory unit, or the transport study's frozen Mode B unit."""
    alias = HISTORICAL_ALIAS.get(condition, condition)
    local = Path(out) / 'exploratory_2017' / f'seed_{seed}' / condition / 'mode_B' / 'metrics.json'
    if condition not in HISTORICAL_ALIAS and local.exists():
        return local
    return resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/{alias}/mode_B/metrics.json')


def priors(seed: int, registry: Registry) -> dict:
    """2017 fitting priors, reused from the transport study; never refitted."""
    selection = read_json(registry.resolve(
        f'results/{TRANSPORT_NAME}/seed_{seed}/reference/selection.json'))
    labels = np.load(registry.resolve(f'results/{TRANSPORT_NAME}/final_labels.npz'))
    out = {}
    for target, path in selection['priors'].items():
        y = labels['y/' + target]
        valid = y >= 0
        model = load_candidate(path)
        p = np.asarray(model.predict_proba(np.zeros((int(valid.sum()), 1))), dtype=float)
        p = np.clip(p, 1e-12, 1.)
        p /= p.sum(1, keepdims=True)
        yy = y[valid].astype(int)
        losses = -np.log(p[np.arange(len(yy)), yy])
        w = labels['weights'][valid].astype(float)
        out[target] = {'test': float(losses.mean()),
                       'test_person_weighted': float((w * losses).sum() / w.sum())}
    return out


def point(records, prior, weight: str, scope: str) -> dict:
    key = 'test' if weight == 'unweighted' else 'test_person_weighted'
    utility, gains, selected = {}, {}, {}
    for row in records:
        if row['role'] == 'utility' and row.get('selected'):
            utility[row['target']] = row['scores'][key]['log_loss']
        elif row['role'] == 'audit' and row.get('audit_budget') == BUDGET \
                and scope in row.get('selected_scopes', []):
            endpoint = row['view'] + '/' + row['target']
            if endpoint in gains:
                raise ValueError('duplicate selected audit candidate ' + endpoint)
            gains[endpoint] = prior[row['target']][key] - row['scores'][key]['log_loss']
            selected[endpoint] = row['candidate_id']
    return {'utility': utility, 'gains': gains, 'selected': selected}


def build(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    registry = Registry.new()
    conditions = tuple(dict.fromkeys(REFERENCES + NEW))
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
    return points, registry, missing


def rows_and_consistency(points, out: Path, seeds):
    """Per-seed rows plus a sign-consistency check against the 2018 development result."""
    rows = []
    for (seed, c, weight, scope), p in points.items():
        h = points.get((seed, 'H', weight, scope))
        if h is None:
            continue
        rows.append({'seed': seed, 'condition': c, 'weight': weight, 'scope': scope,
                     'kind': 'utility_gain_vs_H', 'endpoint': 'same_residence',
                     'value': h['utility']['same_residence'] - p['utility']['same_residence']})
        for endpoint, value in p['gains'].items():
            rows.append({'seed': seed, 'condition': c, 'weight': weight, 'scope': scope,
                         'kind': 'absolute_recovery', 'endpoint': endpoint, 'value': value})
            rows.append({'seed': seed, 'condition': c, 'weight': weight, 'scope': scope,
                         'kind': 'additional_recovery', 'endpoint': endpoint,
                         'value': value - h['gains'][endpoint]})

    dev = {}
    with open(Path(out) / 'PER_SEED.csv', newline='') as handle:
        for r in csv.DictReader(handle):
            if (r['split'] == 'test' and r['budget'] == '360'
                    and r['scope'] == 'kernel_expanded_catchup'
                    and r['kind'] in ('additional_recovery', 'utility_gain_vs_H')):
                dev[int(r['seed']), r['condition'], r['weight'], r['kind'], r['endpoint']] = \
                    float(r['value'])

    checks = []
    for scope in SCOPES:
        for pair in [(condition_name('nonlinear', r, p), condition_name('original', r, p))
                     for r in (16, 8) for p in POLICIES]:
            left, right = pair
            for weight, endpoint in itertools.product(WEIGHTS, ('A/SEX', 'A/RAC1P', 'AB/SEX',
                                                                'AB/RAC1P')):
                agree, total = 0, 0
                for seed in seeds:
                    a = points.get((seed, left, weight, scope))
                    b = points.get((seed, right, weight, scope))
                    dl = dev.get((seed, left, weight, 'additional_recovery', endpoint))
                    dr = dev.get((seed, right, weight, 'additional_recovery', endpoint))
                    if a is None or b is None or dl is None or dr is None:
                        continue
                    x = (a['gains'][endpoint] - points[seed, 'H', weight, scope]['gains'][endpoint]) \
                        - (b['gains'][endpoint] - points[seed, 'H', weight, scope]['gains'][endpoint])
                    total += 1
                    agree += int(np.sign(x) == np.sign(dl - dr) and x != 0)
                if total:
                    checks.append({'scope': scope, 'contrast': f'{left} vs {right}',
                                   'weight': weight, 'endpoint': endpoint,
                                   'seeds_compared': total, 'sign_agrees_with_2018': agree})
    return rows, checks


def csvout(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    out = Path(out)
    points, registry, missing = build(out, seeds)
    rows, checks = rows_and_consistency(points, out, seeds)
    record = {
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT',
        'scopes': list(SCOPES), 'budget': BUDGET, 'seeds': list(seeds),
        'conditions_with_complete_endpoints': sorted({c for (_s, c, _w, _sc) in points}),
        'missing_units': [{'seed': s, 'condition': c} for s, c in missing],
        'sign_consistency_with_2018': checks,
        'no_intervals_note': ('No simultaneous intervals are computed on this partition. Its seal '
                              'is spent, and an interval here would invite a confirmatory reading '
                              'the evidence cannot support.'),
        'scope_note': ('The original frozen 2017 transport result keeps its historical status and '
                       'is neither restated nor overwritten. Years are reported separately and '
                       'rows are never pooled across years.'),
        'inputs': registry.dump(), 'machine': machine_state(),
    }
    write_json(out / 'EXPLORATORY_2017.json', record)
    csvout(out / 'EXPLORATORY_2017.csv', rows)
    return record


def markdown(out: Path, record: dict) -> str:
    """Render EXPLORATORY_2017.md from the produced tables."""
    from collections import defaultdict
    rows = []
    path = Path(out) / 'EXPLORATORY_2017.csv'
    if path.exists():
        with open(path, newline='') as handle:
            rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['scope'], r['weight'], r['condition'], r['kind'], r['endpoint']].append(
            float(r['value']))

    def mean(scope, weight, condition, kind, endpoint):
        v = grouped.get((scope, weight, condition, kind, endpoint))
        return float(np.mean(v)) if v else None

    def fmt(v):
        return '—' if v is None else f'{v:.4f}'

    lines = ['# EXPLORATORY_2017 — cross-year development, not a confirmation', '',
             '**Every number in this file is EXPLORATORY CROSS-YEAR DEVELOPMENT evidence.**', '',
             'The 2017 locked evaluation is finished and its `final_evaluation` partition is spent.',
             'So this file:', '',
             '* is **not** a second confirmation of anything;',
             '* does **not** restate, reinterpret or overwrite the original frozen 2017 transport',
             '  result, which keeps its historical status;',
             '* uses the 2017 `final_evaluation` rows only as **now-exposed development',
             '  evaluation**;',
             '* reports **no simultaneous intervals**. ' + record['no_intervals_note'], '',
             'Probes were fitted on the 2017 attacker/task fitting partitions and selected on the',
             '2017 validation partitions by minimum unweighted validation log loss, so selection',
             'still never saw the evaluation rows. `ev.load_final` is deliberately not called:',
             'reusing a spent seal would misrepresent the evidence. Years are reported separately',
             'and rows are never pooled across years.', '']

    if record['missing_units']:
        lines += [f"**Incomplete: {len(record['missing_units'])} unit(s) missing.** "
                  'Their conditions are omitted from the tables below rather than partially '
                  'reported.', '']

    for scope in SCOPES:
        present = [c for c in (*REFERENCES, *NEW)
                   if mean(scope, 'unweighted', c, 'utility_gain_vs_H', 'same_residence') is not None]
        if not present:
            continue
        lines += [f'## Scope `{scope}`, budget {BUDGET}, seed means, unweighted', '',
                  '| condition | residence gain over H | add. A/SEX | add. A/RAC1P | '
                  'add. AB/SEX | add. AB/RAC1P |', '|---|---|---|---|---|---|']
        for c in present:
            lines.append('| `{}` | {} | {} | {} | {} | {} |'.format(
                c,
                fmt(mean(scope, 'unweighted', c, 'utility_gain_vs_H', 'same_residence')),
                fmt(mean(scope, 'unweighted', c, 'additional_recovery', 'A/SEX')),
                fmt(mean(scope, 'unweighted', c, 'additional_recovery', 'A/RAC1P')),
                fmt(mean(scope, 'unweighted', c, 'additional_recovery', 'AB/SEX')),
                fmt(mean(scope, 'unweighted', c, 'additional_recovery', 'AB/RAC1P'))))
        lines.append('')

    checks = record['sign_consistency_with_2018']
    if checks:
        agree = sum(c['sign_agrees_with_2018'] for c in checks)
        total = sum(c['seeds_compared'] for c in checks)
        lines += ['## Does the 2018 attribution reproduce in direction?', '',
                  'For each nonlinear-versus-original contrast at matched rank and policy, the sign',
                  'of the change in additional recovery is compared against the 2018 development',
                  f'result, per seed: **{agree} of {total} seed-level comparisons agree in sign**.',
                  '', 'This is a direction check on an exposed partition. It is not a replication,',
                  'it carries no interval, and a cross-year change in effect magnitude is not',
                  'attributed to sample size here — the predecessor study saw magnitudes move by',
                  'large factors between years, and its own per-seed sign agreement was weaker than',
                  'its seed-mean statement suggested.', '',
                  '| scope | contrast | weight | endpoint | seeds | sign agrees |',
                  '|---|---|---|---|---|---|']
        for c in checks:
            if c['seeds_compared'] != c['sign_agrees_with_2018']:
                lines.append('| {} | `{}` | {} | {} | {} | {} |'.format(
                    c['scope'], c['contrast'], c['weight'], c['endpoint'],
                    c['seeds_compared'], c['sign_agrees_with_2018']))
        lines += ['', '(only rows where agreement is not unanimous are listed; the full set is in',
                  '`EXPLORATORY_2017.json`)', '']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    r = run(a.out, tuple(a.seeds))
    (Path(a.out) / 'EXPLORATORY_2017.md').write_text(markdown(a.out, r))
    print('conditions with complete endpoints:', len(r['conditions_with_complete_endpoints']),
          '| missing units:', len(r['missing_units']),
          '| consistency checks:', len(r['sign_consistency_with_2018']))


if __name__ == '__main__':
    main()

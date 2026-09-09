"""Independent stdlib replay of the published coalition-strength comparisons.

Reads compact selected scores, frozen rules and derived CSV files.  It imports
neither reporter/comparison helpers nor a numerical/model library, and fits
nothing.  Its arithmetic deliberately starts again from PER_SEED scores.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ('income_binary', 'civilian_at_work', 'public_coverage')
TASKS = (*SOURCE, 'same_residence', 'commute_over20')
VIEWS = dict(zip(TASKS, ('A', 'A', 'B', 'A', 'B')))
ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
         'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
         'AB': ('SEX', 'RAC1P')}
AUDITS = tuple(view+'/'+task for view, tasks in ROLES.items() for task in tasks)
PAIR_ID = ('interface', 'J_beta', 'Iplus_beta', 'J_condition', 'Iplus_condition',
           'attribute', 'weighting', 'scope', 'audit_budget', 'panel', 'delta')
STR_FIELDS = {'J_beta', 'Iplus_beta', 'beta'}
ROUND = 1e-12
REPLAY_ATOL = 5e-14


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def defined(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def conjunction(values):
    values = tuple(values)
    return None if None in values else all(values)


def subtract(a, b):
    return a-b if defined(a) and defined(b) else None


def mean_sd(values):
    if not all(defined(x) for x in values):
        return None, None
    mean = math.fsum(values)/len(values)
    variance = math.fsum((x-mean)**2 for x in values)/(len(values)-1)
    return mean, math.sqrt(variance)


def cell(value, key):
    if value == '':
        return None
    if key in STR_FIELDS:
        return value
    if value[0] in '{[':
        return json.loads(value)
    if value in ('True', 'False'):
        return value == 'True'
    try:
        number = float(value)
        return int(value) if value.lstrip('-').isdigit() else number
    except ValueError:
        return value


class Replay:
    def __init__(self, out):
        self.out = out
        self.inputs = {}
        self.checks = 0
        self.max_error = 0.
        self.counts = {}
        self.compact = self.read_json(out/'COMPACT_EXPORTS.json')['files']

    def bind(self, path):
        path = Path(path).resolve()
        value = digest(path)
        if path in self.inputs:
            assert self.inputs[path] == value, 'Input changed during replay: '+str(path)
        self.inputs[path] = value
        return value

    def read_json(self, path):
        self.bind(path)
        return json.loads(Path(path).read_text())

    def rows(self, name):
        plain, compressed = self.out/name, self.out/(name+'.gz')
        if name in self.compact:
            record = self.compact[name]
            assert self.bind(compressed) == record['gzip_sha256'], name+' gzip identity'
            h = hashlib.sha256()
            with gzip.open(compressed, 'rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == record['plain_sha256'], name+' decompressed identity'
            if plain.exists():
                assert self.bind(plain) == record['plain_sha256'], name+' local plain identity'
            opener, path = gzip.open, compressed
        else:
            self.bind(plain)
            opener, path = open, plain
        with opener(path, 'rt', newline='') as stream:
            for row in csv.DictReader(stream):
                yield {key: cell(value, key) for key, value in row.items()}

    def same(self, actual, expected, where):
        self.checks += 1
        if isinstance(expected, dict):
            assert isinstance(actual, dict), where+' expected mapping'
            for key, value in expected.items():
                assert str(key) in actual, where+'/'+str(key)+' absent'
                self.same(actual[str(key)], value, where+'/'+str(key))
        elif isinstance(expected, (list, tuple)):
            assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), where+' sequence length'
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.same(a, b, where+'/'+str(i))
        elif isinstance(expected, bool) or expected is None:
            assert actual is expected, f'{where}: {actual!r} != {expected!r}'
        elif defined(expected):
            assert defined(actual), where+' missing numeric value'
            error = abs(actual-expected)
            self.max_error = max(self.max_error, error)
            assert error <= REPLAY_ATOL, f'{where}: {actual!r} != {expected!r}'
        else:
            assert actual == expected, f'{where}: {actual!r} != {expected!r}'

    def prepare(self):
        self.rules = rules = self.read_json(self.out/'comparison_rules.json')
        freeze = self.read_json(self.out/'protocol_freeze.json')
        assert digest(self.out/'comparison_rules.json') == freeze['scientific_and_protocol_sha256'][str((self.out/'comparison_rules.json').relative_to(ROOT))]
        assert rules['seeds'] == [0, 1, 2]
        assert rules['beta_decimal_strings'] == ['0', '0.025', '0.05', '0.1', '0.2']
        assert rules['comparison_roundoff_tolerance'] == rules['pareto']['roundoff_tolerance'] == ROUND
        assert rules['source_feasibility']['maximum_per_task_loss_increase'] == .01
        assert rules['pareto']['delta_used_for_dominance'] is False
        assert tuple(rules['fixed_contrast_targets']['authorized_utility']) == TASKS
        assert rules['fixed_contrast_targets']['forbidden_audits'] == {k: list(v) for k, v in ROLES.items()}
        self.entries = self.read_json(self.out/'REUSE_MANIFEST.json')['systems']
        assert len(self.entries) == len({(e['seed'], e['condition']) for e in self.entries}) == 54
        self.alias = {}
        self.names = {e['condition']: e['interface'] for e in self.entries if e['seed'] == 0}
        for e in self.entries:
            beta = format(e['beta'], '.15g')
            for family in (('J', 'Iplus') if e['beta'] == 0 else (e['family'],)):
                key = (e['seed'], e['interface'], family, beta)
                assert key not in self.alias
                self.alias[key] = e['condition']
        expected = set(itertools.product(rules['seeds'], rules['interfaces'], ('J', 'Iplus'), rules['beta_decimal_strings']))
        assert set(self.alias) == expected
        assert sum(e['reused'] for e in self.entries) == 18
        for seed in rules['seeds']:
            for view in ('F', 'P'):
                assert self.alias[seed, view, 'J', '0'] == self.alias[seed, view, 'Iplus', '0']
        self.parent = {}
        for seed, parent in rules['original_parent_metric_identity'].items():
            raw = self.read_json(ROOT/parent['metrics_path'])['raw_metrics']
            selected = self.read_json(ROOT/parent['selection_path'])['head_selections']
            assert digest(ROOT/parent['metrics_path']) == parent['metrics_sha256']
            assert digest(ROOT/parent['selection_path']) == parent['selection_sha256']
            for task, identity in parent['tasks'].items():
                records = [r for r in raw if r['role'] == 'transfer' and r['release'] == 'E_pca' and r['target'] == task and r['selected']]
                assert len(records) == 1
                record = records[0]
                assert record['candidate_id'] == identity['candidate_id'] == selected[identity['selection_key']]
                for split, value in identity['scores'].items():
                    self.same(record[split]['log_loss'], value, 'original parent')
                for weight, split in rules['weights'].items():
                    self.parent[int(seed), weight, task] = record[split]['log_loss']
        self.counts['original_parent_task_split_scores'] = 60
        self.summary = self.read_json(self.out/'summary.json')
        assert self.summary['complete'] is True and self.summary['physical_system_count'] == 54
        assert self.summary['new_completed_systems'] == 36

    def load_scores(self):
        scores, candidates, priors = {}, {}, {}
        for row in self.rows('PER_SEED.csv'):
            if row['condition'] not in self.names or row['role'] not in ('utility', 'audit'):
                continue
            key = tuple(row[k] for k in ('seed', 'condition', 'role', 'view', 'target', 'audit_budget', 'scope', 'split'))
            assert key not in scores, 'Duplicate selected metric'
            assert row['present'] is True and row['selected_candidate'], 'Full report has missing selected system'
            scores[key] = row
            selection_key = key[:-1]
            if selection_key in candidates:
                assert candidates[selection_key] == row['selected_candidate'], 'Different validation/weighted candidate'
            candidates[selection_key] = row['selected_candidate']
            if row['role'] == 'audit':
                self.same(row['signed_prior_relative_gain'], subtract(row['prior_loss'], row['log_loss']), 'selected signed gain')
                prior_key = (row['seed'], row['target'], row['split'])
                if prior_key in priors:
                    self.same(row['prior_loss'], priors[prior_key], 'same target prior across releases/views/budgets/scopes')
                priors[prior_key] = row['prior_loss']
        self.points = {}
        for e, weight, scope, budget in itertools.product(self.entries, self.rules['weights'], self.rules['audit_scopes'], self.rules['audit_budgets']):
            split = 'development_evaluation'+('_person_weighted' if weight == 'person_weighted' else '')
            seed, name = e['seed'], e['condition']
            utility = {task: scores[seed, name, 'utility', VIEWS[task], task, None, 'utility', split]['log_loss'] for task in TASKS}
            attack, ids = {}, {}
            for role in AUDITS:
                view, task = role.split('/')
                row = scores[seed, name, 'audit', view, task, budget, scope, split]
                attack[role] = subtract(row['prior_loss'], row['log_loss'])
                ids[role] = row['selected_candidate']
            self.points[seed, name, weight, scope, budget] = {'utility': utility, 'gain': attack, 'ids': ids}
        self.counts['selected_score_rows'] = len(scores)
        self.counts['fixed_system_weight_scope_budget_points'] = len(self.points)

    def source(self, point, seed, weight):
        tasks = {}
        for task in SOURCE:
            loss, parent = point['utility'][task], self.parent[seed, weight, task]
            delta = subtract(loss, parent)
            tasks[task] = {'loss': loss if defined(loss) else None, 'parent_loss': parent,
                'loss_minus_parent': delta, 'excess_over_allowance': delta-.01 if delta is not None else None,
                'pass': loss <= parent+.01+ROUND if delta is not None else None}
        return {'tasks': tasks, 'pass': conjunction(r['pass'] for r in tasks.values())}

    def pair(self, seed, identity):
        interface, jb, lb, jname, lname, attr, weight, scope, budget, panel, delta = identity
        assert self.alias[seed, interface, 'J', jb] == jname
        assert self.alias[seed, interface, 'Iplus', lb] == lname
        j, local = (self.points[seed, name, weight, scope, budget] for name in (jname, lname))
        source_j, source_l = self.source(j, seed, weight), self.source(local, seed, weight)
        udiff = {task: subtract(j['utility'][task], local['utility'][task]) for task in TASKS}
        adiff = {role: subtract(j['gain'][role], local['gain'][role]) for role in AUDITS}
        panel_tasks = self.rules['panels'][panel]
        close = {task: abs(udiff[task]) <= delta+ROUND if udiff[task] is not None else None for task in panel_tasks}
        directional = {task: udiff[task] <= delta+ROUND if udiff[task] is not None else None for task in panel_tasks}
        missing = [label+'/utility/'+task for label, p in (('J', j), ('Iplus', local)) for task in TASKS
                   if task in set(SOURCE+tuple(panel_tasks)) and not defined(p['utility'][task])]
        missing.extend('parent/'+task for task in SOURCE if not defined(self.parent[seed, weight, task]))
        role = 'AB/'+attr
        missing.extend(label+'/gain/'+role for label, p in (('J', j), ('Iplus', local)) if not defined(p['gain'][role]))
        gain = adiff[role]
        strict = gain < -ROUND if gain is not None else None
        both = conjunction((source_j['pass'], source_l['pass']))
        result = {'source_J': source_j, 'source_Iplus': source_l, 'both_source_feasible': both,
            'utility_differences': udiff, 'audit_gain_differences': adiff, 'close_by_task': close,
            'directional_by_task': directional, 'utility_close': conjunction(close.values()),
            'utility_directional': conjunction(directional.values()), 'J_gain': j['gain'][role], 'Iplus_gain': local['gain'][role],
            'gain_difference': gain, 'strict_gain_improvement': strict, 'missing_required': missing,
            'shared_physical_anchor': jname == lname, 'J_selected_candidate': j['ids'][role],
            'Iplus_selected_candidate': local['ids'][role], 'exclusion_reasons': {}}
        for kind, values in (('close', close), ('directional', directional)):
            excluded = [label+'/source/'+task for label, src in (('J', source_j), ('Iplus', source_l))
                        for task in SOURCE if src['tasks'][task]['pass'] is False]
            excluded += ['utility/'+task for task in panel_tasks if values[task] is False]
            if strict is False:
                excluded.append('no_strict_'+attr+'_gain_improvement')
            excluded += ['missing/'+item for item in missing]
            passed = None if missing else conjunction((both, conjunction(values.values()), strict))
            result['qualifies_'+kind] = passed
            result['assessment_'+kind] = 'unassessable' if passed is None else 'qualifies' if passed else 'excluded'
            result['exclusion_reasons'][kind] = excluded
        return result

    def identities(self):
        for interface, jb, lb, attr, weight, scope, budget, panel, delta in itertools.product(
                self.rules['interfaces'], self.rules['beta_decimal_strings'], self.rules['beta_decimal_strings'],
                ('SEX', 'RAC1P'), self.rules['weights'], self.rules['audit_scopes'], self.rules['audit_budgets'],
                self.rules['panels'], self.rules['delta_values']):
            yield (interface, jb, lb, self.alias[0, interface, 'J', jb], self.alias[0, interface, 'Iplus', lb],
                   attr, weight, scope, budget, panel, delta)

    def cube(self):
        expected_ids = set(itertools.product(self.rules['seeds'], self.identities()))
        count = 0
        for row in self.rows('UTILITY_MATCHES.csv'):
            identity = tuple(row[k] for k in PAIR_ID)
            key = row['seed'], identity
            assert key in expected_ids, 'Unknown/duplicate pair identity '+str(key)
            expected_ids.remove(key)
            self.same(row, self.pair(*key), 'cube/'+str(key))
            count += 1
        assert not expected_ids and count == 57600
        self.counts['pair_rows'] = count

    def aggregates(self):
        expected_ids = set(self.identities())
        count = 0
        for row in self.rows('UTILITY_MATCHES_AGGREGATE.csv'):
            identity = tuple(row[k] for k in PAIR_ID)
            assert identity in expected_ids, 'Unknown/duplicate aggregate identity'
            expected_ids.remove(identity)
            by_seed = {str(seed): self.pair(seed, identity) for seed in self.rules['seeds']}
            both = {s: r['both_source_feasible'] for s, r in by_seed.items()}
            expected = {'expected_seeds': [0, 1, 2], 'present_seeds': [0, 1, 2],
                'both_source_feasible_per_seed': both, 'both_source_feasible_seed_count': sum(x is True for x in both.values())}
            for kind in ('close', 'directional'):
                values = {s: r['qualifies_'+kind] for s, r in by_seed.items()}
                eligible = {s: conjunction((r['both_source_feasible'], r['utility_'+kind])) for s, r in by_seed.items()}
                expected.update({kind+'_source_and_utility_eligible_per_seed': eligible,
                    kind+'_source_and_utility_eligible_seed_count': sum(x is True for x in eligible.values()),
                    kind+'_source_and_utility_assessable_seed_count': sum(x is not None for x in eligible.values()),
                    kind+'_qualifying_seed_count': sum(x is True for x in values.values()),
                    kind+'_assessable_seed_count': sum(x is not None for x in values.values()),
                    kind+'_per_seed': values, kind+'_all_seeds_qualify': conjunction(values.values())})
            gains = {s: r['gain_difference'] for s, r in by_seed.items()}
            mean, sd = mean_sd(tuple(gains.values()))
            expected.update(gain_difference_per_seed=gains, gain_difference_mean=mean, gain_difference_sd=sd)
            for field, components in (('utility_differences', TASKS), ('audit_gain_differences', AUDITS)):
                expected[field] = {}
                for component in components:
                    values = {s: r[field][component] for s, r in by_seed.items()}
                    mean, sd = mean_sd(tuple(values.values()))
                    expected[field][component] = {'per_seed': values, 'mean': mean, 'sample_sd': sd}
            self.same(row, expected, 'aggregate/'+str(identity))
            count += 1
        assert not expected_ids and count == 19200
        self.counts['fixed_pair_aggregate_rows'] = count

    def vector(self, seed, name, weight, scope, budget):
        p = self.points[seed, name, weight, scope, budget]
        return {'utility/'+k: v for k, v in p['utility'].items()} | {'gain/'+k: v for k, v in p['gain'].items()}

    def pareto(self):
        expected = {}
        for weight, scope, budget, domain, vector in itertools.product(self.rules['weights'], self.rules['audit_scopes'],
                self.rules['audit_budgets'], ('F', 'P', 'F_and_P'), self.rules['pareto']['vectors']):
            names = [name for name, interface in self.names.items() if domain == 'F_and_P' or interface == domain]
            assert len(names) == (18 if domain == 'F_and_P' else 9)
            components = self.rules['pareto']['vectors'][vector]
            for seed in (0, 1, 2, None):
                values = {}
                for name in names:
                    if seed is None:
                        vectors = [self.vector(s, name, weight, scope, budget) for s in (0, 1, 2)]
                        values[name] = {c: mean_sd(tuple(v[c] for v in vectors))[0] for c in components}
                    else:
                        values[name] = self.vector(seed, name, weight, scope, budget)
                complete = {name: all(defined(values[name][c]) for c in components) for name in names}
                for name in names:
                    missing = [c for c in components if not defined(values[name][c])]
                    witnesses = []
                    for other in names:
                        if name == other or not complete[name] or not complete[other]:
                            continue
                        deltas = [values[other][c]-values[name][c] for c in components]
                        if all(d <= ROUND for d in deltas) and any(d < -ROUND for d in deltas):
                            witnesses.append(other)
                    status = ('unassessable_point' if missing else 'dominated' if witnesses else
                              'nondominated' if all(complete.values()) else 'not_dominated_among_assessable_points')
                    key = (seed, 'per_seed' if seed is not None else 'three_seed_mean', domain, vector, weight, scope, budget, name)
                    expected[key] = {'components': components, 'values': {c: values[name][c] for c in components},
                        'missing_components': missing, 'dominating_conditions': witnesses,
                        'all_alternatives_complete': all(complete.values()), 'status': status}
        count = 0
        for row in self.rows('NONDOMINATED_POINTS.csv'):
            key = tuple(row[k] for k in ('seed', 'aggregation', 'domain', 'vector', 'weighting', 'scope', 'audit_budget', 'condition'))
            assert key in expected, 'Unknown/duplicate Pareto identity'
            self.same(row, expected.pop(key), 'Pareto/'+str(key))
            count += 1
        assert not expected and count == 22464
        self.counts['pareto_rows'] = count

    def vector_tradeoffs(self):
        identities = {identity for identity in self.identities() if identity[5] == 'SEX' and identity[9] == 'full_authorized' and identity[10] == 0}
        expected_ids = set(itertools.product((0, 1, 2), identities))
        count = 0
        for row in self.rows('VECTOR_TRADEOFF.csv'):
            identity = tuple({'attribute': 'SEX', 'panel': 'full_authorized', 'delta': 0}.get(k, row.get(k)) for k in PAIR_ID)
            key = row['seed'], identity
            assert key in expected_ids
            expected_ids.remove(key)
            pair = self.pair(*key)
            differences = {'utility/'+k: v for k, v in pair['utility_differences'].items()}
            differences.update({'gain/'+k: v for k, v in pair['audit_gain_differences'].items()})
            missing = [k for k, v in differences.items() if not defined(v)]
            worse = [k for k, v in differences.items() if defined(v) and v > ROUND]
            better = [k for k, v in differences.items() if defined(v) and v < -ROUND]
            self.same(row, {'utility_differences': pair['utility_differences'], 'audit_gain_differences': pair['audit_gain_differences'],
                'components_worse_for_J': worse, 'components_better_for_J': better, 'missing_components': missing,
                'J_dominates_Iplus_numeric': None if missing else not worse and bool(better),
                'Iplus_dominates_J_numeric': None if missing else not better and bool(worse),
                'roundoff_tolerance': ROUND, 'delta_used_for_dominance': False}, 'full-vector tradeoff/'+str(key))
            count += 1
        assert not expected_ids and count == 1800
        self.counts['full_vector_pair_rows'] = count

    def run(self):
        for stage in ('prepare', 'load_scores', 'cube', 'aggregates', 'pareto', 'vector_tradeoffs'):
            self.stage = stage
            getattr(self, stage)()
        self.same(self.summary['pair_cube_rows'], 57600, 'summary pair count')
        self.same(self.summary['fixed_pair_aggregate_rows'], 19200, 'summary aggregate count')
        self.same(self.summary['pareto_rows'], 22464, 'summary Pareto count')
        for path, initial_hash in self.inputs.items():
            assert digest(path) == initial_hash, 'Replay input changed: '+str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    report = (args.report or out/'FINAL_COMPARISON_REPLAY.json').resolve()
    if report.exists():
        raise FileExistsError('Preserve existing replay evidence; use a new --report path')
    start = time.perf_counter()
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Independent stdlib arithmetic from published PER_SEED and frozen original-parent scores; no reporter/helper imports, fitted models, raw people or fitting',
        'numeric_tolerance': REPLAY_ATOL, 'decision_roundoff': ROUND,
        'replay_script_sha256': digest(Path(__file__)), 'status': 'failed'}
    verifier = None
    error = None
    try:
        verifier = Replay(out)
        verifier.run()
        result['status'] = 'pass'
    except Exception as exc:
        error = exc
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                             'stage': getattr(verifier, 'stage', 'initialization')}
    finally:
        result['runtime_seconds'] = time.perf_counter()-start
        if verifier is not None:
            result.update(counts=verifier.counts, checked_values=verifier.checks,
                          max_absolute_scalar_error=verifier.max_error,
                          input_sha256={str(p.relative_to(ROOT)): h for p, h in verifier.inputs.items()})
        with report.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'counts', 'runtime_seconds', 'max_absolute_scalar_error')}))
    if error is not None:
        raise error


if __name__ == '__main__':
    main()

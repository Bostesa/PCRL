"""Independent stdlib source-guard arithmetic replay from compact selected scores.

No reporter/comparison helper, numerical library, fitted model or person data is
imported.  The earlier independent verifier supplies CSV/hash/comparison plumbing
only; all source-guard identities, matching, paired and criterion formulas below
start from PER_SEED and the original hash-bound PCA32 scores.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import gzip
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from scripts import verify_acs_coalition_strength_comparisons as plumbing

SOURCE, TASKS, VIEWS, AUDITS = plumbing.SOURCE, plumbing.TASKS, plumbing.VIEWS, plumbing.AUDITS
defined, conjunction, subtract, mean_sd = plumbing.defined, plumbing.conjunction, plumbing.subtract, plumbing.mean_sd
ROUND = 1e-12
ARMS = ('G_J', 'G_L025', 'G_L20', 'T', 'U_J', 'U_L025', 'U_L20', 'I')
NEW = ARMS[:4]
PAIR_FIELDS = ('interface', 'J_arm', 'local_arm', 'J_condition', 'local_condition', 'attribute',
               'evaluation_split', 'weighting', 'scope', 'audit_budget', 'panel', 'delta')
CONTRAST_FIELDS = ('comparison', 'interface', 'left_arm', 'right_arm', 'left', 'right', 'role', 'view',
                   'target', 'audit_budget', 'scope', 'evaluation_split', 'weighting', 'raw_score_split')


def legacy_joint(values):
    """Old descriptive margins retain known-failure precedence, unlike matching."""
    values = tuple(values)
    if any(v is False for v in values): return False
    if not values or any(v is None for v in values): return None
    return True


def statistics(values):
    values = tuple(values)
    complete = bool(values) and all(defined(v) for v in values)
    mean, sd = mean_sd(values) if complete and len(values) > 1 else (values[0], None) if complete else (None, None)
    return {'n_seeds': len(values), 'n_defined': sum(defined(v) for v in values), 'complete': complete,
            'mean': mean, 'sample_sd': sd}


def source_floor(utility, parents, *, old_margin=False):
    rows = {}
    for task in SOURCE:
        value, parent = utility.get(task), parents.get(task)
        delta = subtract(value, parent)
        passed = value <= parent+.01+(0. if old_margin else ROUND) if delta is not None else None
        rows[task] = ({'parent_log_loss': parent, 'erased_log_loss': value, 'difference': delta,
                       'maximum_increase': .01, 'pass': passed} if old_margin else
            {'loss': value if defined(value) else None, 'parent_loss': parent if defined(parent) else None,
             'loss_minus_parent': delta, 'excess_over_allowance': delta-.01 if delta is not None else None, 'pass': passed})
    return {'tasks': rows, 'pass': (legacy_joint if old_margin else conjunction)(r['pass'] for r in rows.values())}


def independent_pair(j, local, parent, panel_tasks, delta, attribute, local_arm):
    sj, sl = source_floor(j['utility'], parent), source_floor(local['utility'], parent)
    utility = {task: subtract(j['utility'].get(task), local['utility'].get(task)) for task in TASKS}
    attacks = {role: subtract(j['gain'].get(role), local['gain'].get(role)) for role in AUDITS}
    close = {task: abs(utility[task]) <= delta+ROUND if utility[task] is not None else None for task in panel_tasks}
    directional = {task: utility[task] <= delta+ROUND if utility[task] is not None else None for task in panel_tasks}
    missing = [label+'/utility/'+task for label, point in (('G_J', j), (local_arm, local)) for task in TASKS
               if task in set(SOURCE+tuple(panel_tasks)) and not defined(point['utility'].get(task))]
    missing += ['parent/'+task for task in SOURCE if not defined(parent.get(task))]
    role = 'AB/'+attribute
    missing += [label+'/gain/'+role for label, point in (('G_J', j), (local_arm, local)) if not defined(point['gain'].get(role))]
    gain = attacks[role]
    strict = gain < -ROUND if gain is not None else None
    both = conjunction((sj['pass'], sl['pass']))
    result = {'source_guarded_J': sj, 'source_guarded_local': sl, 'both_source_feasible': both,
        'utility_differences': utility, 'audit_gain_differences': attacks, 'close_by_task': close,
        'directional_by_task': directional, 'utility_close': conjunction(close.values()),
        'utility_directional': conjunction(directional.values()), 'guarded_J_gain': j['gain'].get(role),
        'guarded_local_gain': local['gain'].get(role), 'gain_difference': gain,
        'strict_gain_improvement': strict, 'missing_required': missing, 'exclusion_reasons': {}}
    for kind, tests in (('close', close), ('directional', directional)):
        reasons = [label+'/source/'+task for label, src in (('G_J', sj), (local_arm, sl)) for task in SOURCE if src['tasks'][task]['pass'] is False]
        reasons += ['utility/'+task for task in panel_tasks if tests[task] is False]
        if strict is False: reasons.append('no_strict_'+attribute+'_gain_improvement')
        reasons += ['missing/'+item for item in missing]
        passed = None if missing else conjunction((both, conjunction(tests.values()), strict))
        result.update({kind_key: value for kind_key, value in (
            ('qualifies_'+kind, passed), ('assessment_'+kind, 'unassessable' if passed is None else 'qualifies' if passed else 'excluded'))})
        result['exclusion_reasons'][kind] = reasons
    return result


def independent_aggregate(by_seed):
    both = {s: by_seed[s]['both_source_feasible'] for s in ('0', '1', '2')}
    result = {'expected_seeds': [0, 1, 2], 'present_seeds': [0, 1, 2],
        'both_source_feasible_per_seed': both, 'both_source_feasible_seed_count': sum(x is True for x in both.values())}
    for kind in ('close', 'directional'):
        eligible = {s: conjunction((r['both_source_feasible'], r['utility_'+kind])) for s, r in by_seed.items()}
        values = {s: r['qualifies_'+kind] for s, r in by_seed.items()}
        result.update({kind+'_source_and_utility_eligible_per_seed': eligible,
            kind+'_source_and_utility_eligible_seed_count': sum(x is True for x in eligible.values()),
            kind+'_source_and_utility_assessable_seed_count': sum(x is not None for x in eligible.values()),
            kind+'_qualifying_seed_count': sum(x is True for x in values.values()),
            kind+'_assessable_seed_count': sum(x is not None for x in values.values()),
            kind+'_per_seed': values, kind+'_all_seeds_qualify': conjunction(values.values())})
    gain = {s: r['gain_difference'] for s, r in by_seed.items()}
    mean, sd = mean_sd(tuple(gain.values()))
    result.update(gain_difference_per_seed=gain, gain_difference_mean=mean, gain_difference_sd=sd)
    for field, components in (('utility_differences', TASKS), ('audit_gain_differences', AUDITS)):
        result[field] = {}
        for component in components:
            values = {s: r[field][component] for s, r in by_seed.items()}
            mean, sd = mean_sd(tuple(values.values()))
            result[field][component] = {'per_seed': values, 'mean': mean, 'sample_sd': sd}
    return result


def ordinary_pareto(values, components):
    complete = {name: all(defined(vector.get(c)) for c in components) for name, vector in values.items()}
    result = {}
    for name, value in values.items():
        missing = [c for c in components if not defined(value.get(c))]
        witnesses = [other for other, vector in values.items() if other != name and complete[name] and complete[other]
            and all(vector[c] <= value[c]+ROUND for c in components)
            and any(vector[c] < value[c]-ROUND for c in components)]
        result[name] = {'components': list(components), 'values': {c: value.get(c) if defined(value.get(c)) else None for c in components},
            'missing_components': missing, 'dominating_conditions': witnesses, 'all_alternatives_complete': all(complete.values()),
            'status': 'unassessable_point' if missing else 'dominated' if witnesses else 'nondominated' if all(complete.values()) else 'not_dominated_among_assessable_points'}
    return result


def headroom(parent, method, reference):
    valid = all(defined(v) for v in (parent, method, reference))
    original = reference-parent if valid else None
    remaining = reference-method if valid else None
    positive = bool(valid and original > 0)
    return {'parent_log_loss': parent, 'erased_log_loss': method, 'reference_log_loss': reference,
        'erased_minus_parent_log_loss': method-parent if valid else None,
        'parent_headroom': original, 'erased_headroom': remaining, 'positive_headroom': positive,
        'retained_fraction': remaining/original if positive else None,
        'headroom_reduction_fraction': 1-remaining/original if positive else None,
        'ratio_defined': positive, 'undefined_reason': None if positive else 'nonpositive_parent_headroom' if valid else 'undefined_loss'}


def residence_margin(parent, method, bank_a, bank_b):
    reference = min(bank_a, bank_b) if defined(bank_a) and defined(bank_b) else None
    result = headroom(parent, method, reference)
    result.update(reference='minimum unprotected B/C residence log loss on this seed and split',
        minimum_retained_fraction=.5, **{'pass': method <= (reference+parent)/2 if result['ratio_defined'] else None})
    return result


def halving_margin(parent, method, prior, covered):
    result = headroom(parent, method, prior)
    result['parent_attribute_gain'] = result.pop('parent_headroom')
    result['erased_attribute_gain'] = result.pop('erased_headroom')
    result.update(maximum_gain_fraction_remaining=.5, coverage_complete=covered,
        numeric_halving_inequality=method >= (prior+parent)/2 if result['ratio_defined'] else None)
    if covered:
        result['pass'] = result['numeric_halving_inequality']
    else:
        result.update(ratio_defined=False, retained_fraction=None, headroom_reduction_fraction=None,
                      undefined_reason='coverage_limited', **{'pass': None})
    return result


class Replay(plumbing.Replay):
    def json_export(self, name):
        path = self.out/name
        if name in self.compact:
            record = self.compact[name]; packed = self.out/(name+'.gz')
            assert self.bind(packed) == record['gzip_sha256']
            data = gzip.decompress(packed.read_bytes())
            assert plumbing.hashlib.sha256(data).hexdigest() == record['plain_sha256']
            if path.exists(): assert self.bind(path) == record['plain_sha256']
            return json.loads(data)
        return self.read_json(path)
    def prepare(self):
        self.rules = rules = self.read_json(self.out/'comparison_rules.json')
        freeze = self.read_json(self.out/'protocol_freeze.json')
        assert plumbing.digest(self.out/'comparison_rules.json') == freeze['scientific_and_protocol_sha256'][str((self.out/'comparison_rules.json').relative_to(ROOT))]
        reporter = ROOT/'scripts/report_acs_source_guard.py'; self.bind(reporter)
        nodes = ast.parse(reporter.read_text()).body
        frozen = freeze['comparison_implementation']
        functions = {node.name: node for node in nodes if isinstance(node, ast.FunctionDef)}
        assignments = {target.id: node for node in nodes if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
        for group, available in (('functions_ast_sha256', functions), ('constants_ast_sha256', assignments)):
            for name, wanted in frozen[group].items():
                actual = plumbing.hashlib.sha256(ast.dump(available[name], include_attributes=False).encode()).hexdigest()
                assert actual == wanted, 'Prospectively frozen comparison AST changed: '+name
        for name, wanted in frozen['historical_helpers_sha256'].items():
            assert self.bind(ROOT/name) == wanted, 'Historical reporting helper changed: '+name
        assert rules['seeds'] == [0, 1, 2] and tuple(rules['method_arms']) == ARMS
        assert rules['fixed_method_pairs'] == [['G_J', 'G_L025'], ['G_J', 'G_L20']]
        assert rules['comparison_roundoff_tolerance'] == rules['pareto']['roundoff_tolerance'] == ROUND
        assert rules['source_feasibility']['maximum_per_task_loss_increase'] == .01
        assert rules['pareto']['delta_used_for_dominance'] is False
        self.entries = self.read_json(self.out/'REUSE_MANIFEST.json')['systems']
        assert plumbing.digest(self.out/'REUSE_MANIFEST.json') == freeze['reuse_manifest_sha256']
        assert len(self.entries) == len({(e['seed'], e['condition']) for e in self.entries}) == 48
        self.alias = {(e['seed'], e['interface'], e['arm']): e['condition'] for e in self.entries}
        assert set(self.alias) == set(itertools.product((0, 1, 2), ('F', 'P'), ARMS))
        assert all(e['reused'] == (e['arm'] not in NEW) for e in self.entries)
        self.names = {e['condition']: e['interface'] for e in self.entries if e['seed'] == 0}
        assert len(self.names) == 16
        self.parent = {}
        for seed, parent in rules['original_parent_metric_identity'].items():
            raw = self.read_json(ROOT/parent['metrics_path'])['raw_metrics']
            choices = self.read_json(ROOT/parent['selection_path'])['head_selections']
            assert plumbing.digest(ROOT/parent['metrics_path']) == parent['metrics_sha256']
            assert plumbing.digest(ROOT/parent['selection_path']) == parent['selection_sha256']
            for task, identity in parent['tasks'].items():
                selected = [r for r in raw if r['role'] == 'transfer' and r['release'] == 'E_pca' and r['target'] == task and r['selected']]
                assert len(selected) == 1
                selected = selected[0]
                assert selected['candidate_id'] == choices[identity['selection_key']] == identity['candidate_id']
                for split, value in identity['scores'].items(): self.same(selected[split]['log_loss'], value, 'original PCA32')
                for split, weight in itertools.product(rules['evaluation_splits'], rules['weights']):
                    self.parent[int(seed), split, weight, task] = selected[rules['evaluation_splits'][split][weight]]['log_loss']
        self.counts['original_parent_task_split_scores'] = 60
        self.summary = self.read_json(self.out/'summary.json')
        for name, wanted in self.summary['report_source_sha256'].items():
            assert self.bind(ROOT/name) == wanted, 'Published render no longer matches reporting source: '+name
        assert self.summary['physical_systems'] == 48 and self.summary['new_interface_systems'] == 24

    def metric(self, seed, name, role, view, target, budget, scope, split, weight):
        raw = self.rules['evaluation_splits'][split][weight]
        if role == 'native': raw = raw.replace('validation', 'source_validation')
        display = raw.replace('test', 'development_evaluation')
        return self.scores[seed, name, role, view, target, budget, scope, display]

    def load_scores(self):
        self.scores = {}; candidates = {}; priors = {}
        for row in self.rows('PER_SEED.csv'):
            if row['condition'] not in (*self.names, 'E'): continue
            key = tuple(row[k] for k in ('seed', 'condition', 'role', 'view', 'target', 'audit_budget', 'scope', 'split'))
            assert key not in self.scores, 'Duplicate selected-score identity'
            self.scores[key] = row
            if key[:-1] in candidates: assert candidates[key[:-1]] == row['selected_candidate'], 'Different weighted/validation winner'
            candidates[key[:-1]] = row['selected_candidate']
            if row['role'] == 'audit':
                self.same(row['signed_prior_relative_gain'], subtract(row['prior_loss'], row['log_loss']), 'selected signed gain')
                prior_key = row['seed'], row['target'], row['split']
                if prior_key in priors: self.same(row['prior_loss'], priors[prior_key], 'fixed target prior')
                priors[prior_key] = row['prior_loss']
        self.points = {}
        for e, split, weight, scope, budget in itertools.product(self.entries, self.rules['evaluation_splits'], self.rules['weights'], self.rules['audit_scopes'], self.rules['audit_budgets']):
            seed, name = e['seed'], e['condition']
            utility = {task: self.metric(seed, name, 'utility', VIEWS[task], task, None, 'utility', split, weight)['log_loss'] for task in TASKS}
            gain, ids = {}, {}
            for role in AUDITS:
                view, task = role.split('/')
                row = self.metric(seed, name, 'audit', view, task, budget, scope, split, weight)
                gain[role] = subtract(row['prior_loss'], row['log_loss']); ids[role] = row['selected_candidate']
            self.points[seed, name, split, weight, scope, budget] = {'utility': utility, 'gain': gain, 'ids': ids}
        self.counts['selected_score_rows'] = len(self.scores)
        self.counts['fixed_system_split_weight_scope_budget_points'] = len(self.points)
        assert len(self.points) == 1152

    def identities(self):
        for interface, pair, attribute, split, weight, scope, budget, panel, delta in itertools.product(
                self.rules['interfaces'], self.rules['fixed_method_pairs'], ('SEX', 'RAC1P'), self.rules['evaluation_splits'],
                self.rules['weights'], self.rules['audit_scopes'], self.rules['audit_budgets'], self.rules['panels'], self.rules['delta_values']):
            j, local = pair
            yield (interface, j, local, self.alias[0, interface, j], self.alias[0, interface, local], attribute,
                   split, weight, scope, budget, panel, delta)

    def pair(self, seed, identity):
        interface, jarm, larm, jname, lname, attr, split, weight, scope, budget, panel, delta = identity
        assert self.alias[seed, interface, jarm] == jname and self.alias[seed, interface, larm] == lname
        j, local = [self.points[seed, name, split, weight, scope, budget] for name in (jname, lname)]
        parent = {task: self.parent[seed, split, weight, task] for task in SOURCE}
        result = independent_pair(j, local, parent, self.rules['panels'][panel], delta, attr, larm)
        result.update(J_selected_candidate=j['ids']['AB/'+attr], local_selected_candidate=local['ids']['AB/'+attr])
        return result

    def cube(self):
        expected = set(itertools.product((0, 1, 2), self.identities())); count = 0
        for row in self.rows('UTILITY_MATCHES.csv'):
            key = row['seed'], tuple(row[k] for k in PAIR_FIELDS)
            assert key in expected, 'Unknown/duplicate fixed pair'
            expected.remove(key); self.same(row, self.pair(*key), 'pair/'+str(key)); count += 1
        assert not expected and count == 9216
        self.counts['pair_rows'] = count

    def aggregates(self):
        expected = set(self.identities()); count = 0
        for row in self.rows('UTILITY_MATCHES_AGGREGATE.csv'):
            identity = tuple(row[k] for k in PAIR_FIELDS)
            assert identity in expected, 'Unknown/duplicate fixed-pair aggregate'
            expected.remove(identity)
            self.same(row, independent_aggregate({str(s): self.pair(s, identity) for s in (0, 1, 2)}), 'pair aggregate/'+str(identity))
            count += 1
        assert not expected and count == 3072
        self.counts['fixed_pair_aggregate_rows'] = count

    def source_feasibility(self):
        expected = {(e['seed'], e['condition'], split, weight): e for e, split, weight in itertools.product(self.entries, self.rules['evaluation_splits'], self.rules['weights'])}
        for row in self.read_json(self.out/'SOURCE_FEASIBILITY.json'):
            key = tuple(row[k] for k in ('seed', 'condition', 'evaluation_split', 'weighting'))
            assert key in expected
            entry = expected.pop(key); seed, name, split, weight = key
            self.same(row, {k: entry[k] for k in ('interface', 'arm', 'reused')}, 'source identity')
            self.same(row['source_feasibility'], source_floor(self.points[(*key, 'expanded_catchup', 360)]['utility'],
                {t: self.parent[seed, split, weight, t] for t in SOURCE}), 'source floor/'+str(key))
        assert not expected
        self.counts['source_feasibility_rows'] = 192

    def vector(self, seed, name, split, weight, scope, budget):
        p = self.points[seed, name, split, weight, scope, budget]
        return {'utility/'+k: v for k, v in p['utility'].items()} | {'gain/'+k: v for k, v in p['gain'].items()}

    def pareto(self):
        expected = {}
        for split, weight, scope, budget, domain, vector in itertools.product(self.rules['evaluation_splits'], self.rules['weights'],
                self.rules['audit_scopes'], self.rules['audit_budgets'], ('F', 'P', 'F_and_P'), self.rules['pareto']['vectors']):
            names = [name for name, interface in self.names.items() if domain == 'F_and_P' or interface == domain]
            assert len(names) == (16 if domain == 'F_and_P' else 8)
            components = self.rules['pareto']['vectors'][vector]
            for seed in (0, 1, 2, None):
                values = {}
                for name in names:
                    if seed is None:
                        points = [self.vector(s, name, split, weight, scope, budget) for s in (0, 1, 2)]
                        values[name] = {c: mean_sd(tuple(p[c] for p in points))[0] for c in components}
                    else: values[name] = self.vector(seed, name, split, weight, scope, budget)
                for name, row in ordinary_pareto(values, components).items():
                    expected[seed, 'per_seed' if seed is not None else 'three_seed_mean', split, weight, scope, budget, domain, vector, name] = row
        count = 0
        for row in self.rows('NONDOMINATED_POINTS.csv'):
            key = tuple(row[k] for k in ('seed', 'aggregation', 'evaluation_split', 'weighting', 'scope', 'audit_budget', 'domain', 'vector', 'condition'))
            assert key in expected, 'Unknown/duplicate Pareto identity'
            self.same(row, expected.pop(key), 'Pareto/'+str(key)); count += 1
        assert not expected and count == 39936
        self.counts['pareto_rows'] = count

    def vector_tradeoffs(self):
        ids = {identity for identity in self.identities() if identity[5] == 'SEX' and identity[10] == 'full_authorized' and identity[11] == 0}
        expected = set(itertools.product((0, 1, 2), ids)); count = 0
        for row in self.rows('VECTOR_TRADEOFF.csv'):
            identity = tuple({'attribute': 'SEX', 'panel': 'full_authorized', 'delta': 0}.get(k, row.get(k)) for k in PAIR_FIELDS)
            key = row['seed'], identity
            assert key in expected; expected.remove(key)
            pair = self.pair(*key)
            components = {'utility/'+k: v for k, v in pair['utility_differences'].items()} | {'gain/'+k: v for k, v in pair['audit_gain_differences'].items()}
            missing = [k for k, v in components.items() if not defined(v)]
            worse = [k for k, v in components.items() if defined(v) and v > ROUND]
            better = [k for k, v in components.items() if defined(v) and v < -ROUND]
            self.same(row, {'utility_differences': pair['utility_differences'], 'audit_gain_differences': pair['audit_gain_differences'],
                'components_worse_for_guarded_J': worse, 'components_better_for_guarded_J': better, 'missing_components': missing,
                'guarded_J_dominates_local_numeric': None if missing else not worse and bool(better),
                'local_dominates_guarded_J_numeric': None if missing else not better and bool(worse),
                'comparison_roundoff_tolerance': ROUND, 'delta_used_for_dominance': False}, 'full vector/'+str(key))
            count += 1
        assert not expected and count == 288
        self.counts['full_vector_pair_rows'] = count

    def endpoints(self):
        for seed, interface in itertools.product((0, 1, 2), ('F', 'P')):
            pairs = [('guarded_minus_unguarded', g, u) for g, u in self.rules['guarded_unguarded_pairs']]
            pairs += [('guarded_minus_T', g, 'T') for g, _ in self.rules['guarded_unguarded_pairs']]
            pairs += [('guarded_J_minus_guarded_local', g, local) for g, local in self.rules['fixed_method_pairs']]
            pairs += [('T_minus_I', 'T', 'I')]
            for comparison, left, right in pairs:
                yield (seed, comparison, interface, left, right, self.alias[seed, interface, left], self.alias[seed, interface, right])
        for seed, arm in itertools.product((0, 1, 2), NEW):
            yield (seed, 'F_minus_P', 'F_minus_P', arm, arm, self.alias[seed, 'F', arm], self.alias[seed, 'P', arm])

    def contrast_expected(self):
        for seed, comparison, interface, left_arm, right_arm, left, right in self.endpoints():
            for role in ('utility', 'native', 'audit'):
                targets = [(r.split('/')) for r in AUDITS] if role == 'audit' else [(VIEWS[t], t) for t in SOURCE if role == 'native'] if role == 'native' else [(VIEWS[t], t) for t in TASKS]
                for view, target in targets:
                    for budget, scope in (itertools.product(self.rules['audit_budgets'], self.rules['audit_scopes']) if role == 'audit' else [(None, role)]):
                        for split, weight in itertools.product(self.rules['evaluation_splits'], self.rules['weights']):
                            raw = self.rules['evaluation_splits'][split][weight]
                            if role == 'native': raw = raw.replace('validation', 'source_validation')
                            a, b = [self.metric(seed, name, role, view, target, budget, scope, split, weight) for name in (left, right)]
                            delta = subtract(a['log_loss'], b['log_loss'])
                            identity = (comparison, interface, left_arm, right_arm, left, right, role, view, target, budget, scope, split, weight, raw)
                            yield seed, identity, {'left_loss': a['log_loss'], 'right_loss': b['log_loss'], 'left_minus_right': delta,
                                'signed_gain_left_minus_right': -delta if role == 'audit' and delta is not None else None,
                                'present': a['present'] and b['present']}

    def paired(self):
        expected = {(seed, identity): row for seed, identity, row in self.contrast_expected()}; aggregates = {}
        for (seed, identity), row in expected.items(): aggregates.setdefault(identity, {})[str(seed)] = row
        count = 0
        for row in self.rows('PAIRED.csv'):
            key = row['seed'], tuple(row[k] for k in CONTRAST_FIELDS)
            assert key in expected, 'Unknown/duplicate paired identity'
            self.same(row, expected.pop(key), 'paired/'+str(key)); count += 1
        assert not expected and count == 19536
        self.counts['paired_rows'] = count; count = 0
        for row in self.rows('PAIRED_AGGREGATE.csv'):
            key = tuple(row[k] for k in CONTRAST_FIELDS)
            assert key in aggregates, 'Unknown/duplicate paired aggregate'
            values = aggregates.pop(key)
            strict = statistics(values[str(seed)]['left_minus_right'] for seed in (0, 1, 2))
            present = [v['left_minus_right'] for v in values.values() if v['present']]
            observed = statistics(present)
            result = {**strict, 'observed_mean': observed['mean'], 'observed_sample_sd': observed['sample_sd'],
                'n_present': len(present), 'present_scores_defined': observed['complete'], 'partial_seed_summary': len(present) < 3,
                'signed_gain_difference_mean': -strict['mean'] if key[6] == 'audit' and strict['mean'] is not None else None}
            self.same(row, result, 'paired aggregate/'+str(key)); count += 1
        assert not aggregates and count == 6512
        self.counts['paired_aggregate_rows'] = count

    def criteria(self):
        context = {}
        for row in self.rows('CONTEXTUAL.csv'):
            key = tuple(row[k] for k in ('seed', 'historical_release', 'role', 'target', 'report_audit_budget', 'original_scope', 'split'))
            assert key not in context
            context[key] = row['log_loss']
        def dimensions(display):
            weight = 'person_weighted' if display.endswith('_person_weighted') else 'unweighted'
            return display.removesuffix('_person_weighted'), weight
        def utility(seed, name, display):
            split, weight = dimensions(display)
            return {t: self.metric(seed, name, 'utility', VIEWS[t], t, None, 'utility', split, weight)['log_loss'] for t in TASKS}
        def original(seed, display):
            split, weight = dimensions(display)
            return {t: self.parent[seed, split, weight, t] for t in TASKS}
        def residence(seed, name, display):
            banks = [context[seed, bank, 'transfer', 'same_residence', None, 'primary', display] for bank in ('B_rich_bank', 'C_tree_bank')]
            return residence_margin(original(seed, display)['same_residence'], utility(seed, name, display)['same_residence'], *banks)
        criteria = self.json_export('criteria.json')
        expected_margins = {'source_log_loss_increase': .01, 'residential_headroom_retained': .5,
            'attribute_gain_remaining': .5, 'feature_bank_utility_advantage': .01, 'feature_bank_attribute_gain_increase': .005}
        self.same(criteria['margins'], expected_margins, 'unchanged margins')
        assert criteria['fixed_parent'] == 'E_pca' and criteria['parent_actual_audit_budget'] == 120
        assert criteria['opposing_task_threshold'] is None and criteria['all_target_certificate'] is False
        expected_utilities = {(seed, name, split) for seed, name, split in itertools.product((0, 1, 2), (*self.names, 'E'),
            ('validation', 'validation_person_weighted', 'development_evaluation', 'development_evaluation_person_weighted'))}
        count = 0
        for row in criteria['per_seed_split_utility']:
            key = row['seed'], row['condition'], row['split']; assert key in expected_utilities; expected_utilities.remove(key)
            u, p = utility(*key), original(key[0], key[2])
            self.same(row, {'fixed_parent': 'E_pca', 'source_preservation': source_floor(u, p, old_margin=True),
                'residential_retention': residence(*key), 'task_minus_pca32': {t: subtract(u[t], p[t]) for t in TASKS}}, 'utility criteria/'+str(key))
            count += 1
        assert not expected_utilities and count == 204
        self.counts['utility_criteria_rows'] = count
        expected_policies = {(seed, name, view, budget, scope, split) for seed, name, view, budget, scope, split in itertools.product(
            (0, 1, 2), (*self.names, 'E'), ('A', 'B', 'AB'), (120, 360), self.rules['audit_scopes'],
            ('validation', 'validation_person_weighted', 'development_evaluation', 'development_evaluation_person_weighted'))}
        count = 0
        for row in criteria['per_seed_view_budget_scope_policy']:
            seed, name, view, budget, scope, display = key = tuple(row[k] for k in ('seed', 'condition', 'view', 'audit_budget', 'scope', 'split'))
            assert key in expected_policies; expected_policies.remove(key)
            u, p = utility(seed, name, display), original(seed, display)
            source, residential = source_floor(u, p, old_margin=True), residence(seed, name, display)
            split, weight = dimensions(display); attributes = {}
            for attr in ('SEX', 'RAC1P'):
                selected = self.metric(seed, name, 'audit', view, attr, budget, scope, split, weight)
                parent = context[seed, 'E_pca', 'audit', attr, budget, 'primary', display]
                # Category coverage is independently checked by model/score
                # replay; this arithmetic uses its reported boolean explicitly.
                covered = row['attribute_halving'][attr]['coverage_complete']
                assert isinstance(covered, bool)
                if attr == 'RAC1P': assert covered is False
                attributes[attr] = halving_margin(parent, selected['log_loss'], selected['prior_loss'], covered)
            self.same(row, {'utility_differences': {t: subtract(u[t], p[t]) for t in TASKS},
                'source_preservation': source, 'residential_retention': residential, 'attribute_halving': attributes,
                'commute_margin': None, 'coverage_complete': all(a['coverage_complete'] for a in attributes.values()),
                'joint_pass': legacy_joint([source['pass'], residential['pass'], *[a['pass'] for a in attributes.values()]])}, 'policy criteria/'+str(key))
            count += 1
        assert not expected_policies and count == 3672
        self.counts['parent_policy_rows'] = count
        feature = self.json_export('feature_capability.json')
        self.same(feature['margins'], expected_margins, 'feature margins'); assert feature['tasks_decided_separately'] is True
        expected_features = set()
        for seed, arm, task, budget, scope, display in itertools.product((0, 1, 2), ARMS, ('same_residence', 'commute_over20'), (120, 360),
                self.rules['audit_scopes'], ('validation', 'validation_person_weighted', 'development_evaluation', 'development_evaluation_person_weighted')):
            for right, view in itertools.product((self.alias[seed, 'P', arm], 'E'), (VIEWS[task], 'AB')):
                expected_features.add((seed, arm, self.alias[seed, 'F', arm], right, task, view, budget, scope, display))
        count = 0
        for row in feature['per_seed']:
            key = tuple(row[k] for k in ('seed', 'arm', 'left', 'comparator', 'task', 'sensitive_access', 'audit_budget', 'scope', 'split'))
            assert key in expected_features; expected_features.remove(key)
            seed, arm, left, right, task, view, budget, scope, display = key
            split, weight = dimensions(display)
            left_loss, right_loss = utility(seed, left, display)[task], utility(seed, right, display)[task]
            delta = subtract(left_loss, right_loss)
            u = {'feature_log_loss': left_loss, 'bank_log_loss': right_loss, 'feature_minus_bank': delta,
                 'minimum_advantage': .01, 'pass': left_loss <= right_loss-.01 if delta is not None else None}
            attrs = {}
            for attr in ('SEX', 'RAC1P'):
                a, b = [self.metric(seed, name, 'audit', view, attr, budget, scope, split, weight) for name in (left, right)]
                valid = all(defined(v) for v in (a['log_loss'], b['log_loss'], a['prior_loss']))
                covered = row['attributes'][attr]['coverage_complete']; assert isinstance(covered, bool)
                if attr == 'RAC1P': assert covered is False
                ga, gb = (a['prior_loss']-a['log_loss'], a['prior_loss']-b['log_loss']) if valid else (None, None)
                numeric = a['log_loss'] >= b['log_loss']-.005 if valid else None
                attrs[attr] = {'feature_gain': ga, 'bank_gain': gb, 'feature_minus_bank_gain': subtract(ga, gb),
                    'maximum_extra_gain': .005, 'coverage_complete': covered, 'numeric_inequality': numeric,
                    'pass': numeric if covered else None, 'undefined_reason': 'coverage_limited' if not covered else None if valid else 'undefined_loss'}
            self.same(row, {'utility': u, 'attributes': attrs, 'coverage_complete': all(a['coverage_complete'] for a in attrs.values()),
                'joint_pass': legacy_joint([u['pass'], *[a['pass'] for a in attrs.values()]]),
                'numeric_joint_inequalities': legacy_joint([u['pass'], *[a['numeric_inequality'] for a in attrs.values()]]),
                'separate_reserved_task_decision': True}, 'feature criteria/'+str(key))
            count += 1
        assert not expected_features and count == 4608
        self.counts['feature_task_criteria_rows'] = count

    def run(self):
        for stage in ('prepare', 'load_scores', 'source_feasibility', 'cube', 'aggregates', 'paired', 'pareto', 'vector_tradeoffs', 'criteria'):
            self.stage = stage; getattr(self, stage)()
        for key, count in (('pair_cube_rows', 9216), ('fixed_pair_aggregate_rows', 3072), ('pareto_rows', 39936), ('vector_rows', 288)):
            self.same(self.summary[key], count, 'summary/'+key)
        for path, value in self.inputs.items(): assert plumbing.digest(path) == value, 'Replay input changed: '+str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    report = (args.report or out/'FINAL_COMPARISON_REPLAY.json').resolve()
    if report.exists(): raise FileExistsError('Preserve earlier replay; choose a fresh --report')
    started = time.perf_counter(); verifier = None; failure = None
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'failed',
        'scope': 'Independent stdlib from PER_SEED and original selected PCA32, both evaluation pools and weightings; no reporter/comparison helper imports or fitting',
        'replay_script_sha256': plumbing.digest(Path(__file__)),
        'independent_csv_hash_plumbing_sha256': plumbing.digest(Path(plumbing.__file__)),
        'scalar_replay_tolerance': plumbing.REPLAY_ATOL, 'decision_roundoff': ROUND}
    try:
        verifier = Replay(out); verifier.run(); result['status'] = 'pass'
    except Exception as error:
        failure = error
        result['failure'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'stage': getattr(verifier, 'stage', 'initialization')}
    finally:
        result['runtime_seconds'] = time.perf_counter()-started
        if verifier is not None:
            result.update(counts=verifier.counts, checked_values=verifier.checks, max_absolute_scalar_error=verifier.max_error,
                          input_sha256={str(p.relative_to(ROOT)): h for p, h in verifier.inputs.items()})
        with report.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'counts', 'runtime_seconds', 'max_absolute_scalar_error')}))
    if failure is not None: raise failure


if __name__ == '__main__': main()

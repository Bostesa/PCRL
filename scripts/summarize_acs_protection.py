"""Saved ACS protection-feasibility comparisons; no fitting or test selection.

The pure helpers implement the provisional, empirical margins exactly. A
missing score, nonpositive reference headroom, or inadequate class coverage is
undefined rather than a pass. Evaluation is run independently for each split.
"""
from __future__ import annotations

import math
from numbers import Real


UTILITY_TASKS = ('same_residence', 'commute_over20', 'income_binary',
                 'civilian_at_work', 'public_coverage')
SOURCE_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
ATTRIBUTES = ('SEX', 'RAC1P')
PARENTS = ('A_binary_bank', 'B_rich_bank', 'C_tree_bank', 'D_compressed', 'E_pca')
MARGINS = {'source_log_loss_increase': .01, 'residential_headroom_retained': .5,
           'attribute_gain_remaining': .5, 'feature_bank_utility_advantage': .01,
           'feature_bank_attribute_gain_increase': .005}


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def joint_status(checks):
    """Known failure dominates; incomplete evidence cannot produce a pass."""
    checks = list(checks)
    if any(value is False for value in checks):
        return False
    if not checks or any(value is None for value in checks):
        return None
    if not all(value is True for value in checks):
        raise ValueError('Policy checks must be True, False, or None')
    return True


def loss_headroom(parent_loss, erased_loss, reference_loss):
    """Signed loss changes, with ratios only for positive parent headroom."""
    defined = all(finite(v) for v in (parent_loss, erased_loss, reference_loss))
    headroom = reference_loss - parent_loss if defined else None
    remaining = reference_loss - erased_loss if defined else None
    positive = bool(defined and headroom > 0)
    return {'parent_log_loss': parent_loss, 'erased_log_loss': erased_loss,
            'reference_log_loss': reference_loss,
            'erased_minus_parent_log_loss': erased_loss - parent_loss if defined else None,
            'parent_headroom': headroom, 'erased_headroom': remaining,
            'positive_headroom': positive,
            'retained_fraction': remaining / headroom if positive else None,
            'headroom_reduction_fraction': 1 - remaining / headroom if positive else None,
            'ratio_defined': positive,
            'undefined_reason': None if positive else ('nonpositive_parent_headroom' if defined else 'undefined_loss')}


def source_preservation(parent_losses, erased_losses):
    """All three source tasks must individually lose no more than .01 nats."""
    tasks = {}
    for target in SOURCE_TASKS:
        parent, erased = parent_losses.get(target), erased_losses.get(target)
        defined = finite(parent) and finite(erased)
        tasks[target] = {'parent_log_loss': parent, 'erased_log_loss': erased,
                         'difference': erased - parent if defined else None,
                         'maximum_increase': MARGINS['source_log_loss_increase'],
                         'pass': bool(erased <= parent + MARGINS['source_log_loss_increase']) if defined else None}
    return {'tasks': tasks, 'pass': joint_status(v['pass'] for v in tasks.values())}


def residential_retention(parent_loss, erased_loss, unprotected_B_loss, unprotected_C_loss):
    """Retain half the positive residence advantage over the stronger bank."""
    reference = min(unprotected_B_loss, unprotected_C_loss) if all(
        finite(v) for v in (unprotected_B_loss, unprotected_C_loss)) else None
    result = loss_headroom(parent_loss, erased_loss, reference)
    result['reference'] = 'minimum unprotected B/C residence log loss on this seed and split'
    result['minimum_retained_fraction'] = MARGINS['residential_headroom_retained']
    result['pass'] = bool(erased_loss <= (reference + parent_loss) / 2) if result['ratio_defined'] else None
    return result


def coverage_status(n_classes, fit_support, validation_support, evaluation_support, exposed_recall):
    """Require every class in all audit pools and a working exposed control."""
    if not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError('A fixed class schema of at least two classes is required')
    inputs = {'fit': fit_support, 'validation': validation_support,
              'evaluation': evaluation_support, 'exposed_recall': exposed_recall}
    limitations = {}
    for name, values in inputs.items():
        if values is None or len(values) != n_classes:
            limitations[name] = list(range(n_classes))
        else:
            limitations[name] = [k for k, value in enumerate(values) if not finite(value) or value <= 0]
    limitations = {name: indices for name, indices in limitations.items() if indices}
    return {'n_classes': n_classes, 'class_schema': list(range(n_classes)),
            'complete': not limitations, 'limitations': limitations,
            'fit_support': fit_support, 'validation_support': validation_support,
            'evaluation_support': evaluation_support, 'exposed_primary_recall': exposed_recall}


def attribute_halving(parent_attack_loss, erased_attack_loss, prior_loss, coverage):
    """Halve positive prior-relative attacker gain, with adequate coverage."""
    result = loss_headroom(parent_attack_loss, erased_attack_loss, prior_loss)
    result['parent_attribute_gain'] = result.pop('parent_headroom')
    result['erased_attribute_gain'] = result.pop('erased_headroom')
    result['maximum_gain_fraction_remaining'] = MARGINS['attribute_gain_remaining']
    result['coverage_complete'] = bool(coverage.get('complete', False))
    result['coverage_limitations'] = coverage.get('limitations', {})
    result['numeric_halving_inequality'] = bool(erased_attack_loss >= (prior_loss + parent_attack_loss) / 2) if result['ratio_defined'] else None
    if not result['coverage_complete']:
        result['pass'] = None
        result['ratio_defined'] = False
        result['retained_fraction'] = None
        result['headroom_reduction_fraction'] = None
        result['undefined_reason'] = 'coverage_limited'
    else:
        result['pass'] = bool(erased_attack_loss >= (prior_loss + parent_attack_loss) / 2) if result['ratio_defined'] else None
    return result


def feature_bank_comparison(feature_utility_loss, bank_utility_loss,
                            feature_attack_losses, bank_attack_losses, prior_losses,
                            attribute_coverage):
    """Require .01-nat utility benefit and at most .005 more gain per attribute."""
    defined = finite(feature_utility_loss) and finite(bank_utility_loss)
    utility = {'feature_log_loss': feature_utility_loss, 'bank_log_loss': bank_utility_loss,
               'feature_minus_bank': feature_utility_loss - bank_utility_loss if defined else None,
               'minimum_advantage': MARGINS['feature_bank_utility_advantage'],
               'pass': bool(feature_utility_loss <= bank_utility_loss - MARGINS['feature_bank_utility_advantage']) if defined else None}
    attributes = {}
    for target in ATTRIBUTES:
        feature, bank, prior = feature_attack_losses.get(target), bank_attack_losses.get(target), prior_losses.get(target)
        valid = all(finite(v) for v in (feature, bank, prior))
        complete = bool(attribute_coverage.get(target, {}).get('complete', False))
        feature_gain, bank_gain = (prior - feature, prior - bank) if valid else (None, None)
        attributes[target] = {'feature_gain': feature_gain, 'bank_gain': bank_gain,
            'feature_minus_bank_gain': feature_gain - bank_gain if valid else None,
            'maximum_extra_gain': MARGINS['feature_bank_attribute_gain_increase'],
            'coverage_complete': complete,
            'numeric_inequality': bool(feature >= bank - MARGINS['feature_bank_attribute_gain_increase']) if valid else None,
            'pass': bool(feature >= bank - MARGINS['feature_bank_attribute_gain_increase']) if valid and complete else None,
            'undefined_reason': 'coverage_limited' if not complete else None if valid else 'undefined_loss'}
    return {'utility': utility, 'attributes': attributes,
            'coverage_complete': all(v['coverage_complete'] for v in attributes.values()),
            'joint_pass': joint_status([utility['pass'], *[v['pass'] for v in attributes.values()]])}


def parent_comparison(parent_utility, erased_utility, unprotected_bank_residence,
                      parent_attack, erased_attack, prior_attack, attribute_coverage):
    """Evaluate one seed/split independently; commute has no invented margin."""
    source = source_preservation(parent_utility, erased_utility)
    residence = residential_retention(parent_utility.get('same_residence'), erased_utility.get('same_residence'),
                                     unprotected_bank_residence.get('B_rich_bank'), unprotected_bank_residence.get('C_tree_bank'))
    attributes = {target: attribute_halving(parent_attack.get(target), erased_attack.get(target),
                                            prior_attack.get(target), attribute_coverage.get(target, {}))
                  for target in ATTRIBUTES}
    differences = {target: erased_utility[target] - parent_utility[target] if finite(erased_utility.get(target))
                   and finite(parent_utility.get(target)) else None for target in UTILITY_TASKS}
    return {'utility_differences': differences, 'source_preservation': source,
            'residential_retention': residence, 'attribute_halving': attributes,
            'commute_margin': None, 'commute_note': 'reported separately; no provisional commute threshold',
            'coverage_complete': all(v['coverage_complete'] for v in attributes.values()),
            'joint_pass': joint_status([source['pass'], residence['pass'], *[v['pass'] for v in attributes.values()]])}


# Everything below reads saved aggregate evidence. No estimator, raw records,
# releases, prediction arrays, or fitting module is loaded by this reporter.
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as stat


RELEASES = tuple(release for parent in PARENTS for release in (parent, parent + '_leace')) + ('F_covariates', 'prior')
SPLITS = {'validation': 'validation', 'test': 'development_evaluation',
          'validation_person_weighted': 'validation_person_weighted',
          'test_person_weighted': 'development_evaluation_person_weighted'}
SCORE_METRICS = ('log_loss', 'auroc', 'macro_auroc', 'accuracy', 'balanced_accuracy',
                 'observed_balanced_accuracy', 'observed_macro_auroc')
DISPLAY = {'A_binary_bank': 'A binary bank', 'B_rich_bank': 'B rich neural bank',
           'C_tree_bank': 'C rich tree bank', 'D_compressed': 'D compressed features',
           'E_pca': 'E PCA features', 'F_covariates': 'F full covariates',
           'prior': 'Fitting prior', 'exposed': 'Exposed attribute'}
TASK_DISPLAY = {'same_residence': 'Same residence', 'commute_over20': 'Commute >20 min',
                'income_binary': 'Income >$50k', 'civilian_at_work': 'Civilian at work',
                'public_coverage': 'Public coverage', 'SEX': 'SEX', 'RAC1P': 'RAC1P'}


def label(release):
    return DISPLAY.get(release.removesuffix('_leace'), release) + (' + LEACE' if release.endswith('_leace') else '')


def number(value):
    return f'{value:.6f}' if finite(value) else 'undefined'


def statistics(values):
    """Do not silently average away a seed with an undefined fixed-schema score."""
    values = list(values)
    valid = [float(v) for v in values if finite(v)]
    complete = bool(values) and len(valid) == len(values)
    return {'n_seeds': len(values), 'n_defined': len(valid), 'complete': complete,
            'mean': stat.mean(valid) if complete else None,
            'sample_sd': stat.stdev(valid) if complete and len(valid) > 1 else None}


def mean_sd(record):
    if not record['complete']:
        return f"undefined ({record['n_defined']}/{record['n_seeds']})"
    return f"{number(record['mean'])} ± {number(record['sample_sd'])}"


def status(value):
    return {True: 'pass', False: 'fail', None: 'undefined'}[value]


def table(headers, rows):
    def line(row):
        return '| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |'
    return [line(headers), line(['---'] * len(headers)), *[line(row) for row in rows]]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def selectors(row):
    result = ['candidate:' + row['candidate_id']]
    if row['selected']:
        result.append('selected')
    if row['selected_within_family']:
        result.append('family:' + row['family'])
    if row['auc_selected']:
        result.append('auc_selected')
    return result


def make_index(records):
    result = {}
    for record in records:
        for row in record['raw_metrics']:
            if row['seed'] != record['seed']:
                raise ValueError('Seed identity mismatch')
            for selector in selectors(row):
                key = (row['seed'], row['role'], row['release'], row['target'], selector)
                if key in result:
                    raise ValueError('Duplicate selected/candidate score: ' + str(key))
                result[key] = row
    return result


def get_score(index, seed, role, release, target, split, selector='selected'):
    return index.get((seed, role, release, target, selector), {}).get(split, {})


def get_value(index, seed, role, release, target, split, metric='log_loss', selector='selected'):
    return get_score(index, seed, role, release, target, split, selector).get(metric)


def candidate_metadata(selection, row):
    key = '/'.join(row[k] for k in ('role', 'release', 'target'))
    return selection['fitting_records'][key]['candidates'][row['candidate_id']]


def flatten(records, selections):
    rows, classes, curves, accounting = [], [], [], []
    for record in records:
        selection = selections[record['seed']]
        for raw in record['raw_metrics']:
            common = {key: raw[key] for key in ('seed', 'role', 'release', 'parent', 'erased', 'target',
                       'candidate_id', 'family', 'selected', 'selected_within_family', 'auc_selected')}
            meta = candidate_metadata(selection, raw)
            common.update(fit_support=meta.get('fit_support'), fit_rows=meta.get('fit_rows'),
                          fit_coverage_complete=meta.get('fit_coverage_complete'))
            counts = {key: meta.get(key) for key in ('selected_epoch', 'selected_optimizer_steps', 'optimizer_steps',
                      'boosting_iterations', 'training_row_exposures', 'row_exposure_min', 'row_exposure_max',
                      'initialization_seed', 'schedule_seed', 'schedule_hash', 'restart_index', 'restart_seed_offset',
                      'fit_runtime_seconds', 'fallback_reason', 'warnings')}
            accounting.append({**common, **counts, 'parameters': meta.get('parameters')})
            for curve in meta.get('validation_curve', []):
                curves.append({**common, **curve, 'checkpoint_selected': curve.get('epoch') == meta.get('selected_epoch'),
                               'total_optimizer_steps': meta.get('optimizer_steps'),
                               'total_training_row_exposures': meta.get('training_row_exposures')})
            for raw_split, split_name in SPLITS.items():
                score = raw[raw_split]
                row = {**common, 'split': split_name,
                       'evaluation_status': 'DEVELOPMENT EVALUATION; reused original test households',
                       **{key: value for key, value in score.items() if key != 'per_class'}}
                rows.append(row)
                for cls in score['per_class']:
                    classes.append({**common, 'split': split_name, **cls,
                                    'original_census_code': cls['class_index'] + 1 if raw['role'] == 'audit' else None,
                                    'fit_class_support': meta.get('fit_support', [None] * score['n_classes'])[cls['class_index']],
                                    'schema_coverage_complete': score['coverage_complete']})
    return rows, classes, curves, accounting


def audit_coverage(index, selection, seed, release, target, split, selector='selected'):
    row = index.get((seed, 'audit', release, target, selector))
    fit = candidate_metadata(selection, row).get('fit_support') if row else None
    val = row.get('validation', {}).get('support') if row else None
    evaluated = row.get(split, {}).get('support') if row else None
    control_row = index.get((seed, 'audit', 'exposed', target, 'selected'))
    control = control_row.get(split, {}) if control_row else {}
    recalls = [cls.get('recall') for cls in control.get('per_class', [])]
    result = coverage_status(2 if target == 'SEX' else 9, fit, val, evaluated, recalls)
    result.update(exposed_candidate=control_row['candidate_id'] if control_row else None,
                  audit_candidate=row['candidate_id'] if row else None,
                  limitation_census_codes={k: [c + 1 for c in v] for k, v in result['limitations'].items()})
    return result


def combined_coverage(*records):
    """A comparison needs support and a competent control for both fitted attacks."""
    limitations = {}
    for index, record in enumerate(records):
        for key, value in record['limitations'].items():
            limitations[f'arm_{index}/{key}'] = value
    return {'complete': bool(records) and all(record['complete'] for record in records),
            'limitations': limitations, 'arms': list(records)}


def compare_all(records, selections, index):
    criteria, features, coverage_rows = [], [], []
    for record in records:
        seed = record['seed']
        for split in ('validation', 'test'):
            cov = {release: {target: audit_coverage(index, selections[seed], seed, release, target, split)
                            for target in ATTRIBUTES} for release in RELEASES}
            for release, attrs in cov.items():
                for target, coverage in attrs.items():
                    coverage_rows.append({'seed': seed, 'split': SPLITS[split], 'release': release,
                                          'target': target, **coverage})
            utility = lambda release: {target: get_value(index, seed, 'transfer', release, target, split)
                                       for target in UTILITY_TASKS}
            attack = lambda release: {target: get_value(index, seed, 'audit', release, target, split)
                                      for target in ATTRIBUTES}
            reference = {bank: utility(bank)['same_residence'] for bank in ('B_rich_bank', 'C_tree_bank')}
            for parent in PARENTS:
                erased = parent + '_leace'
                coverage = {target: combined_coverage(cov[parent][target], cov[erased][target]) for target in ATTRIBUTES}
                value = parent_comparison(utility(parent), utility(erased), reference,
                                          attack(parent), attack(erased), attack('prior'), coverage)
                meta = record['release_metadata'][erased]['eraser']
                eraser_gaps = {target: [k for k, n in enumerate(meta['coverage'][target]['support_complete_cases']) if n <= 0]
                               for target in ATTRIBUTES}
                eraser_gaps = {k: v for k, v in eraser_gaps.items() if v}
                criteria.append({'seed': seed, 'split': SPLITS[split], 'parent': parent, 'erased': erased,
                                 'head_selector': 'selected', 'audit_selector': 'selected',
                                 'eraser_fit_schema_complete': not eraser_gaps,
                                 'eraser_fit_missing_census_codes': {k: [v + 1 for v in values] for k, values in eraser_gaps.items()},
                                 'attribute_coverage': coverage, **value})
            feature_arms = ('D_compressed', 'D_compressed_leace', 'E_pca', 'E_pca_leace')
            bank_arms = ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')
            for feature in feature_arms:
                for bank in bank_arms:
                    coverage = {target: combined_coverage(cov[feature][target], cov[bank][target]) for target in ATTRIBUTES}
                    f_util, b_util = utility(feature), utility(bank)
                    comparison = feature_bank_comparison(f_util['same_residence'], b_util['same_residence'],
                                                         attack(feature), attack(bank), attack('prior'), coverage)
                    differences = {target: f_util[target] - b_util[target] if finite(f_util[target]) and finite(b_util[target]) else None
                                   for target in UTILITY_TASKS}
                    features.append({'seed': seed, 'split': SPLITS[split], 'feature': feature, 'bank': bank,
                                     'utility_target_for_margin': 'same_residence',
                                     'all_task_feature_minus_bank_log_loss': differences,
                                     'numeric_joint_inequalities': joint_status([comparison['utility']['pass'],
                                         *[v['numeric_inequality'] for v in comparison['attributes'].values()]]), **comparison})
    return criteria, features, coverage_rows


def paired_rows(records, index):
    rows = []
    comparisons = [('erased_minus_parent', p + '_leace', p) for p in PARENTS]
    comparisons += [('feature_minus_bank', feature, bank)
                    for feature in ('D_compressed', 'D_compressed_leace', 'E_pca', 'E_pca_leace')
                    for bank in ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')]
    comparisons += [('compressed_minus_pca', feature, pca)
                    for feature in ('D_compressed', 'D_compressed_leace')
                    for pca in ('E_pca', 'E_pca_leace')]
    for record in records:
        seed = record['seed']
        for comparison, left, right in comparisons:
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                if comparison == 'erased_minus_parent':
                    choices = ('selected', 'family:logistic', 'family:mlp')
                    if role == 'audit':
                        choices += ('family:histgb', 'auc_selected')
                else:
                    choices = ('selected',)
                    targets = ('same_residence',) if role == 'transfer' else ATTRIBUTES
                for target in targets:
                    for selector in choices:
                        left_row = index.get((seed, role, left, target, selector))
                        right_row = index.get((seed, role, right, target, selector))
                        for split in SPLITS:
                            for metric in SCORE_METRICS:
                                a = left_row.get(split, {}).get(metric) if left_row else None
                                b = right_row.get(split, {}).get(metric) if right_row else None
                                rows.append({'seed': seed, 'comparison': comparison, 'left': left, 'right': right,
                                    'role': role, 'target': target, 'selector': selector, 'split': SPLITS[split],
                                    'metric': metric, 'left_candidate': left_row['candidate_id'] if left_row else None,
                                    'right_candidate': right_row['candidate_id'] if right_row else None,
                                    'left_value': a, 'right_value': b,
                                    'left_minus_right': a - b if finite(a) and finite(b) else None})
    return rows


def aggregate_metrics(records, index):
    seeds = [r['seed'] for r in records]
    groups = sorted({key[1:] for key in index})
    rows = []
    for role, release, target, selector in groups:
        for split in SPLITS:
            for metric in SCORE_METRICS:
                vals = [get_value(index, seed, role, release, target, split, metric, selector) for seed in seeds]
                rows.append({'role': role, 'release': release, 'target': target, 'selector': selector,
                             'split': SPLITS[split], 'metric': metric, **statistics(vals)})
    return rows


def aggregate_attribute_gains(records, index):
    """Pair each candidate with its fitting prior before taking seed statistics."""
    seeds = [r['seed'] for r in records]
    groups = sorted({key[1:] for key in index if key[1] == 'audit'})
    rows = []
    for role, release, target, selector in groups:
        for split in SPLITS:
            values = []
            for seed in seeds:
                prior = get_value(index, seed, role, 'prior', target, split)
                attack = get_value(index, seed, role, release, target, split, selector=selector)
                values.append(prior - attack if finite(prior) and finite(attack) else None)
            rows.append({'release': release, 'target': target, 'selector': selector, 'split': SPLITS[split],
                         'definition': 'signed prior log loss minus attack log loss', **statistics(values)})
    return rows


def aggregate_pairs(rows, seeds):
    fields = ('comparison', 'left', 'right', 'role', 'target', 'selector', 'split', 'metric')
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row[k] for k in fields)][row['seed']] = row['left_minus_right']
    return [{**dict(zip(fields, key)), **statistics(values.get(seed) for seed in seeds)} for key, values in sorted(groups.items())]


def counts(values):
    values = list(values)
    return {'pass': sum(v is True for v in values), 'fail': sum(v is False for v in values),
            'undefined': sum(v is None for v in values), 'n_seeds': len(values)}


def criterion_counts(criteria):
    rows = []
    for split in ('validation', 'development_evaluation'):
        for parent in PARENTS:
            selected = [r for r in criteria if r['parent'] == parent and r['split'] == split]
            accessors = {'source_all': lambda r: r['source_preservation']['pass'],
                         **{target: lambda r, t=target: r['source_preservation']['tasks'][t]['pass'] for target in SOURCE_TASKS},
                         'residential_retention': lambda r: r['residential_retention']['pass'],
                         **{target + '_halving': lambda r, t=target: r['attribute_halving'][t]['pass'] for target in ATTRIBUTES},
                         'joint': lambda r: r['joint_pass']}
            rows.extend({'split': split, 'parent': parent, 'criterion': name, **counts(fn(row) for row in selected)}
                        for name, fn in accessors.items())
    return rows


def report_tables(out, records, index, criteria, features, aggregates, pairs, gains):
    seeds = [r['seed'] for r in records]
    ai = {(r['role'], r['release'], r['target'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    pi = {(r['comparison'], r['left'], r['right'], r['role'], r['target'], r['selector'], r['split'], r['metric']): r for r in pairs}
    def avg(role, release, target, split, metric='log_loss', selector='selected'):
        return mean_sd(ai.get((role, release, target, selector, split, metric), statistics([None] * len(seeds))))
    def diff(left, right, role, target, split, selector='selected'):
        return mean_sd(pi['erased_minus_parent', left, right, role, target, selector, split, 'log_loss'])
    lines = ['# ACS erasure feasibility — saved development evidence', '',
             'The original test households informed this question; all development columns reuse them. '
             'Selections use only the designated validation pool. Natural log loss is primary. '
             'Lower task loss is better; higher attack loss means less recovery by this finite auditor set. '
             'Means ± sample SD are descriptive across shared-cohort seeds. Undefined SD with one completed seed is retained.', '',
             f"Completed seeds: {', '.join(map(str, seeds))}. "
             '[Criteria and interpretation](ANALYSIS.md), [support and exposed controls](SUPPORT.md), '
             '[all candidates](PER_TARGET.csv), [all classes](PER_CLASS.csv), [paired scores](PAIRED.csv).', '']
    for split, title in (('validation', 'Validation'), ('development_evaluation', 'Development evaluation')):
        lines += [f'## {title}: primary log loss, mean ± sample SD', '']
        lines += table(['Release', 'Residence', 'Commute', 'SEX attack', 'RAC1P attack'],
                       [[label(release), avg('transfer', release, 'same_residence', split),
                         avg('transfer', release, 'commute_over20', split),
                         *[avg('audit', release, target, split) for target in ATTRIBUTES]] for release in RELEASES])
        lines += ['', 'Source tasks use newly fitted, matched downstream heads on each frozen release.', '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in SOURCE_TASKS]],
                       [[label(release), *[avg('transfer', release, target, split) for target in SOURCE_TASKS]] for release in RELEASES])
        lines += ['', f'## {title}: every seed, primary log loss', '']
        raw_split = 'validation' if split == 'validation' else 'test'
        lines += table(['Seed', 'Release', 'Residence', 'Commute', 'Income', 'Civilian at work', 'Public coverage', 'SEX', 'RAC1P'],
                       [[seed, label(release),
                         *[number(get_value(index, seed, 'transfer', release, target, raw_split)) for target in UTILITY_TASKS],
                         *[number(get_value(index, seed, 'audit', release, target, raw_split)) for target in ATTRIBUTES]]
                        for seed in seeds for release in RELEASES])
        lines += ['', f'## {title}: paired erased − parent log loss', '',
                  'Positive differences lose task utility but reduce measured attribute recovery. No differences are clipped.', '']
        lines += table(['Parent', 'Residence', 'Commute', 'Income', 'Civilian at work', 'Public coverage', 'SEX', 'RAC1P'],
                       [[label(parent), *[diff(parent + '_leace', parent, 'transfer', target, split) for target in UTILITY_TASKS],
                         *[diff(parent + '_leace', parent, 'audit', target, split) for target in ATTRIBUTES]] for parent in PARENTS])
        lines += ['', f'## {title}: provisional criteria by seed', '',
                  'Undefined is not a pass. Known failures remain failures when another condition is undefined. '
                  'Residential retention requires positive headroom over the stronger unprotected rich bank; '
                  'attribute halving requires positive parent gain and full support/competent exposed control.', '']
        lines += table(['Seed', 'Parent', 'Source all ≤+.01', 'Residence retain half', 'SEX halve gain', 'RAC1P halve gain', 'Joint'],
                       [[r['seed'], label(r['parent']), status(r['source_preservation']['pass']),
                         status(r['residential_retention']['pass']),
                         *[status(r['attribute_halving'][target]['pass']) for target in ATTRIBUTES], status(r['joint_pass'])]
                        for r in criteria if r['split'] == split])
        lines += ['']
    lines += ['## Stored release properties', '',
              'Dimensions and ranks are reported on representation-fitting rows. Rank uses the frozen relative covariance threshold; '
              'stored dimensions are preserved even where a map becomes constant.', '']
    lines += table(['Seed', 'Release', 'Stored dim', 'Bytes/person', 'Numerical rank', 'Spectral effective rank', 'Float32 calibration max |cov|'],
                   [[r['seed'], label(name), meta['dimension'], meta['bytes_per_record'], meta['centered_covariance_rank'],
                     number(meta['entropy_effective_rank']),
                     number(meta['eraser']['after_float32']['cross_covariance_max_abs']) if meta['eraser'] else '—']
                    for r in records for name, meta in r['release_metadata'].items()])
    lines += ['', 'Calibration covariance is a fitting-sample statistic. It provides no classification-accuracy or nonlinear-privacy guarantee.', '']
    (out/'TABLE.md').write_text('\n'.join(lines))

    analysis = ['# ACS erasure feasibility analysis', '',
                'This report summarizes frozen outputs without refitting or selecting on development outcomes. '
                'The provisional comparisons are descriptive empirical checks, not privacy budgets or statistical noninferiority tests. '
                'Validation and development evaluation are assessed independently, seed by seed. '
                'These seeds share the same cohort, so the SD is not population uncertainty.', '',
                '[Primary tables](TABLE.md), [coverage and exposed controls](SUPPORT.md), [all scores](PER_TARGET.csv), '
                '[per-class scores](PER_CLASS.csv), [training curves](CURVES.csv), [fit accounting](FITTING.csv), '
                '[paired comparisons](PAIRED.csv), [exact criteria](criteria.json), '
                '[feature/bank comparisons](feature_comparisons.json), [all means and SDs](summary.json).', '',
                'Attack gain is prior log loss minus attack log loss, retaining its sign. A nonpositive parent gain '
                'makes fractional reduction undefined. Incomplete fit/validation/evaluation class support or a zero/undefined '
                'class recall for the validation-selected exposed control limits the attribute assessment; it cannot pass. '
                'Raw scalar inequalities are still published. Eraser calibration coverage gaps are flagged separately.', '']
    cc = criterion_counts(criteria)
    analysis += ['## Provisional criterion counts', '', 'Cells are pass / fail / undefined across completed seeds.', '']
    ci = {(r['split'], r['parent'], r['criterion']): r for r in cc}
    def count_cell(split, parent, criterion):
        v = ci[split, parent, criterion]
        return f"{v['pass']} / {v['fail']} / {v['undefined']}"
    analysis += table(['Split', 'Parent', 'Source all', 'Residence retention', 'SEX halving', 'RAC1P halving', 'Joint'],
                      [[split.replace('_', ' '), label(parent),
                        *[count_cell(split, parent, c) for c in ('source_all', 'residential_retention', 'SEX_halving', 'RAC1P_halving', 'joint')]]
                       for split in ('validation', 'development_evaluation') for parent in PARENTS])
    gi = {(r['release'], r['target'], r['selector'], r['split']): r for r in gains}
    analysis += ['', '## Signed attribute gain, mean ± sample SD', '',
                 'Each seed pairs the selected attack with that seed’s fitting-prior loss before aggregation. '
                 'Negative gains remain negative. Gains do not imply a protection pass when coverage is limited.', '']
    analysis += table(['Release', 'Validation SEX', 'Validation RAC1P', 'Development SEX', 'Development RAC1P'],
                      [[label(release), *[mean_sd(gi[release, target, 'selected', split])
                        for split in ('validation', 'development_evaluation') for target in ATTRIBUTES]] for release in RELEASES])
    analysis += ['', '## Compressed features versus PCA: explicit paired contrasts', '',
                 'These contrasts use validation-selected primary heads and attacks. Each cell is the mean ± sample SD '
                 'of compressed-feature minus PCA log loss paired within seed. Lower residence loss is better; '
                 'higher attack loss means less measured recovery. This comparison has no additional provisional threshold.', '']
    analysis += table(['Split', 'Compressed arm', 'PCA arm', 'Residence LL difference', 'SEX attack LL difference', 'RAC1P attack LL difference'],
                      [[split.replace('_', ' '), label(feature), label(pca),
                        mean_sd(pi['compressed_minus_pca', feature, pca, 'transfer', 'same_residence', 'selected', split, 'log_loss']),
                        *[mean_sd(pi['compressed_minus_pca', feature, pca, 'audit', target, 'selected', split, 'log_loss']) for target in ATTRIBUTES]]
                       for split in ('validation', 'development_evaluation')
                       for feature in ('D_compressed', 'D_compressed_leace') for pca in ('E_pca', 'E_pca_leace')])
    for split in ('validation', 'development_evaluation'):
        analysis += ['', f"## {split.replace('_', ' ').capitalize()}: headroom and signed attribute gains", '']
        analysis += table(['Seed', 'Parent', 'Residence H before', 'Residence H after', 'Residence retained',
                          'SEX gain before → after', 'SEX retained', 'RAC1P gain before → after', 'RAC1P retained'],
                         [[r['seed'], label(r['parent']), number(r['residential_retention']['parent_headroom']),
                           number(r['residential_retention']['erased_headroom']), number(r['residential_retention']['retained_fraction']),
                           *[v for target in ATTRIBUTES for v in (
                               number(r['attribute_halving'][target]['parent_attribute_gain']) + ' → ' + number(r['attribute_halving'][target]['erased_attribute_gain']),
                               number(r['attribute_halving'][target]['retained_fraction']))]]
                          for r in criteria if r['split'] == split])
        analysis += ['', f"## {split.replace('_', ' ').capitalize()}: every feature/bank margin", '',
                     'Count cells are pass / fail / undefined. Numeric attribute and joint columns disregard coverage only to '
                     'show the scalar inequalities; the final policy column retains coverage limitations. '
                     'Utility means residence loss at least .01 lower; each attribute permits at most .005 extra signed gain.', '']
        frows = []
        for feature in ('D_compressed', 'D_compressed_leace', 'E_pca', 'E_pca_leace'):
            for bank in ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace'):
                rs = [r for r in features if r['split'] == split and r['feature'] == feature and r['bank'] == bank]
                getters = [lambda r: r['utility']['pass'],
                           *[lambda r, t=t: r['attributes'][t]['numeric_inequality'] for t in ATTRIBUTES],
                           lambda r: r['numeric_joint_inequalities'], lambda r: r['joint_pass']]
                cells = []
                for fn in getters:
                    v = counts(fn(r) for r in rs)
                    cells.append(f"{v['pass']} / {v['fail']} / {v['undefined']}")
                frows.append([label(feature), label(bank), *cells])
        analysis += table(['Feature', 'Bank', 'Utility', 'SEX numeric', 'RAC1P numeric', 'Joint numeric', 'Joint with coverage'], frows)
    analysis += ['', '## All primary scoring metrics, mean ± sample SD', '',
                 'Full-schema macro AUROC and balanced accuracy stay undefined if a category is unsupported. '
                 'Observed-class alternatives remain explicitly labeled in CSV/JSON. PWGTP sensitivity uses the same '
                 'predictions and unweighted validation selections; it is not a Census population estimate.', '']
    for split in SPLITS.values():
        analysis += [f"### {split.replace('_', ' ').capitalize()}", '']
        analysis += table(['Release', 'Target', 'Log loss', 'Accuracy', 'Balanced accuracy', 'AUROC (binary) / macro AUROC (RAC1P)'],
                          [[label(release), TASK_DISPLAY[target], *[avg(role, release, target, split, metric)
                            for metric in ('log_loss', 'accuracy', 'balanced_accuracy', 'macro_auroc' if target == 'RAC1P' else 'auroc')]]
                           for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES))
                           for target in targets for release in RELEASES])
        analysis += ['']
    analysis += ['## Matched-family paired log losses', '',
                 'These are within-family validation selections, separate from the primary minimum-log-loss candidate across families. '
                 'Every restart/configuration and its validation-AUROC diagnostic selection appears in the raw candidate CSV. '
                 'Paired CSV/JSON retain all erased-parent task/attribute metrics and person-weighted sensitivity. '
                 'Feature/bank and compressed/PCA cross-comparisons retain primary residence/attribute contrasts; '
                 'every other raw candidate score remains available separately.', '']
    for split in ('validation', 'development_evaluation'):
        analysis += [f"### {split.replace('_', ' ').capitalize()}", '']
        analysis += table(['Parent', 'Role', 'Family', 'Target', 'Erased − parent log loss'],
                          [[label(parent), role, selector.removeprefix('family:'), TASK_DISPLAY[target],
                            diff(parent + '_leace', parent, role, target, split, selector)]
                           for parent in PARENTS for role, targets, families in (
                               ('transfer', UTILITY_TASKS, ('logistic', 'mlp')),
                               ('audit', ATTRIBUTES, ('logistic', 'mlp', 'histgb')))
                           for selector in ['family:' + family for family in families] for target in targets])
        analysis += ['']
    analysis += ['## Runtime and invariants', '']
    runtime_keys = ('total_seconds', 'parent_loading_verification_seconds', 'erasure_and_freeze_seconds',
                    'utility_fitting_seconds', 'audit_fitting_seconds', 'controls_seconds',
                    'evaluation_serialization_integrity_seconds')
    analysis += table(['Seed', 'Total s', 'Parent load s', 'Erasure s', 'Heads s', 'Audits s', 'Controls s', 'Evaluation s'],
                      [[r['seed'], *[number(r['runtime'].get(k)) for k in runtime_keys]] for r in records])
    analysis += ['', f"Sum of recorded internal runtimes: {number(sum(r['runtime']['total_seconds'] for r in records))} seconds. "
                 'Wall-clock execution timing is recorded by the orchestrator separately. All loaded per-seed integrity flags '
                 'are retained in summary.json, including source/map/output hashes and selection-before-evaluation times.', '',
                 'No source model, map, task, or protection strength is selected by these comparisons. '
                 'Empirical covariance removal and bounded-auditor failure do not establish privacy. '
                 'A task-probability bank is a practical reference, not an information lower bound. '
                 'Commute eligibility itself may reveal information beyond the audited numerical release.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))


def support_report(out, records, selections, index, coverage_rows):
    lines = ['# Fixed-schema support and control competence', '',
             'Support vectors retain original class order: SEX codes 1–2 and RAC1P codes 1–9. '
             'Zero counts and zero/undefined class recalls are explicit. All original test pools below are development evaluation. '
             'Per-class predictions, weighted support, recall, precision, F1 and AUROC for every candidate are in '
             '[PER_CLASS.csv](PER_CLASS.csv). Fitting support uses the actual capped audit/head subset; '
             'pool support also includes rows not selected into a fitting subset.', '', '## Household pools and valid labels', '']
    lines += table(['Seed', 'Pool', 'People', 'Households', 'Target', 'Valid', 'Missing', 'Class support'],
                   [[r['seed'], 'development_evaluation' if pool == 'test' else pool, meta['persons'], meta['households'],
                     target, support['valid'], support['missing_or_inapplicable'], json.dumps(support['class_counts'])]
                    for r in records for pool, meta in r['support_by_pool'].items()
                    for group in ('tasks', 'attributes') for target, support in meta[group].items()])
    lines += ['', '## Every exposed auditor and restart', '',
              'Primary and within-family selections minimize validation log loss. An exposed-control failure is not erased by '
              'a better result from a different diagnostic configuration; the primary coverage rule uses the primary selected control. '
              'Displayed failing codes have evaluation support zero or recall zero/undefined.', '']
    rows = []
    for record in records:
        for raw in record['raw_metrics']:
            if raw['release'] != 'exposed':
                continue
            meta = candidate_metadata(selections[record['seed']], raw)
            for split in ('validation', 'test'):
                score = raw[split]
                failures = [v['class_index'] + 1 for v in score['per_class'] if not finite(v['recall']) or v['recall'] <= 0 or v['support'] <= 0]
                rows.append([raw['seed'], SPLITS[split].replace('_', ' '), raw['target'], raw['candidate_id'],
                    raw['selected'], raw['selected_within_family'], raw['auc_selected'], number(score['log_loss']),
                    number(score['accuracy']), number(score['balanced_accuracy']),
                    number(score['macro_auroc'] if raw['target'] == 'RAC1P' else score['auroc']),
                    json.dumps(meta.get('fit_support')), json.dumps(score['support']), json.dumps(failures),
                    ', '.join(number(v['recall']) for v in score['per_class'])])
    lines += table(['Seed', 'Split', 'Target', 'Candidate', 'Primary', 'Family selected', 'AUC selected', 'LL', 'Accuracy',
                    'Balanced acc', 'AUROC / macro', 'Fit support', 'Eval support', 'Failing codes', 'Recall by code'], rows)
    lines += ['', '## Primary audit coverage on every release', '',
              'A limited attribute assessment cannot be a policy pass. Limitation keys use original Census codes. '
              'These checks require actual fit support, attacker-validation support, evaluated support, and a positive recall '
              'for each class from the primary exposed control. Race gaps are never replaced by an observed-class privacy claim.', '']
    lines += table(['Seed', 'Split', 'Release', 'Target', 'Primary audit', 'Primary exposed', 'Complete', 'Limitations (Census codes)'],
                   [[r['seed'], r['split'].replace('_', ' '), label(r['release']), r['target'], r['audit_candidate'],
                     r['exposed_candidate'], r['complete'], json.dumps(r['limitation_census_codes'], separators=(',', ':'))]
                    for r in coverage_rows])
    lines += ['', '## Eraser calibration coverage and covariance', '',
              'All eleven one-hot columns stay in the fixed joint schema. A fitting gap remains a gap even with zero residual '
              'covariance. Covariance statistics use complete fitting cases and the n−1 denominator, independently of predictive scoring.', '']
    lines += table(['Seed', 'Erased release', 'Target', 'Complete-case support', 'Missing input rows', 'Schema complete',
                    'After float64 max |cov|', 'After float32 max |cov|', 'Fitting covariance criterion'],
                   [[r['seed'], label(name), target, json.dumps(meta['eraser']['coverage'][target]['support_complete_cases']),
                     meta['eraser']['coverage'][target]['missing_rows'], meta['eraser']['after_float32']['schema_complete'],
                     f"{meta['eraser']['after_float64']['cross_covariance_max_abs']:.3e}",
                     f"{meta['eraser']['after_float32']['cross_covariance_max_abs']:.3e}",
                     meta['eraser']['after_float32']['empirical_covariance_within_tolerance']]
                    for r in records for name, meta in r['release_metadata'].items() if meta['eraser'] for target in ATTRIBUTES])
    lines += ['', 'Raw pool row hashes and selection subset hashes are retained in the per-seed support/selection JSON, linked '
              'by summary.json provenance. No raw person records or prediction arrays are loaded by this report.', '']
    (out/'SUPPORT.md').write_text('\n'.join(lines))


def plot_tradeoffs(out, aggregates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    ai = {(r['role'], r['release'], r['target'], r['split']): r for r in aggregates
          if r['selector'] == 'selected' and r['metric'] == 'log_loss'}
    palette = dict(zip(PARENTS, ('#4978ac', '#d17b2d', '#659845', '#ae4b62', '#8663ab')))
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        for ax, split, title in zip(axes, ('validation', 'development_evaluation'), ('Validation', 'Development evaluation')):
            def point(release, color, marker, prefix=None):
                x = ai.get(('transfer', release, 'same_residence', split), {})
                y = ai.get(('audit', release, target, split), {})
                if not finite(x.get('mean')) or not finite(y.get('mean')):
                    return None
                xy = x['mean'], y['mean']
                ax.errorbar(*xy, xerr=x.get('sample_sd') or 0., yerr=y.get('sample_sd') or 0.,
                            marker=marker, color=color, markersize=6, capsize=2, linewidth=1, alpha=.9)
                if prefix:
                    ax.annotate(prefix, xy, xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)
                return xy
            for parent in PARENTS:
                a = point(parent, palette[parent], 'o')
                b = point(parent + '_leace', palette[parent], 'D')
                if a and b:
                    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': palette[parent], 'lw': 1.3, 'alpha': .75})
            point('F_covariates', '#333333', 's', 'F')
            point('prior', '#777777', 'X', 'Prior')
            ax.set_title(title)
            ax.set_xlabel('Same-residence log loss (lower is better)')
            ax.set_ylabel(f'{target} attack log loss (higher: less measured recovery)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='both', style='plain', useOffset=False)
            ax.grid(alpha=.2)
            ax.margins(.12)
        dimensions = {'A_binary_bank': 3, 'B_rich_bank': 26, 'C_tree_bank': 26, 'D_compressed': 26, 'E_pca': 32}
        handles = [Line2D([0], [0], color=palette[p], marker='o', label=f'{label(p)} ({dimensions[p]}D)') for p in PARENTS]
        handles += [Line2D([0], [0], color='#333333', marker='s', linestyle='', label='F full covariates (75–77D)'),
                    Line2D([0], [0], color='#777777', marker='X', linestyle='', label='Fitting prior'),
                    Line2D([0], [0], color='#555555', marker='D', linestyle='', label='LEACE (arrow destination)')]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5, bbox_to_anchor=(.5, .015))
        fig.suptitle(f'Frozen releases: residential utility and {target} recovery', fontsize=12)
        note = ('Mean ± sample SD across shared-cohort seeds; empirical audit, not a privacy certificate.'
                if target == 'SEX' else
                'Mean ± sample SD across shared-cohort seeds; race-category coverage incomplete; no privacy certificate.')
        fig.text(.5, .13, note, ha='center', fontsize=8.5)
        fig.tight_layout(rect=(0, .17, 1, .95))
        fig.savefig(out/f'tradeoff_{target}.png', dpi=200)
        fig.savefig(out/f'tradeoff_{target}.pdf')
        plt.close(fig)


def summarize(out):
    out = Path(out).resolve()
    metric_paths = sorted(out.glob('seed_*/metrics.json'), key=lambda p: int(p.parent.name.removeprefix('seed_')))
    if not metric_paths:
        raise ValueError('No completed seed metrics; reporting performs no fitting')
    records = [read_json(path) for path in metric_paths]
    seeds = [record['seed'] for record in records]
    config = read_json(out/'config.json')
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config['seeds']):
        raise ValueError('Duplicate or undeclared completed seed')
    selections = {record['seed']: read_json(path.parent/'selection_before_test.json')
                  for path, record in zip(metric_paths, records)}
    index = make_index(records)
    rows, classes, curves, accounting = flatten(records, selections)
    raw_split_names = {name: raw for raw, name in SPLITS.items()}
    for row in rows:
        if row['role'] == 'audit':
            prior = get_value(index, row['seed'], 'audit', 'prior', row['target'], raw_split_names[row['split']])
            row['prior_log_loss'] = prior
            row['prior_relative_attribute_gain'] = prior - row['log_loss'] if finite(prior) and finite(row['log_loss']) else None
    criteria, features, coverage = compare_all(records, selections, index)
    paired = paired_rows(records, index)
    aggregates = aggregate_metrics(records, index)
    gains = aggregate_attribute_gains(records, index)
    paired_aggregates = aggregate_pairs(paired, seeds)
    for name, values in (('PER_TARGET.csv', rows), ('PER_CLASS.csv', classes), ('CURVES.csv', curves),
                         ('FITTING.csv', accounting), ('PAIRED.csv', paired)):
        save_csv(out/name, values)
    save_json(out/'criteria.json', {'evaluation_status': config['evaluation_status'], 'margins': MARGINS,
              'per_seed_split': criteria, 'counts': criterion_counts(criteria)})
    save_json(out/'feature_comparisons.json', {'utility_margin_target': 'same_residence', 'margins': MARGINS,
              'scope': 'Numeric inequalities are descriptive; coverage still gates each policy assessment.',
              'per_seed_split': features})
    inputs = [out/'config.json', out/'PROTOCOL.md', *metric_paths,
              *[p.parent/'selection_before_test.json' for p in metric_paths]]
    summary = {'evaluation_status': config['evaluation_status'], 'completed_seeds': seeds,
               'declared_seeds': config['seeds'], 'all_declared_seeds_complete': set(seeds) == set(config['seeds']),
               'scorer': {'primary': 'unweighted natural log loss; p floor 1e-12 then renormalize',
                 'selection': 'saved validation log-loss selections, never recomputed from development outcomes',
                 'attribute_gain': 'signed prior log loss minus attack log loss',
                 'fixed_schemas': {'SEX': 2, 'RAC1P': 9, 'utility': 2},
                 'undefined': 'null; never averaged away or counted as a policy pass',
                 'uncertainty': 'sample SD across shared-cohort seeds; descriptive, not population uncertainty'},
               'margins': MARGINS, 'aggregate_metrics': aggregates, 'paired_aggregate_metrics': paired_aggregates,
               'attribute_gain_aggregates': gains,
               'criterion_counts': criterion_counts(criteria), 'coverage': coverage,
               'release_metadata_paths': {r['seed']: f"seed_{r['seed']}/release_freeze.json" for r in records},
               'support_paths': {r['seed']: f"seed_{r['seed']}/support.json" for r in records},
               'integrity': {r['seed']: r['integrity'] for r in records},
               'runtime': {r['seed']: r['runtime'] for r in records},
               'internal_runtime_total_seconds': sum(r['runtime']['total_seconds'] for r in records),
               'candidate_selection_counts': dict(Counter(row['candidate_id'] for r in records for row in r['raw_metrics'] if row['selected'])),
               'row_counts': {'per_target': len(rows), 'per_class': len(classes), 'curves': len(curves),
                              'fitting_candidates': len(accounting), 'paired': len(paired)},
               'report_source_sha256': sha(__file__),
               'input_sha256': {str(path.relative_to(out)): sha(path) for path in inputs}}
    save_json(out/'summary.json', summary)
    report_tables(out, records, index, criteria, features, aggregates, paired_aggregates, gains)
    support_report(out, records, selections, index, coverage)
    plot_tradeoffs(out, aggregates)
    print(json.dumps({'out': str(out), 'seeds': seeds, **summary['row_counts'],
                      'internal_runtime_seconds': summary['internal_runtime_total_seconds']}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)


if __name__ == '__main__':
    main()

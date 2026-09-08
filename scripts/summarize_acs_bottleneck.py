"""Summarize saved ACS bottleneck evidence; no fitting or outcome selection.

Usage: python scripts/summarize_acs_bottleneck.py --out RESULTS_DIRECTORY
All original-test outcomes are development evaluation. The strongest saved
validation-selected independent/catch-up attack is the primary audit; the saved
training adversary is a diagnostic and can never become the primary attack.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, MARGINS, SPLITS, SCORE_METRICS,
    TASK_DISPLAY, finite, joint_status, parent_comparison, feature_bank_comparison,
    combined_coverage, audit_coverage, get_score, get_value, candidate_metadata,
    statistics, mean_sd, number, status, table, read_json, save_json, save_csv, sha,
    aggregate_metrics, aggregate_attribute_gains, aggregate_pairs,
)


LEARNED = ('C_bottleneck', 'D_protected')
BANKS = ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')
RELEASES = ('E_pca', 'E_pca_leace', *BANKS, *LEARNED, 'prior')
DISPLAY = {'E_pca': 'Original PCA', 'E_pca_leace': 'PCA + LEACE',
           'B_rich_bank': 'Rich neural bank', 'B_rich_bank_leace': 'Neural bank + LEACE',
           'C_tree_bank': 'Rich tree bank', 'C_tree_bank_leace': 'Tree bank + LEACE',
           'C_bottleneck': 'C task-only bottleneck', 'D_protected': 'D protected bottleneck',
           'prior': 'Fitting prior', 'exposed': 'Exposed attribute'}
AUDIT_SELECTORS = ('selected', 'independent_selected')
INHERITED_FIELDS = ('fit_pool', 'fit_rows', 'fit_coverage', 'common_warm_adversary_optimizer_steps',
                    'common_warm_adversary_row_exposures', 'continuation_adversary_optimizer_steps',
                    'continuation_adversary_row_exposures', 'total_optimizer_steps', 'total_row_exposures',
                    'total_row_passes_equivalent', 'total_known_label_exposures', 'continuation_schedule_hash',
                    'fit_raw_row_sha256', 'training_label_sha256')


def label(release):
    return DISPLAY.get(release, release)


def selectors(row):
    """Read persisted decisions; never rank candidates by development scores."""
    cid = row['candidate_id']
    # References have no catch-up trajectory: their original primary is also
    # their original independent primary. The runner preserves their scores.
    independent = row.get('independent_selected', row['selected'])
    if row['role'] == 'audit' and cid == 'saved_adversary':
        if row['selected'] or independent or row.get('auc_selected', False):
            raise ValueError('Saved training adversaries must remain diagnostic-only')
        # The raw runner may mark its singleton diagnostic family as selected.
        # Preserve that raw flag on export, without making it a fitted-family arm.
        return ['candidate:saved_adversary']
    if cid == 'catchup' and independent:
        raise ValueError('Catch-up is not one of the five fresh independent candidates')
    result = ['candidate:' + cid]
    if row['selected']:
        result.append('selected')
    if independent and row['role'] == 'audit':
        result.append('independent_selected')
    if row.get('selected_within_family', False):
        result.append('family:' + row['family'])
    if row.get('auc_selected', False):
        result.append('auc_selected')
    return result


def make_index(records):
    index = {}
    for record in records:
        for row in record['raw_metrics']:
            if row['seed'] != record['seed']:
                raise ValueError('Seed mismatch in saved metrics')
            for selector in selectors(row):
                key = (row['seed'], row['role'], row['release'], row['target'], selector)
                if key in index:
                    raise ValueError('Duplicate saved selection: ' + str(key))
                index[key] = row
    return index


def pca_policy(method_utility, pca_utility, bank_residence,
               method_attack, pca_attack, prior_attack, coverage):
    """All parent-relative margins use original PCA, including the protected arm.

    C is an ablation comparator for D, not a replacement parent for the
    preservation or fractional-gain criteria. Fractions remain undefined for
    nonpositive original-PCA headroom or inadequate fixed-schema coverage.
    """
    return {'fixed_parent': 'E_pca', **parent_comparison(
        pca_utility, method_utility, bank_residence, pca_attack, method_attack,
        prior_attack, coverage)}


def paired_rows(records, index):
    """D−C for every task/attribute/family; compact primary reference contrasts."""
    comparisons = [('protected_minus_task_only', 'D_protected', 'C_bottleneck')]
    comparisons += [('learned_minus_reference', method, reference) for method in LEARNED
                    for reference in ('E_pca', 'E_pca_leace', *BANKS)]
    rows = []
    for record in records:
        seed = record['seed']
        for comparison, left, right in comparisons:
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                if role == 'transfer':
                    choices = ('selected', 'family:logistic', 'family:mlp') if comparison == 'protected_minus_task_only' else ('selected',)
                else:
                    choices = (*AUDIT_SELECTORS, 'family:logistic', 'family:mlp', 'family:histgb',
                               'candidate:catchup', 'candidate:saved_adversary', 'auc_selected') if comparison == 'protected_minus_task_only' else AUDIT_SELECTORS
                for target in targets:
                    for selector in choices:
                        a = index.get((seed, role, left, target, selector))
                        b = index.get((seed, role, right, target, selector))
                        for raw_split, split in SPLITS.items():
                            for metric in SCORE_METRICS:
                                av = a.get(raw_split, {}).get(metric) if a else None
                                bv = b.get(raw_split, {}).get(metric) if b else None
                                rows.append({'seed': seed, 'comparison': comparison, 'left': left, 'right': right,
                                    'role': role, 'target': target, 'selector': selector, 'split': split, 'metric': metric,
                                    'left_candidate': a['candidate_id'] if a else None,
                                    'right_candidate': b['candidate_id'] if b else None,
                                    'left_value': av, 'right_value': bv,
                                    'left_minus_right': av - bv if finite(av) and finite(bv) else None})
    return rows


def catchup_diagnostics(records, index, selections=None):
    """Distinguish saved adversary, fresh attack, catch-up, and primary choice."""
    rows = []
    for record in records:
        for method in LEARNED:
            for target in ATTRIBUTES:
                for raw_split, split in SPLITS.items():
                    seed = record['seed']
                    get = lambda selector: get_value(index, seed, 'audit', method, target, raw_split, selector=selector)
                    saved, fresh, catchup, primary = [get(selector) for selector in (
                        'candidate:saved_adversary', 'independent_selected', 'candidate:catchup', 'selected')]
                    prior = get_value(index, seed, 'audit', 'prior', target, raw_split)
                    row = index.get((seed, 'audit', method, target, 'selected'))
                    independent = index.get((seed, 'audit', method, target, 'independent_selected'))
                    result = {'seed': seed, 'release': method, 'target': target, 'split': split,
                        'saved_adversary_log_loss': saved, 'independent_log_loss': fresh,
                        'catchup_log_loss': catchup, 'primary_log_loss': primary,
                        'primary_candidate': row['candidate_id'] if row else None,
                        'independent_candidate': independent['candidate_id'] if independent else None,
                        'prior_log_loss': prior,
                        'primary_attribute_gain': prior - primary if finite(prior) and finite(primary) else None,
                        'catchup_minus_saved': catchup - saved if finite(catchup) and finite(saved) else None,
                        'catchup_minus_independent': catchup - fresh if finite(catchup) and finite(fresh) else None,
                        'primary_minus_independent': primary - fresh if finite(primary) and finite(fresh) else None}
                    if selections is not None:
                        meta = selections[seed]['fitting_records'][f'audit/{method}/{target}']['candidates']['catchup']
                        result.update({'catchup_' + key: meta.get(key) for key in (
                            'fit_rows', 'optimizer_steps', 'training_row_exposures', 'selected_epoch',
                            'selected_optimizer_steps', 'initial_fidelity_exact')})
                        result.update({'inherited_' + key: meta['inherited_exposure'].get(key) for key in INHERITED_FIELDS})
                    rows.append(result)
    return rows


def flatten(records, selections, index):
    """Export every candidate and restart, preserving saved selector flags."""
    scores, classes, curves, fits = [], [], [], []
    for record in records:
        for raw in record['raw_metrics']:
            common = {key: raw.get(key) for key in ('seed', 'role', 'release', 'target', 'candidate_id', 'family',
                                                   'parent', 'erased', 'selected', 'selected_within_family', 'auc_selected')}
            common['independent_selected'] = 'independent_selected' in selectors(raw)
            common['diagnostic_only'] = raw['candidate_id'] == 'saved_adversary'
            meta = candidate_metadata(selections[record['seed']], raw)
            common.update(fit_rows=meta.get('fit_rows'), fit_support=meta.get('fit_support'),
                          fit_coverage_complete=meta.get('fit_coverage_complete'),
                          reused_reference=raw['release'] not in LEARNED)
            fit_fields = ('selected_epoch', 'selected_optimizer_steps', 'optimizer_steps', 'boosting_iterations',
                          'training_row_exposures', 'row_exposure_min', 'row_exposure_max', 'initialization_seed',
                          'schedule_seed', 'schedule_hash', 'restart_index', 'restart_seed_offset', 'fit_runtime_seconds',
                          'fallback_reason', 'warnings', 'parameters', 'checkpoint_criterion', 'selection',
                          'audit_kind', 'provided_attacker_fit_rows', 'provided_attacker_fit_support',
                          'preprocessing', 'preprocessing_fit_rows', 'initial_fidelity_exact')
            inherited = meta.get('inherited_exposure') or {}
            fits.append({**common, **{key: meta.get(key) for key in fit_fields},
                         **{'inherited_' + key: inherited.get(key) for key in INHERITED_FIELDS}})
            for point in meta.get('validation_curve', []):
                curves.append({**common, **point,
                               'checkpoint_selected': point.get('epoch') == meta.get('selected_epoch'),
                               'total_optimizer_steps': meta.get('optimizer_steps'),
                               'total_training_row_exposures': meta.get('training_row_exposures')})
            for raw_split, split in SPLITS.items():
                score = raw[raw_split]
                row = {**common, 'split': split, **{k: v for k, v in score.items() if k != 'per_class'}}
                if raw['role'] == 'audit':
                    prior = get_value(index, raw['seed'], 'audit', 'prior', raw['target'], raw_split)
                    row['prior_log_loss'] = prior
                    row['prior_relative_attribute_gain'] = prior - score['log_loss'] if finite(prior) and finite(score['log_loss']) else None
                scores.append(row)
                fit_support = meta.get('fit_support') or [None] * score['n_classes']
                for cls in score['per_class']:
                    classes.append({**common, 'split': split, **cls,
                        'original_census_code': cls['class_index'] + 1 if raw['role'] == 'audit' else None,
                        'fit_class_support': fit_support[cls['class_index']],
                        'schema_coverage_complete': score['coverage_complete']})
    return scores, classes, curves, fits


def compare_all(records, selections, index):
    criteria, banks, coverage = [], [], []
    for record in records:
        seed = record['seed']
        for split in ('validation', 'test'):
            utility = lambda release: {t: get_value(index, seed, 'transfer', release, t, split) for t in UTILITY_TASKS}
            bank_reference = {b: utility(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')}
            for selector in AUDIT_SELECTORS:
                attack = lambda release: {t: get_value(index, seed, 'audit', release, t, split, selector=selector) for t in ATTRIBUTES}
                cov = {release: {target: audit_coverage(index, selections[seed], seed, release, target, split, selector)
                                 for target in ATTRIBUTES} for release in RELEASES}
                for release, targets in cov.items():
                    for target, value in targets.items():
                        coverage.append({'seed': seed, 'split': SPLITS[split], 'release': release, 'target': target,
                                         'audit_selector': selector, **value})
                for method in LEARNED:
                    policy_cov = {t: combined_coverage(cov[method][t], cov['E_pca'][t]) for t in ATTRIBUTES}
                    policy = pca_policy(utility(method), utility('E_pca'), bank_reference,
                                        attack(method), attack('E_pca'), attack('prior'), policy_cov)
                    criteria.append({'seed': seed, 'split': SPLITS[split], 'release': method,
                                     'audit_selector': selector, 'primary_assessment': selector == 'selected',
                                     'attribute_coverage': policy_cov, **policy})
                    for bank in BANKS:
                        comparison_cov = {t: combined_coverage(cov[method][t], cov[bank][t]) for t in ATTRIBUTES}
                        comparison = feature_bank_comparison(utility(method)['same_residence'], utility(bank)['same_residence'],
                                                             attack(method), attack(bank), attack('prior'), comparison_cov)
                        banks.append({'seed': seed, 'split': SPLITS[split], 'release': method, 'bank': bank,
                                      'audit_selector': selector, 'primary_assessment': selector == 'selected',
                                      'numeric_joint_inequalities': joint_status([comparison['utility']['pass'],
                                          *[v['numeric_inequality'] for v in comparison['attributes'].values()]]), **comparison})
    return criteria, banks, coverage


def criterion_counts(criteria):
    rows = []
    for split in ('validation', 'development_evaluation'):
        for method in LEARNED:
            for selector in AUDIT_SELECTORS:
                subset = [r for r in criteria if r['split'] == split and r['release'] == method and r['audit_selector'] == selector]
                access = {'source_all': lambda r: r['source_preservation']['pass'],
                          **{t: lambda r, t=t: r['source_preservation']['tasks'][t]['pass'] for t in SOURCE_TASKS},
                          'residential_retention': lambda r: r['residential_retention']['pass'],
                          **{t + '_halving': lambda r, t=t: r['attribute_halving'][t]['pass'] for t in ATTRIBUTES},
                          'joint': lambda r: r['joint_pass']}
                for name, fn in access.items():
                    values = [fn(r) for r in subset]
                    rows.append({'split': split, 'release': method, 'audit_selector': selector, 'criterion': name,
                                 'pass': sum(v is True for v in values), 'fail': sum(v is False for v in values),
                                 'undefined': sum(v is None for v in values), 'n_seeds': len(values)})
    return rows


def support_report(out, records, selections, coverage):
    lines = ['# Support and exposed-control competence', '',
             'This stage reuses the original household pools and fixed two-class SEX/nine-class RAC1P schema. '
             'All original test outcomes are DEVELOPMENT EVALUATION. Every candidate’s category support, recall, '
             'precision, F1 and AUROC appears in [PER_CLASS.csv](PER_CLASS.csv). '
             'Class codes below are the original Census codes, not zero-based indices.', '', '## Original pools', '']
    lines += table(['Seed', 'Pool', 'People', 'Households', 'Target', 'Valid', 'Missing/inapplicable', 'Class support'],
                   [[r['seed'], 'development_evaluation' if p == 'test' else p, meta['persons'], meta['households'],
                     t, value['valid'], value['missing_or_inapplicable'], json.dumps(value['class_counts'])]
                    for r in records for p, meta in r['support_by_pool'].items()
                    for group in ('tasks', 'attributes') for t, value in meta[group].items()])
    lines += ['', '## Reused exposed controls: every family and restart', '',
              'The primary exposed control is selected by saved validation log loss. Every failed configuration remains visible. '
              'Positive recall for every class is necessary for a complete attribute assessment; unsupported classes cannot pass.', '']
    rows = []
    for record in records:
        for raw in record['raw_metrics']:
            if raw['release'] != 'exposed':
                continue
            meta = candidate_metadata(selections[record['seed']], raw)
            for split in ('validation', 'test'):
                score = raw[split]
                failed = [c['class_index'] + 1 for c in score['per_class'] if not finite(c['recall']) or c['recall'] <= 0 or c['support'] <= 0]
                rows.append([raw['seed'], SPLITS[split].replace('_', ' '), raw['target'], raw['candidate_id'], raw['selected'],
                             number(score['log_loss']), number(score['accuracy']), number(score['balanced_accuracy']),
                             number(score['macro_auroc'] if raw['target'] == 'RAC1P' else score['auroc']),
                             json.dumps(meta.get('fit_support')), json.dumps(score['support']), json.dumps(failed),
                             ', '.join(number(c['recall']) for c in score['per_class'])])
    lines += table(['Seed', 'Split', 'Target', 'Candidate', 'Primary', 'LL', 'Accuracy', 'Balanced accuracy', 'AUROC / macro',
                    'Fit support', 'Eval support', 'Failing codes', 'Recall by code'], rows)
    lines += ['', '## Primary and independent-only coverage checks', '',
              'Fit support is from the selected candidate’s actual fitting rows. Coverage also requires every category '
              'in attacker validation and evaluated rows plus positive primary exposed-control recall. '
              'Coverage limits apply even when the scalar log-loss inequality is satisfied.', '']
    lines += table(['Seed', 'Split', 'Release', 'Selector', 'Target', 'Candidate', 'Complete', 'Limitations (Census codes)'],
                   [[r['seed'], r['split'].replace('_', ' '), label(r['release']), r['audit_selector'], r['target'],
                     r['audit_candidate'], r['complete'], json.dumps(r['limitation_census_codes'], separators=(',', ':'))]
                    for r in coverage])
    lines += ['', 'Training and catch-up support are recorded in the associated fitting metadata and training evidence; '
              'their history is distinct from the fresh attacker-fit pool. Missing race categories are never merged, '
              'resampled into a new split, or represented as a privacy pass.', '']
    (out/'SUPPORT.md').write_text('\n'.join(lines))


def report_tables(out, records, index, aggregates, gains, pairs, criteria, banks, catchup):
    seeds = [r['seed'] for r in records]
    ai = {(r['role'], r['release'], r['target'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    gi = {(r['release'], r['target'], r['selector'], r['split']): r for r in gains}
    def avg(role, release, target, split, metric='log_loss', selector='selected'):
        return mean_sd(ai.get((role, release, target, selector, split, metric), statistics([None] * len(seeds))))
    lines = ['# ACS bottleneck development evidence', '',
             'This report reads saved metrics and selections only. The original test households informed this research direction; '
             'they are development evidence. Means ± sample SD describe seeds sharing the same cohort. '
             'Natural log loss is primary: lower for utility, higher for less measured attribute recovery. '
             'The primary C/D auditor is the saved validation-log-loss winner among five fresh candidates and catch-up; '
             'the saved training adversary is diagnostic-only. Reference audit scores are reused from the prior frozen stage.', '',
             f"Completed seeds: {', '.join(map(str, seeds))}. [Analysis](ANALYSIS.md), [coverage](SUPPORT.md), "
             '[all candidates](PER_TARGET.csv), [all classes](PER_CLASS.csv), [paired scores](PAIRED.csv), '
             '[catch-up diagnostics](CATCHUP.csv).', '']
    for split in ('validation', 'development_evaluation'):
        lines += [f"## {split.replace('_', ' ').capitalize()}: primary log loss, mean ± sample SD", '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[label(release), *[avg('transfer', release, target, split) for target in UTILITY_TASKS],
                         *[avg('audit', release, target, split) for target in ATTRIBUTES]] for release in RELEASES])
        lines += ['', f"## {split.replace('_', ' ').capitalize()}: every seed", '']
        raw_split = 'validation' if split == 'validation' else 'test'
        lines += table(['Seed', 'Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[seed, label(release), *[number(get_value(index, seed, 'transfer', release, target, raw_split)) for target in UTILITY_TASKS],
                         *[number(get_value(index, seed, 'audit', release, target, raw_split)) for target in ATTRIBUTES]]
                        for seed in seeds for release in RELEASES])
        lines += ['', f"## {split.replace('_', ' ').capitalize()}: PCA-parent criteria", '',
                  'PCA means the unchanged original PCA parent for both learned methods. Source tasks each permit at most +.01 nats. '
                  'Residence must retain half the original PCA advantage over the stronger unprotected rich bank, if that advantage '
                  'is positive. Attribute gain must halve relative to original PCA, when positive and adequately supported. '
                  'Race coverage limits cannot become a pass. A known failure dominates an undefined condition.', '']
        lines += table(['Seed', 'Method', 'Source all', 'Residence retention', 'SEX half gain', 'RAC1P half gain', 'Joint'],
                       [[r['seed'], label(r['release']), status(r['source_preservation']['pass']),
                         status(r['residential_retention']['pass']),
                         *[status(r['attribute_halving'][t]['pass']) for t in ATTRIBUTES], status(r['joint_pass'])]
                        for r in criteria if r['split'] == split and r['primary_assessment']])
        lines += ['']
    lines += ['## Independent, catch-up, and saved-adversary losses', '',
              'Primary selection uses validation only, so primary and independent development rankings may differ. '
              'These rows expose that difference instead of taking the development minimum.', '']
    lines += table(['Seed', 'Split', 'Method', 'Target', 'Saved adversary', 'Fresh independent', 'Catch-up', 'Primary', 'Primary candidate'],
                   [[r['seed'], r['split'].replace('_', ' '), label(r['release']), r['target'],
                     *[number(r[k]) for k in ('saved_adversary_log_loss', 'independent_log_loss', 'catchup_log_loss', 'primary_log_loss')],
                     r['primary_candidate']] for r in catchup if 'person_weighted' not in r['split']])
    lines += ['', 'Full weighted sensitivity, classification metrics, supported/unsupported category scores, '
              'and every candidate/restart are preserved in CSV/JSON. No covariance or accuracy privacy certificate is used.', '']
    (out/'TABLE.md').write_text('\n'.join(lines))

    analysis = ['# ACS bottleneck analysis', '',
                'Saved development evidence only; this report neither fits models nor reselects on development outcomes. '
                'C is the matched task-only training ablation; D adds the fixed protection objective. '
                'All parent-relative margins use original PCA. The original five-task policy and reserved-task boundary remain fixed. '
                'Descriptive seed SDs are not survey uncertainty or a statistical noninferiority test.', '',
                '[Primary tables](TABLE.md), [support and controls](SUPPORT.md), [all candidates](PER_TARGET.csv), '
                '[classes](PER_CLASS.csv), [paired metrics](PAIRED.csv), [training curves](TRAINING_CURVES.csv), '
                '[attacker curves](CURVES.csv), [catch-up](CATCHUP.csv), [criteria](criteria.json), '
                '[bank comparisons](bank_comparisons.json), [all aggregates](summary.json).', '',
                'The primary six-candidate audit and five-candidate independent audit are distinct. '
                'The saved adversary receives no primary/independent selector. A catch-up audit can lower validation loss '
                'and still have higher development loss; the persisted choice is retained. Prior-relative gains remain signed.', '']
    for split in ('validation', 'development_evaluation'):
        analysis += [f"## {split.replace('_', ' ').capitalize()}: exact PCA-parent comparisons", '']
        analysis += table(['Seed', 'Method', 'Audit selector', 'Source LL differences (income / work / coverage)',
                           'Residence headroom before → after', 'Residence retained', 'SEX gain before → after', 'RAC1P gain before → after', 'Joint'],
                          [[r['seed'], label(r['release']), r['audit_selector'],
                            ' / '.join(number(r['source_preservation']['tasks'][t]['difference']) for t in SOURCE_TASKS),
                            number(r['residential_retention']['parent_headroom']) + ' → ' + number(r['residential_retention']['erased_headroom']),
                            number(r['residential_retention']['retained_fraction']),
                            *[number(r['attribute_halving'][t]['parent_attribute_gain']) + ' → ' + number(r['attribute_halving'][t]['erased_attribute_gain']) for t in ATTRIBUTES],
                            status(r['joint_pass'])] for r in criteria if r['split'] == split])
        analysis += ['', f"## {split.replace('_', ' ').capitalize()}: rich-bank margins by seed", '',
                     'Utility requires .01 lower residence loss. Each attribute permits at most .005 extra signed gain. '
                     'Numeric columns publish the raw inequalities even when coverage prevents policy assessment.', '']
        analysis += table(['Seed', 'Method', 'Bank', 'Selector', 'Residence LL difference', 'Utility', 'SEX numeric', 'RAC1P numeric', 'Numeric joint', 'Policy joint'],
                          [[r['seed'], label(r['release']), label(r['bank']), r['audit_selector'], number(r['utility']['feature_minus_bank']),
                            status(r['utility']['pass']), *[status(r['attributes'][t]['numeric_inequality']) for t in ATTRIBUTES],
                            status(r['numeric_joint_inequalities']), status(r['joint_pass'])] for r in banks if r['split'] == split])
        analysis += ['', f"## {split.replace('_', ' ').capitalize()}: signed gains and independent audit sensitivity", '']
        analysis += table(['Release', 'Selector', 'SEX gain', 'RAC1P gain', 'SEX attack LL', 'RAC1P attack LL'],
                          [[label(release), selector,
                            *[mean_sd(gi[release, t, selector, split]) for t in ATTRIBUTES],
                            *[avg('audit', release, t, split, selector=selector) for t in ATTRIBUTES]]
                           for release in RELEASES for selector in AUDIT_SELECTORS])
        analysis += ['']
    analysis += ['## Paired D − C: every task and attribute', '',
                 'Positive loss differences reduce task utility and reduce measured attribute recovery. '
                 'Within-family choices are saved validation selections. Catch-up and training-adversary pairs are diagnostics.', '']
    analysis += table(['Split', 'Role', 'Target', 'Selector', 'D − C log loss'],
                      [[r['split'].replace('_', ' '), r['role'], TASK_DISPLAY[r['target']], r['selector'], mean_sd(r)]
                       for r in pairs if r['comparison'] == 'protected_minus_task_only' and r['metric'] == 'log_loss'
                       and 'person_weighted' not in r['split']])
    analysis += ['', '## Every primary metric and person-weighted sensitivity', '',
                 'PWGTP sensitivity uses identical predictions and the same unweighted validation selection. '
                 'Full-schema macro AUROC/balanced accuracy remain undefined with absent categories; '
                 'observed-class alternatives are separately named in CSV/JSON.', '']
    for split in SPLITS.values():
        analysis += [f"### {split.replace('_', ' ').capitalize()}", '']
        analysis += table(['Release', 'Target', 'Log loss', 'Accuracy', 'Balanced accuracy', 'AUROC / macro'],
                          [[label(release), TASK_DISPLAY[t], *[avg(role, release, t, split, metric)
                            for metric in ('log_loss', 'accuracy', 'balanced_accuracy', 'macro_auroc' if t == 'RAC1P' else 'auroc')]]
                           for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)) for t in targets for release in RELEASES])
        analysis += ['']
    analysis += ['## Scope', '',
                 'The five-task policy is unchanged. Missing race support is an assessment limit, not protection. '
                 'Task-only C versus protected D tests the contribution of this standard protection objective under the fixed recipe. '
                 'Residual recovery and utility losses must be considered together. No PCRL novelty, universal privacy, '
                 'coalition protection, official Census estimate, or untouched-household confirmation follows from these measurements.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))


def flatten_training(training):
    curves, accounting = [], []
    for seed, meta in training.items():
        def curve_rows(phase, release, curve):
            for point in curve:
                row = {'seed': seed, 'phase': phase, 'release': release}
                for key, value in point.items():
                    if isinstance(value, dict):
                        row.update({key + '_' + target: v for target, v in value.items()})
                    else:
                        row[key] = value
                curves.append(row)
        for phase, curve in meta['shared_curves'].items():
            curve_rows(phase, 'shared', curve)
        for release, arm in meta['arms'].items():
            curve_rows('continuation', release, arm['curve'])
            accounting.append({'seed': seed, 'release': release, 'fit_pool': meta['fit_pool'],
                'fit_rows': meta['preprocessing']['fit_rows'], 'attribute_fit_coverage': meta['attribute_fit_coverage'],
                'source_fit_coverage': meta['source_fit_coverage'], 'prior_entropies': meta['prior_entropies'],
                'common_base_row_exposures': meta['common_base_row_exposures'],
                'common_adversary_row_exposures': meta['common_adversary_row_exposures'],
                'common_source_valid_label_exposures': meta.get('common_source_valid_label_exposures'),
                'common_attribute_valid_label_exposures': meta.get('common_attribute_valid_label_exposures'),
                'continuation_mapper_row_exposures': arm['mapper_row_exposures'],
                'continuation_adversary_row_exposures': arm['adversary_row_exposures'],
                'continuation_source_valid_label_exposures': arm.get('source_valid_label_exposures'),
                'continuation_attribute_valid_label_exposures': arm.get('attribute_valid_label_exposures'),
                'total_mapper_row_exposures': meta['common_base_row_exposures'] + arm['mapper_row_exposures'],
                'total_adversary_row_exposures_before_catchup': meta['common_adversary_row_exposures'] + arm['adversary_row_exposures'],
                'total_mapper_optimizer_steps': arm['optimizer_counts_including_common']['mapper_optimizer_steps'],
                'total_adversary_optimizer_steps_before_catchup': arm['optimizer_counts_including_common']['adversary_optimizer_steps'],
                'selected_epoch': arm['selected_epoch'], 'selection': arm['selection'],
                'fork_equal_to_shared': arm['fork_hashes'] == meta['shared_fork_hashes'],
                'schedule_hash': arm['schedule_hash'], 'empty_minibatch_heads': arm['empty_minibatch_heads'],
                'first_batch_gradient_diagnostics_at_shared_fork': arm['first_batch_gradient_diagnostics_at_shared_fork'],
                'potential_protection_mapper_l2_at_shared_fork': arm.get('potential_protection_mapper_l2_at_shared_fork'),
                'applied_protection_mapper_l2_at_shared_fork': arm.get('applied_protection_mapper_l2_at_shared_fork'),
                'mapper_loss_uses_protection_gradient': arm.get('mapper_loss_uses_protection_gradient'),
                'reserved_labels_received': meta['reserved_labels_received'],
                'source_validation_role': meta['source_validation_role']})
    return curves, accounting


def append_training_report(out, training, accounting, records, fits):
    lines = ['', '## Training phases and exposure accounting', '',
             'Training curves are diagnostics on representation-fitting and source-validation rows. '
             'They do not select a checkpoint, and they do not contain reserved-task labels. '
             'C/D use the fixed final continuation epoch. Step and exposure totals below include shared warm-up. '
             'Shared work is inherited by both arms; totals are model histories, not duplicated runtime claims. '
             'Each adversary update processes both attribute networks on the same minibatch. '
             '[Full curves](TRAINING_CURVES.csv) and [training accounting](TRAINING.csv) retain phase-specific details.', '']
    lines += table(['Seed', 'Method', 'Mapper steps', 'Adversary steps before catch-up', 'Mapper row exposures',
                    'Adversary row exposures before catch-up', 'Final epoch', 'Exact shared fork'],
                   [[r['seed'], label(r['release']), r['total_mapper_optimizer_steps'],
                     r['total_adversary_optimizer_steps_before_catchup'], r['total_mapper_row_exposures'],
                     r['total_adversary_row_exposures_before_catchup'], r['selected_epoch'], r['fork_equal_to_shared']] for r in accounting])
    lines += ['', 'Catch-up resets Adam and adds attacker-pool training after the inherited representation-pool history. '
              'The saved adversary performs zero new fitting updates. Its provided attacker-fit arrays serve fidelity/scoring checks, '
              'not an additional fitting phase. Fresh independent MLPs have no inherited warm-up history.', '']
    lines += table(['Seed', 'Method', 'Target', 'Candidate', 'New fitting rows', 'New updates', 'New row exposures',
                    'Selected epoch', 'Identity-input fidelity'],
                   [[r['seed'], label(r['release']), r['target'], r['candidate_id'], r['fit_rows'],
                     r['optimizer_steps'], r['training_row_exposures'], r['selected_epoch'], r['initial_fidelity_exact']]
                    for r in fits if r['release'] in LEARNED and r['candidate_id'] in ('catchup', 'saved_adversary')])
    lines += ['', '## Fixed-final training diagnostics', '',
              'These source losses are from the jointly trained source heads, separate from the independently fitted '
              'utility heads in the primary tables. Reconstruction is standardized-PCA MSE. '
              'Attribute cross-entropy here is from the current training adversary on fitting rows, not a held-out privacy result.', '']
    lines += table(['Seed', 'Method', 'Fit source CE', 'Source-validation CE', 'Fit reconstruction MSE',
                    'Fit SEX adversary CE', 'Fit RAC1P adversary CE', 'Normalized mean adversary CE'],
                   [[seed, label(release), *[number(arm['curve'][-1][key]) for key in (
                       'fit_source_ce', 'source_validation_ce', 'fit_reconstruction_mse')],
                     *[number(arm['curve'][-1]['fit_adversary_ce'][target]) for target in ATTRIBUTES],
                     number(arm['curve'][-1]['fit_normalized_adversary_ce'])]
                    for seed, meta in training.items() for release, arm in meta['arms'].items()])
    lines += ['', '## Training support and gradient checks', '',
              'Training support is reported separately from attacker support. Inherited exposure to a rare category does not '
              'replace missing support in the fixed independent attack/validation pools. Gradient norms are measured on '
              'the first shared continuation minibatch; they are not a record of every optimizer update.', '']
    lines += table(['Seed', 'Target', 'Training support', 'Missing rows', 'Schema complete', 'Fitting entropy'],
                   [[seed, target, json.dumps(meta['attribute_fit_coverage'][target]['support']),
                     meta['attribute_fit_coverage'][target]['missing'], meta['attribute_fit_coverage'][target]['schema_complete'],
                     number(meta['prior_entropies'][target]['entropy'])]
                    for seed, meta in training.items() for target in ATTRIBUTES])
    lines += ['', 'The potential protection gradient measures the common objective path; C applies zero protection '
              'gradient and only D applies the potential term.', '']
    lines += table(['Seed', 'Method', 'Base mapper gradient L2', 'Potential protection gradient L2', 'Applied protection gradient L2',
                    'Source mapper gradient L2', 'Weighted reconstruction mapper gradient L2'],
                   [[seed, label(release), number(arm['first_batch_gradient_diagnostics_at_shared_fork']['base_mapper_l2']),
                     number(arm['potential_protection_mapper_l2_at_shared_fork']), number(arm['applied_protection_mapper_l2_at_shared_fork']),
                     *[number(arm['first_batch_gradient_diagnostics_at_shared_fork'][key]) for key in ('source_mapper_l2', 'reconstruction_mapper_l2')]]
                    for seed, meta in training.items() for release, arm in meta['arms'].items()])
    lines += ['', '## Recorded internal runtime', '']
    phase_names = ('total_seconds', 'reference_loading_seconds', 'representation_training_seconds',
                   'reference_prediction_verification_seconds', 'utility_fitting_seconds', 'independent_audit_seconds',
                   'catchup_seconds', 'evaluation_serialization_integrity_seconds')
    lines += table(['Seed', 'Total s', 'Reference load s', 'Training s', 'Reuse verification s', 'Utility s', 'Fresh audit s', 'Catch-up s', 'Evaluation s'],
                   [[r['seed'], *[number(r['runtime'].get(key)) for key in phase_names]] for r in records])
    lines += ['', f"Sum of internal seed runtimes: {number(sum(r['runtime']['total_seconds'] for r in records))} seconds. "
              'The orchestrator records process wall time separately. All states and historical references remain frozen during auditing.', '']
    with (out/'ANALYSIS.md').open('a') as handle:
        handle.write('\n'.join(lines))
    support_lines = ['', '## Representation-fitting attribute support', '']
    support_lines += table(['Seed', 'Target', 'Support', 'Missing', 'Schema complete'],
                           [[seed, target, json.dumps(meta['attribute_fit_coverage'][target]['support']),
                             meta['attribute_fit_coverage'][target]['missing'], meta['attribute_fit_coverage'][target]['schema_complete']]
                            for seed, meta in training.items() for target in ATTRIBUTES])
    support_lines += ['', 'No training-category support gap is repaired by changing the cohort or schema. '
                      'All policy criteria still require adequate fixed audit-pool and exposed-control coverage.', '']
    with (out/'SUPPORT.md').open('a') as handle:
        handle.write('\n'.join(support_lines))


def plot_tradeoffs(out, records, aggregates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    ai = {(r['role'], r['release'], r['target'], r['split']): r for r in aggregates
          if r['selector'] == 'selected' and r['metric'] == 'log_loss'}
    pairs = [('E_pca', 'E_pca_leace', '#73569a'), ('B_rich_bank', 'B_rich_bank_leace', '#b47e38'),
             ('C_tree_bank', 'C_tree_bank_leace', '#609768'), ('C_bottleneck', 'D_protected', '#245b8c')]
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
        for ax, split, title in zip(axes, ('validation', 'development_evaluation'), ('Validation', 'Development evaluation')):
            def point(release, color, marker):
                x = ai.get(('transfer', release, 'same_residence', split), {})
                y = ai.get(('audit', release, target, split), {})
                if not finite(x.get('mean')) or not finite(y.get('mean')):
                    return None
                xy = (x['mean'], y['mean'])
                ax.errorbar(*xy, xerr=x.get('sample_sd') or 0, yerr=y.get('sample_sd') or 0,
                            color=color, marker=marker, capsize=2, markersize=6, linewidth=1)
                return xy
            for left, right, color in pairs:
                a, b = point(left, color, 'o'), point(right, color, 'D')
                if a and b:
                    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.5})
            point('prior', '#777777', 'X')
            ax.set_title(title)
            ax.set_xlabel('Same-residence log loss (lower is better)')
            ax.set_ylabel(f'{target} attack log loss (higher: less measured recovery)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='both', style='plain', useOffset=False)
            ax.grid(alpha=.2)
            ax.margins(.1)
        labels = ('PCA → PCA + LEACE (32D)', 'Neural bank → + LEACE (26D)',
                  'Tree bank → + LEACE (26D)', 'C task-only → D protected (16D)')
        handles = [Line2D([0], [0], color=color, marker='o', label=name) for (_, _, color), name in zip(pairs, labels)]
        handles += [Line2D([0], [0], color='#777777', marker='X', linestyle='', label='Fitting prior'),
                    Line2D([0], [0], color='#555555', marker='D', linestyle='', label='Arrow destination')]
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.7, bbox_to_anchor=(.5, .01))
        note = 'Mean ± sample SD; bounded empirical audit, not a privacy certificate.'
        if target == 'RAC1P':
            note = 'Mean ± sample SD; race-category coverage incomplete; no all-attribute privacy claim.'
        fig.text(.5, .13, note, ha='center', fontsize=8.5)
        fig.suptitle(f'Fixed nonlinear bottleneck: residential utility and {target} recovery', fontsize=12)
        fig.tight_layout(rect=(0, .18, 1, .95))
        fig.savefig(out/f'tradeoff_{target}.png', dpi=200)
        fig.savefig(out/f'tradeoff_{target}.pdf')
        plt.close(fig)


def plot_training(out, training, selections, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    colors = {'C_bottleneck': '#b2793a', 'D_protected': '#326b9f'}
    metrics = [('source_validation_ce', 'Source-validation CE'), ('fit_source_ce', 'Fitting source CE'),
               ('fit_reconstruction_mse', 'Fitting reconstruction MSE'),
               ('fit_adversary_ce_SEX', 'Fitting SEX adversary CE'), ('fit_adversary_ce_RAC1P', 'Fitting RAC1P adversary CE'),
               ('fit_normalized_adversary_ce', 'Fitting normalized adversary CE')]
    curves, _ = flatten_training(training)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for release in LEARNED:
            grouped = defaultdict(list)
            for row in curves:
                if row['release'] == release and row['phase'] == 'continuation':
                    grouped[row['epoch']].append(row.get(metric))
            xs = sorted(grouped)
            stats = [statistics(grouped[x]) for x in xs]
            ys = np.array([r['mean'] for r in stats], dtype=float)
            sd = np.array([r['sample_sd'] or 0. for r in stats])
            ax.plot(xs, ys, color=colors[release], label=label(release))
            ax.fill_between(xs, ys - sd, ys + sd, color=colors[release], alpha=.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Continuation epoch (fixed final iterate)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Training diagnostics: mean ± sample SD; no checkpoint selection', fontsize=12)
    fig.tight_layout(rect=(0, .02, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'training_curves.{suffix}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    for rownum, target in enumerate(ATTRIBUTES):
        for colnum, release in enumerate(LEARNED):
            ax = axes[rownum, colnum]
            grouped = defaultdict(list)
            selected_points = []
            for seed, selection in selections.items():
                meta = selection['fitting_records'][f'audit/{release}/{target}']['candidates']['catchup']
                for point in meta['validation_curve']:
                    grouped[point['epoch']].append(point['validation_log_loss'])
                selected_points.append((meta['selected_epoch'], meta['validation_scores']['log_loss']))
            xs = sorted(grouped)
            stats = [statistics(grouped[x]) for x in xs]
            ys = np.array([r['mean'] for r in stats])
            sd = np.array([r['sample_sd'] or 0 for r in stats])
            ax.plot(xs, ys, color=colors[release], label='Catch-up mean')
            ax.fill_between(xs, ys - sd, ys + sd, color=colors[release], alpha=.15, label='Sample SD')
            ax.scatter(*zip(*selected_points), marker='x', color=colors[release], s=25, label='Per-seed selected epoch', zorder=3)
            independent = statistics(get_value(index, seed, 'audit', release, target, 'validation', selector='independent_selected') for seed in selections)
            if finite(independent['mean']):
                ax.axhline(independent['mean'], linestyle='--', color='#555555', label='Fresh independent primary mean')
            ax.set_title(f'{label(release)}: {target}', fontsize=10)
            ax.set_xlabel('Catch-up epoch; saved weights at epoch 0')
            ax.set_ylabel('Attacker-validation log loss')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle('Frozen-release catch-up: inherited weights, reset Adam, validation selection', fontsize=11)
    fig.text(.5, .01, 'Catch-up has additional inherited training exposure; these are not equal-lifetime comparisons.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .035, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'catchup_curves.{suffix}', dpi=200)
    plt.close(fig)


def summarize(out):
    out = Path(out).resolve()
    paths = sorted(out.glob('seed_*/metrics.json'), key=lambda p: int(p.parent.name.removeprefix('seed_')))
    if not paths:
        raise ValueError('No completed seed metrics; this report performs no fitting')
    records = [read_json(p) for p in paths]
    seeds = [r['seed'] for r in records]
    config = read_json(out/'config.json')
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config['seeds']):
        raise ValueError('Duplicate or undeclared completed seed')
    selections = {r['seed']: read_json(p.parent/'selection_before_test.json') for r, p in zip(records, paths)}
    training = {r['seed']: read_json(p.parent/'training/training.json') for r, p in zip(records, paths)}
    index = make_index(records)
    scores, classes, curves, fits = flatten(records, selections, index)
    training_curves, training_accounting = flatten_training(training)
    criteria, banks, coverage = compare_all(records, selections, index)
    paired = paired_rows(records, index)
    aggregates = aggregate_metrics(records, index)
    gains = aggregate_attribute_gains(records, index)
    pair_aggregates = aggregate_pairs(paired, seeds)
    catchup = catchup_diagnostics(records, index, selections)
    for filename, rows in (('PER_TARGET.csv', scores), ('PER_CLASS.csv', classes), ('CURVES.csv', curves),
                           ('FITTING.csv', fits), ('TRAINING_CURVES.csv', training_curves),
                           ('TRAINING.csv', training_accounting), ('PAIRED.csv', paired), ('CATCHUP.csv', catchup)):
        save_csv(out/filename, rows)
    save_json(out/'criteria.json', {'fixed_parent': 'E_pca', 'margins': MARGINS,
              'per_seed_split': criteria, 'counts': criterion_counts(criteria)})
    save_json(out/'bank_comparisons.json', {'margins': MARGINS, 'per_seed_split': banks})
    source_paths = [Path(__file__), Path(__file__).with_name('summarize_acs_protection.py')]
    input_paths = [out/'config.json', out/'PROTOCOL.md', *paths,
                   *[p.parent/'selection_before_test.json' for p in paths], *[p.parent/'training/training.json' for p in paths]]
    summary = {'evaluation_status': config['evaluation_status'], 'completed_seeds': seeds,
        'declared_seeds': config['seeds'], 'all_declared_seeds_complete': set(seeds) == set(config['seeds']),
        'scorer': {'primary': 'unweighted natural log loss; p floor 1e-12 then renormalize',
                   'primary_audit': 'persisted validation winner among five fresh independent candidates plus learned-arm catch-up',
                   'independent_audit': 'persisted validation winner among matched five fresh candidates only',
                   'saved_adversary': 'diagnostic-only, never an eligible primary candidate',
                   'gain': 'signed fitting-prior log loss minus attack log loss',
                   'uncertainty': 'sample SD over shared-cohort seeds, not population uncertainty',
                   'undefined': 'null, never averaged away or counted as protection'},
        'margins': MARGINS, 'fixed_parent': 'E_pca', 'aggregate_metrics': aggregates,
        'attribute_gain_aggregates': gains, 'paired_aggregate_metrics': pair_aggregates,
        'criterion_counts': criterion_counts(criteria), 'coverage': coverage,
        'catchup_diagnostics': catchup, 'training_accounting': training_accounting,
        'release_metadata_paths': {r['seed']: f"seed_{r['seed']}/release_freeze.json" for r in records},
        'support_paths': {r['seed']: f"seed_{r['seed']}/support.json" for r in records},
        'training_paths': {r['seed']: f"seed_{r['seed']}/training/training.json" for r in records},
        'runtime': {r['seed']: r['runtime'] for r in records},
        'internal_runtime_total_seconds': sum(r['runtime']['total_seconds'] for r in records),
        'integrity': {r['seed']: r['integrity'] for r in records},
        'primary_candidate_counts': dict(Counter(row['candidate_id'] for r in records for row in r['raw_metrics'] if row['selected'])),
        'row_counts': {'per_target': len(scores), 'per_class': len(classes), 'curves': len(curves), 'fitting': len(fits),
                       'training_curves': len(training_curves), 'training_arms': len(training_accounting),
                       'paired': len(paired), 'catchup': len(catchup)},
        'report_source_sha256': {str(p.name): sha(p) for p in source_paths},
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in input_paths}}
    save_json(out/'summary.json', summary)
    report_tables(out, records, index, aggregates, gains, pair_aggregates, criteria, banks, catchup)
    support_report(out, records, selections, coverage)
    append_training_report(out, training, training_accounting, records, fits)
    plot_tradeoffs(out, records, aggregates)
    plot_training(out, training, selections, index)
    print(json.dumps({'out': str(out), 'seeds': seeds, **summary['row_counts'],
                      'internal_runtime_seconds': summary['internal_runtime_total_seconds']}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)


if __name__ == '__main__':
    main()

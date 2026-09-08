"""Read-only stage reporting for the fixed PCA16-initialized bottleneck.

Usage: python scripts/summarize_acs_pca16_init.py --out RESULTS_DIRECTORY
No fitting, prediction arrays, or outcome-based reselection. Initialization I
and warm state W have utility probes only; their attributes are not audited.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, SPLITS, SCORE_METRICS, TASK_DISPLAY,
    source_preservation, residential_retention, audit_coverage, get_score,
    get_value, candidate_metadata, statistics, number, finite, status, table,
    read_json, save_json, save_csv, sha, aggregate_pairs, parent_comparison,
    feature_bank_comparison, combined_coverage, joint_status,
)
from scripts.summarize_acs_bottleneck import make_index as original_index, INHERITED_FIELDS

NEW = ('I', 'W', 'C_init', 'D_init')
FINAL = ('C_init', 'D_init')
LEARNED_AUDITS = (*FINAL, 'C_bottleneck', 'D_protected')
RELEASES = ('PCA16', *NEW, 'C_bottleneck', 'D_protected', 'E_pca', 'E_pca_leace',
            'B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace', 'prior')
LABELS = {'I': 'I initialized mapper', 'W': 'W after common warm-up', 'C_init': 'C initialized task-only',
          'D_init': 'D initialized protected', 'C_bottleneck': 'Historical C task-only',
          'D_protected': 'Historical D protected', 'PCA16': 'Frozen PCA16', 'E_pca': 'Original PCA32',
          'E_pca_leace': 'PCA32 + LEACE', 'B_rich_bank': 'Rich neural bank',
          'B_rich_bank_leace': 'Neural bank + LEACE', 'C_tree_bank': 'Rich tree bank',
          'C_tree_bank_leace': 'Tree bank + LEACE', 'prior': 'Fitting prior'}
STAGE_PAIRS = (('initialization', 'I', 'PCA16'), ('warmup', 'W', 'I'),
               ('task_continuation', 'C_init', 'W'), ('protection', 'D_init', 'C_init'),
               ('task_initialization_change', 'C_init', 'C_bottleneck'),
               ('protected_initialization_change', 'D_init', 'D_protected'))
BANKS = ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')


def precise(value):
    """Preserve tiny nonzero identity differences instead of printing zero."""
    if finite(value) and 0 < abs(value) < 1e-6:
        return f'{value:.3e}'
    return number(value)


def mean_sd(record):
    if not record['complete']:
        return f"undefined ({record['n_defined']}/{record['n_seeds']})"
    return f"{precise(record['mean'])} ± {precise(record['sample_sd'])}"


def make_index(records):
    index = original_index(records)
    for seed, role, release, target in {key[:4] for key in index}:
        if role == 'audit' and release in ('I', 'W'):
            raise ValueError('I/W have no attribute audits in this protocol')
        selector = 'independent_selected' if role == 'audit' and release in LEARNED_AUDITS else 'selected'
        row = index.get((seed, role, release, target, selector))
        if row is not None:
            index[seed, role, release, target, 'primary'] = row
        if role == 'audit' and release in LEARNED_AUDITS:
            row = index.get((seed, role, release, target, 'selected'))
            if row is not None:
                index[seed, role, release, target, 'catchup_inclusive'] = row
    return index


def utility_criteria(method, pca32, bank_residence):
    return {'fixed_parent': 'E_pca', 'source_preservation': source_preservation(pca32, method),
            'residential_retention': residential_retention(pca32.get('same_residence'), method.get('same_residence'),
                bank_residence.get('B_rich_bank'), bank_residence.get('C_tree_bank')),
            'task_minus_pca32': {t: method[t] - pca32[t] if finite(method.get(t)) and finite(pca32.get(t)) else None
                                for t in UTILITY_TASKS},
            'scope': 'unchanged descriptive source/residence references; no all-attribute admission gate'}


def final_criteria(seeds, index, selections):
    policies, banks = [], []
    for seed in seeds:
        for raw_split, split in (('validation', 'validation'), ('test', 'development_evaluation')):
            utility = lambda release: {t: get_value(index, seed, 'transfer', release, t, raw_split, selector='primary') for t in UTILITY_TASKS}
            bank_residence = {b: utility(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')}
            for selector in ('primary', 'catchup_inclusive'):
                def attack(release):
                    chosen = selector if release in FINAL else 'primary'
                    return {t: get_value(index, seed, 'audit', release, t, raw_split, selector=chosen) for t in ATTRIBUTES}
                def coverage(release, target):
                    return audit_coverage(index, selections[seed], seed, release, target, raw_split,
                                          selector if release in FINAL else 'primary')
                for release in FINAL:
                    cov = {t: combined_coverage(coverage(release, t), coverage('E_pca', t)) for t in ATTRIBUTES}
                    policy = parent_comparison(utility('E_pca'), utility(release), bank_residence,
                                               attack('E_pca'), attack(release), attack('prior'), cov)
                    policies.append({'seed': seed, 'split': split, 'release': release, 'audit_selector': selector,
                                     'fixed_parent': 'E_pca', 'attribute_coverage': cov, **policy})
                    for bank in BANKS:
                        cov = {t: combined_coverage(coverage(release, t), coverage(bank, t)) for t in ATTRIBUTES}
                        comparison = feature_bank_comparison(utility(release)['same_residence'], utility(bank)['same_residence'],
                                                             attack(release), attack(bank), attack('prior'), cov)
                        banks.append({'seed': seed, 'split': split, 'release': release, 'bank': bank,
                                      'audit_selector': selector,
                                      'numeric_joint_inequalities': joint_status([comparison['utility']['pass'],
                                          *[v['numeric_inequality'] for v in comparison['attributes'].values()]]), **comparison})
    return policies, banks


def paired_rows(seeds, index):
    rows = []
    for seed in seeds:
        for comparison, left, right in STAGE_PAIRS:
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                if role == 'audit' and (left not in LEARNED_AUDITS or right not in LEARNED_AUDITS):
                    continue
                choices = ('primary', 'family:logistic', 'family:mlp')
                if role == 'audit':
                    choices += ('family:histgb', 'catchup_inclusive')
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
                                    'left_candidate': a['candidate_id'] if a else None, 'right_candidate': b['candidate_id'] if b else None,
                                    'left_value': av, 'right_value': bv,
                                    'left_minus_right': av - bv if finite(av) and finite(bv) else None})
    return rows


def flatten_new(records, selections):
    scores, classes, curves, fits = [], [], [], []
    for record in records:
        for raw in record['raw_metrics']:
            if raw['release'] not in NEW:
                continue
            meta = candidate_metadata(selections[record['seed']], raw)
            common = {k: raw.get(k) for k in ('seed', 'role', 'release', 'target', 'candidate_id', 'family',
                                            'selected', 'independent_selected', 'selected_within_family', 'auc_selected')}
            common.update(diagnostic_only=raw['candidate_id'] == 'saved_adversary',
                          fit_rows=meta.get('fit_rows'), fit_support=meta.get('fit_support'),
                          fit_coverage_complete=meta.get('fit_coverage_complete'))
            inherited = meta.get('inherited_exposure') or {}
            fits.append({**common, **{k: meta.get(k) for k in (
                'optimizer_steps', 'selected_epoch', 'selected_optimizer_steps', 'training_row_exposures',
                'schedule_hash', 'initialization_seed', 'schedule_seed', 'restart_index', 'parameters',
                'fit_runtime_seconds', 'warnings', 'fallback_reason', 'initial_fidelity_exact')},
                **{'inherited_' + k: inherited.get(k) for k in INHERITED_FIELDS}})
            curves.extend({**common, **point, 'checkpoint_selected': point['epoch'] == meta.get('selected_epoch')}
                          for point in meta.get('validation_curve', []))
            for raw_split, split in SPLITS.items():
                score = raw[raw_split]
                scores.append({**common, 'split': split, **{k: v for k, v in score.items() if k != 'per_class'}})
                fit_support = meta.get('fit_support') or [None] * score['n_classes']
                for cls in score['per_class']:
                    classes.append({**common, 'split': split, **cls,
                        'original_census_code': cls['class_index'] + 1 if raw['role'] == 'audit' else None,
                        'fit_class_support': fit_support[cls['class_index']],
                        'schema_coverage_complete': score['coverage_complete']})
    return scores, classes, curves, fits


def aggregate(seeds, index):
    result = []
    for release in RELEASES:
        for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
            if role == 'audit' and release in ('I', 'W'):
                continue
            choices = ('primary', 'catchup_inclusive') if role == 'audit' and release in LEARNED_AUDITS else ('primary',)
            for target in targets:
                for selector in choices:
                    for raw_split, split in SPLITS.items():
                        for metric in SCORE_METRICS:
                            values = [get_value(index, seed, role, release, target, raw_split, metric, selector) for seed in seeds]
                            result.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                           'split': split, 'metric': metric, **statistics(values)})
                        if role == 'audit':
                            values = []
                            for seed in seeds:
                                prior = get_value(index, seed, role, 'prior', target, raw_split, selector='primary')
                                attack = get_value(index, seed, role, release, target, raw_split, selector=selector)
                                values.append(prior - attack if finite(prior) and finite(attack) else None)
                            result.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                           'split': split, 'metric': 'signed_prior_relative_gain', **statistics(values)})
    return result


def write_reports(out, seeds, index, aggregates, pairs, paired, criteria, coverage, policies, bank_comparisons):
    ai = {(r['release'], r['role'], r['target'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    def avg(release, role, target, split, metric='log_loss', selector='primary'):
        if role == 'audit' and release in ('I', 'W'):
            return 'not audited'
        return mean_sd(ai[release, role, target, selector, split, metric])
    intro = ['# PCA16-initialized bottleneck — stage evidence', '',
             'These are development results on the original households, not untouched confirmation. '
             'I is the initialized mapper, W the common warm state, C_init the final task-only continuation, '
             'and D_init the final protected continuation. Each stage has independent utility probes after all training '
             'finishes. I/W have no attribute audits. Primary attribute comparisons use matched five-candidate '
             'independent audits; catch-up-inclusive selections remain separate. No saved selection is recomputed '
             'from development outcomes. Means ± sample SD describe seeds sharing a cohort.', '',
             f"Completed seeds: {', '.join(map(str, seeds))}. [Analysis](ANALYSIS.md), [paired metrics](PAIRED.csv), "
             '[new candidate scores](PER_TARGET.csv), [new classes](PER_CLASS.csv), [new fitting curves](CURVES.csv), '
             '[PCA32 criteria](criteria.json), [bank comparisons](bank_comparisons.json).', '']
    lines = intro.copy()
    for split in ('validation', 'development_evaluation'):
        lines += [f"## {split.replace('_', ' ').capitalize()}: primary log loss, mean ± sample SD", '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[LABELS[r], *[avg(r, 'transfer', t, split) for t in UTILITY_TASKS],
                         *[avg(r, 'audit', t, split) for t in ATTRIBUTES]] for r in RELEASES])
        lines += ['', f"## {split.replace('_', ' ').capitalize()}: every seed, new stages", '']
        raw_split = 'validation' if split == 'validation' else 'test'
        lines += table(['Seed', 'Stage', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[seed, LABELS[r], *[number(get_value(index, seed, 'transfer', r, t, raw_split, selector='primary')) for t in UTILITY_TASKS],
                         *['not audited' if r in ('I', 'W') else number(get_value(index, seed, 'audit', r, t, raw_split, selector='primary')) for t in ATTRIBUTES]]
                        for seed in seeds for r in NEW])
        lines += ['', f"## {split.replace('_', ' ').capitalize()}: residence stage differences", '',
                  'Positive differences lose utility. Tiny nonzero differences are shown in scientific notation. '
                  'These comparisons localize observed changes; they do not select a stage.', '']
        lines += table(['Contrast (left − right)', 'Mean ± sample SD'],
                       [[LABELS[left] + ' − ' + LABELS[right], mean_sd(next(r for r in pairs
                         if r['left'] == left and r['right'] == right and r['split'] == split and r['target'] == 'same_residence'
                         and r['metric'] == 'log_loss' and r['selector'] == 'primary'))] for _, left, right in STAGE_PAIRS])
        lines += ['']
    lines += ['## Catch-up-inclusive audit sensitivity', '',
              'Only additional catch-up candidates inherit representation-training exposure before fitting on the attacker pool. '
              'The inclusive selector may still choose a fresh independent candidate. Saved training adversaries are diagnostic-only.', '']
    lines += table(['Split', 'Release', 'SEX independent', 'SEX inclusive', 'RAC1P independent', 'RAC1P inclusive'],
                   [[split.replace('_', ' '), LABELS[r], *[avg(r, 'audit', t, split, selector=selector)
                     for t in ATTRIBUTES for selector in ('primary', 'catchup_inclusive')]]
                    for split in ('validation', 'development_evaluation') for r in LEARNED_AUDITS])
    lines += ['', '## Original-PCA32 source and residence references', '',
              'Each source loss permits at most +.01 nats relative to original PCA32. Residence retains half the positive '
              'PCA32 advantage over the stronger unprotected rich bank. Nonpositive headroom stays undefined. '
              'These descriptive references are not an all-attribute admission gate.', '']
    lines += table(['Seed', 'Split', 'Stage', 'Income Δ', 'Work Δ', 'Coverage Δ', 'Source all', 'Residence retained', 'Residence half'],
                   [[r['seed'], r['split'].replace('_', ' '), LABELS[r['release']],
                     *[precise(r['source_preservation']['tasks'][t]['difference']) for t in SOURCE_TASKS],
                     status(r['source_preservation']['pass']), precise(r['residential_retention']['retained_fraction']),
                     status(r['residential_retention']['pass'])] for r in criteria])
    lines += ['', 'Full race-schema coverage is incomplete. No absent class, unaudited stage, or unsuccessful finite attacker '
              'counts as protection. Historical scores and flags remain unchanged.', '']
    lines += ['## Final-stage descriptive policy criteria', '',
              'The original PCA32 is the parent for source preservation, residential retention, and attribute-gain halving. '
              'Halving requires positive parent gain and complete audit/exposed-control coverage. Known failures remain failures '
              'when another criterion is undefined. Both saved audit candidate sets are shown separately.', '']
    lines += table(['Seed', 'Split', 'Final release', 'Audit selector', 'Source all', 'Residence half', 'SEX half gain', 'RAC1P half gain', 'Joint'],
                   [[r['seed'], r['split'].replace('_', ' '), LABELS[r['release']], r['audit_selector'],
                     status(r['source_preservation']['pass']), status(r['residential_retention']['pass']),
                     *[status(r['attribute_halving'][t]['pass']) for t in ATTRIBUTES], status(r['joint_pass'])] for r in policies])
    lines += ['']
    (out/'TABLE.md').write_text('\n'.join(lines))

    analysis = ['# PCA16 initialization: paired stage analysis', '',
                'All comparisons use saved post-freeze heads and audit choices. Primary is the matched independent '
                'audit for both initialized and historical learned releases. Utility comparisons retain every task. '
                'The stage contrasts separate initialization, shared warm-up, task-only continuation, and added protection. '
                'C_init and D_init are separate continuations from W; D_init is not trained from C_init. '
                'They are descriptive differences, not an outcome-driven checkpoint-selection rule.', '',
                '[Main table](TABLE.md), [all paired metrics](PAIRED.csv), [new fit accounting](FITTING.csv), '
                '[new classes](PER_CLASS.csv), [new curves](CURVES.csv), [full aggregates](summary.json).', '',
                '## Every stage contrast, per-seed task log loss', '']
    analysis += table(['Seed', 'Split', 'Left − right', *[TASK_DISPLAY[t] for t in UTILITY_TASKS]],
                      [[seed, split.replace('_', ' '), LABELS[left] + ' − ' + LABELS[right],
                        *[precise(next(r['left_minus_right'] for r in paired if r['seed'] == seed and r['split'] == split
                            and r['left'] == left and r['right'] == right and r['target'] == t and r['selector'] == 'primary' and r['metric'] == 'log_loss'))
                          for t in UTILITY_TASKS]] for split in ('validation', 'development_evaluation') for seed in seeds
                       for _, left, right in STAGE_PAIRS])
    analysis += ['', '## All task/family stage contrasts, mean ± sample SD', '',
                 'Within-family selections use their designated validation pool. Fresh MLP family comparisons exclude catch-up.', '']
    analysis += table(['Split', 'Left − right', 'Target', 'Selector', 'Log-loss difference'],
                      [[r['split'].replace('_', ' '), LABELS[r['left']] + ' − ' + LABELS[r['right']], TASK_DISPLAY[r['target']],
                        r['selector'], mean_sd(r)] for r in pairs if r['metric'] == 'log_loss' and 'person_weighted' not in r['split']])
    analysis += ['', '## Person-weighted sensitivity: primary paired log losses', '',
                 'PWGTP weights change scoring only; fitting and validation selection remain unweighted. '
                 'These are sensitivity calculations, not official Census estimates or design-based uncertainty.', '']
    analysis += table(['Split', 'Left − right', 'Target', 'Log-loss difference'],
                      [[r['split'].replace('_', ' '), LABELS[r['left']] + ' − ' + LABELS[r['right']], TASK_DISPLAY[r['target']], mean_sd(r)]
                       for r in pairs if r['metric'] == 'log_loss' and r['selector'] == 'primary' and 'person_weighted' in r['split']])
    analysis += ['', '## Signed attribute gains', '',
                 'Gain is fitting-prior log loss minus attack log loss, paired per seed before aggregation. '
                 'Negative gains remain negative. I/W are not audited. A scalar gain does not resolve missing category support.', '']
    analysis += table(['Release', 'Selector', 'Validation SEX', 'Validation RAC1P', 'Development SEX', 'Development RAC1P'],
                      [[LABELS[r], selector, *[avg(r, 'audit', t, split, 'signed_prior_relative_gain', selector)
                        for split in ('validation', 'development_evaluation') for t in ATTRIBUTES]]
                       for r in ('PCA16', *LEARNED_AUDITS, 'E_pca', 'E_pca_leace')
                       for selector in (('primary', 'catchup_inclusive') if r in LEARNED_AUDITS else ('primary',))])
    analysis += ['', '## Final-stage audit coverage', '',
                 'The fixed two-class SEX/nine-class RAC1P schema, actual attacker-fit support, validation/evaluation support '
                 'and validation-selected exposed-control recalls are required. Missing race categories remain limitations.', '']
    analysis += table(['Seed', 'Split', 'Release', 'Selector', 'Target', 'Candidate', 'Complete', 'Limitations (Census codes)'],
                      [[r['seed'], r['split'].replace('_', ' '), LABELS[r['release']], r['selector'], r['target'],
                        r['audit_candidate'], r['complete'], json.dumps(r['limitation_census_codes'], separators=(',', ':'))] for r in coverage])
    analysis += ['', '## Attribute halving versus original PCA32', '',
                 'Signed gain is reported even when a coverage limit prevents fractional interpretation or a policy pass. '
                 'Numeric halvings are shown separately from coverage-gated assessments.', '']
    analysis += table(['Seed', 'Split', 'Final release', 'Selector', 'Attribute', 'Parent gain', 'Final gain',
                       'Numeric halving', 'Coverage-gated halving'],
                      [[r['seed'], r['split'].replace('_', ' '), LABELS[r['release']], r['audit_selector'], t,
                        precise(r['attribute_halving'][t]['parent_attribute_gain']), precise(r['attribute_halving'][t]['erased_attribute_gain']),
                        status(r['attribute_halving'][t]['numeric_halving_inequality']), status(r['attribute_halving'][t]['pass'])]
                       for r in policies for t in ATTRIBUTES])
    analysis += ['', '## Final releases versus all four bank references', '',
                 'Residence must be at least .01 nats lower while each attribute gain is at most .005 nats higher. '
                 'Numeric inequalities are published even when race coverage limits the joint interpretation. '
                 'These descriptive references do not select or reject future method work.', '']
    analysis += table(['Seed', 'Split', 'Final release', 'Bank', 'Selector', 'Residence Δ', 'Utility', 'SEX numeric', 'RAC1P numeric', 'Numeric joint', 'Joint with coverage'],
                      [[r['seed'], r['split'].replace('_', ' '), LABELS[r['release']], LABELS[r['bank']], r['audit_selector'],
                        precise(r['utility']['feature_minus_bank']), status(r['utility']['pass']),
                        *[status(r['attributes'][t]['numeric_inequality']) for t in ATTRIBUTES],
                        status(r['numeric_joint_inequalities']), status(r['joint_pass'])] for r in bank_comparisons])
    analysis += ['', '## New-stage classification metrics', '',
                 'Full-schema balanced accuracy/macro AUROC remain undefined if a category is unsupported. '
                 'Per-class and observed-class alternatives are explicitly named in CSV.', '']
    analysis += table(['Split', 'Stage', 'Target', 'Log loss', 'Accuracy', 'Balanced accuracy', 'AUROC / macro'],
                      [[split.replace('_', ' '), LABELS[r], TASK_DISPLAY[t], *[avg(r, role, t, split, metric)
                        for metric in ('log_loss', 'accuracy', 'balanced_accuracy', 'macro_auroc' if t == 'RAC1P' else 'auroc')]]
                       for split in SPLITS.values() for r in NEW for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES))
                       if role != 'audit' or r in FINAL for t in targets])
    analysis += ['', 'All initialization identities, training metadata, inherited exposures, original references and runtime '
                'remain in the per-seed evidence linked by summary.json. No privacy, novelty, or population claim follows from '
                'this stage comparison. Every release map remains fixed before its downstream heads are fitted.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))


def plot_stages(out, seeds, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    stages = ('PCA16', 'I', 'W', 'C_init', 'D_init')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, split, title in zip(axes, ('validation', 'test'), ('Validation', 'Development evaluation')):
        values = [[get_value(index, seed, 'transfer', release, 'same_residence', split, selector='primary') for release in stages]
                  for seed in seeds]
        for seed, row in zip(seeds, values):
            ax.plot(range(len(stages)), row, alpha=.45, marker='.', linestyle='none', label=f'Seed {seed}')
        stats = [statistics(row[j] for row in values) for j in range(len(stages))]
        ax.errorbar(range(len(stages)), [r['mean'] for r in stats], yerr=[r['sample_sd'] or 0 for r in stats],
                    color='#245c8e', marker='o', linestyle='none', linewidth=2, capsize=3, label='Mean ± sample SD')
        ax.set_xticks(range(len(stages)), ('PCA16', 'I: initial', 'W: warm', 'C: task-only', 'D: protected'))
        ax.set_title(title)
        ax.set_ylabel('Same-residence log loss (lower is better)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen stage snapshots; diagnostic probes after complete training', fontsize=11)
    fig.text(.5, .015, 'C and D are parallel continuations from W; D is not trained after C.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(out/'residence_stages.png', dpi=200)
    fig.savefig(out/'residence_stages.pdf')
    plt.close(fig)


def summarize(out):
    out = Path(out).resolve()
    paths = sorted(out.glob('seed_*/metrics.json'), key=lambda p: int(p.parent.name.removeprefix('seed_')))
    if not paths:
        raise ValueError('No completed seed metrics; reporting never starts training')
    config = read_json(out/'config.json')
    records = [read_json(p) for p in paths]
    seeds = [r['seed'] for r in records]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config['seeds']):
        raise ValueError('Duplicate or undeclared completed seed')
    selections = {r['seed']: read_json(p.parent/'selection_before_test.json') for r, p in zip(records, paths)}
    index = make_index(records)
    scores, classes, curves, fits = flatten_new(records, selections)
    aggregates = aggregate(seeds, index)
    paired = paired_rows(seeds, index)
    pairs = aggregate_pairs(paired, seeds)
    criteria, coverage = [], []
    for seed in seeds:
        for raw_split, split in (('validation', 'validation'), ('test', 'development_evaluation')):
            loss = lambda release: {t: get_value(index, seed, 'transfer', release, t, raw_split, selector='primary') for t in UTILITY_TASKS}
            banks = {b: loss(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')}
            criteria.extend({'seed': seed, 'split': split, 'release': release, **utility_criteria(loss(release), loss('E_pca'), banks)} for release in NEW)
            for release in FINAL:
                for selector in ('primary', 'catchup_inclusive'):
                    for target in ATTRIBUTES:
                        coverage.append({'seed': seed, 'split': split, 'release': release, 'selector': selector, 'target': target,
                            **audit_coverage(index, selections[seed], seed, release, target, raw_split, selector)})
    policies, bank_comparisons = final_criteria(seeds, index, selections)
    for filename, rows in (('PER_TARGET.csv', scores), ('PER_CLASS.csv', classes), ('CURVES.csv', curves), ('FITTING.csv', fits), ('PAIRED.csv', paired)):
        save_csv(out/filename, rows)
    save_json(out/'criteria.json', {'scope': 'unchanged PCA32 source/residence references for all snapshots; final-stage full criteria separately',
              'per_seed_split': criteria, 'final_policy_per_seed_split': policies})
    save_json(out/'bank_comparisons.json', {'scope': 'final C_init/D_init versus each bank under both audit candidate sets',
                                          'per_seed_split': bank_comparisons})
    input_paths = [out/'config.json', out/'PROTOCOL.md', *paths, *[p.parent/'selection_before_test.json' for p in paths]]
    sources = [Path(__file__), Path(__file__).with_name('summarize_acs_protection.py'), Path(__file__).with_name('summarize_acs_bottleneck.py')]
    summary = {'evaluation_status': config['evaluation_status'], 'completed_seeds': seeds,
        'declared_seeds': config['seeds'], 'all_declared_seeds_complete': set(seeds) == set(config['seeds']),
        'primary_audits': 'five independent candidates for all audited releases; saved choices only',
        'unaudited_stages': ['I', 'W'], 'catchup_inclusive': 'separate saved selector, never a replacement for primary',
        'aggregate_metrics': aggregates, 'paired_aggregate_metrics': pairs, 'coverage': coverage,
        'final_policy_counts': {split: {release: {selector: {
            name: {'pass': sum(getter(r) is True for r in policies if r['split'] == split and r['release'] == release and r['audit_selector'] == selector),
                   'fail': sum(getter(r) is False for r in policies if r['split'] == split and r['release'] == release and r['audit_selector'] == selector),
                   'undefined': sum(getter(r) is None for r in policies if r['split'] == split and r['release'] == release and r['audit_selector'] == selector)}
            for name, getter in {'source': lambda r: r['source_preservation']['pass'],
                                 'residence': lambda r: r['residential_retention']['pass'],
                                 'SEX': lambda r: r['attribute_halving']['SEX']['pass'],
                                 'RAC1P': lambda r: r['attribute_halving']['RAC1P']['pass'],
                                 'joint': lambda r: r['joint_pass']}.items()}
            for selector in ('primary', 'catchup_inclusive')} for release in FINAL}
            for split in ('validation', 'development_evaluation')},
        'runtime': {r['seed']: r.get('runtime') for r in records}, 'integrity': {r['seed']: r.get('integrity') for r in records},
        'runtime_path': 'runtime.json',
        'total_experiment_process_seconds': read_json(out/'runtime.json').get('total_experiment_process_seconds') if (out/'runtime.json').exists() else None,
        'release_metadata_paths': {seed: f'seed_{seed}/release_freeze.json' for seed in seeds},
        'training_paths': {seed: f'seed_{seed}/training/training.json' for seed in seeds},
        'support_paths': {seed: f'seed_{seed}/support.json' for seed in seeds},
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in input_paths},
        'report_source_sha256': {p.name: sha(p) for p in sources},
        'row_counts': {'new_scores': len(scores), 'new_classes': len(classes), 'new_curves': len(curves), 'new_fits': len(fits), 'paired': len(paired)}}
    save_json(out/'summary.json', summary)
    write_reports(out, seeds, index, aggregates, pairs, paired, criteria, coverage, policies, bank_comparisons)
    plot_stages(out, seeds, index)
    print(json.dumps({'out': str(out), 'seeds': seeds, **summary['row_counts']}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)


if __name__ == '__main__':
    main()

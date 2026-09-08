"""Compact, read-only PCA16 diagnostic reporting from saved scores.

Usage: python scripts/summarize_acs_pca16.py --out RESULTS_DIRECTORY
Only new PCA16 candidate/class/curve records are exported. Historical evidence
is read by reference and hashed, never relabeled in place or refitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, SPLITS, SCORE_METRICS, TASK_DISPLAY,
    source_preservation, residential_retention, audit_coverage, get_score,
    get_value, candidate_metadata, statistics, mean_sd, number, finite,
    status, table, read_json, save_json, save_csv, sha, aggregate_pairs,
)
from scripts.summarize_acs_bottleneck import make_index as original_index

ROOT = Path(__file__).resolve().parents[1]
NEW_RELEASE = 'PCA16'
LEARNED = ('C_bottleneck', 'D_protected')
RELEASES = ('E_pca', NEW_RELEASE, 'E_pca_leace', *LEARNED,
            'B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace', 'prior')
LABELS = {'E_pca': 'Original PCA32', NEW_RELEASE: 'PCA16', 'E_pca_leace': 'PCA32 + LEACE',
          'C_bottleneck': 'C task-only bottleneck (16D)', 'D_protected': 'D protected bottleneck (16D)',
          'B_rich_bank': 'Rich neural bank (26D)', 'B_rich_bank_leace': 'Neural bank + LEACE',
          'C_tree_bank': 'Rich tree bank (26D)', 'C_tree_bank_leace': 'Tree bank + LEACE', 'prior': 'Fitting prior'}


def primary_selector(role, release):
    return 'independent_selected' if role == 'audit' and release in LEARNED else 'selected'


def make_index(records):
    """Add report aliases while keeping original dictionaries and flags intact."""
    index = original_index(records)
    groups = {key[:4] for key in index}
    for seed, role, release, target in groups:
        selected = primary_selector(role, release)
        row = index.get((seed, role, release, target, selected))
        if row is not None:
            index[seed, role, release, target, 'primary'] = row
        if role == 'audit' and release in LEARNED:
            row = index.get((seed, role, release, target, 'selected'))
            if row is not None:
                index[seed, role, release, target, 'historical_catchup_inclusive'] = row
    return index


def utility_criteria(pca16, pca32, banks):
    """Unchanged source/residence references; this is a dimension diagnostic."""
    return {'parent': 'E_pca', 'source_preservation': source_preservation(pca32, pca16),
            'residential_retention': residential_retention(pca32.get('same_residence'), pca16.get('same_residence'),
                                                         banks.get('B_rich_bank'), banks.get('C_tree_bank')),
            'all_task_pca16_minus_parent': {target: pca16[target] - pca32[target]
                if finite(pca16.get(target)) and finite(pca32.get(target)) else None for target in UTILITY_TASKS},
            'scope': 'descriptive source/residence reference margins; not an all-attribute policy admission gate'}


def paired_rows(seeds, index):
    rows = []
    for seed in seeds:
        for reference in ('E_pca', *LEARNED):
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                selectors = ('primary', 'family:logistic', 'family:mlp') + (('family:histgb',) if role == 'audit' else ())
                for target in targets:
                    for selector in selectors:
                        a = index.get((seed, role, NEW_RELEASE, target, selector))
                        b = index.get((seed, role, reference, target, selector))
                        for raw_split, split in SPLITS.items():
                            for metric in SCORE_METRICS:
                                av = a.get(raw_split, {}).get(metric) if a else None
                                bv = b.get(raw_split, {}).get(metric) if b else None
                                rows.append({'seed': seed, 'comparison': 'pca16_minus_reference', 'left': NEW_RELEASE,
                                    'right': reference, 'role': role, 'target': target, 'selector': selector, 'split': split,
                                    'metric': metric, 'left_candidate': a['candidate_id'] if a else None,
                                    'right_candidate': b['candidate_id'] if b else None, 'left_value': av, 'right_value': bv,
                                    'left_minus_right': av - bv if finite(av) and finite(bv) else None})
    return rows


def flatten_new(records, selections):
    scores, classes, curves, fits = [], [], [], []
    for record in records:
        for raw in record['raw_metrics']:
            if raw['release'] != NEW_RELEASE:
                raise ValueError('New metrics must contain PCA16 only; historical scores stay referenced')
            meta = candidate_metadata(selections[record['seed']], raw)
            common = {k: raw.get(k) for k in ('seed', 'role', 'release', 'target', 'candidate_id', 'family',
                                            'selected', 'independent_selected', 'selected_within_family', 'auc_selected')}
            common.update(fit_rows=meta.get('fit_rows'), fit_support=meta.get('fit_support'),
                          fit_coverage_complete=meta.get('fit_coverage_complete'))
            fits.append({**common, **{k: meta.get(k) for k in (
                'optimizer_steps', 'selected_epoch', 'selected_optimizer_steps', 'training_row_exposures',
                'schedule_hash', 'initialization_seed', 'schedule_seed', 'restart_index', 'parameters',
                'fit_runtime_seconds', 'warnings', 'fallback_reason')}})
            curves.extend({**common, **point, 'checkpoint_selected': point['epoch'] == meta.get('selected_epoch')}
                          for point in meta.get('validation_curve', []))
            for raw_split, split in SPLITS.items():
                score = raw[raw_split]
                scores.append({**common, 'split': split, **{k: v for k, v in score.items() if k != 'per_class'}})
                for cls in score['per_class']:
                    classes.append({**common, 'split': split, **cls,
                        'original_census_code': cls['class_index'] + 1 if raw['role'] == 'audit' else None,
                        'fit_class_support': meta['fit_support'][cls['class_index']],
                        'schema_coverage_complete': score['coverage_complete']})
    return scores, classes, curves, fits


def aggregate(seeds, index):
    rows = []
    for release in RELEASES:
        for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
            selectors = ('primary', 'historical_catchup_inclusive') if role == 'audit' and release in LEARNED else ('primary',)
            for target in targets:
                for selector in selectors:
                    for raw_split, split in SPLITS.items():
                        for metric in SCORE_METRICS:
                            values = [get_value(index, seed, role, release, target, raw_split, metric, selector) for seed in seeds]
                            rows.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                         'split': split, 'metric': metric, **statistics(values)})
                        if role == 'audit':
                            values = []
                            for seed in seeds:
                                prior = get_value(index, seed, role, 'prior', target, raw_split, selector='primary')
                                attack = get_value(index, seed, role, release, target, raw_split, selector=selector)
                                values.append(prior - attack if finite(prior) and finite(attack) else None)
                            rows.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                         'split': split, 'metric': 'signed_prior_relative_gain', **statistics(values)})
    return rows


def write_reports(out, seeds, index, aggregates, pairs, criteria, coverage, reference):
    ai = {(r['release'], r['role'], r['target'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    avg = lambda release, role, target, split, metric='log_loss', selector='primary': mean_sd(ai[release, role, target, selector, split, metric])
    relative_reference = Path('..')/reference.name
    support_link = (relative_reference/'SUPPORT.md').as_posix()
    previous_table_link = (relative_reference/'TABLE.md').as_posix()
    intro = ['# PCA16 dimension diagnostic — development evidence', '',
             'PCA16 is the fixed first16-column slice of the original PCA32 release, before head-specific standardization. '
             'All original test households are development evaluation. No fitting or selection is performed by this report. '
             'The primary comparison uses matched five-candidate independent auditors for every release: original '
             '`independent_selected` for C/D and saved `selected` for PCA16 and other references. '
             'Historical catch-up-inclusive C/D choices are shown separately; original flags and files remain unchanged. '
             'Natural log loss is primary; means ± sample SD describe seeds sharing a cohort.', '',
             f"Completed seeds: {', '.join(map(str, seeds))}. [Analysis](ANALYSIS.md), [new PCA16 candidates](PER_TARGET.csv), "
             '[new class scores](PER_CLASS.csv), [paired scores](PAIRED.csv), '
             f'[historical support/exposed controls]({support_link}), [historical tables]({previous_table_link}).', '']
    lines = intro.copy()
    for split in ('validation', 'development_evaluation'):
        lines += [f"## {split.replace('_', ' ').capitalize()}: primary log loss, mean ± sample SD", '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[LABELS[release], *[avg(release, 'transfer', t, split) for t in UTILITY_TASKS],
                         *[avg(release, 'audit', t, split) for t in ATTRIBUTES]] for release in RELEASES])
        lines += ['', f"## {split.replace('_', ' ').capitalize()}: every seed", '']
        raw_split = 'validation' if split == 'validation' else 'test'
        lines += table(['Seed', 'Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
                       [[seed, LABELS[release],
                         *[number(get_value(index, seed, 'transfer', release, t, raw_split, selector='primary')) for t in UTILITY_TASKS],
                         *[number(get_value(index, seed, 'audit', release, t, raw_split, selector='primary')) for t in ATTRIBUTES]]
                        for seed in seeds for release in RELEASES])
        lines += ['']
    lines += ['## Original-PCA source and residence references', '',
              'Each source loss permits at most +.01 nats relative to original PCA32. Residence retains half of the '
              'positive original-PCA advantage over the stronger unprotected rich bank. Nonpositive headroom is undefined. '
              'These are descriptive dimension-diagnostic references, not all-attribute policy admission checks.', '']
    lines += table(['Seed', 'Split', 'Income difference', 'Work difference', 'Coverage difference', 'Source all within margin',
                    'Residence retained fraction', 'Residence half-headroom'],
                   [[r['seed'], r['split'].replace('_', ' '),
                     *[number(r['source_preservation']['tasks'][t]['difference']) for t in SOURCE_TASKS],
                     status(r['source_preservation']['pass']), number(r['residential_retention']['retained_fraction']),
                     status(r['residential_retention']['pass'])] for r in criteria])
    lines += ['', '## Historical catch-up-inclusive C/D audit sensitivity', '',
              'These preserve the previous stage’s different candidate set; they do not replace the matched independent primary table. '
              'Each historical catch-up trajectory inherited260 representation-fitting passes and added120 attacker-fitting epochs. '
              'Fresh independent candidates have no inherited training history, so these are different lifetime budgets.', '']
    lines += table(['Split', 'Method', 'SEX independent', 'SEX historical inclusive', 'RAC1P independent', 'RAC1P historical inclusive'],
                   [[split.replace('_', ' '), LABELS[release], *[avg(release, 'audit', t, split, selector=selector)
                     for t in ATTRIBUTES for selector in ('primary', 'historical_catchup_inclusive')]]
                    for split in ('validation', 'development_evaluation') for release in LEARNED])
    lines += ['', 'Race-category coverage remains incomplete. Lower finite-auditor gains are not a privacy guarantee. '
              'Every new candidate, weighted sensitivity, and undefined full-schema statistic is preserved.', '']
    (out/'TABLE.md').write_text('\n'.join(lines))

    analysis = ['# PCA16 paired diagnostic analysis', '',
                'This diagnostic adds only the fixed PCA16 interface. All older releases, heads, attackers, '
                'selections and scores are read unchanged. Primary C/D comparisons use their matched independent attackers; '
                'no development minimum is substituted. Sample SDs are descriptive across shared-cohort seeds.', '',
                '[Main table](TABLE.md), [all paired metrics](PAIRED.csv), [new fitting accounting](FITTING.csv), '
                '[new curves](CURVES.csv), [new classes](PER_CLASS.csv), [exact source/residence criteria](criteria.json), '
                f'[historical coverage and exposed controls]({support_link}).', '',
                'Positive PCA16-minus-reference loss differences lose utility or reduce measured attribute recovery. '
                'Negative differences retain their sign. All five tasks remain reported; no alternative task or '
                'attribute threshold is selected from this diagnostic.', '', '## Primary paired log losses', '']
    analysis += table(['Split', 'Reference', 'Task / attribute', 'PCA16 − reference'],
                      [[r['split'].replace('_', ' '), LABELS[r['right']], TASK_DISPLAY[r['target']], mean_sd(r)]
                       for r in pairs if r['selector'] == 'primary' and r['metric'] == 'log_loss' and 'person_weighted' not in r['split']])
    analysis += ['', '## Matched-family paired log losses', '',
                 'Within-family choices are the previously saved validation choices. The MLP family comparison '
                 'uses fresh independent candidates, excluding historical catch-up and saved adversaries.', '']
    analysis += table(['Split', 'Reference', 'Task / attribute', 'Family', 'PCA16 − reference'],
                      [[r['split'].replace('_', ' '), LABELS[r['right']], TASK_DISPLAY[r['target']], r['selector'].removeprefix('family:'), mean_sd(r)]
                       for r in pairs if r['selector'].startswith('family:') and r['metric'] == 'log_loss' and 'person_weighted' not in r['split']])
    analysis += ['', '## Signed attribute gain and coverage', '',
                 'Gain is fitting-prior log loss minus attack log loss, paired within each seed before aggregation. '
                 'Negative gains stay negative. Fixed-schema missing support and exposed-control failures limit '
                 'race interpretation; no all-attribute protection pass is asserted.', '']
    analysis += table(['Release', 'Validation SEX gain', 'Validation RAC1P gain', 'Development SEX gain', 'Development RAC1P gain'],
                      [[LABELS[release], *[avg(release, 'audit', t, split, 'signed_prior_relative_gain')
                        for split in ('validation', 'development_evaluation') for t in ATTRIBUTES]] for release in RELEASES])
    analysis += ['', 'New PCA16 primary coverage checks use actual fitting support and the unchanged exposed-control evidence.', '']
    analysis += table(['Seed', 'Split', 'Attribute', 'Primary candidate', 'Complete', 'Limitations (Census codes)'],
                      [[r['seed'], r['split'].replace('_', ' '), r['target'], r['audit_candidate'], r['complete'],
                        json.dumps(r['limitation_census_codes'], separators=(',', ':'))] for r in coverage])
    analysis += ['', '## PCA16 metrics and weighted sensitivity', '',
                 'PWGTP sensitivity uses the same predictions and unweighted validation selections. '
                 'Full-schema balanced accuracy/macro AUROC remain undefined when a class is unsupported; '
                 'observed-class alternatives are separately named in CSV. These are not official Census estimates.', '']
    analysis += table(['Split', 'Target', 'Log loss', 'Accuracy', 'Balanced accuracy', 'AUROC / macro'],
                      [[split.replace('_', ' '), TASK_DISPLAY[t], *[avg(NEW_RELEASE, role, t, split, metric)
                        for metric in ('log_loss', 'accuracy', 'balanced_accuracy', 'macro_auroc' if t == 'RAC1P' else 'auroc')]]
                       for split in SPLITS.values() for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)) for t in targets])
    analysis += ['', 'No catch-up fitting, eraser, protection strength, new encoder, or task selection is introduced here. '
                 'The purpose is to measure the fixed dimensionality control under the same scorer and independent audit budget.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))


def summarize(out):
    out = Path(out).resolve()
    paths = sorted(out.glob('seed_*/metrics.json'), key=lambda p: int(p.parent.name.removeprefix('seed_')))
    if not paths:
        raise ValueError('No completed PCA16 metrics; this reporter never fits models')
    config = read_json(out/'config.json')
    reference = Path(config['reference_results'])
    reference = reference.resolve() if reference.is_absolute() else (ROOT/reference).resolve()
    new = [read_json(p) for p in paths]
    seeds = [r['seed'] for r in new]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config['seeds']):
        raise ValueError('Duplicate or undeclared seed')
    reference_paths = [reference/f'seed_{seed}'/'metrics.json' for seed in seeds]
    reference_selection_paths = [p.parent/'selection_before_test.json' for p in reference_paths]
    old = [read_json(p) for p in reference_paths]
    new_selections = {r['seed']: read_json(p.parent/'selection_before_test.json') for r, p in zip(new, paths)}
    old_selections = {seed: read_json(p) for seed, p in zip(seeds, reference_selection_paths)}
    combined = []
    selections = {}
    for r, previous in zip(new, old):
        if r['seed'] != previous['seed']:
            raise ValueError('Historical seed mismatch')
        combined.append({'seed': r['seed'], 'raw_metrics': [*r['raw_metrics'], *previous['raw_metrics']]})
        selections[r['seed']] = {'fitting_records': {**old_selections[r['seed']]['fitting_records'],
                                                    **new_selections[r['seed']]['fitting_records']}}
    index = make_index(combined)
    scores, classes, curves, fits = flatten_new(new, new_selections)
    aggregates = aggregate(seeds, index)
    paired = paired_rows(seeds, index)
    pairs = aggregate_pairs(paired, seeds)
    criteria, coverage = [], []
    for seed in seeds:
        for raw_split, split in (('validation', 'validation'), ('test', 'development_evaluation')):
            loss = lambda release: {t: get_value(index, seed, 'transfer', release, t, raw_split, selector='primary') for t in UTILITY_TASKS}
            criteria.append({'seed': seed, 'split': split, **utility_criteria(loss(NEW_RELEASE), loss('E_pca'),
                {b: loss(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')})})
            for target in ATTRIBUTES:
                coverage.append({'seed': seed, 'split': split, 'target': target,
                    **audit_coverage(index, selections[seed], seed, NEW_RELEASE, target, raw_split, 'primary')})
    for name, rows in (('PER_TARGET.csv', scores), ('PER_CLASS.csv', classes), ('CURVES.csv', curves),
                       ('FITTING.csv', fits), ('PAIRED.csv', paired)):
        save_csv(out/name, rows)
    save_json(out/'criteria.json', {'scope': 'source/residence references only; no all-policy admission gate',
                                  'per_seed_split': criteria})
    inputs = [out/'config.json', out/'PROTOCOL.md', *paths, *[p.parent/'selection_before_test.json' for p in paths]]
    summary = {'evaluation_status': config['evaluation_status'], 'completed_seeds': seeds,
        'declared_seeds': config['seeds'], 'all_declared_seeds_complete': set(seeds) == set(config['seeds']),
        'primary_audit': 'matched five independent candidates; historical C/D independent_selected, other saved selected',
        'historical_catchup_inclusive': 'separate diagnostic alias; original flags unchanged',
        'score_scope': 'unweighted natural log loss primary; signed prior gains; person-weighted sensitivity; descriptive sample SD',
        'aggregate_metrics': aggregates, 'paired_aggregate_metrics': pairs, 'coverage': coverage,
        'runtime': {r['seed']: r.get('runtime') for r in new},
        'integrity': {r['seed']: r.get('integrity') for r in new},
        'release_metadata_paths': {r['seed']: f"seed_{r['seed']}/release_freeze.json" for r in new},
        'reference_results': str(reference.relative_to(ROOT)) if reference.is_relative_to(ROOT) else str(reference),
        'reference_read_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                                  for p in [*reference_paths, *reference_selection_paths]},
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in inputs},
        'report_source_sha256': {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('summarize_acs_protection.py'),
                                                        Path(__file__).with_name('summarize_acs_bottleneck.py'))},
        'row_counts': {'new_scores': len(scores), 'new_classes': len(classes), 'new_curves': len(curves),
                       'new_fits': len(fits), 'paired': len(paired)}}
    save_json(out/'summary.json', summary)
    write_reports(out, seeds, index, aggregates, pairs, criteria, coverage, reference)
    print(json.dumps({'out': str(out), 'seeds': seeds, **summary['row_counts']}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)


if __name__ == '__main__':
    main()

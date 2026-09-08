"""Read-only reporting for the bounded PCA16-output preservation study.

Consumes saved scores and selections only. Raw beta-unit identifiers are mapped
to unambiguous report identifiers without changing the scientific artifacts.
No models, raw people, releases, or prediction arrays are loaded here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, MARGINS, SPLITS, SCORE_METRICS,
    TASK_DISPLAY, audit_coverage, combined_coverage, parent_comparison,
    feature_bank_comparison, joint_status, candidate_metadata, get_value,
    statistics, finite, number, status, table, read_json, save_json, save_csv, sha,
)
from scripts.summarize_acs_bottleneck import make_index as base_index, INHERITED_FIELDS
from scripts.summarize_acs_pca16_init import utility_criteria

BETAS = ('beta_0p1', 'beta_1')
ARMS = ('C_warmup_only', 'D_warmup_only', 'C_persistent', 'D_persistent')
FINAL = tuple(f'{beta}_{arm}' for beta in BETAS for arm in ARMS)
STAGES = ('I', 'W_historical', *(f'{beta}_W' for beta in BETAS))
HISTORICAL_LEARNED = ('C_init', 'D_init', 'C_bottleneck', 'D_protected')
LEARNED = (*FINAL, *HISTORICAL_LEARNED)
BANKS = ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')
REFERENCES = ('C_init', 'D_init', 'PCA16', 'E_pca', 'E_pca_leace',
              'C_bottleneck', 'D_protected', *BANKS, 'prior')
RELEASES = (*FINAL, *REFERENCES, *STAGES)
LABELS = {'I': 'I: shared initialization', 'W_historical': 'W: historical beta=0',
          'C_init': 'beta=0 C_init', 'D_init': 'beta=0 D_init', 'PCA16': 'Fixed PCA16 teacher',
          'E_pca': 'Original PCA32', 'E_pca_leace': 'PCA32 + LEACE',
          'C_bottleneck': 'Historical C16', 'D_protected': 'Historical D16',
          'B_rich_bank': 'Rich neural bank', 'B_rich_bank_leace': 'Neural bank + LEACE',
          'C_tree_bank': 'Rich tree bank', 'C_tree_bank_leace': 'Tree bank + LEACE',
          'prior': 'Fitting prior', 'exposed': 'Exposed control'}
for _beta in BETAS:
    _value = '.1' if _beta == 'beta_0p1' else '1'
    LABELS[f'{_beta}_W'] = f'beta={_value} W'
    for _arm in ARMS:
        LABELS[f'{_beta}_{_arm}'] = f'beta={_value} {_arm[0]} ' + (
            'warmup-only' if _arm.endswith('warmup_only') else 'persistent')


def canonical(release, beta):
    return f'{beta}_{release}' if release in ('W', *ARMS) else release


def normalize_unit(record, selection, beta):
    """Copy only report labels; preserve all scores and original source hashes."""
    record, selection = copy.deepcopy(record), copy.deepcopy(selection)
    for row in record['raw_metrics']:
        row['release'] = canonical(row['release'], beta)
    selection['fitting_records'] = {
        '/'.join((parts[0], canonical(parts[1], beta), parts[2])): value
        for key, value in selection['fitting_records'].items()
        for parts in [key.split('/')]
    }
    return record, selection


def merge_units(units):
    """Historical evidence can repeat across beta units only if it is identical."""
    rows, selections = {}, defaultdict(lambda: {'fitting_records': {}})
    for record, selection in units:
        seed = record['seed']
        for row in record['raw_metrics']:
            key = (seed, row['role'], row['release'], row['target'], row['candidate_id'])
            if key in rows and rows[key] != row:
                raise ValueError('Inconsistent reused evidence: ' + str(key))
            rows[key] = row
        for key, value in selection['fitting_records'].items():
            previous = selections[seed]['fitting_records'].get(key)
            if previous is not None and previous != value:
                raise ValueError('Inconsistent reused fitting metadata: ' + str((seed, key)))
            selections[seed]['fitting_records'][key] = value
    seeds = sorted({key[0] for key in rows})
    return [{'seed': seed, 'raw_metrics': [row for key, row in rows.items() if key[0] == seed]}
            for seed in seeds], dict(selections)


def make_index(records):
    index = base_index(records)
    for seed, role, release, target in sorted({key[:4] for key in index}):
        if role == 'audit' and release in STAGES:
            raise ValueError('I and W snapshots have no independently executed attribute audits')
        selector = 'independent_selected' if role == 'audit' else 'selected'
        row = index.get((seed, role, release, target, selector))
        if row is not None:
            index[seed, role, release, target, 'primary'] = row
        if role == 'audit':
            row = index.get((seed, role, release, target, 'selected'))
            if row is not None:
                index[seed, role, release, target, 'catchup_inclusive'] = row
    return index


def comparison_matrix():
    pairs = [('historical_warmup', 'W_historical', 'I')]
    for beta in BETAS:
        warm = f'{beta}_W'
        pairs += [('warmup', warm, 'I'), ('warmup_minus_historical', warm, 'W_historical')]
        for arm in ARMS:
            final = f'{beta}_{arm}'
            pairs += [('continuation', final, warm),
                      ('preservation_minus_beta0', final, f'{arm[0]}_init')]
        for arm in ('C', 'D'):
            pairs.append(('persistent_minus_warmup_only', f'{beta}_{arm}_persistent', f'{beta}_{arm}_warmup_only'))
        for schedule in ('warmup_only', 'persistent'):
            pairs.append(('protection_D_minus_C', f'{beta}_D_{schedule}', f'{beta}_C_{schedule}'))
    pairs += [('beta1_minus_beta0p1', f'beta_1_{arm}', f'beta_0p1_{arm}') for arm in ARMS]
    return pairs


def paired_rows(seeds, index):
    rows = []
    for seed in seeds:
        for comparison, left, right in comparison_matrix():
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                if role == 'audit' and (left in STAGES or right in STAGES):
                    continue
                selectors = ('primary', 'family:logistic', 'family:mlp')
                if role == 'audit':
                    selectors += ('family:histgb', 'catchup_inclusive', 'candidate:catchup', 'candidate:saved_adversary')
                for target in targets:
                    for selector in selectors:
                        a = index.get((seed, role, left, target, selector))
                        b = index.get((seed, role, right, target, selector))
                        for raw_split, split in SPLITS.items():
                            av = a.get(raw_split, {}).get('log_loss') if a else None
                            bv = b.get(raw_split, {}).get('log_loss') if b else None
                            rows.append({'seed': seed, 'comparison': comparison, 'left': left, 'right': right,
                                'role': role, 'target': target, 'selector': selector, 'split': split,
                                'metric': 'log_loss', 'left_candidate': a['candidate_id'] if a else None,
                                'right_candidate': b['candidate_id'] if b else None,
                                'left_value': av, 'right_value': bv,
                                'left_minus_right': av - bv if finite(av) and finite(bv) else None})
    return rows


def summarize_pairs(rows):
    groups = defaultdict(list)
    fields = ('comparison', 'left', 'right', 'role', 'target', 'selector', 'split', 'metric')
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return [{**dict(zip(fields, key)), **statistics(row['left_minus_right'] for row in values),
             'per_seed': {str(row['seed']): row['left_minus_right'] for row in values}}
            for key, values in groups.items()]


def aggregate(seeds, index):
    rows = []
    for release in RELEASES:
        for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
            if role == 'audit' and release in STAGES:
                continue
            selectors = ('primary', 'catchup_inclusive') if role == 'audit' and release in LEARNED else ('primary',)
            for target in targets:
                for selector in selectors:
                    for raw_split, split in SPLITS.items():
                        for metric in SCORE_METRICS:
                            values = [get_value(index, seed, role, release, target, raw_split, metric, selector) for seed in seeds]
                            rows.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                         'split': split, 'metric': metric, **statistics(values)})
                        if role == 'audit':
                            gains = []
                            for seed in seeds:
                                prior = get_value(index, seed, role, 'prior', target, raw_split, selector='primary')
                                value = get_value(index, seed, role, release, target, raw_split, selector=selector)
                                gains.append(prior - value if finite(prior) and finite(value) else None)
                            rows.append({'release': release, 'role': role, 'target': target, 'selector': selector,
                                'split': split, 'metric': 'signed_prior_relative_gain', **statistics(gains)})
    return rows


def flatten(records, selections, audit_budget=120):
    scores, classes, fits, curves = [], [], [], []
    for record in records:
        for raw in record['raw_metrics']:
            meta = candidate_metadata(selections[record['seed']], raw)
            common = {k: raw.get(k) for k in ('seed', 'role', 'release', 'target', 'candidate_id', 'family',
                       'selected', 'independent_selected', 'selected_within_family', 'auc_selected')}
            common.update(audit_budget=audit_budget if raw['role'] == 'audit' else None,
                reused_reference=raw.get('reused_reference', raw['release'] not in (*FINAL, *(f'{b}_W' for b in BETAS))),
                diagnostic_only=raw['candidate_id'] == 'saved_adversary',
                fit_rows=meta.get('fit_rows'), fit_support=meta.get('fit_support'),
                fit_coverage_complete=meta.get('fit_coverage_complete'))
            inherited = meta.get('inherited_exposure') or {}
            fits.append({**common, **{k: meta.get(k) for k in ('optimizer_steps', 'selected_epoch',
                'selected_optimizer_steps', 'training_row_exposures', 'schedule_hash', 'initialization_seed',
                'schedule_seed', 'restart_index', 'parameters', 'fit_runtime_seconds', 'warnings', 'fit_warnings',
                'fallback_reason', 'initial_fidelity_exact')},
                **{'inherited_' + k: inherited.get(k) for k in INHERITED_FIELDS}})
            curves.extend({**common, **point, 'checkpoint_selected': point['epoch'] == meta.get('selected_epoch')}
                          for point in meta.get('validation_curve', []))
            for raw_split, split in SPLITS.items():
                score = raw.get(raw_split)
                if score is None:
                    continue
                scores.append({**common, 'split': split, **{k: v for k, v in score.items() if k != 'per_class'}})
                fit_support = meta.get('fit_support') or [None] * score['n_classes']
                classes.extend({**common, 'split': split, **cls,
                    'original_census_code': cls['class_index'] + 1 if raw['role'] == 'audit' else None,
                    'fit_class_support': fit_support[cls['class_index']],
                    'schema_coverage_complete': score['coverage_complete']} for cls in score['per_class'])
    return scores, classes, fits, curves


def criteria_rows(seeds, index, selections):
    stages, policies, banks, coverage = [], [], [], []
    for seed in seeds:
        for raw_split, split in SPLITS.items():
            utility = lambda r: {t: get_value(index, seed, 'transfer', r, t, raw_split, selector='primary') for t in UTILITY_TASKS}
            bank_residence = {b: utility(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')}
            for release in RELEASES:
                stages.append({'seed': seed, 'split': split, 'release': release,
                    **utility_criteria(utility(release), utility('E_pca'), bank_residence)})
            for selector in ('primary', 'catchup_inclusive'):
                attack = lambda r: {t: get_value(index, seed, 'audit', r, t, raw_split,
                    selector=selector if r in LEARNED else 'primary') for t in ATTRIBUTES}
                cov = lambda r, t: audit_coverage(index, selections.get(seed, {'fitting_records': {}}), seed, r, t, raw_split,
                    selector if r in LEARNED else 'primary')
                for release in FINAL:
                    attr_cov = {t: combined_coverage(cov(release, t), cov('E_pca', t)) for t in ATTRIBUTES}
                    policies.append({'seed': seed, 'split': split, 'release': release, 'audit_selector': selector,
                        'fixed_parent': 'E_pca', 'attribute_coverage': attr_cov,
                        **parent_comparison(utility('E_pca'), utility(release), bank_residence,
                            attack('E_pca'), attack(release), attack('prior'), attr_cov)})
                    coverage.extend({'seed': seed, 'split': split, 'release': release, 'selector': selector,
                                     'target': t, **cov(release, t)} for t in ATTRIBUTES)
                    for bank in BANKS:
                        bank_cov = {t: combined_coverage(cov(release, t), cov(bank, t)) for t in ATTRIBUTES}
                        comparison = feature_bank_comparison(utility(release)['same_residence'], utility(bank)['same_residence'],
                            attack(release), attack(bank), attack('prior'), bank_cov)
                        banks.append({'seed': seed, 'split': split, 'release': release, 'bank': bank,
                            'audit_selector': selector, 'numeric_joint_inequalities': joint_status([
                                comparison['utility']['pass'], *[v['numeric_inequality'] for v in comparison['attributes'].values()]]),
                            **comparison})
    return stages, policies, banks, coverage


def primary_rows(seeds, index, criteria):
    rows = []
    for release in (*FINAL, *REFERENCES):
        result = {'release': release, 'label': LABELS[release], 'n_declared_seeds': len(seeds)}
        for raw_split, short in (('test', 'unweighted'), ('test_person_weighted', 'PWGTP')):
            split = SPLITS[raw_split]
            selected = [r for r in criteria if r['release'] == release and r['split'] == split]
            result[f'{short}_source_all_pass'] = sum(r['source_preservation']['pass'] is True for r in selected)
            result[f'{short}_source_all_defined'] = sum(r['source_preservation']['pass'] is not None for r in selected)
            result[f'{short}_residence_half_pass'] = sum(r['residential_retention']['pass'] is True for r in selected)
            changes = [v['difference'] for r in selected for v in r['source_preservation']['tasks'].values()]
            result[f'{short}_worst_source_delta'] = max(changes) if changes and all(finite(v) for v in changes) else None
            for target in UTILITY_TASKS:
                values = [get_value(index, seed, 'transfer', release, target, raw_split, selector='primary') for seed in seeds]
                result.update({f'{short}_{target}_{k}': v for k, v in statistics(values).items()})
            for target in ATTRIBUTES:
                for selector in ('primary', 'catchup_inclusive'):
                    gains = []
                    for seed in seeds:
                        prior = get_value(index, seed, 'audit', 'prior', target, raw_split, selector='primary')
                        value = get_value(index, seed, 'audit', release, target, raw_split, selector=selector)
                        gains.append(prior - value if finite(prior) and finite(value) else None)
                    result.update({f'{short}_{target}_{selector}_gain_{k}': v for k, v in statistics(gains).items()})
        result['race_assessment'] = 'unassessable: missing fitting/validation race code 4; all 9 categories retained'
        rows.append(result)
    return rows


def catchup_rows(seeds, index):
    rows = []
    for seed in seeds:
        for release in LEARNED:
            for target in ATTRIBUTES:
                for raw_split, split in SPLITS.items():
                    result = {'seed': seed, 'release': release, 'target': target, 'split': split}
                    for name, selector in (('saved', 'candidate:saved_adversary'), ('fresh', 'primary'),
                                           ('catchup', 'candidate:catchup'), ('inclusive', 'catchup_inclusive')):
                        row = index.get((seed, 'audit', release, target, selector))
                        result[f'{name}_log_loss'] = row.get(raw_split, {}).get('log_loss') if row else None
                        result[f'{name}_candidate'] = row['candidate_id'] if row else None
                    for right in ('saved', 'fresh'):
                        a, b = result['catchup_log_loss'], result[f'{right}_log_loss']
                        result[f'catchup_minus_{right}'] = a - b if finite(a) and finite(b) else None
                    rows.append(result)
    return rows


def mean_sd(row):
    if not row['complete']:
        return f"undefined ({row['n_defined']}/{row['n_seeds']})"
    return number(row['mean']) + (' ± ' + number(row['sample_sd']) if row['sample_sd'] is not None else ' (one seed)')


def write_reports(out, seeds, primary, aggregates, pairs, criteria, policies, banks, coverage, units, extended):
    ai = {(r['release'], r['role'], r['target'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    def avg(release, role, target, split, metric='log_loss', selector='primary'):
        record = ai.get((release, role, target, selector, split, metric))
        return mean_sd(record) if record else 'not audited'
    lines = ['# PCA16 preservation: primary comparison', '',
        'DEVELOPMENT EVALUATION on the same cohort and original households. Lower task loss and lower '
        'signed prior-relative attack gain are better under this finite audit. Primary attributes use five fresh '
        'validation-selected candidates. PWGTP scores beside them use the SAME selected predictions. '
        'Seed SDs describe shared-cohort variation; they are not population or design-based uncertainty.', '',
        '[Decision](OVERNIGHT_RESEARCH_DECISION.md), [full analysis](ANALYSIS.md), [all selected source/task scores](SELECTED_TASKS.csv), '
        '[all candidates](PER_TARGET.csv), [paired contrasts](PAIRED.csv), [classes and controls](PER_CLASS.csv), '
        '[fitting curves](CURVES.csv), [audit budgets](AUDIT_BUDGET.md), [coordinate diagnostics](PRESERVATION_DIAGNOSTICS.csv).', '',
        f'Completed beta/seed units: {len(units)}/6. Declared seeds: {seeds}. Missing units remain undefined.', '',
        'Values below are development means; all-three-source counts apply the unchanged original-PCA32 +.01-nat '
        'allowance separately to each source and seed. Full task losses and descriptive SDs follow. '
        'Every RAC1P assessment remains support-limited, including numerically small gains.', '']
    lines += table(['Release', 'Source all U / W', 'Residence U / W', 'SEX gain U / W', 'RAC1P gain U / W'],
        [[r['label'], f"{r['unweighted_source_all_pass']}/{len(seeds)} / {r['PWGTP_source_all_pass']}/{len(seeds)}",
          number(r['unweighted_same_residence_mean']) + ' / ' + number(r['PWGTP_same_residence_mean']),
          number(r['unweighted_SEX_primary_gain_mean']) + ' / ' + number(r['PWGTP_SEX_primary_gain_mean']),
          number(r['unweighted_RAC1P_primary_gain_mean']) + ' / ' + number(r['PWGTP_RAC1P_primary_gain_mean'])] for r in primary])
    lines += ['', 'U = unweighted primary; W = PWGTP sensitivity. Gain = fitting-prior log loss minus selected attack log loss. '
        'Signed gains remain signed; a negative predictive gain is not negative information.', '']
    for split in SPLITS.values():
        lines += [f"## {split.replace('_', ' ').capitalize()}: task losses, mean ± sample SD", '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX attack', 'RAC1P attack'],
            [[LABELS[r], *[avg(r, 'transfer', t, split) for t in UTILITY_TASKS],
              *[avg(r, 'audit', t, split) for t in ATTRIBUTES]] for r in RELEASES])
        lines += ['']
    (out/'TABLE.md').write_text('\n'.join(lines))

    analysis = ['# PCA16 preservation: fixed contrasts and descriptive criteria', '',
        'Positive left-minus-right task losses lose utility; positive attribute losses reduce measured recovery. '
        'Every contrast uses saved unweighted-validation selections. No mapper snapshot, attacker, weighting '
        'or experiment was selected by development outcomes. All five tasks remain in the comparison.', '',
        '[Primary table](TABLE.md), [per-seed paired scores](PAIRED.csv), [paired aggregates](PAIRED_AGGREGATE.csv), '
        '[source/residence and policy criteria](criteria.json), [all bank comparisons](bank_comparisons.json).', '',
        '## Prespecified task and attribute contrasts', '']
    for split in ('development_evaluation', 'development_evaluation_person_weighted', 'validation', 'validation_person_weighted'):
        analysis += [f"### {split.replace('_', ' ').capitalize()}", '']
        selected = [r for r in pairs if r['selector'] in ('primary', 'catchup_inclusive') and r['split'] == split]
        analysis += table(['Left − right', 'Target', 'Audit/head selector', 'Log-loss difference'],
            [[LABELS[r['left']] + ' − ' + LABELS[r['right']], TASK_DISPLAY[r['target']], r['selector'], mean_sd(r)] for r in selected])
        analysis += ['']
    analysis += ['## Unchanged original-PCA32 references', '',
        'Each source loss permits +.01 nats. Residence must retain half the positive original-PCA32 advantage '
        'over the stronger unprotected rich bank on that seed and split. Nonpositive denominators are undefined. '
        'Attribute halving requires positive original-PCA32 gain and complete fitting, validation, evaluation '
        'and exposed-control coverage. These margins are descriptive references, not privacy budgets or '
        'statistical noninferiority tests. Known failures remain failures when race is undefined.', '']
    analysis += table(['Seed', 'Split', 'Release', 'Selector', 'Source all', 'Residence half', 'SEX half', 'RAC1P half', 'Joint'],
        [[r['seed'], r['split'], LABELS[r['release']], r['audit_selector'], status(r['source_preservation']['pass']),
          status(r['residential_retention']['pass']), *[status(r['attribute_halving'][t]['pass']) for t in ATTRIBUTES],
          status(r['joint_pass'])] for r in policies])
    analysis += ['', '## Feature/bank comparisons and support', '',
        'Every final arm is compared with all four unprotected/erased rich banks. Residence must be at least '
        '.01 nats lower and each signed attribute gain at most .005 nats higher. Raw numerical inequalities '
        'and coverage-gated results, including failures and undefined cases, are preserved in bank_comparisons.json.', '',
        'All nine RAC1P codes remain in PER_CLASS.csv. Code 4 (Alaska Native alone) is absent from attacker '
        'fitting and validation in every seed, absent from development seeds 0/1, and has one development '
        'example in seed 2. Longer fitting cannot resolve the missing support. Original minimum-leaf-20 '
        'exposed controls also fail on supported code 5; every control candidate remains published.', '',
        '[Original full pool and exposed-control evidence](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md). '
        'SUPPORT.csv records coverage for every new final selected attack and split; PER_CLASS.csv also retains '
        'the original controls and every new candidate. I and W have utility and label-free diagnostics only. '
        'Their teacher audit is not relabeled as a new snapshot audit.', '',
        '## Saved, fresh, and catch-up audit accounting', '',
        'CATCHUP.csv separates the zero-update saved training adversary, five-candidate fresh selection, '
        'direct-coordinate catch-up, and pooled validation selection. Catch-up inherits 20 warm plus 240 '
        'continuation representation-fitting passes, then resets Adam for the attacker pool. It is not an '
        'equal-lifetime-exposure comparison. FITTING.csv and CURVES.csv preserve exposure, fidelity and '
        'all saved validation curves. A validation winner may have worse development loss.', '',
        '## Coordinate movement and recoverable teacher structure', '',
        'Direct error is normalized squared distance to immutable raw PCA16 using saved representation-fitting '
        'scales. The independent affine decoder predicts standardized teacher coordinates; its intercept '
        'and coefficients use representation-fitting examples only. Source-validation/development errors, '
        'per-coordinate errors, rank, fixed rank tolerance and fitting-prior errors are retained in '
        'PRESERVATION_DIAGNOSTICS.csv and COORDINATE_ERRORS.csv. No outcome chooses a decoder or release.', '',
        'Low affine error despite direct movement means teacher structure remains linearly recoverable. '
        'High affine error does not rule out nonlinear recovery. Neither diagnostic replaces utility heads '
        'or attribute audits, and retaining a leaky teacher is not a protection advance.', '',
        '## Completion and audit extension', '',
        f'{len(units)}/6 complete core units. Extended budget status: {extended["status"]}. '
        'See AUDIT_BUDGET.md for nested trajectory comparisons and runtime.json for measured phase time.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))
    support = ['# Fixed attribute schemas and support limits', '',
        'Recorded SEX retains Census codes 1–2; RAC1P retains all nine original codes. '
        'Every candidate is scored in the full declared schema. No missing class or undefined '
        'balanced-accuracy/macro-AUROC statistic is treated as protection.', '',
        'RAC1P code 4 (Alaska Native alone) is absent from attacker fitting and attacker validation '
        'in all three seeds. It is absent from development seeds 0/1 and has one development '
        'observation in seed 2. More epochs cannot recover a class with no fitting examples. '
        'The corresponding complete race assessment remains unassessable.', '',
        'Coverage requires positive fitting, validation and evaluation support for every class, '
        'plus positive class recall for the validation-selected exposed-label control. Original '
        'minimum-leaf-20 exposed trees also miss supported race code 5. These failures remain '
        'visible for every candidate, even when another exposed-control candidate succeeds.', '',
        '- [Every new final selected-attack coverage flag](SUPPORT.csv).',
        '- [Every primary-budget candidate/class score, including historical priors and exposed controls](PER_CLASS.csv).',
        '- [Every extended-budget candidate/class score and extended exposed control](EXTENDED_PER_CLASS.csv).',
        '- [Exact fixed household-pool supports](beta_0p1/seed_0/support.json), '
        '[seed 1](beta_0p1/seed_1/support.json), [seed 2](beta_0p1/seed_2/support.json).',
        '- [Historical pool tables and original exposed-control failures](../redesign_20260908_acs_bottleneck_v1/SUPPORT.md).', '',
        'All original test fields are DEVELOPMENT EVALUATION. PWGTP sensitivity changes scoring '
        'on the same selected predictions; it does not fill a missing class. I/W have no independent '
        'attribute audits in this study and cannot inherit a coverage pass from their PCA16 teacher.', '']
    (out/'SUPPORT.md').write_text('\n'.join(support))


def diagnostic_rows(out):
    rows, coordinates, shared_initial = [], [], {}
    for beta in BETAS:
        for path in sorted((out/beta).glob('seed_*/preservation_diagnostics.json')):
            if not (path.parent/'metrics.json').exists():
                continue
            seed = int(path.parent.name.removeprefix('seed_'))
            data = read_json(path)
            for stage, snapshot in data['snapshots'].items():
                release = canonical(stage, beta)
                for split in ('fit', 'source_validation', 'development_evaluation'):
                    score = snapshot.get(split)
                    if score is None:
                        continue
                    if release == 'I':
                        key = (seed, split)
                        identity = {'rank': snapshot.get('rank'), 'rank_tolerance': snapshot.get('rank_tolerance'), **score}
                        if key in shared_initial:
                            if shared_initial[key] != identity:
                                raise ValueError('Repeated initial affine diagnostic differs between beta units')
                            continue
                        shared_initial[key] = identity
                    common = {'beta': beta, 'seed': seed, 'release': release, 'split': split,
                              'rank': snapshot.get('rank'), 'rank_tolerance': snapshot.get('rank_tolerance')}
                    rows.append({**common, **{k: v for k, v in score.items() if not isinstance(v, (list, dict))}})
                    for key, value in score.items():
                        if isinstance(value, list):
                            coordinates.extend({**common, 'metric': key, 'coordinate': j, 'value': v} for j, v in enumerate(value))
    return rows, coordinates


def training_rows(out):
    curves, gradients = [], []
    for beta in BETAS:
        for path in sorted((out/beta).glob('seed_*/training/training.json')):
            if not (path.parents[1]/'metrics.json').exists():
                continue
            seed = int(path.parents[1].name.removeprefix('seed_'))
            record = read_json(path)
            for phase, points in record['shared_curves'].items():
                curves.extend({'beta': beta, 'seed': seed, 'phase': phase, 'release': canonical('W', beta), **p} for p in points)
            for point, values in record.get('stage_gradient_diagnostics', {}).items():
                gradients.append({'beta': beta, 'seed': seed, 'release': 'shared', 'point': point, **values})
            for arm, meta in record['arms'].items():
                curves.extend({'beta': beta, 'seed': seed, 'phase': 'continuation', 'release': canonical(arm, beta), **p}
                              for p in meta['curve'])
                for point in ('shared_fork', 'final'):
                    gradients.append({'beta': beta, 'seed': seed, 'release': canonical(arm, beta), 'point': point,
                                      **meta.get('fixed_batch_gradient_diagnostics_at_' + point, {})})
    return curves, gradients


def extended_reports(out, core_records, core_selections, core_index):
    """Nested 120/360 comparisons; static candidates are supplied by the runner."""
    paths = sorted((out/'extended').glob('seed_*/metrics.json'))
    if not paths:
        status_record = read_json(out/'progress.json') if (out/'progress.json').exists() else {}
        result = {'status': status_record.get('extended_status', 'not run'), 'reason': status_record.get('extended_reason',
            'No completed extended metrics are present; no extended check is claimed.'), 'completed_seeds': []}
        (out/'AUDIT_BUDGET.md').write_text('# Audit budget sensitivity\n\n' + result['status'] + ': ' + result['reason'] + '\n')
        return result
    records, selections, exports = {}, {}, [[], [], [], []]
    for budget in (120, 360):
        records[budget], selections[budget] = [], {}
        for path in paths:
            data = read_json(path)
            seed = data['seed']
            records[budget].append({'seed': seed, 'raw_metrics': [r for r in data['raw_metrics'] if r['audit_budget'] == budget]})
            selected = read_json(path.parent/'selection_before_test.json')
            selections[budget][seed] = selected['budgets'][str(budget)]
        for destination, new in zip(exports, flatten(records[budget], selections[budget], budget)):
            destination.extend(new)
    for filename, rows in zip(('EXTENDED_PER_TARGET.csv', 'EXTENDED_PER_CLASS.csv', 'EXTENDED_FITTING.csv', 'EXTENDED_CURVES.csv'), exports):
        save_csv(out/filename, rows)
    indices = {budget: make_index(records[budget]) for budget in records}
    rows = []
    for seed, role, release, target, selector in indices[360]:
        if role != 'audit' or selector not in ('primary', 'catchup_inclusive', 'candidate:catchup', 'candidate:mlp_a', 'candidate:mlp_b',
                'candidate:mlp_0', 'candidate:mlp_1', 'family:mlp'):
            continue
        for raw_split, split in SPLITS.items():
            result = {'seed': seed, 'release': release, 'target': target, 'selector': selector, 'split': split}
            for budget in (120, 360):
                raw = indices[budget].get((seed, role, release, target, selector))
                result[f'loss_{budget}'] = raw.get(raw_split, {}).get('log_loss') if raw else None
                result[f'candidate_{budget}'] = raw['candidate_id'] if raw else None
            a, b = result['loss_360'], result['loss_120']
            result['loss_360_minus_120'] = a - b if finite(a) and finite(b) else None
            rows.append(result)
    save_csv(out/'AUDIT_BUDGET.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['release'], row['target'], row['selector'], row['split']].append(row)
    aggregated = [{'release': k[0], 'target': k[1], 'selector': k[2], 'split': k[3],
                   **statistics(r['loss_360_minus_120'] for r in v)} for k, v in groups.items()]
    budget_policies, budget_banks = {}, {}
    for budget in (120, 360):
        replacement = {(r['seed'], raw['role'], raw['release'], raw['target']) for r in records[budget] for raw in r['raw_metrics']}
        completed = set(selections[budget])
        combined, combined_selections = [], {s: copy.deepcopy(v) for s, v in core_selections.items() if s in completed}
        for original in core_records:
            seed = original['seed']
            if seed not in completed:
                continue
            extra = next((r['raw_metrics'] for r in records[budget] if r['seed'] == seed), [])
            combined.append({'seed': seed, 'raw_metrics': [r for r in original['raw_metrics']
                if (seed, r['role'], r['release'], r['target']) not in replacement] + extra})
            combined_selections[seed]['fitting_records'].update(selections[budget].get(seed, {}).get('fitting_records', {}))
        _, budget_policies[budget], budget_banks[budget], _ = criteria_rows(
            sorted(combined_selections), make_index(combined), combined_selections)
    save_json(out/'EXTENDED_CRITERIA.json', {'fixed_parent': 'E_pca', 'parent_audit_budget': 'historical unchanged 120',
        'note': 'PCA32 was not authorized for the extension. Its unchanged scores remain the fixed parent; '
                'both unprotected banks and new learned arms receive matched extended fitting budgets.',
        'budgets': {str(b): {'policies': budget_policies[b], 'bank_comparisons': budget_banks[b]} for b in (120, 360)}})
    changes = []
    bank_keys = ('seed', 'split', 'release', 'bank', 'audit_selector')
    previous = {tuple(r[k] for k in bank_keys): r for r in budget_banks[120]}
    for row in budget_banks[360]:
        key = tuple(row[k] for k in bank_keys)
        old = previous[key]
        if row['numeric_joint_inequalities'] != old['numeric_joint_inequalities']:
            changes.append({**dict(zip(bank_keys, key)), 'numeric_120': old['numeric_joint_inequalities'],
                            'numeric_360': row['numeric_joint_inequalities'], 'coverage_joint_360': row['joint_pass']})
    save_csv(out/'AUDIT_BUDGET_FEASIBILITY_CHANGES.csv', changes)
    parent_changes = []
    policy_keys = ('seed', 'split', 'release', 'audit_selector')
    previous = {tuple(r[k] for k in policy_keys): r for r in budget_policies[120]}
    for row in budget_policies[360]:
        key = tuple(row[k] for k in policy_keys)
        for target in ATTRIBUTES:
            old, new = previous[key]['attribute_halving'][target], row['attribute_halving'][target]
            if old['numeric_halving_inequality'] != new['numeric_halving_inequality']:
                parent_changes.append({**dict(zip(policy_keys, key)), 'target': target,
                    'numeric_120': old['numeric_halving_inequality'], 'numeric_360': new['numeric_halving_inequality'],
                    'gain_120': old['erased_attribute_gain'], 'gain_360': new['erased_attribute_gain'],
                    'parent_gain': new['parent_attribute_gain'], 'coverage_120': old['pass'], 'coverage_360': new['pass']})
    save_csv(out/'AUDIT_BUDGET_PARENT_CHANGES.csv', parent_changes)
    ranking = []
    ranking_releases = (*FINAL, 'C_init', 'D_init', 'PCA16', 'B_rich_bank', 'C_tree_bank')
    completed_seeds = sorted(selections[360])
    for seed in (*completed_seeds, 'mean'):
        for target in ATTRIBUTES:
            for selector in ('primary', 'catchup_inclusive'):
                for raw_split, split in SPLITS.items():
                    values = {}
                    for budget in (120, 360):
                        values[budget] = {}
                        for release in ranking_releases:
                            members = completed_seeds if seed == 'mean' else [seed]
                            losses = [get_value(indices[budget], s, 'audit', release, target, raw_split, selector=selector) for s in members]
                            values[budget][release] = statistics(losses)['mean']
                    for release in ranking_releases:
                        before, after = values[120][release], values[360][release]
                        rank = lambda budget, value: 1 + sum(finite(other) and other < value for other in values[budget].values()) if finite(value) else None
                        first, last = rank(120, before), rank(360, after)
                        ranking.append({'seed': seed, 'release': release, 'target': target, 'selector': selector, 'split': split,
                            'loss_120': before, 'loss_360': after, 'rank_120': first, 'rank_360': last,
                            'rank_360_minus_120': last - first if first is not None and last is not None else None,
                            'rank_changed': first != last,
                            'ranking_definition': 'ascending attack log loss among all 13 eligible interfaces; 1 = most measured recovery'})
    save_csv(out/'AUDIT_BUDGET_RANKINGS.csv', ranking)
    trajectory_fits = [r for r in exports[2] if r['audit_budget'] == 360 and
                       r['candidate_id'] in ('mlp_0', 'mlp_1', 'catchup')]
    noncontrol = [r for r in trajectory_fits if r['release'] not in ('prior', 'exposed')]
    controls = [r for r in trajectory_fits if r['release'] == 'exposed']
    endpoints = defaultdict(list)
    for fit in noncontrol:
        points = [r for r in exports[3] if r['audit_budget'] == 360 and
                  all(r[k] == fit[k] for k in ('seed', 'release', 'target', 'candidate_id'))]
        final = next(r['validation_log_loss'] for r in points if r['epoch'] == 360)
        best = min(r['validation_log_loss'] for r in points)
        endpoints[fit['candidate_id'], fit['target']].append(final - best)
    saturation = {'noncontrol_trajectories': len(noncontrol),
        'all_noncontrol_best_within_120': bool(noncontrol) and all(r['selected_epoch'] <= 120 for r in noncontrol),
        'noncontrol_selected_epoch_range': [min(r['selected_epoch'] for r in noncontrol), max(r['selected_epoch'] for r in noncontrol)] if noncontrol else None,
        'exposed_trajectories': len(controls), 'exposed_selected_at_360': sum(r['selected_epoch'] == 360 for r in controls),
        'endpoint_minus_best_validation_loss': [{'candidate_id': k[0], 'target': k[1], **statistics(v)} for k, v in endpoints.items()],
        'scope': 'fixed-trajectory epoch-budget sensitivity only; no universal or model-family saturation claim'}
    lines = ['# Nested audit budget sensitivity', '',
        'Fresh MLP and direct-coordinate catch-up checkpoints at 120 and 360 are selected from each same '
        '360-epoch trajectory by unweighted attacker-validation loss. Logistic/tree candidates remain fixed. '
        'Negative loss differences mean additional measured recovery. Validation gains need not transfer '
        'to development. Fresh and catch-up exposure remain separate; saved observers are diagnostic only.', '',
        '[Every candidate](EXTENDED_PER_TARGET.csv), [classes and exposed controls](EXTENDED_PER_CLASS.csv), '
        '[all curves](EXTENDED_CURVES.csv), [exposure and selected epochs](EXTENDED_FITTING.csv), '
        '[every paired budget score](AUDIT_BUDGET.csv).', '']
    for split in SPLITS.values():
        lines += [f"## {split.replace('_', ' ').capitalize()}: 360 − 120 loss", '']
        lines += table(['Release', 'Attribute', 'Selection', 'Mean ± sample SD'],
            [[LABELS.get(r['release'], r['release']), r['target'], r['selector'], mean_sd(r)] for r in aggregated
             if r['split'] == split and r['selector'] in ('primary', 'catchup_inclusive', 'candidate:catchup')])
        lines += ['']
    lines += ['## Changed numerical bank feasibility', '',
        'The utility margins stay fixed. Both unprotected rich banks and each new final release receive the '
        'same extended fresh fitting budget. PCA32, PCA32+LEACE and erased banks remain their named historical '
        'audit references because extending them was outside the authorized matrix. All budget-specific '
        'parent/bank criteria are in [EXTENDED_CRITERIA.json](EXTENDED_CRITERIA.json).', '']
    lines += table(['Seed', 'Split', 'Release', 'Bank', 'Selector', 'Numeric 120', 'Numeric 360', 'Coverage 360'],
        [[r['seed'], r['split'], LABELS[r['release']], LABELS[r['bank']], r['audit_selector'],
          status(r['numeric_120']), status(r['numeric_360']), status(r['coverage_joint_360'])] for r in changes])
    if not changes:
        lines += ['', 'No numerical joint feature/bank inequality changed under the saved budget selections.']
    lines += ['', '## Changed individual PCA32-parent numerical halvings', '',
        'PCA32 remains its historical 120-epoch audit parent. Source and residential predictions never change '
        'with an audit budget. Numeric attribute halvings are separated from support-gated assessments; '
        'all race assessments remain unassessable.', '']
    lines += table(['Seed', 'Split', 'Release', 'Selector', 'Attribute', 'Numeric 120', 'Numeric 360', 'Coverage 360'],
        [[r['seed'], r['split'], LABELS[r['release']], r['audit_selector'], r['target'],
          status(r['numeric_120']), status(r['numeric_360']), status(r['coverage_360'])] for r in parent_changes])
    if not parent_changes:
        lines += ['', 'No individual numerical parent-halving inequality changed.']
    lines += ['', '## Changed development recovery rankings', '',
        'Ranks are descriptive ascending mean attack log loss across all 13 authorized extension interfaces: '
        'rank 1 means most measured recovery. Utility is fixed. These are not combined utility/privacy scores, '
        'outcome-based candidate selections, or an ordering of protection guarantees. '
        '[Every seed, split, weighting and ranking](AUDIT_BUDGET_RANKINGS.csv) retains both budgets.', '']
    lines += table(['Split', 'Release', 'Attribute', 'Selector', 'Rank 120', 'Rank 360'],
        [[r['split'], LABELS[r['release']], r['target'], r['selector'], r['rank_120'], r['rank_360']]
         for r in ranking if r['seed'] == 'mean' and r['rank_changed'] and r['split'].startswith('development_evaluation')])
    if not any(r['rank_changed'] for r in ranking):
        lines += ['', 'No ranking changed for any seed, mean, split, weighting or audit selector.']
    lines += ['', '## Fitting-budget saturation and its limits', '',
        f"There are {len(noncontrol)} non-control fresh/catch-up trajectories. "
        f"Their selected epochs span {saturation['noncontrol_selected_epoch_range']}; "
        f"all keep their first-120 best checkpoint: {saturation['all_noncontrol_best_within_120']}. "
        f"Of {len(controls)} exposed-control MLP trajectories, {saturation['exposed_selected_at_360']} select epoch 360. "
        'Exact nested-prefix verification and actual last optimizer states are recorded separately in '
        '[EXTENDED_SCORE_REPLAY.json](EXTENDED_SCORE_REPLAY.json).', '',
        'The following fixed-trajectory endpoint costs distinguish continued fitting from a silently shortened run. '
        'Each row summarizes its release/seed trajectories; SD is descriptive variation across those trajectories. '
        'Positive values mean the final iterate has worse validation loss than its saved best checkpoint.', '']
    lines += table(['Candidate', 'Attribute', 'Epoch 360 − best validation loss, mean ± sample SD'],
        [[r['candidate_id'], r['target'], mean_sd(r)] for r in saturation['endpoint_minus_best_validation_loss']])
    lines += ['', 'The selected release audits receive no demonstrated benefit from this additional epoch budget. '
        'That resolves this predeclared budget sensitivity, not general audit saturation across architectures, '
        'data or optimization methods. Exposed-control fitting can still improve supported-class loss; '
        'race code 4 remains unassessable. No finite audit establishes privacy, and no further extension was run.', '']
    (out/'AUDIT_BUDGET.md').write_text('\n'.join(lines))
    return {'status': 'completed' if len(paths) == 3 else 'partial',
            'completed_seeds': [read_json(p)['seed'] for p in paths], 'paired_aggregates': aggregated,
            'score_rows': len(exports[0]), 'indices': indices, 'rows': rows, 'feasibility_changes': changes,
            'parent_halving_changes': parent_changes,
            'saturation': saturation,
            'mean_development_ranking_changes': [r for r in ranking if r['seed'] == 'mean' and r['rank_changed']
                                                and r['split'].startswith('development_evaluation')]}


def plot_reports(out, seeds, index, diagnostics, extended, curves=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {r: plt.get_cmap('tab10')(j % 10) for j, r in enumerate(FINAL)}
    reference_colors = {'PCA16': '#202020', 'E_pca': '#777777', 'E_pca_leace': '#9c755f',
                        'B_rich_bank': '#888888', 'C_tree_bank': '#bbbbbb', 'C_init': '#cc79a7', 'D_init': '#0072b2',
                        'C_bottleneck': '#b3b3b3', 'D_protected': '#555555'}
    shown = (*FINAL, 'C_init', 'D_init', 'PCA16', 'E_pca', 'E_pca_leace', 'C_bottleneck', 'D_protected', 'B_rich_bank', 'C_tree_bank')
    def mean_value(release, role, target, split, selector='primary'):
        values = [get_value(index, s, role, release, target, split, selector=selector) for s in seeds]
        return statistics(values)['mean']
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        for ax, split, title in zip(axes, ('test', 'test_person_weighted'), ('Unweighted primary', 'PWGTP: same predictions')):
            prior = mean_value('prior', 'audit', target, split)
            for j, r in enumerate(shown):
                x, attack = mean_value(r, 'transfer', 'same_residence', split), mean_value(r, 'audit', target, split)
                if not all(finite(v) for v in (x, attack, prior)):
                    continue
                color = colors.get(r, reference_colors.get(r))
                ax.scatter(x, prior - attack, color=color, marker='o' if r in FINAL else 'X', s=45, label=LABELS[r])
                offset = {'beta_1_C_persistent': (4, 7), 'beta_1_D_persistent': (4, -10),
                          'beta_0p1_D_persistent': (4, -10), 'PCA16': (-9, 7)}.get(r, (4, 3))
                ax.annotate(str(j+1), (x, prior-attack), xytext=offset, textcoords='offset points', fontsize=8)
            ax.set(title=title, xlabel='Residence log loss (lower is better)',
                   ylabel=f'{target} prior-relative gain (lower is better)')
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, [f'{j+1}. {v}' for j, v in enumerate(labels)], loc='lower center', ncol=4, fontsize=7)
        fig.suptitle(f'{target}: independent 120-epoch audit, development evaluation' + ('; all-race assessment support-limited' if target == 'RAC1P' else ''))
        fig.tight_layout(rect=(0, .25, 1, .93))
        fig.savefig(out/f'tradeoff_{target}.png', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    stage_order = ('I', 'W_historical', 'beta_0p1_W', 'beta_1_W', 'C_init', 'D_init', *FINAL)
    for ax, target in zip(axes.flat, UTILITY_TASKS):
        for split, marker, color, text in (('test', 'o', '#0072b2', 'Unweighted'), ('test_person_weighted', 's', '#d55e00', 'PWGTP')):
            values = [mean_value(r, 'transfer', target, split) for r in stage_order]
            ax.plot(range(len(stage_order)), [v if finite(v) else np.nan for v in values], marker=marker, linestyle='none', color=color, label=text)
        ax.set_title(TASK_DISPLAY[target]); ax.set_ylabel('Log loss')
        ax.set_xticks(range(len(stage_order)), ['I', 'W0', 'W.1', 'W1', 'C0', 'D0', '.1Cw', '.1Dw', '.1Cp', '.1Dp', '1Cw', '1Dw', '1Cp', '1Dp'], rotation=60, fontsize=7)
        ax.grid(axis='y', alpha=.2)
    axes.flat[-1].axis('off'); axes.flat[0].legend(fontsize=8)
    fig.suptitle('Fixed snapshot utility: all five tasks; w = warmup-only, p = persistent; C/D are parallel forks')
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out/'utility_stages.png', dpi=180); plt.close(fig)
    if diagnostics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        stages = ('I', 'beta_0p1_W', 'beta_1_W', *FINAL)
        for ax, split in zip(axes, ('fit', 'source_validation', 'development_evaluation')):
            for metric, marker, label in (('direct_mean_mse', 'o', 'Direct coordinate error'), ('affine_mean_mse', 's', 'Affine teacher prediction error'), ('prior_mean_mse', '^', 'Fitting prior')):
                values = [statistics(r.get(metric) for r in diagnostics if r['release'] == release and r['split'] == split)['mean'] for release in stages]
                ax.plot(range(len(stages)), [v if finite(v) else np.nan for v in values], marker=marker, linestyle='none', label=label)
            ax.set_xticks(range(len(stages)), ['I', 'W.1', 'W1', '.1Cw', '.1Dw', '.1Cp', '.1Dp', '1Cw', '1Dw', '1Cp', '1Dp'], rotation=60, fontsize=8)
            ax.set(title=split.replace('_', ' '), ylabel='Mean standardized squared error')
            ax.set_yscale('symlog', linthresh=.005); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle('Frozen snapshot teacher diagnostics; vertical scale linear below .005, logarithmic above')
        fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out/'preservation_affine.png', dpi=180); plt.close(fig)
    if extended.get('rows'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        releases = (*FINAL, 'C_init', 'D_init', 'PCA16', 'B_rich_bank', 'C_tree_bank')
        for ax, target in zip(axes, ATTRIBUTES):
            for selector, offset, label in (('primary', -.15, 'Fresh'), ('catchup_inclusive', .15, 'Fresh + catch-up')):
                values = [statistics(r['loss_360_minus_120'] for r in extended['rows'] if r['release'] == release and r['target'] == target and r['selector'] == selector and r['split'] == 'development_evaluation')['mean'] for release in releases]
                ax.scatter(np.arange(len(releases))+offset, [v if finite(v) else np.nan for v in values], s=30, label=label)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(title=target, ylabel='360 − 120 attack loss; negative = more recovery')
            ax.set_xticks(range(len(releases)), [LABELS[r] for r in releases], rotation=70, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'audit_budget.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    releases = (*FINAL, 'C_init', 'D_init')
    for ax, target in zip(axes, ATTRIBUTES):
        for selector, offset, label in (('candidate:saved_adversary', -.25, 'Saved before catch-up'),
                ('primary', -.08, 'Fresh selected'), ('candidate:catchup', .08, 'Catch-up'),
                ('catchup_inclusive', .25, 'Pooled selected')):
            values = [mean_value(r, 'audit', target, 'test', selector) for r in releases]
            ax.scatter(np.arange(len(releases))+offset, [v if finite(v) else np.nan for v in values], s=25, label=label)
        ax.set(title=target, ylabel='Development attack log loss (higher = less recovery)')
        ax.set_xticks(range(len(releases)), [LABELS[r] for r in releases], rotation=65, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'saved_fresh_catchup.png', dpi=180); plt.close(fig)
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax, release in zip(axes.flat, FINAL):
            for candidate, label in (('mlp_0', 'Fresh MLP 0'), ('mlp_1', 'Fresh MLP 1'), ('catchup', 'Catch-up')):
                subset = [r for r in curves if r['release'] == release and r['target'] == target and r['candidate_id'] == candidate]
                epochs = sorted({r['epoch'] for r in subset})
                vals = [statistics(r['validation_log_loss'] for r in subset if r['epoch'] == epoch)['mean'] for epoch in epochs]
                ax.plot(epochs, vals, label=label, linewidth=1.2)
            ax.set(title=LABELS[release], xlabel='Attacker-fitting epoch', ylabel='Validation log loss')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{target}: all fresh/catch-up learning curves, mean over seeds; inherited exposure differs')
        fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out/f'audit_curves_{target}.png', dpi=160); plt.close(fig)


def summarize(out):
    out = Path(out).resolve()
    config = read_json(out/'config.json')
    seeds = list(config.get('seeds', [0, 1, 2]))
    units, inputs, unit_status = [], [out/'config.json', out/'PROTOCOL.md'], []
    for beta in BETAS:
        for seed in seeds:
            path = out/beta/f'seed_{seed}'/'metrics.json'
            if not path.exists():
                continue
            record = read_json(path)
            if record['seed'] != seed:
                raise ValueError('Seed/path identity mismatch')
            selection_path = path.parent/'selection_before_test.json'
            units.append(normalize_unit(record, read_json(selection_path), beta))
            inputs.extend((path, selection_path))
            inputs.extend(p for p in (path.parent/'training/training.json', path.parent/'preservation_diagnostics.json') if p.exists())
            unit_status.append({'beta': beta, 'seed': seed, 'metrics': str(path.relative_to(out)),
                                'runtime': record.get('runtime'), 'integrity': record.get('integrity')})
    if not units:
        raise ValueError('No complete beta/seed metrics; reporter never launches scientific fitting')
    records, selections = merge_units(units)
    index = make_index(records)
    scores, classes, fits, curves = flatten(records, selections)
    aggregates = aggregate(seeds, index)
    paired = paired_rows(seeds, index)
    pairs = summarize_pairs(paired)
    criteria, policies, banks, coverage = criteria_rows(seeds, index, selections)
    primary = primary_rows(seeds, index, criteria)
    diagnostics, coordinates = diagnostic_rows(out)
    training, gradients = training_rows(out)
    catchup = catchup_rows(seeds, index)
    for filename, rows in (('PER_TARGET.csv', scores), ('PER_CLASS.csv', classes), ('FITTING.csv', fits),
            ('CURVES.csv', curves), ('PAIRED.csv', paired), ('PAIRED_AGGREGATE.csv', pairs),
            ('AGGREGATE.csv', aggregates), ('PRIMARY_COMPARISON.csv', primary), ('SUPPORT.csv', coverage),
            ('CATCHUP.csv', catchup), ('PRESERVATION_DIAGNOSTICS.csv', diagnostics), ('COORDINATE_ERRORS.csv', coordinates),
            ('TRAINING_CURVES.csv', training), ('GRADIENT_DIAGNOSTICS.csv', gradients)):
        save_csv(out/filename, rows)
    save_csv(out/'SELECTED_TASKS.csv', [r for r in scores if r['role'] == 'transfer' and r['selected']])
    save_json(out/'criteria.json', {'margins': MARGINS, 'fixed_parent': 'E_pca', 'per_seed_split': criteria,
                                  'final_policy_per_seed_split': policies})
    save_json(out/'bank_comparisons.json', {'margins': MARGINS, 'per_seed_split': banks})
    extended = extended_reports(out, records, selections, index)
    inputs.extend(p for p in (out/'progress.json',) if p.exists())
    inputs.extend(p for directory in (out/'extended').glob('seed_*') if (directory/'metrics.json').exists()
                  for p in (directory/'metrics.json', directory/'selection_before_test.json'))
    write_reports(out, seeds, primary, aggregates, pairs, criteria, policies, banks, coverage, unit_status, extended)
    plot_reports(out, seeds, index, diagnostics, extended, curves)
    extended.pop('indices', None)
    extended.pop('rows', None)
    summary = {'evaluation_status': 'DEVELOPMENT EVALUATION', 'declared_seeds': seeds, 'completed_units': unit_status,
        'core_complete': len(unit_status) == 6, 'margins': MARGINS, 'fixed_parent': 'E_pca',
        'unaudited_stages': list(STAGES), 'primary_audit': 'five independent candidates, unweighted validation choice',
        'weighting': 'unweighted primary and PWGTP sensitivity on identical selected predictions',
        'aggregate_metrics': aggregates, 'paired_aggregate_metrics': pairs, 'primary_comparison': primary,
        'extended_audit': extended, 'coordinate_diagnostics': diagnostics,
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in inputs},
        'report_source_sha256': {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('summarize_acs_protection.py'),
            Path(__file__).with_name('summarize_acs_bottleneck.py'), Path(__file__).with_name('summarize_acs_pca16_init.py'))},
        'row_counts': {'scores': len(scores), 'classes': len(classes), 'fits': len(fits), 'curves': len(curves),
                       'paired': len(paired), 'coordinate_scores': len(diagnostics)}}
    save_json(out/'summary.json', summary)
    print(json.dumps({'out': str(out), 'complete_units': len(unit_status), **summary['row_counts']}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)


if __name__ == '__main__':
    main()

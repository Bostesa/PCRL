"""Summarize saved ACS transfer metrics only; never fit or reselect models.

Usage: python scripts/summarize_acs_transfer.py --out RESULTS_DIRECTORY
The original per-seed metrics, selections and protocol are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

RELEASE_ORDER = ('A_binary_bank', 'B_rich_bank', 'C_tree_bank', 'D_features',
                 'D_compressed', 'E_pca', 'F_covariates', 'prior', 'exposed')
TARGET_ORDER = ('same_residence', 'commute_over20', 'SEX', 'RAC1P')
METRICS = ('log_loss', 'auroc', 'balanced_accuracy', 'accuracy', 'macro_auroc',
           'observed_balanced_accuracy', 'observed_macro_auroc')
SPLITS = ('validation', 'test', 'test_person_weighted')
COMPARATORS = ('B_rich_bank', 'C_tree_bank', 'E_pca', 'F_covariates')


def finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and math.isfinite(value)


def number(value):
    return f'{value:.6f}' if finite(value) else 'undefined'


def statistics(values):
    values = list(values)
    valid = [float(v) for v in values if finite(v)]
    complete = len(valid) == len(values) and bool(values)
    return {'n_seeds': len(values), 'n_defined': len(valid), 'complete': complete,
            'mean': float(np.mean(valid)) if complete else None,
            'sample_sd': float(np.std(valid, ddof=1)) if complete and len(valid) > 1 else None}


def mean_sd(record):
    if not record['complete']:
        return f"undefined ({record['n_defined']}/{record['n_seeds']} defined)"
    return f"{number(record['mean'])} ± {number(record['sample_sd'])}"


def order_key(value, choices):
    return (choices.index(value) if value in choices else len(choices), value)


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def save_csv(path, records):
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def schema_info(schema, seed, role, target, split):
    pool = 'test' if split.startswith('test') else ('downstream_validation' if role == 'transfer' else 'attacker_validation')
    return schema.get('seeds', {}).get(str(seed), {}).get('pools', {}).get(pool, {}).get('support', {}).get(target, {})


def class_name(schema, target, index):
    names = schema.get('audit_category_names', {}).get(target)
    if names is None:
        names = {'same_residence': ['different residence', 'same residence'],
                 'commute_over20': ['at most 20 minutes', 'over 20 minutes']}.get(target)
    return names[index] if names and index < len(names) else str(index)


def flatten(records, selections, schema):
    rows, classes, fallbacks = [], [], []
    seen = set()
    for record in records:
        seed = record['seed']
        for raw in record['raw_metrics']:
            key = f"{raw['role']}/{raw['release']}/{raw['target']}"
            identity = (seed, key, raw['family'])
            if identity in seen:
                raise ValueError(f'Duplicate candidate metric: {identity}')
            seen.add(identity)
            saved = selections.get(seed, {})
            if saved and raw['selected'] != (saved['head_selections'][key] == raw['family']):
                raise ValueError(f'Metric selection differs from saved selection: {identity}')
            metadata = saved.get('fitting_records', {}).get(key, {}).get('candidates', {}).get(raw['family'], {})
            common = {k: raw[k] for k in ('seed', 'role', 'release', 'target', 'family', 'selected')}
            common.update(effective_family=metadata.get('effective_family', raw['family']),
                          fallback_reason=metadata.get('fallback_reason'),
                          fit_rows=metadata.get('fit_rows'), fit_support=metadata.get('fit_support'),
                          fit_coverage_complete=metadata.get('fit_coverage_complete'))
            if metadata.get('fallback_reason'):
                fallbacks.append({**common, 'reason': metadata['fallback_reason']})
            for split in SPLITS:
                score = raw[split]
                coverage = schema_info(schema, seed, raw['role'], raw['target'], split)
                row = {**common, 'split': split,
                       'eligible_n': score['n'],
                       'missing_or_inapplicable': coverage.get('missing_or_inapplicable'),
                       **{k: v for k, v in score.items() if k != 'per_class'}}
                rows.append(row)
                for item in score.get('per_class', []):
                    j = item['class_index']
                    fit_support = metadata.get('fit_support')
                    classes.append({**common, 'split': split, 'class_name': class_name(schema, raw['target'], j),
                                    'fit_class_support': fit_support[j] if fit_support is not None else None,
                                    'evaluation_missing_or_inapplicable': coverage.get('missing_or_inapplicable'),
                                    **item})
    selected_groups = defaultdict(list)
    expected_groups = set()
    for row in rows:
        group_key = (row['seed'], row['role'], row['release'], row['target'], row['split'])
        expected_groups.add(group_key)
        if row['selected']:
            selected_groups[group_key].append(row)
    if set(selected_groups) != expected_groups or any(len(group) != 1 for group in selected_groups.values()):
        raise ValueError('Expected exactly one preselected family per seed/role/release/target')
    return rows, classes, fallbacks


def aggregate(rows, selected=False):
    groups = defaultdict(list)
    for row in rows:
        if selected and not row['selected']:
            continue
        family = 'validation_selected' if selected else row['family']
        groups[(row['role'], row['release'], row['target'], family, row['split'])].append(row)
    result = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row['seed'])
        result.append(dict(zip(('role', 'release', 'target', 'family', 'split'), key),
                           seeds=[row['seed'] for row in group],
                           selected_families={str(row['seed']): row['family'] for row in group} if selected else None,
                           coverage_complete_by_seed={str(row['seed']): row['coverage_complete'] for row in group},
                           metrics={metric: statistics(row.get(metric) for row in group) for metric in METRICS}))
    return result


def paired(rows):
    index = {(r['seed'], r['role'], r['release'], r['target'], r['split'], r['family']): r for r in rows}
    selected_index = {(r['seed'], r['role'], r['release'], r['target'], r['split']): r for r in rows if r['selected']}
    results = []
    for (seed, role, release, target, split), left in sorted(selected_index.items()):
        if release != 'D_features':
            continue
        for comparator in COMPARATORS:
            right = selected_index.get((seed, role, comparator, target, split))
            if right is None:
                continue
            modes = [('validation_selected', left, right)]
            for family in ('logistic', 'mlp', 'histgb'):
                a = index.get((seed, role, release, target, split, family))
                b = index.get((seed, role, comparator, target, split, family))
                if a is not None and b is not None:
                    modes.append((family, a, b))
            for mode, a, b in modes:
                results.append({'seed': seed, 'role': role, 'target': target, 'split': split,
                                'comparison': f'D_features minus {comparator}', 'family_mode': mode,
                                'D_family': a['family'], 'comparator_family': b['family'],
                                'differences': {metric: float(a[metric] - b[metric]) if finite(a.get(metric)) and finite(b.get(metric)) else None
                                                for metric in METRICS}})
    groups = defaultdict(list)
    for row in results:
        key = tuple(row[k] for k in ('role', 'target', 'split', 'comparison', 'family_mode'))
        groups[key].append(row)
    means = []
    for key, group in sorted(groups.items()):
        means.append(dict(zip(('role', 'target', 'split', 'comparison', 'family_mode'), key),
                          seeds=[r['seed'] for r in group],
                          differences={metric: statistics(r['differences'][metric] for r in group) for metric in METRICS}))
    return results, means


def selected_tables(rows, aggregates):
    text = ['# ACS transfer and attribute audit', '',
            'All primary rows use the family fixed by the relevant validation log loss. Lower log loss is better; higher AUROC/accuracy is better. Values are unweighted unless marked otherwise. Mean ± sample SD is descriptive across split seeds; undefined full-schema metrics remain undefined.', '']
    for role, title in (('transfer', 'Withheld tasks'), ('audit', 'Attribute audit')):
        text += ['## ' + title + ': selected models, per seed', '']
        current = [r for r in rows if r['selected'] and r['role'] == role and r['split'] == 'test']
        current.sort(key=lambda r: (order_key(r['target'], TARGET_ORDER), order_key(r['release'], RELEASE_ORDER), r['seed']))
        text += table(['Task', 'Release', 'Seed', 'Family', 'Val LL', 'Test LL', 'AUROC', 'Bal. acc.', 'Accuracy', 'N / missing', 'Class support'], [
            [r['target'], r['release'], r['seed'], r['family'],
             number(next(v['log_loss'] for v in rows if v['seed'] == r['seed'] and v['role'] == role and v['release'] == r['release'] and v['target'] == r['target'] and v['family'] == r['family'] and v['split'] == 'validation')),
             number(r['log_loss']), number(r['auroc']), number(r['balanced_accuracy']), number(r['accuracy']),
             f"{r['n']} / {r['missing_or_inapplicable'] if r['missing_or_inapplicable'] is not None else 'unknown'}", ','.join(map(str, r['support']))]
            for r in current])
        text += ['', '## ' + title + ': selected-model means ± sample SD', '']
        means = [r for r in aggregates if r['role'] == role and r['split'] == 'test']
        means.sort(key=lambda r: (order_key(r['target'], TARGET_ORDER), order_key(r['release'], RELEASE_ORDER)))
        text += table(['Task', 'Release', 'Seeds', 'Log loss', 'AUROC', 'Balanced accuracy', 'Accuracy'], [
            [r['target'], r['release'], len(r['seeds']), *[mean_sd(r['metrics'][m]) for m in ('log_loss', 'auroc', 'balanced_accuracy', 'accuracy')]] for r in means])
        text += ['']
    return text


def diagnostics_tables(records, source_details):
    text = ['## Source models and release costs', '',
            'A/B/D share the source encoder cost; table times are component totals per seed, not additive per-release costs. PCA/compression and downstream/audit costs are separate. Source test is descriptive and selects nothing.', '']
    text += table(['Seed', 'Encoder LR', 'Tree leaves', 'Encoder source LL', 'Tree source LL', 'Total s', 'Encoder search s', 'Tree search s', 'Downstream/audit s'], [
        [r['seed'], r['source_selection']['lr'], r['source_selection']['tree_leaves'],
         number(r['source_test']['encoder']['mean_source_loss']), number(r['source_test']['tree']['mean_source_loss']),
         *[number(r['runtime'][k]) for k in ('total_seconds', 'source_encoder_search_seconds', 'source_tree_search_seconds', 'downstream_and_audit_seconds')]] for r in records])
    text += ['', '### Release size and extraction', '']
    text += table(['Seed', 'Release', 'Dimensions', 'Bytes/record', 'Test rows', 'Test extraction s', 'Joint bank supported'], [
        [r['seed'], name, meta['dimension'], meta['bytes_per_record'], meta['test_rows'], number(meta['test_inference_seconds']),
         source_details.get(r['seed'], {}).get('joint_supported', 'unknown') if name in ('B_rich_bank', 'C_tree_bank') else 'n/a']
        for r in records for name, meta in sorted(r['release_metadata'].items(), key=lambda item: order_key(item[0], RELEASE_ORDER))])
    text += ['', '### Source task test scores', '']
    text += table(['Seed', 'Source family', 'Target', 'Log loss', 'Accuracy', 'Valid / missing', 'Class support'], [
        [r['seed'], family, target, number(score['log_loss']), number(score['accuracy']),
         f"{score['known']} / {score['missing']}", ','.join(map(str, score['class_support']))]
        for r in records for family, source in r['source_test'].items() for target, score in source['heads'].items()])
    return text + ['']


def analysis_text(summary, rows):
    text = ['# ACS transfer evidence summary', '',
            'This report summarizes the frozen scientific comparison. See [RESEARCH_DECISION.md](RESEARCH_DECISION.md) for the research interpretation and next-step decision. No release is selected using final test outcomes. The comparison concerns two specified withheld task identities on contemporaneous survey records; it does not establish reusable utility for arbitrary tasks, longitudinal prediction, privacy protection, or PCRL novelty.', '',
            '## Paired selected-head comparisons', '',
            'Differences are D_features minus the named comparator on identical test examples. Negative log-loss differences favor D; positive AUROC/accuracy differences favor D. The predeclared −0.01-nat reference is descriptive, not a significance test. Fitting and selection remain unweighted.', '']
    chosen = [r for r in summary['paired_per_seed'] if r['role'] == 'transfer' and r['split'] == 'test' and r['family_mode'] == 'validation_selected']
    text += table(['Task', 'Comparison', 'Seed', 'D / comparator head', 'Δ log loss', 'Δ AUROC', 'Δ balanced acc.', 'Δ accuracy'], [
        [r['target'], r['comparison'], r['seed'], f"{r['D_family']} / {r['comparator_family']}",
         *[number(r['differences'][m]) for m in ('log_loss', 'auroc', 'balanced_accuracy', 'accuracy')]] for r in chosen])
    text += ['', '### Paired means ± sample SD', '']
    pairs = [r for r in summary['paired_aggregates'] if r['role'] == 'transfer' and r['split'] == 'test' and r['family_mode'] == 'validation_selected']
    text += table(['Task', 'Comparison', 'Δ log loss', 'Δ AUROC', 'Δ balanced acc.', 'Δ accuracy'], [
        [r['target'], r['comparison'], *[mean_sd(r['differences'][m]) for m in ('log_loss', 'auroc', 'balanced_accuracy', 'accuracy')]] for r in pairs])
    text += ['', '### Matched-family transfer checks', '',
             'Both members of each pair use the named family, with each model checkpoint still selected on its own validation data. All per-seed matched-family differences are retained in summary.json.', '']
    pairs = [r for r in summary['paired_aggregates'] if r['role'] == 'transfer' and r['split'] == 'test' and r['family_mode'] != 'validation_selected']
    text += table(['Task', 'Comparison', 'Head family', 'Δ log loss', 'Δ AUROC'], [
        [r['target'], r['comparison'], r['family_mode'], mean_sd(r['differences']['log_loss']), mean_sd(r['differences']['auroc'])] for r in pairs])
    text += ['', '## PWGTP sensitivity on unchanged selected predictions', '',
             'Weighted scores are sensitivity analyses of this sampled benchmark cohort, not official survey estimates. No weighted validation score exists because selection was unweighted.', '']
    weighted = [r for r in summary['selected_aggregates'] if r['split'] == 'test_person_weighted']
    text += table(['Role', 'Task', 'Release', 'Weighted LL', 'Weighted AUROC', 'Weighted balanced acc.', 'Weighted accuracy'], [
        [r['role'], r['target'], r['release'], *[mean_sd(r['metrics'][m]) for m in ('log_loss', 'auroc', 'balanced_accuracy', 'accuracy')]] for r in weighted])
    text += ['', '## Coverage, controls, and interpretation limits', '',
             'PER_TARGET.csv retains every candidate family, its saved selected flag, fitting support/fallback, unweighted validation and test metrics, weighted test sensitivity, prevalence and missing counts. PER_CLASS.csv retains every fixed-schema class, including absent classes. Observed-class summaries have distinct names and do not replace undefined full-schema balanced accuracy or macro AUROC.', '',
             'Prior controls use only designated fitting-label frequencies with the declared smoothing. Exposed SEX/RAC1P one-hot controls test attacker competence; they are diagnostics with label access. Raw covariates and PCA are information-retention references. Probability-vector simplex redundancy means matching stored dimensions does not match intrinsic capacity.', '',
             'The three splits reuse a common sampled cohort, so their sample SD is descriptive rather than independent-sample or survey-design uncertainty. Age, hours, and retained covariates may correlate with source tasks and audited attributes. Attribute recoverability is measured without a protection objective or privacy pass/fail threshold.', '',
             f"Completed seeds: {summary['seeds']}. Declared seeds: {summary['declared_seeds']}. Missing completed seed records: {summary['missing_seeds']}.",
             f"Candidate prior fallbacks recorded: {len(summary['fallbacks'])}. Source/release/selection integrity passed for every completed seed: {summary['integrity_complete']}.", '',
             'Undefined aggregates are not averaged away: summary.json records n_defined and n_seeds; a mean is null if any contributing seed is undefined. A one-seed sample SD is undefined.', '',
             '## Runtime and provenance', '',
             f"Sum of recorded per-seed total runtime: {number(summary['runtime_total_seconds'])} seconds. Process wall timings, schema preparation, and artificial plumbing-check costs are separate artifacts. Configuration and exact input file hashes are copied into summary.json; raw records and fitted objects are never opened by this script.", '']
    return text


def plot_transfer(aggregates, out):
    targets = TARGET_ORDER[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, target in zip(axes, targets):
        current = [r for r in aggregates if r['role'] == 'transfer' and r['target'] == target and r['split'] == 'test']
        current.sort(key=lambda r: order_key(r['release'], RELEASE_ORDER))
        labels = [r['release'] for r in current]
        means = [r['metrics']['log_loss']['mean'] if finite(r['metrics']['log_loss']['mean']) else np.nan for r in current]
        errors = [r['metrics']['log_loss']['sample_sd'] or 0. for r in current]
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=errors, fmt='o', color='#245b91', capsize=3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel('Selected-head test log loss (nats; lower is better)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(target.replace('_', ' '))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Frozen ACS task-identity transfer: mean ± sample SD', fontsize=12)
    fig.text(.5, .01, 'Same sampled cohort across seeds; SD is descriptive. Single-seed SD is unavailable.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(out/'transfer.png', dpi=180)
    fig.savefig(out/'transfer.pdf')
    plt.close(fig)


def summarize(out):
    out = Path(out)
    paths = sorted(out.glob('seed_*/metrics.json'), key=lambda p: int(p.parent.name.split('_')[-1]))
    if not paths:
        raise ValueError(f'No completed seed metrics found in {out}')
    records = [read_json(path) for path in paths]
    seeds = [r['seed'] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seed records')
    cfg = read_json(out/'config.json', {})
    schema = read_json(out/'schema_support.json', {})
    selection_paths = {r['seed']: out/f"seed_{r['seed']}/selection_before_test.json" for r in records}
    selections = {seed: read_json(path, {}) for seed, path in selection_paths.items()}
    source_paths = {r['seed']: out/f"seed_{r['seed']}/source_selection.json" for r in records}
    source_details = {seed: {k: v for k, v in read_json(path, {}).items()
                            if k not in ('encoder_candidates', 'tree_candidates')}
                      for seed, path in source_paths.items()}
    flat, per_class, fallbacks = flatten(records, selections, schema)
    all_aggregates = aggregate(flat)
    selected_aggregates = aggregate(flat, selected=True)
    differences, paired_means = paired(flat)
    input_paths = [*paths, *[p for p in [*selection_paths.values(), *source_paths.values()] if p.exists()],
                   *[out/name for name in ('config.json', 'schema_support.json', 'protocol_freeze.json') if (out/name).exists()]]
    summary = {'seeds': seeds, 'declared_seeds': cfg.get('seeds', seeds),
               'missing_seeds': sorted(set(cfg.get('seeds', seeds)) - set(seeds)),
               'configuration': cfg, 'input_sha256': {str(p.relative_to(out)): sha(p) for p in input_paths},
               'metric_definitions': {'log_loss': 'Natural-log loss; fixed probability floor from raw scorer, lower better.',
                    'auroc': 'Binary positive-class AUROC or full fixed-schema macro OvR AUROC.',
                    'weighted': 'PWGTP changes test scoring only; fitting and selection unweighted.',
                    'selection': 'Saved validation-family selection, no recomputation from test.',
                    'difference': 'D_features minus comparator; negative log-loss difference favors D.'},
               'all_family_aggregates': all_aggregates, 'selected_aggregates': selected_aggregates,
               'paired_per_seed': differences, 'paired_aggregates': paired_means, 'fallbacks': fallbacks,
               'source_metrics': {str(r['seed']): r['source_test'] for r in records},
               'source_selection': {str(r['seed']): r['source_selection'] for r in records},
               'source_protocol': {str(seed): value for seed, value in source_details.items()},
               'release_metadata': {str(r['seed']): r['release_metadata'] for r in records},
               'runtime': {str(r['seed']): r['runtime'] for r in records},
               'runtime_total_seconds': sum(r['runtime']['total_seconds'] for r in records),
               'integrity': {str(r['seed']): r['integrity'] for r in records},
               'integrity_complete': all(all(r['integrity'].get(k, False) for k in
                    ('source_unchanged', 'development_releases_unchanged', 'selection_still_unchanged')) for r in records)}
    save_csv(out/'PER_TARGET.csv', flat)
    save_csv(out/'PER_CLASS.csv', per_class)
    save_json(out/'summary.json', summary)
    content = selected_tables(flat, selected_aggregates) + diagnostics_tables(records, source_details)
    (out/'TABLE.md').write_text('\n'.join(content).rstrip() + '\n')
    (out/'ANALYSIS.md').write_text('\n'.join(analysis_text(summary, flat)).rstrip() + '\n')
    plot_transfer(selected_aggregates, out)
    print(json.dumps({'seeds': seeds, 'metric_rows': len(flat), 'class_rows': len(per_class),
                      'paired_rows': len(differences), 'output': str(out)}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.out)


if __name__ == '__main__':
    main()

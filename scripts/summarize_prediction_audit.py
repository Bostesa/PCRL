"""Report immutable prediction audit metrics; performs no fitting or selection."""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = ('p1_V', 'p1_S', 'p2_U', 'p2_S', 'combined_S')
FAMILIES = ('linear', 'mlp', 'histgb')


def values(row, split='test'):
    e = row[split]
    result = {f'utility_{t}': e['task']['mlp'][t]['r2'] for t in ('p1_U', 'p2_V')}
    for family in FAMILIES:
        result[f'individual_{family}'] = max(e['attack'][family][t]['r2'] for t in TARGETS[:4])
        result[f'combined_{family}'] = e['attack'][family]['combined_S']['r2']
        for t in TARGETS:
            result[f'{family}_{t}'] = e['attack'][family][t]['r2']
    if 'direct_task' in row:
        result.update({f'direct_{t}': row['direct_task'][split][t]['r2'] for t in ('p1_U', 'p2_V')})
    return result


def stats(rows):
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'sample_sd': float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else None,
                  'n_seeds': len(rows)} for key in rows[0]}


def text_stat(value):
    return f"{value['mean']:.6f}" + (f" ± {value['sample_sd']:.6f}" if value['sample_sd'] is not None else ' (n=1)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    seeds = [json.loads(p.read_text()) for p in sorted(out.glob('seed_*/metrics.json'))]
    if not seeds:
        raise RuntimeError('No completed audit seeds')
    groups = {}
    for seed in seeds:
        for key, row in seed['methods'].items():
            groups.setdefault(key, []).append(values(row))
    summary = {key: stats(rows) for key, rows in groups.items()}
    paired = []
    for seed in seeds:
        left, right = (values(seed['methods'][k]) for k in ('prediction_only', 'oracle'))
        paired.append({'seed': seed['seed'], **{key: left[key]-right[key] for key in left}})
    paired_summary = stats([{k: v for k, v in r.items() if k != 'seed'} for r in paired])
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'paired_comparisons.json').write_text(json.dumps({'prediction_minus_oracle': paired,
                                             'summary': paired_summary}, indent=2)+'\n')
    lines = ['# Stronger frozen prediction audit', '',
        'Utility is independently fitted MLP task R²; direct native predictions appear separately. ',
        'Individual leakage is the largest of the four prohibited relationships within the named family.',
        'All attackers/checkpoints/family choices were frozen using validation before the fresh test.', '',
        '| Seed | Release | U | V | Individual linear | MLP | Trees | Combined S linear | MLP | Trees | Validation feasible | Test feasible |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for seed in seeds:
        for key, row in seed['methods'].items():
            v = values(row)
            cells = [v[f'utility_{t}'] for t in ('p1_U', 'p2_V')]
            cells += [v[f'individual_{f}'] for f in FAMILIES]
            cells += [v[f'combined_{f}'] for f in FAMILIES]
            lines.append(f"| {seed['seed']} | {key} | " + ' | '.join(f'{x:.6f}' for x in cells) +
                         f" | {row['assessment_validation']['feasible']} | {row['assessment_test']['feasible']} |")
    lines += ['', 'Oracle and exposed-target rows are diagnostics. Full E is the unchanged seed2 E_dual_0.01 checkpoint; it was protection-infeasible historically.',
              '', '## Native scalar prediction utility', '', '| Seed | Validation U | Validation V | Test U | Test V |', '|---|---:|---:|---:|---:|']
    for seed in seeds:
        row = seed['methods']['prediction_only']
        v = [row['direct_task'][split][t]['r2'] for split in ('validation', 'test') for t in ('p1_U', 'p2_V')]
        lines.append(f"| {seed['seed']} | " + ' | '.join(f'{x:.6f}' for x in v) + ' |')
    lines += ['', '## Across-seed mean ± sample SD', '', '| Release | U | V | Worst individual linear / MLP / trees | Combined S linear / MLP / trees |',
              '|---|---:|---:|---|---|']
    for key, row in summary.items():
        cells = [text_stat(row[f'utility_{t}']) for t in ('p1_U', 'p2_V')]
        cells += [' / '.join(text_stat(row[f'{role}_{f}']) for f in FAMILIES) for role in ('individual', 'combined')]
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Three seeds are preliminary; no significance claim. Mean/SD never replaces per-seed feasibility. Negative R² is not negative information.',
              'Every target, family and validation-selected family test score is in PER_TARGET.md/csv and per-seed metrics. All fixed-family test results are shown; test does not select attackers.',
              '', '## Compute and interface', '', '| Seed | Release | Dimensions P1/P2/combined | Cached bytes per row | Fit + task probe seconds | Release extraction 4096 rows, seconds | Audit + task inference 4096 rows, seconds |',
              '|---|---|---|---|---:|---:|---:|']
    for seed in seeds:
        for key, row in seed['methods'].items():
            lines.append(f"| {seed['seed']} | {key} | {row['release_dimensions']} | {row['actual_cached_bytes_per_row']} | {row['audit_and_task_fit_seconds']:.3f} | {row['release_inference_seconds_test_4096_rows']:.6f} | {row['audit_and_task_inference_seconds_test_4096_rows']:.6f} |")
    lines += ['', 'Cache arrays are float64. Native prediction values originate in float32, so scalar releases can be serialized in4 bytes/purpose (8 combined); oracle/full-reference precision is not changed for evaluation.',
              'Short extraction timings are local single-batch observations including Python overhead, not deployment throughput guarantees. No representation training or eraser fitting was performed.']
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')
    detail = ['# Every forbidden relationship', '', '| Seed | Release | Target | Family | Validation R² | Fresh-test R² | Validation-selected family |',
              '|---|---|---|---|---:|---:|---|']
    csv_rows = []
    for seed in seeds:
        for key, row in seed['methods'].items():
            for target in TARGETS:
                for family in FAMILIES:
                    val, test = (row[split]['attack'][family][target] for split in ('validation', 'test'))
                    selected = row['selected_family'][target] == family
                    detail.append(f"| {seed['seed']} | {key} | {target} | {family} | {val['r2']:.6f} | {test['r2']:.6f} | {selected} |")
                    csv_rows.append({'seed': seed['seed'], 'release': key, 'target': target, 'family': family,
                                     'validation_r2': val['r2'], 'test_r2': test['r2'], 'selected_on_validation': selected,
                                     'test_mse': test['mse'], 'test_variance': test['variance'], 'test_n': test['n']})
    (out/'PER_TARGET.md').write_text('\n'.join(detail)+'\n')
    with (out/'PER_TARGET.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader(); writer.writerows(csv_rows)
    scalar_passes = sum(s['methods']['prediction_only']['assessment_test']['feasible'] for s in seeds)
    scalar_val = sum(s['methods']['prediction_only']['assessment_validation']['feasible'] for s in seeds)
    analysis = ['# Closing audit analysis', '',
        f'Prediction-only passes {scalar_val}/{len(seeds)} validation and {scalar_passes}/{len(seeds)} fresh-test audits under the frozen per-purpose utility and three-family leakage criteria.',
        'The unchanged native scalar predictions, longer fresh MLP fits, and complementary trees close the declared audit. This is empirical sufficiency for the two fixed tasks under tested attacks, not universal privacy or sufficiency for unknown future tasks.', '',
        '## Per-target scalar versus oracle differences', '',
        'Paired prediction-only minus oracle on the common fresh test; mean ± sample SD, no significance claim.', '',
        '| Metric | Paired difference |', '|---|---:|']
    for key, row in paired_summary.items():
        analysis.append(f'| {key} | {text_stat(row)} |')
    analysis += ['', '## Frozen full-representation positive control', '',
                 '| Target | Saved training validation/test | Saved continued validation/test | Fresh MLP validation/test | Trees validation/test |',
                 '|---|---|---|---|---|']
    for seed in seeds:
        for key, row in seed['methods'].items():
            if 'prior_adversary_references' not in row:
                continue
            for target in TARGETS:
                cells = [' / '.join(f"{row['prior_adversary_references'][split][kind][target]['r2']:.6f}" for split in ('validation', 'test'))
                         for kind in ('saved_training', 'saved_continued')]
                cells += [' / '.join(f"{row[split]['attack'][kind][target]['r2']:.6f}" for split in ('validation', 'test')) for kind in ('mlp', 'histgb')]
                analysis.append(f'| {target} | ' + ' | '.join(cells) + ' |')
    analysis += ['', 'The full control retains its historical infeasibility; the audit does not improve protection. No final map was refitted. The saved continued weights retain prior exposure and are compatible only with the full release.', '',
        '## Budget and limits', '',
        'Fresh MLPs use two1800-step trajectories per view, each460800 row presentations, totaling921600. This exceeds900+640 updates per successful continued trajectory but uses2048 distinct attacker examples; original training weights also saw3072 other examples. Input dimensions and output sharing differ, so equal nominal updates cannot establish equal attack strength. HistGB uses200 full fitting-data passes per target; these are separately counted, not converted into Adam updates.',
        'Task probes fit4096 representation-training examples, attackers fit2048 attacker examples, validation2048 selects checkpoints/restarts, and fresh test4096 is generated after saved selection. Observation preprocessing and all release tensors remain unchanged. Covariance certificates play no role.',
        'Reused validation makes this an exploratory diagnosis. A failed attacker does not bound all attacks. Exposed-target controls check competence including S, and oracle controls check finite-sample spurious predictive success.', '',
        'The original toy benchmark stops here. The research decision must depend on a defensible real application and strong simple controls, not another protection-strength sweep.', '',
        '## Measured internal runtime', '']
    analysis += [f"- Seed{s['seed']}: {s['runtime_seconds']:.3f}s." for s in seeds]
    analysis.append(f"- Sum: {sum(s['runtime_seconds'] for s in seeds):.3f}s; process wall timings are in execution logs.")
    (out/'ANALYSIS.md').write_text('\n'.join(analysis)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    colors = {'oracle': '#4c78a8', 'prediction_only': '#59a14f', 'full_E_dual_0.01': '#e15759'}
    for ax, role, limit in zip(axes, ('individual', 'combined'), (.05, .10)):
        for index, key in enumerate(('oracle', 'prediction_only', 'full_E_dual_0.01')):
            if key not in summary: continue
            means = [summary[key][f'{role}_{f}']['mean'] for f in FAMILIES]
            sds = [summary[key][f'{role}_{f}']['sample_sd'] or 0 for f in FAMILIES]
            ax.errorbar(np.arange(3)+(index-1)*.12, means, yerr=sds, fmt='o', capsize=3, color=colors[key], label=key)
        ax.axhline(limit, linestyle='--', color='black', linewidth=1, label='Declared leakage threshold')
        ax.set_xticks(range(3), ['Affine OLS', 'MLP', 'Boosted trees'])
        ax.set_title('Worst individual forbidden R²' if role == 'individual' else 'Combined S predictive R²')
        ax.set_ylabel('Fresh-test R² (negative scores retained)')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Unchanged releases; three-seed mean ± sample SD (full reference: seed2 only)')
    fig.savefig(out/'audit_scores.png', dpi=180)
    fig.savefig(out/'audit_scores.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()

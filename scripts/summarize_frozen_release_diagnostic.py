"""Report immutable frozen-release diagnostics; no fitting or selection."""
import argparse
import json
from pathlib import Path
import statistics


def plot_learning_curves(out, raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed = next((r for r in raw if r['seed'] == 2), None)
    if seed is None:
        return
    row = seed['methods']['E_dual_0.01']
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharex=True)
    for target, ax in zip(TARGETS, axes):
        ax.axhline(row['validation']['A'][target]['r2'], color='black', linestyle=':', label='A saved')
        for method, color in [('B', '#0072B2'), ('C', '#D55E00')]:
            curves = [c['validation_curve'] for c in row['selection'][method]['candidates']]
            steps = [p['optimizer_steps'] for p in curves[0]]
            values = [max(c[j]['validation_r2'][target] for c in curves) for j in range(len(steps))]
            ax.plot(steps, values, label=f'{method} '+('continued' if method == 'B' else 'fresh'), color=color)
        p = 2 if target.startswith('combined') else int(target[1])-1
        meta = row['independent_audit_reference'][f'{p}_mlp']
        j = meta['target_names'].index(target)
        variance = row['validation']['D'][target]['variance']
        grouped = {}
        for point in meta['history']:
            grouped.setdefault(point['optimizer_steps'], []).append(1-point['validation_mse'][j]/variance)
        ax.plot(sorted(grouped), [max(grouped[s]) for s in sorted(grouped)], color='#009E73', label='D original audit')
        ax.set_title(target)
        ax.set_xlabel('Updates per trajectory')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation predictive R²')
    axes[-1].legend(fontsize=8)
    figure.suptitle('Seed 2 E .01: fixed final release; strongest candidate at each common update count')
    figure.tight_layout()
    figure.savefig(out/'learning_curves.png', dpi=160)
    figure.savefig(out/'learning_curves.pdf')
    plt.close(figure)

TARGETS = ('p1_V', 'p1_S', 'p2_U', 'p2_S', 'combined_S')
METHODS = ('A', 'B', 'C', 'D')


def summarize(out):
    raw = [json.loads(p.read_text()) for p in sorted(out.glob('seed_*/metrics.json'))]
    table = ['# Frozen-release adversary diagnostic', '',
        'A: saved training adversary; B: continued saved weights; C: fresh weights; D: saved independent auditor.',
        'B/C share two reset-Adam trajectories, 640 added updates each, fixed saved coordinates and validation-only per-target selection.',
        'D is re-evaluated on the common new test, not copied from old test scores. All releases are unchanged.', '',
        '## Fresh diagnostic test: every forbidden target', '',
        '| Seed | Release | Target | A | B | C | D | B−A | C−A | D−B | D−C |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    paired = []
    for seed in raw:
        for key, row in seed['methods'].items():
            for target in TARGETS:
                vals = [row['test'][m][target]['r2'] for m in METHODS]
                a, b, c, d = vals
                deltas = [b-a, c-a, d-b, d-c]
                table.append(f'| {seed["seed"]} | {key} | {target} | '+' | '.join(f'{v:.6f}' for v in vals+deltas)+' |')
                paired.append({'seed': seed['seed'], 'release': key, 'target': target,
                    **dict(zip(METHODS, vals)), **dict(zip(('B_minus_A','C_minus_A','D_minus_B','D_minus_C'), deltas))})
    table += ['', 'Seed1 D/E weight.1 rows remain utility-infeasible diagnostic fallbacks; seed0/2 weight.01 were validation utility-eligible but protection-infeasible.', '',
        '## Same-example validation comparison', '',
        '| Seed | Release | Target | A | B | C | D |', '|---|---|---|---:|---:|---:|---:|']
    for seed in raw:
        for key, row in seed['methods'].items():
            for target in TARGETS:
                table.append(f'| {seed["seed"]} | {key} | {target} | '+' | '.join(f'{row["validation"][m][target]["r2"]:.6f}' for m in METHODS)+' |')
    table += ['', '## Means ± sample SD across seeds (including explicitly diagnostic seed1)', '',
        '| Release family | Target | A | B | C | D |', '|---|---|---:|---:|---:|---:|']
    for family in ('D_', 'E_'):
        for target in TARGETS:
            chosen = [r for r in paired if r['release'].startswith(family) and r['target'] == target]
            if len(chosen) > 1:
                cells = [f'{statistics.mean(r[m] for r in chosen):.6f} ± {statistics.stdev(r[m] for r in chosen):.6f}' for m in METHODS]
                table.append(f'| {family[0]} | {target} | '+' | '.join(cells)+' |')
    analysis = ['# Frozen-release diagnostic analysis', '',
        'The frozen protocol and INSPECTION.md describe the original discrepancy, training exposure and coordinate conventions. Tables show each target on identical examples.', '',
        '## Last-training versus final eraser (validation only)', '',
        '| Seed | Release | Target | Map375/stats375 | Map400/stats400 | Final/stats375 | Final/stats400 (A) |',
        '|---|---|---|---:|---:|---:|---:|']
    findings = ['## What the evidence supports', '',
        f'Continuing the saved weights improves fresh-test R² on {sum(r["B"] > r["A"] for r in paired)}/{len(paired)} identical-target/checkpoint pairs, '
        f'with gains from {min(r["B_minus_A"] for r in paired):.6f} to {max(r["B_minus_A"] for r in paired):.6f}. '
        f'B exceeds fresh C on {sum(r["B"] > r["C"] for r in paired)}/{len(paired)} pairs, and exceeds independent audit D on '
        f'{sum(r["B"] > r["D"] for r in paired)}/{len(paired)}. Fresh C exceeds D on {sum(r["C"] > r["D"] for r in paired)}/{len(paired)}.', '',
        'Thus saved weights can recover the missed leakage with additional fitting on the fixed final release. Fresh initialization does not help substantially more; inherited weights usually help under the matched added budget. This is not a comparison of equal lifetime exposure.', '',
        'The original hidden widths, prohibited-only output heads, loss and saved preprocessing are sufficient to match/exceed D after fitting. Architecture or coordinate differences are therefore not necessary to explain this miss. Their separate effects remain unmeasured; no capacity/loss factorial experiment was run. The predeclared seed2 preprocessing bridge did not trigger because no residual validation gap exceeded .05.', '',
        'The final eraser does create a transfer mismatch: on seed2 E.01, P2_U validation R² falls from .192596 (map375/stats375) to .069501 (map400/stats400) to .047024 (finalmap/stats400). Merely using375stats on the finalmap gives .052381. The exact original audit .252487 is also P2_U on the same2048examples. Map changes can change released information, so these sensitivity results do not establish a unique causal decomposition.', '',
        'The bounded conclusion is that an under-fitted/deployment-mismatched adversary missed recoverable leakage; it was not incapable of representing a stronger attack. Evidence cannot identify moving representations as the sole cause or separate the added fitting pool, optimizer reset and validation selection from extra updates.', '',
        'One next action: add a mandatory frozen-final-release catch-up audit before future protection decisions, using this saved-coordinate replay and per-target validation selection. Validate that evaluation change on existing checkpoints before considering encoder retraining. Keep prediction-only release as the simpler fixed-task solution.', '',
        'Learning curves are saved as learning_curves.png/pdf. Lines show the strongest candidate at each common count, not final-test-selected curves. B/C counts are added updates (B already had900 prior updates); D counts are original audit updates. Each method has two trajectories.', '']
    analysis[4:4] = findings
    for seed in raw:
        for key, row in seed['methods'].items():
            mm = row['eraser_mismatch_validation']
            for t in TARGETS:
                vals = [mm[f'training_map_{s}_own_coordinates'][t]['r2'] for s in (375, 400)]
                vals += [mm[f'final_map_coordinates_{s}'][t]['r2'] for s in (375, 400)]
                analysis.append(f'| {seed["seed"]} | {key} | {t} | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    analysis += ['', 'All map comparisons hold the final encoder and saved adversary fixed. Map375 was fitted to an earlier encoder; these are deployment-interface sensitivities, not reconstruction of unsaved last joint-training examples. The final50 adversary updates used map375, then encoder400/refresh400 occurred with no further adversary update. Changing maps may change information as well as coordinates.', '',
        '## Oracle and exposed-target controls: fresh test', '',
        '| Seed | Control | Family | Target | R² |', '|---|---|---|---|---:|']
    for seed in raw:
        for key, row in seed['controls'].items():
            for kind, values in row['test'].items():
                for target, score in values.items():
                    analysis.append(f'| {seed["seed"]} | {key} | {kind} | {target} | {score["r2"]:.6f} |')
    analysis += ['', '## Budget, bridge and integrity', '']
    for seed in raw:
        analysis.append(f'- Seed{seed["seed"]}: {seed["runtime_seconds"]:.3f}s internal runtime.')
        for key, row in seed['methods'].items():
            analysis.append(f'  {key}: frozen state/output checks {row["frozen_unchanged"]}/{row["representative_outputs_unchanged"]}; bridge run={row["bridge"]["run"]}.')
            if row['bridge']['run']:
                analysis += ['', '| Target | C validation | Bridge validation | C test | Bridge test |', '|---|---:|---:|---:|---:|']
                for t in TARGETS:
                    analysis.append('| '+t+' | '+' | '.join(f'{row[split][m][t]["r2"]:.6f}' for split in ('validation','test') for m in ('C','C_bridge'))+' |')
    analysis += ['', '## Interpretation boundaries', '',
        'B−A measures the result of added fitting on a frozen final release, a new attacker fitting pool, reset optimizer and validation checkpoint selection together. B retains prior900-update exposure; C does not. B/C isolate inherited weights under matched added fitting, not total lifetime training.', '',
        'D differs from B/C in output target sharing (all U,V,S), preprocessing, initialization and minibatch streams. No architecture or loss sweep was run. Catch-up cannot alone prove that moving representations caused the original miss, because stopping movement and adding training co-occur.', '',
        'Validation has been reused across exploratory pilots. The fresh test assesses these selected attacks, not an independent confirmation of the entire research hypothesis. Negative predictive R² is not negative information. Successful attacks improve measurement, not protection: encoders, heads and final releases never change.', '',
        'The broader result remains: prediction-only release passed the fixed-task pilot and no full representation did. This diagnostic establishes neither PCRL novelty nor a need for reusable representations.']
    (out/'TABLE.md').write_text('\n'.join(table)+'\n')
    (out/'ANALYSIS.md').write_text('\n'.join(analysis)+'\n')
    (out/'paired_target_metrics.json').write_text(json.dumps(paired, indent=2)+'\n')
    plot_learning_curves(out, raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    summarize(parser.parse_args().out)

"""Compact figures for the addendum. Reads the report tables only; computes no new science."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .inputs import OUT, read_json, write_json
from .maps import condition_name
from .objective import POLICIES

FAMILY_LABEL = {'original': 'original moments', 'nonlinear': 'nonlinear refinement'}
COLORS = {'original': '#4b6bab', 'nonlinear': '#c2603b', 'J': '#3f7d55', 'H': '#7a7a7a'}


def _read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tradeoff_figure(out: Path, dest: Path):
    """Residence gain over H against additional A/RAC1P recovery, the study's core tradeoff."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read_csv(out / 'PER_SEED.csv')
    main = [r for r in rows if r['split'] == 'test' and r['budget'] == '360'
            and r['scope'] == 'kernel_expanded_catchup']
    gains = defaultdict(dict)
    for r in main:
        if r['kind'] == 'utility_gain_vs_H' and r['endpoint'] == 'same_residence':
            gains[r['weight'], r['condition']].setdefault('residence', []).append(_num(r['value']))
        if r['kind'] == 'additional_recovery' and r['endpoint'] == 'A/RAC1P':
            gains[r['weight'], r['condition']].setdefault('race', []).append(_num(r['value']))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, weight in zip(axes, ('unweighted', 'person_weighted')):
        for family in ('original', 'nonlinear'):
            for rank, marker in ((16, 'o'), (8, 's')):
                xs, ys, labels = [], [], []
                for policy in sorted(POLICIES):
                    key = (weight, condition_name(family, rank, policy))
                    if key not in gains or 'race' not in gains[key]:
                        continue
                    xs.append(np.mean(gains[key]['residence']))
                    ys.append(np.mean(gains[key]['race']))
                    labels.append(policy)
                if xs:
                    ax.scatter(xs, ys, marker=marker, s=58, color=COLORS[family],
                               edgecolor='white', linewidth=.7, zorder=3,
                               label=f'{FAMILY_LABEL[family]}, r={rank}')
                    for x, y, text in zip(xs, ys, labels):
                        ax.annotate(text, (x, y), fontsize=7, xytext=(4, 3),
                                    textcoords='offset points', color='#333')
        for reference, color in (('J', COLORS['J']),):
            key = (weight, reference)
            if key in gains and 'race' in gains[key]:
                ax.scatter([np.mean(gains[key]['residence'])], [np.mean(gains[key]['race'])],
                           marker='*', s=190, color=color, zorder=4, label='frozen neural J')
        ax.axvline(.01, ls=':', lw=1, color='#999')
        ax.axhline(0, ls='-', lw=.8, color='#ccc')
        ax.set_xlabel('residence gain over H (nats)')
        ax.set_title(weight.replace('_', ' '), fontsize=10)
        ax.grid(alpha=.18, lw=.5)
    axes[0].set_ylabel('ADDITIONAL A/RAC1P recovery over H (nats)')
    axes[0].legend(fontsize=7.5, loc='best', framealpha=.9)
    fig.suptitle('2018 development tradeoff — seed means, budget 360, kernel_expanded_catchup\n'
                 'right and down is better; dotted line is the .01 residence reference',
                 fontsize=9.5)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(dest / f'tradeoff_2018.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def rotation_figure(out: Path, dest: Path):
    """The share of training-objective gain reachable by rotation alone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = read_json(out / 'DIAGNOSTICS_SUMMARY.json')['seeds']
    names, shares = [], []
    for rank in (16, 8):
        for policy in sorted(POLICIES):
            name = condition_name('nonlinear', rank, policy)
            values = [summary[s]['rotation'][name]['rotation_only_share']
                      for s in sorted(summary) if name in summary[s]['rotation']]
            values = [v for v in values if v is not None]
            if values:
                names.append(f'r={rank} {policy}')
                shares.append(values)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    positions = np.arange(len(names))
    for i, values in enumerate(shares):
        ax.scatter([positions[i]] * len(values), values, s=46, color='#c2603b',
                   edgecolor='white', linewidth=.6, zorder=3)
        ax.plot([positions[i] - .22, positions[i] + .22], [np.mean(values)] * 2,
                color='#4b4b4b', lw=1.6, zorder=4)
    ax.axhline(1.0, ls='--', lw=1, color='#888')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('share of training gain\nreachable by rotation alone')
    ax.set_title('Rotation decomposition: how much of the penalty reduction changes\n'
                 'nothing an attacker can recover (1.0 = all of it)', fontsize=9.5)
    ax.grid(alpha=.18, lw=.5, axis='y')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(dest / f'rotation_share.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def objective_vs_attack_figure(out: Path, dest: Path):
    """Surrogate test: training-objective gain against realised attack-loss change."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fits = {s: read_json(out / f'seed_{s}' / 'fit_diagnostics.json') for s in (0, 1, 2)}
    rows = _read_csv(out / 'PER_SEED.csv')
    main = [r for r in rows if r['split'] == 'test' and r['budget'] == '360'
            and r['scope'] == 'kernel_expanded_catchup' and r['weight'] == 'unweighted'
            and r['kind'] == 'additional_recovery' and r['endpoint'] == 'A/RAC1P']
    recovery = {(int(r['seed']), r['condition']): _num(r['value']) for r in main}

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for rank, marker in ((16, 'o'), (8, 's')):
        xs, ys = [], []
        for seed in (0, 1, 2):
            for policy in sorted(POLICIES):
                nl = condition_name('nonlinear', rank, policy)
                lin = condition_name('original', rank, policy)
                record = fits[seed]['conditions'].get(nl)
                if record is None or record['initial_training_objective'] is None:
                    continue
                if (seed, nl) not in recovery or (seed, lin) not in recovery:
                    continue
                xs.append(record['initial_training_objective'] - record['training_objective'])
                ys.append(recovery[seed, nl] - recovery[seed, lin])
        if xs:
            ax.scatter(xs, ys, marker=marker, s=52, color='#c2603b', edgecolor='white',
                       linewidth=.6, zorder=3, label=f'r={rank}')
    ax.axhline(0, color='#888', lw=1, ls='--')
    ax.set_xlabel('training-objective gain of the refinement (nats, surrogate)')
    ax.set_ylabel('change in ADDITIONAL A/RAC1P recovery\n(negative = less leakage)')
    ax.set_title('Surrogate versus independent attack\n'
                 'points on or above the dashed line bought no disclosure reduction', fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=.18, lw=.5)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(dest / f'surrogate_vs_attack.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def build(out: Path = OUT) -> dict:
    out = Path(out)
    dest = out / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    made, failed = [], {}
    for name, fn in (('tradeoff_2018', tradeoff_figure),
                     ('rotation_share', rotation_figure),
                     ('surrogate_vs_attack', objective_vs_attack_figure)):
        try:
            fn(out, dest)
            made.append(name)
        except Exception as exc:                                    # pragma: no cover
            failed[name] = repr(exc)
    record = {'figures': made, 'failed': failed,
              'files': sorted(p.name for p in dest.iterdir() if p.is_file())}
    write_json(out / 'FIGURES.json', record)
    return record


if __name__ == '__main__':
    print(build())

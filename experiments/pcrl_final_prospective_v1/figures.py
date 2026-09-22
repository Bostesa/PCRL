"""Task-versus-disclosure small multiples (one panel per role x weighting)."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import PANEL, PRIMARY_ROLES, WEIGHTINGS

HIGHLIGHT = {'Q': '#2a78d6', 'D17': '#eb6834', 'J': '#1baf7a'}   # reference palette slots 1-3 (all-pairs safe)
OTHER = '#8a8983'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def task_vs_disclosure(table, out_dir):
    rows = {(r['family'], r['role'], r['weighting']): r for r in table}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey='row', constrained_layout=True)
    for i, w in enumerate(WEIGHTINGS):
        for j, role in enumerate(PRIMARY_ROLES):
            ax = axes[i, j]
            ax.axvline(0, color=GRID, lw=1, zorder=0); ax.axhline(0, color=GRID, lw=1, zorder=0)
            for m in PANEL:
                if m == 'H':
                    ax.plot(0, 0, marker='s', ms=7, color=INK, zorder=3)
                    ax.annotate('H', (0, 0), xytext=(4, 4), textcoords='offset points', fontsize=8, color=INK)
                    continue
                a = rows[(m, role, w)]; t = rows[(m, 'utility:A/same_residence', w)]
                x, y = a['recovery_over_H'], t['task_minus_H']
                xe = [[x-a['recovery_over_H_ci95_unadjusted'][0]], [a['recovery_over_H_ci95_unadjusted'][1]-x]]
                ye = [[y-t['task_minus_H_ci95_unadjusted'][0]], [t['task_minus_H_ci95_unadjusted'][1]-y]]
                c = HIGHLIGHT.get(m, OTHER)
                ax.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', ms=8 if m in HIGHLIGHT else 6, color=c, ecolor=c,
                            elinewidth=1, capsize=0, alpha=1 if m in HIGHLIGHT else .8, zorder=4 if m in HIGHLIGHT else 2,
                            mec='white', mew=1.5)
                ax.annotate(m, (x, y), xytext=(5, -10 if m in ('D17', 'RR75') else 4), textcoords='offset points',
                            fontsize=8, color=INK if m in HIGHLIGHT else MUTED)
            ax.set_title(f'{role.split(":")[1]} · {w}', fontsize=10, color=INK)
            ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(GRID)
            if j == 0:
                ax.set_ylabel('Residence CE − H (nats; lower = better task)', fontsize=9, color=MUTED)
            if i == 1:
                ax.set_xlabel('Sensitive recovery over H: CE_H − CE_M (nats)', fontsize=9, color=MUTED)
    fig.suptitle('ACS 2016 final pool: task gain versus measured sensitive recovery, per role and weighting\n'
                 'mean of three anchors; bars = unadjusted marginal 95% household-bootstrap intervals (descriptive)',
                 fontsize=11, color=INK)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'task_vs_disclosure.{ext}', dpi=160)
    plt.close(fig)
    return [str(out/f'task_vs_disclosure.{e}') for e in ('png', 'pdf', 'svg')]

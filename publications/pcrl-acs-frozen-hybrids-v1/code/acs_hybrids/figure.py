"""Figure: coalition SEX recovery versus income and occupation utility for every service and seed (from DEV_REPORT_RESULTS).

x: the task's worst utility clause as a fraction of its cap (max over U0/Ucal x accuracy loss, NLL excess, summed Brier excess;
<= 1 means all six point clauses pass). y: coalition (pair) AUC with unadjusted marginal 95% intervals. H_LJ is the
preselected primary; NEURAL_RAW and TASK_REFIT are the external controls; H_JL, H_JJ and T_TREE are diagnostic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common as C

STYLE = {'H_LJ': dict(marker='*', s=260, c='#d62728', label='H_LJ (preselected primary)', zorder=6),
         'H_LL': dict(marker='o', s=70, c='#1f77b4', label='H_LL = LOCAL_TREE'),
         'H_JL': dict(marker='v', s=70, c='#9467bd', label='H_JL (diagnostic)'),
         'H_JJ': dict(marker='s', s=70, c='#8c564b', label='H_JJ = JOINT_TREE (diagnostic)'),
         'NEURAL_RAW': dict(marker='D', s=110, c='#2ca02c', label='NEURAL_RAW (eligible neural control)', zorder=5),
         'TASK_REFIT': dict(marker='P', s=110, c='#ff7f0e', label='TASK_REFIT E20 (matched control)'),
         'T_TREE': dict(marker='x', s=70, c='#7f7f7f', label='T_TREE (teacher, diagnostic)')}


def main(results, out):
    res = json.loads(Path(results).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, task in zip(axes, C.TASKS):
        for svc, st in STYLE.items():
            for s in C.SEEDS:
                v = res['per_seed'][f'S{s}']['services'][svc]
                x = max(u['point'] / u['cap'] for k, u in v['utility'].items() if k.startswith(task))
                r = v['recovery']['pair']; lo, hi = r['marginal95']
                ax.errorbar([x], [r['auc']], yerr=[[r['auc'] - lo], [hi - r['auc']]], fmt='none', ecolor=st['c'], alpha=0.5, lw=1)
                ax.scatter([x], [r['auc']], label=st['label'] if s == 0 else None, edgecolors='black' if s == 1 else 'none', linewidths=1.2,
                           **{k: val for k, val in st.items() if k != 'label'})
        ax.axvline(1.0, color='black', ls='--', lw=1)
        ax.text(1.0, ax.get_ylim()[0], '  cap', va='bottom', fontsize=9)
        ax.set_xlabel(f'{task}: worst utility clause / its cap (<= 1 passes all six point clauses)')
        ax.set_title(f'Coalition recovery vs {task} utility')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('coalition (pair) SEX-recovery AUC, DEV_REPORT (lower = reveals less)')
    axes[1].legend(fontsize=8, loc='upper right')
    fig.suptitle('ACS frozen hybrids (exploratory development follow-up; already-exposed people). Filled: seed 0; black edge: seed 1. Bars: unadjusted 95%.', fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); main(a.results, a.out)

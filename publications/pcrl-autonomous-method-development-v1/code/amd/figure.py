"""UTILITY_RECOVERY.png: two separate measurements, never one frontier.

Left: MODEL_SELECT development proxies (adaptive; household halves) for every fitted configuration and seed: linked-pair proxy
against the worst normalized utility value (max over the 12 clauses and both guards, each divided by its cap). Right: fresh
DEV_REPORT readers (ATTACK_FIT-fitted, ATTACK_VALIDATION-selected) for the final bank: linked-pair AUC with marginal 95%
intervals, filled when the release meets the 12 DEV utility clauses at the point on that seed. Families use fixed hues and
distinct marker shapes (secondary encoding); the tables COMPARISON_TABLE.csv and RECOVERY_TABLE.csv are the table view.
"""
from __future__ import annotations
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common as C

INK, MUTED, GRID = '#1f1f1f', '#5f5f5f', '#d9d9d9'
FAM = {'A': ('#0072B2', 'o', 'A: constrained leaf correction'), 'B': ('#E69F00', 's', 'B: coarse / coupled income code'),
       'C': ('#009E73', '^', 'C: SEX-mixture marginalization'), 'REF': ('#5f5f5f', 'D', 'frozen references')}


def main():
    rows = list(csv.DictReader(open(C.RESULTS / 'COMPARISON_TABLE.csv')))
    d = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios': [1.0, 1.15]})
    for ax in (a1, a2):
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    seen = set()
    for r in rows:
        if r['family'] == 'REF' and r['config_id'] not in ('NEURAL_RAW', 'H_LJ', 'H_JJ', 'T_TREE'):
            continue
        worst = max(float(r['ms_worst_normalized']), float(r['ms_guard_men']) / C.GUARD, float(r['ms_guard_women']) / C.GUARD)
        col, mk, lab = FAM[r['family']]
        a1.scatter(min(worst, 2.2), float(r['proxy_pair']), s=46, marker=mk, facecolor=col if r['seed'] == '0' else 'white', edgecolor=col,
                   linewidth=1.6, label=lab if r['family'] not in seen else None, zorder=3)
        seen.add(r['family'])
        if r['family'] == 'REF' and r['seed'] == '0':
            a1.annotate(r['config_id'], (min(worst, 2.2), float(r['proxy_pair'])), textcoords='offset points', xytext=(6, -3), fontsize=8, color=MUTED)
        if r['config_id'] == 'B3CTL_UNCOUPLED_MIX50_C6A' and r['seed'] == '0':
            a1.annotate('finalist', (worst, float(r['proxy_pair'])), textcoords='offset points', xytext=(-48, -14), fontsize=9, color=INK,
                        arrowprops={'arrowstyle': '-', 'color': MUTED, 'linewidth': 0.8})
    a1.axvline(1.0, color=MUTED, linestyle='--', linewidth=1.0)
    a1.text(1.02, 0.567, 'cap', color=MUTED, fontsize=8)
    a1.set_xlabel('worst utility value / its cap (MODEL_SELECT; >1 fails; clipped at 2.2)', color=INK)
    a1.set_ylabel('linked-pair SEX recovery proxy (AUC)', color=INK)
    a1.set_title('MODEL_SELECT development proxies (adaptive)\nfilled = seed 0, open = seed 1', fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8, loc='upper right')
    order = ['FINALIST', 'H_JJ', 'B2TWIN', 'PUB_IBSI', 'NEURAL_RAW', 'H_LJ', 'TASKONLY', 'T_TREE']
    for s, off in ((0, -0.13), (1, 0.13)):
        ps = d['per_seed'][f'S{s}']
        for i, role in enumerate(order):
            v = ps['services'][ps['service_of_role'][role]]
            r = v['recovery']['pair']; feas = v['utility_point_feasible_12_this_seed']
            col = '#E69F00' if role == 'FINALIST' else '#5f5f5f'
            a2.errorbar(i + off, r['auc'], yerr=[[r['auc'] - r['marginal95'][0]], [r['marginal95'][1] - r['auc']]], fmt='o' if s == 0 else 's',
                        color=col, mfc=col if feas else 'white', mec=col, ms=6, elinewidth=1.4, capsize=0, zorder=3)
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels(['finalist', 'H_JJ\n(infeasible)', 'LOCAL-code\ntwin', 'PUB_IBSI\n(adapted)', 'NEURAL_RAW', 'H_LJ', 'task-only\ncontrol', 'T_TREE'],
                       fontsize=8, color=INK)
    a2.set_ylabel('linked-pair SEX recovery, fresh readers (AUC)', color=INK)
    a2.set_title('DEV_REPORT final bank (one logged evaluation)\ncircle = seed 0, square = seed 1; open = fails a DEV utility point clause', fontsize=10, color=INK)
    fig.suptitle('Florida ACS 2019 development roles: recovery of the recorded SEX column from the released income and occupation outputs',
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(C.RESULTS / 'UTILITY_RECOVERY.png', dpi=150)
    print('UTILITY_RECOVERY.png written')


if __name__ == '__main__':
    main()

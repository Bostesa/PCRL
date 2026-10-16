"""Public tables and figures from DEV_REPORT_RESULTS.json (aggregate numbers only; no person-level data)."""
from __future__ import annotations
import csv
import json
from pathlib import Path

GROUP = {'LOCAL_TREE': 'primary', 'TASK_REFIT': 'matched control', 'JOINT_TREE': 'diagnostic', 'T_TREE': 'teacher', 'U0': 'reference'}


def group(s):
    if s in GROUP:
        return GROUP[s]
    return 'neural control' if s.startswith('NEURAL_RAW') else 'published tree baseline' if s.startswith('FAGTB') else 'official FARE' if s.startswith('FARE') else 'other'


COLORS = {'primary': 'crimson', 'matched control': 'forestgreen', 'diagnostic': 'darkorange', 'teacher': 'darkgreen', 'reference': 'black',
          'neural control': 'royalblue', 'published tree baseline': 'purple', 'official FARE': 'grey', 'other': 'brown'}


def main(results_dir):
    rd = Path(results_dir); r = json.loads((rd / 'DEV_REPORT_RESULTS.json').read_text())
    # ---------- tables
    with open(rd / 'PRIMARY_ENDPOINTS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['endpoint', 'kind', 'margin', 'point', 'se', 'iut_lower', 'iut_upper', 'iut_pass', 'sim95_lower', 'sim95_upper', 'sim_status',
                    'point_pass', 'sensitivity_pass', 'bootstrap_skewness'])
        for k, v in r['endpoints'].items():
            w.writerow([k, v['kind'], v['margin'], v['point'], v['se'], v['iut_lower'], v['iut_upper'], v['iut_pass'], v['sim95_lower'], v['sim95_upper'],
                        v['sim_status'], v['point_pass'], v['sensitivity_pass'], v['bootstrap_skewness']])
    rows = []
    for seed, sr in r['per_seed'].items():
        for svc, s in sr['services'].items():
            row = {'service': svc, 'group': group(svc), 'seed': seed}
            for view in ('income', 'occupation', 'pair'):
                row[f'{view}_auc'] = s['recovery'][view]['auc']; row[f'{view}_reader'] = s['recovery'][view]['reader']
                row[f'{view}_pwgtp_auc_point'] = s['recovery'][view]['pwgtp_weighted_auc_point']
            for t in ('income', 'occupation'):
                u = s['task_summary'][t]['unweighted']
                for m in ('accuracy', 'nll', 'brier', 'balanced_accuracy'):
                    row[f'{t}_{m}'] = u[m]
            fails = [k for k, v in s['utility'].items() if not v['point_pass']]
            row['utility_point_fail_count_of_12'] = len(fails); row['utility_point_fails'] = ';'.join(fails)
            for g in ('sex0', 'sex1'):
                row[f'occupation_brier_excess_vs_U0_{g}'] = s['subgroup'][g]['point']
            rows.append(row)
    with open(rd / 'SERVICE_TABLE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(rd / 'SUBGROUP_TABLE.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['service', 'seed', 'sex_group', 'occupation_brier_excess_vs_U0', 'se', 'iut_upper', 'sim95_lower', 'sim95_upper', 'sim_status',
                    'group_nll_excess_vs_U0', 'group_accuracy', 'group_accuracy_loss_vs_U0', 'group_households', 'in_primary_family'])
        for seed, sr in r['per_seed'].items():
            for svc, s in sr['services'].items():
                for g in ('sex0', 'sex1'):
                    v = s['subgroup'][g]
                    w.writerow([svc, seed, g, v['point'], v['se'], v['iut_upper'], v['sim95_lower'], v['sim95_upper'], v['sim_status'],
                                v['group_nll_excess_vs_U0'], v['group_accuracy'], v['group_accuracy_loss_vs_U0'], v['group_households'], v['in_primary_family']])
    # ---------- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = rd / 'figures'; out.mkdir(exist_ok=True)
    svcs = sorted({x['service'] for x in rows}); mean = lambda s, k: sum(x[k] for x in rows if x['service'] == s) / 2
    u0 = {k: mean('U0', k) for k in ('income_nll', 'occupation_nll', 'income_brier', 'occupation_brier', 'income_accuracy', 'occupation_accuracy')}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, cap in ((axes[0, 0], 'occupation_brier', 0.005), (axes[0, 1], 'income_brier', 0.005), (axes[1, 0], 'occupation_nll', 0.010), (axes[1, 1], 'income_nll', 0.010)):
        for s in svcs:
            ax.scatter(mean(s, 'pair_auc'), mean(s, metric), color=COLORS[group(s)], s=70 if s == 'LOCAL_TREE' else 30, zorder=3)
            ax.annotate(s, (mean(s, 'pair_auc'), mean(s, metric)), fontsize=7, xytext=(3, 2), textcoords='offset points')
        ax.axhline(u0[metric], color='black', lw=0.8, label='U0'); ax.axhline(u0[metric] + cap, color='red', lw=0.8, ls='--', label=f'U0 + cap ({cap})')
        ax.set_xlabel('coalition SEX AUC (ATTACK_VALIDATION winner, DEV_REPORT, seed mean)'); ax.set_ylabel(metric.replace('_', ' ') + ' (DEV_REPORT, seed mean)')
        ax.legend(fontsize=7)
    fig.suptitle('Florida 2019 ACS: task utility vs coalition recovery (lower AUC = less SEX recovery)')
    fig.tight_layout(); fig.savefig(out / 'UTILITY_VS_COALITION_RECOVERY.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    for s in svcs:
        xs = [x['income_auc'] for x in rows if x['service'] == s]; ys = [x['occupation_auc'] for x in rows if x['service'] == s]
        ax.scatter(xs, ys, color=COLORS[group(s)], s=25, alpha=0.6); ax.scatter(sum(xs) / 2, sum(ys) / 2, color=COLORS[group(s)], s=60, marker='x')
        ax.annotate(s, (sum(xs) / 2, sum(ys) / 2), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel('income recipient SEX AUC (DEV_REPORT)'); ax.set_ylabel('occupation recipient SEX AUC (DEV_REPORT)')
    ax.set_title('Individual recipient recovery (dots = seeds, x = mean)')
    fig.tight_layout(); fig.savefig(out / 'RECIPIENT_RECOVERY.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [s for s in svcs if s != 'U0']
    for i, s in enumerate(order):
        for j, g in enumerate(('sex0', 'sex1')):
            pts = [x[f'occupation_brier_excess_vs_U0_{g}'] for x in rows if x['service'] == s]
            ax.scatter([i + (j - 0.5) * 0.25] * len(pts), pts, color=('teal' if g == 'sex0' else 'magenta'), s=25, label=(f'{g} (0 = male, 1 = female)' if i == 0 else None))
    ax.axhline(0.005, color='red', ls='--', lw=0.8, label='subgroup guard +0.005'); ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('occupation summed-Brier excess vs U0 within SEX group'); ax.legend(fontsize=7)
    ax.set_title('Subgroup usefulness (points = seeds; DEV_REPORT)')
    fig.tight_layout(); fig.savefig(out / 'SUBGROUP_OCCUPATION_BRIER.png', dpi=130); plt.close(fig)
    return [str(p) for p in sorted(out.glob('*.png'))]


if __name__ == '__main__':
    import sys
    print(main(sys.argv[1]))

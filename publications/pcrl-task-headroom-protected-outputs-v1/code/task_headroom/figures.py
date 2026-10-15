"""Figures from the frozen INNER report (aggregate numbers only; no person-level data)."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main(report_path, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads(Path(report_path).read_text()); out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rows = r['service_table']; primary = r['primary_service']
    svc = sorted({x['service'] for x in rows if 'pair_auc' in x})
    mean = lambda s, k: sum(x[k] for x in rows if x['service'] == s) / 2
    refs = {n: {k: mean(n, k) for k in ('income_nll', 'occupation_nll', 'income_accuracy', 'occupation_accuracy')} for n in ('U0', 'Ucal')}
    group = lambda s: ('primary' if s == primary else 'protected tree' if s in ('J_TREE', 'L_TREE', 'L_STRONG', 'S12_TREE', 'S21_TREE', 'DIAG_PRIV')
                       else 'tree task-only' if s in ('T_TREE', 'T_REFIT') else 'neural' if s in ('N_EQ', 'RAW_PROJ') else 'historical')
    colors = {'primary': 'crimson', 'protected tree': 'darkorange', 'tree task-only': 'forestgreen', 'neural': 'royalblue', 'historical': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, cap in ((axes[0, 0], 'occupation_nll', 0.010), (axes[0, 1], 'income_nll', 0.010), (axes[1, 0], 'occupation_accuracy', -0.010), (axes[1, 1], 'income_accuracy', -0.010)):
        for s in svc:
            ax.scatter(mean(s, 'pair_auc'), mean(s, metric), color=colors[group(s)], s=60 if s == primary else 30, zorder=3)
            ax.annotate(s, (mean(s, 'pair_auc'), mean(s, metric)), fontsize=7, xytext=(3, 2), textcoords='offset points')
        worst = max(refs['U0'][metric], refs['Ucal'][metric]) if cap > 0 else min(refs['U0'][metric], refs['Ucal'][metric])
        ax.axhline(refs['U0'][metric], color='black', lw=0.8, ls='-', label='U0')
        ax.axhline(refs['Ucal'][metric], color='black', lw=0.8, ls=':', label='Ucal')
        ax.axhline(min(refs['U0'][metric], refs['Ucal'][metric]) + cap if cap > 0 else max(refs['U0'][metric], refs['Ucal'][metric]) + cap,
                   color='red', lw=0.8, ls='--', label='strictest cap (both refs)')
        ax.set_xlabel('coalition SEX AUC (ATT-validation winner, INNER, seed mean)'); ax.set_ylabel(metric.replace('_', ' ') + ' (INNER, seed mean)')
        ax.legend(fontsize=7)
    fig.suptitle('Task utility vs coalition recovery, all attacked services (lower AUC = less recovery)')
    fig.tight_layout(); fig.savefig(out / 'UTILITY_VS_COALITION_RECOVERY.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    for s in svc:
        xs = [x['income_auc'] for x in rows if x['service'] == s]; ys = [x['occupation_auc'] for x in rows if x['service'] == s]
        ax.scatter(xs, ys, color=colors[group(s)], s=25, alpha=0.6); ax.scatter(sum(xs) / 2, sum(ys) / 2, color=colors[group(s)], s=60, marker='x')
        ax.annotate(s, (sum(xs) / 2, sum(ys) / 2), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel('income recipient SEX AUC (INNER)'); ax.set_ylabel('occupation recipient SEX AUC (INNER)')
    ax.set_title('Individual recipient recovery (dots = seeds, x = mean)')
    fig.tight_layout(); fig.savefig(out / 'RECIPIENT_RECOVERY.png', dpi=130); plt.close(fig)
    return [str(out / 'UTILITY_VS_COALITION_RECOVERY.png'), str(out / 'RECIPIENT_RECOVERY.png')]


if __name__ == '__main__':
    print(main(sys.argv[1], sys.argv[2]))

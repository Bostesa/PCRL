"""Public aggregate figures from committed INNER_REPORT.json (attacked services only); no person-level data."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['RAW_PROJ', 'O_PREVIOUS', 'T_BOTH', 'L_BOTH', 'J_BOTH', 'S12', 'S21', 'I_ONLY', 'HYB_JI_RO', 'HYB_RI_JO']


def main(res):
    res = Path(res); d = json.loads((res / 'INNER_REPORT.json').read_text())
    desc = {(r['service'], r['seed'], r['view']): r for r in d['descriptive_per_seed_auc']}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, view, title in zip(axes, ('r1', 'r2', 'pair'), ('income wire (21)', 'occupation wire (29)', 'aligned pair (50)')):
        for k, seed in enumerate((0, 1)):
            xs = [i + (k - 0.5) * 0.35 for i in range(len(ORDER))]
            ys = [desc[(s, seed, view)]['auc'] for s in ORDER]
            lo = [desc[(s, seed, view)]['auc'] - desc[(s, seed, view)]['lower95'] for s in ORDER]
            ax.errorbar(xs, ys, yerr=lo, fmt='o', ms=4, capsize=2, label=f'seed {seed}')
        base = sum(desc[('RAW_PROJ', s, view)]['auc'] for s in (0, 1)) / 2
        ax.axhline(base, color='grey', lw=0.8, ls='--'); ax.axhline(base - 0.02, color='red', lw=0.8, ls=':')
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=60, ha='right', fontsize=8)
        ax.set_title(f'Fresh-reader SEX recovery: {title}'); ax.set_ylabel('complete-envelope AUC (INNER)')
    axes[0].legend(fontsize=8)
    fig.text(0.5, -0.02, 'All ten attacked services per seed; unadjusted 95% bootstrap bands. Grey: RAW_PROJ mean; red dotted: RAW_PROJ - 0.020 (required coalition benefit).', ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(res / 'figures/RECOVERY_BY_SERVICE.png', dpi=150, bbox_inches='tight')
    ep = {r['endpoint_id']: r for r in d['primary']['endpoints']}
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [r for r in d['primary']['endpoints'] if r['kind'] == 'privacy' and r['view'] == 'pair']
    for i, r in enumerate(rows):
        ax.errorbar([r['point']], [i], xerr=[[r['point'] - r['lower']], [r['upper'] - r['point']]], fmt='o', capsize=3)
    ax.axvline(0.020, color='red', ls=':', label='required lower bound > 0.020'); ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"J_BOTH vs {r['control']}" for r in rows])
    ax.set_xlabel('coalition (pair) recovery benefit, AUC (K=39 simultaneous)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(res / 'figures/PRIMARY_PAIR_BENEFIT.png', dpi=150)
    st = {(r['service'], r['seed']): r for r in d['service_table']}
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in ORDER:
        for seed, mk in ((0, 'o'), (1, 's')):
            r = st[(s, seed)]
            ax.scatter(r['income__U0__nll_excess'] + r['occupation__U0__nll_excess'], r['pair__complete_auc'], marker=mk)
            ax.annotate(f'{s}/{seed}', (r['income__U0__nll_excess'] + r['occupation__U0__nll_excess'], r['pair__complete_auc']), fontsize=6)
    ax.set_xlabel('INNER NLL excess vs U0, income + occupation (point)'); ax.set_ylabel('pair complete-envelope AUC (INNER)')
    ax.set_title('Attacked services: utility cost vs coalition recovery')
    fig.tight_layout(); fig.savefig(res / 'figures/UTILITY_VS_PAIR_RECOVERY.png', dpi=150)


if __name__ == '__main__':
    main(sys.argv[1])

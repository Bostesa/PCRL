"""Figures from committed public aggregates only (no person arrays)."""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(sys.argv[1])
rep = json.loads((R / 'INNER_REPORT.json').read_text())
fit2 = json.loads((R / 'FIT2_UTILITY.json').read_text())
sel = json.loads((R / 'SELECTION_LOCK.json').read_text())
ARMS = ['J_TRUST', 'J_FREE', 'L_TRUST', 'L_FREE', 'TASK_TRUST']
COL = {'J_TRUST': 'tab:red', 'J_FREE': 'tab:orange', 'L_TRUST': 'tab:blue', 'L_FREE': 'tab:cyan', 'TASK_TRUST': 'tab:green',
       'RAW_PROJ': 'black', 'RAW_HYBRID': 'gray', 'PRIOR_SEPARATE_GUARD_E5': 'tab:purple', 'PRIOR_GLOBAL_UPDATE_GUARD_E5': 'tab:brown',
       'PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ': 'tab:pink'}
fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
for ax, seed in zip(axes, (0, 1)):
    for row in rep['service_table']:
        if row['seed'] != seed:
            continue
        s = row['service']; elig = all(v for k, v in row.items() if k.endswith('point_pass'))
        status = sel['attack_roster']['status'][s]
        x = row['occupation__U0__nll_excess']; y = row['pair__complete_auc']
        ax.scatter(x, y, s=70, color=COL[s], marker='o' if elig else 'x', label=f'{s} ({status.split("_E")[0][:28]})')
        ax.annotate(s.replace('PRIOR_', 'P_').replace('_UPDATE_GUARD', 'G').replace('_GUARD', 'G'), (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axvline(0.010, color='k', ls='--', lw=0.8); ax.text(0.0102, ax.get_ylim()[0] if False else 0.823, 'NLL cap 0.010', fontsize=7, rotation=90)
    ax.set_title(f'Source seed {seed}: INNER utility vs fresh coalition recovery')
    ax.set_xlabel('occupation NLL excess vs U0 on INNER (point)'); ax.grid(alpha=.3)
axes[0].set_ylabel('pair complete-view AUC (18 fresh readers + ignore aliases)')
h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, loc='lower center', ncol=3)
fig.suptitle('o = all 24 original INNER point clauses pass; x = fails a point clause. Diagnostic points are not eligible services; recovery shown only for attacked checkpoints.', fontsize=8)
fig.tight_layout(rect=(0, 0.16, 1, 0.97)); fig.savefig(R / 'figures/UTILITY_RECOVERY_PAIRED.png', dpi=140); plt.close(fig)
# Utility-only trajectories (FIT2, every saved epoch).
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for j, seed in enumerate((0, 1)):
    for i, metric in enumerate(('nll_excess', 'brier_excess')):
        ax = axes[i, j]
        for arm in ARMS:
            ep = [0, 1, 3, 5, 10]
            vals = [next(c['value'] for c in fit2['table'][f'{seed}|{arm}|{e}']['clauses'] if c['task'] == 'occupation' and c['reference'] == 'U0' and c['metric'] == metric) for e in ep]
            ax.plot(ep, vals, marker='o', color=COL[arm], label=arm)
        cap = 0.010 if metric == 'nll_excess' else 0.005
        ax.axhline(cap, color='k', ls='--', lw=.8); ax.set_title(f'seed {seed}: FIT2 occupation {metric} vs U0'); ax.grid(alpha=.3)
axes[1, 0].set_xlabel('epoch'); axes[1, 1].set_xlabel('epoch'); axes[0, 0].legend(fontsize=7)
fig.suptitle('Utility-only trajectories on FIT2 (exposed development selection role); no recovery measured for unattacked epochs', fontsize=9)
fig.tight_layout(); fig.savefig(R / 'figures/FIT2_UTILITY_TRAJECTORIES.png', dpi=140); plt.close(fig)
# Fitting-pool vs FIT2 loss.
traj = list(csv.DictReader((R / 'TRAINING_TRAJECTORIES.csv').open()))
start = json.loads((R / 'TRAINING_COMPLETE_LOCK.json').read_text())['start_pool']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, seed in zip(axes, (0, 1)):
    for arm in ARMS:
        ep = [1, 3, 5, 10]
        fp = [float(next(r['nll'] for r in traj if r['seed'] == str(seed) and r['arm'] == arm and r['pool'] == 'TASK_POOL' and r['epoch'] == str(e))) - start[str(seed)]['nll'] for e in ep]
        f2 = [fit2['table'][f'{seed}|{arm}|{e}']['metrics']['occupation']['nll'] - fit2['table'][f'{seed}|{arm}|0']['metrics']['occupation']['nll'] for e in ep]
        ax.plot(ep, fp, ls='-', marker='o', color=COL[arm], label=f'{arm} fitting pool')
        ax.plot(ep, f2, ls=':', marker='s', color=COL[arm], label=f'{arm} FIT2')
    ax.axhline(0.004, color='k', ls='--', lw=.8); ax.set_title(f'seed {seed}: occupation NLL increase vs start'); ax.set_xlabel('epoch'); ax.grid(alpha=.3)
axes[0].legend(fontsize=6, ncol=2)
fig.suptitle('Fitting-pool (solid) vs FIT2 (dotted) NLL increase; dashed line = internal +0.004 fitting budget', fontsize=9)
fig.tight_layout(); fig.savefig(R / 'figures/FITTING_VS_FIT2_LOSS.png', dpi=140); plt.close(fig)
print('figures written')

"""Figures of the PCRL correction-transfer campaign (label-free: they read saved ladder, selection and reader-grid records).
Run with the torch interpreter (matplotlib 3.10.8).

    <TORCH python> -m cts.figures cost --partitions D0,D1 --direction REFIT_PAIR [--stem COST_TRANSFER_D_REFIT_PAIR]
    <TORCH python> -m cts.figures frontier --bank-id ID --partitions R0,R1 [--stem FRONTIER_R]

Every figure is written as SVG and PNG with a CSV table view beside it (results/<campaign>/figures/). Categorical hues in fixed order
(validated light palette: blue, orange, aqua, yellow on surface #fcfcfb; contrast below 3:1 for aqua and yellow, so every series is
also identified by marker shape and a direct label).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

import numpy as np

from . import common as C
from . import rules as RL
from . import trackf as TF

SURFACE = '#fcfcfb'
INK = '#1f1f1e'
MUTED = '#6b6a63'
GRID = '#e4e3dc'
HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
OUT = C.RESULTS / 'figures'


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'svg.fonttype': 'none'})
    return plt


def _save(fig, stem, rows, header):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.png', dpi=160, bbox_inches='tight')
    with open(OUT / f'{stem}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _end_labels(ax, ends, min_gap=0.075):
    """Direct labels at the line ends, nudged apart vertically (at least min_gap of the y range) so that none overlap."""
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    placed = []
    for x, y, lab in sorted(ends, key=lambda t: t[1]):
        yy = max(y, placed[-1] + min_gap * (hi - lo)) if placed else y
        placed.append(yy)
        ax.text(x + 0.03 * (x1 - x0), yy, lab, color=INK, fontsize=7, va='center', clip_on=False)


def cost(partitions, did, stem):
    plt = _plt()
    stats = [('men_brier', "Men's summed-Brier increase"), ('women_brier', "Women's summed-Brier increase"), ('nll', 'NLL increase (nats)')]
    ests = [('OOF', 'OOF (source estimator)'), ('GUARD', 'GUARD'), ('CHECK', 'CHECK'), ('MS', 'MODEL_SELECT')]
    fig, axes = plt.subplots(len(partitions), len(stats), figsize=(3.3 * len(stats), 2.6 * len(partitions)), squeeze=False)
    rows = []
    for i, P in enumerate(partitions):
        f = TF.pdir(P) / ('LADDER_CHECK' if C.PARTITIONS[P]['kind'] == 'repair' else 'LADDER') / f'{did}.json'
        if not f.is_file():
            f = TF.pdir(P) / 'LADDER' / f'{did}.json'
        lad = json.loads(f.read_text())['estimators']
        sel = json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())
        ss = RL.stat_slack(sel['baseline'])
        for j, (k, title) in enumerate(stats):
            ax = axes[i][j]
            ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
            ends = []
            for m, (e, lab) in enumerate(ests):
                if e not in lad:
                    continue
                xs = [float(s) for s in lad[e]]
                ys = [lad[e][s]['stats'][k] for s in lad[e]]
                es = [lad[e][s]['se'][k] for s in lad[e]]
                ax.errorbar(xs, ys, yerr=es, color=HUES[m], marker=MARKERS[m], markersize=4.5, linewidth=1.6, elinewidth=0.7, capsize=0,
                            label=lab)
                ends.append((xs[-1], ys[-1], e))
                rows += [[P, did, k, e, x, y, s] for x, y, s in zip(xs, ys, es)]
            ax.axhline(ss[k], color=MUTED, linewidth=0.9, linestyle='--')
            ax.axhline(0, color=MUTED, linewidth=0.6)
            _end_labels(ax, ends)
            ax.set_title(f'{P}: {title}\n(dashed: reserve slack {ss[k]:+.4f})', color=INK, fontsize=8.5)
            ax.set_xlabel('correction strength s')
    axes[0][0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle(f'Incremental occupation costs of {did} against the uncorrected witness (household-clustered SE bars)', color=INK, fontsize=9)
    fig.tight_layout()
    _save(fig, stem, rows, ['partition', 'direction', 'statistic', 'estimator', 'strength', 'cost', 'se'])
    print(json.dumps({'figure': stem, 'rows': len(rows)}))


def _grid_auc(bank_id, seed):
    g = json.loads((C.RUNS / 'GRID' / bank_id / f'S{seed}' / 'GRID.json').read_text())
    out = {}
    for cell, cres in g['cells'].items():
        if '@' in cell:
            continue
        for n, ent in cres.items():
            out.setdefault(n, []).append(ent['pair']['score_auc'])
    return {n: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for n, v in out.items()}


def _ms_cost(P, name):
    """Realized MODEL_SELECT incremental costs (vs the uncorrected witness) of a ladder point or compression code, and the registered
    utilization u = max over statistics of cost / reserve slack (cts.rules.utilization: infinite where a slack is <= 0 and the cost
    exceeds it)."""
    if name == 'W':
        st = {k: 0.0 for k in RL.STATS}
    elif name.startswith('COMP:'):
        cr = json.loads((TF.pdir(P) / 'COMP' / 'COMP_REALIZED.json').read_text())['codes']
        did, rk = name[5:].split('@')
        st = cr[f'COMP__{did}__{rk.replace("|", "_")}']['MS']['stats']
    else:
        did, spec = name.split('@')
        f = TF.pdir(P) / 'LADDER_CHECK' / f'{did}.json'
        if not f.is_file():
            f = TF.pdir(P) / 'LADDER' / f'{did}.json'
        lad = json.loads(f.read_text())['estimators']['MS']
        key = spec[1:]
        if key not in lad:
            return None, None
        st = lad[key]['stats']
    base = json.loads((TF.pdir(P) / 'SELECT' / 'REFIT_PAIR.json').read_text())['baseline']
    return st, RL.utilization(st, RL.stat_slack(base))


def frontier(bank_id, partitions, stem):
    """Pair SCORE AUC against the realized MODEL_SELECT men's summed-Brier increase (the binding occupation-side cost of every
    partition), with the men's reserve slack marked; the registered utilization u is in the table view."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(partitions), figsize=(4.6 * len(partitions), 3.6), squeeze=False)
    rows = []
    for i, P in enumerate(partitions):
        s = TF.seed_of(P)
        auc = _grid_auc(bank_id, s)
        comp_bank = 'R_COMP0' if (C.RUNS / 'GRID' / 'R_COMP0' / f'S{s}' / 'GRID.json').is_file() and bank_id == 'R_FRONTIER' else None
        if comp_bank:
            auc.update({n: v for n, v in _grid_auc(comp_bank, s).items() if n.startswith('COMP:')})
        slack_men = RL.stat_slack(json.loads((TF.pdir(P) / 'SELECT' / 'REFIT_PAIR.json').read_text())['baseline'])['men_brier']
        ax = axes[0][i]
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for m, arm in enumerate(('PAIR', 'LOC', 'CONT')):
            pts = sorted((float(n.split('@s')[1].split('|')[0]), n) for n in auc if n.startswith(f'REFIT_{arm}@s'))
            xy = []
            for _, n in pts:
                st, u = _ms_cost(P, n)
                if st is not None:
                    xy.append((1e3 * st['men_brier'], auc[n][0], n, u))
            if xy:
                ax.plot([p[0] for p in xy], [p[1] for p in xy], color=HUES[m], marker=MARKERS[m], markersize=4.5, linewidth=1.6, label=arm)
                ex, ey = (xy[0][0], xy[0][1]) if arm == 'CONT' else (xy[-1][0], xy[-1][1])
                ax.annotate(arm, (ex, ey), xytext=(-6, 7) if arm == 'CONT' else (4, 0), textcoords='offset points', fontsize=7, color=INK,
                            va='center', ha='right' if arm == 'CONT' else 'left')
                rows += [[P, n, x, u, a] for x, a, n, u in xy]
        comp = []
        for n in auc:
            if n.startswith('COMP:'):
                st, u = _ms_cost(P, n)
                comp.append((n, 1e3 * st['men_brier'], auc[n][0], u))
        if comp:
            ax.scatter([c[1] for c in comp], [c[2] for c in comp], color=HUES[3], marker=MARKERS[3], s=22, label='COMP (R0 only)', zorder=3)
            rows += [[P, n, x, u, a] for n, x, a, u in comp]
        if 'W' in auc:
            ax.scatter([0.0], [auc['W'][0]], color=INK, marker='*', s=60, label='witness', zorder=4)
            rows.append([P, 'W', 0.0, _ms_cost(P, 'W')[1], auc['W'][0]])
        for r in ('SUCC', 'H_LJ', 'NEURAL_RAW', 'RETIRED_B3CTL'):
            if r in auc:
                ax.axhline(auc[r][0], color=MUTED, linewidth=0.8, linestyle=':')
                rows.append([P, r, None, None, auc[r][0]])
        xmin, xmax = ax.get_xlim()
        for j, r in enumerate(sorted((r for r in ('SUCC', 'H_LJ', 'NEURAL_RAW', 'RETIRED_B3CTL') if r in auc), key=lambda r: auc[r][0])):
            right = j % 2 == 0                          # alternate sides so that close reference lines keep readable labels
            ax.annotate(r, (xmax if right else xmin, auc[r][0]), xytext=(-2 if right else 2, 2), textcoords='offset points',
                        fontsize=6.5, color=MUTED, ha='right' if right else 'left')
        ax.axvline(1e3 * slack_men, color=MUTED, linewidth=0.9, linestyle='--')
        ax.annotate(f"men's reserve slack {1e3 * slack_men:+.2f}", (1e3 * slack_men, ax.get_ylim()[0]), xytext=(3, 3),
                    textcoords='offset points', fontsize=6.5, color=MUTED, rotation=90, va='bottom')
        ax.set_title(f'{P} (seed {s}): linked-pair SCORE AUC vs realized men\'s cost', fontsize=8.5, color=INK)
        ax.set_xlabel("realized MODEL_SELECT men's summed-Brier increase (x 1e-3)")
        ax.set_ylabel('mean pair SCORE AUC, 4 cells\n(lower = less recovery)')
    axes[0][0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.tight_layout()
    _save(fig, stem, rows, ['partition', 'release', 'men_brier_increase_x1e3', 'registered_utilization_u', 'mean_pair_score_auc'])
    print(json.dumps({'figure': stem, 'rows': len(rows)}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('cost'); x.add_argument('--partitions', required=True); x.add_argument('--direction', required=True)
    x.add_argument('--stem', default=None)
    y = sub.add_parser('frontier'); y.add_argument('--bank-id', required=True); y.add_argument('--partitions', default='R0,R1')
    y.add_argument('--stem', default=None)
    a = ap.parse_args(argv)
    parts = a.partitions.split(',')
    if a.cmd == 'cost':
        cost(parts, a.direction, a.stem or f'COST_TRANSFER_{"_".join(parts)}_{a.direction}')
    else:
        frontier(a.bank_id, parts, a.stem or f'FRONTIER_{a.bank_id}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

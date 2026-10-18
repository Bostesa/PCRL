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
            for m, (e, lab) in enumerate(ests):
                if e not in lad:
                    continue
                xs = [float(s) for s in lad[e]]
                ys = [lad[e][s]['stats'][k] for s in lad[e]]
                es = [lad[e][s]['se'][k] for s in lad[e]]
                ax.errorbar(xs, ys, yerr=es, color=HUES[m], marker=MARKERS[m], markersize=4.5, linewidth=1.6, elinewidth=0.7, capsize=0,
                            label=lab)
                ax.annotate(e, (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points', color=INK, fontsize=7, va='center')
                rows += [[P, did, k, e, x, y, s] for x, y, s in zip(xs, ys, es)]
            if ss[k] > 0:
                ax.axhline(ss[k], color=MUTED, linewidth=0.9, linestyle='--')
                ax.annotate('reserve slack', (0.0, ss[k]), xytext=(2, 3), textcoords='offset points', color=MUTED, fontsize=7)
            ax.axhline(0, color=MUTED, linewidth=0.6)
            ax.set_title(f'{P}: {title}', color=INK, fontsize=8.5)
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


def _util(P, name):
    """Realized MODEL_SELECT utilization u = max over the 8 conditions of cost / reserve slack (slack > 0 only), for a ladder point
    or rule selection of a REFIT direction (post-freeze ladder) or a compression code."""
    from . import releases as REL
    sel = json.loads((TF.pdir(P) / 'SELECT' / 'REFIT_PAIR.json').read_text())
    base = sel['baseline']
    if name == 'W':
        return 0.0
    if name.startswith('COMP:'):
        cr = json.loads((TF.pdir(P) / 'COMP' / 'COMP_REALIZED.json').read_text())['codes']
        did, rk = name[5:].split('@')
        st = cr[f'COMP__{did}__{rk.replace("|", "_")}']['MS']['stats']
    else:
        did, spec = name.split('@')
        s = float(spec[1:]) if spec.startswith('s') else REL.strength(P, did, spec)
        f = TF.pdir(P) / 'LADDER_CHECK' / f'{did}.json'
        if not f.is_file():
            f = TF.pdir(P) / 'LADDER' / f'{did}.json'
        lad = json.loads(f.read_text())['estimators']['MS']
        key = str(s) if str(s) in lad else None
        if key is None:
            return None
        st = lad[key]['stats']
    u = 0.0
    for c in RL.OCC_CONDITIONS:
        rho = base[c]['slack']
        if rho > 0:
            u = max(u, st[RL.STAT_OF[c]] / rho)
    return u


def frontier(bank_id, partitions, stem):
    plt = _plt()
    fig, axes = plt.subplots(1, len(partitions), figsize=(4.6 * len(partitions), 3.6), squeeze=False)
    rows = []
    for i, P in enumerate(partitions):
        s = TF.seed_of(P)
        auc = _grid_auc(bank_id, s)
        ax = axes[0][i]
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for m, arm in enumerate(('PAIR', 'LOC', 'CONT')):
            pts = sorted((float(n.split('@s')[1].split('|')[0]), n) for n in auc if n.startswith(f'REFIT_{arm}@s'))
            xy = [(_util(P, n), auc[n][0], n) for _, n in pts]
            xy = [p for p in xy if p[0] is not None]
            if xy:
                ax.plot([p[0] for p in xy], [p[1] for p in xy], color=HUES[m], marker=MARKERS[m], markersize=4.5, linewidth=1.6, label=arm)
                ax.annotate(arm, (xy[-1][0], xy[-1][1]), xytext=(4, 0), textcoords='offset points', fontsize=7, color=INK, va='center')
                rows += [[P, n, u, a] for u, a, n in xy]
        comp = [(n, _util(P, n), auc[n][0]) for n in auc if n.startswith('COMP:')]
        comp = [c for c in comp if c[1] is not None]
        if comp:
            ax.scatter([c[1] for c in comp], [c[2] for c in comp], color=HUES[3], marker=MARKERS[3], s=22, label='COMP', zorder=3)
            rows += [[P, n, u, a] for n, u, a in comp]
        if 'W' in auc:
            ax.scatter([0.0], [auc['W'][0]], color=INK, marker='*', s=60, label='witness', zorder=4)
            rows.append([P, 'W', 0.0, auc['W'][0]])
        for r in ('SUCC', 'H_LJ', 'NEURAL_RAW', 'RETIRED_B3CTL'):
            if r in auc:
                ax.axhline(auc[r][0], color=MUTED, linewidth=0.8, linestyle=':')
                ax.annotate(r, (ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.0, auc[r][0]), xytext=(-2, 2), textcoords='offset points',
                            fontsize=6.5, color=MUTED, ha='right')
                rows.append([P, r, None, auc[r][0]])
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle='--')
        ax.set_title(f'{P} (seed {s}): linked-pair SCORE AUC vs realized reserve use', fontsize=8.5, color=INK)
        ax.set_xlabel('realized MODEL_SELECT reserve use (max cost / reserve slack)')
        ax.set_ylabel('mean pair SCORE AUC over 4 cells (lower = less recovery)')
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, stem, rows, ['partition', 'release', 'realized_reserve_use', 'mean_pair_score_auc'])
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

"""Utility-recovery figure of the PCRL finite-routing comparison campaign (label-free; reads saved evaluation records and the saved
registered-grid outputs only; adapted from nrm.figures).

    <TORCH> -m frc.figures frontier --bank FRC1_MAIN                   figures/FRONTIER.{svg,png,csv}

x = utility spend: total NLL excess vs U0 (income + occupation, nats), mean over the two versions, from frc.evaluate (lower is more
useful). y = linked-pair recovery: mean registered-grid pair SCORE AUC over the 8 registered cells (lower is less recovery).
Left panel: every release and the admitted references. Right panel: the family's matched-control paths, all anchored at the task-only
twin: protected routing (PRC), the task-cell propensity grouping control (HQ), the SEX-blind untargeted re-routing controls (SHUF,
UNT) and SEX-blind coarsening (ORD). Series differ by colour and marker shape and are direct-labelled; the CSV is the table view.
Development evidence on adaptively used MODEL_SELECT; point estimates only (uncertainty is in the claims records).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

from nrm import rules as NR
from . import common as C

INK, MUTED, GRID, SURFACE = '#1f1f1e', '#6b6a63', '#e4e3dc', '#fcfcfb'
SERIES = (('PRC', 'Protected routing (PRC)', '#2a78d6', 'o', ('TWIN', 'PRC_S2')),
          ('HQ', 'Task-cell propensity grouping (HQ)', '#eb6834', 's', ('TWIN', 'HQ_S2')),
          ('UNT', 'SEX-blind untargeted re-routing (SHUF, UNT)', '#1baf7a', '^', ('TWIN', 'SHUF_S2', 'UNT_BR')),
          ('ORD', 'SEX-blind coarsening (ORD)', '#8a5cd6', 'v', ('TWIN', 'ORD_S2', 'ORD_S3')))
FAMILY = {'TWIN': ('FA', 'TWB_C64'), 'PRC_S2': ('FA', 'PRC_S2'), 'HQ_S2': ('FA', 'HQ_S2'), 'SHUF_S2': ('FA', 'SHUF_S2'),
          'UNT_BR': ('FA', 'UNT_BR'), 'ORD_S2': ('FA', 'ORD_S2'), 'ORD_S3': ('FA', 'ORD_S3'), 'TWB_C128': ('FA', 'TWB_C128'),
          'A_TWIN': ('SOURCE/FA', 'PRC_TWIN'), 'A_TWIN_C128': ('SOURCE/FA', 'PRC_TWIN_C128')}
REFS = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL', 'SUCC': 'REF:SUCC',
        'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'U0': 'REF:U0', 'Ucal': 'REF:Ucal'}


def _spend(u):
    v = {f"{c['task']}|{c['reference']}|{c['metric']}": float(c['value']) for c in u['clauses']}
    return v['income|U0|nll_excess'] + v['occupation|U0|nll_excess']


def points(bank):
    cells = {}
    for v in C.VERSIONS:
        g = json.loads((C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())
        for cell, cres in g['cells'].items():
            if cell in NR.REGISTERED:
                for name, ent in cres.items():
                    cells.setdefault(name, {})[cell] = ent['pair']['score_auc']
    full = {n for n, c in cells.items() if len(c) == len(NR.REGISTERED)}
    out = {}
    refs = json.loads((C.RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())['references']
    for name, spec in REFS.items():
        if name in full and spec in refs:
            out[name] = {'x': sum(_spend(refs[spec][str(v)]['utility']) for v in C.VERSIONS) / 2,
                         'y': sum(cells[name].values()) / len(NR.REGISTERED), 'kind': 'reference'}
    for name, (fam, cfg) in FAMILY.items():
        p = C.RUNS / fam / cfg / 'EVAL_MS.json'
        if name in full and p.is_file():
            e = json.loads(p.read_text())['versions']
            out[name] = {'x': sum(_spend(e[str(v)]['utility']) for v in C.VERSIONS) / 2,
                         'y': sum(cells[name].values()) / len(NR.REGISTERED), 'kind': 'family'}
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(MUTED); ax.spines[spine].set_linewidth(0.8)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def frontier(bank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts = points(bank)
    out = C.RESULTS / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 5.6), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.0, 1.15]})
    for ax in (axl, axr):
        _style(ax)
    table = []
    for n, p in sorted(pts.items()):
        table.append({'kind': p['kind'], 'point': n, 'total_nll_excess_vs_U0': round(p['x'], 6), 'pair_auc_registered_mean': round(p['y'], 6)})
    fam = {n: p for n, p in pts.items() if p['kind'] == 'family'}
    ref = {n: p for n, p in pts.items() if p['kind'] == 'reference'}
    axl.scatter([p['x'] for p in fam.values()], [p['y'] for p in fam.values()], s=26, color='#2a78d6', marker='o',
                edgecolors=SURFACE, linewidths=1.2, zorder=3, label='family A releases and controls (right panel)')
    axl.scatter([p['x'] for p in ref.values()], [p['y'] for p in ref.values()], s=34, color=MUTED, marker='D', edgecolors=SURFACE,
                linewidths=1.2, zorder=3, label='admitted references, U0 / Ucal')
    for n, p in ref.items():
        axl.annotate(n, (p['x'], p['y']), xytext=(4, 3), textcoords='offset points', color=MUTED, fontsize=7)
    xs = [p['x'] for p in fam.values()]; ys = [p['y'] for p in fam.values()]
    pad_x, pad_y = 0.0012, 0.0012
    axl.add_patch(matplotlib.patches.Rectangle((min(xs) - pad_x, min(ys) - pad_y), max(xs) - min(xs) + 2 * pad_x,
                                               max(ys) - min(ys) + 2 * pad_y, fill=False, edgecolor=INK, linewidth=0.8, linestyle='--'))
    axl.set_xlabel('Utility spend: income + occupation NLL excess vs U0 (nats)', color=INK, fontsize=8.5)
    axl.set_ylabel('Linked-pair SEX recovery: registered-grid pair AUC', color=INK, fontsize=8.5)
    axl.legend(loc='lower left', frameon=False, fontsize=7.5, labelcolor=INK)
    axl.set_title('All releases (dashed box = right panel)', color=INK, fontsize=9.5, loc='left')
    for key, label, color, marker, path in SERIES:
        names = [n for n in path if n in pts]
        if len(names) < 2:
            continue
        axr.plot([pts[n]['x'] for n in names], [pts[n]['y'] for n in names], color=color, linewidth=1.8, zorder=2, label=label,
                 marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.6)
    for n, p in fam.items():
        if n in ('TWB_C128', 'A_TWIN', 'A_TWIN_C128'):
            axr.scatter([p['x']], [p['y']], s=30, color=MUTED, marker='x', zorder=3)
        axr.annotate(n, (p['x'], p['y']), xytext=(5, -3 if n != 'TWIN' else 5), textcoords='offset points', color=INK, fontsize=7.5)
    axr.set_xlabel('Utility spend (nats; lower = more useful)', color=INK, fontsize=8.5)
    axr.set_title('Matched-control paths from the task-only twin (x = descriptive points)', color=INK, fontsize=9.5, loc='left')
    axr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle('Utility and linked-pair recovery on MODEL_SELECT (development evidence; lower is better on both axes)', color=INK,
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'FRONTIER.svg'); fig.savefig(out / 'FRONTIER.png', dpi=160)
    with open(out / 'FRONTIER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'point', 'total_nll_excess_vs_U0', 'pair_auc_registered_mean'])
        w.writeheader(); w.writerows(table)
    print(json.dumps({'points': len(table)}))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('frontier'); f.add_argument('--bank', required=True)
    a = ap.parse_args(argv)
    frontier(a.bank)
    return 0


if __name__ == '__main__':
    sys.exit(main())

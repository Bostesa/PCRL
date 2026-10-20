"""Utility-recovery frontier figure of the PCRL new-release-mechanisms campaign (label-free; reads saved evaluation records and the
saved registered-grid outputs only).

    <TORCH> -m nrm.figures frontier --bank NRM1_MAIN [--bank NRM1_CTL ...]     figures/FRONTIER.{svg,png,csv}

One axis. x = utility spend: total NLL excess vs U0 (income + occupation, nats), mean over the two versions, from nrm.evaluate (lower is
more useful). y = linked-pair recovery: mean registered-grid pair SCORE AUC over the 8 registered cells (lower is less recovery).
Series (validated categorical slots 1-3: blue, orange, aqua; aqua is below 3:1 contrast on the light surface, so every series is
direct-labelled, markers differ by shape and the CSV is the table view): the SEX-targeted channel path (base -> S3 -> S5), the
SEX-blind same-form MINF path and the SEX-blind deterministic CTL_REM path, all anchored at the frozen base; the nine benchmark
references and the utility references U0 / Ucal as labelled neutral points. Development evidence on adaptively used MODEL_SELECT;
the figure shows point estimates only (uncertainty is in the claims records).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

from . import common as C

INK, MUTED, GRID, SURFACE = '#1f1f1e', '#6b6a63', '#e4e3dc', '#fcfcfb'
SERIES = (('SEX', 'SEX-targeted channel (FB2)', '#2a78d6', 'o', ('FB2_BASE', 'FB2_SEX_S3', 'FB2_SEX_S5')),
          ('MINF', 'SEX-blind channel, same form (MINF)', '#eb6834', 's', ('FB2_BASE', 'FB2_MINF_S3', 'FB2_MINF_S5')),
          ('CTL', 'SEX-blind remainder flattening (CTL_REM)', '#1baf7a', '^', None))
REFS = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL', 'SUCC': 'REF:SUCC',
        'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'U0': 'REF:U0', 'Ucal': 'REF:Ucal'}
REGISTERED = tuple(f'S{v}{r}k{k}' for v in (0, 1) for r in ('R1', 'R2') for k in (0, 1))
SHORT = {'SEX': ('FB2 SEX', (0.0008, -0.0013)), 'MINF': ('MINF', (0.0008, 0.0010)), 'CTL': ('CTL_REM', (0.0008, 0.0))}
REF_OFFSET = {'U0': (0.0006, -0.0016), 'Ucal': (-0.0034, -0.0016), 'TR100': (0.0006, -0.0018), 'W': (-0.0030, -0.0016),
              'TT100': (0.0006, 0.0004), 'TWJ': (-0.0030, -0.0016)}   # label offsets (nats, AUC) that keep labels off the lines


def _spend(u):
    v = {f"{c['task']}|{c['reference']}|{c['metric']}": float(c['value']) for c in u['clauses']}
    return v['income|U0|nll_excess'] + v['occupation|U0|nll_excess']


def points(banks):
    cells = {}
    for bank in banks:
        for v in C.VERSIONS:
            g = json.loads((C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())
            for cell, cres in g['cells'].items():
                if cell in REGISTERED:
                    for name, ent in cres.items():
                        cells.setdefault(name, {})[cell] = ent['pair']['score_auc']
    rec = json.loads((C.RUNS / 'REFERENCES' / 'EVAL_MS__WIRES.json').read_text())['references']
    out = {}
    for name, spec in REFS.items():
        if name in cells and spec in rec and len(cells[name]) == len(REGISTERED):
            out[name] = {'x': sum(_spend(rec[spec][str(v)]['utility']) for v in C.VERSIONS) / 2,
                         'y': sum(cells[name].values()) / len(REGISTERED), 'kind': 'reference'}
    from .claims import eval_records                        # the newest record carrying wires (tagged records included)
    for name in cells:
        if name in out or '__d' in name or len(cells[name]) != len(REGISTERED):
            continue
        fam = next((f for f in ('FA', 'FB', 'CTL') if (C.RUNS / f / name).is_dir()), None)   # by saved folder (ENG2-09)
        if fam is None:
            continue
        e, _ = eval_records(fam, name)
        out[name] = {'x': sum(_spend(e['versions'][str(v)]['utility']) for v in C.VERSIONS) / 2,
                     'y': sum(cells[name].values()) / len(REGISTERED), 'kind': 'family'}
    return out


def frontier(banks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts = points(banks)
    out = C.RESULTS / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(MUTED); ax.spines[spine].set_linewidth(0.8)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    table = []
    for key, label, color, marker, path in SERIES:
        names = list(path) if path else ['FB2_BASE'] + sorted(n for n in pts if n.startswith('CTL_'))
        xs = [pts[n]['x'] for n in names if n in pts]; ys = [pts[n]['y'] for n in names if n in pts]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=2, label=label, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2.0)
        short, (dx, dy) = SHORT[key]
        ax.text(xs[-1] + dx, ys[-1] + dy, short, color=INK, fontsize=8, va='center')
        for n in names:
            if n in pts:
                table.append({'series': key, 'point': n, 'total_nll_excess_vs_U0': round(pts[n]['x'], 6),
                              'pair_auc_registered_mean': round(pts[n]['y'], 6)})
    drawn = {r['point'] for r in table}
    for n, p in sorted(pts.items()):                      # descriptive family points (not on a path): table view only
        if p['kind'] == 'family' and n not in drawn:
            table.append({'series': 'descriptive (table only)', 'point': n, 'total_nll_excess_vs_U0': round(p['x'], 6),
                          'pair_auc_registered_mean': round(p['y'], 6)})
    for n, p in pts.items():
        if p['kind'] == 'reference':
            ax.scatter([p['x']], [p['y']], s=36, color=MUTED, marker='D', edgecolors=SURFACE, linewidths=1.5, zorder=3,
                       label='admitted references and utility references' if n == 'NEURAL_RAW' else None)
            dx, dy = REF_OFFSET.get(n, (0.0006, 0.0006))
            ax.text(p['x'] + dx, p['y'] + dy, n, color=MUTED, fontsize=7.5, va='bottom')
            table.append({'series': 'reference', 'point': n, 'total_nll_excess_vs_U0': round(p['x'], 6),
                          'pair_auc_registered_mean': round(p['y'], 6)})
    ax.set_xlabel('Utility spend: income + occupation NLL excess vs U0 (nats; lower = more useful)', color=INK, fontsize=9)
    ax.set_ylabel('Linked-pair SEX recovery: registered-grid pair AUC\n(base instance per cell)', color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_xlim(right=max(p['x'] for p in pts.values()) + 0.006)
    ax.legend(loc='lower left', frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title('Utility-recovery frontier on MODEL_SELECT (development evidence)', color=INK, fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'FRONTIER.svg'); fig.savefig(out / 'FRONTIER.png', dpi=160)
    with open(out / 'FRONTIER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['series', 'point', 'total_nll_excess_vs_U0', 'pair_auc_registered_mean'])
        w.writeheader(); w.writerows(table)
    print(json.dumps({'points': len(table)}))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('frontier'); f.add_argument('--bank', action='append', required=True)
    a = ap.parse_args(argv)
    frontier(a.bank)
    return 0


if __name__ == '__main__':
    sys.exit(main())

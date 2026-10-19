"""Training-size figure (label-free; reads size/SIZE_TABLE.csv only).

    <TORCH> -m tsz.figures size      figures/SIZE_CURVE.{svg,png,csv}

Two panels, one axis each, x = training fraction of MODEL_FIT (50 / 70 / 100%): (a) men's occupation summed-Brier excess vs U0
(x 1e-3; the original guard cap 5 marked), (b) minimum occupation-side reserve ratio (k = 2 marked). Dots: every fitted cell (four
orderings x two versions; one fit per version at 100%); lines: the mean per recipe. Series: A1_PO_16 (the recipe), TASK_REFIT (the
equally exposed control) and T_TREE (its first stage). Palette: the validated categorical slots 1-3 (blue, orange, aqua); aqua is
below 3:1 contrast on the light surface, so every series is also direct-labelled and the CSV is the table view.
"""
from __future__ import annotations
import csv
import sys

from . import common as C

SERIES = (('A1', 'A1_PO_16 (recipe)', '#2a78d6', 'o'), ('TR', 'TASK_REFIT (control)', '#eb6834', 's'),
          ('TT', 'T_TREE (control, first stage)', '#1baf7a', '^'))
INK, MUTED, GRID, SURFACE = '#1f1f1e', '#6b6a63', '#e4e3dc', '#fcfcfb'


def size():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(C.RESULTS / 'size' / 'SIZE_TABLE.csv')))
    out = C.RESULTS / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    panels = (('guard|sex0|value', 1e3, "Men's Brier excess vs U0 (x 1e-3)", 5.0, 'original guard cap 5'),
              ('min_occ_ratio', 1.0, 'Minimum occupation-side reserve ratio', 2.0, 'reserve k = 2'))
    for ax, (col, scale, title, ref, ref_label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(MUTED); ax.spines[spine].set_linewidth(0.8)
        ax.grid(axis='y', color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.axhline(ref, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(117, ref, ref_label, color=MUTED, fontsize=8, va='bottom', ha='right')
        for j, (key, label, color, marker) in enumerate(SERIES):
            xs, ys, means = [], [], []
            for f in (50, 70, 100):
                vals = [float(r[col]) * scale for r in rows if r['recipe'] == key and int(r['fraction']) == f]
                off = (j - 1) * 1.2
                xs += [f + off] * len(vals); ys += vals
                means.append((f + off, sum(vals) / len(vals)))
                for r in rows:
                    if r['recipe'] == key and int(r['fraction']) == f:
                        table.append({'panel': title, 'recipe': label, 'fraction_percent': f, 'subset': r['subset'],
                                      'version': r['version'], 'value': float(r[col]) * scale})
            ax.scatter(xs, ys, s=22, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.0, zorder=3, alpha=0.9)
            ax.plot([m[0] for m in means], [m[1] for m in means], color=color, linewidth=2.0, zorder=2, label=label)
            ax.text(means[-1][0] + 3.0, means[-1][1], label.split(' (')[0], color=INK, fontsize=8, va='center')
        ax.set_xticks([50, 70, 100]); ax.set_xticklabels(['50%', '70%', '100%'], color=INK)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(44, 118)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('training fraction of MODEL_FIT (nested household subsets)', color=MUTED, fontsize=8.5)
    axes[0].legend(loc='lower left', frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle('Training size: occupation subgroup margin of A1_PO_16 and the equally exposed control '
                 '(MODEL_SELECT, development evidence)', color=INK, fontsize=10.5, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'dots: each fit (4 orderings x 2 versions; one fit per version at 100%); lines: means. Lower excess and '
                          'higher ratio are better.', color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ('svg', 'png'):
        fig.savefig(out / f'SIZE_CURVE.{ext}', dpi=160, facecolor=SURFACE)
    with open(out / 'SIZE_CURVE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print('ok', len(table))


if __name__ == '__main__':
    if sys.argv[1:] != ['size']:
        raise SystemExit(__doc__)
    size()

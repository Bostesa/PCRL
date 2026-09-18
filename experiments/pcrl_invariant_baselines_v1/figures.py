"""Stage 8c: figures. Reads committed evidence only; computes no new endpoint.

Form: **forest plot**. The data's job is polarity with uncertainty -- an estimate and
its adjusted simultaneous interval, read against zero -- which is what a forest plot is
for. A bar chart would be wrong: the quantity is signed and the interval is the point.

Colour: **diverging**, because the encoded variable is polarity (better / worse / not
distinguishable), not identity. Blue `#2a78d6` and red `#e34948` are the validated
diverging poles; the neutral is `#52514e`.

Validated rather than eyeballed (`scripts/validate_palette.js`, light surface
`#fcfcfb`, `--pairs all`): the blue/red pair passes every check -- lightness band,
chroma floor, CVD separation (worst pair Delta E 21.6 protan, 34.5 tritan),
normal-vision floor (32.3) and contrast. The neutral is a diverging midpoint rather
than a categorical slot, so it is held to contrast instead of the chroma floor:
7.73:1 against the surface.

**Identity is never carried by colour alone.** Whether an interval crosses zero is
visible in the geometry, and every row is directly labelled with its estimate. The
figures are static PDF/PNG for a LaTeX manuscript, matching the completed study's
convention, so there is no hover layer and no dark mode -- a print surface has one mode.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .run_fit import OUT

BETTER = '#2a78d6'      # validated diverging pole: interval entirely below zero
WORSE = '#e34948'       # validated diverging pole: interval entirely above zero
NEUTRAL = '#52514e'     # diverging midpoint: interval spans zero
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#dedcd6'

SHORT = {'utility/same_residence': 'residence (utility)', 'recovery/A/SEX': 'A / SEX',
         'recovery/AB/SEX': 'AB / SEX', 'recovery/A/RAC1P': 'A / RAC1P',
         'recovery/AB/RAC1P': 'AB / RAC1P'}
ORDER = ('recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P',
         'utility/same_residence')


def read_intervals(out: Path) -> list:
    with open(out / 'PAIRED_INTERVALS.csv') as handle:
        return list(csv.DictReader(handle))


def colour_of(row: dict) -> str:
    if row['significantly_better'] in ('True', True):
        return BETTER
    if row['significantly_worse'] in ('True', True):
        return WORSE
    return NEUTRAL


def forest(ax, rows, contrasts, weight, title, captions=True):
    import numpy as np

    labels, y = [], 0
    ticks, ticklabels = [], []
    for left, right, caption in contrasts:
        for endpoint in ORDER:
            match = [r for r in rows if r['left'] == left and r['right'] == right
                     and r['weight'] == weight and r['endpoint'] == endpoint]
            if not match:
                continue
            r = match[0]
            est = float(r['estimate'])
            low, high = float(r['adjusted_low']), float(r['adjusted_high'])
            colour = colour_of(r)
            # 2px interval line, >=8px marker, both in the polarity colour
            ax.plot([low, high], [y, y], color=colour, linewidth=2.0,
                    solid_capstyle='round', zorder=3)
            ax.plot([est], [y], marker='o', markersize=6, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
            # direct label: identity and magnitude never depend on colour alone
            ax.annotate(f'{est:+.4f}', (high, y), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=7.0,
                        color=INK)
            ticks.append(y)
            ticklabels.append(SHORT[endpoint])
            y -= 1
        labels.append((y + len(ORDER) / 2.0, caption))
        y -= 0.9

    ax.axvline(0.0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=7.5, color=INK)
    ax.tick_params(axis='x', labelsize=7.5, colors=MUTED, length=3)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc='left', pad=8)
    if captions:
        # Group captions on the first panel only: the two panels share rows, so
        # repeating them crowds the gap between panels and adds no information.
        for position, caption in labels:
            ax.annotate(caption, (0.0, position), xycoords=('axes fraction', 'data'),
                        xytext=(-118, 0), textcoords='offset points', fontsize=7.5,
                        color=INK, fontweight='bold', va='center', annotation_clip=False)
    return ax


def pad_right(axes):
    """Headroom on the right so the direct labels never collide with the panel edge."""
    low, high = axes[0].get_xlim()
    axes[0].set_xlim(low, high + 0.32 * (high - low))


def legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BETTER, lw=2, marker='o', markersize=6,
                      label='left arm significantly better'),
               Line2D([], [], color=WORSE, lw=2, marker='o', markersize=6,
                      label='left arm significantly worse'),
               Line2D([], [], color=NEUTRAL, lw=2, marker='o', markersize=6,
                      label='interval spans zero')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))


def build(out: Path) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = read_intervals(out)
    dest = out / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    written = []

    # Figure 1 -- the headline negative: the repair makes measured recovery worse.
    contrasts = [('spectral_riv16_C1', 'spectral_nlr16_C1', 'rank 16, C1'),
                 ('spectral_riv8_C1', 'spectral_nlr8_C1', 'rank 8, C1')]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), facecolor=SURFACE, sharex=True)
    for index, (ax, weight) in enumerate(zip(axes, ('unweighted', 'person_weighted'))):
        forest(ax, rows, contrasts, weight, weight.replace('_', '-'), captions=index == 0)
        ax.set_xlabel('repaired minus defective   (negative = repaired better)',
                      fontsize=7.5, color=MUTED)
    pad_right(axes)
    fig.suptitle('Repairing the rotation defect did not reduce measured disclosure',
                 fontsize=10.5, color=INK, x=0.012, ha='left')
    legend(fig)
    fig.tight_layout(rect=(0.10, 0.07, 1, 0.94))
    for suffix in ('pdf', 'png'):
        path = dest / f'repair_vs_defective.{suffix}'
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        written.append(path)
    plt.close(fig)

    # Figure 2 -- the external baselines against the repaired mechanism.
    contrasts = [('leace_A0', 'spectral_riv16_C1', 'LEACE'),
                 ('splince_A0', 'spectral_riv16_C1', 'SPLINCE'),
                 ('optnet16_C1', 'spectral_riv16_C1', 'OptNet-ARL')]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), facecolor=SURFACE, sharex=True)
    for index, (ax, weight) in enumerate(zip(axes, ('unweighted', 'person_weighted'))):
        forest(ax, rows, contrasts, weight, weight.replace('_', '-'), captions=index == 0)
        ax.set_xlabel('baseline minus repaired   (negative = baseline better)',
                      fontsize=7.5, color=MUTED)
    pad_right(axes)
    fig.suptitle('Three published adaptations beat the developed mechanism',
                 fontsize=10.5, color=INK, x=0.012, ha='left')
    legend(fig)
    fig.tight_layout(rect=(0.10, 0.05, 1, 0.95))
    for suffix in ('pdf', 'png'):
        path = dest / f'baselines_vs_repaired.{suffix}'
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        written.append(path)
    plt.close(fig)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    for path in build(a.out):
        print('wrote', path)


if __name__ == '__main__':
    main()

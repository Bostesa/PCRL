"""Utility panel: residence loss of each view relative to the same-host J comparator.

Form choice: the job is magnitude comparison across a small set of named views, where the
meaningful differences (0.005-0.02 nats) are tiny against the ~0.50 absolute loss. Plotting the
*difference versus the registered comparator* gives a real zero reference instead of a truncated
axis. Error bars are the simultaneous (Bonferroni, frozen family of 68) half-widths.

Deliberately NOT called a frontier: no disclosure axis was measured, because the gate closed the
branch before any audit ran.
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RES = pathlib.Path('/Users/nathansamson/PCRL-terminal-1-replacement/results/'
                   'pcrl_stochastic_replacement_overnight_v1')
iv = json.load(open(RES / 'stage_R' / 'INTERVALS.json'))
adj = json.load(open(RES / 'stage_R' / 'INTERVALS_ADJUSTED.json'))
Z = adj['critical_value_adjusted']

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
UNW = '#2a78d6'     # categorical slot 1
PW = '#eb6834'      # categorical slot 2
GRID = '#dedcd5'

ROWS = [
    ('H only  — no channel',                     'pca32|k64|J_minus_H',   -1),
    ('H + T   zj, k=256  — fallback',            'zj|k256|T_minus_J',     +1),
    ('H + T   zj, k=64',                              'zj|k64|T_minus_J',      +1),
    ('H + T   pca32, k=256  — fallback',         'pca32|k256|T_minus_J',  +1),
    ('H + T   pca32, k=64  — best code tested',  'pca32|k64|T_minus_J',   +1),
    ('H + raw  zj  — identity check (= J)',      'zj|k64|raw_minus_J',    +1),
    ('H + raw  pca32  — unquantized',            'pca32|k64|raw_minus_J', +1),
]

fig, ax = plt.subplots(figsize=(10.2, 6.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = np.arange(len(ROWS))
off = 0.19

lo, hi = 0.0, 0.0
for i, (label, key, sign) in enumerate(ROWS):
    for series, colour, dy in (('unw', UNW, +off), ('pw', PW, -off)):
        r = iv[f'{key}|{series}']
        est, se = sign * r['estimate'], r['bootstrap_se']
        half = Z * se
        lo, hi = min(lo, est - half), max(hi, est + half)
        ax.errorbar(est, y[i] + dy, xerr=half, fmt='o', ms=8, color=colour,
                    ecolor=colour, elinewidth=2.2, capsize=0, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        # Label beyond the bar end, on the side away from zero, so labels never stack.
        outward = 1 if est >= 0 else -1
        ax.annotate(f'{est:+.4f}', (est + outward * half, y[i] + dy),
                    textcoords='offset points', xytext=(7 * outward, 0),
                    ha='left' if outward > 0 else 'right', va='center',
                    fontsize=8.6, color=INK2)

pad = (hi - lo) * 0.19
ax.set_xlim(lo - pad, hi + pad)

ax.axvline(0.0, color=INK, lw=1.5, zorder=2)
ax.axvline(0.001, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.6, color=INK)
ax.set_xlabel('residence log loss relative to same-host J  (nats)'
              '        ← better              worse →',
              fontsize=9.8, color=INK2, labelpad=9)

fig.suptitle('Replacement-channel utility on residence',
             fontsize=13, color=INK, x=0.012, ha='left', y=0.975)
fig.text(0.012, 0.923,
         '2018 development pools, three anchors pooled over 5,443 shared households.  '
         f'Bars = simultaneous one-sided half-width (Bonferroni, frozen family of 68, z={Z:.2f}).',
         fontsize=8.7, color=INK2, ha='left')

ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(axis='x', colors=INK2, labelsize=8.8)
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.75, len(ROWS) - 0.25)

# Reference-line labels in the top margin, clear of every mark.
top = len(ROWS) - 0.42
ax.annotate('same-host J', (0.0, top), xytext=(5, 0), textcoords='offset points',
            fontsize=8.8, color=INK, va='center', ha='left')
ax.annotate('.001 screen allowance', (0.001, top - 0.30), xytext=(6, 0),
            textcoords='offset points', fontsize=8.4, color=INK2, va='center', ha='left')

handles = [plt.Line2D([], [], marker='o', ls='', ms=8, color=UNW, label='unweighted'),
           plt.Line2D([], [], marker='o', ls='', ms=8, color=PW, label='person-weighted (PWGTP)')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.115), ncol=2,
          frameon=False, fontsize=9.4, labelcolor=INK2, handletextpad=0.5, columnspacing=2.2)

fig.subplots_adjust(top=0.875, bottom=0.165, left=0.305, right=0.965)
for ext in ('png', 'pdf'):
    fig.savefig(RES / f'UTILITY_PANEL.{ext}', dpi=200, facecolor=SURFACE)
print('wrote UTILITY_PANEL.png / .pdf')

table = []
for label, key, sign in ROWS:
    row = {'view': label}
    for series in ('unw', 'pw'):
        r = iv[f'{key}|{series}']
        row[series] = {'estimate_vs_J': sign * r['estimate'], 'bootstrap_se': r['bootstrap_se'],
                       'simultaneous_half_width': Z * r['bootstrap_se'],
                       'households_union': r['households_union']}
    table.append(row)
json.dump({'rows': table, 'adjusted_z': Z, 'family_size': adj['family_size'],
           'comparator': 'same-host J = [H_A, Z_J], ancestor-inclusive',
           'not_a_frontier': ('no disclosure axis was measured; the gate closed the branch before '
                              'any audit ran, so this is a utility panel only')},
          open(RES / 'UTILITY_PANEL.json', 'w'), indent=1)
print('wrote UTILITY_PANEL.json')

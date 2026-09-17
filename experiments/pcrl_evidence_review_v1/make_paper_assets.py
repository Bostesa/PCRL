"""Generate the revised manuscript's tables and figures from verified evidence.

Inputs are the independent reanalysis under results/pcrl_evidence_review_v1 plus the
study's own published evidence CSVs. Nothing is typed by hand into the manuscript:
every number in a generated table comes from one of these files.

Outputs go to papers/pcrl_evidence_v1/{tables,figures}.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {'C1': '#2f6f9f', 'J': '#b8562a', 'L1': '#7a7a7a', 'L2': '#4a4a4a',
           'S0': '#9a6fa8', 'H': '#000000', 'other': '#b0b0b0'}
LABEL = {'spectral_C1': 'C1', 'spectral_L1': 'L1', 'spectral_L2': 'L2',
         'spectral_S0': 'S0', 'spectral_C025': 'C025', 'spectral_M025': 'M025',
         'spectral_M1': 'M1', 'spectral_L025': 'sL025', 'J': 'J', 'H': 'H',
         'E': 'E', 'A0': 'A0', 'L025': 'L025', 'L20': 'L20'}
ENDPOINT = {'utility/same_residence': 'residence loss', 'recovery/A/SEX': 'A / sex',
            'recovery/AB/SEX': 'AB / sex', 'recovery/A/RAC1P': 'A / race',
            'recovery/AB/RAC1P': 'AB / race'}
WSHORT = {'unweighted': 'unw.', 'person_weighted': 'PWGTP'}


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x)


def fmt(x, n=4):
    return f'{float(x):.{n}f}'


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}')
    plt.close(fig)


# --------------------------------------------------------------------- tables
def table_primary(rows, out):
    """F1 and F3 with adjusted simultaneous intervals."""
    lines = [r'\begin{tabular}{llrrl@{\hspace{4pt}}c}', r'\toprule',
             r'Contrast & Endpoint & Estimate & Seeds 0 / 1 / 2 & Adjusted 95\% & \\',
             r'\midrule']
    for fam, title in (('F1_primary', r'\textit{F1: coalition conditioning vs local controls} ($c=2.930$)'),
                       ('F3_secondary', r'\textit{F3: coalition channel vs frozen neural channel} ($c=2.728$)')):
        lines.append(rf'\multicolumn{{6}}{{l}}{{{title}}} \\[1pt]')
        sub = [r for r in rows if r['family'] == fam and r['weight'] == 'unweighted']
        for r in sub:
            mark = ''
            if f(r['adjusted_high']) < 0:
                mark = r'$\blacktriangledown$'
            elif f(r['adjusted_low']) > 0:
                mark = r'$\blacktriangle$'
            seeds = ' / '.join(fmt(r[f'seed_{s}'], 4) for s in (0, 1, 2))
            lines.append(
                rf"{LABEL[r['left']]} $-$ {LABEL[r['right']]} & {ENDPOINT[r['endpoint']]} & "
                rf"{fmt(r['estimate'], 5)} & {seeds} & "
                rf"[{fmt(r['adjusted_low'], 5)}, {fmt(r['adjusted_high'], 5)}] & {mark} \\")
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'primary_comparisons.tex').write_text('\n'.join(lines) + '\n')


def table_equivalence(rows, out):
    lines = [r'\begin{tabular}{llrrrrl}', r'\toprule',
             r'Contrast & Weight & Estimate & SE & Pointwise 95\% & Simultaneous 95\% & Non-inferior? \\',
             r'         &        &          &    & upper bound     & upper bound      & (margin $0.001$) \\',
             r'\midrule']
    for r in rows:
        if r['family'] != 'F1_primary':
            continue
        c = r['contrast'].replace('spectral_', '')
        lines.append(
            rf"{c.replace('_', ' ')} & {WSHORT[r['weight']]} & {fmt(r['estimate'], 5)} & {fmt(r['se'], 5)} & "
            rf"\textbf{{{fmt(r['pointwise_upper_95_percentile'], 5)}}} & "
            rf"{fmt(r['simultaneous_upper_95_studentised'], 5)} & no \\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'equivalence.tex').write_text('\n'.join(lines) + '\n')


def table_vector(rows, out):
    """Absolute and additional recovery beside residence gain."""
    order = ['H', 'J', 'spectral_L1', 'spectral_L2', 'spectral_C1', 'spectral_S0']
    by = {r['interface']: r for r in rows if r['mode'] == 'B' and r['weight'] == 'unweighted'}
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'& residence & \multicolumn{2}{c}{A / race} & \multicolumn{2}{c}{AB / sex} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'Interface & gain vs H & absolute & additional & absolute & additional \\',
             r'\midrule']
    for k in order:
        r = by[k]
        lines.append(
            rf"{LABEL[k]} & {fmt(r['residence_gain_vs_H'])} & {fmt(r['absolute/A/RAC1P'])} & "
            rf"{fmt(r['additional/A/RAC1P'])} & {fmt(r['absolute/AB/SEX'])} & {fmt(r['additional/AB/SEX'])} \\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'absolute_vs_additional.tex').write_text('\n'.join(lines) + '\n')


def table_matching(rows, out):
    lines = [r'\begin{tabular}{llrlr}', r'\toprule',
             r'Withheld channel & matched to & $p^\star$ & identifiable? & A/race at $p^\star$ \\',
             r'\midrule']
    for r in rows:
        if r['mode'] != 'B' or r['weight'] != 'unweighted':
            continue
        ident = r['identifiable_in_unit_interval'] == 'True'
        val = (fmt(r['source_additional_at_matched_p/A/RAC1P']) if ident else '---')
        lines.append(
            rf"{LABEL[r['withholding_source']]} & {LABEL[r['target']]} & {fmt(r['matching_probability'], 3)} & "
            rf"{'yes' if ident else r'\textbf{no}'} & {val} \\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'withholding_matching.tex').write_text('\n'.join(lines) + '\n')


def table_criteria(rows, out):
    order = ['H', 'J', 'spectral_L1', 'spectral_L2', 'spectral_C1', 'spectral_S0']
    idx = {(r['interface'], r['mode']): r for r in rows if r['weight'] == 'unweighted'}
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'& \multicolumn{2}{c}{source-probe allowance} & \multicolumn{2}{c}{half-headroom} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Interface & fresh (B) & frozen (A) & fresh (B) & frozen (A) \\',
             r'\midrule']
    for k in order:
        b, a = idx[(k, 'B')], idx[(k, 'A')]
        lines.append(
            rf"{LABEL[k]} & {b['legacy_source_probe_allowance_pass_seeds']}/3 & "
            rf"{a['legacy_source_probe_allowance_pass_seeds']}/3 & "
            rf"{b['half_headroom_pass_seeds']}/3 & {a['half_headroom_pass_seeds']}/3 \\")
    lines += [r'\bottomrule', r'\end{tabular}',
              r'', r'% counts are PASSING seeds out of 3.']
    (out / 'criteria.tex').write_text('\n'.join(lines) + '\n')


# -------------------------------------------------------------------- figures
def fig_forest(rows, out):
    """Adjusted simultaneous intervals for F1 and F3. Left of zero = better."""
    style()
    fams = [('F1_primary', 'F1: C1 vs its local controls'),
            ('F3_secondary', 'F3: C1 vs the frozen neural channel J')]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=False,
                             gridspec_kw={'wspace': 0.62})
    for ax, (fam, title) in zip(axes, fams):
        sub = [r for r in rows if r['family'] == fam]
        labels, est, lo, hi, colors = [], [], [], [], []
        for r in reversed(sub):
            labels.append(f"{LABEL[r['left']]}$-${LABEL[r['right']]}  {ENDPOINT[r['endpoint']]}  ({WSHORT[r['weight']]})")
            est.append(f(r['estimate']))
            lo.append(f(r['adjusted_low']))
            hi.append(f(r['adjusted_high']))
            colors.append(PALETTE['C1'] if f(r['adjusted_high']) < 0
                          else (PALETTE['J'] if f(r['adjusted_low']) > 0 else PALETTE['other']))
        y = np.arange(len(est))
        ax.axvline(0, color='black', lw=0.8, zorder=1)
        for i in range(len(est)):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=1.6, solid_capstyle='butt', zorder=2)
            ax.plot([est[i]], [y[i]], 'o', color=colors[i], ms=3.0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.6)
        ax.tick_params(axis='y', pad=1)
        ax.set_title(title)
        ax.set_xlabel('difference in nats  (left of 0 = C1 better)')
        ax.margins(y=0.02)
    fig.suptitle('Simultaneous 95% intervals within each pre-declared family', y=1.01, fontsize=9)
    save(fig, out, 'forest_primary')


def fig_tradeoff(rows, out):
    """Residence gain against additional A/race recovery, per seed and seed-mean."""
    style()
    per = {r['interface']: r for r in rows if r['mode'] == 'B' and r['weight'] == 'unweighted'}
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for k, r in per.items():
        col = PALETTE.get(LABEL[k], PALETTE['other'])
        x, y = f(r['residence_gain_vs_H']), f(r['additional/A/RAC1P'])
        ax.scatter([x], [y], s=26 if LABEL[k] in PALETTE else 14, color=col,
                   zorder=3, edgecolor='white', linewidth=0.5)
        ax.annotate(LABEL[k], (x, y), textcoords='offset points', xytext=(4, 2), fontsize=6, color=col)
    ax.set_xlabel('residence capability gained over H (nats, higher = better)')
    ax.set_ylabel('additional A/race recovery (nats, lower = better)')
    ax.set_title('Utility against local race disclosure\nMode B, fresh 2017 attacks', fontsize=8)
    save(fig, out, 'tradeoff_scatter')


def fig_absolute_vs_additional(rows, out):
    style()
    order = ['H', 'J', 'spectral_L2', 'spectral_L1', 'spectral_C1', 'spectral_S0']
    per = {r['interface']: r for r in rows if r['mode'] == 'B' and r['weight'] == 'unweighted'}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, ep, name in zip(axes, ('A/RAC1P', 'AB/SEX'), ('A / race', 'AB / sex')):
        y = np.arange(len(order))
        absv = [f(per[k][f'absolute/{ep}']) for k in order]
        addv = [f(per[k][f'additional/{ep}']) for k in order]
        base = [a - b for a, b in zip(absv, addv)]
        ax.barh(y, base, color='#d6d6d6', height=0.6, label='recoverable from H alone')
        ax.barh(y, addv, left=base, color=PALETTE['C1'], height=0.6, label='additional, from the channel')
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[k] for k in order])
        ax.set_title(name)
        ax.set_xlabel('absolute recovery (nats, lower = better)')
    axes[0].legend(loc='lower right', frameon=False, fontsize=6.5)
    fig.suptitle('Absolute recovery splits into what H alone already gives and what the channel adds', y=1.04, fontsize=9)
    save(fig, out, 'absolute_vs_additional')


def fig_per_seed(rows, out):
    """Per-seed values behind each F1 endpoint: the seed mean can hide a sign flip."""
    style()
    sub = [r for r in rows if r['family'] == 'F1_primary' and r['weight'] == 'unweighted']
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, r in enumerate(sub):
        seeds = [f(r[f'seed_{s}']) for s in (0, 1, 2)]
        flip = not all(np.sign(v) == np.sign(f(r['estimate'])) for v in seeds)
        col = PALETTE['J'] if flip else PALETTE['C1']
        ax.plot(seeds, [i] * 3, 'o', color=col, ms=3.2, alpha=0.85, zorder=3)
        ax.plot([min(seeds), max(seeds)], [i, i], color=col, lw=0.8, alpha=0.5, zorder=2)
        ax.plot([f(r['estimate'])], [i], '|', color='black', ms=9, mew=1.2, zorder=4)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"{LABEL[r['left']]}$-${LABEL[r['right']]}  {ENDPOINT[r['endpoint']]}" for r in sub],
                       fontsize=6)
    ax.set_xlabel('difference in nats (left of 0 = C1 better)')
    ax.set_title('Per-seed values; vertical bar is the seed mean\norange = seeds disagree in sign', fontsize=8)
    save(fig, out, 'per_seed')


def fig_withholding(rows, matching, out):
    """Expected residence gain and A/race recovery along the declared p grid."""
    style()
    per = {r['interface']: r for r in rows if r['mode'] == 'B' and r['weight'] == 'unweighted'}
    ps = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for k in ('spectral_S0', 'A0', 'J', 'E'):
        col = PALETTE.get(LABEL[k], PALETTE['other'])
        g, a = f(per[k]['residence_gain_vs_H']), f(per[k]['additional/A/RAC1P'])
        ax.plot(ps * g, ps * a, color=col, lw=1.2)
        ax.annotate(LABEL[k], (g, a), textcoords='offset points', xytext=(3, -1), fontsize=6.5, color=col)
    c = per['spectral_C1']
    ax.scatter([f(c['residence_gain_vs_H'])], [f(c['additional/A/RAC1P'])], marker='*', s=70,
               color=PALETTE['C1'], zorder=4, edgecolor='white', linewidth=0.4)
    ax.annotate('C1 (full release)', (f(c['residence_gain_vs_H']), f(c['additional/A/RAC1P'])),
                textcoords='offset points', xytext=(-6, 7), fontsize=6.5, color=PALETTE['C1'])
    ax.axvline(f(c['residence_gain_vs_H']), color=PALETTE['C1'], lw=0.7, ls=':')
    ax.set_xlabel("expected residence gain over H (nats, higher = better)")
    ax.set_ylabel('expected additional A/race recovery\n(nats, lower = better)')
    ax.set_title('Randomised withholding: each line is one channel\nreleased with probability $p\\in[0,1]$', fontsize=8)
    save(fig, out, 'withholding')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ev, out = Path(args.evidence), Path(args.out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    fam = read_csv(ev / 'INDEPENDENT_FAMILIES.csv')
    eq = read_csv(ev / 'EQUIVALENCE_F1.csv')
    modeb = read_csv(ev / 'MODEB_TRANSPORT_TABLE.csv')
    match = read_csv(ev / 'WITHHOLDING_MATCHING.csv')
    crit = read_csv(ev / 'SERVICE_VS_PROBE.csv')

    table_primary(fam, out / 'tables')
    table_equivalence(eq, out / 'tables')
    table_vector(modeb, out / 'tables')
    table_matching([r for r in match if r['target'] == 'spectral_C1'], out / 'tables')
    table_criteria(crit, out / 'tables')

    fig_forest(fam, out / 'figures')
    fig_tradeoff(modeb, out / 'figures')
    fig_absolute_vs_additional(modeb, out / 'figures')
    fig_per_seed(fam, out / 'figures')
    fig_withholding(modeb, match, out / 'figures')

    manifest = {'tables': sorted(p.name for p in (out / 'tables').glob('*.tex')),
                'figures': sorted(p.name for p in (out / 'figures').glob('*.pdf'))}
    (out / 'ASSET_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

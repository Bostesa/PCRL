"""Discrete Pareto plots: residence gain over H against each sensitive increment, separately.

One figure per weighting, four panels (A/SEX, AB/SEX, A/RAC1P, AB/RAC1P), one y-scale
each; the four risks are never averaged. Seed means from `POINTS_2018.csv`; the same
rows are written to `PARETO_2018.csv` as the machine-readable backing. Group identity
is carried by marker shape AND colour (validated palette; contrast WARN relieved by
shapes, direct labels on references/panel, and the CSV table). Hollow markers fail the
historical source allowance on at least one anchor; retained rank is labelled on panel
members.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import OUT, read_json, sha_file, write_json_atomic

GROUPS = [('Track N, A0 init', '#2a78d6', 'o'), ('Track N, J init', '#eb6834', 's'),
          ('Track E coalition C', '#1baf7a', '^'), ('Track E local L / LX', '#eda100', 'v'),
          ('Controls: marginal / PCA / random', '#e87ba4', 'D'),
          ('References', '#4a3aa7', 'X')]
REFERENCES = ('A0', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1', 'leace_J', 'splince_J',
              'ref_A0', 'ref_J')
ENDPOINTS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')


def group(condition: str) -> str:
    if condition in REFERENCES:
        return 'References'
    if condition.startswith('N_A0'):
        return 'Track N, A0 init'
    if condition.startswith('N_J'):
        return 'Track N, J init'
    if condition.startswith('E_') and '_C_' in condition:
        return 'Track E coalition C'
    if condition.startswith('E_') and ('_L_' in condition or '_LX_' in condition):
        return 'Track E local L / LX'
    return 'Controls: marginal / PCA / random'


def seed_mean_rows():
    rows = defaultdict(list)
    with open(OUT / 'POINTS_2018.csv') as fh:
        for r in csv.DictReader(fh):
            rows[r['condition'], r['weight']].append(r)
    out = []
    for (condition, weight), rs in rows.items():
        if condition == 'H' or condition.endswith('_full') and False:
            continue
        row = {'condition': condition, 'weight': weight, 'group': group(condition),
               'residence_gain_vs_H': sum(float(r['residence_gain_vs_H']) for r in rs) / len(rs),
               'source_allowance_all_anchors': all(r['source_allowance_pass'] == 'True' for r in rs)}
        for e in ENDPOINTS:
            row['increment/' + e] = sum(float(r['increment/' + e]) for r in rs) / len(rs)
        out.append(row)
    return out


def run(panel_units=()) -> dict:
    rows = seed_mean_rows()
    with open(OUT / 'PARETO_2018.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = {}
    for weight in ('unweighted', 'person_weighted'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
        for ax, endpoint in zip(axes.ravel(), ENDPOINTS):
            for name, color, marker in GROUPS:
                pts = [r for r in rows if r['weight'] == weight and r['group'] == name]
                for filled in (True, False):
                    sub = [r for r in pts if r['source_allowance_all_anchors'] == filled]
                    if not sub:
                        continue
                    ax.scatter([r['increment/' + endpoint] for r in sub],
                               [r['residence_gain_vs_H'] for r in sub], s=36, marker=marker,
                               facecolors=color if filled else 'none', edgecolors=color,
                               linewidths=1.2, label=name if filled and endpoint == 'A/SEX' else None,
                               alpha=0.9)
            for r in rows:
                if r['weight'] != weight:
                    continue
                if r['condition'] in ('A0', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1') \
                        or r['condition'] in panel_units:
                    ax.annotate(r['condition'], (r['increment/' + endpoint],
                                                 r['residence_gain_vs_H']),
                                fontsize=7, color='#333333', xytext=(3, 3),
                                textcoords='offset points')
            ax.axvline(0, color='#bbbbbb', lw=0.8)
            ax.axhline(0.01, color='#bbbbbb', lw=0.8, ls='--')
            ax.set_title(f'{endpoint}: sensitive increment over H (lower = less disclosure)',
                         fontsize=9)
            ax.set_xlabel('increment over H (nats)', fontsize=8)
            ax.set_ylabel('residence gain over H (nats; dashed = .01 reference)', fontsize=8)
            ax.grid(color='#eeeeee', lw=0.5)
        fig.legend(loc='lower center', ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f'2018 development frontier, seed means, {weight} — hollow = fails '
                     'historical source allowance on >=1 anchor', fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        path = OUT / f'PARETO_2018_{weight}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files[path.name] = sha_file(path)
    files['PARETO_2018.csv'] = sha_file(OUT / 'PARETO_2018.csv')
    write_json_atomic(OUT / 'FIGURES.json', {'figures': files,
                                             'backing': 'PARETO_2018.csv (seed means of POINTS_2018.csv)',
                                             'points_sha256': sha_file(OUT / 'POINTS_2018.csv')})
    return files

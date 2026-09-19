"""Render the result documents from the machine-readable ledgers.

Numbers are read out of DEVELOPMENT_2018.json, TRACK_N_FITS.json, TRACK_E_FITS.json,
COUNTS.json, VALIDATION.json, PANEL.json, STRESS.json and EXPLORATORY_2017.json; the
source hashes are printed in each document. Interpretation lives in
RESEARCH_DECISION.md, which is written by hand against these tables.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import OUT, read_json, sha_file, utcnow, write_text_atomic

SENS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
REFS = ('A0', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1', 'leace_J', 'splince_J')


def f(x, d=4):
    return 'n/a' if x is None else f'{x:+.{d}f}'


def means(dev, condition, weight='unweighted'):
    return dev['seed_means'].get(f'{condition}|{weight}')


def row(dev, condition, weight='unweighted'):
    m = means(dev, condition, weight)
    if m is None:
        return None
    return (f"| `{condition}` | " + ' | '.join(f(m['increment/' + e]) for e in SENS)
            + f" | {f(m['residence_gain_vs_H'])} | {m['source_allowance_pass_seeds']}/3 |")


HEADER = ('| condition | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain | source ok |\n'
          '|---|---|---|---|---|---|---|')


def development_md(dev, n_fits, e_fits, counts, panel) -> str:
    L = ['# DEVELOPMENT_2018 — full grid and panel, matched-exposure scope, budget 360', '',
         f'Generated {utcnow()}. Backing: `DEVELOPMENT_2018.json` '
         f'(`{sha_file(OUT / "DEVELOPMENT_2018.json")}`), `POINTS_2018.csv`, '
         '`INTERVALS_P.csv`, `INTERVALS_X.csv`. Seed means, increments over the `H` view '
         '(negative observed increments are kept: finite-selection artifacts, not removal of '
         'information already in `H`). **All numbers are 2018 DEVELOPMENT on repeatedly used '
         'pools; residence is reserved-from-training capability, not a blind endpoint.**', '']
    for weight in ('unweighted', 'person_weighted'):
        L += [f'## References ({weight})', '', HEADER]
        L += [r for r in (row(dev, c, weight) for c in REFS + ('ref_A0', 'ref_J')) if r]
        L += ['']
    L += ['## Track N (unweighted)', '', HEADER]
    for init in ('A0', 'J'):
        for g in ('g000', 'g001', 'g010', 'g100'):
            for pol in ('C1', 'L1', 'L2'):
                for b in ('b100', 'b300'):
                    r = row(dev, f'N_{init}_{g}_{pol}_{b}')
                    if r:
                        L.append(r)
            r = row(dev, f'N_{init}_{g}_none')
            if r:
                L.append(r)
    L += ['', '## Track E (unweighted)', '', HEADER]
    for ch in ('A0', 'J'):
        for k in (1, 2, 4, 6, 8):
            for pol in ('C', 'L', 'LX', 'marginal', 'pca', 'rand'):
                r = row(dev, f'E_{ch}_{pol}_k{k}')
                if r:
                    L.append(r)
        for pol in ('C', 'L', 'LX'):
            r = row(dev, f'E_{ch}_{pol}_full')
            if r:
                L.append(r)
    L += ['', f"Family sizes: P = {dev['family_sizes'].get('P')}, "
              f"X = {dev['family_sizes'].get('X')} rows (contrasts x 5 endpoints x 2 weightings)."]
    return '\n'.join(L) + '\n'


def main():
    dev = read_json(OUT / 'DEVELOPMENT_2018.json')
    n_fits = read_json(OUT / 'TRACK_N_FITS.json')
    e_fits = read_json(OUT / 'TRACK_E_FITS.json')
    counts = read_json(OUT / 'COUNTS.json')
    panel = read_json(OUT / 'PANEL.json') if (OUT / 'PANEL.json').exists() else None
    write_text_atomic(OUT / 'DEVELOPMENT_2018.md', development_md(dev, n_fits, e_fits, counts,
                                                                   panel))
    print('wrote DEVELOPMENT_2018.md')


if __name__ == '__main__':
    main()

"""Generate the integrated manuscript's tables and figures from committed evidence.

Two completed studies feed this generator:

  * the locked 2017 transport study and its independent reanalysis
    (results/redesign_20260917_acs_spectral_transport_v1, results/pcrl_evidence_review_v1)
  * the 2018 nonlinear-penalty / rank development study and its exploratory 2017 reuse
    (results/pcrl_nonlinear_rank_v1)

No number is typed by hand into the manuscript. Every generated table and figure records
its source files in MANIFEST.json with a sha256, so a reader can trace a printed digit
back to a committed artifact.

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2 \
        --repo . --out papers/pcrl_manuscript_v2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# One BLAS/OpenMP thread: this machine has been observed at swap capacity.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------- shared

TRANSPORT = 'results/redesign_20260917_acs_spectral_transport_v1'
REVIEW = 'results/pcrl_evidence_review_v1'
NONLIN = 'results/pcrl_nonlinear_rank_v1'

SENSITIVE = ['recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P']
ENDPOINT = {'utility/same_residence': 'residence loss', 'recovery/A/SEX': 'A / sex',
            'recovery/AB/SEX': 'AB / sex', 'recovery/A/RAC1P': 'A / race',
            'recovery/AB/RAC1P': 'AB / race'}
SHORT_EP = {'A/SEX': 'A / sex', 'AB/SEX': 'AB / sex', 'A/RAC1P': 'A / race',
            'AB/RAC1P': 'AB / race'}
LABEL = {'spectral_C1': 'C1', 'spectral_L1': 'L1', 'spectral_L2': 'L2',
         'spectral_S0': 'S0', 'spectral_C025': 'C025', 'spectral_M025': 'M025',
         'spectral_M1': 'M1', 'spectral_L025': 'sL025', 'J': 'J', 'H': 'H',
         'E': 'E', 'A0': 'A0', 'L025': 'L025', 'L20': 'L20',
         'spectral_lin16_C1': 'lin16-C1', 'spectral_lin16_L1': 'lin16-L1',
         'spectral_lin16_L2': 'lin16-L2', 'spectral_nlr16_C1': 'nlr16-C1',
         'spectral_nlr16_L1': 'nlr16-L1', 'spectral_nlr16_L2': 'nlr16-L2',
         'spectral_lin8_C1': 'lin8-C1', 'spectral_lin8_L1': 'lin8-L1',
         'spectral_lin8_L2': 'lin8-L2', 'spectral_nlr8_C1': 'nlr8-C1',
         'spectral_nlr8_L1': 'nlr8-L1', 'spectral_nlr8_L2': 'nlr8-L2'}
WSHORT = {'unweighted': 'unw.', 'person_weighted': 'PWGTP'}

PALETTE = {'C1': '#2f6f9f', 'J': '#b8562a', 'L1': '#7a7a7a', 'L2': '#4a4a4a',
           'S0': '#9a6fa8', 'H': '#000000', 'nlr': '#2f6f9f', 'lin': '#b8562a',
           'other': '#b0b0b0'}

MANIFEST: dict[str, dict] = {}
_SEEN: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Repo:
    """Reads evidence files and records every one it touched."""

    def __init__(self, root: Path):
        self.root = root
        self.used: dict[str, str] = {}

    def _record(self, rel: str) -> Path:
        p = self.root / rel
        if rel not in self.used:
            self.used[rel] = sha256(p)
        return p

    def csv(self, rel: str) -> list[dict]:
        p = self._record(rel)
        opener = gzip.open if rel.endswith('.gz') else open
        with opener(p, 'rt', newline='') as fh:
            return list(csv.DictReader(fh))

    def json(self, rel: str):
        p = self._record(rel)
        with open(p) as fh:
            return json.load(fh)


def f(x) -> float:
    return float(x)


def fmt(x, n=4, signed=False) -> str:
    v = float(x)
    s = f'{v:+.{n}f}' if signed else f'{v:.{n}f}'
    return s.replace('-', '$-$')


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def emit(out: Path, name: str, lines: list[str], repo: Repo, command: str):
    path = out / 'tables' / f'{name}.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    MANIFEST[f'tables/{name}.tex'] = {
        'generated_by': command,
        'sources': dict(repo.used),
    }


def save_fig(fig, out: Path, name: str, repo: Repo, command: str):
    d = out / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(d / f'{name}.{ext}')
    plt.close(fig)
    MANIFEST[f'figures/{name}.pdf'] = {
        'generated_by': command,
        'sources': dict(repo.used),
    }


def scoped(repo: Repo) -> Repo:
    """A fresh source-recording view over the same root."""
    return Repo(repo.root)


# ============================================================ derived counters

def family_counts(decision: dict) -> dict:
    """Sensitive-cell counts per family, from the study's frozen decision record.

    The denominator is contrasts x 4 sensitive roles x 2 weightings. The registered
    `advantage` rule is a different object: >=1 strictly better and none strictly worse,
    per weighting. Both are returned so the manuscript can quote the one it means.
    """
    out = {}
    for fam, fd in decision['families'].items():
        better = worse = cells = 0
        adv_cells = 0
        for cell, cd in fd['decisions'].items():
            better += len([e for e in cd['significantly_better'] if e in SENSITIVE])
            worse += len([e for e in cd['significantly_worse'] if e in SENSITIVE])
            cells += len(SENSITIVE)
            adv_cells += 1 if cd['advantage'] else 0
        out[fam] = {
            'sensitive_cells': cells, 'better': better, 'worse': worse,
            'unresolved': cells - better - worse,
            'advantage_cells_fired': adv_cells, 'advantage_cells_total': len(fd['decisions']),
            'critical_value': fd['critical'], 'family_endpoints': fd['endpoints'],
        }
    return out


def paired_counts(rows: list[dict], family: str) -> dict:
    sub = [r for r in rows if r['comparison_family'] == family and r['endpoint'] in SENSITIVE]
    b = sum(r['significantly_better'] == 'True' for r in sub)
    w = sum(r['significantly_worse'] == 'True' for r in sub)
    res = [r for r in rows if r['comparison_family'] == family
           and r['endpoint'] == 'utility/same_residence']
    return {
        'sensitive_cells': len(sub), 'better': b, 'worse': w, 'unresolved': len(sub) - b - w,
        'worse_rows': [f"{r['left']} - {r['right']} / {r['endpoint']} / {r['weight']}"
                       for r in sub if r['significantly_worse'] == 'True'],
        'residence_cells': len(res),
        'residence_better': sum(r['significantly_better'] == 'True' for r in res),
        'residence_worse': sum(r['significantly_worse'] == 'True' for r in res),
    }


# ==================================================================== tables

def table_conditions(repo: Repo, out: Path):
    """The compact condition table: what each interface is, and what it may be compared to."""
    r = scoped(repo)
    fit = r.json(f'{NONLIN}/FIT_SUMMARY.json')
    matrix = r.json(f'{NONLIN}/MATRIX.json')
    spec = r.json(f'{NONLIN}/RANK_SPECTRUM.json')
    alias = fit['historical_alias_map']

    rows = [
        # name, objective, rank, fitting labels, released to A, inference inputs, source year, status
        ('H', 'none (no channel)', '--', 'none', r'$H_A$ only', r'$H_A$', '2018', 'reference'),
        ('E', 'adversarial, frozen', '16', 'sex, race', r'$H_A,Z$', r'$X$', '2018', 'context'),
        ('A0', 'adversarial, frozen', '16', 'sex, race', r'$H_A,Z$', r'$X$', '2018', 'context'),
        ('L025', 'adversarial, frozen', '16', 'sex, race', r'$H_A,Z$', r'$X$', '2018', 'F2 control'),
        ('L20', 'adversarial, frozen', '16', 'sex, race', r'$H_A,Z$', r'$X$', '2018', 'F2 control'),
        ('J', 'adversarial, frozen', '16', 'sex, race', r'$H_A,Z$', r'$X$', '2018', 'neural ref.'),
        ('S0', r'$U$ only ($\lambda=0$)', '16', 'none', r'$H_A,Z$', r'$T,H_A$', '2018', 'context'),
        ('M025 / M1', r'$U-\lambda P_{\mathrm{marg}}$', '16', 'sex, race', r'$H_A,Z$', r'$T,H_A$', '2018', 'context'),
        ('L1 / L2 / L025', r'$U-\lambda P_{\mathrm{loc}}$', '16', 'sex, race, cov.', r'$H_A,Z$', r'$T,H_A$', '2018', 'F1/F4 control'),
        ('C025 / C1', r'$U-\lambda(P_{\mathrm{loc}}{+}P_{AB})$', '16', r'$+$ coal.\ sex, race', r'$H_A,Z$', r'$T,H_A$', '2018', 'candidate'),
        (r'lin16-$\ast$', r'$U-\lambda P$ (alias)', '16', 'as above', r'$H_A,Z$', r'$T,H_A$', '2018', 'alias, not refitted'),
        (r'nlr16-$\ast$', 'nonlinear refinement', '16', 'as above', r'$H_A,Z$', r'$T,H_A$', '2018', 'dev.\\ only'),
        (r'lin8-$\ast$', r'$U-\lambda P$, compressed', '8', 'as above', r'$H_A,Z$', r'$T,H_A$', '2018', 'dev.\\ only'),
        (r'nlr8-$\ast$', 'nonlinear ref., compressed', '8', 'as above', r'$H_A,Z$', r'$T,H_A$', '2018', 'dev.\\ only'),
    ]
    lines = [r'\begin{tabular}{@{}llclll@{}}', r'\toprule',
             r'Interface & Objective & $r$ & Fitting labels & Inf.\ inputs & Role \\',
             r'\midrule']
    years = set()
    for name, obj, rank, lab, _rel, inf, yr, role in rows:
        years.add(yr)
        lines.append(rf'\texttt{{{name}}} & {obj} & {rank} & {lab} & {inf} & {role} \\')
    assert years == {'2018'}, years
    lines += [r'\bottomrule', r'\end{tabular}']
    # sanity: the alias map and the rank branch must agree with what the table asserts
    assert set(alias) == {'spectral_lin16_L1', 'spectral_lin16_L2', 'spectral_lin16_C1'}
    assert spec['all_seeds_r_plus_equals_16'] and matrix['rank_branch_trigger']['label'].startswith('compression')
    emit(out, 'conditions', lines, r, 'make_assets_v2.table_conditions')


def table_primary(repo: Repo, out: Path):
    """F1 and F3 on the locked year, with the corrected sensitive-cell denominator."""
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/INDEPENDENT_FAMILIES.csv')
    dec = r.json(f'{TRANSPORT}/TRANSPORT_DECISION.json')
    counts = family_counts(dec)

    lines = [r'\begin{tabular}{@{}llrrl@{\hspace{3pt}}c@{}}', r'\toprule',
             r'Contrast & Endpoint & Estimate & Seeds 0 / 1 / 2 & Adjusted 95\% & \\',
             r'\midrule']
    for fam, title in (
            ('F1_primary', r'\textit{F1 (registered primary): coalition penalty vs local controls}, '
                           rf'$c={counts["F1_primary"]["critical_value"]:.3f}$ over '
                           rf'{counts["F1_primary"]["family_endpoints"]} endpoints'),
            ('F3_secondary', r'\textit{F3 (registered secondary): coalition channel vs frozen neural channel J}, '
                             rf'$c={counts["F3_secondary"]["critical_value"]:.3f}$ over '
                             rf'{counts["F3_secondary"]["family_endpoints"]} endpoints')):
        lines.append(rf'\multicolumn{{6}}{{@{{}}l}}{{{title}}} \\[1.5pt]')
        for row in [x for x in rows if x['family'] == fam and x['weight'] == 'unweighted']:
            lo, hi = f(row['adjusted_low']), f(row['adjusted_high'])
            mark = r'$\blacktriangledown$' if hi < 0 else (r'$\blacktriangle$' if lo > 0 else '')
            seeds = ' / '.join(f'{f(row[f"seed_{i}"]):+.4f}'.replace('-', '$-$') for i in range(3))
            lines.append(
                rf'\texttt{{{LABEL[row["left"]]}}}$-$\texttt{{{LABEL[row["right"]]}}} & '
                rf'{ENDPOINT[row["endpoint"]]} & {fmt(row["estimate"], 5, signed=True)} & '
                rf'{seeds} & [{fmt(lo, 5, signed=True)}, {fmt(hi, 5, signed=True)}] & {mark} \\')
        lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'primary_comparisons', lines, r, 'make_assets_v2.table_primary')

    # the denominator audit, as its own small table
    r2 = scoped(repo)
    dec = r2.json(f'{TRANSPORT}/TRANSPORT_DECISION.json')
    c = family_counts(dec)
    nice = {'F1_primary': r'F1 primary (C1 vs L1, L2; Mode B)',
            'F2_neural_replication': r'F2 replication (J vs L025, L20; Mode B)',
            'F3_secondary': r'F3 secondary (C1 vs J; Mode B)',
            'F4_frozen_transfer': r'F4 frozen transfer (all four; Mode A)'}
    lines = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
             r'Family & Sens.\ cells & Better & Worse & Unresolved & '
             r'\texttt{advantage} fired & $c$ \\', r'\midrule']
    for fam in ['F1_primary', 'F2_neural_replication', 'F3_secondary', 'F4_frozen_transfer']:
        k = c[fam]
        lines.append(
            rf'{nice[fam]} & {k["sensitive_cells"]} & {k["better"]} & {k["worse"]} & '
            rf'{k["unresolved"]} & {k["advantage_cells_fired"]} / {k["advantage_cells_total"]} & '
            rf'{k["critical_value"]:.3f} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'denominators', lines, r2, 'make_assets_v2.table_primary')
    return counts


def _equivalence_block(rows, fams):
    lines = [r'\begin{tabular}{@{}llrrcrc@{}}', r'\toprule',
             r'Contrast & Weight & Point est. & Pointwise 95\% up. & NI? & '
             r'Simult.\ 95\% up. & NI? \\', r'\midrule']
    for row in rows:
        if row['family'] not in fams:
            continue
        lines.append(
            rf'\texttt{{{row["contrast"].replace("spectral_", "").replace("_", "-")}}} & '
            rf'{WSHORT[row["weight"]]} & {fmt(row["estimate"], 5, signed=True)} & '
            rf'{fmt(row["pointwise_upper_95_percentile"], 5, signed=True)} & '
            rf'{"yes" if row["noninferior_pointwise"] == "True" else r"\textbf{no}"} & '
            rf'{fmt(row["simultaneous_upper_95_studentised"], 5, signed=True)} & '
            rf'{"yes" if row["noninferior_simultaneous"] == "True" else r"\textbf{no}"} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return lines


def table_equivalence(repo: Repo, out: Path):
    """The registered rule is a one-sided POINT rule; these are the interval outcomes."""
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/EQUIVALENCE_F1.csv')
    emit(out, 'equivalence', _equivalence_block(rows, {'F1_primary'}), r,
         'make_assets_v2.table_equivalence')
    r2 = scoped(repo)
    rows2 = r2.csv(f'{REVIEW}/EQUIVALENCE_F1.csv')
    emit(out, 'equivalence_f4', _equivalence_block(rows2, {'F4_frozen_transfer'}), r2,
         'make_assets_v2.table_equivalence')
    summary = {}
    for fam in ('F1_primary', 'F4_frozen_transfer'):
        sub = [x for x in rows if x['family'] == fam]
        summary[fam] = {
            'comparisons': len(sub),
            'registered_point_rule_passed': sum(x['registered_rule_point_le_margin'] == 'True'
                                                for x in sub),
            'noninferior_pointwise': sum(x['noninferior_pointwise'] == 'True' for x in sub),
            'noninferior_simultaneous': sum(x['noninferior_simultaneous'] == 'True' for x in sub),
        }
    return summary


def table_absolute_vs_additional(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/MODEB_TRANSPORT_TABLE.csv')
    sub = [x for x in rows if x['mode'] == 'B' and x['scope'] == 'transport_all'
           and x['budget'] == '360' and x['weight'] == 'unweighted']
    order = ['H', 'J', 'E', 'A0', 'spectral_S0', 'spectral_L1', 'spectral_L2', 'spectral_C1']
    by = {x['interface']: x for x in sub}
    lines = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
             r'& & \multicolumn{2}{c}{A / race} & \multicolumn{2}{c}{AB / sex} & \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'Interface & Residence gain & absolute & additional & absolute & additional & '
             r'\multicolumn{1}{c}{} \\', r'\midrule']
    for name in order:
        if name not in by:
            continue
        x = by[name]
        lines.append(
            rf'\texttt{{{LABEL[name]}}} & {fmt(x["residence_gain_vs_H"], 4)} & '
            rf'{fmt(x["absolute/A/RAC1P"], 4)} & {fmt(x["additional/A/RAC1P"], 4, signed=True)} & '
            rf'{fmt(x["absolute/AB/SEX"], 4)} & {fmt(x["additional/AB/SEX"], 4, signed=True)} & \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'absolute_vs_additional', lines, r, 'make_assets_v2.table_absolute_vs_additional')
    return by


def table_withholding(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/WITHHOLDING_MATCHING.csv')
    sub = [x for x in rows if x['mode'] == 'B' and x['weight'] == 'unweighted'
           and x['target'] == 'spectral_C1']
    lines = [r'\begin{tabular}{@{}lrrrc@{}}', r'\toprule',
             r'Comparator & gain at $p{=}1$ & $p^\star$ & '
             r'A/race at $p^\star$ & reachable? \\', r'\midrule']
    for x in sub:
        pstar = f(x['matching_probability'])
        reach = x['identifiable_in_unit_interval'] == 'True'
        cell = fmt(x['source_additional_at_matched_p/A/RAC1P'], 4) if reach else '--'
        lines.append(
            rf'\texttt{{{LABEL.get(x["withholding_source"], x["withholding_source"])}}} & '
            rf'{fmt(x["source_residence_gain_at_p1"], 4)} & {pstar:.3f} & {cell} & '
            rf'{"yes" if reach else r"\textbf{no}"} \\')
    if sub:
        lines.append(r'\midrule')
        lines.append(rf'\texttt{{C1}} (target) & {fmt(sub[0]["target_residence_gain"], 4)} & -- & '
                     rf'{fmt(sub[0]["target_additional/A/RAC1P"], 4)} & -- \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'withholding_matching', lines, r, 'make_assets_v2.table_withholding')


def table_criteria(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/SERVICE_VS_PROBE.csv')
    q = r.csv(f'{REVIEW}/SERVICE_QUALITY_SUMMARY.csv')
    sub = [x for x in rows if x['weight'] == 'unweighted']
    order = ['H', 'E', 'A0', 'J', 'spectral_S0', 'spectral_L1', 'spectral_L2', 'spectral_C1']
    lines = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{source-probe allowance} & '
             r'\multicolumn{2}{c}{half-headroom residence} \\[1pt]',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Interface & Mode A (frozen) & Mode B (fresh) & Mode A & Mode B \\', r'\midrule']
    idx = {(x['interface'], x['mode']): x for x in sub}
    for name in order:
        a, b = idx.get((name, 'A')), idx.get((name, 'B'))
        if not (a and b):
            continue
        lines.append(
            rf'\texttt{{{LABEL[name]}}} & {a["legacy_source_probe_allowance_pass_seeds"]} / 3 & '
            rf'{b["legacy_source_probe_allowance_pass_seeds"]} / 3 & '
            rf'{a["half_headroom_pass_seeds"]} / 3 & {b["half_headroom_pass_seeds"]} / 3 \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'criteria', lines, r, 'make_assets_v2.table_criteria')
    return {x['service_task']: float(x['change_2017_minus_2018']) for x in q}


def table_attribution(repo: Repo, out: Path):
    """The 2018 development study: each factor separately, with honest denominators."""
    r = scoped(repo)
    pi = r.csv(f'{NONLIN}/PAIRED_INTERVALS.csv')
    fams = {
        'C1_nonlinear_vs_original_rank16': ('Nonlinear penalty', r'$r=16$'),
        'C6_nonlinear_vs_original_rank8': ('Nonlinear penalty', r'$r=8$'),
        'C3_rank8_vs_rank16': ('Rank-8 compression', 'both penalties'),
        'C4_C1_vs_local_control': ('Coalition vs local penalty', 'both ranks'),
    }
    lines = [r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
             r'Factor & Held at & Sens.\ cells & Better & Worse & Unres. & '
             r'Residence better/worse \\', r'\midrule']
    counts = {}
    for fam, (label, held) in fams.items():
        k = paired_counts(pi, fam)
        counts[fam] = k
        lines.append(
            rf'{label} & {held} & {k["sensitive_cells"]} & {k["better"]} & '
            rf'\textbf{{{k["worse"]}}} & {k["unresolved"]} & '
            rf'{k["residence_better"]} / {k["residence_worse"]} '
            rf'(of {k["residence_cells"]}) \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'attribution', lines, r, 'make_assets_v2.table_attribution')

    # the individual significant cells, for the appendix
    r2 = scoped(repo)
    pi = r2.csv(f'{NONLIN}/PAIRED_INTERVALS.csv')
    lines = [r'\begin{tabular}{@{}llllrl@{}}', r'\toprule',
             r'Family & Contrast & Endpoint & Weight & Estimate & Adjusted 95\% \\', r'\midrule']
    for fam in fams:
        for row in pi:
            if row['comparison_family'] != fam or row['endpoint'] not in SENSITIVE:
                continue
            if row['significantly_better'] != 'True' and row['significantly_worse'] != 'True':
                continue
            tag = r'$\blacktriangledown$' if row['significantly_better'] == 'True' else r'$\blacktriangle$'
            lines.append(
                rf'{fam.split("_")[0]} & \texttt{{{LABEL[row["left"]]}}}$-$'
                rf'\texttt{{{LABEL[row["right"]]}}} & {ENDPOINT[row["endpoint"]]} & '
                rf'{WSHORT[row["weight"]]} & {fmt(row["estimate"], 5, signed=True)}~{tag} & '
                rf'[{fmt(row["adjusted_low"], 5, signed=True)}, '
                rf'{fmt(row["adjusted_high"], 5, signed=True)}] \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'attribution_cells', lines, r2, 'make_assets_v2.table_attribution')
    return counts


def table_candidate_vs_J(repo: Repo, out: Path):
    r = scoped(repo)
    pi = r.csv(f'{NONLIN}/PAIRED_INTERVALS.csv')
    sub = [x for x in pi if x['comparison_family'] == 'C5_candidate_vs_reference'
           and x['left'] == 'spectral_nlr8_C1' and x['right'] == 'J']
    lines = [r'\begin{tabular}{@{}lllrl@{\hspace{3pt}}c@{}}', r'\toprule',
             r'Endpoint & Weight & & Estimate & Adjusted 95\% & \\', r'\midrule']
    for x in sub:
        lo, hi = f(x['adjusted_low']), f(x['adjusted_high'])
        mark = r'$\blacktriangledown$' if hi < 0 else (r'$\blacktriangle$' if lo > 0 else '')
        lines.append(
            rf'{ENDPOINT[x["endpoint"]]} & {WSHORT[x["weight"]]} & & '
            rf'{fmt(x["estimate"], 5, signed=True)} & [{fmt(lo, 5, signed=True)}, '
            rf'{fmt(hi, 5, signed=True)}] & {mark} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'candidate_vs_j', lines, r, 'make_assets_v2.table_candidate_vs_J')
    nworse = sum(1 for x in sub if f(x['adjusted_low']) > 0)
    nbetter = sum(1 for x in sub if f(x['adjusted_high']) < 0)
    return {'cells': len(sub), 'worse': nworse, 'better': nbetter}


def table_rank(repo: Repo, out: Path):
    r = scoped(repo)
    spec = r.json(f'{NONLIN}/RANK_SPECTRUM.json')
    lines = [r'\begin{tabular}{@{}ccccccc@{}}', r'\toprule',
             r'Seed & $q$ & $U$: pos / neg / $\approx 0$ & L1 pos & L2 pos & C1 pos & $r_+$ \\',
             r'\midrule']
    checks = {'nonpositive_among_top_16': set(), 'r_plus': set()}
    for s in sorted(spec['per_seed']):
        d = spec['per_seed'][s]
        u = d['utility_matrix_spectrum']
        obj = d['objectives']
        lines.append(
            rf'{s} & {d["whitened_rank"]} & {u["positive_count"]} / {u["negative_count"]} / '
            rf'{u["within_tolerance_count"]} & {obj["L1"]["positive_count"]} & '
            rf'{obj["L2"]["positive_count"]} & {obj["C1"]["positive_count"]} & '
            rf'\textbf{{{d["r_plus"]}}} \\')
        for o in obj.values():
            checks['nonpositive_among_top_16'].add(o['nonpositive_among_top_16'])
        checks['r_plus'].add(d['r_plus'])
    lines += [r'\bottomrule', r'\end{tabular}']
    emit(out, 'rank_spectrum', lines, r, 'make_assets_v2.table_rank')
    return {'r_plus_values': sorted(checks['r_plus']),
            'nonpositive_among_top_16_values': sorted(checks['nonpositive_among_top_16']),
            'alias_of_rank_16': spec['reduced_rank_recipe_is_alias_of_rank_16'],
            'sensitivity_label': spec['sensitivity_label']}


def table_evidence_status(repo: Repo, out: Path):
    r = scoped(repo)
    adm = r.json(f'{REVIEW}/ACS_2016_ADMISSION_MANIFEST.json')
    exp = r.json(f'{NONLIN}/EXPLORATORY_2017.json')
    dev = r.json(f'{NONLIN}/DEVELOPMENT_2018.json')
    lines = [r'\begin{tabular}{@{}lllll@{}}', r'\toprule',
             r'Data pool & Role & Selection saw it? & Intervals? & Used by \\', r'\midrule',
             r'ACS 2018 & development & yes, repeatedly & yes (dev.\ bootstrap) & '
             r'both studies \\',
             r'ACS 2017, original seal & \textbf{confirmatory} & no & '
             r'yes, simultaneous & transport study \\',
             r'ACS 2017, after the seal & exploratory reuse & '
             r'2017 fitting/val.\ only & \textbf{none, by design} & '
             r'nonlinear study \\',
             r'ACS 2016 & admitted, \textbf{unscored} & no & n/a & '
             r'nothing yet \\',
             r'\bottomrule', r'\end{tabular}']
    emit(out, 'evidence_status', lines, r, 'make_assets_v2.table_evidence_status')
    assert exp['evaluation_status'].startswith('EXPLORATORY')
    assert dev['evaluation_status'].startswith('DEVELOPMENT')
    return adm


# =================================================================== figures

def fig_forest(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/INDEPENDENT_FAMILIES.csv')
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    for ax, fam, title in (
            (axes[0], 'F1_primary', 'F1: C1 against its local controls'),
            (axes[1], 'F3_secondary', 'F3: C1 against the frozen neural channel J')):
        sub = [x for x in rows if x['family'] == fam]
        labels, ys = [], []
        for i, x in enumerate(sub):
            lo, hi, est = f(x['adjusted_low']), f(x['adjusted_high']), f(x['estimate'])
            col = PALETTE['C1'] if hi < 0 else ('#b8562a' if lo > 0 else '#9a9a9a')
            y = len(sub) - i
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle='round')
            ax.plot([est], [y], 'o', color=col, ms=3.2)
            labels.append(f'{LABEL[x["left"]]}$-${LABEL[x["right"]]} '
                          f'{ENDPOINT[x["endpoint"]]} ({WSHORT[x["weight"]]})')
            ys.append(y)
        ax.axvline(0, color='k', lw=0.8, ls=(0, (3, 3)))
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=5.6)
        ax.set_title(title, fontsize=8)
    fig.supxlabel('paired difference (nats); left of zero favours the candidate. '
                  'Blue: interval entirely below zero. Orange: entirely above. '
                  'Grey: crosses zero (unresolved, not unaffected).', fontsize=7, y=0.015)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_fig(fig, out, 'forest_primary', r, 'make_assets_v2.fig_forest')


def fig_absolute_vs_additional(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/MODEB_TRANSPORT_TABLE.csv')
    sub = {x['interface']: x for x in rows if x['mode'] == 'B'
           and x['scope'] == 'transport_all' and x['budget'] == '360'
           and x['weight'] == 'unweighted'}
    order = ['H', 'J', 'E', 'A0', 'spectral_S0', 'spectral_L2', 'spectral_L1', 'spectral_C1']
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, role, title in ((axes[0], 'A/RAC1P', 'A / race'),
                            (axes[1], 'AB/SEX', 'AB / sex')):
        names = [n for n in order if n in sub]
        base = np.array([f(sub[n][f'H_baseline/{role}']) for n in names])
        add = np.array([f(sub[n][f'additional/{role}']) for n in names])
        y = np.arange(len(names))[::-1]
        ax.barh(y, base, color='#c8c8c8', height=0.62,
                label='already reachable from $H$ alone')
        ax.barh(y, np.clip(add, 0, None), left=base, color=PALETTE['C1'], height=0.62,
                label='added by the channel')
        neg = np.clip(add, None, 0)
        ax.barh(y, -neg, left=base + neg, color='#e0b7a0', height=0.62,
                hatch='///', edgecolor='#b8562a', linewidth=0.4,
                label='negative increment (selection artifact)')
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[n] for n in names], fontsize=7)
        ax.set_xlabel('recovery (nats), lower is better')
        ax.set_title(title, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_fig(fig, out, 'absolute_vs_additional', r, 'make_assets_v2.fig_absolute_vs_additional')


def _declutter(ax, points, fontsize=5.6):
    """Annotate without overlapping: alternate the offset when labels collide."""
    placed = []
    for x, y, text, col in sorted(points, key=lambda p: (-p[1], p[0])):
        dx, dy = 4.5, 2.0
        for px, py in placed:
            if abs(px - x) < 0.008 and abs(py - y) < 0.0013:
                dy = -7.5 if dy > 0 else 2.0
        ax.annotate(text, (x, y), textcoords='offset points', xytext=(dx, dy),
                    fontsize=fontsize, color=col)
        placed.append((x, y))


def _style_of(cond):
    """One colour scheme across every trade-off panel.

    J is the frozen adversarial reference; coalition-penalty arms are the candidates;
    everything else is a control. Rank 8 is drawn hollow so compression is visible
    without spending a second colour on it.
    """
    if cond == 'J':
        return PALETTE['J'], 'D', 4.6, True
    rank8 = '8_' in cond or cond.endswith('8')
    filled = not rank8
    if cond.endswith('C1') or cond.endswith('C025'):
        return PALETTE['C1'], 'o', 4.8, filled
    if cond in ('spectral_S0', 'E', 'A0'):
        return '#b0b0b0', 's', 3.4, True
    return '#7a7a7a', 'o', 3.6, filled


def _plot_point(ax, x, y, cond):
    col, marker, ms, filled = _style_of(cond)
    ax.plot([x], [y], marker, color=col, ms=ms,
            markerfacecolor=col if filled else 'none', markeredgewidth=1.0)
    return col


def fig_tradeoff(repo: Repo, out: Path):
    """Residence capability against sensitive recovery, both 2017 roles, individual race."""
    r = scoped(repo)
    tr = [x for x in r.csv(f'{REVIEW}/MODEB_TRANSPORT_TABLE.csv')
          if x['scope'] == 'transport_all' and x['budget'] == '360']
    ex = r.csv(f'{NONLIN}/EXPLORATORY_2017.csv')
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)

    # left: the locked 2017 evaluation, historical frozen slate only
    sub = [x for x in tr if x['mode'] == 'B' and x['weight'] == 'unweighted']
    ax, pts = axes[0], []
    for x in sub:
        n = x['interface']
        if n == 'H':
            continue
        gx, gy = f(x['additional/A/RAC1P']), f(x['residence_gain_vs_H'])
        pts.append((gx, gy, LABEL[n], _plot_point(ax, gx, gy, n)))
    _declutter(ax, pts, 6.0)
    ax.set_xlabel('additional A / race recovery (nats)')
    ax.set_ylabel('residence gain over $H$ (nats)')
    ax.set_title('2017 under its original seal: confirmatory\n(14 frozen interfaces, '
                 'simultaneous intervals elsewhere)', fontsize=7.5)

    # right: the same year after the seal was spent, with the new arms
    ax, pts = axes[1], []
    agg = defaultdict(list)
    for x in ex:
        if x['weight'] != 'unweighted' or x['scope'] != 'transport_all':
            continue
        agg[(x['condition'], x['kind'], x['endpoint'])].append(f(x['value']))
    kinds = {k for (_, k, _) in agg}
    ukind = next(k for k in kinds if 'utility' in k or 'gain' in k)
    akind = next(k for k in kinds if 'additional' in k)

    def mean(cond, kind, ep):
        v = agg.get((cond, kind, ep))
        return float(np.mean(v)) if v else None

    for cond in sorted({c for (c, _, _) in agg}):
        gx, gy = mean(cond, akind, 'A/RAC1P'), mean(cond, ukind, 'same_residence')
        if gx is None or gy is None or cond == 'H':
            continue
        pts.append((gx, gy, LABEL.get(cond, cond), _plot_point(ax, gx, gy, cond)))
    _declutter(ax, pts, 5.2)
    ax.set_xlabel('additional A / race recovery (nats)')
    ax.set_title('2017 after the seal was spent: exploratory\n'
                 '(new arms added; no intervals computed)', fontsize=7.5)

    handles = [
        plt.Line2D([], [], color=PALETTE['J'], marker='D', ls='', ms=4.6, label='J (frozen adversarial)'),
        plt.Line2D([], [], color=PALETTE['C1'], marker='o', ls='', ms=4.8, label='coalition penalty, $r=16$'),
        plt.Line2D([], [], color=PALETTE['C1'], marker='o', ls='', ms=4.8,
                   markerfacecolor='none', label='coalition penalty, $r=8$'),
        plt.Line2D([], [], color='#7a7a7a', marker='o', ls='', ms=3.6, label='local penalty'),
        plt.Line2D([], [], color='#b0b0b0', marker='s', ls='', ms=3.4, label='unpenalised / neural context'),
    ]
    axes[0].legend(handles=handles, loc='lower right', frameon=False, fontsize=5.8)
    fig.tight_layout()
    save_fig(fig, out, 'tradeoff', r, 'make_assets_v2.fig_tradeoff')


def fig_attribution(repo: Repo, out: Path):
    """Nonlinear penalty and rank-8 compression, separated by factor and by rank."""
    r = scoped(repo)
    pi = r.csv(f'{NONLIN}/PAIRED_INTERVALS.csv')
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.3), sharex=True)
    panels = [('C1_nonlinear_vs_original_rank16', 'Nonlinear penalty, $r=16$'),
              ('C6_nonlinear_vs_original_rank8', 'Nonlinear penalty, $r=8$'),
              ('C3_rank8_vs_rank16', 'Rank 8 vs rank 16')]
    for ax, (fam, title) in zip(axes, panels):
        sub = [x for x in pi if x['comparison_family'] == fam
               and x['endpoint'] in SENSITIVE and x['weight'] == 'unweighted']
        ys = np.arange(len(sub))[::-1]
        for y, x in zip(ys, sub):
            lo, hi, est = f(x['adjusted_low']), f(x['adjusted_high']), f(x['estimate'])
            col = PALETTE['C1'] if hi < 0 else ('#b8562a' if lo > 0 else '#9a9a9a')
            ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle='round')
            ax.plot([est], [y], 'o', color=col, ms=3)
        ax.axvline(0, color='k', lw=0.8, ls=(0, (3, 3)))
        ax.set_yticks(ys)
        ax.set_yticklabels([f'{LABEL[x["left"]]}: {ENDPOINT[x["endpoint"]]}' for x in sub],
                           fontsize=6.0)
        ax.set_title(title, fontsize=8)
    fig.supxlabel('change in additional recovery, candidate minus comparator (nats)',
                  fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_fig(fig, out, 'attribution', r, 'make_assets_v2.fig_attribution')


def fig_rotation(repo: Repo, out: Path):
    r = scoped(repo)
    diag = r.json(f'{NONLIN}/DIAGNOSTICS_SUMMARY.json')
    style()
    conds, shares, stops = [], [], set()
    for seed in sorted(diag['seeds']):
        for cond, d in diag['seeds'][seed]['rotation'].items():
            if not d.get('applicable'):
                continue
            conds.append((cond, int(seed)))
            shares.append(d['rotation_only_share'])
            stops.add(d.get('rotation_stop_reason', 'unknown'))
    order = sorted(set(c for c, _ in conds), key=lambda c: LABEL.get(c, c))
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for i, cond in enumerate(order):
        vals = [s for (c, _), s in zip(conds, shares) if c == cond]
        ax.plot([i] * len(vals), vals, 'o', color=PALETTE['nlr'], ms=4, alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color='k', lw=1.2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL.get(c, c) for c in order], rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('share of training gain also reached\nby a rotation-only search')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, out, 'rotation', r, 'make_assets_v2.fig_rotation')
    return {'stop_reasons': sorted(stops),
            'mean': float(np.mean(shares)), 'min': float(np.min(shares)),
            'max': float(np.max(shares)), 'n': len(shares)}


def fig_per_seed(repo: Repo, out: Path):
    r = scoped(repo)
    rows = r.csv(f'{REVIEW}/PER_SEED_DIRECTION.csv')
    sub = [x for x in rows if x['family'] == 'F1_primary']
    style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ys = np.arange(len(sub))[::-1]
    for y, x in zip(ys, sub):
        seeds = [f(x[f'seed_{i}']) for i in range(3)]
        m = f(x['seed_mean'])
        agree = x['all_seeds_match_seed_mean_sign'] == 'True'
        col = PALETTE['C1'] if agree else '#b8562a'
        ax.plot(seeds, [y] * 3, 'o', color=col, ms=4.2, alpha=0.85)
        ax.plot([m, m], [y - 0.32, y + 0.32], color='k', lw=1.4)
    ax.axvline(0, color='k', lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{LABEL[x["contrast"].split(" - ")[0]]}$-$'
                        f'{LABEL[x["contrast"].split(" - ")[1]]} '
                        f'{ENDPOINT[x["endpoint"]]} ({WSHORT[x["weight"]]})' for x in sub],
                       fontsize=7.0)
    ax.set_xlabel('paired difference (nats); bar is the seed mean that enters the estimand')
    fig.tight_layout()
    save_fig(fig, out, 'per_seed', r, 'make_assets_v2.fig_per_seed')


def fig_withholding(repo: Repo, out: Path):
    """Post-hoc average-utility matching, including the comparators that cannot reach C1."""
    r = scoped(repo)
    match = r.csv(f'{REVIEW}/WITHHOLDING_MATCHING.csv')
    vec = r.csv(f'{REVIEW}/MODEB_TRANSPORT_TABLE.csv')
    full = {x['interface']: x for x in vec if x['mode'] == 'B' and x['weight'] == 'unweighted'
            and x['scope'] == 'transport_all' and x['budget'] == '360'}
    sub = [x for x in match if x['mode'] == 'B' and x['weight'] == 'unweighted'
           and x['target'] == 'spectral_C1']
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    pts = []
    tgt_gain = f(sub[0]['target_residence_gain'])
    tgt_leak = f(sub[0]['target_additional/A/RAC1P'])
    for x in sub:
        src = x['withholding_source']
        if src not in full:
            continue
        g1 = f(full[src]['residence_gain_vs_H'])
        leak1 = f(full[src]['additional/A/RAC1P'])
        p = f(x['matching_probability'])
        reach = x['identifiable_in_unit_interval'] == 'True'
        col = PALETTE['J'] if src == 'J' else ('#7a7a7a' if reach else '#c07a55')
        ax.plot([0, leak1], [0, g1], '-', color=col, lw=1.3 if src == 'J' else 1.0,
                alpha=0.95 if reach or src == 'J' else 0.8)
        ax.plot([leak1], [g1], 'D' if src == 'J' else 'o', color=col, ms=4.4)
        pts.append((leak1, g1,
                    f'{LABEL.get(src, src)}  $p^\\star={p:.2f}$'
                    + ('' if reach else r' $(>1)$'), col))
    _declutter(ax, pts, 5.8)
    ax.axhline(tgt_gain, color=PALETTE['C1'], lw=1.0, ls=(0, (2, 2)))
    ax.plot([tgt_leak], [tgt_gain], 's', color=PALETTE['C1'], ms=5.5)
    ax.annotate("C1 (target)", (tgt_leak, tgt_gain), textcoords='offset points',
                xytext=(5, 3), fontsize=6.5, color=PALETTE['C1'])
    ax.set_xlabel('expected additional A / race recovery (nats)')
    ax.set_ylabel('expected residence gain over $H$ (nats)')
    ax.set_title(r'Each channel traces a straight line from the origin as $p$ runs 0 to 1;'
                 '\n' r'a line ending below the dashed level cannot reach C1 at any schedule',
                 fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, 'withholding', r, 'make_assets_v2.fig_withholding')
    return {'sources': {x['withholding_source']: {
        'p_star': f(x['matching_probability']),
        'reachable': x['identifiable_in_unit_interval'] == 'True'} for x in sub}}


# ====================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='papers/pcrl_manuscript_v2')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Repo(root)

    facts = {}
    facts['family_counts'] = table_primary(repo, out)
    table_conditions(repo, out)
    facts['equivalence'] = table_equivalence(repo, out)
    table_absolute_vs_additional(repo, out)
    table_withholding(repo, out)
    facts['service_accuracy'] = table_criteria(repo, out)
    facts['attribution'] = table_attribution(repo, out)
    facts['candidate_vs_J'] = table_candidate_vs_J(repo, out)
    facts['rank'] = table_rank(repo, out)
    table_evidence_status(repo, out)

    fig_forest(repo, out)
    fig_absolute_vs_additional(repo, out)
    fig_tradeoff(repo, out)
    fig_attribution(repo, out)
    facts['rotation'] = fig_rotation(repo, out)
    fig_per_seed(repo, out)
    facts['withholding'] = fig_withholding(repo, out)

    (out / 'MANIFEST.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': 'experiments/pcrl_manuscript_v2/make_assets_v2.py',
        'command': 'PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2'
                   ' --repo . --out papers/pcrl_manuscript_v2',
        'assets': MANIFEST,
    }, indent=1) + '\n')
    (out / 'DERIVED_FACTS.json').write_text(json.dumps(facts, indent=1) + '\n')
    print(f'wrote {len(MANIFEST)} assets to {out}')
    for k, v in facts.items():
        print(' ', k, json.dumps(v)[:300])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

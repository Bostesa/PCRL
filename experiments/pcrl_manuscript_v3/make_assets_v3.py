"""Generate the v3 manuscript's NEW tables and figures from commit-pinned evidence.

This generator adds the third completed study -- the rotation-invariant repair plus the
first executed external-method adaptations -- to the assets the v2 generator already
produces.  Run the v2 generator first with ``--out papers/pcrl_manuscript_v3``; this
script then merges its own assets into the same MANIFEST.

Every byte of study-3 evidence is read through ``git cat-file`` at a pinned commit, so
the historical worktree is never touched and the provenance of a printed digit is a
(commit, path, sha256) triple rather than a filesystem location.

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.make_assets_v3 \
        --repo . --out papers/pcrl_manuscript_v3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------------ pinned evidence

STUDY3 = '73903b7f28df68284285f0610a4036beb32b208f'   # research/pcrl-invariant-baselines-v1
STUDY2 = 'c37807e4f568ef38e5528fc09c1506083278bf4d'   # research/pcrl-nonlinear-rank-v1
STUDY1 = '349efa454afd907389760fd1f59fd8806a215efd'   # locked 2017 transport
EVIDENCE = '0d8f4b67b6d4961dfa133289d0167c874d2f4794'  # evidence-paper branch
D3 = 'results/pcrl_invariant_baselines_v1'

# The scope, budget and split in which every study-3 paired interval was computed.
# Located independently by reproducing a published paired estimate from PER_SEED.csv
# (see results/pcrl_manuscript_review_v3/VALIDATION.md, check V3).
SCOPE, BUDGET, SPLIT = 'expanded_catchup', '360', 'test'

SENS = ['recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P']
EPSHORT = {'recovery/A/SEX': 'A / sex', 'recovery/AB/SEX': 'AB / sex',
           'recovery/A/RAC1P': 'A / race', 'recovery/AB/RAC1P': 'AB / race',
           'utility/same_residence': 'residence'}
ARM = {'leace_A0': 'LEACE', 'splince_A0': 'SPLINCE', 'optnet16_C1': 'OptNet-C1',
       'optnet16_L1': 'OptNet-L1', 'optnet16_L2': 'OptNet-L2',
       'spectral_riv16_C1': 'riv16-C1', 'spectral_riv16_L1': 'riv16-L1',
       'spectral_riv16_L2': 'riv16-L2', 'spectral_riv8_C1': 'riv8-C1',
       'spectral_riv8_L1': 'riv8-L1', 'spectral_riv8_L2': 'riv8-L2',
       'spectral_nlr16_C1': 'nlr16-C1', 'spectral_nlr8_C1': 'nlr8-C1',
       'spectral_C1': 'C1', 'J': 'J', 'H': 'H', 'A0': 'A0', 'E': 'E'}

MANIFEST: dict[str, dict] = {}


class Pinned:
    """Reads evidence out of the object store at a fixed commit and hashes it."""

    def __init__(self, root: Path):
        self.root = root
        self.used: dict[str, dict] = {}

    def _blob(self, commit: str, rel: str) -> bytes:
        raw = subprocess.run(['git', 'cat-file', '-p', f'{commit}:{rel}'],
                             cwd=self.root, capture_output=True, check=True).stdout
        key = f'{commit[:12]}:{rel}'
        if key not in self.used:
            self.used[key] = {'commit': commit, 'path': rel,
                              'sha256': hashlib.sha256(raw).hexdigest(),
                              'bytes': len(raw)}
        return raw

    def csv(self, rel: str, commit: str = STUDY3) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self._blob(commit, rel).decode())))

    def json(self, rel: str, commit: str = STUDY3):
        return json.loads(self._blob(commit, rel).decode())

    def text(self, rel: str, commit: str = STUDY3) -> str:
        return self._blob(commit, rel).decode()


def fmt(x, n=4, signed=True) -> str:
    s = f'{float(x):+.{n}f}' if signed else f'{float(x):.{n}f}'
    return s.replace('-', '$-$')


def sci(x, n=1) -> str:
    """LaTeX scientific notation, e.g. 1.4e-15 -> $1.4\\times10^{-15}$."""
    mant, exp = f'{float(x):.{n}e}'.split('e')
    return r'$' + mant + r'\times10^{' + str(int(exp)) + r'}$'


def mark(row) -> str:
    if row['significantly_better'] == 'True':
        return r'$\blacktriangledown$'
    if row['significantly_worse'] == 'True':
        return r'$\blacktriangle$'
    return ''


def save_table(out: Path, name: str, body: str, repo: Pinned, fn: str, note: str = ''):
    """Write the tabular. A long note goes to <name>_note.tex for the LaTeX caption:
    putting it inside the tabular would stretch the table to the note's width and
    inflate whichever column absorbs the slack."""
    d = out / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.tex').write_text(body)
    MANIFEST[f'tables/{name}.tex'] = {'generator': fn, 'sources': dict(repo.used)}
    if note:
        # LaTeX control sequences are letters only, so digits are spelled out.
        digits = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
                  '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
        macro = ''.join(w.capitalize() for w in name.split('_'))
        macro = ''.join(digits.get(c, c) for c in macro if c.isalnum())
        (d / f'{name}_note.tex').write_text(
            '\\newcommand{\\note' + macro + '}{%\n' + note.strip() + '%\n}\n')
        MANIFEST[f'tables/{name}_note.tex'] = {'generator': fn, 'sources': dict(repo.used),
                                               'macro': '\\note' + macro}


def save_fig(fig, out: Path, name: str, repo: Pinned, fn: str):
    d = out / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(d / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    MANIFEST[f'figures/{name}.pdf'] = {'generator': fn, 'sources': dict(repo.used)}


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'figure.dpi': 200, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    })


def pairs(rows, left, right, weight):
    return {r['endpoint']: r for r in rows
            if r['left'] == left and r['right'] == right and r['weight'] == weight}


# ============================================================ study chronology

def table_chronology(repo: Pinned, out: Path):
    """Four studies, what each is evidence for, and which pools it consumed."""
    repo.text(f'{D3}/RESEARCH_DECISION.md')  # record provenance of the verdict column
    rows = [
        (r'1. Locked 2017 transport', STUDY1[:7],
         r'2018 fit $\rightarrow$ \textbf{2017 sealed}',
         r'\textbf{Confirmatory}', 'prespecified rules, no directional forecast',
         r'coalition penalty beats both local controls'),
        (r'2. Nonlinear / rank refinement', STUDY2[:7],
         r'2018 development; 2017 reuse',
         r'Development', '6 forecasts, 3 wrong or ambiguous',
         r'moved recovery, failed 8/8 coordination; objective later found coordinate-dependent'),
        (r'3. Invariant repair + externals', STUDY3[:7],
         r'2018 development; 2017 reuse',
         r'Development', '7 forecasts, 3 wrong',
         r'repair exact; measured disclosure worse; three external arms beat it'),
        (r'4. Direct adversarial refinement', r'\texttt{86f142c4}',
         r'2018 development; 2017 panel',
         r'Development', r'3 questions, 5 weak directional priors',
         r'\emph{fitting in progress; 126 fits registered, no result read}'),
    ]
    b = [r'\begin{tabular}{@{}p{2.5cm}p{1.2cm}p{2.2cm}p{1.9cm}p{2.5cm}p{3.5cm}@{}}', r'\toprule',
         r'Study & Commit & Data pools & Evidence role & Registration & Outcome \\',
         r'\midrule']
    for r in rows:
        b.append(' & '.join([r[0], r'\texttt{' + r[1] + '}' if not r[1].startswith(r'\emph')
                             else r[1], r[2], r[3], r[4], r[5]]) + r' \\')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'ACS 2016 is admitted on '
        r'provenance and schema only and remains \textbf{unscored}: no label, output, '
        r'transform or performance figure for that year has been inspected by any study.')
    save_table(out, 'study_chronology', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_chronology', note=NOTE)


# ============================================== the main comparator table (study 3)

def table_comparator(repo: Pinned, out: Path):
    pi = repo.csv(f'{D3}/PAIRED_INTERVALS.csv')
    crit = repo.csv(f'{D3}/CRITERIA_SUMMARY.csv')
    src = {r['condition']: r['source_allowance_pass_seeds']
           for r in crit if r['weight'] == 'unweighted'}
    gain = {r['condition']: float(r['residence_gain_mean'])
            for r in crit if r['weight'] == 'unweighted'}

    arms = ['leace_A0', 'splince_A0', 'optnet16_C1', 'optnet16_L1', 'optnet16_L2']
    b = [r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule',
         r'& & \multicolumn{4}{c}{additional recovery vs.\ \texttt{riv16-C1}}'
         r' & residence & source \\',
         r'\cmidrule(lr){3-6}',
         r'Arm & wt. & A / sex & AB / sex & A / race & AB / race & loss diff. & allow. \\',
         r'\midrule']
    facts = {}
    for a in arms:
        for w, wl in (('unweighted', 'unw.'), ('person_weighted', 'PWGTP')):
            p = pairs(pi, a, 'spectral_riv16_C1', w)
            cells = [fmt(p[e]['estimate']) + mark(p[e]) for e in SENS]
            res = p['utility/same_residence']
            rescell = fmt(res['estimate']) + mark(res)
            b.append(' & '.join([r'\texttt{' + ARM[a] + '}' if w == 'unweighted' else '',
                                 wl, *cells, rescell,
                                 (src.get(a, '--') + '/3') if w == 'unweighted' else '']) + r' \\')
            if w == 'unweighted':
                facts[a] = {'residence_gain_mean': gain.get(a),
                            'source_allowance_pass_seeds': int(src.get(a, -1)),
                            'residence_diff_vs_riv16C1': float(res['estimate']),
                            'residence_diff_adj_interval': [float(res['adjusted_low']),
                                                            float(res['adjusted_high'])],
                            'residence_diff_significant': res['significantly_worse'] == 'True'}
        b.append(r'\addlinespace[1pt]')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'$\blacktriangledown$ adjusted simultaneous interval entirely below zero (the arm '
        r'recovers \emph{less} than \texttt{riv16-C1}); $\blacktriangle$ entirely above. '
        r'Residence is a \emph{log loss} difference, so a positive value means the arm '
        r'supplies \emph{less} residence capability. ACS 2018 development pools; three seeds. '
        r'The last column counts seeds clearing the legacy service-probe allowance, out of 3 '
        r'(\texttt{riv16-C1} clears 1/3).')
    save_table(out, 'external_vs_repaired', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_comparator', note=NOTE)
    return facts


def table_vs_J(repo: Pinned, out: Path):
    pi = repo.csv(f'{D3}/PAIRED_INTERVALS.csv')
    arms = ['leace_A0', 'splince_A0', 'optnet16_C1', 'optnet16_L1', 'optnet16_L2',
            'spectral_riv16_C1', 'spectral_riv8_C1']
    b = [r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
         r'Arm $-$ $J$ & wt. & A / sex & AB / sex & A / race & AB / race & residence \\',
         r'\midrule']
    facts = {}
    for a in arms:
        for w, wl in (('unweighted', 'unw.'), ('person_weighted', 'PWGTP')):
            p = pairs(pi, a, 'J', w)
            cells = [fmt(p[e]['estimate']) + mark(p[e]) for e in SENS]
            res = p['utility/same_residence']
            b.append(' & '.join([r'\texttt{' + ARM[a] + '}' if w == 'unweighted' else '',
                                 wl, *cells, fmt(res['estimate']) + mark(res)]) + r' \\')
            facts.setdefault(a, {})[w] = {
                'n_worse': sum(1 for e in SENS if p[e]['significantly_worse'] == 'True'),
                'n_better': sum(1 for e in SENS if p[e]['significantly_better'] == 'True'),
                'n_unresolved': sum(1 for e in SENS
                                    if p[e]['significantly_worse'] == 'False'
                                    and p[e]['significantly_better'] == 'False'),
                'residence_estimate': float(res['estimate']),
                'residence_significant': (res['significantly_worse'] == 'True'
                                          or res['significantly_better'] == 'True')}
        b.append(r'\addlinespace[1pt]')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'A blank cell is an \textbf{unresolved} difference at three seeds, not a demonstrated '
        r'equality: the adjusted interval contains zero and also contains effects of the size this '
        r'table reports elsewhere. No equivalence or non-inferiority test was run for these cells. '
        r'Negative residence means the arm supplies \emph{more} residence capability than $J$.')
    save_table(out, 'external_vs_J', '\n'.join(b) + '\n', repo, 'make_assets_v3.table_vs_J',
               note=NOTE)
    return facts


# ============================================ repair: objective vs. empirical protection

def table_repair(repo: Pinned, out: Path):
    gate = repo.json(f'{D3}/MECHANISM_GATE.json')
    pi = repo.csv(f'{D3}/PAIRED_INTERVALS.csv')
    rep = [r for r in pi if r['comparison_family'] == 'repair_vs_defective']
    sens = [r for r in rep if r['endpoint'] in SENS]
    res = [r for r in rep if r['endpoint'] == 'utility/same_residence']
    worse = [r for r in sens if r['significantly_worse'] == 'True']
    better = [r for r in sens if r['significantly_better'] == 'True']
    res_sig = [r for r in res if r['significantly_worse'] == 'True'
               or r['significantly_better'] == 'True']
    largest = max(sens, key=lambda r: float(r['estimate']))

    b = [r'\begin{tabular}{@{}p{6.4cm}p{4.2cm}p{4.4cm}@{}}', r'\toprule',
         r'Quantity & Defective objective (Study 2) & Repaired objective (Study 3) \\',
         r'\midrule',
         r'Rotation-only share of training gain, attained '
         r'(\emph{a property of the objective}) & mean $0.628$, range $0.367$--$0.907$ '
         r'over 18 cells & max.\ abs.\ ' + sci(gate['gate']['max_abs_share'])
         + r' over ' + str(gate['gate']['cells']) + r' cells \\',
         r'Direct invariance identity $|L(WQ)-L(W)|$ on real data & up to '
         r'$7.5\times10^{-3}$ & ' + sci(gate['max_direct_invariance_change']) + r' \\',
         r'\addlinespace[2pt]',
         r'\multicolumn{3}{@{}l}{\emph{Measured disclosure, repaired minus defective, over 6 '
         r'matched contrasts $\times$ 4 sensitive endpoints $\times$ 2 weightings:}} \\',
         r'Sensitive cells significantly \textbf{worse} (more recovery) & \multicolumn{2}{l}{'
         + f'{len(worse)} of {len(sens)}' + r'} \\',
         r'Sensitive cells significantly \textbf{better} (less recovery) & \multicolumn{2}{l}{'
         + f'{len(better)} of {len(sens)}' + r'} \\',
         r'Largest single increase in recovery & \multicolumn{2}{l}{'
         + fmt(largest['estimate']) + r' ' + EPSHORT[largest['endpoint']] + r', '
         + ARM[largest['left']] + r'} \\',
         r'Residence cells resolved in either direction & \multicolumn{2}{l}{'
         + f'{len(res_sig)} of {len(res)}' + r'} \\',
         r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'The upper rows are a diagnostic of the \emph{objective}; the lower block is a '
        r'measurement of the \emph{release}. They are reported separately because an exactly '
        r'invariant objective did not produce less measured disclosure. The attained rotation '
        r'share is not a supremum -- every rotation search stopped on budget exhaustion or a '
        r'failed line search -- and it is not a causal decomposition of what the optimiser did.')
    save_table(out, 'repair_diagnostic', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_repair', note=NOTE)
    return {'largest_increase': {'estimate': float(largest['estimate']),
                                 'endpoint': largest['endpoint'],
                                 'contrast': f"{largest['left']}-{largest['right']}",
                                 'weight': largest['weight']},
            'sensitive_cells': len(sens), 'residence_cells': len(res),
            'residence_cells_resolved': len(res_sig),
            'gate_max_abs_share': gate['gate']['max_abs_share'],
            'gate_cells': gate['gate']['cells'],
            'direct_invariance_change': gate['max_direct_invariance_change'],
            'repair_vs_defective_significantly_worse': len(worse),
            'repair_vs_defective_significantly_better': len(better),
            'repair_vs_defective_cells': len(sens),
            'worse_cells': [f"{r['left']}-{r['right']} {r['weight']} {r['endpoint']}"
                            for r in worse]}


# ==================================================== erasure rank and class support

def table_erasure_rank(repo: Pinned, out: Path):
    eb = repo.json(f'{D3}/ERASURE_BASELINES.json')
    b = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
         r'Seed & directions deleted & realised width & race class 3 support &'
         r' cross-cov.\ after & idempotency & mean pres. \\', r'\midrule']
    facts = {}
    seeds = eb['seeds']
    for s in sorted(seeds):
        le = seeds[s]['leace_A0']
        kz = le['expected_rank_loss']
        rank = le['realised_projection_rank']
        sup = le['coverage']['RAC1P']['support_complete_cases'][3]
        after = le['cross_covariance_max_abs_after']
        idem = le['idempotent_max_abs_error']
        mp = le['mean_preservation_max_abs_error']
        b.append(' & '.join([
            s, str(kz), str(rank), str(sup),
            sci(after),
            sci(idem),
            sci(mp)]) + r' \\')
        facts[s] = {'directions_deleted': kz, 'realised_width': rank,
                    'race_class3_support_in_fitting_pool': sup,
                    'channel_width': le['channel_width']}
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'Linear erasure deletes exactly '
        r'$\mathrm{rank}(\Sigma_{XZ})$ directions. A centred joint one-hot over sex (2), race '
        r'(9) and coverage (2) has rank $1+8+1=10$, leaving 6 of 16. \textbf{Seed 2 differs '
        r'because one race category has no support in \emph{that seed\textquotesingle s own fitting pool}}, '
        r'so the centred one-hot has rank 9 and the channel keeps 7 dimensions. Support counts '
        r'refer to fitting pools, not to the population: a category missing from one pool is '
        r'not absent from California.')
    save_table(out, 'erasure_rank', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_erasure_rank', note=NOTE)
    return facts


# ======================================================== release-view schematic

def fig_release_view(repo: Pinned, out: Path):
    style()
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off'); ax.grid(False)

    def box(x, y, w, h, label, fc, ec, fs=7.5, tc='black'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                    linewidth=0.9, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fs, color=tc)

    def arrow(x0, y0, x1, y1, style='-|>', ls='-', col='#555555'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=8, linewidth=0.8,
                                     linestyle=ls, color=col, shrinkA=1, shrinkB=1))

    box(0.1, 1.35, 1.25, 0.7, '$X$\ncovariates', '#f4f4f4', '#999999')
    box(1.95, 2.25, 1.75, 0.66, r'$H_A(X)\in[0,1]^4$', '#e9e3d5', '#8a7a55')
    box(1.95, 1.35, 1.75, 0.66, r'$H_B(X)\in[0,1]^2$', '#e9e3d5', '#8a7a55')
    box(1.95, 0.30, 1.75, 0.66, '$Z \\in \\mathsf{R}^{16}$', '#d8e6f2', '#2f6f9f')
    ax.text(2.82, 3.06, r'\textbf{immutable, already published}' if False
            else 'immutable, already published', ha='center', fontsize=6.6,
            color='#8a7a55')
    ax.text(2.82, 0.08, 'appended, never mixed in', ha='center', fontsize=6.6,
            color='#2f6f9f')

    for y0, y1 in ((2.58, 2.58), (1.68, 1.68), (0.63, 0.63)):
        arrow(1.35, 1.7, 1.95, y1)

    box(4.35, 2.05, 1.5, 0.86, 'recipient A\n$(H_A, Z)$', '#ffffff', '#333333')
    box(4.35, 0.55, 1.5, 0.86, 'recipient B\n$(H_B)$', '#ffffff', '#333333')
    arrow(3.70, 2.58, 4.35, 2.55); arrow(3.70, 0.63, 4.35, 1.05)
    arrow(3.70, 1.68, 4.35, 0.98)

    box(6.60, 2.05, 1.5, 0.86, 'audit: A alone\n$(H_A,Z)$', '#f7f2ea', '#b8562a')
    box(6.60, 0.55, 1.5, 0.86, 'audit: AB\n$(H_A,Z,H_B)$', '#f7f2ea', '#b8562a')
    arrow(5.85, 2.48, 6.60, 2.48)
    arrow(5.85, 2.20, 6.60, 1.25); arrow(5.85, 0.98, 6.60, 0.98)

    box(8.55, 1.30, 1.35, 0.86, 'baseline\naudit: $H$ only', '#f0f0f0', '#666666')
    ax.text(9.22, 1.12, 'everything is measured\nrelative to this', ha='center',
            fontsize=6.4, color='#666666', va='top')

    ax.text(5.0, 3.25, 'Only recipient A receives a learned channel. '
                       'Nothing here validates arbitrary coalitions.',
            ha='center', fontsize=6.8, style='italic', color='#444444')
    save_fig(fig, out, 'release_view', repo, 'make_assets_v3.fig_release_view')


# ============================= utility against recovery, per sensitive endpoint

def fig_utility_recovery(repo: Pinned, out: Path):
    rows = repo.csv(f'{D3}/PER_SEED.csv')
    crit = repo.csv(f'{D3}/CRITERIA_SUMMARY.csv')
    srcpass = {r['condition']: int(r['source_allowance_pass_seeds'])
               for r in crit if r['weight'] == 'unweighted'}

    want = ['H', 'J', 'A0', 'E', 'leace_A0', 'splince_A0', 'optnet16_C1',
            'spectral_C1', 'spectral_nlr16_C1', 'spectral_riv16_C1', 'spectral_riv8_C1']
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r['scope'], r['budget'], r['split'], r['weight']) != \
                (SCOPE, BUDGET, SPLIT, 'unweighted'):
            continue
        if r['condition'] not in want:
            continue
        if r['kind'] == 'additional_recovery' and f"recovery/{r['endpoint']}" in SENS:
            agg[r['condition']][f"recovery/{r['endpoint']}"].append(float(r['value']))
        elif r['kind'] == 'utility_gain_vs_H' and r['endpoint'] == 'same_residence':
            agg[r['condition']]['gain'].append(float(r['value']))
    mean = {c: {k: sum(v) / len(v) for k, v in d.items()} for c, d in agg.items()}

    style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    # Per-point text labels collided in the dense spectral cluster, so the arms are
    # identified by a shared legend and the markers carry the distinction instead.
    spec = {
        'H':                 ('#000000', 'o'),
        'J':                 ('#b8562a', 'D'),
        'A0':                ('#444444', 's'),
        'E':                 ('#888888', 's'),
        'leace_A0':          ('#2f7f4f', '^'),
        'splince_A0':        ('#5fa37a', 'v'),
        'optnet16_C1':       ('#7a5ba6', 'P'),
        'spectral_C1':       ('#b0b0b0', 'o'),
        'spectral_nlr16_C1': ('#9fc4dd', 'o'),
        'spectral_riv16_C1': ('#2f6f9f', 'o'),
        'spectral_riv8_C1':  ('#5f9fc9', 'o'),
    }
    handles = {}
    for ax, ep in zip(axes, SENS):
        for c in want:
            if c not in mean or ep not in mean[c] or 'gain' not in mean[c]:
                continue
            col, mk = spec.get(c, ('#999999', 'o'))
            filled = srcpass.get(c, 0) == 3
            h, = ax.plot([mean[c][ep]], [mean[c]['gain']], mk, ms=4.8, color=col,
                         markerfacecolor=col if filled else 'white', markeredgewidth=1.0,
                         linestyle='none')
            handles.setdefault(c, h)
        ax.set_title(EPSHORT[ep], fontsize=7.5)
        ax.set_xlabel('additional recovery (nats)', fontsize=6.8)
        ax.margins(x=0.14, y=0.16)
    axes[0].set_ylabel('residence gain over $H$\n(nats, higher better)', fontsize=6.8)
    order = ['H', 'J', 'leace_A0', 'splince_A0', 'optnet16_C1', 'spectral_C1',
             'spectral_nlr16_C1', 'spectral_riv16_C1', 'spectral_riv8_C1', 'E', 'A0']
    fig.legend([handles[c] for c in order if c in handles],
               [ARM.get(c, c) for c in order if c in handles],
               loc='lower center', ncol=6, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.17), handletextpad=0.4, columnspacing=1.2)
    fig.suptitle('One panel per sensitive endpoint; no score is averaged across them. '
                 'Filled markers clear the service-probe allowance in 3/3 seeds, hollow do not.',
                 fontsize=6.6, y=1.04)
    fig.tight_layout()
    save_fig(fig, out, 'utility_recovery_v3', repo, 'make_assets_v3.fig_utility_recovery')
    return {c: mean[c] for c in mean}


# ================================== absolute vs additional, and the scope dependence

def table_scope(repo: Pinned, out: Path):
    """The subtracted H baseline depends on the attack scope; paired contrasts do not."""
    rows = repo.csv(f'{D3}/PER_SEED.csv')
    conds = ['H', 'J', 'leace_A0', 'spectral_riv16_C1', 'A0']
    scopes = ['standard_independent', 'expanded_catchup']
    agg = defaultdict(list)
    for r in rows:
        if (r['budget'], r['split'], r['weight'], r['endpoint']) != \
                (BUDGET, SPLIT, 'unweighted', 'A/SEX'):
            continue
        if r['condition'] in conds and r['scope'] in scopes and \
                r['kind'] in ('absolute_recovery', 'additional_recovery'):
            agg[(r['condition'], r['scope'], r['kind'])].append(float(r['value']))
    m = {k: sum(v) / len(v) for k, v in agg.items()}

    b = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
         r'& \multicolumn{2}{c}{fresh independent attacks}'
         r' & \multicolumn{2}{c}{$+$ historical catch-up (reporting scope)} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Condition & absolute & additional & absolute & additional \\', r'\midrule']
    for c in conds:
        b.append(' & '.join([
            r'\texttt{' + ARM.get(c, c) + '}',
            fmt(m[(c, 'standard_independent', 'absolute_recovery')], signed=False),
            fmt(m[(c, 'standard_independent', 'additional_recovery')]),
            fmt(m[(c, 'expanded_catchup', 'absolute_recovery')], signed=False),
            fmt(m[(c, 'expanded_catchup', 'additional_recovery')])]) + r' \\')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'A / sex, 2018, unweighted, '
        r'seed means. The reporting scope gives $H$ and the frozen historical interfaces their '
        r'saved-observer catch-up attacks; the new arms have none, so their \emph{absolute} '
        r'recovery is unchanged between scopes and only the \emph{subtracted} $H$ baseline '
        r'moves ($0.0014\rightarrow0.0071$). Every paired contrast between two arms is '
        r'therefore identical on both scales, because the shared baseline cancels; only '
        r'$H$-relative levels and ratios are scope-dependent. The absolute recovery of $J$ '
        r'\emph{falls} when it is given more attack candidates -- a validation-selected '
        r'maximum is a floor on leakage, never a bound.')
    save_table(out, 'scope_decomposition', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_scope', note=NOTE)
    return {f'{c}|{s}|{k}': m[(c, s, k)] for (c, s, k) in m}


# ================================================ limitations / claim-support table

def table_claim_support(repo: Pinned, out: Path):
    rows = [
        (r'$H$ preserved exactly in every release',
         r'Structural identity', r'Verified bitwise, 210/210 views',
         r'Says nothing about service \emph{accuracy}, which moved by up to $0.015$ nats'),
        (r'Appended channel adds authorised capability',
         r'Empirical, development \& sealed', r'$0.017$--$0.032$ nats residence gain over $H$',
         r'Half-headroom reference fails in 1--3 of 3 seeds for every arm'),
        (r'Coalition penalty beats both local controls',
         r'\textbf{Confirmatory} (2017 sealed)', r'14/16 cells better, 0 worse, 2 unresolved',
         r'One year, one state, three seeds; residence cost not bounded below $0.001$'),
        (r'Developed mechanism vs.\ frozen neural channel',
         r'Empirical, development', r'Worse on 3 of 4 sensitive endpoints',
         r'$J$ is internal, and it supplies \emph{less} residence capability: neither dominates'),
        (r'Rotation defect repaired exactly',
         r'Elementary algebra $+$ numerics', r'Invariance to $10^{-15}$, 18/18 cells',
         r'Invariance of an objective is not evidence the objective measures the right thing'),
        (r'Repair reduces measured disclosure',
         r'\textbf{Refuted}', r'4 cells significantly worse, 0 better',
         r'Attributable to the repaired \emph{package}, not to any single changed ingredient'),
        (r'External adaptations beat the developed mechanism',
         r'Empirical, development', r'LEACE / SPLINCE / OptNet-C1 lower on all four endpoints',
         r'Adaptations, not replicas; LEACE costs residence against its own source channel'),
        (r'Any bound on $I(S;Z\mid H)$',
         r'\textbf{Not claimed}', r'--',
         r'Finite validation-selected attacks attain no infimum in either direction'),
    ]
    b = [r'\begin{tabular}{@{}p{4.0cm}p{2.5cm}p{3.7cm}p{4.9cm}@{}}', r'\toprule',
         r'Claim & Status & Supporting measurement & What it does not establish \\',
         r'\midrule']
    for r in rows:
        b.append(' & '.join(r) + r' \\' + '\n' + r'\addlinespace[1.5pt]')
    b += [r'\bottomrule', r'\end{tabular}']
    save_table(out, 'claim_support', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_claim_support')


# =============================================================== pending study 4

def table_pending(repo: Pinned, out: Path):
    b = [r'\begin{tabular}{@{}p{4.6cm}p{3.0cm}p{3.0cm}p{4.4cm}@{}}', r'\toprule',
         r'Reported quantity & Value & Interval & Precondition checked before integration \\',
         r'\midrule']
    pend = [
        (r'Adversarially refined arm $-$ \texttt{riv16-C1}, four sensitive endpoints',
         r'\emph{pending}', r'\emph{pending}',
         r'no residence/commute label in representation selection'),
        (r'Adversarially refined arm $-$ $J$, four sensitive endpoints',
         r'\emph{pending}', r'\emph{pending}',
         r'attacker view equals the recipient view; $H_B$ not in the release to $A$'),
        (r'Coalition vs.\ local controls, coordination cells',
         r'\emph{pending}', r'\emph{pending}',
         r'candidate-selection multiplicity covers the whole search'),
        (r'Residence loss vs.\ $J$ and vs.\ \texttt{riv16-C1}',
         r'\emph{pending}', r'\emph{pending}',
         r'training attackers distinct from fresh auditors'),
        (r'Penalty-strength and width ablations',
         r'\emph{pending}', r'\emph{pending}',
         r'optimiser repeats nested in three fixed anchors, not extra seeds'),
        (r'Erasure and single-attacker controls',
         r'\emph{pending}', r'\emph{pending}',
         r'executed cell count read from the committed matrix, not the plan'),
    ]
    for r in pend:
        b.append(' & '.join(r) + r' \\')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'Cells are deliberately empty. They are filled only from a committed artifact whose '
        r'hash and executed-unit count have been checked, and no forecast is entered in the '
        r'meantime. The study registered its protocol, method and 126-fit matrix at '
        r'\texttt{86f142c4} while this revision was being written; all ten preconditions above '
        r'were checked against that registration and are satisfied, with one watch item '
        r'(\texttt{PENDING\_EXPERIMENT\_INTEGRATION.md} P6). No result had been read.')
    save_table(out, 'pending_study4', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_pending', note=NOTE)


# ==================================== related-work comparison (authored, not evidence-derived)

def table_related(repo: Pinned, out: Path):
    """The five-feature comparison. Entries come from RELATED_WORK_COMPARISON.md, which
    records for each source whether it was read in full text or only as an abstract.
    Anything unsettled by what was read is printed as UNVERIFIED rather than guessed."""
    Y, N, U = r'\checkmark', r'--', r'\textsc{unv.}'
    rows = [
        (r'\textbf{This paper}', Y, Y, Y, Y, Y),
        (r'Madras et al.\ 2018 (LAFTR)', N, Y, N, N, r'part.'),
        (r'LEACE \citep{belrose2023leace}', N, N, N, N, N),
        (r'SPLINCE \citep{holstege2025splince}', N, N, N, N, N),
        (r'SARL \citep{sadeghi2019sarl}', N, N, N, N, r'part.'),
        (r'OptNet-ARL \citep{sadeghi2021optnetarl}', N, N, N, N, r'part.'),
        (r'K-TOpt \citep{sadeghi2022ktopt}', N, N, N, N, r'part.'),
        (r'U-FaTE \citep{dehdashtian2024ufate}', N, N, N, N, Y),
        (r'Sankar et al.\ \citeyearpar{sankar2013utility}', N, U, U, U, N),
        (r'Stadler et al.\ 2024', N, N, N, N, r'n/a'),
        (r'Elazar \& Goldberg 2018', N, N, N, N, Y),
    ]
    b = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
         r'& fixed prior & reusable & recipient & coalition-aware & independent \\',
         r'Work & output & extra features & permissions & training & attack eval. \\',
         r'\midrule']
    for r in rows:
        b.append(' & '.join(r) + r' \\')
    b += [r'\bottomrule', r'\end{tabular}']
    NOTE = (
        r'\checkmark\ the text demonstrates it; -- its setting excludes it; '
        r'\textquotedblleft part.\textquotedblright\ a partial form (an adversarial '
        r'objective or a transfer test, not an '
        r'independent attacker slate); \textsc{unv.}\ could not be settled from what was '
        r'read and is \textbf{not} guessed from a title or abstract. '
        r'Several of these methods can be \emph{adapted} into this setting; three were, and '
        r'they beat the mechanism developed here. Provenance for every row, including which '
        r'sources were read in full text and which only as abstracts, is in '
        r'\texttt{RELATED\_WORK\_COMPARISON.md}.')
    save_table(out, 'related_work', '\n'.join(b) + '\n', repo,
               'make_assets_v3.table_related', note=NOTE)


# ====================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='papers/pcrl_manuscript_v3')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Pinned(root)

    facts = {'reporting_scope': {'scope': SCOPE, 'budget': BUDGET, 'split': SPLIT,
                                 'located_by': 'independent reproduction of a published '
                                               'paired estimate from PER_SEED.csv'}}
    table_chronology(repo, out)
    table_related(repo, out)
    facts['external_vs_repaired'] = table_comparator(repo, out)
    facts['vs_J'] = table_vs_J(repo, out)
    facts['repair'] = table_repair(repo, out)
    facts['erasure_rank'] = table_erasure_rank(repo, out)
    facts['scope_decomposition'] = table_scope(repo, out)
    table_claim_support(repo, out)
    table_pending(repo, out)
    fig_release_view(repo, out)
    facts['utility_recovery'] = fig_utility_recovery(repo, out)

    # merge into the manifest the v2 generator already wrote
    mpath = out / 'MANIFEST.json'
    base = json.loads(mpath.read_text()) if mpath.exists() else {'assets': {}}
    base.setdefault('assets', {}).update(MANIFEST)
    base['v3_generated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    base['v3_generator'] = 'experiments/pcrl_manuscript_v3/make_assets_v3.py'
    base['v3_pinned_commits'] = {'study1_locked_2017': STUDY1, 'study2_nonlinear_rank': STUDY2,
                                 'study3_invariant_baselines': STUDY3,
                                 'evidence_branch': EVIDENCE}
    base['v3_pinned_sources'] = repo.used
    mpath.write_text(json.dumps(base, indent=1) + '\n')

    (out / 'DERIVED_FACTS_V3.json').write_text(json.dumps(facts, indent=1) + '\n')
    print(f'wrote {len(MANIFEST)} v3 assets to {out}')
    print(f'pinned {len(repo.used)} evidence blobs at {STUDY3[:12]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

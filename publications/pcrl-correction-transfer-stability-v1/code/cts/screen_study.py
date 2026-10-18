"""Screen operating-characteristic study (question C) of the PCRL correction-transfer campaign
(`pcrl_correction_transfer_stability_v1`; reader investigator; frozen and run BEFORE any new method candidate is scored with
readers; EXPLORATORY DEVELOPMENT evidence on adaptively used MODEL_SELECT records, never confirmation).

Public-use ACS 2019 1-Year PUMS Florida person records. Development readers estimate the recorded SEX column (sex 0 = men,
sex 1 = women) from released outputs only: each recipient's float64 probability vector and argmax decision (income view,
occupation view) and the linked pair. The question is how two recipient screens behave, not whether any release protects anyone:
  OLD   the source's per-cell recipient gate (cts.rules.per_cell_gate): in every one of the 8 registered cells, candidate
        single-output SCORE AUC minus comparator's <= +0.010, income and occupation views, every required comparator
        (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ);
  AGG1  the statistical reviewer's predeclared aggregate paired rule (cts.rules.aggregate_paired_screen; RULES.json
        screens.aggregate_paired_rule; RULES.md section 10): U = dbar + z sqrt(phi v_samp) <= 0.010 per (candidate, comparator,
        view), phi = max(1, phi_prior(view), phi_emp) with the registered gross phi_prior, exact aliases pass by construction; a
        configuration passes iff every required comparator and both views pass.
Both screens and law L1's cell generator are imported from cts.rules; nothing here re-defines them.

Laws (RULES.md section 10, copied; simulated or empirical pass rates under these laws only, never universal error rates):
  L0a  exact alias under shared fitting randomness (frozen source grid, label-free): every alias pair of frozen releases (an
       H3-anchored release's income view against SUCC; candidate income views against each other; shared occupation wires).
  L0b  identical release under separately fitted readers (MODEL_SELECT SEX via the logged access layer, operation
       'screen_study'): the same frozen release is candidate and comparator; the candidate's readers are refitted with new reader
       seeds (k2, k3 instead of k0, k1; also k4, k5 as extra replicates) on the same FIT / VALIDATION / SCORE splits, no cache
       sharing; releases SUCC, H_LJ, NEURAL_RAW, the witness A1_PO_16@H3 and the source protection output P2A_PAIR_B80@H3;
       income, occupation and pair views. True difference 0; noise = reader fitting and winner selection only.
  L1   cell-level synthetic law (exact truth; primary; cts.rules.synthetic_cells / oc_rates): d_{s,R,k} = mu + A_{s,R} +
       B_{s,R,k}; per view the frozen-grid components (sd_A = NET split SD, sd_k = reader-seed SD, se_cell = rms cell SE);
       rho_seed in {0, 0.76}; mu in {-0.030, -0.020, -0.010, 0, +0.005, +0.010, +0.020}; n_R in {2, 3}; 20,000 simulations
       per setting. Multiplicity version: release-level effects (the candidate's shared across its comparisons; per-release SD
       = sd_A / sqrt 2 and sd_k / sqrt 2), 6 required comparators x 2 views (income, occupation) x 2 seeds: P(pass all) at mu = 0
       and mu = -0.010 everywhere, and P(pass all) while one comparison has mu = +0.010 or +0.020 (the IUT error), the others at
       0 or -0.010; n_R = 3 as well. Investigator addition (labelled): the same with the required comparators' ACTUAL wire
       structure for an H3-anchored candidate (SUCC an exact income alias; 5 comparators share one occupation wire).
  L2   release-level semi-synthetic law (realism check; truth approximate; SEX via operation 'screen_study'): base view p =
       the occupation view of SUCC (= H_LJ's, the same JOINT wire bitwise, so done once), of NEURAL_RAW, the income view of
       H_LJ, and (investigator addition) the occupation view of the witness A1_PO_16@H3; candidate log p' = log p + eta (2 SEX
       - 1) v, renormalized, v = e_1 - (1/K) 1 (first class against the rest, centred-logit space; MODEL_SELECT people only, this
       simulation only); eta by bisection so that the fixed reader LOG_LR (fit on R1 FIT, oriented on R1 VALIDATION, scored on R1
       SCORE, reader seed k0) gains Delta_AUC in {0.005, 0.010, 0.020}; then the registered grid (both repetitions, both reader
       seeds, all 16 recipes with winner selection) for both screens, next to the slate's realized mean shift and the fixed-recipe
       shift over 10 extra household splits (tags CTS1_screen_L2_split<i>).

Outputs: results/pcrl_correction_transfer_stability_v1/screen/{SCREEN_STUDY.json, SCREEN_STUDY.md, SCREEN_TABLE.csv,
SCREEN_OC.csv}; private intermediates under .private/<campaign>/runs/SCREEN/. A screen pass is permission for further
development, not evidence of protection; the source label NO_QUALIFYING_CONFIGURATION is never rewritten.

    python3 -m cts.screen_study selftest       invented data only (no real data, no label)
    python3 -m cts.screen_study source         part A: L0a, source configurations, what-if, distinct wires (label-free)
    python3 -m cts.screen_study simulate       L1 and its multiplicity version (no data)
    python3 -m cts.screen_study l0b            L0b refits (labels; through cts.jobqueue with the locked interpreter)
    python3 -m cts.screen_study l2             L2 semi-synthetic law (labels; through cts.jobqueue with the locked interpreter)
    python3 -m cts.screen_study report         merge into the results files
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from task_conditioned_representation.inference import auc
from . import common as C
from . import devreaders as DR
from . import rules as RU

SCHEMA = 'CTS1-screen-study-v1'
OUT = C.RESULTS / 'screen'
PRIV_OUT = C.RUNS / 'SCREEN'
SEEDS = (0, 1)
REPS = ('R1', 'R2')
FULL_CELLS = tuple(f'S{s}R{r}k{k}' for s in SEEDS for r in (1, 2) for k in (0, 1))
VIEWS = ('income', 'occupation')
ALL_VIEWS = ('income', 'occupation', 'pair')
REQUIRED = RU.REQUIRED
ALLOWANCE = RU.RECIPIENT_ALLOWANCE
SOURCE_GRID = C.OHN_RUNS / 'GRID'
SOURCE_BANKS = ('REF_BANK', 'WITNESS_BANK', 'INC_BANK', 'PROT_P1', 'PROT_P2', 'PROT_P2A', 'PROT_P2L', 'PROT_P2AL', 'PROT_P3',
                'PROT_P3A', 'RR_LC_WIT', 'RR_LC_PROT_P1', 'RR_LC_PROT_P2', 'RR_LC_PROT_P2B50', 'RR_LC_PROT_P2A')
PHI_PRIOR_BANKS = ('REF_BANK', 'WITNESS_BANK', 'RR_LC_WIT', 'INC_BANK', 'PROT_P1', 'RR_LC_PROT_P1', 'PROT_P2', 'RR_LC_PROT_P2',
                   'RR_LC_PROT_P2B50', 'PROT_P2A', 'RR_LC_PROT_P2A', 'PROT_P3', 'PROT_P3A')   # the source INCOME_NOISE banks
NOMINATION = C.OHN_RESULTS / 'NOMINATION.json'
# ---- L1 components (difference level), computed label-free from the frozen grid over PHI_PRIOR_BANKS (rechecked at run time)
COMPONENTS = {
    'income': {'sd_A': 0.02123682861190588, 'sd_k': 0.0018486366498115536, 'se_cell': 0.008962013990683278},
    'occupation': {'sd_A': 0.007565469096113031, 'sd_k': 0.0036816255632697186, 'se_cell': 0.009069907677839946},
    'pair': {'sd_A': 0.0075047647225215, 'sd_k': 0.005923903453354604, 'se_cell': 0.008244617631262354},
}
L1_MU = (-0.030, -0.020, -0.010, 0.0, 0.005, 0.010, 0.020)
L1_RHO = (0.0, 0.76)
L1_NREP = (2, 3)
N_SIM = 20000
SIM_SEED = 2026101820
MULTI_PATTERNS = ('ALL_0', 'ALL_-0.010', 'IUT_income_+0.010_others_0', 'IUT_income_+0.010_others_-0.010',
                  'IUT_income_+0.020_others_0', 'IUT_income_+0.020_others_-0.010', 'IUT_occupation_+0.010_others_0',
                  'IUT_occupation_+0.010_others_-0.010', 'IUT_occupation_+0.020_others_0', 'IUT_occupation_+0.020_others_-0.010')
MULTI_SETTINGS = tuple((rho, nr) for rho, nr in ((0.76, 2), (0.0, 2), (0.76, 3)))
STRUCT_PATTERNS = ('ALL_0', 'ALL_-0.010', 'IUT_income_+0.010_others_0', 'IUT_occupation_+0.010_others_0')
# ---- L0b
L0B_MEMBERS = {
    'SUCC': {'type': 'reference', 'name': 'SUCC'}, 'H_LJ': {'type': 'reference', 'name': 'H_LJ'},
    'NEURAL_RAW': {'type': 'reference', 'name': 'NEURAL_RAW'},
    'A/A1_PO_16@H3': {'type': 'occ', 'family': 'A', 'config': 'A1_PO_16', 'anchor': 'H3'},
    'PROT/P2A_PAIR_B80@H3': {'type': 'occ', 'family': 'PROT', 'config': 'P2A_PAIR_B80', 'anchor': 'H3'},
}
L0B_NEW_K = (2, 3, 4, 5)
SEED_SLOTS = ((0, 1), (2, 3), (4, 5))      # slot 0 = the registered (cached) reader seeds
L0B_PRIMARY = (1, 0)                       # the law's design: candidate k2, k3 against comparator k0, k1
L0B_MAX_FITS = 4000
# ---- L2
L2_BASES = (
    {'id': 'JOINT_OCC', 'member': 'SUCC', 'view': 'occupation', 'law': True,
     'note': "occupation view of SUCC = H_LJ's occupation view (the same JOINT wire bitwise): the law's two bases are one"},
    {'id': 'NEURAL_RAW_OCC', 'member': 'NEURAL_RAW', 'view': 'occupation', 'law': True, 'note': ''},
    {'id': 'H_LJ_INC', 'member': 'H_LJ', 'view': 'income', 'law': True, 'note': ''},
    {'id': 'A1_PO_16_OCC', 'member': 'A/A1_PO_16@H3', 'view': 'occupation', 'law': False,
     'note': 'investigator addition (a task-only witness occupation view), replacing the duplicate JOINT base'},
)
L2_MEMBERS = {'SUCC': L0B_MEMBERS['SUCC'], 'NEURAL_RAW': L0B_MEMBERS['NEURAL_RAW'], 'H_LJ': L0B_MEMBERS['H_LJ'],
              'A/A1_PO_16@H3': L0B_MEMBERS['A/A1_PO_16@H3']}
L2_DELTAS = (0.005, 0.010, 0.020)
L2_TOL = 2e-4
L2_EXTRA_SPLITS = 10
L2_SPLIT_TAG = 'CTS1_screen_L2_split{}'
L2_ETA_GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
STUDY_FILES = ('cts/screen_study.py', 'cts/devreaders.py', 'cts/rules.py',
               'results/pcrl_correction_transfer_stability_v1/roles/statistical_reviewer/RULES.json')


# ================================================================ shared helpers
def frozen_check():
    """The study runs only from committed, unchanged code and rules (frozen before any new candidate is scored with readers)."""
    bad = [f for f in STUDY_FILES if not C.committed(C.ROOT / f)]
    if bad:
        raise PermissionError(f'screen study code / rules not committed: {bad}')


def code_record():
    return {f: C.sha(C.ROOT / f) for f in STUDY_FILES} | {'rules_json_sha256_loaded': RU.load_rules()['_sha256']}


def mc_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else None


def screen_entry(e, phi, alias):
    """OLD and AGG1 for one (candidate, comparator, view) entry {'cells', 'cell_se'} (cts.rules, unchanged)."""
    return {'old': RU.per_cell_gate(e['cells']), 'agg': RU.aggregate_paired_screen(e['cells'], e['cell_se'], phi, alias=alias)}


def _row(e, r):
    a = r['agg']
    return {'old_pass': r['old']['pass'], 'old_cells_above': r['old'].get('cells_above', []), 'old_max': r['old'].get('max'),
            'mean': float(np.mean(list(e['cells'].values()))), 'agg_pass': a['pass'], 'agg_label': a['label'], 'U': a.get('U'),
            'dbar': a.get('dbar'), 'phi': a.get('phi'), 'phi_emp': a.get('phi_emp'), 'se_total': a.get('se_total'),
            'agg_flags': a.get('flags', []),
            'rms_cell_se': math.sqrt(float(np.mean(np.square(list(e['cell_se'].values()))))) if e.get('cell_se') else None}


# ================================================================ part A: frozen source grid (label-free; L0a, configurations)
def load_source_grid(banks=SOURCE_BANKS, root=SOURCE_GRID):
    """cells[(cell, member, view)] = entry, boot[(cell, member, view)] = bootstrap draws, specs[member] = definition. Identical
    (cell, member, view) entries from several banks come from the same cached readers and must agree."""
    cells, boot, specs, problems, files = {}, {}, {}, [], {}
    for b in banks:
        for s in SEEDS:
            d = Path(root) / b / f'S{s}'
            if not (d / 'GRID.json').is_file():
                continue
            files[f'{b}/S{s}/GRID.json'] = C.sha(d / 'GRID.json'); files[f'{b}/S{s}/BOOT.npz'] = C.sha(d / 'BOOT.npz')
            g = json.loads((d / 'GRID.json').read_text())
            for name, m in g['spec']['members'].items():
                if name in specs and specs[name] != m:
                    problems.append(f'{name}: member definition differs between banks')
                specs.setdefault(name, m)
            with np.load(d / 'BOOT.npz') as z:
                bt = {k: z[k].copy() for k in z.files}
            for ck, cres in g['cells'].items():
                for name, ent in cres.items():
                    for v, e in ent.items():
                        key = (ck, name, v)
                        if key in cells:
                            if abs(cells[key]['score_auc'] - e['score_auc']) > 1e-12 or abs(cells[key]['val_auc'] - e['val_auc']) > 1e-12:
                                problems.append(f'{key}: {b} differs from an earlier bank')
                            continue
                        cells[key] = e
                        boot[key] = bt[f'{ck}__{name}__{v}']
    return {'cells': cells, 'boot': boot, 'specs': specs, 'problems': problems, 'files_sha256': files}


def members_full(cells, view):
    """Members with all 8 registered full cells of the view (grid cell ids 'S<s>R<r>k<k>' are the rules' cell ids)."""
    return sorted({k[1] for k in cells if all((c, k[1], view) in cells for c in FULL_CELLS)})


def comparison(cells, boot, cand, comp, view):
    """Paired per-cell difference candidate - comparator (the screens' convention, every view) and household-bootstrap SE."""
    d, se = {}, {}
    for c in FULL_CELLS:
        d[c] = float(cells[(c, cand, view)]['score_auc'] - cells[(c, comp, view)]['score_auc'])
        se[c] = float(np.std(boot[(c, cand, view)] - boot[(c, comp, view)], ddof=1))
    return {'cells': d, 'cell_se': se}


def entry_signature(cells, name, view):
    return tuple(round(cells[(c, name, view)]['score_auc'], 12) for c in FULL_CELLS) + \
        tuple(round(cells[(c, name, view)]['val_auc'], 12) for c in FULL_CELLS)


def member_wires(specs, names=None):
    """Label-free: SHA-256 of each member's released wires (float64 vector + argmax) per seed, from the frozen releases."""
    out = {}
    for name in sorted(names if names is not None else specs):
        out[name] = {}
        for s in SEEDS:
            try:
                _, prov = DR.load_member(specs[name], s)
            except FileNotFoundError:
                continue                                          # a seed-0-only source configuration
            inc, occ = prov['wire_sha256']['income'], prov['wire_sha256']['occupation']
            out[name][s] = {'income': inc, 'occupation': occ, 'pair': hashlib.sha256(f'{inc}|{occ}'.encode()).hexdigest()}
    return out


def is_alias(wires, a, b, view):
    """Exact alias in the rule's sense: the view's wire bitwise equal on both seeds."""
    return all(s in wires.get(a, {}) and s in wires.get(b, {}) and wires[a][s][view] == wires[b][s][view] for s in SEEDS)


def wire_groups(cells, wires, view):
    names = [n for n in members_full(cells, view) if all(s in wires.get(n, {}) for s in SEEDS)]
    by_sig, by_wire = {}, {}
    for n in names:
        by_sig.setdefault(entry_signature(cells, n, view), []).append(n)
        by_wire.setdefault(tuple(wires[n][s][view] for s in SEEDS), []).append(n)
    a = sorted(sorted(v) for v in by_sig.values())
    b = sorted(sorted(v) for v in by_wire.values())
    return {'members': len(names), 'groups_by_entries': len(a), 'groups_by_wires': len(b), 'agree': a == b,
            'alias_groups': [g for g in b if len(g) > 1], 'representatives': sorted(v[0] for v in by_wire.values())}


def phi_prior_components(cells, boot, view):
    """RULES.json's phi_prior construction on the frozen source grid: groups of identical grid entries (members with all 8 cells),
    every pair of distinct groups. Gross (registered): sd_split^2 = mean over seeds of (x_{s,R1} - x_{s,R2})^2 / 2, x = mean over
    reader seeds; net (source pooled, not used): max(0, mean(dr^2)/2 - var_k/2). Also the L1 components (net sd_A, sd_k, se)."""
    groups = {}
    for n in members_full(cells, view):
        groups.setdefault(entry_signature(cells, n, view), []).append(n)
    reps = sorted(tuple(v)[0] for v in groups.values())
    sd2, se2, dks, drs = [], [], [], []
    for a, b in itertools.combinations(reps, 2):
        e = comparison(cells, boot, a, b, view)
        d = e['cells']
        x = {(s, r): (d[f'S{s}R{r}k0'] + d[f'S{s}R{r}k1']) / 2 for s in SEEDS for r in (1, 2)}
        sd2.append(float(np.mean([(x[(s, 1)] - x[(s, 2)]) ** 2 / 2 for s in SEEDS])))
        se2.append(float(np.mean([v ** 2 for v in e['cell_se'].values()])))
        dks += [d[f'S{s}R{r}k0'] - d[f'S{s}R{r}k1'] for s in SEEDS for r in (1, 2)]
        drs += [x[(s, 1)] - x[(s, 2)] for s in SEEDS]
    if not sd2:
        return {'groups': len(reps), 'pairs': 0}
    var_k = float(np.mean(np.square(dks)) / 2)
    var_net = max(0.0, float(np.mean(np.square(drs)) / 2) - var_k / 2)
    return {'groups': len(reps), 'pairs': len(sd2), 'phi_gross': float(np.mean(sd2) / np.mean(se2)),
            'phi_net': var_net / float(np.mean(se2)), 'sd_split_gross': math.sqrt(float(np.mean(sd2))),
            'sd_A_net': math.sqrt(var_net), 'rms_cell_se': math.sqrt(float(np.mean(se2))), 'sd_k': math.sqrt(var_k)}


def phi_priors(computed=None):
    """The registered gross phi_prior per view (refused if not a number or if it differs from cts.rules.PHI_PRIOR); the value
    recomputed here from the frozen source grid is recorded beside it."""
    reg = RU.load_rules()['screens']['aggregate_paired_rule']['phi_prior']
    out = {}
    for v in ALL_VIEWS:
        x = reg.get(v)
        if not (isinstance(x, (int, float)) and math.isfinite(x)) or x != RU.PHI_PRIOR[v]:
            raise PermissionError(f'phi_prior[{v}] is not the registered number of RULES.json / cts.rules ({x!r})')
        rec = {'value': float(x), 'source': 'RULES.json (registered, gross construction)'}
        if computed is not None:
            g = computed[v].get('phi_gross')
            rec.update({'recomputed_gross': g, 'recomputed_agrees': bool(g is not None and abs(g - x) < 1e-9)})
        out[v] = rec
    return out


def screen_configuration(cells, boot, wires, cand, phis, comparators=REQUIRED):
    per = {}
    for comp in comparators:
        for v in VIEWS:
            e = comparison(cells, boot, cand, comp, v)
            alias = is_alias(wires, cand, comp, v)
            per[f'{comp}|{v}'] = {'comparator': comp, 'view': v, 'alias': alias, 'cells': e['cells'], 'cell_se': e['cell_se'],
                                  **_row(e, screen_entry(e, phis[v]['value'], alias))}
    return {'per_comparison': per, 'old_pass': all(x['old_pass'] for x in per.values()),
            'agg_pass': all(x['agg_pass'] for x in per.values()),
            'old_failed': sorted(k for k, x in per.items() if not x['old_pass']),
            'agg_failed': sorted(k for k, x in per.items() if not x['agg_pass'])}


def what_if_label(rec, recipient_pass, screen_name):
    """The source nomination order (ohn.rules.evaluate_config) with the recipient-gate result supplied: labels decided before the
    recipient gate are unchanged; then utility point, recipient screen, reserve, benefit stability, matched component."""
    lab = rec['label']
    if lab in ('NOT_ADMISSIBLE', 'NOT_CONSIDERED_ONE_SEED_ONLY') or lab.startswith('ALIAS_OF_'):
        return lab
    if rec.get('utility_point_failed'):
        return 'UTILITY_POINT_INELIGIBLE'
    if not recipient_pass:
        return 'RECIPIENT_POINT_FAILED' if screen_name == 'OLD' else 'RECIPIENT_SCREEN_FAILED_AGG1'
    if (rec.get('outcomes') or {}).get('reserve') != 'RESERVE_MET':
        return 'RESERVE_NOT_MET'
    if rec.get('benefit_not_stable'):
        return 'BENEFIT_NOT_STABLE'
    if (rec.get('outcomes') or {}).get('matched_component_comparison') != 'PRESENT':
        return 'NO_MATCHED_COMPONENT_COMPARISON'
    return 'QUALIFYING'


BANDS = ((-1.0, -0.02), (-0.02, -0.01), (-0.01, -0.005), (-0.005, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 1.0))


def _bands(rows):
    out = []
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r['mean'] < hi]
        out.append({'band_of_observed_8_cell_mean': f'[{lo:+.3f}, {hi:+.3f})', 'n': len(sel),
                    'old_pass_rate': (sum(r['old_pass'] for r in sel) / len(sel)) if sel else None,
                    'agg_pass_rate': (sum(r['agg_pass'] for r in sel) / len(sel)) if sel else None})
    return out


def source_part(banks=SOURCE_BANKS):
    """Part A (label-free): L0a, the H3 income record, every two-seed source configuration under both screens with the
    descriptive what-if, every pair of distinct wires (descriptive), the phi_prior and L1 components recomputed."""
    t0 = time.process_time()
    G = load_source_grid(banks)
    cells, boot = G['cells'], G['boot']
    wires = member_wires(G['specs'])
    Gp = load_source_grid(PHI_PRIOR_BANKS)
    comps = {v: phi_prior_components(Gp['cells'], Gp['boot'], v) for v in ALL_VIEWS}
    phis = phi_priors(comps)
    comp_check = {v: all(abs(comps[v][a] - COMPONENTS[v][b]) < 1e-12 for a, b in (('sd_A_net', 'sd_A'), ('sd_k', 'sd_k'), ('rms_cell_se', 'se_cell')))
                  for v in ALL_VIEWS}
    groups = {v: wire_groups(cells, wires, v) for v in ALL_VIEWS}
    # L0a: every ordered alias pair of frozen releases with all 8 cells and both seeds
    l0a = {}
    for v in ALL_VIEWS:
        rows = []
        for g in groups[v]['alias_groups']:
            for a, b in itertools.permutations(g, 2):
                e = comparison(cells, boot, a, b, v)
                r_alias = screen_entry(e, phis[v]['value'], True)
                r_plain = screen_entry(e, phis[v]['value'], False)
                rows.append({'candidate': a, 'comparator': b, 'max_abs_cell': max(abs(x) for x in e['cells'].values()),
                             'max_cell_se': max(e['cell_se'].values()), 'old_pass': r_alias['old']['pass'],
                             'agg_label': r_alias['agg']['label'], 'agg_pass': r_alias['agg']['pass'],
                             'agg_without_alias_flag': r_plain['agg']['label'], 'agg_without_alias_flag_U': r_plain['agg']['U']})
        l0a[v] = {'ordered_alias_pairs': len(rows), 'old_pass_rate': (sum(r['old_pass'] for r in rows) / len(rows)) if rows else None,
                  'agg_pass_rate': (sum(r['agg_pass'] for r in rows) / len(rows)) if rows else None,
                  'all_cells_exactly_zero': all(r['max_abs_cell'] == 0.0 for r in rows),
                  'agg_without_alias_flag_labels': sorted({r['agg_without_alias_flag'] for r in rows}), 'rows': rows}
    # the H3 income record (RULES.md section 11): SUCC's income wire is H3's frozen income wire
    from ohn import devrun as D
    h3_ok = all(C.SB.wire_sha(D.anchor_income('H3', s, 'MODEL_SELECT')) == wires['SUCC'][s]['income'] for s in SEEDS)
    h3 = {}
    for comp in REQUIRED:
        e = comparison(cells, boot, 'SUCC', comp, 'income')
        al = is_alias(wires, 'SUCC', comp, 'income')
        h3[comp] = {'alias': al, 'cells': e['cells'], 'cell_se': e['cell_se'], **_row(e, screen_entry(e, phis['income']['value'], al))}
    # source configurations and the descriptive what-if
    nom = json.loads(NOMINATION.read_text())
    configs, whatif = {}, {}
    full_inc = set(members_full(cells, 'income'))
    for rec in nom['configurations']:
        cid = rec['config_id']
        two_seed = all(s in wires.get(cid, {}) for s in SEEDS) and cid in full_inc and all(c in full_inc for c in REQUIRED)
        if not rec.get('per_comparator') or not two_seed:
            whatif[cid] = {'source_label': rec['label'], 'old_reproduced_label': rec['label'], 'what_if_label_AGG1': rec['label'],
                           'note': 'decided before the recipient gate (unchanged)'}
            continue
        sc = screen_configuration(cells, boot, wires, cid, phis)
        src_failed = sorted(rec.get('recipient_point_failed') or [])
        configs[cid] = {**sc, 'source_label': rec['label'], 'source_recipient_point_failed': src_failed,
                        'old_reproduces_source': sorted(sc['old_failed']) == src_failed}
        whatif[cid] = {'source_label': rec['label'], 'old_reproduced_label': what_if_label(rec, sc['old_pass'], 'OLD'),
                       'what_if_label_AGG1': what_if_label(rec, sc['agg_pass'], 'AGG1'), 'agg_pass': sc['agg_pass'],
                       'agg_failed': sc['agg_failed'], 'old_failed': sc['old_failed'],
                       'reserve': (rec.get('outcomes') or {}).get('reserve'),
                       'benefit_not_stable': rec.get('benefit_not_stable'),
                       'matched_component': (rec.get('outcomes') or {}).get('matched_component_comparison')}
    qualifying = [c for c, w in whatif.items() if w['what_if_label_AGG1'] == 'QUALIFYING']
    # every pair of distinct wires (descriptive; truth unknown)
    pairs = {}
    for v in VIEWS:
        reps = groups[v]['representatives']
        rows = []
        for a, b in itertools.permutations(reps, 2):
            e = comparison(cells, boot, a, b, v)
            rows.append({'candidate': a, 'comparator': b, **_row(e, screen_entry(e, phis[v]['value'], False))})
        pairs[v] = {'distinct_wires': len(reps), 'oriented_pairs': len(rows),
                    'old_pass_rate': sum(r['old_pass'] for r in rows) / len(rows),
                    'agg_pass_rate': sum(r['agg_pass'] for r in rows) / len(rows),
                    'disagree_old_pass_agg_fail': sum(r['old_pass'] and not r['agg_pass'] for r in rows),
                    'disagree_old_fail_agg_pass': sum(r['agg_pass'] and not r['old_pass'] for r in rows),
                    'by_mean_band': _bands(rows), 'rows': rows}
    req_groups = {v: sorted(sorted(g) for g in _group(REQUIRED, wires, v).values()) for v in ALL_VIEWS}
    return {'schema': SCHEMA + '/source', 'utc': C.now(), 'labels_read': False, 'inputs': G['files_sha256'],
            'merge_problems': G['problems'] + Gp['problems'], 'phi_prior_components': comps, 'phi_prior_used': phis,
            'l1_components_recomputed_equal': comp_check, 'wire_groups': groups, 'required_comparator_wire_groups': req_groups,
            'L0a': l0a, 'h3_income_record': {'h3_anchor_income_is_SUCC_income_both_seeds': h3_ok, 'per_comparator': h3},
            'configurations': configs, 'what_if': whatif, 'what_if_qualifying': qualifying,
            'what_if_outcome': ('NO_QUALIFYING_CONFIGURATION (unchanged under AGG1)' if not qualifying else
                                f'FINALIST_WOULD_BE_NOMINATED (descriptive what-if): {qualifying}'),
            'old_gate_reproduces_source_all': all(c['old_reproduces_source'] for c in configs.values()),
            'distinct_wire_pairs': pairs, 'cpu_s': round(time.process_time() - t0, 1),
            'note': 'label-free re-analysis of frozen source grid outputs; the source label is never rewritten; a screen pass is '
                    'permission for further development, not evidence of protection'}


def _group(names, wires, view):
    g = {}
    for n in names:
        g.setdefault(tuple(wires[n][s][view] for s in SEEDS), []).append(n)
    return g


# ================================================================ L1 and its multiplicity version (no data)
def l1_single(view, mu, rho, n_rep, n_sim, seed):
    comp = COMPONENTS[view]
    r = RU.oc_rates(mu, n_sim, seed, sd_split=comp['sd_A'], sd_reader=comp['sd_k'], se_cell=comp['se_cell'], rho_seed=rho,
                    phi_prior=RU.PHI_PRIOR[view], n_rep=n_rep)
    o, a = r['per_cell_gate_pass'], r['aggregate_pass']
    return {'law': 'L1', 'view': view, 'mu': mu, 'rho_seed': rho, 'n_R': n_rep, 'n_sim': n_sim, 'seed': seed, **comp,
            'phi_prior': RU.PHI_PRIOR[view], 'old_pass_rate': o, 'old_mc_se': mc_se(o, n_sim), 'agg_pass_rate': a,
            'agg_mc_se': mc_se(a, n_sim)}


def pattern_mu(pattern, n_comp=6):
    """{view: [mu for each comparator]} for a multiplicity pattern; the IUT comparison is comparator 0 of the named view."""
    if pattern == 'ALL_0':
        return {v: [0.0] * n_comp for v in VIEWS}
    if pattern == 'ALL_-0.010':
        return {v: [-0.010] * n_comp for v in VIEWS}
    _, view, bad, _, others = pattern.split('_')
    out = {v: [float(others)] * n_comp for v in VIEWS}
    out[view][0] = float(bad)
    return out


def multi_draw(mus, rng, rho, n_rep, structure='independent'):
    """One configuration under the multiplicity law: per view, release-level effects (candidate x = 0 shared by its comparisons,
    comparators x = 1..6 own effects): split effect a_{x,s,R} (per-release SD sd_A / sqrt 2, seeds correlated rho, independent
    across R) and reader-seed effect e_{x,s,R,k} (SD sd_k / sqrt 2); cell difference = mu + (a_0 - a_x) + (e_0 - e_x); reported
    cell SE = se_cell. structure 'h3_actual' (investigator addition; the required comparators' wires for an H3-anchored
    candidate): income comparator 5 (SUCC) is an exact alias (cells 0, alias flag); occupation comparators 0-4 (H_LJ, SUCC, TWJ,
    B2TWIN, RETIRED_B3CTL) share one comparator effect (one JOINT wire), comparator 5 (NEURAL_RAW) has its own."""
    out = []
    for view in VIEWS:
        comp = COMPONENTS[view]
        sd_a, sd_e = comp['sd_A'] / math.sqrt(2.0), comp['sd_k'] / math.sqrt(2.0)
        n_rel = 1 + len(mus[view])
        z = rng.standard_normal((n_rel, n_rep, 2))
        A = np.empty_like(z)
        A[..., 0] = sd_a * z[..., 0]
        A[..., 1] = sd_a * (rho * z[..., 0] + math.sqrt(1 - rho ** 2) * z[..., 1])
        E = sd_e * rng.standard_normal((n_rel, n_rep, 2, 2))
        for j, mu in enumerate(mus[view]):
            x = 1 + j
            alias = False
            if structure == 'h3_actual':
                if view == 'income' and j == 5:
                    alias = True
                if view == 'occupation' and j <= 4:
                    x = 1                                         # one shared JOINT comparator wire
            if alias:
                cells = {f'S{s}R{r + 1}k{k}': 0.0 for s in SEEDS for r in range(n_rep) for k in (0, 1)}
                ses = {c: 0.0 for c in cells}
            else:
                cells = {f'S{s}R{r + 1}k{k}': float(mu + A[0, r, s] - A[x, r, s] + E[0, r, s, k] - E[x, r, s, k])
                         for s in SEEDS for r in range(n_rep) for k in (0, 1)}
                ses = {c: comp['se_cell'] for c in cells}
            out.append((view, j, cells, ses, alias))
    return out


def l1_multi(pattern, rho, n_rep, n_sim, seed, structure='independent'):
    mus = pattern_mu(pattern)
    if structure == 'h3_actual':
        mus['income'][5] = 0.0                                    # the alias comparison (SUCC) is exactly 0
        mus['occupation'][1:5] = [mus['occupation'][0]] * 4       # one shared JOINT wire: one true shift for its 5 comparisons
    rng = np.random.default_rng(seed)
    po = pa = 0
    single_old = {v: 0 for v in VIEWS}; single_agg = {v: 0 for v in VIEWS}
    for _ in range(n_sim):
        all_o = all_a = True
        for view, j, cells, ses, alias in multi_draw(mus, rng, rho, n_rep, structure):
            o = RU.per_cell_gate(cells)['pass']
            a = RU.aggregate_paired_screen(cells, ses, RU.PHI_PRIOR[view], alias=alias)['pass']
            all_o &= o; all_a &= a
            if j == 0:
                single_old[view] += o; single_agg[view] += a
        po += all_o; pa += all_a
    o, a = po / n_sim, pa / n_sim
    return {'law': 'L1-multiplicity' if structure == 'independent' else 'L1-multiplicity-h3-actual-structure (investigator addition)',
            'pattern': pattern, 'mu': mus, 'rho_seed': rho, 'n_R': n_rep, 'n_sim': n_sim, 'seed': seed,
            'comparisons': '6 required comparators x 2 views (income, occupation) x 2 seeds', 'structure': structure,
            'old_pass_all_rate': o, 'old_mc_se': mc_se(o, n_sim), 'agg_pass_all_rate': a, 'agg_mc_se': mc_se(a, n_sim),
            'first_comparison_pass_rate': {v: {'old': single_old[v] / n_sim, 'agg': single_agg[v] / n_sim} for v in VIEWS}}


def simulate_part(n_sim=N_SIM, seed=SIM_SEED, views=ALL_VIEWS, mus=L1_MU, rhos=L1_RHO, nreps=L1_NREP,
                  patterns=MULTI_PATTERNS, settings=MULTI_SETTINGS, struct_patterns=STRUCT_PATTERNS):
    t0 = time.process_time()
    single, multi, struct = [], [], []
    i = 0
    for view in views:
        for rho in rhos:
            for nr in nreps:
                for mu in mus:
                    i += 1
                    single.append(l1_single(view, mu, rho, nr, n_sim, seed + i))
    for rho, nr in settings:
        for p in patterns:
            i += 1
            multi.append(l1_multi(p, rho, nr, n_sim, seed + i))
    for p in struct_patterns:
        i += 1
        struct.append(l1_multi(p, 0.76, 2, n_sim, seed + i, structure='h3_actual'))
    return {'schema': SCHEMA + '/simulate', 'utc': C.now(), 'components': COMPONENTS, 'phi_prior': dict(RU.PHI_PRIOR),
            'z': RU.Z, 'allowance': ALLOWANCE, 'L1_single': single, 'L1_multiplicity': multi,
            'L1_multiplicity_h3_actual_structure': struct, 'cpu_s': round(time.process_time() - t0, 1),
            'note': 'simulated pass rates under these laws only; never universal error rates'}


# ================================================================ L0b: identical releases, separately fitted readers (labels)
def _view_key(prov_by_seed, view):
    return tuple((p['wire_sha256']['income'] if view == 'income' else p['wire_sha256']['occupation'] if view == 'occupation'
                  else (p['wire_sha256']['income'], p['wire_sha256']['occupation'])) for p in prov_by_seed)


def wire_representatives(provs, names, view):
    """{representative: [members sharing the view's wire on both seeds]} in the given member order."""
    groups = {}
    for n in names:
        groups.setdefault(_view_key([provs[f'{n}|S{s}'] for s in SEEDS], view), []).append(n)
    return {g[0]: g for g in groups.values()}


def zero_shift_entries(res, reps_by_view, slots=SEED_SLOTS):
    """For every representative release and view, every ordered pair of distinct reader-seed slots (candidate slot a, comparator
    slot b): d_{s,R,j} = AUC(winner, k = a_j) - AUC(winner, k = b_j), paired bootstrap SEs (a cell's weights do not depend on the
    reader seed). Cells are named S<s>R<r>k<j> (j = position in the slot)."""
    out = {}
    for view, reps in reps_by_view.items():
        for name, sharing in reps.items():
            for ia, ib in itertools.permutations(range(len(slots)), 2):
                a, b = slots[ia], slots[ib]
                cells, ses, wins = {}, {}, {}
                for s in SEEDS:
                    for r in (1, 2):
                        for j in (0, 1):
                            ea = res[f'S{s}R{r}k{a[j]}'][name][view]; eb = res[f'S{s}R{r}k{b[j]}'][name][view]
                            c = f'S{s}R{r}k{j}'
                            cells[c] = float(ea['score_auc'] - eb['score_auc'])
                            ses[c] = float(np.std(np.asarray(ea['boot']) - np.asarray(eb['boot']), ddof=1))
                            wins[c] = [ea['winner'], eb['winner']]
                out[f'{name}|{view}|P{ia}vsP{ib}'] = {'release': name, 'sharing_wire': sharing, 'view': view, 'slots': [ia, ib],
                                                      'candidate_seeds': list(a), 'comparator_seeds': list(b), 'cells': cells,
                                                      'cell_se': ses, 'winners': wins,
                                                      'cells_same_winner': sum(x[0] == x[1] for x in wins.values()),
                                                      'cells_exact_zero': sum(x == 0.0 for x in cells.values())}
    return out


def zero_shift_screens(entries, primary=L0B_PRIMARY):
    rows = []
    for key, e in entries.items():
        r = screen_entry(e, RU.PHI_PRIOR[e['view']], False)
        rows.append({'entry': key, 'release': e['release'], 'view': e['view'], 'slots': e['slots'],
                     'primary': tuple(e['slots']) == tuple(primary), 'cells_same_winner': e['cells_same_winner'],
                     'cells_exact_zero': e['cells_exact_zero'], 'max_abs_cell': max(abs(x) for x in e['cells'].values()),
                     **_row(e, r)})
    by_view = {}
    for v in ALL_VIEWS:
        for kind, sel in (('primary', [r for r in rows if r['view'] == v and r['primary']]), ('all_ordered_slot_pairs', [r for r in rows if r['view'] == v])):
            if sel:
                n = len(sel)
                po, pa = sum(r['old_pass'] for r in sel) / n, sum(r['agg_pass'] for r in sel) / n
                by_view.setdefault(v, {})[kind] = {
                    'entries': n, 'old_pass_rate': po, 'agg_pass_rate': pa,
                    'max_abs_cell': max(r['max_abs_cell'] for r in sel), 'median_rms_cell_se': float(np.median([r['rms_cell_se'] for r in sel])),
                    'share_cells_exact_zero': sum(r['cells_exact_zero'] for r in sel) / (8 * n),
                    'phi_emp': [r['phi_emp'] for r in sel], 'U': [r['U'] for r in sel],
                    'note': 'entries are not independent (shared releases, cells and reader seeds); counts, not rates with an MC SE'}
    return {'rows': rows, 'by_view': by_view}


def expected_new_fits(bank, s, sex, unit, ks, views, reps=REPS, caches=None):
    """Recipe fits that cell_readers would perform (no fit here): unique (view, wire, cell) without a cached entry, x 16."""
    caches = caches or DR.default_caches()
    sx = C.arr_sha(np.asarray(sex))
    need = set()
    for rep in reps:
        for k in ks:
            rs = DR.reader_seed(s, k)
            for rel in bank.values():
                for v in views:
                    key = DR.cache_key(rep, rs, 1.0, DR.wire_hash(v, rel['income'], rel['occupation']), sx)
                    if not (Path(caches['own']) / f'{key}.npz').is_file() and not (Path(caches['source']) / f'{key}.npz').is_file():
                        need.add(key)
    return len(need) * len(DR.RECIPES)


def l0b_part(out_dir=None):
    DR.lock_check()
    frozen_check()
    out = Path(out_dir) if out_dir else PRIV_OUT / 'L0B'
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load('MODEL_SELECT', ('features', 'sex'), operation='screen_study',
               purpose='screen study law L0b: identical frozen releases read by separately fitted readers (new reader seeds k2-k5 '
                       'beside the cached k0/k1, same FIT/VALIDATION/SCORE splits); operating characteristics only, never selection')
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    del S                                                   # readers never see Census inputs
    t0 = time.process_time()
    stats = DR.new_stats()
    res_all, provs, banks = {}, {}, {}
    for s in SEEDS:
        banks[s] = {}
        for name, m in L0B_MEMBERS.items():
            banks[s][name], provs[f'{name}|S{s}'] = DR.load_member(m, s, n=len(sex), row_id=row_id)
    expected = sum(expected_new_fits(banks[s], s, sex, unit, (0, 1) + L0B_NEW_K, ALL_VIEWS) for s in SEEDS)
    if expected > L0B_MAX_FITS:
        raise PermissionError(f'L0b would fit {expected} recipes (> {L0B_MAX_FITS}); refused (budget)')
    for s in SEEDS:
        res, stats = DR.evaluate_bank(banks[s], s, sex, unit, stats=stats, reps=REPS, ks=(0, 1) + L0B_NEW_K, views=ALL_VIEWS)
        res_all.update(res)
    reps_by_view = {v: wire_representatives(provs, list(L0B_MEMBERS), v) for v in ALL_VIEWS}
    entries = zero_shift_entries(res_all, reps_by_view)
    scr = zero_shift_screens(entries)
    out.mkdir(parents=True, mode=0o700)
    rec = {'schema': SCHEMA + '/L0b', 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'members': L0B_MEMBERS,
           'reader_seeds_new': list(L0B_NEW_K), 'reader_seeds_cached': [0, 1], 'slots': SEED_SLOTS, 'primary_design': L0B_PRIMARY,
           'views': list(ALL_VIEWS), 'wire_representatives': reps_by_view, 'expected_new_fits': expected,
           'cells': DR.strip(res_all), 'entries': entries, 'screens': scr, 'provenance': provs, 'stats': stats,
           'cpu_s': round(time.process_time() - t0, 1), 'versions': DR.versions(), 'code': code_record(),
           'note': 'same release, same cells, separately fitted readers: a true zero shift; split noise cancels by construction'}
    C.write_json(out / 'L0B.json', rec)
    C.unit_stats(reader_fits=stats['reader_fits'], cache_hits_own=stats['cache_hits_own'], cache_hits_source=stats['cache_hits_source'],
                 cache_hits_source_r3=stats['cache_hits_source_r3'], reader_cpu_s=round(stats['reader_cpu_s'], 3))
    return rec


# ================================================================ L2: release-level semi-synthetic law (labels)
def plant(p, sex, eta):
    """log p' = log p + eta (2 SEX - 1) v, renormalized; v = e_1 - (1/K) 1 (the first class against the rest, centred logits)."""
    p = np.asarray(p, dtype=np.float64)
    K = p.shape[1]
    v = np.full(K, -1.0 / K); v[0] += 1.0
    z = np.log(np.maximum(p, 1e-300)) + float(eta) * (2.0 * np.asarray(sex, dtype=np.float64) - 1.0)[:, None] * v[None, :]
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def with_view(rel, view, arr):
    out = dict(rel)
    out[view] = DR._check_prob(arr, C.K[view], f'L2 synthetic {view}', len(arr))
    return out


def lr_score_auc(view, rel, sex, roles, rs):
    """The fixed reader LOG_LR of the slate (cts.devreaders._fit_one): fit on FIT, oriented on VALIDATION, AUC on SCORE."""
    s, _ = DR._fit_one('LOG_LR', view, rel['income'], rel['occupation'], sex, roles['FIT'], roles['VALIDATION'], rs)
    return float(auc(sex[roles['SCORE']], s[roles['SCORE']]))


def calibrate_eta(rel, view, sex, roles, rs, target, tol=L2_TOL, max_iter=40):
    """eta such that LOG_LR's SCORE AUC gain over the base view is target (within tol): first grid point of L2_ETA_GRID (then
    the same points negated) reaching the target, then bisection between it and the previous point."""
    a0 = lr_score_auc(view, rel, sex, roles, rs)
    fits = [1]

    def gain(eta):
        fits[0] += 1
        return lr_score_auc(view, with_view(rel, view, plant(rel[view], sex, eta)), sex, roles, rs) - a0
    for sign in (1.0, -1.0):
        lo, g_lo = 0.0, 0.0
        for e in L2_ETA_GRID:
            g = gain(sign * e)
            if g >= target:
                hi, g_hi = sign * e, g
                for _ in range(max_iter):
                    if abs(g_hi - target) <= tol:
                        break
                    mid = (lo + hi) / 2
                    gm = gain(mid)
                    if gm >= target:
                        hi, g_hi = mid, gm
                    else:
                        lo, g_lo = mid, gm
                return {'eta': hi, 'gain': g_hi, 'base_auc_LOG_LR_R1': a0, 'fits': fits[0], 'converged': abs(g_hi - target) <= tol,
                        'sign': sign}
            lo, g_lo = sign * e, g
    return {'eta': None, 'gain': None, 'base_auc_LOG_LR_R1': a0, 'fits': fits[0], 'converged': False}


def extra_split_shift(base, cand, view, sex, unit, rs, n=L2_EXTRA_SPLITS):
    vals = []
    for i in range(n):
        part = C.household_folds(unit, 10, L2_SPLIT_TAG.format(i))
        roles = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}
        vals.append(lr_score_auc(view, cand, sex, roles, rs) - lr_score_auc(view, base, sex, roles, rs))
    return {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)), 'se': float(np.std(vals, ddof=1) / math.sqrt(n)),
            'n_splits': n, 'values': vals, 'recipe': 'LOG_LR (fixed)'}


def l2_part(out_dir=None):
    DR.lock_check()
    frozen_check()
    out = Path(out_dir) if out_dir else PRIV_OUT / 'L2'
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load('MODEL_SELECT', ('features', 'sex'), operation='screen_study',
               purpose='screen study law L2: SEX-dependent synthetic score shift of frozen comparator views (simulation only), eta '
                       'calibrated with the fixed LOG_LR reader, then the registered reader grid for both screens; never selection')
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    del S
    t0 = time.process_time()
    caches = {'own': out / 'cache', 'source': DR.SOURCE_CACHE, 'source_r3': DR.SOURCE_R3_CACHE}  # synthetic wires kept apart
    stats = DR.new_stats()
    lr_fits = 0
    rels = {(name, s): DR.load_member(m, s, n=len(sex), row_id=row_id)[0] for name, m in L2_MEMBERS.items() for s in SEEDS}
    roles1 = DR.roles_of(unit, 'R1')
    rows = []
    for base in L2_BASES:
        view = base['view']
        for delta in L2_DELTAS:
            cells, ses, per_seed = {}, {}, {}
            for s in SEEDS:
                rel = rels[(base['member'], s)]
                rs = DR.reader_seed(s, 0)
                cal = calibrate_eta(rel, view, sex, roles1, rs, delta)
                lr_fits += cal['fits']
                if cal['eta'] is None:
                    per_seed[s] = {'calibration': cal}
                    continue
                cand = with_view(rel, view, plant(rel[view], sex, cal['eta']))
                res, stats = DR.evaluate_bank({'CAND': cand, 'BASE': rel}, s, sex, unit, stats=stats, reps=REPS, ks=(0, 1),
                                              views=(view,), caches=caches)
                for cell, cr in res.items():               # evaluate_bank cell ids 'S<s>R<r>k<k>' are the rules' cell ids
                    cells[cell] = float(cr['CAND'][view]['score_auc'] - cr['BASE'][view]['score_auc'])
                    ses[cell] = float(np.std(cr['CAND'][view]['boot'] - cr['BASE'][view]['boot'], ddof=1))
                ext = extra_split_shift(rel, cand, view, sex, unit, rs)
                lr_fits += 2 * L2_EXTRA_SPLITS
                per_seed[s] = {'calibration': cal, 'extra_splits_fixed_recipe': ext,
                               'winners': {cell: [cr['CAND'][view]['winner'], cr['BASE'][view]['winner']] for cell, cr in res.items()}}
            row = {'law': 'L2', 'base': base['id'], 'base_member': base['member'], 'view': view, 'target_delta_LOG_LR_R1': delta,
                   'in_law': base['law'], 'note': base['note'], 'per_seed': per_seed}
            if set(cells) == set(FULL_CELLS):
                e = {'cells': cells, 'cell_se': ses}
                r = screen_entry(e, RU.PHI_PRIOR[view], False)
                ext_means = [per_seed[s]['extra_splits_fixed_recipe']['mean'] for s in SEEDS]
                row.update({'cells': cells, 'cell_se': ses, **_row(e, r), 'realized_slate_mean_shift': float(np.mean(list(cells.values()))),
                            'fixed_recipe_shift_10_extra_splits': float(np.mean(ext_means)), 'complete': True})
            else:
                row['complete'] = False
            rows.append(row)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    rec = {'schema': SCHEMA + '/L2', 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV), 'bases': L2_BASES, 'deltas': L2_DELTAS,
           'rows': rows, 'stats': stats, 'calibration_and_split_LOG_LR_fits': lr_fits, 'cpu_s': round(time.process_time() - t0, 1),
           'versions': DR.versions(), 'code': code_record(),
           'note': 'one grid realization per setting (pass / fail, not a rate); truth approximate; synthetic releases exist only here'}
    C.write_json(out / 'L2.json', rec)
    C.unit_stats(reader_fits=stats['reader_fits'], calibration_reader_fits=lr_fits, cache_hits_own=stats['cache_hits_own'],
                 cache_hits_source=stats['cache_hits_source'], reader_cpu_s=round(stats['reader_cpu_s'], 3))
    return rec


# ================================================================ report
def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def table_rows(source):
    rows = []
    for cid, sc in source['configurations'].items():
        wi = source['what_if'][cid]
        for k, x in sc['per_comparison'].items():
            rows.append({'configuration': cid, 'comparator': x['comparator'], 'view': x['view'], 'exact_alias': x['alias'],
                         'old_pass': x['old_pass'], 'old_max_cell': x['old_max'], 'old_cells_above_0.010': x['old_cells_above'],
                         'mean_8_cells': x['mean'], 'agg1_label': x['agg_label'], 'agg1_U': x['U'], 'agg1_dbar': x['dbar'],
                         'agg1_phi': x['phi'], 'agg1_phi_emp': x['phi_emp'], 'agg1_se_total': x['se_total'], 'agg1_flags': x['agg_flags'],
                         'source_label': wi['source_label'], 'what_if_label_AGG1': wi['what_if_label_AGG1']})
    return rows


def oc_rows(source, sim, l0b, l2):
    rows = []
    if source:
        for v, x in source['L0a'].items():
            if x['ordered_alias_pairs']:
                rows.append({'law': 'L0a', 'parameters': f'view={v}; exact aliases of frozen releases (shared cached readers)',
                             'n': x['ordered_alias_pairs'], 'old_pass_rate': x['old_pass_rate'], 'agg_pass_rate': x['agg_pass_rate'],
                             'kind': 'empirical (frozen grid; every alias pair)'})
    if l0b:
        for v, d in l0b['screens']['by_view'].items():
            for kind, x in d.items():
                rows.append({'law': 'L0b', 'parameters': f'view={v}; {kind}; candidate readers k2..k5 vs comparator k0..k5 slots; '
                             f'members={list(L0B_MEMBERS)}', 'n': x['entries'], 'old_pass_rate': x['old_pass_rate'],
                             'agg_pass_rate': x['agg_pass_rate'], 'kind': 'empirical refit (entries not independent)'})
    if sim:
        for r in sim['L1_single']:
            rows.append({'law': 'L1', 'parameters': f"view={r['view']}; mu={r['mu']:+.3f}; rho_seed={r['rho_seed']}; n_R={r['n_R']}; "
                         f"sd_A={r['sd_A']:.6f}; sd_k={r['sd_k']:.6f}; se_cell={r['se_cell']:.6f}; phi_prior={r['phi_prior']:.6f}",
                         'n': r['n_sim'], 'old_pass_rate': r['old_pass_rate'], 'old_mc_se': r['old_mc_se'],
                         'agg_pass_rate': r['agg_pass_rate'], 'agg_mc_se': r['agg_mc_se'], 'kind': 'simulated (exact truth)'})
        for key in ('L1_multiplicity', 'L1_multiplicity_h3_actual_structure'):
            for r in sim[key]:
                rows.append({'law': r['law'], 'parameters': f"pattern={r['pattern']}; rho_seed={r['rho_seed']}; n_R={r['n_R']}; "
                             f"12 comparisons (6 comparators x income, occupation); structure={r['structure']}",
                             'n': r['n_sim'], 'old_pass_rate': r['old_pass_all_rate'], 'old_mc_se': r['old_mc_se'],
                             'agg_pass_rate': r['agg_pass_all_rate'], 'agg_mc_se': r['agg_mc_se'],
                             'kind': 'simulated configuration-level pass-all (exact truth)'})
    if l2:
        for r in l2['rows']:
            rows.append({'law': 'L2', 'parameters': f"base={r['base']} ({r['view']}); target LOG_LR gain={r['target_delta_LOG_LR_R1']:+.3f}"
                         + ('' if r['in_law'] else '; investigator addition'),
                         'n': 1 if r.get('complete') else 0, 'old_pass_rate': float(r['old_pass']) if r.get('complete') else None,
                         'agg_pass_rate': float(r['agg_pass']) if r.get('complete') else None,
                         'realized_slate_mean_shift': r.get('realized_slate_mean_shift'),
                         'fixed_recipe_shift_10_extra_splits': r.get('fixed_recipe_shift_10_extra_splits'), 'U': r.get('U'),
                         'kind': 'semi-synthetic, one grid realization (truth approximate)'})
    return rows


OC_FIELDS = ['law', 'parameters', 'kind', 'n', 'old_pass_rate', 'old_mc_se', 'agg_pass_rate', 'agg_mc_se', 'U',
             'realized_slate_mean_shift', 'fixed_recipe_shift_10_extra_splits']
TABLE_FIELDS = ['configuration', 'comparator', 'view', 'exact_alias', 'old_pass', 'old_max_cell', 'old_cells_above_0.010',
                'mean_8_cells', 'agg1_label', 'agg1_U', 'agg1_dbar', 'agg1_phi', 'agg1_phi_emp', 'agg1_se_total', 'agg1_flags',
                'source_label', 'what_if_label_AGG1']


def _read(p):
    return json.loads(Path(p).read_text()) if Path(p).is_file() else None


def report(priv=PRIV_OUT, out=OUT):
    source, sim = _read(priv / 'SOURCE.json'), _read(priv / 'SIM.json')
    l0b, l2 = _read(priv / 'L0B' / 'L0B.json'), _read(priv / 'L2' / 'L2.json')
    rec = {'schema': SCHEMA, 'utc': C.now(), 'code': code_record(),
           'definitions': {'old': 'cts.rules.per_cell_gate: every one of the 8 cells <= +0.010 (inclusive), income and occupation views, '
                                  'every required comparator', 'AGG1': 'cts.rules.aggregate_paired_screen (RULES.md section 10)',
                           'phi_prior': dict(RU.PHI_PRIOR), 'z': RU.Z, 'allowance': ALLOWANCE, 'required_comparators': list(REQUIRED)},
           'parts_present': {'source': source is not None, 'simulate': sim is not None, 'L0b': l0b is not None, 'L2': l2 is not None},
           'statement': 'simulated or empirical pass rates under the stated laws only, never universal error rates; an AGG1 pass is '
                        'permission for further development, not evidence of protection; the source label NO_QUALIFYING_CONFIGURATION '
                        'is never rewritten (the what-if is descriptive)'}
    if source:
        rec['part_A'] = {k: source[k] for k in ('phi_prior_used', 'l1_components_recomputed_equal', 'required_comparator_wire_groups',
                                                'what_if_outcome', 'what_if_qualifying', 'old_gate_reproduces_source_all', 'merge_problems')}
        rec['part_A']['inputs_sha256'] = source['inputs']
        rec['part_A']['L0a'] = {v: {k: x[k] for k in x if k != 'rows'} for v, x in source['L0a'].items()}
        rec['part_A']['h3_income_record'] = {'h3_anchor_income_is_SUCC_income_both_seeds': source['h3_income_record']['h3_anchor_income_is_SUCC_income_both_seeds'],
                                             'per_comparator': {c: {k: x[k] for k in ('alias', 'old_pass', 'old_max', 'old_cells_above', 'agg_label', 'U', 'dbar', 'phi_emp', 'agg_flags')}
                                                                for c, x in source['h3_income_record']['per_comparator'].items()}}
        rec['part_A']['configurations'] = {c: {k: x[k] for k in ('old_pass', 'agg_pass', 'old_failed', 'agg_failed', 'source_label', 'old_reproduces_source')}
                                           for c, x in source['configurations'].items()}
        rec['part_A']['what_if'] = source['what_if']
        rec['part_A']['distinct_wire_pairs'] = {v: {k: x[k] for k in x if k != 'rows'} for v, x in source['distinct_wire_pairs'].items()}
        _csv(out / 'SCREEN_TABLE.csv', table_rows(source), TABLE_FIELDS)
    if sim:
        rec['L1'] = {k: sim[k] for k in ('components', 'phi_prior', 'L1_single', 'L1_multiplicity', 'L1_multiplicity_h3_actual_structure', 'cpu_s')}
    if l0b:
        rec['L0b'] = {'members': l0b['members'], 'wire_representatives': l0b['wire_representatives'], 'by_view': l0b['screens']['by_view'],
                      'rows': l0b['screens']['rows'], 'stats': l0b['stats'], 'cpu_s': l0b['cpu_s'], 'unit_id': l0b.get('unit_id')}
    if l2:
        rec['L2'] = {'rows': [{k: v for k, v in r.items() if k not in ('cell_se',)} for r in l2['rows']], 'stats': l2['stats'],
                     'calibration_and_split_LOG_LR_fits': l2['calibration_and_split_LOG_LR_fits'], 'cpu_s': l2['cpu_s'],
                     'unit_id': l2.get('unit_id')}
    _csv(out / 'SCREEN_OC.csv', oc_rows(source, sim, l0b, l2), OC_FIELDS)
    C.write_json(out / 'SCREEN_STUDY.json', rec)
    (out / 'SCREEN_STUDY.md').write_text(markdown(rec, source, sim, l0b, l2))
    return rec


def _f(x, nd=3):
    return '-' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def markdown(rec, source, sim, l0b, l2):
    L = ['# Screen operating-characteristic study (question C)', '',
         'Reader investigator, PCRL correction-transfer campaign. Public-use ACS 2019 1-Year PUMS Florida person records, MODEL_SELECT '
         'development role (adaptively used). Development readers estimate the recorded SEX column from released probability vectors '
         'only. Frozen and run before any new method candidate was scored with readers.', '',
         '**What this is.** The source\'s per-cell recipient gate (OLD: every one of the 8 cells of candidate-minus-comparator '
         'single-output SCORE AUC <= +0.010, income and occupation views, each of the 6 required comparators) beside the statistical '
         'reviewer\'s predeclared aggregate paired rule (AGG1: U = dbar + z sqrt(phi v_samp) <= 0.010 per comparison, phi = max(1, '
         'phi_prior, phi_emp), gross phi_prior income 5.6365 / occupation 0.7782 / pair 1.0867, exact aliases pass). Both come from '
         '`cts.rules`. All rates are pass rates under the stated laws only, never universal error rates. An AGG1 pass is permission '
         'for further development, not evidence of protection. The source label NO_QUALIFYING_CONFIGURATION is never rewritten.', '']
    if source:
        h3 = source['h3_income_record']['per_comparator']
        L += ['## Pre-outcome record: H3\'s frozen income view (every H3-anchored release)', '',
              '| Comparator | Old gate | Old max cell | AGG1 | U | dbar | phi_emp |', '|---|---|---|---|---|---|---|']
        for c, x in h3.items():
            L.append(f"| {c} | {'pass' if x['old_pass'] else 'FAIL'} | {_f(x['old_max'], 4)} | {x['agg_label']} | {_f(x['U'], 4)} | "
                     f"{_f(x['dbar'], 4)} | {_f(x['phi_emp'], 2)} |")
        L += ['', 'Every H3-anchored release fails both configuration-level screens through its income view alone, whatever its '
              'occupation output (old gate: against H_LJ and TWJ; AGG1: against H_LJ and RETIRED_B3CTL).', '',
              '## Part A: frozen source releases (label-free)', '',
              f"- Old gate reproduces the source's recipient failures for all {len(source['configurations'])} two-seed configurations: "
              f"{source['old_gate_reproduces_source_all']}.",
              f"- Descriptive what-if (recipient gate replaced by AGG1, everything else as recorded): **{source['what_if_outcome']}**.",
              f"- Required-comparator wire groups: occupation {source['required_comparator_wire_groups']['occupation']} "
              f"(2 distinct wires); income {source['required_comparator_wire_groups']['income']}.", '',
              '| Configuration | Source label | Old fails | AGG1 fails | What-if label (AGG1) |', '|---|---|---|---|---|']
        for c, x in source['configurations'].items():
            L.append(f"| {c} | {x['source_label']} | {len(x['old_failed'])} | {len(x['agg_failed'])} | {source['what_if'][c]['what_if_label_AGG1']} |")
        L += ['', '**L0a (exact aliases, shared cached readers).**', '', '| View | Ordered alias pairs | All cells exactly 0 | Old pass | AGG1 pass |',
              '|---|---|---|---|---|']
        for v, x in source['L0a'].items():
            L.append(f"| {v} | {x['ordered_alias_pairs']} | {x['all_cells_exactly_zero']} | {_f(x['old_pass_rate'])} | {_f(x['agg_pass_rate'])} |")
        L += ['', '**Every pair of distinct frozen wires (descriptive; the truth is unknown).** Pass rates by the observed 8-cell mean:', '']
        for v, x in source['distinct_wire_pairs'].items():
            L += [f"- {v}: {x['distinct_wires']} wires, {x['oriented_pairs']} oriented pairs; old {x['old_pass_rate']:.3f}, AGG1 "
                  f"{x['agg_pass_rate']:.3f}; old pass / AGG1 fail {x['disagree_old_pass_agg_fail']}, old fail / AGG1 pass "
                  f"{x['disagree_old_fail_agg_pass']}."]
            L += ['', '  | Observed mean band | n | Old pass | AGG1 pass |', '  |---|---|---|---|']
            for b in x['by_mean_band']:
                L.append(f"  | {b['band_of_observed_8_cell_mean']} | {b['n']} | {_f(b['old_pass_rate'])} | {_f(b['agg_pass_rate'])} |")
            L.append('')
    if l0b:
        L += ['## L0b: identical releases under separately fitted readers (true zero shift)', '',
              f"Members {list(L0B_MEMBERS)}; candidate readers refitted with new reader seeds, comparator = the same release with the "
              'registered (cached) or other new seeds, same FIT / VALIDATION / SCORE splits. Split noise cancels by construction, so this '
              'isolates reader fitting and winner selection. Entries share releases and seeds (counts, not independent rates).', '',
              '| View | Design | Entries | Old pass | AGG1 pass | Max abs cell | Median rms cell SE | Share of cells exactly 0 |',
              '|---|---|---|---|---|---|---|---|']
        for v, d in l0b['screens']['by_view'].items():
            for kind, x in d.items():
                L.append(f"| {v} | {kind} | {x['entries']} | {_f(x['old_pass_rate'])} | {_f(x['agg_pass_rate'])} | {_f(x['max_abs_cell'], 4)} | "
                         f"{_f(x['median_rms_cell_se'], 4)} | {_f(x['share_cells_exact_zero'])} |")
        L.append('')
    if sim:
        L += ['## L1: cell-level synthetic law (exact truth; 20,000 simulations per setting)', '',
              'd = mu + A_{s,R} + B_{s,R,k}; A ~ N(0, sd_A^2) correlated rho_seed across seeds; B ~ N(0, sd_k^2); reported SE = se_cell. '
              'Components (difference level, frozen grid): ' + '; '.join(
                  f"{v}: sd_A {c['sd_A']:.6f}, sd_k {c['sd_k']:.6f}, se {c['se_cell']:.6f}" for v, c in sim['components'].items()) + '.', '',
              '| View | rho_seed | n_R | mu | Old pass (MC SE) | AGG1 pass (MC SE) |', '|---|---|---|---|---|---|']
        for r in sim['L1_single']:
            L.append(f"| {r['view']} | {r['rho_seed']} | {r['n_R']} | {r['mu']:+.3f} | {r['old_pass_rate']:.4f} ({r['old_mc_se']:.4f}) | "
                     f"{r['agg_pass_rate']:.4f} ({r['agg_mc_se']:.4f}) |")
        L += ['', '**Multiplicity (configuration level: all 12 comparisons must pass; release-level effects, candidate shared).**', '',
              '| Structure | Pattern | rho_seed | n_R | Old pass-all (MC SE) | AGG1 pass-all (MC SE) |', '|---|---|---|---|---|---|']
        for r in sim['L1_multiplicity'] + sim['L1_multiplicity_h3_actual_structure']:
            L.append(f"| {r['structure']} | {r['pattern']} | {r['rho_seed']} | {r['n_R']} | {r['old_pass_all_rate']:.4f} ({r['old_mc_se']:.4f}) | "
                     f"{r['agg_pass_all_rate']:.4f} ({r['agg_mc_se']:.4f}) |")
        L.append('')
    if l2:
        L += ['## L2: semi-synthetic SEX-dependent shift of frozen views (one grid realization per row; truth approximate)', '',
              '| Base view | Target LOG_LR gain | Realized slate mean shift | Fixed-recipe shift (10 extra splits) | Old | AGG1 | U |',
              '|---|---|---|---|---|---|---|']
        for r in l2['rows']:
            if r.get('complete'):
                L.append(f"| {r['base']}{'' if r['in_law'] else ' (addition)'} | {r['target_delta_LOG_LR_R1']:+.3f} | "
                         f"{r['realized_slate_mean_shift']:+.4f} | {r['fixed_recipe_shift_10_extra_splits']:+.4f} | "
                         f"{'pass' if r['old_pass'] else 'FAIL'} | {'pass' if r['agg_pass'] else 'FAIL'} | {r['U']:+.4f} |")
            else:
                L.append(f"| {r['base']} | {r['target_delta_LOG_LR_R1']:+.3f} | calibration failed | - | - | - | - |")
        L.append('')
    L += ['## Limits', '',
          '- Rates hold under the stated laws only. L0b and L2 are single realizations of the registered cells; L0b isolates reader '
          'fitting (split noise cancels for identical releases), L2 truth is approximate.',
          '- The frozen-source rows are adaptive development evidence on MODEL_SELECT; readers are the registered slate only.',
          '- The what-if is descriptive: it changes one component of the source nomination and rewrites nothing.', '']
    return '\n'.join(L)


# ================================================================ invented-data selftest (no real data, no label)
def _fake_grid(rng, members, wires_inc, n_boot=50):
    """Invented grid cells / boot: members with a 'level' per view; identical wires share identical entries."""
    cells, boot = {}, {}
    lvl = {}
    for m in members:
        for v in ALL_VIEWS:
            key = (v, wires_inc[m] if v == 'income' else m)
            lvl.setdefault(key, (rng.normal(0.55, 0.02), rng.normal(size=(len(FULL_CELLS),)) * 0.01, rng.normal(size=(len(FULL_CELLS), n_boot)) * 0.01))
            base, cell_noise, bnoise = lvl[key]
            for i, c in enumerate(FULL_CELLS):
                cells[(c, m, v)] = {'score_auc': float(base + cell_noise[i]), 'val_auc': float(base + 0.5 * cell_noise[i]), 'winner': 'LOG_LR'}
                boot[(c, m, v)] = base + cell_noise[i] + bnoise[i]
    return cells, boot


def selftest():
    rng = np.random.default_rng(777)
    chk = {}
    # 1. comparisons, aliases, phi construction, configuration screen on an invented grid
    members = ['CAND', 'SUCC', 'H_LJ', 'NEURAL_RAW', 'B2TWIN', 'RETIRED_B3CTL', 'TWJ']
    winc = {'CAND': 'H3', 'SUCC': 'H3', 'H_LJ': 'LJ', 'NEURAL_RAW': 'NR', 'B2TWIN': 'B2', 'RETIRED_B3CTL': 'B3', 'TWJ': 'TW'}
    cells, boot = _fake_grid(rng, members, winc)
    wires = {m: {s: {'income': winc[m], 'occupation': m, 'pair': winc[m] + m} for s in SEEDS} for m in members}
    e = comparison(cells, boot, 'CAND', 'H_LJ', 'income')
    c0 = FULL_CELLS[0]
    chk['comparison_sign_candidate_minus_comparator'] = abs(e['cells'][c0] - (cells[(c0, 'CAND', 'income')]['score_auc'] - cells[(c0, 'H_LJ', 'income')]['score_auc'])) < 1e-15
    chk['comparison_se_paired'] = abs(e['cell_se'][c0] - float(np.std(boot[(c0, 'CAND', 'income')] - boot[(c0, 'H_LJ', 'income')], ddof=1))) < 1e-15
    chk['alias_detected'] = is_alias(wires, 'CAND', 'SUCC', 'income') and not is_alias(wires, 'CAND', 'SUCC', 'occupation')
    phis = {v: {'value': RU.PHI_PRIOR[v]} for v in ALL_VIEWS}
    sc = screen_configuration(cells, boot, wires, 'CAND', phis)
    chk['alias_comparison_EXACT_ALIAS'] = sc['per_comparison']['SUCC|income']['agg_label'] == 'EXACT_ALIAS' and sc['per_comparison']['SUCC|income']['old_pass']
    chk['config_level_is_intersection'] = sc['agg_pass'] == all(x['agg_pass'] for x in sc['per_comparison'].values())
    bad = {k: dict(v) for k, v in cells.items()}
    bad[(c0, 'SUCC', 'income')] = dict(bad[(c0, 'SUCC', 'income')], score_auc=bad[(c0, 'SUCC', 'income')]['score_auc'] + 1e-6)
    try:
        screen_configuration(bad, boot, wires, 'CAND', phis); chk['nonzero_alias_refused'] = False
    except ValueError:
        chk['nonzero_alias_refused'] = True
    comps = phi_prior_components(cells, boot, 'occupation')
    sd2, se2 = [], []                                           # direct gross computation over every pair of distinct wires
    for a, b in itertools.combinations(sorted(members), 2):
        ee = comparison(cells, boot, a, b, 'occupation')
        x = {(s, r): (ee['cells'][f'S{s}R{r}k0'] + ee['cells'][f'S{s}R{r}k1']) / 2 for s in SEEDS for r in (1, 2)}
        sd2.append(np.mean([(x[(s, 1)] - x[(s, 2)]) ** 2 / 2 for s in SEEDS])); se2.append(np.mean(np.square(list(ee['cell_se'].values()))))
    chk['phi_gross_formula'] = abs(comps['phi_gross'] - float(np.mean(sd2) / np.mean(se2))) < 1e-12 and comps['pairs'] == len(sd2)
    chk['phi_net_le_gross'] = comps['phi_net'] <= comps['phi_gross'] + 1e-15
    # 2. what-if order
    base = {'label': 'RECIPIENT_POINT_FAILED', 'utility_point_failed': [], 'outcomes': {'reserve': 'RESERVE_MET', 'matched_component_comparison': 'PRESENT'},
            'benefit_not_stable': {}}
    chk['what_if_qualifying_when_all_else_holds'] = what_if_label(base, True, 'AGG1') == 'QUALIFYING'
    chk['what_if_order'] = (what_if_label(base, False, 'AGG1') == 'RECIPIENT_SCREEN_FAILED_AGG1'
                            and what_if_label(dict(base, outcomes={'reserve': 'RESERVE_ONE_SEED_ONLY'}), True, 'AGG1') == 'RESERVE_NOT_MET'
                            and what_if_label(dict(base, benefit_not_stable={'H_LJ': 'SIGN_CHANGES'}), True, 'AGG1') == 'BENEFIT_NOT_STABLE'
                            and what_if_label(dict(base, label='NOT_ADMISSIBLE'), True, 'AGG1') == 'NOT_ADMISSIBLE'
                            and what_if_label(dict(base, utility_point_failed=['s0|x']), True, 'AGG1') == 'UTILITY_POINT_INELIGIBLE'
                            and what_if_label(dict(base, outcomes={'reserve': 'RESERVE_MET', 'matched_component_comparison': 'MISSING'}), True, 'AGG1') == 'NO_MATCHED_COMPONENT_COMPARISON')
    # 3. zero-shift entries from an invented evaluate_bank-like result over reader seeds 0..5
    res = {}
    W = rng.normal(size=(30,))
    for s in SEEDS:
        for r in (1, 2):
            for k in range(6):
                res[f'S{s}R{r}k{k}'] = {'X': {v: {'score_auc': 0.6 + 0.001 * k, 'boot': W + 0.001 * k * (1 + W), 'winner': f'W{k % 2}'} for v in ALL_VIEWS}}
    ents = zero_shift_entries(res, {v: {'X': ['X']} for v in ALL_VIEWS})
    chk['zero_shift_entry_count'] = len(ents) == 3 * 6
    e0 = ents['X|income|P1vsP0']
    chk['zero_shift_cells_and_naming'] = sorted(e0['cells']) == sorted(FULL_CELLS) and abs(e0['cells']['S0R1k0'] - 0.002) < 1e-12 \
        and abs(e0['cells']['S0R1k1'] - 0.002) < 1e-12
    zs = zero_shift_screens(ents)
    chk['zero_shift_primary_rows'] = sum(r['primary'] for r in zs['rows']) == 3
    # 4. L2 machinery: plant and eta calibration with the slate's LOG_LR on invented data
    n = 1500
    unit = np.repeat(np.arange(n // 2), 2)[:n]
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); occ = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-rng.normal(size=n))); inc = np.column_stack((1 - t, t))
    rel = {'income': inc, 'occupation': occ}
    q0 = plant(occ, sex, 0.0)
    chk['plant_eta0_identity'] = bool(np.allclose(q0, occ, atol=1e-12))
    q = plant(occ, sex, 0.3)
    chk['plant_rows_sum_to_one'] = bool(np.allclose(q.sum(1), 1, atol=1e-12) and (q > 0).all())
    lr = np.log(q[:, 0] / q[:, 1]) - np.log(occ[:, 0] / occ[:, 1])
    chk['plant_direction_women_up_class0'] = bool(np.allclose(lr[sex == 1], 0.3) and np.allclose(lr[sex == 0], -0.3))
    roles = DR.roles_of(unit, 'R1')
    cal = calibrate_eta(rel, 'occupation', sex, roles, DR.reader_seed(0, 0), 0.010)
    chk['eta_calibration_converged'] = bool(cal['converged'] and abs(cal['gain'] - 0.010) <= L2_TOL)
    ext = extra_split_shift(rel, with_view(rel, 'occupation', plant(occ, sex, cal['eta'])), 'occupation', sex, unit, DR.reader_seed(0, 0), n=3)
    chk['extra_splits_positive_shift'] = ext['mean'] > 0
    # 5. simulation machinery: extremes, and the multiplicity generator equals L1 for one comparison (same law)
    lo = l1_single('income', -0.08, 0.76, 2, 300, 5); hi = l1_single('income', 0.08, 0.76, 2, 300, 6)
    chk['l1_extremes'] = lo['old_pass_rate'] > 0.95 and lo['agg_pass_rate'] > 0.95 and hi['old_pass_rate'] < 0.05 and hi['agg_pass_rate'] < 0.05
    ref = l1_single('income', 0.0, 0.76, 2, 4000, 11)
    rng2 = np.random.default_rng(12)
    g = a = 0
    for _ in range(4000):
        d = multi_draw({'income': [0.0], 'occupation': [0.0]}, rng2, 0.76, 2)
        cells1, ses1 = d[0][2], d[0][3]
        g += RU.per_cell_gate(cells1)['pass']; a += RU.aggregate_paired_screen(cells1, ses1, RU.PHI_PRIOR['income'])['pass']
    tol = 4 * math.sqrt(0.25 / 4000) * math.sqrt(2)
    chk['multiplicity_generator_matches_L1_one_comparison'] = abs(g / 4000 - ref['old_pass_rate']) < tol and abs(a / 4000 - ref['agg_pass_rate']) < tol
    chk['multiplicity_runs'] = 0.0 <= l1_multi('ALL_0', 0.76, 2, 50, 13)['agg_pass_all_rate'] <= 1.0 and \
        0.0 <= l1_multi('ALL_0', 0.76, 2, 50, 14, structure='h3_actual')['agg_pass_all_rate'] <= 1.0
    chk['pattern_parse'] = pattern_mu('IUT_occupation_+0.020_others_-0.010') == {'income': [-0.01] * 6, 'occupation': [0.02] + [-0.01] * 5}
    # reviewer's invented-data check (RULES.md section 10): gate false failure at mu = 0, independent seeds, income components
    ff = 1 - l1_single('income', 0.0, 0.0, 2, 4000, 15)['old_pass_rate']
    chk['reviewer_check_gate_false_failure_0807'] = abs(ff - 0.807) < 0.03
    # 6. report writer on invented parts
    tmp = Path(tempfile.mkdtemp(prefix='cts_screen_selftest_'))
    src = {'configurations': {'CAND': {**sc, 'source_label': 'X', 'old_reproduces_source': True}},
           'what_if': {'CAND': {'source_label': 'X', 'what_if_label_AGG1': 'Y'}}}
    _csv(tmp / 'T.csv', table_rows(src), TABLE_FIELDS)
    chk['csv_written'] = len((tmp / 'T.csv').read_text().splitlines()) == 1 + 12
    keys = [k for k, v in chk.items() if isinstance(v, bool)]
    chk['all_ok'] = all(chk[k] for k in keys)
    chk['checked'] = keys
    chk['details'] = {'eta_calibration': cal, 'L1_income_mu0_rho076': ref, 'gate_false_failure_rho0': ff}
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('source')
    sm = sub.add_parser('simulate'); sm.add_argument('--n-sim', type=int, default=N_SIM)
    for name in ('l0b', 'l2'):
        x = sub.add_parser(name); x.add_argument('--out', default=None)
    sub.add_parser('report')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        chk = selftest()
        C.write_json(C.RESULTS / 'roles' / 'reader_investigator' / 'checks' / 'SCREEN_STUDY_SELFTEST.json', {**chk, 'utc': C.now()})
        print(json.dumps(chk, indent=1, default=str))
        if not chk['all_ok']:
            raise SystemExit(1)
        return
    frozen_check()
    if a.cmd == 'source':
        rec = source_part()
        C.write_json(PRIV_OUT / 'SOURCE.json', rec)
        print(json.dumps({'what_if_outcome': rec['what_if_outcome'], 'old_reproduces_source': rec['old_gate_reproduces_source_all'],
                          'L0a': {v: (x['ordered_alias_pairs'], x['old_pass_rate'], x['agg_pass_rate']) for v, x in rec['L0a'].items()},
                          'cpu_s': rec['cpu_s']}, indent=1))
    elif a.cmd == 'simulate':
        rec = simulate_part(n_sim=a.n_sim)
        C.write_json(PRIV_OUT / 'SIM.json', rec)
        print(json.dumps({'settings': len(rec['L1_single']) + len(rec['L1_multiplicity']) + len(rec['L1_multiplicity_h3_actual_structure']),
                          'cpu_s': rec['cpu_s']}))
    elif a.cmd == 'l0b':
        rec = l0b_part(a.out)
        print(json.dumps({'stats': rec['stats'], 'cpu_s': rec['cpu_s'], 'by_view': {v: {k: (x['old_pass_rate'], x['agg_pass_rate']) for k, x in d.items()}
                                                                                for v, d in rec['screens']['by_view'].items()}}, indent=1))
    elif a.cmd == 'l2':
        rec = l2_part(a.out)
        print(json.dumps({'stats': rec['stats'], 'cpu_s': rec['cpu_s'], 'rows': len(rec['rows'])}))
    elif a.cmd == 'report':
        rec = report()
        print(json.dumps(rec['parts_present']))


if __name__ == '__main__':
    main()

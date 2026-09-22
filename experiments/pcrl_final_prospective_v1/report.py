"""Aggregate public report files from a completed run (no fitting, no selection)."""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from .audit_panel import roles_for, unit_dir
from .common import ANCHORS, CANDIDATES, OUT, PANEL, PRIMARY_ROLES, SECONDARY_COMPARATORS, TASK_ROLE, WEIGHTINGS, atomic_json, now, read_json

WKEY = {'unweighted': 'unweighted', 'PWGTP': 'weighted'}


def fmt(x, d=5):
    return '—' if x is None else f'{x:+.{d}f}'


def final_table(root, inf, absolute):
    desc = {r['id']: r for r in inf['descriptive']}
    rows = []
    for m in PANEL:
        for role in PRIMARY_ROLES+(TASK_ROLE,):
            for w in WEIGHTINGS:
                ce = [absolute[f'{m}|{a}|{role}']['ce'][WKEY[w]] for a in ANCHORS]
                row = {'family': m, 'frozen_release': PANEL[m], 'role': role, 'weighting': w,
                       'ce_mean_over_anchors': float(np.mean(ce)), 'ce_per_anchor': ce,
                       'rows_per_anchor': [absolute[f'{m}|{a}|{role}']['rows'] for a in ANCHORS],
                       'selection_per_anchor': [absolute[f'{m}|{a}|{role}']['selection'] for a in ANCHORS]}
                if role == TASK_ROLE:
                    fd = [absolute[f'{m}|{a}|{role}']['fixed_decoder_ce'] for a in ANCHORS]
                    ind = [absolute[f'{m}|{a}|{role}']['independent_ce'][WKEY[w]] for a in ANCHORS]
                    row['fixed_decoder_ce_mean'] = None if any(v is None for v in fd) else float(np.mean([v[WKEY[w]] for v in fd]))
                    row['independent_probe_ce_mean'] = float(np.mean(ind))
                    if m != 'H':
                        e = desc[f'descriptive|task_minus_H|{m}|{w}']
                        row['task_minus_H'] = e['estimate']; row['task_minus_H_ci95_unadjusted'] = [e['lower_unadjusted_95'], e['upper_unadjusted_95']]
                    if m not in ('H', 'J'):
                        e = desc[f'descriptive|task_minus_J|{m}|{w}']
                        row['task_minus_J'] = e['estimate']; row['task_minus_J_ci95_unadjusted'] = [e['lower_unadjusted_95'], e['upper_unadjusted_95']]
                elif m != 'H':
                    e = desc[f'descriptive|recovery_over_H|{m}|{role}|{w}']
                    row['recovery_over_H'] = e['estimate']
                    row['recovery_over_H_ci95_unadjusted'] = [e['lower_unadjusted_95'], e['upper_unadjusted_95']]
                    if m != 'J':
                        j = desc[f'descriptive|recovery_over_H|J|{role}|{w}']['estimate']
                        row['recovery_increment_over_J_point'] = e['estimate']-j
                rows.append(row)
    return rows


def competence(root):
    """Validation-only diagnostics of attacker/probe competence."""
    out = {'selection_counts': defaultdict(Counter), 'catchup_improves_validation': Counter(),
           'nonlinear_beats_logistic': Counter(), 'units': 0}
    val = {}
    for m in PANEL:
        for a in ANCHORS:
            for role in roles_for(m):
                reg = joblib.load(unit_dir(root, m, a, role)/'registry.joblib')
                s = reg['validation_scores']
                out['units'] += 1
                sel = reg['selection']
                fam = 'H_ancestor' if sel.startswith('H__') else 'A/B_ancestor' if sel.startswith('ancestor_') else \
                    'fixed_decoder' if sel == 'fixed_decoder' else 'H_recalibrator' if sel.startswith('global_offset') else sel
                out['selection_counts'][role][fam] += 1
                if 'mlp_360' in s and 'mlp_120' in s:
                    out['catchup_improves_validation'][role] += int(s['mlp_360']['balanced'] < s['mlp_120']['balanced'])
                nonlin = [k for k in ('mlp_120', 'mlp_360', 'hist_gb_20', 'hist_gb_5', 'sampled_hist_gb_20', 'sampled_hist_gb_5') if k in s]
                if 'logistic' in s and nonlin:
                    out['nonlinear_beats_logistic'][role] += int(min(s[k]['balanced'] for k in nonlin) < s['logistic']['balanced'])
                val[f'{m}|{a}|{role}'] = s[sel]
    # Validation recovery of controls over H (selected predictors, balanced CE).
    rec = {}
    for m in ('J', 'C', 'Q', 'D17', 'D33', 'E', 'S', 'RR75', 'W75'):
        for role in PRIMARY_ROLES:
            rec[f'{m}|{role}'] = float(np.mean([val[f'H|{a}|{role}']['balanced']-val[f'{m}|{a}|{role}']['balanced'] for a in ANCHORS]))
    out['validation_recovery_over_H_balanced'] = rec
    out['selection_counts'] = {k: dict(v) for k, v in out['selection_counts'].items()}
    out['catchup_improves_validation'] = dict(out['catchup_improves_validation'])
    out['nonlinear_beats_logistic'] = dict(out['nonlinear_beats_logistic'])
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def primary_md(inf):
    lines = []
    for m in CANDIDATES:
        p = inf['primary'][m]
        lines.append(f"### {m} ({PANEL[m]}): **{p['decision'].upper()}** — {p['clauses_passed']}/{p['clauses_total']} clauses pass\n")
        lines.append('| Clause | Weighting | Estimate | SE | One-sided 97.5% UB | Threshold | Status | Per-anchor estimates |')
        lines.append('|---|---|---:|---:|---:|---:|---|---|')
        for c in p['clauses']:
            lines.append(f"| {c['role']} | {c['weighting']} | {fmt(c.get('estimate'))} | {c.get('bootstrap_se', float('nan')):.5f} | "
                         f"{fmt(c.get('upper_bound_97_5'))} | {c['threshold']:+.3f} | {c['status']} | "
                         f"{', '.join(fmt(x) for x in (c.get('anchor_estimates') or []))} |")
        lines.append('')
    return '\n'.join(lines)


def secondary_md(inf):
    sec = inf['secondary']
    lines = [f"Family size {sec['family_size']} (generated), two-sided simultaneous 95% Bonferroni, z = {sec['z']:.4f}. "
             'Task rows: CE_Q − CE_comp. Sensitive rows: CE_comp − CE_Q = recovery(Q) − recovery(comp). Negative favors Q.\n',
             '| Comparator | Role | Weighting | Estimate | Simultaneous 95% interval | Excludes 0 |', '|---|---|---|---:|---|---|']
    for r in sec['rows']:
        lines.append(f"| {r['comparator']} | {r['role']} | {r['weighting']} | {fmt(r['estimate'])} | "
                     f"[{fmt(r['lower'])}, {fmt(r['upper'])}] | {'yes' if r['excludes_zero'] else 'no'} |")
    lines += ['', '| Comparator | Task non-inferior at .001 (both) | Task strictly better (both) | Task strictly worse (any) | '
              'No sensitive cost > .001 (all 8) | Strictly lower recovery | Strictly higher recovery |', '|---|---|---|---|---|---|---|']
    for c, v in inf['secondary_interpretation'].items():
        lines.append(f"| {c} | {v['task_noninferior_at_001_both_weightings']} | {v['task_strictly_better_both_weightings']} | "
                     f"{v['task_strictly_worse_any_weighting']} | {v['no_sensitive_cost_beyond_001_all_eight']} | "
                     f"{', '.join(v['strictly_lower_recovery_endpoints']) or 'none'} | {', '.join(v['strictly_higher_recovery_endpoints']) or 'none'} |")
    return '\n'.join(lines)


def per_anchor_md(table, inf):
    lines = ['# Per-anchor heterogeneity (anchors are not independent samples)\n',
             'Absolute expected log loss of the validation-selected predictor on the final pool, per anchor. '
             'The primary estimand is the equal mean of the three anchors.\n',
             '| Family | Role | Weighting | Anchor 0 | Anchor 1 | Anchor 2 | Mean |', '|---|---|---|---:|---:|---:|---:|']
    for r in table:
        a = r['ce_per_anchor']
        lines.append(f"| {r['family']} | {r['role']} | {r['weighting']} | {a[0]:.5f} | {a[1]:.5f} | {a[2]:.5f} | {r['ce_mean_over_anchors']:.5f} |")
    lines += ['', '## Primary clause estimates by anchor\n', primary_md(inf)]
    return '\n'.join(lines)


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inf = read_json(root/'INFERENCE.json')
    absolute = read_json(root/'ABSOLUTE_SCORES.json')
    table = final_table(root, inf, absolute)
    atomic_json(out_dir/'FINAL_TABLE.json', {'created_utc': now(), 'rows': table,
                'note': 'absolute CE = equal mean of anchor-level expected log loss; recovery_over_H = CE_H - CE_M (not clamped); unadjusted intervals are descriptive'})
    write_csv(out_dir/'FINAL_TABLE.csv', table)
    comp = competence(root)
    atomic_json(out_dir/'ATTACK_COMPETENCE.json', comp)
    atomic_json(out_dir/'SECONDARY_COMPARISONS.json', {'family': inf['secondary'], 'interpretation': inf['secondary_interpretation']})
    (out_dir/'SECONDARY_COMPARISONS.md').write_text('# Secondary comparisons (separate family)\n\n'+secondary_md(inf)+'\n')
    (out_dir/'PER_ANCHOR.md').write_text(per_anchor_md(table, inf)+'\n')
    (out_dir/'PRIMARY_CLAIMS_RESULTS.md').write_text('# Primary claims\n\n'+primary_md(inf)+'\n')
    return inf, table, comp

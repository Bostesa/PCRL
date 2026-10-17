"""Presentation of frozen records (no new computation on labels): UTILITY_TABLE.csv, RECOVERY_TABLE.csv, PRIMARY_ENDPOINTS.csv,
SELECTION_AND_MATCHING.md and UTILITY_RECOVERY_FRONTIER.png, from SELECTION_LOCK.json, TRAJECTORY_SUMMARY.json and
report/DEV_REPORT_RESULTS.json.

The figure keeps the two kinds of measurement apart: the top row shows MODEL_SELECT cross-fitted reader PROXIES for every
registered checkpoint (lines join a trajectory's checkpoints in merge order as an illustration only; no interpolated release
exists); the bottom row shows FRESH DEV_REPORT reader results for the final evaluation bank only.
"""
from __future__ import annotations
import csv
import json
import shutil

import numpy as np

from . import common as C
from .trajectory import TRAJ_NAMES

COL = {'SCM': '#2a78d6', 'ORD': '#eb6834', 'IBSI': '#1baf7a', 'SHUF': '#9a9993', 'FROZEN': '#0b0b0b'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ROLE_FAMILY = {'SCM': 'SCM', 'HIST_SCM': 'SCM', 'ORD_NLLM': 'ORD', 'ORD_CNT': 'ORD', 'IBSI_NLLM': 'IBSI', 'IBSI_CNT': 'IBSI', 'SHUF_NLLM': 'SHUF',
               'NEURAL_RAW': 'FROZEN', 'H_LJ': 'FROZEN', 'C0': 'FROZEN'}


def family(name):
    return 'SCM' if name == 'SCM' else 'ORD' if name == 'ORD' else 'IBSI' if name.startswith('IBSI') else 'SHUF'


def main():
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    tsum = json.loads((C.RESULTS / 'TRAJECTORY_SUMMARY.json').read_text())
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    UT = [(t, r, m) for t in C.TASKS for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]
    # ---------------- UTILITY_TABLE.csv
    with open(C.RESULTS / 'UTILITY_TABLE.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['measurement_role', 'seed', 'trajectory_or_role', 'checkpoint_labels', 'release', 'nominal_cells', 'fitted_occupation_nll_F',
                    'occupation_nll_expenditure_vs_C0'] + [f'{t}|{r}|{m}' for t, r, m in UT] +
                   ['point_pass_12', 'subgroup_occ_brier_excess_vs_U0_sex0', 'subgroup_occ_brier_excess_vs_U0_sex1', 'occupation_nll_sex0', 'occupation_nll_sex1',
                    'occupation_brier_sex0', 'occupation_brier_sex1', 'pair_reader_proxy_or_fresh_auc'])
        for s in C.SEEDS:
            per = sel['per_seed_measurements'][str(s)]; cps = tsum['per_seed'][f'S{s}']['checkpoints']; rel = tsum['per_seed'][f'S{s}']['releases']
            for name in TRAJ_NAMES:
                seen = {}
                for label, cp in cps[name].items():
                    if cp['release'] is not None:
                        seen.setdefault(cp['release'], []).append(label)
                for rid, labels in sorted(seen.items(), key=lambda x: cps[name][x[1][0]]['m']):
                    m = per[rid]; cl = {f"{c['task']}|{c['reference']}|{c['metric']}": c['value'] for c in m['clauses']}
                    w.writerow(['MODEL_SELECT', s, name, ';'.join(labels), rid, rel[rid]['live_cells'], rel[rid]['fitted_occupation_nll_F'],
                                m['occupation_nll_expenditure_vs_C0']] + [cl[f'{t}|{r}|{mm}'] for t, r, mm in UT] +
                               [m['point_pass_12'], m['subgroup_occupation_brier_excess_vs_U0']['sex0'], m['subgroup_occupation_brier_excess_vs_U0']['sex1'],
                                m['by_sex']['sex0']['occupation_nll'], m['by_sex']['sex1']['occupation_nll'], m['by_sex']['sex0']['occupation_brier'],
                                m['by_sex']['sex1']['occupation_brier'], m['proxy']['arm_proxy']])
            for name in ('NEURAL_RAW', 'H_LJ'):
                m = sel['frozen_measurements'][str(s)][name]; cl = {f"{c['task']}|{c['reference']}|{c['metric']}": c['value'] for c in m['clauses']}
                w.writerow(['MODEL_SELECT', s, name, 'frozen', f'FROZEN_{name}', '', '', m['occupation_nll_expenditure_vs_C0']] + [cl[f'{t}|{r}|{mm}'] for t, r, mm in UT] +
                           [m['point_pass_12'], m['subgroup_occupation_brier_excess_vs_U0']['sex0'], m['subgroup_occupation_brier_excess_vs_U0']['sex1'],
                            m['by_sex']['sex0']['occupation_nll'], m['by_sex']['sex1']['occupation_nll'], m['by_sex']['sex0']['occupation_brier'],
                            m['by_sex']['sex1']['occupation_brier'], m['proxy']['arm_proxy']])
            ps = rep['per_seed'][f'S{s}']; c0 = ps['service_of_role']['C0']
            for svc, v in ps['services'].items():
                if svc in ('U0', 'Ucal'):
                    continue
                roles = ps['aliases'].get(svc, [svc])
                rid = sel['roles'][str(s)][roles[0]].get('release')
                exp = v['task_summary']['occupation']['nll'] - ps['services'][c0]['task_summary']['occupation']['nll']
                w.writerow(['DEV_REPORT', s, '+'.join(roles), 'final bank', rid, '', '', exp] + [v['utility'][f'{t}|{r}|{mm}']['point'] for t, r, mm in UT] +
                           [v['utility_point_feasible_12_this_seed'], v['subgroup']['sex0']['occupation_brier_excess_vs_U0'], v['subgroup']['sex1']['occupation_brier_excess_vs_U0'],
                            v['by_sex']['occupation']['sex0']['nll'], v['by_sex']['occupation']['sex1']['nll'], v['by_sex']['occupation']['sex0']['brier'],
                            v['by_sex']['occupation']['sex1']['brier'], v['recovery']['pair']['auc']])
    # ---------------- RECOVERY_TABLE.csv
    with open(C.RESULTS / 'RECOVERY_TABLE.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['seed', 'service', 'roles', 'role_status', 'release', 'cells', 'MODEL_SELECT_proxy', 'MODEL_SELECT_occ_nll_expenditure', 'MODEL_SELECT_delta_vs_SCM']
                   + [f'DEV_{v}_{x}' for v in ('pair', 'income', 'occupation') for x in ('auc', 'se', 'winner')] + ['DEV_pair_auc_slate7', 'DEV_occ_nll_expenditure_vs_C0',
                                                                                                                     'DEV_occ_nll_minus_SCM', 'DEV_point_feasible_12'])
        for s in C.SEEDS:
            ps = rep['per_seed'][f'S{s}']; c0 = ps['service_of_role']['C0']; scm = ps['service_of_role']['SCM']
            for svc, roles in ps['aliases'].items():
                v = ps['services'][svc]; r0 = sel['roles'][str(s)][roles[0]]; rc = v['recovery']
                w.writerow([s, svc, '+'.join(roles), r0.get('status'), r0.get('release'), r0.get('cells'), r0.get('proxy'), r0.get('expenditure', ''), r0.get('delta')]
                           + [x for vv in ('pair', 'income', 'occupation') for x in (rc[vv]['auc'], rc[vv]['auc_se'], rc[vv]['winner'])]
                           + [v['recovery_slate7']['pair']['auc'], v['task_summary']['occupation']['nll'] - ps['services'][c0]['task_summary']['occupation']['nll'],
                              v['task_summary']['occupation']['nll'] - ps['services'][scm]['task_summary']['occupation']['nll'], v['utility_point_feasible_12_this_seed']])
    shutil.copyfile(C.RESULTS / 'report' / 'PRIMARY_ENDPOINTS.csv', C.RESULTS / 'PRIMARY_ENDPOINTS.csv')
    figure(sel, tsum, rep)
    selection_md(sel)


def figure(sel, tsum, rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    for j, s in enumerate(C.SEEDS):
        ax = axes[0, j]; per = sel['per_seed_measurements'][str(s)]; cps = tsum['per_seed'][f'S{s}']['checkpoints']
        for name in TRAJ_NAMES:
            pts = {}
            for label, cp in cps[name].items():
                if cp['release'] is not None:
                    pts[cp['m']] = (per[cp['release']]['occupation_nll_expenditure_vs_C0'], per[cp['release']]['proxy']['arm_proxy'])
            ms = sorted(pts)
            xs, ys = [pts[m][0] for m in ms], [pts[m][1] for m in ms]
            fam = family(name); col = COL[fam]
            ls = {'IBSI_g0.5': ':', 'IBSI_g1.0': '-', 'IBSI_g2.0': '--'}.get(name, '-')
            ax.plot(xs, ys, ls, color=col, lw=2 if fam in ('SCM', 'ORD') else 1.2, alpha=1 if fam != 'SHUF' else 0.7, zorder=3 if fam in ('SCM', 'ORD') else 2,
                    marker='o', ms=3.5, markeredgecolor='white', markeredgewidth=0.5,
                    label=(name if fam in ('SCM', 'ORD', 'IBSI') else ('SHUF_r0..r4' if name == 'SHUF_r0' else None)))
        roles = sel['roles'][str(s)]
        for rn, mk in (('SCM', '*'), ('ORD_NLLM', 'D'), ('ORD_CNT', 's'), ('IBSI_NLLM', 'D'), ('SHUF_NLLM', 'D')):
            r = roles.get(rn, {})
            if r.get('release') and r['release'] in per:
                x, y = per[r['release']]['occupation_nll_expenditure_vs_C0'], per[r['release']]['proxy']['arm_proxy']
                ax.plot(x, y, mk, ms=11 if mk == '*' else 7, color=COL[ROLE_FAMILY[rn]], markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
                off = {'SCM': (8, -16), 'ORD_NLLM': (6, 7), 'ORD_CNT': (6, 7), 'IBSI_NLLM': (-62, -16), 'SHUF_NLLM': (8, -12)}[rn]
                ax.annotate(rn, (x, y), xytext=off, textcoords='offset points', fontsize=7.5, color=INK,
                            arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
        for name in ('NEURAL_RAW', 'H_LJ'):
            m = sel['frozen_measurements'][str(s)][name]
            ax.plot(m['occupation_nll_expenditure_vs_C0'], m['proxy']['arm_proxy'], 'X', ms=8, color=COL['FROZEN'], zorder=5)
            ax.annotate(name, (m['occupation_nll_expenditure_vs_C0'], m['proxy']['arm_proxy']), xytext=(-78, 14) if name == 'NEURAL_RAW' else (14, 12),
                        textcoords='offset points', fontsize=7.5, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
        e = sel['nomination']['e_star'][str(s)]
        ax.axvspan(e - 0.001, e + 0.001, color=COL['SCM'], alpha=0.08, lw=0, zorder=0)
        ax.set_title(f'MODEL_SELECT reader proxy (cross-fitted, max of 3 readers), seed {s}', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('MODEL_SELECT occupation-NLL expenditure vs C0 (nats)'); ax.set_ylabel('pair proxy AUC (lower = less SEX recovery)')
        ax.grid(color=GRID, lw=0.6); ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax2 = axes[1, j]; ps = rep['per_seed'][f'S{s}']; c0 = ps['service_of_role']['C0']
        for svc, rnames in ps['aliases'].items():
            v = ps['services'][svc]
            x = v['task_summary']['occupation']['nll'] - ps['services'][c0]['task_summary']['occupation']['nll']
            y, se = v['recovery']['pair']['auc'], v['recovery']['pair']['auc_se']
            fam = ROLE_FAMILY[rnames[0]]
            ax2.errorbar(x, y, yerr=1.96 * se, fmt='o' if fam != 'FROZEN' else 'X', ms=7, color=COL[fam], ecolor=COL[fam], elinewidth=1.2, capsize=0,
                         markeredgecolor=INK if fam != 'FROZEN' else COL[fam], markeredgewidth=0.6, zorder=4)
            off = {'SCM': (8, -12), 'IBSI_CNT': (8, 6), 'IBSI_NLLM': (8, 6), 'SHUF_NLLM': (8, -12)}.get(rnames[0], (8, 3))
            ax2.annotate('+'.join(rnames), (x, y), xytext=off, textcoords='offset points', fontsize=7.2, color=INK)
        ax2.set_title(f'DEV_REPORT fresh readers (8-reader slate, ±1.96 SE), seed {s}', fontsize=9, color=INK, loc='left')
        ax2.set_xlabel('DEV_REPORT occupation-NLL expenditure vs C0 (nats)'); ax2.set_ylabel('pair AUC, selected fresh reader')
        ax2.grid(color=GRID, lw=0.6)
    fig.suptitle('Utility-recovery frontier: MODEL_SELECT proxies (top; every registered checkpoint) and fresh DEV_REPORT readers (bottom; final bank only)\n'
                 'Lines join one trajectory\'s checkpoints in merge order for illustration only; shaded band: ±0.001 nats around the SCM nominee\'s expenditure',
                 fontsize=9.5, color=INK)
    fig.savefig(C.RESULTS / 'UTILITY_RECOVERY_FRONTIER.png', dpi=160, facecolor='#fcfcfb')


def selection_md(sel):
    nom = sel['nomination']; L = []
    L.append('# Selection and matching (MODEL_SELECT only; frozen in SELECTION_LOCK.json)\n')
    L.append('"MATCHED" below is the MODEL_SELECT role status. None of these matches carried over to DEV_REPORT (`RESEARCH_DECISION.md` §6).\n')
    L.append(f"**Status:** `{sel['status']}`; common threshold t* = {nom['t_star']}; cells K* = {nom['K_star']['0']} / {nom['K_star']['1']}; expenditure e* = "
             + ', '.join(f"{float(v):.5f}" for v in nom['e_star'].values()) + ' nats (seed 0, seed 1).\n')
    L.append('## SCM nomination grid (both seeds; eligible = 24 clauses + occupation NLL excess ≤ 0.008 + both within-SEX guards)\n')
    L.append('| t | cells s0 / s1 | eligible | worst normalised | mean proxy | expenditure s0 / s1 | failed conditions |')
    L.append('|---|---|---|---|---|---|---|')
    for g in nom['grid']:
        fails = '; '.join(f"s{s}: {', '.join(v)}" for s, v in g['failed_conditions'].items() if v) or '—'
        L.append(f"| {g['t']:.3f} | {g['cells']['0']} / {g['cells']['1']} | {g['eligible']} | {g['worst_normalized']:.3f} | {g['mean_proxy']:.4f} | "
                 f"{g['expenditure']['0']:.5f} / {g['expenditure']['1']:.5f} | {fails} |")
    L.append('\n## Roles (per seed)\n')
    L.append('| seed | role | status | trajectory | labels | cells | release | Δ expenditure (comparator − SCM) | proxy | 12 clauses |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for s, rr in sel['roles'].items():
        for rn in sel['role_order']:
            r = rr.get(rn, {})
            d = r.get('delta'); p = r.get('proxy')
            L.append(f"| {s} | {rn} | {r.get('status')} | {r.get('trajectory', '—')} | {','.join(r.get('labels', []) or [])} | {r.get('cells', '—')} | {r.get('release')} | "
                     f"{'' if d is None else f'{d:+.5f}'} | {'' if p is None else f'{p:.4f}'} | {r.get('point_pass_12', '—')} |")
    m = sel['matching']
    L.append('\n## Matching outcome\n')
    L.append('- ORD_NLLM available (MODEL_SELECT match within 0.001 and 12 clauses): seed 0 ' + str(m['ORD_NLLM_available']['0']) + ', seed 1 ' + str(m['ORD_NLLM_available']['1']))
    L.append(f"- IBSI_NLLM: trajectory {m['IBSI_NLLM']['gamma_trajectory']}, matched in both seeds: {m['IBSI_NLLM']['available_both_seeds']}, admissible: {', '.join(m['IBSI_NLLM']['admissible'])}")
    L.append(f"- SHUF_NLLM: trajectory {m['SHUF_NLLM']['r_trajectory']}, matched in both seeds: {m['SHUF_NLLM']['available_both_seeds']}, admissible: {', '.join(m['SHUF_NLLM']['admissible'])}")
    L.append(f"- ORD_CNT available at exactly K* cells in both seeds: {m['ORD_CNT_available_both_seeds']}")
    L.append(f"- IBSI_CNT: {m['IBSI_CNT']['trajectory']} (24-clause eligible: {m['IBSI_CNT']['eligible_24']})")
    L.append('\n## Selection effort (distinct candidate releases examined)\n')
    L.append('```\n' + json.dumps(sel['selection_effort'], indent=1) + '\n```')
    L.append('\n## Candidates nearest the SCM expenditure (per comparator trajectory and seed)\n')
    L.append('| seed | trajectory | labels | cells | Δ expenditure (candidate − SCM) | match | 12 clauses | proxy |')
    L.append('|---|---|---|---|---|---|---|---|')
    for name, per in m['candidates'].items():
        for s, cands in per.items():
            for c in sorted(cands, key=lambda c: c['abs_delta'])[:3]:
                L.append(f"| {s} | {name} | {','.join(c['labels'])} | {c['cells']} | {c['delta']:+.5f} | {c['match']} | {c['point_pass_12']} | {c['proxy']:.4f} |")
    (C.RESULTS / 'SELECTION_AND_MATCHING.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()

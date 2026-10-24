"""Anchor table of the strong-anchor release campaign (coordinator; LABEL-FREE: reads only saved records of the input-and-pair
release campaign, no label file and no new reader fit).

    python3 -m sar.anchors table        -> results/pcrl_strong_anchor_release_v1/anchors/ANCHOR_TABLE.json and .md

Every recovery number comes from ONE reader bank, the input-and-pair campaign's registered grid IPR1_MAIN (8 registered cells:
2 versions x repetitions R1, R2 x reader seeds k0, k1; the admitted 16-recipe bank; validation-selected winner SCORE AUC on
MODEL_SELECT), so the rows are comparable. Paired differences use that grid's stored household-bootstrap replicates (same weights
for both releases inside a cell) and the inherited aggregate (nrm.rules.aggregate: block means over version x repetition,
se_agg = sqrt(se_hh^2 + se_rs^2 + se_mc^2), floor 0.001). Utility comes from the same campaign's MODEL_SELECT evaluation records
(EVAL_MS.json; 12 task clauses + 2 within-SEX occupation guards per version, the original caps). Provenance, exposure, selection
history and known reader sensitivity are quoted from the cited source records (curated below; nothing here re-derives them).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from nrm import rules as NR

ROOT = Path(__file__).resolve().parent.parent
IPR_PRIV = ROOT / '.private' / 'pcrl_input_and_pair_release_v1' / 'runs'
IPR_RES = ROOT / 'results' / 'pcrl_input_and_pair_release_v1'
OUT = ROOT / 'results' / 'pcrl_strong_anchor_release_v1' / 'anchors'
BANK = 'IPR1_MAIN'
VIEWS = ('pair', 'income', 'occupation')
ROWS = ('RETIRED_B3CTL', 'B2TWIN', 'SUCC', 'H_LJ', 'DDR_PTC_R1', 'NEURAL_RAW', 'TWJ', 'W', 'TT100', 'TR100', 'U0', 'Ucal', 'CTL_TWIN')
VERSUS = ('NEURAL_RAW', 'H_LJ', 'U0')

# Curated provenance (sources: results/pcrl_correction_transfer_stability_v1/EXPOSURE_MAP.md section 1; ipr EXPOSURE_MAP.md;
# the START_HERE / RESEARCH_DECISION of the named campaigns). 'package' = raw-input deployable artefact known to exist locally.
PROV = {
    'RETIRED_B3CTL': {
        'contract': 'income: 6 decision-aligned levels of 0.5 LOCAL_TREE + 0.5 JOINT_TREE income log-odds (3 quantile bins per side of '
                    '0, MF thresholds), per-level probabilities from an L2 logistic map h fitted on all CALIBRATION; occupation: JOINT_TREE '
                    'E20 (bitwise the H_LJ / SUCC occupation output). 6 distinct income vectors.',
        'exposure': 'JOINT / LOCAL: MF income, occupation, SEX of all 20,000 (in-sample); h: CAL income (all 2,982); the 0.5 mix chosen '
                    'on a CAL screen; uncoupled twin and finalist promotion were adaptive steps on MS development proxies.',
        'selection': 'autonomous-method-development finalist B3CTL_UNCOUPLED_MIX50_C6A (adaptive, MS proxies); frozen-finalist '
                     'validation reproduced it exactly and RETIRED it as a competitive candidate (2026-10-17).',
        'reader_sensitivity': 'READER_SENSITIVE (frozen-finalist validation, enhanced bank: claim B fails on the seed-1 income clause vs '
                              'H_LJ through H_LJ\'s unstable income winner); DEV benefits +0.027/+0.022 vs NEURAL_RAW, +0.028/+0.026 vs '
                              'H_LJ, none above 0.020 at its clause bounds.',
        'package': 'amd/ffv deployment packages (B3CTL; admitted copy under the ihs mirror admitted/amd_deploy); MS wire admitted by hash.',
        'limiting': 'utility slack ~0: the binding income Brier clauses sit 0.03-0.04 SE below the cap on DEV and the men\'s occupation '
                    'guard 0.32 SE (v0); confirmation infeasible (frozen-finalist validation). Coarse 6-level income.'},
    'B2TWIN': {
        'contract': 'income: 6-level aligned code of LOCAL_TREE income log-odds with CAL-fitted per-level probabilities; occupation: '
                    'JOINT_TREE E20. 6 distinct income vectors.',
        'exposure': 'as RETIRED_B3CTL (LOCAL only in the code); h on all CAL; the uncoupled twin was an adaptive step on MS proxies.',
        'selection': 'task-only-code twin of the B3CTL finalist in the autonomous-method-development campaign (a control, adaptive).',
        'reader_sensitivity': 'not separately established; a 6-level code view (few distinct inputs: reader seeds partly effective).',
        'package': 'admitted MS / MF wires (ihs mirror admitted/amd_runs); raw-input package via the amd code path.',
        'limiting': 'the shared JOINT occupation output\'s men\'s occupation Brier guard (reserve 0.32 / 0.62 SE).'},
    'SUCC': {
        'contract': 'income: H3_STRONG_LJDIR_PB = Platt(a (s_B + D_L + beta D_J) + b), s_B the HGB + T_TREE stacked task-only score, '
                    'D_L / D_J the LOCAL / JOINT leaf-correction directions, beta 0.25 (v0) / 0.20 (v1) chosen on CAL; occupation: '
                    'JOINT_TREE E20 (leaf-score corrections on the frozen T_TREE CatBoost routing, trained with income, occupation '
                    'and linked-pair SEX critics). Continuous probabilities on both outputs.',
        'exposure': 'MF income, occupation and SEX of all 20,000 (JOINT, D_L, D_J, in-sample); HGB / stack / beta / Platt: MF and CAL '
                    'income; H3 nominated on MS by its source rule and evaluated once on DEV_REPORT.',
        'selection': 'the income-headroom successor (H3) composed with JOINT occupation by the occupation-headroom campaign ("the source '
                     'successor"); fixed composition, not tuned here.',
        'reader_sensitivity': 'income-headroom successor: P (pair) claim fails and reader-sensitive; occupation binds (ihs decision).',
        'package': 'H3 income (ihs mirror runs/H/H3_STRONG_LJDIR_PB) + JOINT occupation (acs_replication STAGE_A routing + STAGE_B '
                   'JOINT_TREE__E20.npz); both stored locally.',
        'limiting': 'men\'s within-SEX occupation summed-Brier excess vs U0 (JOINT occupation output) at 0.883 (v0) / 0.788 (v1) of '
                    'the 0.005 cap, reserve 0.32 / 0.62 SE, while the women\'s guard has wide slack; the pair (0.607) sits 0.034 above '
                    'its occupation view (0.573): the H3 income vector was never jointly protected with JOINT occupation.'},
    'H_LJ': {
        'contract': 'income: LOCAL_TREE E20 (leaf-score corrections on the T_TREE routing with own-wire SEX critics); occupation: '
                    'JOINT_TREE E20.',
        'exposure': 'MF income, occupation, SEX of all 20,000 (in-sample); fixed composition chosen after its components\' DEV results.',
        'selection': 'frozen-hybrids preselected primary (H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED); a material-target reference.',
        'reader_sensitivity': 'its income winner is unstable (frozen-finalist validation: chosen on an ATTACK_VALIDATION near tie).',
        'package': 'acs_replication STAGE_A/STAGE_B (LOCAL_TREE / JOINT_TREE E20) and deploy/LOCAL_TREE_S<v>.',
        'limiting': 'the shared JOINT occupation men\'s guard (0.32 / 0.62 SE); pair 0.619 vs occupation view 0.573.'},
    'NEURAL_RAW': {
        'contract': 'both outputs from the U0 network (d-64-64-16-K per task) trained from U0 with the joint objective (same online '
                    'critics, row draws and calibration as the tree arms; Adam 1e-4, clip 5), rho 1, checkpoint E2. Continuous.',
        'exposure': 'MF task labels and SEX of all 20,000 (in-sample); rho / checkpoint chosen on MS by the replication\'s registered '
                    'rule (post-selection MS values).',
        'selection': 'the replication\'s eligible neural control; a material-target reference (frozen-hybrids claim B).',
        'reader_sensitivity': 'none recorded beyond the general reader-bank caveat.',
        'package': 'checkpoints NEURAL_RAW__rho{1,3,10}__E{0,2,5,10,20}.pt for both seeds (acs_replication runs/NEURAL_RAW_<v>); a '
                   'raw-input deploy package exists for S0 only.',
        'limiting': 'protection is weak on the occupation view (0.625 = its pair; the pair recovery is the occupation view); its stronger '
                    'grid points (rho 3, 10 / later epochs) were not chosen by the replication rule; reserve 1.95 / 1.56 (men\'s guard '
                    'v0, income Brier vs U0 v1) but point margins 0.34 / 0.57 of cap.'},
    'DDR_PTC_R1': {
        'contract': 'dual-design adaptive revision: HGB task teachers with a linear SEX-propensity tilt (closed idea family).',
        'exposure': 'MF income, occupation, SEX (all 20,000); adaptive revision chosen on the same MS cells (winner\'s curse).',
        'selection': 'DB_PTC_R1, the dual-design campaign\'s strongest release; never separated from its release-only tilt control.',
        'reader_sensitivity': 'not separated from DB_ZTILT_R1 (release-only).',
        'package': 'ddr deployment packages (local).',
        'limiting': 'idea family closed (propensity tilts, release-only tilts); not separated from its release-only control.'},
    'TWJ': {'contract': 'HC_STRONG (task-only) income + JOINT_TREE E20 occupation.', 'exposure': 'as HC_STRONG and JOINT.',
            'selection': 'occupation-headroom combined reference.', 'reader_sensitivity': '-', 'package': 'local (ihs mirror + Stage B).',
            'limiting': 'task-only income re-opens the pair (0.636).'},
    'W': {'contract': 'A1_PO_16 occupation witness + H3 income.', 'exposure': 'MF occupation (witness), as H3.',
          'selection': 'occupation-headroom witness reference.', 'reader_sensitivity': '-', 'package': 'local (ohn mirror).',
          'limiting': 'task-only occupation (0.644 occupation view).'},
    'TT100': {'contract': 'H3 income + T_TREE occupation.', 'exposure': 'MF; as H3.', 'selection': 'training-size reference.',
              'reader_sensitivity': '-', 'package': 'local.', 'limiting': 'task-only occupation.'},
    'TR100': {'contract': 'H3 income + TASK_REFIT occupation.', 'exposure': 'MF; as H3.', 'selection': 'training-size reference.',
              'reader_sensitivity': '-', 'package': 'local.', 'limiting': 'task-only occupation.'},
    'U0': {'contract': 'utility reference U0 (neural, task-only).', 'exposure': 'MF task labels.', 'selection': '-',
           'reader_sensitivity': '-', 'package': 'U0_MODEL.pt', 'limiting': 'utility reference, not a protection.'},
    'Ucal': {'contract': 'U0 with one CAL temperature per task.', 'exposure': 'MF, CAL task labels.', 'selection': '-',
             'reader_sensitivity': '-', 'package': 'U0 + temperatures', 'limiting': 'utility reference.'},
    'CTL_TWIN': {'contract': 'input-and-pair shared twin: cached HGB task teachers (task-only).', 'exposure': 'MF task labels.',
                 'selection': 'the recent campaigns\' weak task-only start.', 'reader_sensitivity': '-', 'package': 'ipr deploy.',
                 'limiting': 'weak start (pair 0.648).'},
}


def load_bank():
    cells, boot = {}, {}
    for v in (0, 1):
        d = IPR_PRIV / 'GRID' / BANK / f'S{v}'
        g = json.loads((d / 'GRID.json').read_text())
        cells.update(g['cells'])
        with np.load(d / 'BOOT.npz') as z:
            for k in z.files:
                boot[k] = z[k].copy()
    return cells, boot


def level(cells, boot, m, view):
    e = {'cells': {}, 'cell_se': {}}
    for c in NR.REGISTERED:
        e['cells'][c] = float(cells[c][m][view]['score_auc'])
        e['cell_se'][c] = float(np.std(boot[f'{c}__{m}__{view}'], ddof=1))
    return NR.aggregate(e)


def paired(cells, boot, cand, comp):
    """Pair benefit = comp - cand (positive favours cand); single views r = cand - comp (the inherited sign conventions)."""
    out = {}
    for view in VIEWS:
        e = {'cells': {}, 'cell_se': {}}
        for c in NR.REGISTERED:
            a, b = cells[c][cand][view], cells[c][comp][view]
            ba, bb = boot[f'{c}__{cand}__{view}'], boot[f'{c}__{comp}__{view}']
            d, db = (b['score_auc'] - a['score_auc'], bb - ba) if view == 'pair' else (a['score_auc'] - b['score_auc'], ba - bb)
            e['cells'][c] = float(d); e['cell_se'][c] = float(np.std(db, ddof=1))
        out[view] = e
    return NR.comparison(out)


def utility_rows():
    e = json.loads((IPR_PRIV / 'REFERENCES' / 'EVAL_MS.json').read_text())['references']
    out = {}
    names = {'REF:B2TWIN': 'B2TWIN', 'REF:H_LJ': 'H_LJ', 'REF:NEURAL_RAW': 'NEURAL_RAW', 'REF:RETIRED_B3CTL': 'RETIRED_B3CTL',
             'REF:SUCC': 'SUCC', 'REF:TWJ': 'TWJ', 'REF:U0': 'U0', 'REF:Ucal': 'Ucal', 'W': 'W', 'REF:H3+T_TREE': 'TT100',
             'REF:H3+TASK_REFIT': 'TR100'}
    for k, n in names.items():
        out[n] = e[k]
    for n, p in (('DDR_PTC_R1', IPR_PRIV / 'SOURCE_DDR' / 'DB' / 'DB_PTC_R1' / 'EVAL_MS.json'),
                 ('CTL_TWIN', IPR_PRIV / 'CTL' / 'CTL_TWIN' / 'EVAL_MS.json')):
        out[n] = json.loads(p.read_text())
    return out


def util_summary(rec):
    """Per version: point status, max value / cap, binding conditions, guards (value, se, margin in SE), every clause."""
    out = {}
    for v in ('0', '1'):
        r = (rec.get('versions') or rec).get(v) if isinstance(rec, dict) else None
        if r is None:
            out[v] = None; continue
        u = r['utility']
        cl = {f"{c['task']}|{c['reference']}|{c['metric']}": {'value': c['value'], 'cap': c['cap'], 'se': c.get('se'),
                                                             'margin_se': c.get('margin_se_units')} for c in u['clauses']}
        gd = {g['group']: {'value': g['value'], 'cap': g['cap'], 'se': g.get('se'), 'margin_se': g.get('margin_se_units')}
              for g in u['guards']}
        out[v] = {'point_pass_28_of_28': bool(r['point']['point_pass_12_plus_2']), 'max_value_over_cap': r['point']['max_value_over_cap'],
                  'failed': r['point'].get('failed'), 'reserve': r.get('reserve'), 'guards': gd, 'clauses': cl,
                  'distinct_vectors': r.get('distinct_vectors')}
    return out


def table():
    cells, boot = load_bank()
    util = utility_rows()
    rows = {}
    for m in ROWS:
        lv = {view: level(cells, boot, m, view) for view in VIEWS}
        cmp_ = {}
        for comp in VERSUS:
            if comp != m:
                c = paired(cells, boot, m, comp)
                cmp_[comp] = {'pair_benefit': c['benefit']['aggregate']['mean'], 'se_agg': c['benefit']['aggregate']['se_agg'],
                              'label': c['benefit']['label'],
                              'v_means': {str(v): c['benefit']['aggregate']['per_version'][v]['mean'] for v in (0, 1)},
                              'recipient': {vw: c['recipient'][vw]['label'] for vw in ('income', 'occupation')},
                              'target_0020': NR.target_screen(_entry(cells, boot, m, comp))['label']}
        rows[m] = {'auc': {view: {'mean': lv[view]['mean'], 'se_agg': lv[view]['se_agg'],
                                  'v_means': {str(v): lv[view]['per_version'][v]['mean'] for v in (0, 1)}} for view in VIEWS},
                   'pair_minus_max_single': lv['pair']['mean'] - max(lv['income']['mean'], lv['occupation']['mean']),
                   'versus': cmp_, 'utility': util_summary(util[m]) if m in util else None, 'provenance': PROV.get(m)}
    rec = {'schema': 'SAR1-anchor-table-v1', 'bank': BANK, 'cells': list(NR.REGISTERED), 'labels_read': False,
           'sources': {'grid': f'.private/pcrl_input_and_pair_release_v1/runs/GRID/{BANK}/S<v>/GRID.json + BOOT.npz',
                       'utility': '.private/pcrl_input_and_pair_release_v1/runs/REFERENCES/EVAL_MS.json (+ SOURCE_DDR, CTL)',
                       'aggregate': 'nrm.rules.aggregate / comparison / target_screen (unchanged)'},
           'rows': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ANCHOR_TABLE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    (OUT / 'ANCHOR_TABLE.md').write_text(render(rec))
    print(render(rec))


def _entry(cells, boot, cand, comp):
    e = {'cells': {}, 'cell_se': {}}
    for c in NR.REGISTERED:
        d = cells[c][comp]['pair']['score_auc'] - cells[c][cand]['pair']['score_auc']
        db = boot[f'{c}__{comp}__pair'] - boot[f'{c}__{cand}__pair']
        e['cells'][c] = float(d); e['cell_se'][c] = float(np.std(db, ddof=1))
    return e


def _f(x, n=4):
    return '-' if x is None else f'{x:+.{n}f}' if isinstance(x, float) and x < 0 else f'{x:.{n}f}'


def render(rec):
    R = rec['rows']
    L = ['# Anchor table (campaign `pcrl_strong_anchor_release_v1`; label-free, from saved records)', '',
         f"One reader bank: {rec['bank']} (the input-and-pair campaign's registered grid; 8 registered cells, admitted 16-recipe bank, "
         'validation-selected winner SCORE AUC on the 8,001 MODEL_SELECT people). Utility: that campaign\'s MODEL_SELECT records, 12 task '
         'clauses + 2 within-SEX occupation guards per version (24 + 4 = 28), original caps (accuracy loss 0.010, NLL excess 0.010, summed '
         'Brier excess 0.005, guards 0.005 vs U0). Development evidence on adaptively reused people; lower recovery for these readers is '
         'not protection against every reader.', '',
         '## 1. Recovery (AUC, mean (se_agg) over the 8 cells; per-version means in the JSON)', '',
         '| Release | Pair | Income view | Occupation view | Pair - max single view |', '|---|---|---|---|---|']
    for m, r in R.items():
        a = r['auc']
        L.append(f"| {m} | {a['pair']['mean']:.4f} ({a['pair']['se_agg']:.4f}) | {a['income']['mean']:.4f} ({a['income']['se_agg']:.4f}) | "
                 f"{a['occupation']['mean']:.4f} ({a['occupation']['se_agg']:.4f}) | {_f(r['pair_minus_max_single'])} |")
    L += ['', '## 2. Paired pair benefit against the material-target references (benefit = comparator AUC - release AUC; positive '
          'favours the release; se_agg; label of the inherited screen; 0.020 target screen)', '',
          '| Release | vs NEURAL_RAW | vs H_LJ | vs U0 |', '|---|---|---|---|']
    for m, r in R.items():
        cs = []
        for comp in VERSUS:
            c = r['versus'].get(comp)
            cs.append('-' if c is None else f"{_f(c['pair_benefit'])} ({c['se_agg']:.4f}) {c['label']}; target {c['target_0020']}")
        L.append(f"| {m} | " + ' | '.join(cs) + ' |')
    L += ['', '## 3. Utility (per version v0 / v1: max value / cap over the 28; binding condition; whole and occupation-side '
          'two-SE reserve ratio; men\'s and women\'s occupation Brier excess vs U0 (cap 0.005))', '',
          '| Release | max/cap v0 / v1 | whole binding v0 / v1 | reserve whole v0 / v1 | men\'s guard v0 / v1 (se) | women\'s guard v0 / v1 | '
          'income NLL vs U0 / Ucal v0 | occ NLL vs U0 / Ucal v0 | distinct income / occ vectors v0 |', '|---|---|---|---|---|---|---|---|---|']
    for m, r in R.items():
        u = r['utility']
        if not u or not u.get('0'):
            L.append(f'| {m} | - | - | - | - | - | - | - | - |'); continue
        u0, u1 = u['0'], u['1']

        def g(x, s):
            gg = x['guards'][s]
            return f"{gg['value']:+.4f} ({gg['se']:.4f})"
        rs = lambda x: x['reserve'] or {}
        L.append(f"| {m} | {u0['max_value_over_cap']:.3f} / {u1['max_value_over_cap']:.3f} | {rs(u0).get('whole_binding')} / "
                 f"{rs(u1).get('whole_binding')} | {rs(u0).get('min_whole_ratio', float('nan')):.2f} / {rs(u1).get('min_whole_ratio', float('nan')):.2f} | "
                 f"{g(u0, 'sex0')} / {g(u1, 'sex0')} | {g(u0, 'sex1')} / {g(u1, 'sex1')} | "
                 f"{u0['clauses']['income|U0|nll_excess']['value']:+.4f} / {u0['clauses']['income|Ucal|nll_excess']['value']:+.4f} | "
                 f"{u0['clauses']['occupation|U0|nll_excess']['value']:+.4f} / {u0['clauses']['occupation|Ucal|nll_excess']['value']:+.4f} | "
                 f"{(u0.get('distinct_vectors') or {}).get('income', '-')} / {(u0.get('distinct_vectors') or {}).get('occupation', '-')} |")
    L += ['', '## 4. Contract, exposure, selection history, reader sensitivity, package, limiting condition', '']
    for m, r in R.items():
        p = r['provenance'] or {}
        L += [f'### {m}', ''] + [f'- **{k}:** {p.get(k, "-")}' for k in ('contract', 'exposure', 'selection', 'reader_sensitivity',
                                                                          'package', 'limiting')] + ['']
    L += ['Point eligibility (all 28 at the point) is not confidence preservation (the reserve ratios), and neither is a competitive '
          'claim. A retired release can be a starting point or comparator without recovering its retired claim.', '']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    if sys.argv[1:] == ['table']:
        table()
    else:
        print(__doc__)

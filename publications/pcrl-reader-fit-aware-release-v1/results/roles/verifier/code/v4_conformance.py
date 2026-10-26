#!/usr/bin/env python3
"""RFA1 verifier, wave 4 (a): conformance of RESEARCH_DECISION.md, START_HERE.md and HANDOFF.md with the governing records (label-free).

Every numeric token on every line of the three documents is matched, in order, against a value formatted from a record (or declared a
literal: a section number, version index, date or count named in the text). Labels are checked against the records by explicit
statement checks, and every upper-snake token is required to occur in some record or file name. Also: TABLES.md / ENDPOINTS.json
against the final records, and the reviewer's REQUIRED items R1-R7 (roles/statistical_reviewer/HANDBACK_3.md) as text checks.

Sources: claims/RFA_v1_final, claims/RFA_R1_final (DECISIONS, FALSIFIER_DETAIL, FAMILY_CHANGE_vs_RF_FULL), claims/RFA_v1_D2R,
claims/RFA_R1_D2R, claims/CAMPAIGN_SUMMARY.json, claims/PER_RECIPE_PAIR_MEANS.json, the plans, fam/FIT_RF_*.json, PROTOCOL.md, the two
cards, PROTOCOL_DEVIATIONS.md, RECEIPTS.json, RESOURCE_ACCOUNTING.json, ACCESS_RECEIPT.json, custody/, deploy/; the saved grids'
aggregate SCORE AUCs (.private/.../runs/GRID/*/S*/GRID.json, reproduced exactly by this verifier in waves 2-3) and the reference
utility record (runs/REFERENCES/EVAL_MS.json, aggregate); the verifier's own outputs (out/V2_*.json, V3_MAIN.json, V4_*.json).
No label, array or model file is read.

    <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v4_conformance.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
PRIV = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1'
OUTD = HERE.parent / 'out'
OUT = OUTD / 'V4_CONFORMANCE.json'
sys.path.insert(0, str(ROOT))

DOCS = {'RD': 'RESEARCH_DECISION.md', 'SH': 'START_HERE.md', 'HO': 'HANDOFF.md'}
REG = [f'S{v}R{r}k{k}' for v in (0, 1) for r in (1, 2) for k in (0, 1)]
NUM = re.compile(r'(?<![A-Za-z_0-9.])[+-]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?(?![A-Za-z_0-9])')
LAB = re.compile(r'\b[A-Z][A-Z0-9]*(?:_[A-Za-z0-9]+)+\b')


def jl(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def s(x, nd):
    return f'{x:+.{nd}f}'


def u(x, nd):
    return f'{x:.{nd}f}'


def ci(n):
    return f'{int(n):,}'


def L(t, why='literal'):
    return ('LIT', t, why)


def R(t, why):
    return ('REC', t, why)


# ----------------------------------------------------------------------------------------------------------------------------- records
t0 = time.process_time()
C = RES / 'claims'
DV = jl(C / 'RFA_v1_final' / 'DECISIONS.json')
DR = jl(C / 'RFA_R1_final' / 'DECISIONS.json')
dv, dr = DV['decisions'], DR['decisions']
FV = jl(C / 'RFA_v1_final' / 'FALSIFIER_DETAIL.json')['detail']
FR = jl(C / 'RFA_R1_final' / 'FALSIFIER_DETAIL.json')['detail']
FC = jl(C / 'RFA_R1_final' / 'FAMILY_CHANGE_vs_RF_FULL.json')['reading']
XV = jl(C / 'RFA_v1_D2R' / 'D2R.json')
XR = jl(C / 'RFA_R1_D2R' / 'D2R.json')
xv, xr = XV['d2r']['per_comparator'], XR['d2r']['per_comparator']
CS = jl(C / 'CAMPAIGN_SUMMARY.json')
PRM = jl(C / 'PER_RECIPE_PAIR_MEANS.json')
PV, PR = jl(C / 'PLAN_RFA_v1.json'), jl(C / 'PLAN_RFA_R1.json')
FIT = {('v1', 0): jl(RES / 'fam' / 'FIT_RF_V0.json'), ('v1', 1): jl(RES / 'fam' / 'FIT_RF_V1.json'),
       ('R1', 0): jl(RES / 'fam' / 'FIT_RF_R1_V0.json'), ('R1', 1): jl(RES / 'fam' / 'FIT_RF_R1_V1.json')}
END = jl(RES / 'ENDPOINTS.json')
EV, ER = END['readings'][0], END['readings'][1]
RACC = jl(RES / 'RESOURCE_ACCOUNTING.json')
ACC = jl(RES / 'ACCESS_RECEIPT.json')
RCPT = jl(RES / 'RECEIPTS.json')['receipts'][0]
BK = jl(RES / 'custody' / 'BACKUP_VERIFICATION.json')
OFF = jl(RES / 'custody' / 'OFFDEVICE_COPY.json')
DEP = jl(RES / 'deploy' / 'DEPLOYMENT_CHECK_MAIN.json')
REFEV = jl(PRIV / 'runs' / 'REFERENCES' / 'EVAL_MS.json')['references']
PROTO = (RES / 'PROTOCOL.md').read_text()
CARD1 = (RES / 'cards' / 'RF_RFIT_v1.md').read_text()
CARDR = (RES / 'cards' / 'RF_RFIT_R1.md').read_text()
DEVS = (RES / 'PROTOCOL_DEVIATIONS.md').read_text()
HB3 = (RES / 'roles' / 'statistical_reviewer' / 'HANDBACK_3.md').read_text()
MU = jl(OUTD / 'V2_USEFULNESS.json')          # verifier's own usefulness recomputation (v1 configs, H_LJ, NEURAL_RAW)
MG = jl(OUTD / 'V2_GRID.json')                # verifier's own v1 main-grid comparisons (recipient labels and version means)
M3 = jl(OUTD / 'V3_MAIN.json')                # verifier's own R1 comparisons and usefulness
M4 = jl(OUTD / 'V4_RECIPE_MEANS.json')        # verifier's own per-recipe means (exact falls)
ACCESS_LOG = [json.loads(x) for x in (PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
TEXT = {k: (RES / f).read_text() for k, f in DOCS.items()}
LINES = {k: t.splitlines() for k, t in TEXT.items()}


def grid_cells(bank):
    cells, exact = {}, []
    for v in (0, 1):
        g = jl(PRIV / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json')
        cells.update(g['cells'])
        exact.append(g.get('exact_aliases'))
    return cells, exact


def vmeans(cells, member, view):
    return [sum(cells[c][member][view]['score_auc'] for c in REG if c.startswith(f'S{v}')) / 4 for v in (0, 1)]


GV, GV_ALIAS = grid_cells('RFA1_MAIN')
GR, GR_ALIAS = grid_cells('RFA1_R1_MAIN')
gv = {m: {view: vmeans(GV, m, view) for view in ('pair', 'income', 'occupation')} for m in GV['S0R1k0']}
gr = {m: {view: vmeans(GR, m, view) for view in ('pair', 'income', 'occupation')} for m in GR['S0R1k0']}
mgc = MG['comparisons']                                    # v1, verifier's own
m3c = M3['own_labels_R1']['comparisons']                   # R1, verifier's own
TREE = ('B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ')


def cmp1(reading):
    return (dv if reading == 'v1' else dr)['D2']['comparisons']


def bnd(reading):
    return (dv if reading == 'v1' else dr)['D2']['bounds']['per_comparison']


def ref_util(name, v):
    r = REFEV.get(name) or REFEV.get(f'REF:{name}')
    rv = r[str(v)]
    g = {x['group']: x['value'] for x in rv['utility']['guards']}
    return {'max': rv['point']['max_value_over_cap'], 'men': g['sex0'], 'pass': rv['point'].get('point_pass_12_plus_2')}


def mu_cfg(name, v):
    src = MU['configs'].get(name) or MU['references'][name]
    pv = src['per_version'][str(v)]
    return {'max': pv['max_value_over_cap'], 'men': pv['guards']['sex0']['value'], 'G1': src['G1']['G1']}


def r1_util(name, v):
    return M3['usefulness_R1'][name]['worst_over_cap'][str(v)]


def cd(reading, v, cond):
    return (dv if reading == 'v1' else dr)['D1']['confidence_detail_cand']['per_version'][str(v)][cond]


def same_ck(reading, v, arm):
    f = FIT[(reading, v)]
    k = f['selection']['k_star']
    return f['selection']['path'][f'{arm}_S{k}']['tot_nll']['estimate'] - f['selection']['path'][f'FULL_S{k}']['tot_nll']['estimate']


cs_bind = CS['binding_model_fit_budget_components']
f_orth = {key: f['mechanism_gate']['orthogonal_share'] for key, f in FIT.items()}
f_cos = {key: f['mechanism_gate']['cos_tot_dir'] for key, f in FIT.items()}
same = {(rd, v, a): same_ck(rd, v, a) for rd in ('v1', 'R1') for v in (0, 1) for a in ('STOPFIT', 'ALTCRIT')}
F1min = min(x for F in (FV, FR) for k in ('F1_V0', 'F1_V1') for x in F[k]['orthogonal_shares_to_k_star'])
occ_fall = {m: [gv['NEURAL_RAW']['occupation'][v] - gv[m]['occupation'][v] for v in (0, 1)] for m in ('RF_FULL', 'RF_STOP_HI', 'RF_RAWJ_HI')}
rv_occ = {rd: [src['B2TWIN']['recipient_version_means']['occupation'][str(v)] for v in (0, 1)] for rd, src in (('v1', mgc), ('R1', m3c))}
inc_r = {(rd, ref, v): src[ref]['recipient_version_means']['income'][str(v)]
         for rd, src in (('v1', mgc), ('R1', m3c)) for ref in ('B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
         for v in (0, 1)}
falls = M4['falls_vs_START']
deg = [math.degrees(math.acos(f_cos[('v1', v)])) for v in (0, 1)]


def card_probs(text, header):
    i = text.index(header)
    rows = []
    for line in text[i:].splitlines()[2:]:
        line = line.strip()
        if not line.startswith('|'):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells[0] in ('Prediction', '---') or set(cells[0]) <= set('-'):
            continue
        rows.append((cells[0], cells[1]))
    return rows


C1P = card_probs(CARD1, '- **Predictions, registered:**')
CRP = card_probs(CARDR, '- **Predictions, registered before the R1 fit:**')
PROTO12 = PROTO[PROTO.index('## 12. Registered expectations'):PROTO.index('## 13.')]
PP = re.findall(r'about (0\.\d+)', PROTO12)

# --------------------------------------------------------------------------------------------------- per-line numeric token expectations
E = {'RD': {}, 'SH': {}, 'HO': {}}
rd, sh, ho = E['RD'], E['SH'], E['HO']
v1s, r1s = cmp1('v1'), cmp1('R1')
b1, br = bnd('v1'), bnd('R1')
rd[3] = [L('2026', 'date'), L('10', 'date'), L('25', 'date')]
rd[4] = [L('10', 'deviation 10')]
rd[8] = [R(str(len(GV['S0R1k0']['RF_FULL']['occupation']['per_recipe_score_auc'])), 'recipes per view in the bank (per_recipe_score_auc keys)')]
rd[16] = [R(u(min(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 F0 orthogonal share min'), R(u(max(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 F0 orthogonal share max')]
for i in range(29, 32):
    rd[i] = 'VERBATIM'                                     # campaign summary blockquote, checked as one string below
rd[37] = [R(str(CS['summary']['d2_looks']), 'summary d2_looks')]
rd[41] = [L('1', 'section')]
rd[43] = [R(str(MU['configs']['RF_FULL']['G1']['n_conditions']), 'G1 n_conditions')]
rd[44] = [L('7', 'deviation 7'), R(f"{max(MU['income_identity'][v]['max_abs_RF_FULL_income_minus_NEURAL_RAW_bank'] for v in ('0', '1')):.1e}".replace('e-08', 'e-8'), 'verifier income identity max')]
rd[46] = [R(str(dv['D1']['confidence']['cand']['n_pass']), 'v1 n_pass'), R('28', 'conditions'), L('1', 'version'),
          R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['value'], 5), 'v1 v1 occ NLL vs Ucal value'),
          R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['se'], 5), 'se'), R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['cap'], 3), 'cap')]
rd[47] = [R(u(cd('v1', 1, 'guard|sex0')['value'], 5), 'v1 v1 men guard value'), R(u(cd('v1', 1, 'guard|sex0')['se'], 5), 'se'),
          R(u(cd('v1', 1, 'guard|sex0')['cap'], 3), 'cap')]
rd[48] = [R(str(dr['D1']['confidence']['cand']['n_pass']), 'R1 n_pass'), R('28', 'conditions'), L('1', 'version'),
          R(u(cd('R1', 1, 'guard|sex0')['value'], 5), 'R1 v1 men guard'), R(u(cd('R1', 1, 'guard|sex0')['cap'], 3), 'cap')]
rd[50] = [L('1', 'version'), R(u(mu_cfg('RF_FULL', 1)['max'], 3), 'v1 RF_FULL v1 max value/cap (verifier)')]
rd[51] = [R(u(r1_util('RF1_FULL', 1), 3), 'R1 RF1_FULL v1 worst over cap (verifier)')]
rd[52] = [L('2', 'z = 2 (selection rule)')]
rd[56] = [L('2', 'z = 2')]
rd[58] = [L('0', 'version'), R(u(cs_bind['FIT_RF_V0']['failing_components']['occ_brier_women']['estimate_plus_2se'], 5), 'CS v1 V0'),
          R(u(cs_bind['FIT_RF_V0']['failing_components']['occ_brier_women']['budget'], 4), 'budget')]
rd[59] = [L('1', 'version'), R(u(cs_bind['FIT_RF_V1']['failing_components']['occ_nll']['estimate_plus_2se'], 5), 'CS v1 V1 occ_nll'),
          R(u(cs_bind['FIT_RF_V1']['failing_components']['occ_nll']['budget'], 4), 'budget'),
          R(u(cs_bind['FIT_RF_V1']['failing_components']['occ_brier_women']['estimate_plus_2se'], 5), 'occ_brier_women'),
          R(u(cs_bind['FIT_RF_V1']['failing_components']['occ_brier_women']['budget'], 4), 'budget')]
rd[60] = [L('0', 'version'), R(u(cs_bind['FIT_RF_R1_V0']['failing_components']['occ_brier_women']['estimate_plus_2se'], 5), 'CS R1 V0'),
          R(u(cs_bind['FIT_RF_R1_V0']['failing_components']['occ_brier_women']['budget'], 4), 'budget')]
rd[61] = [L('1', 'version'), R(u(cs_bind['FIT_RF_R1_V1']['failing_components']['occ_nll']['estimate_plus_2se'], 5), 'CS R1 V1'),
          R(u(cs_bind['FIT_RF_R1_V1']['failing_components']['occ_nll']['budget'], 4), 'budget')]
rd[63] = [L('0', 'version')]
rd[64] = [R(u(FIT[('v1', 0)]['cand']['spend_over_start']['value'], 4), 'v1 V0 CAND spend'), R(u(FIT[('v1', 0)]['cand']['spend_over_start']['se'], 4), 'se')]
rd[66] = [L('0', 'version')]
rd[68] = [L('1', 'version')]
rd[69] = [R(s(dv['D2']['targeting']['NEURAL_RAW']['version_means']['0'], 4), 'v1 START benefit version 0 (D2.targeting)'),
          R(s(dv['D2']['targeting']['NEURAL_RAW']['version_means']['1'], 4), 'v1 START benefit version 1')]
rd[71] = [L('2', 'section')]
rd[75] = [R(s(v1s['START']['benefit_mean'], 4), 'v1 START')]
rd[76] = [R(u(v1s['START']['benefit_se'], 4), 'se')]
for ln, key, comp in ((80, 'START', 'NEURAL_RAW'), (81, 'CONT', 'RF_TASK'), (82, 'STOPFIT', 'RF_STOP_HI'), (83, 'ALTCRIT', 'RF_ALTC_HI'),
                      (84, 'RAWJ', 'RF_RAWJ_HI')):
    rd[ln] = [R(s(v1s[key]['benefit_mean'], 4), f'v1 {key}'), R(u(v1s[key]['benefit_se'], 4), 'se'),
              R(s(xv[comp]['robustness_mean'], 4), f'v1 D2R {comp}'), R(u(xv[comp]['robustness_se'], 4), 'rob se')]
rd[87] = [L('1', 'version'), R(u(mu_cfg('RF_RAWJ_HI', 1)['max'], 3), 'RF_RAWJ_HI v1 max/cap (verifier)')]
rd[88] = [L('1', 'version')]
rd[92] = [R(dict(C1P)['FULL vs STOPFIT at matched spend: tie (NOT_SEPARATED)'], 'card v1 tie probability')]
rd[93] = [R(u(b1['STOPFIT']['upper'], 3), 'v1 STOPFIT upper')]
rd[94] = [R(s(b1['STOPFIT']['upper'], 4), 'v1 STOPFIT upper'), R(u(b1['ALTCRIT']['upper'], 4), 'v1 ALTCRIT upper')]
rd[96] = [R(u(min(same.values()), 4), 'same-checkpoint spend min'), R(u(max(same.values()), 4), 'max')]
rd[100] = [R(ci(XV['refit_checks']), 'v1 refit checks')]
rd[102] = [R(s(xv['NEURAL_RAW']['robustness_mean'], 4), 'v1 D2R START'), R(u(xv['NEURAL_RAW']['robustness_se'], 4), 'se')]
rd[104] = [R(s(xv['RF_STOP_HI']['registered_mean'], 4), 'reg'), R(s(xv['RF_STOP_HI']['robustness_mean'], 4), 'rob'),
           R(s(xv['RF_ALTC_HI']['registered_mean'], 4), 'reg'), R(s(xv['RF_ALTC_HI']['robustness_mean'], 4), 'rob'),
           R(s(xv['SUCC']['registered_mean'], 4), 'reg')]
rd[105] = [R(s(xv['SUCC']['robustness_mean'], 4), 'rob')]
rd[117] = [R(s(r1s['START']['benefit_mean'], 4), 'R1 START'), R(u(r1s['START']['benefit_se'], 4), 'se')]
for ln, key, comp in ((121, 'START', 'NEURAL_RAW'), (122, 'CONT', 'RF1_TASK'), (123, 'STOPFIT', 'RF1_STOP_HI'), (124, 'ALTCRIT', 'RF1_ALTC_HI'),
                      (125, 'RAWJ', 'RF1_RAWJ_HI')):
    rd[ln] = [R(s(r1s[key]['benefit_mean'], 4), f'R1 {key}'), R(u(r1s[key]['benefit_se'], 4), 'se'),
              R(s(xr[comp]['robustness_mean'], 4), f'R1 D2R {comp}'), R(u(xr[comp]['robustness_se'], 4), 'rob se')]
rd[127] = [L('1', 'version'), R(u(r1_util('RF1_RAWJ_HI', 1), 3), 'RF1_RAWJ_HI v1 worst over cap (verifier)')]
rd[129] = [R(s(xr['RF1_ALTC_HI']['registered_mean'], 4), 'reg'), R(s(xr['RF1_ALTC_HI']['robustness_mean'], 4), 'rob'),
           R(s(xr['SUCC']['registered_mean'], 4), 'reg'), R(s(xr['SUCC']['robustness_mean'], 4), 'rob')]
rd[139] = [R(s(FC['benefit_mean'], 4), 'FC mean'), R(u(FC['benefit_se_agg'], 4), 'FC se'),
           R(s(FC['benefit_version_means']['0'], 4), 'FC v0'), R(s(FC['benefit_version_means']['1'], 4), 'FC v1')]
rd[140] = [R(s(FC['primary_cost_difference']['value'], 5), 'FC spend'), R(u(FC['primary_cost_difference']['se'], 5), 'se')]
rd[141] = [R(s(FC['secondary']['occupation|men_brier']['value'], 5), 'FC men Brier'), R(u(FC['secondary']['occupation|men_brier']['se'], 5), 'se')]
rd[150] = [R(ci(DV['lc_grid_validity']['n_checks']), 'v1 LC checks'), R(ci(DR['lc_grid_validity']['n_checks']), 'R1 LC checks')]
lcv, lcr = dv['LC']['per_comparator'], dr['LC']['per_comparator']
rd[152] = [x for c in ('RF_TASK', 'RF_ALTC_HI', 'H_LJ') for x in (R(s(lcv[c]['change_mean'], 4), f'LC {c}'), R(u(lcv[c]['change_se'], 4), 'se'))]
rd[154] = [R(s(lcv['NEURAL_RAW']['change_mean'], 4), 'LC START'), R(u(lcv['NEURAL_RAW']['change_se'], 4), 'se')]
rd[155] = [R(s(lcr['H_LJ']['change_mean'], 4), 'R1 LC H_LJ'), R(u(lcr['H_LJ']['change_se'], 4), 'se')]
rd[156] = [R(s(lcr['RF1_STOP_HI']['change_mean'], 4), 'R1 LC STOP'), R(s(lcr['RF1_ALTC_HI']['change_mean'], 4), 'R1 LC ALTC')]
rd[158] = [R(ci(dv['LC']['inner_episode_fit_size']), 'inner fit size'), R(ci(dv['LC']['budgets']['FULL']), 'FULL budget')]
rd[165] = [R(u(F1min, 3), 'min F1 orthogonal share up to k*')]
rd[169] = [R(s(FV['F3']['linear'], 4), 'v1 F3 LOG_LR'), R(s(FV['F3']['tree_table_mean'], 4), 'v1 tree/table mean')]
rd[170] = [R(s(FR['F3']['linear'], 4), 'R1 F3 LOG_LR'), R(s(FR['F3']['tree_table_mean'], 4), 'R1 tree/table mean')]
rd[171] = [R(u(FV['F3']['mean_gain_over_stopfit_by_recipe']['LOOKUP'], 1), 'LOOKUP gain')]
rd[175] = [R(u(max(F[k]['negative_orientation_fraction'] for F in (FV, FR) for k in ('F5_V0', 'F5_V1')), 1), 'F5 fraction')]
rd[178] = [R(s(br['START']['lower'], 4), 'R1 START lower')]
rd[179] = [L('0', 'version'), R(str(FV['F6_V0']['logged_steps']), 'v1 F6_V0 logged steps'), R(str(FR['F6_V0']['logged_steps']), 'R1'),
           R(str(FV['F6_V0']['monitors']), 'monitors')]
rd[181] = [L('3', 'section')]
rd[190] = [R(u(gv['RF_FULL']['occupation'][0], 4), 'grid'), R(u(gv['RF_FULL']['occupation'][1], 4), 'grid')]
rd[191] = [R(u(gr['RF1_FULL']['occupation'][0], 4), 'grid'), R(u(gr['RF1_FULL']['occupation'][1], 4), 'grid'),
           R(u(gv['H_LJ']['occupation'][0], 4), 'tree refs occ'), R(u(gv['H_LJ']['occupation'][1], 4), 'tree refs occ'),
           R(s(rv_occ['v1'][0], 3), 'v1 occ r v0 (verifier)')]
rd[192] = [R(s(rv_occ['v1'][1], 3), 'v1 occ r v1'), R(s(rv_occ['R1'][0], 3), 'R1 occ r v0'), R(s(rv_occ['R1'][1], 3), 'R1 occ r v1'),
           R(u(FC['recipient']['occupation']['limit'], 3), 'recipient limit')]
rd[193] = [L('0', 'version'), R(s(inc_r[('v1', 'RETIRED_B3CTL', 0)], 4), 'RETIRED_B3CTL income r v0')]
rd[196] = [R(u(ref_util('H_LJ', 0)['max'], 3), 'ref max/cap v0'), R(u(ref_util('H_LJ', 1)['max'], 3), 'v1'), L('0', 'version'), L('1', 'version'),
           R(u(ref_util('H_LJ', 0)['men'], 5), 'men guard v0'), R(u(ref_util('H_LJ', 1)['men'], 5), 'v1')]
rd[197] = [R(u(cd('v1', 0, 'guard|sex0')['cap'], 3), 'guard cap')]
rd[198] = [R(u(ref_util('RETIRED_B3CTL', 0)['max'], 3), 'RETIRED_B3CTL max/cap')]
rd[201] = [L('4', 'section')]
rd[204] = [R(s(dv['D3']['per_reference']['NEURAL_RAW']['mean'], 4), 'D3'), R(s(dv['D3']['per_reference']['H_LJ']['mean'], 4), 'D3'),
           L('0.020', 'D4 target (PROTOCOL 6 / rules)')]
rd[205] = [R(s(dr['D3']['per_reference']['NEURAL_RAW']['mean'], 4), 'D3'), R(s(dr['D3']['per_reference']['H_LJ']['mean'], 4), 'D3')]
rd[207] = [L('5', 'section')]
rd[215] = [L('12', 'PROTOCOL section')]
rd[219] = [R(PP[0], 'PROTOCOL 12')]
rd[220] = [R(PP[1], 'PROTOCOL 12'), R(u(gv['RF_FULL']['occupation'][0], 4), 'grid'), R(u(gv['RF_FULL']['occupation'][1], 4), 'grid'),
           R(u(gv['RF_STOP_HI']['occupation'][0], 4), 'grid'), R(u(gv['RF_STOP_HI']['occupation'][1], 4), 'grid'),
           R(u(gr['RF1_FULL']['occupation'][0], 4), 'grid'), R(u(gr['RF1_FULL']['occupation'][1], 4), 'grid'),
           R(u(gr['RF1_STOP_HI']['occupation'][0], 4), 'grid'), R(u(gr['RF1_STOP_HI']['occupation'][1], 4), 'grid')]
rd[221], rd[222], rd[223], rd[224] = [R(PP[2], 'P12')], [R(PP[3], 'P12')], [R(PP[4], 'P12')], [R(PP[5], 'P12')]
rd[226] = [L('0.005', 'PROTOCOL 12 quoted range'), L('0.015', 'PROTOCOL 12 quoted range')]
rd[228] = [R(u(occ_fall['RF_FULL'][0], 3), 'occ fall v0'), R(u(occ_fall['RF_FULL'][1], 3), 'occ fall v1'), L('1', 'version')]
rd[229] = [R(u(occ_fall['RF_STOP_HI'][0], 3), 'STOP fall v0'), R(u(occ_fall['RF_STOP_HI'][1], 3), 'v1'),
           R(u(occ_fall['RF_RAWJ_HI'][0], 3), 'RAWJ fall v0'), R(u(occ_fall['RF_RAWJ_HI'][1], 4), 'RAWJ fall v1 (4 dp)')]
rd[232] = [L('6', 'card section'), R(str(len(C1P)), 'card v1 predictions')]
for ln, (_, p) in zip(range(236, 247), C1P):
    rd[ln] = [R(p, 'card v1 probability')]
rd[239] = [L('+0.003', 'card range'), L('+0.020', 'card range'), R(C1P[3][1], 'card v1 probability'), R(s(v1s['START']['benefit_mean'], 4), 'v1 START')]
rd[248] = [L('3', 'card section'), R(str(len(CRP)), 'card R1 predictions')]
for ln, (_, p) in zip(range(252, 261), CRP):
    rd[ln] = [R(p, 'card R1 probability')]
rd[252] = [R(CRP[0][1], 'card'), R(u(f_orth[('R1', 0)], 3), 'R1 share v0'), R(u(f_orth[('R1', 1)], 3), 'R1 share v1'),
           R(u(f_orth[('v1', 0)], 3), 'v1 share v0'), R(u(f_orth[('v1', 1)], 3), 'v1 share v1')]
rd[255] = [R(CRP[3][1], 'card'), R(s(FC['benefit_mean'], 4), 'FC'), R(u(gr['RF1_FULL']['pair'][0], 4), 'grid'), R(u(gr['RF1_FULL']['pair'][1], 4), 'grid'),
           R(u(gr['RF_FULL']['pair'][0], 4), 'grid'), R(u(gr['RF_FULL']['pair'][1], 4), 'grid')]
rd[256] = [R(CRP[4][1], 'card')] + [R(s(falls[k][r], 4), f'{k} {r} (verifier exact)') for r in ('ET25', 'CATI1', 'HGB15')
                                    for k in ('R1 (RFA1_R1_MAIN)', 'v1 (RFA1_MAIN)')]
rd[273] = [R(u(sum(rv_occ['v1']) / 2, 2), 'occupation gap (about)')]

sh[3] = [L('2026', 'date'), L('10', 'date'), L('25', 'date'), R(RACC['start_utc'][11:13], 'start hour'), R(RACC['start_utc'][14:16], 'start minute'),
         L('2019', 'ACS year (PROTOCOL 0)')]
sh[7] = [L('2', 'income classes (PROTOCOL 0)'), L('5', 'occupation groups (PROTOCOL 0)')]
sh[10] = [R('28', 'conditions')]
sh[16] = [R(ci(FIT[('v1', 0)]['full_logs_FULL'][0]['n_A']), 'n_A'), L('2,400', 'n_E about 2,400 (logs 2,400-2,401)')]
sh[34] = [R(u(min(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 share'), R(u(max(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 share'),
          R(str(round(sum(deg) / 2)), f'turn in degrees from cos_tot_dir {deg}')]
sh[35] = [R('28', 'conditions')]
sh[36] = [R(s(v1s['START']['benefit_mean'], 4), 'v1 START')]
sh[37] = [R(u(v1s['START']['benefit_se'], 4), 'se'), R(s(xv['NEURAL_RAW']['robustness_mean'], 4), 'D2R'), R(u(xv['NEURAL_RAW']['robustness_se'], 4), 'se')]
sh[38] = [R(s(v1s['STOPFIT']['benefit_mean'], 4), 'v1 STOPFIT'), R(u(v1s['STOPFIT']['benefit_se'], 4), 'se'), R(s(v1s['ALTCRIT']['benefit_mean'], 4), 'ALTCRIT')]
sh[39] = [R(u(v1s['ALTCRIT']['benefit_se'], 4), 'se')]
sh[42] = [R(s(FC['benefit_mean'], 4), 'FC'), R(u(FC['benefit_se_agg'], 4), 'FC se')]
sh[43] = [R(u(sum(rv_occ['v1']) / 2, 2), 'occupation gap (about)'), L('0.020', 'D4 target')]
sh[48] = [L('1', 'list')]
sh[53] = [L('1e-14', 'envelope identity; card v1 section 3 and the investigator record say 1.4e-14')]
sh[54] = [L('2', 'list')]
sh[55] = [R(u(min(f_orth.values()), 2), 'share min (both readings)'), R(u(max(f_orth.values()), 2), 'share max')]
sh[56] = [R(u(min(f_cos.values()), 3), 'cos min'), R(u(max(f_cos.values()), 3), 'cos max'), L('0.05', 'F1 threshold (card)')]
sh[58] = [R(u(gv['RF_FULL']['occupation'][0], 4), 'grid'), R(u(gv['RF_FULL']['occupation'][1], 4), 'grid'),
          R(u(gv['RF_STOP_HI']['occupation'][0], 4), 'grid'), R(u(gv['RF_STOP_HI']['occupation'][1], 4), 'grid')]
sh[59] = [R(u(min(same.values()), 4), 'same-checkpoint min'), R(u(max(same.values()), 4), 'max')]
sh[61] = [L('0', 'version'), L('0', 'version'), R(u(FIT[('v1', 0)]['cand']['spend_over_start']['value'], 4), 'v1 V0 spend')]
sh[62] = [L('4', 'answer number')]
sh[63] = [L('3', 'list'), R('28', 'conditions'), R('28', 'conditions')]
sh[65] = [L('1', 'version'), R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['value'], 5), 'v'), R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['se'], 5), 'se'),
          R(u(cd('v1', 1, 'occupation|Ucal|nll_excess')['cap'], 3), 'cap'), L('1', 'version'), R(u(cd('v1', 1, 'guard|sex0')['value'], 5), 'v'),
          R(u(cd('v1', 1, 'guard|sex0')['se'], 5), 'se'), R(u(cd('v1', 1, 'guard|sex0')['cap'], 3), 'cap')]
sh[66] = [L('1', 'version'), R(u(cd('R1', 1, 'guard|sex0')['value'], 5), 'R1 guard'), R(u(cd('R1', 1, 'guard|sex0')['cap'], 3), 'cap')]
sh[67] = [L('1', 'version'), R(u(mu_cfg('RF_FULL', 1)['max'], 3), 'v1'), R(u(r1_util('RF1_FULL', 1), 3), 'R1')]
sh[69] = [L('4', 'list')]
sh[74] = [R(s(v1s['STOPFIT']['benefit_mean'], 4), 'v1'), R(u(v1s['STOPFIT']['benefit_se'], 4), 'se'), R(s(v1s['ALTCRIT']['benefit_mean'], 4), 'v1'),
          R(u(v1s['ALTCRIT']['benefit_se'], 4), 'se')]
sh[75] = [R(s(r1s['STOPFIT']['benefit_mean'], 4), 'R1'), R(u(r1s['STOPFIT']['benefit_se'], 4), 'se'), R(s(r1s['ALTCRIT']['benefit_mean'], 4), 'R1'),
          R(u(r1s['ALTCRIT']['benefit_se'], 4), 'se')]
sh[77] = [R(u(b1['STOPFIT']['upper'], 3), 'v1 STOPFIT upper'), R(u(b1['ALTCRIT']['upper'], 4), 'v1 ALTCRIT upper')]
sh[78] = [L('5', 'list')]
sh[82] = [R(s(xv['RF_STOP_HI']['registered_mean'], 4), 'reg'), R(s(xv['RF_STOP_HI']['robustness_mean'], 4), 'rob'),
          R(s(xv['RF_ALTC_HI']['registered_mean'], 4), 'reg'), R(s(xv['RF_ALTC_HI']['robustness_mean'], 4), 'rob'),
          R(s(xv['SUCC']['registered_mean'], 4), 'reg')]
sh[83] = [R(s(xv['SUCC']['robustness_mean'], 4), 'rob')]
sh[84] = [R(s(xv['NEURAL_RAW']['robustness_mean'], 4), 'D2R'), R(u(xv['NEURAL_RAW']['robustness_se'], 4), 'se')]
sh[88] = [R(s(xr['RF1_ALTC_HI']['registered_mean'], 4), 'reg'), R(s(xr['RF1_ALTC_HI']['robustness_mean'], 4), 'rob')]
sh[89] = [R(s(xr['NEURAL_RAW']['robustness_mean'], 4), 'R1 D2R'), R(u(xr['NEURAL_RAW']['robustness_se'], 4), 'se')]
sh[94] = [L('6', 'list')]
sh[98] = [L('7', 'list')]
sh[100] = [L('8', 'list')]
sh[102] = [R(u(b1['STOPFIT']['upper'], 3), 'v1 STOPFIT upper'), R(u(b1['ALTCRIT']['upper'], 4), 'v1 ALTCRIT upper')]
sh[105] = [R(u(sum(rv_occ['v1']) / 2, 2), 'occupation gap')]
sh[106] = [R(u(ref_util('H_LJ', 0)['max'], 3), 'ref v0'), R(u(ref_util('H_LJ', 1)['max'], 3), 'ref v1')]
sh[109] = [L('0', 'version'), L('0', 'version')]
sh[110] = [L('1', 'version'), L('1', 'version')]
sh[111] = [L('1', 'version'), R(u(mu_cfg('RF_FULL', 1)['max'], 3), 'v1'), R(u(r1_util('RF1_FULL', 1), 3), 'R1')]

ho[3] = [L('2026', 'date'), L('10', 'date'), L('25', 'date')]
ho[4] = [R(RACC['utc'][11:13], 'RESOURCE_ACCOUNTING utc hour')]
ho[9] = []
ho[12] = [R(RCPT['verified_utc'][11:13], 'receipt hour'), R(RCPT['verified_utc'][14:16], 'receipt minute')]
ho[17] = [L('10', 'deviation 10')]
nv1 = sum(1 for x in RACC['configurations']['configurations'] if x['card'].endswith('RF_RFIT_v1.md'))
nr1 = sum(1 for x in RACC['configurations']['configurations'] if x['card'].endswith('RF_RFIT_R1.md'))
ho[18] = [R(str(len(RACC['configurations']['configurations'])), 'configurations'), R(str(nv1), 'v1 configurations'), R(str(nr1), 'R1 configurations')]
ho[21] = [L('0', 'version'), L('1', 'version')]
ho[26] = [L('1e-9', 'FD relative errors 1.8e-10..1.4e-9 (cards, investigator, verifier)')]
ho[27] = [R(u(min(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 share'), R(u(max(f_orth[('v1', 0)], f_orth[('v1', 1)]), 2), 'v1 share'),
          R(u(b1['STOPFIT']['upper'], 3), 'v1 STOPFIT upper')]
ho[28] = [R(u(b1['ALTCRIT']['upper'], 4), 'v1 ALTCRIT upper')]
ks = [FIT[k]['selection']['k_star'] for k in FIT]
ho[31] = [R(str(min(ks)), 'k* min'), R(str(max(ks)), 'k* max')]
ho[33] = [L('1', 'version'), R(u(mu_cfg('RF_FULL', 1)['max'], 3), 'v1'), R(u(r1_util('RF1_FULL', 1), 3), 'R1')]
ho[35] = [R(u(sum(rv_occ['v1']) / 2, 2), 'occupation gap')]
ho[36] = [L('1', 'deviations'), L('10', 'deviations')]
ho[38] = [R(f"{max(MU['income_identity'][v]['max_abs_RF_FULL_income_minus_NEURAL_RAW_bank'] for v in ('0', '1')):.1e}".replace('e-08', 'e-8'), 'income identity')]
ho[44] = [R(RACC['utc'][11:13], 'queue close hour')]
ho[46] = [R(RACC['start_utc'][11:13], 'start hour'), R(RACC['start_utc'][14:16], 'start minute'), R(u(RACC['elapsed_h'], 2), 'elapsed h'),
          R(f"{RACC['elapsed_limit_h']:g}", 'elapsed limit')]
st = RACC['cpu_s_by_stage']
ho[47] = [R(u(RACC['charged_cpu_h'], 2), 'charged CPU-h'), R(f"{RACC['cpu_limit_h']:g}", 'CPU limit'),
          R(f"{sum(x['charged_cpu_s'] for x in RACC['agent_charges']):g}", 'agent charges')]
ho[48], ho[49], ho[50] = [R(ci(round(st['readers'])), 'readers')], [R(ci(round(st['fit'])), 'fit')], [R(ci(round(st['custody'])), 'custody')]
ho[51], ho[52] = [R(ci(round(st['verify'])), 'verify')], [R(ci(round(st['deployment_check'])), 'deployment check')]
ho[53], ho[54] = [R(ci(round(st['evaluate'])), 'evaluate')], [R(ci(round(st['claims'])), 'claims')]
ho[55] = [R(str(RACC['units_done']), 'units done'), R(str(RACC['units_failed']), 'units failed')]
ho[56] = [L('7', 'deviation 7')]
ho[57] = [L('4', 'wave 4'), L('4', 'hand-back 4')]
kinds = RACC['configurations']['by_family_and_kind']
ho[60] = [R(str(len(RACC['configurations']['configurations'])), 'configurations'), L('24', 'limit (PROTOCOL 10)'),
          R(str(kinds['FAM|candidate']), 'candidates'), R(str(kinds['FAM|twin']), 'twins'), R(str(kinds['FAM|control']), 'controls')]
byop = ACC['by_operation_role_fields_entry_card']
nfit = sorted({x['reads'] for x in byop if x['operation'] == 'fit'})
ho[64] = [R(str(ACC['label_reads']), 'label reads'), R(ACC['registration_receipt_verified_utc'][11:13], 'receipt hour'),
          R(ACC['registration_receipt_verified_utc'][14:16], 'receipt minute')]
ho[65] = [R(str(nfit[0]) if len(nfit) == 1 else str(nfit), 'reads per fit entry point')]
ho[66] = [R(str(sum(x['reads'] for x in byop if x['operation'] == 'benchmark')), 'benchmark'),
          R(str(sum(x['reads'] for x in byop if x['operation'] == 'readers')), 'readers'),
          R(str(sum(x['reads'] for x in byop if x['operation'] == 'verify')), 'verify')]
ho[73] = [R(ci(BK['size_bytes']), 'backup size')]
avail = OFF['drive']['availability_note']
ho[78] = [R(re.search(r'about (\d\d):(\d\d)Z', avail).group(1), 'left /Volumes hour'), R(re.search(r'by (\d\d):(\d\d):\d\dZ', avail).group(1), 'back hour'),
          R(re.search(r'by (\d\d):(\d\d):\d\dZ', avail).group(2), 'back minute')]
ho[80] = [R(re.search(r'the (\d+) existing campaign folders', OFF['untouched']).group(1), 'untouched folders')]

# ------------------------------------------------------------------------------------------------------------------- numeric sweep
numeric = {}
for d, lines in LINES.items():
    for i, line in enumerate(lines, 1):
        toks = [t.rstrip(',') for t in NUM.findall(line)]
        spec = E[d].get(i)
        if not toks and not spec:
            continue
        if spec == 'VERBATIM':
            numeric[f'{d}:{i}'] = {'status': 'VERBATIM_CHECKED_BELOW', 'tokens': toks}
            continue
        if spec is None:
            numeric[f'{d}:{i}'] = {'status': 'UNCHECKED', 'tokens': toks}
            continue
        exp = [x[1] for x in spec]
        if toks == exp:
            numeric[f'{d}:{i}'] = {'status': 'MATCH', 'tokens': toks, 'n_record': sum(1 for x in spec if x[0] == 'REC')}
        else:
            numeric[f'{d}:{i}'] = {'status': 'MISMATCH', 'tokens': toks, 'expected': exp, 'sources': [x[2] for x in spec],
                                   'line': line.strip()}

# ------------------------------------------------------------------------------------------------------------------- statement checks
chk = {}


def ok(key, cond, note=''):
    chk[key] = {'pass': bool(cond), 'note': note}


def has(d, ln, text):
    return text in LINES[d][ln - 1]


def joined(d, a, b):
    return ' '.join(x.strip().lstrip('>').strip() for x in LINES[d][a - 1:b])


# RD decision paragraph and table
ok('RD:16-17 not separated from any mechanism-specific control (both readings)', set(dv['D2']['failed']) >= {'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ'}
   and set(dr['D2']['failed']) >= {'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ'})
ok('RD:18 v1 pair recovery resolved below NEURAL_RAW on the registered cells', v1s['START']['label'] == 'AGG_POSITIVE' and v1s['START']['pass'])
ok('RD:18-19 "This holds for ... the one adaptive revision" (R1 START resolved, if read literally)', r1s['START']['label'] == 'AGG_POSITIVE',
   f"R1 START is {r1s['START']['label']} (lower {br['START']['lower']:+.4f}); the sentence is accurate for R1 only for 'not separated' and 'not confirmed'")
row24 = [x.strip() for x in LINES['RD'][23].strip('|').split('|')]
row25 = [x.strip() for x in LINES['RD'][24].strip('|').split('|')]
ok('RD:24 v1 table row labels', row24[1] == f"{dv['outcome_class']['label']} (partial {dv['outcome_class']['partial']})" and row24[2] == dv['D1']['label']
   and row24[3] == dv['D2']['label'] and row24[4] == dv['D3']['label'] and row24[5] == dv['D4']['label']
   and row24[6] == f"{dv['LC']['label']} (point rule)" and row24[7] == dv['D2R']['label'] == XV['d2r']['label'])
ok('RD:25 R1 table row labels (unprefixed)', row25[1].startswith(dr['outcome_class']['label_unprefixed']) and row25[2] == dr['D1']['label_unprefixed']
   and row25[3] == dr['D2']['label_unprefixed'] and row25[4] == dr['D3']['label_unprefixed'] and row25[5] == dr['D4']['label_unprefixed']
   and row25[6] == f"{dr['LC']['label_unprefixed']} (point rule)" and row25[7] == dr['D2R']['label'] == XR['d2r']['label']
   and all(dr[k]['label'].startswith('ADAPTIVE_R1__') for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'LC', 'outcome_class')))
ok('RD:25 "(labels carry the ADAPTIVE_R1 prefix)" also true of the R1 D2R label', dr['D2R']['label'].startswith('ADAPTIVE_R1__'),
   f"R1 D2R label is '{dr['D2R']['label']}' (unprefixed in claims/RFA_R1_final and claims/RFA_R1_D2R); every other R1 label carries the prefix")
ok('RD:10 D5 not applicable, D6 NOT_CLAIMED', dv['D5']['label'] == 'NOT_APPLICABLE_NOT_JOINT' and dv['D6']['label'] == 'NOT_CLAIMED'
   and dr['D5']['label_unprefixed'] == 'NOT_APPLICABLE_NOT_JOINT' and dr['D6']['label_unprefixed'] == 'NOT_CLAIMED')
ok('RD:29-35 campaign summary quoted verbatim', joined('RD', 29, 35) == CS['summary']['statement'])
ok('RD:37 K = 2 = d2_looks', CS['summary']['d2_looks'] == 2 and len(CS['summary']['d2_met']) == 0)
# usefulness
ok('RD:43 / SH:35 / SH:63 point-feasible on 28 conditions, both readings and versions',
   MU['configs']['RF_FULL']['G1']['G1'] and M3['usefulness_R1']['RF1_FULL']['G1'] and dv['D1']['cand'] == dr['D1']['cand'] == 'POINT_ELIGIBLE'
   and dv['D1']['cand_failed'] == dr['D1']['cand_failed'] == [])
ok('RD:44 deviation 7 states 6.8e-8', '6.8e-8' in DEVS.split('\n7. ')[1].split('\n8. ')[0])
ok('RD:46-47 v1 unresolved = version-1 occupation NLL vs Ucal and men guard', dv['D1']['confidence']['cand']['unresolved'] == ['v1|occupation|Ucal|nll_excess', 'v1|guard|sex0'])
ok('RD:48 R1 unresolved = version-1 men guard only', dr['D1']['confidence']['cand']['unresolved'] == ['v1|guard|sex0'])
ok('RD:49 not reserve-qualified (both)', not dv['D1']['reserve']['cand']['qualified'] and not dr['D1']['reserve']['cand']['qualified'])
for rdg, D in (('v1', dv), ('R1', dr)):
    pv1 = D['D1']['confidence_detail_cand']['per_version']['1']
    worst = max(pv1, key=lambda c: pv1[c]['value'] / pv1[c]['cap'])
    allv = {(v, c): x['value'] / x['cap'] for v, d1 in D['D1']['confidence_detail_cand']['per_version'].items() for c, x in d1.items()}
    ok(f'RD:50 / SH:67 {rdg} tightest condition at the point is the version-1 men guard (guard|sex0 = men, ohn/common.py)',
       worst == 'guard|sex0' and max(allv, key=allv.get) == ('1', 'guard|sex0'))
for key, (fit, comps) in {'RD:58': ('FIT_RF_V0', ['occ_brier_women']), 'RD:59': ('FIT_RF_V1', ['occ_nll', 'occ_brier_women']),
                          'RD:60': ('FIT_RF_R1_V0', ['occ_brier_women']), 'RD:61': ('FIT_RF_R1_V1', ['occ_nll'])}.items():
    b = cs_bind[fit]
    rk = {'FIT_RF_V0': ('v1', 0), 'FIT_RF_V1': ('v1', 1), 'FIT_RF_R1_V0': ('R1', 0), 'FIT_RF_R1_V1': ('R1', 1)}[fit]
    ln = LINES['RD'][int(key[3:]) - 1]
    ok(f'{key} k* / first failing / components ({fit})', sorted(b['failing_components']) == sorted(comps) and b['k_star'] == FIT[rk]['selection']['k_star']
       and f"| S{b['k_star']} | S{b['first_failing_step']} |" in ln and all(b['failing_components'][c]['estimate_plus_2se'] > b['failing_components'][c]['budget'] for c in comps))
ok('RD:52-53 selection rule description vs fit record rule', 'S_10..S_k all keep estimate + 2 household-clustered SE <= BUDGET_OCC' in FIT[('v1', 0)]['selection']['rule'],
   'doc paraphrase "the last checkpoint whose estimate + 2 SE stays within" is looser than "the largest S_k such that S_10..S_k all keep"; equivalent here because first_failing = k* + 10 in all four fits')
ok('RD:52-61 first failing checkpoint = k* + 10 in all four fits', all(cs_bind[f]['first_failing_step'] == cs_bind[f]['k_star'] + 10 for f in cs_bind))
fv0 = FIT[('v1', 0)]['paths']
ok('RD:65 STOPFIT and ALTCRIT HI points at S20 (= k*) in v1 version 0', fv0['STOPFIT']['hi'] == fv0['ALTCRIT']['hi'] == 20 == FIT[('v1', 0)]['selection']['k_star'])
ok('RD:66 LO points S10, saved for STOPFIT only', fv0['STOPFIT']['lo'] == fv0['ALTCRIT']['lo'] == 10 and 'RF_STOP_LO' in FIT[('v1', 0)]['saved']
   and 'LO_NOT_SAVED_SPEC_BUDGET' in fv0['ALTCRIT']['flags'] and 'LO_NOT_SAVED_SPEC_BUDGET' not in fv0['STOPFIT']['flags'])
from rfa import rules as RU  # noqa: E402  (frozen rules, label-free use)
gfv, gfr = RU.grid_fineness(PV['fineness_inputs']), RU.grid_fineness(PR['fineness_inputs'])
fin_v = {k: {str(v): x for v, x in p.items()} for k, p in gfv['per_path'].items()}
fin_r = {k: {str(v): x for v, x in p.items()} for k, p in gfr['per_path'].items()}
ok('RD:67-68 fineness: v1 v0 COARSE_BELOW (STOPFIT, ALTCRIT), COARSE_BOTH (RAWJ); v1 v1 FINE; R1 all FINE (frozen rules on plan inputs)',
   fin_v == {'STOPFIT': {'0': 'COARSE_BELOW', '1': 'FINE'}, 'ALTCRIT': {'0': 'COARSE_BELOW', '1': 'FINE'}, 'RAWJ': {'0': 'COARSE_BOTH', '1': 'FINE'}}
   and all(x == 'FINE' for p in fin_r.values() for x in p.values()) and set(gfv['flags']) <= set(dv['D2']['flags']) and gfr['flags'] == [],
   f'v1 {fin_v}; R1 {fin_r}')
ok('RD:69 version means of the START benefit from the grid equal D2.targeting (difference of unrounded grid means)',
   all(abs((gv['NEURAL_RAW']['pair'][v] - gv['RF_FULL']['pair'][v]) - dv['D2']['targeting']['NEURAL_RAW']['version_means'][str(v)]) < 1e-12 for v in (0, 1)),
   f"exact {gv['NEURAL_RAW']['pair'][0] - gv['RF_FULL']['pair'][0]:+.6f} / {gv['NEURAL_RAW']['pair'][1] - gv['RF_FULL']['pair'][1]:+.6f}; "
   f"the document's +0.0101 is 0.6200 - 0.6099 from the 4-dp TABLES means")
# v1 D2
ok('RD:75/80 v1 START resolved with both recipient gates; D2R not confirmed', v1s['START']['label'] == 'AGG_POSITIVE' and v1s['START']['recipient_gate_pass']
   and xv['NEURAL_RAW']['label'] == 'ROBUSTNESS_NOT_CONFIRMED')
ok('RD:82-83 sign changes STOPFIT / ALTCRIT (v1)', xv['RF_STOP_HI']['sign_change'] and xv['RF_ALTC_HI']['sign_change'] and not xv['RF_RAWJ_HI']['sign_change'] and not xv['RF_TASK']['sign_change'])
ok('RD:87 RF_RAWJ_HI fails G1 in version 1 only', MU['configs']['RF_RAWJ_HI']['G1']['failed'] == {'0': [], '1': ['income|U0|brier_excess', 'income|Ucal|brier_excess']})
ok('RD:88 "the ALT comparison" = RAWJ (PROTOCOL 6 / card 4 name RF_RAWJ_HI the ALT control)', '| ALT | NEURAL_RAW\'s own RAW-J recipe' in PROTO and v1s['RAWJ']['point'] == 'RF_RAWJ_HI',
   'naming: the tables call it RAWJ; this sentence calls it ALT')
ok('RD:89 CONT (RF_TASK) at S_k*, simple comparison', all(FIT[k]['cont']['strength'] == FIT[k]['selection']['k_star'] for k in FIT) and v1s['CONT']['kind'] == 'SIMPLE')
ok('RD:90 STOPFIT and ALTCRIT points G1-feasible both versions (v1 and R1)', MU['configs']['RF_STOP_HI']['G1']['G1'] and MU['configs']['RF_ALTC_HI']['G1']['G1']
   and M3['usefulness_R1']['RF1_STOP_HI']['G1'] and M3['usefulness_R1']['RF1_ALTC_HI']['G1'])
ok('RD:91-92 F2 triggered; card tie prediction 0.75', dv['falsifiers']['per_falsifier']['F2'] == 'TRIGGERED' and dict(C1P)['FULL vs STOPFIT at matched spend: tie (NOT_SEPARATED)'] == '0.75')
ok('RD:96 same-checkpoint differences: path differences equal the recorded paired estimates',
   all(abs(same[(rdg, v, a)] - FIT[(rdg, v)]['selection']['same_checkpoint_spend_vs_cand'][a]['tot_nll']['estimate']) < 1e-12 for (rdg, v, a) in same))
zs = {f'{rdg}|v{v}|{a}': round(FIT[(rdg, v)]['selection']['same_checkpoint_spend_vs_cand'][a]['tot_nll']['estimate']
                               / FIT[(rdg, v)]['selection']['same_checkpoint_spend_vs_cand'][a]['tot_nll']['se'], 2) for (rdg, v, a) in same}
ok('RD:97-98 / SH:59-60 "their paired uncertainty is not recorded" / "(point estimates ...)"',
   not all('se' in FIT[k]['selection']['same_checkpoint_spend_vs_cand'][a]['tot_nll'] for k in FIT for a in ('STOPFIT', 'ALTCRIT')),
   f'the paired household-clustered SEs ARE recorded: fam/FIT_RF_*.json selection.same_checkpoint_spend_vs_cand.<ARM>.tot_nll.se; estimate/se = {zs}')
hi_at_k = {f'{rdg}|v{v}': {cfg: FIT[(rdg, v)]['saved'][cfg]['checkpoint'] for cfg in FIT[(rdg, v)]['saved'] if cfg.endswith(('STOP_HI', 'ALTC_HI', '_FULL'))}
           for (rdg, v) in FIT}
ok('RD:99 "The registered comparisons are made at the bracket points, not at the same checkpoint"',
   not all(len(set(x.values())) == 1 for x in hi_at_k.values()),
   f'in every fit the STOPFIT and ALTCRIT HI bracket points (the D2 comparison points) are saved at S_k*, the same checkpoint as CAND: {hi_at_k}; '
   'RD:65 states this for v1 version 0 only')
ok('RD:100 v1 refit genuinely effective, 6,912 checks', XV['d2r']['refit'] == 'REFIT_GENUINELY_EFFECTIVE' and XV['refit_checks'] == 6912)
ok('RD:103 v1 START all cells point only', xv['NEURAL_RAW']['all_cells_label'] == 'AGG_POSITIVE_POINT_ONLY')
ok('RD:104-105 v1 sign changes = RF_STOP_HI, RF_ALTC_HI, SUCC', sorted(XV['d2r']['sign_changes']) == sorted(['RF_STOP_HI', 'RF_ALTC_HI', 'SUCC']))
fails = lambda p: sorted(vw for vw, lab in p['recipient_robustness'].items() if lab in ('RECIPIENT_AGG_POINT_FAILED', 'RECIPIENT_AGG_RESOLVED_WORSE'))  # noqa: E731
ok('RD:106-110 v1 robustness recipient failures and views', fails(xv['RF_RAWJ_HI']) == ['income'] and fails(xv['H_LJ']) == ['occupation']
   and fails(xv['RETIRED_B3CTL']) == ['income', 'occupation'] and fails(xv['SUCC']) == ['occupation'] and sorted(XV['d2r']['recipient_failures']) == sorted(['RF_RAWJ_HI', 'H_LJ', 'RETIRED_B3CTL', 'SUCC']))
ok('RD:111 / SH:91 coverage: B2TWIN, TWJ, W, TR100, TT100 NOT_RUN (both readings)', all(xv[c]['label'] == xr[c]['label'] == 'NOT_RUN' for c in ('B2TWIN', 'TWJ', 'W', 'TR100', 'TT100'))
   and sorted(c for c in xv if xv[c]['label'] == 'NOT_RUN') == sorted(['B2TWIN', 'TWJ', 'W', 'TR100', 'TT100']))
ok('RD:112-113 READER_SENSITIVE_ENHANCED_RF_RAWJ_HI; RAWJ not positive registered', 'READER_SENSITIVE_ENHANCED_RF_RAWJ_HI' in dv['D2']['flags'] and v1s['RAWJ']['benefit_mean'] < 0)
# R1 D2
ok('RD:117/121 R1 START point only', r1s['START']['label'] == 'AGG_POSITIVE_POINT_ONLY' and not r1s['START']['pass'])
ok('RD:124 R1 sign change ALTCRIT only among decisive', xr['RF1_ALTC_HI']['sign_change'] and not any(xr[c]['sign_change'] for c in ('NEURAL_RAW', 'RF1_TASK', 'RF1_STOP_HI', 'RF1_RAWJ_HI')))
ok('RD:127 RF1_RAWJ_HI fails G1 in version 1; CONT simple', not M3['usefulness_R1']['RF1_RAWJ_HI']['G1'] and r1s['CONT']['kind'] == 'SIMPLE'
   and M3['usefulness_R1']['RF1_RAWJ_HI']['worst_over_cap']['0'] <= 1 < M3['usefulness_R1']['RF1_RAWJ_HI']['worst_over_cap']['1'])
ok('RD:128 R1 refit genuinely effective', XR['d2r']['refit'] == 'REFIT_GENUINELY_EFFECTIVE' and XR['refit_checks'] == 6912)
ok('RD:129 R1 sign changes = RF1_ALTC_HI, SUCC', sorted(XR['d2r']['sign_changes']) == ['RF1_ALTC_HI', 'SUCC'])
ok('RD:130-134 R1 robustness recipient failures and views (resolved worse where stated)',
   xr['RF1_RAWJ_HI']['recipient_robustness'] == {'income': 'RECIPIENT_AGG_POINT_FAILED', 'occupation': 'RECIPIENT_AGG_RESOLVED'}
   and xr['H_LJ']['recipient_robustness']['occupation'] == 'RECIPIENT_AGG_RESOLVED_WORSE' and fails(xr['H_LJ']) == ['occupation']
   and xr['RETIRED_B3CTL']['recipient_robustness'] == {'income': 'RECIPIENT_AGG_POINT_FAILED', 'occupation': 'RECIPIENT_AGG_RESOLVED_WORSE'}
   and xr['SUCC']['recipient_robustness']['occupation'] == 'RECIPIENT_AGG_RESOLVED_WORSE' and fails(xr['SUCC']) == ['occupation'])
ok('RD:136-137 R1 READER_SENSITIVE_ENHANCED flags for STOP, ALTC, RAWJ; none passed', all(f'READER_SENSITIVE_ENHANCED_{c}' in dr['D2']['flags'] for c in ('RF1_STOP_HI', 'RF1_ALTC_HI', 'RF1_RAWJ_HI'))
   and dr['D2']['passed'] == [])
ok('RD:139-144 FC_R1 label, spend state, men Brier resolved higher, income alias, occupation edge, not lower than v1',
   FC['label'] == 'ADAPTIVE_R1__FAMILY_CHANGE_NOT_POSITIVE' and FC['spend_state'] == 'R1_SPENDS_MORE_UNRESOLVED' and FC['secondary_resolved_higher'] == ['occupation|men_brier']
   and FC['recipient']['income']['label'] == 'ALIAS_ZERO' and FC['recipient']['occupation']['label'] == 'RECIPIENT_AGG_EDGE' and FC['benefit_mean'] < 0
   and abs(FC['primary_cost_difference']['value'] - FC['secondary']['occupation|nll']['value']) < 1e-12 and FC['secondary']['income|nll']['value'] == 0.0)
ok('RD:145-146 R1 card committed after the first MODEL_SELECT read', DR['registration_status']['card_first_commit'] == '2026-10-25T18:23:27-04:00'
   and DR['registration_status']['first_model_select_read'] == '2026-10-25T21:59:53+00:00')
# learning curve
ok('RD:150 half-budget validity effective (both)', DV['lc_grid_validity']['label'] == DR['lc_grid_validity']['label'] == 'READER_SEEDS_EFFECTIVE')
ok('RD:152-154 v1 LC limitations and START larger at full budget', dv['LC']['limitations'] == ['RF_TASK', 'RF_ALTC_HI', 'H_LJ']
   and all(lcv[c]['label'] == 'BENEFIT_SHRINKS_POINT' for c in dv['LC']['limitations']) and lcv['NEURAL_RAW']['change_mean'] > 0)
ok('RD:155-157 R1 LC: H_LJ only; STOP / ALTC fell from non-positive full benefits', dr['LC']['limitations'] == ['H_LJ'] and lcr['H_LJ']['full_label'] == 'AGG_POSITIVE_POINT_ONLY'
   and lcr['RF1_STOP_HI']['full_label'] == lcr['RF1_ALTC_HI']['full_label'] == 'AGG_NOT_POSITIVE' and not lcr['RF1_STOP_HI']['limitation'] and not lcr['RF1_ALTC_HI']['limitation'])
ok('RD:158 / SH:16 inner episodes n_A 4,096, n_E about 2,400', all(FIT[k]['full_logs_FULL'][0]['n_A'] == 4096 and 2399 <= FIT[k]['full_logs_FULL'][0]['n_E'] <= 2402 for k in FIT))
# falsifiers
ok('RD:165 MECHANISM_ACTIVE in every fit', all(FIT[k]['mechanism_gate']['label'] == 'MECHANISM_ACTIVE' for k in FIT))
ok('RD:165 "stays at or above 0.086" at full precision', F1min >= 0.086, f'minimum logged share {F1min:.6f} (R1 version 0) rounds to 0.086 but is below it')
ok('RD:167 F2 triggered both; no resolved benefit over STOPFIT', all(D['falsifiers']['per_falsifier']['F2'] == 'TRIGGERED' and not D['D2']['comparisons']['STOPFIT']['pass'] for D in (dv, dr)))
ok('RD:168-172 F3 triggered both; tree/table mean includes LOOKUP 0.0',
   all(D['falsifiers']['per_falsifier']['F3'] == 'TRIGGERED' for D in (dv, dr))
   and all(abs(F['F3']['tree_table_mean'] - sum(F['F3']['mean_gain_over_stopfit_by_recipe'][r] for r in ('CATI1', 'ET25', 'HGB15', 'LOOKUP')) / 4) < 1e-15 for F in (FV, FR)))
ok('RD:174 F4 triggered both', all(D['falsifiers']['per_falsifier']['F4'] == 'TRIGGERED' for D in (dv, dr)))
ok('RD:175-178 F5 not triggered; orientation fraction 0.0 everywhere; v1 START resolved; R1 START point only',
   all(D['falsifiers']['per_falsifier']['F5'] == 'NOT_TRIGGERED' for D in (dv, dr)) and v1s['START']['label'] == 'AGG_POSITIVE' and r1s['START']['label'] == 'AGG_POSITIVE_POINT_ONLY')
ok('RD:179 F6 not triggered; version 0 had 2-3 logged steps and 1 monitor', all(D['falsifiers']['per_falsifier']['F6'] == 'NOT_TRIGGERED' for D in (dv, dr))
   and FV['F6_V0']['logged_steps'] == 2 and FR['F6_V0']['logged_steps'] == 3 and FV['F6_V0']['monitors'] == FR['F6_V0']['monitors'] == 1)
ok('RD:165-166 F0 and F1 not triggered both', all(D['falsifiers']['per_falsifier'][f] == 'NOT_TRIGGERED' for D in (dv, dr) for f in ('F0', 'F1')))
# strong references
def gates(src, ref):
    return {vw: lab for vw, lab in src[ref]['recipient_labels'].items()}


ok('RD:187 / SH:95-97 v1 D3 table', dv['D3']['passed'] == ['NEURAL_RAW', 'W', 'TR100', 'TT100']
   and dv['D3']['per_reference']['TWJ']['benefit'] == 'AGG_POSITIVE' and gates(mgc, 'TWJ')['occupation'] == 'RECIPIENT_AGG_POINT_FAILED'
   and all(dv['D3']['per_reference'][c]['benefit'] == 'AGG_POSITIVE_POINT_ONLY' and gates(mgc, c)['occupation'] == 'RECIPIENT_AGG_POINT_FAILED' for c in ('B2TWIN', 'H_LJ', 'SUCC'))
   and dv['D3']['per_reference']['RETIRED_B3CTL']['benefit'] == 'AGG_NOT_POSITIVE')
ok('RD:188 R1 D3 table', dr['D3']['passed'] == ['W', 'TR100', 'TT100'] and dr['D3']['per_reference']['NEURAL_RAW']['benefit'] == 'AGG_POSITIVE_POINT_ONLY'
   and dr['D3']['per_reference']['TWJ']['benefit'] == 'AGG_POSITIVE' and gates(m3c, 'TWJ')['occupation'] == 'RECIPIENT_AGG_POINT_FAILED'
   and all(dr['D3']['per_reference'][c]['benefit'] == 'AGG_POSITIVE_POINT_ONLY' and gates(m3c, c)['occupation'] == 'RECIPIENT_AGG_POINT_FAILED' for c in ('B2TWIN', 'H_LJ', 'SUCC'))
   and dr['D3']['per_reference']['RETIRED_B3CTL']['benefit'] == 'AGG_NOT_POSITIVE')
ok('RD:190-191 the five tree references share one occupation output (identical grid occupation means)',
   all(gv[c]['occupation'] == gv['H_LJ']['occupation'] and gr[c]['occupation'] == gr['H_LJ']['occupation'] for c in TREE))
incv = {(ref, v): inc_r[('v1', ref, v)] for ref in ('B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100') for v in (0, 1)}
ok('RD:193 registered income gates pass for every reference; largest income r = RETIRED_B3CTL version 0',
   all(gates(src, c)['income'] in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO') for src in (mgc, m3c) for c in ('B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100'))
   and max(incv, key=incv.get) == ('RETIRED_B3CTL', 0))
ok('RD:194 RETIRED_B3CTL income gate fails on the robustness cells (both readings)', xv['RETIRED_B3CTL']['recipient_robustness']['income'] == xr['RETIRED_B3CTL']['recipient_robustness']['income'] == 'RECIPIENT_AGG_POINT_FAILED')
ok('RD:196-198 reference utility (EVAL_MS) and its tightest condition', all(abs(ref_util(c, v)['max'] - ref_util('H_LJ', v)['max']) < 1e-12 for c in ('B2TWIN', 'SUCC', 'TWJ') for v in (0, 1))
   and all(abs(ref_util('H_LJ', v)['max'] - ref_util('H_LJ', v)['men'] / 0.005) < 1e-12 for v in (0, 1)) and round(ref_util('RETIRED_B3CTL', 1)['max'], 3) == 0.883
   and abs(mu_cfg('H_LJ', 0)['max'] - ref_util('H_LJ', 0)['max']) < 1e-12)
ok('RD:203-205 D4 not met (both), per-reference TARGET_AGG_NOT_MET', dv['D4']['label'] == 'NOT_MET' and dr['D4']['label_unprefixed'] == 'NOT_MET'
   and all(x['label'] == 'TARGET_AGG_NOT_MET' for D in (dv, dr) for x in D['D4']['per_reference'].values()))
# registered expectations
ok('RD:219-224 PROTOCOL 12 probabilities in order', PP == ['0.7', '0.9', '0.10', '0.06', '0.02', '0.01'])
occ_alias = {}
for bank in ('RFA1_MAIN', 'RFA1_ROB', 'RFA1_LC', 'RFA1_R1_MAIN', 'RFA1_R1_ROB', 'RFA1_R1_LC'):
    for v in (0, 1):
        ea = jl(PRIV / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json').get('exact_aliases') or {}
        occ_alias[f'{bank}/S{v}'] = [grp for vw in ('occupation', 'pair') for grp in (ea.get(vw) or [])]
ok('RD:220 CAND and STOPFIT occupation wires differ; no occupation or pair exact alias of RF_FULL / RF_STOP_HI or RF1_FULL / RF1_STOP_HI in any grid',
   all(MU['income_identity'][v]['max_abs_RF_FULL_minus_RF_STOP_HI_occupation'] > 1e-3 for v in ('0', '1'))
   and not any(({'RF_FULL', 'RF_STOP_HI'} <= set(grp)) or ({'RF1_FULL', 'RF1_STOP_HI'} <= set(grp)) for gs in occ_alias.values() for grp in gs)
   and len(occ_alias) == 12, f'occupation/pair alias groups: {sorted({tuple(g) for gs in occ_alias.values() for g in gs})}')
mlo = 'Most likely outcome: CAND lowers NEURAL_RAW\'s occupation view by 0.005-0.015 at the point, close to STOPFIT and ALT at matched spend;\n  a bounded negative experiment.'
ok('RD:226-227 most likely outcome quoted verbatim', mlo in PROTO and 'CAND lowers NEURAL_RAW\'s occupation view by 0.005-0.015 at the point, close to STOPFIT and' in LINES['RD'][225]
   and 'ALT at matched spend; a bounded negative experiment."' in LINES['RD'][226])
ok('RD:228 version 1 just above 0.015, version 0 inside', 0.005 <= occ_fall['RF_FULL'][0] <= 0.015 < occ_fall['RF_FULL'][1])
ok('RD:234-246 card v1: all 11 predictions in card order', len(C1P) == 11 and [x.split('|')[2].strip() for x in LINES['RD'][235:246]] == [p for _, p in C1P])
ok('RD:250-260 card R1: all 9 predictions in card order', len(CRP) == 9 and [x.split('|')[2].strip() for x in LINES['RD'][251:260]] == [p for _, p in CRP])
ok('RD:237 k* S20 / S70', FIT[('v1', 0)]['selection']['k_star'] == 20 and FIT[('v1', 1)]['selection']['k_star'] == 70)
ok('RD:252 R1 shares not larger than v1', f_orth[('R1', 0)] < f_orth[('v1', 0)] and f_orth[('R1', 1)] < f_orth[('v1', 1)])
ok('RD:256 prediction not met, only CATI1 fell more (verifier exact means; record flags)', M4['prediction_met'] is False and PRM['prediction_met'] is False
   and {r: x['R1_fell_more'] for r, x in M4['prediction'].items()} == {'ET25': False, 'CATI1': True, 'HGB15': False})
ok('RD:260 R1 LC limitation again (H_LJ, point)', dr['LC']['label_unprefixed'] == 'LEARNABILITY_LIMITATION' and lcr['H_LJ']['label'] == 'BENEFIT_SHRINKS_POINT')
# START_HERE extra
ok('SH:3 campaign start 20:58:41Z', RACC['start_utc'] == '2026-10-25T20:58:41+00:00' and 'Campaign start 2026-10-25T20:58:41Z' in PROTO)
ok('SH:7-8 six fields and classes as PROTOCOL', 'AGEP, WKHP, COW, SCHL, MAR, POBP' in PROTO and 'income class (2) and an occupation\n  group (5)' in PROTO)
ok('SH:34 "a turn of about 6 degrees" (arccos of cos_tot_dir, v1)', all(5.5 <= x <= 6.5 for x in deg), f'{deg}')
ok('SH:53 envelope identity value as recorded (card v1 section 3: 1.4e-14)', '1.4e-14' in CARD1 and 'checked to 1.4e-14' in TEXT['SH'],
   'the card and the investigator hand-back record 1.4e-14; the document says "checked to 1e-14" (order of magnitude)')
ok('SH:56 share never below 0.05 before k*', F1min >= 0.05)
cw_v1 = sorted({GV[c]['RF_FULL']['pair']['winner'] for c in REG})
cw_r1 = sorted({GR[c]['RF1_FULL']['pair']['winner'] for c in REG})
ok('SH:80 pair winners: categorical-interaction, extra-trees, MLP and ignore-one (v1 RF_FULL)', cw_v1 == ['CATI1', 'ET25', 'IGNORE_income__ET25', 'MLP64'],
   f'v1 RF_FULL {cw_v1}; R1 RF1_FULL {cw_r1} (no ignore-one winner in R1)')
ok('SH:94-96 NEURAL_RAW resolved below in v1 only', dv['D3']['per_reference']['NEURAL_RAW']['benefit'] == 'AGG_POSITIVE' and dr['D3']['per_reference']['NEURAL_RAW']['benefit'] != 'AGG_POSITIVE')
ok('SH:108 k* S20 / S70 (v1) and S30 / S60 (R1)', [FIT[k]['selection']['k_star'] for k in (('v1', 0), ('v1', 1), ('R1', 0), ('R1', 1))] == [20, 70, 30, 60]
   and 'S20 / S70 (v1) and S30 / S60 (R1)' in LINES['SH'][107])
# HANDOFF extra
ok('HO:9 base 913eb5c91 = DRF1 research tip (PROTOCOL 0)', 'research tip 913eb5c91 = the base of this branch' in PROTO)
ok('HO:11-12 registration commit and receipt', RCPT['research_commit'] == '43830ae19' and RCPT['receipt_commit'].startswith('34dc42b51')
   and RCPT['remote_sha_verified'] and RCPT['verified_utc'] == '2026-10-25T21:53:27+00:00')
ok('HO:11-12 / HO:64 every label read after the receipt verification', min(x['utc'] for x in ACCESS_LOG) > RCPT['verified_utc'] and ACC['label_reads_before_registration_receipt'] == 0
   and len(ACCESS_LOG) == ACC['label_reads'] == 40)
ok('HO:12 / HO:5 PUBLICATION_RECEIPT.json exists', (RES / 'PUBLICATION_RECEIPT.json').is_file(), 'not yet written (publication pending; expected at task 4)')
ok('HO:14-15 earlier records kept', all((C / d / 'DECISIONS.json').is_file() for d in ('RFA_v1', 'RFA_v1_rebuilt', 'RFA_R1')))
lab_keys = ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'LC')
reb, first = jl(C / 'RFA_v1_rebuilt' / 'DECISIONS.json')['decisions'], jl(C / 'RFA_R1' / 'DECISIONS.json')['decisions']
ok('HO:17 / deviation 10 final records: labels unchanged from the earlier records (D2R attached)',
   all(reb[k]['label'] == dv[k]['label'] and first[k]['label'] == dr[k]['label'] for k in lab_keys)
   and reb['outcome_class']['label'] == dv['outcome_class']['label'] and first['outcome_class']['label'] == dr['outcome_class']['label']
   and reb['D2R']['label'] == first['D2R']['label'] == 'NOT_RUN')
ok('HO:18 / HO:60 24 configurations, 12 per card, n_saved_new 6 per fit', len(RACC['configurations']['configurations']) == 24 and nv1 == nr1 == 12
   and all(FIT[k]['n_saved_new'] == 6 for k in FIT) and RACC['internal_fit_counters_from_units']['configurations_saved'] == 24)
pk = sorted(DEP['packages'])
ok('HO:20-21 six deployment packages', pk == sorted(f'{c}_V{v}' for c in ('RF_FULL', 'RF_STOP_HI', 'RF1_FULL') for v in (0, 1)) and DEP['all_bitwise'] and DEP['all_refused'], f'{pk}')
ok('HO:25-26 FD checks about 1e-9 (cards 1.9e-10 / 1.2e-9 / 8.8e-10; verifier 2.2e-10 .. 1.4e-9)', all(x in CARD1 for x in ('1.9e-10', '1.2e-9')) and '8.8e-10' in CARDR)
ok('HO:27-28 bounds "about 0.012 / 0.0025 ... in either reading"', abs(br['STOPFIT']['upper'] - 0.012) < 0.001 and abs(br['ALTCRIT']['upper'] - 0.0025) < 0.0005,
   f"the bounds are v1's; R1's are {br['STOPFIT']['upper']:+.4f} (STOPFIT) and {br['ALTCRIT']['upper']:+.4f} (ALTCRIT)")
ok('HO:31 components women Brier and occupation NLL; k* 20-70', {c for b in cs_bind.values() for c in b['failing_components']} == {'occ_brier_women', 'occ_nll'})
dev_named = {1: 'ALTCRIT inner-fit count', 2: 'refusal', 3: 'log label', 4: 'time stamp', 5: 'card draft', 6: 'rebuilt v1 record', 7: '6.8e-8',
             8: 'episode count', 9: 'family-change field', 10: 'final records carrying the second splits'}
ho_def = ' '.join(LINES['HO'][35:42])
dev_missing = [n for n, kw in dev_named.items() if kw not in ho_def]
ok('HO:36-42 known record defects list names every deviation 1-10', not dev_missing,
   f'deviations not named: {dev_missing} (2 ReaderRefused not caught, 3 log-label cadence, 4 coordinator time stamps, 5 interrupted card draft)')
ok('HO:46-60 resources (RESOURCE_ACCOUNTING)', RACC['units_started_without_end'] == [] and RACC['processes_left_running'] == [] and RACC['heavy_process_limit'] == 2
   and RACC['threads_per_job'] == 1 and RACC['cloud_or_paid_compute_usd'] == 0
   and sorted(x['unit'] for x in RACC['failed_units']) == ['RFA_VER_USEFULNESS_v1', 'RFA_VER_W4_R1ROB_v1']
   and abs(sum(st.values()) + sum(x['charged_cpu_s'] for x in RACC['agent_charges']) - RACC['charged_cpu_h'] * 3600) < 3.6)
ok('HO:64-68 data access (ACCESS_RECEIPT)', ACC['CALIBRATION_label_reads'] == [] and ACC['ATTACK_or_DEV_REPORT_label_reads'] == []
   and ACC['MODEL_SELECT_reads_outside_benchmark_readers_verify'] == [] and ACC['fitting_reads_with_card_committed_after_the_read'] == []
   and ACC['lineage_dev_report_log']['unchanged'] and all(x.get('unit_id') or x.get('unit') for x in ACCESS_LOG)
   and not any(x['role'] == 'MODEL_SELECT' and x['operation'] == 'fit' for x in byop) and nfit == [2])
ok('HO:72-74 final backup', BK['tip_commit'].startswith('42ef29832') and BK['all_ok'] and sorted(BK['restore_check_from_backup']['passes']) == ['python3', 'torch_venv']
   and BK['restore_check_from_backup']['all_ok'] and '20261025T232718Z' in BK['backup'])
ok('HO:75-80 off-device copy WARM, manifest equal, drive identity', OFF['all_ok'] and OFF['manifest_equal_to_source'] and OFF['read_back'].startswith('WARM')
   and OFF['drive']['name'] == 'YOTUO' and OFF['drive']['volume_uuid'].startswith('3D74932B') and OFF['tip_commit'] == BK['tip_commit'],
   f"copy size {OFF['size_bytes']:,} vs backup {BK['size_bytes']:,} bytes (not quoted in HANDOFF; for the closeout check)")
# referenced files
refs = ['roles/statistical_reviewer/HANDBACK_2.md', 'roles/statistical_reviewer/HANDBACK_3.md', 'MEASUREMENT_CONTRACT.md', 'PROTOCOL.md', 'cards/RF_RFIT_v1.md',
        'cards/RF_RFIT_R1.md', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES_ADDENDUM_1_R1.md', 'PROTOCOL_DEVIATIONS.md', 'QUICKSTART.md',
        'HANDOFF.md', 'claims/CAMPAIGN_SUMMARY.json', 'claims/PER_RECIPE_PAIR_MEANS.json', 'claims/RFA_v1_final/FALSIFIER_DETAIL.json',
        'claims/RFA_R1_final/FALSIFIER_DETAIL.json', 'claims/RFA_R1_final/FAMILY_CHANGE_vs_RF_FULL.json', 'TABLES.md', 'ENDPOINTS.json',
        'deploy/DEPLOYMENT_CHECK_MAIN.json', 'RESOURCE_ACCOUNTING.json', 'ACCESS_RECEIPT.json', 'custody/BACKUP_VERIFICATION.json', 'custody/OFFDEVICE_COPY.json']
ok('referenced files exist', all((RES / p).is_file() for p in refs), str([p for p in refs if not (RES / p).is_file()]))
ok('RD:3 / RD:4 reviewer hand-backs and deviation 10 exist', '10. **Final decision records carry the second split' in DEVS)

# -------------------------------------------------------------------------------------------------------- reviewer REQUIRED R1-R7
req = {}
sh_t, rd_t = TEXT['SH'], TEXT['RD']
req['R1'] = {'not_confirmed_sentence_RD': 'the registered benefit over NEURAL_RAW was\nnot confirmed on the second split' in rd_t or
             'the registered benefit over NEURAL_RAW was not confirmed on the second split' in rd_t.replace('\n', ' '),
             'not_confirmed_sentence_SH': 'The registered benefit over NEURAL_RAW was not confirmed on the second split' in sh_t.replace('\n  ', ' '),
             'SUCC_sign_change_both': 'SUCC (+0.0037 to\n    -0.0036)' in rd_t or 'SUCC (+0.0037 to -0.0036)' in rd_t.replace('\n    ', ' '),
             'SUCC_sign_change_SH': '(+0.0037\n       to -0.0036)' in sh_t or '+0.0037 to -0.0036' in sh_t.replace('\n       ', ' '),
             'four_failures_with_views_RD': all(x in rd_t for x in ('RF_RAWJ_HI (income view)', 'H_LJ (occupation)', 'RETIRED_B3CTL (both views)', 'SUCC (occupation)')),
             'four_failures_with_views_SH': 'RF_RAWJ_HI (income), H_LJ (occupation), RETIRED_B3CTL (both views) and SUCC' in sh_t.replace('\n       ', ' '),
             'replacement_sentence_SH': 'the point edges over STOPFIT and ALTCRIT changed\n  sign, and the benefit over NEURAL_RAW was not confirmed' in sh_t,
             'old_phrase_gone': 'small point edges reversed' not in sh_t}
req['R2'] = {'final_records_D2R': dv['D2R']['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and dr['D2R']['label'] == 'ROBUSTNESS_NOT_CONFIRMED',
             'flags_in_records': all(f in D['outcome_class']['flags'] for D in (dv, dr) for f in ('D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE', 'D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE')),
             'flags_in_ENDPOINTS': all(f in E_['outcome_flags'] for E_ in (EV, ER) for f in ('D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE', 'D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE')),
             'R1_D2R_reported_not_pending': 'pending' not in (rd_t + sh_t).lower() and (C / 'RFA_R1_D2R' / 'D2R.json').is_file()}
req['R3'] = {'not_separated_wording': 'was not separated from any mechanism-specific control' in rd_t and '"Not separated" is not "no effect"' in rd_t,
             'old_wording_gone': 'beyond any mechanism-specific control at comparable usefulness' not in rd_t and 'It did not lower recovery' not in rd_t,
             'comparable_usefulness_disclosed': '"At comparable usefulness" does not hold for every control' in rd_t and 'RF_RAWJ_HI max value/cap 1.059' in rd_t
             and 'RF1_RAWJ_HI fails G1 in version 1 (max value/cap 1.137)' in rd_t and 'CONT (RF_TASK, at S_k\\*) is not spend-matched' in rd_t,
             'ALT_not_feasible_disclosed': 'The ALT comparison is therefore against a control\n    that is not itself usefulness-feasible in version 1' in rd_t}
req['R4'] = {'new_sentence': "R1's learning-curve limitation is against H_LJ only (-0.0067, se 0.0116, point only)" in rd_t
             and 'also fell, but from non-positive full-budget benefits, so they are not limitations' in rd_t.replace('\n  ', ' '),
             'old_sentence_gone': 'Only the benefit against H_LJ shrank at the point' not in rd_t}
req['R5'] = {'fit_path_judged_on_outputs': 'CAND and STOPFIT outputs differ beyond rounding | 0.9 | yes: the occupation wires differ' in rd_t,
             'card_v1_all_11': chk['RD:234-246 card v1: all 11 predictions in card order']['pass'],
             'card_R1_all_9': chk['RD:250-260 card R1: all 9 predictions in card order']['pass']}
req['R6'] = {'no_evidence_wording': 'No evidence that the fit path helps.** The candidate was not separated from the stopped-fit control (development bound about\n     0.012 on this grid) or from a lagging critic of the same family' in sh_t,
             'reproduces_gone': 'reproduces the gain' not in sh_t.lower() and 'the fit path does not help' not in sh_t.lower(),
             'by_this_evidence': 'are\n     not supported by this evidence' in sh_t}
req['R7'] = {'margins_and_values': 'F3: triggered in both readings,** on tiny descriptive margins with no uncertainty' in rd_t and '+0.0007, against a tree/table mean of -0.0002' in rd_t
             and 'R1: +0.0003 against -0.0005' in rd_t,
             'LOOKUP_structural_zero': "LOOKUP's gain is a structural 0.0, and that zero enters the tree/table mean" in rd_t,
             'different_contrast': "The R1 card's per-recipe falls are measured against START, a different contrast from F3's" in rd_t,
             'F3_cited_elsewhere': [d for d in ('SH', 'HO') if re.search(r'\bF3\b', TEXT[d])]}
req_pass = {k: all(x if isinstance(x, bool) else (x == []) for x in v.values()) for k, v in req.items()}

# --------------------------------------------------------------------------------------------------------- label tokens vs records
corpus = set()


def harvest(x):
    if isinstance(x, dict):
        for k, v in x.items():
            corpus.add(str(k))
            harvest(v)
    elif isinstance(x, list):
        for v in x:
            harvest(v)
    elif isinstance(x, str):
        corpus.update(re.findall(r'[A-Za-z0-9_]+', x))


for obj in (DV, DR, XV, XR, CS, PRM, PV, PR, END, RACC, ACC, FC, jl(C / 'RFA_v1_final' / 'FALSIFIER_DETAIL.json'), BK, OFF, DEP):
    harvest(obj)
for f in RES.rglob('*'):
    corpus.update(re.findall(r'[A-Za-z0-9_]+', f.name))
for t in (PROTO, CARD1, CARDR, DEVS, HB3, (RES / 'roles' / 'statistical_reviewer' / 'RULES.md').read_text(), (RES / 'MEASUREMENT_CONTRACT.md').read_text(),
          (RES / 'roles' / 'statistical_reviewer' / 'RULES_ADDENDUM_1_R1.md').read_text(), (ROOT / 'rfa' / 'rules.py').read_text()):
    corpus.update(re.findall(r'[A-Za-z0-9_]+', t))
labels = {}
for d, lines in LINES.items():
    for i, line in enumerate(lines, 1):
        for t in LAB.findall(line):
            if t not in corpus:
                labels.setdefault(f'{d}:{i}', []).append(t)

# ---------------------------------------------------------------------------------------------------- TABLES.md / ENDPOINTS.json
te = {}
for E_, D, X, fits, g, name in ((EV, dv, XV, ('v1',), gv, 'RFA_v1'), (ER, dr, XR, ('R1',), gr, 'RFA_R1')):
    te[f'{name}|labels'] = all(E_[k] == D[k]['label'] for k in lab_keys) and E_['outcome_class'] == D['outcome_class']['label'] and E_['D2R'] == X['d2r']['label']
    te[f'{name}|D2_bounds_exact'] = E_['D2_bounds'] == D['D2']['bounds']['per_comparison']
    te[f'{name}|D3_rounded'] = all(E_['D3_per_reference'][c] == {'benefit': x['benefit'], 'mean': round(x['mean'], 4), 'recipient_gate_pass': x['recipient_gate_pass'],
                                                                 'se_agg': round(x['se_agg'], 4)} for c, x in D['D3']['per_reference'].items())
    te[f'{name}|D2R_per_comparator'] = all(E_['D2R_per_comparator'][c] == {k: p.get(k) for k in ('label', 'registered_mean', 'robustness_mean', 'robustness_se', 'sign_change')}
                                           for c, p in X['d2r']['per_comparator'].items())
    te[f'{name}|grid_version_means'] = all(E_['grid_version_means'][m][vw] == [round(x, 4) for x in g[m][vw]] for m in g for vw in ('pair', 'income', 'occupation'))
    te[f'{name}|falsifiers_LC'] = E_['falsifiers'] == D['falsifiers']['per_falsifier'] and E_['LC_limitations'] == D['LC']['limitations']
    rk = fits[0]
    te[f'{name}|fit'] = all(E_['fit'][str(v)]['k_star'] == FIT[(rk, v)]['selection']['k_star'] and E_['fit'][str(v)]['gate_orthogonal_share'] == FIT[(rk, v)]['mechanism_gate']['orthogonal_share']
                            and E_['fit'][str(v)]['cand_spend_over_start'] == FIT[(rk, v)]['cand']['spend_over_start'] for v in (0, 1))
te['RFA_v1|utility_vs_verifier'] = all(abs(EV['utility'][c][str(v)]['max_over_cap'] - round(mu_cfg(c, v)['max'], 3)) < 1e-12
                                       and EV['utility'][c][str(v)]['men_guard'] == round(mu_cfg(c, v)['men'], 5)
                                       for c in ('RF_FULL', 'RF_TASK', 'RF_STOP_LO', 'RF_STOP_HI', 'RF_ALTC_HI', 'RF_RAWJ_HI', 'NEURAL_RAW') for v in (0, 1))
te['RFA_R1|utility_vs_verifier'] = all(abs(ER['utility'][c][str(v)]['max_over_cap'] - round(r1_util(c, v), 3)) < 1e-12
                                       for c in ('RF1_FULL', 'RF1_TASK', 'RF1_STOP_LO', 'RF1_STOP_HI', 'RF1_ALTC_HI', 'RF1_RAWJ_HI') for v in (0, 1))
te['RFA_R1|family_change'] = ER['family_change_vs_RF_FULL'] == {k: FC.get(k) for k in ER['family_change_vs_RF_FULL']}
te['decision_records_are_final'] = EV['decision_record'] == 'claims/RFA_v1_final/DECISIONS.json' and ER['decision_record'] == 'claims/RFA_R1_final/DECISIONS.json'
TB = (RES / 'TABLES.md').read_text()
tbl = []
for E_, D, name in ((EV, dv, 'RFA_v1'), (ER, dr, 'RFA_R1')):
    for k, b in E_['D2_bounds'].items():
        tbl.append(f"| {k} | {b['at']} | {b['mean']:+.4f} | {b['se_agg']:.4f} | {b['lower']:+.4f} | {b['upper']:+.4f} |")
    for c, x in E_['D3_per_reference'].items():
        tbl.append(f"| {c} | {x['mean']:+.4f} | {x['se_agg']:.4f} | {x['recipient_gate_pass']} | {x['benefit']} |")
    for c, p in E_['D2R_per_comparator'].items():
        if p['label'] != 'NOT_RUN':
            tbl.append(f"| {c} | {p['label']} | {p['registered_mean']:+.4f} | {p['robustness_mean']:+.4f} | {p['robustness_se']:.4f} | {p['sign_change']} |")
    for m, x in E_['grid_version_means'].items():
        tbl.append(f"| {m} | {x['pair']} | {x['occupation']} | {x['income']} |")
for E_, D, X, rk, name, cand in ((EV, dv, XV, 'v1', 'RFA_v1', 'RF_FULL'), (ER, dr, XR, 'R1', 'RFA_R1', 'RF1_FULL')):
    for m, x in E_['utility'].items():
        tbl.append(f"| {m} | {x['0']['point_pass']} / {x['1']['point_pass']} | {x['0']['max_over_cap']} / {x['1']['max_over_cap']} | "
                   f"{x['0']['men_guard']} / {x['1']['men_guard']} |")
    for m, x in E_['reference_utility'].items():
        tbl.append(f"| {m} | {x['0']['point_pass']} / {x['1']['point_pass']} | {x['0']['max_over_cap']} / {x['1']['max_over_cap']} | "
                   f"{x['0']['men_guard']} / {x['1']['men_guard']} |".replace('NaN', 'nan'))
    tbl.append(f"Falsifiers: {D['falsifiers']['per_falsifier']}. Learning-curve limitations against: {D['LC']['limitations']}.")
    tbl.append(f"D2R caveat: {X['d2r']['headline_caveat']}.")
    for v in (0, 1):
        f = FIT[(rk, v)]
        k = f['selection']['k_star']
        sc = {a: round(f['selection']['path'][f'{a}_S{k}']['tot_nll']['estimate'], 6) for a in ('FULL', 'STOPFIT', 'ALTCRIT', 'TASK')}
        tbl.append(f"- fit v{v}: k* S{k}, F0 {f['mechanism_gate']['label']} (orthogonal share {f['mechanism_gate']['orthogonal_share']:.3f}, "
                   f"cos {f['mechanism_gate']['cos_tot_dir']:.4f}), lambda {f['full_records']['FULL']['lambda']:.3f}, "
                   f"brackets {({kk: {'lo': p.get('lo'), 'hi': p.get('hi')} for kk, p in f['paths'].items()})}, "
                   f"ALTCRIT lag median {f['altcrit_rel_lag_median_deployed']}, CAND MODEL_FIT out-of-fold total-NLL spend over the start "
                   f"{f['cand']['spend_over_start']['value']:.4f} (se {f['cand']['spend_over_start']['se']:.4f}), MODEL_FIT out-of-fold total NLL over "
                   f"the fold start at S{k} {sc}, fit CPU {round(f['cpu_s'], 1)} s")
tbl.append(f"Family-change reading FC_R1 (beside; never gating): {FC['label']}, pair benefit {FC['benefit_mean']:+.4f} (se {FC['benefit_se_agg']:.4f}; "
           f"version means {FC['benefit_version_means']}), spend state {FC['spend_state']}, secondary metrics resolved higher: {FC['secondary_resolved_higher']}.")
te['TABLES_rows_present'] = [r for r in tbl if r not in TB] == []
te['TABLES_rows_missing'] = [r for r in tbl if r not in TB]
te['TABLES_rows_checked'] = len(tbl)
te['reference_utility_vs_EVAL_MS'] = all(EV['reference_utility'][c][str(v)]['max_over_cap'] == round(ref_util(c, v)['max'], 3)
                                         and EV['reference_utility'][c][str(v)]['men_guard'] == round(ref_util(c, v)['men'], 5)
                                         for c in ('B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W') for v in (0, 1))
ok('RD:239 card range "+0.003 and +0.020"', 'between +0.003 and +0.020' in CARD1)
ok('SH:56 F1 threshold 0.05 (card section 6)', 'the orthogonal share below 0.05 before S_k*' in CARD1)
ok('HO:60 limit 24 (PROTOCOL 10)', 'At most 24 new version-specific release specifications' in PROTO)
ok('RD:204 D4 target 0.020 (PROTOCOL 6)', 'D4 = 0.020 vs NEURAL_RAW and H_LJ' in PROTO)

res = {'schema': 'RFA1-verifier-conformance-v4', 'documents_sha256': {f: sha(RES / f) for f in DOCS.values()},
       'records_sha256': {p: sha(RES / p) for p in ('claims/RFA_v1_final/DECISIONS.json', 'claims/RFA_R1_final/DECISIONS.json', 'claims/RFA_v1_final/FALSIFIER_DETAIL.json',
                                                     'claims/RFA_R1_final/FALSIFIER_DETAIL.json', 'claims/RFA_R1_final/FAMILY_CHANGE_vs_RF_FULL.json',
                                                     'claims/RFA_v1_D2R/D2R.json', 'claims/RFA_R1_D2R/D2R.json', 'claims/CAMPAIGN_SUMMARY.json',
                                                     'claims/PER_RECIPE_PAIR_MEANS.json', 'claims/PLAN_RFA_v1.json', 'claims/PLAN_RFA_R1.json',
                                                     'fam/FIT_RF_V0.json', 'fam/FIT_RF_V1.json', 'fam/FIT_RF_R1_V0.json', 'fam/FIT_RF_R1_V1.json',
                                                     'TABLES.md', 'ENDPOINTS.json', 'RESOURCE_ACCOUNTING.json', 'ACCESS_RECEIPT.json', 'PROTOCOL.md',
                                                     'cards/RF_RFIT_v1.md', 'cards/RF_RFIT_R1.md', 'PROTOCOL_DEVIATIONS.md',
                                                     'roles/statistical_reviewer/HANDBACK_3.md')},
       'numeric_lines': numeric,
       'numeric_summary': {st_: sum(1 for x in numeric.values() if x['status'] == st_) for st_ in ('MATCH', 'MISMATCH', 'UNCHECKED', 'VERBATIM_CHECKED_BELOW')},
       'numeric_record_tokens_matched': sum(x.get('n_record', 0) for x in numeric.values() if x['status'] == 'MATCH'),
       'numeric_tokens_total': sum(len(x['tokens']) for x in numeric.values()),
       'statement_checks': chk, 'statement_failures': {k: x['note'] for k, x in chk.items() if not x['pass']},
       'reviewer_required': req, 'reviewer_required_pass': req_pass, 'labels_not_in_records': labels, 'tables_endpoints': te,
       'derived': {'same_checkpoint_spend_minus_FULL': {f'{a}|{b}|{c}': x for (a, b, c), x in same.items()}, 'same_checkpoint_z': zs,
                   'occupation_falls_vs_START_v1_bank': occ_fall, 'tree_ref_occupation_r': rv_occ, 'F1_min_share': F1min, 'turn_degrees_v1': deg,
                   'cand_pair_winners': {'v1_RF_FULL': cw_v1, 'R1_RF1_FULL': cw_r1}, 'fineness_v1': gfv['per_path'], 'fineness_R1': gfr['per_path'],
                   'grid_version_means_v1': gv, 'grid_version_means_R1': gr},
       'cpu_s': None}
res['cpu_s'] = time.process_time() - t0
OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + '\n')
print(json.dumps({'numeric_summary': res['numeric_summary'], 'record_tokens_matched': res['numeric_record_tokens_matched'],
                  'tokens_total': res['numeric_tokens_total'],
                  'mismatch': {k: v for k, v in numeric.items() if v['status'] in ('MISMATCH', 'UNCHECKED')},
                  'statement_failures': res['statement_failures'], 'reviewer_required_pass': req_pass,
                  'labels_not_in_records': labels, 'tables_endpoints_fail': {k: v for k, v in te.items() if v not in (True, [])},
                  'cpu_s': res['cpu_s']}, indent=1, default=str))

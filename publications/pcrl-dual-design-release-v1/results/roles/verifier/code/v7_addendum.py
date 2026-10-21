"""Verifier addendum, label-free part (no label is read here; deployment spot-check on admitted DEV_REPORT raw inputs only).

A1x  DB_PTC_R1_OCC: income bitwise DA_TWIN's and occupation bitwise DB_PTC_R1's on all six roles, both versions.
A3   the revision readings DB_R1 (bank DDR1_R1) and DA_R1 (bank DDR1_DA_R1): D1-D4 re-derived with own arithmetic (v4_decisions
     functions: aggregate, benefit / recipient screens, brackets from pooled total-NLL costs, secondary metrics, cheaper-control check,
     targeting) and the adaptive tag from the card's first commit against the first MODEL_SELECT read; compared with DECISIONS.json.
A4   D2R records: per comparator (TWIN, the release's bracketing controls; NEURAL_RAW / H_LJ beside) the robustness-cell and all-cell
     aggregates (own arithmetic on the merged registered + refit grids), refit effectiveness (own split hashes and SCORE overlaps of
     R1-R4; registered reader and bootstrap seeds; seed-dependent recipe equalities on the minimum set for k0 / k1 and k2 / k3 with
     the prospective extra-trees explanation), and the ablation comparisons; compared with D2R.json.
A5   the strongest feasible nonzero release per family (RULES.md 8; split strength alpha_I + alpha_O) from registered-cell pair
     benefits against DA_TWIN; compared with claims/STRONGEST.json.
A6   DB_PTC_R1 vs DB_ZTILT_R1 on bank DDR1_R1_ZTILT (registered cells and all 32 cells) and the ZTILT_R1 spend matching rule
     (first grid pair whose nested MODEL_FIT spend reaches the candidate's) from db/ZTILT_R1_V<v>.json.
A8   deployment spot-check: two packages loaded with ddr.deploy.release on admitted DEV_REPORT raw rows, compared bitwise with the
     saved RELEASE.npz DEV_REPORT arrays; one refusal probe each.

    python3 -m ddr.jobqueue run --unit DDR_VER_ADD_FREE_V<n> --stage verify --rss-gib 1 -- python3 <this file> --label-record <V6 record>
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402
import v4_decisions as D4  # noqa: E402

VERSIONS = (0, 1)
REG = D4.REG
ALL_REPS = ('R1', 'R2', 'R3', 'R4')
SEED_DEP = D4.SEED_DEP
MIN_LEAF = D4.MIN_LEAF
CLAIMS = V.RESULTS / 'claims'


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip()


def load_grids(*banks):
    g = {'paired': {}, 'cells': {}, 'members': {}}
    for bank in banks:
        for v in VERSIONS:
            p = V.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
            if not p.is_file():
                raise FileNotFoundError(f'{bank} S{v} GRID.json missing')
            x = json.loads(p.read_text())
            g['members'].setdefault(v, {}).update(x['members'])
            for cell, cres in x['cells'].items():
                g['cells'].setdefault(cell, {}).update(cres)
            for cand, comps in x['paired'].items():
                for comp, views in comps.items():
                    for view, e in views.items():
                        d = g['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                        d['cells'].update(e['cells']); d['cell_se'].update(e['cell_se'])
    return g


def first_ms_read():
    rows = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    return min(V.parse_utc(r['utc']) for r in rows if r['role'] == 'MODEL_SELECT')


def card_first_commit(card):
    out = git('log', '--diff-filter=A', '--format=%cI', '--', card)
    return V.parse_utc(out.splitlines()[-1]) if out else None


def g1_map(label_records):
    g1 = {}
    for p in label_records:
        r = json.loads(Path(p).read_text())
        if 'U' in r:
            g1.update({k: x['G1_24_plus_4'] for k, x in r['U']['G1'].items()})
        if 'U2' in r:
            g1.update(r['U2']['G1'])
    return g1


# ---------------------------------------------------------------- A1x
def a1x():
    out = {}
    for v in VERSIONS:
        with np.load(V.RUNS / 'DB' / 'DB_PTC_R1_OCC' / f'V{v}' / 'RELEASE.npz') as z, \
                np.load(V.RUNS / 'DA' / 'DA_TWIN' / f'V{v}' / 'RELEASE.npz') as t, \
                np.load(V.RUNS / 'DB' / 'DB_PTC_R1' / f'V{v}' / 'RELEASE.npz') as c:
            for k in z.files:
                role, task = k.split('__')
                ref = t[k] if task == 'income' else c[k]
                out[f'V{v}|{k}'] = bool(z[k].dtype == ref.dtype and z[k].shape == ref.shape and z[k].tobytes() == ref.tobytes())
    return {'checks': out, 'all_bitwise': all(out.values()), 'n': len(out)}


# ---------------------------------------------------------------- A3
def a3(g1, published):
    res = {}
    t_ms = first_ms_read()
    for pid, bank in (('DB_R1', 'DDR1_R1'), ('DA_R1', 'DDR1_DA_R1')):
        plan = json.loads((CLAIMS / f'PLAN_{pid}.json').read_text())
        grid = load_grids(bank)
        mine = D4.reading(plan, grid, g1, published)
        minimum = [plan['cand'], plan['twin']] + [x for k in ('SBC', 'SPEND') for x in plan['paths'][k]]
        mine['seed_validity'] = D4.seed_validity(grid['cells'], minimum)
        t_card = card_first_commit(plan['card'])
        kind = 'REVISED' if t_card < t_ms else 'ADAPTIVE'
        mine['tag'] = f"{kind}_R{int(plan['revision'])}" if int(plan['revision']) else (None if kind == 'REVISED' else 'ADAPTIVE_R0_LATE_CARD')
        mine['card_first_commit'] = t_card.isoformat(); mine['first_model_select_read'] = t_ms.isoformat()
        rec = {'mine': mine}
        dp = CLAIMS / pid / 'DECISIONS.json'
        if dp.is_file():
            dec = json.loads(dp.read_text())
            rec['comparison'] = D4.compare(mine, dec)
            rec['recorded_tag'] = dec['decisions'].get('adaptive_tag')
            rec['tag_agrees'] = rec['recorded_tag'] == mine['tag']
            rec['prefix_ok'] = all(str(dec['decisions'][k]['label']).startswith(f"{mine['tag']}__") for k in ('D1', 'D2', 'D3', 'D4', 'outcome_class'))
            rec['recorded_flags'] = dec['decisions']['outcome_class'].get('flags')
            rec['recorded_grid_validity'] = dec['grid_validity'].get('label')
        else:
            rec['comparison'] = 'NOT_RUN: DECISIONS.json missing'
        res[pid] = rec
    return res


# ---------------------------------------------------------------- A4
def parse(c):
    return int(c[1]), c[2:4], int(c[5:])


def agg_cells(entry, which):
    cells = [c for c in (entry or {}).get('cells', {}) if '@' not in c]
    if which == 'registered':
        sel = [c for c in cells if c in REG]
    elif which == 'robustness':
        sel = [c for c in cells if c not in REG]
    else:
        sel = cells
    if not sel:
        return None
    a = D4.agg(entry, cells=sel)
    if a is None:
        return None
    blocks = {}
    for c in sel:
        v, r, k = parse(c)
        blocks.setdefault((v, r), []).append(float(entry['cells'][c]))
    a['n_eff'] = sum(len(set(x)) for x in blocks.values())
    kd = [len(set(x)) > 1 for x in blocks.values() if len(x) > 1]
    a['qualifier'] = ('READER_SEEDS_EFFECTIVE_ALL_BLOCKS' if kd and all(kd) else 'READER_SEED_REPLICATION_ABSENT' if not any(kd)
                      else 'READER_SEEDS_PARTLY_EFFECTIVE')
    a['resolved'] = (not a['alias']) and a['lower'] > 0 and all(a['per_version'][v]['mean'] > 0 for v in VERSIONS)
    a['label'] = ('ALIAS_ZERO' if a['alias'] else 'AGG_POSITIVE' if a['resolved'] else
                  'AGG_POSITIVE_POINT_ONLY' if a['mean'] > 0 else 'AGG_NOT_POSITIVE')
    a['n_cells'] = len(sel)
    return a


def split_records(unit):
    roles = {r: V.split_masks(unit, r) for r in ALL_REPS}
    out = {}
    for r, m in roles.items():
        a = np.where(m['FIT'], 0, np.where(m['VALIDATION'], 1, 2)).astype(np.int8)
        sc = set(np.unique(unit[m['SCORE']]).tolist())
        ov = {q: len(sc & set(np.unique(unit[mq['SCORE']]).tolist())) / len(sc) for q, mq in roles.items() if q != r}
        out[r] = {'assignment_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'score_overlap': ov, 'score_households': len(sc)}
    return out


def seed_checks(cells, minimum, pairs=((0, 1), (2, 3))):
    fp = {}
    for c, cres in cells.items():
        if '@' in c:
            continue
        v, r, k = parse(c)
        for m in minimum:
            if m in cres:
                for w in ('income', 'occupation', 'pair'):
                    fp[(m, v, r, w, k)] = (cres[m][w].get('extras') or {}).get('score_sha256') or {}
    dc = {(m, v): D4.distinct_counts(m, v) for m in minimum for v in VERSIONS}
    res = {}
    for a, b in pairs:
        eqs, unexplained, checked = [], [], 0
        for m in minimum:
            for v in VERSIONS:
                for r in ALL_REPS:
                    for w in ('income', 'occupation', 'pair'):
                        f0, f1 = fp.get((m, v, r, w, a)), fp.get((m, v, r, w, b))
                        if f0 is None or f1 is None:
                            unexplained.append(f'{m}|V{v}|{r}|{w}: fingerprints missing for k{a}/k{b}')
                            continue
                        for rec in SEED_DEP:
                            if rec not in f0 or rec not in f1:
                                unexplained.append(f'{m}|V{v}|{r}|{w}|{rec}: unchecked')
                                continue
                            checked += 1
                            if f0[rec] == f1[rec]:
                                ea = any(f0.get(o) != f1.get(o) for o in SEED_DEP if o != rec and o in f0 and o in f1)
                                eb = any((fp.get((m, v, r, ww, a)) or {}).get(rec) != (fp.get((m, v, r, ww, b)) or {}).get(rec)
                                         for ww in ('income', 'occupation', 'pair') if ww != w)
                                ok = rec in MIN_LEAF and ea and eb and dc[(m, v)][w] <= 8001 // MIN_LEAF[rec]
                                eqs.append({'m': m, 'v': v, 'r': r, 'w': w, 'recipe': rec, 'explained': ok})
                                if not ok:
                                    unexplained.append(f'{m}|V{v}|{r}|{w}|{rec}: unexplained equality')
        res[f'k{a}k{b}'] = {'checked': checked, 'equalities': len(eqs), 'unexplained': unexplained[:20], 'n_unexplained': len(unexplained)}
    return res


def a4(unit):
    res = {}
    splits = split_records(unit)
    hs = [splits[r]['assignment_sha256'] for r in ALL_REPS]
    split_ok = len(set(hs)) == 4 and all(0 <= splits[r]['score_overlap'][q] < 1 for r in ('R3', 'R4') for q in ALL_REPS if q != r)
    man = json.loads((V.RESULTS / 'READER_SPLITS.json').read_text())
    want_seeds = {f'S{v}k{k}': int(man['reader_seeds'][f'V{v}k{k}']) for v in VERSIONS for k in range(4)}
    want_boot = {f'S{v}{r}': V.BOOT_BASE + 100 * v + V.REP_INDEX[r] for v in VERSIONS for r in ALL_REPS}
    for pid in ('DA_REG_D2R', 'DB_REG_D2R', 'DB_R1_D2R'):
        dp = CLAIMS / pid / 'D2R.json'
        if not dp.is_file():
            res[pid] = 'NOT_RUN: D2R.json missing'
            continue
        d = json.loads(dp.read_text())
        plan = json.loads((CLAIMS / f'PLAN_{pid}.json').read_text())
        g = load_grids(d['main_bank'], d['rob_bank']) if d['main_bank'] != d['rob_bank'] else load_grids(d['main_bank'])
        rel, twin = plan['cand'], plan['twin']
        br = plan.get('bracket_ids') or {}
        decisive = list(dict.fromkeys(['TWIN'] + [br.get('SBC'), br.get('SPEND')]))
        decisive = [x for x in decisive if x]
        sc = seed_checks(g['cells'], [rel, twin] + [x for x in decisive if x != 'TWIN'])
        refit_ok = split_ok and d.get('seeds') == want_seeds and d.get('boot_seeds') == want_boot and \
            all(x['n_unexplained'] == 0 for x in sc.values())
        per, cmpr = {}, {}
        for c in decisive + [r for r in ('NEURAL_RAW', 'H_LJ') if r not in decisive]:
            comp = twin if c == 'TWIN' else c
            e = (g['paired'].get(rel) or {}).get(comp)
            if not e:
                per[c] = {'label': 'NOT_RUN'}
                continue
            rob, allc = agg_cells(e['pair'], 'robustness'), agg_cells(e['pair'], 'all')
            has_reg = all(x in e['pair']['cells'] for x in REG)
            conf = bool(rob and allc and rob['resolved'] and allc['resolved'])
            lab = 'ROBUSTNESS_CONFIRMED' if (conf and refit_ok and has_reg) else 'ROBUSTNESS_NOT_CONFIRMED'
            per[c] = {'label': lab, 'robustness_mean': rob['mean'], 'robustness_se': rob['se_agg'], 'robustness_label': rob['label'],
                      'all_cells_label': allc['label'], 'all_mean': allc['mean'], 'all_se': allc['se_agg'], 'n_eff': rob['n_eff'],
                      'qualifier': rob['qualifier'], 'n_rob_cells': rob['n_cells'], 'n_all_cells': allc['n_cells']}
            rc = d['d2r']['per_comparator'].get(c) or {}
            cmpr[c] = {'label': (lab, rc.get('label')), 'all_cells_label': (allc['label'], rc.get('all_cells_label')),
                       'qualifier': (rob['qualifier'], rc.get('reader_seed_qualifier')), 'n_eff': (rob['n_eff'], rc.get('n_eff_robustness_cells')),
                       'd_mean': abs(rob['mean'] - rc['robustness_mean']) if rc.get('robustness_mean') is not None else None,
                       'd_se': abs(rob['se_agg'] - rc['robustness_se']) if rc.get('robustness_se') is not None else None}
        labs = [per[c]['label'] for c in decisive]
        fam = ('NOT_RUN' if any(x == 'NOT_RUN' for x in labs) else
               'ROBUSTNESS_CONFIRMED' if all(x == 'ROBUSTNESS_CONFIRMED' for x in labs) else 'ROBUSTNESS_NOT_CONFIRMED')
        abl, abl_cmp = {}, {}
        for a in plan.get('ablations') or []:
            for comp, key in ((rel, 'vs_release'), (twin, 'vs_twin')):
                e = (g['paired'].get(a) or {}).get(comp)
                if not e:
                    continue
                regb, rob, allc = agg_cells(e['pair'], 'registered'), agg_cells(e['pair'], 'robustness'), agg_cells(e['pair'], 'all')
                rl = 'ROBUSTNESS_CONFIRMED' if (rob and allc and rob['resolved'] and allc['resolved']) else 'ROBUSTNESS_NOT_CONFIRMED'
                rcp = {w: D4.recipient(e[w])['gate'] for w in ('income', 'occupation')}
                abl[f'{a}_{key}'] = {'registered_label': regb['label'], 'mean': regb['mean'], 'se': regb['se_agg'],
                                     'recipient_gates': rcp, 'robustness': rl}
                ra = (d.get('ablation') or {}).get(f'{a}_{key}')
                if ra:
                    rb = ra['registered']['benefit']
                    abl_cmp[f'{a}_{key}'] = {'label': (regb['label'], rb['label']), 'robustness': (rl, ra['robustness']['label']),
                                             'd_mean': abs(regb['mean'] - rb['aggregate']['mean']),
                                             'd_se': abs(regb['se_agg'] - rb['aggregate']['se_agg'])}
        res[pid] = {'release': rel, 'decisive': decisive, 'family_label': (fam, d['d2r']['label']), 'per_comparator': per,
                    'comparison': cmpr, 'refit': {'mine_ok': refit_ok, 'recorded': d['refit_effectiveness']['label'],
                                                  'split_ok': split_ok, 'seed_checks': sc,
                                                  'seeds_match': d.get('seeds') == want_seeds, 'boot_match': d.get('boot_seeds') == want_boot},
                    'ablation': abl, 'ablation_comparison': abl_cmp, 'recorded_release_field': d['d2r'].get('release'),
                    'banks': [d['main_bank'], d['rob_bank']]}
    res['splits'] = splits
    return res


# ---------------------------------------------------------------- A5
def a5(g1):
    s = json.loads((CLAIMS / 'STRONGEST.json').read_text())
    grids = {b: load_grids(b) for b in ('DDR1_MAIN', 'DDR1_R1', 'DDR1_DA_R1')}
    readings = {'DA': [('DA_STD', 'DDR1_MAIN', {'0': 1.0, '1': 1.0}, 0), ('DA_IPW', 'DDR1_DA_R1', {'0': 1.0, '1': 0.75}, 1)],
                'DB': [('DB_PTC', 'DDR1_MAIN', {'0': 0.35, '1': 0.35}, 0), ('DB_PTC_R1', 'DDR1_R1', {'0': 0.35 + 0.65, '1': 0.35 + 0.65}, 1)]}
    out = {}
    for fam, rows in readings.items():
        recs = []
        for cfg, bank, strength, rev in rows:
            e = grids[bank]['paired'][cfg]['DA_TWIN']['pair']
            b = D4.benefit(e)
            elig = bool(g1.get(cfg)) and all(float(x) > 0 for x in strength.values()) and b['label'] not in ('ALIAS_ZERO', 'INCOMPLETE')
            recs.append({'config_id': cfg, 'x': b.get('mean'), 'label': b['label'], 'eligible': elig, 'strength_sum': sum(strength.values()),
                         'revision': rev})
        el = [r for r in recs if r['eligible']]
        best = max(r['x'] for r in el)
        near = sorted((r for r in el if r['x'] >= best - 1e-12), key=lambda r: (r['strength_sum'], r['revision'], r['config_id']))
        rf = s['families'][fam]
        rx = {r['config_id']: r.get('x') for r in rf['records']}
        out[fam] = {'mine_selected': near[0]['config_id'], 'recorded_selected': rf['selected'], 'records': recs,
                    'max_dx': max(abs(r['x'] - rx[r['config_id']]) for r in recs if rx.get(r['config_id']) is not None),
                    'recorded_strengths': {r['config_id']: r['strength'] for r in rf['records']}}
    return out


# ---------------------------------------------------------------- A6
def a6():
    out = {'spend_matching': {}}
    for v in VERSIONS:
        z = json.loads((V.RESULTS / 'db' / f'ZTILT_R1_V{v}.json').read_text())
        pairs = sorted(z['spends'], key=lambda k: float(k.split('|')[0]))
        first = next((p for p in pairs if z['spends'][p] >= z['cand_spend']), None)
        chosen = '|'.join(repr(float(x)) for x in z['chosen'])
        rel = json.loads((V.RUNS / 'DB' / 'DB_ZTILT_R1' / f'V{v}' / 'RELEASE.json').read_text())
        ks = [float(p.split('|')[0]) / float(pairs[2].split('|')[0]) for p in pairs]          # k relative to the k = 1 pair
        out['spend_matching'][v] = {'cand_spend': z['cand_spend'], 'first_reaching_pair': first,
                                    'chosen_pair': z['chosen'], 'rule_followed': first is not None and
                                    all(abs(float(a) - float(b)) < 1e-12 for a, b in zip(first.split('|'), z['chosen'])),
                                    'chosen_spend': z['spends'].get(first), 'overshoot_ratio': z['spends'].get(first) / z['cand_spend'],
                                    'k_grid': [round(k, 6) for k in ks],
                                    'ratio_alpha_O_over_alpha_I': [float(p.split('|')[1]) / float(p.split('|')[0]) for p in pairs],
                                    'release_record_strength': {k: rel.get(k) for k in ('alpha_inc', 'alpha_occ') if k in rel},
                                    'saved_pair_equals_chosen': [float(rel.get('alpha_inc', 'nan')), float(rel.get('alpha_occ', 'nan'))]
                                    == [float(x) for x in z['chosen']]}
    p = CLAIMS / 'BESIDE_DDR1_R1_ZTILT.json'
    gp = V.RUNS / 'GRID' / 'DDR1_R1_ZTILT'
    if all((gp / f'S{v}' / 'GRID.json').is_file() for v in VERSIONS):
        g = load_grids('DDR1_R1_ZTILT')
        e = g['paired']['DB_PTC_R1']['DB_ZTILT_R1']
        out['mine'] = {w: {'registered': agg_cells(e['pair'], 'registered'), 'all': agg_cells(e['pair'], 'all')} for w in ('pair',)}
        out['mine']['recipient_registered'] = {w: D4.recipient(e[w]) for w in ('income', 'occupation')}
        out['mine_summary'] = {k: {kk: (x[kk] if kk in x else None) for kk in ('label', 'mean', 'se_agg', 'lower', 'n_cells')}
                               for k, x in out['mine']['pair'].items()}
        out['mine_summary']['version_means'] = {k: {v: x['per_version'][v]['mean'] for v in VERSIONS} for k, x in out['mine']['pair'].items()}
        e2 = load_grids('DDR1_R1_BESIDE')['paired']['DB_PTC_R1'].get('DB_ZTILT')
        if e2:
            out['vs_registered_ZTILT_registered_cells'] = {kk: agg_cells(e2['pair'], 'registered')[kk] for kk in ('label', 'mean', 'se_agg')}
        del out['mine']
    else:
        out['grid'] = 'NOT_RUN: DDR1_R1_ZTILT GRID.json missing'
    out['record'] = json.loads(p.read_text()) if p.is_file() else 'NOT_RUN: BESIDE_DDR1_R1_ZTILT.json missing'
    if isinstance(out['record'], dict) and 'mine_summary' in out:
        g = load_grids('DDR1_R1_ZTILT')
        cmp_ = {}
        for comp, rc in out['record']['comparisons'].items():
            e = g['paired']['DB_PTC_R1'][comp]
            a = agg_cells(e['pair'], 'registered')
            al = agg_cells(e['pair'], 'all')
            si = {w: agg_cells(e[w], 'registered')['mean'] for w in ('income', 'occupation')}
            sa = {w: float(np.mean([float(x) for x in e[w]['cells'].values()])) for w in ('income', 'occupation')}
            cmp_[comp] = {'label': (a['label'], rc['label']), 'd_mean': abs(a['mean'] - rc['mean']), 'd_se': abs(a['se_agg'] - rc['se_agg']),
                          'd_version_means': max(abs(a['per_version'][v]['mean'] - rc['version_means'][str(v)]) for v in VERSIONS),
                          'd_income': abs(si['income'] - rc['income_cand_minus_comp']),
                          'd_occupation': abs(si['occupation'] - rc['occupation_cand_minus_comp']),
                          'd_income_vs_plain_mean_all_cells': abs(sa['income'] - rc['income_cand_minus_comp']),
                          'd_occupation_vs_plain_mean_all_cells': abs(sa['occupation'] - rc['occupation_cand_minus_comp']),
                          'n_cells_single_views': {w: len(e[w]['cells']) for w in ('income', 'occupation')},
                          'mine_registered': (a['mean'], a['se_agg'], a['lower']), 'mine_all_32': (al['label'], al['mean'], al['se_agg'], al['lower'],
                                                                                                  al['n_cells'])}
        out['record_comparison'] = cmp_
    return out


# ---------------------------------------------------------------- A8
def a8(n_rows=25):
    from ddr import common as C
    from ddr import deploy as DP
    R = C.features('DEV_REPORT', raw=True)
    raw = R['raw_inputs']
    rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in DP.PERMITTED} for i in range(n_rows)]
    out = {}
    for pkg_name, fam, cfg, v in (('DA_STD_V1', 'DA', 'DA_STD', 1), ('DB_PTC_R1_V0', 'DB', 'DB_PTC_R1', 0)):
        pkg = V.PRIV / 'deploy' / pkg_name
        got = DP.release(str(pkg), rows)
        with np.load(V.RUNS / fam / cfg / f'V{v}' / 'RELEASE.npz') as z:
            ref = {t: z[f'DEV_REPORT__{t}'][:n_rows] for t in V.TASKS}
        same = {t: bool(np.asarray(got[t]['probability']).tobytes() == np.ascontiguousarray(ref[t]).tobytes()) for t in V.TASKS}
        dec = {t: bool(np.array_equal(got[t]['decision'], V.argmax_first(ref[t]))) for t in V.TASKS}
        probes = {}
        for label, bad in (('SEX_field', [{**rows[0], 'SEX': 1}]), ('extra_field', [{**rows[0], 'PUMA': 1}])):
            try:
                DP.release(str(pkg), bad); probes[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                probes[label] = f'refused: {type(e).__name__}'
        out[pkg_name] = {'rows': n_rows, 'bitwise_vs_saved_DEV_REPORT': same, 'decisions_equal': dec, 'probes': probes,
                         'keys_exported': {t: sorted(got[t]) for t in V.TASKS}}
    return out


def main():
    t0 = time.process_time()
    recs = []
    if '--label-record' in sys.argv:
        recs = sys.argv[sys.argv.index('--label-record') + 1].split(',')
    recs = [str(V.RECORDS / 'V2_LABEL__DDR_VER_LABEL_V1.json')] + recs
    g1 = g1_map(recs)
    published = json.loads((V.ROOT / 'results' / 'pcrl_new_release_mechanisms_v1' / 'roles' / 'statistical_reviewer' /
                            'RULES.json').read_text())['references']['published_labels']
    from ddr import common as C
    unit = np.asarray(C.features('MODEL_SELECT')['unit'])
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'g1_sources': recs, 'errors': []}
    for name, fn in (('A1x', a1x), ('A3', lambda: a3(g1, published)), ('A4', lambda: a4(unit)), ('A5', lambda: a5(g1)),
                     ('A6', a6), ('A8', a8)):
        try:
            out[name] = fn()
        except Exception as e:
            out['errors'].append({'part': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V7_ADD_FREE__{V.unit_id() or "NOQUEUE_" + V.utc().replace(":", "")}.json', out)
    print(json.dumps({'record': str(p), 'errors': [e['part'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s']}))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())

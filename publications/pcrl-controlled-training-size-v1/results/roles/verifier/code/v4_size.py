"""Verifier stage V4: the size table, occupation-side reserve of every fit, all paired size contrasts (50->70, 70->100, 50->100), the
ordering summaries and labels, the comparison with the equally exposed TASK_REFIT / T_TREE fits (level, DiD, control-explains,
equal reserve), reproduction, encoding sensitivity, subset-to-subset variation and the label-free training records, recomputed with
the verifier's own code from the saved MODEL_SELECT release arrays, then compared with results/<campaign>/size/SIZE_RESULTS.json and
SIZE_TABLE.csv.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records;
MODEL_SELECT is adaptively used development data). Labels: MODEL_SELECT income, occupation and SEX, and MODEL_FIT occupation (class
counts of the registered subsets), read through tsz.common.load operation 'verify'.

    python3 -m tsz.jobqueue run --unit VER_SIZE_V1 --stage verification --rss-gib 3 -- python3 .../v4_size.py --tag V1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
SIZE = PRIV / 'runs' / 'SIZE'
CONTROL = PRIV / 'runs' / 'CONTROL'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
CONTRASTS = {'50->70': (50, 70), '70->100': (70, 100), '50->100': (50, 100)}
M5 = V.METRICS5


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sid_of(f, o):
    return 'S100' if f == 100 else f'S{f}_{o}'


def load_releases():
    """{(recipe, sid, v): MODEL_SELECT occupation array} with every file hash-checked against its own record; recipe in A1, A1_ADM,
    TR (TASK_REFIT occupation), TT (T_TREE occupation)."""
    rel, files, missing = {}, [], []
    for v in (0, 1):
        for sid in [f'S{f}_{o}' for f in (50, 70) for o in V.ORDERINGS] + ['S100']:
            d = SIZE / f'A1__{sid}__V{v}'
            if (d / 'FIT.json').is_file():
                rec = json.loads((d / 'FIT.json').read_text())
                if V.file_sha(d / 'RELEASE.npz') != rec['files_sha256']['RELEASE.npz']:
                    raise PermissionError(f'{d.name}: RELEASE.npz differs from FIT.json')
                rel[('A1', sid, v)] = npz(d / 'RELEASE.npz')['MODEL_SELECT']; files += [d / 'RELEASE.npz', d / 'FIT.json']
            else:
                missing.append(('A1', sid, v))
            dc = CONTROL / f'{sid}__V{v}'
            tj, sj = dc / 'TEACHERS' / 'TEACHERS.json', dc / 'STAGE_B' / 'STAGE_B.json'
            if tj.is_file():
                t = json.loads(tj.read_text())
                if V.file_sha(dc / 'TEACHERS' / 'TEACHER_OUTPUTS.npz') != t['files_sha256']['TEACHER_OUTPUTS.npz']:
                    raise PermissionError(f'{dc.name}: TEACHER_OUTPUTS.npz differs')
                rel[('TT', sid, v)] = npz(dc / 'TEACHERS' / 'TEACHER_OUTPUTS.npz')['T_TREE__MODEL_SELECT__occupation']
                files += [dc / 'TEACHERS' / 'TEACHER_OUTPUTS.npz', tj]
            else:
                missing.append(('TT', sid, v))
            if sj.is_file():
                s = json.loads(sj.read_text())
                if V.file_sha(dc / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz') != s['files_sha256']['STAGE_B_OUTPUTS_FINAL.npz']:
                    raise PermissionError(f'{dc.name}: STAGE_B_OUTPUTS_FINAL.npz differs')
                rel[('TR', sid, v)] = npz(dc / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz')['TASK_REFIT__MODEL_SELECT__occupation']
                files += [dc / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz', sj]
            else:
                missing.append(('TR', sid, v))
    for o in V.ORDERINGS:
        d = SIZE / f'A1__S50_{o}__V0__ADM'
        if (d / 'FIT.json').is_file():
            rec = json.loads((d / 'FIT.json').read_text())
            if V.file_sha(d / 'RELEASE.npz') != rec['files_sha256']['RELEASE.npz']:
                raise PermissionError(f'{d.name}: RELEASE.npz differs from FIT.json')
            rel[('A1_ADM', f'S50_{o}', 0)] = npz(d / 'RELEASE.npz')['MODEL_SELECT']; files += [d / 'RELEASE.npz', d / 'FIT.json']
        else:
            missing.append(('A1_ADM', f'S50_{o}', 0))
    return rel, files, missing


def keyname(k):
    return f'{k[0]}|{k[1]}|V{k[2]}'


def summary_block(xs_by_o, ses_by_o):
    """Improvement-orientation values per ordering -> own summary (vlib.summarize_orderings)."""
    os_ = sorted(xs_by_o)
    return V.summarize_orderings([xs_by_o[o] for o in os_], [ses_by_o[o] for o in os_])


def _pv(d, v):
    return d.get(str(v), d.get(v)) if isinstance(d, dict) else None


def compare(tag, mine_name):
    """Label-free: own V4 numbers (records/<mine_name>) vs results/<campaign>/size/SIZE_RESULTS.json and SIZE_TABLE.csv."""
    mine = json.loads((V.RECORDS / mine_name).read_text())['mine']
    prod_p = RES / 'size' / 'SIZE_RESULTS.json'; tab_p = RES / 'size' / 'SIZE_TABLE.csv'
    P = json.loads(prod_p.read_text())
    with open(tab_p, newline='') as f:
        table = {(r['recipe'], r['subset'], int(r['version'])): r for r in csv.DictReader(f)}
    tol, rtol = 1e-12, 1e-9
    out = {}
    # (1) per-fit table and occupation-side reserve
    fit_bad, fit_max = [], {'value': 0.0, 'own_se': 0.0, 'ratio': 0.0, 'abs_metric': 0.0}
    for key, f in mine['fits'].items():
        pr = P['occupation_reserve'].get(key)
        rc, sid, vv = key.split('|'); row = table.get((rc, sid, int(vv[1:])))
        if pr is None or row is None:
            fit_bad.append(f'{key}: missing in production'); continue
        for c, x in f['conditions'].items():
            q = pr['conditions'][c]
            for k in ('value', 'own_se'):
                fit_max[k] = max(fit_max[k], abs(x[k] - q[k]))
            fit_max['ratio'] = max(fit_max['ratio'], abs(x['ratio'] - q['ratio']))
            if x['met'] != q['met'] or x['point_pass'] != q['point_pass']:
                fit_bad.append(f'{key}|{c}: met/point differs')
        if f['occ_reserve_met'] != pr['met'] or f['binding'] != pr['binding'] or abs(f['min_occ_ratio'] - pr['min_ratio']) > rtol:
            fit_bad.append(f'{key}: status / binding / min ratio differs')
        if abs(f['income_min_ratio'] - pr['income']['min_income_ratio']) > rtol or f['income_met'] != pr['income']['income_reserve_met']:
            fit_bad.append(f'{key}: income block differs')
        for m in ('accuracy', 'nll', 'brier', 'men_brier', 'women_brier'):
            fit_max['abs_metric'] = max(fit_max['abs_metric'], abs(f[m] - float(row[m])))
    if fit_max['value'] > tol or fit_max['own_se'] > tol or fit_max['ratio'] > rtol or fit_max['abs_metric'] > tol:
        fit_bad.append(f'max differences {fit_max}')
    with open(tab_p, newline='') as f:
        rows_all = list(csv.DictReader(f))
    s100_rows = {}
    for r in rows_all:
        if r['subset'] == 'S100':
            s100_rows[(r['recipe'], r['version'])] = s100_rows.get((r['recipe'], r['version']), 0) + 1
    if any(n != 1 for n in s100_rows.values()):
        fit_bad.append(f'S100 listed more than once: {s100_rows}')
    inc_ratios = {}
    for key, f in mine['fits'].items():
        inc_ratios.setdefault(key.split('|')[2], set()).add(round(f['income_min_ratio'], 12))
    if any(len(s) != 1 for s in inc_ratios.values()):
        fit_bad.append(f'income conditions differ across fits of one version: {inc_ratios}')
    out['fits'] = {'n': len(mine['fits']), 'n_production': len(P['occupation_reserve']), 'max_abs_diff': fit_max, 'problems': fit_bad,
                   'same_key_set': sorted(mine['fits']) == sorted(P['occupation_reserve']),
                   's100_rows_per_recipe_version': {f'{k[0]}|V{k[1]}': n for k, n in s100_rows.items()},
                   'income_min_ratio_per_version': {k: sorted(s) for k, s in inc_ratios.items()}}
    # (2) size contrasts and labels
    sz_bad, sz_max = [], {'change': 0.0, 'se': 0.0, 'mean': 0.0, 'se_comb': 0.0}
    for rc in ('A1', 'TR', 'TT'):
        for c in CONTRASTS:
            for m in M5:
                mb, pb = mine['size'][rc][c][m], P['size_summary'][rc]['contrasts'][c][m]
                if mb['across'] != pb['across']:
                    sz_bad.append(f'{rc}|{c}|{m}: across {mb["across"]} vs {pb["across"]}')
                for v in (0, 1):
                    ms, ps = mb[str(v)]['summary'], _pv(pb['per_version'], v)
                    if ms['label'] != ps['label']:
                        sz_bad.append(f'{rc}|{c}|{m}|V{v}: label {ms["label"]} vs {ps["label"]}')
                    if ms['label'] != 'INCOMPLETE':
                        sz_max['mean'] = max(sz_max['mean'], abs(ms['mean'] - ps['mean']))
                        sz_max['se_comb'] = max(sz_max['se_comb'], abs(ms['se_comb'] - ps['se_comb']))
                        if ms['sign_reversals'] != ps['sign_reversals'] or bool(ms['flags']) != bool(ps['sign_reversal']):
                            sz_bad.append(f'{rc}|{c}|{m}|V{v}: sign reversal count / flag differs')
                    pc = _pv(P['size_changes'][rc][c][m], v)
                    for o, x in mb[str(v)]['change_values'].items():
                        sz_max['change'] = max(sz_max['change'], abs(x - pc['values'][o]))
                        sz_max['se'] = max(sz_max['se'], abs(mb[str(v)]['ses'][o] - pc['se'][o]))
        if mine['size_effect'][rc] != P['size_summary'][rc]['size_effect']:
            sz_bad.append(f'{rc}: size_effect {mine["size_effect"][rc]} vs {P["size_summary"][rc]["size_effect"]}')
    if any(x > tol for x in sz_max.values()):
        sz_bad.append(f'max differences {sz_max}')
    out['size'] = {'max_abs_diff': sz_max, 'problems': sz_bad, 'primary': {'mine': mine['size_effect']['A1'], 'production': P['size_summary']['A1']['size_effect']}}
    # (3) control comparison
    ct_bad, ct_max = [], {'mean': 0.0, 'se_comb': 0.0, 'value': 0.0}
    for ctl in ('TR', 'TT'):
        mc, pc = mine['control'][ctl], P['control'][ctl]
        for f in ('50', '70', '100'):
            for m in M5:
                a, b = mc['level'][f][m], pc['level'][f][m]
                if a['label'] != b['label'] or a['across'] != b['across']:
                    ct_bad.append(f'{ctl}|level|{f}|{m}: {a["label"]} vs {b["label"]}')
                for v in (0, 1):
                    sa, sb = a[str(v)]['summary'], _pv(b['per_version'], v)
                    if sa['label'] != sb['label']:
                        ct_bad.append(f'{ctl}|level|{f}|{m}|V{v}: {sa["label"]} vs {sb["label"]}')
                    elif sa['label'] != 'INCOMPLETE':
                        ct_max['mean'] = max(ct_max['mean'], abs(sa['mean'] - sb['mean'])); ct_max['se_comb'] = max(ct_max['se_comb'], abs(sa['se_comb'] - sb['se_comb']))
        for c in CONTRASTS:
            for m in M5:
                a, b = mc['did'][c][m], pc['did'][c][m]
                if a['across'] != b['across']:
                    ct_bad.append(f'{ctl}|did|{c}|{m}: {a["across"]} vs {b["across"]}')
                for v in (0, 1):
                    sa, sb = a[str(v)]['summary'], _pv(b['per_version'], v)
                    if sa['label'] != sb['label']:
                        ct_bad.append(f'{ctl}|did|{c}|{m}|V{v}: {sa["label"]} vs {sb["label"]}')
                    elif sa['label'] != 'INCOMPLETE':
                        ct_max['mean'] = max(ct_max['mean'], abs(sa['mean'] - sb['mean'])); ct_max['se_comb'] = max(ct_max['se_comb'], abs(sa['se_comb'] - sb['se_comb']))
        for m in M5:
            if mc['explains_50_100'][m] != pc['control_explains_50_100'][m]:
                ct_bad.append(f'{ctl}|explains|{m}: {mc["explains_50_100"][m]} vs {pc["control_explains_50_100"][m]}')
        for f in ('50', '70', '100'):
            lit = mc['also_meets_reserve'][f]['CONTROL_ALSO_MEETS_RESERVE']
            rm = mc['also_meets_reserve'][f]['recipe_met']
            prod_reading = bool(any(rm.values())) and bool(lit)
            pv_ = pc['control_also_meets_reserve'][f]
            if pv_ != prod_reading:
                ct_bad.append(f'{ctl}|also|{f}: production {pv_} vs own (production reading) {prod_reading}')
            if pv_ != lit:
                ct_bad.append(f'{ctl}|also|{f}: TEXT_READING_DIFFERS literal {lit} (A1 meets the reserve in no cell: vacuous) vs production {pv_}')
    if any(x > tol for x in ct_max.values()):
        ct_bad.append(f'max differences {ct_max}')
    out['control'] = {'max_abs_diff': ct_max, 'problems': ct_bad}
    # (4) pattern, explanation, reproduction, encoding
    pp = P['margin_pattern']
    lab_bad = []
    if mine['margin_pattern']['pattern'] != pp['pattern'] or sorted(mine['margin_pattern']['flags']) != sorted(pp.get('flags', [])):
        lab_bad.append(f'pattern {mine["margin_pattern"]} vs {pp}')
    if mine['size_explanation'] != P['interpretation']['size_explanation']:
        lab_bad.append(f'explanation {mine["size_explanation"]} vs {P["interpretation"]["size_explanation"]}')
    if mine['reproduction']['label'] != P['reproduction']['label']['label']:
        lab_bad.append(f'reproduction {mine["reproduction"]["label"]} vs {P["reproduction"]["label"]["label"]}')
    if ('MARGIN_REPRODUCED' if mine['reproduction']['MARGIN_REPRODUCED'] else 'MARGIN_NOT_REPRODUCED') != P['reproduction']['margin']:
        lab_bad.append('margin reproduced differs')
    enc_max = 0.0
    for m in M5:
        a, b = mine['encoding'][m], P['encoding']['summary'][m]
        if a['label'] != b['encoding_label']:
            lab_bad.append(f'encoding {m}: {a["label"]} vs {b["encoding_label"]}')
        if a['summary'].get('mean') is not None and b.get('mean') is not None:
            enc_max = max(enc_max, abs(a['summary']['mean'] - b['mean']), abs(a['summary']['se_comb'] - b['se_comb']))
    pw_max = 0.0
    for k, d in mine['pairwise'].items():
        q = P['subset_to_subset_pairwise'].get(k)
        if q is None:
            lab_bad.append(f'pairwise {k} missing'); continue
        pw_max = max([pw_max] + [abs(d[m]['value'] - q[m]['value']) for m in M5] + [abs(d[m]['se'] - q[m]['se']) for m in M5])
    vd_max = 0.0
    for k, d in mine['version_difference'].items():
        q = P['version_difference_same_rows'].get(k)
        if q is None:
            lab_bad.append(f'version difference {k} missing'); continue
        vd_max = max([vd_max] + [abs(d[m]['value'] - q[m]['value']) for m in M5])
    if max(enc_max, pw_max, vd_max) > tol:
        lab_bad.append(f'max differences encoding {enc_max} pairwise {pw_max} version {vd_max}')
    out['labels'] = {'problems': lab_bad, 'pattern': [mine['margin_pattern'], pp], 'explanation': [mine['size_explanation'], P['interpretation']['size_explanation']],
                     'reproduction': [mine['reproduction']['label'], P['reproduction']['label']], 'encoding_share_mine': {m: mine['encoding'][m]['share_of_50_100_improvement_v0'] for m in M5}}
    code = V.code_hashes(__file__, V.__file__)
    inputs = V.inputs_sha([prod_p, tab_p, V.RECORDS / mine_name])
    text_only = [p for p in ct_bad if 'TEXT_READING_DIFFERS' in p]
    hard_ct = [p for p in ct_bad if 'TEXT_READING_DIFFERS' not in p]
    checks = [
        {'id': f'V4.SIZE.FIT_TABLE_AND_RESERVE__{tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'every fit (A1_PO_16, encoding fits, TASK_REFIT, T_TREE; with H3 income): absolute occupation metrics, the 6 occupation clauses '
                 'and 2 guards (value, SE), reserve ratio / status / binding (k = 2, OHN1 floors) recomputed with own code vs SIZE_RESULTS.json '
                 'and SIZE_TABLE.csv (tolerance 1e-12 on values and SEs, 1e-9 on ratios)',
         'inputs': inputs, 'expected': 'agreement', 'observed': out['fits'], 'status': 'PASS' if (not fit_bad and out['fits']['same_key_set']) else 'FAIL'},
        {'id': f'V4.SIZE.CONTRASTS_AND_LABELS__{tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'paired size contrasts 50->70, 70->100, 50->100 per method, ordering, version and metric (loss changes and SEs), the ordering '
                 'summaries (mean, se_comb, sign reversals), per-version and across-version labels, and the size_effect labels incl. the '
                 'registered primary endpoint, own code vs SIZE_RESULTS.json',
         'inputs': inputs, 'expected': 'agreement', 'observed': out['size'], 'status': 'PASS' if not sz_bad else 'FAIL'},
        {'id': f'V4.SIZE.CONTROL_COMPARISON__{tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'level comparison A1_PO_16 vs TASK_REFIT / T_TREE at 50, 70, 100 (summaries and labels), DiD per contrast, control-explains '
                 'rule (50->100), CONTROL_ALSO_MEETS_RESERVE per size; own code vs SIZE_RESULTS.json; the RULES.md text reading of the '
                 'equal-reserve flag (vacuously true when A1_PO_16 meets the reserve in no cell) is reported beside the production reading',
         'inputs': inputs, 'expected': 'agreement', 'observed': out['control'],
         'status': 'FAIL' if hard_ct else ('ATTENTION' if text_only else 'PASS')},
        {'id': f'V4.SIZE.PATTERN_EXPLANATION_REPRO_ENCODING__{tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'margin pattern (with sub-flags), size explanation, reproduction label and MARGIN_REPRODUCED, encoding-sensitivity labels and '
                 'summaries, subset-to-subset pairwise differences and the version difference on shared rows; own code vs SIZE_RESULTS.json',
         'inputs': inputs, 'expected': 'agreement', 'observed': out['labels'], 'status': 'PASS' if not lab_bad else 'FAIL'}]
    for c in checks:
        c['code_sha256'] = code
    V.write_record(f'V4_SIZE_COMPARE__{tag}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))
    print(json.dumps({k: (v.get('problems') if isinstance(v, dict) else v) for k, v in out.items()}, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--compare', default=None)
    a = ap.parse_args()
    if a.compare:
        compare(a.tag, a.compare)
        return
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier V4: size table, reserve, size contrasts and labels recomputed with own code from the saved size-fit arrays')
    from ohn import common as OC
    from ohn import devrun as OD
    floors = json.loads((ROOT / V.OHN1_RULES).read_text())['reserve']['floor']['se']
    unit = np.asarray(S['unit']); sex = np.asarray(S['sex']); y = np.asarray(S['occupation'])
    labels = {'income': np.asarray(S['income']), 'occupation': y}
    G = len(np.unique(unit))
    refs, inc = {}, {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank row order differs')
        refs[v] = {r: {t: np.asarray(bank[r]['MODEL_SELECT'][t], dtype=np.float64) for t in ('income', 'occupation')} for r in ('U0', 'Ucal')}
        inc[v] = np.asarray(OD.anchor_income('H3', v, 'MODEL_SELECT'), dtype=np.float64)
    rel, files, missing = load_releases()
    L = {k: V.loss_vectors(y, p, sex) for k, p in rel.items()}
    men, women = sex == 0, sex == 1
    # 1. per-fit table and occupation-side reserve
    fits = {}
    for k, p in rel.items():
        v = k[2]
        conds = V.clauses_and_guards(labels, {'income': inc[v], 'occupation': p}, refs[v], sex, unit)
        ro = V.reserve(conds, floors[str(v)], G, names=V.OCC_SIDE)
        ri = V.reserve(conds, floors[str(v)], G, names=V.INCOME_SIDE)
        lv = L[k]
        fits[keyname(k)] = {'accuracy': float(1.0 - lv['acc_loss'].mean()), 'nll': float(lv['nll'].mean()), 'brier': float(lv['brier'].mean()),
                            'men_brier': float(lv['brier'][men].mean()), 'women_brier': float(lv['brier'][women].mean()),
                            'occ_point_pass': ro['point_pass'], 'occ_reserve_met': ro['met'], 'min_occ_ratio': ro['min_ratio'],
                            'binding': ro['binding'], 'failed': ro['failed'],
                            'conditions': {c: {f: ro['conditions'][c][f] for f in ('value', 'own_se', 'floor_se', 'se_used', 'margin', 'ratio', 'rho',
                                                                                  'met', 'point_pass')} for c in V.OCC_SIDE},
                            'income_min_ratio': ri['min_ratio'], 'income_met': ri['met'], 'income_binding': ri['binding']}
    met = {k: fits[keyname(k)]['occ_reserve_met'] for k in rel}
    # 2. size contrasts per method (loss orientation change; improvement summaries)
    def change(ka, kb):
        return V.paired_change(L[ka], L[kb], unit)[0]
    size = {}
    for rc in ('A1', 'TR', 'TT'):
        size[rc] = {}
        for c, (fa, fb) in CONTRASTS.items():
            size[rc][c] = {}
            for m in M5:
                blk = {}
                for v in (0, 1):
                    vals, ses = {}, {}
                    for o in V.ORDERINGS:
                        ka, kb = (rc, sid_of(fa, o), v), (rc, sid_of(fb, o), v)
                        if ka in L and kb in L:
                            d = change(ka, kb)[m]
                            vals[o], ses[o] = d['value'], d['se']
                    if len(vals) == 4:
                        s = summary_block({o: -x for o, x in vals.items()}, ses)
                    else:
                        s = {'label': 'INCOMPLETE'}
                    blk[str(v)] = {'change_values': vals, 'ses': ses, 'summary': s}
                blk['across'] = V.across_versions(blk['0']['summary']['label'], blk['1']['summary']['label'])
                size[rc][c][m] = blk
    size_effect = {rc: V.size_effect(size[rc]['50->100']['men_brier']['across']) for rc in size}
    # 3. control comparison
    control = {}
    for ctl in ('TR', 'TT'):
        lev = {}
        for f in (50, 70, 100):
            lev[f] = {}
            for m in M5:
                blk = {}
                for v in (0, 1):
                    vals, ses = {}, {}
                    for o in (V.ORDERINGS if f != 100 else ('ALL',)):
                        ka, kc = ('A1', sid_of(f, o), v), (ctl, sid_of(f, o), v)
                        if ka in L and kc in L:
                            d = change(ka, kc)[m]                     # loss(control) - loss(A1) = g (positive: recipe better)
                            vals[o], ses[o] = d['value'], d['se']
                    need = 4 if f != 100 else 1
                    blk[str(v)] = {'g_values': vals, 'ses': ses, 'summary': summary_block(vals, ses) if len(vals) == need else {'label': 'INCOMPLETE'}}
                blk['across'] = V.across_versions(blk['0']['summary']['label'], blk['1']['summary']['label'])
                blk['label'] = V.control_level_label(blk['across'])
                lev[f][m] = blk
        did = {}
        for c, (fa, fb) in CONTRASTS.items():
            did[c] = {}
            per_m = {m: {0: ({}, {}), 1: ({}, {})} for m in M5}
            for v in (0, 1):
                for o in V.ORDERINGS:
                    ks = [('A1', sid_of(fa, o), v), ('A1', sid_of(fb, o), v), (ctl, sid_of(fa, o), v), (ctl, sid_of(fb, o), v)]
                    if not all(k in L for k in ks):
                        continue
                    a1a, a1b, ca, cb = (L[k] for k in ks)
                    dd = {mm: (a1a[mm] - a1b[mm]) - (ca[mm] - cb[mm]) for mm in ('acc_loss', 'nll', 'brier')}   # g_A1 - g_ctl per person
                    dd['men_brier'] = dd['brier']; dd['women_brier'] = dd['brier']
                    st = V.diff_stats(dd, men, women, unit)
                    for m in M5:
                        per_m[m][v][0][o] = st[m]['value']; per_m[m][v][1][o] = st[m]['se']
            for m in M5:
                blk = {}
                for v in (0, 1):
                    vals, ses = per_m[m][v]
                    blk[str(v)] = {'did_values': vals, 'ses': ses, 'summary': summary_block(vals, ses) if len(vals) == 4 else {'label': 'INCOMPLETE'}}
                blk['across'] = V.across_versions(blk['0']['summary']['label'], blk['1']['summary']['label'])
                did[c][m] = blk
        explains = {m: V.control_explains(size['A1']['50->100'][m]['across'], size[ctl]['50->100'][m]['across'], did['50->100'][m]['across'])
                    for m in M5}
        also = {}
        for f in (50, 70, 100):
            cells = [(o, v) for o in (V.ORDERINGS if f != 100 else ('ALL',)) for v in (0, 1)]
            rm = {(o, v): met.get(('A1', sid_of(f, o), v)) for o, v in cells}
            cm = {(o, v): met.get((ctl, sid_of(f, o), v)) for o, v in cells}
            also[f] = {'CONTROL_ALSO_MEETS_RESERVE': V.control_also_meets(rm, cm) if None not in rm.values() and None not in cm.values() else None,
                       'recipe_met': {f'{o}|{v}': x for (o, v), x in rm.items()}, 'control_met': {f'{o}|{v}': x for (o, v), x in cm.items()}}
        control[ctl] = {'level': lev, 'did': did, 'explains_50_100': explains, 'also_meets_reserve': also}
    # 4. margin pattern, explanation
    m100 = {v: met.get(('A1', 'S100', v)) for v in (0, 1)}
    m50 = {(o, v): met.get(('A1', f'S50_{o}', v)) for o in V.ORDERINGS for v in (0, 1)}
    m70 = {(o, v): met.get(('A1', f'S70_{o}', v)) for o in V.ORDERINGS for v in (0, 1)}
    pattern = V.margin_pattern(m100, m50, m70)
    explanation, ex_flags = V.size_explanation(pattern['pattern'], size_effect['A1'])
    # 5. reproduction
    repro = {}
    for v in (0, 1):
        d = SIZE / f'A1__S100__V{v}'
        if not (d / 'FIT.json').is_file():
            continue
        mine_rel = npz(d / 'RELEASE.npz'); rec = json.loads((d / 'FIT.json').read_text())
        src = {k.split('__')[0]: x for k, x in npz(OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.npz').items()}
        src_rec = json.loads((OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.json').read_text())
        roles = {r: {'bitwise': bool(mine_rel[r].shape == src[r].shape and np.array_equal(mine_rel[r], src[r])),
                     'max_abs': float(np.max(np.abs(mine_rel[r] - src[r]))), 'decisions': bool(np.array_equal(mine_rel[r].argmax(1), src[r].argmax(1)))}
                 for r in src}
        repro[v] = {'bitwise': all(x['bitwise'] for x in roles.values()), 'max_abs': max(x['max_abs'] for x in roles.values()),
                    'decisions_identical': all(x['decisions'] for x in roles.values()), 'tstar_equal': rec['T_star'] == src_rec.get('T_star'),
                    'roles': roles, 'source_roles': sorted(src)}
    repro_label = V.reproduction_label(repro)
    margin_reproduced = all(m100.get(v) is True for v in (0, 1))
    # 6. encoding sensitivity (v = 0): E_o = loss(FIT-only) - loss(admitted)
    enc_vals = {m: ({}, {}) for m in M5}
    for o in V.ORDERINGS:
        ka, kb = ('A1_ADM', f'S50_{o}', 0), ('A1', f'S50_{o}', 0)
        if ka in L and kb in L:
            d = change(ka, kb)
            for m in M5:
                enc_vals[m][0][o] = d[m]['value']; enc_vals[m][1][o] = d[m]['se']
    enc = {}
    for m in M5:
        vals, ses = enc_vals[m]
        s = summary_block(vals, ses) if len(vals) == 4 else {'label': 'INCOMPLETE'}
        g100 = size['A1']['50->100'][m]['0']['summary']
        share = (s['mean'] / g100['mean']) if (s.get('mean') is not None and g100.get('mean') is not None and s['mean'] > 0 and g100['mean'] > 0) else None
        enc[m] = {'E_values': vals, 'ses': ses, 'summary': s, 'label': V.encoding_label(s['label']), 'share_of_50_100_improvement_v0': share}
    enc_reserve = {o: met.get(('A1_ADM', f'S50_{o}', 0)) for o in V.ORDERINGS}
    # 7. subset-to-subset variation (A1, TR, TT): spread of absolute metrics; pairwise paired differences (A1)
    spread = {}
    for rc in ('A1', 'TR', 'TT'):
        for f in (50, 70):
            for v in (0, 1):
                ks = [(rc, f'S{f}_{o}', v) for o in V.ORDERINGS]
                if not all(k in L for k in ks):
                    continue
                spread[f'{rc}|S{f}|V{v}'] = {}
                for m in M5:
                    xs = [float((L[k][m] if m in ('acc_loss', 'nll', 'brier') else L[k]['brier'][men if m == 'men_brier' else women]).mean()) for k in ks]
                    xb = sum(xs) / 4
                    spread[f'{rc}|S{f}|V{v}'][m] = {'sd_ord': math.sqrt(sum((x - xb) ** 2 for x in xs) / 3), 'range': [min(xs), max(xs)]}
    pairwise = {}
    for f in (50, 70):
        for v in (0, 1):
            for i, oa in enumerate(V.ORDERINGS):
                for ob in V.ORDERINGS[i + 1:]:
                    ka, kb = ('A1', f'S{f}_{oa}', v), ('A1', f'S{f}_{ob}', v)
                    if ka in L and kb in L:
                        pairwise[f'S{f}|V{v}|{ob}-{oa}'] = change(ka, kb)
    vdiff = {}
    for sid in [f'S{f}_{o}' for f in (50, 70) for o in V.ORDERINGS] + ['S100']:
        ka, kb = ('A1', sid, 0), ('A1', sid, 1)
        if ka in L and kb in L:
            vdiff[sid] = change(ka, kb)
    mine = {'fits': fits, 'size': size, 'size_effect': size_effect, 'control': control, 'margin_pattern': pattern,
            'size_explanation': explanation, 'size_explanation_flags': ex_flags, 'reproduction': {'label': repro_label, 'per_version': repro,
                                                                                               'MARGIN_REPRODUCED': margin_reproduced},
            'encoding': enc, 'encoding_fits_reserve_met': enc_reserve, 'spread': spread, 'pairwise': pairwise, 'version_difference': vdiff,
            'missing': [keyname(k) for k in missing], 'n_fits': len(rel), 'G': G}
    out_rec = V.write_record(f'V4_SIZE_MINE__{a.tag}.json', {'mine': mine, 'inputs': V.inputs_sha(files), 'utc': V.utc()})
    print(json.dumps({'n_fits': len(rel), 'missing': mine['missing'][:10], 'size_effect': size_effect, 'pattern': pattern,
                      'explanation': explanation, 'repro': repro_label, 'record': str(out_rec.name)}, default=str))


if __name__ == '__main__':
    main()

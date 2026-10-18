"""Independent verifier, checks 3, 4 and 6 (MODEL_SELECT development record; own code).

Public-use ACS 2019 Florida person records. Service: income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR,
POBP; readers estimate the recorded SEX column from released probability vectors.

3. Utility clauses (12 per seed) and the two within-SEX occupation guards with household-clustered SEs, recomputed from the
   releases and MODEL_SELECT labels (logged read) for every configuration with a release and 4 references; compared with
   EVAL_MS.json / EVAL_MS_REFERENCES.json (points 1e-12, SEs 1e-9).
4. Reserve status re-derived from the rule text (UTILITY_RESERVE_RULE.md, RESERVE_RULE.json floors) with own code, from the
   own recomputed values and SEs; compared with each EVAL_MS reserve screen and NOMINATION.json b_reserve.
6. Nomination labels, flags, score, near-tie set and the choice re-derived from the EVAL_MS records with own code
   (NOMINATION_RULE.md); compared with NOMINATION.json.
Not imported: ihs.evaluate, ihs.reserve, ihs.nominate.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ihs import common as C                       # admitted loaders only
from ihs.devrun import load_release               # allowed release loader (hash-checked)

RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
RUNS = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'runs'
OUT = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'verifier'
TASKS = ('income', 'occupation'); K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
CELLS = ('A0', 'A1', 'B0', 'B1'); READERS = ('OUT_LR', 'OUT_HGB', 'LOG_MLP', 'ET25', 'HGB31', 'CATI1', 'LOOKUP')


def wire(a):
    """Documented wire definition: float64 probability bytes followed by the int64 argmax decision bytes (V2; V1 omitted the
    decision bytes, so its wire comparison was a definition mismatch, not a data mismatch)."""
    p = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    h = hashlib.sha256(p.tobytes()); h.update(np.ascontiguousarray(np.argmax(p, axis=1).astype(np.int64)).tobytes())
    return h.hexdigest()


def vecs(y, p, k):
    p = np.asarray(p, dtype=np.float64)
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1.0
    return {'accuracy': (np.argmax(p, 1) == y).astype(float), 'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - oh) ** 2).sum(1)}


def cl_se(d, inv, G):
    n = len(d); s = np.bincount(inv, weights=d - d.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1.0) * (s ** 2).sum()) / n)


def sub_se(d, mask, inv, G):
    n = len(d); p = mask.mean(); m = d[mask].mean()
    u = np.where(mask, (d - m) / p, 0.0)
    s = np.bincount(inv, weights=u, minlength=G)
    return float(np.sqrt(G / (G - 1.0) * (s ** 2).sum()) / n)


def utility(lab, sex, inv, G, outs, refs):
    own = {t: vecs(lab[t], outs[t], K[t]) for t in TASKS}
    cl = {}
    for t in TASKS:
        for ref in ('U0', 'Ucal'):
            r = vecs(lab[t], refs[ref][t], K[t]); o = own[t]
            val = {'accuracy_loss': r['accuracy'].mean() - o['accuracy'].mean(), 'nll_excess': o['nll'].mean() - r['nll'].mean(),
                   'brier_excess': o['brier'].mean() - r['brier'].mean()}
            dv = {'accuracy_loss': r['accuracy'] - o['accuracy'], 'nll_excess': o['nll'] - r['nll'], 'brier_excess': o['brier'] - r['brier']}
            for m in METRICS:
                cl[f'{t}|{ref}|{m}'] = {'value': float(val[m]), 'se': cl_se(dv[m], inv, G), 'cap': CAPS[m]}
    u0 = vecs(lab['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    gd = {}
    for g in (0, 1):
        mask = sex == g
        gd[f'guard|sex{g}'] = {'value': float(d[mask].mean()), 'se': sub_se(d, mask, inv, G), 'cap': 0.005}
    return cl, gd


def reserve(cl, gd, seed, rule):
    """Own reading of UTILITY_RESERVE_RULE.md section 1 (k = 2, se_used = max(own, H_LJ floor), inclusive)."""
    k = rule['multiplier_k']; floor = rule['floor']['se'][str(seed)]
    occ = rule['occupation']; inh_vals = occ['inherited_values'][str(seed)]; inh_marg = occ['inherited_margins'][str(seed)]
    inc = {}
    for c in rule['income_conditions']:
        v, se, cap = cl[c]['value'], cl[c]['se'], cl[c]['cap']
        su = max(se, floor[c]); m = cap - v
        ok = bool(np.isfinite(v) and np.isfinite(se) and m >= k * su)
        inc[c] = {'margin': m, 'se_used': su, 'ratio': m / su, 'met': ok}
    allv = {**cl, **gd}
    inherited = all(abs(allv[c]['value'] - inh_vals[c]) <= occ['identity_tolerance'] for c in occ['conditions'])
    occ_failed = []
    if not inherited:
        for c in occ['conditions']:
            m = allv[c]['cap'] - allv[c]['value']; su = max(allv[c]['se'], floor[c])
            if not (m >= k * su or m >= inh_marg[c]):
                occ_failed.append(c)
    failed = [c for c, r in inc.items() if not r['met']] + occ_failed
    return {'status': 'RESERVE_MET' if not failed else 'RESERVE_NOT_MET', 'failed': failed, 'inherited': inherited,
            'min_income_ratio': min(r['ratio'] for r in inc.values()), 'income': inc}


def two_seed(st):
    if len(st) == 2:
        n = sum(s == 'RESERVE_MET' for s in st)
        return 'RESERVE_MET' if n == 2 else 'RESERVE_ONE_SEED_ONLY' if n == 1 else 'RESERVE_NOT_MET'
    if len(st) == 1:
        return 'RESERVE_PENDING_SECOND_SEED' if st[0] == 'RESERVE_MET' else 'RESERVE_NOT_MET'
    return 'NONE'


def pair_winner(cell):
    best, lab = -1.0, None
    for view in ('pair', 'income', 'occupation'):
        for r in READERS:
            v = cell[view][r]
            if v > best:
                best, lab = v, f'{view}:{r}'
    return lab, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nom = json.loads((RES / 'NOMINATION.json').read_text())
    rule = json.loads((RES / 'roles' / 'statistical_reviewer' / 'RESERVE_RULE.json').read_text())
    refs_eval = {s: json.loads((RUNS / 'REF' / f'S{s}' / 'EVAL_MS_REFERENCES.json').read_text()) for s in (0, 1)}
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'),
                    purpose='independent verification: MODEL_SELECT utility clauses, guards, SEs and reserve (IHS verifier checks 3, 4, 6)')
    lab = {t: S[t] for t in TASKS}; sex = S['sex']; _, inv = np.unique(S['unit'], return_inverse=True); G = int(inv.max() + 1)
    out = {'G': G, 'n': int(len(sex)), 'check3': {}, 'check4': {}, 'check6': {}}
    tol = {'point': 0.0, 'se': 0.0}
    # ---------------- references (own outputs from the admitted banks)
    ref_outs = {}
    for s in (0, 1):
        fb, ids = C.frozen_bank(s, roles=('MODEL_SELECT',))
        assert np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id'])
        ref_outs[s] = {n: {t: fb[n]['MODEL_SELECT'][t] for t in TASKS} for n in ('U0', 'Ucal', 'H_LJ', 'NEURAL_RAW')}
        for n, cfg in (('B2TWIN', 'B2CTL_UNCOUPLED_C6A'), ('RETIRED_B3CTL', 'B3CTL_UNCOUPLED_MIX50_C6A')):
            ref_outs[s][n] = C.amd_release(cfg, s, roles=('MODEL_SELECT',))['MODEL_SELECT']
    own_ut = {}
    def compare(name, s, cl, gd, rec_ut):
        st = {c['task'] + '|' + c['reference'] + '|' + c['metric']: c for c in rec_ut['clauses']}
        sg = {'guard|' + g['group']: g for g in rec_ut['guards']}
        dp = max(max(abs(cl[k]['value'] - st[k]['value']) for k in cl), max(abs(gd[k]['value'] - sg[k]['value']) for k in gd))
        ds = max(max(abs(cl[k]['se'] - st[k]['se']) for k in cl), max(abs(gd[k]['se'] - sg[k]['se']) for k in gd))
        pp = all((cl[k]['value'] <= cl[k]['cap']) == st[k]['point_pass'] for k in cl) and all((gd[k]['value'] <= 0.005) == sg[k]['point_pass'] for k in gd)
        tol['point'] = max(tol['point'], dp); tol['se'] = max(tol['se'], ds)
        out['check3'][f'{name}|s{s}'] = {'max_point_diff': dp, 'max_se_diff': ds, 'point_pass_flags_match': pp,
                                         'pass_point_1e-12': dp <= 1e-12, 'pass_se_1e-9': ds <= 1e-9}
    for s in (0, 1):
        refs = {r: ref_outs[s][r] for r in ('U0', 'Ucal')}
        for n in ('H_LJ', 'NEURAL_RAW', 'B2TWIN', 'RETIRED_B3CTL'):
            cl, gd = utility(lab, sex, inv, G, ref_outs[s][n], refs)
            own_ut[(f'REF/{n}', s)] = (cl, gd)
            compare(f'REF/{n}', s, cl, gd, refs_eval[s][n]['utility'])
            out['check3'][f'REF/{n}|s{s}']['wire_match'] = all(wire(ref_outs[s][n][t]) == refs_eval[s][n]['wire_sha256'][t] for t in TASKS)
    # ---------------- configurations with releases
    evals, recs = {}, {}
    for c in nom['configurations']:
        fam, cid = c['family'], c['config_id']
        for s in (0, 1):
            p = RUNS / fam / cid / f'S{s}' / 'EVAL_MS.json'
            if not p.is_file():
                continue
            evals[(fam, cid, s)] = json.loads(p.read_text())
            outs, rec = load_release(fam, cid, s, roles=('MODEL_SELECT',)); recs[(fam, cid, s)] = rec
            refs = {r: ref_outs[s][r] for r in ('U0', 'Ucal')}
            cl, gd = utility(lab, sex, inv, G, outs['MODEL_SELECT'], refs)
            own_ut[(f'{fam}/{cid}', s)] = (cl, gd)
            compare(f'{fam}/{cid}', s, cl, gd, evals[(fam, cid, s)]['utility'])
            out['check3'][f'{fam}/{cid}|s{s}']['wire_match'] = all(wire(outs['MODEL_SELECT'][t]) == evals[(fam, cid, s)]['wire_sha256'][t] for t in TASKS)
    out['check3_max'] = tol
    # ---------------- check 4: reserve from own values / SEs
    for key, (cl, gd) in own_ut.items():
        name, s = key
        r = reserve(cl, gd, s, rule)
        if name.startswith('REF/'):
            stored = rule['reference_screen'][name[4:]]
            out['check4'][f'{name}|s{s}'] = {'own': {k: r[k] for k in ('status', 'failed', 'min_income_ratio', 'inherited')}, 'stored_reference_screen': stored.get(str(s), stored) if isinstance(stored, dict) else stored}
            continue
        fam, cid = name.split('/')
        st = evals[(fam, cid, s)]['reserve']
        out['check4'][f'{name}|s{s}'] = {'own_status': r['status'], 'stored_status': st['status'], 'status_match': r['status'] == st['status'],
                                         'own_failed': r['failed'], 'stored_failed': st['failed'], 'failed_match': sorted(r['failed']) == sorted(st['failed']),
                                         'own_min_ratio': r['min_income_ratio'], 'stored_min_ratio': st['min_income_ratio'],
                                         'ratio_diff': abs(r['min_income_ratio'] - st['min_income_ratio']),
                                         'own_inherited': r['inherited'], 'stored_occupation_side': st['occupation_side']}
    # two-seed status vs NOMINATION.json
    for c in nom['configurations']:
        name = f"{c['family']}/{c['config_id']}"
        st = [reserve(*own_ut[(name, s)], s, rule)['status'] for s in (0, 1) if (name, s) in own_ut]
        mins = [reserve(*own_ut[(name, s)], s, rule)['min_income_ratio'] for s in (0, 1) if (name, s) in own_ut]
        b = c.get('b_reserve') or {}
        out['check4'][f'{name}|two_seed'] = {'own': two_seed(st), 'stored': b.get('status'), 'match': two_seed(st) == b.get('status') or (b.get('status') is None),
                                             'own_min_ratio_two_seed': min(mins) if mins else None, 'stored_min_ratio': c.get('min_income_ratio')}
    # ---------------- check 6: nomination from the EVAL_MS records (stored values)
    ledger = list(csv.DictReader(open(RES / 'EXPERIMENT_LEDGER.csv')))
    first_entry = {}
    for i, row in enumerate(ledger):
        k = (row['family'], row['config_id'], row['seed'])
        if k not in first_entry:
            first_entry[k] = i + 1
    ref_wires = {s: {n: v['wire_sha256'] for n, v in refs_eval[s].items()} for s in (0, 1)}
    rows = {}
    for c in nom['configurations']:
        fam, cid = c['family'], c['config_id']; name = f'{fam}/{cid}'
        seeds = [s for s in (0, 1) if (fam, cid, s) in evals]
        r = {'stored_label': c['label']}
        if len(seeds) < 2:
            r['label'] = 'NOT_CONSIDERED_ONE_SEED_ONLY'; rows[name] = r; continue
        ev = {s: evals[(fam, cid, s)] for s in (0, 1)}
        alias = [n for n in ref_wires[0] if all(ev[s]['wire_sha256'][t] == ref_wires[s][n][t] for s in (0, 1) for t in TASKS)]
        a_fail = [f"s{s}|{x['task']}|{x['reference']}|{x['metric']}" for s in (0, 1) for x in ev[s]['utility']['clauses'] if not x['value'] <= x['cap']]
        a_fail += [f"s{s}|guard|{g['group']}" for s in (0, 1) for g in ev[s]['utility']['guards'] if not g['value'] <= 0.005]
        A = not a_fail
        # reserve from the stored clause values / SEs with own rule code
        st_res = []
        mins = []
        for s in (0, 1):
            cl = {f"{x['task']}|{x['reference']}|{x['metric']}": {'value': x['value'], 'se': x['se'], 'cap': x['cap']} for x in ev[s]['utility']['clauses']}
            gd = {f"guard|{g['group']}": {'value': g['value'], 'se': g['se'], 'cap': 0.005} for g in ev[s]['utility']['guards']}
            rr_ = reserve(cl, gd, s, rule); st_res.append(rr_['status']); mins.append(rr_['min_income_ratio'])
        Bst = two_seed(st_res); Bok = Bst == 'RESERVE_MET'; mir = min(mins)
        cells = {s: ev[s]['proxy']['cells'] for s in (0, 1)}
        refc = {(n, s): refs_eval[s][n]['proxy']['cells'] for n in ('NEURAL_RAW', 'H_LJ', 'B2TWIN') for s in (0, 1)}
        ben, recd = {}, {}
        for n in ('NEURAL_RAW', 'H_LJ'):
            for s in (0, 1):
                ben[(n, s)] = [refc[(n, s)][k]['pair']['max_incl_single'] - cells[s][k]['pair']['max_incl_single'] for k in CELLS]
                for v in TASKS:
                    recd[(n, s, v)] = [cells[s][k][v]['view_max'] - refc[(n, s)][k][v]['view_max'] for k in CELLS]
        c1 = all(x > 0 for b in ben.values() for x in b)
        c2 = all(x <= 0.010 for d in recd.values() for x in d)
        c3cells = {s: [refc[('B2TWIN', s)][k]['pair']['max_incl_single'] - cells[s][k]['pair']['max_incl_single'] for k in CELLS] for s in (0, 1)}
        c3 = all(x > 0 for v in c3cells.values() for x in v)
        score = float(np.mean([cells[s][k]['pair']['max_incl_single'] for s in (0, 1) for k in CELLS]))
        maxrec = max(x for d in recd.values() for x in d)
        flags = []
        if any(np.mean(b) < 0.005 for b in ben.values()):
            flags.append('PAIR_NEAR_TIE')
        wv = {s: [pair_winner(cells[s][k])[0] for k in CELLS] for s in (0, 1)}
        rw = {(n, s): [pair_winner(refc[(n, s)][k])[0] for k in CELLS] for n in ('NEURAL_RAW', 'H_LJ') for s in (0, 1)}
        unstable = (any((max(b) - min(b)) > np.mean(b) for b in ben.values()) or any(len(set(v)) > 1 for v in wv.values())
                    or any(len(set(v)) > 1 for v in rw.values()))
        if unstable:
            flags.append('READER_UNSTABLE')
        if maxrec > 0:
            flags.append(f'RECIPIENT_EDGE (max cell difference {maxrec:+.4f})')
        if alias:
            lab_ = f'ALIAS_OF_{alias[0]}'
        elif not A:
            lab_ = 'POINT_INELIGIBLE'
        elif not Bok:
            lab_ = 'RESERVE_NOT_MET'
        elif not c1:
            lab_ = 'UTILITY_HEADROOM_ONLY'
        elif not c2:
            lab_ = 'RECIPIENT_SCREEN_FAILED'
        elif not c3:
            lab_ = 'NO_GAIN_OVER_B2TWIN'
        else:
            lab_ = 'QUALIFYING'
        nfits = sum(int(recs[(fam, cid, s)].get('n_model_fits', 0) or 0) for s in (0, 1))
        r.update({'label': lab_, 'a': A, 'a_failed': a_fail, 'b': Bst, 'min_income_ratio': mir, 'c1': c1, 'c2': c2, 'c3': c3, 'score': score,
                  'flags': flags, 'stored_flags': c.get('flags'), 'stored_score': c.get('score_two_seed_mean_pair_proxy'),
                  'stored_min_income_ratio': c.get('min_income_ratio'), 'max_recipient_cell_difference': maxrec,
                  'stored_max_recipient_cell_difference': c.get('max_recipient_cell_difference'),
                  'n_model_fits_two_seeds': nfits, 'stored_n_model_fits_two_seeds': c.get('n_model_fits_two_seeds'),
                  'seed0_entry': first_entry.get((fam, cid, '0')), 'stored_seed0_entry': c.get('seed0_entry'),
                  'winners_match': wv == {int(k): [v[x] for x in CELLS] for k, v in c.get('pair_winner_per_cell', {}).items()} if c.get('pair_winner_per_cell') else None})
        rows[name] = r
    q = {n: r for n, r in rows.items() if r['label'] == 'QUALIFYING'}
    choice = None
    if q:
        best = min(r['score'] for r in q.values())
        near = {n: r for n, r in q.items() if r['score'] <= best + 0.005}
        choice = sorted(near, key=lambda n: (-near[n]['min_income_ratio'], near[n]['n_model_fits_two_seeds'], near[n]['seed0_entry']))[0]
        for n in q:
            rows[n]['label'] = 'NOMINATED_FINALIST' if n == choice else 'QUALIFYING_NOT_CHOSEN'
        out['check6_near_tie_set'] = sorted(near); out['check6_best_score'] = best
    out['check6_choice'] = choice; out['check6_stored_choice'] = nom['nominated']
    for n, r in rows.items():
        r['label_match'] = r['label'] == r['stored_label']
        if 'score' in r and r.get('stored_score') is not None:
            r['score_diff'] = abs(r['score'] - r['stored_score']); r['ratio_diff'] = abs(r['min_income_ratio'] - r['stored_min_income_ratio'])
            r['flags_match'] = sorted(r['flags']) == sorted(r['stored_flags'] or [])
    out['check6'] = rows
    (OUT / 'MS_CHECKS.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    summ = {'check3_max': tol, 'check3_all_pass': all(v['pass_point_1e-12'] and v['pass_se_1e-9'] and v['point_pass_flags_match'] and v.get('wire_match', True) for v in out['check3'].values()),
            'check3_n': len(out['check3']),
            'check4_seed_mismatch': [k for k, v in out['check4'].items() if 'status_match' in v and not (v['status_match'] and v['failed_match'] and v['ratio_diff'] <= 1e-9)],
            'check4_two_seed_mismatch': [k for k, v in out['check4'].items() if k.endswith('two_seed') and not v['match']],
            'check6_label_mismatch': [n for n, r in rows.items() if not r['label_match']],
            'check6_flag_mismatch': [n for n, r in rows.items() if r.get('flags_match') is False],
            'check6_other_mismatch': [n for n, r in rows.items() if r.get('score_diff', 0) > 1e-12 or r.get('ratio_diff', 0) > 1e-9
                                      or (r.get('stored_n_model_fits_two_seeds') not in (None, '') and r.get('n_model_fits_two_seeds') != r.get('stored_n_model_fits_two_seeds'))
                                      or (r.get('stored_seed0_entry') is not None and r.get('seed0_entry') != r.get('stored_seed0_entry'))
                                      or r.get('winners_match') is False],
            'check6_choice': choice, 'stored_choice': nom['nominated'], 'near_tie_set': out.get('check6_near_tie_set')}
    print(json.dumps(summ, indent=1, default=float))


if __name__ == '__main__':
    main()

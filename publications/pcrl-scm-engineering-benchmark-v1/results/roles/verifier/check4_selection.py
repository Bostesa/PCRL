#!/usr/bin/env python
"""Independent verifier, Check 4: MODEL_SELECT utility clauses, cross-fitted privacy proxy and nomination (PROTOCOL.md section 7).

Own code. It does not import scm_bench, task_conditioned_representation or acs_replication. The registered definitions it
re-implements (read from the frozen code, then written here independently):
- clauses: unweighted person means on MODEL_SELECT; accuracy loss = ref acc - arm acc (argmax, first index on ties); NLL
  excess = arm NLL - ref NLL with p_y floored at 1e-12; summed Brier excess = arm - ref; inclusive caps 0.010/0.010/0.005;
  normalised = value / cap.
- proxy: household halves A/B from np.random.default_rng(int(sha256('SCM1proxysplit')[:8 hex], 16)).permutation over the
  sorted unique household ids, A = first floor(G/2); pair wire 'out' features (income p, income argmax, occupation p,
  occupation argmax); a finite scaler (mean, population std, std 0 -> 1); OUT_LR = LogisticRegression(C=1, lbfgs, 500 iter);
  OUT_HGB = HistGradientBoostingClassifier(15 leaves, 200 iter, lr 0.05, no early stop, random_state 2026101205 + s);
  LOOKUP keyed on exact float64 bytes of the (income, occupation) vectors, score (n_female + 2 pi)/(n + 2), unseen -> pi;
  orientation +1; AUC = Mann-Whitney with ties counted one half; reader proxy = mean over the two directions (A->B, B->A);
  arm proxy = max over the three readers.
Writes check4_selection.json (aggregates only; no person-level arrays).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIV = ROOT / '.private' / 'pcrl_scm_engineering_benchmark_v1'
RES = ROOT / 'results' / 'pcrl_scm_engineering_benchmark_v1'
TASKS = ('income', 'occupation'); KC = {'income': 2, 'occupation': 5}
CAPS = (('accuracy_loss', 0.010), ('nll_excess', 0.010), ('brier_excess', 0.005))
BETAS = ('0.5', '1.0', '1.5'); GAMMAS = ('0.5', '1.0', '2.0'); NSHUF = 5
SEED_BASE = 2026101205


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# ---------------------------------------------------------------- utility clauses
def task_metrics(y, p):
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64); n, k = p.shape
    assert y.shape == (n,) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-9, rtol=0)
    oh = np.zeros_like(p); oh[np.arange(n), y] = 1.0
    return {'acc': float(np.mean(np.argmax(p, axis=1) == y)),
            'nll': float(np.mean(-np.log(np.maximum(p[np.arange(n), y], 1e-12)))),
            'brier': float(np.mean(np.sum((p - oh) ** 2, axis=1)))}


def my_clauses(labels, arm, refs):
    out = {}
    for t in TASKS:
        own = task_metrics(labels[t], arm[t])
        for r in ('U0', 'Ucal'):
            ref = task_metrics(labels[t], refs[r][t])
            v = {'accuracy_loss': ref['acc'] - own['acc'], 'nll_excess': own['nll'] - ref['nll'], 'brier_excess': own['brier'] - ref['brier']}
            for m, cap in CAPS:
                out[f'{t}|{r}|{m}'] = {'value': v[m], 'cap': cap, 'normalized': v[m] / cap, 'pass': bool(v[m] <= cap)}
    return out


def occ_brier_vec(y, p):
    p = np.asarray(p, dtype=np.float64); n = len(y); oh = np.zeros_like(p); oh[np.arange(n), y] = 1.0
    return np.sum((p - oh) ** 2, axis=1)


# ---------------------------------------------------------------- proxy readers
def seed_for8(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:8], 16)


def my_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum()); assert n1 > 0 and n0 > 0 and n1 + n0 == len(y)
    r = rankdata(s, method='average')
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def out_features(inc, occ):
    inc = np.asarray(inc, dtype=np.float64); occ = np.asarray(occ, dtype=np.float64)
    return np.column_stack((inc, inc.argmax(1).astype(np.float64), occ, occ.argmax(1).astype(np.float64)))


def scale_fit(x):
    mu = x.mean(0); sd = x.std(0); return mu, np.where(sd > 0, sd, 1.0)


def fit_score(name, xtr, ytr, xte, seed):
    mu, sc = scale_fit(xtr)
    if name == 'OUT_LR':
        m = LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    else:
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit((xtr - mu) / sc, ytr)
    assert list(m.classes_) == [0, 1]
    return m.predict_proba((xte - mu) / sc)[:, 1].astype(np.float64)


def lookup_score(ktr, ytr, kte):
    pi = float(np.mean(ytr)); tab = {}
    for k, v in zip(ktr, ytr):
        a, b = tab.get(k, (0, 0)); tab[k] = (a + int(v), b + 1)
    return np.array([(tab[k][0] + 2.0 * pi) / (tab[k][1] + 2.0) if k in tab else pi for k in kte], dtype=np.float64)


def my_proxy(inc, occ, sex, halfA, seed):
    X = out_features(inc, occ)
    keys = [r.tobytes() for r in np.ascontiguousarray(np.column_stack((np.asarray(inc, np.float64), np.asarray(occ, np.float64))))]
    keys = np.array(keys, dtype=object)
    res = {}
    for name in ('OUT_LR', 'OUT_HGB', 'LOOKUP'):
        d = []
        for tr, te in ((halfA, ~halfA), (~halfA, halfA)):
            if name == 'LOOKUP':
                s = lookup_score(list(keys[tr]), sex[tr], list(keys[te]))
            else:
                s = fit_score(name, X[tr], sex[tr], X[te], seed)
            d.append(my_auc(sex[te], s))
        res[name] = {'directions': d, 'mean': float(np.mean(d))}
    res['arm_proxy'] = max(res[n]['mean'] for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP'))
    return res


# ---------------------------------------------------------------- nomination (PROTOCOL.md section 7)
def nominate(elig, mean_proxy, worst, cells):
    scm = [f'SCM_b{b}' for b in BETAS]
    el = [a for a in scm if elig[a]]
    if el:
        nominee = min(el, key=lambda a: (mean_proxy[a], scm.index(a))); status = 'NOMINEE_SELECTED'
    else:
        nominee = min(scm, key=lambda a: (worst[a], scm.index(a))); status = 'INELIGIBLE_DIAGNOSTIC_FALLBACK'
    b = nominee.split('_b')[1]; twin = f'TWIN_b{b}'
    matched = lambda a: all(cells[s][a] == cells[s][nominee] for s in (0, 1))
    ib = [f'IBSI_b{b}_g{g}' for g in GAMMAS]; ib_el = [a for a in ib if elig[a]]
    ibsi = min(ib_el, key=lambda a: (mean_proxy[a], ib.index(a))) if ib_el else min(ib, key=lambda a: (worst[a], ib.index(a)))
    sh = [f'SHUF_b{b}_r{r}' for r in range(NSHUF)]; sh_el = [a for a in sh if elig[a] and matched(a)]
    shuf = min(sh_el, key=lambda a: (mean_proxy[a], sh.index(a))) if sh_el else None
    return {'status': status, 'eligible_scm': el, 'nominee': nominee, 'twin': twin, 'twin_matched_capacity': matched(twin),
            'ibsi': ibsi, 'ibsi_eligible': ib_el, 'ibsi_matched_capacity': matched(ibsi), 'shuf': shuf, 'shuf_eligible_and_matched': sh_el}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--proxy-arms', choices=('required', 'nomination', 'all'), default='all')
    ap.add_argument('--out', default=str(HERE / 'check4_selection.json')); a = ap.parse_args()
    t_start = time.process_time()
    rep = {'check': 'CHECK_4_MODEL_SELECT_UTILITY_AND_NOMINATION', 'discrepancies': [], 'notes': []}
    disc = rep['discrepancies']
    SL = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    TL = json.loads((RES / 'TRAINING_COMPLETE_LOCK.json').read_text())
    ext = json.loads((ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'EXTRACTION_SUMMARY.json').read_text())['export_file_sha256']['MODEL_SELECT']
    fbm = json.loads((ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1' / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    # ---- inputs, hashes and row alignment
    hashes = {}
    lab = {}
    ex = PRIV / 'source' / 'exports' / 'MODEL_SELECT'
    with np.load(ex / 'features.npz') as z:
        rid, unit = z['row_id'].copy(), z['unit'].copy()
    hashes['MODEL_SELECT/features.npz'] = sha(ex / 'features.npz') == ext['features']
    for name in ('income', 'occupation', 'sex'):
        hashes[f'MODEL_SELECT/{name}.npz'] = sha(ex / f'{name}.npz') == ext[name]
        with np.load(ex / f'{name}.npz') as z:
            if not (np.array_equal(z['row_id'], rid) and np.array_equal(z['unit'], unit)):
                disc.append(f'MODEL_SELECT {name} export row_id/unit pairing differs from features'); continue
            lab[name] = z[name].astype(np.int64)
    sex = lab['sex']; labels = {t: lab[t] for t in TASKS}
    rep['model_select'] = {'persons': int(len(rid)), 'households': int(len(np.unique(unit))), 'female_share': float(sex.mean()),
                           'sex_values': sorted(map(int, np.unique(sex)))}
    outs, refs, frozen, cells = {}, {}, {}, {}
    for s in (0, 1):
        p = PRIV / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz'
        hashes[f'FIT_{s}/ARM_OUTPUTS.npz==TRAINING_COMPLETE_LOCK'] = sha(p) == TL['arm_outputs_sha256'][f'S{s}']
        hashes[f'FIT_{s}/ARM_OUTPUTS.npz==SELECTION_LOCK'] = sha(p) == SL['arm_output_files_sha256'][f'runs/FIT_{s}/ARM_OUTPUTS.npz']
        with np.load(p) as z:
            al = np.array_equal(z['row_id__MODEL_SELECT'], rid) and np.array_equal(z['unit__MODEL_SELECT'], unit)
            rep.setdefault('row_alignment', {})[f'FIT_{s}/ARM_OUTPUTS MODEL_SELECT'] = bool(al)
            if not al:
                disc.append(f'seed {s}: ARM_OUTPUTS MODEL_SELECT rows differ from the export')
            outs[s] = {}
            for k in z.files:
                parts = k.split('__')
                if len(parts) == 3 and parts[1] == 'MODEL_SELECT':
                    outs[s].setdefault(parts[0], {})[parts[2]] = np.asarray(z[k], dtype=np.float64)
        fb = PRIV / 'source' / 'bank' / f'FROZEN_BANK_S{s}.npz'
        hashes[f'FROZEN_BANK_S{s}==FROZEN_BANK_MANIFEST'] = sha(fb) == fbm[f'S{s}']
        with np.load(fb) as z:
            al = np.array_equal(z['row_id__MODEL_SELECT'], rid) and np.array_equal(z['unit__MODEL_SELECT'], unit)
            rep['row_alignment'][f'FROZEN_BANK_S{s} MODEL_SELECT'] = bool(al)
            if not al:
                disc.append(f'seed {s}: frozen bank MODEL_SELECT rows differ from the export')
            refs[s] = {r: {t: np.asarray(z[f'{r}__MODEL_SELECT__{t}'], dtype=np.float64) for t in TASKS} for r in ('U0', 'Ucal')}
            frozen[s] = {n: {t: np.asarray(z[f'{n}__MODEL_SELECT__{t}'], dtype=np.float64) for t in TASKS} for n in ('NEURAL_RAW', 'H_LJ')}
        with np.load(PRIV / 'runs' / f'FIT_{s}' / 'ARM_TABLES.npz') as z:
            cells[s] = {arm: int(len(np.unique(z[f'{arm}__cell_of_initial']))) for arm in outs[s]}
    rep['input_hashes_match'] = hashes
    for k, v in hashes.items():
        if not v:
            disc.append(f'hash mismatch: {k}')
    arms = sorted(outs[0])
    if sorted(outs[1]) != arms:
        disc.append('arm sets differ between seeds')
    rep['arms'] = arms
    # cell counts vs SELECTION_LOCK final_cells
    cell_diff = [(s, arm, cells[s][arm], SL['final_cells'][str(s)][arm]) for s in (0, 1) for arm in arms if cells[s][arm] != SL['final_cells'][str(s)][arm]]
    rep['final_cells_from_tables_equal_lock'] = not cell_diff
    for d in cell_diff:
        disc.append(f'seed {d[0]} {d[1]}: distinct cell_of_initial {d[2]} vs SELECTION_LOCK final_cells {d[3]}')
    # ---- (1) clauses for every arm (and frozen comparators), compared to the lock
    clause_cmp = {'max_abs_value_diff': 0.0, 'max_abs_normalized_diff': 0.0, 'pass_flag_disagreements': 0, 'n_clauses_compared': 0,
                  'min_abs_margin_to_cap': None, 'arm_feasible_disagreements': 0, 'max_normalized_disagreements': 0}
    mine = {s: {} for s in (0, 1)}
    margins = []
    for s in (0, 1):
        for arm in arms + ['FROZEN_NEURAL_RAW', 'FROZEN_H_LJ']:
            o = outs[s][arm] if not arm.startswith('FROZEN_') else frozen[s][arm[len('FROZEN_'):]]
            cl = my_clauses(labels, o, refs[s])
            locked = {f"{c['task']}|{c['reference']}|{c['metric']}": c for c in SL['per_seed'][str(s)][arm]['clauses']}
            if set(locked) != set(cl) or len(SL['per_seed'][str(s)][arm]['clauses']) != 12:
                disc.append(f'seed {s} {arm}: locked clause keys differ'); continue
            for k, c in cl.items():
                L = locked[k]
                clause_cmp['n_clauses_compared'] += 1
                clause_cmp['max_abs_value_diff'] = max(clause_cmp['max_abs_value_diff'], abs(c['value'] - L['value']))
                clause_cmp['max_abs_normalized_diff'] = max(clause_cmp['max_abs_normalized_diff'], abs(c['normalized'] - L['normalized']))
                if c['pass'] != L['point_pass'] or L['cap'] != c['cap'] or L['seed'] != s:
                    clause_cmp['pass_flag_disagreements'] += 1; disc.append(f'seed {s} {arm} {k}: pass/cap/seed differs (mine {c}, locked {L})')
                margins.append((abs(c['value'] - c['cap']), f's{s}|{arm}|{k}', c['value'], c['cap']))
            mn = max(c['normalized'] for c in cl.values()); feas = all(c['pass'] for c in cl.values())
            rec = {'clauses': cl, 'max_normalized': mn, 'feasible_12': feas}
            if not arm.startswith('FROZEN_'):
                L = SL['per_seed'][str(s)][arm]
                if abs(mn - L['max_normalized']) > 1e-12:
                    clause_cmp['max_normalized_disagreements'] += 1; disc.append(f'seed {s} {arm}: max_normalized {mn} vs locked {L["max_normalized"]}')
                if feas != L['point_feasible_24']:
                    clause_cmp['arm_feasible_disagreements'] += 1; disc.append(f'seed {s} {arm}: per-seed feasibility {feas} vs locked {L["point_feasible_24"]}')
                # subgroup (descriptive) occupation Brier excess vs U0 per SEX
                d = occ_brier_vec(labels['occupation'], o['occupation']) - occ_brier_vec(labels['occupation'], refs[s]['U0']['occupation'])
                sub = {f'sex{g}': float(d[sex == g].mean()) for g in (0, 1)}
                rec['subgroup'] = sub
                for g in ('sex0', 'sex1'):
                    dd = abs(sub[g] - L['subgroup_occupation_brier_excess_vs_U0'][g])
                    clause_cmp['max_abs_subgroup_diff'] = max(clause_cmp.get('max_abs_subgroup_diff', 0.0), dd)
                    if (sub[g] <= 0.005) != L['subgroup_pass_0.005'][g]:
                        disc.append(f'seed {s} {arm} subgroup {g}: pass flag differs')
            mine[s][arm] = rec
    mm = min(margins, key=lambda x: x[0])
    clause_cmp['min_abs_margin_to_cap'] = float(mm[0]); clause_cmp['closest_clause_to_cap'] = {'clause': mm[1], 'value': mm[2], 'cap': mm[3]}
    nom_arms = [x for x in margins if x[1].split('|')[1] in ('SCM_b1.0', 'TWIN_b1.0', 'IBSI_b1.0_g1.0', 'SHUF_b1.0_r2')]
    mn_nom = min(nom_arms, key=lambda x: x[0])
    clause_cmp['closest_clause_to_cap_bank_arms'] = {'clause': mn_nom[1], 'value': mn_nom[2], 'cap': mn_nom[3], 'margin': mn_nom[0]}
    rep['clauses'] = clause_cmp
    elig = {arm: all(mine[s][arm]['feasible_12'] for s in (0, 1)) for arm in arms}
    worst = {arm: max(mine[s][arm]['max_normalized'] for s in (0, 1)) for arm in arms}
    rep['eligibility_mine'] = elig
    rep['worst_normalized_per_seed_mine'] = {arm: [mine[0][arm]['max_normalized'], mine[1][arm]['max_normalized']] for arm in arms}
    # ---- (2) proxy recomputation
    groups = np.unique(unit); split_seed = seed_for8('SCM1proxysplit')
    perm = np.random.default_rng(split_seed).permutation(len(groups)); halfA = np.isin(unit, groups[perm[:len(groups) // 2]])
    rep['proxy_split'] = {'seed': split_seed, 'households_A': int(len(groups) // 2), 'households_B': int(len(groups) - len(groups) // 2),
                          'persons_A': int(halfA.sum()), 'persons_B': int((~halfA).sum()),
                          'household_overlap': int(len(set(unit[halfA]) & set(unit[~halfA])))}
    required = ['SCM_b1.0', 'TWIN_b1.0', 'IBSI_b1.0_g1.0', 'SHUF_b1.0_r2']
    nomination_rel = required + ['SCM_b0.5', 'SCM_b1.5', 'IBSI_b1.0_g0.5', 'IBSI_b1.0_g2.0'] + [f'SHUF_b1.0_r{r}' for r in range(NSHUF)]
    pa = {'required': required, 'nomination': list(dict.fromkeys(nomination_rel)), 'all': arms + ['FROZEN_NEURAL_RAW', 'FROZEN_H_LJ']}[a.proxy_arms]
    proxy_cmp = {'arms_recomputed': pa, 'max_abs_direction_diff': 0.0, 'max_abs_arm_proxy_diff': 0.0, 'per_reader_max_abs_diff': {}}
    my_px = {s: {} for s in (0, 1)}
    t0 = time.process_time()
    for s in (0, 1):
        for arm in pa:
            o = outs[s][arm] if not arm.startswith('FROZEN_') else frozen[s][arm[len('FROZEN_'):]]
            r = my_proxy(o['income'], o['occupation'], sex, halfA, SEED_BASE + s)
            my_px[s][arm] = r
            L = SL['per_seed'][str(s)][arm]['proxy']
            for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP'):
                dd = max(abs(x - y) for x, y in zip(r[n]['directions'], L[n]['directions']))
                proxy_cmp['per_reader_max_abs_diff'][n] = max(proxy_cmp['per_reader_max_abs_diff'].get(n, 0.0), dd)
                proxy_cmp['max_abs_direction_diff'] = max(proxy_cmp['max_abs_direction_diff'], dd)
                if dd > 1e-9:
                    disc.append(f'seed {s} {arm} proxy {n}: directions mine {r[n]["directions"]} vs locked {L[n]["directions"]}')
            proxy_cmp['max_abs_arm_proxy_diff'] = max(proxy_cmp['max_abs_arm_proxy_diff'], abs(r['arm_proxy'] - L['arm_proxy']))
            argmax_mine = max(('OUT_LR', 'OUT_HGB', 'LOOKUP'), key=lambda n: r[n]['mean'])
            argmax_lock = max(('OUT_LR', 'OUT_HGB', 'LOOKUP'), key=lambda n: L[n]['mean'])
            if argmax_mine != argmax_lock:
                disc.append(f'seed {s} {arm}: proxy max reader differs ({argmax_mine} vs {argmax_lock})')
    proxy_cmp['cpu_s'] = round(time.process_time() - t0, 2)
    proxy_cmp['required_arms'] = {arm: {f's{s}': {'mine': my_px[s][arm]['arm_proxy'], 'locked': SL['per_seed'][str(s)][arm]['proxy']['arm_proxy'],
                                                  'max_reader_mine': max(('OUT_LR', 'OUT_HGB', 'LOOKUP'), key=lambda n: my_px[s][arm][n]['mean']),
                                                  'readers_mine': {n: my_px[s][arm][n]['mean'] for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP')}}
                                        for s in (0, 1)} for arm in required}
    rep['proxy'] = proxy_cmp
    # ---- (3) nomination: (a) from the locked values, (b) from my recomputed values (where recomputed; locked otherwise)
    lk_elig = {arm: all(SL['per_seed'][str(s)][arm]['point_feasible_24'] for s in (0, 1)) for arm in arms}
    lk_mean = {arm: float(np.mean([SL['per_seed'][str(s)][arm]['proxy']['arm_proxy'] for s in (0, 1)])) for arm in arms}
    lk_worst = {arm: max(SL['per_seed'][str(s)][arm]['max_normalized'] for s in (0, 1)) for arm in arms}
    lk_cells = {s: {arm: SL['final_cells'][str(s)][arm] for arm in arms} for s in (0, 1)}
    nom_locked = nominate(lk_elig, lk_mean, lk_worst, lk_cells)
    my_mean = {arm: (float(np.mean([my_px[s][arm]['arm_proxy'] for s in (0, 1)])) if arm in my_px[0] else lk_mean[arm]) for arm in arms}
    nom_mine = nominate(elig, my_mean, worst, cells)
    rep['nomination_from_locked_values'] = nom_locked
    rep['nomination_from_my_values'] = nom_mine
    rep['nomination_proxy_source'] = 'recomputed for all arms' if a.proxy_arms == 'all' else f'recomputed for {pa}; locked otherwise'
    lock_nom = {'status': SL['status'], 'nominee': SL['nominee']['arm'], 'twin': SL['twin']['arm'], 'twin_matched_capacity': SL['twin_matched_capacity'],
                'ibsi': SL['ibsi']['arm'], 'ibsi_eligible': SL['ibsi']['eligible_gammas'], 'ibsi_matched_capacity': SL['ibsi_matched_capacity'],
                'shuf': SL['shuf']['arm'], 'shuf_eligible_and_matched': SL['shuf'].get('eligible', []), 'eligible_scm': SL['eligible_scm_settings']}
    for src, nom in (('locked values', nom_locked), ('my values', nom_mine)):
        for k, v in lock_nom.items():
            if nom[k] != v:
                disc.append(f'nomination from {src}: {k} = {nom[k]} vs SELECTION_LOCK {v}')
    # lock's summary numbers
    for key, arm in (('nominee', SL['nominee']['arm']), ('twin', SL['twin']['arm']), ('ibsi', SL['ibsi']['arm']), ('shuf', SL['shuf']['arm'])):
        if abs(SL[key]['mean_proxy'] - my_mean[arm]) > 1e-9:
            disc.append(f'SELECTION_LOCK {key} mean_proxy {SL[key]["mean_proxy"]} vs mine {my_mean[arm]}')
        if SL[key]['model_select_point_feasible_24'] != elig[arm]:
            disc.append(f'SELECTION_LOCK {key} feasibility flag differs')
    sm = {arm: SL['shuffled_matched_capacity'][arm] == all(cells[s][arm] == cells[s][SL['nominee']['arm']] for s in (0, 1)) for arm in SL['shuffled_matched_capacity']}
    if not all(sm.values()):
        disc.append(f'shuffled_matched_capacity differs: {sm}')
    # near-ties that could change a nomination
    def gap(cands, val):
        v = sorted(val[c] for c in cands); return float(v[1] - v[0]) if len(v) > 1 else None
    rep['nomination_margins_mine'] = {'scm_eligible_mean_proxy_gap': gap(nom_mine['eligible_scm'], my_mean),
                                      'ibsi_eligible_mean_proxy_gap': gap(nom_mine['ibsi_eligible'], my_mean),
                                      'shuf_eligible_mean_proxy_gap': gap(nom_mine['shuf_eligible_and_matched'], my_mean),
                                      'mean_proxy': {arm: my_mean[arm] for arm in arms}}
    # F1
    tw, nm = SL['twin']['arm'], SL['nominee']['arm']
    f1 = {s: my_px[s][tw]['arm_proxy'] - my_px[s][nm]['arm_proxy'] for s in (0, 1)}
    rep['F1'] = {'twin_minus_scm_mine': f1, 'locked': SL['falsifier_F1_check'], 'holds_mine': all(v >= 0.020 for v in f1.values()),
                 'max_abs_diff': max(abs(f1[s] - SL['falsifier_F1_check'][str(s)]) for s in (0, 1))}
    if rep['F1']['max_abs_diff'] > 1e-9:
        disc.append(f'F1 differs: mine {f1} vs locked {SL["falsifier_F1_check"]}')
    # ---- RESULTS_TABLES.md section 1
    txt = (RES / 'RESULTS_TABLES.md').read_text()
    sec = txt.split('## 1.')[1].split('## 2.')[0]
    rows = [l for l in sec.splitlines() if l.startswith('| ') and not l.startswith('| Arm') and not l.startswith('|---')]
    tab = {'rows_checked': 0, 'mismatches': []}
    def chk(label, shown, value, decimals):
        tol = 0.5 * 10 ** (-decimals) + 1e-12
        if shown is None or abs(float(shown) - value) > tol:
            tab['mismatches'].append(f'{label}: table {shown} vs recomputed {value:.6f}')
    for l in rows:
        c = [x.strip() for x in l.strip('|').split('|')]
        arm = c[0].replace(' (frozen)', '')
        tab['rows_checked'] += 1
        if arm.startswith('FROZEN_'):
            px = c[4].split(' / ')
            for s in (0, 1):
                v = my_px[s][arm]['arm_proxy'] if arm in my_px[s] else SL['per_seed'][str(s)][arm]['proxy']['arm_proxy']
                chk(f'{arm} proxy s{s}', px[s], v, 4)
            continue
        cs = c[1].split(' / ')
        for s in (0, 1):
            if int(cs[s]) != cells[s][arm]:
                tab['mismatches'].append(f'{arm} cells s{s}: table {cs[s]} vs tables {cells[s][arm]}')
        if (c[2] == 'True') != elig[arm]:
            tab['mismatches'].append(f'{arm} feasible: table {c[2]} vs recomputed {elig[arm]}')
        w = c[3].split(' / '); px = c[4].split(' / '); mb = c[5].split(' / ')
        for s in (0, 1):
            chk(f'{arm} worst s{s}', w[s], mine[s][arm]['max_normalized'], 3)
            v = my_px[s][arm]['arm_proxy'] if arm in my_px[s] else SL['per_seed'][str(s)][arm]['proxy']['arm_proxy']
            chk(f'{arm} proxy s{s}', px[s], v, 4)
            chk(f'{arm} men occ Brier s{s}', mb[s], mine[s][arm]['subgroup']['sex0'], 4)
    m = re.search(r'Nominee (\S+); twin (\S+) \(matched capacity: (\w+)\); IBSI (\S+) \(eligible gammas \[(.*?)\]\); SHUF (\S+)\. Registered F1 check on the proxy \(TWIN - SCM\): ([0-9.]+) / ([0-9.]+)\.', sec)
    if not m:
        tab['mismatches'].append('nominee line not parsed')
    else:
        if m.group(1) != nom_mine['nominee'] or m.group(2) != nom_mine['twin'] or (m.group(3) == 'True') != nom_mine['twin_matched_capacity'] \
                or m.group(4) != nom_mine['ibsi'] or m.group(6) != nom_mine['shuf']:
            tab['mismatches'].append(f'nominee line differs: {m.group(0)}')
        if [x.strip().strip("'") for x in m.group(5).split(',')] != nom_mine['ibsi_eligible']:
            tab['mismatches'].append('eligible gammas list differs')
        for s, g in ((0, 7), (1, 8)):
            chk(f'F1 s{s}', m.group(g), f1[s], 4)
    rep['results_tables_section1'] = tab
    for x in tab['mismatches']:
        disc.append('RESULTS_TABLES.md section 1: ' + x)
    # ---- verdicts
    rep['verdicts'] = {
        'clauses': 'PASS' if (clause_cmp['pass_flag_disagreements'] == 0 and clause_cmp['arm_feasible_disagreements'] == 0
                              and clause_cmp['max_abs_value_diff'] < 1e-12 and clause_cmp['max_normalized_disagreements'] == 0) else 'FAIL',
        'proxy': 'PASS' if proxy_cmp['max_abs_direction_diff'] < 1e-9 else 'FAIL',
        'nomination': 'PASS' if all(nom_locked[k] == v and nom_mine[k] == v for k, v in lock_nom.items()) and rep['F1']['max_abs_diff'] < 1e-9 else 'FAIL',
        'results_tables_section1': 'PASS' if not tab['mismatches'] else 'FAIL'}
    rep['cpu_s_total'] = round(time.process_time() - t_start, 2)
    # drop the per-clause detail of every arm from the JSON except the nominee bank arms (keeps the file small; aggregates only)
    keep = {SL['nominee']['arm'], SL['twin']['arm'], SL['ibsi']['arm'], SL['shuf']['arm'], 'C0'}
    rep['clauses_detail_bank_arms'] = {f's{s}': {arm: mine[s][arm] for arm in sorted(keep)} for s in (0, 1)}
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=float))
    print(json.dumps({'verdicts': rep['verdicts'], 'clauses': clause_cmp, 'proxy': {k: v for k, v in proxy_cmp.items() if k != 'required_arms'},
                      'nomination_mine': nom_mine, 'F1': rep['F1'], 'n_discrepancies': len(disc), 'discrepancies': disc[:40],
                      'cpu_s_total': rep['cpu_s_total']}, indent=1, default=float))


if __name__ == '__main__':
    main()

"""Independent verifier: the ONE logged DEV_REPORT replay of the income-headroom successor campaign (check 8, DEV part).

Public-use ACS 2019 Florida person records (development roles). Readers estimate the recorded SEX column from released income
and occupation probability vectors. This script recomputes, with the verifier's own code, the utility clauses, within-SEX
guards, pair benefits, recipient differences, twin and component contrasts, the 0.020 target and the claim labels
E, R (R2, R3), P, N, M, C, T, O, D and the composite under CLAIMS_DRAFT.md, and compares them with
report/DEV_REPORT_RESULTS.json.

Own code: AUC (rank / tie-group formula), household bootstrap weights (registered scheme: seed 2026101705, B = 2048, household
indices drawn uniformly with replacement; checked against task_headroom.inference.group_weights), weighted bootstrap AUC
(per-replicate tie-group sums), task vectors, endpoint bounds, claim logic. Not imported: ihs.report, ihs.claims, amd.report.

--invented: identical pipeline on invented labels (no DEV_REPORT truth, no access event) for a dry run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ihs import common as C                                    # admitted loaders + the logged dev_report_truth gate only

RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
PRIV = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
EVENT = 'DEV_REPORT_TRUTH_REREAD_BY_IHS_INDEPENDENT_VERIFIER'
TASKS = ('income', 'occupation'); K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD = 0.005; VIEWS = ('income', 'occupation', 'pair'); SC = ('ENHANCED', 'ORIGINAL')
R2_FACTOR = 2.0 * np.sqrt(3955 / 5306)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wire(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


# ------------------------------------------------------------------ own statistics
def own_weights(unit, seed, B):
    groups, inv = np.unique(np.asarray(unit), return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(B, G))
    cnt = np.empty((B, G), dtype=np.float64)
    for b in range(B):
        cnt[b] = np.bincount(idx[b], minlength=G)
    return cnt[:, inv], G


def own_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    r = stats.rankdata(s, method='average')
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def own_boot_auc(y, s, W):
    """Weighted AUC per replicate: sum over tie groups of P_g * (N_below_g + N_g / 2) / (P * N)."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    _, g = np.unique(s, return_inverse=True); m = g.max() + 1
    pos = (y == 1); neg = ~pos
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        w = W[b]
        P = np.bincount(g[pos], weights=w[pos], minlength=m); N = np.bincount(g[neg], weights=w[neg], minlength=m)
        below = np.concatenate(([0.0], np.cumsum(N)[:-1]))
        out[b] = float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))
    return out


def tv(y, p, k):
    p = np.asarray(p, dtype=np.float64); y = np.asarray(y)
    onehot = np.zeros_like(p); onehot[np.arange(len(y)), y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - onehot) ** 2).sum(1)}


def tcrit(G, one_sided):
    return float(stats.t.ppf(1.0 - one_sided, G - 1))


def ep(point, draws, kind, margin, c1, cF):
    se = float(np.std(draws, ddof=1))
    r = {'point': float(point), 'se': se, 'clause_L1': point - c1 * se, 'clause_U1': point + c1 * se,
         'family_lower': point - cF * se, 'family_upper': point + cF * se, 'kind': kind, 'margin': margin}
    if kind == 'benefit':
        r['point_pass'] = bool(point > margin); r['clause_pass'] = bool(r['clause_L1'] > margin)
    elif kind == 'no_excess':
        r['point_pass'] = bool(point <= margin); r['clause_pass'] = bool(r['clause_U1'] <= margin)
    else:
        r['point_pass'] = None; r['clause_pass'] = None
    return r


# ------------------------------------------------------------------ claims (own reading of CLAIMS_DRAFT.md)
def claims(E, CV, declared, c_marg):
    """E: endpoint dict for one scorecard (keys as ANALYSIS_LOCK). Returns labels and details."""
    out = {}
    util_keys = [k for k in E if k.startswith('UTILITY|S|')]
    guard_keys = [k for k in E if k.startswith('SUBGROUP|S|')]
    rec_keys = [f'NW|{x}|{v}|s{s}' for x in ('NEURAL_RAW', 'H_LJ') for v in TASKS for s in (0, 1)]
    UG = all(E[k]['point'] <= E[k]['margin'] for k in util_keys) and len(util_keys) == 24
    SG = all(E[k]['point'] <= E[k]['margin'] for k in guard_keys) and len(guard_keys) == 4
    RG = all(E[k]['point'] <= 0.010 for k in rec_keys)
    E_POINT = UG and SG and RG
    failed = [n for n, ok in (('UG', UG), ('SG', SG), ('RG', RG)) if not ok]
    conf28 = util_keys + guard_keys
    unres = [k for k in conf28 if E[k]['family_upper'] > E[k]['margin']]
    viol = [k for k in conf28 if E[k]['family_lower'] > E[k]['margin']]
    uc = 'ESTABLISHED_UTILITY_VIOLATION' if viol else (f'CONFIDENCE_UNRESOLVED ({len(unres)} of 28)' if unres else 'CONFIDENCE_FEASIBLE')
    out['E'] = {'label': 'DEV_POINT_ELIGIBLE' if E_POINT else 'DEV_POINT_INELIGIBLE', 'failed_in_order': failed,
                'utility_confidence': uc, 'unresolved_keys': unres}
    # R2 / R3
    inc = [k for k in util_keys if k.split('|')[2] == 'income']
    r2 = {}
    for k in inc:
        hk = k.replace('UTILITY|S|', 'UTILITY|H_LJ|')
        m = E[k]['margin'] - E[k]['point']; se_u = max(E[k]['se'], E[hk]['se'])
        r2[k] = {'margin': m, 'se_S': E[k]['se'], 'se_H_LJ': E[hk]['se'], 'required': R2_FACTOR * se_u, 'ratio_se_units': m / se_u,
                 'holds': bool(m >= R2_FACTOR * se_u)}
    if any(v['margin'] < 0 for v in r2.values()):
        R2 = 'CAP_EXCEEDED_AT_POINT'
    elif all(v['holds'] for v in r2.values()):
        R2 = 'RESERVE_REPLICATED'
    else:
        R2 = 'RESERVE_ERODED'
    r3u = [k for k in inc if E[k]['family_upper'] > E[k]['margin']]; r3v = [k for k in inc if E[k]['family_lower'] > E[k]['margin']]
    R3 = 'ESTABLISHED_INCOME_VIOLATION' if r3v else (f'INCOME_CONFIDENCE_UNRESOLVED ({len(r3u)} of 12)' if r3u else 'INCOME_CONFIDENCE_RESOLVED')
    out['R'] = {'R2': R2, 'R3': R3, 'R2_detail': r2, 'R3_unresolved': r3u,
                'R4_projected_margin_se_units_17510': {k: v['margin'] / (v['se_S'] * np.sqrt(3955 / 17510)) for k, v in r2.items()}}
    # P / N / T
    ben = {f'BEN|{x}|s{s}': E[f'BEN|{x}|s{s}'] for x in ('NEURAL_RAW', 'H_LJ') for s in (0, 1)}
    nw = {k: E[k] for k in rec_keys}
    cond = {'CV': CV, 'E_POINT': E_POINT, **{k: v['clause_pass'] for k, v in ben.items()}, **{k: v['clause_pass'] for k, v in nw.items()}}
    P = all(cond.values())
    if P:
        Pl = 'SUPERIOR_TO_ALL_ELIGIBLE'
    elif not CV:
        Pl = 'CONTROL_FAILURE'
    elif not UG:
        Pl = 'FAILS_DEV_UTILITY_GATE'
    elif not SG:
        Pl = 'FAILS_DEV_SUBGROUP_POINT_GUARD'
    elif not RG:
        Pl = 'FAILS_DEV_RECIPIENT_POINT_GATE'
    else:
        Pl = 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
    out['P'] = {'label': Pl, 'holds': P, 'condition_vector': cond, 'failing': [k for k, v in cond.items() if not v],
                'benefit_L1': {k: v['clause_L1'] for k, v in ben.items()}}
    if all(v['clause_U1'] <= 0.010 for v in nw.values()):
        Nl = 'NON_WORSENING_ESTABLISHED'
    elif all(v['point'] <= 0.010 for v in nw.values()):
        Nl = 'NON_WORSENING_UNRESOLVED'
    else:
        Nl = 'NON_WORSENING_VIOLATED_AT_POINT'
    out['N'] = {'label': Nl, 'max_point': max(v['point'] for v in nw.values()), 'U1': {k: v['clause_U1'] for k, v in nw.items()}}
    if P and all(v['clause_L1'] > 0.020 for v in ben.values()):
        Tl = 'MATERIAL_TARGET_ESTABLISHED'
    elif all(v['point'] > 0.020 for v in ben.values()):
        Tl = 'TARGET_POINT_ONLY'
    else:
        Tl = 'TARGET_NOT_ESTABLISHED'
    out['T'] = {'label': Tl, 'points': {k: v['point'] for k, v in ben.items()}}
    # M
    tw = [E[f'BEN|TW|s{s}']['clause_pass'] for s in (0, 1)] + [E[f'NW|TW|{v}|s{s}']['clause_pass'] for v in TASKS for s in (0, 1)]
    step1 = CV and E_POINT and declared['AV_TW'] and all(tw)
    Ml = ('MECHANISM_ATTRIBUTION_ESTABLISHED' if step1 and P else 'TASK_ONLY_ATTRIBUTION_ONLY' if step1 else 'NOT_ESTABLISHED')
    out['M'] = {'label': Ml, 'step1': step1, 'step2': Pl if step1 else 'NOT_TESTED_SEQUENCE_STOPPED'}
    # C
    if declared['CC']:
        cc = [E[f'BEN|CC|s{s}']['clause_pass'] for s in (0, 1)] + [E[f'NW|CC|{v}|s{s}']['clause_pass'] for v in TASKS for s in (0, 1)]
        Cl = 'COMPONENT_ATTRIBUTION_ESTABLISHED' if (CV and E_POINT and declared['AV_CC'] and all(cc)) else 'NOT_ESTABLISHED'
    else:
        Cl = 'NOT_CLAIMED'
    out['C'] = {'label': Cl}
    # D
    lb = [E[f'DESC|BEN|B2TWIN|s{s}']['point'] - c_marg * E[f'DESC|BEN|B2TWIN|s{s}']['se'] for s in (0, 1)]
    out['D'] = {'GAIN_OVER_B2TWIN_DESCRIPTIVE': bool(all(x > 0 for x in lb)), 'marginal_lower': lb}
    # composite
    comp = 'COMPETITIVE_SUCCESSOR' if (E_POINT and R2 == 'RESERVE_REPLICATED' and P) else ('SUPERIOR_WITHOUT_RESERVE' if (E_POINT and P) else 'NOT_COMPETITIVE')
    out['COMPOSITE'] = {'label': comp}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--invented', action='store_true'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time(); out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    lock = json.loads((RES / 'ANALYSIS_LOCK.json').read_text()); rlock = json.loads((RES / 'REPORTING_LOCK.json').read_text())
    man = json.loads((RES / 'SERVICE_BANK_MANIFEST.json').read_text()); stored = json.loads((RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    pre = {'locks': {n: sha(RES / n) == h for n, h in rlock['locks_sha256'].items()}}
    # ---- everything label-free first (any loading failure happens BEFORE the access event)
    D = C.load_role('DEV_REPORT', ('features',)); unit, row_id = D['unit'], D['row_id']; del D
    inf = lock['inference']; B, bseed = inf['bootstrap_B'], inf['bootstrap_seed']
    W, G = own_weights(unit, bseed, B)
    from task_headroom.inference import group_weights            # cross-check of the registered scheme only
    W2, G2 = group_weights(unit, seed=bseed, draws=B)
    pre['weights_equal_group_weights'] = bool(np.array_equal(W, W2) and G == G2); del W2
    pre['G'] = int(G); pre['n'] = int(len(unit))
    banks, scores, rr, frozen = {}, {}, {}, {}
    for s in (0, 1):
        p = PRIV / 'runs' / 'FINAL' / f'SERVICE_BANK_S{s}.npz'
        pre[f'bank_S{s}_sha_ok'] = sha(p) == man['bank_files_sha256'][f'S{s}']
        bk = {}
        with np.load(p) as z:
            for k in z.files:
                parts = k.split('__')
                if len(parts) == 3 and parts[1] == 'DEV_REPORT':
                    bk.setdefault(parts[0], {})[parts[2]] = z[k].copy()
                elif k == 'row_id__DEV_REPORT':
                    pre[f'bank_S{s}_rows_ok'] = bool(np.array_equal(z[k], row_id))
        banks[s] = bk
        rd = PRIV / 'runs' / f'READERS_S{s}'
        pre[f'readers_S{s}_dev_scores_sha_ok'] = sha(rd / 'DEV_REPORT_SCORES.npz') == rlock['readers'][f'S{s}']['dev_report_scores_sha256']
        pre[f'readers_S{s}_av_scores_sha_ok'] = sha(rd / 'AV_SCORES.npz') == rlock['readers'][f'S{s}']['av_scores_sha256']
        pre[f'readers_S{s}_record_sha_ok'] = sha(rd / 'READERS_SEED.json') == rlock['readers'][f'S{s}']['record_sha256']
        with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
            scores[s] = {k: z[k].copy() for k in z.files}
        pre[f'scores_S{s}_rows_ok'] = bool(np.array_equal(scores[s]['row_id'], row_id))
        rr[s] = json.loads((rd / 'READERS_SEED.json').read_text())
        fb, fids = C.frozen_bank(s, roles=('DEV_REPORT',))
        pre[f'frozen_S{s}_rows_ok'] = bool(np.array_equal(fids['row_id__DEV_REPORT'], row_id))
        frozen[s] = {r: {t: fb[r]['DEV_REPORT'][t] for t in TASKS} for r in ('U0', 'Ucal')}
    if not all(v for k, v in pre.items() if isinstance(v, bool)) or not all(pre['locks'].values()):
        raise SystemExit(f'pre-access checks failed: {pre}')
    # ---- truth
    if a.invented:
        rng = np.random.default_rng(99)
        T = {'income': rng.integers(0, 2, len(unit)), 'occupation': rng.integers(0, 5, len(unit)), 'sex': rng.integers(0, 2, len(unit)),
             'unit': unit, 'row_id': row_id}
        event = None
    else:
        T = C.dev_report_truth('independent verification: the one verifier replay of the frozen successor DEV_REPORT evaluation '
                               '(own code: utility, guards, pair benefits, recipient differences, twin/component contrasts, '
                               'target, claim labels)', EVENT)
        event = [e for e in C.dev_access_events() if e.get('event') == EVENT]
    t_access = C.now()
    if not (np.array_equal(T['row_id'], row_id) and np.array_equal(T['unit'], unit)):
        raise SystemExit('truth / features row order differs')
    y = {t: np.asarray(T[t], dtype=np.int64) for t in TASKS}; sex = np.asarray(T['sex'], dtype=np.int64)
    Wsum = W.sum(1)
    n_claims = inf['n_claims']; c1 = tcrit(G, inf['alpha_family_one_sided'] / n_claims)
    cF = {f: tcrit(G, 0.05 / (2 * k)) for f, k in inf['K'].items()}; cm = tcrit(G, 0.025)
    res = {'schema': 'IHS1-verifier-dev-replay-v1', 'invented': a.invented, 'event': EVENT if not a.invented else None,
           'access_utc': t_access, 'access_log_entries_for_event': event, 'pre_access_checks': pre,
           'critical': {'c1': c1, 'c_family': cF, 'marginal': cm}, 'errors': errors}
    # ---- per (service, seed, scorecard, view) recovery; utility vectors
    auc_pt, auc_dr, cache_pt, cache_dr = {}, {}, {}, {}
    vec, wires, comp_per_seed = {}, {}, {}
    try:
        for s in (0, 1):
            bk = banks[s]
            wires[s] = {svc: {t: wire(bk[svc][t]) for t in TASKS} for svc in bk}
            u0 = {t: tv(y[t], frozen[s]['U0'][t], K[t]) for t in TASKS}; ucal = {t: tv(y[t], frozen[s]['Ucal'][t], K[t]) for t in TASKS}
            for svc in sorted(bk):
                for scn, wk in (('ENHANCED', 'enhanced'), ('ORIGINAL', 'original')):
                    for view in VIEWS:
                        key = rr[s]['winners'][svc][wk][view]['reader_key']
                        if (s, key) not in cache_pt:
                            sc = scores[s][key]
                            cache_pt[(s, key)] = own_auc(sex, sc); cache_dr[(s, key)] = own_boot_auc(sex, sc, W)
                        auc_pt[(svc, s, scn, view)] = cache_pt[(s, key)]; auc_dr[(svc, s, scn, view)] = cache_dr[(s, key)]
                for t in TASKS:
                    vec[(svc, s, t)] = tv(y[t], bk[svc][t], K[t])
                vec[('U0', s)] = u0; vec[('Ucal', s)] = ucal
    except Exception:
        errors.append(traceback.format_exc())
    # ---- endpoints
    E_all = {}
    draws_save = {}
    for scn in SC:
        E = {}
        for e in lock['endpoints']:
            try:
                s = e['seed']; ident = False
                if e['type'] == 'auc':
                    A_, B_ = e['a'], e['b']
                    point = auc_pt[(A_, s, scn, e['view'])] - auc_pt[(B_, s, scn, e['view'])]
                    dr = auc_dr[(A_, s, scn, e['view'])] - auc_dr[(B_, s, scn, e['view'])]
                    ident = A_ == B_ or (e['view'] in TASKS and wires[s][A_][e['view']] == wires[s][B_][e['view']])
                elif e['type'] == 'utility':
                    o = vec[(e['service'], s, e['task'])]; r = vec[(e['ref'], s)][e['task']]
                    d = {'accuracy_loss': r['accuracy'] - o['accuracy'], 'nll_excess': o['nll'] - r['nll'], 'brier_excess': o['brier'] - r['brier']}[e['metric']]
                    point = float(d.mean()); dr = (W @ d) / Wsum
                elif e['type'] == 'guard':
                    d = vec[(e['service'], s, 'occupation')]['brier'] - vec[('U0', s)]['occupation']['brier']
                    m_ = sex == e['group']; Wg = W[:, m_]
                    point = float(d[m_].mean()); dr = (Wg @ d[m_]) / Wg.sum(1)
                elif e['type'] == 'loss_diff':
                    d = vec[(e['a'], s, e['task'])][e['metric']] - vec[(e['b'], s, e['task'])][e['metric']]
                    point = float(d.mean()); dr = (W @ d) / Wsum
                    ident = wires[s][e['a']][e['task']] == wires[s][e['b']][e['task']]
                else:
                    raise ValueError(e['type'])
                if ident:
                    r_ = {'point': 0.0, 'se': 0.0, 'clause_L1': 0.0, 'clause_U1': 0.0, 'family_lower': 0.0, 'family_upper': 0.0,
                          'kind': e['kind'], 'margin': e['margin'], 'identical_release': True,
                          'point_pass': e['kind'] == 'no_excess', 'clause_pass': e['kind'] == 'no_excess'}
                else:
                    r_ = ep(point, dr, e['kind'], e['margin'], c1, cF[e['family']]); r_['identical_release'] = False
                    draws_save[f'{scn}|{e["key"]}'] = np.asarray(dr, dtype=np.float64)
                r_['marginal95'] = [r_['point'] - cm * r_['se'], r_['point'] + cm * r_['se']]
                E[e['key']] = r_
            except Exception:
                errors.append(f"{scn} {e['key']}: " + traceback.format_exc())
        E_all[scn] = E
    try:
        np.savez(out_dir / 'VERIFIER_ENDPOINT_DRAWS.npz', **draws_save)
    except Exception:
        errors.append(traceback.format_exc())
    # ---- controls (CV): DEV recomputed, AV from the reader records
    CV = True; cvrec = {}
    try:
        band = lock['controls_bands']
        for s in (0, 1):
            for view in VIEWS:
                for name, c in rr[s]['controls'][view].items():
                    key = f'CONTROL__{view}__{name}'
                    pl = np.where(sex == 1, scores[s][f'{key}__PLANTED_1'], scores[s][f'{key}__PLANTED_0'])
                    sh, pa = own_auc(sex, scores[s][f'{key}__SHUFFLED']), own_auc(sex, pl)
                    ok = (band['shuffled'][0] <= sh <= band['shuffled'][1] and pa >= band['planted_min']
                          and band['shuffled'][0] <= c['shuffled_attack_validation_auc'] <= band['shuffled'][1]
                          and c['planted_attack_validation_auc'] >= band['planted_min'])
                    CV = CV and ok
                    cvrec[f'S{s}|{view}|{name}'] = {'shuffled_dev': sh, 'planted_dev': pa, 'ok': ok}
    except Exception:
        errors.append(traceback.format_exc()); CV = None
    res['CV'] = CV; res['controls'] = cvrec
    # ---- claims per scorecard
    kept = {s: set(man['per_seed'][f'S{s}']['kept']) for s in (0, 1)}
    declared = {'CC': lock['declared']['CC'], 'AV_TW': all('TW' in kept[s] for s in (0, 1)), 'AV_CC': all('CC' in kept[s] for s in (0, 1))}
    labels = {}
    for scn in SC:
        try:
            labels[scn] = claims(E_all[scn], bool(CV), declared, cm)
        except Exception:
            errors.append(f'claims {scn}: ' + traceback.format_exc())
    # reader sensitivity: clause-pass disagreement on any conjunction key of P, M, C
    sens = []
    try:
        conj = {'P': [f'BEN|{x}|s{s}' for x in ('NEURAL_RAW', 'H_LJ') for s in (0, 1)] + [f'NW|{x}|{v}|s{s}' for x in ('NEURAL_RAW', 'H_LJ') for v in TASKS for s in (0, 1)],
                'M': [f'BEN|TW|s{s}' for s in (0, 1)] + [f'NW|TW|{v}|s{s}' for v in TASKS for s in (0, 1)],
                'C': [f'BEN|CC|s{s}' for s in (0, 1)] + [f'NW|CC|{v}|s{s}' for v in TASKS for s in (0, 1)]}
        for cl, keys in conj.items():
            for k in keys:
                a_, b_ = E_all['ENHANCED'][k]['clause_pass'], E_all['ORIGINAL'][k]['clause_pass']
                if a_ != b_:
                    sens.append({'claim': cl, 'key': k, 'ENHANCED': a_, 'ORIGINAL': b_})
    except Exception:
        errors.append(traceback.format_exc())
    res['reader_sensitive'] = sens
    # O: inherited occupation output (bitwise on DEV both seeds) and the 16 conditions vs the R2 rule
    try:
        inh = all(wires[s]['SUCCESSOR']['occupation'] == wires[s]['H_LJ']['occupation'] for s in (0, 1))
        below = []
        det = {}
        for k, r_ in E_all['ENHANCED'].items():
            if (k.startswith('UTILITY|S|occupation|') or k.startswith('SUBGROUP|S|')):
                m = r_['margin'] - r_['point']; req = R2_FACTOR * r_['se']
                det[k] = {'margin': m, 'se': r_['se'], 'margin_se_units': m / r_['se'] if r_['se'] > 0 else None, 'below_rule': bool(m < req)}
                if m < req:
                    below.append(k)
        Ol = ('INHERITED_CONSTRAINTS_BIND' if below else 'INHERITED_CONSTRAINTS_HAVE_RESERVE') if inh else 'OCCUPATION_CHANGED (not evaluated)'
        for scn in labels:
            labels[scn]['O'] = {'label': Ol, 'inherited_bitwise': inh, 'below_rule': below, 'n_conditions': len(det), 'detail': det}
    except Exception:
        errors.append(traceback.format_exc())
    # ---- comparison with the stored report
    cmp = {'endpoint_point_maxdiff': {}, 'endpoint_se_maxdiff': {}, 'endpoint_bound_maxdiff': {}, 'clause_pass_mismatch': [],
           'n_endpoints': {}, 'labels': {}, 'per_seed_recovery_maxdiff': 0.0, 'per_seed_utility_maxdiff': 0.0}
    try:
        for scn in SC:
            st = stored['scorecards'][scn]['endpoints']; mine = E_all[scn]
            cmp['n_endpoints'][scn] = [len(mine), len(st)]
            dp = ds = db = 0.0
            for k, r_ in mine.items():
                q = st[k]
                dp = max(dp, abs(r_['point'] - q['point'])); ds = max(ds, abs(r_['se'] - q['se']))
                for bnd in ('clause_L1', 'clause_U1', 'family_lower', 'family_upper'):
                    db = max(db, abs(r_[bnd] - q[bnd]))
                if r_.get('clause_pass') != q.get('clause_pass'):
                    cmp['clause_pass_mismatch'].append([scn, k, r_.get('clause_pass'), q.get('clause_pass')])
            cmp['endpoint_point_maxdiff'][scn] = dp; cmp['endpoint_se_maxdiff'][scn] = ds; cmp['endpoint_bound_maxdiff'][scn] = db
            sl = stored['labels'][scn]
            mine_l = {c: labels[scn][c]['label'] for c in ('E', 'P', 'N', 'M', 'C', 'T', 'O', 'COMPOSITE')}
            mine_l['R2'] = labels[scn]['R']['R2']; mine_l['R3'] = labels[scn]['R']['R3']
            mine_l['D'] = 'GAIN_OVER_B2TWIN_DESCRIPTIVE' if labels[scn]['D']['GAIN_OVER_B2TWIN_DESCRIPTIVE'] else 'NO_DESCRIPTIVE_GAIN_OVER_B2TWIN'
            if scn == 'ENHANCED':                                  # the qualifier belongs to the label decided on ENHANCED
                for c in ('P', 'M', 'C'):
                    if [x for x in sens if x['claim'] == c]:
                        mine_l[c] += ' [READER_SENSITIVE]'
            cmp['labels'][scn] = {'mine': mine_l, 'stored': sl}
        mr = mu = mus = 0.0
        for s in (0, 1):
            ps = stored['per_seed'][f'S{s}']['services']
            for svc, v in ps.items():
                for scn in SC:
                    for view in VIEWS:
                        q = v['recovery'][scn][view]
                        mr = max(mr, abs(auc_pt[(svc, s, scn, view)] - q['auc']), abs(float(np.std(auc_dr[(svc, s, scn, view)], ddof=1)) - q['auc_se']))
                for t in TASKS:
                    o = vec[(svc, s, t)]
                    for ref in ('U0', 'Ucal'):
                        r = vec[(ref, s)][t]
                        for metric, d in (('accuracy_loss', r['accuracy'] - o['accuracy']), ('nll_excess', o['nll'] - r['nll']), ('brier_excess', o['brier'] - r['brier'])):
                            q = v['utility'][f'{t}|{ref}|{metric}']
                            mu = max(mu, abs(float(d.mean()) - q['point'])); mus = max(mus, abs(float(np.std((W @ d) / Wsum, ddof=1)) - q['se']))
        cmp['per_seed_recovery_maxdiff'] = mr; cmp['per_seed_utility_point_maxdiff'] = mu; cmp['per_seed_utility_se_maxdiff'] = mus
        cmp['stored_CV'] = stored['CV']; cmp['stored_reader_sensitive'] = stored['reader_sensitive']
    except Exception:
        errors.append(traceback.format_exc())
    res['comparison'] = cmp
    res['labels'] = {scn: {c: (v if c not in ('R', 'O', 'P') else {kk: vv for kk, vv in v.items() if kk not in ('detail',)}) for c, v in labels[scn].items()} for scn in labels}
    res['endpoints'] = {scn: {k: {kk: vv for kk, vv in r_.items()} for k, r_ in E_all[scn].items()} for scn in E_all}
    res['wall_s'] = time.time() - t0
    (out_dir / 'VERIFIER_DEV_REPLAY.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'errors': len(errors), 'comparison': {k: v for k, v in cmp.items() if k != 'labels'}, 'labels': cmp.get('labels')}, indent=1, default=float))


if __name__ == '__main__':
    main()

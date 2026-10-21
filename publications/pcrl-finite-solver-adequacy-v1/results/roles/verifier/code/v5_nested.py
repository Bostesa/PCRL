"""V5 (nested solver comparison) and V6 (route S decision inputs and labels) for campaign `pcrl_finite_solver_adequacy_v1`.

V5, per version, from .private/<campaign>/solver/NESTED_SOLVERS_V<v>.npz with the verifier's own code (vcore, v2_replay): every
entry's held-out means (task metrics, fit_pi_pair_mi, men's / women's occupation Brier, pair / income / occupation LOOKUP AUC) and
spends over the twin with household-clustered SEs, against nested/NESTED_SOLVERS_V<v>.json; the original-solver entries (PRC_TWIN,
PRC@<lam>, HQ@<lam>) against the P1 replay's per-person outputs (every shared key, exact); households against the admitted
MODEL_FIT rows (label-free) and SEX against the replay file's SEX (itself checked against the admitted labels in V2: no new read).

V6, from the same files: the route S nested inputs (RULES 4): the select_path S2 choices of the original (PRC@<lam>) and revised
(PRCsv1@<lam>) paths with the S2 budget of SELECTION_V<v>.json; the total-NLL spends over the twin; the bracket (the cheapest
original point with spend >= revised spend - 0.0005, else the most-spending one); for every original point the paired held-out
pair / income / occupation LOOKUP AUC differences (revised - original) with a household-bootstrap SE (B = 400,
default_rng(SeedSequence([2026102200, v, 4])), the same household weights for both) and the total-NLL difference with a
household-clustered SE; compared with decisions/ROUTE_S_INPUT.json. Then fsa.rules.route_s on an input built from the verifier's
own numbers (fitted part from V4's ROUTE_S_FITTED_V<v>.json; nested part from here; flags checked from git, the ledger and the
access log) and fsa.rules.advancement with that S input in place of the recorded one (D and E inputs as recorded); the labels are
compared with decisions/ADVANCEMENT.json.

    python3 .../v5_nested.py --selftest
    python3 .../v5_nested.py --tag V1          (queue unit, stage verify; both versions)
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402
import v2_replay as R2  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/v5_nested.py'
GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TWIN = 'PRC_TWIN'
TOL = 1e-9
B = 400
SEED = 2026102200
TASK_NO_WORSE = 0.0005
VIEWS = ('pair', 'income', 'occupation')
TASKK = ('inc_nll', 'occ_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc', 'fit_pi_pair_mi')


def entries(arr):
    return sorted({k.split('|', 1)[0] for k in arr if '|' in k})


def means_of(arr, e, sex):
    g = lambda m: np.asarray(arr[f'{e}|{m}'], dtype=np.float64)       # noqa: E731
    d = {k: float(g(k).mean()) for k in TASKK if f'{e}|{k}' in arr}
    d['occ_brier_men'] = float(g('occ_brier')[sex == 0].mean())
    d['occ_brier_women'] = float(g('occ_brier')[sex == 1].mean())
    for vw in VIEWS:
        d[f'auc_{vw}'] = V.auc(g(f'look_{vw}'), sex)
    return d


def bracket(path, r_spend):
    pts = [p for p in path if np.isfinite(p['spend'])]
    ok = [p for p in pts if p['spend'] >= r_spend - TASK_NO_WORSE]
    return min(ok, key=lambda p: p['spend'])['id'] if ok else max(pts, key=lambda p: p['spend'])['id']


def route_s_nested(arr, budget_s2, v):
    """Own route S nested inputs for one version."""
    unit = np.asarray(arr['unit'])
    sex = np.asarray(arr['sex']).astype(np.int64)
    orig = [f'PRC@{g}' for g in GRID]
    revd = [f'PRCsv1@{g}' for g in GRID]
    sp = {e: R2.spends(arr, e, unit, sex) for e in orig + revd}
    o2, mo = R2.select(orig, sp, budget_s2)
    rp, mr = R2.select(revd, sp, budget_s2)
    tot = {e: R2.total_nll_spend(arr, e) for e in orig + revd}
    out = {'revised_point': rp, 'revised_spend': tot.get(rp) if rp else None, 'original_s2': o2,
           'original_path': [{'id': e, 'spend': tot[e]} for e in orig], 'revised_path_spends': {e: tot[e] for e in revd},
           'paired': {}, 'alias_of_original_s2': None}
    margins = {'original': mo, 'revised': mr}
    if rp is None:
        return out, margins, None
    W = V.bootstrap_weights(unit, B, [SEED, int(v), 4])
    totp = lambda e: np.asarray(arr[f'{e}|inc_nll'], dtype=np.float64) + np.asarray(arr[f'{e}|occ_nll'], dtype=np.float64)  # noqa
    draws_r, auc_r = {}, {}
    for vw in VIEWS:
        s = np.asarray(arr[f'{rp}|look_{vw}'], dtype=np.float64)
        wa = V.WeightedAUC(s, sex)
        draws_r[vw] = np.array([wa(W[b]) for b in range(B)])
        auc_r[vw] = V.auc(s, sex)
    for e in orig:
        row = {}
        for vw in VIEWS:
            s = np.asarray(arr[f'{e}|look_{vw}'], dtype=np.float64)
            wa = V.WeightedAUC(s, sex)
            d = draws_r[vw] - np.array([wa(W[b]) for b in range(B)])
            row['d_auc' if vw == 'pair' else f'd_auc_{vw}'] = [auc_r[vw] - V.auc(s, sex), float(np.std(d, ddof=1))]
        diff = totp(rp) - totp(e)
        row['d_spend'] = [float(diff.mean()), V.cluster_se(diff, unit)]
        out['paired'][e] = row
    out['bracket_own'] = bracket(out['original_path'], out['revised_spend'])
    return out, margins, None


def compare_nested(mine, rec, tol=TOL):
    bad = []
    for key in ('revised_point', 'original_s2', 'alias_of_original_s2'):
        if mine.get(key) != rec.get(key):
            bad.append(f'{key}: mine {mine.get(key)} recorded {rec.get(key)}')
    if (mine.get('revised_spend') is None) != (rec.get('revised_spend') is None) or (
            mine.get('revised_spend') is not None and abs(mine['revised_spend'] - rec['revised_spend']) > tol):
        bad.append(f'revised_spend mine {mine.get("revised_spend")} recorded {rec.get("revised_spend")}')
    rp = {p['id']: p['spend'] for p in rec.get('original_path', [])}
    for p in mine['original_path']:
        if p['id'] not in rp or abs(p['spend'] - rp[p['id']]) > tol:
            bad.append(f'original_path {p["id"]}')
    for e, x in mine.get('revised_path_spends', {}).items():
        if abs(x - rec.get('revised_path_spends', {}).get(e, np.inf)) > tol:
            bad.append(f'revised_path_spends {e}')
    for e, row in mine['paired'].items():
        r = rec['paired'].get(e, {})
        for k2, (est, se) in row.items():
            if k2 not in r or abs(est - r[k2][0]) > tol or abs(se - r[k2][1]) > tol:
                bad.append(f'paired {e} {k2}: mine {[est, se]} recorded {r.get(k2)}')
    return bad


# ---------------------------------------------------------------- V5
def v5_version(v, feats_unit, tag):
    p = V.PRIV / 'solver' / f'NESTED_SOLVERS_V{v}.npz'
    rp = V.PRIV / 'solver' / f'P1_REPLAY_V{v}.npz'
    rec = V.load_json(V.RES / 'nested' / f'NESTED_SOLVERS_V{v}.json')
    sha = V.sha_file(p)
    with np.load(p, allow_pickle=False) as z:
        arr = {k: z[k].copy() for k in z.files}
    with np.load(rp, allow_pickle=False) as z:
        rep = {k: z[k].copy() for k in z.files}
    unit = np.asarray(arr['unit'])
    sex = np.asarray(arr['sex']).astype(np.int64)
    prob = []
    if sha != rec['per_person_file_sha256']:
        prob.append('npz sha256 differs from the nested record')
    if not np.array_equal(unit, np.asarray(feats_unit)):
        prob.append('households differ from the admitted MODEL_FIT rows')
    if not np.array_equal(sex, np.asarray(rep['sex']).astype(np.int64)) or not np.array_equal(unit, rep['unit']):
        prob.append('SEX / households differ from the P1 replay file (verified in V2 against the admitted labels)')
    ents = entries(arr)
    if set(ents) != set(rec['means']):
        prob.append('entries differ from the nested record')
    # means and AUCs
    mrows, worst = [], 0.0
    for e in ents:
        mine = means_of(arr, e, sex)
        for key, rv in rec['means'][e].items():
            if key not in mine:
                mrows.append(f'{e}|{key}: not recomputable')
                continue
            d = abs(mine[key] - rv)
            worst = max(worst, d)
            if d > TOL:
                mrows.append(f'{e}|{key}: mine {mine[key]} recorded {rv}')
    # spends
    srows, sworst = [], 0.0
    for e, mets in rec['spend_over_twin'].items():
        mine = R2.spends(arr, e, unit, sex)
        for m, (est, se) in mets.items():
            d = max(abs(mine[m][0] - est), abs(mine[m][1] - se))
            sworst = max(sworst, d)
            if d > TOL:
                srows.append(f'{e}|{m}: mine {mine[m]} recorded {[est, se]}')
    # original entries against the replay (exact)
    orig = [TWIN] + [f'{f}@{g}' for f in ('PRC', 'HQ') for g in GRID]
    eq_rows, n_keys, mx = [], 0, 0.0
    for e in orig:
        for k in sorted(x for x in arr if x.startswith(e + '|')):
            if k in rep:
                n_keys += 1
                d = float(np.max(np.abs(np.asarray(arr[k], dtype=np.float64) - np.asarray(rep[k], dtype=np.float64))))
                mx = max(mx, d)
                if d != 0.0:
                    eq_rows.append(f'{k}: max |diff| {d}')
            elif not k.endswith('|fit_pi_pair_mi'):
                eq_rows.append(f'{k}: not in the replay file')
    checks = [
        {'id': f'V5.FILE__V{v}__{tag}', 'block': 'V5', 'kind': 'procedural', 'status': 'PASS' if not prob else 'FAIL',
         'summary': f'nested per-person file of version {v}: {len(ents)} entries, sha256 equal to the record, households equal to the '
                    f'admitted MODEL_FIT rows, SEX equal to the replay file' + (f'; problems {prob}' if prob else ''),
         'detail': {'problems': prob, 'n_entries': len(ents), 'npz_sha256': sha}, 'record': {'entries': ents}, 'script': SCRIPT},
        {'id': f'V5.MEANS_SPENDS__V{v}__{tag}', 'block': 'V5', 'kind': 'numerical', 'status': 'PASS' if not mrows and not srows else 'FAIL',
         'summary': f'own held-out means / LOOKUP AUCs of {len(ents)} entries (max |diff| {worst:.2e}) and spends with clustered SEs of '
                    f'{len(rec["spend_over_twin"])} entries (max |diff| {sworst:.2e}) vs NESTED_SOLVERS_V{v}.json; '
                    f'{len(mrows) + len(srows)} mismatches' + (f': {(mrows + srows)[:4]}' if mrows or srows else ''),
         'detail': {'max_abs_diff_means': worst, 'max_abs_diff_spends': sworst, 'n_mismatch': len(mrows) + len(srows)},
         'record': {'means_mismatches': mrows, 'spend_mismatches': srows}, 'script': SCRIPT},
        {'id': f'V5.ORIGINAL_EQUALS_REPLAY__V{v}__{tag}', 'block': 'V5', 'kind': 'numerical', 'status': 'PASS' if not eq_rows else 'FAIL',
         'summary': f'the nested run\'s original-solver entries ({len(orig)}: twin, PRC and HQ at 6 strengths) against the P1 replay\'s '
                    f'per-person outputs, {n_keys} per-person arrays compared exactly: max |diff| {mx}; {len(eq_rows)} differences',
         'detail': {'n_arrays': n_keys, 'max_abs_diff': mx, 'n_diff': len(eq_rows), 'record_claim': rec.get('original_entries_max_abs_diff_vs_replay')},
         'record': {'differences': eq_rows}, 'script': SCRIPT}]
    return arr, checks, (not eq_rows and not prob)


# ---------------------------------------------------------------- V6 helpers: flags from git, ledger and the access log
def flags_check():
    card = f'results/{V.CAMPAIGN}/cards/FA_FSA_v2.md'
    _, log = V.git('log', '--diff-filter=A', '--format=%cI', '--', card)
    first = V.parse_utc(log.split()[-1]) if log else None
    led = [json.loads(x) for x in (V.COMPUTE / 'ledger.jsonl').read_text().splitlines() if x.strip()]
    starts = {e['unit_id']: V.parse_utc(e['utc']) for e in led if e.get('event') == 'unit_start'
              and str(e['unit_id']).startswith(('FSA_SOURCE_', 'FSA_NESTED_'))}
    acc = [json.loads(x) for x in (V.ACCESS / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
    ms = [a for a in acc if a.get('role') == 'MODEL_SELECT']
    return {'card_committed_before_run': bool(starts) and first is not None and all(t > first for t in starts.values()),
            'model_select_opened': bool(ms), 'card_first_commit_utc': None if first is None else first.isoformat(),
            'unit_starts': {k: t.isoformat() for k, t in starts.items()}, 'model_select_reads': len(ms)}


def main_real(tag):
    C = V.admitted_loaders()
    feats = C.features('MODEL_FIT')
    checks, arrs, eq_ok = [], {}, {}
    for v in (0, 1):
        arr, ch, ok = v5_version(v, feats['unit'], tag)
        arrs[v], eq_ok[v] = arr, ok
        checks += ch
    # ---- V6
    rsi = V.load_json(V.RES / 'decisions' / 'ROUTE_S_INPUT.json')
    adv = V.load_json(V.RES / 'decisions' / 'ADVANCEMENT.json')
    fl = flags_check()
    mine_in = {'revision_id': 'SV1', 'card_committed_before_run': fl['card_committed_before_run'],
               'model_select_opened': fl['model_select_opened'], 'label_derived_statistics_disclosed': True,
               'original_entries_equal_replay': all(eq_ok.values()), 'fitted': {}, 'nested': {}}
    n_bad, nested_rows = [], {}
    for v in (0, 1):
        sel = V.load_json(V.FRC_RES / 'fa' / f'SELECTION_V{v}.json')
        mine, margins, _ = route_s_nested(arrs[v], sel['budgets']['S2'], v)
        fz = V.load_json(V.VER / f'ROUTE_S_FITTED_V{v}.json')
        mine_in['fitted'][str(v)] = fz['fitted']
        nested = {k: x for k, x in mine.items() if k != 'bracket_own'}
        mine_in['nested'][str(v)] = nested
        bad = compare_nested(mine, rsi['nested'][str(v)])
        near = [f'{w}:{c}' for w, mg in margins.items() for c, m in mg.items() if abs(m) <= TOL]
        nested_rows[v] = {'mine': mine, 'margins': margins, 'mismatches': bad, 'near_threshold': near}
        n_bad += [f'V{v} {x}' for x in bad]
    checks.append({'id': f'V6.ROUTE_S_NESTED__{tag}', 'block': 'V6', 'kind': 'numerical', 'status': 'PASS' if not n_bad else 'FAIL',
                   'summary': 'own route S nested inputs (S2 choices of both solvers, total-NLL spends, bracket, paired pair / income / '
                              'occupation LOOKUP AUC differences with household-bootstrap SEs B = 400 seed [2026102200, v, 4], paired '
                              'total-NLL differences with clustered SEs) vs decisions/ROUTE_S_INPUT.json: '
                              + '; '.join(f'V{v}: revised {nested_rows[v]["mine"]["revised_point"]}, original S2 '
                                          f'{nested_rows[v]["mine"]["original_s2"]}, bracket {nested_rows[v]["mine"].get("bracket_own")}'
                                          for v in (0, 1))
                              + f'; {len(n_bad)} mismatches' + (f': {n_bad[:4]}' if n_bad else ''),
                   'detail': {'n_mismatch': len(n_bad), 'near_threshold': {v: r['near_threshold'] for v, r in nested_rows.items()}},
                   'record': nested_rows, 'script': SCRIPT})
    # flags
    fbad = [f'{k}: mine {mine_in[k]} recorded {rsi.get(k)}' for k in ('card_committed_before_run', 'model_select_opened',
                                                                      'original_entries_equal_replay') if mine_in[k] != rsi.get(k)]
    checks.append({'id': f'V6.ROUTE_S_FLAGS__{tag}', 'block': 'V6', 'kind': 'procedural', 'status': 'PASS' if not fbad else 'FAIL',
                   'summary': f'route S validity flags from own sources: card FA_FSA_v2 first committed {fl["card_first_commit_utc"]} before '
                              f'every source / nested unit start; MODEL_SELECT reads in the log {fl["model_select_reads"]}; original '
                              f'entries equal the replay (V5) {mine_in["original_entries_equal_replay"]}; label-derived statistics '
                              'disclosed (record text)' + (f'; mismatches {fbad}' if fbad else ''),
                   'detail': {'mismatches': fbad, **{k: mine_in[k] for k in ('card_committed_before_run', 'model_select_opened',
                                                                              'original_entries_equal_replay')}},
                   'record': fl, 'script': SCRIPT})
    # labels from the frozen rules applied to own inputs
    RU = __import__('fsa.rules', fromlist=['route_s'])
    rs_mine = RU.route_s(copy.deepcopy(mine_in))
    rs_rec = RU.route_s(copy.deepcopy(rsi))
    adv_in = copy.deepcopy(adv['input'])
    adv_in['S'] = {'SV1': copy.deepcopy(mine_in)}
    ad_mine = RU.advancement(adv_in)
    rec_label = adv['result']['label']
    rec_s = adv['result']['labels']['S']['SV1']
    same = rs_mine['label'] == rec_s and ad_mine['label'] == rec_label and rs_rec['label'] == rec_s
    pooled = {k: (rs_mine.get(k), rs_rec.get(k)) for k in ('pooled_d_auc_vs_bracket', 'pooled_d_spend_vs_original_s2',
                                                            'recovery_rationale', 'task_rationale', 'flags', 'improving_strengths_both_versions')}
    pv = {}
    for v in (0, 1):
        a, b = (rs_mine.get('per_version') or {}).get(v), (rs_rec.get('per_version') or {}).get(v)
        pv[v] = {'mine': a, 'recorded_input': b}
    checks.append({'id': f'V6.LABELS__{tag}', 'block': 'V6', 'kind': 'numerical', 'status': 'PASS' if same else 'FAIL',
                   'summary': f'fsa.rules.route_s on own inputs: {rs_mine["label"]} (on the recorded input: {rs_rec["label"]}; recorded '
                              f'{rec_s}); fsa.rules.advancement with own S input (D, E as recorded): {ad_mine["label"]} (recorded '
                              f'{rec_label}); pooled d AUC vs bracket {rs_mine.get("pooled_d_auc_vs_bracket")}, pooled d spend vs '
                              f'original S2 {rs_mine.get("pooled_d_spend_vs_original_s2")}, recovery rationale '
                              f'{rs_mine.get("recovery_rationale")}, task rationale {rs_mine.get("task_rationale")}, flags '
                              f'{rs_mine.get("flags")}',
                   'detail': {'route_s_own': rs_mine['label'], 'route_s_recorded_input': rs_rec['label'], 'route_s_recorded': rec_s,
                              'advancement_own': ad_mine['label'], 'advancement_recorded': rec_label, 'pooled': pooled},
                   'record': {'route_s_own': rs_mine, 'route_s_recorded_input': rs_rec, 'advancement_own': ad_mine, 'per_version': pv,
                              'own_input': mine_in}, 'script': SCRIPT})
    V.write_once_json(V.VER / 'ROUTE_S_VERIFIER_INPUT.json', {'input': mine_in, 'route_s': rs_mine['label'], 'advancement': ad_mine['label'],
                                                              'by': 'independent verifier (v4_objective.py, v5_nested.py)', 'utc': V.utc_now(),
                                                              'unit_id': os.environ.get('FSA_UNIT_ID')})
    V.Checks().append(checks)
    print(json.dumps({c['id']: (c['status'], c['summary'][:600]) for c in checks}, indent=1, default=V._json_default))
    return 0


# ---------------------------------------------------------------- selftest (invented per-person outputs)
def selftest():
    rng = np.random.default_rng(5)
    res = []
    n = 600
    unit = np.sort(rng.integers(0, 230, n)) * 7 + 3
    sex = (rng.random(n) < 0.5).astype(np.int64)
    arr = {'unit': unit, 'sex': sex}
    names = [TWIN] + [f'PRC@{g}' for g in GRID] + [f'PRCsv1@{g}' for g in GRID]
    for j, e in enumerate(names):
        for k in ('inc_nll', 'occ_nll'):
            arr[f'{e}|{k}'] = rng.gamma(2.0, 0.3, n) + 0.0004 * j
        for k in ('inc_brier', 'occ_brier'):
            arr[f'{e}|{k}'] = rng.random(n)
        for k in ('inc_acc', 'occ_acc'):
            arr[f'{e}|{k}'] = (rng.random(n) < 0.6).astype(float)
        cells = rng.integers(0, 9, n)
        for vw in VIEWS:
            arr[f'{e}|look_{vw}'] = (cells + (vw == 'income')) % 9 / 9.0
    budget = {m: 1.0 for m in R2.BUDGET_METRICS}
    mine, margins, _ = route_s_nested(arr, budget, 0)
    res.append(('selects_last_feasible', mine['revised_point'] == 'PRCsv1@8.0' and mine['original_s2'] == 'PRC@8.0', mine['revised_point']))
    # paired AUC SE by brute force (explicit bootstrap loop with the protocol's draw definition)
    g, inv = np.unique(unit, return_inverse=True)
    rngb = np.random.default_rng(np.random.SeedSequence([SEED, 0, 4]))
    Wd = [np.bincount(rngb.integers(0, len(g), len(g)), minlength=len(g))[inv].astype(float) for _ in range(B)]
    a = np.asarray(arr['PRCsv1@8.0|look_pair']); b = np.asarray(arr['PRC@2.0|look_pair'])
    dd = np.array([V._naive_auc(a, sex, w) - V._naive_auc(b, sex, w) for w in Wd[:40]])
    se40 = float(np.std(dd, ddof=1))
    W40 = V.bootstrap_weights(unit, 40, [SEED, 0, 4])
    wa, wb = V.WeightedAUC(a, sex), V.WeightedAUC(b, sex)
    dm = np.array([wa(W40[i]) - wb(W40[i]) for i in range(40)])
    res.append(('paired_bootstrap_vs_bruteforce', abs(float(np.std(dm, ddof=1)) - se40) < 1e-14, (float(np.std(dm, ddof=1)), se40)))
    res.append(('paired_point_difference', abs(mine['paired']['PRC@2.0']['d_auc'][0] - (V._naive_auc(a, sex) - V._naive_auc(b, sex))) < 1e-14, None))
    tot = lambda e: arr[f'{e}|inc_nll'] + arr[f'{e}|occ_nll']           # noqa: E731
    dsp = tot('PRCsv1@8.0') - tot('PRC@1.0')
    res.append(('paired_spend', abs(mine['paired']['PRC@1.0']['d_spend'][0] - dsp.mean()) < 1e-15
                and abs(mine['paired']['PRC@1.0']['d_spend'][1] - V._naive_cluster_se(dsp, unit)) < 1e-14, None))
    # bracket rule
    path = [{'id': 'a', 'spend': 0.001}, {'id': 'b', 'spend': 0.002}, {'id': 'c', 'spend': 0.003}]
    res.append(('bracket_rule', bracket(path, 0.0024) == 'b' and bracket(path, 0.0026) == 'c' and bracket(path, 0.01) == 'c'
                and bracket(path, 0.0) == 'a', None))
    # comparison flags a planted difference
    rec = copy.deepcopy({k: x for k, x in mine.items() if k != 'bracket_own'})
    res.append(('compare_clean', compare_nested(mine, rec) == [], None))
    rec['paired']['PRC@4.0']['d_auc_income'][1] += 1e-8
    res.append(('compare_planted', len(compare_nested(mine, rec)) == 1, None))
    # tight budget: no revised point
    m2, _, _ = route_s_nested(arr, {m: -1.0 for m in R2.BUDGET_METRICS}, 0)
    res.append(('no_feasible_point', m2['revised_point'] is None and m2['paired'] == {}, None))
    n_ok = sum(r[1] for r in res)
    for name, ok, info in res:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  [{info}]'))
    print(f'v5_nested selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tag', default='V1')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not os.environ.get('FSA_UNIT_ID'):
        sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
    sys.exit(main_real(a.tag))

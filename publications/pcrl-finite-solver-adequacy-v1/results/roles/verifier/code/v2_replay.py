"""V2: independent recomputation of the P1 replay (campaign `pcrl_finite_solver_adequacy_v1`; the decisive check; RULES.md 8).

The coordinator's unit FSA_REPLAY_V<v>_V1 reran the previous campaign's nested P1 estimator and saved per-person held-out outputs
(.private/<campaign>/solver/P1_REPLAY_V<v>.npz: 'unit', 'sex', 'income', 'occupation', '<entry>|<metric>') and its summary
replay/P1_REPLAY_V<v>.json in the structure of the recorded P1 record results/pcrl_finite_routing_comparison_v1/fa/
SELECTION_V<v>.json. From the per-person file alone, with the verifier's own code (vcore; no fsa / frc / nrm numerical function),
this script recomputes, and compares with BOTH the replay record and the recorded P1 record:
  (a) MEANS      every entry's held-out means (task metrics, men's / women's occupation Brier, pair / income / occupation LOOKUP AUC)
                 (keys that need training-side partitions are listed as not recomputable);
  (b) SPEND      each candidate's spend over PRC_TWIN per budget metric: paired per-person differences, accuracy as a loss, men's /
                 women's occupation Brier as ratio-estimator subgroup means, household-clustered SEs (audit_spend_over_twin);
  (c) SELECTION  per family and budget: the LAST candidate in path order such that it and every earlier one keep
                 estimate + 2 SE <= budget on every metric; decision margins and boundary ties (RULES 8) reported;
  (d) BRACKETS   the first entry of HQ / SHUF / ORD with nested total-NLL spend >= that of the PRC S2 entry;
  (e) LOOKUP/F4  every entry's pair LOOKUP AUC, its difference from the twin and the household-bootstrap SE (B = 400,
                 default_rng(SeedSequence([2026102100, v, 4])), the same household weights for both, AUC weighted by the bootstrap
                 household counts); F4 = the PRC S2 entry's difference, pass = diff - 2 SE > 0.
Also: FILE (hashes, keys, households against the admitted MODEL_FIT rows; labels against the admitted labels with --labels),
CONSISTENCY (per-person identities), REPRODUCTION (the replay record against the recorded P1 record within the RULES reproduction
tolerance, every discrete output equal; own comparison code), ESTIMATOR (RULES 8 estimator identity), ORIGINAL_CODE (the nested_c
call path of the code the original P1 unit ran against the replay's code; disclosure) and PROVENANCE (ledger, interpreter).
The recomputation is written in the P1 structure (audit_spend_over_twin, selection, F4, nested_lookup_pair, brackets) to
roles/verifier/P1_VERIFIER_V<v>.json for fsa.rules.p1_replay.

    python3 .../v2_replay.py --selftest                        # invented data only
    python3 .../v2_replay.py --version 0 --tag V1 [--labels]   # real (queue unit, stage verify, after the receipt)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/v2_replay.py'
TASK = ('inc_nll', 'occ_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc')
LOOK = ('look_pair', 'look_income', 'look_occupation')
PER_PERSON = TASK + LOOK
NOT_RECOMPUTABLE = ('fit_auc_pair_insample', 'fit_mi_pair_labels_insample', 'fit_mi_pair_pi_fitted',
                    'divergence_auc_pair_insample_minus_heldout', 'mean_fit_rerouted_vs_twin', 'mean_heldout_rerouted_vs_twin')
BUDGET_METRICS = ('occ_nll', 'inc_nll', 'occ_acc', 'inc_acc', 'occ_brier_men', 'occ_brier_women')
TEXT_KEYS = ('rule', 'bootstrap', 'verdict_note', 'note')
TWIN = 'PRC_TWIN'
K_SE = 2.0
B_BOOT = 400
SEED_BASE = 2026102100


def plan_order(sel):
    """Entries in path order: twin, PRC / HQ / SHUF over the lambda grid, ORD over the ladder (as the record lists them)."""
    out = [TWIN]
    for fam in ('PRC', 'HQ', 'SHUF'):
        out += [f'{fam}@{float(g)}' for g in sel['lambda_grid']]
    out += [f'ORD@{int(a)}x{int(b)}' for a, b in sel['ladder']]
    return out


def family_path(order, fam):
    return [e for e in order if e.startswith(fam + '@')]


def spends(arr, e, unit, sex):
    """Spend of entry e over the twin: {metric: [estimate, clustered SE]}; accuracy as a loss; subgroup Brier as ratio means."""
    a = lambda m: np.asarray(arr[f'{e}|{m}'], dtype=np.float64)          # noqa: E731
    t = lambda m: np.asarray(arr[f'{TWIN}|{m}'], dtype=np.float64)       # noqa: E731
    men, women = sex == 0, sex == 1
    out = {'occ_nll': V.mean_se(a('occ_nll') - t('occ_nll'), unit),
           'inc_nll': V.mean_se(a('inc_nll') - t('inc_nll'), unit),
           'occ_acc': V.mean_se(t('occ_acc') - a('occ_acc'), unit),
           'inc_acc': V.mean_se(t('inc_acc') - a('inc_acc'), unit),
           'occ_brier_men': V.subgroup_mean_se(a('occ_brier') - t('occ_brier'), men, unit),
           'occ_brier_women': V.subgroup_mean_se(a('occ_brier') - t('occ_brier'), women, unit)}
    return {k: [float(x), float(y)] for k, (x, y) in out.items()}


def total_nll_spend(arr, e):
    tot = lambda x: np.asarray(arr[f'{x}|inc_nll'], dtype=np.float64) + np.asarray(arr[f'{x}|occ_nll'], dtype=np.float64)  # noqa
    return float(np.mean(tot(e) - tot(TWIN)))


def select(path, sp, budget, k=K_SE):
    """Last candidate in path order such that it and every earlier one keep est + k SE <= budget on every budget metric."""
    pick, margins, stopped = None, {}, False
    for c in path:
        slack = {m: sp[c][m][0] + k * sp[c][m][1] - budget[m] for m in budget}
        margins[c] = max(slack.values())
        if not stopped and all(x <= 0 for x in slack.values()):
            pick = c
        else:
            stopped = True
    return pick, margins


# ---------------------------------------------------------------- own flatten / compare (RULES 8 semantics, own code)
def flat(x, pre=''):
    out = {}
    if isinstance(x, dict):
        for k in x:
            if str(k) in TEXT_KEYS:
                continue
            out.update(flat(x[k], f'{pre}/{k}'))
    elif isinstance(x, (list, tuple)):
        for i, y in enumerate(x):
            out.update(flat(y, f'{pre}[{i}]'))
    else:
        out[pre] = x
    return out


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(ref, mine, fields, tol):
    """Leaves of `mine` against `ref` over the given top-level fields: numbers within tol (NaN = NaN), others equal, none missing."""
    mism, worst, n = [], 0.0, 0
    for f in fields:
        fr, fm = flat(ref.get(f), f'/{f}'), flat(mine.get(f), f'/{f}')
        for k in sorted(set(fr) | set(fm)):
            if k not in fr or k not in fm:
                mism.append({'path': k, 'kind': 'MISSING', 'in_ref': k in fr, 'in_mine': k in fm})
                continue
            x, y = fr[k], fm[k]
            n += 1
            if _isnum(x) and _isnum(y):
                if math.isnan(x) and math.isnan(y):
                    continue
                d = abs(float(x) - float(y))
                worst = max(worst, d) if math.isfinite(d) else worst
                if not d <= tol:
                    mism.append({'path': k, 'kind': 'NUMERIC', 'ref': x, 'mine': y, 'abs_diff': d})
            elif x != y:
                mism.append({'path': k, 'kind': 'DISCRETE', 'ref': x, 'mine': y})
    return mism, worst, n


def recompute(arr, sel, version, frozen_selection=None, tol=1e-9):
    """The P1 fields from the per-person file (own code): nested means (recomputable keys), spends, selection, brackets, LOOKUP, F4.
    If my PRC S2 choice differs from the frozen one only by a boundary tie (RULES 8: the frozen code's value stands), the brackets
    and F4 are computed from the frozen S2 entry."""
    v = int(version)
    order = plan_order(sel)
    unit = np.asarray(arr['unit'])
    sex = np.asarray(arr['sex']).astype(np.int64)
    means = {}
    for e in order:
        g = lambda m: np.asarray(arr[f'{e}|{m}'], dtype=np.float64)      # noqa: E731
        d = {k: float(g(k).mean()) for k in TASK}
        d['occ_brier_men'] = float(g('occ_brier')[sex == 0].mean())
        d['occ_brier_women'] = float(g('occ_brier')[sex == 1].mean())
        for vw in ('pair', 'income', 'occupation'):
            d[f'auc_{vw}'] = V.auc(g(f'look_{vw}'), sex)
        means[e] = d
    sp = {e: spends(arr, e, unit, sex) for e in order if e != TWIN}
    audit = {fam: {c: sp[c] for c in family_path(order, fam)} for fam in ('PRC', 'HQ', 'SHUF', 'ORD')}
    selection, margins = {}, {}
    for fam, rec_choice in sel['selection'].items():                  # the families and budgets the record assesses
        selection[fam], margins[fam] = {}, {}
        for b in rec_choice:
            selection[fam][b], margins[fam][b] = select(family_path(order, fam), sp, sel['budgets'][b])
    s2 = selection.get('PRC', {}).get('S2')
    s2_source = 'own selection'
    if frozen_selection is not None:
        fz = (frozen_selection.get('PRC') or {}).get('S2')
        if fz != s2 and any(abs(m) <= tol for m in margins.get('PRC', {}).get('S2', {}).values()):
            s2, s2_source = fz, 'frozen selection (own S2 choice differs only by a boundary tie)'
    br = {'prc_s2': s2, 'target_total_nll_spend': None, 'brackets': {}, 'spend': {}}
    if s2 is None:
        br['brackets'] = {f: 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH' for f in ('HQ', 'SHUF', 'ORD')}
    else:
        target = total_nll_spend(arr, s2)
        br['target_total_nll_spend'] = target
        for fam in ('HQ', 'SHUF', 'ORD'):
            path = family_path(order, fam)
            spd = {c: total_nll_spend(arr, c) for c in path}
            br['spend'][fam] = spd
            br['brackets'][fam] = next((c for c in path if spd[c] >= target), 'CONTROL_DID_NOT_REACH_SPEND')
    W = V.bootstrap_weights(unit, B_BOOT, [SEED_BASE, v, 4])
    ref = np.asarray(arr[f'{TWIN}|look_pair'], dtype=np.float64)
    wref = V.WeightedAUC(ref, sex)
    bref = np.array([wref(W[b]) for b in range(B_BOOT)])
    a_ref = V.auc(ref, sex)
    look = {TWIN: {'auc': a_ref}}
    for e in order[1:]:
        s = np.asarray(arr[f'{e}|look_pair'], dtype=np.float64)
        we = V.WeightedAUC(s, sex)
        be = np.array([we(W[b]) for b in range(B_BOOT)])
        a = V.auc(s, sex)
        look[e] = {'auc': a, 'diff_twin_minus_entry': a_ref - a, 'se_boot': float(np.std(bref - be, ddof=1))}
    f4 = None
    if s2 is not None:
        r = look[s2]
        f4 = {'entry': s2, 'diff_twin_minus_prc': r['diff_twin_minus_entry'], 'se_boot': r['se_boot'],
              'pass_2se': bool(r['diff_twin_minus_entry'] - K_SE * r['se_boot'] > 0)}
    return {'nested_means': means, 'audit_spend_over_twin': audit, 'selection': selection, 'brackets': br,
            'nested_lookup_pair': look, 'F4': f4, '_margins': margins, '_s2_source': s2_source,
            '_bootstrap': {'twin_auc_mean': float(bref.mean()), 'twin_auc_sd': float(bref.std(ddof=1))}}


def _ties(mine, tol):
    """Recomputed quantities within tol of a decision threshold (RULES 8): est + 2 SE vs budget, diff - 2 SE vs 0, spend vs target."""
    out = []
    for fam, bs in mine['_margins'].items():
        for b, mg in bs.items():
            out += [f'/selection/{fam}/{b}: {c} slack {m:.3e}' for c, m in mg.items() if abs(m) <= tol]
    f = mine['F4']
    if f is not None and abs(f['diff_twin_minus_prc'] - K_SE * f['se_boot']) <= tol:
        out.append('/F4/pass_2se')
    br = mine['brackets']
    if br['target_total_nll_spend'] is not None:
        for fam, spd in br['spend'].items():
            out += [f'/brackets/brackets/{fam}: {c}' for c, s in spd.items() if abs(s - br['target_total_nll_spend']) <= tol]
    return out


def _tie_for(path, ties):
    if path.startswith('/selection/'):
        fam_b = '/'.join(path.split('/')[:4])
        return any(t.startswith(fam_b + ':') for t in ties)
    if path.startswith('/brackets/brackets/'):
        return any(t.startswith(path + ':') for t in ties)
    if path == '/F4/pass_2se':
        return '/F4/pass_2se' in ties
    return False


def verify(arr, sel, rep, version, tol, rtol=1e-12, unit_ref=None, labels_ref=None, npz_sha=None):
    """Pure recomputation and comparison. Returns (checks, mine): checks are dicts for vcore.Checks.append."""
    v = int(version)
    order = plan_order(sel)
    out = []
    unit = np.asarray(arr['unit'])
    sex = np.asarray(arr['sex']).astype(np.int64)
    n = len(unit)
    # ---------------- FILE
    fp = []
    if set(order) != set(sel['nested_means']):
        fp.append('plan order differs from the recorded nested_means entries')
    want = {'unit', 'sex', 'income', 'occupation'} | {f'{e}|{m}' for e in order for m in PER_PERSON}
    have = set(arr)
    if want - have:
        fp.append(f'missing keys: {sorted(want - have)[:6]}')
    bad_len = [k for k in have if np.asarray(arr[k]).shape[0] != n]
    if bad_len:
        fp.append(f'length differs: {bad_len[:4]}')
    if rep is None:
        fp.append('no replay record')
    else:
        if npz_sha is not None and rep.get('per_person_file_sha256') != npz_sha:
            fp.append('npz sha256 differs from the replay record')
        if sorted(rep.get('per_person_keys', [])) != sorted(have):
            fp.append('npz keys differ from the replay record per_person_keys')
        for k in ('T_fixed', 'budgets', 'ladder', 'capacity', 'construction', 'outer_folds'):
            if k in rep and rep.get(k) != sel.get(k):
                fp.append(f'replay {k} differs from the recorded {k}')
        if int(rep.get('version', -1)) != v:
            fp.append('replay version differs')
        if set(rep.get('nested_means', {})) != set(sel['nested_means']):
            fp.append('replay plan entries differ from the recorded plan')
    if not set(np.unique(sex)) <= {0, 1}:
        fp.append('sex not in {0,1}')
    if not set(np.unique(arr['income'])) <= {0, 1} or not set(np.unique(arr['occupation'])) <= set(range(5)):
        fp.append('task labels out of range')
    unit_equal = None if unit_ref is None else bool(np.array_equal(unit, np.asarray(unit_ref)))
    if unit_equal is False:
        fp.append('npz unit differs from the admitted MODEL_FIT households (row order)')
    lab_equal = None
    if labels_ref is not None:
        lab_equal = {k: bool(np.array_equal(np.asarray(arr[k]).astype(np.int64), np.asarray(labels_ref[k]).astype(np.int64)))
                     for k in ('sex', 'income', 'occupation')}
        if not all(lab_equal.values()):
            fp.append(f'npz labels differ from the admitted labels: {lab_equal}')
    out.append({'id': f'V2.FILE__V{v}', 'block': 'V2', 'kind': 'procedural', 'status': 'PASS' if not fp else 'FAIL',
                'summary': f'per-person file: {len(have)} keys, n={n}, {len(np.unique(unit))} households; unit equal to admitted '
                           f'MODEL_FIT: {unit_equal}; labels equal: {lab_equal}' + (f'; problems {fp}' if fp else ''),
                'detail': {'n': n, 'households': int(len(np.unique(unit))), 'unit_equal_admitted': unit_equal,
                           'labels_equal_admitted': lab_equal, 'extra_keys': sorted(have - want), 'npz_sha256': npz_sha},
                'record': {'problems': fp, 'keys': sorted(have), 'plan_order': order}})
    if want - have or bad_len or rep is None:
        return out, None
    # ---------------- CONSISTENCY (per-person identities)
    cons, good = {}, {}
    for e in order:
        g = lambda m: np.asarray(arr[f'{e}|{m}'], dtype=np.float64)      # noqa: E731
        p_inc, p_occ = np.exp(-g('inc_nll')), np.exp(-g('occ_nll'))
        a_inc, a_occ = 1.0 - p_inc, 1.0 - p_occ
        r = {'finite': bool(all(np.isfinite(g(m)).all() for m in PER_PERSON)),
             'acc_binary': bool(set(np.unique(g('inc_acc'))) <= {0.0, 1.0} and set(np.unique(g('occ_acc'))) <= {0.0, 1.0}),
             'nll_range': bool((g('inc_nll') >= 0).all() and (g('occ_nll') >= 0).all()
                               and max(g('inc_nll').max(), g('occ_nll').max()) < 14.0)}       # releases clip at 1e-6
        r['inc_brier_identity_max_abs'] = float(np.max(np.abs(g('inc_brier') - 2.0 * a_inc ** 2)))
        clear = np.abs(p_inc - 0.5) > 1e-9
        r['inc_acc_vs_py_mismatch'] = int(np.sum((g('inc_acc')[clear] == 1.0) != (p_inc[clear] > 0.5)))
        lo, hi = 1.25 * a_occ ** 2 - 1e-12, 2.0 * a_occ ** 2 + 1e-12
        r['occ_brier_bounds_violations'] = int(np.sum((g('occ_brier') < lo) | (g('occ_brier') > hi)))
        r['occ_acc_vs_py_violations'] = int(np.sum((g('occ_acc') == 1.0) & (p_occ < 0.2 - 1e-12))
                                            + np.sum((g('occ_acc') == 0.0) & (p_occ > 0.5 + 1e-12)))
        r['look_range'] = bool(all(((g(m) >= 0) & (g(m) <= 1)).all() for m in LOOK))
        r['look_pair_distinct'] = int(len(np.unique(g('look_pair'))))
        cons[e] = r
        good[e] = (r['finite'] and r['acc_binary'] and r['nll_range'] and r['inc_brier_identity_max_abs'] < 1e-12
                   and r['inc_acc_vs_py_mismatch'] == 0 and r['occ_brier_bounds_violations'] == 0
                   and r['occ_acc_vs_py_violations'] == 0 and r['look_range'] and r['look_pair_distinct'] <= 5 * 1024 + 5)
    bad = [e for e in order if not good[e]]
    out.append({'id': f'V2.CONSISTENCY__V{v}', 'block': 'V2', 'kind': 'numerical', 'status': 'PASS' if not bad else 'FAIL',
                'summary': f'per-person identities on {len(order)} entries: binary Brier = 2(1-p_y)^2 (max dev '
                           f'{max(c["inc_brier_identity_max_abs"] for c in cons.values()):.2e}), accuracy against p_y, occupation '
                           f'Brier bounds, LOOKUP in [0,1] with at most {max(c["look_pair_distinct"] for c in cons.values())} '
                           f'distinct pair scores; failing entries {bad}',
                'detail': {'failing': bad}, 'record': cons})
    # ---------------- recomputation and comparisons
    mine = recompute(arr, sel, v, frozen_selection=rep.get('selection'), tol=tol)
    ties = _ties(mine, tol)
    refs = (('replay', rep), ('recorded', sel))
    # (a) MEANS: recomputable keys only, against both records
    m_rows, na = {}, set()
    for name, ref in refs:
        rows, worst, nn = [], 0.0, 0
        for e in order:
            for key, rv in ref['nested_means'][e].items():
                if key in NOT_RECOMPUTABLE:
                    na.add(key)
                    continue
                if key not in mine['nested_means'][e]:
                    rows.append({'path': f'{e}|{key}', 'kind': 'UNKNOWN_KEY'})
                    continue
                d = abs(mine['nested_means'][e][key] - float(rv))
                worst, nn = max(worst, d), nn + 1
                if not d <= tol:
                    rows.append({'path': f'{e}|{key}', 'kind': 'NUMERIC', 'ref': rv, 'mine': mine['nested_means'][e][key], 'abs_diff': d})
        m_rows[name] = {'mismatches': rows, 'max_abs_diff': worst, 'n_compared': nn}
    st = 'PASS' if not any(m_rows[r]['mismatches'] for r in m_rows) else 'FAIL'
    out.append({'id': f'V2.MEANS__V{v}', 'block': 'V2', 'kind': 'numerical', 'status': st,
                'summary': '(a) recomputed nested means: ' + '; '.join(f'vs {r} {m_rows[r]["n_compared"]} values, max |diff| '
                                                                      f'{m_rows[r]["max_abs_diff"]:.3e}, {len(m_rows[r]["mismatches"])} outside tol {tol:g}'
                                                                      for r in m_rows)
                           + f'; not recomputable from held-out outputs: {sorted(na)}',
                'detail': {r: {k: m_rows[r][k] for k in ('max_abs_diff', 'n_compared')} | {'n_mismatch': len(m_rows[r]['mismatches'])}
                           for r in m_rows} | {'not_recomputable': sorted(na)},
                'record': {'comparisons': m_rows, 'mine': mine['nested_means']}})
    # (b)-(e): one check per field, against both records
    for cid, field, what in (('SPEND', 'audit_spend_over_twin', '(b) spends over the twin with household-clustered SEs'),
                             ('SELECTION', 'selection', '(c) predeclared selection'),
                             ('BRACKETS', 'brackets', '(d) spend brackets of HQ / SHUF / ORD'),
                             ('LOOKUP', 'nested_lookup_pair', '(e) pair LOOKUP AUCs, twin differences and bootstrap SEs'),
                             ('F4', 'F4', '(e) F4')):
        comp, real, tied = {}, [], []
        for name, ref in refs:
            mm, worst, nn = compare(ref, mine, [field], tol)
            t = [m for m in mm if m['kind'] == 'DISCRETE' and _tie_for(m['path'], ties)]
            comp[name] = {'n_leaves': nn, 'max_abs_diff': worst, 'mismatches': [m for m in mm if m not in t], 'boundary_ties': t}
            real += [dict(m, ref_record=name) for m in mm if m not in t]
            tied += [dict(m, ref_record=name) for m in t]
        near = [t for t in ties if t.startswith(f'/{field}')]
        st = 'FAIL' if real else ('ATTENTION' if tied or near else 'PASS')
        extra = ''
        if field == 'selection':
            extra = f' mine {mine["selection"]};'
        elif field == 'F4':
            extra = f' mine {mine["F4"]};'
        elif field == 'brackets':
            extra = f' mine {mine["brackets"]["brackets"]} (PRC S2 {mine["brackets"]["prc_s2"]}, target {mine["brackets"]["target_total_nll_spend"]});'
        out.append({'id': f'V2.{cid}__V{v}', 'block': 'V2', 'kind': 'numerical', 'status': st,
                    'summary': f'{what}:{extra} ' + '; '.join(f'vs {r} {comp[r]["n_leaves"]} leaves, max |diff| {comp[r]["max_abs_diff"]:.3e}, '
                                                               f'{len(comp[r]["mismatches"])} mismatches, {len(comp[r]["boundary_ties"])} '
                                                               'boundary ties' for r in comp)
                               + (f'; within tol of a threshold: {near}' if near else ''),
                    'detail': {r: {'max_abs_diff': comp[r]['max_abs_diff'], 'n_leaves': comp[r]['n_leaves'],
                                   'n_mismatch': len(comp[r]['mismatches']), 'n_boundary_ties': len(comp[r]['boundary_ties'])}
                               for r in comp} | {'near_threshold': near},
                    'record': {'comparisons': comp, 'mine': mine[field],
                               **({'margins_max_slack': mine['_margins'], 'budgets': sel['budgets']} if field == 'selection' else {}),
                               **({'bootstrap': mine['_bootstrap']} if field in ('LOOKUP', 'F4', 'nested_lookup_pair') else {})}})
    # ---------------- REPRODUCTION: the replay record against the recorded P1 record (RULES 8 reproduction; own comparison)
    fields = ('nested_means', 'audit_spend_over_twin', 'brackets', 'nested_lookup_pair', 'F4', 'selection')
    rm, rw, rn = compare(sel, rep, [f for f in fields if f in sel], rtol)
    claims = {k: rep.get(k) for k in ('max_abs_diff_recorded_nested_means', 'selection_equal_recorded', 'brackets_equal_recorded',
                                       'estimator_unchanged', 'per_person_outputs_saved')}
    out.append({'id': f'V2.REPRODUCTION__V{v}', 'block': 'V2', 'kind': 'numerical', 'status': 'PASS' if not rm else 'FAIL',
                'summary': f'replay record vs recorded SELECTION_V{v} over the six P1 fields (own comparison; numbers within {rtol:g}, '
                           f'every discrete output equal, text fields skipped): {rn} leaves, max |diff| {rw:.3e}, {len(rm)} mismatches; '
                           f'replay claims {claims}',
                'detail': {'n_leaves': rn, 'max_abs_diff': rw, 'n_mismatch': len(rm), 'replay_claims': claims},
                'record': {'mismatches': rm}})
    # ---------------- aliases (descriptive)
    alias = {}
    for e in order:
        sig = V.sha_bytes(b''.join(np.ascontiguousarray(np.asarray(arr[f'{e}|{m}'], dtype=np.float64)).tobytes() for m in PER_PERSON))
        alias.setdefault(sig, []).append(e)
    mine['_alias_groups'] = [g for g in alias.values() if len(g) > 1]
    return out, mine


def p1_verifier_record(mine, version):
    """The recomputation in the P1 structure for fsa.rules.p1_replay (fields fully recomputable from the per-person file)."""
    return {'version': int(version), 'audit_spend_over_twin': mine['audit_spend_over_twin'], 'selection': mine['selection'],
            'F4': mine['F4'], 'nested_lookup_pair': mine['nested_lookup_pair'], 'brackets': mine['brackets']}


# ---------------------------------------------------------------- code identity and provenance (real run only)
NESTED_ROOT = 'nested_c'
RULES_P1_FUNCS = ('nested_c', 'fit_pipeline_c', 'chain_inputs', 'tables', 'apply_c', 'emit', 'route_any', 'entry_embedding',
                  'shuf_embedding', '_plan', 'total_spend', 'brackets_for', '_wauc_draws', 'nested_lookup')


def _module_functions(src):
    tree = ast.parse(src)
    funcs, assigns = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigns[t.id] = node
    return funcs, assigns


def _reachable(funcs, assigns, root):
    seen, names, stack = set(), set(), [root]
    while stack:
        f = stack.pop()
        if f in seen or f not in funcs:
            continue
        seen.add(f)
        for node in ast.walk(funcs[f]):
            if isinstance(node, ast.Name):
                if node.id in funcs and node.id not in seen:
                    stack.append(node.id)
                elif node.id in assigns:
                    names.add(node.id)
    return seen, names


def _dump(node):
    return ast.dump(node, include_attributes=False)


def call_path_diff(old_src, new_src, root=NESTED_ROOT):
    fo, ao = _module_functions(old_src)
    fn, an = _module_functions(new_src)
    so, no = _reachable(fo, ao, root)
    sn, nn = _reachable(fn, an, root)
    changed = sorted(f for f in so & sn if _dump(fo[f]) != _dump(fn[f]))
    const_changed = sorted(c for c in no & nn if _dump(ao[c]) != _dump(an[c]))
    return {'functions_old': sorted(so), 'functions_new': sorted(sn), 'only_old': sorted(so - sn), 'only_new': sorted(sn - so),
            'changed': changed, 'constants_changed': const_changed, 'constants_only_new': sorted(nn - no),
            'constants_only_old': sorted(no - nn),
            'source_changed': {f: {'old': ast.get_source_segment(old_src, fo[f]), 'new': ast.get_source_segment(new_src, fn[f])}
                               for f in changed},
            'source_only_new': {f: ast.get_source_segment(new_src, fn[f]) for f in sorted(sn - so)}}


class _Rename(ast.NodeTransformer):
    def __init__(self, m):
        self.m = m

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.m.get(node.id, node.id), ctx=node.ctx), node)


REPLAY_RENAMES = {'sel2': 'sel', 'k2': 'k', 'v2': 'v'}     # the replay's result dict and comprehension variables (local names only)


def _selection_loop(func_node, renames):
    """The `for fam, bud in (...)` selection loop of a stage function, with the given local names renamed."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and [getattr(e, 'id', None) for e in node.target.elts] == ['fam', 'bud']:
            return _dump(_Rename(dict(renames)).visit(ast.parse(ast.unparse(node)).body[0]))
    return None


def _ledger(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def _replay_unit(v):
    es = _ledger(V.COMPUTE / 'ledger.jsonl')
    st = [e for e in es if e.get('event') == 'unit_start' and e.get('argv', [])[1:4] == ['-m', 'fsa.fa.solve', 'replay']
          and e['argv'][e['argv'].index('--version') + 1] == str(v)]
    return es, st


def code_checks(version, rules, rep):
    """ESTIMATOR (RULES 8 identity), ORIGINAL_CODE (disclosure) and PROVENANCE checks for the replay unit of this version."""
    v = int(version)
    es, starts = _replay_unit(v)
    out = []
    pins = rules['p1_replay']['estimator_sha256']
    prob, notes, rec = [], [], {'replay_units': [s['unit_id'] for s in starts]}
    if len(starts) != 1:
        prob.append(f'{len(starts)} replay unit_start lines for version {v}')
    cs = None
    if starts:
        st = starts[-1]
        uid = st['unit_id']
        d = V.COMPUTE / 'units' / uid
        cs = V.load_json(d / 'CODE_AT_LAUNCH.json') if (d / 'CODE_AT_LAUNCH.json').is_file() else V.load_json(d / 'CODE_AT_START.json')
        rec['estimator_files_at_launch'] = {f: cs['files'].get(f) for f in pins}
        for f, h in pins.items():
            if cs['files'].get(f) != h:
                prob.append(f'{f} at launch differs from the RULES pin')
        ends = [e for e in es if e.get('event') == 'unit_end' and e['unit_id'] == uid]
        changed = (ends[-1].get('code_changed_during_run') or []) if ends else []
        rec['code_changed_during_run'] = changed
        if any(f in changed for f in pins) or any(f.startswith(('nrm/', 'frc/', 'fsa/fa/solve.py')) for f in changed):
            prob.append(f'estimator code changed during the run: {changed}')
        route_now = V.git_blob(st['git_head'], 'frc/fa/route.py')
        route_src = (route_now if route_now is not None and V.sha_bytes(route_now) == pins['frc/fa/route.py']
                     else (V.ROOT / 'frc' / 'fa' / 'route.py').read_bytes())
        rec['route_source_sha256'] = V.sha_bytes(route_src)
        if V.sha_bytes(route_src) != pins['frc/fa/route.py']:
            prob.append('no copy of frc/fa/route.py with the pinned hash is available')
        fr, _ = _module_functions(route_src.decode())
        missing = [f for f in RULES_P1_FUNCS if f not in fr]
        if missing:
            prob.append(f'P1 functions missing in the pinned route.py: {missing}')
        seeds = [n.value for n in ast.walk(fr['nested_lookup']) if isinstance(n, ast.Constant) and n.value == SEED_BASE] if 'nested_lookup' in fr else []
        rec['bootstrap_seed_literal_in_nested_lookup'] = bool(seeds)
        if not seeds:
            prob.append('nested_lookup does not carry the seed 2026102100')
        # the selection loop of stage_paths against the replay's loop (fsa/fa/solve.py as launched)
        solve_h = cs['files'].get('fsa/fa/solve.py')
        solve_b = V.git_blob(st['git_head'], 'fsa/fa/solve.py')
        if solve_b is None or V.sha_bytes(solve_b) != solve_h:
            cur = (V.ROOT / 'fsa' / 'fa' / 'solve.py').read_bytes()
            solve_b = cur if V.sha_bytes(cur) == solve_h else None
        rec['solve_py_at_launch_sha256'] = solve_h
        rec['solve_py_source_found'] = solve_b is not None
        if solve_b is None:
            prob.append('the launched fsa/fa/solve.py is not recoverable (neither HEAD blob nor working copy hashes to it)')
        else:
            fs, _ = _module_functions(solve_b.decode())
            a_loop = _selection_loop(fr['stage_paths'], {}) if 'stage_paths' in fr else None
            b_loop = _selection_loop(fs['stage_replay'], REPLAY_RENAMES) if 'stage_replay' in fs else None
            rec['selection_loop_identical_up_to_local_renames'] = a_loop is not None and a_loop == b_loop
            rec['selection_loop_renames'] = REPLAY_RENAMES
            rec['selection_loop_textually_identical'] = (a_loop is not None and a_loop == _selection_loop(fs['stage_replay'], {})
                                                         if 'stage_replay' in fs else False)
            if not rec['selection_loop_identical_up_to_local_renames']:
                prob.append('the replay selection loop differs from the selection loop of stage_paths')
            calls = sorted({f'{n.func.value.id}.{n.func.attr}' for n in ast.walk(fs['stage_replay']) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in ('FR', 'P')})
            rec['replay_calls_into_frc_nrm'] = calls
        rec['replay_record_inputs'] = {k: (rep.get(k) if rep else None) for k in ('construction', 'capacity', 'outer_folds', 'ladder', 'T_fixed')}
        rec['replay_record_identity'] = {k: (rep.get(k) if rep else None) for k in ('estimator_identity', 'estimator_unchanged',
                                                                                  'per_person_outputs_saved')}
        if rep and (rep.get('estimator_unchanged') is not True or rep.get('per_person_outputs_saved') is not True):
            prob.append('the replay record does not state estimator_unchanged and per_person_outputs_saved')
    out.append({'id': f'V2.ESTIMATOR__V{v}', 'block': 'V2', 'kind': 'procedural', 'status': 'FAIL' if prob else 'PASS',
                'summary': 'RULES 8 estimator identity at the replay unit\'s launch: nrm/fa/prc.py and frc/fa/route.py hashes against '
                           'the pins (so every P1 function is textually the pinned one), the seed literal in nested_lookup, the replay\'s '
                           'selection loop against stage_paths\' (identical up to the local names sel2, k2, v2), the replay inputs'
                           + (f'; problems {prob}' if prob else ''),
                'detail': {'problems': prob, 'selection_loop_identical_up_to_local_renames': rec.get('selection_loop_identical_up_to_local_renames'),
                           'selection_loop_textually_identical': rec.get('selection_loop_textually_identical')},
                'record': rec})
    # ORIGINAL_CODE: what the original P1 unit (FRC_PATHS_V<v>_V1) ran against the replay's code
    prob2, notes2, rec2 = [], [], {}
    fes = _ledger(V.FRC_PRIV / 'compute' / 'ledger.jsonl')
    orig = [e for e in fes if e.get('event') == 'unit_start' and e.get('unit_id') == f'FRC_PATHS_V{v}_V1']
    if not orig or cs is None:
        prob2.append('original P1 unit or replay launch record not found')
    else:
        o = orig[0]
        ocs = V.load_json(V.FRC_PRIV / 'compute' / 'units' / o['unit_id'] / 'CODE_AT_START.json')
        rec2['original_unit'] = {'unit_id': o['unit_id'], 'git_head': o['git_head'], 'utc': o['utc'], 'interpreter': o['interpreter']}
        rec2['same_interpreter'] = {k: o['interpreter'].get(k) == starts[-1]['interpreter'].get(k) for k in ('python', 'numpy', 'scipy', 'scikit-learn')}
        if not all(rec2['same_interpreter'].values()):
            notes2.append(f'interpreter differs: {rec2["same_interpreter"]}')
        dep = {f: (h, cs['files'].get(f)) for f, h in ocs['files'].items() if f.startswith(('nrm/', 'ohn/', 'scm_bench/'))
               and cs['files'].get(f) != h}
        rec2['nrm_ohn_scm_bench_files_changed_since_original'] = dep
        if dep:
            notes2.append(f'dependency files differ from the original P1 launch: {sorted(dep)}')
        old_route = V.git_blob(o['git_head'], 'frc/fa/route.py')
        rec2['original_route_sha256'] = None if old_route is None else V.sha_bytes(old_route)
        rec2['original_route_matches_its_unit_record'] = old_route is not None and V.sha_bytes(old_route) == ocs['files'].get('frc/fa/route.py')
        if not rec2['original_route_matches_its_unit_record']:
            prob2.append('the original P1 route.py is not recoverable from git with its recorded hash')
        else:
            new_route = V.git_blob(starts[-1]['git_head'], 'frc/fa/route.py') or (V.ROOT / 'frc' / 'fa' / 'route.py').read_bytes()
            cp = call_path_diff(old_route.decode(), new_route.decode())
            rec2['nested_c_call_path'] = cp
            if cp['changed'] or cp['only_old'] or cp['only_new'] or cp['constants_changed'] or cp['constants_only_old']:
                notes2.append(f'nested_c call path differs from the original P1 code: changed {cp["changed"]}, new {cp["only_new"]}, '
                              f'removed {cp["only_old"]}, constants changed {cp["constants_changed"]}')
    out.append({'id': f'V2.ORIGINAL_CODE__V{v}', 'block': 'V2', 'kind': 'procedural',
                'status': 'FAIL' if prob2 else ('ATTENTION' if notes2 else 'PASS'),
                'summary': 'the code the original P1 unit ran (FRC1 ledger, CODE_AT_START, git) against the replay\'s code: '
                           'interpreter, nrm / ohn / scm_bench files, the nested_c call path of frc/fa/route.py'
                           + (f'; problems {prob2}' if prob2 else '') + (f'; notes {notes2}' if notes2 else ''),
                'detail': {'problems': prob2, 'notes': notes2}, 'record': rec2})
    # PROVENANCE: the replay unit itself
    prob3, notes3, rec3 = [], [], {}
    if starts:
        st = starts[-1]
        ends = [e for e in es if e.get('event') == 'unit_end' and e['unit_id'] == st['unit_id']]
        rec3 = {'unit_start': st, 'unit_end': ends[-1] if ends else None}
        if not ends or ends[-1].get('status') != 'done':
            prob3.append('replay unit not done')
        import v1_access as A                                          # own import-closure scanner (AST)
        clos = A.import_closure(V.ROOT / 'fsa' / 'fa' / 'solve.py', V.ROOT)
        rec3['import_closure'] = sorted(clos)
        unc = st.get('uncommitted_code_paths') or []
        hit_u = [u for u in unc if any(c == u or c.startswith(u.rstrip('/') + '/') for c in clos)]
        if hit_u:
            prob3.append(f'uncommitted paths at launch inside the import closure: {hit_u}')
        elif unc:
            notes3.append(f'uncommitted paths at launch, outside the import closure: {unc}')
        drift = (ends[-1].get('code_changed_during_run') or []) if ends else []
        hit_d = [f for f in drift if f in clos]
        if hit_d:
            prob3.append(f'code inside the import closure changed during the run: {hit_d}')
        elif drift:
            notes3.append(f'code changed during the run, outside the import closure: {drift}')
    else:
        prob3.append('no replay unit')
    out.append({'id': f'V2.PROVENANCE__V{v}', 'block': 'V2', 'kind': 'procedural',
                'status': 'FAIL' if prob3 else ('ATTENTION' if notes3 else 'PASS'),
                'summary': f'replay unit {rec.get("replay_units")}: status, uncommitted paths at launch, code drift during the run'
                           + (f'; problems {prob3}' if prob3 else '') + (f'; notes {notes3}' if notes3 else ''),
                'detail': {'problems': prob3, 'notes': notes3}, 'record': rec3})
    return out


def loop_source(func_node):
    for node in ast.walk(func_node):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and [getattr(e, 'id', None) for e in node.target.elts] == ['fam', 'bud']:
            return ast.unparse(node)
    return None


def estimator_text(tag):
    """Procedural note for RULES 8 'textually identical': the replay's selection loop against stage_paths' (both as launched)."""
    checks = []
    for v in (0, 1):
        _, starts = _replay_unit(v)
        st = starts[-1]
        cs = V.load_json(V.COMPUTE / 'units' / st['unit_id'] / 'CODE_AT_START.json')
        route = V.git_blob(st['git_head'], 'frc/fa/route.py')
        solve = V.git_blob(st['git_head'], 'fsa/fa/solve.py')
        ok_src = (route is not None and solve is not None and V.sha_bytes(route) == cs['files']['frc/fa/route.py']
                  and V.sha_bytes(solve) == cs['files']['fsa/fa/solve.py'])
        fr, _ = _module_functions(route.decode())
        fs, _ = _module_functions(solve.decode())
        a, b = loop_source(fr['stage_paths']), loop_source(fs['stage_replay'])
        same_ren = _selection_loop(fr['stage_paths'], {}) == _selection_loop(fs['stage_replay'], REPLAY_RENAMES)
        rec = {'stage_paths_loop': a, 'stage_replay_loop': b, 'textually_identical': a == b, 'identical_after_renaming': same_ren,
               'renames': REPLAY_RENAMES, 'sources_are_the_launched_files': ok_src, 'replay_unit': st['unit_id']}
        status = 'FAIL' if not (ok_src and same_ren) else ('PASS' if a == b else 'ATTENTION')
        checks.append({'id': f'V2.ESTIMATOR_TEXT__V{v}__{tag}', 'block': 'V2', 'kind': 'procedural', 'status': status,
                       'summary': 'RULES 8 asks the selection loop of stage_paths to be textually identical: the replay (fsa/fa/solve.py '
                                  'stage_replay as launched) carries a copy whose text differs only in three local names (sel2 for sel, '
                                  'k2 / v2 for the comprehension variables k / v); after renaming them its AST equals stage_paths\' '
                                  f'loop ({same_ren}); strict textual identity: {a == b}; every frc P1 function is byte-identical '
                                  '(route.py = the pin). Reported for the reviewer\'s reading',
                       'detail': {'textually_identical': a == b, 'identical_after_renaming': same_ren, 'sources_ok': ok_src},
                       'record': rec, 'script': SCRIPT})
    V.Checks().append(checks)
    print(json.dumps({c['id']: c['status'] for c in checks}))
    return 0


def rules_info():
    """The committed RULES (statistical reviewer): p1_replay tolerances. Refuses to run on uncommitted RULES."""
    info = {}
    for r in (f'results/{V.CAMPAIGN}/roles/statistical_reviewer/RULES.md', f'results/{V.CAMPAIGN}/roles/statistical_reviewer/RULES.json'):
        _, dirty = V.git('status', '--porcelain', '--', r)
        _, log = V.git('log', '-1', '--format=%H %cI', '--', r)
        if not (V.ROOT / r).is_file() or dirty or not log:
            raise PermissionError(f'{r} is not committed and clean; the tolerances come from the committed RULES')
        info[r] = {'sha256': V.sha_file(V.ROOT / r), 'commit': log.split()[0], 'commit_utc': log.split()[1]}
    rules = V.load_json(V.ROOT / f'results/{V.CAMPAIGN}/roles/statistical_reviewer/RULES.json')
    return rules, info


def main_real(version, tag, labels):
    v = int(version)
    rules, rinfo = rules_info()
    tol = float(rules['p1_replay']['verifier_tolerance_abs'])
    rtol = float(rules['p1_replay']['reproduction_tolerance_abs'])
    sel_path = V.FRC_RES / 'fa' / f'SELECTION_V{v}.json'
    adm = V.load_json(V.RES / 'SOURCE_ADMISSION.json')
    if V.sha_file(sel_path) != adm['frc_private_artefacts'][f'frc/results/fa/SELECTION_V{v}.json']['sha256']:
        raise PermissionError('SELECTION record differs from its SOURCE_ADMISSION pin')
    sel = V.load_json(sel_path)
    rep_path = V.RES / 'replay' / f'P1_REPLAY_V{v}.json'
    rep = V.load_json(rep_path)
    npz_path = V.PRIV / 'solver' / f'P1_REPLAY_V{v}.npz'
    npz_sha = V.sha_file(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        arr = {k: z[k].copy() for k in z.files}
    C = V.admitted_loaders()                            # fsa.common: features label-free; load logged, operation 'verify'
    feats = C.features('MODEL_FIT')
    labels_ref = None
    if labels:
        lab = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), operation='verify',
                     purpose=f'independent verifier V2 (P1 replay V{v}): check that the per-person file carries the admitted MODEL_FIT '
                             'labels and households in row order (SEX is the held-out LOOKUP evaluation label); no fitting')
        if not np.array_equal(lab['row_id'], feats['row_id']):
            raise AssertionError('row order of load differs from features')
        labels_ref = {k: lab[k] for k in ('sex', 'income', 'occupation')}
    checks, mine = verify(arr, sel, rep, v, tol, rtol=rtol, unit_ref=feats['unit'], labels_ref=labels_ref, npz_sha=npz_sha)
    checks += code_checks(v, rules, rep)
    inputs = {'selection': V.rel(sel_path), 'selection_sha256': V.sha_file(sel_path), 'replay_record': V.rel(rep_path),
              'replay_record_sha256': V.sha_file(rep_path), 'npz': str(npz_path.relative_to(V.ROOT)), 'npz_sha256': npz_sha,
              'rules': rinfo, 'tolerance_abs': tol, 'reproduction_tolerance_abs': rtol, 'labels_read': bool(labels)}
    if mine is not None:
        pv = p1_verifier_record(mine, v)
        pv.update({'by': 'independent verifier (own code: roles/verifier/code/vcore.py, v2_replay.py)', 'inputs': inputs,
                   'alias_groups': mine['_alias_groups'], 'utc': V.utc_now(), 'unit_id': os.environ.get('FSA_UNIT_ID')})
        V.write_once_json(V.VER / f'P1_VERIFIER_V{v}.json', pv)
    for c in checks:
        c['id'] = f'{c["id"]}__{tag}'
        c['script'] = SCRIPT
        c['record'] = {'inputs': inputs, 'result': c.get('record')}
    V.Checks().append(checks)
    print(json.dumps({'version': v, 'tag': tag, 'statuses': {c['id']: c['status'] for c in checks},
                      'selection': None if mine is None else mine['selection'], 'f4': None if mine is None else mine['F4'],
                      'brackets': None if mine is None else mine['brackets']['brackets'],
                      'alias_groups': None if mine is None else mine['_alias_groups']}, indent=1, default=V._json_default))
    return 0                                     # the script ran to completion; statuses (FAIL included) live in CHECKS.json


# ---------------------------------------------------------------- selftest (invented data; brute-force references)
def _fake(rng, n=700, H=260):
    unit = np.sort(rng.integers(0, H, n)) * 104729 + 17
    sex = (rng.random(n) < 0.47).astype(np.int64)
    inc = rng.integers(0, 2, n)
    occ = rng.integers(0, 5, n)
    sel = {'lambda_grid': [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], 'ladder': [[12, 48], [8, 32], [6, 24]]}
    order = plan_order(sel)
    arr = {'unit': unit, 'sex': sex, 'income': inc, 'occupation': occ}
    base_i = rng.dirichlet(np.ones(2) * 2, n)
    base_o = rng.dirichlet(np.ones(5) * 2, n)
    for j, e in enumerate(order):
        fam = e.split('@')[0]
        step = 0 if e == TWIN else (j % 6) + 1
        eps = {'PRC': 0.004, 'HQ': 0.002, 'SHUF': 0.0005, 'ORD': 0.03}.get(fam, 0.0) * step
        w = min(eps * 4, 0.9)                               # mixture with fresh Dirichlet noise, clipped at 1e-6 as the releases
        Pi = np.clip((1 - w) * base_i + w * rng.dirichlet(np.ones(2), n), 1e-6, 1.0)
        Pi /= Pi.sum(1, keepdims=True)
        Po = np.clip((1 - w) * base_o + w * rng.dirichlet(np.ones(5), n), 1e-6, 1.0)
        Po /= Po.sum(1, keepdims=True)
        if e in ('PRC@0.5', 'PRC@1.0'):                     # an exact alias of PRC@0.25
            Pi, Po = arr['_Pi_PRC@0.25'], arr['_Po_PRC@0.25']
        arr[f'_Pi_{e}'], arr[f'_Po_{e}'] = Pi, Po
        for t, P, y, K in (('inc', Pi, inc, 2), ('occ', Po, occ, 5)):
            oh = np.eye(K)[y]
            arr[f'{e}|{t}_nll'] = -np.log(P[np.arange(n), y])
            arr[f'{e}|{t}_brier'] = ((P - oh) ** 2).sum(1)
            arr[f'{e}|{t}_acc'] = (P.argmax(1) == y).astype(np.float64)
        cells = rng.integers(0, 12, n) if e not in ('PRC@0.5', 'PRC@1.0') else arr['_cells_PRC@0.25']
        arr[f'_cells_{e}'] = cells
        rate = np.array([(sex[cells == c].sum() + 0.47) / ((cells == c).sum() + 1.0) for c in range(12)])
        for vw in LOOK:
            arr[f'{e}|{vw}'] = rate[cells] if vw == 'look_pair' else rate[(cells + (vw == 'look_income')) % 12]
    arr = {k: v for k, v in arr.items() if not k.startswith('_')}
    return arr, sel, order


def _record_bruteforce(arr, sel, order, version):
    """The recorded P1 numbers from the protocol's definitions with brute-force loops (independent of vcore's vectorised code)."""
    unit, sex = arr['unit'], arr['sex']
    groups, inv = np.unique(unit, return_inverse=True)
    G, n = len(groups), len(unit)

    def cse(x):
        x = np.asarray(x, float)
        s = np.zeros(G)
        for i in range(n):
            s[inv[i]] += x[i] - x.mean()
        return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)

    def sub_cse(d, sub):
        m = d[sub].mean()
        z = np.array([(d[i] - m) / sub.mean() if sub[i] else 0.0 for i in range(n)])
        return float(m), cse(z)
    nm = {}
    for e in order:
        g = lambda m: arr[f'{e}|{m}']                               # noqa: E731
        nm[e] = {k: float(np.mean(g(k))) for k in TASK}
        nm[e]['occ_brier_men'] = float(np.mean(g('occ_brier')[sex == 0]))
        nm[e]['occ_brier_women'] = float(np.mean(g('occ_brier')[sex == 1]))
        for vw in ('pair', 'income', 'occupation'):
            nm[e][f'auc_{vw}'] = V._naive_auc(g(f'look_{vw}'), sex)
        nm[e]['fit_auc_pair_insample'] = 0.7
    aud = {}
    for fam in ('PRC', 'HQ', 'SHUF', 'ORD'):
        aud[fam] = {}
        for c in family_path(order, fam):
            a, t = (lambda m: np.asarray(arr[f'{c}|{m}'], float)), (lambda m: np.asarray(arr[f'{TWIN}|{m}'], float))
            row = {}
            for m in ('occ_nll', 'inc_nll'):
                d = a(m) - t(m)
                row[m] = [float(d.mean()), cse(d)]
            for m in ('occ_acc', 'inc_acc'):
                d = t(m) - a(m)
                row[m] = [float(d.mean()), cse(d)]
            row['occ_brier_men'] = list(sub_cse(a('occ_brier') - t('occ_brier'), sex == 0))
            row['occ_brier_women'] = list(sub_cse(a('occ_brier') - t('occ_brier'), sex == 1))
            aud[fam][c] = row
    prc = family_path(order, 'PRC')                         # S2 admits the first three PRC strengths exactly (then stops)
    budgets = {'S1': {m: -1.0 for m in BUDGET_METRICS},
               'S2': {m: max(aud['PRC'][c][m][0] + 2 * aud['PRC'][c][m][1] for c in prc[:3]) + 1e-7 for m in BUDGET_METRICS},
               'S3': {m: 1.0 for m in BUDGET_METRICS}}
    selection = {}
    for fam, bs in (('PRC', ('S1', 'S2', 'S3')), ('HQ', ('S2',)), ('SHUF', ('S2',)), ('ORD', ('S1', 'S2', 'S3'))):
        selection[fam] = {}
        for b in bs:
            pick = None
            for c in family_path(order, fam):
                if all(aud[fam][c][m][0] + 2 * aud[fam][c][m][1] <= budgets[b][m] for m in budgets[b]):
                    pick = c
                else:
                    break
            selection[fam][b] = pick
    s2 = selection['PRC']['S2']
    tot = lambda e: np.asarray(arr[f'{e}|inc_nll']) + np.asarray(arr[f'{e}|occ_nll'])    # noqa: E731
    target = float(np.mean(tot(s2) - tot(TWIN))) if s2 else None
    brk = {'prc_s2': s2, 'target_total_nll_spend': target, 'brackets': {}, 'spend': {}}
    for fam in ('HQ', 'SHUF', 'ORD'):
        path = family_path(order, fam)
        spd = {c: float(np.mean(tot(c) - tot(TWIN))) for c in path}
        brk['spend'][fam] = spd
        brk['brackets'][fam] = next((c for c in path if spd[c] >= target), 'CONTROL_DID_NOT_REACH_SPEND')
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, int(version), 4]))
    Wd = [np.bincount(rng.integers(0, G, G), minlength=G)[inv].astype(float) for _ in range(B_BOOT)]

    def wauc_loop(s, w):                                            # explicit per-value sums
        num, cneg = 0.0, 0.0
        for val in np.unique(s):
            m = s == val
            wp = float(np.sum(w[m] * (sex[m] == 1)))
            wn = float(np.sum(w[m] * (sex[m] == 0)))
            num += wp * (cneg + 0.5 * wn)
            cneg += wn
        return num / (float(np.sum(w * (sex == 1))) * float(np.sum(w * (sex == 0))))
    ref = np.asarray(arr[f'{TWIN}|look_pair'], float)
    bref = np.array([wauc_loop(ref, w) for w in Wd])
    a_ref = V._naive_auc(ref, sex)
    nlp = {TWIN: {'auc': a_ref}}
    for e in order[1:]:
        s = np.asarray(arr[f'{e}|look_pair'], float)
        be = np.array([wauc_loop(s, w) for w in Wd])
        a = V._naive_auc(s, sex)
        nlp[e] = {'auc': a, 'diff_twin_minus_entry': a_ref - a, 'se_boot': float(np.std(bref - be, ddof=1))}
    f4 = {'entry': s2, 'diff_twin_minus_prc': nlp[s2]['diff_twin_minus_entry'], 'se_boot': nlp[s2]['se_boot'],
          'pass_2se': bool(nlp[s2]['diff_twin_minus_entry'] > 2 * nlp[s2]['se_boot']), 'rule': 'text', 'bootstrap': 'text'}
    return {'nested_means': nm, 'audit_spend_over_twin': aud, 'budgets': budgets, 'selection': selection, 'brackets': brk,
            'nested_lookup_pair': nlp, 'F4': f4, 'T_fixed': {'inc': 1}, 'lambda_grid': sel['lambda_grid'], 'ladder': sel['ladder']}


def selftest():
    import copy
    rng = np.random.default_rng(20261021)
    arr, sel0, order = _fake(rng)
    rec = _record_bruteforce(arr, sel0, order, 0)
    rep = copy.deepcopy({k: rec[k] for k in ('nested_means', 'audit_spend_over_twin', 'brackets', 'nested_lookup_pair', 'selection',
                                             'budgets', 'T_fixed', 'ladder')})
    rep['F4'] = {k: rec['F4'][k] for k in ('entry', 'diff_twin_minus_prc', 'se_boot', 'pass_2se')}
    rep.update({'version': 0, 'per_person_keys': sorted(arr), 'per_person_file_sha256': 'X', 'estimator_unchanged': True,
                'per_person_outputs_saved': True})
    res = []

    def run(name, a, s, r, expect, sha='X', tol=1e-9):
        checks, mine = verify(a, s, r, 0, tol, unit_ref=arr['unit'], labels_ref={k: arr[k] for k in ('sex', 'income', 'occupation')},
                              npz_sha=sha)
        st = {c['id'].split('__')[0]: c['status'] for c in checks}
        good = all(st.get(k) == want for k, want in expect.items())
        res.append((name, good, st))
        return checks, mine
    allpass = {f'V2.{k}': 'PASS' for k in ('FILE', 'CONSISTENCY', 'MEANS', 'SPEND', 'SELECTION', 'BRACKETS', 'LOOKUP', 'F4',
                                          'REPRODUCTION')}
    _, mine = run('clean_records_all_pass', arr, rec, rep, allpass)
    res.append(('selection_nontrivial', mine['selection']['PRC']['S2'] == 'PRC@1.0' and mine['selection']['PRC']['S1'] is None
                and mine['selection']['PRC']['S3'] == 'PRC@8.0', mine['selection']))
    res.append(('alias_found', any({'PRC@0.25', 'PRC@0.5', 'PRC@1.0'} <= set(g) for g in mine['_alias_groups']), mine['_alias_groups']))
    pv = p1_verifier_record(mine, 0)
    mm, _, _ = compare(rep, pv, ['audit_spend_over_twin', 'selection', 'F4', 'nested_lookup_pair', 'brackets'], 1e-9)
    res.append(('p1_verifier_record_matches_replay_structure', not mm, mm[:3]))
    r2 = copy.deepcopy(rec); r2['nested_means']['HQ@2.0']['occ_nll'] += 1e-6
    run('planted_mean_error_in_record', arr, r2, rep, {'V2.MEANS': 'FAIL', 'V2.SPEND': 'PASS', 'V2.REPRODUCTION': 'FAIL'})
    p2 = copy.deepcopy(rep); p2['nested_means']['HQ@2.0']['occ_nll'] += 1e-6
    run('planted_mean_error_in_replay', arr, rec, p2, {'V2.MEANS': 'FAIL', 'V2.REPRODUCTION': 'FAIL'})
    p3 = copy.deepcopy(rep); p3['nested_means']['HQ@2.0']['fit_auc_pair_insample'] += 1e-11
    run('reproduction_tolerance_1e12', arr, rec, p3, {'V2.REPRODUCTION': 'FAIL', 'V2.MEANS': 'PASS'})
    p4 = copy.deepcopy(rep); p4['audit_spend_over_twin']['PRC']['PRC@2.0']['occ_brier_women'][1] *= 1.001
    run('planted_subgroup_se_error', arr, rec, p4, {'V2.SPEND': 'FAIL', 'V2.MEANS': 'PASS'})
    r4 = copy.deepcopy(rec); fs = r4['selection']['ORD']; fs['S2'] = 'ORD@6x24' if fs['S2'] != 'ORD@6x24' else 'ORD@12x48'
    run('planted_selection_error', arr, r4, rep, {'V2.SELECTION': 'FAIL', 'V2.REPRODUCTION': 'FAIL'})
    p5 = copy.deepcopy(rep); p5['nested_lookup_pair']['SHUF@4.0']['se_boot'] += 1e-7
    run('planted_bootstrap_se_error', arr, rec, p5, {'V2.LOOKUP': 'FAIL', 'V2.F4': 'PASS'})
    p6 = copy.deepcopy(rep); p6['F4']['pass_2se'] = not p6['F4']['pass_2se']
    run('planted_f4_flag_error', arr, rec, p6, {'V2.F4': 'FAIL'})
    p7 = copy.deepcopy(rep); p7['brackets']['brackets']['HQ'] = 'HQ@0.25' if p7['brackets']['brackets']['HQ'] != 'HQ@0.25' else 'HQ@8.0'
    run('planted_bracket_error', arr, rec, p7, {'V2.BRACKETS': 'FAIL'})
    run('npz_hash_mismatch', arr, rec, rep, {'V2.FILE': 'FAIL', 'V2.MEANS': 'PASS'}, sha='Y')
    a8 = dict(arr); a8['HQ@1.0|inc_brier'] = a8['HQ@1.0|inc_brier'] + 1e-6
    run('planted_brier_identity_break', a8, rec, rep, {'V2.CONSISTENCY': 'FAIL'})
    a9 = dict(arr); a9['sex'] = a9['sex'][::-1].copy()
    run('labels_reordered', a9, rec, rep, {'V2.FILE': 'FAIL'})
    wrong = _record_bruteforce(arr, sel0, order, 1)                     # the bootstrap of another seed changes every SE
    run('wrong_bootstrap_seed_detected', arr, rec, dict(rep, nested_lookup_pair=wrong['nested_lookup_pair']), {'V2.LOOKUP': 'FAIL'})
    # boundary tie: put the S2 budget of one metric exactly at a candidate's est + 2 SE (within 1e-12) and flip the record's choice
    r10 = copy.deepcopy(rec); p10 = copy.deepcopy(rep)
    c4 = family_path(order, 'PRC')[3]
    tied = {m: rec['audit_spend_over_twin']['PRC'][c4][m][0] + 2 * rec['audit_spend_over_twin']['PRC'][c4][m][1] for m in BUDGET_METRICS}
    for m in BUDGET_METRICS:
        r10['budgets']['S2'][m] = max(r10['budgets']['S2'][m], tied[m] + 1e-13)
    p10['budgets'] = r10['budgets']
    _, mt = run('boundary_tie_scan', arr, r10, p10, {'V2.FILE': 'PASS'})
    pick = mt['selection']['PRC']['S2']
    r10['selection'] = copy.deepcopy(mt['selection'])        # the raised S2 budgets move other families too; keep those
    r10['selection']['PRC']['S2'] = family_path(order, 'PRC')[2] if pick == c4 else pick
    p10['selection'] = copy.deepcopy(r10['selection'])
    run('boundary_tie_reported_not_failed', arr, r10, p10, {'V2.SELECTION': 'ATTENTION'})
    # call-path differ: a changed callee is reported, an untouched function is not
    old = 'X = 1\ndef a(e):\n    return b(e) + X\ndef b(e):\n    return e\ndef c():\n    return 0\n'
    new = 'X = 1\ndef a(e):\n    return b(e) + X\ndef b(e):\n    return d(e)\ndef d(e):\n    return e\ndef c():\n    return 1\n'
    cp = call_path_diff(old, new, root='a')
    res.append(('call_path_diff', cp['changed'] == ['b'] and cp['only_new'] == ['d'] and 'c' not in cp['changed'], cp['changed']))
    cp2 = call_path_diff(old, old, root='a')
    res.append(('call_path_same', not cp2['changed'] and not cp2['only_new'] and not cp2['only_old'], cp2['changed']))
    # selection-loop comparison up to the result name
    f1 = ast.parse('def stage_paths():\n    for fam, bud in ((1, 2),):\n        sel[fam] = bud\n').body[0]
    f2 = ast.parse('def stage_replay():\n    for fam, bud in ((1, 2),):\n        sel2[fam] = bud\n').body[0]
    f3 = ast.parse('def stage_replay():\n    for fam, bud in ((1, 3),):\n        sel2[fam] = bud\n').body[0]
    res.append(('selection_loop_rename', _selection_loop(f1, {}) == _selection_loop(f2, REPLAY_RENAMES)
                and _selection_loop(f1, {}) != _selection_loop(f3, REPLAY_RENAMES) and _selection_loop(f1, {}) != _selection_loop(f2, {}), None))
    n_ok = sum(r[1] for r in res)
    for name, good, info in res:
        print(('PASS ' if good else 'FAIL ') + name + ('' if good else f'  [{info}]'))
    print(f'v2_replay selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--version', type=int, choices=(0, 1))
    ap.add_argument('--tag', default='V1')
    ap.add_argument('--labels', action='store_true')
    ap.add_argument('--estimator-text', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.estimator_text:
        if not os.environ.get('FSA_UNIT_ID'):
            sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
        sys.exit(estimator_text(a.tag))
    if a.version is None:
        ap.error('--version is required for a real run')
    if not os.environ.get('FSA_UNIT_ID'):
        sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
    sys.exit(main_real(a.version, a.tag, a.labels))

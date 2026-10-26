#!/usr/bin/env python3
"""RFA1 verifier, wave 2, tasks 1 and 3 (LABEL-FREE: JSON records and ledgers only; no array, no label, no model file).

Task 1, model selection from records:
  - k* of RF_FULL per version recomputed from fam/FIT_RF_V<v>.json selection.path (FULL_S10..S400 spends over the fold start) with
    the card's budgets (cards/RF_RFIT_v1.md section 5: estimate + 2 SE <= budget on 5 occupation components, every S_10..S_k);
  - the STOPFIT / ALTCRIT / ALT bracket points recomputed from paths.<key>.grid with the SAR1 rule (sar/fb/pfd.py bracket, read by
    the verifier and re-implemented here); grid spends cross-checked against selection.path; RULES 4.4 fineness labels;
  - the F0 gate statistics recomputed from their recorded raw cross-group estimates and labelled by the card's rule; F1 (per-epoch
    statistics along FULL) read before S_k*; the ALTCRIT near-alias rule from the recorded median lag;
  - the saved configurations (FIT record 'saved', RELEASE.json records) and the CONFIG_LEDGER (12 configurations).
Task 3, data boundaries: LABEL_ACCESS.jsonl and the queue ledger.

Run (outside the queue: JSON parsing only; CPU reported in the hand-back):
  <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_records.py \
      --commit-times '43830ae19=2026-10-25T21:51:57Z,ff741a72c=2026-10-25T21:59:32Z'
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
PRIV = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1'
OUT = HERE.parent / 'out' / 'V2_RECORDS.json'
CARD = RES / 'cards' / 'RF_RFIT_v1.md'
CARD_BUDGET = {'occ_nll': 0.0060, 'occ_brier': 0.0025, 'occ_brier_men': 0.0028, 'occ_brier_women': 0.0028, 'occ_acc': 0.0060}
STEPS = tuple(range(10, 401, 10))
ALT_STEPS = tuple(range(90, 481, 10))
RECEIPT_UTC = '2026-10-25T21:53:27+00:00'
CLOSED_ROLES = ('CALIBRATION', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
CARDS = {}
PLAN = {'RF_FULL': ('FULL', 'CAND', 'candidate'), 'RF_TASK': ('TASK', 'CONT', 'twin'), 'RF_STOP_LO': ('STOPFIT', 'STOPFIT', 'control'),
        'RF_STOP_HI': ('STOPFIT', 'STOPFIT', 'control'), 'RF_ALTC_HI': ('ALTCRIT', 'ALTCRIT', 'control'), 'RF_RAWJ_HI': ('RAWJ', 'ALT', 'control')}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(dt.timezone.utc)


# ---------------------------------------------------------------- task 1
def recompute_kstar(path, budget, k_se=2.0):
    """Last S_k such that S_10..S_k all meet estimate + k_se SE <= budget on every component; the first failing step and its
    binding components."""
    k, first_fail, binding = None, None, []
    for s in STEPS:
        sp = path[f'FULL_S{s}']
        bad = [c for c, lim in budget.items() if not (sp[c]['estimate'] + k_se * sp[c]['se'] <= lim)]
        if bad:
            first_fail, binding = s, bad
            break
        k = s
    margins = None
    if k is not None:
        sp = path[f'FULL_S{k}']
        margins = {c: lim - (sp[c]['estimate'] + k_se * sp[c]['se']) for c, lim in budget.items()}
    fail_detail = None
    if first_fail is not None:
        sp = path[f'FULL_S{first_fail}']
        fail_detail = {c: {'estimate_plus_2se': sp[c]['estimate'] + k_se * sp[c]['se'], 'budget': budget[c]} for c in binding}
    return k, first_fail, fail_detail, margins


def recompute_bracket(grid_rows, cand_value):
    """SAR1 bracket rule (sar/fb/pfd.py bracket) on recorded rows {step: spend_over_start, se_diff_vs_cand}: LO = largest spend <=
    CAND's; HI = smallest spend >= CAND's + 2 SE(point - CAND) (ties: first in grid order); no such point: HI = largest spend,
    CONTROL_PATH_MAY_NOT_BRACKET; LO None: LO_IS_ORIGIN; LO == HI: LO None, LO_EQUALS_HI; CAND <= 0: CTRL_BRACKET_IS_ORIGIN."""
    steps = list(grid_rows)
    flags = []
    if cand_value <= 0:
        return None, None, ['CTRL_BRACKET_IS_ORIGIN'], None
    below = [g for g in steps if grid_rows[g]['spend_over_start'] <= cand_value]
    lo = None
    for g in below:                                   # max by spend, first in grid order on ties (python max semantics)
        if lo is None or grid_rows[g]['spend_over_start'] > grid_rows[lo]['spend_over_start']:
            lo = g
    above = [g for g in steps if grid_rows[g]['spend_over_start'] >= cand_value + 2.0 * grid_rows[g]['se_diff_vs_cand']]
    if above:
        hi = None
        for g in above:
            if hi is None or grid_rows[g]['spend_over_start'] < grid_rows[hi]['spend_over_start']:
                hi = g
    else:
        hi = None
        for g in steps:
            if hi is None or grid_rows[g]['spend_over_start'] > grid_rows[hi]['spend_over_start']:
                hi = g
        flags.append('CONTROL_PATH_MAY_NOT_BRACKET')
    if lo is None:
        flags.append('LO_IS_ORIGIN')
    elif lo == hi:
        lo = None
        flags.append('LO_EQUALS_HI')
    sp = [r['spend_over_start'] for r in grid_rows.values()]
    fineness = {'below': any(0.75 * cand_value <= s <= cand_value for s in sp), 'above': any(cand_value <= s <= 1.25 * cand_value for s in sp)}
    return lo, hi, flags, fineness


def rules_fineness(lo, hi, rows, c):
    """RULES.md 4.4 (descriptive): FINE iff LO >= 0.75 s (and <= s) and s + 2 SE <= HI <= s + 2 SE + 0.25 s (SE: the HI point's paired
    SE against CAND); a missing (unsaved) LO is read from START (0)."""
    lo_s = 0.0 if lo is None else rows[lo]['spend_over_start']
    below_ok = 0.75 * c <= lo_s <= c
    if hi is None:
        return 'INCOMPLETE'
    se = rows[hi]['se_diff_vs_cand']
    hs = rows[hi]['spend_over_start']
    if hs < c + 2 * se:
        return 'CONTROL_PATH_MAY_NOT_BRACKET'
    above_ok = hs <= c + 2 * se + 0.25 * c
    return 'FINE' if below_ok and above_ok else 'COARSE_BOTH' if not below_ok and not above_ok else \
        'COARSE_BELOW' if not below_ok else 'COARSE_ABOVE'


def gate_check(g, rule):
    dd, ff, fd, tt = g['dir_sq'], g['fit_sq'], g['fit_dot_dir'], g['tot_sq']
    td = dd + fd                                     # cross(T, D) = cross(D, D) + cross(F, D) (bilinear cross-group estimator)
    mine = {'systematic_share': math.sqrt(max(ff, 0.0)) / math.sqrt(dd), 'orthogonal_share': math.sqrt(max(ff - fd * fd / dd, 0.0)) / math.sqrt(dd),
            'cos_tot_dir': td / math.sqrt(tt * dd), 'parallel_coefficient': fd / dd, 'tot_sq_from_parts': dd + 2 * fd + ff}
    inert = mine['orthogonal_share'] < rule['orthogonal_below'] and mine['cos_tot_dir'] > rule['cos_above']
    label = 'MECHANISM_INERT' if inert else 'MECHANISM_ACTIVE'
    diffs = {k: abs(mine[k] - g[k]) for k in ('systematic_share', 'orthogonal_share', 'cos_tot_dir', 'parallel_coefficient')}
    diffs['tot_sq_consistency'] = abs(mine['tot_sq_from_parts'] - tt)
    return mine, label, diffs


def task1(v, rec, card_sha):
    out, chk = {}, {}
    sel = rec['selection']
    path = sel['path']
    chk['budget_equals_card'] = sel['budget'] == CARD_BUDGET
    k, ff, fdet, margins = recompute_kstar(path, CARD_BUDGET)
    out['k_star'] = {'recorded': sel['k_star'], 'verifier': k, 'first_failing_step': ff, 'binding_at_first_failure': fdet,
                     'margins_at_k_star': margins}
    chk['k_star_reproduced'] = k == sel['k_star']
    cand = rec['cand']
    chk['cand_record_consistent'] = (cand['strength'] == k and cand['checkpoint'] == f'S{k}' and cand['status'] == 'FITTED'
                                     and cand['config'] == 'RF_FULL')
    start_tot = path[f'RAWJ_S80']['tot_nll']['estimate']
    c_from_path = path[f'FULL_S{k}']['tot_nll']['estimate'] - start_tot
    out['cand_spend_over_start'] = {'recorded': cand['spend_over_start']['value'], 'from_selection_path': c_from_path}
    chk['cand_spend_over_start_consistent_with_path (abs < 1e-12)'] = abs(c_from_path - cand['spend_over_start']['value']) < 1e-12
    # brackets
    br = {}
    pre = {'STOPFIT': ('STOPFIT', STEPS), 'ALTCRIT': ('ALTCRIT', STEPS), 'ALT': ('RAWJ', ALT_STEPS)}
    for key, (arm, steps) in pre.items():
        b = rec['paths'][key]
        rows = {s: b['grid'][str(s)] for s in steps}
        c = b['cand_spend_over_start']['value']
        lo, hi, flags, fin = recompute_bracket(rows, c)
        rec_flags = [f for f in b['flags'] if f != 'LO_NOT_SAVED_SPEC_BUDGET']
        expected_extra = ['LO_NOT_SAVED_SPEC_BUDGET'] if key in ('ALTCRIT', 'ALT') and lo is not None else []
        cons = max(abs(rows[s]['spend_over_start'] - (path[f'{arm}_S{s}']['tot_nll']['estimate'] - start_tot)) for s in steps)
        cons_d = max(abs(rows[s]['diff_vs_cand'] - (path[f'{arm}_S{s}']['tot_nll']['estimate'] - path[f'FULL_S{k}']['tot_nll']['estimate']))
                     for s in steps)
        br[key] = {'recorded': {'lo': b['lo'], 'hi': b['hi'], 'flags': b['flags'], 'fineness': b.get('fineness')},
                   'verifier': {'lo': lo, 'hi': hi, 'flags': flags + expected_extra, 'fineness': fin},
                   'rules_4_4_fineness': rules_fineness(lo, hi, rows, c),
                   'lo_spend': None if lo is None else rows[lo]['spend_over_start'], 'hi_spend': None if hi is None else rows[hi]['spend_over_start'],
                   'cand_spend': c, 'hi_threshold': None if hi is None else c + 2 * rows[hi]['se_diff_vs_cand'],
                   'max_abs_grid_vs_selection_path': cons, 'max_abs_diff_vs_cand_vs_selection_path': cons_d}
        chk[f'bracket_{key}_reproduced'] = (lo == b['lo'] and hi == b['hi'] and sorted(flags + expected_extra) == sorted(b['flags'])
                                            and fin == b.get('fineness'))
        chk[f'bracket_{key}_grid_consistent_with_selection_path (abs < 1e-12)'] = cons < 1e-12 and cons_d < 1e-12
        chk[f'bracket_{key}_cand_spend_is_same_as_cand_record'] = abs(c - cand['spend_over_start']['value']) == 0.0
        del rec_flags
    out['brackets'] = br
    # gate
    g = rec['mechanism_gate']
    mine, label, diffs = gate_check(g, g['gate'])
    out['gate'] = {'recorded_label': g['label'], 'verifier_label': label, 'verifier_stats': mine, 'abs_diffs': diffs, 'n': g['n'],
                   'groups': g['groups'], 'recorded_rule': g['gate']}
    chk['gate_rule_equals_card'] = g['gate']['orthogonal_below'] == 0.05 and g['gate']['cos_above'] == 0.995 and g['gate']['episodes'] == 64 \
        and g['gate']['groups'] == 8
    chk['gate_statistics_recorded_64_episodes_8_groups'] = g['n'] == 64 and g['groups'] == 8 and g['status'] == 'OK'
    chk['gate_label_by_card_rule'] = label == g['label']
    chk['gate_derived_statistics_consistent (abs < 1e-12)'] = all(x < 1e-12 for x in diffs.values())
    # F1 along FULL before S_k* (deployed fit and folds)
    f1 = {'deployed': [{k2: e.get(k2) for k2 in ('epoch_end_step', 'n', 'orthogonal_share', 'cos_tot_dir', 'status', 'partial_epoch')}
                       for e in rec['full_records']['FULL'].get('mechanism_by_epoch') or []]}
    for j, fr in enumerate(rec['fold_records']):
        f1[f'fold{j}'] = [{k2: e.get(k2) for k2 in ('epoch_end_step', 'n', 'orthogonal_share', 'cos_tot_dir', 'status')}
                          for e in fr['FULL'].get('mechanism_by_epoch') or [] if e.get('epoch_end_step', 0) <= max(k, 40)]
    below = [(w, e['epoch_end_step'], e['orthogonal_share']) for w, es in f1.items() for e in es
             if isinstance(e.get('orthogonal_share'), (int, float)) and e['orthogonal_share'] < 0.05]
    out['F1_epochs_up_to_k_star_or_first_epoch'] = f1
    out['F1_orthogonal_share_below_0.05'] = below
    lag = rec.get('altcrit_rel_lag_median_deployed')
    out['altcrit_lag'] = {'median_rel_lag_deployed': lag, 'near_alias_by_card_rule': None if lag is None else lag < 0.01}
    chk['altcrit_lag_recorded'] = isinstance(lag, (int, float))
    # saved configurations (FIT record)
    sv = rec['saved']
    exp_ck = {'RF_FULL': k, 'RF_TASK': k, 'RF_STOP_LO': br['STOPFIT']['verifier']['lo'], 'RF_STOP_HI': br['STOPFIT']['verifier']['hi'],
              'RF_ALTC_HI': br['ALTCRIT']['verifier']['hi'], 'RF_RAWJ_HI': br['ALT']['verifier']['hi']}
    want = {c for c, s in exp_ck.items() if s is not None}
    chk['saved_set_equals_plan'] = set(sv) == want and len(sv) <= 6
    ok_saved, lam = True, {}
    for c, e in sv.items():
        arm, pth, kind = PLAN[c]
        ok_saved &= (e['arm'] == arm and e['path'] == pth and e['strength'] == exp_ck[c] and e['checkpoint'] == f'S{exp_ck[c]}' and
                     e['selection_k_star'] == k)
        lam[c] = e['lambda']
    chk['saved_points_at_k_star_and_bracket_points'] = bool(ok_saved)
    chk['lambda_shared_FULL_STOPFIT_ALTCRIT_and_TASK_zero'] = (lam['RF_FULL'] == lam['RF_STOP_HI'] == lam['RF_ALTC_HI'] ==
                                                              lam.get('RF_STOP_LO', lam['RF_FULL']) > 0 and lam['RF_TASK'] == 0.0)
    out['lambda'] = lam
    out['n_episodes_deployed_fit'] = {c: (e.get('n_episodes_deployed_fit'), e['strength']) for c, e in sv.items()}
    # RELEASE.json records
    rel = {}
    okr = True
    for c in sv:
        r = json.loads((PRIV / 'runs' / 'FAM' / c / f'V{v}' / 'RELEASE.json').read_text())
        arm, pth, kind = PLAN[c]
        good = (r['kind'] == kind and r['arm'] == arm and r['checkpoint'] == f'S{exp_ck[c]}' and r['card_sha256'] == card_sha and
                r['mechanism_card'] == 'results/pcrl_reader_fit_aware_release_v1/cards/RF_RFIT_v1.md' and
                r['roles_read'] == 'MODEL_FIT: income, occupation, sex' and r['unit_id'] == f'RFA_FIT_RF_V{v}' and r['version'] == v and
                r['randomized'] is False and r['training_people'] == 20000 and r['config_id'] == c and r['family'] == 'FAM')
        okr &= good
        rel[c] = {'release_sha256': r['release_sha256'], 'kind': r['kind'], 'checkpoint': r['checkpoint'], 'ok': good,
                  'roles_read': r['roles_read'], 'unit_id': r['unit_id'], 'card_commit': r.get('card_commit')}
    chk['release_records_match_plan_card_and_MODEL_FIT_only'] = bool(okr)
    out['release_records'] = rel
    return out, chk


def ledger_check(fits, card_sha):
    C = [json.loads(l) for l in (PRIV / 'runs' / 'CONFIG_LEDGER.jsonl').read_text().splitlines() if l.strip()]
    keys = [(c['family'], c['config_id'], c['version']) for c in C]
    chk = {'twelve_configurations': len(C) == 12, 'unique': len(set(keys)) == len(keys),
           'numbers_1_to_12': sorted(c['configuration_number'] for c in C) == list(range(1, 13)),
           'card_sha_equals_card_file': all(c['card_sha256'] == card_sha for c in C)}
    ok = True
    for c in C:
        v = c['version']
        r = fits[v]['release_records'].get(c['config_id'])
        ok &= r is not None and r['release_sha256'] == c['release_sha256'] and r['kind'] == c['kind'] and c['unit_id'] == f'RFA_FIT_RF_V{v}'
    chk['ledger_matches_release_records'] = bool(ok)
    chk['six_per_version_as_card_section_7'] = all(sum(1 for c in C if c['version'] == v) == 6 for v in (0, 1))
    return {'n': len(C), 'entries': [{k: c[k] for k in ('configuration_number', 'config_id', 'version', 'kind', 'release_sha256', 'unit_id', 'utc')}
                                     for c in C]}, chk


# ---------------------------------------------------------------- task 3
def task3(card_sha, commit_times):
    L = [json.loads(l) for l in (PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if l.strip()]
    Q = [json.loads(l) for l in (PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    starts = {e['unit_id']: e for e in Q if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: e for e in Q if e.get('event') == 'unit_end'}
    receipt = ts(RECEIPT_UTC)
    rec_file = json.loads((PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json').read_text())
    chk = {'receipt_verified_record': rec_file.get('remote_sha_verified') is True and rec_file.get('verified_utc') == RECEIPT_UTC
           and rec_file.get('research_commit') == '43830ae19'}
    rows, problems = [], []
    for r in L:
        t = ts(r['utc'])
        u = r.get('unit_id')
        st, en = starts.get(u), ends.get(u)
        chain = r.get('module_chain') or []
        fam_in_chain = any(str(m).startswith('rfa.fam') for m in chain)
        row = {'utc': r['utc'], 'unit': u, 'role': r['role'], 'fields': r['fields'], 'operation': r['operation'], 'entry': r['entry_point'],
               'after_receipt': t >= receipt, 'inside_unit': bool(st) and ts(st['utc']) <= t and (en is None or t <= ts(en['utc'])),
               'unit_stage': None if not st else st.get('stage'), 'fam_in_chain': fam_in_chain}
        rows.append(row)
        if not row['after_receipt']:
            problems.append(f'{r["utc"]} {u}: label read before the verified receipt')
        if r['role'] in CLOSED_ROLES:
            problems.append(f'{r["utc"]} {u}: closed role {r["role"]} read')
        if not row['inside_unit']:
            problems.append(f'{r["utc"]} {u}: read outside its queue unit window')
        if r['role'] == 'MODEL_SELECT' and (fam_in_chain or str(r['entry_point']).startswith('rfa.fam')):
            problems.append(f'{r["utc"]} {u}: MODEL_SELECT read with a fitting module on the stack')
        if r['operation'] == 'fit':
            cd = CARDS.get(r.get('card'))
            if r['role'] != 'MODEL_FIT' or not str(r['entry_point']).startswith('rfa.fam') or cd is None or \
                    r.get('card_sha256') != cd['sha256'] or r.get('card_commit') != cd['commit'] or not ts(cd['commit_utc']) < t:
                problems.append(f'{r["utc"]} {u}: fit read not MODEL_FIT / not rfa.fam / card not a committed RF card before the read')
        if r['role'] == 'MODEL_FIT' and r['operation'] not in ('fit', 'verify'):
            problems.append(f'{r["utc"]} {u}: MODEL_FIT labels read by operation {r["operation"]}')
    fit_units = sorted({r['unit_id'] for r in L if str(r.get('unit_id', '')).startswith('RFA_FIT')})
    fit_roles = {u: sorted({r['role'] for r in L if r.get('unit_id') == u}) for u in fit_units}
    ms_first = min((ts(r['utc']) for r in L if r['role'] == 'MODEL_SELECT'), default=None)
    chk['every_label_read_after_receipt'] = all(r['after_receipt'] for r in rows)
    chk['no_closed_role_label_read'] = not any(r['role'] in CLOSED_ROLES for r in L)
    chk['no_MODEL_SELECT_read_by_fitting_entry_point'] = not any(r['role'] == 'MODEL_SELECT' and (r['fam_in_chain'] or str(r['entry']).startswith('rfa.fam')) for r in rows)
    chk['fit_units_read_only_MODEL_FIT'] = all(v == ['MODEL_FIT'] for v in fit_roles.values()) and len(fit_units) >= 2
    chk['every_read_inside_a_queue_unit'] = all(r['inside_unit'] for r in rows)
    chk['fit_reads_name_a_committed_RF_card_committed_before_the_read'] = all(
        r.get('card') in CARDS and r.get('card_sha256') == CARDS[r['card']]['sha256'] and r.get('card_commit') == CARDS[r['card']]['commit']
        and ts(CARDS[r['card']]['commit_utc']) < ts(r['utc']) for r in L if r['operation'] == 'fit')
    chk['MODEL_SELECT_read_only_by_benchmark_readers_verify'] = all(r['operation'] in ('benchmark', 'readers', 'verify')
                                                                     for r in L if r['role'] == 'MODEL_SELECT')
    if 'ff741a72c' in commit_times and ms_first is not None:
        chk['fits_and_plan_committed_before_first_MODEL_SELECT_read'] = ts(commit_times['ff741a72c']) < ms_first
    if '43830ae19' in commit_times:
        chk['card_committed_before_first_fit_read'] = all(ts(commit_times['43830ae19']) < ts(r['utc']) for r in L if r['operation'] == 'fit')
    unit_ok = []
    for u, st in starts.items():
        en = ends.get(u)
        unit_ok.append({'unit': u, 'stage': st.get('stage'), 'start': st['utc'], 'end': None if not en else en['utc'],
                        'status': None if not en else en.get('status'), 'charged_cpu_s': None if not en else en.get('charged_cpu_s'),
                        'uncommitted_code_paths': st.get('uncommitted_code_paths'), 'git_head': st.get('git_head')})
    return {'n_label_reads': len(L), 'reads': rows, 'problems': problems, 'fit_units_roles': fit_roles,
            'first_MODEL_SELECT_read': None if ms_first is None else ms_first.isoformat(), 'queue_units': unit_ok,
            'charges': [e for e in Q if e.get('event') == 'charge']}, chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit-times', default='')
    ap.add_argument('--cards', default='', help='card path=commit sha=commit utc;... (committed RF cards)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    global OUT
    if a.out:
        OUT = HERE.parent / 'out' / a.out
    for item in [x for x in a.cards.split(';') if x]:
        pth, commit, cutc = item.split('=')
        CARDS[pth] = {'sha256': sha256(ROOT / pth), 'commit': commit, 'commit_utc': cutc}
    if not CARDS:
        CARDS['results/pcrl_reader_fit_aware_release_v1/cards/RF_RFIT_v1.md'] = {
            'sha256': sha256(CARD), 'commit': '43830ae19f0f098e6f8ce51d91326655b7fe6422', 'commit_utc': '2026-10-25T21:51:57Z'}
    commit_times = dict(x.split('=') for x in a.commit_times.split(',') if x)
    t0 = time.process_time()
    card_sha = sha256(CARD)
    res = {'schema': 'RFA1-verifier-records-v2', 'card_sha256': card_sha, 'cards': CARDS, 'checks': {}, 'versions': {},
           'inputs_sha256': {f'fam/FIT_RF_V{v}.json': sha256(RES / 'fam' / f'FIT_RF_V{v}.json') for v in (0, 1)}}
    fits = {}
    for v in (0, 1):
        rec = json.loads((RES / 'fam' / f'FIT_RF_V{v}.json').read_text())
        out, chk = task1(v, rec, card_sha)
        fits[v] = out
        res['versions'][str(v)] = out
        for kk, vv in chk.items():
            res['checks'][f'V{v}|{kk}'] = vv
    led, chk = ledger_check(fits, card_sha)
    res['config_ledger'] = led
    res['checks'].update({f'ledger|{k}': v for k, v in chk.items()})
    acc, chk = task3(card_sha, commit_times)
    res['access'] = acc
    res['checks'].update({f'access|{k}': v for k, v in chk.items()})
    res['all_pass'] = all(bool(v) for v in res['checks'].values())
    res['cpu_s'] = time.process_time() - t0
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, v in res['checks'].items() if not v],
                      'n_checks': len(res['checks']), 'problems': acc['problems'], 'cpu_s': res['cpu_s']}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

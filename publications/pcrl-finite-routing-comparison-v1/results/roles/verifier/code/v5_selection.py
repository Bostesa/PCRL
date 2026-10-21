"""V5a P1 selection, brackets and F4 from the recorded numbers (label-free rule replay with internal cross-checks), own code.

P1 (frc.fa.route paths) writes results/<campaign>/fa/SELECTION_V<v>.json without person-level outputs, so the nested spends, their
household SEs, the nested LOOKUP AUCs and the F4 bootstrap SE cannot be recomputed independently here. What this script does:
- the registered constants: budgets S1-S3 (card FA_FRC_v1 / FA_PRC_v1 table, typed here), the lambda grid, the ORD ladder of the
  protection capacity, the construction / capacity of the committed qualification decision;
- the selection rule replayed on the recorded audit table (per family in REGISTERED path order: the last candidate such that it
  and every earlier candidate keep estimate + 2 SE <= budget on every metric; None if the first fails); aliases (a budget whose
  selection equals an earlier budget's);
- every audit ESTIMATE cross-checked against the recorded nested means (mean of paired differences = difference of means, up to
  rounding; accuracy as a loss; men's / women's occupation Brier from the subgroup means);
- the brackets: the target = total-NLL spend of the PRC configuration selected at S2 (from the nested means), every recorded path
  spend against the nested means, the bracket = first strength in path order with spend >= target, else
  CONTROL_DID_NOT_REACH_SPEND;
- F4: the entry is the S2 selection, diff = AUC_twin - AUC_entry against the recorded LOOKUP AUCs and nested means, pass iff
  diff - 2 se_boot > 0;
- the SEX-side cache hash against its own record and the selection record; the task T* against the source task cache.

    python3 .../v5_selection.py --tag V1
    python3 .../v5_selection.py --selftest
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

BUDGETS = {
    'S1': {'occ_nll': 0.004, 'inc_nll': 0.002, 'occ_brier_men': 0.0015, 'occ_brier_women': 0.0015, 'occ_acc': 0.003, 'inc_acc': 0.002},
    'S2': {'occ_nll': 0.008, 'inc_nll': 0.004, 'occ_brier_men': 0.0030, 'occ_brier_women': 0.0030, 'occ_acc': 0.006, 'inc_acc': 0.004},
    'S3': {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008, 'inc_acc': 0.006}}
GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BASE_LADDER = ((12, 48), (8, 32), (6, 24), (4, 16), (3, 12), (2, 8))
LADDERS = {'C64': BASE_LADDER, 'C128': ((16, 64),) + BASE_LADDER, 'C256': ((24, 128), (16, 64)) + BASE_LADDER}
FAMILY_BUDGETS = {'PRC': ('S1', 'S2', 'S3'), 'HQ': ('S2',), 'SHUF': ('S2',), 'ORD': ('S1', 'S2', 'S3')}
TOL = 1e-12
TWIN = 'PRC_TWIN'


def path(fam, cid):
    if fam == 'ORD':
        return [f'ORD@{a}x{b}' for a, b in LADDERS[cid]]
    return [f'{fam}@{g}' for g in GRID]


def select(audit, cands, budget, z=2.0):
    pick = None
    for c in cands:
        if all(audit[c][k][0] + z * audit[c][k][1] <= lim for k, lim in budget.items()):
            pick = c
        else:
            break
    return pick


def spend_from_means(nm, c, key):
    a, b = nm[c], nm[TWIN]
    if key.endswith('_acc'):
        return b[key] - a[key]
    return a[key] - b[key]


def total_from_means(nm, c):
    return (nm[c]['inc_nll'] + nm[c]['occ_nll']) - (nm[TWIN]['inc_nll'] + nm[TWIN]['occ_nll'])


def replay(rec, cid):
    out, mm = {}, []
    sel_mine = {}
    for fam, buds in FAMILY_BUDGETS.items():
        cands = path(fam, cid)
        missing = [c for c in cands if c not in rec['audit_spend_over_twin'].get(fam, {})]
        if missing:
            mm.append(f'{fam}: audit lacks {missing}')
            continue
        sel_mine[fam] = {b: select(rec['audit_spend_over_twin'][fam], cands, BUDGETS[b]) for b in buds}
        for b in buds:
            if sel_mine[fam][b] != (rec['selection'].get(fam) or {}).get(b):
                mm.append(f'selection {fam} {b}: mine {sel_mine[fam][b]} recorded {(rec["selection"].get(fam) or {}).get(b)}')
    out['selection_mine'] = sel_mine
    aliases = {}
    for fam, s in sel_mine.items():
        for b in sorted(s):
            earlier = [x for x in sorted(s) if x < b and s[x] is not None and s[x] == s[b]]
            if earlier:
                aliases[f'{fam}_{b}'] = f'ALIAS_OF_{fam}_{earlier[0]}'
    out['aliases'] = aliases
    nm = rec['nested_means']
    worst = 0.0
    for fam in FAMILY_BUDGETS:
        for c, rows in rec['audit_spend_over_twin'].get(fam, {}).items():
            for k, (est, se) in rows.items():
                worst = max(worst, abs(est - spend_from_means(nm, c, k)))
    out['max_abs_audit_estimate_minus_means'] = worst
    if worst > TOL:
        mm.append(f'audit estimates differ from the nested means by {worst}')
    s2 = sel_mine.get('PRC', {}).get('S2')
    br = rec.get('brackets') or {}
    if s2 is None:
        exp = {f: 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH' for f in ('HQ', 'SHUF', 'ORD')}
        if br.get('brackets') != exp:
            mm.append(f'brackets: mine {exp} recorded {br.get("brackets")}')
        out['brackets_mine'] = exp
    else:
        target = total_from_means(nm, s2)
        out['target_mine'] = target
        if br.get('target_total_nll_spend') is None or abs(br['target_total_nll_spend'] - target) > TOL:
            mm.append(f'bracket target mine {target} recorded {br.get("target_total_nll_spend")}')
        bm, sw = {}, 0.0
        for fam in ('HQ', 'SHUF', 'ORD'):
            cands = path(fam, cid)
            sp = {c: total_from_means(nm, c) for c in cands}
            for c, x in (br.get('spend', {}).get(fam) or {}).items():
                sw = max(sw, abs(x - sp[c]))
            bm[fam] = next((c for c in cands if sp[c] >= target), 'CONTROL_DID_NOT_REACH_SPEND')
            if bm[fam] != br.get('brackets', {}).get(fam):
                mm.append(f'bracket {fam}: mine {bm[fam]} recorded {br.get("brackets", {}).get(fam)}')
        out['brackets_mine'] = bm
        out['max_abs_bracket_spend_minus_means'] = sw
        if sw > TOL:
            mm.append(f'bracket spends differ from the nested means by {sw}')
    f4 = rec.get('F4')
    look = rec.get('nested_lookup_pair') or {}
    lw = 0.0
    for c, r in look.items():
        if 'auc' in r and 'auc_pair' in nm.get(c, {}):
            lw = max(lw, abs(r['auc'] - nm[c]['auc_pair']))
        if 'diff_twin_minus_entry' in r:
            lw = max(lw, abs(r['diff_twin_minus_entry'] - (look[TWIN]['auc'] - r['auc'])))
    out['max_abs_lookup_consistency'] = lw
    if lw > TOL:
        mm.append(f'nested LOOKUP entries inconsistent by {lw}')
    if s2 is not None:
        if f4 is None or f4.get('entry') != s2:
            mm.append(f'F4 entry mine {s2} recorded {(f4 or {}).get("entry")}')
        else:
            d = look[TWIN]['auc'] - look[s2]['auc']
            p = bool(d - 2 * f4['se_boot'] > 0)
            out['F4_mine'] = {'diff': d, 'se_boot_recorded': f4['se_boot'], 'pass_2se': p, 'margin': d - 2 * f4['se_boot']}
            if abs(d - f4['diff_twin_minus_prc']) > TOL or p != f4.get('pass_2se') or abs(f4['se_boot'] - look[s2]['se_boot']) > 0:
                mm.append(f'F4 mine {out["F4_mine"]} recorded {f4}')
    return out, mm


def run(tag):
    qp = V.RESULTS / 'decisions' / 'QUAL_DECISION.json'
    qdec = json.loads(qp.read_text())
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    for v in (0, 1):
        p = V.RESULTS / 'fa' / f'SELECTION_V{v}.json'
        if not p.is_file():
            continue
        rec = json.loads(p.read_text())
        cid = rec['capacity_id']
        proc = []
        if rec['budgets'] != BUDGETS:
            proc.append('budgets differ from the registered table')
        if tuple(rec['lambda_grid']) != GRID:
            proc.append('lambda grid differs')
        if [tuple(x) for x in rec['ladder']] != list(LADDERS[cid]):
            proc.append('ORD ladder differs from the registered ladder of the capacity')
        if (rec['construction'], cid, list(rec['capacity'])) != (qdec.get('construction'), qdec.get('capacity_id'), qdec.get('capacity')):
            proc.append('construction / capacity differ from the committed qualification decision')
        sx = V.FA_CACHE / f'V{v}' / 'SEXSIDE.pkl'
        sx_h = V.file_sha(sx) if sx.is_file() else None
        sx_rec = (V.FA_CACHE / f'V{v}' / 'SEXSIDE.sha256').read_text().strip() if sx.is_file() else None
        if not (sx_h and sx_h == sx_rec == rec.get('sexside_cache_sha256')):
            proc.append('SEX-side cache hash differs from its record or the selection record')
        task = pickle.loads((V.SRC_FA_CACHE / f'V{v}' / 'TASK.pkl').read_bytes())
        if {k: int(rec['T_fixed'][k]) for k in ('inc', 'occ')} != {k: int(task['T'][k]) for k in ('inc', 'occ')}:
            proc.append('task T* differ from the source task cache')
        if sx.is_file():
            ss = pickle.loads(sx.read_bytes())
            if {k: int(rec['T_fixed'][k]) for k in ('sex', 'occ_s')} != {k: int(ss['T'][k]) for k in ('sex', 'occ_s')}:
                proc.append('SEX-side T* differ from the SEX-side cache')
        out, mm = replay(rec, cid)
        record['per'][f'V{v}'] = {**out, 'mismatches': mm, 'procedural_findings': proc}
        checks.append({'id': f'V5A.SELECTION_V{v}__{tag}',
                       'what': f'P1 version {v}: registered budgets / grid / ladder / decision, selection rule replay in registered path '
                               f'order, aliases, audit estimates vs nested means, bracket target / spends / brackets, F4 decision, '
                               f'LOOKUP entry consistency, SEX-side cache hash and T* (recorded numbers; SEs, AUCs and the bootstrap '
                               f'SE themselves are not recomputed: no person-level P1 output exists)',
                       'inputs': V.inputs_sha([p, qp, sx]), 'result': {**out, 'mismatches': mm},
                       'numerical_status': 'PASS' if not mm else 'FAIL', 'procedural_status': 'PASS' if not proc else 'FAIL',
                       'procedural_findings': proc})
    checks.append({'id': f'V5A.P1_PERSON_LEVEL_REPLAY__{tag}',
                   'what': 'independent recomputation of the P1 nested spends, household SEs, nested held-out LOOKUP AUCs and the F4 '
                           'household-bootstrap SE', 'inputs': {},
                   'result': {'reason': 'P1 saved no person-level nested outputs (per-entry metrics, held-out LOOKUP scores); only '
                                        'aggregates are in SELECTION_V<v>.json, so these numbers can be checked for internal '
                                        'consistency (V5A.SELECTION_*) but not recomputed without a re-fit'},
                   'numerical_status': 'NOT_RUN', 'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V5A_SELECTION__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], c['result'].get('mismatches'), c.get('procedural_findings'))
    return 0


def selftest():
    ok = {}
    cid = 'C64'
    nm = {TWIN: {'inc_nll': 0.5, 'occ_nll': 1.2, 'inc_acc': 0.75, 'occ_acc': 0.5, 'occ_brier_men': 0.6, 'occ_brier_women': 0.55,
                 'auc_pair': 0.63}}
    audit = {}
    for fam in FAMILY_BUDGETS:
        audit[fam] = {}
        for i, c in enumerate(path(fam, cid)):
            dlt = 0.001 * (i + 1) if fam != 'ORD' else 0.003 * (i + 1)
            nm[c] = {'inc_nll': 0.5 + dlt / 4, 'occ_nll': 1.2 + dlt, 'inc_acc': 0.75 - dlt / 10, 'occ_acc': 0.5 - dlt / 10,
                     'occ_brier_men': 0.6 + dlt / 10, 'occ_brier_women': 0.55 + dlt / 10, 'auc_pair': 0.63 - dlt}
            audit[fam][c] = {k: [spend_from_means(nm, c, k), 0.0005] for k in BUDGETS['S1']}
    sel = {fam: {b: select(audit[fam], path(fam, cid), BUDGETS[b]) for b in buds} for fam, buds in FAMILY_BUDGETS.items()}
    look = {c: {'auc': nm[c]['auc_pair']} for c in nm}
    for c in look:
        if c != TWIN:
            look[c].update(diff_twin_minus_entry=look[TWIN]['auc'] - look[c]['auc'], se_boot=0.001)
    s2 = sel['PRC']['S2']
    target = total_from_means(nm, s2)
    br = {'brackets': {}, 'spend': {}, 'target_total_nll_spend': target, 'prc_s2': s2}
    for fam in ('HQ', 'SHUF', 'ORD'):
        sp = {c: total_from_means(nm, c) for c in path(fam, cid)}
        br['spend'][fam] = sp
        br['brackets'][fam] = next((c for c in path(fam, cid) if sp[c] >= target), 'CONTROL_DID_NOT_REACH_SPEND')
    f4 = {'entry': s2, 'diff_twin_minus_prc': look[TWIN]['auc'] - look[s2]['auc'], 'se_boot': 0.001,
          'pass_2se': (look[TWIN]['auc'] - look[s2]['auc']) - 0.002 > 0}
    rec = {'audit_spend_over_twin': audit, 'selection': sel, 'nested_means': nm, 'brackets': br, 'F4': f4, 'nested_lookup_pair': look}
    out, mm = replay(rec, cid)
    ok['consistent_record_passes'] = mm == []
    ok['selection_nonempty'] = s2 is not None
    r2 = json.loads(json.dumps(rec))
    r2['selection']['PRC']['S2'] = path('PRC', cid)[0] if s2 != path('PRC', cid)[0] else path('PRC', cid)[1]
    ok['flags_wrong_selection'] = bool(replay(r2, cid)[1])
    r3 = json.loads(json.dumps(rec))
    r3['audit_spend_over_twin']['HQ'][path('HQ', cid)[2]]['occ_nll'][0] += 1e-9
    ok['flags_audit_vs_means'] = bool(replay(r3, cid)[1])
    r4 = json.loads(json.dumps(rec))
    r4['F4']['pass_2se'] = not r4['F4']['pass_2se']
    ok['flags_f4_flip'] = bool(replay(r4, cid)[1])
    r5 = json.loads(json.dumps(rec))
    r5['brackets']['brackets']['ORD'] = 'CONTROL_DID_NOT_REACH_SPEND' if br['brackets']['ORD'] != 'CONTROL_DID_NOT_REACH_SPEND' else path('ORD', cid)[0]
    ok['flags_bracket'] = bool(replay(r5, cid)[1])
    # a failing first candidate gives None; a failure stops the walk even if a later candidate passes
    a2 = {c: {k: [0.0, 0.0] for k in BUDGETS['S1']} for c in path('PRC', cid)}
    a2[path('PRC', cid)[1]]['occ_nll'] = [1.0, 0.0]
    ok['walk_stops_at_first_failure'] = select(a2, path('PRC', cid), BUDGETS['S2']) == path('PRC', cid)[0]
    a2[path('PRC', cid)[0]]['occ_nll'] = [1.0, 0.0]
    ok['first_failure_gives_none'] = select(a2, path('PRC', cid), BUDGETS['S2']) is None
    ok['ord_path_in_registered_order'] = path('ORD', 'C128')[:2] == ['ORD@16x64', 'ORD@12x48']
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {k: bool(x) for k, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())

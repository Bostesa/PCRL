"""VER check 7a: nomination replay (SELECTION_RULE.md section 6 / RULES.json nomination) with the verifier's own code.

Inputs, all recomputed or derived independently:
- utility points and the 28-condition whole-release reserve: the verifier's own tables (records/VER_UTILITY_CONFIGS.json);
- 8-cell pair benefits, recipient cells and stability labels: rebuilt from GRID.json / BOOT.npz of exactly the nomination's banks
  (v_grid.collect_cells / build_labels, own code);
- matched component: a privacy-stage configuration's witness (OCC.json 'witness') with the same anchor; an income-branch complete
  release: the same occupation output with H3 income (RELEASE.json 'fixed_h3_comparison', e.g. B/B2_SINGLE@H3); a task-only
  configuration: SUCC;
- runnable package / six raw fields: a deployment-check record (results/<campaign>/deploy/DEPLOYMENT_CHECKS*.json) showing both
  seeds bitwise with every refusal (the campaign's record; the verifier's own harness re-checked a sample, check 8);
- alias: identical MODEL_SELECT wires (both tasks, both seeds) with a required comparator, from the evaluator wire hashes that the
  verifier recomputed and matched in check 3.
Gates in order: NOT_CONSIDERED_ONE_SEED_ONLY, NOT_ADMISSIBLE, ALIAS_OF_<X>, UTILITY_POINT_INELIGIBLE, RECIPIENT_POINT_FAILED,
RESERVE_NOT_MET, BENEFIT_NOT_STABLE, NO_MATCHED_COMPONENT_COMPARISON, QUALIFYING; score = min over the 6 required comparators of the
8-cell mean pair benefit; near-tie 0.005; tie-breaks leximin 28-ratio vector, fewer fitted components, earliest seed-0 ledger entry.
Compared label by label with results/<campaign>/NOMINATION.json. Light (aggregate numbers only).
"""
from __future__ import annotations

import csv
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402
import v_grid as G  # noqa: E402  (verifier's own grid code)

RULES = V.load_rules()
REQ = RULES['comparators']['required']
CELLS = G.CELLS


def tables():
    db = json.loads((V.RECORDS / 'VER_UTILITY_CONFIGS.json').read_text())['entries']
    out = {}
    for key, e in db.items():
        fam, cfg, s, anchor = key.split('/')
        out.setdefault(f'{fam}/{cfg}@{anchor}', {})[int(s[1:])] = e['table']
    return out


def record(fam, cfg, s, anchor):
    d = V.RUNS / fam / cfg / f'S{s}'
    p = d / ('RELEASE.json' if anchor == 'COMPLETE' else 'OCC.json')
    return json.loads(p.read_text()) if p.is_file() else None


def eval_wires(fam, cfg, s, anchor):
    p = V.RUNS / fam / cfg / f'S{s}' / f'EVAL_MS_{anchor}.json'
    return json.loads(p.read_text())['wire_sha256'] if p.is_file() else None


def first_entry(fam, cfg):
    rows = list(csv.DictReader(open(V.RES / 'EXPERIMENT_LEDGER.csv')))
    e = [int(r['entry']) for r in rows if r['family'] == fam and r['config_id'] == cfg and r['seed'] == '0'
         and r['kind'] not in ('reader_grid', 'reference', 'decision')]
    return min(e) if e else None


def deploy_status():
    """{(family, config, anchor): {seed: ok}} from every deployment-check record."""
    out = {}
    for p in sorted(V.RES.glob('deploy/DEPLOYMENT_CHECKS*.json')):
        d = json.loads(p.read_text())
        for name, r in d.get('packages', {}).items():
            occ = r.get('occupation', '')
            if ':' not in occ:
                continue
            fam, cfg = occ.split(':')
            bit = all(all(v.values()) for v in r.get('bitwise_equal_to_evaluated_DEV_REPORT_release', {}).values())
            ref = all(str(v).startswith('refused') for v in r.get('refusals', {}).values())
            key = (fam, cfg, r.get('anchor'))
            out.setdefault(key, {}).setdefault(int(r['seed']), []).append({'file': p.name, 'ok': bit and ref})
    return out


def ref_wires():
    out = {}
    for s in (0, 1):
        ev = json.loads((V.RUNS / 'REF' / f'S{s}' / 'EVAL_MS_REFERENCES.json').read_text())
        for name in REQ:
            out.setdefault(name, {})[s] = ev[name]['wire_sha256']
    return out


def matched_component(fam, cfg, anchor):
    r0 = record(fam, cfg, 0, anchor) or {}
    if fam == 'PROT':
        return f"{r0.get('witness')}@{anchor}" if r0.get('witness') else None
    if anchor == 'COMPLETE':
        txt = str(r0.get('fixed_h3_comparison', ''))
        return f'{txt.split()[0]}@H3' if txt else None
    return 'SUCC'


def evaluate(cands, banks):
    T = tables()
    per_bank, per_bank_lc, _ = G.collect_cells()
    L = G.build_labels(per_bank, per_bank_lc, banks)
    dep = deploy_status()
    RW = ref_wires()
    out = {}
    for name in cands:
        fam_cfg, anchor = name.split('@')
        fam, cfg = fam_cfg.split('/')
        row = {'name': name}
        per = T.get(name, {})
        evaluated = [s for s in (0, 1) if (V.RUNS / fam / cfg / f'S{s}' / f'EVAL_MS_{anchor}.json').is_file()]
        row['seeds_evaluated'] = evaluated
        recs = [record(fam, cfg, s, anchor) for s in (0, 1)]
        row['n_components'] = sum(int((r or {}).get('n_model_fits') or 0) + int((r or {}).get('n_calibrator_fits') or 0) for r in recs)
        row['entry'] = first_entry(fam, cfg)
        if len(evaluated) < 2:
            row['label'] = 'NOT_CONSIDERED_ONE_SEED_ONLY'
            out[name] = row
            continue
        if set(per) != {0, 1}:
            row['label'] = 'VERIFIER_TABLE_MISSING'
            out[name] = row
            continue
        g = L.get(name, {})
        grid_ok = all(c in g and 'mean' in g[c] for c in REQ)
        d = dep.get((fam, cfg, anchor), {})
        runnable = all(any(x['ok'] for x in d.get(s, [])) for s in (0, 1))
        row['admissibility'] = {'grid_all_required': grid_ok, 'missing_grid': [c for c in REQ if not (c in g and 'mean' in g[c])],
                                'runnable_package_both_seeds': runnable, 'deploy_records': {str(s): d.get(s, []) for s in (0, 1)}}
        alias = None
        for comp in REQ:
            if all(eval_wires(fam, cfg, s, anchor) == RW[comp][s] for s in (0, 1)):
                alias = comp
        row['alias_of'] = alias
        res = {s: V.reserve_seed(per[s], s, RULES, 5306) for s in (0, 1)}
        row['ratio_vector_28'] = sorted(c['ratio'] for s in (0, 1) for c in res[s].values())
        row['reserve'] = V.reserve_status({s: all(c['met'] for c in res[s].values()) for s in (0, 1)})
        row['utility_point_failed'] = [f's{s}|{k}' for s in (0, 1) for k, c in res[s].items() if not c['point_pass']]
        mc = matched_component(fam, cfg, anchor)
        row['matched_component'] = mc
        row['matched_component_present'] = bool(mc) and mc in g and 'mean' in g[mc]
        if grid_ok:
            row['score'] = min(g[c]['mean'] for c in REQ)
            row['per_comparator'] = {c: {'mean': g[c]['mean'], 'benefit': g[c]['benefit'], 'recipient_income': g[c]['recipient_income'],
                                         'recipient_occupation': g[c]['recipient_occupation'],
                                         'income_max': g[c]['recipient_income_max'], 'occupation_max': g[c]['recipient_occupation_max']}
                                     for c in REQ}
            row['recipient_point_failed'] = sorted(f'{c}|{v}' for c in REQ for v, k in (('income', 'recipient_income'), ('occupation', 'recipient_occupation'))
                                                   if g[c][k] == 'RECIPIENT_POINT_FAILED')
            row['benefit_not_stable'] = {c: g[c]['benefit'] for c in REQ if g[c]['benefit'] != 'STABLE_POSITIVE'}
        if not (grid_ok and runnable):
            lab = 'NOT_ADMISSIBLE'
        elif alias:
            lab = f'ALIAS_OF_{alias}'
        elif row['utility_point_failed']:
            lab = 'UTILITY_POINT_INELIGIBLE'
        elif row['recipient_point_failed']:
            lab = 'RECIPIENT_POINT_FAILED'
        elif row['reserve'] != 'RESERVE_MET':
            lab = 'RESERVE_NOT_MET'
        elif row['benefit_not_stable']:
            lab = 'BENEFIT_NOT_STABLE'
        elif not row['matched_component_present']:
            lab = 'NO_MATCHED_COMPONENT_COMPARISON'
        else:
            lab = 'QUALIFYING'
        row['label'] = lab
        out[name] = row
    qual = [r for r in out.values() if r.get('label') == 'QUALIFYING']
    if qual:
        best = max(r['score'] for r in qual)
        near = [r for r in qual if r['score'] >= best - RULES['nomination']['near_tie_width']]

        def cmp(a, b):
            for x, y in zip(a['ratio_vector_28'], b['ratio_vector_28']):
                if abs(x - y) > 1e-9:
                    return -1 if x > y else 1
            if a['n_components'] != b['n_components']:
                return -1 if a['n_components'] < b['n_components'] else 1
            return -1 if (a['entry'] or 10 ** 9) < (b['entry'] or 10 ** 9) else 1
        win = sorted(near, key=functools.cmp_to_key(cmp))[0]
        for r in qual:
            r['label'] = 'NOMINATED_FINALIST' if r is win else 'QUALIFYING_NOT_CHOSEN'
        decision = {'outcome': 'FINALIST_NOMINATED', 'nominated': win['name']}
    else:
        decision = {'outcome': 'NO_QUALIFYING_CONFIGURATION', 'nominated': None}
    return out, decision


def main():
    p = V.RES / 'NOMINATION.json'
    theirs = json.loads(p.read_text())
    cands, banks = list(theirs['candidates']), list(theirs['banks'])
    mine, decision = evaluate(cands, banks)
    tconf = {c['config_id']: c for c in theirs['configurations']}
    cmp_rows = {}
    for name in cands:
        m, t = mine.get(name, {}), tconf.get(name, {})
        row = {'label_mine': m.get('label'), 'label_theirs': t.get('label'), 'label_equal': m.get('label') == t.get('label')}
        if 'score' in m or t.get('score_min_required_mean_benefit') is not None:
            ts = t.get('score_min_required_mean_benefit')
            row.update({'score_mine': m.get('score'), 'score_theirs': ts,
                        'score_abs_diff': abs(m['score'] - ts) if (m.get('score') is not None and ts is not None) else None})
            row['reserve_mine'] = m.get('reserve')
            row['reserve_theirs'] = (t.get('reserve') or {}).get('status') if isinstance(t.get('reserve'), dict) else t.get('reserve')
            row['matched_component_mine'] = m.get('matched_component')
            mct = t.get('matched_component')
            row['matched_component_theirs'] = mct.get('config_id') if isinstance(mct, dict) else mct
            row['recipient_point_failed_mine'] = m.get('recipient_point_failed')
            row['recipient_point_failed_theirs'] = t.get('recipient_point_failed')
            row['benefit_not_stable_equal'] = (m.get('benefit_not_stable') == t.get('benefit_not_stable'))
            row['utility_point_failed_mine'] = m.get('utility_point_failed')
        cmp_rows[name] = row
    two_seed_grid = [n for n, r in cmp_rows.items() if r.get('score_mine') is not None]
    rec = {'check': 'VER_NOMINATION_REPLAY', 'utc': V.utc(), 'nomination_file_sha256': V.file_sha256(p), 'nomination_utc': theirs.get('utc'),
           'banks': banks, 'n_candidates': len(cands), 'decision_mine': decision,
           'decision_theirs': {'outcome': theirs.get('outcome'), 'nominated': theirs.get('nominated'), 'dev_report': theirs.get('dev_report')},
           'label_agreement': f"{sum(r['label_equal'] for r in cmp_rows.values())}/{len(cmp_rows)}",
           'two_seed_grid_candidates': two_seed_grid,
           'max_score_abs_diff': max((r['score_abs_diff'] for r in cmp_rows.values() if r.get('score_abs_diff') is not None), default=None),
           'comparison': cmp_rows, 'mine': mine}
    rec['status'] = 'PASS' if (all(r['label_equal'] for r in cmp_rows.values()) and decision['outcome'] == theirs.get('outcome')
                               and (rec['max_score_abs_diff'] or 0) < 1e-12) else 'ATTENTION'
    V.write_record('VER_NOMINATION_REPLAY', rec)
    print(json.dumps({k: rec[k] for k in ('status', 'decision_mine', 'decision_theirs', 'label_agreement', 'max_score_abs_diff')}, indent=1))
    for n in two_seed_grid:
        r = cmp_rows[n]
        print(n, r['label_mine'], '| score', round(r['score_mine'], 5), '| reserve', r['reserve_mine'], '| rp_fail', len(r['recipient_point_failed_mine']),
              '| mc', r['matched_component_mine'], r['matched_component_mine'] == r['matched_component_theirs'], '| bns_eq', r['benefit_not_stable_equal'])
    for n, r in cmp_rows.items():
        if not r['label_equal']:
            print('LABEL MISMATCH', n, r['label_mine'], r['label_theirs'])


if __name__ == '__main__':
    main()

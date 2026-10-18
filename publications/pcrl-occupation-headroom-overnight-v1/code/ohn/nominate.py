"""Whole-release nomination with the frozen rule (ohn.rules.nominate), from stored MODEL_SELECT evaluations, the development reader
grid and the deployment checks. No label read.

    python -m ohn.nominate --banks BANK_ID [BANK_ID ...] --candidates NAME [NAME ...] [--matched NAME=COMPONENT ...]
        -> results/<campaign>/NOMINATION.json (write-once) and NOMINATION_TABLE.csv, plus a ledger decision row

Candidate names follow the bank-spec convention '<FAMILY>/<CONFIG>@<ANCHOR>' ('@COMPLETE' for income-branch complete releases).
For each candidate: evals = the seed-0 / seed-1 EVAL_MS_<ANCHOR>.json (complete release); grid = the 8-cell (candidate, comparator)
entries for every REQUIRED comparator, merged across the given banks; matched_component = the (candidate, component) grid entry
(default: a privacy-stage configuration's witness as recorded in its OCC.json 'witness'; SUCC for a task-only configuration);
runnable_package = a deployment check record shows both seeds bitwise with every refusal; six_raw_fields from OCC.json /
RELEASE.json; alias_of = a required comparator with identical wires on both tasks and seeds; fit counts summed over both seeds;
entry = the ledger entry of the seed-0 evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json

from . import common as C
from . import devrun as D
from . import grid_summary as GS
from . import ledger as L
from . import rules as RU


def _parse(name):
    fam_cfg, anchor = name.split('@')
    fam, cfg = fam_cfg.split('/')
    return fam, cfg, anchor


def _record(fam, cfg, seed, anchor):
    d = D.occ_dir(fam, cfg, seed)
    if anchor == 'COMPLETE':
        return json.loads((d / 'RELEASE.json').read_text())
    return json.loads((d / 'OCC.json').read_text())


def _deploy_ok(fam, cfg, anchor):
    """Both seeds' packages checked bitwise with every refusal, in any results/<campaign>/deploy*/DEPLOYMENT_CHECKS*.json."""
    ok = {s: False for s in C.SEEDS}
    for p in C.RESULTS.glob('**/DEPLOYMENT_CHECKS*.json'):
        rec = json.loads(p.read_text())
        for r in rec.get('packages', {}).values():
            occ = r.get('occupation', '')
            if occ == f'{fam}:{cfg}' and r.get('anchor') == anchor:
                bit = all(all(v.values()) for v in r['bitwise_equal_to_evaluated_DEV_REPORT_release'].values())
                ref = all(str(v).startswith('refused') for v in r['refusals'].values())
                ok[int(r['seed'])] = ok[int(r['seed'])] or (bit and ref)
    return all(ok.values())


def _alias(name, fam, cfg, anchor):
    """A required comparator with identical MODEL_SELECT wires on both tasks and both seeds."""
    for comp in RU.REQUIRED:
        same = True
        for s in C.SEEDS:
            ref = D.reference_release(comp, s, 'MODEL_SELECT')
            if anchor == 'COMPLETE':
                rel = D.load_complete(fam, cfg, s, roles=('MODEL_SELECT',))[0]['MODEL_SELECT']
            else:
                rel = {'income': D.anchor_income(anchor, s, 'MODEL_SELECT'),
                       'occupation': D.load_occupation(fam, cfg, s, roles=('MODEL_SELECT',))[0]['MODEL_SELECT']}
            if any(C.wire_sha(rel[t]) != C.wire_sha(ref[t]) for t in C.TASKS):
                same = False; break
        if same:
            return comp
    return None


def build_table(banks, candidates, matched):
    entries, _, _ = GS.merge(banks)
    first = {}
    for r in L.rows():
        if r['seed'] == '0' and r['kind'] not in ('reader_grid', 'reference', 'decision'):
            first.setdefault((r['family'], r['config_id']), int(r['entry']))
    table = []
    for name in candidates:
        fam, cfg, anchor = _parse(name)
        evals, recs = {}, {}
        for s in C.SEEDS:
            p = D.occ_dir(fam, cfg, s) / f'EVAL_MS_{anchor}.json'
            if p.is_file():
                evals[s] = json.loads(p.read_text()); recs[s] = _record(fam, cfg, s, anchor)
        grid = {comp: entries[(name, comp)] for comp in RU.REQUIRED if (name, comp) in entries}
        comp_name = matched.get(name)
        if comp_name is None:
            w = (recs.get(0) or {}).get('witness')
            comp_name = f'{w}@{anchor}' if w else 'SUCC'
        mc = {'config_id': comp_name, 'grid': entries.get((name, comp_name))} if (name, comp_name) in entries else None
        kinds = {r.get('kind') for r in recs.values()}
        runnable = _deploy_ok(fam, cfg, anchor)
        table.append({'config_id': name, 'family': fam, 'kind': 'control' if 'control' in kinds else next(iter(kinds), None),
                      'entry': first.get((fam, cfg), 10 ** 9), 'evals': evals, 'grid': grid, 'matched_component': mc,
                      'runnable_package': runnable,
                      # computed from the six permitted raw fields = verified by the raw-field deployment check itself (the
                      # documented wrapper accepts exactly AGEP, WKHP, COW, SCHL, MAR, POBP and reproduced the evaluated release)
                      'six_raw_fields': runnable,
                      'alias_of': _alias(name, fam, cfg, anchor) if len(evals) == 2 else None,
                      'n_model_fits': sum(int(r.get('n_model_fits') or 0) for r in recs.values()),
                      'n_calibrator_fits': sum(int(r.get('n_calibrator_fits') or 0) for r in recs.values())})
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--candidates', nargs='+', required=True)
    ap.add_argument('--matched', nargs='*', default=[]); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    matched = dict(m.split('=', 1) for m in a.matched)
    table = build_table(a.banks, a.candidates, matched)
    res = RU.nominate(table)
    if a.dry_run:                                      # plumbing test only: nothing written, no ledger row
        print(json.dumps({'dry_run': True, 'outcome': res['outcome'], 'labels': {r['config_id']: (r['label'], r.get('flags'),
              (r.get('outcomes') or {}).get('reader_stability')) for r in res['configurations']}}, indent=1, default=str))
        return
    res.update({'utc': C.now(), 'banks': a.banks, 'candidates': a.candidates,
                'note': 'frozen rule (RULES.json); development selection on adaptively used MODEL_SELECT data'})
    C.write_json_x(C.RESULTS / 'NOMINATION.json', res)
    rows = [{'configuration': r['config_id'], 'label': r['label'], 'score_min_required_mean_benefit': r.get('score_min_required_mean_benefit'),
             **(r.get('outcomes') or {}), 'flags': ';'.join(r.get('flags', [])),
             'recipient_point_failed': ';'.join(r.get('recipient_point_failed', []))[:300],
             'benefit_not_stable': json.dumps(r.get('benefit_not_stable', {}))[:300],
             'reserve_min_ratio': (r.get('reserve') or {}).get('min_ratio')} for r in res['configurations']]
    if rows:
        with open(C.RESULTS / 'NOMINATION_TABLE.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    L.append({'family': 'SELECT', 'version': 'nomination', 'config_id': 'NOMINATION', 'kind': 'decision', 'budget_class': 'NONE',
              'counts': 0, 'decision': f"{res['outcome']}: {res['nominated']}", 'notes': f"DEV_REPORT {res['dev_report']}"})
    print(json.dumps({'outcome': res['outcome'], 'nominated': res['nominated'],
                      'labels': {r['config_id']: r['label'] for r in res['configurations']}}, indent=1))


if __name__ == '__main__':
    main()

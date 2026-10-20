"""V2b record consistency of draw-averaged evaluations (label-free; reads the EVAL_MS files only): for randomized releases nrm.evaluate
copies the FIRST draw's utility block and overwrites value / se / se_mc / draw_values; this audit lists every derived field that
still describes draw 0 instead of the draw-averaged value: per clause / guard 'point_pass' (vs value <= cap), 'margin' (vs cap -
value), 'normalized' (vs value / cap), 'margin_se_units' (vs margin / se), and the block-level 'point_pass_12' / 'guard_pass'.
The decision fields (value, se, se_mc, the point summary 'point' and the reserve) are checked separately by v2_utility.

    python3 .../v2b_stale_fields.py --tag V1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def audit(u):
    stale = []
    allc = [('clause', c) for c in u['clauses']] + [('guard', g) for g in u['guards']]
    for kind, c in allc:
        name = f"{c['task']}|{c['reference']}|{c['metric']}" if kind == 'clause' else f"guard|{c['group']}"
        val, cap, se = float(c['value']), float(c['cap']), float(c.get('se', float('nan')))
        if 'point_pass' in c and bool(c['point_pass']) != (val <= cap):
            stale.append((name, 'point_pass'))
        if 'margin' in c and abs(float(c['margin']) - (cap - val)) > 1e-12:
            stale.append((name, 'margin'))
        if 'normalized' in c and abs(float(c['normalized']) - val / cap) > 1e-9:
            stale.append((name, 'normalized'))
        if 'margin_se_units' in c and se > 0 and abs(float(c['margin_se_units']) - (cap - val) / se) > 1e-6:
            stale.append((name, 'margin_se_units'))
    p12 = all(float(c['value']) <= float(c['cap']) for c in u['clauses'])
    gp = all(float(g['value']) <= float(g['cap']) for g in u['guards'])
    if 'point_pass_12' in u and bool(u['point_pass_12']) != p12:
        stale.append(('utility', 'point_pass_12'))
    if 'guard_pass' in u and bool(u['guard_pass']) != gp:
        stale.append(('utility', 'guard_pass'))
    return stale


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    files = sorted(V.RUNS.glob('*/*/EVAL_MS*.json'))
    res, inputs = {}, {}
    status_flips = []
    for p in files:
        d = json.loads(p.read_text())
        inputs[V.rel(p)] = V.file_sha(p)
        for vs, x in d.get('versions', {}).items():
            if x.get('missing'):
                continue
            st = audit(x['utility'])
            if st or int(x.get('draws', 1)) > 1:
                res[f"{d['family']}/{d['config_id']}/{p.name}/V{vs}"] = {'draws': x.get('draws'), 'stale_fields': [f'{a_}:{b}' for a_, b in st],
                                                                          'n_stale': len(st)}
                status_flips += [f"{d['family']}/{d['config_id']}/V{vs}:{n}" for n, f in st if f == 'point_pass']
    det_stale = {k: x for k, x in res.items() if x['draws'] in (None, 1) and x['n_stale']}
    rnd = {k: x for k, x in res.items() if x['draws'] not in (None, 1)}
    chk = {'id': f'V2B.EVAL_DERIVED_FIELDS__{a.tag}',
           'what': 'derived per-clause / guard fields of the evaluation records (point_pass, margin, normalized, margin_se_units, '
                   'point_pass_12, guard_pass) against their own draw-averaged values; deterministic releases must be consistent',
           'inputs': inputs,
           'result': {'n_files': len(files), 'randomized_versions': len(rnd),
                      'randomized_versions_with_stale_fields': sum(1 for x in rnd.values() if x['n_stale']),
                      'per_clause_point_pass_flips': status_flips, 'deterministic_inconsistent': det_stale, 'detail': res,
                      'reading': 'for randomized releases the derived fields describe draw 0 (copied block), not the draw average; '
                                 'value, se, se_mc, the point summary and the reserve are draw-averaged and are replayed by v2_utility'},
           'numerical_status': 'FAIL' if det_stale else ('ATTENTION' if any(x['n_stale'] for x in rnd.values()) else 'PASS'),
           'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps({k: chk['result'][k] for k in ('n_files', 'randomized_versions', 'randomized_versions_with_stale_fields',
                                                     'per_clause_point_pass_flips', 'deterministic_inconsistent')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

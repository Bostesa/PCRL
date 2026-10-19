"""Verifier stage V4 (label-free cross-check): class-support records of the size fits vs the verifier's own counts.

PCRL controlled training-size campaign (TSZ1). The verifier's refit records (records/V3_A1__*.json, written by v3_refit.py from
MODEL_FIT occupation labels of each subset read under operation 'verify') carry the subset's minimum occupation class count and the
minimum class count of the inner training sets; they are compared with FIT.json class_support of every refitted fit, and every
saved fit's class-support record is checked against the fixed rule (all five classes in the subset and in every inner training set;
no SUPPORT_FAILURE recorded).

    python3 .../v4_classcounts.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
SIZE = ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'runs' / 'SIZE'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    mine, files = {}, []
    for p in sorted(V.RECORDS.glob('V3_A1__*.json')):
        files.append(p)
        for c in json.loads(p.read_text())['checks']:
            o = c['observed']
            mine[o['fit_id']] = {'subset_min_class': o['subset_min_class'], 'inner_training_min_class': o['inner_training_min_class'],
                                 'check': c['id'], 'status': c['status']}
    rows, bad = {}, []
    for d in sorted(p for p in SIZE.iterdir() if p.is_dir()):
        if (d / 'SUPPORT_FAILURE.json').exists():
            bad.append(f'{d.name}: SUPPORT_FAILURE recorded')
            continue
        if not (d / 'FIT.json').is_file():
            continue
        rec = json.loads((d / 'FIT.json').read_text()); files.append(d / 'FIT.json')
        cs = rec['class_support']
        ok_rule = bool(cs['ok'] and cs['subset_min_class_count'] > 0 and cs['inner_training_min_class_count'] > 0)
        r = {'production_subset_min': cs['subset_min_class_count'], 'production_inner_min': cs['inner_training_min_class_count'], 'rule_ok': ok_rule}
        if d.name in mine:
            r['mine_subset_min'] = mine[d.name]['subset_min_class']; r['mine_inner_min'] = mine[d.name]['inner_training_min_class']
            if r['mine_subset_min'] != r['production_subset_min'] or r['mine_inner_min'] != r['production_inner_min']:
                bad.append(f'{d.name}: class counts differ')
        if not ok_rule:
            bad.append(f'{d.name}: class-support rule not met')
        rows[d.name] = r
    n_cmp = sum(1 for r in rows.values() if 'mine_subset_min' in r)
    chk = {'id': f'V4.RECORDS.CLASS_SUPPORT__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'class support of every saved A1_PO_16 fit: fixed rule (all five occupation classes in the subset and in every inner training '
                   'set, no SUPPORT_FAILURE) and, for every fit the verifier refitted, the subset and inner-training minimum class counts from '
                   'the verifier\'s own label read vs FIT.json',
           'inputs': V.inputs_sha(files), 'expected': 'rule met; counts equal', 'observed': {'fits': rows, 'compared_with_own_counts': n_cmp, 'problems': bad},
           'status': 'PASS' if not bad else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    import os
    V.write_record(f'V4_CLASSCOUNTS__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'n': len(rows), 'compared': n_cmp, 'problems': bad}))


if __name__ == '__main__':
    main()

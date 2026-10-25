"""Verifier check 1r (DRF1; label-free): the clean re-run of check 1 on committed code reproduces the first run exactly: every item
of DRF_VER_C1_V1 (v1, R1; both versions) has the same name, the same verifier value and the same result in DRF_VER_C1_V3.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check1r_reproduce.py --a UNIT_A --b UNIT_B --out FILE
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

OUT = HERE.parent / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='DRF_VER_C1_V1')
    ap.add_argument('--b', default='DRF_VER_C1_V3')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'files': {}}
    for rev in ('v1', 'R1'):
        for v in (0, 1):
            pa, pb = OUT / f'{args.a}_check1_{rev}_V{v}.json', OUT / f'{args.b}_check1_{rev}_V{v}.json'
            A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
            info['files'][f'{rev}_V{v}'] = {'a': V.sha256_file(pa), 'b': V.sha256_file(pb)}
            ia = {i['name']: (i.get('verifier'), i['result']) for i in A['items']}
            ib = {i['name']: (i.get('verifier'), i['result']) for i in B['items']}
            items.append(V.cmp_eq(f'{rev}_V{v}:same_item_names', sorted(ia), sorted(ib)))
            diff = [k for k in ia if k in ib and ia[k] != ib[k]]
            items.append(V.cmp_eq(f'{rev}_V{v}:identical_verifier_values_and_results', diff, []))
            items.append(V.cmp_eq(f'{rev}_V{v}:per_config_values_identical', A['per_config'], B['per_config']))
    out = {'schema': 'DRF1-verifier-check1r-v1', 'info': info, 'overall': V.summarize(items), 'summary': {'R': V.summarize(items)},
           'items': items}
    if args.out:
        V.dump(args.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall']}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

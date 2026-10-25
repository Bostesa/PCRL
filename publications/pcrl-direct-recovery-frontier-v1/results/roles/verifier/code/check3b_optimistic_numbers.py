"""Verifier check 3b (DRF1; label-free): the optimistic diagnostic's per-construction numbers of my check-3 outputs (bracket step,
spends, pair gain, single-view excesses, within-budget flag, per mode and version) against the readings' DECISIONS.json.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check3b_optimistic_numbers.py --mine OUT1 [OUT2 ...] --out FILE
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'inputs': {}}
    for m in a.mine:
        mp = Path(m)
        mine = json.loads(mp.read_text())
        dp = REPO / mine['info']['decisions']['path']
        rec = json.loads(dp.read_text())['decisions']['optimistic']
        tag = mine['info']['plan']['path'].split('/')[-1].replace('.json', '')
        info['inputs'][tag] = {'mine': str(mp), 'mine_sha256': V.sha256_file(mp), 'decisions': str(dp.relative_to(REPO)),
                               'decisions_sha256': V.sha256_file(dp)}
        for x, modes in mine['optimistic']['per_construction'].items():
            for mode, e in modes.items():
                rr = rec['per_construction'][x][mode]
                items.append(V.cmp_eq(f'{tag}:{x}:{mode}:label', e['label'], rr['label']))
                for v, b in e['by_version'].items():
                    r = rr['by_version'][str(v)]
                    pre = f'{tag}:{x}:{mode}:V{v}'
                    items.append(V.cmp_eq(pre + ':label', b['label'], r.get('label')))
                    items.append(V.cmp_eq(pre + ':bracket', b.get('bracket'), r.get('bracket')))
                    for k in ('spend', 'bracket_spend', 'gain_pair'):
                        items.append(V.cmp_num(pre + f':{k}', b.get(k), r.get(k), 1e-12))
                    for vw in ('income', 'occupation'):
                        items.append(V.cmp_num(pre + f':single_view_excess_{vw}', b.get('single_view_excess', {}).get(vw),
                                               r.get('single_view_excess', {}).get(vw), 1e-12))
                    items.append(V.cmp_eq(pre + ':within_budget', b.get('within_budget'), r.get('within_budget')))
        items.append(V.cmp_eq(f'{tag}:held_out_survives', sorted(mine['optimistic']['held_out_survives']), sorted(rec['held_out_survives'])))
    out = {'schema': 'DRF1-verifier-check3b-v1', 'info': info, 'overall': V.summarize(items), 'items': items,
           'summary': {'OPT': V.summarize(items)}}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall']}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

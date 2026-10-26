#!/usr/bin/env python3
"""RFA1 verifier, wave 2: invented-data test of vagg (aggregate screens, recipient gate, target screen, frontier / Q*) against the
frozen NRM1 functions (nrm.rules) and of vagg.auc_fast against vlib.auc. Label-free (vguard installed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vguard  # noqa: E402

vguard.install()
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vagg as A  # noqa: E402
import vlib as V  # noqa: E402
from nrm import rules as NR  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    bad, n, worst = 0, 0, 0.0
    for trial in range(400):
        mu, sc = rng.normal(0, 0.01), rng.choice([0.0005, 0.002, 0.008])
        e = {'cells': {c: float(mu + rng.normal(0, sc)) for c in A.REGISTERED},
             'cell_se': {c: float(abs(rng.normal(0.004, 0.002))) for c in A.REGISTERED}}
        if trial % 50 == 0:
            e = {'cells': {c: 0.0 for c in A.REGISTERED}, 'cell_se': {c: 0.0 for c in A.REGISTERED}}
        if trial % 37 == 0:
            e = {'cells': {c: 0.0125 + float(rng.normal(0, 1e-4)) for c in A.REGISTERED}, 'cell_se': {c: 1e-5 for c in A.REGISTERED}}
        a, b = A.aggregate(e), NR.aggregate(e)
        d = max([abs(a['mean'] - b['mean']), abs(a['se_agg'] - b['se_agg'])] +
                [abs(a['per_version'][v]['mean'] - b['per_version'][v]['mean']) for v in (0, 1)] +
                [abs(a['per_version'][v]['se_agg'] - b['per_version'][v]['se_agg']) for v in (0, 1)])
        worst = max(worst, d)
        same = (A.benefit_screen(e)['label'] == NR.benefit_screen(e)['label'] and A.target_screen(e)['label'] == NR.target_screen(e)['label']
                and A.recipient_screen(e)['label'] == NR.recipient_screen(e)['label'])
        n += 1
        bad += (d > 1e-15) or not same

    def costs(tot):
        return {v: {**{m: {'value': float(rng.normal(0, 0.001)), 'se': 0.0005} for m in A.SECONDARY},
                    'total|nll': {'value': tot[v], 'se': 0.0004}} for v in (0, 1)}
    fbad, labels = 0, {}
    for trial in range(300):
        path = ['START', 'P1', 'P2']
        cv = {'START': costs({0: 0.004, 1: 0.005}), 'P1': costs({0: float(rng.normal(0.001, 0.002)), 1: float(rng.normal(0.001, 0.002))}),
              'P2': costs({0: float(rng.normal(-0.001, 0.002)), 1: float(rng.normal(-0.001, 0.002))})}
        gv = {q: {'pair': {'cells': {c: float(rng.normal(0.004, 0.006)) for c in A.REGISTERED}, 'cell_se': {c: 0.003 for c in A.REGISTERED}}}
              for q in path}
        mine = A.frontier(cv, path, gv)
        theirs = NR.frontier_comparison({'id': 'CAND', 'costs_vs': cv, 'grid_vs': gv}, [{'id': q} for q in path])
        same = (mine['label'] == theirs['label'] and mine.get('bracket') == theirs.get('bracketing_control') and
                sorted(mine.get('secondary_more_spending') or []) == sorted(theirs.get('secondary_more_spending') or []))
        labels[theirs['label']] = labels.get(theirs['label'], 0) + 1
        fbad += not same
    abad = 0
    for _ in range(50):
        y = rng.integers(0, 2, 300)
        s = np.round(rng.normal(size=300) + 0.3 * y, 1)
        abad += abs(A.auc_fast(y, s) - V.auc(y, s)) > 1e-14
    res = {'aggregate_screen_trials': n, 'mismatches': int(bad), 'worst_abs_diff': worst, 'frontier_trials': 300,
           'frontier_mismatches': int(fbad), 'frontier_label_mix': labels, 'auc_mismatches': int(abad), 'refused': vguard.REFUSED}
    res['all_pass'] = bad == 0 and fbad == 0 and abad == 0
    (HERE.parent / 'out' / 'V2_TEST_VAGG.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    print(json.dumps(res))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

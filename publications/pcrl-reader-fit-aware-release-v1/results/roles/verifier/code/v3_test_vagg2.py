#!/usr/bin/env python3
"""RFA1 verifier, wave 3: invented-data test of vagg2 (registered / robustness / all cell sets) against nrm.rules aggregate,
benefit_screen, recipient_screen and robustness. Label-free (vguard)."""
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

import vagg2 as B  # noqa: E402
from nrm import rules as NR  # noqa: E402


def main():
    rng = np.random.default_rng(11)
    cells = [f'S{v}{r}k{k}' for v in (0, 1) for r in ('R1', 'R2', 'R3', 'R4') for k in (0, 1, 2, 3)]
    bad, worst, n = 0, 0.0, 0
    for t in range(300):
        mu, sc = rng.normal(0, 0.01), rng.choice([0.0005, 0.003, 0.01])
        e = {'cells': {c: float(mu + rng.normal(0, sc)) for c in cells}, 'cell_se': {c: float(abs(rng.normal(0.004, 0.002))) for c in cells}}
        if t % 60 == 0:
            e = {'cells': {c: 0.0 for c in cells}, 'cell_se': {c: 0.0 for c in cells}}
        for which in ('registered', 'robustness', 'all'):
            a, b = B.aggregate(e, which), NR.aggregate(e, which)
            d = max([abs(a['mean'] - b['mean']), abs(a['se_agg'] - b['se_agg'])] +
                    [abs(a['per_version'][v]['mean'] - b['per_version'][v]['mean']) for v in (0, 1)])
            worst = max(worst, d)
            same = (B.benefit(e, which)['label'] == NR.benefit_screen(e, cells=which)['label'] and
                    B.recipient(e, which)['label'] == NR.recipient_screen(e, cells=which)['label'])
            n += 1
            bad += (d > 1e-15) or not same
        r_m = B.d2r_comparator({'pair': e, 'income': e, 'occupation': e}, True)
        r_n = NR.robustness({'pair': e})
        bad += r_m['robustness_label'] != r_n['robustness_cells']['label'] or r_m['all_cells_label'] != r_n['all_cells']['label']
    res = {'trials': n, 'mismatches': int(bad), 'worst_abs_diff': worst, 'refused': vguard.REFUSED, 'all_pass': bad == 0}
    (HERE.parent / 'out' / 'V3_TEST_VAGG2.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

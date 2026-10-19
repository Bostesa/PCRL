"""Verifier stage V5 (invented data only; no real record, no label): RULES.md 8.1 requires that under guard 'none' the correction
parameters theta are bitwise invariant to the `targets` argument. The source P2 recipe (ohn.protect.train_correction) is run on
invented inputs for the three arms (CONT, LOC, PAIR) with three different targets dicts; theta must be bitwise equal; as a
sensitivity control, guard 'static' (which uses targets as weights) must change theta.

PCRL controlled training-size campaign (TSZ1).

    <torch venv python> .../v5_targets_invariance.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    from ohn import protect as PR
    rng = np.random.default_rng(20261019)
    n, T, L = 700, 12, 16
    base_z = rng.normal(size=(n, 5))
    leaves = rng.integers(0, L, size=(n, T))
    y = rng.integers(0, 5, size=n)
    sex = (rng.random(n) < 0.5).astype(np.int64)
    inc = rng.dirichlet(np.ones(2), size=n)
    inc_wire = PR.income_wire(inc)
    targets_list = [{'men': 0.005, 'women': 0.005}, {'men': 1.0, 'women': 1.0}, {'men': 0.0001, 'women': 0.5}]
    out = {}
    for arm in ('CONT', 'LOC', 'PAIR'):
        thetas = []
        for tg in targets_list:
            r = PR.train_correction(base_z, leaves, y, sex, inc_wire, T, L, arm, 'VERIFIER_INVENTED', 'inv', tg, epochs=2, steps=3, guard='none')
            thetas.append(np.asarray(r['theta'], dtype=np.float64))
        st = []
        # V2 correction: the sensitivity control must use targets that differ BETWEEN the SEX groups (equal men / women targets give
        # equal per-person weights after normalization, so 'static' is then also invariant; V1 compared two equal-ratio dicts)
        for tg in (targets_list[0], targets_list[2]):
            r = PR.train_correction(base_z, leaves, y, sex, inc_wire, T, L, arm, 'VERIFIER_INVENTED', 'inv', tg, epochs=2, steps=3, guard='static')
            st.append(np.asarray(r['theta'], dtype=np.float64))
        out[arm] = {'none_bitwise_invariant': all(np.array_equal(thetas[0], t) for t in thetas[1:]),
                    'static_changes_with_targets': not np.array_equal(st[0], st[1]),
                    'theta_sha256': V.arr_sha(thetas[0])}
    ok = all(x['none_bitwise_invariant'] for x in out.values())
    ctrl = all(x['static_changes_with_targets'] for x in out.values())
    chk = {'id': f'V5.CORRECTION.TARGETS_INVARIANCE__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'RULES.md 8.1: under guard none, theta of the source P2 recipe is bitwise invariant to targets (invented data; three '
                   'targets dicts incl. the registered constant {0.005, 0.005}; arms CONT, LOC, PAIR); guard static changes theta as a '
                   'sensitivity control',
           'inputs': {}, 'expected': 'bitwise invariant under none; static sensitive', 'observed': out,
           'status': 'PASS' if (ok and ctrl) else ('ATTENTION' if ok else 'FAIL'), 'code_sha256': V.code_hashes(__file__, V.__file__),
           'labels_read': False}
    V.write_record(f'V5_TARGETS_INVARIANCE__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **out}))


if __name__ == '__main__':
    main()

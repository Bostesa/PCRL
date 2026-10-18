"""Coordinator helper (no label read): wait until the ledger has new rows for given families, then print a compact summary.

    python -m ohn.watch --families A B C PROT INC --after ENTRY --min-new 1 --timeout 580
"""
from __future__ import annotations
import argparse
import csv
import time

from . import common as C

FIELDS = ('entry', 'family', 'config_id', 'seed', 'kind', 'budget_class', 'point_pass_12', 'guard_pass', 'occ_nll_excess_U0',
          'occ_brier_excess_U0', 'occ_accuracy_loss_U0', 'guard_men', 'guard_women', 'occ_nll_vs_T_TREE', 'occ_nll_vs_TASK_REFIT',
          'reserve_min_ratio', 'reserve_binding', 'deployed_minus_foldavg_nll', 'decision')


def rows():
    p = C.RESULTS / 'EXPERIMENT_LEDGER.csv'
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', required=True); ap.add_argument('--after', type=int, required=True)
    ap.add_argument('--min-new', type=int, default=1); ap.add_argument('--timeout', type=float, default=580)
    a = ap.parse_args(argv)
    t0 = time.time()
    while True:
        new = [r for r in rows() if int(r['entry']) > a.after and r['family'] in a.families]
        if len(new) >= a.min_new or time.time() - t0 > a.timeout:
            break
        time.sleep(15)
    print(C.now(), f'{len(new)} new rows')
    for r in new:
        print(' | '.join(f'{k}={r.get(k, "")}' for k in FIELDS if r.get(k, '') != '')[:400])


if __name__ == '__main__':
    main()

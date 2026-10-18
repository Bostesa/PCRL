"""Household partitions of MODEL_FIT for the PCRL correction-transfer campaign (definitions in cts.common.PARTITIONS).

    python3 -m cts.partitions manifest     label-free: PARTITIONS.json (household counts, set hashes, overlaps, structural checks)
    python3 -m cts.partitions support      after the registration receipt: PARTITION_SUPPORT.json (occupation class x SEX counts per
                                           partition role; counts only, no outcome; operation 'support_count')

Support rule (registered): every SEX group >= 500 people in each GUARD / CHECK role and in C0 / C1; every occupation class x SEX
cell >= 3 people in every role and >= 10 in every FIT. A failing partition takes the next tag suffix ('b', 'c', ...) before any fit;
the repair panels C0 / C1 are fixed (a failing panel is reported, not redrawn).
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from . import common as C

MIN_SEX_EVAL = 500
MIN_CELL = 3
MIN_CELL_FIT = 10


def manifest():
    F = C.features('MODEL_FIT')
    unit = F['unit']
    rec = {'schema': 'CTS1-partitions-v1', 'pool': 'MODEL_FIT (20,000 people, 13,337 households; admitted export)',
           'assignment': ('households ordered by SHA-256(<tag>|<household id>) and dealt round-robin into equal parts '
                          '(ohn.common.household_folds); label-free'),
           'blocks': {'n': C.N_BLOCKS, 'tag': C.BLOCK_TAG}, 'panels': {k: list(v) for k, v in C.PANELS.items()},
           'pool_blocks': list(C.POOL_BLOCKS),
           'partitions': {pid: {'kind': s['kind'], 'associated_source_seed': s['seed'], 'tag': s['tag'], 'parts': s['parts'],
                                'FIT_parts': list(s['FIT']), 'GUARD_parts': list(s['GUARD']),
                                'CHECK': list(s['CHECK']) if s['kind'] == 'diagnosis' else s['CHECK']}
                          for pid, s in C.PARTITIONS.items()},
           'support_rule': {'min_sex_group_guard_check_panel': MIN_SEX_EVAL, 'min_class_by_sex_cell_any_role': MIN_CELL,
                            'min_class_by_sex_cell_fit': MIN_CELL_FIT, 'on_failure': 'next tag suffix b, c, ... before any fit; panels fixed'},
           'summary': C.partition_summary(unit), 'structural_problems': C.check_partitions(unit), 'labels_read': False,
           'utc': C.now()}
    C.write_json(C.RESULTS / 'PARTITIONS.json', rec)
    print(json.dumps({'problems': rec['structural_problems'], 'summary': {k: v for k, v in rec['summary'].items() if k != 'FIT_overlap_households'}}))
    return rec


def support():
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'support_count',
               'registered support check: occupation class x SEX counts per partition role before any fit (no outcome)')
    y, sex, unit = F['occupation'], F['sex'], F['unit']
    m = C.partition_masks(unit)
    out, fails = {}, []

    def cells(mask):
        return {f'class{c}|sex{g}': int(((y == c) & (sex == g) & mask).sum()) for c in range(C.K_OCC) for g in (0, 1)}
    for name in ('C0', 'C1'):
        cc = cells(m[name]); men = int(((sex == 0) & m[name]).sum()); women = int(((sex == 1) & m[name]).sum())
        out[name] = {'cells': cc, 'men': men, 'women': women}
        if min(men, women) < MIN_SEX_EVAL or min(cc.values()) < MIN_CELL:
            fails.append(name)
    for pid in C.PARTITIONS:
        out[pid] = {}
        for r in C.PROLES:
            mask = m[pid][r]
            cc = cells(mask); men = int(((sex == 0) & mask).sum()); women = int(((sex == 1) & mask).sum())
            out[pid][r] = {'cells': cc, 'men': men, 'women': women}
            need = MIN_CELL_FIT if r == 'FIT' else MIN_CELL
            if min(cc.values()) < need or (r != 'FIT' and min(men, women) < MIN_SEX_EVAL):
                fails.append(f'{pid}:{r}')
    rec = {'schema': 'CTS1-partition-support-v1', 'counts': out, 'failures': fails, 'ok': not fails, 'utc': C.now(),
           'note': 'counts only; no model output is evaluated'}
    C.write_json(C.RESULTS / 'PARTITION_SUPPORT.json', rec)
    print(json.dumps({'ok': rec['ok'], 'failures': fails, 'min_cell': {k: (min(v['cells'].values()) if 'cells' in v else
                                                                          {r: min(x['cells'].values()) for r, x in v.items()}) for k, v in out.items()}}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('manifest', 'support'))
    a = ap.parse_args(argv)
    rec = manifest() if a.cmd == 'manifest' else support()
    return 0 if (rec.get('ok', True) and not rec.get('structural_problems')) else 1


if __name__ == '__main__':
    sys.exit(main())

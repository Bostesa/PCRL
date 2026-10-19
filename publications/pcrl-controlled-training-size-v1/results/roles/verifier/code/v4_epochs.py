"""Verifier stage V4 (label-free): the Stage B update accounting of every control fit vs deviation 3 (PROTOCOL_DEVIATIONS.md).

PCRL controlled training-size campaign (TSZ1). Fixed effort: 20 epochs x 40 updates x 256 rows. Recomputed from the subset size n
(own subset reconstruction): source slices of one permutation would leave batch sizes n - 256 b (last partial / empty batches) when
n < 10,240; the control then uses the cyclic extension and 10,240 - n people are used twice per epoch; coverage = 10,240 / n capped
at 1. Compared with every STAGE_B.json (n, epoch_order, people_used_twice_per_epoch, source_slice_batches_not_full,
source_slice_empty_batches, epoch_coverage).

    python3 .../v4_epochs.py --tag V1
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
from tsz import common as C  # noqa: E402   (label-free features only)

CONTROL = ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'runs' / 'CONTROL'
B, NB = 256, 40


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    masks = V.subset_masks(C.features('MODEL_FIT')['unit'])
    rows, bad, files = {}, [], []
    for p in sorted(CONTROL.glob('*/STAGE_B/STAGE_B.json')):
        s = json.loads(p.read_text()); files.append(p)
        sid = p.parts[-3].split('__')[0]
        n = int(masks[sid].sum())
        per_epoch = B * NB
        if n < per_epoch:
            partial = {str(b): max(0, n - B * b) for b in range(NB) if n - B * b < B}
            exp = {'n': n, 'cyclic': True, 'twice': per_epoch - n, 'not_full': partial,
                   'empty': sum(1 for x in partial.values() if x == 0), 'coverage': 1.0}
        else:
            exp = {'n': n, 'cyclic': False, 'twice': 0, 'not_full': {}, 'empty': 0, 'coverage': per_epoch / n}
        obs = {'n': s.get('n'), 'cyclic': 'cyclic' in str(s.get('epoch_order')), 'twice': s.get('people_used_twice_per_epoch'),
               'not_full': s.get('source_slice_batches_not_full'), 'empty': s.get('source_slice_empty_batches'), 'coverage': s.get('epoch_coverage')}
        ok = (obs['n'] == exp['n'] and obs['cyclic'] == exp['cyclic'] and obs['twice'] == exp['twice'] and obs['not_full'] == exp['not_full']
              and obs['empty'] == exp['empty'] and abs(float(obs['coverage']) - exp['coverage']) < 1e-12)
        rows[p.parts[-3]] = {'expected': exp, 'observed': obs, 'ok': ok}
        if not ok:
            bad.append(p.parts[-3])
    chk = {'id': f'V4.RECORDS.STAGE_B_UPDATES__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'Stage B effort of every control fit (20 x 40 x 256 updates): subset size, cyclic epoch order below 10,240 people, people used '
                   'twice per epoch, short / empty source-slice batches and coverage, recomputed from the own subset sizes vs STAGE_B.json and '
                   'PROTOCOL_DEVIATIONS item 3',
           'inputs': V.inputs_sha(files), 'expected': 'agreement', 'observed': {'fits': rows, 'problems': bad},
           'status': 'PASS' if (rows and not bad) else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    import os
    V.write_record(f'V4_EPOCHS__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'n': len(rows), 'problems': bad}))


if __name__ == '__main__':
    main()

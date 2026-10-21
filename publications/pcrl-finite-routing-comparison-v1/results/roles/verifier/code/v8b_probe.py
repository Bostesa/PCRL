"""V8b (optional, label-free): the PROBE_STRENGTH partition at strength 8 against my own V5b routing of the saved PRC_S2 book.

For each version: my out-of-fold partition (k, l) of PRC_S2 (v5b_protected_tables.route_book on the cached out-of-fold statistics and
the saved book), hashed as SHA-256 over (dtype string, shape string, bytes) of k then of l in one digest, against
results/<campaign>/fa/PROBE_STRENGTH_V<v>.json per_strength['8.0']['partition_sha256'].

    python3 .../v8b_probe.py --tag V1
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import v5b_protected_tables as T  # noqa: E402


def part_sha(k, l):
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(k), np.ascontiguousarray(l)):
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def run(tag):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    from ohn import fam_a as FA
    F = C.features('MODEL_FIT')
    Z = FA.compact_po(np.asarray(F['X'], dtype=np.float64), tuple(map(str, F['feature_names'])))
    res, ok = {}, True
    for v in (0, 1):
        task = pickle.loads((V.SRC_FA_CACHE / f'V{v}' / 'TASK.pkl').read_bytes())
        ss = pickle.loads((V.FA_CACHE / f'V{v}' / 'SEXSIDE.pkl').read_bytes())
        E = {'qI': np.asarray(task['oof']['inc'], dtype=np.float64), 'qO': np.asarray(task['oof']['occ'], dtype=np.float64), 'Z': Z,
             'pi': np.minimum(np.maximum(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], 0.01), 0.99),
             'qO0': np.asarray(ss['oof']['occ_s'][0], dtype=np.float64), 'qO1': np.asarray(ss['oof']['occ_s'][1], dtype=np.float64)}
        M = pickle.loads((V.RUNS / 'FA' / 'PRC_S2' / f'V{v}' / 'MODEL.pkl').read_bytes())
        k, l, gap = T.route_book(E, M['book'])
        mine = part_sha(k, l)
        probe = json.loads((V.RESULTS / 'fa' / f'PROBE_STRENGTH_V{v}.json').read_text())
        rec = (probe.get('per_strength') or {}).get('8.0') or {}
        res[f'V{v}'] = {'mine': mine, 'probe_8': rec.get('partition_sha256'), 'equal': mine == rec.get('partition_sha256'),
                        'selected': M.get('selected'), 'min_gap': gap, 'dtype': str(k.dtype)}
        ok &= res[f'V{v}']['equal']
    chk = {'id': f'V8B.PROBE_STRENGTH_8__{tag}', 'what': 'PROBE_STRENGTH partition at strength 8 vs my own routing of the saved PRC_S2 '
           'book on the cached out-of-fold statistics (partition hash)', 'inputs': V.inputs_sha([V.RESULTS / 'fa' / f'PROBE_STRENGTH_V{v}.json'
                                                                                               for v in (0, 1)]),
           'result': res, 'numerical_status': 'PASS' if ok else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(chk['id'], chk['numerical_status'], json.dumps(res))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())

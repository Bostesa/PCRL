"""VER check 1 (supplement): admitted source releases without a committed file hash (runs/O/*, runs/H/HC_TTREE_LJDIR_PB, ...):
search the source's committed records for their WIRE hashes (per role and task) and array hashes, label-free."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402


def main():
    texts = []
    for root in (V.SRC_RESULTS, V.ROOT / 'results' / 'pcrl_frozen_finalist_validation_v1', V.ROOT / 'results' / 'pcrl_autonomous_method_development_v1'):
        if root.is_dir():
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.txt', '.jsonl'):
                    texts.append(p.read_text(errors='ignore'))
    blob = '\n'.join(texts)
    out = {}
    for p in sorted(V.MIRROR.glob('runs/*/*/S*/RELEASE.npz')):
        rel = str(p.relative_to(V.MIRROR))
        rec = {'file_sha_found': V.file_sha256(p) in blob, 'wires_found': {}, 'n_wires': 0}
        with np.load(p) as z:
            for k in z.files:
                rec['n_wires'] += 1
                rec['wires_found'][k] = V.wire_hash(z[k]) in blob
        rec['n_wires_found'] = sum(rec['wires_found'].values())
        out[rel] = rec
    res = {'check': 'VER_SOURCE_PINS', 'utc': V.utc(), 'unit_id': V.unit_id(), 'releases': out,
           'summary': {k: f"file {'pinned' if v['file_sha_found'] else 'unpinned'}; wires pinned {v['n_wires_found']}/{v['n_wires']}" for k, v in out.items()}}
    V.write_record('VER_SOURCE_PINS', res)
    print(json.dumps(res['summary'], indent=1))


if __name__ == '__main__':
    main()

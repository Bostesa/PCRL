"""Unit FFV_MAPPING_CONFLICTS_V2: label-free supplement to the output-mapping check (claims review M12, M13).

Per seed, over the union of the six admitted roles (exact transmitted float64 bytes; encoded features from the label-free
exports): the exact number of occupation vectors shared by two or more distinct encoded inputs (and the excess of distinct inputs
over distinct occupation vectors, which counts each shared vector k - 1 times); and, for every occupation vector that carries more
than one income level, the identifier-free description: roles, income levels, and the raw-variable names whose encoding differs.
No person identifier, raw value or label is written. Replaces the session-scratchpad script of unit FFV_MAPPING_CONFLICTS_V1.
"""
from __future__ import annotations
import json

import numpy as np

from . import common as C
from .mapping import level_of, wire_keys

FIN = 'B3CTL_UNCOUPLED_MIX50_C6A'


def main():
    out = {'schema': 'FFV1-mapping-conflicts-v1', 'utc': C.now(), 'labels_read': 'none', 'per_seed': {}}
    for s in C.SEEDS:
        with np.load(C.admitted(f'runs/B/{FIN}/S{s}/RELEASE.npz')) as z:
            rel = {k: z[k].copy() for k in z.files}
        man = json.loads((C.RESULTS / 'FROZEN_CANDIDATE_MANIFEST.json').read_text())['per_seed'][f'S{s}']
        lv = np.array([[float.fromhex(x) for x in row] for row in man['per_level_income_probability_hex']])
        occ_map, names = {}, None
        for r in C.ROLES6:
            X = C.load_role(r, ('features',)); fx = X['X'].astype(np.float32); names = X['feature_names']
            ko = wire_keys(rel[f'{r}__occupation']); lev = level_of(rel[f'{r}__income'], lv)
            for i, k in enumerate(ko):
                occ_map.setdefault(k, []).append((r, int(lev[i]), fx[i].tobytes()))
        inputs_per_vec = {k: len({x[2] for x in v}) for k, v in occ_map.items()}
        shared = [k for k, n in inputs_per_vec.items() if n > 1]
        conflicts = []
        for k, v in occ_map.items():
            if len({x[1] for x in v}) > 1:
                arrs = [np.frombuffer(b, dtype=np.float32) for b in {x[2] for x in v}]
                fields = sorted({names[j].split('=')[0] for a in arrs for b in arrs for j in np.flatnonzero(a != b)})
                conflicts.append({'roles_and_levels': sorted({(x[0], x[1]) for x in v}), 'distinct_encoded_inputs': len(arrs),
                                  'differing_raw_variables': fields})
        out['per_seed'][f'S{s}'] = {
            'union_occupation_vectors_shared_by_distinct_encoded_inputs': len(shared),
            'union_distinct_inputs_minus_distinct_occupation_vectors': int(sum(n - 1 for n in inputs_per_vec.values())),
            'max_distinct_inputs_on_one_occupation_vector': int(max(inputs_per_vec.values())),
            'cross_role_conflicts': conflicts, 'source_units': ['FFV_MAPPING_V2 (counts)', 'FFV_MAPPING_CONFLICTS_V1 (first, scratchpad version)']}
    C.write_json_x(C.RESULTS / 'MAPPING_CONFLICTS.json', out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()

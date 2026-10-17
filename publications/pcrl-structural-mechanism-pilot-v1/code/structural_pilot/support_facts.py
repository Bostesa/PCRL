"""Label-free support facts (unit SMP_SUPPORT_FACTS): distinct encoded inputs per role, overlap with MODEL_FIT, and
per-field level counts. Reads features only (no task label, no SEX). Output: aggregate counts only."""
from __future__ import annotations
import json
import sys

import numpy as np

from . import common as C


def rows_as_void(X):
    X = np.ascontiguousarray(X)
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


def main(out):
    roles = {r: C.load_role(r, ('features',)) for r in C.ROLES}
    names = roles['MODEL_FIT']['feature_names']
    mf = set(rows_as_void(roles['MODEL_FIT']['X']).tolist())
    rec = {'schema': 'SMP1-support-facts-v1', 'input_width': len(names), 'roles': {}}
    for r, d in roles.items():
        v = rows_as_void(d['X'])
        u, c = np.unique(v, return_counts=True)
        rec['roles'][r] = {'persons': int(len(v)), 'households': int(len(np.unique(d['unit']))), 'distinct_encoded_inputs': int(len(u)),
                           'share_of_persons_with_unique_input_in_role': float((c == 1).sum() / len(v)),
                           'share_of_persons_whose_input_occurs_in_MODEL_FIT': float(np.mean([x in mf for x in v.tolist()]))}
    groups = {}
    for i, n in enumerate(names):
        g = n.split('=')[0]
        groups.setdefault(g, []).append(i)
    rec['columns_per_field'] = {g: len(ix) for g, ix in groups.items()}
    X = roles['MODEL_FIT']['X']
    rec['distinct_values_MODEL_FIT'] = {g: int(len(np.unique(rows_as_void(X[:, ix])))) for g, ix in groups.items()}
    with open(out, 'w') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])

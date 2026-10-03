"""Exact feature-vector equality between a query set and a reference set, reported BOTH as distinct vectors and as
affected query rows (repairs the output-aware amendment-A1 field that counted distinct vectors under a row label)."""
import hashlib

import numpy as np


def _row_keys(X):
    X = np.ascontiguousarray(X, dtype=np.float64)
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=16).hexdigest() for r in X])


def feature_equality_counts(X_query, X_ref) -> dict:
    q, r = _row_keys(X_query), set(_row_keys(X_ref))
    hit = np.array([k in r for k in q], dtype=bool)
    return {"query_rows": int(len(q)), "affected_rows": int(hit.sum()),
            "distinct_query_vectors": int(len(set(q))), "distinct_vectors_shared_with_reference": int(len(set(q[hit])))}

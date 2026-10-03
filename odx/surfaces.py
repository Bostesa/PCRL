"""Score decomposition of a task-head output vector (standard algebra; see SCORE_DECOMPOSITION.md).

Binary logits (l0, l1): margin d = l1 - l0, offset c = (l0 + l1)/2; (l0, l1) = (c - d/2, c + d/2).
Multiclass logits l (K): offset c = mean_k l_k, centred = l - c (sum 0); l = centred + c.
softmax(l) depends only on the centred vector (binary: on d), so probabilities, the argmax decision and proper task
losses are unchanged by removing c.
"""
from __future__ import annotations

import numpy as np


def decompose(L):
    L = np.asarray(L, dtype=np.float64)
    c = L.mean(1)
    centred = L - c[:, None]
    out = {"offset": c[:, None], "centred_vec": centred}
    if L.shape[1] == 2:
        d = L[:, 1] - L[:, 0]
        out["margin"] = d[:, None]
    return out


def softmax(L):
    L = np.asarray(L, dtype=np.float64)
    Z = L - L.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def onehot(ids, K):
    X = np.zeros((len(ids), K))
    X[np.arange(len(ids)), np.asarray(ids, int)] = 1.0
    return X


def surfaces(L) -> dict:
    """Release surfaces of one output vector. 'centred' is non-redundant for binary (the margin d, 1 column) and the
    full centred K-vector for multiclass; 'dc' is the invertible (centred, offset) recoding of 'full'."""
    L = np.asarray(L, dtype=np.float64)
    K = L.shape[1]
    D = decompose(L)
    cen = D["margin"] if K == 2 else D["centred_vec"]
    return {"full": L, "dc": np.hstack([cen, D["offset"]]), "centred": cen, "prob": softmax(L),
            "offset": D["offset"], "hard": onehot(L.argmax(1), K)}


def exactness(L) -> dict:
    """Numerical checks of the identities on actual arrays (float64)."""
    L = np.asarray(L, dtype=np.float64)
    K = L.shape[1]
    D = decompose(L)
    P = softmax(L)
    rec = D["centred_vec"] + D["offset"]
    out = {"K": K, "n": int(len(L)), "reconstruct_full_from_centred_plus_offset_maxabs": float(np.max(np.abs(rec - L))),
           "softmax_of_centred_minus_softmax_full_maxabs": float(np.max(np.abs(softmax(D["centred_vec"]) - P))),
           "offset_sd": float(np.std(D["offset"]))}
    if K == 2:
        d = D["margin"][:, 0]
        sig = 1.0 / (1.0 + np.exp(-d))
        out.update({"sigmoid_margin_minus_softmax_p1_maxabs": float(np.max(np.abs(sig - P[:, 1]))),
                    "argmax_equals_margin_positive": bool(np.array_equal(L.argmax(1), (d > 0).astype(int))),
                    "ties_d_equal_0": int((d == 0).sum()),
                    "p1_saturated_exact_0_or_1": int(((P[:, 1] == 0.0) | (P[:, 1] == 1.0)).sum()),
                    "rows_d_ge_36_7368_float64_p1_eq_1": int((d >= 36.7368005696771).sum()),
                    "rows_abs_d_ge_16_6355_float32_saturation": int((np.abs(d) >= 16.635532333438686).sum()),
                    "abs_margin_max": float(np.max(np.abs(d)))})
    return out

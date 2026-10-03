"""Canonical task-output transformation (a standard algebraic step, not a new method or a privacy guarantee).

Removes the common logit offset c = mean_k l_k from a released logit vector. softmax, the argmax decision and every
proper task loss are mathematically unchanged; the removed scalar is returned separately so it can be withheld.

Example
-------
>>> import numpy as np
>>> from odx.canonical import canonical_logits
>>> L = np.array([[2.0, 3.5], [-1.0, 0.25]])
>>> out = canonical_logits(L)
>>> out["centred"].tolist()                          # (-d/2, d/2) per row
[[-0.75, 0.75], [-0.625, 0.625]]
>>> bool(out["softmax_unchanged"]) and bool(out["decision_unchanged"])
True

Measured effect in this study (development data only): see SCORE_DECOMPOSITION.md. Withholding the offset changes
what a fitted attacker can recover; it does not bound what any attacker could recover from the remaining output.
"""
from __future__ import annotations

import numpy as np


def _softmax(L):
    Z = L - L.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def canonical_logits(L, atol: float = 1e-12) -> dict:
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] < 2:
        raise ValueError("expected an (n, K>=2) logit matrix")
    c = L.mean(1, keepdims=True)
    centred = L - c
    if L.shape[1] == 2:                                # exact (-d/2, d/2) from the margin
        d = L[:, 1] - L[:, 0]
        centred = np.stack([-d / 2, d / 2], 1)
    sm_unchanged = bool(np.max(np.abs(_softmax(centred) - _softmax(L))) <= atol)
    dec_unchanged = bool(np.array_equal(centred.argmax(1), L.argmax(1)))
    return {"centred": centred, "removed_offset": c[:, 0], "softmax_unchanged": sm_unchanged,
            "decision_unchanged": dec_unchanged}

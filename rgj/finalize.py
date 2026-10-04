"""Deployment finalisation under the new roles (identical for every neural arm; heads part reused for FARE/LEACE).

Release r_i = g_i(X) (identity map; no erasure in any trained arm). Deployed head: StandardScaler + LogisticRegression
fitted on DEFENSE_FIT, C in {0.01, 0.1, 1, 10, 100} selected by HEAD_VALIDATION log loss (ties -> smaller C). Released
outputs: centred logits from the head's decision function (predecessor amendment A1 form), probabilities, hard
decision. The primary recipient view is [r_i, centred logits_i]; both are functions of r_i only (output closure).
Releases are computed only for rows of the five new roles (rgj.data drops every other row at load).
"""
from __future__ import annotations

import numpy as np
import torch

from jcv.finalize import HEAD_C, fit_head, outputs, save_unit, unit_complete  # noqa: F401  (pinned helpers)

KS = [2, 6]
TASKS = ("income", "occupation_group")


def heads_and_outputs(Rs, D):
    tr, va = D["idx"]["DEFENSE_FIT"], D["idx"]["HEAD_VALIDATION"]
    out, heads, meta = {}, {}, {}
    for i in (0, 1):
        R = np.asarray(Rs[i], dtype=np.float64)
        head, hm = fit_head(R, D["y"][TASKS[i]], tr, va, KS[i])
        cen, P, hard = outputs(head, R)
        out.update({f"r{i + 1}": R, f"c{i + 1}": cen, f"p{i + 1}": P, f"hard{i + 1}": hard})
        heads[i], meta[i] = head, hm
    return out, heads, meta


def encode_all(model, X):
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        return [model.encode(i, Xt).double().numpy() for i in (0, 1)]


def finalize_model(model, D):
    """Identity-map release + refitted deployed heads on the new roles."""
    return heads_and_outputs(encode_all(model, D["X"]), D)


def views_from_release(z):
    v1 = np.hstack([z["r1"], z["c1"]])
    v2 = np.hstack([z["r2"], z["c2"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": z["p1"], "p2": z["p2"], "hard1": z["hard1"], "hard2": z["hard2"]}}

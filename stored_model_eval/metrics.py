"""Leakage, recovery and health metrics computed from stored arrays.

Conventions (each documented where it is implemented):
  * R2_onehot (PCRL convention, pcrl/purposes/verification.py LinearComplianceCertificate and
    experiments/rlace_diagnostic.py::linear_r2 at origin/main): one-hot Y with K = max(y)+1 columns,
    H and Y centred, closed-form ridge W = (Hc'Hc + lam*I)^-1 Hc'Yc with an ABSOLUTE penalty lam=1e-6,
    R2 = 1 - SS_res/SS_tot pooled over the K columns, clamped at 0, fit and scored on the SAME rows.
    Deviation: PCRL keeps the input dtype (float32 for torch representations); we compute in float64
    unless dtype=np.float32 is passed to reproduce the historical arithmetic.
  * A fixed absolute ridge penalty makes R2 scale-dependent; OLS R2 (lam=0, minimum-norm lstsq with an
    intercept) is invariant to any invertible linear map of H.
  * AUCs are rank statistics (ties count 1/2) and accept optional non-negative row weights, which is how
    the cluster bootstrap re-weights rows without copying arrays.
  * A quantity that cannot be estimated returns a NotEstimable sentinel with counts and coverage. It has
    no truth value and cannot be compared with a number, so it can never be read as pass/fail/0/1.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------------
# NOT_ESTIMABLE sentinel
# --------------------------------------------------------------------------------------------------


class NotEstimable:
    """Explicit 'quantity not estimable' result. Never 0, 1, pass or fail."""

    __slots__ = ("reason", "counts", "coverage", "partial")

    def __init__(self, reason: str, counts: dict | None = None, coverage: tuple | None = None,
                 partial: dict | None = None):
        self.reason = reason
        self.counts = dict(counts or {})
        self.coverage = coverage  # (n_supported, n_total) where meaningful
        self.partial = dict(partial or {})  # explicitly labelled partial quantities, never the headline

    def __bool__(self):
        raise TypeError("NotEstimable has no truth value: it is neither pass nor fail")

    def __float__(self):
        raise TypeError("NotEstimable cannot be converted to a number")

    def __lt__(self, other):
        raise TypeError("NotEstimable cannot be compared with a bar")

    __le__ = __gt__ = __ge__ = __lt__

    def __repr__(self):
        return f"NotEstimable({self.reason!r}, counts={self.counts}, coverage={self.coverage})"

    def to_json(self) -> dict:
        out = {"status": "NOT_ESTIMABLE", "reason": self.reason, "counts": _jsonable(self.counts)}
        if self.coverage is not None:
            n_s, n_t = self.coverage
            out["coverage"] = {"supported": int(n_s), "total": int(n_t),
                               "fraction": (n_s / n_t) if n_t else None}
        if self.partial:
            out["partial_not_headline"] = _jsonable(self.partial)
        return out


def is_estimable(x: Any) -> bool:
    return not isinstance(x, NotEstimable)


def _jsonable(x):
    if isinstance(x, NotEstimable):
        return x.to_json()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


to_jsonable = _jsonable

# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _idx(n: int, idx) -> np.ndarray:
    if idx is None:
        return np.arange(n)
    idx = np.asarray(idx)
    if idx.dtype == bool:
        return np.flatnonzero(idx)
    return idx.astype(np.int64)


def class_counts(y: np.ndarray, classes=None, w: np.ndarray | None = None) -> dict[int, float]:
    y = np.asarray(y).astype(np.int64)
    if classes is None:
        classes = np.unique(y)
    if w is None:
        return {int(k): int((y == k).sum()) for k in classes}
    return {int(k): float(w[y == k].sum()) for k in classes}


# --------------------------------------------------------------------------------------------------
# R2 family
# --------------------------------------------------------------------------------------------------


def r2_onehot_ridge(H, y, lam: float = 1e-6, fit_idx=None, score_idx=None, clamp: bool = True,
                    dtype=np.float64, n_classes: int | None = None):
    """Pooled one-hot ridge R2 (PCRL convention when fit_idx == score_idx, the default).

    Held-out variant (score_idx given and different): centre with fit-row means, predict score rows,
    SS_tot around the score-row mean. Returns NotEstimable when the scored rows hold < 2 classes.
    """
    H = np.asarray(H, dtype=dtype)
    y = np.asarray(y).astype(np.int64)
    fi = _idx(len(y), fit_idx)
    si = fi if score_idx is None else _idx(len(y), score_idx)
    K = int(n_classes if n_classes is not None else y.max() + 1)
    if len(np.unique(y[si])) < 2 or len(np.unique(y[fi])) < 2:
        return NotEstimable("fewer than 2 classes in fit or score rows",
                            counts={"fit": class_counts(y[fi]), "score": class_counts(y[si])})
    Y = np.eye(K, dtype=dtype)[y]
    muH, muY = H[fi].mean(0, keepdims=True), Y[fi].mean(0, keepdims=True)
    Hc, Yc = H[fi] - muH, Y[fi] - muY
    G = Hc.T @ Hc + dtype(lam) * np.eye(H.shape[1], dtype=dtype)
    W = np.linalg.solve(G, Hc.T @ Yc)
    if score_idx is None or np.array_equal(si, fi):
        res = Yc - Hc @ W
        tot = Yc
    else:
        pred = (H[si] - muH) @ W + muY
        res = Y[si] - pred
        tot = Y[si] - Y[si].mean(0, keepdims=True)
    ss_tot = float((tot ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / max(ss_tot, 1e-12)
    return max(0.0, r2) if clamp else r2


def _design(H):
    return np.c_[H, np.ones(len(H))]


def ols_r2(H, y, fit_idx=None, score_idx=None, onehot: bool = True):
    """OLS (lam = 0, intercept, minimum-norm lstsq) R2 of one-hot y (or a real target if onehot=False).

    Invariant to any invertible linear map of H (incl. rescaling). Not clamped (held-out can be < 0).
    """
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y)
    fi = _idx(len(y), fit_idx)
    si = fi if score_idx is None else _idx(len(y), score_idx)
    if onehot:
        yi = y.astype(np.int64)
        if len(np.unique(yi[si])) < 2:
            return NotEstimable("fewer than 2 classes in score rows", counts={"score": class_counts(yi[si])})
        Y = np.eye(int(yi.max()) + 1)[yi]
    else:
        Y = y.astype(np.float64).reshape(len(y), -1)
    B, *_ = np.linalg.lstsq(_design(H[fi]), Y[fi], rcond=None)
    res = Y[si] - _design(H[si]) @ B
    tot = Y[si] - Y[si].mean(0, keepdims=True)
    return 1.0 - float((res ** 2).sum()) / max(float((tot ** 2).sum()), 1e-300)


def ols_heldout_scores(H, y, fit_idx, score_idx) -> np.ndarray:
    """Held-out linear least-squares discriminant scores for each class column (n_score x K).

    Rank-based AUCs of these scores are exactly invariant to invertible linear maps of H.
    """
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    fi, si = _idx(len(yi), fit_idx), _idx(len(yi), score_idx)
    Y = np.eye(int(yi.max()) + 1)[yi]
    B, *_ = np.linalg.lstsq(_design(H[fi]), Y[fi], rcond=None)
    return _design(H[si]) @ B


def contrast_r2(H, y, a) -> float:
    """OLS R2 of the class contrast t = sum_k a_k 1[y=k] on H (a real-valued target)."""
    y = np.asarray(y).astype(np.int64)
    a = np.asarray(a, dtype=np.float64)
    return ols_r2(H, a[y], onehot=False)


def cross_covariance_norm(H, y) -> float:
    """Frobenius norm of the cross-covariance between centred H and centred one-hot y."""
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    Y = np.eye(int(yi.max()) + 1)[yi]
    C = (H - H.mean(0)).T @ (Y - Y.mean(0)) / len(yi)
    return float(np.linalg.norm(C))


# --------------------------------------------------------------------------------------------------
# canonical correlation
# --------------------------------------------------------------------------------------------------


def cca_rho2(H, y, eps_rel: float = 1e-6, tol: float = 1e-10, drop_class: int | None = None):
    """Top squared canonical correlation between H and the class indicators of y (float64).

    One-hot with one present class dropped (default: the smallest present label); classes with zero
    support are dropped and reported. Sigma_HH is regularised by eps = eps_rel * tr(Sigma_HH)/d.
    The eigenvalue must lie in [-tol, 1+tol] (then clipped to [0,1]); otherwise a numerical error is
    raised rather than a number returned. Returns dict(value, contrast, details) or NotEstimable.
    rho1^2 = max over ALL class contrasts a of the linear R2 of a'Y on H, so it bounds every per-class
    OvR R2 and the pooled one-hot R2 from above.
    """
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    n, d = H.shape
    present = sorted(int(k) for k in np.unique(yi))
    if len(present) < 2:
        return NotEstimable("fewer than 2 classes present", counts=class_counts(yi))
    ref = present[0] if drop_class is None else int(drop_class)
    if ref not in present:
        raise ValueError(f"drop_class {ref} has no support")
    keep = [k for k in present if k != ref]
    Yd = (yi[:, None] == np.asarray(keep)[None, :]).astype(np.float64)
    Yc = Yd - Yd.mean(0)
    Hc = H - H.mean(0)
    Shh = Hc.T @ Hc / n
    eps = eps_rel * np.trace(Shh) / d
    Shh = Shh + eps * np.eye(d)
    Syy = Yc.T @ Yc / n
    Syh = Yc.T @ Hc / n
    ev, U = np.linalg.eigh(Syy)
    if ev.min() <= 0:
        return NotEstimable("singular class covariance", counts=class_counts(yi))
    Syy_mh = U @ np.diag(ev ** -0.5) @ U.T
    M = Syy_mh @ Syh @ np.linalg.solve(Shh, Syh.T) @ Syy_mh
    lam, V = np.linalg.eigh((M + M.T) / 2)
    top = float(lam[-1])
    if top < -tol or top > 1 + tol:
        raise FloatingPointError(f"rho1^2 = {top!r} outside [0,1] beyond tol {tol}")
    top = min(max(top, 0.0), 1.0)
    a_keep = Syy_mh @ V[:, -1]
    a = np.zeros(int(yi.max()) + 1)
    a[keep] = a_keep
    absent = [k for k in range(int(yi.max()) + 1) if k not in present]
    return {"value": top, "contrast": a, "dropped_reference_class": ref, "absent_classes": absent,
            "eps": float(eps), "min_class_cov_eig": float(ev.min())}


# --------------------------------------------------------------------------------------------------
# AUC family
# --------------------------------------------------------------------------------------------------


def auc_binary(y01, s, w: np.ndarray | None = None):
    """Weighted Mann-Whitney AUC with ties counted 1/2. Returns NotEstimable if a side has no weight."""
    y01 = np.asarray(y01).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=np.float64)
    wp, wn = float(w[y01].sum()), float(w[~y01].sum())
    if wp <= 0 or wn <= 0:
        return NotEstimable("one side empty", counts={"pos_weight": wp, "neg_weight": wn})
    u, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, weights=w * y01, minlength=len(u))
    gn = np.bincount(inv, weights=w * (~y01), minlength=len(u))
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (wp * wn))


def _scores_matrix(P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = np.c_[1.0 - P, P]
    return P


def per_class_ovr_auc(y, P, min_support: int, w=None, classes=None) -> dict:
    """Per-class one-vs-rest AUC of column k. A class is supported iff it has >= min_support rows AND
    its complement has >= min_support rows (unweighted counts of distinct rows)."""
    P = _scores_matrix(P)
    yi = np.asarray(y).astype(np.int64)
    K = P.shape[1]
    classes = range(K) if classes is None else classes
    out = {}
    for k in classes:
        n_pos = int((yi == k).sum())
        n_neg = len(yi) - n_pos
        if n_pos < min_support or n_neg < min_support:
            kind = "absent" if n_pos == 0 else "singleton" if n_pos == 1 else "below_min_support"
            out[int(k)] = NotEstimable(f"class {k} {kind}", counts={"n_pos": n_pos, "n_neg": n_neg,
                                                                     "min_support": min_support})
        else:
            out[int(k)] = auc_binary(yi == k, P[:, k], w)
    return out


def macro_ovr_auc(y, P, min_support: int, w=None, over: str = "all_declared"):
    """Macro one-vs-rest AUC. Binary: AUC of column 1. Multiclass: mean of per-class OvR AUCs over the
    DECLARED class set (columns of P). If any declared class is unsupported -> NotEstimable carrying the
    supported-class mean as an explicitly labelled partial quantity and the coverage."""
    P = _scores_matrix(P)
    yi = np.asarray(y).astype(np.int64)
    if P.shape[1] == 2:
        n1 = int((yi == 1).sum())
        n0 = len(yi) - n1
        if min(n0, n1) < min_support:
            return NotEstimable("binary class below min_support", counts={"n0": n0, "n1": n1,
                                                                          "min_support": min_support})
        return auc_binary(yi == 1, P[:, 1], w)
    pc = per_class_ovr_auc(yi, P, min_support, w)
    sup = {k: v for k, v in pc.items() if is_estimable(v)}
    if len(sup) == len(pc):
        return float(np.mean(list(sup.values())))
    if over == "supported" and len(sup) >= 2:
        # methodology protocol variant: mean over supported classes; coverage must be reported alongside
        return float(np.mean(list(sup.values())))
    partial = {"macro_over_supported_classes": float(np.mean(list(sup.values())))} if sup else {}
    return NotEstimable("declared classes unsupported",
                        counts={k: v.counts for k, v in pc.items() if not is_estimable(v)},
                        coverage=(len(sup), len(pc)), partial=partial)


def worst_class_auc(y, P, min_support: int, w=None) -> dict:
    """Max over SUPPORTED classes of per-class OvR AUC, with argmax and coverage."""
    pc = per_class_ovr_auc(y, P, min_support, w)
    sup = {k: v for k, v in pc.items() if is_estimable(v)}
    unsup = {k: v.counts for k, v in pc.items() if not is_estimable(v)}
    if not sup:
        val = NotEstimable("no supported class", counts=unsup, coverage=(0, len(pc)))
        return {"value": val, "argmax": None, "coverage": (0, len(pc)), "unsupported": unsup}
    k = max(sup, key=sup.get)
    return {"value": sup[k], "argmax": k, "coverage": (len(sup), len(pc)), "unsupported": unsup}


def pair_score(P, i, j, rows) -> np.ndarray:
    den = P[rows, i] + P[rows, j]
    return np.where(den > 0, P[rows, j] / np.maximum(den, 1e-300), 0.5)


def supported_pairs(y, K: int, min_support: int) -> tuple[list, dict]:
    yi = np.asarray(y).astype(np.int64)
    cnt = np.bincount(yi, minlength=K)
    sup, unsup = [], {}
    for i, j in combinations(range(K), 2):
        if cnt[i] >= min_support and cnt[j] >= min_support:
            sup.append((i, j))
        else:
            unsup[(i, j)] = {"n_i": int(cnt[i]), "n_j": int(cnt[j]), "min_support": min_support}
    return sup, unsup


def pairwise_auc(y, P, min_support: int, w=None, orientation_free: bool = False) -> dict:
    """Pairwise AUC for each class pair (i<j) on rows of classes i,j, scored by p_j/(p_i+p_j), target j.

    Pairs where either class has < min_support rows are NotEstimable. Returns per-pair values, the max
    over supported pairs (worst pair), and coverage = supported pairs / total pairs. orientation_free
    takes max(a, 1-a); it is biased upward under the null and is OFF by default.
    """
    P = _scores_matrix(P)
    yi = np.asarray(y).astype(np.int64)
    K = P.shape[1]
    sup, unsup = supported_pairs(yi, K, min_support)
    per = {}
    for (i, j), c in unsup.items():
        per[(i, j)] = NotEstimable(f"pair ({i},{j}) unsupported", counts=c)
    for i, j in sup:
        rows = np.flatnonzero((yi == i) | (yi == j))
        a = auc_binary(yi[rows] == j, pair_score(P, i, j, rows), None if w is None else w[rows])
        if orientation_free and is_estimable(a):
            a = max(a, 1.0 - a)
        per[(i, j)] = a
    est = {k: v for k, v in per.items() if is_estimable(v)}
    total = K * (K - 1) // 2
    if not est:
        mx = NotEstimable("no supported pair", counts={str(k): v for k, v in unsup.items()},
                          coverage=(0, total))
        return {"per_pair": per, "max": mx, "argmax": None, "coverage": (0, total)}
    k = max(est, key=est.get)
    rows_cov = float(np.isin(yi, sorted({c for pr in est for c in pr})).mean())
    return {"per_pair": per, "max": est[k], "argmax": k, "coverage": (len(est), total), "row_coverage": rows_cov}


def label_only_reference_auc(s, label, fit_idx, score_idx, min_support: int, alpha: float = 1.0):
    """AUC of predicting attribute s from the task label alone (empirical P(s | label) on fit rows,
    Laplace smoothing alpha). The reference that a perfectly useful output cannot avoid."""
    s = np.asarray(s).astype(np.int64)
    lab = np.asarray(label).astype(np.int64)
    fi, si = _idx(len(s), fit_idx), _idx(len(s), score_idx)
    K, L = int(s.max()) + 1, int(lab.max()) + 1
    T = np.full((L, K), alpha)
    np.add.at(T, (lab[fi], s[fi]), 1.0)
    P = T / T.sum(1, keepdims=True)
    return macro_ovr_auc(s[si], P[lab[si]], min_support)


# --------------------------------------------------------------------------------------------------
# health (scale-dependent)
# --------------------------------------------------------------------------------------------------


def health(H, std_min: float = 0.5, er_min: float = 2.0) -> dict:
    """PCRL repr_health (run_v2_dataset.py@b158abc54:153-175): per_dim_std_mean (ddof=0) and
    effective_rank = exp(entropy(s^2/sum s^2)) of the centred matrix. Labelled with the PCRL thresholds.

    SCALE-DEPENDENT: per_dim_std scales linearly with H, so a harmless rescaling flips the label while
    every invariant leakage measure is unchanged. Never use the label as evidence about leakage.
    """
    H = np.asarray(H, dtype=np.float64)
    std = float(H.std(axis=0).mean())
    s = np.linalg.svd(H - H.mean(0, keepdims=True), compute_uv=False)
    p = s ** 2 / max(float((s ** 2).sum()), 1e-12)
    er = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    label = "healthy" if (std >= std_min and er >= er_min) else "collapsed"
    return {"per_dim_std_mean": std, "effective_rank": er, "label": label,
            "thresholds": {"per_dim_std_mean": std_min, "effective_rank": er_min},
            "scale_dependent": True}


def per_class_r2(H, y, lam: float = 1e-6) -> dict:
    """PCRL dominant-axis convention (pcrl/evaluation/certificates.py::compute_dominant_axis_r2):
    in-sample ridge R2 of each indicator 1[y=k] on centred H; returns per-class list, max (R2_DA), argmax,
    priors. Pooled one-hot R2 = sum_k w_k R2_k with w_k = pi_k(1-pi_k)/sum_j pi_j(1-pi_j) (same lam)."""
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    K = int(yi.max()) + 1
    Hc = H - H.mean(0)
    G = Hc.T @ Hc + lam * np.eye(H.shape[1])
    out = []
    for k in range(K):
        z = (yi == k).astype(np.float64)
        zc = z - z.mean()
        w = np.linalg.solve(G, Hc.T @ zc)
        out.append(max(0.0, 1.0 - float(((zc - Hc @ w) ** 2).sum()) / max(float((zc ** 2).sum()), 1e-12)))
    pri = np.bincount(yi, minlength=K) / len(yi)
    return {"per_class": out, "max": float(max(out)), "argmax": int(np.argmax(out)), "priors": pri.tolist()}


def contrast_report(H, y, P, min_support: int, lam: float = 1e-6, eps_rel: float = 1e-6) -> dict:
    """Side-by-side, DISTINCT quantities for class-contrast leakage (never merged into one number):
    pooled one-hot R2, per-class R2 (mean and max = dominant axis), rho1^2 and the R2 of its canonical
    contrast (linear-recoverability scale), macro OvR AUC and the max supported pairwise AUC (ranking scale
    of an attacker's probabilities P)."""
    pc = per_class_r2(H, y, lam)
    c = cca_rho2(H, y, eps_rel)
    wp = pairwise_auc(y, P, min_support)
    return {"pooled_onehot_r2": r2_onehot_ridge(H, y, lam),
            "per_class_r2_mean": float(np.mean(pc["per_class"])), "per_class_r2_max": pc["max"],
            "rho1_sq": c["value"] if isinstance(c, dict) else c,
            "canonical_contrast_r2": contrast_r2(H, y, c["contrast"]) if isinstance(c, dict) else c,
            "macro_ovr_auc": macro_ovr_auc(y, P, min_support),
            "max_pair_auc": wp["max"], "max_pair": wp["argmax"], "pair_coverage": wp["coverage"]}


def r2_onehot_relridge(H, y, fit_idx, score_idx, rho: float = 0.0, floor_rel: float = 1e-6):
    """Methodology recipe R02: affine one-hot least squares fit on fit rows (means from fit rows), numerical
    floor (drop covariance directions with eigenvalue < floor_rel * largest) and RELATIVE ridge
    lambda = rho * tr(S)/d; scored on score rows with SS_tot around the fit-row means. Invariant to rescaling
    H (both the floor and the penalty are relative). Not clamped."""
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    fi, si = _idx(len(yi), fit_idx), _idx(len(yi), score_idx)
    if len(np.unique(yi[si])) < 2:
        return NotEstimable("fewer than 2 classes in score rows", counts={"score": class_counts(yi[si])})
    Y = np.eye(int(yi.max()) + 1)[yi]
    muH, muY = H[fi].mean(0), Y[fi].mean(0)
    Hc, Yc = H[fi] - muH, Y[fi] - muY
    S = Hc.T @ Hc / len(fi)
    ev, U = np.linalg.eigh(S)
    keep = ev >= floor_rel * ev.max()
    U, ev = U[:, keep], ev[keep]
    lam = rho * float(np.trace(S)) / H.shape[1]
    B = U @ np.diag(1.0 / (ev + lam)) @ U.T @ (Hc.T @ Yc / len(fi))
    pred = (H[si] - muH) @ B + muY
    res = Y[si] - pred
    tot = Y[si] - muY
    return 1.0 - float((res ** 2).sum()) / max(float((tot ** 2).sum()), 1e-300)


def cca_rho2_heldout(H, y, fit_idx, score_idx, eps_rel: float = 1e-6):
    """Held-out rho1^2: canonical directions (class contrast a, representation direction b) estimated on fit
    rows, squared correlation of (H b, Y a) measured on score rows."""
    H = np.asarray(H, dtype=np.float64)
    yi = np.asarray(y).astype(np.int64)
    fi, si = _idx(len(yi), fit_idx), _idx(len(yi), score_idx)
    c = cca_rho2(H[fi], yi[fi], eps_rel)
    if not isinstance(c, dict):
        return c
    a = c["contrast"]
    if len(a) < int(yi.max()) + 1:
        a = np.r_[a, np.zeros(int(yi.max()) + 1 - len(a))]
    t_fit = a[yi[fi]]
    Hc = H[fi] - H[fi].mean(0)
    Shh = Hc.T @ Hc / len(fi)
    Shh += eps_rel * np.trace(Shh) / H.shape[1] * np.eye(H.shape[1])
    b = np.linalg.solve(Shh, Hc.T @ (t_fit - t_fit.mean()) / len(fi))
    u, v = H[si] @ b, a[yi[si]]
    if u.std() == 0 or v.std() == 0:
        return NotEstimable("degenerate held-out projection", counts={"score": class_counts(yi[si])})
    return float(np.corrcoef(u, v)[0, 1] ** 2)


def prior_from_fit(y_fit, K: int, alpha: float = 0.0) -> np.ndarray:
    c = np.bincount(np.asarray(y_fit).astype(np.int64), minlength=K).astype(np.float64) + alpha
    return c / c.sum()


def brier_skill(y, P, prior) -> float:
    """1 - SSE(P vs one-hot) / SSE(prior vs one-hot); prior from attacker_fit frequencies (methodology)."""
    yi = np.asarray(y).astype(np.int64)
    P = _scores_matrix(P)
    Y = np.eye(P.shape[1])[yi]
    return 1.0 - float(((P - Y) ** 2).sum()) / float(((np.asarray(prior)[None, :] - Y) ** 2).sum())


def logloss_reduction(y, P, prior, eps: float = 1e-12) -> float:
    """Prior log-loss minus model log-loss (nats per row); unclipped (negative = worse than prior)."""
    yi = np.asarray(y).astype(np.int64)
    P = _scores_matrix(P)
    ll_m = -np.log(np.clip(P[np.arange(len(yi)), yi], eps, 1)).mean()
    ll_p = -np.log(np.clip(np.asarray(prior)[yi], eps, 1)).mean()
    return float(ll_p - ll_m)

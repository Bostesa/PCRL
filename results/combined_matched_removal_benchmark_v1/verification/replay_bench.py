#!/usr/bin/env python3
"""Independent replay of the matched removal benchmark (ROLE 4, independent verifier).

Written from notes/BENCH_DESIGN.md and the pilot's own independent replay
(results/combined_stored_model_pilot_v1/verification/replay.py).  It does NOT import or read
stored_model_eval.  It never refits an attacker or a probe and never tunes on assessment rows.
Everything it reports is recomputed from

  * the hash-pinned inputs listed in INPUTS_INDEX.json (record keys, role arrays, labels,
    forward caches holding row_id / rep_p<i> / logits_<purpose>),
  * the runner's saved predictions units/<unit_id>/preds.npz (assessment and attacker_val rows),
  * the runner's saved LEACE maps (proj_left, proj_right, mean/bias), applied in numpy,
  * closed-form quantities (fixed-ridge G1, LEACE re-derivation, cross-covariances, OLS R^2).

Dependencies: numpy and the standard library only.

Frozen interpretation choices (made before the real run; see FROZEN_READINGS below):
  * role uniform u = int(sha256(salt + key)[:8], 16) / 2**32 (pilot convention); the 16-hex variant
    is computed only as a diagnostic when a role mismatches;
  * macro OvR AUC = mean over supported classes of AUC(P[:, c] ; y == c vs all other assessment rows);
    worst-class / worst-pair = MAXIMUM (most recoverable); pair (j < k) score p_k / (p_j + p_k), positives k;
  * replicate summary = mean over encoder seeds x release seeds x attacker seeds of the per-unit statistic,
    inside every bootstrap replicate on the same resampled assessment units (one bootstrap per dataset);
  * primary bounds: one-sided percentile bounds LCB = q(alpha), UCB = q(1 - alpha), alpha = 0.05 / 24,
    B = 20000, numpy 'linear' quantiles; exploratory: 90 % two-sided percentile, B = 2000;
  * worst-class / worst-pair derived bounds: LCB(max) = max_k q_k(a / K), UCB(max) = max_k q_k(1 - a / K),
    q_k = quantiles of the replicate-summary statistic of class / pair k, a = one-sided level of the
    interval in question (0.05 for the exploratory 90 % interval).  These bound max_k theta_k where
    theta_k is the seed-averaged class-k AUC; the point estimate consistent with that target is
    max_k (seed-mean AUC_k).  The seed-mean of per-unit maxima is also computed and named.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- frozen constants (BENCH_DESIGN)
ROLE_SALTS = {"adult": "pilot-roles-v1|", "hmda": "bench-roles-hmda-v1|"}
FALLBACK_SALT = "bench-defense-fallback-v1|"
FALLBACK_SHARE = 0.30
ROLE_CUTS = (("attacker_fit", 0.50), ("attacker_val", 0.65), ("assessment", 1.0))
TEST_ROLES = ("attacker_fit", "attacker_val", "assessment")
SUPPORT_MIN = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
DEFENSE_MIN = 100
EXPECTED_COUNTS = {"adult": {"attacker_fit": 7571, "attacker_val": 2239, "assessment": 5250},   # pilot-roles-v1
                   "hmda": {"attacker_fit": 6794, "attacker_val": 2089, "assessment": 4778}}    # admission, 2026-10-02
CELLS = {
    "adult": {"purpose": "income_prediction", "task": "income", "target": "sex", "policy": ("race", "sex")},
    "hmda": {"purpose": "underwriting", "task": "loan_decision", "target": "race", "policy": ("race", "ethnicity")},
}
SIGMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS3 = (0, 1, 2)
TAU = 0.05
BAR = 0.55
SECONDARY_BARS = (0.52, 0.60)
NI_MARGIN = -0.01
SIGMA_STAR_BAR = 0.55
FAMILY_SIZE = 24
ALPHA_FAMILY = 0.05 / FAMILY_SIZE
B_EXPL, SEED_EXPL, LEVEL_EXPL = 2000, 20261003, 0.90
B_PRIM, SEED_PRIM = 20000, 20261004
LAMBDA = 1e-6
LL_CLIP = 1e-12
LEACE_SVD_TOL = 0.01          # concept-erasure 0.2.4 LeaceFitter default

# ---------------------------------------------------------------- tolerances (frozen BEFORE the real run)
POINT_TOL = 1e-9              # point estimates recomputed from saved predictions (exact arithmetic up to fp)
BOUND_SE_MULT = 6.5           # bound tolerance = 6.5 x replay MC SE + 5e-5: independent RNG streams at equal B;
BOUND_FLOOR = 5e-5            #   a difference of two independent percentile estimates has sd ~ sqrt(2) SE,
                              #   so 6.5 SE ~ 4.6 sd (per-comparison false alarm ~4e-6)
G1_REDERIVE_TOL = 1e-6        # saved G1_pred vs closed-form fixed-ridge re-derivation (max abs)
LEACE_P_TOL = 1e-7            # (unit test only) P re-derivation vs official on well-conditioned problems
LEACE_REDERIVE_TOL = 1e-6     # erased defense_fit rows: saved map vs numpy re-derivation (max abs, relative to max|h|)
LEACE_MEAN_TOL = 1e-9         # saved bias vs mean of defense_fit representations (max abs, relative to max|h|)
LEACE_XCOV_TOL = 1e-9         # (synthetic expectation only) exact-zero cross-covariance when nothing is truncated
LEACE_OLS_TOL = 1e-9          # (synthetic expectation only) fit-row OLS R^2 when nothing is truncated
# native B/C status (coordinator decision): ||W P Sigma_xz||_2 <= svd_tol (+1e-9 relative, 1e-12 absolute)
TRANSFORM_TOL = 1e-9          # saved transformed rows vs X - (X - mean) R^T L^T (relative to max|h|)
ALIAS_TOL = 1e-10             # B == C alias: max |x_B - x_C| over all rows, relative to max|h| (P descriptive)
N0_MIXED_TOL = 1e-6          # N0 via float32 centring/Gram (historical mixed precision): BLAS-order sensitive
PROB_TOL = 1e-9               # identity of saved probability arrays (aliases, ignore candidates)

FROZEN_READINGS = {
    "role_uniform": "int(sha256(salt+key).hexdigest()[:8],16)/2**32",
    "worst": "maximum over supported classes/pairs",
    "replicate_summary": "mean over encoder x release x attacker seeds inside each replicate",
    "derived_max_bound": "LCB=max_k q_k(a/K), UCB=max_k q_k(1-a/K); a=0.05 (exploratory 90%)",
    "worst_point_consistent": "max_k seed-mean AUC_k",
    "primary_bounds": "one-sided percentile q(alpha), q(1-alpha), alpha=0.05/24, B=20000",
    "support": "scoring support per role 100/30/100 (attacker_fit/attacker_val/assessment) for every arm; "
               "eraser concept = full declared one-hot when every class has >=100 defense_fit rows "
               "(coordinator decision 2026-10-02)",
    "u2ni": "mean_s[acc_m(s)] - mean_s[acc_A(s)] (paired on encoder seed and on assessment units)",
    "leace_native": "primary B/C native status = ||W P Sigma_xz||_2 <= svd_tol (official bound), from saved arrays "
                    "and recomputed rows; exact-zero cross-covariance / OLS R^2 descriptive (coordinator 2026-10-02)",
    "bootstrap_rng": "numpy default_rng([seed, dataset_index]); units drawn in batches of 200",
}


# ---------------------------------------------------------------- small utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def fmt_sigma(s):
    s = float(s)
    return str(int(s)) if s == int(s) else repr(s)


# ---------------------------------------------------------------- roles and support
def u01(salt, key, hexdigits=8):
    return int(hashlib.sha256((salt + str(key)).encode("utf-8")).hexdigest()[:hexdigits], 16) / 16 ** hexdigits


def role_of_key(salt, key, hexdigits=8):
    u = u01(salt, key, hexdigits)
    for name, cut in ROLE_CUTS:
        if u < cut:
            return name
    return ROLE_CUTS[-1][0]


def recompute_roles(salt, keys, split, defense_source, hexdigits=8):
    """Expected role per row.  Test rows: hash roles.  Train rows: defense_fit unless the record key also occurs
    in the test split (then 'excluded').  Fallback: 30 % of attacker_fit units become defense_fit."""
    keys = np.asarray(keys).astype(str)
    split = np.asarray(split).astype(str)
    out = np.full(len(keys), "", dtype=object)
    test = split == "test"
    test_keys = set(keys[test].tolist())
    for i in np.flatnonzero(test):
        out[i] = role_of_key(salt, keys[i], hexdigits)
    if defense_source == "fallback":
        for i in np.flatnonzero(test):
            if out[i] == "attacker_fit" and u01(FALLBACK_SALT, keys[i], hexdigits) < FALLBACK_SHARE:
                out[i] = "defense_fit"
        out[~test] = "unused_train"
    else:
        for i in np.flatnonzero(~test):
            out[i] = "excluded" if keys[i] in test_keys else "defense_fit"
    return out.astype(str)


def support_sets(y, roles, need_defense):
    y = np.asarray(y).astype(int)
    classes = sorted(int(c) for c in np.unique(y[np.isin(roles, list(TEST_ROLES) + ["defense_fit"])]))
    mins = dict(SUPPORT_MIN)
    if need_defense:
        mins["defense_fit"] = DEFENSE_MIN
    counts = {c: {r: int(np.sum((y == c) & (roles == r))) for r in list(SUPPORT_MIN) + ["defense_fit"]}
              for c in classes}
    sup, reasons = [], {}
    for c in classes:
        bad = [f"{r}<{m}" for r, m in mins.items() if counts[c][r] < m]
        if bad:
            reasons[c] = bad
        else:
            sup.append(c)
    return {"classes": classes, "counts": counts, "supported_classes": sup,
            "supported_pairs": [list(p) for p in combinations(sup, 2)],
            "unsupported_reasons": reasons, "estimable": len(sup) >= 2}


def _find_list(obj, names):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names and isinstance(v, list):
                return v
        for v in obj.values():
            r = _find_list(v, names)
            if r is not None:
                return r
    return None


def parse_runner_supported(js):
    sens = js.get("sensitive", js) if isinstance(js, dict) else js
    cls = _find_list(sens, {"supported_classes", "classes_supported"})
    prs = _find_list(sens, {"supported_pairs", "pairs_supported"})
    out = {}
    if cls is not None:
        out["supported_classes"] = sorted(int(float(c)) for c in cls)
    if prs is not None:
        out["supported_pairs"] = sorted(sorted(int(float(x)) for x in p) for p in prs)
    return out


# ---------------------------------------------------------------- weighted statistics
class WAUC:
    """Weighted Mann-Whitney AUC with ties averaged; weights = cluster multiplicities (rows of W)."""

    def __init__(self, score, pos, rows=None):
        score = np.asarray(score, dtype=np.float64)
        idx = np.arange(len(score)) if rows is None else np.flatnonzero(rows)
        s = score[idx]
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        self.cols = idx[order]
        self.pos = np.asarray(pos)[idx][order].astype(np.float64)
        self.neg = 1.0 - self.pos
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]]) if len(ss) else np.array([0])

    def __call__(self, W):
        Ws = W[:, self.cols]
        Pw = np.add.reduceat(Ws * self.pos, self.starts, axis=1)
        Nw = np.add.reduceat(Ws * self.neg, self.starts, axis=1)
        below = np.cumsum(Nw, axis=1) - Nw
        num = (Pw * (below + 0.5 * Nw)).sum(axis=1)
        den = Pw.sum(axis=1) * Nw.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


class Lazy:
    """Defers construction of an expensive evaluator (sorting) until first use."""

    def __init__(self, factory):
        self.factory, self.obj = factory, None

    def __call__(self, W):
        if self.obj is None:
            self.obj = self.factory()
        return self.obj(W)


class Ctx:
    def __init__(self, W):
        self.W = W
        self.cache = {}

    def get(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn(self.W)
        return self.cache[key]


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def ll_rows(ptrue, eps=LL_CLIP):
    return -np.log(np.clip(np.asarray(ptrue, dtype=np.float64), eps, 1.0 - eps))


def onehot(y, classes):
    classes = list(classes)
    pos = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y), len(classes)))
    for i, v in enumerate(np.asarray(y).astype(int)):
        j = pos.get(int(v))
        if j is not None:
            Y[i, j] = 1.0
    return Y


# ---------------------------------------------------------------- closed forms
def ridge_onehot(H_fit, Y_fit, H_score, lam=LAMBDA):
    """Unnormalised centred-Gram fixed ridge (lambda = 1e-6), float64.  Returns predictions on H_score and the
    fit means (the SS_tot reference)."""
    Hf = np.asarray(H_fit, dtype=np.float64)
    mu = Hf.mean(axis=0, keepdims=True)
    Hc = Hf - mu
    ybar = Y_fit.mean(axis=0, keepdims=True)
    Wst = np.linalg.solve(Hc.T @ Hc + lam * np.eye(Hc.shape[1]), Hc.T @ (Y_fit - ybar))
    return (np.asarray(H_score, dtype=np.float64) - mu) @ Wst + ybar, ybar.ravel()


def ridge_onehot_r2_insample(H, Y, lam=LAMBDA, float32_gram=False):
    """Historical native check: in-sample one-hot ridge R^2 (float32 centring/Gram when float32_gram)."""
    if float32_gram:
        Hf = np.asarray(H, dtype=np.float32)
        Hc = Hf - Hf.mean(axis=0, keepdims=True)
        gram = Hc.T @ Hc + lam * np.eye(Hc.shape[1])
    else:
        Hc = np.asarray(H, dtype=np.float64)
        Hc = Hc - Hc.mean(axis=0, keepdims=True)
        gram = Hc.T @ Hc + lam * np.eye(Hc.shape[1])
    ybar = Y.mean(axis=0, keepdims=True)
    Wst = np.linalg.solve(np.asarray(gram, np.float64), np.asarray(Hc.T @ (Y - ybar), np.float64))
    pred = Hc @ Wst + ybar
    return 1.0 - float(np.sum((Y - pred) ** 2)) / max(float(np.sum((Y - ybar) ** 2)), 1e-12)


def cross_cov(X, Z):
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64)
    return (X - X.mean(0)).T @ (Z - Z.mean(0)) / (len(X) - 1)


def affine_ols_r2(X, Z):
    """Pooled one-hot R^2 of Z on [1, X], in sample, minimum-norm least squares (handles rank deficiency)."""
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64)
    Xc = X - X.mean(0)
    Zc = Z - Z.mean(0)
    beta, *_ = np.linalg.lstsq(Xc, Zc, rcond=None)
    res = Zc - Xc @ beta
    return 1.0 - float(np.sum(res ** 2)) / max(float(np.sum(Zc ** 2)), 1e-300)


def _shrink(S, n):
    """Optimal linear shrinkage (concept-erasure 0.2.4 shrinkage.optimal_linear_shrinkage), re-typed in numpy."""
    p = S.shape[-1]
    eye = np.eye(p)
    trS = np.trace(S)
    sigma0 = eye * trS / p
    s0n = np.sum(sigma0 ** 2)
    Sn = np.sum(S ** 2)
    prod = np.trace(S @ sigma0)
    top = trS * trS * s0n / n
    bottom = Sn * s0n - prod * prod
    eps = np.finfo(np.float64).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod + eps) / (s0n + eps)
    return alpha * S + beta * sigma0


def leace_numpy(X, Z, svd_tol=LEACE_SVD_TOL, shrinkage=True, constrain_cov_trace=True):
    """Numpy re-derivation of concept-erasure 0.2.4 LeaceFitter(method='leace', affine=True) on one batch.
    Returns P (= I - proj_left @ proj_right), bias, and diagnostics."""
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64).reshape(len(X), -1)
    n, d = X.shape
    mx, mz = X.mean(0), Z.mean(0)
    Sxx = X.T @ (X - mx)              # Welford from zero mean, single batch
    Sxz = X.T @ (Z - mz)
    S_hat = (Sxx + Sxx.T) / 2
    sigma = _shrink(S_hat / n, n) if shrinkage else S_hat / (n - 1)
    sxz = Sxz / (n - 1)
    L, V = np.linalg.eigh(sigma)
    mask = L > (L[-1] * d * np.finfo(np.float64).eps)
    L = np.clip(L, 0.0, None)
    with np.errstate(divide="ignore"):
        W = (V * np.where(mask, 1.0 / np.sqrt(np.where(mask, L, 1.0)), 0.0)) @ V.T
    Winv = (V * np.where(mask, np.sqrt(L), 0.0)) @ V.T
    u, s, _ = np.linalg.svd(W @ sxz, full_matrices=False)
    keep = s > svd_tol
    u = u * keep
    Pl, Pr = Winv @ u, u.T @ W
    eye = np.eye(d)
    P = eye - Pl @ Pr
    trace_mixed, alpha_mix = False, None
    if constrain_cov_trace:
        old, new = np.trace(sigma), np.trace(P @ sigma @ P.T)
        if new > old:
            trace_mixed = True
            Q = eye - u @ u.T
            x, y_, z_, w = new, 2 * np.trace(P @ sigma @ Q.T), np.trace(Q @ sigma @ Q.T), old
            disc = np.sqrt(4 * w * x - 4 * w * y_ + 4 * w * z_ - 4 * x * z_ + y_ ** 2)
            a1 = (-y_ / 2 + z_ - disc / 2) / (x - y_ + z_)
            a2 = (-y_ / 2 + z_ + disc / 2) / (x - y_ + z_)
            alpha_mix = float(np.clip(a1 if a1 > 0 else a2, 0, 1))
            P = alpha_mix * P + (1 - alpha_mix) * Q
    return {"P": P, "bias": mx, "singular_values": s, "n_truncated": int(np.sum(~keep)),
            "n_kept": int(np.sum(keep)), "trace_mixed": trace_mixed, "alpha_mix": alpha_mix,
            "whitening_rank": int(mask.sum())}


def leace_apply(X, proj_left, proj_right, bias):
    X = np.asarray(X, np.float64)
    return X - ((X - bias) @ proj_right.T) @ proj_left.T


# ---------------------------------------------------------------- bootstrap engine
def cluster_index(units):
    uniq, inv = np.unique(units, return_inverse=True)
    return inv, len(uniq)


def run_bootstrap(stats, inv, n_units, B, seed, batch=200):
    """Cluster bootstrap over assessment record units; predictors fixed.  seed may be an int or a list."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in stats}
    done = 0
    while done < B:
        b = min(batch, B - done)
        draws = rng.integers(0, n_units, size=(b, n_units))
        flat = (draws + (np.arange(b)[:, None] * n_units)).ravel()
        counts = np.bincount(flat, minlength=b * n_units).reshape(b, n_units).astype(np.float64)
        ctx = Ctx(counts[:, inv])
        for k, fn in stats.items():
            out[k][done:done + b] = fn(ctx)
        done += b
    return out


def mc_se_quantile(x, p):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    B = len(x)
    if B < 10:
        return float("nan")
    d = math.sqrt(p * (1 - p) / B)
    lo, hi = max(p - 2 * d, 0.0), min(p + 2 * d, 1.0)
    slope = (np.quantile(x, hi, method="linear") - np.quantile(x, lo, method="linear")) / (hi - lo)
    return float(slope * d)


def qbound(x, p):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None, None
    return float(np.quantile(x, p, method="linear")), mc_se_quantile(x, p)


def interval(x, lo_p, hi_p):
    lo, sl = qbound(x, lo_p)
    hi, sh = qbound(x, hi_p)
    if lo is None:
        return None
    return {"lower": lo, "upper": hi, "mcse_lower": sl, "mcse_upper": sh,
            "n_finite": int(np.sum(np.isfinite(x)))}


def derived_max_bounds(comp_reps, a_side):
    """Simultaneous per-component bounds at a / K and their maxima (the frozen worst-class / pair rule)."""
    K = len(comp_reps)
    los, his = {}, {}
    for k, x in comp_reps.items():
        los[k] = qbound(x, a_side / K)
        his[k] = qbound(x, 1 - a_side / K)
    klo = max(los, key=lambda k: los[k][0])
    khi = max(his, key=lambda k: his[k][0])
    return {"lower": los[klo][0], "upper": his[khi][0], "K": K, "a_side": a_side,
            "argmax_lower": klo, "argmax_upper": khi,
            "mcse_lower": max(v[1] for v in los.values()), "mcse_upper": max(v[1] for v in his.values()),
            "per_component": {str(k): [los[k][0], his[k][0]] for k in comp_reps}}


def ds_of(k):
    """Dataset of a statistic name: '<ds>__...' for unit statistics, 'SUM|<ds>|...' / 'U2NI|<ds>|...' otherwise."""
    return k.split("|")[1] if k.startswith(("SUM|", "U2NI|")) else k.split("__", 1)[0]


def bound_tol(se):
    return BOUND_SE_MULT * (se if se is not None and np.isfinite(se) else 0.0) + BOUND_FLOOR


# ---------------------------------------------------------------- decisions
def decide_bar(lcb, ucb, bar):
    if lcb > bar:
        return "ABOVE"
    if ucb < bar:
        return "BELOW"
    return "UNRESOLVED"


def decide_ni(lcb, ucb, margin=NI_MARGIN):
    if lcb >= margin:
        return "NONINFERIOR"
    if ucb < margin:
        return "INFERIOR"
    return "UNRESOLVED"


# ---------------------------------------------------------------- report items
class Report:
    """Items; quiet=True PASS/INFO items are counted (by status and scope) but not stored, to keep the output
    compact.  Every non-PASS item is always stored."""

    def __init__(self):
        self.items = []
        self.quiet_counts = {}

    def add(self, check, scope, runner, replay, tol=None, status=None, note=None, kind="value", quiet=False):
        diff = None
        if isinstance(runner, (int, float, np.floating)) and isinstance(replay, (int, float, np.floating)) \
                and not isinstance(runner, bool) and not isinstance(replay, bool) \
                and np.isfinite(runner) and np.isfinite(replay):
            diff = abs(float(runner) - float(replay))
        if status is None:
            if kind == "value" and diff is not None:
                status = "PASS" if diff <= (tol if tol is not None else POINT_TOL) else "FAIL"
            else:
                status = "PASS" if _jsonable(runner) == _jsonable(replay) else "FAIL"
        if quiet and status in ("PASS", "INFO"):
            sc = self.quiet_counts.setdefault(scope, {})
            sc[status] = sc.get(status, 0) + 1
            return None
        it = {"check": check, "scope": scope, "runner_value": _jsonable(runner), "replay_value": _jsonable(replay),
              "abs_diff": diff, "tolerance": tol, "status": status}
        if note:
            it["note"] = note
        self.items.append(it)
        return it


# ---------------------------------------------------------------- layout parsing
UNIT_RE = re.compile(r"^(?P<ds>[a-z0-9]+)__s(?P<seed>\d+)__(?P<purpose>[a-z0-9_]+?)__(?P<attr>[a-z0-9_]+?)__"
                     r"(?P<arm>A|B|C|D_sigma(?P<sigma>[0-9.]+)_rs(?P<rs>\d+))$")


def parse_unit_id(uid):
    m = UNIT_RE.match(uid)
    if not m:
        return None
    d = m.groupdict()
    arm = d["arm"]
    return {"ds": d["ds"], "seed": int(d["seed"]), "purpose": d["purpose"], "attr": d["attr"],
            "arm": "D" if arm.startswith("D_") else arm,
            "sigma": float(d["sigma"]) if d["sigma"] else None, "rs": int(d["rs"]) if d["rs"] else None,
            "group": f"D_sigma{fmt_sigma(d['sigma'])}" if d["sigma"] else arm}


MAP_RE = re.compile(r"^(?P<ds>[a-z0-9]+)__s(?P<seed>\d+)__(?P<purpose>[a-z0-9_]+?)__(?P<kind>B|C)_(?P<attrs>[a-z0-9_+]+)$")


def parse_map_id(mid):
    m = MAP_RE.match(mid)
    if not m:
        return None
    return {"ds": m["ds"], "seed": int(m["seed"]), "purpose": m["purpose"], "kind": m["kind"],
            "attrs": m["attrs"].split("+")}


def canon_surface(s):
    t = s.lower().replace("-", "_")
    if "rep" in t and "out" in t:
        return "repPLUSoutputs"
    if t.startswith("rep"):
        return "rep"
    if t.startswith("out"):
        return "outputs"
    return s


PKEY_RE = re.compile(r"^(?P<kind>P|V|VAL)__(?P<surf>[A-Za-z]+)__(?P<recipe>[A-Za-z0-9_]+?)(?:__as(?P<as>\d+))?$")


def parse_pkey(k):
    m = PKEY_RE.match(k)
    if not m:
        return None
    return ("V" if m["kind"] in ("V", "VAL") else "P"), canon_surface(m["surf"]), m["recipe"], \
        (int(m["as"]) if m["as"] is not None else None)


def _entry_path(v):
    if isinstance(v, dict):
        return v.get("path"), v.get("sha256")
    return v, None


@dataclass
class Config:
    sigma_grid: tuple = SIGMA_GRID
    encoder_seeds: tuple = SEEDS3
    release_seeds: tuple = SEEDS3
    attacker_seeds: tuple = SEEDS3
    b_prim: int = B_PRIM
    b_expl: int = B_EXPL
    seed_prim: int = SEED_PRIM
    seed_expl: int = SEED_EXPL
    expected_counts: dict | None = field(default_factory=lambda: {k: dict(v) for k, v in EXPECTED_COUNTS.items()})
    cells: dict = field(default_factory=lambda: {k: dict(v) for k, v in CELLS.items()})
    role_salts: dict = field(default_factory=lambda: dict(ROLE_SALTS))
    expl_scope: str = "summaries"     # "summaries" or "all"


# ---------------------------------------------------------------- dataset inputs
class Dataset:
    """Inputs of one dataset, read from the admission INPUTS_INDEX.json schema (bench_v1.inputs_index/v1):
    labels npz (row_id, split, unit, record_key, canon_key, role, <attrs>, task_<task>), roles npz
    (<role>__row_id / <role>__unit), forward npz (row_id, split, rep_p<i>, logits_<purpose>), tier1 cell."""

    def __init__(self, name, index, spec, rep, cfg):
        self.name, self.index, self.spec, self.cfg = name, index, spec, cfg
        self.cell = cfg.cells[name]
        t1 = spec.get("tier1", {})
        rep.add(f"{name}: Tier-1 cell in the index == frozen design", "inputs",
                [t1.get("purpose"), t1.get("task"), t1.get("target"), sorted(t1.get("policy", []))],
                [self.cell["purpose"], self.cell["task"], self.cell["target"], sorted(self.cell["policy"])],
                kind="exact")
        pur = spec["purposes"][self.cell["purpose"]]
        rep.add(f"{name}: policy set == disallowed_attrs of the purpose", "inputs", sorted(pur["disallowed_attrs"]),
                sorted(self.cell["policy"]), kind="exact")
        self.rep_key, self.logits_key = pur["rep_key"], pur["logits_key"]
        self.task_key = pur.get("labels_task_key", f"task_{self.cell['task']}")
        self.attr_dims = dict(pur.get("disallowed_attr_dims", {}))

        def pinned(path, sha):
            p = Path(path).expanduser()
            rep.add(f"{name}: input sha256 {p.name}", "inputs", sha, sha256_file(p), kind="exact")
            return p
        lb = load_npz(pinned(spec["labels_npz"], spec["labels_sha256"]))
        rl = load_npz(pinned(spec["roles_npz"], spec["roles_sha256"]))
        self.row_id = np.asarray(lb["row_id"]).astype(np.int64)
        self.pos = {int(r): i for i, r in enumerate(self.row_id)}
        rka = spec.get("record_key_array", {})
        self.record_key = np.asarray(lb[rka.get("key", "record_key")]).astype(str)
        self.canon_key = np.asarray(lb[rka.get("canon_key", "canon_key")]).astype(str) \
            if rka.get("canon_key", "canon_key") in lb else self.record_key
        for k, arr in (("sha256_record_key", self.record_key), ("sha256_canon_key", self.canon_key)):
            if rka.get(k):
                rep.add(f"{name}: {k} ('\\n'.join convention)", "inputs", rka[k],
                        hashlib.sha256("\n".join(arr.tolist()).encode()).hexdigest(), kind="exact")
        self.split = np.asarray(lb["split"]).astype(str)
        ra = spec.get("role_array", {})
        self.role = np.asarray(lb[ra.get("key", "role")]).astype(str)
        self.unit = np.asarray(lb[ra.get("unit_key", "unit")]).astype(np.int64)
        if ra.get("sha256"):
            rep.add(f"{name}: role column sha256", "inputs", ra["sha256"],
                    hashlib.sha256("\n".join(self.role.tolist()).encode()).hexdigest(), kind="exact")
        # per-role arrays in the roles npz == the role column; index counts and hashes
        for r, meta in spec.get("roles", {}).items():
            rid = np.asarray(rl.get(f"{r}__row_id", np.array([], np.int64))).astype(np.int64)
            mine = np.sort(self.row_id[self.role == r])
            rep.add(f"{name}: roles npz {r} row ids == role column", "ids_roles",
                    int(len(np.setxor1d(rid, mine))), 0, kind="exact")
            if f"{r}__unit" in rl:
                ui = np.array([self.pos[int(x)] for x in rid], dtype=np.int64)
                rep.add(f"{name}: roles npz {r} units == unit column", "ids_roles",
                        int(np.sum(np.asarray(rl[f"{r}__unit"]).astype(np.int64) != self.unit[ui])) if len(ui) else 0,
                        0, kind="exact")
                if meta.get("unit_sha256"):
                    rep.add(f"{name}: index {r} unit_sha256", "inputs", meta["unit_sha256"],
                            hashlib.sha256(np.asarray(rl[f"{r}__unit"]).astype("<i8").tobytes()).hexdigest(), kind="exact")
            if meta.get("row_id_sha256"):
                rep.add(f"{name}: index {r} row_id_sha256", "inputs", meta["row_id_sha256"],
                        hashlib.sha256(rid.astype("<i8").tobytes()).hexdigest(), kind="exact")
            rep.add(f"{name}: index {r} n_rows / n_units", "inputs", [meta.get("n_rows"), meta.get("n_units")],
                    [int(np.sum(self.role == r)), int(len(np.unique(self.unit[self.role == r])))], kind="exact")
        self.labels = {k: np.asarray(lb[k]) for k in lb
                       if k not in ("row_id", "split", "unit", "record_key", "canon_key", "role")}
        self.defense_source = "fallback" if str(spec.get("defense_route", "PREFERRED")).upper() == "FALLBACK" \
            else "train_split"
        self.encoders = {}
        for s, e in spec.get("encoders", {}).items():
            self.encoders[int(s)] = {"meta": e, "forward_cache": pinned(e["forward_npz"], e["forward_sha256"])}
        self._fwd = {}

    def task_label(self, purpose):
        pu = self.spec["purposes"][purpose]
        return np.asarray(self.labels[pu.get("labels_task_key", f"task_{pu['task']}")]).astype(int)

    def label(self, attr):
        if attr == self.cell["task"] and self.task_key in self.labels:
            return np.asarray(self.labels[self.task_key]).astype(int)
        for k in (attr, f"task_{attr}"):
            if k in self.labels:
                return np.asarray(self.labels[k]).astype(int)
        raise KeyError(f"{self.name}: no label column for {attr}")

    def forward(self, seed, purpose=None):
        """(rep, logits, cache row_id order) with rep / logits in labels row order, float64."""
        purpose = purpose or self.cell["purpose"]
        if (seed, purpose) not in self._fwd:
            fc = load_npz(self.encoders[seed]["forward_cache"])
            pu = self.spec["purposes"][purpose]
            rk, lk = pu.get("rep_key", f"rep_p{pu['index']}"), pu.get("logits_key", f"logits_{purpose}")
            rid = np.asarray(fc["row_id"]).astype(np.int64)
            idx = np.array([self.pos[int(r)] for r in rid])
            H = np.empty((len(self.row_id), fc[rk].shape[1]))
            H[idx] = np.asarray(fc[rk], np.float64)
            Lg = None
            if lk in fc:
                Lg = np.empty((len(self.row_id), fc[lk].shape[1]))
                Lg[idx] = np.asarray(fc[lk], np.float64)
            self._fwd[(seed, purpose)] = (H, Lg, rid)
        return self._fwd[(seed, purpose)]


def check_roles(ds, rep):
    """Independent role recomputation.  Test rows: sha256(salt + record_key) (pilot key, verbatim).  Train rows:
    'excluded_dup' when their canon_key occurs in the test split, else defense_fit (preferred route).  Units:
    one per canon_key across both splits."""
    cfg = ds.cfg
    salt = cfg.role_salts[ds.name]
    test = ds.split == "test"
    exp = np.full(len(ds.row_id), "", dtype=object)
    for i in np.flatnonzero(test):
        exp[i] = role_of_key(salt, ds.record_key[i])
    test_canon = set(ds.canon_key[test].tolist())
    if ds.defense_source == "fallback":
        for i in np.flatnonzero(test):
            if exp[i] == "attacker_fit" and u01(FALLBACK_SALT, ds.record_key[i]) < FALLBACK_SHARE:
                exp[i] = "defense_fit"
        for i in np.flatnonzero(~test):
            exp[i] = "excluded_dup" if ds.canon_key[i] in test_canon else "unused_train"
    else:
        for i in np.flatnonzero(~test):
            exp[i] = "excluded_dup" if ds.canon_key[i] in test_canon else "defense_fit"
    exp = exp.astype(str)
    mism = int(np.sum(exp[test] != ds.role[test]))
    note = None
    if mism:
        alt = np.array([role_of_key(salt, k, 16) for k in ds.record_key[test]])
        note = f"16-hex uniform variant mismatches: {int(np.sum(alt != ds.role[test]))}"
    rep.add(f"{ds.name}: stored test-split roles == sha256('{salt}'+record_key) roles", "ids_roles", mism, 0,
            kind="exact", note=note)
    rep.add(f"{ds.name}: train-split roles == defense_fit / excluded_dup (canon_key overlap with test)", "ids_roles",
            int(np.sum(exp[~test] != ds.role[~test])), 0, kind="exact",
            note=f"recomputed excluded_dup rows: {int(np.sum(exp == 'excluded_dup'))}")
    df_saved = ds.role == "defense_fit"
    test_units = set(ds.unit[test].tolist())
    assess_units = set(ds.unit[ds.role == "assessment"].tolist())
    rep.add(f"{ds.name}: defense_fit canon keys disjoint from every test-split role", "ids_roles",
            int(sum(k in test_canon for k in ds.canon_key[df_saved])), 0, kind="exact")
    rep.add(f"{ds.name}: defense_fit units disjoint from assessment units", "ids_roles",
            int(sum(int(u) in assess_units for u in ds.unit[df_saved])), 0, kind="exact")
    if ds.defense_source != "fallback":
        rep.add(f"{ds.name}: defense_fit units disjoint from all test-split units", "ids_roles",
                int(sum(int(u) in test_units for u in ds.unit[df_saved])), 0, kind="exact")
    k2u, u2k, bad = {}, {}, 0
    for k, u in zip(ds.canon_key, ds.unit):
        if k2u.setdefault(k, int(u)) != int(u) or u2k.setdefault(int(u), k) != k:
            bad += 1
    rep.add(f"{ds.name}: record unit <-> canon_key one-to-one (both splits)", "ids_roles", bad, 0, kind="exact")
    k2r, badr = {}, 0
    for k, r in zip(ds.canon_key[test], ds.role[test]):
        if k2r.setdefault(k, r) != r:
            badr += 1
    rep.add(f"{ds.name}: duplicate test records share one role", "ids_roles", badr, 0, kind="exact")
    if ds.name == "adult":
        rep.add("adult: pilot record_key and canon_key give the same test-split partition", "ids_roles",
                int(len({(a, b) for a, b in zip(ds.record_key[test], ds.canon_key[test])}) -
                    len(set(ds.record_key[test].tolist()))), 0, kind="exact")
    exposure = int(sum(k in set(ds.canon_key[exp == "excluded_dup"].tolist()) for k in ds.canon_key[test]))
    counts = {r: int(np.sum(ds.role == r)) for r in TEST_ROLES + ("defense_fit", "excluded_dup")}
    rep.add(f"{ds.name}: role row counts", "ids_roles", None, counts, status="INFO", kind="info",
            note=f"test rows with an identical record among excluded train rows (disclosed exposure): {exposure}")
    want = (cfg.expected_counts or {}).get(ds.name)
    if want:
        want = dict(want)
        if ds.defense_source == "fallback":
            want["attacker_fit"] = want["attacker_fit"] - int(np.sum(exp == "defense_fit"))
        for r, n in want.items():
            rep.add(f"{ds.name}: role count {r}", "ids_roles", n, counts[r], kind="exact")
    return exp


# ---------------------------------------------------------------- units
class Unit:
    """units/<unit_id>/: preds.npz (assessment rows), val_preds.npz (attacker_val rows, VAL__ keys, val_row_id,
    val_y_s), supported.json, fit_records.json, COMPLETE.json; optional ALIAS.json."""

    def __init__(self, uid, path, meta):
        self.uid, self.path, self.meta = uid, Path(path), meta
        self.alias_of = None
        al = self.path / "ALIAS.json"
        if al.exists():
            self.alias_of = json.loads(al.read_text()).get("alias_of")
        self.preds = load_npz(self.path / "preds.npz") if (self.path / "preds.npz").exists() else None
        vp = self.path / "val_preds.npz"
        if self.preds is not None and vp.exists():
            for k, v in load_npz(vp).items():
                if k in self.preds:
                    raise ValueError(f"{uid}: key {k} in both preds.npz and val_preds.npz")
                self.preds[k] = v
        sp = self.path / "supported.json"
        self.runner_supported = json.loads(sp.read_text()) if sp.exists() else None
        fr = self.path / "fit_records.json"
        self.fit_records = json.loads(fr.read_text()) if fr.exists() else {}
        ac = self.fit_records.get("alias_check") or {}
        self.alias_claim = bool(self.alias_of) or bool(ac.get("alias", False))
        if ac.get("alias") and not self.alias_of:
            self.alias_of = ac.get("alias_of")
        mp = ((self.fit_records.get("release") or {}).get("map") or {})
        self.map_id = mp.get("map_id")
        self.shared = self.fit_records.get("shared") or {}
        self.pkeys, self.vkeys = {}, {}
        for k in (self.preds or {}):
            pk = parse_pkey(k)
            if pk:
                (self.pkeys if pk[0] == "P" else self.vkeys)[pk[1:]] = k


def _sha_pairs(obj):
    hexre = re.compile(r"^[0-9a-f]{64}$")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and hexre.match(v) and k != "sha256":
                yield k, v
            elif isinstance(v, dict) and isinstance(v.get("sha256"), str) and hexre.match(v["sha256"]):
                yield v.get("path", k), v["sha256"]
            else:
                yield from _sha_pairs(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and isinstance(v.get("sha256"), str) and ("path" in v or "file" in v):
                yield v.get("path", v.get("file")), v["sha256"]
            else:
                yield from _sha_pairs(v)


def verify_complete(u, rep):
    f = u.path / "COMPLETE.json"
    if not f.exists():
        rep.add(f"{u.uid}: COMPLETE.json present", "ids_roles", False, True, kind="exact")
        return
    pairs = list(_sha_pairs(json.loads(f.read_text())))
    bad, missing = [], []
    for rel, sha in pairs:
        p = Path(rel) if Path(rel).is_absolute() else u.path / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != sha:
            bad.append(rel)
    rep.add(f"{u.uid}: COMPLETE.json sha256s match files", "ids_roles", len(bad) + len(missing), 0, kind="exact",
            note=f"{len(pairs)} listed; mismatched={bad} missing={missing}")


# ---------------------------------------------------------------- per-unit statistics
def register_unit(stats, uid, preds, pkeys, y, yt, sup, prior, t_prior=None, extras=None):
    """Weighted statistics of one unit on the assessment rows, names '<uid>|<tag>|<metric>'.
    Also returns the per-class / per-pair component names for derived bounds."""
    y = np.asarray(y).astype(int)
    supc, supp = sup["supported_classes"], sup["supported_pairs"]
    comps = {}

    def reg(name, fn):
        stats[f"{uid}|{name}"] = fn

    def auc_block(tag, P):
        if not sup["estimable"]:
            return
        P = np.asarray(P, np.float64)
        cnames = []
        for c in supc:
            if c >= P.shape[1]:
                continue
            key = f"{uid}|{tag}|AUC_class_{c}"
            lz = Lazy(lambda c=c, P=P: WAUC(P[:, c], y == c))
            stats[key] = (lambda key=key, lz=lz: lambda ctx: ctx.get(key, lz))()
            cnames.append(key)
        pnames = []
        for j, k in supp:
            if k >= P.shape[1]:
                continue
            key = f"{uid}|{tag}|AUC_pair_{j}_{k}"

            def mk(j=j, k=k, P=P):
                pj, pk = P[:, j], P[:, k]
                den = pj + pk
                with np.errstate(invalid="ignore", divide="ignore"):
                    sc = np.where(den > 0, pk / np.where(den > 0, den, 1.0), 0.5)
                return WAUC(sc, y == k, (y == j) | (y == k))
            lz = Lazy(mk)
            stats[key] = (lambda key=key, lz=lz: lambda ctx: ctx.get(key, lz))()
            pnames.append(key)
        comps[f"{uid}|{tag}|classes"] = cnames
        comps[f"{uid}|{tag}|pairs"] = pnames
        cf = [stats[n] for n in cnames]
        pf = [stats[n] for n in pnames]
        reg(f"{tag}|AUC_macro", (lambda cf=cf: lambda ctx: np.mean([f(ctx) for f in cf], axis=0))())
        reg(f"{tag}|AUC_worst_class", (lambda cf=cf: lambda ctx: np.max([f(ctx) for f in cf], axis=0))())
        if pf:
            reg(f"{tag}|AUC_worst_pair", (lambda pf=pf: lambda ctx: np.max([f(ctx) for f in pf], axis=0))())

    def wmean(x):
        x = np.asarray(x, np.float64)
        return lambda ctx: (ctx.W @ x) / ctx.W.sum(axis=1)

    def wskill(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return lambda ctx: 1.0 - (ctx.W @ a) / (ctx.W @ b)

    def prob_scores(tag, P, pri):
        P = np.asarray(P, np.float64)
        K = P.shape[1]
        ok = (y >= 0) & (y < K)
        ptrue = np.where(ok, P[np.arange(len(y)), np.clip(y, 0, K - 1)], 0.0)
        prtrue = np.where(ok, pri[np.clip(y, 0, K - 1)], 0.0)
        lm, lp = ll_rows(ptrue), ll_rows(prtrue)
        reg(f"{tag}|logloss", wmean(lm))
        reg(f"{tag}|LL_skill", wskill(lm, lp))
        reg(f"{tag}|LLR_nats", wmean(lp - lm))
        Y = onehot(y, range(K))
        reg(f"{tag}|brier_skill", wskill(((P - Y) ** 2).sum(1), ((pri[None, :] - Y) ** 2).sum(1)))

    for (surf, recipe, a), key in pkeys.items():
        tag = f"P|{surf}|{recipe}" + (f"|as{a}" if a is not None else "|fixed")
        P = preds[key]
        auc_block(tag, P)
        prob_scores(tag, P, prior[:P.shape[1]] if len(prior) >= P.shape[1] else np.r_[prior, np.zeros(P.shape[1] - len(prior))])
    for g in ("G1", "G2"):
        if f"{g}_pred" in preds and f"{g}_prior" in preds:
            pred = np.asarray(preds[f"{g}_pred"], np.float64)
            pri = np.asarray(preds[f"{g}_prior"], np.float64).ravel()
            Y = onehot(y, range(pred.shape[1]))
            res = ((Y - pred) ** 2).sum(1)
            tot = ((Y - pri[None, :]) ** 2).sum(1)
            reg(f"{g}|R2", (lambda res=res, tot=tot: lambda ctx: 1.0 - (ctx.W @ res) / (ctx.W @ tot))())
            if g == "G1":
                auc_block("G1pred", pred)          # metric contrast: the ridge scores read as an attacker
    if "RHO_u" in preds and "RHO_v" in preds:
        ru = np.asarray(preds["RHO_u"], np.float64).ravel()
        rv = np.asarray(preds["RHO_v"], np.float64).ravel()
        ru, rv = ru - ru.mean(), rv - rv.mean()

        def rho2(ctx, ru=ru, rv=rv):
            W = ctx.W
            sw = W.sum(1)
            mu, mv = (W @ ru) / sw, (W @ rv) / sw
            cuv = (W @ (ru * rv)) / sw - mu * mv
            return cuv * cuv / (((W @ (ru * ru)) / sw - mu * mu) * ((W @ (rv * rv)) / sw - mv * mv))
        reg("RHO1SQ_heldout", rho2)
    if "LO_P" in preds:
        P = np.asarray(preds["LO_P"], np.float64)
        auc_block("LO", P)
        prob_scores("LO", P, prior[:P.shape[1]])
    # constant reference: prior for every person (AUC undefined -> 0.5 by ties; skills = 0 by construction)
    ex = extras or {}
    # constant reference (attacker_fit prior for everyone): ties give AUC 0.5, skills 0 by construction
    for m_, v_ in (("AUC_macro", 0.5), ("AUC_worst_class", 0.5), ("AUC_worst_pair", 0.5), ("LLR_nats", 0.0),
                   ("LL_skill", 0.0), ("brier_skill", 0.0)):
        reg(f"CONST|{m_}", (lambda v_=v_: lambda ctx: np.full(ctx.W.shape[0], v_))())
    if yt is not None:
        yt = np.asarray(yt).astype(int)
        tsup = ex.get("task_supported")
        if ex.get("majority_fit") is not None:
            reg("Uconst|accuracy", wmean((yt == ex["majority_fit"]).astype(np.float64)))
        for name, arr, is_logit in (("U1", preds.get("U1_logits"), True), ("U2", preds.get("U2_P"), False),
                                    ("CLEAN", ex.get("clean_logits"), True)):
            if arr is None:
                continue
            arr = np.asarray(arr, np.float64)
            prob = np.exp(log_softmax(arr)) if is_logit else arr
            K = prob.shape[1]
            ok = (yt >= 0) & (yt < K)
            pred_lab = prob.argmax(1)
            acc = (pred_lab == yt).astype(np.float64)
            reg(f"{name}|accuracy", wmean(acc))
            ptrue = np.where(ok, prob[np.arange(len(yt)), np.clip(yt, 0, K - 1)], 0.0)
            reg(f"{name}|logloss", wmean(ll_rows(ptrue)))
            if is_logit:
                lsm = log_softmax(arr)
                reg(f"{name}|logloss_unclipped", wmean(-lsm[np.arange(len(yt)), np.clip(yt, 0, K - 1)]))
            cls = [c for c in (tsup if tsup is not None else range(K)) if c < K]
            if cls:
                fs_ = []
                for c in cls:
                    key = f"{uid}|{name}|tauc{c}"
                    lz = Lazy(lambda c=c, prob=prob: WAUC(prob[:, c], yt == c))
                    fs_.append((lambda key=key, lz=lz: lambda ctx: ctx.get(key, lz))())
                reg(f"{name}|macro_auc", (lambda fs_=fs_: lambda ctx: np.mean([f(ctx) for f in fs_], axis=0))())
                parts = [((pred_lab == c) & (yt == c), (pred_lab == c) & (yt != c), (pred_lab != c) & (yt == c))
                         for c in cls]
                parts = [tuple(x.astype(np.float64) for x in p_) for p_ in parts]

                def macro_f1(ctx, parts=parts):
                    out = []
                    for tp, fp, fn_ in parts:
                        a_, b_, c_ = ctx.W @ tp, ctx.W @ fp, ctx.W @ fn_
                        d_ = 2 * a_ + b_ + c_
                        out.append(np.where(d_ > 0, 2 * a_ / np.where(d_ > 0, d_, 1.0), 0.0))
                    return np.mean(out, axis=0)
                reg(f"{name}|macro_f1", macro_f1)
    # held-out cross-covariance of the released representation with the target / policy one-hots (assessment rows)
    for cname, (X, Z) in (ex.get("xcov") or {}).items():
        X = np.asarray(X, np.float64)
        Z = np.asarray(Z, np.float64)
        d_, k_ = X.shape[1], Z.shape[1]
        XZ = (X[:, :, None] * Z[:, None, :]).reshape(len(X), d_ * k_)
        key = f"{uid}|XCOVM|{cname}"

        def covm(W, X=X, Z=Z, XZ=XZ, d_=d_, k_=k_):
            sw = W.sum(1)[:, None]
            mx, mz = (W @ X) / sw, (W @ Z) / sw
            C = (W @ XZ) / sw - (mx[:, :, None] * mz[:, None, :]).reshape(len(W), d_ * k_)
            return C, sw[:, 0]
        for nm_, fn_ in (("fro", lambda C, sw: np.sqrt((C ** 2).sum(1))),
                         ("max_abs", lambda C, sw: np.abs(C).max(1)),
                         ("fro_ddof1", lambda C, sw: np.sqrt((C ** 2).sum(1)) * sw / (sw - 1)),
                         ("max_abs_ddof1", lambda C, sw: np.abs(C).max(1) * sw / (sw - 1))):
            reg(f"XCOV|{cname}|{nm_}", (lambda key=key, covm=covm, fn_=fn_:
                                         lambda ctx: fn_(*ctx.get(key, covm)))())
    return comps


# ---------------------------------------------------------------- LEACE map checks
def find_maps(root):
    """{map_id: (dir, npz dict, merged meta, npz path, parsed id)} from defenses/<map_id>/map/leace_map.npz,
    map/leace_map.json and MAP_RECORD.json (the coordinator's layout)."""
    out = {}
    base = root / "defenses"
    if not base.exists():
        return out
    for f in sorted(base.rglob("*.npz")):
        try:
            d = load_npz(f)
        except Exception:  # noqa: BLE001
            continue
        if "proj_left" not in d or "proj_right" not in d:
            continue
        mdir = f.parent.parent if f.parent.name == "map" else f.parent
        pm = parse_map_id(mdir.name)
        if pm is None:
            continue
        meta = {}
        for j in [mdir / "MAP_RECORD.json", *sorted(f.parent.glob("*.json"))]:
            if j.exists():
                try:
                    meta.update(json.loads(j.read_text()))
                except Exception:  # noqa: BLE001
                    pass
        out[mdir.name] = (mdir, d, meta, f, pm)
    return out


def design_map_id(ds, seed, purpose, attr, arm):
    if arm == "B":
        return f"{ds.name}__s{seed}__{purpose}__B_{attr}"
    pol = ds.spec["purposes"][purpose]["disallowed_attrs"]
    return f"{ds.name}__s{seed}__{purpose}__C_{'+'.join(pol)}"


def map_bias(d):
    for k in ("mean_x", "mean", "bias"):
        if k in d:
            return np.asarray(d[k], np.float64).ravel()
    return None


def concept_matrix(ds, meta, pm, rep=None, mid=""):
    """Design concept from the map id: B = one-hot of its attribute; C = concatenated marginal one-hots of the
    purpose's disallowed_attrs (checked against the id), each over the FULL declared class range
    (disallowed_attr_dims) -- the coordinator's rule when every declared class has >= 100 defense_fit rows.
    The runner's concept_spec is compared, never used."""
    pur = ds.spec["purposes"][pm["purpose"]]
    attrs = list(pm["attrs"])
    if rep is not None and pm["kind"] == "C":
        rep.add(f"{mid}: C concept attributes == disallowed_attrs of the purpose", "leace", attrs,
                list(pur["disallowed_attrs"]), kind="exact")
    dims = dict(pur.get("disallowed_attr_dims", {}))
    fit = ds.role == "defense_fit"
    blocks, classes = [], {}
    for a in attrs:
        y = ds.label(a)
        K = int(dims.get(a, int(y.max()) + 1))
        cnt = [int(np.sum(fit & (y == c))) for c in range(K)]
        classes[a] = list(range(K))
        if rep is not None:
            rep.add(f"{mid}: every declared {a} class has >= {DEFENSE_MIN} defense_fit rows (full one-hot rule)",
                    "leace", min(cnt), DEFENSE_MIN, status="PASS" if min(cnt) >= DEFENSE_MIN else "FAIL", kind="info",
                    note=f"defense_fit counts per class: {cnt}")
        blocks.append(onehot(y, classes[a]))
    spec = meta.get("concept_spec")
    if rep is not None and spec is not None:
        bl = spec.get("blocks") if isinstance(spec, dict) else spec
        if isinstance(bl, list):
            got = [[b.get("name"), list(range(int(b.get("n_classes", 0))))] for b in bl]
        else:
            ra = spec.get("attributes") or []
            got = [[a, [int(c) for c in (spec.get("classes") or {}).get(a, [])]] for a in ra]
        rep.add(f"{mid}: saved concept_spec == design concept (attributes, full declared classes, column order)",
                "leace", got, [[a, classes[a]] for a in attrs], kind="exact")
    return np.concatenate(blocks, axis=1), attrs


def whitening(sigma):
    """W = sigma^{-1/2} and W^{-1} with the official pinv-style eigenvalue mask (concept-erasure 0.2.4)."""
    S = (np.asarray(sigma, np.float64) + np.asarray(sigma, np.float64).T) / 2
    L, V = np.linalg.eigh(S)
    mask = L > (L[-1] * S.shape[-1] * np.finfo(np.float64).eps)
    L = np.clip(L, 0.0, None)
    W = (V * np.where(mask, 1.0 / np.sqrt(np.where(mask, L, 1.0)), 0.0)) @ V.T
    Winv = (V * np.where(mask, np.sqrt(L), 0.0)) @ V.T
    return W, Winv, int(mask.sum())


def _get(d, *names):
    for n in names:
        if n in d:
            return np.asarray(d[n], np.float64)
    return None


def leace_checks(ds, maps, rep, results, units=None):
    """Key LEACE properties recomputed from the saved maps (numpy only).

    Primary status (coordinator decision 2026-10-02): the official implementation bound -- the spectral norm of the
    whitened residual cross-covariance ||W P Sigma_xz||_2 <= svd_tol (W = Sigma_xx_used^{-1/2}), computed from the
    saved arrays and again from independently recomputed defense_fit rows.  The exact-zero cross-covariance of the
    erased defense_fit features with the concept one-hot and the fit-row OLS R^2 are descriptive (INFO)."""
    out = {}
    fit = ds.role == "defense_fit"
    for mid, (mdir, d, meta, f, pm) in sorted(maps.items()):
        if pm["ds"] != ds.name:
            continue
        seed = pm["seed"]
        H = ds.forward(seed, pm["purpose"])[0]
        Lm, Rm = np.asarray(d["proj_left"], np.float64), np.asarray(d["proj_right"], np.float64)
        b = map_bias(d)
        Z, attrs = concept_matrix(ds, meta, pm, rep, mid)
        Hf, Zf = H[fit], Z[fit]
        n, dim = Hf.shape
        scale = max(1.0, float(np.max(np.abs(Hf))))
        P = np.eye(dim) - Lm @ Rm
        st = meta.get("settings_used") or meta.get("settings") or \
            (meta.get("map_metadata") or {}).get("settings_used") or {}
        svd_tol = float(st.get("svd_tol", LEACE_SVD_TOL))
        rec = {"concept_attributes": attrs, "n_fit_rows": int(n), "rank_saved": meta.get("rank"),
               "settings_saved": st}
        rep.add(f"{mid}: official LeaceFitter defaults (svd_tol=0.01, shrinkage, leace, affine, trace constraint)",
                "leace", [svd_tol, st.get("shrinkage", True), st.get("method", "leace"), st.get("affine", True),
                          st.get("constrain_cov_trace", True)], [LEACE_SVD_TOL, True, "leace", True, True], kind="exact")
        # (1) the saved mean is the original fitting mean
        dm = float(np.max(np.abs(b - Hf.mean(0)))) if b is not None else None
        rep.add(f"{mid}: saved mean_x == mean of recomputed defense_fit h (original fitting mean)", "leace", dm, 0.0,
                tol=LEACE_MEAN_TOL * scale, status=None if dm is not None else "FAIL", note="runner_value = max abs diff")
        mz = _get(d, "mean_z")
        if mz is not None and mz.shape == Zf.mean(0).shape:
            rep.add(f"{mid}: saved mean_z == concept one-hot means on defense_fit", "leace",
                    float(np.max(np.abs(mz - Zf.mean(0)))), 0.0, tol=1e-12)
        # (2) saved statistics vs recomputation from the recomputed defense_fit rows
        S_hat = Hf.T @ (Hf - Hf.mean(0))
        S_hat = (S_hat + S_hat.T) / 2
        sig_rec = _shrink(S_hat / n, n) if st.get("shrinkage", True) else S_hat / (n - 1)
        sxz_rec = Hf.T @ (Zf - Zf.mean(0)) / (n - 1)
        sig_saved = _get(d, "sigma_xx_used", "sigma_xx")
        sxz_saved = _get(d, "sigma_xz")
        for nm, sv, rv in (("sigma_xx_used", sig_saved, sig_rec), ("sigma_xz", sxz_saved, sxz_rec)):
            if sv is None:
                rep.add(f"{mid}: saved {nm} present", "leace", False, True, kind="exact")
                continue
            dd = float(np.max(np.abs(sv - rv)))
            rep.add(f"{mid}: saved {nm} == recomputation on defense_fit rows", "leace", dd, 0.0,
                    tol=1e-9 * max(1.0, float(np.max(np.abs(rv)))), note="runner_value = max abs diff")
        sig = sig_saved if sig_saved is not None else sig_rec
        sxz = sxz_saved if sxz_saved is not None else sxz_rec
        W, Winv, wrank = whitening(sig)
        s_mine = np.linalg.svd(W @ sxz, compute_uv=False)
        s_saved = _get(d, "singular_values_whitened_xz", "singular_values", "s", "svals", "sv")
        if s_saved is not None:
            k = min(len(s_saved), len(s_mine))
            rep.add(f"{mid}: saved singular values == svd(W Sigma_xz) from saved arrays", "leace",
                    float(np.max(np.abs(np.sort(s_saved.ravel())[::-1][:k] - s_mine[:k]))), 0.0,
                    tol=1e-9 * max(1.0, float(s_mine.max())))
        n_trunc = int(np.sum(s_mine <= svd_tol))
        rec.update({"whitened_singular_values": s_mine.tolist(), "n_truncated_directions": n_trunc,
                    "whitening_rank": wrank, "fit_rank_sample_cov": int(np.linalg.matrix_rank(S_hat))})
        # (3) PRIMARY native bound: ||W P Sigma_xz||_2 <= svd_tol (saved arrays) and on recomputed erased rows
        Xe = leace_apply(Hf, Lm, Rm, b)
        r_saved = float(np.linalg.norm(W @ P @ sxz, 2))
        W_rec, _, _ = whitening(sig_rec)
        r_rec = float(np.linalg.norm(W_rec @ cross_cov(Xe, Zf), 2))
        rec.update({"whitened_residual_spectral_saved": r_saved, "whitened_residual_spectral_recomputed": r_rec})
        bt = svd_tol * (1 + 1e-9) + 1e-12
        rep.add(f"{mid}: NATIVE ||W P Sigma_xz||_2 <= svd_tol (saved arrays)", "leace", r_saved, svd_tol, tol=None,
                status="PASS" if r_saved <= bt else "FAIL", kind="info",
                note=f"largest truncated whitened singular value; {n_trunc} direction(s) <= svd_tol")
        rep.add(f"{mid}: NATIVE ||W Cov(erased defense_fit h, concept)||_2 <= svd_tol (recomputed rows)", "leace",
                r_rec, svd_tol, status="PASS" if r_rec <= bt else "FAIL", kind="info")
        # (4) re-derivation of the official algorithm (data action on the fit rows; P itself is descriptive because
        #     near-null whitening directions make P ill-conditioned)
        nd = leace_numpy(Hf, Zf, svd_tol=svd_tol, shrinkage=st.get("shrinkage", True),
                         constrain_cov_trace=st.get("constrain_cov_trace", True))
        dP = float(np.max(np.abs(P - nd["P"])))
        Xe2 = Hf - (Hf - nd["bias"]) @ (np.eye(dim) - nd["P"]).T
        dX = float(np.max(np.abs(Xe - Xe2)))
        rec.update({"P_maxabs_diff_vs_rederivation": dP, "erased_fit_maxabs_diff_vs_rederivation": dX,
                    "trace_constraint_mixed": nd["trace_mixed"], "alpha_mix": nd["alpha_mix"]})
        rep.add(f"{mid}: erased defense_fit rows == numpy re-derivation of LeaceFitter (recomputed rows)", "leace",
                dX, 0.0, tol=LEACE_REDERIVE_TOL * scale,
                note=f"max |P_saved - P_rederived| = {dP:.3g} (descriptive); trace-constraint mix = {nd['trace_mixed']}")
        # (5) descriptive: exact-zero cross-covariance and fit-row OLS R^2
        c0 = float(np.max(np.abs(cross_cov(Hf, Zf))))
        c1 = float(np.max(np.abs(cross_cov(Xe, Zf))))

        def _maxcorr(X_, Z_):
            C_ = cross_cov(X_, Z_)
            sx, sz = Hf.std(0, ddof=1), Z_.std(0, ddof=1)       # untreated h scale (matches runner records)
            den = np.outer(sx, sz)
            ok_ = den > 1e-12 * max(1.0, float(den.max()))
            return float(np.max(np.abs(np.where(ok_, C_ / np.where(ok_, den, 1.0), 0.0))))
        ev = np.linalg.eigvalsh(S_hat / (n - 1))
        thr_ = ev[-1] * dim * np.finfo(np.float64).eps
        Xc_, Zc_ = Xe - Xe.mean(0), Zf - Zf.mean(0)
        ols_var = {}
        for rc_ in (None, 1e-12, 1e-10, 1e-8, 1e-6):
            b_, *_ = np.linalg.lstsq(Xc_, Zc_, rcond=rc_)
            ols_var[str(rc_)] = 1.0 - float(np.sum((Zc_ - Xc_ @ b_) ** 2)) / max(float(np.sum(Zc_ ** 2)), 1e-300)
        rec["fit_ols_r2_rcond_variants"] = ols_var
        dgr = (meta.get("diagnostics") or {}).get("sample_cov_rank")
        if dgr is not None:
            rep.add(f"{mid}: defense_fit sample-covariance rank (pinv rule) == map diagnostics", "leace", int(dgr),
                    int(np.sum(np.linalg.eigvalsh(S_hat / (n - 1)) >
                               np.linalg.eigvalsh(S_hat / (n - 1))[-1] * dim * np.finfo(np.float64).eps)),
                    kind="exact", quiet=True)
        rec.update({"xcov_after_maxcorr": _maxcorr(Xe, Zf), "xcov_before_maxcorr": _maxcorr(Hf, Zf),
                    "sample_cov_rank_pinv": int(np.sum(ev > thr_)),
                    "sample_cov_rank_band": [int(np.sum(ev > thr_ * 1e3)), int(np.sum(ev > thr_ / 1e3))],
                    "structural_zero_sv_threshold": float(((meta.get("diagnostics") or {}).get(
                        "structural_zero_sv_threshold")) or 1e-10)})
        r2 = affine_ols_r2(Xe, Zf)
        rec.update({"xcov_before_maxabs": c0, "xcov_after_maxabs": c1, "xcov_after_relative": c1 / max(c0, 1e-300),
                    "fit_ols_r2_after": r2, "fit_ols_r2_before": affine_ols_r2(Hf, Zf)})
        rep.add(f"{mid}: descriptive max |Cov(erased defense_fit h, concept)|", "leace", None, c1, status="INFO",
                kind="info", note=f"relative to pre-erasure {c1 / max(c0, 1e-300):.3g}; nonzero when sub-tolerance "
                                  f"directions are truncated ({n_trunc}); not a failure (coordinator decision)")
        rep.add(f"{mid}: descriptive fit-row affine OLS R^2 of the concept on erased h", "leace", None, r2,
                status="INFO", kind="info")
        nc = meta.get("native_check") or {}
        for k_r, mine in (("whitened_residual_spectral", r_saved), ("xcov_maxabs", c1), ("ols_r2", r2)):
            if k_r in nc:
                rep.add(f"{mid}: runner native_check {k_r}", "leace", nc[k_r], mine, tol=max(1e-9, 1e-6 * abs(mine)))
        # (6) held-out cross-covariance (assessment rows), exploratory values
        am = ds.role == "assessment"
        xh = cross_cov(leace_apply(H[am], Lm, Rm, b), Z[am])
        rec["heldout_xcov_fro"] = float(np.linalg.norm(xh))
        rec["heldout_xcov_maxabs"] = float(np.max(np.abs(xh)))
        rec["heldout_whitened_spectral"] = float(np.linalg.norm(W @ xh, 2))
        # (7) saved transformed rows of other roles use the original fitting mean
        for tf in sorted(mdir.glob("*.npz")):
            if tf == f:
                continue
            t = load_npz(tf)
            if "row_id" not in t:
                continue
            arr = next((t[k] for k in t if k != "row_id" and t[k].ndim == 2 and t[k].shape[1] == dim), None)
            if arr is None:
                continue
            idx = np.array([ds.pos[int(r)] for r in t["row_id"]])
            mine = leace_apply(H[idx], Lm, Rm, b)
            dd = float(np.max(np.abs(arr - mine)))
            roles_here = sorted(set(ds.role[idx].tolist()))
            note = f"rows: {len(idx)} roles {roles_here}"
            if dd > TRANSFORM_TOL * scale:
                diag = []
                for r in roles_here:
                    sel = ds.role[idx] == r
                    alt = leace_apply(H[idx][sel], Lm, Rm, H[idx][sel].mean(0))
                    if np.max(np.abs(arr[sel] - alt)) <= TRANSFORM_TOL * scale:
                        diag.append(r)
                if diag:
                    note += f"; rows reproduced with their OWN role mean instead of the fitting mean: {diag}"
            rep.add(f"{mid}: saved transformed rows ({tf.name}) == x - (x - mean_x) R^T L^T", "leace", dd, 0.0,
                    tol=TRANSFORM_TOL * scale, note=note)
        # (8) fit row-id hash
        ids = np.sort(ds.row_id[fit]).astype("<i8")
        uids = np.sort(np.unique(ds.unit[fit])).astype("<i8")
        cands = {"int64_bytes_sorted_row_ids": hashlib.sha256(ids.tobytes()).hexdigest(),
                 "comma_joined_sorted": hashlib.sha256(",".join(map(str, ids.tolist())).encode()).hexdigest(),
                 "newline_joined_sorted": hashlib.sha256("\n".join(map(str, ids.tolist())).encode()).hexdigest(),
                 "json_list_sorted": hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
                 "int64_bytes_sorted_units": hashlib.sha256(uids.tobytes()).hexdigest()}
        for fld, required in (("fit_row_ids_sha256", True), ("defense_fit_row_ids_sha256", False)):
            frh = meta.get(fld) or ((meta.get("map_metadata") or {}).get(fld) if required else None)
            if not frh:
                continue
            hit = [k for k, v in cands.items() if v == frh]
            rep.add(f"{mid}: {fld} reproduced from recomputed defense_fit ids", "leace", frh,
                    cands[hit[0]] if hit else None,
                    status="PASS" if hit else ("FAIL" if required else "INFO"), kind="exact",
                    note=f"convention={hit[0]}" if hit else ("no known hashing convention matched" +
                                                             ("" if required else " (record field; informational)")))
        out[mid] = rec
    # aliases B vs C of the Tier-1 cell (action on all rows decides; P compared descriptively)
    c = ds.cell
    for seed in sorted(ds.encoders):
        bid = design_map_id(ds, seed, c["purpose"], c["target"], "B")
        cid = design_map_id(ds, seed, c["purpose"], c["target"], "C")
        cu = [u for u in (units or {}).values() if u.meta["ds"] == ds.name and u.meta["seed"] == seed
              and u.meta["arm"] == "C"]
        unit_claim = any(u.alias_claim for u in cu)
        if bid not in maps or cid not in maps:
            if unit_claim or cu:
                rep.add(f"{ds.name}__s{seed}: B and C maps saved for the alias decision", "leace", False, True,
                        kind="exact", note=f"looked for {bid}, {cid}")
            continue
        _, db, _, _, _ = maps[bid]
        _, dc, mc, _, _ = maps[cid]
        H = ds.forward(seed)[0]
        scale = max(1.0, float(np.max(np.abs(H[fit]))))
        xb = leace_apply(H, np.asarray(db["proj_left"]), np.asarray(db["proj_right"]), map_bias(db))
        xc = leace_apply(H, np.asarray(dc["proj_left"]), np.asarray(dc["proj_right"]), map_bias(dc))
        dd = float(np.max(np.abs(xb - xc))) / scale
        dP = float(np.max(np.abs((db["proj_left"] @ db["proj_right"]) - (dc["proj_left"] @ dc["proj_right"]))))
        mine = dd <= ALIAS_TOL
        claim = mc.get("alias_of") is not None or bool(mc.get("alias", False)) or unit_claim
        st_ = None
        atol_r = float(((mc.get("tolerances") or {}).get("alias_atol")) or 1e-10)
        if claim != mine and claim == (dP <= atol_r):
            st_ = "PASS"      # the runner's stated rule (max |P_B - P_C| <= alias_atol) decides; replay agrees under it
        rep.add(f"{ds.name}__s{seed}: B==C alias claim", "leace", claim, mine, kind="exact", status=st_,
                note=f"max |x_B - x_C| / scale over all rows = {dd:.3g} (replay alias tol {ALIAS_TOL}, action); "
                     f"max |P_B - P_C| = {dP:.3g}")
        out[f"{ds.name}__s{seed}__alias"] = {"maxabs_action_rel": dd, "maxabs_P": dP, "alias_replay": mine,
                                             "alias_claimed": claim}
    results.setdefault("leace", {}).update(out)
    return out


# ---------------------------------------------------------------- main replay
def replay(root, report_dir=None, out_path=None, results_path=None, cfg=None, verbose=True,
           historical=None):
    t0 = time.time()
    cfg = cfg or Config()
    root = Path(root).expanduser()
    rep = Report()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    results = {"config": _jsonable(asdict(cfg)), "frozen_readings": FROZEN_READINGS}
    rep.add("replay configuration == frozen design (B, seeds, grids)", "inference",
            [B_PRIM, SEED_PRIM, B_EXPL, SEED_EXPL, list(SIGMA_GRID)],
            [cfg.b_prim, cfg.seed_prim, cfg.b_expl, cfg.seed_expl, list(cfg.sigma_grid)],
            status="PASS" if (cfg.b_prim, cfg.seed_prim, cfg.b_expl, cfg.seed_expl, tuple(cfg.sigma_grid)) ==
            (B_PRIM, SEED_PRIM, B_EXPL, SEED_EXPL, SIGMA_GRID) else "INFO", kind="exact",
            note="INFO = reduced validation configuration")
    index = json.loads((root / "inputs" / "INPUTS_INDEX.json").read_text())
    dsets = {}
    for i, (name, spec) in enumerate(sorted(index["datasets"].items())):
        dsets[name] = Dataset(name, i, spec, rep, cfg)
        exp_roles = check_roles(dsets[name], rep)
        dsets[name].role_saved = dsets[name].role
        dsets[name].role = exp_roles      # every downstream quantity uses the independently recomputed roles
        for s, e in dsets[name].encoders.items():
            st = e["meta"].get("lineage_status")
            rep.add(f"{name}: encoder s{s} lineage status", "inputs", st, st,
                    status="PASS" if str(st).upper() in ("VERIFIED", "PASS", "ADMITTED") else "INFO", kind="info")

    # ---- units
    udir = root / "units"
    present = {p.name: p for p in sorted(udir.iterdir()) if p.is_dir()} if udir.exists() else {}
    expected = []
    for name, ds in dsets.items():
        c = ds.cell
        for s in sorted(ds.encoders):
            base = f"{name}__s{s}__{c['purpose']}__{c['target']}__"
            expected += [base + "A", base + "B", base + "C"]
            expected += [base + f"D_sigma{fmt_sigma(sg)}_rs{k}" for sg in cfg.sigma_grid for k in cfg.release_seeds]
    missing = [u for u in expected if u not in present]
    extra = [u for u in present if u not in expected]
    rep.add("Tier-1 unit IDs present (A, B, C, D sigma x release seed per dataset x encoder seed)", "ids_roles",
            len(missing), 0, kind="exact", note=f"missing={missing}; expected {len(expected)}")
    rep.add("other unit directories present (Tier 2 / extensions; not part of the primary family)", "ids_roles",
            None, len(extra), status="INFO", kind="info", note=f"first: {sorted(extra)[:10]}")
    units_all = {}
    for uid in sorted(present):
        pm = parse_unit_id(uid)
        if pm is None or pm["ds"] not in dsets:
            rep.add(f"{uid}: unit directory name parses", "ids_roles", uid, None, status="FAIL", kind="info")
            continue
        units_all[uid] = Unit(uid, present[uid], pm)
    units = {uid: units_all[uid] for uid in expected if uid in units_all}     # Tier 1 (primary family)
    results["inventory"] = tier2_inventory(root, report_dir, units_all, expected, rep)
    # alias resolution: an aliased C has no preds of its own and points to B
    for uid, u in units_all.items():
        if u.preds is None and u.alias_claim and u.meta["arm"] == "C" and not u.alias_of:
            u.alias_of = u.uid[:-1] + "B"
        if u.preds is None and u.alias_of:
            tgt = units_all.get(u.alias_of)
            ok = tgt is not None and tgt.meta["arm"] == "B" and u.meta["arm"] == "C" and \
                tgt.meta["seed"] == u.meta["seed"] and tgt.meta["ds"] == u.meta["ds"]
            rep.add(f"{uid}: alias target is the B unit of the same dataset/seed", "aliases", u.alias_of,
                    u.alias_of if ok else None, kind="exact")
            if ok:
                u.preds, u.pkeys, u.vkeys = tgt.preds, tgt.pkeys, tgt.vkeys
                u.runner_supported = u.runner_supported or tgt.runner_supported
        elif u.preds is None:
            rep.add(f"{uid}: preds.npz present", "ids_roles", False, True, kind="exact")

    stats, comps, sup_by_unit, inv_by_ds, prior_by_unit, val_info = {}, {}, {}, {}, {}, {}
    assess_ref = {}
    maps = find_maps(root)
    task_sup_cache = {}
    results["task_support"] = task_sup_cache
    for uid, u in units_all.items():
        if u.preds is None:
            continue
        m, ds = u.meta, dsets[u.meta["ds"]]
        pr = u.preds
        if not (u.alias_of and not (u.path / "preds.npz").exists()):
            verify_complete(u, rep)
        arid = np.asarray(pr["assess_row_id"]).astype(np.int64)
        exp_assess = np.sort(ds.row_id[ds.role == "assessment"])
        rep.add(f"{uid}: assess_row_id set == assessment role", "ids_roles",
                int(len(np.setxor1d(arid, exp_assess))) + int(len(arid) - len(np.unique(arid))), 0, kind="exact")
        if m["ds"] not in assess_ref:
            assess_ref[m["ds"]] = arid
        else:
            rep.add(f"{uid}: assessment row order identical across methods/seeds", "ids_roles",
                    bool(np.array_equal(arid, assess_ref[m["ds"]])), True, kind="exact")
        idx = np.array([ds.pos[int(r)] for r in arid])
        y = ds.label(m["attr"])
        yt = ds.task_label(m["purpose"])
        if "assess_unit" in pr:
            rep.add(f"{uid}: assess_unit == record unit", "ids_roles",
                    int(np.sum(np.asarray(pr["assess_unit"]).astype(np.int64) != ds.unit[idx])), 0, kind="exact")
        if "y_s" in pr:
            rep.add(f"{uid}: y_s == labels[{m['attr']}]", "ids_roles",
                    int(np.sum(np.asarray(pr["y_s"]).astype(int) != y[idx])), 0, kind="exact")
        if "y_task" in pr:
            rep.add(f"{uid}: y_task == task labels of {m['purpose']}", "ids_roles",
                    int(np.sum(np.asarray(pr["y_task"]).astype(int) != yt[idx])), 0, kind="exact")
        sup = support_sets(y, ds.role, need_defense=False)     # per role 100/30/100 for every arm
        sup_by_unit[uid] = sup
        if u.runner_supported is not None:
            rs = parse_runner_supported(u.runner_supported)
            alt = support_sets(y, ds.role, need_defense=True)
            note = None
            if rs.get("supported_classes") != sup["supported_classes"] and \
                    rs.get("supported_classes") == alt["supported_classes"]:
                note = "runner additionally requires >=100 defense_fit rows per scored class"
            rep.add(f"{uid}: supported classes", "support", rs.get("supported_classes"), sup["supported_classes"],
                    kind="exact", note=note)
            rep.add(f"{uid}: supported pairs", "support", rs.get("supported_pairs"),
                    sorted(sorted(p) for p in sup["supported_pairs"]), kind="exact")
        else:
            rep.add(f"{uid}: supported.json present", "support", False, True, kind="exact")
        fit_y = y[ds.role == "attacker_fit"]
        K = int(y.max()) + 1
        prior = np.array([np.mean(fit_y == c) for c in range(K)])
        prior_by_unit[uid] = prior
        for g in ("G1", "G2"):
            if f"{g}_prior" in pr:
                gp = np.ravel(pr[f"{g}_prior"]).astype(np.float64)
                rep.add(f"{uid}: {g}_prior == attacker_fit one-hot means", "estimates",
                        float(np.max(np.abs(gp - prior[:len(gp)]))), 0.0, tol=1e-12)
        if "s_prior_fit" in pr:
            sp = np.ravel(pr["s_prior_fit"]).astype(np.float64)
            rep.add(f"{uid}: s_prior_fit == attacker_fit frequencies (constant reference)", "estimates",
                    float(np.max(np.abs(sp - prior[:len(sp)]))), 0.0, tol=1e-12)
        if "LO_P" in pr:   # Laplace alpha = 1 label-only table on attacker_fit
            fm = ds.role == "attacker_fit"
            KL = pr["LO_P"].shape[1]
            mine = np.empty((len(idx), KL))
            for t in np.unique(yt[idx]):
                sel = fm & (yt == t)
                mine[yt[idx] == t] = [(np.sum(y[sel] == c) + 1.0) / (sel.sum() + KL) for c in range(KL)]
            rep.add(f"{uid}: LO_P == Laplace(1) attacker_fit table P(s | task label)", "estimates",
                    float(np.max(np.abs(mine - pr["LO_P"]))), 0.0, tol=1e-12)
        H, Lg, _ = ds.forward(m["seed"], m["purpose"])
        if m["arm"] == "A" and "U1_logits" in pr and Lg is not None:
            rep.add(f"{uid}: U1_logits (untreated) == clean head logits", "estimates",
                    float(np.max(np.abs(pr["U1_logits"] - Lg[idx]))), 0.0, tol=1e-9)
        if "OUT_logits" in pr and Lg is not None:
            rep.add(f"{uid}: OUT_logits == clean head logits from the forward cache", "estimates",
                    float(np.max(np.abs(pr["OUT_logits"] - Lg[idx]))), 0.0, tol=1e-9)
        if (m["ds"], m["purpose"]) not in task_sup_cache:
            ts = support_sets(yt, ds.role, need_defense=False)
            vals, cnts = np.unique(yt[ds.role == "attacker_fit"], return_counts=True)
            task_sup_cache[(m["ds"], m["purpose"])] = (ts, int(vals[np.argmax(cnts)]))
        tsup, maj = task_sup_cache[(m["ds"], m["purpose"])]
        ex = {"task_supported": tsup["supported_classes"], "majority_fit": maj,
              "clean_logits": Lg[idx] if Lg is not None else None}
        if m["arm"] in ("A", "B", "C"):
            X = H[idx]
            if m["arm"] in ("B", "C"):
                _, mp = _map_for_unit(u, ds, maps)
                X = None if mp is None else leace_apply(H[idx], np.asarray(mp[1]["proj_left"]),
                                                         np.asarray(mp[1]["proj_right"]), map_bias(mp[1]))
            if X is not None:
                pur = ds.spec["purposes"][m["purpose"]]
                dims = pur.get("disallowed_attr_dims", {})
                Zt = onehot(y[idx], range(int(dims.get(m["attr"], y.max() + 1))))
                Zp = np.concatenate([onehot(ds.label(a)[idx], range(int(dims.get(a, ds.label(a).max() + 1))))
                                     for a in pur["disallowed_attrs"]], axis=1)
                ex["xcov"] = {"target": (X, Zt), "policy": (X, Zp)}
        comps.update(register_unit(stats, uid, pr, u.pkeys, y[idx], yt[idx], sup, prior, extras=ex))
        if "val_row_id" in pr:
            vrid = np.asarray(pr["val_row_id"]).astype(np.int64)
            rep.add(f"{uid}: val_row_id set == attacker_val role", "ids_roles",
                    int(len(np.setxor1d(vrid, ds.row_id[ds.role == 'attacker_val']))), 0, kind="exact")
            val_info[uid] = np.array([ds.pos[int(r)] for r in vrid])
        if m["ds"] not in inv_by_ds:
            inv_by_ds[m["ds"]] = cluster_index(ds.unit[idx])

    # ---- surface / candidate / alias identities (all completed units)
    identity_checks(units_all, dsets, rep, root, maps)
    # ---- held-out G1 re-derivation (A, B, C from forward reps / saved maps; D under the pilot noise convention)
    for name, ds in dsets.items():
        leace_checks(ds, maps, rep, results, units)
    g1_rederive(units_all, dsets, maps, rep)
    native_A(units_all, dsets, rep, results, historical)

    # ---- point estimates
    n_by_ds = {d: len(assess_ref[d]) for d in assess_ref}
    point = {}
    for k, fn in stats.items():
        point[k] = float(fn(Ctx(np.ones((1, n_by_ds[ds_of(k)]))))[0])
    log(f"[replay] {len(stats)} unit statistics; point estimates at {time.time() - t0:.1f}s")

    # ---- sigma* from attacker_val predictions only
    sigma_star = compute_sigma_star(units, dsets, sup_by_unit, val_info, cfg, rep)
    results["sigma_star"] = sigma_star

    # ---- replicate summaries
    summaries, sum_members = register_summaries(units, stats, comps, sup_by_unit, cfg)
    # ---- primary family (registers the paired U2NI differences into summaries)
    endpoints = primary_endpoints(dsets, units, sigma_star, summaries, sup_by_unit, sum_members, maps, cfg, rep)
    for k, fn in summaries.items():
        point[k] = float(fn(Ctx(np.ones((1, n_by_ds[ds_of(k)]))))[0])
    allstats = {**stats, **summaries}
    results["primary_family"] = {"size": len(endpoints), "alpha_each": ALPHA_FAMILY}
    rep.add("primary family size", "primary", len(endpoints), FAMILY_SIZE, kind="exact")
    log(f"[replay] primary bootstrap B={cfg.b_prim} ...")
    prim_reps = {}
    for name, ds in dsets.items():
        if name not in inv_by_ds:
            continue
        need = {}
        for e in endpoints:
            if e["dataset"] == name and e.get("stat"):
                need[e["stat"]] = allstats[e["stat"]]
        inv, nu = inv_by_ds[name]
        prim_reps.update(run_bootstrap(need, inv, nu, cfg.b_prim, [cfg.seed_prim, ds.index]))
    primary = []
    for e in endpoints:
        rec = dict(e)
        if e["decision"] == "NE" or not e.get("stat"):
            rec["decision"] = "NE"
            primary.append(rec)
            continue
        x = prim_reps[e["stat"]]
        lo, slo = qbound(x, ALPHA_FAMILY)
        hi, shi = qbound(x, 1 - ALPHA_FAMILY)
        rec.update({"estimate": point[e["stat"]], "lower": lo, "upper": hi, "mcse_lower": slo, "mcse_upper": shi,
                    "n_nonfinite": int(np.sum(~np.isfinite(x)))})
        if e["family"] == "G1":
            rec["decision"] = decide_bar(lo, hi, TAU)
        elif e["family"] in ("Rrep", "Rplus"):
            rec["decision"] = decide_bar(lo, hi, BAR)
        else:
            rec["decision"] = decide_ni(lo, hi)
        thr = TAU if e["family"] == "G1" else (NI_MARGIN if e["family"] == "U2NI" else BAR)
        rec["mc_borderline"] = bool(min(abs(lo - thr), abs(hi - thr)) <= max(bound_tol(slo), bound_tol(shi)))
        primary.append(rec)
    results["primary"] = primary
    log(f"[replay] primary done at {time.time() - t0:.1f}s")

    # ---- exploratory (90 %, B = 2000) on replicate summaries (+ per-unit statistics if requested)
    expl_names = [] if cfg.expl_scope == "none" else \
        ([k for k in summaries] if cfg.expl_scope == "summaries" else list(allstats))
    expl_reps = {}
    for name, ds in dsets.items():
        if name not in inv_by_ds:
            continue
        need = {k: allstats[k] for k in expl_names if ds_of(k) == name}
        inv, nu = inv_by_ds[name]
        expl_reps.update(run_bootstrap(need, inv, nu, cfg.b_expl, [cfg.seed_expl, ds.index]))
    lo_e, hi_e = (1 - LEVEL_EXPL) / 2, 1 - (1 - LEVEL_EXPL) / 2
    exploratory = {}
    for k in expl_names:
        iv = interval(expl_reps[k], lo_e, hi_e)
        exploratory[k] = {"estimate": point[k], **(iv or {})}
        if k.endswith("AUC_macro") and iv:
            exploratory[k]["vs_bars"] = {str(b): decide_bar(iv["lower"], iv["upper"], b)
                                         for b in (*SECONDARY_BARS, BAR)}
    # derived worst-class / worst-pair bounds for the summaries
    worst = {}
    for k, members in sum_members.items():
        if not k.endswith("|AUC_worst_class_derived") and not k.endswith("|AUC_worst_pair_derived"):
            continue
        cks = members
        if not cks or any(c not in expl_reps for c in cks):
            continue
        db = derived_max_bounds({c.rsplit("|", 1)[1]: expl_reps[c] for c in cks}, lo_e)
        worst[k] = {"estimate_max_of_seedmeans": max(point[c] for c in cks),
                    "estimate_seedmean_of_unit_max": point.get(k.replace("_derived", "")), **db}
    results["worst_derived"] = worst
    results["exploratory"] = exploratory
    log(f"[replay] exploratory done at {time.time() - t0:.1f}s")

    # ---- per-seed tables: attacker-seed SD and encoder-seed SD of point estimates
    results["seed_variation"] = seed_variation(point, units_all, cfg)
    results["support"] = {u: {"supported_classes": s["supported_classes"], "estimable": s["estimable"],
                              "counts": {str(c): v for c, v in s["counts"].items()},
                              "unsupported_reasons": {str(c): r for c, r in s["unsupported_reasons"].items()}}
                          for u, s in sup_by_unit.items()}
    results["n_assessment"] = {d: {"rows": int(n_by_ds[d]), "units": int(inv_by_ds[d][1])} for d in n_by_ds}
    results["point_summaries"] = {k: point[k] for k in summaries}
    results["point_units"] = {k: point[k] for k in stats}   # aggregate statistics only (no per-person values)
    results["point_units"] = {k: v for k, v in results["point_units"].items()
                              if not k.split("|", 1)[1].startswith(("XCOV", "CONST"))}
    if report_dir and (Path(report_dir) / "RECOVERY.csv").exists():
        table_replay(report_dir, root, units_all, dsets, stats, point, sup_by_unit, inv_by_ds, n_by_ds, results, rep,
                     cfg, log, primary, sigma_star)
    elif report_dir:
        compare_runner_reports(Path(report_dir), results, rep)
    return finish(rep, results, out_path, results_path, t0)


def _map_for_unit(u, ds, maps):
    m = u.meta
    mid = u.map_id or design_map_id(ds, m["seed"], m["purpose"], m["attr"], "B" if (u.alias_of and m["arm"] == "C")
                                    else m["arm"])
    if u.alias_of and m["arm"] == "C":
        mid = design_map_id(ds, m["seed"], m["purpose"], m["attr"], "B")
    return mid, maps.get(mid)


def _pk(u, surf, recipe):
    """Seed-free key first (grid fit), then __as0."""
    return u.pkeys.get((surf, recipe, None)) or u.pkeys.get((surf, recipe, 0))


def identity_checks(units, dsets, rep, root=None, maps=None):
    """Selection, alias and shared-store identities, all from saved arrays and the runner's own records."""
    maps = maps or {}
    by_dsseed = {}
    for uid, u in units.items():
        if u.preds is None or (u.alias_of and u.meta["arm"] == "C" and not (u.path / "preds.npz").exists()):
            continue
        by_dsseed.setdefault((u.meta["ds"], u.meta["seed"], u.meta["purpose"], u.meta["attr"]), []).append(u)
        pr = u.preds
        ds = dsets[u.meta["ds"]]
        for surf in ("rep", "outputs", "repPLUSoutputs"):
            nl = u.pkeys.get((surf, "NL", 0))
            if nl is None:
                continue
            cands = {r: k for (s_, r, a), k in u.pkeys.items() if s_ == surf and r not in ("NL",) and a in (0, None)}
            same = sorted({r for r, k in cands.items() if np.array_equal(pr[k], pr[nl])})
            rep.add(f"{uid}: {surf} NL-selected (as0) equals a slate candidate", "selection", same,
                    same if same else None, status="PASS" if same else "FAIL", kind="info")
            # selection recomputed from attacker_val predictions, only when every candidate's val predictions exist
            vc = {r: k for (s_, r, a), k in u.vkeys.items() if s_ == surf and r != "NL" and a in (0, None)}
            if vc and same and set(r for r in cands if r not in ("L", "Lslate")) <= set(vc) and "val_row_id" in pr:
                yv = ds.label(u.meta["attr"])[[ds.pos[int(r)] for r in pr["val_row_id"]]]
                lls = {r: float(np.mean(ll_rows(np.asarray(pr[k])[np.arange(len(yv)), yv]))) for r, k in vc.items()}
                best = min(lls, key=lls.get)
                rep.add(f"{uid}: {surf} NL selection == argmin attacker_val log loss over the slate", "selection",
                        same, best, status="PASS" if best in same else "FAIL", kind="info")
            for (s_, r, a), k in u.pkeys.items():
                if s_ == surf and r in ("L", "Lslate") and a not in (None, 0) and (surf, r, 0) in u.pkeys:
                    rep.add(f"{uid}: {surf} {r} as{a} == as0 (deterministic recipe aliased)", "selection",
                            bool(np.array_equal(pr[k], pr[u.pkeys[(surf, r, 0)]])), True, kind="exact")
        if "val_y_s" in pr and "val_row_id" in pr:
            rep.add(f"{uid}: val_y_s == labels[{u.meta['attr']}]", "ids_roles",
                    int(np.sum(np.asarray(pr["val_y_s"]).astype(int) !=
                               ds.label(u.meta["attr"])[[ds.pos[int(r)] for r in pr["val_row_id"]]])), 0, kind="exact")
        # the runner's recorded selections: argmin of the recorded attacker_val log losses, and the saved NL as0
        # predictions equal the selected candidate's predictions
        for r in (u.fit_records.get("recipes") or []):
            surf = canon_surface(str(r.get("surface", "")))
            sels = []
            if r.get("recipe") == "NL" and isinstance(r.get("nl_selection"), dict):
                sels.append(("NL", r["nl_selection"].get("selected_family"), r["nl_selection"].get("candidates")))
            if r.get("recipe") == "NL" and isinstance(r.get("plus_selection"), dict):
                for slate, v in r["plus_selection"].items():
                    if isinstance(v, dict):
                        sels.append(("NL" if slate == "NL" else "Lslate" if slate == "L" else slate,
                                     v.get("selected"), v.get("candidates_attacker_val_log_loss")))
            for tgt, sel, cl in sels:
                if not sel or not isinstance(cl, dict) or not cl:
                    continue
                am = min(cl, key=cl.get)
                rep.add(f"{uid}: {surf} {tgt} recorded selection == argmin recorded attacker_val log loss", "selection",
                        sel, am, kind="exact")
                tk, ck = u.pkeys.get((surf, tgt, 0)), _pk(u, surf, sel)
                if tgt == "Lslate" and sel != "L":
                    ck = None     # an ignore candidate of the L slate is an L-family model not saved under its own key
                if tk and ck:
                    rep.add(f"{uid}: {surf} {tgt} as0 predictions == selected candidate '{sel}'", "selection",
                            bool(np.array_equal(pr[tk], pr[ck])), True, kind="exact")
        # ignore candidates
        for cand, src in (("ignore_rep", "outputs"), ("ignore_out", "rep")):
            ck, sk = _pk(u, "repPLUSoutputs", cand), u.pkeys.get((src, "NL", 0))
            if ck and sk:
                rep.add(f"{uid}: repPLUSoutputs '{cand}' candidate == {src} NL as0 predictions", "selection",
                        bool(np.allclose(pr[ck], pr[sk], atol=PROB_TOL, rtol=0)), True, kind="exact")
        # shared stores: outputs_only and U2
        if root is not None:
            for kind, keys in (("outputs_only", [k for k in pr if k.startswith("P__outputs__")]), ("U2", ["U2_P"])):
                key = (u.shared.get(kind) or {}).get("key")
                if not key:
                    continue
                f = root / "shared" / kind / key / "preds.npz"
                if not f.exists():
                    rep.add(f"{uid}: shared/{kind}/{key}/preds.npz present", "aliases", False, True, kind="exact")
                    continue
                sh = load_npz(f)
                bad = [k for k in keys if k not in sh or not np.array_equal(sh[k], pr[k])]
                if "assess_row_id" in sh:
                    bad += [] if np.array_equal(sh["assess_row_id"], pr["assess_row_id"]) else ["assess_row_id"]
                rep.add(f"{uid}: {kind} arrays == shared/{kind}/{key}", "aliases", bad, [], kind="exact")
        # HX (transformed assessment representation) and ZP (policy concept one-hot), when saved
        H, _, _ = ds.forward(u.meta["seed"], u.meta["purpose"])
        idx = np.array([ds.pos[int(r)] for r in pr["assess_row_id"]])
        scale = max(1.0, float(np.max(np.abs(H[idx]))))
        if "HX" in pr and u.meta["arm"] in ("A", "B", "C"):
            exp = H[idx]
            if u.meta["arm"] in ("B", "C"):
                mid, mp = _map_for_unit(u, ds, maps)
                exp = None if mp is None else leace_apply(H[idx], np.asarray(mp[1]["proj_left"]),
                                                           np.asarray(mp[1]["proj_right"]), map_bias(mp[1]))
            if exp is not None:
                dd = float(np.max(np.abs(np.asarray(pr["HX"]) - exp)))
                note = None
                if dd > TRANSFORM_TOL * scale and u.meta["arm"] in ("B", "C"):
                    alt = leace_apply(H[idx], np.asarray(mp[1]["proj_left"]), np.asarray(mp[1]["proj_right"]),
                                      H[idx].mean(0))
                    if np.max(np.abs(np.asarray(pr["HX"]) - alt)) <= TRANSFORM_TOL * scale:
                        note = "HX reproduced with the ASSESSMENT mean instead of the fitting mean"
                rep.add(f"{uid}: saved HX == transform of assessment rows with the saved map (fitting mean)", "leace",
                        dd, 0.0, tol=TRANSFORM_TOL * scale, note=note)
        if "ZP" in pr:
            pur = ds.spec["purposes"][u.meta["purpose"]]
            dims = pur.get("disallowed_attr_dims", {})
            Zd = np.concatenate([onehot(ds.label(a)[idx], range(int(dims.get(a, ds.label(a).max() + 1))))
                                 for a in pur["disallowed_attrs"]], axis=1)
            rep.add(f"{uid}: saved ZP == policy-set marginal one-hots of the assessment rows", "leace",
                    float(np.max(np.abs(np.asarray(pr["ZP"]) - Zd))) if np.shape(pr["ZP"]) == Zd.shape else None,
                    0.0, tol=0.0)
    # outputs-only surface fitted once per (dataset, encoder seed, purpose, attribute), aliased across methods
    for (d, s, _p, _a), us in by_dsseed.items():
        ref = next((u for u in us if u.meta["arm"] == "A"), us[0])
        for u in us:
            if u is ref:
                continue
            for key3, k in u.pkeys.items():
                if key3[0] == "outputs" and key3 in ref.pkeys:
                    rep.add(f"{u.uid}: outputs/{key3[1]}/as{key3[2]} identical to {ref.uid}", "aliases",
                            bool(np.array_equal(u.preds[k], ref.preds[ref.pkeys[key3]])), True, kind="exact")


def g1_rederive(units, dsets, maps, rep):
    for uid, u in units.items():
        if u.preds is None or "G1_pred" not in u.preds:
            continue
        m, ds = u.meta, dsets[u.meta["ds"]]
        H = ds.forward(m["seed"], m["purpose"])[0]
        if m["arm"] in ("B", "C"):
            mid, mp = _map_for_unit(u, ds, maps)
            if mp is None:
                rep.add(f"{uid}: G1 re-derivation", "estimates", None, mid, status="FAIL", kind="info",
                        note="saved map not found")
                continue
            d = mp[1]
            H = leace_apply(H, np.asarray(d["proj_left"]), np.asarray(d["proj_right"]), map_bias(d))
        elif m["arm"] == "D":
            rid = ds.forward(m["seed"], m["purpose"])[2]
            noise = np.random.default_rng(m["rs"]).normal(0.0, m["sigma"], size=(len(rid), H.shape[1]))
            nl = np.zeros_like(H)
            nl[np.array([ds.pos[int(r)] for r in rid])] = noise   # one draw over the full cache matrix, cache order
            H = H + nl
        y = ds.label(m["attr"])
        K = u.preds["G1_pred"].shape[1]
        Y = onehot(y, range(K))
        fm = ds.role == "attacker_fit"
        idx = np.array([ds.pos[int(r)] for r in u.preds["assess_row_id"]])
        pred, ybar = ridge_onehot(H[fm], Y[fm], H[idx])
        dd = float(np.max(np.abs(pred - u.preds["G1_pred"])))
        st, note = None, None
        if m["arm"] == "D" and dd > G1_REDERIVE_TOL:
            st, note = "NOTE", ("noise draw convention default_rng(rs).normal(0, sigma, full matrix in forward-cache "
                                "row order) not reproduced; the saved G1_pred is still scored as saved")
        rep.add(f"{uid}: saved G1_pred == fixed-ridge(1e-6) re-derivation on attacker_fit -> assessment", "estimates",
                dd, 0.0, tol=G1_REDERIVE_TOL, status=st, note=note)


def native_A(units, dsets, rep, results, historical):
    """Arm A native check per (dataset, purpose, attribute, encoder seed) with an A unit: historical in-sample
    one-hot ridge R^2 on the test split (float32 centring/Gram = historical; float64 = exact), tau = 0.05."""
    out = {}
    done = set()
    for uid, u in units.items():
        m = u.meta
        if m["arm"] != "A" or (m["ds"], m["purpose"], m["attr"], m["seed"]) in done:
            continue
        done.add((m["ds"], m["purpose"], m["attr"], m["seed"]))
        ds = dsets[m["ds"]]
        test = ds.split == "test"
        H = ds.forward(m["seed"], m["purpose"])[0][test]
        y = ds.label(m["attr"])[test]
        Y = np.eye(int(y.max()) + 1)[y]
        r32 = ridge_onehot_r2_insample(H, Y, float32_gram=True)
        r64 = ridge_onehot_r2_insample(H, Y)
        out[uid] = {"N0_float32": r32, "N0_float64": r64, "N0_rows": int(test.sum()), "historical_pass": r32 <= TAU}
        h = (historical or {}).get((m["ds"], m["seed"], m["purpose"], m["attr"]))
        if h is not None:
            out[uid]["historical_r2_onehot"] = h
            rep.add(f"{uid}: N0 (float32 historical path, clamped) reproduces dominant_axis_audit r2_onehot", "native",
                    h, max(0.0, r32), tol=1e-5, note="historical value computed in float32")
    results["native_A"] = out


def compute_sigma_star(units, dsets, sup_by_unit, val_info, cfg, rep):
    out = {}
    for name, ds in dsets.items():
        curve = {}
        for sg in cfg.sigma_grid:
            vals = []
            for s in sorted(ds.encoders):
                for k in cfg.release_seeds:
                    uid = f"{name}__s{s}__{ds.cell['purpose']}__{ds.cell['target']}__D_sigma{fmt_sigma(sg)}_rs{k}"
                    u = units.get(uid)
                    if u is None or u.preds is None or uid not in val_info:
                        continue
                    vk = u.vkeys.get(("rep", "NL", 0)) or u.vkeys.get(("rep", "NL", None))
                    if vk is None:
                        continue
                    yv = ds.label(ds.cell["target"])[val_info[uid]]
                    P = np.asarray(u.preds[vk], np.float64)
                    W = np.ones((1, len(yv)))
                    aucs = [WAUC(P[:, c], yv == c)(W)[0] for c in sup_by_unit[uid]["supported_classes"]]
                    vals.append(float(np.mean(aucs)))
            curve[fmt_sigma(sg)] = {"mean_val_macro_auc": float(np.mean(vals)) if vals else None, "n": len(vals)}
        q = [sg for sg in sorted(cfg.sigma_grid) if curve[fmt_sigma(sg)]["mean_val_macro_auc"] is not None
             and curve[fmt_sigma(sg)]["mean_val_macro_auc"] <= SIGMA_STAR_BAR]
        star = q[0] if q else max(cfg.sigma_grid)
        out[name] = {"sigma_star": star, "flagged_no_sigma_qualifies": not q, "val_curve": curve}
        rep.add(f"{name}: sigma* (attacker_val, encoder x release seeds, attacker seed 0)", "sigma_star",
                None, out[name], status="INFO", kind="info")
    return out


def register_summaries(units, stats, comps, sup_by_unit, cfg):
    """SUM|<ds>|<group>|<quantity> = mean over (encoder seeds x release seeds x attacker seeds) of unit stats."""
    groups = {}
    for uid, u in units.items():
        if u.preds is None:
            continue
        groups.setdefault((u.meta["ds"], u.meta["group"]), []).append(uid)
    summaries, members = {}, {}
    by_uid = {}
    for k in stats:
        uid, q = k.split("|", 1)
        by_uid.setdefault(uid, []).append((k, q))
    for (d, g), uids in sorted(groups.items()):
        per_base = {}
        for uid in uids:
            for k, q in by_uid.get(uid, []):
                base = re.sub(r"\|as\d+(?=\|)", "", q)
                per_base.setdefault(base, {}).setdefault(uid, []).append(k)
        for base, mp in per_base.items():
            if len(mp) != len(uids):
                continue        # incomplete coverage across members: not summarised
            names = [n for uid in uids for n in mp[uid]]
            fns = [stats[n] for n in names]
            key = f"SUM|{d}|{g}|{base}"
            summaries[key] = (lambda fns=fns: lambda ctx: np.mean([f(ctx) for f in fns], axis=0))()
            members[key] = names
        # derived worst bounds use the per-class summaries
        for base in list(per_base):
            for kind, pat in (("class", r"\|AUC_class_\d+$"), ("pair", r"\|AUC_pair_\d+_\d+$")):
                if re.search(pat, base):
                    head = re.sub(pat, "", base)
                    dk = f"SUM|{d}|{g}|{head}|AUC_worst_{kind}_derived"
                    sk = f"SUM|{d}|{g}|{base}"
                    if sk in summaries:
                        members.setdefault(dk, []).append(sk)
    return summaries, members


def primary_endpoints(dsets, units, sigma_star, summaries, sup_by_unit, sum_members, maps, cfg, rep):
    eps = []
    for name in sorted(dsets):
        ds = dsets[name]
        ss = sigma_star.get(name, {}).get("sigma_star")
        dgrp = f"D_sigma{fmt_sigma(ss)}" if ss is not None else None

        def est(grp):
            us = [u for u in units.values() if u.meta["ds"] == name and u.meta["group"] == grp and u.preds is not None]
            return bool(us) and all(sup_by_unit[u.uid]["estimable"] for u in us)

        def alias(grp):
            cs = [u for u in units.values() if u.meta["ds"] == name and u.meta["group"] == "C"]
            return grp == "C" and bool(cs) and all(u.alias_claim for u in cs)
        rows = [("G1", "A", f"SUM|{name}|A|G1|R2")]
        for fam, surf in (("Rrep", "rep"), ("Rplus", "repPLUSoutputs")):
            for arm in ("A", "B", "C", "D"):
                g = dgrp if arm == "D" else arm
                rows.append((fam, arm, f"SUM|{name}|{g}|P|{surf}|NL|AUC_macro"))
        for arm in ("B", "C", "D"):
            g = dgrp if arm == "D" else arm
            a, b = f"SUM|{name}|{g}|U2|accuracy", f"SUM|{name}|A|U2|accuracy"
            k = f"U2NI|{name}|{g}"
            if a in summaries and b in summaries:
                fa, fb = summaries[a], summaries[b]
                summaries[k] = (lambda fa=fa, fb=fb: lambda ctx: fa(ctx) - fb(ctx))()
                sa = {u.meta["seed"] for u in units.values() if u.meta["ds"] == name and u.meta["group"] == g}
                sb = {u.meta["seed"] for u in units.values() if u.meta["ds"] == name and u.meta["group"] == "A"}
                rep.add(f"U2NI[{name},{arm}]: paired on identical encoder seeds", "primary", sorted(sa), sorted(sb),
                        kind="exact")
            rows.append(("U2NI", arm, k))
        for fam, arm, k in rows:
            g = dgrp if arm == "D" else arm
            ok = k in summaries and (fam in ("G1", "U2NI") or est(g))
            n_mem = len([u for u in units.values() if u.meta["ds"] == name and u.meta["group"] == g])
            eps.append({"id": f"{fam}[{name},{arm}]", "family": fam, "dataset": name, "arm": arm, "group": g,
                        "n_members": n_mem,
                        "stat": k if ok else None, "decision": None if ok else "NE",
                        "alias_of": f"{fam}[{name},B]" if arm == "C" and alias("C") else None,
                        "reason": None if ok else ("statistic missing" if k not in summaries else
                                                   "not estimable (<2 supported classes)")})
    return eps


def seed_variation(point, units, cfg):
    out = {}
    pat = re.compile(r"^(?P<uid>[^|]+)\|(?P<head>.*?)\|as(?P<a>\d+)\|(?P<m>AUC_macro|AUC_worst_class|LL_skill|brier_skill)$")
    tmp = {}
    for k, v in point.items():
        m = pat.match(k)
        if m:
            tmp.setdefault((m["uid"], m["head"], m["m"]), []).append(v)
    for (uid, head, met), vals in tmp.items():
        out[f"{uid}|{head}|{met}|attacker_seed_sd"] = {
            "points": vals, "sd_ddof1": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "sd_ddof0": float(np.std(vals))}
    # encoder-seed SD of per-seed means (over release x attacker seeds)
    enc = {}
    for (uid, head, met), vals in tmp.items():
        u = units[uid]
        enc.setdefault((u.meta["ds"], u.meta["group"], head, met), {}).setdefault(u.meta["seed"], []).extend(vals)
    for (d, g, head, met), byseed in enc.items():
        means = [float(np.mean(v)) for s, v in sorted(byseed.items())]
        out[f"SUM|{d}|{g}|{head}|{met}|encoder_seed_sd"] = {
            "per_seed_means": means, "sd_ddof1": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "sd_ddof0": float(np.std(means))}
    return out


# ---------------------------------------------------------------- runner-report comparison
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(s):
    try:
        if s is None or str(s).strip() == "":
            return None
        return float(s)
    except ValueError:
        return None


def _print_tol(s):
    s = str(s).strip().lower()
    if "e" in s:
        mant, ex = s.split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return max(POINT_TOL, 0.5 * 10 ** (int(ex) - dec))
    dec = len(s.split(".")[1]) if "." in s else 0
    return max(POINT_TOL, 0.5 * 10 ** (-dec)) if dec < 15 else POINT_TOL


def compare_runner_reports(report_dir, res, rep):
    """Compare runner tables.  Layout used for the synthetic validation (adapted to the real headers after the
    run): PRIMARY_ENDPOINTS.csv (id, point, lower, upper, decision, alpha_each, B, seed, alias_of),
    SIGMA_STAR.json ({dataset: {sigma_star, flagged}}), WORST_CLASS.csv (key, point, lower, upper)."""
    mine = {r["id"]: r for r in res["primary"]}
    f = report_dir / "PRIMARY_ENDPOINTS.csv"
    rows = _read_csv(f) if f.exists() else []
    rep.add("PRIMARY_ENDPOINTS rows == family size 24", "primary", len(rows), FAMILY_SIZE, kind="exact")
    rep.add("PRIMARY_ENDPOINTS ids == replay family ids", "primary", sorted(r["id"] for r in rows), sorted(mine),
            kind="exact")
    for row in rows:
        m = mine.get(row["id"])
        if m is None:
            continue
        lab = row["id"]
        if row.get("alpha_each"):
            rep.add(f"{lab}: alpha_each", "primary", _num(row["alpha_each"]), ALPHA_FAMILY, tol=1e-15)
        rep.add(f"{lab}: alias_of", "aliases", row.get("alias_of") or None, m.get("alias_of"), kind="exact")
        if m["decision"] == "NE":
            rep.add(f"{lab}: decision", "primary", row["decision"], "NE", kind="exact")
            continue
        rep.add(f"{lab}: point", "primary", _num(row["point"]), m["estimate"], tol=_print_tol(row["point"]))
        for side in ("lower", "upper"):
            rep.add(f"{lab}: simultaneous {side} bound", "primary", _num(row[side]), m[side],
                    tol=bound_tol(m[f"mcse_{side}"]), note="MC tolerance (independent RNG stream)")
        st = "PASS" if row["decision"] == m["decision"] else ("MC_BORDERLINE" if m["mc_borderline"] else "FAIL")
        rep.add(f"{lab}: decision", "primary", row["decision"], m["decision"], status=st, kind="exact")
    f = report_dir / "SIGMA_STAR.json"
    if f.exists():
        js = json.loads(f.read_text())
        for d, v in res["sigma_star"].items():
            r = js.get(d, {})
            rep.add(f"{d}: sigma*", "sigma_star", _num(r.get("sigma_star")), v["sigma_star"], tol=0.0)
            rep.add(f"{d}: sigma* fallback flag", "sigma_star", bool(r.get("flagged")), v["flagged_no_sigma_qualifies"],
                    kind="exact")
    f = report_dir / "WORST_CLASS.csv"
    for row in (_read_csv(f) if f.exists() else []):
        w = res["worst_derived"].get(row["key"])
        if w is None:
            rep.add(f"{row['key']}: known to replay", "exploratory", row["key"], None, status="FAIL", kind="info")
            continue
        rep.add(f"{row['key']}: point (max of seed-mean class AUCs)", "exploratory", _num(row["point"]),
                w["estimate_max_of_seedmeans"], tol=_print_tol(row["point"]))
        for side in ("lower", "upper"):
            rep.add(f"{row['key']}: derived {side} bound (max of per-class a/K bounds)", "exploratory",
                    _num(row[side]), w[side], tol=bound_tol(w[f"mcse_{side}"]))


# ================================================================ runner-table replay (real report layout)
# Mapping layer only: every runner row is resolved to a statistic defined above (unit statistics registered by
# register_unit, means over member units and attacker seeds, max-of-per-class bounds), computed independently.
RUNNER_SURF = {"rep": "rep", "rep+outputs": "repPLUSoutputs", "outputs": "outputs"}
RUNNER_METRIC = {"macro_auc": "AUC_macro", "worst_class_auc": "AUC_worst_class", "worst_pair_auc": "AUC_worst_pair",
                 "LLR_nats": "LLR_nats", "LL_skill": "LL_skill", "brier_skill": "brier_skill"}
# naming ambiguities resolved by majority point agreement, applied uniformly (listed as adaptations)
RECIPE_ALTS = {("repPLUSoutputs", "L"): ["Lslate", "L"], ("repPLUSoutputs", "Lconcat"): ["L", "Lslate"]}


def tier2_inventory(root, report_dir, units_all, expected, rep):
    out = {}
    bi = root / "infer" / "BENCH_INFER.json"
    led = root / "logs" / "BUDGET_LEDGER.json"
    present = set(units_all)
    if bi.exists():
        b = json.loads(bi.read_text())
        found, miss = set(b.get("units_found", [])), set(b.get("units_missing", []))
        rep.add("BENCH_INFER units_found == unit directories present", "ids_roles", sorted(found ^ present), [],
                kind="exact", note=f"found {len(found)}, present {len(present)}")
        t1miss = [u for u in miss if u in set(expected)]
        rep.add("no Tier-1 unit missing", "ids_roles", t1miss, [], kind="exact")
        cells_t1 = {(parse_unit_id(u)["ds"], parse_unit_id(u)["purpose"], parse_unit_id(u)["attr"]) for u in expected}
        non_e3 = [u for u in miss if not (parse_unit_id(u) and parse_unit_id(u)["arm"] == "D" and
                                          (parse_unit_id(u)["ds"], parse_unit_id(u)["purpose"],
                                           parse_unit_id(u)["attr"]) not in cells_t1)]
        rep.add("every missing unit is an E3 (Tier-2 noise) unit", "ids_roles", sorted(non_e3), [], kind="exact")
        stops = json.loads(led.read_text()).get("stops", []) if led.exists() else []
        stop_units = [s_.get("at_unit") for s_ in stops]
        rep.add("budget stop recorded in BUDGET_LEDGER and the stop unit is not run", "ids_roles",
                [bool(stop_units), all(s_ in miss for s_ in stop_units)], [True, True], kind="exact",
                note=f"stops at {stop_units}")
        rep.add("E3 units not run (frozen budget rule)", "ids_roles", None, len(miss), status="NOT_RUN_BUDGET",
                kind="info", note=f"{len(miss)} absent by design (budget stop at {stop_units}); not a failure")
        out = {"units_present": len(present), "units_expected": b.get("units_expected"), "not_run_budget": len(miss),
               "budget_stops": stops}
    return out


class TableReplay:
    def __init__(self, units_all, dsets, stats, point, sup_by_unit, n_by_ds, rep):
        self.U, self.dsets, self.stats, self.point = units_all, dsets, stats, point
        self.sup, self.n_by_ds, self.rep = sup_by_unit, n_by_ds, rep
        self.cells = {}
        for uid, u in units_all.items():
            if u.preds is not None:
                m = u.meta
                self.cells.setdefault(f"{m['ds']}__{m['purpose']}__{m['attr']}", []).append(uid)
        self.by_uid = {}
        for k in stats:
            uid, q = k.split("|", 1)
            self.by_uid.setdefault(uid, set()).add(q)
        self.req = {}          # name -> fn
        self.ptcache = {}

    # ---- member sets
    def members(self, cell, ag, unit=""):
        if unit:
            return [unit] if unit in self.U and self.U[unit].preds is not None else []
        us = self.cells.get(cell, [])
        if ag == "CELL":
            return [u for u in us if self.U[u].meta["arm"] == "A"]
        if re.match(r"^D_sigma[0-9.]+_rs\d+$", ag):
            return [u for u in us if u.endswith("__" + ag)]
        return [u for u in us if self.U[u].meta["group"] == ag]

    # ---- unit quantity: list of unit-statistic names whose mean is the quantity (attacker-seed mean)
    def unit_names(self, uid, base, metric):
        qs = self.by_uid.get(uid, set())
        if base.startswith("P|"):
            _, surf, recipe = base.split("|")
            seeded = sorted(q for q in qs if re.fullmatch(rf"P\|{surf}\|{recipe}\|as\d+\|{re.escape(metric)}", q))
            if seeded:
                return [f"{uid}|{q}" for q in seeded]
            q = f"P|{surf}|{recipe}|fixed|{metric}"
            return [f"{uid}|{q}"] if q in qs else []
        q = f"{base}|{metric}" if metric else base
        return [f"{uid}|{q}"] if q in qs else []

    def group_fn(self, mem, base, metric):
        names = []
        for u in mem:
            nn = self.unit_names(u, base, metric)
            if not nn:
                return None
            names.append(nn)
        if not names:
            return None
        fns = [[self.stats[n] for n in nn] for nn in names]
        return lambda ctx: np.mean([np.mean([f(ctx) for f in ff], axis=0) for ff in fns], axis=0)

    def register(self, key, fn):
        if fn is not None:
            self.req[key] = fn
        return key if fn is not None else None

    def pt(self, key):
        if key not in self.ptcache:
            d = key.split("::", 1)[0]
            self.ptcache[key] = float(self.req[key](Ctx(np.ones((1, self.n_by_ds[d]))))[0])
        return self.ptcache[key]

    def stat(self, ds, cell, ag, unit, base, metric, tag=""):
        mem = self.members(cell, ag, unit)
        if not mem:
            return None, mem
        key = f"{ds}::{cell}|{ag}|{unit}|{base}|{metric}{tag}"
        if key in self.req:
            return key, mem
        return self.register(key, self.group_fn(mem, base, metric)), mem

    def diff(self, k1, k2):
        if k1 is None or k2 is None:
            return None
        key = f"{k1}::minus::{k2}"
        f1, f2 = self.req[k1], self.req[k2]
        return self.register(key, lambda ctx: f1(ctx) - f2(ctx))


def _dec_map(d):
    return {"ABOVE": "ESTABLISHED_ABOVE", "BELOW": "ESTABLISHED_BELOW"}.get(d, d)


def table_replay(report_dir, root, units_all, dsets, stats, point, sup_by_unit, inv_by_ds, n_by_ds, results, rep, cfg,
                 log, primary, sigma_star):
    rd = Path(report_dir)
    T = TableReplay(units_all, dsets, stats, point, sup_by_unit, n_by_ds, rep)
    adapt = results.setdefault("mapping_adaptations", [])
    rows_by_table = {}
    for name in ("RECOVERY.csv", "UTILITY.csv", "MATCHED_COMPARISONS.csv", "NATIVE_AND_HELDOUT_CHECKS.csv",
                 "SEED_VARIATION.csv", "SUPPORT_COVERAGE.csv", "PRIMARY_ENDPOINTS.csv"):
        f = rd / name
        rows_by_table[name] = _read_csv(f) if f.exists() else None
        rep.add(f"{name} present", "tables", f.exists(), True, kind="exact")

    def base_for(surface, recipe, metric):
        """Runner (surface, recipe, metric) -> list of (base, my_metric) alternatives."""
        if recipe in ("G1", "G2") and metric == "r2":
            return [(f"{recipe}", "R2")]
        if recipe == "RHO1":
            return [("RHO1SQ_heldout", "")]
        if recipe == "G1_scores":
            return [("G1pred", RUNNER_METRIC.get(metric, metric))]
        if recipe in ("crosscov_target", "crosscov_policy"):
            c = recipe.split("_")[1]
            return [(f"XCOV|{c}", metric), (f"XCOV|{c}", metric + "_ddof1")]
        mm = RUNNER_METRIC.get(metric)
        if mm is None:
            mc = re.fullmatch(r"cls(\d+)", metric)
            mp = re.fullmatch(r"pair(\d+)-(\d+)", metric)
            mm = f"AUC_class_{mc[1]}" if mc else (f"AUC_pair_{mp[1]}_{mp[2]}" if mp else None)
        if mm is None:
            return []
        if surface == "label_only":
            return [("LO", mm)]
        if surface == "constant":
            return [("CONST", mm)]
        s_ = RUNNER_SURF.get(surface)
        if s_ is None:
            return []
        return [(f"P|{s_}|{r}", mm) for r in RECIPE_ALTS.get((s_, recipe), [recipe])]

    def util_base(contract, metric):
        b = {"U1": "U1", "U2": "U2", "clean_output": "CLEAN", "Uconst": "Uconst"}.get(contract)
        if b is None:
            return []
        if metric == "log_loss":
            return [(b, "logloss"), (b, "logloss_unclipped")]
        return [(b, metric)]

    # ---------- pass 1: resolve rows to candidate statistics
    resolved = []     # (table, row, label, [cand keys], kind)
    lo_e, hi_e = (1 - LEVEL_EXPL) / 2, 1 - (1 - LEVEL_EXPL) / 2
    worst_specs = {}

    def worst_comp_keys(ds, cell, ag, base, kind):
        mem = T.members(cell, ag)
        if not mem:
            return None
        sup = sup_by_unit[mem[0]]
        comps_ = [f"AUC_class_{c}" for c in sup["supported_classes"]] if kind == "class" else \
            [f"AUC_pair_{j}_{k}" for j, k in sup["supported_pairs"]]
        ks = []
        for mt in comps_:
            k, _ = T.stat(ds, cell, ag, "", base, mt)
            if k is None:
                return None
            ks.append(k)
        return ks

    def rho_note(label):
        """Conditioning of the saved canonical variates: |mean| / sd of RHO_u and RHO_v for the units in the row."""
        us = [u for u in T.U if u in label] or [u for c, uu in T.cells.items() if c in label for u in uu]
        worst = 0.0
        for u in us[:9]:
            pr = T.U[u].preds
            if pr is not None and "RHO_u" in pr:
                for k_ in ("RHO_u", "RHO_v"):
                    a_ = np.asarray(pr[k_], np.float64)
                    worst = max(worst, abs(float(a_.mean())) / max(float(a_.std()), 1e-300))
        return f"max |mean|/sd of saved RHO variates = {worst:.3g}" if worst else None

    for tname in ("RECOVERY.csv", "UTILITY.csv"):
        for r in rows_by_table.get(tname) or []:
            ds, cell, ag, unit = r["dataset"], r["cell"], r["arm_group"], r.get("unit", "")
            label = r.get("id") or f"{cell}|{ag}|{unit}|{r['surface']}|{r['recipe']}|{r['metric']}"
            cands = util_base(r["contract"], r["metric"]) if tname == "UTILITY.csv" else \
                base_for(r["surface"], r["recipe"], r["metric"])
            keys = []
            for b, mt in cands:
                if mt in ("AUC_worst_class", "AUC_worst_pair"):
                    kind = "class" if mt.endswith("class") else "pair"
                    ck = worst_comp_keys(ds, cell, ag, b, kind)
                    k, _ = T.stat(ds, cell, ag, unit, b, mt)
                    if ck is not None:
                        worst_specs[(label, b)] = ck
                    keys.append(k)
                else:
                    k, _ = T.stat(ds, cell, ag, unit, b, mt)
                    keys.append(k)
            resolved.append((tname, r, label, keys, cands))
    for r in rows_by_table.get("MATCHED_COMPARISONS.csv") or []:
        ds, cell, comp, lev = r["dataset"], r["cell"], r["comparison"], r["level"]
        label = r["id"] or f"{cell}|{comp}"
        keys = []
        if lev == "primary":
            resolved.append(("MATCHED_COMPARISONS.csv", r, label, [], "primary"))
            continue
        m1 = re.fullmatch(r"(.+)-A", comp)
        m2 = re.fullmatch(r"(.+)\|plus_NL_minus_outputs_NL\|macro_auc", comp)
        m3 = re.fullmatch(r"CELL\|outputs_NL_minus_LO\|macro_auc", comp)
        if m1 and lev == "paired_diff_vs_A":
            cands = util_base(r["surface"], r["metric"]) if r["surface"] in ("U1", "U2") else \
                base_for(r["surface"], r["recipe"], r["metric"])
            seeds_x = {T.U[u].meta["seed"] for u in T.members(cell, m1[1])}
            for b, mt in cands:
                ka, _ = T.stat(ds, cell, m1[1], "", b, mt)
                mem_a = [u for u in T.members(cell, "A") if T.U[u].meta["seed"] in seeds_x]
                kb = T.register(f"{ds}::{cell}|A[seeds {sorted(seeds_x)}]||{b}|{mt}", T.group_fn(mem_a, b, mt)) \
                    if mem_a else None
                keys.append(T.diff(ka, kb))
        elif m2:
            ka, _ = T.stat(ds, cell, m2[1], "", "P|repPLUSoutputs|NL", "AUC_macro")
            kb, _ = T.stat(ds, cell, m2[1], "", "P|outputs|NL", "AUC_macro")
            keys.append(T.diff(ka, kb))
        elif m3:
            ka, _ = T.stat(ds, cell, "CELL", "", "P|outputs|NL", "AUC_macro")
            kb, _ = T.stat(ds, cell, "CELL", "", "LO", "AUC_macro")
            keys.append(T.diff(ka, kb))
        resolved.append(("MATCHED_COMPARISONS.csv", r, label, keys, "matched"))
    for r in rows_by_table.get("NATIVE_AND_HELDOUT_CHECKS.csv") or []:
        unit = r["unit"]
        if r.get("group_metric"):
            mg = re.fullmatch(r"(.+)\|(.+) \(seed mean\)", unit)
            if not mg:
                resolved.append(("NATIVE_AND_HELDOUT_CHECKS.csv", r, unit, [], "unmapped"))
                continue
            rec_, met = r["group_metric"].split("|")
            keys = [T.stat(r["dataset"], mg[1], mg[2], "", b, mt)[0] for b, mt in base_for("rep", rec_, met)]
            resolved.append(("NATIVE_AND_HELDOUT_CHECKS.csv", r, unit + " " + r["group_metric"], keys, "native_group"))
        else:
            ks = {}
            for col, (b, mt) in (("heldout_G1", ("G1", "R2")), ("heldout_G2", ("G2", "R2")),
                                 ("heldout_rho1sq", ("RHO1SQ_heldout", ""))):
                ks[col] = [T.stat(r["dataset"], r["cell"], "", unit, b, mt)[0]]
            for col, c in (("heldout_crosscov_target_fro", "target"), ("heldout_crosscov_policy_fro", "policy")):
                ks[col] = [T.stat(r["dataset"], r["cell"], "", unit, f"XCOV|{c}", m_)[0] for m_ in ("fro", "fro_ddof1")]
            resolved.append(("NATIVE_AND_HELDOUT_CHECKS.csv", r, unit, ks, "native_unit"))
    log(f"[tables] {len(resolved)} runner rows resolved; {len(T.req)} statistics requested")

    # ---------- uniform resolution of naming ambiguities (majority point agreement)
    tallies = {}
    for tname, r, label, keys, cands in resolved:
        if tname not in ("RECOVERY.csv", "UTILITY.csv") or len(keys) < 2:
            continue
        pv = _num(r.get("point"))
        if pv is None:
            continue
        tol = _print_tol(r["point"])
        grp = (tname, r.get("surface"), r.get("recipe"), r.get("contract") if tname == "UTILITY.csv" else "",
               r["metric"] if r["recipe"].startswith("crosscov") or r["metric"] == "log_loss" else "")
        t = tallies.setdefault(grp, [0] * len(keys))
        for i, k in enumerate(keys):
            if k is not None and abs(T.pt(k) - pv) <= tol:
                t[i] += 1
    choice = {g: int(np.argmax(t)) for g, t in tallies.items()}
    for g, t in tallies.items():
        adapt.append({"what": "naming ambiguity resolved uniformly by majority point agreement",
                      "group": list(g), "alternative_match_counts": t, "chosen_index": choice[g]})
    # worst-class point convention
    wc = [0, 0]
    for tname, r, label, keys, cands in resolved:
        if r.get("metric") in ("worst_class_auc", "worst_pair_auc") and tname == "RECOVERY.csv":
            ck = worst_specs.get((label, cands[0][0])) if cands else None
            pv = _num(r.get("point"))
            if ck and pv is not None:
                wc[0] += abs(max(T.pt(k) for k in ck) - pv) <= _print_tol(r["point"])
                wc[1] += keys[0] is not None and abs(T.pt(keys[0]) - pv) <= _print_tol(r["point"])
    adapt.append({"what": "worst-class/pair point convention matched",
                  "max_of_seedmean_class_auc": wc[0], "seedmean_of_unit_max": wc[1]})
    worst_conv = 0 if wc[0] >= wc[1] else 1

    # ---------- bootstrap (B = 2000, seed 20261003) over every requested statistic
    need = {}
    for tname, r, label, keys, cands in resolved:
        ks = []
        if isinstance(keys, dict):
            for v in keys.values():
                ks += v
        else:
            ks = list(keys)
        if tname in ("RECOVERY.csv", "UTILITY.csv") and len(ks) > 1:
            g = (tname, r.get("surface"), r.get("recipe"), r.get("contract") if tname == "UTILITY.csv" else "",
                 r["metric"] if r["recipe"].startswith("crosscov") or r["metric"] == "log_loss" else "")
            ks = [ks[choice.get(g, 0)]]
        if tname == "RECOVERY.csv" and r.get("metric") in ("worst_class_auc", "worst_pair_auc") and cands:
            ks += worst_specs.get((label, cands[choice.get((tname, r.get("surface"), r.get("recipe"), "", ""), 0)][0]),
                                  [])
        for k in ks:
            if k is not None:
                need[k] = T.req[k]
    t0 = time.time()
    reps_ = {}
    for name, ds in dsets.items():
        if name not in inv_by_ds:
            continue
        sub = {k: f for k, f in need.items() if k.split("::", 1)[0] == name}
        inv, nu = inv_by_ds[name]
        log(f"[tables] bootstrap {name}: {len(sub)} statistics, B={cfg.b_expl}")
        reps_.update(run_bootstrap(sub, inv, nu, cfg.b_expl, [cfg.seed_expl, ds.index]))
    log(f"[tables] bootstrap done in {time.time() - t0:.0f}s")

    def iv(k):
        return interval(reps_[k], lo_e, hi_e) if k in reps_ else None

    def cmp_point(scope, label, rv_s, k, note=None):
        if note is None and ("RHO1" in label or "rho1sq" in label):
            note = rho_note(label)
        rv = _num(rv_s)
        if k is None:
            rep.add(f"{label}: replay statistic resolved", scope, rv_s, None, status="FAIL", kind="info",
                    note="runner row could not be mapped to a replay statistic")
            return
        mine = T.pt(k)
        if rv is None:
            st = "PASS" if not np.isfinite(mine) else "FAIL"
            rep.add(f"{label}: point", scope, rv_s, mine, status=st, kind="info", note="runner point empty", quiet=True)
            return
        rep.add(f"{label}: point", scope, rv, mine, tol=_print_tol(rv_s), note=note, quiet=True)

    def cmp_bounds(scope, label, row, lcol, ucol, ivv):
        for side, col in (("lower", lcol), ("upper", ucol)):
            rv = _num(row.get(col))
            if rv is None or ivv is None or ivv.get(side) is None:
                continue
            rep.add(f"{label}: 90% {side}", scope, rv, ivv[side], tol=bound_tol(ivv[f"mcse_{side}"]),
                    note="MC tolerance (independent RNG stream)", quiet=True)

    def cmp_decisions(scope, label, dec_json, ivv, r2=False):
        try:
            d = json.loads(dec_json) if dec_json else {}
        except json.JSONDecodeError:
            return
        if ivv is None:
            return
        for kk, rd_ in d.items():
            if kk.startswith("NI_margin"):
                thr = NI_MARGIN
                md = decide_ni(ivv["lower"], ivv["upper"], thr)
            else:
                try:
                    thr = 0.0 if kk == "vs_0" else float(kk.split("=")[-1])
                except ValueError:
                    rep.add(f"{label}: decision key {kk} understood", scope, kk, None, status="FAIL", kind="info")
                    continue
                md = _dec_map(decide_bar(ivv["lower"], ivv["upper"], thr))
            tol = max(bound_tol(ivv["mcse_lower"]), bound_tol(ivv["mcse_upper"]))
            border = min(abs(ivv["lower"] - thr), abs(ivv["upper"] - thr)) <= tol
            st = "PASS" if rd_ == md else ("MC_BORDERLINE" if border else "FAIL")
            rep.add(f"{label}: decision {kk}", scope, rd_, md, status=st, kind="exact", quiet=True)

    def _compare_row(tname, r, label, keys, cands):
        scope = {"RECOVERY.csv": "recovery", "UTILITY.csv": "utility", "MATCHED_COMPARISONS.csv": "matched",
                 "NATIVE_AND_HELDOUT_CHECKS.csv": "native_heldout"}[tname]
        if tname in ("RECOVERY.csv", "UTILITY.csv"):
            if not keys:
                rep.add(f"{label}: mapped", scope, label, None, status="FAIL", kind="info",
                        note=f"no replay definition for {tname} row")
                return
            g = (tname, r.get("surface"), r.get("recipe"), r.get("contract") if tname == "UTILITY.csv" else "",
                 r["metric"] if r["recipe"].startswith("crosscov") or r["metric"] == "log_loss" else "")
            ci = choice.get(g, 0) if len(keys) > 1 else 0
            k = keys[ci]
            if r.get("metric") in ("worst_class_auc", "worst_pair_auc") and tname == "RECOVERY.csv":
                ck = worst_specs.get((label, cands[ci][0]))
                if ck is None or any(c not in reps_ for c in ck):
                    rep.add(f"{label}: worst components resolved", scope, None, None, status="FAIL", kind="info")
                    return
                pv = _num(r["point"])
                mine = max(T.pt(c) for c in ck) if worst_conv == 0 else T.pt(k)
                rep.add(f"{label}: point", scope, pv, mine, tol=_print_tol(r["point"]), quiet=True)
                K = len(ck)
                if _num(r.get("K")) is not None:
                    rep.add(f"{label}: K", scope, int(_num(r["K"])), K, kind="exact", quiet=True)
                db = derived_max_bounds({c: reps_[c] for c in ck}, lo_e)
                cmp_bounds(scope, label, r, "lower90", "upper90", db)
                cmp_decisions(scope, label, r.get("decisions"), db)
                return
            cmp_point(scope, label, r.get("point"), k)
            if r.get("boot_B") and _num(r.get("boot_B")) is not None:
                rep.add(f"{label}: B / seed", scope, [int(_num(r["boot_B"])), int(_num(r["boot_seed"]))],
                        [cfg.b_expl, cfg.seed_expl], kind="exact", quiet=True)
            if k is not None:
                ivv = iv(k)
                cmp_bounds(scope, label, r, "lower90", "upper90", ivv)
                cmp_decisions(scope, label, r.get("decisions"), ivv)
        elif tname == "MATCHED_COMPARISONS.csv":
            if cands == "primary":
                ds = r["dataset"]
                arm = r["comparison"].split("-")[0]
                arm = "D" if arm.startswith("D_") else arm
                m = next((p_ for p_ in primary if p_["id"] == f"U2NI[{ds},{arm}]"), None)
                if m is None or m.get("estimate") is None:
                    rep.add(f"{label}: primary U2NI known", scope, label, None, status="FAIL", kind="info")
                    return
                rep.add(f"{label}: primary point", scope, _num(r["point"]), m["estimate"], tol=_print_tol(r["point"]))
                for side, col in (("lower", "primary_lower"), ("upper", "primary_upper")):
                    if _num(r.get(col)) is not None:
                        rep.add(f"{label}: primary {side}", scope, _num(r[col]), m[side],
                                tol=bound_tol(m[f"mcse_{side}"]))
                if r.get("decision"):
                    st = "PASS" if r["decision"] == m["decision"] else ("MC_BORDERLINE" if m["mc_borderline"] else "FAIL")
                    rep.add(f"{label}: primary decision", scope, r["decision"], m["decision"], status=st, kind="exact")
                return
            k = keys[0] if keys and len(keys) == 1 else (keys[choice.get(("UTILITY.csv", r["surface"], r["recipe"],
                                                                            r["surface"], "log_loss"), 0)]
                                                         if keys else None)
            cmp_point(scope, label, r.get("point"), k)
            if k is not None:
                ivv = iv(k)
                cmp_bounds(scope, label, r, "lower90", "upper90", ivv)
                cmp_decisions(scope, label, r.get("decisions"), ivv)
        elif cands == "native_group":
            k = keys[0] if len(keys) == 1 else keys[choice.get(("RECOVERY.csv", "rep", r["group_metric"].split("|")[0],
                                                                "", r["group_metric"].split("|")[1]), 0)]
            cmp_point(scope, label, r.get("group_point"), k)
            if k is not None:
                ivv = iv(k)
                cmp_bounds(scope, label, r, "group_lower90", "group_upper90", ivv)
                if r.get("group_decision_tau") and ivv:
                    md = _dec_map(decide_bar(ivv["lower"], ivv["upper"], TAU))
                    tol = max(bound_tol(ivv["mcse_lower"]), bound_tol(ivv["mcse_upper"]))
                    border = min(abs(ivv["lower"] - TAU), abs(ivv["upper"] - TAU)) <= tol
                    st = "PASS" if r["group_decision_tau"] == md else ("MC_BORDERLINE" if border else "FAIL")
                    rep.add(f"{label}: decision tau", scope, r["group_decision_tau"], md, status=st, kind="exact",
                            quiet=True)
        elif cands == "native_unit":
            for col, ks in keys.items():
                if _num(r.get(col)) is None:
                    continue
                if col.startswith("heldout_crosscov"):
                    c = col.split("_")[2]
                    k = ks[choice.get(("RECOVERY.csv", "rep", f"crosscov_{c}", "", "fro"), 0)]
                else:
                    k = ks[0]
                cmp_point(scope, f"{label} {col}", r[col], k)
                if k is not None and _num(r.get(col + "_lower90")) is not None:
                    cmp_bounds(scope, f"{label} {col}", r, col + "_lower90", col + "_upper90", iv(k))
    # ---------- pass 2: compare
    for tname, r, label, keys, cands in resolved:
        try:
            _compare_row(tname, r, label, keys, cands)
        except Exception as e:  # noqa: BLE001
            rep.add(f"{label}: comparison executed", "tables", None, repr(e)[:200], status="FAIL", kind="info",
                    note="replay error while comparing this row")

    for nm_, fn_ in (("native fields", lambda: native_unit_fields(rows_by_table.get("NATIVE_AND_HELDOUT_CHECKS.csv")
                                                                 or [], units_all, dsets, results, rep)),
                     ("seed variation", lambda: seed_variation_table(rows_by_table.get("SEED_VARIATION.csv") or [], T,
                                                                     rep)),
                     ("support coverage", lambda: support_table(rows_by_table.get("SUPPORT_COVERAGE.csv") or [], T,
                                                                dsets, results, rep)),
                     ("primary table", lambda: primary_table(rd, rows_by_table.get("PRIMARY_ENDPOINTS.csv") or [],
                                                             primary, sigma_star, root, rep))):
        try:
            fn_()
        except Exception as e:  # noqa: BLE001
            import traceback
            rep.add(f"{nm_} comparison executed", "tables", None, repr(e)[:300], status="FAIL", kind="info",
                    note=traceback.format_exc()[-600:])
    results["tables"] = {"rows_resolved": len(resolved), "statistics_bootstrapped": len(need)}


def native_unit_fields(rows, units_all, dsets, results, rep):
    """Arm-A N0 and B/C LEACE fields per unit vs the replay's own native computations."""
    nA = results.get("native_A", {})
    lc = results.get("leace", {})
    for r in rows:
        if r.get("group_metric"):
            continue
        uid = r["unit"]
        u = units_all.get(uid)
        if u is None:
            continue
        if r["arm"] == "A":
            mine = nA.get(uid)
            if mine is None:
                continue
            for col, v in (("N0_mixed_clamped", max(0.0, mine["N0_float32"])), ("N0_mixed_raw", mine["N0_float32"]),
                           ("N0_float64_raw", mine["N0_float64"])):
                if _num(r.get(col)) is not None:
                    rep.add(f"{uid}: {col}", "native", _num(r[col]), v,
                            tol=1e-9 if "64" in col else N0_MIXED_TOL, quiet=True,
                            note=None if "64" in col else "float32 centring/Gram; BLAS-order sensitive")
            if _num(r.get("N0_rows")) is not None:
                rep.add(f"{uid}: N0_rows", "native", int(_num(r["N0_rows"])), mine["N0_rows"], kind="exact", quiet=True)
            if _num(r.get("historical_r2_onehot")) is not None and "historical_r2_onehot" in mine:
                rep.add(f"{uid}: historical_r2_onehot as read", "native", _num(r["historical_r2_onehot"]),
                        mine["historical_r2_onehot"], tol=1e-12, quiet=True)
        elif r["arm"] in ("B", "C"):
            mid = u.map_id
            mine = lc.get(mid)
            if mine is None:
                rep.add(f"{uid}: map {mid} replayed", "native", mid, None, status="FAIL", kind="info")
                continue
            sv = np.asarray(mine["whitened_singular_values"])
            zthr = mine.get("structural_zero_sv_threshold", 1e-10)
            for col, v, tol in (("bound_whitened_residual", mine["whitened_residual_spectral_saved"], 1e-9),
                                ("crosscov_max_abs_rel_erased", mine["xcov_after_maxcorr"], 1e-6),
                                ("ols_r2_joint_erased", mine["fit_ols_r2_after"], 1e-9),
                                ("n_singular_values_nonzero_truncated",
                                 int(np.sum((sv <= 0.01) & (sv > zthr))), 0)):
                rv = _num(r.get(col))
                if rv is None:
                    continue
                st_, note_ = None, None
                if col in ("crosscov_max_abs_rel_erased", "ols_r2_joint_erased") and max(abs(rv), abs(v)) < 1e-6:
                    st_, note_ = "PASS", "both at numerical zero (< 1e-6); descriptive"
                if col == "ols_r2_joint_erased" and st_ is None and abs(rv - v) > max(tol, 1e-6 * abs(v)):
                    var_ = mine.get("fit_ols_r2_rcond_variants", {})
                    hit_ = [k_ for k_, x_ in var_.items() if abs(x_ - rv) <= max(1e-9, 1e-6 * abs(rv))]
                    if hit_:
                        st_, note_ = "PASS", f"matches the least-squares solution at rcond={hit_[0]}"
                    elif var_ and min(var_.values()) - 1e-9 <= rv <= max(var_.values()) + 1e-9:
                        st_, note_ = "NOTE", ("rank-deficient erased design: OLS R^2 depends on the pseudo-inverse "
                                             f"cutoff; replay range over rcond {sorted(var_.values())}")
                if col == "crosscov_max_abs_rel_erased" and st_ is None and abs(rv - v) > max(tol, 1e-6 * abs(v)):
                    alt = mine["xcov_after_relative"]
                    note_ = f"max |corr| convention; replay max|cov|/max|cov_untreated| = {alt:.4g}"
                rep.add(f"{uid}: {col}", "native", rv, v, tol=max(tol, 1e-6 * abs(v)) if tol else 0, status=st_,
                        note=note_, quiet=True)
            rv = _num(r.get("defense_fit_cov_rank"))
            if rv is not None:
                lo_, hi_ = mine["sample_cov_rank_band"]
                st_ = "PASS" if rv == mine["sample_cov_rank_pinv"] else ("NOTE" if lo_ <= rv <= hi_ else "FAIL")
                rep.add(f"{uid}: defense_fit_cov_rank", "native", int(rv), mine["sample_cov_rank_pinv"], kind="exact",
                        status=st_, quiet=True,
                        note=f"tolerance-dependent rank of a numerically rank-deficient covariance (the replay's pinv "
                             f"rule reproduces the map's own diagnostics.sample_cov_rank; this table column uses "
                             f"another cutoff); replay ranks within a 1e3 band of the pinv threshold: [{lo_}, {hi_}]")
            if r.get("bound_holds"):
                rep.add(f"{uid}: bound_holds", "native", r["bound_holds"] == "True",
                        bool(mine["whitened_residual_spectral_saved"] <= 0.01 * (1 + 1e-9) + 1e-12), kind="exact",
                        quiet=True)
            if r.get("native_status"):
                rep.add(f"{uid}: native_status", "native", r["native_status"],
                        "PASS" if mine["whitened_residual_spectral_saved"] <= 0.01 * (1 + 1e-9) + 1e-12 else "FAIL",
                        kind="exact", quiet=True)


def seed_variation_table(rows, T, rep):
    for r in rows:
        cell, ag = r["cell"], r["arm_group"]
        s_ = RUNNER_SURF.get(r["surface"])
        mm = RUNNER_METRIC.get(r["metric"])
        label = f"SEED_VARIATION {cell}|{ag}|{r['surface']}|{r['recipe']}|{r['metric']}"
        alts = RECIPE_ALTS.get((s_, r["recipe"]), [r["recipe"]])
        best = None
        for rc in alts:
            pts = {}
            ok = True
            for uid in T.members(cell, ag):
                nn = T.unit_names(uid, f"P|{s_}|{rc}", mm)
                if not nn:
                    ok = False
                    break
                m = T.U[uid].meta
                pts[f"s{m['seed']}_rs{m['rs'] if m['rs'] is not None else 'None'}"] = [T.point[n] for n in nn]
            if not ok:
                continue
            try:
                rp = json.loads(r["per_unit_attacker_seed_points"])
            except Exception:  # noqa: BLE001
                rp = {}
            d = max((abs(a - b) for k in rp for a, b in zip(rp[k], pts.get(k, []))), default=np.inf) \
                if set(rp) == set(pts) else np.inf
            if best is None or d < best[0]:
                best = (d, rc, pts)
        if best is None:
            rep.add(f"{label}: mapped", "seed_variation", label, None, status="FAIL", kind="info")
            continue
        d, rc, pts = best
        rep.add(f"{label}: per-unit attacker-seed points", "seed_variation", d, 0.0, tol=1e-9, quiet=True,
                note=f"recipe mapped to {rc}")
        enc = {}
        for k, v in pts.items():
            enc.setdefault(k.split("_")[0][1:], []).append(v)
        per_enc = {e: float(np.mean([np.mean(v) for v in vs])) for e, vs in enc.items()}
        try:
            rpe = json.loads(r["per_encoder_seed_point"])
            dd = max(abs(rpe[e] - per_enc[e]) for e in rpe)
            rep.add(f"{label}: per-encoder-seed points", "seed_variation", dd, 0.0, tol=1e-9, quiet=True)
        except Exception:  # noqa: BLE001
            pass
        vals = list(per_enc.values())
        for col, cands_ in (("encoder_seed_sd", [np.std(vals, ddof=1) if len(vals) > 1 else np.nan, np.std(vals)]),
                            ("attacker_seed_sd_mean",
                             [np.mean([np.std(v, ddof=1) for v in pts.values()]) if all(len(v) > 1 for v in pts.values())
                              else np.nan, np.mean([np.std(v) for v in pts.values()])]),
                            ("attacker_seed_sd_max",
                             [np.max([np.std(v, ddof=1) for v in pts.values()]) if all(len(v) > 1 for v in pts.values())
                              else np.nan, np.max([np.std(v) for v in pts.values()])])):
            rv = _num(r.get(col))
            if rv is None:
                continue
            hit = [i for i, c in enumerate(cands_) if np.isfinite(c) and abs(c - rv) <= 1e-9]
            rep.add(f"{label}: {col}", "seed_variation", rv, float(cands_[hit[0]] if hit else cands_[0]), tol=1e-9,
                    quiet=True, note=("ddof=1" if hit and hit[0] == 0 else "ddof=0" if hit else None))
        rv = _num(r.get("release_seed_sd_mean"))
        if rv is not None:
            c1, c0 = [], []
            for e in enc:
                rs_means = [np.mean(v) for k, v in pts.items() if k.startswith(f"s{e}_")]
                if len(rs_means) > 1:
                    c1.append(np.std(rs_means, ddof=1))
                    c0.append(np.std(rs_means))
            cand = [float(np.mean(c1)) if c1 else np.nan, float(np.mean(c0)) if c0 else np.nan]
            hit = [i for i, c in enumerate(cand) if np.isfinite(c) and abs(c - rv) <= 1e-9]
            rep.add(f"{label}: release_seed_sd_mean", "seed_variation", rv, cand[hit[0]] if hit else cand[0], tol=1e-9,
                    quiet=True)


def support_table(rows, T, dsets, results, rep):
    for r in rows:
        if r.get("unit"):
            rep.add(f"SUPPORT_COVERAGE unit row {r['unit']}", "support", r.get("unit") in T.U, False, kind="exact",
                    quiet=True, note="unit-level rows list units without outputs (not run)")
            continue
        cell = r["cell"]
        ds_, purpose, attr = cell.split("__")
        ds = dsets[ds_]
        if _num(r.get("class")) is None:
            continue
        what, c = r["what"], int(float(r["class"]))
        if what == "sensitive":
            y = ds.label(attr)
        elif what == "task":
            y = ds.task_label(purpose)
        else:
            y = None
        label = f"SUPPORT_COVERAGE {cell} {what} class {c}"
        if y is not None:
            cnt = [int(np.sum((y == c) & (ds.role == role_))) for role_ in TEST_ROLES]
            rep.add(f"{label}: counts fit/val/assessment", "support",
                    [int(_num(r[f"n_{x}"])) if _num(r.get(f"n_{x}")) is not None else None for x in TEST_ROLES], cnt,
                    kind="exact", quiet=True)
            sup = c in support_sets(y, ds.role, need_defense=False)["supported_classes"]
            if r.get("supported", "").strip():
                rep.add(f"{label}: supported", "support", r["supported"].strip() == "True", sup, kind="exact",
                        quiet=True)
        elif _num(r.get("n_defense_fit")) is not None:
            attrs = [attr] if what.endswith("_B") else ds.spec["purposes"][purpose]["disallowed_attrs"]
            # class index refers to the concatenated concept columns for C
            off, found = 0, None
            for a in attrs:
                K = int(ds.spec["purposes"][purpose].get("disallowed_attr_dims", {}).get(a, ds.label(a).max() + 1))
                if c < off + K:
                    found = (a, c - off)
                    break
                off += K
            if found is None:
                rep.add(f"{label}: mapped", "support", None, None, status="FAIL", kind="info")
                continue
            n = int(np.sum((ds.label(found[0]) == found[1]) & (ds.role == "defense_fit")))
            rep.add(f"{label}: n_defense_fit ({found[0]} class {found[1]})", "support", int(_num(r["n_defense_fit"])), n,
                    kind="exact", quiet=True)


def primary_table(rd, rows, primary, sigma_star, root, rep):
    mine = {}
    for p_ in primary:
        fam, rest = p_["id"].split("[")
        ds, arm = rest[:-1].split(",")
        mine[f"P-{ds}-{fam}-{'Dstar' if arm == 'D' else arm}"] = p_
    rep.add("PRIMARY_ENDPOINTS ids == replay family of 24", "primary", sorted(r["id"] for r in rows), sorted(mine),
            kind="exact")
    for r in rows:
        m = mine.get(r["id"])
        if m is None:
            continue
        lab = r["id"]
        rep.add(f"{lab}: alpha_each / B / seed", "primary",
                [_num(r.get("alpha_each")), int(_num(r.get("B")) or 0), int(_num(r.get("seed")) or 0)],
                [ALPHA_FAMILY, B_PRIM, SEED_PRIM], kind="exact")
        rep.add(f"{lab}: alias_of", "primary", r.get("alias_of") or None,
                (f"P-{m['dataset']}-{m['family']}-B" if m.get("alias_of") else None), kind="exact")
        if m["decision"] == "NE":
            rep.add(f"{lab}: decision", "primary", r["decision"], "NE", kind="exact")
            continue
        rep.add(f"{lab}: point", "primary", _num(r["point"]), m["estimate"], tol=_print_tol(r["point"]))
        for side in ("lower", "upper"):
            rep.add(f"{lab}: simultaneous {side} bound (alpha=0.05/24, B=20000)", "primary", _num(r[side]), m[side],
                    tol=bound_tol(m[f"mcse_{side}"]), note=f"replay MC SE {m[f'mcse_{side}']:.2e}; independent stream")
        md = _dec_map(m["decision"])
        st = "PASS" if r["decision"] == md else ("MC_BORDERLINE" if m["mc_borderline"] else "FAIL")
        rep.add(f"{lab}: decision", "primary", r["decision"], md, status=st, kind="exact")
        if r.get("n_members"):
            rep.add(f"{lab}: n_members", "primary", int(_num(r["n_members"])), m.get("n_members"), kind="exact")
        if m["arm"] == "D":
            rep.add(f"{lab}: sigma*", "primary", _num(r.get("sigma_star")), sigma_star[m["dataset"]]["sigma_star"],
                    tol=0.0)
    f = rd / "PRIMARY_FAMILY.json"
    if f.exists():
        pf = json.loads(f.read_text())
        rep.add("PRIMARY_FAMILY family_size / alpha_each / B / seed", "primary",
                [pf.get("family_size"), pf.get("alpha_each"), pf.get("B"), pf.get("seed")],
                [FAMILY_SIZE, ALPHA_FAMILY, B_PRIM, SEED_PRIM], kind="exact")
    for src in (root / "infer" / "SIGMA_STAR.json", rd / "SIGMA_STAR_SUMMARY.json"):
        if not src.exists():
            continue
        js = json.loads(src.read_text())
        blk = js.get("datasets", {})
        for d, v in sigma_star.items():
            rv = (blk.get(d) or {}).get("sigma_star", (js.get("values") or {}).get(d))
            if isinstance(rv, dict):
                rv = rv.get("sigma_star")
            rep.add(f"{d}: sigma* ({src.name})", "sigma_star", _num(rv), v["sigma_star"], tol=0.0)
            curve = (blk.get(d) or {}).get("val_curve_mean")
            if curve:
                for sg, val in curve.items():
                    mv = v["val_curve"].get(sg, {}).get("mean_val_macro_auc")
                    rep.add(f"{d}: attacker_val curve sigma={sg} ({src.name})", "sigma_star", val, mv, tol=1e-9)
            fl = (blk.get(d) or {}).get("flagged_fallback")
            if fl is not None:
                rep.add(f"{d}: sigma* fallback flag ({src.name})", "sigma_star", bool(fl),
                        v["flagged_no_sigma_qualifies"], kind="exact")


# ---------------------------------------------------------------- output
# (regex on check, cause): filled only after the discrepancy was investigated on the real outputs (2026-10-03)
DIAGNOSES = [
    (r"__marital_status\|D_sigma(0\.5|1|2|4|8)-A\|.*: (point|90% lower|90% upper)$",
     "RUNNER DEFECT (exploratory, Tier-2 E3 partial groups only): the budget stop left encoder seed s2 of adult "
     "employment_analysis/marital_status unrun for sigma >= 0.5, so these D groups hold s0/s1 only; the runner's "
     "'paired' D - A difference subtracts the arm-A mean over s0,s1,s2 (unpaired in encoder seed). The replay restricts "
     "A to the group's own encoder seeds; the two differ exactly by the A seed-composition gap."),
    (r"\|plus_NL_minus_outputs_NL\|macro_auc: (point|90% lower|90% upper)$",
     "RUNNER DEFECT (exploratory, Tier-2 E3 only): for adult employment_analysis/marital_status the budget stop left "
     "encoder seed s2 unrun for sigma >= 0.5, so the D groups hold s0/s1 only; the runner subtracts the CELL "
     "(s0,s1,s2) outputs-only mean from the group's (s0,s1) rep+outputs mean. The runner value equals exactly "
     "mean(outputs NL; s0,s1) - mean(outputs NL; s0,s1,s2) = 0.013953 (replay check to 1e-13), i.e. a seed-composition "
     "gap, not a paired contrast. The replay pairs on the group's own members (difference 0: rep+outputs NL selected "
     "the ignore-rep candidate in every member)."),
    (r"(RHO1\|rho1sq|rho1sq)",
     "RUNNER NUMERICAL CANCELLATION (exploratory): the saved canonical variate RHO_u of these units carries a huge "
     "offset (|mean|/sd up to ~4e6, e.g. mean -3.5e5, sd 0.079), so uncentred second moments lose ~1e-2 relative "
     "precision; the replay centres before forming weighted moments. The runner's own bootstrap interval "
     "[0.0010, 0.0035] excludes its point 0.0048 for adult s1 education_assessment C, confirming the instability."),
    (r"^(adult__employment_analysis__race\|C\|rep\+outputs\|L\|cls1: 90% lower|adult__s2__employment_analysis__race__"
     r"D_sigma1_rs2\|rep\|LRT_A2\|macro_auc: 90% upper|hmda__underwriting__race\|A\|rep\|L\|LL_skill: 90% lower|"
     r"adult__s1__education_assessment__(income|race|sex)__C\|U2\|U2\|accuracy: 90% lower|adult__s2__income_prediction__"
     r"(race|sex)__D_sigma2_rs0\|U2\|U2\|accuracy: 90% upper)$",
     "Monte Carlo exceedance (exploratory, B=2000): |runner - replay| exceeds the frozen 6.5 MC-SE tolerance by 0.3-3.5 "
     "%. Five distinct statistics (the U2 rows repeat one shared probe across attributes). A B=20000 reference on an "
     "independent seed puts the runner's B=2000 bound 1.3-3.4 SD (SD of a B=2000 percentile) from the reference "
     "quantile, and the replay's own B=2000 bound also deviates; MC noise across ~37,000 bound comparisons, not an "
     "implementation discrepancy (verification/mc_exceedance_reference.json)."),
    (r": defense_fit_cov_rank$",
     "tolerance-dependent rank of a numerically rank-deficient defense_fit covariance (eigenvalues spread over 1e-13.."
     "8 with no gap); the replay's pinv rule reproduces each map's own diagnostics.sample_cov_rank exactly, the table "
     "column uses a different cutoff; descriptive only"),
    (r": ols_r2_joint_erased$",
     "rank-deficient erased design: the fit-row OLS R^2 depends on the pseudo-inverse cutoff; descriptive only "
     "(native status is decided by the official implementation bound, which passes)"),
]


def public_summary(items, quiet_counts=None):
    by_status, by_scope = {}, {}
    for sc_, d_ in (quiet_counts or {}).items():
        for st_, n_ in d_.items():
            by_status[st_] = by_status.get(st_, 0) + n_
            by_scope.setdefault(sc_, {})[st_] = by_scope.setdefault(sc_, {}).get(st_, 0) + n_
    for it in items:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1
        sc = by_scope.setdefault(it["scope"], {})
        sc[it["status"]] = sc.get(it["status"], 0) + 1
    residual = []
    for it in items:
        if it["status"] not in ("PASS", "INFO"):
            cause = next((c for rx, c in DIAGNOSES if re.search(rx, it["check"])), None)
            if cause is None and it["status"] == "MC_BORDERLINE":
                cause = ("Monte Carlo: the replay bound lies within the frozen MC tolerance of the bar, so the call "
                         "flips between independent bootstrap streams; not an implementation discrepancy")
            if it["status"] == "NOT_RUN_BUDGET":
                cause = cause or it.get("note")
            residual.append({"check": it["check"], "scope": it["scope"], "status": it["status"],
                             "runner_value": it["runner_value"] if not isinstance(it["runner_value"], list) else "list",
                             "replay_value": it["replay_value"] if not isinstance(it["replay_value"], (list, dict)) else "structured",
                             "abs_diff": it["abs_diff"], "tolerance": it["tolerance"],
                             "cause": cause or it.get("note") or "undiagnosed"})
    return {"n_items": sum(by_status.values()), "by_status": by_status, "by_scope": by_scope,
            "non_pass_items": residual,
            "contains_per_person_values": False}


_S = "/"
FORBIDDEN_PATH_STRINGS = (_S + "Users" + _S, _S + "private" + _S + "tmp", _S + "Volumes" + _S)   # built, not literal


def scrub_text(txt, extra=()):
    """Public outputs carry no absolute private paths: replace known roots with placeholders, then refuse to write
    if any FORBIDDEN_PATH_STRINGS entry (home, system tmp, external volume roots) remains."""
    import os
    import tempfile
    for root, ph in [*[(e, "<scratch>") for e in extra], (tempfile.gettempdir(), "<tmp>"),
                     (str(Path("~/PCRL_eval_cache_private").expanduser()), "~/PCRL_eval_cache_private"),
                     (os.path.expanduser("~"), "~")]:
        if root and len(root) > 1:
            txt = txt.replace(root, ph)
    txt = txt.replace(_S + "private" + _S + "var" + _S + "folders", "<tmp>").replace(FORBIDDEN_PATH_STRINGS[1], "<tmp>")
    bad = [f for f in FORBIDDEN_PATH_STRINGS if f in txt]
    if bad:
        raise ValueError(f"refusing to write public output containing private path strings {bad}")
    return txt


def finish(rep, results, out_path, results_path, t0):
    ps = public_summary(rep.items, rep.quiet_counts)
    summ = ps["by_status"]
    payload = {"schema": "pcrl.matched_removal_benchmark.independent_verification/v1",
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "implementation": "results/combined_matched_removal_benchmark_v1/verification/replay_bench.py "
                                 "(numpy only; independent of stored_model_eval)",
               "tolerances": {"point": POINT_TOL, "bound": f"{BOUND_SE_MULT} x replay MC SE + {BOUND_FLOOR}",
                              "g1_rederive": G1_REDERIVE_TOL, "leace_rederive_rel": LEACE_REDERIVE_TOL,
                              "leace_mean_rel": LEACE_MEAN_TOL,
                              "leace_native": "||W P Sigma_xz||_2 <= svd_tol*(1+1e-9)+1e-12",
                              "transform_rel": TRANSFORM_TOL, "alias": ALIAS_TOL},
               "summary": summ, "public_summary": ps,
               "items_note": "stored items: every non-PASS item plus the non-quiet checks; quiet PASS/INFO items are "
                             "counted in public_summary only",
               "elapsed_s": round(time.time() - t0, 1), "items": rep.items}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(scrub_text(json.dumps(_jsonable(payload), indent=1)))
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(scrub_text(json.dumps(_jsonable(results), indent=1)))
    return payload, results


def load_historical(paths):
    """{(dataset, seed, purpose, attribute): r2_onehot} from dominant_axis_audit.json (final.pt audit)."""
    out = {}
    for ds, p in paths.items():
        d = json.loads(Path(p).read_text())
        per = d.get("per_seed", {"0": d})
        for s, blk in per.items():
            for r in blk.get("rows", []):
                out[(ds, int(s), r.get("purpose"), r.get("attribute"))] = float(r["r2_onehot"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/PCRL_eval_cache_private/bench_v1")   # private root, never in git
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--b-prim", type=int, default=B_PRIM)
    ap.add_argument("--b-expl", type=int, default=B_EXPL)
    ap.add_argument("--expl-scope", default="none", choices=("none", "summaries", "all"))
    ap.add_argument("--rediagnose", default=None, help="re-apply DIAGNOSES to an existing verification JSON (no "
                                                       "recomputation) and rewrite it")
    a = ap.parse_args(argv)
    if a.rediagnose:
        pth = Path(a.rediagnose)
        pl = json.loads(pth.read_text())
        rp = Report()
        rp.items = pl["items"]
        qc = {}
        for sc_, d_ in pl["public_summary"]["by_scope"].items():
            stored = {}
            for it in rp.items:
                if it["scope"] == sc_:
                    stored[it["status"]] = stored.get(it["status"], 0) + 1
            qc[sc_] = {st_: n_ - stored.get(st_, 0) for st_, n_ in d_.items() if n_ - stored.get(st_, 0) > 0}
        pl["public_summary"] = public_summary(rp.items, qc)
        pl["summary"] = pl["public_summary"]["by_status"]
        pth.write_text(scrub_text(json.dumps(_jsonable(pl), indent=1)))
        print(json.dumps(pl["summary"]))
        return 0
    cfg = Config(b_prim=a.b_prim, b_expl=a.b_expl, expl_scope=a.expl_scope)
    res_dir = Path(__file__).resolve().parents[2]
    hist_paths = {d: res_dir / f"v2_{d}_ROUND4" / "dominant_axis_audit.json" for d in CELLS}
    hist = load_historical({d: p for d, p in hist_paths.items() if p.exists()})
    payload, _ = replay(a.root, a.report_dir, a.out, a.results_out, cfg, historical=hist)
    print(json.dumps(payload["summary"]))
    for i in [i for i in payload["items"] if i["status"] == "FAIL"][:50]:
        print("FAIL:", i["check"], "| runner:", i["runner_value"], "| replay:", i["replay_value"])
    return 1 if any(i["status"] == "FAIL" for i in payload["items"]) else 0


if __name__ == "__main__":
    sys.exit(main())

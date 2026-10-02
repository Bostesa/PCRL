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
    def __init__(self):
        self.items = []

    def add(self, check, scope, runner, replay, tol=None, status=None, note=None, kind="value"):
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


def canon_surface(s):
    t = s.lower().replace("-", "_")
    if "rep" in t and "out" in t:
        return "repPLUSoutputs"
    if t.startswith("rep"):
        return "rep"
    if t.startswith("out"):
        return "outputs"
    return s


PKEY_RE = re.compile(r"^(?P<kind>P|V)__(?P<surf>[A-Za-z]+)__(?P<recipe>[A-Za-z0-9_]+?)(?:__as(?P<as>\d+))?$")


def parse_pkey(k):
    m = PKEY_RE.match(k)
    if not m:
        return None
    return m["kind"], canon_surface(m["surf"]), m["recipe"], (int(m["as"]) if m["as"] is not None else None)


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

    def label(self, attr):
        if attr == self.cell["task"] and self.task_key in self.labels:
            return np.asarray(self.labels[self.task_key]).astype(int)
        for k in (attr, f"task_{attr}"):
            if k in self.labels:
                return np.asarray(self.labels[k]).astype(int)
        raise KeyError(f"{self.name}: no label column for {attr}")

    def forward(self, seed):
        """(rep, logits, cache row_id order) with rep / logits in labels row order, float64."""
        if seed not in self._fwd:
            fc = load_npz(self.encoders[seed]["forward_cache"])
            rid = np.asarray(fc["row_id"]).astype(np.int64)
            idx = np.array([self.pos[int(r)] for r in rid])
            H = np.empty((len(self.row_id), fc[self.rep_key].shape[1]))
            H[idx] = np.asarray(fc[self.rep_key], np.float64)
            Lg = None
            if self.logits_key in fc:
                Lg = np.empty((len(self.row_id), fc[self.logits_key].shape[1]))
                Lg[idx] = np.asarray(fc[self.logits_key], np.float64)
            self._fwd[seed] = (H, Lg, rid)
        return self._fwd[seed]


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
    def __init__(self, uid, path, meta):
        self.uid, self.path, self.meta = uid, Path(path), meta
        self.alias_of = None
        al = self.path / "ALIAS.json"
        if al.exists():
            self.alias_of = json.loads(al.read_text()).get("alias_of")
        self.preds = load_npz(self.path / "preds.npz") if (self.path / "preds.npz").exists() else None
        sp = self.path / "supported.json"
        self.runner_supported = json.loads(sp.read_text()) if sp.exists() else None
        fr = self.path / "fit_records.json"
        self.fit_records = json.loads(fr.read_text()) if fr.exists() else {}
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
def register_unit(stats, uid, preds, pkeys, y, yt, sup, prior, t_prior=None):
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
        tag = f"P|{surf}|{recipe}" + (f"|as{a}" if a is not None else "")
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
    if yt is not None:
        yt = np.asarray(yt).astype(int)
        for name, arr, is_logit in (("U1", preds.get("U1_logits"), True), ("U2", preds.get("U2_P"), False)):
            if arr is None:
                continue
            arr = np.asarray(arr, np.float64)
            prob = np.exp(log_softmax(arr)) if is_logit else arr
            K = prob.shape[1]
            ok = (yt >= 0) & (yt < K)
            acc = (prob.argmax(1) == yt).astype(np.float64)
            reg(f"{name}|accuracy", wmean(acc))
            ptrue = np.where(ok, prob[np.arange(len(yt)), np.clip(yt, 0, K - 1)], 0.0)
            reg(f"{name}|logloss", wmean(ll_rows(ptrue)))
    return comps


# ---------------------------------------------------------------- LEACE map checks
def find_maps(root):
    """{(ds, seed, purpose, attr, arm): (dir, npz dict, meta dict)} for every saved LEACE map."""
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
        meta = {}
        for j in sorted(f.parent.glob("*.json")):
            try:
                meta.update(json.loads(j.read_text()))
            except Exception:  # noqa: BLE001
                pass
        pm = None
        for part in [f.parent.name, *[p.name for p in f.parents]]:
            pm = parse_unit_id(part)
            if pm:
                break
        if pm is None:
            continue
        out[(pm["ds"], pm["seed"], pm["purpose"], pm["attr"], pm["arm"])] = (f.parent, d, meta, f)
    return out


def map_bias(d):
    for k in ("mean_x", "mean", "bias"):
        if k in d:
            return np.asarray(d[k], np.float64).ravel()
    return None


def concept_matrix(ds, meta, arm, rep=None, mid=""):
    """Design concept: B = one-hot of the target attribute, C = concatenated marginal one-hots of the policy set,
    each over the FULL declared class range (disallowed_attr_dims), the coordinator's rule when every declared class
    has >= 100 defense_fit rows.  The runner's concept_spec (if saved) is compared, not used."""
    attrs = [ds.cell["target"]] if arm == "B" else list(ds.cell["policy"])
    fit = ds.role == "defense_fit"
    blocks, classes = [], {}
    for a in attrs:
        y = ds.label(a)
        K = int(ds.attr_dims.get(a, int(y.max()) + 1))
        cnt = [int(np.sum(fit & (y == c))) for c in range(K)]
        classes[a] = list(range(K))
        if rep is not None:
            rep.add(f"{mid}: every declared {a} class has >= {DEFENSE_MIN} defense_fit rows (full one-hot rule)",
                    "leace", min(cnt), DEFENSE_MIN, status="PASS" if min(cnt) >= DEFENSE_MIN else "FAIL", kind="info",
                    note=f"defense_fit counts per class: {cnt}")
        blocks.append(onehot(y, classes[a]))
    spec = meta.get("concept_spec")
    if rep is not None and isinstance(spec, dict):
        ra = spec.get("attributes") or spec.get("attrs")
        rc = {a: [int(c) for c in v] for a, v in (spec.get("classes") or {}).items()}
        rep.add(f"{mid}: saved concept_spec == design concept (attributes, full declared classes)", "leace",
                [list(ra or []), {a: rc.get(a) for a in (ra or [])}], [attrs, classes], kind="exact")
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
    for (dsn, seed, purpose, attr, arm), (mdir, d, meta, f) in sorted(maps.items()):
        if dsn != ds.name:
            continue
        mid = f"{dsn}__s{seed}__{purpose}__{attr}__{arm}"
        H = ds.forward(seed)[0]
        Lm, Rm = np.asarray(d["proj_left"], np.float64), np.asarray(d["proj_right"], np.float64)
        b = map_bias(d)
        Z, attrs = concept_matrix(ds, meta, arm, rep, mid)
        Hf, Zf = H[fit], Z[fit]
        n, dim = Hf.shape
        scale = max(1.0, float(np.max(np.abs(Hf))))
        P = np.eye(dim) - Lm @ Rm
        st = meta.get("settings", {}) or {}
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
        s_saved = _get(d, "singular_values", "s", "svals", "sv")
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
        frh = meta.get("fit_row_ids_sha256") or meta.get("fit_row_hash")
        if frh:
            ids = np.sort(ds.row_id[fit]).astype("<i8")
            cands = {"int64_bytes_sorted": hashlib.sha256(ids.tobytes()).hexdigest(),
                     "comma_joined_sorted": hashlib.sha256(",".join(map(str, ids.tolist())).encode()).hexdigest(),
                     "newline_joined_sorted": hashlib.sha256("\n".join(map(str, ids.tolist())).encode()).hexdigest(),
                     "json_list_sorted": hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()}
            hit = [k for k, v in cands.items() if v == frh]
            rep.add(f"{mid}: fit row-id sha256 reproduced from recomputed defense_fit row ids", "leace",
                    frh, cands[hit[0]] if hit else None, status="PASS" if hit else "FAIL", kind="exact",
                    note=f"convention={hit[0]}" if hit else "no known hashing convention matched; diagnose")
        out[mid] = rec
    # aliases B vs C (action on the fit rows decides; P compared descriptively)
    for seed in sorted(ds.encoders):
        kb = [k for k in maps if k[0] == ds.name and k[1] == seed and k[4] == "B"]
        kc = [k for k in maps if k[0] == ds.name and k[1] == seed and k[4] == "C"]
        cu = [u for u in (units or {}).values() if u.meta["ds"] == ds.name and u.meta["seed"] == seed
              and u.meta["arm"] == "C"]
        unit_claim = any(bool(u.alias_of) for u in cu)
        if not kb or not kc:
            if unit_claim:
                rep.add(f"{ds.name}__s{seed}: B==C alias claim verifiable from saved maps", "leace", True, False,
                        kind="exact", note="C unit claims an alias but the B or C map is not saved")
            continue
        _, db, mb, _ = maps[kb[0]]
        _, dc, mc, _ = maps[kc[0]]
        H = ds.forward(seed)[0]
        scale = max(1.0, float(np.max(np.abs(H[fit]))))
        xb = leace_apply(H, np.asarray(db["proj_left"]), np.asarray(db["proj_right"]), map_bias(db))
        xc = leace_apply(H, np.asarray(dc["proj_left"]), np.asarray(dc["proj_right"]), map_bias(dc))
        dd = float(np.max(np.abs(xb - xc))) / scale
        dP = float(np.max(np.abs((db["proj_left"] @ db["proj_right"]) - (dc["proj_left"] @ dc["proj_right"]))))
        mine = dd <= ALIAS_TOL
        claim = mc.get("alias_of") is not None or bool(mc.get("alias", False)) or unit_claim
        rep.add(f"{ds.name}__s{seed}: B==C alias claim", "leace", claim, mine, kind="exact",
                note=f"max |x_B - x_C| / scale over all rows = {dd:.3g} (alias tol {ALIAS_TOL}); max |P_B - P_C| = {dP:.3g}")
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
            len(missing) + len(extra), 0, kind="exact", note=f"missing={missing} extra={extra}")
    units = {}
    for uid in expected:
        if uid in present:
            units[uid] = Unit(uid, present[uid], parse_unit_id(uid))
    # alias resolution: an aliased C has no preds of its own and points to B
    for uid, u in units.items():
        if u.preds is None and u.alias_of:
            tgt = units.get(u.alias_of)
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
    for uid, u in units.items():
        if u.preds is None:
            continue
        m, ds = u.meta, dsets[u.meta["ds"]]
        pr = u.preds
        if not u.alias_of:
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
        yt = ds.label(ds.cell["task"])
        if "assess_unit" in pr:
            rep.add(f"{uid}: assess_unit == record unit", "ids_roles",
                    int(np.sum(np.asarray(pr["assess_unit"]).astype(np.int64) != ds.unit[idx])), 0, kind="exact")
        if "y_s" in pr:
            rep.add(f"{uid}: y_s == labels[{m['attr']}]", "ids_roles",
                    int(np.sum(np.asarray(pr["y_s"]).astype(int) != y[idx])), 0, kind="exact")
        if "y_task" in pr:
            rep.add(f"{uid}: y_task == labels[{ds.cell['task']}]", "ids_roles",
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
        if m["arm"] == "A" and "U1_logits" in pr:
            Lg = ds.forward(m["seed"])[1]
            if Lg is not None:
                rep.add(f"{uid}: U1_logits (untreated) == clean head logits", "estimates",
                        float(np.max(np.abs(pr["U1_logits"] - Lg[idx]))), 0.0, tol=1e-9)
        comps.update(register_unit(stats, uid, pr, u.pkeys, y[idx], yt[idx], sup, prior))
        if "val_row_id" in pr:
            vrid = np.asarray(pr["val_row_id"]).astype(np.int64)
            rep.add(f"{uid}: val_row_id set == attacker_val role", "ids_roles",
                    int(len(np.setxor1d(vrid, ds.row_id[ds.role == 'attacker_val']))), 0, kind="exact")
            val_info[uid] = np.array([ds.pos[int(r)] for r in vrid])
        if m["ds"] not in inv_by_ds:
            inv_by_ds[m["ds"]] = cluster_index(ds.unit[idx])

    # ---- surface / candidate / alias identities
    identity_checks(units, dsets, rep)
    # ---- held-out G1 re-derivation (A, B, C from forward reps / saved maps; D under the pilot noise convention)
    maps = find_maps(root)
    for name, ds in dsets.items():
        leace_checks(ds, maps, rep, results, units)
    g1_rederive(units, dsets, maps, rep)
    native_A(units, dsets, rep, results, historical)

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
    expl_names = [k for k in summaries] if cfg.expl_scope == "summaries" else list(allstats)
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
    results["seed_variation"] = seed_variation(point, units, cfg)
    results["support"] = {u: {"supported_classes": s["supported_classes"], "estimable": s["estimable"],
                              "counts": {str(c): v for c, v in s["counts"].items()},
                              "unsupported_reasons": {str(c): r for c, r in s["unsupported_reasons"].items()}}
                          for u, s in sup_by_unit.items()}
    results["n_assessment"] = {d: {"rows": int(n_by_ds[d]), "units": int(inv_by_ds[d][1])} for d in n_by_ds}
    results["point_summaries"] = {k: point[k] for k in summaries}
    results["point_units"] = {k: point[k] for k in stats}   # aggregate statistics only (no per-person values)
    if report_dir:
        compare_runner_reports(Path(report_dir), results, rep)
    return finish(rep, results, out_path, results_path, t0)


def identity_checks(units, dsets, rep):
    by_dsseed = {}
    for uid, u in units.items():
        if u.preds is None or u.alias_of:
            continue
        by_dsseed.setdefault((u.meta["ds"], u.meta["seed"]), []).append(u)
        pr = u.preds
        # NL-selected equals one of the slate candidates; selection on attacker_val log loss when saved
        for surf in ("rep", "outputs", "repPLUSoutputs"):
            nl = u.pkeys.get((surf, "NL", 0)) or u.pkeys.get((surf, "NL", None))
            if nl is None:
                continue
            cands = {r: k for (s, r, a), k in u.pkeys.items() if s == surf and r != "NL" and a in (0, None)}
            same = [r for r, k in cands.items() if np.array_equal(pr[k], pr[nl])]
            rep.add(f"{uid}: {surf} NL-selected (as0) equals a slate candidate", "selection", same,
                    same if same else None, status="PASS" if same else "FAIL", kind="info")
            vc = {r: k for (s, r, a), k in u.vkeys.items() if s == surf and r != "NL" and a in (0, None)}
            if vc and same and "val_row_id" in pr:
                dsx = dsets[u.meta["ds"]]
                yv = dsx.label(u.meta["attr"])[[dsx.pos[int(r)] for r in pr["val_row_id"]]]
                lls = {r: float(np.mean(ll_rows(np.asarray(pr[k])[np.arange(len(yv)), yv]))) for r, k in vc.items()}
                best = min(lls, key=lls.get)
                rep.add(f"{uid}: {surf} NL selection == argmin attacker_val log loss over the slate", "selection",
                        sorted(same), best, status="PASS" if best in same else "FAIL", kind="info",
                        note=f"val log loss: {json.dumps({k: round(v, 6) for k, v in lls.items()})}")
            # attacker seeds: deterministic recipes aliased across seeds
            for (s, r, a), k in u.pkeys.items():
                if s == surf and r == "L" and a not in (None, 0) and (surf, "L", 0) in u.pkeys:
                    rep.add(f"{uid}: {surf} L as{a} == as0 (deterministic recipe aliased)", "selection",
                            bool(np.array_equal(pr[k], pr[u.pkeys[(surf, 'L', 0)]])), True, kind="exact")
        # ignore candidates
        for cand, src in (("ignore_rep", "outputs"), ("ignore_out", "rep")):
            ck = u.pkeys.get(("repPLUSoutputs", cand, 0)) or u.pkeys.get(("repPLUSoutputs", cand, None))
            sk = u.pkeys.get((src, "NL", 0)) or u.pkeys.get((src, "NL", None))
            if ck and sk:
                rep.add(f"{uid}: repPLUSoutputs '{cand}' candidate == {src} NL model predictions", "selection",
                        bool(np.allclose(pr[ck], pr[sk], atol=PROB_TOL, rtol=0)), True, kind="exact")
    # outputs-only surface computed once per (dataset, seed) and aliased across methods
    for (d, s), us in by_dsseed.items():
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
        H = ds.forward(m["seed"])[0]
        if m["arm"] in ("B", "C"):
            mk = (m["ds"], m["seed"], m["purpose"], m["attr"], "B" if u.alias_of else m["arm"])
            if mk not in maps:
                rep.add(f"{uid}: G1 re-derivation", "estimates", None, None, status="SKIPPED", kind="info",
                        note="no saved map found")
                continue
            _, d, _, _ = maps[mk]
            H = leace_apply(H, np.asarray(d["proj_left"]), np.asarray(d["proj_right"]), map_bias(d))
        elif m["arm"] == "D":
            rid = ds.forward(m["seed"])[2]
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
    """Arm A native check: historical in-sample one-hot ridge R^2 on the test split (<= 0.05)."""
    out = {}
    for name, ds in dsets.items():
        test = ds.split == "test"
        for s in sorted(ds.encoders):
            H = ds.forward(s)[0][test]
            y = ds.label(ds.cell["target"])[test]
            Y = np.eye(int(y.max()) + 1)[y]
            r32 = ridge_onehot_r2_insample(H, Y, float32_gram=True)
            r64 = ridge_onehot_r2_insample(H, Y)
            out[f"{name}__s{s}"] = {"N0_float32": r32, "N0_float64": r64, "historical_pass": r32 <= TAU}
            rep.add(f"{name}__s{s}: arm A native check, in-sample one-hot ridge R^2 on the test split <= {TAU}",
                    "native", None, {"N0_float32": r32, "N0_float64": r64}, status="INFO", kind="info",
                    note="the historical certificate as reproduced; its pass/fail is a reported result")
            if historical and (name, s) in historical:
                rep.add(f"{name}__s{s}: N0 reproduces dominant_axis_audit r2_onehot", "native",
                        historical[(name, s)], max(0.0, r32), tol=1e-5)
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
            return grp == "C" and bool(cs) and all(u.alias_of for u in cs)
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
            eps.append({"id": f"{fam}[{name},{arm}]", "family": fam, "dataset": name, "arm": arm, "group": g,
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


# ---------------------------------------------------------------- output
DIAGNOSES = []   # (regex on check, cause) -- filled only after a discrepancy has been investigated


def public_summary(items):
    by_status, by_scope = {}, {}
    for it in items:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1
        sc = by_scope.setdefault(it["scope"], {})
        sc[it["status"]] = sc.get(it["status"], 0) + 1
    residual = []
    for it in items:
        if it["status"] not in ("PASS", "INFO"):
            cause = next((c for rx, c in DIAGNOSES if re.search(rx, it["check"])), None)
            residual.append({"check": it["check"], "scope": it["scope"], "status": it["status"],
                             "runner_value": it["runner_value"] if not isinstance(it["runner_value"], list) else "list",
                             "replay_value": it["replay_value"] if not isinstance(it["replay_value"], (list, dict)) else "structured",
                             "abs_diff": it["abs_diff"], "tolerance": it["tolerance"],
                             "cause": cause or it.get("note") or "undiagnosed"})
    return {"n_items": len(items), "by_status": by_status, "by_scope": by_scope, "non_pass_items": residual,
            "contains_per_person_values": False}


def finish(rep, results, out_path, results_path, t0):
    summ = {}
    for it in rep.items:
        summ[it["status"]] = summ.get(it["status"], 0) + 1
    payload = {"schema": "pcrl.matched_removal_benchmark.independent_verification/v1",
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "implementation": "results/combined_matched_removal_benchmark_v1/verification/replay_bench.py "
                                 "(numpy only; independent of stored_model_eval)",
               "tolerances": {"point": POINT_TOL, "bound": f"{BOUND_SE_MULT} x replay MC SE + {BOUND_FLOOR}",
                              "g1_rederive": G1_REDERIVE_TOL, "leace_rederive_rel": LEACE_REDERIVE_TOL,
                              "leace_mean_rel": LEACE_MEAN_TOL,
                              "leace_native": "||W P Sigma_xz||_2 <= svd_tol*(1+1e-9)+1e-12",
                              "transform_rel": TRANSFORM_TOL, "alias": ALIAS_TOL},
               "summary": summ, "public_summary": public_summary(rep.items),
               "elapsed_s": round(time.time() - t0, 1), "items": rep.items}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(_jsonable(payload), indent=1))
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(json.dumps(_jsonable(results), indent=1))
    return payload, results


def load_historical(paths):
    """{(dataset, seed): r2_onehot of the target attribute for the cell purpose} from dominant_axis_audit.json."""
    out = {}
    for ds, p in paths.items():
        d = json.loads(Path(p).read_text())
        per = d.get("per_seed", {"0": d})
        for s, blk in per.items():
            for r in blk.get("rows", []):
                if r.get("purpose") == CELLS[ds]["purpose"] and r.get("attribute") == CELLS[ds]["target"]:
                    out[(ds, int(s))] = float(r["r2_onehot"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/PCRL_eval_cache_private/bench_v1")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--b-prim", type=int, default=B_PRIM)
    ap.add_argument("--b-expl", type=int, default=B_EXPL)
    ap.add_argument("--expl-scope", default="summaries", choices=("summaries", "all"))
    a = ap.parse_args(argv)
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

#!/opt/homebrew/bin/python3
"""Independent replay of the CELL-A stored-model pilot (ROLE 3).

Written from notes/FROZEN_DESIGN.md only.  It does not import or read the
runner (stored_model_eval).  It never refits an attacker and never tunes on
assessment rows.  Everything it reports is recomputed from

  * the input labels (labels.npz; task labels from task_labels_v1.npz, keys y_task_<purpose>) and manifests,
  * the runner's saved predictions run_v1/units/<unit>/preds.npz,
  * the closed-form native quantities (N0, N1, G1 re-derivation), which are a
    deterministic ridge solve on the stored representations and need no fitting
    choices.

Dependencies: numpy, scipy (optional), standard library.  No torch, no repo imports.

Usage (real run):
  /opt/homebrew/bin/python3 replay.py \
      --inputs-dir ~/PCRL_eval_cache_private/pilot_adult_s0 \
      --run-dir   ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1 \
      --report-dir results/combined_stored_model_pilot_v1 \
      --out results/combined_stored_model_pilot_v1/INDEPENDENT_VERIFICATION.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- frozen constants
ROLE_SALT = "pilot-roles-v1|"
ROLE_CUTS = (("attacker_fit", 0.50), ("attacker_val", 0.65), ("assessment", 1.0))
SUPPORT_MIN = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
EXPECTED_ROLE_COUNTS = {"attacker_fit": 7571, "attacker_val": 2239, "assessment": 5250}
UNTREATED_PAIRS = (
    ("income_prediction", "race"), ("income_prediction", "sex"),
    ("employment_analysis", "race"), ("employment_analysis", "age_group"),
    ("employment_analysis", "marital_status"),
    ("education_assessment", "sex"), ("education_assessment", "race"),
    ("education_assessment", "income"),
)
NOISE_PAIR = ("income_prediction", "sex")
NOISE_P = 0
SIGMAS = ("0.25", "0.5", "1", "2", "4", "8")
SEEDS = (0, 1, 2)
PURPOSE_REP = {"income_prediction": "rep_p0", "employment_analysis": "rep_p1",
               "education_assessment": "rep_p2"}
PURPOSE_TASK_DEFAULT = {"income_prediction": "income"}
TAU = 0.05
TAU_GRID = (0.01, 0.02, 0.05, 0.10)
BAR = 0.55
SECONDARY_BARS = (0.52, 0.60)
FAMILY_SIZE = 16
ALPHA_FAMILY = 0.05 / FAMILY_SIZE
B_EXPL, SEED_EXPL, LEVEL_EXPL = 2000, 20261002, 0.90
B_PRIM, SEED_PRIM = 20000, 20261003
LAMBDA = 1e-6
EPS64 = float(np.finfo(np.float64).eps)
EPS32 = float(np.finfo(np.float32).eps)
# log-loss clipping: the locked EFFECTIVE_PROTOCOL.json fixes log_loss_clip = prob_clip = 1e-12 (primary).
# Other conventions are computed as named sensitivity values only; a runner value that matches only a
# non-frozen convention is a FAIL.
LL_CLIP = 1e-12
LL_VARIANTS = (("", LL_CLIP), ("_eps64", EPS64), ("_eps32", EPS32), ("_clip1e-15", 1e-15))


def ll_rows(ptrue, eps):
    return -np.log(np.clip(np.asarray(ptrue, dtype=np.float64), eps, 1.0 - eps))
POINT_TOL = 1e-9

NL_SELECTED_NAMES = {"nl", "nl_selected", "nlsel", "nl_sel", "nlselected", "selected_nl", "nl-selected"}


# ---------------------------------------------------------------- unit identifiers
def expected_unit_ids():
    ids = [f"{p}__{a}" for p, a in UNTREATED_PAIRS]
    for s in SIGMAS:
        for k in SEEDS:
            ids.append(f"{NOISE_PAIR[0]}__{NOISE_PAIR[1]}__p{NOISE_P}_sigma{s}_seed{k}")
    return ids


UNIT_RE = re.compile(
    r"^(?:manifest_)?(?P<purpose>[a-z]+_[a-z]+)__(?P<attr>[a-z_]+?)"
    r"(?:__p(?P<p>\d)_sigma(?P<sigma>[0-9.]+)_seed(?P<seed>\d+))?(?:\.json)?$")


def parse_unit_id(uid):
    m = UNIT_RE.match(uid)
    if not m:
        return None
    d = m.groupdict()
    return {"purpose": d["purpose"], "attr": d["attr"],
            "kind": "noise" if d["sigma"] is not None else "untreated",
            "sigma": d["sigma"], "seed": int(d["seed"]) if d["seed"] is not None else None}


def norm_name(s):
    """Normalise an identifier for matching runner tables (case, separators)."""
    s = str(s).strip().lower().replace("manifest_", "")
    s = re.sub(r"[\s/:|]+", "__", s)
    return s


# ---------------------------------------------------------------- roles and support
def role_of_key(record_key):
    u = int(hashlib.sha256((ROLE_SALT + str(record_key)).encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32
    for name, cut in ROLE_CUTS:
        if u < cut:
            return name
    return ROLE_CUTS[-1][0]


def recompute_roles(record_keys):
    return np.array([role_of_key(k) for k in record_keys])


def support_from_counts(y, roles):
    classes = sorted(int(c) for c in np.unique(y))
    counts = {int(c): {r: int(np.sum((y == c) & (roles == r))) for r in SUPPORT_MIN} for c in classes}
    supported, reasons = [], {}
    for c in classes:
        bad = [f"{r}<{SUPPORT_MIN[r]}" for r in SUPPORT_MIN if counts[c][r] < SUPPORT_MIN[r]]
        if bad:
            reasons[c] = bad
        else:
            supported.append(c)
    pairs = [list(p) for p in combinations(supported, 2)]
    return {"classes": classes, "counts": counts, "supported_classes": supported,
            "supported_pairs": pairs, "unsupported_reasons": reasons,
            "estimable": len(supported) >= 2}


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
    cls = _find_list(js, {"supported_classes", "classes_supported", "supported"})
    prs = _find_list(js, {"supported_pairs", "pairs_supported", "pairs"})
    out = {}
    if cls is not None:
        out["supported_classes"] = sorted(int(float(c)) for c in cls)
    if prs is not None:
        out["supported_pairs"] = sorted(sorted(int(float(x)) for x in p) for p in prs)
    return out


# ---------------------------------------------------------------- weighted statistics
class WAUC:
    """Weighted Mann-Whitney AUC with ties averaged; weights = cluster multiplicities."""

    def __init__(self, score, pos, rows=None):
        score = np.asarray(score, dtype=np.float64)
        idx = np.arange(len(score)) if rows is None else np.flatnonzero(rows)
        s = score[idx]
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        self.cols = idx[order]
        self.pos = np.asarray(pos)[idx][order].astype(np.float64)
        self.neg = 1.0 - self.pos
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])

    def __call__(self, W):
        Ws = W[:, self.cols]
        Pw = np.add.reduceat(Ws * self.pos, self.starts, axis=1)
        Nw = np.add.reduceat(Ws * self.neg, self.starts, axis=1)
        below = np.cumsum(Nw, axis=1) - Nw
        num = (Pw * (below + 0.5 * Nw)).sum(axis=1)
        den = Pw.sum(axis=1) * Nw.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def wmean(x):
    x = np.asarray(x, dtype=np.float64)
    return lambda ctx: (ctx.W @ x) / ctx.W.sum(axis=1)


def wratio_skill(model_loss, ref_loss):
    a = np.asarray(model_loss, dtype=np.float64)
    b = np.asarray(ref_loss, dtype=np.float64)
    return lambda ctx: 1.0 - (ctx.W @ a) / (ctx.W @ b)


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


# ---------------------------------------------------------------- closed-form native R²
def ridge_onehot_r2(H_fit, Y_fit, H_score=None, Y_score=None, lam=LAMBDA, float32_gram=False,
                    return_pred=False):
    """Unnormalised centred-Gram ridge, one-hot pooled R².

    In-sample (H_score None): SS_tot around the fit means (= historical check).
    Held out: SS_tot around the fit means, unclamped.
    float32_gram=True mimics the historical float32 path: centre and form H^T H in float32.
    """
    if float32_gram:
        Hf = np.asarray(H_fit, dtype=np.float32)
        mu = Hf.mean(axis=0, keepdims=True)
        Hc = Hf - mu
        gram = (Hc.T @ Hc) + lam * np.eye(Hc.shape[1])
        cross = Hc.T @ (Y_fit - Y_fit.mean(axis=0, keepdims=True))
    else:
        Hf = np.asarray(H_fit, dtype=np.float64)
        mu = Hf.mean(axis=0, keepdims=True)
        Hc = Hf - mu
        gram = Hc.T @ Hc + lam * np.eye(Hc.shape[1])
        cross = Hc.T @ (Y_fit - Y_fit.mean(axis=0, keepdims=True))
    Wst = np.linalg.solve(np.asarray(gram, dtype=np.float64), np.asarray(cross, dtype=np.float64))
    ybar = Y_fit.mean(axis=0, keepdims=True)
    if H_score is None:
        H_score, Y_score = H_fit, Y_fit
        Hs = Hc
    else:
        Hs = np.asarray(H_score, dtype=np.float32 if float32_gram else np.float64) - mu
    pred = Hs @ Wst + ybar
    ss_res = float(np.sum((Y_score - pred) ** 2))
    ss_tot = float(np.sum((Y_score - ybar) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    if return_pred:
        return r2, pred, ybar.ravel()
    return r2


def onehot(y, classes):
    classes = list(classes)
    Y = np.zeros((len(y), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(y):
        j = pos.get(int(v))
        if j is not None:
            Y[i, j] = 1.0
    return Y


# ---------------------------------------------------------------- bootstrap engine
def cluster_index(units):
    uniq, inv = np.unique(units, return_inverse=True)
    return inv, len(uniq)


def run_bootstrap(stats, inv, n_units, B, seed, batch=200):
    """Cluster bootstrap over assessment units, predictors held fixed.

    Each replicate draws n_units units with replacement; each row gets the
    multiplicity of its unit.  All statistics share the replicate weights.
    """
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
    """MC standard error of an empirical p-quantile via order-statistic spread."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    B = len(x)
    d = math.sqrt(p * (1 - p) / B)
    lo, hi = max(p - 2 * d, 0.0), min(p + 2 * d, 1.0)
    slope = (np.quantile(x, hi, method="linear") - np.quantile(x, lo, method="linear")) / (hi - lo)
    return float(slope * d)


def interval(x, lo_p, hi_p):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {"lower": float(np.quantile(x, lo_p, method="linear")), "upper": float(np.quantile(x, hi_p, method="linear")),
            "mcse_lower": mc_se_quantile(x, lo_p), "mcse_upper": mc_se_quantile(x, hi_p),
            "n_finite": int(len(x))}


# ---------------------------------------------------------------- unit loading
def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def canon_surface(s):
    t = s.lower().replace("-", "_")
    if "rep" in t and "out" in t:
        return "rep+outputs"
    if t.startswith("rep"):
        return "rep"
    if t.startswith("out"):
        return "outputs"
    return s


def parse_pkey(k):
    parts = k.split("__")
    if parts[0] != "P" or len(parts) < 3:
        return None
    return canon_surface(parts[1]), "__".join(parts[2:])


def is_nl_selected(recipe):
    return recipe.lower() in NL_SELECTED_NAMES


def find_task_labels(lab, purpose):
    cands = [f"y_task_{purpose}", f"task_{purpose}", f"task__{purpose}", f"{purpose}_task",
             f"task_label_{purpose}", f"task_labels_{purpose}", purpose]
    for c in cands:
        if c in lab:
            return c, lab[c]
    d = PURPOSE_TASK_DEFAULT.get(purpose)
    if d and d in lab:
        return d, lab[d]
    return None, None


class Report:
    def __init__(self):
        self.items = []

    def add(self, check, scope, runner, replay, tol=None, status=None, note=None, kind="value"):
        diff = None
        if isinstance(runner, (int, float)) and isinstance(replay, (int, float)) \
                and runner is not None and replay is not None \
                and np.isfinite(runner) and np.isfinite(replay):
            diff = abs(float(runner) - float(replay))
        if status is None:
            if kind == "value":
                if diff is None:
                    status = "PASS" if (runner == replay or (runner is None and replay is None)) else "FAIL"
                else:
                    status = "PASS" if diff <= (tol if tol is not None else POINT_TOL) else "FAIL"
            else:
                status = "PASS" if runner == replay else "FAIL"
        it = {"check": check, "scope": scope, "runner_value": _jsonable(runner),
              "replay_value": _jsonable(replay), "abs_diff": diff, "tolerance": tol, "status": status}
        if note:
            it["note"] = note
        self.items.append(it)
        return it


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    return v


# ---------------------------------------------------------------- per-unit statistics
class Unit:
    def __init__(self, uid, path, meta):
        self.uid = uid
        self.path = Path(path)
        self.meta = meta
        self.preds = load_npz(self.path / "preds.npz")
        sp = self.path / "supported.json"
        self.runner_supported_raw = json.loads(sp.read_text()) if sp.exists() else None
        self.pkeys = {}
        for k in self.preds:
            pk = parse_pkey(k)
            if pk:
                self.pkeys[pk] = k


def class_columns(ncols, fit_classes, all_classes):
    # Saved format (coordinator, 2026-10-02): columns are class labels 0..K-1 (full K columns).
    if ncols >= max(all_classes) + 1:
        return list(range(ncols))
    if ncols == len(fit_classes):
        return list(fit_classes)
    if ncols == len(all_classes):
        return list(all_classes)
    raise ValueError(f"cannot map {ncols} probability columns to classes {all_classes}")


def build_unit_stats(u, y, yt, sup, prior_vec_by_cols, stats, task_info=None):
    """Register lazily-evaluated weighted statistics for one unit.  Names: '<uid>|<quantity>'.

    y / yt are the sensitive and task labels taken from the input labels file (not from preds.npz),
    aligned to the saved assessment row order.
    """
    y = np.asarray(y).astype(int)
    supc = sup["supported_classes"]
    supp = sup["supported_pairs"]
    est = sup["estimable"]
    uid = u.uid

    def reg(name, fn):
        stats[f"{uid}|{name}"] = fn

    def auc_block(tag, P, cols):
        """macro / worst-class / worst-pair over supported classes; P columns = cols."""
        if not est:
            return
        colpos = {c: i for i, c in enumerate(cols)}
        class_fns = {}
        for c in supc:
            if c not in colpos:
                continue
            fn = WAUC(P[:, colpos[c]], y == c)
            key = f"{uid}|{tag}|aucc{c}"
            class_fns[c] = (key, fn)
        pair_fns = {}
        for j, k in supp:
            if j not in colpos or k not in colpos:
                continue
            # protocol orientation (EFFECTIVE_PROTOCOL recovery_metrics.worst_pair): pair (i, j), i < j,
            # score p_j / (p_i + p_j), positives = class j.  Mathematically symmetric, but floating-point
            # rounding of 1 - s changes ties, so the frozen orientation is used.
            pj, pk = P[:, colpos[j]].astype(np.float64), P[:, colpos[k]].astype(np.float64)
            den = pj + pk
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = np.where(den > 0, pk / np.where(den > 0, den, 1.0), 0.5)
            rows = (y == j) | (y == k)
            pair_fns[(j, k)] = (f"{uid}|{tag}|aucp{j}_{k}", WAUC(sc, y == k, rows))

        def get_classes(ctx):
            return np.stack([ctx.get(kk, ff) for kk, ff in class_fns.values()], axis=0)

        def get_pairs(ctx):
            return np.stack([ctx.get(kk, ff) for kk, ff in pair_fns.values()], axis=0)

        for c, (kk, ff) in class_fns.items():
            reg(f"{tag}|AUC_class_{c}", (lambda kk=kk, ff=ff: lambda ctx: ctx.get(kk, ff))())
        reg(f"{tag}|AUC_macro", lambda ctx: get_classes(ctx).mean(axis=0))
        # 'worst' = worst case for the guarantee = most recoverable class/pair (max AUC). This reading was
        # fixed after the real run showed the runner uses it; the minimum is kept as AUC_min_class/pair.
        reg(f"{tag}|AUC_worst_class", lambda ctx: get_classes(ctx).max(axis=0))
        reg(f"{tag}|AUC_min_class", lambda ctx: get_classes(ctx).min(axis=0))
        if pair_fns:
            reg(f"{tag}|AUC_worst_pair", lambda ctx: get_pairs(ctx).max(axis=0))
            reg(f"{tag}|AUC_min_pair", lambda ctx: get_pairs(ctx).min(axis=0))

    def prob_scores(tag, P, cols, prior):
        colpos = {c: i for i, c in enumerate(cols)}
        idx = np.array([colpos.get(int(v), -1) for v in y])
        P = np.asarray(P, dtype=np.float64)
        ptrue = np.where(idx >= 0, P[np.arange(len(y)), np.clip(idx, 0, None)], 0.0)
        prior_true = np.where(idx >= 0, prior[np.clip(idx, 0, None)], 0.0)
        for suf, eps in LL_VARIANTS:   # D1 #2: LLR_nats = LL0 - LL, LL_skill = 1 - LL/LL0, LL0 = fit prior
            l_model, l_prior = ll_rows(ptrue, eps), ll_rows(prior_true, eps)
            reg(f"{tag}|logloss{suf}", wmean(l_model))
            reg(f"{tag}|LL_skill{suf}", wratio_skill(l_model, l_prior))
            reg(f"{tag}|LLR_nats{suf}", wmean(l_prior - l_model))
        Y = onehot(y, cols)
        b_model = ((P - Y) ** 2).sum(axis=1)
        b_prior = ((prior[None, :] - Y) ** 2).sum(axis=1)
        reg(f"{tag}|brier_skill", wratio_skill(b_model, b_prior))

    fit_cols_prior = prior_vec_by_cols
    # attacker probability matrices
    for (surface, recipe), key in u.pkeys.items():
        P = u.preds[key]
        cols = class_columns(P.shape[1], *fit_cols_prior["classes_info"])
        prior = fit_cols_prior["prior_for"](cols)
        tag = f"P|{surface}|{recipe}"
        auc_block(tag, P, cols)
        prob_scores(tag, P, cols, prior)

    # G1 / G2 held-out R² from saved one-hot predictions, SS_tot around saved priors, unclamped
    for g in ("G1", "G2"):
        if f"{g}_pred" in u.preds and f"{g}_prior" in u.preds:
            pred = np.asarray(u.preds[f"{g}_pred"], dtype=np.float64)
            pri = np.asarray(u.preds[f"{g}_prior"], dtype=np.float64).ravel()
            cols = class_columns(pred.shape[1], *fit_cols_prior["classes_info"])
            Y = onehot(y, cols)
            res = ((Y - pred) ** 2).sum(axis=1)
            tot = ((Y - pri[None, :]) ** 2).sum(axis=1)
            reg(f"{g}|R2", (lambda res=res, tot=tot: lambda ctx: 1.0 - (ctx.W @ res) / (ctx.W @ tot))())
            if g == "G1":
                auc_block("G1pred", pred, cols)   # pure metric contrast F2
    # held-out rho_1^2 from saved per-row canonical variates (D1 #6): squared weighted correlation
    if "RHO_u" in u.preds and "RHO_v" in u.preds:
        ru = np.asarray(u.preds["RHO_u"], dtype=np.float64).ravel()
        rv = np.asarray(u.preds["RHO_v"], dtype=np.float64).ravel()
        ru, rv = ru - ru.mean(), rv - rv.mean()

        def rho2(ctx, ru=ru, rv=rv):
            W = ctx.W
            sw = W.sum(axis=1)
            mu, mv = (W @ ru) / sw, (W @ rv) / sw
            cuv = (W @ (ru * rv)) / sw - mu * mv
            cuu = (W @ (ru * ru)) / sw - mu * mu
            cvv = (W @ (rv * rv)) / sw - mv * mv
            return cuv * cuv / (cuu * cvv)
        reg("RHO1SQ_heldout", rho2)
    # label-only reference
    if "LO_P" in u.preds:
        P = np.asarray(u.preds["LO_P"], dtype=np.float64)
        cols = class_columns(P.shape[1], *fit_cols_prior["classes_info"])
        auc_block("LO", P, cols)
        prob_scores("LO", P, cols, fit_cols_prior["prior_for"](cols))
    # utility
    if yt is not None:
        yt = np.asarray(yt).astype(int)
        if (task_info or {}).get("majority_fit") is not None:
            reg("Uconst|accuracy", wmean((yt == task_info["majority_fit"]).astype(np.float64)))
        for name, arr, is_logit in (("U1", u.preds.get("U1_logits"), True), ("U2", u.preds.get("U2_P"), False)):
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                arr = np.stack([-arr / 2, arr / 2], axis=1) if is_logit else np.stack([1 - arr, arr], axis=1)
            prob = np.exp(log_softmax(arr)) if is_logit else arr
            ncls = arr.shape[1]
            ok = (yt >= 0) & (yt < ncls)
            ptrue = np.where(ok, prob[np.arange(len(yt)), np.clip(yt, 0, ncls - 1)], 0.0)
            pred_lab = prob.argmax(axis=1)
            acc = (pred_lab == yt).astype(np.float64)
            reg(f"{name}|accuracy", wmean(acc))
            ti = task_info or {}
            tsup = [c for c in ti.get("supported", []) if c < ncls]
            if tsup:   # D1 9f: macro-F1 and per-class utility over task classes meeting 100/30/100
                parts = []
                for c in tsup:
                    tp = ((pred_lab == c) & (yt == c)).astype(float)
                    fp = ((pred_lab == c) & (yt != c)).astype(float)
                    fn_ = ((pred_lab != c) & (yt == c)).astype(float)
                    parts.append((tp, fp, fn_))

                def macro_f1(ctx, parts=parts):
                    W = ctx.W
                    f1s = []
                    for tp, fp, fn_ in parts:
                        a_, b_, c_ = W @ tp, W @ fp, W @ fn_
                        d_ = 2 * a_ + b_ + c_
                        f1s.append(np.where(d_ > 0, 2 * a_ / np.where(d_ > 0, d_, 1.0), 0.0))
                    return np.mean(f1s, axis=0)
                reg(f"{name}|macro_F1", macro_f1)
            if ti.get("majority_fit") is not None:   # D1 9g: lift over the constant (fit-majority) predictor
                const = (yt == ti["majority_fit"]).astype(np.float64)
                reg(f"{name}|lift_over_constant", wmean(acc - const))
            for suf, eps in LL_VARIANTS:
                reg(f"{name}|logloss{suf}", wmean(ll_rows(ptrue, eps)))
            if is_logit:   # exact cross-entropy from logits, no clipping
                lsm = log_softmax(arr)
                reg(f"{name}|logloss_unclipped",
                    wmean(-np.where(ok, lsm[np.arange(len(yt)), np.clip(yt, 0, ncls - 1)], np.log(EPS64))))
            # macro OvR over supported task classes (binary included: mean of both class AUCs, which differs
            # from the p1-AUC only where the softmax saturates); variant: all classes present
            present = [c for c in range(ncls) if 0 < np.sum(yt == c) < len(yt)]
            fs_all = {c: (f"{uid}|{name}|uauc{c}", WAUC(prob[:, c], yt == c)) for c in present}
            fs_sup = [fs_all[c] for c in tsup if c in fs_all] if tsup else list(fs_all.values())
            reg(f"{name}|AUC", (lambda fs=fs_sup: lambda ctx: np.mean([ctx.get(k, f) for k, f in fs], axis=0))())
            reg(f"{name}|AUC_allpresent",
                (lambda fs=list(fs_all.values()): lambda ctx: np.mean([ctx.get(k, f) for k, f in fs], axis=0))())
    return stats


def _sha_pairs(obj, prefix=""):
    """Yield (relative path, sha256) pairs from a COMPLETE.json-like structure of unknown shape."""
    hexre = re.compile(r"^[0-9a-f]{64}$")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and hexre.match(v) and k not in ("sha256",):
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_complete(unit_dir, uid, rep):
    f = Path(unit_dir) / "COMPLETE.json"
    if not f.exists():
        rep.add(f"{uid}: COMPLETE.json present", "ids_roles", False, True, kind="exact")
        return
    pairs = list(_sha_pairs(json.loads(f.read_text())))
    bad, missing = [], []
    for rel, sha in pairs:
        p = Path(rel) if Path(rel).is_absolute() else Path(unit_dir) / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != sha:
            bad.append(rel)
    rep.add(f"{uid}: COMPLETE.json sha256s match files", "ids_roles", len(bad) + len(missing), 0, kind="exact",
            note=f"{len(pairs)} listed; mismatched={bad} missing={missing}")
    if not any(Path(r).name == "preds.npz" for r, _ in pairs):
        rep.add(f"{uid}: COMPLETE.json covers preds.npz", "ids_roles", False, True, kind="exact")


_SHA_CACHE = {}


def find_manifest(uid, dirs):
    """manifest_<uid>.json, or any *.json whose name, after dropping a prefix ending just before the
    purpose name, equals <uid> exactly (covers v2 manifest prefixes)."""
    strip = re.compile(r"^.*?(?=(income_prediction|employment_analysis|education_assessment)__)")
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        if (d / f"manifest_{uid}.json").exists():
            return d / f"manifest_{uid}.json"
        hits = [m for m in sorted(d.glob("*.json")) if strip.sub("", m.stem) == uid]
        if hits:
            return hits[0]
    return None


def assign_categories(n0_f32, n0_f64, p1_dec, p2_dec, outputs_lower):
    """Category rules (FROZEN_DESIGN + Addendum D1 #8).

    C1: reproduced N0 (historical mixed precision) > tau; otherwise 'historical_check_passes'.
    C2: P1 'fails to generalise' -- not assigned when C1 holds (the fitting-sample check already fails).
    C3: P2 lower bound > 0.55 (rep), or the outputs-surface bound (same construction) > 0.55.
    C5_NE / C5_UNRESOLVED: any primary endpoint of the unit NE / UNRESOLVED.
    """
    cats, notes = [], []
    c1 = None
    if n0_f32 is not None:
        c1 = n0_f32 > TAU
        cats.append("C1" if c1 else "historical_check_passes")
        if n0_f64 is not None and (n0_f64 > TAU) != c1:
            cats.append("C1_float32_float64_disagree")
    if p1_dec == "FAILS_TO_GENERALISE":
        if c1:
            notes.append("P1 fails to generalise but C1 already holds: not read as C2 (D1 #8)")
        else:
            cats.append("C2")
    if p2_dec == "RECOVERY_OUTSIDE_SCOPE_ESTABLISHED" or (outputs_lower is not None and outputs_lower > BAR):
        cats.append("C3")
        if p2_dec != "RECOVERY_OUTSIDE_SCOPE_ESTABLISHED":
            notes.append("C3 from the outputs surface only")
    decs = [d for d in (p1_dec, p2_dec) if d]
    if "NE" in decs:
        cats.append("C5_NE")
    if "UNRESOLVED" in decs:
        cats.append("C5_UNRESOLVED")
    return cats, notes


# ---------------------------------------------------------------- main replay
def replay(run_dir, inputs_dir, labels_path=None, report_dir=None, out_path=None,
           b_expl=B_EXPL, b_prim=B_PRIM, seed_expl=SEED_EXPL, seed_prim=SEED_PRIM,
           expected_role_counts=EXPECTED_ROLE_COUNTS, historical_json=None, results_path=None,
           closed_form=True, verbose=True, task_labels_path=None):
    t0 = time.time()
    run_dir, inputs_dir = Path(run_dir).expanduser(), Path(inputs_dir).expanduser()
    rep = Report()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- labels and roles
    if labels_path is None:
        cand = [inputs_dir / "labels.npz", inputs_dir / "labels_v2.npz"]
        labels_path = next((c for c in cand if c.exists()), None)
    labels_path = Path(labels_path)
    lab = load_npz(labels_path)
    rep.add("labels file used", "inputs", None, str(labels_path), status="INFO", kind="info")
    roles = recompute_roles(lab["record_key"])
    row_ids = np.asarray(lab["row_id"]).astype(np.int64)
    rid_pos = {int(r): i for i, r in enumerate(row_ids)}
    # task labels (Addendum D1 #5): separate file, keys y_task_<purpose>, aligned by row_id
    tl_cands = [run_dir / "inputs" / "task_labels_v1.npz", inputs_dir / "task_labels_v1.npz"]
    tl_path = Path(task_labels_path) if task_labels_path else next((c for c in tl_cands if c.exists()), tl_cands[0])
    if tl_path.exists():
        tl = load_npz(tl_path)
        if "row_id" in tl:
            tpos = np.array([rid_pos.get(int(r), -1) for r in tl["row_id"]])
            rep.add("task_labels row_id set == labels row_id set", "inputs",
                    int(len(np.setxor1d(tl["row_id"], row_ids))), 0, kind="exact")
        else:
            tpos = np.arange(len(row_ids))
        for k, v in tl.items():
            if k.startswith("y_task_") and len(v) == len(tpos):
                arr = np.full(len(row_ids), -1, dtype=np.int64)
                ok = tpos >= 0
                arr[tpos[ok]] = np.asarray(v)[ok].astype(np.int64)
                lab[k] = arr
        rep.add("task labels file used", "inputs", None, str(tl_path), status="INFO", kind="info")
    else:
        rep.add("task labels file present", "inputs", None, str(tl_path), status="INFO", kind="info",
                note="not found; falling back to keys inside the labels file / preds y_task")
    if "role" in lab:
        rep.add("stored role column equals recomputed hash roles (all rows)", "ids_roles",
                int(np.sum(lab["role"] != roles)), 0, kind="exact")
    counts = {r: int(np.sum(roles == r)) for r, _ in ROLE_CUTS}
    if expected_role_counts:
        for r, n in expected_role_counts.items():
            rep.add(f"role count {r}", "ids_roles", n, counts[r], kind="exact",
                    note="runner value = frozen design count")
    # units constant within record key
    uk = {}
    bad_unit = 0
    for u_, k_ in zip(lab["unit"], lab["record_key"]):
        if uk.setdefault(int(u_), k_) != k_:
            bad_unit += 1
    rep.add("record unit id maps to one record key", "ids_roles", 0, bad_unit, kind="exact")
    expected_assess = np.sort(row_ids[roles == "assessment"])

    # ---- units
    units_dir = run_dir / "units"
    present = {}
    for p in sorted(units_dir.iterdir()) if units_dir.exists() else []:
        if p.is_dir():
            present[norm_name(p.name)] = p
    exp_ids = expected_unit_ids()
    missing = [u for u in exp_ids if u not in present]
    extra = [u for u in present if u not in exp_ids]
    rep.add("exactly the 26 expected unit IDs present", "ids_roles",
            sorted(present), sorted(exp_ids), status="PASS" if not missing and not extra else "FAIL",
            note=f"missing={missing} extra={extra}", kind="ids")

    units = {}
    for uid in exp_ids:
        if uid in present and (present[uid] / "preds.npz").exists():
            units[uid] = Unit(uid, present[uid], parse_unit_id(uid))
        elif uid in present:
            rep.add(f"{uid}: preds.npz present", "ids_roles", False, True, kind="exact")

    sup_by_unit, fitinfo, ylabs, task_infos = {}, {}, {}, {}
    ref_assess = None
    for uid, u in units.items():
        m = u.meta
        pr = u.preds
        arid = np.asarray(pr.get("assess_row_id", np.array([], dtype=np.int64))).astype(np.int64)
        # ID checks
        rep.add(f"{uid}: assess_row_id set == recomputed assessment role", "ids_roles",
                int(len(np.setxor1d(arid, expected_assess))), 0, kind="exact",
                note=f"n_saved={len(arid)} n_expected={len(expected_assess)}")
        rep.add(f"{uid}: assess_row_id unique", "ids_roles", int(len(arid) - len(np.unique(arid))), 0, kind="exact")
        if ref_assess is None:
            ref_assess = arid
        else:
            same = len(arid) == len(ref_assess) and bool(np.all(arid == ref_assess))
            rep.add(f"{uid}: assessment rows aligned identically with first unit", "ids_roles", same, True, kind="exact")
        idx = np.array([rid_pos.get(int(r), -1) for r in arid])
        okidx = idx >= 0
        rep.add(f"{uid}: all assess_row_id in labels", "ids_roles", int(np.sum(~okidx)), 0, kind="exact")
        idx = idx[okidx]
        if "assess_unit" in pr:
            rep.add(f"{uid}: assess_unit == labels unit", "ids_roles",
                    int(np.sum(np.asarray(pr["assess_unit"])[okidx].astype(np.int64) != lab["unit"][idx].astype(np.int64))),
                    0, kind="exact")
        y_lab = np.asarray(lab[m["attr"]]).astype(int)
        if "y_s" in pr:
            rep.add(f"{uid}: y_s == labels[{m['attr']}]", "ids_roles",
                    int(np.sum(np.asarray(pr["y_s"])[okidx].astype(int) != y_lab[idx])), 0, kind="exact")
        tkey, tlab = find_task_labels(lab, m["purpose"])
        y_al = np.full(len(arid), -1)
        y_al[okidx] = y_lab[idx]
        if tlab is not None:
            t_al = np.full(len(arid), -1)
            t_al[okidx] = np.asarray(tlab).astype(int)[idx]
        else:
            t_al = np.asarray(pr["y_task"]).astype(int) if "y_task" in pr else None
        ylabs[uid] = (y_al, t_al, f"labels key {tkey}" if tlab is not None else "preds.npz y_task (no label key)")
        if tlab is not None:
            ta = np.asarray(tlab).astype(int)
            tsup = support_from_counts(ta, roles)["supported_classes"]
            vals, cnts = np.unique(ta[roles == "attacker_fit"], return_counts=True)
            tsp = support_from_counts(ta, roles)
            task_infos[uid] = {"supported": tsup, "majority_fit": int(vals[np.argmax(cnts)]), "key": tkey,
                               "counts": {str(c): cc for c, cc in tsp["counts"].items()}}
        else:
            task_infos[uid] = {}
        if "y_task" in pr and tlab is not None:
            rep.add(f"{uid}: y_task == labels[{tkey}]", "ids_roles",
                    int(np.sum(np.asarray(pr["y_task"])[okidx].astype(int) != np.asarray(tlab)[idx].astype(int))),
                    0, kind="exact")
        elif "y_task" in pr:
            rep.add(f"{uid}: y_task cross-check", "ids_roles", None, None, status="SKIPPED",
                    note=f"no task-label key for {m['purpose']} in {labels_path.name}", kind="info")
        # support
        sup = support_from_counts(y_lab, roles)
        sup_by_unit[uid] = sup
        rs = parse_runner_supported(u.runner_supported_raw) if u.runner_supported_raw is not None else {}
        rep.add(f"{uid}: supported classes", "support", rs.get("supported_classes"), sup["supported_classes"], kind="exact")
        rep.add(f"{uid}: supported pairs", "support", rs.get("supported_pairs"),
                sorted(sorted(p) for p in sup["supported_pairs"]), kind="exact")
        # fit-role class info, prior (attacker_fit frequencies)
        fit_y = y_lab[roles == "attacker_fit"]
        fit_classes = sorted(int(c) for c in np.unique(fit_y))
        all_classes = sorted(int(c) for c in np.unique(y_lab))

        def prior_for(cols, fit_y=fit_y):
            return np.array([np.mean(fit_y == c) for c in cols], dtype=np.float64)
        fitinfo[uid] = {"classes_info": (fit_classes, all_classes), "prior_for": prior_for}
        for g in ("G1", "G2"):
            if f"{g}_prior" in pr:
                cols = class_columns(len(np.ravel(pr[f"{g}_prior"])), fit_classes, all_classes)
                d = float(np.max(np.abs(np.ravel(pr[f"{g}_prior"]) - prior_for(cols))))
                rep.add(f"{uid}: {g}_prior == attacker_fit one-hot means", "estimates", d, 0.0, tol=1e-9,
                        status="PASS" if d <= 1e-9 else "FAIL", note="runner_value = max abs diff")
        if "s_prior_fit" in pr:
            sp_ = np.ravel(pr["s_prior_fit"]).astype(np.float64)
            d = float(np.max(np.abs(sp_ - prior_for(list(range(len(sp_)))))))
            rep.add(f"{uid}: s_prior_fit == attacker_fit class frequencies", "estimates", d, 0.0, tol=1e-9,
                    status="PASS" if d <= 1e-9 else "FAIL", note="runner_value = max abs diff")
        if "t_prior_fit" in pr and tlab is not None:
            tp_ = np.ravel(pr["t_prior_fit"]).astype(np.float64)
            tf = np.asarray(tlab).astype(int)[roles == "attacker_fit"]
            d = float(np.max(np.abs(tp_ - np.array([np.mean(tf == c) for c in range(len(tp_))]))))
            rep.add(f"{uid}: t_prior_fit == attacker_fit task-class frequencies", "estimates", d, 0.0, tol=1e-9,
                    status="PASS" if d <= 1e-9 else "FAIL", note="runner_value = max abs diff")
        verify_complete(u.path, uid, rep)
        # LO table recompute
        if "LO_P" in pr and tlab is not None and "y_task" in pr:
            t_all = np.asarray(tlab).astype(int)
            fm = roles == "attacker_fit"
            cols = class_columns(pr["LO_P"].shape[1], fit_classes, all_classes)
            K = len(cols)
            yt = np.asarray(pr["y_task"]).astype(int)[okidx]
            mine = np.empty((len(yt), K))
            for t in np.unique(yt):   # Laplace alpha=1: (n_st + 1) / (n_t + K), on attacker_fit
                sel = fm & (t_all == t)
                mine[yt == t] = np.array([(np.sum(y_lab[sel] == c) + 1.0) / (sel.sum() + K) for c in cols])
            d = float(np.max(np.abs(mine - np.asarray(pr["LO_P"], dtype=np.float64)[okidx])))
            rep.add(f"{uid}: LO_P == Laplace(alpha=1) attacker_fit table P(s|y_task)", "estimates", d, 0.0,
                    tol=1e-9, status="PASS" if d <= 1e-9 else "FAIL", note="runner_value = max abs diff")
        # NL selection identity
        for surf in ("rep", "outputs", "rep+outputs"):
            nl = [k for (s, r), k in u.pkeys.items() if s == surf and is_nl_selected(r)]
            comps = {r: k for (s, r), k in u.pkeys.items() if s == surf and r.lower() in ("gbt", "mlp")}
            if nl and comps:
                same = [r for r, k in comps.items() if np.array_equal(pr[k], pr[nl[0]])]
                rep.add(f"{uid}: {surf} NL-selected equals one of GBT/MLP", "estimates", same, same,
                        status="PASS" if len(same) >= 1 else "FAIL", kind="info")
        # untreated U1 = stored clean logits
        if m["kind"] == "untreated" and "U1_logits" in pr:
            for cache in (inputs_dir / "cache").glob("*.npz"):
                c = load_npz(cache)
                key = f"logits_{m['purpose']}"
                if key in c:
                    cpos = {int(r): i for i, r in enumerate(c["row_id"])}
                    ci = np.array([cpos[int(r)] for r in arid[okidx]])
                    d = float(np.max(np.abs(np.asarray(pr["U1_logits"])[okidx] - c[key][ci])))
                    rep.add(f"{uid}: U1_logits == stored clean logits", "estimates", d, 0.0, tol=1e-6,
                            status="PASS" if d <= 1e-6 else "FAIL", note="runner_value = max abs diff")
                    break
    # reused outputs surface on noise units
    ref = units.get(f"{NOISE_PAIR[0]}__{NOISE_PAIR[1]}")
    if ref is not None:
        for uid, u in units.items():
            if u.meta["kind"] != "noise":
                continue
            for (s, r), k in u.pkeys.items():
                if s == "outputs" and (s, r) in ref.pkeys:
                    same = np.array_equal(u.preds[k], ref.preds[ref.pkeys[(s, r)]])
                    rep.add(f"{uid}: outputs/{r} reused from untreated {ref.uid}", "estimates", same, True, kind="exact")

    # ---- closed-form native quantities (inputs only, deterministic)
    native = {}
    if closed_form:
        native = closed_form_native(units, lab, roles, rid_pos, inputs_dir, rep, log,
                                    manifest_dirs=[run_dir / "inputs", inputs_dir])
    hist = None
    if historical_json:
        try:
            hist = load_historical(historical_json)
            rep.add("historical audit loaded", "native", None, len(hist), status="INFO", kind="info")
        except Exception as e:  # noqa: BLE001
            rep.add("historical dominant_axis_audit comparison", "native", None, None, status="SKIPPED",
                    note=repr(e), kind="info")

    # ---- statistics registry
    if ref_assess is None:
        rep.add("any unit loaded", "ids_roles", False, True, kind="exact")
        return finish(rep, {}, out_path, results_path, t0)
    first = next(iter(units.values()))
    fa = np.asarray(first.preds["assess_row_id"]).astype(np.int64)
    inv, n_clusters = cluster_index(np.asarray(lab["unit"]).astype(np.int64)[[rid_pos[int(r)] for r in fa]])
    rep.add("assessment cluster count (unique record units)", "inference",
            None, int(n_clusters), status="INFO", kind="info",
            note=f"rows={len(inv)}; duplicate rows collapse to {len(inv) - n_clusters} fewer units")

    stats = {}
    for uid, u in units.items():
        build_unit_stats(u, ylabs[uid][0], ylabs[uid][1], sup_by_unit[uid], fitinfo[uid], stats, task_infos[uid])
        if not ylabs[uid][2].startswith("labels key y_task_"):
            rep.add(f"{uid}: task label source", "estimates", None, ylabs[uid][2], status="INFO", kind="info")
    # leakage beyond label-only (outputs NL - LO), paired on identical IDs
    for uid, u in units.items():
        nlk = [r for (s, r) in u.pkeys if s == "outputs" and is_nl_selected(r)]
        a, b = (f"{uid}|P|outputs|{nlk[0]}|AUC_macro" if nlk else None), f"{uid}|LO|AUC_macro"
        if a in stats and b in stats:
            fa, fb = stats[a], stats[b]
            stats[f"{uid}|LEAK_outputsNL_minus_LO|AUC"] = (lambda fa=fa, fb=fb: lambda ctx: fa(ctx) - fb(ctx))()
    # decomposition step deltas (paired, same replicate)
    for uid, u in units.items():
        if u.meta["kind"] != "untreated":
            continue

        def _nl(surf, u=u):
            r = [r for (s_, r) in u.pkeys if s_ == surf and is_nl_selected(r)]
            return f"{uid}|P|{surf}|{r[0]}|AUC_macro" if r else None
        lk = [r for (s_, r) in u.pkeys if s_ == "rep" and r.lower() == "l"]
        fk = {"F2": f"{uid}|G1pred|AUC_macro", "F3": f"{uid}|P|rep|{lk[0]}|AUC_macro" if lk else None,
              "F4": _nl("rep"), "F5": _nl("outputs"), "F6": _nl("rep+outputs")}
        for a_, b_ in (("F3", "F2"), ("F4", "F3"), ("F5", "F4"), ("F6", "F4")):
            if fk[a_] in stats and fk[b_] in stats:
                fa, fb = stats[fk[a_]], stats[fk[b_]]
                stats[f"{uid}|DECOMP|{a_}_minus_{b_}"] = (lambda fa=fa, fb=fb: lambda ctx: fa(ctx) - fb(ctx))()
    # utility paired differences vs the untreated reference of the same purpose
    refuid = f"{NOISE_PAIR[0]}__{NOISE_PAIR[1]}"
    for uid, u in units.items():
        if u.meta["kind"] != "noise":
            continue
        for q in ("U1|accuracy", "U1|logloss", "U1|AUC", "U1|macro_F1", "U1|lift_over_constant",
                  "U2|accuracy", "U2|logloss", "U2|AUC", "U2|macro_F1", "U2|lift_over_constant"):
            a, b = f"{uid}|{q}", f"{refuid}|{q}"
            if a in stats and b in stats:
                fa, fb = stats[a], stats[b]
                stats[f"{uid}|DIFF_vs_untreated|{q}"] = (lambda fa=fa, fb=fb: lambda ctx: fa(ctx) - fb(ctx))()
        for pr_ in ("U1", "U2"):   # D1 9g: normalised lift = lift(noise) / lift(clean reference)
            a, b = f"{uid}|{pr_}|lift_over_constant", f"{refuid}|{pr_}|lift_over_constant"
            if a in stats and b in stats:
                fa, fb = stats[a], stats[b]
                stats[f"{uid}|{pr_}|normalised_lift"] = (lambda fa=fa, fb=fb: lambda ctx: fa(ctx) / fb(ctx))()
    # noise-arm seed aggregation (mean over seeds within replicate)
    for s in SIGMAS:
        seeds = [f"{NOISE_PAIR[0]}__{NOISE_PAIR[1]}__p{NOISE_P}_sigma{s}_seed{k}" for k in SEEDS]
        if not all(x in units for x in seeds):
            continue
        qs = None
        for x in seeds:
            q = {k.split("|", 1)[1] for k in stats if k.startswith(x + "|")}
            qs = q if qs is None else qs & q
        for q in sorted(qs):
            fns = [stats[f"{x}|{q}"] for x in seeds]
            stats[f"noise_sigma{s}_seedmean|{q}"] = (lambda fns=fns: lambda ctx: np.mean([f(ctx) for f in fns], axis=0))()

    # point estimates
    n = len(inv)
    pctx = Ctx(np.ones((1, n)))
    point = {k: float(fn(pctx)[0]) for k, fn in stats.items()}
    # seed spread
    spread = {}
    for k in point:
        if k.startswith("noise_sigma"):
            s = k.split("_")[1][5:]
            q = k.split("|", 1)[1]
            vals = [point.get(f"{NOISE_PAIR[0]}__{NOISE_PAIR[1]}__p{NOISE_P}_sigma{s}_seed{j}|{q}") for j in SEEDS]
            if all(v is not None for v in vals):
                spread[k] = {"points": vals, "min": min(vals), "max": max(vals),
                             "sd_ddof1": float(np.std(vals, ddof=1)), "sd_ddof0": float(np.std(vals))}
    lift_flags = {}
    for pr_ in ("U1", "U2"):
        v = point.get(f"{refuid}|{pr_}|lift_over_constant")
        if v is not None:
            lift_flags[pr_] = {"clean_lift": v, "normalised_lift_reportable": bool(v >= 0.03)}
    log(f"[replay] {len(stats)} statistics; point estimates done at {time.time() - t0:.1f}s")

    # ---- primary family
    prim_stats, endpoints = {}, []
    for p, a in UNTREATED_PAIRS:
        uid = f"{p}__{a}"
        endpoints.append(("P1", uid, f"{uid}|G1|R2", TAU))
        rk = None
        if uid in units:
            nl = [r for (s, r) in units[uid].pkeys if s == "rep" and is_nl_selected(r)]
            rk = f"{uid}|P|rep|{nl[0]}|AUC_macro" if nl else None
        endpoints.append(("P2", uid, rk, BAR))
    c3_out = {}
    for p, a in UNTREATED_PAIRS:
        uid = f"{p}__{a}"
        if uid in units:
            nl = [r for (s, r) in units[uid].pkeys if s == "outputs" and is_nl_selected(r)]
            if nl and f"{uid}|P|outputs|{nl[0]}|AUC_macro" in stats:
                c3_out[uid] = f"{uid}|P|outputs|{nl[0]}|AUC_macro"
    for _, uid, k, _ in endpoints:
        if k and k in stats and sup_by_unit.get(uid, {}).get("estimable", False):
            prim_stats[k] = stats[k]
    for k in c3_out.values():
        prim_stats[k] = stats[k]
    log(f"[replay] primary bootstrap B={b_prim} over {len(prim_stats)} statistics ...")
    prim = run_bootstrap(prim_stats, inv, n_clusters, b_prim, seed_prim)
    log(f"[replay] primary done at {time.time() - t0:.1f}s; exploratory B={b_expl} over {len(stats)} ...")
    expl = run_bootstrap(stats, inv, n_clusters, b_expl, seed_expl)
    log(f"[replay] exploratory done at {time.time() - t0:.1f}s")

    a = ALPHA_FAMILY
    lo_e, hi_e = (1 - LEVEL_EXPL) / 2, 1 - (1 - LEVEL_EXPL) / 2
    primary = []
    for fam, uid, k, thr in endpoints:
        name = f"{fam}-{uid}"
        rec = {"endpoint": name, "unit": uid, "statistic": k, "threshold": thr}
        if uid not in units or not sup_by_unit[uid]["estimable"] or k is None or k not in prim:
            rec.update({"decision": "NE", "estimate": point.get(k) if k else None,
                        "reason": "unit not estimable (<2 supported classes)" if uid in sup_by_unit
                        and not sup_by_unit[uid]["estimable"] else "statistic missing"})
            primary.append(rec)
            continue
        iv = interval(prim[k], a, 1 - a)
        rec.update({"estimate": point[k], **iv})
        if fam == "P1":
            rec["decision"] = ("FAILS_TO_GENERALISE" if iv["lower"] > thr else
                               "GENERALISES" if iv["upper"] <= thr else "UNRESOLVED")
        else:
            rec["decision"] = ("RECOVERY_OUTSIDE_SCOPE_ESTABLISHED" if iv["lower"] > thr else
                               "BELOW_BAR_ESTABLISHED" if iv["upper"] < thr else "UNRESOLVED")
        rec["mc_borderline"] = bool(min(abs(iv["lower"] - thr), abs(iv["upper"] - thr))
                                    <= 5 * max(iv["mcse_lower"], iv["mcse_upper"]) + 5e-5)
        primary.append(rec)

    # categories per untreated unit
    categories = {}
    pmap = {r["endpoint"]: r for r in primary}
    for p, a_ in UNTREATED_PAIRS:
        uid = f"{p}__{a_}"
        nv = native.get(uid, {})
        p1, p2 = pmap.get(f"P1-{uid}"), pmap.get(f"P2-{uid}")
        c3o = None
        if uid in c3_out:
            ivo = interval(prim[c3_out[uid]], ALPHA_FAMILY, 1 - ALPHA_FAMILY)
            c3o = {"estimate": point[c3_out[uid]], **ivo, "established_above": ivo["lower"] > BAR}
        cats, notes = assign_categories(nv.get("N0_float32"), nv.get("N0_float64"),
                                        p1["decision"] if p1 else None, p2["decision"] if p2 else None,
                                        c3o["lower"] if c3o else None)
        categories[uid] = {"categories": cats, "notes": notes, "outputs_surface_bound": c3o}

    # tau sensitivity for G1 (D1 9e): exploratory 90% intervals, tau in {0.01,0.02,0.05,0.10}
    tau_sens = {}
    for uid in units:
        k = f"{uid}|G1|R2"
        if k in expl:
            iv = interval(expl[k], lo_e, hi_e)
            tau_sens[uid] = {str(t): ("FAILS_TO_GENERALISE" if iv["lower"] > t else
                                      "GENERALISES" if iv["upper"] <= t else "UNRESOLVED")
                             for t in TAU_GRID}

    # decomposition
    decomposition = {}
    for p, a_ in UNTREATED_PAIRS:
        uid = f"{p}__{a_}"
        if uid not in units:
            continue
        u = units[uid]

        def nlk(surf):
            r = [r for (s, r) in u.pkeys if s == surf and is_nl_selected(r)]
            return f"{uid}|P|{surf}|{r[0]}|AUC_macro" if r else None
        lk = [r for (s, r) in u.pkeys if s == "rep" and r.lower() in ("l", "linear", "lr", "logreg")]
        keys = {"F0": None, "F1": f"{uid}|G1|R2", "F2": f"{uid}|G1pred|AUC_macro",
                "F3": f"{uid}|P|rep|{lk[0]}|AUC_macro" if lk else None,
                "F4": nlk("rep"), "F5": nlk("outputs"), "F6": nlk("rep+outputs")}
        row = {}
        for f, k in keys.items():
            if f == "F0":
                row["F0"] = native.get(uid, {}).get("N0_float32")
                row["F0_float64"] = native.get(uid, {}).get("N0_float64")
                continue
            row[f] = point.get(k) if k else None
            row[f + "_key"] = k
            if k and k in expl:
                iv = interval(expl[k], lo_e, hi_e)
                row[f + "_ci90"] = [iv["lower"], iv["upper"]] if iv else None
        decomposition[uid] = row

    exploratory = {}
    for k in stats:
        iv = interval(expl[k], lo_e, hi_e)
        exploratory[k] = {"estimate": point[k], **(iv or {})}
        if k.endswith("AUC_macro"):
            exploratory[k]["vs_bars"] = {str(b): ("above" if iv and iv["lower"] > b else
                                                  "below" if iv and iv["upper"] < b else "unresolved")
                                         for b in (*SECONDARY_BARS, BAR)}

    results = {"primary": primary, "categories": categories, "decomposition": decomposition,
               "tau_sensitivity_G1": tau_sens, "normalised_lift_flags": lift_flags,
               "exploratory": exploratory, "seed_spread": spread, "native": native,
               "support": {u: {k: (v if k != "counts" else {str(c): cc for c, cc in v.items()})
                               for k, v in s.items() if k != "unsupported_reasons"}
                           | {"unsupported_reasons": {str(c): r for c, r in s["unsupported_reasons"].items()}}
                           for u, s in sup_by_unit.items()},
               "support_task": {u: {"supported_classes": t.get("supported"), "counts": t.get("counts")}
                                for u, t in task_infos.items()},
               "config": {"B_expl": b_expl, "seed_expl": seed_expl, "B_prim": b_prim, "seed_prim": seed_prim,
                          "alpha_family": ALPHA_FAMILY, "rng": "numpy default_rng (PCG64), unit draws batched 200",
                          "n_assessment_rows": int(n), "n_clusters": int(n_clusters)}}

    if report_dir:
        compare_runner_reports(Path(report_dir), results, rep, hist=hist)
    return finish(rep, results, out_path, results_path, t0)


# Diagnosed causes for residual FAIL/MC_BORDERLINE items: (regex on check, cause).  Filled only after a
# discrepancy has been investigated; anything unmatched is reported as 'undiagnosed'.
DIAGNOSES = [
    (r": decision (tau=)?[0-9.]+$",
     "secondary exploratory decision (unadjusted 90% interval): runner and replay bounds agree within MC tolerance, "
     "but the replay bound lies within that tolerance of the bar (observed distance 2e-5 to 1.6e-3), so the call "
     "flips between independent bootstrap streams; Monte Carlo sensitivity, not an implementation discrepancy"),
    (r"^P2-income_prediction__race category$",
     "interpretation, not arithmetic: decision ESTABLISHED_ABOVE agrees; the runner labels an established-above P2 "
     "on a known-C1 unit as C1, the replay lists the unit as C1 + C3 (D1 #8 states the C1 override for P1 only)"),
]


def public_summary(items):
    by_status, by_scope = {}, {}
    for it in items:
        by_status[it["status"]] = by_status.get(it["status"], 0) + 1
        sc = by_scope.setdefault(it["scope"], {})
        sc[it["status"]] = sc.get(it["status"], 0) + 1
    residual = []
    for it in items:
        if it["status"] in ("FAIL", "MC_BORDERLINE", "NOTE"):
            cause = next((c for rx, c in DIAGNOSES if re.search(rx, it["check"])), None)
            residual.append({"check": it["check"], "scope": it["scope"], "status": it["status"],
                             "runner_value": it["runner_value"], "replay_value": it["replay_value"],
                             "abs_diff": it["abs_diff"], "tolerance": it["tolerance"],
                             "cause": cause or it.get("note") or "undiagnosed"})
    return {"n_items": len(items), "by_status": by_status, "by_scope": by_scope,
            "residual_items": residual,
            "contains_per_person_values": False}


def finish(rep, results, out_path, results_path, t0):
    summ = {}
    for it in rep.items:
        summ[it["status"]] = summ.get(it["status"], 0) + 1
    payload = {"schema": "pcrl.cell_a.independent_verification/v1",
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "implementation": "results/combined_stored_model_pilot_v1/verification/replay.py (independent of stored_model_eval)",
               "summary": summ, "public_summary": public_summary(rep.items),
               "elapsed_s": round(time.time() - t0, 1), "items": rep.items}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=1, default=_jsonable))
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(json.dumps(results, indent=1, default=_jsonable))
    return payload, results


def load_historical(path_or_ref):
    p = Path(path_or_ref)
    if p.exists():
        d = json.loads(p.read_text())
    else:
        txt = subprocess.run(["git", "show", path_or_ref], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent).stdout
        d = json.loads(txt)
    if "per_seed" in d:
        d = d["per_seed"]["0"]
    rows = d["rows"] if isinstance(d, dict) else d
    return {(r["purpose"], r["attribute"]): float(r["r2_onehot"]) for r in rows}


def closed_form_native(units, lab, roles, rid_pos, inputs_dir, rep, log, manifest_dirs=None):
    """N0 (all rows, f32/f64), N1 (assessment in-sample), and a G1 re-derivation."""
    native = {}
    for uid, u in units.items():
        man = find_manifest(uid, manifest_dirs or [inputs_dir])
        if man is None:
            rep.add(f"{uid}: manifest present for closed-form check", "native", None, None,
                    status="SKIPPED", note="no manifest found in run_v1/inputs or inputs dir", kind="info")
            continue
        mj = json.loads(man.read_text())
        for fk_, fv_ in mj.get("files", {}).items():   # input files hash-pinned in the manifest
            fp_ = Path(fv_.get("path", "")).expanduser()
            want = fv_.get("sha256", "")
            if fp_.exists() and re.match(r"^[0-9a-f]{64}$", want or "") and fp_.suffix != ".pt":
                got = _SHA_CACHE.setdefault(str(fp_), sha256_file(fp_))
                rep.add(f"{uid}: manifest input '{fk_}' sha256", "inputs", want, got, kind="exact")
        spec = mj["arrays"]["representations"]
        fpath = Path(mj["files"][spec["file"]]["path"]).expanduser()
        if not fpath.is_absolute():
            fpath = man.parent / fpath if (man.parent / fpath).exists() else inputs_dir / fpath
        rep.add(f"{uid}: manifest used for closed-form check", "native", None, str(man), status="INFO", kind="info")
        arr = load_npz(fpath)
        H = arr[spec["key"]]
        hid = arr[spec.get("ids", "row_id")].astype(np.int64)
        order = np.array([rid_pos[int(r)] for r in hid])
        Hl = np.empty_like(H)
        Hl[order] = H   # rows in labels order
        y = np.asarray(lab[u.meta["attr"]]).astype(int)
        K = int(y.max()) + 1          # historical: num_classes = max + 1
        Y = np.eye(K)[y]
        nv = {}
        if u.meta["kind"] == "untreated":
            nv["N0_float32"] = ridge_onehot_r2(Hl, Y, float32_gram=True)
            nv["N0_float64"] = ridge_onehot_r2(Hl.astype(np.float64), Y)
            nv["N0_rows"] = int(len(Hl))
        else:
            nv["N0_status"] = "unverified"   # D1 #7: no historical approval on these rows
        am = roles == "assessment"
        fm = roles == "attacker_fit"
        nv["N1_float64"] = ridge_onehot_r2(Hl[am].astype(np.float64), Y[am])
        nv["N1_mixed"] = ridge_onehot_r2(Hl[am], Y[am], float32_gram=True)
        g1, pred, ybar = ridge_onehot_r2(Hl[fm].astype(np.float64), Y[fm], Hl[am].astype(np.float64), Y[am],
                                         return_pred=True)
        nv["G1_rederived_float64"] = g1
        pr = u.preds
        if "G1_pred" in pr and "assess_row_id" in pr:
            arid = np.asarray(pr["assess_row_id"]).astype(np.int64)
            ai = np.array([rid_pos[int(r)] for r in arid])
            # map my prediction (labels order, assessment subset) to saved order
            pos_in_am = {int(r): i for i, r in enumerate(np.flatnonzero(am))}
            if not all(int(i) in pos_in_am for i in ai):
                rep.add(f"{uid}: G1 re-derivation (assessment rows)", "native", None, None, status="FAIL",
                        note="saved assess_row_id contains non-assessment rows; re-derivation not comparable",
                        kind="info")
                native[uid] = nv
                continue
            saved = np.asarray(pr["G1_pred"], dtype=np.float64)
            fit_classes = sorted(int(c) for c in np.unique(y[fm]))
            cols = class_columns(saved.shape[1], fit_classes, sorted(int(c) for c in np.unique(y)))
            mine = pred[[pos_in_am[int(i)] for i in ai]][:, cols]   # ridge is column-separable
            if saved.shape == mine.shape:
                d = float(np.max(np.abs(saved - mine)))
                rep.add(f"{uid}: saved G1_pred == closed-form ridge re-derivation (float64)", "native",
                        d, 0.0, tol=1e-4, status="PASS" if d <= 1e-4 else "FAIL",
                        note="runner_value = max abs diff over assessment x classes")
            else:
                rep.add(f"{uid}: G1_pred shape", "native", list(saved.shape), list(mine.shape), kind="exact")
        native[uid] = nv
        log(f"[native] {uid}: " + ", ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in nv.items()))
    return native


# ---------------------------------------------------------------- runner report comparison
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(s):
    try:
        if s is None or str(s).strip() == "":
            return None
        v = float(s)
        return v
    except ValueError:
        return None


def _print_tol(s):
    """Half a unit in the last printed decimal (rounding tolerance) with a 1e-9 floor."""
    s = str(s).strip().lower()
    if "e" in s:
        mant, ex = s.split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return max(POINT_TOL, 0.5 * 10 ** (int(ex) - dec))
    dec = len(s.split(".")[1]) if "." in s else 0
    return max(POINT_TOL, 0.5 * 10 ** (-dec)) if dec < 15 else POINT_TOL


def _col(row, *aliases):
    low = {k.lower().strip(): k for k in row}
    for a in aliases:
        if a in low:
            return low[a]
    return None


# ---- runner-table comparison (layouts read from the real report headers, 2026-10-02)
BOUND_SE_MULT = 6.5      # bound tolerance = 6.5 x replay MC SE + 5e-5 (about 4.6 sd of a difference of two
BOUND_FLOOR = 5e-5       # independent percentile bounds with equal B; ~10k bound comparisons -> <0.1 false alarm)

M_MAP = {"macro_auc": "AUC_macro", "worst_class_auc": "AUC_worst_class", "worst_pair_auc": "AUC_worst_pair",
         "LLR_nats": "LLR_nats", "LL_skill": "LL_skill", "brier_skill": "brier_skill"}
U_MAP = {"accuracy": "accuracy", "log_loss": "logloss", "macro_f1": "macro_F1", "macro_auc": "AUC"}


def _unit_prefix(unit):
    m = re.match(r"^income_prediction__sex__p0_sigma([0-9.]+)__seedmean$", unit)
    return f"noise_sigma{m.group(1)}_seedmean" if m else unit


def exploratory_key(unit, quantity, surface, recipe, metric, rid=""):
    """Map a runner (unit, quantity, surface, recipe, metric) to the replay statistic name."""
    up = _unit_prefix(unit)
    q = quantity
    if q in ("G1", "G2") and metric == "r2":
        return f"{up}|{q}|R2"
    if q == "RHO1":
        return f"{up}|RHO1SQ_heldout"
    if q == "pure_metric_G1" and metric == "macro_auc":
        return f"{up}|G1pred|AUC_macro"
    if q == "recovery" and metric in M_MAP:
        return f"{up}|P|{canon_surface(surface)}|{recipe}|{M_MAP[metric]}"
    if q == "label_only" and metric in M_MAP:
        return f"{up}|LO|{M_MAP[metric]}"
    if q == "Uconst":
        return f"{up}|Uconst|accuracy"
    if q in ("U1", "U2") and metric in U_MAP:
        return f"{up}|{q}|{U_MAP[metric]}"
    if q in ("U1_lift", "U2_lift"):
        return f"{up}|{q[:2]}|lift_over_constant"
    if q in ("U1_diff", "U2_diff") and metric in U_MAP:
        return f"{up}|DIFF_vs_untreated|{q[:2]}|{U_MAP[metric]}"
    if q in ("U1_normalised_lift", "U2_normalised_lift"):
        return f"{up}|{q[:2]}|normalised_lift"
    if q == "output_leakage_beyond_label_only":
        return f"{up}|LEAK_outputsNL_minus_LO|AUC"
    if q == "decomposition_step":
        return f"{up}|DECOMP|{rid.rsplit('|', 1)[-1]}"
    return None


def _bound_tol(ex_rec, side):
    se = ex_rec.get(f"mcse_{side}")
    return None if se is None else BOUND_SE_MULT * se + BOUND_FLOOR


def _cmp_triplet(rep, scope, label, row, pcol, lcol, ucol, key, ex):
    """Point (exact up to print rounding, log-loss conventions named) and both 90% bounds (MC tolerance)."""
    if key not in ex:
        rep.add(f"{label}: replay statistic exists", scope, key, None, status="FAIL", kind="info",
                note="runner reports a quantity the replay does not compute")
        return None
    pv = _num(row.get(pcol))
    if pv is None:
        if ex[key].get("estimate") is not None and np.isfinite(ex[key]["estimate"]):
            rep.add(f"{label}: point", scope, row.get(pcol), ex[key]["estimate"], status="FAIL", kind="info",
                    note="runner point missing")
        return None
    used = key   # frozen convention only; other conventions are diagnostic
    tol = _print_tol(row[pcol])
    note = None
    if abs(pv - ex[key]["estimate"]) > tol:
        alt = [k for k in _variant_keys(key, ex)[1:] if abs(pv - ex[k]["estimate"]) <= tol]
        note = (f"runner matches only the non-frozen convention '{alt[0].rsplit('|', 1)[-1]}'" if alt
                else "no clipping convention matches")
    rep.add(f"{label}: point", scope, pv, ex[used]["estimate"], tol=tol, note=note)
    if any(key.endswith(t) for t in ("|logloss", "|LL_skill", "|LLR_nats")) and key + "_eps64" in ex:
        d64 = abs(ex[key + "_eps64"]["estimate"] - ex[key]["estimate"])
        if d64 > 1e-9:
            rep.add(f"{label}: sensitivity, eps64 clipping instead of 1e-12", scope, None,
                    ex[key + "_eps64"]["estimate"], status="INFO", kind="info",
                    note=f"named sensitivity value, |eps64 - 1e-12| = {d64:.3g}; not a comparison")
    for side, col in (("lower", lcol), ("upper", ucol)):
        rv = _num(row.get(col)) if col else None
        if rv is not None and ex[used].get(side) is not None:
            rep.add(f"{label}: 90% {side}", scope, rv, ex[used][side], tol=_bound_tol(ex[used], side),
                    note="MC tolerance (independent RNG stream)")
    return used


def _decide(lower, upper, thr, below_inclusive):
    if lower > thr:
        return "ESTABLISHED_ABOVE"
    if (upper <= thr) if below_inclusive else (upper < thr):
        return "ESTABLISHED_BELOW"
    return "UNRESOLVED"


def _cmp_decisions(rep, scope, label, dec_json, rec, is_r2):
    try:
        d = json.loads(dec_json) if dec_json else {}
    except json.JSONDecodeError:
        rep.add(f"{label}: decisions parseable", scope, dec_json, None, status="FAIL", kind="info")
        return
    for k, rd in d.items():
        thr = 0.0 if k == "vs_0" else float(k.split("=")[-1])
        md = _decide(rec["lower"], rec["upper"], thr, below_inclusive=is_r2)
        tol = max(_bound_tol(rec, "lower"), _bound_tol(rec, "upper"))
        border = min(abs(rec["lower"] - thr), abs(rec["upper"] - thr)) <= tol
        st = "PASS" if rd == md else ("MC_BORDERLINE" if border else "FAIL")
        rep.add(f"{label}: decision {k}", scope, rd, md, status=st, kind="exact")


P_DEC = {("P1", "GENERALISES"): "ESTABLISHED_BELOW", ("P1", "FAILS_TO_GENERALISE"): "ESTABLISHED_ABOVE",
         ("P2", "BELOW_BAR_ESTABLISHED"): "ESTABLISHED_BELOW",
         ("P2", "RECOVERY_OUTSIDE_SCOPE_ESTABLISHED"): "ESTABLISHED_ABOVE"}


def compare_runner_reports(report_dir, res, rep, hist=None):
    ex = res["exploratory"]
    cfg = res["config"]
    bP, sP, bE, sE = cfg["B_prim"], cfg["seed_prim"], cfg["B_expl"], cfg["seed_expl"]
    rep.add("replay ran at the frozen B / seeds", "inference", [B_PRIM, SEED_PRIM, B_EXPL, SEED_EXPL], [bP, sP, bE, sE],
            status="PASS" if [bP, sP, bE, sE] == [B_PRIM, SEED_PRIM, B_EXPL, SEED_EXPL] else "INFO", kind="exact",
            note="INFO = reduced-B validation run")
    mine_p = {r["endpoint"]: r for r in res["primary"]}
    # PRIMARY_FAMILY.json
    f = report_dir / "PRIMARY_FAMILY.json"
    if f.exists():
        pf = json.loads(f.read_text())
        rep.add("PRIMARY_FAMILY ids == replay 16-endpoint family", "primary",
                sorted(x["id"] for x in pf["family"]), sorted(mine_p), kind="exact")
        rep.add("PRIMARY_FAMILY family_size", "primary", pf.get("family_size"), FAMILY_SIZE, kind="exact")
        rep.add("PRIMARY_FAMILY alpha_each", "primary", pf.get("alpha_each"), ALPHA_FAMILY, tol=1e-15)
        rep.add("PRIMARY_FAMILY B / seed", "primary", [pf.get("B"), pf.get("seed")], [bP, sP], kind="exact")
        rep.add("PRIMARY_FAMILY quantile method", "primary", pf.get("quantile_method"), "linear", kind="exact")
        for x in pf["family"]:
            want = ("G1_r2", TAU) if x["id"].startswith("P1-") else ("rep__NL__macro_auc", BAR)
            rep.add(f"{x['id']} statistic/bar", "primary", [x["statistic"], x["bar"]], list(want), kind="exact")
    else:
        rep.add("PRIMARY_FAMILY.json present", "primary", False, True, kind="exact")
    # PRIMARY_ENDPOINTS.csv
    f = report_dir / "PRIMARY_ENDPOINTS.csv"
    rows = _read_csv(f) if f.exists() else []
    rep.add("PRIMARY_ENDPOINTS rows == 16", "primary", len(rows), FAMILY_SIZE, kind="exact")
    for row in rows:
        m = mine_p.get(row["id"])
        if m is None:
            rep.add(f"{row['id']} known to replay", "primary", row["id"], None, status="FAIL", kind="info")
            continue
        fam = row["id"][:2]
        rep.add(f"{row['id']} alpha_each / B / seed", "primary",
                [_num(row.get("alpha_each")), int(_num(row.get("B")) or 0), int(_num(row.get("seed")) or 0)],
                [ALPHA_FAMILY, bP, sP], kind="exact")
        if m["decision"] == "NE":
            rep.add(f"{row['id']} decision", "primary", row["decision"], "NE",
                    status="PASS" if row["decision"].upper() in ("NE", "NOT_ESTIMABLE") else "FAIL", kind="exact")
            continue
        pv = _num(row["point"])
        rep.add(f"{row['id']} point estimate", "primary", pv, m["estimate"], tol=_print_tol(row["point"]))
        for side in ("lower", "upper"):
            rep.add(f"{row['id']} simultaneous {side} bound (alpha=0.05/16, B=20000)", "primary", _num(row[side]),
                    m[side], tol=BOUND_SE_MULT * m[f"mcse_{side}"] + BOUND_FLOOR,
                    note=f"replay MC SE {m[f'mcse_{side}']:.2e}; independent RNG stream")
        md = P_DEC.get((fam, m["decision"]), m["decision"])
        st = "PASS" if row["decision"] == md else ("MC_BORDERLINE" if m.get("mc_borderline") else "FAIL")
        rep.add(f"{row['id']} decision", "primary", row["decision"], md, status=st, kind="exact")
        rep.add(f"{row['id']} non-finite replicates", "primary", int(_num(row.get("n_ne_replicates")) or 0),
                bP - m["n_finite"], kind="exact")
        # endpoint category vs replay unit categories
        cats = res["categories"].get(m["unit"], {}).get("categories", [])
        rc = row.get("category", "")
        tok = re.match(r"^(C\d|none)", rc.strip())
        tok = tok.group(1) if tok else rc
        if md == "UNRESOLVED":
            exp = "C5"
        elif md == "ESTABLISHED_BELOW":
            exp = "none"
        elif fam == "P1":
            exp = "C1" if "C1" in cats else "C2"
        else:
            exp = "C3"
        if tok == exp:
            st, note = "PASS", None
        elif fam == "P2" and tok == "C1" and "C1" in cats and "C3" in cats:
            st, note = "NOTE", ("runner reads an established-above P2 on a C1 unit as C1; the replay assigns the unit "
                               "both C1 and C3 (D1 #8 states the C1 override for P1 only)")
        else:
            st, note = "FAIL", None
        rep.add(f"{row['id']} category", "categories", rc, exp, status=st, kind="exact", note=note)
        nc = row.get("native_N0_category", "")
        rep.add(f"{row['id']} native N0 category", "categories", nc,
                "C1" if "C1" in cats else "historical check passes",
                status="PASS" if (nc.startswith("C1") == ("C1" in cats)) else "FAIL", kind="exact")
    # EXPLORATORY_ENDPOINTS.csv
    f = report_dir / "EXPLORATORY_ENDPOINTS.csv"
    rows = _read_csv(f) if f.exists() else []
    if not rows:
        rep.add("EXPLORATORY_ENDPOINTS.csv present and non-empty", "exploratory", False, True, kind="exact")
    for row in rows:
        key = exploratory_key(row["unit"], row["quantity"], row["surface"], row["recipe"], row["metric"], row["id"])
        label = row["id"]
        if key is None:
            rep.add(f"{label}: mapped to a replay statistic", "exploratory", label, None, status="FAIL", kind="info")
            continue
        used = _cmp_triplet(rep, "exploratory", label, row, "point", "lower90", "upper90", key, ex)
        if used is None:
            continue
        rep.add(f"{label}: B / seed", "exploratory", [int(_num(row["boot_B"])), int(_num(row["boot_seed"]))],
                [bE, sE], kind="exact")
        if row.get("decisions"):
            _cmp_decisions(rep, "exploratory", label, row["decisions"], ex[used], row["metric"] == "r2")
        if row.get("per_seed_points"):
            sp = res["seed_spread"].get(used)
            rp = json.loads(row["per_seed_points"])
            if sp:
                d = max(abs(a - b) for a, b in zip(rp, sp["points"]))
                rep.add(f"{label}: per-seed points", "exploratory", d, 0.0, tol=1e-9,
                        status="PASS" if d <= 1e-9 else "FAIL", note="runner_value = max abs diff")
        if row.get("seed_sd"):
            sp = res["seed_spread"].get(used)
            if sp:
                rv = _num(row["seed_sd"])
                ok1 = abs(rv - sp["sd_ddof1"]) <= 1e-9
                ok0 = abs(rv - sp["sd_ddof0"]) <= 1e-9
                rep.add(f"{label}: seed sd", "exploratory", rv, sp["sd_ddof1"] if ok1 or not ok0 else sp["sd_ddof0"],
                        tol=1e-9, note=None if ok1 else ("matched with ddof=0" if ok0 else None))
    # DECOMPOSITION.csv
    f = report_dir / "DECOMPOSITION.csv"
    rows = _read_csv(f) if f.exists() else []
    nat = res["native"]
    for row in rows:
        uid, step = row["unit"], row["step"]
        label = f"{uid} {step}"
        if step == "F0":
            nv = nat.get(uid, {})
            if "N0_float32" not in nv:
                rep.add(f"{label}: replay N0", "decomposition", None, None, status="FAIL", kind="info")
                continue
            rep.add(f"{label}: N0 mixed precision, clamped", "decomposition", _num(row["point"]),
                    max(0.0, nv["N0_float32"]), tol=N0_MIXED_TOL, note=N0_NOTE)
            rep.add(f"{label}: N0 mixed precision, raw", "decomposition", _num(row["point_raw_mixed"]),
                    nv["N0_float32"], tol=N0_MIXED_TOL, note=N0_NOTE)
            rep.add(f"{label}: N0 float64", "decomposition", _num(row["point_float64"]),
                    max(0.0, nv["N0_float64"]), tol=1e-9)
            if hist and (uid.split("__")[0], uid.split("__")[1]) in hist:
                h = hist[(uid.split("__")[0], uid.split("__")[1])]
                rep.add(f"{label}: historical r2_onehot (as read)", "decomposition", _num(row["historical_r2_onehot"]),
                        h, tol=1e-12)
                dh = abs(h - max(0.0, nv["N0_float32"]))
                same_as_runner = abs(_num(row["point"]) - max(0.0, nv["N0_float32"])) <= 1e-12
                rep.add(f"{label}: replay N0 (mixed, clamped) reproduced within float32 rounding", "native", h,
                        max(0.0, nv["N0_float32"]), tol=1e-5,
                        note=f"|diff| = {dh:.3g}; historical value computed in float32 (tolerance 1e-5, set by the "
                             f"coordinator after the run); replay vs runner N0 agree to 1e-12: {same_as_runner}")
            cats = res["categories"].get(uid, {}).get("categories", [])
            rep.add(f"{label}: N0 category", "categories", row.get("category"),
                    "C1" if "C1" in cats else "historical check passes", kind="exact")
            continue
        fk = res["decomposition"].get(uid, {}).get(f"{step}_key")
        if fk:
            _cmp_triplet(rep, "decomposition", label, row, "point", "lower90", "upper90", fk, ex)
        if row.get("delta_name"):
            _cmp_triplet(rep, "decomposition", f"{label} {row['delta_name']}", row, "delta_point",
                         "delta_lower90", "delta_upper90", f"{uid}|DECOMP|{row['delta_name']}", ex)
    if not rows:
        rep.add("DECOMPOSITION.csv present", "decomposition", False, True, kind="exact")
    # UTILITY.csv
    f = report_dir / "UTILITY.csv"
    rows = _read_csv(f) if f.exists() else []
    for row in rows:
        kind, metric = row["kind"], row["metric"]
        key = exploratory_key(row["unit"], kind, "", "", metric)
        label = f"{row['unit']} {kind} {metric}"
        if key is None:
            rep.add(f"{label}: mapped", "utility", label, None, status="FAIL", kind="info")
            continue
        _cmp_triplet(rep, "utility", label, row, "point", "lower90", "upper90", key, ex)
        if row.get("diff_point") and kind in ("U1", "U2"):
            dk = f"{_unit_prefix(row['unit'])}|DIFF_vs_untreated|{kind}|{U_MAP[metric]}"
            _cmp_triplet(rep, "utility", f"{label} paired diff vs {row.get('reference_unit')}", row,
                         "diff_point", "diff_lower90", "diff_upper90", dk, ex)
    if not rows:
        rep.add("UTILITY.csv present", "utility", False, True, kind="exact")
    # SUPPORT_COVERAGE.csv
    f = report_dir / "SUPPORT_COVERAGE.csv"
    rows = _read_csv(f) if f.exists() else []
    for row in rows:
        uid, what, c = row["unit"], row["what"], str(int(_num(row["class"])))
        src = res["support"].get(uid) if what == "sensitive" else res.get("support_task", {}).get(uid)
        if not src or not src.get("counts") or c not in src["counts"]:
            rep.add(f"{uid} {what} class {c}: known to replay", "support", c, None, status="FAIL", kind="info")
            continue
        cnt = src["counts"][c]
        rep.add(f"{uid} {what} class {c}: counts fit/val/assessment", "support",
                [int(_num(row[f"n_{r}"])) for r in SUPPORT_MIN], [cnt[r] for r in SUPPORT_MIN], kind="exact")
        rep.add(f"{uid} {what} class {c}: supported", "support", row["supported"].strip().lower() == "true",
                int(c) in src["supported_classes"], kind="exact")
        if what == "sensitive":
            rep.add(f"{uid} {what} class {c}: unit status", "support", row["unit_status"],
                    "ESTIMABLE" if res["support"][uid]["estimable"] else "NE", kind="exact")
    if not rows:
        rep.add("SUPPORT_COVERAGE.csv present", "support", False, True, kind="exact")
    # NATIVE_CHECKS.csv
    f = report_dir / "NATIVE_CHECKS.csv"
    rows = _read_csv(f) if f.exists() else []
    for row in rows:
        uid = row["unit"]
        nv = nat.get(uid, {})
        if parse_unit_id(uid)["kind"] == "noise":
            rep.add(f"{uid}: N0 status", "native", row["N0_status"], "unverified", kind="exact")
        elif "N0_float32" in nv:
            rep.add(f"{uid}: N0 mixed clamped", "native", _num(row["N0_mixed_clamped"]), max(0.0, nv["N0_float32"]),
                    tol=N0_MIXED_TOL, note=N0_NOTE)
            rep.add(f"{uid}: N0 mixed raw", "native", _num(row["N0_mixed_raw"]), nv["N0_float32"],
                    tol=N0_MIXED_TOL, note=N0_NOTE)
            rep.add(f"{uid}: N0 float64 clamped", "native", _num(row["N0_float64_clamped"]),
                    max(0.0, nv["N0_float64"]), tol=1e-9)
            rep.add(f"{uid}: N0 rows", "native", int(_num(row["N0_rows"])), nv.get("N0_rows"), kind="exact")
        if "N1_float64" in nv and row.get("N1_float64_raw") is None:
            rep.add(f"{uid}: N1 columns present", "native", False, True, kind="exact")
        elif "N1_float64" in nv:
            rep.add(f"{uid}: N1 float64 raw", "native", _num(row["N1_float64_raw"]), nv["N1_float64"], tol=1e-9)
            rep.add(f"{uid}: N1 mixed raw", "native", _num(row["N1_mixed_raw"]), nv["N1_mixed"],
                    tol=N0_MIXED_TOL, note=N0_NOTE)
    if not rows:
        rep.add("NATIVE_CHECKS.csv present", "native", False, True, kind="exact")


N0_MIXED_TOL = 1e-6
N0_NOTE = ("float32 centring/Gram (historical mixed precision); agreement beyond ~1e-6 depends on BLAS summation "
           "order, and the raw value is ill-conditioned where the float32 Gram loses rank")


def _variant_keys(key, ex):
    """Primary key first, then clipping-convention variants of a log-loss quantity."""
    out = [key]
    if any(key.endswith(t) for t in ("|logloss", "|LL_skill", "|LLR_nats")):
        for suf in [v[0] for v in LL_VARIANTS[1:]] + ["_unclipped"]:
            if key + suf in ex:
                out.append(key + suf)
    return out


def _add_with_variants(rep, check, scope, runner_val, key, ex, tol):
    """Compare against the primary convention; if it fails, a named convention variant may match."""
    if key not in ex:
        return
    keys = _variant_keys(key, ex)
    for k in keys:
        if abs(runner_val - ex[k]["estimate"]) <= tol:
            note = None if k == key else f"matched under log-loss convention variant '{k.split('logloss')[-1]}'"
            rep.add(check, scope, runner_val, ex[k]["estimate"], tol=tol, note=note)
            return
    rep.add(check, scope, runner_val, ex[key]["estimate"], tol=tol,
            note=f"no convention variant matched ({len(keys)} tried)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs-dir", default="~/PCRL_eval_cache_private/pilot_adult_s0")
    ap.add_argument("--run-dir", default="~/PCRL_eval_cache_private/pilot_adult_s0/run_v1")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-out", default=None, help="aggregate replay results JSON (no row-level data)")
    ap.add_argument("--historical", default="origin/main:results/v2_adult_ROUND4/dominant_axis_audit.json")
    ap.add_argument("--b-expl", type=int, default=B_EXPL)
    ap.add_argument("--b-prim", type=int, default=B_PRIM)
    ap.add_argument("--no-expected-counts", action="store_true")
    a = ap.parse_args(argv)
    payload, _ = replay(a.run_dir, a.inputs_dir, a.labels, a.report_dir, a.out, a.b_expl, a.b_prim,
                        expected_role_counts=None if a.no_expected_counts else EXPECTED_ROLE_COUNTS,
                        historical_json=a.historical, results_path=a.results_out)
    print(json.dumps(payload["summary"]))
    fails = [i for i in payload["items"] if i["status"] == "FAIL"]
    for i in fails[:50]:
        print("FAIL:", i["check"], "| runner:", i["runner_value"], "| replay:", i["replay_value"])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

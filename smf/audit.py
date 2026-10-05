"""Independent attacker slates for the strength-matched feedback study (audit/baseline owner; not the training critics).

Adapted from rgj.audit (same interface and semantics) with three prompt-section-12 changes: a precision-justified
defense-aware reader (the old 1e-9 variance cut is gone), a rotated 1e-6 planted clue sent through the release
serialisation and view transform, and shuffled-label nulls scored on a separate diagnostic half of INNER_SELECTION.

Roles. Every attacker is FITTED on AUDIT_FIT and SELECTED on INNER_SELECTION. Inner (selection) audits report the
selected attacker's INNER_SELECTION AUC (= the bank maximum; an optimistic selection statistic, used identically for
every candidate; it is NOT a null test or an unbiased recovery estimate). Final audits refit the selected attackers at
attacker seeds 0, 1, 2 and score them on rows passed in by the caller (`score_idx`); only smf.assess passes
DEVELOPMENT_ASSESSMENT rows, after verifying a pushed EVALUATION_LOCK.json. This module never names or indexes that
role, and every fitting/selection entry point refuses labels < 0 (smf.data's sealed-label mask) on the rows it uses.

Recovery metric. AUC of P(S = 1 | view) for binary SEX (fixed orientation: column 1 of predict_proba, never flipped; an
AUC below 0.5 stays below 0.5); macro one-vs-rest AUC over the supplied classes for multiclass race. Proper-loss
recovery: cross-entropy (probabilities clipped at 1e-12).

Dual selection (every slate). PRIMARY attacker of a view = highest INNER_SELECTION AUC; a separate PROPER-LOSS attacker
= lowest INNER_SELECTION cross-entropy (log-loss reporting only). Ties go to the earlier candidate in bank order.

Coalition bank. For the pair view [v1, v2] the candidates are, in this order: `coalition` (every slate member fitted on
the pair view), `ignore_recipient_2` (every member fitted on v1 alone), `ignore_recipient_1` (every member on v2 alone).
Inner coalition recovery is therefore >= the better inner local recovery by construction (a property of the attack
family, not a clamp); on scored rows the coalition estimate is never clamped to the local estimates. All tables are
returned even when an ignore candidate wins.

Slates (LR/MLP/HGB/CC factories are the pinned predecessor helpers in jcv.audit, imported unchanged).
  INNER      LR(C=1) | MLP(64,64) | HGB(lr 0.1, 31 leaves, 200 it.) | DA_LR                        finite: + CC x3
  FINAL      LR (StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10, 100})
             MLP (StandardScaler + MLP hidden in {(64,), (128,), (64,64), (128,128)}, adam, alpha 1e-4, early stopping
                  on a 10% split of AUDIT_FIT, max_iter 300)
             HGB (lr in {0.05, 0.1} x max_leaf_nodes in {15, 31}, 200 iterations)
             DA_LR, DA_MLP (below)                                                                  finite: + CC x3
  SECONDARY  INNER + DA_MLP (+ CC on finite views): output-only views and the race stress audit.
  CC = exact cell-conditional attacker, Laplace alpha in {0.1, 1, 10} (finite releases: FARE cells, hard decisions).

Defense-aware reader (DA_LR = canonical view -> LogisticRegression(C=1); DA_MLP = canonical view -> MLP(128,128)).
  What "defense-aware" means here, exactly: the attacker knows the PUBLIC release construction -- a recipient view is
  [r_i, centred logits_i], the centred logits are an exact affine function of r_i through the public deployed head and
  sum to zero, a LEACE release has exactly collapsed directions, a FARE release is a one-hot cell code -- and therefore
  knows (i) the released coordinates carry exact algebraic redundancy and (ii) their relative scales are arbitrary, so a
  low-scale direction can be as informative as a large one. It uses that knowledge only to canonicalise the frozen view
  it receives: centre, SVD on AUDIT_FIT, drop the exact null directions, whiten every other direction to unit variance.
  It is an ordinary supervised attacker trained on released views of AUDIT_FIT people with their SEX labels. It uses no
  encoder weights, gradients, critic or controller state, training data, raw permitted inputs of the assessed people,
  labels at inference, repeated or adaptive queries, or any view other than the declared one. It is not a white-box,
  adaptive-query or repeated-query attacker, and it is not the strongest possible attacker.
  Canonicalisation (`Canon`), all float64: mu = AUDIT_FIT column mean; SVD Xc = X - mu = U diag(s) V'; keep direction j
  iff s_j > tol with
      tol = max(n, d) * eps64 * max(s_1, ||X||_2),     eps64 = 2.22e-16, ||X||_2 = spectral norm of the UNCENTRED view;
  kept directions are whitened (x - mu) V_j sqrt(n - 1) / s_j. Justification: this is numpy.linalg.matrix_rank's
  tolerance (SVD backward error O(max(n, d) eps s_1)), with the scale taken as the larger of the centred and uncentred
  norms because rounding of stored and derived entries is relative to their uncentred magnitude (an affine logit block
  computed in float64 carries errors <= ~d eps |x|, i.e. a perturbation of spectral norm <= ~d^1.5 eps ||X||_2, below
  tol whenever n >= d^1.5; here n = 6065 and d <= 44, d^1.5 <= 292). By Weyl's inequality an exactly redundant direction
  then has computed singular value below tol. Measured on predecessor releases restricted to this study's AUDIT_FIT rows
  (U and LEACE views v1, v2, pair): every exact null direction (affine logits, centring, LEACE collapse) had s_j / s_1
  <= 3.5e-15, the tolerance ratio was 1.5e-12 .. 3.7e-12 (>= 400x above), and the smallest genuine direction had
  s_j / s_1 >= 2.6e-3 (nine orders above). A rotated 1e-6 planted clue sits near s_j / s_1 ~ 1e-7, five orders above
  tol, and is kept. No direction at or below tol is ever amplified; directions above tol that hold only float32
  rounding of r would be kept but carry no label signal, and attacker selection on INNER_SELECTION rows that were not
  used for fitting decides whether a kept direction helps. Nothing here claims that every small direction is
  estimable from finite data. The predecessor reader (jcv.audit.Canon) dropped explained variance <= 1e-9 x max
  (s_j / s_1 < 3.2e-5) and so discarded rotated low-scale clues; tests document that blind spot and the repair.

Controls (inner roles only; `controls`, `rotated_plant`; thresholds calibrated on synthetic data, see NULL_CALIBRATION).
  Diagnostic split. INNER_SELECTION is split by exact-record group into halves A and B (u = sha256("20261022|null-split|
  <unit>") / 2^64, A iff u < 0.5). Controls fit on AUDIT_FIT, SELECT the attacker on A (AUC, bank order ties) and
  EVALUATE that one selected attacker on B. Because B was used neither for fitting nor for selection, its AUC is an
  unbiased held-out estimate for the selected attacker, so the selection over the slate (and coalition bank) is
  accounted for. The bank maximum on A (or on all of INNER_SELECTION) is archived as a descriptive selection statistic
  and is never called a null test.
  (a) Shuffled-label null: SEX permuted within AUDIT_FIT, within A and within B (CONTROL_SEED stream, drawn and hashed
      before any fit). Pass iff AUC_B(selected on A) <= 0.5 + NULL_Z * sd0, sd0 = sqrt((n0 + n1 + 1) / (12 n0 n1)) on B
      (Mann-Whitney null sd; conservative with ties). NULL_Z = 3.5 (one-sided 2.3e-4 per view).
  (b) Planted leaks. Plants are declared transformations of the PERMUTED label S* of (a), and are scored against S*: the
      release itself carries no information about S*, so the plant is the only signal and detection is meaningful
      even for a release that already leaks real SEX (a leaky release would trivially clear a fixed AUC bar against
      true SEX). PLANT_ONEHOT = appended one-hot of noisy S* (20% of rows replaced by a uniformly random class);
      PLANT_TINY = appended 1e-6 * (noisy S* - 0.5), coordinate-aligned. Pass iff AUC_B > PLANT_MIN = 0.75 (a perfect
      reader of noisy S* has AUC 0.9). Pair: the plant goes into the pair copy only (v1, v2 clean), so detection must
      come from the coalition's own joint attackers.
  (c) Rotated tiny clue through the release path (`rotated_plant`): r_i of the release is replaced by
      [r_i, 1e-6 * (noisy S* - 0.5)] Q with Q a seeded Haar-random orthogonal matrix (the clue is spread over every
      released coordinate at ~1e-7 of their scale, so no single coordinate exposes it), the planted release is written
      with rgj.finalize.save_unit / np.savez_compressed, re-read hash-checked from disk, transformed with
      rgj.finalize.views_from_release and audited against S*. Pass iff the arrays survive the round trip bit-exactly and
      the selected attacker's AUC_B > PLANT_MIN on v_i and on the pair bank. Per-attacker tables are archived (only the
      scale-aware DA members are expected to see it; the predecessor reader could not).
  Failure flags invalidate the corresponding audit; they are recorded and never dropped.

NULL_CALIBRATION (synthetic; no release of this study existed; NULL_Z = 3.5 was fixed beforehand and not tuned; an
engineering smoke test on one predecessor release, inner roles only, preceded it). `calibrate_null` on study-shaped
null releases (AUDIT_FIT 6065 / INNER_SELECTION 2235 rows, P(S = 1) = 0.68, views [r (16, float32-computed), centred
affine logits (2 | 6)], SEX independent of the views; one shuffled-label null per rep and view; every rep kept):
  slate   tests  exceed NULL_Z  held-out AUC_B mean  z mean  z sd   max AUC_B  bank max (all INNER_SEL / half A)
  inner    300     0            0.5007               +0.04   1.006  0.546      0.513 / 0.517
  final     60     0            0.4982               -0.10   1.020  0.532      0.521 / 0.530
The held-out statistic behaves as N(0, 1) in z units (fraction z > 2: 2.7% inner, 0% final; nominal 2.3%), so the
analytic threshold 0.5 + 3.5 sd0 (~0.565 on the real half B of 1,091 rows) is retained. The bank maximum is biased
upward by 0.013-0.032 over chance, which is why it is never used as the null test.
Degenerate views: a view with no direction above tol (constant) gives DA a prior-only predictor (AUC 0.5).
"""
from __future__ import annotations

import hashlib
import math
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from jcv import audit as JA   # pinned predecessor factories (LR/MLP/HGB/CellConditional), unchanged

warnings.filterwarnings("ignore")

ATT_SEEDS = (0, 1, 2)
CLIP = 1e-12
TIE = 1e-12
EPS64 = float(np.finfo(np.float64).eps)
TOL_MULT = 1.0                    # multiple of max(n, d) * eps64 * scale (numpy matrix_rank convention)
CONTROL_SEED = 20261021
SPLIT_SEED = 20261022
NULL_Z = 3.5
PLANT_MIN = 0.75
PLANT_NOISE = 0.20
PLANT_TINY_AMP = 1e-6
FIT_ROLE, SEL_ROLE = "AUDIT_FIT", "INNER_SELECTION"
PRIMARY_VIEWS = ("v1", "v2", "pair")
IGNORE_NAMES = ("ignore_recipient_2", "ignore_recipient_1")   # local_keys[0] keeps recipient 1 only, [1] recipient 2
NULL_CALIBRATION = {             # receipts of `calibrate_null` (seed0 = 1000, CONTROL_SEED + rep permutations)
    "inner": {"reps": 100, "tests": 300, "exceedances": 0, "heldout_mean": 0.50066, "z_mean": 0.036,
              "z_sd": 1.0065, "heldout_max": 0.54638, "frac_z_gt_2": 0.02667, "bank_max_full_mean": 0.51260,
              "max_auc_A_mean": 0.51721},
    "final": {"reps": 20, "tests": 60, "exceedances": 0, "heldout_mean": 0.49821, "z_mean": -0.097,
              "z_sd": 1.0203, "heldout_max": 0.53197, "frac_z_gt_2": 0.0, "bank_max_full_mean": 0.52065,
              "max_auc_A_mean": 0.52999},
}


# ------------------------------------------------------------------ defense-aware canonical reader
def rank_tolerance(X):
    """(tol, s, scale): singular values s of the centred float64 view and the precision tolerance (see docstring)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    scale = max(float(s[0]) if s.size else 0.0, float(np.linalg.norm(X, 2)))
    return TOL_MULT * max(n, d) * EPS64 * scale, s, scale


class Canon:
    """Float64 canonicalisation: centre, SVD on the fitting rows, drop s_j <= tol, whiten the rest to unit variance."""

    def __init__(self, tol_mult=TOL_MULT):
        self.tol_mult = tol_mult

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu_ = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu_, full_matrices=False)
        self.scale_ = max(float(s[0]) if s.size else 0.0, float(np.linalg.norm(X, 2)))
        self.tol_ = self.tol_mult * max(n, d) * EPS64 * self.scale_
        keep = s > self.tol_
        self.s_, self.keep_ = s, keep
        self.rank_ = int(keep.sum())
        self.W_ = (Vt[keep].T / s[keep]) * math.sqrt(max(n - 1, 1))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu_) @ self.W_

    def receipt(self):
        s1 = float(self.s_[0]) if self.s_.size else 0.0
        dropped = self.s_[~self.keep_]
        return {"rank_kept": self.rank_, "dim": int(len(self.s_)), "tol": float(self.tol_), "scale": self.scale_,
                "tol_over_s1": float(self.tol_ / s1) if s1 else None,
                "max_dropped_over_s1": float(dropped.max() / s1) if dropped.size and s1 else None,
                "min_kept_over_s1": float(self.s_[self.keep_].min() / s1) if self.rank_ and s1 else None}


class _DA:
    """Canonical view -> classifier (sklearn-style fit / predict_proba / classes_)."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        self.c_ = Canon().fit(X)
        Z = self.c_.transform(X)
        if self.c_.rank_ == 0:                       # no direction above tolerance: the view is constant -> prior
            from sklearn.dummy import DummyClassifier
            self.m_ = DummyClassifier(strategy="prior")
            Z = np.zeros((len(Z), 1))
        elif self.kind == "lr":
            self.m_ = LogisticRegression(C=1.0, max_iter=3000)
        else:
            self.m_ = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m_.fit(Z, y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        Z = self.c_.transform(X)
        return self.m_.predict_proba(Z if self.c_.rank_ else np.zeros((len(Z), 1)))


def da_lr(seed):
    return _DA("lr", seed)


def da_mlp(seed):
    return _DA("mlp", seed)


# ------------------------------------------------------------------ slates
def _cc(finite):
    if not finite:
        return []
    return [(f"CC_alpha{a}", (lambda s, a=a: JA.CellConditional(a))) for a in (0.1, 1.0, 10.0)]


def inner_slate(finite=False):
    return JA.slate("inner") + [("DA_LR", da_lr)] + _cc(finite)


def final_slate(finite=False):
    base = [x for x in JA.slate("final", False) if not x[0].startswith("DA")]   # LR x5, MLP x4, HGB x4 (old DA out)
    return base + [("DA_LR", da_lr), ("DA_MLP", da_mlp)] + _cc(finite)


def secondary_slate(finite=False):
    return JA.slate("inner") + [("DA_LR", da_lr), ("DA_MLP", da_mlp)] + _cc(finite)


SLATES = {"inner": inner_slate, "final": final_slate, "secondary": secondary_slate}


# ------------------------------------------------------------------ metrics (fixed orientation)
def proba(m, X, K):
    return JA.proba(m, X, K)


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1))))


def auc_fixed(y, P, classes=(0, 1)):
    """Binary: AUC of P[:, 1] for y == 1 (never flipped). Multiclass: macro OvR AUC of P[:, c] over `classes`."""
    classes = list(classes)
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], P[:, classes[1]]))
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


def null_sd(y, classes=(0, 1)):
    """Mann-Whitney null standard deviation of a binary AUC on labels y."""
    n1 = int((np.asarray(y) == classes[1]).sum())
    n0 = int(len(y) - n1)
    return math.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))


def check_labels(y, *idx_sets):
    """Refuse sealed (negative) or non-integer labels on any row an attacker is fitted, selected or scored on."""
    for ix in idx_sets:
        yy = np.asarray(y)[np.asarray(ix)]
        if yy.size and (yy.min() < 0 or not np.issubdtype(yy.dtype, np.integer)):
            raise ValueError("REFUSED: sealed or invalid labels on attacker rows (labels must be integers >= 0)")


# ------------------------------------------------------------------ core: fit a slate on one view, select a bank
def _finite_for(finite, w):
    return bool(finite.get(w, False)) if isinstance(finite, dict) else bool(finite)


def _fit_table(X, y, fit_idx, sel_idx, slate, K, classes, halves=None):
    rows, models = [], {}
    ys = y[sel_idx]
    for order, (name, fac) in enumerate(slate):
        t0 = time.time()
        m = fac(0).fit(X[fit_idx], y[fit_idx])
        P = proba(m, X[sel_idx], K)
        row = {"attacker": name, "order": order, "inner_auc": auc_fixed(ys, P, classes), "inner_ce": logloss(ys, P)}
        if halves is not None:
            a, b = halves
            row["auc_A"], row["auc_B"] = auc_fixed(ys[a], P[a], classes), auc_fixed(ys[b], P[b], classes)
        row["fit_s"] = round(time.time() - t0, 3)
        rows.append(row)
        models[name] = (fac, m)
    return rows, models


def _pick(cands, key, maximize):
    best = None
    for j, c in enumerate(cands):
        v = c[key]
        if best is None or (v > cands[best][key] + TIE if maximize else v < cands[best][key] - TIE):
            best = j
    return best


def _brief(b):
    out = {"candidate": b["candidate"], "view": b["view"], "attacker": b["attacker"],
           "inner_auc": b["inner_auc"], "inner_ce": b["inner_ce"]}
    for k in ("auc_A", "auc_B"):
        if k in b:
            out[k] = b[k]
    return out


def _label(sel):
    return f"{sel['candidate']}:{sel['view']}:{sel['attacker']}"


def select_bank(V, y, fit_idx, sel_idx, slate_fn, finite=False, K=2, classes=(0, 1), coalition_key="pair",
                local_keys=("v1", "v2"), halves=None):
    """Fit `slate_fn` on every view in V (fit_idx), score on sel_idx, select per view by AUC (primary) and CE.

    Views other than `coalition_key` are local views. If coalition_key and both local_keys are in V, the coalition
    bank = own pair attackers + ignore-recipient candidates; local_keys are ordered (recipient-1 view, recipient-2
    view). halves = (positions A, positions B) into sel_idx: additionally select on A and report that attacker on B
    (selection["split"]). Returns (record, models) with models[(view, attacker)] = (factory, seed-0 model).
    """
    fit_idx, sel_idx = np.asarray(fit_idx), np.asarray(sel_idx)
    check_labels(y, fit_idx, sel_idx)
    tables, models = {}, {}
    for w, X in V.items():
        assert len(X) == len(y), f"view {w} has {len(X)} rows, labels have {len(y)}"
        rows, ms = _fit_table(np.asarray(X), y, fit_idx, sel_idx, slate_fn(_finite_for(finite, w)), K, classes, halves)
        tables[w] = rows
        models.update({(w, nm): m for nm, m in ms.items()})
    sel = {}
    for w in V:
        if w == coalition_key and all(lk in V for lk in local_keys):
            bank = [{"candidate": "coalition", "view": w, **r} for r in tables[w]]
            for lk, cname in zip(local_keys, IGNORE_NAMES):
                bank += [{"candidate": cname, "view": lk, **r} for r in tables[lk]]
        else:
            bank = [{"candidate": "own", "view": w, **r} for r in tables[w]]
        ia, ic = _pick(bank, "inner_auc", True), _pick(bank, "inner_ce", False)
        sel[w] = {"auc": {**_brief(bank[ia]), "bank_index": ia}, "ce": {**_brief(bank[ic]), "bank_index": ic}}
        if halves is not None:
            j = _pick(bank, "auc_A", True)
            sel[w]["split"] = {**_brief(bank[j]), "bank_index": j,
                               "max_auc_A": bank[j]["auc_A"], "heldout_auc_B": bank[j]["auc_B"]}
        if w == coalition_key and len(bank) > len(tables[w]):
            sel[w]["bank"] = [_brief(b) for b in bank]
    rec = {"tables": tables, "selection": sel, "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
           "classes": [int(c) for c in classes]}
    return rec, models


def _views(V, views):
    keys = views or [w for w in ("v1", "v2", "pair", "v") if w in V]
    return {w: np.asarray(V[w]) for w in keys}


# ------------------------------------------------------------------ INNER (selection) audit
def inner_audit(V, D, finite=False, views=None, y=None, slate="inner"):
    """Inner audit of one frozen release (AUDIT_FIT -> INNER_SELECTION; never assessment rows).

    V: dict of row-aligned views over all D rows, e.g. rgj.finalize.views_from_release(z) (keys v1, v2, pair; other
       keys such as "out" are ignored). Single-view mode (one FARE purpose): V = {"v": X}.
    finite: bool or {view: bool}; finite views add the exact cell-conditional attacker.
    Returns a JSON-serialisable dict:
      auc[w]          INNER_SELECTION AUC of the AUC-selected attacker (local views; coalition bank for "pair")
      ce[w]           cross-entropy of the separately CE-selected attacker
      selected[w]     "<candidate>:<view>:<attacker>" of the AUC selection; ce_selected[w] likewise
      worse_local / mean_local / coalition_minus_best_local   (when v1, v2 [and pair] are present)
      tables, selection (incl. the full coalition bank), n_fit, n_select, slate, wall_s
    """
    t0 = time.time()
    VV = _views(V, views)
    yy = D["sex"] if y is None else y
    rec, _ = select_bank(VV, yy, D["idx"][FIT_ROLE], D["idx"][SEL_ROLE], SLATES[slate], finite)
    out = {"auc": {}, "ce": {}, "selected": {}, "ce_selected": {}}
    for w, s in rec["selection"].items():
        out["auc"][w], out["ce"][w] = s["auc"]["inner_auc"], s["ce"]["inner_ce"]
        out["selected"][w], out["ce_selected"][w] = _label(s["auc"]), _label(s["ce"])
    if "v1" in out["auc"] and "v2" in out["auc"]:
        out["worse_local"] = max(out["auc"]["v1"], out["auc"]["v2"])
        out["mean_local"] = (out["auc"]["v1"] + out["auc"]["v2"]) / 2
        if "pair" in out["auc"]:
            out["coalition_minus_best_local"] = out["auc"]["pair"] - out["worse_local"]
    out.update({"slate": slate, "slate_members": [nm for nm, _ in SLATES[slate](False)],
                "finite": finite if isinstance(finite, bool) else dict(finite),
                "roles": {"fit": FIT_ROLE, "select_and_score": SEL_ROLE},
                "statistic": "bank maximum on INNER_SELECTION (selection-optimistic; not a null test)",
                "orientation": "P(S=1), fixed; never flipped", **rec, "wall_s": round(time.time() - t0, 2)})
    return out


def inner_local(X, D, finite=False, slate="inner"):
    """Single-view inner audit (e.g. one FARE purpose): returns (AUC of the AUC-selected attacker, full record)."""
    r = inner_audit({"v": X}, D, finite=finite, slate=slate)
    return r["auc"]["v"], r


# ------------------------------------------------------------------ FINAL slate (dual selection + seed refits)
def final_audit(V, y, fit_idx, sel_idx, score_idx, slate_fn=final_slate, finite=False, K=2, classes=(0, 1),
                coalition_key="pair", local_keys=("v1", "v2"), seeds=ATT_SEEDS):
    """Fit on fit_idx, dual-select on sel_idx, refit each selected attacker at `seeds`, score on score_idx.

    score_idx must be disjoint from fit_idx and sel_idx and carry unsealed labels. Returns (record, probs) with
    probs[f"{crit}_{w}"] an array (len(seeds), len(score_idx), K) of per-row probabilities for crit in {"auc", "ce"}
    and each view w. The record holds every slate table and the coalition bank, the selections with their inner
    values, and the scored-row AUC/CE per seed (descriptive; endpoints are recomputed from the saved probabilities).
    """
    fit_idx, sel_idx, score_idx = (np.asarray(a) for a in (fit_idx, sel_idx, score_idx))
    assert not np.intersect1d(score_idx, fit_idx).size and not np.intersect1d(score_idx, sel_idx).size, \
        "scored rows overlap the attacker fitting or selection rows"
    check_labels(y, score_idx)
    t0 = time.time()
    rec, models = select_bank(V, y, fit_idx, sel_idx, slate_fn, finite, K, classes, coalition_key, local_keys)
    cache, probs, scored = {}, {}, {}
    for w, s in rec["selection"].items():
        scored[w] = {}
        for crit in ("auc", "ce"):
            src, att = s[crit]["view"], s[crit]["attacker"]
            Ps = []
            for sd in seeds:
                key = (src, att, sd)
                if key not in cache:
                    fac, m0 = models[(src, att)]
                    m = m0 if sd == 0 else fac(sd).fit(np.asarray(V[src])[fit_idx], y[fit_idx])
                    cache[key] = proba(m, np.asarray(V[src])[score_idx], K)
                Ps.append(cache[key])
            P = np.stack(Ps)
            probs[f"{crit}_{w}"] = P
            ys = y[score_idx]
            scored[w][crit] = {"source_view": src, "attacker": att, "candidate": s[crit]["candidate"],
                               "seeds": list(seeds), "auc_per_seed": [auc_fixed(ys, p, classes) for p in P],
                               "ce_per_seed": [logloss(ys, p) for p in P]}
            scored[w][crit]["auc_mean"] = float(np.mean(scored[w][crit]["auc_per_seed"]))
            scored[w][crit]["ce_mean"] = float(np.mean(scored[w][crit]["ce_per_seed"]))
    rec.update({"scored": scored, "n_score": int(len(score_idx)), "refits": len(cache),
                "slate_members": [nm for nm, _ in slate_fn(False)],
                "orientation": "P(class) columns, fixed; never flipped on scored rows",
                "wall_s": round(time.time() - t0, 2)})
    return rec, probs


# ------------------------------------------------------------------ controls (inner roles only)
def _group_u(groups, seed, salt):
    return np.array([int(hashlib.sha256(f"{seed}|{salt}|{int(g)}".encode()).hexdigest()[:16], 16) / 2.0 ** 64
                     for g in groups])


def null_split(D, seed=SPLIT_SEED):
    """Group-wise halves of INNER_SELECTION: (positions A, positions B) into D["idx"][SEL_ROLE]."""
    ix = D["idx"][SEL_ROLE]
    groups = D["unit"][ix] if "unit" in D else ix
    u = _group_u(groups, seed, "null-split")
    return np.flatnonzero(u < 0.5), np.flatnonzero(u >= 0.5)


def frozen_permutation(S, D, halves=None, seed=CONTROL_SEED):
    """SEX permuted within AUDIT_FIT and within each INNER_SELECTION half (drawn before any fit). (S_perm, sha256)."""
    rng = np.random.default_rng(seed)
    Sp = S.copy()
    sel = D["idx"][SEL_ROLE]
    blocks = [D["idx"][FIT_ROLE]] + ([sel[h] for h in halves] if halves is not None else [sel])
    for ix in blocks:
        Sp[ix] = S[ix][rng.permutation(len(ix))]
    used = np.concatenate([D["idx"][FIT_ROLE], sel])
    return Sp, hashlib.sha256(np.ascontiguousarray(Sp[used]).astype(np.int64).tobytes()).hexdigest()


def planted_columns(S, seed=CONTROL_SEED):
    """Declared transformations of a label S (controls pass S*): one-hot of noisy S and a tiny column (noisy 20%)."""
    noisy = noisy_sex(S, seed)
    return {"PLANT_ONEHOT": np.eye(2)[noisy], "PLANT_TINY": PLANT_TINY_AMP * (noisy[:, None] - 0.5)}


def noisy_sex(S, seed=CONTROL_SEED):
    rng = np.random.default_rng(seed + 1)
    noisy = np.asarray(S).copy()
    flip = rng.random(len(noisy)) < PLANT_NOISE
    noisy[flip] = rng.integers(0, 2, int(flip.sum()))
    return noisy


def null_threshold(Sp, D, halves):
    b = D["idx"][SEL_ROLE][halves[1]]
    sd = null_sd(Sp[b])
    return 0.5 + NULL_Z * sd, sd


def _split_entry(sel_w):
    s = sel_w["split"]
    return {"selected_on_A": _label(s), "max_auc_A": s["max_auc_A"], "heldout_auc_B": s["heldout_auc_B"]}


def controls(V, D, finite=False, slate="final", views=None):
    """Shuffled-label null (held-out half) and planted-leak controls on one release (inner roles only)."""
    t0 = time.time()
    VV = _views(V, views)
    S = D["sex"]
    f, v = D["idx"][FIT_ROLE], D["idx"][SEL_ROLE]
    check_labels(S, f, v)
    halves = null_split(D)
    Sp, perm_sha = frozen_permutation(S, D, halves)       # frozen before fitting
    thr, sd = null_threshold(Sp, D, halves)
    plants = planted_columns(Sp)                         # plants encode the permuted label S* and are scored on S*
    out = {"slate": slate, "slate_members": [nm for nm, _ in SLATES[slate](False)], "control_seed": CONTROL_SEED,
           "permutation_sha256": perm_sha, "split": _split_receipt(D, halves),
           "plant_definition": {"label": "S* = SEX permuted within AUDIT_FIT, half A, half B (the null's label)",
                                "PLANT_ONEHOT": "one-hot(noisy S*), noisy = S* with 20% rows set to a random class",
                                "PLANT_TINY": f"{PLANT_TINY_AMP} * (noisy S* - 0.5), one appended column"},
           "thresholds": {"null_z": NULL_Z, "null_sd_B": sd, "null_max_heldout_auc": thr, "plant_min": PLANT_MIN},
           "statistic": "AUC on half B of the attacker selected on half A (fit AUDIT_FIT); bank maxima are descriptive",
           "views": {}}
    null_rec, _ = select_bank(VV, Sp, f, v, SLATES[slate], finite, halves=halves)
    for w in VV:
        e = _split_entry(null_rec["selection"][w])
        out["views"][w] = {"null_heldout_auc_B": e["heldout_auc_B"], "null_selected_on_A": e["selected_on_A"],
                           "null_selection_max_auc_A_descriptive": e["max_auc_A"],
                           "null_bank_max_inner_auc_descriptive": null_rec["selection"][w]["auc"]["inner_auc"]}
    for pname, col in plants.items():
        for w in VV:
            Vp = dict(VV)
            Vp[w] = np.hstack([VV[w], col])
            sub = {w: Vp[w]} if w != "pair" else Vp     # pair: bank with clean local copies
            r, _ = select_bank(sub, Sp, f, v, SLATES[slate], finite, halves=halves)
            e = _split_entry(r["selection"][w])
            out["views"][w].update({f"{pname}_heldout_auc_B": e["heldout_auc_B"],
                                    f"{pname}_selected_on_A": e["selected_on_A"],
                                    f"{pname}_bank_max_inner_auc": r["selection"][w]["auc"]["inner_auc"]})
    for w, d in out["views"].items():
        d["null_ok"] = d["null_heldout_auc_B"] <= thr
        d["plant_ok"] = all(d[f"{p}_heldout_auc_B"] > PLANT_MIN for p in plants)
    out["failures"] = [f"{w}:{k}" for w, d in out["views"].items() for k in ("null_ok", "plant_ok") if not d[k]]
    out["all_ok"] = not out["failures"]
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def _split_receipt(D, halves):
    sel = D["idx"][SEL_ROLE]
    a, b = sel[halves[0]], sel[halves[1]]
    rid = D.get("row_id", np.arange(len(D["sex"])))
    h = lambda ix: hashlib.sha256(np.sort(rid[ix]).astype(np.int64).tobytes()).hexdigest()   # noqa: E731
    return {"seed": SPLIT_SEED, "rule": "exact-record group hash, A iff u < 0.5", "rows_A": int(len(a)),
            "rows_B": int(len(b)), "row_id_sha256_A": h(a), "row_id_sha256_B": h(b)}


def random_rotation(d, seed):
    """Seeded Haar-random orthogonal d x d matrix (QR of a Gaussian with sign correction)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def planted_release(z, S, recipient=1, amp=PLANT_TINY_AMP, seed=CONTROL_SEED):
    """Release copy whose r_i is [r_i, amp * (noisy S - 0.5)] Q (rotated low-scale clue; c_i, p_i, hard_i unchanged)."""
    zp = {k: np.asarray(z[k]) for k in (z.files if hasattr(z, "files") else z)}
    R = np.asarray(zp[f"r{recipient}"], dtype=np.float64)
    t = amp * (noisy_sex(S, seed)[:, None] - 0.5)
    Q = random_rotation(R.shape[1] + 1, seed + 2 + recipient)
    zp[f"r{recipient}"] = np.hstack([R, t]) @ Q
    return zp, {"recipient": recipient, "amplitude": amp, "rotation_seed": seed + 2 + recipient,
                "rotation_dim": int(R.shape[1] + 1), "clue": "amp * (noisy label - 0.5), noisy share 20%",
                "max_coordinate_share_of_clue_direction": float(np.abs(Q[-1]).max())}


def rotated_plant(z, D, slate="final", recipients=(1,), finite=False, workdir=None):
    """Control (c): rotated 1e-6 clue in r_i, through save_unit/np.load/views_from_release, audited (inner roles)."""
    from rgj import finalize as FN
    t0 = time.time()
    S = D["sex"]
    f, v = D["idx"][FIT_ROLE], D["idx"][SEL_ROLE]
    check_labels(S, f, v)
    halves = null_split(D)
    Sp, perm_sha = frozen_permutation(S, D, halves)       # the plant encodes S*; the release carries no S* signal
    out = {"slate": slate, "plant_min": PLANT_MIN, "split": _split_receipt(D, halves), "label": "S* (permuted SEX)",
           "permutation_sha256": perm_sha, "recipients": {}}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        for i in recipients:
            zp, meta = planted_release(z, Sp, i)
            ud = Path(tmp) / f"planted_r{i}"
            FN.save_unit(ud, {"release.npz": lambda p, zp=zp: np.savez_compressed(p, **zp)}, {"control": meta})
            if not FN.unit_complete(ud):
                raise RuntimeError("planted release is not hash-complete after writing")
            z2 = np.load(ud / "release.npz")
            exact = all(np.array_equal(z2[k], zp[k]) for k in zp)
            V2 = FN.views_from_release(z2)
            Vs = {w: np.asarray(V2[w]) for w in PRIMARY_VIEWS}
            r, _ = select_bank(Vs, Sp, f, v, SLATES[slate], finite, halves=halves)
            wi = f"v{i}"
            res = {"plant": meta, "serialisation": {"writer": "rgj.finalize.save_unit + np.savez_compressed",
                                                   "reread_hash_complete": True, "reread_arrays_identical": exact,
                                                   "transform": "rgj.finalize.views_from_release"},
                   "canon_receipt": {w: Canon().fit(Vs[w][f]).receipt() for w in (wi, "pair")}}
            for w in (wi, "pair"):
                e = _split_entry(r["selection"][w])
                res[w] = {**e, "bank_max_inner_auc": r["selection"][w]["auc"]["inner_auc"],
                          "selected_full": _label(r["selection"][w]["auc"]),
                          "per_attacker_auc_B": {x["attacker"]: x["auc_B"] for x in r["tables"][w]}}
            res["pair"]["coalition_own_max_auc_B"] = max(x["auc_B"] for x in r["tables"]["pair"])
            res["ok"] = bool(exact and all(res[w]["heldout_auc_B"] > PLANT_MIN for w in (wi, "pair")))
            out["recipients"][str(i)] = res
    out["all_ok"] = all(x["ok"] for x in out["recipients"].values())
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ synthetic null calibration
def synthetic_D(n_fit=6065, n_sel=2235, p1=0.68, seed=0, dims=(16, 16), K=(2, 6)):
    """Study-shaped synthetic release (no SEX information): correlated features + affine centred logits."""
    rng = np.random.default_rng(seed)
    n = n_fit + n_sel
    D = {"sex": (rng.random(n) < p1).astype(np.int64), "unit": np.arange(n, dtype=np.int64),
         "row_id": np.arange(n, dtype=np.int64), "idx": {FIT_ROLE: np.arange(n_fit), SEL_ROLE: np.arange(n_fit, n)}}
    z = {}
    for i, (d, k) in enumerate(zip(dims, K), start=1):
        A = rng.normal(size=(d, d)) * np.exp(rng.uniform(-3, 1, d))      # scales spanning ~e^4
        R = (rng.normal(size=(n, d)) @ A).astype(np.float32).astype(np.float64)
        Wt = rng.normal(size=(d, k))
        L = R @ Wt + rng.normal(size=k)
        z[f"r{i}"], z[f"c{i}"] = R, L - L.mean(1, keepdims=True)
        z[f"p{i}"] = np.full((n, k), 1.0 / k)
        z[f"hard{i}"] = np.zeros(n, dtype=np.int64)
    return D, z


def calibrate_null(reps=40, slate="inner", seed0=1000, **kw):
    """Run the null procedure on `reps` synthetic null releases; returns every rep (failures preserved) + summary."""
    from rgj import finalize as FN
    rows = []
    for k in range(reps):
        D, z = synthetic_D(seed=seed0 + k, **kw)
        V = FN.views_from_release(z)
        S = D["sex"]
        halves = null_split(D)
        Sp, _ = frozen_permutation(S, D, halves, seed=CONTROL_SEED + k)
        thr, sd = null_threshold(Sp, D, halves)
        r, _ = select_bank({w: V[w] for w in PRIMARY_VIEWS}, Sp, D["idx"][FIT_ROLE], D["idx"][SEL_ROLE],
                           SLATES[slate], False, halves=halves)
        for w in PRIMARY_VIEWS:
            s = r["selection"][w]
            rows.append({"rep": k, "view": w, "heldout_auc_B": s["split"]["heldout_auc_B"],
                         "max_auc_A": s["split"]["max_auc_A"], "bank_max_full": s["auc"]["inner_auc"],
                         "selected_on_A": _label(s["split"]), "sd0": sd, "threshold": thr,
                         "exceeds": s["split"]["heldout_auc_B"] > thr})
    hb = np.array([x["heldout_auc_B"] for x in rows])
    zz = (hb - 0.5) / np.array([x["sd0"] for x in rows])
    summ = {"reps": reps, "slate": slate, "tests": len(rows), "exceedances": int(sum(x["exceeds"] for x in rows)),
            "heldout_mean": float(hb.mean()), "heldout_z_mean": float(zz.mean()), "heldout_z_sd": float(zz.std(ddof=1)),
            "heldout_q99": float(np.quantile(hb, 0.99)), "heldout_max": float(hb.max()),
            "frac_z_gt_2": float((zz > 2).mean()), "nominal_frac_z_gt_2": 0.02275,
            "bank_max_full_mean": float(np.mean([x["bank_max_full"] for x in rows])),
            "max_auc_A_mean": float(np.mean([x["max_auc_A"] for x in rows])),
            "threshold_mean": float(np.mean([x["threshold"] for x in rows])), "null_z": NULL_Z}
    return {"summary": summ, "rows": rows}


# ------------------------------------------------------------------ helpers for predecessor/test releases
def align_release(z, D, keys=None):
    """Rows of a release npz (with row_id) re-ordered to D's rows by row_id; raises if any D row is missing."""
    rid = np.asarray(z["row_id"])
    pos = {int(r): j for j, r in enumerate(rid)}
    try:
        take = np.array([pos[int(r)] for r in D["row_id"]])
    except KeyError as e:
        raise ValueError(f"release lacks D row_id {e}") from None
    return {k: np.asarray(z[k])[take] for k in (keys or z.files) if k != "row_id"} | {"row_id": rid[take]}


def main(argv=None):
    """python -m smf.audit {controls|inner|rotated} --unit <name> [--units-dir DIR] [--slate final|inner] [--out F]
    python -m smf.audit calibrate --reps 40 [--slate inner] [--out F]       (synthetic only)"""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("controls", "inner", "rotated", "calibrate"))
    ap.add_argument("--unit")
    ap.add_argument("--units-dir", default=str(Path.home() / "PCRL_eval_cache_private" / "smf_v1" / "run" / "units"))
    ap.add_argument("--slate", default=None)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--recipients", nargs="+", type=int, default=[1])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    dump = lambda o: json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))  # noqa: E731
    if a.cmd == "calibrate":
        res = calibrate_null(a.reps, a.slate or "inner")
        txt = dump(res)
        if a.out:
            Path(a.out).write_text(txt)
        print(dump(res["summary"]))
        return
    from rgj import finalize as FN
    from smf import data as DA
    D = DA.load()
    d = Path(a.units_dir) / a.unit
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: {d.name} is not hash-complete")
    z = align_release(np.load(d / "release.npz"), D)
    V = FN.views_from_release(z)
    if a.cmd == "controls":
        res = controls(V, D, slate=a.slate or "final")
    elif a.cmd == "rotated":
        res = rotated_plant(z, D, slate=a.slate or "final", recipients=tuple(a.recipients))
    else:
        res = inner_audit(V, D, slate=a.slate or "inner")
    txt = dump(res)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt if a.cmd != "inner" else dump({k: res[k] for k in ("auc", "selected", "wall_s")}))


if __name__ == "__main__":
    main()

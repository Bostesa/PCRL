"""True-label utility metrics and the unchanged inner confidence contract (role D; prompt sections 7 A3, 11, 12).

Provenance: the metric functions are dpc.utility at the source evidence SHA 0a7b05a5 (imported, unchanged; dpc/ is
pinned in this tree and never edited). This module adds the qpc contract on top: the exact-decision-preservation gate,
the headroom preferences, the retention RATIO, the normalised confidence excess, accuracy-identity receipts and the
all-seed aggregation used by the Stage A capacity gate and by inner selection.

Every arm (policy codes, continuous sources, references, U) is measured with the SAME functions on the SAME rows:
  acc                 mean(hard == y)                                            (dpc.utility.metrics)
  logloss             -mean log clip(P[y], 1e-12, 1)   natural log, float64      (dpc.utility.per_row)
  brier               mean_rows sum_k (P_k - 1[y = k])^2   (multiclass Brier)
  balanced_acc        mean recall over classes with >= 30 evaluated rows; recalls of every present class kept
  ece                 10 equal-width bins of max probability (lo, hi]; accuracy of the RELEASED decision
  const_class         OSF_DEFENSE_FIT majority class per task (argmax of bincount, ties -> lower class); const_acc on
                      the evaluated rows (the fitting-prior constant; its probability vector is scored descriptively)
  gain                acc - const_acc
  retention           gain / gain(U)   (None when gain(U) <= 0; the gate itself uses the margin form below, which is
                      the same inequality without a division)

Gate (`gate`, one seed, both tasks; every margin >= 0, no tolerance; U = the U teacher of the SAME seed on the SAME
rows; the decision flag is computed from the release by the caller, e.g. with `decision_preservation`):
  G_acc     acc >= acc(U) - 0.01                         margin = acc - (acc(U) - 0.01)
  G_ll      logloss <= logloss(U) + 0.01                 margin = logloss(U) + 0.01 - logloss
  G_brier   brier <= brier(U) + 0.005                    margin = brier(U) + 0.005 - brier
  G_ret     gain >= 0.8 gain(U)                          margin = gain - 0.8 gain(U)
  G_gain    gain >= 0.03                                 margin = gain - 0.03
  G_dec     released decision == teacher decision on every row (boolean; per task); False or missing -> fail
Headroom preferences (rate selection only, never a replacement for the gate):
  H_ll      logloss <= logloss(U) + 0.0075               margin = logloss(U) + 0.0075 - logloss
  H_brier   brier <= brier(U) + 0.0035                   margin = brier(U) + 0.0035 - brier
Normalised confidence excess (Q* ordering): per task and seed max((logloss - logloss(U)) / 0.01,
(brier - brier(U)) / 0.005); the configuration value is the max over tasks and seeds (signed: < 0 means better than U
on both). Accuracy-identity receipts: acc - acc(U), gain - gain(U) and whether they are exactly zero (class
preservation makes the accuracy gates identities; the receipts keep them visible instead of dropping them).

Labels read: task labels of OSF_DEFENSE_FIT (constant only; allowlist procedure "fitting") and of the evaluated rows
(INNER_SELECTION: procedure "selection"; assessment rows only under procedure "assessment" on an unsealed D). Every read
goes through qpc.data.labels_for (falls back to the identical dpc/osf allowlist only while qpc.data does not exist);
sealed (negative) labels are refused.

Lead API (qpc.gate / qpc.select):
  task_metrics(prob, y, hard, K)                       -> acc, logloss, brier, balanced_acc, recalls, ece, ...
  constant_class(D, task)                              -> OSF_DEFENSE_FIT majority class
  release_inner_utility(probs {1,2}, hard {1,2}, D)    -> {"income": metrics, "occupation": metrics} on INNER_SELECTION
  gate_record(code_u, U_u, preserved {1: bool, 2: bool}) -> ONE seed: per task acc_ok, ll_ok, brier_ok, retention_ok,
                                                          gain_ok, preserved, ll_excess, brier_excess, norm_excess,
                                                          headroom_ll, headroom_brier; plus eligible, headroom
  decision_preservation(release, own_teacher, D)       -> exact per-task preservation on all rows / per role
Internal ("0"/"1"-keyed) forms: release_utility, inner_release_utility, gate, gate_all_seeds (used by audit/assess).
"""
from __future__ import annotations

import numpy as np

from dpc import utility as DU
from dpc.utility import (CLIP, ECE_BINS, KS, SEL_ROLE, SUPPORT_MIN, TASKS, constants, ece, fit_index,  # noqa: F401
                         per_row)

ALLOW_ACC, ALLOW_LL, ALLOW_BRIER, RETAIN, MIN_GAIN = 0.01, 0.01, 0.005, 0.8, 0.03
HEAD_LL, HEAD_BRIER = 0.0075, 0.0035
GATES = ("G_acc", "G_ll", "G_brier", "G_ret", "G_gain", "G_dec")
NUMERIC_GATES = GATES[:-1]
HEADROOM = ("H_ll", "H_brier")
TASK_KEYS = ("0", "1")
RULE = ("acc >= U-0.01; logloss <= U+0.01; brier <= U+0.005; gain >= 0.8 gain(U); gain >= 0.03; exact teacher "
        "decision preservation; each task, each seed (margin >= 0, no tolerance)")
HEADROOM_RULE = "logloss <= U+0.0075 and brier <= U+0.0035, each task, each seed (rate preference only)"


# ------------------------------------------------------------------ label access (qpc.data allowlist)
_LOCAL_ALLOW = {"fitting": ("OSF_DEFENSE_FIT",), "selection": ("INNER_SELECTION",),
                "inner_audit": ("AUDIT_FIT", "INNER_SELECTION"),
                "assessment": ("OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")}
_ALIASES = {"DEFENSE_FIT": "OSF_DEFENSE_FIT", "DEVELOPMENT_ASSESSMENT": "OSF_DEVELOPMENT_ASSESSMENT"}
TASK_NAMES = {"income": 0, "occupation": 1, "occupation_group": 1, 0: 0, 1: 1, "0": 0, "1": 1, 2: 1}
OUT_NAMES = ("income", "occupation")


def _role_index(D, role):
    if role in D["idx"]:
        return np.asarray(D["idx"][role])
    alt = _ALIASES.get(role) or next((a for a, b in _ALIASES.items() if b == role), None)
    return np.asarray(D["idx"][alt])


def label_rows(D, procedure, role):
    """Row positions of `role` whose task labels `procedure` may read: qpc.data.labels_for when qpc.data provides it
    (the study allowlist, owned by role E), otherwise the identical dpc/osf allowlist locally (synthetic tests before
    qpc.data exists). Sealed labels are refused in both paths."""
    try:
        from qpc import data as QD
    except ModuleNotFoundError as e:
        if e.name not in ("qpc.data",):
            raise
        QD = None
    f = getattr(QD, "labels_for", None) if QD is not None else None
    if f is not None:
        return np.asarray(f(D, procedure, role))
    r = _ALIASES.get(role, role)
    if procedure not in _LOCAL_ALLOW or r not in _LOCAL_ALLOW[procedure]:
        raise PermissionError(f"{procedure} may not read labels of {role}")
    ix = _role_index(D, role)
    if D.get("sealed", True) and "role" in D and np.any(np.asarray(D["role"])[ix] == "OSF_DEVELOPMENT_ASSESSMENT"):
        raise PermissionError("assessment labels are sealed until the pushed EVALUATION_LOCK")
    return ix


def task_labels(D, task, procedure, role):
    """(rows, true labels) of one task on one role through the allowlist; negative (sealed) labels refused."""
    i = TASK_NAMES[task]
    rows = label_rows(D, procedure, role)
    y = np.asarray(D["y"][TASKS[i]])[rows]
    return rows, DU._labels_ok(y)


# ------------------------------------------------------------------ release outputs
def as_out(z, D=None):
    """{p1, p2, hard1, hard2} over all D rows from a qpc/dpc release (q_i, hard_i), a teacher unit (p_i, d_i) or an
    already-formed output dict. Refuses a release that is not in D row order."""
    keys = set(z.files if hasattr(z, "files") else z)
    if D is not None and "row_id" in keys and not np.array_equal(np.asarray(z["row_id"]), np.asarray(D["row_id"])):
        raise ValueError("REFUSED: release rows are not aligned with D (row_id order)")
    out = {}
    for i in (1, 2):
        if f"q{i}" in keys:
            out[f"p{i}"], out[f"hard{i}"] = np.asarray(z[f"q{i}"]), np.asarray(z[f"hard{i}"])
        elif f"hard{i}" in keys:
            out[f"p{i}"], out[f"hard{i}"] = np.asarray(z[f"p{i}"]), np.asarray(z[f"hard{i}"])
        else:
            out[f"p{i}"], out[f"hard{i}"] = np.asarray(z[f"p{i}"]), np.asarray(z[f"d{i}"])
    if D is not None:
        n = len(D["row_id"])
        for k, v in out.items():
            if len(v) != n:
                raise ValueError(f"REFUSED: release array {k} has {len(v)} rows, D has {n}")
    return out


def decision_preservation(release, teacher, D=None):
    """Exact pointwise decision preservation of a release against ITS OWN teacher (d_i), per task, on all rows and per
    role when D is given (fitting, selection and assessment rows; no labels read). For a continuous source or reference
    release the release is its own teacher output, so this is True by construction and is still computed."""
    r, t = as_out(release, D), as_out(teacher, D)
    out = {"rule": "released hard_i == teacher d_i on every row (both recipients)", "tasks": {}}
    for j, k in enumerate(TASK_KEYS, start=1):
        h, d = np.asarray(r[f"hard{j}"]), np.asarray(t[f"hard{j}"])
        ok = bool(h.shape == d.shape and np.array_equal(h, d))
        rec = {"ok": ok, "mismatches": int((h != d).sum()) if h.shape == d.shape else None}
        if D is not None and h.shape == d.shape:
            rec["mismatches_by_role"] = {str(role): int((h[np.asarray(ix)] != d[np.asarray(ix)]).sum())
                                         for role, ix in D["idx"].items()}
        out["tasks"][k] = rec
    out["ok"] = all(v["ok"] for v in out["tasks"].values())
    return out


# ------------------------------------------------------------------ metrics
def task_metrics(prob, y, hard, K):
    """acc, logloss (clip 1e-12, natural log), brier (multiclass sum of squares), balanced_acc (classes with >= 30
    rows), recalls, class counts, ece (10 bins; accuracy of the released decision) of one task on one row set."""
    P = np.asarray(prob, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    y = DU._labels_ok(y).astype(np.int64)
    if P.shape != (len(y), K) or hard.shape != (len(y),):
        raise ValueError(f"REFUSED: shapes {P.shape} / {hard.shape} do not match {len(y)} rows x {K} classes")
    pr = per_row(P, y, K)
    counts = {c: int((y == c).sum()) for c in range(K)}
    rec = {c: float((hard[y == c] == c).mean()) for c in range(K) if counts[c]}
    sup = [c for c in range(K) if counts[c] >= SUPPORT_MIN]
    e, rel = ece(P, hard, y)
    return {"n": int(len(y)), "acc": float((hard == y).mean()), "logloss": float(pr["ll"].mean()),
            "brier": float(pr["br"].mean()), "balanced_acc": float(np.mean([rec[c] for c in sup])) if sup else None,
            "recalls": {str(c): v for c, v in rec.items()}, "class_counts": {str(c): v for c, v in counts.items()},
            "supported_classes": sup, "unsupported_classes": [c for c in range(K) if 0 < counts[c] < SUPPORT_MIN],
            "ece": e, "reliability": rel,
            "decision_note": None if np.array_equal(P.argmax(1), hard) else
            "released decision differs from argmax of released probabilities on some rows"}


def constant_class(D, task):
    """OSF_DEFENSE_FIT majority class of `task` (argmax of the bincount; ties -> lower class), reading the true labels
    of OSF_DEFENSE_FIT only (allowlist procedure "fitting")."""
    i = TASK_NAMES[task]
    _, y = task_labels(D, i, "fitting", "OSF_DEFENSE_FIT")
    return int(np.argmax(np.bincount(y, minlength=KS[i])))


def constant_prior(D, task):
    """OSF_DEFENSE_FIT class frequencies of `task` (the fitting-prior probability constant; descriptive)."""
    i = TASK_NAMES[task]
    _, y = task_labels(D, i, "fitting", "OSF_DEFENSE_FIT")
    c = np.bincount(y, minlength=KS[i]).astype(np.float64)
    return c / c.sum()


def metrics(P, hard, y, K, const, prior=None, u=None):
    """task_metrics plus the constant (const_class, const_acc, gain; descriptive prior-vector log loss / Brier when
    `prior` is given) and, when u (U's metrics on the same rows) is given, the U anchors, the retention ratio, signed
    confidence excesses, the normalised excess and the accuracy-identity receipts. Same numbers as dpc.utility.metrics
    for every shared key (tested)."""
    m = task_metrics(P, y, hard, K)
    y = np.asarray(y, dtype=np.int64)
    m.update({"const_class": int(const), "const_acc": float((y == const).mean())})
    m["gain"] = m["acc"] - m["const_acc"]
    sup = m["supported_classes"]
    m["minority_class"] = min(sup, key=lambda c: m["class_counts"][str(c)]) if sup else None
    m["minority_recall"] = m["recalls"].get(str(m["minority_class"])) if m["minority_class"] is not None else None
    if prior is not None:
        pc = per_row(np.repeat(np.asarray(prior, dtype=np.float64)[None], len(y), 0), y, K)
        m["const_logloss"], m["const_brier"] = float(pc["ll"].mean()), float(pc["br"].mean())
    if u is not None:
        if abs(m["const_acc"] - u["const_acc"]) > 1e-15 or m["n"] != u["n"]:
            raise ValueError("candidate and U were evaluated on different rows (constant accuracies differ)")
        _anchor(m, u)
    return m


def _anchor(m, u):
    ug = u["acc"] - u["const_acc"]
    m.update({"u_acc": u["acc"], "u_logloss": u["logloss"], "u_brier": u["brier"], "u_gain": ug,
              "retention": (m["gain"] / ug) if ug > 0 else None,
              "gain_retention_margin": m["gain"] - RETAIN * ug,
              "excess_logloss": m["logloss"] - u["logloss"], "excess_brier": m["brier"] - u["brier"],
              "acc_minus_u": m["acc"] - u["acc"], "gain_minus_u_gain": m["gain"] - ug,
              "acc_identity_exact": bool(m["acc"] == u["acc"])})
    m["normalised_excess"] = max(m["excess_logloss"] / ALLOW_LL, m["excess_brier"] / ALLOW_BRIER)
    return m


def _on_rows(a, rows, n):
    a = np.asarray(a)
    if len(a) == n:
        return a[rows]
    if len(a) == len(rows):
        return a
    raise ValueError(f"REFUSED: array of {len(a)} rows is neither all {n} D rows nor the {len(rows)} evaluated rows")


def release_utility(out, D, rows, u_out=None, procedure="selection", role=None):
    """{"0": income metrics, "1": occupation metrics} of a release on `rows`, U-anchored when u_out is given. rows: a
    role name (its labels read through the allowlist under `procedure`), or explicit positions inside `role` (which
    must itself be allowed for `procedure`; e.g. a bootstrap subset)."""
    out = as_out(out)
    u_out = as_out(u_out) if u_out is not None else None
    n = len(D["row_id"])
    res = {}
    for i, t in enumerate(TASKS):
        if isinstance(rows, str):
            r, y = task_labels(D, i, procedure, rows)
        else:
            if role is None:
                raise ValueError("explicit rows need the role they belong to")
            allowed, _ = task_labels(D, i, procedure, role)
            r = np.asarray(rows)
            if not np.isin(r, allowed).all():
                raise PermissionError(f"{procedure}: explicit rows are not all inside {role}")
            y = DU._labels_ok(np.asarray(D["y"][t])[r])
        cst, prior = constant_class(D, i), constant_prior(D, i)
        u = None
        if u_out is not None:
            u = metrics(_on_rows(u_out[f"p{i + 1}"], r, n), _on_rows(u_out[f"hard{i + 1}"], r, n), y, KS[i], cst,
                        prior)
        res[str(i)] = metrics(_on_rows(out[f"p{i + 1}"], r, n), _on_rows(out[f"hard{i + 1}"], r, n), y, KS[i], cst,
                              prior, u)
    return res


def inner_release_utility(out, D, u_out=None):
    """release_utility on INNER_SELECTION (the gate rows; allowlist procedure "selection")."""
    rows = _role_index(D, SEL_ROLE)
    if "role" in D and np.any(np.asarray(D["role"])[rows] != SEL_ROLE):
        raise ValueError("REFUSED: INNER_SELECTION positions carry another role")
    r = release_utility(out, D, SEL_ROLE, u_out, procedure="selection")
    for v in r.values():
        v["role"] = SEL_ROLE
    return r


def release_inner_utility(probs, hard, D, rows="INNER_SELECTION", u=None):
    """Lead API (qpc.gate / qpc.select). probs = {1: (n, 2), 2: (n, 6)}, hard = {1: (n,), 2: (n,)} over all D rows
    (or exactly the evaluated rows) -> {"income": metrics, "occupation": metrics}; metrics include acc, logloss, brier,
    balanced_acc, recalls, ece, const_class, const_acc, gain. INNER_SELECTION labels are read through the allowlist
    (procedure "selection"). u (optional) = U's {probs, hard} pair to add the U-anchored receipts."""
    if rows != SEL_ROLE:
        raise PermissionError("release_inner_utility evaluates INNER_SELECTION only")
    out = {"p1": probs[1], "p2": probs[2], "hard1": hard[1], "hard2": hard[2]}
    uo = None if u is None else {"p1": u[0][1], "p2": u[0][2], "hard1": u[1][1], "hard2": u[1][2]}
    r = inner_release_utility(out, D, uo)
    return {OUT_NAMES[0]: r["0"], OUT_NAMES[1]: r["1"]}


# ------------------------------------------------------------------ gates
def _dec_flag(decision_preserved, t):
    if isinstance(decision_preserved, dict):
        if "tasks" in decision_preserved:                      # a decision_preservation() record
            return bool(decision_preserved["tasks"][t]["ok"])
        return bool(decision_preserved.get(t, decision_preserved.get(int(t), False)))
    if decision_preserved is None:
        raise ValueError("REFUSED: decision preservation must be passed explicitly (computed from the release)")
    return bool(decision_preserved)


def gate(cand, u, decision_preserved):
    """One seed: cand, u = {"0": metrics, "1": metrics} on the same rows; decision_preserved = bool, {"0","1"} -> bool,
    or a decision_preservation() record. Returns margins, shortfalls, headroom, identities and pass flags per task."""
    out = {}
    for t in TASK_KEYS:
        c, uu = cand[t], u[t]
        const = c["const_acc"]
        if abs(const - uu["const_acc"]) > 1e-15:
            raise ValueError("candidate and U were evaluated on different rows (constant accuracies differ)")
        ug = uu["acc"] - const
        gain = c["acc"] - const
        m = {"G_acc": c["acc"] - (uu["acc"] - ALLOW_ACC),
             "G_ll": (uu["logloss"] + ALLOW_LL) - c["logloss"],
             "G_brier": (uu["brier"] + ALLOW_BRIER) - c["brier"],
             "G_ret": gain - RETAIN * ug,
             "G_gain": gain - MIN_GAIN}
        dec = _dec_flag(decision_preserved, t)
        h = {"H_ll": (uu["logloss"] + HEAD_LL) - c["logloss"], "H_brier": (uu["brier"] + HEAD_BRIER) - c["brier"]}
        ex_ll, ex_br = c["logloss"] - uu["logloss"], c["brier"] - uu["brier"]
        out[t] = {"margins": m, "shortfalls": {g: max(0.0, -v) for g, v in m.items()},
                  "G_dec": dec, "numeric_pass": all(v >= 0 for v in m.values()),
                  "pass": dec and all(v >= 0 for v in m.values()),
                  "worst_gate": min(m, key=m.get), "worst_margin": min(m.values()),
                  "headroom_margins": h, "headroom": all(v >= 0 for v in h.values()),
                  "excess_logloss": ex_ll, "excess_brier": ex_br,
                  "normalised_excess": max(ex_ll / ALLOW_LL, ex_br / ALLOW_BRIER),
                  "receipts": {"acc": c["acc"], "u_acc": uu["acc"], "const_acc": const, "gain": gain, "u_gain": ug,
                               "retention": gain / ug if ug > 0 else None, "acc_minus_u": c["acc"] - uu["acc"],
                               "gain_minus_u_gain": gain - ug, "acc_identity_exact": bool(c["acc"] == uu["acc"]),
                               "logloss": c["logloss"], "u_logloss": uu["logloss"], "brier": c["brier"],
                               "u_brier": uu["brier"]}}
    return {"tasks": out, "pass": all(v["pass"] for v in out.values()),
            "decision_preserved": all(v["G_dec"] for v in out.values()),
            "headroom": all(v["headroom"] for v in out.values()),
            "worst_margin": min(v["worst_margin"] for v in out.values()),
            "normalised_excess": max(v["normalised_excess"] for v in out.values()),
            "total_shortfall": float(sum(sum(v["shortfalls"].values()) for v in out.values()))}


def gate_record(code_u, U_u, preserved):
    """Lead API (ONE seed). code_u, U_u = {"income": metrics, "occupation": metrics} on the same rows (from
    release_inner_utility); preserved = {1: bool, 2: bool} computed from the release. Inclusive comparisons exactly as
    registered. Returns {"income": {...}, "occupation": {...}, "eligible": bool, "headroom": bool, ...}; per task:
    acc_ok, ll_ok, brier_ok, retention_ok, gain_ok, preserved, ll_excess, brier_excess, norm_excess, headroom_ll,
    headroom_brier, the signed margins and the accuracy-identity receipts."""
    cand = {"0": code_u[OUT_NAMES[0]], "1": code_u[OUT_NAMES[1]]}
    uu = {"0": U_u[OUT_NAMES[0]], "1": U_u[OUT_NAMES[1]]}
    dec = {"0": _pres(preserved, 1), "1": _pres(preserved, 2)}
    g = gate(cand, uu, dec)
    out = {}
    for t, name in zip(TASK_KEYS, OUT_NAMES):
        v = g["tasks"][t]
        m, h = v["margins"], v["headroom_margins"]
        out[name] = {"acc_ok": m["G_acc"] >= 0, "ll_ok": m["G_ll"] >= 0, "brier_ok": m["G_brier"] >= 0,
                     "retention_ok": m["G_ret"] >= 0, "gain_ok": m["G_gain"] >= 0, "preserved": v["G_dec"],
                     "ll_excess": v["excess_logloss"], "brier_excess": v["excess_brier"],
                     "norm_excess": v["normalised_excess"],
                     "headroom_ll": h["H_ll"] >= 0, "headroom_brier": h["H_brier"] >= 0,
                     "eligible": v["pass"], "margins": m, "headroom_margins": h, "shortfalls": v["shortfalls"],
                     "worst_gate": v["worst_gate"], "receipts": v["receipts"]}
    return {**out, "eligible": g["pass"], "headroom": g["headroom"], "norm_excess": g["normalised_excess"],
            "total_shortfall": g["total_shortfall"], "rule": RULE, "headroom_rule": HEADROOM_RULE}


def _pres(preserved, i):
    if not isinstance(preserved, dict):
        raise ValueError("REFUSED: preserved must be {1: bool, 2: bool}")
    v = preserved.get(i, preserved.get(str(i)))
    if v is None:
        raise ValueError(f"REFUSED: decision preservation of recipient {i} missing")
    return bool(v)


def gate_all_seeds(cand_by_seed, u_by_seed, dec_by_seed):
    """Eligibility: every gate, each task, EACH seed (dicts keyed by seed; a missing seed fails). Also the headroom
    preference on all seeds, the worst-seed normalised excess, worst-seed log losses (rate-selection tie-breaks) and
    the summed task log loss averaged over seeds (nominee ordering)."""
    ks = sorted(set(map(str, cand_by_seed)) | set(map(str, u_by_seed)) | set(map(str, dec_by_seed)))
    by = lambda d: {str(k): v for k, v in d.items()}  # noqa: E731
    C, U, Dd = by(cand_by_seed), by(u_by_seed), by(dec_by_seed)
    missing = [k for k in ks if k not in C or k not in U or k not in Dd]
    per = {k: gate(C[k], U[k], Dd[k]) for k in ks if k not in missing}
    have = bool(per)
    wl = {t: max(C[k][t]["logloss"] for k in per) for t in TASK_KEYS} if have else None
    return {"per_seed": per, "seeds": sorted(per), "missing_seeds": missing,
            "eligible": have and not missing and all(v["pass"] for v in per.values()),
            "headroom_all": have and not missing and all(v["headroom"] for v in per.values()),
            "decision_preserved_all": have and not missing and all(v["decision_preserved"] for v in per.values()),
            "min_margin": min(v["worst_margin"] for v in per.values()) if have else None,
            "total_shortfall": float(sum(v["total_shortfall"] for v in per.values())),
            "normalised_excess": max(v["normalised_excess"] for v in per.values()) if have else None,
            "worst_seed_logloss": wl,
            "mean_summed_logloss": float(np.mean([C[k]["0"]["logloss"] + C[k]["1"]["logloss"] for k in per]))
            if have else None,
            "rule": RULE, "headroom_rule": HEADROOM_RULE,
            "normalised_excess_rule": "max over tasks and seeds of max(logloss excess / 0.01, brier excess / 0.005)"}


def class_preserved(hard, teacher_d):
    """Pointwise class preservation of one recipient (dpc.utility.class_preserved)."""
    return DU.class_preserved(hard, teacher_d)


# ------------------------------------------------------------------ one-call convenience for the gate / selection
def inner_gate_release(release, teacher_own, u_teacher, D):
    """Inner utility of `release` vs U (same seed) on INNER_SELECTION, decision preservation against its own teacher
    on all rows, and the one-seed gate. Returns {"utility", "u_utility", "decision_preservation", "gate"}."""
    util = inner_release_utility(release, D, u_out=u_teacher)
    uu = inner_release_utility(u_teacher, D)
    dp = decision_preservation(release, teacher_own, D)
    return {"utility": util, "u_utility": uu, "decision_preservation": dp, "gate": gate(util, uu, dp)}


# ------------------------------------------------------------------ synthetic fixtures (tests and timing only)
def synthetic_task_D(n_fit=15434, n_head=1500, n_audit=6065, n_sel=2235, n_assess=13936, seed=0, p_sex=0.68):
    """Study-shaped synthetic D with task labels (income binary, occupation 6-class) and roles; assessment labels
    sealed to -1. Synthetic only (no real data)."""
    rng = np.random.default_rng(seed)
    sizes = {"OSF_DEFENSE_FIT": n_fit, "HEAD_VALIDATION": n_head, "AUDIT_FIT": n_audit, SEL_ROLE: n_sel,
             "OSF_DEVELOPMENT_ASSESSMENT": n_assess}
    n = sum(sizes.values())
    idx, role, o = {}, np.empty(n, dtype=object), 0
    for r, s in sizes.items():
        idx[r] = np.arange(o, o + s)
        role[o:o + s] = r
        o += s
    y1 = (rng.random(n) < 0.24).astype(np.int64)
    y2 = rng.choice(6, n, p=[0.28, 0.25, 0.2, 0.15, 0.1, 0.02]).astype(np.int64)
    a = idx["OSF_DEVELOPMENT_ASSESSMENT"]
    y1[a], y2[a] = -1, -1
    return {"row_id": np.arange(n, dtype=np.int64) * 7 + 3, "role": role.astype(str), "idx": idx,
            "y": {"income": y1, "occupation_group": y2}, "sex": (rng.random(n) < p_sex).astype(np.int64),
            "unit": np.arange(n, dtype=np.int64), "sealed": True}


def synthetic_teacher(D, seed=0, sharp=2.0):
    """Synthetic U-like teacher: logits correlated with the (unsealed) labels, decisions = argmax."""
    rng = np.random.default_rng(seed + 100)
    n = len(D["row_id"])
    t = {"row_id": np.asarray(D["row_id"])}
    for i, (k, K) in enumerate(zip(TASKS, KS), start=1):
        y = np.asarray(D["y"][k])
        yy = np.where(y < 0, rng.integers(0, K, n), y)
        L = rng.normal(0, 1, (n, K)) + sharp * np.eye(K)[yy] * (rng.random(n) < 0.8)[:, None]
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        t[f"p{i}"], t[f"d{i}"] = P, P.argmax(1)
    return t

"""Useful-score metrics and the inner task-eligibility gate (audit/baseline owner; prompt section 12).

Every arm (policy codes, source teachers, references, U) is measured with the SAME functions on the SAME rows:
  accuracy            mean(hard == y)
  log_loss            true-label log loss, -mean log clip(P[y], 1e-12, 1) (the pinned clipping; natural log)
  brier               multiclass Brier, mean over rows of sum_k (P_k - 1[y = k])^2 (the osf / smf convention)
  const               the OSF_DEFENSE_FIT-prior constant: majority class of the task labels on OSF_DEFENSE_FIT
                      (argmax of the bincount; ties -> lower class); const_accuracy = mean(y == const)
  gain_over_const     accuracy - const_accuracy
  gain_retention      accuracy - 0.8 accuracy(U) - 0.2 const_accuracy (needs U on the same rows)
  balanced_accuracy   mean recall over SUPPORTED classes (>= SUPPORT_MIN = 30 rows among the evaluated rows)
  per_class_recall    every class present in the evaluated rows (unsupported ones flagged, never dropped)
  ece_10bin           10 equal-width bins of max probability (lo, hi]; sum_b (n_b / n) |conf_b - acc_b|, where acc_b
                      is the accuracy of the RELEASED decision (= argmax of the released probabilities for every arm
                      of this study)
The fitting-prior PROBABILITY constant (OSF_DEFENSE_FIT class frequencies) is also scored (const_log_loss,
const_brier; descriptive only). No calibration is applied to any arm. KL-to-teacher is never used here.

Inner versions are computed on INNER_SELECTION only (`inner_release_utility`); `per_row` returns the per-person
arrays (ll, br, correct) used by the assessment bootstrap. Labels read: task labels of OSF_DEFENSE_FIT (constant
only) and of the evaluated rows; sealed (negative) labels are refused.

Gate (`gate`, `gate_all_seeds`), each task, each seed, all must hold (margin >= 0, no tolerance):
  G_acc     acc >= acc(U) - 0.01                                 margin = acc - (acc(U) - 0.01)
  G_ll      log_loss <= log_loss(U) + 0.01                       margin = log_loss(U) + 0.01 - log_loss
  G_brier   brier <= brier(U) + 0.005                            margin = brier(U) + 0.005 - brier
  G_ret     acc - const >= 0.8 (acc(U) - const)                  margin = (acc - const) - 0.8 (acc(U) - const)
  G_gain    acc - const >= 0.03                                  margin = (acc - const) - 0.03
Shortfall of a gate = max(0, -margin). Allowances are prospective for this study (registered before fitting).
"""
from __future__ import annotations

import numpy as np

CLIP = 1e-12
ECE_BINS = 10
SUPPORT_MIN = 30
KS = (2, 6)
TASKS = ("income", "occupation_group")
SEL_ROLE = "INNER_SELECTION"
ALLOW_ACC, ALLOW_LL, ALLOW_BRIER, RETAIN, MIN_GAIN = 0.01, 0.01, 0.005, 0.8, 0.03
GATES = ("G_acc", "G_ll", "G_brier", "G_ret", "G_gain")


def _labels_ok(y):
    y = np.asarray(y)
    if y.size and (y.min() < 0 or not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("REFUSED: sealed or invalid task labels on the evaluated rows")
    return y


def fit_index(D):
    for r in ("OSF_DEFENSE_FIT", "DEFENSE_FIT"):
        if r in D["idx"]:
            return np.asarray(D["idx"][r])
    raise KeyError("D has no OSF_DEFENSE_FIT role")


def constants(D):
    """{0: income constant, 1: occupation constant} = OSF_DEFENSE_FIT majority class; plus prior vectors."""
    tr = fit_index(D)
    out, prior = {}, {}
    for i, t in enumerate(TASKS):
        y = _labels_ok(np.asarray(D["y"][t])[tr])
        c = np.bincount(y, minlength=KS[i]).astype(np.float64)
        out[i] = int(np.argmax(c))
        prior[i] = c / c.sum()
    return out, prior


def per_row(P, y, K=None):
    """Per-person arrays: ll (true-label log loss, clipped), br (sum of squares), correct (argmax == y)."""
    P = np.asarray(P, dtype=np.float64)
    y = _labels_ok(y).astype(np.int64)
    K = K or P.shape[1]
    ll = -np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1.0))
    br = np.sum((P - np.eye(K)[y]) ** 2, 1)
    return {"ll": ll, "br": br, "correct": (P.argmax(1) == y)}


def ece(P, hard, y, bins=ECE_BINS):
    conf = np.asarray(P, dtype=np.float64).max(1)
    ok = (np.asarray(hard) == np.asarray(y)).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            c_, a_ = float(conf[m].mean()), float(ok[m].mean())
            e += m.mean() * abs(c_ - a_)
            rel.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "confidence": c_, "accuracy": a_})
    return float(e), rel


def metrics(P, hard, y, K, const, prior=None, u=None):
    """All useful-score metrics of one release on one row set. u = U's metrics dict on the same rows (optional)."""
    P = np.asarray(P, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    y = _labels_ok(y).astype(np.int64)
    if not np.array_equal(P.argmax(1), hard):
        dec_note = "released decision differs from argmax of released probabilities on some rows"
    else:
        dec_note = None
    pr = per_row(P, y, K)
    acc = float((hard == y).mean())
    cacc = float((y == const).mean())
    counts = {int(c): int((y == c).sum()) for c in range(K)}
    rec = {c: float((hard[y == c] == c).mean()) for c in range(K) if counts[c]}
    sup = [c for c in range(K) if counts[c] >= SUPPORT_MIN]
    e, rel = ece(P, hard, y)
    out = {"n": int(len(y)), "acc": acc, "logloss": float(pr["ll"].mean()), "brier": float(pr["br"].mean()),
           "const_class": int(const), "const_acc": cacc, "gain": acc - cacc,
           "balanced_acc": float(np.mean([rec[c] for c in sup])) if sup else None,
           "recalls": {str(c): v for c, v in rec.items()}, "class_counts": {str(c): v for c, v in counts.items()},
           "supported_classes": sup, "unsupported_classes": [c for c in range(K) if 0 < counts[c] < SUPPORT_MIN],
           "minority_class": min(sup, key=lambda c: counts[c]) if sup else None,
           "ece": e, "reliability": rel, "decision_note": dec_note}
    out["minority_recall"] = rec.get(out["minority_class"]) if out["minority_class"] is not None else None
    if prior is not None:
        Pc = np.repeat(np.asarray(prior, dtype=np.float64)[None], len(y), 0)
        pc = per_row(Pc, y, K)
        out["const_logloss"], out["const_brier"] = float(pc["ll"].mean()), float(pc["br"].mean())
    if u is not None:
        out["u_acc"] = u["acc"]
        out["gain_retention"] = acc - RETAIN * u["acc"] - (1 - RETAIN) * cacc
    return out


def release_utility(out, D, rows, u_out=None):
    """{"0": metrics income, "1": metrics occupation} of a release output dict {p1, p2, hard1, hard2} on `rows`."""
    cst, prior = constants(D)
    res = {}
    for i, t in enumerate(TASKS):
        y = np.asarray(D["y"][t])[rows]
        u = None
        if u_out is not None:
            u = metrics(np.asarray(u_out[f"p{i + 1}"])[rows], np.asarray(u_out[f"hard{i + 1}"])[rows], y, KS[i],
                        cst[i], prior[i])
        res[str(i)] = metrics(np.asarray(out[f"p{i + 1}"])[rows], np.asarray(out[f"hard{i + 1}"])[rows], y, KS[i],
                              cst[i], prior[i], u)
    return res


def inner_release_utility(out, D, u_out=None):
    """release_utility on INNER_SELECTION (gates). out: {p1, p2, hard1, hard2} over all D rows."""
    rows = np.asarray(D["idx"][SEL_ROLE])
    if "role" in D and np.any(np.asarray(D["role"])[rows] != SEL_ROLE):
        raise ValueError("REFUSED: INNER_SELECTION positions carry another role")
    r = release_utility(out, D, rows, u_out)
    for v in r.values():
        v["role"] = SEL_ROLE
    return r


# ------------------------------------------------------------------ gates
def gate(cand, u):
    """Per task margins of one seed: cand, u = {"0": metrics, "1": metrics} on the same rows."""
    out = {}
    for t in ("0", "1"):
        c, uu = cand[str(t)], u[str(t)]
        const = c["const_acc"]
        if abs(const - uu["const_acc"]) > 1e-15:
            raise ValueError("candidate and U were evaluated on different rows (constant accuracies differ)")
        m = {"G_acc": c["acc"] - (uu["acc"] - ALLOW_ACC),
             "G_ll": (uu["logloss"] + ALLOW_LL) - c["logloss"],
             "G_brier": (uu["brier"] + ALLOW_BRIER) - c["brier"],
             "G_ret": (c["acc"] - const) - RETAIN * (uu["acc"] - const),
             "G_gain": (c["acc"] - const) - MIN_GAIN}
        out[t] = {"margins": m, "shortfalls": {g: max(0.0, -v) for g, v in m.items()},
                  "pass": all(v >= 0 for v in m.values()), "worst_gate": min(m, key=m.get),
                  "worst_margin": min(m.values())}
    return {"tasks": out, "pass": all(v["pass"] for v in out.values()),
            "worst_margin": min(v["worst_margin"] for v in out.values()),
            "total_shortfall": float(sum(sum(v["shortfalls"].values()) for v in out.values()))}


def gate_all_seeds(cand_by_seed, u_by_seed):
    """Inner task eligibility: every gate, each task, EACH seed (dicts keyed by seed)."""
    per = {str(k): gate(cand_by_seed[k], u_by_seed[k]) for k in sorted(cand_by_seed, key=str)}
    missing = sorted(set(map(str, u_by_seed)) ^ set(map(str, cand_by_seed)))
    return {"per_seed": per, "missing_seeds": missing,
            "eligible": bool(per) and not missing and all(v["pass"] for v in per.values()),
            "min_margin": min(v["worst_margin"] for v in per.values()) if per else None,
            "total_shortfall": float(sum(v["total_shortfall"] for v in per.values())),
            "rule": "acc >= U-0.01; logloss <= U+0.01; brier <= U+0.005; gain >= 0.8 gain(U); gain >= 0.03; each "
                    "task, each seed (margin >= 0)"}


def class_preserved(hard, teacher_d):
    """Pointwise class preservation: the released decision equals the teacher decision on every row."""
    h, d = np.asarray(hard), np.asarray(teacher_d)
    return {"ok": bool(h.shape == d.shape and np.array_equal(h, d)), "mismatches": int((h != d).sum())}

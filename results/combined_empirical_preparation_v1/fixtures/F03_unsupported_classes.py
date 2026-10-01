"""F3 -- unsupported / near-singleton classes in per-class and one-hot scores.

Independent reporter written from definitions. For each DECLARED class k of a
fixed schema [0, K): support_k = #{y == k}. Status:
  absent        support_k == 0                    -> R2_k not estimable
  constant      support_k == n                    -> complement empty, not estimable
  too_few       0 < support_k < MIN_SUPPORT       -> reported with counts, not scored
  estimable     otherwise                         -> in-sample ridge R2 (lam 1e-6)
MIN_SUPPORT = d + 2 (fewer positives than parameters+1 makes the in-sample
indicator R2 a leverage statistic of individual rows). This threshold is a
reporting choice of this fixture, not a PCRL rule.
Aggregate (one-hot, dominant axis) is reported as NOT ESTIMABLE unless every
declared class is estimable; an 'observed over estimable classes' value is given
separately and labelled as such.

Also demonstrates that a singleton class's in-sample R2 is the leverage of one
row: with pure-noise H, moving the singleton row to an outlying position drives
its R2 toward 1.
"""
import hashlib
import json

import numpy as np

LAM = 1e-6


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def r2_indicator(H, t):
    Hc = H - H.mean(0)
    tc = t - t.mean()
    den = (tc ** 2).sum()
    if den == 0:
        return float("nan")
    w = np.linalg.solve(Hc.T @ Hc + LAM * np.eye(H.shape[1]), Hc.T @ tc)
    return float(1 - ((tc - Hc @ w) ** 2).sum() / den)


def report(H, y, K):
    n, d = H.shape
    min_support = d + 2
    rows = []
    for k in range(K):
        s = int((y == k).sum())
        if s == 0:
            status = "absent"
        elif s == n:
            status = "constant"
        elif s < min_support or n - s < min_support:
            status = "too_few"
        else:
            status = "estimable"
        r2 = r2_indicator(H, (y == k).astype(float)) if status == "estimable" else None
        raw = r2_indicator(H, (y == k).astype(float)) if s not in (0, n) else None
        rows.append({"class": k, "support": s, "status": status, "r2_if_estimable": r2,
                     "raw_in_sample_r2_unreliable": raw})
    est = [r for r in rows if r["status"] == "estimable"]
    all_ok = len(est) == K
    return {"n": n, "d": d, "declared_K": K, "min_support_rule": "d+2 positives and negatives",
            "per_class": rows, "coverage_complete": all_ok,
            "dominant_axis": (max(r["r2_if_estimable"] for r in est) if all_ok else "NOT_ESTIMABLE"),
            "observed_max_over_estimable_classes_only": (max(r["r2_if_estimable"] for r in est) if est else None)}


def make_case(name):
    rng = np.random.default_rng({"absent_top_and_singleton": 31, "absent_middle": 32,
                                 "constant": 33, "singleton_outlier": 34}[name])
    n, d, K = 200, 5, 4
    H = rng.normal(size=(n, d))
    if name == "absent_top_and_singleton":
        y = rng.integers(0, 2, n)
        y[17] = 2                                         # class 2 singleton, class 3 absent
    elif name == "absent_middle":
        y = rng.choice([0, 2, 3], size=n)                 # class 1 absent
    elif name == "constant":
        y = np.zeros(n, dtype=np.int64)
    elif name == "singleton_outlier":
        y = rng.integers(0, 2, n)
        y[17] = 2
        H[17] = 8.0                                       # same singleton, now an outlying row
    else:
        raise ValueError(name)
    return H, y.astype(np.int64), K


CASES = ["absent_top_and_singleton", "absent_middle", "constant", "singleton_outlier"]


def main():
    out = {"id": "F3", "cases": {}}
    for name in CASES:
        H, y, K = make_case(name)
        rep = report(H, y, K)
        rep["sha_H"], rep["sha_y"] = sha(H), sha(y)
        out["cases"][name] = rep
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

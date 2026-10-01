"""F9 -- the 20-observation counterexample to the withdrawn accuracy guarantee.

Withdrawn claim (origin/main pcrl/purposes/verification.py:208-358,
'Linear Compliance Guarantee'): R2 < eps implies any linear classifier has
accuracy <= max(pi_maj, pi_maj + sqrt(eps*k*pi_maj*(1-pi_maj))); at R2 = 0 this
is the majority rate.

Independent check (exact arithmetic with fractions, plus float64 OLS):
  A=1: h=+1 (x9), h=-9 (x1);   A=0: h=-1 (x9), h=+9 (x1)   (PCRL doc example)
  and a second example of my own: A=0: h=0 (x9), h=9 (x1); A=1: h=1 (x9), h=0 (x1).
Both have Cov(h, A) = 0 exactly => affine least-squares R2 = 0, yet one linear
threshold classifies 18/20 = 90%.
"""
import hashlib
import json
from fractions import Fraction as Fr

import numpy as np


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def doc_example():
    h = np.array([1.0] * 9 + [-9.0] + [-1.0] * 9 + [9.0])
    A = np.array([1] * 10 + [0] * 10, dtype=np.int64)
    return h, A


def own_example():
    h = np.array([0.0] * 9 + [9.0] + [1.0] * 9 + [0.0])
    A = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    return h, A


def analyse(h, A, thresholds):
    n = len(h)
    hf = [Fr(int(x)) for x in h]
    af = [Fr(int(x)) for x in A]
    mh, ma = sum(hf) / n, sum(af) / n
    cov = sum((x - mh) * (y - ma) for x, y in zip(hf, af)) / n
    X = np.c_[np.ones(n), h]
    b, *_ = np.linalg.lstsq(X, A.astype(float), rcond=None)
    res = A - X @ b
    r2 = 1 - (res ** 2).sum() / ((A - A.mean()) ** 2).sum()
    best = None
    for t in thresholds:
        for sign in (1, -1):
            acc = float(np.mean((sign * (h - t) > 0).astype(int) == A))
            if best is None or acc > best[0]:
                best = (acc, t, sign)
    return {"n": n, "exact_cov_h_A": str(cov), "ols_coefficients": b.tolist(), "ols_r2": float(r2),
            "best_threshold_accuracy": best[0], "threshold": best[1], "direction": best[2],
            "majority_rate": float(max(A.mean(), 1 - A.mean())),
            "withdrawn_bound_at_r2_0": float(max(A.mean(), 1 - A.mean()))}


def main():
    out = {"id": "F9"}
    h, A = doc_example()
    out["doc_example"] = analyse(h, A, thresholds=np.unique(np.r_[h, (h[:, None] + h[None, :]).ravel() / 2]))
    out["doc_example"]["sha_h"] = sha(h)
    h2, A2 = own_example()
    out["own_example"] = analyse(h2, A2, thresholds=np.unique(np.r_[h2, (h2[:, None] + h2[None, :]).ravel() / 2]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

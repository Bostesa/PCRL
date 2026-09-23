"""Interval-arithmetic bracket for the information radius R(Q) = inf_r max_t KL(Q_t || r) = capacity.

Rigour model:
* Each stored float64 kernel entry is an exact dyadic rational. The certified object is the
  row-normalised exact-rational kernel (row sums differ from 1 by at most a few ulps; the
  normalisation is reported).
* The UPPER end is max_t KL(Q_t || r) for a supplied reference r (Terminal 4's, from
  EXISTING_RADIUS.json, also normalised exactly). Every logarithm and product is evaluated in
  mpmath interval arithmetic (mpmath.iv), so the returned upper end is a rigorous upper bound on
  R(Q) for that kernel.
* The LOWER end is I_pi(T;Z) = sum_t pi_t KL(Q_t || pi Q) for a supplied input prior pi (normalised
  exactly). It is a rigorous lower bound on capacity = R(Q).
The bracket is therefore interval-certified, conditional only on the correctness of mpmath's
interval primitives. It says nothing about the true population law.

Usage: python interval_radius.py EXISTING_RADIUS.json KERNEL_ROOT
KERNEL_ROOT holds anchor_{a}/maps/{cfg}/Q.npz. The kernels are private archive members, not
committed. Only aggregate brackets are printed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv, mp

iv.dps = 50
mp.dps = 50
CFG = {"Q": "T0_L_0.01_a17", "D17": "T0_U_unconstrained_a17"}


def exact_rows(M):
    rows = []
    for row in M:
        fr = [Fraction(float(x)) for x in row]
        s = sum(fr)
        rows.append([x / s for x in fr])
    return rows


def exact_vec(v):
    fr = [Fraction(float(x)) for x in v]
    s = sum(fr)
    return [x / s for x in fr]


def ivf(fr):
    return iv.mpf(fr.numerator) / iv.mpf(fr.denominator)


def kl_iv(p, r):
    tot = iv.mpf(0)
    for a, b in zip(p, r):
        if a == 0:
            continue
        if b == 0:
            return None  # infinite
        A, B = ivf(a), ivf(b)
        tot += A * iv.log(A / B)
    return tot


def bracket(Qrows, ref, prior):
    upper_terms = [kl_iv(q, ref) for q in Qrows]
    if any(u is None for u in upper_terms):
        upper = None
    else:
        upper = max(float(u.b) for u in upper_terms)  # right endpoint of each interval
    nz = len(Qrows[0])
    mix = [sum(prior[t] * Qrows[t][z] for t in range(len(Qrows))) for z in range(nz)]
    lower_iv = iv.mpf(0)
    for t, q in enumerate(Qrows):
        if prior[t] == 0:
            continue
        k = kl_iv(q, mix)
        lower_iv += ivf(prior[t]) * k
    lower = float(lower_iv.a)  # left endpoint
    return lower, upper


def main(radius_json, root):
    spec = json.load(open(radius_json))
    out = {}
    for a, kernels in spec["anchors"].items():
        for name, rec in kernels.items():
            path = Path(root) / f"anchor_{a}" / "maps" / CFG[name] / "Q.npz"
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            M = np.load(path)["Q"]
            Qrows = exact_rows(M)
            max_row_sum_err = float(np.abs(M.sum(1) - 1).max())
            ref = exact_vec(rec["reference"])
            prior = exact_vec([max(0.0, x) for x in rec["input_prior"]])
            lo, hi = bracket(Qrows, ref, prior)
            used = int((M > 0).any(0).sum())
            deterministic = bool(np.all((M == 0) | (M == 1)))
            out[f"{a}/{name}"] = {
                "sha256_matches_T4": sha == rec["file_sha256"],
                "interval_lower_capacity": lo, "interval_upper_radius": hi,
                "T4_lower": rec["lower"], "T4_upper": rec["upper"],
                "T4_bracket_contains_interval_bracket": (rec["lower"] <= lo + 1e-12) and (hi <= rec["upper"] + 1e-12),
                "outputs_with_positive_mass": used, "deterministic": deterministic,
                "log_used_outputs": float(np.log(used)), "max_row_sum_error": max_row_sum_err,
            }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

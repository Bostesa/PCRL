"""Distribution-free release guarantees for a fixed public kernel Q(z|t).

Input: a JSON file {"Q": [[...], ...]} (rows = codes t, columns = tokens z), or {"states": {b: Q_b}} for
kernels that depend on a public bin b(H_A). Codes with zero probability may be omitted with
{"support": [t, ...]}. No individual-level data is needed. The kernel itself is enough.

Output, per state and overall (in nats):
  radius   = min_r max_t KL(Q_t || r) = capacity (Blahut-Arimoto; the certificate is the attained max
             over rows against the BA output mixture, which is a valid upper bound for any r)
  eta_TV   = max_{t,t'} TV(Q_t, Q_t')  (Dobrushin)
  bounds   I(S;Z|W) <= radius for every side information W that contains the state;
           I(S;Z|H) <= eta_TV * log|S|;  I(Y;Z|H) <= radius (Bayes task log-loss gain over H)
compared with log 2 (SEX) and log 9 (RAC1P).

Usage: python kernel_guarantees.py kernel.json
"""
from __future__ import annotations

import json
import math
import sys


def kl(p, q):
    s = 0.0
    for a, b in zip(p, q):
        if a > 0:
            if b <= 0:
                return math.inf
            s += a * math.log(a / b)
    return s


def radius_certificate(Q, iters=20000, tol=1e-12):
    nt, nz = len(Q), len(Q[0])
    p = [1.0 / nt] * nt
    for _ in range(iters):
        q = [sum(p[t] * Q[t][z] for t in range(nt)) for z in range(nz)]
        d = [kl(Q[t], q) for t in range(nt)]
        lower = sum(p[t] * d[t] for t in range(nt))
        upper = max(d)
        if upper - lower < tol:
            break
        w = [p[t] * math.exp(d[t]) for t in range(nt)]
        s = sum(w)
        p = [x / s for x in w]
    return {"capacity_lower": lower, "radius_upper_certificate": upper, "gap": upper - lower}


def dobrushin(Q):
    return max(0.5 * sum(abs(a - b) for a, b in zip(Q[t], Q[u])) for t in range(len(Q)) for u in range(len(Q)))


def analyse(Q):
    for row in Q:
        if abs(sum(row) - 1) > 1e-9 or min(row) < -1e-15:
            raise ValueError("rows must be probability vectors")
    rc = radius_certificate(Q)
    eta = dobrushin(Q)
    kappa = rc["radius_upper_certificate"]
    return {**rc, "eta_TV": eta,
            "bound_any_side_info_nats": kappa,
            "bound_sdpi_SEX_nats": eta * math.log(2), "bound_sdpi_RAC1P_nats": eta * math.log(9),
            "utility_ceiling_bayes_gain_nats": kappa,
            "informative_vs_log2": kappa < math.log(2), "informative_vs_log9": kappa < math.log(9)}


def main(path):
    spec = json.load(open(path))
    if "states" in spec:
        per = {b: analyse(Qb) for b, Qb in spec["states"].items()}
        out = {"per_state": per,
               "overall_bound_any_view_containing_state": max(v["bound_any_side_info_nats"] for v in per.values())}
    else:
        Q = spec["Q"]
        if "support" in spec:
            Q = [Q[t] for t in spec["support"]]
        out = analyse(Q)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

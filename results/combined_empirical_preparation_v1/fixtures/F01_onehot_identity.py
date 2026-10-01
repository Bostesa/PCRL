"""F1 -- one-hot aggregation identity (independent; numpy only).

Definitions (written from scratch, not imported):
  Y   one-hot (n x K) of a categorical attribute, Yc = Y - colmean(Y)
  Hc  representation centred per column (affine fit == centred fit)
  B   ridge solution (Hc'Hc + lam I)^-1 Hc'Yc   (lam = 1e-6, as in PCRL)
  SSres_k = ||Yc_k - Hc B_k||^2 ,  SStot_k = ||Yc_k||^2 = n pi_k (1 - pi_k)
  R2_k    = 1 - SSres_k / SStot_k                  (per-indicator, OvR)
  R2_agg  = 1 - sum_k SSres_k / sum_k SStot_k      (multi-output one-hot R2)

Identity claimed by PCRL (NeurIPS draft Prop. 4 'Convex-Combination Identity',
origin/main pcrl/evaluation/certificates.py:30-45):
  R2_agg = sum_k w_k R2_k ,  w_k = pi_k(1-pi_k) / sum_j pi_j(1-pi_j)
Algebra: R2_agg = 1 - sum_k (SStot_k/sum SStot) (SSres_k/SStot_k) = sum_k w_k R2_k,
with w_k = SStot_k / sum_j SStot_j, which equals the prior-based weight only when
SStot_k is computed with the SAME rows and the 1/n (population) convention.
It requires (i) the same per-column estimator in both quantities (multi-output
least squares decomposes by column, so this holds for OLS and ridge), (ii) every
SStot_k > 0, and (iii) no per-class clipping of negative R2_k (clipping matters
only out of sample, where R2_k < 0 is possible).

Prints one JSON object. Deterministic. Runtime: < 2 s.
"""
import hashlib
import json
import sys

import numpy as np

LAM = 1e-6


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def ridge_cols(H_fit, Y_fit, H_eval=None, Y_eval=None, lam=LAM, gram_dtype=np.float64):
    """Per-column SSres/SStot. Fit on (H_fit,Y_fit); evaluate on eval rows (default: fit rows).
    gram_dtype=float32 reproduces a float32 Gram accumulation (precision path only)."""
    H_fit = np.asarray(H_fit)
    mu_h = H_fit.mean(0, keepdims=True)
    mu_y = Y_fit.mean(0, keepdims=True)
    Hc = (H_fit - mu_h)
    if gram_dtype == np.float32:
        Hc32 = Hc.astype(np.float32)
        G = (Hc32.T @ Hc32).astype(np.float64)
        Hc = Hc32.astype(np.float64)
    else:
        Hc = Hc.astype(np.float64)
        G = Hc.T @ Hc
    G = G + lam * np.eye(G.shape[0])
    B = np.linalg.solve(G, Hc.T @ (Y_fit - mu_y))
    if H_eval is None:                      # in-sample
        ssres = ((Y_fit - mu_y - Hc @ B) ** 2).sum(0)
        sstot = ((Y_fit - mu_y) ** 2).sum(0)
    else:                                   # held-out: affine predictor fitted on fit rows
        pred = (np.asarray(H_eval, dtype=np.float64) - mu_h) @ B + mu_y
        ssres = ((Y_eval - pred) ** 2).sum(0)
        sstot = ((Y_eval - Y_eval.mean(0, keepdims=True)) ** 2).sum(0)
    return ssres, sstot


def make_data(seed=11, n=3000, d=6, priors=(0.6, 0.25, 0.1, 0.05)):
    rng = np.random.default_rng(seed)
    K = len(priors)
    y = rng.choice(K, size=n, p=priors)
    means = rng.normal(size=(K, d)) * np.array([0.6, 0.3, 0.2, 0.1, 0.05, 0.0])
    means[3, 4] = 1.5                       # strong signal on the rare class
    H = means[y] + rng.normal(size=(n, d))
    return H, y


def make_stress(tail_sd, offset, seed=0, n=15000, d=64, ntail=61):
    rng3 = np.random.default_rng(seed)
    yb = rng3.integers(0, 2, n)
    Q, _ = np.linalg.qr(rng3.normal(size=(d, d)))
    sd = np.r_[np.ones(d - ntail), np.full(ntail, tail_sd)]
    Zl = rng3.normal(size=(n, d)) * sd
    Zl[:, d - ntail:] += (yb[:, None] - 0.5) * tail_sd * 0.3
    return (Zl @ Q.T + offset).astype(np.float32), yb


STRESS = [(1e-2, 20.0), (3e-3, 5.0), (1e-3, 5.0), (1e-3, 0.0)]


def identity_report(ssres, sstot, n):
    r2k = 1 - ssres / sstot
    agg = 1 - ssres.sum() / sstot.sum()
    pi = sstot / n                           # = pi_k (1-pi_k) under 1/n convention
    w = sstot / sstot.sum()
    return r2k, agg, w


def main():
    out = {"id": "F1"}
    H, y = make_data()
    n, K = len(y), 4
    Y = np.eye(K)[y]
    out["data"] = {"n": n, "d": H.shape[1], "K": K, "sha_H": sha(H), "sha_y": sha(y),
                   "class_counts": np.bincount(y, minlength=K).tolist()}

    # In-sample, float64
    ssres, sstot = ridge_cols(H, Y)
    r2k, agg, w = identity_report(ssres, sstot, n)
    pi = np.bincount(y, minlength=K) / n
    w_prior = pi * (1 - pi) / (pi * (1 - pi)).sum()
    out["in_sample_float64"] = {
        "per_class_r2": r2k.tolist(),
        "pi": pi.tolist(),
        "weights_from_sstot": w.tolist(),
        "weights_from_priors": w_prior.tolist(),
        "max_abs_weight_difference": float(np.abs(w - w_prior).max()),
        "r2_aggregate_direct": float(agg),
        "r2_convex_combination": float(w_prior @ r2k),
        "identity_residual": float(agg - w_prior @ r2k),
        "dominant_axis_max_r2k": float(r2k.max()),
        "argmax_class": int(r2k.argmax()),
        "amplification_DA_over_agg": float(r2k.max() / agg),
        "ceiling_1_over_w_argmax": float(1 / w_prior[r2k.argmax()]),
    }
    # Plain OLS (no ridge) via lstsq with explicit intercept: ridge is immaterial here
    X1 = np.c_[np.ones(n), H]
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    res = Y - X1 @ B
    Yc = Y - Y.mean(0)
    r2_ols = 1 - (res ** 2).sum(0) / (Yc ** 2).sum(0)
    out["in_sample_float64"]["ols_vs_ridge_max_abs_diff_r2k"] = float(np.abs(r2_ols - r2k).max())

    # Out of sample: identity still algebraic if weights use EVAL variances and no clipping
    rng = np.random.default_rng(5)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    # make class 3 signal not generalise for one column to force a negative R2_k
    ssr, sst = ridge_cols(H[tr], Y[tr], H[te], Y[te])
    r2k_te = 1 - ssr / sst
    agg_te = 1 - ssr.sum() / sst.sum()
    pi_te = Y[te].mean(0)
    w_te = pi_te * (1 - pi_te) / (pi_te * (1 - pi_te)).sum()
    pi_tr = Y[tr].mean(0)
    w_tr = pi_tr * (1 - pi_tr) / (pi_tr * (1 - pi_tr)).sum()
    out["held_out"] = {
        "per_class_r2_test": r2k_te.tolist(),
        "r2_aggregate_direct": float(agg_te),
        "convex_with_test_weights_unclipped": float(w_te @ r2k_te),
        "residual_test_weights_unclipped": float(agg_te - w_te @ r2k_te),
        "convex_with_train_weights": float(w_tr @ r2k_te),
        "residual_train_weights": float(agg_te - w_tr @ r2k_te),
        "convex_with_test_weights_clipped_at_0": float(w_te @ np.maximum(r2k_te, 0)),
        "residual_clipped": float(agg_te - w_te @ np.maximum(r2k_te, 0)),
        "any_negative_r2k": bool((r2k_te < 0).any()),
    }
    # pure-noise held-out case so negative per-class R2 is guaranteed to appear
    rng2 = np.random.default_rng(6)
    yn = rng2.choice(3, size=400, p=[0.5, 0.3, 0.2])
    Hn = rng2.normal(size=(400, 30))
    Hn[:, 0] += 3.0 * (yn == 0)            # signal for class 0 only; classes 1,2 go negative held out
    Yn = np.eye(3)[yn]
    ssr, sst = ridge_cols(Hn[:120], Yn[:120], Hn[120:], Yn[120:])
    r2n = 1 - ssr / sst
    pin = Yn[120:].mean(0)
    wn = pin * (1 - pin) / (pin * (1 - pin)).sum()
    aggn = 1 - ssr.sum() / sst.sum()
    out["held_out_mixed_sign"] = {
        "per_class_r2_test": r2n.tolist(),
        "r2_aggregate_direct_unclipped": float(aggn),
        "residual_unclipped": float(aggn - wn @ r2n),
        "pcrl_style_clipped_aggregate": float(max(0.0, aggn)),
        "pcrl_style_clipped_convex": float(wn @ np.maximum(r2n, 0)),
        "residual_after_pcrl_clipping": float(max(0.0, aggn) - wn @ np.maximum(r2n, 0)),
    }

    # Float32: (a) benign data cast to float32; (b) ill-conditioned stress case
    H32 = H.astype(np.float32)
    s64, t64 = ridge_cols(H32.astype(np.float64), Y)
    s32, t32 = ridge_cols(H32, Y, gram_dtype=np.float32)
    a64 = 1 - s64.sum() / t64.sum()
    a32 = 1 - s32.sum() / t32.sum()
    out["float32_benign"] = {"agg_float64_on_float32_inputs": float(a64),
                             "agg_float32_gram": float(a32),
                             "abs_diff": float(abs(a64 - a32))}
    stress = []
    for tail_sd, offset in STRESS:
        Hs, yb = make_stress(tail_sd, offset)
        Ys = np.eye(2)[yb]
        sa, ta = ridge_cols(Hs.astype(np.float64), Ys)
        sb, tb = ridge_cols(Hs, Ys, gram_dtype=np.float32)
        r64 = 1 - sa.sum() / ta.sum()
        r32 = 1 - sb.sum() / tb.sum()
        stress.append({"sha_H": sha(Hs), "tail_sd": tail_sd, "offset": offset, "eigen_ratio_min_over_max":
                       float(tail_sd ** 2), "r2_float64": float(r64), "r2_float32_gram": float(r32),
                       "r2_float32_gram_clipped_at_0": float(max(0.0, r32)),
                       "understatement": float(r64 - max(0.0, r32))})
    out["float32_stress"] = {"description": "binary attribute, d=64, 3 unit-variance dims, 61 low-variance dims "
                             "carrying the signal, random rotation, float32 storage; Gram accumulated in float32 "
                             "(the precision path of origin/main LinearComplianceCertificate when given float32 H)",
                             "cases": stress}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())

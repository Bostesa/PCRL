#!/usr/bin/env python3
"""Standalone synthetic fixture for the PCRL worst-class / dominant-axis audit.

No repository imports; numpy only. Re-implements (as literal replicas) the
scoring conventions found in Bostesa/PCRL origin/main@55e4cb1d1:
  * pooled one-hot R^2 = 1 - sum_k SSE_k / sum_k SST_k, K one-hot columns,
    centred, ridge 1e-6, fit AND scored in-sample on the supplied rows
    (pcrl/purposes/verification.py:89-103, LinearComplianceCertificate.check)
  * per-class OvR R^2 and R^2_DA = max_k (pcrl/evaluation/certificates.py:95-116)
  * training-time per-class R^2 on a minibatch, float32, ridge 1e-4
    (pcrl/training/losses.py:496-565) with tail-padding by zeros
    (pcrl/training/v2_trainer.py:856-867)

Sections
  A  convex-combination identity: when it holds, when it fails
  B  linear-contrast gap: max per-class OvR R^2 small, group contrast large
  C  estimator floors: minibatch in-sample null floor d/(B-1); ridge/scale
     shrinkage; float32 vs float64
  D  unsupported classes: absent middle class scored as R^2 = 1 (pre-fix),
     NaN (post-fix semantics); implied dual accumulation for Diabetes age k=0

Run:  python3 worst_class_fixture.py      (takes a few seconds)
"""
import json

import numpy as np


# ---------------------------------------------------------------- scorers
def pooled_r2(H, y, K=None, reg=1e-6, dtype=np.float64, clamp=True):
    """Replica of LinearComplianceCertificate.check (pre-fix)."""
    H = np.asarray(H, dtype=dtype)
    K = int(y.max()) + 1 if K is None else K
    Z = np.eye(K, dtype=dtype)[y]
    Hc = H - H.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)
    W = np.linalg.solve(Hc.T @ Hc + dtype(reg) * np.eye(H.shape[1], dtype=dtype), Hc.T @ Zc)
    r2 = 1.0 - ((Zc - Hc @ W) ** 2).sum() / max(float((Zc ** 2).sum()), 1e-12)
    return max(0.0, float(r2)) if clamp else float(r2)


def per_class_r2(H, y, K=None, reg=1e-6, dtype=np.float64):
    """Replica of compute_dominant_axis_r2 per-class loop (pre-fix)."""
    H = np.asarray(H, dtype=dtype)
    K = int(y.max()) + 1 if K is None else K
    Hc = H - H.mean(0, keepdims=True)
    G = np.linalg.solve(Hc.T @ Hc + dtype(reg) * np.eye(H.shape[1], dtype=dtype), Hc.T)
    out = []
    for k in range(K):
        z = (y == k).astype(dtype)
        zc = z - z.mean()
        zp = Hc @ (G @ zc)
        out.append(max(0.0, 1.0 - float(((zc - zp) ** 2).sum()) / max(float((zc ** 2).sum()), 1e-12)))
    return np.array(out)


def heldout_scores(Htr, ytr, Hte, yte, K, reg=1e-6):
    """Fit on train, score on test (test means for SST). Returns pooled and per-class (unclamped)."""
    mu = Htr.mean(0, keepdims=True)
    Ztr = np.eye(K)[ytr]
    Zte = np.eye(K)[yte]
    Hc = Htr - mu
    W = np.linalg.solve(Hc.T @ Hc + reg * np.eye(Htr.shape[1]), Hc.T @ (Ztr - Ztr.mean(0)))
    pred = (Hte - mu) @ W + Ztr.mean(0)
    sse = ((Zte - pred) ** 2).sum(0)
    sst = ((Zte - Zte.mean(0)) ** 2).sum(0)
    return 1 - sse.sum() / sst.sum(), 1 - sse / sst, Zte.mean(0)


def top_canonical_r2(H, y, K):
    """max over contrasts c of R^2(H, c^T z) = largest squared canonical correlation."""
    H = H - H.mean(0)
    Z = np.eye(K)[y][:, : K - 1]  # drop one column (simplex constraint)
    Z = Z - Z.mean(0)
    Shh = H.T @ H
    Szz = Z.T @ Z
    Shz = H.T @ Z
    M = np.linalg.solve(Szz, Shz.T @ np.linalg.solve(Shh, Shz))
    return float(np.max(np.real(np.linalg.eigvals(M))))


def weights(priors):
    w = priors * (1 - priors)
    return w / w.sum()


res = {}
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- A identity
n, d, K = 4000, 8, 5
pri = np.array([0.65, 0.06, 0.26, 0.02, 0.01])
y = rng.choice(K, size=n, p=pri)
H = rng.normal(size=(n, d))
H[:, 0] += 3.0 * (y == 4)  # leak only the rarest class
pc = per_class_r2(H, y)
emp = np.bincount(y, minlength=K) / n
A = {
    "pooled_onehot_r2": pooled_r2(H, y),
    "per_class_r2": pc.round(4).tolist(),
    "max_per_class_r2_DA": float(pc.max()),
    "identity_variance_weighted": float(weights(emp) @ pc),
    "identity_residual": float(weights(emp) @ pc - pooled_r2(H, y)),
    "uniform_average_(sklearn_default)": float(pc.mean()),
    "ridge_alpha_equal_n_identity_residual": float(
        weights(emp) @ per_class_r2(H, y, reg=n) - pooled_r2(H, y, reg=n)),
}
# K-1 encoding (drop last column): pooled score changes
Zkm1 = np.eye(K)[y][:, :-1]
Hc = H - H.mean(0)
Zc = Zkm1 - Zkm1.mean(0)
W = np.linalg.lstsq(Hc, Zc, rcond=None)[0]
A["pooled_r2_K_minus_1_columns_drop_rarest"] = float(1 - ((Zc - Hc @ W) ** 2).sum() / (Zc ** 2).sum())
# held-out: identity holds with test priors if per-class scores are NOT clamped
tr = rng.random(n) < 0.5
p_ho, pk_ho, pri_te = heldout_scores(H[tr], y[tr], H[~tr], y[~tr], K)
A["heldout_pooled_r2"] = float(p_ho)
A["heldout_identity_unclamped_residual"] = float(weights(pri_te) @ pk_ho - p_ho)
# small-sample held-out where some per-class R^2 are negative; clamping breaks identity
n2 = 300
y2 = rng.choice(K, size=n2, p=pri)
H2 = rng.normal(size=(n2, 40))
H2[:, 0] += 2.0 * (y2 == 0)  # class 0 generalises (positive held-out R^2); others overfit (negative)
tr2 = np.arange(n2) < 150
p2, pk2, pri2 = heldout_scores(H2[tr2], y2[tr2], H2[~tr2], y2[~tr2], K)
A["small_heldout_pooled_r2_unclamped"] = float(p2)
A["small_heldout_per_class_unclamped"] = np.round(pk2, 4).tolist()
A["small_heldout_identity_with_clamp_at_0_residual"] = float(weights(pri2) @ np.maximum(pk2, 0) - max(p2, 0))
res["A_identity"] = A

# ---------------------------------------------------------------- B contrast gap
def contrast_case(pri, group, target_r2_da, n=200_000, seed=1):
    r = np.random.default_rng(seed)
    K = len(pri)
    y = r.choice(K, size=n, p=pri)
    b = np.isin(y, group).astype(float)
    # choose noise so that max per-class OvR R^2 == target (population calc)
    q = b.mean()
    rk = []
    for k in range(K):
        pk = pri[k]
        cov = pk * (1 - q) if k in group else -pk * q
        rk.append(cov ** 2 / (q * (1 - q) * pk * (1 - pk)))
    rho2 = min(1.0, target_r2_da / max(rk))  # squared corr(h, b) needed
    sig2 = q * (1 - q) * (1 / rho2 - 1)
    h = b + r.normal(scale=np.sqrt(sig2), size=n)
    H = np.column_stack([h, r.normal(size=(n, 3))])
    pc = per_class_r2(H, y)
    hs = np.sort(h)  # best single threshold on h (either direction), in-sample
    cand = hs[:: max(1, n // 2000)]
    acc = max(max(((h > t) == b).mean(), ((h <= t) == b).mean()) for t in cand)
    acc = float(acc)
    return {
        "priors": [round(x, 4) for x in pri],
        "group": list(group),
        "pooled_onehot_r2": round(pooled_r2(H, y), 4),
        "max_per_class_r2_DA": round(float(pc.max()), 4),
        "top_canonical_r2_(best_linear_contrast)": round(top_canonical_r2(H, y, K), 4),
        "binary_group_best_threshold_accuracy": round(acc, 4),
        "binary_group_majority": round(max(q, 1 - q), 4),
    }

res["B_contrast_gap"] = {
    "balanced_K10_group_0to4_vs_5to9_noiseless": contrast_case(np.full(10, 0.1), list(range(5)), 1.0),
    "balanced_K10_calibrated_DA_0.045": contrast_case(np.full(10, 0.1), list(range(5)), 0.045),
    # Diabetes age_bucket priors after first-encounter dedup (computed from
    # data/diabetes/diabetic_data.csv; 71,518 rows). Group = age < 60.
    "diabetes_age_priors_group_age_lt_60_calibrated_DA_0.045": contrast_case(
        np.array([0.00215, 0.00748, 0.01576, 0.03774, 0.09617, 0.17431, 0.22316, 0.25462, 0.16204, 0.02657]) / 1.00000,
        list(range(6)), 0.045),
}

# ---------------------------------------------------------------- C estimator floors
B, d = 256, 64
yb = rng.choice(5, size=B, p=pri)
null = []
for r_ in (64, 32, 12, 4, 1):
    v32, v64 = [], []
    for _ in range(50):  # 50 independent minibatches; H independent of the attribute
        yb = rng.choice(5, size=B, p=pri)
        Hb = rng.normal(size=(B, r_)) @ rng.normal(size=(r_, d))
        v32.append(pooled_r2(Hb.astype(np.float32), yb, K=5, reg=1e-4, dtype=np.float32))
        v64.append(pooled_r2(Hb, yb, K=5, reg=1e-4))
    null.append({"rank": r_, "mean_batch_r2_float32_ridge1e-4": round(float(np.mean(v32)), 4),
                 "mean_batch_r2_float64": round(float(np.mean(v64)), 4),
                 "expected_rank_over_B_minus_1": round(r_ / (B - 1), 4)})
res["C_null_floor_minibatch_B256_d64_independent_of_A"] = null
# scale shrinkage: perfectly informative but tiny-scale representation
n3 = 13661
y3 = rng.choice(5, size=n3, p=pri)
Hinf = np.eye(5)[y3] @ rng.normal(size=(5, 64))
scale_rows = []
for s in (1.0, 1e-3, 1e-5, 1e-6):
    scale_rows.append({"scale": s,
                       "eval_r2_ridge1e-6_float64": round(pooled_r2(s * Hinf, y3, reg=1e-6), 4),
                       "eval_r2_ridge1e-6_float32_input": round(pooled_r2((s * Hinf + 0.18).astype(np.float32).astype(np.float64), y3, reg=1e-6), 4)})
yb2 = y3[:256]
for s in (1.0, 1e-3, 3e-4, 1e-4, 3e-5):
    scale_rows.append({"scale": s, "train_batch_r2_ridge1e-4_float32": round(pooled_r2((s * Hinf[:256]).astype(np.float32), yb2, K=5, reg=1e-4, dtype=np.float32), 4)})
res["C_scale_shrinkage_informative_representation"] = scale_rows

# ---------------------------------------------------------------- D unsupported classes
yd = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 28 + [5, 6, 7, 8])  # class 0 absent, B=256
Hd = rng.normal(size=(len(yd), 64)).astype(np.float32)
pcd32 = per_class_r2(Hd, yd, K=10, reg=1e-4, dtype=np.float32)
yt = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] * 28 + [5, 6, 7, 8])  # class 9 (tail) absent
pct = per_class_r2(Hd, yt, reg=1e-4, dtype=np.float32)  # length 9, then padded with 0
pct = np.concatenate([pct, np.zeros(10 - len(pct))])
p_absent = (1 - 0.00215) ** 256
res["D_unsupported_classes"] = {
    "absent_middle_class0_prefix_R2": float(pcd32[0]),
    "absent_tail_class9_prefix_R2_after_zero_pad": float(pct[9]),
    "postfix_semantics": "NaN score, valid_mask False, coverage_complete False, dual update skipped "
                         "(research/pcrl-submission-finish-v1:tests/test_scoring_support.py:15-27,62-74)",
    "diabetes_age_k0_prior_after_dedup": 0.00215,
    "P_class0_absent_in_batch_of_256": round(p_absent, 3),
    "implied_lambda_k0_after_200_epochs_x_196_steps_eta0.02_from_absent_batches_only":
        round(1 + 200 * 196 * 0.02 * p_absent * (1 - 0.05), 1),
    "stored_lambda_k0_R7_seeds_0_1_2": [420.2, 419.4, 420.2],
}

print(json.dumps(res, indent=1))

"""Synthetic checks for SCOPE_CORRECTIONS items A (composition, Prop 6) and D (scale dependence).

Synthetic data only. No real rows, no checkpoints. Run with system python3 (numpy only):
    python3 scale_composition_check.py > scale_composition_check_output.json

The R^2 function reproduces the arithmetic of PCRL origin/main@55e4cb1d
pcrl/purposes/verification.py:LinearComplianceCertificate.check (lines ~76-100) and
pcrl/training/v2_trainer.py:_linear_r2_train (lines 282-302): per-variable centering of H and of the
one-hot Z, Tikhonov penalty reg*I added to the UNNORMALISED Gram H_c^T H_c (reg = 1e-6), in-sample
fitted values, R^2 = 1 - sum(resid^2)/sum(Z_c^2) pooled over one-hot columns, clamped at 0.
Health metrics reproduce experiments/run_v2_dataset.py:effective_rank/repr_health (lines 168-188)
and thresholds HEALTH_PER_DIM_STD_MIN = 0.5, HEALTH_EFF_RANK_MIN = 2.0 (lines 81-82).
"""
import json

import numpy as np


def pcrl_ridge_r2(H, z, reg=1e-6, dtype=np.float64):
    H = np.asarray(H, dtype=dtype)
    k = int(z.max()) + 1
    Z = np.eye(k, dtype=dtype)[z]
    Hc = H - H.mean(axis=0, keepdims=True)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    G = Hc.T @ Hc + dtype(reg) * np.eye(H.shape[1], dtype=dtype)
    W = np.linalg.solve(G, Hc.T @ Zc)
    res = Zc - Hc @ W
    return float(max(0.0, 1.0 - (res ** 2).sum() / max((Zc ** 2).sum(), 1e-12)))


def ols_r2(H, z):
    """Unrestricted affine least squares (pseudoinverse), float64."""
    k = int(z.max()) + 1
    Z = np.eye(k)[z]
    Hc = H - H.mean(0)
    Zc = Z - Z.mean(0)
    W = np.linalg.lstsq(Hc, Zc, rcond=None)[0]
    res = Zc - Hc @ W
    return float(1.0 - (res ** 2).sum() / (Zc ** 2).sum())


def effective_rank(R):
    c = R - R.mean(0, keepdims=True)
    s = np.linalg.svd(c, compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def health(R):
    return {"per_dim_std_mean": float(R.std(0).mean()), "effective_rank": effective_rank(R)}


def clean_label(r2, h):
    return bool(r2 < 0.05 and h["per_dim_std_mean"] >= 0.5 and h["effective_rank"] >= 2.0)


rng = np.random.default_rng(20261002)
out = {"note": "synthetic only; see docstring for the reproduced PCRL arithmetic"}

# ---------------------------------------------------------------- D1: well-conditioned H
n, d = 5000, 64
z = rng.integers(0, 3, n)
H = 0.3 * rng.standard_normal((n, d))
H[:, 0] += 0.02 * (z == 2)            # weak linear signal, R^2 well below 0.05
scales = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 1.41, 2.0, 10.0]
rows = []
for c in scales:
    Hs = c * H
    h = health(Hs)
    r2 = pcrl_ridge_r2(Hs, z)
    rows.append({"scale": c, "ridge_r2": r2, "ridge_r2_float32": pcrl_ridge_r2(Hs, z, dtype=np.float32),
                 "ols_r2": ols_r2(Hs, z), **h, "clean_label": clean_label(r2, h)})
out["D1_well_conditioned_isotropic_rescale"] = rows

# ---------------------------------------------------------------- D2: signal in a near-singular direction
# Mimics the post-erasure situation where the residual attribute signal lives in directions whose
# singular values are 1e-7..1e-4 of the leading one (as recorded for LEACE-on-raw in the reconciliation).
H2 = 0.3 * rng.standard_normal((n, d))
s_small = 1e-5
H2[:, -1] = s_small * ((z == 1) - (z == 1).mean() + 0.3 * rng.standard_normal(n))
rows = []
for c in [0.1, 0.5, 1.0, 1.15, 1.41, 2.0, 10.0, 100.0]:
    Hs = c * H2
    h = health(Hs)
    r2 = pcrl_ridge_r2(Hs, z)
    gram_small = float(n * (c * s_small) ** 2)
    rows.append({"scale": c, "ridge_r2": r2, "ols_r2": ols_r2(Hs, z),
                 "smallest_direction_gram_eigen_approx": gram_small, "reg": 1e-6,
                 **h, "clean_label": clean_label(r2, h)})
out["D2_near_singular_signal_rescale"] = rows

# ---------------------------------------------------------------- D3: anisotropic rescale (health labels)
H3 = 0.35 * rng.standard_normal((n, 8))
H3[:, :2] *= 6.0                         # two dominant dims
rows = []
for name, M in [("identity", np.ones(8)), ("isotropic_x1.5", 1.5 * np.ones(8)),
                ("anisotropic_shrink_dominant", np.array([1 / 6, 1 / 6, 1, 1, 1, 1, 1, 1]))]:
    Hs = H3 * M
    h = health(Hs)
    rows.append({"transform": name, "ridge_r2": pcrl_ridge_r2(Hs, z), **h})
out["D3_health_under_rescale"] = rows

# ---------------------------------------------------------------- A1: composition
# (a) exact zero sample cross-covariance per release => concatenation has linear R^2 = 0 (OLS) on the
#     same rows; a nonlinear function of the concatenation can still carry A.
a = rng.integers(0, 2, n)
A = np.eye(2)[a]; Ac = A - A.mean(0)
def residualise(X):          # in-sample exact removal of linear cross-covariance with A
    Xc = X - X.mean(0)
    B = np.linalg.lstsq(Ac, Xc, rcond=None)[0]
    return Xc - Ac @ B
u = rng.standard_normal((n, 4))
h1 = residualise(u + 0.8 * a[:, None] + rng.standard_normal((n, 4)) * (1 + a[:, None]))
h2 = residualise(u + rng.standard_normal((n, 4)))
out["A1a_exact_zero_cross_cov"] = {
    "r2_h1": ols_r2(h1, a), "r2_h2": ols_r2(h2, a), "r2_concat": ols_r2(np.hstack([h1, h2]), a),
    "r2_concat_with_squares": ols_r2(np.hstack([h1, h2, h1 ** 2, h2 ** 2]), a),
    "note": "linear R^2 of the concatenation is 0 to rounding; adding squared features (a nonlinear "
            "recipient) recovers signal because A changes the variance of h1",
}
# (b) small but nonzero per-release R^2 does not give small concatenated R^2.
eps_sig = 0.05
uu = 3.0 * rng.standard_normal(n)
g1 = (eps_sig * a + uu)[:, None]
g2 = (uu + 1e-3 * rng.standard_normal(n))[:, None]
r1, r2_, rc = ols_r2(g1, a), ols_r2(g2, a), ols_r2(np.hstack([g1, g2]), a)
G = np.hstack([g1, g2]); C = np.cov(G.T); Dm = np.diag(1 / np.sqrt(np.diag(C)))
lam = float(np.linalg.eigvalsh(Dm @ C @ Dm).min())
out["A1b_small_nonzero_does_not_compose"] = {
    "r2_h1": r1, "r2_h2": r2_, "r2_concat": rc, "lambda_min_R": lam,
    "prop6_bound_sum_over_lambda_min": (r1 + r2_) / lam,
    "note": "each release passes tau=0.05 by orders of magnitude; the concatenation recovers A almost "
            "exactly; Prop 6 is not violated, its bound is simply vacuous because lambda_min(R) ~ 0",
}

# ---------------------------------------------------------------- A2: Prop 6 under regularisation conventions
def cov_parts(Hs, z):
    k = int(z.max()) + 1
    Z = np.eye(k)[z]
    Hc = Hs - Hs.mean(0); Zc = Z - Z.mean(0)
    return Hc, Zc

def plugin_r2_tau(Hc, Zc, tau):   # Prop 6(III) quantity on sample moments, tau on covariance scale
    m = Hc.shape[0]
    S = Hc.T @ Hc / m; C = Hc.T @ Zc / m
    return float(np.trace(C.T @ np.linalg.solve(S + tau * np.eye(S.shape[0]), C)) / (np.sum(Zc ** 2) / m))

def lam_min_R(blocks):
    Hc = np.hstack(blocks); S = Hc.T @ Hc / Hc.shape[0]
    D = np.zeros_like(S); i = 0
    for b in blocks:
        Sp = b.T @ b / b.shape[0]; w, V = np.linalg.eigh(Sp)
        D[i:i + b.shape[1], i:i + b.shape[1]] = V @ np.diag(w ** -0.5) @ V.T; i += b.shape[1]
    return float(np.linalg.eigvalsh(D @ S @ D).min())

worst = {"code_quantity_ratio": 0.0, "plugin_tau_ratio": 0.0}
viol_code = 0; viol_plugin = 0; trials = 400
for t in range(trials):
    m, dp = 300, 3
    zz = rng.integers(0, 3, m)
    base = rng.standard_normal((m, dp))
    s_scale = 10 ** rng.uniform(-4, 0)
    blocks = []
    for p in range(2):
        Bp = 0.7 * base + rng.standard_normal((m, dp)) * 10 ** rng.uniform(-3, 0)
        Bp[:, 0] += 0.3 * (zz == p)
        Bp *= s_scale * 10 ** rng.uniform(-1, 1, dp)
        blocks.append(Bp - Bp.mean(0))
    _, Zc = cov_parts(blocks[0], zz)
    lam = lam_min_R(blocks)
    tau_code = 1e-6 / m                          # reg 1e-6 on the unnormalised Gram == tau = 1e-6/m on covariance
    q_p = [pcrl_ridge_r2(b, zz) for b in blocks]
    q_H = pcrl_ridge_r2(np.hstack(blocks), zz)
    p_p = [plugin_r2_tau(b, Zc, tau_code) for b in blocks]
    p_H = plugin_r2_tau(np.hstack(blocks), Zc, tau_code)
    rc_ = q_H / (sum(q_p) / lam); rp_ = p_H / (sum(p_p) / lam)
    worst["code_quantity_ratio"] = max(worst["code_quantity_ratio"], rc_)
    worst["plugin_tau_ratio"] = max(worst["plugin_tau_ratio"], rp_)
    viol_code += rc_ > 1 + 1e-9; viol_plugin += rp_ > 1 + 1e-9
out["A2_prop6_regularisation_conventions"] = {
    "trials": trials,
    "bound_checked": "R2(H) <= sum_p R2(h_p) / lambda_min(R), R from unregularised sample covariances",
    "plugin_tikhonov_quantity_violations": int(viol_plugin),
    "plugin_tikhonov_max_ratio": worst["plugin_tau_ratio"],
    "pcrl_code_insample_ridge_quantity_violations": int(viol_code),
    "pcrl_code_insample_ridge_max_ratio": worst["code_quantity_ratio"],
    "note": "the code's in-sample ridge R2 equals tr(C'(G+l)^-1 (G+2l)(G+l)^-1 C)/TSS, which lies between "
            "the Tikhonov plug-in form of Prop 6(III) and unregularised OLS; scales are drawn so that "
            "reg is sometimes comparable to Gram eigenvalues",
}

print(json.dumps(out, indent=1))

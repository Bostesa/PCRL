"""Synthetic check of what concept-erasure==0.2.4 LeaceEraser.fit(x, z) guarantees UNDER ITS DEFAULTS
(method="leace", affine=True, constrain_cov_trace=True, shrinkage=True, svd_tol=0.01), which is how
PCRL (pcrl/training/v2_trainer.py:584, laftr_proxy_trainer.py:482) and durable-guarantees
(experiments/baseline_gauntlet.py:317, smart_erasure.py:205) call it, on float32 inputs.

Synthetic data only. Run: /Users/nathansamson/PCRL/.venv/bin/python leace_tolerance_check.py
Reports, on the FIT rows and on held-out rows from the same law:
  max |Cov(r(X), Z)| and the in-sample OLS R^2 of one-hot Z on r(X).
"""
import json

import numpy as np
import torch
from concept_erasure import LeaceEraser


def ols_r2(H, z):
    k = int(z.max()) + 1
    Z = np.eye(k)[z]; Hc = H - H.mean(0); Zc = Z - Z.mean(0)
    W = np.linalg.lstsq(Hc, Zc, rcond=None)[0]
    return float(1 - ((Zc - Hc @ W) ** 2).sum() / (Zc ** 2).sum())


def heldout_r2(Hf, zf, Hs, zs):
    """OLS fit on rows f, scored on rows s (affine, means from f)."""
    k = int(max(zf.max(), zs.max())) + 1
    Zf = np.eye(k)[zf]; Zs = np.eye(k)[zs]
    mh, mz = Hf.mean(0), Zf.mean(0)
    W = np.linalg.lstsq(Hf - mh, Zf - mz, rcond=None)[0]
    pred = (Hs - mh) @ W + mz
    return float(1 - ((Zs - pred) ** 2).sum() / ((Zs - Zs.mean(0)) ** 2).sum())


def pcrl_ridge_r2(H, z, reg=1e-6):
    k = int(z.max()) + 1
    Z = np.eye(k)[z]; Hc = H - H.mean(0); Zc = Z - Z.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + reg * np.eye(H.shape[1]), Hc.T @ Zc)
    return float(max(0.0, 1 - ((Zc - Hc @ W) ** 2).sum() / (Zc ** 2).sum()))


def ols_r2_floor(H, z, rel):
    """OLS after discarding covariance directions with eigenvalue < rel * largest (numerical floor)."""
    Hc = H - H.mean(0)
    w, V = np.linalg.eigh(Hc.T @ Hc / len(z))
    keep = w > rel * w.max()
    return ols_r2(Hc @ V[:, keep], z)


def logreg_auc(Hf, zf, Hs, zs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Hf)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(Hf), zf)
    return float(roc_auc_score(zs, m.predict_proba(sc.transform(Hs)), multi_class="ovr", average="macro"))


def max_xcov(H, z):
    k = int(z.max()) + 1
    Z = np.eye(k)[z]
    return float(np.abs((H - H.mean(0)).T @ (Z - Z.mean(0)) / len(z)).max())


rng = np.random.default_rng(7)
out = {}
cases = {
    # strong signal, well conditioned
    "strong_isotropic": dict(d=16, sig=1.0, aniso=1.0),
    # weak signal: whitened cross-cov singular value below svd_tol=0.01 -> not erased
    "weak_below_svd_tol": dict(d=16, sig=0.015, aniso=1.0),
    # strongly anisotropic covariance: oblique LEACE projection may increase trace -> trace constraint mixes in
    # an orthogonal projection in the original basis
    "anisotropic_trace_constraint": dict(d=16, sig=1.0, aniso=30.0),
}
for name, c in cases.items():
    n = 6000
    z = rng.integers(0, 3, 2 * n)
    scales = np.ones(c["d"]); scales[: c["d"] // 2] = c["aniso"]
    X = rng.standard_normal((2 * n, c["d"])) * scales
    mix = rng.standard_normal((c["d"], c["d"])) / np.sqrt(c["d"])
    X = X @ (np.eye(c["d"]) + 0.5 * mix)
    X[:, 0] += c["sig"] * (z == 1); X[:, 1] -= c["sig"] * (z == 2)
    Xf, zf, Xs, zs = X[:n], z[:n], X[n:], z[n:]
    res = {}
    for label, kw in [("defaults", {}), ("exact_settings", dict(svd_tol=1e-12, constrain_cov_trace=False))]:
        er = LeaceEraser.fit(torch.from_numpy(Xf).float(), torch.from_numpy(np.eye(3)[zf]).float(), **kw)
        Rf = er(torch.from_numpy(Xf).float()).double().numpy()
        Rs = er(torch.from_numpy(Xs).float()).double().numpy()
        ev = np.linalg.eigvalsh(np.cov(Rf.T))
        res[label] = {
            "fit_rows_cov_eigenvalues_smallest3": [float(v) for v in ev[:3]],
            "fit_rows_pcrl_ridge1e-6_r2": pcrl_ridge_r2(Rf, zf),
            "fit_rows_ols_r2_after_dropping_eigs_below_1e-6_rel": ols_r2_floor(Rf, zf, 1e-6),
            "heldout_rows_pcrl_ridge1e-6_r2": pcrl_ridge_r2(Rs, zs),
            "standardized_logreg_heldout_auc_macro": logreg_auc(Rf, zf, Rs, zs),
            "fit_rows_max_abs_xcov": max_xcov(Rf, zf),
            "fit_rows_insample_ols_r2": ols_r2(Rf, zf),
            "heldout_rows_insample_ols_r2": ols_r2(Rs, zs),
            "probe_fit_on_fit_rows_scored_heldout_r2": heldout_r2(Rf, zf, Rs, zs),
        }
    res["raw_insample_ols_r2"] = ols_r2(Xf, zf)
    out[name] = res
print(json.dumps(out, indent=1))

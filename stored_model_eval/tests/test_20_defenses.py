"""Official LEACE wrapper and matched-defense helpers (method owner, 2026-10-02).

Synthetic only. Covers:
  * pins: concept-erasure 0.2.4, installed source sha256 recomputed and asserted;
  * a port of the official upstream test ``tests/test_leace.py::test_linear_erasure`` (tag v0.2.4, commit
    9b18b3d5c73f552798212c51d6533d649fa434cd, file sha256 b9d126f49f35ca284f02a5b215612746eac4ddb0db329bad48341e26913f0dc1)
    restricted to the DEFAULT settings the benchmark uses; the oracle-loss comparison is dropped because the
    benchmark never imports OracleEraser. The unmodified upstream suite was also run against the installed
    package (27 passed; log in results/combined_matched_removal_benchmark_v1/notes/method/upstream_v0.2.4_test_run.txt);
  * an independently solved covariance example (hand-solved 2-d P and a numpy-eigh closed form
    P = I - W^+ P_{W Sxz} W, W = Sxx^{-1/2});
  * fitted cross-covariance zero on fit rows; nonzero but small on fresh rows (finite-sample scope);
  * rank-deficient X: identity on the complement of the fit covariance support (transfer outside support);
  * concatenated marginal one-hots erase each marginal, not their intersection;
  * a nonlinear (XOR) concept stays recoverable by a nonlinear model (scope, not a refutation);
  * svd_tol truncation is reported, never silently "passed";
  * save/load round trip, fixed-map transform, alias test, noise delegation, scale report, fit gate.
"""
import json

import numpy as np
import pytest
import torch

from stored_model_eval import defenses as D
from stored_model_eval.guards import FitAuthorization, ScientificFitRefused
from stored_model_eval.releases import gaussian_release


def _fitter(d, k, **kw):
    from concept_erasure import LeaceFitter

    return LeaceFitter(d, k, dtype=torch.float64, **kw)


def _closed_form_P(Sxx, Sxz, svd_tol=0.0):
    """Independent closed form: P = I - W^+ P_{W Sxz} W, W = Sxx^{-1/2} (numpy eigh), pinv on the support."""
    L, V = np.linalg.eigh(Sxx)
    m = L > L[-1] * len(L) * np.finfo(float).eps
    W = (V * np.where(m, 1 / np.sqrt(np.where(m, L, 1)), 0)) @ V.T
    Wp = (V * np.where(m, np.sqrt(np.where(m, L, 0)), 0)) @ V.T
    A = W @ Sxz
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    U = U[:, s > max(svd_tol, 1e-12)]
    return np.eye(len(L)) - Wp @ (U @ U.T) @ W


# ------------------------------------------------------------------------------------------------
# pins
# ------------------------------------------------------------------------------------------------


def test_official_package_pins():
    prov = D.verify_official_leace()
    sha, _, n = D.installed_source_sha256()
    assert prov["version"] == "0.2.4" and sha == D.LEACE_SOURCE_SHA256 and n == 15
    assert prov["upstream_commit"] == "9b18b3d5c73f552798212c51d6533d649fa434cd"
    # default settings are the pinned ones
    f = _fitter(4, 2)
    assert {"method": f.method, "affine": f.affine, "constrain_cov_trace": f.constrain_cov_trace,
            "shrinkage": f.shrinkage, "svd_tol": f.svd_tol} == D.LEACE_DEFAULTS


# ------------------------------------------------------------------------------------------------
# upstream test port (default settings)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_classes", [1, 2, 3, 5, 10, 20])
def test_upstream_linear_erasure_port_default_settings(num_classes):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    eps = 2e-9
    n, d = 2048, 128
    num_distinct = max(num_classes, 2)
    X, Y = make_classification(n_samples=n, n_features=d, n_classes=num_distinct, n_informative=num_distinct,
                               random_state=42)
    Z = np.eye(num_classes)[Y] if num_classes > 1 else Y.astype(float)[:, None]
    m = D.fit_leace(X, Z, fit_row_ids=np.arange(n))
    X_ = m.transform(X)
    Xt, X_t = torch.from_numpy(X), torch.from_numpy(X_)

    # first-order optimality holds only for shrinkage=False upstream; the default (shrinkage=True) is NOT
    # least-squares optimal w.r.t. the sample covariance (upstream asserts exactly this)
    A, s, B_t = torch.linalg.svd(torch.from_numpy(m.P))
    A, B = A[:, s > 0.5].requires_grad_(True), B_t[s > 0.5].T
    Pc = A @ torch.inverse(B.T @ A) @ B.T
    mu = torch.from_numpy(m.mean_x)
    x_ = (Xt - mu) @ Pc.T + mu
    Lm = torch.randn(d, d, dtype=Xt.dtype, generator=torch.Generator().manual_seed(0)) / d ** 0.5
    torch.nn.functional.mse_loss(x_ @ Lm, Xt @ Lm).backward()
    assert not (A.grad.norm() < eps)

    torch.testing.assert_close(torch.from_numpy(m.transform(X_)), X_t)          # idempotence
    assert int(torch.linalg.svdvals(Xt - X_t).gt(eps).sum()) <= num_classes   # rank of the update
    torch.testing.assert_close(Xt.mean(0), X_t.mean(0))                       # unconditional mean kept
    cm_ = torch.stack([X_t[torch.from_numpy(Y) == c].mean(0) for c in range(num_distinct)])
    torch.testing.assert_close(cm_[1:], cm_[:-1])                             # class means equalised
    cm = torch.stack([Xt[torch.from_numpy(Y) == c].mean(0) for c in range(num_distinct)])
    assert not torch.allclose(cm[1:], cm[:-1])

    # linear guardedness (upstream check_linear_guardedness)
    y = Y.reshape(n, -1) if num_classes == 1 else Y
    y = np.ravel(y)
    lr = LogisticRegression(penalty=None, tol=0.0).fit(X_ - X_.mean(0), y)
    assert abs(lr.coef_).max() < eps
    svm = LinearSVC(dual=False, intercept_scaling=1e6, tol=eps).fit(X_, y)
    assert abs(svm.coef_).max() < eps
    real = LogisticRegression(penalty=None, tol=0.0).fit(X - X.mean(0), y)
    assert abs(real.coef_).max() > 0.05


# ------------------------------------------------------------------------------------------------
# independently solved covariance example
# ------------------------------------------------------------------------------------------------


def _exact_moment_data(n, Sxx_target, c):
    """Rows whose Bessel-corrected sample moments are exactly Sxx_target and Sxz = c*var(z)*e1 (z binary)."""
    rng = np.random.default_rng(7)
    z = np.tile([0.0, 1.0], n // 2)
    zc = z - z.mean()
    vz = zc @ zc / (n - 1)
    R = rng.normal(size=(n, 2))
    Q, _ = np.linalg.qr(np.c_[np.ones(n), zc])
    R -= Q @ (Q.T @ R)
    SE = Sxx_target - (c ** 2) * vz * np.diag([1.0, 0.0])
    CR = R.T @ R / (n - 1)
    Lr, Vr = np.linalg.eigh(CR)
    Le, Ve = np.linalg.eigh(SE)
    E = R @ (Vr / np.sqrt(Lr)) @ Vr.T @ (Ve * np.sqrt(Le)) @ Ve.T
    X = E + c * zc[:, None] * np.array([1.0, 0.0]) + np.array([3.0, -1.0])
    return X, z, vz


def test_hand_solved_two_dim_example():
    # Sxx = [[6,1],[1,2]], Sxz = e1  =>  P x = x - b (b' Sxx^-1 x)/(b' Sxx^-1 b) = [[0, S12/S22], [0, 1]]
    Sxx = np.array([[6.0, 1.0], [1.0, 2.0]])
    n = 4000
    vz = n / (4 * (n - 1))
    X, z, vz2 = _exact_moment_data(n, Sxx, c=1.0 / vz)
    assert np.isclose(vz, vz2)
    assert np.allclose(np.cov(X.T), Sxx, atol=1e-10)
    P_hand = np.array([[0.0, 0.5], [0.0, 1.0]])
    f = _fitter(2, 1, shrinkage=False).update(torch.from_numpy(X), torch.from_numpy(z[:, None]))
    assert np.allclose(f.sigma_xz.numpy().ravel(), [1.0, 0.0], atol=1e-10)
    assert np.allclose(f.eraser.P.numpy(), P_hand, atol=1e-9)
    assert np.allclose(_closed_form_P(Sxx, np.array([[1.0], [0.0]])), P_hand, atol=1e-12)
    # default settings (shrinkage) give a different oblique map, but the erasure condition still holds
    m = D.fit_leace(X, z[:, None])
    assert not np.allclose(m.P, P_hand, atol=1e-3)
    assert np.abs(m.P @ np.array([1.0, 0.0])).max() < 1e-10
    assert m.native_check(X, z[:, None])["status"] == "WITHIN_TOLERANCE"


def test_numpy_closed_form_matches_package():
    rng = np.random.default_rng(3)
    n, d, k = 3000, 6, 3
    y = rng.integers(0, k, n)
    M = rng.normal(size=(d, d))
    X = rng.normal(size=(n, d)) @ M + np.eye(k, d)[y] @ rng.normal(size=(d, d)) * 2
    Z = np.eye(k)[y]
    Xc, Zc = X - X.mean(0), Z - Z.mean(0)
    Sxx, Sxz = Xc.T @ Xc / (n - 1), Xc.T @ Zc / (n - 1)
    f = _fitter(d, k, shrinkage=False).update(torch.from_numpy(X), torch.from_numpy(Z))
    assert np.allclose(f.eraser.P.numpy(), _closed_form_P(Sxx, Sxz, 0.01), atol=1e-9)
    # default (shrunk) map equals the closed form computed from the package's own shrunk covariance
    m = D.fit_leace(X, Z)
    assert np.allclose(m.P, _closed_form_P(m.arrays["sigma_xx_used"], m.arrays["sigma_xz"], 0.01), atol=1e-9)
    assert np.allclose(m.arrays["sigma_xz"], Sxz, atol=1e-12)
    assert m.metadata["diagnostics"]["cov_trace_constraint_fired"] is False
    assert m.metadata["rank"] == k - 1


# ------------------------------------------------------------------------------------------------
# finite-sample scope
# ------------------------------------------------------------------------------------------------


def _law(rng, n, d=10):
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += 0.8 * (y == 1)
    X[:, 1] += 0.6 * (y == 2) - 0.3 * (y == 0)
    return X, y


def test_crosscov_zero_on_fit_rows_small_but_nonzero_on_fresh_rows():
    rng = np.random.default_rng(11)
    resid = {}
    for n_fit in (500, 8000):
        Xf, yf = _law(rng, n_fit)
        m = D.fit_leace(Xf, np.eye(3)[yf], fit_row_ids=np.arange(n_fit))
        nc = m.native_check(Xf, np.eye(3)[yf])
        assert nc["status"] == "WITHIN_TOLERANCE"
        assert nc["crosscov_max_abs_rel_erased"] < 1e-10 and nc["ols_r2_joint_erased"] < 1e-10
        assert nc["whitened_residual_spectral_norm"] < 1e-12 and nc["implementation_bound_holds"]
        Xn, yn = _law(rng, 200_000)
        h = D.crosscov_stats(m.transform(Xn), np.eye(3)[yn], scale_H=Xf.std(0, ddof=1))
        resid[n_fit] = h["max_abs_rel"]
        assert 1e-6 < h["max_abs_rel"] < 6 / np.sqrt(n_fit)   # nonzero, at sampling-error scale
        un = D.crosscov_stats(Xn, np.eye(3)[yn])["max_abs_rel"]
        assert h["max_abs_rel"] < un / 3
    assert resid[8000] < resid[500]


def test_rank_deficient_fit_identity_outside_support():
    rng = np.random.default_rng(5)
    n, d = 4000, 6
    y = rng.integers(0, 2, n)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    S, Nul = basis[:, :3], basis[:, 3:]
    lat = rng.normal(size=(n, 3)) + np.outer(y, [1.0, 0.0, 0.5])
    X = lat @ S.T                                    # fit rows live in a 3-d subspace
    m = D.fit_leace(X, np.eye(2)[y])
    dg = m.metadata["diagnostics"]
    assert dg["sample_cov_rank"] == 3 and dg["whitening_dirs_masked"] == 0   # shrinkage makes Sigma full rank
    assert m.native_check(X, np.eye(2)[y])["status"] == "WITHIN_TOLERANCE"
    # P is the identity on the complement of the fit support
    assert np.allclose(m.P @ Nul, Nul, atol=1e-10)
    # fresh rows that carry the concept OUTSIDE the fit support keep it after erasure
    yn = rng.integers(0, 2, 50_000)
    Xn = (rng.normal(size=(50_000, 3)) + np.outer(yn, [1.0, 0.0, 0.5])) @ S.T + np.outer(yn, Nul[:, 0])
    E = m.transform(Xn)
    c_out = np.corrcoef(E @ Nul[:, 0], yn)[0, 1]
    assert c_out > 0.4
    # same behaviour without shrinkage (pinv mask) in the official fitter
    f = _fitter(d, 2, shrinkage=False).update(torch.from_numpy(X), torch.from_numpy(np.eye(2)[y]))
    assert np.allclose(f.eraser.P.numpy() @ Nul, Nul, atol=1e-6)


def test_concatenated_marginals_erase_each_marginal_not_intersection():
    rng = np.random.default_rng(2)
    n = 20_000
    a, b, c = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 3, n)
    cell = (a != b).astype(float)                      # an intersection-cell contrast, uncorrelated with marginals
    X = rng.normal(size=(n, 8))
    X[:, 0] += a
    X[:, 1] += b
    X[:, 3] += c == 1
    X[:, 2] += 2.0 * cell
    Z, spec = D.concat_marginal_onehots({"a": a, "b": b, "c": c}, {"a": 2, "b": 2, "c": 3})
    assert [s["col_start"] for s in spec] == [0, 2, 4] and Z.shape == (n, 7)
    m = D.fit_leace(X, Z, concept_spec=spec)
    nc = m.native_check(X, Z)
    assert nc["status"] == "WITHIN_TOLERANCE"
    assert {blk["name"] for blk in nc["blocks"]} == {"a", "b", "c"}
    assert all(blk["ols_r2_fit_rows_erased"] < 1e-10 for blk in nc["blocks"])
    assert all(blk["ols_r2_fit_rows_untreated"] > 0.05 for blk in nc["blocks"])
    assert m.metadata["rank"] == 1 + 1 + 2
    # the intersection cell (a != b) is NOT erased: it stays linearly recoverable from the erased representation
    E = m.transform(X)
    assert D._ols_r2(E, cell[:, None]) > 0.4


def test_xor_concept_survives_for_nonlinear_attacker():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # y = XOR of two cluster signs (no linear signal in x0, x1) plus a linear trace of y in x2 that LEACE erases
    rng = np.random.default_rng(0)
    n = 12_000
    q = rng.integers(0, 2, (n, 2))
    y = q[:, 0] ^ q[:, 1]
    X = np.c_[(2 * q - 1) * 2.0 + rng.normal(scale=0.5, size=(n, 2)), y + rng.normal(size=n),
              rng.normal(size=(n, 2))]
    fit, att, te = np.arange(0, 4000), np.arange(4000, 8000), np.arange(8000, n)
    m = D.fit_leace(X[fit], np.eye(2)[y[fit]])
    assert m.native_check(X[fit], np.eye(2)[y[fit]])["status"] == "WITHIN_TOLERANCE"
    assert m.metadata["rank"] == 1
    E = m.transform(X)
    lin0 = LogisticRegression().fit(X[att], y[att]).predict_proba(X[te])[:, 1]
    assert roc_auc_score(y[te], lin0) > 0.7       # linear signal present before erasure
    lin = LogisticRegression().fit(E[att], y[att]).predict_proba(E[te])[:, 1]
    gbt = HistGradientBoostingClassifier(random_state=0).fit(E[att], y[att]).predict_proba(E[te])[:, 1]
    assert roc_auc_score(y[te], lin) < 0.56
    assert roc_auc_score(y[te], gbt) > 0.9      # outside LEACE's linear scope; not a refutation


def test_svd_tol_truncation_is_reported_not_passed():
    # weak signal in a rare class: whitened cross-cov singular value < svd_tol = 0.01 -> left intact
    rng = np.random.default_rng(4)
    n = 20_000
    y = (rng.random(n) < 0.01).astype(int)
    X = rng.normal(size=(n, 5))
    X[:, 0] += 0.06 * y
    m = D.fit_leace(X, np.eye(2)[y])
    dg = m.metadata["diagnostics"]
    assert dg["n_singular_values_nonzero_truncated"] == 1 and m.metadata["rank"] == 0
    nc = m.native_check(X, np.eye(2)[y])
    assert nc["status"] == "OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION"
    assert nc["crosscov_max_abs_rel_erased"] > 1e-4
    # what the default official map does guarantee: whitened residual cross-covariance <= svd_tol
    assert nc["implementation_bound_holds"] and 1e-4 < nc["whitened_residual_spectral_norm"] <= 0.01


# ------------------------------------------------------------------------------------------------
# persistence, fixed map, alias, noise, scale, gate
# ------------------------------------------------------------------------------------------------


def test_save_load_round_trip_identical(tmp_path):
    rng = np.random.default_rng(1)
    X, y = _law(rng, 3000)
    Z, spec = D.concat_marginal_onehots({"s": y}, {"s": 3})
    m = D.fit_leace(X, Z, fit_row_ids=np.arange(1000, 4000), concept_spec=spec)
    info = m.save(tmp_path / "B")
    m2 = D.LeaceMap.load(tmp_path / "B")
    for a in ("proj_left", "proj_right", "mean_x"):
        assert np.array_equal(getattr(m, a), getattr(m2, a))
    for k in m.arrays:
        assert np.array_equal(m.arrays[k], m2.arrays[k])
    Xn, _ = _law(rng, 500)
    assert np.array_equal(m.transform(Xn), m2.transform(Xn))
    meta = json.loads((tmp_path / "B" / "leace_map.json").read_text())
    for key in ("rank", "dtype", "tolerances", "n_fit", "fit_row_ids_sha256", "concept_spec", "provenance",
                "settings_used", "diagnostics"):
        assert key in meta
    assert meta["provenance"]["installed_source_sha256"] == D.LEACE_SOURCE_SHA256
    assert meta["fit_row_ids_sha256"] == D._ids_hash(np.arange(1000, 4000))
    # tampering is detected
    with np.load(tmp_path / "B" / "leace_map.npz") as z:
        arr = {k: z[k] for k in z.files}
    arr["mean_x"] = arr["mean_x"] + 1e-9
    np.savez(tmp_path / "B" / "leace_map.npz", **arr)
    with pytest.raises(ValueError, match="sha256"):
        D.LeaceMap.load(tmp_path / "B")
    assert info["npz_sha256"] == meta["npz_sha256"]


def test_transform_is_fixed_map_with_fitting_mean():
    rng = np.random.default_rng(6)
    X, y = _law(rng, 2000)
    m = D.fit_leace(X, np.eye(3)[y])
    Xs = X[:300] + 5.0                                  # shifted scored rows
    expect = Xs - (Xs - m.mean_x) @ (np.eye(X.shape[1]) - m.P).T
    assert np.allclose(m.transform(Xs), expect, atol=1e-12)
    assert np.allclose(m.transform(Xs)[:10], m.transform(Xs[:10]), rtol=0, atol=1e-12)  # row-wise map


def test_alias_test():
    rng = np.random.default_rng(8)
    X, y = _law(rng, 3000)
    a2 = rng.integers(0, 2, 3000)
    X[:, 3] += a2
    mb = D.fit_leace(X, np.eye(3)[y], fit_row_ids=np.arange(3000))
    Zsame, _ = D.concat_marginal_onehots({"s": y, "s_again": y}, {"s": 3, "s_again": 3})
    mc_same = D.fit_leace(X, Zsame, fit_row_ids=np.arange(3000))
    Zdiff, _ = D.concat_marginal_onehots({"s": y, "t": a2}, {"s": 3, "t": 2})
    mc_diff = D.fit_leace(X, Zdiff, fit_row_ids=np.arange(3000))
    r = D.alias_test(mb, mc_same, H_probe=X[:50])
    assert r["alias"] and r["max_abs_P_diff"] < 1e-10
    r2 = D.alias_test(mb, mc_diff)
    assert not r2["alias"] and r2["rank_c"] == r2["rank_b"] + 1


def test_noise_release_delegates_and_scale_report():
    H = np.random.default_rng(0).normal(size=(400, 8)) * 0.5
    assert np.array_equal(D.noise_release(H, 2.0, 1), gaussian_release(H, 2.0, 1))
    assert np.array_equal(D.noise_release(H, 0.0, 1), H)
    rep = D.scale_report(H)
    assert [s["sigma_abs"] for s in rep["sigmas"]] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert abs(rep["sigma_train_scale_rms_sd"] - 0.5) < 0.05
    s1 = rep["sigmas"][2]
    assert np.isclose(s1["sigma_over_rms_sd"], 1.0 / rep["sigma_train_scale_rms_sd"])
    assert len(rep["per_dim_sd"]["values"]) == 8


def test_fit_gate_and_dtype():
    X = np.random.default_rng(0).normal(size=(100, 3))
    Z = np.eye(2)[np.arange(100) % 2]
    with pytest.raises(ScientificFitRefused):
        D.fit_leace(X, Z, auth=FitAuthorization(), synthetic=False)
    D.fit_leace(X, Z, auth=FitAuthorization.synthetic_only(), synthetic=True)
    with pytest.raises(ValueError):
        D.fit_leace(X, Z, dtype=np.float32)

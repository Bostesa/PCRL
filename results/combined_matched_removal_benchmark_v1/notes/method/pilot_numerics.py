"""Focused numerical check of the pilot's education_assessment fixed-ridge R2 values (versioned diagnostic).

Role 2 (method owner), matched removal benchmark v1, 2026-10-02. NOT a replacement of any primary number.

Reads EXISTING pilot arrays only (no attacker refits; the only "fits" are closed-form least-squares solves of
the registered G1 problem, recomputed for numerical evidence):
  ~/PCRL_eval_cache_private/pilot_adult_s0/cache/adult_s0_test.npz   (rep_p2, float32 as stored)
  ~/PCRL_eval_cache_private/pilot_adult_s0/labels.npz                 (role, sensitive labels)
  ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/units/education_assessment__*/preds.npz (G1_pred/G1_prior/G2_*)

Writes notes/method/pilot_numerics.json next to this file.

Questions answered, per education pair (sex, race, income):
  (i)   historical float32 in-sample computation (N0, mixed precision): conditioning of the Gram, float32 Gram
        error vs the smallest eigenvalues, and whether the mixed-precision W is even a ridge minimiser
        (an exact ridge solution has in-sample R2 >= 0 by construction, because J(W*) <= J(0) = SS_tot).
  (ii)  the fixed-ridge (lambda = 1e-6) predictor's held-out performance (G1): solver residuals in float64 vs
        float32, a stable SVD solution of the SAME fixed-ridge problem, and whether the stable held-out G1
        equals the stored G1 (i.e. negativity is a property of the predictor, not of the solver).
  (iii) the separately evaluated scale-invariant G2 (stored predictions, recomputed statistic).
  (iv)  nothing here bounds the broader class of linear predictors; only these fitted probes are examined.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
sys.path.insert(0, str(WORKTREE))
from stored_model_eval.recipes import historical_native_r2  # noqa: E402  (read-only import)

PILOT = Path(os.path.expanduser("~/PCRL_eval_cache_private/pilot_adult_s0"))
LAM = 1e-6
PAIRS = ("sex", "race", "income")
PURPOSE = "education_assessment"
REP_KEY = "rep_p2"
G2_FLOOR_REL = 1e-6  # registered G2 numerical floor (EFFECTIVE_PROTOCOL linear_closed_form.G2.floor_rel)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def onehot(y, K, dtype=np.float64):
    return np.eye(K, dtype=dtype)[np.asarray(y).astype(int)]


def r2_vs_prior(Y, pred, prior):
    """Pilot r2_stat with all-ones weights: 1 - SS_res / SS_tot(around attacker_fit prior). Unclamped."""
    return 1.0 - float(((Y - pred) ** 2).sum()) / float(((Y - prior[None, :]) ** 2).sum())


def ridge_objective(Yc, Hc, W, lam):
    return float(((Yc - Hc @ W) ** 2).sum() + lam * (W ** 2).sum())


def main() -> dict:
    files = {
        "cache": PILOT / "cache/adult_s0_test.npz",
        "labels": PILOT / "labels.npz",
        "task_labels": PILOT / "run_v1/inputs/task_labels_v1.npz",
    }
    c = np.load(files["cache"])
    L = np.load(files["labels"])
    assert np.array_equal(c["row_id"], L["row_id"])
    H32 = c[REP_KEY]
    assert H32.dtype == np.float32
    H = H32.astype(np.float64)
    role = L["role"]
    F = np.flatnonzero(role == "attacker_fit")
    d = H.shape[1]

    # ---- shared fit-row geometry (attacker_fit rows; G1's fit role) ----
    muF = H[F].mean(0)
    Hc = H[F] - muF
    G = Hc.T @ Hc
    ev, V = np.linalg.eigh(G)
    Gl = G + LAM * np.eye(d)
    geom = {
        "n_attacker_fit": int(len(F)),
        "dim": int(d),
        "stored_dtype": "float32",
        "gram_eigs_smallest_8": ev[:8].tolist(),
        "gram_eigs_largest_3": ev[-3:].tolist(),
        "cond_gram": float(ev[-1] / ev[0]),
        "cond_gram_plus_lambda": float((ev[-1] + LAM) / (ev[0] + LAM)),
        "n_eigs_below_lambda": int((ev < LAM).sum()),
        "n_eigs_below_100lambda": int((ev < 100 * LAM).sum()),
        "n_eigs_below_G2_floor(1e-6*max)": int((ev < G2_FLOOR_REL * ev[-1]).sum()),
        "float32_eps_times_max_eig": float(np.finfo(np.float32).eps * ev[-1]),
        "float64_eps_times_max_eig": float(np.finfo(np.float64).eps * ev[-1]),
        "n_eigs_below_float32_resolution": int((ev < np.finfo(np.float32).eps * ev[-1]).sum()),
        "per_dim_std_fit": {"min": float(H[F].std(0).min()), "median": float(np.median(H[F].std(0))),
                            "max": float(H[F].std(0).max())},
        "row_norm_quantiles_all_rows": dict(zip(["q05", "q50", "q95"],
                                                np.percentile(np.linalg.norm(H, axis=1), [5, 50, 95]).tolist())),
        "lambda_relative_to_trace_over_d": float(LAM / (np.trace(G) / d)),
    }
    # float32 Gram (historical arithmetic: float32 centring + float32 product) vs float64 Gram
    Hc32 = H32[F] - H32[F].mean(axis=0, keepdims=True)
    G32 = (Hc32.T @ Hc32).astype(np.float64)
    dG = G32 - G
    geom["fit_rows_float32_gram_error_spectral"] = float(np.linalg.norm(dG, 2))
    geom["fit_rows_float32_gram_error_over_min_eig"] = float(np.linalg.norm(dG, 2) / ev[0])
    # assessment-row variance along fit near-null directions (why the fixed-ridge predictor transfers badly)
    A = np.flatnonzero(role == "assessment")
    varF = ((Hc @ V) ** 2).mean(0)
    varA = (((H[A] - muF) @ V) ** 2).mean(0)
    low = ev < G2_FLOOR_REL * ev[-1]
    geom["assessment_over_fit_variance_ratio_in_dirs_below_G2_floor"] = {
        "n_dirs": int(low.sum()),
        "median": float(np.median(varA[low] / varF[low])) if low.any() else None,
        "max": float(np.max(varA[low] / varF[low])) if low.any() else None,
    }
    # concentration: share of the near-null-direction sum of squares carried by the top 1% of rows
    def top1_share(M):
        r = (M ** 2).sum(1)
        k = max(1, int(round(0.01 * len(r))))
        return float(np.sort(r)[::-1][:k].sum() / r.sum())
    geom["top1pct_row_share_of_sumsq_in_dirs_below_G2_floor"] = {
        "attacker_fit": top1_share(Hc @ V[:, low]), "assessment": top1_share((H[A] - muF) @ V[:, low])}
    geom["assessment_over_fit_variance_ratio_in_dirs_above_G2_floor"] = {
        "median": float(np.median(varA[~low] / varF[~low])), "max": float(np.max(varA[~low] / varF[~low]))}

    pairs = {}
    for attr in PAIRS:
        u = PILOT / f"run_v1/units/{PURPOSE}__{attr}"
        files[f"preds_{attr}"] = u / "preds.npz"
        P = np.load(u / "preds.npz")
        y = L[attr].astype(int)
        K = int(P["G1_prior"].shape[0])
        assert np.array_equal(P["assess_row_id"], L["row_id"][A])
        Y = onehot(y, K)
        Yc = Y[F] - Y[F].mean(0)
        B = Hc.T @ Yc
        nB = np.linalg.norm(B)

        # (ii-a) the registered float64 solve (as in recipes.fixed_ridge_fit)
        W64 = np.linalg.solve(Gl, B)
        # (ii-b) the same problem, all-float32 arithmetic
        Gl32 = (Hc32.T @ Hc32) + np.float32(LAM) * np.eye(d, dtype=np.float32)
        W32 = np.linalg.solve(Gl32, (Hc32.T @ Yc.astype(np.float32))).astype(np.float64)
        # (ii-c) stable SVD solution of the same fixed-ridge problem (no normal equations formed)
        U_, s, Vt = np.linalg.svd(Hc, full_matrices=False)
        Wsvd = Vt.T @ (np.diag(s / (s ** 2 + LAM)) @ (U_.T @ Yc))
        # (ii-d) eigendecomposition solution
        Weig = V @ (np.diag(1.0 / (ev + LAM)) @ (V.T @ B))

        def resid(W):
            return float(np.linalg.norm(Gl @ W - B) / nB)

        muY = Y[F].mean(0)
        pred = lambda W: (H[A] - muF) @ W + muY  # noqa: E731
        stored_g1 = r2_vs_prior(Y[A], P["G1_pred"], P["G1_prior"])
        heldout = {
            "stored_G1_point_recomputed_from_saved_preds": stored_g1,
            "float64_solve": r2_vs_prior(Y[A], pred(W64), muY),
            "float32_solve": r2_vs_prior(Y[A], pred(W32), muY),
            "stable_svd_solve": r2_vs_prior(Y[A], pred(Wsvd), muY),
            "eigh_solve": r2_vs_prior(Y[A], pred(Weig), muY),
            "max_abs_diff_saved_G1_pred_vs_float64_solve": float(np.abs(P["G1_pred"] - pred(W64)).max()),
            "max_abs_diff_saved_G1_pred_vs_svd_solve": float(np.abs(P["G1_pred"] - pred(Wsvd)).max()),
        }
        r_rows = ((Y[A] - pred(W64)) ** 2).sum(1)
        k1 = max(1, int(round(0.01 * len(r_rows))))
        heldout["top1pct_assessment_rows_share_of_SS_res"] = float(np.sort(r_rows)[::-1][:k1].sum() / r_rows.sum())
        # Truncation decomposition: the SAME fixed lambda, restricted to directions above the G2 floor.
        keep = ~low
        Wtr = V[:, keep] @ (np.diag(1.0 / (ev[keep] + LAM)) @ (V[:, keep].T @ B))
        heldout["fixed_lambda_restricted_to_dirs_above_G2_floor"] = r2_vs_prior(Y[A], pred(Wtr), muY)
        heldout["share_of_W_norm_sq_in_dirs_below_G2_floor"] = float(
            ((V[:, low].T @ W64) ** 2).sum() / (W64 ** 2).sum()) if low.any() else 0.0
        # in-sample (attacker_fit) R2 of each solution, to show it is >= 0 when exact
        insample_fit = {k: 1.0 - float(((Yc - Hc @ W) ** 2).sum()) / float((Yc ** 2).sum())
                        for k, W in (("float64_solve", W64), ("svd_solve", Wsvd), ("float32_solve", W32))}

        solver = {
            "relative_normal_eq_residual_float64_solve": resid(W64),
            "relative_normal_eq_residual_float32_solve": resid(W32),
            "relative_normal_eq_residual_svd_solve": resid(Wsvd),
            "relative_normal_eq_residual_eigh_solve": resid(Weig),
            "rel_diff_W_svd_vs_float64_solve": float(np.linalg.norm(Wsvd - W64) / np.linalg.norm(Wsvd)),
            "rel_diff_W_float32_vs_svd": float(np.linalg.norm(W32 - Wsvd) / np.linalg.norm(Wsvd)),
            "norm_W_svd": float(np.linalg.norm(Wsvd)),
            "norm_W_float32": float(np.linalg.norm(W32)),
            "forward_error_bound_float64(cond*eps)": float(geom["cond_gram_plus_lambda"] * np.finfo(np.float64).eps),
            "forward_error_bound_float32(cond*eps)": float(geom["cond_gram_plus_lambda"] * np.finfo(np.float32).eps),
            "attacker_fit_insample_r2": insample_fit,
        }

        # (i) historical N0: in-sample on ALL test rows (15,060), mixed precision vs float64 vs stable SVD
        hm = historical_native_r2(H32, y, LAM, "historical_mixed_precision")
        h64 = historical_native_r2(H32, y, LAM, "float64")
        Ha = H - H.mean(0)
        Za = onehot(y, int(y.max()) + 1)
        Zc = Za - Za.mean(0)
        Ua, sa, Vta = np.linalg.svd(Ha, full_matrices=False)
        Wa = Vta.T @ (np.diag(sa / (sa ** 2 + LAM)) @ (Ua.T @ Zc))
        n0_svd = 1.0 - float(((Zc - Ha @ Wa) ** 2).sum()) / float((Zc ** 2).sum())
        # reconstruct the mixed-precision W to test whether it minimises the ridge objective
        Ha32 = H32 - H32.mean(axis=0, keepdims=True)
        gram_mixed = Ha32.T @ Ha32 + LAM * np.eye(d)
        Wm = np.linalg.solve(gram_mixed, Ha32.T @ Zc)
        Ga = Ha.T @ Ha
        eva = np.linalg.eigvalsh(Ga)
        n0 = {
            "rows": int(len(y)),
            "historical_mixed_precision_raw": hm["raw"],
            "historical_mixed_precision_clamped": hm["clamped"],
            "float64_raw": h64["raw"],
            "stable_svd_float64": n0_svd,
            "abs_diff_float64_vs_svd": abs(h64["raw"] - n0_svd),
            "all_rows_gram_min_eig": float(eva[0]),
            "all_rows_cond_gram": float(eva[-1] / eva[0]),
            "all_rows_float32_gram_error_spectral": float(np.linalg.norm(
                (Ha32.T @ Ha32).astype(np.float64) - Ga, 2)),
            "ridge_objective_J0_equals_SStot": ridge_objective(Zc, Ha, np.zeros_like(Wa), LAM),
            "ridge_objective_stable_W": ridge_objective(Zc, Ha, Wa, LAM),
            "ridge_objective_mixed_precision_W": ridge_objective(Zc, Ha, Wm, LAM),
            "mixed_W_is_not_a_ridge_minimiser": bool(ridge_objective(Zc, Ha, Wm, LAM)
                                                     > ridge_objective(Zc, Ha, np.zeros_like(Wa), LAM)),
            "norm_W_mixed_over_norm_W_stable": float(np.linalg.norm(Wm) / np.linalg.norm(Wa)),
        }
        g2 = r2_vs_prior(Y[A], P["G2_pred"], P["G2_prior"])
        pairs[f"{PURPOSE}__{attr}"] = {"K": K, "N0_in_sample_all_test_rows": n0, "G1_fixed_ridge_solver": solver,
                                       "G1_heldout_assessment": heldout, "G2_stored_heldout": g2}

    out = {
        "schema": "pilot_numerics_v1",
        "date": "2026-10-02",
        "role": "method owner (role 2), matched removal benchmark v1",
        "status": "versioned diagnostic; does NOT replace any primary pilot number or decision",
        "inputs": {k: {"path": str(v).replace(os.path.expanduser("~"), "~"), "sha256": sha(v)}
                   for k, v in files.items()},
        "representation": REP_KEY,
        "lambda": LAM,
        "fit_role_geometry_attacker_fit": geom,
        "pairs": pairs,
        "numpy": np.__version__,
    }
    (HERE / "pilot_numerics.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps(o["fit_role_geometry_attacker_fit"], indent=1))
    for k, v in o["pairs"].items():
        print(k, json.dumps(v, indent=1))

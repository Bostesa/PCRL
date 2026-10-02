"""Matched removal defenses for the matched removal benchmark v1 (method owner, 2026-10-02).

Arms B / C: OFFICIAL LEACE (EleutherAI concept-erasure 0.2.4; arXiv 2306.03819), wrapped, never re-implemented.
  * Only ``concept_erasure.LeaceFitter`` / ``LeaceEraser`` are used, with the package's DEFAULT settings
    (method="leace", affine=True, constrain_cov_trace=True, shrinkage=True, svd_tol=0.01). Never OracleEraser
    (it needs the concept label at inference), never the in-house ``pcrl.models.baselines.LEACEEraser``,
    never an orthogonal projection or a direction truncation substituted for the official map.
  * Read from the installed source (identical to upstream tag v0.2.4, commit 9b18b3d5c...):
      - the map is OBLIQUE: W = Sigma_xx^{-1/2} (eigh, pinv-style mask L > L_max * d * eps), u = left singular
        vectors of W Sigma_xz with singular value > svd_tol, P = I - W^+ u u' W, x' = x - (x - mean_x) (I - P)'.
      - Sigma_xx is the Ledoit-Wolf-type optimal LINEAR SHRINKAGE estimate (alpha S_n + beta tr(S_n)/d I),
        hence full rank whenever tr(S_n) > 0; Sigma_xz is the Bessel-corrected sample cross-covariance.
        Consequence: P Sigma_xz = 0 still holds exactly (when no singular value is truncated), but the
        least-squares optimality of LEACE is w.r.t. the shrunk covariance, not the sample one (the upstream
        test asserts first-order optimality only for shrinkage=False).
      - svd_tol = 0.01 is an ABSOLUTE threshold on singular values of the whitened cross-covariance
        (invariant to rescaling H, not to the concept coding). Directions below it are NOT erased; the
        package docstring says this "may leave trace correlations intact". We record every singular value
        and the truncation count; the native check reports the consequence.
      - constrain_cov_trace mixes P with Q = I - u u' only if tr(P S P') > tr(S). With the same S used for
        whitening and for the trace this cannot happen in exact arithmetic (tr(P S P') = tr(S) - tr(u' S u));
        we record whether it fired. If it fires, the erasure condition is no longer exact.
      - Inference uses only x (``LeaceEraser.__call__``): no concept labels at transform time.
      - Directions outside the fitting covariance support: Sigma_xz lies in range(S), so u does too and P acts
        as the identity on range(S)^perp. Concept signal that a fresh row carries outside the fit support is
        passed through unchanged (tested in test_20).
  * The map is FIXED after fitting: ``transform`` applies the stored (proj_left, proj_right, mean_x) to any
    rows; it never refits and never recentres on the scored rows.

Arm D: unclipped Gaussian noise, delegated to ``releases.gaussian_release`` (one persistent draw per person).

Public API (frozen for role 3 / the coordinator):
    verify_official_leace() -> dict
    concat_marginal_onehots(labels: Mapping[str, array], n_classes: Mapping[str, int]) -> (Z, concept_spec)
    fit_leace(H_fit, Z_fit_onehot, *, dtype=np.float64, fit_row_ids=None, concept_spec=None,
              auth=None, synthetic=None) -> LeaceMap
    LeaceMap.transform(H) -> ndarray (float64)
    LeaceMap.P -> (d, d) ndarray
    LeaceMap.save(dir) / LeaceMap.load(dir)
    LeaceMap.native_check(H_fit, Z_fit, *, tol_rel=NATIVE_TOL_REL, tol_r2=NATIVE_TOL_R2) -> dict
    crosscov_stats(H, Z, *, scale_H=None) -> dict          (held-out cross-covariance norm on any rows)
    alias_test(map_b, map_c, *, atol=ALIAS_ATOL, H_probe=None) -> dict
    noise_release(H, sigma, seed) -> ndarray
    scale_report(H_train, sigmas=DOCUMENTED_ADULT_SIGMAS) -> dict
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .releases import DOCUMENTED_ADULT_SIGMAS, gaussian_release

# --------------------------------------------------------------------------------------------------
# Pins (BENCH_DESIGN.md "Official LEACE")
# --------------------------------------------------------------------------------------------------
LEACE_PACKAGE = "concept-erasure"
LEACE_PKG_VERSION = "0.2.4"
LEACE_UPSTREAM_REPO = "https://github.com/EleutherAI/concept-erasure"
LEACE_UPSTREAM_TAG = "v0.2.4"
LEACE_UPSTREAM_COMMIT = "9b18b3d5c73f552798212c51d6533d649fa434cd"
# sha256 over the installed .py files, sorted by site-packages-relative path, hashing path bytes then file bytes
# (no separators); see installed_source_sha256().
LEACE_SOURCE_SHA256 = "fffac29d5914f334396f4af1f6b65ba09fcbf658fa8a013972bb50cbc1568597"
LEACE_PAPER = "Belrose et al. 2023, LEACE: Perfect linear concept erasure in closed form, arXiv:2306.03819"
LEACE_DEFAULTS = {"method": "leace", "affine": True, "constrain_cov_trace": True, "shrinkage": True,
                  "svd_tol": 0.01}

# Native-check tolerances (float64). The fitted condition is exact in exact arithmetic; these bound rounding.
NATIVE_TOL_REL = 1e-6   # max_ij |Cov(erased H_j, Z_i)| / (sd(H_j) sd(Z_i)), pre-erasure sds, fit rows
NATIVE_TOL_R2 = 1e-6    # fit-row affine OLS R2 of each concept block on erased H
ALIAS_ATOL = 1e-10      # max |P_B - P_C| and max |mean_B - mean_C| for an exact alias
STRUCTURAL_ZERO_SV = 1e-10  # whitened cross-cov singular values below this are structural zeros (dimensionless)

SCHEMA_LEACE_MAP = "stored_model_eval.leace_map/v1"
_NPZ = "leace_map.npz"
_META = "leace_map.json"


class OfficialLeaceMismatch(RuntimeError):
    """The importable concept_erasure is not the pinned official 0.2.4 source."""


# --------------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_source_sha256() -> tuple[str, str, int]:
    """(sha256, package_dir, n_files) of the importable concept_erasure .py tree."""
    import concept_erasure

    pkg = Path(concept_erasure.__file__).resolve().parent
    base = pkg.parent
    h = hashlib.sha256()
    files = sorted(pkg.rglob("*.py"))
    for p in files:
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest(), str(pkg), len(files)


def verify_official_leace() -> dict:
    """Assert the installed package is concept-erasure 0.2.4 with the pinned source hash; return provenance."""
    from importlib.metadata import version

    import torch

    v = version(LEACE_PACKAGE)
    sha, pkg, n = installed_source_sha256()
    if v != LEACE_PKG_VERSION:
        raise OfficialLeaceMismatch(f"concept-erasure version {v} != pinned {LEACE_PKG_VERSION}")
    if sha != LEACE_SOURCE_SHA256:
        raise OfficialLeaceMismatch(f"installed source sha256 {sha} != pinned {LEACE_SOURCE_SHA256} ({pkg})")
    return {"package": LEACE_PACKAGE, "version": v, "upstream_repo": LEACE_UPSTREAM_REPO,
            "upstream_tag": LEACE_UPSTREAM_TAG, "upstream_commit": LEACE_UPSTREAM_COMMIT,
            "installed_source_sha256": sha, "installed_source_files": n, "installed_path": pkg,
            "torch": torch.__version__, "numpy": np.__version__, "paper": LEACE_PAPER,
            "classes_used": ["concept_erasure.LeaceFitter", "concept_erasure.LeaceEraser"]}


# --------------------------------------------------------------------------------------------------
# Concept encoding
# --------------------------------------------------------------------------------------------------


def concat_marginal_onehots(labels: Mapping[str, np.ndarray], n_classes: Mapping[str, int]
                            ) -> tuple[np.ndarray, list[dict]]:
    """Concatenate the MARGINAL one-hot block of each attribute (in the given order).

    K is the frozen class count of the attribute (not the classes observed in these rows). Erasing these
    columns zeroes the cross-covariance with each marginal; it says nothing about intersections
    (e.g. 1[race=k, sex=j]), which are not linear in the concatenated marginals.
    """
    blocks, spec, start = [], [], 0
    n = None
    for name, y in labels.items():
        y = np.asarray(y).astype(np.int64)
        K = int(n_classes[name])
        if n is None:
            n = len(y)
        if len(y) != n:
            raise ValueError("all label arrays must have the same length")
        if y.min() < 0 or y.max() >= K:
            raise ValueError(f"{name}: labels outside [0, {K})")
        blocks.append(np.eye(K, dtype=np.float64)[y])
        spec.append({"name": str(name), "n_classes": K, "col_start": start, "col_stop": start + K,
                     "encoding": "marginal one-hot", "class_counts": np.bincount(y, minlength=K).tolist()})
        start += K
    return np.concatenate(blocks, axis=1), spec


def _default_spec(k: int) -> list[dict]:
    return [{"name": "concept", "n_classes": k, "col_start": 0, "col_stop": k, "encoding": "as given"}]


def _ids_hash(ids) -> str:
    return _sha256_bytes(np.ascontiguousarray(np.asarray(ids).astype("<i8")).tobytes())


# --------------------------------------------------------------------------------------------------
# LeaceMap
# --------------------------------------------------------------------------------------------------


@dataclass
class LeaceMap:
    """Fitted official LEACE map: x' = x - ((x - mean_x) @ proj_right.T) @ proj_left.T (LeaceEraser.__call__)."""

    proj_left: np.ndarray
    proj_right: np.ndarray
    mean_x: np.ndarray
    metadata: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)  # fitting statistics kept for audit (mean_z, sigma_xx, ...)

    # -- application ------------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.proj_left.shape[0])

    @property
    def P(self) -> np.ndarray:
        return np.eye(self.dim) - self.proj_left @ self.proj_right

    def _eraser(self):
        import torch
        from concept_erasure import LeaceEraser

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))  # noqa: E731
        return LeaceEraser(t(self.proj_left), t(self.proj_right), t(self.mean_x))

    def transform(self, H) -> np.ndarray:
        """Apply the FIXED fitted map (original fitting mean) to any rows, via the official LeaceEraser."""
        import torch

        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 2 or H.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) input, got {H.shape}")
        with torch.no_grad():
            out = self._eraser()(torch.from_numpy(np.ascontiguousarray(H)))
        return out.numpy().astype(np.float64, copy=True)

    # -- checks -----------------------------------------------------------------------------------
    def native_check(self, H_fit, Z_fit, *, tol_rel: float = NATIVE_TOL_REL, tol_r2: float = NATIVE_TOL_R2) -> dict:
        """LEACE's own condition on the FITTING rows: Cov(erased H, Z) = 0 and fit-row affine OLS R2 = 0.

        Scope: the fitting sample only. Uses Bessel-corrected sample statistics (as the package does). The
        relative cross-covariance divides by pre-erasure per-dim sd(H) and sd(Z); columns with sd(Z) = 0
        (class absent in fit rows) are reported and excluded from the relative maximum.
        """
        H_fit = np.asarray(H_fit, dtype=np.float64)
        Z_fit = np.asarray(Z_fit, dtype=np.float64)
        if Z_fit.ndim == 1:
            Z_fit = Z_fit[:, None]
        E = self.transform(H_fit)
        pre = crosscov_stats(H_fit, Z_fit)
        post = crosscov_stats(E, Z_fit, scale_H=H_fit.std(0, ddof=1))
        spec = self.metadata.get("concept_spec") or _default_spec(Z_fit.shape[1])
        blocks = []
        for b in spec:
            Zb = Z_fit[:, b["col_start"]:b["col_stop"]]
            blocks.append({"name": b["name"], "ols_r2_fit_rows_erased": _ols_r2(E, Zb),
                           "ols_r2_fit_rows_untreated": _ols_r2(H_fit, Zb),
                           "crosscov_max_abs_erased": crosscov_stats(E, Zb)["max_abs"]})
        joint = _ols_r2(E, Z_fit)
        # Implementation-level bound of the DEFAULT official map: the whitened cross-covariance left after
        # erasure, W Cov(E, Z) with W = Sigma_used^{-1/2}, has spectral norm <= svd_tol (only truncated
        # directions remain). Exact LEACE (no truncation) makes it 0.
        wres = self._whitened_residual(E, Z_fit)
        trunc = int(self.metadata.get("diagnostics", {}).get("n_singular_values_nonzero_truncated", 0))
        within = (post["max_abs_rel"] <= tol_rel) and all(b["ols_r2_fit_rows_erased"] <= tol_r2 for b in blocks)
        if within:
            status = "WITHIN_TOLERANCE"
        elif trunc > 0:
            status = "OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION"
        elif self.metadata.get("diagnostics", {}).get("cov_trace_constraint_fired"):
            status = "OUTSIDE_TOLERANCE_TRACE_CONSTRAINT"
        else:
            status = "OUTSIDE_TOLERANCE"
        return {"scope": "fitting rows only (finite-sample condition); not a held-out or nonlinear claim",
                "n_rows": int(len(H_fit)), "status": status, "tol_rel": tol_rel, "tol_r2": tol_r2,
                "crosscov_max_abs_erased": post["max_abs"], "crosscov_max_abs_rel_erased": post["max_abs_rel"],
                "crosscov_fro_erased": post["fro"], "crosscov_max_abs_untreated": pre["max_abs"],
                "crosscov_max_abs_rel_untreated": pre["max_abs_rel"], "zero_variance_concept_columns":
                post["zero_variance_concept_columns"], "ols_r2_joint_erased": joint, "blocks": blocks,
                "n_singular_values_nonzero_truncated": trunc,
                "whitened_residual_spectral_norm": wres,
                "svd_tol": float(self.metadata.get("settings_used", {}).get("svd_tol", float("nan"))),
                "implementation_bound_holds": bool(
                    wres <= float(self.metadata.get("settings_used", {}).get("svd_tol", 0.0)) * (1 + 1e-9) + 1e-12)}

    def _whitened_residual(self, E, Z) -> float:
        S = self.arrays.get("sigma_xx_used")
        if S is None:
            return float("nan")
        n = len(E)
        C = (E - E.mean(0)).T @ (Z - Z.mean(0)) / (n - 1)
        L, V = np.linalg.eigh(S)
        mask = L > (L[-1] * len(L) * np.finfo(np.float64).eps)
        W = (V * np.where(mask, 1.0 / np.sqrt(np.where(mask, L, 1.0)), 0.0)) @ V.T
        return float(np.linalg.norm(W @ C, 2))

    # -- persistence ------------------------------------------------------------------------------
    def save(self, directory) -> dict:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / _NPZ, proj_left=self.proj_left, proj_right=self.proj_right, mean_x=self.mean_x,
                 **{k: np.asarray(v) for k, v in self.arrays.items()})
        meta = dict(self.metadata)
        meta["npz_file"] = _NPZ
        meta["npz_sha256"] = _sha256_file(d / _NPZ)
        (d / _META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return {"npz": str(d / _NPZ), "npz_sha256": meta["npz_sha256"], "json": str(d / _META),
                "json_sha256": _sha256_file(d / _META)}

    @classmethod
    def load(cls, directory, *, verify_package: bool = True) -> "LeaceMap":
        d = Path(directory)
        meta = json.loads((d / _META).read_text())
        if meta.get("schema") != SCHEMA_LEACE_MAP:
            raise ValueError(f"unexpected schema {meta.get('schema')}")
        got = _sha256_file(d / meta["npz_file"])
        if got != meta["npz_sha256"]:
            raise ValueError(f"{d / meta['npz_file']} sha256 {got} != recorded {meta['npz_sha256']}")
        if verify_package:
            verify_official_leace()
        with np.load(d / meta["npz_file"]) as z:
            arrs = {k: z[k] for k in z.files}
        pl, pr, mu = arrs.pop("proj_left"), arrs.pop("proj_right"), arrs.pop("mean_x")
        meta = {k: v for k, v in meta.items() if k not in ("npz_file", "npz_sha256")}
        return cls(pl, pr, mu, meta, arrs)


# --------------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------------


def fit_leace(H_fit, Z_fit_onehot, *, dtype=np.float64, fit_row_ids: Sequence[int] | None = None,
              concept_spec: list[dict] | None = None, auth=None, synthetic: bool | None = None) -> LeaceMap:
    """Fit official LEACE (LeaceFitter, default settings) on defense_fit rows only.

    H_fit: (n, d) frozen representations of the defense_fit rows. Z_fit_onehot: (n, k) concept encoding
    (use concat_marginal_onehots for a policy set). fit_row_ids: the defense_fit row/unit ids (hashed into the
    metadata). If ``auth`` is given, ``auth.check("official LEACE eraser", synthetic)`` gates the fit.
    """
    if auth is not None:
        if synthetic is None:
            raise ValueError("pass synthetic=True/False together with auth")
        auth.check("official LEACE eraser", synthetic)
    import torch
    from concept_erasure import LeaceFitter

    prov = verify_official_leace()
    if np.dtype(dtype) != np.float64:
        raise ValueError("the benchmark fits LEACE in float64 only")
    H = np.ascontiguousarray(np.asarray(H_fit, dtype=np.float64))
    Z = np.asarray(Z_fit_onehot, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.ascontiguousarray(Z)
    n, d = H.shape
    if Z.shape[0] != n:
        raise ValueError("H_fit and Z_fit_onehot row counts differ")
    if not (np.isfinite(H).all() and np.isfinite(Z).all()):
        raise ValueError("non-finite values in fitting inputs")
    k = Z.shape[1]
    spec = concept_spec if concept_spec is not None else _default_spec(k)
    if spec[-1]["col_stop"] != k:
        raise ValueError("concept_spec does not cover the Z columns")

    fitter = LeaceFitter(d, k, dtype=torch.float64)  # official defaults, see LEACE_DEFAULTS
    fitter.update(torch.from_numpy(H), torch.from_numpy(Z))
    eraser = fitter.eraser
    used = {"method": fitter.method, "affine": fitter.affine, "constrain_cov_trace": fitter.constrain_cov_trace,
            "shrinkage": fitter.shrinkage, "svd_tol": float(fitter.svd_tol)}
    if used != LEACE_DEFAULTS:
        raise OfficialLeaceMismatch(f"LeaceFitter defaults differ from the pinned defaults: {used}")
    assert eraser.bias is not None  # affine=True

    pl = eraser.proj_left.numpy().copy()
    pr = eraser.proj_right.numpy().copy()
    mu = eraser.bias.numpy().copy()
    sxx = fitter.sigma_xx.numpy().copy()     # shrunk covariance actually used
    sxz = fitter.sigma_xz.numpy().copy()
    S_n = ((fitter.sigma_xx_ + fitter.sigma_xx_.mH) / 2).numpy() / n   # biased sample cov fed to shrinkage
    diag = _diagnostics(sxx, sxz, S_n, pl, pr, used["svd_tol"])

    meta = {
        "schema": SCHEMA_LEACE_MAP,
        "estimator": "official LEACE (concept_erasure.LeaceFitter -> LeaceEraser), oblique whitened projection",
        "settings_used": used,
        "dtype": "float64",
        "n_fit": int(n), "dim": int(d), "z_dim": int(k),
        "fit_row_ids_sha256": _ids_hash(fit_row_ids) if fit_row_ids is not None else None,
        "fit_row_ids_count": int(len(fit_row_ids)) if fit_row_ids is not None else None,
        "fit_row_ids_hash_convention": "sha256 of ids as little-endian int64 bytes, in the given order",
        "H_fit_sha256": _sha256_bytes(H.tobytes()),
        "Z_fit_sha256": _sha256_bytes(Z.tobytes()),
        "concept_spec": spec,
        "rank": diag["erasure_rank"],
        "tolerances": {"svd_tol": used["svd_tol"],
                       "whitening_mask": "eig > eig_max * d * finfo(float64).eps (torch.linalg.pinv rule)",
                       "native_tol_rel": NATIVE_TOL_REL, "native_tol_r2": NATIVE_TOL_R2,
                       "alias_atol": ALIAS_ATOL},
        "diagnostics": diag,
        "provenance": prov,
        "inference_uses_labels": False,
        "transform_contract": "fixed map; original fitting mean; never refit or recentre on scored rows",
    }
    arrays = {"mean_z": fitter.mean_z.numpy().copy(), "sigma_xx_used": sxx, "sigma_xz": sxz,
              "singular_values_whitened_xz": np.asarray(diag["singular_values_whitened_xz"])}
    return LeaceMap(pl, pr, mu, meta, arrays)


def _diagnostics(sxx, sxz, S_n, pl, pr, svd_tol) -> dict:
    """Recompute (read-only) the quantities the official eraser derives, to record support and truncation."""
    d = sxx.shape[0]
    L, V = np.linalg.eigh(sxx)
    mask = L > (L[-1] * d * np.finfo(np.float64).eps)
    W = (V * np.where(mask, 1.0 / np.sqrt(np.clip(L, 1e-300, None)), 0.0)) @ V.T
    u, s_full, _ = np.linalg.svd(W @ sxz, full_matrices=False)
    s = s_full
    keep = s_full > svd_tol
    Winv = (V * np.where(mask, np.sqrt(np.clip(L, 0, None)), 0.0)) @ V.T
    P0 = np.eye(d) - (Winv @ u[:, keep]) @ (u[:, keep].T @ W)
    old_tr = float(np.trace(sxx))
    new_tr = float(np.trace(P0 @ sxx @ P0.T))
    P = np.eye(d) - pl @ pr
    Ls = np.linalg.eigvalsh((S_n + S_n.T) / 2)
    # shrinkage: sxx = a S_n + b I  (least-squares recovery of the package's coefficients)
    A = np.stack([S_n.ravel(), np.eye(d).ravel()], 1)
    (a, b), *_ = np.linalg.lstsq(A, sxx.ravel(), rcond=None)
    k_max = int(min(sxz.shape))
    return {
        "singular_values_whitened_xz": s.tolist(),
        "n_singular_values_truncated": int((~keep).sum()),
        # structural zeros (collinear one-hot columns: each marginal block has rank K-1) are not a loss;
        # a NONZERO singular value at or below svd_tol is a concept direction the official map leaves intact.
        "n_singular_values_nonzero_truncated": int(((~keep) & (s_full > STRUCTURAL_ZERO_SV)).sum()),
        "structural_zero_sv_threshold": STRUCTURAL_ZERO_SV,
        "erasure_rank": int(keep.sum()),
        "max_possible_rank": k_max,
        "cov_trace_constraint_fired": bool(new_tr > old_tr),
        "trace_sigma_before": old_tr, "trace_sigma_after_unconstrained": new_tr,
        "max_abs_P_package_minus_recomputed": float(np.abs(P - P0).max()),
        "sigma_used_eig_min": float(L[0]), "sigma_used_eig_max": float(L[-1]),
        "whitening_dirs_masked": int((~mask).sum()),
        "sample_cov_eig_min": float(Ls[0]), "sample_cov_eig_max": float(Ls[-1]),
        "sample_cov_rank": int((Ls > Ls[-1] * d * np.finfo(np.float64).eps).sum()),
        "shrinkage_alpha": float(a), "shrinkage_beta_abs": float(b),
        "shrinkage_target": "isotropic, trace-matched (tr(S_n)/d I)",
    }


# --------------------------------------------------------------------------------------------------
# Statistics shared by native and held-out checks
# --------------------------------------------------------------------------------------------------


def crosscov_stats(H, Z, *, scale_H=None) -> dict:
    """Bessel-corrected cross-covariance of H with Z on THESE rows (centred on these rows' own means; this is a
    statistic of the scored rows, not a recentring of the map). Relative = |C_ij| / (sd(H_j) sd(Z_i)), sd(H)
    from ``scale_H`` if given (e.g. pre-erasure sds)."""
    H = np.asarray(H, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(H)
    C = (H - H.mean(0)).T @ (Z - Z.mean(0)) / (n - 1)
    sH = np.asarray(scale_H, dtype=np.float64) if scale_H is not None else H.std(0, ddof=1)
    sZ = Z.std(0, ddof=1)
    zv = sZ <= 0
    denom = np.outer(np.where(sH > 0, sH, np.inf), np.where(zv, np.inf, sZ))
    rel = np.abs(C) / denom
    return {"max_abs": float(np.abs(C).max()), "fro": float(np.linalg.norm(C)),
            "max_abs_rel": float(rel.max()) if np.isfinite(denom).any() else float("nan"),
            "zero_variance_concept_columns": np.flatnonzero(zv).tolist(), "n_rows": int(n)}


def _ols_r2(H, Zb) -> float:
    """In-sample affine OLS R2 (minimum-norm lstsq with intercept), pooled over the columns of Zb."""
    H = np.asarray(H, dtype=np.float64)
    Zb = np.asarray(Zb, dtype=np.float64)
    X = np.c_[H - H.mean(0), np.ones(len(H))]
    B, *_ = np.linalg.lstsq(X, Zb, rcond=None)
    tot = float(((Zb - Zb.mean(0)) ** 2).sum())
    if tot <= 0:
        return float("nan")
    return 1.0 - float(((Zb - X @ B) ** 2).sum()) / tot


def alias_test(map_b: LeaceMap, map_c: LeaceMap, *, atol: float = ALIAS_ATOL, H_probe=None) -> dict:
    """B and C are an exact alias iff their full maps (P and fitting mean) agree within atol."""
    if map_b.dim != map_c.dim:
        return {"alias": False, "reason": "dimension differs"}
    dP = float(np.abs(map_b.P - map_c.P).max())
    dmu = float(np.abs(map_b.mean_x - map_c.mean_x).max())
    out = {"alias": bool(dP <= atol and dmu <= atol), "max_abs_P_diff": dP, "max_abs_mean_diff": dmu,
           "atol": atol, "rank_b": map_b.metadata.get("rank"), "rank_c": map_c.metadata.get("rank"),
           "fit_rows_same": map_b.metadata.get("fit_row_ids_sha256") == map_c.metadata.get("fit_row_ids_sha256")}
    if H_probe is not None:
        out["max_abs_transform_diff_probe"] = float(np.abs(map_b.transform(H_probe) - map_c.transform(H_probe)).max())
    return out


# --------------------------------------------------------------------------------------------------
# Arm D and scale reporting
# --------------------------------------------------------------------------------------------------


def noise_release(H, sigma: float, seed: int) -> np.ndarray:
    """Arm D: one persistent Gaussian draw per person (releases.gaussian_release; default_rng(seed) over the
    full matrix). No native certificate (unclipped)."""
    return gaussian_release(H, sigma, seed)


def scale_report(H_train, sigmas: Sequence[float] = DOCUMENTED_ADULT_SIGMAS) -> dict:
    """Scale of the training-role representation, so that absolute sigma can be read relative to it.

    sigma_train_scale := sqrt(tr(Cov(H_train)) / d) (RMS per-dimension sd). Equal absolute sigma across
    encoders/datasets is NOT a matched operating point.
    """
    H = np.asarray(H_train, dtype=np.float64)
    n, d = H.shape
    sd = H.std(0, ddof=1)
    rms = float(np.sqrt((sd ** 2).mean()))
    q = [1, 5, 25, 50, 75, 95, 99]
    nrm = np.linalg.norm(H, axis=1)
    cnrm = np.linalg.norm(H - H.mean(0), axis=1)
    med_cn = float(np.median(cnrm))
    per_sigma = [{"sigma_abs": float(s), "sigma_over_rms_sd": float(s / rms) if rms > 0 else float("inf"),
                  "sigma_over_median_sd": float(s / np.median(sd)) if np.median(sd) > 0 else float("inf"),
                  "expected_noise_norm": float(s * np.sqrt(d)),
                  "noise_norm_over_median_centred_norm": float(s * np.sqrt(d) / med_cn) if med_cn > 0 else float("inf"),
                  "per_dim_snr_rms": float((rms / s) ** 2) if s > 0 else float("inf")} for s in sigmas]
    return {"n_rows": int(n), "dim": int(d),
            "per_dim_sd": {"min": float(sd.min()), "median": float(np.median(sd)), "mean": float(sd.mean()),
                           "max": float(sd.max()), "values": sd.tolist()},
            "sigma_train_scale_rms_sd": rms,
            "norm_quantiles": dict(zip([f"q{p:02d}" for p in q], np.percentile(nrm, q).tolist())),
            "centred_norm_quantiles": dict(zip([f"q{p:02d}" for p in q], np.percentile(cnrm, q).tolist())),
            "sigmas": per_sigma,
            "note": "sigma_train_scale = RMS per-dim sd of the training-role rows; equal sigma != equal protection"}


__all__ = ["LEACE_PKG_VERSION", "LEACE_UPSTREAM_TAG", "LEACE_UPSTREAM_COMMIT", "LEACE_SOURCE_SHA256",
           "LEACE_DEFAULTS", "NATIVE_TOL_REL", "NATIVE_TOL_R2", "ALIAS_ATOL", "OfficialLeaceMismatch",
           "installed_source_sha256", "verify_official_leace", "concat_marginal_onehots", "fit_leace",
           "LeaceMap", "crosscov_stats", "alias_test", "noise_release", "scale_report"]

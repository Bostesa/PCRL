"""Pilot recipes, implemented against the EFFECTIVE_PROTOCOL (every setting read from it; nothing defaulted here).

Selection contract: every selecting function receives attacker_fit and attacker_val arrays ONLY. Assessment rows
are scored afterwards by `predict`-style calls on already selected models (pilot.py separates the two phases and
tests/test_17_pilot_units.py checks that the fit phase never receives an assessment row).

Recipes
  L        StandardScaler + multinomial LogisticRegression, C grid, select on attacker_val log-loss       A1
  GBT      HistGradientBoostingClassifier over the 20 frozen configs (early stopping inside attacker_fit) A1
  MLP      StandardScaler + MLPClassifier over the 18-config grid (early stopping inside attacker_fit)    A1
  NL       GBT-best vs MLP-best, chosen on attacker_val log-loss                                          A1
  LRT_A2   Gaussian class-conditional LRT on RELEASED attacker_fit rows, known sigma subtracted (R10)     A2
  LRT_A4   Gaussian class-conditional LRT on CLEAN attacker_fit rows + sigma^2 I (R09, stress test)       A4
  LO       frequency table P(s | y_task) on attacker_fit (label-only reference)
  G1 / G2 / RHO1   closed-form linear quantities (fixed ridge, relative ridge, held-out rho1^2)
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.metrics import log_loss

from .guards import FitAuthorization
from .metrics import cca_rho2

# --------------------------------------------------------------------------------------------------
# probability helpers
# --------------------------------------------------------------------------------------------------


def full_proba(model, X, K: int) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), K))
    out[:, np.asarray(model.classes_).astype(int)] = p
    return out


def val_log_loss(y, P, K: int, clip: float) -> float:
    P = np.clip(np.asarray(P, dtype=np.float64), clip, 1.0)
    P = P / P.sum(1, keepdims=True)
    return float(log_loss(np.asarray(y).astype(int), P, labels=list(range(K))))


# --------------------------------------------------------------------------------------------------
# attacker families
# --------------------------------------------------------------------------------------------------


def _grid(family: str, cfg) -> list[dict]:
    if family == "L":
        return [{"C": float(c)} for c in cfg["C"]]
    if family == "GBT":
        return [dict(c) for c in cfg["configs"]]
    if family == "MLP":
        return [{"hidden_layer_sizes": list(h), "alpha": float(a), "learning_rate_init": float(lr)}
                for h in cfg["hidden_layer_sizes"] for a in cfg["alpha"] for lr in cfg["learning_rate_init"]]
    raise ValueError(family)


def _make(family: str, hp: dict, cfg, random_state: int | None = None):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if family == "L":
        from sklearn.linear_model import LogisticRegression
        if cfg["scaler"] != "standard":
            raise ValueError("only the standard scaler is supported")
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], solver=cfg["solver"],
                                                                  max_iter=int(cfg["max_iter"])))
    if family == "GBT":
        from sklearn.ensemble import HistGradientBoostingClassifier
        rs = int(cfg["random_state"]) if random_state is None else int(random_state)
        return HistGradientBoostingClassifier(
            learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"],
            min_samples_leaf=hp["min_samples_leaf"], l2_regularization=hp["l2_regularization"],
            max_iter=int(cfg["max_iter"]), early_stopping=bool(cfg["early_stopping"]),
            validation_fraction=float(cfg["validation_fraction"]), n_iter_no_change=int(cfg["n_iter_no_change"]),
            scoring=cfg["scoring"], random_state=rs)
    if family == "MLP":
        from sklearn.neural_network import MLPClassifier
        if cfg["scaler"] != "standard":
            raise ValueError("only the standard scaler is supported")
        rs = int(cfg["random_state"]) if random_state is None else int(random_state)
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(hp["hidden_layer_sizes"]), alpha=hp["alpha"],
            learning_rate_init=hp["learning_rate_init"], max_iter=int(cfg["max_iter"]),
            early_stopping=bool(cfg["early_stopping"]), validation_fraction=float(cfg["validation_fraction"]),
            n_iter_no_change=int(cfg["n_iter_no_change"]), solver=cfg["solver"], batch_size=cfg["batch_size"],
            random_state=rs))
    raise ValueError(family)


def fit_family(family: str, X_fit, y_fit, X_val, y_val, K: int, cfg, clip: float, *, auth: FitAuthorization,
               synthetic: bool, what: str) -> dict:
    """Fit every config of `family` on attacker_fit, score attacker_val log-loss, keep the best.
    Only fit/val arrays enter; the returned selection table holds validation scores only."""
    auth.check(f"{family} ({what})", synthetic)
    y_fit, y_val = np.asarray(y_fit).astype(int), np.asarray(y_val).astype(int)
    table, best = [], None
    t0 = time.process_time()
    for hp in _grid(family, cfg):
        m = _make(family, hp, cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_fit, y_fit)
        ll = val_log_loss(y_val, full_proba(m, X_val, K), K, clip)
        row = {"hp": hp, "attacker_val_log_loss": ll}
        if family == "GBT":
            row["n_iter"] = int(m.n_iter_)
        if family == "MLP":
            row["n_iter"] = int(m[-1].n_iter_)
        table.append(row)
        if best is None or ll < best[0]:
            best = (ll, hp, m)
    return {"family": family, "model": best[2], "selected": best[1], "attacker_val_log_loss": best[0],
            "selection_table": table, "n_configs": len(table), "cpu_s": time.process_time() - t0}


def select_nl(gbt: dict, mlp: dict) -> dict:
    """NL selection between the family winners on attacker_val log-loss (ties -> GBT, listed first)."""
    winner = gbt if gbt["attacker_val_log_loss"] <= mlp["attacker_val_log_loss"] else mlp
    return {"selected_family": winner["family"],
            "candidates": {"GBT": gbt["attacker_val_log_loss"], "MLP": mlp["attacker_val_log_loss"]},
            "criterion": "attacker_val_log_loss"}


# --------------------------------------------------------------------------------------------------
# defense-informed and white-box Gaussian LRTs
# --------------------------------------------------------------------------------------------------


class GaussianClassLRT:
    """Class-conditional Gaussian LRT for r = h + N(0, sigma^2 I).

    mode "A2": fitted on RELEASED rows: C_k = cov_k(r) - sigma^2 I (no clean vectors).
    mode "A4": fitted on CLEAN rows:    C_k = cov_k(h)        (white-box population stress test).
    Eigenvalues of C_k are floored at eig_floor_rel * trace(T_k) / d, where T_k is the released (A2) or clean (A4)
    class covariance (positive by construction; trace(C_k) itself can be negative at large sigma in A2). The
    predictive covariance of a release is C_k + sigma^2 I. Class priors are attacker_fit frequencies.
    """

    def __init__(self, sigma: float, mode: str, eig_floor_rel: float, ddof: int, K: int):
        if mode not in ("A2", "A4"):
            raise ValueError(mode)
        self.sigma, self.mode, self.rel, self.ddof, self.K = float(sigma), mode, float(eig_floor_rel), int(ddof), K

    def fit(self, X_fit, y_fit, *, auth: FitAuthorization, synthetic: bool):
        auth.check(f"LRT_{self.mode}", synthetic)
        X = np.asarray(X_fit, dtype=np.float64)
        y = np.asarray(y_fit).astype(int)
        d = X.shape[1]
        self.params, self.floors = {}, {}
        counts = np.bincount(y, minlength=self.K)[: self.K]
        self.log_prior = np.full(self.K, -np.inf)
        for k in range(self.K):
            Xk = X[y == k]
            if len(Xk) < 2:
                continue
            mu = Xk.mean(0)
            T = np.cov(Xk, rowvar=False, ddof=self.ddof).reshape(d, d)
            C = T - self.sigma ** 2 * np.eye(d) if self.mode == "A2" else T
            floor = self.rel * float(np.trace(T)) / d
            ev, U = np.linalg.eigh((C + C.T) / 2)
            n_floored = int((ev < floor).sum())
            ev = np.maximum(ev, floor) + self.sigma ** 2
            self.params[k] = (mu, U, ev)
            self.floors[k] = {"floor": floor, "n_eigen_floored": n_floored, "n_rows": int(len(Xk))}
            self.log_prior[k] = np.log(counts[k] / counts.sum())
        return self

    def predict_proba(self, R) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        L = np.full((len(R), self.K), -np.inf)
        for k, (mu, U, ev) in self.params.items():
            Z = (R - mu) @ U
            L[:, k] = -0.5 * ((Z ** 2 / ev).sum(1) + np.log(ev).sum()) + self.log_prior[k]
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    def describe(self) -> dict:
        return {"mode": self.mode, "sigma": self.sigma, "eig_floor_rel": self.rel, "ddof": self.ddof,
                "classes_modelled": sorted(int(k) for k in self.params), "floors": self.floors}


# --------------------------------------------------------------------------------------------------
# label-only reference
# --------------------------------------------------------------------------------------------------


def fit_label_only(s_fit, t_fit, K_s: int, K_t: int, alpha: float, *, auth, synthetic) -> np.ndarray:
    """Frequency table P(s | y_task) on attacker_fit with Laplace alpha. Returns (K_t, K_s)."""
    auth.check("label-only reference table", synthetic)
    T = np.full((K_t, K_s), float(alpha))
    np.add.at(T, (np.asarray(t_fit).astype(int), np.asarray(s_fit).astype(int)), 1.0)
    return T / T.sum(1, keepdims=True)


# --------------------------------------------------------------------------------------------------
# closed-form linear quantities
# --------------------------------------------------------------------------------------------------


def onehot(y, K: int, dtype=np.float64) -> np.ndarray:
    return np.eye(K, dtype=dtype)[np.asarray(y).astype(int)]


def fixed_ridge_fit(H_fit, y_fit, K: int, lam: float, dtype=np.float64) -> dict:
    """PCRL native estimator: one-hot Y, centred H and Y, W = (Hc'Hc + lam I)^-1 Hc'Yc (unnormalised Gram)."""
    H = np.asarray(H_fit, dtype=dtype)
    Y = onehot(y_fit, K, dtype)
    muH, muY = H.mean(0), Y.mean(0)
    Hc, Yc = H - muH, Y - muY
    G = Hc.T @ Hc + dtype(lam) * np.eye(H.shape[1], dtype=dtype)
    W = np.linalg.solve(G, Hc.T @ Yc)
    return {"W": W, "muH": muH, "muY": muY, "lam": lam, "dtype": np.dtype(dtype).name}


def linear_predict(model: dict, H) -> np.ndarray:
    H = np.asarray(H, dtype=model["W"].dtype)
    return (H - model["muH"]) @ model["W"] + model["muY"]


def r2_against_prior(y, pred, prior, K: int) -> float:
    """1 - SS_res / SS_tot with SS_tot around the given prior (attacker_fit means). Unclamped."""
    Y = onehot(y, K)
    pred = np.asarray(pred, dtype=np.float64)
    return 1.0 - float(((Y - pred) ** 2).sum()) / float(((Y - np.asarray(prior)[None, :]) ** 2).sum())


def historical_native_r2(H, y, lam: float, variant: str) -> dict:
    """N0/N1 estimator, line-for-line mirror of PCRL b96c412 pcrl/purposes/verification.py:89-103
    (LinearComplianceCertificate.check, called by eval_round4_dominant_axis.py:158-172):

        Z = eye(max(y)+1)[y]                       (float64)
        Hc = H - H.mean(0); Zc = Z - Z.mean(0)
        gram = Hc.T @ Hc + lam * eye(d)            (float32 Hc.T@Hc, then + float64 -> float64)
        W = solve(gram, Hc.T @ Zc); Zp = Hc @ W    (float64)
        R2 = 1 - sum((Zc - Zp)^2) / max(sum(Zc^2), 1e-12), clamped at 0

    variant "historical_mixed_precision": H as float32 (the torch representation dtype), so centring and the Gram
    product are float32 and everything after is float64, exactly as the historical code. "float64": H in float64.
    Returns raw (unclamped) and clamped values."""
    if variant == "historical_mixed_precision":
        Hn = np.asarray(H, dtype=np.float32)
    elif variant == "float64":
        Hn = np.asarray(H, dtype=np.float64)
    else:
        raise ValueError(variant)
    yi = np.asarray(y).astype(int)
    Z = np.eye(int(yi.max()) + 1)[yi]
    d = Hn.shape[1]
    Hc = Hn - Hn.mean(axis=0, keepdims=True)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    gram = Hc.T @ Hc + lam * np.eye(d)
    W = np.linalg.solve(gram, Hc.T @ Zc)
    Zp = Hc @ W
    ss_res = float(np.sum((Zc - Zp) ** 2))
    ss_tot = float(np.sum(Zc ** 2))
    raw = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"raw": raw, "clamped": float(max(0.0, raw)), "variant": variant, "gram_dtype": str(gram.dtype),
            "centring_dtype": str(Hc.dtype), "n_rows": int(len(yi)), "n_classes": int(Z.shape[1])}


def relridge_fit(H_fit, y_fit, K: int, rho: float, floor_rel: float) -> dict:
    """R02: affine one-hot least squares with numerical floor and relative ridge lambda = rho tr(S)/d."""
    H = np.asarray(H_fit, dtype=np.float64)
    Y = onehot(y_fit, K)
    muH, muY = H.mean(0), Y.mean(0)
    Hc, Yc = H - muH, Y - muY
    S = Hc.T @ Hc / len(H)
    ev, U = np.linalg.eigh(S)
    keep = ev >= floor_rel * ev.max()
    U, ev = U[:, keep], ev[keep]
    lam = rho * float(np.trace(S)) / H.shape[1]
    B = U @ np.diag(1.0 / (ev + lam)) @ U.T @ (Hc.T @ Yc / len(H))
    return {"W": B, "muH": muH, "muY": muY, "rho": rho, "n_dirs_kept": int(keep.sum())}


def fit_g2(H_fit, y_fit, H_val, y_val, K: int, cfg, *, auth, synthetic) -> dict:
    auth.check("G2 relative-ridge least squares", synthetic)
    table, best = [], None
    for rho in cfg["rho_grid"]:
        m = relridge_fit(H_fit, y_fit, K, float(rho), float(cfg["floor_rel"]))
        r2v = r2_against_prior(y_val, linear_predict(m, H_val), m["muY"], K)
        table.append({"rho": float(rho), "attacker_val_heldout_r2_fitmean": r2v})
        if best is None or r2v > best[0]:
            best = (r2v, m)
    return {"model": best[1], "selection_table": table, "selected": {"rho": best[1]["rho"]}}


def fit_rho1(H_fit, y_fit, K: int, eps_rel: float, *, auth, synthetic) -> dict:
    """Canonical directions on attacker_fit: class contrast a (length K) and representation direction b."""
    auth.check("held-out rho1^2 directions", synthetic)
    H = np.asarray(H_fit, dtype=np.float64)
    y = np.asarray(y_fit).astype(int)
    c = cca_rho2(H, y, eps_rel)
    if not isinstance(c, dict):
        return {"status": "NE", "reason": c.reason}
    a = np.zeros(K)
    a[: len(c["contrast"])] = c["contrast"]
    t = a[y]
    Hc = H - H.mean(0)
    Shh = Hc.T @ Hc / len(H)
    Shh += eps_rel * np.trace(Shh) / H.shape[1] * np.eye(H.shape[1])
    b = np.linalg.solve(Shh, Hc.T @ (t - t.mean()) / len(H))
    return {"status": "OK", "a": a, "b": b, "in_sample_rho1sq_fit": float(c["value"])}

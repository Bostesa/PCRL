"""fit-attackers -> score -> infer -> report, on admitted arrays (or synthetic fixtures).

Role contract: attackers fit on `attacker_fit`, hyper-parameters selected on `attacker_val`, everything
scored on `evaluation`. Score arrays are stored with the evaluation row ids, so `score` / `infer` are
pure functions of stored arrays (no refitting).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .access import ReleaseContract
from .attackers import FAMILIES
from .guards import FitAuthorization
from .inference import (cluster_bootstrap, decide_all, macro_auc_stat, resolve_units, worst_class_stat,
                        worst_pair_stat)
from .metrics import (cca_rho2, health, is_estimable, macro_ovr_auc, ols_r2, pairwise_auc,
                      r2_onehot_ridge, to_jsonable, worst_class_auc)
from .surfaces import build_surface

AUC_METRICS = ("macro_ovr_auc", "worst_class_auc", "worst_pair_auc")


def _role_idx(arrays, cfg, which):
    name = cfg["roles"][which]
    return np.flatnonzero(arrays["roles"].astype(str) == name)


def fit_attackers(arrays: dict, cfg: dict, auth: FitAuthorization, attackers=("linear", "gbt", "mlp"),
                  surfaces=None, random_state: int = 0) -> dict:
    synthetic = bool(arrays.get("_synthetic", arrays.get("synthetic", False)))
    surfaces = surfaces or cfg["surfaces"]
    fi, vi, ei = (_role_idx(arrays, cfg, k) for k in ("attacker_fit", "attacker_select", "score"))
    y = arrays["labels"] if "labels" in arrays else arrays["S"]
    rep = arrays.get("representations", arrays.get("H"))
    out = {"eval_row_ids": np.asarray(arrays["row_ids"])[ei], "y_eval": np.asarray(y)[ei],
           "units_eval": np.asarray(arrays["units"])[ei],
           "record_keys_eval": None if arrays.get("record_keys") is None else np.asarray(arrays["record_keys"])[ei],
           "probs": {}, "records": [], "timing_s": {}}
    # Owner addition 2026-10-02: closed-form linear quantities on the representation surface (primary endpoint
    # P1 = R02 held-out R2 vs tau). R02 is a least-squares fit on real attacker_fit rows, so it sits behind the
    # same authorization as the attacker slate.
    auth.check("closed-form linear quantities (native check, R02, held-out rho1^2)", synthetic)
    # 2026-10-02 repair: the contract comes from the manifest's "release" block when the arrays were admitted
    # from a manifest; the protocol-wide default (noise="none") applies only to manifest-free synthetic arrays.
    if arrays.get("_has_manifest"):
        contract = ReleaseContract.from_manifest(arrays.get("_release"))
    else:
        contract = ReleaseContract(**{k: v for k, v in cfg["release_contract"].items() if k in ("noise", "sigma")})
    out["closed_form"] = closed_form_linear(rep, np.asarray(y), fi, ei, cfg)
    for s in surfaces:
        if s != "rep" and arrays.get("outputs") is None:
            continue
        X, srec = build_surface(s, rep, arrays.get("outputs"))
        for a in attackers:
            t0 = time.perf_counter()
            att = FAMILIES[a](cfg["attackers"].get(a, {}), random_state)
            att.fit(X[fi], y[fi], X[vi], y[vi], auth=auth, synthetic=synthetic)
            out["probs"][(s, a)] = att.predict_proba(X[ei])
            out["timing_s"][f"{s}|{a}"] = time.perf_counter() - t0
            rec = att.access_record(s, contract)
            out["records"].append({"surface": srec.to_json(), "attacker": a, "selected": att.selected,
                                   "selection_table": att.selection_table, "access": rec.to_json(),
                                   "n_fit": int(len(fi)), "n_val": int(len(vi)), "n_eval": int(len(ei))})
    return out


def closed_form_linear(H, y, fit_idx, score_idx, cfg) -> dict:
    """Native check (PCRL convention: fit = score on the assessment rows), R02 held-out R2 (fit on attacker_fit,
    SS_tot around fit-row means, unclamped, scale-invariant), R02 with SS_tot around score-row means (reported
    sensitivity only), held-out rho1^2 and health (scale-dependent; descriptive only)."""
    from .metrics import r2_onehot_relridge, cca_rho2_heldout
    r = cfg["r2"]
    H = np.asarray(H, dtype=np.float64)
    ei = np.asarray(score_idx)
    native = r2_onehot_ridge(H[ei], y[ei], lam=r["ridge_lambda"])
    r02 = r2_onehot_relridge(H, y, fit_idx, score_idx)
    # sensitivity: same predictor, denominator centred on the scored rows
    yi = np.asarray(y).astype(np.int64)
    sens = None
    if is_estimable(r02):
        K = int(yi.max()) + 1
        Y = np.eye(K)[yi]
        tot_fit = float(((Y[ei] - Y[np.asarray(fit_idx)].mean(0)) ** 2).sum())
        tot_score = float(((Y[ei] - Y[ei].mean(0)) ** 2).sum())
        sens = 1.0 - (1.0 - float(r02)) * tot_fit / max(tot_score, 1e-300)
    rho = cca_rho2_heldout(H, y, fit_idx, score_idx, cfg["cca"]["eps_rel"])
    return {"native_r2_fit_eq_score_on_assessment": native, "r02_heldout_r2_fitmean": r02,
            "r02_heldout_r2_scoremean_sensitivity": sens,
            "rho1sq_heldout": rho.get("value") if isinstance(rho, dict) else rho,
            "health_descriptive_scale_dependent": health(H[ei], cfg["health"]["per_dim_std_min"],
                                                         cfg["health"]["effective_rank_min"])}


def save_scores(res: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrs = {"row_id": res["eval_row_ids"], "y": res["y_eval"], "unit": res["units_eval"]}
    if res.get("record_keys_eval") is not None:
        arrs["record_key"] = res["record_keys_eval"].astype(str)
    keys = []
    for (s, a), P in res["probs"].items():
        k = f"P__{s.replace('+', 'PLUS')}__{a}"
        arrs[k] = P
        keys.append(k)
    np.savez_compressed(out / "scores.npz", **arrs)
    (out / "fit_records.json").write_text(json.dumps(to_jsonable(
        {"records": res["records"], "timing_s": res["timing_s"], "prob_keys": keys,
         "closed_form": res.get("closed_form")}), indent=1, default=str))
    return out / "scores.npz"


def load_scores(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    probs = {}
    for k in z.files:
        if k.startswith("P__"):
            _, s, a = k.split("__")
            probs[(s.replace("PLUS", "+"), a)] = z[k]
    return {"row_ids": z["row_id"], "y": z["y"], "unit": z["unit"],
            "record_key": z["record_key"] if "record_key" in z.files else None, "probs": probs}


def score(y, P, min_support: int, macro_over: str = "all_declared") -> dict:
    wc = worst_class_auc(y, P, min_support)
    wp = pairwise_auc(y, P, min_support)
    return {"macro_ovr_auc": macro_ovr_auc(y, P, min_support, over=macro_over), "macro_over": macro_over,
            "worst_class_auc": wc["value"], "worst_class_argmax": wc["argmax"],
            "worst_class_coverage": wc["coverage"],
            "worst_pair_auc": wp["max"], "worst_pair_argmax": wp["argmax"], "pair_coverage": wp["coverage"],
            "pair_row_coverage": wp.get("row_coverage")}


def score_representation(H, y, cfg) -> dict:
    """Closed-form quantities on a representation (no attacker): R2 conventions, rho1^2, health."""
    r = cfg["r2"]
    c = cca_rho2(H, y, cfg["cca"]["eps_rel"], cfg["cca"]["tol"])
    return {"r2_onehot_ridge_fit_eq_score": r2_onehot_ridge(H, y, lam=r["ridge_lambda"]),
            "ols_r2_fit_eq_score": ols_r2(H, y),
            "cca_rho2": c["value"] if isinstance(c, dict) else c,
            "health": health(H, cfg["health"]["per_dim_std_min"], cfg["health"]["effective_rank_min"])}


def infer(scores: dict, cfg: dict, min_support: int | None = None) -> dict:
    bs = cfg["bootstrap"]
    ms = cfg["support"]["min_class_support"] if min_support is None else min_support
    units = resolve_units(scores["unit"], scores.get("record_key"))
    out = {"units": {k: v for k, v in units.items() if k != "unit_index"}, "results": {}}
    y = scores["y"]
    for (s, a), P in scores["probs"].items():
        mo = cfg["support"].get("macro_over", "all_declared")
        for m, fn in (("macro_ovr_auc", lambda w: macro_ovr_auc(y, P, ms, w, over=mo)),
                      ("worst_class_auc", worst_class_stat(y, P, ms)), ("worst_pair_auc", worst_pair_stat(y, P, ms))):
            r = cluster_bootstrap(fn, units["unit_index"], n_boot=bs["n_boot"], seed=bs["seed"], alpha=bs["alpha"])
            r["decisions"] = decide_all(r, cfg["bars"])
            out["results"][f"{s}|{a}|{m}"] = r
    return out


# --------------------------------------------------------------------------------------------------
# outcome categories
# --------------------------------------------------------------------------------------------------

def classify_outcome(*, fitting_check_passed, heldout_check_passed, attack_decision: str,
                     attacker_family: str, surface: str, scope: dict, metric: str | None = None,
                     under_assumptions: bool = True) -> dict:
    """Map one (check, attack) record to C1..C5 (PREP_CONTEXT labels).

    fitting_check_passed / heldout_check_passed: True / False / None (None = not estimable)
    attack_decision: decision of the recovery statistic vs its bar (ESTABLISHED_ABOVE etc.)
    scope: {"attacker_classes": [...], "surfaces": [...], "population": bool}
    """
    if fitting_check_passed is None:
        return {"category": "C5", "why": "fitting-sample check not estimable"}
    if fitting_check_passed is False:
        return {"category": "C1", "why": "implementation fails its own fitting-sample check"}
    if heldout_check_passed is None:
        return {"category": "C5", "why": "held-out check not estimable"}
    if heldout_check_passed is False:
        return {"category": "C2", "why": "fitting-sample check passes but does not generalise to held-out rows"}
    if attack_decision in ("UNRESOLVED", "NOT_ESTIMABLE"):
        return {"category": "C5", "why": f"recovery {attack_decision.lower()}"}
    if attack_decision == "ESTABLISHED_BELOW":
        return {"category": None, "why": "recovery established below the bar for this attacker/surface"}
    in_scope = (attacker_family in scope.get("attacker_classes", []) and surface in scope.get("surfaces", [])
                and (metric is None or scope.get("metric") in (None, metric)))
    if not in_scope:
        return {"category": "C3", "why": f"{attacker_family}/{metric} on {surface} is outside the stated "
                                         f"guarantee (scope {scope.get('attacker_classes')}/{scope.get('metric')} "
                                         f"on {scope.get('surfaces')})"}
    if scope.get("population") and under_assumptions:
        return {"category": "C4", "why": "in-scope recovery contradicts a population guarantee under its assumptions"}
    return {"category": "C2", "why": "in-scope recovery on held-out rows of a sample-only guarantee"}


def report(score_path: str | Path, infer_json: dict, cfg: dict) -> dict:
    sc = load_scores(score_path)
    ms = cfg["support"]["min_class_support"]
    rows = []
    for (s, a), P in sc["probs"].items():
        point = score(sc["y"], P, ms, cfg["support"].get("macro_over", "all_declared"))
        for m in AUC_METRICS:
            key = f"{s}|{a}|{m}"
            inf = infer_json["results"].get(key, {})
            rows.append({"surface": s, "attacker": a, "metric": m,
                         "point": to_jsonable(point[m]),
                         "interval": to_jsonable(inf.get("interval")),
                         "decisions": inf.get("decisions")})
    return {"n_eval_rows": int(len(sc["y"])), "independent_units": infer_json["units"], "rows": rows,
            "bars": cfg["bars"], "guarantee_scope": cfg["guarantee_scope"]}

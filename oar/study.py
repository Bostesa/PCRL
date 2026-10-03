"""Output-aware removal study runner (stages 4-6).

Reuses the matched benchmark's locked attacker slate, selection rules, utility probe and AUC/bootstrap machinery
(stored_model_eval.*, unchanged) and adds: exposure-cleaned roles with a certification carve-out, three access views
per defense, output-surface variants, release heads fitted only on the protected features, a cell-conditional
(defense-aware) attacker for finite surfaces, and real-data null / planted-leak controls.

Everything that selects anything sees only attacker_fit / attacker_val (or defense_fit for defenses and heads).
Assessment rows are scored only by already selected models (predict phase).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from stored_model_eval.bench import fit_slate, retrain_seeds, select_plus
from stored_model_eval.bench_effective import BENCH_EFFECTIVE, Tracked
from stored_model_eval.guards import FitAuthorization
from stored_model_eval.recipes import fit_family, fit_label_only, full_proba, val_log_loss

HOME = Path.home()
BENCH = HOME / "PCRL_eval_cache_private" / "bench_v1"
RUN = HOME / "PCRL_eval_cache_private" / "oar_v1" / "run"

CELLS = {
    "adult": {"purpose": "income_prediction", "attr": "sex", "rep": "rep_p0", "logits": "logits_income_prediction",
              "task": "task_income", "policy": ["race", "sex"], "K_s": 2, "K_t": 2},
    "hmda": {"purpose": "underwriting", "attr": "race", "rep": "rep_p0", "logits": "logits_underwriting",
             "task": "task_loan_decision", "policy": ["race", "ethnicity"], "K_s": 5, "K_t": 2},
}
SIGMA_STAR = {"adult": 2.0, "hmda": 4.0}          # frozen benchmark values, selected on attacker_val only
SEEDS = (0, 1, 2)
RELEASE_SEEDS = (0, 1, 2)
CERT_SALT, CERT_SHARE = "oar-cert-v1|", 0.20
HEAD_SALT, HEAD_VAL_SHARE = "oar-head-v1|", 0.20
SCORED = ("attacker_fit", "attacker_val", "assessment")
CLIP = 1e-12


def _u01(salt: str, key: str) -> float:
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 2 ** 32


def sha_arr(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- world / roles
def load_world(ds: str) -> dict:
    """Labels and the new oar-roles-v1. Exposure groups (records also present in the encoder-training split) leave
    every scored role; 20 % of the remaining attacker_fit groups (hash of canon_key) become the FARE certification
    set. defense_fit is unchanged. No outcome is read to form roles."""
    L = np.load(BENCH / "inputs" / f"{ds}_labels.npz")
    c = CELLS[ds]
    role0 = L["role"].astype("<U16")
    train_keys = set(L["canon_key"][L["split"] == "train"])
    exposed = (L["split"] == "test") & np.isin(L["canon_key"], list(train_keys))
    role = role0.copy()
    role[exposed & np.isin(role0, SCORED)] = "excluded_exposure"
    fitm = role == "attacker_fit"
    cert = np.array([_u01(CERT_SALT + ds + "|", k) < CERT_SHARE for k in L["canon_key"]]) & fitm
    role[cert] = "cert"
    # head holdout inside defense_fit (owner side; never an attacker role)
    dfm = role == "defense_fit"
    head_val = np.array([_u01(HEAD_SALT + ds + "|", k) < HEAD_VAL_SHARE for k in L["canon_key"]]) & dfm
    W = {"ds": ds, "row_id": L["row_id"], "unit": L["unit"], "canon_key": L["canon_key"], "role_bench": role0,
         "role": role, "head_val": head_val, "s": L[c["attr"]].astype(int), "t": L[c["task"]].astype(int),
         "policy": {a: L[a].astype(int) for a in c["policy"]}, "exposed": exposed}
    W["idx"] = {r: np.flatnonzero(role == r) for r in ("defense_fit", "cert", *SCORED)}
    W["idx"]["head_fit"] = np.flatnonzero(dfm & ~head_val)
    W["idx"]["head_val"] = np.flatnonzero(head_val)
    # benchmark noise draw rows: the ORIGINAL scored roles in ascending row_id (persistent per-person draws)
    sc = np.flatnonzero(np.isin(role0, SCORED))
    W["noise_rows"] = sc[np.argsort(L["row_id"][sc], kind="stable")]
    for r in ("attacker_fit", "attacker_val", "assessment", "cert"):
        u = W["unit"][W["idx"][r]]
        assert len(np.intersect1d(u, W["unit"][W["idx"]["defense_fit"]])) == 0
    return W


def role_summary(W) -> dict:
    out = {}
    for r, ix in W["idx"].items():
        out[r] = {"rows": int(len(ix)), "groups": int(len(np.unique(W["unit"][ix]))),
                  "row_ids_sha256": sha_arr(np.sort(W["row_id"][ix]).astype(np.int64))}
    out["excluded_exposure_rows"] = int((W["role"] == "excluded_exposure").sum())
    return out


def load_seed(ds: str, k: int) -> dict:
    F = np.load(BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
    c = CELLS[ds]
    return {"H": F[c["rep"]].astype(np.float64), "O": F[c["logits"]].astype(np.float64), "row_id": F["row_id"]}


# ------------------------------------------------------------------------------------------------- releases
def leace_map_id(ds, k, kind):
    """Benchmark map naming: C maps join the purpose's disallowed attributes in DECLARED order (not sorted)."""
    c = CELLS[ds]
    return f"{ds}__s{k}__{c['purpose']}__" + (f"B_{c['attr']}" if kind == "B" else "C_" + "+".join(c["policy"]))


def leace_release(ds, k, kind, H):
    from stored_model_eval.defenses import LeaceMap
    mid = leace_map_id(ds, k, kind)
    m = LeaceMap.load(BENCH / "defenses" / mid / "map", verify_package=True)
    return np.asarray(m.transform(H), dtype=np.float64), mid


def noise_release(W, H, sigma, rs):
    """Identical per-person draws to the benchmark: drawn over the ORIGINAL scored-role rows in ascending row_id."""
    from stored_model_eval.defenses import noise_release as nr
    R = np.full_like(H, np.nan)
    rows = W["noise_rows"]
    R[rows] = np.asarray(nr(H[rows], sigma, rs), dtype=np.float64)
    return R


def onehot(ids, K) -> np.ndarray:
    X = np.zeros((len(ids), K))
    X[np.arange(len(ids)), np.asarray(ids, int)] = 1.0
    return X


# ------------------------------------------------------------------------------------------------- heads (view 3)
def fit_head(X, t, W, K_t, E, auth, syn, what):
    """Release head on protected features: LR grid (the benchmark U2 family/budget), fitted on defense_fit minus a
    hash-held-out 20 %, selected on that holdout by log loss. Never sees attacker roles, labels of protected
    attributes, or historical outputs."""
    u2 = E["utility"]["U2"]
    cfg = {"C": u2["C"], "max_iter": u2["max_iter"], "solver": E["attackers"]["L"]["solver"],
           "scaler": E["attackers"]["L"]["scaler"]}
    hf, hv = W["idx"]["head_fit"], W["idx"]["head_val"]
    r = fit_family("L", X[hf], t[hf], X[hv], t[hv], K_t, cfg, E["attackers"]["log_loss_clip"], auth=auth,
                   synthetic=syn, what=what)
    return r


def head_outputs(model, X, K_t):
    P = full_proba(model, X, K_t)
    return np.log(np.clip(P, CLIP, 1.0))


# ------------------------------------------------------------------------------------------------- output variants
def output_variants(O):
    """Historical full logit vector; class probability vector (softmax); frozen-head hard prediction (argmax, the
    head's own decision rule, one-hot)."""
    Z = O - O.max(1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    hard = O.argmax(1)
    return {"full": O, "prob": P, "hard": onehot(hard, O.shape[1])}, hard


def alias_report(O) -> dict:
    """Is the probability vector an invertible function of the logit vector (and vice versa)?"""
    v, hard = output_variants(O)
    K = O.shape[1]
    centred = O - O.mean(1, keepdims=True)
    logp = np.log(np.clip(v["prob"], 1e-300, 1))
    recon = logp - logp.mean(1, keepdims=True)
    rep = {"K": K, "prob_determines_centred_logits_maxabs": float(np.max(np.abs(recon - centred))),
           "logit_sum_sd": float(np.std(O.sum(1))), "logit_sum_is_constant": bool(np.std(O.sum(1)) < 1e-9),
           "full_vs_prob": ("alias (logit offset constant)" if np.std(O.sum(1)) < 1e-9 else
                            "NOT alias: the logit vector carries the per-row offset (sum of logits) that the "
                            "probability vector discards; probability = function of logits, not invertible"),
           "hard_is_function_of_prob": True}
    return rep


# ------------------------------------------------------------------------------------------------- attackers
class CellConditional:
    """Defense-aware attacker for a finite surface: P(s | cell) from attacker_fit counts with Dirichlet smoothing
    alpha toward the attacker_fit prior; alpha selected on attacker_val log loss. Cells unseen in fit -> prior."""

    def __init__(self, alpha: float, K: int):
        self.alpha, self.K = float(alpha), int(K)

    @staticmethod
    def cell_key(X):
        X = np.asarray(X)
        return np.array([hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest()[:16] for r in np.round(X, 9)])

    def fit(self, X, s):
        s = np.asarray(s).astype(int)
        keys = self.cell_key(X)
        self.prior = np.bincount(s, minlength=self.K) / len(s)
        self.table = {}
        for key in np.unique(keys):
            c = np.bincount(s[keys == key], minlength=self.K).astype(float)
            self.table[key] = (c + self.alpha * self.prior) / (c.sum() + self.alpha)
        self.n_cells = len(self.table)
        return self

    def predict_proba(self, X):
        keys = self.cell_key(X)
        return np.stack([self.table.get(k, self.prior) for k in keys])


CC_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def fit_cell_conditional(Xf, sf, Xv, sv, K, clip):
    table, best = [], None
    for a in CC_ALPHAS:
        m = CellConditional(a, K).fit(Xf, sf)
        ll = val_log_loss(np.asarray(sv).astype(int), m.predict_proba(Xv), K, clip)
        table.append({"alpha": a, "attacker_val_log_loss": ll, "n_cells_fit": m.n_cells})
        if best is None or ll < best[0]:
            best = (ll, a, m)
    return {"model": best[2], "selected": {"alpha": best[1]}, "attacker_val_log_loss": best[0],
            "selection_table": table}


def slate(Xf, sf, Xv, sv, K, E, auth, syn, what, finite=False):
    """Base slate (benchmark rule: NL = GBT vs MLP on attacker_val log loss; refit at attacker seeds 0,1,2) plus, for
    finite surfaces, the defense-aware cell-conditional candidate reported separately (NLDA = base NL vs CC)."""
    r = fit_slate(Xf, sf, Xv, sv, K, E, auth, syn, what)
    nlf = r["NL"]["selected_family"]
    sN, rN = retrain_seeds(nlf, r[nlf]["selected"], r[nlf]["model"], Xf, sf, E, auth, syn, what)
    out = {"slate": r, "nl_family": nlf, "nl_seeds": sN, "nl_seed_record": rN, "L": r["L"]["model"],
           "val_ll": {"NL": r[nlf]["attacker_val_log_loss"], "L": r["L"]["attacker_val_log_loss"]}}
    if finite:
        auth.check(f"cell-conditional ({what})", syn)
        cc = fit_cell_conditional(Xf, sf, Xv, sv, K, E["attackers"]["log_loss_clip"])
        out["CC"] = cc
        out["val_ll"]["CC"] = cc["attacker_val_log_loss"]
        out["NLDA"] = "CC" if cc["attacker_val_log_loss"] < r[nlf]["attacker_val_log_loss"] else "NL"
    return out


def _proba(m, X, K):
    return m.predict_proba(X) if isinstance(m, CellConditional) else full_proba(m, X, K)


# ------------------------------------------------------------------------------------------------- units on disk
def unit_dir(uid: str) -> Path:
    return RUN / "units" / uid


def unit_complete(uid: str) -> bool:
    d = unit_dir(uid)
    if not (d / "COMPLETE.json").exists():
        return False
    rec = json.loads((d / "COMPLETE.json").read_text())
    for f, h in rec["files"].items():
        p = d / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            return False
    return True


def write_unit(uid: str, preds: dict, val: dict, models: dict, record: dict) -> None:
    d = unit_dir(uid)
    if d.exists() and not unit_complete(uid):
        d.rename(d.with_name(d.name + f".partial-{int(time.time())}"))
    (d / "models").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "preds.npz", **preds)
    np.savez_compressed(d / "val_preds.npz", **val)
    for name, m in models.items():
        joblib.dump(m, d / "models" / f"{name}.joblib")
    (d / "record.json").write_text(json.dumps(record, indent=1, default=str))
    files = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*"))
             if p.is_file() and p.name != "COMPLETE.json"}
    (d / "COMPLETE.json").write_text(json.dumps({"id": uid, "files": files,
                                                 "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                indent=1))


def attack_unit(uid, X, W, s, K, E, auth, syn, *, finite=False, plus=None, contract=None, cpu_ledger=None):
    """Fit/select on attacker_fit/val, then score assessment. `plus` = dict(rep_unit=..., out_unit=...) adds the
    benchmark's ignore-rep / ignore-outputs candidates (validation log losses of the component units)."""
    if unit_complete(uid):
        return json.loads((unit_dir(uid) / "record.json").read_text())
    t0 = time.process_time()
    f, v, a = W["idx"]["attacker_fit"], W["idx"]["attacker_val"], W["idx"]["assessment"]
    assert not np.isnan(X[np.concatenate([f, v, a])]).any(), f"{uid}: NaN in released surface"
    r = slate(X[f], s[f], X[v], s[v], K, E, auth, syn, uid, finite=finite)
    preds = {"assess_row_id": W["row_id"][a], "assess_unit": W["unit"][a], "y_s": s[a]}
    val = {"val_row_id": W["row_id"][v], "val_y_s": s[v]}
    models = {}
    for k_, m in r["nl_seeds"].items():
        preds[f"P__NL__as{k_}"] = _proba(m, X[a], K)
        models[f"NL_as{k_}"] = m
    preds["P__L"] = _proba(r["L"], X[a], K)
    val["VAL__NL__as0"] = _proba(r["nl_seeds"][0], X[v], K)
    val["VAL__L"] = _proba(r["L"], X[v], K)
    models["L"] = r["L"]
    rec = {"id": uid, "nl_family": r["nl_family"], "nl_selected": r["slate"][r["nl_family"]]["selected"],
           "val_log_loss": r["val_ll"], "selection_tables": {fam: r["slate"][fam]["selection_table"] for fam in
                                                             ("L", "GBT", "MLP")},
           "nl_seed_record": r["nl_seed_record"], "contract": contract, "n_columns": int(X.shape[1]),
           "selection_role": "attacker_val", "fit_role": "attacker_fit"}
    if finite:
        cc = r["CC"]
        preds["P__CC"] = cc["model"].predict_proba(X[a])
        val["VAL__CC"] = cc["model"].predict_proba(X[v])
        models["CC"] = cc["model"]
        rec["CC"] = {"selected": cc["selected"], "selection_table": cc["selection_table"]}
        rec["NLDA_selected"] = r["NLDA"]
    if plus is not None:
        rv = json.loads((unit_dir(plus["rep_unit"]) / "record.json").read_text())["val_log_loss"]
        ov = json.loads((unit_dir(plus["out_unit"]) / "record.json").read_text())["val_log_loss"]
        cand = {"GBT/MLP": r["val_ll"]["NL"], "ignore_rep": ov["NL"], "ignore_out": rv["NL"]}
        sel = select_plus(cand)
        rec["plus_selection"] = {"candidates_attacker_val_log_loss": cand, "selected": sel,
                                 "alias_source": {"ignore_rep": plus["out_unit"], "ignore_out": plus["rep_unit"]}.get(sel)}
    rec["cpu_s"] = time.process_time() - t0
    rec["n_model_fits"] = sum(len(r["slate"][fam]["selection_table"]) for fam in ("L", "GBT", "MLP")) + \
        sum(1 for x in r["nl_seed_record"]["seeds"].values() if x.get("status") == "REFIT") + \
        (len(CC_ALPHAS) if finite else 0)
    write_unit(uid, preds, val, models, rec)
    if cpu_ledger is not None:
        cpu_ledger(uid, rec["cpu_s"], rec["n_model_fits"])
    return rec


def u2_unit(uid, X, W, t, K_t, E, auth, syn, cpu_ledger=None):
    """Common refitted task utility: the benchmark U2 LR probe on the released features (attacker_fit / val)."""
    if unit_complete(uid):
        return json.loads((unit_dir(uid) / "record.json").read_text())
    t0 = time.process_time()
    u2 = E["utility"]["U2"]
    cfg = {"C": u2["C"], "max_iter": u2["max_iter"], "solver": E["attackers"]["L"]["solver"],
           "scaler": E["attackers"]["L"]["scaler"]}
    f, v, a = W["idx"]["attacker_fit"], W["idx"]["attacker_val"], W["idx"]["assessment"]
    r = fit_family("L", X[f], t[f], X[v], t[v], K_t, cfg, E["attackers"]["log_loss_clip"], auth=auth,
                   synthetic=syn, what=uid)
    Pa, Pv = full_proba(r["model"], X[a], K_t), full_proba(r["model"], X[v], K_t)
    rec = {"id": uid, "selected": r["selected"], "selection_table": r["selection_table"],
           "val_accuracy": float((Pv.argmax(1) == t[v]).mean()), "val_log_loss": r["attacker_val_log_loss"],
           "cpu_s": time.process_time() - t0, "n_model_fits": len(r["selection_table"])}
    write_unit(uid, {"assess_row_id": W["row_id"][a], "assess_unit": W["unit"][a], "y_t": t[a], "U2_P": Pa},
               {"val_row_id": W["row_id"][v], "val_y_t": t[v], "VAL__U2_P": Pv}, {"U2": r["model"]}, rec)
    if cpu_ledger is not None:
        cpu_ledger(uid, rec["cpu_s"], rec["n_model_fits"])
    return rec


def head_unit(uid, X, W, t, K_t, E, auth, syn, cpu_ledger=None):
    """View-3 release head on protected features (fit: defense_fit minus holdout; select: head holdout). Saves the
    head's outputs for every row and its task accuracy / log loss on assessment (release utility)."""
    d = unit_dir(uid)
    if unit_complete(uid):
        with np.load(d / "preds.npz") as z:
            return {"outputs": z["head_outputs_all"], "record": json.loads((d / "record.json").read_text())}
    t0 = time.process_time()
    r = fit_head(X, t, W, K_t, E, auth, syn, uid)
    Oh = head_outputs(r["model"], X, K_t)
    a = W["idx"]["assessment"]
    Pa = np.exp(Oh[a])
    rec = {"id": uid, "selected": r["selected"], "selection_table": r["selection_table"],
           "fit_role": "defense_fit minus oar-head-v1 holdout", "select_role": "defense_fit holdout (oar-head-v1)",
           "inputs_at_runtime": "protected features only",
           "assessment_accuracy": float((Pa.argmax(1) == t[a]).mean()),
           "assessment_log_loss": float(-np.mean(np.log(np.clip(Pa[np.arange(len(a)), t[a]], CLIP, 1)))),
           "cpu_s": time.process_time() - t0, "n_model_fits": len(r["selection_table"])}
    write_unit(uid, {"head_outputs_all": Oh, "row_id": W["row_id"], "assess_row_id": W["row_id"][a],
                     "assess_unit": W["unit"][a], "y_t": t[a]}, {}, {"head": r["model"]}, rec)
    if cpu_ledger is not None:
        cpu_ledger(uid, rec["cpu_s"], rec["n_model_fits"])
    return {"outputs": Oh, "record": rec}


def reference_unit(uid, W, s, t, K_s, K_t, E, auth, syn):
    """Label-only reference P(s | true task label) (diagnostic) and the constant prior, fitted on attacker_fit."""
    if unit_complete(uid):
        return json.loads((unit_dir(uid) / "record.json").read_text())
    f, a = W["idx"]["attacker_fit"], W["idx"]["assessment"]
    lo = E["references"]["LO"]
    T = fit_label_only(s[f], t[f], K_s, K_t, lo["laplace_alpha"], auth=auth, synthetic=syn)
    prior = np.bincount(s[f], minlength=K_s) / len(f)
    preds = {"assess_row_id": W["row_id"][a], "assess_unit": W["unit"][a], "y_s": s[a],
             "P__LO": T[t[a]], "P__const": np.tile(prior, (len(a), 1))}
    write_unit(uid, preds, {}, {}, {"id": uid, "LO_alpha": lo["laplace_alpha"], "prior": prior.tolist()})
    return {"id": uid}


def null_and_planted(uid, X, W, s, K, E, auth, syn, finite=False):
    """Real-data controls on attacker_fit / attacker_val only: (a) shuffled-label null (s permuted within each role,
    seed 20261012): validation macro AUC of the base NL should be near 0.5; (b) planted-leak positive control:
    append a column equal to s (one-hot) with 20 % of rows replaced by a random class: validation AUC must be high."""
    from sklearn.metrics import roc_auc_score
    f, v = W["idx"]["attacker_fit"], W["idx"]["attacker_val"]
    rng = np.random.default_rng(20261012)
    out = {"id": uid}
    sf, sv = rng.permutation(s[f]), rng.permutation(s[v])
    r = slate(X[f], sf, X[v], sv, K, E, auth, syn, uid + "/null", finite=finite)
    P = _proba(r["nl_seeds"][0], X[v], K)
    out["null_val_macro_auc"] = float(np.mean([roc_auc_score(sv == c, P[:, c]) for c in np.unique(sv)]))
    noisy = s.copy()
    flip = rng.random(len(s)) < 0.2
    noisy[flip] = rng.integers(0, K, flip.sum())
    Xp = np.hstack([X, onehot(noisy, K)])
    r2 = slate(Xp[f], s[f], Xp[v], s[v], K, E, auth, syn, uid + "/planted", finite=finite)
    P2 = _proba(r2["nl_seeds"][0], Xp[v], K)
    out["planted_val_macro_auc"] = float(np.mean([roc_auc_score(s[v] == c, P2[:, c]) for c in np.unique(s[v])]))
    out["null_flag_above_0.55"] = out["null_val_macro_auc"] > 0.55
    out["planted_detected_above_0.75"] = out["planted_val_macro_auc"] > 0.75
    return out


def effective():
    return Tracked(BENCH_EFFECTIVE)


def auth_real():
    return FitAuthorization(execute_scientific_fits=True)

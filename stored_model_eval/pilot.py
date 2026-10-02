"""CELL-A pilot unit runner: plan / dry-run / execute (checkpointed per unit, resumable).

    python -m stored_model_eval pilot --plan                       # expected vs actual unit IDs; refuses on mismatch
    python -m stored_model_eval pilot                              # dry-run (default): admission, contracts, support
    python -m stored_model_eval pilot --execute-scientific-fits [--resume] [--units a,b]

Root: the worktree is derived from this file's location (or an explicit --worktree that must resolve to the same
tree); execution refuses unless `git rev-parse --abbrev-ref HEAD` there is research/combined-stored-model-pilot-v1
(an explicit test override exists). `--execute-scientific-fits` additionally verifies PILOT_LOCK_v2.json
(lock.py) and refuses on any mismatch.

Per unit (FROZEN_DESIGN "Saved predictions"), written to <run_dir>/units/<unit_id>/ (outside git):
  supported.json   frozen support (written BEFORE any fit)
  preds.npz        aligned assessment predictions (see SAVED_KEYS)
  fit_records.json effective configuration, selection tables (attacker_val only), access records, timing
  models/          joblib-pickled fitted estimators / closed-form parameters (replay needs no retraining)
  COMPLETE.json    sha256 of every file above; `--resume` skips a unit only if all hashes verify
Phases: native_phase (historical in-sample statistics; no selection) -> fit_phase (receives attacker_fit and
attacker_val rows ONLY; all selection happens here) -> predict_phase (assessment rows; no selection).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .access import AccessRecord, ReleaseContract
from .admission import load_admitted, sha256_file
from .effective import (BRANCH, EFFECTIVE_PROTOCOL, OUTPUTS_REUSE_SOURCE, PURPOSES, REGISTERED_UNITS, Tracked,
                        consumption_report, effective_hash, unit_info)
from .guards import FitAuthorization, require_outside_git
from .metrics import to_jsonable
from .recipes import (GaussianClassLRT, fit_family, fit_g2, fit_label_only, fit_rho1, fixed_ridge_fit,
                      full_proba, historical_native_r2, linear_predict, select_nl)
from .support import freeze_support
from .surfaces import build_surface

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = Path.home() / "PCRL_eval_cache_private" / "pilot_adult_s0" / "run_v1"
SURF_KEY = {"rep": "rep", "outputs": "outputs", "rep+outputs": "repPLUSoutputs"}
SAVED_KEYS = ("assess_row_id", "assess_unit", "y_s", "y_task", "G1_pred", "G1_prior", "G2_pred", "G2_prior",
              "LO_P", "U1_logits", "U2_P")


class PilotRefused(SystemExit):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------------
# worktree / branch / unit selection
# --------------------------------------------------------------------------------------------------


def git_branch(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise PilotRefused(f"REFUSED: {root} is not a git work tree ({r.stderr.strip()})")
    return r.stdout.strip()


def resolve_worktree(explicit: str | None = None, allow_other_branch_for_tests: bool = False) -> dict:
    root = PKG_ROOT if not explicit else Path(explicit).expanduser().resolve()
    if root != PKG_ROOT:
        raise PilotRefused(f"REFUSED: --worktree {root} is not the tree this package runs from ({PKG_ROOT})")
    if not (root / "stored_model_eval" / "pilot.py").exists():
        raise PilotRefused(f"REFUSED: {root} does not contain stored_model_eval/")
    branch = git_branch(root)
    ok = branch == BRANCH
    if not ok and not allow_other_branch_for_tests:
        raise PilotRefused(f"REFUSED: worktree {root} is on branch {branch!r}, expected {BRANCH!r}")
    return {"root": str(root), "branch": branch, "branch_ok": ok, "override": (not ok)}


def select_units(manifest_dir: Path, subset: list[str] | None = None) -> dict:
    """Select the registered units (explicit list, by ID) from the manifests present in manifest_dir."""
    manifest_dir = Path(manifest_dir)
    available = sorted(p.name[len("manifest_"):-len(".json")] for p in manifest_dir.glob("manifest_*.json"))
    expected = list(REGISTERED_UNITS)
    if subset:
        bad = [u for u in subset if u not in expected]
        if bad:
            raise PilotRefused(f"REFUSED: --units contains unregistered IDs {bad}")
        expected = [u for u in expected if u in subset]
    selected = [u for u in expected if u in available]
    missing = [u for u in expected if u not in available]
    return {"manifest_dir": str(manifest_dir), "n_available": len(available), "expected": expected,
            "selected": selected, "missing": missing, "match": selected == expected,
            "n_unselected": len([a for a in available if a not in expected]),
            "manifests": {u: str(manifest_dir / f"manifest_{u}.json") for u in selected}}


def plan(manifest_dir: Path, subset=None) -> dict:
    sel = select_units(manifest_dir, subset)
    out = {"mode": "plan", "registered_count": len(REGISTERED_UNITS), **sel,
           "comparison": [{"expected": u, "actual": (u if u in sel["selected"] else None)} for u in sel["expected"]],
           "fits_performed": 0}
    if not sel["match"]:
        out["status"] = "REFUSED"
        raise PilotRefused("REFUSED: registered units missing from manifest dir: " + ", ".join(sel["missing"])
                           + "\n" + json.dumps(out, indent=1))
    out["status"] = "OK"
    return out


# --------------------------------------------------------------------------------------------------
# unit completeness
# --------------------------------------------------------------------------------------------------


def _files_under(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and p.name != "COMPLETE.json")


def write_complete(d: Path, unit_id: str) -> dict:
    rec = {"unit_id": unit_id, "completed_at": _now(),
           "files": {str(p.relative_to(d)): sha256_file(p) for p in _files_under(d)}}
    (d / "COMPLETE.json").write_text(json.dumps(rec, indent=1))
    return rec


def verify_complete(d: Path) -> tuple[bool, list[str]]:
    c = d / "COMPLETE.json"
    if not c.exists():
        return False, ["no COMPLETE.json"]
    rec = json.loads(c.read_text())
    errs = []
    for rel, h in rec["files"].items():
        p = d / rel
        if not p.exists():
            errs.append(f"missing {rel}")
        elif sha256_file(p) != h:
            errs.append(f"hash changed {rel}")
    for need in ("preds.npz", "fit_records.json", "supported.json"):
        if need not in rec["files"]:
            errs.append(f"{need} not recorded")
    return not errs, errs


# --------------------------------------------------------------------------------------------------
# access records
# --------------------------------------------------------------------------------------------------


def _access(tag, attacker, surface, contract: ReleaseContract, fit_inputs, requires=(), status="EXECUTED",
            note=None, valid=True, invalid_reason=None, queries=1):
    return AccessRecord(tag=tag, attacker=attacker, surface=surface, fit_inputs=list(fit_inputs),
                        eval_inputs=["release(assessment row), one draw" if contract.noise != "none"
                                     else "release(assessment row), deterministic"],
                        requires=list(requires), contract=contract.to_json(), valid=valid,
                        invalid_reason=invalid_reason, queries=queries,
                        effective_queries=contract.effective_queries(queries), status=status, note=note).to_json()


def staged_repeated_query(contract: ReleaseContract) -> dict:
    return _access("A3(N)", "repeated_release", "rep", contract,
                   ["release(attacker_fit)", "S(attacker_fit)", "N releases of target"],
                   requires=["query interface"], status="STAGED_NOT_RUN", valid=contract.issues_fresh_noise(),
                   invalid_reason=None if contract.issues_fresh_noise() else
                   f"contract noise={contract.noise} persistent={contract.persistent} release_count="
                   f"{contract.release_count}: N collapses to 1; a fresh-query contract needs a separate amendment",
                   note="staged, not executed")


# --------------------------------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------------------------------


def native_phase(rep_all, s_all, ei, K_s, eff, *, auth, synthetic, untreated: bool) -> dict:
    """Historical in-sample statistics (no selection). N0 on all rows (untreated only), N1 on assessment rows."""
    auth.check("native in-sample statistics", synthetic)
    lc = eff["linear_closed_form"]
    out = {}
    n0 = lc["N0"]
    if untreated and n0["units"] == "untreated":
        out["N0"] = {v: historical_native_r2(rep_all, s_all, n0["lambda"], v) for v in n0["variants"]}
        out["N0"].update(rows=n0["rows"], n_rows=int(len(s_all)), label=n0["label"], tau=n0["tau"],
                         category_variant=n0["category_variant"])
        val = out["N0"][n0["category_variant"]]["clamped"]
        out["N0"]["category"] = "C1" if val > n0["tau"] else "historical check passes"
        cats = {v: (out["N0"][v]["clamped"] > n0["tau"]) for v in n0["variants"]}
        out["N0"]["variants_agree_on_category"] = len(set(cats.values())) == 1
    else:
        out["N0"] = {"status": n0["noise_units"], "label": n0["label"],
                     "reason": "no historical check on these rows (durable-guarantees noise approvals were on their "
                               "own rows); compliance is not inferred from N1"}
    n1 = lc["N1"]
    out["N1"] = {v: historical_native_r2(rep_all[ei], s_all[ei], n1["lambda"], v) for v in n1["variants"]}
    out["N1"].update(rows=n1["rows"], label=n1["label"])
    return out


def fit_phase(view: dict, ctx: dict) -> tuple[dict, list, dict]:
    """All fitting and selection. `view` holds ONLY attacker_fit ("fit") and attacker_val ("val") rows."""
    if set(view) != {"fit", "val"}:
        raise ValueError(f"fit_phase accepts only fit/val views, got {sorted(view)}")
    eff, auth, syn = ctx["eff"], ctx["auth"], ctx["synthetic"]
    contract, K_s, K_t, info = ctx["contract"], ctx["K_s"], ctx["K_t"], ctx["info"]
    F, V = view["fit"], view["val"]
    att = eff["attackers"]
    clip = att["log_loss_clip"]
    models, recs, closed = {}, [], {}
    lc = eff["linear_closed_form"]
    estimable = ctx["support"]["status"] != "NE"
    a1_inputs = ["release(attacker_fit)", "S(attacker_fit)"]

    # closed-form linear quantities on the released representation
    if estimable:
        auth.check("G1 fixed-ridge held-out", syn)
        g1 = lc["G1"]
        models["G1"] = fixed_ridge_fit(F["rep"], F["s"], K_s, g1["lambda"], np.dtype(g1["dtype"]).type)
        models["G1"]["fit_role"], models["G1"]["clamp"] = g1["fit_role"], g1["clamp"]
        closed["G1"] = {"ss_tot_center": g1["ss_tot_center"], "score_role": g1["score_role"],
                        "access": _access("A1", "G1_fixed_ridge", "rep", contract, a1_inputs)}
        g2 = fit_g2(F["rep"], F["s"], V["rep"], V["s"], K_s, lc["G2"], auth=auth, synthetic=syn)
        models["G2"] = g2["model"]
        closed["G2"] = {"selected": g2["selected"], "selection_table": g2["selection_table"],
                        "select_on": lc["G2"]["select_on"], "select_metric": lc["G2"]["select_metric"],
                        "access": _access("A1", "G2_relative_ridge", "rep", contract, a1_inputs)}
        rho = fit_rho1(F["rep"], F["s"], K_s, lc["rho1_heldout"]["eps_rel"], auth=auth, synthetic=syn)
        models["RHO1"] = rho
        closed["RHO1"] = {k: v for k, v in rho.items() if k not in ("a", "b")}
        closed["pure_metric"] = {"predictor": lc["pure_metric"]["predictor"]}

    # attacker slate per surface
    reused = ctx["reused_surfaces"]
    for surf in ctx["surfaces"]:
        if surf in reused or not estimable:
            continue
        Xf, srec = build_surface(surf, F["rep"], F["outputs"], ctx["outputs_kind"])
        Xv, _ = build_surface(surf, V["rep"], V["outputs"], ctx["outputs_kind"])
        res = {}
        for fam in ("L", "GBT", "MLP"):
            r = fit_family(fam, Xf, F["s"], Xv, V["s"], K_s, att[fam], clip, auth=auth, synthetic=syn,
                           what=f"{info['unit_id']}/{surf}")
            res[fam] = r
            models[(surf, fam)] = r["model"]
            recs.append({"surface": surf, "recipe": fam, "status": "EXECUTED", "surface_record": srec.to_json(),
                         "selected": r["selected"], "selection_table": r["selection_table"],
                         "attacker_val_log_loss": r["attacker_val_log_loss"], "cpu_s": r["cpu_s"],
                         "select_metric": att[fam]["select_metric"], "selection_role": att["selection_role"],
                         "access": _access("A1", fam, surf, contract, a1_inputs,
                                           note=None if surf == "rep" or contract.noise == "none"
                                           else "outputs component is the clean model output (deterministic)")})
        nl = select_nl(res["GBT"], res["MLP"])
        if list(att["NL"]["members"]) != ["GBT", "MLP"] or att["NL"]["select_metric"] != "attacker_val_log_loss":
            raise ValueError("NL members/selection must be GBT, MLP on attacker_val log-loss")
        models[(surf, "NL")] = models[(surf, nl["selected_family"])]
        recs.append({"surface": surf, "recipe": "NL", "status": "EXECUTED", "nl_selection": nl,
                     "selected": {"family": nl["selected_family"], **res[nl["selected_family"]]["selected"]},
                     "access": _access("A1", "NL", surf, contract, a1_inputs)})

    # defense-informed (A2) and white-box (A4) noise LRTs
    if info["kind"] == "noise" and estimable:
        for name, X_src in (("LRT_A2", F["rep"]), ("LRT_A4", F["clean_rep"])):
            c = att[name]
            if c["applies_to"] != "noise" or c["surface"] != "rep" or c["class_prior"] != "attacker_fit":
                raise ValueError(f"{name} effective settings not supported")
            exp_trace = "released_class_covariance" if name == "LRT_A2" else "clean_class_covariance"
            if c["floor_trace"] != exp_trace:
                raise ValueError(f"{name}.floor_trace must be {exp_trace}")
            m = GaussianClassLRT(contract.sigma, name[-2:], c["eig_floor_rel"], c["cov_ddof"], K_s)
            m.fit(X_src, F["s"], auth=auth, synthetic=syn)
            models[("rep", name)] = m
            if name == "LRT_A2":
                acc = _access("A2", name, "rep", contract, a1_inputs + ["sigma (public mechanism parameter)"],
                              requires=["mechanism code", "known sigma"], note="no clean vectors; selection: none")
            else:
                acc = _access("A4", name, "rep", contract,
                              ["CLEAN representations(attacker_fit)", "S(attacker_fit)", "known Sigma"],
                              requires=["population clean vectors", "known Sigma"],
                              note="white-box population stress test; never used to refute a guarantee")
            recs.append({"surface": "rep", "recipe": name, "status": "EXECUTED", "selected": None,
                         "lrt": m.describe(), "access": acc})

    # label-only reference
    lo = eff["references"]["LO"]
    if lo["fit_role"] != "attacker_fit":
        raise ValueError("LO must be fitted on attacker_fit")
    models["LO"] = fit_label_only(F["s"], F["t"], K_s, K_t, lo["laplace_alpha"], auth=auth, synthetic=syn)
    recs.append({"surface": "label_only", "recipe": "LO", "status": "EXECUTED", "selected": None,
                 "access": {"tag": "reference", "fit_inputs": ["y_task(attacker_fit)", "S(attacker_fit)"],
                            "eval_inputs": ["true y_task(assessment)"], "contract": None}})

    # U2 utility probe on the released representation
    u2 = eff["utility"]["U2"]
    if u2["fit_role"] != "attacker_fit" or u2["select_role"] != "attacker_val":
        raise ValueError("U2 roles must be attacker_fit / attacker_val")
    u2cfg = {"C": u2["C"], "max_iter": u2["max_iter"], "solver": att["L"]["solver"], "scaler": att["L"]["scaler"]}
    r = fit_family("L", F["rep"], F["t"], V["rep"], V["t"], K_t, u2cfg, clip, auth=auth, synthetic=syn,
                   what=f"{info['unit_id']}/U2")
    models["U2"] = r["model"]
    recs.append({"surface": "rep", "recipe": "U2", "status": "EXECUTED", "selected": r["selected"],
                 "selection_table": r["selection_table"], "attacker_val_log_loss": r["attacker_val_log_loss"],
                 "cpu_s": r["cpu_s"], "target": "y_task",
                 "access": {"tag": "utility_probe", "fit_inputs": ["release(attacker_fit)", "y_task(attacker_fit)"],
                            "contract": contract.to_json()}})
    return models, recs, closed


def predict_phase(models: dict, A: dict, ctx: dict) -> dict:
    """Score assessment rows with already selected models. No selection happens here."""
    K_s = ctx["K_s"]
    out = {}
    if "G1" in models:
        out["G1_pred"] = linear_predict(models["G1"], A["rep"]).astype(np.float64)
        out["G1_prior"] = np.asarray(models["G1"]["muY"], dtype=np.float64)
        out["G2_pred"] = linear_predict(models["G2"], A["rep"])
        out["G2_prior"] = np.asarray(models["G2"]["muY"], dtype=np.float64)
        if models["RHO1"]["status"] == "OK":
            out["RHO_u"] = np.asarray(A["rep"], dtype=np.float64) @ models["RHO1"]["b"]
            out["RHO_v"] = models["RHO1"]["a"][np.asarray(A["s"]).astype(int)]
    for key, m in models.items():
        if not isinstance(key, tuple):
            continue
        surf, rec = key
        if rec.startswith("LRT_"):
            P = m.predict_proba(A["rep"])
        else:
            X, _ = build_surface(surf, A["rep"], A["outputs"], ctx["outputs_kind"])
            P = full_proba(m, X, K_s)
        out[f"P__{SURF_KEY[surf]}__{rec}"] = P
    out["LO_P"] = models["LO"][np.asarray(A["t"]).astype(int)]
    out["U2_P"] = full_proba(models["U2"], A["rep"], ctx["K_t"])
    return out


# --------------------------------------------------------------------------------------------------
# one unit
# --------------------------------------------------------------------------------------------------


def _view(arrays: dict, idx: np.ndarray) -> dict:
    return {k: (None if v is None else np.asarray(v)[idx]) for k, v in arrays.items()}


def run_unit(unit_id: str, manifest_path: Path, units_root: Path, auth: FitAuthorization, *,
             eff: dict | None = None, log=print) -> dict:
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    effp = EFFECTIVE_PROTOCOL if eff is None else eff
    E = Tracked(effp)
    info = unit_info(unit_id)
    if unit_id not in E["units"]["registered"]:
        raise PilotRefused(f"REFUSED: {unit_id} is not a registered unit")
    units_root = require_outside_git(Path(units_root), "pilot unit outputs")
    out = units_root / unit_id
    if out.exists():
        raise PilotRefused(f"REFUSED: {out} exists (use --resume; partial dirs are moved aside, never overwritten)")
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text())
    if man.get("unit_id", unit_id) != unit_id:
        raise PilotRefused(f"REFUSED: manifest {manifest_path} declares unit_id {man.get('unit_id')}")
    arrays, adm = load_admitted(manifest_path)  # verifies every file hash and ID alignment
    synthetic = bool(arrays["_synthetic"])
    contract = ReleaseContract.from_manifest(man.get("release"))
    for pname in E["purposes"]:  # the effective purpose table must equal the constant unit_info() parses with
        P_ = E["purposes"][pname]
        if (P_["index"], P_["task"], P_["n_task_classes"]) != tuple(PURPOSES[pname][k] for k in
                                                                     ("index", "task", "n_task_classes")):
            raise ValueError(f"purpose table mismatch for {pname}")
    if info["kind"] == "noise":
        if contract.noise != "persistent_token" or contract.sigma != info["sigma"] or \
                contract.seed != info["release_seed"]:
            raise PilotRefused(f"REFUSED: {unit_id} release block {man.get('release')} disagrees with the unit id")
    elif contract.noise != "none":
        raise PilotRefused(f"REFUSED: untreated unit {unit_id} carries a release block")
    R = E["roles"]
    roles = arrays["roles"].astype(str)
    fi, vi, ei = (np.flatnonzero(roles == R[k]) for k in ("attacker_fit", "attacker_val", "assessment"))
    s = np.asarray(arrays["labels"]).astype(np.int64)
    if "task_labels" not in arrays:
        raise PilotRefused(f"REFUSED: {unit_id} manifest has no task_labels (build run_v1 inputs first)")
    t = np.asarray(arrays["task_labels"]).astype(np.int64)
    K_s = int(s.max()) + 1
    K_t = int(E["purposes"][info["purpose"]]["n_task_classes"])
    outputs = np.asarray(arrays["outputs"])
    if t.max() >= K_t or outputs.shape[1] != K_t:
        raise PilotRefused(f"REFUSED: task labels/outputs disagree with {K_t} task classes")
    rule = E["support"]
    support = freeze_support(s, roles, rule, K_s, "sensitive")
    task_support = freeze_support(t, roles, rule, K_t, "task")
    out.mkdir(parents=True)
    (out / "models").mkdir()
    (out / "supported.json").write_text(json.dumps({"unit_id": unit_id, "frozen_at": _now(),
                                                    "frozen_before": "any fit or inference",
                                                    "sensitive": support, "task": task_support}, indent=1))
    surfaces = list(E["surfaces"][info["kind"]])
    reused_surfaces = {}
    if info["kind"] == "noise":
        if E["surfaces"]["noise_outputs"] != "reused":
            raise ValueError("noise outputs surface must be reused")
        reused_surfaces["outputs"] = E["units"]["outputs_reuse_source"]
        surfaces = surfaces + ["outputs"]
    ctx = {"eff": E, "auth": auth, "synthetic": synthetic, "contract": contract, "K_s": K_s, "K_t": K_t,
           "info": info, "support": support, "surfaces": surfaces, "reused_surfaces": reused_surfaces,
           "outputs_kind": E["surfaces"]["outputs_kind"]}
    base = {"row_id": arrays["row_ids"], "rep": arrays["representations"], "outputs": outputs, "s": s, "t": t,
            "clean_rep": arrays.get("clean_representations")}
    if info["kind"] == "noise" and base["clean_rep"] is None:
        raise PilotRefused(f"REFUSED: noise unit {unit_id} manifest lacks clean_representations (A4 LRT)")
    native = native_phase(np.asarray(base["rep"]), s, ei, K_s, E, auth=auth, synthetic=synthetic,
                          untreated=info["kind"] == "untreated")
    models, recs, closed = fit_phase({"fit": _view(base, fi), "val": _view(base, vi)}, ctx)
    A = _view(base, ei)
    preds = predict_phase(models, A, ctx)
    # in-memory values before saving, so tests/replay can confirm the saved -> loaded -> inferred path reproduces them
    from .recipes import r2_against_prior
    for nm in ("G1", "G2"):
        if f"{nm}_pred" in preds:
            closed[nm]["r2_in_memory_before_save"] = r2_against_prior(s[ei], preds[f"{nm}_pred"],
                                                                      preds[f"{nm}_prior"], K_s)
    preds.update(assess_row_id=np.asarray(arrays["row_ids"])[ei].astype(np.int64),
                 assess_unit=np.asarray(arrays["units"])[ei].astype(np.int64), y_s=s[ei], y_task=t[ei],
                 s_prior_fit=np.bincount(s[fi], minlength=K_s)[:K_s] / len(fi),
                 t_prior_fit=np.bincount(t[fi], minlength=K_t)[:K_t] / len(fi))

    # reuse of the clean outputs surface (noise units)
    reuse = {}
    if "outputs" in reused_surfaces:
        src = units_root / reused_surfaces["outputs"]
        ok, errs = verify_complete(src)
        if not ok:
            raise PilotRefused(f"REFUSED: outputs reuse source {src.name} incomplete: {errs}")
        srec = json.loads((src / "fit_records.json").read_text())
        mine = _outputs_spec(man)
        if srec["manifest"]["outputs_spec"] != mine:
            raise PilotRefused(f"REFUSED: outputs object differs from reuse source ({mine} vs "
                               f"{srec['manifest']['outputs_spec']})")
        with np.load(src / "preds.npz", allow_pickle=False) as z:
            if not np.array_equal(z["assess_row_id"], preds["assess_row_id"]):
                raise PilotRefused("REFUSED: reuse source assessment IDs differ")
            for k in z.files:
                if k.startswith("P__outputs__"):
                    preds[k] = z[k]
        for r in srec["recipes"]:
            if r["surface"] == "outputs":
                rr = json.loads(json.dumps(r))
                rr["status"] = "REUSED"
                rr["reused_from"] = src.name
                rr["access"]["status"] = "REUSED"
                rr["access"]["source_contract"] = rr["access"]["contract"]
                rr["access"]["contract"] = contract.to_json()  # the arm's release contract (rep component)
                rr["access"]["note"] = ("outputs component is the clean model output (deterministic), identical "
                                        f"across noise arms; reused from {src.name}; not new evidence")
                recs.append(rr)
        reuse = {"outputs": {"source_unit": src.name, "source_preds_sha256": sha256_file(src / "preds.npz"),
                             "outputs_spec": mine, "not_new_evidence": True}}

    # U1 frozen head
    u1cfg = E["utility"]["U1"]
    u1 = {"mode": u1cfg[info["kind"]]}
    ck = man.get("files", {}).get("checkpoint")
    if ck:
        ckp = adm["files"]["checkpoint"]["path"]
        from .forward import frozen_head_logits
        if u1cfg["checkpoint_load"] != "weights_only":
            raise ValueError("U1 checkpoint load must be weights_only")
        clean_src = base["clean_rep"] if info["kind"] == "noise" else base["rep"]
        head_clean, prov = frozen_head_logits(ckp, ck["sha256"], info["purpose"], np.asarray(clean_src)[ei])
        diff = float(np.abs(head_clean.astype(np.float64) - outputs[ei].astype(np.float64)).max())
        u1["consistency_head_on_clean_vs_stored_logits_max_abs"] = diff
        u1["consistency_ok"] = diff <= u1cfg["consistency_atol"]
        u1["checkpoint"] = prov
        if not u1["consistency_ok"]:
            raise PilotRefused(f"REFUSED: frozen head on clean rep disagrees with stored logits (max |d| = {diff})")
        if info["kind"] == "noise":
            preds["U1_logits"], _ = frozen_head_logits(ckp, ck["sha256"], info["purpose"], np.asarray(base["rep"])[ei])
        else:
            preds["U1_logits"] = outputs[ei]
    else:
        u1["consistency_ok"] = None
        if info["kind"] == "noise":
            u1["status"] = "NE"
            u1["reason"] = "no checkpoint declared in manifest"
        else:
            preds["U1_logits"] = outputs[ei]
    preds["U1_logits"] = np.asarray(preds.get("U1_logits", np.zeros((len(ei), 0))), dtype=np.float64)

    np.savez_compressed(out / "preds.npz", **{k: np.asarray(v) for k, v in preds.items()})
    import joblib
    for key, m in models.items():
        name = "__".join(key) if isinstance(key, tuple) else key
        joblib.dump(m, out / "models" / f"{name.replace('+', 'PLUS')}.joblib")
    recs.append({"surface": "rep", "recipe": "A3_repeated_query", "status": E["attackers"]["A3_repeated_query"]["status"],
                 "access": staged_repeated_query(contract)})
    import sklearn
    fr = {"unit_id": unit_id, "info": info, "written_at": _now(),
          "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path),
                       "outputs_spec": _outputs_spec(man), "derived_from": man.get("derived_from")},
          "admission": {"status": adm["status"], "n_rows": adm["n_rows"], "warnings": adm["warnings"],
                        "files": {k: {"sha256": v["sha256"], "match": v["match"]} for k, v in adm["files"].items()}},
          "contract": contract.to_json(), "synthetic": synthetic,
          "effective_protocol_sha256": effective_hash(effp), "effective_protocol": effp,
          "roles": {"attacker_fit": int(len(fi)), "attacker_val": int(len(vi)), "assessment": int(len(ei))},
          "K_s": K_s, "K_t": K_t, "support_status": support["status"],
          "native": native, "closed_form": closed, "recipes": recs, "reuse": reuse, "U1": u1,
          "saved_keys": sorted(preds),
          "n_model_fits": _count_fits(recs, closed),
          "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                          "OMP_NUM_THREADS_required": E["threads"]["OMP_NUM_THREADS"],
                          "python": platform.python_version(),
                          "numpy": np.__version__, "sklearn": sklearn.__version__, "machine": platform.machine()},
          "effective_keys_consumed_in_unit": sorted(E.log),
          "timing": {"wall_s": time.perf_counter() - t_wall, "cpu_s": time.process_time() - t_cpu}}
    (out / "fit_records.json").write_text(json.dumps(to_jsonable(fr), indent=1, default=str))
    comp = write_complete(out, unit_id)
    log(f"[pilot] {unit_id}: done in {fr['timing']['wall_s']:.1f}s wall / {fr['timing']['cpu_s']:.1f}s cpu")
    return {"unit_id": unit_id, "dir": str(out), "timing": fr["timing"], "n_files": len(comp["files"])}


def _count_fits(recs: list, closed: dict) -> dict:
    """Models actually fitted in THIS unit (reused / staged records excluded)."""
    by = {}
    for r in recs:
        if r.get("status") != "EXECUTED" or r["recipe"] == "NL":
            continue
        n = len(r.get("selection_table") or []) or 1
        by[f"{r['surface']}|{r['recipe']}"] = n
    for nm in ("G1", "RHO1"):
        if nm in closed:
            by[nm] = 1
    if "G2" in closed:
        by["G2"] = len(closed["G2"]["selection_table"])
    return {"total": int(sum(by.values())), "by_recipe": by}


def _outputs_spec(man: dict) -> dict:
    a = man["arrays"]["outputs"]
    return {"file_sha256": man["files"][a["file"]]["sha256"], "key": a["key"]}


# --------------------------------------------------------------------------------------------------
# dry run and execute loop
# --------------------------------------------------------------------------------------------------


def dry_run(manifest_dir: Path, subset=None) -> dict:
    """Admission, contracts and frozen support per selected unit. Reads labels; performs NO fit."""
    sel = plan(manifest_dir, subset)
    E = EFFECTIVE_PROTOCOL
    rows = []
    for u in sel["selected"]:
        mp = Path(sel["manifests"][u])
        man = json.loads(mp.read_text())
        arrays, adm = load_admitted(mp)
        info = unit_info(u)
        roles = arrays["roles"].astype(str)
        s = np.asarray(arrays["labels"]).astype(int)
        sup = freeze_support(s, roles, E["support"], int(s.max()) + 1)
        surfaces = E["surfaces"][info["kind"]]
        mlp = E["attackers"]["MLP"]
        n_slate = len(E["attackers"]["L"]["C"]) + len(E["attackers"]["GBT"]["configs"]) + \
            len(mlp["hidden_layer_sizes"]) * len(mlp["alpha"]) * len(mlp["learning_rate_init"])
        rows.append({"unit_id": u, "admission": adm["status"], "contract": ReleaseContract.from_manifest(
            man.get("release")).to_json(), "has_task_labels": "task_labels" in arrays,
            "has_checkpoint": "checkpoint" in man.get("files", {}),
            "support": {"supported_classes": sup["supported_classes"], "status": sup["status"],
                        "unsupported": sup["unsupported_classes"]},
            "surfaces": surfaces, "reused_surfaces": ["outputs"] if info["kind"] == "noise" else [],
            "planned_model_fits": n_slate * len(surfaces) + len(E["linear_closed_form"]["G2"]["rho_grid"])
            + len(E["utility"]["U2"]["C"]) + (2 if info["kind"] == "noise" else 0)})
    return {"mode": "dry-run", "fits_performed": 0, "units": rows, "plan": {k: sel[k] for k in
                                                                            ("n_available", "match", "missing")}}


def execute(manifest_dir: Path, units_root: Path, auth: FitAuthorization, *, subset=None, resume=False,
            log=print) -> dict:
    sel = plan(manifest_dir, subset)
    units_root = require_outside_git(Path(units_root), "pilot unit outputs")
    units_root.mkdir(parents=True, exist_ok=True)
    done, skipped, moved = [], [], []
    for u in sel["selected"]:
        d = units_root / u
        if d.exists():
            ok, errs = verify_complete(d)
            if ok and resume:
                skipped.append(u)
                log(f"[pilot] {u}: complete and hash-verified; skipped (--resume)")
                continue
            if ok:
                raise PilotRefused(f"REFUSED: {u} already complete; pass --resume to skip it")
            if (d / "COMPLETE.json").exists():
                raise PilotRefused(f"REFUSED: {u} COMPLETE.json present but hashes fail ({errs}); investigate")
            if not resume:
                raise PilotRefused(f"REFUSED: partial outputs in {d}; pass --resume to move them aside and rerun")
            aside = d.with_name(f"{u}.partial-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            shutil.move(str(d), str(aside))
            moved.append(str(aside))
        r = run_unit(u, Path(sel["manifests"][u]), units_root, auth, log=log)
        r["n_model_fits"] = json.loads((units_root / u / "fit_records.json").read_text())["n_model_fits"]["total"]
        done.append(r)
        with open(units_root.parent / "RUN_LOG.jsonl", "a") as f:
            f.write(json.dumps({"at": _now(), **r}) + "\n")
    return {"mode": "execute", "completed": done, "skipped_complete": skipped, "moved_partial": moved,
            "fits_performed": int(sum(r["n_model_fits"] for r in done)),
            "consumption": consumption_report(_consumed_union(units_root, sel["selected"]))}


def _consumed_union(units_root: Path, units) -> set:
    s = set()
    for u in units:
        p = Path(units_root) / u / "fit_records.json"
        if p.exists():
            s |= set(json.loads(p.read_text()).get("effective_keys_consumed_in_unit", []))
    return s


__all__ = ["plan", "dry_run", "execute", "run_unit", "select_units", "resolve_worktree", "verify_complete",
           "fit_phase", "predict_phase", "native_phase", "PilotRefused", "SAVED_KEYS", "PURPOSES",
           "OUTPUTS_REUSE_SOURCE"]

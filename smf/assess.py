"""Single outer assessment of frozen releases on NEW_DEVELOPMENT_ASSESSMENT (audit/baseline owner; adapted from rgj.assess).

REFUSES unless results/pcrl_strength_matched_feedback_v1/EVALUATION_LOCK.json is committed, byte-identical to its last
commit in the working tree, and that commit is contained in origin/research/pcrl-strength-matched-feedback-v1
(`git branch -r --contains <commit>`; the local remote-tracking ref, so fetch/push first). `open_assessment(lock)`
performs the check; only after it passes does `load_unsealed()` call smf.data.load(unseal=True), and `outer_unit`
refuses a sealed D and re-verifies the lock (same commit, same sha256, still pushed) before every unit. If the lock lists
`locked_code_files` {relpath: sha256}, every listed file of the scoring chain (CHAIN) must match it; if the lock lists
`seeds[k].unit_file_sha256[unit]`, the unit's COMPLETE.json file hashes must equal it; if it lists
`assessment_role.row_id_sha256`, the unsealed assessment rows must hash to it. This is the only module of the study that
indexes DEVELOPMENT_ASSESSMENT (= NEW_DEVELOPMENT_ASSESSMENT) rows or reads their labels.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m smf.assess --evaluation-lock results/pcrl_strength_matched_feedback_v1/EVALUATION_LOCK.json
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m smf.assess --evaluation-lock <lock> --controls-only   (inner roles only)

Lock schema read: lock["seeds"][str(k)]["score"] = {label: spec}, spec = {"kind": "neural", "unit": "<unit name>"} or
{"kind": "fare", "units": [n_purpose0, n_purpose1]}. A seed entry with "valid_reference": false and an empty score map
is skipped. Unit written: outer__s{k}__{safe(label)} (safe: "*" -> "star", "/" and spaces -> "_"; the label is kept).

Per release (attackers fitted on AUDIT_FIT, selected on INNER_SELECTION, refitted at attacker seeds 0, 1, 2, scored on
DEVELOPMENT_ASSESSMENT; smf.audit.final_audit):
  primary    views v1 = [r1, c1], v2 = [r2, c2], pair = [v1, v2]; FINAL slate (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP;
             finite releases add CC x3); dual selection: AUC-selected (primary endpoint) and CE-selected (log loss);
             coalition bank = own pair attackers + ignore_recipient_2 (v1) + ignore_recipient_1 (v2); every table kept.
  secondary  output-only views p1, p2, ppair (probabilities) and h1, h2, hpair (one-hot hard decisions, finite) with the
             SECONDARY slate (inner + DA_MLP, + CC on finite views), same dual selection and banks.
  race       stress audit on the primary views, SECONDARY slate, supported classes only (>= 30 rows in each of
             AUDIT_FIT, INNER_SELECTION, DEVELOPMENT_ASSESSMENT), codes remapped to 0..K-1 in ascending original code;
             fewer than 2 supported classes -> NOT_ESTIMABLE (never zero leakage).
  utility    deployed heads on DEVELOPMENT_ASSESSMENT: accuracy, balanced accuracy over supported task classes,
             per-class recall, minority recall (smallest supported class), log loss, Brier, 10-bin ECE (max-probability
             confidence), constant accuracy (DEFENSE_FIT majority class) and useful gain.
  probe      common refitted probe per release r_i -> task_i: StandardScaler + LR, C in {0.01, 0.1, 1, 10, 100} fitted on
             AUDIT_FIT, C by INNER_SELECTION log loss; accuracy on DEVELOPMENT_ASSESSMENT (reported separately).
  linear     OLS R^2 of SEX on r1, r2 and [r1, r2]: fitted and scored on DEFENSE_FIT rows (fitted-row diagnostic);
             held-out (fit AUDIT_FIT, score DEVELOPMENT_ASSESSMENT); fixed-penalty ridge (alpha = 1) on the same two
             splits; shuffled-label nulls (SEX permuted within DEFENSE_FIT / within AUDIT_FIT, seed NULL_SEED, drawn
             before fitting). LEACE releases carry their native fitting-row check from the unit record; FARE releases
             carry the certificate status (NOT_AVAILABLE_UNDER_NEW_ROLES, smf.baselines.CERT_STATUS).

preds.npz keys (private; row order = DEVELOPMENT_ASSESSMENT rows in D order, n = number of assessment rows)
  assess_row_id (n,) int64      row ids              assess_unit (n,) exact-record group id (bootstrap unit)
  sex (n,)  race (n,)  y_income (n,)  y_occ (n,)    labels (race = original codes)
  hard1 (n,) hard2 (n,)  p1 (n,2)  p2 (n,6)          deployed-head decisions and probabilities
  P_auc_{w}, P_ce_{w}  (3, n, 2) float64             SEX probabilities of the AUC- / CE-selected attacker at attacker
                                                     seeds 0, 1, 2 for w in v1, v2, pair (primary), p1, p2, ppair,
                                                     h1, h2, hpair (secondary); score = column 1 = P(SEX = 1)
  race_pos (m,) int64                                positions (into the n assessment rows) of supported-race rows
  race_codes (K,) int64                              original race code of each Prace column
  race_y (m,) int64                                  remapped race label (0..K-1) of those rows
  Prace_auc_{w}, Prace_ce_{w} (3, m, K)              race probabilities for w in v1, v2, pair (absent if NOT_ESTIMABLE)
  probe_hard1 (n,), probe_hard2 (n,)                 common LR probe decisions
The attacker, source view and bank candidate behind every P_* key are in record.json under
primary / secondary_prob / secondary_hard / race -> "scored" -> w -> "auc" | "ce".

Controls-only mode (no lock verification needed; sealed D; inner roles only): for each seed and label, smf.audit.controls
(shuffled-label null scored on the held-out half B of INNER_SELECTION after selection on half A; planted one-hot and
1e-6 leaks of the permuted label) and smf.audit.rotated_plant (rotated 1e-6 clue in r1 through save/load/transform) are
written as unit controls__s{k} (all_ok and every failure preserved).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rgj import finalize as FN
from smf import audit as AU

WT = Path(__file__).resolve().parents[1]
STUDY_BRANCH = "research/pcrl-strength-matched-feedback-v1"
LOCK_NAME = "EVALUATION_LOCK.json"
LOCK_REL = "results/pcrl_strength_matched_feedback_v1/EVALUATION_LOCK.json"
UNITS = Path.home() / "PCRL_eval_cache_private" / "smf_v1" / "run" / "units"
ASSESS = "DEVELOPMENT_ASSESSMENT"
FIT, SEL = "AUDIT_FIT", "INNER_SELECTION"
SUPPORT_MIN = 30
NULL_SEED = 20261016
KS = [2, 6]
TASKS = ("income", "occupation_group")
LR_C = (0.01, 0.1, 1.0, 10.0, 100.0)
CHAIN = ("smf/assess.py", "smf/audit.py", "smf/baselines.py", "smf/data.py", "jcv/audit.py", "jcv/finalize.py",
         "rgj/finalize.py", "rgj/data.py")
DEFAULT_CONTROL_LABELS = ("J-F", "L-F", "U", "F")
_OPENED = None


# ------------------------------------------------------------------ the lock gate
def _git(repo, *args, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lock_is_pushed(path, repo=WT, branch=STUDY_BRANCH, rel_required=LOCK_REL):
    """{"ok": bool, "reason" | "commit", "sha256", ...}: the registered lock file, committed, unmodified, on origin."""
    p = Path(path).resolve()
    repo = Path(repo).resolve()
    if p.name != LOCK_NAME:
        return {"ok": False, "reason": f"lock file must be named {LOCK_NAME}"}
    if not p.is_file():
        return {"ok": False, "reason": "lock file missing"}
    try:
        rel = str(p.relative_to(repo))
    except ValueError:
        return {"ok": False, "reason": "lock file is outside the study repository"}
    if rel_required is not None and rel != rel_required:
        return {"ok": False, "reason": f"lock file is not the registered {rel_required}"}
    commit = _git(repo, "log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not commit:
        return {"ok": False, "reason": "lock file is not committed"}
    blob = _git(repo, "show", f"{commit}:{rel}", text=False)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        return {"ok": False, "reason": "working-tree lock differs from its last commit"}
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        return {"ok": False, "reason": "lock file has uncommitted changes"}
    remotes = _git(repo, "branch", "-r", "--contains", commit).stdout.split()
    if f"origin/{branch}" not in remotes:
        return {"ok": False, "reason": f"lock commit {commit[:12]} is not on origin/{branch}"}
    return {"ok": True, "commit": commit, "sha256": _sha(p), "path": str(p), "repo": str(repo), "branch": branch,
            "rel": rel}


def verify_code(lock, repo=WT):
    """Scoring-chain files listed in lock["locked_code_files"] must be unchanged. Returns a receipt; raises on mismatch."""
    listed = lock.get("locked_code_files") or {}
    rec = {}
    for f in CHAIN:
        if f not in listed:
            rec[f] = "not listed in the lock"
            continue
        have = _sha(Path(repo) / f) if (Path(repo) / f).exists() else None
        if have != listed[f]:
            raise SystemExit(f"REFUSED: {f} differs from the locked code hash")
        rec[f] = "matches lock"
    return rec


def open_assessment(lock_path, repo=WT, branch=STUDY_BRANCH, rel_required=LOCK_REL, check_code=True):
    """Verify the pushed lock (and the locked scoring code) and open the assessment for this process."""
    global _OPENED
    v = lock_is_pushed(lock_path, repo, branch, rel_required)
    if not v["ok"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} is not committed and pushed ({v['reason']})")
    lock = json.loads(Path(lock_path).read_text())
    v["code"] = verify_code(lock, repo) if check_code else "not checked (test)"
    v["rel_required"] = rel_required
    v["lock"] = lock
    _OPENED = v
    return lock


def close_assessment():
    global _OPENED
    _OPENED = None


def _require_open():
    if _OPENED is None:
        raise SystemExit(f"REFUSED: the development assessment is sealed; no verified pushed {LOCK_NAME}")
    v = lock_is_pushed(_OPENED["path"], _OPENED["repo"], _OPENED["branch"], _OPENED["rel_required"])
    if not v["ok"] or v["sha256"] != _OPENED["sha256"] or v["commit"] != _OPENED["commit"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} changed or is no longer pushed since it was opened")
    return _OPENED


def load_unsealed():
    """smf.data.load(unseal=True), only after open_assessment() has verified the pushed lock. If the lock records
    assessment_role.row_id_sha256, the unsealed role's row ids must hash to it (smf.data.manifest construction)."""
    opened = _require_open()
    from smf import data as DA
    D = DA.load(unseal=True)
    if D.get("sealed", True):
        raise SystemExit("REFUSED: smf.data did not unseal the assessment labels")
    want = ((opened.get("lock") or {}).get("assessment_role") or {}).get("row_id_sha256")
    if want is not None and DA.manifest(D)["NEW_DEVELOPMENT_ASSESSMENT"]["row_id_sha256"] != want:
        raise SystemExit("REFUSED: the assessment rows differ from those recorded in the lock")
    return D


# ------------------------------------------------------------------ releases
def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace(" ", "_")


def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _unit_names(spec):
    kind = spec.get("kind") or ("fare" if "units" in spec else "neural")
    return kind, (spec["units"] if kind == "fare" else [spec["unit"]])


def release_for(spec, D, units_dir=None, locked_files=None):
    """(views, [r1, r2], finite, native meta, provenance, release dict) for a lock spec."""
    kind, names = _unit_names(spec)
    prov = {}
    for n in names:
        d = udir(n, units_dir)
        if not FN.unit_complete(d):
            raise SystemExit(f"REFUSED: release unit {n} is missing or not hash-complete")
        files = json.loads((d / "COMPLETE.json").read_text())["files"]
        if locked_files is not None and n in locked_files and locked_files[n] != files:
            raise SystemExit(f"REFUSED: unit {n} files differ from the hashes recorded in the lock")
        prov[n] = {"complete_sha256": _sha(d / "COMPLETE.json"),
                   "matches_lock_unit_hashes": (locked_files[n] == files) if locked_files and n in locked_files else None}
    if kind == "fare":
        from smf.baselines import fare_pair_views, fare_release_dict
        for n in names:
            assert np.array_equal(np.load(udir(n, units_dir) / "release.npz")["row_id"], D["row_id"])
        V = fare_pair_views(names[0], names[1], units_dir)
        recs = [json.loads((udir(n, units_dir) / "record.json").read_text()) for n in names]
        native = {"fare_certificate": [r.get("certificate") for r in recs], "configs": [r.get("config") for r in recs]}
        return V, V["r"], True, native, prov, fare_release_dict(names[0], names[1], units_dir)
    z = np.load(udir(names[0], units_dir) / "release.npz")
    assert np.array_equal(z["row_id"], D["row_id"]), "release rows are not aligned with D"
    V = FN.views_from_release(z)
    rec = json.loads((udir(names[0], units_dir) / "record.json").read_text())
    leace = (rec.get("finalize") or {}).get("leace")
    native = {"leace_native_fit_rows": leace} if leace else {}
    return V, [z["r1"], z["r2"]], False, native, prov, {k: z[k] for k in z.files}


# ------------------------------------------------------------------ utility, probe, linear diagnostics
def constants(D):
    tr = D["idx"]["DEFENSE_FIT"]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr]))) for i, t in enumerate(TASKS)}


def supported(y, D, K):
    return [c for c in range(K) if all((y[D["idx"][r]] == c).sum() >= SUPPORT_MIN for r in (FIT, SEL, ASSESS))]


def utility(P, hard, y, D, K, const):
    a = D["idx"][ASSESS]
    yy, Pa, ha = y[a], np.asarray(P)[a], np.asarray(hard)[a]
    AU.check_labels(yy, np.arange(len(yy)))
    sup = supported(y, D, K)
    rec = {int(c): float((ha[yy == c] == c).mean()) for c in range(K) if (yy == c).sum()}
    minority = min(sup, key=lambda c: (yy == c).sum()) if sup else None
    conf = Pa.max(1)
    ece, rel = 0.0, []
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            c_, acc_ = float(conf[m].mean()), float((ha[m] == yy[m]).mean())
            ece += m.mean() * abs(c_ - acc_)
            rel.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "confidence": c_, "accuracy": acc_})
    acc, cacc = float((ha == yy).mean()), float((yy == const).mean())
    return {"accuracy": acc, "balanced_accuracy_supported": float(np.mean([rec[c] for c in sup])) if sup else None,
            "per_class_recall": rec, "supported_classes": sup, "minority_class": minority,
            "minority_recall": rec.get(minority) if minority is not None else None, "log_loss": AU.logloss(yy, Pa),
            "brier": float(np.mean(np.sum((Pa - np.eye(K)[yy]) ** 2, 1))), "ece_10bin": float(ece),
            "reliability": rel, "const_class": int(const), "const_accuracy": cacc, "useful_gain": acc - cacc}


def probe(R, y, D, K):
    from jcv.audit import _lr
    f, v, a = D["idx"][FIT], D["idx"][SEL], D["idx"][ASSESS]
    best, table = None, []
    for C in LR_C:
        m = _lr(C, 0).fit(R[f], y[f])
        ll = AU.logloss(y[v], AU.proba(m, R[v], K))
        table.append({"C": C, "inner_log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    hard = AU.proba(best[2], R[a], K).argmax(1)
    return {"probe": f"LR_C{best[1]}", "table": table, "accuracy": float((hard == y[a]).mean())}, hard


def linear_diagnostics(Rs, D):
    from sklearn.linear_model import LinearRegression, Ridge
    tr, af, a = D["idx"]["DEFENSE_FIT"], D["idx"][FIT], D["idx"][ASSESS]
    S = D["sex"].astype(float)
    rng = np.random.default_rng(NULL_SEED)
    S_tr_null, S_af_null = rng.permutation(S[tr]), rng.permutation(S[af])     # frozen before fitting
    out = {"null_seed": NULL_SEED}
    for nm, Z in (("r1", Rs[0]), ("r2", Rs[1]), ("r1+r2", np.hstack(Rs))):
        Z = np.asarray(Z, dtype=np.float64)
        d = {}
        for est_name, est in (("ols", LinearRegression), ("ridge_alpha1", lambda: Ridge(alpha=1.0))):
            m = est().fit(Z[tr], S[tr])
            mh = est().fit(Z[af], S[af])
            d[f"{est_name}_fit_rows_r2"] = float(m.score(Z[tr], S[tr]))
            d[f"{est_name}_heldout_r2_fit_AUDIT_FIT_score_ASSESS"] = float(mh.score(Z[a], S[a]))
        d["ols_fit_rows_null_r2"] = float(LinearRegression().fit(Z[tr], S_tr_null).score(Z[tr], S_tr_null))
        d["ols_heldout_null_r2"] = float(LinearRegression().fit(Z[af], S_af_null).score(Z[a], S[a]))
        d["dim"] = int(Z.shape[1])
        out[nm] = d
    return out


# ------------------------------------------------------------------ one outer unit
def _audit(V, y, idx, slate_fn, finite, K, classes, ck, lk):
    return AU.final_audit(V, y, idx[FIT], idx[SEL], idx[ASSESS], slate_fn=slate_fn, finite=finite, K=K,
                          classes=classes, coalition_key=ck, local_keys=lk)


def outer_unit(D, k, label, spec, units_dir=None):
    """Score one frozen release on DEVELOPMENT_ASSESSMENT (refuses unless the pushed lock was opened and D unsealed)."""
    opened = _require_open()
    if D.get("sealed", True):
        raise SystemExit("REFUSED: D is sealed; use load_unsealed() after open_assessment()")
    name = f"outer__s{k}__{safe(label)}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    t0 = time.time()
    seed_entry = (opened.get("lock") or {}).get("seeds", {}).get(str(k), {})
    V, Rs, finite, native, prov, _ = release_for(spec, D, units_dir, seed_entry.get("unit_file_sha256"))
    S, race, idx = D["sex"], D["race"], D["idx"]
    a = idx[ASSESS]
    prim, Pp = _audit({w: V[w] for w in ("v1", "v2", "pair")}, S, idx, AU.final_slate, finite, 2, (0, 1),
                      "pair", ("v1", "v2"))
    o = V["out"]
    pv = {"p1": o["p1"], "p2": o["p2"], "ppair": np.hstack([o["p1"], o["p2"]])}
    hv = {"h1": np.eye(KS[0])[o["hard1"]], "h2": np.eye(KS[1])[o["hard2"]]}
    hv["hpair"] = np.hstack([hv["h1"], hv["h2"]])
    secP, Ps1 = _audit(pv, S, idx, AU.secondary_slate, finite, 2, (0, 1), "ppair", ("p1", "p2"))
    secH, Ps2 = _audit(hv, S, idx, AU.secondary_slate, True, 2, (0, 1), "hpair", ("h1", "h2"))
    Kr = int(race.max()) + 1
    rsup = supported(race, D, Kr)
    race_rec, Pr = {"supported_codes": rsup, "support_min": SUPPORT_MIN}, {}
    race_pos, race_y = np.array([], int), np.array([], int)
    if len(rsup) >= 2:
        remap = np.full(Kr, -1)
        remap[rsup] = np.arange(len(rsup))
        yr = remap[race]
        ridx = {r: idx[r][yr[idx[r]] >= 0] for r in (FIT, SEL, ASSESS)}
        ra, Pr = _audit({w: V[w] for w in ("v1", "v2", "pair")}, yr, ridx, AU.secondary_slate, finite, len(rsup),
                        list(range(len(rsup))), "pair", ("v1", "v2"))
        race_rec.update({"status": "ESTIMATED", "audit": ra, "n_assessment": int(len(ridx[ASSESS]))})
        race_pos = np.flatnonzero(yr[a] >= 0)
        race_y = yr[a][race_pos]
    else:
        race_rec["status"] = "NOT_ESTIMABLE (fewer than 2 supported classes)"
    cst = constants(D)
    util, prb, prb_hard = {}, {}, {}
    for i, t in enumerate(TASKS):
        y = D["y"][t]
        util[i] = utility(o[f"p{i + 1}"], o[f"hard{i + 1}"], y, D, KS[i], cst[i])
        prb[i], prb_hard[i] = probe(np.asarray(Rs[i], dtype=np.float64), y, D, KS[i])
    rec = {"unit": name, "seed": k, "label": label, "spec": spec, "release_provenance": prov,
           "evaluation_lock": {"commit": opened["commit"], "sha256": opened["sha256"], "code": opened.get("code")},
           "roles": {"attacker_fit": FIT, "attacker_selection": SEL, "scored": ASSESS},
           "primary": prim, "secondary_prob": secP, "secondary_hard": secH, "race": race_rec,
           "utility_deployed": util, "utility_common_probe": prb, "linear_diagnostics": linear_diagnostics(Rs, D),
           "native": native, "n_assessment": int(len(a)), "wall_s": None}
    preds = {"assess_row_id": D["row_id"][a], "assess_unit": D["unit"][a], "sex": S[a], "race": race[a],
             "y_income": D["y"]["income"][a], "y_occ": D["y"]["occupation_group"][a],
             "hard1": np.asarray(o["hard1"])[a], "hard2": np.asarray(o["hard2"])[a],
             "p1": np.asarray(o["p1"])[a], "p2": np.asarray(o["p2"])[a],
             "race_pos": race_pos.astype(np.int64), "race_codes": np.asarray(rsup, dtype=np.int64),
             "race_y": race_y.astype(np.int64), "probe_hard1": prb_hard[0], "probe_hard2": prb_hard[1]}
    for key, P in {**Pp, **Ps1, **Ps2}.items():
        preds[f"P_{key}"] = P
    for key, P in Pr.items():
        preds[f"Prace_{key}"] = P
    rec["preds_keys"] = sorted(preds)
    rec["wall_s"] = time.time() - t0
    FN.save_unit(udir(name, units_dir), {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    return rec


# ------------------------------------------------------------------ controls (inner roles only; no lock needed)
def controls_for(D, specs, units_dir=None, slate="final", recipients=(1,)):
    out = {}
    for label, spec in specs.items():
        V, _, finite, _, prov, z = release_for(spec, D, units_dir)
        c = AU.controls({w: V[w] for w in ("v1", "v2", "pair")}, D, finite=finite, slate=slate)
        rot = AU.rotated_plant(z, D, slate=slate, recipients=recipients, finite=finite)
        out[label] = {"spec": spec, "provenance": prov, **c, "rotated_plant": rot,
                      "all_ok": bool(c["all_ok"] and rot["all_ok"])}
    out["all_ok"] = all(v["all_ok"] for v in out.values() if isinstance(v, dict))
    return out


def main(argv=None):
    import argparse

    from smf import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--controls-only", action="store_true", help="inner-role controls on the lock's releases (sealed)")
    ap.add_argument("--controls-labels", nargs="*", default=list(DEFAULT_CONTROL_LABELS))
    ap.add_argument("--controls-slate", default="final")
    ap.add_argument("--controls-recipients", nargs="+", type=int, default=[1])
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.controls_only:
        L = json.loads(Path(a.evaluation_lock).read_text())
        D = DA.load()                                   # sealed: inner roles only
        for k in a.seeds:
            sc = L["seeds"][str(k)].get("score", {})
            labels = [lb for lb in a.controls_labels if lb in sc]
            res = controls_for(D, {lb: sc[lb] for lb in labels}, a.units_dir, a.controls_slate,
                               tuple(a.controls_recipients))
            FN.save_unit(udir(f"controls__s{k}", a.units_dir), {}, res)
            print(k, "controls all_ok =", res["all_ok"], {lb: res[lb]["all_ok"] for lb in labels}, flush=True)
        return
    L = open_assessment(a.evaluation_lock)
    D = load_unsealed()
    for k in a.seeds:
        for label, spec in L["seeds"][str(k)].get("score", {}).items():
            if a.labels and label not in a.labels:
                continue
            r = outer_unit(D, k, label, spec, a.units_dir)
            print(r["unit"], {w: round(r["primary"]["scored"][w]["auc"]["auc_mean"], 4) for w in ("v1", "v2", "pair")},
                  flush=True)


if __name__ == "__main__":
    main()

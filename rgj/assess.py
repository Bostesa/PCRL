"""Single outer assessment of frozen releases on DEVELOPMENT_ASSESSMENT (audit/baseline owner).

REFUSES unless EVALUATION_LOCK.json is committed, unchanged in the working tree, and its commit is contained in
origin/research/pcrl-refreshed-guarded-joint-v1 (`git branch -r --contains <commit>`). `open_assessment(lock)` performs
that check; `outer_unit` refuses until it has passed and re-verifies it before every unit. This is the only module of
the study that indexes DEVELOPMENT_ASSESSMENT rows or reads their labels.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.assess --evaluation-lock results/pcrl_refreshed_guarded_joint_v1/EVALUATION_LOCK.json
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.assess --evaluation-lock <lock> --controls-only   (inner roles only)

Lock schema read: lock["seeds"][str(k)]["score"] = {label: spec}, spec = {"kind": "neural", "unit": "<ck__|tl__|lc__...>"}
or {"kind": "fare", "units": [n_purpose0, n_purpose1]}. Unit written: outer__s{k}__{safe(label)} (safe: "*" -> "star",
"/" and spaces -> "_"; the original label is kept in the record).

Per release (all attackers fitted on AUDIT_FIT, selected on INNER_SELECTION, refitted at attacker seeds 0, 1, 2,
scored on DEVELOPMENT_ASSESSMENT; rgj.audit.final_audit):
  primary    views v1 = [r1, c1], v2 = [r2, c2], pair = [v1, v2]; FINAL slate (finite releases add CC); dual selection:
             AUC-selected (primary endpoint) and CE-selected (log-loss reporting); coalition bank = own pair attackers +
             ignore_recipient_2 (v1) + ignore_recipient_1 (v2); every table archived.
  secondary  output-only views p1, p2, ppair (probabilities) and h1, h2, hpair (one-hot hard decisions, finite) with the
             SECONDARY slate (inner + DA, + CC on finite views), same dual selection and banks.
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
             before fitting). LEACE releases also carry their native fitting-row check from the unit record; FARE
             releases carry the certificate status (not recomputed).

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
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rgj import audit as AU
from rgj import finalize as FN

WT = Path(__file__).resolve().parents[1]
STUDY_BRANCH = "research/pcrl-refreshed-guarded-joint-v1"
LOCK_NAME = "EVALUATION_LOCK.json"
UNITS = Path.home() / "PCRL_eval_cache_private" / "rgj_v1" / "run" / "units"
ASSESS = "DEVELOPMENT_ASSESSMENT"
FIT, SEL = "AUDIT_FIT", "INNER_SELECTION"
SUPPORT_MIN = 30
NULL_SEED = 20261016
KS = [2, 6]
TASKS = ("income", "occupation_group")
LR_C = (0.01, 0.1, 1.0, 10.0, 100.0)
_OPENED = None


# ------------------------------------------------------------------ the lock gate
def _git(repo, *args, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def lock_is_pushed(path, repo=WT, branch=STUDY_BRANCH):
    """{"ok": bool, "reason"| "commit", "sha256", ...}: committed, unmodified, and on origin/<branch>."""
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
    return {"ok": True, "commit": commit, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "path": str(p),
            "repo": str(repo), "branch": branch}


def open_assessment(lock_path, repo=WT, branch=STUDY_BRANCH):
    """Verify the pushed lock and open the assessment for this process. Returns the parsed lock."""
    global _OPENED
    v = lock_is_pushed(lock_path, repo, branch)
    if not v["ok"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} is not committed and pushed ({v['reason']})")
    _OPENED = v
    return json.loads(Path(lock_path).read_text())


def close_assessment():
    global _OPENED
    _OPENED = None


def _require_open():
    if _OPENED is None:
        raise SystemExit(f"REFUSED: the development assessment is sealed; no verified pushed {LOCK_NAME}")
    v = lock_is_pushed(_OPENED["path"], _OPENED["repo"], _OPENED["branch"])
    if not v["ok"] or v["sha256"] != _OPENED["sha256"] or v["commit"] != _OPENED["commit"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} changed or is no longer pushed since it was opened")
    return v


# ------------------------------------------------------------------ releases
def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace(" ", "_")


def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def release_for(spec, D, units_dir=None):
    """(views, [r1, r2], finite, native meta, provenance) for a lock spec."""
    kind = spec.get("kind") or ("fare" if "units" in spec else "neural")
    names = spec["units"] if kind == "fare" else [spec["unit"]]
    for n in names:
        if not FN.unit_complete(udir(n, units_dir)):
            raise SystemExit(f"REFUSED: release unit {n} is missing or not hash-complete")
    prov = {n: {"complete_sha256": _sha_file(udir(n, units_dir) / "COMPLETE.json")} for n in names}
    if kind == "fare":
        from rgj.baselines import fare_pair_views
        for n in names:
            assert np.array_equal(np.load(udir(n, units_dir) / "release.npz")["row_id"], D["row_id"])
        V = fare_pair_views(names[0], names[1], units_dir)
        recs = [json.loads((udir(n, units_dir) / "record.json").read_text()) for n in names]
        native = {"fare_certificate": [r.get("certificate") for r in recs], "configs": [r.get("config") for r in recs]}
        return V, V["r"], True, native, prov
    z = np.load(udir(names[0], units_dir) / "release.npz")
    assert np.array_equal(z["row_id"], D["row_id"]), "release rows are not aligned with D"
    V = FN.views_from_release(z)
    rec = json.loads((udir(names[0], units_dir) / "record.json").read_text())
    leace = (rec.get("finalize") or {}).get("leace")
    native = {"leace_native_fit_rows": leace} if leace else {}
    return V, [z["r1"], z["r2"]], False, native, prov


# ------------------------------------------------------------------ utility, probe, linear diagnostics
def constants(D):
    tr = D["idx"]["DEFENSE_FIT"]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr]))) for i, t in enumerate(TASKS)}


def supported(y, D, K):
    return [c for c in range(K) if all((y[D["idx"][r]] == c).sum() >= SUPPORT_MIN for r in (FIT, SEL, ASSESS))]


def utility(P, hard, y, D, K, const):
    a = D["idx"][ASSESS]
    yy, Pa, ha = y[a], np.asarray(P)[a], np.asarray(hard)[a]
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
    """Score one frozen release on DEVELOPMENT_ASSESSMENT (refuses unless the pushed lock was opened)."""
    lock = _require_open()
    name = f"outer__s{k}__{safe(label)}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    t0 = time.time()
    V, Rs, finite, native, prov = release_for(spec, D, units_dir)
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
    # race stress audit (supported classes only)
    Kr = int(race.max()) + 1
    rsup = supported(race, D, Kr)
    race_rec, Pr, race_pos, race_y = {"supported_codes": rsup, "support_min": SUPPORT_MIN}, {}, np.array([], int), np.array([], int)
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
           "evaluation_lock": {"commit": lock["commit"], "sha256": lock["sha256"]},
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
def controls_for(D, specs, units_dir=None, slate="final"):
    out = {}
    for label, spec in specs.items():
        V, _, finite, _, prov = release_for(spec, D, units_dir)
        out[label] = {"spec": spec, "provenance": prov, **AU.controls({w: V[w] for w in ("v1", "v2", "pair")}, D,
                                                                      finite=finite, slate=slate)}
    out["all_ok"] = all(v["all_ok"] for v in out.values() if isinstance(v, dict))
    return out


def main(argv=None):
    import argparse

    from rgj import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--controls-only", action="store_true", help="inner-role controls on the lock's releases")
    ap.add_argument("--controls-labels", nargs="*", default=["J-G", "L-G", "U", "F"])
    ap.add_argument("--controls-slate", default="final")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.controls_only:
        L = json.loads(Path(a.evaluation_lock).read_text())
        D = DA.load()
        for k in a.seeds:
            sc = L["seeds"][str(k)]["score"]
            res = controls_for(D, {lb: sc[lb] for lb in a.controls_labels if lb in sc}, a.units_dir, a.controls_slate)
            FN.save_unit(udir(f"controls__s{k}", a.units_dir), {}, res)
            print(k, "controls all_ok =", res["all_ok"])
        return
    L = open_assessment(a.evaluation_lock)
    D = DA.load()
    for k in a.seeds:
        for label, spec in L["seeds"][str(k)]["score"].items():
            if a.labels and label not in a.labels:
                continue
            r = outer_unit(D, k, label, spec, a.units_dir)
            print(r["unit"], {w: round(r["primary"]["scored"][w]["auc"]["auc_mean"], 4) for w in ("v1", "v2", "pair")},
                  flush=True)


if __name__ == "__main__":
    main()

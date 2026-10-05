"""Single locked assessment of frozen releases on OSF_DEVELOPMENT_ASSESSMENT (audit/baseline owner; adapted from
smf.assess). This is the ONLY module of the study that indexes OSF_DEVELOPMENT_ASSESSMENT rows or reads their labels.

Gate. REFUSES unless results/pcrl_online_strength_frontier_v1/EVALUATION_LOCK.json is committed, byte-identical to its
last commit in the working tree, and that commit is contained in origin/research/pcrl-online-strength-frontier-v1
(`git branch -r --contains <commit>`; the local remote-tracking ref, so fetch/push first). `open_assessment(lock)`
also requires lock["locked_code_files"] = {relpath: sha256} to list EVERY file of the scoring chain (CHAIN) with a
matching hash now; other listed files are re-hashed and any later change is recorded (not refused). Only after this passes does `load_unsealed()` call
osf.data.load(unseal=True); the unsealed role must hash to lock["assessment_role"]["row_id_sha256"] (osf.data.manifest
construction) and have lock["assessment_role"]["rows"] rows when recorded. `outer_unit` refuses a sealed D and
re-verifies the lock (same commit and sha256, still pushed) before every unit; a release unit whose COMPLETE.json
file hashes differ from lock["seeds"][k]["unit_file_sha256"][unit] is refused.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.assess --evaluation-lock \
        results/pcrl_online_strength_frontier_v1/EVALUATION_LOCK.json [--seeds 0 1 2] [--labels ...] [--shard i/n]
    ... -m osf.assess --evaluation-lock <lock> --controls-only        (inner roles only; sealed D; no lock check)

Lock schema read (lead's osf/eval_lock.py contract): lock["seeds"][str(k)] = {"score": {label: spec}, "u_label": "U"
(default), "unit_file_sha256": {unit: COMPLETE.json files map}}; spec = {"kind": "release", "unit": "<unit>"} (any
single rgj-format release unit; "neural" accepted as a synonym; a unit whose record says "finite": true adds the
cell-conditional attackers) or {"kind": "fare", "units": [purpose-0 unit, purpose-1 unit]} (F / F0, as returned by
osf.baselines.reference_candidates(k)[label]["units"]). lock["assessment_role"] = osf.data.manifest(D)
["OSF_DEVELOPMENT_ASSESSMENT"] (rows, groups, row_id_sha256 are re-checked after unsealing);
lock["sex_prior_defense_fit_sha256"] (smf.eval_lock.prior_hash construction) is re-checked when present. Every locked release is scored with the SAME final slate:
nominees, controls, U, references and every locked grid point (the lead's list). Unit written: outer__s{k}__{safe}
(safe: "*" -> "star", "/", "|" and spaces -> "_"; the label is kept in record.json).

Per release (attackers FITTED on AUDIT_FIT, SELECTED on INNER_SELECTION, refitted at attacker seeds 0, 1, 2 and scored
on OSF_DEVELOPMENT_ASSESSMENT via smf.audit.final_audit; orientation fixed, never chosen on scored rows; the scored
coalition attacker is never clamped to a local result):
  primary     complete views v1 = [r1, c1], v2 = [r2, c2], pair = [v1, v2]; FINAL slate (LR x5, MLP x4, HGB x4, DA_LR,
              DA_MLP; finite releases add CC x3); AUC-selected (primary) and CE-selected (proper loss) attackers kept
              separately; coalition bank = own pair attackers + ignore_recipient_2 (v1) + ignore_recipient_1 (v2).
  features    features-only views r1, r2, rpair = [r1, r2] (SECONDARY slate, same dual selection and bank)
  outputs     outputs-only views c1, c2, cpair (centred logits); p1, p2, ppair (probabilities); h1, h2, hpair (one-hot
              hard decisions, finite) -- SECONDARY slate, own coalition banks
  race        supported-race diagnostic on the complete views, SECONDARY slate. Frozen support rule: a race code is
              supported iff it has >= 30 rows in each of AUDIT_FIT, INNER_SELECTION and OSF_DEVELOPMENT_ASSESSMENT;
              supported codes remapped to 0..K-1 ascending; rows of unsupported codes are EXCLUDED from the race audit
              (never pooled into another class); every unsupported / absent code is listed with its role counts and
              status NOT_SUPPORTED (never scored as zero leakage). Reported per scored seed: macro OvR AUC, per-class
              OvR AUC of every supported class, worst class (max OvR AUC) and contrast (max - min OvR AUC). Fewer
              than 2 supported classes -> NOT_ESTIMABLE.
  utility     deployed heads on the assessment: accuracy, constant accuracy (OSF_DEFENSE_FIT-prior majority class),
              gain over the constant, gain retention Acc - 0.8 Acc(U) - 0.2 Acc(const) (U = the seed's locked u_label
              release on the same rows), balanced accuracy over supported task classes, per-class recall, minority
              recall (smallest supported class), Brier score, log loss, 10-bin ECE.
  probe       common refitted probe r_i -> task_i (StandardScaler + LR, C by INNER_SELECTION log loss).
  linear      OLS / ridge R^2 of SEX on r1, r2, [r1, r2] (fit-row diagnostic and held-out AUDIT_FIT -> assessment)
              with shuffled-label nulls -- secondary attacker statistics.

preds.npz (private; row order = assessment rows in D order; n rows)
  assess_row_id, assess_unit (exact-record group = bootstrap unit), sex, race, y_income, y_occ
  const_class (2,) ; const1, const2 (n,)       the fitting-prior constant predictions
  hard1, hard2 (n,), p1 (n,2), p2 (n,6)          deployed-head decisions / probabilities of this release
  u_hard1, u_hard2, u_p1, u_p2                   the seed's U release on the same rows (gain retention)
  P_auc_{w}, P_ce_{w} (3, n, 2)                  P(SEX) of the AUC- / CE-selected attacker at attacker seeds 0, 1, 2
                                                 for w in v1 v2 pair (primary) | r1 r2 rpair (features-only) |
                                                 p1 p2 ppair, h1 h2 hpair (smf outputs-only keys) | c1 c2 cpair
                                                 (centred-logit outputs-only, added); score = column 1 = P(SEX = 1)
  race_pos (m,), race_codes (K,), race_y (m,), Prace_auc_{w}, Prace_ce_{w} (3, m, K) for w in v1 v2 pair
  probe_hard1, probe_hard2
The attacker, source view and bank candidate behind every P_* key are in record.json under primary / features_only /
outputs_only.{logits, probs, hard} / race.audit -> "scored" -> w -> "auc" | "ce".

Controls-only mode: osf.audit.controls + rotated_plant per locked label (inner roles only, sealed D), unit
controls__s{k}.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from osf import audit as OA
from rgj import finalize as FN
from smf import audit as AU

WT = Path(__file__).resolve().parents[1]
STUDY_BRANCH = "research/pcrl-online-strength-frontier-v1"
LOCK_NAME = "EVALUATION_LOCK.json"
LOCK_REL = "results/pcrl_online_strength_frontier_v1/EVALUATION_LOCK.json"
UNITS = Path.home() / "PCRL_eval_cache_private" / "osf_v1" / "run" / "units"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
FIT, SEL = "AUDIT_FIT", "INNER_SELECTION"
SUPPORT_MIN = 30
NULL_SEED = 20261016
KS = [2, 6]
TASKS = ("income", "occupation_group")
LR_C = (0.01, 0.1, 1.0, 10.0, 100.0)
CHAIN = ("osf/assess.py", "osf/audit.py", "osf/baselines.py", "osf/data.py", "smf/audit.py", "smf/data.py",
         "jcv/audit.py", "jcv/finalize.py", "rgj/finalize.py", "rgj/data.py")
SECONDARY_FAMILIES = ("features", "logits", "probs", "hard")
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


def verify_code(lock, repo=WT, chain=CHAIN):
    """Every CHAIN file (the scoring chain imported by this module) must be listed in lock["locked_code_files"] and
    match it now, else REFUSED. Other listed files are checked and their status recorded in the receipt (a later edit
    of e.g. a report module does not block scoring; it is disclosed)."""
    listed = lock.get("locked_code_files") or {}
    missing = [f for f in chain if f not in listed]
    if missing:
        raise SystemExit(f"REFUSED: the lock does not list the scoring-chain code hashes of {missing}")
    rec = {}
    for f, want in sorted(listed.items()):
        have = _sha(Path(repo) / f) if (Path(repo) / f).exists() else None
        if f in chain and have != want:
            raise SystemExit(f"REFUSED: {f} differs from the locked code hash")
        rec[f] = "matches lock" if have == want else ("missing now" if have is None else "CHANGED after lock (not in "
                                                                                           "the scoring chain)")
    return rec


def open_assessment(lock_path, repo=WT, branch=STUDY_BRANCH, rel_required=LOCK_REL, check_code=True, chain=CHAIN):
    """Verify the pushed lock and the locked code, then open the assessment for this process."""
    global _OPENED
    v = lock_is_pushed(lock_path, repo, branch, rel_required)
    if not v["ok"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} is not committed and pushed ({v['reason']})")
    lock = json.loads(Path(lock_path).read_text())
    v["code"] = verify_code(lock, repo, chain) if check_code else "not checked (test)"
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
    """osf.data.load(unseal=True), only after open_assessment() verified the pushed lock."""
    opened = _require_open()
    from osf import data as OD
    D = OD.load(unseal=True)
    if D.get("sealed", True):
        raise SystemExit("REFUSED: osf.data did not unseal the assessment labels")
    ar = (opened.get("lock") or {}).get("assessment_role") or {}
    m = OD.manifest(D)[ASSESS]
    if ar.get("row_id_sha256") is not None and m["row_id_sha256"] != ar["row_id_sha256"]:
        raise SystemExit("REFUSED: the assessment rows differ from those recorded in the lock")
    for key in ("rows", "groups"):
        if ar.get(key) is not None and m[key] != ar[key]:
            raise SystemExit(f"REFUSED: the assessment {key} count differs from the lock")
    want = (opened.get("lock") or {}).get("sex_prior_defense_fit_sha256")
    if want is not None and prior_hash(D) != want:
        raise SystemExit("REFUSED: the OSF_DEFENSE_FIT SEX prior differs from the lock")
    return D


def prior_hash(D):
    """smf.eval_lock.prior_hash construction: sha256 of the int64 bincount of OSF_DEFENSE_FIT SEX."""
    c = np.bincount(D["sex"][D["idx"]["DEFENSE_FIT"]], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


# ------------------------------------------------------------------ releases
def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace("|", "_").replace(" ", "_")


def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _unit_names(spec):
    kind = spec.get("kind") or ("fare" if "units" in spec else "release")
    if kind not in ("release", "neural", "fare"):
        raise SystemExit(f"REFUSED: unknown release spec kind {kind!r}")
    if kind == "fare":
        if len(spec["units"]) != 2:
            raise SystemExit("REFUSED: a fare spec lists exactly two purpose units")
        return kind, list(spec["units"])
    return "release", [spec["unit"]]


def release_for(spec, D, units_dir=None, locked_files=None):
    """(release dict {row_id, r1, c1, p1, hard1, r2, ...}, finite, native meta, provenance) for a lock spec."""
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
    recs = [json.loads((udir(n, units_dir) / "record.json").read_text()) for n in names]
    if kind == "fare":
        a, b = (np.load(udir(n, units_dir) / "release.npz") for n in names)
        z = {"row_id": a["row_id"], "r1": a["r"], "c1": a["c"], "p1": a["p"], "hard1": a["hard"],
             "r2": b["r"], "c2": b["c"], "p2": b["p"], "hard2": b["hard"]}
        assert np.array_equal(a["row_id"], b["row_id"])
        finite = True
        native = {"fare_certificate": [r.get("certificate") for r in recs], "configs": [r.get("config") for r in recs]}
    else:
        zz = np.load(udir(names[0], units_dir) / "release.npz")
        z = {k: zz[k] for k in zz.files}
        finite = bool(spec.get("finite", recs[0].get("finite", False)))
        leace = (recs[0].get("finalize") or {}).get("leace")
        native = {"leace_native_fit_rows": leace} if leace else {}
        if recs[0].get("certificate"):
            native["fare_certificate"] = recs[0]["certificate"]
    if not np.array_equal(z["row_id"], D["row_id"]):
        raise SystemExit("REFUSED: release rows are not aligned with osf D")
    return z, finite, native, prov


# ------------------------------------------------------------------ utility, support, probe, linear diagnostics
def constants(D):
    tr = D["idx"]["DEFENSE_FIT"]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr], minlength=KS[i]))) for i, t in enumerate(TASKS)}


def support_table(y, D, K):
    """Frozen support rule: per code, row counts in AUDIT_FIT, INNER_SELECTION, assessment and the status."""
    out = {}
    for c in range(K):
        n = {r: int((np.asarray(y)[D["idx"][r]] == c).sum()) for r in (FIT, SEL, ASSESS)}
        out[int(c)] = {"counts": n, "status": "SUPPORTED" if min(n.values()) >= SUPPORT_MIN else
                       ("ABSENT_FROM_ASSESSMENT" if n[ASSESS] == 0 else "NOT_SUPPORTED")}
    return out


def supported(y, D, K):
    return [c for c, v in support_table(y, D, K).items() if v["status"] == "SUPPORTED"]


def utility(P, hard, y, D, K, const, u_hard=None):
    a = D["idx"][ASSESS]
    yy, Pa, ha = np.asarray(y)[a], np.asarray(P)[a], np.asarray(hard)[a]
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
    out = {"accuracy": acc, "const_class": int(const), "const_accuracy": cacc, "gain_over_const": acc - cacc,
           "balanced_accuracy_supported": float(np.mean([rec[c] for c in sup])) if sup else None,
           "per_class_recall": rec, "supported_classes": sup, "minority_class": minority,
           "minority_recall": rec.get(minority) if minority is not None else None, "log_loss": AU.logloss(yy, Pa),
           "brier": float(np.mean(np.sum((Pa - np.eye(K)[yy]) ** 2, 1))), "ece_10bin": float(ece), "reliability": rel}
    if u_hard is not None:
        accU = float((np.asarray(u_hard)[a] == yy).mean())
        out.update({"u_accuracy": accU, "acc_minus_u": acc - accU,
                    "gain_retention": acc - 0.8 * accU - 0.2 * cacc})
    return out


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
    out = {"null_seed": NULL_SEED, "kind": "secondary attacker statistic (linear R^2 of SEX)"}
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


def race_diagnostic(V, D, finite=False):
    race = np.asarray(D["race"])
    Kr = int(max(race[D["idx"][r]].max() for r in (FIT, SEL, ASSESS))) + 1
    table = support_table(race, D, Kr)
    rsup = [c for c, v in table.items() if v["status"] == "SUPPORTED"]
    rec = {"support_rule": f">= {SUPPORT_MIN} rows in each of {FIT}, {SEL}, {ASSESS}", "support_table": table,
           "supported_codes": rsup,
           "unsupported_codes": {c: v for c, v in table.items() if v["status"] != "SUPPORTED"},
           "unsupported_handling": "rows of unsupported codes excluded from the race audit; never pooled; never "
                                   "scored as zero leakage"}
    a = D["idx"][ASSESS]
    if len(rsup) < 2:
        rec["status"] = "NOT_ESTIMABLE (fewer than 2 supported classes)"
        return rec, {}, np.array([], np.int64), np.array([], np.int64)
    remap = np.full(Kr, -1)
    remap[rsup] = np.arange(len(rsup))
    yr = remap[race]
    idx = {r: D["idx"][r][yr[D["idx"][r]] >= 0] for r in (FIT, SEL, ASSESS)}
    ra, Pr = AU.final_audit({w: V[w] for w in ("v1", "v2", "pair")}, yr, idx[FIT], idx[SEL], idx[ASSESS],
                            slate_fn=AU.secondary_slate, finite=finite, K=len(rsup), classes=list(range(len(rsup))))
    ys = yr[idx[ASSESS]]
    per = {}
    for w in ("v1", "v2", "pair"):
        P = Pr[f"auc_{w}"]
        cls = [[float(AU.auc_fixed((ys == c).astype(int), np.stack([1 - p[:, c], p[:, c]], 1)))
                for c in range(len(rsup))] for p in P]
        m = np.mean(cls, 0)
        per[w] = {"per_class_ovr_auc_per_seed": {int(rsup[c]): [x[c] for x in cls] for c in range(len(rsup))},
                  "per_class_ovr_auc_mean": {int(rsup[c]): float(m[c]) for c in range(len(rsup))},
                  "worst_class": int(rsup[int(np.argmax(m))]), "worst_class_ovr_auc": float(m.max()),
                  "contrast_max_minus_min_ovr_auc": float(m.max() - m.min()),
                  "macro_ovr_auc_mean": ra["scored"][w]["auc"]["auc_mean"]}
    rec.update({"status": "ESTIMATED", "audit": ra, "per_view": per, "n_assessment": int(len(idx[ASSESS]))})
    pos = np.flatnonzero(yr[a] >= 0)
    return rec, Pr, pos.astype(np.int64), yr[a][pos].astype(np.int64)


# ------------------------------------------------------------------ one outer unit
def _audit(V, y, D, slate_fn, finite, ck, lk):
    idx = D["idx"]
    return AU.final_audit(V, y, idx[FIT], idx[SEL], idx[ASSESS], slate_fn=slate_fn, finite=finite, K=2,
                          classes=(0, 1), coalition_key=ck, local_keys=lk)


def outer_unit(D, k, label, spec, units_dir=None, u_spec=None):
    """Score one frozen release on OSF_DEVELOPMENT_ASSESSMENT (refuses unless the pushed lock was opened and D is
    unsealed). u_spec: the seed's U release spec (default: the lock's seeds[k].score[u_label])."""
    opened = _require_open()
    if D.get("sealed", True):
        raise SystemExit("REFUSED: D is sealed; use load_unsealed() after open_assessment()")
    name = f"outer__s{k}__{safe(label)}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    t0 = time.time()
    seed_entry = (opened.get("lock") or {}).get("seeds", {}).get(str(k), {})
    locked = seed_entry.get("unit_file_sha256")
    z, finite, native, prov = release_for(spec, D, units_dir, locked)
    if u_spec is None:
        u_label = seed_entry.get("u_label", "U")
        u_spec = (seed_entry.get("score") or {}).get(u_label)
    if u_spec is None:
        raise SystemExit(f"REFUSED: seed {k} has no locked U release for gain retention")
    zu, _, _, uprov = release_for(u_spec, D, units_dir, locked)
    S, a = D["sex"], D["idx"][ASSESS]
    V = OA.release_views(z)
    fam = {"complete": _audit({w: V[w] for w in ("v1", "v2", "pair")}, S, D, AU.final_slate, finite, "pair",
                              ("v1", "v2"))}
    for f in SECONDARY_FAMILIES:
        Vf, ck, lk = OA.family_views(V, f)
        fam[f] = _audit(Vf, S, D, AU.secondary_slate, True if f == "hard" else finite, ck, lk)
    race_rec, Pr, race_pos, race_y = race_diagnostic(V, D, finite)
    cst = constants(D)
    util, prb, prb_hard = {}, {}, {}
    for i, t in enumerate(TASKS):
        y = D["y"][t]
        util[i] = utility(z[f"p{i + 1}"], z[f"hard{i + 1}"], y, D, KS[i], cst[i], u_hard=zu[f"hard{i + 1}"])
        prb[i], prb_hard[i] = probe(np.asarray(z[f"r{i + 1}"], dtype=np.float64), y, D, KS[i])
    rec = {"unit": name, "seed": k, "label": label, "spec": spec, "u_spec": u_spec, "release_provenance": prov,
           "u_release_provenance": uprov,
           "evaluation_lock": {"commit": opened["commit"], "sha256": opened["sha256"], "code": opened.get("code")},
           "roles": {"attacker_fit": FIT, "attacker_selection": SEL, "scored": ASSESS},
           "primary": fam["complete"][0], "features_only": fam["features"][0],
           "outputs_only": {f: fam[f][0] for f in ("logits", "probs", "hard")},
           "race": race_rec, "utility_deployed": util, "utility_common_probe": prb,
           "linear_diagnostics": linear_diagnostics([z["r1"], z["r2"]], D), "native": native, "finite": finite,
           "n_assessment": int(len(a)), "n_assessment_groups": int(len(np.unique(D["unit"][a]))), "wall_s": None}
    preds = {"assess_row_id": D["row_id"][a], "assess_unit": D["unit"][a], "sex": S[a], "race": D["race"][a],
             "y_income": D["y"]["income"][a], "y_occ": D["y"]["occupation_group"][a],
             "const_class": np.array([cst[0], cst[1]], dtype=np.int64),
             "const1": np.full(len(a), cst[0], dtype=np.int64), "const2": np.full(len(a), cst[1], dtype=np.int64),
             "hard1": np.asarray(z["hard1"])[a], "hard2": np.asarray(z["hard2"])[a],
             "p1": np.asarray(z["p1"])[a], "p2": np.asarray(z["p2"])[a],
             "u_hard1": np.asarray(zu["hard1"])[a], "u_hard2": np.asarray(zu["hard2"])[a],
             "u_p1": np.asarray(zu["p1"])[a], "u_p2": np.asarray(zu["p2"])[a],
             "race_pos": race_pos, "race_codes": np.asarray(race_rec["supported_codes"], dtype=np.int64),
             "race_y": race_y, "probe_hard1": prb_hard[0], "probe_hard2": prb_hard[1]}
    for f, (_, P) in fam.items():
        for key, arr in P.items():
            preds[f"P_{key}"] = arr
    for key, arr in Pr.items():
        preds[f"Prace_{key}"] = arr
    rec["view_families"] = {f: {"coalition_key": ck, "local_keys": list(lk)} for f, (ck, lk) in OA.FAMILIES.items()}
    rec["preds_keys"] = sorted(preds)
    rec["wall_s"] = time.time() - t0
    FN.save_unit(udir(name, units_dir), {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    return json.loads((udir(name, units_dir) / "record.json").read_text())


# ------------------------------------------------------------------ controls (inner roles only; no lock needed)
def controls_for(D, specs, units_dir=None, slate="final", recipients=(1,)):
    out = {}
    for label, spec in specs.items():
        z, finite, _, prov = release_for(spec, D, units_dir)
        V = OA.release_views(z)
        c = OA.controls({w: V[w] for w in ("v1", "v2", "pair")}, D, finite=finite, slate=slate)
        rot = OA.rotated_plant(z, D, slate=slate, recipients=recipients, finite=finite)
        out[label] = {"spec": spec, "provenance": prov, **c, "rotated_plant": rot,
                      "all_ok": bool(c["all_ok"] and rot["all_ok"])}
    out["all_ok"] = all(v["all_ok"] for v in out.values() if isinstance(v, dict))
    return out


def main(argv=None):
    import argparse

    from osf import data as OD
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--shard", default=None, help="i/n over the (seed, label) list")
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--controls-only", action="store_true", help="inner-role controls on the lock's releases (sealed)")
    ap.add_argument("--controls-labels", nargs="*", default=["U"])
    ap.add_argument("--controls-slate", default="final")
    ap.add_argument("--controls-recipients", nargs="+", type=int, default=[1])
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.controls_only:
        L = json.loads(Path(a.evaluation_lock).read_text())
        D = OD.load()                                   # sealed: inner roles only
        for k in a.seeds:
            sc = L["seeds"][str(k)].get("score", {})
            labels = [lb for lb in a.controls_labels if lb in sc]
            res = controls_for(D, {lb: sc[lb] for lb in labels}, a.units_dir, a.controls_slate,
                               tuple(a.controls_recipients))
            FN.save_unit(udir(f"controls__s{k}", a.units_dir), {}, res)
            print(k, "controls all_ok =", res["all_ok"], flush=True)
        return
    L = open_assessment(a.evaluation_lock)
    D = load_unsealed()
    jobs = [(k, lb, sp) for k in a.seeds for lb, sp in L["seeds"][str(k)].get("score", {}).items()
            if not a.labels or lb in a.labels]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        jobs = [j for t, j in enumerate(jobs) if t % n == i]
    for k, label, spec in jobs:
        r = outer_unit(D, k, label, spec, a.units_dir)
        print(r["unit"], {w: round(r["primary"]["scored"][w]["auc"]["auc_mean"], 4) for w in ("v1", "v2", "pair")},
              flush=True)


if __name__ == "__main__":
    main()

"""Independent attacker interface for the online-strength frontier study (audit/baseline owner; not training critics).

The attacker slates, the defense-aware reader and its repaired numerical treatment are the PINNED predecessor module
smf.audit, imported unchanged (its docstring is the specification; summary below). This module adds only what the new
roles require: an explicit role guard for osf.data's D, release views with features-only / outputs-only / complete
recipient views recorded separately, the inner audit unit written per release, and selection-facing summaries.

Roles (osf.data). Attackers are FITTED on AUDIT_FIT (6,065 rows) and SELECTED on INNER_SELECTION (2,235 rows). Every
inner entry point here (a) refuses a D whose AUDIT_FIT / INNER_SELECTION positions carry another role (in particular
OSF_DEVELOPMENT_ASSESSMENT), (b) refuses sealed (< 0) labels on every row it fits, selects or scores (smf.audit
.check_labels), and (c) never indexes D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"] / "DEVELOPMENT_ASSESSMENT". Only
osf.assess passes assessment rows to `final_audit`, after the pushed EVALUATION_LOCK has been verified.

Recovery metric: AUC of P(S = 1 | view), fixed orientation (column 1 of predict_proba, never flipped; an AUC below
0.5 stays below 0.5; orientation is never chosen on any row, scored or not). Proper-loss recovery: cross-entropy.

Dual selection: the AUC-selected attacker (highest INNER_SELECTION AUC; primary) and the CE-selected attacker (lowest
INNER_SELECTION cross-entropy; proper-loss reporting) are selected and stored separately for every view.

Coalition bank (view "pair" = [v1, v2]): `coalition` (every slate member on the pair view), `ignore_recipient_2`
(every member on v1 alone), `ignore_recipient_1` (every member on v2 alone). An ignore-recipient candidate may win on
INNER_SELECTION; the inner coalition AUC is then >= the better inner local AUC as a property of the attack family (not
a clamp). On scored rows the selected coalition attacker is scored on its own and NEVER clamped to a local result.

Slates (smf.audit, unchanged; the same slate for every release; finite releases add the exact cell-conditional
attacker as in the predecessor):
  INNER      LR(C=1) | MLP(64,64) | HGB(lr 0.1, 31 leaves, 200 it.) | DA_LR                       finite: + CC x3
  FINAL      LR x5 (C 0.01..100) | MLP x4 | HGB x4 | DA_LR | DA_MLP                                  finite: + CC x3
  SECONDARY  INNER + DA_MLP (output-only, features-only views and the race diagnostic)
Defense-aware reader DA_* = smf.audit.Canon (float64 centre, SVD on AUDIT_FIT, drop s_j <= max(n, d) eps64
max(s_1, ||X||_2) -- numpy's matrix_rank tolerance with the uncentred scale -- and whiten every kept direction), so
exact algebraic nulls (centred affine logits, LEACE collapse) are dropped and never amplified while a rotated 1e-6
clue (s_j / s_1 ~ 1e-7, five orders above tol) is kept. Controls (smf.audit.controls / rotated_plant): shuffled-label
null with selection on half A and evaluation on held-out half B of INNER_SELECTION (exact-record-group split), planted
one-hot / 1e-6 leaks of the permuted label, and the rotated 1e-6 clue sent through save_unit / np.load /
views_from_release. Plants live inside the released view (the permitted attacker interface); a hidden raw-input leak
is never used as a control.

Recipient views (`release_views`; z = a release.npz in the rgj format: row_id, r1, c1, p1, hard1, r2, c2, p2, hard2):
  complete (PRIMARY)  v1 = [r1, c1], v2 = [r2, c2], pair = [v1, v2]
  features-only       r1, r2, rpair = [r1, r2]
  outputs-only        c1, c2, cpair (centred logits); p1, p2, ppair (probabilities); h1, h2, hpair (one-hot hard
                      decisions, finite)
Coalition keys / local keys per family: (pair; v1, v2), (rpair; r1, r2), (cpair; c1, c2), (ppair; p1, p2),
(hpair; h1, h2). All primary comparisons and every selection use the complete view only.

Inner audit unit (`inner_unit`): inner__<release unit> with
  of, of_complete_sha256 (COMPLETE.json hash of the audited release; a stale inner unit is refused), recovery =
  inner_audit(v1, v2, pair) on AUDIT_FIT -> INNER_SELECTION (INNER slate), utility = {0: income, 1: occupation_group}
  each {acc, const_acc, const_class, n} on INNER_SELECTION with const = the OSF_DEFENSE_FIT majority class,
  optional recovery_features_only / recovery_outputs_only (extended=True; descriptive), wall_s.
`inner_summary(name)` -> {"auc": {v1, v2, pair}, "ce": {...}, "selected": {...}, "ce_selected": {...},
"utility": {0: {...}, 1: {...}}} for the selection code.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rgj import finalize as FN
from smf import audit as AU
# pinned names re-exported unchanged (the slate and its repaired numerical treatment)
from smf.audit import (ATT_SEEDS, IGNORE_NAMES, NULL_CALIBRATION, NULL_Z, PLANT_MIN, PLANT_TINY_AMP,  # noqa: F401
                       PRIMARY_VIEWS, Canon, auc_fixed, check_labels, final_slate, inner_slate, logloss, null_sd,
                       null_split, planted_release, proba, rank_tolerance, secondary_slate, select_bank)

HOME = Path.home()
UNITS = HOME / "PCRL_eval_cache_private" / "osf_v1" / "run" / "units"
FIT_ROLE, SEL_ROLE = AU.FIT_ROLE, AU.SEL_ROLE            # "AUDIT_FIT", "INNER_SELECTION"
ASSESS_NAMES = ("OSF_DEVELOPMENT_ASSESSMENT", "DEVELOPMENT_ASSESSMENT")
TASKS = ("income", "occupation_group")
KS = (2, 6)
FAMILIES = {"complete": ("pair", ("v1", "v2")), "features": ("rpair", ("r1", "r2")),
            "logits": ("cpair", ("c1", "c2")), "probs": ("ppair", ("p1", "p2")), "hard": ("hpair", ("h1", "h2"))}
SLATE_OF = {"complete": "inner", "features": "secondary", "logits": "secondary", "probs": "secondary",
            "hard": "secondary"}


# ------------------------------------------------------------------ role guard
def guard_inner(D, labels=None):
    """Refuse a D whose attacker roles are not exactly AUDIT_FIT / INNER_SELECTION rows, or sealed labels on them."""
    for r in (FIT_ROLE, SEL_ROLE):
        if r not in D["idx"]:
            raise ValueError(f"REFUSED: D has no {r} role")
        ix = np.asarray(D["idx"][r])
        if "role" in D:
            got = np.unique(np.asarray(D["role"])[ix])
            if got.size and (len(got) != 1 or got[0] != r):
                raise ValueError(f"REFUSED: positions of {r} carry roles {got.tolist()} (assessment rows are never "
                                 "used by inner audits)")
        for a in ASSESS_NAMES:
            if a in D["idx"] and np.intersect1d(ix, D["idx"][a]).size:
                raise ValueError(f"REFUSED: {r} overlaps {a}")
    y = D["sex"] if labels is None else labels
    check_labels(np.asarray(y), D["idx"][FIT_ROLE], D["idx"][SEL_ROLE])


# ------------------------------------------------------------------ views
def release_views(z):
    """All recipient views of a release (dict of arrays over the release's rows). See module docstring."""
    g = {k: np.asarray(z[k]) for k in ("r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2")}
    V = {"v1": np.hstack([g["r1"], g["c1"]]), "v2": np.hstack([g["r2"], g["c2"]])}
    V["pair"] = np.hstack([V["v1"], V["v2"]])
    V.update({"r1": g["r1"], "r2": g["r2"], "rpair": np.hstack([g["r1"], g["r2"]]),
              "c1": g["c1"], "c2": g["c2"], "cpair": np.hstack([g["c1"], g["c2"]]),
              "p1": g["p1"], "p2": g["p2"], "ppair": np.hstack([g["p1"], g["p2"]])})
    h1 = np.eye(g["p1"].shape[1])[g["hard1"].astype(np.int64)]
    h2 = np.eye(g["p2"].shape[1])[g["hard2"].astype(np.int64)]
    V.update({"h1": h1, "h2": h2, "hpair": np.hstack([h1, h2])})
    return V


def family_views(V, family):
    ck, lk = FAMILIES[family]
    return {lk[0]: V[lk[0]], lk[1]: V[lk[1]], ck: V[ck]}, ck, lk


# ------------------------------------------------------------------ inner audit (selection statistic)
def inner_audit(V, D, finite=False, views=None, y=None, slate="inner"):
    """smf.audit.inner_audit after the osf role guard (complete views v1, v2, pair by default)."""
    guard_inner(D, y)
    return AU.inner_audit(V, D, finite=finite, views=views, y=y, slate=slate)


def inner_family(V, D, family, finite=False, slate=None):
    """Inner audit of one non-primary view family (features / logits / probs / hard), own coalition bank."""
    guard_inner(D)
    Vf, ck, lk = family_views(V, family)
    t0 = time.time()
    fin = True if family == "hard" else finite
    rec, _ = select_bank(Vf, D["sex"], D["idx"][FIT_ROLE], D["idx"][SEL_ROLE], AU.SLATES[slate or SLATE_OF[family]],
                         fin, coalition_key=ck, local_keys=lk)
    out = {"family": family, "auc": {}, "ce": {}, "selected": {}, "ce_selected": {}}
    for w, s in rec["selection"].items():
        out["auc"][w], out["ce"][w] = s["auc"]["inner_auc"], s["ce"]["inner_ce"]
        out["selected"][w], out["ce_selected"][w] = AU._label(s["auc"]), AU._label(s["ce"])
    out.update({"slate": slate or SLATE_OF[family], "coalition_key": ck, "local_keys": list(lk), **rec,
                "wall_s": round(time.time() - t0, 2)})
    return out


def constants(D):
    """OSF_DEFENSE_FIT majority class per task (the fitting-prior constant predictor)."""
    tr = D["idx"]["DEFENSE_FIT"]
    if "OSF_DEFENSE_FIT" in D["idx"] and not np.array_equal(D["idx"]["OSF_DEFENSE_FIT"], tr):
        raise ValueError("REFUSED: DEFENSE_FIT is not the OSF_DEFENSE_FIT alias")
    out = {}
    for i, t in enumerate(TASKS):
        y = np.asarray(D["y"][t])[tr]
        check_labels(y, np.arange(len(y)))
        out[i] = int(np.argmax(np.bincount(y, minlength=KS[i])))
    return out


def inner_utility(out, D, purposes=(0, 1)):
    """Deployed-head accuracy and constant accuracy on INNER_SELECTION (const = OSF_DEFENSE_FIT majority class).
    out holds hard1 / hard2 aligned with D's rows."""
    guard_inner(D)
    v = D["idx"][SEL_ROLE]
    cst = constants(D)
    res = {}
    for i in purposes:
        y = np.asarray(D["y"][TASKS[i]])[v]
        check_labels(y, np.arange(len(y)))
        h = np.asarray(out[f"hard{i + 1}"])[v]
        res[i] = {"acc": float((h == y).mean()), "const_acc": float((y == cst[i]).mean()), "const_class": cst[i],
                  "n": int(len(v))}
    return res


# ------------------------------------------------------------------ inner unit per release
def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_release(name, D, units_dir=None):
    d = udir(name, units_dir)
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: {name} is missing or not hash-complete")
    z = np.load(d / "release.npz")
    if not np.array_equal(z["row_id"], D["row_id"]):
        raise SystemExit(f"REFUSED: {name} rows are not aligned with osf D (row_id order)")
    return z


def inner_unit(name, D, units_dir=None, extended=False, finite=False):
    """Write (or return the cached) inner__<name> for one neural-format release unit. See module docstring."""
    iname = f"inner__{name}"
    src = _sha(udir(name, units_dir) / "COMPLETE.json")
    d = udir(iname, units_dir)
    if FN.unit_complete(d):
        rec = json.loads((d / "record.json").read_text())
        if rec.get("of", name) != name:
            raise RuntimeError(f"{iname} audits {rec.get('of')}, not {name}")
        if "of_complete_sha256" in rec and rec["of_complete_sha256"] != src:
            raise RuntimeError(f"{iname} was computed for another version of {name}; move it aside")
        return rec            # (a unit written by osf.inner carries no version hash; same inner_audit procedure)
    z = load_release(name, D, units_dir)
    V = release_views(z)
    t0 = time.time()
    rec = {"unit": iname, "of": name, "of_complete_sha256": src,
           "recovery": inner_audit({w: V[w] for w in PRIMARY_VIEWS}, D, finite=finite),
           "utility": inner_utility({"hard1": z["hard1"], "hard2": z["hard2"]}, D),
           "views": {"primary": "complete [r_i, c_i]; pair = [v1, v2]"}}
    if extended:
        rec["recovery_features_only"] = inner_family(V, D, "features", finite)
        rec["recovery_outputs_only"] = {f: inner_family(V, D, f, finite) for f in ("logits", "probs", "hard")}
    rec["wall_s"] = round(time.time() - t0, 2)
    FN.save_unit(d, {}, rec)
    return json.loads((d / "record.json").read_text())


def inner_summary(name, units_dir=None):
    """Selection-facing summary of inner__<name> (complete views only)."""
    rec = json.loads((udir(f"inner__{name}", units_dir) / "record.json").read_text())
    r = rec["recovery"]
    return {"unit": name, "auc": {w: r["auc"][w] for w in PRIMARY_VIEWS}, "ce": {w: r["ce"][w] for w in PRIMARY_VIEWS},
            "selected": {w: r["selected"][w] for w in PRIMARY_VIEWS},
            "ce_selected": {w: r["ce_selected"][w] for w in PRIMARY_VIEWS},
            "utility": {int(i): v for i, v in rec["utility"].items()}}


# ------------------------------------------------------------------ reader precision receipts (inner roles)
def precision_receipt(V, D, expected_rank=None, ambiguous_factor=1e3):
    """Canon receipt per view on the AUDIT_FIT rows: kept rank, tolerance, largest dropped and smallest kept singular
    values (relative to s_1), the number of kept directions within `ambiguous_factor` x tol (rounding-scale directions
    that would be whitened), and whether the kept rank equals the algebraic rank expected from the release
    construction (expected_rank = {view: rank}, e.g. 16 for [r_i, centred affine logits_i] of a 16-dim identity release)."""
    f = np.asarray(D["idx"][FIT_ROLE])
    out = {}
    for w, X in V.items():
        c = Canon().fit(np.asarray(X, dtype=np.float64)[f])
        rc = c.receipt()
        kept = c.s_[c.keep_]
        rc["kept_within_ambiguous_band"] = int((kept <= ambiguous_factor * c.tol_).sum())
        rc["dropped"] = int((~c.keep_).sum())
        rc["dtype"] = str(np.asarray(X).dtype)
        if expected_rank is not None and w in expected_rank:
            rc["expected_rank"] = int(expected_rank[w])
            rc["rank_matches_expected"] = rc["rank_kept"] == int(expected_rank[w])
        out[w] = rc
    return out


# ------------------------------------------------------------------ final slate and controls (pinned, guarded)
def final_audit(V, y, fit_idx, sel_idx, score_idx, **kw):
    """smf.audit.final_audit unchanged (fit fit_idx, dual-select sel_idx, refit seeds 0,1,2, score score_idx).
    Callers other than osf.assess pass inner/diagnostic rows only."""
    return AU.final_audit(V, y, fit_idx, sel_idx, score_idx, **kw)


def controls(V, D, finite=False, slate="final", views=None):
    guard_inner(D)
    return AU.controls(V, D, finite=finite, slate=slate, views=views)


def rotated_plant(z, D, slate="final", recipients=(1,), finite=False, workdir=None):
    guard_inner(D)
    return AU.rotated_plant(z, D, slate=slate, recipients=recipients, finite=finite, workdir=workdir)


# ------------------------------------------------------------------ pre-lock checks (inner roles only)
def real_null_calibration(V, D, reps=20, slate="inner", seed0=AU.CONTROL_SEED + 100, views=PRIMARY_VIEWS):
    """Shuffled-label nulls on REAL release views: per rep, SEX permuted within AUDIT_FIT and within each half of
    INNER_SELECTION (seed seed0 + rep), the slate fitted on AUDIT_FIT, the attacker selected on half A and evaluated on
    held-out half B (selection over the slate and the coalition bank accounted for). Every rep is kept."""
    guard_inner(D)
    halves = null_split(D)
    rows = []
    for k in range(reps):
        Sp, _ = AU.frozen_permutation(D["sex"], D, halves, seed=seed0 + k)
        thr, sd = AU.null_threshold(Sp, D, halves)
        r, _ = select_bank({w: np.asarray(V[w]) for w in views}, Sp, D["idx"][FIT_ROLE], D["idx"][SEL_ROLE],
                           AU.SLATES[slate], False, halves=halves)
        for w in views:
            sp = r["selection"][w]["split"]
            rows.append({"rep": k, "view": w, "heldout_auc_B": sp["heldout_auc_B"], "max_auc_A": sp["max_auc_A"],
                         "bank_max_full": r["selection"][w]["auc"]["inner_auc"], "sd0": sd, "threshold": thr,
                         "exceeds": sp["heldout_auc_B"] > thr, "selected_on_A": AU._label(sp)})
    hb = np.array([x["heldout_auc_B"] for x in rows])
    zz = (hb - 0.5) / np.array([x["sd0"] for x in rows])
    return {"summary": {"reps": reps, "slate": slate, "tests": len(rows), "exceedances": int(sum(x["exceeds"] for x in rows)),
                        "heldout_mean": float(hb.mean()), "heldout_max": float(hb.max()), "z_mean": float(zz.mean()),
                        "z_sd": float(zz.std(ddof=1)) if len(zz) > 1 else None, "frac_z_gt_2": float((zz > 2).mean()),
                        "bank_max_full_mean_descriptive": float(np.mean([x["bank_max_full"] for x in rows])),
                        "max_auc_A_mean_descriptive": float(np.mean([x["max_auc_A"] for x in rows])),
                        "threshold_mean": float(np.mean([x["threshold"] for x in rows])), "null_z": NULL_Z},
            "rows": rows}


def prelock_checks(D, releases, controls_labels, null_label, null_reps=(20, 5), workdir=None):
    """Pre-lock audit checks on INNER roles only. releases = {label: (release dict, finite, expected ranks | None)};
    None -> expected complete-view rank = the kept rank of its own feature block (logits are affine in r_i).
    controls_labels: labels that receive the full control battery (split null + planted one-hot / 1e-6 leaks with the
    FINAL slate, and the rotated 1e-6 clue in r1 and r2 through save_unit / np.load / views_from_release).
    null_label: the release used for the multi-permutation real-data null calibration (inner and final slates)."""
    guard_inner(D)
    out = {"roles": {"fit": FIT_ROLE, "select_and_heldout": SEL_ROLE, "assessment": "never indexed"},
           "plants": "all planted signals live inside the released view (the permitted attacker interface); no hidden "
                     "raw-input leak is used as a control",
           "precision": {}, "controls": {}, "rotated_plant": {}, "null_calibration": {}, "timing_s": {}}
    for lab, (z, finite, exp) in releases.items():
        V = release_views(z)
        rb = precision_receipt({w: V[w] for w in ("r1", "r2", "rpair")}, D)
        if exp is None:     # algebraic rank: the centred affine logits add no direction to their own feature block
            exp = {"v1": rb["r1"]["rank_kept"], "v2": rb["r2"]["rank_kept"], "pair": rb["rpair"]["rank_kept"]}
        out["precision"][lab] = {"dtypes": {k: str(np.asarray(z[k]).dtype) for k in ("r1", "c1", "r2", "c2")},
                                 "feature_blocks": rb, **precision_receipt({w: V[w] for w in PRIMARY_VIEWS}, D, exp)}
    for lab in controls_labels:
        z, finite, _ = releases[lab]
        V = release_views(z)
        t0 = time.time()
        out["controls"][lab] = controls({w: V[w] for w in PRIMARY_VIEWS}, D, finite=finite, slate="final")
        out["rotated_plant"][lab] = rotated_plant(z, D, slate="final", recipients=(1, 2), finite=finite,
                                                  workdir=workdir)
        out["timing_s"][lab] = round(time.time() - t0, 1)
    z, finite, _ = releases[null_label]
    V = release_views(z)
    for slate, reps in zip(("inner", "final"), null_reps):
        t0 = time.time()
        out["null_calibration"][f"{null_label}|{slate}"] = real_null_calibration(V, D, reps, slate)
        out["timing_s"][f"null|{slate}"] = round(time.time() - t0, 1)
    prec_ok = all(v.get("rank_matches_expected", True) and v["kept_within_ambiguous_band"] == 0
                  for r in out["precision"].values() for w, v in r.items() if w in PRIMARY_VIEWS)
    out["verdict"] = {
        "precision_ok": prec_ok,
        "controls_ok": all(c["all_ok"] for c in out["controls"].values()),
        "rotated_plant_ok": all(c["all_ok"] for c in out["rotated_plant"].values()),
        "real_null_exceedances": {k: v["summary"]["exceedances"] for k, v in out["null_calibration"].items()}}
    out["verdict"]["all_ok"] = bool(prec_ok and out["verdict"]["controls_ok"] and out["verdict"]["rotated_plant_ok"] and
                                    not any(out["verdict"]["real_null_exceedances"].values()))
    return out


PRELOCK_CODE = ("osf/audit.py", "osf/baselines.py", "osf/data.py", "smf/audit.py", "jcv/audit.py", "rgj/finalize.py",
                "jcv/finalize.py")


def run_prelock(D, out_path, units_dir=None, workdir=None):
    """AUDIT_PRELOCK_CHECKS.json: U releases of seeds 0, 1, 2 rebuilt in memory from the admitted tl__s{k}__e40
    (osf.baselines.u_release; inner roles only) receive the full battery; the lead's rel__* release units present at
    run time receive precision receipts only. Code hashes are taken at start; compute is recorded."""
    import platform
    import resource
    from osf import baselines as BL
    t0, c0 = time.time(), time.process_time()
    wt = Path(__file__).resolve().parents[1]
    code = {f: _sha(wt / f) for f in PRELOCK_CODE}
    if not D.get("sealed", False):
        raise SystemExit("REFUSED: pre-lock checks run on the sealed D only")
    rel, prov = {}, {}
    for k in (0, 1, 2):
        out, _, rec, _, _ = BL.u_release(k, D, units_dir)
        rel[f"U|s{k}"] = ({"row_id": D["row_id"], **out}, False, None)
        prov[f"U|s{k}"] = {kk: rec.get(kk) for kk in ("source", "origin", "model_pt_sha256", "replay_exact",
                                                       "lead_unit_max_abs_diff")}
    for d in sorted(Path(units_dir or UNITS).glob("rel__*")):
        if FN.unit_complete(d):
            z = np.load(d / "release.npz")
            if np.array_equal(z["row_id"], D["row_id"]):
                rel[f"lead:{d.name}"] = ({kk: z[kk] for kk in z.files}, False, None)
                prov[f"lead:{d.name}"] = {"complete_sha256": _sha(d / "COMPLETE.json")}
    res = prelock_checks(D, rel, controls_labels=["U|s0", "U|s1", "U|s2"], null_label="U|s0", workdir=workdir)
    from osf import data as OD
    man = OD.manifest(D)
    res.update({"study": "pcrl_online_strength_frontier_v1", "scope": "inner roles only (AUDIT_FIT fit, "
                "INNER_SELECTION select / held-out halves); OSF_DEVELOPMENT_ASSESSMENT never indexed; sealed D",
                "releases": prov, "roles_manifest": {r: man[r] for r in ("AUDIT_FIT", "INNER_SELECTION")},
                "code_sha256_at_start": code, "software": {"python": platform.python_version(), "numpy": np.__version__},
                "compute": {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "processes": 1, "omp_threads": 1}})
    txt = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for bad in (str(Path.home()), "PCRL_eval_cache_private"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter the public pre-lock receipt")
    Path(out_path).write_text(txt + "\n")
    return res


def main(argv=None):
    """python -m osf.audit inner --units <name> [<name> ...] [--units-dir DIR] [--extended]
    python -m osf.audit prelock [--out results/pcrl_online_strength_frontier_v1/AUDIT_PRELOCK_CHECKS.json]"""
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("inner", "prelock"))
    ap.add_argument("--units", nargs="+", default=[])
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" /
                                         "pcrl_online_strength_frontier_v1" / "AUDIT_PRELOCK_CHECKS.json"))
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from osf import data as OD
    D = OD.load()
    if a.cmd == "prelock":
        r = run_prelock(D, a.out, a.units_dir)
        print(json.dumps(r["verdict"], indent=1), r["compute"], flush=True)
        return
    for n in a.units:
        r = inner_unit(n, D, a.units_dir, extended=a.extended)
        print(r["unit"], {w: round(r["recovery"]["auc"][w], 4) for w in PRIMARY_VIEWS}, flush=True)


if __name__ == "__main__":
    main()

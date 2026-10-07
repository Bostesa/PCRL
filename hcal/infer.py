"""Inference of the held-out calibration study (hcal) from saved OSF_DEVELOPMENT_ASSESSMENT predictions and the
EVALUATION_LOCK only (no refits, no selection). Adapted from lra/infer.py at 9762025 (same bootstrap graph, same
per-seed paired statistics, same clause classifier); the family is hcal.family (23 slots).

Per model seed: recovery = SEX AUC (score P(SEX=1), fixed orientation) of the inner-AUC-selected common-bank attacker of
the release's partition (continuous U: its composed bank), averaged over attacker seeds 0-2; accuracy = released
decisions; log loss = mean -log(clip(prob_y, 1e-12)); Brier = mean sum_k (prob_k - 1[y = k])^2; const = OSF_DEFENSE_FIT
majority class. Endpoint = mean over seeds of the per-seed paired statistic. SE = sd (ddof 1) over B = 1999 paired
multinomial bootstrap replicates of exact-record groups (seed 20261011; identical draws for every statistic, arm and
seed); interval = point +- z SE, z = 3.0653831516447343. P slots: clause outcomes; with no nominee they are recorded
NOT_SCORED_NO_NOMINEE (the family size stays 23). D slots: signed diagnostics (SUPPORTS_POSITIVE / SUPPORTS_NEGATIVE /
UNRESOLVED). Supplementary fixed contrasts: nominal 95% (z = 1.959963984540054), descriptive, never primary.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.infer --evaluation-lock results/pcrl_heldout_calibration_v1/EVALUATION_LOCK.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from hcal import family as FAM
from hcal import ids as I

SEEDS = I.SEEDS
EPS = 1e-12
VIEWS = ("v1", "v2", "pair")


def row_losses(prob, y):
    prob = np.asarray(prob, dtype=np.float64)
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], EPS, 1.0))
    br = ((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1)
    return ll, br


def att_name(k, key):
    return f"oatt__s{k}__{I.safe(key)}"


def prob_name(k, rid):
    return f"oprob__s{k}__{I.safe(rid)}"


def partition_key(rid):
    return "SRC|U" if rid.startswith(I.U_ID) else I.parse_release(rid)[0]


class Ctx:
    def __init__(self, EL, units=None):
        self.EL, self.g = EL, G()
        units = Path(units) if units else I.UNITS
        self.labels = list(EL["scored_releases"])
        self.probs, self.att = {}, {}
        z0 = None
        for k in SEEDS:
            for rid in self.labels:
                z = np.load(units / prob_name(k, rid) / "preds.npz", allow_pickle=False)
                p = {x: z[x] for x in z.files}
                self.probs[(k, rid)] = p
                z0 = z0 if z0 is not None else p
                for key in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ", "const_class"):
                    assert np.array_equal(p[key], z0[key]), f"{key} differs across arms"
                pk = partition_key(rid)
                if (k, pk) not in self.att:
                    za = np.load(units / att_name(k, pk) / "preds.npz", allow_pickle=False)
                    a = {x: za[x] for x in za.files}
                    assert np.array_equal(a["assess_row_id"], z0["assess_row_id"])
                    self.att[(k, pk)] = a
        self.finiteness = {f"s{k}|{rid}": {x: int((~np.isfinite(np.asarray(p[x], dtype=np.float64))).sum())
                                            for x in p if x.startswith(("prob", "hard"))}
                           for (k, rid), p in self.probs.items()}
        self.finiteness.update({f"s{k}|att|{pk}": {x: int((~np.isfinite(np.asarray(a[x], dtype=np.float64))).sum())
                                                   for x in a if x.startswith(("P_auc_", "P_ce_"))}
                                for (k, pk), a in self.att.items()})
        self.units, self.sex, self.rows = z0["assess_unit"], z0["sex"], z0["assess_row_id"]
        self.y = {0: z0["y_income"], 1: z0["y_occ"]}
        self.const = {j: int(z0["const_class"][j]) for j in (0, 1)}

    def resolve(self, role):
        return self.EL["resolved"].get(role)

    def rec(self, k, rid, view, crit="auc"):
        pk = partition_key(rid)
        P3 = np.asarray(self.att[(k, pk)][f"P_{crit}_{view}"], dtype=np.float64)
        tag = f"{crit}#{k}#{pk}#{view}"
        if not np.isfinite(P3[..., 1]).all():
            nan = lambda WT: np.full(WT.shape[1], np.nan)                      # noqa: E731
            ids = [self.g.base_once(f"{tag}#{s}", f"{tag}#{s}", nan) for s in range(3)]
        else:
            ids = [self.g.base_once(f"{tag}#{s}", f"{tag}#{s}", class_auc(self.sex, P3[s], 1)) for s in range(3)]
        return self.g.add(f"R#{tag}", "mean", ids)

    def acc(self, k, rid, j):
        p = self.probs[(k, rid)]
        return self.g.base_once(f"acc#{k}#{rid}#{j}", f"acc#{k}#{rid}#{j}", acc_stat(p[f"hard{j + 1}"] == self.y[j]))

    def loss(self, k, rid, j, kind):
        p = self.probs[(k, rid)]
        ll, br = row_losses(p[f"prob{j + 1}"], self.y[j])
        return self.g.base_once(f"{kind}#{k}#{rid}#{j}", f"{kind}#{k}#{rid}#{j}", acc_stat(ll if kind == "ll" else br))

    def constacc(self, j):
        return self.g.base_once(f"const#{j}", f"const#{j}", acc_stat(self.y[j] == self.const[j]))


def build(ctx):
    g, ids = ctx.g, {}
    U0 = I.U_ID
    nom, tref, ucal = ctx.resolve("P*"), ctx.resolve("T*"), ctx.resolve("Ucal*")
    for e in FAM.PRIMARY:
        if e["kind"] == "diag":
            dp = e["partition"]
            a, b = I.release_id(dp, e["minuend"]), I.release_id(dp, e["subtrahend"])
            per = [g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, a, e["task"], e["loss"]),
                                                    ctx.loss(k, b, e["task"], e["loss"])]) for k in SEEDS]
            ids[e["id"]] = g.add(e["id"], "mean", per)
            continue
        if nom is None or tref is None or ucal is None:
            ids[e["id"]] = None
            continue
        ref = {"T*": tref, "U0": U0, "Ucal*": ucal}[e["ref"]]
        per = []
        for k in SEEDS:
            j = e.get("task")
            if e["kind"] == "coalition":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, ref, "pair"), ctx.rec(k, nom, "pair")]))
            elif e["kind"] == "local":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, nom, e["view"]), ctx.rec(k, ref, e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.acc(k, nom, j), ctx.acc(k, ref, j)]))
            elif e["kind"] in ("logloss", "brier"):
                kind = "ll" if e["kind"] == "logloss" else "br"
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, nom, j, kind), ctx.loss(k, ref, j, kind)]))
            else:
                per.append(g.add(f"{e['id']}#{k}", "lin", [ctx.acc(k, nom, j), ctx.acc(k, ref, j), ctx.constacc(j)]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    for rid in ctx.labels:
        for k in SEEDS:
            for v in VIEWS:
                levels[f"R#{k}#{rid}#{v}"] = ctx.rec(k, rid, v)
                levels[f"Rce_sel_auc#{k}#{rid}#{v}"] = ctx.rec(k, rid, v, "ce")
            for j in (0, 1):
                levels[f"acc#{k}#{rid}#{j}"] = ctx.acc(k, rid, j)
                levels[f"ll#{k}#{rid}#{j}"] = ctx.loss(k, rid, j, "ll")
                levels[f"br#{k}#{rid}#{j}"] = ctx.loss(k, rid, j, "br")
        for v in VIEWS:
            levels[f"Rmean#{rid}#{v}"] = g.add(f"Rmean#{rid}#{v}", "mean", [ctx.rec(k, rid, v) for k in SEEDS])
        for j in (0, 1):
            for kind, fn in (("acc", lambda k: ctx.acc(k, rid, j)), ("ll", lambda k: ctx.loss(k, rid, j, "ll")),
                             ("br", lambda k: ctx.loss(k, rid, j, "br"))):
                levels[f"{kind}mean#{rid}#{j}"] = g.add(f"{kind}mean#{rid}#{j}", "mean", [fn(k) for k in SEEDS])
            for kind in ("ll", "br"):
                for refname, refrid in (("U0", U0), ("Ucal", ucal)):
                    if refrid is None or rid == refrid:
                        continue
                    per = [g.add(f"x{kind}{refname}#{k}#{rid}#{j}", "diff",
                                 [ctx.loss(k, rid, j, kind), ctx.loss(k, refrid, j, kind)]) for k in SEEDS]
                    levels[f"x{kind}{refname}mean#{rid}#{j}"] = g.add(f"x{kind}{refname}mean#{rid}#{j}", "mean", per)
    for j in (0, 1):
        levels[f"const#{j}"] = ctx.constacc(j)
    sup = {}
    for dp in FAM.SUPPLEMENTARY_PARTITIONS:
        for name, a_dec, b_dec in FAM.SUPPLEMENTARY_CONTRASTS:
            if a_dec not in I.decoders_of(dp) or b_dec not in I.decoders_of(dp):
                continue
            a, b = I.release_id(dp, a_dec), I.release_id(dp, b_dec)
            if a not in ctx.labels or b not in ctx.labels:
                continue
            for j in (0, 1):
                for kind in ("ll", "br"):
                    per = [g.add(f"sup#{dp}#{name}#{kind}#{j}#{k}", "diff",
                                 [ctx.loss(k, a, j, kind), ctx.loss(k, b, j, kind)]) for k in SEEDS]
                    sup[(dp, name, kind, j)] = g.add(f"sup#{dp}#{name}#{kind}#{j}", "mean", per)
    return ids, levels, sup


def nominee_states(EL):
    st = EL["statuses"]
    return (st["P*"]["state"], st["T*"]["state"])


def main(argv=None, units=None, run_dir=None, pkg_dir=None):
    import argparse
    from hcal import run as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--failures", default=None)
    a = ap.parse_args(argv)
    EL = json.loads(Path(a.evaluation_lock).read_text())
    if (EL.get("technical_validity") or {}).get("engineering_gate") != "ENGINEERING_READY":
        raise SystemExit("REFUSED: the evaluation lock does not bind ENGINEERING_READY")
    failures = json.loads(Path(a.failures).read_text()) if a.failures else {}
    ctx = Ctx(EL, units=units)
    ids, levels, sup = build(ctx)
    boot = UnitBootstrap(ctx.units, FAM.B, FAM.BOOT_SEED, 250)
    want = [i for i in ids.values() if i] + list(levels.values()) + list(sup.values())
    pts, reps = run(ctx.g, boot, list(dict.fromkeys(want)))
    z = FAM.Z_PRIMARY
    nstate, cstate = nominee_states(EL)
    out = {"schema": "hcal-inference-v1", "primary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED, "z": z,
           "z_definition": "NormalDist().inv_cdf(1 - 0.05 / (2 * 23))",
           "resampling_unit": "OSF_DEVELOPMENT_ASSESSMENT exact-record group", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units))), "resolved": EL["resolved"], "statuses": EL["statuses"]}
    outcomes = {}
    for e in FAM.PRIMARY:
        sid = ids[e["id"]]
        base = {k2: v for k2, v in e.items()}
        base["z"] = z
        if sid is None:
            row = {**base, "point": None, "se": None, "lower": None, "upper": None,
                   "outcome": "NOT_SCORED_NO_NOMINEE", "decision": "NOT_SCORED_NO_NOMINEE"}
            out["primary"].append(row)
            outcomes[e["id"]] = row["decision"]
            continue
        r = reps[sid]
        pt = float(pts[sid])
        nonfinite = int((~np.isfinite(r)).sum())
        if nonfinite or not np.isfinite(pt):
            row = {**base, "point": pt, "outcome": "INVALID", "decision": "INVALID", "nonfinite_replicates": nonfinite}
        else:
            se = float(np.std(r, ddof=1))
            lo, hi = pt - z * se, pt + z * se
            oc = FAM.diag_outcome(pt, lo, hi) if e["kind"] == "diag" else FAM.clause_outcome(e["side"], e["target"],
                                                                                              pt, lo, hi)
            row = {**base, "point": pt, "se": se, "lower": lo, "upper": hi, "outcome": oc, "decision": oc,
                   "n_finite_replicates": int(len(r))}
        out["primary"].append(row)
        outcomes[e["id"]] = row["decision"]
    for nm, sid in levels.items():
        r = reps[sid]
        fin = r[np.isfinite(r)]
        out["levels"][nm] = {"point": float(pts[sid]), "se": float(np.std(fin, ddof=1)) if len(fin) > 1 else None,
                             "nonfinite": int((~np.isfinite(r)).sum())}
    z95 = FAM.Z_SUPPLEMENTARY
    out["supplementary"] = {"z": z95, "z_definition": "NormalDist().inv_cdf(0.975) (nominal 95%; descriptive; not "
                                                       "primary slots)", "rows": []}
    for (dp, name, kind, j), sid in sup.items():
        r = reps[sid]
        pt = float(pts[sid])
        se = float(np.std(r, ddof=1)) if np.isfinite(r).all() and np.isfinite(pt) else None
        out["supplementary"]["rows"].append({
            "partition": dp, "contrast": name, "loss": "logloss" if kind == "ll" else "brier",
            "task": FAM.TASKS[j], "point": pt, "se": se, "lower": None if se is None else pt - z95 * se,
            "upper": None if se is None else pt + z95 * se,
            "reading": None if se is None else FAM.diag_outcome(pt, pt - z95 * se, pt + z95 * se)})
    tv = bool((EL.get("technical_validity") or {}).get("ok")) and not failures.get("global")
    orig, cal = FAM.criteria(outcomes, nstate == "ELIGIBLE", tv and "TECHNICAL_FAILURE" not in (nstate, cstate),
                             cstate == "ELIGIBLE")
    label = FAM.overall_label(True, True, tv, nstate, cstate, outcomes)
    out.update({
        "outcomes": outcomes, "OriginalCriterion": orig, "CalibrationMatchedCriterion": cal,
        "FittingRole": {"slots": {i: outcomes[i] for i in FAM.FITTING_IDS},
                        "summary": FAM.diag_summary(outcomes[i] for i in FAM.FITTING_IDS)},
        "ParameterSharing": {"slots": {i: outcomes[i] for i in FAM.SHARING_IDS},
                             "summary": FAM.diag_summary(outcomes[i] for i in FAM.SHARING_IDS)},
        "ExactDecisionPreservation": EL.get("decision_preservation_receipt"),
        "label": label, "technical_valid": tv, "failures_input": failures or None,
        "finiteness_receipt": {"nonfinite_counts": ctx.finiteness,
                               "all_finite": not any(v for d in ctx.finiteness.values() for v in d.values())}})
    out = R._finite(out)
    run_dir = Path(run_dir) if run_dir else R.RUN
    pkg_dir = Path(pkg_dir) if pkg_dir else I.PKG
    (run_dir / "inference.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    cols = ["id", "group", "kind", "stat", "target", "side", "point", "se", "lower", "upper", "z", "outcome", "decision"]
    with open(pkg_dir / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for e in out["primary"]:
            w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    with open(pkg_dir / "ALL_LEVELS.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "seed", "release", "detail", "point", "se"])
        for nm, v in sorted(out["levels"].items()):
            parts = nm.split("#")
            if parts[0] in ("R", "Rce_sel_auc", "acc", "ll", "br"):
                seed, lab_, det = parts[1], parts[2], "|".join(parts[3:])
            elif parts[0] == "const":
                seed, lab_, det = "", "const", parts[1]
            else:
                seed, lab_, det = "mean", parts[1], "|".join(parts[2:])
            f6 = lambda x: "" if x is None else f"{x:.6f}"                       # noqa: E731
            w.writerow([parts[0], seed, lab_, det, f6(v["point"]), f6(v["se"])])
    return out


if __name__ == "__main__":
    main()

"""Report tables of the held-out calibration study (hcal; role A). Separately versioned report code: it reads saved
records only (never refits, never reselects) and writes aggregate public tables; locked raw results are never
overwritten. Per-person probabilities, labels, memberships and private paths never enter these files.

    PYTHONPATH=. <python> -m hcal.report inner    AUDIT_PLAN.json, SELECTION.json, INNER_SELECTION_TABLE.csv,
                                                  COMPLETE_INTERFACE_EQUIVALENCE.json, CALIBRATION_FITS.csv,
                                                  CONTROLS_RESULT.json
    PYTHONPATH=. <python> -m hcal.report assess   CALIBRATION_GENERALIZATION.csv (adds assessment columns)
"""
from __future__ import annotations

import csv
import json
import sys


from hcal import ids as I


def _R():
    from hcal import run as R
    return R


def f6(x):
    return "" if x is None else (f"{x:.6f}" if isinstance(x, float) else x)


def mean3(v):
    return (float(v[0]) + float(v[1]) + float(v[2])) / 3.0


def inner_tables():
    from hcal import stages as ST
    R = _R()
    plan = json.loads((R.RUN / "audit_plan.json").read_text())
    sel = json.loads((R.RUN / "selection.json").read_text())
    table = sel["table"]
    util = {k: R.rec(ST.util_name(k))["releases"] for k in I.SEEDS}
    pub_plan = {k: v for k, v in plan.items()}
    (I.PKG / "AUDIT_PLAN.json").write_text(json.dumps(pub_plan, indent=1) + "\n")
    pub = {k: v for k, v in sel.items() if k != "table"}
    (I.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1) + "\n")
    com = {}
    for k in I.SEEDS:
        for p in plan["audited"]:
            com[(k, p)] = R.rec(ST.com_name(k, p))
        com[(k, "SRC|U")] = R.rec(ST.com_name(k, "SRC|U"))
    cols = ["release", "partition", "decoder", "privacy_trained", "task_only_candidate", "nominee_capable", "audited",
            "eligible_all_seeds", "eligible_u0_all_seeds", "eligible_ucal_all_seeds", "worst_norm_excess",
            "mean_summed_nll", "mean_inner_ll_income", "mean_inner_ll_occupation", "mean_inner_brier_income",
            "mean_inner_brier_occupation", "mean_pair_auc", "mean_v1_auc", "mean_v2_auc"]
    with open(I.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for rid in I.code_release_ids() + I.u_release_ids():
            if rid.startswith(I.U_ID):
                p, dec, key = None, I.parse_release(rid)[1], "SRC|U"
            else:
                p, dec = I.parse_release(rid)
                key = p
            t = table[rid]
            row = {"release": rid, "partition": p or "SRC|U", "decoder": dec,
                   "privacy_trained": bool(p and I.privacy_trained(p)), "task_only_candidate": I.task_only_candidate(rid),
                   "nominee_capable": I.nominee_capable(rid), "audited": key == "SRC|U" or p in plan["audited"],
                   "eligible_all_seeds": t["eligible_all_seeds"], "eligible_u0_all_seeds": t["eligible_u0_all_seeds"],
                   "eligible_ucal_all_seeds": t["eligible_ucal_all_seeds"], "worst_norm_excess": t["worst_norm_excess"],
                   "mean_summed_nll": t["mean_summed_nll"]}
            for task in ("income", "occupation"):
                row[f"mean_inner_ll_{task}"] = mean3([util[k][rid]["inner"][task]["logloss"] for k in I.SEEDS])
                row[f"mean_inner_brier_{task}"] = mean3([util[k][rid]["inner"][task]["brier"] for k in I.SEEDS])
            if (0, key) in com:
                for v in ("pair", "v1", "v2"):
                    row[f"mean_{v}_auc"] = mean3([com[(k, key)]["recovery"]["auc"][v] for k in I.SEEDS])
            w.writerow({c: f6(row.get(c)) for c in cols})
    eq = {"schema": "hcal-complete-interface-equivalence-v1",
          "rule": "one common attack bank per (partition, seed): every decoder variant receives the same selected "
                  "attack predictions, hence identical primary AUC / CE; the fresh complete view carries every "
                  "registered public lookup table; continuous U's three decoder variants share its composed bank",
          "partitions": {}, "u": {}}
    for (k, key), r in com.items():
        if key == "SRC|U":
            eq["u"][str(k)] = {"variants": I.u_release_ids(), "composed_fresh_partitions":
                               len(r.get("composed_fresh_partitions") or []),
                               "auc": r["recovery"]["auc"], "winner": r["recovery"]["winner"]}
        else:
            vr = r["variants_receipt"]
            eq["partitions"].setdefault(key, {})[str(k)] = {"variants": vr["variants"], "ok": vr["ok"],
                                                            "record_sha256": vr["record_sha256"],
                                                            "auc": r["recovery"]["auc"],
                                                            "winner": r["recovery"]["winner"]}
    eq["all_ok"] = all(v["ok"] for p in eq["partitions"].values() for v in p.values())
    eq["skipped_partitions"] = {p: "PREDECLARED_UTILITY_INELIGIBLE (no fresh bank; legacy banks remain in the "
                                   "admitted lra U composition)" for p in plan["skipped"]}
    (I.PKG / "COMPLETE_INTERFACE_EQUIVALENCE.json").write_text(json.dumps(eq, indent=1) + "\n")
    # calibration fits (parameter / token counts, alphas, calibration-row losses)
    cols = ["seed", "partition", "family", "recipient", "tokens", "occupied_by_calibration_rows", "parameter_count",
            "alpha", "alphas", "status_counts"]
    with open(I.PKG / "CALIBRATION_FITS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for k in I.SEEDS:
            for p in I.partitions():
                r = R.rec(ST.cal_name(k, p))
                for key, s in r["tables"].items():
                    fam, rr = key.split("|")
                    w.writerow({"seed": k, "partition": p, "family": fam, "recipient": rr, "tokens": s["tokens"],
                                "occupied_by_calibration_rows": s["occupied_by_calibration_rows"],
                                "parameter_count": s["parameter_count"], "alpha": f6(s["alpha"]),
                                "alphas": json.dumps(s["alphas"]), "status_counts": json.dumps(s["status_counts"])})
            u = R.rec(ST.calu_name(k))
            for key, s in u["tables"].items():
                fam, rr = key.split("|")
                w.writerow({"seed": k, "partition": "SRC|U", "family": fam, "recipient": rr, "tokens": "",
                            "occupied_by_calibration_rows": "", "parameter_count": s["parameter_count"],
                            "alpha": f6(s["alpha"]), "alphas": json.dumps(s["alphas"]),
                            "status_counts": json.dumps(s["status_counts"])})
    ctl = json.loads((R.RUN / "controls.json").read_text())
    (I.PKG / "CONTROLS_RESULT.json").write_text(json.dumps(ctl, indent=1) + "\n")


def generalization(inference=None):
    """CALIBRATION_GENERALIZATION.csv: per registered release (diagnostic partitions, T*, P*, U variants) the mean over
    seeds of fitting-row, train-matched, calibration-row, inner and (when given) assessment log loss and Brier."""
    from hcal import stages as ST
    R = _R()
    util = {k: R.rec(ST.util_name(k))["releases"] for k in I.SEEDS}
    sel = json.loads((R.RUN / "selection.json").read_text())
    rids = [I.release_id(p, d) for p in I.DIAGNOSTIC_PARTITIONS for d in I.decoders_of(p)] + I.u_release_ids()
    for role in ("T*", "P*"):
        if sel[role].get("release"):
            rids.append(sel[role]["release"])
    rids = list(dict.fromkeys(rids))
    lev = (inference or {}).get("levels") or {}
    cols = ["release", "task", "metric", "fit_rows", "train_matched_rows", "calibration_rows", "inner_rows",
            "assessment_rows", "assessment_se"]
    with open(I.PKG / "CALIBRATION_GENERALIZATION.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for rid in rids:
            for j, task in enumerate(("income", "occupation")):
                for metric, key in (("logloss", "ll"), ("brier", "br")):
                    row = {"release": rid, "task": task, "metric": metric}
                    for col, tag in (("fit_rows", "fit"), ("train_matched_rows", "tm"), ("calibration_rows", "cal"),
                                     ("inner_rows", "inner")):
                        row[col] = mean3([util[k][rid][tag][task][metric] for k in I.SEEDS])
                    a = lev.get(f"{key}mean#{rid}#{j}")
                    if a:
                        row["assessment_rows"], row["assessment_se"] = a["point"], a["se"]
                    w.writerow({c: f6(row.get(c)) for c in cols})


if __name__ == "__main__":
    if sys.argv[1:] == ["inner"]:
        inner_tables()
        generalization()
    elif sys.argv[1:] == ["assess"]:
        inf = json.loads((_R().RUN / "inference.json").read_text())
        generalization(inf)

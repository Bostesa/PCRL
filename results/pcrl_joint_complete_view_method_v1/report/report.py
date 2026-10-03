"""Aggregate public tables from unit records (no per-person data).
    ~/PCRL/.venv/bin/python results/pcrl_joint_complete_view_method_v1/report/report.py diagnostics|inner|outer
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jcv import run as R  # noqa: E402


def rec(name):
    return json.loads((R.U(name) / "record.json").read_text())


def diagnostics():
    rows = []
    for d in sorted(R.UNITS.glob("nn__s*")):
        if d.name.endswith(".quarantined") or not R.done(d.name):
            continue
        r = rec(d.name)
        g = r.get("diag", {})
        logs = g.get("stage_logs", [])
        rows.append({"unit": d.name, "arm": r["arm"], "beta": r["beta"], "seed": r["seed"],
                     "wall_s": round(r.get("wall_s") or 0, 1), "cpu_s": round(r.get("cpu_s") or 0, 1),
                     "encoder_updates": g.get("steps"), "protection_steps": g.get("protection_steps_attempted"),
                     "rejected": g.get("protection_steps_rejected"), "backtrack_halvings": g.get("backtracks"),
                     "active_0/1/2": "/".join(str(g.get("projection_active", {}).get(k, "")) for k in ("0", "1", "2")),
                     "max_projection_violation": g.get("max_projection_violation"), "nonfinite": g.get("nonfinite"),
                     "leace_refits": g.get("leace_refits"), "rescue": bool(g.get("rescue")),
                     "final_R_train": json.dumps({k: round(v, 3) for k, v in (logs[-1]["R"] if logs and logs[-1].get("R") else {}).items()}),
                     "head_C": json.dumps({i: h["selected_C"] for i, h in r["finalize"]["heads"].items()}),
                     "leace_native": "/".join(l["native_check_fit_rows"]["status"] for l in r["finalize"]["leace"].values())
                     if r["finalize"]["leace"] else "n/a"})
    with open(R.PKG / "TRAINING_DIAGNOSTICS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def inner_table():
    sel = json.loads((R.RUN / "selection.json").read_text())
    rows = []
    for k, s in sel.items():
        for arm, a in s["arms"].items():
            for t in a.get("table", [a]):
                rows.append({"seed": k, "arm": arm, "unit": t.get("unit"), "beta": t.get("beta"),
                             "gates_ok": t.get("gates_ok"), "worst_gate_margin": t.get("worst_gate_margin"),
                             "R_v1": t.get("R_v1"), "R_v2": t.get("R_v2"), "R_pair": t.get("R_pair"),
                             "selected": t.get("unit") == a.get("unit") and a["status"] == "NOMINEE", "arm_status": a["status"]})
        for i, fs in s["fare_selection"].items():
            for t in fs["table"]:
                rows.append({"seed": k, "arm": f"F(p{i})", "unit": t["unit"], "beta": t["config"], "gates_ok": t["gate_ok"],
                             "worst_gate_margin": t["margin"], "R_v1": t["R_local"] if str(i) == "0" else None,
                             "R_v2": t["R_local"] if str(i) == "1" else None, "R_pair": None,
                             "selected": t["config"] == fs["config"] and fs["status"] == "NOMINEE", "arm_status": fs["status"]})
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def outer_tables():
    util, native = [], []
    for d in sorted(R.UNITS.glob("outer__s*")):
        if d.name.endswith(".quarantined"):
            continue
        r = rec(d.name)
        for i, t in enumerate(("income", "occupation_group")):
            u = r["utility_deployed"][str(i)]
            util.append({"seed": r["seed"], "arm": r["arm"], "status": r["status"], "task": t,
                         "accuracy": u["accuracy"], "balanced_accuracy_supported": u["balanced_accuracy_supported"],
                         "minority_class": u["minority_class"], "minority_recall": u["minority_recall"],
                         "log_loss": u["log_loss"], "brier": u["brier"], "ece_10bin": u["ece_10bin"],
                         "const_accuracy": u["const_accuracy"], "useful_gain": u["useful_gain"],
                         "common_probe_accuracy": r["utility_common_probe"][str(i)]["accuracy"]})
            nat = r["native_vs_audit"]
            ln = (nat.get("leace_native_fit_rows") or {}).get(str(i))
            fc = (nat.get("fare_certificate") or {}).get(str(i)) if isinstance(nat.get("fare_certificate"), dict) else None
            native.append({"seed": r["seed"], "arm": r["arm"], "recipient": i + 1,
                           "native_criterion": "LEACE fit-row cross-covariance" if ln else ("FARE DP certificate" if fc else "none"),
                           "native_status": ln["native_check_fit_rows"]["status"] if ln else (fc or {}).get("status") if fc else "n/a",
                           "native_value": ln["native_check_fit_rows"]["crosscov_max_abs_rel_erased"] if ln else (fc or {}).get("bound") if fc else None,
                           "held_out_max_abs_corr_assessment": nat["held_out_linear"][str(i)]["max_abs_corr_assessment"],
                           "audit_selected_attacker_local": r["primary"][f"v{i + 1}"]["selected"]})
    for fn, rows in (("ACTUAL_TASK_UTILITY.csv", util), ("NATIVE_VS_AUDIT.csv", native)):
        with open(R.PKG / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            for row in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    return util, native


if __name__ == "__main__":
    {"diagnostics": diagnostics, "inner": inner_table, "outer": outer_tables}[sys.argv[1]]()

"""Stage A: recompute every corrected number from committed endpoint arrays (and, for per-seed offset contrasts,
from the saved private predictions; point estimates only). Writes corrections_recomputed.json.
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_analysis_paper_v1/report/stage_a_recompute.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
ODX = WT / "results/combined_output_diagnosis_v1"
OAR = WT / "results/combined_output_aware_removal_v1"
out = {}
rows = lambda p: list(csv.DictReader(open(p)))  # noqa: E731
s5 = {r["id"]: r for r in rows(ODX / "S5_ENDPOINTS.csv")}
fz = s5["S5-FZ-minus-FARE-rep+head"]
out["fare_vs_compression"] = {k: float(fz[k]) for k in ("point", "lower", "upper", "target", "z")} | {"decision": fz["decision"],
    "interval_excludes_zero": float(fz["lower"]) > 0, "meets_target": float(fz["lower"]) > float(fz["target"])}
out["fare_vs_leace"] = {k: float(s5["S5-LEACE-minus-FARE-rep+head"][k]) for k in ("point", "lower", "upper")}
S5 = json.loads((ODX / "S5_SUMMARY.json").read_text())["descriptive"]
out["fare_useful_task_levels"] = {k: round(S5[k]["point"], 4) for k in ("R(A,rep+head)", "R(B,rep+head)", "R(F,rep+head)", "R(FZ,rep+head)", "Acc(A)", "Acc(F)", "const")}
fh = rows(ODX / "FROZEN_HEAD_UTILITY.csv")
fl = [r for r in fh if r["purpose"] == "fair_lending_audit"]
out["fair_lending"] = [{"seed": int(r["seed"]), "constant_prediction": r["is_constant_prediction"], "task_auc_of_scores": float(r["task_auc"]),
                        "log_loss": float(r["log_loss"]), "constant_log_loss": float(r["constant_log_loss"])} for r in fl]
ref = [r for r in fh if r["dataset"] == "adult" and r["purpose"] == "income_prediction" and r["head"] == "refit_head_HEAD__A"]
u2 = [r for r in fh if r["dataset"] == "adult" and r["purpose"] == "income_prediction" and r["head"] == "refit_probe_U2__A"]
out["adult_refit"] = {"HEAD__A_gain_by_seed": [float(r["gain_over_constant"]) for r in ref],
                      "HEAD__A_gain_mean": float(np.mean([float(r["gain_over_constant"]) for r in ref])),
                      "U2_probe_gain_by_seed": [float(r["gain_over_constant"]) for r in u2],
                      "U2_probe_gain_mean": float(np.mean([float(r["gain_over_constant"]) for r in u2]))}
prim = {r["id"]: r for r in rows(ODX / "PRIMARY_ENDPOINTS.csv")}
out["adult_refit"]["U-refit-adult_endpoint"] = float(prim["U-refit-adult"]["point"])
osr = {(r["dataset"], r["key"]): float(r["point"]) for r in rows(ODX / "OUTPUT_SURFACE_RESULTS.csv")}
out["adult_refit"]["RH_offset_free_recovery"] = {k[1]: v for k, v in osr.items() if k[0] == "adult" and "|RH|" in k[1] and "income_prediction|sex" in k[1]}
s3 = rows(ODX / "S3_ENDPOINTS.csv")
fc = [r for r in s3 if r["id"].startswith("S3-FC-")]
out["offset_FC_pairs"] = {"n": len(fc), "pass": sum(r["decision"] == "PASS" for r in fc), "pass_ids": [r["id"] for r in fc if r["decision"] == "PASS"]}
s4 = {r["id"]: r for r in rows(ODX / "S4_ENDPOINTS.csv")}
out["coalition_hard"] = {k: {"point": float(s4[k]["point"]), "lower": float(s4[k]["lower"])} for k in s4 if "hard" in k}
bi = json.loads((ODX / "PRIVATE_BACKUP_INDEX.json").read_text())
out["backup"] = {"index_files": bi["files"], "verification": bi["verification"], "sha256sums": bi["SHA256SUMS_sha256"]}
# per-seed offset contrast (point only) from the saved predictions
import odx.infer as I
from odx import run as R
import oar.study as S
per = {}
for ds, pur, att in (("adult", "income_prediction", "sex"), ("hmda", "underwriting", "race")):
    W = R.world(ds, pur, att)
    a = W["idx"]["assessment"]
    cls = [0, 1] if ds == "adult" else [0, 1, 2]
    D = I.DS(W["row_id"][a], W["unit"][a])
    ids = {}
    for k in S.SEEDS:
        fb = D.recovery(R.uid_of(ds, k, pur, att, "FH", "fullbank"), cls)
        io = D.recovery(R.uid_of(ds, k, pur, att, "FH", "iobank"), cls)
        ids[k] = (D.diff(f"fc{k}", fb, io), fb, io)
    est = I.estimate(D, [x for t in ids.values() for x in t], 2, 1, 1.0)
    per[ds] = {f"s{k}": {"FC": est[v[0]]["point"], "fullbank": est[v[1]]["point"], "iobank": est[v[2]]["point"]} for k, v in ids.items()}
out["FC_per_seed_points"] = per
(Path(__file__).parent / "corrections_recomputed.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

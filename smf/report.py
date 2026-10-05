"""Public aggregate tables and figure from private unit records (reporting only; no fit, selection or inference).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.report [parts...]
Writes GRADIENT_MATCHING.csv, CONTROLLER_ACTIVATION.csv (+ CONTROLLER_SUMMARY.json), CRITIC_TRACKING.csv,
ACTUAL_TASK_UTILITY.csv, LINEAR_DIAGNOSTICS.csv, AUDIT_CONTROLS.json, figures/fig_tradeoff.{pdf,png}.
Activation (per Phase B run): local feedback is ACTIVE iff the applied allocation became asymmetric
(|s1 - s2| > 0.05) at some update, else INACTIVE_OR_ALIAS (common-mode only: identical updates to its twin);
joint feedback is ACTIVE_ASYMMETRIC (|s1 - s2| > 0.05), ACTIVE_COMMON_ONLY (weights moved but stayed equal: only the
local/pair balance changed) or INACTIVE (weights never moved). No-feedback twins report their hypothetical weights.
"""
from __future__ import annotations

import csv
import glob
import json

import numpy as np

from smf import run as R

P = R.PKG


def _w(name, rows, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(P / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _runs(prefix):
    return [json.loads(open(d + "/record.json").read()) for d in sorted(glob.glob(str(R.UNITS / f"{prefix}*")))]


def gradient_matching():
    rows = []
    for r in _runs("run__A__") + _runs("run__B__") + _runs("tplB__"):
        d = r["diag"]
        ep = d["epochs"]
        rat = [[e.get(f"ratio_mean_{i}") for e in ep if e.get(f"ratio_mean_{i}") is not None] for i in (1, 2)]
        rows.append({"unit": r["unit"], "stage": r["stage"], "seed": r["seed"], "base": d["spec"]["base"],
                     "schedule": d["spec"]["schedule"], "feedback": d["spec"].get("feedback"), "rho": d["rho"],
                     "ratio_pre_clip_mean_1": float(np.mean(rat[0])) if rat[0] else None,
                     "ratio_pre_clip_mean_2": float(np.mean(rat[1])) if rat[1] else None,
                     "ratio_pre_clip_rms_mean": float(np.mean([np.sqrt((a * a + b * b) / 2) for a, b in zip(*rat)])) if rat[0] else None,
                     "cos_t_p_mean_1": float(np.mean([e["cos_mean_1"] for e in ep if e.get("cos_mean_1") is not None] or [np.nan])),
                     "cos_t_p_mean_2": float(np.mean([e["cos_mean_2"] for e in ep if e.get("cos_mean_2") is not None] or [np.nan])),
                     "realized_ratio_mean_1": float(np.mean([e["realized_ratio_mean_1"] for e in ep if e.get("realized_ratio_mean_1") is not None] or [np.nan])),
                     "realized_ratio_mean_2": float(np.mean([e["realized_ratio_mean_2"] for e in ep if e.get("realized_ratio_mean_2") is not None] or [np.nan])),
                     "a_max_1": max([e["a_max_1"] for e in ep if e.get("a_max_1") is not None] or [np.nan]),
                     "a_max_2": max([e["a_max_2"] for e in ep if e.get("a_max_2") is not None] or [np.nan]),
                     "kappa_min": min([e.get("kappa_min", 1.0) for e in ep]),
                     "q_norm_mean_1": float(np.mean([e["q_norm_mean_1"] for e in ep if e.get("q_norm_mean_1") is not None] or [np.nan])),
                     "t_norm_mean_1": float(np.mean([e["t_norm_mean_1"] for e in ep if e.get("t_norm_mean_1") is not None] or [np.nan])),
                     "post_clip_enc_norm_mean_1": float(np.mean([e["post_clip_enc_norm_mean_1"] for e in ep if e.get("post_clip_enc_norm_mean_1") is not None] or [np.nan])),
                     "post_clip_enc_norm_mean_2": float(np.mean([e["post_clip_enc_norm_mean_2"] for e in ep if e.get("post_clip_enc_norm_mean_2") is not None] or [np.nan])),
                     "cap_hits_1": d["cap_hits"][0], "cap_hits_2": d["cap_hits"][1], "zero_events_1": d["zero_events"][0],
                     "zero_events_2": d["zero_events"][1], "clip_hits": d["clip_hits"], "encoder_updates": d["encoder_updates"],
                     "update_norm_post_clip_mean": float(np.mean([e["update_norm_post_clip_mean"] for e in ep
                                                                  if e["update_norm_post_clip_mean"] is not None])),
                     "critic_online_updates": d["critic_online_updates"],
                     "critic_matched_extra_updates": d.get("critic_matched_extra_updates", 0),
                     "critic_refit_updates": d.get("critic_refit_updates", 0), "nonfinite": d["nonfinite"],
                     "rescue": d.get("rescue"), "wall_s": d.get("wall_s")})
    for r in _runs("run__raw__"):
        d = r["diag"]
        nt = [n["task"] for n in d["norms"]]
        npn = [n["penalty"] for n in d["norms"]]
        rows.append({"unit": r["unit"], "stage": "raw", "seed": r["seed"], "base": "joint" if r["arm"] == "RAW-J" else "local",
                     "schedule": "ONLINE (raw beta penalty)", "feedback": False, "rho": None, "beta": r["beta"],
                     "raw_penalty_to_task_norm_ratio_mean": float(np.mean(np.array(npn) / np.array(nt))),
                     "clip_hits": d["clip_hits"], "encoder_updates": d["encoder_updates"],
                     "critic_online_updates": d["critic_online_updates"], "nonfinite": d["nonfinite"]})
    _w("GRADIENT_MATCHING.csv", rows)
    return rows


def controller_activation():
    rows, summ = [], {}
    for r in _runs("run__B__"):
        d = r["diag"]
        fb = d["spec"].get("feedback")
        asym = common = 0
        maxdev = 0.0
        for e in d["controller"]:
            w = e["w_after"] if fb else e["w_hypothetical"]
            s = e["allocation"] if fb else [x / np.sqrt((w[0] ** 2 + w[1] ** 2) / 2) for x in w]
            maxdev = max(maxdev, abs(w[0] - 1), abs(w[1] - 1))
            if not e["diagnostic_only"]:
                if abs(s[0] - s[1]) > 0.05:
                    asym += 1
                elif w != [1.0, 1.0]:
                    common += 1
            rows.append({"unit": r["unit"], "seed": r["seed"], "arm": r.get("arm"), "base": d["spec"]["base"],
                         "feedback_applied": fb, "rho": d["rho"], "epoch": e["epoch"], "diagnostic_only": e["diagnostic_only"],
                         "auc_1": e["auc"][0], "auc_2": e["auc"][1], "b_1": e["b"][0], "b_2": e["b"][1],
                         "v_1": e["v"][0], "v_2": e["v"][1], "w_1": w[0], "w_2": w[1], "s_1": s[0], "s_2": s[1],
                         "w_kind": "applied" if fb else "hypothetical (twin)"})
        if fb:
            if d["spec"]["base"] == "local":
                st = "ACTIVE" if asym else "INACTIVE_OR_ALIAS"
            else:
                st = "ACTIVE_ASYMMETRIC" if asym else ("ACTIVE_COMMON_ONLY" if common else "INACTIVE")
        else:
            st = "TWIN (not applied)"
        summ[r["unit"]] = {"status": st, "asymmetric_updates": asym, "common_mode_updates": common,
                           "max_abs_weight_deviation": maxdev}
    _w("CONTROLLER_ACTIVATION.csv", rows)
    (P / "CONTROLLER_SUMMARY.json").write_text(json.dumps(summ, indent=1) + "\n")
    return summ


def critic_tracking():
    rows = []
    for d in sorted(glob.glob(str(R.UNITS / "track__*"))):
        r = json.loads(open(d + "/record.json").read())
        snaps = dict(r["snapshots"])
        if "final_refit_at_theta_T" in r:
            snaps["final_refit_at_theta_T"] = r["final_refit_at_theta_T"]
        for snap, vv in snaps.items():
            for v, x in vv.items():
                for rs in ("calib", "inner"):
                    row = {"seed": r["seed"], "label": r["label"], "run": r["run"], "snapshot": snap, "view": v, "rows": rs,
                           "gap_registered_mean_paired": x[f"gap_registered_{rs}"], "gap_best_of_bank": x[f"gap_best_of_bank_{rs}"],
                           "online_best_ce": x[f"online_best_{rs}"], "fresh_best_ce": x[f"fresh_best_{rs}"],
                           "const_ce": x["const_ce"][rs]}
                    for kind in ("A", "B"):
                        row[f"online_ce_{kind}"] = x["kinds"][kind]["online"][rs]
                        row[f"fresh_ce_{kind}"] = x["kinds"][kind]["fresh"][rs]
                    rows.append(row)
    _w("CRITIC_TRACKING.csv", rows)
    return rows


def utility_and_linear():
    urows, lrows = [], []
    for d in sorted(glob.glob(str(R.UNITS / "outer__s*"))):
        r = json.loads(open(d + "/record.json").read())
        for i, t in enumerate(("income", "occupation_group")):
            u = r["utility_deployed"].get(str(i)) or r["utility_deployed"].get(i)
            pr = r["utility_common_probe"].get(str(i)) or r["utility_common_probe"].get(i) or {}
            urows.append({"seed": r["seed"], "label": r["label"], "task": t, "accuracy": u["accuracy"],
                          "balanced_accuracy_supported": u["balanced_accuracy_supported"], "minority_class": u["minority_class"],
                          "minority_recall": u["minority_recall"], "log_loss": u["log_loss"], "brier": u["brier"],
                          "ece_10bin": u["ece_10bin"], "const_accuracy": u["const_accuracy"], "useful_gain": u["useful_gain"],
                          "probe_accuracy": pr.get("accuracy")})
        flat = {}

        def walk(x, pre=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{pre}{k}.")
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                flat[pre.rstrip(".")] = x
        walk(r.get("linear_diagnostics") or {})
        lrows.append({"seed": r["seed"], "label": r["label"], **flat})
    _w("ACTUAL_TASK_UTILITY.csv", urows)
    _w("LINEAR_DIAGNOSTICS.csv", lrows)


def controls():
    for p in sorted(R.UNITS.glob("controls__s*")):
        (P / "AUDIT_CONTROLS.json").write_text(json.dumps(json.loads((p / "record.json").read_text()), indent=1,
                                                          default=float) + "\n")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = json.loads((R.RUN / "inference.json").read_text())["levels"]
    labs = ["U", "RAW-J", "RAW-L", "J-N", "L-N", "L-F", "J-F", "E", "F", "F0"]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, j, name in ((axes[0], 0, "Income accuracy"), (axes[1], 1, "Occupation-group accuracy")):
        for lab in labs:
            if f"accmean|{lab}|{j}" not in L:
                continue
            x, y = L[f"accmean|{lab}|{j}"]["point"], L[f"Rmean|{lab}|pair"]["point"]
            ax.scatter(x, y, s=28 if lab == "J-F" else 14, color="#d62728" if lab == "J-F" else "k", zorder=3)
            ax.annotate(lab, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
            for k in R.SEEDS:
                ax.scatter(L[f"acc|{k}|{lab}|{j}"]["point"], L[f"R|{k}|{lab}|prim|pair"]["point"], s=4, color="0.6", alpha=0.5)
        ax.axvline(L[f"accmean|U|{j}"]["point"] - 0.01, color="0.5", ls="--", lw=0.7)
        ax.set_xlabel(name + " (new development assessment)")
    axes[0].set_ylabel("Coalition SEX recovery (AUC)")
    fig.tight_layout()
    (P / "figures").mkdir(exist_ok=True)
    fig.savefig(P / "figures" / "fig_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(P / "figures" / "fig_tradeoff.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["gradient_matching", "controller_activation", "critic_tracking", "utility_and_linear",
                            "controls", "figure"]
    for w in what:
        globals()[w]()
    print("done", what)

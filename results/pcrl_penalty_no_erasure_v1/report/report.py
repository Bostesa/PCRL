"""Aggregate public tables and figures from unit records (no per-person data).
    ~/PCRL/.venv/bin/python results/pcrl_penalty_no_erasure_v1/report/report.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pnx import run as R  # noqa: E402

PKG = R.PKG
LABELS = ["U", "E", "F", "F0"] + [f"{a}_b{b}" for a in ("JP", "PN", "LN") for b in ("0.1", "1", "10")]


def rec(path):
    return json.loads((path / "record.json").read_text())


def write(fn, rows):
    with open(PKG / fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def training():
    rows = []
    for d in sorted(R.UNITS.glob("pn__s*")):
        if d.name.endswith(".quarantined") or not R.done(d.name):
            continue
        r = rec(d)
        g = r["diag"]
        last = g["stage_logs"][-1]
        rows.append({"unit": d.name, "arm": r["arm"], "beta": r["beta"], "seed": r["seed"], "wall_s": round(r["wall_s"], 1),
                     "cpu_s": round(r["cpu_s"], 1), "encoder_updates": g["steps"], "protection_steps": g["protection_steps_attempted"],
                     "nonfinite": g["nonfinite"], "rescue": bool(g.get("rescue")), "leace_refits": g["leace_refits"],
                     "final_epoch_R_train": json.dumps({k: round(v, 3) for k, v in last["R"].items()}),
                     "final_guard_losses": json.dumps({k: round(v, 4) for k, v in last["guard"].items()}),
                     "head_C": json.dumps({i: h["selected_C"] for i, h in r["finalize"]["heads"].items()}),
                     "maps": r["maps"]})
    write("TRAINING_DIAGNOSTICS.csv", rows)


def selection():
    S = json.loads((R.RUN / "selection.json").read_text())
    rows = []
    for k, s in S.items():
        for arm, a in s["arms"].items():
            for t in a.get("table", [a]):
                rows.append({"seed": k, "arm": arm, "unit": t.get("unit", a.get("units")), "beta": t.get("beta"),
                             "gates_ok": t.get("gates_ok"), "worst_gate_margin": t.get("worst_gate_margin"),
                             "R_v1": t.get("R_v1"), "R_v2": t.get("R_v2"), "R_pair": t.get("R_pair"),
                             "selected": (t.get("unit") == a.get("unit")) if "table" in a else True, "arm_status": a["status"]})
        rows.append({"seed": k, "arm": "C*", "unit": s["comparator"]["arm"], "beta": None, "gates_ok": None,
                     "worst_gate_margin": None, "R_v1": None, "R_v2": None, "R_pair": None, "selected": True,
                     "arm_status": "candidates: " + ",".join(s["comparator"]["candidates"])})
    write("INNER_SELECTION_TABLE.csv", rows)


def outer():
    util, native = [], []
    for k in (0, 1, 2):
        for lab in LABELS:
            r = rec(R.U(f"outer__s{k}__{lab}"))
            for i, t in enumerate(("income", "occupation_group")):
                u = r["utility_deployed"][str(i)]
                util.append({"seed": k, "label": lab, "task": t, "accuracy": u["accuracy"],
                             "balanced_accuracy_supported": u["balanced_accuracy_supported"], "minority_class": u["minority_class"],
                             "minority_recall": u["minority_recall"], "log_loss": u["log_loss"], "brier": u["brier"],
                             "ece_10bin": u["ece_10bin"], "const_accuracy": u["const_accuracy"], "useful_gain": u["useful_gain"],
                             "common_probe_accuracy": r["utility_common_probe"][str(i)]["accuracy"]})
                nat = r["native_vs_audit"]
                native.append({"seed": k, "label": lab, "recipient": i + 1,
                               "fit_rows_max_abs_rel_crosscov": nat["fit_rows_max_abs_rel_crosscov"][str(i)],
                               "guaranteed_by_construction": "yes (LEACE fitted-moment identity)" if lab in ("E",) or lab.startswith("JP") else "no",
                               "held_out_max_abs_corr_assessment": nat["held_out_linear"][str(i)]["max_abs_corr_assessment"],
                               "audit_local_selected": r["primary"][f"v{i + 1}"]["selected"]})
    write("ACTUAL_TASK_UTILITY.csv", util)
    write("NATIVE_VS_AUDIT.csv", native)


def critic_gap():
    rows = []
    for d in sorted(R.UNITS.glob("critic__pn__*")):
        r = rec(d)
        for w, v in r["views"].items():
            rows.append({"unit": r["unit"], "arm": r["arm"], "beta": r["beta"], "seed": r["seed"], "view": w,
                         "prior_ce": v["prior_ce"], "online_best_ce": v["online_best_ce"], "fresh_def_best_ce": v["fresh_def_best_ce"],
                         "fresh_att_best_ce": v["fresh_att_best_ce"], "slate_ce": v["slate_ce"], "slate_auc": v["slate_auc"],
                         "deployed_view_slate_ce": v["deployed_view_slate_ce"], "deployed_view_slate_auc": v["deployed_view_slate_auc"],
                         "primary_mean_paired_online_minus_fresh_def": v["primary_mean_paired_online_minus_fresh_def"],
                         "sensitivity_thetaT_refit_whitener_primary":
                             v["sensitivity_thetaT_refit_whitener"]["primary_mean_paired_online_minus_fresh_def"],
                         "best_of_bank_online_minus_fresh_def": v["online_best_ce"] - v["fresh_def_best_ce"],
                         "spectrum_condition": v["spectrum"]["condition"], "feature_scale_min": v["feature_scale"]["min"],
                         "feature_scale_max": v["feature_scale"]["max"]})
    write("CRITIC_GAP.csv", rows)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    inf = json.loads((R.RUN / "inference.json").read_text())
    Lv = inf["levels"]
    out = PKG / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    col = {"PN": "#d62728", "LN": "#2ca02c", "JP": "#ff7f0e"}
    for ax, j, task in ((axes[0], 0, "Income accuracy"), (axes[1], 1, "Occupation-group accuracy")):
        for arm in ("PN", "LN", "JP"):
            xs = [Lv[f"accmean|{arm}_b{b}|{j}"]["point"] for b in ("0.1", "1", "10")]
            ys = [Lv[f"Rmean|{arm}_b{b}|pair"]["point"] for b in ("0.1", "1", "10")]
            ax.plot(xs, ys, "-o", color=col[arm], ms=4, label=f"{arm}{' (erased)' if arm == 'JP' else ''} β=0.1→10")
            for k in (0, 1, 2):
                for b in ("0.1", "1", "10"):
                    ax.scatter(Lv[f"acc|{k}|{arm}_b{b}|{j}"]["point"], Lv[f"R|{k}|{arm}_b{b}|prim|pair"]["point"], s=6,
                               color=col[arm], alpha=0.3)
        for lab, mk in (("U", "s"), ("E", "D"), ("F", "^"), ("F0", "v")):
            ax.scatter(Lv[f"accmean|{lab}|{j}"]["point"], Lv[f"Rmean|{lab}|pair"]["point"], marker=mk, s=30, color="k",
                       label=lab, zorder=3)
        accU = Lv[f"accmean|U|{j}"]["point"]
        ax.axvline(accU - 0.01, color="0.5", ls="--", lw=0.7)
        ax.text(accU - 0.0105, 0.52, "U − 1 point", rotation=90, fontsize=6, color="0.4", ha="right", va="bottom")
        ax.axhline(0.5, color="0.7", ls=":", lw=0.6)
        ax.set_xlabel(task + " (assessment)")
    axes[0].set_ylabel("Coalition SEX recovery (AUC)")
    axes[1].legend(frameon=False, fontsize=6, loc="lower left")
    fig.text(0.5, -0.03, "Small points: individual seeds. Lines: seed means over β ∈ {0.1, 1, 10} (descriptive grid; no winner chosen from it).",
             ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_erasure_on_off.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_erasure_on_off.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    what = sys.argv[1:] or ["training", "selection", "outer", "critic_gap", "figures"]
    for w in what:
        globals()[w]()
    print("done", what)

"""Public aggregate tables and figure from private unit records (reporting only; no fit, no selection, no inference).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.report
Writes into results/pcrl_refreshed_guarded_joint_v1/: CRITIC_TRACKING.csv, GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv,
ACTUAL_TASK_UTILITY.csv, LINEAR_DIAGNOSTICS.csv, AUDIT_CONTROLS.json, WHITENING_DIAGNOSTIC.json, ABLATION_CAPPED.csv,
figures/fig_tradeoff.{pdf,png}. Aggregates only (no row-level values).
"""
from __future__ import annotations

import csv
import glob
import json

import numpy as np

from rgj import run as R

P = R.PKG


def _w(name, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0])
    with open(P / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def critic_tracking():
    rows = []
    for d in sorted(glob.glob(str(R.UNITS / "track__s*"))):
        r = json.loads(open(d + "/record.json").read())
        snaps = dict(r["snapshots"])
        if "final_refit_at_theta_T" in r:
            snaps["final_refit_at_theta_T"] = r["final_refit_at_theta_T"]
        for snap, vv in snaps.items():
            for v, x in vv.items():
                for rowset in ("calib", "inner"):
                    row = {"seed": r["seed"], "arm": r["arm"], "selection_status": r["selection_status"],
                           "run": r["run"], "snapshot": snap, "view": v, "rows": rowset,
                           "gap_registered_mean_paired": x[f"gap_registered_{rowset}"],
                           "gap_best_of_bank": x[f"gap_best_of_bank_{rowset}"],
                           "online_best_ce": x[f"online_best_{rowset}"], "fresh_best_ce": x[f"fresh_best_{rowset}"],
                           "const_ce": x["const_ce"][rowset]}
                    for kind in ("A", "B"):
                        row[f"online_ce_{kind}"] = x["kinds"][kind]["online"][rowset]
                        row[f"fresh_ce_{kind}"] = x["kinds"][kind]["fresh"][rowset]
                    rows.append(row)
    _w("CRITIC_TRACKING.csv", rows)
    return rows


def gradient_and_weights():
    rows = []
    for d in sorted(glob.glob(str(R.UNITS / "run__*"))):
        r = json.loads(open(d + "/record.json").read())
        g = r["diag"]
        nt = [n["task"] for n in g["norms"]]
        npn = [n["penalty"] for n in g["norms"]]
        lam = g.get("lambda_trace") or []
        base = g["coefficients_base"]
        b = r["beta"]
        last_cal = g["refits"][-1]["calib"] if g["refits"] else None
        bz = g.get("block_end_max_abs_z") or []
        row = {"unit": r["unit"], "stage": r["stage"], "seed": r["seed"], "arm": r["arm"], "beta": b, "tkind": r["tkind"],
               "wall_s": r["wall_s"], "cpu_s": r["cpu_s"], "encoder_updates": g["encoder_updates"],
               "critic_online_updates": g["critic_online_updates"], "critic_refit_updates": g["critic_refit_updates"],
               "refits": len(g["refits"]), "early_stops": g["early_stops"], "restart_chosen": g["restart_chosen"],
               "continued_chosen": g["continued_chosen"], "clip_hits": g["clip_hits"], "nonfinite": g["nonfinite"],
               "rescue": bool(g.get("rescue")), "task_norm_mean": float(np.mean(nt)), "penalty_norm_mean": float(np.mean(npn)),
               "penalty_norm_max": float(np.max(npn)), "penalty_to_task_ratio_mean": float(np.mean(np.array(npn) / np.array(nt))),
               "base_w_v1": base["v1"], "base_w_v2": base["v2"], "base_w_pair": base["pair"],
               "budget_c1": (g.get("c") or [None, None])[0], "budget_c2": (g.get("c") or [None, None])[1],
               "lambda1_final": lam[-1]["lambda"][0] if lam else 0.0, "lambda2_final": lam[-1]["lambda"][1] if lam else 0.0,
               "lambda_max_over_base": (max(max(x["lambda"]) for x in lam) / (b / 3 if r["arm"].startswith("J") else b / 2)
                                        if lam and b > 0 else 0.0),
               "lambda_trace": ";".join(f"e{x['epoch']}:{x['lambda'][0]:.4g}/{x['lambda'][1]:.4g}" for x in lam),
               "R_minus_c_trace": ";".join(f"e{x['epoch']}:{x['R_calib'][0] - x['c'][0]:+.4f}/{x['R_calib'][1] - x['c'][1]:+.4f}"
                                           for x in lam),
               "final_refit_calib_R_v1": last_cal["v1"]["R"] if last_cal else None,
               "final_refit_calib_R_v2": last_cal["v2"]["R"] if last_cal else None,
               "final_refit_calib_R_pair": last_cal["pair"]["R"] if last_cal else None,
               "block_end_max_abs_z_max": max((max(x[v] for v in ("v1", "v2", "pair")) for x in bz), default=None)}
        rows.append(row)
    _w("GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv", rows)
    return rows


def utility_and_linear():
    urows, lrows = [], []
    for d in sorted(glob.glob(str(R.UNITS / "outer__s*"))):
        r = json.loads(open(d + "/record.json").read())
        for i, t in enumerate(("income", "occupation_group")):
            u = r["utility_deployed"][str(i)] if str(i) in r["utility_deployed"] else r["utility_deployed"][i]
            urows.append({"seed": r["seed"], "label": r["label"], "task": t, "accuracy": u["accuracy"],
                          "balanced_accuracy_supported": u["balanced_accuracy_supported"],
                          "minority_class": u["minority_class"], "minority_recall": u["minority_recall"],
                          "log_loss": u["log_loss"], "brier": u["brier"], "ece_10bin": u["ece_10bin"],
                          "const_accuracy": u["const_accuracy"], "useful_gain": u["useful_gain"],
                          "probe_accuracy": (r["utility_common_probe"].get(str(i)) or r["utility_common_probe"].get(i) or {}).get("accuracy")})
        ld = r.get("linear_diagnostics") or {}
        flat = {}

        def walk(x, pre=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{pre}{k}." if pre or k else k)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                flat[pre.rstrip(".")] = x
        walk(ld)
        lrows.append({"seed": r["seed"], "label": r["label"], **flat})
    _w("ACTUAL_TASK_UTILITY.csv", urows)
    if lrows:
        cols = sorted({k for r in lrows for k in r}, key=lambda c: (c not in ("seed", "label"), c))
        _w("LINEAR_DIAGNOSTICS.csv", lrows, cols)
    return urows


def controls_and_whitening():
    for src, dst in (("controls__s0", "AUDIT_CONTROLS.json"), ("whiten__diag", "WHITENING_DIAGNOSTIC.json")):
        p = R.UNITS / src / "record.json"
        if p.exists():
            (P / dst).write_text(json.dumps(json.loads(p.read_text()), indent=1, default=float) + "\n")


def ablation():
    rows = []
    for k in R.SEEDS:
        for tag, sfx in (("floored (main J-G)", ""), ("capped (registered ablation)", "__capped")):
            for e in R.T.HP["checkpoints"]:
                n = R.ck_name("C", k, "J-G", 0.1, e) + sfx
                if not R.done(f"inner__{n}"):
                    continue
                r = R.rec(f"inner__{n}")
                a = r["recovery"]["auc"]
                rows.append({"seed": k, "transform": tag, "beta": 0.1, "epoch": e, "unit": n, "auc_v1": a["v1"],
                             "auc_v2": a["v2"], "auc_pair": a["pair"], "acc_income": r["utility"]["0"]["acc"],
                             "acc_occ": r["utility"]["1"]["acc"]})
    _w("ABLATION_CAPPED.csv", rows)
    return rows


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inf = json.loads((R.RUN / "inference.json").read_text())
    L = inf["levels"]
    labs = ["U", "L-R", "L-O", "L-G", "J-R", "J-O", "J-G", "E", "F", "F0"]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, j, name in ((axes[0], 0, "Income accuracy"), (axes[1], 1, "Occupation-group accuracy")):
        for lab in labs:
            if f"accmean|{lab}|{j}" not in L:
                continue
            x, y = L[f"accmean|{lab}|{j}"]["point"], L[f"Rmean|{lab}|pair"]["point"]
            ax.scatter(x, y, s=28 if lab == "J-G" else 14, color="#d62728" if lab == "J-G" else "k", zorder=3)
            ax.annotate(lab, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
            for k in R.SEEDS:
                ax.scatter(L[f"acc|{k}|{lab}|{j}"]["point"], L[f"R|{k}|{lab}|prim|pair"]["point"], s=4, color="0.6",
                           alpha=0.5)
        ax.axvline(L[f"accmean|U|{j}"]["point"] - 0.01, color="0.5", ls="--", lw=0.7)
        ax.set_xlabel(name + " (development assessment)")
    axes[0].set_ylabel("Coalition SEX recovery (AUC)")
    fig.text(0.5, -0.04, "Seed means (labelled) and individual seeds (grey). J-G is the descriptive closest checkpoint "
             "(no feasible nominee on any seed). Dashed: U minus 1 point.", ha="center", fontsize=6.5)
    fig.tight_layout()
    (P / "figures").mkdir(exist_ok=True)
    fig.savefig(P / "figures" / "fig_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(P / "figures" / "fig_tradeoff.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["critic_tracking", "gradient_and_weights", "utility_and_linear", "controls_and_whitening",
                            "ablation", "figure"]
    for w in what:
        globals()[w]()
    print("done", what)

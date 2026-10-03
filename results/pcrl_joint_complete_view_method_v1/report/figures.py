"""Figures from aggregate outputs (inference.json levels + ACTUAL_TASK_UTILITY.csv). No per-person data.
    ~/PCRL/.venv/bin/python results/pcrl_joint_complete_view_method_v1/report/figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parents[1]
INF = Path.home() / "PCRL_eval_cache_private/jcv_v1/run/inference.json"
OUT = PKG / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
ARMS = ["U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"]
COL = {"U": "#333333", "E": "#1f77b4", "L": "#2ca02c", "J": "#d62728", "JP": "#ff7f0e", "S12": "#9467bd",
       "S21": "#8c564b", "F": "#e377c2", "F0": "#7f7f7f"}


def diagram():
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    kw = dict(ha="center", va="center", fontsize=6.5)
    box = dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec="0.3")
    rel = dict(boxstyle="round,pad=0.35", fc="#e8f0fb", ec="0.3")
    rec = dict(boxstyle="round,pad=0.35", fc="#fff6d5", ec="0.3")
    ax.text(0.055, 0.5, "permitted\ninputs X\n(83 columns;\nno sex, race,\nincome, occupation)", bbox=box, **kw)
    xs = [0.235, 0.39, 0.55, 0.735]
    for y, i, task in ((0.85, 1, "income"), (0.15, 2, "occupation group")):
        ax.text(xs[0], y, f"g{i}: MLP 64-64-16", bbox=box, **kw)
        ax.text(xs[1], y, f"official LEACE {i}\n(vs SEX, final refit)", bbox=rel, **kw)
        ax.text(xs[2], y, f"affine head h{i}\ncentred logits", bbox=rel, **kw)
        ax.text(xs[3], y, f"recipient {i}: {task}\nv{i} = [r{i}, logits {i}]", bbox=rec, **kw)
        for x0, x1 in ((0.12, 0.17), (0.30, 0.325), (0.455, 0.495), (0.605, 0.65)):
            ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.93, 0.5, "coalition\n[v1, v2]", bbox=dict(boxstyle="round,pad=0.35", fc="#fde0e0", ec="0.3"), **kw)
    ax.annotate("", (0.915, 0.6), (0.80, 0.78), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", (0.915, 0.4), (0.80, 0.22), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.47, 0.5, "training only (never released): critics on v1, v2 and, for J/JP/S, on [v1, v2];\n"
                       "privacy step projected off active task-loss guards (L, J, S) or plain penalty (JP)",
            ha="center", va="center", fontsize=6, style="italic", color="0.25")
    fig.savefig(OUT / "fig_what_runs.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_what_runs.png", dpi=150, bbox_inches="tight")


def tradeoff():
    inf = json.loads(INF.read_text())
    L = inf["levels"]
    util = list(csv.DictReader(open(PKG / "ACTUAL_TASK_UTILITY.csv")))
    sl = json.loads((PKG / "SELECTION_LOCK.json").read_text())
    gain = {}
    for r in util:
        gain.setdefault((r["arm"], int(r["seed"])), {})[r["task"]] = float(r["useful_gain"])
    gU = {(k, t): gain[("U", k)][t] for k in (0, 1, 2) for t in ("income", "occupation_group")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, view, title in ((axes[0], "pair", "Coalition [v1, v2]"), (axes[1], "local", "Worse local view")):
        for arm in ARMS:
            xs, ys = [], []
            for k in (0, 1, 2):
                ret = min(gain[(arm, k)][t] / gU[(k, t)] for t in ("income", "occupation_group"))
                if view == "pair":
                    y = L[f"R|{k}|{arm}|prim|pair"]["point"]
                else:
                    y = max(L[f"R|{k}|{arm}|prim|v1"]["point"], L[f"R|{k}|{arm}|prim|v2"]["point"])
                xs.append(ret)
                ys.append(y)
                infeas = sl["seeds"][str(k)]["arms"][arm]["status"] != "NOMINEE"
                ax.scatter(ret, y, s=14, color=COL[arm], marker="x" if infeas else "o", alpha=0.75, lw=1)
            ax.scatter(np.mean(xs), np.mean(ys), s=40, color=COL[arm], edgecolor="k", lw=0.5, label=arm, zorder=3)
        ax.axvline(0.8, color="0.5", lw=0.7, ls="--")
        ax.text(0.805, 0.51, "80% useful-gain floor", fontsize=6, color="0.4", rotation=90, va="bottom")
        ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("Retained useful gain vs U (worse of the two tasks)")
        ax.set_title(title)
    axes[0].set_ylabel("SEX recovery on assessment (AUC)")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=3, loc="lower left")
    fig.text(0.5, -0.04, "o: per-seed nominee   x: per-seed INFEASIBLE closest-utility configuration   large: seed mean",
             ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_tradeoff.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    diagram()
    if INF.exists():
        tradeoff()
    print(sorted(p.name for p in OUT.iterdir()))

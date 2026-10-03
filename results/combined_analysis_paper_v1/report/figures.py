"""Manuscript figures from committed aggregate tables only (no private data).
    ~/PCRL/.venv/bin/python results/combined_analysis_paper_v1/report/figures.py
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WT = Path(__file__).resolve().parents[3]
PKG = WT / "results/combined_analysis_paper_v1"
ODX = WT / "results/combined_output_diagnosis_v1"
OUT = WT / "papers/combined_empirical_v1/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
ARM_LABEL = {"A": "A untreated", "B": "B LEACE", "F": "F FARE", "F0": "F0 zero-fairness"}
COL = {"A": "#444444", "B": "#1f77b4", "F": "#d62728", "F0": "#ff7f0e"}


def fig_useful_head():
    inf = json.loads((Path.home() / "PCRL_eval_cache_private/cap_v1/run/inference.json").read_text())
    gain = {(r["arm"], r["seed"]): r["gain_over_constant"] for r in inf["utility"]}
    lv, ps = inf["levels"], inf["per_seed"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, surf, title in ((axes[0], "centred", "Output-only: centred margin (= logits, probabilities)"),
                            (axes[1], "hard", "Output-only: hard decision")):
        for arm in ("A", "B", "F", "F0"):
            gs = [gain[arm, k] for k in (0, 1, 2)]
            rs = [ps[f"R|{arm}|out|{surf}|s{k}"] for k in (0, 1, 2)]
            ax.scatter(gs, rs, s=9, color=COL[arm], alpha=0.35, lw=0)
            r = lv[f"R|{arm}|out|{surf}"]
            ax.errorbar(np.mean(gs), r["point"], yerr=1.645 * r["se"], fmt="o", color=COL[arm], label=ARM_LABEL[arm],
                        capsize=2, ms=4.5)
        ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("Deployed head: accuracy gain over constant")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Sex recovery (macro AUC)")
    axes[0].set_ylim(0.48, 0.86)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_useful_head.pdf")


def fig_formats_and_pairs():
    osr = {(r["dataset"], r["key"]): float(r["point"]) for r in csv.DictReader(open(ODX / "OUTPUT_SURFACE_RESULTS.csv"))}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    forms = ["full", "prob", "centred", "hard"]
    x = np.arange(len(forms))
    for i, (ds, key, lab) in enumerate((("adult", "income_prediction|sex", "Adult income / sex"), ("hmda", "underwriting|race", "HMDA underwriting / race"))):
        ax.bar(x + (i - 0.5) * 0.38, [osr[(ds, f"R|{key}|FH|{f}")] for f in forms], 0.38, label=lab, color=["#555555", "#aaaaaa"][i])
    ax.set_xticks(x, ["full\nlogits", "proba-\nbilities", "centred\n(no offset)", "hard\ndecision"])
    ax.set_ylim(0.45, 0.88)
    ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
    ax.set_ylabel("Recovery (macro AUC)")
    ax.set_title("Frozen PCRL heads: one recipient")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    # S4 (MULTI_RECIPIENT_RESULTS.md / S4_ENDPOINTS.csv): Adult race; income, employment, pair
    s4 = {"full": (0.845, 0.775, 0.917), "centred": (0.731, 0.792, 0.883), "hard": (0.546, 0.576, 0.611)}
    x = np.arange(3)
    for j, (lab, c) in enumerate((("income alone", "#bbbbbb"), ("employment alone", "#888888"), ("pair", "#222222"))):
        ax.bar(x + (j - 1) * 0.27, [s4[f][j] for f in ("full", "centred", "hard")], 0.27, label=lab, color=c)
    ax.set_xticks(x, ["full logits", "centred", "hard decision"])
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
    ax.set_title("Adult race: two recipients together")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "fig_formats_pairs.pdf")


if __name__ == "__main__":
    fig_formats_and_pairs()
    if (Path.home() / "PCRL_eval_cache_private/cap_v1/run/inference.json").exists():
        fig_useful_head()
    print(sorted(p.name for p in OUT.iterdir()))

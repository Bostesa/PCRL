"""Figures (ordinary matplotlib). Fig 1: output-surface recovery beside frozen-head task gain, per encoder seed.
Fig 2 (if stage 5 ran): FARE / LEACE / zero-fairness / untreated task gain vs complete-contract recovery."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import infer as I, run as R
PKG = WT / "results/combined_output_diagnosis_v1"
(PKG / "figures").mkdir(exist_ok=True)


def macro(uid, cls):
    p = I.load_preds(uid)
    return float(np.mean([np.mean([roc_auc_score(p["y_s"] == c, p[f"P__NL__as{a}"][:, c]) for c in cls]) for a in (0, 1, 2)]))


util = list(csv.DictReader(open(PKG / "FROZEN_HEAD_UTILITY.csv")))
cells = [("adult", "income_prediction", "sex", [0, 1]), ("hmda", "underwriting", "race", [0, 1, 2])]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [3, 1.2, 3, 1.2]})
surf = [("fullbank", "full logits\n(bank)"), ("iobank", "offset removed\n(centred / prob)"), ("prob", "probabilities"), ("hard", "decision")]
for ci, (ds, pur, att, cls) in enumerate(cells):
    ax = axes[2 * ci]
    for k, mk in zip((0, 1, 2), ("o", "s", "^")):
        ys = [macro(R.uid_of(ds, k, pur, att, "FH", s), cls) for s, _ in surf]
        ax.plot(range(len(surf)), ys, marker=mk, label=f"encoder seed {k}")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xticks(range(len(surf))); ax.set_xticklabels([l for _, l in surf], fontsize=8)
    ax.set_ylabel("attacker macro AUC (recovery of %s)" % att); ax.set_title(f"{ds} {pur}: what the output reveals", fontsize=9)
    ax.set_ylim(0.45, 1.0); ax.legend(fontsize=7)
    ax2 = axes[2 * ci + 1]
    g = [float(r["gain_over_constant"]) for r in util if r["dataset"] == ds and r["purpose"] == pur and r["head"] == "frozen"]
    gr = [float(r["gain_over_constant"]) for r in util if r["dataset"] == ds and r["purpose"] == pur and r["head"] == "refit_probe_U2__A"]
    ax2.bar(np.arange(3) - 0.18, g, 0.36, label="frozen head")
    ax2.bar(np.arange(3) + 0.18, gr, 0.36, label="refitted probe")
    ax2.axhline(0.01, color="red", lw=0.8, ls="--", label="0.01 target")
    ax2.set_xticks(range(3)); ax2.set_xticklabels([f"s{k}" for k in range(3)], fontsize=8)
    ax2.set_ylabel("accuracy gain over constant", fontsize=8); ax2.set_title("task usefulness", fontsize=9); ax2.legend(fontsize=6)
fig.suptitle("Fig. 1. Recovery by output surface (left of each pair; AUC, outputs-only recipient, mean over 3 attacker seeds) "
             "beside the frozen head's accuracy gain over an attacker_fit-majority constant (right; accuracy units). "
             "Assessment rows; development data.", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PKG / "figures/fig1_output_surfaces_vs_gain.png", dpi=150)
print("fig1 written")

# ---------------- Fig 2: FARE / LEACE / zero-fairness / untreated, task gain vs complete-contract recovery
rep = list(csv.DictReader(open(PKG / "FARE_REPLAY_EXISTING.csv")))
s5 = json.loads((PKG / "S5_SUMMARY.json").read_text()) if (PKG / "S5_SUMMARY.json").exists() else None
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
mk = {"A": ("o", "untreated"), "B": ("s", "target LEACE"), "F": ("*", "FARE nominee"), "FZ": ("^", "zero-fairness twin (same tree budget)")}
for ax, ds in zip(axes[:2], ("adult", "hmda")):
    for arm, (m, lab) in mk.items():
        rr = [r for r in rep if r["dataset"] == ds and r["arm"] == arm]
        ax.scatter([float(r["gain_over_constant"]) for r in rr], [float(r["complete_contract_recovery_macro_auc"]) for r in rr],
                   marker=m, s=80 if arm == "F" else 40, label=lab)
        for r in rr:
            if r["constant_release"] == "True":
                ax.annotate("1-cell (constant) release", (float(r["gain_over_constant"]), float(r["complete_contract_recovery_macro_auc"])),
                            fontsize=7, xytext=(5, 10), textcoords="offset points", arrowprops={"arrowstyle": "-"})
    ax.set_xlabel("U2 probe accuracy gain over constant (accuracy units)", fontsize=8)
    ax.set_ylabel("recovery from features + own head (macro AUC)", fontsize=8)
    ax.set_title(f"{ds} ({'sex' if ds == 'adult' else 'race, supported groups 0-2'}): one point per encoder seed", fontsize=9)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":"); ax.legend(fontsize=6)
ax = axes[2]
if s5 and s5["descriptive"]:
    d = s5["descriptive"]
    import itertools
    for arm, key in (("A", "R(A,rep+head)"), ("B", "R(B,rep+head)"), ("F", "R(F,rep+head)"), ("FZ", "R(FZ,rep+head)")):
        ax.bar(mk[arm][1].split(" (")[0], d[key]["point"], yerr=1.6448536 * d[key]["se"], capsize=3)
    ax.set_ylim(0.45, 0.8); ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("recovery of age_group (macro AUC; bar = 90% normal interval)", fontsize=8)
    ax.set_title("stage 5: Adult employment / age_group\ntask gain: untreated %.3f, FARE %.3f" % (d["Acc(A)"]["point"] - d["const"]["point"], d["Acc(F)"]["point"] - d["const"]["point"]), fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
fig.suptitle("Fig. 2. Task gain (U2 refitted-probe accuracy minus attacker_fit-majority constant, accuracy units) versus "
             "complete-contract recovery (features + head fitted only on them; AUC). HMDA race groups 3-4 not estimable. Development data.", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(PKG / "figures/fig2_fare_leace_compression.png", dpi=150)
print("fig2 written")

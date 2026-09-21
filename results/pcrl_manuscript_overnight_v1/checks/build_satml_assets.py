#!/usr/bin/env python3
"""Build the Stage B (capacity diagnostic) assets for the SaTML paper from pinned evidence.

Reads committed evidence through the git object store at the pinned diagnostic commit. Arithmetic
only: no ACS model is fitted, no pool is read, 2016 is untouched. Figures are sized for an IEEEtran
column. Labels carry the development/diagnostic status on the artwork itself.
"""
import hashlib, json, os, statistics as st, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHA = "cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_stochastic_channel_v1/"
OUT = os.path.join(ROOT, "papers/pcrl_satml_v1")
SEEDS = ("0", "1", "2")
sources, manifest = {}, {}


def show(p):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{p}"], check=True, capture_output=True).stdout
    sources[p] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def emit(name, body):
    path = os.path.join(OUT, "tables", name)
    open(path, "w").write(body if body.endswith("\n") else body + "\n")
    manifest["tables/" + name] = hashlib.sha256(open(path, "rb").read()).hexdigest()


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "figures", f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        manifest[f"figures/{stem}.{ext}"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    plt.close(fig)


sb = json.loads(show(RD + "STAGE_B_RESULTS.json"))
g2 = json.loads(show(RD + "gates/G2.json"))
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

rows = {}
for res in ("64", "256"):
    uw = [sb[res][s]["B2_deployable"]["residence_advantage_nats_code_only"] for s in SEEDS]
    pw = [sb[res][s]["B2_deployable"]["residence_advantage_nats_person_weighted"] for s in SEEDS]
    se = [sb[res][s]["B2_interval_code_only"]["bootstrap_se"] for s in SEEDS]
    rows[res] = {"uw": uw, "pw": pw, "se": se}

# ---------------------------------------------------------------- table: the capacity diagnostic
L = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
     r"code size & anchor & unweighted & person-weighted & paired SE \\", r"\midrule"]
for res in ("64", "256"):
    for i, s in enumerate(SEEDS):
        lab = (r"$|T|=%s$" % res) if i == 0 else ""
        L.append(f"{lab} & {s} & ${rows[res]['uw'][i]:+.5f}$ & ${rows[res]['pw'][i]:+.5f}$ "
                 f"& {rows[res]['se'][i]:.5f} " + r"\\")
    L.append(r"\addlinespace[1pt]")
    L.append(r"\multicolumn{2}{@{}l}{\quad anchor mean} & $%+.5f$ & $%+.5f$ & \\" %
             (st.mean(rows[res]["uw"]), st.mean(rows[res]["pw"])))
    if res == "64":
        L.append(r"\midrule")
emit("study7_capacity.tex", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- table: subsumption, as measured
sub = g2["subsumption_check"]["validation_log_loss_nats"]
L = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
     r"anchor & prior$^\dagger$ & $T$ only$^\dagger$ & $J$ only & $J+T$ \\", r"\midrule"]
for s in SEEDS:
    L.append(f"{s} & {sub[s]['prior']:.5f} & {sub[s]['T_only']:.5f} & "
             f"\\textbf{{{sub[s]['J_only']:.5f}}} & {sub[s]['J_plus_T']:.5f} " + r"\\")
emit("study7_subsumption.tex", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, MUTE, ACC, WARN = "#1b1b1b", "#8a8a8a", "#2a6f97", "#a4303f"
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
for ax, res in zip(axes, ("64", "256")):
    x = range(3)
    for dx, key, col, mk, lab in ((-0.12, "uw", INK, "o", "unweighted"),
                                  (0.12, "pw", ACC, "s", "person-weighted")):
        ax.errorbar([i + dx for i in x], rows[res][key], yerr=rows[res]["se"], fmt=mk, ms=3.2,
                    color=col, ecolor=col, elinewidth=0.9, capsize=1.8, lw=0, label=lab)
    ax.axhline(0, color=WARN, lw=0.8, ls="--")
    ax.set_title(r"$|T| = %s$ codes" % res, fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"a{ i }" for i in x])
    ax.set_xlabel("anchor")
axes[0].set_ylabel("residence advantage\nover same-host $J$ (nats)")
axes[0].legend(frameon=False, fontsize=6, loc="lower left")
fig.suptitle("Capacity diagnostic (development; validation split)", fontsize=7.5, y=1.04)
savefig(fig, "study7_capacity")

json.dump({"pinned_commit": SHA, "sources_sha256": sources, "assets": manifest,
           "recomputed": {r: {"seed_mean_unweighted": st.mean(rows[r]["uw"]),
                              "seed_mean_person_weighted": st.mean(rows[r]["pw"]),
                              "seeds_positive_unweighted": sum(v > 0 for v in rows[r]["uw"]),
                              "seeds_positive_person_weighted": sum(v > 0 for v in rows[r]["pw"])}
                          for r in rows}},
          open(os.path.join(ROOT, "results/pcrl_manuscript_overnight_v1/SATML_ASSET_HASHES.json"), "w"),
          indent=1)
print(f"{len(manifest)} assets built from {SHA[:9]}")

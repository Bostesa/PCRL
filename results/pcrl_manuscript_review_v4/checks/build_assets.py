#!/usr/bin/env python3
"""Generate every v4 table and figure from committed machine-readable evidence.
Pending experiments get PENDING cells and empty axes - never an invented curve."""
import csv, json, collections, statistics, os, sys, hashlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EV = "/Users/nathansamson/PCRL-terminal-1-adversarial/results/pcrl_direct_adversarial_v1"
OUT = "/Users/nathansamson/PCRL-terminal-2-manuscript/papers/pcrl_manuscript_v4"
P = lambda n: os.path.join(EV, n)
T = lambda n: os.path.join(OUT, "tables", n)
F = lambda n: os.path.join(OUT, "figures", n)
rd = lambda n: list(csv.DictReader(open(P(n))))
esc = lambda s: str(s).replace("_", r"\_")
tt  = lambda s: r"\texttt{" + esc(s) + "}"
manifest = {}
def emit(path, body):
    open(path, "w").write(body if body.endswith("\n") else body + "\n")
    manifest[os.path.relpath(path, OUT)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
def savefig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(F(f"{stem}.{ext}"), bbox_inches="tight", dpi=200)
        manifest[f"figures/{stem}.{ext}"] = hashlib.sha256(open(F(f"{stem}.{ext}"), "rb").read()).hexdigest()
    plt.close(fig)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, MUTE, ACC, WARN = "#1b1b1b", "#8a8a8a", "#2a6f97", "#a4303f"

# ---------------------------------------------------------------- panels
per = rd("PER_SEED.csv")
SCOPE, BUDGET = "kernel_expanded_independent", "360"
COLS = ["A/SEX", "AB/SEX", "A/RAC1P", "AB/RAC1P", "residence"]
def panel(rows, scope, weight, budget=None):
    d = collections.defaultdict(dict)
    for r in rows:
        if r.get("scope") != scope or r["weight"] != weight: continue
        if budget and r.get("budget") != budget: continue
        if r["kind"] == "additional_recovery" and r["endpoint"] in COLS[:4]:
            d[(r["condition"], r["endpoint"])][r["seed"]] = float(r["value"])
        elif r["kind"] == "utility_gain_vs_H" and r["endpoint"] == "same_residence":
            d[(r["condition"], "residence")][r["seed"]] = float(r["value"])
    return {k: statistics.mean(v.values()) for k, v in d.items() if len(v) == 3}
p18 = panel(per, SCOPE, "unweighted", BUDGET)
p18w = panel(per, SCOPE, "person_weighted", BUDGET)
ex = rd("EXPLORATORY_2017.csv")
p17 = panel(ex, "common_fresh", "unweighted")

# ---------------------------------------------------------------- T1 frontier
ORDER = [("A0","no protection"),("J","frozen neural channel, strongest developed"),
         ("leace_A0","closed-form eraser on $A_0$"),("splince_A0","closed-form eraser on $A_0$"),
         ("optnet16_C1","published-method adaptation"),("spectral_C1","Study 3, previous line"),
         ("dax8_C1_b300","Study 4, best new arm"),("dax16_C1_b100","Study 4, coalition width 16"),
         ("dax16_L2_b300","Study 4, strengthened local"),("leace_dax8_none","erasure at rank 0")]
def frontier_table(p, cap_cols):
    L = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
         r"condition & role & $A$/SEX & $AB$/SEX & $A$/RAC & $AB$/RAC & residence \\", r"\midrule"]
    for c, note in ORDER:
        if (c, "residence") not in p: continue
        v = [p[(c, e)] for e in COLS]
        bold = r"\bfseries " if c in ("J", "dax8_C1_b300") else ""
        L.append(bold + tt(c) + " & " + note + " & " + " & ".join(f"${x:+.4f}$" for x in v) + r" \\")
        if c == "spectral_C1": L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)
emit(T("study4_frontier.tex"), frontier_table(p18, COLS))
emit(T("study4_frontier_2017.tex"), frontier_table(p17, COLS))
emit(T("study4_frontier_note.tex"),
     r"\newcommand{\noteStudyFourFrontier}{Seed means over three anchor seeds, matched-exposure scope "
     r"\texttt{kernel\_expanded\_independent}, audit budget 360, unweighted. Recovery is additional "
     r"recovery over $H$; residence is utility gain over $H$. Lower recovery is less disclosure, higher "
     r"residence is more capability. Recomputed cell by cell from \texttt{PER\_SEED.csv}; all 50 cells "
     r"agree with the study's own table to within $5\times10^{-5}$.}")
emit(T("study4_frontier_2017_note.tex"),
     r"\newcommand{\noteStudyFourFrontierSeventeen}{\textbf{Exploratory cross-year development.} The 2017 "
     r"\texttt{final\_evaluation} partition is spent; no intervals are computed. Computing them would not "
     r"be forbidden---they would be conditional descriptive sampling uncertainty---but they would not "
     r"restore confirmatory status or account for adaptive selection. The original locked 2017 transport "
     r"result keeps its historical status and is neither restated nor overwritten.}")

# ---------------------------------------------------------------- T2 coalition, both levels
iv = rd("PAIRED_INTERVALS.csv")
coal = [r for r in iv if r["comparison_family"] == "coalition_vs_local"]
PAIRS = [("dax16_C1_b100","dax16_L1_b100","width 16, weak local control"),
         ("dax16_C1_b100","dax16_L2_b100","width 16, strength-matched local control"),
         ("dax8_C1_b100","dax8_L1_b100","width 8, weak local control"),
         ("dax8_C1_b100","dax8_L2_b100","width 8, strength-matched local control")]
L = [r"\begin{tabular}{l l cc cc l}", r"\toprule",
     r"\multicolumn{2}{l}{contrast} & \multicolumn{2}{c}{sensitive endpoints better, of 4}"
     r" & \multicolumn{2}{c}{residence difference} & \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"$C_1$ arm & local control & family-adj. & cand.-wide & estimate & family-adj.\ interval & cand.-wide \\",
     r"\midrule"]
for l_, r_, note in PAIRS:
    sub = [x for x in coal if x["left"] == l_ and x["right"] == r_ and x["weight"] == "unweighted"]
    rec = [x for x in sub if x["endpoint"].startswith("recovery")]
    adj = sum(1 for x in rec if x["significantly_better"] == "True")
    cw  = sum(1 for x in rec if x["candidate_wide_better"] == "True")
    res = [x for x in sub if x["endpoint"] == "utility/same_residence"][0]
    lo, hi = float(res["adjusted_low"]), float(res["adjusted_high"])
    cwsig = "worse" if res["candidate_wide_worse"] == "True" else r"n.s."
    L.append(f"{tt(l_)} & {tt(r_)} & {adj} & {cw} & ${float(res['estimate']):+.5f}$ & "
             f"$[{lo:+.5f},\\,{hi:+.5f}]$ & {cwsig} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
emit(T("study4_coalition.tex"), "\n".join(L))
emit(T("study4_coalition_note.tex"),
     r"\newcommand{\noteStudyFourCoalition}{Unweighted, $\beta=1.0$, matched-exposure scope. "
     r"\emph{Family-adjusted} is the single-step max-$|t|$ interval inside the "
     r"\texttt{coalition\_vs\_local} family; \emph{candidate-wide} is the simultaneous studentized "
     r"Bonferroni bound over the full 850-row family, which is the standard the frontier comparison "
     r"in Table~\ref{tab:s4frontier} is held to. The study's own report gives this family at the "
     r"family-adjusted level only and omits the fourth row, which is the width-8 comparison against "
     r"the strength-matched control and is null throughout. The coalition benefit is robust against "
     r"the weak control $L_1$ and does not survive against $L_2$ under the candidate-wide standard. "
     r"$L_2$ matches $C_1$'s nominal total group weight, not its gradients and not its difficulty.}")

# ---------------------------------------------------------------- T3 accounting
mech = json.load(open(P("MECHANISM.json"))); ca = mech["channel_accounting"]
be = rd("MECH_BETA_EQUIVALENCE.csv")
ident = [r for r in be if r["identical_to_no_protection"] == "True"]
bb = collections.Counter(r["beta"] for r in ident)
rows = [("nominal transform fits planned", 126, "the registry, not a count of systems"),
        ("released interfaces", ca["released_units_total"],
         f"{126-ca['released_units_total']} SPLINCE width-8 arms \\textsc{{scoped infeasible}}"),
        ("bitwise duplicates within a seed", ca["duplicate_units_within_seed_total"],
         "preregistered checkpoint rule returned the unmoved channel"),
        ("distinct channels", ca["distinct_channels_total"], "$123-57$"),
        ("of those, bitwise identical to a historical arm", len(ca["duplicates_of_historical_arms"]),
         "max abs difference $0.0$; reused with proof"),
        (r"\textbf{distinct and new}", ca["distinct_channels_total"]-len(ca["duplicates_of_historical_arms"]),
         r"\textbf{the number of new systems this study contributes}")]
L = [r"\begin{tabular}{@{}p{5.2cm} r p{6.4cm}@{}}", r"\toprule", r"quantity & count & note \\", r"\midrule"]
for a, b, c in rows: L.append(f"{a} & {b} & {c} " + r"\\")
L += [r"\midrule",
      f"main arms bitwise identical to no protection & {len(ident)} of {len(be)} & "
      f"{bb['0.1']} at $\\beta=0.1$, {bb['0.3']} at $0.3$, {bb['1.0']} at $1.0$ " + r"\\",
      r"\bottomrule", r"\end{tabular}"]
emit(T("study4_accounting.tex"), "\n".join(L))
emit(T("study4_accounting_note.tex"),
     r"\newcommand{\noteStudyFourAccounting}{Nominal registry slots are not distinct models. The "
     r"duplicates are outcomes of the preregistered checkpoint rule---every unit ran its full 600-update "
     r"budget and selection returned the unmoved initial checkpoint---not failed or abandoned fits. "
     r"Arithmetic verified in both directions and against the per-seed sums. The study's own gloss "
     r"attributes the unmoved arms to $\beta\in\{0.1,0.3\}$; three further arms select the unmoved "
     r"channel at $\beta=1.0$.}")

# ---------------------------------------------------------------- T4 measured erasure rank
ne = json.load(open(P("NEW_ERASURE.json")))
L = [r"\begin{tabular}{l l r r r r}", r"\toprule",
     r"seed & arm & width & nominal & \textbf{measured} & post-erasure \\",
     r" & & & rank loss & \textbf{retained rank} & cross-cov.\ max \\", r"\midrule"]
for sd in "012":
    for arm in ("leace_dax16_none", "leace_dax8_none"):
        v = ne["seeds"][sd][arm]
        L.append(f"{sd} & {tt(arm)} & {v['channel_width']} & {v['expected_rank_loss']} & "
                 f"\\textbf{{{v['realised_projection_rank']}}} & "
                 f"${v['cross_covariance_max_abs_after']:.1e}$ " + r"\\")
    if sd != "2": L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
emit(T("study4_erasure_rank_v4.tex"), "\n".join(L))
sup = ne["seeds"]["0"]["leace_dax16_none"]["coverage"]["RAC1P"]["support_complete_cases"]
emit(T("study4_erasure_rank_v4_note.tex"),
     r"\newcommand{\noteStudyFourErasure}{The nominal rank loss is $1+8+1=10$ centred degrees of freedom "
     r"for the joint $\{$SEX, RAC1P, public\_coverage$\}$ schema. Counting does not determine the "
     r"projection: at width 16 the \emph{measured} loss is $10,10,9$ across seeds, so seed 2 retains "
     r"rank 7 where counting predicts 6. One RAC1P class carries a single complete case in the seed-0 "
     r"fit fold (support " + esc(str(sup)) + r"). Rank-zero erasure at width 8 is established by "
     r"measurement---retained rank 0 with post-erasure cross-covariance at $10^{-35}$ in all three "
     r"seeds---not by the observation that $10>8$.}")

# ---------------------------------------------------------------- T5 chronology
CH = [("1", "locked ACS 2017 transport", tt("349efa45"), "confirmatory", "coalition effect confirmed; the $.001$ rule it did not establish"),
      ("2", "nonlinear / rank", tt("c37807e4"), "development", "negative; penalty was coordinate-dependent, diagnosed after evaluation"),
      ("3", "invariant repair + external baselines", tt("73903b7f"), "development", "negative; a closed-form eraser matched $J$ and beat the developed mechanism"),
      ("4", "direct adversarial channel refinement", tt("69e790af"), "development", r"\textbf{negative}; nominee conjunction \textsc{unsupported} on both years"),
      ("5", "factorial neural + coalition-conditioned projection", r"\emph{unregistered}", r"\textsc{pending}", r"\textsc{no outcome read}; protocol not yet committed")]
L = [r"\begin{tabular}{@{}l p{3.4cm} l l p{5.6cm}@{}}", r"\toprule",
     r"study & construction & evidence commit & status & what it established \\", r"\midrule"]
for a in CH: L.append(" & ".join(a) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
emit(T("study_chronology_v4.tex"), "\n".join(L))
emit(T("study_chronology_v4_note.tex"),
     r"\newcommand{\noteStudyChronologyFour}{Only Study~1 is confirmatory. Studies 2--4 are development "
     r"on pools this project has used repeatedly, and no computation on them can undo that. Study~5 is "
     r"two tracks under review in this revision \emph{before} any outcome exists: its slot registry is "
     r"a plan, not an execution record, and this paper carries its cells as pending with no forecast. "
     r"2016 remains sealed and unscored.}")

# ---------------------------------------------------------------- T6 pending Study 5
PEND = [("neural factorial", r"initialisation $A_0$ vs.\ $J$, common frozen $A_0$ teacher", "168",
         r"\textsc{pending}", "actual $J$ weights loaded and verified; teacher held common"),
        ("", r"teacher coefficient $\gamma$, penalty $\beta$", "", r"\textsc{pending}",
         r"$\gamma=0$ removes preservation, retains authorised source losses"),
        ("", r"local vs.\ coalition policy", "", r"\textsc{pending}",
         "matched contrasts; comparison with untouched $J$, not only with $A_0$"),
        ("projection", "coalition-conditioned rank reduction on frozen $A_0$/$J$", "204",
         r"\textsc{pending}", "measured cross-moment rank and per-class support reported"),
        ("", r"local-conditioned, local-expanded, marginal partial", "", r"\textsc{pending}",
         "mass-scaled local hard projection registered as an alias, not a comparator"),
        ("", "PCA, random compression, ordinary erasers", "", r"\textsc{pending}",
         "compression is a live explanation of Study 4's width effect")]
L = [r"\begin{tabular}{@{}l p{4.3cm} r l p{5.4cm}@{}}", r"\toprule",
     r"track & arm family & nominal slots & result & precondition checked before filling \\", r"\midrule"]
for a in PEND: L.append(" & ".join(a) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
emit(T("study5_pending.tex"), "\n".join(L))
emit(T("study5_pending_note.tex"),
     r"\newcommand{\noteStudyFivePending}{\textbf{No result from either track has been read, and no "
     r"protocol for either has been committed.} At the time of writing the branch carrying this program "
     r"is byte-identical to the Study~4 evidence branch. The slot counts are the nominal registry; they "
     r"are not distinct models and they are not an execution record. Study~4's own registry resolved "
     r"126 nominal slots into 123 releases, 66 distinct channels and 60 distinct-and-new, and the same "
     r"three-way accounting is required here. The finite protocol permits \emph{prospective} reductions "
     r"on measured runtime; retrospective ones would be a different thing. Cells are empty because they "
     r"are empty.}")

# ---------------------------------------------------------------- Fig 1: frontiers
lab = {"A/SEX": r"$A$/SEX", "AB/SEX": r"$AB$/SEX", "A/RAC1P": r"$A$/RAC1P", "AB/RAC1P": r"$AB$/RAC1P"}
GROUP = {"A0": ("no protection", MUTE, "o"), "J": ("frozen $J$", ACC, "D"),
         "leace_A0": ("external eraser", ACC, "s"), "splince_A0": ("external eraser", ACC, "s"),
         "optnet16_C1": ("published adaptation", ACC, "^"), "spectral_C1": ("Study 3", MUTE, "v"),
         "leace_dax8_none": ("erasure at rank 0", MUTE, "x")}
new_arms = [c for (c, e) in p18 if e == "residence" and c.startswith("dax")]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for ax, ep in zip(axes, COLS[:4]):
    for c in new_arms:
        if (c, ep) not in p18: continue
        ax.scatter(p18[(c, ep)], p18[(c, "residence")], s=13, c=WARN, alpha=.55,
                   marker="o", linewidths=0, zorder=2)
    for c, (nm, col, mk) in GROUP.items():
        if (c, ep) not in p18: continue
        ax.scatter(p18[(c, ep)], p18[(c, "residence")], s=42, c=col, marker=mk,
                   edgecolors="white", linewidths=.7, zorder=4)
        if c in ("J", "A0", "leace_A0", "splince_A0", "leace_dax8_none"):
            ax.annotate(c.replace("_A0", "").replace("leace_dax8_none", "rank-0"),
                        (p18[(c, ep)], p18[(c, "residence")]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color=INK)
    ax.set_xlabel(lab[ep] + " additional recovery")
    ax.axvline(0, color=INK, lw=.6, alpha=.4)
axes[0].set_ylabel("residence utility gain over $H$")
handles = [Line2D([], [], ls="", marker="o", ms=4, color=WARN, label="Study 4 arms (60 new channels)"),
           Line2D([], [], ls="", marker="D", ms=5, color=ACC, label="$J$ / external comparators"),
           Line2D([], [], ls="", marker="o", ms=5, color=MUTE, label="no protection / previous line")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=3, frameon=False, fontsize=7)
fig.suptitle("Up and to the left is better: less disclosure at more retained capability", y=1.02, fontsize=8, color=INK)
savefig(fig, "study4_frontier")

# ---------------------------------------------------------------- Fig 2: selection vs training movement
traj = rd("MECH_TRAJECTORY.csv")
print("MECH_TRAJECTORY columns:", list(traj[0].keys()), file=sys.stderr)
emit(T("_tmp"), "")  # placeholder removed below
os.remove(T("_tmp")); manifest.pop("tables/_tmp", None)

# ---------------------------------------------------------------- Fig 3: chronology
fig, ax = plt.subplots(figsize=(9, 2.1))
ys = [("1", "locked transport 2017", "confirmatory", ACC),
      ("2", "nonlinear / rank", "development, negative", MUTE),
      ("3", "invariant repair + externals", "development, negative", MUTE),
      ("4", "direct adversarial refinement", "development, negative", WARN),
      ("5", "factorial + coalition projection", "pending, no outcome", "#cfcfcf")]
for i, (n, name, st, col) in enumerate(ys):
    ax.barh(i, 1, color=col, height=.55, alpha=.85 if col != "#cfcfcf" else 1.0,
            hatch="//" if col == "#cfcfcf" else None, edgecolor="white")
    ax.text(1.03, i, f"Study {n}: {name}  —  {st}", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0, 4.6); ax.set_ylim(-.6, len(ys)-.4); ax.invert_yaxis()
ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("One confirmatory result, three negative development studies, one pending program",
             fontsize=8, loc="left", color=INK)
savefig(fig, "study_chronology_v4")

json.dump(manifest, open("/Users/nathansamson/PCRL-terminal-2-manuscript/results/"
                         "pcrl_manuscript_review_v4/ASSET_HASHES.json", "w"), indent=1)
print(json.dumps({"tables": sorted(k for k in manifest if k.startswith("tables")),
                  "figures": sorted(k for k in manifest if k.startswith("figures"))}, indent=1))

# ================= APPENDED: selection-versus-training movement =================
import glob, numpy as np
recs = []
for p in sorted(glob.glob(os.path.join(EV, "seed_*/fits/*/fit_record.json"))):
    d = json.load(open(p))
    fresh = [c["monitor_score"] for c in d["monitor_scores"]]
    final = [c["monitor_score"] for c in d["final_slate_monitor_scores"]]
    contemp = [c["monitor_score"] for c in d["contemporaneous_monitor_scores"]]
    steps = [c["step"] for c in d["monitor_scores"]]
    recs.append({"seed": d["seed"], "arm": d["arm"], "beta": d["beta"], "policy": d["policy"],
                 "width": d["width"], "initialisation": d["initialisation"],
                 "steps": steps, "fresh": fresh, "final": final, "contemp": contemp,
                 "selected_step": d["selected_step"],
                 "argmin_fresh": steps[int(np.argmin(fresh))],
                 "argmin_final": steps[int(np.argmin(final))],
                 "argmin_contemp": steps[int(np.argmin(contemp))]})
sel = {"n_fits": len(recs),
       "initialisations": dict(collections.Counter(r["initialisation"] for r in recs)),
       "any_initialised_from_J": any("J" in r["initialisation"].split() for r in recs),
       "fresh_rule_selects_step0": sum(1 for r in recs if r["argmin_fresh"] == 0),
       "final_slate_rule_would_select_step0": sum(1 for r in recs if r["argmin_final"] == 0),
       "contemporaneous_rule_would_select_step0": sum(1 for r in recs if r["argmin_contemp"] == 0),
       "selected_step_matches_fresh_argmin": sum(1 for r in recs if r["selected_step"] == r["argmin_fresh"]),
       "selected_step_distribution": dict(collections.Counter(r["selected_step"] for r in recs))}
json.dump(sel, open("/Users/nathansamson/PCRL-terminal-2-manuscript/results/"
                    "pcrl_manuscript_review_v4/SELECTION_AUDIT.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
ax = axes[0]
moved = [r for r in recs if r["selected_step"] > 0]
for r in recs:
    base = r["fresh"][0]
    ax.plot(r["steps"], [x - base for x in r["fresh"]], color=WARN if r["selected_step"] > 0 else MUTE,
            lw=.7, alpha=.45 if r["selected_step"] > 0 else .22, zorder=2)
for r in recs:
    ax.scatter([r["selected_step"]], [r["fresh"][r["steps"].index(r["selected_step"])] - r["fresh"][0]],
               s=9, c=WARN if r["selected_step"] > 0 else MUTE, zorder=3, linewidths=0, alpha=.7)
ax.axhline(0, color=INK, lw=.6)
ax.set_xlabel("mapper update"); ax.set_ylabel("monitor score, change from step 0")
ax.set_title(f"Equal-budget fresh probe (the rule used)\n{len(recs)-len(moved)} of {len(recs)} arms stay at step 0",
             fontsize=7.5, color=INK, loc="left")
ax = axes[1]
lbl = ["equal-budget\nfresh probe\n(Amendment 1)", "single final\nslate", "contemporaneous\nslate"]
vals = [sel["fresh_rule_selects_step0"], sel["final_slate_rule_would_select_step0"],
        sel["contemporaneous_rule_would_select_step0"]]
bars = ax.bar(lbl, vals, color=[ACC, MUTE, MUTE], width=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}/{len(recs)}", ha="center", fontsize=7.5, color=INK)
ax.set_ylim(0, len(recs)*1.12); ax.set_ylabel("arms selecting the unmoved channel")
ax.set_title("Both rejected yardsticks are biased the same way", fontsize=7.5, color=INK, loc="left")
ax.tick_params(axis="x", labelsize=6.5)
ax = axes[2]
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values(): s.set_edgecolor("#cccccc")
ax.add_patch(plt.Rectangle((0.02, 0.02), .96, .96, fill=True, facecolor="#f4f4f4",
                           edgecolor="#bbbbbb", hatch="//", lw=.8))
ax.text(.5, .60, "PENDING", ha="center", va="center", fontsize=13, color="#7a7a7a", weight="bold")
ax.text(.5, .40, "matched $A_0$ vs.\\ $J$ initialisation\nand $\\gamma$ ablation (Study 5)\n"
                 "no protocol committed", ha="center", va="center", fontsize=7, color="#7a7a7a")
ax.set_title("No forecast is drawn here", fontsize=7.5, color=INK, loc="left")
savefig(fig, "study4_selection")
json.dump(manifest, open("/Users/nathansamson/PCRL-terminal-2-manuscript/results/"
                         "pcrl_manuscript_review_v4/ASSET_HASHES.json", "w"), indent=1)
print(json.dumps(sel, indent=1))

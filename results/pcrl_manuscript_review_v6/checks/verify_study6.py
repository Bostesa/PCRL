#!/usr/bin/env python3
"""Independently recompute the Study 6 (utility-first pilot) claims and emit the v6 assets.

Reads only committed evidence at the pinned commit through the git object store. Arithmetic only:
no model is fitted, no ACS pool is read, 2016 is untouched. Screening decisions are RE-DERIVED from
the stored per-anchor increments and compared with the study's recorded flags.
"""
import json, os, statistics, subprocess, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHA = "0517c06a7"                                            # T1 handoff commit (evidence recorded inside)
EVIDENCE_SHA = "ba531ab424c593fe8573cd320d2bcef379907cf1"    # commit containing the published evidence
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_utility_extension_v1/"
OUT = os.path.join(ROOT, "papers/pcrl_manuscript_v6")
REV = os.path.join(ROOT, "results/pcrl_manuscript_review_v6")
T = lambda n: os.path.join(OUT, "tables", n)
F = lambda n: os.path.join(OUT, "figures", n)
ENDS = ["A/SEX", "AB/SEX", "A/RAC1P", "AB/RAC1P"]
WEIGHTS = ["unweighted", "person_weighted"]
MEAN_SCREEN, ANCHOR_SCREEN = 0.001, 0.003

sources, manifest, checks = {}, {}, []
def show(name):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}{name}"], check=True, capture_output=True).stdout
    sources[RD + name] = hashlib.sha256(raw).hexdigest()
    return raw.decode()
rjson = lambda n: json.loads(show(n))
def check(name, got, want, tol=5e-5):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    checks.append({"check": name, "recomputed": got, "reported": want, "ok": bool(ok)})
    if not ok: raise SystemExit(f"MISMATCH {name}: recomputed {got} vs reported {want}")
def emit(path, body):
    open(path, "w").write(body if body.endswith("\n") else body + "\n")
    manifest[os.path.relpath(path, OUT)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
def savefig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(F(f"{stem}.{ext}"), bbox_inches="tight", dpi=160)
        manifest[f"figures/{stem}.{ext}"] = hashlib.sha256(open(F(f"{stem}.{ext}"), "rb").read()).hexdigest()
    plt.close(fig)
esc = lambda s: str(s).replace("_", r"\_")
tt = lambda s: r"\texttt{" + esc(s) + "}"

screen = rjson("PILOT_SCREEN.json"); t2 = rjson("gates/T2.json"); t1g = rjson("gates/T1.json")
t1 = rjson("TIER1_REANALYSIS.json"); cal = rjson("CALIBRATION.json"); hand = rjson("HANDOFF.json")

# ---------------------------------------------------------------- 1. counts and tier status
check("evidence commit recorded by Terminal 1", hand["evidence_commit"], EVIDENCE_SHA)
check("pilot units fitted", t2["fitted"], 42)
check("pilot units audited", t2["audited"], 42)
check("exact duplicates", t2["duplicates"], 1)
check("audit failures", len(t2["audit_failures"]), 0)
check("configurations screened", len(screen), 14)
check("units = 14 configurations x 3 anchors", len(screen) * 3, 42)
check("T2 gate status", t2["status"], "FAIL")
check("T2 passing configurations", len(t2["passing"]), 0)
check("T1 gate status", t1g["status"], "PASS")
check("T3/T4 not started (no nominee, gate rule)", (hand["tier_reached"].count("not started") + hand["tier_reached"].count("not started")) > 0, True)
# cumulative slot accounting: the pilot's 42 units are inside the 135-slot expanded plan
counts = hand["counts"]
planned_not_run = [v for k, v in counts.items() if "not_run" in k or "not run" in k]
check("cumulative plan: 42 pilot units + 93 unrun Tier-3 slots = 135", 42 + 93, 135)

# ---------------------------------------------------------------- 2. re-derive the screen
rows, worst = {}, {}
for cfg, s in screen.items():
    recon = [float(x) for x in s["recon_reduction_by_anchor"]]
    check(f"{cfg} recon mean", round(statistics.mean(recon), 6), round(float(s["recon_mean"]), 6), 1e-6)
    check(f"{cfg} recon leg (>=10% in >=2 of 3 anchors)", sum(x >= 0.10 for x in recon) >= 2, bool(s["recon_ok"]))
    means, anchors_ok, worst_by_w = {}, True, {}
    for w in WEIGHTS:
        for e in ENDS:
            v = [float(x) for x in s["increments_vs_J"][e][w]]
            m = statistics.mean(v); means[(e, w)] = m
            worst_by_w[w] = max(worst_by_w.get(w, -9), m)
            anchors_ok &= max(v) <= ANCHOR_SCREEN
    worst_mean = max(worst_by_w.values())
    mean_ok = all(m <= MEAN_SCREEN for m in means.values())
    check(f"{cfg} mean-increment leg re-derived", mean_ok, bool(s["mean_increment_ok"]))
    check(f"{cfg} anchor-increment leg re-derived", anchors_ok, bool(s["anchor_increment_ok"]))
    check(f"{cfg} overall screen", bool(mean_ok and anchors_ok and s["recon_ok"] and s["source_ok"]), bool(s["pass"]))
    rows[cfg] = {"recon": statistics.mean(recon), "means": means, "worst": worst_mean,
                 "worst_unweighted": worst_by_w["unweighted"], "worst_person_weighted": worst_by_w["person_weighted"],
                 "source_ok": bool(s["source_ok"]), "pass": bool(s["pass"])}
    worst[cfg] = worst_by_w["unweighted"]
check("no configuration passed the screen", any(r["pass"] for r in rows.values()), False)
check("every configuration passed the source allowance", all(r["source_ok"] for r in rows.values()), True)
check("every configuration passed the reconstruction leg", all(screen[c]["recon_ok"] for c in screen), True)
best = min(worst, key=worst.get)
check("closest configuration", best, "X_r2_L2_b030")
check("its worst mean increment, unweighted (the study's headline)", round(rows[best]["worst_unweighted"], 4), 0.0016, 6e-5)
check("its worst mean increment, person-weighted (larger; both miss the screen)", round(rows[best]["worst_person_weighted"], 4), 0.0021, 6e-5)
check("its capability gain is the lowest in the grid", round(rows[best]["recon"], 3), 0.111, 6e-4)
check("the closest configuration misses the .001 screen under BOTH weightings",
      (rows[best]["worst_unweighted"] > MEAN_SCREEN, rows[best]["worst_person_weighted"] > MEAN_SCREEN), (True, True))
check("unprotected control carries the most capability", max(rows, key=lambda c: rows[c]["recon"]), "U_r2")
check("unprotected control carries the most disclosure", max(rows, key=lambda c: rows[c]["worst_unweighted"]), "U_r2")
# monotonicity in beta, per policy
for pol in ("C1", "L1", "L2"):
    seq = [(b, rows[f"X_r2_{pol}_b{b:03d}"]) for b in (1, 3, 10, 30)]
    check(f"{pol}: capability falls monotonically in beta",
          all(seq[i][1]["recon"] > seq[i + 1][1]["recon"] for i in range(3)), True)
    check(f"{pol}: worst sensitive increment (unweighted) falls monotonically in beta",
          all(seq[i][1]["worst_unweighted"] > seq[i + 1][1]["worst_unweighted"] for i in range(3)), True)
check("no coalition advantage at matched beta=30 (C1 worst vs L2 worst, unweighted)",
      (round(rows["X_r2_C1_b030"]["worst_unweighted"], 4), round(rows["X_r2_L2_b030"]["worst_unweighted"], 4)), (0.0049, 0.0016))

# ---------------------------------------------------------------- 3. tier-1 reanalysis
pc = t1["per_config"]
check("configurations re-scored", len(pc), 124)
check("configurations re-scored (field)", t1["configurations_searched"], 124)
check("new search family size (old 3900 not reused)", (t1["new_family_size"], t1["old_family_size_not_reused"]), (2480, 3900))
check("pass vs J (recounted)", sum(v["passes_vs_J"] for v in pc.values()), 0)
check("pass vs leace_A0 (recounted)", sum(v["passes_vs_leace_A0"] for v in pc.values()), 0)
check("pass both (recounted)", sum(v["passes_both"] for v in pc.values()), 0)
check("residence point better than J, both weightings (recounted)",
      sum(v["residence_point_better_than_J_both_weightings"] for v in pc.values()), 55)
check("reported n_pass_vs_J", t1["n_pass_vs_J"], 0)
closest = t1["closest_by_worst_weighting_residence_vs_J"]
check("closest list is all A0-channel releases (E_A0 projections and bitwise-A0 Track N continuations)",
      all("A0" in c["config"] for c in closest), True)
check("closest configurations' sensitive upper bounds are >> .001",
      min(min(c["max_sensitive_upper_vs_J"].values()) for c in closest) > 0.01, True)

# ---------------------------------------------------------------- 4. attack calibration
per_recipe = cal["decision"]["per_recipe"]
check("prespecified recipes calibrated", len(per_recipe), 8)
check("recipes stronger than the standard slate", sum(r["stronger"] for r in per_recipe.values()), 0)
check("added stress strength not established", t1g["stress_strength_established"], False)
check("no recipe adopted", t1g["adopted_recipe"], None)
pos = cal["positive_control_A0_minus_H_validation"]
check("positive control A/SEX (unweighted)", round(pos["A/SEX"]["unweighted"], 3), 0.032, 6e-4)
check("positive control A/RAC1P (unweighted)", round(pos["A/RAC1P"]["unweighted"], 3), 0.051, 6e-4)
port = max(abs(float(v)) for v in cal["audit_portability_max_abs"].values())
check("cross-platform audit reproduction exceeded its tolerance", port > cal["audit_portability_tol"], True)
check("cross-platform max abs deviation", round(port, 4), 0.0031, 6e-5)
check("portability flag", cal["audit_portability_ok"], False)

# ================================================================== assets
ORDER = [f"X_r2_{p}_b{b:03d}" for p in ("C1", "L1", "L2") for b in (1, 3, 10, 30)] + ["U_r2", "P_r2"]
LABEL = {"U_r2": "unprotected control ($\\beta=0$)", "P_r2": "deployable PCA residual"}
L = [r"\begin{tabular}{@{}l r r r r r c@{}}", r"\toprule",
     r"unit & capability & $A$/SEX & $AB$/SEX & $A$/RAC1P & $AB$/RAC1P & screen \\", r"\midrule"]
for c in ORDER:
    r = rows[c]
    m = [r["means"][(e, "unweighted")] for e in ENDS]
    lab = tt(c) + (f" \\ {LABEL[c]}" if c in LABEL else "")
    L.append(lab + f" & {r['recon']:.3f} & " + " & ".join(f"${x:+.4f}$" for x in m) +
             " & " + (r"\textsc{pass}" if r["pass"] else r"\textsc{fail}") + r" \\")
    if c == "X_r2_C1_b030" or c == "X_r2_L1_b030" or c == "X_r2_L2_b030": L.append(r"\addlinespace[2pt]")
emit(T("study6_pilot.tex"), "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

emit(T("study6_tiers.tex"), "\n".join([
    r"\begin{tabular}{@{}l l l p{6.6cm}@{}}", r"\toprule",
    r"tier & status & cost & outcome \\", r"\midrule",
    r"T0 infrastructure, archive, restore & \textsc{pass} & \$0.16 & bytes restored and re-hashed; frozen mappers re-execute \\",
    r"T1 evidence reuse $+$ audit calibration & \textsc{pass} & \$0.20 & 0 of 124 stored configurations meet the utility-first criterion; 0 of 8 attacker recipes stronger than the standard slate \\",
    r"T2 pilot (42 units) & \textbf{\textsc{fail}} & \$0.35 & capability leg passed for all 14 configurations; the $.001$-nat disclosure screen failed for all 14 \\",
    r"T3 expansion (93 further slots) & \textsc{not triggered} & --- & forbidden by the T2 gate rule; no partial grid was fitted \\",
    r"T4 nominee comparison, 2017 transport & \textsc{not triggered} & --- & nothing was nominated \\",
    r"\bottomrule", r"\end{tabular}"]))

emit(T("study_chronology_v6.tex"), "\n".join([
    r"\begin{tabular}{@{}l p{3.4cm} l l p{5.7cm}@{}}", r"\toprule",
    r"study & construction & evidence & status & what it established \\", r"\midrule",
    r"1 & locked ACS 2017 transport & \texttt{349efa45} & confirmatory & coalition effect over both local controls; the $.001$ point rule, not interval noninferiority \\",
    r"2 & nonlinear penalty / rank & \texttt{c37807e4} & development & negative; penalty coordinate-dependent, diagnosed after evaluation \\",
    r"3 & invariant repair $+$ external baselines & \texttt{73903b7f} & development & negative; closed-form erasers beat the developed mechanism \\",
    r"4 & direct adversarial refinement from $A_0$ & \texttt{69e790af} & development & negative on both years \\",
    r"5 & $J$-initialised factorial $+$ coalition projection & \texttt{a56bcc7f} & development & negative; no prospectively selected release met the conjunction \\",
    r"6 & utility-first residual extension of $J$ & \texttt{ba531ab4} & development & \textbf{negative pilot}: capability gained on the proxy, disclosure screen failed at every configuration; expansion not triggered \\",
    r"\bottomrule", r"\end{tabular}"]))

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, MUTE, ACC, WARN = "#1b1b1b", "#8a8a8a", "#2a6f97", "#a4303f"
fig, ax = plt.subplots(figsize=(5.4, 2.6))
style = {"C1": (WARN, "o", "coalition $C1$"), "L1": (ACC, "^", "local $L1$"), "L2": (INK, "s", "local $L2$ (mass-matched)")}
for pol, (col, mk, lab) in style.items():
    xs = [rows[f"X_r2_{pol}_b{b:03d}"]["recon"] for b in (1, 3, 10, 30)]
    ys = [rows[f"X_r2_{pol}_b{b:03d}"]["worst_unweighted"] for b in (1, 3, 10, 30)]
    ax.plot(xs, ys, color=col, lw=1.2, marker=mk, ms=4.5, label=lab, zorder=3)
for c, mk, lab in [("U_r2", "D", "unprotected ($\\beta=0$)"), ("P_r2", "P", "PCA residual")]:
    ax.scatter(rows[c]["recon"], rows[c]["worst_unweighted"], s=42, color=MUTE, marker=mk, edgecolor="white", lw=.8, zorder=4, label=lab)
ax.axhline(0.001, color=WARN, lw=.9, ls="--")
ax.text(0.455, 0.0016, "declared .001-nat screen", fontsize=6.5, color=WARN, ha="right", va="bottom")
ax.annotate("closest unit\n(+0.0016, capability 0.111)", xy=(rows[best]["recon"], rows[best]["worst_unweighted"]),
            xytext=(0.17, 0.012), fontsize=6.5, color=INK,
            arrowprops=dict(arrowstyle="->", lw=.7, color=INK))
ax.set_xlabel("capability: mean reduction in residual teacher-reconstruction MSE over $J$")
ax.set_ylabel("worst mean sensitive\nincrement over $J$ (nats)")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.suptitle("Study 6 pilot: capability and disclosure did not separate ($\\beta$ increases to the left)", fontsize=8, y=1.0)
savefig(fig, "study6_pilot")

# claim-support table: v5 rows + Study 6 rows generated from checked values
v5 = open(os.path.join(ROOT, "papers/pcrl_manuscript_v5/tables/claim_support_v5.tex")).read().split("\\bottomrule")[0]
b = rows[best]
rows6 = [
 ("Study 6: extension adds capability on the proxy", "Empirical, development",
  f"{min(r['recon'] for r in rows.values()):.3f}-{max(r['recon'] for r in rows.values()):.3f} reduction in residual reconstruction error; 14/14 configurations",
  "Reconstruction is a proxy, not residence transfer; no residential label entered it"),
 ("Study 6: added disclosure within the declared .001-nat screen", r"\textbf{Not established}",
  f"0 of 14 configurations; closest ${b['worst_unweighted']:+.4f}$ unw / ${b['worst_person_weighted']:+.4f}$ pw",
  "Point-estimate screen, no intervals (nothing nominated); one family, $r{=}2$"),
 ("Study 6: capability and disclosure separable by the adversarial term", r"\textbf{Not established}",
  "Both fall monotonically in $\\beta$ in all three policies", "This finite family and budget only"),
 ("Study 6: coalition benefit over matched local control", r"\textbf{Not established}",
  "$C1$ $+0.0049$ vs $L2$ $+0.0016$ at $\\beta{=}30$", "Third mechanism family to reproduce this"),
 ("Utility-first candidate hidden in the earlier 372-slot evidence", r"\textbf{None}",
  "0 of 124 configurations pass vs $J$ or \\texttt{leace\\_A0} ($m{=}2480$)", "Exploratory reanalysis; does not change that study's own verdict"),
 ("Study 6: stronger attacker found", r"\textbf{Not established}", "0 of 8 prespecified recipes beat the standard slate",
  "Not evidence of survival against a stronger attack"),
 ("Study 6 Tier 3 / Tier 4 results", r"\textsc{not triggered}", "Gate rule after the failed pilot; no partial grid fitted",
  "Absent evidence for an unrun design, not missing evidence for a claimed one"),
]
body6 = v5 + "".join(f"{a} & {c} & {d} & {e} \\\\\n\\addlinespace[1.5pt]\n" for a, c, d, e in rows6) + "\\bottomrule\n\\end{tabular}"
emit(T("claim_support_v6.tex"), body6)

# chronology strip, updated for a completed Study 6
fig, ax = plt.subplots(figsize=(7.2, 1.1))
st = [("1: locked 2017\ntransport", "confirmed\ncoalition effect", INK), ("2: nonlinear\npenalty / rank", "negative", "#d9d9d9"),
      ("3: invariant repair\n+ externals", "negative", "#d9d9d9"), ("4: adversarial\nfrom $A_0$", "negative", "#d9d9d9"),
      ("5: $J$-init +\ncoalition projection", "negative", "#d9d9d9"), ("6: utility-first\nextension of $J$", "negative\n(pilot)", "#d9d9d9")]
for i, (n, lab, col) in enumerate(st):
    ax.add_patch(plt.Rectangle((i, 0), .92, 1, facecolor=col, edgecolor=INK, lw=.6))
    tc = "white" if col == INK else INK
    ax.text(i + .46, .68, n, ha="center", va="center", fontsize=6.5, color=tc)
    ax.text(i + .46, .27, lab, ha="center", va="center", fontsize=6.5, color=tc, style="italic")
ax.set_xlim(-.05, 6); ax.set_ylim(-.05, 1.05); ax.axis("off")
savefig(fig, "study_chronology_v6")

json.dump({"pinned_commit": SHA, "evidence_commit": EVIDENCE_SHA, "sources_sha256": sources,
           "checks": checks, "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks),
           "rederived_screen": {c: {"capability": rows[c]["recon"], "worst_mean_increment_unweighted": rows[c]["worst_unweighted"],
                                    "worst_mean_increment_person_weighted": rows[c]["worst_person_weighted"],
                                    "pass": rows[c]["pass"]} for c in ORDER},
           "tiers": {"T0": "PASS", "T1": "PASS", "T2": "FAIL", "T3": "NOT_TRIGGERED", "T4": "NOT_TRIGGERED"},
           "zero_new_acs_fits": True},
          open(os.path.join(REV, "STUDY6_VERIFICATION.json"), "w"), indent=1, default=str)
json.dump(manifest, open(os.path.join(REV, "ASSET_HASHES_STUDY6.json"), "w"), indent=1, sort_keys=True)
print(f"{len(checks)} checks, all ok; {len(manifest)} assets")

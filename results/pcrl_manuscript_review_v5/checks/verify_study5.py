#!/usr/bin/env python3
"""Independently recompute the Study 5 claims and emit the v5 tables and figures.

Reads ONLY committed evidence at the pinned commit through the git object store
(`git show <sha>:<path>`), never a live checkout, so nothing depends on another
terminal's working tree. Arithmetic only: no model is fitted, no ACS pool is read,
2016 is not touched. Every assertion below compares an independently recomputed
value with the number the study (or its PAPER_ADDENDUM) prints.
"""
import csv, io, json, os, statistics, subprocess, hashlib, collections
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHA = "7f961d5c7f6f0562efcb25a27a77bb5221c279a7"      # final handoff of Study 5
EVIDENCE_SHA = "a56bcc7ff7a122a6411ac3072b11b00b75f4cdad"  # commit that contains the evidence
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_competitive_method_v1/"
OUT = os.path.join(ROOT, "papers/pcrl_manuscript_v5")
REV = os.path.join(ROOT, "results/pcrl_manuscript_review_v5")
T = lambda n: os.path.join(OUT, "tables", n)
F = lambda n: os.path.join(OUT, "figures", n)

sources = {}
def show(name):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}{name}"], check=True,
                         capture_output=True).stdout
    sources[RD + name] = hashlib.sha256(raw).hexdigest()
    return raw.decode()
def rcsv(name): return list(csv.DictReader(io.StringIO(show(name))))
def rjson(name): return json.loads(show(name))

manifest, checks = {}, []
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
f4 = lambda x: f"${x:+.4f}$".replace("+-", "-")

# ------------------------------------------------------------------ 1. denominators
counts = rjson("COUNTS.json")
N, E = counts["tracks"]["N"], counts["tracks"]["E"]
check("nominal = N + E", N["expected"] + E["expected"], 372)
check("nominal_total field", counts["nominal_total"], 372)
check("fitted N", N["fitted"], 168); check("fitted E", E["fitted"], 204)
check("distinct audited = 102 + 192", N["audited_distinct"] + E["audited_distinct"], 294)
check("duplicates = 66 + 12", N["duplicate_of_audited_release"] + E["duplicate_of_audited_release"], 78)
check("distinct + duplicates = nominal (references NOT inside 372)",
      N["audited_distinct"] + E["audited_distinct"] + N["duplicate_of_audited_release"]
      + E["duplicate_of_audited_release"], 372)
check("references (ref_A0, ref_J x 3 anchors), outside the 372", counts["references_audited"], 6)
stress = rjson("STRESS.json")
stress_rel = sorted({k.split("|")[0] for k in stress["conditions"]})
check("stress release conditions", len(stress_rel), 17)
check("stress interfaces = (17 releases + H-only view) x 3 anchors", (len(stress_rel) + 1) * 3, 54)
neural_in_stress = [c for c in stress_rel if c.startswith("N_J_")]
ex17 = rjson("EXPLORATORY_2017.json")
transported = sorted({r["condition"] for r in ex17["rows"]} - {"A0", "J", "leace_A0", "splince_A0", "optnet16_C1"})
check("2017 transported Track E conditions", len(transported), 6)
check("2017 transported units = 6 x 3 anchors", len(transported) * 3, 18)
check("2017 missing rows", len(ex17["missing"]), 0)

# ------------------------------------------------------------------ 2. seed means, 2018
pts = rcsv("POINTS_2018.csv")
ENDS = ["increment/A/SEX", "increment/AB/SEX", "increment/A/RAC1P", "increment/AB/RAC1P", "residence_gain_vs_H"]
def smean(rows, cond, w):
    r = [x for x in rows if x["condition"] == cond and x["weight"] == w]
    assert len(r) == 3, (cond, w, len(r))
    return [statistics.mean(float(x[e]) for x in r) for e in ENDS]
p18 = {(c, w): smean(pts, c, w) for c in {x["condition"] for x in pts} for w in ("unweighted", "person_weighted")
       if sum(1 for x in pts if x["condition"] == c and x["weight"] == w) == 3}
ADD = {  # PAPER_ADDENDUM table, unweighted; "J (ref)" is ref_J
    "ref_J": [.0015, .0080, .0000, .0000, .0215], "leace_A0": [.0039, .0038, .0097, .0096, .0220],
    "E_J_C_k8": [-.0007, -.0011, -.0011, -.0000, .0114], "E_J_C_k6": [-.0004, .0028, -.0024, -.0017, .0166],
    "leace_J": [.0000, .0000, -.0001, .0000, .0115], "E_A0_C_k2": [.0163, .0201, .0223, .0183, .0309],
    "E_A0_L_k2": [.0112, .0150, .0425, .0436, .0291]}
for c, v in ADD.items():
    for e, got, want in zip(ENDS, p18[(c, "unweighted")], v):
        check(f"2018 {c} {e} unweighted", round(got, 4), want, tol=6e-5)
check("historical J AB/RAC1P differs from ref_J (amendment 4)", round(p18[("J", "unweighted")][3], 4), -0.0006, 6e-5)
# J-initialised neural nominees are bitwise J in every reported number
for c in ("N_J_g000_C1_b100", "N_J_g000_C1_b300"):
    for w in ("unweighted", "person_weighted"):
        check(f"{c} equals ref_J ({w})", p18[(c, w)] == p18[("ref_J", w)], True)

# ------------------------------------------------------------------ 3. intervals
IP, IX = rcsv("INTERVALS_P.csv"), rcsv("INTERVALS_X.csv")
check("family P size", len(IP), 400); check("family X size", len(IX), 3900)
def iv(rows, l, r, e, w):
    m = [x for x in rows if x["left"] == l and x["right"] == r and x["endpoint"] == e and x["weight"] == w]
    assert len(m) == 1, (l, r, e, w, len(m)); x = m[0]
    return dict(est=float(x["estimate"]), lo=float(x["candidate_wide_low"]), hi=float(x["candidate_wide_high"]),
                better=x["candidate_wide_better"] == "True", worse=x["candidate_wide_worse"] == "True",
                family=x["comparison_family"])
k8J = iv(IP, "E_J_C_k8", "ref_J", "recovery/AB/SEX", "unweighted")
check("k8 vs ref_J AB/SEX est", round(k8J["est"], 4), -0.0091); check("k8 vs ref_J AB/SEX better", k8J["better"], True)
k8Jw = iv(IP, "E_J_C_k8", "ref_J", "recovery/AB/SEX", "person_weighted")
check("k8 vs ref_J AB/SEX better (person-weighted)", k8Jw["better"], True)
k8res = iv(IP, "E_J_C_k8", "ref_J", "utility/same_residence", "unweighted")
check("k8 vs ref_J residence est", round(k8res["est"], 4), 0.0101); check("k8 residence significantly worse", k8res["worse"], True)
k8resw = iv(IP, "E_J_C_k8", "ref_J", "utility/same_residence", "person_weighted")
check("k8 residence NOT significant person-weighted", k8resw["worse"], False)
# ref_J vs historical J: identical verdict (amendment 4)
for e in ("recovery/AB/SEX", "utility/same_residence"):
    for w in ("unweighted", "person_weighted"):
        a, b = iv(IP, "E_J_C_k8", "ref_J", e, w), iv(IP, "E_J_C_k8", "J", e, w)
        check(f"k8 verdict same under ref_J and J ({e},{w})", (a["better"], a["worse"]), (b["better"], b["worse"]))
# E_J_C_k8 vs leace_J: point agreement within .0011 holds UNWEIGHTED only
mx_u = max(abs(iv(IP, "E_J_C_k8", "leace_J", e, "unweighted")["est"]) for e in
           ["recovery/A/SEX", "recovery/AB/SEX", "recovery/A/RAC1P", "recovery/AB/RAC1P", "utility/same_residence"])
mx_w = max(abs(iv(IP, "E_J_C_k8", "leace_J", e, "person_weighted")["est"]) for e in
           ["recovery/A/SEX", "recovery/AB/SEX", "recovery/A/RAC1P", "recovery/AB/RAC1P", "utility/same_residence"])
check("k8 vs leace_J max |point diff| unweighted", round(mx_u, 4), 0.0011, 6e-5)
check("k8 vs leace_J max |point diff| person-weighted", round(mx_w, 4), 0.0020, 6e-5)
# A0 k2 coalition race effect: family X only (exploratory), never in P
check("no A0 release in family P", any(x["left"].startswith("E_A0") for x in IP), False)
a0 = {}
for ctl in ("E_A0_L_k2", "E_A0_LX_k2"):
    for e in ("recovery/A/RAC1P", "recovery/AB/RAC1P", "utility/same_residence"):
        for w in ("unweighted", "person_weighted"):
            a0[(ctl, e, w)] = iv(IX, "E_A0_C_k2", ctl, e, w)
            check(f"A0 k2 vs {ctl} {e} {w} family X", a0[(ctl, e, w)]["family"].startswith("E_C_vs_"), True)
            if e != "utility/same_residence":
                check(f"A0 k2 vs {ctl} {e} {w} better (X)", a0[(ctl, e, w)]["better"], True)
            else:
                check(f"A0 k2 vs {ctl} residence {w} unresolved (X)", (a0[(ctl, e, w)]["better"], a0[(ctl, e, w)]["worse"]), (False, False))

# ------------------------------------------------------------------ 4. competitive conjunction, recomputed
PANEL = ["E_J_C_k8", "E_J_C_k6", "N_J_g000_C1_b100", "N_J_g000_C1_b300"]
EXT = ["leace_A0", "splince_A0", "optnet16_C1"]
SENS = ["recovery/A/SEX", "recovery/AB/SEX", "recovery/A/RAC1P", "recovery/AB/RAC1P"]
conj = {}
for c in PANEL:
    per_w = {}
    for w in ("unweighted", "person_weighted"):
        ok_any = False
        for jref in ("ref_J", "J"):
            for ext in EXT:
                for e in SENS:
                    vj, vx = iv(IP, c, jref, e, w), iv(IP, c, ext, e, w)
                    if not (vj["better"] and vx["better"]): continue
                    # (a) residence noninferior at .001 vs that comparator; (b) other endpoints within .001
                    rx = iv(IP, c, ext, "utility/same_residence", w)
                    others = [iv(IP, c, ext, o, w)["hi"] <= 0.001 and iv(IP, c, jref, o, w)["hi"] <= 0.001
                              for o in SENS if o != e]
                    if rx["hi"] <= 0.001 and all(others): ok_any = True
        per_w[w] = ok_any
    conj[c] = per_w
check("no panel release meets the registered conjunction", any(any(v.values()) for v in conj.values()), False)

# ------------------------------------------------------------------ 5. 2017 panel
def m17(cond, w):
    r = [x for x in ex17["rows"] if x["condition"] == cond and x["weight"] == w]
    assert len(r) == 3, (cond, w, len(r))
    return [statistics.mean(x[e] for x in r) for e in ENDS]
p17 = {(c, w): m17(c, w) for c in {x["condition"] for x in ex17["rows"]} for w in ("unweighted", "person_weighted")}
check("2017 J residence (person-weighted)", round(p17[("J", "person_weighted")][4], 4), 0.0171, 6e-5)
check("2017 k6 residence (person-weighted)", round(p17[("E_J_C_k6", "person_weighted")][4], 4), 0.0159, 6e-5)
gap17w = p17[("J", "person_weighted")][4] - p17[("E_J_C_k6", "person_weighted")][4]
gap17u = p17[("J", "unweighted")][4] - p17[("E_J_C_k6", "unweighted")][4]
check("2017 k6 person-weighted residence deficit exceeds .001 at the point", gap17w > 0.001, True)

# ------------------------------------------------------------------ 6. stronger attack weaker than standard audit
weaker = 0; total = 0
for k, v in stress["conditions"].items():
    for e in ("A/SEX", "AB/SEX"):
        total += 1; weaker += v["stress_increment_seed_mean"][e] < v["standard_audit_increment_seed_mean"][e]
check("stress below standard audit on both SEX endpoints, every condition", weaker, total)
refA0 = stress["conditions"]["ref_A0|unweighted"]
check("stress A0 A/SEX", round(refA0["stress_increment_seed_mean"]["A/SEX"], 3), -0.092, 6e-4)

# ------------------------------------------------------------------ 7. projection form (rank-deficiency)
tef = rjson("TRACK_E_FITS.json")
ranks, dropped = [], []
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "supported_rank": ranks.append(v)
            elif k == "dropped_directions": dropped.append(v)
            else: walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(tef)
check("supported rank 16 in every Track E fit", (len(ranks), set(ranks)), (204, {16}))
check("zero off-support directions dropped (2 channels x 3 anchors)", (len(dropped), set(dropped)), (6, {0}))

# ------------------------------------------------------------------ 8. Track N teacher strength
g = {gm: p18[(f"N_A0_g{gm}_C1_b100", "unweighted")] for gm in ("000", "100")}
check("A0 gamma 1 -> 0, A/RAC1P", (round(g["100"][2], 4), round(g["000"][2], 4)), (0.0201, 0.0038))
check("A0 gamma 1 -> 0, AB/RAC1P", (round(g["100"][3], 4), round(g["000"][3], 4)), (0.0182, 0.0078))
jarms = [c for (c, w) in p18 if c.startswith("N_J_g000") or c.startswith("N_J_g001")]
check("every J-arm at gamma in {0,.01} equals ref_J (unweighted)",
      all(p18[(c, "unweighted")] == p18[("ref_J", "unweighted")] for c in jarms), True)


# ------------------------------------------------------------------ 8b. family-X cell counts (sensitive endpoints)
famc = collections.Counter()
for x in IX:
    if x["endpoint"].startswith("recovery"):
        f = x["comparison_family"]; famc[(f, "n")] += 1
        famc[(f, "better")] += x["candidate_wide_better"] == "True"; famc[(f, "worse")] += x["candidate_wide_worse"] == "True"
for f, n, b, w in [("N_C1_vs_L1", 128, 21, 0), ("N_C1_vs_L2", 128, 0, 0), ("N_gamma_vs_1", 288, 22, 1),
                   ("N_J_vs_A0_init", 192, 36, 0), ("E_C_vs_pca", 80, 28, 0), ("E_C_vs_random", 80, 25, 0),
                   ("E_C_vs_marginal", 80, 5, 0), ("E_C_vs_L", 80, 4, 0), ("E_C_vs_LX", 80, 5, 0),
                   ("grid_vs_J", 992, 5, 233)]:
    check(f"family X {f} (n, better, worse)", (famc[(f, "n")], famc[(f, "better")], famc[(f, "worse")]), (n, b, w))

# ================================================================== assets
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
ROWS = [("A0", "untouched weak channel"), ("ref_J", "strongest channel $J$ (matched audit)"),
        ("leace_A0", "LEACE on $A_0$"), ("splince_A0", "SPLINCE on $A_0$"), ("optnet16_C1", "OptNet-ARL adaptation"),
        ("leace_J", "LEACE on $J$ (diagnostic)"), None,
        ("E_J_C_k8", "\\textbf{panel}: coalition projection, $k{=}8$"), ("E_J_L_k8", "local control, $k{=}8$"),
        ("E_J_LX_k8", "expanded local, $k{=}8$"),
        ("E_J_C_k6", "\\textbf{panel}: coalition projection, $k{=}6$"), ("E_J_L_k6", "local control, $k{=}6$"),
        ("E_J_LX_k6", "expanded local, $k{=}6$"),
        ("N_J_g000_C1_b100", "\\textbf{panel}: neural nominees (bitwise $J$)"), None,
        ("E_A0_C_k2", "exploratory: coalition on $A_0$, $k{=}2$"), ("E_A0_L_k2", "local on $A_0$, $k{=}2$"),
        ("E_A0_LX_k2", "expanded local on $A_0$, $k{=}2$")]
def ptable(p, w):
    L = [r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
         r"release & role & $A$/SEX & $AB$/SEX & $A$/RAC1P & $AB$/RAC1P & residence \\", r"\midrule"]
    for row in ROWS:
        if row is None: L.append(r"\midrule"); continue
        c, note = row
        if (c, w) not in p: continue
        L.append(tt(c) + " & " + note + " & " + " & ".join(f4(x) for x in p[(c, w)]) + r" \\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"])
emit(T("study5_panel.tex"), ptable(p18, "unweighted"))
emit(T("study5_panel_pw.tex"), ptable(p18, "person_weighted"))
ROWS17 = [r for r in ROWS if r is None or (r[0] in {k[0] for k in p17})]
ROWS_BAK, ROWS = ROWS, [r for r in ROWS17 if r is not None or True]
emit(T("study5_2017.tex"), ptable(p17, "unweighted")); emit(T("study5_2017_pw.tex"), ptable(p17, "person_weighted"))
ROWS = ROWS_BAK

# contrast table (family P, candidate-wide) + the exploratory A0 rows (family X)
CT = [("E_J_C_k8", "ref_J", "P"), ("E_J_C_k8", "leace_J", "P"), ("E_J_C_k8", "splince_A0", "P"),
      ("E_J_C_k8", "E_J_L_k8", "P"), ("E_J_C_k8", "E_J_LX_k8", "P"),
      ("E_J_C_k6", "ref_J", "P"), ("E_J_C_k6", "splince_A0", "P"), ("E_J_C_k6", "E_J_L_k6", "P"),
      ("E_J_C_k6", "E_J_LX_k6", "P"), ("E_A0_C_k2", "E_A0_L_k2", "X"), ("E_A0_C_k2", "E_A0_LX_k2", "X")]
CE = ["recovery/AB/SEX", "recovery/A/RAC1P", "recovery/AB/RAC1P", "utility/same_residence"]
def cell(x):
    s = f"{x['est']:+.4f}\\,[{x['lo']:+.4f},{x['hi']:+.4f}]"
    if x["better"] and "utility" not in x.get("e", ""): s = r"\textbf{" + s + "}"
    return s
L = [r"\begin{tabular}{@{}l l c l l l l@{}}", r"\toprule",
     r"release & comparator & fam. & $AB$/SEX & $A$/RAC1P & $AB$/RAC1P & residence loss \\", r"\midrule"]
contrast_json = []
for l, r, fam in CT:
    rows = IP if fam == "P" else IX
    cells = []
    for e in CE:
        x = iv(rows, l, r, e, "unweighted"); x["e"] = e
        contrast_json.append({"left": l, "right": r, "family": fam, "endpoint": e, **{k: x[k] for k in ("est", "lo", "hi", "better", "worse")}})
        s = f"${x['est']:+.4f}$ {{\\tiny[{x['lo']:+.4f}, {x['hi']:+.4f}]}}"
        if e != "utility/same_residence" and x["better"]: s = r"\textbf{" + s + "}"
        if e == "utility/same_residence" and x["worse"]: s = r"\textbf{" + s + r"}$^\dagger$"
        cells.append(s)
    L.append(tt(l) + " & " + tt(r) + f" & {fam} & " + " & ".join(cells) + r" \\")
    if (l, r) == ("E_J_C_k6", "E_J_LX_k6"): L.append(r"\midrule")
emit(T("study5_contrasts.tex"), "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# accounting
emit(T("study5_accounting.tex"), "\n".join([
    r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
    r"track & nominal slots & fitted & distinct audited & exact duplicates \\", r"\midrule",
    f"N (neural factorial) & {N['expected']} & {N['fitted']} & {N['audited_distinct']} & {N['duplicate_of_audited_release']} (of untouched start) \\\\",
    f"E (projection) & {E['expected']} & {E['fitted']} & {E['audited_distinct']} & {E['duplicate_of_audited_release']} (constant full-span maps) \\\\",
    r"\midrule",
    f"total & {counts['nominal_total']} & {N['fitted']+E['fitted']} & {N['audited_distinct']+E['audited_distinct']} & {N['duplicate_of_audited_release']+E['duplicate_of_audited_release']} \\\\",
    r"\addlinespace", f"references \\texttt{{ref\\_A0}}, \\texttt{{ref\\_J}} (outside the 372) & 6 & -- & 6 & -- \\\\",
    f"stronger-attack interfaces (17 releases $+$ $H$ view, 3 anchors) & 54 & -- & -- & -- \\\\",
    f"2017 transported Track E units (6 releases, 3 anchors) & 18 & -- & -- & -- \\\\",
    r"\bottomrule", r"\end{tabular}"]))

# chronology v5
emit(T("study_chronology_v5.tex"), "\n".join([
    r"\begin{tabular}{@{}l p{3.5cm} l l p{5.9cm}@{}}", r"\toprule",
    r"study & construction & evidence & status & what it established \\", r"\midrule",
    r"1 & locked ACS 2017 transport & \texttt{349efa45} & confirmatory & coalition effect over both local controls; the $.001$ point rule, not interval noninferiority \\",
    r"2 & nonlinear penalty / rank & \texttt{c37807e4} & development & negative; penalty coordinate-dependent, diagnosed after evaluation \\",
    r"3 & invariant repair $+$ external baselines & \texttt{73903b7f} & development & negative; closed-form erasers beat the developed mechanism \\",
    r"4 & direct adversarial refinement from $A_0$ & \texttt{69e790af} & development & negative on both years \\",
    r"5 & $J$-initialised factorial $+$ coalition projection & \texttt{a56bcc7f} & development & \textbf{negative}: no prospectively selected release met the conjunction \\",
    r"6 & residual extension $[H_A,Z_J,R]$ (utility-first) & \emph{not integrated} & \textsc{pending} & no outcome read here; protocol review only (\S\ref{sec:pending}) \\",
    r"\bottomrule", r"\end{tabular}"]))


# claim-support table: v4 rows carried (Study 1-4 claims) + Study 5 rows generated from checked values
v4 = open(os.path.join(ROOT, "papers/pcrl_manuscript_v4/tables/claim_support.tex")).read().split("\\bottomrule")[0]
# v4 printed "4 cells" for the Study 3 repair; the ledger (V06) and text say 8 of 48 -> corrected (CORRECTIONS_V5 C6)
assert "& 4 cells significantly worse, 0 better &" in v4
v4 = v4.replace("& 4 cells significantly worse, 0 better &", "& 8 of 48 cells significantly worse, 0 better &")
k8u = iv(IP, "E_J_C_k8", "ref_J", "recovery/AB/SEX", "unweighted")
rows5 = [
 ("Study 5: competitive conjunction met by a panel release", r"\textbf{Not established}",
  "0 of 4 nominees, both weightings, ref\\_J or historical $J$", "Not a proof no competitive release exists; one objective, one selection rule"),
 ("$J$-initialised training improves on $J$", r"\textbf{Not supported}",
  r"Selection returned $J$ (step 0) in every $J$-arm at $\gamma\le.01$", "Other selection yardsticks untested"),
 ("Coalition projection on $J$ beats matched local projections", r"\textbf{Not established}",
  "No candidate-wide difference vs $L$, $LX$ (family P)", "Nonsignificance is not equivalence"),
 ("Projection of $J$ lowers coalition sex recovery", "Empirical, development",
  f"$AB$/SEX ${k8u['est']:+.4f}$ vs $J$, residence loss ${k8res['est']:+.4f}$ (unweighted)", "A trade-off, not dominance; ordinary LEACE on $J$ is as close as $\\pm0.0011$ (points)"),
 ("Coalition race effect on $A_0$ at $k{=}2$", "Exploratory (family X)",
  "Below both local controls, both weightings", "Not nominated; residence unresolved, not preserved; far from $J$"),
 ("Stronger attack confirms protection", r"\textbf{Not claimed}", "MLP[256,256,128] weaker than standard audit", "Survival untested"),
]
body = v4 + "".join(f"{a} & {b} & {c} & {d} \\\\\n\\addlinespace[1.5pt]\n" for a, b, c, d in rows5) + "\\bottomrule\n\\end{tabular}"
emit(T("claim_support_v5.tex"), body)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, MUTE, ACC, WARN = "#1b1b1b", "#8a8a8a", "#2a6f97", "#a4303f"
TITLES = ["$A$/SEX", "$AB$/SEX", "$A$/RAC1P", "$AB$/RAC1P"]
def frontier(p, w, stem, year):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    grid = [c for (c, ww) in p if ww == w and c.startswith(("E_J_", "E_A0_", "N_"))]
    for i, ax in enumerate(axs):
        for c in grid:
            ax.scatter(p[(c, w)][i], p[(c, w)][4], s=7, color=MUTE, alpha=.35, lw=0)
        for c, col, mk, lab in [("A0", INK, "s", "$A_0$"), ("ref_J", INK, "D", "$J$"), ("J", INK, "D", "$J$"),
                                ("leace_A0", ACC, "^", "LEACE $A_0$"), ("splince_A0", ACC, "v", "SPLINCE $A_0$"),
                                ("optnet16_C1", ACC, "P", "OptNet"),
                                ("E_J_C_k8", WARN, "o", "panel $k8$"), ("E_J_C_k6", WARN, "X", "panel $k6$")]:
            if (c, w) not in p or (c == "J" and ("ref_J", w) in p): continue
            ax.scatter(p[(c, w)][i], p[(c, w)][4], s=34, color=col, marker=mk, edgecolor="white", lw=.8, zorder=3, label=lab)
        ax.axvline(0, color=MUTE, lw=.6); ax.set_title(TITLES[i], fontsize=8)
    fig.supxlabel("added sensitive recovery over $H$ (nats; lower = less disclosure)", fontsize=8, y=-.1)
    axs[0].set_ylabel("residence gain over $H$ (nats)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(.5, -.26), fontsize=7)
    fig.suptitle(f"Study 5, {year}, {w.replace('_', '-')} seed means; grey = every other Study 5 release", fontsize=8, y=1.02)
    savefig(fig, stem)
frontier(p18, "unweighted", "study5_frontier", "2018 development")
frontier(p18, "person_weighted", "study5_frontier_pw", "2018 development")

# contrast forest (family P rows, candidate-wide intervals, both weightings)
FR = [("E_J_C_k8", "ref_J"), ("E_J_C_k8", "leace_J"), ("E_J_C_k8", "E_J_L_k8"), ("E_J_C_k8", "E_J_LX_k8"),
      ("E_J_C_k6", "ref_J"), ("E_J_C_k6", "splince_A0"), ("E_J_C_k6", "E_J_L_k6"), ("E_J_C_k6", "E_J_LX_k6")]
FE = ["recovery/AB/SEX", "recovery/A/RAC1P", "utility/same_residence"]
FT = ["$AB$/SEX recovery", "$A$/RAC1P recovery", "residence loss (+ = less capability)"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for j, e in enumerate(FE):
    ax = axs[j]
    for i, (l, r) in enumerate(FR):
        for dy, w, col, mk in [(-.15, "unweighted", INK, "o"), (.15, "person_weighted", ACC, "s")]:
            x = iv(IP, l, r, e, w)
            ax.plot([x["lo"], x["hi"]], [i + dy] * 2, color=col, lw=1.2)
            ax.scatter(x["est"], i + dy, color=col, marker=mk, s=14, zorder=3,
                       label=w.replace("_", "-") if (i == 0 and j == 0) else None)
    ax.axvline(0, color=MUTE, lw=.7); ax.set_title(FT[j], fontsize=8)
    if e == "utility/same_residence": ax.axvline(0.001, color=WARN, lw=.7, ls="--")
axs[0].set_yticks(range(len(FR))); axs[0].set_yticklabels([f"{l.replace('E_J_C_', 'C ')} vs {r}" for l, r in FR], fontsize=7)
axs[0].invert_yaxis()
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.06), fontsize=7)
fig.text(.5, -.1, "Candidate minus comparator, candidate-wide simultaneous 95% intervals (family P, m = 400). Dashed: .001 margin.",
         ha="center", fontsize=7)
savefig(fig, "study5_contrasts")

# chronology figure (compact)
fig, ax = plt.subplots(figsize=(7.2, 1.1))
st = [("1: locked 2017\ntransport", "confirmed\ncoalition effect", INK), ("2: nonlinear\npenalty / rank", "negative", "#d9d9d9"),
      ("3: invariant repair\n+ externals", "negative", "#d9d9d9"), ("4: adversarial\nfrom $A_0$", "negative", "#d9d9d9"),
      ("5: $J$-init +\ncoalition projection", "negative", "#d9d9d9"), ("6: residual\nextension", "pending\n(not integrated)", "white")]
for i, (n, s, col) in enumerate(st):
    ax.add_patch(plt.Rectangle((i, 0), .92, 1, facecolor=col, edgecolor=INK, lw=.6))
    tc = "white" if col in (INK,) else INK
    ax.text(i + .46, .68, n, ha="center", va="center", fontsize=6.5, color=tc)
    ax.text(i + .46, .27, s, ha="center", va="center", fontsize=6.5, color=tc, style="italic")
ax.set_xlim(-.05, 6); ax.set_ylim(-.05, 1.05); ax.axis("off")
savefig(fig, "study_chronology_v5")

# ---------------------------------------------------------------- records
json.dump({"pinned_commit": SHA, "evidence_commit": EVIDENCE_SHA, "sources_sha256": sources,
           "checks": checks, "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks),
           "conjunction_recomputed": conj, "contrasts": contrast_json,
           "k6_2017_residence_deficit_vs_J": {"unweighted": gap17u, "person_weighted": gap17w},
           "stress_conditions": stress_rel, "stress_neural_conditions_bitwise_J": neural_in_stress,
           "zero_new_acs_fits": True},
          open(os.path.join(REV, "STUDY5_VERIFICATION.json"), "w"), indent=1, default=str)
json.dump(manifest, open(os.path.join(REV, "ASSET_HASHES_STUDY5.json"), "w"), indent=1, sort_keys=True)
print(f"{len(checks)} checks, all ok; {len(manifest)} assets")

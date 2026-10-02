"""G-AAAI: classify each 'certificate failure' / guarantee in the AAAI 'Outputs Leak' paper as WITHIN
or OUTSIDE the stated scope of the certificate it is said to defeat, pulling the supporting numbers
from stored files. A nonlinear attacker reading above the bar does not refute a linear-scope theorem;
a reading above a bound that is stated for ALL attackers (FNF, FARE-on-binary, Prop 3 ceiling) is
within attacker scope, and then the question is whether the bound's preconditions held.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from F_common import DG, dg_json, save, sha

fnf, i1 = dg_json("results/fnf_gauntlet_hard_pca16.json")
fpca, i2 = dg_json("results/fairpca_gauntlet.json")
fare, i3 = dg_json("results/fare_gauntlet.json")
bg, i4 = dg_json("results/baseline_gauntlet.json")
ob, i5 = dg_json("results/obliviator_gauntlet.json")
dp, i6 = dg_json("results/dp_fullrank.json")
inputs = [i1, i2, i3, i4, i5, i6,
          {"location": "durable-guarantees@956f5c8:results/audit_scatter_rows.csv",
           "sha256": sha(os.path.join(DG, "results/audit_scatter_rows.csv"))}]
rows = list(csv.DictReader(open(os.path.join(DG, "results/audit_scatter_rows.csv"))))
n_att = sum(float(r["cert_r2_attacked"]) > 0.05 for r in rows)
n_rest = sum(r["cert_r2_at_rest"] != "" for r in rows)

fnf_margins = {str(c["gamma"]): round(c["best_bacc"] - c["bound"], 4) for c in fnf["certified"]}
mid = fare["cells"][1]
fare_mid = {"n_rows": len(mid["rows"]), "dp_ub_available": sum(r["dp_ub"] is not None for r in mid["rows"]),
            "dp_ub_zero": sum(r["dp_ub"] == 0.0 for r in mid["rows"]),
            "leaky_rows_macro_ovr_rep_t1": [(r["tag"], round(r["tier1_max"], 4), r["dp_ub"]) for r in mid["rows"]
                                             if r["dp_ub"] == 0.0 and r["tier1_max"] > 0.6]}
leace = [(c["cell"], round(b["tier1"]["closest"]["tier1_max"], 4)) for c in bg["cells"] for b in c["baselines"]
         if b["baseline"] == "LEACE"]
obl = [(c["cell"], round(c["their_orig_probes"]["unwanted"], 4), round(c["tier1"]["closest"]["tier1_max"], 4))
       for c in ob["cells"]]

items = [
 {"certificate": "Linear R^2 certificate (PCRL in-sample ridge one-hot R^2 <= 0.05 on the full matrix; "
                 "paper.tex:228-240, 303-305; appendix.tex:237)",
  "stated_scope": "linear least-squares predictors of s from h, fit and scored on the same rows (fit law / in-sample)",
  "failure_reported": "59/67 approved configurations recovered at AUC > 0.55 by XGB/MLP (paper.tex:56-59, 414)",
  "classification": "OUTSIDE scope (nonlinear attackers). 0/59 failures are linear-scope failures; no held-out linear "
                    "probe was stored for any of the 67, so the certificate was never tested inside its own scope "
                    "out of sample.",
  "extra": {"approved_with_attacked_R2_gt_tau": n_att, "at_rest_reading_stored": f"{n_rest}/67",
            "note": "'attacked R^2' = linear R^2 read after a trained rank-8 LoRA+ReLU adapter (falsification_attack/run_attack); "
                    "it is a nonlinear feature map, so its exceeding tau (41/67; HMDA projections 0.166-0.171, paper.tex:429-431) "
                    "is also outside the linear certificate's scope. paper.tex:429-430 calls this 'do not even pass the "
                    "certificate', which mislabels the attacked instrument as the certificate (build_audit_scatter_rows.py docstring "
                    "itself distinguishes them)."}},
 {"certificate": "LEACE (concept-erasure library; Table 1 'linear attackers')",
  "stated_scope": "no linear (affine) predictor of s beats a constant, on the fit distribution",
  "failure_reported": "Tier-1 .99+ (XGB/MLP/LoRA)",
  "classification": "OUTSIDE scope; paper says so (paper.tex:679-682). Eraser and LR head are fit on all rows "
                    "including attacker-test rows (baseline_gauntlet.py:309-323); the battery has no linear attacker, "
                    "so LEACE's in-scope guarantee is neither tested nor contradicted.",
  "extra": {"tier1_max_by_cell": leace}},
 {"certificate": "Fair PCA (Kleindessner et al.; official code)",
  "stated_scope": "projected group-conditional means equal (no linear classifier separates groups better than chance); "
                  "Sec 3.6 variant also matches covariances; binary s only",
  "failure_reported": "fails Tier 1 at every sweep point incl. 'certified' one (paper.tex:714-716): min 0.677-0.696",
  "classification": "OUTSIDE scope (nonlinear attackers); mean-equalisation invariant reproduced at 1.5e-16 "
                    "(fairpca_gauntlet.json gate). Race cells structurally out of scope (binary s).",
  "extra": {"gate": fpca["gate"]["invariant_mean_equalization"], "applicability": fpca["applicability"]}},
 {"certificate": "FARE (dp_ub: high-probability upper bound on demographic-parity distance of ANY downstream classifier)",
  "stated_scope": "binary s; all downstream classifiers on the finite (leaf-cell) representation; population bound w.h.p.",
  "failure_reported": "'reading 0.000 on the multiclass cell while three configurations leak race pairs at 0.603-0.610' (paper.tex:692-694)",
  "classification": "OUTSIDE the method's binary scope (5-class race; shipped multiclass/Bonferroni path returns dp_ub=0.000 "
                    "on every available config and null on the certified middle point). Inside scope (binary hard cell) the "
                    "certificate is coherent: dp_ub 0.185 vs measured 0.546. Two corrections: (i) 0.603-0.610 are MACRO OvR "
                    "Tier-1 AUCs of non-certified seed-0 sweep rows, not pairwise AUCs; (ii) docs/fare_results.md's scope "
                    "argument 'DP is not recovery' is not right for an all-classifier DP bound, which equals total variation "
                    "and does bound recovery; the valid scope argument is the binary-attribute restriction and the degenerate "
                    "multiclass extension.",
  "extra": fare_mid},
 {"certificate": "FNF (bound (1+Delta)/2 on balanced accuracy of ANY adversary, Delta = statistical distance under the "
                 "ESTIMATED densities)",
  "stated_scope": "all adversaries; conditional on density estimates (Theorem adds TV-to-truth epsilon); binary s; s at inference",
  "failure_reported": "measured balanced accuracy above FNF's own bound at 4/5 gammas by up to 0.174 (paper.tex:710-714)",
  "classification": "WITHIN attacker scope, OUTSIDE the precondition (density-estimate accuracy). Also run on a PCA-16 "
                    "deviation (full-dimension FNF inapplicable). The exceedance is visible in FNF's own logged adversary. "
                    "Not a refutation of the theorem; a demonstration that the computed certificate is unsound on this cell.",
  "extra": {"best_bacc_minus_bound_by_gamma": fnf_margins}},
 {"certificate": "Obliviator (official code; kernel-probe stopping criterion)",
  "stated_scope": "erasure against its RFF-kernel probes",
  "failure_reported": "Tier-1 .97+",
  "classification": "WITHIN its own diagnostic: its own stopping probe ('unwanted') reads 0.975-0.996 at the reported point, "
                    "so the method itself does not claim removal there; not a certificate failure.",
  "extra": {"[cell, their_unwanted_probe, tier1]": obl}},
 {"certificate": "LAFTR / DANN-scrub / VFAE (Table 1 'own adversary' / 'downstream fairness')",
  "stated_scope": "adversarial or MMD objectives against their own adversary/criterion; no recoverability certificate",
  "failure_reported": "Tier-1 failures; VFAE passes easy T1 only",
  "classification": "NO certificate to be within/outside; failures against higher-capacity adversaries are outside the "
                    "training adversary class. Official LAFTR is a binary-attribute method run on 5-class race "
                    "(laftr_official.py), outside its scope on the HMDA cells."},
 {"certificate": "Prop 1 (paper) - linear maps cannot raise linear R^2",
  "stated_scope": "population / in-sample OLS, Sigma_h > 0",
  "failure_reported": "none; used as instrument-blindness argument",
  "classification": "Algebraic identity (variational multiple R^2); correct in its scope; exact only for unregularised "
                    "in-sample OLS (ridge certificate differs by 1.7e-5, appendix.tex:76-83)."},
 {"certificate": "Prop 2 (paper) - second-moment statistics do not identify recoverability",
  "stated_scope": "population existence: for given class moments a matching distribution exists where a Bayes attacker "
                  "nearly fully recovers (asymptotic in separation/sigma)",
  "failure_reported": "used to argue 'no sound certificate computed from these statistics alone can be useful' (paper.tex:271-273, 761-763)",
  "classification": "Population existence claim; asymptotic regime not reached (dmin/sigma 0.016-0.171, appendix caption). "
                    "Main-text 'cannot be useful' exceeds what is shown (witness exceeds bar only at the 3 Tier-1 points). "
                    "Statement conditions on task label Y while the claim concerns s. Note the paper's own Tier-2 LRT is a "
                    "second-moment (Gaussian class-conditional) reader, so by Prop 2 Tier-2 passes are measurements, not certificates."},
 {"certificate": "Prop 3 / Corollary (paper) - clipped full-rank Gaussian channel is mu-GDP, AUC <= Phi(mu/sqrt 2)",
  "stated_scope": "CLIPPED release Pi_C(phi(x)) + sigma Z, one release per input, frozen non-private map, attacker whose "
                  "only s-informative input is the release; population AUC",
  "failure_reported": "none claimed; but Table 1 'Ours, full-rank' checkmarks and 'the certified full-rank channel (Prop 3) under "
                      "Tier 2' (paper.tex:583-584)",
  "classification": "The Table 1 / Fig. 4 full-rank points are UNCLIPPED (mi_ceiling.py:230-256: P = BN(E(x)) + sigma*randn, no "
                    "projection), so Prop 3 does not cover them; their Tier-2 pass is empirical. The clipped variant "
                    "(run_dp_fullrank.py, Table dp) keeps ~0% utility at eps<=3 and was never attacked by the battery. "
                    "Ceiling below 0.55 only at mu=0.103 (0.529).",
  "extra": {"dp_fullrank_keys": list(dp.keys())[:12]}},
 {"certificate": "Stadler et al. 2024 (population leakage floor)",
  "stated_scope": "population statement about representations with high task utility",
  "failure_reported": "'measured floors ... approach the leakage floor established by Stadler' (paper.tex:472-474)",
  "classification": "Population claim used interpretively; finite-sample, in-sample-utility attack readings neither confirm nor "
                    "refute it; 'approach' is not a measured comparison against a computed Stadler bound (none computed)."},
]
p = save("G_aaai_scope_classification.json", {"inputs": inputs, "items": items})
print(p)
print(n_att, n_rest, fnf_margins, fare_mid, leace, obl)

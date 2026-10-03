"""Build ORIGINAL_VS_CORRECTED.csv from the correction outputs (numeric) plus the text ledger (wording)."""
import csv, json
from pathlib import Path
HOME = Path.home()
C = HOME / "PCRL_eval_cache_private" / "oar_v1" / "corrections"
PKG = Path(__file__).resolve().parents[1]
OLDC = "70f978ffc0afff55ecdf49cf1a74908cc3db5f49"
rows = []
for r in json.load(open(C / "C1_seed_pairing.json")):
    if r.get("status") != "CORRECTED":
        continue
    rows.append({"item": "C1_seed_pairing", "kind": "deterministic arithmetic (implementation error)", "location": f"MATCHED_COMPARISONS.csv @{OLDC[:7]}",
                 "id": r["id"], "original": r["original_point"], "corrected": r["corrected_point"],
                 "corrected_interval90": f"[{r['corrected_lower90']:.6f}, {r['corrected_upper90']:.6f}]",
                 "membership": f"encoder seeds {r['encoder_seeds_D']} on both sides; D units {r['n_D_units']}, A units {r['n_A_units_paired']}",
                 "in_original_primary_family": "no (exploratory, Tier-2 E3)"})
c2 = json.load(open(C / "C2_rho_stable.json"))
for r in c2["intervals"]:
    if r["original_point"] is None or abs(r["corrected_point"] - r["original_point"]) <= 1e-9:
        continue
    rows.append({"item": "C2_stable_rho1sq", "kind": "deterministic arithmetic (numerical cancellation)", "location": f"RECOVERY.csv @{OLDC[:7]}",
                 "id": r["id"], "original": r["original_point"], "corrected": r["corrected_point"],
                 "corrected_interval90": f"[{r['corrected_lower90']:.6f}, {r['corrected_upper90']:.6f}]",
                 "membership": "unchanged", "in_original_primary_family": "no (exploratory)"})
for m in json.load(open(PKG.parents[0] / "combined_matched_removal_benchmark_v1/verification/mc_exceedance_reference.json")):
    rows.append({"item": "C3_monte_carlo", "kind": "Monte Carlo simulation error (NOT corrected)", "location": "exploratory bound, B=2000",
                 "id": m["stat"], "original": m["runner_B2000"], "corrected": "",
                 "corrected_interval90": f"B=20000 reference quantile {m['q_B20000']:.6f}; runner is {m['runner_minus_q_in_sd2000']:+.2f} SD of the B=2000 quantile",
                 "membership": "unchanged", "in_original_primary_family": "no (exploratory)"})
TEXT = [
 ("W4_noise_lift", "HANDOFF.json headline", "Noise at sigma* costs 40% (Adult) to 100% (HMDA) of probe lift.",
  "At sigma*, the refitted probe retains about 40% of its lift over the constant predictor on Adult (0.782 vs 0.831, constant 0.749; about 60% lost) and 0% on HMDA (0.886 = constant 0.886; 100% lost)."),
 ("W5_guarantee_transfer", "RESEARCH_DECISION.md l.87, l.159, l.167-168; ADVISOR_BRIEF.md l.23, l.54; HANDOFF.json headline",
  "The linear guarantee transfers to unseen rows / transfers it to held-out rows / generalises",
  "Measured on one held-out assessment split per dataset: held-out top canonical correlation rho1^2 under target LEACE 0.0006 (Adult, 90% interval to 0.013) and 0.0001 (HMDA, to 0.033), against 0.025 and 0.21 untreated. This is an empirical held-out measurement on already-used rows, not a distribution-free guarantee for future people; rank-deficient seeds leave the out-of-support component untouched."),
 ("W6_auc_not_information", "ADVISOR_BRIEF.md l.35", "The outputs carry 0.18-0.24 AUC more information than the true label would.",
  "The outputs-only attacker reaches 0.18-0.24 higher measured AUC than the label-only reference."),
 ("W7_no_more_pairs_claim", "RESEARCH_DECISION.md l.190-192; ADVISOR_BRIEF.md l.62; HANDOFF.json recommendation",
  "More pairs, seeds or noise levels ... would not change the decision / The 14 pairs agree / Do not scale",
  "Removed. The observed 14 pairs agree on these encoders and rows; additional seeds, pairs or datasets could differ. Prioritising a different comparison next is a resource decision, not a claim that more data cannot change the conclusion."),
 ("W8_affine_lookup", "RESEARCH_DECISION.md l.114-115; VALIDATION.md l.94-95",
  "HMDA fair_lending representations take only about 4.7K-6.7K distinct values ... An affine eraser cannot change a lookup on those values.",
  "Observed separately: (i) the fair_lending representation has few distinct rows (assessment: 447 / 315 / 3,690 distinct among 4,778 rows for seeds 0/1/2; 6,703 / 4,742 / 54,315 among all 77,408 rows); (ii) on the assessment rows every LEACE map (B and C) was collision-free: the distinct-row partition is identical before and after. A projection can merge distinct rows in general, so affineness alone implies nothing; the identical recovery on seed 0 is consistent with the observed collision-free partition, not proven by it."),
 ("W9_exposure_roles", "RESEARCH_DECISION.md l.177; HANDOFF.json not_done/limits wording",
  "17 Adult and 42 HMDA assessment rows duplicate training records",
  "17 Adult and 42 HMDA TEST-SPLIT rows (all three roles) have a record equal to an encoder-training record: Adult attacker_fit 6 / attacker_val 4 / assessment 7; HMDA 21 / 7 / 14. PROTOCOL.md stated this correctly."),
]
for item, loc, orig, corr in TEXT:
    rows.append({"item": item, "kind": "wording / reporting (no number changed)" if item != "W4_noise_lift" else "wording (inverted percentage)",
                 "location": f"{loc} @{OLDC[:7]}", "id": "", "original": orig, "corrected": corr, "corrected_interval90": "",
                 "membership": "", "in_original_primary_family": "no"})
cols = list(rows[0].keys())
with open(PKG / "ORIGINAL_VS_CORRECTED.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(len(rows), {k: sum(r["item"] == k for r in rows) for k in sorted({r["item"] for r in rows})})

# Closing statement of the statistical reviewer (campaign `pcrl_dual_design_release_v1`, rules DDR1-rules-v1)

**Status.** Writing started 2026-10-21T19:55:01Z (`date -u`); the completion time is in section 5. Author: the statistical reviewer
(the role that wrote `RULES.md`, `RULES.json`, `ddr/rules.py`, `ddr/test_rules.py` and `REPORT_PREREGISTRATION.md`), returning as a
fresh agent that relied on the files only. I read no label, ran no real-data computation and modified no file other than this one.
My only computations were invented-data checks: `python3 -m ddr.test_rules` (22 of 22 PASS, exit 0, 0.21 CPU-s reported) and a check
of the frozen helpers on invented strength inputs (section 1.4).

**Inputs read** (records and code only): `roles/coordinator/CLOSING_REVIEW_BRIEF.md`; `PROTOCOL.md`, `PROTOCOL_AMENDMENT_1.md`,
`PROTOCOL_DEVIATIONS.md` (deviations 1-12, deviation 12 as written at 19:47Z); every card in `cards/`; `claims/{DA_REG,DB_REG,DB_R1,DA_R1}/DECISIONS.json`;
`claims/PLAN_*.json`; `claims/{DA_REG_D2R,DB_REG_D2R,DB_R1_D2R}/D2R.json`; `claims/STRONGEST.json`, `claims/CAMPAIGN_SUMMARY.json`,
`claims/BESIDE_*.json` (the versions rewritten at 19:47:20Z), `claims/ZTILT_R1_READING.json`, `claims/DB_R2_TRIGGER_*.json`;
`da/R1_TRIGGER_B2.json`, `db/R1_TRIGGER.json`, `db/ZTILT_R1_V*.json`, `db/ABLATION_R1_V*.json`, the SBC entries of `db/FIT_R1_V*.json`;
`rich/RICH_*.json`; `dev/UTILITY_LEDGER.csv`; `TABLES.md` (regenerated 19:43-19:46Z); `roles/verifier/VERIFIER_REPORT.md`,
`roles/verifier/VERIFIER_ADDENDUM_REPORT.md` and `roles/verifier/records/CHECKS_ADDENDUM__2026-10-21T194410Z.json`;
`roles/challenger/REVIEW_DESIGNS.md` (finding 1); `ddr/claims.py` (what the builder passes to the rules) and `ddr/rules.py`. The
coordinator's drafts were read as saved at 19:44:27Z (`RESEARCH_DECISION.md`) and 19:44:41Z (`START_HERE.md`, `ADVISOR_BRIEF.md`,
`HANDOFF.md`); the corrections in section 3 quote those versions.

Sign conventions are the registered ones: pair benefit = comparator AUC - candidate AUC on the 8 registered cells (positive favours
the candidate), reported as mean (se_agg); "resolved" = mean - 2 se_agg > 0 with both version means > 0.

## 1. Do the decision labels follow from the frozen rules and the records?

**Yes, for every reading.** All four DECISIONS.json records, the three D2R records, the strongest-release record and the campaign
summary follow from `ddr/rules.py` applied to the inputs the claims builder passed, and the verifier replayed every one of them
exactly with separate code (first report V4; addendum A3-A5). The frozen rules are intact: `RULES.json` hashes to e4b21944...,
which equals the `rules_sha256` in every DECISIONS.json; `ddr/rules.py` and `RULES.json` were last committed in the registration
commit 8f736cbe3; the invented-data tests pass 22 of 22.

### 1.1 Inputs checked, per reading

| Reading | Registration status (card first commit vs first MODEL_SELECT read 18:09:16Z) | Paths given (aliases) | Q* (pooled MODEL_SELECT total-NLL, CAND - point) | Grid validity | Labels |
|---|---|---|---|---|---|
| DA_REG | registered: 18:06:14Z, before the read | SBC [DA_SBC_LO, DA_SBC_HI], SPEND [DA_SPD_LO, DA_SPD_HI]; alias_of {} | DA_SBC_HI (-0.0074), DA_SPD_HI (-0.0074) | READER_SEEDS_EFFECTIVE (strict NRM1 too; 0 equalities; registered reader and bootstrap seeds) | D1 TASK_FEASIBLE; D2 PARTIAL_SBC_FRONTIER_ONLY; D3 NOT_SUPERIOR; D4 NOT_MET; class TASK_ONLY_FEASIBLE_NO_PROTECTION |
| DB_REG | registered: 18:06:14Z | SBC [DB_SBC_LO, DB_SBC_HI], SPEND [DB_SPD_LO, DB_SPD_HI]; {} | DB_SBC_HI (-0.0043), DB_SPD_HI (-0.0071) | same | D2 NOT_SEPARATED; otherwise as DA_REG |
| DB_R1 | ADAPTIVE_R1: 18:15:30Z, after the utility read, before any grid output | SBC refitted [DB_SBC_R1_LO, DB_SBC_R1_HI]; SPEND reused [DB_SPD_LO, DB_SPD_HI] (PROTOCOL 8 reuse rule); {} | DB_SBC_R1_HI (-0.0008), DB_SPD_HI (-0.0025) | same | ADAPTIVE_R1__ prefix on every label; D2 PARTIAL_TWIN+SPEND_FRONTIER_ONLY; flag POINT_FEASIBLE_NOT_RESERVE_QUALIFIED |
| DA_R1 | ADAPTIVE_R1: 19:13:36Z, after DA_REG's D2 | SBC refitted [DA_IPW_SBC_LO, DA_IPW_SBC_HI]; SPEND reused [DA_SPD_LO, DA_SPD_HI]; {} | DA_IPW_SBC_HI (-0.0038), DA_SPD_HI (-0.0077) | same | ADAPTIVE_R1__NOT_SEPARATED; targeting fails in version 1 against both brackets; flags READER_SENSITIVE_ENHANCED_* (reported only) |

`controls_not_run` is false and `every_nonzero_strength_alias_of_twin` false in every plan, as the records require; every path has a
point costing at least CAND's, so no NO_BRACKET flag arises; no cheaper path point dominates any CAND ((c3) lists empty); the reader
budget flag is READER_BUDGET_COMPARABLE for all nine references, so the credit rule changes nothing; `joint` and `novelty` are null,
so D5 = NOT_ASSESSED_MISSING_CONTROLS and D6 = NOT_CLAIMED by rule. The campaign summary counts K = 4 D2 looks (2 registered,
2 ADAPTIVE_R1) with D2 met in none; DB's R2 trigger did not hold on either reading and the family stop rule did not fire.

### 1.2 Record defects that change no label (to be disclosed; see C19)

1. **DB_R1's D1 record** has `fitted_strengths: {}` and lists DB_PTC_R1 under `ignored_zero_strength_entries`. Cause: the plan gives
   the strength as a per-version pair [0.35, 0.65], which the frozen helper does not read as a nonzero number (the same cause as
   deviation 8). Harmless here: the list is consulted only when CAND fails G1, and DB_PTC_R1 passes G1. It is not yet disclosed.
2. `CAND_STRENGTH_DIFFERS_BY_VERSION` for DA_R1 (lambda* 1.0 / 0.75) is missing from the plan: disclosed as deviation 12 (b).
3. `d2r.release` is null in the three D2R records (the builder's stub passes `config_id`, the rule reads `selected`): deviation 12 (c).
   The releases are DA_STD, DB_PTC and DB_PTC_R1.
4. The beside records mixed cell sets for the single-view fields: corrected at 19:47Z, deviation 12 (a). The pair-view fields, which
   are the only ones quoted with uncertainty, were always registered-cell aggregates.

### 1.3 Interpretation caveats inherited by the labels (they follow the rules; they limit what the labels mean)

- **DA_REG's only passed comparison is weak.** Both DA SBC points are more recoverable than the twin at the point (pair AUC 0.6604 and
  0.6603 against 0.6506; verifier observation 2). DA_STD's +0.0273 (0.0104) over DA_SBC_HI therefore partly reflects a control path
  that adds recoverable signal, while DA_STD against the twin itself is unresolved (+0.0175, 0.0090).
- **SBC constructions differ by version.** DA: PSEUDO2 with gamma 2 (v0) and 1 (v1). DB, registered and R1: permuted SEX in v0, where
  Cov(q, s) = 1.48e-4 is just above the 1e-4 degeneracy threshold, and PSEUDO2 (gamma 1) in v1. The aggregate pools the two versions.
- **Power was far below plan.** The rules assumed se_agg about 0.005; the realized D2 values were 0.0087-0.0178, so the 2-SE threshold
  was 0.017-0.036 and "not separated" is weak evidence of absence.
- **Equal evaluation effort was breached once**: DB_ZTILT had no utility record before the grid (verifier V2.5, procedural FAIL;
  deviation 5). No decision uses it.

### 1.4 The strongest-release choice and its strength mapping (deviation 8)

The rule (RULES.md 8) was applied correctly: DA_STD for DA (x = +0.0175 against DA_IPW's +0.0072) and DB_PTC_R1 for DB (+0.0239
against DB_PTC's +0.0164), on registered cells, from pools of the families' own operation configurations (controls and ablations
excluded), all eligible. **The mapping alpha_I + alpha_O cannot have changed the choice.** In the frozen rule a strength enters only
(i) the nonzero test and (ii) the tie-break, which applies only when |dx| <= 1e-12; here dx = 0.0075. Any positive scalar (the sum,
the larger component, either component) gives the same selection; I checked the helpers on invented inputs: the pair reads as not
nonzero, the sum reads as nonzero. Two consequences for the documents: STRONGEST.json's "strength 1.0" for DB_PTC_R1 is
alpha_I + alpha_O and must never be read as alpha = 1 (full cancellation); and DB_PTC's single shared alpha 0.35 is not on that
scale (its sum would be 0.70), which is immaterial for the same reason. Deviation 8 is acceptable.

### 1.5 D2R minimum sets and refit effectiveness

Each D2R minimum set is {release, TWIN, the release's Q* on SBC and SPEND}: DA_STD with DA_SBC_HI / DA_SPD_HI; DB_PTC with DB_SBC_HI /
DB_SPD_HI; DB_PTC_R1 with DB_SBC_R1_HI / DB_SPD_HI. Each release is its reading's CAND, so copying the brackets from that reading's
D2 equals the `brackets()` rule. Every refit is REFIT_GENUINELY_EFFECTIVE: R1-R4 assignment hashes distinct, SCORE household overlaps
0.28-0.31, both seed pairs (k0 / k1, k2 / k3) judged separately with 0 equalities, registered reader and bootstrap seeds (verifier
A4.refit). The 24 robustness cells (R3, R4 x k0-k3; R1, R2 x k2, k3) were present for every decisive comparator (n_eff 23-24).
**Under RULES.md 8 the family D2Rs are DA_REG_D2R (DA: ROBUSTNESS_NOT_CONFIRMED) and DB_R1_D2R (DB: ROBUSTNESS_CONFIRMED).**
DB_REG_D2R (NOT_CONFIRMED) was computed while DB_PTC was DB's only operation configuration; it is an extra, descriptive reading of
the registered candidate (deviation 12 (d)).

### 1.6 Deviation 7 (DA R1 outside its declared trigger): acceptable, labelled ADAPTIVE_R1

DA_STD_v1's trigger (b) names `NOT_SEPARATED`, and DA_REG is `PARTIAL_SBC_FRONTIER_ONLY`, so the declared trigger did not hold. PROTOCOL 7
does not make a declared trigger the only route to a revision: a revision must address a named, recorded failure, have its own card
committed before its fit, and be read under the same rules with an adaptive tag. All of that holds (failure: D2 not met, with the
MODEL_FIT residual-propensity record `da/R1_TRIGGER_B2.json`; card 00eef8eb7 before its fit; tag ADAPTIVE_R1; counted as one of DA's
two revisions and as one D2 look). It must be described as **a revision run outside its card's declared trigger (deviation 7),
after the registered DA privacy outcome**, so it is adaptive to a privacy outcome, not only to the utility read. Its negative
outcome is reported like any other look and is not evidence about the trigger.

### 1.7 Deviation 9 (DB_PTC_R1's ablation and DB_ZTILT_R1): acceptable, descriptive, beside D2

RULES.md 8 requires the component ablation and the refits of each family's strongest feasible nonzero release, and PROTOCOL_AMENDMENT_1
finding 2 made ZTILT DB's required release-only control. Once the registered rule selected DB_PTC_R1, both were required, although
card DB_PTC_R1 had said they would not be refitted. Each was added under its own card committed before its fit, with its grid bank
written before any of its output, within the 48-configuration budget. Adding a control after a favourable-looking reading can only
take claims away. Required labelling: **both are part of the ADAPTIVE_R1 reading, descriptive and beside D2; neither changes or
upgrades D2; both inherit the winner's curse of the strongest-release choice.** DB_ZTILT_R1's declared reading,
`NOT_RESOLVED_BELOW_ZTILT`, stands as declared. Its overshoot (MODEL_FIT spend 2.08x / 1.73x the release's, because the registered
scale grid jumps from k = 1, at 0.90x / 0.75x, to k = 1.5) is a design limit of the registered grid. It is disclosed beside the
reading. It is never used to set the reading aside, and the reading is never read as "the gain is fully explained".

## 2. Required headline (to be quoted verbatim)

> This is development evidence on previously and adaptively used MODEL_SELECT records of the admitted ACS 2019 Florida cohort (8,001
> people), for the declared readers only: it is not confirmation, and lower recovery for these readers is not protection against
> every reader. Two independently designed release families, which belong to one idea family (SEX-standardized posteriors, STD, and
> propensity-tilt cancellation, PTC, both subtract the propensity-proportional part of the released task posteriors, so their joint
> negative closes one idea, not two), met the original utility limits at the point, but neither reduced the readers' recovery of the
> recorded SEX attribute beyond its matched controls: D2 was met in 0 of 2 registered family readings and in 0 of 2 non-registered
> (ADAPTIVE_R1) readings, 4 D2 looks in total; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under
> six moderate NRM1 known laws and about 8% when one version or split dominates, and with 4 looks the chance that at least one passes
> by chance is at most 4 times the per-look rate. Registered readings (linked-pair benefit = comparator AUC - candidate AUC, mean
> (se_agg) over the 8 registered cells): DA_STD is PARTIAL_SBC_FRONTIER_ONLY, with +0.0175 (0.0090) over the equally trained twin and
> +0.0137 (0.0174) over its SEX-blind spending (SPEND) bracket, both unresolved; its one passed comparison, +0.0273 (0.0104) over its
> SEX-blind-target (SBC) bracket, is against controls that are themselves more recoverable than the twin at the point. DB_PTC is
> NOT_SEPARATED: +0.0164 (0.0107), +0.0173 (0.0171) and +0.0170 (0.0094) over the twin, SBC and SPEND brackets. It is not resolved below its
> release-only control DB_ZTILT (+0.0170 (0.0121)). Both are TASK_ONLY_FEASIBLE_NO_PROTECTION. Neither registered release is robust to
> classifier refitting (D2R ROBUSTNESS_NOT_CONFIRMED; for DB_PTC an extra, descriptive reading). Beside them, never as family results:
> DA's revision (IPW-STD, run outside its declared trigger, deviation 7) is ADAPTIVE_R1__NOT_SEPARATED (+0.0072 (0.0087) over the
> twin). DB's revision DB_PTC_R1 (recipient-split strengths 0.35 / 0.65) was afterwards selected, on the same registered cells, as DB's
> strongest feasible release, so it carries the winner's curse. It is ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY: resolved over
> the twin (+0.0239 (0.0101)), over the SPEND bracket by a 2-SE margin of 0.0001 (+0.0245 (0.0122)), and not separated from its SBC bracket
> (+0.0265 (0.0139)). On refit cells (new household splits and reader seeds of the same people; descriptive, D2 unchanged) its benefits
> over the twin, SBC and SPEND brackets stay resolved (+0.0219 (0.0103), +0.0231 (0.0091), +0.0189 (0.0091); ROBUSTNESS_CONFIRMED).
> By the reading declared in card DB_ZTILT_R1 before that control's fit, it is not separated from the release-only control DB_ZTILT_R1, which tilts by
> P(SEX | the twin's released vectors): +0.0128 (0.0081) on the registered cells and +0.0066 (0.0095) on the refit cells. So its gain
> is not separated from release-only information; that control overspent (0.0169 against 0.0099 nats of MODEL_SELECT total NLL).
> DB_PTC_R1 is point-feasible but not reserve-qualified. No reading is competitive with the nine inherited references (D3
> NOT_SUPERIOR in all four), and none meets the 0.020 material criterion (D4 NOT_MET; the largest values, DB_PTC_R1's, are +0.0110
> (0.0135) over NEURAL_RAW and -0.0058 (0.0151) over H_LJ). The +0.007 to +0.024 differences against the twin are a smaller
> result against a weaker baseline, not the criterion. No release is a method, candidate or component; joint design (D5) was not
> assessed and novelty (D6) is not claimed.

## 3. Required corrections to the coordinator's drafts

### 3.0 DB_PTC_R1: what may and may not be said about the combination

Its registered-cell D2 is `ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY` (SBC frontier missed: +0.0265, se 0.0139, lower 2-SE bound
-0.0013). Its descriptive D2R on refit cells is `ROBUSTNESS_CONFIRMED` against the twin, the SBC bracket and the SPEND bracket.

**May be said** (each with its qualifier in the same sentence):
1. On the registered cells, for the declared readers, its linked-pair benefit is resolved over the twin (+0.0239, se 0.0101, lower
   bound +0.0037). It passes the SPEND frontier comparison with a lower bound of +0.0001. It is not separated from its SBC bracket.
   D2 is not met, and no partial outcome is a claim.
2. On refit cells (R3 / R4 x k0-k3 and R1 / R2 x k2 / k3: new household splits and reader seeds of the same 8,001 people), the
   benefits stay resolved against the twin (+0.0219, se 0.0103; lower bound +0.0013), the SBC bracket (+0.0231, se 0.0091) and the
   SPEND bracket (+0.0189, se 0.0091; lower bound +0.0007). So the direction found on the registered cells is not an artefact of those
   particular reader fits and splits. This is descriptive robustness to classifier refitting.
3. **Always in the same breath as item 2:** against the release-only control DB_ZTILT_R1 the benefit is +0.0128 (se 0.0081) on the
   registered cells and +0.0066 (se 0.0095) on the refit cells, neither resolved. By the reading declared in card DB_ZTILT_R1 before
   its fit (NOT_RESOLVED_BELOW_ZTILT), the gain is not separated from release-only information, PRC's failure repeated. The control
   overspent (MODEL_SELECT total NLL 0.0169 vs 0.0099 nats). That makes the comparison a bracket from above, as the inherited frontier
   rule reads every control: if the control's recovery falls with spend, a spend-matched point would have been weaker. No such point
   exists, no interpolation across ZTILT laws is licensed, and the declared reading stands. Per view the control matches DB_PTC_R1
   (income AUC 0.5329 vs 0.5333, occupation 0.6177 vs 0.6166); the difference is on the linked pair (0.6395 vs 0.6267).
4. It was chosen as DB's strongest feasible nonzero release by the registered rule on the same registered cells (winner's curse). Its
   refit estimates are lower than its registered ones against all three comparators (0.0219 vs 0.0239, 0.0231 vs 0.0265, 0.0189 vs
   0.0245). It is one of 4 D2 looks, all at one idea family, and it is ADAPTIVE_R1, reported beside DB's registered reading.

**May not be said:**
- that DB_PTC_R1 separates from, beats or dominates its SBC control; that it "meets D2 on the refit cells", "after refitting" or "in
  aggregate"; that the refit "resolves" or "overturns" the SBC miss. D2R never changes D2 (RULES.md 8). D2's SBC conjunct is read on
  the registered cells and needs the full frontier comparison: no secondary metric resolved higher, the recipient gate at Q*, (c3)
  and targeting. D2R is a pair-benefit screen only.
- that the SBC miss "is only a power problem", or that the effect "would separate with more data". A miss at the 2-SE threshold is
  a non-separation: evidence neither of equality nor of a hidden effect. The SBC bracket is already within 0.0008 nats of the
  release's cost, so a finer control grid cannot change the comparison. A new reading on these people would be a fifth adaptive look;
  a new population would be confirmation, which was not authorized.
- "replicated", "confirmed" in the confirmation sense, "independent evidence", or "robust" standing alone. The refit cells are new
  splits and seeds of the same people; for k2 / k3 the R1 / R2 splits are the registered ones; the versions share every training row.
- "stable across reader seeds". In one block (version 0, split R1) the benefit over the twin is about +0.034 with seeds k0 / k1 and about
  +0.003 with k2 / k3 (block means in `claims/ZTILT_R1_READING.json`). The aggregate is resolved, but not every block is stable.
- "DB's result", "the family's result" or "the best method". The registered DB reading is DB_PTC, NOT_SEPARATED.
- that the gain "needs propensity information beyond the release", or "removes information". DB_PTC_R1 is not separated from
  DB_ZTILT_R1, and the release writes the propensity into the vector (section 4.1). Equally, not that the gain "is fully explained by"
  or "is reproduced by" release-only information. The pair comparison is unresolved with a positive point estimate in both versions,
  and the control overspent. The card's wording "reproduced" holds per view only.
- "a component", "a candidate", "a method", a protection or privacy claim, "competitive", or "meets 0.020".

### 3.1 Numbered corrections

**C1 (all four documents; REQUIRED by PROTOCOL_AMENDMENT_1 finding 1).** No draft says that both families test one idea family. The
amendment requires every campaign-level statement to say it, and the challenger's finding 1 requires the campaign summary to say it.
Add, in RESEARCH_DECISION "Decision", START_HERE "Result", ADVISOR_BRIEF "Result in one line", and HANDOFF "What can be said":
> "Both families test one idea family: STD and PTC each subtract the propensity-proportional part of the released task posterior
> (the revisions IPW-STD and recipient-split PTC stay inside it), so this joint negative closes one idea, not two."

**C2 (headline placeholder).** RESEARCH_DECISION "[PENDING: quoted verbatim from ...]": replace with section 2 above, verbatim.
START_HERE, ADVISOR_BRIEF and HANDOFF point to it; keep those pointers.

**C3 (registered first; "strongest result").** RESEARCH_DECISION: "The strongest result is DB's adaptive revision, DB_PTC_R1. For
it:"; START_HERE: "The strongest result is DB's adaptive revision, DB_PTC_R1, with recipient-split tilt strengths:". Why: RULES.md 7
makes each family's registered reading its headline, with adaptive readings beside it. "Strongest result" is not a rule term, and it
puts a winner's-curse selection first. Replace with:
> "Registered readings, the family results: DA_STD is PARTIAL_SBC_FRONTIER_ONLY and DB_PTC is NOT_SEPARATED, both
> TASK_ONLY_FEASIBLE_NO_PROTECTION. Beside them, never as a family result, is DB's adaptive revision DB_PTC_R1 (ADAPTIVE_R1;
> recipient-split strengths 0.35 / 0.65). The registered rule selected it afterwards, on the same registered cells, as DB's strongest
> feasible nonzero release (winner's curse; descriptive):"

**C4 (the SPEND pass and the scope of the twin reduction).** RESEARCH_DECISION: "It lowers linked-pair recovery against the equally
trained twin by a resolved 0.024 AUC and dominates the SEX-blind spending path."; START_HERE: "It lowers linked-pair recovery against
the twin by a resolved 0.024 AUC (se 0.010)." and "It dominates the SEX-blind task-spending control path."; ADVISOR_BRIEF: "it dominates
the spending path: +0.0245 (0.0122);". Why: the SPEND pass clears the 2-SE threshold by 0.0001, and "dominates" without the margin
reads as a clear win. Both statements also hold only for the declared readers on the registered cells. Replace with:
> "For the declared readers on the registered cells, its linked-pair AUC is below the twin's by 0.0239 (se_agg 0.0101; lower 2-SE
> bound +0.0037). It is below the SPEND bracket DB_SPD_HI's by 0.0245 (se_agg 0.0122), which passes the SPEND frontier comparison with
> a lower 2-SE bound of only +0.0001."

**C5 (refit confirmation must carry the ZTILT_R1 refit and its limits).** RESEARCH_DECISION: "On refit cells (new household splits
and reader seeds, same people), its benefits against the twin and both bracketing controls are resolved. That reading is
descriptive." (START_HERE has the same sentence with "new splits"); HANDOFF: "On additional refit cells ... its benefits against the twin and both bracketing controls were resolved.";
ADVISOR_BRIEF question 3, "DB_PTC_R1: ROBUSTNESS_CONFIRMED on the refit cells ...". Why: section 3.0, items 2-4. Replace with:
> "On refit cells (new household splits R3 / R4 and reader seeds k2 / k3 of the same 8,001 MODEL_SELECT people), its benefits over the
> twin (+0.0219, se 0.0103), the SBC bracket (+0.0231, se 0.0091) and the SPEND bracket (+0.0189, se 0.0091) stay resolved (D2R
> ROBUSTNESS_CONFIRMED, descriptive). Its benefit over the release-only control DB_ZTILT_R1 roughly halves and is not confirmed
> (+0.0066, se 0.0095). The refit reading never changes D2 and is not a replication. It inherits the winner's curse of choosing
> DB_PTC_R1 on the registered cells: the refit estimates are below the registered ones against all three comparators."

**C6 ("spend-matched" DB_ZTILT_R1).** RESEARCH_DECISION: "its spend-matched release-only control DB_ZTILT_R1: +0.0128, se 0.0081,
which overspends", "A spend-matched ZTILT was also fitted for DB_PTC_R1", "DB_PTC_R1 vs the spend-matched DB_ZTILT_R1",
"its spend-matched release-only control (DB_ZTILT_R1)", "2 for its spend-matched ZTILT"; START_HERE: "its spend-matched release-only
control" and "its spend-matched ZTILT"; ADVISOR_BRIEF: "its spend-matched release-only control DB_ZTILT_R1"; HANDOFF: "Against its
spend-matched release-only control". Why: the control was intended to be spend-matched. By its registered first-reaching rule it
spent 2.08x / 1.73x the release's MODEL_FIT spend and 1.7x its MODEL_SELECT cost, so "spend-matched ... which overspends" contradicts
itself. Use:
> "its release-only control DB_ZTILT_R1 (the registered ZTILT tilt along DB_PTC_R1's split law, intended to match its spend; the
> registered scale grid made it spend more: 0.0169 against 0.0099 nats of MODEL_SELECT total NLL)"

**C7 (how the ZTILT_R1 reading is stated).** RESEARCH_DECISION: "Under the card's declared reading, the gain is not separated from what
the twin's release itself reveals."; START_HERE: "That control uses only what the twin's release reveals."; ADVISOR_BRIEF and HANDOFF,
the corresponding sentences. Why: "what the twin's release itself reveals" can be misread as a statement about the twin's leakage. The
reading must be the declared one, placed beside the refit confirmation (C5), and must carry the overshoot without being discounted by
it. Use:
> "By the reading declared in card DB_ZTILT_R1 before its fit, DB_PTC_R1 is NOT_RESOLVED_BELOW_ZTILT: its pair benefit over
> DB_ZTILT_R1, a tilt whose amplitude is estimated from the twin's released vectors alone, is +0.0128 (se_agg 0.0081; lower 2-SE bound
> -0.0035; version means +0.0149 / +0.0108) on the registered cells and +0.0066 (se 0.0095) on the refit cells. So its gain is not
> separated from release-only information (PRC's failure repeated, as declared). The control overspent, which makes the comparison a
> bracket from above; that is disclosed and does not change the declared reading. On each single view the control matches DB_PTC_R1
> (income AUC 0.5329 vs 0.5333, occupation 0.6177 vs 0.6166); the difference is on the linked pair (0.6395 vs 0.6267)."

**C8 (direction error).** RESEARCH_DECISION: "The 2-SE interval misses zero by 0.0013." Wrong direction: the 2-SE lower bound is
-0.0013, so the interval includes zero. Use:
> "Its 2-SE lower bound is -0.0013: the interval includes zero, so the comparison is not resolved."

**C9 ("a finer SBC control grid").** ADVISOR_BRIEF: "A finer SBC control grid for DB_PTC_R1. Its SBC bracket is nearly spend-matched;
the comparison misses by 0.0013."; HANDOFF: "a finer SBC control grid for DB_PTC_R1". Why: the SBC bracket is already within 0.0008
nats of the release's cost (-0.0008, se 0.0020), so a finer grid cannot change the comparison. The design limit that a finer grid
would address is ZTILT_R1's overshoot. Any new reading on these people would be a fifth adaptive look. Use:
> "A confirmation population, with a registered reading of DB_PTC_R1 against its SBC bracket and against a release-only ZTILT point
> that is spend-matched along its split law (a finer ZTILT scale grid; the registered grid jumped from 0.90x / 0.75x to 2.08x / 1.73x
> of the release's MODEL_FIT spend). A finer SBC grid would not help: the SBC bracket is already within 0.0008 nats of the release."

**C10 ("is the twin's").** RESEARCH_DECISION: "ZTILT's own pair AUC is the twin's (DB_ZTILT vs TWIN -0.0006 +- 0.0079)." Why: "not
separated" is never "equal" (RULES.md 3). Use:
> "DB_ZTILT's pair AUC is not distinguishable from the twin's (point -0.0006, se 0.0079)."

**C11 (recipient gate).** RESEARCH_DECISION: "The recipient gate holds in every reading: against the twin, neither recipient's own
output is more recoverable than the twin's (per-recipient labels `RECIPIENT_AGG_RESOLVED` or `_EDGE`, gate passed)." Why: the gate is
an allowance of 0.010 AUC. RESOLVED means within the allowance at 2 SE, and EDGE means within it at the point only. Neither label says
"not more recoverable". Use:
> "The recipient gate passed in every reading against the twin. No recipient's own output is more than 0.010 AUC more recoverable
> than under the twin, in both versions (RECIPIENT_AGG_RESOLVED: within this allowance at 2 SE; _EDGE: within it at the point). At the
> point every candidate's single-view AUCs are below the twin's."

**C12 (DA_COMP "essential").** RESEARCH_DECISION table: "Fixed SEX weights are essential: propensity-weighted standardization leaks
more than the twin"; ADVISOR_BRIEF: "fixed SEX weights are essential (propensity weights leak more than the twin);". Why: DA_COMP against
the twin is -0.0129 (0.0070), not resolved (upper bound +0.0010). The ablation is descriptive and is read on a release that did not
meet D2. "Essential" is a necessity claim. Use:
> "With the person's own propensity as the SEX weight (DA_COMP), the release is resolved more recoverable than DA_STD (-0.0304,
> se 0.0096). At the point only, it is also more recoverable than the twin (-0.0129, se 0.0070, not resolved). Descriptively,
> DA_STD's point benefit depends on the fixed weights."

**C13 (ablation shares, D5 wording).** RESEARCH_DECISION: "The occupation component carries about 60% of the effect", "carries about
40%", "The income component (alpha_I = 0.35) carries about 60% of DB_PTC_R1's effect, the occupation component about 40%", "Both
recipients contribute, but no comparison here isolates a joint (linked-pair) design ..."; ADVISOR_BRIEF: "the occupation recipient
carries about 60% of DA_STD's effect and about 40% of DB_PTC's and DB_PTC_R1's." and "DB's revision gained only by strengthening the
occupation recipient: +0.0075 over DB_PTC, point only." Why: no ablation difference is resolved, and pair-AUC differences are not
additive over recipients, so these "shares" are ratios of unresolved point estimates. Use:
> "At the point (no ablation difference is resolved, and pair AUC differences are not additive over recipients), the occupation-only
> ablations keep about 60% of DA_STD's, 40% of DB_PTC's and 40% of DB_PTC_R1's registered point benefit over the twin."

and

> "DB's revision differs from DB_PTC only in its occupation strength (its income output is bitwise DB_PTC's). Its pair difference
> from DB_PTC is +0.0075 (se 0.0120), not resolved. At the point both recipients' outputs contribute; no contribution is resolved,
> and no comparison isolates a joint design."

**C14 ("post hoc").** RESEARCH_DECISION: "**Selection was post hoc.** The release was chosen as DB's strongest on the registered cells
after an adaptive revision, so the winner's curse applies."; HANDOFF: "The release was chosen post hoc as DB's strongest"; ADVISOR_BRIEF:
"the release was picked as strongest after an adaptive revision". Why: in this campaign "post hoc" marks a repair made after an
outcome. The choice used the registered rule, and deviation 8's mapping cannot change it (section 1.4). The point to make is
selection on the same cells. Use:
> "**Selected on the same cells.** DB_PTC_R1 was chosen by the registered rule (RULES.md 8) as DB's strongest feasible nonzero
> release on the registered cells, from a pool that included the adaptive revision. Deviation 8 maps its strength pair to
> alpha_I + alpha_O, which can only break exact ties, and there was none. The winner's curse applies: its refit estimates are below
> its registered ones against the twin (0.0219 vs 0.0239), the SBC bracket (0.0231 vs 0.0265) and the SPEND bracket (0.0189 vs
> 0.0245)."

**C15 (gains against the twin).** ADVISOR_BRIEF: "Both lowered linked-pair recovery against the task-only twin by 0.016-0.024 AUC.";
HANDOFF: "The 0.016-0.024 gains are against the twin"; RESEARCH_DECISION: "The gains against the twin (0.016-0.024) are a separate,
smaller kind of result". Why: the range mixes unresolved registered estimates with the resolved adaptive one, and it omits DA_IPW.
Use:
> "At the point, every release had lower linked-pair recovery than the twin. Registered: DA +0.0175 (se 0.0090) and DB +0.0164
> (se 0.0107), neither resolved. Adaptive revisions: DB +0.0239 (se 0.0101), resolved; DA +0.0072 (se 0.0087)."

In the D4 sentences, write "+0.007 to +0.024 against the twin (resolved only for the adaptive DB_PTC_R1)".

**C16 ("came close").** ADVISOR_BRIEF: "Neither came close to the strongest references or to the 0.020 criterion." Why: "close" is not
a rule outcome, and against H_LJ and B2TWIN DB_PTC_R1's differences are unresolved. Use:
> "Neither is competitive with the strongest references (D3 NOT_SUPERIOR in every reading) or meets the 0.020 material criterion (D4
> NOT_MET; the largest point values, DB_PTC_R1's, are +0.0110 (se 0.0135) over NEURAL_RAW and -0.0058 (se 0.0151) over H_LJ)."

**C17 (STD "no longer varies with SEX").** ADVISOR_BRIEF: "SEX-aware task models are averaged over both SEX values with fixed weights, so
the release no longer varies with SEX at fixed inputs." Why: no release here, the twin included, takes SEX at runtime, so this
property is not specific to STD. The sentence implies a removal property that STD does not have. Use:
> "SEX-aware task models are averaged over both SEX values with fixed weights, so the person's SEX propensity no longer enters through
> the mixture weights. The release is still a function of the six fields, which predict SEX. On MODEL_FIT its income log-probabilities
> kept about two thirds of the twin's linear propensity dependence (R^2 0.066 vs 0.100; `da/R1_TRIGGER_B2.json`)."

**C18 (challenger finding).** ADVISOR_BRIEF: "The challenger found the two designs not equivalent, so neither designer was
redirected." Why: the finding also places both designs in one idea family (C1). Use:
> "The challenger found the two designs to be different operations within one idea family (both subtract the propensity-proportional
> part of the posterior), so neither designer was redirected and a joint negative closes one idea, not two."

**C19 (PROTOCOL_DEVIATIONS.md, deviation 12).** Add item (e):
> "(e) DB_R1's D1 record lists DB_PTC_R1 under `ignored_zero_strength_entries`, with `fitted_strengths` empty. Its strength pair
> [0.35, 0.65] is not read as a nonzero number by the frozen helpers, the cause of deviation 8. The list is used only when CAND fails
> G1, and DB_PTC_R1 passes G1, so no label changes. STRONGEST.json's 'strength 1.0' for DB_PTC_R1 is alpha_I + alpha_O, not
> alpha = 1."

**C20 (adaptive labels in the decision table).** RESEARCH_DECISION decision table: the DB_R1 and DA_R1 rows show unprefixed labels
(`TASK_FEASIBLE`, `PARTIAL_TWIN+SPEND_FRONTIER_ONLY`, ...). RULES.md 7 prefixes every D label and the outcome class of a
non-registered reading. Write the prefixed labels in those rows, at least in the D2 and outcome-class cells (for example
`ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY`, `ADAPTIVE_R1__TASK_ONLY_FEASIBLE_NO_PROTECTION`). Keep the registered rows first.

**C21 (campaign-level statement).** RESEARCH_DECISION "Decision": "D2 was met in 0 of 2 registered readings and 0 of 2 non-registered
(adaptive) readings: 4 D2 looks in all (`claims/CAMPAIGN_SUMMARY.json`)."; likewise ADVISOR_BRIEF question 2, START_HERE and HANDOFF.
RULES.md 7 requires every campaign-level sentence to carry the per-look false-pass statement. Append:
> "one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws and about 8% when
> one version or split dominates; with 4 D2 looks the chance that at least one passes by chance is at most 4 times the per-look rate."

**C22 (D2R section).** RESEARCH_DECISION: "is refitted against new household splits R3 / R4 and new reader seeds k2 / k3" should say
that the readers are refitted, on new household splits and reader seeds of the same MODEL_SELECT people. In the D2R table add a row
or a note: "DB_PTC_R1 vs DB_ZTILT_R1 (release-only control, beside D2, outside the decisive set): +0.0066 (0.0095) on the refit cells,
ROBUSTNESS_NOT_CONFIRMED". Mark the DB_PTC row "extra, descriptive (deviation 12 (d))". Add the narrow margins: on the refit cells the
lower 2-SE bounds are +0.0013 against the twin and +0.0007 against the SPEND bracket.

**C23 (data-rich readers).** RESEARCH_DECISION: "They agree in sign with the grid for every decisive comparison of DB_PTC_R1: ... vs the
registered ZTILT +0.017 to +0.025. Every version mean is positive." Add, in the same bullet: these are point values with no
uncertainty. No data-rich readers were trained for DB_ZTILT_R1, the release-only control at which the grid comparison is unresolved.
On the income view the data-rich LOG_LR finds DB_PTC_R1's (= DB_PTC's) output more recoverable than DB_ZTILT's (-0.0040 / -0.0042).
DA_IPW's data-rich benefit over DA_SPD_HI is negative in version 1 (HGB15 -0.0053), which the amendment's rule would have labelled
READER_BUDGET_DEPENDENT had its D2 passed. Agreement in sign is not evidence of information removal (section 4.1).

**C24 (scope of the verifier's reader-dependence check; D3 paragraph).** RESEARCH_DECISION: "The verifier's own LR and HGB readers change
the sign against NEURAL_RAW and H_LJ, so those two differences are reader-dependent. D3 fails under every reader used." Placed after
DB_PTC_R1's table, it reads as covering DB_PTC_R1. The check covered DA_STD and DB_PTC only (first report, V3.8); the addendum ran no
independent readers. Write "For the registered releases DA_STD and DB_PTC, the verifier's own LR and HGB readers ...; no
independent-reader check was run for DB_PTC_R1." After "use less of the utility budget", add "This is not a utility-matched
comparison and changes no label." In the Decision bullets, "It stays 0.005-0.027 AUC above four strong references." is a point
statement: add "resolved only against RETIRED_B3CTL (-0.0267, se 0.0111)".

**C25 (favourable list).** RESEARCH_DECISION: "DA_STD's consistent +0.017-0.018 against the twin on registered and refit cells." Use:
> "DA_STD's point benefit over the twin is similar on registered and refit cells (+0.0175, se 0.0090; +0.0176, se 0.0110) and
> resolved on neither (D2R not confirmed against the twin)."

Also add "(lower 2-SE bound +0.0001)" to "Its domination of the SEX-blind spending frontier". Add "(different split law; the same-law
control DB_ZTILT_R1 is not separated)" to "Its resolved benefit over the registered ZTILT, which was not spend-matched".

**C26 (adverse list; additions).** Add: DB_PTC_R1's benefit over DB_ZTILT_R1 falls to +0.0066 (se 0.0095) on the refit cells. Two of
its three refit confirmations clear the 2-SE threshold by about 0.001. The benefit over the twin in block S0R1 is about +0.034 with
seeds k0 / k1 and +0.003 with k2 / k3. The registered predictions for ZTILT_R1 failed in the direction adverse to DB_PTC_R1: ZTILT_R1's
pair AUC was predicted within +-0.008 of the twin's (p 0.7) and observed 0.0111 lower; DB_PTC_R1 was predicted resolved above it
(p 0.55) and was not.

**C27 (DA R1 description).** RESEARCH_DECISION "Revisions": the DA R1 bullet must say that it followed the registered DA privacy
outcome (DA_REG's D2), not only the utility read (section 1.6). In the adverse list, "DA's revision did worse than the registered
STD" is a point statement: add "(-0.0103, se 0.0107, not resolved)".

**C28 (HANDOFF "What must not be said").** Replace "Not 'removes information beyond the release' for either DB release. ..." with
the two precise statements of section 3.0: not "needs (or uses) propensity information beyond the release", and not "fully explained
by release-only information". Add "Not 'information removal' for any release (section 4.1)". Add the items of section 3.0 that are
not yet listed: separation from the SBC control, D2 on the refit cells, replication, stability across reader seeds, and DB_PTC_R1 as
DB's result.

**C29 (utility wording; lower priority).** "Every release kept the task models useful" (RESEARCH_DECISION, START_HERE), "Both
independently designed families kept the predictions useful." (ADVISOR_BRIEF) and "Both kept the predictions useful" (HANDOFF): say "every saved configuration met all 28
original utility conditions at the point (G1)", with "point feasibility is not confidence preservation". Add that DB_PTC_R1, its
ablation and DB_ZTILT_R1 are not reserve-qualified.

## 4. What the evidence does and does not support

### 4.1 Information removal versus reader budget

- **No release here can carry an information-removal claim.** None is a garbling (post-processing) of the twin's release, so the data
  processing inequality gives no ordering against the twin. All are deterministic functions of the six fields, and any reduction is
  relative to the readers that were tried. PROTOCOL_AMENDMENT_1 (findings 3, 18) made a positive data-rich benefit a necessary
  condition for even discussing removal after a D2 pass. It never made it sufficient, and no D2 passed.
- **DB (PTC, PTC_R1).** The release writes the runtime propensity into the vector and is injective on its support (challenger finding
  3, disclosed in the card before any outcome). A reader able to invert the tilt recovers at least what the twin carries. The measured
  reduction is first-moment cancellation against the declared readers, and it persists for three data-rich recipes trained on 20,000
  MODEL_FIT releases (pair benefits over the twin +0.025 to +0.040, over the brackets +0.013 to +0.040, all version means positive). So
  for those recipes the reduction is not an artefact of the DDR1 readers' smaller FIT samples (half of the MODEL_SELECT households). It is not separated from release-only
  information (DB_ZTILT for DB_PTC: +0.0170 (0.0121); DB_ZTILT_R1 for DB_PTC_R1: +0.0128 (0.0081), refit +0.0066 (0.0095)), so even
  the weaker reading "the gain needs propensity information beyond the release" is not established. **Supported:** the declared and
  data-rich readers recover SEX less well from this re-encoding. **Not supported:** removal, or a guarantee against other readers.
- **DA (STD, IPW-STD).** The fixed-weight mixture removes the propensity from the mixture weights only. On MODEL_FIT the lambda = 1
  release kept about 2/3 (income) and 2/5 (occupation) of the twin's linear propensity dependence (`da/R1_TRIGGER_B2.json`). The
  data-rich readers keep DA_STD's point benefits positive in both versions, but against the SPEND bracket HGB15 sees only
  +0.0018 / +0.0050. The revision aimed at the residual dependence (IPW-STD) did worse at the point (-0.0103 (0.0107) against DA_STD).
  Its data-rich benefit over DA_SPD_HI is negative in version 1. DA's reduction is not separated from SEX-blind task spending on the
  registered cells and is not robust to refitting against the twin or the SPEND bracket. **Supported:** a point reduction for the
  declared readers. **Not supported:** removal or robustness.
- None of these data-rich numbers has an uncertainty estimate; they are descriptive.

### 4.2 The shared idea family

STD and PTC, and both revisions, subtract the propensity-proportional part of the released posterior: the challenger's finding 1,
adopted by PROTOCOL_AMENDMENT_1. The four D2 looks are four looks at one idea. The registered releases are indistinguishable from
each other at the point (DA_STD vs DB_PTC +0.0011 (0.0123)), as one idea family would predict. The joint negative closes one idea,
not two. It says nothing about release families outside it, for example finite or coarsened occupation outputs like those of the
strongest references, which are a different family.

### 4.3 Joint design (D5) and novelty (D6)

D5 is `NOT_ASSESSED_MISSING_CONTROLS` in every reading: no separately designed recipients and no sequential-order controls were built.
The occupation-only ablations are component ablations, not joint-design controls, and their contrasts are unresolved. Nothing here
supports, or refutes, a joint-design (pair super-additivity) advantage. D6 is `NOT_CLAIMED` in every reading: no separate prior-art
review was run, and both designers named close published work (standardization and adjustment formulas; AttriGuard, LEACE,
multiaccuracy-style proxies, exponential-tilt adjustment; inverse-propensity reweighing). Novelty is never inferred from performance.

### 4.4 Component, candidate or method

No release may be called a method, a candidate or a component. Under RULES.md 6, COMPONENT needs D2 met and CANDIDATE needs D2 and D3;
no reading met D2, and every reading is TASK_ONLY_FEASIBLE_NO_PROTECTION. The protocol's own term "component ablation" names a part of
an operation (its income or occupation side). It must never become "X is a component". The permitted description of DB_PTC_R1 is "DB's
strongest feasible nonzero release (ADAPTIVE_R1; selected on the registered cells; descriptive)".

### 4.5 Registered expectations against outcomes (descriptive)

The protocol's most likely outcome was a 0.005-0.015 effect against the twin, not separated from a SEX-blind control at matched spend.
It was approximately realized: the registered effects were +0.0164 to +0.0175, unresolved, and not separated. P(D2 met for at least one
family) was registered at about 0.10; D2 was not met. DB R1's own predictions were largely met (0.35 / 0.65; +0.0075 over DB_PTC). The
ZTILT_R1 predictions failed against DB_PTC_R1 (C26).

## 5. Completion record

- Writing started 2026-10-21T19:55:01Z and was completed at the time in the last line of this section, both read with `date -u`.
- **Correction (disclosed):** the first save of this file, at about 19:57Z, ended with a placeholder completion line reading
  "2026-10-21T20:10:00Z", which was ahead of the clock (the clock read 19:57:38Z right after the save) and was not a `date -u` reading.
  It was replaced within a minute, before this file was handed to anyone. No finding changed with it.
- Rules intact: `RULES.json` e4b21944a3c73d829f6a337be0eb81685a1b04288e401780bb422b047ae64a2c (= every DECISIONS.json's
  rules_sha256); `ddr/rules.py` 6e78dac2..., `ddr/test_rules.py` 7dc3233d..., all three last committed at registration (8f736cbe3)
  and unmodified in the working tree. `python3 -m ddr.test_rules`: 22 of 22 PASS, exit 0.
- No label was read and no real-data computation was run. No repository file other than this statement was written or changed; my
  helper scripts (JSON pretty-printers) lived in the session scratchpad, outside the repository. Nothing was committed.
- CPU: about 6 CPU-seconds in all, by estimate. That covers about 20 short Python processes reading JSON records, each well under
  0.3 CPU-s, plus the invented-data test (0.21 CPU-s reported) and its imports; shell listings and greps are negligible.
- First full draft saved by 2026-10-21T19:57:38Z (`date -u` read after the save).
- Completed 2026-10-21T19:59:56Z (`date -u`, read immediately before this line was written).

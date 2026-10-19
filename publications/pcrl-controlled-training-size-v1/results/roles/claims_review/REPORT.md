<!-- Attribution: written by the engineering / claims reviewer (an agent assignment of this campaign, wave 2) and returned as text in
its final hand-back (its platform rules forbid subagents from writing report .md files). Saved verbatim by the coordinator at about
01:53Z (2026-10-19, date -u); not coordinator analysis. The coordinator's disposition is adjudication A5 in
roles/coordinator/ADJUDICATIONS.md. -->

## Claims review, wave 2

The decision documents report almost every number correctly; checked against the records, I found four factual errors. Each is MUST-FIX:
- M1: the control's DiD is called "not resolved for any metric", but women's Brier is resolved on v1.
- M2: "all original limits" ignores the recipient gate, which fails against every reference.
- M3: the A-SC1 sentence says no stability label had been computed, but the verifier had computed its labels by then.
- M4: the ADVISOR_BRIEF and START_HERE omit the required A-SC1 disclosure.

The other six MUST-FIX items concern wording that is causal or too strong (M5 and M6), or wording that blurs the original limits with the reserve heuristic (M7). The two closure labels hold on every reading of the records: CORRECTION_RECIPE_CLOSED and C1 MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL.

**Header.** Written 01:43:16Z–01:52Z (`date -u`). I read no label, ran no fit and edited no document.
- **Reviewed:** RESEARCH_DECISION.md, ADVISOR_BRIEF.md, START_HERE.md as of b1da02c45. HEAD later moved to 9150ca4dd, which only adds figure references; I spot-checked figures/SIZE_CURVE.csv against SIZE_TABLE.csv.
- **Checked against:** the size/ CSVs and SIZE_RESULTS.json, IDENTITY_CHECKS.json, trigger/TRIGGER.json, SELECTION_V0/V1, COND_UTILITY.json, CLAIMS.json and its three CSVs, READER_SEED_CHECK.json, WIRE_CARDINALITY.json, RULES.md (sections 0, 3, 7, 9–12), ADJUDICATION_SEEDCHECK.md, PROTOCOL_DEVIATIONS.md, ADJUDICATIONS.md, the verifier's CHECKS.json and DEPLOY_W100.json, deploy/DEPLOYMENT_CHECKS.json, and the earlier campaign's RESEARCH_DECISION.md line 118.

**Summary.** 10 MUST-FIX (one of them in two parts, M7a and M7b), 15 SHOULD-FIX, 5 NOTE. All eight questions are answered plainly. C7 NOT_CLAIMABLE, C8 NOT_CLAIMED and the "not claimable" list are stated.

## MUST-FIX

**M1. RESEARCH_DECISION §3, third bullet.**
- Current: "the difference between the two gains (per-person double difference) is not resolved for any metric."
- Record: CONTROL_COMPARISON.csv, TR difference-in-gains, 50->100, women's Brier: v0 +1.21 ± 1.19 UNRESOLVED; v1 +2.44 ± 1.14 RESOLVED_POSITIVE; across versions VERSION_DEPENDENT. SIZE_RESULTS gives women's Brier the label RECIPE_GAIN_CONTROL_UNRESOLVED.
- Replace: "the difference between the two gains (per-person double difference) is not resolved for NLL, Brier, men's Brier or accuracy; for women's Brier it is version-dependent (+2.44 ± 1.14 x 1e-3 in favour of A1_PO_16 on v1, unresolved on v0; label RECIPE_GAIN_CONTROL_UNRESOLVED)."

**M2. ADVISOR_BRIEF, finding 1, third bullet.**
- Current: "The ordinary control shows the same kind of size gain; the difference between the two gains is not resolved."
- Replace: "The ordinary TASK_REFIT control shows a resolved size gain of its own on NLL, Brier and men's Brier. The difference between the two gains is not resolved on those metrics; on women's Brier it is version-dependent (in favour of A1_PO_16 on v1 only)."

**M3. RESEARCH_DECISION, Decision in brief, third bullet** (original limits blurred with the reserve heuristic).
- Current: "At s = 0.3 both arms keep utility on MODEL_SELECT (all original limits and the occupation reserve, both versions)"
- Why it is wrong: RULES §0 lists recipient non-worsening (≤ +0.010 in every cell) as an original limit. The occupation-view recipient gate fails for LOC_SEL and PAIR_SEL against all six references (CLAIMS.json; ORIGINAL_BENCHMARK.csv, occupation_gate RECIPIENT_POINT_FAILED).
- Replace: "At s = 0.3 both arms keep utility on MODEL_SELECT in both versions: all 12 clauses and both within-SEX guards per version at the original point limits, and, reported separately, the occupation-side two-SE reserve (a development heuristic). Both fail the original per-cell recipient gate in the occupation view against every strong reference (section 6)"

**M4. ADVISOR_BRIEF, finding 3, first bullet** (same error).
- Current: "At that strength they pass every original limit and keep the occupation reserve."
- Replace: "At that strength they pass every original point limit (24 clauses + 4 guards) and, separately, keep the occupation-side reserve (a development heuristic). They fail the original per-cell recipient gate in the occupation view against every strong reference."

**M5. The A-SC1 timing sentence contradicts PROTOCOL_DEVIATIONS 11–13** (RESEARCH_DECISION §5, and READER_SEED_CHECK.json's disclosure).
- Current: "Amendment A-SC1 (statistical reviewer), made after this check's outcome and before any reader-stability label was computed or read, limited grid invalidation to unexplained equalities."
- Why it is wrong:
  - The verifier's unit VER_READERS_V1 computed stability, target and recipient labels at 01:32:59–01:33:09Z.
  - The ruling came at 01:36:00Z.
  - The document's own Resources section says the verifier's labels were "computed before the coordinator had examined the seed check (items 11-12)".
  - The adjudication's own top disclosure says "before any reader-stability label was computed by the coordinator".
- Replace: keep the required sentence verbatim, then add: "Qualification (PROTOCOL_DEVIATIONS items 11-13): 'before any reader-stability label was computed or read' holds for the coordinator and the statistical reviewer. The independent verifier had already computed its own reader labels (unit VER_READERS_V1, 01:32:59-01:33:09Z, after the seed-check record and before the ruling at 01:36:00Z) and stored them unread until the claims were computed."
- The statistical reviewer should also amend the section 5 sentence in ADJUDICATION_SEEDCHECK.md, or accept this qualification appended to it.

**M6. ADVISOR_BRIEF, Process notes** (same error).
- Current: "The reader-seed check had to be amended after its outcome, before any stability label: two references' coarse income wires make ExtraTrees seed-invariant."
- Replace: "The reader-seed check had to be amended after its outcome, before the coordinator or the statistical reviewer had computed or read any stability label (the independent verifier had computed its own labels minutes earlier and stored them unread; deviations 11-12): two references' coarse income wires make ExtraTrees seed-invariant."

**M7. The required A-SC1 disclosure is missing where reader labels are reported.** ADJUDICATION_SEEDCHECK.md §5 says "required wherever reader labels are reported".
- **M7a. ADVISOR_BRIEF.** Finding 3 reports reader results ("the registered screen says not stable", "changes sign", the references' recovery, the 0.020 target). Insert after finding 3: the §5 sentence verbatim, plus the M5 qualification.
- **M7b. START_HERE, "Outcome in one line".** It reports "it does not lower the readers' recovery stably". Append: "Reader labels were computed under amendment A-SC1: [the §5 sentence verbatim, plus the M5 qualification]."

**M8. RESEARCH_DECISION, Decision in brief, first bullet** (headline over-claim).
- Current: "The equally exposed task-only control (T_TREE then TASK_REFIT) shows the same kind of size gain, so the gain is a training-size property of this pool and these learners, not a recipe-specific one."
- Why it is wrong:
  - RECIPE_SPECIFIC_SIZE_GAIN needs the DiD to be POSITIVE_BOTH_VERSIONS; failing it means "not shown recipe-specific", not "shown not recipe-specific".
  - The women's Brier DiD is resolved on v1.
  - The control's gains are smaller: NLL 7.8 / 8.1 against 11.5 / 11.4 x 1e-3.
- Replace: "The equally exposed task-only control (T_TREE then TASK_REFIT) shows a resolved size gain of its own on NLL, Brier and men's Brier, and the difference between the two gains is not resolved on those metrics (registered reading IR3, CONTROL_EXPLAINS_GAIN). The size gain is therefore not shown to be specific to the recipe."

**M9. RESEARCH_DECISION §1, short answer** (same over-claim).
- Current: "But the same size gain appears in the ordinary equally exposed control, so "training size" explains it, not the recipe."
- Replace: "But the ordinary equally exposed control shows a resolved size gain of its own (NLL 7.8 / 8.1 against the recipe's 11.5 / 11.4 x 1e-3; men's Brier 2.9 / 4.1 against 4.3 / 4.2), and the difference between the gains is not resolved on those metrics. So the gain is not shown to be specific to the recipe (IR3)."

## SHOULD-FIX

**S1. Causal "explains" / "effect" wording.** RULES 3.6 says a supported size explanation "means only: on this pool, under this recipe, the larger nested training sets had resolved lower loss on MODEL_SELECT and the margin pattern matched."
- (a) START_HERE.
  - Current: "On this pool the lost margin is a training-size effect, though one the ordinary control shares."
  - Replace: "On this pool the registered size explanation is supported (SUPPORTED_ON_THIS_POOL: larger nested training sets had resolved lower loss on MODEL_SELECT and the margin pattern matched; not a population learning curve or causal proof), and the ordinary control shows a resolved size gain of its own."
- (b) RESEARCH_DECISION brief.
  - Current: "the lost margin is explained by training size on this pool, but not specifically by the recipe."
  - Replace: "the registered size explanation is supported on this pool, but the size gain is not shown to be specific to the recipe."
- (c) ADVISOR_BRIEF finding 1.
  - Current: "Training size explains the lost margin on this pool, but not specifically through the recipe."
  - Replace: "On this pool the registered size explanation is supported, but the gain is not shown to be specific to the recipe."
- (d) RESEARCH_DECISION §8.
  - Current: "The ordinary equally exposed control shows the same size gain (IR3)."
  - Replace: "The ordinary equally exposed control shows a resolved size gain of its own on NLL, Brier and men's Brier, and the difference is not resolved there (IR3)."

**S2. RESEARCH_DECISION §8: give C1's registered scope verbatim.**
- Current: "Claim C1: MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL, with that scope."
- Replace: "Claim C1: MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL. Registered scope: within the fixed recipe, this MODEL_FIT pool, adaptively used MODEL_SELECT; not a population learning curve; not causal for every historical difference."

**S3. RESEARCH_DECISION §2: the reserve and trigger scope sentences are missing** (PROTOCOL 6–7; RULES 12). Add at the end of §2: "The two-SE reserve is a development heuristic, not a new scientific cap or population guarantee; it is reported separately from the original limits. The trigger is an occupation-component trigger and does not declare that the H3-anchored service qualifies."

**S4. RESEARCH_DECISION §2, short answer.** RULES 3.5(d) says usefulness is answered only by the correction stage.
- Current: "Yes, exactly."
- Replace: "Yes for reproduction and the registered margin: the refits are bitwise (REPRODUCED_BITWISE, MARGIN_REPRODUCED) and the trigger fired (MARGIN_AVAILABLE). Whether the slack is usable for protection is answered in section 6."

**S5. RESEARCH_DECISION §1, encoding bullet.** This is an unresolved, one-version descriptive result; RULES 6 says "never a decomposition claim".
- Current: "Rare-category pooling does not explain the size effect."
- Replace: "The encoding effect is unresolved; its point estimate is about 5-6% of the 50 -> 100 change (version 0 only; descriptive, not a decomposition)."

**S6. RESEARCH_DECISION §6, strong references.**
- Current: "every candidate reads worse than each of the six references (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ: pair benefits -2.8 to -45.2 x 1e-3, all NOT_POSITIVE)"
- Why: against TWJ, LOC's cells span -6.2 to +5.3 x 1e-3, and W's and CONT's cells also take both signs. The references' original point status is not stated.
- Replace: "against each of the six references (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ) every candidate's 8-cell mean pair benefit is negative (-2.8 to -45.2 x 1e-3, all NOT_POSITIVE; against TWJ the cells take both signs for LOC, W and CONT). Every reference passes the original point limits (24 clauses + 4 guards) but fails the reserve heuristic."

**S7. ADVISOR_BRIEF, finding 3.**
- Current: "Every strong reference has much lower recovery but fails the reserve."
- Replace: "Every strong reference has lower mean recovery (by 2.8e-3 for TWJ against LOC, up to 45e-3), passes the original point limits, and fails the two-SE reserve heuristic."

**S8. Transfer warning: these are point estimates, on different people.** Realized costs are on MODEL_SELECT, predicted costs on out-of-fold MODEL_FIT. Realized SE is 0.33–0.40e-3 and predicted SE 0.19–0.22e-3. Only the LOC v0 gap (1.21e-3) exceeds two combined SEs; the other three gaps are 0.42–0.70e-3, about 1.0–1.6 SE.
- (a) ADVISOR_BRIEF finding 4.
  - Current: "The nested 90% cost estimate under-predicted the deployed model's men's cost about two- to threefold (realized 0.9-1.9e-3 against 0.5-0.8e-3 predicted)."
  - Replace: "In point estimates the nested 90% cost estimate under-predicted the deployed model's men's cost 1.9- to 2.8-fold (realized 0.9-1.9e-3, SE 0.3-0.4e-3, on MODEL_SELECT, against 0.5-0.8e-3 predicted); only LOC v0's gap exceeds two SEs."
- (b) RESEARCH_DECISION §4 bold.
  - Current: "**The nested estimate under-predicted the deployed model's costs.**"
  - Replace: "**In point estimates the realized costs exceeded the nested predictions.**"

**S9. Reader-seed replication per comparison** (RULES 9: when a comparison's qualifier is not ALL_PAIRS, say on how many of the 4 version-repetition pairs the reader seed changed the result).
- Add under the §6 table: "The reader seed changed the result in 3 of the 4 (version, repetition) pairs for every comparison in the table except CONT vs W (2 of 4). Identical cells are counted once (n_eff 7 and 6)."
- ADVISOR_BRIEF, current: "All 8 cells are positive". Replace: "All 8 cells are positive (7 distinct; the reader seed changed the result in 3 of 4 version-repetition pairs)".

**S10. RESEARCH_DECISION §4: the rule description is inexact.**
- Current: "Rule: the largest strength whose every smaller strength keeps estimate + 2 SE within one H_LJ-floor SE on men's and women's Brier, Brier, NLL and accuracy."
- Replace: "Rule: the largest registered strength s such that s and every smaller nonzero strength keep estimate + 2 SE at or below one H_LJ-floor SE (the smallest among the conditions mapped to each statistic) for men's Brier, women's Brier, Brier, NLL and accuracy loss; identity if 0.01 fails."

**S11. Five benchmark units fitted on invented labels before the registration (deviation 7).**
- (a) ADVISOR_BRIEF.
  - Current: "Everything was registered and published on GitHub before any fit."
  - Replace: "Everything was registered and published on GitHub before any real-label fit (five label-free benchmark units on invented labels ran earlier on uncommitted code; deviation 7)."
- (b) START_HERE.
  - Current: "before any fit or label read"
  - Replace: "before any real-label fit or label read".

**S12. ADVISOR_BRIEF overstates the verifier's scope.**
- Current: "An independent verifier replayed everything with its own code."
- Replace: "An independent verifier replayed the registered checks with its own code: all 22 A1_PO_16 fits refitted bitwise, the trigger (within 1.7e-16), the S100 control identity and one T_TREE refit, one nested outer fold end to end, both selections, the utility tables and 39 reader-label comparisons."

**S13. RESEARCH_DECISION has no verifier summary** (PROTOCOL 11: report numerical reproduction and procedural compliance separately; a corrected check keeps its original result). Add to "Resources, failures and unrun stages": "Verifier: every numerical replay passes. Three passes followed adjudicated verifier implementation bugs, with the original results kept: A1 (V0.PINS.PUBLICATION_TREE, FAIL then PASS, CRLF handling) and A4 (V5.CORRECTION.TARGETS_INVARIANCE, ATTENTION then PASS; V6.TABLES, FAIL then PASS). Procedural compliance: V6.PROCESS.CODE_PROVENANCE FAIL (deviation 7); ATTENTION on the two null-unit selftest reads (deviation 4) and on the literal seed check (A-SC1)."

**S14. RESEARCH_DECISION §7 has an exception.** T_TREE (TT100) binds lower, at 2.60 on its own occupation side.
- Current: "but binds the whole-release reserve of every H3-anchored release at 2.86 (v1 income accuracy)"
- Replace: "and caps the whole-release reserve of every H3-anchored release at 2.86 (v1 income accuracy), the binding condition for every candidate, the witness and TASK_REFIT (T_TREE binds lower on its own occupation side, 2.60)"

**S15. Precision fixes.**
- (a) RESEARCH_DECISION brief.
  - Current: "holds in all 8 cells at 70% and at 100%"
  - Replace: "holds in all 8 cells at 70% and in both versions at 100%"
  - ADVISOR_BRIEF has the same phrase "all 8 cells at 70%, at 100%,"; replace with "all 8 cells at 70% and in both versions at 100%,".
- (b) ADVISOR_BRIEF.
  - Current: "falls by 4.2-4.3e-3 on both versions (every ordering)"
  - Replace: "falls by 4.3 / 4.2e-3 on average (v0 / v1; every ordering positive, 3.2-5.3e-3)"
- (c) RESEARCH_DECISION §3, short answer.
  - Current: "At 100% A1_PO_16 is resolved better than T_TREE"
  - Replace: "At 100% A1_PO_16 is resolved better than T_TREE on NLL, Brier and men's Brier (not on accuracy or women's Brier)"
  - ADVISOR_BRIEF, current "(it is better than T_TREE)"; replace with "(it is resolved better than T_TREE on NLL, Brier and men's Brier)".
- (d) RESEARCH_DECISION §6.
  - Current: "(whole-release minimum ratios 0.32-1.95)"
  - Replace: "(whole-release minimum ratio over both versions 0.32-1.56, EQUAL_RESERVE.csv; per version 0.32-1.95)"
- (e) Pointers.
  - RESEARCH_DECISION Decision in brief and §6: add "(reader labels under amendment A-SC1; disclosure in section 5)".
  - START_HERE: the claims review path should read `roles/claims_review/REPORT.md`, not "roles/engineering_review/ … plus the claims review". Also, `roles/verifier/REPORT.md` did not exist at 01:47Z.

## NOTE
- **N1. RESEARCH_DECISION §8.** Current: "reproduces the source package and its margin bitwise". Replace: "reproduces the source package bitwise and with it the occupation-side margin (MARGIN_REPRODUCED)".
- **N2. §1 table.** The v0/v1 gap in the excess rows mostly reflects each version's own reference U0_S<v>. On the same rows the fits differ by at most 0.43e-3 men's Brier. A footnote would prevent misreading.
- **N3. §7 H3 provenance.** H3's RELEASE.json also lists stored replication tree parameters, including a JOINT_TREE income correction trained with MODEL_FIT SEX. Add this for completeness.
- **N4.** The earlier campaign's values (+0.65 / +2.15 / +1.11 / +1.89) match its RESEARCH_DECISION line 118. They come from different partitions, so they are descriptive only.
- **N5. ADVISOR_BRIEF recommendation.** "shared by ordinary learners" rests on two controls, both from the same replication pipeline. Consider "shared by the equally exposed T_TREE / TASK_REFIT control".

## Verified correct against the records
- **§1 size results:**
  - table: men's excess, NLL excess, 4 / 8 / 2 cells meeting the reserve, ratio ranges 1.10–2.87, 2.55–3.73, 4.88 / 4.30;
  - all contrasts and their SEs, labels, and "both steps contribute";
  - FULL_ONLY with FULL_ONLY_70_RETAINS, IR1, SUPPORTED_ON_THIS_POOL;
  - subset ranges, version difference ≤ 0.43e-3, encoding +0.64 ± 0.66 and 5–6%.
- **§2:** T* 168 / 154; REPRODUCED_BITWISE; the verifier's 22 refits bitwise on all roles; the trigger's T0–T3; replay within 1.68e-16; 24 / 24 refusals; slack 5.23 / 3.90e-3.
- **§3:** every level difference; reserve counts 7 / 8, 1 / 8 (TASK_REFIT) and 2 / 2, 4 / 8, 1 / 8 (T_TREE); CONTROL_ALSO_MEETS_RESERVE at 100% only; CONTROL_EXPLAINS_GAIN on NLL / Brier / men's Brier; S100 controls identical to the admitted ones.
- **§4:** 0.3 / 0.3 / first failure 0.5; commit at 01:18:34Z; realized and predicted numbers; correction base bitwise equal to the source.
- **§5–§6:** 14 members; 16 equalities; the minimum set clean; the six-row table (means, minima, maxima, labels); the utility ratios; C2 / C2p / C3 / C3w / C4 / C5 / C6; CORRECTION_RECIPE_CLOSED; mean pair AUC 0.644 / 0.649; 11 packages.
- **§7:** income ratios 3.51 / 2.86; the income gate fails only against RETIRED_B3CTL.

## Not checked
- The figure image itself; only its CSV was spot-checked.
- The closeout files (ledger, resources, handoff), which did not exist yet.
- The reader grids, which are private.
- The investigator reports and QUICKSTART.md.

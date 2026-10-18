# Claims review report (claims reviewer agent; saved by the coordinator)

**Attribution.** The independent claims reviewer wrote this text and returned it at the end of its assignment (21:00-21:22Z). The
tool refuses report files from a subagent, so the reviewer could not write this file itself. The coordinator saved its returned text
verbatim at 21:29Z. This is not coordinator analysis. The coordinator's disposition of each required edit is recorded at the end of
the coordinator's `roles/coordinator/ADJUDICATIONS.md` (A7).

---

# Claims review: PCRL correction-transfer campaign (CTS1)

**Attribution.** Independent claims reviewer (subagent), 2026-10-18. Review ran 21:00–21:22Z (clock read at 21:21:31Z). I did not
write `roles/claims_review/REPORT.md` because subagent rules forbid report files; please save this text there verbatim, with
attribution. I edited no file, read no label, fitted nothing and ran no queue job.

**What was reviewed.** Commit 97de83f0f, plus the working-tree RESEARCH_DECISION.md (it now includes the fallback table).

**A5 status used.** I judged the documents against A5's final status:
- Ruling (b) stands (ADJUDICATIONS A5 resolution, commit 6956507a1 at 21:12:07Z).
- Condition 1 holds: verifier P4_QA_CORRECTION_TRIGGER_INPUTS__V2 = PASS.
- Condition 4 is done: C3.QA.R1.VARIANCE_DETAIL__V3.
- The P4b ATTENTION is disclosed in the A5 resolution.

**Method.**
- Every quoted number was recomputed from the label-free records (diagnosis/, repair/, tables/, screen/, deploy/, preflight/).
- Where no public record holds a value, I read private run summaries that contain no person-level data:
  `runs/TRACKF/P70/EVAL/WITNESS_CURVE_EVAL_MS.json` and `runs/TRACKF/R0|R1/EVAL/TRACKR_EVAL_MS.json`.
- Line numbers below refer to the current files.

**Bottom line.**

These match their records exactly and need no edit:
- the §4 table and the §3 rule numbers;
- the question-A labels and the three flips;
- the claim ladder (rung 0) and NO_RECIPE, including its timing;
- the reserve-qualified frontier and benchmark superiority numbers;
- the 0.020 target, reported separately from superiority;
- the Track R recovery, reserve and income-clause values;
- the P70 values, the source replay and the deployment checks;
- the pre-outcome income-view screen values.

No document claims any of the following: confirmation, novelty, impossibility, a joint two-output design, or a changed
NO_QUALIFYING_CONFIGURATION.

The required edits concern:
1. **The nesting claim.** It rests on one partition, one direction and one of two realized panels. It is contradicted on
   MODEL_SELECT and for LOC, and nesting was never run for CONT.
2. **A5 wording.** The scheme-dependent units need A5's final wording.
3. **Causal training-size language.** The registered bet (TRAINING_SIZE_BET) forbids it.
4. **Numbers that differ from their records.** There are eight.
5. **An incomplete old-gate attribution.**
6. **Disclosure gaps:** a failed unit, deviations 6, 12, 18 and 19, and the P4b window.
7. **Reader-relative wording** missing in several places.

**Dispositions.** 130 statements: 60 SUPPORTED, 67 QUALIFY, 2 REMOVE, 1 not verifiable yet.

---

## A. RESEARCH_DECISION.md

| # | Line | Statement | Disposition | Record and numbers | Required edit |
|---|---|---|---|---|---|
| A1 | 4-9 | Scope: roles used; MODEL_SELECT adaptive; ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels closed; nothing is confirmation; source label unchanged | SUPPORTED | ACCESS_RECEIPT.json `ATTACK_or_DEV_REPORT_label_reads` = [], `lineage_dev_report_log.unchanged` = true; PROTOCOL.md L68 | none |
| A2 | 10 | "Every number below comes from the records listed in START_HERE.md" | QUALIFY | The P70 values in §2 (+3.2 / +1.9, -11.3 / -7.9) are not in TRAINING_SIZE_CURVE.json, which L100 cites and which holds only paired differences. They are in FINDINGS_NOTES F7 and the private P70 evaluation. The Track R income values (§7) are in F11 and the private TRACKR_EVAL_MS. | Cite F7 at L100 and F11 at L193. |
| A3 | 14, 203 | "a validated diagnosis" | QUALIFY | Replicated on four overlapping partitions and replayed numerically by the verifier (ladders within 6.7e-18). But the nesting attribution rests on D0 / CHECK only (A16); the size reading is ADAPTIVE (TRAINING_SIZE_BET `label`); and the torch-trained directions were not refitted (verifier REPORT, "Not covered"). | Say "a replicated, independently replayed diagnosis (development evidence)". Apply the same change at SH L12 and AB L44. |
| A4 | 17-21 | Q A no; OOF biased, men's error changes sign; GUARD unbiased within noise, noise as large as slack | SUPPORTED | TRANSFER_FACTORS REFIT_PAIR NLL MS-OOF -1.93 / -4.18 / -3.16 / -4.46e-3; men's CHECK-OOF +0.03 / -5.14 / +2.09 / -2.76; slack +0.49 / -0.78 / +0.14 / -0.31 | none |
| A5 | 22-23 | Neither PAIR nor LOC better under the registered bootstrap on any partition; no direction on all four under either scheme | SUPPORTED | QUESTION_A_LABELS.json `direction_arms_improved_on_all_four` = []; every REFIT_PAIR / LOC / PAIR_TR row is NOT_IMPROVED | none |
| A6 | 24-25 | "Two R1 labels depend on the scheme. They were corrected after the repair panels had been read with the frozen code (PROTOCOL deviation 16)." | QUALIFY | QUESTION_A_LABELS.csv: R1 REFIT_LOC full q05 +0.105 to -0.020; R1 REFIT_PAIR_TR full +0.050 to -0.020; `flag` SCHEME_DEPENDENT; A5 resolved as (b) | Replace with: "Two R1 labels (REFIT_LOC full, REFIT_PAIR_TR full) are SCHEME_DEPENDENT: an improvement appeared only under the superseded frozen implementation and is not established under the registered scheme, which adjudication A5 makes primary (the correction's trigger used D-stage inputs only; verifier P4 V2 PASS). The correction was committed after the repair panels had been read with the frozen code (deviations 16, 18)." |
| A7 | 28-30 | "every frozen strength rule chose identity or a tiny strength, and recipe selection gave NO_RECIPE before any repair panel was opened" | QUALIFY | Full deployment, correction directions: at most 0.129 (D0 REFIT LOC OLD-B80). Ensemble: SIMPLE chose 0.6 / 0.6 (D0 FROZEN PAIR / LOC), 0.3 / 0.4 (D0 REFIT PAIR / LOC) and 0.5 / 0.3 / 0.2 (R0 PAIR / LOC / PAIR_TR); all crossed (RULE_OUTCOMES.csv; RULE_OUTCOMES_REPAIR.csv, C0 slack use 13.8 / 7.8 / 4.8). The timing is supported: RECIPE_SELECTION 20:06:53Z, before the freeze (20:23:59Z) and the first C0 read (20:25:19Z). | "chose identity or at most 0.13 in the full deployment; in the ensemble deployment SIMPLE chose 0.2-0.6, and every such selection crossed the reserve" |
| A8 | 31-32 | Correction lowers the development readers' recovery repeatably beyond W and CONT, only at strengths many times the slack | SUPPORTED | MATCHED_CONTROLS.csv: STABLE_POSITIVE vs W only at s = 1.0 (+0.0412), vs CONT at s ≥ 0.3; CLAIM_LADDER u = 13.7-70.1 | Optional: "beyond W only at s = 1.0; beyond CONT at s ≥ 0.3" |
| A9 | 35 | "The old per-cell recipient gate rejects most true-zero differences on the income view." | QUALIFY | SCREEN_STUDY.md, law L1, income, n_R = 2, mu = 0: old-gate pass 0.1903 (rho_seed 0) and 0.3120 (rho_seed 0.76) | Add: "under simulation law L1 with two household repetitions (pass 0.19-0.31); a property of that law, not a universal error rate" |
| A10 | 36-38 | Every H3-anchored release fails through the income view alone; every comparator fails the reserve; no protective release qualifies | SUPPORTED | PRE_OUTCOME_H3_INCOME_VIEW.json `old_gate_fails_against` [H_LJ, TWJ], utc 19:17:07Z, labels_read false; RESERVE_QUALIFIED_FRONTIER.csv reference ratios 0.321-1.947 | L38: "no release that lowers the development readers' recovery qualifies" |
| A11 | 40-41 | "The source witness's margin depends on its training-set size. Refits ... gain occupation NLL and men's-guard margin smoothly as the training share grows from 42% to 70% to 100%" | QUALIFY | TRAINING_SIZE_BET `reading_rule.near_log_linear`: "consistent with a size effect being sufficient; cannot exclude a fit-specific selection component at 100%". Its `may_not` list forbids "causal attribution to size". The men's-guard excess on s0 is not monotone: +0.65 (42%), +1.11 (53%), -1.79 (70%), -3.9 (100%, the source fit). Occupation NLL is monotone: -5.76 / -5.99 / -11.27 / -16.1. | "The source witness's margin is consistent with a training-size effect: occupation NLL improves monotonically from 42% to 53% to 70% of MODEL_FIT toward the source fit at 100%; the men's guard improves from 53% to 70% (not from 42% to 53%). A fit-specific selection component at 100% is not excluded." |
| A12 | 46 | "four genuinely different training partitions" | QUALIFY | PROTOCOL.md L78: the D0 / D1 FIT sets share 3,329 households and R0 / R1 share 5,268; all four are rotations of the same MODEL_FIT people | "four different, overlapping training partitions of MODEL_FIT" (also at L157 and L205) |
| A13 | 50 | Heading "A systematic over-prediction, removed by nesting." | QUALIFY | see A16 | "A systematic over-prediction; nesting reduced it on one partition (D0)" |
| A14 | 51-52 | OOF over-predicts MS NLL on all four: -1.9 / -4.2 / -3.2 / -4.5 | SUPPORTED | `{P}\|REFIT_PAIR\|nll` MS-OOF -1.928 / -4.182 / -3.158 / -4.457e-3 (z -1.1 / -2.6 / -2.1 / -2.9) | Name the REFIT-routed PAIR; note that D0 alone is not resolved (z -1.1). |
| A15 | 53 | CONT MS z from -7.1 to -10.3 | SUPPORTED | REFIT_CONT NLL MS-OOF mean_z -7.10 / -10.26 / -10.16 / -9.40 | none |
| A16 | 54-56 | "Nested cross-fitting removes most of the NLL over-prediction on D0 (CHECK minus nested +0.97, z +0.5) ... So the non-nested upstream cross-fitting is the main source of the NLL bias." | QUALIFY (priority 1) | D0\|REFIT_PAIR\|nll: NESTED_OOF−OOF -4.79e-3 (z -3.05); CHECK−NESTED +0.97 (z +0.54). But MS−NESTED is +2.86 (z +1.96), larger than the MS−OOF error it replaces (-1.93, z -1.13). D0\|REFIT_LOC\|nll: CHECK−NESTED +2.96 (z +1.62), MS−NESTED +2.43 (z +1.90). There was no nested run for CONT (D0\|REFIT_CONT NESTED = null; no NESTED_OOF CONT pairs in COST_TRANSFER_SUMMARY.json). In question A with NESTED in place of OOF, GUARD was closer to CHECK (D +0.42 PAIR, +0.20 LOC; unresolved). | Replace with: "On D0, the only partition where it was run, nesting every upstream layer lowered the predicted NLL increase by 4.8e-3 (z -3.1). Against the CHECK panel this removes the over-prediction (+0.97, z +0.5); against MODEL_SELECT it over-corrects (+2.86, z +2.0), and for LOC it under-predicts on both panels (+2.96, +2.43). Non-nested cross-fitting is the likely main source of the bias; one partition cannot establish it." |
| A17 | 58-59 | Men's CHECK−OOF +0.03 / -5.14 (z -4.0) / +2.09 (z +1.5) / -2.76 (z -2.2) | SUPPORTED | REFIT_PAIR men_brier CHECK-OOF +0.029 / -5.142 / +2.089 / -2.760; z +0.09 / -3.96 / +1.53 / -2.22 | Name the REFIT routing. |
| A18 | 60 | "Nesting does not fix it: on D0 the nested estimate under-predicts by +2.70 (z +2.0)." | QUALIFY | On D0 the OOF men's error was +0.03 (z +0.09). Nested: CHECK +2.70 (z +1.98), MS +2.12 (z +2.20). | "On D0, where the out-of-fold men's error was +0.03, the nested estimate errs by +2.70 (z +2.0): nesting does not remove this kind of error and can introduce it." |
| A19 | 61-62 | Source +2.0 / +3.0 on seed 0, -1.2 / -0.9 on seed 1 | SUPPORTED | SOURCE_REPLAY.json, B80, MS−OOF men's: P2_PAIR S0 +2.02, P2A_PAIR S0 +2.96; S1 -1.18 / -0.91 | Add "(B80; MODEL_SELECT minus OOF)": the refit errors above are CHECK minus OOF. |
| A20 | 64-65 | SE about 2e-3 on GUARD / CHECK and 1.1-1.2e-3 on MS | QUALIFY (minor) | COST_TRANSFER.csv, REFIT_PAIR s = 1: GUARD / CHECK men's 1.66-2.19, women's 1.46-2.02; MS men's 1.12-1.29, women's 0.95-1.18 (×1e-3) | "1.5-2.2e-3" and "1.0-1.3e-3" |
| A21 | 66 | Slack +0.49 / -0.78 / +0.14 / -0.31 | SUPPORTED | Verifier REPORT, checks 3-4 | none |
| A22 | 68-72 | "It was biased by non-nested cross-fitting"; the men's cost varies by more than the reserve | QUALIFY (first) / SUPPORTED (second) | First: see A16. Second: realized MS men's cost at s = 1 is 10.52 / 3.86 / 9.91 / 5.01e-3 (COST_TRANSFER.csv), against slack ≤ 0.49e-3. | "It over-predicted, most likely because of non-nested cross-fitting (a one-partition check)." |
| A23 | 78-79 | FROZEN vs REFIT give the same pattern (D1 men's -3.97 vs -5.14) | SUPPORTED | FROZEN_PAIR -3.975 (z -2.93); REFIT_PAIR -5.142 | none |
| A24 | 81-82 | "every CHECK minus GUARD and MS minus GUARD error lies within \|z\| <= 2.4" | QUALIFY (number wrong) | TRANSFER_FACTORS, 180 GUARD entries. Maximum \|z\| is 2.88 (D1\|FROZEN_LOC\|men_brier CHECK−GUARD, -3.46e-3), then 2.55 (R0\|REFIT_PAIR\|brier, +2.76e-3) and 2.42 (R0\|REFIT_PAIR\|nll, +6.01e-3). Nine of the 180 exceed 2. The R0 PAIR CHECK−GUARD errors (Brier, NLL, men's) are all positive, with z 2.2-2.5. | "within \|z\| <= 2.9 (9 of 180 above 2, about the number expected by chance), with signs that vary between partitions" |
| A25 | 83-87 | GUARD better for CONT on D1 and R0, and for FROZEN PAIR D1 ens (A1_NOT_SUPPORTED) | SUPPORTED | QUESTION_A_LABELS.json `prediction_improved_primary` (5 units) | none |
| A26 | 88-89 | "GUARD is never better under the registered scheme. It wins where OOF is biased and loses where its own noise dominates." | QUALIFY | The two sentences contradict each other unless "wins" means point estimates. PAIR full D: +0.16 (D1), +0.38 (R1), -0.51 (D0), -2.21 (R0). q05 < 0 in every PAIR / LOC row. On R0, where the OOF men's error is +2.09, GUARD still loses. | "Its point estimate favours GUARD on D1 and R1 and OOF on D0 and R0; no PAIR or LOC difference is resolved." |
| A27 | 90-91 | Three flips: D0 CONT ens, R1 LOC full, R1 Track R PAIR full | SUPPORTED | CSV `flag` SCHEME_DEPENDENT; superseded q05 +0.010 / +0.105 / +0.050, registered -0.059 / -0.020 / -0.020 | none |
| A28 | 92-93 | "...corrected after the repair panels had been read with the frozen code (deviation 16; adjudication A5). Under the frozen code they are PREDICTION_IMPROVED." | QUALIFY (A5 final) | A5 resolution. V_TRIGGER__V2: P4 PASS, P4b ATTENTION. C3.QA.R1.VARIANCE_DETAIL__V3: the superseded SE is 25% / 20% smaller; the top 10 households carry 5.7% / 4.9% of the variance; the label stays NOT_IMPROVED after dropping 1, 5 or 10 households. | "Under adjudication A5 the registered-scheme labels (NOT_IMPROVED) are primary; the frozen-code labels (PREDICTION_IMPROVED) are superseded output of an implementation that departed from RULES.md section 6. The shift comes from the superseded scheme's smaller SE, not from heavy tails. Disclosed window: the verifier's R1 replays wrote these two labels to its CHECKS.json 29 s before the correction commit; no coordinator read is logged (P4b; deviations 16, 18)." |
| A29 | 94 | Claim ladder unaffected | SUPPORTED | CLAIM_LADDER `IMPROVED_COST_PREDICTION` NOT_MET at every strength | none |
| A30 | 96-99 | Source -3.9 / -2.3; refits +0.65 / +2.15 / +1.11 / +1.89; point limits pass; slack values | SUPPORTED | TRAINING_SIZE_BET `observed_before_the_fit`; verifier | none |
| A31 | 100-103 | P70: men's slack +3.2 / +1.9, NLL -11.3 / -7.9; "at or slightly beyond" the log-linear prediction; "A size effect is sufficient to explain it" | QUALIFY | Values match the P70 evaluation: reserve slack guard\|sex0 +0.00316 / +0.00187; NLL excess -0.01127 / -0.00792; prediction -0.0105 / -0.0069. | "consistent with a size effect being sufficient" (the registered reading). Cite F7 and the P70 evaluation, not TRAINING_SIZE_CURVE.json. |
| A32 | 104 | P70A − P70 NLL -0.7 (SE 0.4) | SUPPORTED | TRAINING_SIZE_CURVE: -0.000716 (SE 0.000391) | none |
| A33 | 108 | §3 answer: "Only at negligible strengths, and never with the reserve." | QUALIFY (plainness) | Point pass (24 clauses + 4 guards) holds for s ≤ 0.5 (RESERVE_QUALIFIED_FRONTIER `point_pass_24_4`). The reserve is kept at no common strength ≥ 0.1 (CLAIM_LADDER `ms_reserve_kept` false). The only non-crossing nonzero choice was R0 OLD-B50 at 0.017 (slack use 0.94 on C0, 0.60 on MS; not reader-evaluated). | "At the point, up to s = 0.5. With the two-SE reserve, at no evaluated nonzero strength; the only non-crossing nonzero choice was 0.017 on R0, and on R1 the witness itself fails the reserve." |
| A34 | 111-112 | R1 identity everywhere; witness ratio 1.82 | SUPPORTED | RULE_OUTCOMES_REPAIR R1: all 0.0; RESERVE_QUALIFIED_FRONTIER W seed-1 ratio 1.821 | none |
| A35 | 113-117 | R0 full: OLD-B50 0.017 (0.94 / 0.60), OLD-B80 0.027 (1.50); SIMPLE and UNCERTAINTY identity; ensemble SIMPLE 0.5 (13.8 / 10.5) | QUALIFY (numbers supported) | RULE_OUTCOMES_REPAIR.csv. Deviation 12: rule selections were not reader-evaluated. Verifier C6.6.DECISION_STATEMENTS__V1: the uncorrected ensemble e_0 already uses 4.0 times W's men's slack (men's cost 0.00057). Deviation 6: the ensemble's "identity" is e_0, and its budget is e_s − e_0. | Add "(rule selections not reader-evaluated; deviation 12)". Add: "Ensemble slack use is the increment over the uncorrected ensemble e_0, which already uses 4.0 times W's men's slack on C0 (deviation 6; verifier ATTENTION), so every ensemble release crosses W's reserve." |
| A36 | 118 | GUARD rules never improved on the source rule on both R partitions | QUALIFY | RULE_COMPARISON.json: 24 of 24 have `a_improves_on_both` false. R1 is SAME_STRENGTH everywhere. The registered note says "identity never improves". | Add: "This was close to forced: on R1 every rule chose identity, and compare_rules never counts identity as an improvement, even where it avoided OLD-B80's C0 crossing (1.50)." |
| A37 | 120-123 | Point pass s ≤ 0.5; min ratio 1.78 / 1.73 to 0.06 / 0.86; u 3.8 and 26.6; R1 infinite | SUPPORTED | RESERVE_QUALIFIED_FRONTIER.csv; CLAIM_LADDER u 3.815 / 26.567 | none |
| A38 | 124-125 | The men's guard binds at every reader-evaluated point | SUPPORTED | FRONTIER_DRIVERS.csv: 36 of 36 `guard\|sex0` | none |
| A39 | 129 | §4 answer (yes vs W and CONT at large strengths only; not vs LOC) | QUALIFY (wording) | MATCHED_CONTROLS.csv | Add "for the declared development readers". |
| A40 | 135-140 | §4 table | SUPPORTED | MATCHED_CONTROLS.csv. W: +0.0412 STABLE_POSITIVE / +0.0279 positive, not stable / +0.0086 positive, not stable. CONT: +0.0417 / +0.0285 / +0.0113, all STABLE_POSITIVE. LOC: +0.0096 sign changes / +0.0131 positive, not stable / +0.0008 sign changes. COMP: +0.0170 / +0.0178 / +0.0043 over 4 cells, marked not cost-matched. D stage (D_FRONTIER_KEY): +0.0405, +0.0342, +0.0133. | none |
| A41 | 142-143 | "CONT alone reads like the witness: -0.0006 to -0.0028 vs W, never positive. So the critic ... lowers recovery." | QUALIFY | R stage: CONT vs W -0.00056 to -0.00275, NOT_POSITIVE. D stage: CONT@1.0 vs W +0.0063, sign changes (READER_STABILITY_D_FRONTIER_KEY; F6). | Make it "On the repair stage ... never positive (diagnostic stage: +0.0063 at s = 1.0, sign changes)". Make the conclusion reader-relative. |
| A42 | 144-145 | LOC +0.032 vs W; the coalition-aware part adds +0.010, not stable | SUPPORTED | +0.0315 positive, not stable; PAIR vs LOC +0.0096, sign changes | none |
| A43 | 146-149 | COMP on R0 only, not cost-matched, 6-55% of target, descriptive | SUPPORTED | Deviations 13 and 17; `NOT_EVALUABLE_4_CELLS` | none |
| A44 | 150-151 | Men's cost 14-70 times the R0 slack at the stable strengths; unbounded on R1 | QUALIFY (minor) | CLAIM_LADDER u = 13.74 (0.3), 26.57 (0.5), 70.05 (1.0) | "where the benefit over CONT is stable (s ≥ 0.3; over W only at s = 1.0)" |
| A45 | 155 | "The recovery effect survives." | QUALIFY | — | "The lower recovery for these development readers survives; utility does not." |
| A46 | 157 | Same ordering on D0, D1, R0 and R1 | SUPPORTED | Per-seed means | Use "overlapping partitions" (A12). |
| A47 | 158 | Routing -0.0023 / +0.0016 | SUPPORTED | D_FRONTIER_KEY, FROZEN_PAIR vs REFIT_PAIR | none |
| A48 | 159-163 | Track R: teachers bitwise identical; +0.047 / +0.023 STABLE_POSITIVE; +0.006 vs Track F, sign changes | SUPPORTED | Verifier identity check; R_TRACKR +0.0470 / +0.0234 / +0.0057 | Optional: add s = 0.5 vs Track F, -0.0047 NOT_POSITIVE. |
| A49 | 164-165 | Track R fails the reserve: R0 income Ucal Brier 1.92; R1 men's guard 1.82 | SUPPORTED | TRACKR_EVAL_MS: 1.922 and 1.821 | none |
| A50 | 172-174 | "It fails the old recipient gate at every strength, through H3's income view." | QUALIFY | ORIGINAL_BENCHMARK.csv `gates`: the income view fails (vs H_LJ and TWJ) at every strength. At s = 0.1-0.5 the occupation view also fails: against six comparators at s = 0.1 / 0.2, and against all except NEURAL_RAW at s = 0.3 / 0.5. Only at s = 1.0 is the income view the sole cause. | "...fails the old gate at every strength: through H3's income view (vs H_LJ, TWJ) at every strength, and at s ≤ 0.5 also through its occupation view (vs five or six comparators)." |
| A51 | 175-178 | Superiority values at s = 1.0 and 0.5 | SUPPORTED | MATCHED_CONTROLS R_FRONTIER, values as quoted | none |
| A52 | 179-180 | The 0.020 target is POINT_ONLY at s = 1.0, where utility fails | SUPPORTED | CLAIM_LADDER `target_label`; `point_pass` false at 1.0 | none |
| A53 | 181-185 | Reserve-qualified frontier; "No reserve-qualified protected control exists." | QUALIFY (numbers supported) | RESERVE_QUALIFIED_FRONTIER.json `qualified_protected_points` = [REFIT_CONT@0.3, @0.5, @1.0]; pair AUC 0.636 / 0.634 / 0.634 vs W 0.634; reference ratios 0.32-1.95 | "The record lists these three CONT points as `qualified_protected_points`; none lowers the readers' recovery relative to W (CONT vs W -0.0006 to -0.0028), so no reserve-qualified point lowers recovery." |
| A54 | 167 | §6 has no lead answer | QUALIFY (plainness) | — | Add: "Neither comparison favours the correction: nothing is competitive on the original benchmark, and no reserve-qualified point lowers recovery." Add a similar lead to §7. |
| A55 | 189-192 | Income view: old gate +0.0309 / +0.0147; AGG1 0.0227 / 0.0159; recorded before any outcome | SUPPORTED | PRE_OUTCOME_H3_INCOME_VIEW.json per_comparator; utc 19:17:07Z | none |
| A56 | 193-197 | "On 75% of MODEL_FIT plus CALIBRATION", with the income values | QUALIFY (number wrong) | PROTOCOL.md L75: R FIT is 75% of the non-panel pool, about 53% of MODEL_FIT (10,523 / 10,506 people). The values are correct (TRACKR_EVAL_MS): H3R Brier +1.63 / +1.12, NLL +0.96 / +0.55, accuracy -1.75 / +1.37; frozen H3 -2.79 / -3.30, -4.69 / -5.15, +0.12 / +1.75 (×1e-3). | "on the R partitions' FIT sets (75% of the non-panel pool, about 53% of MODEL_FIT) plus CALIBRATION". Add: "one refit per partition, at about half H3's training size, so training size and recipe are not separated; a recipe test on different training subsets, not a test of the frozen H3 package". Add an ERRATA entry for F11's "75% of MODEL_FIT". |
| A57 | 198-199 | The income asset "does not survive refitting" | QUALIFY | A56 | "...does not survive a refit on about half of MODEL_FIT" |
| A58 | 205 | "replicated on four training partitions and verified with independent code" | QUALIFY | Verifier REPORT, "Not covered": the torch-trained Stage B and Track F directions were not refitted | "replicated on four overlapping partitions and numerically replayed with independent code (directions not refitted)" |
| A59 | 206-207 | "A systematic over-prediction of NLL and of the continuation's costs comes from non-nested cross-fitting, and nesting removes it." | REMOVE (CONT part) + QUALIFY | Nesting was never run for CONT (A16). | "A systematic over-prediction of NLL, and a larger one of CONT's own small costs. On D0 nesting removed the NLL over-prediction against CHECK but over-corrected against MODEL_SELECT; non-nested cross-fitting is the likely source. Nesting was not run for CONT." |
| A60 | 208 | "A partition-specific men's-cost error of 2-5e-3 is not removed by nesting." | QUALIFY | +0.03 / -5.14 / +2.09 / -2.76; nested on D0 +2.70 | "of up to 5e-3 (about 0 on D0); on D0 the nested estimate itself errs by +2.70 (z +2.0)" |
| A61 | 209 | GUARD unbiased; noise as large as slack | SUPPORTED | A24 | none |
| A62 | 210 | "through a training-size effect" | QUALIFY | TRAINING_SIZE_BET `may_not` | "consistent with a training-size effect (adaptive diagnostic; a selection component at 100% is not excluded)" |
| A63 | 212 | NO_RECIPE: all 32 recipes fail G1 on D1 | SUPPORTED | RECIPE_GATES: 32 of 32 fail G1 on D1; `survivors_G1_G3` = [] | none |
| A64 | 213 | GUARD rules never improve | QUALIFY | A36 | as A36 |
| A65 | 214-215 | No nonzero strength keeps the reserve; no reserve-qualified protective point | SUPPORTED | CLAIM_LADDER `ms_reserve_kept` false everywhere | Wording as A53. |
| A66 | 216-218 | Not competitive; repeatable benefit only at costs far outside the budget | QUALIFY (wording) | — | Add "for the declared development readers". |
| A67 | 219-221 | Rung 0; JOINT_DESIGN_ADVANTAGE not claimable; NOVELTY not claimed; no impossibility claim | SUPPORTED | CLAIM_LADDER `highest_claimable_rung` 0 at every strength | none |
| A68 | 227 | Diagnosis replicated "for PAIR, LOC, CONT and both routings" | QUALIFY | TRANSFER_FACTORS has FROZEN arms on D0 / D1 only | "both routings on D0 / D1; REFIT only on R0 / R1" |
| A69 | 228 | "NLL and CONT over-prediction: systematic (every partition)" | QUALIFY | Holds on MS for every partition. On the R0 CHECK panel: PAIR +0.58 (z +0.3), LOC +0.01. | "systematic on MODEL_SELECT (every partition; the R0 CHECK panel shows none for PAIR or LOC)" |
| A70 | 229 | "D0 only: nesting removes the NLL bias and under-predicts the men's cost" | QUALIFY | A16, A18 | "D0 only: nesting removes PAIR's NLL over-prediction against CHECK but over-corrects against MODEL_SELECT (+2.9e-3, z 2.0) and for LOC; it under-predicts the men's cost (+2.7e-3, z 2.0); not run for CONT; in question A, GUARD was closer to CHECK than the nested estimate (D +0.42 / +0.20, unresolved)" |
| A71 | 230 | Simple-rule row (UNCERTAINTY identity in full / 0.1 ensemble on D0 only; SIMPLE 0.1 LOC on D0, crossing ensemble choices, identity elsewhere; neither improves) | QUALIFY | True only for the correction directions. For the control CONT: UNCERTAINTY full chose 0.2 / 0.5 on D0, and 0.1-0.2 (D0) and 0.1 (R0) in the ensemble; SIMPLE full chose 0.5 / 1.0 on D0 and 0.1 on R0, and 0.2 / 1.0 (D0) and 0.7 (R0) in the ensemble. None of these crossed. | Add "for the correction directions (the control CONT received nonzero, non-crossing strengths)" and the A36 qualifier. |
| A72 | 231 | Men's guard binds; recovery falls only far over budget | SUPPORTED | FRONTIER_DRIVERS: 36 of 36 | Make it reader-relative. |
| A73 | 232 | Screen study done before any new candidate was scored | SUPPORTED | SCREEN_STUDY.json utc 19:51:50Z; the first candidate grid (CTS_GRID_D_FRONTIER_S0_V1) started at 19:56:31Z | none |
| A74 | 233 | "Verified with separate code (bitwise refits)" | QUALIFY | The verifier refitted the witnesses, routing, Track R teachers and Step-1 HGB bitwise; the directions were not refitted | Add "(directions not refitted)". |
| A75 | 234 | P70 packaged; bitwise, every refusal | SUPPORTED | DEPLOYMENT_CHECKS_P70.json: all_bitwise, all_refused, labels_read false | Add "adaptive fit (amendment 9); never a selected release". |
| A76 | 238-239 | Use nested CF instead of a GUARD subset of "25-40%"; "The training-size effect costs more slack than the GUARD estimate saves." | QUALIFY | Untested remedy. On D0 the nested estimate was not closer to CHECK than GUARD (question A NESTED rows: D +0.42 / +0.20), and it under-predicted the men's cost (+2.70) and MS NLL (+2.86). GUARD is 25% of the pool on R and 20% on D (40% with CHECK on D). GUARD never improved PAIR or LOC. | State these caveats; give the fractions as above. Replace the second sentence with: "The training-size loss is large, and the GUARD estimate did not predict better." |
| A77 | 240-242 | 8-16 times as many people; beyond the development roles | SUPPORTED (projection) | SE ∝ 1/√n from 2e-3 at about 1,400 per group; MODEL_FIT 20,000, MODEL_SELECT 8,001 | none |
| A78 | 243-244 | One reserve rule; a screen with a known zero-difference pass rate | SUPPORTED | — | none |
| A79 | 246-259 | Failed attempts list | QUALIFY (incomplete) | Missing: deviation 6 (a registered computation changed for the ensemble gates; ensemble budget and identity not patched); deviations 12, 16, 18 and 19; adjudications A3, A4 and A6; the verifier's genuine FAILs (F3 V2, C2.1); the P4b window. Further time-stamp errors: the A2 heading says "about 20:35Z" but the file was created 20:30:14Z; the A5 heading says "about 20:50Z" but the file was written 21:04:46Z; the verifier's "20:29Z" message was sent between 20:27:28Z and 20:28:04Z. | Add one line for each, plus a pointer: "all 19 items are in PROTOCOL_DEVIATIONS.md". |
| A80 | ~260 | "Units: 0 failed queue units." | REMOVE (correct it) | RESOURCE_ACCOUNTING.json `units_failed` = 1: VER_QA_INFLUENCE_R1_V1 (returncode 1; the access layer refused role 'OOF' before any label read; rerun as V2 / V3; NOTES.jsonl 21:10:46Z). VER_STRUCT_V1 was registered but never launched (NOTES.jsonl). EXPERIMENT_LEDGER.csv (287 rows, all done) lags the accounting (289 done, 1 failed). | "1 failed queue unit (verifier VER_QA_INFLUENCE_R1_V1, refused before any label read; rerun); 1 verifier unit registered but never launched". Regenerate the ledger before closeout. |
| A81 | 262+ | Not claimed | SUPPORTED | — | none |

## B. START_HERE.md

| # | Line | Statement | Disposition | Record and numbers | Required edit |
|---|---|---|---|---|---|
| B1 | 3-10 | Scope, the three questions, source label unchanged | SUPPORTED | — | none |
| B2 | 12 | "A validated diagnosis" | QUALIFY | A3 | as A3 |
| B3 | 12-13 | "biased by non-nested cross-fitting" | QUALIFY | A16 | "over-predicts NLL on every partition, most likely through non-nested cross-fitting (a one-partition check)" |
| B4 | 13-14 | Men's error changes sign; GUARD unbiased but noisy; NO_RECIPE | SUPPORTED | A4, A63 | none |
| B5 | 14-16 | "lowers recovery repeatably ... no release that protects qualifies" | QUALIFY (reader-relative) | MATCHED_CONTROLS; RESERVE_QUALIFIED_FRONTIER | "lowers the declared development readers' recovery ...; no release that lowers that recovery qualifies" |
| B6 | 22-23 | Registration receipt verified before any fit or label read | SUPPORTED | RECEIPTS.json `problems` []; verifier checks 1-2 | none |
| B7 | 30-32 | QUESTION_A_LABELS with superseded labels beside the primary ones | SUPPORTED | CSV `superseded_*`, `flag` | Add "flag SCHEME_DEPENDENT; primary per A2 (D) and A5 (R)". |
| B8-B10 | 38-45 | Track R records; verification records; 10 packages plus P70 | SUPPORTED | DEPLOYMENT_CHECKS.json: 10 packages, all_bitwise, all_refused, labels_read false | Add the REPORT addendum, the V_TRIGGER and V_QA_INFLUENCE records, and A1-A6. |
| B11 | 46-47 | Closeout files | QUALIFY | BACKUP_VERIFICATION.json, HANDOFF.json and PUBLICATION_RECORD.json do not exist. ACCESS_RECEIPT.json, RESOURCE_ACCOUNTING.json and EXPERIMENT_LEDGER.csv were untracked at review time. | Create and commit them before publication, or mark them pending. |
| B12 | 48-49 | FINDINGS_NOTES F1-F12 | SUPPORTED | — | Note that F8 holds superseded labels (F9 corrects them) and that F11 has the "75%" error (ERRATA). |
| B13 | 53-55 | Private tree and sanitized publication branch | Not verifiable yet (PUBLICATION_RECORD.json pending) | — | none now |

## C. ADVISOR_BRIEF.md

| # | Line | Statement | Disposition | Record and numbers | Required edit |
|---|---|---|---|---|---|
| C1 | 3-4 | The source chose its strength from OOF costs and missed the men's cost | SUPPORTED | SOURCE_REPLAY: seed 0 +2.0 / +3.0e-3 | none |
| C2 | 9-10 | Development roles only; ATTACK and DEV_REPORT labels closed | SUPPORTED | ACCESS_RECEIPT | Optional: "MODEL_SELECT used adaptively in earlier campaigns". |
| C3 | 11-12 | "We refitted the whole pipeline on four training partitions, each excluding GUARD and CHECK households" | QUALIFY (scope) | Track F refits only the occupation side. H3's frozen income was fitted on all of MODEL_FIT, including the GUARD and CHECK households. The FROZEN arm uses the admitted teacher (A1_NOT_SUPPORTED). Only Track R (R0, R1) refits the income lineage. | "We refitted the occupation side (witness, routing teacher, correction directions) on four overlapping training partitions that exclude GUARD and CHECK households, with H3's income frozen (Track F); on the two repair partitions we also refitted the income lineage (Track R)." |
| C4-C5 | 13-15 | Rules frozen before any fit; readers; Track R "once"; verifier replay; ten packages | SUPPORTED | — | "once per repair partition" |
| C6 | 19-20 | "Non-nested cross-fitting biases it: ... 2-5e-3 on every partition. Nesting removed most of this on ... D0." | QUALIFY | A14, A16 | "It over-predicts the NLL increase on MODEL_SELECT by 1.9-4.5e-3 on every partition. On D0, the one partition where nesting was run, nesting removed this against the CHECK panel but over-corrected against MODEL_SELECT (+2.9e-3, z 2.0); non-nested cross-fitting is the likely, not established, cause." |
| C7 | 21-22 | "The men's cost of a fixed correction changes ... by 2-5e-3, with either sign." | QUALIFY | Those numbers are the OOF estimate's error (CHECK − OOF: +0.03 / -5.14 / +2.09 / -2.76), not the change in cost. The realized MS cost at s = 1 ranges from 3.9 to 10.5e-3. | "The out-of-fold estimate's men's-cost error ranges from +2.1 to -5.1e-3 across partitions (about 0 on D0): more than the reserve slack (+0.5 to -0.8e-3)." |
| C8 | 23-25 | "...under the frozen code they would read as improved" | QUALIFY (A5 final; the first two sentences are SUPPORTED) | A5 | "Two R1 labels are scheme-dependent: an improvement appeared only under a superseded implementation and is not established under the registered scheme, which adjudication A5 makes primary (the correction, committed after the panels were read, was triggered by D-stage evidence only)." |
| C9 | 27-28 | "The source witness's margin came largely from training on all 20,000 people." | QUALIFY (numbers supported) | TRAINING_SIZE_BET `may_not` | "appears to come largely from ... (adaptive diagnostic; selection not excluded)" |
| C10 | 29 | "every rule picks identity or a tiny strength" | QUALIFY | A7 | as A7 |
| C11 | 30 | "restores most of the margin (+3.2 / +1.9e-3)" | QUALIFY | +3.2 / +1.9 is reserve slack, not margin. The men's-guard excess at 70% (-1.79 / -0.26) closes 51-58% of the gap between the 42-53% refits and the source (-3.9 / -2.3); for NLL it is 52-60%. | "...restores a positive men's reserve slack (+3.2 / +1.9e-3) and about half of the gap to the source witness" |
| C12 | 31-33 | "The correction does lower recovery ... +0.03 to +0.04 AUC" | QUALIFY (numbers supported) | R stage +0.041 / +0.042; D stage +0.041 / +0.034 | Heading: "lowers these development readers' recovery". |
| C13 | 34 | Not stable vs LOC, which alone gives about +0.03 | SUPPORTED | R stage +0.0315; D stage +0.0272 | none |
| C14 | 35 | "At those strengths the men's cost is 14-70 times the available slack." | QUALIFY | At full strength u is 70.05 on MS (94.5 on CHECK) and infinite on R1; the 14-70 range spans s = 0.3-1.0 | "At full strength the men's cost is 70 times the R0 slack (unbounded on R1); 14 times already at s = 0.3." |
| C15 | 36-37 | Survives Track R; "H3's Brier / NLL headroom disappears when its recipe is refitted" | SUPPORTED / QUALIFY | A48; A56 | "...when its recipe is refitted on about half of MODEL_FIT (training size not separated)" |
| C16 | 39 | The income view alone fails the old gate | SUPPORTED | PRE_OUTCOME | none |
| C17 | 40 | "about 19% ... under the source-like law" | QUALIFY | SCREEN_STUDY.md L1: 0.19 (rho 0), 0.31 (rho 0.76). The study does not use the term "source-like". | "about 19-31% under simulation law L1 at two household repetitions (a property of that law)" |
| C18 | 41 | Every reference fails the reserve | SUPPORTED | RESERVE_QUALIFIED_FRONTIER | none |
| C19 | 42 | "only the zero-critic continuation qualifies, and it gives no protection" | QUALIFY (reader-relative) | Pair AUC 0.634-0.636 vs W 0.634 | "does not lower the development readers' recovery" |
| C20 | 44-45 | Classification; rung 0 | QUALIFY ("validated"); rung 0 SUPPORTED | A3 | as A3 |
| C21 | 48 | Do not pursue GUARD-subset rules | SUPPORTED | — | none |
| C22 | 49-50 | Use nested CF; "A 1e-3 reserve needs roughly 8-16 times the people of one partition." | QUALIFY | A76. A panel has about 2,800-3,500 people (about 1,400-1,750 per SEX group); a partition has about 14,000-17,000. | "nested cross-fitting is untested as a remedy (on D0 it was no more accurate than GUARD)"; "8-16 times the people of one GUARD or CHECK panel" |
| C23 | 51-52 | Same reserve; a fresh population is needed | SUPPORTED | — | none |

## D. MANUSCRIPT_CLAIMS_HANDOFF.md

| # | Line | Statement | Disposition | Record and numbers | Required edit |
|---|---|---|---|---|---|
| D1 | 3-9 | Use: development evidence; reader-relative definition of recovery | SUPPORTED | — | none |
| D2 | 13-17 | Statement 1, citing records MS-OOF and CHECK-NESTED_OOF | QUALIFY | The two cited records come from different panels. MS-NESTED_OOF is +2.86 (z +1.96); LOC also under-predicts; there is no CONT nested run. | Headline: "over-predicts the NLL increase on MODEL_SELECT on all four partitions; non-nested cross-fitting is the likely source". Qualifier: "one partition and one panel: against MODEL_SELECT the nested estimate under-predicts (+2.9e-3, z 2.0), for LOC on both panels; not run for CONT". Cite MS-NESTED_OOF. |
| D3 | 18-22 | Statement 2: "Its size exceeds the ... slack" | QUALIFY | D0: +0.03 against a slack of +0.49 | "on three of four partitions (D0: +0.03 against a slack of +0.49)"; name the REFIT routing |
| D4 | 23-28 | Statement 3 | SUPPORTED | QUESTION_A_LABELS | Optional: "unbiased within noise (\|z\| up to 2.9)" |
| D5 | 29-33 | Statement 3 qualifiers: "...Under that code they are PREDICTION_IMPROVED ... Report both." | QUALIFY (A5 final) | A5 resolution | "Primary (registered scheme; A2 for D, A5 for R): NOT_IMPROVED. Superseded frozen-implementation output: PREDICTION_IMPROVED, flagged SCHEME_DEPENDENT. Report both, primary first, with 'an improvement appeared only under the superseded implementation; it is not established under the registered scheme' (deviations 10, 11, 16, 18)." |
| D6 | 34-39 | Statement 4: "...and training-set size accounts for it"; "restores most of the margin" | QUALIFY | TRAINING_SIZE_BET `may_not`; C11 | "...; an adaptive diagnostic is consistent with training-set size being sufficient to explain it (a selection component at 100% is not excluded)"; "restores about half of the gap" |
| D7 | 40-42 | Statement 5 | QUALIFY (NO_RECIPE SUPPORTED) | A36 | Add the "close to forced" qualifier. |
| D8 | 43-49 | Statement 6, "14-70 times" | QUALIFY (the rest SUPPORTED) | C14 | "At full strength its realized men's cost is 70 times the R0 slack (unbounded on R1)." |
| D9 | 50-54 | Statement 7 | SUPPORTED | RESERVE_QUALIFIED_FRONTIER | Add that the record's `qualified_protected_points` are these CONT points. |
| D10 | 55-60 | Statement 8: "19% at two household repetitions" | QUALIFY | 0.19 / 0.31 | "19% (L1, rho_seed 0; 31% at rho_seed 0.76)" |
| D11 | 61-64 | Statement 9; record "Track R evaluations, findings F11" | QUALIFY | A56 | Add: "refit on about 53% of MODEL_FIT plus CALIBRATION, one per partition; training size not separated from recipe". Cite TRACKR_EVAL_MS (private) and F11, with an erratum. |
| D12 | 66-73 | Statements that must not be made | SUPPORTED | — | Add: that nesting removes the bias; that training size causes the margin loss; that the reserve-qualified CONT points are protective. |
| D13 | 75-83 | Required disclosures | QUALIFY (incomplete) | PROTOCOL_DEVIATIONS 1-19; ADJUDICATIONS A1-A6; verifier REPORT and addendum | Add deviation 6 (a registered computation changed for the ensemble gates; ensemble identity is e_0, which crosses W's reserve), deviation 18, deviation 19, A5 final (P4 PASS and the P4b window), the verifier's genuine FAILs (F3 V2, C2.1) and the failed verifier unit. Add a pointer covering deviations 1-5, 7 and 15 and adjudications A1, A3, A4 and A6. Change "deviations 8 and 11" to "8, 11 and 19". |

## Brief checklist

- **Confirmation language.** No statement claims confirmation. "Validated" should be softened (A3), and the overlapping partitions
  should not be called "genuinely different" (A12).
- **Track F scope.** Correct in the handoff (MH) and in RESEARCH_DECISION §7. Wrong in AB L11-12 (C3).
- **Track R scope.** Labelled a recipe test only in MH #9. It needs the training-size caveat in RD §7, AB L36-37 and MH #9.
- **Reader-relative recovery.** Good in MH and RD L31. Missing at SH L14-16, RD L38, L129, L142-143, L155, L216 and L231, and AB
  L31-33 and L42.
- **0.020 target and the frontier.** The target is kept separate from superiority, the reserve-qualified frontier is not presented
  as superiority, and identity earns no protection claim: SUPPORTED.
- **Source label and new screens.** NO_QUALIFYING_CONFIGURATION is untouched, and the AGG1 what-if is descriptive: SUPPORTED.
- **Screen rates as properties of their laws.** Missing at RD L35, AB L40 and MH #8.
- **Compression twins described as not cost-matched.** SUPPORTED (RD L140, L148-149; MH disclosures).
- **Novelty and impossibility.** Neither is claimed: SUPPORTED.
- **The eight questions.**
  - §1 and §2 are plain, apart from the nesting wording.
  - §3's answer is ambiguous (A33).
  - §4 and §5 are plain, but need reader-relative wording.
  - §6 and §7 have no one-line answer (A54).
  - §8 is plain, apart from "validated".
- **A5 consistency.** The dependence is stated wherever the units appear: RD L24-25 and L90-93, AB L24-25, MH #3. None of them yet
  uses A5's final designation (primary and superseded) or the required sentence, and none discloses P4b.

---

## REQUIRED EDITS (priority order)

1. **Nesting.** RD L50, L54-56, L68-69, L206-207 and L229; SH L12-13; AB L19-20; MH #1.
   - Rewrite per A16, A59 and A70.
   - The result holds on one partition (D0), for one direction (PAIR) and against one panel (CHECK, +0.97).
   - Against MODEL_SELECT the nested estimate under-predicts by +2.86e-3 (z +1.96); for LOC it is +2.96 and +2.43.
   - Delete the CONT attribution, because nesting was never run for CONT.
   - Call non-nested cross-fitting the likely source, not the established one.
2. **A5 final status.** RD L24-25 and L92-93; AB L24-25; MH #3 qualifiers and disclosures.
   - The registered labels are primary and the frozen-code labels are superseded and flagged SCHEME_DEPENDENT.
   - Use the sentence: "an improvement appeared only under the superseded implementation; it is not established under the
     registered scheme".
   - The shift comes from the superseded scheme's 20-25% smaller SE, not heavy tails.
   - Disclose the P4b 29-second window and cite deviation 18.
3. **Numbers that differ from their records.**
   - RD L82: \|z\| ≤ 2.4 should be 2.9 (the maximum is 2.88).
   - RD L193: "75% of MODEL_FIT" should be about 53% (75% of the pool). Add an ERRATA entry for F11.
   - RD ~L260: "0 failed queue units" should be 1 (VER_QA_INFLUENCE_R1_V1), and 1 unit was never launched.
   - AB L35 and MH #6: the cost is 70 times the slack at full strength, not 14-70; it is unbounded on R1.
   - AB L50: "one partition" should be one GUARD or CHECK panel.
   - AB L30 and MH #4: +3.2 / +1.9 is slack, not margin, and the diagnostic restores about half of the gap, not most.
   - AB L21-22 and RD L208: these are OOF errors, not cost changes; they go up to 5e-3 and are about 0 on D0.
   - RD L65: SE ranges 1.5-2.2e-3 and 1.0-1.3e-3.
   - RD L100: cite F7 and the P70 evaluation, not TRAINING_SIZE_CURVE.json.
4. **Causal training-size language.** RD L40-41, L102 and L210; AB L27-28; MH #4.
   - Say "consistent with a size effect being sufficient; a selection component at 100% is not excluded".
   - Fix "smoothly ... 42% to 70% to 100%": the 53% point is omitted, and the s0 men's guard is not monotone.
5. **Old-gate attribution.** RD L173-174: at s ≤ 0.5 the occupation view also fails, against five or six comparators.
6. **Strength-rule statements.**
   - RD L29 and AB L29: SIMPLE chose 0.2-0.6 in the ensemble deployment, and every one crossed.
   - RD L117: the ensemble crossings belong to e_0, which already uses 4.0 times W's slack (deviation 6; verifier C6.6).
   - RD L113-117: the rule selections were not reader-evaluated (deviation 12).
   - RD L118, L213, L230 and MH #5: the "never improved" result was close to forced (R1 identity; identity never counts).
   - RD L230: give the CONT exceptions.
7. **Scope.**
   - AB L11-12: Track F refits only the occupation side, with H3's income frozen; Track R ran on R0 and R1 only.
   - RD §7, AB L36-37 and MH #9: the income refit used about half of H3's training size; it is a recipe test, not a test of the
     frozen package.
8. **Screen rates.** RD L35, AB L40 and MH #8: report them as properties of law L1 (0.19 at rho 0, 0.31 at rho 0.76).
9. **Reader-relative wording.** SH L14-16; RD L38, L129, L142-143, L155, L216-218 and L231; AB L31-33 and L42.
10. **"Validated" and "verified".** RD L14, L203, L205 and L233; SH L12; AB L44.
    - Say "replicated and independently replayed"; the directions were not refitted.
    - RD L46, L157 and L205: call the partitions overlapping.
11. **Disclosure completeness.** The RD failed-attempts list and the MH required-disclosures list need:
    - deviations 6, 12, 16, 18 and 19;
    - adjudications A3, A4 and A6;
    - the verifier's genuine FAILs (F3 V2, C2.1) and the P4b window;
    - the failed verifier unit and the never-launched one;
    - the heading-time errors for A2, A5 and the verifier's "20:29Z".
12. **Plain answers.** Rewrite the §3 lead answer (A33) and add one-line answers to §6 and §7 (A54).
13. **START_HERE closeout list.** Some listed files are missing or uncommitted, the ledger lags the accounting, and the list should
    add the verifier REPORT addendum, the V_TRIGGER and V_QA_INFLUENCE records, and A1-A6.
14. **Minor.**
    - RD L142-143: "never positive" holds on the R stage only (D stage +0.0063).
    - RD L150: stable vs CONT at s ≥ 0.3, vs W only at s = 1.0.
    - RD L61-62: add "B80, MS−OOF".
    - RD L185 and MH #7: explain the record's `qualified_protected_points` field.
    - RD L227: both routings on D only.
    - RD L228: "systematic" holds on MS; R0 CHECK is the exception.
    - RD L238: the GUARD fraction ("25-40%").
    - RD L234: the P70 package is an adaptive fit.

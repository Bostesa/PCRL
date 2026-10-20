# Challenger, pre-registration review (saved verbatim with attribution)

Author: the challenger of campaign `pcrl_finite_routing_comparison_v1`. Returned to the coordinator at about 23:24Z on 2026-10-20 (its
own times: written 23:08-23:22Z, CPU charge recorded at 23:22:09Z). Saved verbatim; nothing below the line was edited (the harness's
two-space indentation of each line was removed). The coordinator's adjudication of every finding is in
`roles/coordinator/ADJUDICATION_PREREG.md`. Note: the challenger read `frc/devreaders.py` before the coordinator's constants patch of
about 23:17Z (finding 4 was already fixed when the report arrived; see the adjudication).

---

CHALLENGER: pre-registration review of campaign `pcrl_finite_routing_comparison_v1`. Written 23:08–23:22Z (`date -u`). CPU charge of 20 s recorded at 23:22:09Z. Handed back after that.

**What I read.** AGENT_CONTEXT.md, PROTOCOL.md (the 23:06Z draft), cards/FA_PRC_v1.md, nrm/fa/prc.py, the frc drafts (frc/fa/route.py, frc/readerbudget.py, and the header and constants of frc/devreaders.py), ffv/readers.py and ffv/features.py (CatiDesign), and the NRM1 records listed in the assignment. I read no data, no labels and nothing under `.private/` (the ledger was grepped only to confirm the charge). I wrote no files and spawned no helpers.

**Severity count:** 4 BLOCKER, 9 SHOULD FIX, 8 NOTE.

---

## Findings

### 1. [BLOCKER] QUAL as written is already met by A at (16, 64), and its slack (S1) does not match the budget where D2 is assessed (S2)

**A at (16, 64) qualifies under QUAL as written, from published records.**
- PRC_TWIN's worst value/cap is 0.100 (v0) and 0.510 (v1).
- The largest d/cap in Q2 is 0.40 (occupation NLL, 0.004/0.010); the others are 0.2–0.3.
- So every clause has value + d ≤ 0.91 × cap in both versions. The v1 men's guard gives 0.00255 + 0.0015 = 0.00405 ≤ 0.005.

**Consequences.**
- (a) The protected stage runs whenever MODEL_FIT does not select B1 at 64 cells. The Q-PE outcome cannot stop it.
- (b) Proceeding with A swaps the failed NRM1 capacity gate for a laxer rule after it failed. FA_CAPACITY_DECISION.md refused exactly this: "The registered gate is not widened after its failure."
- (c) D2 and the HQ/SHUF controls sit at S2, but Q2 only checks S1 slack. With S2 slack the v1 men's guard reads 0.00255 + 0.0030 = 0.00555 > 0.005. So D2's G1 conjunct, built on A at 64 cells, is either nearly certain to fail or a near-limit coin flip.

**Verdict:** not acceptable under the brief ("If that can be repaired while preserving useful task margin, proceed"). An unrepaired release with only S1 slack is neither repaired nor carrying useful margin for the comparison that is registered.

**Proposed text, section 5, replace Q2:**
> Q2 (slack at the assessment budget S2): value + d ≤ cap for every condition. d is the S2 budget of FA_PRC_v1: occupation NLL 0.008, income NLL 0.004, occupation accuracy loss 0.006, income accuracy loss 0.004, men's and women's occupation Brier guards 0.0030. For the summed-Brier clauses, d is 0.0030 (occupation) and 0.0020 (income).
>
> Known before registration (published NRM1 utility): A at (16, 64) meets Q1 and fails this Q2 (v1 men's guard 0.00255 + 0.0030 = 0.00555 > 0.005). It meets the S1 form (0.00405), which is not this rule.

**Proposed text, section 13:**
> QUAL(A, (16, 64)) is known from published records; it is not a prediction.

**Proposed text, section 10.1:**
> QUALIFIED_WITHOUT_PROBABILITY_REPAIR can occur only at (24, 128) or (32, 256), where A's MODEL_SELECT utility is unknown.

This rule is strict but reachable. The A decoders' Brier estimation term is roughly (1 − Σp²)/n ≈ 0.7/265 ≈ 0.0026 per cell at 64 cells, so a genuine B1 repair could recover the 0.00055 needed.

**Fallback if S1 is kept.** Register b* = S2 if Q2(S2) holds in both versions, else S1. HQ_b*, SHUF_b* and D2 then sit at b*. State in section 0 that A at 64 cells is known to qualify at S1, so proceeding is a protocol decision, not a test outcome.

### 2. [BLOCKER] The MODEL_FIT check P3 is biased in PRC's favour

**The bias.**
- P3 reuses `nrm.fa.prc.stage_check`, which reads cells "off the released MODEL_FIT vectors themselves". Those vectors come from the deployed full-fit teachers applied to their own training rows.
- For PRC, that routing uses an in-sample π fitted on these same people's SEX labels. Section 0 found only 56–76% agreement between deployed and out-of-fold routing even for the task teachers.
- Only PRC routes on a person-level π. TWIN and ORD use none; HQ uses out-of-fold means within twin cells; SHUF uses a donor's out-of-fold value.
- So the bias is differential: in-sample, PRC sends likely women into men-heavy cells more accurately than it can on new people.
- Card falsifier 1 requires this check before any PRC-vs-ORD credit, so the bias reaches a decision.

**Proposed text, P3 row of the section 3 table and section 10:**
> P3 reads the out-of-fold-routed fit cells (the partition each saved book reproduces on the cached out-of-fold statistics), never the deployed MODEL_FIT release. It is labelled "routing optimized on these people (optimistic)" and is never a credit input.
>
> The MODEL_FIT reader check used by any decision is the nested held-out LOOKUP of FRC_PATHS: held-out people routed by the outer-training deployed rule.

### 3. [BLOCKER] The card's falsifiers and reader-targeting rule are missing from D2, and there is no governing card yet (cards/ is empty)

FA_PRC_v1 section 6 has:
- F1: a PRC-vs-ORD advantage counts only if it also holds for smooth readers and for the MODEL_FIT LOOKUP.
- F4: before any MODEL_SELECT read of a protected release, PRC_S2's nested held-out pair LOOKUP AUC must sit below the twin's by more than 2 SE.
- An advantage seen only by LOOKUP-type readers is labelled reader-targeted.

D2 as drafted asks only for AGG_POSITIVE and DOMINATES. The PRC objective is the π-weighted plug-in LOOKUP statistic itself, so this omission matters.

**Proposed text, append to 10.2:**
> D2 additionally requires:
> (i) gate F4 at the assessed configuration — nested held-out pair LOOKUP AUC below TWIN by more than 2 household-clustered SE in both versions — checked before any MODEL_SELECT read of a protected release. Failure closes the protected stage (FA_PRC_v1 section 10 stop).
> (ii) a point benefit > 0 in both versions against each of TWIN, SHUF, HQ and the bracketing ORD point, for the best non-table pair reader (the registered bank without LOOKUP: the `nolookup` winner of nrm.readerextra) and for the nested MODEL_FIT LOOKUP. Otherwise the comparison is labelled READER_TARGETED and no component is credited.

**Proposed addition:**
> The governing card FRC_PRC_v1 (FA_PRC_v1 plus the construction rule, the SHUF change of finding 6 and the brackets of finding 5) carries section 6 falsifiers 1–4 and is committed with the protocol.

### 4. [BLOCKER] frc/devreaders.py still carries NRM1's grid constants

**What it has now (lines 14, 17, 27, 73–89):**
- `REP_TAG = 'NRM1_devgrid_<R>'`
- `SEED_BASE_OFFSET = 600`
- `BOOT_SEED = 2026102000`
- `GRID_VERSION` / `SCHEMA` = `'NRM1-devgrid-v1'`
- reader cache key prefix `'NRM1-devgrid-v1'`

**What the protocol registers:** `FRC1_devgrid_R<r>`, offset +700, seed 2026102100.

Run as is, the "fresh" grid would be NRM1's exact splits with its reader seeds, which section 9 forbids (offset 600).

**Fix:**
- Set the constants to the registered values before the registration commit.
- Add a selftest asserting that the tags and seeds differ from offsets 300, 400, 500 and 600.
- Record the household overlap with every `NRM1_devgrid_R*` in READER_SPLITS.json.

### 5. [SHOULD FIX] HQ_S2 and SHUF_S2 are not spend-matched to PRC_S2, so "beyond HQ/SHUF" can be credited against a control that spent less

- The S2 rule (largest λ whose nested estimate + 2 SE ≤ budget) does not make a control's spend reach PRC_S2's.
- At λ = 8 the control may still sit below PRC's spend, or the budget may bind early.
- Card falsifiers 2 and 3 are stated "at equal or lower measured spend", but the protocol has no bracket for these two controls.
- The card's ORD contingent (+2) is also missing from the protocol.

**Proposed text, HQ and SHUF rows of section 8:**
> HQ and SHUF are also saved at the bracketing λ: the smallest grid λ whose nested spend over the twin on occupation NLL and on the binding guard is ≥ PRC_S2's. This uses MODEL_FIT only, from the chains the nested estimator already computes, and at most 4 configurations from the reserve.
>
> "Beyond HQ/SHUF" requires a resolved benefit against the bracketing point (or against S2 when S2 already spends ≥ PRC_S2 on MODEL_SELECT). If no grid λ reaches PRC_S2's spend, the label is CONTROL_DID_NOT_REACH_SPEND and nothing beyond that control is credited.
>
> ORD: one bracketing entry (+2), chosen by measured MODEL_SELECT spend only, when nrm.ctl.frontier returns EXTEND_CONTROL or AHEAD_VS_SPARSE_PATH (the FA_PRC_v1 section 9 contingent).

### 6. [SHOULD FIX] SHUF's guard runs on another person's occupation posteriors, so SHUF spends budget on random re-routing that PRC never does

- `_prepare` and `route` replace π, qO0 and qO1 by the donor copy.
- The guard term t·κ·[(1−π)/p0·|qO0 − cO_l|² + π/p1·|qO1 − cO_l|²] then pulls each person toward the donor's occupation posterior. That is large, task-irrelevant re-routing.
- The invented-data selftest, where SHUF spent more than PRC, is consistent with this.
- So "PRC beyond SHUF" would confound targeting with a wasted guard.

**Proposed text, SHUF row of section 8:**
> π is replaced by the keyed-hash donor copy; qO0 and qO1 are replaced by the person's own qO (a SEX-blind task-fidelity guard). Objective, tables and acceptance all use this copy.

It costs no configurations. It changes the card's SHUF before any SHUF fit has ever run; record it as a card change.

### 7. [SHOULD FIX] Reader-budget asymmetry is reported but not controlled

- About 4,000 reader-FIT people face up to 1,024 joint values (16 × 64) or 3,072 (24 × 128). That is a few people per value, with many SCORE values unseen (they get the base rate).
- A PRC that empties or splits joint cells changes the reader's data per value independently of information.
- frc.readerbudget.contrast already has fixed thresholds (ratio 2.0, unseen 0.10) but is "diagnostic only".

**Proposed text, section 9:**
> For every D2 and D3 comparison, the pair-view budget contrast (frc.readerbudget.contrast, thresholds as committed) is recorded. A READER_BUDGET_ASYMMETRIC comparison is credited only if the benefit also holds (point estimate, both versions) for the non-table winner and for the nested MODEL_FIT LOOKUP.
>
> The FIT-fraction curve (0.25 / 0.5 / 1.0, already in the grid code) is reported for the pair view of TWIN, PRC_S2, HQ, SHUF and the ORD bracket.

### 8. [SHOULD FIX] No reader combines the two dense single-view tables

- The pair LOOKUP keys the exact joint value.
- The dense income LOOKUP (16 values) and occupation LOOKUP (64 values) enter only as ignore-one alternatives.
- CATI (pair) is income indicators × occupation-vector features, a smooth reader.
- No recipe combines the two marginal tables additively. That is exactly what PRC's local terms and its rP shrinkage target (an additive-logit fit) model.

**Proposed text, section 9:**
> Supplementary recipe LOOKUP_ADD, the same for every release and outside the locked bank (disclosed): logit = shrunk income-cell logit + shrunk occupation-cell logit − base-rate logit, pseudo-count 2, fitted on FIT and oriented on VALIDATION. Its pair AUC is reported beside the winner.

Cost: seconds of CPU, no configurations.

### 9. [SHOULD FIX] D2 is undefined when PRC_S2 is NOT_FEASIBLE or an alias

**Proposed text, 10.2:**
> - If PRC_S2 is NOT_FEASIBLE: D2 = NOT_ASSESSED_NO_FEASIBLE_STRENGTH.
> - If PRC_S2 is an alias of PRC_S1: it is assessed as that configuration, with the finding 5 brackets.
> - If every nonzero selection is an exact alias of TWIN (no transmitted vector changes): D2 = NOT_ASSESSED_EXACT_ALIAS, and the family stops.

### 10. [SHOULD FIX] The construction choice ignores the binding constraints, and the no-fallback rule is unstated

- Section 6.2 chooses on nested occupation and total NLL only.
- T1 reads no SEX, so the men's guard (binding for A) and accuracy are invisible to the choice.

**Proposed text, section 6.3:**
> The construction chosen at a capacity is final for that capacity. If it fails QUAL, the next capacity is tried even when the other construction would qualify. Whether the other construction would have qualified is recorded, never used.

### 11. [SHOULD FIX] T6 is an open design choice made after outcomes are seen (a forking path)

Preferred: drop T6.

Otherwise, proposed text for section 11:
> A T6 repair is labelled POST_HOC. Its card is committed before any MODEL_SELECT read of its outputs. It can never yield D1 = REPAIRED_TASK_ONLY_RELEASE, only POST_HOC_REPAIR_QUALIFIED.

### 12. [SHOULD FIX] Reproducing the GATE records is necessary but not sufficient

It validates CONT, A, the teachers, the folds and the books. It does not touch the B0 and B1 code paths.

**Add to T0–T2:**
- (a) The verifier recomputes the B1 tables from the cached out-of-fold predictions and the saved books with separate code, and they match bitwise.
- (b) Invented-data selftest: perturbing the outer-test labels leaves every construction's tables and the held-out partition bitwise unchanged; perturbing all labels at fixed teachers leaves B0 and B1 unchanged.
- (c) At a converged synthetic Lloyd fixed point, B1 equals B0.
- (d) Count, per role, the people deployed-routed into cells that are empty in the out-of-fold partition (both A and B1 emit the prototype there).
- (e) The per-fold B1 counts equal A's counts.

Already verified: frc/fa/route.py `nested_twin` builds all tables from `fit_twin(Z[tr], lab[tr])`, i.e. the outer-training part only, and asserts the value-partition identity per fold and view.

### 13. [SHOULD FIX] The ORD ladders in the code differ from protocol section 8

`frc/fa/route.py` `ORD_LADDERS` prepends (16, 64) at C128, and (24, 128) and (16, 64) at C256. Section 8 lists only the card's (12, 48) … (2, 8). Register the per-capacity ladders, ordered finest to coarsest (select_path depends on the order).

### 14. [NOTE] The partition is fixed: yes, by construction and asserted

- `route` depends only on the book and the runtime statistics; A, B0 and B1 change only the table.
- In the protected stage the partition is also independent of the construction, because Lloyd's task term is the cross-entropy to the codewords, not to the emitted tables. Only select_path's spends differ between constructions.
- Caveat: the tables are estimated on the out-of-fold partition but applied to the deployed one (56–76% agreement in-sample). This affects both constructions alike, so it is shared, not differential.

### 15. [NOTE] No hidden label exposure and no unfair advantage for B1

- B1 tables are functions of fold-model predictions, MODEL_FIT only. Excluding full-fit predictions of training rows is correct: those would be in-sample overconfident.
- If anything B1 is handicapped: the fold models (16,000 people; 12,800 inside nested) are flatter than the deployed teacher.
- Shared by every construction: the non-nested full-pool T*. The held-out outer folds' labels chose T*; this is common to CONT, A, B0 and B1. State it in 4.1.
- Nested cells hold about 250 people versus about 312 when deployed (64 cells), so A's estimation term is about 1.25× larger in nested. The MODEL_SELECT conjunct of PE_ERROR_SUPPORTED guards against over-reading this.

### 16. [NOTE] Decisions versus probabilities

- B1 − A differences in NLL are pure probability effects.
- Accuracy differences arise only in cells whose argmax flips. Report per capacity and role: the cells and people whose A and B1 decisions differ, and the accuracy and Brier differences restricted to those people.
- The B1 TWIN's recovery will not equal PRC_TWIN's 0.6509: smooth readers see different coordinates on the same partition. Prediction 6's "0.015–0.018 above NEURAL_RAW and H_LJ" was measured on the A twin. Say so.

### 17. [NOTE] Uncertainty

- The household-clustered paired SE is adequate to first order: with proper probability vectors, the within-cell first-order correlation of per-person differences cancels (Σ_y δ(y) = 0).
- It ignores fit-level dependence: outer-training sets overlap by 75%, T* is non-nested, and routing shifts between out-of-fold and deployed.
- Guard: report the five per-fold means (already in `paired_fold_mean_B1_minus_A_occ_nll`). Require for PE_ERROR_SUPPORTED a negative mean in at least 4 of 5 folds per version, or use max(household SE, sd(fold means)/√5).
- The two versions are procedure repeats, not replications.

### 18. [NOTE] Power

- NRM1's se_agg for candidate-versus-control comparisons was 0.007–0.010, so AGG_POSITIVE needs a benefit of about 0.014–0.020.
- Prediction 6 (0.005–0.020 below TWIN) mostly falls below that.
- Register the minimum detectable effect (2·se_agg ≈ 0.015) and "not separated" as the default reading beside D2.

### 19. [NOTE] Budget arithmetic

- Make E4 conditional on PRC_S2 reaching at least AGG_POSITIVE_POINT_ONLY against TWIN.
- With findings 5 and 13: 40 planned + up to 4 (HQ/SHUF brackets) + up to 2 (ORD bracket) = 46 ≤ 48, or 42 when E4 is not triggered.
- CPU stays small: the brackets come from chains already computed at every grid λ; the extra cost is the configuration saves (about 5 min each) and their reader cells.

### 20. [NOTE] HQ is sparse at C128 and C256

- HQ's surrogate π is the mean out-of-fold π within twin joint cells: 3,072 cells at 24 × 128, about 6.5 people each, pseudo-count 1.
- It is h(q), but noisy and sample-specific. Report HQ's fitted-versus-held-out divergence (already in nested) and read it on that support only.
- By design, HQ's tables and acceptance use the true out-of-fold π; only the deployed rule is restricted. That is the correct form of the published objective.

### 21. [NOTE] MODEL_SELECT SEX enters QUAL through the guards

This is an adaptive use of the people later read for recovery. The selection is on utility only and is common to PRC and its controls, so it does not bias comparisons within a capacity. Disclose it.

---

## FAILURE_MAP.md (draft, ready to save)

(Saved, with the coordinator's edits recorded in its header, as `results/pcrl_finite_routing_comparison_v1/FAILURE_MAP.md`.)

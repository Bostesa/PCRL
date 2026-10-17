# Development review: harness, baselines, controls and promotion

**Role.** Baseline and statistical reviewer of the PCRL autonomous method-development campaign. The campaign uses public-use Florida ACS 2019 person records. The service releases each person's income (2-class) and occupation (5-class) probability vectors and argmax decisions, computed from AGEP, WKHP, COW, SCHL, MAR and POBP. Readers infer the recorded SEX column from each output and from the linked pair.

**Scope.** Written 2026-10-17, 17:30–17:50Z. It covers ledger entries 1–42 and `PROMOTION.json` of 17:23:54Z.

**Read.**
- Campaign records: `PROTOCOL.md`, `FAILURE_MAP.md`, `MECHANISM_CARDS.md`, `EXPERIMENT_LEDGER.csv`, the coordinator's brief and the B/C card revisions.
- Code: `amd/{evaluate,devrun,baselines,common,ledger,fam_a,fam_b,promote,final_bank}.py`, plus `scm_bench/selection.clauses`, `scm_bench/readers` (views, LOOKUP) and `acs_replication/data.clauses`.
- Aggregate development evaluations (`EVAL_MS.json`: clauses, guards, per-reader proxies; no person-level arrays) for A2, B2, B2CTL_UNCO, B2CTL_TASKONLY, B3 and B3CTL_UNCO.
- The previous campaign's FROZEN DESIGN and `scm_um/report.py`.

**Not done.** I read no label array and no ATTACK or DEV_REPORT label. I fitted nothing, committed nothing and edited no code. The only computations were:
- re-applying §6 to the aggregate JSON, using a script in my scratchpad whose logic is copied from `amd.evaluate` / `amd.promote` minus the control exclusion;
- closed-form t quantiles.

---

## 0. Adjudication (coordinator's question): what does the registered §6 rule select?

**Decision: `B3CTL_UNCOUPLED_MIX50_C6A`, labelled `EDGE_SELECTED`.** The decision rests on the registered text only.

**Why the control exclusion is not part of the rule.**
- §6 speaks of "configurations" and "among promotable configurations" without exception.
- §2 defines the configuration count to include "newly fitted controls and ablations".
- Nothing in §5, §6 or the brief bars a control-designated configuration from promotion.
- The CTL/CTRL/TASKREFIT exclusion in `amd/promote.py` was written at 17:23:49Z. That was after B2's controls had been scored (entries 35 and 37 at 17:19–17:20Z; 38 and 39 at 17:22Z).
- Applied now, the exclusion would change the finalist. A rule introduced after outcomes that changes the selection is a protocol deviation, not an implementation detail.

**Verification.** My recomputation covers both seeds, on MODEL_SELECT `max` proxies. Reference pair proxies: NEURAL_RAW 0.6238 / 0.6269, H_LJ 0.6224 / 0.6233.

| Configuration | 12 clauses + 2 guards, both seeds | Recipient screens, both seeds | Pair proxy (s0 / s1) | Values above 0.8 × cap | Mean pair proxy |
|---|---|---|---|---|---|
| B3CTL_UNCOUPLED_MIX50_C6A | pass | pass (income 0.5277 / 0.5286; occupation = H_LJ's) | 0.6000 / 0.6001 | s0 income Brier vs Ucal 0.87; s0 men's guard 0.88; s1 income Brier vs U0 0.87 and vs Ucal 0.88 | **0.60002** |
| B2_CSR_JT_C6A | pass | pass | 0.6131 / 0.6132 | s0 men's guard 0.88 | 0.61314 |
| B2CTL_UNCOUPLED_C6A | pass | pass | 0.6153 / 0.6154 | s0 men's guard 0.88 | 0.61533 |

- No promotable configuration meets the 0.8 × cap margin, so §6.3 goes to the lowest mean pair proxy: B3CTL_UNCO.
- Everything else is not promotable:
  - A2 fails the recipient screens and the seed-1 pair condition.
  - C1 fails the seed-1 men's guard (0.0064).
  - The one-seed configurations (A1, A3, B1, B3, C2, CDIAG) each fail a seed-0 condition.
  - B2CTL_TASKONLY_TT and C1_CTRL_TEMP fail the screens and the pair condition.

**Is there a registered reason to prefer B2? No.**
- **Robustness.** §6.3's robustness preference is binary: either every value is ≤ 0.8 × cap or not. Both configurations are EDGE. The rule does not grade how far a configuration is over 0.8, so B2's larger income margins (worst normalized 0.55 / 0.31) are not a registered criterion.
- **The brief's "reproducible feasible improvements over fragile single-seed gains".** B3CTL's gain holds on both seeds (0.0224 / 0.0232 below H_LJ) and is point-feasible on both. Its fragility is a utility margin, which the registered rule handles only through the EDGE label.
- **§2 revision cap.** B3CTL applies the card's control construction to revision 2, as B2CTL did for B2: the same m, α, bins and occupation output as B3, with φ = none. No design value was chosen after B3's result. Only the decision to run it was adaptive, which §3 and §5 allow as revision feedback. It was the 10th of 10 family-B configurations.
- **Ties to the simpler mechanism.** Not engaged (the gap is 0.013). B3CTL is the simpler mechanism anyway: 5 coefficients and no coupling.
- **§5 "never selected because its own training critic became weak".** Not applicable; this construction has no critic.
- **Reader agreement.** The per-reader pair proxies agree:

  | Seed | OUT_HGB | OUT_LR | LOG_MLP |
  |---|---|---|---|
  | s0 | 0.6000 | 0.5980 | 0.5958 |
  | s1 | 0.5946 | 0.6001 | 0.5974 |

  The selection does not rest on one reader class. It is unlike the B2-versus-B2CTL gap (F2).

**Must be disclosed, whichever configuration is frozen.**
1. **The superseded promotion record.** `PROMOTION.json` of 17:23:54Z named B2. It came from an incomplete trial set (before entries 40–42) and applied the unregistered exclusion. Keep it in the record as superseded, with the reason. Re-run promotion on the registered text after exploration closes, and freeze only that record.
2. **If B2 is promoted anyway.** That is a declared protocol deviation: introduced after the controls' outcomes, it changes the finalist. The registered-text finalist and its numbers must be printed beside it.
3. **Origin of B3CTL.**
   - It was run as the coarse-only twin of B3 after B3 failed the seed-0 MODEL_SELECT income Brier clause (adaptive, logged).
   - α = 0.5 was the largest α inside the caps on a CALIBRATION screen that applied a MODEL_SELECT-derived offset, i.e. edge-seeking.
   - Its MODEL_SELECT income Brier is at 0.87–0.88 of the cap on both seeds.
   - A DEV_REPORT point failure on income Brier is plausible. FAILURE_MAP puts role-drift SEs at 0.0016–0.0020 nats, and SCM's spend moved from 0.0082 to 0.0113 on DEV_REPORT.
   - Register a prediction for E before the REPORTING_LOCK.
4. **What the configuration is.**
   - Income: six decision-aligned levels of 0.5·LOCAL_TREE + 0.5·JOINT_TREE E20 income log-odds. Thresholds are label-free MODEL_FIT quantiles; the levels are histogram-calibrated on CALIBRATION income labels.
   - Occupation: the stored JOINT_TREE E20 output, bitwise H_LJ's.
   - It does not use the family-B card's coupling.
   - Every privacy-trained component is inherited from the replication's Stage B.
5. **The EDGE label on both seeds.** On both seeds it comes from the configuration's own income Brier; on seed 0 also from the men's guard inherited from H_LJ's occupation output.

---

## 1. Findings, ranked (each with its fix)

**F1. The promotion code and record (critical).** `amd/promote.py` excludes control-designated ids by name. That rule is unregistered and post-hoc (§0). `PROMOTION.json` was also produced while family B was still running.
- **Fix:**
  - Remove the exclusion.
  - Keep a `designation` field: candidate / control / diagnostic.
  - Re-run promotion once all families have stopped.
  - Keep the 17:23:54Z record as superseded.

**F2. Coupling explains nothing at the pair (high).**
- B2 and its uncoupled twin B2CTL use the same statistic c. Both pairs are one-to-one functions of (occupation output, c), provided the six calibrated levels are distinct. B2CTL has 6 distinct income values; for B2, check that h's six bin coefficients are distinct in RELEASE.json. So the population-optimal pair recovery is identical for the two.
- The 0.0022 proxy gap comes only from OUT_LR. OUT_HGB ranks them the other way (B2 higher by 0.0036 / 0.0059), and LOG_MLP splits (−0.0008 / +0.0006). B2CTL also has the better income utility (NLL excess vs U0 0.0013 / 0.0006, against 0.0033 / 0.0026).
- The card revision's "information bound log 6" is vacuous for any m ≥ 2, because I(SEX; ·) ≤ H(SEX) ≤ log 2 < log m. It holds only for m = 1.
- **Fix:**
  - No claim, figure or wording attributes pair protection to coupling or to that bound.
  - For a coupled finalist, the uncoupled twin's pair difference is a secondary representation check, expected to be 0. M's simple-control leg becomes a utility leg (see the draft).

**F3. The finalist has no matched task-only control yet (high, needed before the freeze).**
- `B2CTL_TASKONLY_TT_C6A` is coupled (9 coefficients) and binned on T_TREE log-odds. It is not matched to B3CTL, which is uncoupled (5 coefficients) and binned on a mixture statistic.
- **Fix:** before FINALIST_LOCK, fit TO = six decision-aligned, CALIBRATION-histogram-calibrated levels of the TASK_REFIT E20 income log-odds, with label-free MODEL_FIT quantile thresholds, paired with the TASK_REFIT E20 occupation output. That is 2 configurations; the count goes from 19 to 21 of 36.
  - TASK_REFIT is the replication's privacy-off arm of the same E20 leaf-correction loop. It therefore matches the inherited corrections' task-training exposure and optimization effort.
  - Using T_TREE bases (for continuity with B2's control) is acceptable only if M is then worded "relative to the same coarsening of the teacher's outputs".
- The other controls the finalist needs already exist:
  - the strongest simple control on the card (same occupation, uncoupled LOCAL income) is H_LJ itself;
  - the local-statistic twin is B2CTL_UNCO, the J control.

**F4. DEV_REPORT is partly a replay for any B/C finalist (high).**
- The occupation output is H_LJ's, bitwise. Its DEV utility, guards and recovery (24/24; men's occupation Brier 0.0032 / 0.0040; recovery 0.560 / 0.583) were published before this campaign, and the component was chosen knowing them.
- **Fix:**
  - Flag occupation-side DEV endpoints `REPLAY_OF_PUBLISHED`.
  - Flag the occupation-recipient endpoint against H_LJ `IDENTICAL_RELEASE` (point = SE = 0).
  - Qualify claim B against NEURAL_RAW with "occupation component selected with knowledge of its published DEV recovery".

**F5. Edge-selected utility (medium).** See §0, item 3. The registered rule's only protection is the EDGE label.
- **Fix:** no rule change. Print EDGE_SELECTED with every claim, report income Brier per seed beside E, and register the E bet before the lock.

**F6. Winner's curse on a fixed split (medium).**
- All 42 evaluations reuse one household split (`SCM1proxysplit`) and fixed reader seeds. The finalist is the minimum, while the references were measured once. So the development margins are optimistic.
- The two seeds share MODEL_SELECT people and the same split, so "both seeds" is not replication.
- Scale: the information-identical B2/B2CTL pairs differ by up to 0.006 between readers. B3CTL's 0.022 margin is well beyond that.
- **Fix:** never quote MODEL_SELECT margins as effects. The fresh ATTACK_FIT readers on DEV_REPORT are the test. No rule change.

**F7. Family A's controller uses MODEL_SELECT constants (medium; currently moot).**
- **The constants.** All A budgets use the teacher's published MODEL_SELECT headroom (`STAGE_A_HEADROOM.json`). A2 and A3 also use the teacher's MODEL_SELECT per-SEX guard values from `BASELINES_MS`.
- **Effect.** These are a few scalars per seed, computed from MODEL_SELECT labels and SEX, and they set the rollback thresholds. A's MODEL_SELECT U/G passes are therefore tuned rather than held out.
- **What the rules say.** This is allowed as revision feedback, since the scalars are not gradient labels. But the A ledger rows record MODEL_SELECT only as "evaluation".
- **Matched control.** `TASKREFIT80` (defined, never run) carries A1's budgets, not A2's. Its rollback can also return it to θ = 0, because the teacher is in-sample on the holdout.
- **Fix:**
  - Annotate the A rows.
  - Any A control must copy the finalist's budget dictionary verbatim and log its rollbacks.
  - No action is needed unless A is promoted.

**F8. The proxy definition is only implicit (low).** `promote.py` and the cards use `max` consistently: on the pair, the maximum over OUT_LR, OUT_HGB, LOG_MLP and LOOKUP; on each single output, the maximum over the first three. `source_proxy` would flip A1 on seed 0 (0.6193 against H_LJ's 0.6224) and change A2 on seed 1.
- **Fix:** write the definition into the promotion record.

**F9. LOOKUP is degenerate on continuous pairs (low).**
- It reads 0.5033 / 0.5030 for every continuous release checked, because only repeated input profiles contribute. This is symmetric across continuous candidates and references, so it gives no advantage.
- In the final bank, a single-output LOOKUP on B3CTL's 6-level income output is nearly Bayes-optimal, while the references' continuous incomes get no such reader. That is conservative for the finalist.
- **Fix:** no change; disclose in the claims.

**F10. Ledger hygiene (low).**
- The `decision` column is empty in all 42 rows, although §5 requires decisions.
- Candidate rows hash `devrun.py` plus the family module, but not `evaluate.py` or `common.py`. No drift is suspected:
  - `evaluate.py` was last modified at 16:58Z, before the first row;
  - the later edit to `common.py` (17:17Z) only merges `SOURCE_ADMISSION_2.json` into `admission()`.
- The A2 seed-0 and seed-1 hashes differ because the A3 options were added in between; A2's code path did not change.
- **Fix:** append decision rows (family stops, promotion, this adjudication) and record the SHA-256 of `evaluate.py` and `common.py` once.

---

## 2. Checks that pass

**(a) The development harness measures exactly the registered clauses. No change.**
- `amd.evaluate.measure` calls `scm_bench.selection.clauses`. That is the replication's `acs_replication.data.clauses` in substance: 2 tasks × {U0, Ucal} × {accuracy loss ≤ 0.010, NLL excess ≤ 0.010, summed Brier excess ≤ 0.005}, inclusive. It uses the same `task_vectors`: NLL = −log max(p_y, 1e-12), summed Brier, accuracy from argmax.
- The guard is the mean, within recorded SEX g, of the occupation summed-Brier difference against U0, checked ≤ 0.005 inclusive, with sex0 = men.
- The checks run per seed, against the frozen U0 and Ucal MODEL_SELECT outputs. Row order is asserted.

**(b) The reference bar is measured with the same harness. No change.**
- `amd.baselines` calls the same `measure` with the same MODEL_SELECT data, the same household halves, the same reader seeds (`MODEL_SEED_BASE + seed`), the same views and the same float64 casting.
- `promote.py` reads the references' `max` proxies from that output.

**(d) The other families' controls are fair as defined.**
- **Family C.** The SEX-blind control is T_TREE, and the identical recipe reproduced it bitwise. `C1_CTRL_TEMP` is matched on label-free KL on MODEL_FIT features. `CDIAG_TTINC_JOCC` is the right component control for C2.
- **Family B twins.** The coupled and uncoupled twins have the same input access, the same CALIBRATION exposure and the same solver. Their capacity difference (9 versus 5 coefficients) is inherent to the comparison.
- **References.** NEURAL_RAW and H_LJ are not recalibrated on CALIBRATION, whereas family-B incomes are. This is a utility advantage for the candidates, not a recovery bias. Disclose it.

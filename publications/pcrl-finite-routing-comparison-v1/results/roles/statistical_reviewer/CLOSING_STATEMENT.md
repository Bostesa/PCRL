# Closing statement of the statistical reviewer (campaign `pcrl_finite_routing_comparison_v1`)

Author: the statistical reviewer, who wrote the frozen rules (`frc/rules.py`, `frc/test_rules.py`, `RULES.md` / `RULES.json`) and
the post hoc grid-validity repair ER1 (2026-10-21 00:28:29Z-00:30:09Z). Review started 2026-10-21T01:03:14Z (`date -u`). Inputs:
committed files and saved JSON records only (`RESEARCH_DECISION.md` draft, `TABLES.md`, `claims/MAIN_ER1`, `claims/R1_ER1`,
`claims/MAIN`, `claims/R1`, `claims/STABILITY/*`, `decisions/QUAL_DECISION.json`, `fa/PROBE_STRENGTH_V0.json`,
`PROTOCOL.md` section on ablations, `COORDINATOR_LOG.md`, `roles/verifier/REPORT_0058.md`, `frc/rules.py`, `frc/fa/route.py`).
No label was read and no real-data computation was run.

## Required headline

The coordinator must use this wording (numbers may only be re-rounded, not changed):

> Development evidence on adaptively used MODEL_SELECT records, for the declared readers only. Under the post hoc grid-validity
> repair ER1, the outcome class is `ENGINEERING_REPAIR`: a repaired task-only finite release, and no protected-routing benefit
> beyond matched controls. Without ER1, D2-D4 are `NOT_COMPUTED_GRID_INVALID` and the outcome class is `INCOMPLETE`. On identical
> cell partitions, replacing family A's label-refit cell tables with out-of-fold teacher means lowered nested held-out occupation
> NLL by 0.0039-0.0049 nats at 16 x 64 cells and 0.0085-0.0090 nats at 24 x 128 cells (each resolved). That is about 36% and
> 62-68% of the label-refit twin's occupation-NLL spend over the continuous task teachers (point ratios). The original 0.004
> capacity gate still fails at both capacities. With the repaired tables, the 16 x 64 twin qualifies under the original limits
> with S2 slack (D1 `REPAIRED_TASK_ONLY_RELEASE`). The propensity-aware routing was then fitted. The registered strength path
> produced one protected partition per version (strength 8, the top of the grid). That partition re-routes 13-14% of MODEL_SELECT
> people and lowers registered-grid linked-pair AUC from 0.6555 (twin) to 0.6485 (benefit +0.0070, se 0.0040, not resolved). A
> grouping control that sees only task scores reaches 0.6489, and untargeted re-routing at PRC_S2's spend reaches 0.6516 (revision
> R1 path). On every control path the reduction is not separated from task spending (D2 `NO_REDUCTION_BEYOND_ANY_CONTROL`). The
> routing is not superior to the admitted references (D3 `NOT_SUPERIOR`), and on the registered grid it stays 0.018-0.023 pair AUC
> above both target references (D4 `NOT_MET`). No comparison is confirmed on the 32 robustness reader cells (24 new, 26-29
> distinct). No component, joint-design or novelty claim is made.

## Label check

Records: `claims/MAIN_ER1/DECISIONS.json` (registered reading) and `claims/R1_ER1/DECISIONS.json` (revision R1 reading), both at
`rules_sha256` e2a6af23... and computed at 00:32:21Z / 00:32:23Z; `claims/MAIN`, `claims/R1` (strict, pre-ER1);
`claims/STABILITY/CLASSIFIER_STABILITY.json`; `decisions/QUAL_DECISION.json`; `fa/PROBE_STRENGTH_V0.json`. Documents:
`RESEARCH_DECISION.md` (RD) and `TABLES.md` (TB). "Follows" means I re-derived the label from the record's inputs with the frozen
code path (`frc.rules.d1`, `decisions`, `_d3`, `_d4`, `qual`).

| Item | Record value | Document value | Check |
|---|---|---|---|
| Outcome class (ER1, both readings) | `ENGINEERING_REPAIR`, flags [] | RD line 10, TB s5: `ENGINEERING_REPAIR` | OK. Follows: D2 not met, D1 `REPAIRED_TASK_ONLY_RELEASE` |
| Outcome class (strict, pre-ER1) | `INCOMPLETE` (`claims/MAIN`, `claims/R1`) | RD: not stated; RD line 96 says only "the strict labels are kept beside them" | **Discrepancy** (omission, see C1) |
| D1 | `REPAIRED_TASK_ONLY_RELEASE`, capacity 16x64, qpe `PE_ERROR_SUPPORTED` (also in strict records) | RD line 16, TB s2 and s5 | OK. Follows: state PROTECT, C64 (not the T6 repair), construction B1, `PE_ERROR_SUPPORTED`. One-liner wording, see C8 |
| Q-PE | `PE_ERROR_SUPPORTED` at C64 and C128 | RD line 39, TB s2 | OK |
| Original capacity gate | `ORIGINAL_CAPACITY_GATE_FAIL` for every construction at both capacities | RD lines 28, 39; TB s1 FAIL / FAIL | OK |
| QUAL | C64: A `NOT_QUALIFIED` (V1 men's guard 0.00255 > slack limit 0.002), B1 `QUALIFIED` (0.000185); C128: A and B1 `QUALIFIED` | RD line 40 (0.00255, 0.00019), TB s2 | OK |
| D2 | `NO_REDUCTION_BEYOND_ANY_CONTROL`, passed [], g1 `POINT_ELIGIBLE`, task_feasible true, flags [] | RD lines 17, 42; TB s5 | OK. Follows: no frontier passes |
| HQ_FRONTIER (both readings) | `NOT_SEPARATED_FROM_TASK_SPENDING`, bracket HQ_S2, `NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL`, benefit +0.000335 (se 0.005118) | RD line 42 "+0.0003 (se 0.0051)"; TB s4 | OK |
| ORD_FRONTIER (both readings) | `NOT_SEPARATED_FROM_TASK_SPENDING`, bracket ORD_S2, benefit +0.005295 (se 0.005898); interpolation 0.006366 (se 0.004723), "reported, never gating" | RD lines 42, 86 | OK (framing of the interpolation, see C5) |
| SHUF_FRONTIER, registered | `NOT_SEPARATED_FROM_TASK_SPENDING`, point null, `CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT` | RD line 42 "never reaches PRC's spend"; TB s4 | OK |
| SHUF_FRONTIER, revision R1 | `NOT_SEPARATED_FROM_TASK_SPENDING`, bracket UNT_BR, benefit +0.003114 (se 0.007640); interpolation 0.006767 (se 0.005406), "reported, never gating" | RD lines 42, 86 | OK (the interpolation exists only in the R1 reading, see C5) |
| TWIN comparison | benefit `AGG_POSITIVE_POINT_ONLY`, pass false, targeting_pass true, recipient_gate_pass true (no `label` key) | TB s4 lines 73, 75: "TWIN None (bracket TWIN, None)" | **Discrepancy** (rendering, see C10) |
| Reader targeting | registered: HQ_S2 false, ORD_S2 true, TWIN true; R1 adds UNT_BR false | RD line 42; TB s5 | OK |
| F4 | pass; V0 0.007153 (se 0.002245), V1 0.011135 (se 0.002313) | RD line 41 "0.00715 (0.00225) / 0.01113 (0.00231)" | OK |
| D3 | `NOT_SUPERIOR`; all nine references failing; six `NOT_CREDITED` | RD lines 18, 43, 53; TB s5 | OK. Follows: G4 fails on its own results. Framing of "easier to read than six", see C6 |
| D4 | `NOT_MET` (nrm G5 `NOT_MET`; NEURAL_RAW and H_LJ `NOT_CREDITED`) | RD line 19 "0.0177 / 0.0233"; TB s4 -0.0177 / -0.0233 | OK |
| D5 | `NOT_ASSESSED_MISSING_CONTROLS` | RD line 20; TB s5 | OK |
| D6 | `NOT_CLAIMED` | RD line 21; TB s5 | OK |
| Grid validity | ER1: `READER_SEEDS_EFFECTIVE`, 85 explained minimum-set equalities, 0 unexplained; pre_ER1: `READER_SEEDS_NOT_EFFECTIVE` | RD line 95 | OK |
| Robustness, 7 comparators | `ROBUSTNESS_NOT_CONFIRMED` each; all 32 cells: TWIN +0.0069 (0.0040), HQ_S2 +0.0005 (0.0062), UNT_BR +0.0031 (0.0077), H_LJ -0.0206 (0.0089, upper -0.0028), NEURAL_RAW -0.0147 (0.0105); n_eff distinct 26-29 | RD line 44; TB s7 | OK (headline framing, see C4) |
| Ablations | `ROBUSTNESS_NOT_CONFIRMED`; registered cells ABL_LOC vs PRC_S2 -0.0024 (0.0031), ABL_PAIR vs PRC_S2 -0.0032 (0.0042); all 32 cells -0.0006 (0.0035), -0.0022 (0.0038) | RD line 44 | **Discrepancy** (definitions reversed, cell set unlabelled; see C2) |
| Refit effectiveness | `REFIT_NOT_GENUINELY_EFFECTIVE`, reasons [`READER_SEEDS_NOT_EFFECTIVE`] (strict: 240 seed-dependent equalities on the minimum set, 144 explained elsewhere); splits pass (four distinct assignments, SCORE overlap 0.297-0.313) | RD line 45; TB s7 | OK (label). Wording, see C3 and the refit ruling |
| Strength probe | V0: one partition (sha 4257345f...) at all ten strengths 0.25-128 (`SATURATED`); V1: one partition 0.25-8, a new one 16-128 (`MOVES`); no revision R2 by the rule committed at 00:44:34Z (c7b80cb1c), before the probe ran at 00:49:21Z | RD line 46; TB s8 | OK |

Numbers spot-checked against the records and found right: every number in RD lines 39-46 except the items listed below, the
headline's 0.0039-0.0049, 0.0085-0.0090, 36%, 62-68%, 13-14%, 0.6555, 0.6485, +0.0070 (0.0040), 0.6489, 0.6516 and the
18 / 23 thousandths; TB s1-s8 against the claims and stability records. Not checked (not in the records I read): the MODEL_SELECT
income / occupation re-routing split 4.0% / 3.7% and 11.3% / 10.1% (RD line 41), and the 0.0062 maximum B0-B1 probability
difference (RD line 39).

## Discrepancies and required corrections

C1. **RD lines 10-12 (Decision).** The outcome class holds only under ER1. Required: "Outcome class `ENGINEERING_REPAIR` under ER1
(the post hoc grid-validity repair, `roles/statistical_reviewer/RULES_ER1.md`), identical under the registered and revision-R1
readings. Under the frozen strict rule, D2-D4 are `NOT_COMPUTED_GRID_INVALID` and the outcome class is `INCOMPLETE` (`claims/MAIN`,
`claims/R1`)." Also qualify "replayed by the independent verifier": the verifier replayed the grid, both ER1 readings and the
stability record, but did not recompute the P1 person-level numbers (spends, SEs, AUCs, the F4 bootstrap SE; REPORT_0058 line 11).

C2. **RD line 44 (ablations).** The definitions are reversed. Correct (PROTOCOL.md lines 290-291; `frc/fa/route.py` lines 101 and
104): ABL_LOC = pair weight 0, so the income and occupation (per-recipient) dependence terms only; ABL_PAIR = local weights 0, so the
pair term only. The values -0.0024 (se 0.0031) and -0.0032 (se 0.0042) are the 8 registered cells, not "all 32 cells" as the row's
opening suggests. Add the all-32-cell values -0.0006 (0.0035) and -0.0022 (0.0038).

C3. **RD line 45 (refit).** "the 16- or 64-valued views" is wrong for ORD_S2, whose views have 12 and 48 distinct values (48 of the
384 equalities). The 384 equalities are ET5 and ET25 on the income view and ET5 on the occupation view of the eight finite releases in
the refit bank (12 or 16 income values, 48 or 64 occupation values). ET25 never ties on the occupation view. "Seed-invariant
frequency tables by construction" must become "reduce to seed-invariant frequency tables on these views (the ER1 mechanism)". The
tie is a property of these views and leaf sizes, not of the recipe. Give the strict count too: 240 of the 384 are on the minimum
set. For the clause "every other seed-dependent reader changed with its seed", see the refit ruling below.

C4. **RD headline lines 32-33.** "Robustness readers on 24 new cells show the same pattern, and the routing stays 0.018-0.023 above
both target references" mixes two cell sets. 0.018-0.023 is the registered grid (8 cells). Over all 32 cells the figures are 0.015
(NEURAL_RAW, -0.0147, se 0.0105, not resolved) and 0.021 (H_LJ, -0.0206, se 0.0089, resolved worse). The 24 new cells are not 24
distinct evaluations: 26-29 of the 32 cells are distinct, because the extra-trees winners on the finite views do not change with
the reader seed. The required headline above fixes both points.

C5. **RD line 86 (interpolation).** The untargeted interpolation (0.0068, se 0.0054) exists only in the revision-R1 reading. The
registered untargeted path does not reach PRC_S2's spend, so it has no interpolation. Both interpolations are 1.3 se, linear between
two control points and never gating. The gating bracket benefits are smaller: +0.0053 (se 0.0059) at ORD_S2 and +0.0031 (se 0.0076)
at UNT_BR. See the interpolation ruling.

C6. **RD lines 18 and 54 (D3; "easier to read than six of the nine references, including both targets").** These are point
estimates. The difference is resolved at 2 se for H_LJ, B2TWIN, RETIRED_B3CTL and SUCC. It is not resolved for NEURAL_RAW (-0.0177,
se 0.0117) or TWJ (-0.0107, se 0.0080). Required: "has higher point-estimate pair AUC than six of the nine references (resolved
for four: H_LJ, B2TWIN, RETIRED_B3CTL, SUCC)".

C7. **RD line 84 (efficiency).**
- "fewer than half as many people (14% vs 33%, version 0)" holds only in version 0. In version 1 the figures are 12.8% vs 23.9%
  (0.54), so the version-1 numbers must be added and "fewer than half" dropped.
- "at 0.0026 nats lower task spend" is not resolved: se 0.0014, 1.9 se, by version 0.0026 / 0.0027.
- "the same fitted pair MI (0.0332)" holds in version 0 only. In version 1 the values are 0.0327 vs 0.0328.
- "efficiency edge" must go; see the ruling.

C8. **RD line 16 (D1 one-liner).** "meets the original limits with S2 slack and the two-SE reserve" makes the reserve look like part
of D1. QUAL is Q1 + Q2 only, and the reserve is reported separately and never gates (`frc.rules.qual` docstring; TB s3). Required:
"meets the original limits with S2 slack (its two-SE reserve, reported and never gating, is 3.15 / 2.71); the label-refit version
(A) does not (version-1 men's guard 0.00255 > 0.002)". It would also help to add that at 24 x 128 both A and B1 qualify.

C9. **RD lines 63-64 and 81-82.**
- "the targeted-versus-untargeted question is closed for this family by the R1 stopping rule" conflicts with the "Untested" list
  (other optimizers, the S3 budget). Required: "closed for this campaign at the registered optimizer and budget path".
- "The rest ... belongs to the finite partition itself" is not established; see the probability-estimation ruling.

C10. **TABLES.md section 4.**
- Lines 73 and 75 print "TWIN None (bracket TWIN, None)". The TWIN record has no `label` key. Print "TWIN: benefit
  AGG_POSITIVE_POINT_ONLY, pass False, targeting pass True".
- The "Version 0" and "Version 1" columns are "-" in every row. Fill them from the records or drop them. This is presentation only
  and no label changes.

C11. **RD line 89-90.** "Closing that gap" is the gap to parity. D4 needs a further 0.020 beyond parity, so the shortfall from the
D4 criterion is 0.038 (NEURAL_RAW) and 0.043 (H_LJ), 5-6 times the reduction the routing achieved against its own twin (0.0070).
Say so.

C12. **Erratum to my own ER1 text (RULES_ER1.md, "Failure" and "Mechanism"; frozen, not edited).** "16 income, 64 occupation
values" was the coordinator's description. It is exact for PRC_S2, TWIN, HQ_S2, SHUF_S2 and UNT_BR. The recorded distinct-input
counts are 12 / 48 for ORD_S2 and 8 / 32 for ORD_S3 (`claims/MAIN_ER1` `n_eff_implied`). The rule is unaffected, because
explanation (c) uses the recorded count, not a stated value. The documents must not repeat "16 or 64" as a property of all finite
releases.

## Wording rulings

**Refit.**
- The label stays `REFIT_NOT_GENUINELY_EFFECTIVE` (reason `READER_SEEDS_NOT_EFFECTIVE`) under the frozen strict rule. The
  coordinator's 00:31Z decision not to extend ER1 to the refit stands, and the documents must state the label without softening it
  (not "technically", "only formally", or "effective apart from").
- `REFIT_EQUALITIES.json` is a fair descriptive account. Its counts reconcile: 18 recipe-views x 160 checks = 2,880 seed-dependent
  checks; 384 ties = 3 recipe-views x 128 = 8 finite releases x 48; 2,496 differ. It applies the same NRM1 (iii) explanation ER1
  uses, records `labels_read: false`, and says it changes no label.
- Its scope is the FRC1_ROB bank: the eight finite releases plus H_LJ and NEURAL_RAW. The other benchmark references and ORD_S3 are
  not in it.
- Permitted: "Descriptively (`frc/refitcheck.py`, not a rule outcome), every seed-dependent tie in the refit bank (384 of 2,880
  checks) is an extra-trees reader (ET5 or ET25 on the income view, ET5 on the occupation view) on one of the eight finite releases,
  and each meets the NRM1 (iii) explanation. Every other seed-dependent check in the bank (2,496 of 2,880) differed between its two
  reader seeds."
- The clause "every other seed-dependent reader changed with its seed" is supported only in that check-level, bank-scoped form.
- Required consequence, stated next to it: for those recipes the k2 / k3 cells duplicate the k0 / k1 cells, so the robustness screens
  count them once (26-29 distinct of 32 cells). The robustness evidence therefore varies the split fully but the reader seed only
  partly.
- Not permitted: "the reader seeds were effective", "the refit was effective under ER1", "24 independent robustness readers", or
  "seed-invariant by construction".

**Efficiency edge over HQ_S2.**
- Not permitted: "efficiency edge", "matched recovery at lower cost", or any wording that ranks PRC_S2 above HQ_S2. The pair benefit
  is +0.0003 (se 0.0051). Robustness gives +0.0005 (se 0.0062) with 13 of 27 distinct cells opposite. At 25% and 50% reader-FIT
  data HQ_S2 is lower (-0.0013, -0.0040; TB s9). The spend difference is 1.9 se, and the re-routed share is not a registered cost.
- Permitted: "PRC_S2 and HQ_S2 reached about the same recovery (benefit +0.0003, se 0.0051) and the same fitted pair MI (0.0332 /
  0.0327 vs 0.0332 / 0.0328). PRC_S2's primary task spend was 0.0026 nats lower (se 0.0014, not resolved), and it re-routed fewer
  people (14% vs 33% in version 0, 13% vs 24% in version 1; re-routing is not a registered cost). No efficiency difference is
  established."

**Interpolated SEX-blind paths.**
- Permitted only as: "Interpolated linearly at PRC_S2's spend (reported, never gating), the ORD path sits 0.0064 (se 0.0047) above
  PRC_S2 in both readings, and the untargeted path 0.0068 (se 0.0054) above it in the revision-R1 reading only (the registered
  untargeted path never reaches PRC_S2's spend). Both are point estimates of about 1.3 se. The gating bracket benefits are +0.0053
  (se 0.0059) and +0.0031 (se 0.0076)."
- Not permitted: "PRC_S2 beats the SEX-blind paths", pooling the two values, or putting either value in a headline or summary
  without "never gating".
- The sentence on standard errors being 20-40% of the 0.020 target is correct and may stay. Add that the D2 record's minimum
  detectable pair benefit is about 0.015 (2 se_agg).

**Probability-estimation answer.**
- "Probability estimation explained about a third (16 x 64) and about two thirds (24 x 128) of the lost margin" is not permitted:
  "explained" is causal, and "lost margin" covers clauses that were not decomposed.
- Permitted: "Replacing the label-refit tables with out-of-fold teacher means removed about 36% (16 x 64) and 62-68% (24 x 128) of
  the label-refit twin's nested held-out occupation-NLL spend over the continuous task teachers. These are point ratios without
  intervals. The NLL reductions themselves are resolved (3.3-8 se)."
- Replace "The rest ... belongs to the finite partition itself" with: "The remainder, 0.0068 / 0.0088 nats at 16 x 64 and 0.0052 /
  0.0042 at 24 x 128, was not reduced by either teacher-mean construction (B0 and B1 agree within 0.000005 nats). Whether it is the
  partition's own cost or remaining per-cell estimation error was not tested."

## Deviation 9 check

Deviation 9 says that moving the ER1 text changed no rule. **Confirmed.** Checked by script from git objects and the working tree,
reading no labels:
- At 6d6cd0829 the two registration files hash to 0121711e... (RULES.md) and e2a6af23... (RULES.json). Their parent versions hash to
  2b7e0be8... and a0dc0fc7..., the registration receipt.
- The working-tree RULES.md and RULES.json are byte-equal to the receipt versions, and every SHA-256 recorded in `RULES_ER1.json`
  matches.
- RULES.md at 6d6cd0829 differs from the receipt version only by two inserted blocks: the 2-line pointer in section 9 and sections
  15-16. All 43 non-empty inserted lines appear verbatim and in order in `RULES_ER1.md`, with no non-insert change.
- RULES.json at 6d6cd0829, with `reader_seeds.engineering_repair_ER1` and the last `change_record` entry removed, equals the receipt
  RULES.json. Both entries equal the copies in `RULES_ER1.json`.
- `git diff 6d6cd0829 HEAD -- frc/rules.py frc/test_rules.py` is empty, so the code that implements and tests ER1 is untouched.
- ER1 changed `frc/rules.py` in 6d6cd0829 itself. That is the repair, disclosed as deviation 6, not part of the move.
- `python3 -m frc.test_rules`: 18 of 18 PASS (all_pass true, rules_sha256 a0dc0fc7..., invented data only).

One consequence must be disclosed: the claims records carry `rules_sha256` e2a6af23..., which is the post-ER1 RULES.json kept at
6d6cd0829, not the restored file. ER1 adds no constants (`load_rules` checks only registered constants), so the same claims run today
would hash a0dc0fc7... with identical labels. Add one sentence to deviation 9 so a reader can match the hash.

## Additions to limitations

Add to "Limits of this evidence" in RESEARCH_DECISION.md:
1. **ER1 dependence**, stated in the Decision section as well (C1). Every D2-D4 label exists only under a post hoc repair. The
   strict outcome class is `INCOMPLETE`.
2. **One protected point per version.** The registered strength path realized one distinct nonzero partition per version:
   - S2 and S3 both select strength 8, and the S2 budget did not bind (coordinator log 00:34Z).
   - In the label-free full-pool chain, every registered strength from 0.25 to 8 gives the same partition in both versions.
   - D2 therefore compares one protected partition per version with the controls, not a traced protected frontier. "Strongest
     registered setting" must not suggest a dose-response.
3. **Reader seeds partly ineffective in the robustness bank.** The strict refit label applies. 26-29 of 32 cells are distinct, and
   the robustness evidence varies splits more than reader seeds.
4. **Detectability and multiplicity.** The minimum detectable pair benefit is about 0.015 (2 se_agg). The observed control
   differences (0.0003-0.0070) are below it, so "not separated" is the default reading, not evidence of equality. Many comparisons
   were screened at 2 se without adjustment. Only the Q-PE reductions (3.3-8 se) are clear of this.
5. **Data-limited readers.** Every release's pair AUC still rises with reader-FIT data (TB s9). Measured recoveries are therefore
   lower bounds for better-trained readers, and the PRC_S2 / HQ_S2 ordering changes sign across FIT fractions.
6. **Verifier scope.** The P1 person-level spends, SEs, AUCs and the F4 bootstrap SE were not independently recomputed.
7. **Robustness against H_LJ.** Over all 32 cells, PRC_S2 is resolved worse than H_LJ (upper bound -0.0028). Here
   `ROBUSTNESS_NOT_CONFIRMED` means no benefit, not an unstable difference.
8. **Rules hash.** The claims records cite the post-ER1 RULES.json hash e2a6af23... (see the deviation 9 check).

Add to "What stays open":
- Whether the remaining occupation-NLL spend after B1 is the partition's own cost or remaining estimation error. Other per-cell
  estimators were not tried.
- A traced protected-routing frontier: more than one protected point per version, which needs an optimizer that moves below strength
  16. The S3 budget point is already listed.
- The registered-reading untargeted bracket: no SHUF point reaches PRC_S2's spend.
- An effective-reader-seed robustness check. Robustness under new reader seeds for the extra-trees recipes on finite views was not
  achieved, and ER1 was deliberately not extended to the refit.

## Completion record

- Started 2026-10-21T01:03:14Z (`date -u`); file written after the 01:10:50Z reading; the completion time is the `date -u`
  reading in the hand-back report, taken after this file was written and the CPU charge was recorded.
- Tests run: `python3 -m frc.test_rules` once, 18 of 18 PASS, exit 0 (invented data only).
- Other computation: one label-free comparison script (git objects and working-tree registration files, SHA-256 and line
  comparison), `git show` / `git diff`, and small `python3 -c` reads of saved JSON records (claims, stability, QUAL, probe).
- CPU charged with `python3 -m frc.jobqueue charge --label statistical_reviewer`: 15 CPU-s (upper estimate).
- No label was read, no file under `.private/` was opened, `frc.common.load` was not imported, and no real-data computation was
  run. No file other than this one was edited; nothing was committed, pushed or sent.

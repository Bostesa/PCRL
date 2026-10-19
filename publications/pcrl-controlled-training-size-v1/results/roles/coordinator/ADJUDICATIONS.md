# Coordinator adjudications

Each adjudication keeps the original result, states the cause and the decision, and never widens an acceptance rule. Times are
`date -u` readings taken before writing.

## A1 (00:28Z): verifier check V0.PINS.PUBLICATION_TREE, V1 -> V2 (CHECK_CHANGES.json change 1)

- **Before.** V1 FAIL: 21 of the 342 published results files of e32a185b4 did not equal the local copies after the source sanitiser.
- **Cause.** A verifier implementation bug: V1 sanitised the raw bytes of local text files, keeping the CRLF line endings of 21 CSV
  files, whereas the source publication build reads text with universal newlines (CRLF -> LF) before sanitising. The published blobs
  equal the local files after exactly that transformation (the verifier's probe; the coordinator's own admission, tsz.admit, reached
  the same counts 342 / 319 / 23 independently, SOURCE_ADMISSION.json).
- **Decision.** V2 PASS is accepted as the check result. The acceptance rule is unchanged (byte equality, or equality after the
  source build's own text transformation, with counts equal to SOURCE_ADMISSION.json). V1 FAIL stays in CHECKS.json.

## A2 (00:41Z): engineering review wave 1 (roles/engineering_review/REVIEW_WAVE1.md), before any size analysis or selection

| Finding | Severity | Disposition |
|---|---|---|
| ENG1-01 size-table completeness (missing cells silently dropped; write-once record) | BLOCKER (conditional) | APPLIED at 00:39Z (tsz/evaluate.py), before `tsz.evaluate size` was run; carries ENG1-06 / ENG1-07. A dated code amendment (PROTOCOL_DEVIATIONS item 5) implementing RULES.md sections 2 and 3.3 |
| ENG1-02 `tsz.correct realized` could block the other version's selection | MAJOR (operational) | NOT APPLIED: `tsz.correct realized` is never run in this campaign; realized MODEL_SELECT costs come from `tsz.evaluate conditional`, which requires both selections committed. Applying it would change tsz/correct.py while its lanes run |
| ENG1-03 identity release not bitwise W | fixed by its owners before the freeze | verified by the reviewer; no action |
| ENG1-04 trigger does not read the coordinator's second deployment record | MINOR, retrospective | No re-run. The two records agree: verifier DEPLOY_W100.json (sha256 f5de9259...: "0" and "1" bitwise true, refusals_all_pass true) and coordinator deploy/DEPLOYMENT_CHECKS_W100.json (sha256 3c47828e...: all_bitwise true, all_refused true) for the same packages W100_V0_B2 / W100_V1_B2; TRIGGER_FIRES stands |
| ENG1-05 IR4 reading missing when SIZE_EFFECT_UNRESOLVED with another pattern | MINOR | APPLIED at 00:40Z (one line in tsz/rules.py `interpretation`), implementing the registered RULES.md 3.6 text; labels and every other rule unchanged; 16 rules tests pass. PROTOCOL_DEVIATIONS item 5 |
| ENG1-08 label-free `load(..., subset=...)` returns all rows | MINOR | NOT APPLIED: no caller uses it, and tsz/common.py is pinned by the verified deployment packages |
| ENG1-09 nested npz-before-json blocks a retry | MINOR | DEFERRED: applied only if a nested retry is needed (changing tsz/nested.py while its lanes run would mix code versions) |
| ENG1-10 PAIR's critic reads the frozen H3 income (fitted on all MODEL_FIT) in every outer fold | MINOR, disclosure | Disclosed (EXPOSURE_MAP.md, PROTOCOL_DEVIATIONS item 5, decision documents) |
| ENG1-11 selection guard wording | MINOR | Handled by process: corrected-release words kept out of MODEL_SELECT read purposes until both selections are committed (verifier told at 00:33Z) |
| ENG1-12 conditional evaluation and claims wiring | note | no defect |

## A3 (01:38Z): reader-seed check scope (the statistical reviewer's ruling A-SC1)

- **Outcome.** READER_SEED_CHECK.json (01:31:29Z): literal registered label READER_SEEDS_NOT_EFFECTIVE; 16 equalities, all ExtraTrees
  (ET5 / ET25) on the income view of B2TWIN and RETIRED_B3CTL (6 distinct income vectors each).
- **Ruling.** The rule owner ruled A-SC1 (roles/statistical_reviewer/ADJUDICATION_SEEDCHECK.md, completed 01:36:00Z), classified as an
  amendment after the outcome and before any stability label, having read no reader outcome. The coordinator applies it unchanged:
  grid validity comes from the amended criterion (tsz.claims.a_sc1_validity), both labels are reported, the disclosure sentence of
  the ruling's section 5 accompanies every reader label, and COARSE_REFERENCE_INCOME_WIRE is attached as specified (never gating).
- **The coordinator's role.** The coordinator requested the ruling, wrote the cardinality counts (label-free) and implemented the
  criterion; it did not choose the scope. No reader outcome had been read by the coordinator when the ruling was made.

## A4 (written after the verifier's V5 / V6 replay): verifier check changes 2 and 3 (CHECK_CHANGES.json)

- **Change 2, V5.CORRECTION.TARGETS_INVARIANCE V1 (ATTENTION) -> V2 (PASS).** Cause: the verifier's sensitivity control (guard
  'static' with equal men / women targets) could not show sensitivity, because equal targets give equal weights after normalization.
  The registered criterion (theta bitwise invariant to `targets` under guard 'none' for CONT, LOC and PAIR) was met in V1 and is
  unchanged; V2 uses unequal targets in the control. Accepted as a verifier implementation bug; V1 kept.
- **Change 3, V6.TABLES V1 (FAIL) -> V2 (PASS).** Cause: the verifier compared the 0.020 target label for all six references,
  whereas RULES.md sections 9-11 define it only against NEURAL_RAW and H_LJ; all 20 differences were the four empty-by-definition
  references. Accepted as a verifier implementation bug; the acceptance rule is unchanged; V1 FAIL kept in CHECKS.json.
- Neither change widens an acceptance rule.

## A5 (01:57Z): claims review (roles/claims_review/REPORT.md) of RESEARCH_DECISION, ADVISOR_BRIEF and START_HERE

- All 10 MUST-FIX items (M1-M9, with M7a / M7b) and all 15 SHOULD-FIX items were applied as proposed, and the five notes were applied
  where they change text (N1, N2, N3, N5; N4 is reflected as "descriptively in line, different partitions"). Highlights: the
  control's difference in gains is version-dependent on women's Brier (not "not resolved for any metric"); utility is stated as the
  original point limits plus, separately, the reserve heuristic, with the original recipient gate failing in the occupation view
  against every reference; the A-SC1 disclosure sentence is quoted where reader labels are reported, with the qualification that the
  verifier had already computed its own labels; causal and recipe-specific wording replaced by the registered scope.
- The closure labels are unchanged on every reading: CORRECTION_RECIPE_CLOSED and C1 MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL.

## A6 (01:57Z): verifier check change 4 (V6.TEXT_SCOPE_SCAN V1 FAIL -> V2 ATTENTION)

- Cause: the verifier's automated scan matched negated mentions of forbidden terms and missed a line-wrapped phrase (a verifier
  implementation bug). V2 reports no forbidden claim and six wording points, all of which were also claims-review items and are fixed
  under A5. Accepted; V1 kept in CHECKS.json; the acceptance rule is unchanged.

# Coordinator adjudications (append-only)

## A1. Engineering review wave 1 (roles/engineering_review/REVIEW_WAVE1.md), adjudicated about 15:44Z (times from `date -u` at commits)

| Finding | Disposition |
|---|---|
| ENG1-01 truncated dirty paths | **Applied** (442b84694, comment 86a995e6f): `git status --porcelain -z`. Earlier ledger rows are kept as known-truncated, never rewritten. Dirty launches are still recorded, not refused; the commit-before-use rule stays a process rule, audited by the verifier (R9). |
| ENG1-02 MODEL_SELECT gate bypass through coordinator helpers | **Applied** (442b84694): a MODEL_SELECT read is refused when any fitting module (nrm.fa / fb / fc / ctl) is on the call stack; label reads are refused outside queue units (no NRM_UNIT_ID); the full module chain is logged with every read. The registered ACCESS_POLICY.json is unchanged: these only tighten the gate. The verifier's R10 remains the after-the-fact cross-check. |
| ENG1-03 unlocked configuration budget | **Applied** (442b84694): an exclusive file lock spans the budget check, the write and the ledger append. The duplicate `configuration_number` from the concurrent FB saves is deviation 9; the budget counts distinct triples. |
| ENG1-04 restore check on randomized releases and mixed interpreters | **Applied** (fa863c9b4): draw keys are passed; each release and package is re-run only under its own fit interpreter; per-release errors are recorded; custody runs both interpreters and merges the passes. |
| ENG1-05 inconsistent wire hashes | **Applied** (705966524): EVAL_MS.json stores `wire_sha256_eval` per version in the grid's format (scm_bench wire_sha; a list over draws for randomized releases); reference evaluations store theirs too. The selection driver uses these. |
| ENG1-06 stale draw-0 fields; reserve without MC; MC SE | **Applied** (705966524): derived fields are recomputed from draw averages; the governing `nrm.rules.reserve_status` is recorded (`reserve_rules`); per-version reserve columns are descriptive (`reserve_without_mc`); MC SEs come from per-draw means, combined in quadrature. The FB_* evaluations written before this commit keep their original records; FB_* are not claimable (G1 fails). |
| ENG1-07 FA one-version saves | **Moot.** Family A closed before stage 2 (capacity). Designer A acknowledged it. |
| ENG1-08 FB revision file layout | **Required of Designer B before any FB2 fit** (instruction of 15:36Z). |
| ENG1-09 biased or undeployable random-replacement control | **Applied by not using C5.** The funded challenger control is CTL_REM (deterministic remainder flattening), with its shares fitted on CALIBRATION or out-of-fold vectors, as instructed for the CTL v2 card. RAND (C5) is not funded. |
| ENG1-10 fix before the lock | **Applied:** all release.py and devreaders.py changes, and the reader-robustness extras, went in before the lock (1128afd67, 15:43:59Z). |
| ENG1-11 card not tied to family | **Applied** (442b84694): the card prefix must match the family in `load` and in `save`; the template is refused. Closeout's `late_card` check is kept as a vacuous belt-and-braces check and labelled as such. |
| ENG1-12 devreaders draw docstring; MC assembly | **Applied** (7222b0105): docstring corrected. The `mc_cell` assembly from the +2/+4 replicate members is done by the coordinator's claims code (nrm.claims). |
| ENG1-13 stale FB learner docstring; SEX-blind controls | **Required of Designer B** (15:36Z instruction). SEX-blindness of the twin and MINF/ISO controls is cited from the verifier's invariance check. |
| ENG1-14 ORD operating point SEX-informed | **Moot** (family A closed). Recorded wording, if ever needed: "SEX-blind mapping, SEX-informed operating point". |
| ENG1-15 TORCH_NUM_THREADS ignored by torch | **Noted.** Every torch entry point calls `torch.set_num_threads(1)` (FB does). |

## A2. Reader-robustness grid extras (roles/reader_robustness/PROPOSAL_GRID_EXTRAS.md), about 15:43Z

**Accepted in full**, before any candidate outcome on the grid:
- (a) score fingerprints, used for `grid_valid`;
- (b) the NOLOOKUP winner, for the smooth-reader qualifier;
- (c) TABLE_CV and the enhanced winner.

All three are descriptive and never change a registered winner or decision. They were integrated before the lock (7222b0105, lock
1128afd67). Grid validity is computed by the coordinator's claims code from the fingerprints through `nrm.rules.grid_validity`. The
verifier checks it independently.

## A3. No replacement for family A

Family A closed at the capacity stage (FINITE_CELL_CAPACITY_NOT_ESTABLISHED). The coordinator does not launch a replacement family:
- Deterministic many-to-one releases pay a measured capacity cost of about 0.013 nats at readable resolution (64-128 cells).
- Continuous deterministic releases are injective, so any gain would be learnability only (the MMP lesson).
- The remaining information-removing route is randomization, which is family B's revision FB2.

The freed configuration budget goes to FB2's controls, the decisive ablations and a training perturbation.

**Erratum to A3 (added after 16:27Z).** The second bullet above repeats a claim that was withdrawn on 2026-10-16: determinism and
continuity do not imply injectivity, the population injectivity of the actual models is unknown, and a deterministic many-to-one
probability release can remove SEX information inside the current contract. That bullet is withdrawn as a reason. The decision rests on
the first and third bullets. PROTOCOL_DEVIATIONS.md item 12.

## A4. No replacement family after family B closed (16:32Z)

Family B closed by its registered stop rule (F1 holds on both declared paths in claims/FINAL; FB2P NOT_RUN by rule 2(a); MEAN
NOT_FITTED by C5; selection NO_DEVELOPMENT_CANDIDATE). The brief allows one reviewed replacement family. The coordinator does not launch
one, for these reasons, all from this campaign's registered grids (NRM1_MAIN, NRM1_CTL; registered-cell mean SCORE AUC, pair / income /
occupation):

- **The gap is in the predictor, not in post-processing.** The strongest admitted task-only base (FB2_BASE = HC_STRONG income + A1_PO_16
  occupation) reads 0.6605 / 0.5569 / 0.6428. The target references read NEURAL_RAW 0.6361 and H_LJ 0.6327 (H_LJ 0.5278 / 0.5811). Closing
  0.024-0.028 of pair AUC within the original limits is what any new family would need.
- **Post-hoc randomization buys about 0.01 at the affordable spend, with no targeting advantage** (FB2_SEX_S5 0.6496 vs FB2_MINF_S5 0.6505
  at matched spend; benefit +0.0019, se 0.0089).
- **Measured for the declared readers: the partial (invertible) flattening changed recovery by about 0.001; the full (many-to-one)
  collapse was too expensive.** (Wording amended at the statistical reviewer's closing condition; the first version of this heading read
  as a general principle, which the withdrawn injectivity lesson does not support. Commit f728dc1af keeps it.) CTL_REM_0p50 (partial,
  invertible remainder flattening) reads 0.6596 / 0.5569 / 0.6419 at a feasible spend. CTL_REM_1p00 (the remainder replaced by fixed shares: decision and top probability
  only) cuts the occupation view to 0.5603 and the pair to 0.6457 (resolved, +0.0148 vs FB2_BASE), but fails the occupation NLL and Brier
  clauses and the men's guard by about 2.7-3.0 times the cap, and the pair stays above both target references. The linked pair keeps
  SEX information that the occupation remainder alone does not carry.
- **Finite cells pay a measured capacity cost** of about 0.013 nats at 64-128 cells (family A's gate), before any protection.
- **The remaining routes are closed or not authorized:** the leaf-score correction line behind the references is closed by the brief; the
  MMP mixture awaits the advisor's target decision recorded on 2026-10-16 ("the pilot is not authorised"); a coalition-aware correlated
  channel is what FB2's coupled rank-1 construction already tested at this scale.

No candidate route has a plausible path to the 0.024-0.028 gap inside the original limits with the evidence in hand, so the remaining
budget goes to the decisive verification, the robustness readers, deployment, custody and publication. This is a coordinator decision,
not a frozen rule, and it is not evidence that no mechanism can close the gap. Neither the statistical reviewer nor the challenger
objected (their closing reports).

**Routes still untested** (the challenger's closing list; "no replacement" does not mean these were exhausted): propensity-routed
finite cells beyond the twin (FA's HQ, ORD and SHUF never ran); continuous-mixture and temperature-matched controls; randomized response
on the decisions; input-side recoding or erasure of the six fields or of the 145-column encoding; training-time SEX-balanced
reweighting; the MMP mixture (not authorised); an isolated coupled-versus-decoupled randomness contrast.

## A5. Engineering review wave 2, code (ENG2-01 to ENG2-11; report saved verbatim in roles/engineering_review/REVIEW_WAVE2_CODE.md)

None is blocking; none changed a label or decision of this campaign. Fixed in commit 735cde202 unless stated:
- **ENG2-01 (claims path members dropped silently): fixed.** Every declared path member reaches `frontier_comparison`, so a missing
  cost gives INCOMPLETE. In claims/FINAL every path member carried costs, so no label changed.
- **ENG2-02 (failed conjunction recorded as no alternative): fixed.** Rerun as claims/FINAL_V2 (16:52:14Z): exactly one field differs
  from claims/FINAL (FB_SEX_S2's selection-record G2 now NOT_MET, matching its summary); summaries and selection identical. FINAL_V2 is
  the canonical final claims; FINAL is kept with a superseded note. F1 and MEAN re-applied to FINAL_V2 (decisive/F1_FINAL_V2.json:
  F1_HOLDS on every declared path and on the MINF reading; MEAN_RULE_FINAL_V2.json: NOT_FITTED).
- **ENG2-03 (cost merge order): fixed** (time order). No duplicates existed.
- **ENG2-04 (F1 read only the MINF path): fixed.** `decisive.f1` reads every declared path, with the MINF reading beside; both give F1
  holds.
- **ENG2-05: fixed** for the perturbation (a missing input is INCOMPLETE). For F2, NOT_READ_INCOMPLETE is kept as the protective reading
  (an INCOMPLETE screen is completed first, as C6 treats INCOMPLETE records) and documented in the module. Neither path ran.
- **ENG2-06 (custody could fail falsely across interpreters): fixed for future runs.** It did not occur here: the backup's restore check
  (16:46:17Z, before the fix) re-scored all six reader pickles bitwise under both interpreters.
- **ENG2-07, ENG2-10, ENG2-11: recorded, not changed.** None occurred; the CALIBRATION role order is fixed by the access layer.
- **ENG2-08 (base-instance AUCs beside pooled benefits): fixed.** TABLES.md section 1 labels the base instance per cell and adds a Monte
  Carlo pooled pair AUC for randomized releases; section 4 uses the pooled values; documents cite pooled values (FB2_SEX_S3 0.6536,
  FB2_SEX_S5 0.6486, FB2_MINF_S3 0.6564, FB2_MINF_S5 0.6505). The figure keeps the base instance, labelled on its axis. A4 above quotes
  base-instance values, as it was written.
- **ENG2-09: fixed** (family by saved folder).

## A6. Engineering review wave 2, documents (DOC2-01 to DOC2-15; report saved verbatim in roles/engineering_review/REVIEW_WAVE2_DOCUMENTS.md)

None is blocking; none changes the decision. All were corrected in the documents before publication:
- **DOC2-01 (CTL_REM_1p00's failure misdescribed): corrected** in ADVISOR_BRIEF, RESEARCH_DECISION and CLOSEST_WORK: occupation NLL at
  1.8-2.2 times the cap and the men's guard at 2.7-3.0 times; occupation Brier passes. The same imprecision appears in A4 above ("fails
  the occupation NLL and Brier clauses and the men's guard by about 2.7-3.0 times the cap") and in the coordinator's 16:33Z message,
  which the challenger's verbatim report repeats; A4 stays as written with this correction, and the challenger's saved report carries a
  coordinator note.
- **DOC2-02: footnoted.** The reviewer's headline wording ("about 0.011") is kept verbatim; a reading note says it is the base-instance
  value and gives the Monte Carlo pooled +0.0120 (se 0.0073).
- **DOC2-03, DOC2-04: reworded** ("not separated", and interpretations marked "consistent with ... not tested directly"; the readers'
  pair AUC fell about 0.012 under both channels).
- **DOC2-05: corrected** (this campaign's own capacity curve, the Adult finite codes, the gate as estimate + 2 SE ≤ 0.004 nats).
- **DOC2-06: corrected** (FINAL_V2 governs; MAIN_V2, FINAL and FINAL_V2 all replayed by the verifier).
- **DOC2-07: resolved by a final same-device backup** taken at the final research tip, before publication (custody/BACKUP_VERIFICATION.json).
- **DOC2-08: resolved** (VERIFICATION.md, ledgers and receipts now exist).
- **DOC2-09: corrected and tested** (zero-padded codes; unknown codes fall into `__OTHER__`; example run as unit NRM_QUICKSTART_EXAMPLE_V1).
- **DOC2-10: corrected** (figure caption names the base instance). The message of commit b2f0dccce says the figure uses pooled values;
  it does not (the figure keeps the base instance, labelled on its axis). The commit message stays as history.
- **DOC2-11, DOC2-12, DOC2-14, DOC2-15: corrected** (both seed labels named; family A's declared PRIOR_ART_FOUND against the twin's
  invalid-grid G7; the Gaussian-mechanism citation; elapsed time pointed to RESOURCE_ACCOUNTING.json; "verified on the remote"; the
  family-B v1 interpreter; write-once store and build; temperature before noise).
- **DOC2-13: corrected** (nrm.tables computes PRC_TWIN's reserve from its evaluation record: not qualified, minimum ratio 1.29).

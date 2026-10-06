# PCRL confidence-budgeted privacy compression (cbp) — registered protocol

Branch `research/pcrl-confidence-budgeted-privacy-v1`. Source: `research/pcrl-confidence-capacity-v1` (qpc), evidence
commit `9dd06da6b64e558e1c079f76e43982b60b327e63`, handoff tip `d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351`.
Package `cbp/`. Public results `results/pcrl_confidence_budgeted_privacy_v1/`. Private store `<PRIVATE_CACHE>/cbp_v1/`.

This protocol is frozen in `FIT_LOCK.json` and `AUDIT_AND_SELECTION_LOCK.json`, and both are pushed and remote-verified
before any new mapping fit. The executable definitions are the locked code. Where this text and the code disagree, the
locked code and tests are authoritative, and the disagreement is disclosed as an error.

## 1. Question

At the fixed code capacity that preserved confidence in qpc (income 8 / occupation 64 states per teacher-predicted
class, "i8o64"), can a privacy weight smaller than 0.1 keep both tasks' confidence within the ORIGINAL allowances while
still reducing combined-view SEX recovery beyond strong privacy-untrained compression? Any family may win. Joint design
earns nothing unless claims A and B pass.

Scope: a fixed-task output release (token identity, decoded probabilities, unchanged decision per recipient). It is not
general-purpose representation learning.

## 2. Exposure (registered verbatim; see EXPOSURE_LEDGER.md)

"The design is motivated by opened Adult development results, including the completed qpc confidence-capacity study. The
fitting, inner-selection and assessment data have all been used historically. This is an exploratory, locked development
comparison. Its nominal intervals condition on fitted artifacts and do not account for the adaptive research history. It
is not fresh confirmation or a population privacy guarantee."

- **Roles (unchanged):**

  | Role | Rows |
  |---|---:|
  | OSF_DEFENSE_FIT | 15,434 |
  | HEAD_VALIDATION | 1,500 (historical head selection only) |
  | AUDIT_FIT | 6,065 |
  | INNER_SELECTION | 2,235 |
  | OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) |

- **Input:** `adult_jcv.npz`, sha256 `e0d9e54a…2f12`. The 83 permitted columns are used in the pinned order.
- **No new population:** no new state or year is opened, and no confirmation data is spent.
- **Assessment rows:** assessment labels and prior per-person assessment predictions are sealed out of fitting, selection
  and debugging (cbp.data). Only `cbp.assess` can unseal, and only after the pushed EVALUATION_LOCK.

## 3. Budget and resources

- **Time and CPU limits:**
  - 10 h elapsed from 2026-10-06T16:44:45Z (hard ceiling 2026-10-07T02:44:45Z);
  - 20 CPU-h in total;
  - a reserve of the final 2 h and 4 CPU-h, starting 2026-10-07T00:44:45Z.
- **Process limits:**
  - at most two heavy numerical processes in total, through one shared two-slot semaphore (`cbp.sema`, flock, with
    signal forwarding) and one ledger (`SEMA_LOG.jsonl`, `COMPUTE_LEDGER.jsonl`);
  - `OMP_NUM_THREADS=1` and the BLAS equivalents;
  - 8 GiB memory and at least 5 GiB free disk.
- **Cloud spend:** $0.
- **Timing:** the full bank was timed on synthetic data before fitting (B: `TIMING.json["fitting"]`; D:
  `AUDIT_COMPUTE.json`). Timing decides scheduling only. There is no outcome-dependent λ pruning.
- **Reserve boundary:** once the reserve boundary is reached, no new work is launched, state is saved for resume, and
  incomplete work is marked as such.

## 4. Admitted sources (SOURCE_ADMISSION_LOCK.json, pushed at 5cc8ecb)

- **Admitted units:**
  - frozen U teachers and RAW-J β0.3 for seeds 0–2, reproduced bitwise;
  - fine partitions (income 32 / occupation 128 per predicted class);
  - DIRECT-TASK i8o64, FINE-TASK i8o64 and CLASS-ONLY i1o1;
  - LOCAL, SEQ-12, SEQ-21 and JOINT i8o64 at λ 0.01 and 0.1 (24 endpoint units);
  - REF|E (official LEACE), REF|F (official FARE) and REF|F0 (no-fairness FARE);
  - 11 composition-only public maps: the 7 other DIRECT-TASK rates and the four λ=1 i8o64 maps. They exist only so that
    SRC|U can compose them; they are never candidates or scored releases.
- **Legacy JSON:** qpc JSON contains `Infinity` sentinels. The source bytes are kept, and cbp writes finite-or-null JSON
  only.

## 5. Fixed capacity and the intermediate bank (FIT_MANIFEST.json)

- **Fixed settings:** one rate, i8o64. λ grid exactly {0.01, 0.025, 0.04, 0.06, 0.08, 0.1}. Families LOCAL, SEQ-12,
  SEQ-21 and JOINT. Seeds 0, 1 and 2.
- **Unit counts:** 72 logical privacy units, of which 24 endpoint units are reused after parity and 48 new fits.
- **References:** 3 × (DIRECT-TASK, FINE-TASK, CLASS-ONLY), plus the continuous references.
- **Endpoint parity:** every reused unit passes `cbp.fit.endpoint_parity` before any new fit. The gates are:
  - COMPLETE.json hashes;
  - policy integrity;
  - configuration and fingerprint;
  - bitwise re-encode of the release;
  - class preservation on all rows;
  - exact fitting statistics;
  - fitting terms within a relative 1e-12;
  - fine partition;
  - code pins equal to the qpc STAGE_A/B locks;
  - binding;
  - JOINT witness dominance against the witness records.

  Any failure stops the stage. Tolerances are not relaxed.
- **New fits:** every new fit calls `qpc.compress.fit_unit(family, fine, T, OSF_DEFENSE_FIT, S_fit, 8, 64, λ, meta,
  witnesses)` unchanged.
- **Objectives:** D_i is the mean fitting KL(teacher ‖ decoded); I1 and I2 are the plug-in MI between SEX and each token;
  I12 is the same for the aligned pair.
  - F_task = D1 + D2
  - F_local = D1 + D2 + λ(I1 + I2)/2
  - F_joint = D1 + D2 + λ[(I1 + I2)/2 + I12]
- **Corrected sequential design:** the first recipient is optimised under F_joint, holding the other recipient's
  CLASS-ONLY release. That map is frozen, and the second recipient is then optimised under F_joint.
- **JOINT:** starts from the task-only, local, both sequential and joint-greedy policies, with a five-sweep cap. Unchanged
  witnesses are kept as candidates. Dominance is asserted on the fitting objective only, recomputed from the deployed
  release.
- **Not added:** no rescue, occupation-only objective, randomisation, feedback or coordinated-move solver.
- **Compute asymmetry:** JOINT has 5 search paths and 4 witnesses; SEQ has one two-stage path. This is recorded per unit
  and not described as matched compute. The sequential arms are matched adaptations of the source's corrected
  design, not the official Taylor, Vippathalla and Coon solver (PRIOR_ART_AND_CLAIM_SCOPE.md).
- **Order and resume:** within a (seed, λ) group the fit order is LOCAL, SEQ-12, SEQ-21, then JOINT. Resume is by unit
  hash and status, and valid units are never overwritten.

## 6. Release interface and exact properties (METHOD_CARD.md)

- **Per recipient output:** a categorical token, its decoded codebook probability vector and the unchanged teacher
  decision. Token identity is a disclosure.
- **Inputs at inference:** a map reads only its own frozen teacher probabilities and decision.
- **Numerics (source settings):** smoothing, epsilon 1e-12, float64, natural logs, first-index ties and reserved
  fallbacks.
- **Decision preservation:** tested on every authorized role, with row checks and failed checks reported separately. The
  containment argument is in MATH_REVIEW.md.
- **Not implied:** these properties give no SEX AUC bound, no population MI bound and no training-data privacy.

## 7. Matched inner audit (cbp.audit; AUDIT_COMPUTE.json)

- **Attacker slate:** the source FINAL slate (logistic regression, MLP, histogram gradient boosting, defence-aware linear
  and nonlinear readers), plus exact finite-code readers with smoothing {0.5, 1, 5} for codes.
- **Fitting and selection:** fitted on AUDIT_FIT and selected on INNER_SELECTION. AUC and CE winners are selected
  separately, and orientation is fixed.
- **Pair views:** pair readers receive both interfaces, the exact tuple and both ignore-the-other banks.
- **Fallback fractions:** unseen-token and unseen-pair fallback fractions are reported.
- **SRC|U composition:** SRC|U composes every public map of the registered composition bank (27 cbp codes plus the 11
  composition-only maps, same seed) before source selection. Every composed winner is frozen into the EVALUATION_LOCK.
  Protected-code attackers never receive continuous probabilities.
- **Real-data controls:** shuffled SEX (null), within-class confidence signal, two token IDs with identical decoded
  probabilities, coalition XOR, and the rotated-signal plant. Their source pass limits are registered in `cbp.audit`
  CONTROL_LIMITS before fitting.
- **Control failures:** a failed control is an engineering investigation and makes the study technically invalid
  (LABEL_TRUTH_TABLE.json failure_scope). Plants never enter a candidate bank.

## 8. Ordinary eligibility, headroom and roles (HEADROOM_SELECTION_RULES.json)

- **Ordinary eligibility:** applied on EACH seed and EACH task.
  - Accuracy ≥ U − 0.01.
  - Log loss ≤ U + 0.01.
  - Brier ≤ U + 0.005.
  - Gain over the OSF_DEFENSE_FIT-majority constant ≥ 0.8 × U's gain.
  - Gain ≥ 0.03.
  - Exact decision preservation, using `qpc.utility.gate_record` with inclusive comparisons. A decision-preservation
    failure is a technical failure, not a shortfall.
- **Headroom (privacy nominees only):** log-loss excess ≤ 0.006 and Brier excess ≤ 0.0035 for EACH task and seed. These
  are computed from the excesses, not from qpc's 0.0075 flag. They are inner selection rules, not relaxed assessment
  thresholds and not a confidence guarantee.
- **Roles:**
  - **T\*:** the closed list DIRECT-TASK i8o64, FINE-TASK i8o64, SRC|U, CLASS-ONLY and REF|F0, under ordinary
    eligibility. RAW-J is excluded from T\* because it is privacy-trained (a disclosed correction of the qpc legacy
    list). RAW-J stays in C_global.
  - **C_rate:** DIRECT, FINE, LOCAL, SEQ-12 and SEQ-21 at every λ, under ordinary eligibility.
  - **C_global:** every nonjoint release of the bank (C_rate plus CLASS-ONLY, SRC|U, SRC|RAW-J, REF|E, REF|F and REF|F0),
    under ordinary eligibility.
  - **Comparators:** never dropped for failing headroom.
  - **P\*:** headroom-eligible privacy configurations (any family, any λ) with each recipient's inner AUC ≤ T\* + 0.005
    on each seed.
  - **J\*:** headroom-eligible JOINT configurations with each recipient's inner AUC ≤ C_rate + 0.005 AND ≤ C_global +
    0.005 on each seed.
  - **Q:** U|DIRECT-TASK|i8o64, fixed. If it is not ordinarily eligible, Q is NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE and
    its slots are DESCRIPTIVE_ONLY.
- **Ordering:**
  1. Mean INNER pair AUC.
  2. Mean summed true-label log loss.

     Keys 1 and 2 are seed means computed as (s0 + s1 + s2)/3 in seed order, then `round(·, 12)`.
  3. Mean actual token states. Continuous releases count as +∞ for ordering only and are written as null.
  4. Configuration ID.
- **Aliases:** the label names the simplest family of P\*'s alias set. The alias set is the configurations whose qpc
  pair fingerprint (the deployed maps only) equals P\*'s on all three seeds. The order is LOCAL < SEQ-12 = SEQ-21 <
  JOINT. The alias set and the ID tie-break are disclosed.
- **One global configuration:** a single family/λ configuration is used across all seeds. There is no per-seed λ and no
  assessment-based choice.
- **Fallbacks:** used when no candidate is nominable. They are ranked by:
  1. the ordinary shortfall;
  2. the headroom shortfall;
  3. the local-guard shortfall (exact formulas in the JSON);
  4. then the ordering keys.

  A fallback is DESCRIPTIVE_ONLY and can never pass. If a required guard comparator is missing and some candidate is
  eligible, the role is INVALID (MISSING_GUARD_COMPARATOR).
- **No relaxation:** the headroom rule is not relaxed if nothing passes.
- **Prespecified diagnostics:**
  - the ordinary privacy winner without headroom, both T\*-guarded and unguarded;
  - each family's headroom winner (T\*-guarded; the JOINT-family winner is not J\*);
  - the source λ 0.1 JOINT, SEQ-12 and SEQ-21 controls;
  - whether headroom changes the winner, and the pair AUC given up (mean and per seed).

## 9. Primary family (PRIMARY_FAMILY.json; cbp/family.py)

There are 37 fixed slots:

- **A:** J\* vs C_rate (P01–P11).
- **B:** J\* vs C_global (P12–P22).
- **C:** P\* vs T\* (P23–P33).
- **Q:** DIRECT-TASK i8o64 (P34–P37).

Clauses of each 11-clause comparison:

| Clauses | Statistic | Requirement |
|---|---|---|
| 1 | pair AUC(comparator) − AUC(nominee) | lower bound > 0.02 |
| 2–3 | local AUC(nominee) − AUC(comparator) | upper bound < 0.01 |
| 4–5 | accuracy − U | lower bound > −0.01 |
| 6–7 | log loss − U | upper bound < 0.01 |
| 8–9 | Brier − U | upper bound < 0.005 |
| 10–11 | accuracy − 0.8 U − 0.2 constant | lower bound > 0 |

Q uses the four confidence bounds with the same limits.

Inference settings:

- **Recovery:** SEX AUC of the inner-AUC-selected attacker, averaged over attacker seeds 0–2.
- **Endpoint:** the equal-weight mean over model seeds of the per-seed paired statistic.
- **Uncertainty:** the paired multinomial bootstrap over OSF_DEVELOPMENT_ASSESSMENT exact-record groups, with B = 1999
  and RNG seed 20261008, using the same draws for every arm and seed. SE is the sd with ddof 1, and the interval is point
  ± z·SE, with z = NormalDist().inv_cdf(1 − 0.05/74) = 3.2048452050105634 (verified in code).
- **Fixed family:** slots are kept after aliases or absent nominees, and the family is never shrunk.
- **Data-dependent role aliases:** for example C_rate == C_global or T\* == Q config. They are recorded per slot
  (`alias_of_by_role`), and prose never counts one comparison twice.
- **Conditionality:** intervals condition on the fitted artifacts and do not account for adaptive reuse.

## 10. Labels (LABEL_TRUTH_TABLE.json; cbp/family.py; tested exhaustively)

- **Clause outcomes:**
  - PASS: the registered bound clears the target.
  - NOT_ESTABLISHED_PRECISION: the point clears the target, the bound does not.
  - NOT_ESTABLISHED_POINT: the point fails, and the other bound does not exclude the target.
  - MEASURED_VIOLATION: the other bound excludes the target on the failing side.
  - INVALID: a nonfinite point or bound.
- **Claim status precedence:**
  1. A failed required control.
  2. An invalid or missing comparator.
  3. A technically failed nominee.
  4. No eligible nominee: NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE, with the selection reason ORDINARY_UTILITY_FAILURE,
     HEADROOM_SELECTION_FAILURE or LOCAL_GUARD_FAILURE.
  5. No eligible comparator: INCOMPLETE_OR_INVALID.
  6. A missing slot or no clauses: INCOMPLETE.
  7. Any INVALID clause: INCOMPLETE.
  8. All 11 clauses PASS: PASS.
  9. Otherwise NOT_ESTABLISHED, with each failing clause classified. The root cause is
     MEASURED_VIOLATION_SUPPORTED_BY_BOUND, ASSESSMENT_PRECISION_FAILURE or CLAUSE_NOT_ESTABLISHED, plus the per-kind
     detail.
- **Overall label:**
  1. A global technical failure gives INCOMPLETE_OR_INVALID.
  2. C PASS gives PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (winning family).
  3. A and B PASS gives JOINT_DEVELOPMENT_CRITERION_MET.
  4. Otherwise, any INCOMPLETE gives INCOMPLETE_OR_INVALID.
  5. Otherwise, Q PASS gives CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.
  6. Otherwise, EXPERIMENTAL_NO_ADVANTAGE.
- **Display:** every claim status is always displayed. A per-claim failure affects only that claim.
- **Winning family:** a local or sequential winner is reported as such. No joint contribution is earned unless A
  and B both pass.
- **What a C pass does not mean:** a new algorithm, joint superiority, dominance over published defences, chance-level
  privacy, a population certificate, or general-purpose representation learning.

## 11. Stages and locks

- **Stage 0:** admission (SOURCE_ADMISSION_LOCK), engineering, synthetic timing, and the predictions in PREDICTIONS.json,
  committed at 357b300.
- **Stage 1:** freeze this file and the files below, then push and remote-verify FIT_LOCK.json and
  AUDIT_AND_SELECTION_LOCK.json, BEFORE any new fit. Files frozen:
  - EXPOSURE_LEDGER.md, METHOD_CARD.md, PREDICTIONS.json, FIT_MANIFEST.json;
  - HEADROOM_SELECTION_RULES.json, PRIMARY_FAMILY.json, LABEL_TRUTH_TABLE.json;
  - the scientific code, attack definitions, selection, inference and tests.

  Locked code is checked at every stage (`check_loaded_modules`). Review tests live in the unlocked `cbp/review_tests/`,
  so review edits cannot alter locked files.
- **Stage 2:** endpoint parity, then the 48 new fits.
- **Stage 3:**
  - inner audits for every candidate and composition code;
  - SRC|U composition closure;
  - real-data controls;
  - selection;
  - independent selection verification (F) BEFORE the assessment;
  - publication of the aggregate inner tables.
- **Stage 4:** push and remote-verify EVALUATION_LOCK.json (`cbp.eval_lock`).
- **Stage 5:** one assessment opening (`cbp.assess`), then `cbp.infer`. There is no new λ, nominee, restart, threshold or
  attacker selection afterwards. Technical reruns reuse the same locked artifacts, and every attempt is recorded.
- **Stage 6:** independent replay (F, no runner imports), deployment, reports, backup and closeout.

## 12. Scored list (cbp/eval_lock.py scored_labels)

The scored list, with duplicates scored once and every role mapping kept:

- P\*, J\*, T\*, C_rate, C_global and Q, or their registered fallbacks;
- SRC|U and CLASS-ONLY;
- JOINT, SEQ-12 and SEQ-21 at λ 0.1;
- the ordinary privacy winner without headroom;
- each family's headroom winner or fallback;
- SRC|RAW-J β0.3, REF|F (FARE), REF|F0 and REF|E (LEACE).

The full λ curve is an INNER figure only. The assessment never scores the whole grid.

## 13. Diagnostics and figures

- **Per code:**
  - full and occupied alphabets;
  - states per predicted class;
  - entropy, support, singletons and unseen-pair fraction;
  - teacher KL vs true-label LL and Brier;
  - fitting MI with permutation null;
  - objective terms, merges, exchanges, sweeps and unresolved optima;
  - deployed-release reconstruction;
  - every seed.

  These go in OPTIMIZATION_RECEIPTS.json and CLASS_PRESERVATION.json.
- **Figures:**
  1. The inner λ curve, with the headroom and allowance lines.
  2. Assessment pair recovery vs occupation LL excess.
  3. Individual vs pair recovery.
  4. Token counts vs recovery.
- **Bottleneck:** reported from the per-kind clause outcomes.

## 14. Independent verification (F)

F re-implements the protocol from these definitions without importing the runner's helpers. F reproduces:

- admission;
- bindings;
- deployed outputs;
- decision preservation;
- KL, MI and counts;
- eligibility, roles, aliases and fallbacks;
- every primary point, SE, bound and label;
- headline numbers;
- budget;
- backups.

Tolerances are registered, verdicts must agree exactly, and identities and hashes must be exact. The defects tested are
those listed in prompt section 14.

## 15. Release, backup, closeout

- **Commands:** `python -m cbp.deploy`, `cbp.run`, `cbp.infer` and `cbp.closeout`. QUICKSTART.md lists only commands
  that were actually run.
- **Backup:** a versioned same-device copy, with hashes and restore from the copy alone. Off-device backup is reported as
  pending if the drive is absent.
- **Public files:** aggregate evidence only. A privacy scan runs before every push.
- **Shutdown:** only this study's owned workers are stopped.

## 16. Decision rules after the study (prompt section 17)

- **C passes:** package the winner and recommend an independently planned confirmation study.
- **Sequential matches joint:** prefer the simpler story.
- **Headroom removes the benefit:** close this interpolation line, with no further λ tweak on these rows.
- **Technical blocker:** report the exact hash, stage, counts and the smallest remaining action.

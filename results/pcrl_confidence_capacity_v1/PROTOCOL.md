# Protocol: confidence capacity and privacy (qpc)

Registered before Stage A. Sections 7 and 11 are locked by STAGE_A_LOCK; the rest is locked progressively by the named locks of §13. Any change after a lock is a dated, pushed amendment, made before the affected rerun.

| Item | Value |
|---|---|
| Branch | `research/pcrl-confidence-capacity-v1` |
| Source evidence | dpc @ `0a7b05a52746544213742f50efd0a48167efffb1` |
| Teacher provenance | osf @ `925e0fddfcb666116c6179575339728a324ed78e` |
| Code | `qpc/` |
| Public evidence | `results/pcrl_confidence_capacity_v1/` |
| Private store | `<PRIVATE_CACHE>/qpc_v1/` |
| Start | 2026-10-06T03:48:42Z |

## 1. Question and scope

The completed dpc study found:
- class-preserving finite codes kept every decision and lowered SEX recovery;
- but every registered code (m ≤ 8 states per class) lost too much occupation confidence;
- the source review recorded that most fine clustering runs hit their 20-round cap without converging.

This study asks two bounded questions:
1. **Stage A.** Were the confidence failures caused partly by unfinished clustering or by too few occupation states, under the UNCHANGED confidence contract?
2. **Stage B** (only if a capacity passes). At feasible capacities, does privacy training — local, sequential or joint — beat equally capable ordinary compression?

This is a bounded follow-up to dpc. It is not a continuation of the closed critic-refresh, feedback-controller or gradient-strength studies. It promises no algorithmic breakthrough and no fresh confirmation.

**Scoped reading of the source.** These items come from dpc's RESEARCH_DECISION.md and VALIDATION.md and are not reclassified:
- "Every tested code failed" is supported.
- "Every finite code must lose at least 0.02 nats" is not supported.
- Unfinished clustering is an optimisation limitation, not proof that finishing it solves the gap.
- The ~0.74 decision-only AUC is measured recovery by the declared attackers. The structural fact is only that a recipient can recover the decision from a decision-preserving code.
- No joint-over-sequential advantage was established.

**The dpc label caveat is not inherited.** dpc kept the label EXPERIMENTAL_NO_ADVANTAGE although a strict reading of its protocol §8 (C_match NO_FEASIBLE_CONTROL) would give INCOMPLETE_OR_INVALID (dpc RESEARCH_DECISION.md, "Matched-control coverage failure (claim A)"). This protocol removes that ambiguity: §11 and `LABEL_TRUTH_TABLE.json` define every status, and `qpc/family.py` implements exactly that table, with exhaustive tests.

## 2. Exposure statement (registered verbatim)

"The design is motivated by previously opened Adult development results, including the completed decision-preserving compression study. Every assessment row has been used historically. This is an exploratory, locked development comparison. Its nominal intervals condition on fitted artifacts and do not correct for the adaptive research history. It is not fresh confirmation or a population privacy guarantee."

## 3. Data and roles (unchanged from osf/dpc)

**Input.** `adult_jcv.npz`, sha256 `e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12`. The 83 permitted columns are pinned. Tasks are binary income and six-class occupation_group; the protected attribute is SEX. Proxies are preserved as in the source.

| Role | Rows | Use |
|---|---:|---|
| OSF_DEFENSE_FIT | 15,434 | Partitions, prototypes, privacy fitting |
| HEAD_VALIDATION | 1,500 | Historical head selection only |
| AUDIT_FIT | 6,065 | Attacker fitting |
| INNER_SELECTION | 2,235 | All selection, including the Stage A capacity gate |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) | Masked until the pushed EVALUATION_LOCK; opened once |

Counts, group hashes and non-overlap are verified (`SOURCE_ADMISSION.json`, `ROLE_MANIFEST.json`). No group is removed and no role is pooled after results are seen.

## 4. Teachers and references (admission only, no new fits)

- **U, seeds 0–2.** Frozen encoders and deployed heads, restored from the hash-pinned admitted artifacts. Scores, probabilities and decisions are rebuilt by `qpc.admit` and must be bitwise equal to the admitted dpc teacher outputs before any code is fitted.
- **RAW-J β 0.3.** Continuous outputs, as an incumbent comparator only.
- **References:** official FARE (F), no-fairness FARE (F0) and LEACE (E), score-only, admitted by verified copy when their source provenance remains valid.

No RAW-J encoder, task head, recalibration or supervised probability prototype is fitted.

## 5. Release interface and invariants

Each recipient i receives a categorical token, the public probability vector decoded from that token, and the teacher's decision. The pair receives both interfaces aligned. Token identity is audited in full: two tokens with identical decoded vectors are still distinct disclosures. No features, logits, continuous teacher probabilities, fine-cell IDs, distances, residuals or debug IDs accompany a release.

**Construction rules:**
- Partitions are made only within the teacher-predicted class.
- Prototype: smooth(mean_p, d) = (mean_p + ε·1 + ε·e_d)/(1 + (K+1)ε), with ε = 1e-12.
- Arithmetic: float64, natural logs, first-index ties, log-loss clip 1e-12.
- A reserved fallback cell exists for any class absent from the fitting rows.

**The map reads only its own teacher's probabilities and predicted class.** It never reads true labels, SEX, the other recipient's score, person ID, role or row index at inference.

**Invariants, checked and reported:**
- decision preservation on fitting, selection and assessment rows;
- normalisation and strict argmax of every prototype;
- identical assignment after save/restore;
- no hidden fine IDs;
- training statistics equal the deployed assignment's statistics on the training rows;
- degenerate and absent classes reported. Preserving decisions cannot repair the teacher's class-5 recall of 0.

Standard data-processing and decision-containment facts are stated with their assumptions in MATH_REVIEW.md. They are not new theorems, attribute-specific guarantees or training-data privacy.

## 6. Pre-fit predictions

`PREDICTIONS.json` is committed and pushed before STAGE_A_LOCK and never revised.

## 7. Stage A: convergence versus capacity (first scientific gate)

Stage A runs before any privacy-trained code is fitted. The k-means rules (starts, KL k-means++, convergence tolerance, coherence, empty cells, fallbacks) are registered in METHOD_CARD.md §1–5 and §7 and locked by STAGE_A_LOCK.

**Numeric constants:**
- relative tolerance RTOL = 1e-9 on the class-total fitted KL, with patience 3 successive passes;
- cap 200 rounds (20 for the A1 source reproduction);
- start tie threshold 1e-12·max(|J|, 1), with ties going to the earlier start;
- k-means++ generator `numpy.random.default_rng([seed, K, c])` with seeds 20261006 and 20261007;
- KL clip in [−1e-12, 0) applied only to k-means++ sampling weights.

Stop reasons are reported separately everywhere: assignment fixed point, relative tolerance, and cap (non-converged).

**A1, historical convergence diagnostic** (per U seed).
1. Reproduce the source DIRECT-TASK (income 8, occupation 8) code with its source initialisation and 20-round rule. Parity against the admitted dpc `pol__s{k}__U_DIRECT-TASK_m8` release is REQUIRED; a mismatch is an engineering blocker, not evidence.
2. Refit the same initialisation, rate, KL objective, smoothing and assignment rules with at most 200 rounds and the registered convergence rule.

Reported: fit objective, INNER_SELECTION confidence, convergence and work (`CONVERGENCE_DIAGNOSTIC.csv`). The old code is not called incorrect for stopping at its registered cap.

**A2, better task-only codebooks.**
- **Bank:** class-preserving DIRECT-TASK k-means directly on OSF_DEFENSE_FIT rows, for income m1 ∈ {4, 8} and occupation m2 ∈ {8, 16, 32, 64} per predicted class, seeds 0–2. That is 8 global rate configurations and 24 mapping-pair units.
- **Caching:** per-recipient fits are cached and shared across rate pairs. The A1 200-round fixed-initialisation fit is A2's source-start receipt at (8, 8), not an extra nominal unit.
- **Starts:** at most 200 rounds and three fixed starts:
  1. the source deterministic initialisation;
  2. KL k-means++ with seed 20261006;
  3. KL k-means++ with seed 20261007.
- **Winner:** the lowest OSF_DEFENSE_FIT KL distortion per class. No task labels or SEX choose centroids, starts or assignments.
- **Convergence:**
  - stop when assignments are unchanged, or at the registered relative-objective tolerance for three successive rounds after a final consistent assignment/update;
  - cap at 200 and label non-convergence if reached;
  - keep the best fully coherent iterate;
  - apply identical rules at every rate.
- **Reported:** actual states, empty cells, cell populations and convergence. There is no support floor. m2 = 64 means up to 64 real cells per class, limited only by distinct vectors and rows.

**A3, inner confidence only.** No attackers. True-label utility on INNER_SELECTION; the outcome is not a fit-distortion certificate.
- **Eligibility, unchanged:** on EACH seed, for EACH task:
  - accuracy ≥ U − 0.01;
  - log loss ≤ U + 0.01 nats;
  - Brier ≤ U + 0.005;
  - gain over the OSF_DEFENSE_FIT-majority constant ≥ 0.8 × U's gain;
  - gain ≥ 0.03;
  - exact teacher decision preservation.
- **Identities:** class preservation makes several accuracy comparisons identities. They stay in the receipts, and no confidence limit is dropped.
- **Headroom preference** (rate ordering only): log loss ≤ U + 0.0075 and Brier ≤ U + 0.0035.

**A4, conditional decision** (`qpc/gate.py`, locked here).
- **If no global rate qualifies on all seeds:** label CAPACITY_GATE_NOT_MET; Stage B is not run; assessment labels are never opened. The best-shortfall code (smallest worst-seed normalised excess, then fewer states, then configuration ID) is packaged as runnable and INELIGIBLE. We state that this construction did not pass the inner gate, not that compression is impossible.
- **Otherwise, select up to TWO eligible rates, the same on every seed:**
  - order by key = (mean actual total token states over seeds, worst-seed occupation log loss, worst-seed income log loss, configuration ID);
  - H = eligible configurations meeting both headroom preferences on all seeds, ordered by key;
  - E = the remaining eligible configurations, ordered by key;
  - selected = (H + E)[:2].
- **Recording:** the decision is written to `CAPACITY_GATE.json`, and STAGE_B_LOCK is pushed with the selected rates. Stage B continues automatically.

Normalised excess of one seed = max over tasks of (log-loss excess / 0.01, Brier excess / 0.005).

## 8. Stage B: same input, same capacity, privacy comparison

**Bank.**
- The selected U rates, all three seeds, λ ∈ {0.01, 0.1, 1}.
- Before any real fit, synthetic timing (`TIMING.json`) may reduce λ to {0.01, 0.1}, or reduce Stage B to the first eligible rate. That choice is recorded in STAGE_B_LOCK and never made on scientific results.

**Fine partitions.** Task-only, with income ≤ 32 and occupation ≤ 128 states per predicted class. Same starts and convergence rules as Stage A. OSF_DEFENSE_FIT probabilities only; fallbacks and support receipts kept. The Stage A DIRECT-TASK controls are unchanged. DIRECT-TASK may lie outside the fine-partition family and can be stronger; it remains a control.

**Families and objectives.**
- Families: FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT, over fine cells.
- Definitions: D_i = mean fitting KL(p_i ‖ decoded); I1, I2 and I12 = plug-in MI of SEX with the complete released code identities on OSF_DEFENSE_FIT.
- F_task = D1 + D2
- F_local = D1 + D2 + λ(I1 + I2)/2
- F_joint = D1 + D2 + λ((I1 + I2)/2 + I12)

Objective definitions, sufficient statistics, restarts and optimisation budget are the same across arms at each λ.

**Sequential correction.** The other recipient's predicted class is always disclosed.
- When optimising the first recipient, the other map is its CLASS-ONLY release, and the first stage is evaluated under the actual F_joint with that counterpart.
- The first map is then frozen and the second optimised under F_joint.
- dpc's D + 1.5λI stage-one objective is NOT reused. The correction is reported.

**Search.**
- Greedy same-class merges plus exact single-fine-cell exchanges, on cached sufficient statistics.
- At most the registered state cap, including objective-improving extra merges below it. Actual alphabets are reported.
- JOINT starts: the fine-task, local, both sequential and joint-greedy policies, each refined with at most 5 full sweeps. Fitted F_joint must be no worse than the unchanged witnesses. Unresolved local optima are reported.
- DIRECT-TASK need not be contained in this family, so no dominance over it is claimed.

**Units, with two rates and three λ:**

| Units | Count |
|---|---:|
| Privacy mapping pairs (2 rates × 3 λ × 4 families × 3 seeds) | 72 |
| FINE-TASK | 6 |
| Stage A DIRECT-TASK controls | 24 |
| A1 reproductions | 3 |
| **Total logical mapping-pair units** (before aliases, references, auxiliary fine-partition jobs and the 3 CLASS-ONLY releases used as the decisions-alone control) | **105** |

Per-class restarts are counted separately. No neural encoder, randomised token mechanism, feedback controller or critic refresh belongs to this bank.

## 9. Sparse cells and fitted MI

**Reported for every code:**
- occupied local and pair alphabets;
- counts per code and pair;
- fractions of singleton and unseen pair cells;
- fitted MI, a permutation-null MI at the same alphabet, and held-out attack results, each separately;
- the λ-weighted privacy term versus distortion;
- whether improvements mostly track alphabet reduction.

**Permutation diagnostic.** Uses OSF_DEFENSE_FIT SEX only, with a fixed permutation list and seed. It diagnoses MI bias and never changes objectives, λ or nominees.

**Readers and coverage.** Unknown pairs use the prespecified fallback. Neural and tree attackers also receive decoded probabilities. Nominee attack coverage is reported. Any unestimable metric or technical coverage failure blocks its dependent claim.

## 10. Attacks, controls and public-map closure (`qpc/audit.py`; locked in AUDIT_AND_SELECTION_LOCK)

**Slate and selection.**
- The same source FINAL slate for every primary release: pinned LR/MLP/HGB grids, the source defence-aware linear and nonlinear readers, and finite-code conditional readers at smoothing {0.5, 1, 5}.
- Fit on AUDIT_FIT; select on INNER_SELECTION. AUC and CE selection are separate. Orientation is fixed; there is no flipping and no clamping.

**What readers see.**
- Individual readers: token identity (categorical) plus decoded probability and decision.
- Pair readers: both full interfaces and exact tuples, with both ignore-recipient banks.
- Unseen local tokens use the pinned fitting SEX prior; unseen pairs use the prespecified, validation-selected fallback. Use is reported on fit, selection and assessment rows.

**Public-map closure.**
- U continuous-source attack banks include composition with EVERY fitted code of this study, through the cached code-reader candidates.
- They are assembled before any strongest control or nominee is selected.
- Protected-code attackers never read continuous probabilities.
- Final source audits include the full-slate readers of the locked selected codes, and every composed reader selected as a source-bank winner.

**Controls:**
- U continuous probabilities and decisions alone;
- the RAW-J β 0.3 score-only release;
- all eligible Stage A DIRECT-TASK rates;
- FINE-TASK at the selected rates;
- LOCAL and both sequential orders;
- admitted FARE, F0 and LEACE score-only outputs.

All face the same slate and utility contract. FARE's occupation loss is not excused by its low recovery. A clean original output appended to a code is a different contract and is not used.

**Real-data controls, run before assessment, with explicit pass rules in `qpc/audit.py`:**
- SEX shuffled within authorised training roles;
- a within-predicted-class confidence leak;
- two token IDs sharing one decoded vector but carrying different information;
- a coalition XOR plant;
- a small rotated signal.

A failed positive control triggers technical review and an amendment, never a reading of low leakage.

## 11. Inner selection, nominees and validity (`qpc/select.py`, `qpc/family.py`, `LABEL_TRUTH_TABLE.json`)

**General rules.**
- Every selection uses INNER_SELECTION only.
- Configurations are frozen across seeds.
- An ELIGIBLE release passes every utility gate of §7 A3 on every seed. For continuous sources and references, decision preservation is with respect to their own released decisions.
- U continuous is eligible by construction.
- No compact code is eligible because its seed mean passes.

**Ordering for C_rate, C_global, T*, J* and P*:**
1. lower mean inner pair AUC;
2. lower mean summed true-label task log loss;
3. fewer actual total states (continuous releases count as infinite);
4. configuration ID.

**Selections:**
- **Q\*:** the eligible Stage A DIRECT-TASK configuration with the smallest worst-seed normalised excess, then fewer actual states, then configuration ID.
- **C_rate(J):** the strongest eligible nonjoint control at J's U rate caps (DIRECT-TASK, FINE-TASK, LOCAL, SEQ-12, SEQ-21, at any λ).
- **C_global:** the strongest eligible nonjoint release across all rates, sources, references and CLASS-ONLY. JOINT is excluded.
- **T\*:** the strongest eligible privacy-untrained release across DIRECT-TASK, FINE-TASK, continuous sources (U, RAW-J), CLASS-ONLY and F0 (F0 explicitly included).
- **J\*:** an eligible JOINT configuration whose individual inner AUC (v1 and v2) is ≤ C_rate(J) + 0.005 and ≤ C_global + 0.005 for each recipient and seed. The best by the ordering is chosen.
- **P\*:** an eligible LOCAL/SEQ-12/SEQ-21/JOINT configuration whose individual inner AUC is ≤ T* + 0.005 for each recipient and seed. The best by the ordering is chosen; the winning family is recorded.

**Status rules, fixed before fitting.** These agree exactly with `qpc/family.py`; see `LABEL_TRUTH_TABLE.json`.

| Situation | Status |
|---|---|
| No eligible nominee | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE. Cannot pass through a fallback; fallback rows are DESCRIPTIVE_ONLY |
| Missing or technically uncomputable comparator | INVALID_COMPARATOR |
| No eligible same-rate comparator | NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR. NOT a valid head-to-head negative; it counts as missing comparator coverage |
| Nonfinite primary statistic or unestimable metric | INVALID |
| All 11 clauses pass, with an eligible nominee and comparator | PASS |
| Otherwise | NOT_ESTABLISHED, a completed valid negative |

- **Precedence:** technical failure, then no eligible nominee, then no eligible comparator, then INVALID clauses, then PASS / NOT_ESTABLISHED.
- **Interpretation:** a completed confidence failure is not a software failure, and a missing scientific comparison is not a completed competitive comparison.
- **Fallbacks:** when a nominee is absent, its deterministic minimum-shortfall fallback is scored DESCRIPTIVE_ONLY. Among the role's candidate set, take the smallest worst-seed normalised confidence excess (0 if eligible), then the smallest guard shortfall (max over recipients and seeds of local AUC − (bound + 0.005), floored at 0), then the ordering above.

**Labels, in precedence order:**
1. Stage A incomplete or invalid → INCOMPLETE_OR_INVALID.
2. Gate not met → CAPACITY_GATE_NOT_MET.
3. Any required technical validity failed (a failed positive control without amendment, a verification FAIL on a primary quantity, a missing locked unit) → INCOMPLETE_OR_INVALID.
4. A and B both PASS → JOINT_DEVELOPMENT_CRITERION_MET (plus PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET if C also passes).
5. C PASS → PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (winning family named).
6. No favourable label, and any claim in {NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR, INVALID_COMPARATOR, INVALID_NOMINEE, INVALID} or Q* INVALID → INCOMPLETE_OR_INVALID, listing the missing items.
7. Q* PASS → CONFIDENCE_FEASIBILITY_ESTABLISHED.
8. Otherwise → EXPERIMENTAL_NO_ADVANTAGE.

A validly PASSing claim keeps its favourable label even if another claim lacks coverage; that gap is always listed in the missing items and the per-claim table. A missing comparator is never a completed negative.

**Guard-blocked nominees and the J* cell.** TECHNICAL_FAILURE of a nominee also covers an otherwise eligible nominee blocked only because one of its guard comparators (C_rate(J) or C_global for J*; T* for P*) is not a NOMINEE. Guards are never dropped. Claim A's C_rate is resolved at the J* nominee cell, else at the J* descriptive-fallback cell. When J* has neither, C_rate is INVALID_COMPARATOR ("no J* cell"), with the J* failure named as the root cause.

These two clarifications (math review SEL-R1 and SEL-C2) were made at 04:29:47Z (commit 9090348). That was after STAGE_A_LOCK and after the Stage A gate results were known, but before any SEX-recovery, inner-audit, selection or assessment data existed. They are recorded in STAGE_B_LOCK. Because SEL-C2 changes a registered rule, the final label is reported under BOTH the STAGE_A_LOCK rule (any coverage gap gives INCOMPLETE_OR_INVALID before favourable labels) and the amended rule. If the two differ, both are shown, and the amended one is marked as a post-lock amendment.

The per-claim table is always published. All favourable labels refer to exploratory development only.

## 12. Primary family (`PRIMARY_FAMILY.json`, `qpc/family.py`)

**Slots.** 37 fixed slots:
- claim A, J* vs C_rate (P01–P11);
- claim B, J* vs C_global (P12–P22);
- claim C, P* vs T* (P23–P33);
- Q* confidence feasibility (P34–P37).

**Method-claim clauses:**

| Clause | Statistic | Pass rule |
|---|---|---|
| 1 | pair AUC(comparator) − AUC(nominee) | lower bound > 0.02 |
| 2–3 | local AUC(nominee) − AUC(comparator), each recipient | upper bound < 0.01 |
| 4–5 | accuracy(nominee) − accuracy(U), each task | lower bound > −0.01 |
| 6–7 | log loss(nominee) − log loss(U), each task | upper bound < 0.01 |
| 8–9 | Brier(nominee) − Brier(U), each task | upper bound < 0.005 |
| 10–11 | accuracy(nominee) − 0.8 accuracy(U) − 0.2 accuracy(constant), each task | lower bound > 0 |

**Q\* clauses:** income and occupation log-loss excess, each upper bound < 0.01; income and occupation Brier excess, each upper bound < 0.005.

**Inference.**
- z = NormalDist().inv_cdf(1 − 0.05/74) = 3.2048452050105634.
- 1,999 paired exact-record-group bootstrap replicates, seed 20261007, with the same draws for every arm and seed.
- Model seeds are averaged by the source convention and are not treated as independent populations.
- Nonfinite primary statistic → INVALID; no replicate is dropped.
- Bounds are nominal and conditional on the fitted artifacts and the reused data.

**Claim requirements.**
- A joint advantage requires both A and B with valid nominees and comparators.
- A privacy-compression advantage requires C, with the winning family named.
- Passing only against continuous U does not establish a joint contribution.
- Winning below 0.02 is a small measured effect, not the registered advantage.

## 13. Locks and the single exploratory assessment opening

**Locks.** Each is pushed and remote-verified before the stages it governs; `qpc.lock` also refuses any unlocked or changed study module that a stage loads.

| Lock | Contents |
|---|---|
| SOURCE_ADMISSION_LOCK | source pins, hashes, roles, raw-input custody, admission code |
| STAGE_A_LOCK | k-means code, starts, convergence, rate bank, gate and rate-selection rules, protocol, predictions |
| STAGE_B_LOCK | selected rates, full/reduced bank (from synthetic timing), search code |
| AUDIT_AND_SELECTION_LOCK | attacks, source composition, controls, nominees, family, inference |
| EVALUATION_LOCK | exact release, prototype, teacher, attacker, role and code hashes, and the scored list |

**Scored list, frozen in EVALUATION_LOCK before the opening:**
- the locked nominees and comparators (J*, C_rate(J*), C_global, P*, T*), or their fixed minimum-shortfall fallbacks;
- Q*;
- continuous U, the RAW-J score release and the decisions-alone (CLASS-ONLY) release;
- the task-only capacity curve: all 8 Stage A DIRECT-TASK rates;
- FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT at the J* cell (or the J* fallback's cell);
- FARE, F0 and LEACE.

**Opening rules.** No new winner is scored after the table is seen. Technical retries rescore the same locked units. If Stage A fails, the assessment is not opened.

## 14. Reports

Reports must explain:
- the convergence contribution at a fixed rate;
- the occupation capacity contribution;
- income versus occupation costs;
- token caps versus emitted states and entropy;
- true-label confidence versus teacher KL;
- ordinary versus privacy-trained compression;
- joint versus local and both sequential orders;
- the strongest favourable AND the strongest adverse comparison;
- absolute recovery, including decision leakage;
- whether the inner feasibility gate and the assessment confidence bounds agree.

All displayed numbers come from checked aggregate arrays. No empirical AUC difference is called an MI bound, no in-sample MI optimum is called a population guarantee, and no rate increase is called algorithmic novelty.

## 15. Verification, packaging, custody and budget

- **Verification:** an independent verifier (separate code; no qpc imports) reproduces the items in prompt §15.
- **Defects after assessment:** an arithmetic or reporting defect keeps the original output, corrects the same locked estimand, discloses both, and never reselects a nominee.
- **Packages:**
  - U continuous;
  - Q*, if eligible;
  - the best inner-selected eligible privacy code, if one exists;
  - otherwise the best-shortfall code, marked INELIGIBLE.
- **Deployment:** `python -m qpc.deploy` takes the exact 83 pinned columns with teacher and schema binding, and outputs only tokens, decoded probabilities and decisions.
- **Backup:** to the drive if it is identified by content; otherwise a verified versioned same-device copy, with pending commands.
- **Budget:** 10 h elapsed, 20 CPU-h, at most two heavy processes in total (the `qpc/sema.py` semaphore; verifier included), 8 GiB, ≥ 5 GiB free, $0 cloud. The final 2 h and 4 CPU-h are reserved. Measured and estimated CPU are tracked separately.

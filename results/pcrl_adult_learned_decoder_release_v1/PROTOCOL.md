# PCRL Adult learned decoder and constrained release (lra) — registered protocol

| Item | Value |
|---|---|
| Branch | `research/pcrl-adult-learned-decoder-release-v1` |
| Started from | lcr final tip `091afc2` (lcr evidence `418e529`); cbp `7f3ec67`; qpc evidence `9dd06da` / tip `d0c8a45` |
| Package | `lra/` (port of `lcr/`, PORT_LOG.md) |
| Public results | `results/pcrl_adult_learned_decoder_release_v1/` |
| Private store | `<PRIVATE_CACHE>/lra_v1/` |
| Start | 2026-10-07T04:03:43Z |

**Locks.** Four separate locks are used: SOURCE_ADMISSION_LOCK, CORRECTNESS_LOCK, SCIENCE_LOCK and EVALUATION_LOCK. Each
holds this protocol and the actual lra code, and each is pushed and verified before the stage it governs.

**Disagreements.** A code/text disagreement is a DEFECT to disclose and resolve. Code never silently overrides a
registered scientific rule.

## 1. Questions and the predecessor

1. Can learning better probabilities let a more private code preserve useful predictions and confidence on Adult?
2. Separately: does the new constrained assignment search add anything beyond recalibrating existing private maps,
   ordinary compression and equally capable weighted or sequential methods?

**Comparison scope.** The same probability-learning procedure is available to every control. If a calibrated old map is
the best useful release, it is delivered and the result is attributed to it. No favourable outcome is promised. This is
a supervised fixed-task output-release study on historically used development rows. It is not fresh confirmation, not
general-purpose representation learning, not a population privacy guarantee and not an algorithmic novelty claim.

**The predecessor.** lcr's mechanism gate could not trigger (PREDECESSOR_GATE_DIAGNOSIS.md; SOURCE_FIXTURE_GATE.json).
- On F2–F4, CLASS|D1 met the fitting budgets with zero pair information. Every class-preserving release contains the
  classes, so none could reveal 0.01 nats less.
- On F1, no task-only comparator was feasible.
- The historical label stays MECHANISM_GATE_NOT_MET, and lcr produced no Adult result.

**Correction carried forward.** Giving decisions SEX content does NOT make the old "less information than CLASS" gate
attainable whenever CLASS is utility-feasible. This study therefore does not reuse that synthetic superiority gate. Its
launch gate is correctness-only (§9).

## 2. The mechanism and its limits

- **A decoder change cannot erase information.** Changing the probability decoder of an UNCHANGED token removes no
  information from that token. The recipient receives the complete token identity, and a public deterministic decoder
  is a function of it. Bayes information of the complete interface is therefore unchanged. A learned decoder (D1) can
  improve utility and make a more private partition usable. A finite attacker's score may change, but that is never
  credited as information removal without a changed partition. Fixed-token decoder ablations and full-token attacks are
  required throughout.
- **Calibrated-teacher null.** Suppose the teacher probability is the true conditional label distribution for its
  permitted input. Then its conditional mean within a fixed token minimises expected log loss and Brier, and no
  population proper-loss improvement is available from replacing that mean. D1 is useful only because the real frozen
  heads are imperfect (MATH_REVIEW.md).

## 3. Exposure, roles and labels (EXPOSURE_LEDGER.md)

- **Exposure statement.** Registered verbatim in EXPOSURE_LEDGER.md and in every lock.
- **Roles (unchanged).**

  | Role | Rows |
  |---|---|
  | OSF_DEFENSE_FIT | 15,434 |
  | HEAD_VALIDATION | 1,500 (historical only; not repurposed) |
  | AUDIT_FIT | 6,065 |
  | INNER_SELECTION | 2,235 |
  | OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) |

- **Input.** `adult_jcv.npz` (sha256 e0d9e54a…2f12): the 83 permitted columns in pinned order, with duplicate rules and
  group assignments.
- **Label use (a material change).**
  - Task labels and SEX on OSF_DEFENSE_FIT are used for the NEW supervised fitting: D1 decoders and coarse assignments.
    The fine partitions stay fixed and label-blind.
  - AUDIT_FIT trains attackers, and INNER_SELECTION selects.
  - Assessment labels stay sealed until the pushed EVALUATION_LOCK; only `lra.assess` unseals.
- **Cohorts.** No new state or year is opened.

## 4. Budget and resources

- **Time and CPU.**
  - 10 h elapsed from 04:03:43Z (ceiling 2026-10-07T14:03:43Z) and 20 CPU-h in total.
  - The final 2 h (from 12:03:43Z) and 4 CPU-h are reserved for verification and closeout.
- **Concurrency.** At most 2 heavy processes, through one shared semaphore (`lra.sema`; SEMA_LOG.jsonl and
  COMPUTE_LEDGER.jsonl). One thread per process.
- **Memory and disk.** At most 8 GiB aggregate; at least 5 GiB free disk.
- **Cloud.** $0; no AWS, no new data.
- **Measurement.** Estimated and measured CPU are recorded separately.
- **Scheduling.** The full bank is timed on realistic synthetic shapes (TIMING.json, AUDIT_COMPUTE.json). There is no
  result-based pruning. If the mandatory bank cannot fit, scheduling and caching are fixed first. If it still cannot
  fit, the outcome is a pre-fit budget blocker (INCOMPLETE_NOT_RUN).

## 5. Source admission (SOURCE_ADMISSION_LOCK; SOURCE_ADMISSION.json)

- **Admitted.** The lcr admission's 189 source/custody units, by verified copy from the cbp store:
  - 6 teachers;
  - 9 references (official LEACE, FARE, F0);
  - 3 fine partitions;
  - 81 D0 codes: DIRECT-TASK, FINE-TASK, CLASS-ONLY, and the full six-weight LOCAL/SEQ-12/SEQ-21/JOINT bank;
  - 90 cbp inner audits, as custody only.
- **Hash checks.** Every file matches all four of:
  - its cbp COMPLETE.json;
  - the cbp same-device copy SHA256SUMS;
  - cbp's EVALUATION_LOCK, for scored units;
  - the lcr SOURCE_ADMISSION.json at 091afc2.
- **Parity.** Teacher forward passes and release re-encodes are bitwise: 6/6 teachers and 81/81 codes, ADMITTED at
  04:11:54Z.
- **Scope.** The 189 custody units are not the scientific fit count. No new encoder, head or feature extractor is
  fitted. An unchanged teacher forward pass is inference, not a defence fit.
- **Fine partitions.** They stay label-blind. DIRECT-TASK may lie outside the fine-cell family and is kept as a strong
  external control, with no containment claimed.

## 6. Common learned decoder D1 (lra/decoder.py; METHOD_CARD.md; DECODER_CERTIFICATES.json)

**Notation.** For a token t of recipient i:
- n_t is the fitting count;
- y_t are the fitting true-label class counts;
- pbar_t is the mean teacher probability;
- d_t is the common teacher-predicted class.

**Optimisation.**
- Optimise a base vector u in the class-dominant simplex: sum u = 1, u ≥ 0, u[d_t] ≥ u[k].
- Evaluate the ACTUAL released vector q = (u + ε·1 + ε·e_d)/(1 + (K+1)ε), with ε = 1e-12.
- Use float64, natural logs, source ties and log-loss clipping.
- Objective per token, with κ = 32 teacher pseudo-observations:
  - Σ_y y_t[y]·(−log q[y]);
  - + 0.5·Σ_rows ||q − onehot(Y)||²;
  - + κ·KL(pbar_t || q).
- No tuning of κ, the Brier coefficient or ε, and no SEX in this objective.
- The problem is convex for a fixed token. It is solved by a deterministic exact method on cached sufficient
  statistics.

**Certificates.**
- Each solve reports the primal residual, stationarity/KKT residual, active class constraints, convergence and
  objective, checked against an independent trusted solve (role E).
- A deterministic projection is applied to the class-dominant simplex for tiny residuals. The actual q is then
  re-evaluated.
- A released q whose argmax differs from d_t is refused.

**Fallbacks.** Tokens without fitting rows, and absent-class fallbacks, keep the pinned D0 vector. No labels are
invented.

**Naming.** D0 (the mean-teacher codebook) and D1 are distinct artifacts. An admitted D0 map is never overwritten. If
D1 violates a budget, the map is "infeasible under this registered decoder", not mathematically infeasible.

**CLASS|D1 (a registered addition).** The learned decoder is also applied to the unchanged decision-only map,
U|CLASS|i1o1|D1 (prompt §10: CLASS stays in the comparator pool, and its utility is measured after D1). It is a
fixed-map D1 code: audited, in the composition bank and in the T\* pool. This gives 81 D1 fixed-map units against
the prompt's 78, 84 codes per seed, 89 configurations and 267 inner units against the prompt's 88/264.

**Consequence.** If any CLASS variant is eligible, it sits at the decision floor, since I(S; P\*) ≥ I(S; CLASS) for
every class-preserving P\*. Claim A could then pass only through finite-reader behaviour, and is reported that way.

## 7. Fitting budgets and the new mapper (lra/mapper.py; SEARCH_RULES.json)

**Fitting quantities on OSF_DEFENSE_FIT.**
- L_i: actual true-label log loss.
- B_i: source-convention multiclass Brier.
- I_i and I12: plug-in MI of SEX with the full token identities.

**One budget profile, all seeds.**
- L_i ≤ L_i(U) + 0.005 nats.
- B_i ≤ B_i(U) + 0.003.
- Decision preservation.
- State caps of 8/64 per predicted class.

These FITTING budgets are a design choice, not a population certificate, and do not replace the inner/assessment rules.

**Registered consequence (inherited lcr N-9).** There is no restoration phase.
- The U teacher heads were fitted on OSF_DEFENSE_FIT, so L_i(U) and B_i(U) on the fitting rows are in-sample.
- If C-TASK exceeds a fitting budget on some seed, then on that seed:
  - K-LOCAL, whose only start is C-TASK, is INFEASIBLE by construction;
  - K-SEQ and K-JOINT depend on feasible source starts and witnesses.
- C-TASK's deployed budget slack is read right after the ctask stage. It is a pre-registered diagnostic, never a tuning
  point.
- Selection treats an infeasible constrained fit as CONSTRAINED_FIT_INFEASIBLE (SELECTION_RULES.json).

**Search design choices registered in SEARCH_RULES.json (lra.mapper.rules()).**
- "Nearest" target: KL(fine-cell mean teacher probability ‖ the target token's current released q).
- Pair pool: the neighbourhood candidates that satisfy the recipient's own budgets.
- At most one accepted pair per paired step.
- Pair task-loss ranking: L_i + 0.5 B_i.
- Margins: a search margin of 1e-10 on L/B; the deployed check has zero tolerance.
- Infeasible units: an INFEASIBLE constrained unit writes a descriptive release of its first candidate.
- Decoder cache: a per-state memo re-folded on every applied move. It requires bitwise-equal (n, y, s) and is never
  reused across different label counts or teacher sums.

**Arms.**
- **C-TASK.**
  - D1 decoding.
  - Objective T = L_1 + L_2 + 0.5(B_1 + B_2).
  - Recipients are fitted separately, without SEX.
  - Starts include the source FINE-TASK map.
- **Local caps** for the constrained privacy arms: I_i ≤ I_i(C-TASK) of the same seed.
- **Constrained JOINT.** Φ = I12 + 0.5(I1 + I2), subject to both budgets, both caps, class preservation and state caps.
- **LOCAL** minimises each I_i under its own budgets and cap.
- **SEQ-12.**
  - Recipient 1 is fitted under Φ, with the other recipient's CLASS-ONLY view and only recipient 1's constraints.
  - Recipient 1 is frozen, then recipient 2 is fitted under Φ.
  - The final release satisfies both recipients' constraints.
  - The temporary partner is never required to meet a budget and is never a constant.
  - SEQ-21 is the reverse.
- **JOINT-SINGLE** uses single-recipient moves. **JOINT-PAIR** adds atomic paired moves.

## 8. Search rules and matched controls

**Search.** The exact rules, tolerances and ceilings are in SEARCH_RULES.json.
- A move takes a WHOLE fine cell to a same-class token.
- Targets per fine cell: the 4 nearest decoded-probability targets plus the 4 best exact-objective targets.
- The best strictly improving feasible move is accepted, with a restart from the coherent state.
- At most 5 sweeps.
- Pair step, after each single sweep:
  - 8 own-budget proposals per recipient, including non-improving ones;
  - 4 ranked by Φ and 4 by task loss;
  - the Cartesian product is evaluated atomically, both decoders are recomputed, and only a feasible strict Φ
    improvement is accepted.
- Decoder solves are cached by exact sufficient-statistic bytes.
- **Equal TOTAL proposal-evaluation ceilings** for JOINT-SINGLE, both sequential orders (split over their two stages)
  and JOINT-PAIR. Actual proposals and CPU are recorded.

**Starts and witnesses.**
- LOCAL starts from C-TASK.
- SEQ-ab starts from C-TASK and the source SEQ-ab maps (D1 recomputed).
- JOINT witnesses: feasible C-TASK, LOCAL, both SEQ maps and feasible source maps. An infeasible witness is never
  eligible by name.
- The objective and constraints are reported for every start and for the winner. No global optimum is claimed.

**Control bank.** It is complete and none is omitted for being strong.

| Bank | Contents |
|---|---|
| A | D0 legacy DIRECT-TASK, FINE-TASK and the full six-weight privacy bank |
| B | D1 versions of those EXACT maps: calibration-only controls with assignments unchanged |
| C | C-TASK |
| D | True-label WEIGHTED controls with D1: W-LOCAL / W-SEQ-12 / W-SEQ-21 / W-JOINT at λ ∈ {0.01, 0.025, 0.04, 0.06, 0.08, 0.1}, all seeds |
| E | Constrained LOCAL, SEQ-12, SEQ-21, JOINT-SINGLE, JOINT-PAIR |
| F | U continuous, decisions alone, RAW-J, FARE, F0, LEACE |

Notes on the weighted controls (D):
- Objective T + λΦ (T + λ(I1+I2)/2 for LOCAL).
- The same decoders, starts, caps and single-move neighbourhoods as the corresponding new arms.
- They are not constrained by the new fitting budgets, but they face the unchanged inner rules and guards.

New mapping-pair fits: 72 weighted + 18 C-TASK/constrained = 90. The full manifest is in FIT_MANIFEST.json.

**Persistence (new in lra).** Every mapper unit persists:
- the coherent start labels of every start;
- the ordered accepted moves (a paired move is one atomic entry), with their incremental terms and sufficient-statistic hashes;
- the termination receipts.

lra.mapper's replay reproduces each accepted state from scratch and checks the budgets and local caps on it. The
verifier replays the traces with its own engine. This fixes the predecessor's unreplayable mapper checks.

## 9. Correctness-only launch gate (CORRECTNESS_LOCK; ENGINEERING_GATE_RULE.json, ENGINEERING_GATE_RESULT.json)

- **What the gate asks.** It answers "is the implementation correct enough to run the registered development test?".
  It does not ask whether a synthetic example has already proved superiority.
- **Fixtures.** The four hash-pinned lcr fixture laws are reused unchanged as KNOWN, ALREADY-OPENED regression cases:
  - F1 calibrated null;
  - F2 miscalibrated teacher;
  - F3 complementary clues;
  - F4 redundant.

  FIXTURE_LAWS.json has sha256 24f70745… and laws_sha256 5c5e3bda…, with N = 4096 exact expected counts. The laws,
  counts, caps, decoder prior and source budgets are unchanged. The old gate output is kept separately
  (SOURCE_FIXTURE_GATE.json) and is never relabelled as fresh mechanism evidence.
- **Ordering.** CORRECTNESS_LOCK is pushed before the new runner or the independent verifier executes any algorithm or
  oracle on those laws.
- **Mandatory checks (ENGINEERING_GATE_RULE.json gives the exact tolerances).**
  1. Integer atom counts, class routing, label counts and law hashes.
  2. The fixed-token D0/D1 full-interface information identity.
  3. F1 calibrated null: no proper-loss improvement beyond tolerance.
  4. D1 final-vector class, simplex, KKT and projection certificates.
  5. True-label LL and source-convention Brier match a row-level reconstruction.
  6. Fitting budgets and local MI caps hold on EVERY accepted deployed state, including atomic paired moves.
  7. The temporary sequential CLASS partner is not required to be feasible; the final release meets both budgets.
  8. Incremental terms and cached solves match a from-scratch reconstruction from PERSISTED starts and moves.
  9. Exact enumeration reproduces the oracle table. Heuristic gaps are reported; a labelled gap is not a failure.
  10. The decision disclosure floor holds. An affordable CLASS comparator may make privacy superiority impossible
      without making the implementation wrong.
  11. The launch verdict, refusal paths and label truth table are wired to the actual code.
  12. Every inherited review finding has a tested disposition (REVIEW_FINDINGS_DISPOSITION.json).
- **Verdicts.** There are two:
  - ENGINEERING_READY: every mandatory check and every implementation-review obligation passes.
  - ENGINEERING_BLOCKED: otherwise.
- **What does NOT block.** ENGINEERING_READY permits the Adult study even if:
  - CLASS is zero-leakage on a fixture;
  - no fixture shows a superior privacy release;
  - constrained search ties a weighted control;
  - joint ties sequential;
  - the source gate still reads GATE_NOT_MET.

  No 0.01-nat gain, equal-leakage gain, joint advantage or favourable forecast is required.
- **If blocked.**
  - Bounded engineering repairs are made through dated code-only amendments, pushed before reruns, and prior attempts
    are kept.
  - No law, objective, comparator bank or success criterion is changed.
  - If the problem is unresolved at the resource ceiling, the study closes as ENGINEERING_BLOCKED_NOT_RUN with every
    Adult claim NOT RUN. That is not an Adult negative.
- **Wiring.** Every science stage (`lra.run`), the evaluation-lock builder, inference and closeout require a
  lock-bound, pushed ENGINEERING_READY (tests: lra/tests/test_run_gate.py). Readiness is never hard-coded.

## 10. Inner nomination (new registration; no cbp headroom) (SELECTION_RULES.json; lra/select.py)

- **Eligibility.** ORIGINAL inner eligibility for every candidate and comparator, on each task and EACH seed:
  - accuracy ≥ U − 0.01;
  - LL ≤ U + 0.01;
  - Brier ≤ U + 0.005;
  - gain ≥ 0.8 of U's gain;
  - gain ≥ 0.03;
  - decision preservation.

  cbp's 0.006/0.0035 headroom is NOT used. A mean that hides a failing seed never qualifies. One configuration is used
  across seeds.
- **Roles.**

  | Role | Definition |
  |---|---|
  | T\* | Closed privacy-untrained list: C-TASK, D0/D1 DIRECT-TASK and FINE-TASK, U, CLASS-ONLY (D0 and D1), F0. RAW-J is excluded. |
  | P\* | All privacy-trained arms, including calibrated old maps and weighted controls; guard vs T\* |
  | C\* | D0/D1 old privacy maps and weighted controls; no constrained arm |
  | N\* | Constrained arms; guards vs T\* AND C\* |
  | C_pair\* | Every release except K-JOINT-PAIR, and except fit-infeasible constrained units |
  | J\* | K-JOINT-PAIR; guards vs T\* AND C_pair\* |
  | Q | Fixed D0 DIRECT-TASK i8o64 |

- **Guard.** Each recipient's inner AUC ≤ comparator + 0.005 on every seed (inclusive).
- **Ordering.**
  1. Mean inner pair AUC.
  2. Mean summed true-label LL (round 12 decimals).
  3. Actual token states (continuous = +∞ for ordering only).
  4. Configuration ID.
- **Fallbacks.** Deterministic minimum-shortfall, DESCRIPTIVE_ONLY, distinguishing utility, guard, missing-comparator
  and technical failures.
- **Identities and aliases.**
  - Release hashes and role aliases are built on fitting and inner rows only, across all three seeds, for every role
    including comparators and Q.
  - Same tokens with a different decoder are NOT a full-release alias.
  - A privacy-trained release that is an exact alias of an untrained release is disclosed as identical_to_untrained.
  - The named family and construction is one real exact alias, chosen by construction rank, then family rank, then
    config ID.
  - CLASS stays in the T\* pool.
- **Labelling.** Fixed-map D1 privacy releases stay privacy-trained. P\* may be a control: if it wins, it is delivered
  as the useful method, and the new search is reported as adding no demonstrated benefit.

## 11. Matched attacks (lra/audit.py; AUDIT_COMPUTE.json)

- **Slate.** The pinned source FINAL slate (LR, MLP, HGB, defence-aware readers, finite categorical readers with
  smoothing 0.5/1/5).
- **Fitting and selection.** Fitted on AUDIT_FIT. AUC and CE readers are selected separately on INNER_SELECTION.
  Orientation is fixed.
- **Views.** Recipient complete interfaces (token identity, decoded probabilities, decision) and the pair (both, exact
  tuples, both ignore banks). Token-only and probability-only diagnostic families are reported. Nomination uses the
  COMPLETE interface.
- **Composition.** U composes through the COMPLETE registered map/decoder bank (all 84 codes per seed, including CLASS\|D1) before
  selection. Every source-winning composed reader is frozen for the assessment.
- **Real-data controls.** With the source limits:
  - shuffled SEX;
  - confidence within predicted class;
  - distinct-token/same-probability;
  - coalition XOR;
  - rotated signal.

  A failed control triggers engineering review. It is never evidence of privacy.

## 12. Primary family and labels (PRIMARY_FAMILY.json, LABEL_TRUTH_TABLE.json; lra/family.py, lra/infer.py)

- **37 fixed slots.**

  | Claim | Comparison | Slots |
  |---|---|---|
  | A | P\* vs T\* | 11 |
  | B | N\* vs C\* | 11 |
  | C | J\* vs C_pair\* | 11 |
  | Q | the four original confidence bounds (two LL upper bounds < 0.01, two Brier upper bounds < 0.005) | 4 |

- **Clauses.** Each 11-clause claim requires:
  1. Pair AUC(comparator) − AUC(nominee): lower bound > 0.02.
  2–3. Each local AUC(nominee) − AUC(comparator): upper bound < 0.01.
  4–5. Each accuracy − U: lower bound > −0.01.
  6–7. Each LL − U: upper bound < 0.01.
  8–9. Each Brier − U: upper bound < 0.005.
  10–11. Each accuracy − 0.8 U − 0.2 constant: lower bound > 0.
- **Inference.**
  - 1,999 paired exact-record-group bootstrap replicates, with the NEW seed 20261010 and the same draws for every arm
    and model seed.
  - Equal-seed aggregation follows source conventions; seeds are not independent populations.
  - z = NormalDist().inv_cdf(1 − 0.05/74) ≈ 3.2048452050105634, verified.
  - All 37 slots are kept after aliases or missing nominees. There is no post-outcome family reduction.
- **Claim statuses.**
  - PASS: only with a valid nominee and comparator and every clause passing.
  - NOT_ESTABLISHED.
  - NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE.
  - INCOMPLETE_OR_INVALID: missing required work, a missing comparator, a failed required control, nonfinite
    statistics or unresolved verification failures.
- **Overall label, in precedence order.** A, B, C and Q are always displayed separately.
  1. A pre-science blocker:
     - ENGINEERING_BLOCKED_NOT_RUN (a failed correctness gate);
     - INCOMPLETE_NOT_RUN (a pre-fit budget or admission blocker, with its reason).
  2. An unresolved global technical failure after science begins: INCOMPLETE_OR_INVALID. A separately valid Q may be
     reported, but it cannot hide the incomplete work.
  3. Passing method components:
     - A passes: PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (actual family; existing | calibrated | weighted |
       constrained);
     - B passes: + CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED;
     - C passes: + PAIRED_JOINT_INCREMENT_ESTABLISHED.
  4. Q only: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.
  5. Complete valid work with no pass: EXPERIMENTAL_NO_ADVANTAGE.

  The source MECHANISM_GATE_NOT_MET is historical and is never this study's verdict.
- **What does not follow.** A passing A implies neither B nor C. A fitting-budget certificate does not imply an
  assessment clause. A point below a limit with an upper bound above it is unresolved preservation.

## 13. Stages and locks

| Stage | Lock | Content |
|---|---|---|
| 0 | — | admission (done), the 14 review repairs (REVIEW_FINDINGS_DISPOSITION.json), port, timing on synthetic shapes |
| 1 | CORRECTNESS_LOCK | `correctness` → ENGINEERING_GATE_RESULT.json; independent replay (E phase 1) |
| 2 | SCIENCE_LOCK | everything in §§6–12, code and dependency hashes, FIT_MANIFEST, PREDICTIONS; requires ENGINEERING_READY |
| 3 | SCIENCE_LOCK | `d1`: 81 fixed-map D1 units (incl. CLASS\|D1); `ctask`: 3; `fit`: 72 weighted + 15 constrained |
| 4 | SCIENCE_LOCK | `inner` (267 units), `inner_src`, `controls`, `select`; E phase 2 replays selection BEFORE the evaluation lock |
| 5 | EVALUATION_LOCK | built only with valid technical status, controls, admission and the engineering/science locks; no escape flag |
| 6 | EVALUATION_LOCK | one locked exploratory assessment (`lra.assess`) and inference (`lra.infer`) |
| 7 | — | E phase 3, deployment, figures, backup, closeout |

- **Scored list (lra.eval_lock).**
  - The roles P\*, N\*, J\*, T\*, C\*, C_pair\* and Q, or their descriptive fallbacks.
  - U and decisions alone.
  - The strongest inner D1 fixed-map privacy control and its paired D0.
  - The strongest inner weighted control.
  - C-TASK and the five constrained arms.
  - RAW-J, FARE, F0 and LEACE.
  - The paired D0 versions of P\* and of C-TASK, for decoder-only diagnostics.

  Exact aliases are deduplicated; every role and all 37 slots are kept.
- **After the opening.** No new κ, buffer, budget, λ, capacity, candidate, split, attacker selection or stopping rule.
  Technical reruns use the same locked estimands and keep every attempt.

## 14. Diagnostics and independent verification

- **Diagnostics.**
  - D0 vs D1 on IDENTICAL tokens (DECODER_ONLY_ABLATION.csv, DECODER_UTILITY_ABLATION.csv).
  - C-TASK vs fixed old task maps with D1.
  - Weighted vs constrained with the same decoder.
  - LOCAL, sequential orders, JOINT-SINGLE and JOINT-PAIR.
  - Fitting feasibility vs inner and assessment losses.
  - Per-task LL, Brier, utility, recall and alphabet.
  - Pair alongside individual recovery.
  - Support, singleton and unseen-pair rates.
  - Fitted vs permutation-null MI.
  - Training label use and calibration.
  - Decision floor and CLASS feasibility (DECISION_FLOOR_AND_FEASIBILITY.csv).
- **Same-map D1 − D0 confidence contrasts.** These are supplementary, outside the 37 slots, on the same 1,999 draws,
  with nominal 95% intervals (z = NormalDist().inv_cdf(0.975)). They cover the maps named on INNER before the assessment.
- **Verifier (role E).** It imports no study module and reproduces the prompt §14 list, including:
  - the convex decoder certificate;
  - the new gate;
  - incremental replay from persisted traces;
  - every selection value;
  - every endpoint;
  - deployment refusals;
  - restores;
  - the staged exposure boundaries.

## 15. Packaging, backup and claim scope

- **Deployment.**
  - `python -m lra.deploy` takes exactly the pinned 83-column schema and the bound U teacher, map and decoder.
  - It outputs only tokens, decoded probabilities and decisions.
  - Packages: U, Q, P\* (even if it is a control), the constrained candidate or fallback with its truthful status, the
    decoder-only baseline and the sequential winner. Each carries its training-label-use disclosure.
- **Backup.**
  - Off-device if the drive is identified by content; otherwise a verified versioned same-device copy, restored from.
  - Predecessor custody runs only after this study's evaluation opening. The chain cbp → qpc → dpc → osf DOES unseal
    OSF_DEVELOPMENT_ASSESSMENT, inside osf.closeout.backup. lra.closeout therefore runs it only after the lra opening
    (EVALUATION_LOCK on origin plus a bound outer__ unit). A proof from code alone is not available.
- **Prior art.** Privacy Funnel (arXiv:1402.1774) and Taylor–Vippathalla–Coon (arXiv:2601.21859v2) are established. No
  novelty is claimed. The joint discrete program is not convex. Fitted MI is not a population guarantee. The sequential
  arms are matched adaptations.

**Gate result (recorded before SCIENCE_LOCK).** CORRECTNESS_LOCK was pushed at de4495f, and the correctness stage ran at
05:07:54–05:08:58Z. All twelve checks pass: **ENGINEERING_READY** (ENGINEERING_GATE_RESULT.json; evidence 0ea0d7c).
- Checks 6 and 8 replayed 2,043 accepted states from the persisted traces, with a worst difference of 1.5e-15.
- No paired move was accepted on the fixtures. The paired path is covered by registered synthetic tests through check 11.
- The oracle tables are byte-identical to lcr's.

The independent replay (role E, phase 1) is recorded in INDEPENDENT_VERIFICATION.json.

**Independent gate check, and a registered numerical behaviour (recorded before SCIENCE_LOCK).**
- Role E's own replay of checks 1–12 agrees: ENGINEERING_READY, across 120 traces, 2,043 accepted states and 1,921
  certified D1 vectors (INDEPENDENT_VERIFICATION.json, PHASE_1).
- The local cap I_i ≤ I_i(C-TASK) is compared on exact float bits (search margin 0). When the true MI equals the cap,
  float noise of order 1e-16 can exclude a start or witness. On F3, two K-SEQ stage-2 starts and three K-JOINT
  witnesses were excluded this way.
- This is conservative: no violating state is ever accepted.
- The rule is unchanged and applies on Adult as registered. Exclusions at the cap are reported in OPTIMIZATION_RECEIPTS,
  never relaxed.

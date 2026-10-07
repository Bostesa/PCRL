# PCRL learned decoders and confidence-constrained releases (lcr) — registered protocol

| Item | Value |
|---|---|
| Branch | `research/pcrl-learned-decoder-constrained-release-v1` |
| Started from | cbp final tip `7f3ec67b2ecd86d474e2ff27167091af9923f572` |
| qpc lineage | evidence `9dd06da`, tip `d0c8a45` |
| Package | `lcr/` |
| Public results | `results/pcrl_learned_decoder_constrained_release_v1/` |
| Private store | `<PRIVATE_CACHE>/lcr_v1/` |

**What the locks freeze.**
- FIXTURE_LOCK freezes the fixture laws, the gate and the fixture engine before any algorithm runs on them.
- SCIENCE_LOCK freezes every Adult definition before any Adult fit.
- The executable definitions are the locked code.

**Disagreements.** A code/text disagreement is a DEFECT. It is disclosed and resolved, and code never silently overrides
a registered scientific rule.

## 1. Questions

1. Can learning the released probability vectors and code assignments with the actual task losses and explicit budgets
   produce a useful privacy release?
2. Does the new constrained search contribute beyond better decoding, ordinary compression and equally capable
   sequential methods?

A simpler control may answer (1) positively while defeating (2). That is reported honestly. There is no promise of
novelty, chance-level privacy, confirmation or a joint-design win. This is a fixed-task output-release algorithm. It does
not provide general-purpose features.

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

## 3. Exposure, roles, labels (EXPOSURE_LEDGER.md)

- **Exposure statement.** It is registered verbatim in EXPOSURE_LEDGER.md. All roles were used historically. This is
  exploratory development evidence, the intervals are conditional, and no confirmation population is opened.
- **Roles (unchanged).**

  | Role | Rows |
  |---|---|
  | OSF_DEFENSE_FIT | 15,434 |
  | HEAD_VALIDATION | 1,500 (historical only; not repurposed) |
  | AUDIT_FIT | 6,065 |
  | INNER_SELECTION | 2,235 |
  | OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) |

- **Input.** `adult_jcv.npz`, sha256 e0d9e54a…2f12. The 83 permitted columns are used in pinned order.
- **Label use (material change).**
  - Task labels and SEX on OSF_DEFENSE_FIT are used for the NEW supervised fitting: coarse assignments and D1 decoders.
    The fine partitions stay fixed and label-blind.
  - AUDIT_FIT trains attackers, and INNER_SELECTION selects.
  - Assessment labels stay sealed until the pushed EVALUATION_LOCK; only `lcr.assess` unseals.

## 4. Budget and resources

- **Time.**
  - 10 h elapsed from 2026-10-06T23:42:46Z (ceiling 2026-10-07T09:42:46Z).
  - The final 2 h and 4 CPU-h are reserved for verification and closeout, from 07:42:46Z.
- **Compute.**
  - 20 CPU-h in total.
  - At most 2 heavy numerical processes in total, through one shared semaphore (`lcr.sema`, with ledger SEMA_LOG.jsonl
    and COMPUTE_LEDGER.jsonl).
  - One thread per process; aggregate memory at most 8 GiB; at least 5 GiB free disk.
  - Measured and estimated CPU are counted separately.
- **Cloud.** $0.
- **Timing.** The full bank was timed on realistic synthetic shapes before fitting (TIMING.json). There is no
  result-based pruning. If the mandatory bank cannot fit, the first step is to fix scheduling and caching. If it still
  cannot fit, the result is a pre-fit budget blocker, never a selectively weak comparison.

## 5. Source admission (SOURCE_ADMISSION_LOCK, pushed at 4a91947)

- **Admitted by verified copy.** The units below are hash-matched to cbp unit COMPLETE.json, the cbp same-device copy
  SHA256SUMS and the cbp EVALUATION_LOCK:
  - U and RAW-J teachers;
  - deployed heads;
  - fixed fine partitions (income 32 / occupation 128 per predicted class);
  - the D0 bank (DIRECT-TASK i8o64, FINE-TASK i8o64, CLASS-ONLY, and LOCAL / SEQ-12 / SEQ-21 / JOINT at λ ∈ {0.01,
    0.025, 0.04, 0.06, 0.08, 0.1});
  - official LEACE, FARE and F0 references.
- **Parity.** Teacher forward parity and release re-encode parity are bitwise (stage admit, 2026-10-06T23:57:50Z:
  ADMITTED, 6/6 teachers, 81/81 codes).
- **cbp inner audits.** They are admitted as custody only. lcr recomputes every D0 audit under `aud__*`, because the
  lcr audit adds token-only and probability-only diagnostic families. `lcr.audit.cbp_parity` checks the complete-family
  agreement.
- **Not copied:** no outer unit and no per-person assessment prediction.

## 6. Common learned decoder D1 (lcr/decoder.py; METHOD_CARD.md; DECODER_CERTIFICATES.json)

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

## 7. Fitting budgets and the new mapper (lcr/mapper.py; SEARCH_RULES.json)

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

**Registered consequence (N-9).** There is no restoration phase.
- The U teacher heads were fitted on OSF_DEFENSE_FIT, so L_i(U) and B_i(U) on the fitting rows are in-sample.
- If C-TASK exceeds a fitting budget on some seed, then on that seed:
  - K-LOCAL, whose only start is C-TASK, is INFEASIBLE by construction;
  - K-SEQ and K-JOINT depend on feasible source starts and witnesses.
- C-TASK's deployed budget slack is read right after the ctask stage. It is a pre-registered diagnostic, never a tuning
  point.
- Selection treats an infeasible constrained fit as CONSTRAINED_FIT_INFEASIBLE (SELECTION_RULES.json).

**Search design choices registered in SEARCH_RULES.json (lcr.mapper.rules()).**
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

## 9. Mechanism gate (FIXTURE_LAWS.json, FIXTURE_GATE_RULE.json, FIXTURE_GATE.json)

**Fixtures.** Four fixed, known-law fixture families, written before any algorithm runs on them:
1. calibrated-teacher null;
2. miscalibrated teacher;
3. complementary clues;
4. redundant / no coordination.

- N = 4096 exact expected counts (atom probabilities are multiples of 1/4096).
- Exhaustive enumeration of canonical same-class partitions, at most 100,000 pairs per fixture.
- Exact-law MI and proper losses, computed independently.

**Mandatory correctness.**
- Token-preserving decoder changes leave full-token information unchanged.
- The calibrated null gives no population decoder improvement beyond tolerance.
- Budgets are enforced.
- Decision preservation holds.
- Incremental terms match an independent reconstruction.
- Heuristics are labelled as such where they differ from exhaustive references.

**Trigger for Adult.** At least one fixed nontrivial fixture shows privacy-trained partitioning with D1 that:
- satisfies all utility/local budgets;
- reduces pair MI by ≥ 0.01 nats beyond the strongest feasible task-only D1 compression;
- keeps both tasks' accuracy gain ≥ 0.03 over their constants.

**Route classification** (frozen): DECODER_ENABLED_ROUTE if the trigger passes only because D1 makes an old private map
usable; ASSIGNMENT_SEARCH_ROUTE if new assignment search also helps.

**Pre-lock clarifications of FIXTURE_GATE_RULE.json (registered at dd1cf23; changed before FIXTURE_LOCK, laws
unchanged).** A text/code review before the lock found disagreements. Each was resolved before any study algorithm ran:
- **C3:** the text now matches the intended and implemented scope. The budget part of feasible() binds the constrained
  (K-) arms. Unconstrained arms (C-TASK, W-) report FEASIBLE for their own constraints only (decisions and caps); their
  budgets are evaluated at qualification.
- **C5:** the winning start's incrementally updated search-state terms are now compared with the from-scratch rebuild.
  Before, C5 compared two from-scratch computations. The token-loss helper condition is now registered.
- **C7:** on the registered run, the laws and the rule must equal the lock's documents_sha256, and reloaded shard units
  must carry the same laws hash.
- **C2:** the per-arm released q is compared with the D0 decoding of the same pair.
- **C1, C6:** the extra C1 condition (every release of every arm) and the NOT_A_SEARCH label for D1 fixed-map
  controls are now registered.
- **Descriptive flags:** the two descriptive flags (best constrained vs best weighted I12; joint vs sequential I12) are
  now defined and computed on every fixture. They never enter the verdict.

**Registered expectation before the stage runs (structural, not a fit result).**
- **Bound.** Every class-preserving release determines both predicted classes (c1, c2). So I12(R) ≥ I12(CLASS|D1) for
  every release R.
- **Consequence.** Whenever CLASS|D1 is budget-feasible, it is the strongest feasible task-only compression, and no
  release can reduce pair MI below it. A triggering fixture therefore needs CLASS|D1 to violate a fitting budget.
- **Laws.** In all four registered laws, SEX is balanced within every predicted-class pair, so I12(CLASS|D1) = 0
  exactly.
- **E's oracle counts.** Role E's independent oracle was run on the registered laws BEFORE this lock. That is a
  deviation from "push FIXTURE_LOCK before algorithms run on them"; it is disclosed here. Its counts:
  - CLASS|D1 is budget-feasible on F2, F3 and F4;
  - F1 has no budget-feasible task-only map.
- **Expected verdict.** GATE_NOT_MET with NO_FIXTURE_TRIGGERED: F1 for NO_FEASIBLE_TASK_ONLY; F2–F4 because T* =
  CLASS|D1 with I12 = 0, so they are not nontrivial.
- **Laws not amended.** The prompt requires the laws to be written before fixture algorithms run and forbids a hunt
  for a favourable example, and the outcome was already predictable from E's counts. The gate is run exactly as
  registered. Its failure is reported with this precise scope: the registered fixtures could not trigger. It is not
  evidence that the mechanism fails on Adult.

**If the gate fails:**
- no Adult fit;
- finish the fixture decision, oracle replay and deployment/backup;
- the label is MECHANISM_GATE_NOT_MET, not an Adult result.

## 10. Inner nomination (SELECTION_RULES.json; lcr/select.py)

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
  | T\* | Closed privacy-untrained list: C-TASK, D0/D1 DIRECT-TASK and FINE-TASK, U, CLASS-ONLY, F0. RAW-J is excluded. |
  | P\* | All privacy-trained arms, including calibrated old maps and weighted controls; guard vs T\* |
  | C\* | D0/D1 old privacy maps and weighted controls; no constrained arm |
  | N\* | Constrained arms; guards vs T\* AND C\* |
  | C_pair\* | Every release except K-JOINT-PAIR |
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
- **Labelling.** Fixed-map D1 privacy releases stay privacy-trained. P\* may be a control: if it wins, it is delivered
  as the useful method, and the new search is reported as adding no demonstrated benefit.

## 11. Matched attacks (lcr/audit.py; AUDIT_COMPUTE.json)

- **Slate.** The pinned source FINAL slate (LR, MLP, HGB, defence-aware readers, finite categorical readers with
  smoothing 0.5/1/5).
- **Fitting and selection.** Fitted on AUDIT_FIT. AUC and CE readers are selected separately on INNER_SELECTION.
  Orientation is fixed.
- **Views.** Recipient complete interfaces (token identity, decoded probabilities, decision) and the pair (both, exact
  tuples, both ignore banks). Token-only and probability-only diagnostic families are reported. Nomination uses the
  COMPLETE interface.
- **Composition.** U composes through the COMPLETE registered map/decoder bank (all 83 codes per seed) before
  selection. Every source-winning composed reader is frozen for the assessment.
- **Real-data controls.** With the source limits:
  - shuffled SEX;
  - confidence within predicted class;
  - distinct-token/same-probability;
  - coalition XOR;
  - rotated signal.

  A failed control triggers engineering review. It is never evidence of privacy.

## 12. Primary family and labels (PRIMARY_FAMILY.json, LABEL_TRUTH_TABLE.json; lcr/family.py, lcr/infer.py)

- **37 fixed slots.**

  | Claim | Comparison | Slots |
  |---|---|---|
  | A | P\* vs T\* | P01–P11 |
  | B | N\* vs C\* | P12–P22 |
  | C | J\* vs C_pair\* | P23–P33 |
  | Q | four original confidence bounds | P34–P37 |

- **Clauses of each 11-clause claim.**

  | Clauses | Condition |
  |---|---|
  | 1 | pair AUC(comparator) − AUC(nominee): lower bound > 0.02 |
  | 2–3 | local AUC(nominee) − AUC(comparator): upper bound < 0.01 |
  | 4–5 | accuracy − U: lower bound > −0.01 |
  | 6–7 | LL − U: upper bound < 0.01 |
  | 8–9 | Brier − U: upper bound < 0.005 |
  | 10–11 | accuracy − 0.8 U − 0.2 const: lower bound > 0 |

- **Inference.** 1,999 paired exact-record-group bootstrap replicates with seed 20261009 and identical draws. z =
  NormalDist().inv_cdf(1 − 0.05/74) = 3.2048452050105634 (verified). Slots are kept after aliases or missing nominees.
- **Clause outcomes.** PASS / NOT_ESTABLISHED_PRECISION / NOT_ESTABLISHED_POINT / MEASURED_VIOLATION / INVALID.
- **Claim statuses.** PASS / NOT_ESTABLISHED / NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE / INCOMPLETE_OR_INVALID, with exact
  root causes.
- **Overall label.** Components are joined.
  - A passes: PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (family; existing | calibrated | weighted | constrained).
  - B passes: + CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED.
  - C passes: + PAIRED_JOINT_INCREMENT_ESTABLISHED.
  - Otherwise, Q passes: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.
  - Otherwise: EXPERIMENTAL_NO_ADVANTAGE.
  - Any incomplete work without a passing component: INCOMPLETE_OR_INVALID.
  - A gate failure: MECHANISM_GATE_NOT_MET.
- **What does not follow.** A passing A implies neither B nor C. A fitting-budget certificate does not imply a passing
  assessment clause. A point below a limit with an upper bound above it is unresolved preservation, not an established
  violation.

## 13. Stages and locks

| Stage | Content |
|---|---|
| 0 | Admission, engineering, synthetic timing of the full bank |
| 1 | FIXTURE_LOCK pushed; fixture stage; independent oracle replay; gate registered |
| 2 | SCIENCE_LOCK pushed: every definition, code hash, manifest, prediction, control and label rule |
| 3 | d1 (D1 fixed-map decoders) and fit (90 units) |
| 4 | Inner audits; composition; controls; selection; independent selection replay; nominees frozen |
| 5 | EVALUATION_LOCK pushed |
| 6 | One locked assessment |
| 7 | Independent replay, deployment, reports, backup and closeout |

**Scored list (lcr/eval_lock.py).**
- P\*, N\*, J\*, T\*, C\*, C_pair\* and Q (or their fallbacks);
- U and decisions alone;
- the best D1 fixed-map privacy control and its paired D0 release;
- the best weighted privacy control;
- C-TASK;
- the five constrained arms;
- RAW-J, FARE, F0 and LEACE.

**After the assessment.** No new κ, budget, λ, candidate, partition, attacker selection or stopping rule. Technical
reruns use the identical locked artifacts.

## 14. Diagnostics and independent verification

**Diagnostics.**
- D0 vs D1 on IDENTICAL tokens (DECODER_ONLY_ABLATION.csv).
- C-TASK vs the fixed old task maps with D1.
- Weighted vs constrained.
- LOCAL / sequential / JOINT-SINGLE / JOINT-PAIR.
- Fitting feasibility vs inner/assessment losses (BUDGET_FEASIBILITY.csv).
- Per-task LL/Brier/recall/alphabets.
- Pair alongside individual recovery.
- Support/singletons/unseen pairs.
- Fitted MI and permutation null separately.
- Label-use and overfitting diagnostics.

**Verifier (role E).** Imports no runner module and reproduces:
- hashes and groups;
- the D1 decoder and certificates;
- the fixture enumeration and gate;
- incremental terms;
- deployed budgets and caps;
- parity;
- eligibility and selection;
- composition coverage;
- all endpoints and labels;
- tables and figures;
- deployment;
- restores.

Tolerances are registered, verdicts must agree exactly, and hashes, tokens and decisions must be identical.

## 15. Packaging, backup and claim scope

- **Deployment.** `python -m lcr.deploy` binds the teacher, schema, map and decoder, and outputs only tokens, decoded
  probabilities and decisions. Every package discloses its training-label use, status, schema, binding and restore
  recipe.
- **Backup.** A same-device versioned copy, with off-device backup and predecessor custody PENDING if the drive is
  absent.
- **Prior art (PRIOR_ART_AND_CLAIM_SCOPE.md).** These are established ideas:
  - hard utility constraints;
  - privacy-funnel objectives;
  - proper-loss calibration;
  - greedy partition search;
  - sequential collusion accounting.

  Their combination is not automatically novel.
- **Required wording.**
  - The full discrete program is not convex.
  - Fitted MI is not a population guarantee.
  - The sequential arms are matched adaptations, not the official Taylor, Vippathalla and Coon solver.

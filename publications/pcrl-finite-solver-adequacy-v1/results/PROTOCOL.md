# Protocol: solver adequacy of the finite-output benchmark and conditional method development

Campaign `pcrl_finite_solver_adequacy_v1`. Coordinator's protocol, written 2026-10-21 from 02:41Z (`date -u`). Campaign start
02:22:58Z; research window 8 hours, plus 30 minutes for closeout. It is registered, with the statistical reviewer's frozen rules
(`roles/statistical_reviewer/RULES.md` / `RULES.json`), before any new real-label computation, and its sanitized receipt is verified
on the GitHub remote. Registered files are never edited: an amendment is a new dated file stating its scope and timing.

## 1. Purpose and preserved findings

This is statistical machine-learning software research on admitted public-use Census ACS tabular records. It asks whether the
optimizer of the propensity-routed finite-cell release is adequate for its own stated objective. Only if a prospectively defined
condition is met does it develop a solver-repaired method on development data.

Preserved findings (publication 04ac9161f, `pcrl_finite_routing_comparison_v1`), not relitigated here:
- The out-of-fold teacher-mean (B1) tables repaired avoidable task error on fixed partitions.
- The repaired finite twin (TWB_C64, 16 x 64 cells) is a useful task-only release (D1 `REPAIRED_TASK_ONLY_RELEASE`).
- The protected routing ran but did not show benefit beyond matched controls. Pair AUC: PRC_S2 0.6485, task-score-only HQ_S2 0.6489,
  twin 0.6555. It stayed above the strong target references.
- Under the frozen validity rule the previous outcome is `INCOMPLETE`; `ENGINEERING_REPAIR` exists only under the post hoc ER1.
  Both readings stand. This campaign's reader-validity rule (section 9) is prospective and is never applied to the old grid.

## 2. Data, roles and the service

- **Population.** The admitted Florida ACS 2019 cohort (public-use PUMS person records). The real-data roles are MODEL_FIT (20,000
  people, 13,337 households), CALIBRATION (2,982 people, 1,974 households) and MODEL_SELECT (8,001 people, 5,306 households), all
  household-grouped.
- **Fixed definitions.** The exact income (2 classes) and five-class occupation definitions, U0 / Ucal and the source manifests are
  unchanged.
- **Closed.** ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT labels, the sealed assessment and capped-out people. DEV_REPORT features are
  used label-free for deployment parity only. There is no new population or download.
- **Runtime release.**
  - Input: six fields, AGEP, WKHP, COW, SCHL, MAR, POBP (the admitted 145-column encoding).
  - The income recipient receives its own probability vector (2) and argmax decision; the occupation recipient receives its own (5)
    and its decision.
  - No internal identifier, cell index or other feature is transmitted. Identical emitted vectors are one observable value.
- **Illustrative probability-use scenario.** A service scores a person once from the six fields. One office receives the income
  probability, for example to prioritize an income-support review; another receives the occupation probabilities, for example to
  route a job-matching contact. Each office acts on its vector and decision.
- **Access considered.**
  - *Standard recipient access:* one recipient's vector.
  - *Linked-recipient access:* both vectors for the same person, joined. Evaluation classifiers ("readers") estimate the dataset's
    recorded SEX attribute from exactly these released values, and recovery is their AUC.
  - One target release per person (one deterministic instance; no repeated queries).
- **Auxiliary sample sizes for the readers.** Inside MODEL_SELECT, by household: about 4,000 FIT, 1,590 VALIDATION and 2,400 SCORE
  people per repetition (`READER_SPLITS.json`). Fit on FIT, choose settings on VALIDATION, score on SCORE.

## 3. Three separate questions

- **A, optimization.** Does the optimizer find good solutions to its stated objective? Answered on exact invented laws (section 5)
  and on cached MODEL_FIT statistics (section 6).
- **B, usefulness and recovery.** Does that objective, better optimized, improve actual task usefulness and attribute recovery?
  Answered by exact population metrics on the laws and nested MODEL_FIT held-out metrics, and by MODEL_SELECT development evidence
  only in the pilot.
- **C, propensity.** Does person-level propensity add anything beyond an equally optimized HQ (task-score-only routing)?
  - HQ's restriction is explicit: it routes on twin-cell means of the propensity and the SEX-conditional occupation statistics, while
    its acceptance objective uses person-level propensities.
  - Where task scores uniquely identify support states, an unrestricted task-score mapping can express every partition, so no
    structural expressivity advantage of propensity is asserted.

## 4. Objective audit (stage `audit`, card FA_FSA_v2; independent replay by the verifier)

The following are compared with independent calculations:
1. the routing proposal score (shrunk, linearized dependence);
2. the acceptance objective (unshrunk π-weighted plug-in information + guard, with the search prototypes);
3. the probability tables actually emitted (B1);
4. the task losses and per-SEX occupation losses of those emitted vectors.

Each difference is classified as an intentional approximation documented in card FA_PRC_v1, or an implementation defect (behaviour
contradicting the specification). The objective is never silently changed under the name of a solver repair.

Also checked:
- The accepted objective decreases strictly along the original chain. Monotonicity is never read as a local or global optimality
  certificate.
- Runtime routing consistency: the deployed rule is a function of the six fields through the saved teachers. In-sample and
  out-of-fold routing of training rows are compared. An arbitrary reassignment of training rows is not a deployable release, and no
  row identifier is ever a runtime feature.

## 5. Exact references (exact-reference investigator; laws committed before any numerical experiment on them)

- **The laws.** Small finite joint laws P(X, S, Y_income, Y_occupation) with explicit probabilities and permitted runtime mappings
  (X stands for the runtime inputs; m <= 12 support states). Task posteriors, attribute posteriors and population task metrics are
  computed from the law. They include:
  - a calibrated null;
  - laws with genuine task-score collisions and laws without;
  - an independently checked feasible many-to-one witness, whose utility and recovery are recorded before any improvement is
    interpreted.
- **Per-law comparators.** Each law reports the decision-only release, its task feasibility and its attribute recovery. Improvement
  below a feasible zero-recovery comparator is never required, and a null correctly permits no benefit.
- **Narrow reference first.**
  - Hold a refreshed routing book fixed and enumerate every tilt breakpoint on the support.
  - Evaluate the exact objective on the resulting deployable partitions, including the current state and the seven original
    proposals.
  - Compare the best attainable proposal in this declared search class with the old first-improvement choice.
  - This is exact for that proposal step, not for the full alternating optimizer.
- **Broader enumeration (where tractable).** It is reported with its scope. An unrestricted partition optimum is a relaxation unless
  the partition is realizable by the declared runtime rule, shown by running prediction and reproducing assignments. A finite
  parameter grid is named and never called a global continuous optimum.
- **Recovery.** Surrogate information and the exact Bayes AUC are compared separately: lower information need not lower AUC.
  Identical emitted vectors are merged before recovery is computed.
- **Tolerances** (the statistical reviewer may tighten them in RULES):
  - objective ties within 1e-9;
  - Bayes AUC differences below 0.001 are no material change;
  - task-metric changes below 0.0005 are no worse.

## 6. Solver variants (card FA_FSA_v2; at most two technically distinct variants)

- **SV1, exact tilt line searches with best improvement.** At each step, for the refreshed book and for the current book, it
  enumerates every breakpoint of the deployable partitions route(E, book, t), t <= λ (proposal resolution separated from the
  objective strength; zero, the previous tilt and every boundary included), and selects the interval with the best exact objective.
  The current-book line was added before any real-data or registered-law run (card FA_FSA_v2, which supersedes FA_FSA_v1 before use).
  - It preserves the objective, routing family, teachers, capacity and emitted-table rule, so the contrast is solver-only.
  - It is checked for objective arithmetic, finite termination, empty cells, tie handling and rule consistency
    (`fsa.fa.solver.selftest`).
- **SV1_first, the equal-effort control.** The same candidate set with the original first-improvement rule. It gives the old
  selection rule comparable objective-evaluation effort and isolates selection from resolution. No new starts are introduced, so no
  restart control is needed. If starts are added later, an equal-effort original-solver restart control comes with them.
- **A second variant** only if the audit justifies it, by a dated amendment card with its own control, before its fits. A changed
  objective or architecture would be labelled separately and needs its own control.
- **HQ** gets the same variant, the same initialization (the B1 twin) and comparable effort.
- **Source-specific check (stage `source`).** On the cached MODEL_FIT out-of-fold statistics (derived from training labels, never
  described as untouched data), under card FA_FSA_v2, without MODEL_SELECT. The original, SV1 and SV1_first chains of PRC and HQ are
  compared at every registered strength. Stage `nested` adds MODEL_FIT nested held-out task spend and LOOKUP AUC for both solvers in
  the same outer folds.

## 7. P1 replay (closes the previous verification gap)

- **Rerun.** Before relying on the old P1 evidence, the coordinator reruns the previous campaign's nested P1 estimator unchanged
  (`frc.fa.route.nested_c`, with the recorded inputs) and saves the per-person held-out outputs.
- **Independent recomputation.** The independent verifier recomputes the decisive P1 costs, household-clustered uncertainty, the
  selection and the F4 bootstrap from that file, with its own code.
- **Acceptance.** The tolerances are in RULES. A mismatch is reported; it never changes an old label and authorizes no closed-role
  access.

## 8. Prospective advancement condition (defined by the statistical reviewer in RULES before any new outcome)

A conditional pilot is justified only by one of these routes:
- an independently verified implementation defect affecting the real pipeline;
- a strictly better feasible, deployable point found by a revised solver in a valid same-class reference comparison, with evidence
  that the improvement concerns actual usefulness or recovery and not only an irrelevant surrogate;
- the source-specific route on cached MODEL_FIT statistics: a distinct feasible improvement together with a usefulness or recovery
  rationale.

Unrestricted partition enumeration beating a restricted deployable model never advances. If the old solver already matches the
applicable reference, or the new solver only improves a proxy, the optimizer-repair hypothesis closes. That is not an impossibility
result about other mechanisms. If the condition is met, the pilot below runs without a further prompt.

## 9. Conditional development-data pilot (only if section 8 is met; its card is committed before its fits)

**Setup.**
- The chosen solver is frozen before any MODEL_SELECT result.
- The baseline is the repaired B1 twin TWB_C64 at the admitted capacity. The full permitted teacher-training exposure is kept, and
  verified artifacts are reused.
- The operating path, specified in advance: identity and up to three genuinely distinct nonzero points per version, chosen on
  MODEL_FIT nested estimates only. Strengths with identical outputs count once.

**Compared.**
- Original-solver PRC and HQ: the saved frc PRC_S2 / HQ_S2 are reused.
- Revised-solver PRC and HQ, with the same optimization improvement, initialization and comparable effort.
- The same task-only twin.
- One relevant untargeted or ordinary-compression control.
- The nine admitted strong references.

**Utility limits** (the original ones): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005, all 24
source clauses and the 4 occupation subgroup guards. The two-SE reserve is reported separately. The old 0.004 capacity gate is not
reopened, and old verdicts are not altered.

**Evaluation.**
- MODEL_SELECT is used only for recorded evaluation, never to fit release parameters. All findings are adaptive development evidence.
- Readers on frozen household-disjoint FIT / VALIDATION / SCORE roles, with the expanded admitted bank. It includes smoothed
  frequency readers for observable finite values and nonlinear readers. Auxiliary counts, sparse cells and unknown-cell handling are
  reported.
- Readers see only released vectors and decisions. Combined-output evaluation includes the alternatives that ignore either
  recipient.
- Paired comparisons, with the dependence among overlapping splits accounted for. The two source versions are not independent
  replications. A version left unchanged by the revised solver does not cancel experiments on the other; aliases and
  version-specific outcomes are reported, and pooled points are never manufactured from incomparable configurations.

**Diagnostics.** For the strongest feasible nonzero point: one component ablation and effective comparator refits, even if the first
interval crosses zero.

**Claims kept separate:** better optimization; useful task performance; protection beyond matched controls; improvement over strong
references; the 0.020 material margin; joint design (not claimed without separate or sequential controls); novelty (a better line
search is not a novel privacy algorithm).

**Reader validity (prospective; RULES).**
- A deterministic evaluation classifier is legitimate.
- Expected seed invariance on finite outputs does not invalidate a comparison, and duplicates count once.
- Household fitting splits vary, and the recipes that actually change are recorded.
- Genuine schema, access, score or seed-implementation errors still fail.

## 10. Verification, deployment, custody, publication

**Verification.** The independent verifier, with separate numerical code, checks:
- the exact-reference calculations;
- the objective changes;
- the actual emitted metrics;
- the P1 replay;
- the pilot selection and its uncertainty;
- representative reader refits.

Failed attempts are preserved. Numerical reproducibility is reported separately from procedural validity, and no rule is modified
after a failure.

**Deployment.**
- If a feasible real-data release exists, the revised method and its decisive control are packaged next to the baseline.
- If no pilot is justified, the tested solver and reference implementation and the exact findings are delivered without claiming a
  fitted method.
- Deployment reproduces evaluated outputs from the six fields and rejects unsupported schemas at documented and underlying entry
  points.

**Custody.**
- Representative artifacts are restored and run from the backup alone.
- An attached writable external drive is used if available. Otherwise off-device custody is marked pending, with the safe command.
- No forced mount. A warm read is never described as cold.

**Publication.**
- Sanitized code and aggregate records go on `publication/pcrl-finite-solver-adequacy-v1`, with the GitHub SHA and links verified.
- Private-path history stays local. Earlier branches and manuscripts are preserved. Handoffs are written in the repository, and no
  messages are sent to people.

## 11. Budgets, accounting and roles

**Budgets.**
- 8 research hours plus 30 minutes for closeout; 12 CPU-hours; zero paid compute.
- Two heavy jobs at once, one thread each, through the existing queue `fsa.jobqueue`.
- At most 32 new real-data release configurations, controls and genuine training repetitions included.

**Accounting.** Failures, agents' numerical work (charged) and internal fits (unit counters) are counted separately. No duplicate
aliases are run, and runtime is not padded. Resources are reserved for the conditional pilot.

**Agents.** Up to six live agents including the coordinator, with no uncounted recursive helpers:
- coordinator, also the optimizer implementer;
- exact-reference investigator;
- statistical and challenger reviewer;
- engineering and claims reviewer;
- independent verifier.

## 12. What is not claimed

- No result here is a guarantee against every classifier.
- Lower recovery for the declared readers is not protection against every reader.
- A null pilot or a closed optimizer hypothesis is not an impossibility result about other mechanisms.
- All MODEL_SELECT results are adaptive development evidence.

## 13. Threats and what this protocol states (the statistical reviewer's challenge, RULES.md section 9)

1. **Multiplicity.**
   - No family-wise error control is claimed.
   - Every registered law cell, strength and variant is reported, and no route, law or budget is chosen after its result.
   - Pilot screens are 2-SE screens without adjustment; each record states the minimum detectable benefit.
2. **Adaptivity.**
   - SV1 was designed with knowledge of the previous campaign's saturation (one protected partition per version).
   - It was revised once, from FA_FSA_v1 to FA_FSA_v2 at 02:46:39Z, after an invented-data trace that is not a registered law.
   - Every later change is a new variant, with a dated amendment card, counted against the limit of two and disclosed as post hoc.
   - MODEL_FIT and MODEL_SELECT were already used adaptively by every earlier campaign.
3. **Surrogate against recovery.**
   - A lower objective is never reported as protection or usefulness.
   - Every point reports its fitted π-weighted information next to its held-out label LOOKUP AUC.
   - A lower objective with unchanged recovery is a proxy-only result, and it closes the optimizer-repair hypothesis.
4. **Effort.** Each chain records CPU time, route and objective evaluations, accepted steps and its stopping reason. SV1_first is
   the equal-effort decomposition. Revised PRC and revised HQ share variant, initialization, settings and stopping rules.
5. **HQ.**
   - HQ is the same objective searched over a restricted, task-score-measurable rule family. Routing is on twin-cell means of π, qO0
     and qO1, while its tables and acceptance objective use the person-level values.
   - It is neither "the IBSI objective" nor "all of h(q)".
   - A PRC-HQ difference is never called "the information propensity adds".
6. **Expressivity.**
   - The audit records, label-free, the number of distinct six-field support states on MODEL_FIT and the share of people whose
     deployed task scores are shared with another support state.
   - Where that share is about zero, an unrestricted task-score mapping could reproduce any PRC partition on that support. A PRC
     advantage over HQ is then about rule parametrization and search inside two restricted deployable families.
7. **Invented laws.**
   - Laws can favour a variant, so route E needs registration before any solver run, and route S supersedes route E for the same
     variant.
   - The exact Bayes AUC is reader-optimal only within functions of the release.
   - On the source, readers are trained and data-limited, so measured recoveries are lower bounds.
8. **Matching.** Comparisons match on realized (measured) spend, never on budget labels.
9. **Uncertainty.** Household-clustered SEs treat refitted pipelines as fixed, and the LOOKUP bootstrap is conditional on the
   fitted tables. Both understate uncertainty.
10. **Winner's curse.** Selected points are chosen by rule, and their benefits are not unbiased. Diagnostics on them are
    descriptive.

**Order of events (disclosure).**
- The investigator committed the invented laws at 02:42:16Z (c27a91477), before any numerical run on them.
- Its first solver runs on them (the original solver only; the narrow reference and enumeration) started at 02:49:12Z. That was
  after the revised solver's code (91ba9a27e, 02:46:39Z) and before the statistical reviewer committed the rules (71f8de33b,
  02:59:46Z).
- The rules' declared inputs do not include those outcomes. The coordinator did not read them and relayed nothing about them
  until the rules were committed.
- They are committed files and are therefore part of the registration package.

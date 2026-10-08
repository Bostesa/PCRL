# Protocol — ccm (confidence-constrained mechanism: design and feasibility sprint)

| Item | Value |
|---|---|
| Study | `pcrl_confidence_constrained_mechanism_v1` |
| Branch | `research/pcrl-confidence-constrained-mechanism-v1` |
| Package | `ccm/` |
| Tests | `tests/pcrl_confidence_constrained_mechanism_v1/` |
| Results | this folder |
| Source | hcal tip 1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5; evidence 159537dcb61d030c870ae42e7addff4796651723 (ancestor verified) |
| Start | 2026-10-08T07:47:22Z |
| Private store | `<PRIVATE_CACHE>/ccm_v1`, environment variable `PCRL_CCM_PRIVATE_CACHE` |
| Budget | 8 elapsed hours, 20 CPU-hours, at most 2 heavy processes, $0 cloud |

Only role A edits this file. Sections marked [frozen at FEASIBILITY_LOCK] cannot change after that lock is pushed.

## 0. Purpose and scope

This is a design and feasibility sprint for ONE next mechanism, not a confirmatory study and not a repeat of hcal. The
first candidate is a confidence-constrained information release: choose which distinctions of a frozen reference
score to disclose, subject to an explicit, pointwise utility contract, instead of fitting new decoders to unchanged
tokens. The first question is whether the contract leaves any useful room to operate. If it forces an essentially
identity release, we say so and do not launch a pilot.

**Scope statement.**
- This is a narrow score-sharing setting: two recipients receive confidence channels derived from frozen teacher
  scores.
- It is not a general multi-purpose representation-learning method.
- Bounded-distortion quantization and privacy under distortion constraints are established prior work
  (PRIOR_WORK_AND_NOVELTY.md).

## 1. Exposure (verbatim; EXPOSURE_STATEMENT.md)

"This sprint is motivated by repeatedly used Adult development results and by an inner comparison observed after those
results. Existing models, partitions, thresholds and past outcomes are known. Any real-data prototype result is
exploratory development evidence. New procedural locks do not undo historical exposure. No old assessment is reopened
and no new confirmation population is opened."

## 2. Data roles (DATA_ACCESS_MANIFEST.json; ccm/data.py access guard)

**Inherited roles.** The exact hcal / lra / qpc / osf manifests, unchanged:

| Role | Rows | Groups |
|---|---|---|
| OSF_DEFENSE_FIT | 15,434 | 15,428 |
| HEAD_VALIDATION | 1,500 | 1,499 |
| AUDIT_FIT | 6,065 | 6,061 |
| INNER_SELECTION | 2,235 | 2,234 |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 | 13,929 |

**hcal subroles reused unchanged:** CALIBRATION_HELDOUT (2,000 representatives), ATTACK_FIT_NEW (4,064 rows) and
CALIBRATION_TRAIN_MATCHED. The 83 permitted columns, row order and group identities are those of the source manifests.

**Permissions.**
- Old assessment labels, old assessment predictions and every person-level assessment array are FORBIDDEN. The access
  guard drops assessment rows before any array is returned, and reference vectors are computed only on permitted rows.
- HEAD_VALIDATION labels are never read.
- OSF_DEFENSE_FIT reference vectors may be used for engineering geometry and defense fitting. SEX on OSF_DEFENSE_FIT
  is allowed only for privacy-aware assignment fitting, which is pilot-only.
- CALIBRATION_HELDOUT reference vectors are used for held-out geometry (fallback rates; label-free). The frozen source
  calibrated teacher is reused; its temperature is never refitted.
- ATTACK_FIT_NEW is for new attacker fitting only (pilot only).
- INNER_SELECTION is for the bounded exploratory pilot only, after a pushed PILOT_LOCK.
- No new population is opened.

**Reference teachers.** Per seed k ∈ {0, 1, 2} and recipient i (income K=2, occupation K=6):
- **U0:** the admitted frozen U teacher probability vectors, computed by the pinned forward pass on permitted rows only.
- **Ucal (PRIMARY reference):** U0 transformed by the frozen hcal H-GLOBAL-TEMP inverse temperatures of seed k (the
  α values in the hcal calU__s{k} records), using the frozen log-input rule of hcal.calib.apply_u.
- The decision is the source tie rule, d = numpy argmax of U0 (first index). Ucal preserves it.

## 3. Utility contract G(d, b) [frozen at FEASIBILITY_LOCK] (UTILITY_CONTRACT.md)

**Constants.** One setting only: d = 0.005, b = 0.0025. This is stricter than hcal's average-loss allowances
(0.010 / 0.005). A negative result under G does not show that the average-loss problem is impossible.

**Conditions.** For an input with reference vector p (Ucal, PRIMARY) and released vector q:
- **(NLL)** q_k ≥ exp(−d)·p_k for every class k. This implies log(p_y/q_y) ≤ d for EVERY label y. The clipped-score
  version (1e-12) also holds; B verifies this.
- **(Brier)** Σ_k q_k² − Σ_k p_k² − 2(q_y − p_y) ≤ b for EVERY label y. This is the full multiclass Brier, with no
  factor ½, as in the source convention.
- **(Class)** q has the strict argmax d(p), the source decision: q_d > q_k for every k ≠ d. Ucal preserves the
  first-index argmax but not strictness: tied Ucal tops are possible and are counted label-free.
- **Float64 semantics** (UTILITY_CONTRACT.md §6, after MATH_REVIEW.md B2):
  - the canonical predicate is `ccm.guard.check_release`, with E = exp(−d), NLL `q >= E*p`, numpy-sum Brier and
    simplex tolerance 1e-12;
  - representatives are built against E·p·(1 + 1e-9), b − 1e-9 and a 1e-9 class gap, then checked at d and b;
  - closed forms use the conservative margin e^d·(1 + 1e-12).

**Scope.**
- The guarantee is all-label and pointwise per input. It is not an expected or average guarantee, and it is a utility
  statement relative to Ucal only.
- A bound against U0 is NOT implied. Claiming both requires the intersection of the two feasible sets for every input.
- **Randomised releases:** the conditions must hold for EVERY supported output.

**Bins.** A bin (a token with one representative q) is admissible iff one q satisfies G for all its members.
- **Necessary condition:** Σ_k max_{p∈bin} p_k ≤ exp(d). Pairwise overlap is not sufficient for a whole bin when
  K ≥ 3; for K = 2 pairwise (extreme-pair) feasibility suffices (MATH_REVIEW.md R4.7).
- **Certification:** a returned q is checked against every member and label with the canonical predicate.
- **Infeasibility:** certified by the closed-form necessary condition only (declared iff Σ_k max p_k > e^d·(1 + 1e-12)).
- **Representative choice:** an input is served by the FIRST representative, in the registered cover order, of its
  decision class that certifies it. The rule is p-only.

**Unseen inputs.** For a deployment-time input with no admissible registered representative, the contract defines a
DISCLOSED FALLBACK with a fallback flag. Refusal is NOT used.
- If the Ucal top is strict, the fallback releases Ucal(p) itself; Ucal(p) satisfies G iff its top is strict.
- If the top is tied, it releases (1 − η)·Ucal(p) + η·e_d with the fixed η = 1e-6, admissible by MATH_REVIEW.md R3.3.
- Fallback flags and continuous outputs are part of the recipient's view and must be attacked.

## 4. Capacity bound [frozen at FEASIBILITY_LOCK]

- Every arm, for every recipient, gets at most 8 tokens per predicted class for income and 64 per predicted class for
  occupation. This is the i8o64 family of every incumbent code.
- Fallback outputs are not tokens; they are reported separately.
- Identical bound, inputs, teachers and solver for task-only, local, sequential and joint arms.

## 5. Stage D feasibility metrics (label-free) [frozen at FEASIBILITY_LOCK]

Per seed, recipient and predicted class, on Ucal vectors:

| Metric | Definition |
|---|---|
| F1 cover | certified greedy admissible cover of all OSF_DEFENSE_FIT rows (fixed label-free order: decreasing max probability, then row id). Reports the number of bins (upper bound), the bin-size distribution and the compression ratio bins/rows. |
| F2 packing | lower bound on the minimum number of bins: a greedy set of rows that are pairwise NLL-infeasible (Σ_k max(p_k, p'_k) > e^d·(1 + 1e-12)) in the same order. Every such row needs its own bin. |
| F3 capacity coverage | the fraction of OSF_DEFENSE_FIT rows covered by at most 8 / 64 certified bins per class. For occupation (K = 6) it is a constructive lower bound: greedy maximum coverage over the F1 bins. For income (K = 2) it is the EXACT maximum (F3x): an O(nC) dynamic program over the sorted score, which uses the one-dimensional interval structure of K = 2 admissibility (MATH_REVIEW.md R4.7, R5.5) with the tightened construction targets; every chosen bin is then certified by an exhibited q. The income greedy value is reported as F3_greedy. |
| F3u certified coverage bound | an upper bound on the coverage ANY admissible code can reach at the registered capacity. For each row r, N(r) is the number of rows r′ of its class with Σ_k max(p_k, p′_k) ≤ e^d·(1 + 1e-12) (every bin containing r lies in this NLL neighbourhood). Then F3u = Σ over classes of the sum of the C largest N(r) (C = 8 or 64), divided by the rows, capped at 1. It can be loose (MATH_REVIEW.md R5). A second valid bound, F3u_packing = 1 − Σ_c max(0, \|F2 pack_c\| − C)/n, holds because every packed row needs its own bin. It was added before the lock after role C found the neighbourhood bound vacuous (1.0) on synthetic data. The REGISTERED bound is F3u_registered = min(F3u, F3u_packing). |
| F4 held-out fallback | the fraction of CALIBRATION_HELDOUT representatives (2,000) whose Ucal vector satisfies G with NO registered representative from F3 (representative choice rule above): the deployment fallback rate. Tied-top counts are reported for both roles. |
| F5 decision floor and CLASS | the decision-only release carries no confidence vector, so it is confidence-INELIGIBLE under G (premise check). CLASS (one token per predicted class) is eligible iff each whole class is an admissible bin (checked). |

**Go rule (registered; correctness-independent; no privacy outcome involved).** A nontrivial admissible channel exists
under the declared contract iff, for EVERY seed and BOTH recipients:
- F4 held-out fallback ≤ 0.05; and
- F3 capacity coverage ≥ 0.95.

If the rule fails, a bound says whether the failure is intrinsic to the contract or could be a construction
limitation:
- **Income (K = 2):** F3 is exact (F3x), so F3 < 0.95 is intrinsic.
- **Occupation (K = 6):** F3u_registered < 0.95 for some seed is intrinsic: no admissible code at the registered
  capacity can cover 95% of fitting rows. F3u_registered ≥ 0.95 with F3 < 0.95 is reported as INCOMPLETE for that
  recipient.
- **F4 > 0.05:** reported as FALLBACK_DOMINATED irrespective of the bounds.

Otherwise the mechanism is NEAR_IDENTITY or FALLBACK_DOMINATED under G. There is then no pilot, and the sprint label is
PREMISE_NOT_SUPPORTED.

**Secondary diagnostic** (registered here, label-free, NOT the selected contract, NOT a pass, never a prototype).
F1–F4 are also computed under the TEACHER-EXPECTED guard G_exp(d, b):
- KL(p‖q) ≤ d (expected log-loss excess under y ~ p);
- Σ_k (q_k − p_k)² ≤ b (expected Brier excess under y ~ p);
- class preserved.

This informs which contract a next study could use. Its validity rests on calibration and is weaker than G.
- It never reuses the G neighbourhood or the NLL closed forms (MATH_REVIEW.md R7.4).
- Its F3u is either a G_exp-valid bound or null with a reason.

**Theoretical expectation (MATH_REVIEW.md R6, recorded before any real-array run).** If P(SEX | p) is L-Lipschitz in
total variation, every admissible arm has SEX Bayes accuracy within L·(e^d − 1) ≈ 0.005·L of releasing p itself.

## 6. Finite-law oracles [laws pinned by role F before any candidate code evaluates them]

**Laws** (TOY_LAWS.json, sha256-pinned):
- NULL: privacy cannot usefully improve under G;
- POSITIVE: admissible coarsening can remove an unnecessary sensitive distinction;
- COMPLEMENTARY: combining the two releases creates a coalition clue;
- NO_COALITION: no extra coalition clue;
- JOINT_HEADROOM: purpose-built so that joint design beats both non-adaptive sequential orders. It was added by role F
  before any candidate code evaluated the laws (TOY_LAWS.json amendment A1). It is a capability fixture, not evidence
  of headroom on real data.

Each law lists inputs x with probabilities, SEX laws, and reference vectors p₁(x), p₂(x). Construction notes are
recorded. Laws are never changed after a candidate result.

**Oracles.** Exact enumeration over admissible partitions of the reference values, under G and the capacity bound.
Arms:
- decision-only (floor; confidence-ineligible);
- CLASS (eligibility check);
- task-only (SEX-blind: fewest admissible tokens, ties by a fixed label-free order);
- local (each recipient minimises its own MI);
- sequential 1→2 and 2→1 (the first recipient local, the second minimising coalition MI given the first). These are
  NON-ADAPTIVE designs: the second partition is chosen given the first PARTITION, not conditioned on realised tokens.
  They are not Taylor et al.'s adaptive algorithm (PRIOR_WORK_AND_NOVELTY.md);
- joint (minimises coalition MI over admissible pairs);
- stochastic local (an LP minimising each recipient's own Bayes SEX accuracy over channels whose every supported output
  satisfies G). Every law has at most 6 inputs, so capacity never binds and the LP is uncapped and exact. A law that
  could bind capacity is refused rather than relaxed (MATH_REVIEW.md R9.5).

**Measures.** Exact plug-in mutual information (nats) and Bayes accuracy of SEX from t₁, t₂ and (t₁, t₂).

**Solver bracket.** Exhaustive over admissible set partitions; the LP via scipy.optimize.linprog (HiGHS) with an
independent re-check. The correctness gate does not depend on any favourable value.

## 7. Stages, locks and gates

| Stage | Content | Gate |
|---|---|---|
| A | INCUMBENT_TRIAGE.md from committed aggregates only (no refits, no rescoring) | — |
| B | mechanism decision table + PRIOR_WORK_AND_NOVELTY.md (role D, primary sources) | — |
| C | UTILITY_CONTRACT.md + independent math review (role B) | no blocking finding |
| FEASIBILITY_LOCK | binds §3–§6, code, the data guard, TOY_LAWS.json, PREDICTIONS.json | pushed and verified before any real-array geometry or oracle run |
| D | geometry F1–F5 (+ secondary) and finite-law oracles; independent replay (role E) | go rule (§5) |
| E | MECHANISM_DECISION.md (A, B, D) | — |
| F | restricted exploratory pilot | only if the go rule holds, math review is clear, the mechanism is materially different from a closed recipe, and PILOT_LOCK.json is pushed first |

**Pilot (only if authorised).**
- One compact code bank, one configuration, 3 paired teacher seeds.
- Required controls 1–8 of the handoff (§12).
- New attackers on ATTACK_FIT_NEW (linear, nonlinear, token/cell, both ignore-recipient coalition readers,
  defense-aware, and fallback-metadata readers); scored once on INNER_SELECTION.
- Old 0.02 pair benefit and 0.01 local guard thresholds retained prospectively.
- No assessment opening.

**Labels:** PREMISE_NOT_SUPPORTED, MECHANISM_CORRECT_NO_HEADROOM, EXPLORATORY_PROTOTYPE_NO_ADVANTAGE,
EXPLORATORY_PROTOTYPE_PROMISING, INCOMPLETE.

**Amendments.** A post-lock coding bug requires a dated amendment, pushed before any governed rerun. Original outputs
are kept. Thresholds are never changed to obtain a pass.

## 8. Roles and file ownership

| Role | Owns |
|---|---|
| A | protocol, locks, decision files, ccm/ids.py, ccm/lock.py, ccm/sema.py, ccm/run.py, triage, integration |
| B | MATH_REVIEW.md, scratch counterexamples (no locked file edits) |
| C | ccm/guard.py, ccm/geometry.py, ccm/oracle.py, tests/…/test_guard.py, test_geometry.py, test_oracle.py |
| D | PRIOR_WORK_AND_NOVELTY.md |
| E | verification/ (independent code; no import of ccm.guard, geometry, oracle, run or selection code) |
| F | ccm/data.py, tests/…/test_data.py, DATA_ACCESS_MANIFEST.json, TOY_LAWS.json (+ its construction notes) |

**Shared semaphore.** At most 2 heavy processes, one BLAS thread each.
